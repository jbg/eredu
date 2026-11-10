//! Retained typed V3 target and prediction ownership and its explicit projections.
use super::{
    moe::RoutedSpec,
    v3_specs::{self, AttentionSpec, PredictionSpec, QuerySpec},
    LayerPolicy, V3Args,
};
use eredu_nn::{
    DeclareLogicalParameters, Error, LinearSpec, NormalizationConstructionSpec as Norm,
    NormalizationScale,
};

#[derive(Debug, Clone)]
pub(crate) struct Block {
    root: String,
    pub norms: [Norm; 2],
    pub attention: AttentionSpec,
    pub dense: Option<eredu_nn::SplitGatedProductSpec>,
    pub routed: Option<RoutedSpec>,
    prediction: Option<PredictionSpec>,
}
#[derive(Debug, Clone)]
pub(crate) struct Parameters {
    pub statics: crate::decoder::StaticModuleConstruction,
    blocks: Vec<Block>,
    target_count: usize,
}
impl Parameters {
    pub fn declare(args: &V3Args) -> Result<Self, Error> {
        let target_count = args.num_hidden_layers as usize;
        let total = target_count
            .checked_add(args.num_nextn_predict_layers as usize)
            .ok_or_else(|| Error::backend("V3 invocation count overflow"))?;
        let blocks = (0..total)
            .map(|layer| {
                let root = format!("model.layers.{layer}");
                let dense = layer < target_count
                    && args.layer_schedule.get(layer) == Some(&LayerPolicy::DenseMlp);
                Ok(Block {
                    norms: [
                        v3_specs::declare_normalization_spec(
                            args,
                            &format!("{root}.input_layernorm.weight"),
                            args.hidden_size,
                        )?,
                        v3_specs::declare_normalization_spec(
                            args,
                            &format!("{root}.post_attention_layernorm.weight"),
                            args.hidden_size,
                        )?,
                    ],
                    attention: v3_specs::declare_attention_spec(
                        args,
                        &format!("{root}.self_attn"),
                    )?,
                    dense: dense
                        .then(|| v3_specs::declare_dense_parameters(args, &format!("{root}.mlp")))
                        .transpose()?,
                    routed: (!dense)
                        .then(|| super::v3::declare_routed_spec(args, layer))
                        .transpose()?,
                    prediction: (layer >= target_count)
                        .then(|| v3_specs::declare_prediction_spec(args, &root))
                        .transpose()?,
                    root,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let p = Self {
            statics: super::v3::declare_static_spec(args).construction_specs()?,
            blocks,
            target_count,
        };
        eredu_nn::validate_logical_parameter_topology(&p.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(p)
    }
    pub fn block(&self, root: &str) -> Result<&Block, Error> {
        self.blocks
            .iter()
            .find(|b| b.root == root)
            .ok_or_else(|| Error::backend(format!("undeclared V3 invocation {root}")))
    }
    pub fn prediction(&self, root: &str) -> Result<&PredictionSpec, Error> {
        self.block(root)?
            .prediction
            .as_ref()
            .ok_or_else(|| Error::backend("V3 invocation is not a predictor"))
    }
    pub fn target_only(&self) -> Self {
        let mut p = self.clone();
        p.blocks.truncate(p.target_count);
        p
    }
    pub fn normalization(&self, name: &str) -> Result<&Norm, Error> {
        let mut norms = vec![self.statics.normalization_spec()];
        for b in &self.blocks {
            norms.extend(&b.norms);
            norms.push(&b.attention.kv_norm);
            if let QuerySpec::LowRank(q) = &b.attention.query {
                norms.push(&q.normalization);
            }
            if let Some(p) = &b.prediction {
                norms.extend([&p.embedding_norm, &p.hidden_norm, &p.output_norm]);
            }
        }
        norms
            .into_iter()
            .find(|n| match &n.scale {
                NormalizationScale::Learned(p)
                | NormalizationScale::LearnedOffset { weight: p, .. } => p.id.as_str() == name,
                NormalizationScale::Unit => false,
            })
            .ok_or_else(|| Error::backend(format!("undeclared V3 normalization {name}")))
    }
    pub fn with_formats(&self, args: &V3Args) -> Result<Self, Error> {
        let mut p = self.clone();
        let project = |s: &mut LinearSpec| -> Result<(), Error> {
            s.format = crate::linear_format::project_matrix_format(
                &s.weight,
                &s.format,
                args.linear_format_for(s.weight.id.as_str()),
            )?;
            Ok(())
        };
        let mut e = p.statics.embedding_spec().clone();
        e.format = crate::linear_format::project_matrix_format(
            &e.weight,
            &e.format,
            args.linear_format_for(e.weight.id.as_str()),
        )?;
        let mut h = p.statics.head_spec().cloned();
        if let Some(h) = &mut h {
            project(h)?;
        }
        p.statics = crate::decoder::StaticModuleConstruction::new(
            e,
            h,
            p.statics.normalization_spec().clone(),
        )?;
        for b in &mut p.blocks {
            match &mut b.attention.query {
                QuerySpec::Direct(q) => project(q)?,
                QuerySpec::LowRank(q) => {
                    if let Some(a) = &mut q.first {
                        project(a)?;
                    }
                    project(&mut q.second)?;
                }
            }
            for s in [
                &mut b.attention.kv_a,
                &mut b.attention.kv_b,
                &mut b.attention.output,
            ] {
                project(s)?;
            }
            if let Some(d) = &mut b.dense {
                for s in [&mut d.gate, &mut d.up, &mut d.down] {
                    project(s)?;
                }
            }
            if let Some(r) = &mut b.routed {
                let selector = &r.parameters.selector;
                let format = args
                    .linear_formats
                    .get(selector.weight().id.as_str())
                    .copied()
                    .unwrap_or(eredu_checkpoint::LinearFormat::Dense);
                r.parameters.selector =
                    selector
                        .clone()
                        .with_format(crate::linear_format::project_matrix_format(
                            selector.weight(),
                            selector.format(),
                            format,
                        )?)?;
                r.parameters.experts = crate::linear_format::project_packed_gated_formats(
                    &r.parameters.experts,
                    |name| args.linear_format_for(name),
                )?;
                for s in [&mut r.shared.gate, &mut r.shared.up, &mut r.shared.down] {
                    project(s)?;
                }
            }
            if let Some(d) = &mut b.prediction {
                project(&mut d.fusion)?;
                project(&mut d.output)?;
            }
        }
        eredu_nn::validate_logical_parameter_topology(&p.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(p)
    }
    pub fn with_geometry(&self, heads: i32, dense: i32, routed: i32) -> Result<Self, Error> {
        let mut p = self.clone();
        for b in &mut p.blocks {
            let a = &mut b.attention;
            let old = a.geometry;
            let geometry = eredu_nn::LatentAttentionGeometry::new(
                heads,
                old.non_positional_dimensions(),
                old.positional_dimensions(),
                old.value_dimensions(),
                old.latent_dimensions(),
            )?;
            match &mut a.query {
                QuerySpec::Direct(q) => q.output = geometry.query_width(),
                QuerySpec::LowRank(q) => q.second.output = geometry.query_width(),
            }
            a.kv_b.output = geometry.reconstruction_width()?;
            a.output.input = geometry.value_width();
            a.geometry = geometry;
            if let Some(d) = &mut b.dense {
                d.gate.output = dense;
                d.up.output = dense;
                d.down.input = dense;
            }
            if let Some(r) = &mut b.routed {
                let ratio = r.shared.gate.output / r.parameters.experts.intermediate_dimensions();
                let shared = ratio
                    .checked_mul(routed)
                    .ok_or_else(|| Error::backend("V3 selected shared width overflow"))?;
                r.parameters.experts = r
                    .parameters
                    .experts
                    .clone()
                    .with_group_geometry(r.parameters.experts.group_count(), routed)?;
                r.shared.gate.output = shared;
                r.shared.up.output = shared;
                r.shared.down.input = shared;
            }
        }
        eredu_nn::validate_logical_parameter_topology(&p.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(p)
    }
}
impl DeclareLogicalParameters for Parameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.statics.logical_parameters()?;
        for b in &self.blocks {
            p.extend(b.norms.as_slice().logical_parameters()?);
            p.extend(b.attention.logical_parameters()?);
            if let Some(d) = &b.dense {
                p.extend(d.logical_parameters()?);
            }
            if let Some(r) = &b.routed {
                p.extend(r.logical_parameters()?);
            }
            if let Some(d) = &b.prediction {
                p.extend(d.logical_parameters()?);
            }
        }
        Ok(p)
    }
}
impl DeclareLogicalParameters for V3Args {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.parameters().logical_parameters()
    }
}
