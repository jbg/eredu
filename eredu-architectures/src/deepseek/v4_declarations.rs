//! Retained V4 target, sequential prediction and DSpark parameter ownership.
use super::{
    v4_specs::{self, AttentionSpec, DsparkSpec, HyperBlockSpec, MoeSpec, PredictionSpec},
    V4Args,
};
use eredu_nn::{
    DeclareLogicalParameters, EmbeddingSpec, Error, HyperHeadSpec, LinearSpec,
    NormalizationConstructionSpec as Norm,
};

#[derive(Debug, Clone)]
pub(crate) struct Block {
    root: String,
    layer: usize,
    pub norms: [Norm; 2],
    pub hyper: HyperBlockSpec,
    pub attention: AttentionSpec,
    pub moe: MoeSpec,
    prediction: Option<PredictionSpec>,
}
#[derive(Debug, Clone)]
pub(crate) struct Parameters {
    pub statics: crate::decoder::StaticModuleConstruction,
    pub head: HyperHeadSpec,
    pub dspark: Option<DsparkSpec>,
    blocks: Vec<Block>,
    target_count: usize,
}
impl Parameters {
    pub fn declare(args: &V4Args) -> Result<Self, Error> {
        let target_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
        let prediction = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        let total = target_count
            .checked_add(prediction)
            .ok_or_else(|| Error::backend("V4 invocation count overflow"))?;
        let blocks = (0..total)
            .map(|layer| {
                let root = if layer < target_count {
                    format!("layers.{layer}")
                } else {
                    format!("mtp.{}", layer - target_count)
                };
                Ok(Block {
                    norms: v4_specs::declare_block_normalizations(args, &root)?,
                    hyper: v4_specs::declare_hyper_block_spec(args, &root)?,
                    attention: v4_specs::declare_attention_spec(
                        args,
                        layer,
                        &format!("{root}.attn"),
                    )?,
                    moe: v4_specs::declare_moe_spec(args, layer, &format!("{root}.ffn"))?,
                    prediction: (layer >= target_count && args.dspark.is_none())
                        .then(|| v4_specs::declare_prediction_spec(args, &root))
                        .transpose()?,
                    root,
                    layer,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let p = Self {
            statics: super::v4::declare_text_static_spec(args).construction_specs()?,
            head: v4_specs::declare_hyper_head_spec(args, "hc_head")?,
            dspark: args
                .dspark
                .as_ref()
                .map(|_| v4_specs::declare_dspark_spec(args))
                .transpose()?,
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
            .ok_or_else(|| Error::backend(format!("undeclared V4 invocation {root}")))
    }
    pub fn block_at(&self, layer: usize, root: &str) -> Result<&Block, Error> {
        let b = self.block(root)?;
        if b.layer != layer {
            return Err(Error::backend(
                "V4 layer identity differs from retained owner",
            ));
        }
        Ok(b)
    }
    pub fn prediction(&self, root: &str) -> Result<&PredictionSpec, Error> {
        self.block(root)?
            .prediction
            .as_ref()
            .ok_or_else(|| Error::backend("V4 invocation has no sequential prediction owner"))
    }
    pub fn hyper_head(&self, prefix: &str) -> Result<&HyperHeadSpec, Error> {
        let identity = format!("{prefix}_fn");
        std::iter::once(&self.head)
            .chain(self.dspark.iter().map(|s| &s.hyper_head))
            .chain(
                self.blocks
                    .iter()
                    .filter_map(|b| b.prediction.as_ref().map(|p| &p.hyper_head)),
            )
            .find(|s| s.function.id.as_str() == identity)
            .ok_or_else(|| Error::backend(format!("undeclared V4 hyper-head {prefix}")))
    }
    pub fn target_only(&self) -> Self {
        let mut p = self.clone();
        p.blocks.truncate(p.target_count);
        p.dspark = None;
        p
    }
    pub fn with_formats(&self, args: &V4Args) -> Result<Self, Error> {
        let mut p = self.clone();
        let project = |s: &mut LinearSpec| -> Result<(), Error> {
            s.format = crate::linear_format::project_matrix_format(
                &s.weight,
                &s.format,
                args.linear_format_for(s.weight.id.as_str()),
            )?;
            Ok(())
        };
        let embedding = |s: &mut EmbeddingSpec| -> Result<(), Error> {
            s.format = crate::linear_format::project_matrix_format(
                &s.weight,
                &s.format,
                args.linear_format_for(s.weight.id.as_str()),
            )?;
            Ok(())
        };
        let compressor = |s: &mut v4_specs::CompressorSpec| -> Result<(), Error> {
            project(&mut s.wkv)?;
            project(&mut s.wgate)
        };
        let mut e = p.statics.embedding_spec().clone();
        embedding(&mut e)?;
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
            let a = &mut b.attention;
            if let Some(q) = &mut a.query.first {
                project(q)?;
            }
            for s in [
                &mut a.query.second,
                &mut a.kv,
                &mut a.output_a,
                &mut a.output_b,
            ] {
                project(s)?;
            }
            if let Some(c) = &mut a.compressor {
                compressor(c)?;
            }
            if let Some(i) = &mut a.indexer {
                project(&mut i.query)?;
                project(&mut i.weights)?;
                compressor(&mut i.compressor)?;
            }
            let r = &mut b.moe.routed;
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
            if let Some(d) = &mut b.prediction {
                project(&mut d.embedding_projection)?;
                project(&mut d.hidden_projection)?;
            }
        }
        if let Some(d) = &mut p.dspark {
            project(&mut d.main_projection)?;
            embedding(&mut d.markov_embedding)?;
            project(&mut d.markov_output)?;
            project(&mut d.confidence_head)?;
        }
        eredu_nn::validate_logical_parameter_topology(&p.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(p)
    }
    pub fn with_geometry(&self, heads: i32, groups: i32, routed: i32) -> Result<Self, Error> {
        let mut p = self.clone();
        for b in &mut p.blocks {
            let a = &mut b.attention;
            let old = a.geometry;
            let rank = a.output_rank();
            let geometry = eredu_nn::AttentionHeadGeometry::new(
                heads,
                old.key_value_heads(),
                old.key_dimensions(),
                old.value_dimensions(),
            )?;
            a.query.second.output = geometry.query_width();
            a.output_a.input = geometry.attended_width() / groups;
            let output = groups
                .checked_mul(rank)
                .ok_or_else(|| Error::backend("V4 selected output width overflow"))?;
            a.output_a.output = output;
            a.output_b.input = output;
            a.geometry = geometry;
            a.groups = groups;
            let r = &mut b.moe.routed;
            let ratio = r.shared.gate.output / r.parameters.experts.intermediate_dimensions();
            let shared = ratio
                .checked_mul(routed)
                .ok_or_else(|| Error::backend("V4 selected shared width overflow"))?;
            r.parameters.experts = r
                .parameters
                .experts
                .clone()
                .with_group_geometry(r.parameters.experts.group_count(), routed)?;
            r.shared.gate.output = shared;
            r.shared.up.output = shared;
            r.shared.down.input = shared;
        }
        eredu_nn::validate_logical_parameter_topology(&p.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(p)
    }
}
impl DeclareLogicalParameters for Parameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.statics.logical_parameters()?;
        p.extend(self.head.logical_parameters()?);
        if let Some(d) = &self.dspark {
            p.extend(d.logical_parameters()?);
        }
        for b in &self.blocks {
            p.extend(b.norms.as_slice().logical_parameters()?);
            p.extend(b.hyper.logical_parameters()?);
            p.extend(b.attention.logical_parameters()?);
            p.extend(b.moe.logical_parameters()?);
            if let Some(d) = &b.prediction {
                p.extend(d.logical_parameters()?);
            }
        }
        Ok(p)
    }
}
impl DeclareLogicalParameters for V4Args {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        self.parameters().logical_parameters()
    }
}
