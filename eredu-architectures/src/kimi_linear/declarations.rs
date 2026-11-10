//! Family-owned Kimi parameter scopes, separate from invocation state and placement.
use super::{AttentionKind, FeedForwardPolicy, ModelArgs};
use eredu_nn::{DeclareLogicalParameters, Error, LogicalParameterDeclaration};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub(crate) enum MixerParameters {
    Kda(super::specs::KdaSpec),
    Mla(super::specs::MlaSpec),
}
impl MixerParameters {
    pub(crate) fn state_policy(&self) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        match self {
            Self::Kda(spec) => spec.state_policy(),
            Self::Mla(spec) => eredu_core::cache::LayerCachePolicy::compressed_latent_rotary(
                eredu_core::AttentionPolicy::Full,
                spec.geometry.latent_dimensions(),
                spec.geometry.positional_dimensions(),
            )
            .map_err(Error::backend_source),
        }
    }
}
impl DeclareLogicalParameters for MixerParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Kda(spec) => spec.logical_parameters(),
            Self::Mla(spec) => spec.logical_parameters(),
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum FeedForwardParameters {
    Dense(eredu_nn::SplitGatedProductSpec),
    Sparse(super::moe::SparseParameters),
}
impl DeclareLogicalParameters for FeedForwardParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        match self {
            Self::Dense(spec) => spec.logical_parameters(),
            Self::Sparse(spec) => spec.logical_parameters(),
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) struct BlockParameters {
    pub mixer: MixerParameters,
    pub feed_forward: FeedForwardParameters,
    pub input_norm: eredu_nn::NormalizationConstructionSpec,
    pub post_attention_norm: eredu_nn::NormalizationConstructionSpec,
}
impl BlockParameters {
    fn new(args: &ModelArgs, layer: usize) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("Kimi invocation is outside its schedule"))?;
        let root = format!("model.layers.{layer}");
        Ok(Self {
            mixer: match policy.attention {
                AttentionKind::Kda => MixerParameters::Kda(super::specs::kda_spec(
                    args,
                    &format!("{root}.self_attn"),
                    args.kda_config.num_heads,
                )?),
                AttentionKind::Mla => MixerParameters::Mla(super::specs::mla_spec(
                    args,
                    &format!("{root}.self_attn"),
                    args.num_attention_heads,
                )?),
            },
            feed_forward: match policy.feed_forward {
                FeedForwardPolicy::Dense => {
                    FeedForwardParameters::Dense(super::specs::dense_parameters(
                        args,
                        &format!("{root}.mlp"),
                        args.intermediate_size,
                    )?)
                }
                FeedForwardPolicy::SparseMoe => {
                    FeedForwardParameters::Sparse(super::moe::sparse_parameters(args, layer)?)
                }
            },
            input_norm: super::specs::normalization_spec(
                args,
                &format!("{root}.input_layernorm.weight"),
                args.hidden_size,
            )?,
            post_attention_norm: super::specs::normalization_spec(
                args,
                &format!("{root}.post_attention_layernorm.weight"),
                args.hidden_size,
            )?,
        })
    }
    pub(crate) fn geometry(&self, args: &ModelArgs) -> super::BlockGeometry {
        let mut geometry = super::BlockGeometry {
            kda_heads: args.kda_config.num_heads,
            mla_heads: args.num_attention_heads,
            dense_intermediate: args.intermediate_size,
            routed_intermediate: args.moe_intermediate_size,
            shared_intermediate: args.moe_intermediate_size,
        };
        match &self.mixer {
            MixerParameters::Kda(p) => geometry.kda_heads = p.geometry.value_heads(),
            MixerParameters::Mla(p) => geometry.mla_heads = p.geometry.heads(),
        }
        match &self.feed_forward {
            FeedForwardParameters::Dense(p) => geometry.dense_intermediate = p.down.input,
            FeedForwardParameters::Sparse(p) => {
                geometry.routed_intermediate = p.routed.experts.intermediate_dimensions();
                geometry.shared_intermediate = p.shared.down.input;
            }
        }
        geometry
    }
    pub(crate) fn local(
        mut self,
        geometry: super::BlockGeometry,
        routed: Option<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        use super::specs::{LatentSpec, QuerySpec};
        match &mut self.mixer {
            MixerParameters::Kda(p) => {
                let old = p.geometry;
                if geometry.kda_heads > old.value_heads() {
                    return Err(Error::backend("local Kimi KDA exceeds declared heads"));
                }
                let heads = eredu_nn::GatedDeltaHeadGeometry::new(
                    geometry.kda_heads,
                    geometry.kda_heads,
                    old.key_dimensions(),
                    old.value_dimensions(),
                )?;
                for linear in [&mut p.q, &mut p.k, &mut p.v, &mut p.f_b, &mut p.g_b] {
                    linear.output = heads.value_width();
                }
                p.beta.output = heads.value_heads();
                p.output.input = heads.value_width();
                for convolution in &mut p.convolutions {
                    convolution.channels = heads.value_width();
                }
                p.geometry = heads;
            }
            MixerParameters::Mla(p) => {
                let old = p.geometry;
                if geometry.mla_heads > old.heads() {
                    return Err(Error::backend("local Kimi MLA exceeds declared heads"));
                }
                let heads = eredu_nn::LatentAttentionGeometry::new(
                    geometry.mla_heads,
                    old.non_positional_dimensions(),
                    old.positional_dimensions(),
                    old.value_dimensions(),
                    old.latent_dimensions(),
                )?;
                match &mut p.query {
                    QuerySpec::Direct(p) => p.output = heads.query_width(),
                    QuerySpec::LowRank(p) => p.second.output = heads.query_width(),
                }
                match &mut p.kv_b {
                    LatentSpec::Fused(p) => p.output = heads.reconstruction_width()?,
                    LatentSpec::Split { key, value } => {
                        key.output = heads.key_width();
                        value.output = heads.value_width();
                    }
                }
                p.output.input = heads.value_width();
                p.geometry = heads;
            }
        }
        self.feed_forward = match self.feed_forward {
            FeedForwardParameters::Dense(mut p) => {
                if routed.is_some()
                    || geometry.dense_intermediate <= 0
                    || geometry.dense_intermediate > p.down.input
                {
                    return Err(Error::backend("invalid local Kimi dense scope"));
                }
                p.gate.output = geometry.dense_intermediate;
                p.up.output = geometry.dense_intermediate;
                p.down.input = geometry.dense_intermediate;
                FeedForwardParameters::Dense(p)
            }
            FeedForwardParameters::Sparse(p) => {
                let source = &p.routed.experts;
                if geometry.routed_intermediate <= 0
                    || geometry.routed_intermediate > source.intermediate_dimensions()
                {
                    return Err(Error::backend(
                        "local Kimi expert width exceeds declaration",
                    ));
                }
                let bank = match routed {
                    Some(bank) => {
                        if bank.group_count() > source.group_count() {
                            return Err(Error::backend(
                                "local Kimi expert count exceeds declaration",
                            ));
                        }
                        let expected = source.clone().with_group_geometry(
                            bank.group_count(),
                            geometry.routed_intermediate,
                        )?;
                        if bank != expected {
                            return Err(Error::backend(
                                "local Kimi bank differs from declared identities, formats, geometry or equations",
                            ));
                        }
                        bank
                    }
                    None => source
                        .clone()
                        .with_group_geometry(source.group_count(), geometry.routed_intermediate)?,
                };
                FeedForwardParameters::Sparse(
                    p.with_local_geometry(bank, geometry.shared_intermediate)?,
                )
            }
        };
        self.logical_parameters()?;
        Ok(self)
    }
    fn with_formats(mut self, args: &ModelArgs) -> Result<Self, Error> {
        use super::specs::{LatentSpec, QuerySpec};
        match &mut self.mixer {
            MixerParameters::Kda(spec) => {
                for p in [
                    &mut spec.q,
                    &mut spec.k,
                    &mut spec.v,
                    &mut spec.f_a,
                    &mut spec.f_b,
                    &mut spec.beta,
                    &mut spec.g_a,
                    &mut spec.g_b,
                    &mut spec.output,
                ] {
                    project(p, args)?;
                }
            }
            MixerParameters::Mla(spec) => {
                match &mut spec.query {
                    QuerySpec::Direct(p) => project(p, args)?,
                    QuerySpec::LowRank(p) => {
                        if let Some(first) = &mut p.first {
                            project(first, args)?;
                        }
                        project(&mut p.second, args)?;
                    }
                }
                project(&mut spec.kv_a, args)?;
                match &mut spec.kv_b {
                    LatentSpec::Fused(p) => project(p, args)?,
                    LatentSpec::Split { key, value } => {
                        project(key, args)?;
                        project(value, args)?;
                    }
                }
                project(&mut spec.output, args)?;
            }
        }
        match &mut self.feed_forward {
            FeedForwardParameters::Dense(p) => project_dense(p, args)?,
            FeedForwardParameters::Sparse(p) => {
                project_dense(&mut p.shared, args)?;
                let weight = p.routed.selector.weight().id.as_str();
                p.routed.selector = p.routed.selector.clone().with_format(
                    crate::linear_format::project_matrix_format(
                        p.routed.selector.weight(),
                        p.routed.selector.format(),
                        args.weight_quantization_for(weight).into(),
                    )?,
                )?;
                p.routed.experts = crate::linear_format::project_packed_gated_formats(
                    &p.routed.experts,
                    |name| args.weight_quantization_for(name).into(),
                )?;
            }
        }
        self.logical_parameters()?;
        Ok(self)
    }
}
fn project(p: &mut eredu_nn::LinearSpec, args: &ModelArgs) -> Result<(), Error> {
    p.format = crate::linear_format::project_matrix_format(
        &p.weight,
        &p.format,
        args.weight_quantization_for(p.weight.id.as_str()).into(),
    )?;
    Ok(())
}
fn project_dense(p: &mut eredu_nn::SplitGatedProductSpec, args: &ModelArgs) -> Result<(), Error> {
    for p in [&mut p.gate, &mut p.up, &mut p.down] {
        project(p, args)?;
    }
    Ok(())
}
impl DeclareLogicalParameters for BlockParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.mixer.logical_parameters()?;
        p.extend(self.input_norm.logical_parameters()?);
        p.extend(self.post_attention_norm.logical_parameters()?);
        p.extend(self.feed_forward.logical_parameters()?);
        eredu_nn::validate_logical_parameter_topology(&p).map_err(Error::backend_source)?;
        Ok(p)
    }
}
/// Validated Kimi parameter ownership and independent invocation/state declarations.
#[derive(Debug, Clone)]
pub struct Declaration {
    args: ModelArgs,
    pub(crate) statics: crate::decoder::StaticModuleConstruction,
    pub(crate) blocks: Vec<BlockParameters>,
    definition: Arc<eredu_runtime::ComponentDefinition>,
}
impl Declaration {
    /// Builds typed scopes without source payloads or backend resources.
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        args.validate().map_err(Error::backend_source)?;
        let statics = super::static_spec(&args).construction_specs()?;
        let blocks = (0..args.layer_schedule.len())
            .map(|i| BlockParameters::new(&args, i))
            .collect::<Result<_, _>>()?;
        Self::from_parts(args, statics, blocks)
    }
    fn from_parts(
        args: ModelArgs,
        statics: crate::decoder::StaticModuleConstruction,
        blocks: Vec<BlockParameters>,
    ) -> Result<Self, Error> {
        let parameters =
            super::parallel::parameter_description_from_scopes(&args, &statics, &blocks)
                .map_err(Error::backend_source)?;
        let policies = blocks
            .iter()
            .map(|p| p.mixer.state_policy())
            .collect::<Result<_, _>>()?;
        let state = eredu_runtime::StateLayout::new(
            eredu_core::LayerSchedule::new(blocks.len(), policies)
                .map_err(Error::backend_source)?,
        )
        .map_err(Error::backend_source)?;
        let scope = Self {
            args,
            statics,
            blocks,
            definition: Arc::new(super::partitioned::component_definition(parameters, state)?),
        };
        eredu_nn::validate_logical_parameter_topology(&scope.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let formats = scope
            .args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.iter())
            .map(|(name, format)| (name.as_str(), Some((*format).into())));
        crate::format_references::validate(
            &scope,
            formats,
            || Ok(Vec::new()),
            || super::safetensors_plan(&scope),
        )?;
        Ok(scope)
    }
    /// Projects mutable state from retained mixers at admitted local dimensions.
    pub fn state_layout_with_geometry(
        &self,
        geometry: &[super::BlockGeometry],
    ) -> Result<eredu_runtime::StateLayout, Error> {
        if geometry.len() != self.blocks.len() {
            return Err(Error::backend(
                "Kimi state geometry differs from declared invocation count",
            ));
        }
        let policies = self
            .blocks
            .iter()
            .cloned()
            .zip(geometry)
            .map(|(scope, geometry)| scope.local(*geometry, None)?.mixer.state_policy())
            .collect::<Result<_, _>>()?;
        eredu_runtime::StateLayout::new(
            eredu_core::LayerSchedule::new(self.blocks.len(), policies)
                .map_err(Error::backend_source)?,
        )
        .map_err(Error::backend_source)
    }
    pub(crate) fn expert_banks(
        &self,
    ) -> std::collections::BTreeMap<usize, eredu_nn::GroupedGatedProductSpec> {
        self.blocks
            .iter()
            .enumerate()
            .filter_map(|(unit, scope)| match &scope.feed_forward {
                FeedForwardParameters::Sparse(p) => Some((unit, p.routed.experts.clone())),
                _ => None,
            })
            .collect()
    }
    /// Explicit routed hook sites for one declared execution invocation.
    pub fn routed_observation_points(
        &self,
        unit_path: &str,
        layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoints> {
        let FeedForwardParameters::Sparse(p) = &self.blocks.get(layer)?.feed_forward else {
            return None;
        };
        Some(eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("{unit_path}.mlp"),
            p.routed.experts.group_count(),
        ))
    }
    /// Normalized family and encoding policy.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// Shared parameter, invocation and mutable-state projection.
    pub fn definition(&self) -> &Arc<eredu_runtime::ComponentDefinition> {
        &self.definition
    }
    /// Projects selected encodings, rejecting changes to family equations or identity.
    pub fn with_formats(&self, args: ModelArgs) -> Result<Self, Error> {
        fn logical(mut a: ModelArgs) -> ModelArgs {
            a.weight_quantization = None;
            a.quantized_weight_configs = None;
            a
        }
        args.validate().map_err(Error::backend_source)?;
        if logical(args.clone()) != logical(self.args.clone()) {
            return Err(Error::backend(
                "Kimi format projection changes retained architecture policy",
            ));
        }
        let mut embedding = self.statics.embedding_spec().clone();
        embedding.format = crate::linear_format::project_matrix_format(
            &embedding.weight,
            &embedding.format,
            args.weight_quantization_for(embedding.weight.id.as_str())
                .into(),
        )?;
        let head = self
            .statics
            .head_spec()
            .cloned()
            .map(|mut p| {
                project(&mut p, &args)?;
                Ok::<_, Error>(p)
            })
            .transpose()?;
        let statics = crate::decoder::StaticModuleConstruction::new(
            embedding,
            head,
            self.statics.normalization_spec().clone(),
        )?;
        let blocks = self
            .blocks
            .iter()
            .cloned()
            .map(|p| p.with_formats(&args))
            .collect::<Result<_, _>>()?;
        Self::from_parts(args, statics, blocks)
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.statics.logical_parameters()?;
        for scope in &self.blocks {
            p.extend(scope.logical_parameters()?);
        }
        Ok(p)
    }
}
