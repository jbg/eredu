//! Immutable LFM2 neural scopes and their ownership projection.
use super::{FeedForwardPolicy, ModelArgs, OperatorPolicy};
use eredu_nn::{
    DeclareLogicalParameters, Error, LogicalParameterDeclaration, NormalizationConstructionSpec,
};
use std::sync::Arc;
#[derive(Debug, Clone)]
pub(crate) enum MixerParameters {
    Attention(
        super::block::AttentionParameters,
        eredu_core::AttentionPolicy,
    ),
    Convolution(eredu_nn::GatedShortConvolutionSpec),
}
impl MixerParameters {
    pub(crate) fn new(
        args: &ModelArgs,
        layer: usize,
        geometry: super::BlockGeometry,
    ) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("LFM2 invocation is outside its schedule"))?;
        let root = format!("model.layers.{layer}");
        Ok(match policy.operator {
            OperatorPolicy::SelfAttention(p) => Self::Attention(
                super::block::AttentionParameters::new(args, &root, geometry)?,
                p,
            ),
            OperatorPolicy::CausalConvolution => Self::Convolution(
                super::block::short_convolution_spec(args, &root, geometry.convolution_channels)?,
            ),
        })
    }
    pub(crate) fn state_policy(&self) -> Result<eredu_core::cache::LayerCachePolicy, Error> {
        use eredu_core::cache::{
            LayerCachePolicy, MutableStateResidency, StateTensorDimension as D, StateTensorDtype,
            StateTensorPolicy, StateTensorRole,
        };
        match self {
            Self::Attention(spec, policy) => LayerCachePolicy::key_value(
                *policy,
                spec.projection.geometry.key_value_heads(),
                spec.projection.geometry.key_dimensions(),
            )
            .map_err(Error::backend_source),
            Self::Convolution(spec) => {
                let history = spec.convolution.history_len()?;
                if history == 0 {
                    return Ok(LayerCachePolicy::NoState);
                }
                let fixed = |n| D::fixed(n).map_err(Error::backend_source);
                LayerCachePolicy::fixed_only(vec![StateTensorPolicy::new(
                    StateTensorRole::Convolution { slot: 0 },
                    vec![D::Batch, fixed(history)?, fixed(spec.convolution.channels)?],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map_err(Error::backend_source)?])
                .map_err(Error::backend_source)
            }
        }
    }
}
#[derive(Debug, Clone)]
pub(crate) enum FeedForwardParameters {
    Dense(eredu_nn::SplitGatedProductSpec),
    Routed(eredu_nn::RoutedGatedProductSpec),
}
#[derive(Debug, Clone)]
pub(crate) struct BlockParameters {
    pub mixer: MixerParameters,
    pub feed_forward: FeedForwardParameters,
    pub operator_norm: NormalizationConstructionSpec,
    pub feed_forward_norm: NormalizationConstructionSpec,
}
impl BlockParameters {
    fn new(args: &ModelArgs, layer: usize) -> Result<Self, Error> {
        let policy = args
            .layer_policy(layer)
            .ok_or_else(|| Error::backend("LFM2 invocation is outside its schedule"))?;
        let root = format!("model.layers.{layer}");
        let mixer = MixerParameters::new(args, layer, super::BlockGeometry::replicated(args))?;
        let feed_forward =
            match policy.feed_forward {
                FeedForwardPolicy::Dense => FeedForwardParameters::Dense(
                    super::moe::dense_parameters(args, layer, args.dense_intermediate_size)?,
                ),
                FeedForwardPolicy::SparseMoe => {
                    FeedForwardParameters::Routed(super::moe::routed_spec(args, layer)?)
                }
            };
        Ok(Self {
            mixer,
            feed_forward,
            operator_norm: super::block::normalization_spec(
                args,
                &format!("{root}.operator_norm.weight"),
                args.hidden_size,
            )?,
            feed_forward_norm: super::block::normalization_spec(
                args,
                &format!("{root}.ffn_norm.weight"),
                args.hidden_size,
            )?,
        })
    }
    pub(crate) fn geometry(&self, args: &ModelArgs) -> super::BlockGeometry {
        let mut geometry = super::BlockGeometry::replicated(args);
        match &self.mixer {
            MixerParameters::Attention(spec, _) => {
                geometry.query_heads = spec.projection.geometry.query_heads();
                geometry.key_value_heads = spec.projection.geometry.key_value_heads();
            }
            MixerParameters::Convolution(spec) => geometry.convolution_channels = spec.channels,
        }
        match &self.feed_forward {
            FeedForwardParameters::Dense(spec) => geometry.dense_intermediate = spec.down.input,
            FeedForwardParameters::Routed(spec) => {
                geometry.expert_intermediate = spec.experts.intermediate_dimensions()
            }
        }
        geometry
    }
    pub(crate) fn local(
        mut self,
        geometry: super::BlockGeometry,
        routed: Option<eredu_nn::GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        self.mixer = match self.mixer {
            MixerParameters::Attention(mut spec, policy) => {
                let old = spec.projection.geometry;
                if geometry.query_heads > old.query_heads()
                    || geometry.key_value_heads > old.key_value_heads()
                {
                    return Err(Error::backend(
                        "local LFM2 attention exceeds declared geometry",
                    ));
                }
                let heads = eredu_nn::AttentionHeadGeometry::new(
                    geometry.query_heads,
                    geometry.key_value_heads,
                    old.key_dimensions(),
                    old.value_dimensions(),
                )?;
                spec.projection.query.output = heads.query_width();
                spec.projection.key.output = heads.key_width();
                spec.projection.value.output = heads.value_width();
                spec.projection.output.input = heads.attended_width();
                spec.projection.geometry = heads;
                MixerParameters::Attention(spec, policy)
            }
            MixerParameters::Convolution(mut spec) => {
                let channels = geometry.convolution_channels;
                if channels <= 0 || channels > spec.channels {
                    return Err(Error::backend(
                        "local LFM2 convolution exceeds declared channels",
                    ));
                }
                spec.input_projection.output =
                    eredu_nn::GatedShortConvolutionSpec::input_layout(channels)?.output_width();
                spec.output_projection.input = channels;
                spec.convolution.channels = channels;
                spec.channels = channels;
                MixerParameters::Convolution(spec)
            }
        };
        self.feed_forward = match self.feed_forward {
            FeedForwardParameters::Dense(mut spec) => {
                if routed.is_some() {
                    return Err(Error::backend("dense LFM2 scope cannot own a routed bank"));
                }
                let width = geometry.dense_intermediate;
                if width <= 0 || width > spec.down.input {
                    return Err(Error::backend(
                        "local LFM2 dense width exceeds its declaration",
                    ));
                }
                spec.gate.output = width;
                spec.up.output = width;
                spec.down.input = width;
                FeedForwardParameters::Dense(spec)
            }
            FeedForwardParameters::Routed(mut spec) => {
                if geometry.expert_intermediate <= 0
                    || geometry.expert_intermediate > spec.experts.intermediate_dimensions()
                {
                    return Err(Error::backend(
                        "local LFM2 expert width exceeds declared geometry",
                    ));
                }
                let bank = match routed {
                    Some(bank) => {
                        if bank.group_count() > spec.experts.group_count()
                            || bank.intermediate_dimensions()
                                > spec.experts.intermediate_dimensions()
                        {
                            return Err(Error::backend(
                                "local LFM2 bank exceeds declared geometry",
                            ));
                        }
                        let expected = spec.experts.clone().with_group_geometry(
                            bank.group_count(),
                            bank.intermediate_dimensions(),
                        )?;
                        if bank != expected {
                            return Err(Error::backend(
                                "local LFM2 bank differs from declared identities, formats or equations",
                            ));
                        }
                        bank
                    }
                    None => spec.experts.clone().with_group_geometry(
                        spec.experts.group_count(),
                        geometry.expert_intermediate,
                    )?,
                };
                if bank.intermediate_dimensions() != geometry.expert_intermediate {
                    return Err(Error::backend(
                        "LFM2 bank and selected intermediate geometry differ",
                    ));
                }
                spec.experts = bank;
                FeedForwardParameters::Routed(spec)
            }
        };
        eredu_nn::validate_logical_parameter_topology(&self.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(self)
    }
    fn with_formats(mut self, args: &ModelArgs) -> Result<Self, Error> {
        let project = |spec: &mut eredu_nn::LinearSpec| -> Result<(), Error> {
            spec.format = crate::linear_format::project_matrix_format(
                &spec.weight,
                &spec.format,
                args.weight_quantization_for(spec.weight.id.as_str()).into(),
            )?;
            Ok(())
        };
        match &mut self.mixer {
            MixerParameters::Attention(spec, _) => {
                project(&mut spec.projection.query)?;
                project(&mut spec.projection.key)?;
                project(&mut spec.projection.value)?;
                project(&mut spec.projection.output)?;
            }
            MixerParameters::Convolution(spec) => {
                project(&mut spec.input_projection)?;
                project(&mut spec.output_projection)?;
            }
        }
        match &mut self.feed_forward {
            FeedForwardParameters::Dense(spec) => {
                project(&mut spec.gate)?;
                project(&mut spec.up)?;
                project(&mut spec.down)?;
            }
            FeedForwardParameters::Routed(spec) => {
                let weight = spec.selector.weight().id.as_str();
                spec.selector = spec.selector.clone().with_format(
                    crate::linear_format::project_matrix_format(
                        spec.selector.weight(),
                        spec.selector.format(),
                        args.weight_quantization_for(weight).into(),
                    )?,
                )?;
                spec.experts =
                    crate::linear_format::project_packed_gated_formats(&spec.experts, |name| {
                        args.weight_quantization_for(name).into()
                    })?;
            }
        }
        Ok(self)
    }
}
impl DeclareLogicalParameters for BlockParameters {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = match &self.mixer {
            MixerParameters::Attention(s, _) => s.logical_parameters()?,
            MixerParameters::Convolution(s) => s.logical_parameters()?,
        };
        p.extend(self.operator_norm.logical_parameters()?);
        p.extend(self.feed_forward_norm.logical_parameters()?);
        p.extend(match &self.feed_forward {
            FeedForwardParameters::Dense(s) => s.logical_parameters()?,
            FeedForwardParameters::Routed(s) => s.logical_parameters()?,
        });
        Ok(p)
    }
}
/// Validated static/block scopes paired with exact LFM2 ownership and state contracts.
#[derive(Debug, Clone)]
pub struct Declaration {
    args: ModelArgs,
    pub(crate) statics: crate::decoder::StaticModuleConstruction,
    pub(crate) blocks: Vec<BlockParameters>,
    definition: Arc<eredu_runtime::ComponentDefinition>,
}
impl Declaration {
    /// Declares family parameters without artifact values or backend allocation.
    pub fn new(args: ModelArgs) -> Result<Self, Error> {
        args.validate().map_err(Error::backend_source)?;
        let statics = super::static_spec(&args).construction_specs()?;
        let blocks = (0..args.num_hidden_layers as usize)
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
            .map(|block| block.mixer.state_policy())
            .collect::<Result<Vec<_>, _>>()?;
        let state = eredu_runtime::StateLayout::new(
            eredu_core::LayerSchedule::new(blocks.len(), policies)
                .map_err(Error::backend_source)?,
        )
        .map_err(Error::backend_source)?;
        let definition = Arc::new(super::partitioned::component_definition(parameters, state)?);
        let scope = Self {
            args,
            statics,
            blocks,
            definition,
        };
        eredu_nn::validate_logical_parameter_topology(&scope.logical_parameters()?)
            .map_err(Error::backend_source)?;
        let formats = scope
            .args
            .quantized_weight_configs
            .iter()
            .flat_map(|formats| formats.iter())
            .map(|(name, format)| (name.as_str(), Some((*format).into())));
        let selected = scope
            .args
            .quantized_weights
            .iter()
            .flat_map(|names| names.iter())
            .map(|name| {
                (
                    name.as_str(),
                    scope.args.weight_quantization_for(name).map(Into::into),
                )
            });
        crate::format_references::validate(
            &scope,
            formats.chain(selected),
            || Ok(Vec::new()),
            || super::safetensors_plan(&scope, true),
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
                "LFM2 state geometry differs from declared invocation count",
            ));
        }
        let policies = self
            .blocks
            .iter()
            .cloned()
            .zip(geometry)
            .map(|(scope, geometry)| scope.local(*geometry, None)?.mixer.state_policy())
            .collect::<Result<Vec<_>, _>>()?;
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
                FeedForwardParameters::Routed(spec) => Some((unit, spec.experts.clone())),
                FeedForwardParameters::Dense(_) => None,
            })
            .collect()
    }
    /// Explicit routed hook sites for one declared execution invocation.
    pub fn routed_observation_points(
        &self,
        unit_path: &str,
        layer: usize,
    ) -> Option<eredu_runtime::RoutedObservationPoints> {
        let FeedForwardParameters::Routed(spec) = &self.blocks.get(layer)?.feed_forward else {
            return None;
        };
        Some(eredu_runtime::RoutedObservationPoints::new(
            eredu_runtime::RoutedBankId::new(0),
            format!("{unit_path}.feed_forward"),
            spec.experts.group_count(),
        ))
    }
    /// Normalized family and storage policy.
    pub fn args(&self) -> &ModelArgs {
        &self.args
    }
    /// Parameter ownership, invocations and mutable state projected from the scope.
    pub fn definition(&self) -> &Arc<eredu_runtime::ComponentDefinition> {
        &self.definition
    }
    /// Projects selected formats without rebuilding identities or logical dimensions.
    pub fn with_formats(&self, args: ModelArgs) -> Result<Self, Error> {
        fn logical(mut a: ModelArgs) -> ModelArgs {
            a.weight_quantization = None;
            a.quantized_weights = None;
            a.quantized_weight_configs = None;
            a
        }
        args.validate().map_err(Error::backend_source)?;
        if logical(args.clone()) != logical(self.args.clone()) {
            return Err(Error::backend(
                "LFM2 format projection changes retained architecture policy",
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
            .map(|mut head| {
                head.format = crate::linear_format::project_matrix_format(
                    &head.weight,
                    &head.format,
                    args.weight_quantization_for(head.weight.id.as_str()).into(),
                )?;
                Ok::<_, Error>(head)
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
            .map(|b| b.with_formats(&args))
            .collect::<Result<_, _>>()?;
        Self::from_parts(args, statics, blocks)
    }
}
impl DeclareLogicalParameters for Declaration {
    fn logical_parameters(&self) -> Result<Vec<LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.statics.logical_parameters()?;
        for b in &self.blocks {
            p.extend(b.logical_parameters()?);
        }
        Ok(p)
    }
}
