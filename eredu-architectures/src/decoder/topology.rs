//! Shared construction specifications used by module builders and cold preparation.
use super::*;
use eredu_runtime::execution_topology::*;

pub(super) fn linear<C: Config>(
    config: &C,
    prefix: &str,
    field: &str,
    input: i32,
    output: i32,
    bias: bool,
) -> Result<LinearSpec, Error> {
    let weight = format!("{prefix}.{field}.weight");
    Ok(LinearSpec {
        input,
        output,
        weight: parameter_spec(config, &weight).map_err(Error::backend)?,
        bias: bias
            .then(|| parameter_spec(config, format!("{prefix}.{field}.bias")))
            .transpose()
            .map_err(Error::backend)?,
        format: crate::linear_format::standard_linear_format(
            &weight,
            config.linear_format(&weight),
        )?
        .with_compute_element(config.linear_compute_element())?,
    })
}

pub(crate) use super::attention_parameters::{AttentionInputParameters, AttentionParameters};

/// Complete neural declaration consumed by ordinary gated construction and cold projections.
pub(crate) fn gated_parameters<C: Config>(
    config: &C,
    layer: usize,
) -> Result<eredu_nn::GatedProductSpec, Error> {
    let f = config.block_parameter_fields().validate()?;
    let prefix = format!(
        "{}.layers.{layer}.{}",
        config.parameter_root(),
        f.feed_forward
    );
    let affine = |field: &str, input, output| {
        linear(config, &prefix, field, input, output, config.mlp_bias())
    };
    let down = affine(
        f.feed_forward_output,
        config.intermediate_size(),
        config.hidden_size(),
    )?;
    let policy = config.gated_product_policy().unwrap_or_default();
    let parameters = match config.gated_projection_layout() {
        GatedProjectionLayout::Split => {
            eredu_nn::GatedProductSpec::Split(eredu_nn::SplitGatedProductSpec {
                gate: affine(
                    f.feed_forward_gate,
                    config.hidden_size(),
                    config.intermediate_size(),
                )?,
                up: affine(
                    f.feed_forward_up,
                    config.hidden_size(),
                    config.intermediate_size(),
                )?,
                down,
                policy,
            })
        }
        GatedProjectionLayout::Fused { field } => {
            if field.trim().is_empty() {
                return Err(Error::backend(
                    "fused gate/up projection field must not be empty",
                ));
            }
            eredu_nn::GatedProductSpec::Fused(eredu_nn::FusedGatedProductSpec {
                gate_up: affine(
                    field,
                    config.hidden_size(),
                    config
                        .intermediate_size()
                        .checked_mul(2)
                        .ok_or_else(|| Error::backend("fused gate width overflow"))?,
                )?,
                down,
                policy,
            })
        }
    };
    parameters.logical_parameters()?;
    Ok(parameters)
}

pub(super) fn text_from_declaration(
    config: &impl Config,
    declaration: &DecoderDeclaration,
) -> Result<TextExecutionTopology, Error> {
    declaration.validate_for(config)?;
    text_with_scopes(config, declaration, |layer| {
        let parameters = declaration.blocks[layer]
            .feed_forward
            .as_ref()
            .ok_or_else(|| Error::backend("dense topology has no gated declaration"))?;
        let intermediate_size = u64::try_from(parameters.down().input)
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::backend("gated intermediate width must be positive"))?;
        Ok(FeedForwardTopology::Gated {
            intermediate_size,
            projections: parameters
                .projections()
                .map(ProjectionTopology::from_spec)
                .collect::<Result<_, _>>()?,
        })
    })
}

pub(crate) fn text_with_scopes(
    config: &impl Config,
    declaration: &DecoderDeclaration,
    mut feed_forward: impl FnMut(usize) -> Result<FeedForwardTopology, Error>,
) -> Result<TextExecutionTopology, Error> {
    config.validate_config()?;
    let positive = |n: i32| {
        u64::try_from(n)
            .ok()
            .filter(|n| *n > 0)
            .ok_or_else(|| Error::backend("text topology dimensions must be positive"))
    };
    let mut missing = Vec::new();
    let layers = (0..usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?)
        .map(|layer| {
            let owner = config.block_parameter_owner(layer);
            let attention = &declaration.blocks[owner].attention;
            let geometry = attention.geometry;
            if matches!(
                &attention.input,
                AttentionInputParameters::Split { value: None, .. }
            ) {
                missing.push(format!(
                    "layer {layer}: external attention-value provider invocation is not described"
                ));
            }
            if attention.gate.is_some() {
                missing.push(format!(
                    "layer {layer}: gated attention output activation is not described"
                ));
            }
            Ok(TextLayerTopology {
                input_projections: Vec::new(),
                mixer: TokenMixerTopology::Attention {
                    query_heads: positive(geometry.query_heads())?,
                    kv_heads: positive(geometry.key_value_heads())?,
                    key_width: positive(geometry.key_dimensions())?,
                    value_width: positive(geometry.value_dimensions())?,
                    input_scores: config.attention_arithmetic()
                        == eredu_nn::AttentionArithmetic::InputScores,
                    float32_attention: config.attention_arithmetic()
                        == eredu_nn::AttentionArithmetic::Float32,
                    softcap: config.attention_softcap().is_some(),
                    sinks: config.learned_attention_sinks(),
                    projections: attention
                        .projections()
                        .into_iter()
                        .map(ProjectionTopology::from_spec)
                        .collect::<Result<_, _>>()?,
                    output_gate: false,
                    query_key_normalization: config.query_key_norm_epsilon().is_some(),
                    rotary: config.rotary_enabled(),
                },
                feed_forward: feed_forward(owner)?,
                normalization_count: declaration.blocks[layer].normalizations.iter().count() as u64
                    + u64::from(declaration.blocks[layer].terminal.is_some()),
            })
        })
        .collect::<Result<_, Error>>()?;
    let output = ProjectionTopology::from_spec(&declaration.statics.output_projection_spec())?;
    Ok(TextExecutionTopology {
        iterative: None,
        hidden_size: positive(declaration.statics.embedding_spec().dimensions)?,
        vocabulary_size: positive(declaration.statics.embedding_spec().vocabulary)?,
        layers,
        output,
        output_invocations: 1,
        output_softcap: config.output_softcap().is_some(),
        selected_parameter_promotion_bytes: None,
        selected_parameter_promotion_payloads: Default::default(),
        projection_storage: Default::default(),
        projection_input_normalizations: Default::default(),
        f32_rms_normalization_gains: Default::default(),
        missing,
    })
}

/// Full residual normalization policy used by construction and cold declarations.
pub(crate) fn normalization_spec(
    config: &impl Config,
    name: impl Into<String>,
) -> Result<NormalizationConstructionSpec, Error> {
    let spec = NormalizationConstructionSpec {
        reduction: eredu_nn::NormalizationReduction::Native,
        groups: config.normalization_groups(),
        dimensions: config.hidden_size(),
        epsilon: config.rms_norm_epsilon(),
        scale: normalization_scale(
            parameter_spec(config, name).map_err(Error::backend)?,
            config.normalization_offset(),
        ),
    };
    spec.validate()?;
    Ok(spec)
}

pub(crate) fn terminal_normalization_spec(
    config: &impl Config,
    layer: usize,
) -> Result<Option<NormalizationConstructionSpec>, Error> {
    config
        .block_output_normalization(layer)
        .map(|name| {
            let spec = NormalizationConstructionSpec::learned(
                config.hidden_size(),
                config.rms_norm_epsilon(),
                parameter_spec(config, name).map_err(Error::backend)?,
            );
            spec.validate()?;
            Ok(spec)
        })
        .transpose()
}

/// Independent Q/K normalization declarations, including head-owned scale geometry.
pub(crate) fn attention_normalizations(
    config: &impl Config,
    layer: usize,
    geometry: eredu_nn::AttentionHeadGeometry,
) -> Result<[Option<NormalizationConstructionSpec>; 2], Error> {
    let Some(epsilon) = config.query_key_norm_epsilon() else {
        return Ok([None, None]);
    };
    let fields = config.block_parameter_fields().validate()?;
    let prefix = format!(
        "{}.layers.{layer}.{}",
        config.parameter_root(),
        fields.attention
    );
    let mut specs = [None, None];
    for (destination, (field, heads)) in specs.iter_mut().zip([
        (fields.attention_query_norm, geometry.query_heads()),
        (fields.attention_key_norm, geometry.key_value_heads()),
    ]) {
        let dimensions = if config.query_key_norm_per_head_weights() {
            heads
                .checked_mul(geometry.key_dimensions())
                .ok_or_else(|| Error::backend("attention normalization width overflow"))?
        } else {
            geometry.key_dimensions()
        };
        let weight =
            parameter_spec(config, format!("{prefix}.{field}.weight")).map_err(Error::backend)?;
        let mut spec = NormalizationConstructionSpec::learned(dimensions, epsilon, weight.clone());
        spec.groups = if config.query_key_norm_per_head_weights() {
            Some(heads)
        } else {
            config.query_key_normalization_groups()
        };
        spec.scale = normalization_scale(weight, config.query_key_normalization_offset());
        spec.validate()?;
        *destination = Some(spec);
    }
    Ok(specs)
}

/// Residual normalizations shared by construction and logical projections.
#[derive(Debug, Clone)]
pub(crate) struct BlockNormalizations {
    pub input: NormalizationConstructionSpec,
    pub post_attention: NormalizationConstructionSpec,
    pub attention_output: Option<NormalizationConstructionSpec>,
    pub feed_forward_output: Option<NormalizationConstructionSpec>,
}
impl BlockNormalizations {
    pub fn iter(&self) -> impl Iterator<Item = &NormalizationConstructionSpec> {
        [&self.input, &self.post_attention]
            .into_iter()
            .chain(self.attention_output.iter())
            .chain(self.feed_forward_output.iter())
    }
}
pub(crate) fn block_normalizations(
    config: &impl Config,
    layer: usize,
) -> Result<BlockNormalizations, Error> {
    let fields = config.block_parameter_fields().validate()?;
    let prefix = format!("{}.layers.{layer}", config.parameter_root());
    Ok(BlockNormalizations {
        input: normalization_spec(config, format!("{prefix}.{}.weight", fields.input_norm))?,
        post_attention: normalization_spec(
            config,
            format!("{prefix}.{}.weight", fields.post_attention_norm),
        )?,
        attention_output: config
            .attention_output_normalization(layer)
            .map(|name| normalization_spec(config, name))
            .transpose()?,
        feed_forward_output: config
            .feed_forward_output_normalization(layer)
            .map(|name| normalization_spec(config, name))
            .transpose()?,
    })
}
pub(crate) fn final_normalization_spec(
    config: &impl Config,
) -> Result<NormalizationConstructionSpec, Error> {
    normalization_spec(config, format!("{}.norm.weight", config.parameter_root()))
}
/// Head sink ownership; presence does not establish a collector or backend capability.
#[derive(Debug, Clone)]
pub(crate) struct AttentionSinkSpec {
    pub weight: ParameterSpec,
    pub heads: i32,
}
impl DeclareLogicalParameters for AttentionSinkSpec {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        Ok(vec![eredu_nn::LogicalParameterDeclaration::new(
            &self.weight,
            &[self.heads],
            eredu_nn::LogicalParameterRole::AttentionSink,
        )?])
    }
}
pub(crate) fn attention_sink(
    config: &impl Config,
    layer: usize,
    geometry: eredu_nn::AttentionHeadGeometry,
) -> Result<Option<AttentionSinkSpec>, Error> {
    if !config.learned_attention_sinks() {
        return Ok(None);
    }
    let fields = config.block_parameter_fields().validate()?;
    Ok(Some(AttentionSinkSpec {
        weight: parameter_spec(
            config,
            format!(
                "{}.layers.{layer}.{}.{}",
                config.parameter_root(),
                fields.attention,
                fields.attention_sinks
            ),
        )
        .map_err(Error::backend)?,
        heads: geometry.query_heads(),
    }))
}
/// Owned ordinary block slots composed from the factories used by neural construction.
#[derive(Debug, Clone)]
pub(crate) struct BlockParameters {
    pub attention: AttentionParameters,
    pub feed_forward: Option<eredu_nn::GatedProductSpec>,
    pub normalizations: BlockNormalizations,
    pub query_key: [Option<NormalizationConstructionSpec>; 2],
    pub sinks: Option<AttentionSinkSpec>,
    pub terminal: Option<NormalizationConstructionSpec>,
}
impl BlockParameters {
    /// Projects selected widths through retained identities and arithmetic contracts.
    pub(crate) fn with_geometry(
        &self,
        geometry: eredu_nn::AttentionHeadGeometry,
        intermediate: Option<i32>,
        per_head_normalization: bool,
    ) -> Result<Self, Error> {
        let old = self.attention.geometry;
        if geometry.query_heads() > old.query_heads()
            || geometry.key_value_heads() > old.key_value_heads()
            || geometry.key_dimensions() != old.key_dimensions()
            || geometry.value_dimensions() != old.value_dimensions()
        {
            return Err(Error::backend(
                "local attention exceeds retained head geometry",
            ));
        }
        let mut local = self.clone();
        match &mut local.attention.input {
            AttentionInputParameters::Split { query, key, value } => {
                query.output = geometry.query_width();
                key.output = geometry.key_width();
                if let Some(value) = value {
                    value.output = geometry.value_width();
                }
            }
            AttentionInputParameters::Fused { projection, layout } => {
                *layout = FusedProjectionLayout::new(
                    layout
                        .segments()
                        .iter()
                        .map(|segment| {
                            let width = match segment.name() {
                                "query" => geometry.query_width(),
                                "key" => geometry.key_width(),
                                "value" => geometry.value_width(),
                                _ => return Err(Error::backend("unknown declared QKV segment")),
                            };
                            FusedProjectionSegment::new(segment.name(), width)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                projection.output = layout.output_width();
            }
        }
        if let Some((gate, _)) = &mut local.attention.gate {
            gate.output = geometry.attended_width();
        }
        local.attention.output.input = geometry.attended_width();
        local.attention.geometry = geometry;
        for (norm, (heads, old_heads)) in local.query_key.iter_mut().zip([
            (geometry.query_heads(), old.query_heads()),
            (geometry.key_value_heads(), old.key_value_heads()),
        ]) {
            if let Some(norm) = norm {
                if per_head_normalization {
                    if norm.dimensions != old_heads * old.key_dimensions()
                        || norm.groups != Some(old_heads)
                    {
                        return Err(Error::backend("retained normalization is not head-owned"));
                    }
                    norm.dimensions = heads * geometry.key_dimensions();
                    norm.groups = Some(heads);
                }
                norm.validate()?;
            }
        }
        if let Some(sinks) = &mut local.sinks {
            sinks.heads = geometry.query_heads();
        }
        match (&mut local.feed_forward, intermediate) {
            (Some(spec), Some(width)) => {
                *spec = spec.with_intermediate_dimensions(width)?;
            }
            (None, None) => {}
            _ => {
                return Err(Error::backend(
                    "local feed-forward does not match its declared slots",
                ));
            }
        }
        eredu_nn::validate_logical_parameter_scope(&local.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(local)
    }

    pub fn new(config: &impl Config, layer: usize) -> Result<Self, Error> {
        Self::with_feed_forward(config, layer, Some(gated_parameters(config, layer)?))
    }
    pub fn with_feed_forward(
        config: &impl Config,
        layer: usize,
        feed_forward: Option<eredu_nn::GatedProductSpec>,
    ) -> Result<Self, Error> {
        let attention = AttentionParameters::new(config, layer)?;
        let geometry = attention.geometry;
        Ok(Self {
            attention,
            feed_forward,
            normalizations: block_normalizations(config, layer)?,
            query_key: attention_normalizations(config, layer, geometry)?,
            sinks: attention_sink(config, layer, geometry)?,
            terminal: terminal_normalization_spec(config, layer)?,
        })
    }
}
impl DeclareLogicalParameters for BlockParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut p = Vec::new();
        p.extend(self.attention.logical_parameters()?);
        if let Some(spec) = &self.feed_forward {
            p.extend(spec.logical_parameters()?);
        }
        for n in self
            .normalizations
            .iter()
            .chain(self.query_key.iter().flatten())
            .chain(self.terminal.iter())
        {
            p.extend(n.logical_parameters()?);
        }
        if let Some(s) = &self.sinks {
            p.extend(s.logical_parameters()?);
        }
        Ok(p)
    }
}
/// Supplied-embedding dense stack, without a vocabulary table or language-model head.
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedParameters {
    identity: String,
    pub blocks: Vec<BlockParameters>,
    pub normalization: NormalizationConstructionSpec,
}
impl EmbeddedParameters {
    pub fn new(config: &impl Config) -> Result<Self, Error> {
        config.validate_config()?;
        let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
        let spec = Self {
            identity: config.architecture_fingerprint(),
            blocks: (0..layers)
                .map(|layer| BlockParameters::new(config, layer))
                .collect::<Result<_, _>>()?,
            normalization: final_normalization_spec(config)?,
        };
        eredu_nn::validate_logical_parameter_topology(&spec.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(spec)
    }
    /// Projects retained block slots into a selected policy without rebuilding identities.
    /// The caller owns the transformation; topology and block coverage remain validated.
    pub fn map_blocks(
        &self,
        config: &impl Config,
        mut project: impl FnMut(usize, &BlockParameters) -> Result<BlockParameters, Error>,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        if self.blocks.len() != config.num_hidden_layers() as usize {
            return Err(Error::backend("embedded projection changes block coverage"));
        }
        let selected = Self {
            identity: config.architecture_fingerprint(),
            blocks: self
                .blocks
                .iter()
                .enumerate()
                .map(|(i, b)| project(i, b))
                .collect::<Result<_, _>>()?,
            normalization: self.normalization.clone(),
        };
        eredu_nn::validate_logical_parameter_topology(&selected.logical_parameters()?)
            .map_err(Error::backend_source)?;
        Ok(selected)
    }
    pub fn validate_for(&self, config: &impl Config) -> Result<(), Error> {
        if self.identity != config.architecture_fingerprint() {
            return Err(Error::backend(
                "embedded declaration differs from retained construction policy",
            ));
        }
        Ok(())
    }
}
impl DeclareLogicalParameters for EmbeddedParameters {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut p = self.blocks[..].logical_parameters()?;
        p.extend(self.normalization.logical_parameters()?);
        Ok(p)
    }
}
