//! Backend-neutral Gemma 4 vision configuration and reusable projection units.

use std::collections::{HashMap, HashSet};

use crate::{rotary::RopeValue, decoder::identity::Metadata};
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_nn::{
    multimodal::{
        multi_axis_rotary_embeddings, multi_axis_rotary_embeddings_with_metadata, MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    },
    Error, Index, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, Parameter, Parameterized, Tensor,
};
use serde::Deserialize;

/// Invalid Gemma vision configuration.
#[derive(Debug, thiserror::Error)]
pub enum VisionConfigError {
    /// JSON decoding failed.
    #[error("invalid Gemma 4 vision configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Geometry or activation changes unsupported semantics.
    #[error("{0}")]
    Invalid(String),
}

/// Validated Gemma 4 image-tower geometry.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    /// Encoder hidden width.
    pub hidden_size: i32,
    /// Gated-GELU intermediate width.
    pub intermediate_size: i32,
    /// Encoder block count.
    pub num_hidden_layers: i32,
    /// Query head count.
    pub num_attention_heads: i32,
    /// Key/value head count.
    pub num_key_value_heads: i32,
    /// Per-head width.
    pub head_dim: i32,
    /// Input image patch edge.
    pub patch_size: i32,
    /// Square spatial pooling kernel.
    pub pooling_kernel_size: i32,
    /// Learned position-table length per axis.
    pub position_embedding_size: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Exact supported gated activation spelling.
    #[serde(default = "default_hidden_activation")]
    pub hidden_activation: String,
    /// Whether pooled encoder output uses learned standardization.
    #[serde(default)]
    pub standardize: bool,
    /// Optional rotary base metadata.
    #[serde(default)]
    pub rope_parameters: Option<HashMap<String, RopeValue>>,
    /// Preferred model-wide media weight encoding.
    #[serde(default)]
    pub weight_quantization: Option<WeightQuantization>,
    /// Exact quantized media weights in a mixed artifact.
    #[serde(default)]
    pub quantized_weights: Option<HashSet<String>>,
    /// Per-weight mixed encodings.
    #[serde(default)]
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

fn default_hidden_activation() -> String {
    "gelu_pytorch_tanh".into()
}

impl VisionConfig {
    /// Parses and validates a standalone embedded vision config object.
    pub fn from_json(bytes: &[u8]) -> Result<Self, VisionConfigError> {
        let config: Self = serde_json::from_slice(bytes)?;
        config.validate()?;
        Ok(config)
    }

    /// Validates exact executable geometry.
    pub fn validate(&self) -> Result<(), VisionConfigError> {
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("patch_size", self.patch_size),
            ("pooling_kernel_size", self.pooling_kernel_size),
            ("position_embedding_size", self.position_embedding_size),
        ] {
            if value <= 0 {
                return Err(VisionConfigError::Invalid(format!(
                    "Gemma 4 vision {name} must be positive"
                )));
            }
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 || self.head_dim % 4 != 0 {
            return Err(VisionConfigError::Invalid(
                "Gemma 4 vision requires integral grouped-query heads and head_dim divisible by four"
                    .into(),
            ));
        }
        if !self.rms_norm_eps.is_finite()
            || self.rms_norm_eps <= 0.0
            || !self.rope_theta().is_finite()
            || self.rope_theta() <= 0.0
        {
            return Err(VisionConfigError::Invalid(
                "Gemma 4 vision epsilon and rotary base must be finite and positive".into(),
            ));
        }
        if !matches!(
            self.hidden_activation.as_str(),
            "gelu_pytorch_tanh" | "gelu_new"
        ) {
            return Err(VisionConfigError::Invalid(format!(
                "unsupported Gemma 4 vision activation {:?}",
                self.hidden_activation
            )));
        }
        Ok(())
    }

    /// Returns the normalized two-axis rotary base.
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.get("rope_theta"))
            .and_then(|value| match value {
                RopeValue::Float(value) => Some(*value),
                RopeValue::String(value) => value.parse().ok(),
                RopeValue::Bool(_) => None,
            })
            .unwrap_or(100.0)
    }

    /// Returns a weight's exact physical format, retaining the released
    /// projection alignment restriction.
    pub fn linear_format_for(&self, name: &str, input: i32) -> LinearFormat {
        let format = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|formats| formats.get(name))
            .copied()
            .or_else(|| {
                self.weight_quantization.filter(|_| {
                    self.quantized_weights
                        .as_ref()
                        .is_none_or(|weights| weights.contains(name))
                })
            });
        match format {
            Some(format) if input > 0 && input % format.group_size() == 0 && input % 32 == 0 => {
                format.into()
            }
            _ => LinearFormat::Dense,
        }
    }
}

/// Architecture-specific projection clipped by four learned scalar bounds.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ClippedLinear<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Ordinary general linear operator.
    pub linear: B::Linear,
    /// Minimum input value.
    pub input_min: Parameter<B::Tensor>,
    /// Maximum input value.
    pub input_max: Parameter<B::Tensor>,
    /// Minimum output value.
    pub output_min: Parameter<B::Tensor>,
    /// Maximum output value.
    pub output_max: Parameter<B::Tensor>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> ClippedLinear<B> {
    /// Builds one unloaded clipped projection under its checkpoint prefix.
    pub fn new(
        config: &VisionConfig,
        prefix: &str,
        input: i32,
        output: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &<B::Tensor as Tensor>::Context, LinearFormat)>()?;
        let weight_name = metadata.text(format_args!("{prefix}.linear.weight"))?;
        Self::from_format(
            prefix,
            input,
            output,
            config.linear_format_for(&weight_name, input),
            context,
        )
    }

    /// Builds a clipped projection from an already-normalized physical format.
    pub fn from_format(
        prefix: &str,
        input: i32,
        output: i32,
        format: LinearFormat,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &<B::Tensor as Tensor>::Context, LinearSpec, Parameter<B::Tensor>, LinearFormat)>()?;
        let parameter = |suffix: &str| {
            let spec =
                metadata.named_parameter(format_args!("{prefix}.{suffix}"))?;
            Parameter::unloaded(spec, &[], context)
        };
        metadata.borrowed_controls(&parameter)?;
        let weight_name = metadata.text(format_args!("{prefix}.linear.weight"))?;
        Ok(Self {
            linear: B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: metadata.plain_parameter(&weight_name)?,
                    bias: None,
                    format: metadata.format(&weight_name, format)?,
                },
                context,
            )?,
            input_min: parameter("input_min")?,
            input_max: parameter("input_max")?,
            output_min: parameter("output_min")?,
            output_max: parameter("output_max")?,
        })
    }

    /// Applies input clip, projection, and output clip without host access.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let input = input.clip(self.input_min.as_ref(), self.input_max.as_ref(), context)?;
        self.linear.forward(&input, context)?.clip(
            self.output_min.as_ref(),
            self.output_max.as_ref(),
            context,
        )
    }
}

/// Prepared image patch request. Position IDs must already replace padded
/// negative coordinates with zero; `position_valid` and `key_mask` materialize
/// the architecture-owned [`super::VisionIngressBatchPlan`] without
/// backend-to-host inspection.
pub struct VisionInput<'a, T> {
    /// Flattened RGB patches shaped `[batch, patches, 3 * patch_size^2]`.
    pub patches: &'a T,
    /// Sanitized X/Y coordinates shaped `[batch, patches, 2]`.
    pub position_ids: &'a T,
    /// Numeric valid-position mask shaped `[batch, patches, 1]`.
    pub position_valid: &'a T,
    /// Additive key padding mask broadcastable to `[batch, heads, query, patches]`.
    pub key_mask: &'a T,
    /// Unpadded `(height, width)` for each batch item.
    pub grid_extents: &'a [(i32, i32)],
}

/// Patch projection plus learned two-axis positions.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct PatchEmbedder<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Flattened patch projection.
    pub input_projection: B::Linear,
    /// Learned table shaped `[2, positions, hidden]`.
    pub position_table: Parameter<B::Tensor>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> PatchEmbedder<B> {
    fn new(config: &VisionConfig, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, LinearSpec, [i32; 3], eredu_checkpoint::LinearFormat)>()?;
        let input = config.patch_size.checked_mul(config.patch_size)
            .and_then(|area| area.checked_mul(3))
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let weight = "model.vision_tower.patch_embedder.input_proj.weight";
        Ok(Self {
            input_projection: B::linear(
                LinearSpec {
                    input: input,
                    output: config.hidden_size,
                    weight: metadata.plain_parameter(weight)?,
                    bias: None,
                    format: metadata.format(
                        weight,
                        config.linear_format_for(weight, input),
                    )?,
                },
                context,
            )?,
            position_table: Parameter::unloaded(
                metadata.plain_parameter("model.vision_tower.patch_embedder.position_embedding_table")?,
                &[2, config.position_embedding_size, config.hidden_size],
                context,
            )?,
        })
    }

    fn forward(
        &mut self,
        input: &VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let x = input
            .position_ids
            .index(&[Index::Full, Index::Full, Index::At(0)], context)?;
        let y = input
            .position_ids
            .index(&[Index::Full, Index::Full, Index::At(1)], context)?;
        let x_table = self
            .position_table
            .as_ref()
            .index(&[Index::At(0), Index::Full, Index::Full], context)?;
        let y_table = self
            .position_table
            .as_ref()
            .index(&[Index::At(1), Index::Full, Index::Full], context)?;
        let positions = x_table
            .take_axis(&x, 0, context)?
            .add(&y_table.take_axis(&y, 0, context)?, context)?
            .multiply(input.position_valid, context)?;
        let patches = input
            .patches
            .multiply_scalar(2.0, context)?
            .add(&B::Tensor::full_f32(-1.0, &[1], context)?, context)?;
        self.input_projection
            .forward(&patches, context)?
            .add(&positions, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct VisionAttention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip, metadata)]
    query_heads: i32,
    #[parameter(skip, metadata)]
    key_value_heads: i32,
    #[parameter(skip, metadata)]
    head_dim: i32,
    query: ClippedLinear<B>,
    key: ClippedLinear<B>,
    value: ClippedLinear<B>,
    output: ClippedLinear<B>,
    query_norm: B::Normalization,
    key_norm: B::Normalization,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionAttention<B> {
    fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &<B::Tensor as Tensor>::Context, NormalizationConstructionSpec, i32, i32)>()?;
        let query_width = config.num_attention_heads.checked_mul(config.head_dim)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let key_value_width = config.num_key_value_heads.checked_mul(config.head_dim)
            .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
        let prefix = metadata.text(format_args!("model.vision_tower.encoder.layers.{layer}.self_attn"))?;
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    config.head_dim,
                    config.rms_norm_eps,
                    metadata.named_parameter(format_args!("{prefix}.{field}.weight"))?,
                ),
                context,
            )
        };
        metadata.borrowed_controls(&norm)?;
        Ok(Self {
            query_heads: config.num_attention_heads,
            key_value_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            query: ClippedLinear::new(
                config,
                &metadata.text(format_args!("{prefix}.q_proj"))?,
                config.hidden_size,
                query_width,
                context,
            )?,
            key: ClippedLinear::new(
                config,
                &metadata.text(format_args!("{prefix}.k_proj"))?,
                config.hidden_size,
                key_value_width,
                context,
            )?,
            value: ClippedLinear::new(
                config,
                &metadata.text(format_args!("{prefix}.v_proj"))?,
                config.hidden_size,
                key_value_width,
                context,
            )?,
            output: ClippedLinear::new(
                config,
                &metadata.text(format_args!("{prefix}.o_proj"))?,
                query_width,
                config.hidden_size,
                context,
            )?,
            query_norm: norm("q_norm")?,
            key_norm: norm("k_norm")?,
        })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        key_mask: &B::Tensor,
        cosine: &B::Tensor,
        sine: &B::Tensor,
        epsilon: f32,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let query = self.query_norm.forward(
            &self
                .query
                .forward(hidden, context)?
                .reshape(&[batch, sequence, self.query_heads, self.head_dim], context)?,
            context,
        )?;
        let key = self.key_norm.forward(
            &self.key.forward(hidden, context)?.reshape(
                &[batch, sequence, self.key_value_heads, self.head_dim],
                context,
            )?,
            context,
        )?;
        let value = B::rms_norm_without_weight(
            &self.value.forward(hidden, context)?.reshape(
                &[batch, sequence, self.key_value_heads, self.head_dim],
                context,
            )?,
            epsilon,
            context,
        )?;
        let query = apply_two_axis_rotary(&query, cosine, sine, context, metadata)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let key = apply_two_axis_rotary(&key, cosine, sine, context, metadata)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let value = value.transpose_axes(&[0, 2, 1, 3], context)?;
        let output = B::attention(query, key, value, 1.0, Some(key_mask), context)?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)?;
        self.output.forward(&output, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct VisionMlp<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    gate: ClippedLinear<B>,
    up: ClippedLinear<B>,
    down: ClippedLinear<B>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionMlp<B> {
    fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &<B::Tensor as Tensor>::Context)>()?;
        let prefix = metadata.text(format_args!("model.vision_tower.encoder.layers.{layer}.mlp"))?;
        Ok(Self {
            gate: ClippedLinear::new(
                config,
                &metadata.text(format_args!("{prefix}.gate_proj"))?,
                config.hidden_size,
                config.intermediate_size,
                context,
            )?,
            up: ClippedLinear::new(
                config,
                &metadata.text(format_args!("{prefix}.up_proj"))?,
                config.hidden_size,
                config.intermediate_size,
                context,
            )?,
            down: ClippedLinear::new(
                config,
                &metadata.text(format_args!("{prefix}.down_proj"))?,
                config.intermediate_size,
                config.hidden_size,
                context,
            )?,
        })
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = B::Tensor::gelu(&self.gate.forward(input, context)?, context)?;
        let up = self.up.forward(input, context)?;
        self.down.forward(&gate.multiply(&up, context)?, context)
    }
}

/// One vision encoder residual block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionLayer<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    attention: VisionAttention<B>,
    mlp: VisionMlp<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
    pre_feed_forward_norm: B::Normalization,
    post_feed_forward_norm: B::Normalization,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionLayer<B> {
    /// Builds one unloaded image encoder block.
    pub fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &<B::Tensor as Tensor>::Context, NormalizationConstructionSpec)>()?;
        let prefix = metadata.text(format_args!("model.vision_tower.encoder.layers.{layer}"))?;
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    config.hidden_size,
                    config.rms_norm_eps,
                    metadata.named_parameter(format_args!("{prefix}.{field}.weight"))?,
                ),
                context,
            )
        };
        metadata.borrowed_controls(&norm)?;
        Ok(Self {
            attention: VisionAttention::new(config, layer, context)?,
            mlp: VisionMlp::new(config, layer, context)?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
            pre_feed_forward_norm: norm("pre_feedforward_layernorm")?,
            post_feed_forward_norm: norm("post_feedforward_layernorm")?,
        })
    }

    /// Applies rotary attention and the gated-GELU residual branch.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        key_mask: &B::Tensor,
        cosine: &B::Tensor,
        sine: &B::Tensor,
        epsilon: f32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_metadata(hidden, key_mask, cosine, sine, epsilon, context, Metadata::new(B::construction_metadata(context)))
    }

    pub(super) fn forward_with_metadata(
        &mut self,
        hidden: &B::Tensor,
        key_mask: &B::Tensor,
        cosine: &B::Tensor,
        sine: &B::Tensor,
        epsilon: f32,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        super::model::forward::with_metadata(metadata, || {
        metadata.controls::<(&Self, &B::Tensor, B::Tensor, B::Tensor, B::Tensor)>()?;
        let attention = self.attention.forward(
            &self.input_norm.forward(hidden, context)?,
            key_mask,
            cosine,
            sine,
            epsilon,
            context, metadata,
        )?;
        let hidden = hidden.add(
            &self.post_attention_norm.forward(&attention, context)?,
            context,
        )?;
        let feed_forward = self.mlp.forward(
            &self.pre_feed_forward_norm.forward(&hidden, context)?,
            context,
        )?;
        hidden.add(
            &self
                .post_feed_forward_norm
                .forward(&feed_forward, context)?,
            context,
        )

        })
    }
}

/// Prepared rotary and mask values retained across streamed image blocks.
#[derive(Debug, Clone)]
pub struct VisionState<T> {
    key_mask: T,
    cosine: T,
    sine: T,
    grid_extents: crate::replicated_text::SharedCompositeConfig<Vec<(i32, i32)>>,
}

impl<T> VisionState<T> {
    /// Returns tensors that must remain live while image blocks are submitted.
    pub fn retained_values(&self) -> [&T; 3] {
        [&self.key_mask, &self.cosine, &self.sine]
    }
}

/// Pinned patch, position, pooling, and standardization modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    patch_embedder: PatchEmbedder<B>,
    standardization_bias: Option<Parameter<B::Tensor>>,
    standardization_scale: Option<Parameter<B::Tensor>>,
    #[parameter(skip, metadata)]
    config: crate::replicated_text::SharedCompositeConfig<VisionConfig>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionStatic<B> {
    /// Builds unloaded pinned image modules.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if B::construction_metadata(context).is_some_and(|m| m.uses_checked_metadata()) {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        config.validate().map_err(Error::backend)?;
        Self::from_source(crate::replicated_text::SharedCompositeConfig::new(
            config, B::construction_metadata(context),
        )?, context)
    }

    // The family constructor has validated this exact immutable child source.
    pub(super) fn from_source(
        config: crate::replicated_text::SharedCompositeConfig<VisionConfig>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, PatchEmbedder<B>, Option<Parameter<B::Tensor>>,
            Option<Parameter<B::Tensor>>, crate::replicated_text::SharedCompositeConfig<VisionConfig>)>()?;
        let patch_embedder = PatchEmbedder::new(&config, context)?;
        let parameter = |name: &str| {
            Parameter::unloaded(
                metadata.plain_parameter(name)?,
                &[config.hidden_size],
                context,
            )
        };
        metadata.borrowed_controls(&parameter)?;
        let (standardization_bias, standardization_scale) = if config.standardize {
            (
                Some(parameter("model.vision_tower.std_bias")?),
                Some(parameter("model.vision_tower.std_scale")?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            config,
            patch_embedder,
            standardization_bias,
            standardization_scale,
        })
    }

    /// Embeds patches and constructs reusable rotary/mask state.
    pub fn begin(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        self.begin_with_metadata(input, context, Metadata::new(B::construction_metadata(context)))
    }

    pub(super) fn begin_with_metadata(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        super::model::forward::with_metadata(metadata, || {
        metadata.controls::<(&Self, VisionInput<'_, B::Tensor>, B::Tensor, VisionState<B::Tensor>)>()?;
        validate_input(&self.config, &input, metadata)?;
        let hidden = self.patch_embedder.forward(&input, context)?;
        let state = self.prepare_state_with_metadata(input, context, metadata)?;
        Ok((hidden, state))

        })
    }

    /// Constructs parameter-free rotary and mask state for a downstream
    /// pipeline owner that receives patch activations from an earlier rank.
    pub fn prepare_state(
        &self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionState<B::Tensor>, Error> {
        self.prepare_state_with_metadata(input, context, Metadata::new(B::construction_metadata(context)))
    }

    pub(super) fn prepare_state_with_metadata(
        &self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<VisionState<B::Tensor>, Error> {
        super::model::forward::with_metadata(metadata, || {
        metadata.controls::<(&Self, VisionInput<'_, B::Tensor>, VisionState<B::Tensor>, MultiAxisRotarySpec, Vec<RotaryAxisSpec>, Vec<(i32,i32)>)>()?;
        validate_input(&self.config, &input, metadata)?;
        let mut axes = metadata.vector(2)?;
        axes.push(RotaryAxisSpec { dimensions: self.config.head_dim / 2, position_offset: 0 });
        axes.push(RotaryAxisSpec { dimensions: self.config.head_dim / 2, position_offset: 0 });
        let spec = MultiAxisRotarySpec {
            axes, base: self.config.rope_theta(), minimum_position: 0,
            layout: MultiAxisRotaryLayout::IndependentAxes,
        };
        let (cosine, sine) = match metadata.context() {
            Some(source) => multi_axis_rotary_embeddings_with_metadata(input.position_ids, &spec, context, source),
            None => multi_axis_rotary_embeddings(input.position_ids, &spec, context),
        }?;
        let mut grids = metadata.vector(input.grid_extents.len())?;
        grids.extend_from_slice(input.grid_extents);
        let grid_extents = crate::replicated_text::SharedCompositeConfig::new(grids, metadata.context())?;
        Ok(VisionState {
            key_mask: input.key_mask.clone(),
            cosine,
            sine,
            grid_extents,
        })

        })
    }

    /// Applies one streamed image block with retained position state.
    pub fn forward_layer(
        &self,
        layer: &mut VisionLayer<B>,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_layer_with_metadata(layer, hidden, state, context, Metadata::new(B::construction_metadata(context)))
    }

    pub(super) fn forward_layer_with_metadata(
        &self,
        layer: &mut VisionLayer<B>,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        super::model::forward::with_metadata(metadata, || {
        metadata.controls::<(&Self, &VisionLayer<B>, &B::Tensor, &VisionState<B::Tensor>)>()?;
        layer.forward_with_metadata(
            hidden,
            &state.key_mask,
            &state.cosine,
            &state.sine,
            self.config.rms_norm_eps,
            context, metadata,
        )

        })
    }

    /// Pools spatial patches and applies optional learned standardization.
    pub fn finish(
        &self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_with_metadata(hidden, state, context, Metadata::new(B::construction_metadata(context)))
    }

    pub(super) fn finish_with_metadata(
        &self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        super::model::forward::with_metadata(metadata, || {
        metadata.controls::<(&Self, &B::Tensor, &VisionState<B::Tensor>, Vec<B::Tensor>, B::Tensor, [Index;3], [i32;6], [i32;3])>()?;
        let kernel = self.config.pooling_kernel_size;
        let mut pooled = metadata.vector(state.grid_extents.len())?;
        for (batch, (height, width)) in state.grid_extents.iter().copied().enumerate() {
            let real_patches = height.checked_mul(width).ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
            let pooled_height = height / kernel;
            let pooled_width = width / kernel;
            let item = hidden
                .index(
                    &[
                        Index::Range(batch as i32, batch as i32 + 1),
                        Index::Range(0, real_patches),
                        Index::Full,
                    ],
                    context,
                )?
                .reshape(
                    &[
                        1,
                        pooled_height,
                        kernel,
                        pooled_width,
                        kernel,
                        self.config.hidden_size,
                    ],
                    context,
                )?;
            let item = B::Tensor::mean_axis(&item, 4, false, context)?;
            pooled.push(B::Tensor::mean_axis(&item, 2, false, context)?.reshape(
                &[1, pooled_height * pooled_width, self.config.hidden_size],
                context,
            )?);
        }
        let mut hidden = B::Tensor::concatenate(&pooled, 1, context)?
            .multiply_scalar((self.config.hidden_size as f32).sqrt(), context)?;
        if let (Some(bias), Some(scale)) = (
            self.standardization_bias.as_ref(),
            self.standardization_scale.as_ref(),
        ) {
            hidden = hidden
                .subtract(bias.as_ref(), context)?
                .multiply(scale.as_ref(), context)?;
        }
        Ok(hidden)

        })
    }
}

/// Resident Gemma 4 vision tower built from the same static phases and streamable blocks.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionTower<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Pinned patch/pooling modules.
    pub static_modules: VisionStatic<B>,
    /// Independently streamable encoder blocks.
    pub layers: Vec<VisionLayer<B>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionTower<B> {
    /// Builds the unloaded image tower.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, &<B::Tensor as Tensor>::Context)>()?;
        let count = usize::try_from(config.num_hidden_layers)
            .map_err(|_| metadata.error(format_args!("invalid Gemma 4 vision layer count")))?;
        let mut layers = metadata.vector(count)?;
        for layer in 0..count { layers.push(VisionLayer::new(&config, layer, context)?); }
        Ok(Self {
            static_modules: VisionStatic::new(config, context)?,
            layers,
        })
    }

    /// Encodes prepared flattened patches and returns pooled media features.
    pub fn forward(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (mut hidden, state) = self.static_modules.begin(input, context)?;
        for layer in &mut self.layers {
            hidden = self
                .static_modules
                .forward_layer(layer, &hidden, &state, context)?;
        }
        self.static_modules.finish(&hidden, &state, context)
    }
}

fn validate_input<T: Tensor>(
    config: &VisionConfig,
    input: &VisionInput<'_, T>,
    metadata: Metadata<'_>,
) -> Result<(), Error> {
    metadata.controls::<(&VisionConfig, &VisionInput<'_,T>, i32)>()?;
    let patch_width = config.patch_size.checked_mul(config.patch_size).and_then(|v|v.checked_mul(3))
        .ok_or(eredu_nn::workspace::WorkspaceMetadataError::Overflow)?;
    if input.patches.shape().len() != 3
        || input.position_ids.shape() != [input.patches.dim(0), input.patches.dim(1), 2]
        || input.position_valid.shape() != [input.patches.dim(0), input.patches.dim(1), 1]
        || input.key_mask.shape() != [input.patches.dim(0), 1, 1, input.patches.dim(1)]
        || input.patches.dim(2) != patch_width
        || input.grid_extents.len() != input.patches.dim(0) as usize
        || input.grid_extents.iter().any(|(height, width)| {
            *height <= 0
                || *width <= 0
                || height.checked_mul(*width).is_none_or(|n|n > input.patches.dim(1))
                || *height % config.pooling_kernel_size != 0
                || *width % config.pooling_kernel_size != 0
                || *height > config.position_embedding_size
                || *width > config.position_embedding_size
        })
    {
        return Err(metadata.error(format_args!("invalid Gemma 4 prepared vision geometry")));
    }
    Ok(())
}

fn apply_two_axis_rotary<T: Tensor>(
    input: &T,
    cosine: &T,
    sine: &T,
    context: &T::Context,
    metadata: Metadata<'_>,
) -> Result<T, Error> {
    metadata.controls::<(Vec<T>, T, T, T, T, T, T, [Index;4], [Index;3], [T;2])>()?;
    let half = input.dim(3) / 2;
    let quarter = half / 2;
    let mut output = metadata.vector(2)?;
    for axis in 0..2 {
        let start = axis * half;
        let end = start + half;
        let part = input.index(
            &[
                Index::Full,
                Index::Full,
                Index::Full,
                Index::Range(start, end),
            ],
            context,
        )?;
        let left = part.index(
            &[
                Index::Full,
                Index::Full,
                Index::Full,
                Index::Range(0, quarter),
            ],
            context,
        )?;
        let right = part.index(
            &[
                Index::Full,
                Index::Full,
                Index::Full,
                Index::Range(quarter, half),
            ],
            context,
        )?;
        let rotated = T::concatenate(&[right.multiply_scalar(-1.0, context)?, left], -1, context)?;
        let axis_cosine = cosine
            .index(
                &[Index::Full, Index::Full, Index::Range(start, end)],
                context,
            )?
            .expand_dims(2, context)?;
        let axis_sine = sine
            .index(
                &[Index::Full, Index::Full, Index::Range(start, end)],
                context,
            )?
            .expand_dims(2, context)?;
        output.push(
            part.multiply(&axis_cosine, context)?
                .add(&rotated.multiply(&axis_sine, context)?, context)?,
        );
    }
    T::concatenate(&output, -1, context)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> VisionConfig {
        VisionConfig::from_json(
            br#"{
                "hidden_size":16,"intermediate_size":32,"num_hidden_layers":2,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "patch_size":4,"pooling_kernel_size":2,"position_embedding_size":16,
                "rms_norm_eps":0.000001
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn validates_two_axis_geometry_activation_and_rotary_base() {
        let config = valid();
        assert_eq!(config.rope_theta(), 100.0);
        let mut invalid = config.clone();
        invalid.head_dim = 6;
        assert!(invalid.validate().is_err());
        invalid = config;
        invalid.hidden_activation = "silu".into();
        assert!(invalid.validate().is_err());
    }
}
