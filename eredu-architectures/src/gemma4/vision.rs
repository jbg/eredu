//! Backend-neutral Gemma 4 vision configuration and reusable projection units.

pub(crate) mod parameters;
pub mod specs;

use std::collections::{HashMap, HashSet};

use crate::rotary::RopeValue;
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_nn::{
    multimodal::{
        multi_axis_rotary_embeddings, MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    },
    AttentionArithmetic, AttentionRequest, DeclareLogicalParameters, Error, Index, LinearOperator,
    NeuralBackend, NormalizationOperator, Parameter, Parameterized, Tensor, TensorElementType,
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

/// Normalized shared Gemma image-tower geometry and storage policy.
#[derive(Debug, Clone, Deserialize)]
pub struct VisionPolicy {
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
    /// Whether attention rounds scores and probabilities to the input precision.
    #[serde(default)]
    pub input_score_arithmetic: bool,
    /// Whether pooled encoder output uses learned standardization.
    #[serde(default)]
    pub standardize: bool,
    /// Whether projections carry learned input and output clipping bounds.
    #[serde(default = "default_clipped_linears")]
    pub use_clipped_linears: bool,
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

fn default_clipped_linears() -> bool {
    true
}

impl VisionPolicy {
    /// Validates exact executable geometry.
    pub fn validate(&self) -> Result<(), VisionConfigError> {
        for format in self.weight_quantization.iter().chain(
            self.quantized_weight_configs
                .iter()
                .flat_map(|f| f.values()),
        ) {
            format
                .validate()
                .map_err(|e| VisionConfigError::Invalid(e.to_string()))?;
        }

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

/// Family ownership of the shared image equations and canonical parameter roots.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionOwner {
    /// Gemma 4's independently leased image tower.
    Gemma4,
    /// DiffusionGemma's encoder-owned image tower.
    DiffusionGemma,
}
impl VisionOwner {
    /// Authoritative parameter root used by source extensions and invocation binding.
    pub const fn root(self) -> &'static str {
        match self {
            Self::Gemma4 => "model.vision_tower",
            Self::DiffusionGemma => "model.encoder.vision_tower",
        }
    }
}
/// Validated image policy and immutable parameter scopes under one family owner.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "VisionPolicy")]
pub struct VisionConfig {
    policy: VisionPolicy,
    owner: VisionOwner,
    pinned: std::sync::Arc<specs::StaticSpec>,
    layers: std::sync::Arc<Vec<specs::BlockSpec>>,
}
impl VisionConfig {
    /// Validates external policy and prepares all parameter owners before allocation.
    pub fn new(policy: VisionPolicy, owner: VisionOwner) -> Result<Self, VisionConfigError> {
        policy.validate()?;
        let pinned =
            specs::StaticSpec::at(&policy, owner.root()).map_err(vision_declaration_error)?;
        let layers = (0..policy.num_hidden_layers as usize)
            .map(|layer| specs::BlockSpec::at(&policy, layer, owner.root()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(vision_declaration_error)?;
        let exact = Self {
            policy,
            owner,
            pinned: pinned.into(),
            layers: layers.into(),
        };
        exact.validate()?;
        Ok(exact)
    }
    /// Parses a standalone Gemma 4 image configuration and prepares its owners.
    pub fn from_json(bytes: &[u8]) -> Result<Self, VisionConfigError> {
        Ok(serde_json::from_slice(bytes)?)
    }
    /// Source geometry, arithmetic and storage policy.
    pub fn policy(&self) -> &VisionPolicy {
        &self.policy
    }
    /// Owning family, independent of mutable invocation state.
    pub const fn owner(&self) -> VisionOwner {
        self.owner
    }
    pub(crate) fn pinned(&self, root: &str) -> Result<&specs::StaticSpec, Error> {
        self.require_root(root)?;
        Ok(&self.pinned)
    }
    pub(crate) fn layer(&self, index: usize, root: &str) -> Result<&specs::BlockSpec, Error> {
        self.require_root(root)?;
        self.layers
            .get(index)
            .ok_or_else(|| Error::backend("vision block outside its declared stack"))
    }
    pub(crate) fn require_root(&self, root: &str) -> Result<(), Error> {
        if root != self.owner.root() {
            return Err(Error::backend(
                "vision reference belongs to another declared component",
            ));
        }
        Ok(())
    }
    /// Validates complete geometry, topology and storage-extension references.
    pub fn validate(&self) -> Result<(), VisionConfigError> {
        self.policy.validate()?;
        let parameters = self
            .logical_parameters()
            .map_err(vision_declaration_error)?;
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(|e| VisionConfigError::Invalid(e.to_string()))?;
        crate::format_references::validate_matrices(
            parameters,
            super::text::specs::format_references(
                self.weight_quantization,
                self.quantized_weights.as_ref(),
                self.quantized_weight_configs.as_ref(),
            )
            .filter(|(name, _)| {
                !(self.owner == VisionOwner::Gemma4
                    && *name == super::media::ProjectionKind::Vision.gemma_weight())
            }),
        )
        .map_err(vision_declaration_error)?;
        Ok(())
    }
    /// Projects encodings onto retained geometry, preserving alignment and equations.
    pub fn with_formats(
        &self,
        default: Option<WeightQuantization>,
        names: Option<HashSet<String>>,
        formats: Option<HashMap<String, WeightQuantization>>,
    ) -> Result<Self, VisionConfigError> {
        if default == self.weight_quantization
            && names == self.quantized_weights
            && formats == self.quantized_weight_configs
        {
            return Ok(self.clone());
        }
        let mut exact = self.clone();
        exact.policy.weight_quantization = default;
        exact.policy.quantized_weights = names;
        exact.policy.quantized_weight_configs = formats;
        exact.policy.validate()?;
        let mut pinned = self.pinned.as_ref().clone();
        pinned
            .project_formats(&exact.policy)
            .map_err(vision_declaration_error)?;
        let mut layers = self.layers.as_ref().clone();
        for layer in &mut layers {
            layer
                .project_formats(&exact.policy)
                .map_err(vision_declaration_error)?;
        }
        exact.pinned = pinned.into();
        exact.layers = layers.into();
        exact.validate()?;
        Ok(exact)
    }
}
fn vision_declaration_error(error: Error) -> VisionConfigError {
    VisionConfigError::Invalid(error.to_string())
}
impl TryFrom<VisionPolicy> for VisionConfig {
    type Error = VisionConfigError;
    fn try_from(policy: VisionPolicy) -> Result<Self, Self::Error> {
        Self::new(policy, VisionOwner::Gemma4)
    }
}
impl std::ops::Deref for VisionConfig {
    type Target = VisionPolicy;
    fn deref(&self) -> &VisionPolicy {
        &self.policy
    }
}
impl DeclareLogicalParameters for VisionConfig {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.pinned.logical_parameters()?;
        parameters.extend(self.layers.as_slice().logical_parameters()?);
        Ok(parameters)
    }
}

/// Atomic learned input and output clipping bounds.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "T")]
pub struct ClipBounds<T: Tensor> {
    /// Minimum input value.
    pub input_min: Parameter<T>,
    /// Maximum input value.
    pub input_max: Parameter<T>,
    /// Minimum output value.
    pub output_min: Parameter<T>,
    /// Maximum output value.
    pub output_max: Parameter<T>,
}

/// Media projection with an optional, complete set of learned clipping bounds.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct MediaLinear<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Ordinary general linear operator.
    pub linear: B::Linear,
    /// Learned clipping selected by the artifact's equations.
    pub bounds: Option<ClipBounds<B::Tensor>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> MediaLinear<B> {
    /// Builds a projection and its complete clipping slots from a validated declaration.
    pub fn new(
        spec: eredu_nn::ClippedLinearSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.validate()?;
        Ok(Self {
            linear: B::linear(spec.linear, context)?,
            bounds: spec
                .bounds
                .map(|bounds| {
                    let shape = eredu_nn::LinearClippingSpec::SHAPE;
                    Ok::<_, Error>(ClipBounds {
                        input_min: Parameter::unloaded(bounds.input_min, shape, context)?,
                        input_max: Parameter::unloaded(bounds.input_max, shape, context)?,
                        output_min: Parameter::unloaded(bounds.output_min, shape, context)?,
                        output_max: Parameter::unloaded(bounds.output_max, shape, context)?,
                    })
                })
                .transpose()?,
        })
    }

    /// Applies input clip, projection, and output clip without host access.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &self.bounds {
            Some(bounds) => {
                let input = input.clip(
                    bounds.input_min.as_ref(),
                    bounds.input_max.as_ref(),
                    context,
                )?;
                self.linear.forward(&input, context)?.clip(
                    bounds.output_min.as_ref(),
                    bounds.output_max.as_ref(),
                    context,
                )
            }
            None => self.linear.forward(input, context),
        }
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
    fn new(
        projection: eredu_nn::LinearSpec,
        position: specs::PositionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            input_projection: B::linear(projection, context)?,
            position_table: Parameter::unloaded(position.table, &position.shape, context)?,
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
            .multiply(
                &input.position_valid.cast_float(
                    self.position_table
                        .as_ref()
                        .element_type()
                        .ok_or_else(|| Error::backend("unknown vision position precision"))?,
                    context,
                )?,
                context,
            )?;
        let patches = input
            .patches
            .subtract(
                &B::Tensor::full_f32(0.5, &[1], context)?.cast_float(
                    input
                        .patches
                        .element_type()
                        .ok_or_else(|| Error::backend("unknown vision patch precision"))?,
                    context,
                )?,
                context,
            )?
            .multiply_scalar(2.0, context)?
            .cast_float(
                self.position_table
                    .as_ref()
                    .element_type()
                    .ok_or_else(|| Error::backend("unknown vision working precision"))?,
                context,
            )?;
        self.input_projection
            .forward(&patches, context)?
            .add(&positions, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct VisionAttention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    query_heads: i32,
    #[parameter(skip)]
    input_score_arithmetic: bool,
    #[parameter(skip)]
    key_value_heads: i32,
    #[parameter(skip)]
    head_dim: i32,
    query: MediaLinear<B>,
    key: MediaLinear<B>,
    value: MediaLinear<B>,
    output: MediaLinear<B>,
    query_norm: B::Normalization,
    key_norm: B::Normalization,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionAttention<B> {
    fn new(
        spec: specs::AttentionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.logical_parameters()?;
        let heads = spec.projections.projection.geometry;
        let [query, key, value, output] = spec.projections.into_projections()?;
        Ok(Self {
            query_heads: heads.query_heads(),
            input_score_arithmetic: spec.input_score_arithmetic,
            key_value_heads: heads.key_value_heads(),
            head_dim: heads.key_dimensions(),
            query: MediaLinear::new(query, context)?,
            key: MediaLinear::new(key, context)?,
            value: MediaLinear::new(value, context)?,
            output: MediaLinear::new(output, context)?,
            query_norm: B::normalization(spec.query_norm, context)?,
            key_norm: B::normalization(spec.key_norm, context)?,
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
        let query = apply_two_axis_rotary(&query, cosine, sine, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let key = apply_two_axis_rotary(&key, cosine, sine, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let value = value.transpose_axes(&[0, 2, 1, 3], context)?;
        let output = B::attention_with_sinks(
            AttentionRequest {
                arithmetic: if self.input_score_arithmetic {
                    AttentionArithmetic::InputScores
                } else {
                    AttentionArithmetic::Fused
                },
                queries: query,
                keys: key,
                values: value,
                scale: 1.0,
                softcap: None,
                mask: Some(key_mask),
                sinks: None,
            },
            context,
        )?
        .transpose_axes(&[0, 2, 1, 3], context)?
        .reshape(&[batch, sequence, -1], context)?;
        self.output.forward(&output, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct VisionMlp<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    policy: eredu_nn::GatedProductPolicy,
    gate: MediaLinear<B>,
    up: MediaLinear<B>,
    down: MediaLinear<B>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionMlp<B> {
    fn new(spec: specs::MlpSpec, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let policy = spec.product.policy;
        let [gate, up, down] = spec.into_projections()?;
        Ok(Self {
            policy,
            gate: MediaLinear::new(gate, context)?,
            up: MediaLinear::new(up, context)?,
            down: MediaLinear::new(down, context)?,
        })
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let gate = match self.policy.activation() {
            eredu_nn::GatedProductActivation::GeluApproximate => {
                B::gelu_approximate(gate, context)?
            }
            _ => return Err(Error::backend("unsupported Gemma vision gated equation")),
        };
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
        Self::new_at(config, layer, "model.vision_tower", context)
    }

    /// Declares the same vision equations under an architecture-owned tower root.
    pub(crate) fn new_at(
        config: &VisionConfig,
        layer: usize,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = config.layer(layer, root)?.clone();
        let [input, post_attention, pre_feed_forward, post_feed_forward] = spec.norms;
        Ok(Self {
            attention: VisionAttention::new(spec.attention, context)?,
            mlp: VisionMlp::new(spec.mlp, context)?,
            input_norm: B::normalization(input, context)?,
            post_attention_norm: B::normalization(post_attention, context)?,
            pre_feed_forward_norm: B::normalization(pre_feed_forward, context)?,
            post_feed_forward_norm: B::normalization(post_feed_forward, context)?,
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
        let attention = self.attention.forward(
            &self.input_norm.forward(hidden, context)?,
            key_mask,
            cosine,
            sine,
            epsilon,
            context,
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
    }
}

/// Prepared rotary and mask values retained across streamed image blocks.
#[derive(Debug, Clone)]
pub struct VisionState<T> {
    key_mask: T,
    cosine: T,
    sine: T,
    grid_extents: Vec<(i32, i32)>,
}

impl<T> VisionState<T> {
    /// Returns tensors that must remain live while image blocks are submitted.
    pub fn retained_values(&self) -> [&T; 3] {
        [&self.key_mask, &self.cosine, &self.sine]
    }
}

impl<T: Tensor> VisionState<T> {
    /// Prepares parameter-free rotary positions and visibility for any image-stack owner.
    pub fn prepare(
        config: &VisionConfig,
        input: VisionInput<'_, T>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        config.validate().map_err(Error::backend_source)?;
        validate_input(config, &input)?;
        let (cosine, sine) = multi_axis_rotary_embeddings(
            input.position_ids,
            &MultiAxisRotarySpec {
                axes: vec![
                    RotaryAxisSpec {
                        dimensions: config.head_dim / 2,
                        position_offset: 0,
                    },
                    RotaryAxisSpec {
                        dimensions: config.head_dim / 2,
                        position_offset: 0,
                    },
                ],
                base: config.rope_theta(),
                minimum_position: 0,
                layout: MultiAxisRotaryLayout::IndependentAxes,
            },
            context,
        )?;
        Ok(VisionState {
            key_mask: input.key_mask.clone(),
            cosine,
            sine,
            grid_extents: input.grid_extents.to_vec(),
        })
    }

    /// Executes a leased image layer with the request-local rotary and mask state.
    pub fn forward_layer<B: NeuralBackend<Tensor = T> + eredu_nn::DistributedNeuralBackend>(
        &self,
        layer: &mut VisionLayer<B>,
        epsilon: f32,
        hidden: &T,
        context: &T::Context,
    ) -> Result<T, Error> {
        layer.forward(
            hidden,
            &self.key_mask,
            &self.cosine,
            &self.sine,
            epsilon,
            context,
        )
    }
    /// Pools request geometry using only the declared optional standardization parameters.
    pub fn finish(
        &self,
        config: &VisionConfig,
        hidden: &T,
        bias: Option<&T>,
        scale: Option<&T>,
        context: &T::Context,
    ) -> Result<T, Error> {
        if bias.is_some() != scale.is_some() || config.standardize != bias.is_some() {
            return Err(Error::backend(
                "vision pooling standardization differs from its configuration",
            ));
        }

        let element = hidden
            .element_type()
            .ok_or_else(|| Error::backend("unknown pooled vision precision"))?;
        let kernel = config.pooling_kernel_size;
        let mut pooled = Vec::with_capacity(self.grid_extents.len());
        for (batch, (height, width)) in self.grid_extents.iter().copied().enumerate() {
            let real_patches = height * width;
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
                        config.hidden_size,
                    ],
                    context,
                )?;
            // Average each spatial cell as a weighted FP32 product, rounding
            // once to the tower precision before magnitude scaling.
            let item = item
                .transpose_axes(&[0, 1, 3, 2, 4, 5], context)?
                .reshape(
                    &[
                        pooled_height * pooled_width,
                        kernel * kernel,
                        config.hidden_size,
                    ],
                    context,
                )?
                .cast_float(TensorElementType::F32, context)?;
            let weights = T::full_f32(
                1.0 / (kernel * kernel) as f32,
                &[1, kernel * kernel],
                context,
            )?;
            pooled.push(T::matmul(&weights, &item, context)?.reshape(
                &[1, pooled_height * pooled_width, config.hidden_size],
                context,
            )?);
        }
        let mut hidden = T::concatenate(&pooled, 1, context)?
            .cast_float(element, context)?
            .cast_float(TensorElementType::F32, context)?
            .multiply_scalar((config.hidden_size as f32).sqrt(), context)?;
        if let (Some(bias), Some(scale)) = (bias, scale) {
            hidden = hidden
                .subtract(&bias.cast_float(TensorElementType::F32, context)?, context)?
                .multiply(&scale.cast_float(TensorElementType::F32, context)?, context)?;
        }
        hidden.cast_float(element, context)
    }
}

/// Pinned patch, position, pooling, and standardization modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    config: VisionConfig,
    patch_embedder: PatchEmbedder<B>,
    standardization_bias: Option<Parameter<B::Tensor>>,
    standardization_scale: Option<Parameter<B::Tensor>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionStatic<B> {
    /// Builds unloaded pinned image modules.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(config, "model.vision_tower", context)
    }

    /// Declares pinned vision parameters under their exact architecture root.
    pub(crate) fn new_at(
        config: VisionConfig,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = config.pinned(root)?.clone();
        let patch_embedder = PatchEmbedder::new(spec.patch, spec.position, context)?;
        let (standardization_bias, standardization_scale) = match spec.standardization {
            Some(spec) => (
                Some(Parameter::unloaded(spec.bias, &[spec.dimensions], context)?),
                Some(Parameter::unloaded(
                    spec.scale,
                    &[spec.dimensions],
                    context,
                )?),
            ),
            None => (None, None),
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
        validate_input(&self.config, &input)?;
        let hidden = self.patch_embedder.forward(&input, context)?;
        let state = self.prepare_state(input, context)?;
        Ok((hidden, state))
    }

    /// Constructs parameter-free rotary and mask state for a downstream
    /// pipeline owner that receives patch activations from an earlier rank.
    pub fn prepare_state(
        &self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionState<B::Tensor>, Error> {
        VisionState::prepare(&self.config, input, context)
    }

    /// Applies one streamed image block with retained position state.
    pub fn forward_layer(
        &self,
        layer: &mut VisionLayer<B>,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        state.forward_layer(layer, self.config.rms_norm_eps, hidden, context)
    }

    /// Pools spatial patches and applies optional learned standardization.
    pub fn finish(
        &self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        state.finish(
            &self.config,
            hidden,
            self.standardization_bias
                .as_ref()
                .map(|parameter| parameter.as_ref()),
            self.standardization_scale
                .as_ref()
                .map(|parameter| parameter.as_ref()),
            context,
        )
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
        let layers = (0..config.num_hidden_layers as usize)
            .map(|layer| VisionLayer::new(&config, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
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
) -> Result<(), Error> {
    let patch_width = 3 * config.patch_size * config.patch_size;
    if input.patches.shape().len() != 3
        || input.position_ids.shape() != [input.patches.dim(0), input.patches.dim(1), 2]
        || input.position_valid.shape() != [input.patches.dim(0), input.patches.dim(1), 1]
        || input.key_mask.shape() != [input.patches.dim(0), 1, 1, input.patches.dim(1)]
        || input.patches.dim(2) != patch_width
        || input.grid_extents.len() != input.patches.dim(0) as usize
        || input.grid_extents.iter().any(|(height, width)| {
            *height <= 0
                || *width <= 0
                || *height * *width > input.patches.dim(1)
                || *height % config.pooling_kernel_size != 0
                || *width % config.pooling_kernel_size != 0
                || *height > config.position_embedding_size
                || *width > config.position_embedding_size
        })
    {
        return Err(Error::backend("invalid Gemma 4 prepared vision geometry"));
    }
    Ok(())
}

fn apply_two_axis_rotary<T: Tensor>(
    input: &T,
    cosine: &T,
    sine: &T,
    context: &T::Context,
) -> Result<T, Error> {
    let element = input
        .element_type()
        .ok_or_else(|| Error::backend("unknown vision rotary precision"))?;
    let cosine = cosine.cast_float(element, context)?;
    let sine = sine.cast_float(element, context)?;
    let half = input.dim(3) / 2;
    let quarter = half / 2;
    let mut output = Vec::with_capacity(2);
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
