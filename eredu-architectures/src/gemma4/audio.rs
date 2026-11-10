//! Backend-neutral Gemma 4 audio encoder equations.

pub(crate) mod specs;

use std::collections::{HashMap, HashSet};

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_nn::{
    DeclareLogicalParameters, Error, Index, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationOperator, PadMode, Parameter, Parameterized, Tensor,
};
use serde::Deserialize;

use super::vision::MediaLinear;

/// Invalid Gemma audio configuration.
#[derive(Debug, thiserror::Error)]
pub enum AudioConfigError {
    /// JSON decoding failed.
    #[error("invalid Gemma 4 audio configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Geometry or scalar policy is not executable.
    #[error("{0}")]
    Invalid(String),
}

/// Normalized Gemma 4 audio-tower geometry and storage policy.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioPolicy {
    /// Encoder hidden width.
    pub hidden_size: i32,
    /// Encoder layer count.
    pub num_hidden_layers: i32,
    /// Attention head count.
    pub num_attention_heads: i32,
    /// Output feature width consumed by the media projector.
    pub output_proj_dims: i32,
    /// Depthwise temporal convolution kernel.
    pub conv_kernel_size: i32,
    /// Local-attention query chunk width.
    pub attention_chunk_size: i32,
    /// Left attention context including the current position.
    pub attention_context_left: i32,
    /// Right attention context; released Gemma 4 audio uses zero.
    pub attention_context_right: i32,
    /// Additive value used for invalid attention entries.
    pub attention_invalid_logits_value: f32,
    /// Symmetric tanh cap applied to attention logits.
    pub attention_logit_cap: f32,
    /// Feed-forward residual branch multiplier.
    pub residual_weight: f32,
    /// RMS and scale-only layer-normalization epsilon.
    pub rms_norm_eps: f32,
    /// Exactly two convolution channel counts.
    pub subsampling_conv_channels: Vec<i32>,
    /// Whether the checkpoint supplies the audio output-projection bias.
    #[serde(default = "default_true")]
    pub output_projection_bias: bool,
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

const fn default_true() -> bool {
    true
}

impl AudioPolicy {
    /// Validates the complete executable geometry.
    pub fn validate(&self) -> Result<(), AudioConfigError> {
        for format in self.weight_quantization.iter().chain(
            self.quantized_weight_configs
                .iter()
                .flat_map(|formats| formats.values()),
        ) {
            format
                .validate()
                .map_err(|e| AudioConfigError::Invalid(e.to_string()))?;
        }
        for (name, value) in [
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("output_proj_dims", self.output_proj_dims),
            ("conv_kernel_size", self.conv_kernel_size),
            ("attention_chunk_size", self.attention_chunk_size),
        ] {
            if value <= 0 {
                return Err(AudioConfigError::Invalid(format!(
                    "Gemma 4 audio {name} must be positive"
                )));
            }
        }
        if self.hidden_size % self.num_attention_heads != 0 || self.hidden_size % 2 != 0 {
            return Err(AudioConfigError::Invalid(
                "Gemma 4 audio hidden width must be even and divisible by its attention heads"
                    .into(),
            ));
        }
        if self.attention_context_right != 0 || self.attention_context_left <= 1 {
            return Err(AudioConfigError::Invalid(
                "Gemma 4 audio requires zero right context and left context greater than one"
                    .into(),
            ));
        }
        if self.subsampling_conv_channels.len() != 2
            || self
                .subsampling_conv_channels
                .iter()
                .any(|value| *value <= 0)
        {
            return Err(AudioConfigError::Invalid(
                "Gemma 4 audio requires exactly two positive subsampling channel counts".into(),
            ));
        }
        for (name, value) in [
            (
                "attention_invalid_logits_value",
                self.attention_invalid_logits_value,
            ),
            ("attention_logit_cap", self.attention_logit_cap),
            ("residual_weight", self.residual_weight),
            ("rms_norm_eps", self.rms_norm_eps),
        ] {
            if !value.is_finite() {
                return Err(AudioConfigError::Invalid(format!(
                    "Gemma 4 audio {name} must be finite"
                )));
            }
        }
        if self.rms_norm_eps <= 0.0 || self.attention_logit_cap == 0.0 {
            return Err(AudioConfigError::Invalid(
                "Gemma 4 audio epsilon must be positive and its logit cap nonzero".into(),
            ));
        }
        Ok(())
    }

    /// Per-head width.
    pub fn head_dim(&self) -> i32 {
        self.hidden_size / self.num_attention_heads
    }

    /// Returns a weight's exact physical format, retaining released alignment.
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

/// Validated audio policy with immutable, architecture-owned parameter scopes.
#[derive(Debug, Clone, Deserialize)]
#[serde(try_from = "AudioPolicy")]
pub struct AudioConfig {
    policy: AudioPolicy,
    pinned: std::sync::Arc<specs::StaticSpec>,
    layers: std::sync::Arc<Vec<specs::BlockSpec>>,
}
impl AudioConfig {
    /// Validates external policy and declares every source parameter before allocation.
    pub fn new(policy: AudioPolicy) -> Result<Self, AudioConfigError> {
        policy.validate()?;
        let pinned = specs::StaticSpec::new(&policy).map_err(audio_declaration_error)?;
        let layers = (0..policy.num_hidden_layers as usize)
            .map(|layer| specs::BlockSpec::at(&policy, layer))
            .collect::<Result<Vec<_>, _>>()
            .map_err(audio_declaration_error)?;
        let config = Self {
            policy,
            pinned: pinned.into(),
            layers: layers.into(),
        };
        config.validate()?;
        Ok(config)
    }
    /// Parses one embedded audio configuration and retains its validated declarations.
    pub fn from_json(bytes: &[u8]) -> Result<Self, AudioConfigError> {
        Ok(serde_json::from_slice(bytes)?)
    }
    /// Normalized source policy, independent of the retained parameter scopes.
    pub fn policy(&self) -> &AudioPolicy {
        &self.policy
    }
    pub(crate) fn pinned(&self) -> &specs::StaticSpec {
        &self.pinned
    }
    pub(crate) fn layers(&self) -> &[specs::BlockSpec] {
        &self.layers
    }
    pub(crate) fn layer(&self, index: usize) -> Result<&specs::BlockSpec, Error> {
        self.layers
            .get(index)
            .ok_or_else(|| Error::backend("audio block outside its declared stack"))
    }
    /// Validates identities, sharing, geometry, and storage-extension references.
    pub fn validate(&self) -> Result<(), AudioConfigError> {
        self.policy.validate()?;
        let parameters = self.logical_parameters().map_err(audio_declaration_error)?;
        eredu_nn::validate_logical_parameter_topology(&parameters)
            .map_err(|e| AudioConfigError::Invalid(e.to_string()))?;
        crate::format_references::validate_matrices(
            parameters,
            super::text::specs::format_references(
                self.weight_quantization,
                self.quantized_weights.as_ref(),
                self.quantized_weight_configs.as_ref(),
            )
            .filter(|(name, _)| *name != super::media::ProjectionKind::Audio.gemma_weight()),
        )
        .map_err(audio_declaration_error)?;
        Ok(())
    }
    /// Projects encodings onto retained geometry, preserving released input alignment.
    pub fn with_formats(
        &self,
        default: Option<WeightQuantization>,
        weights: Option<HashSet<String>>,
        formats: Option<HashMap<String, WeightQuantization>>,
    ) -> Result<Self, AudioConfigError> {
        if default == self.weight_quantization
            && weights == self.quantized_weights
            && formats == self.quantized_weight_configs
        {
            return Ok(self.clone());
        }
        let mut exact = self.clone();
        exact.policy.weight_quantization = default;
        exact.policy.quantized_weights = weights;
        exact.policy.quantized_weight_configs = formats;
        exact.policy.validate()?;
        let mut pinned = self.pinned.as_ref().clone();
        pinned
            .project_formats(&exact.policy)
            .map_err(audio_declaration_error)?;
        let mut layers = self.layers.as_ref().clone();
        for layer in &mut layers {
            layer
                .project_formats(&exact.policy)
                .map_err(audio_declaration_error)?;
        }
        exact.pinned = pinned.into();
        exact.layers = layers.into();
        exact.validate()?;
        Ok(exact)
    }
    /// Selects the admitted optional bias while retaining its source slot and all layers.
    pub fn with_output_projection_bias(&self, present: bool) -> Self {
        if present == self.output_projection_bias {
            return self.clone();
        }
        let mut exact = self.clone();
        exact.policy.output_projection_bias = present;
        std::sync::Arc::make_mut(&mut exact.pinned).output_bias_enabled = present;
        exact
    }
}
fn audio_declaration_error(error: Error) -> AudioConfigError {
    AudioConfigError::Invalid(error.to_string())
}
impl TryFrom<AudioPolicy> for AudioConfig {
    type Error = AudioConfigError;
    fn try_from(policy: AudioPolicy) -> Result<Self, Self::Error> {
        Self::new(policy)
    }
}
impl std::ops::Deref for AudioConfig {
    type Target = AudioPolicy;
    fn deref(&self) -> &AudioPolicy {
        &self.policy
    }
}
impl DeclareLogicalParameters for AudioConfig {
    fn logical_parameters(&self) -> Result<Vec<eredu_nn::LogicalParameterDeclaration<'_>>, Error> {
        let mut parameters = self.pinned.logical_parameters()?;
        parameters.extend(self.layers.as_slice().logical_parameters()?);
        Ok(parameters)
    }
}

/// Prepared audio input. Subsampling validity materializes the
/// architecture-owned [`super::AudioIngressBatchPlan`] so tower execution
/// never synchronizes a backend mask to the host.
pub struct AudioInput<'a, T> {
    /// Filter-bank features shaped `[1, frames, 128]`.
    pub features: &'a T,
    /// Numeric mask shaped `[1, frames, 1]` applied before the first convolution.
    pub input_mask: &'a T,
    /// Numeric mask shaped `[1, ceil(frames / 2), 1, 1]` applied between convolutions.
    pub first_stage_mask: &'a T,
    /// Valid frame count after the two stride-two convolution layers for each batch item.
    pub valid_subsampled_frames: &'a [i32],
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct SubsampleLayer<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    weight: Parameter<B::Tensor>,
    norm_weight: Parameter<B::Tensor>,
    #[parameter(skip)]
    epsilon: f32,
    #[parameter(skip)]
    arithmetic: eredu_nn::LayerNormalizationArithmetic,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> SubsampleLayer<B> {
    fn new(
        spec: specs::SubsampleSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            weight: Parameter::unloaded(spec.weight, &spec.shape, context)?,
            norm_weight: Parameter::unloaded(
                spec.norm
                    .weight
                    .ok_or_else(|| Error::backend("audio subsampling requires a learned gain"))?,
                &[spec.norm.dimensions],
                context,
            )?,
            epsilon: spec.norm.epsilon,
            arithmetic: spec.norm.arithmetic,
        })
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let convolved = B::Tensor::conv2d(
            input,
            self.weight.as_ref(),
            (2, 2),
            (1, 1),
            (1, 1),
            1,
            context,
        )?;
        B::Tensor::layer_norm(
            &convolved,
            Some(self.norm_weight.as_ref()),
            None,
            self.epsilon,
            self.arithmetic,
            context,
        )?
        .maximum_scalar(0.0, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct SubsampleProjection<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    layer0: SubsampleLayer<B>,
    layer1: SubsampleLayer<B>,
    input_projection: B::Linear,
    #[parameter(skip)]
    input_width: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> SubsampleProjection<B> {
    fn new(
        layers: [specs::SubsampleSpec; 2],
        input: LinearSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let [first, second] = layers;
        let input_width = input.input;
        Ok(Self {
            layer0: SubsampleLayer::new(first, context)?,
            layer1: SubsampleLayer::new(second, context)?,
            input_projection: B::linear(input, context)?,
            input_width,
        })
    }

    fn forward(
        &mut self,
        input: &AudioInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = input
            .features
            .multiply(input.input_mask, context)?
            .expand_dims(-1, context)?;
        let hidden = self.layer0.forward(&hidden, context)?;
        let hidden = self
            .layer1
            .forward(&hidden.multiply(input.first_stage_mask, context)?, context)?;
        let shape = hidden.shape();
        debug_assert_eq!(shape[2] * shape[3], self.input_width);
        self.input_projection.forward(
            &hidden.reshape(&[shape[0], shape[1], shape[2] * shape[3]], context)?,
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct AudioFeedForward<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    pre_norm: B::Normalization,
    first: MediaLinear<B>,
    second: MediaLinear<B>,
    post_norm: B::Normalization,
    #[parameter(skip)]
    residual_weight: f32,
    #[parameter(skip)]
    activation: eredu_nn::UnaryActivation,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> AudioFeedForward<B> {
    fn new(
        spec: specs::FeedForwardSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        eredu_nn::DeclareLogicalParameters::logical_parameters(&spec)?;
        Ok(Self {
            pre_norm: B::normalization(spec.pre, context)?,
            first: MediaLinear::new(
                eredu_nn::ClippedLinearSpec {
                    linear: spec.network.up,
                    bounds: spec.first_bounds,
                },
                context,
            )?,
            second: MediaLinear::new(
                eredu_nn::ClippedLinearSpec {
                    linear: spec.network.down,
                    bounds: spec.second_bounds,
                },
                context,
            )?,
            post_norm: B::normalization(spec.post, context)?,
            residual_weight: spec.residual_weight,
            activation: spec.network.activation,
        })
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.pre_norm.forward(input, context)?;
        let hidden = self
            .activation
            .apply::<B>(self.first.forward(&hidden, context)?, context)?;
        let hidden = self.second.forward(&hidden, context)?;
        input.add(
            &self
                .post_norm
                .forward(&hidden, context)?
                .multiply_scalar(self.residual_weight, context)?,
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct LightConv1d<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    pre_norm: B::Normalization,
    linear_start: MediaLinear<B>,
    depthwise_weight: Parameter<B::Tensor>,
    conv_norm: B::Normalization,
    linear_end: MediaLinear<B>,
    #[parameter(skip)]
    kernel_size: i32,
    #[parameter(skip)]
    hidden_size: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> LightConv1d<B> {
    fn new(
        spec: specs::ConvolutionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let shape = spec.depthwise.shape();
        Ok(Self {
            pre_norm: B::normalization(spec.pre, context)?,
            linear_start: MediaLinear::new(spec.input, context)?,
            depthwise_weight: Parameter::unloaded(spec.depthwise.weight, &shape, context)?,
            conv_norm: B::normalization(spec.norm, context)?,
            linear_end: MediaLinear::new(spec.output, context)?,
            kernel_size: spec.depthwise.kernel_size,
            hidden_size: spec.depthwise.channels,
        })
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let projected = self
            .linear_start
            .forward(&self.pre_norm.forward(input, context)?, context)?;
        let left = projected.index(
            &[Index::Full, Index::Full, Index::Range(0, self.hidden_size)],
            context,
        )?;
        let right = projected.index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(self.hidden_size, 2 * self.hidden_size),
            ],
            context,
        )?;
        let gated = left.multiply(&B::sigmoid(right, context)?, context)?;
        let padded = B::Tensor::pad(
            &gated,
            &[(0, 0), (self.kernel_size - 1, 0), (0, 0)],
            PadMode::Constant,
            context,
        )?;
        let hidden = B::Tensor::conv1d(
            &padded,
            self.depthwise_weight.as_ref(),
            1,
            0,
            1,
            self.hidden_size,
            context,
        )?;
        let hidden = B::silu(self.conv_norm.forward(&hidden, context)?, context)?;
        input.add(&self.linear_end.forward(&hidden, context)?, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct AudioAttention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    query: MediaLinear<B>,
    key: MediaLinear<B>,
    value: MediaLinear<B>,
    output: MediaLinear<B>,
    relative_key: B::Linear,
    per_dimension_scale: Parameter<B::Tensor>,
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    head_dim: i32,
    #[parameter(skip)]
    hidden_size: i32,
    #[parameter(skip)]
    chunk_size: i32,
    #[parameter(skip)]
    past: i32,
    #[parameter(skip)]
    logit_cap: f32,
    #[parameter(skip)]
    invalid_logits: f32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> AudioAttention<B> {
    fn new(
        spec: specs::AttentionSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        spec.logical_parameters()?;
        let heads = spec.projections.projection.geometry;
        let hidden_size = spec.projections.projection.query.input;
        let [query, key, value, output] = spec.projections.into_projections()?;
        Ok(Self {
            query: MediaLinear::new(query, context)?,
            key: MediaLinear::new(key, context)?,
            value: MediaLinear::new(value, context)?,
            output: MediaLinear::new(output, context)?,
            relative_key: B::linear(spec.relative_key, context)?,
            per_dimension_scale: Parameter::unloaded(
                spec.scale.weight,
                &[spec.scale.dimensions],
                context,
            )?,
            heads: heads.query_heads(),
            head_dim: heads.key_dimensions(),
            hidden_size,
            chunk_size: spec.chunk_size,
            past: spec.past,
            logit_cap: spec.logit_cap,
            invalid_logits: spec.invalid_logits,
        })
    }

    fn relative_embeddings(
        &self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let values = relative_embedding_values(self.hidden_size, self.past);
        B::Tensor::from_f32_slice(&values, &[self.past + 1, self.hidden_size], context)
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        valid: &[i32],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let batch = input.dim(0);
        let sequence = input.dim(1);
        let padded_sequence =
            ((sequence + self.chunk_size - 1) / self.chunk_size) * self.chunk_size;
        let hidden = if padded_sequence == sequence {
            input.clone()
        } else {
            B::Tensor::pad(
                input,
                &[(0, 0), (0, padded_sequence - sequence), (0, 0)],
                PadMode::Constant,
                context,
            )?
        };
        let query_scale = B::softplus(self.per_dimension_scale.as_ref().clone(), 1.0, context)?
            .multiply_scalar(
                (self.head_dim as f32).powf(-0.5) / std::f32::consts::LN_2,
                context,
            )?;
        let key_scale = 1.0_f32.exp().ln_1p() / std::f32::consts::LN_2;
        let query = self
            .query
            .forward(&hidden, context)?
            .reshape(
                &[batch, padded_sequence, self.heads, self.head_dim],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .multiply(&query_scale, context)?;
        let key = self
            .key
            .forward(&hidden, context)?
            .reshape(
                &[batch, padded_sequence, self.heads, self.head_dim],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .multiply_scalar(key_scale, context)?;
        let value = self
            .value
            .forward(&hidden, context)?
            .reshape(
                &[batch, padded_sequence, self.heads, self.head_dim],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let relative = self
            .relative_key
            .forward(&self.relative_embeddings(context)?, context)?
            .reshape(&[self.past + 1, self.heads, self.head_dim], context)?
            .transpose_axes(&[1, 0, 2], context)?
            .expand_dims(0, context)?
            .swap_axes(2, 3, context)?;
        let mut outputs = Vec::new();
        for start in (0..padded_sequence).step_by(self.chunk_size as usize) {
            let key_start = (start - self.past).max(0);
            let key_end = (start + self.chunk_size).min(padded_sequence);
            let query_chunk = query.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(start, start + self.chunk_size),
                    Index::Full,
                ],
                context,
            )?;
            let key_chunk = key.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(key_start, key_end),
                    Index::Full,
                ],
                context,
            )?;
            let value_chunk = value.index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(key_start, key_end),
                    Index::Full,
                ],
                context,
            )?;
            let mut logits =
                B::Tensor::matmul(&query_chunk, &key_chunk.swap_axes(2, 3, context)?, context)?;
            let relative_logits = B::Tensor::matmul(&query_chunk, &relative, context)?;
            let key_count = key_end - key_start;
            let mut rows = Vec::with_capacity(self.chunk_size as usize);
            for query_index in 0..self.chunk_size {
                let absolute_query = start + query_index;
                let mut columns = Vec::with_capacity(key_count as usize);
                for key_index in key_start..key_end {
                    let distance = (absolute_query - key_index).clamp(0, self.past - 1);
                    columns.push(relative_logits.index(
                        &[
                            Index::Full,
                            Index::Full,
                            Index::At(query_index),
                            Index::At(self.past - distance),
                        ],
                        context,
                    )?);
                }
                rows.push(B::Tensor::stack(&columns, -1, context)?);
            }
            logits = logits.add(&B::Tensor::stack(&rows, 2, context)?, context)?;
            let masks = valid
                .iter()
                .map(|valid| {
                    let mut mask = Vec::with_capacity((self.chunk_size * key_count) as usize);
                    for query_index in 0..self.chunk_size {
                        let absolute_query = start + query_index;
                        for key_index in key_start..key_end {
                            mask.push(audio_attention_mask_value(
                                absolute_query,
                                key_index,
                                *valid,
                                self.past,
                                self.invalid_logits,
                            ));
                        }
                    }
                    B::Tensor::from_f32_slice(&mask, &[1, self.chunk_size, key_count], context)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mask = B::Tensor::stack(&masks, 0, context)?;
            logits = logits
                .multiply_scalar(1.0 / self.logit_cap, context)?
                .tanh(context)?
                .multiply_scalar(self.logit_cap, context)?
                .add(&mask, context)?;
            let probabilities = logits.softmax_axis(-1, false, context)?;
            outputs.push(B::Tensor::matmul(&probabilities, &value_chunk, context)?);
        }
        let output = B::Tensor::concatenate(&outputs, 2, context)?
            .index(
                &[
                    Index::Full,
                    Index::Full,
                    Index::Range(0, sequence),
                    Index::Full,
                ],
                context,
            )?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, self.heads * self.head_dim], context)?;
        self.output.forward(&output, context)
    }
}

/// One Gemma 4 audio encoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioLayer<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    feed_forward1: AudioFeedForward<B>,
    pre_attention_norm: B::Normalization,
    attention: AudioAttention<B>,
    post_attention_norm: B::Normalization,
    light_convolution: LightConv1d<B>,
    feed_forward2: AudioFeedForward<B>,
    output_norm: B::Normalization,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> AudioLayer<B> {
    /// Builds one unloaded encoder layer with released checkpoint identities.
    pub fn new(
        config: &AudioConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = config.layer(layer)?.clone();
        Ok(Self {
            feed_forward1: AudioFeedForward::new(spec.first, context)?,
            pre_attention_norm: B::normalization(spec.pre_attention, context)?,
            attention: AudioAttention::new(spec.attention, context)?,
            post_attention_norm: B::normalization(spec.post_attention, context)?,
            light_convolution: LightConv1d::new(spec.convolution, context)?,
            feed_forward2: AudioFeedForward::new(spec.second, context)?,
            output_norm: B::normalization(spec.output, context)?,
        })
    }

    /// Applies feed-forward, attention, light convolution, and final normalization.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        valid: &[i32],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.feed_forward1.forward(input, context)?;
        let attended = self.attention.forward(
            &self.pre_attention_norm.forward(&hidden, context)?,
            valid,
            context,
        )?;
        let hidden = hidden.add(
            &self.post_attention_norm.forward(&attended, context)?,
            context,
        )?;
        let hidden = self.light_convolution.forward(&hidden, context)?;
        let hidden = self.feed_forward2.forward(&hidden, context)?;
        self.output_norm.forward(&hidden, context)
    }
}

/// Pinned subsampling and media-to-decoder output projection.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    #[parameter(skip)]
    config: AudioConfig,
    subsampling: SubsampleProjection<B>,
    output_projection: B::Linear,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> AudioStatic<B> {
    /// Builds unloaded pinned audio modules.
    pub fn new(
        config: AudioConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = config.pinned().clone();
        let output_projection = B::linear(spec.output_construction(), context)?;
        let subsampling = SubsampleProjection::new(spec.subsampling, spec.input, context)?;
        Ok(Self {
            config,
            subsampling,
            output_projection,
        })
    }

    /// Subsamples prepared features and returns the host-known valid output extent.
    pub fn begin(
        &mut self,
        input: AudioInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, Vec<i32>), Error> {
        validate_input(&input)?;
        let valid = input.valid_subsampled_frames.to_vec();
        Ok((self.subsampling.forward(&input, context)?, valid))
    }

    /// Crops padding and projects encoder output into the declared media width.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        valid: &[i32],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = valid
            .iter()
            .copied()
            .enumerate()
            .map(|(batch, valid)| {
                hidden.index(
                    &[
                        Index::Range(batch as i32, batch as i32 + 1),
                        Index::Range(0, valid),
                        Index::Full,
                    ],
                    context,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let hidden = B::Tensor::concatenate(&hidden, 1, context)?;
        self.output_projection.forward(&hidden, context)
    }

    /// Validated configuration retained by the tower.
    pub const fn config(&self) -> &AudioConfig {
        &self.config
    }
}

/// Resident Gemma 4 audio tower built from pinned phases and streamable encoder blocks.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct AudioTower<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Pinned subsampling and output projection.
    pub static_modules: AudioStatic<B>,
    /// Independently streamable encoder blocks.
    pub layers: Vec<AudioLayer<B>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> AudioTower<B> {
    /// Builds the unloaded audio tower.
    pub fn new(
        config: AudioConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let layers = (0..config.num_hidden_layers as usize)
            .map(|layer| AudioLayer::new(&config, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            static_modules: AudioStatic::new(config, context)?,
            layers,
        })
    }

    /// Encodes prepared features and crops padded frames before output projection.
    pub fn forward(
        &mut self,
        input: AudioInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let (mut hidden, valid) = self.static_modules.begin(input, context)?;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &valid, context)?;
        }
        self.static_modules.finish(&hidden, &valid, context)
    }

    /// Validated configuration retained by the tower.
    pub const fn config(&self) -> &AudioConfig {
        self.static_modules.config()
    }
}

fn validate_input<T: Tensor>(input: &AudioInput<'_, T>) -> Result<(), Error> {
    let frames = input.features.shape().get(1).copied().unwrap_or(0);
    let first_frames = (frames + 1) / 2;
    let output_frames = (frames + 3) / 4;
    let batch = input.features.shape().first().copied().unwrap_or(0);
    if input.features.shape() != [batch, frames, 128]
        || input.input_mask.shape() != [batch, frames, 1]
        || input.first_stage_mask.shape() != [batch, first_frames, 1, 1]
        || input.valid_subsampled_frames.len() != batch as usize
        || input
            .valid_subsampled_frames
            .iter()
            .any(|valid| *valid < 0 || *valid > output_frames)
    {
        return Err(Error::backend("invalid Gemma 4 prepared audio geometry"));
    }
    Ok(())
}

fn relative_embedding_values(hidden_size: i32, past: i32) -> Vec<f32> {
    let timescales = hidden_size / 2;
    let increment = 10_000.0_f32.ln() / (timescales - 1).max(1) as f32;
    let mut values = Vec::with_capacity(((past + 1) * hidden_size) as usize);
    for position in (0..=past).rev() {
        for index in 0..timescales {
            values.push((position as f32 * (-increment * index as f32).exp()).sin());
        }
        for index in 0..timescales {
            values.push((position as f32 * (-increment * index as f32).exp()).cos());
        }
    }
    values
}

fn audio_attention_mask_value(query: i32, key: i32, valid: i32, past: i32, invalid: f32) -> f32 {
    if key <= query && query - key < past && key < valid && query < valid {
        0.0
    } else {
        invalid
    }
}
