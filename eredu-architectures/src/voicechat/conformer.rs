//! Released causal FastConformer layer equations and bounded streaming state.
//!
//! Attention retains normalized, unprojected inputs; convolution retains GLU
//! outputs. These caches are independent of checkpoint parameters and are
//! replaced only after the complete layer transition succeeds.

use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use eredu_nn::RetainedTensorRoots;

use eredu_nn::{
    Error, Index, LayerNorm, LinearFormatSpec, LinearOperator, LinearSpec, NeuralBackend, PadMode,
    Parameter, ParameterSpec, Parameterized, Tensor,
};

use super::config::PerceptionEncoderConfig;

#[path = "conformer_positions.rs"]
mod positions;
use positions::RelativePositionProjection;

/// Geometry of one released causal FastConformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformerDimensions {
    /// Hidden feature width.
    pub width: i32,
    /// Number of equal-width attention heads.
    pub heads: i32,
    /// Intermediate width of each feed-forward branch.
    pub ff_hidden: i32,
    /// Odd causal depthwise convolution kernel width.
    pub conv_kernel: i32,
    /// Maximum retained attention context, excluding the current frame.
    pub left_context: i32,
}

impl ConformerDimensions {
    /// Resolves the admitted released encoder policy into layer geometry.
    pub fn from_config(config: &PerceptionEncoderConfig) -> Result<Self, ConformerError> {
        if config.self_attention_model != "rel_pos"
            || config.att_context_style != "chunked_limited"
            || config.att_context_size[1] != 0
            || config.conv_context_size != "causal"
            || config.conv_norm_type != "layer_norm"
            || config.use_bias
            || !config.untie_biases
        {
            return Err(ConformerError::Configuration(
                "encoder changes the released causal layer equations",
            ));
        }
        let convert = |value: usize| {
            i32::try_from(value).map_err(|_| {
                ConformerError::Configuration("Conformer dimension exceeds tensor indexing")
            })
        };
        let value = Self {
            width: convert(config.d_model)?,
            heads: convert(config.n_heads)?,
            ff_hidden: convert(
                config
                    .d_model
                    .checked_mul(config.ff_expansion_factor)
                    .ok_or(ConformerError::Configuration("feed-forward width overflow"))?,
            )?,
            conv_kernel: convert(config.conv_kernel_size)?,
            left_context: i32::try_from(config.att_context_size[0]).map_err(|_| {
                ConformerError::Configuration("attention context exceeds tensor indexing")
            })?,
        };
        value.validate()?;
        Ok(value)
    }

    /// Checks geometry before native parameter allocation.
    pub fn validate(self) -> Result<(), ConformerError> {
        if self.width <= 0
            || self.width % 2 != 0
            || self.heads <= 0
            || self.width % self.heads != 0
            || self.ff_hidden <= 0
            || self.width.checked_mul(2).is_none()
            || self.conv_kernel <= 1
            || self.conv_kernel % 2 != 1
            || self.left_context <= 0
        {
            return Err(ConformerError::Configuration(
                "invalid causal Conformer geometry",
            ));
        }
        Ok(())
    }
}

/// Rank-local complete attention heads and feed-forward intermediate channels.
/// Residual features, normalization, convolution and streaming caches retain
/// the global width on every tensor-parallel rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformerLocalDimensions {
    /// Complete local attention heads, including relative-position projections.
    pub heads: i32,
    /// Local intermediate channels in each feed-forward branch.
    pub ff_hidden: i32,
}

impl ConformerLocalDimensions {
    /// Checks equal, nonempty head and intermediate partitions before allocation.
    pub fn validate(self, global: ConformerDimensions) -> Result<(), ConformerError> {
        global.validate()?;
        if self.heads <= 0
            || self.ff_hidden <= 0
            || global.heads % self.heads != 0
            || global.ff_hidden % self.ff_hidden != 0
            || global.heads / self.heads != global.ff_hidden / self.ff_hidden
        {
            return Err(ConformerError::Configuration(
                "Conformer heads and feed-forward channels require equal complete partitions",
            ));
        }
        Ok(())
    }
}

/// A rejected configuration, restored cache or neural operation.
#[derive(Debug, thiserror::Error)]
pub enum ConformerError {
    /// Unsupported equations or overflowing geometry.
    #[error("{0}")]
    Configuration(&'static str),
    /// Input or retained cache does not match the selected block.
    #[error("{0}")]
    State(&'static str),
    /// Input or parameters differ from the selected floating execution policy.
    #[error("{0}")]
    Precision(&'static str),
    /// Backend-neutral numerical failure retaining its original source.
    #[error(transparent)]
    Neural(#[from] Error),
}

/// Retained streaming values for one layer and one uniformly advanced batch.
#[derive(Debug, Clone)]
pub struct ConformerState<T> {
    dimensions: Option<ConformerDimensions>,
    channel: Option<T>,
    convolution: Option<T>,
    valid_channel: usize,
}

impl<T: SnapshotValue> SnapshotValue for ConformerState<T> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        snapshot_value_estimate::<Self>([
            self.channel.estimate_snapshot_value(),
            self.convolution.estimate_snapshot_value(),
        ])
    }
    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            channel: self.channel.copy_snapshot_value(retention)?,
            convolution: self.convolution.copy_snapshot_value(retention)?,
            dimensions: self.dimensions,
            valid_channel: self.valid_channel,
        })
    }
}

impl<T> RetainedTensorRoots<T> for ConformerState<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        if let Some(channel) = &self.channel {
            visitor(channel);
        }
        if let Some(convolution) = &self.convolution {
            visitor(convolution);
        }
    }
}
impl<T> Default for ConformerState<T> {
    fn default() -> Self {
        Self {
            dimensions: None,
            channel: None,
            convolution: None,
            valid_channel: 0,
        }
    }
}
impl<T> ConformerState<T> {
    /// Fixed `[batch, left_context, width]` attention cache; leading unused rows are zero.
    pub fn channel(&self) -> Option<&T> {
        self.channel.as_ref()
    }
    /// Fixed `[batch, kernel - 1, width]` cache of post-GLU convolution inputs.
    pub fn convolution(&self) -> Option<&T> {
        self.convolution.as_ref()
    }
    /// Number of valid suffix rows in the attention cache.
    pub fn valid_channel(&self) -> usize {
        self.valid_channel
    }
}

impl<T: Tensor> ConformerState<T> {
    /// Checks the exact streaming cache before a parameter lease is acquired.
    pub(super) fn validate_retained(
        &self,
        dimensions: ConformerDimensions,
        batch: Option<i32>,
        frames: u64,
    ) -> Result<(), ConformerError> {
        let valid = frames.min(dimensions.left_context as u64) as usize;
        match (self.dimensions, &self.channel, &self.convolution) {
            (None, None, None) if frames == 0 && self.valid_channel == 0 => Ok(()),
            (Some(geometry), Some(channel), Some(convolution))
                if frames > 0
                    && geometry == dimensions
                    && channel.shape().len() == 3
                    && channel.dim(0) > 0
                    && batch.is_none_or(|batch| channel.dim(0) == batch)
                    && channel.shape()
                        == [channel.dim(0), dimensions.left_context, dimensions.width]
                    && convolution.shape()
                        == [channel.dim(0), dimensions.conv_kernel - 1, dimensions.width]
                    && channel.element_type() == Some(eredu_nn::TensorElementType::F32)
                    && convolution.element_type() == Some(eredu_nn::TensorElementType::F32)
                    && self.valid_channel == valid =>
            {
                Ok(())
            }
            _ => Err(ConformerError::State(
                "Conformer retained cache geometry, precision or frontier differs",
            )),
        }
    }
}

/// One independently materializable perception layer, using ordinary neutral mechanisms.
#[derive(Debug, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ConformerBlock<B: NeuralBackend> {
    #[parameter(skip)]
    precision: super::precision::SpeechPrecision,
    #[parameter(skip)]
    dimensions: ConformerDimensions,
    #[parameter(skip)]
    local: ConformerLocalDimensions,
    norm_feed_forward1: LayerNorm<B::Tensor>,
    ff1_up: B::Linear,
    ff1_down: B::Linear,
    norm_self_att: LayerNorm<B::Tensor>,
    query: B::Linear,
    key: B::Linear,
    value: B::Linear,
    attention_output: B::Linear,
    position: RelativePositionProjection<B>,
    pos_bias_u: Parameter<B::Tensor>,
    pos_bias_v: Parameter<B::Tensor>,
    norm_conv: LayerNorm<B::Tensor>,
    pointwise1: Parameter<B::Tensor>,
    depthwise: Parameter<B::Tensor>,
    conv_norm: LayerNorm<B::Tensor>,
    pointwise2: Parameter<B::Tensor>,
    norm_feed_forward2: LayerNorm<B::Tensor>,
    ff2_up: B::Linear,
    ff2_down: B::Linear,
    norm_out: LayerNorm<B::Tensor>,
}

impl<B: NeuralBackend> Clone for ConformerBlock<B> {
    fn clone(&self) -> Self {
        Self {
            precision: self.precision,
            dimensions: self.dimensions,
            local: self.local,
            norm_feed_forward1: self.norm_feed_forward1.clone(),
            ff1_up: self.ff1_up.clone(),
            ff1_down: self.ff1_down.clone(),
            norm_self_att: self.norm_self_att.clone(),
            query: self.query.clone(),
            key: self.key.clone(),
            value: self.value.clone(),
            attention_output: self.attention_output.clone(),
            position: self.position.clone(),
            pos_bias_u: self.pos_bias_u.clone(),
            pos_bias_v: self.pos_bias_v.clone(),
            norm_conv: self.norm_conv.clone(),
            pointwise1: self.pointwise1.clone(),
            depthwise: self.depthwise.clone(),
            conv_norm: self.conv_norm.clone(),
            pointwise2: self.pointwise2.clone(),
            norm_feed_forward2: self.norm_feed_forward2.clone(),
            ff2_up: self.ff2_up.clone(),
            ff2_down: self.ff2_down.clone(),
            norm_out: self.norm_out.clone(),
        }
    }
}

impl<B: NeuralBackend> ConformerBlock<B> {
    /// Selects projection arithmetic; parameters are supplied by the matching retained plan.
    pub fn with_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, ConformerError> {
        precision.validate().map_err(ConformerError::Precision)?;
        self.ff1_up
            .set_compute_element(precision.projection_element())?;
        self.ff1_down
            .set_compute_element(precision.projection_element())?;
        self.query
            .set_compute_element(precision.projection_element())?;
        self.key
            .set_compute_element(precision.projection_element())?;
        self.value
            .set_compute_element(precision.projection_element())?;
        self.position
            .set_compute_element(precision.projection_element())?;
        self.attention_output
            .set_compute_element(precision.projection_element())?;
        self.ff2_up
            .set_compute_element(precision.projection_element())?;
        self.ff2_down
            .set_compute_element(precision.projection_element())?;
        self.precision = precision;
        Ok(self)
    }

    /// Constructs exact canonical parameter identities for one encoder layer.
    pub fn new(
        dimensions: ConformerDimensions,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, ConformerError> {
        Self::new_parallel(
            dimensions,
            ConformerLocalDimensions {
                heads: dimensions.heads,
                ff_hidden: dimensions.ff_hidden,
            },
            layer,
            context,
        )
    }

    /// Constructs complete local heads and intermediate channels while keeping
    /// global residual, convolution and streaming-state geometry.
    pub fn new_parallel(
        dimensions: ConformerDimensions,
        local: ConformerLocalDimensions,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, ConformerError> {
        local.validate(dimensions)?;
        let d = dimensions.width;
        let attention_width = local.heads * (d / dimensions.heads);
        let prefix = format!("stt_model.perception.encoder.layers.{layer}");
        let parameter = |name: &str| {
            let mut spec = ParameterSpec::trainable(format!("{prefix}.{name}"))
                .expect("valid canonical identifier");
            spec.group = Some(format!("voicechat.perception.layer.{layer}"));
            spec
        };
        let norm = |name: &str| {
            LayerNorm::unloaded(
                d,
                1e-5,
                Some(parameter(&format!("{name}.weight"))),
                Some(parameter(&format!("{name}.bias"))),
                context,
            )
        };
        let linear = |name: &str, input: i32, output: i32| {
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: parameter(&format!("{name}.weight")),
                    bias: None,
                    format: LinearFormatSpec::unscaled(eredu_checkpoint::LinearFormat::Dense)?,
                },
                context,
            )
        };
        Ok(Self {
            precision: Default::default(),
            dimensions,
            local,
            norm_feed_forward1: norm("norm_feed_forward1")?,
            ff1_up: linear("feed_forward1.linear1", d, local.ff_hidden)?,
            ff1_down: linear("feed_forward1.linear2", local.ff_hidden, d)?,
            norm_self_att: norm("norm_self_att")?,
            query: linear("self_attn.linear_q", d, attention_width)?,
            key: linear("self_attn.linear_k", d, attention_width)?,
            value: linear("self_attn.linear_v", d, attention_width)?,
            attention_output: linear("self_attn.linear_out", attention_width, d)?,
            position: RelativePositionProjection::new(
                linear("self_attn.linear_pos", d, attention_width)?,
                dimensions.left_context.saturating_add(1),
            ),
            pos_bias_u: Parameter::unloaded(
                parameter("self_attn.pos_bias_u"),
                &[local.heads, d / dimensions.heads],
                context,
            )?,
            pos_bias_v: Parameter::unloaded(
                parameter("self_attn.pos_bias_v"),
                &[local.heads, d / dimensions.heads],
                context,
            )?,
            norm_conv: norm("norm_conv")?,
            pointwise1: Parameter::unloaded(
                parameter("conv.pointwise_conv1.weight"),
                &[2 * d, d, 1],
                context,
            )?,
            depthwise: Parameter::unloaded(
                parameter("conv.depthwise_conv.weight"),
                &[d, 1, dimensions.conv_kernel],
                context,
            )?,
            conv_norm: norm("conv.batch_norm")?,
            pointwise2: Parameter::unloaded(
                parameter("conv.pointwise_conv2.weight"),
                &[d, d, 1],
                context,
            )?,
            norm_feed_forward2: norm("norm_feed_forward2")?,
            ff2_up: linear("feed_forward2.linear1", d, local.ff_hidden)?,
            ff2_down: linear("feed_forward2.linear2", local.ff_hidden, d)?,
            norm_out: norm("norm_out")?,
        })
    }

    /// Logical layer geometry, independent of materialization and execution context.
    pub fn dimensions(&self) -> ConformerDimensions {
        self.dimensions
    }

    /// Selected local projection geometry, independent of native resources.
    pub fn local_dimensions(&self) -> ConformerLocalDimensions {
        self.local
    }

    fn validate(
        &self,
        input: &B::Tensor,
        state: &ConformerState<B::Tensor>,
    ) -> Result<(), ConformerError> {
        let d = self.dimensions;
        let shape = input.shape();
        if shape.len() != 3
            || shape[0] <= 0
            || shape[1] <= 0
            || shape[2] != d.width
            || shape[1]
                .checked_add(d.left_context)
                .and_then(|n| n.checked_mul(2))
                .is_none()
        {
            return Err(ConformerError::State(
                "Conformer input expects positive [batch, time, width]",
            ));
        }
        match (state.dimensions, &state.channel, &state.convolution) {
            (None, None, None) if state.valid_channel == 0 => Ok(()),
            (Some(geometry), Some(channel), Some(convolution))
                if geometry == d
                    && channel.shape() == [shape[0], d.left_context, d.width]
                    && convolution.shape() == [shape[0], d.conv_kernel - 1, d.width]
                    && state.valid_channel <= d.left_context as usize =>
            {
                Ok(())
            }
            _ => Err(ConformerError::State(
                "Conformer cache geometry differs from its layer",
            )),
        }
    }

    fn initialize(
        &self,
        input: &B::Tensor,
        state: &ConformerState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ConformerState<B::Tensor>, ConformerError> {
        if state.channel.is_some() {
            return Ok(state.clone());
        }
        let d = self.dimensions;
        let zero = input
            .index(&[Index::Full, Index::Range(0, 1), Index::Full], context)?
            .zeros_like(context)?;
        Ok(ConformerState {
            dimensions: Some(d),
            channel: Some(
                zero.broadcast_to(&[input.dim(0), d.left_context, d.width], context)?
                    .compact(context)?,
            ),
            convolution: Some(
                zero.broadcast_to(&[input.dim(0), d.conv_kernel - 1, d.width], context)?
                    .compact(context)?,
            ),
            valid_channel: 0,
        })
    }

    /// Computes one causal chunk without mutating the supplied parent cache.
    ///
    /// Batch members have the same valid length. Host framing supplies complete
    /// encoder frames; there are no padded current frames or lookahead to discard.
    pub fn forward(
        &mut self,
        input: &B::Tensor,
        previous: &ConformerState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, ConformerState<B::Tensor>), ConformerError> {
        if self.local.heads != self.dimensions.heads {
            return Err(ConformerError::Configuration(
                "partitioned Conformer projections require their tensor communicator",
            ));
        }
        self.forward_selected(input, previous, None, context)
    }

    /// Executes local projections and reduces all three residual contributions.
    /// The supplied parent cache remains unchanged until the caller commits the
    /// returned full-width successor state.
    pub fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        previous: &ConformerState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, ConformerState<B::Tensor>), ConformerError> {
        if B::parallel_size(parallel) != (self.dimensions.heads / self.local.heads) as usize {
            return Err(ConformerError::Configuration(
                "Conformer local projections differ from the tensor communicator",
            ));
        }
        self.forward_selected(input, previous, Some(parallel), context)
    }

    fn forward_selected(
        &mut self,
        input: &B::Tensor,
        previous: &ConformerState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, ConformerState<B::Tensor>), ConformerError> {
        self.validate(input, previous)?;
        self.precision
            .require(self, input)
            .map_err(ConformerError::Precision)?;
        let mut next = self.initialize(input, previous, context)?;
        let d = self.dimensions;
        let batch = input.dim(0);
        let count = input.dim(1);
        let ff = self.norm_feed_forward1.forward(input, context)?;
        let ff = self.ff1_up.forward(&ff, context)?;
        let ff = residual_projection::<B>(
            &mut self.ff1_down,
            &B::silu(ff, context)?,
            parallel,
            context,
        )?;
        let residual = input.add(&ff.multiply_scalar(0.5, context)?, context)?;
        let normalized = self.norm_self_att.forward(&residual, context)?;
        let channel = B::Tensor::concatenate(
            &[
                next.channel.as_ref().expect("initialized").clone(),
                normalized.clone(),
            ],
            1,
            context,
        )?;
        let attention =
            self.attention(&normalized, &channel, next.valid_channel, parallel, context)?;
        next.channel = Some(
            channel
                .index(
                    &[
                        Index::Full,
                        Index::Range(count, count + d.left_context),
                        Index::Full,
                    ],
                    context,
                )?
                .compact(context)?,
        );
        next.valid_channel = (next.valid_channel + count as usize).min(d.left_context as usize);
        let residual = residual.add(&attention, context)?;
        let conv = self.norm_conv.forward(&residual, context)?;
        let conv = B::Tensor::conv1d(
            &self.precision.conv_input(&conv, context)?,
            &self
                .pointwise1
                .as_ref()
                .transpose_axes(&[0, 2, 1], context)?,
            1,
            0,
            1,
            1,
            context,
        )?;
        let conv = self.precision.restore(conv, context)?;
        let gate = B::sigmoid(
            conv.index(
                &[Index::Full, Index::Full, Index::Range(d.width, 2 * d.width)],
                context,
            )?,
            context,
        )?;
        let conv = conv
            .index(
                &[Index::Full, Index::Full, Index::Range(0, d.width)],
                context,
            )?
            .multiply(&gate, context)?;
        let history = B::Tensor::concatenate(
            &[
                next.convolution.as_ref().expect("initialized").clone(),
                conv,
            ],
            1,
            context,
        )?;
        next.convolution = Some(
            history
                .index(
                    &[
                        Index::Full,
                        Index::Range(count, count + d.conv_kernel - 1),
                        Index::Full,
                    ],
                    context,
                )?
                .compact(context)?,
        );
        let conv = B::Tensor::conv1d(
            &self.precision.conv_input(&history, context)?,
            &self
                .depthwise
                .as_ref()
                .transpose_axes(&[0, 2, 1], context)?,
            1,
            0,
            1,
            d.width,
            context,
        )?;
        let conv = self.precision.restore(conv, context)?;
        let conv = B::silu(self.conv_norm.forward(&conv, context)?, context)?;
        let conv = B::Tensor::conv1d(
            &self.precision.conv_input(&conv, context)?,
            &self
                .pointwise2
                .as_ref()
                .transpose_axes(&[0, 2, 1], context)?,
            1,
            0,
            1,
            1,
            context,
        )?;
        let conv = self.precision.restore(conv, context)?;
        let residual = residual.add(&conv, context)?;
        let ff = self.norm_feed_forward2.forward(&residual, context)?;
        let ff = self.ff2_up.forward(&ff, context)?;
        let ff = residual_projection::<B>(
            &mut self.ff2_down,
            &B::silu(ff, context)?,
            parallel,
            context,
        )?;
        let output = self.norm_out.forward(
            &residual.add(&ff.multiply_scalar(0.5, context)?, context)?,
            context,
        )?;
        debug_assert_eq!(output.shape(), &[batch, count, d.width]);
        Ok((output, next))
    }

    fn attention(
        &mut self,
        query: &B::Tensor,
        channel: &B::Tensor,
        valid: usize,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ConformerError> {
        let d = self.dimensions;
        let batch = query.dim(0);
        let count = query.dim(1);
        let total = channel.dim(1);
        let head = d.width / d.heads;
        let heads = self.local.heads;
        let q = self
            .query
            .forward(query, context)?
            .reshape(&[batch, count, heads, head], context)?;
        let q_u = q
            .add(self.pos_bias_u.as_ref(), context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let q_v = q
            .add(self.pos_bias_v.as_ref(), context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let k = self
            .key
            .forward(channel, context)?
            .reshape(&[batch, total, heads, head], context)?
            .transpose_axes(&[0, 2, 3, 1], context)?;
        let v = self
            .value
            .forward(channel, context)?
            .reshape(&[batch, total, heads, head], context)?
            .transpose_axes(&[0, 2, 1, 3], context)?;
        let p = self
            .position
            .forward(total, d.width, query, context)?
            .reshape(&[1, 2 * total - 1, heads, head], context)?
            .transpose_axes(&[0, 2, 3, 1], context)?;
        let relative = B::Tensor::matmul(&q_v, &p, context)?;
        // NVIDIA's relative shift: prepend a zero column, reinterpret rows,
        // discard the first reinterpreted row, then restore query ordering.
        let relative = B::Tensor::pad(
            &relative,
            &[(0, 0), (0, 0), (0, 0), (1, 0)],
            PadMode::Constant,
            context,
        )?
        .reshape(&[batch, heads, 2 * total, count], context)?
        .index(
            &[
                Index::Full,
                Index::Full,
                Index::Range(1, 2 * total),
                Index::Full,
            ],
            context,
        )?
        .reshape(&[batch, heads, count, 2 * total - 1], context)?
        .index(
            &[
                Index::Full,
                Index::Full,
                Index::Full,
                Index::Range(0, total),
            ],
            context,
        )?;
        let scores = B::Tensor::matmul(&q_u, &k, context)?
            .add(&relative, context)?
            .multiply_scalar(1.0 / (head as f32).sqrt(), context)?;
        let mut allowed = Vec::with_capacity(count as usize * total as usize);
        for q in 0..count {
            for k in 0..total {
                allowed.push(i32::from(
                    k >= d.left_context - valid as i32 && k >= q && k <= d.left_context + q,
                ));
            }
        }
        let mask = B::Tensor::from_i32_slice(&allowed, &[1, 1, count, total], context)?
            .equal_i32(0, context)?;
        let floor = float_like::<B::Tensor>(-10000.0, &scores, context)?;
        let masked = B::Tensor::where_condition(&mask, &floor, &scores, context)?;
        let probabilities = masked.softmax_axis(-1, false, context)?;
        let probabilities = B::Tensor::where_condition(
            &mask,
            &probabilities.zeros_like(context)?,
            &probabilities,
            context,
        )?;
        let output = B::Tensor::matmul(&probabilities, &v, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, count, heads * head], context)?;
        Ok(residual_projection::<B>(
            &mut self.attention_output,
            &output,
            parallel,
            context,
        )?)
    }
}

fn residual_projection<B: NeuralBackend>(
    linear: &mut B::Linear,
    input: &B::Tensor,
    parallel: Option<&B::ParallelContext>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    match parallel {
        Some(parallel) => B::row_parallel_linear(linear, input, parallel, context),
        None => linear.forward(input, context),
    }
}

fn float_like<T: Tensor>(value: f32, prototype: &T, context: &T::Context) -> Result<T, Error> {
    let tensor = T::full_f32(value, &[], context)?;
    match prototype.element_type() {
        Some(dtype) if dtype != eredu_nn::TensorElementType::F32 => {
            tensor.cast_float(dtype, context)
        }
        _ => Ok(tensor),
    }
}

