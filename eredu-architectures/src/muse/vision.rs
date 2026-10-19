//! Backend-neutral Muse-Glimmer vision tower and language adapter.

use eredu_nn::{
    multimodal::{
        multi_axis_rotary_embeddings, MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    },
    sequence_layout::{
        attention_chunk_lengths, bilinear_interpolation_samples, inverse_permutation,
        patch_positions, validate_patch_grid, window_partition, InterpolationMode, PatchTraversal,
    },
    EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator, LinearSpec, NeuralBackend,
    Parameter, ParameterSpec, Parameterized, Tensor,
};

use super::{VisionArithmetic, VisionAttentionPolicy, VisionConfig};

/// Read-only boundaries inside one ordinary vision block.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionBlockPhase {
    /// Affine query projection before rotary embedding.
    QueryProjection,
    /// Affine key projection before rotary embedding.
    KeyProjection,
    /// Affine value projection.
    ValueProjection,
    /// Rotary-embedded query heads.
    RotatedQuery,
    /// Rotary-embedded key heads.
    RotatedKey,
    /// Aggregated head channels before the attention output projection.
    AttentionChannels,
    /// Normalized input to attention.
    AttentionInput,
    /// Attention write before residual addition.
    AttentionOutput,
    /// Normalized input to the feed-forward expansion.
    FeedForwardInput,
    /// Affine expansion before activation.
    Expansion,
    /// Activated input to the contraction.
    Activation,
    /// Feed-forward write before residual addition.
    Contraction,
}
impl VisionBlockPhase {
    /// Stable semantic name within the block.
    pub const fn name(self) -> &'static str {
        match self {
            Self::QueryProjection => "query_projection",
            Self::KeyProjection => "key_projection",
            Self::ValueProjection => "value_projection",
            Self::RotatedQuery => "rotated_query",
            Self::RotatedKey => "rotated_key",
            Self::AttentionChannels => "attention_channels",
            Self::AttentionInput => "attention_input",
            Self::AttentionOutput => "attention_output",
            Self::FeedForwardInput => "feed_forward_input",
            Self::Expansion => "expansion",
            Self::Activation => "activation",
            Self::Contraction => "contraction",
        }
    }
}

/// Borrowed ordinary tower observations; callers own retention and copy policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionObservation {
    /// Embedded and permuted patches before the first encoder block.
    Ingress,
    /// Encoder output before the next block or tower readout.
    LayerOutput(usize),
    /// Normalization, merging and language-adapter boundary.
    Readout(VisionReadoutPhase),
    /// One semantic boundary within an encoder block.
    Block {
        /// Owning block index.
        layer: usize,
        /// Boundary within the block.
        phase: VisionBlockPhase,
    },
}

/// Read-only boundaries after the vision encoder.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum VisionReadoutPhase {
    /// Final tower normalization before spatial merging.
    PostNormalization,
    /// Spatially merged tower vectors.
    PixelShuffle,
    /// First adapter projection.
    Expansion,
    /// First adapter activation.
    FirstActivation,
    /// Second adapter projection.
    Contraction,
    /// Second adapter activation.
    SecondActivation,
    /// Projection into decoder width before perception normalization.
    LanguageProjection,
}
impl VisionReadoutPhase {
    /// Stable semantic name within the readout.
    pub const fn name(self) -> &'static str {
        match self {
            Self::PostNormalization => "post_normalization",
            Self::PixelShuffle => "pixel_shuffle",
            Self::Expansion => "adapter_expansion",
            Self::FirstActivation => "adapter_first_activation",
            Self::Contraction => "adapter_contraction",
            Self::SecondActivation => "adapter_second_activation",
            Self::LanguageProjection => "language_projection",
        }
    }
}

/// Prepared flattened RGB patches and their image/video grid metadata.
pub struct VisionInput<'a, T> {
    /// Flattened patches shaped `[patches, temporal * 3 * patch_size^2]`.
    pub pixels: &'a T,
    /// Packed media grids in `(time, height, width)` order.
    pub grid: &'a [(i32, i32, i32)],
}

/// Parameter-free placement values retained between vision blocks.
#[derive(Debug, Clone)]
pub struct VisionState<T> {
    grid: Vec<(i32, i32, i32)>,
    full_chunks: Vec<i32>,
    window_chunks: Vec<i32>,
    window_permutation: Vec<i32>,
    cosine: T,
    sine: T,
}

impl<T> VisionState<T> {
    /// Packed full-attention chunk lengths, one per temporal item.
    pub fn full_chunks(&self) -> &[i32] {
        &self.full_chunks
    }

    /// Window-attention chunk lengths after the stable window permutation.
    pub fn window_chunks(&self) -> &[i32] {
        &self.window_chunks
    }

    /// Stable permutation from raster order to window order.
    pub fn window_permutation(&self) -> &[i32] {
        &self.window_permutation
    }

    /// Returns backend tensors that must remain live while blocks are submitted.
    pub fn retained_values(&self) -> [&T; 2] {
        [&self.cosine, &self.sine]
    }

    fn chunks(&self, policy: VisionAttentionPolicy) -> &[i32] {
        match policy {
            VisionAttentionPolicy::Full => &self.full_chunks,
            VisionAttentionPolicy::Window => &self.window_chunks,
        }
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct CenteredLayerNorm<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    weight: Parameter<B::Tensor>,
    bias: Parameter<B::Tensor>,
    #[parameter(skip)]
    epsilon: f32,
    #[parameter(skip)]
    arithmetic: VisionArithmetic,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> CenteredLayerNorm<B> {
    fn new(
        prefix: &str,
        dimensions: i32,
        epsilon: f32,
        arithmetic: VisionArithmetic,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if arithmetic == VisionArithmetic::Rounded {
            B::require_operator_capabilities(
                "rounded vision normalization",
                eredu_nn::NeuralOperatorCapabilities::CASCADED_LAYER_NORM,
            )?;
        }
        let parameter = |suffix: &str| {
            Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.{suffix}")).map_err(Error::backend)?,
                &[dimensions],
                context,
            )
        };
        Ok(Self {
            weight: parameter("weight")?,
            bias: parameter("bias")?,
            epsilon,
            arithmetic,
        })
    }

    fn forward(
        &self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        B::Tensor::layer_norm(
            input,
            Some(self.weight.as_ref()),
            Some(self.bias.as_ref()),
            self.epsilon,
            match self.arithmetic {
                VisionArithmetic::Native => eredu_nn::LayerNormalizationArithmetic::Native,
                VisionArithmetic::Rounded => {
                    eredu_nn::LayerNormalizationArithmetic::CascadedMoments
                }
            },
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct PatchEmbedder<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    projection: B::Linear,
    position_table: B::Embedding,
    #[parameter(skip)]
    input_width: i32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> PatchEmbedder<B> {
    fn new(config: &VisionConfig, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let input_width = config.temporal_patch_size * 3 * config.patch_size * config.patch_size;
        let weight = "model.vision_tower.patch_embedder.patch_embedding.weight";
        let position = "model.vision_tower.patch_embedder.position_embedding_table.weight";
        Ok(Self {
            projection: B::linear(
                LinearSpec {
                    input: input_width,
                    output: config.hidden_size,
                    weight: ParameterSpec::trainable(weight).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        weight,
                        config.linear_format_for(weight),
                    )?,
                },
                context,
            )?,
            position_table: B::embedding(
                EmbeddingSpec {
                    vocabulary: config.position_height * config.position_width,
                    dimensions: config.hidden_size,
                    weight: ParameterSpec::trainable(position).map_err(Error::backend)?,
                    format: crate::linear_format::standard_linear_format(
                        position,
                        config.linear_format_for(position),
                    )?,
                },
                context,
            )?,
            input_width,
        })
    }

    fn forward(
        &mut self,
        pixels: &B::Tensor,
        grid: &[(i32, i32, i32)],
        config: &VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if pixels.shape().len() != 2 || pixels.dim(1) != self.input_width {
            return Err(Error::backend(format!(
                "Muse-Glimmer vision pixels must be shaped [patches, {}], got {:?}",
                self.input_width,
                pixels.shape()
            )));
        }
        let pixels =
            config
                .arithmetic
                .input_precision()
                .prepare(pixels, &self.projection, context)?;
        let hidden = self.projection.forward(&pixels, context)?;
        let transposed;
        let position_grid = match config.position_layout {
            super::config::VisionPositionLayout::Raster => grid,
            super::config::VisionPositionLayout::TransposedRaster => {
                transposed = grid
                    .iter()
                    .map(|&(time, height, width)| (time, width, height))
                    .collect::<Vec<_>>();
                &transposed
            }
        };
        let positions = interpolated_positions::<B>(
            &mut self.position_table,
            position_grid,
            config.position_height,
            config.position_width,
            context,
        )?;
        let positions = match config.arithmetic {
            VisionArithmetic::Native => positions,
            VisionArithmetic::Rounded => positions.cast_float(
                hidden
                    .element_type()
                    .ok_or_else(|| Error::backend("Muse vision patch precision is unavailable"))?,
                context,
            )?,
        };
        hidden.add(&positions, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct VisionAttention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    query: B::Linear,
    key: B::Linear,
    value: B::Linear,
    output: B::Linear,
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    head_dimensions: i32,
    #[parameter(skip)]
    scale: f32,
    #[parameter(skip)]
    arithmetic: VisionArithmetic,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionAttention<B> {
    fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.vision_tower.layers.{layer}.attn");
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: Some(
                        ParameterSpec::trainable(format!("{prefix}.{field}.bias"))
                            .map_err(Error::backend)?,
                    ),
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        config.linear_format_for(&weight),
                    )?
                    .with_arithmetic(match config.arithmetic {
                        VisionArithmetic::Native => eredu_nn::LinearArithmetic::Native,
                        VisionArithmetic::Rounded => eredu_nn::LinearArithmetic::RowMajorAffine,
                    })?,
                },
                context,
            )
        };
        let head_dimensions = config.hidden_size / config.num_heads;
        Ok(Self {
            query: linear("q_proj", config.hidden_size, config.hidden_size)?,
            key: linear("k_proj", config.hidden_size, config.hidden_size)?,
            value: linear("v_proj", config.hidden_size, config.hidden_size)?,
            output: linear("proj", config.hidden_size, config.hidden_size)?,
            heads: config.num_heads,
            head_dimensions,
            scale: match config.arithmetic {
                VisionArithmetic::Native => (head_dimensions as f32).sqrt().recip(),
                VisionArithmetic::Rounded => (head_dimensions as f64).powf(-0.5) as f32,
            },
            arithmetic: config.arithmetic,
        })
    }

    fn forward<F>(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cosine: &B::Tensor,
        sine: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnMut(
            VisionBlockPhase,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), Error>,
    {
        let sequence = hidden.dim(0);
        let query = self.query.forward(hidden, context)?;
        hook(VisionBlockPhase::QueryProjection, &query, context)?;
        let query = query.reshape(&[sequence, self.heads, self.head_dimensions], context)?;
        let key = self.key.forward(hidden, context)?;
        hook(VisionBlockPhase::KeyProjection, &key, context)?;
        let key = key.reshape(&[sequence, self.heads, self.head_dimensions], context)?;
        let value = self.value.forward(hidden, context)?;
        hook(VisionBlockPhase::ValueProjection, &value, context)?;
        let value = value.reshape(&[sequence, self.heads, self.head_dimensions], context)?;
        let query = apply_rotary(&query, cosine, sine, self.arithmetic, context)?;
        let key = apply_rotary(&key, cosine, sine, self.arithmetic, context)?;
        hook(VisionBlockPhase::RotatedQuery, &query, context)?;
        hook(VisionBlockPhase::RotatedKey, &key, context)?;
        let mut outputs = Vec::with_capacity(chunks.len());
        let mut start = 0;
        for &length in chunks {
            if length <= 0 || start + length > sequence {
                return Err(Error::backend(
                    "invalid Muse-Glimmer vision attention chunk",
                ));
            }
            let end = start + length;
            let prepare = |value: &B::Tensor| {
                value
                    .index(
                        &[Index::Range(start, end), Index::Full, Index::Full],
                        context,
                    )?
                    .transpose_axes(&[1, 0, 2], context)?
                    .expand_dims(0, context)
            };
            let output = B::attention_with_sinks(
                eredu_nn::AttentionRequest {
                    arithmetic: self.arithmetic.attention(),
                    queries: prepare(&query)?,
                    keys: prepare(&key)?,
                    values: prepare(&value)?,
                    scale: self.scale,
                    mask: None,
                    sinks: None,
                    softcap: None,
                },
                context,
            )?
            .squeeze_axes(&[0], context)?
            .transpose_axes(&[1, 0, 2], context)?
            .reshape(&[length, self.heads * self.head_dimensions], context)?;
            outputs.push(output);
            start = end;
        }
        if start != sequence {
            return Err(Error::backend(format!(
                "Muse-Glimmer vision chunks cover {start} tokens, expected {sequence}"
            )));
        }
        let channels = B::Tensor::concatenate(&outputs, 0, context)?;
        hook(VisionBlockPhase::AttentionChannels, &channels, context)?;
        self.output.forward(&channels, context)
    }
}

/// One exact Muse-Glimmer vision transformer block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionBlock<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    norm1: CenteredLayerNorm<B>,
    attention: VisionAttention<B>,
    norm2: CenteredLayerNorm<B>,
    fc1: B::Linear,
    fc2: B::Linear,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionBlock<B> {
    /// Builds one unloaded vision block.
    pub fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.vision_tower.layers.{layer}");
        let linear = |field: &str, input, output| {
            let weight = format!("{prefix}.mlp.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
                    bias: Some(
                        ParameterSpec::trainable(format!("{prefix}.mlp.{field}.bias"))
                            .map_err(Error::backend)?,
                    ),
                    format: crate::linear_format::standard_linear_format(
                        &weight,
                        config.linear_format_for(&weight),
                    )?
                    .with_arithmetic(match config.arithmetic {
                        VisionArithmetic::Native => eredu_nn::LinearArithmetic::Native,
                        VisionArithmetic::Rounded => eredu_nn::LinearArithmetic::RowMajorAffine,
                    })?,
                },
                context,
            )
        };
        Ok(Self {
            norm1: CenteredLayerNorm::new(
                &format!("{prefix}.norm1"),
                config.hidden_size,
                config.layer_norm_eps,
                config.arithmetic,
                context,
            )?,
            attention: VisionAttention::new(config, layer, context)?,
            norm2: CenteredLayerNorm::new(
                &format!("{prefix}.norm2"),
                config.hidden_size,
                config.layer_norm_eps,
                config.arithmetic,
                context,
            )?,
            fc1: linear("fc1", config.hidden_size, config.intermediate_size)?,
            fc2: linear("fc2", config.intermediate_size, config.hidden_size)?,
        })
    }

    /// Applies this block using architecture-selected attention chunks.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cosine: &B::Tensor,
        sine: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_observed(hidden, chunks, cosine, sine, context, |_, _, _| Ok(()))
    }

    fn forward_observed<F>(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cosine: &B::Tensor,
        sine: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        mut hook: F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnMut(
            VisionBlockPhase,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), Error>,
    {
        let normalized = self.norm1.forward(hidden, context)?;
        hook(VisionBlockPhase::AttentionInput, &normalized, context)?;
        let attention =
            self.attention
                .forward(&normalized, chunks, cosine, sine, context, &mut hook)?;
        hook(VisionBlockPhase::AttentionOutput, &attention, context)?;
        let hidden = hidden.add(&attention, context)?;
        let normalized = self.norm2.forward(&hidden, context)?;
        hook(VisionBlockPhase::FeedForwardInput, &normalized, context)?;
        let expansion = self.fc1.forward(&normalized, context)?;
        hook(VisionBlockPhase::Expansion, &expansion, context)?;
        let feed_forward = B::Tensor::gelu(&expansion, context)?;
        hook(VisionBlockPhase::Activation, &feed_forward, context)?;
        let write = self.fc2.forward(&feed_forward, context)?;
        hook(VisionBlockPhase::Contraction, &write, context)?;
        hidden.add(&write, context)
    }

    /// Applies this block through its normalized full/window schedule entry.
    pub fn forward_scheduled(
        &mut self,
        hidden: &B::Tensor,
        policy: VisionAttentionPolicy,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward(
            hidden,
            state.chunks(policy),
            &state.cosine,
            &state.sine,
            context,
        )
    }
}

/// Pinned patch, position, normalization, merge-adapter, and projection modules.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    patch_embedder: PatchEmbedder<B>,
    pre_norm: CenteredLayerNorm<B>,
    post_norm: CenteredLayerNorm<B>,
    adapter_fc1: B::Linear,
    adapter_fc2: B::Linear,
    projection: B::Linear,
    #[parameter(skip)]
    config: VisionConfig,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionStatic<B> {
    /// Builds unloaded pinned vision modules.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if matches!(
            config.perception_normalization,
            super::PerceptionNormalization::Float32 { .. }
        ) {
            B::require_operator_capabilities(
                "ordered perception RMS normalization",
                eredu_nn::NeuralOperatorCapabilities::CASCADED_RMS
                    .union(eredu_nn::NeuralOperatorCapabilities::CAST_FLOAT),
            )?;
        }
        let linear = |name: &str, input, output| {
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(name).map_err(Error::backend)?,
                    bias: None,
                    format: crate::linear_format::standard_linear_format(
                        name,
                        config.linear_format_for(name),
                    )?,
                },
                context,
            )
        };
        let shuffled = config.hidden_size * config.merge_size * config.merge_size;
        Ok(Self {
            patch_embedder: PatchEmbedder::new(&config, context)?,
            pre_norm: CenteredLayerNorm::new(
                "model.vision_tower.ln_pre",
                config.hidden_size,
                config.layer_norm_eps,
                config.arithmetic,
                context,
            )?,
            post_norm: CenteredLayerNorm::new(
                "model.vision_tower.ln_post",
                config.hidden_size,
                config.layer_norm_eps,
                config.arithmetic,
                context,
            )?,
            adapter_fc1: linear(
                "model.vision_adapter.fc1.weight",
                shuffled,
                config.projector_hidden_size,
            )?,
            adapter_fc2: linear(
                "model.vision_adapter.fc2.weight",
                config.projector_hidden_size,
                config.projector_hidden_size,
            )?,
            projection: linear(
                "model.vision_projection.weight",
                config.projector_hidden_size,
                config.language_hidden_size,
            )?,
            config,
        })
    }

    /// Embeds and window-permutes patches and constructs reusable placement state.
    pub fn begin(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        validate_patch_grid(
            input.grid,
            self.config.merge_size,
            Some(input.pixels.dim(0)),
        )
        .map_err(Error::backend)?;
        let hidden =
            self.patch_embedder
                .forward(input.pixels, input.grid, &self.config, context)?;
        let hidden = self.pre_norm.forward(&hidden, context)?;
        let full_chunks = attention_chunk_lengths(input.grid).map_err(Error::backend)?;
        let window = window_partition(
            input.grid,
            1,
            self.config.position_height * self.config.patch_size,
            self.config.patch_size,
        )
        .map_err(Error::backend)?;
        let permutation = B::Tensor::from_i32_slice(
            &window.permutation,
            &[window.permutation.len() as i32],
            context,
        )?;
        let hidden = hidden.take_axis(&permutation, 0, context)?;
        let (cosine, sine) = rotary_embeddings::<B::Tensor>(input.grid, &self.config, context)?;
        let (cosine, sine) = match self.config.arithmetic {
            VisionArithmetic::Native => (cosine, sine),
            VisionArithmetic::Rounded => {
                let element = hidden.element_type().ok_or_else(|| {
                    Error::backend("Muse vision rotary coefficient precision is unavailable")
                })?;
                (
                    cosine.cast_float(element, context)?,
                    sine.cast_float(element, context)?,
                )
            }
        };
        Ok((
            hidden,
            VisionState {
                grid: input.grid.to_vec(),
                full_chunks,
                window_chunks: window.chunk_lengths,
                window_permutation: window.permutation,
                cosine: cosine.take_axis(&permutation, 0, context)?,
                sine: sine.take_axis(&permutation, 0, context)?,
            },
        ))
    }

    /// Restores raster order, pixel-shuffles, adapts, projects, and normalizes.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_observed(hidden, state, context, &mut |_, _, _| Ok(()))
    }

    fn finish_observed<F>(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnMut(
            VisionObservation,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), Error>,
    {
        let reverse = inverse_permutation(&state.window_permutation).map_err(Error::backend)?;
        let reverse = B::Tensor::from_i32_slice(&reverse, &[reverse.len() as i32], context)?;
        let hidden = hidden.take_axis(&reverse, 0, context)?;
        let hidden = self.post_norm.forward(&hidden, context)?;
        hook(
            VisionObservation::Readout(VisionReadoutPhase::PostNormalization),
            &hidden,
            context,
        )?;
        let hidden = pixel_shuffle(&hidden, &state.grid, self.config.merge_size, context)?;
        hook(
            VisionObservation::Readout(VisionReadoutPhase::PixelShuffle),
            &hidden,
            context,
        )?;
        self.adapt_observed(&hidden, context, hook)
    }

    /// Adapts pixel-shuffled tower vectors through the shared media projection.
    pub fn adapt(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.adapt_observed(hidden, context, &mut |_, _, _| Ok(()))
    }

    fn adapt_observed<F>(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        hook: &mut F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnMut(
            VisionObservation,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), Error>,
    {
        if hidden.shape().len() != 2
            || hidden.dim(0) <= 0
            || hidden.dim(1)
                != self.config.hidden_size * self.config.merge_size * self.config.merge_size
        {
            return Err(Error::backend(
                "Muse media adapter input has incompatible geometry",
            ));
        }
        let hidden = self.adapter_fc1.forward(hidden, context)?;
        hook(
            VisionObservation::Readout(VisionReadoutPhase::Expansion),
            &hidden,
            context,
        )?;
        let hidden = B::Tensor::gelu(&hidden, context)?;
        hook(
            VisionObservation::Readout(VisionReadoutPhase::FirstActivation),
            &hidden,
            context,
        )?;
        let hidden = self.adapter_fc2.forward(&hidden, context)?;
        hook(
            VisionObservation::Readout(VisionReadoutPhase::Contraction),
            &hidden,
            context,
        )?;
        let hidden = B::Tensor::gelu(&hidden, context)?;
        hook(
            VisionObservation::Readout(VisionReadoutPhase::SecondActivation),
            &hidden,
            context,
        )?;
        let hidden = self.projection.forward(&hidden, context)?;
        hook(
            VisionObservation::Readout(VisionReadoutPhase::LanguageProjection),
            &hidden,
            context,
        )?;
        match self.config.perception_normalization {
            super::PerceptionNormalization::Backend { epsilon } => {
                B::rms_norm_without_weight(&hidden, epsilon, context)
            }
            super::PerceptionNormalization::Float32 { epsilon } => {
                super::equations::weightless_rms_norm::<B>(&hidden, epsilon, context)
            }
        }
    }
}

/// Resident neutral vision tower; streamed runtimes reuse its static phases and block units.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionTower<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Pinned media modules.
    pub static_modules: VisionStatic<B>,
    /// Independently streamable encoder blocks.
    pub blocks: Vec<VisionBlock<B>>,
    #[parameter(skip)]
    schedule: Vec<VisionAttentionPolicy>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionTower<B> {
    /// Builds the complete unloaded native vision tower and language adapter.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let blocks = (0..config.layer_count())
            .map(|layer| VisionBlock::new(&config, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
        let schedule = config.schedule.clone();
        Ok(Self {
            static_modules: VisionStatic::new(config, context)?,
            blocks,
            schedule,
        })
    }

    /// Runs the complete resident tower through the same reusable phases.
    pub fn forward(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_observed(input, context, |_, _, _| Ok(()))
    }

    /// Runs the ordinary tower while observing ingress and block boundaries.
    /// The hook borrows live values;
    /// callers own any retention or host-copy policy.
    pub fn forward_observed<F>(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        mut hook: F,
    ) -> Result<B::Tensor, Error>
    where
        F: FnMut(
            VisionObservation,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), Error>,
    {
        let (mut hidden, state) = self.static_modules.begin(input, context)?;
        hook(VisionObservation::Ingress, &hidden, context)?;
        for layer in 0..self.blocks.len() {
            hidden = self.blocks[layer].forward_observed(
                &hidden,
                state.chunks(self.schedule[layer]),
                &state.cosine,
                &state.sine,
                context,
                |phase, value, context| {
                    hook(VisionObservation::Block { layer, phase }, value, context)
                },
            )?;
            hook(VisionObservation::LayerOutput(layer), &hidden, context)?;
        }
        self.static_modules
            .finish_observed(&hidden, &state, context, &mut hook)
    }
}

fn interpolated_positions<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    embedding: &mut B::Embedding,
    grid: &[(i32, i32, i32)],
    height: i32,
    width: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let samples = bilinear_interpolation_samples(
        grid,
        height,
        width,
        InterpolationMode::HalfPixel,
        PatchTraversal::Raster,
    )
    .map_err(Error::backend)?;
    let length = samples.len() as i32;
    let mut output: Option<B::Tensor> = None;
    for corner in 0..4 {
        let indices = samples
            .iter()
            .map(|sample| sample.indices[corner] as i32)
            .collect::<Vec<_>>();
        let weights = samples
            .iter()
            .map(|sample| sample.weights[corner])
            .collect::<Vec<_>>();
        let indices = B::Tensor::from_i32_slice(&indices, &[length], context)?;
        let weights = B::Tensor::from_f32_slice(&weights, &[length, 1], context)?;
        let value = embedding
            .forward(&indices, context)?
            .multiply(&weights, context)?;
        output = Some(match output {
            Some(current) => current.add(&value, context)?,
            None => value,
        });
    }
    output.ok_or_else(|| Error::backend("Muse-Glimmer vision grid is empty"))
}

fn rotary_embeddings<T: Tensor>(
    grid: &[(i32, i32, i32)],
    config: &VisionConfig,
    context: &T::Context,
) -> Result<(T, T), Error> {
    let positions = patch_positions(grid, PatchTraversal::Raster).map_err(Error::backend)?;
    let positions = positions
        .into_iter()
        .flat_map(|[_, y, x]| [x, y])
        .collect::<Vec<_>>();
    let position_ids = T::from_i32_slice(&positions, &[positions.len() as i32 / 2, 2], context)?;
    let axis_dimensions = (config.hidden_size / config.num_heads) / 2;
    multi_axis_rotary_embeddings(
        &position_ids,
        &MultiAxisRotarySpec {
            axes: vec![
                RotaryAxisSpec {
                    dimensions: axis_dimensions,
                    position_offset: 1,
                },
                RotaryAxisSpec {
                    dimensions: axis_dimensions,
                    position_offset: 1,
                },
            ],
            base: config.rope_theta,
            minimum_position: 0,
            layout: MultiAxisRotaryLayout::SplitHalves,
        },
        context,
    )
}

fn apply_rotary<T: Tensor>(
    value: &T,
    cosine: &T,
    sine: &T,
    arithmetic: VisionArithmetic,
    context: &T::Context,
) -> Result<T, Error> {
    let element = value.element_type();
    let wide;
    let value = match arithmetic {
        VisionArithmetic::Native => value,
        VisionArithmetic::Rounded => {
            wide = value.cast_float(eredu_nn::TensorElementType::F32, context)?;
            &wide
        }
    };
    let half = value.dim(2) / 2;
    let first = value.index(&[Index::Full, Index::Full, Index::Range(0, half)], context)?;
    let second = value.index(
        &[Index::Full, Index::Full, Index::Range(half, value.dim(2))],
        context,
    )?;
    let rotated = T::concatenate(&[second.multiply_scalar(-1.0, context)?, first], 2, context)?;
    let cosine = cosine.expand_dims(1, context)?;
    let sine = sine.expand_dims(1, context)?;
    let result = value
        .multiply(&cosine, context)?
        .add(&rotated.multiply(&sine, context)?, context)?;
    match arithmetic {
        VisionArithmetic::Native => Ok(result),
        VisionArithmetic::Rounded => result.cast_float(
            element.ok_or_else(|| {
                Error::backend("Muse vision rotary input precision is unavailable")
            })?,
            context,
        ),
    }
}

fn pixel_shuffle<T: Tensor>(
    hidden: &T,
    grid: &[(i32, i32, i32)],
    factor: i32,
    context: &T::Context,
) -> Result<T, Error> {
    let dimensions = hidden.dim(1);
    let mut offset = 0;
    let mut outputs = Vec::with_capacity(grid.len());
    for &(time, height, width) in grid {
        let count = time * height * width;
        let chunk = hidden.index(
            &[Index::Range(offset, offset + count), Index::Full],
            context,
        )?;
        let mut permutation = Vec::with_capacity(count as usize);
        for frame in 0..time {
            for outer_y in 0..height / factor {
                for outer_x in 0..width / factor {
                    for inner_y in 0..factor {
                        for inner_x in 0..factor {
                            permutation.push(
                                frame * height * width
                                    + (outer_y * factor + inner_y) * width
                                    + outer_x * factor
                                    + inner_x,
                            );
                        }
                    }
                }
            }
        }
        let permutation = T::from_i32_slice(&permutation, &[count], context)?;
        let output_count = time * (height / factor) * (width / factor);
        outputs.push(
            chunk
                .take_axis(&permutation, 0, context)?
                .reshape(&[output_count, factor * factor, dimensions], context)?
                .transpose_axes(&[0, 2, 1], context)?
                .reshape(&[output_count, dimensions * factor * factor], context)?,
        );
        offset += count;
    }
    if offset != hidden.dim(0) {
        return Err(Error::backend(
            "Muse-Glimmer pixel shuffle grid does not cover hidden states",
        ));
    }
    T::concatenate(&outputs, 0, context)
}
