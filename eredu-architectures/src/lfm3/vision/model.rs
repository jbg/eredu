//! Hybrid vision equations over ordinary neutral neural mechanisms.

use eredu_nn::{
    multimodal::multi_axis_rotary_embeddings,
    sequence_layout::{patch_positions, validate_patch_grid, PatchTraversal},
    AttentionArithmetic, AttentionRequest, Error, Index, LayerNorm, LinearOperator, NeuralBackend,
    NormalizationOperator, Parameter, Parameterized, Tensor, TensorElementType,
};

use super::{
    declarations::{AttentionSpec, BlockSpec, ConvolutionSpec, MixerSpec},
    Declaration, VisionConfig,
};

/// Prepared flattened patches with their ordered temporal/height/width grids.
pub struct VisionInput<'a, T> {
    /// `[patches, channels * temporal_patch_size * patch_size²]` values.
    pub pixels: &'a T,
    /// Temporal patch groups and spatial dimensions, in merge-major order.
    pub grid: &'a [(i32, i32, i32)],
}

/// Parameter-free continuation between separately leased vision units.
#[derive(Debug, Clone)]
pub struct VisionState<T> {
    config: VisionConfig,
    patches: i32,
    frame_shapes: Vec<(i32, i32)>,
    cosine: T,
    sine: T,
}

impl<T: Tensor> VisionState<T> {
    fn prepare(
        config: &VisionConfig,
        input: &VisionInput<'_, T>,
        context: &T::Context,
    ) -> Result<Self, Error> {
        config.validate()?;
        let patches = validate_patch_grid(input.grid, config.spatial_merge_size, None)
            .map_err(Error::backend_source)?;
        if input.pixels.shape() != [patches, config.patch_width()?] {
            return Err(Error::backend(
                "LFM3 vision pixels do not match the declared patch grid",
            ));
        }
        let positions = patch_positions(
            input.grid,
            PatchTraversal::MergeMajor(config.spatial_merge_size),
        )
        .map_err(Error::backend_source)?;
        let coordinates = positions
            .iter()
            .flat_map(|p| [p[1], p[2]])
            .collect::<Vec<_>>();
        let positions = T::from_i32_slice(&coordinates, &[patches, 2], context)?;
        let (cosine, sine) =
            multi_axis_rotary_embeddings(&positions, &config.rotary_spec(), context)?;
        Ok(Self {
            config: config.clone(),
            patches,
            frame_shapes: input
                .grid
                .iter()
                .flat_map(|&(time, height, width)| {
                    std::iter::repeat_n((height, width), time as usize)
                })
                .collect(),
            cosine,
            sine,
        })
    }

    /// Spatial shape of each independently mixed temporal patch group.
    pub fn frame_shapes(&self) -> &[(i32, i32)] {
        &self.frame_shapes
    }

    /// Rotary roots retained across unit boundaries, in cosine/sine order.
    pub fn retained_values(&self) -> [&T; 2] {
        [&self.cosine, &self.sine]
    }

    /// Replaces exactly the transported cosine and sine roots after checking geometry.
    pub fn replace_retained_values(&mut self, values: [T; 2]) -> Result<(), Error> {
        let head = self.config.hidden_size / self.config.num_attention_heads;
        if values.iter().any(|value| {
            value.shape() != [self.patches, head]
                || value.element_type() != Some(TensorElementType::F32)
        }) {
            return Err(Error::backend(
                "LFM3 vision rotary transport has incompatible geometry or dtype",
            ));
        }
        [self.cosine, self.sine] = values;
        Ok(())
    }

    fn validate_hidden(&self, hidden: &T, config: &VisionConfig) -> Result<(), Error> {
        if config != &self.config || hidden.shape() != [self.patches, config.hidden_size] {
            return Err(Error::backend(
                "LFM3 vision continuation does not match its declared geometry",
            ));
        }
        Ok(())
    }
}

/// Linear flattened-patch embedding and initial learned RMSNorm.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionPatchifier<B: NeuralBackend> {
    projection: B::Linear,
    norm: B::Normalization,
    #[parameter(skip)]
    config: VisionConfig,
}

impl<B: NeuralBackend> VisionPatchifier<B> {
    /// Constructs the unloaded input unit using retained source identities.
    pub fn new(
        declaration: &Declaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            projection: B::linear(declaration.patch.clone(), context)?,
            norm: B::normalization(declaration.embedding_norm.clone(), context)?,
            config: declaration.config().clone(),
        })
    }

    /// Creates hidden patches and the exact frame-local positional continuation.
    pub fn forward(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        let state = VisionState::prepare(&self.config, &input, context)?;
        let pixels = match self.projection.parameter_element_type() {
            Some(dtype) if input.pixels.element_type() != Some(dtype) => {
                input.pixels.cast_float(dtype, context)?
            }
            _ => input.pixels.clone(),
        };
        let hidden = self.projection.forward(&pixels, context)?;
        Ok((self.norm.forward(&hidden, context)?, state))
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct Attention<B: NeuralBackend> {
    query: B::Linear,
    key: B::Linear,
    value: B::Linear,
    output: B::Linear,
    query_norm: B::Normalization,
    key_norm: B::Normalization,
    #[parameter(skip)]
    query_heads: i32,
    #[parameter(skip)]
    kv_heads: i32,
    #[parameter(skip)]
    head_dim: i32,
}

impl<B: NeuralBackend> Attention<B> {
    fn new(spec: AttentionSpec, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        let geometry = spec.projections.geometry;
        Ok(Self {
            query_heads: geometry.query_heads(),
            kv_heads: geometry.key_value_heads(),
            head_dim: geometry.key_dimensions(),
            query: B::linear(spec.projections.query, context)?,
            key: B::linear(spec.projections.key, context)?,
            value: B::linear(spec.projections.value, context)?,
            output: B::linear(spec.projections.output, context)?,
            query_norm: B::normalization(spec.query_norm, context)?,
            key_norm: B::normalization(spec.key_norm, context)?,
        })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let sequence = hidden.dim(0);
        let query = self
            .query
            .forward(hidden, context)?
            .reshape(&[sequence, self.query_heads, self.head_dim], context)?;
        let key = self
            .key
            .forward(hidden, context)?
            .reshape(&[sequence, self.kv_heads, self.head_dim], context)?;
        let value = self
            .value
            .forward(hidden, context)?
            .reshape(&[sequence, self.kv_heads, self.head_dim], context)?;
        let query = rotary(
            &self.query_norm.forward(&query, context)?,
            &state.cosine,
            &state.sine,
            context,
        )?;
        let key = rotary(
            &self.key_norm.forward(&key, context)?,
            &state.cosine,
            &state.sine,
            context,
        )?;
        // Eager BF16 attention rounds the scores and softmax probabilities
        // before the value projection.
        let arithmetic = if query.element_type() == Some(TensorElementType::Bf16) {
            AttentionArithmetic::InputScores
        } else {
            AttentionArithmetic::Fused
        };
        let mut start = 0;
        let mut outputs = Vec::with_capacity(state.frame_shapes.len());
        for &(height, width) in &state.frame_shapes {
            let end = start + height * width;
            let slice = |value: &B::Tensor| {
                value
                    .index(
                        &[Index::Range(start, end), Index::Full, Index::Full],
                        context,
                    )?
                    .transpose_axes(&[1, 0, 2], context)?
                    .expand_dims(0, context)
            };
            // Every frame has its complete key set. There is no causal text
            // mask, sliding-window retention or cross-frame KV state here.
            let output = B::attention_with_sinks(
                AttentionRequest {
                    arithmetic,
                    queries: slice(&query)?,
                    keys: slice(&key)?,
                    values: slice(&value)?,
                    scale: (self.head_dim as f32).powf(-0.5),
                    softcap: None,
                    mask: None,
                    sinks: None,
                },
                context,
            )?;
            outputs.push(
                output
                    .squeeze_axes(&[0], context)?
                    .transpose_axes(&[1, 0, 2], context)?,
            );
            start = end;
        }
        let attended = B::Tensor::concatenate(&outputs, 0, context)?
            .reshape(&[sequence, self.query_heads * self.head_dim], context)?;
        self.output.forward(&attended, context)
    }
}

fn rotary<T: Tensor>(value: &T, cosine: &T, sine: &T, context: &T::Context) -> Result<T, Error> {
    let dtype = value
        .element_type()
        .ok_or_else(|| Error::backend("vision rotary requires an explicit floating dtype"))?;
    let value = value.cast_float(TensorElementType::F32, context)?;
    let half = value.dim(2) / 2;
    let first = value.index(&[Index::Full, Index::Full, Index::Range(0, half)], context)?;
    let second = value.index(
        &[Index::Full, Index::Full, Index::Range(half, value.dim(2))],
        context,
    )?;
    let rotated = T::concatenate(&[second.multiply_scalar(-1.0, context)?, first], 2, context)?;
    value
        .multiply(&cosine.expand_dims(1, context)?, context)?
        .add(
            &rotated.multiply(&sine.expand_dims(1, context)?, context)?,
            context,
        )?
        .cast_float(dtype, context)
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
struct Convolution<B: NeuralBackend> {
    input: B::Linear,
    output: B::Linear,
    weight: Parameter<B::Tensor>,
    #[parameter(skip)]
    channels: i32,
    #[parameter(skip)]
    kernel: i32,
}

impl<B: NeuralBackend> Convolution<B> {
    fn new(spec: ConvolutionSpec, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        Ok(Self {
            input: B::linear(spec.input, context)?,
            output: B::linear(spec.output, context)?,
            weight: Parameter::unloaded(
                spec.weight,
                &[spec.channels, 1, spec.kernel, spec.kernel],
                context,
            )?,
            channels: spec.channels,
            kernel: spec.kernel,
        })
    }

    fn forward(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let fused = self.input.forward(hidden, context)?;
        let part = |i| {
            fused.index(
                &[
                    Index::Full,
                    Index::Range(i * self.channels, (i + 1) * self.channels),
                ],
                context,
            )
        };
        let gated = part(0)?.multiply(&part(2)?, context)?;
        // Official source kernels are OIHW. The neutral convolution mechanism
        // consumes OHWI; this is a per-invocation view, not a retained replica.
        let weight = self
            .weight
            .as_ref()
            .transpose_axes(&[0, 2, 3, 1], context)?;
        // The released BF16 mixer accumulates spatial products in F32 and
        // rounds once before the output gate.
        let widen = gated.element_type() == Some(TensorElementType::Bf16);
        let weight = if widen {
            weight.cast_float(TensorElementType::F32, context)?
        } else {
            weight
        };
        let merge = state.config.spatial_merge_size;
        let mut start = 0;
        let mut outputs = Vec::with_capacity(state.frame_shapes.len());
        for &(height, width) in &state.frame_shapes {
            let end = start + height * width;
            let raster = gated
                .index(&[Index::Range(start, end), Index::Full], context)?
                .reshape(
                    &[height / merge, width / merge, merge, merge, self.channels],
                    context,
                )?
                .transpose_axes(&[0, 2, 1, 3, 4], context)?
                .reshape(&[1, height, width, self.channels], context)?;
            let raster = if widen {
                raster.cast_float(TensorElementType::F32, context)?
            } else {
                raster
            };
            let output = B::Tensor::conv2d(
                &raster,
                &weight,
                (1, 1),
                (self.kernel / 2, self.kernel / 2),
                (1, 1),
                self.channels,
                context,
            )?;
            let output = if widen {
                output.cast_float(TensorElementType::Bf16, context)?
            } else {
                output
            };
            outputs.push(
                output
                    .reshape(
                        &[height / merge, merge, width / merge, merge, self.channels],
                        context,
                    )?
                    .transpose_axes(&[0, 2, 1, 3, 4], context)?
                    .reshape(&[height * width, self.channels], context)?,
            );
            start = end;
        }
        let convolved = B::Tensor::concatenate(&outputs, 0, context)?;
        self.output
            .forward(&part(1)?.multiply(&convolved, context)?, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
enum Mixer<B: NeuralBackend> {
    Attention(Attention<B>),
    Convolution(Convolution<B>),
}

/// One independently constructible hybrid vision residual block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionBlock<B: NeuralBackend> {
    #[parameter(skip)]
    layer: usize,
    mixer: Mixer<B>,
    operator_norm: B::Normalization,
    ffn_norm: B::Normalization,
    gate: B::Linear,
    up: B::Linear,
    down: B::Linear,
    #[parameter(skip)]
    config: VisionConfig,
    #[parameter(skip)]
    gated_product: eredu_nn::GatedProductPolicy,
}

impl<B: NeuralBackend> VisionBlock<B> {
    /// Constructs one unloaded encoder layer from retained declaration slots.
    pub fn new(
        declaration: &Declaration,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = declaration
            .blocks
            .get(layer)
            .ok_or_else(|| Error::backend("LFM3 vision block is outside its declaration"))?
            .clone();
        let BlockSpec {
            mixer,
            operator_norm,
            ffn_norm,
            feed_forward,
        } = spec;
        Ok(Self {
            layer,
            mixer: match mixer {
                MixerSpec::Attention(s) => Mixer::Attention(Attention::new(s, context)?),
                MixerSpec::Convolution(s) => Mixer::Convolution(Convolution::new(s, context)?),
            },
            operator_norm: B::normalization(operator_norm, context)?,
            ffn_norm: B::normalization(ffn_norm, context)?,
            gate: B::linear(feed_forward.gate, context)?,
            up: B::linear(feed_forward.up, context)?,
            down: B::linear(feed_forward.down, context)?,
            gated_product: feed_forward.policy,
            config: declaration.config().clone(),
        })
    }

    pub(super) fn layer(&self) -> usize {
        self.layer
    }

    /// Executes the exact frame-local mixer and dense SwiGLU residual equations.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        state.validate_hidden(hidden, &self.config)?;
        let normalized = self.operator_norm.forward(hidden, context)?;
        let mixed = match &mut self.mixer {
            Mixer::Attention(a) => a.forward(&normalized, state, context)?,
            Mixer::Convolution(c) => c.forward(&normalized, state, context)?,
        };
        let hidden = hidden.add(&mixed, context)?;
        let normalized = self.ffn_norm.forward(&hidden, context)?;
        let product = B::gated_product(
            self.gate.forward(&normalized, context)?,
            self.up.forward(&normalized, context)?,
            self.gated_product,
            context,
        )?;
        hidden.add(&self.down.forward(&product, context)?, context)
    }
}

/// Affine LayerNorm, contiguous spatial merger and exact-GELU language projector.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionProjector<B: NeuralBackend> {
    norm: LayerNorm<B::Tensor>,
    up: B::Linear,
    down: B::Linear,
    #[parameter(skip)]
    input_width: i32,
    #[parameter(skip)]
    config: VisionConfig,
}

impl<B: NeuralBackend> VisionProjector<B> {
    /// Constructs an unloaded projector independently of the encoder units.
    pub fn new(
        declaration: &Declaration,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            norm: LayerNorm::new(declaration.projector_norm.clone(), context)?,
            up: B::linear(declaration.projector_up.clone(), context)?,
            down: B::linear(declaration.projector_down.clone(), context)?,
            input_width: declaration.projector_up.input,
            config: declaration.config().clone(),
        })
    }

    /// Projects complete merge groups without mixing adjacent media frames.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        state: &VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        state.validate_hidden(hidden, &self.config)?;
        // The publisher evaluates normalization and the learned affine in
        // FP32 before rounding its BF16 output.
        let normalized = if hidden.element_type() == Some(TensorElementType::Bf16) {
            let wide = hidden.cast_float(TensorElementType::F32, context)?;
            self.norm
                .forward(&wide, context)?
                .cast_float(TensorElementType::Bf16, context)?
        } else {
            self.norm.forward(hidden, context)?
        };
        let merged = normalized.reshape(&[-1, self.input_width], context)?;
        self.down.forward(
            &B::Tensor::gelu(&self.up.forward(&merged, context)?, context)?,
            context,
        )
    }
}
