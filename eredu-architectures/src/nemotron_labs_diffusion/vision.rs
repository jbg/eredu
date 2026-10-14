//! Pixtral image blocks and the released spatial projector.
use super::{FamilyConfig, PixtralConfig};
use crate::decoder::{ComponentInstrumentation, Config};
use eredu_nn::{
    Error, Index, LinearOperator, LinearSpec, NeuralBackend, NormalizationConstructionSpec,
    NormalizationOperator, NormalizationScale, Parameter, ParameterSpec, Parameterized, Tensor,
};

fn linear<B: NeuralBackend>(
    name: &str,
    input: i32,
    output: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, Error> {
    B::linear(
        LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(format!("{name}.weight")).map_err(Error::backend)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(
                &format!("{name}.weight"),
                eredu_checkpoint::LinearFormat::Dense,
            )?,
        },
        context,
    )
}
fn norm<B: NeuralBackend>(
    name: &str,
    width: i32,
    epsilon: f32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Normalization, Error> {
    B::normalization(
        NormalizationConstructionSpec {
            groups: None,
            dimensions: width,
            epsilon,
            scale: NormalizationScale::Learned(
                ParameterSpec::trainable(format!("{name}.weight")).map_err(Error::backend)?,
            ),
        },
        context,
    )
}

/// Packed channel-major flattened patches, with one positive grid per image.
pub struct ImageInput<'a, T> {
    /// Patch rows shaped `[sum(height * width), channels * patch_size²]`.
    pub patches: &'a T,
    /// Ordered image patch grids; rotary positions reset for each image.
    pub grids: &'a [(i32, i32)],
}

/// Invocation-local positions and independent image spans. No vision KV cache exists.
#[derive(Debug, Clone)]
pub struct ImageState<T> {
    cosine: T,
    sine: T,
    grids: Vec<(i32, i32)>,
}
impl<T> ImageState<T> {
    /// Values retained while independently loaded image layers execute.
    pub fn retained_values(&self) -> [&T; 2] {
        [&self.cosine, &self.sine]
    }
}

impl<T: Tensor> ImageState<T> {
    /// Constructs parameter-free geometry for a downstream pipeline image owner.
    pub fn prepare(
        config: &FamilyConfig,
        grids: &[(i32, i32)],
        context: &T::Context,
    ) -> Result<ImageState<T>, Error> {
        config.validate_config()?;
        let merge = i32::try_from(config.spatial_merge_size).map_err(Error::backend)?;
        let config = config
            .vision
            .as_ref()
            .ok_or_else(|| Error::backend("text artifact has no Pixtral tower"))?;
        if grids.is_empty()
            || grids.iter().any(|&(h, w)| {
                h <= 0
                    || w <= 0
                    || h % merge != 0
                    || w % merge != 0
                    || h > config.image_size / config.patch_size
                    || w > config.image_size / config.patch_size
            })
        {
            return Err(Error::backend("invalid Pixtral image patch grids"));
        }
        let count = grids
            .iter()
            .try_fold(0i32, |n, &(h, w)| {
                h.checked_mul(w).and_then(|v| n.checked_add(v))
            })
            .ok_or_else(|| Error::backend("Pixtral patch count overflow"))?;
        let head = config.head_dim;
        let theta = config
            .rope_parameters
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .ok_or_else(|| Error::backend("missing Pixtral rotary base"))?;
        let mut cosine = Vec::new();
        let mut sine = Vec::new();
        for &(height, width) in grids {
            for row in 0..height {
                for column in 0..width {
                    let phases: Vec<_> = (0..head / 2)
                        .map(|i| {
                            let (position, frequency_index) = if i < head / 4 {
                                (row, 2 * i)
                            } else {
                                (column, 2 * (i - head / 4) + 1)
                            };
                            position as f32
                                * theta.powf(-(2 * frequency_index) as f64 / head as f64) as f32
                        })
                        .collect();
                    for _ in 0..2 {
                        for &phase in &phases {
                            cosine.push(phase.cos());
                            sine.push(phase.sin());
                        }
                    }
                }
            }
        }
        Ok(ImageState {
            cosine: T::from_f32_slice(&cosine, &[1, 1, count, head], context)?,
            sine: T::from_f32_slice(&sine, &[1, 1, count, head], context)?,
            grids: grids.to_vec(),
        })
    }
}

/// Patch convolution and pre-normalization owned by the first image stage.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ImageEmbedding<B: NeuralBackend> {
    /// Published convolution kernel; flattening is a storage view.
    pub patch_weight: Parameter<B::Tensor>,
    /// RMS normalization before image attention.
    pub ln_pre: B::Normalization,
    #[parameter(skip)]
    config: FamilyConfig,
}
impl<B: NeuralBackend> ImageEmbedding<B> {
    /// Declares only the image ingress parameters.
    pub fn new(
        config: &FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        let vision = config
            .vision
            .as_ref()
            .ok_or_else(|| Error::backend("text artifact has no Pixtral tower"))?;
        Ok(Self {
            patch_weight: Parameter::unloaded(
                ParameterSpec::trainable("model.vision_tower.patch_conv.weight")
                    .map_err(Error::backend)?,
                &[
                    vision.hidden_size,
                    vision.num_channels,
                    vision.patch_size,
                    vision.patch_size,
                ],
                context,
            )?,
            ln_pre: norm::<B>(
                "model.vision_tower.ln_pre",
                vision.hidden_size,
                1e-5,
                context,
            )?,
            config: config.clone(),
        })
    }
    /// Projects prepared patches without duplicating the convolution kernel.
    pub fn begin(
        &mut self,
        input: ImageInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, ImageState<B::Tensor>), Error> {
        let state = ImageState::<B::Tensor>::prepare(&self.config, input.grids, context)?;
        if input.patches.dim(0) != state.cosine.dim(2) {
            return Err(Error::backend("Pixtral grids and packed patches disagree"));
        }
        let vision = self
            .config
            .vision
            .as_ref()
            .expect("validated image embedding");
        let element = self
            .patch_weight
            .as_ref()
            .element_type()
            .ok_or_else(|| Error::backend("unknown Pixtral patch projection precision"))?;
        let patches = input.patches.cast_float(element, context)?;
        let hidden = eredu_nn::multimodal::project_flattened_patches(
            &patches,
            self.patch_weight.as_ref(),
            None,
            eredu_nn::multimodal::FlattenedPatchSpec {
                channels: vision.num_channels,
                temporal: 1,
                height: vision.patch_size,
                width: vision.patch_size,
                output: vision.hidden_size,
            },
            context,
        )?
        .reshape(&[1, input.patches.dim(0), vision.hidden_size], context)?;
        Ok((self.ln_pre.forward(&hidden, context)?, state))
    }
}

/// Spatial merger and image-to-text projector owned by the final image stage.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ImageProjector<B: NeuralBackend> {
    /// RMS normalization before spatial patch merging.
    pub projector_norm: B::Normalization,
    /// Channel-major spatial merger.
    pub merger: B::Linear,
    /// Image-to-text projection before exact GELU.
    pub linear_1: B::Linear,
    /// Final text-width projection.
    pub linear_2: B::Linear,
    #[parameter(skip)]
    config: PixtralConfig,
    #[parameter(skip)]
    merge: i32,
}
impl<B: NeuralBackend> ImageProjector<B> {
    /// Declares only the terminal image projection parameters.
    pub fn new(
        config: &FamilyConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate_config()?;
        let vision = config
            .vision
            .as_ref()
            .ok_or_else(|| Error::backend("text artifact has no Pixtral tower"))?;
        let root = "model.multi_modal_projector";
        let merge = i32::try_from(config.spatial_merge_size).map_err(Error::backend)?;
        Ok(Self {
            projector_norm: norm::<B>(
                &format!("{root}.norm"),
                vision.hidden_size,
                config.rms_norm_epsilon(),
                context,
            )?,
            merger: linear::<B>(
                &format!("{root}.patch_merger.merging_layer"),
                vision.hidden_size * merge * merge,
                vision.hidden_size,
                context,
            )?,
            linear_1: linear::<B>(
                &format!("{root}.linear_1"),
                vision.hidden_size,
                config.hidden_size(),
                context,
            )?,
            linear_2: linear::<B>(
                &format!("{root}.linear_2"),
                config.hidden_size(),
                config.hidden_size(),
                context,
            )?,
            config: vision.clone(),
            merge,
        })
    }
    /// Applies pre-merger normalization, spatial unfold, and the exact GELU projector.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &ImageState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.finish_observed(hidden, state, context, &mut eredu_runtime::NoopObserver)
    }

    /// Records and intervenes at the values consumed by the spatial projector.
    pub fn finish_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        hidden: &B::Tensor,
        state: &ImageState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let mut hooks = ComponentInstrumentation::new("model.multi_modal_projector", &mut observer);
        if hidden.shape() != [1, state.cosine.dim(2), self.config.hidden_size] {
            return Err(Error::backend("invalid Pixtral projector input"));
        }
        let normalized =
            hooks.apply("norm.output", self.projector_norm.forward(hidden, context)?)?;
        let mut offset = 0;
        let mut images = Vec::new();
        for &(height, width) in &state.grids {
            let image = normalized
                .index(
                    &[
                        Index::Full,
                        Index::Range(offset, offset + height * width),
                        Index::Full,
                    ],
                    context,
                )?
                .reshape(
                    &[
                        height / self.merge,
                        self.merge,
                        width / self.merge,
                        self.merge,
                        self.config.hidden_size,
                    ],
                    context,
                )?
                .transpose_axes(&[0, 2, 4, 1, 3], context)?
                .reshape(
                    &[
                        1,
                        (height / self.merge) * (width / self.merge),
                        self.config.hidden_size * self.merge * self.merge,
                    ],
                    context,
                )?;
            images.push(self.merger.forward(&image, context)?);
            offset += height * width;
        }
        let merged = hooks.apply(
            "patch_merger.output",
            B::Tensor::concatenate(&images, 1, context)?,
        )?;
        let projected = hooks.apply("linear_1.output", self.linear_1.forward(&merged, context)?)?;
        let activated = hooks.apply("gelu.output", B::Tensor::gelu(&projected, context)?)?;
        hooks.apply(
            "linear_2.output",
            self.linear_2.forward(&activated, context)?,
        )
    }
}

/// One independently addressable Pixtral residual block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct ImageLayer<B: NeuralBackend> {
    /// Pre-attention RMS scale.
    pub attention_norm: B::Normalization,
    /// Query projection.
    pub query: B::Linear,
    /// Key projection.
    pub key: B::Linear,
    /// Value projection.
    pub value: B::Linear,
    /// Attention output projection.
    pub output: B::Linear,
    /// Pre-MLP RMS scale.
    pub ffn_norm: B::Normalization,
    /// SwiGLU gate projection.
    pub gate: B::Linear,
    /// SwiGLU up projection.
    pub up: B::Linear,
    /// SwiGLU down projection.
    pub down: B::Linear,
    #[parameter(skip)]
    index: usize,
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    head_dim: i32,
    #[parameter(skip)]
    width: i32,
}
impl<B: NeuralBackend> ImageLayer<B> {
    /// Declares one layer under its exact source identity.
    pub fn new(
        config: &PixtralConfig,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::build(
            config,
            index,
            config.num_attention_heads,
            config.intermediate_size,
            context,
        )
    }

    /// Constructs only head and intermediate-channel shards selected by the
    /// neutral parameter plan. Residuals and RMS reductions retain full width.
    pub fn new_parallel(
        config: &PixtralConfig,
        index: usize,
        layout: &eredu_runtime::LocalModelLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let (heads, intermediate) =
            super::parallel::vision_local_widths(config, index, layout).map_err(Error::backend)?;
        Self::build(config, index, heads, intermediate, context)
    }

    fn build(
        config: &PixtralConfig,
        index: usize,
        heads: i32,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate().map_err(Error::backend)?;
        if index >= config.num_hidden_layers as usize
            || heads <= 0
            || heads > config.num_attention_heads
            || intermediate <= 0
            || intermediate > config.intermediate_size
        {
            return Err(Error::backend(
                "Pixtral layer or local geometry outside tower",
            ));
        }
        let root = format!("model.vision_tower.transformer.layers.{index}");
        let projection = |field: &str, input, output| {
            linear::<B>(&format!("{root}.{field}"), input, output, context)
        };
        let width = config.hidden_size;
        Ok(Self {
            attention_norm: norm::<B>(&format!("{root}.attention_norm"), width, 1e-5, context)?,
            query: projection("attention.q_proj", width, heads * config.head_dim)?,
            key: projection("attention.k_proj", width, heads * config.head_dim)?,
            value: projection("attention.v_proj", width, heads * config.head_dim)?,
            output: projection("attention.o_proj", heads * config.head_dim, width)?,
            ffn_norm: norm::<B>(&format!("{root}.ffn_norm"), width, 1e-5, context)?,
            gate: projection("feed_forward.gate_proj", width, intermediate)?,
            up: projection("feed_forward.up_proj", width, intermediate)?,
            down: projection("feed_forward.down_proj", intermediate, width)?,
            index,
            heads,
            head_dim: config.head_dim,
            width,
        })
    }

    /// Bidirectional attention is independent per image, equivalent to the packed
    /// block-diagonal visibility without allocating a cross-image square mask.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        state: &ImageState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_observed(
            hidden,
            state,
            None,
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    /// Executes selected head/channel shards and reduces their residual writes.
    pub fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &ImageState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_observed(
            hidden,
            state,
            Some(parallel),
            context,
            &mut eredu_runtime::NoopObserver,
        )
    }
    /// Observes complete residual writes after any tensor-parallel reduction.
    pub fn forward_observed<O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized>(
        &mut self,
        hidden: &B::Tensor,
        state: &ImageState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error> {
        let mut observer = eredu_runtime::BorrowedActivationObserver(observer);
        let path = format!("model.vision_tower.transformer.layers.{}", self.index);
        let mut hooks = ComponentInstrumentation::new(&path, &mut observer);
        if hidden.shape() != [1, state.cosine.dim(2), self.width] {
            return Err(Error::backend("invalid Pixtral layer input"));
        }
        let normalized = hooks.apply(
            "attention_norm.output",
            self.attention_norm.forward(hidden, context)?,
        )?;
        let reshape = |tensor: B::Tensor| {
            tensor
                .reshape(&[1, hidden.dim(1), self.heads, self.head_dim], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let query = rotate(
            &reshape(self.query.forward(&normalized, context)?)?,
            state,
            context,
        )?;
        let key = rotate(
            &reshape(self.key.forward(&normalized, context)?)?,
            state,
            context,
        )?;
        let value = reshape(self.value.forward(&normalized, context)?)?;
        let mut offset = 0;
        let mut attended = Vec::new();
        for &(height, width) in &state.grids {
            let slice = |tensor: &B::Tensor| {
                tensor.index(
                    &[
                        Index::Full,
                        Index::Full,
                        Index::Range(offset, offset + height * width),
                        Index::Full,
                    ],
                    context,
                )
            };
            attended.push(B::attention_with_sinks(
                eredu_nn::AttentionRequest {
                    queries: slice(&query)?,
                    keys: slice(&key)?,
                    values: slice(&value)?,
                    scale: 1.0 / (self.head_dim as f32).sqrt(),
                    mask: None,
                    sinks: None,
                    softcap: None,
                    arithmetic: eredu_nn::AttentionArithmetic::InputScores,
                },
                context,
            )?);
            offset += height * width;
        }
        let attended = B::Tensor::concatenate(&attended, 2, context)?
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[1, hidden.dim(1), self.heads * self.head_dim], context)?;
        let attention = match parallel {
            Some(parallel) => {
                B::row_parallel_linear(&mut self.output, &attended, parallel, context)?
            }
            None => self.output.forward(&attended, context)?,
        };
        let attention = hooks.apply("attention.output", attention)?;
        let hidden = hooks.apply("attention.residual", hidden.add(&attention, context)?)?;
        let normalized =
            hooks.apply("ffn_norm.output", self.ffn_norm.forward(&hidden, context)?)?;
        let gate = B::silu(self.gate.forward(&normalized, context)?, context)?;
        let feed = gate.multiply(&self.up.forward(&normalized, context)?, context)?;
        let feed = match parallel {
            Some(parallel) => B::row_parallel_linear(&mut self.down, &feed, parallel, context)?,
            None => self.down.forward(&feed, context)?,
        };
        let feed = hooks.apply("feed_forward.output", feed)?;
        hooks.apply("feed_forward.residual", hidden.add(&feed, context)?)
    }
}

fn rotate<T: Tensor>(input: &T, state: &ImageState<T>, context: &T::Context) -> Result<T, Error> {
    let element = input
        .element_type()
        .ok_or_else(|| Error::backend("unknown Pixtral rotary precision"))?;
    let half = input.dim(3) / 2;
    let first = input.index(
        &[Index::Full, Index::Full, Index::Full, Index::Range(0, half)],
        context,
    )?;
    let second = input.index(
        &[
            Index::Full,
            Index::Full,
            Index::Full,
            Index::Range(half, 2 * half),
        ],
        context,
    )?;
    let negative = T::full_f32(-1.0, &[], context)?.cast_float(element, context)?;
    let rotated = T::concatenate(&[second.multiply(&negative, context)?, first], 3, context)?;
    input
        .multiply(&state.cosine.cast_float(element, context)?, context)?
        .add(
            &rotated.multiply(&state.sine.cast_float(element, context)?, context)?,
            context,
        )
}
