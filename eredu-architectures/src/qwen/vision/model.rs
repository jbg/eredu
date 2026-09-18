//! Shared backend-neutral Qwen vision equations.

use eredu_nn::{
    EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator, LinearSpec, NeuralBackend,
    Parameter, ParameterSpec, Parameterized, SegmentedAttentionInput, Tensor,
    multimodal::{
        FlattenedPatchSpec, MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
        multi_axis_rotary_embeddings, project_flattened_patches,
    },
    sequence_layout::{
        InterpolationMode, PatchTraversal, attention_chunk_lengths, bilinear_interpolation_samples,
        inverse_permutation, patch_positions, validate_patch_grid, visit_gathered_patch_positions,
        window_partition,
    },
};

use super::{VisionAttentionPolicy, VisionConfig, VisionMode};
use eredu_runtime::input::PreparedModelInputOwner;
mod prepared_tables;

/// Prepared flattened patches and their `(time, height, width)` grids.
pub struct VisionInput<'a, T> {
    /// Patches shaped `[patches, channels * temporal * patch_size²]`.
    pub pixels: &'a T,
    /// Ordered media grids.
    pub grid: &'a [(i32, i32, i32)],
}

/// Parameter-free state retained between streamable vision blocks.
#[derive(Debug, Clone)]
pub struct VisionState<T> {
    tables: VisionStateTables<T>,
    cosine: T,
    sine: T,
    deepstack: Vec<T>,
    metadata: Option<eredu_nn::workspace::WorkspaceContext>,
}
#[derive(Debug, Clone)]
enum VisionStateTables<T> {
    Ordinary {
        full_chunks: Vec<i32>,
        window_chunks: Vec<i32>,
        permutation: Vec<i32>,
    },
    Original(PreparedModelInputOwner<T>),
    Projected(eredu_runtime::input::OriginalEncoderTableProjection),
}
impl<T> VisionStateTables<T> {
    fn full_chunks(&self) -> &[i32] {
        match self {
            Self::Ordinary { full_chunks, .. } => full_chunks,
            Self::Original(owner) => owner
                .original_encoder_tables()
                .expect("original tables")
                .full_chunks(),
            Self::Projected(owner) => owner.tables().full_chunks(),
        }
    }
    fn window_chunks(&self) -> &[i32] {
        match self {
            Self::Ordinary { window_chunks, .. } => window_chunks,
            Self::Original(owner) => owner
                .original_encoder_tables()
                .expect("original tables")
                .window_chunks(),
            Self::Projected(owner) => owner.tables().window_chunks(),
        }
    }
    fn permutation(&self) -> &[i32] {
        match self {
            Self::Ordinary { permutation, .. } => permutation,
            Self::Original(owner) => owner
                .original_encoder_tables()
                .expect("original tables")
                .permutation(),
            Self::Projected(owner) => owner.tables().permutation(),
        }
    }
    fn inverse(
        &self,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<std::borrow::Cow<'_, [i32]>, Error> {
        metadata.controls::<std::borrow::Cow<'_, [i32]>>()?;
        match self {
            Self::Ordinary { permutation, .. } => match metadata.context() {
                Some(context) => eredu_nn::sequence_layout::inverse_permutation_with_metadata(
                    permutation,
                    context,
                )
                .map(std::borrow::Cow::Owned),
                None => inverse_permutation(permutation)
                    .map(std::borrow::Cow::Owned)
                    .map_err(Error::backend),
            },
            Self::Original(owner) => Ok(std::borrow::Cow::Borrowed(
                owner
                    .original_encoder_tables()
                    .expect("original tables")
                    .inverse(),
            )),
            Self::Projected(owner) => Ok(std::borrow::Cow::Borrowed(owner.tables().inverse())),
        }
    }
}

impl<T> VisionState<T> {
    /// Full-attention contiguous segment lengths.
    pub fn full_chunks(&self) -> &[i32] {
        self.tables.full_chunks()
    }
    /// Window-attention contiguous segment lengths.
    pub fn window_chunks(&self) -> &[i32] {
        self.tables.window_chunks()
    }
    /// Merge-group execution permutation.
    pub fn permutation(&self) -> &[i32] {
        self.tables.permutation()
    }
    /// Captured DeepStack features.
    pub fn deepstack_features(&self) -> &[T] {
        &self.deepstack
    }
    /// Backend tensors that must remain live while streamable blocks execute.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        std::iter::once(&self.cosine)
            .chain(std::iter::once(&self.sine))
            .chain(self.deepstack.iter())
    }
    /// Replaces the backend tensors transported between component owners.
    pub fn replace_retained_values(&mut self, mut values: Vec<T>) -> Result<(), Error> {
        if values.len() < 2 {
            return Err(Error::backend(
                "vision continuation requires rotary cosine and sine tensors",
            ));
        }
        let mut prefix = values.drain(..2);
        let cosine = prefix.next().expect("validated cosine");
        let sine = prefix.next().expect("validated sine");
        drop(prefix);
        self.cosine = cosine;
        self.sine = sine;
        self.deepstack = values;
        Ok(())
    }
    fn chunks(&self, policy: VisionAttentionPolicy) -> &[i32] {
        match policy {
            VisionAttentionPolicy::Full => self.tables.full_chunks(),
            VisionAttentionPolicy::Windowed => self.tables.window_chunks(),
        }
    }
}

/// Final media embeddings and selected intermediate features.
#[derive(Debug, Clone)]
pub struct VisionOutput<T> {
    /// Final `[1, media_tokens, text_hidden]` embeddings.
    pub embeddings: T,
    /// DeepStack features in merger-bank order.
    pub deepstack_features: Vec<T>,
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(super) struct LayerNorm<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    weight: Parameter<B::Tensor>,
    bias: Parameter<B::Tensor>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> LayerNorm<B> {
    fn new(
        prefix: &str,
        width: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let modules = crate::decoder::ModuleMetadata::new::<B>(context);
        modules.controls::<(Self, ParameterSpec, [i32; 1])>()?;
        let make = |field: &str| {
            Parameter::unloaded(
                modules.named_parameter(format_args!("{prefix}.{field}"))?,
                &[width],
                context,
            )
        };
        Ok(Self {
            weight: make("weight")?,
            bias: make("bias")?,
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
            1e-6,
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(super) struct PatchEmbed<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    weight: Parameter<B::Tensor>,
    bias: Parameter<B::Tensor>,
    #[parameter(skip, metadata)]
    spec: FlattenedPatchSpec,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> PatchEmbed<B> {
    fn new(
        config: &VisionConfig,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let modules = crate::decoder::ModuleMetadata::new::<B>(context);
        modules.controls::<(Self, ParameterSpec, [i32; 5], FlattenedPatchSpec)>()?;
        let spec = FlattenedPatchSpec {
            channels: config.in_channels,
            temporal: config.temporal_patch_size,
            height: config.patch_size,
            width: config.patch_size,
            output: config.hidden_size,
        };
        Ok(Self {
            weight: Parameter::unloaded(
                modules.named_parameter(format_args!(
                    "{}",
                    RootedName(root, "patch_embed.proj.weight")
                ))?,
                &[
                    config.hidden_size,
                    config.in_channels,
                    config.temporal_patch_size,
                    config.patch_size,
                    config.patch_size,
                ],
                context,
            )?,
            bias: Parameter::unloaded(
                modules.named_parameter(format_args!(
                    "{}",
                    RootedName(root, "patch_embed.proj.bias")
                ))?,
                &[config.hidden_size],
                context,
            )?,
            spec,
        })
    }
    fn forward(
        &self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        project_flattened_patches(
            input,
            self.weight.as_ref(),
            Some(self.bias.as_ref()),
            self.spec,
            context,
        )
    }
}

fn linear<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    config: &VisionConfig,
    prefix: &str,
    input: i32,
    output: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, Error> {
    let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
    let modules = crate::decoder::ModuleMetadata::new::<B>(context);
    modules.controls::<(B::Linear, LinearSpec, String)>()?;
    let weight = metadata.format(format_args!("{prefix}.weight"))?;
    B::linear(
        LinearSpec {
            input,
            output,
            weight: modules.plain_parameter(&weight)?,
            bias: Some(modules.named_parameter(format_args!("{prefix}.bias"))?),
            format: modules.format(
                &weight,
                config.linear_format_with_metadata(relative_vision_name(&weight), metadata)?,
            )?,
        },
        context,
    )
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(super) struct Attention<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    pub(super) qkv: B::Linear,
    pub(super) output: B::Linear,
    #[parameter(skip, metadata)]
    heads: i32,
    #[parameter(skip, metadata)]
    head_dim: i32,
    #[parameter(skip, metadata)]
    scale: f32,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> Attention<B> {
    fn new_with_heads(
        config: &VisionConfig,
        parameter_root: &str,
        layer: usize,
        heads: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(Self, String)>()?;
        let root = rooted_with_metadata(
            parameter_root,
            &metadata.format(format_args!("blocks.{layer}.attn"))?,
            metadata,
        )?;
        let head_dim = config.hidden_size / config.num_heads;
        Ok(Self {
            qkv: linear::<B>(
                config,
                &metadata.format(format_args!("{root}.qkv"))?,
                config.hidden_size,
                3 * heads * head_dim,
                context,
            )?,
            output: linear::<B>(
                config,
                &metadata.format(format_args!("{root}.proj"))?,
                heads * head_dim,
                config.hidden_size,
                context,
            )?,
            heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }
    fn forward(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        let sequence = hidden.dim(0);
        let qkv = self
            .qkv
            .forward(hidden, context)?
            .reshape(&[sequence, 3, self.heads, self.head_dim], context)?;
        let select = |n| {
            qkv.index(
                &[
                    Index::Full,
                    Index::Range(n, n + 1),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .squeeze_axes(&[1], context)
        };
        let query = apply_rotary(&select(0)?, cos, sin, context)?;
        let key = apply_rotary(&select(1)?, cos, sin, context)?;
        let value = select(2)?;
        let attended = segmented_attention::<B>(
            SegmentedAttentionInput {
                queries: &query,
                keys: &key,
                values: &value,
                segment_lengths: chunks,
                scale: self.scale,
            },
            context,
            metadata,
        )?;
        self.forward_output(&attended, None, context)
    }

    fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        let sequence = hidden.dim(0);
        let qkv = self
            .qkv
            .forward(hidden, context)?
            .reshape(&[sequence, 3, self.heads, self.head_dim], context)?;
        let select = |n| {
            qkv.index(
                &[
                    Index::Full,
                    Index::Range(n, n + 1),
                    Index::Full,
                    Index::Full,
                ],
                context,
            )?
            .squeeze_axes(&[1], context)
        };
        let query = apply_rotary(&select(0)?, cos, sin, context)?;
        let key = apply_rotary(&select(1)?, cos, sin, context)?;
        let value = select(2)?;
        let attended = segmented_attention::<B>(
            SegmentedAttentionInput {
                queries: &query,
                keys: &key,
                values: &value,
                segment_lengths: chunks,
                scale: self.scale,
            },
            context,
            metadata,
        )?;
        self.forward_output(&attended, Some(parallel), context)
    }

    fn forward_output(
        &mut self,
        attended: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let sequence = attended.dim(0);
        let attended = attended.reshape(&[sequence, self.heads * self.head_dim], context)?;
        match parallel {
            Some(parallel) => {
                B::row_parallel_linear(&mut self.output, &attended, parallel, context)
            }
            None => self.output.forward(&attended, context),
        }
    }
}

fn segmented_attention<B: NeuralBackend>(
    input: SegmentedAttentionInput<'_, B::Tensor>,
    context: &<B::Tensor as Tensor>::Context,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<B::Tensor, Error> {
    match metadata.context() {
        Some(metadata) => B::segmented_attention_with_metadata(input, context, metadata),
        None => B::segmented_attention(input, context),
    }
}

/// One shared, independently streamable vision transformer block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionBlock<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    pub(super) norm1: LayerNorm<B>,
    pub(super) attention: Attention<B>,
    pub(super) norm2: LayerNorm<B>,
    pub(super) fc1: B::Linear,
    pub(super) fc2: B::Linear,
    #[parameter(skip, metadata)]
    activation: String,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionBlock<B> {
    /// Builds one unloaded canonical block.
    pub fn new(
        config: &VisionConfig,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_root(config, "", layer, context)
    }

    /// Builds one block below an explicit canonical parameter root.
    pub fn new_with_root(
        config: &VisionConfig,
        parameter_root: &str,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_parallel_with_root(
            config,
            parameter_root,
            layer,
            config.num_heads,
            config.intermediate_size,
            context,
        )
    }

    /// Builds rank-local attention heads and MLP channels at canonical paths.
    pub fn new_parallel_with_root(
        config: &VisionConfig,
        parameter_root: &str,
        layer: usize,
        heads: i32,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(Self, String)>()?;
        let root = rooted_with_metadata(
            parameter_root,
            &metadata.format(format_args!("blocks.{layer}"))?,
            metadata,
        )?;
        Ok(Self {
            norm1: LayerNorm::new(
                &metadata.format(format_args!("{root}.norm1"))?,
                config.hidden_size,
                context,
            )?,
            attention: Attention::new_with_heads(config, parameter_root, layer, heads, context)?,
            norm2: LayerNorm::new(
                &metadata.format(format_args!("{root}.norm2"))?,
                config.hidden_size,
                context,
            )?,
            fc1: linear::<B>(
                config,
                &metadata.format(format_args!("{root}.mlp.linear_fc1"))?,
                config.hidden_size,
                intermediate,
                context,
            )?,
            fc2: linear::<B>(
                config,
                &metadata.format(format_args!("{root}.mlp.linear_fc2"))?,
                intermediate,
                config.hidden_size,
                context,
            )?,
            activation: metadata.text(&config.hidden_act)?,
        })
    }
    /// Applies exact attention/MLP residual equations over validated segments.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_with_metadata(
            hidden,
            chunks,
            cos,
            sin,
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }

    fn forward_with_metadata(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        let normed = self.norm1.forward(hidden, context)?;
        let hidden = hidden.add(
            &self
                .attention
                .forward(&normed, chunks, cos, sin, context, metadata)?,
            context,
        )?;
        let normed = self.norm2.forward(&hidden, context)?;
        let projected = self.fc1.forward(&normed, context)?;
        let activated = match self.activation.as_str() {
            "silu" => B::silu(projected, context)?,
            "gelu" => B::Tensor::gelu(&projected, context)?,
            "gelu_pytorch_tanh" => B::gelu_approximate(projected, context)?,
            other => {
                return Err(metadata.error(format_args!("unsupported vision activation {other:?}")));
            }
        };
        hidden.add(&self.fc2.forward(&activated, context)?, context)
    }

    /// Applies the same block with local heads/channels and two exact row reductions.
    pub fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel_with_metadata(
            hidden,
            chunks,
            cos,
            sin,
            parallel,
            context,
            crate::decoder::identity::Metadata::new(B::construction_metadata(context)),
        )
    }

    fn forward_parallel_with_metadata(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        metadata: crate::decoder::identity::Metadata<'_>,
    ) -> Result<B::Tensor, Error> {
        let normed = self.norm1.forward(hidden, context)?;
        let hidden = hidden.add(
            &self
                .attention
                .forward_parallel(&normed, chunks, cos, sin, parallel, context, metadata)?,
            context,
        )?;
        let normed = self.norm2.forward(&hidden, context)?;
        let projected = self.fc1.forward(&normed, context)?;
        let activated = match self.activation.as_str() {
            "silu" => B::silu(projected, context)?,
            "gelu" => B::Tensor::gelu(&projected, context)?,
            "gelu_pytorch_tanh" => B::gelu_approximate(projected, context)?,
            other => {
                return Err(metadata.error(format_args!("unsupported vision activation {other:?}")));
            }
        };
        hidden.add(
            &B::row_parallel_linear(&mut self.fc2, &activated, parallel, context)?,
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(super) struct Merger<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    pub(super) norm: LayerNorm<B>,
    pub(super) fc1: B::Linear,
    pub(super) fc2: B::Linear,
    #[parameter(skip, metadata)]
    unit: i32,
    #[parameter(skip, metadata)]
    width: i32,
    #[parameter(skip, metadata)]
    postshuffle: bool,
    #[parameter(skip, metadata)]
    approximate: bool,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> Merger<B> {
    fn new_with_intermediate(
        config: &VisionConfig,
        prefix: &str,
        postshuffle: bool,
        approximate: bool,
        intermediate: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(Self, String)>()?;
        let unit = config.spatial_merge_size * config.spatial_merge_size;
        let width = config.hidden_size * unit;
        Ok(Self {
            norm: LayerNorm::new(
                &metadata.format(format_args!("{prefix}.norm"))?,
                if postshuffle {
                    width
                } else {
                    config.hidden_size
                },
                context,
            )?,
            fc1: linear::<B>(
                config,
                &metadata.format(format_args!("{prefix}.linear_fc1"))?,
                width,
                intermediate,
                context,
            )?,
            fc2: linear::<B>(
                config,
                &metadata.format(format_args!("{prefix}.linear_fc2"))?,
                intermediate,
                config.out_hidden_size,
                context,
            )?,
            unit,
            width,
            postshuffle,
            approximate,
        })
    }
    fn forward(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if hidden.dim(0) % self.unit != 0 {
            return Err(
                crate::decoder::identity::Metadata::new(B::construction_metadata(context)).error(
                    format_args!("vision merge geometry does not divide sequence"),
                ),
            );
        }
        let hidden = if self.postshuffle {
            self.norm
                .forward(&hidden.reshape(&[-1, self.width], context)?, context)?
        } else {
            self.norm
                .forward(hidden, context)?
                .reshape(&[-1, self.width], context)?
        };
        let hidden = self.fc1.forward(&hidden, context)?;
        let hidden = if self.approximate {
            B::gelu_approximate(hidden, context)?
        } else {
            B::Tensor::gelu(&hidden, context)?
        };
        self.fc2.forward(&hidden, context)
    }

    fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if hidden.dim(0) % self.unit != 0 {
            return Err(
                crate::decoder::identity::Metadata::new(B::construction_metadata(context)).error(
                    format_args!("vision merge geometry does not divide sequence"),
                ),
            );
        }
        let hidden = if self.postshuffle {
            self.norm
                .forward(&hidden.reshape(&[-1, self.width], context)?, context)?
        } else {
            self.norm
                .forward(hidden, context)?
                .reshape(&[-1, self.width], context)?
        };
        let hidden = self.fc1.forward(&hidden, context)?;
        let hidden = if self.approximate {
            B::gelu_approximate(hidden, context)?
        } else {
            B::Tensor::gelu(&hidden, context)?
        };
        B::row_parallel_linear(&mut self.fc2, &hidden, parallel, context)
    }
}

/// Pinned patch/position/merger modules used by resident and bounded execution.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionStatic<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    pub(super) position: B::Embedding,
    pub(super) patch: PatchEmbed<B>,
    pub(super) merger: Merger<B>,
    pub(super) deepstack_mergers: Vec<Merger<B>>,
    #[parameter(skip, metadata)]
    config: crate::replicated_text::SharedCompositeConfig<VisionConfig>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionStatic<B> {
    /// Builds static modules for one explicit vision mode.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_root(config, "", context)
    }

    /// Builds pinned vision modules below an explicit canonical root.
    pub fn new_with_root(
        config: VisionConfig,
        parameter_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_config(
            crate::replicated_text::CompositeModelConfig::Owned(config),
            parameter_root,
            context,
        )
    }

    pub(crate) fn new_with_config(
        config: crate::replicated_text::CompositeModelConfig<VisionConfig>,
        parameter_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(
            Self,
            crate::replicated_text::CompositeModelConfig<VisionConfig>,
            Vec<i32>,
        )>()?;
        let width = config.hidden_size * config.spatial_merge_size * config.spatial_merge_size;
        let count = config.deepstack_layer_count() + 1;
        let mut intermediates = metadata.vector(count)?;
        intermediates.resize(count, width);
        Self::new_parallel_with_config(config, parameter_root, &intermediates, context)
    }

    /// Builds pinned modules with rank-local merger intermediate widths.
    pub fn new_parallel_with_root(
        config: VisionConfig,
        parameter_root: &str,
        merger_intermediates: &[i32],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_parallel_with_config(
            crate::replicated_text::CompositeModelConfig::Owned(config),
            parameter_root,
            merger_intermediates,
            context,
        )
    }

    pub(crate) fn new_parallel_with_config(
        config: crate::replicated_text::CompositeModelConfig<VisionConfig>,
        parameter_root: &str,
        merger_intermediates: &[i32],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        let modules = crate::decoder::ModuleMetadata::new::<B>(context);
        metadata.controls::<(Self, EmbeddingSpec, Vec<Merger<B>>)>()?;
        match B::construction_metadata(context).filter(|context| context.uses_checked_metadata()) {
            Some(context) => config.validate_with_metadata(context)?,
            None => config.validate().map_err(Error::backend)?,
        }
        if merger_intermediates.len() != config.deepstack_layer_count() + 1
            || merger_intermediates.iter().any(|width| *width <= 0)
        {
            return Err(
                metadata.error(format_args!("vision merger local-width plan is incomplete"))
            );
        }
        let position = B::embedding(
            EmbeddingSpec {
                vocabulary: config.num_position_embeddings,
                dimensions: config.hidden_size,
                weight: modules.named_parameter(format_args!(
                    "{}",
                    RootedName(parameter_root, "pos_embed.weight")
                ))?,
                format: modules.format(
                    &rooted_with_metadata(parameter_root, "pos_embed.weight", metadata)?,
                    eredu_checkpoint::LinearFormat::Dense,
                )?,
            },
            context,
        )?;
        let patch = PatchEmbed::new(&config, parameter_root, context)?;
        let merger = Merger::new_with_intermediate(
            &config,
            &rooted_with_metadata(parameter_root, "merger", metadata)?,
            false,
            config.mode == VisionMode::WindowScheduled,
            merger_intermediates[0],
            context,
        )?;
        let mut deepstack_mergers = metadata.vector(config.deepstack_layer_count())?;
        for i in 0..config.deepstack_layer_count() {
            let relative = metadata.format(format_args!("deepstack_merger_list.{i}"))?;
            let name = rooted_with_metadata(parameter_root, &relative, metadata)?;
            deepstack_mergers.push(Merger::new_with_intermediate(
                &config,
                &name,
                true,
                false,
                merger_intermediates[i + 1],
                context,
            )?);
        }
        Ok(Self {
            position,
            patch,
            merger,
            deepstack_mergers,
            config: config.into_shared(B::construction_metadata(context))?,
        })
    }

    /// Embeds patches and constructs reusable continuation state.
    pub fn begin(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, VisionState<B::Tensor>), Error> {
        validate_patch_grid(
            input.grid,
            self.config.spatial_merge_size,
            Some(input.pixels.dim(0)),
        )
        .map_err(Error::backend)?;
        let mut hidden = self.patch.forward(input.pixels, context)?;
        let positions = match self.config.mode {
            VisionMode::DeepStack => interpolated_positions::<B>(
                &mut self.position,
                input.grid,
                self.config.num_position_embeddings,
                self.config.spatial_merge_size,
                context,
            )?,
            VisionMode::WindowScheduled => gathered_positions::<B>(
                &mut self.position,
                input.grid,
                self.config.num_position_embeddings,
                context,
            )?,
        };
        hidden = hidden.add(&positions, context)?;
        let (hidden, state) =
            self.continuation_state_with_hidden(input.grid, hidden.dim(0), Some(hidden), context)?;
        Ok((hidden.expect("patch input was supplied"), state))
    }

    /// Builds parameter-free state for a received patch activation, without
    /// rerunning patch or learned-position projection on continuation owners.
    pub(crate) fn continuation_state(
        &self,
        grid: &[(i32, i32, i32)],
        sequence: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionState<B::Tensor>, Error> {
        validate_patch_grid(grid, self.config.spatial_merge_size, Some(sequence))
            .map_err(Error::backend)?;
        self.continuation_state_with_hidden(grid, sequence, None, context)
            .map(|(_, state)| state)
    }

    fn continuation_state_with_hidden(
        &self,
        grid: &[(i32, i32, i32)],
        sequence: i32,
        hidden: Option<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(Option<B::Tensor>, VisionState<B::Tensor>), Error> {
        let full_chunks = attention_chunk_lengths(grid).map_err(Error::backend)?;
        let unit = self.config.spatial_merge_size * self.config.spatial_merge_size;
        let (permutation, window_chunks) = match self.config.mode {
            VisionMode::DeepStack => ((0..sequence / unit).collect(), full_chunks.clone()),
            VisionMode::WindowScheduled => {
                let p = window_partition(
                    grid,
                    self.config.spatial_merge_size,
                    self.config.window_size,
                    self.config.patch_size,
                )
                .map_err(Error::backend)?;
                (p.permutation, p.chunk_lengths)
            }
        };
        self.reorder_continuation(
            sequence,
            hidden,
            VisionStateTables::Ordinary {
                full_chunks,
                window_chunks,
                permutation,
            },
            || rotary_embeddings::<B::Tensor>(grid, &self.config, context),
            context,
            B::construction_metadata(context),
        )
    }

    fn reorder_continuation(
        &self,
        sequence: i32,
        hidden: Option<B::Tensor>,
        tables: VisionStateTables<B::Tensor>,
        rotary: impl FnOnce() -> Result<(B::Tensor, B::Tensor), Error>,
        context: &<B::Tensor as Tensor>::Context,
        metadata: Option<&eredu_nn::workspace::WorkspaceContext>,
    ) -> Result<(Option<B::Tensor>, VisionState<B::Tensor>), Error> {
        let metadata = crate::decoder::identity::Metadata::new(metadata);
        metadata.controls::<(
            VisionState<B::Tensor>,
            VisionStateTables<B::Tensor>,
            [i32; 3],
        )>()?;
        let deepstack = metadata.vector(self.config.deepstack_layer_count())?;
        let unit = self.config.spatial_merge_size * self.config.spatial_merge_size;
        let permutation = tables.permutation();
        let indexes = B::Tensor::from_i32_slice(permutation, &[permutation.len() as i32], context)?;
        let reorder = |value: B::Tensor| {
            value
                .reshape(&[-1, unit, value.dim(1)], context)?
                .take_axis(&indexes, 0, context)?
                .reshape(&[sequence, -1], context)
        };
        let hidden = hidden.map(reorder).transpose()?;
        let (cosine, sine) = rotary()?;
        Ok((
            hidden,
            VisionState {
                metadata: metadata.context().cloned(),
                tables,
                cosine: reorder(cosine)?,
                sine: reorder(sine)?,
                deepstack,
            },
        ))
    }

    /// Executes one scheduled block and captures selected DeepStack output.
    pub fn forward_block(
        &mut self,
        block: &mut VisionBlock<B>,
        layer: usize,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let retained_metadata = state.metadata.clone();
        let metadata = crate::decoder::identity::Metadata::new(
            retained_metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(B::Tensor, super::VisionLayerPolicy)>()?;
        let policy = *self
            .config
            .layer_policy(layer)
            .ok_or_else(|| metadata.error(format_args!("vision schedule has no block {layer}")))?;
        let hidden = block.forward_with_metadata(
            hidden,
            state.chunks(policy.attention),
            &state.cosine,
            &state.sine,
            context,
            metadata,
        )?;
        if let Some(index) = policy.deepstack_merger {
            if let Some(context) = metadata.context() {
                context.reserve_metadata_vec(&mut state.deepstack, 1)?;
            } else {
                state.deepstack.reserve(1);
            }
            let feature = self
                .deepstack_mergers
                .get_mut(index as usize)
                .ok_or_else(|| metadata.error(format_args!("missing DeepStack merger")))?
                .forward(&hidden, context)?
                .expand_dims(0, context)?;
            state.deepstack.push(feature);
        }
        Ok(hidden)
    }

    /// Executes one rank-local block with architecture-owned segmented attention state.
    pub fn forward_block_parallel(
        &mut self,
        block: &mut VisionBlock<B>,
        layer: usize,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let retained_metadata = state.metadata.clone();
        let metadata = crate::decoder::identity::Metadata::new(
            retained_metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(B::Tensor, super::VisionLayerPolicy)>()?;
        let policy = *self
            .config
            .layer_policy(layer)
            .ok_or_else(|| metadata.error(format_args!("vision schedule has no block {layer}")))?;
        let hidden = block.forward_parallel_with_metadata(
            hidden,
            state.chunks(policy.attention),
            &state.cosine,
            &state.sine,
            parallel,
            context,
            metadata,
        )?;
        if let Some(index) = policy.deepstack_merger {
            if let Some(context) = metadata.context() {
                context.reserve_metadata_vec(&mut state.deepstack, 1)?;
            } else {
                state.deepstack.reserve(1);
            }
            let feature = self
                .deepstack_mergers
                .get_mut(index as usize)
                .ok_or_else(|| metadata.error(format_args!("missing DeepStack merger")))?
                .forward_parallel(&hidden, parallel, context)?
                .expand_dims(0, context)?;
            state.deepstack.push(feature);
        }
        Ok(hidden)
    }

    /// Restores original media order and returns all language-space features.
    pub fn finish(
        &mut self,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionOutput<B::Tensor>, Error> {
        let retained_metadata = state.metadata.clone();
        let metadata = crate::decoder::identity::Metadata::new(
            retained_metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(VisionOutput<B::Tensor>, B::Tensor)>()?;
        let merged = self.merger.forward(hidden, context)?;
        let inverse = state.tables.inverse(metadata)?;
        let inverse = B::Tensor::from_i32_slice(&inverse, &[inverse.len() as i32], context)?;
        Ok(VisionOutput {
            embeddings: merged
                .take_axis(&inverse, 0, context)?
                .expand_dims(0, context)?,
            deepstack_features: std::mem::take(&mut state.deepstack),
        })
    }

    /// Completes a rank-local tower and reduces the final merger exactly once.
    pub fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionOutput<B::Tensor>, Error> {
        let retained_metadata = state.metadata.clone();
        let metadata = crate::decoder::identity::Metadata::new(
            retained_metadata
                .as_ref()
                .or_else(|| B::construction_metadata(context)),
        );
        metadata.controls::<(VisionOutput<B::Tensor>, B::Tensor)>()?;
        let merged = self.merger.forward_parallel(hidden, parallel, context)?;
        let inverse = state.tables.inverse(metadata)?;
        let inverse = B::Tensor::from_i32_slice(&inverse, &[inverse.len() as i32], context)?;
        Ok(VisionOutput {
            embeddings: merged
                .take_axis(&inverse, 0, context)?
                .expand_dims(0, context)?,
            deepstack_features: std::mem::take(&mut state.deepstack),
        })
    }
}

/// Resident tower executing the same streamable block lifecycle.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct VisionTower<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Pinned vision modules.
    pub static_modules: VisionStatic<B>,
    /// Independently streamable blocks.
    pub blocks: Vec<VisionBlock<B>>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> VisionTower<B> {
    /// Builds one shared tower.
    pub fn new(
        config: VisionConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_root(config, "", context)
    }

    /// Builds the tower below an explicit canonical parameter root.
    pub fn new_with_root(
        config: VisionConfig,
        parameter_root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let metadata = crate::decoder::identity::Metadata::new(B::construction_metadata(context));
        metadata.controls::<(Self, Vec<VisionBlock<B>>)>()?;
        let mut blocks = metadata.vector(config.layer_count())?;
        for i in 0..config.layer_count() {
            blocks.push(VisionBlock::new_with_root(
                &config,
                parameter_root,
                i,
                context,
            )?);
        }
        Ok(Self {
            static_modules: VisionStatic::new_with_root(config, parameter_root, context)?,
            blocks,
        })
    }
    /// Runs the complete resident tower.
    pub fn forward(
        &mut self,
        input: VisionInput<'_, B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionOutput<B::Tensor>, Error> {
        let (mut hidden, mut state) = self.static_modules.begin(input, context)?;
        for i in 0..self.blocks.len() {
            hidden = self.static_modules.forward_block(
                &mut self.blocks[i],
                i,
                &hidden,
                &mut state,
                context,
            )?;
        }
        self.static_modules.finish(&hidden, &mut state, context)
    }
}

struct RootedName<'a>(&'a str, &'a str);
impl std::fmt::Display for RootedName<'_> {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if !self.0.is_empty() {
            write!(output, "{}.", self.0)?;
        }
        output.write_str(self.1)
    }
}
fn rooted(root: &str, relative: &str) -> String {
    rooted_with_metadata(
        root,
        relative,
        crate::decoder::identity::Metadata::new(None),
    )
    .expect("ordinary rooted-name construction is infallible")
}

fn rooted_with_metadata(
    root: &str,
    relative: &str,
    metadata: crate::decoder::identity::Metadata<'_>,
) -> Result<String, Error> {
    metadata.format(format_args!("{}", RootedName(root, relative)))
}

fn relative_vision_name(name: &str) -> &str {
    ["blocks.", "merger.", "deepstack_merger_list."]
        .into_iter()
        .filter_map(|marker| name.find(marker).map(|index| &name[index..]))
        .next()
        .unwrap_or(name)
}

fn gathered_positions<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    embedding: &mut B::Embedding,
    grid: &[(i32, i32, i32)],
    count: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let side = (count as f64).sqrt() as i32;
    if side * side != count {
        return Err(Error::backend("vision position table is not square"));
    }
    let mut ids = Vec::new();
    visit_gathered_patch_positions(grid.iter().copied(), side, side, |index| ids.push(index))
        .map_err(Error::backend)?;
    embedding.forward(
        &B::Tensor::from_i32_slice(&ids, &[ids.len() as i32], context)?,
        context,
    )
}

fn interpolated_positions<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    embedding: &mut B::Embedding,
    grid: &[(i32, i32, i32)],
    count: i32,
    merge: i32,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    let side = (count as f64).sqrt() as i32;
    if side * side != count {
        return Err(Error::backend("vision position table is not square"));
    }
    let samples = bilinear_interpolation_samples(
        grid,
        side,
        side,
        InterpolationMode::AlignCorners,
        PatchTraversal::MergeMajor(merge),
    )
    .map_err(Error::backend)?;
    let length = samples.len() as i32;
    project_interpolated_positions::<B>(embedding, context, |corner| {
        let ids = samples
            .iter()
            .map(|s| s.indices[corner] as i32)
            .collect::<Vec<_>>();
        let weights = samples
            .iter()
            .map(|s| s.weights[corner])
            .collect::<Vec<_>>();
        Ok((
            B::Tensor::from_i32_slice(&ids, &[length], context)?,
            B::Tensor::from_f32_slice(&weights, &[length, 1], context)?,
        ))
    })
}

// Ordinary collected and original borrowed tables feed the same ordered native
// embedding/multiply/add equation. The producer changes only host table access.
fn project_interpolated_positions<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    embedding: &mut B::Embedding,
    context: &<B::Tensor as Tensor>::Context,
    mut corner_values: impl FnMut(usize) -> Result<(B::Tensor, B::Tensor), Error>,
) -> Result<B::Tensor, Error> {
    let mut output: Option<B::Tensor> = None;
    for corner in 0..4 {
        let (ids, weights) = corner_values(corner)?;
        let part = embedding
            .forward(&ids, context)?
            .multiply(&weights, context)?;
        output = Some(match output {
            Some(current) => current.add(&part, context)?,
            None => part,
        });
    }
    output.ok_or_else(|| Error::backend("vision grid is empty"))
}

fn rotary_embeddings<T: Tensor>(
    grid: &[(i32, i32, i32)],
    config: &VisionConfig,
    context: &T::Context,
) -> Result<(T, T), Error> {
    let positions = patch_positions(grid, PatchTraversal::MergeMajor(config.spatial_merge_size))
        .map_err(Error::backend)?
        .into_iter()
        .flat_map(|[_, y, x]| [y, x])
        .collect::<Vec<_>>();
    let ids = T::from_i32_slice(&positions, &[positions.len() as i32 / 2, 2], context)?;
    let dimensions = (config.hidden_size / config.num_heads) / 2;
    multi_axis_rotary_embeddings(
        &ids,
        &MultiAxisRotarySpec {
            axes: vec![
                RotaryAxisSpec {
                    dimensions,
                    position_offset: 0,
                },
                RotaryAxisSpec {
                    dimensions,
                    position_offset: 0,
                },
            ],
            base: 10_000.0,
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
    context: &T::Context,
) -> Result<T, Error> {
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
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn window_order_has_exact_inverse() {
        let layout = window_partition(&[(1, 4, 4), (2, 2, 4)], 2, 8, 2).unwrap();
        let inverse = inverse_permutation(&layout.permutation).unwrap();
        for (execution, source) in layout.permutation.iter().enumerate() {
            assert_eq!(inverse[*source as usize], execution as i32);
        }
    }
}
