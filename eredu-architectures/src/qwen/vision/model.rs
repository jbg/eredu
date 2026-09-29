//! Shared backend-neutral Qwen vision equations.

use eredu_nn::{
    multimodal::{
        multi_axis_rotary_embeddings, project_flattened_patches, FlattenedPatchSpec,
        MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
    },
    sequence_layout::{
        attention_chunk_lengths, bilinear_interpolation_samples, inverse_permutation,
        patch_positions, validate_patch_grid, window_partition, InterpolationMode, PatchTraversal,
    },
    EmbeddingOperator, EmbeddingSpec, Error, Index, LinearOperator, LinearSpec, NeuralBackend,
    Parameter, ParameterSpec, Parameterized, SegmentedAttentionInput, Tensor,
};

use super::{VisionAttentionPolicy, VisionConfig, VisionMode};

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
    full_chunks: Vec<i32>,
    window_chunks: Vec<i32>,
    permutation: Vec<i32>,
    cosine: T,
    sine: T,
    deepstack: Vec<T>,
}

/// Semantic roots retained between shared vision execution units.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionStateTensorRole {
    /// Rotary cosine products.
    RotaryCosine,
    /// Rotary sine products.
    RotarySine,
    /// Captured features in retained capture order, not merger-bank identity.
    DeepStack(usize),
}

/// Semantic roots returned by the shared vision tower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionOutputTensorRole {
    /// Final decoder-width embeddings.
    Embeddings,
    /// Captured features in retained capture order, not merger-bank identity.
    DeepStack(usize),
}

impl<T> VisionState<T> {
    /// Full-attention contiguous segment lengths.
    pub fn full_chunks(&self) -> &[i32] {
        &self.full_chunks
    }
    /// Window-attention contiguous segment lengths.
    pub fn window_chunks(&self) -> &[i32] {
        &self.window_chunks
    }
    /// Merge-group execution permutation.
    pub fn permutation(&self) -> &[i32] {
        &self.permutation
    }
    /// Captured DeepStack features.
    pub fn deepstack_features(&self) -> &[T] {
        &self.deepstack
    }
    /// Backend tensors that must remain live while streamable blocks execute.
    pub fn retained_values(&self) -> impl Iterator<Item = &T> {
        self.storage_values().map(|(_, value)| value)
    }
    /// Named borrowed roots; combine with request/output roots before surveying
    /// to observe aliases across categories in the same index namespace.
    pub fn storage_values(&self) -> impl Iterator<Item = (VisionStateTensorRole, &T)> {
        [
            (VisionStateTensorRole::RotaryCosine, &self.cosine),
            (VisionStateTensorRole::RotarySine, &self.sine),
        ]
        .into_iter()
        .chain(
            self.deepstack
                .iter()
                .enumerate()
                .map(|(index, value)| (VisionStateTensorRole::DeepStack(index), value)),
        )
    }
    /// Replaces the backend tensors transported between component owners.
    pub fn replace_retained_values(&mut self, values: Vec<T>) -> Result<(), Error> {
        if values.len() < 2 {
            return Err(Error::backend(
                "vision continuation requires rotary cosine and sine tensors",
            ));
        }
        let mut values = values.into_iter();
        self.cosine = values.next().expect("validated cosine");
        self.sine = values.next().expect("validated sine");
        self.deepstack = values.collect();
        Ok(())
    }
    fn chunks(&self, policy: VisionAttentionPolicy) -> &[i32] {
        match policy {
            VisionAttentionPolicy::Full => &self.full_chunks,
            VisionAttentionPolicy::Windowed => &self.window_chunks,
        }
    }
}

/// Final media embeddings and selected intermediate features.
#[derive(Debug, Clone)]
pub struct VisionOutput<T> {
    /// Final `[1, media_tokens, text_hidden]` embeddings.
    pub embeddings: T,
    /// DeepStack features in retained capture order.
    pub deepstack_features: Vec<T>,
}

impl<T> VisionOutput<T> {
    /// Named output roots, borrowed without evaluating or cloning tensors.
    pub fn storage_values(&self) -> impl Iterator<Item = (VisionOutputTensorRole, &T)> {
        std::iter::once((VisionOutputTensorRole::Embeddings, &self.embeddings)).chain(
            self.deepstack_features
                .iter()
                .enumerate()
                .map(|(index, value)| (VisionOutputTensorRole::DeepStack(index), value)),
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(super) struct LayerNorm<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    weight: Parameter<B::Tensor>,
    bias: Parameter<B::Tensor>,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> LayerNorm<B> {
    const EPSILON: f32 = 1e-6;
    fn new(
        prefix: &str,
        width: i32,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let make = |field: &str| {
            Parameter::unloaded(
                ParameterSpec::trainable(format!("{prefix}.{field}")).map_err(Error::backend)?,
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
            Self::EPSILON,
            context,
        )
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(super) struct PatchEmbed<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    weight: Parameter<B::Tensor>,
    bias: Parameter<B::Tensor>,
    #[parameter(skip)]
    spec: FlattenedPatchSpec,
}

impl<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> PatchEmbed<B> {
    fn new(
        config: &VisionConfig,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let spec = FlattenedPatchSpec {
            channels: config.in_channels,
            temporal: config.temporal_patch_size,
            height: config.patch_size,
            width: config.patch_size,
            output: config.hidden_size,
        };
        Ok(Self {
            weight: Parameter::unloaded(
                ParameterSpec::trainable(rooted(root, "patch_embed.proj.weight"))
                    .map_err(Error::backend)?,
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
                ParameterSpec::trainable(rooted(root, "patch_embed.proj.bias"))
                    .map_err(Error::backend)?,
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
    let weight = format!("{prefix}.weight");
    B::linear(
        LinearSpec {
            input,
            output,
            weight: ParameterSpec::trainable(&weight).map_err(Error::backend)?,
            bias: Some(ParameterSpec::trainable(format!("{prefix}.bias")).map_err(Error::backend)?),
            format: crate::linear_format::standard_linear_format(
                &weight,
                config.linear_format(relative_vision_name(&weight)),
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
    #[parameter(skip)]
    heads: i32,
    #[parameter(skip)]
    head_dim: i32,
    #[parameter(skip)]
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
        let root = rooted(parameter_root, &format!("blocks.{layer}.attn"));
        let head_dim = config.hidden_size / config.num_heads;
        Ok(Self {
            qkv: linear::<B>(
                config,
                &format!("{root}.qkv"),
                config.hidden_size,
                3 * heads * head_dim,
                context,
            )?,
            output: linear::<B>(
                config,
                &format!("{root}.proj"),
                heads * head_dim,
                config.hidden_size,
                context,
            )?,
            heads,
            head_dim,
            scale: (head_dim as f32).sqrt().recip(),
        })
    }
    fn forward_impl(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
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
        let attention = SegmentedAttentionInput {
            queries: &query,
            keys: &key,
            values: &value,
            segment_lengths: chunks,
            scale: self.scale,
        };

        let attended = B::segmented_attention(attention, context)?;
        let attended = attended.reshape(&[sequence, self.heads * self.head_dim], context)?;

        match parallel {
            Some(parallel) => {
                B::row_parallel_linear(&mut self.output, &attended, parallel, context)
            }
            None => self.output.forward(&attended, context),
        }
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
    #[parameter(skip)]
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
        let root = rooted(parameter_root, &format!("blocks.{layer}"));
        Ok(Self {
            norm1: LayerNorm::new(&format!("{root}.norm1"), config.hidden_size, context)?,
            attention: Attention::new_with_heads(config, parameter_root, layer, heads, context)?,
            norm2: LayerNorm::new(&format!("{root}.norm2"), config.hidden_size, context)?,
            fc1: linear::<B>(
                config,
                &format!("{root}.mlp.linear_fc1"),
                config.hidden_size,
                intermediate,
                context,
            )?,
            fc2: linear::<B>(
                config,
                &format!("{root}.mlp.linear_fc2"),
                intermediate,
                config.hidden_size,
                context,
            )?,
            activation: config.hidden_act.clone(),
        })
    }
    /// Executes the shared attention/MLP residual equations.
    pub fn forward(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_impl(hidden, chunks, cos, sin, None, context)
    }

    /// Executes the shared block with local heads/channels and row reductions.
    pub fn forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_impl(hidden, chunks, cos, sin, Some(parallel), context)
    }

    fn forward_impl(
        &mut self,
        hidden: &B::Tensor,
        chunks: &[i32],
        cos: &B::Tensor,
        sin: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normed = self.norm1.forward(hidden, context)?;
        let hidden = hidden.add(
            &self
                .attention
                .forward_impl(&normed, chunks, cos, sin, parallel, context)?,
            context,
        )?;
        let normed = self.norm2.forward(&hidden, context)?;

        let projected = self.fc1.forward(&normed, context)?;
        let activated = match self.activation.as_str() {
            "silu" => B::silu(projected, context)?,
            "gelu" => B::Tensor::gelu(&projected, context)?,
            "gelu_pytorch_tanh" => B::gelu_approximate(projected, context)?,
            other => {
                return Err(Error::backend(format!(
                    "unsupported vision activation {other:?}"
                )))
            }
        };

        let projected = match parallel {
            Some(parallel) => B::row_parallel_linear(&mut self.fc2, &activated, parallel, context)?,
            None => self.fc2.forward(&activated, context)?,
        };
        hidden.add(&projected, context)
    }
}

#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub(super) struct Merger<B: NeuralBackend + eredu_nn::DistributedNeuralBackend> {
    pub(super) norm: LayerNorm<B>,
    pub(super) fc1: B::Linear,
    pub(super) fc2: B::Linear,
    #[parameter(skip)]
    unit: i32,
    #[parameter(skip)]
    width: i32,
    #[parameter(skip)]
    postshuffle: bool,
    #[parameter(skip)]
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
        let unit = config.spatial_merge_size * config.spatial_merge_size;
        let width = config.hidden_size * unit;
        Ok(Self {
            norm: LayerNorm::new(
                &format!("{prefix}.norm"),
                if postshuffle {
                    width
                } else {
                    config.hidden_size
                },
                context,
            )?,
            fc1: linear::<B>(
                config,
                &format!("{prefix}.linear_fc1"),
                width,
                intermediate,
                context,
            )?,
            fc2: linear::<B>(
                config,
                &format!("{prefix}.linear_fc2"),
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
    fn forward_impl(
        &mut self,
        hidden: &B::Tensor,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if hidden.dim(0) % self.unit != 0 {
            return Err(Error::backend(
                "vision merge geometry does not divide sequence",
            ));
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

        match parallel {
            Some(parallel) => B::row_parallel_linear(&mut self.fc2, &hidden, parallel, context),
            None => self.fc2.forward(&hidden, context),
        }
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
    #[parameter(skip)]
    config: VisionConfig,
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
        let width = config.hidden_size * config.spatial_merge_size * config.spatial_merge_size;
        let intermediates = vec![width; config.deepstack_layer_count() + 1];
        Self::new_parallel_with_root(config, parameter_root, &intermediates, context)
    }

    /// Builds pinned modules with rank-local merger intermediate widths.
    pub fn new_parallel_with_root(
        config: VisionConfig,
        parameter_root: &str,
        merger_intermediates: &[i32],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        config.validate().map_err(Error::backend)?;
        if merger_intermediates.len() != config.deepstack_layer_count() + 1
            || merger_intermediates.iter().any(|width| *width <= 0)
        {
            return Err(Error::backend(
                "vision merger local-width plan is incomplete",
            ));
        }
        let position = B::embedding(
            EmbeddingSpec {
                vocabulary: config.num_position_embeddings,
                dimensions: config.hidden_size,
                weight: ParameterSpec::trainable(rooted(parameter_root, "pos_embed.weight"))
                    .map_err(Error::backend)?,
                format: crate::linear_format::standard_linear_format(
                    &rooted(parameter_root, "pos_embed.weight"),
                    eredu_checkpoint::LinearFormat::Dense,
                )?,
            },
            context,
        )?;
        let patch = PatchEmbed::new(&config, parameter_root, context)?;
        let merger = Merger::new_with_intermediate(
            &config,
            &rooted(parameter_root, "merger"),
            false,
            config.mode == VisionMode::WindowScheduled,
            merger_intermediates[0],
            context,
        )?;
        let deepstack_mergers = (0..config.deepstack_layer_count())
            .map(|i| {
                Merger::new_with_intermediate(
                    &config,
                    &rooted(parameter_root, &format!("deepstack_merger_list.{i}")),
                    true,
                    false,
                    merger_intermediates[i + 1],
                    context,
                )
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            position,
            patch,
            merger,
            deepstack_mergers,
            config,
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
        let state = self.prepare_state(input.grid, context)?;
        let unit = self.config.spatial_merge_size * self.config.spatial_merge_size;
        let indexes = B::Tensor::from_i32_slice(
            &state.permutation,
            &[state.permutation.len() as i32],
            context,
        )?;
        let sequence = hidden.dim(0);
        hidden = hidden
            .reshape(&[-1, unit, hidden.dim(1)], context)?
            .take_axis(&indexes, 0, context)?
            .reshape(&[sequence, -1], context)?;
        Ok((hidden, state))
    }

    /// Builds rotary products and attention geometry without reading any model
    /// parameters. Receiving pipeline owners pair this with the upstream activation.
    pub fn prepare_state(
        &self,
        grid: &[(i32, i32, i32)],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionState<B::Tensor>, Error> {
        validate_patch_grid(grid, self.config.spatial_merge_size, None).map_err(Error::backend)?;
        let full_chunks = attention_chunk_lengths(grid).map_err(Error::backend)?;
        let sequence = full_chunks
            .iter()
            .try_fold(0_i32, |total, length| total.checked_add(*length))
            .ok_or_else(|| Error::backend("vision continuation sequence exceeds i32"))?;
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
        let indexes =
            B::Tensor::from_i32_slice(&permutation, &[permutation.len() as i32], context)?;
        let reorder = |value: B::Tensor| {
            value
                .reshape(&[-1, unit, value.dim(1)], context)?
                .take_axis(&indexes, 0, context)?
                .reshape(&[sequence, -1], context)
        };
        let (cosine, sine) = rotary_embeddings::<B::Tensor>(grid, &self.config, context)?;
        Ok(VisionState {
            full_chunks,
            window_chunks,
            permutation,
            cosine: reorder(cosine)?,
            sine: reorder(sine)?,
            deepstack: Vec::new(),
        })
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
        self.forward_block_impl(block, layer, hidden, state, None, context)
    }

    /// Executes one scheduled rank-local block and captures selected DeepStack output.
    pub fn forward_block_parallel(
        &mut self,
        block: &mut VisionBlock<B>,
        layer: usize,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_block_impl(block, layer, hidden, state, Some(parallel), context)
    }

    fn forward_block_impl(
        &mut self,
        block: &mut VisionBlock<B>,
        layer: usize,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let policy = *self
            .config
            .layer_policy(layer)
            .ok_or_else(|| Error::backend(format!("vision schedule has no block {layer}")))?;
        let hidden = block.forward_impl(
            hidden,
            state.chunks(policy.attention),
            &state.cosine,
            &state.sine,
            parallel,
            context,
        )?;
        if let Some(index) = policy.deepstack_merger {
            let merger = self
                .deepstack_mergers
                .get_mut(index as usize)
                .ok_or_else(|| Error::backend("missing DeepStack merger"))?;
            let feature = merger
                .forward_impl(&hidden, parallel, context)?
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
        self.finish_impl(hidden, state, None, context)
    }
    /// Completes a rank-local tower and reduces the final merger exactly once.
    pub fn finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionOutput<B::Tensor>, Error> {
        self.finish_impl(hidden, state, Some(parallel), context)
    }

    fn finish_impl(
        &mut self,
        hidden: &B::Tensor,
        state: &mut VisionState<B::Tensor>,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<VisionOutput<B::Tensor>, Error> {
        let merged = self.merger.forward_impl(hidden, parallel, context)?;
        let inverse = inverse_permutation(&state.permutation).map_err(Error::backend)?;
        let inverse = B::Tensor::from_i32_slice(&inverse, &[inverse.len() as i32], context)?;
        let embeddings = merged
            .take_axis(&inverse, 0, context)?
            .expand_dims(0, context)?;
        Ok(VisionOutput {
            embeddings,
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
        let blocks = (0..config.layer_count())
            .map(|i| VisionBlock::new_with_root(&config, parameter_root, i, context))
            .collect::<Result<_, _>>()?;
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

fn rooted(root: &str, relative: &str) -> String {
    if root.is_empty() {
        relative.to_owned()
    } else {
        format!("{root}.{relative}")
    }
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
    for &(time, height, width) in grid {
        if height > side || width > side {
            return Err(Error::backend("vision grid exceeds position table"));
        }
        for _ in 0..time {
            for y in 0..height {
                for x in 0..width {
                    ids.push(y * side + x);
                }
            }
        }
    }
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
    let mut output: Option<B::Tensor> = None;
    for corner in 0..4 {
        let ids = samples
            .iter()
            .map(|s| s.indices[corner] as i32)
            .collect::<Vec<_>>();
        let weights = samples
            .iter()
            .map(|s| s.weights[corner])
            .collect::<Vec<_>>();
        let ids = B::Tensor::from_i32_slice(&ids, &[length], context)?;
        let weights = B::Tensor::from_f32_slice(&weights, &[length, 1], context)?;
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
    let spec = config.rotary_spec()?;
    let patches =
        validate_patch_grid(grid, config.spatial_merge_size, None).map_err(Error::backend)?;
    let rows = i32::try_from(patches).map_err(Error::backend)?;
    let positions = patch_positions(grid, PatchTraversal::MergeMajor(config.spatial_merge_size))
        .map_err(Error::backend)?
        .into_iter()
        .flat_map(|[_, y, x]| [y, x])
        .collect::<Vec<_>>();
    let ids = T::from_i32_slice(&positions, &[rows, 2], context)?;
    multi_axis_rotary_embeddings(&ids, &spec, context)
}

impl VisionConfig {
    /// Exact spatial rotary policy shared by execution and resource descriptions.
    pub fn rotary_spec(&self) -> Result<MultiAxisRotarySpec, Error> {
        self.validate().map_err(Error::backend)?;
        let dimensions = (self.hidden_size / self.num_heads) / 2;
        let spec = MultiAxisRotarySpec {
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
        };
        if spec.dimensions()? != self.hidden_size / self.num_heads {
            return Err(Error::backend(
                "vision rotary axes do not cover the head width",
            ));
        }
        Ok(spec)
    }
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
