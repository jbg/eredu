//! Shared backend-neutral decoder mechanics.
//!
//! Architecture families retain configuration, checkpoint naming, identity, and
//! policy while reusing these statically dispatched decoder operations.

use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::cache::LayerCachePolicy;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, EmbeddingOperator, EmbeddingSpec, Error, LinearOperator, LinearSpec,
    NeuralBackend, NormalizationOperator, NormalizationSpec, ParameterSpec, RotaryOperator,
    RotaryPosition, RotarySpec, RotarySubspace, SwiGluLimit, Tensor,
};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_projection_group,
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, MemberSharding, ParallelPlanError,
    ParameterGroupSpec, ParameterRole, ProjectionSharding, StateLayout,
};

/// Geometry and policy required by the shared decoder mechanics.
pub trait Config: 'static {
    /// Stable normalized model identity.
    fn model_identity(&self) -> &str;
    /// Canonical parameter namespace for this decoder body.
    fn parameter_root(&self) -> &str {
        "model"
    }
    /// Validates architecture-owned configuration policy.
    fn validate_config(&self) -> Result<(), Error>;
    /// Transformer hidden size.
    fn hidden_size(&self) -> i32;
    /// Number of decoder layers.
    fn num_hidden_layers(&self) -> i32;
    /// SwiGLU intermediate width.
    fn intermediate_size(&self) -> i32;
    /// Number of query heads.
    fn num_attention_heads(&self) -> i32;
    /// Number of key/value heads.
    fn num_key_value_heads(&self) -> i32;
    /// Per-head width.
    fn head_dim(&self) -> i32;
    /// RMSNorm epsilon.
    fn rms_norm_epsilon(&self) -> f32;
    /// Vocabulary size.
    fn vocabulary_size(&self) -> i32;
    /// Whether one attention projection owns a learned bias.
    fn attention_bias(&self, projection: AttentionProjection) -> bool;
    /// Optional per-head Q/K RMS-normalization epsilon.
    fn query_key_norm_epsilon(&self) -> Option<f32> {
        None
    }
    /// Whether projections own MLP biases.
    fn mlp_bias(&self) -> bool;
    /// Optional bound applied before each dense SwiGLU product.
    fn swiglu_limit(&self) -> Option<SwiGluLimit> {
        None
    }
    /// Whether the language-model head is tied to input embeddings.
    fn tie_word_embeddings(&self) -> bool;
    /// Exact per-layer attention policy.
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy>;
    /// Physical encoding selected for one canonical checkpoint parameter.
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization>;
    /// Complete rotary-position construction specification.
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec<'_>;
}

/// Semantic attention projection selected by architecture policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AttentionProjection {
    /// Query projection.
    Query,
    /// Key projection.
    Key,
    /// Value projection.
    Value,
    /// Output projection.
    Output,
}

/// Derives the canonical backend-neutral cache layout for this decoder.
pub fn cache_layout<C: Config>(config: &C) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    cache_layout_with_key_value_heads(
        config,
        std::iter::repeat_n(
            config.num_key_value_heads(),
            config.attention_schedule().len(),
        ),
    )
}

/// Declares the complete mutable-state geometry consumed by either resident
/// or bounded-residency execution.
pub fn state_layout<C: Config>(config: &C) -> Result<StateLayout, Error> {
    StateLayout::new(cache_layout(config)?).map_err(Error::backend)
}

/// Declares the complete mutable-state geometry consumed by resident or bounded execution.
pub fn cache_layout_with_key_value_heads<C: Config>(
    config: &C,
    key_value_heads: impl IntoIterator<Item = i32>,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    let key_value_heads = key_value_heads.into_iter().collect::<Vec<_>>();
    if key_value_heads.len() != layers {
        return Err(Error::backend(format!(
            "decoder cache geometry has {} layers, expected {layers}",
            key_value_heads.len()
        )));
    }
    let policies = config
        .attention_schedule()
        .iter()
        .zip(key_value_heads)
        .map(|(attention, key_value_heads)| {
            LayerCachePolicy::key_value(*attention, key_value_heads, config.head_dim())
                .map_err(Error::backend)
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies).map_err(Error::backend)
}

/// Creates one concrete backend cache per decoder layer from the neutral policy.
///
/// Cache construction is outside inference. The closure is monomorphized and
/// returns the backend's native cache type without boxing or tensor conversion.
pub fn create_caches<C: Config, K>(
    config: &C,
    mut create: impl FnMut(usize, Option<i32>) -> K,
) -> Result<Vec<Option<K>>, Error> {
    validate_schedule(config)?;
    config
        .attention_schedule()
        .iter()
        .enumerate()
        .map(|(layer, policy)| {
            let window = policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(Error::backend)?;
            Ok(Some(create(layer, window)))
        })
        .collect()
}

/// Validates that concrete backend caches implement the architecture's policy.
pub fn validate_caches<B, C, K>(config: &C, caches: &[Option<K>]) -> Result<(), Error>
where
    B: NeuralBackend,
    C: Config,
    K: AttentionCache<B::Tensor>,
{
    validate_schedule(config)?;
    if caches.len() != config.attention_schedule().len() {
        return Err(Error::backend(format!(
            "decoder cache has {} layers, expected {}",
            caches.len(),
            config.attention_schedule().len()
        )));
    }
    for (layer, (cache, policy)) in caches
        .iter()
        .zip(config.attention_schedule().iter())
        .enumerate()
    {
        let cache = cache
            .as_ref()
            .ok_or_else(|| Error::backend(format!("decoder cache is missing layer {layer}")))?;
        let expected = policy
            .window()
            .map(|window| i32::try_from(window.get()))
            .transpose()
            .map_err(Error::backend)?;
        if cache.max_size() != expected {
            return Err(Error::backend(format!(
                "decoder cache policy mismatch at layer {layer}: expected {policy:?}, cache window is {:?}",
                cache.max_size()
            )));
        }
    }
    Ok(())
}

fn validate_schedule<C: Config>(config: &C) -> Result<(), Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    if config.attention_schedule().len() != layers {
        return Err(Error::backend(format!(
            "decoder attention schedule has {} layers, expected {layers}",
            config.attention_schedule().len()
        )));
    }
    Ok(())
}

/// Hidden-state input for one decoder block.
pub struct AttentionInput<'a, T, C> {
    /// Hidden states shaped `[batch, sequence, hidden]`.
    pub hidden: &'a T,
    /// Optional additive or boolean attention mask.
    pub mask: Option<&'a T>,
    /// Optional mutable layer cache.
    pub cache: Option<&'a mut C>,
    /// Whether the block may select its mask-free sliding prefill kernel.
    pub allow_sliding_prefill: bool,
    /// Optional caller-provided explicit rotary position data.
    pub rotary_position: Option<RotaryPosition<'a, T>>,
}

/// Shared grouped-query self attention.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Attention<B: NeuralBackend> {
    /// Number of query heads.
    #[parameter(skip)]
    pub query_heads: i32,
    /// Number of key/value heads.
    #[parameter(skip)]
    pub key_value_heads: i32,
    /// Inverse square-root head scaling.
    #[parameter(skip)]
    pub scale: f32,
    /// Query projection.
    pub query: B::Linear,
    /// Key projection.
    pub key: B::Linear,
    /// Value projection.
    pub value: B::Linear,
    /// Output projection.
    pub output: B::Linear,
    /// Optional per-head query normalization.
    pub query_norm: Option<B::Normalization>,
    /// Optional per-head key normalization.
    pub key_norm: Option<B::Normalization>,
    /// Rotary-position operator.
    pub rotary: B::Rotary,
    /// Layer-local sliding window.
    #[parameter(skip)]
    pub sliding_window: Option<i32>,
}

struct AttentionProjections<T> {
    queries: T,
    keys: T,
    values: T,
    batch: i32,
    sequence: i32,
}

impl<B: NeuralBackend> Attention<B> {
    /// Builds unloaded grouped-query attention for one global layer.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{}.layers.{layer}.self_attn", config.parameter_root());
        let hidden = config.hidden_size();
        let head = config.head_dim();
        let query_heads = config.num_attention_heads();
        let key_value_heads = config.num_key_value_heads();
        let linear = |field: &str, input, output, bias: bool| {
            let weight_name = format!("{prefix}.{field}.weight");
            let bias = bias
                .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                .transpose()
                .map_err(Error::backend)?;
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight_name).map_err(Error::backend)?,
                    bias,
                    format: config.weight_quantization(&weight_name).into(),
                },
                context,
            )
        };
        let policy = config.attention_schedule().get(layer).ok_or_else(|| {
            Error::backend(format!(
                "decoder attention schedule has no policy for layer {layer}"
            ))
        })?;
        Ok(Self {
            query_heads,
            key_value_heads,
            scale: (head as f32).sqrt().recip(),
            query: linear(
                "q_proj",
                hidden,
                query_heads * head,
                config.attention_bias(AttentionProjection::Query),
            )?,
            key: linear(
                "k_proj",
                hidden,
                key_value_heads * head,
                config.attention_bias(AttentionProjection::Key),
            )?,
            value: linear(
                "v_proj",
                hidden,
                key_value_heads * head,
                config.attention_bias(AttentionProjection::Value),
            )?,
            output: linear(
                "o_proj",
                query_heads * head,
                hidden,
                config.attention_bias(AttentionProjection::Output),
            )?,
            query_norm: config
                .query_key_norm_epsilon()
                .map(|epsilon| {
                    B::rms_norm(
                        NormalizationSpec {
                            dimensions: head,
                            epsilon,
                            weight: ParameterSpec::trainable(format!("{prefix}.q_norm.weight"))
                                .map_err(Error::backend)?,
                        },
                        context,
                    )
                })
                .transpose()?,
            key_norm: config
                .query_key_norm_epsilon()
                .map(|epsilon| {
                    B::rms_norm(
                        NormalizationSpec {
                            dimensions: head,
                            epsilon,
                            weight: ParameterSpec::trainable(format!("{prefix}.k_norm.weight"))
                                .map_err(Error::backend)?,
                        },
                        context,
                    )
                })
                .transpose()?,
            rotary: B::rotary(config.rotary_spec(head), context)?,
            sliding_window: policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(Error::backend)?,
        })
    }

    fn projections(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<AttentionProjections<B::Tensor>, Error> {
        let batch = hidden.dim(0);
        let sequence = hidden.dim(1);
        let reshape = |tensor: B::Tensor, heads| {
            tensor
                .reshape(&[batch, sequence, heads, -1], context)?
                .transpose_axes(&[0, 2, 1, 3], context)
        };
        let mut queries = reshape(self.query.forward(hidden, context)?, self.query_heads)?;
        if let Some(norm) = &mut self.query_norm {
            queries = norm.forward(&queries, context)?;
        }
        let mut keys = reshape(self.key.forward(hidden, context)?, self.key_value_heads)?;
        if let Some(norm) = &mut self.key_norm {
            keys = norm.forward(&keys, context)?;
        }
        let values = reshape(self.value.forward(hidden, context)?, self.key_value_heads)?;
        Ok(AttentionProjections {
            queries,
            keys,
            values,
            batch,
            sequence,
        })
    }

    fn attend<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        mut cache: Option<&mut C>,
        allow_sliding_prefill: bool,
        rotary_position: Option<RotaryPosition<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let AttentionProjections {
            queries,
            keys,
            values,
            batch,
            sequence,
        } = self.projections(hidden, context)?;
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let position = rotary_position.unwrap_or(RotaryPosition::Offset(offset));
        let queries =
            self.rotary
                .forward_subspace(&queries, RotarySubspace::Full, position, context)?;
        let keys = self
            .rotary
            .forward_subspace(&keys, RotarySubspace::Full, position, context)?;
        let (keys, values) = match cache.as_mut() {
            Some(cache) => cache.update_for_attention(keys, values, context)?,
            None => (keys, values),
        };
        if let Some(window) = self
            .sliding_window
            .filter(|_| allow_sliding_prefill && sequence > 1)
        {
            return B::sliding_window_attention(
                queries, keys, values, self.scale, window, offset, context,
            );
        }
        let attended = if let Some(cache) = cache {
            cache.attention(queries, keys, values, self.scale, mask, context)?
        } else {
            B::attention(queries, keys, values, self.scale, mask, context)?
        };
        attended
            .transpose_axes(&[0, 2, 1, 3], context)?
            .reshape(&[batch, sequence, -1], context)
    }

    fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(
            input.hidden,
            input.mask,
            input.cache,
            input.allow_sliding_prefill,
            input.rotary_position,
            context,
        )?;
        self.output.forward(&attended, context)
    }

    fn forward_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let attended = self.attend(
            input.hidden,
            input.mask,
            input.cache,
            input.allow_sliding_prefill,
            input.rotary_position,
            context,
        )?;
        B::row_parallel_linear(&mut self.output, &attended, parallel, context)
    }
}

/// Shared dense SwiGLU feed-forward network.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: NeuralBackend> {
    /// Gating projection.
    pub gate: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    /// Down projection.
    pub down: B::Linear,
    /// Optional shared pre-activation bound.
    #[parameter(skip)]
    pub limit: Option<SwiGluLimit>,
}

impl<B: NeuralBackend> Mlp<B> {
    /// Builds an unloaded dense SwiGLU network for one global layer.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("{}.layers.{layer}.mlp", config.parameter_root());
        let build = |field: &str, input, output| {
            let weight_name = format!("{prefix}.{field}.weight");
            let bias = config
                .mlp_bias()
                .then(|| ParameterSpec::trainable(format!("{prefix}.{field}.bias")))
                .transpose()
                .map_err(Error::backend)?;
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: ParameterSpec::trainable(&weight_name).map_err(Error::backend)?,
                    bias,
                    format: config.weight_quantization(&weight_name).into(),
                },
                context,
            )
        };
        Ok(Self {
            gate: build(
                "gate_proj",
                config.hidden_size(),
                config.intermediate_size(),
            )?,
            up: build("up_proj", config.hidden_size(), config.intermediate_size())?,
            down: build(
                "down_proj",
                config.intermediate_size(),
                config.hidden_size(),
            )?,
            limit: config.swiglu_limit(),
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        B::swiglu(gate, up, self.limit, context)
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        self.down.forward(&hidden, context)
    }

    fn forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.hidden(input, context)?;
        B::row_parallel_linear(&mut self.down, &hidden, parallel, context)
    }
}

/// Feed-forward policy executed by the shared residual decoder block.
pub trait FeedForwardOperator<B: NeuralBackend>: eredu_nn::Parameterized<B::Tensor> {
    /// Executes replicated feed-forward computation.
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;

    /// Executes tensor-parallel feed-forward computation.
    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>;
}

impl<B: NeuralBackend> FeedForwardOperator<B> for Mlp<B> {
    fn forward_feed_forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward(input, context)
    }

    fn forward_feed_forward_parallel(
        &mut self,
        input: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_parallel(input, parallel, context)
    }
}

/// One RMS-pre-norm residual decoder block.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlock<B: NeuralBackend, F = Mlp<B>> {
    /// Self-attention operator.
    pub self_attention: Attention<B>,
    /// Feed-forward operator.
    pub mlp: F,
    /// Pre-attention RMSNorm.
    pub input_norm: B::Normalization,
    /// Pre-MLP RMSNorm.
    pub post_attention_norm: B::Normalization,
}

impl<B: NeuralBackend> TransformerBlock<B, Mlp<B>> {
    /// Builds an unloaded block for one global layer index.
    pub fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            self_attention: Attention::new(config, layer, context)?,
            mlp: Mlp::new(config, layer, context)?,
            input_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: config.hidden_size(),
                    epsilon: config.rms_norm_epsilon(),
                    weight: ParameterSpec::trainable(format!(
                        "{}.layers.{layer}.input_layernorm.weight",
                        config.parameter_root()
                    ))
                    .map_err(Error::backend)?,
                },
                context,
            )?,
            post_attention_norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: config.hidden_size(),
                    epsilon: config.rms_norm_epsilon(),
                    weight: ParameterSpec::trainable(format!(
                        "{}.layers.{layer}.post_attention_layernorm.weight",
                        config.parameter_root()
                    ))
                    .map_err(Error::backend)?,
                },
                context,
            )?,
        })
    }
}

impl<B, F> TransformerBlock<B, F>
where
    B: NeuralBackend,
    F: FeedForwardOperator<B>,
{
    /// Executes this block with replicated projections.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = self.mlp.forward_feed_forward(&normalized, context)?;
        hidden.add(&mlp, context)
    }

    /// Executes attention and residuals while delegating feed-forward execution.
    pub fn forward_with_feed_forward<C, H>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        H: FnOnce(&mut F, &B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = feed_forward(&mut self.mlp, &normalized, context)?;
        hidden.add(&mlp, context)
    }

    /// Executes a block with rank-local column projections and reduced row projections.
    pub fn forward_tensor_parallel<C: AttentionCache<B::Tensor>>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward_parallel(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            parallel,
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = self
            .mlp
            .forward_feed_forward_parallel(&normalized, parallel, context)?;
        hidden.add(&mlp, context)
    }

    /// Executes tensor-parallel attention and residuals with delegated feed-forward execution.
    pub fn forward_tensor_parallel_with_feed_forward<C, H>(
        &mut self,
        input: AttentionInput<'_, B::Tensor, C>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        C: AttentionCache<B::Tensor>,
        H: FnOnce(&mut F, &B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input.hidden, context)?;
        let attention = self.self_attention.forward_parallel(
            AttentionInput {
                hidden: &normalized,
                mask: input.mask,
                cache: input.cache,
                allow_sliding_prefill: input.allow_sliding_prefill,
                rotary_position: input.rotary_position,
            },
            parallel,
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = feed_forward(&mut self.mlp, &normalized, context)?;
        hidden.add(&mlp, context)
    }
}

/// Declares attention and normalization groups shared by dense and routed blocks.
pub fn block_common_parallel_parameter_groups<B: NeuralBackend, F>(
    block: &TransformerBlock<B, F>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}", config.parameter_root());
    let query_heads = usize::try_from(config.num_attention_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder query-head count exceeds usize".into())
    })?;
    let key_value_heads = usize::try_from(config.num_key_value_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder key/value-head count exceeds usize".into())
    })?;
    let head_dimension = usize::try_from(config.head_dim()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder head dimension exceeds usize".into())
    })?;
    if head_dimension == 0 || key_value_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "decoder attention geometry q={query_heads}, kv={key_value_heads}, dim={head_dimension} does not form positive integral GQA groups"
        )));
    }
    let group_width = (query_heads / key_value_heads)
        .checked_mul(head_dimension)
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup("decoder GQA group width overflowed".into())
        })?;
    let attention_alignment = config
        .weight_quantization(&format!("{prefix}.self_attn.o_proj.weight"))
        .map_or(Ok(1), |quantization| {
            usize::try_from(quantization.group_size()).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "decoder output-projection quantization group exceeds usize".into(),
                )
            })
        })?;
    let attention_units = aligned_partition_units(
        &format!("{prefix}.self_attn"),
        key_value_heads,
        group_width,
        attention_alignment,
    )?;
    let attention = partitioned_projection_group::<B::Tensor, B::Linear>(
        format!("{prefix}.self_attn.projections"),
        ParameterRole::AttentionHeads,
        &[
            (&block.self_attention.query, ProjectionSharding::Column),
            (&block.self_attention.key, ProjectionSharding::Column),
            (&block.self_attention.value, ProjectionSharding::Column),
            (&block.self_attention.output, ProjectionSharding::Row),
        ],
        attention_units,
    )?;

    let input_norm = module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.input_layernorm"),
        ParameterRole::Replicated,
        &block.input_norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?;
    let post_attention_norm = module_parameter_group::<B::Tensor, _>(
        format!("{prefix}.post_attention_layernorm"),
        ParameterRole::Replicated,
        &block.post_attention_norm,
        |_, _| Ok(MemberSharding::Replicated),
    )?;
    let mut groups = vec![attention];
    if let Some(norm) = &block.self_attention.query_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{prefix}.self_attn.q_norm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    if let Some(norm) = &block.self_attention.key_norm {
        groups.push(module_parameter_group::<B::Tensor, _>(
            format!("{prefix}.self_attn.k_norm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?);
    }
    groups.extend([input_norm, post_attention_norm]);
    Ok(groups)
}

/// Declares the dense SwiGLU placement group shared by dense decoder families.
pub fn dense_mlp_parallel_parameter_group<B: NeuralBackend>(
    mlp: &Mlp<B>,
    config: &impl Config,
    layer: usize,
) -> Result<ParameterGroupSpec, ParallelPlanError> {
    let prefix = format!("{}.layers.{layer}.mlp", config.parameter_root());
    let intermediate = usize::try_from(config.intermediate_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("decoder feed-forward width exceeds usize".into())
    })?;
    let alignment = config
        .weight_quantization(&format!("{prefix}.down_proj.weight"))
        .map_or(Ok(1), |quantization| {
            usize::try_from(quantization.group_size()).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "decoder down-projection quantization group exceeds usize".into(),
                )
            })
        })?;
    let units = aligned_partition_units(&prefix, intermediate, 1, alignment)?;
    partitioned_projection_group::<B::Tensor, B::Linear>(
        format!("{prefix}.projections"),
        ParameterRole::FeedForwardIntermediate,
        &[
            (&mlp.gate, ProjectionSharding::Column),
            (&mlp.up, ProjectionSharding::Column),
            (&mlp.down, ProjectionSharding::Row),
        ],
        units,
    )
}

/// Declares every rank-local placement group for one dense shared decoder block.
pub fn layer_parallel_parameter_groups<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = block_common_parallel_parameter_groups(block, config, layer)?;
    groups.push(dense_mlp_parallel_parameter_group(
        &block.mlp, config, layer,
    )?);
    Ok(groups)
}

/// Derives the rank-local construction geometry of one tensor-parallel block
/// from the neutral placement layout.
pub fn static_parallel_parameter_groups<B: NeuralBackend>(
    embeddings: &B::Embedding,
    norm: &B::Normalization,
    head: Option<&B::Linear>,
    parameter_root: &str,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            format!("{parameter_root}.embed_tokens"),
            ParameterRole::Vocabulary,
            embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "decoder embedding parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            format!("{parameter_root}.norm"),
            ParameterRole::Replicated,
            norm,
            |_, _| Ok(MemberSharding::Replicated),
        )?,
    ];
    if let Some(head) = head {
        groups.push(module_parameter_group::<B::Tensor, _>(
            "lm_head",
            ParameterRole::Vocabulary,
            head,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "decoder language-model head parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

/// Shared transformer body without its language-model head.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Decoder<B: NeuralBackend, F = Mlp<B>> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Decoder blocks.
    pub layers: Vec<TransformerBlock<B, F>>,
    /// Final RMSNorm.
    pub norm: B::Normalization,
}

impl<B: NeuralBackend> Decoder<B, Mlp<B>> {
    /// Builds an unloaded decoder.
    pub fn new<C: Config>(
        config: &C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let layers = (0..config.num_hidden_layers() as usize)
            .map(|layer| TransformerBlock::new(config, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embeddings: B::embedding(
                EmbeddingSpec {
                    vocabulary: config.vocabulary_size(),
                    dimensions: config.hidden_size(),
                    weight: ParameterSpec::trainable(format!(
                        "{}.embed_tokens.weight",
                        config.parameter_root()
                    ))
                    .map_err(Error::backend)?,
                    quantization: config.weight_quantization(&format!(
                        "{}.embed_tokens.weight",
                        config.parameter_root()
                    )),
                },
                context,
            )?,
            layers,
            norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: config.hidden_size(),
                    epsilon: config.rms_norm_epsilon(),
                    weight: ParameterSpec::trainable(format!(
                        "{}.norm.weight",
                        config.parameter_root()
                    ))
                    .map_err(Error::backend)?,
                },
                context,
            )?,
        })
    }
}

impl<B, F> Decoder<B, F>
where
    B: NeuralBackend,
    F: FeedForwardOperator<B>,
{
    /// Builds an unloaded decoder with an architecture-selected block factory.
    pub fn new_with_factory<C, P>(
        config: &C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error>
    where
        C: Config,
        P: BlockFactory<B, C, FeedForward = F>,
    {
        let layers = (0..config.num_hidden_layers() as usize)
            .map(|layer| P::build(config, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
        let static_modules: StaticModules<B> = StaticModules::new(config, context)?;
        Ok(Self {
            embeddings: static_modules.embeddings,
            layers,
            norm: static_modules.norm,
        })
    }

    /// Executes the transformer body with a caller-prepared mask and per-layer caches.
    pub fn forward<C: AttentionCache<B::Tensor>>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        caches: &mut [Option<C>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::backend(format!(
                "decoder cache has {} layers, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        let hidden = self.embed(tokens, context)?;
        self.forward_embedded(hidden, mask, mask.is_none(), caches, context)
    }

    /// Embeds token ids without materializing them outside the backend graph.
    pub fn embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.embeddings.forward(tokens, context)
    }

    /// Executes decoder layers from already embedded hidden states.
    pub fn forward_embedded<C: AttentionCache<B::Tensor>>(
        &mut self,
        hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        allow_sliding_prefill: bool,
        caches: &mut [Option<C>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_embedded_with_rotary(
            hidden,
            mask,
            allow_sliding_prefill,
            None,
            caches,
            context,
        )
    }

    /// Executes decoder layers with caller-provided explicit rotary embeddings.
    pub fn forward_embedded_with_rotary<C: AttentionCache<B::Tensor>>(
        &mut self,
        mut hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        allow_sliding_prefill: bool,
        rotary_embeddings: Option<(&B::Tensor, &B::Tensor)>,
        caches: &mut [Option<C>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::backend(format!(
                "decoder cache has {} layers, expected {}",
                caches.len(),
                self.layers.len()
            )));
        }
        for (layer, cache) in self.layers.iter_mut().zip(caches) {
            hidden = layer.forward(
                AttentionInput {
                    hidden: &hidden,
                    mask,
                    cache: cache.as_mut(),
                    allow_sliding_prefill,
                    rotary_position: rotary_embeddings
                        .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
                },
                context,
            )?;
        }
        self.norm.forward(&hidden, context)
    }
}

/// Complete shared causal language model.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Model<B: NeuralBackend> {
    /// Transformer body.
    pub decoder: Decoder<B>,
    /// Optional untied output projection.
    pub lm_head: Option<B::Linear>,
}

impl<B: NeuralBackend> Model<B> {
    /// Builds an unloaded model.
    pub fn new<C: Config>(
        config: &C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let decoder = Decoder::new(config, context)?;
        let lm_head = if config.tie_word_embeddings() {
            None
        } else {
            Some(B::linear(
                LinearSpec {
                    input: config.hidden_size(),
                    output: config.vocabulary_size(),
                    weight: ParameterSpec::trainable("lm_head.weight").map_err(Error::backend)?,
                    bias: None,
                    format: config.weight_quantization("lm_head.weight").into(),
                },
                context,
            )?)
        };
        Ok(Self { decoder, lm_head })
    }

    /// Projects normalized hidden states to vocabulary logits.
    pub fn logits(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        match &mut self.lm_head {
            Some(head) => head.forward(hidden, context),
            None => self.decoder.embeddings.as_linear(hidden, context),
        }
    }

    /// Executes token embedding, cache-aware masking, the decoder, and the
    /// vocabulary projection without leaving backend-native tensor storage.
    pub fn forward<C, K>(
        &mut self,
        config: &C,
        tokens: &B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        caches: &mut [Option<K>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: Config,
        K: AttentionCache<B::Tensor>,
    {
        validate_caches::<B, _, _>(config, caches)?;
        let hidden = self.decoder.embed(tokens, context)?;
        let sequence = hidden.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if sequence > 1 {
            let cache = caches
                .first()
                .and_then(Option::as_ref)
                .expect("validated decoder cache has a first layer");
            let window = cache.max_size();
            let offset = window.map_or(cache.offset(), |window| cache.offset().min(window));
            Some(B::causal_mask(sequence, offset, window, context)?)
        } else {
            None
        };
        let hidden = self.decoder.forward_embedded(
            hidden,
            mask.as_ref(),
            supplied_mask.is_none(),
            caches,
            context,
        )?;
        self.logits(&hidden, context)
    }
}

/// Pinned modules shared by resident and bounded-residency execution.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: NeuralBackend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Final RMSNorm.
    pub norm: B::Normalization,
    /// Optional untied vocabulary projection.
    pub lm_head: Option<B::Linear>,
}

/// Architecture-supplied identities and geometry for the shared pinned text
/// modules.
#[derive(Debug, Clone)]
pub struct StaticModuleSpec {
    /// Token embedding parameter identity.
    pub embedding_weight: String,
    /// Final normalization parameter identity.
    pub normalization_weight: String,
    /// Untied output-head parameter identity.
    pub head_weight: String,
    /// Vocabulary row count.
    pub vocabulary: i32,
    /// Hidden width.
    pub hidden_size: i32,
    /// Final RMS normalization epsilon.
    pub normalization_epsilon: f32,
    /// Packed embedding format, when supported by the general embedding operator.
    pub embedding_quantization: Option<WeightQuantization>,
    /// Complete output-head physical encoding.
    pub head_format: LinearFormat,
    /// Whether output logits reuse the embedding table.
    pub tied_head: bool,
}

impl<B: NeuralBackend> StaticModules<B> {
    /// Builds unloaded pinned modules from architecture-owned parameter
    /// identities and physical formats.
    pub fn from_spec(
        spec: StaticModuleSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let embeddings = B::embedding(
            EmbeddingSpec {
                vocabulary: spec.vocabulary,
                dimensions: spec.hidden_size,
                weight: ParameterSpec::trainable(&spec.embedding_weight).map_err(Error::backend)?,
                quantization: spec.embedding_quantization,
            },
            context,
        )?;
        let norm = B::rms_norm(
            NormalizationSpec {
                dimensions: spec.hidden_size,
                epsilon: spec.normalization_epsilon,
                weight: ParameterSpec::trainable(&spec.normalization_weight)
                    .map_err(Error::backend)?,
            },
            context,
        )?;
        let lm_head = if spec.tied_head {
            None
        } else {
            Some(B::linear(
                LinearSpec {
                    input: spec.hidden_size,
                    output: spec.vocabulary,
                    weight: ParameterSpec::trainable(&spec.head_weight).map_err(Error::backend)?,
                    bias: None,
                    format: spec.head_format,
                },
                context,
            )?)
        };
        Ok(Self {
            embeddings,
            norm,
            lm_head,
        })
    }

    /// Builds unloaded pinned modules for a decoder family.
    pub fn new<C: Config>(
        config: &C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let embedding_name = format!("{}.embed_tokens.weight", config.parameter_root());
        let norm_name = format!("{}.norm.weight", config.parameter_root());
        Self::from_spec(
            StaticModuleSpec {
                embedding_weight: embedding_name.clone(),
                normalization_weight: norm_name,
                head_weight: "lm_head.weight".into(),
                vocabulary: config.vocabulary_size(),
                hidden_size: config.hidden_size(),
                normalization_epsilon: config.rms_norm_epsilon(),
                embedding_quantization: config.weight_quantization(&embedding_name),
                head_format: config.weight_quantization("lm_head.weight").into(),
                tied_head: config.tie_word_embeddings(),
            },
            context,
        )
    }
}

/// Borrowed token input for the shared layered lifecycle.
pub struct LayeredInput<'a, T> {
    /// Token ids shaped `[batch, sequence]`.
    pub tokens: &'a T,
    /// Optional caller-provided attention mask.
    pub mask: Option<&'a T>,
}

/// Shared declaration and validation for one ordered decoder execution group.
#[derive(Debug, Clone)]
pub struct SequentialGroup {
    name: &'static str,
    parameter_root: &'static str,
    units: usize,
}

/// Shared declaration for a target group followed by zero or more ordered
/// single-unit prediction groups.
#[derive(Debug, Clone)]
pub struct SequentialPredictionGroups {
    target: SequentialGroup,
    prediction_roots: Vec<String>,
}

impl SequentialPredictionGroups {
    /// Creates the target group and `mtp.{depth}` prediction groups.
    pub fn new(
        target_parameter_root: &'static str,
        target_units: usize,
        prediction_roots: impl IntoIterator<Item = String>,
    ) -> Result<Self, Error> {
        Ok(Self {
            target: SequentialGroup::new("target", target_parameter_root, target_units)?,
            prediction_roots: prediction_roots.into_iter().collect(),
        })
    }

    /// Builds one dependency chain from target through every prediction depth.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain(
            std::iter::once("target".to_owned())
                .chain((0..self.prediction_roots.len()).map(|depth| format!("mtp.{depth}"))),
        )
        .map_err(Error::backend)
    }

    /// Returns the number of units in one group.
    pub fn unit_count(&self, group: usize) -> Result<usize, Error> {
        if group == 0 {
            self.target.unit_count(0)
        } else if group <= self.prediction_roots.len() {
            Ok(1)
        } else {
            Err(Error::backend(format!(
                "execution group {group} is outside target plus {} prediction groups",
                self.prediction_roots.len()
            )))
        }
    }

    /// Returns one stable target or prediction unit path.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        if group == 0 {
            return self.target.unit_path(0, index);
        }
        self.unit_count(group)?;
        if index != 0 {
            return Err(Error::backend(format!(
                "prediction group {group} contains one unit, received index {index}"
            )));
        }
        Ok(self.prediction_roots[group - 1].clone())
    }

    /// Selects the activation carried into a ready chain group.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group)?;
        if group == 0 {
            return self.target.begin(0, initial, dependencies);
        }
        match dependencies {
            [dependency] => Ok((*dependency).clone()),
            _ => Err(Error::backend(format!(
                "prediction group {group} expected one dependency, received {}",
                dependencies.len()
            ))),
        }
    }

    /// Returns the number of appended prediction groups.
    pub fn prediction_count(&self) -> usize {
        self.prediction_roots.len()
    }
}

impl SequentialGroup {
    /// Creates one non-empty ordered group.
    pub fn new(
        name: &'static str,
        parameter_root: &'static str,
        units: usize,
    ) -> Result<Self, Error> {
        if name.is_empty() || parameter_root.is_empty() || units == 0 {
            return Err(Error::backend(
                "sequential decoder group requires non-empty names and units",
            ));
        }
        Ok(Self {
            name,
            parameter_root,
            units,
        })
    }

    /// Builds the corresponding one-group dependency graph.
    pub fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Error> {
        eredu_runtime::ExecutionGraph::chain([self.name]).map_err(Error::backend)
    }

    /// Validates the group ordinal and returns its unit count.
    pub fn unit_count(&self, group: usize) -> Result<usize, Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "execution group {group} is outside {}",
                self.name
            )));
        }
        Ok(self.units)
    }

    /// Returns one validated stable unit path.
    pub fn unit_path(&self, group: usize, index: usize) -> Result<String, Error> {
        let count = self.unit_count(group)?;
        if index >= count {
            return Err(Error::backend(format!(
                "unit {index} is outside {count} {} units",
                self.name
            )));
        }
        Ok(format!("{}.{index}", self.parameter_root))
    }

    /// Starts the sole group from the initial activation.
    pub fn begin<T: Clone>(
        &self,
        group: usize,
        initial: &T,
        dependencies: &[&T],
    ) -> Result<T, Error> {
        self.unit_count(group)?;
        if !dependencies.is_empty() {
            return Err(Error::backend(format!(
                "{} received {} unexpected dependencies",
                self.name,
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }
}

/// Architecture-owned values retained across one layered forward pass.
pub struct ForwardContext<T> {
    mask: Option<T>,
    allow_sliding_prefill: bool,
    rotary_embeddings: Option<(T, T)>,
}

/// Statically dispatched construction policy for one decoder block family.
pub trait BlockFactory<B: NeuralBackend, C: Config>: 'static {
    /// Architecture-selected feed-forward policy inside the shared block.
    type FeedForward: FeedForwardOperator<B>;

    /// Builds one unloaded decoder block.
    fn build(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, Self::FeedForward>, Error>;
}

/// Dense SwiGLU block factory used by Llama and other all-dense decoders.
pub struct DenseBlockFactory;

impl<B: NeuralBackend, C: Config> BlockFactory<B, C> for DenseBlockFactory {
    type FeedForward = Mlp<B>;

    fn build(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<TransformerBlock<B, Self::FeedForward>, Error> {
        TransformerBlock::new(config, layer, context)
    }
}

/// Shared layered decoder lifecycle over architecture configuration and block policy.
pub struct LayeredModel<B: NeuralBackend, C: Config, P = DenseBlockFactory> {
    args: C,
    static_modules: StaticModules<B>,
    block_factory: std::marker::PhantomData<fn() -> P>,
}

impl<B, C, P> LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
{
    /// Builds unloaded pinned modules from normalized architecture arguments.
    pub fn new(args: C, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        args.validate_config()?;
        let static_modules = StaticModules::new(&args, context)?;
        Ok(Self {
            args,
            static_modules,
            block_factory: std::marker::PhantomData,
        })
    }

    /// Returns the normalized architecture arguments.
    pub const fn args(&self) -> &C {
        &self.args
    }

    /// Borrows pinned modules for neutral checkpoint loading.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned modules for neutral checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Prepares architecture-owned mask state after an execution policy has
    /// produced embeddings, including vocabulary-parallel embeddings.
    pub fn begin_embedded<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let expected = state_layout(&self.args)?;
        self.begin_embedded_with_layout(hidden, supplied_mask, state, &expected, context)
    }

    /// Prepares architecture-owned mask state against an explicitly realized
    /// state layout, such as the rank-local KV geometry produced by tensor
    /// parallel planning.
    pub fn begin_embedded_with_layout<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        self.begin_embedded_with_layout_and_rotary(
            hidden,
            supplied_mask,
            None,
            state,
            expected,
            context,
        )
    }

    /// Prepares a layered pass with caller-provided explicit rotary embeddings.
    pub fn begin_embedded_with_layout_and_rotary<S>(
        &mut self,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        rotary_embeddings: Option<(&B::Tensor, &B::Tensor)>,
        state: &mut S,
        expected: &StateLayout,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "decoder runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let sequence = hidden.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if sequence > 1 {
            let cache = state.layer(0).map_err(Error::backend)?;
            let window = cache.max_size();
            let offset = window.map_or(cache.offset(), |window| cache.offset().min(window));
            Some(B::causal_mask(sequence, offset, window, context)?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                mask,
                allow_sliding_prefill: supplied_mask.is_none(),
                rotary_embeddings: rotary_embeddings
                    .map(|(cosine, sine)| (cosine.clone(), sine.clone())),
            },
        })
    }

    /// Executes one replicated block using architecture-owned forward state.
    pub fn forward_block<S>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            context,
        )
    }

    /// Executes one replicated block while delegating its feed-forward policy
    /// to a composition-supplied executor.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_with_feed_forward<S, H>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        H: FnOnce(
            &mut P::FeedForward,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_with_feed_forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            context,
            feed_forward,
        )
    }

    /// Executes one tensor-parallel block using the same architecture-owned
    /// mask and state semantics as replicated execution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_parallel<S>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_tensor_parallel(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            parallel,
            context,
        )
    }

    /// Executes one tensor-parallel block while delegating its feed-forward
    /// policy to a composition-supplied executor.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_block_parallel_with_feed_forward<S, H>(
        &mut self,
        index: usize,
        block: &mut TransformerBlock<B, P::FeedForward>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        feed_forward: H,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor>,
        H: FnOnce(
            &mut P::FeedForward,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let cache = state.layer(index).map_err(Error::backend)?;
        block.forward_tensor_parallel_with_feed_forward(
            AttentionInput {
                hidden,
                mask: forward.mask.as_ref(),
                cache: Some(cache),
                allow_sliding_prefill: forward.allow_sliding_prefill,
                rotary_position: forward
                    .rotary_embeddings
                    .as_ref()
                    .map(|(cosine, sine)| RotaryPosition::Embeddings { cosine, sine }),
            },
            parallel,
            context,
            feed_forward,
        )
    }

    /// Applies final normalization and the tied or untied output projection.
    pub fn finish_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.norm.forward(hidden, context)?;
        match &mut self.static_modules.lm_head {
            Some(head) => head.forward(&hidden, context),
            None => self.static_modules.embeddings.as_linear(&hidden, context),
        }
    }
}

impl<B, C, P, S> LayeredArchitecture<B, S> for LayeredModel<B, C, P>
where
    B: NeuralBackend,
    C: Config,
    P: BlockFactory<B, C>,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor>,
{
    type Input<'a> = LayeredInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = TransformerBlock<B, P::FeedForward>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = std::option::Iter<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn model_identity(&self) -> &str {
        self.args.model_identity()
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        eredu_runtime::ExecutionGraph::chain(["text_decoder"]).map_err(Error::backend)
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        usize::try_from(self.args.num_hidden_layers()).map_err(Error::backend)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        let count = usize::try_from(self.args.num_hidden_layers()).map_err(Error::backend)?;
        if index >= count {
            return Err(Error::backend(format!(
                "decoder unit {index} is outside {count} decoder layers"
            )));
        }
        Ok(format!("{}.layers.{index}", self.args.parameter_root()))
    }

    fn static_modules(&self) -> &Self::StaticModules {
        &self.static_modules
    }

    fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
        &mut self.static_modules
    }

    fn build_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Unit, Self::Error> {
        if group != 0 {
            return Err(Error::backend(format!(
                "decoder execution group {group} is outside the text decoder"
            )));
        }
        P::build(&self.args, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let hidden = self
            .static_modules
            .embeddings
            .forward(input.tokens, context)?;
        self.begin_embedded(hidden, input.mask, state, context)
    }

    fn begin_execution_group(
        &mut self,
        group: usize,
        initial: &B::Tensor,
        dependencies: &[&B::Tensor],
        _state: &mut S,
        _forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group != 0 || !dependencies.is_empty() {
            return Err(Error::backend(format!(
                "text decoder group {group} received {} dependencies",
                dependencies.len()
            )));
        }
        Ok(initial.clone())
    }

    fn forward_unit(
        &mut self,
        _group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.forward_block(index, unit, hidden, state, forward, context)
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        _forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.finish_hidden(hidden, context)
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        forward.mask.iter()
    }
}
