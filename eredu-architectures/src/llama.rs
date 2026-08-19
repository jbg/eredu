//! Backend-neutral Llama/Mistral decoder implementation.

mod checkpoint;
mod config;

pub use checkpoint::{
    gguf_plan, safetensors_plan, translate_gguf_weight_name, SafetensorsPlanError,
};
pub use config::{
    model_args_from_config_reader, model_args_from_config_value, model_args_from_gguf_catalog,
    prompt_cache_architecture_fingerprint, ConfigError, GgufTensorCatalog, ModelArgs,
};

use eredu_checkpoint::WeightQuantization;
use eredu_core::cache::LayerCachePolicy;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, EmbeddingOperator, EmbeddingSpec, Error, LinearOperator, LinearSpec,
    NeuralBackend, NormalizationOperator, NormalizationSpec, ParameterSpec, RotaryOperator,
    RotarySpec, Tensor,
};
use eredu_runtime::{
    aligned_partition_units, module_parameter_group, partitioned_projection_group, MemberSharding,
    ModelStateIdentity, ParallelPlanError, ParameterGroupSpec, ParameterRole, ProjectionSharding,
    StateLayout,
};

/// Geometry consumed by the shared Llama implementation.
pub trait Config {
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
    /// Whether projections own attention biases.
    fn attention_bias(&self) -> bool;
    /// Whether projections own MLP biases.
    fn mlp_bias(&self) -> bool;
    /// Whether the language-model head is tied to input embeddings.
    fn tie_word_embeddings(&self) -> bool;
    /// Exact per-layer attention policy.
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy>;
    /// Physical encoding selected for one canonical checkpoint parameter.
    fn weight_quantization(&self, name: &str) -> Option<WeightQuantization>;
    /// Complete rotary-position construction specification.
    fn rotary_spec(&self, dimensions: i32) -> RotarySpec<'_>;
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

/// Declares Llama's cache compatibility identity independently of its state
/// storage backend.
pub fn state_identity(
    args: &ModelArgs,
    layout: &StateLayout,
    global_layer_start: usize,
    topology: eredu_core::cache::PromptCacheTopology,
) -> Result<ModelStateIdentity, Error> {
    let global_layer_end = global_layer_start
        .checked_add(layout.len())
        .ok_or_else(|| Error::backend("Llama owned layer range overflowed"))?;
    let layer_count = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    if global_layer_end > layer_count {
        return Err(Error::backend(format!(
            "Llama owns layers {global_layer_start}..{global_layer_end}, outside {layer_count} layers"
        )));
    }
    Ok(ModelStateIdentity {
        model_family: "llama".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start,
        sink_tokens: 0,
        layer_prefix_offsets: vec![0; layout.len()],
        topology,
    })
}

/// Derives a cache layout with backend/distribution-local key/value head counts.
pub fn cache_layout_with_key_value_heads<C: Config>(
    config: &C,
    key_value_heads: impl IntoIterator<Item = i32>,
) -> Result<LayerSchedule<LayerCachePolicy>, Error> {
    let layers = usize::try_from(config.num_hidden_layers()).map_err(Error::backend)?;
    let key_value_heads = key_value_heads.into_iter().collect::<Vec<_>>();
    if key_value_heads.len() != layers {
        return Err(Error::backend(format!(
            "Llama cache geometry has {} layers, expected {layers}",
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
            "Llama cache has {} layers, expected {}",
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
            .ok_or_else(|| Error::backend(format!("Llama cache is missing layer {layer}")))?;
        let expected = policy
            .window()
            .map(|window| i32::try_from(window.get()))
            .transpose()
            .map_err(Error::backend)?;
        if cache.max_size() != expected {
            return Err(Error::backend(format!(
                "Llama cache policy mismatch at layer {layer}: expected {policy:?}, cache window is {:?}",
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
            "Llama attention schedule has {} layers, expected {layers}",
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
}

/// Llama grouped-query self attention.
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
    fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.self_attn");
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
                    quantization: config.weight_quantization(&weight_name),
                },
                context,
            )
        };
        let policy = config.attention_schedule().get(layer).ok_or_else(|| {
            Error::backend(format!(
                "Llama attention schedule has no policy for layer {layer}"
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
                config.attention_bias(),
            )?,
            key: linear(
                "k_proj",
                hidden,
                key_value_heads * head,
                config.attention_bias(),
            )?,
            value: linear(
                "v_proj",
                hidden,
                key_value_heads * head,
                config.attention_bias(),
            )?,
            output: linear(
                "o_proj",
                query_heads * head,
                hidden,
                config.attention_bias(),
            )?,
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
        let queries = reshape(self.query.forward(hidden, context)?, self.query_heads)?;
        let keys = reshape(self.key.forward(hidden, context)?, self.key_value_heads)?;
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
        let queries = self.rotary.forward(&queries, offset, context)?;
        let keys = self.rotary.forward(&keys, offset, context)?;
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
            context,
        )?;
        B::row_parallel_linear(&mut self.output, &attended, parallel, context)
    }
}

/// Llama SwiGLU feed-forward network.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Mlp<B: NeuralBackend> {
    /// Gating projection.
    pub gate: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    /// Down projection.
    pub down: B::Linear,
}

impl<B: NeuralBackend> Mlp<B> {
    fn new<C: Config>(
        config: &C,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.mlp");
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
                    quantization: config.weight_quantization(&weight_name),
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
        })
    }

    fn hidden(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = B::silu(self.gate.forward(input, context)?, context)?;
        let up = self.up.forward(input, context)?;
        gate.multiply(&up, context)
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

/// One Llama decoder block.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct TransformerBlock<B: NeuralBackend> {
    /// Self-attention operator.
    pub self_attention: Attention<B>,
    /// Feed-forward operator.
    pub mlp: Mlp<B>,
    /// Pre-attention RMSNorm.
    pub input_norm: B::Normalization,
    /// Pre-MLP RMSNorm.
    pub post_attention_norm: B::Normalization,
}

impl<B: NeuralBackend> TransformerBlock<B> {
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
                        "model.layers.{layer}.input_layernorm.weight"
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
                        "model.layers.{layer}.post_attention_layernorm.weight"
                    ))
                    .map_err(Error::backend)?,
                },
                context,
            )?,
        })
    }

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
            },
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = self.mlp.forward(&normalized, context)?;
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
            },
            parallel,
            context,
        )?;
        let hidden = input.hidden.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&hidden, context)?;
        let mlp = self.mlp.forward_parallel(&normalized, parallel, context)?;
        hidden.add(&mlp, context)
    }
}

/// Declares every rank-local placement group for one Llama decoder block.
pub fn layer_parallel_parameter_groups<B: NeuralBackend>(
    block: &TransformerBlock<B>,
    config: &impl Config,
    layer: usize,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let prefix = format!("model.layers.{layer}");
    let query_heads = usize::try_from(config.num_attention_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Llama query-head count exceeds usize".into())
    })?;
    let key_value_heads = usize::try_from(config.num_key_value_heads()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Llama key/value-head count exceeds usize".into())
    })?;
    let head_dimension = usize::try_from(config.head_dim()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Llama head dimension exceeds usize".into())
    })?;
    if head_dimension == 0 || key_value_heads == 0 || !query_heads.is_multiple_of(key_value_heads) {
        return Err(ParallelPlanError::InvalidGroup(format!(
            "Llama attention geometry q={query_heads}, kv={key_value_heads}, dim={head_dimension} does not form positive integral GQA groups"
        )));
    }
    let group_width = (query_heads / key_value_heads)
        .checked_mul(head_dimension)
        .ok_or_else(|| {
            ParallelPlanError::InvalidGroup("Llama GQA group width overflowed".into())
        })?;
    let attention_alignment = config
        .weight_quantization(&format!("{prefix}.self_attn.o_proj.weight"))
        .map_or(Ok(1), |quantization| {
            usize::try_from(quantization.group_size()).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "Llama output-projection quantization group exceeds usize".into(),
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

    let intermediate = usize::try_from(config.intermediate_size()).map_err(|_| {
        ParallelPlanError::InvalidGroup("Llama feed-forward width exceeds usize".into())
    })?;
    let mlp_alignment = config
        .weight_quantization(&format!("{prefix}.mlp.down_proj.weight"))
        .map_or(Ok(1), |quantization| {
            usize::try_from(quantization.group_size()).map_err(|_| {
                ParallelPlanError::InvalidGroup(
                    "Llama down-projection quantization group exceeds usize".into(),
                )
            })
        })?;
    let mlp_units =
        aligned_partition_units(&format!("{prefix}.mlp"), intermediate, 1, mlp_alignment)?;
    let mlp = partitioned_projection_group::<B::Tensor, B::Linear>(
        format!("{prefix}.mlp.projections"),
        ParameterRole::FeedForwardIntermediate,
        &[
            (&block.mlp.gate, ProjectionSharding::Column),
            (&block.mlp.up, ProjectionSharding::Column),
            (&block.mlp.down, ProjectionSharding::Row),
        ],
        mlp_units,
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
    Ok(vec![attention, mlp, input_norm, post_attention_norm])
}

/// Declares embedding, final normalization, and output-head placement groups.
pub fn static_parallel_parameter_groups<B: NeuralBackend>(
    embeddings: &B::Embedding,
    norm: &B::Normalization,
    head: Option<&B::Linear>,
) -> Result<Vec<ParameterGroupSpec>, ParallelPlanError> {
    let mut groups = vec![
        module_parameter_group::<B::Tensor, _>(
            "model.embed_tokens",
            ParameterRole::Vocabulary,
            embeddings,
            |_, shape| {
                if shape.is_empty() {
                    Err(ParallelPlanError::InvalidTensor(
                        "Llama embedding parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?,
        module_parameter_group::<B::Tensor, _>(
            "model.norm",
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
                        "Llama language-model head parameter is scalar".into(),
                    ))
                } else {
                    Ok(MemberSharding::Balanced { axis: 0 })
                }
            },
        )?);
    }
    Ok(groups)
}

/// Llama transformer body without its language-model head.
#[derive(Debug, Clone, eredu_nn::Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct Decoder<B: NeuralBackend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Decoder blocks.
    pub layers: Vec<TransformerBlock<B>>,
    /// Final RMSNorm.
    pub norm: B::Normalization,
}

impl<B: NeuralBackend> Decoder<B> {
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
                    weight: ParameterSpec::trainable("model.embed_tokens.weight")
                        .map_err(Error::backend)?,
                    quantization: config.weight_quantization("model.embed_tokens.weight"),
                },
                context,
            )?,
            layers,
            norm: B::rms_norm(
                NormalizationSpec {
                    dimensions: config.hidden_size(),
                    epsilon: config.rms_norm_epsilon(),
                    weight: ParameterSpec::trainable("model.norm.weight")
                        .map_err(Error::backend)?,
                },
                context,
            )?,
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
                "Llama cache has {} layers, expected {}",
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
        mut hidden: B::Tensor,
        mask: Option<&B::Tensor>,
        allow_sliding_prefill: bool,
        caches: &mut [Option<C>],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        if caches.len() != self.layers.len() {
            return Err(Error::backend(format!(
                "Llama cache has {} layers, expected {}",
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
                },
                context,
            )?;
        }
        self.norm.forward(&hidden, context)
    }
}

/// Complete Llama causal language model.
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
                    quantization: config.weight_quantization("lm_head.weight"),
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
                .expect("validated Llama cache has a first layer");
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
