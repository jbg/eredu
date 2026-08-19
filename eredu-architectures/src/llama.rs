//! Backend-neutral Llama/Mistral decoder implementation.

use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{
    AttentionCache, Backend, EmbeddingOperator, Error, LinearOperator, LinearSpec,
    NormalizationOperator, RotaryOperator, Tensor,
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
#[derive(Debug, Clone)]
pub struct Attention<B: Backend> {
    /// Number of query heads.
    pub query_heads: i32,
    /// Number of key/value heads.
    pub key_value_heads: i32,
    /// Inverse square-root head scaling.
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
    pub sliding_window: Option<i32>,
}

struct AttentionProjections<T> {
    queries: T,
    keys: T,
    values: T,
    batch: i32,
    sequence: i32,
}

impl<B> Attention<B>
where
    B: Backend,
    B::Config: Config,
{
    fn new(
        config: &B::Config,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.self_attn");
        let hidden = config.hidden_size();
        let head = config.head_dim();
        let query_heads = config.num_attention_heads();
        let key_value_heads = config.num_key_value_heads();
        let linear = |field: &str, input, output, bias| {
            let weight_name = format!("{prefix}.{field}.weight");
            B::linear(
                config,
                LinearSpec {
                    input,
                    output,
                    bias,
                    weight_name: &weight_name,
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
            rotary: B::rotary(config, head, context)?,
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
#[derive(Debug, Clone)]
pub struct Mlp<B: Backend> {
    /// Gating projection.
    pub gate: B::Linear,
    /// Up projection.
    pub up: B::Linear,
    /// Down projection.
    pub down: B::Linear,
}

impl<B> Mlp<B>
where
    B: Backend,
    B::Config: Config,
{
    fn new(
        config: &B::Config,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let prefix = format!("model.layers.{layer}.mlp");
        let build = |field: &str, input, output| {
            let weight_name = format!("{prefix}.{field}.weight");
            B::linear(
                config,
                LinearSpec {
                    input,
                    output,
                    bias: config.mlp_bias(),
                    weight_name: &weight_name,
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
#[derive(Debug, Clone)]
pub struct TransformerBlock<B: Backend> {
    /// Self-attention operator.
    pub self_attention: Attention<B>,
    /// Feed-forward operator.
    pub mlp: Mlp<B>,
    /// Pre-attention RMSNorm.
    pub input_norm: B::Normalization,
    /// Pre-MLP RMSNorm.
    pub post_attention_norm: B::Normalization,
}

impl<B> TransformerBlock<B>
where
    B: Backend,
    B::Config: Config,
{
    /// Builds an unloaded block for one global layer index.
    pub fn new(
        config: &B::Config,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Ok(Self {
            self_attention: Attention::new(config, layer, context)?,
            mlp: Mlp::new(config, layer, context)?,
            input_norm: B::rms_norm(config.hidden_size(), config.rms_norm_epsilon(), context)?,
            post_attention_norm: B::rms_norm(
                config.hidden_size(),
                config.rms_norm_epsilon(),
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

/// Llama transformer body without its language-model head.
#[derive(Debug, Clone)]
pub struct Decoder<B: Backend> {
    /// Token embedding table.
    pub embeddings: B::Embedding,
    /// Decoder blocks.
    pub layers: Vec<TransformerBlock<B>>,
    /// Final RMSNorm.
    pub norm: B::Normalization,
}

impl<B> Decoder<B>
where
    B: Backend,
    B::Config: Config,
{
    /// Builds an unloaded decoder.
    pub fn new(
        config: &B::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let layers = (0..config.num_hidden_layers() as usize)
            .map(|layer| TransformerBlock::new(config, layer, context))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            embeddings: B::embedding(
                config,
                config.vocabulary_size(),
                config.hidden_size(),
                "model.embed_tokens.weight",
                context,
            )?,
            layers,
            norm: B::rms_norm(config.hidden_size(), config.rms_norm_epsilon(), context)?,
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
#[derive(Debug, Clone)]
pub struct Model<B: Backend> {
    /// Transformer body.
    pub decoder: Decoder<B>,
    /// Optional untied output projection.
    pub lm_head: Option<B::Linear>,
}

impl<B> Model<B>
where
    B: Backend,
    B::Config: Config,
{
    /// Builds an unloaded model.
    pub fn new(
        config: &B::Config,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let decoder = Decoder::new(config, context)?;
        let lm_head = if config.tie_word_embeddings() {
            None
        } else {
            Some(B::linear(
                config,
                LinearSpec {
                    input: config.hidden_size(),
                    output: config.vocabulary_size(),
                    bias: false,
                    weight_name: "lm_head.weight",
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
}
