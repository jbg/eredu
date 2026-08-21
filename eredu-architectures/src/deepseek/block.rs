//! Shared DeepSeek decoder-block sequencing.

use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, Error, HyperConnection,
    HyperConnectionSpec, HyperNeuralBackend, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationOperator, NormalizationSpec, Parameter, ParameterSpec, Parameterized,
    PoolingAttentionCache, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    observe_and_intervene, ActivationObserver, ExpertPass, ResidentExpertProvider,
    RoutedExpertProvider,
};

use super::{
    attention::v3::Attention as V3Attention,
    attention::v4::Attention as V4Attention,
    moe::{RouteSource, RoutedPlusShared},
    LayerPolicy, V3Args, V4Args,
};

/// Ordinary DeepSeek SwiGLU used by dense-prefix V3 layers.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseSwiGlu<B: NeuralBackend> {
    gate: B::Linear,
    up: B::Linear,
    down: B::Linear,
}

/// One V4 hyper-connected decoder block with scheduled local/compressed
/// attention and shared routed-expert execution.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct V4Block<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    /// Scheduled V4 attention operator.
    pub attention: V4Attention<B>,
    /// Learned or token-selected routed plus shared experts.
    pub feed_forward: RoutedPlusShared<B>,
    attention_norm: B::Normalization,
    feed_forward_norm: B::Normalization,
    attention_connection: HyperConnection<B>,
    feed_forward_connection: HyperConnection<B>,
    token_experts: Option<Parameter<B::Tensor>>,
    #[parameter(skip)]
    normalization_epsilon: f32,
}

impl<B> V4Block<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    /// Builds one unloaded target block, including its optional token-to-expert
    /// table for hash-routed prefix layers.
    pub fn new(
        args: &V4Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(args, layer, &format!("layers.{layer}"), context)
    }

    /// Builds one appended DSpark block from the canonical prediction root.
    pub fn new_dspark(
        args: &V4Args,
        layer: usize,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(args, layer, &format!("mtp.{depth}"), context)
    }

    pub(crate) fn new_at(
        args: &V4Args,
        layer: usize,
        root: &str,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let total = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
            .map_err(Error::backend)?;
        if layer >= total {
            return Err(Error::backend(format!("V4 layer {layer} is out of range")));
        }
        let norm = |name: String| {
            B::rms_norm(
                NormalizationSpec {
                    dimensions: args.hidden_size,
                    epsilon: args.rms_norm_eps,
                    weight: parameter(name)?,
                },
                context,
            )
        };
        let connection = |kind: &str| {
            HyperConnection::new(
                HyperConnectionSpec {
                    streams: args.hc_mult,
                    hidden_size: args.hidden_size,
                    sinkhorn_iterations: usize::try_from(args.hc_sinkhorn_iters)
                        .map_err(Error::backend)?,
                    epsilon: args.hc_eps,
                    function: parameter(format!("{root}.hc_{kind}_fn"))?,
                    base: parameter(format!("{root}.hc_{kind}_base"))?,
                    scale: parameter(format!("{root}.hc_{kind}_scale"))?,
                },
                context,
            )
        };
        Ok(Self {
            attention: V4Attention::new_at(args, layer, &format!("{root}.attn"), context)?,
            feed_forward: RoutedPlusShared::new(
                &super::v4::moe_policy_at(args, layer, &format!("{root}.ffn"))?,
                context,
            )?,
            attention_norm: norm(format!("{root}.attn_norm.weight"))?,
            feed_forward_norm: norm(format!("{root}.ffn_norm.weight"))?,
            attention_connection: connection("attn")?,
            feed_forward_connection: connection("ffn")?,
            token_experts: (layer < args.num_hash_layers as usize)
                .then(|| {
                    Parameter::unloaded_i32(
                        parameter(format!("{root}.ffn.gate.tid2eid"))?,
                        &[args.vocab_size, args.num_experts_per_tok],
                        context,
                    )
                })
                .transpose()?,
            normalization_epsilon: args.rms_norm_eps,
        })
    }

    /// Executes both V4 hyper-connection residual cycles.
    pub fn forward<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let residual = input;
        let state =
            self.attention_connection
                .collapse(input, self.normalization_epsilon, context)?;
        let normalized = self.attention_norm.forward(&state.collapsed, context)?;
        let attention = self.attention.forward(&normalized, mask, cache, context)?;
        let hidden = self
            .attention_connection
            .expand(&attention, residual, &state, context)?;

        let residual = &hidden;
        let state =
            self.feed_forward_connection
                .collapse(&hidden, self.normalization_epsilon, context)?;
        let normalized = self.feed_forward_norm.forward(&state.collapsed, context)?;
        let selected = self
            .token_experts
            .as_ref()
            .map(|table| {
                let tokens = input_ids.reshape(&[-1], context)?;
                table.as_ref().take_axis(&tokens, 0, context)
            })
            .transpose()?;
        let source = selected
            .as_ref()
            .map_or(RouteSource::Learned, RouteSource::Selected);
        let feed_forward = self.feed_forward.forward(&normalized, source, context)?;
        self.feed_forward_connection
            .expand(&feed_forward, residual, &state, context)
    }

    /// Executes the V4 block with routed experts supplied by runtime policy.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider<C, P>(
        &mut self,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let residual = input;
        let state =
            self.attention_connection
                .collapse(input, self.normalization_epsilon, context)?;
        let normalized = self.attention_norm.forward(&state.collapsed, context)?;
        let attention = self.attention.forward(&normalized, mask, cache, context)?;
        let hidden = self
            .attention_connection
            .expand(&attention, residual, &state, context)?;
        let state =
            self.feed_forward_connection
                .collapse(&hidden, self.normalization_epsilon, context)?;
        let normalized = self.feed_forward_norm.forward(&state.collapsed, context)?;
        let selected = self
            .token_experts
            .as_ref()
            .map(|table| {
                table
                    .as_ref()
                    .take_axis(&input_ids.reshape(&[-1], context)?, 0, context)
            })
            .transpose()?;
        let source = selected
            .as_ref()
            .map_or(RouteSource::Learned, RouteSource::Selected);
        let feed_forward = self.feed_forward.forward_with_provider(
            &normalized,
            source,
            pass,
            provider,
            context,
        )?;
        self.feed_forward_connection
            .expand(&feed_forward, &hidden, &state, context)
    }

    /// Executes a tensor-partitioned V4 block, reducing the partial attention
    /// and feed-forward projections before each hyper-connection expansion.
    pub fn forward_parallel<C, F>(
        &mut self,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        mut reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let state =
            self.attention_connection
                .collapse(input, self.normalization_epsilon, context)?;
        let normalized = self.attention_norm.forward(&state.collapsed, context)?;
        let attention = reduce(
            self.attention.forward(&normalized, mask, cache, context)?,
            context,
        )?;
        let hidden = self
            .attention_connection
            .expand(&attention, input, &state, context)?;
        let state =
            self.feed_forward_connection
                .collapse(&hidden, self.normalization_epsilon, context)?;
        let normalized = self.feed_forward_norm.forward(&state.collapsed, context)?;
        let selected = self
            .token_experts
            .as_ref()
            .map(|table| {
                table
                    .as_ref()
                    .take_axis(&input_ids.reshape(&[-1], context)?, 0, context)
            })
            .transpose()?;
        let source = selected
            .as_ref()
            .map_or(RouteSource::Learned, RouteSource::Selected);
        let feed_forward = reduce(
            self.feed_forward.forward(&normalized, source, context)?,
            context,
        )?;
        self.feed_forward_connection
            .expand(&feed_forward, &hidden, &state, context)
    }

    /// Tensor-partitioned V4 execution with runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<C, P, F>(
        &mut self,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        mut reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let state =
            self.attention_connection
                .collapse(input, self.normalization_epsilon, context)?;
        let normalized = self.attention_norm.forward(&state.collapsed, context)?;
        let attention = reduce(
            self.attention.forward(&normalized, mask, cache, context)?,
            context,
        )?;
        let hidden = self
            .attention_connection
            .expand(&attention, input, &state, context)?;
        let state =
            self.feed_forward_connection
                .collapse(&hidden, self.normalization_epsilon, context)?;
        let normalized = self.feed_forward_norm.forward(&state.collapsed, context)?;
        let selected = self
            .token_experts
            .as_ref()
            .map(|table| {
                table
                    .as_ref()
                    .take_axis(&input_ids.reshape(&[-1], context)?, 0, context)
            })
            .transpose()?;
        let source = selected
            .as_ref()
            .map_or(RouteSource::Learned, RouteSource::Selected);
        let feed_forward = self.feed_forward.forward_tensor_parallel_with_provider(
            &normalized,
            source,
            pass,
            provider,
            context,
            &mut reduce,
        )?;
        self.feed_forward_connection
            .expand(&feed_forward, &hidden, &state, context)
    }

    /// Projects one target capture into this block's attention cache without
    /// advancing its residual or feed-forward path.
    ///
    /// Fused block drafters use this to commit accepted target context before
    /// proposing the next block with ordinary local-attention execution.
    pub fn prefill_attention_cache<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error> {
        let state =
            self.attention_connection
                .collapse(input, self.normalization_epsilon, context)?;
        let normalized = self.attention_norm.forward(&state.collapsed, context)?;
        self.attention
            .forward(&normalized, None, Some(cache), context)?;
        Ok(())
    }

    /// Executes the V4 block with stable compressed-attention,
    /// hyper-connection, routing, and output observation points.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed<C, O>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut provider = ResidentExpertProvider;
        self.forward_observed_with_provider(
            path,
            input,
            input_ids,
            mask,
            cache,
            ExpertPass::Decode,
            &mut provider,
            context,
            observer,
        )
    }

    /// Executes the observed V4 block with runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<C, O, P>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let input = observe_and_intervene(observer, &format!("{path}.input"), input)?;
        let attention_state =
            self.attention_connection
                .collapse(&input, self.normalization_epsilon, context)?;
        let collapsed = observe_and_intervene(
            observer,
            &format!("{path}.hyper.attention.collapsed"),
            &attention_state.collapsed,
        )?;
        let normalized = self.attention_norm.forward(&collapsed, context)?;
        let attention = self.attention.forward_observed(
            &format!("{path}.compressed_attention"),
            &normalized,
            mask,
            cache,
            context,
            observer,
        )?;
        let attention = observe_and_intervene(
            observer,
            &format!("{path}.compressed_attention.output"),
            &attention,
        )?;
        let hidden =
            self.attention_connection
                .expand(&attention, &input, &attention_state, context)?;
        let hidden = observe_and_intervene(
            observer,
            &format!("{path}.hyper.attention.streams"),
            &hidden,
        )?;

        let feed_forward_state =
            self.feed_forward_connection
                .collapse(&hidden, self.normalization_epsilon, context)?;
        let collapsed = observe_and_intervene(
            observer,
            &format!("{path}.hyper.feed_forward.collapsed"),
            &feed_forward_state.collapsed,
        )?;
        let normalized = self.feed_forward_norm.forward(&collapsed, context)?;
        let selected = self
            .token_experts
            .as_ref()
            .map(|table| {
                table
                    .as_ref()
                    .take_axis(&input_ids.reshape(&[-1], context)?, 0, context)
            })
            .transpose()?;
        if let Some(selected) = &selected {
            observer.observe(&format!("{path}.routing.selected_indexes"), selected)?;
        }
        let source = selected
            .as_ref()
            .map_or(RouteSource::Learned, RouteSource::Selected);
        let feed_forward = self.feed_forward.forward_with_provider_observed(
            &format!("{path}.feed_forward"),
            &normalized,
            source,
            pass,
            provider,
            context,
            observer,
        )?;
        let output = self.feed_forward_connection.expand(
            &feed_forward,
            &hidden,
            &feed_forward_state,
            context,
        )?;
        observe_and_intervene(observer, &format!("{path}.output"), &output)
    }
}

impl<B: NeuralBackend> DenseSwiGlu<B> {
    fn new(
        args: &V3Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let root = format!("model.layers.{layer}.mlp");
        let linear = |field: &str, input, output| {
            let name = format!("{root}.{field}.weight");
            B::linear(
                LinearSpec {
                    input,
                    output,
                    weight: parameter(&name)?,
                    bias: None,
                    format: args.linear_format_for(&name),
                },
                context,
            )
        };
        Ok(Self {
            gate: linear("gate_proj", args.hidden_size, args.intermediate_size)?,
            up: linear("up_proj", args.hidden_size, args.intermediate_size)?,
            down: linear("down_proj", args.intermediate_size, args.hidden_size)?,
        })
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        let activated =
            B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
        self.down.forward(&activated, context)
    }
}

/// Dense-prefix or routed/shared V3 feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum V3FeedForward<B: RoutedNeuralBackend> {
    /// Dense-prefix SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed plus shared experts.
    Routed(RoutedPlusShared<B>),
}

/// One backend-neutral V3 target or MTP decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct V3Block<B: RoutedNeuralBackend + BlockwiseAttentionBackend> {
    /// V3 multi-head latent attention.
    pub attention: V3Attention<B>,
    /// Schedule-selected dense or sparse feed-forward layer.
    pub feed_forward: V3FeedForward<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
}

impl<B: RoutedNeuralBackend + BlockwiseAttentionBackend> V3Block<B> {
    /// Builds one unloaded target block from the validated layer schedule.
    pub fn new(
        args: &V3Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = *args
            .layer_schedule
            .get(layer)
            .ok_or_else(|| Error::backend(format!("missing V3 layer policy {layer}")))?;
        Self::new_with_policy(args, layer, policy, context)
    }

    pub(crate) fn new_prediction(
        args: &V3Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_with_policy(args, layer, LayerPolicy::SparseMoe, context)
    }

    fn new_with_policy(
        args: &V3Args,
        layer: usize,
        policy: LayerPolicy,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let root = format!("model.layers.{layer}");
        let norm = |field: &str| {
            B::rms_norm(
                NormalizationSpec {
                    dimensions: args.hidden_size,
                    epsilon: args.rms_norm_eps,
                    weight: parameter(format!("{root}.{field}.weight"))?,
                },
                context,
            )
        };
        Ok(Self {
            attention: V3Attention::new(args, layer, context)?,
            feed_forward: match policy {
                LayerPolicy::DenseMlp => {
                    V3FeedForward::Dense(DenseSwiGlu::new(args, layer, context)?)
                }
                LayerPolicy::SparseMoe => {
                    let policy = if layer < args.layer_schedule.len() {
                        super::v3::moe_policy(args, layer)?
                    } else {
                        super::v3::prediction_moe_policy(args, layer)?
                    };
                    V3FeedForward::Routed(RoutedPlusShared::new(&policy, context)?)
                }
            },
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    /// Executes pre-norm attention and feed-forward residual sequencing.
    pub fn forward<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let normalized = self.input_norm.forward(input, context)?;
        let attention = self.attention.forward(&normalized, mask, cache, context)?;
        let residual = input.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&residual, context)?;
        let feed_forward = match &mut self.feed_forward {
            V3FeedForward::Dense(mlp) => mlp.forward(&normalized, context)?,
            V3FeedForward::Routed(moe) => {
                moe.forward(&normalized, RouteSource::Learned, context)?
            }
        };
        residual.add(&feed_forward, context)
    }

    /// Executes the V3 block with routed experts supplied by runtime policy.
    pub fn forward_with_provider<C, P>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let normalized = self.input_norm.forward(input, context)?;
        let attention = self.attention.forward(&normalized, mask, cache, context)?;
        let residual = input.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&residual, context)?;
        let feed_forward = match &mut self.feed_forward {
            V3FeedForward::Dense(mlp) => mlp.forward(&normalized, context)?,
            V3FeedForward::Routed(moe) => moe.forward_with_provider(
                &normalized,
                RouteSource::Learned,
                pass,
                provider,
                context,
            )?,
        };
        residual.add(&feed_forward, context)
    }

    /// Executes a tensor-partitioned V3 block, reducing partial output
    /// projections before both residual additions.
    pub fn forward_parallel<C, F>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        mut reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input, context)?;
        let attention = reduce(
            self.attention.forward(&normalized, mask, cache, context)?,
            context,
        )?;
        let residual = input.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&residual, context)?;
        let feed_forward = match &mut self.feed_forward {
            V3FeedForward::Dense(mlp) => mlp.forward(&normalized, context)?,
            V3FeedForward::Routed(moe) => {
                moe.forward(&normalized, RouteSource::Learned, context)?
            }
        };
        residual.add(&reduce(feed_forward, context)?, context)
    }

    /// Tensor-partitioned V3 execution with runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<C, P, F>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        mut reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let normalized = self.input_norm.forward(input, context)?;
        let attention = reduce(
            self.attention.forward(&normalized, mask, cache, context)?,
            context,
        )?;
        let residual = input.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&residual, context)?;
        let feed_forward = match &mut self.feed_forward {
            V3FeedForward::Dense(mlp) => reduce(mlp.forward(&normalized, context)?, context)?,
            V3FeedForward::Routed(moe) => moe.forward_tensor_parallel_with_provider(
                &normalized,
                RouteSource::Learned,
                pass,
                provider,
                context,
                &mut reduce,
            )?,
        };
        residual.add(&feed_forward, context)
    }

    /// Executes the V3 block with stable MLA, routing, and intervention points.
    pub fn forward_observed<C, O>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut provider = ResidentExpertProvider;
        self.forward_observed_with_provider(
            path,
            input,
            mask,
            cache,
            ExpertPass::Decode,
            &mut provider,
            context,
            observer,
        )
    }

    /// Executes the observed V3 block with runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<C, O, P>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let input = observe_and_intervene(observer, &format!("{path}.input"), input)?;
        let normalized = self.input_norm.forward(&input, context)?;
        let attention = self.attention.forward(&normalized, mask, cache, context)?;
        let attention = observe_and_intervene(
            observer,
            &format!("{path}.compressed_attention.output"),
            &attention,
        )?;
        let residual = input.add(&attention, context)?;
        let normalized = self.post_attention_norm.forward(&residual, context)?;
        let feed_forward = match &mut self.feed_forward {
            V3FeedForward::Dense(mlp) => mlp.forward(&normalized, context)?,
            V3FeedForward::Routed(moe) => moe.forward_with_provider_observed(
                &format!("{path}.feed_forward"),
                &normalized,
                RouteSource::Learned,
                pass,
                provider,
                context,
                observer,
            )?,
        };
        let output = residual.add(&feed_forward, context)?;
        observe_and_intervene(observer, &format!("{path}.output"), &output)
    }
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
