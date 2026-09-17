//! Shared DeepSeek decoder-block sequencing.

use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, Error, GroupedNeuralBackend,
    HyperConnection, HyperConnectionSpec, HyperNeuralBackend, LinearOperator, LinearSpec,
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, Parameter, ParameterSpec,
    Parameterized, PoolingAttentionCache, Tensor,
};
use eredu_runtime::{
    ActivationObserver, ExpertPass, ResidentExpertProvider, RoutedExpertProvider,
    observe_and_intervene,
};

use super::{
    LayerPolicy, V3Args, V4Args,
    attention::v3::Attention as V3Attention,
    attention::v4::Attention as V4Attention,
    moe::{RouteSource, RoutedPlusShared},
};
use crate::decoder::ComponentInstrumentation;

mod construction;
mod v4_construction;
pub(crate) use construction::{V3BlockSpec, V3PredictionBlockSpec};
pub(crate) use v4_construction::V4BlockSpec;

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
    B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend,
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
    #[parameter(skip, metadata)]
    normalization_epsilon: f32,
}

impl<B> V4Block<B>
where
    B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend,
{
    /// Builds one unloaded target block, including its optional token-to-expert
    /// table for hash-routed prefix layers.
    pub fn new(
        args: &V4Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        Self::new_at(args, layer, &format!("layers.{layer}"), None, context)
    }

    pub(crate) fn new_with_expert_spec(
        args: &V4Args,
        layer: usize,
        expert_spec: eredu_nn::GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        Self::new_at(
            args,
            layer,
            &format!("layers.{layer}"),
            Some(expert_spec),
            context,
        )
    }

    /// Builds one appended DSpark block from the canonical prediction root.
    pub fn new_dspark(
        args: &V4Args,
        layer: usize,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        Self::new_at(args, layer, &format!("mtp.{depth}"), None, context)
    }

    pub(crate) fn new_at(
        args: &V4Args,
        layer: usize,
        root: &str,
        expert_spec: Option<eredu_nn::GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        V4BlockSpec::new(args, layer, root, expert_spec)?.instantiate::<B>(context)
    }

    /// The common serial/parallel/provider driver. Coefficients are existing
    /// mechanism outputs; disabled instrumentation creates no diagnostic views.
    #[allow(clippy::too_many_arguments)]
    fn forward_cycle<C, R, F>(
        &mut self,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        mut reduce: R,
        feed_forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        F: FnOnce(
            &mut RoutedPlusShared<B>,
            &B::Tensor,
            RouteSource<'_, B::Tensor>,
            &<B::Tensor as Tensor>::Context,
            &mut R,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let attention_state =
            self.attention_connection
                .collapse(input, self.normalization_epsilon, context)?;
        let collapsed = observe_v4_collapse(instrumentation, "hyper.attention", &attention_state)?;
        let normalized = self.attention_norm.forward(
            collapsed.as_ref().unwrap_or(&attention_state.collapsed),
            context,
        )?;
        let normalized = instrumentation.apply("compressed_attention.input", normalized)?;
        let attention = instrumentation.with_scope("compressed_attention", |instrumentation| {
            self.attention
                .forward_instrumented(&normalized, mask, cache, context, instrumentation)
        })?;
        let attention = reduce(attention, context)?;
        let attention = instrumentation.apply("compressed_attention.output", attention)?;
        let hidden =
            self.attention_connection
                .expand(&attention, input, &attention_state, context)?;
        let hidden = instrumentation.apply("hyper.attention.streams", hidden)?;

        let feed_forward_state =
            self.feed_forward_connection
                .collapse(&hidden, self.normalization_epsilon, context)?;
        let collapsed =
            observe_v4_collapse(instrumentation, "hyper.feed_forward", &feed_forward_state)?;
        let normalized = self.feed_forward_norm.forward(
            collapsed.as_ref().unwrap_or(&feed_forward_state.collapsed),
            context,
        )?;
        let normalized = instrumentation.apply("feed_forward.input", normalized)?;
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
            instrumentation.observe("routing.selected_indexes", selected)?;
        }
        let source = selected
            .as_ref()
            .map_or(RouteSource::Learned, RouteSource::Selected);
        let feed_forward = feed_forward(
            &mut self.feed_forward,
            &normalized,
            source,
            context,
            &mut reduce,
            instrumentation,
        )?;
        let feed_forward = instrumentation.apply("feed_forward.contribution", feed_forward)?;
        self.feed_forward_connection
            .expand(&feed_forward, &hidden, &feed_forward_state, context)
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
        let mut provider = ResidentExpertProvider;
        self.forward_with_provider(
            input,
            input_ids,
            mask,
            cache,
            ExpertPass::Decode,
            &mut provider,
            context,
        )
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
        self.forward_cycle(
            input,
            input_ids,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
            |value, _| Ok(value),
            |feed_forward, normalized, source, context, _, _| {
                feed_forward.forward_with_provider(normalized, source, pass, provider, context)
            },
        )
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
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_cycle(
            input,
            input_ids,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
            reduce,
            |feed_forward, normalized, source, context, reduce, _| {
                reduce(feed_forward.forward(normalized, source, context)?, context)
            },
        )
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
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_cycle(
            input,
            input_ids,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
            reduce,
            |feed_forward, normalized, source, context, reduce, _| {
                feed_forward.forward_tensor_parallel_with_provider(
                    normalized, source, pass, provider, context, reduce,
                )
            },
        )
    }

    /// Observes resident tensor-parallel units before the ordinary fused
    /// expert/shared reduction and records complete hyper-stream expansions.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed<C, O, F>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(path, &mut borrowed);
        let input = instrumentation.apply("input", input.clone())?;
        let output = self.forward_cycle(
            &input,
            input_ids,
            mask,
            cache,
            context,
            &mut instrumentation,
            reduce,
            |feed_forward, normalized, source, context, reduce, instrumentation| {
                feed_forward.forward_tensor_parallel_resident_observed(
                    &format!("{path}.feed_forward"),
                    normalized,
                    source,
                    ExpertPass::Decode,
                    context,
                    instrumentation.observer().expect("observed V4 block"),
                    reduce,
                )
            },
        )?;
        instrumentation.apply("output", output)
    }

    /// The same component boundaries with runtime-supplied tensor-parallel banks.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<C, O, P, F>(
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
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(path, &mut borrowed);
        let input = instrumentation.apply("input", input.clone())?;
        let output = self.forward_cycle(
            &input,
            input_ids,
            mask,
            cache,
            context,
            &mut instrumentation,
            reduce,
            |feed_forward, normalized, source, context, reduce, instrumentation| {
                feed_forward.forward_tensor_parallel_with_provider_observed(
                    &format!("{path}.feed_forward"),
                    normalized,
                    source,
                    pass,
                    provider,
                    context,
                    instrumentation.observer().expect("observed V4 block"),
                    reduce,
                )
            },
        )?;
        instrumentation.apply("output", output)
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
        self.prefill_attention_cache_instrumented(
            input,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Observes only the inputs that actually affect a context-cache update.
    /// The discarded attention write and the unexecuted FFN are not exposed as
    /// causal component seams in this invocation.
    pub fn prefill_attention_cache_observed<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut dyn ActivationObserver<B::Tensor, Error>,
    ) -> Result<(), Error> {
        self.prefill_attention_cache_instrumented(
            input,
            cache,
            context,
            &mut ComponentInstrumentation::new(path, observer),
        )
    }

    fn prefill_attention_cache_instrumented<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<(), Error> {
        let effective = if instrumentation.enabled() {
            Some(instrumentation.apply("input", input.clone())?)
        } else {
            None
        };
        let state = self.attention_connection.collapse(
            effective.as_ref().unwrap_or(input),
            self.normalization_epsilon,
            context,
        )?;
        let collapsed = instrumentation.apply("collapsed", state.collapsed)?;
        let normalized = self.attention_norm.forward(&collapsed, context)?;
        let normalized = instrumentation.apply("normalized", normalized)?;
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
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        let mut instrumentation = ComponentInstrumentation::new(path, &mut borrowed);
        let input = instrumentation.apply("input", input.clone())?;
        let output = self.forward_cycle(
            &input,
            input_ids,
            mask,
            cache,
            context,
            &mut instrumentation,
            |value, _| Ok(value),
            |feed_forward, normalized, source, context, _, instrumentation| {
                feed_forward.forward_with_provider_observed(
                    &format!("{path}.feed_forward"),
                    normalized,
                    source,
                    pass,
                    provider,
                    context,
                    instrumentation.observer().expect("observed V4 block"),
                )
            },
        )?;
        instrumentation.apply("output", output)
    }
}

fn observe_v4_collapse<T: Tensor>(
    instrumentation: &mut ComponentInstrumentation<'_, T>,
    scope: &str,
    state: &eredu_nn::HyperConnectionState<T>,
) -> Result<Option<T>, Error> {
    if !instrumentation.enabled() {
        return Ok(None);
    }
    instrumentation.with_scope(scope, |instrumentation| {
        instrumentation.observe("pre", &state.pre)?;
        instrumentation.observe("post", &state.post)?;
        instrumentation.observe("combination", &state.combination)?;
        instrumentation
            .apply("collapsed", state.collapsed.clone())
            .map(Some)
    })
}

impl<B: NeuralBackend> DenseSwiGlu<B> {
    fn new(
        args: &V3Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        construction::DenseSwiGluSpec::new(args, layer)?.instantiate::<B>(context)
    }

    fn forward(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(input, context, &mut ComponentInstrumentation::disabled())
    }

    fn forward_instrumented(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let write = self.forward_partial_instrumented(input, context, instrumentation)?;
        let write = instrumentation.apply("feed_forward.write", write)?;
        instrumentation.apply("feed_forward.output", write)
    }

    fn forward_partial_instrumented(
        &mut self,
        input: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let gate = self.gate.forward(input, context)?;
        let up = self.up.forward(input, context)?;
        let activated =
            B::gated_product(gate, up, eredu_nn::GatedProductPolicy::default(), context)?;
        let activated = instrumentation.apply("feed_forward.units", activated)?;
        instrumentation.project::<B>(
            "feed_forward.write_input",
            &mut self.down,
            &activated,
            None,
            context,
        )
    }
}

fn forward_v3_block<B, C, F>(
    attention: &mut V3Attention<B>,
    input_norm: &mut B::Normalization,
    post_attention_norm: &mut B::Normalization,
    input: &B::Tensor,
    mask: Option<&B::Tensor>,
    cache: Option<&mut C>,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    feed_forward: F,
) -> Result<B::Tensor, Error>
where
    B: BlockwiseAttentionBackend,
    C: CompressedAttentionCache<B::Tensor>,
    F: FnOnce(
        &B::Tensor,
        &<B::Tensor as Tensor>::Context,
        &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error>,
{
    forward_v3_block_reduced(
        attention,
        input_norm,
        post_attention_norm,
        input,
        mask,
        cache,
        context,
        instrumentation,
        |value, _| Ok(value),
        |input, context, instrumentation, _| feed_forward(input, context, instrumentation),
    )
}

fn forward_v3_block_reduced<B, C, F, R>(
    attention: &mut V3Attention<B>,
    input_norm: &mut B::Normalization,
    post_attention_norm: &mut B::Normalization,
    input: &B::Tensor,
    mask: Option<&B::Tensor>,
    cache: Option<&mut C>,
    context: &<B::Tensor as Tensor>::Context,
    instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    mut reduce: R,
    feed_forward: F,
) -> Result<B::Tensor, Error>
where
    R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    B: BlockwiseAttentionBackend,
    C: CompressedAttentionCache<B::Tensor>,
    F: FnOnce(
        &B::Tensor,
        &<B::Tensor as Tensor>::Context,
        &mut ComponentInstrumentation<'_, B::Tensor>,
        &mut R,
    ) -> Result<B::Tensor, Error>,
{
    let normalized = input_norm.forward(input, context)?;
    let normalized = instrumentation.apply("attention.input", normalized)?;
    let attention =
        attention.forward_instrumented(&normalized, mask, cache, context, instrumentation)?;
    let attention = instrumentation.apply("attention.write", reduce(attention, context)?)?;
    // Retain the existing complete MLA-output intervention identity.
    let attention = instrumentation.apply("compressed_attention.output", attention)?;
    let residual = input.add(&attention, context)?;
    let residual = instrumentation.apply("attention.residual", residual)?;
    let normalized = post_attention_norm.forward(&residual, context)?;
    let normalized = instrumentation.apply("feed_forward.input", normalized)?;
    let feed_forward = feed_forward(&normalized, context, instrumentation, &mut reduce)?;
    instrumentation.apply(
        "feed_forward.residual",
        residual.add(&feed_forward, context)?,
    )
}

/// One V3 decoder block whose validated schedule contains no routed experts.
///
/// Keeping this type separate from [`V3Block`] lets ordinary dense V3 models
/// use the neutral compressed-attention path without imposing grouped or
/// distributed backend capabilities that only sparse layers need.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DenseV3Block<B: BlockwiseAttentionBackend> {
    /// V3 multi-head latent attention.
    pub attention: V3Attention<B>,
    feed_forward: DenseSwiGlu<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
}

impl<B: BlockwiseAttentionBackend> DenseV3Block<B> {
    /// Builds one unloaded dense target block from the validated layer schedule.
    pub fn new(
        args: &V3Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        match args.layer_schedule.get(layer) {
            Some(LayerPolicy::DenseMlp) => {}
            Some(LayerPolicy::SparseMoe) => {
                return Err(Error::backend(format!(
                    "dense V3 block received routed layer {layer}"
                )));
            }
            None => return Err(Error::backend(format!("missing V3 layer policy {layer}"))),
        }
        let root = format!("model.layers.{layer}");
        let norm = |field: &str| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    parameter(format!("{root}.{field}.weight"))?,
                ),
                context,
            )
        };
        Ok(Self {
            attention: V3Attention::new(args, layer, context)?,
            feed_forward: DenseSwiGlu::new(args, layer, context)?,
            input_norm: norm("input_layernorm")?,
            post_attention_norm: norm("post_attention_layernorm")?,
        })
    }

    /// Executes pre-norm MLA and dense SwiGLU residual sequencing.
    pub fn forward<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.forward_instrumented(
            input,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
        )
    }

    /// Runs the same compressed-attention and dense-unit boundaries with an
    /// admitted observer. The traversal owns block input/output publication.
    pub fn forward_instrumented<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
    ) -> Result<B::Tensor, Error> {
        let Self {
            attention,
            feed_forward,
            input_norm,
            post_attention_norm,
        } = self;
        forward_v3_block(
            attention,
            input_norm,
            post_attention_norm,
            input,
            mask,
            cache,
            context,
            instrumentation,
            |normalized, context, instrumentation| {
                feed_forward.forward_instrumented(normalized, context, instrumentation)
            },
        )
    }
}

/// Dense-prefix or routed/shared V3 feed-forward policy.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum V3FeedForward<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend> {
    /// Dense-prefix SwiGLU.
    Dense(DenseSwiGlu<B>),
    /// Routed plus shared experts.
    Routed(RoutedPlusShared<B>),
}

/// One backend-neutral V3 target or MTP decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct V3Block<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
> {
    /// V3 multi-head latent attention.
    pub attention: V3Attention<B>,
    /// Schedule-selected dense or sparse feed-forward layer.
    pub feed_forward: V3FeedForward<B>,
    input_norm: B::Normalization,
    post_attention_norm: B::Normalization,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend>
    V3Block<B>
{
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
        Self::new_with_policy(args, layer, policy, None, context)
    }

    pub(crate) fn new_with_expert_spec(
        args: &V3Args,
        layer: usize,
        expert_spec: eredu_nn::GroupedGatedProductSpec,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let policy = *args
            .layer_schedule
            .get(layer)
            .ok_or_else(|| Error::backend(format!("missing V3 layer policy {layer}")))?;
        if policy != LayerPolicy::SparseMoe {
            return Err(Error::backend(format!(
                "V3 expert realization names dense layer {layer}"
            )));
        }
        Self::new_with_policy(args, layer, policy, Some(expert_spec), context)
    }

    pub(crate) fn new_prediction(
        args: &V3Args,
        layer: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if B::construction_metadata(context)
            .is_some_and(|metadata| metadata.uses_checked_metadata())
        {
            return Err(eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into());
        }
        V3PredictionBlockSpec::new(args, layer)?.instantiate::<B>(context)
    }

    fn new_with_policy(
        args: &V3Args,
        layer: usize,
        policy: LayerPolicy,
        expert_spec: Option<eredu_nn::GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::decoder::construction_specs::require_source_compiler::<B>(context)?;
        V3BlockSpec::new(args, layer, policy, expert_spec)?.instantiate::<B>(context)
    }

    /// Executes pre-norm attention and feed-forward residual sequencing.
    pub fn forward<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let Self {
            attention,
            feed_forward,
            input_norm,
            post_attention_norm,
        } = self;
        forward_v3_block(
            attention,
            input_norm,
            post_attention_norm,
            input,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::disabled(),
            |normalized, context, _| match feed_forward {
                V3FeedForward::Dense(mlp) => mlp.forward(normalized, context),
                V3FeedForward::Routed(moe) => {
                    moe.forward(normalized, RouteSource::Learned, context)
                }
            },
        )
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

    /// Observes the ordinary resident TP path. Dense writes are already reduced;
    /// shared sparse writes supply declared additive terms before the fused sum.
    pub fn forward_parallel_observed<C, O, F>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        forward_v3_block_reduced(
            &mut self.attention,
            &mut self.input_norm,
            &mut self.post_attention_norm,
            input,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            reduce,
            |input, context, instrumentation, reduce| match &mut self.feed_forward {
                V3FeedForward::Dense(mlp) => {
                    let partial =
                        mlp.forward_partial_instrumented(input, context, instrumentation)?;
                    let write =
                        instrumentation.apply("feed_forward.write", reduce(partial, context)?)?;
                    instrumentation.apply("feed_forward.output", write)
                }
                V3FeedForward::Routed(moe) => {
                    let output = moe.forward_tensor_parallel_resident_observed(
                        &format!("{path}.feed_forward"),
                        input,
                        RouteSource::Learned,
                        ExpertPass::Decode,
                        context,
                        instrumentation
                            .observer()
                            .expect("observed V3 block retains its observer"),
                        reduce,
                    )?;
                    instrumentation.apply("feed_forward.contribution", output)
                }
            },
        )
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
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
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

    /// Provider-aware TP hooks at the actual component boundaries. The shared
    /// branch emits additive write terms; the existing fused reduction supplies
    /// the complete feed-forward output before the residual addition.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<C, P, O, F>(
        &mut self,
        path: &str,
        input: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: Option<&mut C>,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        forward_v3_block_reduced(
            &mut self.attention,
            &mut self.input_norm,
            &mut self.post_attention_norm,
            input,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            reduce,
            |input, context, instrumentation, reduce| match &mut self.feed_forward {
                V3FeedForward::Dense(mlp) => {
                    let partial =
                        mlp.forward_partial_instrumented(input, context, instrumentation)?;
                    let write =
                        instrumentation.apply("feed_forward.write", reduce(partial, context)?)?;
                    instrumentation.apply("feed_forward.output", write)
                }
                V3FeedForward::Routed(moe) => {
                    let output = moe.forward_tensor_parallel_with_provider_observed(
                        &format!("{path}.feed_forward"),
                        input,
                        RouteSource::Learned,
                        pass,
                        provider,
                        context,
                        instrumentation
                            .observer()
                            .expect("observed V3 block retains its observer"),
                        reduce,
                    )?;
                    instrumentation.apply("feed_forward.contribution", output)
                }
            },
        )
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
        let output = self.forward_internal_observed_with_provider(
            path, &input, mask, cache, pass, provider, context, observer,
        )?;
        observe_and_intervene(observer, &format!("{path}.output"), &output)
    }

    /// Internal hooks; the traversal owns the unit input and output observations.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_internal_observed_with_provider<C, O, P>(
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
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        forward_v3_block(
            &mut self.attention,
            &mut self.input_norm,
            &mut self.post_attention_norm,
            input,
            mask,
            cache,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |normalized, context, instrumentation| match &mut self.feed_forward {
                V3FeedForward::Dense(mlp) => {
                    mlp.forward_instrumented(normalized, context, instrumentation)
                }
                V3FeedForward::Routed(moe) => {
                    let output = moe.forward_with_provider_observed(
                        &format!("{path}.feed_forward"),
                        normalized,
                        RouteSource::Learned,
                        pass,
                        provider,
                        context,
                        instrumentation
                            .observer()
                            .expect("observed V3 block retains its observer"),
                    )?;
                    instrumentation.apply("feed_forward.contribution", output)
                }
            },
        )
    }
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
