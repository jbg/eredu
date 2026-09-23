//! Embedded prediction layers shared by target capture and speculative draft
//! execution.

use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, Error, GroupedNeuralBackend, HyperHead,
    HyperHeadSpec, HyperNeuralBackend, LinearOperator, LinearSpec, NeuralBackend,
    NormalizationConstructionSpec, NormalizationOperator, ParameterSpec, Parameterized,
    PoolingAttentionCache, Tensor,
};
use eredu_runtime::{ExpertPass, RoutedExpertProvider};

use crate::decoder::ComponentInstrumentation;

use super::{block::V3Block, block::V4Block, V3Args, V4Args};

/// Borrowed input selecting target execution or one embedded prediction depth.
pub enum EmbeddedInput<'a, T> {
    /// Execute all target layers and retain their final hidden capture.
    Target {
        /// Token ids shaped `[batch, sequence]`.
        tokens: &'a T,
        /// Optional caller-prepared target attention mask.
        mask: Option<&'a T>,
    },
    /// Execute exactly one appended prediction group from a target capture.
    Draft {
        /// Token ids embedded for the proposed position.
        tokens: &'a T,
        /// Target or prior prediction hidden state.
        hidden: &'a T,
        /// Zero-based embedded prediction depth.
        depth: usize,
    },
    /// Commit target captures into every DSpark draft-layer attention cache.
    DsparkContext {
        /// Concatenated target captures in configured layer order.
        captures: &'a T,
    },
    /// Execute one fused DSpark proposal block.
    DsparkProposal {
        /// One anchor token shaped `[batch, 1]`.
        anchor: &'a T,
        /// Number of positions in the proposed block.
        capacity: usize,
    },
}

impl<'a, T> EmbeddedInput<'a, T> {
    /// Creates target-model input.
    pub const fn target(tokens: &'a T, mask: Option<&'a T>) -> Self {
        Self::Target { tokens, mask }
    }

    /// Creates one embedded prediction input.
    pub const fn draft(tokens: &'a T, hidden: &'a T, depth: usize) -> Self {
        Self::Draft {
            tokens,
            hidden,
            depth,
        }
    }

    /// Creates a DSpark context-commit input.
    pub const fn dspark_context(captures: &'a T) -> Self {
        Self::DsparkContext { captures }
    }

    /// Creates a fused DSpark block-proposal input.
    pub const fn dspark_proposal(anchor: &'a T, capacity: usize) -> Self {
        Self::DsparkProposal { anchor, capacity }
    }
}

/// Execution selection retained in a DeepSeek forward context.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ForwardMode {
    /// Target layers execute and prediction groups are bypassed.
    Target,
    /// Exactly one zero-based prediction depth executes.
    Draft(usize),
    /// Populate every DSpark draft attention cache from target captures.
    DsparkContext,
    /// Run every DSpark draft block over an anchor/noise proposal tensor.
    DsparkProposal,
}

/// Small allocation-free iterator over request-local tensors retained through
/// one layered submission.
pub struct RetainedValues<'a, T> {
    values: [Option<&'a T>; 6],
    next: usize,
    extras: Option<std::slice::Iter<'a, Option<T>>>,
}

impl<'a, T> RetainedValues<'a, T> {
    /// Creates an iterator from the fixed DeepSeek forward-value slots.
    pub const fn new(values: [Option<&'a T>; 6]) -> Self {
        Self {
            values,
            next: 0,
            extras: None,
        }
    }

    /// Appends a borrowed set of optional capture tensors.
    pub fn with_extras(mut self, extras: &'a [Option<T>]) -> Self {
        self.extras = Some(extras.iter());
        self
    }
}

impl<'a, T> Iterator for RetainedValues<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.next < self.values.len() {
            let value = self.values[self.next];
            self.next += 1;
            if value.is_some() {
                return value;
            }
        }
        self.extras
            .as_mut()
            .and_then(|extras| extras.find_map(Option::as_ref))
    }
}

/// One embedded prediction result retained by the speculative runtime.
#[derive(Debug, Clone)]
pub struct PredictionOutput<T> {
    /// Vocabulary logits for the proposed position.
    pub logits: T,
    /// Hidden state passed to the next prediction depth.
    pub hidden: T,
    /// Token tensor that conditioned this prediction.
    pub tokens: T,
}

/// One V3/R1 embedded prediction layer.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct V3PredictionLayer<
    B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend,
> {
    embedding_norm: B::Normalization,
    hidden_norm: B::Normalization,
    fusion: B::Linear,
    pub(crate) decoder: V3Block<B>,
    output_norm: B::Normalization,
    output_head: B::Linear,
}

impl<B: GroupedNeuralBackend + eredu_nn::DistributedNeuralBackend + BlockwiseAttentionBackend>
    V3PredictionLayer<B>
{
    /// Builds one unloaded V3 prediction depth.
    pub fn new(
        args: &V3Args,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let count = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        if depth >= count {
            return Err(Error::backend(format!(
                "V3 prediction depth {depth} is outside {count} layers"
            )));
        }
        let global = usize::try_from(args.num_hidden_layers).map_err(Error::backend)? + depth;
        let root = format!("model.layers.{global}");
        let norm = |name: String| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    parameter(name)?,
                ),
                context,
            )
        };
        Ok(Self {
            embedding_norm: norm(format!("{root}.enorm.weight"))?,
            hidden_norm: norm(format!("{root}.hnorm.weight"))?,
            fusion: linear::<B>(
                format!("{root}.eh_proj.weight"),
                2 * args.hidden_size,
                args.hidden_size,
                args.linear_format_for(&format!("{root}.eh_proj.weight")),
                context,
            )?,
            decoder: V3Block::new_prediction(args, global, context)?,
            output_norm: norm(format!("{root}.shared_head.norm.weight"))?,
            output_head: linear::<B>(
                format!("{root}.shared_head.head.weight"),
                args.hidden_size,
                args.vocab_size,
                args.linear_format_for(&format!("{root}.shared_head.head.weight")),
                context,
            )?,
        })
    }

    /// Executes one prediction depth over the target capture and current token
    /// embedding.
    pub fn forward<C: CompressedAttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionOutput<B::Tensor>, Error> {
        self.forward_with_decoder(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| decoder.forward(fused, None, Some(cache), context),
        )
    }

    /// Executes one V3 prediction depth with runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_with_decoder(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| {
                decoder.forward_with_provider(fused, None, Some(cache), pass, provider, context)
            },
        )
    }

    /// Executes a tensor-partitioned V3 prediction block while retaining its
    /// checkpoint-owned replicated prediction head.
    pub fn forward_parallel<C, F>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_with_decoder(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| {
                decoder.forward_parallel(fused, None, Some(cache), context, reduce)
            },
        )
    }

    /// Tensor-partitioned V3 prediction with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<C, P, F>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_with_decoder(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| {
                decoder.forward_parallel_with_provider(
                    fused,
                    None,
                    Some(cache),
                    pass,
                    provider,
                    context,
                    reduce,
                )
            },
        )
    }

    /// Observes actual prediction fusion, decoder and head values. The caller
    /// supplies the canonical unit path and owns speculative phase attribution.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<C, P, O>(
        &mut self,
        path: &str,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_decoder(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |decoder, fused, context, instrumentation| {
                decoder.forward_internal_observed_with_provider(
                    path,
                    fused,
                    None,
                    Some(cache),
                    pass,
                    provider,
                    context,
                    instrumentation
                        .observer()
                        .expect("observed prediction retains observer"),
                )
            },
        )
    }

    /// Observes the existing resident TP decoder with a replicated prediction head.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed<C, O, F>(
        &mut self,
        path: &str,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: F,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_decoder(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |decoder, fused, context, instrumentation| {
                decoder.forward_parallel_observed(
                    path,
                    fused,
                    None,
                    Some(cache),
                    context,
                    instrumentation
                        .observer()
                        .expect("observed prediction retains observer"),
                    reduce,
                )
            },
        )
    }

    /// Provider-aware TP prediction hooks reuse the target block's ordinary
    /// reductions and routed expert observations.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<C, P, O, F>(
        &mut self,
        path: &str,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: F,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_decoder(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |decoder, fused, context, instrumentation| {
                decoder.forward_parallel_observed_with_provider(
                    path,
                    fused,
                    None,
                    Some(cache),
                    pass,
                    provider,
                    context,
                    instrumentation
                        .observer()
                        .expect("observed prediction retains observer"),
                    reduce,
                )
            },
        )
    }

    fn forward_with_decoder<F>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        decoder: F,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        F: FnOnce(
            &mut V3Block<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let embedded = instrumentation.apply(
            "prediction.embedding.normalized",
            self.embedding_norm.forward(embedded, context)?,
        )?;
        let hidden = instrumentation.apply(
            "prediction.hidden.normalized",
            self.hidden_norm.forward(hidden, context)?,
        )?;
        let fused = B::Tensor::concatenate(&[embedded, hidden], -1, context)?;
        let fused = instrumentation.project::<B>(
            "prediction.fusion.input",
            &mut self.fusion,
            &fused,
            None,
            context,
        )?;
        let fused = instrumentation.apply("prediction.fusion.output", fused)?;
        let hidden = decoder(&mut self.decoder, &fused, context, instrumentation)?;
        let hidden = instrumentation.apply("prediction.readout.residual", hidden)?;
        let normalized = instrumentation.apply(
            "prediction.readout.normalized",
            self.output_norm.forward(&hidden, context)?,
        )?;
        let logits = instrumentation.project::<B>(
            "prediction.readout.projection_input",
            &mut self.output_head,
            &normalized,
            None,
            context,
        )?;
        let logits = instrumentation.apply("prediction.readout.linear", logits)?;
        Ok(PredictionOutput {
            logits,
            hidden,
            tokens: tokens.clone(),
        })
    }
}

/// One V4 embedded prediction layer reusing the ordinary local-attention mHC
/// decoder block.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct V4PredictionLayer<B>
where
    B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend,
{
    embedding_projection: B::Linear,
    hidden_projection: B::Linear,
    embedding_norm: B::Normalization,
    hidden_norm: B::Normalization,
    pub(crate) decoder: V4Block<B>,
    output_norm: B::Normalization,
    hyper_head: HyperHead<B>,
}

impl<B> V4PredictionLayer<B>
where
    B: HyperNeuralBackend + eredu_nn::DistributedNeuralBackend + GroupedNeuralBackend,
{
    /// Builds one unloaded V4 prediction depth.
    pub fn new(
        args: &V4Args,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        if args.dspark.is_some() {
            return Err(Error::backend(
                "fused DSpark checkpoints do not expose sequential V4 prediction layers",
            ));
        }
        let count = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?;
        if depth >= count {
            return Err(Error::backend(format!(
                "V4 prediction depth {depth} is outside {count} layers"
            )));
        }
        let global = usize::try_from(args.num_hidden_layers).map_err(Error::backend)? + depth;
        let root = format!("mtp.{depth}");
        let norm = |name: String| {
            B::normalization(
                NormalizationConstructionSpec::learned(
                    args.hidden_size,
                    args.rms_norm_eps,
                    parameter(name)?,
                ),
                context,
            )
        };
        Ok(Self {
            embedding_projection: linear::<B>(
                format!("{root}.e_proj.weight"),
                args.hidden_size,
                args.hidden_size,
                args.linear_format_for(&format!("{root}.e_proj.weight")),
                context,
            )?,
            hidden_projection: linear::<B>(
                format!("{root}.h_proj.weight"),
                args.hidden_size,
                args.hidden_size,
                args.linear_format_for(&format!("{root}.h_proj.weight")),
                context,
            )?,
            embedding_norm: norm(format!("{root}.enorm.weight"))?,
            hidden_norm: norm(format!("{root}.hnorm.weight"))?,
            decoder: V4Block::new_at(args, global, &root, None, context)?,
            output_norm: norm(format!("{root}.norm.weight"))?,
            hyper_head: HyperHead::new(
                HyperHeadSpec {
                    streams: args.hc_mult,
                    hidden_size: args.hidden_size,
                    norm_epsilon: args.rms_norm_eps,
                    epsilon: args.hc_eps,
                    function: parameter(format!("{root}.hc_head_fn"))?,
                    base: parameter(format!("{root}.hc_head_base"))?,
                    scale: parameter(format!("{root}.hc_head_scale"))?,
                },
                context,
            )?,
        })
    }

    /// Executes one V4 prediction depth using the target model's shared
    /// vocabulary head.
    pub fn forward<C: PoolingAttentionCache<B::Tensor>>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        output_head: &mut B::Linear,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionOutput<B::Tensor>, Error> {
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| decoder.forward(fused, tokens, None, Some(cache), context),
            |normalized, context, _| output_head.forward(normalized, context),
        )
    }

    /// Executes one V4 prediction depth with runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_provider<C, P>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        output_head: &mut B::Linear,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| {
                decoder.forward_with_provider(
                    fused,
                    tokens,
                    None,
                    Some(cache),
                    pass,
                    provider,
                    context,
                )
            },
            |normalized, context, _| output_head.forward(normalized, context),
        )
    }

    /// Executes a tensor-partitioned sequential predictor and delegates the
    /// vocabulary projection to the distributed composition.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel<C, R, H>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        reduce: R,
        mut project: H,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        H: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| {
                decoder.forward_parallel(fused, tokens, None, Some(cache), context, reduce)
            },
            |normalized, context, _| project(normalized, context),
        )
    }

    /// Tensor-partitioned sequential predictor with supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_with_provider<C, P, R, H>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        reduce: R,
        mut project: H,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        H: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::disabled(),
            |decoder, fused, context, _| {
                decoder.forward_parallel_with_provider(
                    fused,
                    tokens,
                    None,
                    Some(cache),
                    pass,
                    provider,
                    context,
                    reduce,
                )
            },
            |normalized, context, _| project(normalized, context),
        )
    }
    /// Observes the separate prediction projections, decoder streams and actual head input.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed<C, O>(
        &mut self,
        path: &str,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        output_head: &mut B::Linear,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |decoder, fused, context, instrumentation| {
                decoder.forward_observed(
                    &format!("{path}.decoder"),
                    fused,
                    tokens,
                    None,
                    Some(cache),
                    context,
                    instrumentation.observer().expect("observed V4 prediction"),
                )
            },
            |normalized, context, instrumentation| {
                instrumentation.project::<B>(
                    "projection_input",
                    output_head,
                    normalized,
                    None,
                    context,
                )
            },
        )
    }

    /// Observes the separate prediction projections, decoder streams and actual head input.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_observed_with_provider<C, O, P>(
        &mut self,
        path: &str,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        output_head: &mut B::Linear,
        pass: ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |decoder, fused, context, instrumentation| {
                decoder.forward_observed_with_provider(
                    &format!("{path}.decoder"),
                    fused,
                    tokens,
                    None,
                    Some(cache),
                    pass,
                    provider,
                    context,
                    instrumentation.observer().expect("observed V4 prediction"),
                )
            },
            |normalized, context, instrumentation| {
                instrumentation.project::<B>(
                    "projection_input",
                    output_head,
                    normalized,
                    None,
                    context,
                )
            },
        )
    }

    /// Observes the separate prediction projections, decoder streams and actual head input.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed<C, O, R>(
        &mut self,
        path: &str,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        output_head: &mut B::Linear,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: R,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |decoder, fused, context, instrumentation| {
                decoder.forward_parallel_observed(
                    &format!("{path}.decoder"),
                    fused,
                    tokens,
                    None,
                    Some(cache),
                    context,
                    instrumentation.observer().expect("observed V4 prediction"),
                    reduce,
                )
            },
            |normalized, context, instrumentation| {
                instrumentation.project_vocabulary::<B>(
                    "projection_input",
                    output_head,
                    normalized,
                    parallel,
                    context,
                )
            },
        )
    }

    /// Observes the separate prediction projections, decoder streams and actual head input.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_parallel_observed_with_provider<C, O, P, R>(
        &mut self,
        path: &str,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        output_head: &mut B::Linear,
        pass: ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        reduce: R,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: eredu_runtime::TensorParallelRoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let mut borrowed = eredu_runtime::BorrowedActivationObserver(observer);
        self.forward_with_execution(
            hidden,
            embedded,
            tokens,
            context,
            &mut ComponentInstrumentation::new(path, &mut borrowed),
            |decoder, fused, context, instrumentation| {
                decoder.forward_parallel_observed_with_provider(
                    &format!("{path}.decoder"),
                    fused,
                    tokens,
                    None,
                    Some(cache),
                    pass,
                    provider,
                    context,
                    instrumentation.observer().expect("observed V4 prediction"),
                    reduce,
                )
            },
            |normalized, context, instrumentation| {
                instrumentation.project_vocabulary::<B>(
                    "projection_input",
                    output_head,
                    normalized,
                    parallel,
                    context,
                )
            },
        )
    }

    fn forward_with_execution<D, H>(
        &mut self,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
        instrumentation: &mut ComponentInstrumentation<'_, B::Tensor>,
        execute: D,
        project: H,
    ) -> Result<PredictionOutput<B::Tensor>, Error>
    where
        D: FnOnce(
            &mut V4Block<B>,
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
        H: FnOnce(
            &B::Tensor,
            &<B::Tensor as Tensor>::Context,
            &mut ComponentInstrumentation<'_, B::Tensor>,
        ) -> Result<B::Tensor, Error>,
    {
        let effective_input = if instrumentation.enabled() {
            Some(instrumentation.apply("input", hidden.clone())?)
        } else {
            None
        };
        let hidden = effective_input.as_ref().unwrap_or(hidden);
        instrumentation.observe("capture", hidden)?;
        instrumentation.observe("prediction.embedding", embedded)?;
        instrumentation.observe("prediction.hidden", hidden)?;
        let embedded = instrumentation.apply(
            "prediction.embedding.normalized",
            self.embedding_norm.forward(embedded, context)?,
        )?;
        let hidden = instrumentation.apply(
            "prediction.hidden.normalized",
            self.hidden_norm.forward(hidden, context)?,
        )?;
        let embedded = instrumentation.project::<B>(
            "prediction.embedding.projection_input",
            &mut self.embedding_projection,
            &embedded,
            None,
            context,
        )?;
        let embedded = instrumentation
            .apply("prediction.embedding.projected", embedded)?
            .expand_dims(2, context)?
            .broadcast_to(hidden.shape(), context)?;
        let hidden = instrumentation.project::<B>(
            "prediction.hidden.projection_input",
            &mut self.hidden_projection,
            &hidden,
            None,
            context,
        )?;
        let hidden = instrumentation.apply("prediction.hidden.projected", hidden)?;
        let fused =
            instrumentation.apply("prediction.fusion.output", embedded.add(&hidden, context)?)?;
        let hidden = execute(&mut self.decoder, &fused, context, instrumentation)?;
        let logits = instrumentation.with_scope("prediction.readout", |instrumentation| {
            let effective = if instrumentation.enabled() {
                Some(instrumentation.apply("streams", hidden.clone())?)
            } else {
                None
            };
            let collapsed = instrumentation.collapse_streams::<B>(
                "stream_coefficients",
                &mut self.hyper_head,
                effective.as_ref().unwrap_or(&hidden),
                context,
            )?;
            let normalized =
                instrumentation.normalize_readout(&collapsed, &mut self.output_norm, context)?;
            let logits = project(&normalized, context, instrumentation)?;
            instrumentation.apply("linear", logits)
        })?;
        Ok(PredictionOutput {
            logits,
            hidden: instrumentation.apply("output", hidden)?,
            tokens: tokens.clone(),
        })
    }
}

fn linear<B: NeuralBackend + eredu_nn::DistributedNeuralBackend>(
    name: impl Into<String>,
    input: i32,
    output: i32,
    format: eredu_checkpoint::LinearFormat,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, Error> {
    let name = name.into();
    B::linear(
        LinearSpec {
            input,
            output,
            weight: parameter(&name)?,
            bias: None,
            format: crate::linear_format::standard_linear_format(&name, format)?,
        },
        context,
    )
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}
