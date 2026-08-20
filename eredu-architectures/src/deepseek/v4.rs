//! Thin DeepSeek-V4 architecture policy.

use std::num::NonZeroU32;

use eredu_core::{
    cache::{
        LayerCachePolicy, MutableStateResidency, PoolingStateComponent, StateResidencyClass,
        StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
    },
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::{
    EmbeddingOperator, EmbeddingSpec, Error, HyperHead, HyperHeadSpec, HyperNeuralBackend,
    LinearOperator, LinearSpec, NormalizationOperator, NormalizationSpec, ParameterSpec,
    Parameterized, PoolingAttentionCache, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{LayerRuntimeState, LayeredArchitecture, LayeredForwardState, StateLayout};

use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat};

use crate::decoder::{
    SequentialPredictionGroups, StaticModuleSpec, StaticModules as TextStaticModules,
};

use super::{
    block::V4Block,
    moe::MoePolicy,
    mtp::{EmbeddedInput, ForwardMode, RetainedValues, V4PredictionLayer},
    DsparkConfig, ExpertFormat, V4Args, V4AttentionPolicy,
};

/// Pinned DSpark projections and heads shared by its ordinary draft blocks.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct DsparkStatic<B: HyperNeuralBackend> {
    main_projection: B::Linear,
    main_norm: B::Normalization,
    output_norm: B::Normalization,
    hyper_head: HyperHead<B>,
    markov_embedding: B::Embedding,
    markov_output: B::Linear,
    confidence_head: B::Linear,
}

/// One target, sequential MTP, or fused DSpark execution unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    /// Ordinary target decoder block.
    Target(V4Block<B>),
    /// One sequential embedded prediction layer.
    Prediction(V4PredictionLayer<B>),
    /// One ordinary local-attention block in the fused DSpark chain.
    Dspark(V4Block<B>),
}

/// V4 pinned modules shared by resident and bounded layer execution.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub struct StaticModules<B: HyperNeuralBackend> {
    /// Shared embedding, final normalization, and vocabulary head lifecycle.
    pub text: TextStaticModules<B>,
    /// Learned collapse from hyper-connection streams to final hidden state.
    pub hyper_head: HyperHead<B>,
    /// Optional fused-drafter projections and heads.
    pub dspark: Option<DsparkStatic<B>>,
}

/// V4 values retained for one target-model forward.
pub struct ForwardContext<T> {
    input_ids: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    target_capture: Option<T>,
    draft_logits: Option<T>,
    draft_hidden: Option<T>,
    captures: Vec<Option<T>>,
}

impl<T> ForwardContext<T> {
    /// Borrows the final target hidden state or configured DSpark captures.
    pub const fn target_capture(&self) -> Option<&T> {
        self.target_capture.as_ref()
    }

    /// Borrows logits emitted by a sequential or fused draft pass.
    pub const fn draft_logits(&self) -> Option<&T> {
        self.draft_logits.as_ref()
    }

    /// Borrows the hidden state emitted by sequential prediction execution.
    pub const fn draft_hidden(&self) -> Option<&T> {
        self.draft_hidden.as_ref()
    }
}

/// Thin V4 target-model architecture over the shared layered runtime.
pub struct Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    args: V4Args,
    static_modules: StaticModules<B>,
    groups: SequentialPredictionGroups,
}

impl<B> Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
{
    /// Builds unloaded pinned V4 modules.
    pub fn new(args: V4Args, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let text = TextStaticModules::from_spec(
            StaticModuleSpec {
                embedding_weight: "embed.weight".into(),
                normalization_weight: "norm.weight".into(),
                head_weight: "head.weight".into(),
                vocabulary: args.vocab_size,
                hidden_size: args.hidden_size,
                normalization_epsilon: args.rms_norm_eps,
                embedding_quantization: None,
                head_format: args.linear_format_for("head.weight"),
                tied_head: false,
            },
            context,
        )?;
        let hyper_head = HyperHead::new(
            HyperHeadSpec {
                streams: args.hc_mult,
                hidden_size: args.hidden_size,
                norm_epsilon: args.rms_norm_eps,
                epsilon: args.hc_eps,
                function: parameter("hc_head_fn")?,
                base: parameter("hc_head_base")?,
                scale: parameter("hc_head_scale")?,
            },
            context,
        )?;
        let dspark = args
            .dspark
            .as_ref()
            .map(|config| DsparkStatic::new(&args, config, context))
            .transpose()?;
        Ok(Self {
            groups: SequentialPredictionGroups::new(
                "layers",
                usize::try_from(args.num_hidden_layers).map_err(Error::backend)?,
                (0..usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?)
                    .map(|depth| format!("mtp.{depth}")),
            )?,
            args,
            static_modules: StaticModules {
                text,
                hyper_head,
                dspark,
            },
        })
    }

    /// Returns the normalized V4 arguments.
    pub const fn args(&self) -> &V4Args {
        &self.args
    }

    /// Borrows pinned modules for checkpoint binding.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned modules for checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Replaces the pinned module set after a backend composition has loaded
    /// exactly the static parameters owned by its placement.
    pub fn replace_static_modules(&mut self, static_modules: StaticModules<B>) {
        self.static_modules = static_modules;
    }

    /// Embeds tokens and broadcasts them across hyper-connection streams for
    /// a pipeline-partitioned target pass.
    pub fn pipeline_embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(tokens, context)?;
        broadcast_streams::<B>(&embedded, &self.args, context)
    }

    /// Broadcasts a composition-owned embedding shard result across the
    /// model's replicated hyper-connection streams.
    pub fn pipeline_broadcast_embedding(
        &self,
        embedded: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        broadcast_streams::<B>(embedded, &self.args, context)
    }

    /// Executes one V4 target block in a pipeline-owned layer range and
    /// returns an optional DSpark capture for that global layer.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_target<C>(
        &mut self,
        layer: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
    {
        let Unit::Target(block) = unit else {
            return Err(Error::backend(
                "a V4 draft unit cannot execute in the target pipeline range",
            ));
        };
        let output = block.forward(hidden, input_ids, mask, Some(cache), context)?;
        let capture = self
            .args
            .dspark
            .as_ref()
            .and_then(|config| {
                config
                    .target_layer_ids
                    .iter()
                    .any(|wanted| usize::try_from(*wanted).ok() == Some(layer))
                    .then(|| B::Tensor::mean_axis(&output, 2, false, context))
            })
            .transpose()?;
        Ok((output, capture))
    }

    /// Executes one V4 pipeline target block with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_target_with_provider<C, P>(
        &mut self,
        layer: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let Unit::Target(block) = unit else {
            return Err(Error::backend(
                "a V4 draft unit cannot execute in the target pipeline range",
            ));
        };
        let output = block.forward_with_provider(
            hidden,
            input_ids,
            mask,
            Some(cache),
            pass,
            provider,
            context,
        )?;
        let capture = self
            .args
            .dspark
            .as_ref()
            .and_then(|config| {
                config
                    .target_layer_ids
                    .iter()
                    .any(|wanted| usize::try_from(*wanted).ok() == Some(layer))
                    .then(|| B::Tensor::mean_axis(&output, 2, false, context))
            })
            .transpose()?;
        Ok((output, capture))
    }

    /// Collapses hyper streams, applies the final norm, and projects logits on
    /// the final pipeline stage.
    pub fn pipeline_finish(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.hyper_head.forward(hidden, context)?;
        let hidden = self.static_modules.text.norm.forward(&hidden, context)?;
        self.static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head")
            .forward(&hidden, context)
    }

    /// Collapses hyper streams and applies the final target normalization
    /// without projecting vocabulary logits.
    pub fn pipeline_finish_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.hyper_head.forward(hidden, context)?;
        self.static_modules.text.norm.forward(&hidden, context)
    }

    /// Executes one tensor-partitioned V4 target block.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_target_parallel<C, F>(
        &mut self,
        layer: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let Unit::Target(block) = unit else {
            return Err(Error::backend(
                "a V4 draft unit cannot execute in the target pipeline range",
            ));
        };
        let output =
            block.forward_parallel(hidden, input_ids, mask, Some(cache), context, reduce)?;
        let capture = self
            .args
            .dspark
            .as_ref()
            .and_then(|config| {
                config
                    .target_layer_ids
                    .iter()
                    .any(|wanted| usize::try_from(*wanted).ok() == Some(layer))
                    .then(|| B::Tensor::mean_axis(&output, 2, false, context))
            })
            .transpose()?;
        Ok((output, capture))
    }

    /// Tensor-partitioned V4 target execution with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_target_parallel_with_provider<C, P, F>(
        &mut self,
        layer: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        input_ids: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<(B::Tensor, Option<B::Tensor>), Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let Unit::Target(block) = unit else {
            return Err(Error::backend(
                "a V4 draft unit cannot execute in the target pipeline range",
            ));
        };
        let output = block.forward_parallel_with_provider(
            hidden,
            input_ids,
            mask,
            Some(cache),
            pass,
            provider,
            context,
            reduce,
        )?;
        let capture = self
            .args
            .dspark
            .as_ref()
            .and_then(|config| {
                config
                    .target_layer_ids
                    .iter()
                    .any(|wanted| usize::try_from(*wanted).ok() == Some(layer))
                    .then(|| B::Tensor::mean_axis(&output, 2, false, context))
            })
            .transpose()?;
        Ok((output, capture))
    }

    /// Executes one sequential embedded-prediction unit owned by the final
    /// pipeline stage.
    pub fn pipeline_forward_prediction<C>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
    {
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(tokens, context)?;
        let output_head = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head");
        match unit {
            Unit::Prediction(unit) => {
                unit.forward(hidden, &embedded, tokens, cache, output_head, context)
            }
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes one sequential embedded-prediction unit with runtime-supplied
    /// routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_with_provider<C, P>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(tokens, context)?;
        let output_head = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head");
        match unit {
            Unit::Prediction(unit) => unit.forward_with_provider(
                hidden,
                &embedded,
                tokens,
                cache,
                output_head,
                pass,
                provider,
                context,
            ),
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes a tensor-partitioned sequential predictor using
    /// composition-owned embedding and vocabulary shards.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_parallel<C, R, H>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        reduce: R,
        project: H,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        H: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        match unit {
            Unit::Prediction(unit) => {
                unit.forward_parallel(hidden, embedded, tokens, cache, context, reduce, project)
            }
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Tensor-partitioned sequential predictor with supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_parallel_with_provider<C, P, R, H>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        reduce: R,
        project: H,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        H: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel_with_provider(
                hidden, embedded, tokens, cache, pass, provider, context, reduce, project,
            ),
            Unit::Target(_) | Unit::Dspark(_) => Err(Error::backend(
                "a non-sequential V4 unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Rebuilds the fused DSpark context caches from concatenated target-layer
    /// captures. The caller controls transactionality by choosing the cache
    /// slice supplied here.
    pub fn pipeline_prefill_dspark_context<C, M>(
        &mut self,
        units: &mut [M],
        captures: &B::Tensor,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
    {
        if units.len() != caches.len() {
            return Err(Error::backend("DSpark unit/cache count mismatch"));
        }
        let dspark = self
            .static_modules
            .dspark
            .as_mut()
            .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
        let main = dspark
            .main_norm
            .forward(&dspark.main_projection.forward(captures, context)?, context)?;
        let hidden = broadcast_streams::<B>(&main, &self.args, context)?;
        for (unit, cache) in units.iter_mut().zip(caches) {
            let Unit::Dspark(unit) = unit.as_mut() else {
                return Err(Error::backend("DSpark context received a non-DSpark unit"));
            };
            unit.prefill_attention_cache(&hidden, cache, context)?;
        }
        Ok(())
    }

    /// Executes one transactional fused DSpark proposal block.
    pub fn pipeline_dspark_proposal<C, M>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
    {
        self.pipeline_dspark_proposal_inner(
            units,
            anchor,
            capacity,
            caches,
            context,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward(hidden, tokens, Some(mask), Some(cache), context)
            },
        )
    }

    /// Executes one transactional fused DSpark proposal block with
    /// runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_dspark_proposal_with_provider<C, M, P>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.pipeline_dspark_proposal_inner(
            units,
            anchor,
            capacity,
            caches,
            context,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward_with_provider(
                    hidden,
                    tokens,
                    Some(mask),
                    Some(cache),
                    pass,
                    provider,
                    context,
                )
            },
        )
    }

    /// Executes a tensor-partitioned fused DSpark proposal with
    /// composition-owned vocabulary embedding and output projection.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_dspark_proposal_parallel<C, M, E, R, H>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
        embed: E,
        mut reduce: R,
        project: H,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        E: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        H: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.pipeline_dspark_proposal_parallel_inner(
            units,
            anchor,
            capacity,
            caches,
            context,
            embed,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward_parallel(
                    hidden,
                    tokens,
                    Some(mask),
                    Some(cache),
                    context,
                    &mut reduce,
                )
            },
            project,
        )
    }

    /// Tensor-partitioned DSpark proposal with supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_dspark_proposal_parallel_with_provider<C, M, P, E, R, H>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        embed: E,
        mut reduce: R,
        project: H,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        E: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        R: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        H: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        self.pipeline_dspark_proposal_parallel_inner(
            units,
            anchor,
            capacity,
            caches,
            context,
            embed,
            |unit, hidden, tokens, mask, cache, context| {
                unit.forward_parallel_with_provider(
                    hidden,
                    tokens,
                    Some(mask),
                    Some(cache),
                    pass,
                    provider,
                    context,
                    &mut reduce,
                )
            },
            project,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline_dspark_proposal_parallel_inner<C, M, E, F, H>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
        mut embed: E,
        mut forward: F,
        mut project: H,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        E: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
        F: FnMut(
            &mut V4Block<B>,
            &B::Tensor,
            &B::Tensor,
            &B::Tensor,
            &mut C,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
        H: FnMut(&B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        let config = self
            .args
            .dspark
            .as_ref()
            .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
        if capacity == 0 || anchor.shape().len() != 2 || anchor.dim(1) != 1 {
            return Err(Error::backend(
                "DSpark proposal requires a positive capacity and [batch, 1] anchor",
            ));
        }
        if units.len() != caches.len() {
            return Err(Error::backend("DSpark unit/cache count mismatch"));
        }
        let input_ids = if capacity == 1 {
            anchor.clone()
        } else {
            let noise = B::Tensor::full_i32(
                config.noise_token_id,
                &[
                    anchor.dim(0),
                    i32::try_from(capacity - 1).map_err(Error::backend)?,
                ],
                context,
            )?;
            B::Tensor::concatenate(&[anchor.clone(), noise], 1, context)?
        };
        let embedded = embed(&input_ids, context)?;
        let mut hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
        for (unit, cache) in units.iter_mut().zip(caches) {
            let Unit::Dspark(unit) = unit.as_mut() else {
                return Err(Error::backend("DSpark proposal received a non-DSpark unit"));
            };
            let keys = (cache.offset() + i32::try_from(capacity).map_err(Error::backend)?)
                .min(self.args.sliding_window);
            let mask = B::Tensor::full_f32(
                0.0,
                &[i32::try_from(capacity).map_err(Error::backend)?, keys],
                context,
            )?;
            hidden = forward(unit, &hidden, &input_ids, &mask, cache, context)?;
        }
        let dspark = self
            .static_modules
            .dspark
            .as_mut()
            .expect("validated DSpark static modules");
        let collapsed = dspark.hyper_head.forward(&hidden, context)?;
        let normalized = dspark.output_norm.forward(&collapsed, context)?;
        let mut logits = project(&normalized, context)?;
        let markov = dspark.markov_embedding.forward(anchor, context)?;
        let adjustment = dspark.markov_output.forward(&markov, context)?;
        logits = logits.add(&adjustment.broadcast_to(logits.shape(), context)?, context)?;
        Ok(logits)
    }

    fn pipeline_dspark_proposal_inner<C, M, F>(
        &mut self,
        units: &mut [M],
        anchor: &B::Tensor,
        capacity: usize,
        caches: &mut [C],
        context: &<B::Tensor as Tensor>::Context,
        mut forward: F,
    ) -> Result<B::Tensor, Error>
    where
        C: PoolingAttentionCache<B::Tensor>,
        M: AsMut<Unit<B>>,
        F: FnMut(
            &mut V4Block<B>,
            &B::Tensor,
            &B::Tensor,
            &B::Tensor,
            &mut C,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<B::Tensor, Error>,
    {
        let config = self
            .args
            .dspark
            .as_ref()
            .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
        if capacity == 0 || anchor.shape().len() != 2 || anchor.dim(1) != 1 {
            return Err(Error::backend(
                "DSpark proposal requires a positive capacity and [batch, 1] anchor",
            ));
        }
        if units.len() != caches.len() {
            return Err(Error::backend("DSpark unit/cache count mismatch"));
        }
        let input_ids = if capacity == 1 {
            anchor.clone()
        } else {
            let noise = B::Tensor::full_i32(
                config.noise_token_id,
                &[
                    anchor.dim(0),
                    i32::try_from(capacity - 1).map_err(Error::backend)?,
                ],
                context,
            )?;
            B::Tensor::concatenate(&[anchor.clone(), noise], 1, context)?
        };
        let embedded = self
            .static_modules
            .text
            .embeddings
            .forward(&input_ids, context)?;
        let mut hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
        for (unit, cache) in units.iter_mut().zip(caches) {
            let Unit::Dspark(unit) = unit.as_mut() else {
                return Err(Error::backend("DSpark proposal received a non-DSpark unit"));
            };
            let keys = (cache.offset() + i32::try_from(capacity).map_err(Error::backend)?)
                .min(self.args.sliding_window);
            let mask = B::Tensor::full_f32(
                0.0,
                &[i32::try_from(capacity).map_err(Error::backend)?, keys],
                context,
            )?;
            hidden = forward(unit, &hidden, &input_ids, &mask, cache, context)?;
        }
        let dspark = self
            .static_modules
            .dspark
            .as_mut()
            .expect("validated DSpark static modules");
        let collapsed = dspark.hyper_head.forward(&hidden, context)?;
        let normalized = dspark.output_norm.forward(&collapsed, context)?;
        let mut logits = self
            .static_modules
            .text
            .lm_head
            .as_mut()
            .expect("validated V4 models have an untied output head")
            .forward(&normalized, context)?;
        let markov = dspark.markov_embedding.forward(anchor, context)?;
        let adjustment = dspark.markov_output.forward(&markov, context)?;
        logits = logits.add(&adjustment.broadcast_to(logits.shape(), context)?, context)?;
        Ok(logits)
    }

    /// Executes one target or prediction unit with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: PoolingAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => {
                let hidden = unit.forward_with_provider(
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    pass,
                    provider,
                    context,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        forward.captures[position] =
                            Some(B::Tensor::mean_axis(&hidden, 2, false, context)?);
                    }
                }
                Ok(hidden)
            }
            Unit::Prediction(unit) if group > 0 => {
                let output_head = self
                    .static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head");
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    output_head,
                    pass,
                    provider,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            Unit::Dspark(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        Ok(hidden.clone())
                    }
                    ForwardMode::DsparkProposal => unit.forward_with_provider(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        pass,
                        provider,
                        context,
                    ),
                    _ => Err(Error::backend("DSpark unit selected outside DSpark mode")),
                }
            }
            _ => Err(Error::backend(format!(
                "V4 execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one graph unit with stable target/MTP/DSpark observation and
    /// intervention points.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_observed<S, O>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match unit {
            Unit::Target(unit) if group == 0 => {
                let output = unit.forward_observed(
                    &format!("layers.{index}"),
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    context,
                    observer,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        let capture = B::Tensor::mean_axis(&output, 2, false, context)?;
                        observer
                            .observe(&format!("dspark.target_captures.{position}"), &capture)?;
                        forward.captures[position] = Some(capture);
                    }
                }
                Ok(output)
            }
            Unit::Prediction(_) | Unit::Dspark(_) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let output = <Self as LayeredArchitecture<B, S>>::forward_unit(
                    self, group, index, unit, hidden, state, forward, context,
                )?;
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output,
                )
            }
            _ => Err(Error::backend(format!(
                "V4 observed execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one observed graph unit with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_observed_with_provider<S, O, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, Error>
    where
        S: LayerRuntimeState<B>,
        S::LayerState: PoolingAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match unit {
            Unit::Target(unit) if group == 0 => {
                let output = unit.forward_observed_with_provider(
                    &format!("layers.{index}"),
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    pass,
                    provider,
                    context,
                    observer,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        let capture = B::Tensor::mean_axis(&output, 2, false, context)?;
                        observer
                            .observe(&format!("dspark.target_captures.{position}"), &capture)?;
                        forward.captures[position] = Some(capture);
                    }
                }
                Ok(output)
            }
            Unit::Prediction(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output_head = self
                    .static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head");
                let output = unit.forward_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    output_head,
                    pass,
                    provider,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output.hidden,
                )
            }
            Unit::Dspark(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                let output = match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        hidden.clone()
                    }
                    ForwardMode::DsparkProposal => unit.forward_with_provider(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        pass,
                        provider,
                        context,
                    )?,
                    _ => return Err(Error::backend("DSpark unit selected outside DSpark mode")),
                };
                eredu_runtime::observe_and_intervene(
                    observer,
                    &format!("mtp.{}.output", group - 1),
                    &output,
                )
            }
            _ => Err(Error::backend(format!(
                "V4 observed execution unit does not match group {group}"
            ))),
        }
    }
}

impl<B: HyperNeuralBackend> DsparkStatic<B> {
    fn new(
        args: &V4Args,
        config: &DsparkConfig,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        let last = usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)? - 1;
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
        Ok(Self {
            main_projection: projection::<B>(
                "mtp.0.main_proj.weight",
                args.hidden_size
                    * i32::try_from(config.target_layer_ids.len()).map_err(Error::backend)?,
                args.hidden_size,
                args.linear_format_for("mtp.0.main_proj.weight"),
                context,
            )?,
            main_norm: norm("mtp.0.main_norm.weight".into())?,
            output_norm: norm(format!("mtp.{last}.norm.weight"))?,
            hyper_head: HyperHead::new(
                HyperHeadSpec {
                    streams: args.hc_mult,
                    hidden_size: args.hidden_size,
                    norm_epsilon: args.rms_norm_eps,
                    epsilon: args.hc_eps,
                    function: parameter(format!("mtp.{last}.hc_head_fn"))?,
                    base: parameter(format!("mtp.{last}.hc_head_base"))?,
                    scale: parameter(format!("mtp.{last}.hc_head_scale"))?,
                },
                context,
            )?,
            markov_embedding: B::embedding(
                EmbeddingSpec {
                    vocabulary: args.vocab_size,
                    dimensions: config.markov_rank,
                    weight: parameter(format!("mtp.{last}.markov_head.markov_w1.weight"))?,
                    quantization: None,
                },
                context,
            )?,
            markov_output: projection::<B>(
                format!("mtp.{last}.markov_head.markov_w2.weight"),
                config.markov_rank,
                args.vocab_size,
                args.linear_format_for(&format!("mtp.{last}.markov_head.markov_w2.weight")),
                context,
            )?,
            confidence_head: projection::<B>(
                format!("mtp.{last}.confidence_head.proj.weight"),
                args.hidden_size + config.markov_rank,
                1,
                args.linear_format_for(&format!("mtp.{last}.confidence_head.proj.weight")),
                context,
            )?,
        })
    }
}

impl<B, S> LayeredArchitecture<B, S> for Model<B>
where
    B: HyperNeuralBackend + RoutedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: PoolingAttentionCache<B::Tensor>,
{
    type Input<'a> = EmbeddedInput<'a, B::Tensor>;
    type StaticModules = StaticModules<B>;
    type Unit = Unit<B>;
    type ForwardContext = ForwardContext<B::Tensor>;
    type RetainedContextValues<'a>
        = RetainedValues<'a, B::Tensor>
    where
        B::Tensor: 'a;
    type Error = Error;

    fn model_identity(&self) -> &str {
        &self.args.model_type
    }

    fn execution_graph(&self) -> Result<eredu_runtime::ExecutionGraph, Self::Error> {
        self.groups.execution_graph()
    }

    fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
        self.groups.unit_count(group)
    }

    fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
        self.groups.unit_path(group, index)
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
        self.groups.unit_count(group)?;
        if group == 0 {
            Ok(Unit::Target(V4Block::new(&self.args, index, context)?))
        } else if self.args.dspark.is_some() {
            let global =
                usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)? + group - 1;
            Ok(Unit::Dspark(V4Block::new_at(
                &self.args,
                global,
                &format!("mtp.{}", group - 1),
                context,
            )?))
        } else {
            Ok(Unit::Prediction(V4PredictionLayer::new(
                &self.args,
                group - 1,
                context,
            )?))
        }
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = state_layout(&self.args)?;
        if state.layout() != &expected {
            return Err(Error::backend(format!(
                "V4 runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let (input_ids, embedded, hidden, mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => {
                let embedded = self
                    .static_modules
                    .text
                    .embeddings
                    .forward(tokens, context)?;
                let hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
                (
                    tokens.clone(),
                    embedded,
                    hidden,
                    mask.cloned(),
                    ForwardMode::Target,
                )
            }
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if self.args.dspark.is_some() || depth >= self.groups.prediction_count() {
                    return Err(Error::backend(format!(
                        "V4 sequential prediction depth {depth} is unavailable"
                    )));
                }
                (
                    tokens.clone(),
                    self.static_modules
                        .text
                        .embeddings
                        .forward(tokens, context)?,
                    hidden.clone(),
                    None,
                    ForwardMode::Draft(depth),
                )
            }
            EmbeddedInput::DsparkContext { captures } => {
                let dspark = self
                    .static_modules
                    .dspark
                    .as_mut()
                    .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
                let main = dspark
                    .main_norm
                    .forward(&dspark.main_projection.forward(captures, context)?, context)?;
                let hidden = broadcast_streams::<B>(&main, &self.args, context)?;
                (
                    captures.clone(),
                    main,
                    hidden,
                    None,
                    ForwardMode::DsparkContext,
                )
            }
            EmbeddedInput::DsparkProposal { anchor, capacity } => {
                let config = self
                    .args
                    .dspark
                    .as_ref()
                    .ok_or_else(|| Error::backend("V4 checkpoint has no DSpark module"))?;
                if capacity == 0 || anchor.shape().len() != 2 || anchor.dim(1) != 1 {
                    return Err(Error::backend(
                        "DSpark proposal requires a positive capacity and [batch, 1] anchor",
                    ));
                }
                let input_ids = if capacity == 1 {
                    anchor.clone()
                } else {
                    let noise = B::Tensor::full_i32(
                        config.noise_token_id,
                        &[
                            anchor.dim(0),
                            i32::try_from(capacity - 1).map_err(Error::backend)?,
                        ],
                        context,
                    )?;
                    B::Tensor::concatenate(&[anchor.clone(), noise], 1, context)?
                };
                let embedded = self
                    .static_modules
                    .text
                    .embeddings
                    .forward(&input_ids, context)?;
                let hidden = broadcast_streams::<B>(&embedded, &self.args, context)?;
                let draft_start =
                    usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?;
                let offset = state.layer(draft_start).map_err(Error::backend)?.offset();
                let keys = (offset + i32::try_from(capacity).map_err(Error::backend)?)
                    .min(self.args.sliding_window);
                let mask = Some(B::Tensor::full_f32(
                    0.0,
                    &[i32::try_from(capacity).map_err(Error::backend)?, keys],
                    context,
                )?);
                (
                    input_ids,
                    embedded,
                    hidden,
                    mask,
                    ForwardMode::DsparkProposal,
                )
            }
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                input_ids,
                embedded,
                mask,
                mode,
                target_capture: None,
                draft_logits: None,
                draft_hidden: None,
                captures: self
                    .args
                    .dspark
                    .as_ref()
                    .map_or_else(Vec::new, |config| vec![None; config.target_layer_ids.len()]),
            },
        })
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
        self.groups.begin(group, initial, dependencies)
    }

    fn should_execute_group(&self, group: usize, forward: &Self::ForwardContext) -> bool {
        match forward.mode {
            ForwardMode::Target => group == 0,
            ForwardMode::Draft(depth) => group == depth + 1,
            ForwardMode::DsparkContext | ForwardMode::DsparkProposal => group > 0,
        }
    }

    fn forward_unit(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => {
                let hidden = unit.forward(
                    hidden,
                    &forward.input_ids,
                    forward.mask.as_ref(),
                    Some(state.layer(index).map_err(Error::backend)?),
                    context,
                )?;
                if let Some(config) = &self.args.dspark {
                    if let Some(position) = config
                        .target_layer_ids
                        .iter()
                        .position(|wanted| usize::try_from(*wanted).ok() == Some(index))
                    {
                        forward.captures[position] =
                            Some(B::Tensor::mean_axis(&hidden, 2, false, context)?);
                    }
                }
                Ok(hidden)
            }
            Unit::Prediction(unit) if group > 0 => {
                let output_head = self
                    .static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head");
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward(
                    hidden,
                    &forward.embedded,
                    &forward.input_ids,
                    state.layer(layer).map_err(Error::backend)?,
                    output_head,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            Unit::Dspark(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let cache = state.layer(layer).map_err(Error::backend)?;
                match forward.mode {
                    ForwardMode::DsparkContext => {
                        unit.prefill_attention_cache(hidden, cache, context)?;
                        Ok(hidden.clone())
                    }
                    ForwardMode::DsparkProposal => unit.forward(
                        hidden,
                        &forward.input_ids,
                        forward.mask.as_ref(),
                        Some(cache),
                        context,
                    ),
                    _ => Err(Error::backend("DSpark unit selected outside DSpark mode")),
                }
            }
            _ => Err(Error::backend(format!(
                "V4 execution unit does not match group {group}"
            ))),
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 && matches!(forward.mode, ForwardMode::Target) {
            forward.target_capture = if self.args.dspark.is_some() {
                Some(B::Tensor::concatenate(
                    &forward
                        .captures
                        .iter()
                        .map(|capture| {
                            capture.clone().ok_or_else(|| {
                                Error::backend("configured DSpark target capture was not produced")
                            })
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    -1,
                    context,
                )?)
            } else {
                Some(hidden.clone())
            };
        }
        if group == self.groups.prediction_count()
            && matches!(forward.mode, ForwardMode::DsparkProposal)
        {
            let dspark = self
                .static_modules
                .dspark
                .as_mut()
                .expect("DSpark proposal mode has pinned DSpark modules");
            let collapsed = dspark.hyper_head.forward(hidden, context)?;
            let normalized = dspark.output_norm.forward(&collapsed, context)?;
            let mut logits = self
                .static_modules
                .text
                .lm_head
                .as_mut()
                .expect("validated V4 models have an untied output head")
                .forward(&normalized, context)?;
            let anchor = forward.input_ids.index(
                &[eredu_nn::Index::Full, eredu_nn::Index::Range(0, 1)],
                context,
            )?;
            let markov = dspark.markov_embedding.forward(&anchor, context)?;
            let adjustment = dspark.markov_output.forward(&markov, context)?;
            let adjustment = adjustment.broadcast_to(logits.shape(), context)?;
            logits = logits.add(&adjustment, context)?;
            forward.draft_logits = Some(logits);
        }
        if group > 0 && matches!(forward.mode, ForwardMode::Draft(_)) {
            forward.draft_hidden = Some(hidden.clone());
        }
        Ok(hidden.clone())
    }

    fn finish_forward(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => {
                let hidden = self.static_modules.hyper_head.forward(hidden, context)?;
                let hidden = self.static_modules.text.norm.forward(&hidden, context)?;
                self.static_modules
                    .text
                    .lm_head
                    .as_mut()
                    .expect("validated V4 models have an untied output head")
                    .forward(&hidden, context)
            }
            ForwardMode::Draft(_) | ForwardMode::DsparkProposal => forward
                .draft_logits
                .clone()
                .ok_or_else(|| Error::backend("V4 draft group produced no logits")),
            ForwardMode::DsparkContext => Ok(hidden.clone()),
        }
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.input_ids),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_capture.as_ref(),
            forward.draft_logits.as_ref(),
            forward.draft_hidden.as_ref(),
        ])
        .with_extras(&forward.captures)
    }
}

/// Builds the shared learned or token-selected MoE policy for one target
/// layer. The caller supplies token-table IDs to `RouteSource::Selected` for
/// hash layers.
pub fn moe_policy(args: &V4Args, layer: usize) -> Result<MoePolicy, Error> {
    moe_policy_at(args, layer, &format!("layers.{layer}.ffn"))
}

pub(crate) fn moe_policy_at(args: &V4Args, layer: usize, root: &str) -> Result<MoePolicy, Error> {
    if layer
        >= usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
            .map_err(Error::backend)?
    {
        return Err(Error::backend(format!(
            "V4 target or prediction layer {layer} is out of range"
        )));
    }
    let expert_format = match args.expert_format {
        ExpertFormat::Dense => LinearFormat::Dense,
        ExpertFormat::MxFp4 => LinearFormat::MxFp4,
        ExpertFormat::BlockFp8 => match args.linear_format {
            format @ LinearFormat::E4M3BlockFp8(_) => format,
            _ => LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::Ue8m0)
                    .map_err(Error::backend)?,
            ),
        },
    };
    Ok(MoePolicy {
        layer,
        hidden: args.hidden_size,
        expert_count: args.n_routed_experts,
        routes_per_token: args.num_experts_per_tok,
        expert_width: args.moe_intermediate_size,
        shared_width: args
            .moe_intermediate_size
            .checked_mul(args.n_shared_experts)
            .ok_or_else(|| Error::backend("V4 shared expert width overflowed"))?,
        scoring: eredu_nn::RoutingScoring::SqrtSoftplus,
        normalize_routes: args.norm_topk_prob,
        normalization_epsilon: 1e-20,
        routed_scaling: args.routed_scaling_factor,
        expert_groups: 1,
        selected_groups: 1,
        router_weight: format!("{root}.gate.weight"),
        correction_bias: (layer >= args.num_hash_layers as usize)
            .then(|| format!("{root}.gate.bias")),
        expert_gate_up: format!("{root}.switch_mlp.gate_up_proj"),
        expert_down: format!("{root}.switch_mlp.down_proj"),
        shared_gate: format!("{root}.shared_experts.w1.weight"),
        shared_up: format!("{root}.shared_experts.w3.weight"),
        shared_down: format!("{root}.shared_experts.w2.weight"),
        shared_gate_format: args.linear_format_for(&format!("{root}.shared_experts.w1.weight")),
        shared_up_format: args.linear_format_for(&format!("{root}.shared_experts.w3.weight")),
        shared_down_format: args.linear_format_for(&format!("{root}.shared_experts.w2.weight")),
        expert_gate_up_format: args
            .linear_formats
            .get(&format!("{root}.switch_mlp.gate_up_proj"))
            .copied()
            .unwrap_or(expert_format),
        expert_down_format: args
            .linear_formats
            .get(&format!("{root}.switch_mlp.down_proj"))
            .copied()
            .unwrap_or(expert_format),
        shared_limit: None,
        limit: args.swiglu_limit,
    })
}

/// Declares bounded local keys and every append-only pooling component for V4
/// target and prediction layers.
pub fn state_layout(args: &V4Args) -> Result<StateLayout, Error> {
    args.validate().map_err(Error::backend)?;
    let layers = usize::try_from(
        args.num_hidden_layers
            .checked_add(args.num_nextn_predict_layers)
            .ok_or_else(|| Error::backend("V4 total state layer count overflowed"))?,
    )
    .map_err(Error::backend)?;
    let attention =
        AttentionPolicy::sliding(u32::try_from(args.sliding_window).map_err(Error::backend)?)
            .map_err(Error::backend)?;
    let policies = (0..layers)
        .map(|layer| {
            let fixed = match args.attention_policy(layer) {
                Some(V4AttentionPolicy::Local) => Vec::new(),
                Some(V4AttentionPolicy::Compressed { ratio }) => {
                    let mut tensors = pooling_stream(0, ratio, args.head_dim, ratio == 4)?;
                    if ratio == 4 {
                        tensors.extend(pooling_stream(1, ratio, args.index_head_dim, true)?);
                    }
                    tensors
                }
                None => return Err(Error::backend(format!("missing V4 layer policy {layer}"))),
            };
            if fixed.is_empty() {
                LayerCachePolicy::key_only(attention, 1, args.head_dim).map_err(Error::backend)
            } else {
                LayerCachePolicy::key_only_with_fixed_state(attention, 1, args.head_dim, fixed)
                    .map_err(Error::backend)
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(LayerSchedule::new(layers, policies).map_err(Error::backend)?)
        .map_err(Error::backend)
}

fn pooling_stream(
    stream: u32,
    ratio: i32,
    pooled_width: i32,
    overlapping: bool,
) -> Result<Vec<StateTensorPolicy>, Error> {
    let ratio = NonZeroU32::new(u32::try_from(ratio).map_err(Error::backend)?)
        .ok_or_else(|| Error::backend("V4 pooling ratio must be positive"))?;
    let source_width = if overlapping {
        pooled_width
            .checked_mul(2)
            .ok_or_else(|| Error::backend("V4 pooling source width overflowed"))?
    } else {
        pooled_width
    };
    let role = |component| StateTensorRole::Pooling { stream, component };
    let pending = |component| {
        StateTensorPolicy::new(
            role(component),
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensRem(ratio),
                StateTensorDimension::fixed(source_width).map_err(Error::backend)?,
            ],
            StateTensorDtype::Floating,
            MutableStateResidency::AlwaysDeviceMutable,
        )
        .map(|policy| policy.when_prefix_remainder_nonzero(ratio))
        .map_err(Error::backend)
    };
    let mut tensors = vec![
        pending(PoolingStateComponent::PendingValues)?,
        pending(PoolingStateComponent::PendingGates)?,
        StateTensorPolicy::new_with_residency(
            role(PoolingStateComponent::Pooled),
            vec![
                StateTensorDimension::Batch,
                StateTensorDimension::PrefixTokensDiv(ratio),
                StateTensorDimension::fixed(pooled_width).map_err(Error::backend)?,
            ],
            StateTensorDtype::Floating,
            StateResidencyClass::SealablePaged,
        )
        .map(|policy| policy.when_prefix_at_least(ratio))
        .map_err(Error::backend)?,
    ];
    if overlapping {
        for component in [
            PoolingStateComponent::OverlapValues,
            PoolingStateComponent::OverlapGates,
        ] {
            tensors.push(
                StateTensorPolicy::new(
                    role(component),
                    vec![
                        StateTensorDimension::Batch,
                        StateTensorDimension::Fixed(ratio),
                        StateTensorDimension::fixed(pooled_width).map_err(Error::backend)?,
                    ],
                    StateTensorDtype::Floating,
                    MutableStateResidency::AlwaysDeviceMutable,
                )
                .map(|policy| policy.when_prefix_at_least(ratio))
                .map_err(Error::backend)?,
            );
        }
    }
    Ok(tensors)
}

fn parameter(name: impl Into<String>) -> Result<ParameterSpec, Error> {
    ParameterSpec::trainable(name).map_err(Error::backend)
}

fn projection<B: eredu_nn::NeuralBackend>(
    name: impl Into<String>,
    input: i32,
    output: i32,
    format: LinearFormat,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Linear, Error> {
    B::linear(
        LinearSpec {
            input,
            output,
            weight: parameter(name)?,
            bias: None,
            format,
        },
        context,
    )
}

fn broadcast_streams<B: HyperNeuralBackend>(
    hidden: &B::Tensor,
    args: &V4Args,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<B::Tensor, Error> {
    hidden.expand_dims(2, context)?.broadcast_to(
        &[hidden.dim(0), hidden.dim(1), args.hc_mult, args.hidden_size],
        context,
    )
}
