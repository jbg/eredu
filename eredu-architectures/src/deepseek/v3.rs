//! Thin DeepSeek-V3/R1 architecture policy.

use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, EmbeddingOperator, Error, LinearOperator,
    NormalizationOperator, Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{LayerRuntimeState, LayeredArchitecture, LayeredForwardState, StateLayout};

use crate::decoder::{SequentialPredictionGroups, StaticModuleSpec, StaticModules};

use super::{
    block::V3Block,
    moe::MoePolicy,
    mtp::{EmbeddedInput, ForwardMode, RetainedValues, V3PredictionLayer},
    LayerPolicy, V3Args,
};

/// One target or appended prediction execution unit.
#[derive(Debug, Clone, Parameterized)]
#[parameterized(tensor = "B::Tensor")]
pub enum Unit<B: RoutedNeuralBackend + BlockwiseAttentionBackend> {
    /// Ordinary target decoder block.
    Target(V3Block<B>),
    /// One embedded prediction depth.
    Prediction(V3PredictionLayer<B>),
}

/// V3 values retained for one target-model forward.
pub struct ForwardContext<T> {
    tokens: T,
    embedded: T,
    mask: Option<T>,
    mode: ForwardMode,
    target_capture: Option<T>,
    draft_logits: Option<T>,
    draft_hidden: Option<T>,
}

impl<T> ForwardContext<T> {
    /// Borrows the final target hidden state when this was a target pass.
    pub const fn target_capture(&self) -> Option<&T> {
        self.target_capture.as_ref()
    }

    /// Borrows prediction logits when this was a draft pass.
    pub const fn draft_logits(&self) -> Option<&T> {
        self.draft_logits.as_ref()
    }

    /// Borrows the hidden state emitted by the selected prediction group.
    pub const fn draft_hidden(&self) -> Option<&T> {
        self.draft_hidden.as_ref()
    }
}

/// Thin target-model architecture over the shared V3 block and generic
/// resident/layerwise execution lifecycle.
pub struct Model<B: RoutedNeuralBackend + BlockwiseAttentionBackend> {
    args: V3Args,
    static_modules: StaticModules<B>,
    groups: SequentialPredictionGroups,
}

impl<B: RoutedNeuralBackend + BlockwiseAttentionBackend> Model<B> {
    /// Builds the unloaded pinned V3 modules.
    pub fn new(args: V3Args, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let static_modules = StaticModules::from_spec(
            StaticModuleSpec {
                embedding_weight: "model.embed_tokens.weight".into(),
                normalization_weight: "model.norm.weight".into(),
                head_weight: "lm_head.weight".into(),
                vocabulary: args.vocab_size,
                hidden_size: args.hidden_size,
                normalization_epsilon: args.rms_norm_eps,
                embedding_quantization: None,
                // Published V3/R1 checkpoints keep the output head dense,
                // including otherwise block-FP8 models.
                head_format: eredu_checkpoint::LinearFormat::Dense,
                tied_head: false,
            },
            context,
        )?;
        Ok(Self {
            groups: SequentialPredictionGroups::new(
                "model.layers",
                usize::try_from(args.num_hidden_layers).map_err(Error::backend)?,
                (0..usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?).map(
                    |depth| {
                        format!(
                            "model.layers.{}",
                            usize::try_from(args.num_hidden_layers).expect("validated layer count")
                                + depth
                        )
                    },
                ),
            )?,
            args,
            static_modules,
        })
    }

    /// Returns the normalized V3 arguments.
    pub const fn args(&self) -> &V3Args {
        &self.args
    }

    /// Borrows pinned modules for neutral checkpoint binding.
    pub const fn static_modules(&self) -> &StaticModules<B> {
        &self.static_modules
    }

    /// Mutably borrows pinned modules for neutral checkpoint binding.
    pub fn static_modules_mut(&mut self) -> &mut StaticModules<B> {
        &mut self.static_modules
    }

    /// Replaces the pinned module set after a backend composition has loaded
    /// exactly the static parameters owned by its placement.
    pub fn replace_static_modules(&mut self, static_modules: StaticModules<B>) {
        self.static_modules = static_modules;
    }

    /// Embeds tokens at the ingress of a pipeline-partitioned target pass.
    pub fn pipeline_embed(
        &mut self,
        tokens: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.embeddings.forward(tokens, context)
    }

    /// Executes one target block inside a pipeline-owned layer range.
    pub fn pipeline_forward_target<C>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
    {
        match unit {
            Unit::Target(block) => block.forward(hidden, mask, Some(cache), context),
            Unit::Prediction(_) => Err(Error::backend(
                "a V3 prediction unit cannot execute in the target pipeline range",
            )),
        }
    }

    /// Executes one pipeline target block with runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_target_with_provider<C, P>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match unit {
            Unit::Target(block) => {
                block.forward_with_provider(hidden, mask, Some(cache), pass, provider, context)
            }
            Unit::Prediction(_) => Err(Error::backend(
                "a V3 prediction unit cannot execute in the target pipeline range",
            )),
        }
    }

    /// Applies the shared target norm and vocabulary head on the final stage.
    pub fn pipeline_finish(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.static_modules.norm.forward(hidden, context)?;
        self.static_modules
            .lm_head
            .as_mut()
            .expect("validated V3 models have an untied output head")
            .forward(&hidden, context)
    }

    /// Applies the final target normalization without projecting vocabulary
    /// logits, allowing a distributed composition to own a sharded head.
    pub fn pipeline_finish_hidden(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.static_modules.norm.forward(hidden, context)
    }

    /// Executes one tensor-partitioned target block with a composition-owned
    /// reduction of partial output projections.
    pub fn pipeline_forward_target_parallel<C, F>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        match unit {
            Unit::Target(block) => {
                block.forward_parallel(hidden, mask, Some(cache), context, reduce)
            }
            Unit::Prediction(_) => Err(Error::backend(
                "a V3 prediction unit cannot execute in the target pipeline range",
            )),
        }
    }

    /// Tensor-partitioned target execution with runtime-supplied experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_target_parallel_with_provider<C, P, F>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        mask: Option<&B::Tensor>,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<B::Tensor, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        match unit {
            Unit::Target(block) => block.forward_parallel_with_provider(
                hidden,
                mask,
                Some(cache),
                pass,
                provider,
                context,
                reduce,
            ),
            Unit::Prediction(_) => Err(Error::backend(
                "a V3 prediction unit cannot execute in the target pipeline range",
            )),
        }
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
        C: CompressedAttentionCache<B::Tensor>,
    {
        let embedded = self.static_modules.embeddings.forward(tokens, context)?;
        match unit {
            Unit::Prediction(unit) => unit.forward(hidden, &embedded, tokens, cache, context),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
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
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = self.static_modules.embeddings.forward(tokens, context)?;
        match unit {
            Unit::Prediction(unit) => unit
                .forward_with_provider(hidden, &embedded, tokens, cache, pass, provider, context),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes a tensor-partitioned embedded predictor using an embedding
    /// already reduced by the distributed composition.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_parallel<C, F>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        match unit {
            Unit::Prediction(unit) => {
                unit.forward_parallel(hidden, embedded, tokens, cache, context, reduce)
            }
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Tensor-partitioned embedded predictor with supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_parallel_with_provider<C, P, F>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        embedded: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
        reduce: F,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
        F: FnMut(B::Tensor, &<B::Tensor as Tensor>::Context) -> Result<B::Tensor, Error>,
    {
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel_with_provider(
                hidden, embedded, tokens, cache, pass, provider, context, reduce,
            ),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
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
        S::LayerState: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_with_provider(
                hidden,
                forward.mask.as_ref(),
                Some(state.layer(index).map_err(Error::backend)?),
                pass,
                provider,
                context,
            ),
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state.layer(layer).map_err(Error::backend)?,
                    pass,
                    provider,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 execution unit does not match group {group}"
            ))),
        }
    }

    /// Executes one graph unit with stable target/MTP observation and
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
        S::LayerState: CompressedAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
    {
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_observed(
                &format!("model.layers.{index}"),
                hidden,
                forward.mask.as_ref(),
                Some(state.layer(index).map_err(Error::backend)?),
                context,
                observer,
            ),
            Unit::Prediction(_) if group > 0 => {
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
                "V3 observed execution unit does not match group {group}"
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
        S::LayerState: CompressedAttentionCache<B::Tensor>,
        O: eredu_runtime::ActivationObserver<B::Tensor, Error> + ?Sized,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_observed_with_provider(
                &format!("model.layers.{index}"),
                hidden,
                forward.mask.as_ref(),
                Some(state.layer(index).map_err(Error::backend)?),
                pass,
                provider,
                context,
                observer,
            ),
            Unit::Prediction(unit) if group > 0 => {
                observer.observe(&format!("mtp.{}.capture", group - 1), hidden)?;
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state.layer(layer).map_err(Error::backend)?,
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
            _ => Err(Error::backend(format!(
                "V3 observed execution unit does not match group {group}"
            ))),
        }
    }
}

impl<B, S> LayeredArchitecture<B, S> for Model<B>
where
    B: RoutedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
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
            Ok(Unit::Target(V3Block::new(&self.args, index, context)?))
        } else {
            Ok(Unit::Prediction(V3PredictionLayer::new(
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
                "V3 runtime state layout {:?} does not match architecture layout {expected:?}",
                state.layout()
            )));
        }
        let (tokens, supplied_hidden, supplied_mask, mode) = match input {
            EmbeddedInput::Target { tokens, mask } => (tokens, None, mask, ForwardMode::Target),
            EmbeddedInput::Draft {
                tokens,
                hidden,
                depth,
            } => {
                if depth >= self.groups.prediction_count() {
                    return Err(Error::backend(format!(
                        "V3 prediction depth {depth} is outside {} groups",
                        self.groups.prediction_count()
                    )));
                }
                (tokens, Some(hidden), None, ForwardMode::Draft(depth))
            }
            EmbeddedInput::DsparkContext { .. } | EmbeddedInput::DsparkProposal { .. } => {
                return Err(Error::backend(
                    "DSpark input is only supported by DeepSeek-V4",
                ));
            }
        };
        let embedded = self.static_modules.embeddings.forward(tokens, context)?;
        let hidden = supplied_hidden.cloned().unwrap_or_else(|| embedded.clone());
        let sequence = embedded.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if matches!(mode, ForwardMode::Target) && sequence > 1 {
            let offset = state.layer(0).map_err(Error::backend)?.offset();
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        Ok(LayeredForwardState {
            hidden,
            context: ForwardContext {
                tokens: tokens.clone(),
                embedded,
                mask,
                mode,
                target_capture: None,
                draft_logits: None,
                draft_hidden: None,
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
            ForwardMode::DsparkContext | ForwardMode::DsparkProposal => false,
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
                let cache = state.layer(index).map_err(Error::backend)?;
                unit.forward(hidden, forward.mask.as_ref(), Some(cache), context)
            }
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state.layer(layer).map_err(Error::backend)?,
                    context,
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 execution unit does not match group {group}"
            ))),
        }
    }

    fn complete_execution_group(
        &mut self,
        group: usize,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &mut Self::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        if group == 0 && matches!(forward.mode, ForwardMode::Target) {
            forward.target_capture = Some(hidden.clone());
        } else if matches!(forward.mode, ForwardMode::Draft(_)) {
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
                let hidden = self.static_modules.norm.forward(hidden, context)?;
                self.static_modules
                    .lm_head
                    .as_mut()
                    .expect("validated V3 models have an untied output head")
                    .forward(&hidden, context)
            }
            ForwardMode::Draft(_) => forward
                .draft_logits
                .clone()
                .ok_or_else(|| Error::backend("V3 draft group produced no logits")),
            ForwardMode::DsparkContext | ForwardMode::DsparkProposal => {
                Err(Error::backend("DSpark mode reached the V3 output path"))
            }
        }
    }

    fn retained_context_values<'a>(
        &'a self,
        forward: &'a Self::ForwardContext,
        _group: usize,
        _index: usize,
    ) -> Self::RetainedContextValues<'a> {
        RetainedValues::new([
            Some(&forward.tokens),
            Some(&forward.embedded),
            forward.mask.as_ref(),
            forward.target_capture.as_ref(),
            forward.draft_logits.as_ref(),
            forward.draft_hidden.as_ref(),
        ])
    }
}

/// Builds the shared MoE assembly policy for one sparse target layer.
pub fn moe_policy(args: &V3Args, layer: usize) -> Result<MoePolicy, Error> {
    if args.layer_schedule.get(layer) != Some(&LayerPolicy::SparseMoe) {
        return Err(Error::backend(format!("V3 layer {layer} is not sparse")));
    }
    moe_policy_for_layer(args, layer)
}

pub(crate) fn prediction_moe_policy(args: &V3Args, layer: usize) -> Result<MoePolicy, Error> {
    let start = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let end = usize::try_from(args.num_hidden_layers + args.num_nextn_predict_layers)
        .map_err(Error::backend)?;
    if !(start..end).contains(&layer) {
        return Err(Error::backend(format!(
            "V3 prediction layer {layer} is outside {start}..{end}"
        )));
    }
    moe_policy_for_layer(args, layer)
}

fn moe_policy_for_layer(args: &V3Args, layer: usize) -> Result<MoePolicy, Error> {
    let root = format!("model.layers.{layer}.mlp");
    Ok(MoePolicy {
        layer,
        hidden: args.hidden_size,
        expert_count: args.n_routed_experts,
        routes_per_token: args.num_experts_per_tok,
        expert_width: args.moe_intermediate_size,
        shared_width: args
            .moe_intermediate_size
            .checked_mul(args.n_shared_experts)
            .ok_or_else(|| Error::backend("V3 shared expert width overflowed"))?,
        scoring: eredu_nn::RoutingScoring::Sigmoid,
        normalize_routes: args.norm_topk_prob,
        normalization_epsilon: 1e-20,
        routed_scaling: args.routed_scaling_factor,
        expert_groups: args.n_group,
        selected_groups: args.topk_group,
        router_weight: format!("{root}.gate.weight"),
        correction_bias: Some(format!("{root}.gate.e_score_correction_bias")),
        expert_gate_up: format!("{root}.experts.gate_up_proj"),
        expert_down: format!("{root}.experts.down_proj"),
        shared_gate: format!("{root}.shared_experts.gate_proj.weight"),
        shared_up: format!("{root}.shared_experts.up_proj.weight"),
        shared_down: format!("{root}.shared_experts.down_proj.weight"),
        shared_gate_format: args
            .linear_format_for(&format!("{root}.shared_experts.gate_proj.weight")),
        shared_up_format: args.linear_format_for(&format!("{root}.shared_experts.up_proj.weight")),
        shared_down_format: args
            .linear_format_for(&format!("{root}.shared_experts.down_proj.weight")),
        expert_gate_up_format: args.linear_format_for(&format!("{root}.experts.gate_up_proj")),
        expert_down_format: args.linear_format_for(&format!("{root}.experts.down_proj")),
        shared_limit: None,
        limit: None,
    })
}

/// Declares compressed latent plus rotary state for target and embedded MTP
/// layers without exposing a backend cache implementation.
pub fn state_layout(args: &V3Args) -> Result<StateLayout, Error> {
    args.validate().map_err(Error::backend)?;
    let layers = usize::try_from(
        args.num_hidden_layers
            .checked_add(args.num_nextn_predict_layers)
            .ok_or_else(|| Error::backend("V3 total state layer count overflowed"))?,
    )
    .map_err(Error::backend)?;
    let policies = (0..layers)
        .map(|_| {
            LayerCachePolicy::compressed_latent_rotary(
                AttentionPolicy::Full,
                args.kv_lora_rank,
                args.qk_rope_head_dim,
            )
            .map_err(Error::backend)
        })
        .collect::<Result<Vec<_>, _>>()?;
    StateLayout::new(LayerSchedule::new(layers, policies).map_err(Error::backend)?)
        .map_err(Error::backend)
}
