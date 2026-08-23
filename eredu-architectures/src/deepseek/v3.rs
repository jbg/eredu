//! Thin DeepSeek-V3/R1 architecture policy.

use std::sync::Arc;

use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
use eredu_nn::{
    BlockwiseAttentionBackend, CompressedAttentionCache, EmbeddingLookupPolicy, EmbeddingOperator,
    Error, LinearOperator, NormalizationOperator, Parameterized, RoutedNeuralBackend, Tensor,
};
use eredu_runtime::{
    LayerRuntimeState, LayeredArchitecture, LayeredForwardState, ParallelLayeredArchitecture,
    ParallelRoutedLayeredArchitecture, RoutedExpertProvider, RoutedLayeredArchitecture,
    RuntimeStateComponents, StateLayout, StateSegmentLifetime, StateSegmentSpec,
};

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

impl<B, S> RoutedLayeredArchitecture<B, S> for Model<B>
where
    B: RoutedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
{
    fn forward_unit_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        Model::forward_unit_with_provider(
            self, group, index, unit, hidden, state, forward, pass, provider, context,
        )
    }
}

impl<B, S> ParallelRoutedLayeredArchitecture<B, S> for Model<B>
where
    B: RoutedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
{
    fn forward_unit_parallel_with_provider<P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>
    where
        P: RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        Model::forward_unit_parallel_with_provider(
            self, group, index, unit, hidden, state, forward, pass, parallel, provider, context,
        )
    }
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
    /// Borrows the architecture-prepared target mask.
    pub const fn mask(&self) -> Option<&T> {
        self.mask.as_ref()
    }

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

/// Immutable target-pass values transported across pipeline partitions.
#[derive(Debug, Clone)]
pub struct TargetBoundary<T> {
    tokens: T,
    embedded: T,
}

/// Family-owned wire schema for immutable V3 target context.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetBoundarySchema {
    hidden_size: i32,
}

impl TargetBoundarySchema {
    /// Derives the boundary schema from the normalized target configuration.
    pub const fn from_args(args: &V3Args) -> Self {
        Self {
            hidden_size: args.hidden_size,
        }
    }
}

impl<T> TargetBoundary<T> {
    /// Creates the typed target boundary in canonical wire order.
    pub const fn new(tokens: T, embedded: T) -> Self {
        Self { tokens, embedded }
    }

    /// Borrows original target token ids.
    pub const fn tokens(&self) -> &T {
        &self.tokens
    }

    /// Borrows the input-rank embeddings retained for prediction groups.
    pub const fn embedded(&self) -> &T {
        &self.embedded
    }

    /// Decomposes the boundary without cloning backend tensors.
    pub fn into_parts(self) -> (T, T) {
        (self.tokens, self.embedded)
    }
}

impl eredu_runtime::ArchitectureBoundary for TargetBoundarySchema {
    type Boundary<T> = TargetBoundary<T>;

    const IDENTITY: &'static str = "deepseek_v3.target";

    fn tensor_specs(&self) -> Vec<eredu_runtime::BoundaryTensorSpec> {
        use eredu_runtime::{BoundaryTensorDimension as Dim, BoundaryTensorDtype as Dtype};
        vec![
            eredu_runtime::BoundaryTensorSpec::new(
                "tokens",
                [Dim::Batch, Dim::Sequence],
                Dtype::Uint32,
            ),
            eredu_runtime::BoundaryTensorSpec::new(
                "embedded",
                [Dim::Batch, Dim::Sequence, Dim::Fixed(self.hidden_size)],
                Dtype::Activation,
            ),
        ]
    }

    fn encode<T>(
        &self,
        boundary: Self::Boundary<T>,
    ) -> Result<Vec<T>, eredu_runtime::ArchitectureBoundaryError> {
        Ok(vec![boundary.tokens, boundary.embedded])
    }

    fn decode<T>(
        &self,
        tensors: Vec<T>,
    ) -> Result<Self::Boundary<T>, eredu_runtime::ArchitectureBoundaryError> {
        eredu_runtime::validate_boundary_tensor_count(self, &tensors)?;
        let mut tensors = tensors.into_iter();
        Ok(TargetBoundary {
            tokens: tensors.next().expect("validated target tokens"),
            embedded: tensors.next().expect("validated target embeddings"),
        })
    }
}

/// Target input owned by either the first or a downstream pipeline rank.
pub enum TargetPartitionInput<'a, T> {
    /// Token ids embedded by the input-owning partition.
    Tokens(&'a T),
    /// Hidden state plus immutable input-rank context on later partitions.
    Hidden {
        /// Upstream decoder activation.
        hidden: T,
        /// Original token ids and embeddings.
        boundary: TargetBoundary<T>,
    },
}

/// Thin target-model architecture over the shared V3 block and generic
/// resident/layerwise execution lifecycle.
pub struct Model<B: RoutedNeuralBackend + BlockwiseAttentionBackend> {
    args: V3Args,
    static_modules: StaticModules<B>,
    groups: SequentialPredictionGroups,
    parallel_geometry: Option<Arc<super::parallel::V3LocalGeometry>>,
}

impl<B: RoutedNeuralBackend + BlockwiseAttentionBackend> Model<B> {
    /// Builds the unloaded pinned V3 modules.
    pub fn new(args: V3Args, context: &<B::Tensor as Tensor>::Context) -> Result<Self, Error> {
        args.validate().map_err(Error::backend)?;
        let static_modules = StaticModules::from_spec(static_spec(&args), context)?;
        Ok(Self {
            groups: prediction_groups(&args)?,
            args,
            static_modules,
            parallel_geometry: None,
        })
    }

    /// Builds unloaded V3 modules using one authoritative rank-local plan.
    pub fn new_parallel(
        args: V3Args,
        geometry: super::parallel::V3LocalGeometry,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self, Error> {
        crate::operator_requirements::require::<B>(
            "DeepSeek-V3 tensor parallelism",
            eredu_nn::NeuralOperatorCapabilities::SUM_PARALLEL,
        )?;
        args.validate().map_err(Error::backend)?;
        geometry.validate_for(&args).map_err(Error::backend)?;
        let static_modules = StaticModules::from_parallel_spec(
            static_spec(&args),
            geometry.embedding_range().clone(),
            Some(geometry.output_range().clone()),
            context,
        )?;
        Ok(Self {
            groups: prediction_groups(&args)?,
            args,
            static_modules,
            parallel_geometry: Some(Arc::new(geometry)),
        })
    }

    /// Returns the state layout matching the modules this instance builds.
    pub fn runtime_state_layout(&self) -> Result<StateLayout, Error> {
        match &self.parallel_geometry {
            Some(geometry) => Ok(geometry.state_layout().clone()),
            None => state_layout(&self.args),
        }
    }

    /// Borrows the shared rank-local geometry used by unit factories.
    pub fn shared_parallel_geometry(&self) -> Option<Arc<super::parallel::V3LocalGeometry>> {
        self.parallel_geometry.clone()
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

    /// Constructs one target or prediction unit from this model's authoritative
    /// global or rank-local geometry.
    pub fn construct_unit(
        &self,
        group: usize,
        index: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Unit<B>, Error> {
        self.groups.unit_count(group)?;
        let args = self
            .parallel_geometry
            .as_ref()
            .map_or(&self.args, |geometry| geometry.args());
        if group == 0 {
            Ok(Unit::Target(V3Block::new(args, index, context)?))
        } else {
            Ok(Unit::Prediction(V3PredictionLayer::new(
                args,
                group - 1,
                context,
            )?))
        }
    }

    /// Starts a replicated target partition and returns its typed immutable
    /// cross-partition boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partition_target<S>(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            TargetBoundary<B::Tensor>,
        ),
        Error,
    >
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        let (tokens, embedded, hidden) = match input {
            TargetPartitionInput::Tokens(tokens) => {
                let embedded = self.static_modules.embeddings.forward(tokens, context)?;
                (tokens.clone(), embedded.clone(), embedded)
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let (tokens, embedded) = boundary.into_parts();
                (tokens, embedded, hidden)
            }
        };
        self.begin_partition_target_embedded(
            tokens,
            embedded,
            hidden,
            supplied_mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    /// Starts a tensor-parallel target partition with input-rank vocabulary
    /// lookup and no downstream re-embedding.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_partition_target_parallel<S>(
        &mut self,
        input: TargetPartitionInput<'_, B::Tensor>,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            TargetBoundary<B::Tensor>,
        ),
        Error,
    >
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        let (tokens, embedded, hidden) = match input {
            TargetPartitionInput::Tokens(tokens) => {
                let embedded = B::vocabulary_parallel_lookup(
                    &mut self.static_modules.embeddings,
                    tokens,
                    EmbeddingLookupPolicy::Strict,
                    parallel,
                    context,
                )?;
                (tokens.clone(), embedded.clone(), embedded)
            }
            TargetPartitionInput::Hidden { hidden, boundary } => {
                let (tokens, embedded) = boundary.into_parts();
                (tokens, embedded, hidden)
            }
        };
        self.begin_partition_target_embedded(
            tokens,
            embedded,
            hidden,
            supplied_mask,
            state,
            expected,
            first_state_ordinal,
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn begin_partition_target_embedded<S>(
        &self,
        tokens: B::Tensor,
        embedded: B::Tensor,
        hidden: B::Tensor,
        supplied_mask: Option<&B::Tensor>,
        state: &mut S,
        expected: &StateLayout,
        first_state_ordinal: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (
            LayeredForwardState<B::Tensor, ForwardContext<B::Tensor>>,
            TargetBoundary<B::Tensor>,
        ),
        Error,
    >
    where
        S: LayerRuntimeState<B>,
        S::LayerState: RuntimeStateComponents<B>,
    {
        if state.layout() != expected {
            return Err(Error::backend(format!(
                "V3 partition state layout {:?} does not match architecture partition layout {expected:?}",
                state.layout()
            )));
        }
        let sequence = embedded.dim(1);
        let mask = if let Some(mask) = supplied_mask {
            Some(mask.clone())
        } else if sequence > 1 {
            let offset = state
                .layer(first_state_ordinal)
                .map_err(Error::backend)?
                .position();
            Some(B::causal_mask(sequence, offset, None, context)?)
        } else {
            None
        };
        let boundary = TargetBoundary::new(tokens.clone(), embedded.clone());
        Ok((
            LayeredForwardState {
                hidden,
                context: ForwardContext {
                    tokens,
                    embedded,
                    mask,
                    mode: ForwardMode::Target,
                    target_capture: None,
                    draft_logits: None,
                    draft_hidden: None,
                },
            },
            boundary,
        ))
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

    /// Applies rank-local normalization and complete vocabulary projection.
    pub fn pipeline_finish_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        let hidden = self.pipeline_finish_hidden(hidden, context)?;
        B::vocabulary_parallel_project(
            self.static_modules
                .lm_head
                .as_mut()
                .expect("validated V3 models have an untied output head"),
            &hidden,
            parallel,
            context,
        )
    }

    /// Finishes the serial target partition through the architecture output boundary.
    pub fn finish_partition_target(
        &mut self,
        hidden: &B::Tensor,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish(hidden, context)
    }

    /// Finishes the tensor-parallel target partition through the architecture output boundary.
    pub fn finish_partition_target_parallel(
        &mut self,
        hidden: &B::Tensor,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Error> {
        self.pipeline_finish_parallel(hidden, parallel, context)
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

    /// Executes a tensor-parallel predictor with architecture-owned embedding
    /// and collective semantics.
    pub fn pipeline_forward_prediction_neutral_parallel<C>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
    {
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.static_modules.embeddings,
            tokens,
            eredu_nn::EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel(
                hidden,
                &embedded,
                tokens,
                cache,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Target(_) => Err(Error::backend(
                "a V3 target unit cannot execute as an embedded predictor",
            )),
        }
    }

    /// Executes a tensor-parallel predictor with architecture-owned embedding,
    /// collectives, and runtime-supplied routed experts.
    #[allow(clippy::too_many_arguments)]
    pub fn pipeline_forward_prediction_neutral_parallel_with_provider<C, P>(
        &mut self,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        tokens: &B::Tensor,
        cache: &mut C,
        pass: eredu_runtime::ExpertPass,
        provider: &mut P,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<super::mtp::PredictionOutput<B::Tensor>, Error>
    where
        C: CompressedAttentionCache<B::Tensor>,
        P: eredu_runtime::RoutedExpertProvider<B>,
        P::Error: std::fmt::Display,
    {
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.static_modules.embeddings,
            tokens,
            eredu_nn::EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
        match unit {
            Unit::Prediction(unit) => unit.forward_parallel_with_provider(
                hidden,
                &embedded,
                tokens,
                cache,
                pass,
                provider,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
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

    /// Executes one tensor-partitioned target or prediction unit with
    /// runtime-supplied experts. All row-parallel partials are reduced on the
    /// supplied tensor-parallel context, while routed expert execution stays
    /// delegated to `provider` (and may therefore use a distinct EP group).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_unit_parallel_with_provider<S, P>(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Unit<B>,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut ForwardContext<B::Tensor>,
        pass: eredu_runtime::ExpertPass,
        parallel: &B::ParallelContext,
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
            Unit::Target(unit) if group == 0 => unit.forward_parallel_with_provider(
                hidden,
                forward.mask.as_ref(),
                Some(state.layer(index).map_err(Error::backend)?),
                pass,
                provider,
                context,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_parallel_with_provider(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state.layer(layer).map_err(Error::backend)?,
                    pass,
                    provider,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 parallel execution unit does not match group {group}"
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

fn static_spec(args: &V3Args) -> StaticModuleSpec {
    StaticModuleSpec {
        embedding_weight: "model.embed_tokens.weight".into(),
        normalization_weight: "model.norm.weight".into(),
        head_weight: "lm_head.weight".into(),
        vocabulary: args.vocab_size,
        hidden_size: args.hidden_size,
        normalization_epsilon: args.rms_norm_eps,
        normalization_offset: 0.0,
        embedding_quantization: None,
        // Published V3/R1 checkpoints keep the output head dense, including
        // otherwise block-FP8 models.
        head_format: eredu_checkpoint::LinearFormat::Dense,
        tied_head: false,
    }
}

fn prediction_groups(args: &V3Args) -> Result<SequentialPredictionGroups, Error> {
    let targets = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    SequentialPredictionGroups::new(
        "model.layers",
        targets,
        (0..usize::try_from(args.num_nextn_predict_layers).map_err(Error::backend)?)
            .map(|depth| format!("model.layers.{}", targets + depth)),
    )
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

    fn group_transport(&self, group: usize) -> eredu_runtime::ArchitectureGroupTransport {
        if group == 0 {
            eredu_runtime::ArchitectureGroupTransport::decoder()
        } else {
            let mut transport = eredu_runtime::ArchitectureGroupTransport::prediction();
            if group == 1 {
                transport.first_owner_static_roles.push("mtp".into());
            }
            transport
        }
    }

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
        self.construct_unit(group, index, context)
    }

    fn begin_forward<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self.runtime_state_layout()?;
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

impl<B, S> ParallelLayeredArchitecture<B, S> for Model<B>
where
    B: RoutedNeuralBackend + BlockwiseAttentionBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: CompressedAttentionCache<B::Tensor>,
{
    fn begin_forward_parallel<'a>(
        &mut self,
        input: Self::Input<'a>,
        state: &mut S,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<LayeredForwardState<B::Tensor, Self::ForwardContext>, Self::Error> {
        let expected = self
            .parallel_geometry
            .as_ref()
            .ok_or_else(|| Error::backend("V3 model was not built with local geometry"))?
            .state_layout()
            .clone();
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
        let embedded = B::vocabulary_parallel_lookup(
            &mut self.static_modules.embeddings,
            tokens,
            EmbeddingLookupPolicy::Strict,
            parallel,
            context,
        )?;
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

    fn forward_unit_parallel(
        &mut self,
        group: usize,
        index: usize,
        unit: &mut Self::Unit,
        hidden: &B::Tensor,
        state: &mut S,
        forward: &mut Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        self.groups.unit_count(group)?;
        match unit {
            Unit::Target(unit) if group == 0 => unit.forward_parallel(
                hidden,
                forward.mask.as_ref(),
                Some(state.layer(index).map_err(Error::backend)?),
                context,
                |value, context| B::sum_parallel(value, parallel, context),
            ),
            Unit::Prediction(unit) if group > 0 => {
                let layer = usize::try_from(self.args.num_hidden_layers).map_err(Error::backend)?
                    + group
                    - 1;
                let output = unit.forward_parallel(
                    hidden,
                    &forward.embedded,
                    &forward.tokens,
                    state.layer(layer).map_err(Error::backend)?,
                    context,
                    |value, context| B::sum_parallel(value, parallel, context),
                )?;
                forward.draft_logits = Some(output.logits);
                Ok(output.hidden)
            }
            _ => Err(Error::backend(format!(
                "V3 parallel execution unit does not match group {group}"
            ))),
        }
    }

    fn finish_forward_parallel(
        &mut self,
        hidden: &B::Tensor,
        _state: &mut S,
        forward: &Self::ForwardContext,
        parallel: &B::ParallelContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error> {
        match forward.mode {
            ForwardMode::Target => {
                let hidden = self.static_modules.norm.forward(hidden, context)?;
                B::vocabulary_parallel_project(
                    self.static_modules
                        .lm_head
                        .as_mut()
                        .expect("validated V3 models have an untied output head"),
                    &hidden,
                    parallel,
                    context,
                )
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
    let target_layers = usize::try_from(args.num_hidden_layers).map_err(Error::backend)?;
    let mut segments = vec![StateSegmentSpec::new(
        super::TARGET_STATE_SEGMENT,
        0..target_layers,
        StateSegmentLifetime::Persistent,
    )
    .map_err(Error::backend)?];
    if layers > target_layers {
        segments.push(
            StateSegmentSpec::new(
                super::PREDICTION_STATE_SEGMENT,
                target_layers..layers,
                StateSegmentLifetime::Persistent,
            )
            .map_err(Error::backend)?,
        );
    }
    StateLayout::segmented(
        LayerSchedule::new(layers, policies).map_err(Error::backend)?,
        segments,
    )
    .map_err(Error::backend)
}

#[cfg(test)]
mod boundary_tests {
    use super::*;
    use eredu_runtime::{ArchitectureBoundary, BoundaryTensorDtype};

    #[test]
    fn target_wire_geometry_is_architecture_owned() {
        let schema = TargetBoundarySchema { hidden_size: 16 };
        let tensors = schema.wire_schema().unwrap().resolve(2, 3).unwrap();
        assert_eq!(tensors[0].role(), "tokens");
        assert_eq!(tensors[0].shape(), [2, 3]);
        assert_eq!(tensors[0].dtype(), BoundaryTensorDtype::Uint32);
        assert_eq!(tensors[1].shape(), [2, 3, 16]);
        assert_eq!(tensors[1].dtype(), BoundaryTensorDtype::Activation);
    }
}
