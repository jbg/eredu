//! Architecture-owned Moshi realtime realization selection.

use std::{borrow::Borrow, collections::BTreeMap, num::NonZeroUsize};

use eredu_checkpoint::{
    recipe::{AtomicRecipeSet, DerivedWeightRecipe, RecipeDtype, RecipeMetadata},
    store::{TensorMetadata, TensorSourceProvenance},
    LinearFormat, SourceTensorEncoding, StoredDtype, WeightQuantization,
};
use eredu_core::{
    ParallelRankTopology, QuantizationRequest, RealtimeSampling, RealtimeSpeechConfig,
};
use eredu_nn::{DistributedNeuralBackend, NeuralBackend, Tensor};
use eredu_runtime::{
    observe_and_intervene, observe_model_logits, select_realtime_realization, ActivationObserver,
    CacheResidencyPolicy, CommunicationCompletionPolicy, CompositeLayeredTraversalHook,
    ExecutionResidency, GenerationSampler, LayerWeightResidency, LayeredTraversalHook,
    PipelineActivationDtype, PreparedRealtimeModelContract, RealtimeArchitectureProof,
    RealtimeArchitectureRequirements, RealtimeDecisionExecution, RealtimeExecutionRequirements,
    RealtimeIdentity, RealtimeMaterializationComponent, RealtimeMaterializationTask,
    RealtimeMechanism, RealtimeMechanismCapabilities, RealtimeMechanismRequirements,
    RealtimeObservationRequirements, RealtimeSelectionError, RealtimeSelectionRequest,
    RealtimeTopologyPolicy, RealtimeWeightComponentRequirement, RealtimeWeightComponentRole,
    RealtimeWeightLoweringRequirement, ResettableRuntimeState, RuntimeState,
    SelectedRealtimeRealization, WeightLoweringDescriptor, WeightLoweringKind,
};

use super::{
    observation_points, parameter_contract, select_parallel_execution, state_layout,
    DecisionBoundary, MoshiConfig, MoshiParallelSelection, MoshiParameterContract,
    RealtimePreparationPlan,
};

/// Derives portable input validation from normalized Moshi architecture policy.
pub fn realtime_ingress_contract(
    config: &MoshiConfig,
) -> Result<eredu_runtime::RealtimeIngressContract, String> {
    let boundary = DecisionBoundary::new(config).map_err(|error| error.to_string())?;
    eredu_runtime::RealtimeIngressContract::new(
        config.frame_schedule().clone(),
        boundary.text_token_domain(),
        boundary.audio_token_domain(),
    )
    .map_err(|error| error.to_string())
}

/// Builds canonical text-first Moshi sampler state from portable controls.
///
/// The architecture owns prediction cardinality and ordering: one text
/// decision followed by every depth-transformer audio decision. Concrete
/// backends supply only sampling and random-number mechanisms.
pub fn realtime_generation_samplers(
    schedule: &RealtimeSpeechConfig,
    sampling: RealtimeSampling,
) -> Result<Vec<GenerationSampler>, MoshiRealtimeSamplingError> {
    let sampler = |top_k: Option<usize>| {
        let top_k = top_k
            .map(|top_k| {
                i32::try_from(top_k)
                    .map_err(|_| MoshiRealtimeSamplingError::TopKExceedsI32 { top_k })
            })
            .transpose()?
            .unwrap_or(0);
        Ok(GenerationSampler::new().top_k(top_k).top_p(1.0).min_p(0.0))
    };
    std::iter::once(sampler(sampling.text_top_k()))
        .chain(
            std::iter::repeat_with(|| sampler(sampling.audio_top_k()))
                .take(schedule.depth_audio_codebooks()),
        )
        .collect()
}

/// Selects Moshi's architecture-proven fully-forced depth-tail optimization.
///
/// The neutral sequential driver still disables the optimization whenever
/// diagnostics are requested or any remaining target is sampled/existing.
pub const fn realtime_decision_execution() -> RealtimeDecisionExecution {
    RealtimeDecisionExecution::new(true)
}

/// Invalid portable controls for architecture-owned Moshi sampler state.
#[derive(Debug, Clone, Copy, Eq, PartialEq, thiserror::Error)]
pub enum MoshiRealtimeSamplingError {
    /// Runtime samplers represent top-k with a signed 32-bit value.
    #[error("realtime top-k {top_k} exceeds i32")]
    TopKExceedsI32 {
        /// Rejected portable top-k value.
        top_k: usize,
    },
}

/// Typed Moshi layered architecture admitted to the realtime decision slice.
pub trait MoshiRealtimeExecutionArchitecture<B, S>:
    'static
    + for<'a> eredu_runtime::LayeredArchitecture<
        B,
        S,
        Input<'a> = super::Input<'a, B::Tensor>,
        ForwardContext = super::ForwardContext<B::Tensor>,
        Error = eredu_nn::Error,
    >
    + eredu_runtime::ParallelLayeredArchitecture<B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    /// Constructs the architecture-owned boundary for one decision traversal.
    fn realtime_decision_boundary(&self) -> Result<DecisionBoundary, eredu_nn::Error>;

    /// Returns canonical text-plus-audio temporal input cardinality.
    fn realtime_temporal_cardinality(&self) -> usize;
}

impl<B, S> MoshiRealtimeExecutionArchitecture<B, S> for super::LayeredModel<B>
where
    B: NeuralBackend + DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B> + ResettableRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
{
    fn realtime_decision_boundary(&self) -> Result<DecisionBoundary, eredu_nn::Error> {
        DecisionBoundary::new(self.config())
    }

    fn realtime_temporal_cardinality(&self) -> usize {
        self.config().frame_schedule().total_audio_codebooks() + 1
    }
}

/// Architecture-owned bridge from a prepared frame to detached Moshi execution.
pub struct MoshiPreparedRealtimeFrameExecutor<A, B, M, O = eredu_runtime::NoopObserver>
where
    B: eredu_runtime::SubmissionBackend<
        Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
    >,
    M: eredu_runtime::RealtimeModelConstructionMechanisms<A, B>,
    A: eredu_runtime::LayeredArchitecture<B, M::State>,
{
    execution: eredu_runtime::ConstructedRealtimeExecution<A, B, M>,
    observer: O,
}

impl<A, B, M> MoshiPreparedRealtimeFrameExecutor<A, B, M, eredu_runtime::NoopObserver>
where
    B: eredu_runtime::SubmissionBackend<
        Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
    >,
    M: eredu_runtime::RealtimeModelConstructionMechanisms<A, B>,
    A: eredu_runtime::LayeredArchitecture<B, M::State>,
{
    /// Adopts one selected and fully constructed detached execution.
    pub const fn new(execution: eredu_runtime::ConstructedRealtimeExecution<A, B, M>) -> Self {
        Self {
            execution,
            observer: eredu_runtime::NoopObserver,
        }
    }
}

impl<A, B, M, O> MoshiPreparedRealtimeFrameExecutor<A, B, M, O>
where
    B: eredu_runtime::SubmissionBackend<
        Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
    >,
    M: eredu_runtime::RealtimeModelConstructionMechanisms<A, B>,
    A: eredu_runtime::LayeredArchitecture<B, M::State>,
{
    /// Adopts selected execution with one statically dispatched observer.
    pub const fn with_observer(
        execution: eredu_runtime::ConstructedRealtimeExecution<A, B, M>,
        observer: O,
    ) -> Self {
        Self {
            execution,
            observer,
        }
    }

    /// Borrows the detached selected execution.
    pub const fn execution(&self) -> &eredu_runtime::ConstructedRealtimeExecution<A, B, M> {
        &self.execution
    }

    /// Recovers the detached execution and its construction mechanisms.
    pub fn into_execution(self) -> eredu_runtime::ConstructedRealtimeExecution<A, B, M> {
        self.execution
    }
}

impl<A, B, M, O, SB, S, T> eredu_runtime::PreparedRealtimeFrameExecutor<SB, S, T>
    for MoshiPreparedRealtimeFrameExecutor<A, B, M, O>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + DistributedNeuralBackend,
    SB: eredu_runtime::SamplingBackend<
        Logits = <B as NeuralBackend>::Tensor,
        Token = <B as NeuralBackend>::Tensor,
        Context = <<B as NeuralBackend>::Tensor as Tensor>::Context,
    >,
    A: MoshiRealtimeExecutionArchitecture<B, M::State> + 'static,
    M: eredu_runtime::RealtimeModelConstructionMechanisms<A, B>,
    M::State: eredu_runtime::LayerRuntimeState<B> + ResettableRuntimeState<B>,
    <M::State as eredu_runtime::LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<<B as NeuralBackend>::Tensor>,
    M::PolicyError: std::fmt::Display,
    SB::Error: std::fmt::Display,
    S: eredu_runtime::Sampler<SB>,
    T: std::ops::DerefMut<Target = M::State>,
    O: ActivationObserver<<B as NeuralBackend>::Tensor, eredu_nn::Error>,
{
    type Error = MoshiRealtimeExecutionError<M::PolicyError>;
    type Retained = (
        <B as NeuralBackend>::Tensor,
        super::ForwardContext<<B as NeuralBackend>::Tensor>,
    );

    fn execute(
        &mut self,
        model_state: &mut T,
        temporal: &[<B as NeuralBackend>::Tensor],
        driver: &mut eredu_runtime::SequentialDecisionDriver<SB, S>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::Retained, Self::Error> {
        execute_detached_replicated_moshi_realtime_with_observer(
            &mut self.execution,
            &mut **model_state,
            temporal,
            driver,
            context,
            &mut self.observer,
        )
    }
}

/// Runs one prepared replicated temporal token list through detached execution.
///
/// The list is canonical text-first followed by every temporal audio codebook.
/// Schedule advancement, delayed history, and output publication remain outside
/// this architecture execution slice.
pub fn execute_detached_replicated_moshi_realtime<A, B, M, SB, S>(
    constructed: &mut eredu_runtime::ConstructedRealtimeExecution<A, B, M>,
    state: &mut M::State,
    temporal: &[B::Tensor],
    driver: &mut eredu_runtime::SequentialDecisionDriver<SB, S>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (B::Tensor, super::ForwardContext<B::Tensor>),
    MoshiRealtimeExecutionError<M::PolicyError>,
>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + DistributedNeuralBackend,
    A: MoshiRealtimeExecutionArchitecture<B, M::State> + 'static,
    M: eredu_runtime::RealtimeModelConstructionMechanisms<A, B>,
    M::State: eredu_runtime::LayerRuntimeState<B> + ResettableRuntimeState<B>,
    <M::State as eredu_runtime::LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor>,
    M::PolicyError: std::fmt::Display,
    SB: eredu_runtime::SamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = <B::Tensor as Tensor>::Context,
    >,
    SB::Error: std::fmt::Display,
    S: eredu_runtime::Sampler<SB>,
{
    execute_detached_replicated_moshi_realtime_with_observer(
        constructed,
        state,
        temporal,
        driver,
        context,
        &mut eredu_runtime::NoopObserver,
    )
}

/// Runs one prepared replicated frame with causal observation/intervention.
///
/// Declared Moshi activations are observed in production traversal order.
/// Each optional replacement is installed before the next layer, decision, or
/// completion consumes it. The caller retains publication authority over the
/// model-state branch if observation fails.
pub fn execute_detached_replicated_moshi_realtime_with_observer<A, B, M, SB, S, O>(
    constructed: &mut eredu_runtime::ConstructedRealtimeExecution<A, B, M>,
    state: &mut M::State,
    temporal: &[B::Tensor],
    driver: &mut eredu_runtime::SequentialDecisionDriver<SB, S>,
    context: &<B::Tensor as Tensor>::Context,
    observer: &mut O,
) -> Result<
    (B::Tensor, super::ForwardContext<B::Tensor>),
    MoshiRealtimeExecutionError<M::PolicyError>,
>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + DistributedNeuralBackend,
    A: MoshiRealtimeExecutionArchitecture<B, M::State> + 'static,
    M: eredu_runtime::RealtimeModelConstructionMechanisms<A, B>,
    M::State: eredu_runtime::LayerRuntimeState<B> + ResettableRuntimeState<B>,
    <M::State as eredu_runtime::LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor>,
    M::PolicyError: std::fmt::Display,
    SB: eredu_runtime::SamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = <B::Tensor as Tensor>::Context,
    >,
    SB::Error: std::fmt::Display,
    S: eredu_runtime::Sampler<SB>,
    O: ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
{
    let expected = constructed
        .selected()
        .requirements()
        .speech_schedule()
        .total_audio_codebooks()
        .checked_add(1)
        .ok_or(MoshiRealtimeExecutionError::TemporalCardinalityOverflow)?;
    if temporal.len() != expected {
        return Err(MoshiRealtimeExecutionError::TemporalCardinality {
            expected,
            actual: temporal.len(),
        });
    }
    let (text, audio) = temporal
        .split_first()
        .expect("validated Moshi temporal cardinality is positive");
    let audio = audio.iter().collect::<Vec<_>>();
    let input = super::Input {
        text,
        audio: &audio,
        mask: None,
    };
    let (output, mut forward) = match constructed.execution_mut() {
        eredu_runtime::RealtimeLayerwiseRuntime::Resident(runtime) => {
            let mut boundary = runtime
                .architecture()
                .realtime_decision_boundary()
                .map_err(MoshiRealtimeExecutionError::DecisionBoundary)?;
            let observations = MoshiRealtimeObservationTraversal { observer };
            let decisions = eredu_runtime::SequentialDecisionTraversal::new(driver, &mut boundary);
            let mut traversal = CompositeLayeredTraversalHook::new(observations, decisions);
            runtime
                .forward_with_traversal_hook(input, state, context, &mut traversal)
                .map_err(MoshiRealtimeExecutionError::Execution)
        }
        eredu_runtime::RealtimeLayerwiseRuntime::Bounded(runtime) => {
            let mut boundary = runtime
                .architecture()
                .realtime_decision_boundary()
                .map_err(MoshiRealtimeExecutionError::DecisionBoundary)?;
            let observations = MoshiRealtimeObservationTraversal { observer };
            let decisions = eredu_runtime::SequentialDecisionTraversal::new(driver, &mut boundary);
            let mut traversal = CompositeLayeredTraversalHook::new(observations, decisions);
            runtime
                .forward_with_traversal_hook(input, state, context, &mut traversal)
                .map_err(MoshiRealtimeExecutionError::Execution)
        }
    }?;
    let output = observe_model_logits(observer, &output).map_err(|error| {
        MoshiRealtimeExecutionError::Execution(eredu_runtime::LayerwiseRuntimeError::Architecture(
            error,
        ))
    })?;
    forward.replace_text_logits(output.clone());
    Ok((output, forward))
}

struct MoshiRealtimeObservationTraversal<'a, O: ?Sized> {
    observer: &'a mut O,
}

impl<B, O> LayeredTraversalHook<B, super::ForwardContext<B::Tensor>, eredu_nn::Error>
    for MoshiRealtimeObservationTraversal<'_, O>
where
    B: NeuralBackend,
    O: ActivationObserver<B::Tensor, eredu_nn::Error> + ?Sized,
{
    fn after_group_begin(
        &mut self,
        group: usize,
        value: &mut B::Tensor,
        _forward: &mut super::ForwardContext<B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_nn::Error> {
        if group == 0 {
            *value = observe_and_intervene(
                self.observer,
                &super::ObservationPoint::TemporalInput.path(),
                value,
            )?;
        }
        Ok(())
    }

    fn after_unit(
        &mut self,
        group: usize,
        index: usize,
        value: &mut B::Tensor,
        _forward: &mut super::ForwardContext<B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_nn::Error> {
        let point = match group {
            0 => super::ObservationPoint::TemporalLayer { layer: index },
            1 => super::ObservationPoint::DepthSliceLogits { slice: index },
            _ => return Ok(()),
        };
        *value = observe_and_intervene(self.observer, &point.path(), value)?;
        Ok(())
    }

    fn after_group(
        &mut self,
        group: usize,
        _value: &mut B::Tensor,
        forward: &mut super::ForwardContext<B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), eredu_nn::Error> {
        if group == 0 {
            let logits = forward
                .text_logits()
                .ok_or_else(|| eredu_nn::Error::backend("Moshi text logits are unavailable"))?;
            let logits = observe_and_intervene(
                self.observer,
                &super::ObservationPoint::TextLogits.path(),
                logits,
            )?;
            forward.replace_text_logits(logits);
        }
        Ok(())
    }
}

/// Runs one prepared temporal token list through an architecture-selected
/// pure tensor-parallel traversal.
#[allow(clippy::type_complexity)]
pub fn execute_detached_partitioned_moshi_realtime<A, B, State, Policy, G, R, I, T, U, V, SB, S>(
    runtime: &mut eredu_runtime::LayerwiseTraversalRuntime<
        eredu_runtime::LayerwiseRuntime<A, B, State, Policy>,
        Box<
            eredu_runtime::PartitionedTextRuntime<
                A,
                B,
                State,
                (),
                eredu_runtime::LayerwiseTraversalPartitionExecutor<A, B, State, Policy>,
                G,
                R,
                I,
                T,
                U,
                V,
            >,
        >,
    >,
    state: &mut State,
    temporal: &[B::Tensor],
    driver: &mut eredu_runtime::SequentialDecisionDriver<SB, S>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<(B::Tensor, super::ForwardContext<B::Tensor>), MoshiRealtimeExecutionError<Policy::Error>>
where
    B: eredu_runtime::CommunicationBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + DistributedNeuralBackend,
    State: eredu_runtime::LayerRuntimeState<B> + ResettableRuntimeState<B>,
    State::LayerState: eredu_nn::AttentionCache<B::Tensor>,
    A: MoshiRealtimeExecutionArchitecture<B, State>
        + eredu_runtime::ParallelLayeredArchitecture<B, State>,
    Policy: eredu_runtime::LayerwisePolicy<B, A::Unit>,
    Policy::Error: std::fmt::Display,
    G: Borrow<B::CommunicationGroup>,
    R: Borrow<B::CommunicationRoute>,
    I: eredu_runtime::CommunicationTensorMetadata<B>,
    T: eredu_runtime::PartitionBoundaryTransport<B, G, R, I>,
    U: eredu_runtime::PartitionOutputPublisher<B, G, R, I>,
    V: eredu_runtime::PartitionCommitAgreement<B, G, R, I>,
    B::ParallelContext: Sized,
    SB: eredu_runtime::SamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = <B::Tensor as Tensor>::Context,
    >,
    SB::Error: std::fmt::Display,
    S: eredu_runtime::Sampler<SB>,
{
    let expected = runtime.architecture().realtime_temporal_cardinality();
    if temporal.len() != expected {
        return Err(MoshiRealtimeExecutionError::TemporalCardinality {
            expected,
            actual: temporal.len(),
        });
    }
    let (text, audio) = temporal
        .split_first()
        .expect("validated Moshi temporal cardinality is positive");
    let audio = audio.iter().collect::<Vec<_>>();
    let input = super::Input {
        text,
        audio: &audio,
        mask: None,
    };
    let mut boundary = runtime
        .architecture()
        .realtime_decision_boundary()
        .map_err(MoshiRealtimeExecutionError::DecisionBoundary)?;
    let mut traversal = eredu_runtime::SequentialDecisionTraversal::new(driver, &mut boundary);
    runtime
        .forward_with_traversal_hook(input, state, context, &mut traversal)
        .map_err(|error| match error {
            eredu_runtime::PartitionedTraversalError::Contract(message) => {
                MoshiRealtimeExecutionError::PartitionedContract(message)
            }
            eredu_runtime::PartitionedTraversalError::Execution(error) => {
                MoshiRealtimeExecutionError::Execution(error)
            }
        })
}

/// Runs one prepared temporal token list through a combined constructed model.
///
/// New orchestration should split construction and retain mutable state in its
/// transaction owner, then call [`execute_detached_replicated_moshi_realtime`].
pub fn execute_replicated_moshi_realtime<A, B, M, SB, S>(
    model: &mut eredu_runtime::ConstructedRealtimeModel<A, B, M>,
    temporal: &[B::Tensor],
    driver: &mut eredu_runtime::SequentialDecisionDriver<SB, S>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<
    (B::Tensor, super::ForwardContext<B::Tensor>),
    MoshiRealtimeExecutionError<M::PolicyError>,
>
where
    B: eredu_runtime::SubmissionBackend<
            Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context,
        > + DistributedNeuralBackend,
    A: MoshiRealtimeExecutionArchitecture<B, M::State> + 'static,
    M: eredu_runtime::RealtimeModelConstructionMechanisms<A, B>,
    M::State: eredu_runtime::LayerRuntimeState<B> + ResettableRuntimeState<B>,
    <M::State as eredu_runtime::LayerRuntimeState<B>>::LayerState:
        eredu_nn::AttentionCache<B::Tensor>,
    M::PolicyError: std::fmt::Display,
    SB: eredu_runtime::SamplingBackend<
        Logits = B::Tensor,
        Token = B::Tensor,
        Context = <B::Tensor as Tensor>::Context,
    >,
    SB::Error: std::fmt::Display,
    S: eredu_runtime::Sampler<SB>,
{
    let (constructed, state) = model.constructed_execution_and_state_mut();
    execute_detached_replicated_moshi_realtime(constructed, state, temporal, driver, context)
}

/// Failure while running one already-prepared Moshi temporal execution slice.
#[derive(Debug, thiserror::Error)]
pub enum MoshiRealtimeExecutionError<P: std::fmt::Display> {
    /// Text plus audio cardinality overflowed the host representation.
    #[error("Moshi temporal token cardinality overflowed")]
    TemporalCardinalityOverflow,
    /// The prepared token list did not contain text plus every audio codebook.
    #[error("Moshi temporal token count is {actual}, expected {expected}")]
    TemporalCardinality {
        /// Required text-plus-audio tensor count.
        expected: usize,
        /// Supplied tensor count.
        actual: usize,
    },
    /// Architecture decision domains could not be constructed.
    #[error("Moshi decision boundary failed: {0}")]
    DecisionBoundary(eredu_nn::Error),
    /// The selected rank-local traversal no longer matches its communication plan.
    #[error("Moshi partitioned traversal contract failed: {0}")]
    PartitionedContract(String),
    /// Layered resident or bounded execution failed.
    #[error(transparent)]
    Execution(eredu_runtime::LayerwiseRuntimeError<eredu_nn::Error, P>),
}

/// Complete architecture-owned request made before realtime construction.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MoshiRealtimeRequest {
    quantization: Option<QuantizationRequest>,
    residency: LayerWeightResidency,
    state: CacheResidencyPolicy,
    rank: ParallelRankTopology,
    maximum_batch_size: NonZeroUsize,
    maximum_sequence_length: NonZeroUsize,
    activation_dtype: PipelineActivationDtype,
    completion: CommunicationCompletionPolicy,
    observations: RealtimeObservationRequirements,
    independently_addressable_parameters: bool,
}

impl MoshiRealtimeRequest {
    /// Creates one request from already validated topology, rank, and finite bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        quantization: Option<QuantizationRequest>,
        residency: LayerWeightResidency,
        state: CacheResidencyPolicy,
        rank: ParallelRankTopology,
        maximum_batch_size: NonZeroUsize,
        maximum_sequence_length: NonZeroUsize,
        activation_dtype: PipelineActivationDtype,
        completion: CommunicationCompletionPolicy,
        observations: RealtimeObservationRequirements,
    ) -> Self {
        Self {
            quantization,
            residency,
            state,
            rank,
            maximum_batch_size,
            maximum_sequence_length,
            activation_dtype,
            completion,
            observations,
            independently_addressable_parameters: false,
        }
    }

    /// Records a request for independently addressable parameter banks.
    pub const fn with_independently_addressable_parameters(mut self, requested: bool) -> Self {
        self.independently_addressable_parameters = requested;
        self
    }

    /// Returns the optional load-time transformation request.
    pub const fn quantization(&self) -> Option<QuantizationRequest> {
        self.quantization
    }

    /// Returns the exact immutable-weight residency policy.
    pub const fn residency(&self) -> LayerWeightResidency {
        self.residency
    }

    /// Returns the exact mutable-state residency policy.
    pub const fn state(&self) -> &CacheResidencyPolicy {
        &self.state
    }

    /// Returns the validated topology and global rank.
    pub const fn rank(&self) -> ParallelRankTopology {
        self.rank
    }

    /// Returns the one requested bounded-wait and timeout disposition policy.
    pub const fn completion(&self) -> CommunicationCompletionPolicy {
        self.completion
    }

    /// Returns exact requested observations.
    pub const fn observations(&self) -> &RealtimeObservationRequirements {
        &self.observations
    }
}

/// Architecture-owned result selected before backend construction.
#[derive(Debug)]
pub struct PreparedMoshiRealtime {
    preparation: RealtimePreparationPlan,
    execution_config: MoshiConfig,
    source_parameters: MoshiParameterContract,
    execution_parameters: MoshiParameterContract,
    parallel: Option<MoshiParallelSelection>,
    contract: PreparedRealtimeModelContract,
}

/// Architecture-owned selected Moshi execution paired with backend mechanisms.
///
/// The wrapper keeps model-family configuration and session identity out of a
/// concrete backend's executor. `E` is only the backend mechanism bundle which
/// materialized and erases the already selected typed architecture.
pub struct MoshiRealtimeExecution<E> {
    source_config: MoshiConfig,
    execution_config: MoshiConfig,
    selected: SelectedRealtimeRealization,
    executor: E,
}

/// Unforgeable architecture descriptor retained while backend mechanisms materialize.
pub struct MoshiRealtimeExecutionDescriptor {
    source_config: MoshiConfig,
    execution_config: MoshiConfig,
    selected: SelectedRealtimeRealization,
}

impl MoshiRealtimeExecutionDescriptor {
    /// Binds the materialized mechanism bundle without reopening selection.
    pub fn bind<E>(self, executor: E) -> MoshiRealtimeExecution<E> {
        MoshiRealtimeExecution {
            source_config: self.source_config,
            execution_config: self.execution_config,
            selected: self.selected,
            executor,
        }
    }
}

impl<E> MoshiRealtimeExecution<E> {
    /// Admitted source artifact configuration.
    pub const fn source_config(&self) -> &MoshiConfig {
        &self.source_config
    }

    /// Selected execution configuration, including load-time transforms.
    pub const fn execution_config(&self) -> &MoshiConfig {
        &self.execution_config
    }

    /// Authoritative backend-neutral selected realization.
    pub const fn selected(&self) -> &SelectedRealtimeRealization {
        &self.selected
    }

    /// Borrows the materialized backend mechanism bundle.
    pub const fn executor(&self) -> &E {
        &self.executor
    }

    /// Mutably borrows the materialized backend mechanism bundle.
    pub fn executor_mut(&mut self) -> &mut E {
        &mut self.executor
    }

    /// Consumes the architecture wrapper into its mechanism bundle.
    pub fn into_executor(self) -> E {
        self.executor
    }
}

/// Typed architecture handoff produced only after exact realtime selection.
pub struct PreparedMoshiRealtimeArchitecture<A> {
    architecture: Option<A>,
    source_architecture: Option<A>,
    contract: Option<PreparedRealtimeModelContract>,
    parallel: Option<MoshiParallelSelection>,
}

impl<A> PreparedMoshiRealtimeArchitecture<A> {
    /// Takes the selected execution-format architecture exactly once.
    pub fn take_architecture(&mut self) -> A {
        self.architecture
            .take()
            .expect("realtime architecture already taken")
    }

    /// Takes the source-format architecture required by a selected transform.
    pub fn take_source_architecture(&mut self) -> Option<A> {
        self.source_architecture.take()
    }

    /// Takes the validated neutral construction contract exactly once.
    pub fn take_contract(&mut self) -> PreparedRealtimeModelContract {
        self.contract
            .take()
            .expect("realtime model contract already taken")
    }

    /// Takes the architecture-selected rank-local pure-TP execution plan.
    pub fn take_parallel(&mut self) -> Option<MoshiParallelSelection> {
        self.parallel.take()
    }
}

/// Family-blind visitor for one selected replicated realtime architecture.
pub trait MoshiRealtimeArchitectureVisitor<B, S>: Sized
where
    B: NeuralBackend + DistributedNeuralBackend,
    S: RuntimeState<B>,
{
    /// Completed construction output.
    type Output;
    /// Generic mechanism-binding failure.
    type Error;

    /// Records the exact point at which backend module construction may begin.
    fn construction_started(&mut self) {}

    /// Receives an opaque layered architecture and exact materialization work.
    fn visit<A>(
        self,
        prepared: PreparedMoshiRealtimeArchitecture<A>,
        store: eredu_checkpoint::store::SharedCheckpointSource,
    ) -> Result<Self::Output, Self::Error>
    where
        A: MoshiRealtimeExecutionArchitecture<B, S>
            + eredu_runtime::RealtimeArchitectureIdentity
            + 'static,
        A::Error: std::fmt::Display;
}

/// Failure before or during the selected architecture/mechanism handoff.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MoshiRealtimeDispatchError<E> {
    /// Selected structure, store metadata, or module construction was invalid.
    #[error("Moshi realtime architecture construction failed: {0}")]
    Architecture(String),
    /// The family-blind mechanism visitor rejected the prepared architecture.
    #[error("realtime mechanism binding failed: {0}")]
    Mechanism(E),
}

impl PreparedMoshiRealtime {
    /// Captures architecture-owned model/session semantics before mechanism materialization.
    pub fn execution_descriptor(&self) -> MoshiRealtimeExecutionDescriptor {
        MoshiRealtimeExecutionDescriptor {
            source_config: self.source_config().clone(),
            execution_config: self.execution_config().clone(),
            selected: self.selected().clone(),
        }
    }

    /// Returns the submitted artifact root retained by header-only inspection.
    pub fn artifact_root(&self) -> &std::path::Path {
        self.preparation.artifact_root()
    }

    /// Returns the exact selected SafeTensors source retained by inspection.
    pub fn checkpoint_source(&self) -> &std::path::Path {
        self.preparation.checkpoint_source()
    }

    /// Returns the strict source checkpoint schema retained by inspection.
    pub fn checkpoint_plan(&self) -> &eredu_checkpoint::schema::SafetensorsCheckpointPlan {
        self.preparation.checkpoint_plan()
    }

    /// Returns the exact checkpoint resolution admitted during inspection.
    pub fn resolved_checkpoint_plan(
        &self,
    ) -> &eredu_checkpoint::validation::ResolvedCheckpointPlan {
        self.preparation.resolved_checkpoint_plan()
    }

    /// Returns the exact canonical shard set admitted during path inspection.
    pub fn admitted_shards(&self) -> Option<&eredu_checkpoint::safetensors::SafetensorsShards> {
        self.preparation.admitted_shards()
    }

    /// Returns the admitted source configuration.
    pub fn source_config(&self) -> &MoshiConfig {
        self.preparation.config()
    }

    /// Returns the architecture-selected execution configuration.
    pub const fn execution_config(&self) -> &MoshiConfig {
        &self.execution_config
    }

    /// Returns the exact source parameter topology and formats.
    pub const fn source_parameter_contract(&self) -> &MoshiParameterContract {
        &self.source_parameters
    }

    /// Returns the exact execution parameter topology and formats.
    pub const fn execution_parameter_contract(&self) -> &MoshiParameterContract {
        &self.execution_parameters
    }

    /// Returns the canonical recipes, including exact logical aliases.
    pub fn recipes(&self) -> &AtomicRecipeSet {
        self.preparation.recipes()
    }

    /// Returns exact admitted physical source metadata.
    pub fn source_metadata(&self) -> &BTreeMap<String, TensorMetadata> {
        self.preparation.source_metadata()
    }

    /// Returns inferred metadata for every canonical recipe owner.
    pub fn recipe_outputs(&self) -> &BTreeMap<String, RecipeMetadata> {
        self.preparation.recipe_outputs()
    }

    /// Returns the rank-local pure-TP plan, if tensor parallelism was selected.
    pub const fn parallel(&self) -> Option<&MoshiParallelSelection> {
        self.parallel.as_ref()
    }

    /// Returns the authoritative family-blind realtime realization.
    pub const fn selected(&self) -> &SelectedRealtimeRealization {
        self.contract.selected()
    }

    /// Returns exact selected materialization work, including physical source provenance.
    pub fn materialization_tasks(&self) -> &[RealtimeMaterializationTask] {
        self.contract.tasks()
    }

    /// Consumes the selection into all architecture-owned construction inputs.
    pub fn into_parts(
        self,
    ) -> (
        RealtimePreparationPlan,
        MoshiConfig,
        MoshiParameterContract,
        MoshiParameterContract,
        Option<MoshiParallelSelection>,
        PreparedRealtimeModelContract,
    ) {
        (
            self.preparation,
            self.execution_config,
            self.source_parameters,
            self.execution_parameters,
            self.parallel,
            self.contract,
        )
    }
}

/// Failure before a Moshi realtime realization becomes constructible.
#[derive(Debug, thiserror::Error)]
pub enum MoshiRealtimeSelectionError {
    /// A requested transform or normalized configuration is invalid.
    #[error("invalid Moshi realtime request: {0}")]
    InvalidRequest(String),
    /// Architecture parameter, state, or parallel planning failed.
    #[error("invalid Moshi realtime architecture: {0}")]
    InvalidArchitecture(String),
    /// Family-blind mechanism selection rejected the exact realization.
    #[error(transparent)]
    Selection(#[from] RealtimeSelectionError),
}

/// Header-only Moshi/PersonaPlex candidate awaiting backend mechanism selection.
pub struct InspectedMoshiRealtime {
    candidate: Candidate,
}

impl InspectedMoshiRealtime {
    /// Returns exact architecture requirements without opening tensor payloads.
    pub const fn requirements(&self) -> &RealtimeArchitectureRequirements {
        &self.candidate.requirements
    }
}

/// Resolves source/execution contracts without invoking backend construction.
pub fn inspect_moshi_realtime(
    preparation: RealtimePreparationPlan,
    request: MoshiRealtimeRequest,
) -> Result<InspectedMoshiRealtime, MoshiRealtimeSelectionError> {
    build_candidate(preparation, request).map(|candidate| InspectedMoshiRealtime { candidate })
}

/// Applies one family-blind mechanism report to a header-only candidate.
pub fn select_inspected_moshi_realtime(
    inspected: InspectedMoshiRealtime,
    capabilities: &RealtimeMechanismCapabilities,
) -> Result<PreparedMoshiRealtime, MoshiRealtimeSelectionError> {
    let candidate = inspected.candidate;
    let selected = select_realtime_realization(
        &candidate.requirements,
        &candidate.selection_request,
        capabilities,
    )?;
    let tasks = materialization_tasks(
        &candidate.execution_parameters,
        &selected,
        candidate.preparation.source_metadata(),
    )?;
    let contract = PreparedRealtimeModelContract::new(selected, tasks)
        .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?;
    Ok(PreparedMoshiRealtime {
        preparation: candidate.preparation,
        execution_config: candidate.execution_config,
        source_parameters: candidate.source_parameters,
        execution_parameters: candidate.execution_parameters,
        parallel: candidate.parallel,
        contract,
    })
}

/// Selects Moshi or PersonaPlex execution before any backend construction.
pub fn select_moshi_realtime(
    preparation: RealtimePreparationPlan,
    request: MoshiRealtimeRequest,
    capabilities: &RealtimeMechanismCapabilities,
) -> Result<PreparedMoshiRealtime, MoshiRealtimeSelectionError> {
    let inspected = inspect_moshi_realtime(preparation, request)?;
    select_inspected_moshi_realtime(inspected, capabilities)
}

/// Constructs and visits one selected Moshi-family architecture.
///
/// Store validation and task assembly inspect metadata only. The visitor is
/// the first code allowed to materialize checkpoint payloads.
pub fn visit_selected_moshi_realtime_architecture<B, S, V>(
    prepared: PreparedMoshiRealtime,
    store: eredu_checkpoint::store::SharedCheckpointSource,
    context: &<B::Tensor as Tensor>::Context,
    mut visitor: V,
) -> Result<V::Output, MoshiRealtimeDispatchError<V::Error>>
where
    B: NeuralBackend + DistributedNeuralBackend,
    S: eredu_runtime::LayerRuntimeState<B> + ResettableRuntimeState<B>,
    S::LayerState: eredu_nn::AttentionCache<B::Tensor>,
    V: MoshiRealtimeArchitectureVisitor<B, S>,
{
    validate_store_metadata(&prepared, store.as_ref())
        .map_err(MoshiRealtimeDispatchError::Architecture)?;
    let (
        preparation,
        execution_config,
        _source_parameters,
        _execution_parameters,
        parallel,
        contract,
    ) = prepared.into_parts();
    let selected = contract.selected();
    if selected.topology().is_replicated() != parallel.is_none() {
        return Err(MoshiRealtimeDispatchError::Architecture(
            "Moshi realtime topology and rank-local parallel plan differ".into(),
        ));
    }
    let transforms = selected.weight_lowerings().iter().any(|lowering| {
        matches!(
            lowering.kind(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        )
    });
    let source_config = transforms.then(|| preparation.config().clone());

    visitor.construction_started();
    let source_architecture = source_config
        .map(|config| super::LayeredModel::<B>::new(config, context))
        .transpose()
        .map_err(|error| MoshiRealtimeDispatchError::Architecture(error.to_string()))?;
    let architecture = match parallel.as_ref() {
        Some(parallel) => super::LayeredModel::<B>::new_selected_parallel(
            execution_config,
            parallel.geometry().clone(),
            parallel.execution_identity().to_owned(),
            context,
        ),
        None => super::LayeredModel::<B>::new(execution_config, context),
    }
    .map_err(|error| MoshiRealtimeDispatchError::Architecture(error.to_string()))?;
    visitor
        .visit(
            PreparedMoshiRealtimeArchitecture {
                architecture: Some(architecture),
                source_architecture,
                contract: Some(contract),
                parallel,
            },
            store,
        )
        .map_err(MoshiRealtimeDispatchError::Mechanism)
}

fn validate_store_metadata(
    prepared: &PreparedMoshiRealtime,
    store: &dyn eredu_checkpoint::store::CheckpointSource,
) -> Result<(), String> {
    let expected = prepared.source_metadata();
    let actual_keys = store.source_keys();
    let actual = actual_keys
        .iter()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    let expected_keys = expected
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    if actual != expected_keys || actual.len() != actual_keys.len() {
        return Err(format!(
            "selected realtime source catalog differs from supplied store: missing {:?}, unexpected {:?}",
            expected_keys.difference(&actual).collect::<Vec<_>>(),
            actual.difference(&expected_keys).collect::<Vec<_>>()
        ));
    }
    for (key, metadata) in expected {
        let actual = store
            .source_metadata(key)
            .map_err(|error| format!("selected realtime source {key:?} is unavailable: {error}"))?;
        if &actual != metadata {
            return Err(format!(
                "selected realtime source metadata differs for {key:?}"
            ));
        }
    }
    for provenance in prepared
        .materialization_tasks()
        .iter()
        .flat_map(|task| task.components())
        .flat_map(RealtimeMaterializationComponent::source_provenance)
    {
        let actual = store
            .source_provenance(&provenance.catalog_key)
            .map_err(|error| {
                format!(
                    "selected realtime source provenance {:?} is unavailable: {error}",
                    provenance.catalog_key
                )
            })?;
        if &actual != provenance {
            return Err(format!(
                "selected realtime source provenance differs for {:?}",
                provenance.catalog_key
            ));
        }
    }
    Ok(())
}

fn materialization_tasks(
    execution: &MoshiParameterContract,
    selected: &SelectedRealtimeRealization,
    source_metadata: &BTreeMap<String, TensorMetadata>,
) -> Result<Vec<RealtimeMaterializationTask>, MoshiRealtimeSelectionError> {
    let mut owners = BTreeMap::new();
    for group in execution.description().groups() {
        for member in group.group().members() {
            if owners
                .insert(member.target(), group.owner().clone())
                .is_some()
            {
                return Err(invalid_architecture(format!(
                    "execution parameter {:?} has duplicate ownership",
                    member.target()
                )));
            }
        }
    }
    selected
        .weight_lowerings()
        .iter()
        .map(|lowering| {
            let owner = owners.get(lowering.target().as_str()).ok_or_else(|| {
                invalid_architecture(format!(
                    "selected target {:?} has no execution owner",
                    lowering.target().as_str()
                ))
            })?;
            let components = lowering
                .components()
                .iter()
                .cloned()
                .map(|requirement| {
                    let provenance = requirement
                        .source_occurrences()
                        .iter()
                        .map(|source| {
                            let metadata =
                                source_metadata.get(source.as_str()).ok_or_else(|| {
                                    invalid_architecture(format!(
                                        "selected realtime source {:?} has no admitted metadata",
                                        source.as_str()
                                    ))
                                })?;
                            Ok(TensorSourceProvenance {
                                catalog_key: source.as_str().to_owned(),
                                physical_tensor: metadata.name.clone(),
                                output: source.as_str().to_owned(),
                                backing_shard: metadata.backing_shard.clone(),
                                source_encoding: SourceTensorEncoding::Safetensors(
                                    metadata.stored_dtype.clone(),
                                ),
                            })
                        })
                        .collect::<Result<Vec<_>, MoshiRealtimeSelectionError>>()?;
                    RealtimeMaterializationComponent::new(requirement, provenance)
                        .map_err(|error| invalid_architecture(error.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            RealtimeMaterializationTask::new(lowering.clone(), owner.clone(), components)
                .map_err(|error| invalid_architecture(error.to_string()))
        })
        .collect()
}

#[derive(Debug)]
struct Candidate {
    preparation: RealtimePreparationPlan,
    execution_config: MoshiConfig,
    source_parameters: MoshiParameterContract,
    execution_parameters: MoshiParameterContract,
    parallel: Option<MoshiParallelSelection>,
    requirements: RealtimeArchitectureRequirements,
    selection_request: RealtimeSelectionRequest,
}

fn build_candidate(
    preparation: RealtimePreparationPlan,
    request: MoshiRealtimeRequest,
) -> Result<Candidate, MoshiRealtimeSelectionError> {
    if request.independently_addressable_parameters {
        return Err(MoshiRealtimeSelectionError::InvalidRequest(
            "independently addressable parameter banks require routed execution units".into(),
        ));
    }
    let source_config = preparation.config().clone();
    let execution_config = execution_config(&source_config, request.quantization)?;
    let source_parameters = parameter_contract(&source_config)
        .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?;
    let execution_parameters = parameter_contract(&execution_config)
        .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?;

    validate_observations(&execution_config, request.observations())?;

    let topology = request.rank.topology();
    let parallel = if topology.is_replicated() {
        None
    } else {
        if topology.pipeline() != 1 || topology.expert() != 1 || topology.data() != 1 {
            return Err(MoshiRealtimeSelectionError::InvalidRequest(format!(
                "Moshi realtime admits replicated or pure tensor parallel topology, got TP/PP/EP/DP={}/{}/{}/{}",
                topology.tensor(),
                topology.pipeline(),
                topology.expert(),
                topology.data()
            )));
        }
        let maximum_batch_size = i32::try_from(request.maximum_batch_size.get()).map_err(|_| {
            MoshiRealtimeSelectionError::InvalidRequest(
                "maximum tensor-parallel batch size exceeds i32".into(),
            )
        })?;
        let maximum_sequence_length = i32::try_from(request.maximum_sequence_length.get())
            .map_err(|_| {
                MoshiRealtimeSelectionError::InvalidRequest(
                    "maximum tensor-parallel sequence length exceeds i32".into(),
                )
            })?;
        Some(
            select_parallel_execution(
                &execution_config,
                execution_parameters.description(),
                request.rank,
                maximum_batch_size,
                maximum_sequence_length,
                request.activation_dtype,
                request.completion,
                preparation.recipes().aliases(),
            )
            .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?,
        )
    };

    let lowerings = weight_lowerings(&preparation, &source_parameters, &execution_parameters)?;
    let source_identity = identity(preparation.metadata_contract_identity())?;
    let execution_identity = identity(execution_config.architecture_fingerprint())?;
    let architecture_identity = identity(format!(
        "moshi.realtime:{}",
        execution_config.architecture_fingerprint()
    ))?;
    let schedule_identity = identity(format!(
        "moshi.schedule:{}",
        execution_config.architecture_fingerprint()
    ))?;
    let state_identity = identity(match parallel.as_ref() {
        Some(parallel) => format!("moshi.state:{}", parallel.execution_identity()),
        None => format!(
            "moshi.state:{};replicated",
            execution_config.architecture_fingerprint()
        ),
    })?;
    let topology_identity = identity(format!(
        "moshi.topology:{};tp={}",
        execution_config.architecture_fingerprint(),
        topology.tensor()
    ))?;
    let state = parallel.as_ref().map_or_else(
        || {
            state_layout(&execution_config).map_err(|error| {
                MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string())
            })
        },
        |parallel| Ok(parallel.geometry().state_layout().clone()),
    )?;
    let mut mechanisms = vec![
        RealtimeMechanism::TensorOperations,
        RealtimeMechanism::NeuralOperations,
        RealtimeMechanism::ParameterMaterialization,
        RealtimeMechanism::ParameterStorage,
        RealtimeMechanism::StateStorage,
        RealtimeMechanism::CoordinateStorage,
        RealtimeMechanism::Sampling,
        RealtimeMechanism::Randomness,
        RealtimeMechanism::HostConversion,
        RealtimeMechanism::ExactCompletion,
        RealtimeMechanism::ResourceRetention,
        RealtimeMechanism::Transfer,
    ];
    if request.observations.requires_activations() {
        mechanisms.push(RealtimeMechanism::Observation);
    }
    if !topology.is_replicated() {
        mechanisms.push(RealtimeMechanism::Collectives);
    }
    let source_description = source_parameters.description().clone();
    let execution_description = execution_parameters.description().clone();
    let execution_graph = execution_description.graph().clone();
    let execution_units = execution_description.unit_layout().clone();
    let requirements = RealtimeArchitectureRequirements::new(
        architecture_identity.clone(),
        source_identity.clone(),
        execution_graph,
        execution_units,
        source_description,
        [RealtimeExecutionRequirements::new(
            execution_identity.clone(),
            execution_description,
            lowerings,
        )
        .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?],
        eredu_nn::NeuralOperatorCapabilities::NONE,
        schedule_identity.clone(),
        execution_config.frame_schedule().clone(),
        state_identity.clone(),
        state,
        RealtimeMechanismRequirements::new(mechanisms)
            .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?,
        RealtimeTopologyPolicy::new(
            topology_identity.clone(),
            true,
            (!topology.is_replicated()).then_some(topology.tensor()),
        )
        .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?,
        [
            ExecutionResidency::FullyResident,
            ExecutionResidency::LayerwiseHost,
            ExecutionResidency::DenseDiskStream,
        ],
    )
    .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))?;
    let proof = RealtimeArchitectureProof::new(
        architecture_identity,
        schedule_identity,
        state_identity,
        topology_identity,
        topology,
        request.rank.global_rank(),
    );
    let selection_request = RealtimeSelectionRequest::new(
        source_identity,
        execution_identity,
        request.residency,
        request.state,
        Some(proof),
        request.completion,
        request.observations,
    );
    Ok(Candidate {
        preparation,
        execution_config,
        source_parameters,
        execution_parameters,
        parallel,
        requirements,
        selection_request,
    })
}

fn execution_config(
    source: &MoshiConfig,
    request: Option<QuantizationRequest>,
) -> Result<MoshiConfig, MoshiRealtimeSelectionError> {
    let requested = request.map(requested_quantization).transpose()?;
    match (source.native_quantization(), requested) {
        (_, None) => Ok(source.clone()),
        (Some(native), Some(requested)) if native == requested => Ok(source.clone()),
        (Some(_), Some(_)) => Err(MoshiRealtimeSelectionError::InvalidRequest(
            "Moshi packed SafeTensors cannot be transcoded to another packed format".into(),
        )),
        (None, Some(requested)) => source
            .with_native_quantization(Some(requested))
            .map_err(|error| MoshiRealtimeSelectionError::InvalidRequest(error.to_string())),
    }
}

fn requested_quantization(
    request: QuantizationRequest,
) -> Result<WeightQuantization, MoshiRealtimeSelectionError> {
    match request {
        QuantizationRequest::Affine { group_size, bits } => {
            let group_size = i32::try_from(group_size).map_err(|_| {
                MoshiRealtimeSelectionError::InvalidRequest("affine group size exceeds i32".into())
            })?;
            let affine = eredu_checkpoint::AffineQuantization::new(group_size, i32::from(bits))
                .map_err(|error| MoshiRealtimeSelectionError::InvalidRequest(error.to_string()))?;
            Ok(WeightQuantization::Affine(affine))
        }
        QuantizationRequest::MxFp4 => Ok(WeightQuantization::MxFp4),
        _ => Err(MoshiRealtimeSelectionError::InvalidRequest(
            "unknown Moshi load-time quantization request".into(),
        )),
    }
}

fn weight_lowerings(
    preparation: &RealtimePreparationPlan,
    source: &MoshiParameterContract,
    execution: &MoshiParameterContract,
) -> Result<Vec<RealtimeWeightLoweringRequirement>, MoshiRealtimeSelectionError> {
    let source_members = parameter_members(source)?;
    let execution_members = parameter_members(execution)?;
    execution_members
        .iter()
        .filter(|(_, member)| member.linear_companion().is_none())
        .map(|(target, target_member)| {
            let source_member = source_members.get(target.as_str()).ok_or_else(|| {
                invalid_architecture(format!(
                    "execution parameter {target:?} has no source contract"
                ))
            })?;
            match (
                source.matrices().get(target.as_str()),
                execution.matrices().get(target.as_str()),
            ) {
                (Some(source_matrix), Some(target_matrix)) => matrix_lowering(
                    preparation,
                    target,
                    source_matrix,
                    target_matrix,
                    source_member,
                    target_member,
                    &source_members,
                    &execution_members,
                ),
                (None, None) => {
                    non_matrix_lowering(preparation, target, source_member, target_member)
                }
                _ => Err(invalid_architecture(format!(
                    "parameter {target:?} changes matrix role between source and execution"
                ))),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn matrix_lowering(
    preparation: &RealtimePreparationPlan,
    target: &str,
    source_matrix: &super::MoshiMatrixContract,
    target_matrix: &super::MoshiMatrixContract,
    source_member: &eredu_runtime::ParameterMemberSpec,
    target_member: &eredu_runtime::ParameterMemberSpec,
    source_members: &BTreeMap<String, eredu_runtime::ParameterMemberSpec>,
    target_members: &BTreeMap<String, eredu_runtime::ParameterMemberSpec>,
) -> Result<RealtimeWeightLoweringRequirement, MoshiRealtimeSelectionError> {
    if source_matrix.logical_shape() != target_matrix.logical_shape()
        || source_member.global_shape() != source_matrix.physical_shape()
        || target_member.global_shape() != target_matrix.physical_shape()
    {
        return Err(invalid_architecture(format!(
            "matrix {target:?} source and execution geometry differ"
        )));
    }

    let primary = recipe_component(
        preparation,
        target,
        target,
        source_matrix.physical_shape(),
        RealtimeWeightComponentRole::Primary,
    )?;
    let mut components = vec![primary.requirement.clone()];
    append_matrix_companion(
        preparation,
        target,
        source_matrix.scale(),
        target_matrix.scale(),
        source_members,
        target_members,
        RealtimeWeightComponentRole::Scale,
        &mut components,
    )?;
    append_matrix_companion(
        preparation,
        target,
        source_matrix.affine_bias(),
        target_matrix.affine_bias(),
        source_members,
        target_members,
        RealtimeWeightComponentRole::AffineBias,
        &mut components,
    )?;

    let transforms = source_matrix.format() != target_matrix.format();
    if !transforms
        && (source_matrix.scale() != target_matrix.scale()
            || source_matrix.affine_bias() != target_matrix.affine_bias())
    {
        return Err(invalid_architecture(format!(
            "matrix {target:?} changes physical companions without changing format"
        )));
    }
    let descriptor = WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(recipe_stored_dtype(&primary.output.dtype)?),
        target_matrix.format(),
        primary.output.shape.clone(),
        target_matrix.logical_shape().to_vec(),
        target_matrix.packed_axis(),
    )
    .map_err(|error| invalid_architecture(error.to_string()))?;
    let kind = match (primary.direct, transforms) {
        (true, false) => WeightLoweringKind::Direct,
        (false, false) => WeightLoweringKind::Derived,
        (true, true) => WeightLoweringKind::Transform,
        (false, true) => WeightLoweringKind::DerivedTransform,
    };
    RealtimeWeightLoweringRequirement::new(identity(target)?, components, descriptor, kind)
        .map_err(|error| invalid_architecture(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn append_matrix_companion(
    preparation: &RealtimePreparationPlan,
    primary: &str,
    source_target: Option<&str>,
    execution_target: Option<&str>,
    source_members: &BTreeMap<String, eredu_runtime::ParameterMemberSpec>,
    execution_members: &BTreeMap<String, eredu_runtime::ParameterMemberSpec>,
    role: RealtimeWeightComponentRole,
    components: &mut Vec<RealtimeWeightComponentRequirement>,
) -> Result<(), MoshiRealtimeSelectionError> {
    let Some(execution_target) = execution_target else {
        if source_target.is_some() {
            return Err(invalid_architecture(format!(
                "matrix {primary:?} drops a source {role:?} companion"
            )));
        }
        return Ok(());
    };
    let execution_member = execution_members.get(execution_target).ok_or_else(|| {
        invalid_architecture(format!(
            "matrix {primary:?} has no execution member for {role:?} companion {execution_target:?}"
        ))
    })?;
    let member_role = match role {
        RealtimeWeightComponentRole::Scale => eredu_nn::LinearCompanionRole::Scale,
        RealtimeWeightComponentRole::AffineBias => eredu_nn::LinearCompanionRole::AffineBias,
        RealtimeWeightComponentRole::Primary => {
            return Err(invalid_architecture(
                "a primary matrix component cannot be appended as a companion",
            ));
        }
    };
    if execution_member.linear_companion_of() != Some(primary)
        || execution_member.linear_companion() != Some(member_role)
    {
        return Err(invalid_architecture(format!(
            "matrix {primary:?} has incorrectly owned execution {role:?} companion {execution_target:?}"
        )));
    }
    let requirement = if let Some(source_target) = source_target {
        if source_target != execution_target {
            return Err(invalid_architecture(format!(
                "matrix {primary:?} renames source {role:?} companion {source_target:?} to {execution_target:?}"
            )));
        }
        let source_member = source_members.get(source_target).ok_or_else(|| {
            invalid_architecture(format!(
                "matrix {primary:?} has no source member for {role:?} companion {source_target:?}"
            ))
        })?;
        if source_member.global_shape() != execution_member.global_shape()
            || source_member.linear_companion_of() != Some(primary)
            || source_member.linear_companion() != Some(member_role)
        {
            return Err(invalid_architecture(format!(
                "matrix {primary:?} source and execution {role:?} companion geometry differ"
            )));
        }
        recipe_component(
            preparation,
            execution_target,
            source_target,
            source_member.global_shape(),
            role,
        )?
        .requirement
    } else {
        RealtimeWeightComponentRequirement::new(
            identity(execution_target)?,
            None,
            None,
            None,
            None,
            [],
            execution_member.global_shape().to_vec(),
            role,
        )
        .map_err(|error| invalid_architecture(error.to_string()))?
    };
    components.push(requirement);
    Ok(())
}

fn non_matrix_lowering(
    preparation: &RealtimePreparationPlan,
    target: &str,
    source_member: &eredu_runtime::ParameterMemberSpec,
    target_member: &eredu_runtime::ParameterMemberSpec,
) -> Result<RealtimeWeightLoweringRequirement, MoshiRealtimeSelectionError> {
    let primary = recipe_component(
        preparation,
        target,
        target,
        source_member.global_shape(),
        RealtimeWeightComponentRole::Primary,
    )?;
    if source_member.global_shape() != target_member.global_shape()
        || primary.output.shape != target_member.global_shape()
    {
        return Err(invalid_architecture(format!(
            "parameter {target:?} source, recipe, and execution geometry differ"
        )));
    }
    let descriptor = WeightLoweringDescriptor::new(
        SourceTensorEncoding::Safetensors(recipe_stored_dtype(&primary.output.dtype)?),
        LinearFormat::Dense,
        primary.output.shape.clone(),
        target_member.global_shape().to_vec(),
        None,
    )
    .map_err(|error| invalid_architecture(error.to_string()))?;
    RealtimeWeightLoweringRequirement::new(
        identity(target)?,
        [primary.requirement],
        descriptor,
        if primary.direct {
            WeightLoweringKind::Direct
        } else {
            WeightLoweringKind::Derived
        },
    )
    .map_err(|error| invalid_architecture(error.to_string()))
}

struct RecipeComponent {
    requirement: RealtimeWeightComponentRequirement,
    output: RecipeMetadata,
    direct: bool,
}

fn recipe_component(
    preparation: &RealtimePreparationPlan,
    execution_target: &str,
    recipe_target: &str,
    expected_shape: &[usize],
    role: RealtimeWeightComponentRole,
) -> Result<RecipeComponent, MoshiRealtimeSelectionError> {
    let (owner, recipe) = preparation
        .recipes()
        .get_resolved(recipe_target)
        .ok_or_else(|| {
            invalid_architecture(format!(
                "execution component {execution_target:?} has no admitted recipe"
            ))
        })?;
    let output = preparation.recipe_outputs().get(owner).ok_or_else(|| {
        invalid_architecture(format!("recipe owner {owner:?} has no inferred metadata"))
    })?;
    if output.shape != expected_shape {
        return Err(invalid_architecture(format!(
            "component {execution_target:?} source recipe and parameter geometry differ"
        )));
    }
    let sources = recipe.source_occurrences();
    for source in &sources {
        if !preparation.source_metadata().contains_key(*source) {
            return Err(invalid_architecture(format!(
                "recipe owner {owner:?} names unadmitted source {source:?}"
            )));
        }
    }
    let direct = matches!(
        recipe,
        DerivedWeightRecipe::Source {
            key,
            selection: eredu_checkpoint::store::TensorSelection::Full,
        } if sources.as_slice() == [key.as_str()] && owner == recipe_target
    );
    let requirement = RealtimeWeightComponentRequirement::new(
        identity(execution_target)?,
        Some(identity(owner)?),
        Some(identity(format!(
            "moshi.recipe:{}:{owner}",
            preparation.config().architecture_fingerprint()
        ))?),
        Some(recipe.clone()),
        Some(output.clone()),
        sources
            .into_iter()
            .map(identity)
            .collect::<Result<Vec<_>, _>>()?,
        output.shape.clone(),
        role,
    )
    .map_err(|error| invalid_architecture(error.to_string()))?;
    Ok(RecipeComponent {
        requirement,
        output: output.clone(),
        direct,
    })
}

fn parameter_members(
    contract: &MoshiParameterContract,
) -> Result<BTreeMap<String, eredu_runtime::ParameterMemberSpec>, MoshiRealtimeSelectionError> {
    let mut members = BTreeMap::new();
    for member in contract
        .description()
        .groups()
        .iter()
        .flat_map(|group| group.group().members())
    {
        if members
            .insert(member.target().to_owned(), member.clone())
            .is_some()
        {
            return Err(invalid_architecture(format!(
                "parameter contract repeats target {:?}",
                member.target()
            )));
        }
    }
    Ok(members)
}

fn validate_observations(
    config: &MoshiConfig,
    requested: &RealtimeObservationRequirements,
) -> Result<(), MoshiRealtimeSelectionError> {
    let admitted = observation_points(config)
        .into_iter()
        .map(|point| point.path())
        .collect::<std::collections::BTreeSet<_>>();
    if let Some(unknown) = requested
        .activations()
        .iter()
        .find(|observation| !admitted.contains(observation.as_str()))
    {
        return Err(MoshiRealtimeSelectionError::InvalidRequest(format!(
            "Moshi activation observation {:?} is not architecture-declared",
            unknown.as_str()
        )));
    }
    Ok(())
}

fn identity(value: impl Into<String>) -> Result<RealtimeIdentity, MoshiRealtimeSelectionError> {
    RealtimeIdentity::new(value)
        .map_err(|error| MoshiRealtimeSelectionError::InvalidArchitecture(error.to_string()))
}

fn invalid_architecture(message: impl Into<String>) -> MoshiRealtimeSelectionError {
    MoshiRealtimeSelectionError::InvalidArchitecture(message.into())
}

fn recipe_stored_dtype(dtype: &RecipeDtype) -> Result<StoredDtype, MoshiRealtimeSelectionError> {
    Ok(match dtype {
        RecipeDtype::Bool => StoredDtype::Bool,
        RecipeDtype::U8 => StoredDtype::U8,
        RecipeDtype::I8 => StoredDtype::I8,
        RecipeDtype::I16 => StoredDtype::I16,
        RecipeDtype::U16 => StoredDtype::U16,
        RecipeDtype::F16 => StoredDtype::F16,
        RecipeDtype::BF16 => StoredDtype::BF16,
        RecipeDtype::I32 => StoredDtype::I32,
        RecipeDtype::U32 => StoredDtype::U32,
        RecipeDtype::F32 => StoredDtype::F32,
        RecipeDtype::F64 => StoredDtype::F64,
        RecipeDtype::I64 => StoredDtype::I64,
        RecipeDtype::U64 => StoredDtype::U64,
        RecipeDtype::C64 => StoredDtype::C64,
        RecipeDtype::F8E4M3 => StoredDtype::F8E4M3,
        RecipeDtype::F8E5M2 => StoredDtype::F8E5M2,
        RecipeDtype::F4 => StoredDtype::F4,
        RecipeDtype::F8E8M0 => StoredDtype::F8E8M0,
        RecipeDtype::Other(value) => {
            return Err(invalid_architecture(format!(
                "unsupported Moshi recipe dtype {value:?}"
            )))
        }
        _ => {
            return Err(invalid_architecture(
                "unsupported future Moshi recipe dtype",
            ))
        }
    })
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, num::NonZeroUsize, time::Duration};

    use eredu_checkpoint::{
        recipe::RecipeCatalog,
        schema::{SafetensorsCheckpointPlan, StoredDtypeConstraint},
        store::{StoreError, TensorMetadata},
        validation::{CatalogTensorMetadata, SafetensorsCatalog},
        AffineQuantization, WeightQuantization,
    };
    use eredu_core::{
        CompletionCancellationMode, ParallelTopology, RealtimeFrameConvention, SessionCapabilities,
    };
    use eredu_runtime::{
        CommunicationCompletionCapabilities, RealtimeMechanismCapabilities, RealtimeSelectionIssue,
        StateComponentMechanism, StateComponentPlacement, StateMechanismCapabilities,
        WeightLoweringCapability,
    };

    #[test]
    fn realtime_samplers_are_text_first_with_exact_depth_cardinality() {
        let schedule = RealtimeSpeechConfig::new(
            4,
            1,
            3,
            3,
            0,
            1,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0; 5],
        )
        .unwrap();
        let sampling = RealtimeSampling::new(0.7, 0.8, 9)
            .unwrap()
            .with_top_k(Some(17), Some(29))
            .unwrap();

        let samplers = realtime_generation_samplers(&schedule, sampling).unwrap();

        assert_eq!(samplers.len(), 4);
        assert_eq!(samplers[0].top_k, 17);
        assert!(samplers[1..].iter().all(|sampler| sampler.top_k == 29));
        assert!(samplers.iter().all(|sampler| sampler.top_p == 1.0));
        assert!(samplers.iter().all(|sampler| sampler.min_p == 0.0));

        let greedy = realtime_generation_samplers(&schedule, RealtimeSampling::greedy()).unwrap();
        assert_eq!(greedy.len(), 4);
        assert!(greedy.iter().all(|sampler| sampler.top_k == 0));
    }

    #[test]
    fn realtime_sampler_conversion_rejects_unrepresentable_portable_top_k() {
        let schedule = RealtimeSpeechConfig::new(
            2,
            1,
            1,
            1,
            0,
            1,
            RealtimeFrameConvention::FeedbackAlignedHistory,
            vec![0, 0, 0],
        )
        .unwrap();
        let top_k = usize::try_from(i32::MAX).unwrap() + 1;
        let sampling = RealtimeSampling::greedy()
            .with_top_k(Some(top_k), None)
            .unwrap();

        assert!(matches!(
            realtime_generation_samplers(&schedule, sampling),
            Err(MoshiRealtimeSamplingError::TopKExceedsI32 {
                top_k: rejected
            }) if rejected == top_k
        ));
    }

    use crate::moshi::{prepare_realtime_model_from_catalog, safetensors_plan};

    use super::*;

    #[test]
    fn ingress_contract_uses_architecture_schedule_and_padding_domains() {
        let config = native_config(32);
        let ingress = realtime_ingress_contract(&config).unwrap();
        assert_eq!(ingress.schedule(), config.frame_schedule());
        assert_eq!(
            ingress.text_domain().cardinality(),
            usize::try_from(config.text_vocabulary_size() + 1).unwrap()
        );
        assert_eq!(
            ingress.audio_domain().cardinality(),
            usize::try_from(config.audio_vocabulary_size() + 1).unwrap()
        );
    }

    struct MetadataCatalog {
        tensors: BTreeMap<String, TensorMetadata>,
    }

    impl MetadataCatalog {
        fn from_plan(plan: &SafetensorsCheckpointPlan) -> Self {
            let tensors = plan
                .common_tensors
                .iter()
                .map(|tensor| {
                    let stored_dtype = match &tensor.dtype {
                        StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                        StoredDtypeConstraint::Floating => StoredDtype::F32,
                        StoredDtypeConstraint::OneOf(dtypes) => dtypes[0].clone(),
                    };
                    (
                        tensor.key.clone(),
                        TensorMetadata {
                            name: tensor.key.clone(),
                            logical_shape: tensor.shape.clone(),
                            physical_shape: tensor.shape.clone(),
                            stored_dtype,
                            encoded_byte_len: 0,
                            backing_shard: None,
                        },
                    )
                })
                .collect();
            Self { tensors }
        }
    }

    impl SafetensorsCatalog for MetadataCatalog {
        fn keys(&self) -> Vec<String> {
            self.tensors.keys().cloned().collect()
        }

        fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
            self.tensors
                .get(key)
                .map(|metadata| CatalogTensorMetadata {
                    shape: metadata.logical_shape.clone(),
                    stored_dtype: metadata.stored_dtype.clone(),
                })
                .ok_or_else(|| format!("unknown tensor {key:?}"))
        }
    }

    impl RecipeCatalog for MetadataCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.tensors
                .get(key)
                .cloned()
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }

    fn native_config(dimensions: usize) -> MoshiConfig {
        let feed_forward = dimensions * 3 / 2;
        MoshiConfig::from_json(&format!(
            r#"{{
                "model_type":"moshi", "dim":{dimensions}, "text_card":17,
                "n_q":2, "dep_q":1, "generated_audio_codebooks":1, "card":16,
                "num_heads":1, "num_layers":1, "dim_feedforward":{feed_forward},
                "causal":true, "context":3, "max_period":10000.0,
                "positional_embedding":"rope", "depformer_dim":{dimensions},
                "depformer_dim_feedforward":{feed_forward}, "depformer_num_heads":1,
                "depformer_num_layers":1, "depformer_context":2,
                "depformer_max_period":10000.0, "depformer_pos_emb":"none",
                "delays":[0,0,1]
            }}"#
        ))
        .unwrap()
    }

    fn preparation(config: MoshiConfig) -> RealtimePreparationPlan {
        preparation_at(config, "logical/artifact", "logical/checkpoint")
    }

    fn preparation_at(
        config: MoshiConfig,
        artifact_root: &str,
        checkpoint_source: &str,
    ) -> RealtimePreparationPlan {
        let plan = safetensors_plan(&config).unwrap();
        let catalog = MetadataCatalog::from_plan(&plan);
        prepare_realtime_model_from_catalog(artifact_root, checkpoint_source, config, &catalog)
            .unwrap()
    }

    fn affine_config(dimensions: usize) -> MoshiConfig {
        native_config(dimensions)
            .with_native_quantization(Some(WeightQuantization::Affine(
                AffineQuantization::new(16, 4).unwrap(),
            )))
            .unwrap()
    }

    fn request(
        quantization: Option<QuantizationRequest>,
        activations: impl IntoIterator<Item = RealtimeIdentity>,
    ) -> MoshiRealtimeRequest {
        request_with_tensor_parallel(quantization, activations, 1)
    }

    fn request_with_tensor_parallel(
        quantization: Option<QuantizationRequest>,
        activations: impl IntoIterator<Item = RealtimeIdentity>,
        tensor_parallel: usize,
    ) -> MoshiRealtimeRequest {
        let topology = ParallelTopology::new(tensor_parallel, 1, 1, 1).unwrap();
        MoshiRealtimeRequest::new(
            quantization,
            LayerWeightResidency::FullyResident,
            CacheResidencyPolicy::Device,
            ParallelRankTopology::new(topology, 0).unwrap(),
            NonZeroUsize::new(1).unwrap(),
            NonZeroUsize::new(1).unwrap(),
            PipelineActivationDtype::Float32,
            CommunicationCompletionPolicy::new(
                Duration::from_secs(1),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
            RealtimeObservationRequirements::new(true, activations),
        )
    }

    fn capabilities(candidate: &Candidate) -> RealtimeMechanismCapabilities {
        let requirements = &candidate.requirements;
        let state = StateMechanismCapabilities::new(
            (0..requirements.state_layout().len()).flat_map(|layer| {
                requirements
                    .state_layout()
                    .components(layer)
                    .unwrap()
                    .iter()
                    .cloned()
                    .map(move |component| {
                        StateComponentMechanism::new(
                            layer,
                            component,
                            Some(StateComponentPlacement::Device),
                            None,
                        )
                    })
                    .collect::<Vec<_>>()
            }),
        )
        .with_transactions(true, true)
        .with_reset(true)
        .with_observation_retention(true);
        RealtimeMechanismCapabilities::new(
            requirements.operators(),
            requirements.mechanisms().mechanisms().iter().copied(),
            requirements.residencies().iter().copied(),
            requirements.executions()[0]
                .weight_lowerings()
                .iter()
                .map(|lowering| {
                    WeightLoweringCapability::new(lowering.descriptor().clone(), lowering.kind())
                })
                .collect(),
            state,
            NonZeroUsize::new(128).unwrap(),
            CommunicationCompletionCapabilities::new([
                CompletionCancellationMode::QuarantineUntilComplete,
            ])
            .unwrap(),
            SessionCapabilities::new(true, true, true),
        )
        .with_observation_identities(
            observation_points(&candidate.execution_config)
                .into_iter()
                .map(|point| RealtimeIdentity::new(point.path()).unwrap()),
        )
    }

    #[test]
    fn native_metadata_selects_exact_dense_realization_before_construction() {
        let config = native_config(4);
        let observation = RealtimeIdentity::new("temporal.input").unwrap();
        let request = request(None, [observation.clone()]);
        let candidate = build_candidate(preparation(config.clone()), request.clone()).unwrap();
        let capabilities = capabilities(&candidate);

        let prepared = select_moshi_realtime(preparation(config), request, &capabilities).unwrap();

        assert_eq!(
            prepared.source_config().architecture_fingerprint(),
            prepared.execution_config().architecture_fingerprint()
        );
        assert_eq!(
            prepared.selected().topology(),
            ParallelTopology::new(1, 1, 1, 1).unwrap()
        );
        assert_eq!(prepared.selected().rank(), 0);
        assert_eq!(
            prepared.selected().requirements().speech_schedule(),
            prepared.source_config().frame_schedule()
        );
        assert!(prepared.parallel().is_none());
        assert!(prepared
            .selected()
            .weight_lowerings()
            .iter()
            .all(|lowering| lowering.components().len() == 1
                && lowering.primary().source_occurrences().len() == 1
                && lowering.descriptor().executable() == LinearFormat::Dense));
        assert!(prepared
            .selected()
            .observations()
            .activations()
            .contains(&observation));
    }

    #[test]
    fn selected_moshi_and_personaplex_retain_exact_task_source_provenance() {
        let configs = [
            native_config(4),
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap(),
        ];
        for config in configs {
            let request = request(None, std::iter::empty());
            let candidate = build_candidate(preparation(config.clone()), request.clone()).unwrap();
            let prepared =
                select_moshi_realtime(preparation(config), request, &capabilities(&candidate))
                    .unwrap();

            assert_eq!(
                prepared.materialization_tasks().len(),
                prepared.selected().weight_lowerings().len()
            );
            for (task, lowering) in prepared
                .materialization_tasks()
                .iter()
                .zip(prepared.selected().weight_lowerings())
            {
                assert_eq!(task.lowering(), lowering);
                for component in task.components() {
                    assert_eq!(
                        component
                            .source_provenance()
                            .iter()
                            .map(|source| source.catalog_key.as_str())
                            .collect::<Vec<_>>(),
                        component
                            .requirement()
                            .source_occurrences()
                            .iter()
                            .map(RealtimeIdentity::as_str)
                            .collect::<Vec<_>>()
                    );
                    for source in component.source_provenance() {
                        let metadata = &prepared.source_metadata()[&source.catalog_key];
                        assert_eq!(source.physical_tensor, metadata.name);
                        assert_eq!(source.output, source.catalog_key);
                        assert_eq!(source.backing_shard, metadata.backing_shard);
                        assert_eq!(
                            source.source_encoding,
                            SourceTensorEncoding::Safetensors(metadata.stored_dtype.clone())
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn realtime_source_identity_is_the_relocation_stable_metadata_contract() {
        let config = native_config(4);
        let first_preparation =
            preparation_at(config.clone(), "first/artifact", "first/checkpoint");
        let expected = first_preparation.metadata_contract_identity().to_owned();
        let first = build_candidate(first_preparation, request(None, std::iter::empty())).unwrap();
        let relocated = build_candidate(
            preparation_at(config, "relocated/artifact", "relocated/checkpoint"),
            request(None, std::iter::empty()),
        )
        .unwrap();

        assert_eq!(first.requirements.source().as_str(), expected);
        assert_eq!(relocated.requirements.source(), first.requirements.source());
    }

    #[test]
    fn personaplex_aliases_resolve_to_exact_recipe_owners_and_sources() {
        let config =
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
        let request = request(None, std::iter::empty());
        let candidate = build_candidate(preparation(config.clone()), request.clone()).unwrap();
        let capabilities = capabilities(&candidate);

        let prepared = select_moshi_realtime(preparation(config), request, &capabilities).unwrap();
        let alias = "depformer.slices.1.transformer.layers.0.norm1.weight";
        let lowering = prepared
            .selected()
            .weight_lowerings()
            .iter()
            .find(|lowering| lowering.target().as_str() == alias)
            .unwrap();

        assert_eq!(prepared.source_metadata().len(), 475);
        assert_eq!(prepared.recipes().aliases().count(), 180);
        assert_eq!(
            lowering.primary().recipe_owner().unwrap().as_str(),
            "depformer.slices.0.transformer.layers.0.norm1.weight"
        );
        assert_eq!(
            lowering.primary().source_occurrences()[0].as_str(),
            "depformer.layers.0.norm1.alpha"
        );
        assert_eq!(lowering.kind(), WeightLoweringKind::Derived);
    }

    #[test]
    fn personaplex_aliases_remain_valid_under_pure_tensor_parallel_selection() {
        let config =
            MoshiConfig::from_json(r#"{"model_type":"personaplex","version":"7b-v1"}"#).unwrap();
        let request = request_with_tensor_parallel(None, std::iter::empty(), 2);
        let candidate = build_candidate(preparation(config.clone()), request.clone()).unwrap();
        let prepared =
            select_moshi_realtime(preparation(config), request, &capabilities(&candidate)).unwrap();
        let alias = "depformer.slices.1.transformer.layers.0.norm1.weight";

        assert!(prepared.parallel().is_some());
        assert_eq!(
            prepared.selected().topology(),
            ParallelTopology::new(2, 1, 1, 1).unwrap()
        );
        assert_eq!(
            prepared
                .selected()
                .weight_lowerings()
                .iter()
                .find(|lowering| lowering.target().as_str() == alias)
                .unwrap()
                .primary()
                .recipe_owner()
                .unwrap()
                .as_str(),
            "depformer.slices.0.transformer.layers.0.norm1.weight"
        );
    }

    #[test]
    fn native_affine_metadata_selects_exact_recipe_backed_component_family() {
        let config = affine_config(32);
        let request = request(None, std::iter::empty());
        let candidate = build_candidate(preparation(config.clone()), request.clone()).unwrap();
        let prepared =
            select_moshi_realtime(preparation(config), request, &capabilities(&candidate)).unwrap();
        let target = "transformer.layers.0.self_attn.out_proj.weight";
        let lowering = prepared
            .selected()
            .weight_lowerings()
            .iter()
            .find(|lowering| lowering.target().as_str() == target)
            .unwrap();

        assert_eq!(lowering.kind(), WeightLoweringKind::Direct);
        assert_eq!(lowering.descriptor().physical_shape(), &[32, 4]);
        assert_eq!(lowering.descriptor().logical_shape(), &[32, 32]);
        assert_eq!(lowering.descriptor().packed_axis(), Some(1));
        assert_eq!(
            lowering.descriptor().executable(),
            prepared.execution_parameter_contract().matrices()[target].format()
        );
        assert_eq!(
            lowering
                .components()
                .iter()
                .map(|component| component.role())
                .collect::<Vec<_>>(),
            vec![
                RealtimeWeightComponentRole::Primary,
                RealtimeWeightComponentRole::Scale,
                RealtimeWeightComponentRole::AffineBias,
            ]
        );
        assert_eq!(lowering.primary().physical_shape(), &[32, 4]);
        assert_eq!(lowering.scale().unwrap().physical_shape(), &[32, 2]);
        assert_eq!(lowering.affine_bias().unwrap().physical_shape(), &[32, 2]);
        for component in lowering.components() {
            assert!(component.is_recipe_backed());
            assert_eq!(component.source_occurrences().len(), 1);
            assert_eq!(
                component.recipe_owner().unwrap().as_str(),
                component.target().as_str()
            );
        }
    }

    #[test]
    fn dense_to_affine_selects_source_primary_and_generated_target_companions() {
        let source = native_config(32);
        let source_identity = source.architecture_fingerprint().to_owned();
        let transform = Some(QuantizationRequest::Affine {
            group_size: 16,
            bits: 4,
        });
        let request = request(transform, std::iter::empty());
        let candidate = build_candidate(preparation(source.clone()), request.clone()).unwrap();
        let prepared =
            select_moshi_realtime(preparation(source), request, &capabilities(&candidate)).unwrap();
        let target = "transformer.layers.0.self_attn.out_proj.weight";
        let lowering = prepared
            .selected()
            .weight_lowerings()
            .iter()
            .find(|lowering| lowering.target().as_str() == target)
            .unwrap();

        assert_eq!(
            prepared.source_config().architecture_fingerprint(),
            source_identity
        );
        assert_ne!(
            prepared.execution_config().architecture_fingerprint(),
            source_identity
        );
        assert_eq!(lowering.kind(), WeightLoweringKind::Transform);
        assert_eq!(lowering.descriptor().physical_shape(), &[32, 32]);
        assert_eq!(lowering.descriptor().logical_shape(), &[32, 32]);
        assert_eq!(lowering.descriptor().packed_axis(), Some(1));
        assert!(lowering.primary().is_recipe_backed());
        assert_eq!(lowering.primary().physical_shape(), &[32, 32]);
        assert_eq!(lowering.primary().source_occurrences().len(), 1);
        assert!(!lowering.scale().unwrap().is_recipe_backed());
        assert!(lowering.scale().unwrap().source_occurrences().is_empty());
        assert_eq!(lowering.scale().unwrap().physical_shape(), &[32, 2]);
        assert!(!lowering.affine_bias().unwrap().is_recipe_backed());
        assert_eq!(lowering.affine_bias().unwrap().physical_shape(), &[32, 2]);
    }

    #[test]
    fn wrong_primary_component_geometry_is_rejected_by_selection_before_construction() {
        let source = native_config(32);
        let transform = Some(QuantizationRequest::Affine {
            group_size: 16,
            bits: 4,
        });
        let request = request(transform, std::iter::empty());
        let candidate = build_candidate(preparation(source.clone()), request.clone()).unwrap();
        let target = "text_emb.weight";
        let required = candidate.requirements.executions()[0]
            .weight_lowerings()
            .iter()
            .find(|lowering| lowering.target().as_str() == target)
            .unwrap();
        let wrong = WeightLoweringDescriptor::new(
            required.descriptor().source().clone(),
            required.descriptor().executable(),
            required.scale().unwrap().physical_shape().to_vec(),
            required.descriptor().logical_shape().to_vec(),
            required.descriptor().packed_axis(),
        )
        .unwrap();
        let lowerings = candidate.requirements.executions()[0]
            .weight_lowerings()
            .iter()
            .filter(|lowering| lowering.descriptor() != required.descriptor())
            .map(|lowering| {
                WeightLoweringCapability::new(lowering.descriptor().clone(), lowering.kind())
            })
            .chain([WeightLoweringCapability::new(wrong, required.kind())])
            .collect();
        let mut capabilities = capabilities(&candidate);
        capabilities = RealtimeMechanismCapabilities::new(
            capabilities.operators(),
            candidate
                .requirements
                .mechanisms()
                .mechanisms()
                .iter()
                .copied(),
            candidate.requirements.residencies().iter().copied(),
            lowerings,
            capabilities.state().clone(),
            capabilities.maximum_tensor_parallel_size(),
            capabilities.completion().clone(),
            capabilities.session(),
        );

        let error = select_moshi_realtime(preparation(source), request, &capabilities).unwrap_err();
        let MoshiRealtimeSelectionError::Selection(error) = error else {
            panic!("wrong component-derived geometry did not reach neutral selection")
        };
        assert!(error.issues().iter().any(|issue| matches!(
            issue,
            RealtimeSelectionIssue::WeightLoweringUnavailable { target: missing, .. }
                if missing.as_str() == target
        )));
    }
}
