//! Backend-neutral replicated-text execution and session ownership.

#![allow(clippy::type_complexity)]

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    path::Path,
};

use eredu_core::cache::{
    validate_prompt_cache_model_identity, PromptCacheDescriptor, PromptCacheError,
    PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions, PromptCacheTopology,
};
use eredu_core::{DistributedCommitEpoch, DistributedCommitOutcome, DistributedCommitPhase};
use eredu_nn::{NeuralBackend, Tensor};

use crate::{
    observe_model_logits, partitioned_replicated_text_materialization_tasks,
    replicated_text_materialization_tasks, ActivationObserver, ArchitecturePartition,
    CommunicationManifest, ExecutionResidency, ExpertPass, LayerWeightResidency,
    LayeredArchitecture, LayerwisePolicy, LayerwiseRuntime, LayerwiseRuntimeError,
    ParameterGroupOwner, PartitionState, PreparedInputCacheIdentity, ReplicatedTextArchitecture,
    ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion,
    ReplicatedTextOutputSelection, ReplicatedTextParameterOwner, ReplicatedTextParameterPresence,
    RoutedExpertProvider, RoutedLayeredArchitecture, RuntimeState,
    SelectedReplicatedTextRealization, SelectedStateRealization, StateError, SubmissionBackend,
    WeightLoweringKind,
};

/// Backend mechanisms used by the generic replicated-text constructor.
///
/// Implementations allocate native state, prepare exact materialization tasks,
/// supply a bounded residency policy, persist opaque state bytes, apply a
/// requested native tensor index, and retain resources through final
/// completion. The trait receives selected values but no family identity or
/// caller selection request.
pub trait ReplicatedTextSessionMechanisms<A, B>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, Self::State>,
    Self::State: RuntimeState<B>,
    Self::ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
    Self::BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
{
    /// Concrete mutable-state realization paired with the architecture.
    type State: RuntimeState<B>;
    /// Shared failure type for resident and bounded runtime policies.
    type PolicyError;
    /// Concrete policy that owns fully resident bound units.
    type ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Concrete policy used for host-windowed or disk-streamed traversal.
    type BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Opaque state checkpoint owned by the backend mechanism.
    type StateCheckpoint;
    /// Backend-native mutable-state residency report.
    type StateReport;
    /// Backend-native parameter/runtime residency report.
    type ExecutionReport;
    /// Mechanism failure.
    type Error;

    /// Consumes the exact selected parameter tasks before runtime construction.
    #[allow(clippy::too_many_arguments)]
    fn prepare_materialization(
        &mut self,
        architecture: &mut A,
        layout: &crate::ExecutionUnitLayout,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        addressable_parameters: &[String],
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Realizes exactly the selected mutable-state components and placements.
    fn realize_state(
        &mut self,
        selected: &SelectedStateRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::State, Self::Error>;

    /// Creates the concrete policy owning fully resident bound units.
    fn resident_policy(
        &mut self,
        architecture: &mut A,
        units: Vec<A::Unit>,
        selected: &SelectedReplicatedTextRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::ResidentPolicy, Self::Error>;

    /// Creates the concrete bounded-unit policy selected for this session.
    fn bounded_policy(
        &mut self,
        architecture: &mut A,
        selected: &SelectedReplicatedTextRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::BoundedPolicy, Self::Error>;

    /// Applies one neutral sequence-axis index to a complete architecture output.
    fn index_text_output(
        &mut self,
        output: B::Tensor,
        sequence_index: i32,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, Self::Error>;

    /// Captures an opaque checkpoint of every live state component.
    fn checkpoint_state(
        &mut self,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::StateCheckpoint, Self::Error>;

    /// Restores every component from an opaque checkpoint.
    fn restore_state(
        &mut self,
        state: &mut Self::State,
        checkpoint: Self::StateCheckpoint,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Restores native state bytes from a validated prompt-cache artifact.
    fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        identity: &PromptCacheModelIdentity,
        prefix_token_ids: &[u32],
        selected: &SelectedStateRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(Self::State, PromptCacheManifest), Self::Error>;

    /// Serializes native state bytes for a neutrally validated cache identity.
    fn save_prompt_cache(
        &mut self,
        state: &mut Self::State,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, Self::Error>;

    /// Reports the realized mutable-state storage.
    fn state_report(&self, state: &Self::State) -> Result<Self::StateReport, Self::Error>;

    /// Reports the selected resident or bounded runtime realization.
    fn execution_report(
        &self,
        residency: LayerWeightResidency,
        bounded: Option<&Self::BoundedPolicy>,
    ) -> Result<Self::ExecutionReport, Self::Error>;

    /// Retains final output and mutable state through exact completion.
    fn complete(
        &mut self,
        output: &B::Tensor,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;
}

/// One typed adapter-owned operation over the authoritative prediction target.
///
/// Implementations may execute prediction-only units against target-owned static modules and the
/// currently installed lane state. They cannot replace the target architecture or take ownership
/// of its ordinary prefill/decode lifecycle.
pub trait PredictionTargetOperation<A, B, S>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    /// Operation result retained by the prediction adapter.
    type Output;

    /// Executes against the exact architecture and mutable state owned by the neutral session.
    fn apply(
        self,
        architecture: &mut A,
        state: &mut S,
        parallel: Option<&B::ParallelContext>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Self::Output, A::Error>;
}

/// Reversible prompt-cache publication used by distributed session control.
///
/// Preparation must not make the destination visible. Publication may replace
/// an existing destination, but the returned transaction must retain enough
/// ownership to restore that destination exactly until [`Self::commit_prompt_cache_save`]
/// is called. Commit and rollback are deliberately infallible: an implementation
/// that cannot provide an exact reversible publication must not implement this
/// capability.
pub trait TransactionalPromptCacheMechanisms<A, B>: ReplicatedTextSessionMechanisms<A, B>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, Self::State>,
    Self::State: RuntimeState<B>,
    Self::ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
    Self::BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>,
{
    /// Opaque staged publication retaining any superseded destination.
    type PromptCacheSaveTransaction;

    /// Serializes a candidate without publishing or replacing the destination.
    #[allow(clippy::too_many_arguments)]
    fn prepare_prompt_cache_save(
        &mut self,
        state: &mut Self::State,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::PromptCacheSaveTransaction, Self::Error>;

    /// Returns the fully validated candidate manifest before publication.
    fn prepared_prompt_cache_manifest(
        transaction: &Self::PromptCacheSaveTransaction,
    ) -> &PromptCacheManifest;

    /// Makes the staged candidate visible while retaining reversible ownership.
    fn publish_prompt_cache_save(
        &mut self,
        transaction: &mut Self::PromptCacheSaveTransaction,
    ) -> Result<(), Self::Error>;

    /// Finalizes a globally successful publication and releases its backup.
    fn commit_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction);

    /// Removes an unpublished candidate or exactly restores a published destination.
    fn rollback_prompt_cache_save(&mut self, transaction: Self::PromptCacheSaveTransaction);
}

/// Resident or bounded execution selected before construction.
enum ReplicatedTextRuntimeKind<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
{
    /// Every architecture unit remains resident.
    Resident(LayerwiseRuntime<A, B, S, R>),
    /// Units are acquired through a bounded backend policy.
    Bounded(LayerwiseRuntime<A, B, S, P>),
}

/// Resident or bounded layered runtime paired before session construction.
///
/// The wrapper lets additive execution strategies reuse one text-session
/// lifecycle without exposing the selected runtime branch or permitting a
/// backend to reconstruct it.
pub struct ReplicatedTextRuntime<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
{
    kind: ReplicatedTextRuntimeKind<A, B, S, R, P>,
}

impl<A, B, S, R, P> ReplicatedTextRuntime<A, B, S, R, P>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    fn forward_with_observer<'a, O>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime
                .forward_with_observer_and_context(input, state, context, observer)
                .map_err(map_layerwise_error),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                .forward_with_observer_and_context(input, state, context, observer)
                .map_err(map_layerwise_error),
        }
    }

    fn forward_with_provider_and_observer<'a, Provider, Observer>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        provider: &mut Provider,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut Observer,
    ) -> Result<
        (B::Tensor, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        B: eredu_nn::GroupedNeuralBackend,
        A: RoutedLayeredArchitecture<B, S>,
        Provider: RoutedExpertProvider<B>,
        Provider::Error: std::fmt::Display,
        Observer: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime
                .forward_with_provider_and_observer_and_context(
                    input, state, pass, provider, context, observer,
                )
                .map_err(map_layerwise_error),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                .forward_with_provider_and_observer_and_context(
                    input, state, pass, provider, context, observer,
                )
                .map_err(map_layerwise_error),
        }
    }

    fn bounded_policy(&self) -> Option<&P> {
        match &self.kind {
            ReplicatedTextRuntimeKind::Resident(_) => None,
            ReplicatedTextRuntimeKind::Bounded(runtime) => Some(runtime.policy()),
        }
    }

    fn prediction_target_capture(
        &mut self,
        forward: &A::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<Option<B::Tensor>, A::Error> {
        Ok(<A as crate::LayeredArchitecture<B, S>>::prediction_target_capture(forward).cloned())
    }

    fn apply_prediction_target_operation<O>(
        &mut self,
        state: &mut S,
        operation: O,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<O::Output, A::Error>
    where
        O: PredictionTargetOperation<A, B, S>,
    {
        match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => {
                operation.apply(runtime.architecture_mut(), state, None, context)
            }
            ReplicatedTextRuntimeKind::Bounded(runtime) => {
                operation.apply(runtime.architecture_mut(), state, None, context)
            }
        }
    }
}

/// Statically dispatched extension point for one replicated text unit strategy.
///
/// Ordinary execution and routed execution share the surrounding session,
/// state, prompt-cache, observation, report, rollback, and completion logic.
pub trait ReplicatedTextExecutionStrategy<A, B, S, R, P>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    R::Error: std::fmt::Display,
{
    /// Whether this strategy executes one rank of a partitioned session.
    const PARTITIONED_SESSION: bool = false;
    /// Whether control phases use a selected bounded all-rank agreement.
    const DISTRIBUTED_PHASE_AGREEMENT: bool = false;

    /// Concrete execution runtime paired before the shared session lifecycle begins.
    type Runtime;

    /// Returns bounded residency state for the shared session report.
    fn bounded_policy(runtime: &Self::Runtime) -> Option<&P>;

    /// Returns the execution residency actually installed on this rank.
    fn execution_residency(
        runtime: &Self::Runtime,
        selected: &SelectedReplicatedTextRealization,
    ) -> ExecutionResidency;

    /// Executes one complete layered pass through the selected unit strategy.
    #[allow(clippy::too_many_arguments)]
    fn forward_with_observer<'a, O>(
        &mut self,
        runtime: &mut Self::Runtime,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized;

    /// Applies the architecture's final-logits observation on the rank that
    /// owns the authoritative output. Ordinary replicated strategies observe
    /// locally; partitioned strategies may suppress the seam on destinations.
    fn observe_output<O>(
        _runtime: &mut Self::Runtime,
        output: &B::Tensor,
        observer: &mut O,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        observe_model_logits(observer, output).map_err(ReplicatedTextSessionError::Architecture)
    }

    /// Publishes an already-observed authoritative output after every rank has
    /// agreed that final observation succeeded.
    fn publish_observed_output(
        _runtime: &mut Self::Runtime,
        output: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>>
    {
        Ok(output)
    }

    /// Resolves the rank-local tensor used by an additive prediction extension.
    ///
    /// Direct execution returns the retained target value. Partitioned
    /// strategies may instead produce an exact placeholder on ranks that do
    /// not own target projection.
    fn prediction_target_capture(
        _runtime: &mut Self::Runtime,
        forward: &A::ForwardContext,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<B::Tensor>,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    > {
        Ok(<A as crate::LayeredArchitecture<B, S>>::prediction_target_capture(forward).cloned())
    }

    /// Publishes the output-owner capture to every prediction participant.
    fn publish_prediction_target_capture(
        _runtime: &mut Self::Runtime,
        capture: B::Tensor,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>>
    {
        Ok(capture)
    }

    /// Runs one typed prediction-only operation against session-owned target modules and state.
    fn apply_prediction_target_operation<O>(
        _runtime: &mut Self::Runtime,
        _state: &mut S,
        _operation: O,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<O::Output>,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: PredictionTargetOperation<A, B, S>,
    {
        Ok(None)
    }

    /// Performs strategy-specific distributed commit only after output
    /// intervention and exact mechanism completion have succeeded.
    fn commit_after_completion(
        _runtime: &mut Self::Runtime,
        epoch: DistributedCommitEpoch,
        _context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> DistributedCommitOutcome {
        DistributedCommitOutcome::Committed(epoch)
    }

    /// Propagates one local shared-session phase result before the lifecycle
    /// can advance. Direct and unsupported strategies retain the local result.
    fn agree_distributed_phase(
        _runtime: &mut Self::Runtime,
        _phase: crate::DistributedExecutionPhase,
        local_success: bool,
        _context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>>
    {
        Ok(local_success)
    }
}

/// Narrow constructor seam for strategies that use the ordinary full replicated runtime.
///
/// Partitioned strategies intentionally do not implement this trait; their rank-local runtime is
/// supplied through the partitioned constructor instead of accepting an impossible full-runtime
/// conversion.
pub trait ReplicatedRuntimeExecutionStrategy<A, B, S, R, P>:
    ReplicatedTextExecutionStrategy<A, B, S, R, P, Runtime = ReplicatedTextRuntime<A, B, S, R, P>>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    R::Error: std::fmt::Display,
{
}

/// Ordinary unit execution for replicated text architectures.
#[derive(Debug, Default, Clone, Copy)]
pub struct DirectReplicatedTextExecution;

impl<A, B, S, R, P> ReplicatedTextExecutionStrategy<A, B, S, R, P> for DirectReplicatedTextExecution
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    type Runtime = ReplicatedTextRuntime<A, B, S, R, P>;

    fn bounded_policy(runtime: &Self::Runtime) -> Option<&P> {
        runtime.bounded_policy()
    }

    fn execution_residency(
        _runtime: &Self::Runtime,
        selected: &SelectedReplicatedTextRealization,
    ) -> ExecutionResidency {
        selected.residency().execution_residency()
    }

    fn forward_with_observer<'a, O>(
        &mut self,
        runtime: &mut Self::Runtime,
        input: A::Input<'a>,
        state: &mut S,
        _pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.forward_with_observer(input, state, context, observer)
    }

    fn prediction_target_capture(
        runtime: &mut Self::Runtime,
        forward: &A::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<B::Tensor>,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    > {
        runtime
            .prediction_target_capture(forward, context)
            .map_err(ReplicatedTextSessionError::Architecture)
    }

    fn apply_prediction_target_operation<O>(
        runtime: &mut Self::Runtime,
        state: &mut S,
        operation: O,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        Option<O::Output>,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: PredictionTargetOperation<A, B, S>,
    {
        runtime
            .apply_prediction_target_operation(state, operation, context)
            .map(Some)
            .map_err(ReplicatedTextSessionError::Architecture)
    }
}

impl<A, B, S, R, P> ReplicatedRuntimeExecutionStrategy<A, B, S, R, P>
    for DirectReplicatedTextExecution
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
}

/// Provider-backed routed unit execution using the shared replicated session.
pub struct RoutedReplicatedTextExecution<P> {
    provider: P,
}

impl<P> RoutedReplicatedTextExecution<P> {
    /// Creates routed unit execution from one neutral provider strategy.
    pub const fn new(provider: P) -> Self {
        Self { provider }
    }

    /// Returns the live provider for mechanism telemetry and reports.
    pub const fn provider(&self) -> &P {
        &self.provider
    }
}

impl<A, B, S, R, P, Provider> ReplicatedTextExecutionStrategy<A, B, S, R, P>
    for RoutedReplicatedTextExecution<Provider>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::GroupedNeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S> + RoutedLayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    Provider: RoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
    type Runtime = ReplicatedTextRuntime<A, B, S, R, P>;

    fn bounded_policy(runtime: &Self::Runtime) -> Option<&P> {
        runtime.bounded_policy()
    }

    fn execution_residency(
        _runtime: &Self::Runtime,
        selected: &SelectedReplicatedTextRealization,
    ) -> ExecutionResidency {
        selected.residency().execution_residency()
    }

    fn forward_with_observer<'a, O>(
        &mut self,
        runtime: &mut Self::Runtime,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.forward_with_provider_and_observer(
            input,
            state,
            pass,
            &mut self.provider,
            context,
            observer,
        )
    }
}

impl<A, B, S, R, P, Provider> ReplicatedRuntimeExecutionStrategy<A, B, S, R, P>
    for RoutedReplicatedTextExecution<Provider>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + eredu_nn::GroupedNeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S> + RoutedLayeredArchitecture<B, S>,
    R: LayerwisePolicy<B, A::Unit>,
    P: LayerwisePolicy<B, A::Unit, Error = R::Error>,
    Provider: RoutedExpertProvider<B>,
    Provider::Error: std::fmt::Display,
    A::Error: std::fmt::Display,
    P::Error: std::fmt::Display,
{
}

/// Complete backend-neutral replicated-text session.
pub struct ReplicatedTextSession<A, B, M, D = DirectReplicatedTextExecution>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
{
    selected: SelectedReplicatedTextRealization,
    selected_state: SessionStateRealization,
    execution: D::Runtime,
    driver: D,
    state: M::State,
    mechanisms: M,
    prompt_cache_identity: Option<PromptCacheModelIdentity>,
    committed_prompt_input_identity: Option<PreparedInputCacheIdentity>,
    next_commit_epoch: DistributedCommitEpoch,
    active_commit_epoch: Option<DistributedCommitEpoch>,
    last_commit_outcome: Option<DistributedCommitOutcome>,
    control_fence: Option<crate::DistributedExecutionPhase>,
    output_selection: ReplicatedTextOutputSelection,
    backend: PhantomData<fn() -> B>,
}

/// Exact mutable-state ownership bound to one shared text session.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SessionStateRealization {
    /// This rank owns the selected local state components and geometry.
    Stateful(SelectedStateRealization),
    /// This rank owns no mutable state components or prompt-cache shard payload.
    Stateless,
}

/// Rank-local runtime, state, and cache identity prepared by partition construction.
///
/// The shared session consumes this value so partition execution reuses the ordinary
/// checkpoint, rollback, reset, observation, publication, and reporting lifecycle.
pub struct PreparedPartitionedSessionRuntime<R, S> {
    selected: SelectedReplicatedTextRealization,
    runtime: R,
    state: S,
    selected_state: SessionStateRealization,
    prompt_cache_identity: Option<PromptCacheModelIdentity>,
    output_selection: ReplicatedTextOutputSelection,
}

impl<R, S> PreparedPartitionedSessionRuntime<R, S> {
    /// Returns the architecture-derived prompt-cache identity retained by this
    /// exact partition, when the rank owns mutable prompt state.
    pub const fn prompt_cache_identity(&self) -> Option<&PromptCacheModelIdentity> {
        self.prompt_cache_identity.as_ref()
    }
}

/// Exact architecture, partition, communication manifest, and payload work received by a
/// partition-runtime factory.
///
/// This value is assembled only after the architecture has been checked against its partition
/// and the payload tasks have been re-derived from that same architecture/partition pair. A
/// factory consumes all four authorities together instead of receiving independently assembled
/// runtime inputs.
pub struct PartitionedSessionFactoryInput<A, G, W> {
    architecture: A,
    partition: ArchitecturePartition<G, W>,
    communication: CommunicationManifest,
    tasks: Vec<ReplicatedTextMaterializationTask>,
}

impl<A, G, W> PartitionedSessionFactoryInput<A, G, W> {
    /// Exact architecture-owned payload tasks for this rank.
    pub fn materialization_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        &self.tasks
    }

    /// Consumes the causal handoff into the values needed to build the rank-local runtime.
    pub fn into_parts(
        self,
    ) -> (
        A,
        ArchitecturePartition<G, W>,
        CommunicationManifest,
        Vec<ReplicatedTextMaterializationTask>,
    ) {
        (
            self.architecture,
            self.partition,
            self.communication,
            self.tasks,
        )
    }
}

/// Failure while consuming architecture authority into a rank-local runtime.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedSessionPreparationError<E> {
    /// Architecture, partition, selection, or payload authority disagreed.
    #[error("partitioned session authority mismatch: {0}")]
    Contract(String),
    /// The rank-local runtime factory rejected the exact handoff.
    #[error("partitioned session runtime factory failed: {0}")]
    Factory(E),
}

/// Consumes exact architecture authority into a rank-local runtime and state.
///
/// The selected realization, architecture, partition, communication manifest, and optional
/// architecture-precomputed task proof are consumed in one operation. Payload work is re-derived
/// from the consumed architecture and partition before the factory runs. A task proof, when
/// supplied, must match that derivation exactly. The factory therefore cannot be paired with a
/// different architecture after admission.
#[allow(clippy::too_many_arguments)]
pub fn prepare_partitioned_session_runtime<A, B, R, S, G, W, E, F>(
    architecture: A,
    selected: SelectedReplicatedTextRealization,
    partition: ArchitecturePartition<G, W>,
    communication: CommunicationManifest,
    expected_tasks: Option<&[ReplicatedTextMaterializationTask]>,
    topology: PromptCacheTopology,
    output_selection: ReplicatedTextOutputSelection,
    context: &<B::Tensor as Tensor>::Context,
    factory: F,
) -> Result<PreparedPartitionedSessionRuntime<R, S>, PartitionedSessionPreparationError<E>>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    F: FnOnce(
        PartitionedSessionFactoryInput<A, G, W>,
        &SelectedReplicatedTextRealization,
        &<B::Tensor as Tensor>::Context,
    ) -> Result<(R, S), E>,
{
    prepare_partitioned_session_runtime_with_exclusions(
        architecture,
        selected,
        partition,
        communication,
        expected_tasks,
        &std::collections::BTreeSet::new(),
        topology,
        output_selection,
        context,
        factory,
    )
}

/// Consumes partition authority while excluding exact parameters supplied by
/// an independently addressable store from ordinary materialization.
#[allow(clippy::too_many_arguments)]
pub fn prepare_partitioned_session_runtime_with_exclusions<A, B, R, S, G, W, E, F>(
    architecture: A,
    selected: SelectedReplicatedTextRealization,
    partition: ArchitecturePartition<G, W>,
    communication: CommunicationManifest,
    expected_tasks: Option<&[ReplicatedTextMaterializationTask]>,
    excluded_parameter_targets: &std::collections::BTreeSet<&str>,
    topology: PromptCacheTopology,
    output_selection: ReplicatedTextOutputSelection,
    context: &<B::Tensor as Tensor>::Context,
    factory: F,
) -> Result<PreparedPartitionedSessionRuntime<R, S>, PartitionedSessionPreparationError<E>>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
    F: FnOnce(
        PartitionedSessionFactoryInput<A, G, W>,
        &SelectedReplicatedTextRealization,
        &<B::Tensor as Tensor>::Context,
    ) -> Result<(R, S), E>,
{
    partition
        .validate_architecture::<B, S, A>(&architecture)
        .map_err(|error| PartitionedSessionPreparationError::Contract(error.to_string()))?;
    let parameters = architecture
        .parameter_description(context)
        .map_err(|error| PartitionedSessionPreparationError::Contract(error.to_string()))?;
    let mut tasks =
        partitioned_replicated_text_materialization_tasks(&selected, &parameters, &partition)
            .map_err(|error| PartitionedSessionPreparationError::Contract(error.to_string()))?;
    tasks.retain(|task| !excluded_parameter_targets.contains(task.name()));
    if expected_tasks.is_some_and(|expected| expected != tasks) {
        let derived_names = tasks
            .iter()
            .map(ReplicatedTextMaterializationTask::name)
            .collect::<std::collections::BTreeSet<_>>();
        let first_missing = expected_tasks
            .expect("task proof was checked as present")
            .iter()
            .map(ReplicatedTextMaterializationTask::name)
            .find(|name| !derived_names.contains(name));
        let parameter_group = first_missing.and_then(|name| {
            parameters
                .groups()
                .iter()
                .find(|group| group.members().iter().any(|member| member.target() == name))
        });
        let partition_group = first_missing.and_then(|name| {
            partition
                .parameter_bindings()
                .iter()
                .find(|group| group.members().iter().any(|member| member.target() == name))
        });
        return Err(PartitionedSessionPreparationError::Contract(
            format!(
                "precomputed local materialization tasks differ from consumed partition authority: expected {:?}, derived {:?}, first missing current group {parameter_group:?}, admitted group {partition_group:?}",
                expected_tasks
                    .expect("task proof was checked as present")
                    .iter()
                    .map(ReplicatedTextMaterializationTask::name)
                    .collect::<Vec<_>>(),
                tasks
                    .iter()
                    .map(ReplicatedTextMaterializationTask::name)
                    .collect::<Vec<_>>()
            ),
        ));
    }
    let partition_state = partition.state().cloned();
    let (selected_state, prompt_cache_identity) = match partition_state.as_ref() {
        Some(partition_state) => (
            SessionStateRealization::Stateful(
                selected
                    .state()
                    .for_partitioned_geometry(partition_state)
                    .map_err(|error| {
                        PartitionedSessionPreparationError::Contract(error.to_string())
                    })?,
            ),
            Some(
                partition_state
                    .prompt_cache_identity::<B, A>(&architecture, topology)
                    .map_err(|error| {
                        PartitionedSessionPreparationError::Contract(error.to_string())
                    })?,
            ),
        ),
        None => (SessionStateRealization::Stateless, None),
    };
    let (runtime, state) = factory(
        PartitionedSessionFactoryInput {
            architecture,
            partition,
            communication,
            tasks,
        },
        &selected,
        context,
    )
    .map_err(PartitionedSessionPreparationError::Factory)?;
    match selected_state.state() {
        Some(local) if state.optional_layout() != Some(local.layout()) => {
            return Err(PartitionedSessionPreparationError::Contract(
                "partition runtime state differs from canonical local geometry".into(),
            ));
        }
        None if state.optional_layout().is_some() => {
            return Err(PartitionedSessionPreparationError::Contract(
                "stateless partition binding contains mutable state geometry".into(),
            ));
        }
        _ => {}
    }
    Ok(PreparedPartitionedSessionRuntime {
        selected,
        runtime,
        state,
        selected_state,
        prompt_cache_identity,
        output_selection,
    })
}

impl SessionStateRealization {
    /// Returns the selected local state realization when this rank owns state.
    pub const fn state(&self) -> Option<&SelectedStateRealization> {
        match self {
            Self::Stateful(state) => Some(state),
            Self::Stateless => None,
        }
    }
}

/// Complete transactional checkpoint including composite prompt-input identity.
pub struct ReplicatedTextSessionCheckpoint<C> {
    state: C,
    prompt_input_identity: Option<PreparedInputCacheIdentity>,
    next_commit_epoch: DistributedCommitEpoch,
    last_commit_outcome: Option<DistributedCommitOutcome>,
}

/// Rank-local state checkpoint whose presence was agreed by a partitioned session.
pub struct DistributedStateCheckpoint<C> {
    state: Option<C>,
}

/// Complete rank-local checkpoint whose presence was agreed by a partitioned session.
pub struct DistributedSessionCheckpoint<C> {
    state: Option<C>,
    prompt_input_identity: Option<PreparedInputCacheIdentity>,
    next_commit_epoch: DistributedCommitEpoch,
    last_commit_outcome: Option<DistributedCommitOutcome>,
}

/// Cold-path failure from replicated-text construction or session control.
#[derive(Debug, thiserror::Error)]
pub enum ReplicatedTextSessionError<A, P, M>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    /// The prepared architecture disagreed with its selected realization.
    #[error("replicated text contract mismatch: {0}")]
    Contract(String),
    /// Architecture construction or execution failed.
    #[error("replicated text architecture failed: {0}")]
    Architecture(A),
    /// Bounded residency failed.
    #[error("replicated text residency failed: {0}")]
    Policy(P),
    /// A native mechanism failed.
    #[error("replicated text mechanism failed: {0}")]
    Mechanism(M),
    /// Mutable-state access failed.
    #[error(transparent)]
    State(#[from] StateError),
    /// Prompt-cache identity or manifest validation failed.
    #[error(transparent)]
    PromptCache(#[from] PromptCacheError),
    /// The globally fixed final decision was an abort.
    #[error("distributed transaction epoch {epoch:?} was aborted")]
    CommitAborted {
        /// Durable transaction identity.
        epoch: DistributedCommitEpoch,
    },
    /// This rank may have contributed to a final decision it could not observe.
    #[error("distributed transaction epoch {epoch:?} is indeterminate at {phase:?}")]
    CommitIndeterminate {
        /// Durable transaction identity.
        epoch: DistributedCommitEpoch,
        /// Exact final-decision cut which was not observed.
        phase: DistributedCommitPhase,
    },
}

/// Immutable session and residency report produced by neutral orchestration.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ReplicatedTextSessionReport<E, S> {
    execution: ExecutionResidency,
    execution_report: E,
    state_report: S,
    distributed_commit: Option<DistributedCommitOutcome>,
}

/// Opaque proof that one concrete architecture agrees with an authoritative
/// replicated-text selection and its exact materialization tasks.
///
/// The value can only be created by [`prepare_replicated_text_contract`].
pub struct PreparedReplicatedTextContract {
    selected: SelectedReplicatedTextRealization,
    tasks: Vec<ReplicatedTextMaterializationTask>,
    addressable_parameters: Vec<String>,
    prompt_cache_identity: PromptCacheModelIdentity,
    output_selection: ReplicatedTextOutputSelection,
}

impl PreparedReplicatedTextContract {
    /// Returns the authoritative selected realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        &self.selected
    }

    /// Returns the validated exact materialization tasks.
    pub fn materialization_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        &self.tasks
    }

    /// Returns the architecture-derived identity coupled to this proof.
    pub const fn prompt_cache_identity(&self) -> &PromptCacheModelIdentity {
        &self.prompt_cache_identity
    }

    /// Returns the architecture-declared causal output projection.
    pub const fn output_selection(&self) -> ReplicatedTextOutputSelection {
        self.output_selection
    }

    fn into_parts(
        self,
    ) -> (
        SelectedReplicatedTextRealization,
        Vec<ReplicatedTextMaterializationTask>,
        Vec<String>,
        PromptCacheModelIdentity,
        ReplicatedTextOutputSelection,
    ) {
        (
            self.selected,
            self.tasks,
            self.addressable_parameters,
            self.prompt_cache_identity,
            self.output_selection,
        )
    }
}

/// Validates a concrete architecture against its authoritative selection and
/// produces the unforgeable contract consumed by the neutral constructor.
pub fn prepare_replicated_text_contract<A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextContract, String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    prepare_replicated_text_contract_with_addressable_parameters::<A, B, S>(
        architecture,
        source_architecture,
        selected,
        expected_prompt_cache_architecture_identity,
        std::iter::empty::<&str>(),
        context,
    )
}

/// Validates a concrete architecture while assigning an exact parameter set
/// to independently addressable storage.
///
/// Addressable parameters remain part of full topology, shape, owner, source,
/// and executable-format validation. Only their ordinary materialization tasks
/// are removed after that validation succeeds.
pub fn prepare_replicated_text_contract_with_addressable_parameters<'a, A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    addressable_parameters: impl IntoIterator<Item = &'a str>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextContract, String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: ReplicatedTextArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    prepare_layered_text_contract_with_addressable_parameters::<A, B, S>(
        architecture,
        source_architecture,
        selected,
        expected_prompt_cache_architecture_identity,
        architecture.text_output_selection(),
        addressable_parameters,
        context,
    )
}

/// Validates a layered causal architecture against an authoritative text selection.
///
/// Composite ingress supplies its architecture-owned input directly, while this
/// proof preserves the same parameter, state, identity, and output-selection
/// authority as ordinary text construction.
pub fn prepare_layered_text_contract<A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    output_selection: ReplicatedTextOutputSelection,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextContract, String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    prepare_layered_text_contract_with_addressable_parameters::<A, B, S>(
        architecture,
        source_architecture,
        selected,
        expected_prompt_cache_architecture_identity,
        output_selection,
        std::iter::empty::<&str>(),
        context,
    )
}

/// Validates a layered causal architecture with independently addressable parameters.
pub fn prepare_layered_text_contract_with_addressable_parameters<'a, A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    output_selection: ReplicatedTextOutputSelection,
    addressable_parameters: impl IntoIterator<Item = &'a str>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextContract, String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let mut addressable_parameters = addressable_parameters
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    validate_selected_state(&selected)?;
    validate_architecture_geometry::<A, B, S>(architecture, &selected)?;
    if let Some(source) = source_architecture {
        validate_architecture_geometry::<A, B, S>(source, &selected)?;
    }
    let has_transform = selected.parameters().iter().any(|parameter| {
        matches!(
            parameter.lowering(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        )
    });
    if has_transform != source_architecture.is_some() {
        return Err(
            "selected transform tasks and source-format architecture ownership disagree".into(),
        );
    }
    if let Some(source) = source_architecture {
        validate_architecture_parameters::<A, B, S>(source, &selected, false, context)?;
    }
    let mut output_companions =
        validate_architecture_parameters::<A, B, S>(architecture, &selected, true, context)?;
    let mut tasks =
        replicated_text_materialization_tasks(&selected).map_err(|error| error.to_string())?;
    let selected_parameter_names = tasks
        .iter()
        .map(|task| task.name().to_owned())
        .collect::<BTreeSet<_>>();
    if !addressable_parameters.is_subset(&selected_parameter_names) {
        return Err(format!(
            "addressable parameter catalog contains unknown selected parameters: {:?}",
            addressable_parameters
                .difference(&selected_parameter_names)
                .collect::<Vec<_>>()
        ));
    }
    let addressable_companions = addressable_parameters
        .iter()
        .filter_map(|name| output_companions.get(name))
        .flatten()
        .map(|companion| companion.name().to_owned())
        .collect::<Vec<_>>();
    addressable_parameters.extend(addressable_companions);
    let companion_names = output_companions
        .values()
        .flatten()
        .map(|companion| companion.name().to_owned())
        .collect::<BTreeSet<_>>();
    let companion_tasks = tasks
        .iter()
        .filter(|task| companion_names.contains(task.name()))
        .map(|task| (task.name().to_owned(), task.clone()))
        .collect::<BTreeMap<_, _>>();
    for task in &mut tasks {
        let companions = output_companions
            .remove(task.name())
            .unwrap_or_default()
            .into_iter()
            .map(
                |companion| match companion_tasks.get(companion.name()).cloned() {
                    Some(exact) => companion
                        .with_materialization_task(exact)
                        .map_err(|error| error.to_string()),
                    None => Ok(companion),
                },
            )
            .collect::<Result<Vec<_>, String>>()?;
        task.set_output_companions(companions)
            .map_err(|error| error.to_string())?;
    }
    if !output_companions.is_empty() {
        return Err(format!(
            "output companion catalog contains unknown materialization tasks: {:?}",
            output_companions.keys().collect::<Vec<_>>()
        ));
    }
    // Checkpoint-native companions participate in selected-parameter
    // validation, but their primary packed output owns their one causal
    // materialization task. Do not consume them again as standalone tasks.
    tasks.retain(|task| !companion_names.contains(task.name()));
    tasks.retain(|task| !addressable_parameters.contains(task.name()));
    let state = PartitionState::new(selected.state().layout().clone(), 0)
        .map_err(|error| error.to_string())?;
    let prompt_cache_identity = state
        .prompt_cache_identity::<B, A>(architecture, Default::default())
        .map_err(|error| error.to_string())?;
    if prompt_cache_identity.architecture_fingerprint()
        != expected_prompt_cache_architecture_identity
        || prompt_cache_identity.layer_count() != selected.state().layout().len()
        || prompt_cache_identity.global_layer_start() != 0
        || prompt_cache_identity.global_layer_end() != selected.state().layout().len()
        || prompt_cache_identity.topology() != &Default::default()
    {
        return Err("architecture prompt-cache identity differs from selection".into());
    }
    Ok(PreparedReplicatedTextContract {
        selected,
        tasks,
        addressable_parameters: addressable_parameters.into_iter().collect(),
        prompt_cache_identity,
        output_selection,
    })
}

impl<E, S> ReplicatedTextSessionReport<E, S> {
    /// Returns the selected resident or bounded execution class.
    pub const fn execution(&self) -> ExecutionResidency {
        self.execution
    }

    /// Returns backend-native parameter/runtime residency details.
    pub const fn execution_report(&self) -> &E {
        &self.execution_report
    }

    /// Returns backend-native mutable-state residency details.
    pub const fn state_report(&self) -> &S {
        &self.state_report
    }

    /// Returns this rank's durable observation of the latest transaction decision.
    pub const fn distributed_commit(&self) -> Option<DistributedCommitOutcome> {
        self.distributed_commit
    }
}

/// Constructs one complete replicated-text session from selected policy and
/// mechanism implementations.
pub fn construct_replicated_text_session<A, B, M>(
    architecture: A,
    source_architecture: Option<A>,
    prepared: PreparedReplicatedTextContract,
    mechanisms: M,
    context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
) -> Result<
    ReplicatedTextSession<A, B, M>,
    ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    construct_replicated_text_session_with_execution(
        architecture,
        source_architecture,
        prepared,
        mechanisms,
        DirectReplicatedTextExecution,
        context,
    )
}

/// Constructs one replicated text session with an additive unit-execution strategy.
///
/// Architecture-owned prepared execution classes use this shared entry point
/// after validating their additional proof. The surrounding lifecycle remains
/// identical to ordinary replicated text construction.
pub fn construct_replicated_text_session_with_execution<A, B, M, D>(
    mut architecture: A,
    mut source_architecture: Option<A>,
    prepared: PreparedReplicatedTextContract,
    mut mechanisms: M,
    driver: D,
    context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
) -> Result<
    ReplicatedTextSession<A, B, M, D>,
    ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedRuntimeExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let (selected, tasks, addressable_parameters, prompt_cache_identity, output_selection) =
        prepared.into_parts();
    let mut units = construct_units::<A, B, M::State>(
        &architecture,
        selected.requirements().execution_units(),
        context,
    )
    .map_err(ReplicatedTextSessionError::Architecture)?;
    let mut source_units = source_architecture
        .as_ref()
        .map(|source| {
            construct_units::<A, B, M::State>(
                source,
                selected.requirements().execution_units(),
                context,
            )
        })
        .transpose()
        .map_err(ReplicatedTextSessionError::Architecture)?;
    mechanisms
        .prepare_materialization(
            &mut architecture,
            selected.requirements().execution_units(),
            &mut units,
            source_architecture.as_mut(),
            source_units.as_deref_mut(),
            &tasks,
            &addressable_parameters,
            context,
        )
        .map_err(ReplicatedTextSessionError::Mechanism)?;
    let state = mechanisms
        .realize_state(selected.state(), context)
        .map_err(ReplicatedTextSessionError::Mechanism)?;
    validate_realized_state(&state, selected.state())?;
    let selected_state = SessionStateRealization::Stateful(selected.state().clone());
    let execution = match selected.residency() {
        LayerWeightResidency::FullyResident => {
            let policy = mechanisms
                .resident_policy(&mut architecture, units, &selected, context)
                .map_err(ReplicatedTextSessionError::Mechanism)?;
            ReplicatedTextRuntime {
                kind: ReplicatedTextRuntimeKind::Resident(LayerwiseRuntime::new(
                    architecture,
                    policy,
                )),
            }
        }
        LayerWeightResidency::LayerwiseHost(_) | LayerWeightResidency::DenseDiskStream(_) => {
            let policy = mechanisms
                .bounded_policy(&mut architecture, &selected, context)
                .map_err(ReplicatedTextSessionError::Mechanism)?;
            ReplicatedTextRuntime {
                kind: ReplicatedTextRuntimeKind::Bounded(LayerwiseRuntime::new(
                    architecture,
                    policy,
                )),
            }
        }
    };
    Ok(ReplicatedTextSession {
        selected,
        selected_state,
        execution,
        driver,
        state,
        mechanisms,
        prompt_cache_identity: Some(prompt_cache_identity),
        committed_prompt_input_identity: None,
        next_commit_epoch: DistributedCommitEpoch::FIRST,
        active_commit_epoch: None,
        last_commit_outcome: None,
        control_fence: None,
        output_selection,
        backend: PhantomData,
    })
}

/// Constructs a partitioned strategy through the ordinary replicated-text session lifecycle.
///
/// Rank-local architecture construction prepares `binding`; this function only validates its
/// state/cache ownership and installs it behind the same session implementation used by ordinary
/// replicated execution. Partition strategies need not and cannot manufacture a full-graph
/// [`ReplicatedTextRuntime`].
pub fn construct_replicated_text_session_with_runtime<A, B, M, D>(
    binding: PreparedPartitionedSessionRuntime<D::Runtime, M::State>,
    mechanisms: M,
    driver: D,
) -> Result<
    ReplicatedTextSession<A, B, M, D>,
    ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    let PreparedPartitionedSessionRuntime {
        selected,
        runtime,
        state,
        selected_state,
        prompt_cache_identity,
        output_selection,
    } = binding;
    match selected_state.state() {
        Some(local) => {
            validate_realized_state(&state, local)?;
            let identity = prompt_cache_identity.as_ref().ok_or_else(|| {
                ReplicatedTextSessionError::Contract(
                    "stateful partition is missing its rank-local cache identity".into(),
                )
            })?;
            let local_layers = identity
                .global_layer_end()
                .checked_sub(identity.global_layer_start());
            if identity.layer_count() != selected.state().layout().len()
                || local_layers != Some(local.layout().len())
                || identity.global_layer_end() > identity.layer_count()
            {
                return Err(ReplicatedTextSessionError::Contract(
                    "rank-local cache identity differs from selected state geometry".into(),
                ));
            }
            let partition =
                PartitionState::new(local.layout().clone(), identity.global_layer_start())
                    .map_err(|error| ReplicatedTextSessionError::Contract(error.to_string()))?;
            let expected = selected
                .state()
                .for_partitioned_geometry(&partition)
                .map_err(|error| ReplicatedTextSessionError::Contract(error.to_string()))?;
            if &expected != local {
                return Err(ReplicatedTextSessionError::Contract(
                    "rank-local state realization is not the selected global interval".into(),
                ));
            }
        }
        None => {
            if prompt_cache_identity.is_some() || state.optional_layout().is_some() {
                return Err(ReplicatedTextSessionError::Contract(
                    "stateless partition owns state or a prompt-cache shard identity".into(),
                ));
            }
        }
    }
    Ok(ReplicatedTextSession {
        selected,
        selected_state,
        execution: runtime,
        driver,
        state,
        mechanisms,
        prompt_cache_identity,
        committed_prompt_input_identity: None,
        next_commit_epoch: DistributedCommitEpoch::FIRST,
        active_commit_epoch: None,
        last_commit_outcome: None,
        control_fence: None,
        output_selection,
        backend: PhantomData,
    })
}

fn construct_units<A, B, S>(
    architecture: &A,
    layout: &crate::ExecutionUnitLayout,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<Vec<A::Unit>, A::Error>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
{
    (0..layout.len())
        .map(|ordinal| {
            let address = layout
                .address(ordinal)
                .expect("validated replicated layout contains every ordinal");
            architecture.build_unit(address.group(), address.index(), context)
        })
        .collect()
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Borrows the statically paired unit-execution strategy for generic telemetry.
    pub const fn execution_strategy(&self) -> &D {
        &self.driver
    }

    /// Runs one direct forward and returns the complete architecture output.
    pub fn forward(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        A: ReplicatedTextArchitecture<B, M::State>,
    {
        self.forward_with_observer(tokens, mask, context, &mut crate::NoopObserver)
    }

    /// Runs one direct forward with unit and final-logits observation and intervention.
    pub fn forward_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        A: ReplicatedTextArchitecture<B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let pass = tokens
            .shape()
            .last()
            .copied()
            .filter(|length| *length > 1)
            .map_or(ExpertPass::Decode, |_| ExpertPass::Prefill);
        let (output, checkpoint, forward_context) =
            self.execute_with_observer(tokens, mask, pass, context, observer)?;
        self.publish(output, checkpoint, forward_context, context)
    }

    /// Runs prompt processing and selects the architecture-declared text output.
    pub fn prefill(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        A: ReplicatedTextArchitecture<B, M::State>,
    {
        self.prefill_with_observer(tokens, mask, context, &mut crate::NoopObserver)
    }

    /// Runs observed prompt processing and selects the declared text output.
    pub fn prefill_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        A: ReplicatedTextArchitecture<B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let input = A::text_input(tokens, mask);
        self.prefill_input_with_observer(input, context, observer)
    }

    /// Runs ordinary target prefill and returns its architecture-owned prediction capture.
    ///
    /// Both tensors come from the same transaction and are returned only after
    /// canonical output publication succeeds.  Missing capture rolls state back
    /// exactly like a failed output projection.
    pub fn prefill_prediction_target(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        (B::Tensor, B::Tensor),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        A: ReplicatedTextArchitecture<B, M::State>,
    {
        let input = A::text_input(tokens, mask);
        self.prefill_input_prediction_target(input, context)
    }

    /// Runs architecture-prepared target prefill and returns its exact additive capture.
    pub fn prefill_input_prediction_target<'a>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        (B::Tensor, B::Tensor),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let (output, checkpoint, forward_context) = self.execute_input_before_publication(
            input,
            ExpertPass::Prefill,
            context,
            &mut crate::NoopObserver,
        )?;
        let capture = D::prediction_target_capture(&mut self.execution, &forward_context, context)
            .map_err(widen_infallible);
        let local_success = matches!(&capture, Ok(Some(_)));
        let agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::PredictionTargetCapture,
            local_success,
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        let capture = match capture {
            Ok(Some(capture)) if agreed => capture,
            Ok(Some(_)) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank could not prepare the prediction target capture".into(),
                    ),
                    context,
                )
            }
            Ok(None) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "prediction target pass did not retain its declared hidden capture".into(),
                    ),
                    context,
                )
            }
            Err(error) => return self.rollback_failure(checkpoint, error, context),
        };
        let capture_publication =
            D::publish_prediction_target_capture(&mut self.execution, capture, context);
        let capture_publication_agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::PredictionTargetCapturePublication,
            capture_publication.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        let capture = match capture_publication {
            Ok(capture) if capture_publication_agreed => capture,
            Ok(_) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank failed to publish the prediction target capture".into(),
                    ),
                    context,
                )
            }
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        let (output, checkpoint, forward_context) =
            self.publish_observed_output_transaction(output, checkpoint, forward_context, context)?;
        self.publish(output, checkpoint, forward_context, context)
            .map(|output| (output, capture))
    }

    /// Runs prompt processing from an architecture-prepared input.
    ///
    /// Additive ingress drivers use this entry after architecture admission has
    /// coupled native tensors to their semantic identity. Output selection,
    /// rollback, observation, state publication, and completion remain owned by
    /// this session.
    pub fn prefill_input<'a>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.prefill_input_with_observer(input, context, &mut crate::NoopObserver)
    }

    /// Runs one ordinary non-partitioned target pass and atomically retains an additive capture.
    ///
    /// The capture is derived from the same forward context and observed unit outputs as the
    /// canonical target logits. Capture failure restores target state before either value is
    /// published. Partitioned capture requires an admitted multi-tensor publication contract and
    /// therefore remains unavailable through this local-only seam.
    pub fn prefill_input_with_capture<'a, O, C, F>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
        capture: F,
    ) -> Result<(B::Tensor, C), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        F: FnOnce(&A::ForwardContext) -> Result<C, A::Error>,
    {
        if D::PARTITIONED_SESSION {
            return Err(ReplicatedTextSessionError::Contract(
                "partitioned prediction capture requires a selected bundle publication contract"
                    .into(),
            ));
        }
        let (output, checkpoint, forward_context) =
            self.execute_input_before_publication(input, ExpertPass::Prefill, context, observer)?;
        let captured = match capture(&forward_context) {
            Ok(captured) => captured,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Architecture(error),
                    context,
                )
            }
        };
        let (output, checkpoint, forward_context) =
            self.publish_observed_output_transaction(output, checkpoint, forward_context, context)?;
        self.publish(output, checkpoint, forward_context, context)
            .map(|output| (output, captured))
    }

    /// Runs composite prompt processing and commits its cache-relevant input identity only after
    /// successful state publication and exact completion.
    pub fn prefill_input_with_cache_identity<'a>(
        &mut self,
        input: A::Input<'a>,
        identity: PreparedInputCacheIdentity,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.prefill_input_with_observer_and_cache_identity(
            input,
            identity,
            context,
            &mut crate::NoopObserver,
        )
    }

    /// Runs observed prompt processing and commits its exact prepared-input identity on success.
    pub fn prefill_input_with_observer_and_cache_identity<'a, O>(
        &mut self,
        input: A::Input<'a>,
        identity: PreparedInputCacheIdentity,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let output = self.prefill_input_with_observer(input, context, observer)?;
        self.committed_prompt_input_identity = Some(identity);
        Ok(output)
    }

    /// Runs observed prompt processing from an architecture-prepared input.
    pub fn prefill_input_with_observer<'a, O>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint, forward_context) =
            self.execute_input_with_observer(input, ExpertPass::Prefill, context, observer)?;
        let sequence_index = self.output_selection.sequence_index();
        let output = match self
            .mechanisms
            .index_text_output(output, sequence_index, context)
        {
            Ok(output) => output,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Mechanism(error),
                    context,
                )
            }
        };
        self.publish(output, checkpoint, forward_context, context)
    }

    /// Runs one decode step from an architecture-prepared input.
    pub fn decode_input<'a>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.decode_input_with_observer(input, context, &mut crate::NoopObserver)
    }

    /// Runs one ordinary non-partitioned decode pass and atomically retains an additive capture.
    pub fn decode_input_with_capture<'a, O, C, F>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
        capture: F,
    ) -> Result<(B::Tensor, C), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
        F: FnOnce(&A::ForwardContext) -> Result<C, A::Error>,
    {
        if D::PARTITIONED_SESSION {
            return Err(ReplicatedTextSessionError::Contract(
                "partitioned prediction capture requires a selected bundle publication contract"
                    .into(),
            ));
        }
        let (output, checkpoint, forward_context) =
            self.execute_input_before_publication(input, ExpertPass::Decode, context, observer)?;
        let captured = match capture(&forward_context) {
            Ok(captured) => captured,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Architecture(error),
                    context,
                )
            }
        };
        let (output, checkpoint, forward_context) =
            self.publish_observed_output_transaction(output, checkpoint, forward_context, context)?;
        self.publish(output, checkpoint, forward_context, context)
            .map(|output| (output, captured))
    }

    /// Runs one observed decode step from an architecture-prepared input.
    pub fn decode_input_with_observer<'a, O>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint, forward_context) =
            self.execute_input_with_observer(input, ExpertPass::Decode, context, observer)?;
        let sequence_index = self.output_selection.sequence_index();
        let output = match self
            .mechanisms
            .index_text_output(output, sequence_index, context)
        {
            Ok(output) => output,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Mechanism(error),
                    context,
                )
            }
        };
        self.publish(output, checkpoint, forward_context, context)
    }

    /// Runs one decode step and selects the architecture-declared text output.
    pub fn decode(
        &mut self,
        tokens: &B::Tensor,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        A: ReplicatedTextArchitecture<B, M::State>,
    {
        self.decode_with_observer(tokens, context, &mut crate::NoopObserver)
    }

    /// Runs ordinary target decode and returns its architecture-owned prediction capture.
    pub fn decode_prediction_target(
        &mut self,
        tokens: &B::Tensor,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        (B::Tensor, B::Tensor),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        A: ReplicatedTextArchitecture<B, M::State>,
    {
        let input = A::text_input(tokens, None);
        self.decode_input_prediction_target(input, context)
    }

    /// Runs architecture-prepared target decode and returns its exact additive capture.
    pub fn decode_input_prediction_target<'a>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        (B::Tensor, B::Tensor),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let (output, checkpoint, forward_context) = self.execute_input_before_publication(
            input,
            ExpertPass::Decode,
            context,
            &mut crate::NoopObserver,
        )?;
        let capture = D::prediction_target_capture(&mut self.execution, &forward_context, context)
            .map_err(widen_infallible);
        let local_success = matches!(&capture, Ok(Some(_)));
        let agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::PredictionTargetCapture,
            local_success,
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        let capture = match capture {
            Ok(Some(capture)) if agreed => capture,
            Ok(Some(_)) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank could not prepare the prediction target capture".into(),
                    ),
                    context,
                )
            }
            Ok(None) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "prediction target pass did not retain its declared hidden capture".into(),
                    ),
                    context,
                )
            }
            Err(error) => return self.rollback_failure(checkpoint, error, context),
        };
        let capture_publication =
            D::publish_prediction_target_capture(&mut self.execution, capture, context);
        let capture_publication_agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::PredictionTargetCapturePublication,
            capture_publication.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        let capture = match capture_publication {
            Ok(capture) if capture_publication_agreed => capture,
            Ok(_) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank failed to publish the prediction target capture".into(),
                    ),
                    context,
                )
            }
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        let (output, checkpoint, forward_context) =
            self.publish_observed_output_transaction(output, checkpoint, forward_context, context)?;
        self.publish(output, checkpoint, forward_context, context)
            .map(|output| (output, capture))
    }

    /// Runs one observed decode step and selects the declared text output.
    pub fn decode_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        A: ReplicatedTextArchitecture<B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint, forward_context) =
            self.execute_with_observer(tokens, None, ExpertPass::Decode, context, observer)?;
        let sequence_index = self.output_selection.sequence_index();
        let output = match self
            .mechanisms
            .index_text_output(output, sequence_index, context)
        {
            Ok(output) => output,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Mechanism(error),
                    context,
                )
            }
        };
        self.publish(output, checkpoint, forward_context, context)
    }

    /// Captures all mutable state for a later transactional rollback.
    pub fn checkpoint(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<M::StateCheckpoint, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.ensure_commit_resolved()?;
        self.mechanisms
            .checkpoint_state(&self.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)
    }

    /// Exchanges the canonical target state with one prediction-lane state after all-rank proof.
    ///
    /// The returned state is the previously installed target state. A speculative adapter uses
    /// this operation before and after a target pass so lane-local caches never become a second
    /// owner of target execution. Validation or agreement failure leaves the canonical state
    /// untouched.
    pub fn exchange_prediction_target_state(
        &mut self,
        replacement: &mut M::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        let validation = match self.selected_state.state() {
            Some(selected) => validate_realized_state(replacement, selected),
            None if replacement.optional_layout().is_none() => Ok(()),
            None => Err(ReplicatedTextSessionError::Contract(
                "stateless prediction target received stateful lane state".into(),
            )),
        };
        let phase = crate::DistributedExecutionPhase::PredictionTargetStatePreparation;
        let agreed =
            D::agree_distributed_phase(&mut self.execution, phase, validation.is_ok(), context)
                .map_err(widen_infallible)?;
        match validation {
            Ok(()) if agreed => {
                std::mem::swap(&mut self.state, replacement);
                Ok(())
            }
            Ok(()) => Err(ReplicatedTextSessionError::Contract(
                "another rank rejected its prediction target lane state".into(),
            )),
            Err(error) => Err(error),
        }
    }

    /// Restores local target-state ownership after a failed prediction-lane pass.
    ///
    /// This is a one-shot ownership repair, not a second distributed operation:
    /// the preceding successful exchange already proved both states, and a
    /// failed pass may poison the selected communication authority before the
    /// ordinary agreement-backed exchange can run again. The caller must still
    /// return the original distributed failure; this swap does not clear a
    /// poison, publish state, or make the session reusable.
    pub fn recover_prediction_target_state_after_failure(
        &mut self,
        replacement: &mut M::State,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let selected = self.selected_state.state().ok_or_else(|| {
            ReplicatedTextSessionError::Contract(
                "stateless prediction target cannot recover lane ownership".into(),
            )
        })?;
        validate_realized_state(&self.state, selected)?;
        validate_realized_state(replacement, selected)?;
        std::mem::swap(&mut self.state, replacement);
        Ok(())
    }

    /// Realizes one blank prediction-lane target state without replacing canonical state.
    ///
    /// Every rank prepares and validates its local shard before any caller may retain the lane.
    pub fn prepare_prediction_target_state(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<M::State, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        let provisional = self.selected_state.state().map_or_else(
            || {
                Err(ReplicatedTextSessionError::Contract(
                    "stateless session cannot prepare prediction target state".into(),
                ))
            },
            |selected| {
                self.mechanisms
                    .realize_state(selected, context)
                    .map_err(ReplicatedTextSessionError::Mechanism)
                    .and_then(|state| {
                        validate_realized_state(&state, selected)?;
                        Ok(state)
                    })
            },
        );
        let phase = crate::DistributedExecutionPhase::PredictionTargetStatePreparation;
        let agreed =
            D::agree_distributed_phase(&mut self.execution, phase, provisional.is_ok(), context)
                .map_err(widen_infallible)?;
        match provisional {
            Ok(state) if agreed => Ok(state),
            Ok(_) => Err(ReplicatedTextSessionError::Contract(
                "another rank could not prepare prediction target lane state".into(),
            )),
            Err(error) => Err(error),
        }
    }

    /// Runs one typed prediction-only operation against the neutral target modules and lane state.
    ///
    /// The operation is checkpointed and agreed independently of ordinary output publication. Any
    /// local or remote failure restores the installed lane state before returning.
    pub fn apply_prediction_target_operation<O>(
        &mut self,
        operation: O,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<O::Output, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: PredictionTargetOperation<A, B, M::State>,
    {
        self.ensure_commit_resolved()?;
        let checkpoint = self
            .mechanisms
            .checkpoint_state(&self.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        let execution = D::apply_prediction_target_operation(
            &mut self.execution,
            &mut self.state,
            operation,
            context,
        )
        .map_err(widen_infallible)
        .and_then(|output| {
            output.ok_or_else(|| {
                ReplicatedTextSessionError::Contract(
                    "selected target execution has no typed prediction-extension handoff".into(),
                )
            })
        });
        let phase = crate::DistributedExecutionPhase::PredictionExtensionExecution;
        let agreed =
            D::agree_distributed_phase(&mut self.execution, phase, execution.is_ok(), context)
                .map_err(widen_infallible);
        match (execution, agreed) {
            (Ok(output), Ok(true)) => Ok(output),
            (execution, agreement) => {
                self.mechanisms
                    .restore_state(&mut self.state, checkpoint, context)
                    .map_err(ReplicatedTextSessionError::Mechanism)?;
                match (execution, agreement) {
                    (_, Err(error)) => Err(error),
                    (Err(error), _) => Err(error),
                    (Ok(_), Ok(false)) => Err(ReplicatedTextSessionError::Contract(
                        "another rank failed during prediction-extension execution".into(),
                    )),
                    (Ok(_), Ok(true)) => unreachable!("successful extension returned above"),
                }
            }
        }
    }

    /// Captures mutable state together with the committed composite prompt identity.
    pub fn checkpoint_complete(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        ReplicatedTextSessionCheckpoint<M::StateCheckpoint>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        Ok(ReplicatedTextSessionCheckpoint {
            state: self.checkpoint(context)?,
            prompt_input_identity: self.committed_prompt_input_identity.clone(),
            next_commit_epoch: self.next_commit_epoch,
            last_commit_outcome: self.last_commit_outcome,
        })
    }

    /// Captures a state-only checkpoint only when every partition rank succeeds.
    pub fn checkpoint_distributed(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        DistributedStateCheckpoint<M::StateCheckpoint>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        self.require_cache_control_agreement()?;
        let checkpoint = self.selected_state.state().map(|_| {
            self.mechanisms
                .checkpoint_state(&self.state, context)
                .map_err(ReplicatedTextSessionError::Mechanism)
        });
        let success = checkpoint.as_ref().is_none_or(Result::is_ok);
        let phase = crate::DistributedExecutionPhase::SessionCheckpoint;
        let agreed = self.agree_cache_control_phase(phase, success, context)?;
        let state = match checkpoint {
            Some(Ok(checkpoint)) if agreed => Some(checkpoint),
            None if agreed => None,
            Some(Ok(_)) | None => return self.fence_remote_cache_control_failure(phase),
            Some(Err(error)) => {
                self.control_fence = Some(phase);
                return Err(error);
            }
        };
        Ok(DistributedStateCheckpoint { state })
    }

    /// Captures state and session commit metadata only on all-rank success.
    pub fn checkpoint_complete_distributed(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        DistributedSessionCheckpoint<M::StateCheckpoint>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let state = self.checkpoint_distributed(context)?.state;
        Ok(DistributedSessionCheckpoint {
            state,
            prompt_input_identity: self.committed_prompt_input_identity.clone(),
            next_commit_epoch: self.next_commit_epoch,
            last_commit_outcome: self.last_commit_outcome,
        })
    }

    /// Restores every mutable component from a session checkpoint.
    pub fn rollback(
        &mut self,
        checkpoint: M::StateCheckpoint,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        self.mechanisms
            .restore_state(&mut self.state, checkpoint, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        // A state-only checkpoint cannot prove which multimodal prompt produced its bytes.
        self.committed_prompt_input_identity = None;
        Ok(())
    }

    /// Restores every mutable component and its committed composite prompt identity atomically.
    pub fn rollback_complete(
        &mut self,
        checkpoint: ReplicatedTextSessionCheckpoint<M::StateCheckpoint>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        self.mechanisms
            .restore_state(&mut self.state, checkpoint.state, context)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        self.committed_prompt_input_identity = checkpoint.prompt_input_identity;
        self.next_commit_epoch = checkpoint.next_commit_epoch;
        self.last_commit_outcome = checkpoint.last_commit_outcome;
        self.active_commit_epoch = None;
        Ok(())
    }

    /// Restores a state-only partition checkpoint after all ranks prepare it provisionally.
    pub fn rollback_distributed(
        &mut self,
        checkpoint: DistributedStateCheckpoint<M::StateCheckpoint>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.restore_distributed_state(checkpoint.state, None, context)
    }

    /// Restores partition state and session metadata after all ranks prepare it provisionally.
    pub fn rollback_complete_distributed(
        &mut self,
        checkpoint: DistributedSessionCheckpoint<M::StateCheckpoint>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.restore_distributed_state(
            checkpoint.state,
            Some((
                checkpoint.prompt_input_identity,
                checkpoint.next_commit_epoch,
                checkpoint.last_commit_outcome,
            )),
            context,
        )
    }

    /// Replaces every rank-local state only after all ranks realize a provisional replacement.
    pub fn reset_distributed(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        self.require_cache_control_agreement()?;
        let provisional = self.selected_state.state().map(|selected| {
            self.mechanisms
                .realize_state(selected, context)
                .map_err(ReplicatedTextSessionError::Mechanism)
                .and_then(|state| {
                    validate_realized_state(&state, selected)?;
                    Ok(state)
                })
        });
        let success = provisional.as_ref().is_none_or(Result::is_ok);
        let phase = crate::DistributedExecutionPhase::SessionResetPreparation;
        let agreed = self.agree_cache_control_phase(phase, success, context)?;
        match provisional {
            Some(Ok(state)) if agreed => self.state = state,
            None if agreed => {}
            Some(Ok(_)) | None => return self.fence_remote_cache_control_failure(phase),
            Some(Err(error)) => {
                self.control_fence = Some(phase);
                return Err(error);
            }
        }
        self.committed_prompt_input_identity = None;
        Ok(())
    }

    /// Replaces mutable state with a newly realized selected state.
    pub fn reset(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        if let Some(selected_state) = self.selected_state.state() {
            let state = self
                .mechanisms
                .realize_state(selected_state, context)
                .map_err(ReplicatedTextSessionError::Mechanism)?;
            validate_realized_state(&state, selected_state)?;
            self.state = state;
        } else if self.state.optional_layout().is_some() {
            return Err(ReplicatedTextSessionError::Contract(
                "stateless session owns a stateful mechanism realization".into(),
            ));
        }
        self.committed_prompt_input_identity = None;
        Ok(())
    }

    /// Validates and replaces state from a reusable prompt cache.
    pub fn load_prompt_cache(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.ensure_control_unfenced()?;
        let identity = self.prompt_cache_identity()?.clone();
        validate_prompt_cache_model_identity(expected, &identity)?;
        let selected_state = self.selected_state.state().ok_or_else(|| {
            ReplicatedTextSessionError::Contract(
                "this partition rank owns no prompt-cache state shard".into(),
            )
        })?;
        let (state, manifest) = self
            .mechanisms
            .load_prompt_cache(
                directory,
                expected,
                &identity,
                prefix_token_ids,
                selected_state,
                context,
            )
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        manifest.validate_compatibility(expected, prefix_token_ids)?;
        validate_realized_state(&state, selected_state)?;
        self.state = state;
        self.committed_prompt_input_identity = None;
        self.restore_distributed_commit(manifest.distributed_commit)?;
        Ok(manifest)
    }

    /// Opens a prompt cache only when its content identity matches the admitted prepared input.
    pub fn load_prompt_cache_for_input(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: PreparedInputCacheIdentity,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.validate_prompt_input_descriptor(expected, &input_identity)?;
        let manifest = self.load_prompt_cache(directory, expected, prefix_token_ids, context)?;
        self.committed_prompt_input_identity = Some(input_identity);
        Ok(manifest)
    }

    /// Atomically replaces partition state from rank-local cache shards.
    ///
    /// Stateful ranks load into provisional state while stateless ranks still
    /// participate in both selected-session agreements. No live state or
    /// distributed-commit metadata changes unless every rank validates its
    /// preflight and provisional shard.
    pub fn load_prompt_cache_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.load_prompt_cache_distributed_inner(
            directory,
            expected,
            prefix_token_ids,
            None,
            context,
        )
    }

    /// Atomically loads partition state after all ranks validate one prepared input.
    pub fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: PreparedInputCacheIdentity,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.load_prompt_cache_distributed_inner(
            directory,
            expected,
            prefix_token_ids,
            Some(input_identity),
            context,
        )
    }

    /// Validates identity and persists the current state through native bytes.
    pub fn save_prompt_cache(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.ensure_control_unfenced()?;
        validate_prompt_cache_model_identity(&descriptor, self.prompt_cache_identity()?)?;
        let descriptor = descriptor.with_distributed_commit(self.last_commit_outcome);
        let manifest = self
            .mechanisms
            .save_prompt_cache(
                &mut self.state,
                destination,
                descriptor.clone(),
                prefix_token_ids,
                options,
                context,
            )
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        manifest.validate_compatibility(&descriptor, prefix_token_ids)?;
        Ok(manifest)
    }

    /// Persists state only when the descriptor names the successfully committed prepared input.
    pub fn save_prompt_cache_for_input(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &PreparedInputCacheIdentity,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.validate_prompt_input_descriptor(&descriptor, input_identity)?;
        if self.committed_prompt_input_identity.as_ref() != Some(input_identity) {
            return Err(ReplicatedTextSessionError::Contract(
                "prompt-cache prepared-input identity differs from the committed prompt".into(),
            ));
        }
        self.save_prompt_cache(destination, descriptor, prefix_token_ids, options, context)
    }

    fn load_prompt_cache_distributed_inner(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: Option<PreparedInputCacheIdentity>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        self.require_cache_control_agreement()?;
        let preflight = (|| {
            if let Some(input_identity) = input_identity.as_ref() {
                self.validate_prompt_input_descriptor(expected, input_identity)?;
            }
            match (
                self.selected_state.state(),
                self.prompt_cache_identity.as_ref(),
            ) {
                (Some(selected_state), Some(identity)) => {
                    if !self.selected.prompt_cache() {
                        return Err(ReplicatedTextSessionError::Contract(
                            "prompt-cache persistence was not selected for this session".into(),
                        ));
                    }
                    validate_prompt_cache_model_identity(expected, identity)?;
                    Ok(Some((selected_state.clone(), identity.clone())))
                }
                (None, None) => Ok(None),
                _ => Err(ReplicatedTextSessionError::Contract(
                    "partition cache state and rank-local identity ownership disagree".into(),
                )),
            }
        })();
        let phase = crate::DistributedExecutionPhase::PromptCacheLoadPreflight;
        let agreement = self.agree_cache_control_phase(phase, preflight.is_ok(), context);
        let local = match (preflight, agreement) {
            (Ok(local), Ok(true)) => local,
            (Ok(_), Ok(false)) => return self.fence_remote_cache_control_failure(phase),
            (Ok(_), Err(error)) => return Err(error),
            (Err(error), _) => {
                self.control_fence = Some(phase);
                return Err(error);
            }
        };

        let provisional = local.map(|(selected_state, identity)| {
            self.mechanisms
                .load_prompt_cache(
                    directory,
                    expected,
                    &identity,
                    prefix_token_ids,
                    &selected_state,
                    context,
                )
                .map_err(ReplicatedTextSessionError::Mechanism)
                .and_then(|(state, manifest)| {
                    manifest.validate_compatibility(expected, prefix_token_ids)?;
                    validate_realized_state(&state, &selected_state)?;
                    validate_distributed_commit_restore(manifest.distributed_commit)?;
                    Ok((state, manifest))
                })
        });
        let local_success = provisional.as_ref().is_none_or(Result::is_ok);
        let phase = crate::DistributedExecutionPhase::PromptCacheLoadPreparation;
        let agreement = self.agree_cache_control_phase(phase, local_success, context);
        let provisional = match (provisional, agreement) {
            (Some(Ok(provisional)), Ok(true)) => Some(provisional),
            (None, Ok(true)) => None,
            (Some(Ok(_)) | None, Ok(false)) => {
                return self.fence_remote_cache_control_failure(phase);
            }
            (Some(Ok(_)) | None, Err(error)) => return Err(error),
            (Some(Err(error)), _) => {
                self.control_fence = Some(phase);
                return Err(error);
            }
        };
        let manifest = provisional.map(|(state, manifest)| {
            self.state = state;
            self.committed_prompt_input_identity = input_identity;
            self.active_commit_epoch = None;
            self.last_commit_outcome = manifest.distributed_commit;
            if let Some(outcome) = manifest.distributed_commit {
                self.next_commit_epoch = outcome.epoch().next().unwrap_or_else(|| {
                    unreachable!("provisional commit epoch was validated before agreement")
                });
            }
            manifest
        });
        Ok(manifest)
    }

    fn restore_distributed_state(
        &mut self,
        checkpoint: Option<M::StateCheckpoint>,
        metadata: Option<(
            Option<PreparedInputCacheIdentity>,
            DistributedCommitEpoch,
            Option<DistributedCommitOutcome>,
        )>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        self.require_cache_control_agreement()?;
        let provisional = match (self.selected_state.state(), checkpoint) {
            (Some(selected), Some(checkpoint)) => Some(
                self.mechanisms
                    .realize_state(selected, context)
                    .map_err(ReplicatedTextSessionError::Mechanism)
                    .and_then(|mut state| {
                        self.mechanisms
                            .restore_state(&mut state, checkpoint, context)
                            .map_err(ReplicatedTextSessionError::Mechanism)?;
                        validate_realized_state(&state, selected)?;
                        Ok(state)
                    }),
            ),
            (None, None) => None,
            _ => Some(Err(ReplicatedTextSessionError::Contract(
                "distributed checkpoint presence differs from rank-local state ownership".into(),
            ))),
        };
        let metadata_valid = metadata.as_ref().is_none_or(|(_, next, outcome)| {
            outcome.is_none_or(|outcome| {
                outcome
                    .epoch()
                    .next()
                    .is_some_and(|expected| expected == *next)
            })
        });
        let success = provisional.as_ref().is_none_or(Result::is_ok) && metadata_valid;
        let phase = crate::DistributedExecutionPhase::SessionRollbackPreparation;
        let agreed = self.agree_cache_control_phase(phase, success, context)?;
        let provisional = match provisional {
            Some(Ok(state)) if agreed => Some(state),
            None if agreed => None,
            Some(Ok(_)) | None => return self.fence_remote_cache_control_failure(phase),
            Some(Err(error)) => {
                self.control_fence = Some(phase);
                return Err(error);
            }
        };
        if !metadata_valid {
            self.control_fence = Some(phase);
            return Err(ReplicatedTextSessionError::Contract(
                "distributed checkpoint commit metadata is inconsistent".into(),
            ));
        }
        if let Some(state) = provisional {
            self.state = state;
        }
        match metadata {
            Some((identity, next, outcome)) => {
                self.committed_prompt_input_identity = identity;
                self.next_commit_epoch = next;
                self.last_commit_outcome = outcome;
                self.active_commit_epoch = None;
            }
            None => self.committed_prompt_input_identity = None,
        }
        Ok(())
    }

    /// Returns the prepared-input identity associated with the currently committed prompt state.
    pub const fn committed_prompt_input_identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.committed_prompt_input_identity.as_ref()
    }

    /// Returns one coherent execution and state residency report.
    pub fn report(
        &self,
    ) -> Result<
        ReplicatedTextSessionReport<M::ExecutionReport, M::StateReport>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let execution_report = self
            .mechanisms
            .execution_report(
                self.selected.residency(),
                D::bounded_policy(&self.execution),
            )
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        let state_report = self
            .mechanisms
            .state_report(&self.state)
            .map_err(ReplicatedTextSessionError::Mechanism)?;
        Ok(ReplicatedTextSessionReport {
            execution: D::execution_residency(&self.execution, &self.selected),
            execution_report,
            state_report,
            distributed_commit: self.last_commit_outcome,
        })
    }

    fn execute_with_observer<O>(
        &mut self,
        tokens: &B::Tensor,
        mask: Option<&B::Tensor>,
        pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        A: ReplicatedTextArchitecture<B, M::State>,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let input = A::text_input(tokens, mask);
        self.execute_input_with_observer(input, pass, context, observer)
    }

    fn execute_input_with_observer<'a, O>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint, forward_context) =
            self.execute_input_before_publication(input, pass, context, observer)?;
        self.publish_observed_output_transaction(output, checkpoint, forward_context, context)
    }

    fn execute_input_before_publication<'a, O>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.begin_commit_epoch()?;
        let checkpoint = self.mechanisms.checkpoint_state(&self.state, context);
        let checkpoint_agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::StateCheckpoint,
            checkpoint.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => return self.abort_without_rollback(widen_infallible(error)),
        };
        let checkpoint = match checkpoint {
            Ok(checkpoint) if checkpoint_agreed => checkpoint,
            Ok(_) => {
                return self.abort_without_rollback(ReplicatedTextSessionError::Contract(
                    "another rank failed to capture its distributed state checkpoint".into(),
                ));
            }
            Err(error) => {
                return self.abort_without_rollback(ReplicatedTextSessionError::Mechanism(error));
            }
        };
        let execution = self
            .driver
            .forward_with_observer(
                &mut self.execution,
                input,
                &mut self.state,
                pass,
                context,
                observer,
            )
            .map_err(widen_infallible);
        let execution_agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::Execution,
            execution.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                let error = match execution {
                    Err(local) => local,
                    Ok(_) => widen_infallible(error),
                };
                return self.rollback_failure(checkpoint, error, context);
            }
        };
        let (output, forward_context) = match execution {
            Ok(output) if execution_agreed => output,
            Ok(_) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank failed during distributed execution".into(),
                    ),
                    context,
                )
            }
            Err(error) => return self.rollback_failure(checkpoint, error, context),
        };
        let observation = D::observe_output(&mut self.execution, &output, observer, context);
        let observation_agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::OutputObservation,
            observation.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        let output = match observation {
            Ok(output) if observation_agreed => output,
            Ok(_) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank failed during distributed output observation".into(),
                    ),
                    context,
                )
            }
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        Ok((output, checkpoint, forward_context))
    }

    fn publish_observed_output_transaction(
        &mut self,
        output: B::Tensor,
        checkpoint: M::StateCheckpoint,
        forward_context: A::ForwardContext,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        (B::Tensor, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let publication = D::publish_observed_output(&mut self.execution, output, context);
        let publication_agreement = D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::OutputPublication,
            publication.is_ok(),
            context,
        );
        let output = match (publication, publication_agreement) {
            (Err(error), _) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
            (Ok(_), Err(error)) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
            (Ok(output), Ok(true)) => output,
            (Ok(_), Ok(false)) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank failed during distributed output publication".into(),
                    ),
                    context,
                )
            }
        };
        Ok((output, checkpoint, forward_context))
    }

    fn publish(
        &mut self,
        output: B::Tensor,
        checkpoint: M::StateCheckpoint,
        _forward_context: A::ForwardContext,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let completion = self
            .selected
            .exact_completion()
            .then(|| self.mechanisms.complete(&output, &self.state, context))
            .transpose();
        let completion_agreed = match D::agree_distributed_phase(
            &mut self.execution,
            crate::DistributedExecutionPhase::MechanismCompletion,
            completion.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context)
            }
        };
        match completion {
            Ok(_) if completion_agreed => {}
            Ok(_) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank failed during distributed mechanism completion".into(),
                    ),
                    context,
                )
            }
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Mechanism(error),
                    context,
                )
            }
        }
        let epoch = self.active_commit_epoch.ok_or_else(|| {
            ReplicatedTextSessionError::Contract(
                "distributed transaction lost its active commit epoch".into(),
            )
        })?;
        match D::commit_after_completion(&mut self.execution, epoch, context) {
            DistributedCommitOutcome::Committed(committed) if committed == epoch => {
                self.last_commit_outcome = Some(DistributedCommitOutcome::Committed(epoch));
                self.active_commit_epoch = None;
            }
            DistributedCommitOutcome::Aborted(aborted) if aborted == epoch => {
                self.last_commit_outcome = Some(DistributedCommitOutcome::Aborted(epoch));
                self.active_commit_epoch = None;
                self.mechanisms
                    .restore_state(&mut self.state, checkpoint, context)
                    .map_err(ReplicatedTextSessionError::Mechanism)?;
                return Err(ReplicatedTextSessionError::CommitAborted { epoch });
            }
            DistributedCommitOutcome::Indeterminate {
                epoch: uncertain,
                phase,
            } if uncertain == epoch => {
                self.last_commit_outcome =
                    Some(DistributedCommitOutcome::Indeterminate { epoch, phase });
                self.active_commit_epoch = None;
                return Err(ReplicatedTextSessionError::CommitIndeterminate { epoch, phase });
            }
            outcome => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(format!(
                        "distributed commit returned epoch {} for active epoch {}",
                        outcome.epoch().value(),
                        epoch.value()
                    )),
                    context,
                );
            }
        }
        Ok(output)
    }

    fn rollback_failure<T>(
        &mut self,
        checkpoint: M::StateCheckpoint,
        error: ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let restored = self
            .mechanisms
            .restore_state(&mut self.state, checkpoint, context)
            .map_err(ReplicatedTextSessionError::Mechanism);
        if let Some(epoch) = self.active_commit_epoch.take() {
            self.last_commit_outcome = Some(DistributedCommitOutcome::Aborted(epoch));
        }
        restored?;
        Err(error)
    }

    fn begin_commit_epoch(
        &mut self,
    ) -> Result<
        DistributedCommitEpoch,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        if self.active_commit_epoch.is_some() {
            return Err(ReplicatedTextSessionError::Contract(
                "distributed transaction already has an active commit epoch".into(),
            ));
        }
        let epoch = self.next_commit_epoch;
        self.next_commit_epoch = epoch.next().ok_or_else(|| {
            ReplicatedTextSessionError::Contract("distributed commit epoch overflow".into())
        })?;
        self.active_commit_epoch = Some(epoch);
        Ok(epoch)
    }

    fn ensure_commit_resolved(
        &self,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_control_unfenced()?;
        if let Some(DistributedCommitOutcome::Indeterminate { epoch, phase }) =
            self.last_commit_outcome
        {
            return Err(ReplicatedTextSessionError::CommitIndeterminate { epoch, phase });
        }
        Ok(())
    }

    fn ensure_control_unfenced(
        &self,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if let Some(phase) = self.control_fence {
            return Err(ReplicatedTextSessionError::Contract(format!(
                "distributed session is fenced after failed cache control at {phase:?}"
            )));
        }
        Ok(())
    }

    fn agree_cache_control_phase(
        &mut self,
        phase: crate::DistributedExecutionPhase,
        local_success: bool,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        D::agree_distributed_phase(&mut self.execution, phase, local_success, context)
            .map_err(widen_infallible)
            .inspect_err(|_| self.control_fence = Some(phase))
    }

    fn require_cache_control_agreement(
        &self,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if D::PARTITIONED_SESSION && !D::DISTRIBUTED_PHASE_AGREEMENT {
            return Err(ReplicatedTextSessionError::Contract(
                "partitioned cache control requires the selected bounded failure agreement".into(),
            ));
        }
        Ok(())
    }

    fn fence_remote_cache_control_failure<T>(
        &mut self,
        phase: crate::DistributedExecutionPhase,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.control_fence = Some(phase);
        Err(ReplicatedTextSessionError::Contract(format!(
            "another rank failed distributed cache control at {phase:?}"
        )))
    }

    fn abort_without_rollback<T>(
        &mut self,
        error: ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if let Some(epoch) = self.active_commit_epoch.take() {
            self.last_commit_outcome = Some(DistributedCommitOutcome::Aborted(epoch));
        }
        Err(error)
    }

    fn restore_distributed_commit(
        &mut self,
        outcome: Option<DistributedCommitOutcome>,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.active_commit_epoch = None;
        self.last_commit_outcome = outcome;
        if let Some(outcome) = outcome {
            self.next_commit_epoch = outcome.epoch().next().ok_or_else(|| {
                ReplicatedTextSessionError::Contract("distributed commit epoch overflow".into())
            })?;
        }
        Ok(())
    }

    fn validate_prompt_input_descriptor(
        &self,
        descriptor: &PromptCacheDescriptor,
        input_identity: &PreparedInputCacheIdentity,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if descriptor.prefix_content_fingerprint() != input_identity.prefix_content_fingerprint() {
            return Err(ReplicatedTextSessionError::Contract(
                "prompt-cache content identity differs from the prepared input".into(),
            ));
        }
        Ok(())
    }

    fn prompt_cache_identity(
        &self,
    ) -> Result<
        &PromptCacheModelIdentity,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        if !self.selected.prompt_cache() {
            return Err(ReplicatedTextSessionError::Contract(
                "prompt-cache persistence was not selected for this session".into(),
            ));
        }
        self.prompt_cache_identity.as_ref().ok_or_else(|| {
            ReplicatedTextSessionError::Contract(
                "this partition rank owns no prompt-cache model identity".into(),
            )
        })
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: TransactionalPromptCacheMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Atomically publishes rank-local cache shards after all ranks prepare them.
    pub fn save_prompt_cache_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.save_prompt_cache_distributed_inner(
            destination,
            descriptor,
            prefix_token_ids,
            options,
            None,
            context,
        )
    }

    /// Atomically publishes shards only for the globally committed prepared input.
    pub fn save_prompt_cache_for_input_distributed(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: &PreparedInputCacheIdentity,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.save_prompt_cache_distributed_inner(
            destination,
            descriptor,
            prefix_token_ids,
            options,
            Some(input_identity),
            context,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn save_prompt_cache_distributed_inner(
        &mut self,
        destination: &Path,
        descriptor: PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        options: &PromptCacheOptions,
        input_identity: Option<&PreparedInputCacheIdentity>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        self.require_cache_control_agreement()?;
        let preflight = (|| {
            if let Some(input_identity) = input_identity {
                self.validate_prompt_input_descriptor(&descriptor, input_identity)?;
                if self.committed_prompt_input_identity.as_ref() != Some(input_identity) {
                    return Err(ReplicatedTextSessionError::Contract(
                        "prompt-cache prepared-input identity differs from the committed prompt"
                            .into(),
                    ));
                }
            }
            match (
                self.selected_state.state(),
                self.prompt_cache_identity.as_ref(),
            ) {
                (Some(_), Some(identity)) => {
                    if !self.selected.prompt_cache() {
                        return Err(ReplicatedTextSessionError::Contract(
                            "prompt-cache persistence was not selected for this session".into(),
                        ));
                    }
                    validate_prompt_cache_model_identity(&descriptor, identity)?;
                    Ok(Some(
                        descriptor
                            .clone()
                            .with_distributed_commit(self.last_commit_outcome),
                    ))
                }
                (None, None) => Ok(None),
                _ => Err(ReplicatedTextSessionError::Contract(
                    "partition cache state and rank-local identity ownership disagree".into(),
                )),
            }
        })();
        let phase = crate::DistributedExecutionPhase::PromptCacheSavePreflight;
        let agreement = self.agree_cache_control_phase(phase, preflight.is_ok(), context);
        let local_descriptor = match (preflight, agreement) {
            (Ok(local), Ok(true)) => local,
            (Ok(_), Ok(false)) => return self.fence_remote_cache_control_failure(phase),
            (Ok(_), Err(error)) => return Err(error),
            (Err(error), _) => {
                self.control_fence = Some(phase);
                return Err(error);
            }
        };

        let mut transaction = local_descriptor.map(|descriptor| {
            self.mechanisms
                .prepare_prompt_cache_save(
                    &mut self.state,
                    destination,
                    descriptor.clone(),
                    prefix_token_ids,
                    options,
                    context,
                )
                .map_err(ReplicatedTextSessionError::Mechanism)
                .and_then(|transaction| {
                    M::prepared_prompt_cache_manifest(&transaction)
                        .validate_compatibility(&descriptor, prefix_token_ids)?;
                    Ok(transaction)
                })
        });
        let local_success = transaction.as_ref().is_none_or(Result::is_ok);
        let phase = crate::DistributedExecutionPhase::PromptCacheSavePreparation;
        let agreement = self.agree_cache_control_phase(phase, local_success, context);
        let mut transaction = match (transaction.take(), agreement) {
            (Some(Ok(transaction)), Ok(true)) => Some(transaction),
            (None, Ok(true)) => None,
            (Some(Ok(transaction)), Ok(false)) => {
                self.mechanisms.rollback_prompt_cache_save(transaction);
                return self.fence_remote_cache_control_failure(phase);
            }
            (None, Ok(false)) => return self.fence_remote_cache_control_failure(phase),
            (Some(Ok(transaction)), Err(error)) => {
                self.mechanisms.rollback_prompt_cache_save(transaction);
                return Err(error);
            }
            (None, Err(error)) => return Err(error),
            (Some(Err(error)), _) => {
                self.control_fence = Some(phase);
                return Err(error);
            }
        };

        let publication = transaction
            .as_mut()
            .map(|transaction| self.mechanisms.publish_prompt_cache_save(transaction));
        let local_success = publication.as_ref().is_none_or(Result::is_ok);
        let phase = crate::DistributedExecutionPhase::PromptCacheSavePublication;
        let agreement = self.agree_cache_control_phase(phase, local_success, context);
        let agreed = match (publication, agreement) {
            (Some(Err(error)), _) => {
                self.mechanisms.rollback_prompt_cache_save(
                    transaction
                        .take()
                        .expect("failed publication retains its transaction"),
                );
                self.control_fence = Some(phase);
                return Err(ReplicatedTextSessionError::Mechanism(error));
            }
            (Some(Ok(())) | None, Err(error)) => {
                if let Some(transaction) = transaction.take() {
                    self.mechanisms.rollback_prompt_cache_save(transaction);
                }
                return Err(error);
            }
            (Some(Ok(())) | None, Ok(agreed)) => agreed,
        };
        if !agreed {
            if let Some(transaction) = transaction {
                self.mechanisms.rollback_prompt_cache_save(transaction);
            }
            return self.fence_remote_cache_control_failure(phase);
        }
        Ok(transaction.map(|transaction| {
            let manifest = M::prepared_prompt_cache_manifest(&transaction).clone();
            self.mechanisms.commit_prompt_cache_save(transaction);
            manifest
        }))
    }
}

fn validate_distributed_commit_restore<A, P, M>(
    outcome: Option<DistributedCommitOutcome>,
) -> Result<(), ReplicatedTextSessionError<A, P, M>>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    if outcome.is_some_and(|outcome| outcome.epoch().next().is_none()) {
        return Err(ReplicatedTextSessionError::Contract(
            "distributed commit epoch overflow".into(),
        ));
    }
    Ok(())
}

fn validate_architecture_geometry<A, B, S>(
    architecture: &A,
    selected: &SelectedReplicatedTextRealization,
) -> Result<(), String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let requirements = selected.requirements();
    let graph = architecture
        .execution_graph()
        .map_err(|error| error.to_string())?;
    if &graph != requirements.execution_graph() {
        return Err("architecture execution graph differs from selection".into());
    }
    if requirements.execution_units().group_count() != graph.groups().len() {
        return Err("selected execution-unit groups differ from architecture graph".into());
    }
    for group in 0..graph.groups().len() {
        let actual = architecture
            .group_unit_count(group)
            .map_err(|error| error.to_string())?;
        let expected = requirements
            .execution_units()
            .group_range(group)
            .expect("validated requirement exposes every graph group")
            .len();
        if actual != expected
            || architecture.group_transport(group) != requirements.group_transports()[group]
        {
            return Err(format!(
                "architecture execution group {group} differs from selection"
            ));
        }
    }
    let layout = architecture
        .state_layout()
        .map_err(|error| error.to_string())?;
    if &layout != selected.state().layout() {
        return Err("architecture state layout differs from selection".into());
    }
    Ok(())
}

fn validate_architecture_parameters<A, B, S>(
    architecture: &A,
    selected: &SelectedReplicatedTextRealization,
    selected_formats: bool,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<BTreeMap<String, Vec<ReplicatedTextOutputCompanion>>, String>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let requirements = selected.requirements();
    let description = architecture
        .parameter_description(context)
        .map_err(|error| error.to_string())?;
    let mut actual = BTreeMap::<String, (Vec<usize>, ParameterGroupOwner)>::new();
    let mut companions = Vec::new();
    for group in description.groups() {
        for member in group.group().members() {
            match (member.linear_companion(), member.linear_companion_of()) {
                (None, None) => {
                    if actual
                        .insert(
                            member.target().to_owned(),
                            (member.global_shape().to_vec(), group.owner().clone()),
                        )
                        .is_some()
                    {
                        return Err(format!(
                            "constructed architecture repeats primary parameter {:?}",
                            member.target()
                        ));
                    }
                }
                (Some(role), Some(primary)) => companions.push((
                    member.target().to_owned(),
                    member.global_shape().to_vec(),
                    group.owner().clone(),
                    role,
                    primary.to_owned(),
                )),
                _ => {
                    return Err(format!(
                        "constructed parameter {:?} has incomplete linear companion metadata",
                        member.target()
                    ));
                }
            }
        }
    }
    let expected = requirements
        .parameters()
        .iter()
        .filter(|parameter| {
            !matches!(
                parameter.presence(),
                ReplicatedTextParameterPresence::OptionalAbsent
                    | ReplicatedTextParameterPresence::Tied { .. }
            )
        })
        .map(|parameter| parameter.name())
        .collect::<BTreeSet<_>>();
    for (name, shape, owner, _, _) in &companions {
        if expected.contains(name.as_str())
            && actual
                .insert(name.clone(), (shape.clone(), owner.clone()))
                .is_some()
        {
            return Err(format!(
                "constructed architecture repeats selected parameter {name:?}"
            ));
        }
    }
    let actual_names = actual.keys().map(String::as_str).collect::<BTreeSet<_>>();
    if expected != actual_names {
        return Err(format!(
            "selected parameter catalog differs from constructed architecture: missing {:?}, unexpected {:?}",
            expected.difference(&actual_names).collect::<Vec<_>>(),
            actual_names.difference(&expected).collect::<Vec<_>>()
        ));
    }
    for parameter in requirements.parameters().iter().filter(|parameter| {
        !matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::OptionalAbsent
                | ReplicatedTextParameterPresence::Tied { .. }
        )
    }) {
        let (shape, owner) = actual
            .get(parameter.name())
            .expect("equal parameter-name sets contain every requirement");
        let owner_matches = match (owner, parameter.owner()) {
            (
                ParameterGroupOwner::StaticRole(actual),
                ReplicatedTextParameterOwner::StaticRole(expected),
            ) => actual == expected,
            (
                ParameterGroupOwner::StaticAnyOf(actual),
                ReplicatedTextParameterOwner::StaticRole(expected),
            ) => actual.iter().any(|role| role == expected),
            (
                ParameterGroupOwner::ExecutionUnit {
                    group, global_unit, ..
                },
                ReplicatedTextParameterOwner::ExecutionUnit {
                    group: expected_group,
                    unit: expected_unit,
                },
            ) => group.as_str() == expected_group && global_unit == expected_unit,
            _ => false,
        };
        let mut expected_shape = parameter.logical_shape().to_vec();
        let realization = selected_formats
            .then(|| {
                selected
                    .parameters()
                    .iter()
                    .find(|realization| realization.name() == parameter.name())
            })
            .flatten();
        let executable = realization.map_or(parameter.native_executable(), |realization| {
            realization.executable()
        });
        if (!selected_formats
            || realization.is_some_and(|realization| {
                realization.lowering() == crate::WeightLoweringKind::Direct
            }))
            && executable == eredu_checkpoint::LinearFormat::MxFp4
            && matches!(
                parameter.source_encoding(),
                Some(eredu_checkpoint::SourceTensorEncoding::Safetensors(
                    eredu_checkpoint::StoredDtype::U8
                ))
            )
        {
            expected_shape = parameter
                .physical_shape()
                .expect("direct native realization has admitted physical geometry")
                .to_vec();
        } else if let Some(last) = expected_shape.last_mut() {
            match executable {
                eredu_checkpoint::LinearFormat::Affine(config) => {
                    *last = last.saturating_mul(config.bits as usize) / 32;
                }
                eredu_checkpoint::LinearFormat::MxFp4 => {
                    *last = last.saturating_mul(4) / 32;
                }
                eredu_checkpoint::LinearFormat::GgufIQuant { ggml_type, .. } => {
                    if let Ok((block, bytes)) = ggml_type.block_and_bytes() {
                        if let (Ok(block), Ok(bytes)) =
                            (usize::try_from(block), usize::try_from(bytes))
                        {
                            *last = last.saturating_mul(bytes) / block;
                        }
                    }
                }
                eredu_checkpoint::LinearFormat::Dense
                | eredu_checkpoint::LinearFormat::E4M3BlockFp8(_) => {}
            }
        }
        if shape != &expected_shape || !owner_matches {
            return Err(format!(
                "selected parameter {:?} expects shape {expected_shape:?} and owner {:?}, constructed shape {shape:?} and owner {owner:?}",
                parameter.name(),
                parameter.owner()
            ));
        }
    }

    let mut output_companions = BTreeMap::<String, Vec<ReplicatedTextOutputCompanion>>::new();
    for (name, shape, owner, role, primary) in companions {
        if name == primary || !expected.contains(primary.as_str()) {
            return Err(format!(
                "constructed companion {name:?} names unselected primary {primary:?}"
            ));
        }
        if selected_formats {
            let recipe = selected.requirements().derived_recipes().get(&name);
            let output = selected.requirements().derived_recipe_outputs().get(&name);
            let companion = match (recipe, output) {
                (Some(recipe), Some(output)) => {
                    ReplicatedTextOutputCompanion::new(name, role, shape, owner).map(|companion| {
                        companion.with_derived_recipe(recipe.clone(), output.clone())
                    })
                }
                (None, None) => ReplicatedTextOutputCompanion::new(name, role, shape, owner),
                _ => {
                    return Err(format!(
                        "constructed companion {name:?} has incomplete derived metadata"
                    ))
                }
            }
            .map_err(|error| error.to_string())?;
            let companion = match requirements
                .parameters()
                .iter()
                .find(|parameter| parameter.name() == companion.name())
            {
                Some(parameter)
                    if matches!(
                        parameter.source_encoding(),
                        Some(eredu_checkpoint::SourceTensorEncoding::Gguf { .. })
                    ) =>
                {
                    let [source] = parameter.physical_sources() else {
                        return Err(format!(
                            "translated catalog companion {:?} has ambiguous provenance",
                            companion.name()
                        ));
                    };
                    companion.with_catalog_source(source.clone())
                }
                _ => companion,
            };
            output_companions
                .entry(primary)
                .or_default()
                .push(companion);
        }
    }
    Ok(output_companions)
}

fn validate_selected_state(selected: &SelectedReplicatedTextRealization) -> Result<(), String> {
    use eredu_core::cache::StateResidencyClass;

    if !selected.topology().is_replicated() {
        return Err("selected replicated-text topology is not replicated".into());
    }
    if selected.state().layout() != selected.requirements().state_layout()
        || selected.state().access() != selected.requirements().state_access()
    {
        return Err("selected state contract differs from architecture requirements".into());
    }
    let mut cursor = 0;
    for layer in 0..selected.state().layout().len() {
        for expected in selected
            .state()
            .layout()
            .components(layer)
            .expect("validated state layout exposes every layer")
        {
            let component = selected.state().components().get(cursor).ok_or_else(|| {
                format!("selected state omits component {cursor} at layer {layer}")
            })?;
            if component.layer() != layer || component.component() != expected {
                return Err(format!(
                    "selected state component {cursor} differs from layer {layer} requirements"
                ));
            }
            let expected_placement = match selected.state().policy() {
                crate::CacheResidencyPolicy::Device => crate::StateComponentPlacement::Device,
                crate::CacheResidencyPolicy::Paged(_) => match expected.residency() {
                    StateResidencyClass::SealablePaged => crate::StateComponentPlacement::Paged,
                    StateResidencyClass::AlwaysDeviceMutable
                    | StateResidencyClass::LayerScopedOffloadable => {
                        crate::StateComponentPlacement::Device
                    }
                },
            };
            if component.placement() != expected_placement {
                return Err(format!(
                    "selected state component {cursor} has {:?} placement, expected {expected_placement:?}",
                    component.placement()
                ));
            }
            cursor += 1;
        }
    }
    if cursor != selected.state().components().len() {
        return Err("selected state contains components beyond its architecture layout".into());
    }
    if !selected.state().checkpoint() || !selected.state().rollback() || !selected.state().reset() {
        return Err("selected state omits a required transactional lifecycle facility".into());
    }
    if selected.state().prompt_cache() != selected.prompt_cache()
        || selected.state().observation_retention()
            != (selected.session().output_observation()
                || selected.session().activation_inspection())
    {
        return Err("selected state lifecycle differs from selected session facilities".into());
    }
    if selected.grouped_operations() != selected.requirements().grouped_operations() {
        return Err("selected grouped operations differ from architecture requirements".into());
    }
    Ok(())
}

fn validate_realized_state<A, P, M, B, S>(
    state: &S,
    selected: &SelectedStateRealization,
) -> Result<(), ReplicatedTextSessionError<A, P, M>>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    if state.layout() != selected.layout() {
        return Err(ReplicatedTextSessionError::Contract(
            "realized state layout differs from selection".into(),
        ));
    }
    Ok(())
}

fn map_layerwise_error<A, P>(
    error: LayerwiseRuntimeError<A, P>,
) -> ReplicatedTextSessionError<A, P, std::convert::Infallible>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
{
    match error {
        LayerwiseRuntimeError::Architecture(error) => {
            ReplicatedTextSessionError::Architecture(error)
        }
        LayerwiseRuntimeError::State(error) => ReplicatedTextSessionError::State(error),
        LayerwiseRuntimeError::Layout(error) => {
            ReplicatedTextSessionError::Contract(error.to_string())
        }
        LayerwiseRuntimeError::Policy(error) => ReplicatedTextSessionError::Policy(error),
        LayerwiseRuntimeError::Submission(error) => ReplicatedTextSessionError::Contract(error),
    }
}

fn widen_infallible<A, P, M>(
    error: ReplicatedTextSessionError<A, P, std::convert::Infallible>,
) -> ReplicatedTextSessionError<A, P, M>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    match error {
        ReplicatedTextSessionError::Contract(error) => ReplicatedTextSessionError::Contract(error),
        ReplicatedTextSessionError::Architecture(error) => {
            ReplicatedTextSessionError::Architecture(error)
        }
        ReplicatedTextSessionError::Policy(error) => ReplicatedTextSessionError::Policy(error),
        ReplicatedTextSessionError::Mechanism(error) => match error {},
        ReplicatedTextSessionError::State(error) => ReplicatedTextSessionError::State(error),
        ReplicatedTextSessionError::PromptCache(error) => {
            ReplicatedTextSessionError::PromptCache(error)
        }
        ReplicatedTextSessionError::CommitAborted { epoch } => {
            ReplicatedTextSessionError::CommitAborted { epoch }
        }
        ReplicatedTextSessionError::CommitIndeterminate { epoch, phase } => {
            ReplicatedTextSessionError::CommitIndeterminate { epoch, phase }
        }
    }
}
