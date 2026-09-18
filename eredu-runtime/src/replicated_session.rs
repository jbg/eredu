//! Backend-neutral replicated-text execution and session ownership.

#![allow(clippy::type_complexity)]

mod contract_metadata;
pub use contract_metadata::PreparedTextContractError;
use contract_metadata::{ContractMetadata, DebugRows, ValidatedParameterCatalog};
mod materialization_source;
pub use materialization_source::PreparedContractMaterialization;

use std::{
    collections::{BTreeMap, BTreeSet},
    marker::PhantomData,
    path::Path,
};

use crate::working_memory::InferenceStateRetention;
use eredu_core::cache::{
    PromptCacheDescriptor, PromptCacheError, PromptCacheManifest, PromptCacheModelIdentity,
    PromptCacheOptions, PromptCacheTopology, validate_prompt_cache_model_identity,
};
use eredu_core::{DistributedCommitEpoch, DistributedCommitOutcome, DistributedCommitPhase};
use eredu_nn::{NeuralBackend, Tensor};

mod control;
mod execution_inspection;
mod parallel_context;
mod parallel_control;
pub use parallel_control::{ParallelControlEvent, ParallelControlIdentity, ParallelControlCursor, ParallelControlClaim};
pub use parallel_context::{PreparedParallelContextCause, PreparedParallelContextFailure};
pub use execution_inspection::RuntimeInspectionBoundary;
mod media_semantic_binding;
pub(crate) mod observation_paths;
mod resident_reset;
pub use media_semantic_binding::{MediaSemanticBindingError, OriginalMediaBindingError};
pub use observation_paths::PreparedSessionObservationError;
mod media_prefill;
mod prediction;
pub use prediction::PublishedPredictionPrefill;
mod prediction_loan;
pub use prediction_loan::PredictionStateLoanError;
mod prefill;
pub use control::{
    ControlBranchSource, ControlBranchPlacement, ControlExchangeResult,
    PreparedControlBindingError, PreparedControlExchangeError, ReplicatedTextControlOrigin,
    ReplicatedTextControlState, ReplicatedTextSnapshotMechanisms,
};
pub use prefill::{
    MediaPrefillSpan, OrdinaryPrefillSpan, PrefillScoreLayout, PrefillSourceOutcome, PrefillSourceProgress,
    PrefillSpanOperation, PreparedPrefillSource, SessionPrefill, SettledPrefillCompletion,
};

use crate::{
    ActivationObserver, ArchitecturePartition, CommunicationManifest, ExecutionResidency,
    ExpertPass, LayerWeightResidency, LayeredArchitecture, LayerwisePolicy, LayerwiseRuntime,
    LayerwiseRuntimeError, ParameterGroupOwner, PartitionState, PreparedInputCacheIdentity,
    ReplicatedTextArchitecture, ReplicatedTextMaterializationTask, ReplicatedTextOutputCompanion,
    ReplicatedTextOutputSelection, ReplicatedTextParameterPresence, RoutedExpertProvider,
    RoutedLayeredArchitecture, RuntimeState, SelectedReplicatedTextRealization,
    SelectedStateRealization, SharedPreparedInputCacheIdentity, StateError, SubmissionBackend,
    WeightLoweringKind, observe_model_logits, partitioned_replicated_text_materialization_tasks,
    plan_local_replicated_text_materialization_tasks, replicated_text_materialization_tasks,
};

use crate::parameter_operations::LayeredParameterOwner;

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
    /// Lends a prepared backend context only around the existing selected
    /// numerical forward. The outer session keeps admission, checkpoint,
    /// agreement, publication and failure recovery. Ordinary mechanisms call
    /// the same closure directly. A context owner must outlive enclosing native
    /// completion; restoring a context does not claim completion or refund.
    fn with_execution_parallel<T,E,F>(
        &self, context:&<B::Tensor as Tensor>::Context, run:F,
    )->Result<Result<T,E>,Self::Error>
    where F:FnOnce(Option<(&mut B::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)->Result<T,E>,
    {
        let _=context;
        Ok(run(None))
    }

    /// Reborrows the enclosing numerical invocation for its quoted output
    /// publication after forward. This does not rebind or reopen the invocation.
    fn with_execution_parallel_publication<T,E,F>(&self,context:&<B::Tensor as Tensor>::Context,run:F)
        ->Result<Result<T,E>,Self::Error>
    where F:FnOnce(Option<(&B::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)->Result<T,E> {
        let _=context;Ok(run(None))
    }

    /// Lends a control-only request context around the existing numerical
    /// forward. This is independent of the architecture's optional neural TP
    /// context and retains no authority to alter tensor-parallel selection.
    fn with_execution_parallel_control_context<T, E, F>(
        &self, context: &<B::Tensor as Tensor>::Context, run: F,
    ) -> Result<Result<T, E>, Self::Error>
    where F: FnOnce(Option<(&mut Option<Box<B::ParallelContext>>,
        &eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        let _ = context;
        Ok(run(None))
    }

    /// Lends the exact request-owned control context for this actual lifecycle
    /// event. This includes phases before the numerical forward. Ordinary
    /// mechanisms pass None; prepared contexts must never fall back to ordinary
    /// communication. Native errors retain their source through BackendFailure.
    fn with_execution_parallel_control<T, E, F>(
        &self, event: ParallelControlEvent, context: &<B::Tensor as Tensor>::Context, run: F,
    ) -> Result<Result<T, E>, eredu_core::BackendFailure>
    where F: FnOnce(Option<(&B::ParallelContext, &eredu_nn::workspace::HostMetadataFunding)>) -> Result<T, E>,
    {
        let _ = (event, context);
        Ok(run(None))
    }

    /// Reads the retained decoder frontier in the architecture-selected state
    /// realization without native allocation or submission. `None` is reserved
    /// for an explicit stateless rank; unknown stateful positions are errors.
    fn prefill_state_frontier(&self, state: &Self::State) -> Result<Option<u64>, Self::Error>;

    /// Allocation-free original-request inspection of the same actual selected
    /// frontier. This companion must not format, box errors, initialize state or
    /// settle native work. The default preserves the missing mechanism as a
    /// temporary admission gap, not an architectural restriction.
    fn original_prefill_state_frontier(
        &self,
        _state: &Self::State,
    ) -> Result<Option<u64>, crate::working_memory::WorkingMemoryError> {
        Err(crate::working_memory::WorkingMemoryError::UnknownBound)
    }

    /// Owns the admitted reservation through native preparation, completion,
    /// failure recovery and teardown. Dropping an unresolved guard must retain
    /// the reservation until independent native release evidence exists. Failure
    /// Drop must be nonblocking; it must not use successful-completion waiting.
    type PrefillReservationGuard;

    /// Opens native reservation retention with one bounded, nonblocking attempt
    /// before input preparation or an admitted transaction. Err accepted no work
    /// under this attempt; older work must keep its independent owners. A failed
    /// attempt cannot retain or authorize a new agreement vote.
    fn begin_prefill_reservation(
        &mut self,
        reservation: crate::working_memory::InferenceRequest,
    ) -> Result<Self::PrefillReservationGuard, Self::Error>;

    /// Names the exact shared-prefill role before its existing scope begins.
    /// Ordinary implementations delegate unchanged. An installed original bank
    /// must validate the request/source and consume its corresponding fixed slot
    /// before any constructor; it must never fall back to ordinary allocation.
    fn begin_prefill_control(
        &mut self,
        reservation: crate::working_memory::InferenceRequest,
        _role: crate::prefill::PrefillControlRole,
    ) -> Result<Self::PrefillReservationGuard, Self::Error> {
        self.begin_prefill_reservation(reservation)
    }

    /// Coordinates caller access before the one-shot reservation entry.
    ///
    /// The default immediately uses the existing bounded entry. A backend may
    /// serialize ordinary constructor access here, retaining the same request
    /// and a real runtime loan through that single attempt. It must release the
    /// loan before replacing old roots, invoking callbacks, preparing inputs,
    /// voting, executing or waiting for native work. Installed original roles
    /// must keep their existing bounded attempt and exact slot ownership.
    /// This is not a retry of a failed entry, an admission or a completion proof.
    /// Shared cancellation boundaries sample their token after this call.
    fn coordinate_prefill_entry(
        &mut self,
        reservation: crate::working_memory::InferenceRequest,
        role: Option<crate::prefill::PrefillControlRole>,
    ) -> Result<Self::PrefillReservationGuard, Self::Error> {
        match role {
            Some(role) => self.begin_prefill_control(reservation, role),
            None => self.begin_prefill_reservation(reservation),
        }
    }

    /// Settles successful preparation/execution and closes its retention scope.
    /// An error must preserve unresolved retention through the guard's teardown.
    fn finish_prefill_reservation(
        &mut self,
        guard: Self::PrefillReservationGuard,
    ) -> Result<(), Self::Error>;

    /// Opts into the guarded current-source boundary before each input chunk.
    /// The default performs no source traversal and never calls the hook below.
    fn requires_prefill_opening_sources(&self) -> bool {
        false
    }

    /// Borrows the actual state and selected execution after the observer begins
    /// the chunk, before observer opening/retention and input preparation. The
    /// original preparation guard remains live; an error enters the existing
    /// input agreement as a mechanism failure before model/state work.
    ///
    /// Neither source access nor the canonical context grants memory or native
    /// work authority. Implementations must account for any inventory they build,
    /// validate its completeness and preserve unresolved resources on failure.
    /// The loans end before the observer or input source runs; no model lookup or
    /// second session loan is needed. Opting in does not activate a capture gate.
    fn prepare_prefill_opening_sources(
        &mut self,
        _state: &Self::State,
        _context: &crate::inspection::PrefillChunkRetentionContext<'_>,
        _execution: &crate::inspection::PrefillOpeningExecution<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Require validation of the actual stored prepared runtime token before
    /// either source callback. Defaults preserve legacy source traversal.
    fn requires_prepared_prefill_sources(&self) -> bool {
        false
    }

    /// Borrows the same session's completed state and current execution after
    /// canonical chunk completion and before observer source retirement. The
    /// existing cancellation-retention guard is live. Errors participate in its
    /// existing readiness vote and preserve the original cause; no extra vote,
    /// completion certificate or storage authority is issued by this callback.
    fn prepare_prefill_retirement_sources(
        &mut self,
        _state: &Self::State,
        _ticket: &crate::inspection::SettledPrefillChunkRetention,
        _execution: &crate::inspection::PrefillOpeningExecution<'_, B::Tensor>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Exact slot facts retained at selected materialization preparation.
    /// Empty means that this mechanism does not expose prepared slot metadata.
    fn prepared_parameter_slots(&self) -> &[crate::parameter_operations::PreparedParameterSlot] {
        &[]
    }
    /// Architecture parameter metadata retained before ordinary/bank ownership separation.
    fn parameter_declarations(&self) -> &[eredu_nn::ParameterMetadata] {
        &[]
    }
    /// Concrete mutable-state realization paired with the architecture.
    type State: RuntimeState<B> + crate::working_memory::InferenceStateRetention;
    /// Shared failure type for resident and bounded runtime policies.
    type PolicyError;
    /// Concrete policy that owns fully resident bound units.
    type ResidentPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Concrete policy used for host-windowed or disk-streamed traversal.
    type BoundedPolicy: LayerwisePolicy<B, A::Unit, Error = Self::PolicyError>;
    /// Opaque state checkpoint owned by the backend mechanism.
    type StateCheckpoint: InferenceStateRetention;
    /// Backend-native mutable-state residency report.
    type StateReport;
    /// Backend-native parameter/runtime residency report.
    type ExecutionReport;
    /// Mechanism failure.
    type Error;

    /// Takes the aggregate report produced while realizing the selected
    /// materialization tasks.
    ///
    /// The neutral constructor calls this exactly once after successful
    /// preparation and retains the value with the completed session. This
    /// keeps report handoff in the same typed sequencing path as preparation
    /// instead of requiring an adapter-owned synchronization side channel.
    fn take_materialization_report(
        &mut self,
    ) -> Result<Option<crate::WeightMaterializationReport>, Self::Error> {
        Ok(None)
    }

    /// Configures rank-local neutral placement facts before partition payload
    /// preparation and state realization.
    fn configure_partition(
        &mut self,
        _target_layout: crate::LocalModelLayout,
        _source_layout: Option<crate::LocalModelLayout>,
        _rank: eredu_core::cache::CacheRankIdentity,
        _global_layer_start: usize,
    ) {
    }

    /// Prepares exact payload work for an explicitly selected rank-local unit
    /// sequence.
    ///
    /// The default is suitable for mechanisms whose ordinary preparation
    /// already accepts the global layout. Backends with distinct local-unit
    /// binding storage override this method while retaining the neutral
    /// construction sequencing.
    #[allow(clippy::too_many_arguments)]
    fn prepare_partition_materialization(
        &mut self,
        architecture: &mut A,
        global_layout: &crate::ExecutionUnitLayout,
        addresses: &[crate::ExecutionUnitAddress],
        task_partition: &crate::ReplicatedTextMaterializationPartitionPlan,
        units: &mut [A::Unit],
        source_architecture: Option<&mut A>,
        source_units: Option<&mut [A::Unit]>,
        tasks: &[ReplicatedTextMaterializationTask],
        addressable_parameters: &[String],
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error> {
        let _ = (addresses, task_partition);
        self.prepare_materialization(
            architecture,
            global_layout,
            units,
            source_architecture,
            source_units,
            tasks,
            addressable_parameters,
            context,
        )
    }

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

    /// Copies checkpoint storage; the shared wrapper attaches backing charges.
    fn copy_checkpoint_state(
        &mut self,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::StateCheckpoint, Self::Error>;

    /// Captures every live component and retains its exact inference charges.
    fn checkpoint_state(
        &mut self,
        state: &Self::State,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::StateCheckpoint, Self::Error> {
        let mut checkpoint = self.copy_checkpoint_state(state, context)?;
        checkpoint.inherit_inference_retention(state);
        Ok(checkpoint)
    }

    /// Restores storage after the shared wrapper has retained both charge sets.
    fn restore_checkpoint_state(
        &mut self,
        state: &mut Self::State,
        checkpoint: Self::StateCheckpoint,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error>;

    /// Restores every component from an opaque checkpoint.
    fn restore_state(
        &mut self,
        state: &mut Self::State,
        mut checkpoint: Self::StateCheckpoint,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<(), Self::Error> {
        // Preserve current charges even if the mechanism replaces all state,
        // and imported charges even if native restoration fails partway through.
        // Empty branches have no charges/admission to transfer. Avoid creating
        // transient revision identities which restore_admission and the final
        // guard immediately invalidate; the same restore worker still runs.
        if !state.inference_retention().is_empty() || !checkpoint.inference_retention().is_empty() {
            state.inherit_inference_retention(&checkpoint);
            checkpoint.inherit_inference_retention(state);
        }
        state
            .inference_retention_mut()
            .restore_admission(checkpoint.inference_retention());
        // A mechanism can replace the whole state with its checkpoint, including
        // the checkpoint's old revision. Invalidate the final installed owner on
        // success, error or unwind so equal-frontier restoration cannot revive it.
        struct Restoring<'a, S: InferenceStateRetention>(&'a mut S);
        impl<S: InferenceStateRetention> Drop for Restoring<'_, S> {
            fn drop(&mut self) {
                self.0.inference_retention_mut().invalidate_revision();
            }
        }
        let restoring = Restoring(state);
        self.restore_checkpoint_state(&mut *restoring.0, checkpoint, context)
    }

    /// Forks canonical state for one independently advanceable prediction lane.
    ///
    /// The default composes ordinary realization and checkpoint restoration. Backends whose
    /// state carries immutable transaction identity, such as a paged-cache residency session,
    /// override this operation so copied content receives one coherent independent identity
    /// without weakening unrelated cross-session restore validation.
    fn fork_prediction_target_state(
        &mut self,
        state: &Self::State,
        selected: &SelectedStateRealization,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<Self::State, Self::Error> {
        let checkpoint = self.checkpoint_state(state, context)?;
        let mut fork = self.realize_state(selected, context)?;
        self.restore_state(&mut fork, checkpoint, context)?;
        Ok(fork)
    }

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

    /// Complete retained future media roots in addition to ordinary output/state.
    /// Defaults reject the media protocol before source work; implementations
    /// must not infer completion of unused roots from the current span output.
    fn supports_media_ingress_completion(&self) -> bool {
        false
    }

    /// Exact ordinary completion over a closed borrowed retained-root source.
    /// `None` means unavailable; no implicit fallback or newly funded work.
    fn complete_media_ingress(
        &mut self,
        _output: Option<&B::Tensor>,
        _state: &Self::State,
        _roots: &crate::media_prefill::RetainedMediaRoots<'_, B::Tensor>,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Option<Result<(), Self::Error>> {
        None
    }

    /// Retains optional final output and all mutable state through exact completion.
    /// A state-only chunk supplies no vocabulary output; state/token-validation
    /// dependencies still have to settle before the next chunk may start.
    fn complete(
        &mut self,
        output: Option<&B::Tensor>,
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

    /// Whether this operation preserves target geometry, parameter topology and
    /// observation declarations on success, error and unwind. Stateful neural
    /// execution may update its supplied lane and module execution state. It
    /// must not replace the architecture or change those declarations.
    ///
    /// The default invalidates prepared bindings before exposing the target.
    /// Closed architecture-owned prediction operations opt in; this is a
    /// semantic execution contract, not source or allocation authority.
    fn preserves_architecture_declarations(&self) -> bool {
        false
    }

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
    fn with_parameter_slots(
        &mut self,
        location: &crate::parameter_operations::PreparedParameterLocation,
        operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            B::Tensor,
            R::Error,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, crate::LayerwiseAcquireError<A::Error, R::Error>> {
        match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => {
                runtime.with_parameter_slots(location, operation, context)
            }
            ReplicatedTextRuntimeKind::Bounded(runtime) => {
                runtime.with_parameter_slots(location, operation, context)
            }
        }
    }

    fn publish_parameter_replacements(
        &mut self,
        values: &BTreeMap<String, B::Tensor>,
        active: bool,
    ) -> Result<bool, R::Error> {
        match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => {
                runtime.publish_parameter_replacements(values, active)
            }
            ReplicatedTextRuntimeKind::Bounded(runtime) => {
                runtime.publish_parameter_replacements(values, active)
            }
        }
    }

    fn visit_loaded_parameters(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => {
                runtime.visit_loaded_parameters(visitor)
            }
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.visit_loaded_parameters(visitor),
        }
    }

    /// Borrows the actual selected static aggregate without loading a unit.
    /// The aggregate is a component source, not complete architecture storage.
    pub fn static_modules_ref(&self) -> &A::StaticModules {
        match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime.architecture().static_modules(),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.architecture().static_modules(),
        }
    }

    /// Visits the exact currently retained ordinary runtime owners. No unit
    /// acquisition, state mutation or native work is authorized. False preserves
    /// incomplete coverage even after a known prefix has been visited.
    pub fn visit_retained_values(&self, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime.visit_retained_values(visitor),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime.visit_retained_values(visitor),
        }
    }

    fn forward_with_observer<'a, O>(
        &mut self,
        input: A::Input<'a>,
        state: &mut S,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        match &mut self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => runtime
                .forward_with_observer_and_context_with_readout(
                    input, state, context, observer, demand,
                )
                .map_err(map_layerwise_error),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                .forward_with_observer_and_context_with_readout(
                    input, state, context, observer, demand,
                )
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
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
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
                .forward_with_provider_and_observer_and_context_with_readout(
                    input, state, pass, provider, context, observer, demand,
                )
                .map_err(map_layerwise_error),
            ReplicatedTextRuntimeKind::Bounded(runtime) => runtime
                .forward_with_provider_and_observer_and_context_with_readout(
                    input, state, pass, provider, context, observer, demand,
                )
                .map_err(map_layerwise_error),
        }
    }

    fn resident_policy(&self) -> Option<&R> {
        match &self.kind {
            ReplicatedTextRuntimeKind::Resident(runtime) => Some(runtime.policy()),
            ReplicatedTextRuntimeKind::Bounded(_) => None,
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
                runtime.apply_prediction_target_operation(operation, state, None, context)
            }
            ReplicatedTextRuntimeKind::Bounded(runtime) => {
                runtime.apply_prediction_target_operation(operation, state, None, context)
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

    /// Whether the selected strategy invokes the architecture's ordinary unit
    /// equations. Provider and partition strategies must supply their own exact
    /// workspace adapter; matching residency or shapes alone is insufficient.
    /// This is a descriptive mechanism fact, not execution authority.
    const ORDINARY_UNIT_EQUATIONS: bool = false;

    /// Equation-parity fact for the actual retained execution strategy.
    /// The default preserves existing statically declared ordinary strategies;
    /// provider strategies may forward their selected provider's closed fact.
    /// This descriptive query constructs, submits and acquires nothing.
    fn uses_ordinary_unit_equations(&self) -> bool {
        Self::ORDINARY_UNIT_EQUATIONS
    }

    /// Concrete execution runtime paired before the shared session lifecycle begins.
    type Runtime;

    /// Exchanges a caller-owned context loan with the selected runtime. A
    /// successful exchange promises the same allocation-free inverse. No new
    /// context is copied or retained by this hook.
    fn exchange_parallel_context(_runtime:&mut Self::Runtime,_context:&mut B::ParallelContext)->bool { false }

    /// Executes the same forward under a lexical context loan and restores
    /// both values on return, failure or unwind. This creates no source grant.
    fn with_borrowed_parallel_context<T,F>(
        runtime:&mut Self::Runtime,context:&mut B::ParallelContext,
        funding:&eredu_nn::workspace::HostMetadataFunding,run:F,
    )->Result<T,PreparedParallelContextCause>
    where F:FnOnce(&mut Self::Runtime)->T {
        parallel_context::with_borrowed_runtime(runtime,context,funding,Self::exchange_parallel_context,run)
    }

    /// Exchanges only the runtime's request-control slot. Success promises
    /// allocation-free restoration and leaves neural parallel selection intact.
    fn exchange_parallel_control_context(
        _runtime: &mut Self::Runtime, _context: &mut Option<Box<B::ParallelContext>>,
    ) -> bool { false }

    /// Reuses the scoped restoration worker for a separate control-only loan.
    fn with_borrowed_parallel_control_context<T, F>(
        runtime: &mut Self::Runtime, context: &mut Option<Box<B::ParallelContext>>,
        funding: &eredu_nn::workspace::HostMetadataFunding, run: F,
    ) -> Result<T, PreparedParallelContextCause>
    where F: FnOnce(&mut Self::Runtime) -> T,
    {
        parallel_context::with_borrowed_runtime(
            runtime, context, funding, Self::exchange_parallel_control_context, run)
    }

    /// Replaces only the selected executor's opaque parallel context. Success
    /// returns its exact prior value and must remain reversible until restored.
    /// Refusal returns `replacement` unchanged before mutation. No source,
    /// completion, allocation or communication authority is created here.
    fn replace_parallel_context(_runtime:&mut Self::Runtime,replacement:B::ParallelContext)
        -> Result<B::ParallelContext,B::ParallelContext>
    where B::ParallelContext:Sized { Err(replacement) }

    /// Temporarily lends one already prepared backend context to the actual
    /// selected runtime. Callers own source/admission and completion custody;
    /// this shared worker guarantees restoration on return, error or unwind.
    fn with_prepared_parallel_context<T,F>(
        runtime:&mut Self::Runtime, context:B::ParallelContext,
        funding:&eredu_nn::workspace::HostMetadataFunding, run:F,
    )->Result<T,PreparedParallelContextFailure<B::ParallelContext>>
    where B::ParallelContext:Sized, F:FnOnce(&mut Self::Runtime)->T,
    {
        parallel_context::with_runtime(runtime,context,funding,Self::replace_parallel_context,run)
    }

    /// Permanently fences this strategy's retained communication incarnations.
    /// This required operation submits, allocates, waits and retires nothing.
    /// The enclosing session stores the phase; existing epochs remain unchanged.
    fn mark_terminal_failure(runtime: &Self::Runtime, phase: crate::DistributedExecutionPhase);

    /// Exact hook coverage of the retained executor, including specialized unit calls.
    fn observation_hooks(_runtime: &Self::Runtime) -> crate::inspection::ObservationHookSupport {
        Default::default()
    }

    /// Visits actual loaded slots without reopening or rematerializing a source.
    /// False means unavailable and must be returned before visiting any slot.
    fn visit_loaded_parameters(
        _runtime: &mut Self::Runtime,
        _visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        false
    }

    /// Borrows the declared static aggregate of this retained execution.
    /// None is unavailable coverage, not an empty aggregate. No construction,
    /// replacement, source resolution, completion or native work is authorized.
    /// Extra architecture fields and policy units remain separate owners.
    fn static_modules_ref(_runtime: &Self::Runtime) -> Option<&A::StaticModules> {
        None
    }

    /// Borrows exact parameter declarations/values from this retained strategy.
    /// The default uses its static/resident pair; specialized executors lend
    /// their actual policy through the same source worker. A partial traversal
    /// never authorizes source installation or any parameter/native operation.
    fn visit_parameter_sources<V>(
        runtime: &Self::Runtime,
        visitor: &mut V,
        context: &eredu_nn::workspace::WorkspaceContext,
    ) -> Result<bool, eredu_nn::Error>
    where
        V: for<'source> eredu_nn::ParameterSourceVisitor<'source, B::Tensor>,
    {
        context.charge_metadata(std::mem::size_of::<(
            &Self::Runtime,
            &mut V,
            &eredu_nn::workspace::WorkspaceContext,
            Option<&A::StaticModules>,
            Option<&R>,
            Result<bool, eredu_nn::Error>,
        )>())?;
        let (Some(static_modules), Some(policy)) = (
            Self::static_modules_ref(runtime),
            Self::resident_policy(runtime),
        ) else {
            return Ok(false);
        };
        crate::parameter_operations::visit_parameter_sources_in_parts::<B, A::Unit, R, V>(
            static_modules,
            policy,
            visitor,
            context,
        )
    }

    /// Traverses retained parameters and numerical helpers through the paired owner,
    /// without replacement access, reloading, evaluation or completion polling.
    /// False means incomplete coverage, possibly after visiting known values.
    fn visit_retained_values(runtime: &Self::Runtime, visitor: &mut dyn FnMut(&B::Tensor)) -> bool;

    /// Checked ceiling from this actual retained runtime topology. Unknown is
    /// never inferred from a traversal count. The default leaves strategy-local
    /// inventory binding unfinished and grants no acquisition or storage rights.
    fn retained_value_slot_bound(_runtime: &Self::Runtime) -> Option<usize> {
        None
    }

    /// Performs parameter work inside one selected residency owner.
    fn with_parameter_slots(
        _runtime: &mut Self::Runtime,
        _location: &crate::parameter_operations::PreparedParameterLocation,
        _operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            B::Tensor,
            R::Error,
        >,
        _context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, crate::LayerwiseAcquireError<A::Error, R::Error>> {
        Ok(false)
    }

    /// Atomically publishes completed replacements in the selected ownership mechanism.
    fn publish_parameter_replacements(
        _runtime: &mut Self::Runtime,
        _values: &BTreeMap<String, B::Tensor>,
        _active: bool,
    ) -> Result<bool, R::Error> {
        Ok(false)
    }

    /// Borrows the actual permanently resident policy without changing selection.
    /// Specialized strategies expose it only when this is their installed owner.
    fn resident_policy(_runtime: &Self::Runtime) -> Option<&R> {
        None
    }

    /// Actual forward worker's group protocol, independent of the borrowed
    /// residency owner and its mechanism-specific final completion.
    fn group_submission_mechanism(runtime: &Self::Runtime) -> crate::GroupSubmissionMechanism;

    /// Returns bounded residency state for the shared session report.
    fn bounded_policy(runtime: &Self::Runtime) -> Option<&P>;

    /// Returns the execution residency actually installed on this rank.
    fn execution_residency(
        runtime: &Self::Runtime,
        selected: &SelectedReplicatedTextRealization,
    ) -> ExecutionResidency;

    /// Build this strategy's exact immutable observation source during initial
    /// loading. None means that its prepared traversal has no source producer.
    /// Source publication and finite execution permission remain separate.
    fn prepare_observation_paths(
        _runtime: &Self::Runtime,
    ) -> Result<Option<crate::PreparedLayeredObservationPaths>,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        Ok(None)
    }

    /// Coldly bind an existing physical source during authorized preparation.
    /// Default rejection prevents a specialized executor from silently falling
    /// back to an allocating traversal when a prepared observer is required.
    fn bind_observation_paths(
        _runtime: &Self::Runtime,
        _source: &crate::SharedLayeredObservationPaths,
        metadata: Option<crate::layered::LayeredMetadata<A::Error>>) -> Result<
        crate::PreparedLayeredObservationPaths,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    > {
        Err(ReplicatedTextSessionError::PreparedObservation(
            PreparedSessionObservationError::Unavailable,
        ))
    }

    /// Check the existing runtime binding without allocating or model work.
    fn validate_observation_paths(
        _runtime: &Self::Runtime,
        _paths: &crate::PreparedLayeredObservationPaths,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        Err(ReplicatedTextSessionError::PreparedObservation(
            PreparedSessionObservationError::Unavailable,
        ))
    }

    /// Execute through the prepared ordinary hook after shared input agreement.
    #[allow(clippy::too_many_arguments)]
    fn forward_with_prepared_observer<'a, O>(
        &mut self,
        _runtime: &mut Self::Runtime,
        _input: A::Input<'a>,
        _state: &mut S,
        _pass: ExpertPass,
        _context: &<B::Tensor as Tensor>::Context,
        _observer: &mut O,
        _paths: &crate::PreparedLayeredObservationPaths,
        _demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        Err(ReplicatedTextSessionError::PreparedObservation(
            PreparedSessionObservationError::Unavailable,
        ))
    }

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
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
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

    /// Publishes through the same policy using a retained native occurrence.
    /// Strategies without that producer refuse an explicitly prepared context.
    fn publish_observed_output_with_parallel(runtime:&mut Self::Runtime,output:B::Tensor,
        context:&<B::Tensor as Tensor>::Context,
        prepared:Option<(&B::ParallelContext,&eredu_nn::workspace::HostMetadataFunding)>)
        ->Result<B::Tensor,ReplicatedTextSessionError<A::Error,R::Error,std::convert::Infallible>> {
        if prepared.is_some(){return Err(ReplicatedTextSessionError::ParallelContext(PreparedParallelContextCause::Unsupported));}
        Self::publish_observed_output(runtime,output,context)
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

    /// Exact native operation selected for this actual control event.
    /// None is a local phase and must not create a native control claim.
    fn parallel_control_operation(
        _runtime: &Self::Runtime, _event: ParallelControlEvent,
    ) -> Option<crate::CommunicationOperation> { None }

    /// Performs strategy-specific distributed commit only after output
    /// intervention and exact mechanism completion have succeeded.
    fn commit_after_completion(
        _runtime: &mut Self::Runtime,
        epoch: DistributedCommitEpoch,
        _context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> DistributedCommitOutcome {
        DistributedCommitOutcome::Committed(epoch)
    }

    /// Final decision under an exact prepared control context. The default
    /// preserves ordinary behavior and explicitly refuses an unconsumed context.
    fn commit_after_completion_with_parallel(
        runtime: &mut Self::Runtime, epoch: DistributedCommitEpoch,
        context: &<B::Tensor as Tensor>::Context,
        prepared: Option<(&B::ParallelContext, &eredu_nn::workspace::HostMetadataFunding)>,
    ) -> Result<DistributedCommitOutcome, ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        if prepared.is_some() {
            return Err(ReplicatedTextSessionError::ParallelContext(PreparedParallelContextCause::Unsupported));
        }
        Ok(Self::commit_after_completion(runtime, epoch, context))
    }

    /// Same phase policy with an explicit prepared communication context.
    fn agree_distributed_phase_with_parallel(
        runtime: &mut Self::Runtime, phase: crate::DistributedExecutionPhase,
        local_success: bool, context: &<B::Tensor as Tensor>::Context,
        prepared: Option<(&B::ParallelContext, &eredu_nn::workspace::HostMetadataFunding)>,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        if prepared.is_some() {
            return Err(ReplicatedTextSessionError::ParallelContext(PreparedParallelContextCause::Unsupported));
        }
        Self::agree_distributed_phase(runtime, phase, local_success, context)
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
    fn group_submission_mechanism(_runtime: &Self::Runtime) -> crate::GroupSubmissionMechanism {
        crate::GroupSubmissionMechanism::LayeredGraph
    }

    fn mark_terminal_failure(_runtime: &Self::Runtime, _phase: crate::DistributedExecutionPhase) {}

    const ORDINARY_UNIT_EQUATIONS: bool = true;

    type Runtime = ReplicatedTextRuntime<A, B, S, R, P>;

    fn bind_observation_paths(
        runtime: &Self::Runtime,
        source: &crate::SharedLayeredObservationPaths,
        metadata: Option<crate::layered::LayeredMetadata<A::Error>>) -> Result<
        crate::PreparedLayeredObservationPaths,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    > {
        runtime.bind_observation_paths(source, metadata)
    }

    fn validate_observation_paths(
        runtime: &Self::Runtime,
        paths: &crate::PreparedLayeredObservationPaths,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        runtime.validate_observation_paths(paths)
    }

    fn forward_with_prepared_observer<'a, O>(
        &mut self,
        runtime: &mut Self::Runtime,
        input: A::Input<'a>,
        state: &mut S,
        _pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: &crate::PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.forward_with_prepared_observer(input, state, context, observer, paths, demand)
    }

    fn static_modules_ref(runtime: &Self::Runtime) -> Option<&A::StaticModules> {
        Some(runtime.static_modules_ref())
    }

    fn visit_retained_values(runtime: &Self::Runtime, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        runtime.visit_retained_values(visitor)
    }

    fn retained_value_slot_bound(runtime: &Self::Runtime) -> Option<usize> {
        runtime.retained_value_slot_bound()
    }

    fn visit_loaded_parameters(
        runtime: &mut Self::Runtime,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        runtime.visit_loaded_parameters(visitor)
    }

    fn with_parameter_slots(
        runtime: &mut Self::Runtime,
        location: &crate::parameter_operations::PreparedParameterLocation,
        operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            B::Tensor,
            R::Error,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, crate::LayerwiseAcquireError<A::Error, R::Error>> {
        runtime.with_parameter_slots(location, operation, context)
    }

    fn publish_parameter_replacements(
        runtime: &mut Self::Runtime,
        values: &BTreeMap<String, B::Tensor>,
        active: bool,
    ) -> Result<bool, R::Error> {
        runtime.publish_parameter_replacements(values, active)
    }

    fn resident_policy(runtime: &Self::Runtime) -> Option<&R> {
        runtime.resident_policy()
    }

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
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.forward_with_observer(input, state, context, observer, demand)
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
    fn group_submission_mechanism(_runtime: &Self::Runtime) -> crate::GroupSubmissionMechanism {
        crate::GroupSubmissionMechanism::LayeredGraph
    }

    fn mark_terminal_failure(_runtime: &Self::Runtime, _phase: crate::DistributedExecutionPhase) {}

    // The retained provider type supplies this descriptive fact. Addressable
    // and custom providers cannot borrow a resident equation quote by matching
    // the target layout or selecting the same residency enum.
    fn uses_ordinary_unit_equations(&self) -> bool {
        <Provider as RoutedExpertProvider<B>>::resident_unit_equations()
    }

    type Runtime = ReplicatedTextRuntime<A, B, S, R, P>;

    fn bind_observation_paths(
        runtime: &Self::Runtime,
        source: &crate::SharedLayeredObservationPaths,
        metadata: Option<crate::layered::LayeredMetadata<A::Error>>) -> Result<
        crate::PreparedLayeredObservationPaths,
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    > {
        runtime.bind_observation_paths(source, metadata)
    }

    fn validate_observation_paths(
        runtime: &Self::Runtime,
        paths: &crate::PreparedLayeredObservationPaths,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>> {
        runtime.validate_observation_paths(paths)
    }

    fn forward_with_prepared_observer<'a, O>(
        &mut self,
        runtime: &mut Self::Runtime,
        input: A::Input<'a>,
        state: &mut S,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        paths: &crate::PreparedLayeredObservationPaths,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, R::Error, std::convert::Infallible>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        runtime.forward_with_prepared_provider_observer(
            input,
            state,
            pass,
            &mut self.provider,
            context,
            observer,
            paths,
            demand,
        )
    }

    fn static_modules_ref(runtime: &Self::Runtime) -> Option<&A::StaticModules> {
        Some(runtime.static_modules_ref())
    }

    fn visit_retained_values(runtime: &Self::Runtime, visitor: &mut dyn FnMut(&B::Tensor)) -> bool {
        runtime.visit_retained_values(visitor)
    }

    fn retained_value_slot_bound(runtime: &Self::Runtime) -> Option<usize> {
        runtime.retained_value_slot_bound()
    }

    fn visit_loaded_parameters(
        runtime: &mut Self::Runtime,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        runtime.visit_loaded_parameters(visitor)
    }

    fn with_parameter_slots(
        runtime: &mut Self::Runtime,
        location: &crate::parameter_operations::PreparedParameterLocation,
        operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            B::Tensor,
            R::Error,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, crate::LayerwiseAcquireError<A::Error, R::Error>> {
        runtime.with_parameter_slots(location, operation, context)
    }

    fn publish_parameter_replacements(
        runtime: &mut Self::Runtime,
        values: &BTreeMap<String, B::Tensor>,
        active: bool,
    ) -> Result<bool, R::Error> {
        runtime.publish_parameter_replacements(values, active)
    }

    fn resident_policy(runtime: &Self::Runtime) -> Option<&R> {
        runtime.resident_policy()
    }

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
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (Option<B::Tensor>, A::ForwardContext),
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
            demand,
        )
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

fn record_successful_restoration<E>(
    generation: &mut Option<u64>,
    restored: Result<(), E>,
) -> Result<(), E> {
    restored?;
    *generation = generation.and_then(|value| value.checked_add(1));
    Ok(())
}

#[cfg(test)]
mod restoration_witness_tests {
    use super::record_successful_restoration;

    #[test]
    fn failed_restore_and_stale_snapshot_do_not_prove_new_restoration() {
        let mut generation = Some(0);
        let before = generation;
        assert!(record_successful_restoration(&mut generation, Err("restore failed")).is_err());
        assert_eq!(generation, before);
        record_successful_restoration(&mut generation, Ok::<_, ()>(())).unwrap();
        assert_eq!(generation, Some(1));
        let prior_restore = generation;
        assert!(
            record_successful_restoration(&mut generation, Err("later restore failed")).is_err()
        );
        assert_eq!(generation, prior_restore);
    }

    #[test]
    fn restoration_counter_overflow_permanently_disables_the_witness() {
        let mut generation = Some(u64::MAX);
        record_successful_restoration(&mut generation, Ok::<_, ()>(())).unwrap();
        assert_eq!(generation, None);
        record_successful_restoration(&mut generation, Ok::<_, ()>(())).unwrap();
        assert_eq!(generation, None);
    }
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
    // The runtime owns its private binding; metadata projections bind this
    // same physical source independently during their own cold preparation.
    observation_paths: Option<crate::PreparedLayeredObservationPaths>,
    driver: D,
    state: M::State,
    mechanisms: M,
    materialization_report: Option<crate::WeightMaterializationReport>,
    partition_parameters: Option<std::sync::Arc<crate::ArchitectureParameterDescription>>,
    prompt_cache_identity: Option<PromptCacheModelIdentity>,
    committed_prompt_input_identity: Option<SharedPreparedInputCacheIdentity>,
    next_commit_epoch: DistributedCommitEpoch,
    active_commit_epoch: Option<DistributedCommitEpoch>,
    last_commit_outcome: Option<DistributedCommitOutcome>,
    successful_state_restorations: Option<u64>,
    control_identity: std::sync::Arc<()>,
    prefill_identity: crate::working_memory::InferenceExecutionIdentity,
    inference_guard: Option<M::PrefillReservationGuard>,
    active_prefill_control: Option<crate::prefill::PrefillControlRole>,
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
    parameters: std::sync::Arc<crate::ArchitectureParameterDescription>,
    runtime: R,
    state: S,
    selected_state: SessionStateRealization,
    prompt_cache_identity: Option<PromptCacheModelIdentity>,
    output_selection: ReplicatedTextOutputSelection,
}

impl<R, S> PreparedPartitionedSessionRuntime<R, S> {
    /// Complete selected parameter topology checked against the architecture and
    /// local materialization tasks, before the native runtime factory executes.
    /// This includes parameters owned by other ranks and no native handles.
    pub fn parameter_description(
        &self,
    ) -> &std::sync::Arc<crate::ArchitectureParameterDescription> {
        &self.parameters
    }

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

/// Unit set constructed by the neutral partition lifecycle.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum PartitionedUnitScope {
    /// Construct every unit in the selected global execution layout.
    All,
    /// Construct only units owned by the admitted rank-local partition.
    Owned,
}

/// Architecture, policy, state, and communication authority completed by the
/// neutral partition construction lifecycle.
pub struct PreparedPartitionedRuntimeComponents<A, G, W, S, P> {
    architecture: A,
    partition: ArchitecturePartition<G, W>,
    communication: CommunicationManifest,
    execution_policy: P,
    bounded_policy: Option<P>,
    state: S,
}

impl<A, G, W, S, P> PreparedPartitionedRuntimeComponents<A, G, W, S, P> {
    /// Consumes the completed neutral lifecycle into a backend-native runtime
    /// factory. The factory cannot repeat task partitioning, materialization,
    /// state selection, or residency selection.
    pub fn into_parts(
        self,
    ) -> (
        A,
        ArchitecturePartition<G, W>,
        CommunicationManifest,
        P,
        Option<P>,
        S,
    ) {
        (
            self.architecture,
            self.partition,
            self.communication,
            self.execution_policy,
            self.bounded_policy,
            self.state,
        )
    }
}

/// Failure from the reusable rank-local preparation lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedRuntimeConstructionError<
    A = std::convert::Infallible,
    M = std::convert::Infallible,
> {
    /// Architecture, selection, task, or partition authority disagreed.
    #[error("partitioned runtime contract mismatch: {0}")]
    Contract(String),
    /// Architecture unit construction failed.
    #[error("partitioned runtime architecture construction failed: {0}")]
    Architecture(#[source] A),
    /// The concrete mechanism rejected materialization, state, or policy work.
    #[error("partitioned runtime mechanism failed: {0}")]
    Mechanism(#[source] M),
}

/// Runs the reusable cold-path lifecycle for one exact partition.
///
/// Architecture authority and tasks arrive together from
/// [`prepare_partitioned_session_runtime`]. This driver chooses the exact unit
/// set, constructs target and optional transform-source units, prepares and
/// partitions materialization, realizes local state, and selects the resident
/// or bounded policy. The caller receives only the completed typed components
/// needed to create native communication and executor mechanisms.
#[allow(clippy::too_many_arguments)]
pub fn prepare_default_partitioned_runtime<A, B, M, G, W, P>(
    input: PartitionedSessionFactoryInput<A, G, W>,
    mut source_architecture: Option<A>,
    target_parallel_layout: crate::LocalModelLayout,
    source_parallel_layout: Option<crate::LocalModelLayout>,
    selected: &SelectedReplicatedTextRealization,
    topology: &PromptCacheTopology,
    scope: PartitionedUnitScope,
    addressable_parameters: &[String],
    mechanisms: &mut M,
    context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
) -> Result<
    PreparedPartitionedRuntimeComponents<A, G, W, M::State, P>,
    PartitionedRuntimeConstructionError<A::Error, M::Error>,
>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSessionMechanisms<A, B, ResidentPolicy = P, BoundedPolicy = P>,
    A: LayeredArchitecture<B, M::State>,
    A::Error: std::fmt::Display,
    M::Error: std::fmt::Display,
    P: LayerwisePolicy<B, A::Unit, Error = M::PolicyError> + Clone,
{
    let (mut architecture, partition, communication, tasks) = input.into_parts();
    let global_layout = partition.unit_layout().clone();
    let addresses=partitioned_materialization_addresses(&partition,scope)
        .map_err(PartitionedRuntimeConstructionError::Contract)?;
    let task_partition =
        plan_local_replicated_text_materialization_tasks(&tasks, &global_layout, &addresses)
            .map_err(|error| PartitionedRuntimeConstructionError::Contract(error.to_string()))?;
    let mut units=crate::layered::ordinary_addressed_units::<A,B,M::State>(&architecture,&addresses,context)
        .map_err(PartitionedRuntimeConstructionError::Architecture)?;
    let mut source_units=source_architecture.as_ref().map(|source|
        crate::layered::ordinary_addressed_units::<A,B,M::State>(source,&addresses,context)
            .map_err(PartitionedRuntimeConstructionError::Architecture)).transpose()?;
    let local_state = partition.state().ok_or_else(|| {
        PartitionedRuntimeConstructionError::Contract(
            "partition owns no local mutable state".into(),
        )
    })?;
    let selected_state = selected
        .state()
        .for_partitioned_geometry(local_state)
        .map_err(|error| PartitionedRuntimeConstructionError::Contract(error.to_string()))?;
    let rank = eredu_core::cache::CacheRankIdentity::new(
        topology.stage().map(|(_, rank)| rank),
        topology.shard().map(|(_, rank)| rank),
        topology.addressable().map(|(_, rank)| rank),
    );
    mechanisms.configure_partition(
        target_parallel_layout,
        source_parallel_layout,
        rank,
        local_state.global_layer_offset(),
    );
    mechanisms
        .prepare_partition_materialization(
            &mut architecture,
            &global_layout,
            &addresses,
            &task_partition,
            &mut units,
            source_architecture.as_mut(),
            source_units.as_deref_mut(),
            &tasks,
            addressable_parameters,
            context,
        )
        .map_err(|error| PartitionedRuntimeConstructionError::Mechanism(error))?;
    let state = mechanisms
        .realize_state(&selected_state, context)
        .map_err(|error| PartitionedRuntimeConstructionError::Mechanism(error))?;
    if state.optional_layout() != Some(selected_state.layout()) {
        return Err(PartitionedRuntimeConstructionError::Contract(
            "realized partition state differs from selected local geometry".into(),
        ));
    }
    let (execution_policy, bounded_policy) = match selected.residency() {
        LayerWeightResidency::FullyResident => (
            mechanisms
                .resident_policy(&mut architecture, units, selected, context)
                .map_err(|error| PartitionedRuntimeConstructionError::Mechanism(error))?,
            None,
        ),
        LayerWeightResidency::LayerwiseHost(_) | LayerWeightResidency::DenseDiskStream(_) => {
            drop(units);
            let policy = mechanisms
                .bounded_policy(&mut architecture, selected, context)
                .map_err(|error| PartitionedRuntimeConstructionError::Mechanism(error))?;
            (policy.clone(), Some(policy))
        }
    };
    Ok(PreparedPartitionedRuntimeComponents {
        architecture,
        partition,
        communication,
        execution_policy,
        bounded_policy,
        state,
    })
}

/// Exact unit addresses consumed by the existing partition materializer.
/// The architecture-selected scope preserves global parameter names while its
/// local residency policy assigns slots in this order.
pub fn partitioned_materialization_addresses<G,W>(
    partition:&ArchitecturePartition<G,W>, scope:PartitionedUnitScope,
)->Result<Vec<crate::ExecutionUnitAddress>,String> {
    let layout=partition.unit_layout();
    let addresses=match scope {
        PartitionedUnitScope::All => (0..layout.len()).map(|ordinal|layout.address(ordinal)
            .ok_or_else(||format!("global unit ordinal {ordinal} has no canonical address")))
            .collect::<Result<Vec<_>,_>>()?,
        PartitionedUnitScope::Owned => partition.units().collect(),
    };
    if addresses.is_empty(){return Err("partition owns no execution units".into());}
    Ok(addresses)
}

/// The policy-local unit layout of the same ordered global partition addresses.
/// This is descriptive construction metadata, not source or native authority.
pub fn partitioned_materialization_unit_layout(
    graph:&crate::ExecutionGraph, addresses:&[crate::ExecutionUnitAddress],
)->Result<crate::ExecutionUnitLayout,String> {
    let counts=(0..graph.groups().len()).map(|group|addresses.iter()
        .filter(|address|address.group()==group).count()).collect::<Vec<_>>();
    crate::ExecutionUnitLayout::new(graph,counts).map_err(|cause|cause.to_string())
}

/// Failure while consuming architecture authority into a rank-local runtime.
#[derive(Debug, thiserror::Error)]
pub enum PartitionedSessionPreparationError<E> {
    /// Retaining an owned architecture declaration failed in its metadata destination.
    #[error("partitioned session metadata construction failed: {0}")]
    Metadata(#[source] eredu_nn::Error),
    /// Architecture, partition, selection, or payload authority disagreed.
    #[error("partitioned session authority mismatch: {0}")]
    Contract(String),
    /// The rank-local runtime factory rejected the exact handoff.
    #[error("partitioned session runtime factory failed: {0}")]
    Factory(#[source] E),
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
    let parameters = crate::ArchitectureParameterDescription::into_owned(parameters,B::construction_metadata(context))
        .map_err(PartitionedSessionPreparationError::Metadata)?;
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
        return Err(PartitionedSessionPreparationError::Contract(format!(
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
        )));
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
        parameters: std::sync::Arc::new(parameters),
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
    prompt_input_identity: Option<SharedPreparedInputCacheIdentity>,
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
    prompt_input_identity: Option<SharedPreparedInputCacheIdentity>,
    next_commit_epoch: DistributedCommitEpoch,
    last_commit_outcome: Option<DistributedCommitOutcome>,
}

/// Fixed shared prediction-target preparation refusal. Original destinations
/// retain it without allocating the ordinary contract diagnostic String.
#[derive(Debug,Clone,Copy,PartialEq,thiserror::Error)]
pub enum PredictionTargetPreparationError {
    /// Existing commit/fence boundary, retained without legacy formatting.
    #[error(transparent)]
    Boundary(#[from] RuntimeInspectionBoundary),
    /// Stateless selection cannot produce a prediction lane cache.
    #[error("stateless session cannot prepare prediction target state")]
    Stateless,
    /// Copied state must retain the exact selected layout.
    #[error("realized state layout differs from selection")]
    Layout,
    /// All ranks must accept their local destination before it escapes.
    #[error("another rank could not prepare prediction target lane state")]
    Peer,
}

/// Cold-path failure from replicated-text construction or session control.
#[derive(Debug)]
pub enum ReplicatedTextSessionError<A, P, M>
where
    A: std::fmt::Display,
    P: std::fmt::Display,
    M: std::fmt::Display,
{
    /// The prepared architecture disagreed with its selected realization.
    Contract(String),
    /// Graph submission or dependency ordering failed with retained source custody.
    Submission(eredu_core::BackendFailure),
    /// Required borrowed observation traversal is unavailable or stale.
    PreparedObservation(PreparedSessionObservationError),
    /// A prepared backend context could not be installed before forward.
    ParallelContext(PreparedParallelContextCause),
    /// Native control-source preparation or binding failed with retained custody.
    ParallelControl(eredu_core::BackendFailure),
    /// Shared inference working-memory admission rejected the request.
    WorkingMemory(crate::working_memory::WorkingMemoryError),
    /// Portable distributed execution or agreement failed.
    Partition(crate::PartitionExecutionError),
    /// Architecture construction or execution failed.
    Architecture(A),
    /// Bounded residency failed.
    Policy(P),
    /// A native mechanism failed.
    Mechanism(M),
    /// Shared admission failed before the driver received mutable model state.
    /// This proves state preservation only; native completion/recovery is separate.
    BeforeStateMutation(Box<ReplicatedTextSessionError<A, P, M>>),
    /// Mutable-state access failed.
    State(StateError),
    /// Prompt-cache identity or manifest validation failed.
    PromptCache(PromptCacheError),
    /// The globally fixed final decision was an abort.
    CommitAborted {
        /// Durable transaction identity.
        epoch: DistributedCommitEpoch,
    },
    /// This rank may have contributed to a final decision it could not observe.
    CommitIndeterminate {
        /// Durable transaction identity.
        epoch: DistributedCommitEpoch,
        /// Exact final-decision cut which was not observed.
        phase: DistributedCommitPhase,
    },
}

impl<A: std::fmt::Display, P: std::fmt::Display, M: std::fmt::Display> std::fmt::Display
    for ReplicatedTextSessionError<A, P, M>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Contract(error) => write!(f, "replicated text contract mismatch: {error}"),
            Self::Submission(error) => std::fmt::Display::fmt(error, f),
            Self::PreparedObservation(error) => std::fmt::Display::fmt(error, f),
            Self::ParallelContext(error) => std::fmt::Display::fmt(error, f),
            Self::ParallelControl(error) => std::fmt::Display::fmt(error, f),
            Self::WorkingMemory(error) => std::fmt::Display::fmt(error, f),
            Self::Partition(error) => std::fmt::Display::fmt(error, f),
            Self::Architecture(error) => write!(f, "replicated text architecture failed: {error}"),
            Self::Policy(error) => write!(f, "replicated text residency failed: {error}"),
            Self::Mechanism(error) => write!(f, "replicated text mechanism failed: {error}"),
            Self::BeforeStateMutation(error) => write!(
                f,
                "replicated text admission rejected before state mutation: {error}"
            ),
            Self::State(error) => std::fmt::Display::fmt(error, f),
            Self::PromptCache(error) => std::fmt::Display::fmt(error, f),
            Self::CommitAborted { epoch } => {
                write!(f, "distributed transaction epoch {epoch:?} was aborted")
            }
            Self::CommitIndeterminate { epoch, phase } => write!(
                f,
                "distributed transaction epoch {epoch:?} is indeterminate at {phase:?}"
            ),
        }
    }
}

impl<A, P, M> std::error::Error for ReplicatedTextSessionError<A, P, M>
where
    A: std::error::Error + 'static,
    P: std::error::Error + 'static,
    M: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Submission(error) => Some(error),
            Self::PreparedObservation(error) => Some(error),
            Self::ParallelContext(error) => Some(error),
            Self::ParallelControl(error) => Some(error),
            Self::WorkingMemory(error) => Some(error),
            Self::Partition(error) => Some(error),
            Self::Architecture(error) => Some(error),
            Self::Policy(error) => Some(error),
            Self::Mechanism(error) => Some(error),
            Self::BeforeStateMutation(error) => Some(error.as_ref()),
            Self::State(error) => Some(error),
            Self::PromptCache(error) => Some(error),
            Self::Contract(_) | Self::CommitAborted { .. } | Self::CommitIndeterminate { .. } => {
                None
            }
        }
    }
}

impl<A: std::fmt::Display, P: std::fmt::Display, M: std::fmt::Display> From<StateError>
    for ReplicatedTextSessionError<A, P, M>
{
    fn from(error: StateError) -> Self {
        Self::State(error)
    }
}

impl<A: std::fmt::Display, P: std::fmt::Display, M: std::fmt::Display> From<PromptCacheError>
    for ReplicatedTextSessionError<A, P, M>
{
    fn from(error: PromptCacheError) -> Self {
        Self::PromptCache(error)
    }
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
    // Completed ordinary construction retains its exact filtered task source.
    // Checked plain quotes can borrow the full selection without another owner.
    materialization: Option<PreparedContractMaterialization>,
    prompt_cache_identity: PromptCacheModelIdentity,
    output_selection: ReplicatedTextOutputSelection,
}

/// Move-only graph and unit layout from a validated text construction or the
/// actual completed workspace unit builder. This is descriptive metadata only;
/// it retains no native source, readiness or submission authority.
pub struct PreparedReplicatedTextExecutionGeometry {
    source: PreparedTextGeometrySource,
}
enum PreparedTextGeometrySource {
    Selected(SelectedReplicatedTextRealization),
    Workspace {graph:crate::ExecutionGraph, units:crate::ExecutionUnitLayout},
}
impl PreparedReplicatedTextExecutionGeometry {
    /// The exact graph compared with or emitted by the architecture constructor.
    pub fn graph(&self) -> &crate::ExecutionGraph {
        match &self.source {
            PreparedTextGeometrySource::Selected(selected)=>selected.requirements().execution_graph(),
            PreparedTextGeometrySource::Workspace{graph,..}=>graph,
        }
    }
    /// Canonical group/unit layout of the same completed constructor.
    pub fn units(&self) -> &crate::ExecutionUnitLayout {
        match &self.source {
            PreparedTextGeometrySource::Selected(selected)=>selected.requirements().execution_units(),
            PreparedTextGeometrySource::Workspace{units,..}=>units,
        }
    }
    pub(crate) fn from_workspace_units(
        graph:crate::ExecutionGraph, counts:&[usize],
        context:&eredu_nn::workspace::WorkspaceContext,
    )->Result<Self,eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(Self,Result<Self,eredu_nn::Error>)>())?;
        let units=crate::ExecutionUnitLayout::new_with_metadata(&graph,counts,context)?;
        Ok(Self{source:PreparedTextGeometrySource::Workspace{graph,units}})
    }
}

impl PreparedReplicatedTextContract {
    /// Consumes the validated contract and retains its same immutable selected
    /// graph and unit layout without cloning names, declarations or topology.
    pub fn into_execution_geometry(self) -> PreparedReplicatedTextExecutionGeometry {
        PreparedReplicatedTextExecutionGeometry {
            source: PreparedTextGeometrySource::Selected(self.selected),
        }
    }
    /// Returns the authoritative selected realization.
    pub const fn selected(&self) -> &SelectedReplicatedTextRealization {
        &self.selected
    }

    /// Returns the validated exact materialization tasks.
    pub fn materialization_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        self.materialization.as_ref()
            .map(PreparedContractMaterialization::tasks)
            .unwrap_or_else(|| self.selected.materialization_tasks())
    }

    /// Exact destinations assigned to independently addressable storage.
    /// Both cold and native binding exclude this same validated population.
    pub fn addressable_parameters(&self) -> &[String] {
        self.materialization.as_ref()
            .map(PreparedContractMaterialization::addressable).unwrap_or(&[])
    }

    /// Successful source-owned task/exclusion result, available after initial
    /// ordinary construction or reuse of an exact retained materialization.
    pub fn materialization_source(&self) -> Option<&PreparedContractMaterialization> {
        self.materialization.as_ref()
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
        // A native consuming constructor still receives owned tasks. The
        // checked quote path consumes into_execution_geometry instead.
        let (tasks, addressable_parameters) = self.materialization
            .map(PreparedContractMaterialization::into_parts)
            .unwrap_or_else(|| (self.selected.materialization_tasks().to_vec(), Vec::new()));
        (
            self.selected,
            tasks,
            addressable_parameters,
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
    prepare_layered_text_contract_impl::<A, B, S>(
        architecture,
        source_architecture,
        selected,
        expected_prompt_cache_architecture_identity,
        output_selection,
        addressable_parameters,
        None,
        context,
        ContractMetadata::new(None),
    )
    .map_err(PreparedTextContractError::into_legacy)
}

/// Uses the same contract worker with this realization's optional host metadata
/// destination. Typed capacity failures remain owned errors without formatting.
pub fn prepare_layered_text_contract_with_metadata<'a, A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    output_selection: ReplicatedTextOutputSelection,
    addressable_parameters: impl IntoIterator<Item = &'a str>,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextContract, PreparedTextContractError>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    prepare_layered_text_contract_impl::<A, B, S>(
        architecture,
        source_architecture,
        selected,
        expected_prompt_cache_architecture_identity,
        output_selection,
        addressable_parameters,
        None,
        context,
        ContractMetadata::new(B::construction_metadata(context)),
    )
}

/// Revalidates the same architecture and companions while reusing a completed
/// exact selected materialization. A changed selection or exclusion set refuses
/// before parameter construction or source work.
pub fn prepare_layered_text_contract_with_materialization<'a, A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    output_selection: ReplicatedTextOutputSelection,
    addressable_parameters: impl IntoIterator<Item = &'a str>,
    materialization: &PreparedContractMaterialization,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<PreparedReplicatedTextContract, PreparedTextContractError>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    prepare_layered_text_contract_impl::<A, B, S>(
        architecture,
        source_architecture,
        selected,
        expected_prompt_cache_architecture_identity,
        output_selection,
        addressable_parameters,
        Some(materialization),
        context,
        ContractMetadata::new(B::construction_metadata(context)),
    )
}

fn prepare_layered_text_contract_impl<'a, A, B, S>(
    architecture: &A,
    source_architecture: Option<&A>,
    selected: SelectedReplicatedTextRealization,
    expected_prompt_cache_architecture_identity: &str,
    output_selection: ReplicatedTextOutputSelection,
    addressable_parameters: impl IntoIterator<Item = &'a str>,
    materialization: Option<&PreparedContractMaterialization>,
    context: &<B::Tensor as Tensor>::Context,
    metadata: ContractMetadata<'_>,
) -> Result<PreparedReplicatedTextContract, PreparedTextContractError>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let mut declarations = addressable_parameters.into_iter();
    metadata.controls::<(Option<&PreparedContractMaterialization>, BTreeSet<String>,
        Option<Vec<String>>, Option<PreparedContractMaterialization>)>()?;
    let mut addressable_parameters = if let Some(source) = materialization {
        source.validate(&selected, declarations, metadata)?;
        BTreeSet::new()
    } else if metadata.is_checked() {
        if declarations.next().is_some() {
            return Err(PreparedTextContractError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Unqualified.into(),
            ));
        }
        BTreeSet::new()
    } else {
        declarations.map(str::to_owned).collect::<BTreeSet<_>>()
    };
    let declared = (materialization.is_none() && !metadata.is_checked())
        .then(|| addressable_parameters.iter().cloned().collect::<Vec<_>>());
    validate_selected_state(&selected, metadata)?;
    validate_architecture_geometry::<A, B, S>(architecture, &selected, metadata)?;
    if let Some(source) = source_architecture {
        validate_architecture_geometry::<A, B, S>(source, &selected, metadata)?;
    }
    let has_transform = selected.parameters().iter().any(|parameter| {
        matches!(
            parameter.lowering(),
            WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
        )
    });
    if has_transform != source_architecture.is_some() {
        return Err(metadata.message(format_args!(
            "selected transform tasks and source-format architecture ownership disagree"
        )));
    }
    if let Some(source) = source_architecture {
        validate_architecture_parameters::<A, B, S>(source, &selected, false, context, metadata)?;
    }
    let constructed_companions = validate_architecture_parameters::<A, B, S>(
        architecture,
        &selected,
        true,
        context,
        metadata,
    )?;
    use std::borrow::Cow;
    metadata.controls::<Cow<'_, [ReplicatedTextMaterializationTask]>>()?;
    if !selected.has_authoritative_materialization_tasks() {
        return Err(metadata.message(format_args!(
            "invalid replicated text contract: selected realization omitted its authoritative materialization tasks"
        )));
    }
    let tasks: Cow<'_, [ReplicatedTextMaterializationTask]> = if metadata.is_checked() || materialization.is_some()
    {
        Cow::Borrowed(selected.materialization_tasks())
    } else {
        Cow::Owned(
            replicated_text_materialization_tasks(&selected).map_err(|error| error.to_string())?,
        )
    };
    // Empty exclusions cannot introduce unknown names or companion exclusions.
    // Avoid rebuilding a selected-name catalog in the ordinary plain case too.
    if !addressable_parameters.is_empty() {
        let selected_parameter_names = tasks
            .iter()
            .flat_map(|task| {
                std::iter::once(task.name().to_owned()).chain(
                    task.output_companions()
                        .iter()
                        .map(|companion| companion.name().to_owned()),
                )
            })
            .collect::<BTreeSet<_>>();
        if !addressable_parameters.is_subset(&selected_parameter_names) {
            return Err(PreparedTextContractError::Contract(format!(
                "addressable parameter catalog contains unknown selected parameters: {:?}",
                addressable_parameters
                    .difference(&selected_parameter_names)
                    .collect::<Vec<_>>()
            )));
        }
        let addressable_companions = addressable_parameters
            .iter()
            .filter_map(|name| tasks.iter().find(|task| task.name() == name))
            .flat_map(|task| task.output_companions())
            .map(|companion| companion.name().to_owned())
            .collect::<Vec<_>>();
        addressable_parameters.extend(addressable_companions);
    }
    for task in tasks.iter() {
        let actual = constructed_companions.for_primary(task.name());
        let expected = task.output_companions();
        let agrees = actual.len() == expected.len()
            && actual.clone().zip(expected).all(|(actual, expected)| {
                actual.name() == expected.name()
                    && actual.role() == expected.role()
                    && actual.logical_shape() == expected.logical_shape()
                    && actual.owner().refines_storage_owner(expected.owner())
            });
        if !agrees {
            return Err(metadata.message(format_args!(
                "constructed output companions for {:?} differ from authoritative selection: lowering={:?}, executable={:?}, constructed={:?}, selected={:?}, retained={:?}",
                task.name(), task.lowering(), task.executable(),
                DebugRows(actual.map(|companion| (companion.name(), companion.role(),
                    companion.logical_shape(), companion.owner()))),
                DebugRows(expected.iter().map(|companion| (companion.name(), companion.role(),
                    companion.logical_shape(), companion.owner()))),
                DebugRows(selected.requirements().parameters().iter().filter_map(|parameter| {
                    parameter.linear_companion().filter(|(_, primary)| *primary == task.name())
                        .map(|(role, primary)| (parameter.name(), role, primary))
                })),
            )));
        }
    }
    let unknown = constructed_companions.unknown_primaries(&tasks);
    if unknown.clone().next().is_some() {
        return Err(metadata.message(format_args!(
            "output companion catalog contains unknown materialization tasks: {:?}",
            DebugRows(unknown),
        )));
    }

    drop(unknown);

    let prompt_cache_identity = if let Some(context) = metadata.context() {
        metadata.controls::<PartitionState>()?;
        let state = PartitionState::new(selected.state().layout().clone_workspace(context)?, 0)
            .map_err(|error| metadata.message(format_args!("{error}")))?;
        // The architecture still supplies its actual validated configuration identity.
        let identity = <A as crate::ArchitectureParameters<B>>::state_identity(
            architecture,
            &state,
            Default::default(),
            Some(context),
        )
        .map_err(|error| {
            metadata.architecture_error(error, "neutral architecture state is invalid: ")
        })?;
        identity.into_prompt_cache_identity_workspace(state.layout(), context)?
    } else {
        let state = PartitionState::new(selected.state().layout().clone(), 0)
            .map_err(|error| error.to_string())?;
        state
            .prompt_cache_identity::<B, A>(architecture, Default::default())
            .map_err(|error| error.to_string())?
    };
    if prompt_cache_identity.architecture_fingerprint()
        != expected_prompt_cache_architecture_identity
        || prompt_cache_identity.layer_count() != selected.state().layout().len()
        || prompt_cache_identity.global_layer_start() != 0
        || prompt_cache_identity.global_layer_end() != selected.state().layout().len()
        || prompt_cache_identity.topology() != &Default::default()
    {
        return Err(metadata.message(format_args!(
            "architecture prompt-cache identity differs from selection"
        )));
    }
    let materialization = match (materialization, tasks) {
        (Some(source), _) => Some(source.clone()),
        (None, Cow::Borrowed(_)) => None,
        (None, Cow::Owned(mut tasks)) => {
            tasks.retain(|task| !addressable_parameters.contains(task.name()));
            Some(PreparedContractMaterialization::new(
                selected.clone(), tasks, declared.unwrap_or_default(),
                addressable_parameters.into_iter().collect(),
            ))
        }
    };
    Ok(PreparedReplicatedTextContract {
        selected, materialization, prompt_cache_identity, output_selection,
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
    let materialization_report = mechanisms
        .take_materialization_report()
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
    let observation_paths = Some(execution.prepare_observation_paths().map_err(
        |error| match error {
            crate::PreparedLayeredObservationError::Execution(error) => {
                ReplicatedTextSessionError::Architecture(error)
            }
            other => ReplicatedTextSessionError::Contract(other.to_string()),
        },
    )?);
    Ok(ReplicatedTextSession {
        selected,
        selected_state,
        execution,
        observation_paths,
        driver,
        state,
        mechanisms,
        materialization_report,
        partition_parameters: None,
        prompt_cache_identity: Some(prompt_cache_identity),
        committed_prompt_input_identity: None,
        next_commit_epoch: DistributedCommitEpoch::FIRST,
        active_commit_epoch: None,
        last_commit_outcome: None,
        successful_state_restorations: Some(0),
        control_identity: std::sync::Arc::new(()),
        prefill_identity: Default::default(),
        inference_guard: None,
        active_prefill_control: None,
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
    mut mechanisms: M,
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
        parameters,
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
    let materialization_report = mechanisms
        .take_materialization_report()
        .map_err(ReplicatedTextSessionError::Mechanism)?;
    let observation_paths = D::prepare_observation_paths(&runtime)
        .map_err(widen_infallible)?;
    Ok(ReplicatedTextSession {
        selected,
        selected_state,
        execution: runtime,
        observation_paths,
        driver,
        state,
        mechanisms,
        materialization_report,
        partition_parameters: Some(parameters),
        prompt_cache_identity,
        committed_prompt_input_identity: None,
        next_commit_epoch: DistributedCommitEpoch::FIRST,
        active_commit_epoch: None,
        last_commit_outcome: None,
        successful_state_restorations: Some(0),
        control_identity: std::sync::Arc::new(()),
        prefill_identity: Default::default(),
        inference_guard: None,
        active_prefill_control: None,
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
    /// Returns the aggregate report captured by the neutral construction
    /// driver after exact materialization preparation completed.
    pub const fn materialization_report(&self) -> Option<&crate::WeightMaterializationReport> {
        self.materialization_report.as_ref()
    }

    /// Adds completed native observations for separately prepared auxiliary
    /// modules. This changes telemetry only; selected requirements and resource
    /// authority remain with the construction driver.
    pub fn record_auxiliary_materialization(&mut self, report: crate::WeightMaterializationReport) {
        self.materialization_report
            .get_or_insert_with(Default::default)
            .merge(report);
    }

    /// Exact materialization contracts for parameters actually selected at construction.
    pub fn parameter_materialization_tasks(&self) -> &[ReplicatedTextMaterializationTask] {
        self.selected.materialization_tasks()
    }

    /// Complete global parameter topology retained by partition construction.
    /// Clones share immutable declarations; no residency unit or native tensor
    /// is acquired. Ordinary construction returns `None`.
    pub fn partition_parameter_description(
        &self,
    ) -> Option<&std::sync::Arc<crate::ArchitectureParameterDescription>> {
        self.partition_parameters.as_ref()
    }

    /// Architecture and executor hook facts retained with partition construction.
    pub fn partition_observation_hooks(&self) -> Option<crate::inspection::ObservationHookSupport> {
        self.partition_parameters
            .as_ref()
            .map(|_| D::observation_hooks(&self.execution))
    }

    /// Describes prepared physical slots without acquiring their residency units.
    pub fn parameter_declarations(&self) -> &[eredu_nn::ParameterMetadata] {
        self.mechanisms.parameter_declarations()
    }

    /// Exact prepared source slots.
    pub fn prepared_parameter_slots(
        &self,
    ) -> &[crate::parameter_operations::PreparedParameterSlot] {
        self.mechanisms.prepared_parameter_slots()
    }

    /// Borrows the statically paired unit-execution strategy for generic telemetry.
    pub const fn execution_strategy(&self) -> &D {
        &self.driver
    }

    /// Traverses loaded parameter slots at a caller-established quiescent boundary.
    /// The visitor must prepare fallible work before publishing replacement handles.
    pub fn visit_loaded_parameters(
        &mut self,
        visitor: &mut dyn eredu_nn::ParameterSlotVisitor<B::Tensor>,
    ) -> bool {
        D::visit_loaded_parameters(&mut self.execution, visitor)
    }

    /// Borrows retained module numerical values at a resolved session boundary. The
    /// traversal does not grant native completion, mutation or submission
    /// authority. Physical storage and future allocation must be priced by the
    /// selected backend; values may still have unknown native backing.
    pub fn visit_retained_values(
        &self,
        visitor: &mut dyn FnMut(&B::Tensor),
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.ensure_commit_resolved()?;
        if self.active_commit_epoch.is_some() {
            return Err(ReplicatedTextSessionError::Contract(
                "cannot inspect retained values during an active transaction".into(),
            ));
        }
        Ok(D::visit_retained_values(&self.execution, visitor))
    }

    /// Performs bounded work while the selected static or unit owner is retained.
    pub fn with_parameter_slots(
        &mut self,
        location: &crate::parameter_operations::PreparedParameterLocation,
        operation: &mut crate::parameter_operations::ParameterSlotOperation<
            '_,
            B::Tensor,
            M::PolicyError,
        >,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, crate::LayerwiseAcquireError<A::Error, M::PolicyError>> {
        D::with_parameter_slots(&mut self.execution, location, operation, context)
    }

    /// Publishes completed replacements after resetting incompatible mutable state.
    pub fn publish_parameter_replacements(
        &mut self,
        values: &BTreeMap<String, B::Tensor>,
        active: bool,
    ) -> Result<bool, M::PolicyError> {
        D::publish_parameter_replacements(&mut self.execution, values, active)
    }

    /// Invalidates native snapshots after a completed parameter publication.
    /// Call only after resetting incompatible mutable state and publishing all slots.
    pub fn invalidate_parameter_snapshots(&mut self) {
        self.control_identity = std::sync::Arc::new(());
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
        self.with_observation_transaction(observer, |session, observer| {
            let pass = tokens
                .shape()
                .last()
                .copied()
                .filter(|length| *length > 1)
                .map_or(ExpertPass::Decode, |_| ExpertPass::Prefill);
            let (output, checkpoint, forward_context) =
                session.execute_with_observer(tokens, mask, pass, context, observer)?;
            session.publish(output, checkpoint, forward_context, context, observer)
        })
    }

    /// Runs an ordinary transaction and retains every sequence logit row for
    /// independent-draft verification. Output publication, all-rank agreement,
    /// state rollback and completion use the same lifecycle as ordinary decoding.
    pub fn sequence_logits<'a>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.with_observation_transaction(&mut crate::NoopObserver, |session, observer| {
            let (output, checkpoint, forward_context) =
                session.execute_input_with_observer(input, pass, context, observer)?;
            session.publish(output, checkpoint, forward_context, context, observer)
        })
    }

    /// Runs the same sequence transaction with an exact caller-owned completion
    /// producer. Completion still precedes the shared agreement, rollback and
    /// commit worker; the callback may retain additional readout roots.
    pub fn sequence_logits_with_completion<'a, F>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        complete: F,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        F: FnOnce(&B::Tensor, &M::State, &<B::Tensor as Tensor>::Context) -> Result<(), M::Error>,
    {
        self.sequence_logits_with_optional_checkpoint(input, pass, context, None, complete)
    }

    /// Uses a caller-prepared checkpoint at the existing checkpoint agreement
    /// phase. The caller must supply the complete copy of this current state;
    /// completion, rollback and commit follow the same sequence transaction.
    pub fn sequence_logits_with_checkpoint_and_completion<'a, F>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        checkpoint: M::StateCheckpoint,
        complete: F,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        F: FnOnce(&B::Tensor, &M::State, &<B::Tensor as Tensor>::Context) -> Result<(), M::Error>,
    {
        self.sequence_logits_with_optional_checkpoint(
            input,
            pass,
            context,
            Some(checkpoint),
            complete,
        )
    }

    fn sequence_logits_with_optional_checkpoint<'a, F>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        checkpoint: Option<M::StateCheckpoint>,
        complete: F,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        F: FnOnce(&B::Tensor, &M::State, &<B::Tensor as Tensor>::Context) -> Result<(), M::Error>,
    {
        self.output_with_optional_checkpoint_and_completion(
            input,
            pass,
            eredu_core::OutputDemand::Sequence,
            context,
            checkpoint,
            |output, state, context| {
                complete(output.expect("sequence output checked"), state, context)
            },
            || {
                ReplicatedTextSessionError::Contract(
                    "score-producing session received state-only output".into(),
                )
            },
        )
        .map(|output| output.expect("sequence publication preserves output"))
    }

    /// Runs one selected prefill span with its exact readout demand and a
    /// caller-prepared rollback copy. State-only draft spans never construct
    /// vocabulary output. The same publication/completion/agreement worker
    /// commits the span only after the supplied exact completion succeeds.
    pub fn prefill_span_with_checkpoint_and_completion<'a, F>(
        &mut self,
        input: A::Input<'a>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        checkpoint: M::StateCheckpoint,
        funding: &eredu_nn::workspace::HostMetadataFunding,
        complete: F,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        M::Error: From<eredu_nn::workspace::HostMetadataFundingError>,
        F: FnOnce(
            Option<&B::Tensor>,
            &M::State,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), M::Error>,
    {
        let controls = [
            std::mem::size_of::<F>(),
            std::mem::size_of::<Option<M::StateCheckpoint>>(),
            std::mem::size_of::<Option<B::Tensor>>(),
            std::mem::size_of::<Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>>(),
            std::mem::size_of::<Result<(Option<B::Tensor>, M::StateCheckpoint, A::ForwardContext), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>>(),
            std::mem::size_of::<(&mut Self, A::Input<'a>, eredu_core::OutputDemand, &<B::Tensor as Tensor>::Context, M::StateCheckpoint, &eredu_nn::workspace::HostMetadataFunding, F)>(),
        ];
        let bytes = controls.into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| ReplicatedTextSessionError::Mechanism(
                eredu_nn::workspace::HostMetadataFundingError::Overflow.into(),
            ))?;
        funding.reserve_metadata(bytes).map_err(|cause| ReplicatedTextSessionError::Mechanism(cause.into()))?;
        self.output_with_optional_checkpoint_and_completion(
            input,
            ExpertPass::Prefill,
            demand,
            context,
            Some(checkpoint),
            complete,
            || {
                ReplicatedTextSessionError::WorkingMemory(
                    crate::working_memory::WorkingMemoryError::IdentityMismatch,
                )
            },
        )
    }

    fn output_with_optional_checkpoint_and_completion<'a, F, E>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        checkpoint: Option<M::StateCheckpoint>,
        complete: F,
        output_error: E,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        F: FnOnce(
            Option<&B::Tensor>,
            &M::State,
            &<B::Tensor as Tensor>::Context,
        ) -> Result<(), M::Error>,
        E: FnOnce() -> ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    {
        self.with_observation_transaction(&mut crate::NoopObserver, |session, observer| {
            let (output, checkpoint, forward) = session
                .execute_input_result_before_publication_with_readout(
                    Ok(input),
                    pass,
                    context,
                    observer,
                    demand,
                    checkpoint,
                )?;
            if output.is_some() != (demand != eredu_core::OutputDemand::StateOnly) {
                return session.rollback_failure(checkpoint, output_error(), context);
            }
            let (output, checkpoint, forward) = session
                .publish_observed_output_transaction_with_readout(
                    output, checkpoint, forward, context,
                )?;
            let completion = complete(output.as_ref(), &session.state, context)
                .map_err(ReplicatedTextSessionError::Mechanism);
            session.finish_publication(output, checkpoint, forward, context, observer, completion)
        })
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
        self.prefill_input_prediction_target_observed(input, context, &mut crate::NoopObserver)
    }

    /// Runs target prefill with internal hooks and the ordinary observation
    /// transaction, retaining the exact target capture from that same forward.
    pub fn prefill_input_prediction_target_observed<'a, O>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, B::Tensor),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.with_observation_transaction(observer, |session, observer| {
            session.prediction_target_input_inner(input, ExpertPass::Prefill, context, observer)
        })
    }

    fn prediction_target_input_inner<'a, O>(
        &mut self,
        input: A::Input<'a>,
        pass: ExpertPass,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, B::Tensor),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint, forward_context) =
            self.execute_input_before_publication(input, pass, context, observer)?;
        let capture = match self.prepare_prediction_target_capture(&forward_context, context) {
            Ok(capture) => capture,
            Err(error) => return self.rollback_failure(checkpoint, error, context),
        };
        let (output, checkpoint, forward_context) =
            self.publish_observed_output_transaction(output, checkpoint, forward_context, context)?;
        self.publish(output, checkpoint, forward_context, context, observer)
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
        self.with_observation_transaction(observer, |session, observer| {
            if D::PARTITIONED_SESSION {
                return Err(ReplicatedTextSessionError::Contract(
                "partitioned prediction capture requires a selected bundle publication contract"
                    .into(),
            ));
            }
            let (output, checkpoint, forward_context) = session.execute_input_before_publication(
                input,
                ExpertPass::Prefill,
                context,
                observer,
            )?;
            let captured = match capture(&forward_context) {
                Ok(captured) => captured,
                Err(error) => {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::Architecture(error),
                        context,
                    );
                }
            };
            let (output, checkpoint, forward_context) = session
                .publish_observed_output_transaction(
                    output,
                    checkpoint,
                    forward_context,
                    context,
                )?;
            session
                .publish(output, checkpoint, forward_context, context, observer)
                .map(|output| (output, captured))
        })
    }

    /// Runs composite prompt processing and commits its cache-relevant input identity only after
    /// successful state publication and exact completion.
    /// This legacy owned-value adapter may allocate shared ownership metadata;
    /// already shared inputs use [`Self::prefill_input_with_shared_cache_identity`].
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
    /// This legacy adapter may allocate; use the shared variant to preserve an
    /// existing allocation owner and its custody.
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
        self.prefill_input_with_observer_and_shared_cache_identity(
            input,
            SharedPreparedInputCacheIdentity::new(identity),
            context,
            observer,
        )
    }

    /// Commits an existing shared input identity without copying its metadata.
    pub fn prefill_input_with_shared_cache_identity<'a>(
        &mut self,
        input: A::Input<'a>,
        identity: SharedPreparedInputCacheIdentity,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.prefill_input_with_observer_and_shared_cache_identity(
            input,
            identity,
            context,
            &mut crate::NoopObserver,
        )
    }

    /// Observes prompt processing while preserving the actual input-identity owner.
    pub fn prefill_input_with_observer_and_shared_cache_identity<'a, O>(
        &mut self,
        input: A::Input<'a>,
        identity: SharedPreparedInputCacheIdentity,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_input_result_transaction(Ok(input), Some(identity), context, observer, false)
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
        self.prefill_input_result_transaction(Ok(input), None, context, observer, false)
    }

    /// Agrees local input preparation before observation admission or model work.
    ///
    /// Partitioned execution requires selected bounded all-rank phase agreement,
    /// even when this rank's input succeeded and its observer is absent.
    /// Callers retain local preparation failures in `input` instead of returning
    /// early. Every rank must enter the same step, including ranks without an
    /// observer. Failed preparation consumes the transaction epoch but preserves
    /// installed state and the previously committed prompt identity. Local native
    /// work remains subject to the backend's ordinary completion/recovery owner.
    /// Wrapping a supplied raw identity may allocate; shared sources use
    /// [`Self::prefill_input_result_with_shared_identity`] directly.
    pub fn prefill_input_result_with_observer<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        identity: Option<PreparedInputCacheIdentity>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_input_result_with_shared_identity(
            input,
            identity.map(SharedPreparedInputCacheIdentity::new),
            context,
            observer,
        )
    }

    /// Agrees prepared input and commits its shared identity after exact completion.
    /// No identity payload is copied or newly wrapped by this entrypoint.
    pub fn prefill_input_result_with_shared_identity<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_input_result_transaction(input, identity, context, observer, true)
    }

    fn prefill_input_result_transaction<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        require_input_agreement: bool,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let output = self.with_observation_transaction(observer, |session, observer| {
            session.require_input_result_agreement(require_input_agreement)?;
            let (output, checkpoint, forward_context) = session
                .execute_input_result_with_observer(
                    input,
                    ExpertPass::Prefill,
                    context,
                    observer,
                )?;
            let sequence_index = session.output_selection.sequence_index();
            let output = match session
                .mechanisms
                .index_text_output(output, sequence_index, context)
            {
                Ok(output) => output,
                Err(error) => {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::Mechanism(error),
                        context,
                    );
                }
            };
            session.publish(output, checkpoint, forward_context, context, observer)
        })?;
        if let Some(identity) = identity {
            self.committed_prompt_input_identity = Some(identity);
        }
        Ok(output)
    }

    /// Executes one prepared decoder span with explicit readout demand and exact
    /// completion, using the ordinary session transaction. The caller owns
    /// semantic input preparation and request reservation; this method neither
    /// slices media nor schedules another chunk. Returned score axes are retained.
    /// A sequence observer can promote demand on every participating rank.
    pub fn prefill_input_with_readout<'a, O>(
        &mut self,
        input: A::Input<'a>,
        identity: Option<PreparedInputCacheIdentity>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_input_result_with_readout(Ok(input), identity, demand, context, observer)
    }

    /// Propagates a local span-preparation failure to participating ranks before
    /// any state mutation, with the same completion and readout contract.
    /// This raw-identity adapter may allocate shared ownership metadata. It is
    /// not an allocation-bound proof for a finite request.
    pub fn prefill_input_result_with_readout<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        identity: Option<PreparedInputCacheIdentity>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_input_result_with_shared_identity_and_readout(
            input,
            identity.map(SharedPreparedInputCacheIdentity::new),
            demand,
            context,
            observer,
        )
    }

    /// Executes a decoder span with explicit readout and an existing shared
    /// prompt-identity owner. The owner is installed only on successful commit.
    pub fn prefill_input_with_shared_identity_and_readout<'a, O>(
        &mut self,
        input: A::Input<'a>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_input_result_with_shared_identity_and_readout(
            Ok(input),
            identity,
            demand,
            context,
            observer,
        )
    }

    /// Agrees span preparation before execution, retaining the supplied shared
    /// identity through publication without allocating a replacement payload.
    pub fn prefill_input_result_with_shared_identity_and_readout<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.prefill_typed_input_with_shared_identity_and_readout(
            input.map_err(ReplicatedTextSessionError::Architecture),
            identity,
            demand,
            context,
            observer,
        )
    }

    // Internal preparation failures (including exact retention association) enter
    // the same original input agreement and transaction as architecture errors.
    fn prefill_typed_input_with_shared_identity_and_readout<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        identity: Option<SharedPreparedInputCacheIdentity>,
        demand: eredu_core::OutputDemand,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        if !self.selected.exact_completion_available() {
            return Err(ReplicatedTextSessionError::Contract(
                "bounded prefill requires exact state completion".into(),
            ));
        }
        let output = self.with_observation_transaction(observer, |session, observer| {
            session.require_input_result_agreement(true)?;
            let (output, checkpoint, forward) = session
                .execute_input_result_before_publication_with_readout(
                    input,
                    ExpertPass::Prefill,
                    context,
                    observer,
                    demand,
                    None,
                )?;
            let (output, checkpoint, forward) = session
                .publish_observed_output_transaction_with_readout(
                    output, checkpoint, forward, context,
                )?;
            session.publish_with_readout(output, checkpoint, forward, context, observer, true)
        })?;
        if let Some(identity) = identity {
            self.committed_prompt_input_identity = Some(identity);
        }
        Ok(output)
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
        self.with_observation_transaction(observer, |session, observer| {
            if D::PARTITIONED_SESSION {
                return Err(ReplicatedTextSessionError::Contract(
                "partitioned prediction capture requires a selected bundle publication contract"
                    .into(),
            ));
            }
            let (output, checkpoint, forward_context) = session.execute_input_before_publication(
                input,
                ExpertPass::Decode,
                context,
                observer,
            )?;
            let captured = match capture(&forward_context) {
                Ok(captured) => captured,
                Err(error) => {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::Architecture(error),
                        context,
                    );
                }
            };
            let (output, checkpoint, forward_context) = session
                .publish_observed_output_transaction(
                    output,
                    checkpoint,
                    forward_context,
                    context,
                )?;
            session
                .publish(output, checkpoint, forward_context, context, observer)
                .map(|output| (output, captured))
        })
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
        self.decode_input_result_transaction(Ok(input), context, observer, false)
    }

    /// Runs decode after agreeing each rank's local input preparation result.
    /// See [`Self::prefill_input_result_with_observer`] for failure ownership.
    pub fn decode_input_result_with_observer<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.decode_input_result_transaction(input, context, observer, true)
    }

    fn decode_input_result_transaction<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        require_input_agreement: bool,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.with_observation_transaction(observer, |session, observer| {
            session.require_input_result_agreement(require_input_agreement)?;
            let (output, checkpoint, forward_context) = session
                .execute_input_result_with_observer(input, ExpertPass::Decode, context, observer)?;
            let sequence_index = session.output_selection.sequence_index();
            let output = match session
                .mechanisms
                .index_text_output(output, sequence_index, context)
            {
                Ok(output) => output,
                Err(error) => {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::Mechanism(error),
                        context,
                    );
                }
            };
            session.publish(output, checkpoint, forward_context, context, observer)
        })
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
        self.decode_input_prediction_target_observed(input, context, &mut crate::NoopObserver)
    }

    /// Runs target verification/replay with internal hooks through the same
    /// observation, capture, publication and rollback transaction as prefill.
    pub fn decode_input_prediction_target_observed<'a, O>(
        &mut self,
        input: A::Input<'a>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, B::Tensor),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.with_observation_transaction(observer, |session, observer| {
            session.prediction_target_input_inner(input, ExpertPass::Decode, context, observer)
        })
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
        self.with_observation_transaction(observer, |session, observer| {
            let (output, checkpoint, forward_context) = session
                .execute_input_result_with_observer(
                    Ok(A::text_input(tokens, None)),
                    ExpertPass::Decode,
                    context,
                    observer,
                )?;
            let sequence_index = session.output_selection.sequence_index();
            let output = match session
                .mechanisms
                .index_text_output(output, sequence_index, context)
            {
                Ok(output) => output,
                Err(error) => {
                    return session.rollback_failure(
                        checkpoint,
                        ReplicatedTextSessionError::Mechanism(error),
                        context,
                    );
                }
            };
            session.publish(output, checkpoint, forward_context, context, observer)
        })
    }

    /// Snapshots successful state-restoration evidence for one execution call.
    ///
    /// Snapshots of this counter prove only a successful neutral state restore,
    /// never completion of backend work. Overflow permanently disables the
    /// witness; checkpoint restoration never rewinds it.
    pub const fn successful_state_restoration_generation(&self) -> Option<u64> {
        self.successful_state_restorations
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
    /// The returned state is the previously installed target state. This is a
    /// persistent replacement and invalidates prior bindings on both states.
    /// Scoped lane execution uses `with_prediction_target_state` instead.
    /// Validation or agreement failure leaves the canonical state untouched.
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
        let agreed = self
            .agree_execution_phase(phase, validation.is_ok(), context)
            .map_err(widen_infallible)?;
        match validation {
            Ok(()) if agreed => {
                crate::working_memory::exchange_inference_state(&mut self.state, replacement);
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
        crate::working_memory::exchange_inference_state(&mut self.state, replacement);
        Ok(())
    }

    /// Forks one prediction-lane target state from the exact canonical state.
    ///
    /// This preserves any prompt-cache restoration already installed in the ordinary target.
    /// Every rank realizes, restores, and validates its local fork before any caller may retain
    /// the lane; failure leaves the canonical state untouched.
    pub fn prepare_prediction_target_state(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<M::State, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.prepare_prediction_target_state_with(
            context,
            |mechanisms, source, selected| {
                let mut state=mechanisms.fork_prediction_target_state(source,selected,context)
                    .map_err(ReplicatedTextSessionError::Mechanism)?;
                state.inherit_inference_retention(source);
                Ok(state)
            },
            |state|state,
            |error|error,
            |cause|match cause {
                PredictionTargetPreparationError::Boundary(cause)=>cause.into_legacy(),
                cause=>ReplicatedTextSessionError::Contract(cause.to_string()),
            },
        )
    }

    /// Shared state-preparation mutation boundary for an actual paid destination.
    /// The callback may copy native state under its separately admitted copy
    /// authority. Unlike read-only inspection, this is the existing preparation
    /// transaction: commit resolution, local copy/layout check and rank agreement
    /// complete before any destination is returned. The callback must preserve
    /// the exact inherited storage/account custody required by its state type.
    pub fn prepare_prediction_target_state_with<T,E>(
        &mut self,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        prepare: impl FnOnce(&mut M,&M::State,&SelectedStateRealization)->Result<T,E>,
        state: impl FnOnce(&T)->&M::State,
        session_error: impl Fn(ReplicatedTextSessionError<A::Error,M::PolicyError,M::Error>)->E,
        contract_error: impl Fn(PredictionTargetPreparationError)->E,
    )->Result<T,E> {
        RuntimeInspectionBoundary::resolved(self.control_fence,self.last_commit_outcome)
            .map_err(|cause|contract_error(PredictionTargetPreparationError::Boundary(cause)))?;
        let provisional=match self.selected_state.state() {
            None=>Err(contract_error(PredictionTargetPreparationError::Stateless)),
            Some(selected)=>prepare(&mut self.mechanisms,&self.state,selected).and_then(|prepared| {
                if !realized_state_layout_matches::<B,M::State>(state(&prepared),selected) {
                    return Err(contract_error(PredictionTargetPreparationError::Layout));
                }
                Ok(prepared)
            }),
        };
        let phase=crate::DistributedExecutionPhase::PredictionTargetStatePreparation;
        let agreed=self.agree_execution_phase(phase,provisional.is_ok(),context)
            .map_err(|cause|session_error(widen_infallible(cause)))?;
        match provisional {
            Ok(prepared) if agreed=>Ok(prepared),
            Ok(_)=>Err(contract_error(PredictionTargetPreparationError::Peer)),
            Err(cause)=>Err(cause),
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
        let checkpoint = self.checkpoint_observed_state(context)?;
        let execution = {
            // The typed operation may replace the complete state, or unwind
            // after doing so. Preserve the original backing owners and invalidate
            // the final revision before another cold inspection is possible.
            struct Mutating<'a, S: InferenceStateRetention> {
                state: &'a mut S,
                retained: crate::working_memory::InferenceRetention,
            }
            impl<S: InferenceStateRetention> Drop for Mutating<'_, S> {
                fn drop(&mut self) {
                    let retention = self.state.inference_retention_mut();
                    retention.extend_from(&self.retained);
                    retention.invalidate_revision();
                }
            }
            let retained = self.state.inference_retention().clone();
            self.state.inference_retention_mut().invalidate_revision();
            let mutating = Mutating {
                state: &mut self.state,
                retained,
            };
            D::apply_prediction_target_operation(
                &mut self.execution,
                &mut *mutating.state,
                operation,
                context,
            )
        }
        .map_err(widen_infallible)
        .and_then(|output| {
            output.ok_or_else(|| {
                ReplicatedTextSessionError::Contract(
                    "selected target execution has no typed prediction-extension handoff".into(),
                )
            })
        });
        let phase = crate::DistributedExecutionPhase::PredictionExtensionExecution;
        let agreed = self
            .agree_execution_phase(phase, execution.is_ok(), context)
            .map_err(widen_infallible);
        match (execution, agreed) {
            (Ok(output), Ok(true)) => Ok(output),
            (execution, agreement) => {
                let error = match (execution, agreement) {
                    (Err(error), _) | (_, Err(error)) => error,
                    (Ok(_), Ok(false)) => ReplicatedTextSessionError::Contract(
                        "another rank failed during prediction-extension execution".into(),
                    ),
                    (Ok(_), Ok(true)) => unreachable!("successful extension returned above"),
                };
                if self.control_fence.is_some() {
                    return Err(error);
                }
                self.mechanisms
                    .restore_state(&mut self.state, checkpoint, context)
                    .map_err(ReplicatedTextSessionError::Mechanism)?;
                Err(error)
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
                self.control_fence.get_or_insert(phase);
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
        self.next_commit_epoch = self.next_commit_epoch.max(checkpoint.next_commit_epoch);
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
                self.control_fence.get_or_insert(phase);
                return Err(error);
            }
        }
        self.state.inference_retention_mut().invalidate_revision();
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
        self.state.inference_retention_mut().invalidate_revision();
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
        self.state.inference_retention_mut().invalidate_revision();
        self.committed_prompt_input_identity = None;
        self.restore_distributed_commit(manifest.distributed_commit)?;
        Ok(manifest)
    }

    /// Opens a prompt cache only when its content identity matches the admitted prepared input.
    /// A shared identity retains its existing payload/custody; a raw identity is
    /// moved into one shared owner before validation.
    pub fn load_prompt_cache_for_input(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: impl Into<SharedPreparedInputCacheIdentity>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<PromptCacheManifest, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        let input_identity = input_identity.into();
        self.validate_prompt_input_descriptor(expected, input_identity.as_ref())?;
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
    /// An existing shared identity is installed without copying its payload.
    pub fn load_prompt_cache_for_input_distributed(
        &mut self,
        directory: &Path,
        expected: &PromptCacheDescriptor,
        prefix_token_ids: &[u32],
        input_identity: impl Into<SharedPreparedInputCacheIdentity>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.load_prompt_cache_distributed_inner(
            directory,
            expected,
            prefix_token_ids,
            Some(input_identity.into()),
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
        if self.committed_prompt_input_identity() != Some(input_identity) {
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
        input_identity: Option<SharedPreparedInputCacheIdentity>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        Option<PromptCacheManifest>,
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        self.ensure_commit_resolved()?;
        self.require_cache_control_agreement()?;
        let preflight = (|| {
            if let Some(input_identity) = input_identity.as_ref() {
                self.validate_prompt_input_descriptor(expected, input_identity.as_ref())?;
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
                self.control_fence.get_or_insert(phase);
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
                self.control_fence.get_or_insert(phase);
                return Err(error);
            }
        };
        let manifest = provisional.map(|(state, manifest)| {
            self.state = state;
            self.state.inference_retention_mut().invalidate_revision();
            self.committed_prompt_input_identity = input_identity;
            self.active_commit_epoch = None;
            self.last_commit_outcome = manifest.distributed_commit;
            if let Some(outcome) = manifest.distributed_commit {
                self.next_commit_epoch =
                    self.next_commit_epoch
                        .max(outcome.epoch().next().unwrap_or_else(|| {
                            unreachable!("provisional commit epoch was validated before agreement")
                        }));
            }
            manifest
        });
        if manifest.is_none() {
            self.state.inference_retention_mut().invalidate_revision();
        }
        Ok(manifest)
    }

    fn restore_distributed_state(
        &mut self,
        checkpoint: Option<M::StateCheckpoint>,
        metadata: Option<(
            Option<SharedPreparedInputCacheIdentity>,
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
                    .is_some_and(|expected| expected <= *next)
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
                self.control_fence.get_or_insert(phase);
                return Err(error);
            }
        };
        if !metadata_valid {
            self.control_fence.get_or_insert(phase);
            return Err(ReplicatedTextSessionError::Contract(
                "distributed checkpoint commit metadata is inconsistent".into(),
            ));
        }
        if let Some(state) = provisional {
            self.state = state;
        }
        self.state.inference_retention_mut().invalidate_revision();
        match metadata {
            Some((identity, next, outcome)) => {
                self.committed_prompt_input_identity = identity;
                self.next_commit_epoch = self.next_commit_epoch.max(next);
                self.last_commit_outcome = outcome;
                self.active_commit_epoch = None;
            }
            None => self.committed_prompt_input_identity = None,
        }
        Ok(())
    }

    /// Returns the prepared-input identity associated with the currently committed prompt state.
    pub fn committed_prompt_input_identity(&self) -> Option<&PreparedInputCacheIdentity> {
        self.committed_prompt_input_identity
            .as_ref()
            .map(AsRef::as_ref)
    }

    /// Borrows the actual committed identity owner for shared storage inventory.
    pub fn committed_shared_prompt_input_identity(
        &self,
    ) -> Option<&SharedPreparedInputCacheIdentity> {
        self.committed_prompt_input_identity.as_ref()
    }

    /// Projects the current state through a read-only backend inspection.
    /// An unresolved commit or fenced control transaction cannot be quoted as
    /// a stable starting state. This grants no native completion or submission
    /// authority; inspectors must preserve unknown allocation/completion facts.
    /// A returned view may borrow this exact state. The shared borrow prevents
    /// state replacement; native work still needs its independent submission
    /// authority after the inspector returns.
    pub fn inspect_runtime_state<'source, T>(
        &'source self,
        inspect: impl FnOnce(&'source M::State) -> Result<T, M::Error>,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.inspect_runtime(|_, state| inspect(state))
    }

    /// Inspects retained backend mechanisms and state at the same quiescent
    /// session boundary. This does not grant submission or completion authority;
    /// an inspector must not load, evaluate, poll, or mutate native resources.
    pub fn inspect_runtime<'source, T>(
        &'source self,
        inspect: impl FnOnce(&'source M, &'source M::State) -> Result<T, M::Error>,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.inspect_runtime_execution(|mechanisms, state, _| inspect(mechanisms, state))
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

    fn fence_terminal(&mut self, phase: crate::DistributedExecutionPhase) {
        self.control_fence.get_or_insert(phase);
        D::mark_terminal_failure(&self.execution, phase);
    }

    fn agree_execution_phase(
        &mut self,
        phase: crate::DistributedExecutionPhase,
        local_success: bool,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, M::PolicyError, std::convert::Infallible>>
    {
        // Terminal errors must never enter a recovery vote or native follow-up.
        if self.control_fence.is_some() {
            return Err(ReplicatedTextSessionError::Partition(
                crate::PartitionExecutionError::CommunicationTerminal,
            ));
        }
        let execution = &mut self.execution;
        let event = ParallelControlEvent::Phase(phase);
        let result = if D::parallel_control_operation(execution, event).is_some() {
            self.mechanisms.with_execution_parallel_control(
                event, context,
                |prepared| D::agree_distributed_phase_with_parallel(
                    execution, phase, local_success, context, prepared),
            ).map_err(ReplicatedTextSessionError::ParallelControl).and_then(|result| result)
        } else {
            D::agree_distributed_phase(execution, phase, local_success, context)
        };
        if result.is_err() {
            self.fence_terminal(phase);
        }
        result
    }

    fn finish_terminal_unwind<T>(
        &mut self,
        phase: crate::DistributedExecutionPhase,
        result: std::thread::Result<T>,
    ) -> T {
        match result {
            Ok(value) => value,
            Err(payload) => {
                // All operation loans have unwound before cleanup. The mark and
                // unresolved guard Drop perform no new work or completion wait.
                self.fence_terminal(phase);
                drop(self.inference_guard.take());
                std::panic::resume_unwind(payload);
            }
        }
    }

    fn with_terminal_unwind<T>(
        &mut self,
        phase: crate::DistributedExecutionPhase,
        operation: impl FnOnce(&mut Self) -> T,
    ) -> T {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| operation(self)));
        self.finish_terminal_unwind(phase, result)
    }

    fn with_observation_transaction<O, V>(
        &mut self,
        observer: &mut O,
        operation: impl FnOnce(
            &mut Self,
            &mut O,
        ) -> Result<
            V,
            ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        >,
    ) -> Result<V, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let guard =
            crate::inspection::ObservationTransactionGuard::new(observer, self.next_commit_epoch);
        let result = self.with_terminal_unwind(
            crate::DistributedExecutionPhase::InferenceWorkspace,
            |session| operation(session, guard.observer),
        );
        let result = if let Some(retention) = self.inference_guard.take() {
            match result {
                Err(error) => {
                    // A host error or deadline is not native completion. Drop
                    // transfers the already owned request into recovery without
                    // spinning on a still-pending collective, even when nested.
                    drop(retention);
                    Err(error)
                }
                Ok(output) => match self.with_terminal_unwind(
                    crate::DistributedExecutionPhase::InferenceWorkspace,
                    |session| session.mechanisms.finish_prefill_reservation(retention),
                ) {
                    Ok(()) => Ok(output),
                    Err(error) => {
                        self.fence_terminal(crate::DistributedExecutionPhase::InferenceWorkspace);
                        Err(ReplicatedTextSessionError::Mechanism(error))
                    }
                },
            }
        } else {
            result
        };
        self.with_terminal_unwind(
            crate::DistributedExecutionPhase::ObservationDelivery,
            |_| guard.finish(result.is_ok()),
        );
        result
    }

    fn require_input_result_agreement(
        &mut self,
        required: bool,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if required && D::PARTITIONED_SESSION && !D::DISTRIBUTED_PHASE_AGREEMENT {
            self.begin_commit_epoch()?;
            return self.abort_without_rollback(ReplicatedTextSessionError::BeforeStateMutation(
                Box::new(ReplicatedTextSessionError::Contract(
                    "fallible partitioned input requires bounded all-rank preparation agreement"
                        .into(),
                )),
            ));
        }
        Ok(())
    }

    fn checkpoint_observed_state(
        &mut self,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<M::StateCheckpoint, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.checkpoint_observed_state_with_prepared(context, None)
    }

    fn checkpoint_observed_state_with_prepared(
        &mut self,
        context: &<B::Tensor as Tensor>::Context,
        prepared: Option<M::StateCheckpoint>,
    ) -> Result<M::StateCheckpoint, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        let checkpoint = match prepared {
            Some(checkpoint) => Ok(checkpoint),
            None => self.mechanisms.checkpoint_state(&self.state, context),
        };
        let checkpoint_agreed = match self.agree_execution_phase(
            crate::DistributedExecutionPhase::StateCheckpoint,
            checkpoint.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                let error = match checkpoint {
                    Err(prior) => ReplicatedTextSessionError::Mechanism(prior),
                    Ok(_) => widen_infallible(error),
                };
                return self.abort_without_rollback(error);
            }
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
        Ok(checkpoint)
    }

    fn prepare_observation_transaction<O>(
        &mut self,
        observer: &mut O,
        epoch: DistributedCommitEpoch,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        use crate::DistributedExecutionPhase as Phase;
        let result = (|| {
            let active = observer.transactional();
            // All ranks execute the first vote, even if their observer is absent.
            // A mixed configuration must fail before a callback enters a collective.
            let absent = self
                .agree_execution_phase(Phase::ObservationParticipation, !active, context)
                .map_err(widen_infallible)?;
            if absent {
                return Ok(());
            }
            let present = self
                .agree_execution_phase(Phase::ObservationParticipationRequired, active, context)
                .map_err(widen_infallible)?;
            if !present || !active || (D::PARTITIONED_SESSION && !D::DISTRIBUTED_PHASE_AGREEMENT) {
                return Err(ReplicatedTextSessionError::Contract(
                    "transactional observation requires agreeing participation on every rank"
                        .into(),
                ));
            }
            for (phase, local) in [
                (Phase::ObservationPreparation, true),
                (Phase::ObservationCoordination, false),
            ] {
                let prepared = if local && !self.selected.exact_completion_available() {
                    Err(ReplicatedTextSessionError::Contract(
                        "transactional observation requires exact completion support".into(),
                    ))
                } else if local {
                    observer
                        .prepare_transaction(epoch, pass)
                        .map_err(ReplicatedTextSessionError::Architecture)
                } else {
                    observer
                        .coordinate_transaction(epoch)
                        .map_err(ReplicatedTextSessionError::Architecture)
                };
                let agreed = self
                    .agree_execution_phase(phase, prepared.is_ok(), context)
                    .map_err(widen_infallible);
                match (prepared, agreed) {
                    (Err(error), _) | (_, Err(error)) => return Err(error),
                    (Ok(()), Ok(true)) => (),
                    (Ok(()), Ok(false)) => {
                        return Err(ReplicatedTextSessionError::Contract(format!(
                            "another rank rejected observation at {phase:?}"
                        )));
                    }
                }
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(()),
            Err(error) => self.abort_without_rollback(
                ReplicatedTextSessionError::BeforeStateMutation(Box::new(error)),
            ),
        }
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

    fn execute_input_result_with_observer<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<
        (B::Tensor, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint, forward_context) = self.execute_input_result_before_publication(
            input,
            pass,
            context,
            observer,
            eredu_core::OutputDemand::LastPosition,
        )?;
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
        self.execute_input_result_before_publication(
            Ok(input),
            pass,
            context,
            observer,
            eredu_core::OutputDemand::Sequence,
        )
    }

    fn execute_input_result_before_publication<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, A::Error>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        demand: eredu_core::OutputDemand,
    ) -> Result<
        (B::Tensor, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let (output, checkpoint, forward) = self
            .execute_input_result_before_publication_with_readout(
                input.map_err(ReplicatedTextSessionError::Architecture),
                pass,
                context,
                observer,
                demand,
                None,
            )?;
        match output {
            Some(output) => Ok((output, checkpoint, forward)),
            None => self.rollback_failure(
                checkpoint,
                ReplicatedTextSessionError::Contract(
                    "score-producing session received state-only output".into(),
                ),
                context,
            ),
        }
    }

    fn execute_input_result_before_publication_with_readout<'a, O>(
        &mut self,
        input: Result<A::Input<'a>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        demand: eredu_core::OutputDemand,
        checkpoint: Option<M::StateCheckpoint>,
    ) -> Result<
        (Option<B::Tensor>, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        self.execute_input_operation_before_publication(
            input,
            pass,
            context,
            observer,
            demand,
            checkpoint,
            A::inference_input_shape,
            |driver, runtime, state, paths, input, observer, demand| {
                match paths {
                    Some(paths) => driver.forward_with_prepared_observer(
                        runtime, input, state, pass, context, observer, paths, demand,
                    ),
                    None => driver.forward_with_observer(
                        runtime, input, state, pass, context, observer, demand,
                    ),
                }
                .map_err(widen_infallible)
            },
        )
    }

    fn execute_input_operation_before_publication<O, I, Shape, Execute>(
        &mut self,
        input: Result<I, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
        pass: ExpertPass,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        demand: eredu_core::OutputDemand,
        checkpoint: Option<M::StateCheckpoint>,
        shape: Shape,
        execute: Execute,
    ) -> Result<
        (Option<B::Tensor>, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    >
    where
        Shape: FnOnce(&I) -> Result<Option<[u64; 2]>, A::Error>,
        Execute: FnOnce(
            &mut D,
            &mut D::Runtime,
            &mut M::State,
            Option<&crate::PreparedLayeredObservationPaths>,
            I,
            &mut O,
            eredu_core::OutputDemand,
        ) -> Result<
            (Option<B::Tensor>, A::ForwardContext),
            ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        >,
        O: ActivationObserver<B::Tensor, A::Error> + ?Sized,
    {
        let epoch = self.begin_commit_epoch()?;
        let retention = match self.state.inference_retention().admission() {
            Some(admission) => self
                .mechanisms
                .coordinate_prefill_entry(admission.request().clone(), self.active_prefill_control)
                .map(Some),
            None => Ok(None),
        };
        let input = match retention {
            Ok(retention) => {
                self.inference_guard = retention;
                input
            }
            Err(error) => {
                self.fence_terminal(crate::DistributedExecutionPhase::InputPreparation);
                // Input preparation may already have failed before this attempt.
                // Keep that first cause; neither cause authorizes a false vote.
                return Err(match input {
                    Err(prior) => prior,
                    Ok(_) => ReplicatedTextSessionError::Mechanism(error),
                });
            }
        };
        let prepared_traversal = observer.requires_prepared_traversal();
        let input = input.and_then(|input| {
            if prepared_traversal {
                let paths = self.observation_paths.as_ref().ok_or(
                    ReplicatedTextSessionError::PreparedObservation(
                        PreparedSessionObservationError::Unavailable,
                    ),
                )?;
                D::validate_observation_paths(&self.execution, paths).map_err(widen_infallible)?;
            }
            Ok(input)
        });
        let agreement = self
            .agree_execution_phase(
                crate::DistributedExecutionPhase::InputPreparation,
                input.is_ok(),
                context,
            )
            .map_err(widen_infallible);
        let input = match (input, agreement) {
            (Ok(input), Ok(true)) => input,
            (input, agreement) => {
                let error = match (input, agreement) {
                    (Err(error), _) => error,
                    (_, Err(error)) => error,
                    _ => ReplicatedTextSessionError::Contract(
                        "another rank rejected input preparation".into(),
                    ),
                };
                return self.abort_without_rollback(
                    ReplicatedTextSessionError::BeforeStateMutation(Box::new(error)),
                );
            }
        };
        let selective = self.agree_execution_phase(
            crate::DistributedExecutionPhase::ReadoutSelection,
            demand != eredu_core::OutputDemand::Sequence && !observer.requires_sequence_readout(),
            context,
        );
        let demand = match selective {
            Ok(true) => match self.agree_execution_phase(
                crate::DistributedExecutionPhase::StateOnlyReadout,
                demand == eredu_core::OutputDemand::StateOnly,
                context,
            ) {
                Ok(true) => eredu_core::OutputDemand::StateOnly,
                Ok(false) => eredu_core::OutputDemand::LastPosition,
                Err(error) => {
                    return self.abort_without_rollback(
                        ReplicatedTextSessionError::BeforeStateMutation(Box::new(
                            widen_infallible(error),
                        )),
                    );
                }
            },
            Ok(false) => eredu_core::OutputDemand::Sequence,
            Err(error) => {
                return self.abort_without_rollback(
                    ReplicatedTextSessionError::BeforeStateMutation(Box::new(widen_infallible(
                        error,
                    ))),
                );
            }
        };
        let span_end = match self.state.inference_retention().admission() {
            Some(admission) => shape(&input)
                .map_err(ReplicatedTextSessionError::Architecture)
                .and_then(|shape| {
                    let position = self
                        .mechanisms
                        .prefill_state_frontier(&self.state)
                        .map_err(ReplicatedTextSessionError::Mechanism)?;
                    admission
                        .validate_span(
                            &self.prefill_identity,
                            shape,
                            pass == ExpertPass::Prefill,
                            demand,
                            position,
                        )
                        .map(Some)
                        .map_err(ReplicatedTextSessionError::WorkingMemory)
                }),
            None => Ok(None),
        };
        let workspace_agreed = self
            .agree_execution_phase(
                crate::DistributedExecutionPhase::InferenceWorkspace,
                span_end.is_ok(),
                context,
            )
            .map_err(widen_infallible);
        let span_end = match (span_end, workspace_agreed) {
            (Ok(end), Ok(true)) => end,
            (local, agreement) => {
                let error = match (local, agreement) {
                    (Err(error), _) | (_, Err(error)) => error,
                    _ => ReplicatedTextSessionError::Contract(
                        "another rank rejected inference workspace admission".into(),
                    ),
                };
                return self.abort_without_rollback(
                    ReplicatedTextSessionError::BeforeStateMutation(Box::new(error)),
                );
            }
        };
        self.prepare_observation_transaction(observer, epoch, pass, context)?;
        let checkpoint = self.checkpoint_observed_state_with_prepared(context, checkpoint)?;
        // Text and retained media borrow the same admitted model and control
        // contexts. Their callbacks select only the input equation; partition
        // boundaries, one-use source claims and context restoration stay here.
        let paths = prepared_traversal.then(|| self.observation_paths.as_ref()
            .expect("binding checked before input agreement"));
        let execution = self.mechanisms.with_execution_parallel_control_context(context, |control| {
            let execute_context = |execution: &mut D::Runtime| {
                self.mechanisms.with_execution_parallel(context, |parallel| {
                    let run = |runtime: &mut D::Runtime| {
                        execute(&mut self.driver, runtime, &mut self.state, paths,
                            input, observer, demand)
                    };
                    match parallel {
                        Some((parallel, funding)) => D::with_borrowed_parallel_context(
                            execution, parallel, funding, run,
                        ).map_err(ReplicatedTextSessionError::ParallelContext)?,
                        None => run(execution),
                    }
                }).map_err(ReplicatedTextSessionError::Mechanism)?
            };
            match control {
                Some((control, funding)) => D::with_borrowed_parallel_control_context(
                    &mut self.execution, control, funding, execute_context,
                ).map_err(ReplicatedTextSessionError::ParallelContext)?,
                None => execute_context(&mut self.execution),
            }
        }).map_err(ReplicatedTextSessionError::Mechanism).and_then(|result| result);
        let execution = execution.and_then(
            |(output, forward)| {
                if let Some(expected) = span_end {
                    if let Some(actual) = self
                        .mechanisms
                        .prefill_state_frontier(&self.state)
                        .map_err(ReplicatedTextSessionError::Mechanism)?
                    {
                        if actual != expected {
                            return Err(ReplicatedTextSessionError::WorkingMemory(
                                crate::working_memory::WorkingMemoryError::StateFrontierMismatch {
                                    expected,
                                    actual,
                                },
                            ));
                        }
                    }
                }
                if output.is_some() == (demand != eredu_core::OutputDemand::StateOnly) {
                    Ok((output, forward))
                } else {
                    Err(ReplicatedTextSessionError::Contract(
                        "execution output presence differs from agreed readout demand".into(),
                    ))
                }
            },
        );
        let execution_agreed = match self.agree_execution_phase(
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
            Ok((output, forward)) if execution_agreed => (output, forward),
            Ok(_) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Partition(
                        crate::PartitionExecutionError::RemotePhaseFailure(
                            crate::DistributedExecutionPhase::Execution,
                        ),
                    ),
                    context,
                );
            }
            Err(error) => return self.rollback_failure(checkpoint, error, context),
        };
        if let Some(end) = span_end {
            self.state.inference_retention_mut().commit_span(end);
        } else {
            // Branch identity follows numerical execution even when no request
            // reservation supplies a logical admission frontier.
            self.state.inference_retention_mut().invalidate_revision();
        }
        let observation = output
            .as_ref()
            .map(|output| D::observe_output(&mut self.execution, output, observer, context))
            .transpose();
        let observation_agreed = match self.agree_execution_phase(
            crate::DistributedExecutionPhase::OutputObservation,
            observation.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                return self.rollback_failure(
                    checkpoint,
                    widen_infallible(observation.err().unwrap_or(error)),
                    context,
                );
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
                );
            }
            Err(error) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context);
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
        self.publish_observed_output_transaction_with_readout(
            Some(output),
            checkpoint,
            forward_context,
            context,
        )
        .map(|(output, checkpoint, forward)| {
            (
                output.expect("score publication preserves output"),
                checkpoint,
                forward,
            )
        })
    }

    fn publish_observed_output_transaction_with_readout(
        &mut self,
        output: Option<B::Tensor>,
        checkpoint: M::StateCheckpoint,
        forward_context: A::ForwardContext,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<
        (Option<B::Tensor>, M::StateCheckpoint, A::ForwardContext),
        ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    > {
        let publication = output.map(|output| {
            self.mechanisms.with_execution_parallel_publication(context,|prepared| {
                D::publish_observed_output_with_parallel(&mut self.execution,output,context,prepared)
                    .map_err(widen_infallible)
            }).map_err(ReplicatedTextSessionError::Mechanism).and_then(|result|result)
        }).transpose();
        let publication_agreement = self.agree_execution_phase(
            crate::DistributedExecutionPhase::OutputPublication,
            publication.is_ok(),
            context,
        );
        let output = match (publication, publication_agreement) {
            (Err(error), _) => {
                return self.rollback_failure(checkpoint, error, context);
            }
            (Ok(_), Err(error)) => {
                return self.rollback_failure(checkpoint, widen_infallible(error), context);
            }
            (Ok(output), Ok(true)) => output,
            (Ok(_), Ok(false)) => {
                return self.rollback_failure(
                    checkpoint,
                    ReplicatedTextSessionError::Contract(
                        "another rank failed during distributed output publication".into(),
                    ),
                    context,
                );
            }
        };
        Ok((output, checkpoint, forward_context))
    }

    fn publish<O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        output: B::Tensor,
        checkpoint: M::StateCheckpoint,
        forward_context: A::ForwardContext,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<B::Tensor, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.publish_with_readout(
            Some(output),
            checkpoint,
            forward_context,
            context,
            observer,
            false,
        )
        .map(|output| output.expect("score publication preserves output"))
    }

    fn publish_with_readout<O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        output: Option<B::Tensor>,
        checkpoint: M::StateCheckpoint,
        _forward_context: A::ForwardContext,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
        force_completion: bool,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        self.publish_with_media_roots(
            output,
            checkpoint,
            _forward_context,
            context,
            observer,
            force_completion,
            None,
        )
    }

    fn publish_with_media_roots<O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        output: Option<B::Tensor>,
        checkpoint: M::StateCheckpoint,
        _forward_context: A::ForwardContext,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
        observer: &mut O,
        force_completion: bool,
        roots: Option<&crate::media_prefill::RetainedMediaRoots<'_, B::Tensor>>,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        let completion = if let Some(roots) = roots {
            self.mechanisms
                .complete_media_ingress(output.as_ref(), &self.state, roots, context)
                .ok_or_else(|| {
                    ReplicatedTextSessionError::Contract(
                        "retained media completion is unavailable".into(),
                    )
                })
                .and_then(|result| result.map_err(ReplicatedTextSessionError::Mechanism))
        } else if force_completion
            || self.inference_guard.is_some()
            || self.selected.exact_completion()
            || observer.transactional()
        {
            self.mechanisms
                .complete(output.as_ref(), &self.state, context)
                .map_err(ReplicatedTextSessionError::Mechanism)
        } else {
            Ok(())
        };
        self.finish_publication(
            output,
            checkpoint,
            _forward_context,
            context,
            observer,
            completion,
        )
    }

    fn finish_publication<O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        output: Option<B::Tensor>,
        checkpoint: M::StateCheckpoint,
        _forward_context: A::ForwardContext,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
        completion: Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>,
    ) -> Result<Option<B::Tensor>, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>>
    {
        let completion_agreed = match self.agree_execution_phase(
            crate::DistributedExecutionPhase::MechanismCompletion,
            completion.is_ok(),
            context,
        ) {
            Ok(agreed) => agreed,
            Err(error) => {
                let error = match completion {
                    Err(prior) => prior,
                    Ok(_) => widen_infallible(error),
                };
                return self.rollback_failure(checkpoint, error, context);
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
                );
            }
            Err(error) => {
                return self.rollback_failure(checkpoint, error, context);
            }
        }
        self.commit_observation_transaction(checkpoint, context, observer)?;
        Ok(output)
    }

    fn commit_observation_transaction<O: ActivationObserver<B::Tensor, A::Error> + ?Sized>(
        &mut self,
        checkpoint: M::StateCheckpoint,
        context: &<B::Tensor as Tensor>::Context,
        observer: &mut O,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let epoch = self.active_commit_epoch.ok_or_else(|| {
            ReplicatedTextSessionError::Contract(
                "distributed transaction lost its active commit epoch".into(),
            )
        })?;
        if observer.transactional() {
            let delivered = observer
                .complete_transaction(epoch)
                .map_err(ReplicatedTextSessionError::Architecture);
            let agreed = self
                .agree_execution_phase(
                    crate::DistributedExecutionPhase::ObservationDelivery,
                    delivered.is_ok(),
                    context,
                )
                .map_err(widen_infallible);
            let delivery = match (delivered, agreed) {
                (Err(error), _) | (_, Err(error)) => Err(error),
                (Ok(()), Ok(true)) => Ok(()),
                (Ok(()), Ok(false)) => Err(ReplicatedTextSessionError::Contract(
                    "another rank rejected completed observation delivery".into(),
                )),
            };
            if let Err(error) = delivery {
                return self.rollback_failure(checkpoint, error, context);
            }
        }
        let execution = &mut self.execution;
        let outcome = if D::parallel_control_operation(execution, ParallelControlEvent::Commit).is_some() {
            self.mechanisms.with_execution_parallel_control(
                ParallelControlEvent::Commit, context,
                |prepared| D::commit_after_completion_with_parallel(execution, epoch, context, prepared),
            ).map_err(ReplicatedTextSessionError::ParallelControl)
                .and_then(|result| result.map_err(widen_infallible))
        } else {
            Ok(D::commit_after_completion(execution, epoch, context))
        };
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(error) => {
                self.last_commit_outcome = Some(DistributedCommitOutcome::Indeterminate {
                    epoch, phase: DistributedCommitPhase::DecisionSubmission,
                });
                self.active_commit_epoch = None;
                self.fence_terminal(crate::DistributedExecutionPhase::Commit);
                return Err(error);
            }
        };
        match outcome {
            DistributedCommitOutcome::Committed(committed) if committed == epoch => {
                self.last_commit_outcome = Some(DistributedCommitOutcome::Committed(epoch));
                self.active_commit_epoch = None;
            }
            DistributedCommitOutcome::Aborted(aborted) if aborted == epoch => {
                self.last_commit_outcome = Some(DistributedCommitOutcome::Aborted(epoch));
                self.active_commit_epoch = None;
                let restored = self
                    .mechanisms
                    .restore_state(&mut self.state, checkpoint, context)
                    .map_err(ReplicatedTextSessionError::Mechanism);
                record_successful_restoration(&mut self.successful_state_restorations, restored)?;
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
        Ok(())
    }

    fn rollback_failure<T>(
        &mut self,
        checkpoint: M::StateCheckpoint,
        error: ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if self.control_fence.is_some() {
            // Neither local restoration nor a fabricated Aborted epoch is a
            // valid follow-up to failed terminal communication. Native recovery
            // retains submitted graph ownership while the checkpoint drops.
            return Err(error);
        }
        self.restore_failed_work(checkpoint, context)?;
        Err(error)
    }

    fn restore_failed_work(
        &mut self,
        checkpoint: M::StateCheckpoint,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        let restored = self
            .mechanisms
            .restore_state(&mut self.state, checkpoint, context)
            .map_err(ReplicatedTextSessionError::Mechanism);
        if let Some(epoch) = self.active_commit_epoch.take() {
            self.last_commit_outcome = Some(DistributedCommitOutcome::Aborted(epoch));
        }
        record_successful_restoration(&mut self.successful_state_restorations, restored)?;
        Ok(())
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
        RuntimeInspectionBoundary::resolved(self.control_fence, self.last_commit_outcome)
            .map_err(RuntimeInspectionBoundary::into_legacy)
    }

    fn ensure_control_unfenced(
        &self,
    ) -> Result<(), ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        RuntimeInspectionBoundary::unfenced(self.control_fence)
            .map_err(RuntimeInspectionBoundary::into_legacy)
    }

    fn agree_cache_control_phase(
        &mut self,
        phase: crate::DistributedExecutionPhase,
        local_success: bool,
        context: &<<B as NeuralBackend>::Tensor as Tensor>::Context,
    ) -> Result<bool, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        self.agree_execution_phase(phase, local_success, context)
            .map_err(widen_infallible)
            .inspect_err(|_| {
                self.control_fence.get_or_insert(phase);
            })
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
        self.control_fence.get_or_insert(phase);
        Err(ReplicatedTextSessionError::Contract(format!(
            "another rank failed distributed cache control at {phase:?}"
        )))
    }

    fn abort_without_rollback<T>(
        &mut self,
        error: ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>,
    ) -> Result<T, ReplicatedTextSessionError<A::Error, M::PolicyError, M::Error>> {
        if self.control_fence.is_none() {
            if let Some(epoch) = self.active_commit_epoch.take() {
                self.last_commit_outcome = Some(DistributedCommitOutcome::Aborted(epoch));
            }
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
            self.next_commit_epoch =
                self.next_commit_epoch
                    .max(outcome.epoch().next().ok_or_else(|| {
                        ReplicatedTextSessionError::Contract(
                            "distributed commit epoch overflow".into(),
                        )
                    })?);
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
                if self.committed_prompt_input_identity() != Some(input_identity) {
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
                self.control_fence.get_or_insert(phase);
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
                self.control_fence.get_or_insert(phase);
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
                self.control_fence.get_or_insert(phase);
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
    metadata: ContractMetadata<'_>,
) -> Result<(), PreparedTextContractError>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let requirements = selected.requirements();
    metadata.controls::<(
        crate::ArchitectureExecutionGraph<'_>,
        crate::ArchitectureGroupTransportDeclaration<'_>,
        usize,
        bool,
    )>()?;
    let graph = architecture.execution_graph()
    .map_err(|error| metadata.architecture_error(error, ""))?;
    if !graph.matches(requirements.execution_graph()) {
        return Err(metadata.message(format_args!(
            "architecture execution graph differs from selection"
        )));
    }
    if requirements.execution_units().group_count() != graph.group_count() {
        return Err(metadata.message(format_args!(
            "selected execution-unit groups differ from architecture graph"
        )));
    }
    for group in 0..graph.group_count() {
        let actual = match metadata.context() {
            Some(context) => architecture.group_unit_count(group, Some(context)),
            None => architecture.group_unit_count(group, None),
        }
        .map_err(|error| metadata.architecture_error(error, ""))?;
        let expected = requirements
            .execution_units()
            .group_range(group)
            .expect("validated requirement exposes every graph group")
            .len();
        if actual != expected
            || !architecture.group_transport_matches(group, &requirements.group_transports()[group])
        {
            return Err(metadata.message(format_args!(
                "architecture execution group {group} differs from selection"
            )));
        }
    }
    let layout = architecture.state_layout(metadata.context())
    .map_err(|error| metadata.architecture_error(error, ""))?;
    if &layout != selected.state().layout() {
        return Err(metadata.message(format_args!(
            "architecture state layout differs from selection"
        )));
    }
    Ok(())
}

fn validate_architecture_parameters<'model, A, B, S>(
    architecture: &'model A,
    selected: &SelectedReplicatedTextRealization,
    selected_formats: bool,
    context: &<B::Tensor as Tensor>::Context,
    metadata: ContractMetadata<'_>,
) -> Result<ValidatedParameterCatalog<'model>, PreparedTextContractError>
where
    B: NeuralBackend,
    S: RuntimeState<B>,
    A: LayeredArchitecture<B, S>,
    A::Error: std::fmt::Display,
{
    let requirements = selected.requirements();
    let description = architecture
        .parameter_description(context)
        .map_err(|error| metadata.architecture_error(error, ""))?;
    type Row<'a> = (&'a str, &'a [usize], &'a ParameterGroupOwner);
    type Companion<'a> = (
        (usize, usize),
        Row<'a>,
        eredu_nn::LinearCompanionRole,
        &'a str,
    );
    metadata.controls::<(Vec<Row<'_>>, Vec<Companion<'_>>, Vec<&str>)>()?;
    let rows = description
        .groups()
        .iter()
        .map(|group| group.group().members().len())
        .try_fold(0usize, usize::checked_add)
        .ok_or_else(|| {
            PreparedTextContractError::Metadata(
                eredu_nn::workspace::WorkspaceMetadataError::Overflow.into(),
            )
        })?;
    let mut actual = metadata.vector::<Row<'_>>(rows)?;
    let mut companions = metadata.vector::<Companion<'_>>(rows)?;
    for (group_index, group) in description.groups().iter().enumerate() {
        for (member_index, member) in group.group().members().iter().enumerate() {
            let row = (member.target(), member.global_shape(), group.owner());
            match (member.linear_companion(), member.linear_companion_of()) {
                (None, None) => match actual.binary_search_by_key(&row.0, |value| value.0) {
                    Ok(_) => {
                        return Err(metadata.message(format_args!(
                            "constructed architecture repeats primary parameter {:?}",
                            row.0
                        )));
                    }
                    Err(index) => actual.insert(index, row),
                },
                (Some(role), Some(primary)) => {
                    companions.push(((group_index, member_index), row, role, primary))
                }
                _ => {
                    return Err(metadata.message(format_args!(
                        "constructed parameter {:?} has incomplete linear companion metadata",
                        row.0
                    )));
                }
            }
        }
    }
    let expected_rows = || {
        requirements
            .parameters()
            .iter()
            .filter(|parameter| {
                !matches!(
                    parameter.presence(),
                    ReplicatedTextParameterPresence::OptionalAbsent
                        | ReplicatedTextParameterPresence::Tied { .. }
                ) && parameter.role() != crate::ReplicatedTextParameterRole::FormatCompanion
            })
            .map(|parameter| parameter.name())
    };
    let mut expected = metadata.vector(expected_rows().count())?;
    expected.extend(expected_rows());
    expected.sort_unstable();
    expected.dedup();
    for &(_, row, _, _) in &companions {
        if expected.binary_search(&row.0).is_ok() {
            match actual.binary_search_by_key(&row.0, |value| value.0) {
                Ok(_) => {
                    return Err(metadata.message(format_args!(
                        "constructed architecture repeats selected parameter {:?}",
                        row.0
                    )));
                }
                Err(index) => actual.insert(index, row),
            }
        }
    }
    if !expected.iter().copied().eq(actual.iter().map(|row| row.0)) {
        return Err(metadata.message(format_args!(
            "selected parameter catalog differs from constructed architecture: missing {:?}, unexpected {:?}",
            DebugRows(expected.iter().copied().filter(|name|
                actual.binary_search_by_key(name, |row| row.0).is_err())),
            DebugRows(actual.iter().map(|row| row.0).filter(|name|
                expected.binary_search(name).is_err())),
        )));
    }
    for parameter in requirements.parameters().iter().filter(|parameter| {
        !matches!(
            parameter.presence(),
            ReplicatedTextParameterPresence::OptionalAbsent
                | ReplicatedTextParameterPresence::Tied { .. }
        ) && parameter.role() != crate::ReplicatedTextParameterRole::FormatCompanion
    }) {
        let (_, shape, owner) = actual[actual
            .binary_search_by_key(&parameter.name(), |row| row.0)
            .expect("equal parameter-name sets contain every requirement")];
        let expected_owner = parameter
            .owner()
            .parameter_group_owner()
            .map_err(|error| metadata.message(format_args!("{error}")))?;
        let owner_matches = owner.refines_storage_owner(&expected_owner);
        let mut expected_shape = parameter.logical_shape();
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
                matches!(
                    realization.lowering(),
                    crate::WeightLoweringKind::Direct | crate::WeightLoweringKind::Derived
                )
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
                .expect("direct native realization has admitted physical geometry");
        }
        // Architecture descriptions remain semantic and may therefore expose
        // the logical module geometry. Backends whose constructed module owns
        // an already-packed transform expose the exact selected executable
        // geometry instead. Keep both identities distinct and accept only the
        // two shapes derived from this selected task.
        let selected_lowering = realization.map_or(crate::WeightLoweringKind::Direct, |selected| {
            selected.lowering()
        });
        let selected_executable_shape = Some(
            (|| -> Result<Vec<usize>, PreparedTextContractError> {
                let descriptor =
                    parameter
                        .lowering_descriptor_view(executable)
                        .map_err(|error| {
                            metadata
                                .message(format_args!("invalid replicated text contract: {error}"))
                        })?;
                let mut packed = metadata.vector(descriptor.logical_shape().len())?;
                packed.extend_from_slice(descriptor.logical_shape());
                let Some(axis) = descriptor.packed_axis() else {
                    return Ok(packed);
                };
                let bits = match executable {
                    eredu_checkpoint::LinearFormat::Affine(config) => usize::try_from(config.bits)
                        .map_err(|_| {
                            metadata.message(format_args!(
                                "selected parameter {:?} has invalid affine packing bits",
                                parameter.name()
                            ))
                        })?,
                    eredu_checkpoint::LinearFormat::MxFp4 => 4,
                    eredu_checkpoint::LinearFormat::Dense
                    | eredu_checkpoint::LinearFormat::E4M3BlockFp8(_) => return Ok(packed),
                    eredu_checkpoint::LinearFormat::GgufIQuant { ggml_type, .. } => {
                        if matches!(
                            selected_lowering,
                            crate::WeightLoweringKind::Transform
                                | crate::WeightLoweringKind::DerivedTransform
                        ) {
                            return Err(metadata.message(format_args!(
                                "selected parameter {:?} cannot apply a load-time GGUF transform",
                                parameter.name()
                            )));
                        }
                        let (block, bytes) = ggml_type
                            .block_and_bytes()
                            .map_err(|error| metadata.message(format_args!("{error}")))?;
                        let block = usize::try_from(block).map_err(|_| {
                            metadata.message(format_args!("GGUF block width is not representable"))
                        })?;
                        let bytes = usize::try_from(bytes).map_err(|_| {
                            metadata.message(format_args!("GGUF block bytes are not representable"))
                        })?;
                        let packed_bytes = packed[axis].checked_mul(bytes).ok_or_else(|| {
                            metadata.message(format_args!("GGUF executable geometry overflowed"))
                        })?;
                        if !packed_bytes.is_multiple_of(block) {
                            return Err(metadata.message(format_args!(
                            "selected parameter {:?} GGUF executable geometry is not block aligned",
                            parameter.name()
                        )));
                        }
                        packed[axis] = packed_bytes / block;
                        return Ok(packed);
                    }
                };
                let packed_bits = packed[axis].checked_mul(bits).ok_or_else(|| {
                    metadata.message(format_args!(
                        "selected parameter {:?} executable geometry overflowed",
                        parameter.name()
                    ))
                })?;
                if !packed_bits.is_multiple_of(32) {
                    return Err(metadata.message(format_args!(
                    "selected parameter {:?} executable geometry {}x{} bits is not U32 aligned (logical {:?}, physical {:?})",
                    parameter.name(),
                    packed[axis],
                    bits,
                    descriptor.logical_shape(),
                    descriptor.physical_shape()
                )));
                }
                packed[axis] = packed_bits / 32;
                Ok(packed)
            })()?,
        );
        let shape_matches = shape == expected_shape
            || selected_executable_shape
                .as_ref()
                .is_some_and(|selected| shape == selected.as_slice());
        if !shape_matches || !owner_matches {
            return Err(metadata.message(format_args!(
                "selected parameter {:?} expects logical shape {expected_shape:?} or selected executable shape {selected_executable_shape:?} and owner {:?}, constructed shape {shape:?} and owner {owner:?}",
                parameter.name(),
                parameter.owner()
            )));
        }
    }

    let mut output_companions = metadata.vector(if selected_formats {
        companions.len()
    } else {
        0
    })?;
    for (index, (name, shape, _owner), _role, primary) in companions {
        if name == primary || expected.binary_search(&primary).is_err() {
            return Err(metadata.message(format_args!(
                "constructed companion {name:?} names unselected primary {primary:?}"
            )));
        }
        if selected_formats {
            let recipe = requirements.derived_recipes().get(name);
            let output = requirements.derived_recipe_outputs().get(name);
            if recipe.is_some() != output.is_some() {
                return Err(metadata.message(format_args!(
                    "constructed companion {name:?} has incomplete derived metadata"
                )));
            }
            if !ReplicatedTextOutputCompanion::valid_identity(name, shape) {
                return Err(metadata.message(format_args!(
                    "invalid replicated text contract: materialization output companion identity or geometry is invalid"
                )));
            }
            if let Some(parameter) = requirements.parameters().iter().find(|parameter| {
                parameter.name() == name
                    && matches!(
                        parameter.source_encoding(),
                        Some(eredu_checkpoint::SourceTensorEncoding::Gguf { .. })
                    )
            }) {
                if !matches!(parameter.physical_sources(), [_]) {
                    return Err(metadata.message(format_args!(
                        "translated catalog companion {name:?} has ambiguous provenance"
                    )));
                }
            }
            output_companions.push(index);
        }
    }
    drop(actual);
    drop(expected);
    metadata.controls::<ValidatedParameterCatalog<'_>>()?;
    let mut catalog = ValidatedParameterCatalog {
        description,
        companions: output_companions,
    };
    // Sorting indices is allocation-free. Descriptions have unique target names,
    // so an unstable sort preserves the previous primary/role/name ordering.
    let description = &catalog.description;
    catalog.companions.sort_unstable_by_key(|&(group, member)| {
        let member = &description.groups()[group].group().members()[member];
        (
            member
                .linear_companion_of()
                .expect("validated companion primary"),
            member.linear_companion().expect("validated companion role"),
            member.target(),
        )
    });
    Ok(catalog)
}

fn validate_selected_state(
    selected: &SelectedReplicatedTextRealization,
    metadata: ContractMetadata<'_>,
) -> Result<(), PreparedTextContractError> {
    use eredu_core::cache::StateResidencyClass;

    if !selected.topology().is_replicated() {
        return Err(metadata.message(format_args!(
            "selected replicated-text topology is not replicated"
        )));
    }
    if selected.state().layout() != selected.requirements().state_layout()
        || selected.state().access() != selected.requirements().state_access()
    {
        return Err(metadata.message(format_args!(
            "selected state contract differs from architecture requirements"
        )));
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
                metadata.message(format_args!(
                    "selected state omits component {cursor} at layer {layer}"
                ))
            })?;
            if component.layer() != layer || component.component() != expected {
                return Err(metadata.message(format_args!(
                    "selected state component {cursor} differs from layer {layer} requirements"
                )));
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
                return Err(metadata.message(format_args!(
                    "selected state component {cursor} has {:?} placement, expected {expected_placement:?}",
                    component.placement()
                )));
            }
            cursor += 1;
        }
    }
    if cursor != selected.state().components().len() {
        return Err(metadata.message(format_args!(
            "selected state contains components beyond its architecture layout"
        )));
    }
    if !selected.state().checkpoint() || !selected.state().rollback() || !selected.state().reset() {
        return Err(metadata.message(format_args!(
            "selected state omits a required transactional lifecycle facility"
        )));
    }
    if selected.state().prompt_cache() != selected.prompt_cache()
        || selected.state().observation_retention()
            != (selected.session().output_observation()
                || selected.session().activation_inspection())
    {
        return Err(metadata.message(format_args!(
            "selected state lifecycle differs from selected session facilities"
        )));
    }
    if selected.grouped_operations() != selected.requirements().grouped_operations() {
        return Err(metadata.message(format_args!(
            "selected grouped operations differ from architecture requirements"
        )));
    }
    Ok(())
}

fn realized_state_layout_matches<B, S>(state: &S, selected: &SelectedStateRealization) -> bool
where
    B: NeuralBackend,
    S: RuntimeState<B>,
{
    state.optional_layout() == Some(selected.layout())
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
    if !realized_state_layout_matches::<B, S>(state, selected) {
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
        LayerwiseRuntimeError::Submission(error) => ReplicatedTextSessionError::Submission(error),
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
        ReplicatedTextSessionError::WorkingMemory(error) => {
            ReplicatedTextSessionError::WorkingMemory(error)
        }
        ReplicatedTextSessionError::PreparedObservation(error) => {
            ReplicatedTextSessionError::PreparedObservation(error)
        }
        ReplicatedTextSessionError::ParallelContext(error) => {
            ReplicatedTextSessionError::ParallelContext(error)
        }
        ReplicatedTextSessionError::ParallelControl(error) => {
            ReplicatedTextSessionError::ParallelControl(error)
        }
        ReplicatedTextSessionError::Submission(error) => {
            ReplicatedTextSessionError::Submission(error)
        }
        ReplicatedTextSessionError::Contract(error) => ReplicatedTextSessionError::Contract(error),
        ReplicatedTextSessionError::Partition(error) => {
            ReplicatedTextSessionError::Partition(error)
        }
        ReplicatedTextSessionError::Architecture(error) => {
            ReplicatedTextSessionError::Architecture(error)
        }
        ReplicatedTextSessionError::Policy(error) => ReplicatedTextSessionError::Policy(error),
        ReplicatedTextSessionError::Mechanism(error) => match error {},
        ReplicatedTextSessionError::BeforeStateMutation(error) => {
            ReplicatedTextSessionError::BeforeStateMutation(Box::new(widen_infallible(*error)))
        }
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
