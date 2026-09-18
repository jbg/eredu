//! Shared composition of complete ordinary model/sampling/input continuations.
//!
//! The facade adds its semantic pipeline, output cursor and lifecycle checkpoint.
//! This module never reconstructs a tokenizer, replays a prompt or owns a native
//! completion object. Copy mechanisms finish through the backend's existing owner.

use super::{SnapshotBudget, SnapshotReservation};
mod host_copy;
mod resume;
use crate::capture::{
    CaptureCheckpoint, CaptureForkRequest, CaptureSession, InterventionForkRequest,
};
use crate::working_memory::WorkspaceCopyLimits;
use eredu_core::{
    capture::CaptureError,
    execution_control::{
        ExecutionControlError, NativeTextStateBackend, SnapshotEstimate, SnapshotResourceKind,
    },
    BackendFailure, HostPreparationAuthority, ModelRuntime, PendingTextInput,
    TextContinuationBoundary, TextContinuationIdentity, TextSnapshotSource, TokenFilterController,
};
use host_copy::CallbackHostCopy;
pub use host_copy::{PreparedTextHostCopy, PreparedTextHostJournal, TextHostCopyError};
pub use resume::PendingSnapshotResumeRetention;

/// Allocation policy for the complete saved component requested by a hook.
/// A sampling-only hook covers sampler/input; a paired hook also covers decoder
/// state. Neither scope includes controller or facade state, and saved copy
/// custody never authorizes resuming a generation run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SamplingCopyPolicy {
    /// Use actual backend unquoted allocation authority. This never bypasses a
    /// managed domain's exclusion of unquoted work during finite reservations.
    Unquoted,
    /// Require complete authenticated component sources and a checked physical
    /// copy bound before allocating any destination payload.
    Bounded(WorkspaceCopyLimits),
}

/// Native ordinary-generation mechanisms used by the portable snapshot driver.
/// Capture/intervention ownership stays in the shared `CaptureSession`; adapters
/// expose that owner and perform native copying, never duplicate its bookkeeping.
pub trait TextSnapshotBackend: NativeTextStateBackend {
    /// Sampling parameters, penalties/history, adaptive state, exact RNG and
    /// absolute next-prediction position. Copies must have isolated mutable state.
    type SamplingState;

    /// Independently owned, immutable sampler and pending input. This is saved
    /// numerical data, not a runnable sampler or a replayable execution grant.
    /// Duplication goes through `copy_saved_sampling`, not an allocating Clone.
    type SavedSamplingState;

    /// One opaque immutable decoder/sampler/input pair from a checked source.
    /// Its implementation must prevent independently replacing either component
    /// or extracting an installable native slot. This is saved data and custody,
    /// never a live run, prediction permit, or restoration authority. Independent
    /// duplication uses the paired copy hook, not an allocating Clone.
    type SavedTextComponents;

    /// Same backend domain used by the concrete original host-copy provider.
    /// This immutable loan grants neither bytes nor native submission authority.
    fn original_snapshot_host_pool(
        _runtime: &ModelRuntime<Self>,
    ) -> Option<&crate::working_memory::WorkingMemoryPool> {
        None
    }

    /// Allocation-free logical estimates for the exact live decoder, sampler,
    /// and pending input, in that order. This pre-grant query must not allocate
    /// inventories, clone native handles, format diagnostics, reap, or submit
    /// work. Unknown facts or an unavailable immutable source loan return None.
    /// These logical costs neither qualify physical copy fit nor grant bytes.
    /// The default deliberately does not invoke the ordinary estimate hooks.
    fn original_snapshot_estimates(
        _runtime: &ModelRuntime<Self>,
        _sampling: &Self::SamplingState,
        _input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Option<[SnapshotEstimate; 3]> {
        None
    }

    /// Complete native saved-copy preparation storage while borrowing this exact
    /// live sampler/input and installed decoder. This query must allocate no
    /// metadata or diagnostic and mutate no source. Include every temporary
    /// projection/registry/program Vec/map, descriptor shape, fixed call frame,
    /// and error prefix created before the separate native copy admission.
    /// The original host provider admits these bytes before invoking the shared
    /// saved-copy worker and retains custody on escaping native failures.
    /// Pre-grant refusals remain fixed neutral facts; backend error construction
    /// and erasure are permitted only after their enclosing host admission.
    fn original_saved_components_preparation_bytes(
        _runtime: &ModelRuntime<Self>,
        _sampling: &Self::SamplingState,
        _input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Result<Option<u64>, crate::working_memory::WorkingMemoryError> {
        Ok(None)
    }

    /// Same pre-grant estimates with the actual complete generation state. A
    /// backend-owned funded capture checkpoint contributes its fixed destination
    /// here; the shared snapshot budget reserves it before any copy begins.
    fn original_generation_snapshot_estimates(
        runtime: &ModelRuntime<Self>,
        state: &Self::TextGenerationState,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Option<[SnapshotEstimate; 3]> {
        Self::original_snapshot_estimates(runtime, Self::sampling_state(state), input)
    }

    /// Complete original preparation for the actual generation source, including
    /// any backend-owned funded capture checkpoint. The default preserves the
    /// sampling-only contract and does not infer a capture source from custody.
    fn original_saved_generation_preparation_bytes(
        runtime: &ModelRuntime<Self>,
        state: &Self::TextGenerationState,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Result<Option<u64>, crate::working_memory::WorkingMemoryError> {
        Self::original_saved_components_preparation_bytes(
            runtime,
            Self::sampling_state(state),
            input,
        )
    }

    /// Allocation-free constructor contribution for a fresh original run from
    /// this exact immutable pair. Include cold planning, native preparation and
    /// escaped failure controls not covered by the new execution account. H is
    /// accepted independently before calling the original resume admission hook.
    fn original_saved_components_resume_preparation_bytes<C: TokenFilterController>(
        _runtime: &ModelRuntime<Self>,
        _saved: &Self::SavedTextComponents,
        _config: eredu_core::TextGenerationConfig,
        _controller: &C,
        _options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Result<Option<u64>, crate::working_memory::WorkingMemoryError> {
        Ok(None)
    }

    /// Allocation-free logical destination copy estimate for that same fresh
    /// resume. Future execution is admitted separately; unknown stays unknown.
    fn original_saved_components_resume_estimate(
        _runtime: &ModelRuntime<Self>,
        _saved: &Self::SavedTextComponents,
        _config: eredu_core::TextGenerationConfig,
        _options: &eredu_core::OriginalTextResumeOptions<'_>,
    ) -> Option<SnapshotEstimate> {
        None
    }

    /// Captures decoder, sampler and pending input as one checked pair without
    /// advancing or invalidating the borrowed live source. Bounded implementations
    /// must admit every component before any destination allocation; an incomplete
    /// paired route must reject, never fall back to separate unquoted workers.
    /// The caller's host authority still covers enclosing controller/facade work.
    fn capture_saved_components(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Self::SamplingState,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: SamplingCopyPolicy,
    ) -> Result<Self::SavedTextComponents, Self::Error>;

    /// Same paired copy worker with the enclosing accepted host preparation
    /// lifetime. Source-carrier metadata that may survive through recovery must
    /// retain this authority. It grants no numerical copy or new request; those
    /// still require the separate bounded copy admission.
    fn capture_saved_components_with_host(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Self::SamplingState,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: SamplingCopyPolicy,
        _host: &HostPreparationAuthority,
    ) -> Result<Self::SavedTextComponents, Self::Error> {
        Self::capture_saved_components(runtime, sampling, input, policy)
    }

    /// Same saved-pair transaction, borrowing the complete source while copying
    /// its funded capture checkpoint. An implementation must derive that source
    /// from this state, validate its exact frontier, and retain its original
    /// cumulative authority. Host custody alone cannot qualify a capture source.
    fn capture_saved_generation_components_with_host(
        runtime: &mut ModelRuntime<Self>,
        state: &Self::TextGenerationState,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: SamplingCopyPolicy,
        host: &HostPreparationAuthority,
    ) -> Result<Self::SavedTextComponents, Self::Error> {
        Self::capture_saved_components_with_host(
            runtime,
            Self::sampling_state(state),
            input,
            policy,
            host,
        )
    }

    /// Independently duplicates one immutable saved pair. Its source provenance
    /// is the saved owner, not the original live frontier or request lifetime.
    /// Bounded copying requires a complete paired route before either part copies.
    fn copy_saved_components(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        policy: SamplingCopyPolicy,
    ) -> Result<Self::SavedTextComponents, Self::Error>;

    /// Borrows immutable sampling/input diagnostics from the exact saved pair.
    /// This gives no mutable extraction, native installation or run permission.
    fn saved_sampling(saved: &Self::SavedTextComponents) -> &Self::SavedSamplingState;

    /// Checks the saved pair's exact source/domain compatibility without copying,
    /// allocating native data or changing either the saved or installed state.
    fn validate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<(), Self::Error>;

    /// Combined logical copy cost of this decoder/sampler/input pair. Unknown
    /// remains unknown; this estimate is not physical allocation permission.
    fn estimate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<Option<SnapshotEstimate>, Self::Error>;

    /// Decoder-only growth for future input tokens from this exact saved pair.
    /// Shared policy separately adds saved sampler/input growth and host costs.
    fn estimate_saved_native_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        input_tokens: u64,
    ) -> Result<Option<u64>, Self::Error>;

    /// Stages independent runnable decoder/sampler/input state together without
    /// installation. All fallible copying finishes before returning the tuple.
    /// The backend must establish fresh authority: frozen copy custody and
    /// historical grants alone cannot authorize a resumed run. Funded saved
    /// sources reject until that complete fresh resume contract exists.
    #[allow(clippy::type_complexity)]
    fn prepare_saved_components_resume(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<
        (
            Self::NativeTextState,
            Self::SamplingState,
            Option<PendingTextInput<Self::Prompt, Self::Token>>,
        ),
        Self::Error,
    >;

    /// Captures sampler and pending input together under one copy policy, without
    /// advancing, mutating or invalidating the borrowed live source. Bounded
    /// implementations admit the complete component before either part is copied.
    fn capture_saved_sampling(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Self::SamplingState,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        policy: SamplingCopyPolicy,
    ) -> Result<Self::SavedSamplingState, Self::Error>;
    /// Independently copies an immutable saved component. Advancing or retiring
    /// its original live source must not invalidate its saved numerical contents.
    fn copy_saved_sampling(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
        policy: SamplingCopyPolicy,
    ) -> Result<Self::SavedSamplingState, Self::Error>;
    /// Absolute next prediction represented by this saved component.
    fn saved_sampling_prediction(saved: &Self::SavedSamplingState) -> u64;
    /// Combined logical copy cost of saved sampler and pending input. This is not
    /// a physical copy certificate or allocation permission.
    fn estimate_saved_sampling(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
    ) -> Result<Option<SnapshotEstimate>, Self::Error>;
    /// Input tokens submitted by the next decisions from this saved component.
    fn saved_input_tokens(saved: &Self::SavedSamplingState, predictions: u64) -> Option<u64>;
    /// Additional sampler/input retention through those future decisions.
    fn estimate_saved_sampling_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
        predictions: u64,
    ) -> Result<Option<u64>, Self::Error>;
    /// Stages one independent runnable sampler/input copy without installing it
    /// or changing the saved source. The backend must establish fresh authority;
    /// saved copy custody and historical grants do not authorize a resumed run.
    /// Funded sources must fail until a checked fresh resume contract is available.
    /// Existing unquoted callers retain their whole-host preparation authority
    /// through staging and installation. Escaping component payloads require
    /// custody from their owner; the logical snapshot budget grants none.
    #[allow(clippy::type_complexity)]
    fn prepare_saved_sampling_resume(
        runtime: &mut ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
    ) -> Result<
        (
            Self::SamplingState,
            Option<PendingTextInput<Self::Prompt, Self::Token>>,
        ),
        Self::Error,
    >;

    /// Borrows the native sampling component without changing it.
    fn sampling_state(state: &Self::TextGenerationState) -> &Self::SamplingState;
    /// Installs an independently copied sampler after native restoration succeeds.
    /// Must be infallible, submit no work and leave capture ownership unchanged.
    fn install_sampling_state(state: &mut Self::TextGenerationState, sampling: Self::SamplingState);
    /// Creates child backend state from prepared sampling and shared capture owners.
    /// No new random draw, model execution, or admission occurs here.
    fn assemble_generation_state(
        sampling: Self::SamplingState,
        capture: Option<CaptureSession>,
    ) -> Self::TextGenerationState;
    /// Authenticate a copied pending prompt against the actual saved capture
    /// checkpoint and destination run before either restore or fork publishes it.
    /// This host-only hook must retain copied-source custody on failure. It may
    /// rebind only the derived ordinary capture source; it grants no new original
    /// allocation authority and never refunds copy or capture consumption.
    fn rebind_pending_capture(
        _runtime: &ModelRuntime<Self>,
        _saved: Option<&CaptureCheckpoint>,
        _capture: Option<&CaptureSession>,
        _pending: &mut Option<PendingTextInput<Self::Prompt, Self::Token>>,
    ) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Absolute next prediction, including the inherited prefix.
    fn sampling_prediction(sampling: &Self::SamplingState) -> u64;
    /// Complete known logical sampling-state copy cost, without native allocation.
    fn estimate_sampling_state(
        runtime: &ModelRuntime<Self>,
        sampling: &Self::SamplingState,
    ) -> Result<Option<SnapshotEstimate>, Self::Error>;
    /// Isolates all mutable sampling state and establishes exact native completion.
    fn copy_sampling_state(
        runtime: &mut ModelRuntime<Self>,
        sampling: &Self::SamplingState,
    ) -> Result<Self::SamplingState, Self::Error>;
    /// Complete known cost for preserving pending prefill/decode input. Unsupported
    /// input kinds, including media for a text-only realization, return unknown.
    fn estimate_pending_input(
        runtime: &ModelRuntime<Self>,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Result<Option<SnapshotEstimate>, Self::Error>;
    /// Upper bound on input tokens submitted by the next `predictions` decisions,
    /// including the pending prompt or last committed token exactly once.
    fn continuation_input_tokens(
        _input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
        _predictions: u64,
    ) -> Option<u64> {
        None
    }
    /// Additional sampler and pending-input retention through those decisions.
    /// Includes future history and lazily created native RNG/input storage.
    fn estimate_sampling_growth(
        _runtime: &ModelRuntime<Self>,
        _sampling: &Self::SamplingState,
        _predictions: u64,
    ) -> Result<Option<u64>, Self::Error> {
        Ok(None)
    }
    /// Copies pending input without executing it, sampling or retokenizing.
    #[allow(clippy::type_complexity)]
    fn copy_pending_input(
        runtime: &mut ModelRuntime<Self>,
        input: Option<PendingTextInput<&Self::Prompt, &Self::Token>>,
    ) -> Result<Option<PendingTextInput<Self::Prompt, Self::Token>>, Self::Error>;
    /// Existing shared observation/intervention owner, when enabled.
    fn capture_run(state: &Self::TextGenerationState) -> Option<&CaptureSession>;
    /// Mutably borrows that same owner for validated, non-refunding restoration.
    fn capture_run_mut(state: &mut Self::TextGenerationState) -> Option<&mut CaptureSession>;

    /// Actual native capture facts for shared child re-admission.
    fn estimate_child_capture(
        runtime: &ModelRuntime<Self>,
        shape: &[u64],
        selection: &eredu_core::capture::CaptureSelection,
        slice: &eredu_core::capture::ResolvedCaptureSlice,
    ) -> Result<eredu_core::capture::CaptureUsage, CaptureError>;
    /// Actual intervention estimator for this loaded backend, kept out of saved metadata.
    fn child_intervention_estimator(
        runtime: &ModelRuntime<Self>,
    ) -> Result<std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>, CaptureError>;
}

/// Explicit independent-copy contract for the canonical constraint owner. A
/// shallow `Clone` of grammar or mutable handles does not provide this guarantee.
/// Implementations must retain host custody on independently escaping aliases,
/// including copies obtained through a borrowed controller. The enclosing
/// snapshot's authority protects only payloads that retire with that snapshot;
/// its logical storage estimate does not provide physical allocation permission.
pub trait SnapshotTokenController: TokenFilterController + Sized {
    /// Complete logical controller storage, including mutable grammar state.
    /// Unknown state costs disable snapshots before any copying occurs.
    fn snapshot_storage_bytes(&self) -> Option<u64>;
    /// Copies without committing tokens, changing the source or sharing mutable
    /// parser state. Errors preserve the source and all existing snapshots.
    fn fork_snapshot(&self) -> Result<Self, String>;

    /// Logical inline storage for a source-retained immutable controller copy.
    /// The default leaves original capture unsupported. Implementations must
    /// copy no mutable/deep payload and retain every immutable source alias.
    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        None
    }
    /// Copies only the immutable owner described above, without allocation or
    /// formatting. Destination inline controls are prepaid by the host provider.
    fn fork_original_snapshot(&self) -> Option<Self> {
        None
    }
}

/// Native copy failure retaining its independently accepted host preparation
/// account. The cause cannot be detached from custody; mapping it preserves the
/// same owner and runs while that owner still protects any diagnostic allocation.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct RetainedSnapshotBackendError<E: std::error::Error + 'static> {
    #[source]
    cause: E,
    _custody: HostPreparationAuthority,
}
impl<E: std::error::Error + 'static> RetainedSnapshotBackendError<E> {
    /// Translates only the error representation while preserving exact custody.
    pub fn map_cause<F: std::error::Error + 'static>(
        self,
        convert: impl FnOnce(E) -> F,
    ) -> RetainedSnapshotBackendError<F> {
        let cause = convert(self.cause);
        RetainedSnapshotBackendError {
            cause,
            _custody: self._custody,
        }
    }
}

/// Failure before or during native/portable snapshot composition.
#[derive(Debug, thiserror::Error)]
pub enum TextSnapshotError<E: std::error::Error + 'static> {
    /// Exact original destination account refusal; logical copy work remains spent.
    #[error("host continuation copy admission failed: {0}")]
    HostAdmission(#[source] crate::working_memory::WorkingMemoryError),
    /// Exact backend-domain host preparation was rejected before copying and
    /// before consuming any logical snapshot budget.
    #[error("snapshot host preparation failed: {0}")]
    HostPreparation(#[source] BackendFailure),
    /// The owning facade could not independently copy its complete semantic state.
    #[error("host continuation snapshot failed: {0}")]
    Host(String),
    /// A prepared host copy retained its exact typed source and partial custody.
    #[error("host continuation copy failed: {0}")]
    HostCopy(#[source] BackendFailure),
    /// The canonical host constraint owner could not be copied independently.
    #[error("constraint snapshot failed: {0}")]
    Controller(String),
    /// Exact native copy, completion or compatibility error.
    #[error("snapshot backend operation failed: {0}")]
    Backend(#[source] E),
    /// A bounded native copy failed after its host preparation account accepted.
    /// Any planner/error payload remains protected until the final cause retires.
    #[error(transparent)]
    RetainedBackend(#[from] RetainedSnapshotBackendError<E>),
    /// Fresh original resume failed after independent host admission. The
    /// erased original backend/controller cause retains its destination custody.
    #[error("snapshot resume failed: {0}")]
    Resume(#[source] BackendFailure),
    /// Portable capture/admission state is not a valid boundary.
    #[error(transparent)]
    Capture(#[from] CaptureError),
    /// Unknown estimates, arithmetic or resource limits.
    #[error(transparent)]
    Control(#[from] ExecutionControlError),
    /// Same-run restoration must not import another run's sampler/accounting.
    #[error("snapshot belongs to another continuation")]
    IncompatibleRun,
    /// Portable and native schedule components disagree.
    #[error("snapshot prediction or capture ownership differs")]
    InconsistentState,
    /// Configuration lacks a required continuation mechanism or retained admission.
    #[error("unsupported continuation: {0}")]
    Unsupported(&'static str),
}

impl<E: std::error::Error + 'static> TextSnapshotError<E> {
    /// The original neutral provider failure beneath an independently funded
    /// resume transport. Borrowing it preserves both source and host custody.
    pub fn resume_backend_failure(&self) -> Option<&BackendFailure> {
        let Self::Resume(error) = self else {
            return None;
        };
        let mut source = std::error::Error::source(error);
        while let Some(cause) = source {
            if let Some(error) = cause.downcast_ref::<BackendFailure>() {
                return Some(error);
            }
            source = cause.source();
        }
        None
    }
}

impl<E: std::error::Error + 'static> From<TextHostCopyError> for TextSnapshotError<E> {
    fn from(error: TextHostCopyError) -> Self {
        match error {
            TextHostCopyError::Admission(error) => Self::HostAdmission(error),
            TextHostCopyError::Message(message) => Self::Host(message),
            TextHostCopyError::Source(source) => Self::HostCopy(source),
        }
    }
}

/// Explicit child admission and logical storage bounds. Future model/host growth
/// must be priced by the owning composition before creating a runnable branch.
pub struct TextBranchRequest<'a> {
    /// Fresh facade session identity for re-admitted interventions.
    pub session_id: &'a str,
    /// Absolute prediction limit, including inherited committed predictions.
    pub max_predictions: u64,
    /// Required for a captured/intervened source; includes inherited consumption.
    pub capture_limits: Option<eredu_core::capture::CaptureLimits>,
    /// None inherits operations; Some replaces only future scheduled operations.
    pub intervention: Option<eredu_core::intervention::InterventionPlan>,
    /// Complete child facade semantic/output storage estimate. Controller costs
    /// are supplied separately by `SnapshotTokenController`.
    pub host_bytes: Option<u64>,
    /// Additional retained allowance for state growth through the admitted limit.
    /// This is not charged as already-copied data. Unknown growth fails closed.
    pub continuation_growth_bytes: Option<u64>,
}

/// Independently prepared branch slot. Exchange is serial and moves the previous
/// installed continuation into this slot; immutable weights stay in one runtime.
pub struct TextContinuationBranch<B: TextSnapshotBackend, C: TokenFilterController> {
    native: B::NativeTextState,
    // The continuation's final host authority outlives the native slot too.
    continuation: ManagedTextContinuation<B, C>,
}

/// Ordinary continuation plus its logical branch-retention ownership. Moving or
/// swapping native slots must keep this lease with the logical child, including
/// while that child is installed in the shared executable.
pub struct ManagedTextContinuation<B: eredu_core::TextGenerationBackend, C: TokenFilterController> {
    state: eredu_core::TextGenerationContinuation<B, C>,
    reservation: Option<SnapshotReservation>,
}

impl<B: eredu_core::TextGenerationBackend, C: TokenFilterController> ManagedTextContinuation<B, C> {
    /// Wraps the initial ordinary run, whose baseline execution state is owned by
    /// the caller's loaded runtime. Forked children are constructed with leases.
    pub fn root(state: eredu_core::TextGenerationContinuation<B, C>) -> Self {
        Self {
            state,
            reservation: None,
        }
    }
    /// Uses the exact continuation source for a shared preparation/delivery
    /// agreement. Snapshot storage does not create or reset this protocol owner.
    pub fn finish_text_preparation_cancellable<T, E>(
        &self,
        driver: &eredu_core::TextGenerationDriver<'_, B>,
        stage: eredu_core::run_preparation::TextPreparationStage,
        local: Result<Option<T>, E>,
        map_backend: impl FnOnce(eredu_core::BackendFailure) -> E,
    ) -> Result<Option<T>, E> {
        driver.finish_text_preparation_cancellable(&self.state, stage, local, map_backend)
    }
    /// Advances the installed continuation using the existing ordinary driver.
    #[allow(clippy::type_complexity)]
    pub fn advance(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B>,
    ) -> Result<
        Option<eredu_core::ControlledToken<B::Token>>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.advance(&mut self.state)
    }
    /// Advances with the caller's live cancellation token, outside snapshots.
    #[allow(clippy::type_complexity)]
    pub fn advance_cancellable(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B>,
        cancellation: &eredu_core::GenerationCancellationToken,
    ) -> Result<
        Option<eredu_core::ControlledToken<B::Token>>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.advance_cancellable(&mut self.state, cancellation)
    }
    /// Settles exact completion and moves the retained capture owner.
    /// Shared frames keep their original custody; this adds no allocation or
    /// authority and never clears an earlier execution or drain failure.
    /// A failed drain preserves the pending frame for a later delivery attempt.
    pub fn take_completed_delivery(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B>,
    ) -> Result<
        Option<eredu_core::capture::SharedCapturedStep>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.take_completed_delivery(&mut self.state)
    }
    /// Lends a completed boundary while retaining this run's branch reservation.
    pub fn boundary<'d, 's>(
        &'s mut self,
        driver: &'d mut eredu_core::TextGenerationDriver<'_, B>,
    ) -> Result<
        TextContinuationBoundary<'d, 's, B, C>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.quiescent(&mut self.state)
    }
    /// Canonical logical constraint state.
    pub fn controller(&self) -> &C {
        self.state.controller()
    }
    /// Queries termination through the same core controller without revising
    /// its bound policy. Errors and internal query mutation remain unchanged;
    /// no mutable controller, new step or completed boundary is exposed.
    pub fn controller_is_complete(&mut self) -> Result<bool, C::Error> {
        self.state.controller_is_complete()
    }
    /// Mutably borrows constraint policy, invalidating its previous identity.
    pub fn controller_mut(&mut self) -> &mut C {
        self.state.controller_mut()
    }
    /// Retained logical branch allowance, absent for the original baseline run.
    pub fn retained_branch_bytes(&self) -> Option<u64> {
        self.reservation
            .as_ref()
            .map(SnapshotReservation::retained_bytes)
    }
}
impl<B: TextSnapshotBackend, C: TokenFilterController> TextContinuationBranch<B, C> {
    /// Swaps this branch with the installed ordinary continuation. The facade
    /// must also swap its matching semantic state before executing a prediction.
    pub fn exchange(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B>,
        active: &mut ManagedTextContinuation<B, C>,
    ) -> Result<(), eredu_core::TextContinuationError<B::Error, C::Error>> {
        driver
            .quiescent(&mut active.state)?
            .exchange_branch(&mut self.continuation.state, &mut self.native)?;
        std::mem::swap(&mut active.reservation, &mut self.continuation.reservation);
        Ok(())
    }
}

/// Opaque reusable model/sampler/input/controller/capture checkpoint. The facade
/// pairs this with its semantic/output/lifecycle checkpoint before exposing a full
/// generation snapshot. Native state is never serialized or shallow-cloned.
pub struct TextContinuationSnapshot<B: TextSnapshotBackend, C: TokenFilterController> {
    driver: Option<eredu_core::TextDriverIdentity>,
    identity: TextContinuationIdentity,
    saved: B::SavedTextComponents,
    controller: C,
    remaining_tokens: Option<usize>,
    capture: Option<CaptureCheckpoint>,
    host_bytes: u64,
    _reservation: SnapshotReservation,
    // Must retire after controller/capture/sampler/pending/native payloads.
    _host_preparation: HostPreparationAuthority,
}

impl<B: TextSnapshotBackend, C: SnapshotTokenController> TextContinuationSnapshot<B, C> {
    /// Remaining admitted output positions represented by the saved frontier.
    /// A fresh run may shorten this allowance, but cannot refund or extend it.
    pub fn remaining_tokens(&self) -> Option<usize> {
        self.remaining_tokens
    }
    /// Logical reservation retained by this snapshot, including facade host data.
    pub fn retained_bytes(&self) -> u64 {
        self._reservation.retained_bytes()
    }
    /// Shared storage reservation for host metadata that can escape the saved
    /// native state. Retaining it grants no restoration or execution authority.
    pub fn storage_reservation(&self) -> &SnapshotReservation {
        &self._reservation
    }
    /// Existing destination custody for escaping immutable metadata aliases.
    /// This permits no allocation or copy and preserves the same paid owner.
    pub fn host_preparation(&self) -> &HostPreparationAuthority {
        &self._host_preparation
    }
    /// Absolute next decision represented by this reusable snapshot.
    pub fn next_prediction(&self) -> u64 {
        B::saved_sampling_prediction(B::saved_sampling(&self.saved))
    }

    /// Immutable canonical constraint state for facade-specific growth facts.
    pub fn controller(&self) -> &C {
        &self.controller
    }

    /// Retained source admission/accounting provenance for facade lineage.
    pub fn capture_checkpoint(&self) -> Option<&CaptureCheckpoint> {
        self.capture.as_ref()
    }

    /// Native model, sampler and pending-input growth through an absolute child
    /// decision limit. Facade/controller growth is separate and must also be
    /// reserved before publishing a runnable child. No copy or prediction occurs.
    pub fn native_continuation_growth(
        &self,
        runtime: &ModelRuntime<B>,
        max_predictions: u64,
    ) -> Result<u64, TextSnapshotError<B::Error>> {
        let predictions = max_predictions
            .checked_sub(self.next_prediction())
            .ok_or(TextSnapshotError::InconsistentState)?;
        let saved_sampling = B::saved_sampling(&self.saved);
        let input = B::saved_input_tokens(saved_sampling, predictions)
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let native = B::estimate_saved_native_growth(runtime, &self.saved, input)
            .map_err(TextSnapshotError::Backend)?
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let sampling = B::estimate_saved_sampling_growth(runtime, saved_sampling, predictions)
            .map_err(TextSnapshotError::Backend)?
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        native
            .checked_add(sampling)
            .ok_or_else(|| ExecutionControlError::Overflow.into())
    }

    /// Saves an ordinary continuation after acquiring independent backend host
    /// authority and reserving logical native, capture and caller-owned host
    /// storage. `host_bytes` covers facade semantic/output checkpoint storage;
    /// the controller supplies its own logical estimate. Neither estimate proves
    /// a finite physical bound. Separately exported host state needs its own
    /// custody from the owning composition.
    pub fn capture(
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
        host_bytes: Option<u64>,
    ) -> Result<Self, TextSnapshotError<B::Error>> {
        Self::capture_host(
            boundary,
            budget,
            CallbackHostCopy::new(host_bytes, |_| Ok(())),
        )
        .map(|(snapshot, ())| snapshot)
    }

    /// Copies a borrowed complete host plan inside the same snapshot transaction.
    /// Logical budget is consumed before either host or native destination copy.
    /// The ordinary host authority covers construction and failure cleanup; an
    /// original provider still needs its separately authenticated physical grant.
    pub fn capture_host<H: PreparedTextHostCopy>(
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
        host: H,
    ) -> Result<(Self, H::Copied), TextSnapshotError<B::Error>> {
        Self::capture_source(
            &mut boundary.snapshot_source(),
            budget,
            host,
            SamplingCopyPolicy::Unquoted,
        )
    }

    /// Captures an original borrowed source using independent native and host
    /// copy admission. No ordinary host exclusion or inferred driver identity is
    /// introduced. Captured/intervened sources require their own bounded host
    /// checkpoint producer through the complete-generation source hook.
    pub fn capture_original_host<H: PreparedTextHostCopy>(
        source: &mut TextSnapshotSource<'_, B, C>,
        budget: &SnapshotBudget,
        host: H,
        limits: WorkspaceCopyLimits,
    ) -> Result<(Self, H::Copied), TextSnapshotError<B::Error>> {
        let (runtime, state, pending) = source.parts();
        let preparation = B::original_saved_generation_preparation_bytes(runtime, state, pending)
            .map_err(TextSnapshotError::HostAdmission)?
            .ok_or(TextSnapshotError::Unsupported(
                "original native copy preparation",
            ))?;
        if host
            .original_preparation_bytes()
            .is_none_or(|bytes| bytes < preparation)
        {
            return Err(TextSnapshotError::Unsupported(
                "original native preparation storage",
            ));
        }
        let required = Self::original_capture_control_bytes::<H::Copied>()
            .ok_or(ExecutionControlError::Overflow)?;
        if host
            .original_control_bytes()
            .is_none_or(|bytes| bytes < required)
        {
            return Err(TextSnapshotError::Unsupported(
                "original snapshot host controls",
            ));
        }
        Self::capture_source(source, budget, host, SamplingCopyPolicy::Bounded(limits))
    }

    /// Fixed capture transaction frames, excluding source-derived host/native
    /// payloads and provider-owned admission/custody allocations.
    pub fn original_capture_control_bytes<H>() -> Option<usize> {
        use std::mem::size_of;
        [
            super::PendingSnapshotReservation::control_bytes()?,
            size_of::<Self>(),
            size_of::<C>(),
            size_of::<Option<C>>(),
            size_of::<TextSnapshotSource<'_, B, C>>(),
            size_of::<[Option<SnapshotEstimate>; 3]>(),
            size_of::<Option<[SnapshotEstimate; 3]>>(),
            size_of::<SnapshotEstimate>(),
            size_of::<SnapshotReservation>(),
            size_of::<HostPreparationAuthority>(),
            size_of::<Option<HostPreparationAuthority>>(),
            size_of::<SamplingCopyPolicy>(),
            size_of::<TextSnapshotError<B::Error>>(),
            size_of::<RetainedSnapshotBackendError<B::Error>>(),
            BackendFailure::source_retention_peak_bytes::<B::Error>()?,
            size_of::<Result<(Self, H), TextSnapshotError<B::Error>>>(),
            size_of::<Result<(H, HostPreparationAuthority), TextHostCopyError>>(),
            size_of::<Result<B::SavedTextComponents, B::Error>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    fn capture_source<H: PreparedTextHostCopy>(
        boundary: &mut TextSnapshotSource<'_, B, C>,
        budget: &SnapshotBudget,
        host: H,
        policy: SamplingCopyPolicy,
    ) -> Result<(Self, H::Copied), TextSnapshotError<B::Error>> {
        let original = matches!(policy, SamplingCopyPolicy::Bounded(_));
        let host_bytes = host
            .storage_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let identity = boundary.identity();
        let driver = if original {
            None
        } else {
            boundary.driver_identity()
        };
        let remaining_tokens = boundary.remaining_tokens();
        let (runtime, state, pending) = boundary.parts();
        let native_estimates = if original {
            B::original_generation_snapshot_estimates(runtime, state, pending)
                .ok_or(ExecutionControlError::UnknownEstimate)?
                .map(Some)
        } else {
            [
                B::estimate_native_text_state(runtime, None).map_err(TextSnapshotError::Backend)?,
                B::estimate_sampling_state(runtime, B::sampling_state(state))
                    .map_err(TextSnapshotError::Backend)?,
                B::estimate_pending_input(runtime, pending).map_err(TextSnapshotError::Backend)?,
            ]
        };
        let host_storage = host_bytes
            .checked_add(
                (if original {
                    boundary.controller().original_snapshot_storage_bytes()
                } else {
                    boundary.controller().snapshot_storage_bytes()
                })
                .ok_or(ExecutionControlError::UnknownEstimate)?,
            )
            .ok_or(ExecutionControlError::Overflow)?;
        // Validate known estimates before authority acquisition; owned discovery
        // and estimator metadata below may allocate even though they are cold.
        combine_estimates(
            native_estimates,
            host_storage,
            0,
            std::mem::size_of::<Self>(),
        )?;
        if original && B::capture_run(state).is_some() {
            return Err(TextSnapshotError::Unsupported(
                "original capture checkpoint producer",
            ));
        }
        let preparation = if original {
            None
        } else {
            Some(B::acquire_host_preparation(runtime).map_err(TextSnapshotError::HostPreparation)?)
        };
        let discovery = B::capture_run(state)
            .map(|_| B::capture_discovery(runtime))
            .transpose()?;
        let capture_bytes = match (B::capture_run(state), discovery.as_ref()) {
            (Some(run), Some(discovery)) => run
                .checkpoint_storage_bytes(discovery)
                .ok_or(ExecutionControlError::UnknownEstimate)?,
            _ => 0,
        };
        let estimate = combine_estimates(
            native_estimates,
            host_storage,
            capture_bytes,
            std::mem::size_of::<Self>(),
        )?;
        let reservation = budget.reserve_pending(SnapshotResourceKind::Snapshot, Some(estimate))?;
        // Host exclusion precedes copying; the reservation is logical accounting,
        // not a finite physical proof. The local guard outlives failure cleanup.
        // Failures consume copy allowance without changing reusable snapshots.
        // Declare destination custody first so every copied host/controller
        // prefix retires before it on each later failure, including ordinary
        // callbacks whose output relies on this enclosing exclusion authority.
        let host_preparation;
        let copied_host;
        match preparation {
            Some(authority) => {
                host_preparation = authority;
                copied_host = host.copy(estimate.retained_bytes)?;
            }
            None => {
                let (copied, authority) = host.copy_original(estimate.retained_bytes)?;
                host_preparation = authority;
                copied_host = copied;
            }
        }
        // The shared reservation Rc is born only while the accepted destination
        // host account (or ordinary exclusion) covers its exact control layout.
        let reservation = reservation.publish();
        let controller = if original {
            boundary
                .controller()
                .fork_original_snapshot()
                .ok_or(TextSnapshotError::Unsupported("original controller copy"))?
        } else {
            boundary
                .controller()
                .fork_snapshot()
                .map_err(TextSnapshotError::Controller)?
        };
        let (runtime, state, pending) = boundary.copy_mechanism_parts();
        let capture = match (B::capture_run(state), discovery.as_ref()) {
            (Some(run), Some(discovery)) => Some(run.checkpoint(discovery)?),
            _ => None,
        };
        if let Some(capture) = &capture {
            capture.retain_host_preparation(&host_preparation)?;
        }
        if capture.as_ref().is_some_and(|capture| {
            capture.next_prediction() != B::sampling_prediction(B::sampling_state(state))
        }) {
            return Err(TextSnapshotError::InconsistentState);
        }
        let saved = match B::capture_saved_generation_components_with_host(
            runtime,
            state,
            pending,
            policy,
            &host_preparation,
        ) {
            Ok(saved) => saved,
            Err(cause) if original => {
                return Err(TextSnapshotError::RetainedBackend(
                    RetainedSnapshotBackendError {
                        cause,
                        _custody: host_preparation,
                    },
                ));
            }
            Err(cause) => return Err(TextSnapshotError::Backend(cause)),
        };
        Ok((
            Self {
                driver,
                identity,
                saved,
                controller,
                remaining_tokens,
                capture,
                host_bytes,
                _reservation: reservation,
                _host_preparation: host_preparation,
            },
            copied_host,
        ))
    }

    fn copy_estimate(
        &self,
        runtime: &ModelRuntime<B>,
    ) -> Result<SnapshotEstimate, TextSnapshotError<B::Error>> {
        combine_estimates(
            [B::estimate_saved_components(runtime, &self.saved)
                .map_err(TextSnapshotError::Backend)?],
            self.host_bytes
                .checked_add(
                    self.controller
                        .snapshot_storage_bytes()
                        .ok_or(ExecutionControlError::UnknownEstimate)?,
                )
                .ok_or(ExecutionControlError::Overflow)?,
            match &self.capture {
                Some(capture) => capture
                    .logical_storage_bytes()
                    .ok_or(ExecutionControlError::UnknownEstimate)?,
                None => 0,
            },
            std::mem::size_of::<Self>(),
        )
        .map_err(Into::into)
    }

    /// Restores the same run. All compatibility checks and independent copies
    /// precede installation. Native exchange, prepared capture commit and host
    /// replacement then perform no fallible native work or fresh sampling.
    /// Cumulative capture and copy usage are never restored from old values.
    pub fn restore(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
    ) -> Result<(), TextSnapshotError<B::Error>> {
        self.restore_with(boundary, budget, || Ok(()))
    }

    /// Stages the facade's semantic/cursor copy under backend host authority and
    /// within the logical reservation,
    /// before any native installation. `prepare_host` must preserve its source;
    /// its cost must have been included in the snapshot's `host_bytes`. The
    /// returned host state is installed infallibly by the owning composition.
    /// Arbitrary `H`, callback exports and error payloads must retain their own
    /// custody if they can outlive the installed continuation; the callback does
    /// not receive transferable allocation permission from the logical estimate.
    pub fn restore_with<H>(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
        prepare_host: impl FnOnce() -> Result<H, String>,
    ) -> Result<H, TextSnapshotError<B::Error>> {
        self.restore_host(
            boundary,
            budget,
            CallbackHostCopy::new(Some(self.host_bytes), |_| prepare_host()),
        )
    }

    /// Stages the exact borrowed host source before native installation, using
    /// the same nonrefundable restore reservation and existing exchange worker.
    /// A prepared plan cannot exceed the host amount retained by this snapshot.
    /// Its logical estimate conveys no physical allocation authority.
    pub fn restore_host<H: PreparedTextHostCopy>(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
        host: H,
    ) -> Result<H::Copied, TextSnapshotError<B::Error>> {
        let host_bytes = host
            .storage_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        if host_bytes > self.host_bytes {
            return Err(TextSnapshotError::InconsistentState);
        }
        if self.driver.is_none() {
            return Err(TextSnapshotError::Unsupported(
                "original snapshot fresh resume",
            ));
        }
        if boundary.identity() != self.identity {
            return Err(TextSnapshotError::IncompatibleRun);
        }
        let (runtime, state, _) = boundary.parts();
        B::validate_saved_components(runtime, &self.saved).map_err(TextSnapshotError::Backend)?;
        match (B::capture_run(state), &self.capture) {
            (Some(run), Some(saved)) => run.validate_restore(saved)?,
            (None, None) => {}
            _ => return Err(TextSnapshotError::InconsistentState),
        }
        let estimate = self.copy_estimate(runtime)?;
        let host_preparation =
            B::acquire_host_preparation(runtime).map_err(TextSnapshotError::HostPreparation)?;
        let _reservation = budget.reserve(SnapshotResourceKind::Restore, Some(estimate))?;
        let host = host.copy(estimate.retained_bytes)?;
        let controller = self
            .controller
            .fork_snapshot()
            .map_err(TextSnapshotError::Controller)?;
        let (runtime, _, _) = boundary.copy_mechanism_parts();
        let (mut native, sampling, mut pending) =
            B::prepare_saved_components_resume(runtime, &self.saved)
                .map_err(TextSnapshotError::Backend)?;
        // Installation may unwind after a payload has moved into the active
        // state. Retain first, and keep the local clone until all displaced
        // payloads retire. This custody does not certify successful settlement.
        boundary.retain_host_preparation(host_preparation.clone());
        let (runtime, state, _) = boundary.mechanism_parts();
        if let Some(run) = B::capture_run(state) {
            run.retain_host_preparation(&host_preparation)?;
        }
        B::rebind_pending_capture(
            runtime,
            self.capture.as_ref(),
            B::capture_run(state),
            &mut pending,
        )
        .map_err(TextSnapshotError::Backend)?;
        let capture_restore = match (B::capture_run_mut(state), &self.capture) {
            (Some(run), Some(saved)) => Some(run.prepare_restore(saved)?),
            (None, None) => None,
            _ => return Err(TextSnapshotError::InconsistentState),
        };
        B::exchange_native_text_state(runtime, &mut native).map_err(TextSnapshotError::Backend)?;
        if let Some(restore) = capture_restore {
            restore.commit();
        }
        B::install_sampling_state(state, sampling);
        boundary.install_host_state(controller, pending, self.remaining_tokens);
        Ok(host)
    }

    /// Copies an isolated child and re-admits its shared record owner using actual
    /// backend discovery/estimator facts. It is initially inactive; no model input
    /// is replayed and no original run accounting is reset.
    pub fn fork(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
        request: TextBranchRequest<'_>,
    ) -> Result<TextContinuationBranch<B, C>, TextSnapshotError<B::Error>> {
        self.fork_with(boundary, budget, request, |_, _| Ok(()))
            .map(|(branch, ())| branch)
    }

    /// Stages complete facade state and optional prospective sampler changes
    /// under independent backend host authority and the logical branch
    /// reservation. Preparation receives the independently
    /// copied child sampler and re-admitted capture owner; it cannot replace the
    /// installed parent continuation. On failure no runnable child is published.
    /// The owning composition must retain custody for arbitrary `H`, callback
    /// exports or error payloads that can outlive the returned branch.
    #[allow(clippy::type_complexity)]
    pub fn fork_with<H>(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
        request: TextBranchRequest<'_>,
        prepare: impl FnOnce(
            &mut ModelRuntime<B>,
            &mut B::TextGenerationState,
        ) -> Result<H, TextSnapshotError<B::Error>>,
    ) -> Result<(TextContinuationBranch<B, C>, H), TextSnapshotError<B::Error>> {
        if self.driver.as_ref() != Some(&boundary.driver_identity()) {
            return Err(TextSnapshotError::IncompatibleRun);
        }
        if request.session_id.is_empty() || request.max_predictions < self.next_prediction() {
            return Err(TextSnapshotError::InconsistentState);
        }
        let host_bytes = request
            .host_bytes
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let growth_bytes = request
            .continuation_growth_bytes
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let remaining = usize::try_from(request.max_predictions - self.next_prediction())
            .map_err(|_| ExecutionControlError::Overflow)?;
        let (runtime, _, _) = boundary.parts();
        B::validate_saved_components(runtime, &self.saved).map_err(TextSnapshotError::Backend)?;
        if self.capture.is_none() && request.intervention.is_some() {
            return Err(TextSnapshotError::Unsupported(
                "adding interventions requires retained request admission geometry",
            ));
        }
        let mut estimate = self.copy_estimate(runtime)?;
        let host_preparation =
            B::acquire_host_preparation(runtime).map_err(TextSnapshotError::HostPreparation)?;
        // These cold discovery/estimator constructors can own allocated host
        // metadata, so they must run under the destination's fresh exclusion.
        let discovery = self
            .capture
            .as_ref()
            .map(|_| B::capture_discovery(runtime))
            .transpose()?;
        let needs_intervention = self
            .capture
            .as_ref()
            .is_some_and(|saved| saved.intervention_plan().is_some())
            || request.intervention.is_some();
        let intervention_discovery = needs_intervention
            .then(|| B::intervention_discovery(runtime))
            .transpose()?;
        let child = match discovery.as_ref() {
            Some(discovery) => Some(CaptureForkRequest {
                discovery,
                max_predictions: request.max_predictions,
                limits: request
                    .capture_limits
                    .ok_or(TextSnapshotError::Unsupported(
                        "captured branches require explicit child limits",
                    ))?,
                intervention: match intervention_discovery.as_ref() {
                    Some(discovery) => Some(InterventionForkRequest {
                        discovery,
                        session_id: request.session_id,
                        replacement: request.intervention,
                        estimator: B::child_intervention_estimator(runtime)?,
                    }),
                    None => None,
                },
            }),
            None => None,
        };
        let child_bytes = match (&self.capture, &child) {
            (Some(saved), Some(child)) => saved
                .fork_storage_bytes(child)
                .ok_or(ExecutionControlError::UnknownEstimate)?,
            _ => 0,
        };
        let extra = host_bytes
            .checked_add(child_bytes)
            .ok_or(ExecutionControlError::Overflow)?;
        estimate.retained_bytes = estimate
            .retained_bytes
            .checked_add(extra)
            .and_then(|n| n.checked_add(growth_bytes))
            .ok_or(ExecutionControlError::Overflow)?;
        estimate.copy_bytes = estimate
            .copy_bytes
            .checked_add(extra)
            .ok_or(ExecutionControlError::Overflow)?;
        let reservation = budget.reserve(SnapshotResourceKind::Branch, Some(estimate))?;
        let capture = match (&self.capture, child) {
            (Some(saved), Some(child)) => Some(saved.fork(child, |shape, selection, slice| {
                B::estimate_child_capture(runtime, shape, selection, slice)
            })?),
            _ => None,
        };
        if let Some(capture) = &capture {
            capture.retain_host_preparation(&host_preparation)?;
        }
        let controller = self
            .controller
            .fork_snapshot()
            .map_err(TextSnapshotError::Controller)?;
        let (runtime, _, _) = boundary.copy_mechanism_parts();
        let (native, sampling, mut pending) =
            B::prepare_saved_components_resume(runtime, &self.saved)
                .map_err(TextSnapshotError::Backend)?;
        B::rebind_pending_capture(
            runtime,
            self.capture.as_ref(),
            capture.as_ref(),
            &mut pending,
        )
        .map_err(TextSnapshotError::Backend)?;
        let mut generation = B::assemble_generation_state(sampling, capture);
        let host = prepare(runtime, &mut generation)?;
        let mut state = boundary.fork_host_state(generation, controller, pending, Some(remaining));
        state.retain_host_preparation(host_preparation);
        Ok((
            TextContinuationBranch {
                continuation: ManagedTextContinuation {
                    state,
                    reservation: Some(reservation),
                },
                native,
            },
            host,
        ))
    }
}

fn combine_estimates<const N: usize>(
    estimates: [Option<SnapshotEstimate>; N],
    host: u64,
    capture: u64,
    inline: usize,
) -> Result<SnapshotEstimate, ExecutionControlError> {
    let base = host
        .checked_add(capture)
        .and_then(|n| n.checked_add(u64::try_from(inline).ok()?))
        .ok_or(ExecutionControlError::Overflow)?;
    let mut total = SnapshotEstimate {
        retained_bytes: base,
        copy_bytes: base,
    };
    for estimate in estimates {
        let estimate = estimate.ok_or(ExecutionControlError::UnknownEstimate)?;
        total.retained_bytes = total
            .retained_bytes
            .checked_add(estimate.retained_bytes)
            .ok_or(ExecutionControlError::Overflow)?;
        total.copy_bytes = total
            .copy_bytes
            .checked_add(estimate.copy_bytes)
            .ok_or(ExecutionControlError::Overflow)?;
    }
    Ok(total)
}
