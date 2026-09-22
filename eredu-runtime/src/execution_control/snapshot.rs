//! Shared composition of complete ordinary model/sampling/input continuations.
//!
//! The facade adds its semantic pipeline, output cursor and lifecycle checkpoint.
//! This module never reconstructs a tokenizer, replays a prompt or owns a native
//! completion object. Copy mechanisms finish through the backend's existing owner.

use super::{SnapshotBudget, SnapshotReservation};
mod host_copy;
mod resume;
use crate::capture::{CaptureCheckpoint, CaptureSession};
use crate::working_memory::WorkspaceCopyLimits;
use eredu_core::{
    capture::CaptureError,
    execution_control::{
        ExecutionControlError, NativeTextStateBackend, SnapshotEstimate, SnapshotResourceKind,
    },
    BackendFailure, HostPreparationAuthority, ModelRuntime, PendingTextInput,
    TextContinuationBoundary, TextSnapshotSource, TokenFilterController,
};
pub use host_copy::{PreparedTextHostCopy, PreparedTextHostJournal, TextHostCopyError};
pub use resume::PendingSnapshotResumeRetention;

/// Allocation policy for the complete saved component requested by a hook.
/// A sampling-only hook covers sampler/input; a paired hook also covers decoder
/// state. Neither scope includes controller or facade state, and saved copy
/// custody never authorizes resuming a generation run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SamplingCopyPolicy {
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
    /// This view retains the same independently copied complete saved pair.
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
    ) -> Option<&crate::working_memory::MemoryLedger> {
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
    ) -> Result<Self::SavedTextComponents, Self::Error>;

    /// Borrows immutable sampling/input diagnostics from the exact saved pair.
    /// This gives no mutable extraction, native installation or run permission.
    fn saved_sampling(saved: &Self::SavedTextComponents) -> &Self::SavedSamplingState;

    /// Funded capture provenance retained by the same immutable saved pair.
    /// This loan grants no copy, source replacement, or execution authority.
    fn saved_capture_checkpoint(
        _saved: &Self::SavedTextComponents,
    ) -> Option<&crate::capture::FundedCaptureCheckpoint> {
        None
    }

    /// Checks the saved pair's exact source/domain compatibility without copying,
    /// allocating native data or changing either the saved or installed state.
    fn validate_saved_components(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
    ) -> Result<(), Self::Error>;

    /// Decoder-only growth for future input tokens from this exact saved pair.
    /// Shared policy separately adds saved sampler/input growth and host costs.
    fn estimate_saved_native_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedTextComponents,
        input_tokens: u64,
    ) -> Result<Option<u64>, Self::Error>;

    /// Absolute next prediction represented by this saved component.
    fn saved_sampling_prediction(saved: &Self::SavedSamplingState) -> u64;

    /// Input tokens submitted by the next decisions from this saved component.
    fn saved_input_tokens(saved: &Self::SavedSamplingState, predictions: u64) -> Option<u64>;
    /// Additional sampler/input retention through those future decisions.
    fn estimate_saved_sampling_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Self::SavedSamplingState,
        predictions: u64,
    ) -> Result<Option<u64>, Self::Error>;
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
    /// Existing shared observation/intervention owner, when enabled.
    fn capture_run(state: &Self::TextGenerationState) -> Option<&CaptureSession>;
    /// Cumulative capture consumption of the actual installed collector.
    fn capture_usage(state: &Self::TextGenerationState) -> eredu_core::capture::CaptureUsage {
        Self::capture_run(state).map_or_else(Default::default, CaptureSession::cumulative_usage)
    }
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
/// Opaque reusable model/sampler/input/controller/capture checkpoint. The facade
/// pairs this with its semantic/output/lifecycle checkpoint before exposing a full
/// generation snapshot. Native state is never serialized or shallow-cloned.
pub struct TextContinuationSnapshot<B: TextSnapshotBackend, C: TokenFilterController> {
    saved: B::SavedTextComponents,
    controller: C,
    remaining_tokens: Option<usize>,
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
    pub fn capture_checkpoint(&self) -> Option<&crate::capture::FundedCaptureCheckpoint> {
        B::saved_capture_checkpoint(&self.saved)
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

    /// Copies a borrowed complete host plan inside the same snapshot transaction.
    /// Logical budget is consumed before either host or native destination copy.
    /// The ordinary host authority covers construction and failure cleanup; an
    /// original provider still needs its separately authenticated physical grant.
    pub fn capture_host<H: PreparedTextHostCopy>(
        boundary: &mut TextContinuationBoundary<'_, '_, B, C>,
        budget: &SnapshotBudget,
        host: H,
    ) -> Result<(Self, H::Copied), TextSnapshotError<B::Error>> {
        Self::capture_original_host(
            &mut boundary.snapshot_source(),
            budget,
            host,
            WorkspaceCopyLimits::new(Default::default()),
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
        let host_bytes = host
            .storage_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let remaining_tokens = boundary.remaining_tokens();
        let (runtime, state, pending) = boundary.parts();
        let native_estimates = B::original_generation_snapshot_estimates(runtime, state, pending)
            .ok_or(ExecutionControlError::UnknownEstimate)?
            .map(Some);
        let host_storage = host_bytes
            .checked_add(
                boundary
                    .controller()
                    .original_snapshot_storage_bytes()
                    .ok_or(ExecutionControlError::UnknownEstimate)?,
            )
            .ok_or(ExecutionControlError::Overflow)?;
        if B::capture_run(state).is_some() {
            return Err(TextSnapshotError::Unsupported(
                "original capture checkpoint producer",
            ));
        }
        let estimate = combine_estimates(
            native_estimates,
            host_storage,
            0,
            std::mem::size_of::<Self>(),
        )?;
        let reservation = budget.reserve_pending(SnapshotResourceKind::Snapshot, Some(estimate))?;
        let (copied_host, host_preparation) = host.copy_original(estimate.retained_bytes)?;
        let reservation = reservation.publish();
        let controller = boundary
            .controller()
            .fork_original_snapshot()
            .ok_or(TextSnapshotError::Unsupported("original controller copy"))?;
        let (runtime, state, pending) = boundary.copy_mechanism_parts();
        let saved = match B::capture_saved_generation_components_with_host(
            runtime,
            state,
            pending,
            policy,
            &host_preparation,
        ) {
            Ok(saved) => saved,
            Err(cause) => {
                return Err(TextSnapshotError::RetainedBackend(
                    RetainedSnapshotBackendError {
                        cause,
                        _custody: host_preparation,
                    },
                ))
            }
        };
        Ok((
            Self {
                saved,
                controller,
                remaining_tokens,
                _reservation: reservation,
                _host_preparation: host_preparation,
            },
            copied_host,
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
