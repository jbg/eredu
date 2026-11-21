//! Shared composition of complete ordinary model/sampling/input continuations.
//!
//! The facade adds its semantic pipeline, output cursor and lifecycle checkpoint.
//! This module never reconstructs a tokenizer, replays a prompt or owns a native
//! completion object. Copy mechanisms finish through the backend's existing owner.

use super::{SnapshotBudget, SnapshotReservation};
use crate::capture::CaptureSession;
#[cfg(feature = "tensor-capture")]
use crate::capture::{CaptureCheckpoint, CaptureForkRequest, InterventionForkRequest};
use eredu_core::{
    capture::CaptureError,
    execution_control::{
        ExecutionControlError, NativeTextStateSession, SnapshotEstimate, SnapshotResourceKind,
    },
    ModelRuntime, PendingTextInput, TextContinuationBoundary, TextContinuationIdentity,
    TokenFilterController,
};

/// Native ordinary-generation mechanisms used by the portable snapshot driver.
/// Capture/intervention ownership stays in the shared `CaptureSession`; adapters
/// expose that owner and perform native copying, never duplicate its bookkeeping.
pub trait TextSnapshotBackend<
    S: NativeTextStateSession<Self> + eredu_core::backend::TextOperationDriver<Self>,
>: eredu_core::TextGenerationBackend
{
    /// Sampling parameters, penalties/history, adaptive state, exact RNG and
    /// absolute next-prediction position. Copies must have isolated mutable state.
    type SamplingState;

    /// Borrows the native sampling component without changing it.
    fn sampling_state(state: &S::TextGenerationState) -> &Self::SamplingState;
    /// Installs an independently copied sampler after native restoration succeeds.
    /// Must be infallible, submit no work and leave capture ownership unchanged.
    fn install_sampling_state(state: &mut S::TextGenerationState, sampling: Self::SamplingState);
    /// Creates child backend state from prepared sampling and shared capture owners.
    /// No new random draw, model execution, or admission occurs here.
    fn assemble_generation_state(
        sampling: Self::SamplingState,
        #[cfg(feature = "tensor-capture")] capture: Option<CaptureSession>,
    ) -> S::TextGenerationState;
    /// Absolute next prediction, including the inherited prefix.
    fn sampling_prediction(sampling: &Self::SamplingState) -> u64;
    /// Complete known logical sampling-state copy cost, without native allocation.
    fn estimate_sampling_state(
        runtime: &ModelRuntime<Self, S>,
        sampling: &Self::SamplingState,
    ) -> Result<Option<SnapshotEstimate>, Self::Error>;
    /// Isolates all mutable sampling state and establishes exact native completion.
    fn copy_sampling_state(
        runtime: &mut ModelRuntime<Self, S>,
        sampling: &Self::SamplingState,
    ) -> Result<Self::SamplingState, Self::Error>;
    /// Complete known cost for preserving pending prefill/decode input. Unsupported
    /// input kinds, including media for a text-only realization, return unknown.
    fn estimate_pending_input(
        runtime: &ModelRuntime<Self, S>,
        input: Option<PendingTextInput<&S::Prompt, &Self::Token>>,
    ) -> Result<Option<SnapshotEstimate>, Self::Error>;
    /// Upper bound on input tokens submitted by the next `predictions` decisions,
    /// including the pending prompt or last committed token exactly once.
    fn continuation_input_tokens(
        _runtime: &ModelRuntime<Self, S>,
        _sampling: &Self::SamplingState,
        _input: Option<PendingTextInput<&S::Prompt, &Self::Token>>,
        _predictions: u64,
    ) -> Option<u64> {
        None
    }
    /// Committed causal input appended at a queued mode's settled boundary.
    /// Includes a block that finishes before the handoff. No work, draws or copies occur.
    fn decoding_settlement_input_tokens(
        _runtime: &ModelRuntime<Self, S>,
        _sampling: &Self::SamplingState,
        _input: Option<PendingTextInput<&S::Prompt, &Self::Token>>,
    ) -> Option<u64> {
        None
    }
    /// Additional sampler and pending-input retention through those decisions.
    /// Includes future history and lazily created native RNG/input storage.
    fn estimate_sampling_growth(
        _runtime: &ModelRuntime<Self, S>,
        _sampling: &Self::SamplingState,
        _predictions: u64,
    ) -> Result<Option<u64>, Self::Error> {
        Ok(None)
    }
    /// Copies pending input without executing it, sampling or retokenizing.
    #[allow(clippy::type_complexity)]
    fn copy_pending_input(
        runtime: &mut ModelRuntime<Self, S>,
        input: Option<PendingTextInput<&S::Prompt, &Self::Token>>,
    ) -> Result<Option<PendingTextInput<S::Prompt, Self::Token>>, Self::Error>;
    /// Existing shared observation/intervention owner, when enabled.
    #[cfg(feature = "tensor-capture")]
    fn capture_run(state: &S::TextGenerationState) -> Option<&CaptureSession>;
    /// Mutably borrows that same owner for validated, non-refunding restoration.
    #[cfg(feature = "tensor-capture")]
    fn capture_run_mut(state: &mut S::TextGenerationState) -> Option<&mut CaptureSession>;

    /// Actual native capture facts for shared child re-admission.
    #[cfg(feature = "tensor-capture")]
    fn estimate_child_capture(
        runtime: &ModelRuntime<Self, S>,
        shape: &[u64],
        selection: &eredu_core::capture::CaptureSelection,
        slice: &eredu_core::capture::ResolvedCaptureSlice,
    ) -> Result<eredu_core::capture::CaptureUsage, CaptureError>;
    /// Actual intervention estimator for this loaded backend, kept out of saved metadata.
    #[cfg(feature = "tensor-capture")]
    fn child_intervention_estimator(
        runtime: &ModelRuntime<Self, S>,
    ) -> Result<std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>, CaptureError>;
}

/// Explicit independent-copy contract for the canonical constraint owner. A
/// shallow `Clone` of grammar or mutable handles does not provide this guarantee.
pub trait SnapshotTokenController: TokenFilterController + Sized {
    /// Complete logical controller storage, including mutable grammar state.
    /// Unknown state costs disable snapshots before any copying occurs.
    fn snapshot_storage_bytes(&self) -> Option<u64>;
    /// Copies without committing tokens, changing the source or sharing mutable
    /// parser state. Errors preserve the source and all existing snapshots.
    fn fork_snapshot(&self) -> Result<Self, String>;
}

/// Failure before or during native/portable snapshot composition.
#[derive(Debug, thiserror::Error)]
pub enum TextSnapshotError<E: std::error::Error + 'static> {
    /// The owning facade could not independently copy its complete semantic state.
    #[error("host continuation snapshot failed: {0}")]
    Host(String),
    /// The canonical host constraint owner could not be copied independently.
    #[error("constraint snapshot failed: {0}")]
    Controller(String),
    /// Exact native copy, completion or compatibility error.
    #[error("snapshot backend operation failed: {0}")]
    Backend(#[source] E),
    /// Portable capture/admission state is not a valid boundary.
    #[error(transparent)]
    #[cfg(feature = "tensor-capture")]
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

/// Explicit child admission and logical storage bounds. Future model/host growth
/// must be priced by the owning composition before creating a runnable branch.
pub struct TextBranchRequest<'a> {
    /// Fresh facade session identity for re-admitted interventions.
    pub session_id: &'a str,
    /// Absolute prediction limit, including inherited committed predictions.
    pub max_predictions: u64,
    /// Required for a captured/intervened source; includes inherited consumption.
    #[cfg(feature = "tensor-capture")]
    pub capture_limits: Option<eredu_core::capture::CaptureLimits>,
    /// None inherits operations; Some replaces only future scheduled operations.
    #[cfg(feature = "tensor-capture")]
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
pub struct TextContinuationBranch<
    B: TextSnapshotBackend<S>,
    C: TokenFilterController,
    S: eredu_core::TextExecutionSession<B> + NativeTextStateSession<B>,
> {
    continuation: ManagedTextContinuation<B, C, S>,
    native: <S as NativeTextStateSession<B>>::NativeTextState,
}

/// Ordinary continuation plus its logical branch-retention ownership. Moving or
/// swapping native slots must keep this lease with the logical child, including
/// while that child is installed in the shared executable.
pub struct ManagedTextContinuation<
    B: eredu_core::TextGenerationBackend,
    C: TokenFilterController,
    S: eredu_core::TextExecutionSession<B>,
> {
    state: eredu_core::TextGenerationContinuation<B, C, S>,
    reservation: Option<SnapshotReservation>,
}

impl<
        B: eredu_core::TextGenerationBackend,
        C: TokenFilterController,
        S: eredu_core::TextExecutionSession<B>,
    > ManagedTextContinuation<B, C, S>
{
    /// Wraps the initial ordinary run, whose baseline execution state is owned by
    /// the caller's loaded runtime. Forked children are constructed with leases.
    pub fn root(state: eredu_core::TextGenerationContinuation<B, C, S>) -> Self {
        Self {
            state,
            reservation: None,
        }
    }
    /// Associates the active facade run's cancellation token after branch exchange.
    pub fn set_cancellation_token(
        &mut self,
        cancellation: eredu_core::GenerationCancellationToken,
    ) {
        self.state.set_cancellation_token(cancellation);
    }
    /// Advances the installed continuation using the existing ordinary driver.
    #[allow(clippy::type_complexity)]
    pub fn advance(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B, S>,
    ) -> Result<
        Option<eredu_core::ControlledToken<B::Token>>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.advance(&mut self.state)
    }
    /// Advances one physical operation through the ordinary generation machine.
    pub fn advance_operation(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B, S>,
    ) -> Result<
        Option<eredu_core::TextAdvancement<B::Token>>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.advance_operation(&mut self.state)
    }
    /// Requests selected scores for the next refinement operation.
    pub fn configure_refinement_readout(
        &mut self,
        driver: &eredu_core::TextGenerationDriver<'_, B, S>,
        selections: Vec<eredu_core::refinement::RefinementScoreSelection>,
    ) -> Result<(), eredu_core::refinement::RefinementReadoutError> {
        driver.configure_refinement_readout(&mut self.state, selections)
    }

    /// Drains selected scores after completion.
    pub fn take_refinement_readout(
        &mut self,
        driver: &eredu_core::TextGenerationDriver<'_, B, S>,
    ) -> Result<
        Option<eredu_core::refinement::RefinementScoreRows>,
        eredu_core::refinement::RefinementReadoutError,
    > {
        driver.take_refinement_readout(&mut self.state)
    }

    /// Completes one nonfinal prompt chunk through the ordinary driver.
    pub fn advance_prefill(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B, S>,
    ) -> Result<bool, eredu_core::TextContinuationError<B::Error, C::Error>> {
        driver.advance_prefill(&mut self.state)
    }
    /// Settles exact native completion at the current boundary.
    #[cfg(not(feature = "tensor-capture"))]
    pub fn take_completed_step(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B, S>,
    ) -> Result<(), eredu_core::TextContinuationError<B::Error, C::Error>> {
        driver.take_completed_step(&mut self.state)
    }

    /// Settles and drains this continuation's bounded record step.
    #[cfg(feature = "tensor-capture")]
    pub fn take_completed_step(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B, S>,
    ) -> Result<
        Option<eredu_core::capture::CapturedStep>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.take_completed_step(&mut self.state)
    }
    /// Lends a completed boundary while retaining this run's branch reservation.
    pub fn boundary<'d, 's>(
        &'s mut self,
        driver: &'d mut eredu_core::TextGenerationDriver<'_, B, S>,
    ) -> Result<
        TextContinuationBoundary<'d, 's, B, C, S>,
        eredu_core::TextContinuationError<B::Error, C::Error>,
    > {
        driver.quiescent(&mut self.state)
    }
    /// Completed physical-work and authoritative-output coordinates.
    pub fn position(&self) -> eredu_core::refinement::GenerationPosition {
        self.state.position()
    }
    /// Active/pending modes and last settled transition.
    pub fn decoding_state(&self) -> eredu_core::decoding::DecodingState {
        self.state.decoding_state()
    }
    /// Exact modes realized before this run began.
    pub fn decoding_capabilities(&self) -> eredu_core::decoding::DecodingCapabilities {
        self.state.decoding_capabilities()
    }
    /// Queues one validated change for the shared driver.
    pub fn request_decoding_mode(
        &self,
        mode: eredu_core::decoding::DecodingMode,
    ) -> Result<(), eredu_core::decoding::DecodingModeError> {
        self.state.request_decoding_mode(mode)
    }
    /// Binds a thread-safe command handle to this logical run.
    pub fn bind_control_handle(
        &self,
        handle: &eredu_core::execution_control::GenerationControlHandle,
    ) -> Result<(), eredu_core::decoding::DecodingModeError> {
        self.state.bind_control_handle(handle)
    }
    /// Stops mode commands when facade termination precedes the backend allowance.
    pub fn finish_decoding(&self) {
        self.state.finish_decoding();
    }

    /// Canonical logical constraint state.
    pub fn controller(&self) -> &C {
        self.state.controller()
    }
    /// Mutable constraint queries used by the ordinary facade semantic driver.
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
impl<
        B: TextSnapshotBackend<S>,
        C: TokenFilterController,
        S: eredu_core::TextExecutionSession<B> + NativeTextStateSession<B>,
    > TextContinuationBranch<B, C, S>
{
    /// Binds commands to the independently owned logical child, even while inactive.
    pub fn bind_control_handle(
        &self,
        handle: &eredu_core::execution_control::GenerationControlHandle,
    ) -> Result<(), eredu_core::decoding::DecodingModeError> {
        self.continuation.bind_control_handle(handle)
    }
    /// Swaps this branch with the installed ordinary continuation. The facade
    /// must also swap its matching semantic state before executing a prediction.
    pub fn exchange(
        &mut self,
        driver: &mut eredu_core::TextGenerationDriver<'_, B, S>,
        active: &mut ManagedTextContinuation<B, C, S>,
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
pub struct TextContinuationSnapshot<
    B: TextSnapshotBackend<S>,
    C: TokenFilterController,
    S: eredu_core::TextExecutionSession<B> + NativeTextStateSession<B>,
> {
    driver: eredu_core::TextDriverIdentity,
    identity: TextContinuationIdentity,
    native: <S as NativeTextStateSession<B>>::NativeTextState,
    sampling: B::SamplingState,
    pending: Option<PendingTextInput<S::Prompt, B::Token>>,
    controller: C,
    sequence: eredu_core::GenerationSequence,
    remaining_tokens: Option<usize>,
    position: eredu_core::refinement::GenerationPosition,
    decoding: eredu_core::decoding::DecodingState,
    #[cfg(feature = "tensor-capture")]
    capture: Option<CaptureCheckpoint>,
    host_bytes: u64,
    _reservation: SnapshotReservation,
}

impl<
        B: TextSnapshotBackend<S>,
        C: SnapshotTokenController,
        S: eredu_core::TextExecutionSession<B> + NativeTextStateSession<B>,
    > TextContinuationSnapshot<B, C, S>
{
    /// Logical policy at the same boundary as the saved sampler and cache.
    pub fn decoding_state(&self) -> eredu_core::decoding::DecodingState {
        self.decoding
    }
    /// Logical reservation retained by this snapshot, including facade host data.
    pub fn retained_bytes(&self) -> u64 {
        self._reservation.retained_bytes()
    }
    /// Absolute next decision represented by this reusable snapshot.
    pub fn next_prediction(&self) -> u64 {
        B::sampling_prediction(&self.sampling)
    }

    /// Immutable canonical constraint state for facade-specific growth facts.
    pub fn controller(&self) -> &C {
        &self.controller
    }

    /// Retained source admission/accounting provenance for facade lineage.
    #[cfg(feature = "tensor-capture")]
    pub fn capture_checkpoint(&self) -> Option<&CaptureCheckpoint> {
        self.capture.as_ref()
    }

    /// Native model, sampler and pending-input growth through an absolute child
    /// decision limit. Facade/controller growth is separate and must also be
    /// reserved before publishing a runnable child. No copy or prediction occurs.
    pub fn native_continuation_growth(
        &self,
        runtime: &ModelRuntime<B, S>,
        max_predictions: u64,
    ) -> Result<u64, TextSnapshotError<B::Error>> {
        let predictions = max_predictions
            .checked_sub(self.next_prediction())
            .ok_or(TextSnapshotError::InconsistentState)?;
        let input = B::continuation_input_tokens(
            runtime,
            &self.sampling,
            self.pending.as_ref().map(PendingTextInput::as_ref),
            predictions,
        )
        .ok_or(ExecutionControlError::UnknownEstimate)?;
        let native = S::estimate_native_text_growth(runtime, &self.native, input)
            .map_err(TextSnapshotError::Backend)?
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let sampling = B::estimate_sampling_growth(runtime, &self.sampling, predictions)
            .map_err(TextSnapshotError::Backend)?
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        native
            .checked_add(sampling)
            .and_then(|bytes| bytes.checked_add(predictions.checked_mul(4)?))
            .ok_or_else(|| ExecutionControlError::Overflow.into())
    }

    /// Saves a complete ordinary continuation after reserving native, portable
    /// capture and caller-owned host state. `host_bytes` covers facade semantic/
    /// output checkpoint storage; the controller supplies its own known estimate.
    pub fn capture(
        boundary: &mut TextContinuationBoundary<'_, '_, B, C, S>,
        budget: &SnapshotBudget,
        host_bytes: Option<u64>,
    ) -> Result<Self, TextSnapshotError<B::Error>> {
        let host_bytes = host_bytes.ok_or(ExecutionControlError::UnknownEstimate)?;
        let identity = boundary.identity();
        let driver = boundary.driver_identity();
        let remaining_tokens = boundary.remaining_tokens();
        let position = boundary.position();
        let decoding = boundary.decoding_state();
        let sequence_bytes = boundary
            .sequence()
            .snapshot_storage_bytes()
            .ok_or(ExecutionControlError::UnknownEstimate)?;
        let (runtime, state, pending) = boundary.parts();
        #[cfg(feature = "tensor-capture")]
        let discovery = B::capture_run(state)
            .map(|_| runtime.session().capture_discovery())
            .transpose()?;
        #[cfg(feature = "tensor-capture")]
        let capture_bytes = match (B::capture_run(state), discovery.as_ref()) {
            (Some(run), Some(discovery)) => run
                .checkpoint_storage_bytes(discovery)
                .ok_or(ExecutionControlError::UnknownEstimate)?,
            _ => 0,
        };
        let estimate = combine_estimates(
            [
                S::estimate_native_text_state(runtime, None).map_err(TextSnapshotError::Backend)?,
                B::estimate_sampling_state(runtime, B::sampling_state(state))
                    .map_err(TextSnapshotError::Backend)?,
                B::estimate_pending_input(runtime, pending).map_err(TextSnapshotError::Backend)?,
            ],
            host_bytes
                .checked_add(sequence_bytes)
                .ok_or(ExecutionControlError::Overflow)?
                .checked_add(
                    boundary
                        .controller()
                        .snapshot_storage_bytes()
                        .ok_or(ExecutionControlError::UnknownEstimate)?,
                )
                .ok_or(ExecutionControlError::Overflow)?,
            #[cfg(feature = "tensor-capture")]
            capture_bytes,
            #[cfg(not(feature = "tensor-capture"))]
            0,
            std::mem::size_of::<Self>(),
        )?;
        let reservation = budget.reserve(SnapshotResourceKind::Snapshot, Some(estimate))?;
        // Everything below is admitted. Failures consume copy allowance but do
        // not change the source or any existing reusable snapshot.
        let sequence = boundary.sequence().clone();
        let controller = boundary
            .controller()
            .fork_snapshot()
            .map_err(TextSnapshotError::Controller)?;
        let (runtime, state, pending) = boundary.mechanism_parts();
        #[cfg(feature = "tensor-capture")]
        let capture = match (B::capture_run(state), discovery.as_ref()) {
            (Some(run), Some(discovery)) => Some(run.checkpoint(discovery)?),
            _ => None,
        };
        #[cfg(feature = "tensor-capture")]
        if capture.as_ref().is_some_and(|capture| {
            capture.next_prediction() != B::sampling_prediction(B::sampling_state(state))
        }) {
            return Err(TextSnapshotError::InconsistentState);
        }
        let pending =
            B::copy_pending_input(runtime, pending).map_err(TextSnapshotError::Backend)?;
        let sampling = B::copy_sampling_state(runtime, B::sampling_state(state))
            .map_err(TextSnapshotError::Backend)?;
        let native = S::capture_native_text_state(runtime).map_err(TextSnapshotError::Backend)?;
        Ok(Self {
            driver,
            identity,
            native,
            sampling,
            pending,
            controller,
            sequence,
            remaining_tokens,
            position,
            decoding,
            #[cfg(feature = "tensor-capture")]
            capture,
            host_bytes,
            _reservation: reservation,
        })
    }

    fn copy_estimate(
        &self,
        runtime: &ModelRuntime<B, S>,
    ) -> Result<SnapshotEstimate, TextSnapshotError<B::Error>> {
        combine_estimates(
            [
                S::estimate_native_text_state(runtime, Some(&self.native))
                    .map_err(TextSnapshotError::Backend)?,
                B::estimate_sampling_state(runtime, &self.sampling)
                    .map_err(TextSnapshotError::Backend)?,
                B::estimate_pending_input(
                    runtime,
                    self.pending.as_ref().map(PendingTextInput::as_ref),
                )
                .map_err(TextSnapshotError::Backend)?,
            ],
            self.host_bytes
                .checked_add(
                    self.sequence
                        .snapshot_storage_bytes()
                        .ok_or(ExecutionControlError::UnknownEstimate)?,
                )
                .ok_or(ExecutionControlError::Overflow)?
                .checked_add(
                    self.controller
                        .snapshot_storage_bytes()
                        .ok_or(ExecutionControlError::UnknownEstimate)?,
                )
                .ok_or(ExecutionControlError::Overflow)?,
            #[cfg(feature = "tensor-capture")]
            match &self.capture {
                Some(capture) => capture
                    .logical_storage_bytes()
                    .ok_or(ExecutionControlError::UnknownEstimate)?,
                None => 0,
            },
            #[cfg(not(feature = "tensor-capture"))]
            0,
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
        boundary: &mut TextContinuationBoundary<'_, '_, B, C, S>,
        budget: &SnapshotBudget,
    ) -> Result<(), TextSnapshotError<B::Error>> {
        self.restore_with(boundary, budget, || Ok(()))
    }

    /// Stages the facade's semantic/cursor copy within the complete reservation,
    /// before any native installation. `prepare_host` must preserve its source;
    /// its cost must have been included in the snapshot's `host_bytes`. The
    /// returned host state is installed infallibly by the owning composition.
    pub fn restore_with<H>(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C, S>,
        budget: &SnapshotBudget,
        prepare_host: impl FnOnce() -> Result<H, String>,
    ) -> Result<H, TextSnapshotError<B::Error>> {
        if boundary.identity() != self.identity {
            return Err(TextSnapshotError::IncompatibleRun);
        }
        let (runtime, state, _) = boundary.parts();
        S::validate_native_text_state(runtime, &self.native).map_err(TextSnapshotError::Backend)?;
        #[cfg(feature = "tensor-capture")]
        match (B::capture_run(state), &self.capture) {
            (Some(run), Some(saved)) => run.validate_restore(saved)?,
            (None, None) => {}
            _ => return Err(TextSnapshotError::InconsistentState),
        }
        let _reservation = budget.reserve(
            SnapshotResourceKind::Restore,
            Some(self.copy_estimate(runtime)?),
        )?;
        let host = prepare_host().map_err(TextSnapshotError::Host)?;
        let controller = self
            .controller
            .fork_snapshot()
            .map_err(TextSnapshotError::Controller)?;
        let (runtime, state, _) = boundary.mechanism_parts();
        let pending =
            B::copy_pending_input(runtime, self.pending.as_ref().map(PendingTextInput::as_ref))
                .map_err(TextSnapshotError::Backend)?;
        let sampling =
            B::copy_sampling_state(runtime, &self.sampling).map_err(TextSnapshotError::Backend)?;
        let mut native =
            S::copy_native_text_state(runtime, &self.native).map_err(TextSnapshotError::Backend)?;
        #[cfg(feature = "tensor-capture")]
        let capture_restore = match (B::capture_run_mut(state), &self.capture) {
            (Some(run), Some(saved)) => Some(run.prepare_restore(saved)?),
            (None, None) => None,
            _ => return Err(TextSnapshotError::InconsistentState),
        };
        S::exchange_native_text_state(runtime, &mut native).map_err(TextSnapshotError::Backend)?;
        #[cfg(feature = "tensor-capture")]
        if let Some(restore) = capture_restore {
            restore.commit();
        }
        B::install_sampling_state(state, sampling);
        boundary.install_host_state(
            controller,
            pending,
            self.remaining_tokens,
            self.position,
            self.decoding,
            self.sequence.clone(),
        );
        Ok(host)
    }

    /// Copies an isolated child and re-admits its shared record owner using actual
    /// backend discovery/estimator facts. It is initially inactive; no model input
    /// is replayed and no original run accounting is reset.
    pub fn fork(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C, S>,
        budget: &SnapshotBudget,
        request: TextBranchRequest<'_>,
    ) -> Result<TextContinuationBranch<B, C, S>, TextSnapshotError<B::Error>> {
        self.fork_with(boundary, budget, request, |_, _| Ok(()))
            .map(|(branch, ())| branch)
    }

    /// Stages complete facade state and optional prospective sampler changes
    /// under the branch reservation. Preparation receives the independently
    /// copied child sampler and re-admitted capture owner; it cannot replace the
    /// installed parent continuation. On failure no runnable child is published.
    #[allow(clippy::type_complexity)]
    pub fn fork_with<H>(
        &self,
        boundary: &mut TextContinuationBoundary<'_, '_, B, C, S>,
        budget: &SnapshotBudget,
        request: TextBranchRequest<'_>,
        prepare: impl FnOnce(
            &mut ModelRuntime<B, S>,
            &mut S::TextGenerationState,
        ) -> Result<H, TextSnapshotError<B::Error>>,
    ) -> Result<(TextContinuationBranch<B, C, S>, H), TextSnapshotError<B::Error>> {
        if self.driver != boundary.driver_identity() {
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
        S::validate_native_text_state(runtime, &self.native).map_err(TextSnapshotError::Backend)?;
        #[cfg(feature = "tensor-capture")]
        if self.capture.is_none() && request.intervention.is_some() {
            return Err(TextSnapshotError::Unsupported(
                "adding interventions requires retained request admission geometry",
            ));
        }
        #[cfg(feature = "tensor-capture")]
        let discovery = self
            .capture
            .as_ref()
            .map(|_| runtime.session().capture_discovery())
            .transpose()?;
        #[cfg(feature = "tensor-capture")]
        let needs_intervention = self
            .capture
            .as_ref()
            .is_some_and(|saved| saved.intervention_plan().is_some())
            || request.intervention.is_some();
        #[cfg(feature = "tensor-capture")]
        let intervention_discovery = needs_intervention
            .then(|| runtime.session().intervention_discovery())
            .transpose()?;
        #[cfg(feature = "tensor-capture")]
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
        #[cfg(feature = "tensor-capture")]
        let child_bytes = match (&self.capture, &child) {
            (Some(saved), Some(child)) => saved
                .fork_storage_bytes(child)
                .ok_or(ExecutionControlError::UnknownEstimate)?,
            _ => 0,
        };
        #[cfg(not(feature = "tensor-capture"))]
        let child_bytes = 0;
        let mut estimate = self.copy_estimate(runtime)?;
        let extra = host_bytes
            .checked_add(child_bytes)
            // A child has an independent mode mailbox in addition to saved data.
            .and_then(|n| n.checked_add(crate::generation::mode_mailbox_bytes()))
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
        #[cfg(feature = "tensor-capture")]
        let capture = match (&self.capture, child) {
            (Some(saved), Some(child)) => Some(saved.fork(child, |shape, selection, slice| {
                B::estimate_child_capture(runtime, shape, selection, slice)
            })?),
            _ => None,
        };
        let controller = self
            .controller
            .fork_snapshot()
            .map_err(TextSnapshotError::Controller)?;
        let (runtime, _, _) = boundary.mechanism_parts();
        let pending =
            B::copy_pending_input(runtime, self.pending.as_ref().map(PendingTextInput::as_ref))
                .map_err(TextSnapshotError::Backend)?;
        let sampling =
            B::copy_sampling_state(runtime, &self.sampling).map_err(TextSnapshotError::Backend)?;
        let native =
            S::copy_native_text_state(runtime, &self.native).map_err(TextSnapshotError::Backend)?;
        let mut generation = B::assemble_generation_state(
            sampling,
            #[cfg(feature = "tensor-capture")]
            capture,
        );
        let host = prepare(runtime, &mut generation)?;
        let state = boundary.fork_host_state(
            generation,
            controller,
            pending,
            Some(remaining),
            self.position,
            self.decoding,
            self.sequence.with_remaining(remaining),
        );
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
