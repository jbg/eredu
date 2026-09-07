//! Selected-local wrappers; all control policy lives in the portable facade.

use super::{LocalBackendError, LocalModel, SelectedBackend};
use crate::api::{
    ControlledGenerationError, ControlledGenerationRecord, GenerationBranchOptions,
    GenerationOutputCheckpoint, GenerationSnapshotMetadata, PreparedObservedGeneration,
    SamplingOverride, SamplingStateFacts,
};
use eredu_core::{execution_control::*, generation::FinishReason};
use std::ops::ControlFlow;

type Result<T> = std::result::Result<T, ControlledGenerationError<LocalBackendError>>;

/// A controllable ordinary local run, exclusively borrowing its loaded model.
/// Native handles stay on their worker; `control_handle` may cross threads.
pub struct LocalControlledGenerationSession<'a> {
    inner: crate::api::ControlledGenerationSession<'a, SelectedBackend>,
}

/// Opaque reusable in-process snapshot for its exact source run and model.
/// Immutable metadata may be serialized; the continuation itself may not.
pub struct LocalGenerationSnapshot {
    inner: crate::api::ControlledGenerationSnapshot<SelectedBackend>,
}

/// Inactive isolated local continuation, exchanged serially under one model owner.
/// After exchange this handle holds the previously active run.
pub struct LocalGenerationBranch {
    inner: crate::api::ControlledGenerationBranch<SelectedBackend>,
}
impl LocalGenerationBranch {
    /// Logical run currently retained in this slot.
    pub fn run_id(&self) -> &str {
        self.inner.run_id()
    }
    /// Exact canonical history of the inactive run.
    pub fn token_ids(&self) -> &[u32] {
        self.inner.token_ids()
    }
    /// Inactive lifecycle, including normal completion or cancellation.
    pub fn status(&self) -> GenerationStatus {
        self.inner.status()
    }
    /// Thread-safe requests for this logical run, including while inactive.
    pub fn control_handle(&self) -> GenerationControlHandle {
        self.inner.control_handle()
    }
}

impl LocalGenerationSnapshot {
    /// Immutable canonical prompt shared by this complete continuation.
    pub fn prompt_token_ids(&self) -> &[u32] {
        self.inner.prompt_token_ids()
    }
    /// Versioned attribution, boundary and logical retention facts.
    pub fn metadata(&self) -> &GenerationSnapshotMetadata {
        self.inner.metadata()
    }
    /// Canonical history saved with the incremental semantic state.
    pub fn token_ids(&self) -> &[u32] {
        self.inner.token_ids()
    }
}

impl LocalModel {
    /// Starts ordinary completed-token control with existing prepared capture and
    /// intervention admissions. No prediction or random draw occurs at startup.
    pub fn start_controlled_chat<'a>(
        &'a mut self,
        prepared: PreparedObservedGeneration,
        caller_stop_sequences: &[String],
        control: GenerationControlHandle,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<LocalControlledGenerationSession<'a>> {
        self.inner
            .start_controlled_chat(prepared, caller_stop_sequences, control, emit)
            .map(|inner| LocalControlledGenerationSession { inner })
            .map_err(map_error)
    }
}

impl LocalControlledGenerationSession<'_> {
    /// Creates a complete independent child with explicit budgets and future
    /// choices. Its first record includes lineage and inherited semantic output.
    pub fn fork(
        &mut self,
        snapshot: &LocalGenerationSnapshot,
        options: GenerationBranchOptions,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<LocalGenerationBranch> {
        self.inner
            .fork(&snapshot.inner, options, emit)
            .map(|inner| LocalGenerationBranch { inner })
            .map_err(map_error)
    }
    /// Exchanges every native and facade state owner with an inactive branch.
    /// No prediction, random draw, copy or output replay occurs.
    pub fn exchange(
        &mut self,
        branch: &mut LocalGenerationBranch,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<()> {
        self.inner
            .exchange(&mut branch.inner, emit)
            .map_err(map_error)
    }
    /// Support for the actual loaded execution and complete semantic state.
    pub fn capabilities(&self) -> ExecutionControlCapabilities {
        self.inner.capabilities()
    }
    /// Current prepared, paused, running or terminal state.
    pub fn status(&self) -> GenerationStatus {
        self.inner.status()
    }
    /// Next absolute prediction; zero denotes prompt prefill.
    pub fn next_prediction(&self) -> u64 {
        self.inner.next_prediction()
    }
    /// Canonical committed history, including terminal special tokens.
    pub fn token_ids(&self) -> &[u32] {
        self.inner.token_ids()
    }
    /// Ordinary terminal reason, absent while resumable.
    pub fn finish_reason(&self) -> Option<FinishReason> {
        self.inner.finish_reason()
    }
    /// Thread-safe pause and permanent cancellation requests for this run.
    pub fn control_handle(&self) -> GenerationControlHandle {
        self.inner.control_handle()
    }
    /// Cumulative bounded transport consumption; restore never refunds it.
    pub fn emitted_bytes(&self) -> u64 {
        self.inner.emitted_bytes()
    }
    /// Current consumer journal checkpoint; this does not capture execution state.
    pub fn output_checkpoint(&self) -> GenerationOutputCheckpoint {
        self.inner.output_checkpoint()
    }
    /// Cumulative copy usage and current snapshot/branch retention.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.inner.snapshot_usage()
    }
    /// Complete mutable semantic/history storage, when its estimate is known.
    pub fn semantic_snapshot_bytes(&self) -> Option<u64> {
        self.inner.semantic_snapshot_bytes()
    }
    /// Advances at most one prediction, completing and delivering its records.
    pub fn step(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus> {
        self.inner.step(emit).map_err(map_error)
    }
    /// Pauses at the current completed boundary without flushing partial text.
    pub fn pause(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<()> {
        self.inner.pause(emit).map_err(map_error)
    }
    /// Advances until pause, cancellation or a normal terminal outcome.
    pub fn run(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus> {
        self.inner.run(emit).map_err(map_error)
    }
    /// Acknowledges a pause and continues the same ordinary run.
    pub fn resume(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus> {
        self.inner.resume(emit).map_err(map_error)
    }
    /// Permanently cancels; cancellation is never interpreted as a pause.
    pub fn cancel(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<GenerationStatus> {
        self.inner.cancel(emit).map_err(map_error)
    }
    /// Validates a canonical ID and restricts the next ordinary sampling decision.
    pub fn force_next_token(&mut self, token: u32) -> Result<()> {
        self.inner.force_next_token(token).map_err(map_error)
    }
    /// Clears a pending choice without advancing model, sampler or semantics.
    pub fn clear_forced_token(&mut self) -> Result<bool> {
        self.inner.clear_forced_token().map_err(map_error)
    }
    /// Canonical choice waiting for commitment.
    pub fn pending_forced_token(&self) -> Option<u32> {
        self.inner.pending_forced_token()
    }
    /// Side-effect-free temperature and RNG compatibility facts.
    pub fn sampling_state(&mut self) -> Result<SamplingStateFacts> {
        self.inner.sampling_state().map_err(map_error)
    }
    /// Changes only future temperature/randomness and emits bounded provenance.
    pub fn override_sampling(
        &mut self,
        request: SamplingOverride,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<()> {
        self.inner
            .override_sampling(request, emit)
            .map_err(map_error)
    }
    /// Configures non-resettable limits after checking every required component.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<()> {
        self.inner.enable_snapshots(limits).map_err(map_error)
    }
    /// Deeply saves the current quiescent continuation, with bounded attribution.
    pub fn snapshot(
        &mut self,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<LocalGenerationSnapshot> {
        self.inner
            .snapshot(emit)
            .map(|inner| LocalGenerationSnapshot { inner })
            .map_err(map_error)
    }
    /// Restores its exact source run; output sequence and consumed budgets remain
    /// monotone. The callback receives the prefix checkpoint to reconcile.
    pub fn restore(
        &mut self,
        snapshot: &LocalGenerationSnapshot,
        emit: impl FnMut(ControlledGenerationRecord) -> ControlFlow<()>,
    ) -> Result<()> {
        self.inner.restore(&snapshot.inner, emit).map_err(map_error)
    }
}

fn map_error(
    error: ControlledGenerationError<eredu_backend_mlx::backend::error::Error>,
) -> ControlledGenerationError<LocalBackendError> {
    use eredu_core::{
        ControlledTextGenerationError as Generation, TextContinuationError as Continuation,
    };
    use eredu_runtime::execution_control::{
        SamplingOverrideError as Sampling, TextSnapshotError as Snapshot,
    };
    use ControlledGenerationError as Control;
    let native = |error| LocalBackendError::new("execution control", error);
    match error {
        Control::Generation(error) => Control::Generation(super::map_prepared_chat_error(error)),
        Control::Choice(error) => Control::Choice(error),
        Control::Control(error) => Control::Control(error),
        Control::Capture(error) => Control::Capture(error),
        Control::Sampling(error) => Control::Sampling(match error {
            Sampling::Invalid(reason) => Sampling::Invalid(reason),
            Sampling::Backend(error) => Sampling::Backend(native(error)),
        }),
        Control::Continuation(error) => Control::Continuation(match error {
            Continuation::IncompatibleDriver => Continuation::IncompatibleDriver,
            Continuation::Failed => Continuation::Failed,
            Continuation::NotQuiescent => Continuation::NotQuiescent,
            Continuation::Generation(Generation::Backend(error)) => {
                Continuation::Generation(Generation::Backend(native(error)))
            }
            Continuation::Generation(Generation::Controller(error)) => {
                Continuation::Generation(Generation::Controller(error))
            }
        }),
        Control::Snapshot(error) => Control::Snapshot(match error {
            Snapshot::Host(error) => Snapshot::Host(error),
            Snapshot::Controller(error) => Snapshot::Controller(error),
            Snapshot::Backend(error) => Snapshot::Backend(native(error)),
            Snapshot::Capture(error) => Snapshot::Capture(error),
            Snapshot::Control(error) => Snapshot::Control(error),
            Snapshot::IncompatibleRun => Snapshot::IncompatibleRun,
            Snapshot::InconsistentState => Snapshot::InconsistentState,
            Snapshot::Unsupported(reason) => Snapshot::Unsupported(reason),
        }),
    }
}
