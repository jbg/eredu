//! Portable discovery and resource contracts for completed-token execution control.

use crate::generation::GenerationCancellationToken;
use serde::{Deserialize, Serialize};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Native persistent-state mechanisms for ordinary text generation.
///
/// These operations cover the model's complete mutable execution state and
/// native metadata, not the sampler, pending input or facade semantic state.
/// Portable composition must reserve known costs before copying and combine all
/// of those owners before advertising a complete generation snapshot. Backends
/// cooperate with their existing submission authority and recovery owner.
pub trait NativeTextStateBackend: crate::TextGenerationBackend {
    /// Independently writable, opaque, in-process native state slot. Sharing
    /// immutable weights is allowed; cloning mutable storage handles is not.
    type NativeTextState;

    /// Side-effect-free support facts for this exact loaded execution. This is
    /// a primitive report, not full generation-snapshot capability discovery.
    fn native_text_state_support(runtime: &crate::ModelRuntime<Self>) -> ControlSupport;

    /// Estimates copying installed state (`None`) or a compatible saved slot.
    /// Unknown costs remain explicit and must be rejected before copying.
    fn estimate_native_text_state(
        runtime: &crate::ModelRuntime<Self>,
        saved: Option<&Self::NativeTextState>,
    ) -> Result<Option<SnapshotEstimate>, Self::Error>;

    /// Conservative additional logical retention through at most this many new
    /// input tokens, starting from the saved state. Includes cache capacity growth
    /// and initially absent recurrent/convolution components, without executing
    /// input. Unknown growth disables runnable branches, not immutable snapshots.
    fn estimate_native_text_growth(
        _runtime: &crate::ModelRuntime<Self>,
        _saved: &Self::NativeTextState,
        _additional_input_tokens: u64,
    ) -> Result<Option<u64>, Self::Error> {
        Ok(None)
    }

    /// Captures independent state after portable resource reservation. Returns
    /// only after successful exact native completion. Failure never changes
    /// the logical source state; unresolved work remains retained and fenced.
    fn capture_native_text_state(
        runtime: &mut crate::ModelRuntime<Self>,
    ) -> Result<Self::NativeTextState, Self::Error>;

    /// Copies a reusable saved slot without installing it or replaying input.
    /// The source remains unchanged even if allocation or completion fails.
    fn copy_native_text_state(
        runtime: &mut crate::ModelRuntime<Self>,
        saved: &Self::NativeTextState,
    ) -> Result<Self::NativeTextState, Self::Error>;

    /// Validates exact executable identity, geometry and safe native boundary
    /// before any state mutation. A different compatible-looking model fails.
    fn validate_native_text_state(
        runtime: &crate::ModelRuntime<Self>,
        saved: &Self::NativeTextState,
    ) -> Result<(), Self::Error>;

    /// Atomically exchanges installed and saved state, without copying tensor
    /// data. The old installed state is returned in `slot`. On error neither
    /// logical state changes; unresolved native failure may fence the engine.
    fn exchange_native_text_state(
        runtime: &mut crate::ModelRuntime<Self>,
        slot: &mut Self::NativeTextState,
    ) -> Result<(), Self::Error>;
}

/// Version of execution-control metadata and lifecycle records.
pub const EXECUTION_CONTROL_SCHEMA_VERSION: u32 = 1;

/// Observable lifecycle of one controllable ordinary generation session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationStatus {
    /// Prepared, with no model prediction yet committed.
    Prepared,
    /// Quiescent, resumable boundary; no further work advances without a request.
    Paused,
    /// A prediction or its associated delivery is being completed.
    Running,
    /// Normal terminal outcome; restoration preserves that outcome.
    Completed,
    /// Cooperatively cancelled, distinct from a resumable pause.
    Cancelled,
    /// Failed, including unresolved native work; not a snapshot boundary.
    Failed,
}

/// Support for one operation on the actual selected execution configuration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "support", rename_all = "snake_case")]
pub enum ControlSupport {
    /// Implemented with the scope and guarantees in the enclosing report.
    Supported,
    /// Rejected before work begins.
    Unsupported {
        /// Concrete missing mechanism or unsupported execution combination.
        reason: String,
    },
}

/// Mutable-state isolation promised by an opaque native in-process snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotIsolation {
    /// Mutable state uses independent storage; immutable weights may be shared.
    DeepCopy,
    /// Storage is shared only until writing, with independent semantic state.
    CopyOnWrite,
}

/// Exact loaded-session execution-control capabilities. Native handles and
/// estimators never enter this serializable report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionControlCapabilities {
    /// Schema version.
    pub schema_version: u32,
    /// At most one ordinary committed prediction per step.
    pub step: ControlSupport,
    /// Pause and resume at a completed, delivered token boundary.
    pub pause_resume: ControlSupport,
    /// Complete in-process snapshot creation.
    pub snapshot: ControlSupport,
    /// Same-session state restoration from an immutable reusable snapshot.
    pub restore: ControlSupport,
    /// Isolated child creation without prompt replay or weight reload.
    pub fork: ControlSupport,
    /// Prospective canonical token restriction through ordinary sampling.
    pub force_next_token: ControlSupport,
    /// Prospective temperature changes and explicit native RNG reseeding.
    pub sampling_overrides: ControlSupport,
    /// Native snapshot isolation, absent when snapshots are unsupported.
    pub isolation: Option<SnapshotIsolation>,
    /// Determinism, scope, execution restrictions and compatibility conditions.
    pub conditions: Vec<String>,
}

impl ExecutionControlCapabilities {
    /// Explicitly rejects all operations for a configuration lacking support.
    pub fn unsupported(reason: impl Into<String>) -> Self {
        let support = ControlSupport::Unsupported {
            reason: reason.into(),
        };
        Self {
            schema_version: EXECUTION_CONTROL_SCHEMA_VERSION,
            step: support.clone(),
            pause_resume: support.clone(),
            snapshot: support.clone(),
            restore: support.clone(),
            fork: support.clone(),
            force_next_token: support.clone(),
            sampling_overrides: support,
            isolation: None,
            conditions: Vec::new(),
        }
    }
}

/// Thread-safe requests for a worker whose native session can remain thread-affine.
/// Pausing is sticky until the owning worker explicitly resumes. Cancellation
/// retains the ordinary permanent cancellation-token semantics.
#[derive(Debug, Clone, Default)]
pub struct GenerationControlHandle {
    pause: Arc<AtomicBool>,
    cancellation: GenerationCancellationToken,
}

impl GenerationControlHandle {
    /// Creates a handle using the ordinary caller's cancellation token.
    pub fn new(cancellation: GenerationCancellationToken) -> Self {
        Self {
            pause: Arc::new(AtomicBool::new(false)),
            cancellation,
        }
    }
    /// Requests pause at the next successful completed-token boundary.
    pub fn request_pause(&self) {
        self.pause.store(true, Ordering::Release);
    }
    /// Whether a pause request is pending. This does not itself prove completion.
    pub fn pause_requested(&self) -> bool {
        self.pause.load(Ordering::Acquire)
    }
    /// Acknowledges an explicit resume before checking for a new pause request.
    pub fn acknowledge_resume(&self) {
        self.pause.store(false, Ordering::Release);
    }
    /// Permanently cancels this run, without reinterpreting cancellation as pause.
    pub fn cancel(&self) {
        self.cancellation.cancel();
    }
    /// Ordinary cancellation token, shared with existing generation consumers.
    pub fn cancellation(&self) -> &GenerationCancellationToken {
        &self.cancellation
    }
}

/// Logical storage facts supplied without allocating or evaluating native state.
/// Values include conservative copy-on-write allowances where applicable. They
/// do not promise a physical allocator/private-workspace ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotEstimate {
    /// Incremental logical state retained for the lifetime of this object.
    pub retained_bytes: u64,
    /// Logical copying/materialization allowance charged cumulatively per attempt.
    pub copy_bytes: u64,
}

/// Independently bounded snapshot and child-state resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotLimits {
    /// Maximum simultaneously retained snapshot objects, including live handles.
    pub max_snapshots: u64,
    /// Maximum simultaneously retained child states in this budget owner.
    pub max_branches: u64,
    /// Maximum sum of retained logical state estimates.
    pub retained_bytes: u64,
    /// Cumulative logical copying allowance; restore and failed attempts never refund it.
    pub cumulative_copy_bytes: u64,
}

/// Kind of state retained under a snapshot resource reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SnapshotResourceKind {
    /// Immutable reusable saved state.
    Snapshot,
    /// Independently mutable child execution state.
    Branch,
    /// Provisional replacement during restoration; no additional persistent object.
    Restore,
}

/// Observed logical snapshot accounting, separate from capture/transport limits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotUsage {
    /// Retained snapshot count.
    pub snapshots: u64,
    /// Retained branch count.
    pub branches: u64,
    /// Sum of retained logical estimates, including provisional copies.
    pub retained_bytes: u64,
    /// Monotone copying allowance consumed by all admitted attempts.
    pub cumulative_copy_bytes: u64,
}

/// Portable invalid-state or budget failure, before a native operation is called.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ExecutionControlError {
    /// The current state does not permit the requested lifecycle edge.
    #[error("invalid generation transition from {from:?} to {to:?}")]
    Transition {
        /// Current lifecycle state.
        from: GenerationStatus,
        /// Requested lifecycle state.
        to: GenerationStatus,
    },
    /// A complete estimate is required before copying or retaining state.
    #[error("complete snapshot resource estimate is unavailable")]
    UnknownEstimate,
    /// Checked arithmetic failed.
    #[error("execution-control accounting overflow")]
    Overflow,
    /// The explicitly admitted logical resource limit was exceeded.
    #[error("snapshot resource limit exceeded: {0}")]
    Limit(&'static str),
}
