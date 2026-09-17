//! Ownership observations for native work accepted during a host operation.

use std::{marker::PhantomData, rc::Rc};

pub use crate::allocation_retention::scope::{
    PreparedSubmissionScopeOwner, SubmissionScopeOwnerCause, SubmissionScopeOwnerError,
    SubmissionScopeOwnerLayout,
};

use crate::{
    error::Exception,
    utils::{guard::Guarded, runtime_lock},
};

/// Try to retire terminal submission resources without waiting for the runtime.
///
/// The callback runs with the native runtime lock held and housekeeping
/// suppressed, so dropping native handles cannot block on another runtime user
/// or recursively invoke application cleanup. If the lock is unavailable, the
/// callback is not invoked and `None` is returned; the caller must retain its
/// resources. The callback itself must not wait for workers or other locks.
/// It must only release resources with independently established terminal proof.
pub fn try_with_submission_retirement<R>(retire: impl FnOnce() -> R) -> Option<R> {
    runtime_lock::try_retire(retire)
}

/// Result of one ordinary completed-record retirement attempt.
///
/// This reports traversal, not native completion, an empty registry, released
/// backing storage, or available memory. Retained aliases still own their charge.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SubmissionRetirement {
    /// Inspected the full entry-time registry and destroyed only already settled
    /// records owned by this thread. Other records and independent aliases remain.
    CompleteSnapshot,
    /// The runtime or registry lock was unavailable; this call retired nothing.
    Busy,
}

impl SubmissionRetirement {
    fn from_raw(raw: safemlx_sys::mlx_submission_retirement) -> Result<Self, Exception> {
        match raw {
            safemlx_sys::mlx_submission_retirement__MLX_SUBMISSION_RETIREMENT_COMPLETE_SNAPSHOT => {
                Ok(Self::CompleteSnapshot)
            }
            safemlx_sys::mlx_submission_retirement__MLX_SUBMISSION_RETIREMENT_BUSY => {
                Ok(Self::Busy)
            }
            _ => Err(Exception::custom(
                "unknown native submission retirement result",
            )),
        }
    }
}

/// Try one ordinary owner-thread pass over completed native submission records.
///
/// Both lock acquisitions are nonblocking and there is no polling, evaluation,
/// native-work wait, retry, tensor allocation, or runtime housekeeping. Record
/// destruction happens outside the registry lock under the native runtime lock;
/// arbitrary primitive destructors can take time, so this is not bounded scope
/// progress. Initial registry/error bookkeeping is not a tensor allocation.
///
/// `CompleteSnapshot` does not certify any scope or prove all backing references
/// retired. Callers must establish their own exact completion and retain all
/// unresolved resources on `Busy` or error. The pass neither releases caller
/// roots nor reports/refunds memory. Queued allocation-owned Rust resources are
/// still reclaimed separately by [`crate::reclaim_allocation_owners`]. Native
/// errors retain the existing [`Exception`] rather than becoming `Busy`.
#[track_caller]
pub fn try_retire_completed_submissions() -> Result<SubmissionRetirement, Exception> {
    // Only this closed operation runs under the private retirement lock. It
    // exposes no arbitrary callback or authority to release unresolved records.
    runtime_lock::try_retire(|| {
        let raw = u32::try_from_op(|out| {
            // SAFETY: out points to the guard's initialized C-compatible enum
            // scalar. Native code writes it only on success and retains no
            // pointer; the runtime lock serializes the native error channel.
            unsafe { safemlx_sys::mlx_submission_retire_completed(out) }
        })?;
        SubmissionRetirement::from_raw(raw)
    })
    .unwrap_or(Ok(SubmissionRetirement::Busy))
}

/// Whether ordinary host code may run arbitrary retired-resource destructors.
///
/// This is false during unwinding or while this thread owns the native runtime
/// lock. Such destructors may wait for a worker which needs that lock. This is
/// not terminal-work evidence and must not be used to authorize resource release.
pub fn can_reclaim_submission_resources() -> bool {
    runtime_lock::can_reclaim_submission_resources()
}

/// Whether native work accepted by a submission scope has stopped.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SubmissionActivity {
    /// The scope accepted no native work.
    None,
    /// Some accepted work has not been proved terminal.
    Pending,
    /// Every accepted operation has reached a terminal state.
    Terminal,
    /// The observation itself was unavailable; resources must remain owned.
    Unobservable,
}

/// Native lifetime evidence, independent of an operation's returned error.
///
/// An execution failure is not evidence that its work has stopped. Conversely,
/// terminal work may have failed and left model state unsuitable for reuse.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionStatus {
    activity: SubmissionActivity,
    failed: bool,
    blocked: bool,
}

impl SubmissionStatus {
    /// The accepted work's lifetime state.
    pub const fn activity(self) -> SubmissionActivity {
        self.activity
    }

    /// Whether releasing this scope's retained resources is safe.
    pub const fn is_settled(self) -> bool {
        matches!(
            self.activity,
            SubmissionActivity::None | SubmissionActivity::Terminal
        )
    }

    /// Whether work was accepted, or an observation could not establish otherwise.
    pub const fn has_work(self) -> bool {
        !matches!(self.activity, SubmissionActivity::None)
    }

    /// Whether a native submission or execution failure was recorded.
    pub const fn failed(self) -> bool {
        self.failed
    }

    /// Whether native recovery prevents reuse of affected execution resources.
    pub const fn blocked(self) -> bool {
        self.blocked
    }

    const fn unobservable() -> Self {
        Self {
            activity: SubmissionActivity::Unobservable,
            failed: true,
            blocked: true,
        }
    }

    fn from_raw(raw: safemlx_sys::mlx_submission_status) -> Self {
        let activity = match raw.activity {
            safemlx_sys::mlx_submission_activity__MLX_SUBMISSION_ACTIVITY_NONE => {
                SubmissionActivity::None
            }
            safemlx_sys::mlx_submission_activity__MLX_SUBMISSION_ACTIVITY_PENDING => {
                SubmissionActivity::Pending
            }
            safemlx_sys::mlx_submission_activity__MLX_SUBMISSION_ACTIVITY_TERMINAL => {
                SubmissionActivity::Terminal
            }
            _ => return Self::unobservable(),
        };
        Self {
            activity,
            failed: raw.failed,
            blocked: raw.blocked,
        }
    }
}

/// Cold status of entered records, independent of structural child scope lifetime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SubmissionRecordActivity {
    /// No entered record lacks its existing terminal evidence. Live child scopes,
    /// retained graphs, backing aliases and future submissions may still exist.
    Quiescent,
    /// At least one record has not reached its existing terminal transition.
    Pending,
    /// The exact scope observation was unavailable; retain prior ownership.
    Unobservable,
}

/// Point-in-time native record facts for one scope and its durable descendants.
///
/// These facts cannot certify scope lifetime, free resources, authorize a write,
/// or supply a budget. In particular, quiescent records with a live empty child
/// leave the ordinary [`SubmissionStatus`] pending. A quiescent failed/blocked
/// scope is not healthy. Independent ancestors, siblings and earlier detached
/// preparation scopes must be observed through their own original handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SubmissionRecordStatus {
    activity: SubmissionRecordActivity,
    failed: bool,
    blocked: bool,
}
impl SubmissionRecordStatus {
    /// Whether records remain unresolved; this excludes structural children.
    pub const fn activity(self) -> SubmissionRecordActivity {
        self.activity
    }
    /// Sticky native failure, also true when observation is unavailable.
    pub const fn failed(self) -> bool {
        self.failed
    }
    /// Sticky blocked recovery, also true when observation is unavailable.
    pub const fn blocked(self) -> bool {
        self.blocked
    }
    const fn unobservable() -> Self {
        Self {
            activity: SubmissionRecordActivity::Unobservable,
            failed: true,
            blocked: true,
        }
    }
    fn from_raw(raw: safemlx_sys::mlx_submission_record_status) -> Self {
        Self {
            activity: if raw.pending {
                SubmissionRecordActivity::Pending
            } else {
                SubmissionRecordActivity::Quiescent
            },
            failed: raw.failed,
            blocked: raw.blocked,
        }
    }
}

/// A nonblocking scope attempt could not establish native recovery ownership.
///
/// Neither variant means work completed. No scope or work was accepted by the
/// failed attempt; callers must preserve any independently owned earlier work.
#[derive(Debug)]
#[non_exhaustive]
pub enum SubmissionScopeBeginError {
    /// Another thread owns the native runtime. No native call was made.
    Busy,
    /// The native constructor rejected creation before accepting work.
    Native(Exception),
}

impl std::fmt::Display for SubmissionScopeBeginError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Busy => "native runtime is busy; no submission scope was entered",
            Self::Native(_) => "native submission scope could not be created",
        })
    }
}

impl std::error::Error for SubmissionScopeBeginError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Busy => None,
            Self::Native(error) => Some(error),
        }
    }
}

/// Fixed result of scoped observation, never an allocation or completion grant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScopedSubmissionProgress {
    /// Only this scope's audited record/event frontiers were observed.
    Observed,
    /// The exact pending receipt needs a separately funded producer operation.
    NeedsFundedProgress,
    /// This scope or one of its concrete side-effect hooks cannot be observed.
    Unobservable,
    /// No observation began because the runtime or registry is borrowed.
    Busy,
}
impl ScopedSubmissionProgress {
    fn from_raw(value: u32) -> Self {
        match value {
            0 => Self::Observed,
            1 => Self::NeedsFundedProgress,
            3 => Self::Busy,
            _ => Self::Unobservable,
        }
    }
}

/// A thread-affine owner observing all native submissions in one host operation.
///
/// Begin a scope before the first potentially eager operation, including input
/// preparation. Nested scopes are supported. Sealing ends capture but retains
/// lifetime evidence; it neither waits nor cancels work. Errors may be converted
/// to text or discarded without discarding that evidence.
///
/// Dropping a scope does not wait. Native recovery retains unresolved resources
/// independently, including when a caller drops an error or exits its thread.
/// Consumers must separately retain application-owned resources and exclude
/// conflicting mutation until [`SubmissionStatus::is_settled`] is true.
///
/// This type is intentionally neither `Send` nor `Sync`, matching MLX stream
/// ownership. No process-termination recovery policy is imposed.
pub struct SubmissionScope {
    raw: safemlx_sys::mlx_submission_scope,
    sealed: bool,
    observation_failed: bool,
    _thread_affine: PhantomData<Rc<()>>,
}

impl std::fmt::Debug for SubmissionScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubmissionScope")
            .field("sealed", &self.sealed)
            .field("status", &self.status())
            .finish()
    }
}

impl SubmissionScope {
    /// Allocate and enter a recovery scope before submitting native work.
    pub fn begin() -> Result<Self, Exception> {
        let guard = runtime_lock::enter();
        Self::begin_under_runtime(&guard)
    }

    /// Try to allocate and enter a scope without waiting for the runtime lock.
    ///
    /// Busy makes no native call and allocates no scope or error payload. The
    /// successful path performs ordinary host allocation for native bookkeeping,
    /// but neither polls, evaluates, waits, retries, reclaims nor invokes/registers
    /// runtime housekeeping. Allocation and destructor latency is not bounded.
    /// Native registry publication does not wait for another initializer.
    ///
    /// Success establishes a new empty capture scope, not admission, completion
    /// of prior work, or permission to submit an unretained failure agreement.
    pub fn try_begin() -> Result<Self, SubmissionScopeBeginError> {
        let guard =
            runtime_lock::try_enter_for_recovery().ok_or(SubmissionScopeBeginError::Busy)?;
        Self::begin_under_runtime(&guard).map_err(SubmissionScopeBeginError::Native)
    }

    /// Begin with one constructor-owned payload, preserving it on every failure.
    ///
    /// The nonblocking runtime loan skips housekeeping. Native construction may
    /// allocate its Scope and initial registry/thread bookkeeping. Success consumes
    /// the preparation once; scope/record readiness does not release its owner.
    /// The final intrusive native reference deletes the Scope before queueing the
    /// owner for unlocked host reclamation. T must not retain this Scope, any
    /// descendant, or a record that retains it, directly or indirectly.
    ///
    /// This standalone primitive supplies no original budget, bounded population,
    /// source or tensor custody, native completion, or managed admission. There is
    /// no attachment to an already-created Scope. Plain begin remains unchanged.
    pub fn try_begin_retaining<T: Send + 'static>(
        owner: PreparedSubmissionScopeOwner<T>,
    ) -> Result<Self, SubmissionScopeOwnerError<PreparedSubmissionScopeOwner<T>>> {
        let raw = owner.begin()?;
        Ok(Self {
            raw,
            sealed: false,
            observation_failed: false,
            _thread_affine: PhantomData,
        })
    }

    /// Enter independently paid control work beneath the exact active original
    /// observer. Both arenas must be supplied by the prepared owner.
    ///
    /// This is a construction primitive, not admission. Native validation
    /// rejects stale/foreign observers and active capture/dispatch/worker loans.
    /// The child's durable ancestry retains the parent through completion and
    /// failure; sealing restores the previous active scope. Backing still must
    /// match the parent's immutable origin. Default nesting remains strict.
    pub fn try_begin_original_child<T: Send + 'static>(
        owner: PreparedSubmissionScopeOwner<T>,
        parent: &OriginalScopeObserver,
    ) -> Result<Self, SubmissionScopeOwnerError<PreparedSubmissionScopeOwner<T>>> {
        let raw = owner.begin_original_child(parent)?;
        Ok(Self {
            raw,
            sealed: false,
            observation_failed: false,
            _thread_affine: PhantomData,
        })
    }

    fn begin_under_runtime(_guard: &runtime_lock::RuntimeLockGuard) -> Result<Self, Exception> {
        let mut raw = safemlx_sys::mlx_submission_scope {
            ctx: std::ptr::null_mut(),
        };
        // The native constructor accepts no work and does not use the textual
        // error channel, which might itself allocate during allocation failure.
        // SAFETY: the borrowed guard holds the runtime. raw is an initialized
        // null output handle; C publishes only a scope owned by this thread.
        let status = unsafe { safemlx_sys::mlx_submission_scope_new(&mut raw) };
        if status != 0 || raw.ctx.is_null() {
            if !raw.ctx.is_null() {
                unsafe { safemlx_sys::mlx_submission_scope_free(raw) };
            }
            return Err(Exception::custom(
                "could not prepare native submission recovery before execution",
            ));
        }
        Ok(Self {
            raw,
            sealed: false,
            observation_failed: false,
            _thread_affine: PhantomData,
        })
    }

    /// Select observation-only progress on this genuine current empty scope.
    /// No work, new owner or source authority is manufactured. Existing work,
    /// wrong thread/current scope or exhausted identity gives a fixed refusal.
    pub fn enable_scoped_observation(&mut self) -> Result<(), ScopedSubmissionProgress> {
        let result = unsafe { safemlx_sys::mlx_submission_scope_enable_scoped(self.raw) };
        match ScopedSubmissionProgress::from_raw(result) {
            ScopedSubmissionProgress::Observed => Ok(()),
            other => Err(other),
        }
    }

    /// Observe only this actual scope and its durable descendants. This never
    /// commits a native encoder or retires unrelated records. The fixed result
    /// is separate from the independently retained lifetime status.
    pub fn progress_scoped(&self) -> (ScopedSubmissionProgress, SubmissionStatus) {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return (ScopedSubmissionProgress::Busy, self.status());
        };
        let mut raw = safemlx_sys::mlx_submission_status {
            activity: safemlx_sys::mlx_submission_activity__MLX_SUBMISSION_ACTIVITY_PENDING,
            failed: false,
            blocked: true,
        };
        let result =
            unsafe { safemlx_sys::mlx_submission_scope_progress_scoped(&mut raw, self.raw) };
        (
            ScopedSubmissionProgress::from_raw(result),
            SubmissionStatus::from_raw(raw),
        )
    }

    pub(crate) fn raw(&self) -> safemlx_sys::mlx_submission_scope {
        self.raw
    }

    /// Stop capturing child submissions without waiting or discarding ownership.
    pub fn seal(&mut self) {
        if !self.sealed {
            // Native sealing only changes this thread's scope bookkeeping.
            // In particular, it must not acquire the process-wide runtime lock.
            let status = unsafe { safemlx_sys::mlx_submission_scope_seal(self.raw) };
            if status == 0 {
                self.sealed = true;
            } else {
                self.observation_failed = true;
            }
        }
    }

    /// Read lifetime evidence without waiting, allocating, or clearing failures.
    pub fn status(&self) -> SubmissionStatus {
        self.observe(false)
    }

    /// Observe entered native records without locks, allocation or housekeeping.
    ///
    /// Counts this scope and its durable descendants, including records entered
    /// before sealing and descendants entered after an outer scope was sealed.
    /// A live empty child alone is excluded. No CPU/GPU/event/communication poll,
    /// progress, evaluation, retry, wait, resource destruction or error formatting
    /// occurs. Terminal native work stays conservatively pending until the
    /// existing record-progress path publishes its terminal transition.
    ///
    /// This is neither completion authority nor exclusion of future submission.
    /// Query the real ancestor/preparation handles covering the intended work;
    /// a newly created scope cannot cover older unrelated work.
    pub fn accepted_records(&self) -> SubmissionRecordStatus {
        if self.observation_failed {
            return SubmissionRecordStatus::unobservable();
        }
        let mut raw = safemlx_sys::mlx_submission_record_status {
            pending: true,
            failed: true,
            blocked: true,
        };
        // SAFETY: this thread-affine live handle belongs to the calling thread;
        // out is initialized and borrowed only for the nonallocating C query.
        let result = unsafe { safemlx_sys::mlx_submission_scope_query_records(&mut raw, self.raw) };
        if result == 0 {
            SubmissionRecordStatus::from_raw(raw)
        } else {
            SubmissionRecordStatus::unobservable()
        }
    }

    /// Actual bytes of the native scope object, excluding allocator/registry and
    /// retained record/graph payloads. This cold size fact performs no allocation,
    /// initialization or locking and is not a reservation or an accounting grant.
    pub fn native_control_bytes() -> usize {
        // SAFETY: the C function returns only sizeof(Scope), without pointers or
        // runtime state. Supported native hosts have the same size_t ABI as usize.
        unsafe { safemlx_sys::mlx_submission_scope_control_bytes() }
    }

    /// Make one nonblocking attempt to advance native completion evidence.
    ///
    /// If another host call owns the runtime lock, only existing evidence is
    /// read. Contention or an observation error never establishes completion.
    /// Terminal graph owners are reclaimed by ordinary native operations, not
    /// by this bounded observation call.
    pub fn progress(&self) -> SubmissionStatus {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return self.status();
        };
        self.observe(true)
    }

    fn observe(&self, progress: bool) -> SubmissionStatus {
        if self.observation_failed {
            return SubmissionStatus::unobservable();
        }
        let mut raw = safemlx_sys::mlx_submission_status {
            activity: safemlx_sys::mlx_submission_activity__MLX_SUBMISSION_ACTIVITY_PENDING,
            failed: false,
            blocked: true,
        };
        let result = unsafe {
            if progress {
                safemlx_sys::mlx_submission_scope_progress(&mut raw, self.raw)
            } else {
                safemlx_sys::mlx_submission_scope_query(&mut raw, self.raw)
            }
        };
        if result == 0 {
            SubmissionStatus::from_raw(raw)
        } else {
            SubmissionStatus::unobservable()
        }
    }
}

impl Drop for SubmissionScope {
    fn drop(&mut self) {
        self.seal();
        // Free retires pending native ownership into a preallocated quarantine;
        // it never waits, formats an error, or destroys unresolved resources.
        unsafe { safemlx_sys::mlx_submission_scope_free(self.raw) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{transforms::async_eval_with_event, Array, Device, DeviceType, Stream};

    fn settle(scope: &SubmissionScope) -> SubmissionStatus {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let status = scope.progress();
            if status.is_settled() {
                return status;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "native scope did not settle: {status:?}"
            );
            std::thread::yield_now();
        }
    }

    #[test]
    fn native_cpu_scope_tracks_submitted_work_through_terminal_completion() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let mut scope = SubmissionScope::begin().unwrap();
        let values = Array::ones::<f32>(&[32, 32], &stream).unwrap();
        let output = values.square(&stream).unwrap();
        let completion = async_eval_with_event([&output]).unwrap();
        scope.seal();
        assert!(scope.status().has_work());
        completion.synchronize().unwrap();
        let status = settle(&scope);
        assert_eq!(status.activity(), SubmissionActivity::Terminal);
        assert!(!status.failed());
        assert!(!status.blocked());
        assert_eq!(output.evaluated().unwrap().as_slice::<f32>(), &[1.0; 1024]);
    }

    #[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
    #[test]
    fn successive_gpu_stream_waits_each_reach_terminal_completion() {
        let device = Device::new(DeviceType::Gpu, 0);
        let producer = Stream::new_with_device(&device);
        let consumer = Stream::new_with_device(&device);
        for _ in 0..3 {
            let mut scope = SubmissionScope::begin().unwrap();
            let value = Array::ones::<f32>(&[8], &producer).unwrap();
            let completion = async_eval_with_event([&value]).unwrap();
            completion.synchronize().unwrap();
            completion.wait_on(&consumer).unwrap();
            scope.seal();
            let status = settle(&scope);
            assert_eq!(status.activity(), SubmissionActivity::Terminal);
            assert!(!status.failed());
            assert!(!status.blocked());
        }
    }

    #[test]
    fn sealed_outer_scope_waits_for_its_active_child_before_releasing_ownership() {
        let mut outer = SubmissionScope::begin().unwrap();
        let mut inner = SubmissionScope::begin().unwrap();
        outer.seal();
        assert!(!outer.progress().is_settled());
        assert!(inner.status().is_settled());
        inner.seal();
        assert_eq!(settle(&outer).activity(), SubmissionActivity::None);
    }

    #[test]
    fn dropping_outer_scope_does_not_invalidate_a_live_child_submission() {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let outer = SubmissionScope::begin().unwrap();
        let mut inner = SubmissionScope::begin().unwrap();
        drop(outer);
        let value = Array::ones::<f32>(&[8], &stream).unwrap();
        let completion = async_eval_with_event([&value]).unwrap();
        inner.seal();
        completion.synchronize().unwrap();
        assert_eq!(settle(&inner).activity(), SubmissionActivity::Terminal);
        assert_eq!(value.evaluated().unwrap().as_slice::<f32>(), &[1.0; 8]);
    }

    #[test]
    fn unknown_native_activity_never_establishes_terminal_ownership() {
        let status = SubmissionStatus::from_raw(safemlx_sys::mlx_submission_status {
            activity: u32::MAX,
            failed: false,
            blocked: false,
        });
        assert_eq!(status.activity(), SubmissionActivity::Unobservable);
        assert!(!status.is_settled());
        assert!(status.failed());
        assert!(status.blocked());
    }

    #[test]
    fn only_explicit_none_or_terminal_evidence_is_settled() {
        for activity in [
            SubmissionActivity::None,
            SubmissionActivity::Pending,
            SubmissionActivity::Terminal,
            SubmissionActivity::Unobservable,
        ] {
            for failed in [false, true] {
                let status = SubmissionStatus {
                    activity,
                    failed,
                    blocked: false,
                };
                assert_eq!(
                    status.is_settled(),
                    matches!(
                        activity,
                        SubmissionActivity::None | SubmissionActivity::Terminal
                    )
                );
            }
        }
    }
}

#[cfg(test)]
mod retirement_tests;

#[cfg(test)]
#[path = "submission/try_begin_tests.rs"]
mod try_begin_tests;

#[cfg(test)]
mod record_tests;

/// Fixed native original-control refusal. No status carries completion authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum OriginalNativeControlError {
    /// There is no current Scope to configure.
    #[error("original native controls require an active scope")]
    MissingScope,
    /// This Scope is sealed, foreign, nonempty or not configured for work.
    #[error("original native scope is not configured for work")]
    InvalidScope,
    /// The one configuration attempt was already spent.
    #[error("original native controls were already attempted")]
    AlreadyConfigured,
    /// No same-scope original graph arena was supplied.
    #[error("original native controls require their Graph quota")]
    MissingGraph,
    /// No same-scope original record arena was supplied.
    #[error("original native controls require their Record quota")]
    MissingRecord,
    /// No original failure carrier was bound.
    #[error("original native controls require their failure carrier")]
    MissingFailure,
    /// Source, scope, arena or stream receipt differs.
    #[error("original native control domain differs")]
    ForeignDomain,
    /// The actual dispatch record lacks this prepared stream.
    #[error("original native task has no prepared stream receipt")]
    MissingReceipt,
    /// Stream worker preparation has not occurred.
    #[error("original native task has no prepared worker")]
    MissingWorker,
    /// Reading the prepared worker table would block.
    #[error("original native worker lookup is busy")]
    BusyWorker,
    /// This prepared worker has a prior failure.
    #[error("original native worker has failed")]
    FailedWorker,
    /// This worker cannot accept more work.
    #[error("original native worker is stopped or blocked")]
    StoppedWorker,
    /// Concurrent blocking won before the task acceptance.
    #[error("original CPU stream blocked during submission")]
    BlockedDuringSubmit,
    /// The physical worker's sequence counter cannot advance.
    #[error("original native worker sequence is exhausted")]
    SequenceExhausted,
    /// The actual native backend or shared-control ABI lacks this component.
    #[error("original native controls are unavailable for this native mechanism")]
    UnsupportedBackend,
    /// A previously erased callable cannot claim original inline storage.
    #[error("original native task requires its concrete callable")]
    ErasedCallable,
    /// The concrete control sum cannot be represented in the host address domain.
    #[error("original native control layout overflows")]
    LayoutOverflow,
    /// An unrecognized C status is retained without allocation or reinterpretation.
    #[error("invalid original native control status {0}")]
    InvalidStatus(u32),
}
impl OriginalNativeControlError {
    pub(crate) fn from_status(status: u32) -> Result<(), Self> {
        Err(match status {
            0 => return Ok(()),
            1 => Self::MissingScope,
            2 => Self::InvalidScope,
            3 => Self::AlreadyConfigured,
            4 => Self::MissingGraph,
            5 => Self::MissingRecord,
            6 => Self::MissingFailure,
            7 => Self::ForeignDomain,
            8 => Self::MissingReceipt,
            9 => Self::MissingWorker,
            10 => Self::BusyWorker,
            11 => Self::FailedWorker,
            12 => Self::StoppedWorker,
            13 => Self::BlockedDuringSubmit,
            14 => Self::SequenceExhausted,
            15 => Self::UnsupportedBackend,
            16 => Self::ErasedCallable,
            value => Self::InvalidStatus(value),
        })
    }
}

/// Native per-object facts; neither event counts nor complete graph fit are inferred.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OriginalNativeControlLayout {
    /// Size of the selected CPU/Metal Event implementation itself.
    pub event_object_bytes: usize,
    /// Actual supported C++ shared-control allocation including its implementation.
    pub event_shared_bytes: usize,
    /// Alignment of that physical shared-control allocation.
    pub event_shared_alignment: usize,
    /// Worst-alignment Graph extent for one such allocation, not a total event budget.
    pub event_graph_extent: usize,
    /// Accelerator shared-event objects (one for Metal, zero for a CPU counter).
    /// Constructor/destructor autorelease pools are separate transient platform objects;
    /// their native wrapper controls are included in fixed_control_bytes.
    /// CPU mutex/condition-variable inline storage is included in the object size.
    pub event_platform_objects: usize,
    /// Inline common task header; the concrete F remains part of TaskNode<F>.
    pub task_header_bytes: usize,
    /// Alignment of that inline header, not a universal callable alignment.
    pub task_header_alignment: usize,
    /// Concrete fixed transport/dispatch/retirement representations for this component.
    pub fixed_control_bytes: usize,
}
impl OriginalNativeControlLayout {
    /// Read concrete native layout without allocation, callback or device construction.
    pub fn inspect() -> Result<Self, OriginalNativeControlError> {
        let mut out =
            std::mem::MaybeUninit::<safemlx_sys::mlx_submission_native_control_layout>::uninit();
        // SAFETY: this fixed no-hooks query initializes all fields only on success.
        let status =
            unsafe { safemlx_sys::mlx_submission_native_control_layout_for(out.as_mut_ptr()) };
        OriginalNativeControlError::from_status(status)?;
        // SAFETY: successful native status guarantees complete initialization.
        let out = unsafe { out.assume_init() };
        let fixed_control_bytes = out
            .fixed_controls
            .checked_add(std::mem::size_of_val(&out))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<Result<Self, OriginalNativeControlError>>())
            })
            .and_then(|n| {
                n.checked_add(std::mem::size_of::<Result<(), OriginalNativeControlError>>())
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<OriginalNativeControlError>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<u32>()))
            .ok_or(OriginalNativeControlError::LayoutOverflow)?;
        Ok(Self {
            event_object_bytes: out.event_object_bytes,
            event_shared_bytes: out.event_shared_bytes,
            event_shared_alignment: out.event_shared_alignment,
            event_graph_extent: out.event_graph_extent,
            event_platform_objects: out.event_platform_objects,
            task_header_bytes: out.task_header_bytes,
            task_header_alignment: out.task_header_alignment,
            fixed_control_bytes,
        })
    }
}
impl SubmissionScope {
    /// Mark an original role before its operation. Until actual carrier/quota
    /// configuration, any Event/task producer receives a fixed refusal.
    pub fn require_original_native_controls(&mut self) -> Result<(), OriginalNativeControlError> {
        // SAFETY: fixed no-hooks update of this live thread-affine Scope.
        let status =
            unsafe { safemlx_sys::mlx_submission_scope_require_original_controls(self.raw) };
        OriginalNativeControlError::from_status(status)
    }
    /// Enable the stronger original Event/task contract on the current empty
    /// Scope. Both original quotas and its bound carrier are mandatory. A
    /// refused attempt is spent; scoped ordinary observations remain unchanged.
    pub fn enable_original_native_controls(&mut self) -> Result<(), OriginalNativeControlError> {
        // SAFETY: self retains this thread-affine native Scope; the call only
        // reads/sets fixed fields, never invokes native work or an error handler.
        let status =
            unsafe { safemlx_sys::mlx_submission_scope_enable_original_controls(self.raw) };
        OriginalNativeControlError::from_status(status)
    }

    /// Bind the fixed native allowance to this same current empty role after
    /// original-control enablement. Native retains it; late/duplicate/foreign
    /// bindings reject. No allocation, fallback or neutral reservation occurs.
    pub fn bind_original_buffer_budget(
        &mut self,
        budget: &crate::OriginalBufferBudget,
    ) -> Result<(), crate::OriginalBufferCause> {
        budget.bind_to_scope(self.raw)
    }
}

mod observer;
pub use observer::OriginalScopeObserver;

mod ordinary_entry;
pub use ordinary_entry::OrdinarySubmissionEntry;
