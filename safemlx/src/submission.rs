//! Ownership observations for native work accepted during a host operation.

use std::{marker::PhantomData, rc::Rc};

use crate::{error::Exception, utils::runtime_lock};

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
        let _guard = runtime_lock::enter();
        let mut raw = safemlx_sys::mlx_submission_scope {
            ctx: std::ptr::null_mut(),
        };
        // The native constructor accepts no work and does not use the textual
        // error channel, which might itself allocate during allocation failure.
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
