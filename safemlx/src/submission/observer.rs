use super::{ScopedSubmissionProgress, SubmissionRetirement, SubmissionStatus};
use crate::{
    error::{Exception, ScopedEvaluationCause},
    utils::runtime_lock,
    Array, EvaluatedArray, RetainedPrefillFailure,
};
use std::{marker::PhantomData, ptr, rc::Rc};

/// An observation alias of the current configured original submission role.
///
/// This creates no child scope or carrier. Dropping it never seals the role.
/// Observation and terminal record retirement remain valid after that role is
/// sealed; every producer independently requires its exact current authority.
#[derive(Debug)]
pub struct OriginalScopeObserver {
    pub(crate) raw: safemlx_sys::mlx_submission_observer,
    _thread: PhantomData<Rc<()>>,
}

impl OriginalScopeObserver {
    /// Named fixed native/Rust observation controls. This neither counts nodes
    /// nor prices arbitrary native destructor, exception or platform payloads.
    pub fn control_bytes() -> Option<usize> {
        use std::mem::size_of;
        [
            size_of::<Self>(),
            size_of::<Result<Self, Exception>>(),
            size_of::<Option<Self>>(),
            size_of::<Result<Option<Self>, Exception>>(),
            size_of::<SubmissionStatus>(),
            size_of::<ScopedSubmissionProgress>(),
            size_of::<Result<(ScopedSubmissionProgress, SubmissionStatus), Exception>>(),
            size_of::<Result<SubmissionRetirement, Exception>>(),
            size_of::<runtime_lock::RuntimeLockGuard>(),
            size_of::<Option<runtime_lock::RuntimeLockGuard>>(),
            size_of::<Option<RetainedPrefillFailure>>(),
            size_of::<&Array>(),
            size_of::<&'static std::panic::Location<'static>>() * 3,
            size_of::<EvaluatedArray<'static>>(),
            size_of::<Result<EvaluatedArray<'static>, Exception>>(),
            size_of::<Result<(), Exception>>(),
        ]
        .into_iter()
        .try_fold(
            unsafe { safemlx_sys::mlx_submission_observer_control_bytes() },
            usize::checked_add,
        )
    }
    /// A fixed observation refusal retaining the actual original native cause,
    /// when one has been published. `Observed` is not an error constructor.
    #[track_caller]
    pub fn observation_error(&self, outcome: ScopedSubmissionProgress) -> Option<Exception> {
        Some(self.error(match outcome {
            ScopedSubmissionProgress::Observed => return None,
            ScopedSubmissionProgress::NeedsFundedProgress => 8,
            ScopedSubmissionProgress::Unobservable => 9,
            ScopedSubmissionProgress::Busy => 10,
        }))
    }
    /// Reports exhausted host or native capacity owned by this original role.
    /// Retains its existing failure carrier without allocating diagnostic text.
    /// This is a refusal only; it neither marks completion nor grants capacity.
    #[track_caller]
    pub fn capacity_error(&self) -> Exception {
        self.error(2)
    }

    /// Refuse a mismatched explicit original role without formatting a new
    /// diagnostic or changing either role's completion/accounting state.
    #[track_caller]
    pub fn domain_error(&self) -> Exception {
        self.error(4)
    }

    /// Refuse invalid input geometry or a consumed finite input/view slot.
    /// The same prepaid failure carrier outlives this fixed error; no native
    /// work is submitted and no budget or retry authority is created.
    #[track_caller]
    pub fn invalid_input_error(&self) -> Exception {
        self.error(1)
    }

    /// Report a failed native lifetime status through its retained cause.
    /// Failure still does not imply settlement or permission to release owners.
    pub fn retained_failure(&self) -> Option<Exception> {
        self.status().failed().then(|| self.error(7))
    }
    /// Retain the actual current original role, or return `None` in ordinary
    /// execution. An unconfigured original child is a fixed refusal.
    #[track_caller]
    pub fn try_current() -> Result<Option<Self>, Exception> {
        let mut raw = safemlx_sys::mlx_submission_observer {
            ctx: ptr::null_mut(),
        };
        // SAFETY: TLS-only validation and an intrusive reference increment.
        // No runtime lock, allocation, hooks, progress or callback is involved.
        match unsafe { safemlx_sys::mlx_submission_observer_current(&mut raw) } {
            0 => Ok(Some(Self {
                raw,
                _thread: PhantomData,
            })),
            11 => Ok(None),
            status => {
                let mut failure = safemlx_sys::mlx_prefill_failure {
                    ctx: ptr::null_mut(),
                };
                unsafe { safemlx_sys::mlx_array_eval_scoped_failure(&mut failure) };
                Err(Exception::from_scoped_evaluation(
                    ScopedEvaluationCause::from_status(status),
                    unsafe { RetainedPrefillFailure::from_owned_raw(failure) },
                ))
            }
        }
    }

    /// Validate the actual current original role for an explicitly selected
    /// original caller. Ordinary execution is a fixed domain refusal here;
    /// this does not select an allocation mode or construct native ownership.
    #[track_caller]
    pub fn require_current() -> Result<Self, Exception> {
        Self::try_current()?
            .ok_or_else(|| Exception::from_scoped_evaluation(ScopedEvaluationCause::Domain, None))
    }

    #[track_caller]
    pub(crate) fn error(&self, status: u32) -> Exception {
        let mut raw = safemlx_sys::mlx_prefill_failure {
            ctx: ptr::null_mut(),
        };
        // SAFETY: this live thread-affine alias pins the exact bound carrier.
        unsafe { safemlx_sys::mlx_submission_observer_failure(&mut raw, self.raw) };
        Exception::from_scoped_evaluation(ScopedEvaluationCause::from_status(status), unsafe {
            RetainedPrefillFailure::from_owned_raw(raw)
        })
    }

    /// Compare two retained aliases of the same actual native role. Both
    /// aliases keep their Scope alive, so equality cannot be an address-reuse
    /// accident. Equal quotas, geometry, source shapes or capacity do not match.
    pub fn same_scope(&self, other: &Self) -> bool {
        self.raw.ctx == other.raw.ctx
    }

    /// Match a role held by its actual owning Scope. This is an identity check,
    /// not current-submission or original-request authorization by itself.
    pub fn belongs_to(&self, scope: &super::SubmissionScope) -> bool {
        self.raw.ctx == scope.raw().ctx
    }

    /// Validate an already completed array under this retained original role,
    /// including after the role is sealed. This submits no work, runs no hooks
    /// and performs no progress or wait. It detaches a completed matching event;
    /// unscheduled arrays and foreign pending descriptors remain unchanged.
    /// Pending, busy and retained native failures remain errors, not completion.
    #[track_caller]
    pub fn validate_completed_array(&self, array: &Array) -> Result<(), Exception> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Err(self.error(10));
        };
        // SAFETY: both owners are live, this observer is thread-affine, and the
        // no-hooks runtime guard serializes the native descriptor transition.
        match unsafe {
            safemlx_sys::mlx_submission_observer_validate_array(self.raw, array.as_ptr())
        } {
            0 => Ok(()),
            status => Err(self.error(status)),
        }
    }

    /// Cold structural lifetime status, including durable children. This is
    /// deliberately not the record-only counter and makes no future-work claim.
    pub fn status(&self) -> SubmissionStatus {
        let mut raw = safemlx_sys::mlx_submission_status {
            activity: 0,
            failed: false,
            blocked: false,
        };
        if unsafe { safemlx_sys::mlx_submission_observer_query(&mut raw, self.raw) } == 0 {
            SubmissionStatus::from_raw(raw)
        } else {
            SubmissionStatus::unobservable()
        }
    }

    /// Attempt only the exact owner's audited native frontiers. Busy, funded
    /// progress and unavailable observation remain distinct from settlement.
    #[track_caller]
    pub fn progress(&self) -> Result<(ScopedSubmissionProgress, SubmissionStatus), Exception> {
        let Some(_guard) = runtime_lock::try_enter_for_recovery() else {
            return Ok((ScopedSubmissionProgress::Busy, self.status()));
        };
        let mut raw = safemlx_sys::mlx_submission_status {
            activity: 0,
            failed: false,
            blocked: false,
        };
        let result = unsafe { safemlx_sys::mlx_submission_observer_progress(&mut raw, self.raw) };
        let progress = match result {
            0 => ScopedSubmissionProgress::Observed,
            8 => ScopedSubmissionProgress::NeedsFundedProgress,
            9 => ScopedSubmissionProgress::Unobservable,
            10 => ScopedSubmissionProgress::Busy,
            status => return Err(self.error(status)),
        };
        Ok((progress, SubmissionStatus::from_raw(raw)))
    }

    /// Detach this owner's already-terminal records and destroy them outside
    /// the registry lock. No other owner is progressed or retired. A completed
    /// pass is not proof that independent array/event/manager aliases are gone.
    pub fn retire_completed_records(&self) -> Result<SubmissionRetirement, Exception> {
        runtime_lock::try_retire(|| {
            match unsafe { safemlx_sys::mlx_submission_observer_retire(self.raw) } {
                0 => Ok(SubmissionRetirement::CompleteSnapshot),
                10 => Ok(SubmissionRetirement::Busy),
                status => Err(self.error(status)),
            }
        })
        .unwrap_or(Ok(SubmissionRetirement::Busy))
    }
}

impl Clone for OriginalScopeObserver {
    fn clone(&self) -> Self {
        // The safe type cannot move to a foreign thread, change identity, or
        // invalidate configuration. A clone only increments the retained ref.
        let status = unsafe { safemlx_sys::mlx_submission_observer_retain(self.raw) };
        assert_eq!(status, 0);
        Self {
            raw: self.raw,
            _thread: PhantomData,
        }
    }
}
impl Drop for OriginalScopeObserver {
    fn drop(&mut self) {
        // Final native custody release only queues the existing Rust owner.
        // This is intentionally not mlx_submission_scope_free (which seals).
        unsafe { safemlx_sys::mlx_submission_observer_release(self.raw) };
    }
}
