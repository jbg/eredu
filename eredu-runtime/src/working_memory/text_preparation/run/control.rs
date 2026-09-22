//! Exact completed-run gate shared by sampler changes and branch movement.
use super::*;
// Preserve the original request-start -> preparation lock order. The closure
// performs only fixed validation/counter changes, never allocation or callbacks.
pub(super) fn validate<T>(
    request: &InferenceRequest,
    context: &TextStepContext,
    f: impl FnOnce(&mut RunProgress) -> Result<T, WorkingMemoryError>,
) -> Result<T, WorkingMemoryError> {
    let start = request
        .start_state()
        .lock()
        .map_err(|_| WorkingMemoryError::Poisoned)?;
    let actual = request
        .preparation
        .as_ref()
        .ok_or(WorkingMemoryError::IdentityMismatch)?;
    match &*start {
        RequestStart::Preparing(expected) | RequestStart::Started(Some(expected))
            if Arc::ptr_eq(actual, expected) => {}
        _ => return Err(WorkingMemoryError::IdentityMismatch),
    }
    let mut state = actual
        .state
        .lock()
        .map_err(|_| WorkingMemoryError::Poisoned)?;
    if state.prompt != READY || state.sampling != READY {
        return Err(WorkingMemoryError::PreparationNotReady);
    }
    let run = state
        .run
        .as_mut()
        .ok_or(WorkingMemoryError::TextRunUnbound)?;
    if run.initial.run_identity() != context.run_identity()
        || run.initial.policy_identity() != context.policy_identity()
    {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    if run.fenced {
        return Err(WorkingMemoryError::ExecutionFenced);
    }
    if run.active.is_some() {
        return Err(WorkingMemoryError::TextStepActive);
    }
    if context.attempt() != run.next_attempt {
        return Err(WorkingMemoryError::TextStepOrdinalMismatch {
            expected: run.next_attempt,
            actual: context.attempt(),
        });
    }
    if run.completed != run.next_attempt.checked_sub(1) {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    f(run)
}
