//! Monotone logical prediction permission within an existing preparation owner.

use super::*;
use eredu_core::{PendingTextInput, TextStepContext};

#[derive(Debug)]
pub(super) struct RunProgress {
    initial: TextStepContext,
    next_attempt: u64,
    active: Option<u64>,
    completed: Option<u64>,
    fenced: bool,
    supersession_claimed: bool,
}

/// Move-only logical permission for one prediction of a bound text run.
///
/// This retains the original request charge, consumes one output slot at issue,
/// and fences the run if abandoned. It does not certify a complete memory quote
/// or authorize a native allocator. Backends must separately validate the priced
/// mechanisms, input/frontier, domains and parameter identity before granting
/// native entry. Native completion independently retains unresolved resources.
#[derive(Debug)]
#[must_use = "finish only after successful submission and required token commitment"]
pub struct InferenceTextStep {
    request: InferenceRequest,
    attempt: u64,
    finished: bool,
}

/// Payload-free evidence identifying one output of an exact prepared run.
///
/// Clones grant no operation authority or memory charge. A receipt can accompany
/// a provisional native token, but cannot authorize the next step until its
/// issuing permit finishes. The token/completion must retain its own storage.
#[derive(Debug, Clone)]
pub struct InferenceTextStepReceipt {
    authority: ReceiptAuthority,
    attempt: u64,
}

#[derive(Debug, Clone)]
enum ReceiptAuthority {
    // Canonical identity only, retaining the existing pool but no request or
    // preparation allocation. A retired account ID can never be issued again.
    Reserved {
        pool: super::super::WorkingMemoryPool,
        id: u64,
    },
    Ordinary(Arc<TextPreparationAuthority>),
}
impl ReceiptAuthority {
    fn new(request: &InferenceRequest) -> Self {
        match request.memory_reservation() {
            Some(reservation) => Self::Reserved {
                pool: reservation.0.pool.clone(),
                id: reservation.0.account_id,
            },
            None => Self::Ordinary(Arc::clone(
                request.preparation.as_ref().expect("preparation owner"),
            )),
        }
    }
    fn matches_request(&self, request: &InferenceRequest) -> Result<bool, WorkingMemoryError> {
        let Some(actual) = request.preparation.as_ref() else {
            return Ok(false);
        };
        match self {
            Self::Ordinary(authority) => {
                Ok(request.memory_reservation().is_none() && Arc::ptr_eq(actual, authority))
            }
            Self::Reserved { pool, id } => {
                let Some(reservation) = request.memory_reservation() else {
                    return Ok(false);
                };
                if reservation.0.account_id != *id || !pool.same_domain(&reservation.0.pool) {
                    return Ok(false);
                }
                {
                    let usage = pool
                        .0
                        .usage
                        .lock()
                        .map_err(|_| WorkingMemoryError::Poisoned)?;
                    usage
                        .funding
                        .validate_metadata(*id, &reservation.0.execution)?;
                }
                // Exact authority remains in the canonical reservation after
                // prefill starts; a same-account replaced preparation is not valid.
                let start = reservation
                    .0
                    .start
                    .lock()
                    .map_err(|_| WorkingMemoryError::Poisoned)?;
                Ok(match &*start {
                    RequestStart::Preparing(expected) | RequestStart::Started(Some(expected)) => {
                        Arc::ptr_eq(actual, expected)
                    }
                    RequestStart::Fresh | RequestStart::Started(None) => false,
                })
            }
        }
    }
}
fn receipt_identity(
    request: &InferenceRequest,
    input: &PendingTextInput<(), &InferenceTextStepReceipt>,
) -> Result<bool, WorkingMemoryError> {
    match input {
        PendingTextInput::Prefill(()) => Ok(true),
        PendingTextInput::Decode(receipt) => receipt.authority.matches_request(request),
    }
}

impl InferenceTextPreparation {
    /// Binds the original core-issued run and policy before any preparation
    /// stage is claimed. Call from the backend's cold run-binding hook, never
    /// learn this identity from a later step after mutable policy exposure.
    /// Clones share this one-use binding. It supplies no missing workspace bound.
    pub fn bind_run(&self, context: &TextStepContext) -> Result<(), WorkingMemoryError> {
        if context.attempt() != 0 {
            return Err(WorkingMemoryError::TextStepOrdinalMismatch {
                expected: 0,
                actual: context.attempt(),
            });
        }
        let mut state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.run.is_some() || state.prompt != UNCLAIMED || state.sampling != UNCLAIMED {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        state.run = Some(RunProgress {
            initial: context.clone(),
            next_attempt: 0,
            active: None,
            completed: None,
            fenced: false,
            supersession_claimed: false,
        });
        Ok(())
    }

    /// Rechecks an original, fully prepared run before its first step is issued.
    ///
    /// This compares the actual stored core run and policy identities and requires
    /// attempt zero, ready prompt/sampler stages, and no issued, active, completed
    /// or fenced step. A request that already entered prefill is also rejected.
    /// It binds nothing, consumes no output allowance, and changes no progress.
    /// Successful checks may be repeated, including for a zero-output request.
    ///
    /// This is a point-in-time logical check, not an installation or allocation
    /// grant. Callers separately validate exact request/executable/geometry/domain
    /// with the existing request and reservation methods, and independently check
    /// funding/source health. No numerical source, quote or completion is proved.
    pub fn validate_initial_ready_context(
        &self,
        context: &TextStepContext,
    ) -> Result<(), WorkingMemoryError> {
        // Preserve begin_prefill's existing request-start -> preparation order.
        // Holding both prevents legacy prefill from racing this initial check.
        // Neither lock owns a newly cloned payload or a caller callback.
        let start = self
            .request
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        match &*start {
            RequestStart::Preparing(authority)
                if std::ptr::eq(authority.as_ref(), self.authority()) => {}
            RequestStart::Started(_) => return Err(WorkingMemoryError::AlreadyStarted),
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        }
        let state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if run.initial.run_identity() != context.run_identity()
            || run.initial.policy_identity() != context.policy_identity()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if context.attempt() != 0 {
            return Err(WorkingMemoryError::TextStepOrdinalMismatch {
                expected: 0,
                actual: context.attempt(),
            });
        }
        if state.prompt != READY || state.sampling != READY {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        if run.fenced {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        if run.active.is_some() {
            return Err(WorkingMemoryError::TextStepActive);
        }
        if run.next_attempt != 0 || run.completed.is_some() || run.supersession_claimed {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        Ok(())
    }

    /// Issues one prediction after prompt and sampler preparation succeeded.
    /// Prefill requires the initial ordinal; decode requires the immediately
    /// preceding completed permit's receipt. Rejection changes no progress.
    /// Successful issuance consumes its ordinal and output slot before callbacks
    /// or peer agreement. Restored state and preparation clones cannot refund it.
    /// Existing span admission remains responsible for actual model geometry.
    pub fn claim_step(
        &self,
        context: &TextStepContext,
        input: PendingTextInput<(), &InferenceTextStepReceipt>,
    ) -> Result<InferenceTextStep, WorkingMemoryError> {
        // Canonical identity is inspected before the preparation mutex, keeping
        // the existing start -> state lock order. Report it at the old input check.
        let identity = receipt_identity(&self.request, &input);
        let mut state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if run.initial.run_identity() != context.run_identity()
            || run.initial.policy_identity() != context.policy_identity()
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if state.prompt != READY || state.sampling != READY {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        let run = state.run.as_mut().expect("bound run checked above");
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
        let limit = self.request.geometry().max_output_tokens;
        if run.next_attempt >= limit {
            return Err(WorkingMemoryError::TextOutputAllowanceExceeded {
                issued: run.next_attempt,
                limit,
            });
        }
        validate_step_input(identity, run.next_attempt, run.completed, input)?;
        let next = run
            .next_attempt
            .checked_add(1)
            .ok_or(WorkingMemoryError::Overflow)?;
        run.active = Some(run.next_attempt);
        run.next_attempt = next;
        Ok(InferenceTextStep {
            request: self.request.clone(),
            attempt: context.attempt(),
            finished: false,
        })
    }
}

impl InferenceTextStep {
    /// Issuing ordinal within this exact prepared run. Initial prefill is zero;
    /// subsequent attempts select their corresponding cached decode geometry.
    /// Reading this ordinal does not grant submission or completion authority.
    pub fn attempt(&self) -> u64 {
        self.attempt
    }

    // The step itself is move-only core-run authority. Initial prefill may still
    // be Preparing; cached decode is Started. Never reuse the preparation-only
    // predicate or accept a public receipt/ordinal in place of this object.
    pub(in crate::working_memory) fn original_scope_identity(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(Arc<TextPreparationAuthority>, u64), WorkingMemoryError> {
        let actual_reservation = self
            .request
            .memory_reservation()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !actual_reservation.0.same(&reservation.0) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let start = self
            .request
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let authority = self
            .request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        match &*start {
            RequestStart::Preparing(expected) | RequestStart::Started(Some(expected))
                if Arc::ptr_eq(authority, expected) => {}
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        }
        let state = authority
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if self.finished
            || run.fenced
            || run.active != Some(self.attempt)
            || run.next_attempt
                != self
                    .attempt
                    .checked_add(1)
                    .ok_or(WorkingMemoryError::Overflow)?
        {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        if state.prompt != READY || state.sampling != READY {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        Ok((Arc::clone(authority), self.attempt))
    }

    /// Exact original request retained through this prediction.
    pub fn request(&self) -> &InferenceRequest {
        &self.request
    }

    /// Permanently retires a predecessor run at an explicitly checked state
    /// handoff. This replacement must own its active initial prefill permit.
    /// Both requests must belong to `execution`, with distinct preparations and
    /// core run identities. A predecessor with an active step cannot retire.
    /// An already fenced predecessor is permitted only because the backend
    /// independently validates its installed state. Each replacement permit can
    /// perform at most one successful supersession, even through shared borrows.
    ///
    /// The backend must first validate the exact installed predecessor branch,
    /// native frontier, domain, and replacement quote under its submission
    /// authority. This method proves only the shared logical transition: it
    /// does not infer normal termination, authorize native work, publish state,
    /// release charges, or refund either run's issued output allowance.
    ///
    /// All checks happen while both run states are locked. Rejection leaves
    /// both runs unchanged. After success, failure to install the replacement
    /// cannot reopen the predecessor; native recovery retains its own charges.
    pub fn supersede_predecessor(
        &self,
        predecessor: &InferenceRequest,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        self.request.validate(execution, self.request.geometry())?;
        predecessor.validate(execution, predecessor.geometry())?;
        if self.request.validate_same_request(predecessor).is_ok() {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if self.attempt != 0 {
            return Err(WorkingMemoryError::InvocationPhaseMismatch);
        }
        let replacement = self
            .request
            .preparation
            .as_ref()
            .expect("preparation owner");
        let previous = predecessor
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if Arc::ptr_eq(replacement, previous) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }

        // Opposing handoffs must acquire the same pair in the same order. No
        // accounting lock, callback, provider operation or allocation occurs
        // while either preparation state is locked.
        if Arc::as_ptr(replacement) < Arc::as_ptr(previous) {
            let mut replacement = replacement
                .state
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            let mut previous = previous
                .state
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            supersede_runs(&mut replacement, &mut previous)
        } else {
            let mut previous = previous
                .state
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            let mut replacement = replacement
                .state
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            supersede_runs(&mut replacement, &mut previous)
        }
    }

    /// Rechecks the actual submitted input against this still-active step.
    /// A composed backend must call this if submission receives input separately
    /// from permit acquisition. Decode requires the same authority and preceding
    /// completed receipt; presence of any receipt is insufficient. This is
    /// read-only and neither consumes nor refunds progress or output allowance.
    pub fn validate_input(
        &self,
        input: PendingTextInput<(), &InferenceTextStepReceipt>,
    ) -> Result<(), WorkingMemoryError> {
        let identity = receipt_identity(&self.request, &input);
        let authority = self
            .request
            .preparation
            .as_ref()
            .expect("preparation owner");
        let state = authority
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if run.fenced || run.active != Some(self.attempt) {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        validate_step_input(identity, self.attempt, run.completed, input)
    }

    /// Evidence for the output being constructed. It remains provisional until
    /// this permit finishes; dropping the permit makes it permanently unusable.
    pub fn receipt(&self) -> InferenceTextStepReceipt {
        InferenceTextStepReceipt {
            authority: ReceiptAuthority::new(&self.request),
            attempt: self.attempt,
        }
    }

    /// Finalizes logical success, without claiming native completion or freeing
    /// its storage. This does not refund the issued ordinal or output allowance.
    pub fn finish(mut self) -> Result<(), WorkingMemoryError> {
        {
            let mut state = self
                .request
                .preparation
                .as_ref()
                .expect("preparation owner")
                .state
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            let run = state
                .run
                .as_mut()
                .ok_or(WorkingMemoryError::TextRunUnbound)?;
            if run.fenced || run.active != Some(self.attempt) {
                return Err(WorkingMemoryError::ExecutionFenced);
            }
            run.active = None;
            run.completed = Some(self.attempt);
        }
        self.finished = true;
        Ok(())
    }
}

fn supersede_runs(
    replacement: &mut TextPreparationState,
    predecessor: &mut TextPreparationState,
) -> Result<(), WorkingMemoryError> {
    let next = replacement
        .run
        .as_ref()
        .ok_or(WorkingMemoryError::TextRunUnbound)?;
    if replacement.prompt != READY || replacement.sampling != READY {
        return Err(WorkingMemoryError::PreparationNotReady);
    }
    if next.fenced || next.active != Some(0) || next.next_attempt != 1 || next.completed.is_some() {
        return Err(WorkingMemoryError::ExecutionFenced);
    }
    if next.supersession_claimed {
        return Err(WorkingMemoryError::AlreadyStarted);
    }
    let previous = predecessor
        .run
        .as_ref()
        .ok_or(WorkingMemoryError::TextRunUnbound)?;
    if next.initial.run_identity() == previous.initial.run_identity() {
        return Err(WorkingMemoryError::IdentityMismatch);
    }
    if predecessor.prompt != READY || predecessor.sampling != READY {
        return Err(WorkingMemoryError::PreparationNotReady);
    }
    if previous.active.is_some() {
        return Err(WorkingMemoryError::TextStepActive);
    }
    predecessor
        .run
        .as_mut()
        .expect("bound predecessor checked above")
        .fenced = true;
    replacement
        .run
        .as_mut()
        .expect("bound replacement checked above")
        .supersession_claimed = true;
    Ok(())
}

fn validate_step_input(
    identity: Result<bool, WorkingMemoryError>,
    attempt: u64,
    completed: Option<u64>,
    input: PendingTextInput<(), &InferenceTextStepReceipt>,
) -> Result<(), WorkingMemoryError> {
    match input {
        PendingTextInput::Prefill(()) if attempt == 0 => Ok(()),
        PendingTextInput::Decode(receipt) if attempt != 0 => {
            if identity? && Some(receipt.attempt) == completed && receipt.attempt == attempt - 1 {
                Ok(())
            } else {
                Err(WorkingMemoryError::IdentityMismatch)
            }
        }
        _ => Err(WorkingMemoryError::InvocationPhaseMismatch),
    }
}

impl Drop for InferenceTextStep {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let authority = self
            .request
            .preparation
            .as_ref()
            .expect("preparation owner");
        if let Ok(mut state) = authority.state.lock() {
            if let Some(run) = &mut state.run {
                run.fenced = true;
                if run.active == Some(self.attempt) {
                    run.active = None;
                }
            }
        }
        // A poisoned owner already rejects every later claim. Completion and
        // backend recovery retain any unresolved native work independently.
    }
}

impl InferenceTextStepReceipt {
    /// Validates the latest completed output as read-only source evidence.
    /// This neither issues a step nor consumes/refunds an output slot. Native
    /// callers must separately prove source storage, settlement and session
    /// identity before copying; a completed receipt grants no allocation.
    pub fn validate_completed_source(
        &self,
        request: &InferenceRequest,
        next_prediction: u64,
    ) -> Result<(), WorkingMemoryError> {
        let authority = request
            .preparation
            .as_ref()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !self.authority.matches_request(request)? {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let state = authority
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if run.fenced {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        if run.active.is_some() {
            return Err(WorkingMemoryError::TextStepActive);
        }
        if run.completed != Some(self.attempt)
            || self
                .attempt
                .checked_add(1)
                .ok_or(WorkingMemoryError::Overflow)?
                != next_prediction
            || run.next_attempt != next_prediction
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }

    /// Issuing ordinal within its exact prepared run; not a globally unique ID.
    pub fn attempt(&self) -> u64 {
        self.attempt
    }
}

impl InferenceTextPreparation {
    // Original Admission extraction, after bind and before Prompt/Sampling.
    // No callback, allocation, accounting mutation or agreement occurs here.
    pub(in crate::working_memory) fn claim_generation_sequence(
        &self,
        context: &TextStepContext,
    ) -> Result<(), WorkingMemoryError> {
        self.initial_sequence_context(context, true)
    }
    pub(in crate::working_memory) fn validate_token_input_context(
        &self,
        context: &TextStepContext,
    ) -> Result<(), WorkingMemoryError> {
        self.initial_sequence_context(context, false)
    }
    fn initial_sequence_context(
        &self,
        context: &TextStepContext,
        claim: bool,
    ) -> Result<(), WorkingMemoryError> {
        let start = self
            .request
            .start_state()
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        match &*start {
            RequestStart::Preparing(authority)
                if std::ptr::eq(authority.as_ref(), self.authority()) => {}
            RequestStart::Started(_) => return Err(WorkingMemoryError::AlreadyStarted),
            _ => return Err(WorkingMemoryError::IdentityMismatch),
        }
        let mut state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let run = state
            .run
            .as_ref()
            .ok_or(WorkingMemoryError::TextRunUnbound)?;
        if &run.initial != context || context.attempt() != 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if run.fenced {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        if state.prompt != UNCLAIMED
            || state.sampling != UNCLAIMED
            || run.next_attempt != 0
            || run.active.is_some()
            || run.completed.is_some()
            || run.supersession_claimed
        {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        if claim {
            state.prompt = SEQUENCE_CLAIMED;
        }
        Ok(())
    }

    // Finishes only the dormant host owner construction, never token readiness.
    pub(in crate::working_memory) fn finish_generation_sequence(
        &self,
    ) -> Result<(), WorkingMemoryError> {
        let mut state = self
            .authority()
            .state
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if state.prompt != SEQUENCE_CLAIMED || state.sampling != UNCLAIMED {
            return Err(WorkingMemoryError::PreparationNotReady);
        }
        state.prompt = SEQUENCE_CONSTRUCTED;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
