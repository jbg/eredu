//! Exact control occurrences of one shared output transaction.
use super::{super::*, ParallelControlEvent};
use crate::{CommunicationOperation, DistributedExecutionPhase as Phase};
use std::mem::{size_of, size_of_val};

const SLOTS: usize = 17;
const RETURN: usize = SLOTS - 1;

/// One source-selected native control call in the shared session driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionTransactionControlOccurrence {
    ordinal: usize,
    event: ParallelControlEvent,
    operation: CommunicationOperation,
}
impl SessionTransactionControlOccurrence {
    pub(super) const fn new(
        ordinal: usize,
        event: ParallelControlEvent,
        operation: CommunicationOperation,
    ) -> Self {
        Self {
            ordinal,
            event,
            operation,
        }
    }
    /// Position in this plan's retained occurrence population.
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }
    /// Actual phase or final decision, including separate loan occurrences.
    pub const fn event(self) -> ParallelControlEvent {
        self.event
    }
    /// Operation selected by the actual execution strategy and runtime.
    pub const fn operation(self) -> CommunicationOperation {
        self.operation
    }
}

/// Descriptive source for one output transaction and optional prediction loan.
/// This owns no execution, reservation, or submission authority.
#[derive(Clone, Debug)]
pub struct SessionTransactionControlPlan {
    rows: [Option<SessionTransactionControlOccurrence>; SLOTS],
    transactional: bool,
    prediction_loan: bool,
}
impl SessionTransactionControlPlan {
    fn prepare(
        demand: eredu_core::OutputDemand,
        transactional: bool,
        requires_sequence: bool,
        prediction_loan: bool,
        mut selected: impl FnMut(ParallelControlEvent) -> Option<CommunicationOperation>,
    ) -> Self {
        let phase = |phase| Some(ParallelControlEvent::Phase(phase));
        let loan = prediction_loan.then_some(super::super::prediction_loan::CONTROL_PHASE);
        let events = [
            loan.map(ParallelControlEvent::Phase),
            phase(Phase::InputPreparation),
            phase(Phase::ReadoutSelection),
            (demand != eredu_core::OutputDemand::Sequence && !requires_sequence)
                .then_some(ParallelControlEvent::Phase(Phase::StateOnlyReadout)),
            phase(Phase::InferenceWorkspace),
            phase(Phase::ObservationParticipation),
            // An inactive observer can reach this vote when another rank is
            // active. That path must fail before observation preparation.
            phase(Phase::ObservationParticipationRequired),
            transactional.then_some(ParallelControlEvent::Phase(Phase::ObservationPreparation)),
            transactional.then_some(ParallelControlEvent::Phase(Phase::ObservationCoordination)),
            phase(Phase::StateCheckpoint),
            phase(Phase::Execution),
            phase(Phase::OutputObservation),
            phase(Phase::OutputPublication),
            phase(Phase::MechanismCompletion),
            transactional.then_some(ParallelControlEvent::Phase(Phase::ObservationDelivery)),
            Some(ParallelControlEvent::Commit),
            loan.map(ParallelControlEvent::Phase),
        ];
        let mut ordinal = 0;
        let rows = events.map(|event| {
            let event = event?;
            let operation = selected(event)?;
            let row = SessionTransactionControlOccurrence {
                ordinal,
                event,
                operation,
            };
            ordinal += 1; // At most SLOTS entries in this fixed population.
            Some(row)
        });
        Self {
            rows,
            transactional,
            prediction_loan,
        }
    }
    /// Exact possible occurrences, with local-only phases omitted.
    pub fn occurrences(&self) -> impl Iterator<Item = SessionTransactionControlOccurrence> + '_ {
        self.rows.iter().flatten().copied()
    }
    /// Starts one move-only validation cursor over this descriptive source.
    pub fn cursor(&self) -> SessionTransactionControlCursor {
        SessionTransactionControlCursor {
            plan: self.clone(),
            next: 0,
            state: CursorState::Transaction,
            failed: false,
        }
    }
    /// Fixed plan, cursor, and validation transports; no dynamic table is built.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<SessionTransactionControlCursor>(),
            size_of::<[Option<ParallelControlEvent>; SLOTS]>(),
            size_of::<Result<usize, SessionTransactionControlError>>(),
            size_of::<Result<(), SessionTransactionControlError>>(),
            size_of::<(&Self, ParallelControlEvent, bool, usize)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

/// A session control did not match its exact retained lifecycle source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SessionTransactionControlError {
    /// A repeated, reordered, or foreign control occurrence was attempted.
    #[error("session control occurrence differs from its source")]
    Occurrence,
    /// Success was reported before the required phases or after a failed path.
    #[error("session control completion differs from its source")]
    Completion,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CursorState {
    Transaction,
    Return,
    LoanFinish,
    Finished,
    Invalid,
}

/// One-use event validation, separate from the request's monotonic claim issuer.
/// Rejected claims poison this cursor; failed prefixes never become success.
#[derive(Debug)]
pub struct SessionTransactionControlCursor {
    plan: SessionTransactionControlPlan,
    next: usize,
    state: CursorState,
    failed: bool,
}
impl SessionTransactionControlCursor {
    /// Consumes the next actual occurrence and returns its retained ordinal.
    pub fn claim(
        &mut self,
        event: ParallelControlEvent,
    ) -> Result<usize, SessionTransactionControlError> {
        if self.failed && self.state == CursorState::Transaction {
            return self.reject(SessionTransactionControlError::Occurrence);
        }
        let range = match self.state {
            CursorState::Transaction => self.next..RETURN,
            CursorState::Return => RETURN..SLOTS,
            _ => return self.reject(SessionTransactionControlError::Occurrence),
        };
        for slot in range {
            let Some(row) = self.plan.rows[slot] else {
                continue;
            };
            if row.event == event {
                self.next = slot + 1;
                if slot == 6 && !self.plan.transactional {
                    self.failed = true;
                }
                if slot == RETURN {
                    self.state = CursorState::LoanFinish;
                }
                return Ok(row.ordinal);
            }
            // Peer readout promotion can skip the local selective vote.
            // Unanimously absent observation skips its required-presence vote.
            if slot != 3 && !(slot == 6 && !self.plan.transactional) {
                return self.reject(SessionTransactionControlError::Occurrence);
            }
        }
        self.reject(SessionTransactionControlError::Occurrence)
    }
    /// Records the real callback result before the enclosing state-loan return.
    /// An error may end any reached prefix; success requires all required calls.
    pub fn finish_transaction(
        &mut self,
        success: bool,
    ) -> Result<(), SessionTransactionControlError> {
        if self.state != CursorState::Transaction
            || (self.plan.rows[0].is_some() && self.next == 0)
            || (success
                && (self.failed
                    || self.plan.rows[self.next..RETURN].iter().enumerate().any(
                        |(offset, row)| {
                            let slot = self.next + offset;
                            row.is_some() && slot != 3 && !(slot == 6 && !self.plan.transactional)
                        },
                    )))
        {
            return self.reject(SessionTransactionControlError::Completion);
        }
        self.failed |= !success;
        self.next = RETURN;
        self.state = if self.plan.prediction_loan {
            if self.plan.rows[RETURN].is_some() {
                CursorState::Return
            } else {
                CursorState::LoanFinish
            }
        } else {
            CursorState::Finished
        };
        Ok(())
    }
    /// Records the actual outer loan result. Failure may fence before return.
    /// Successful return cannot clear an earlier transaction failure.
    pub fn finish_loan(&mut self, success: bool) -> Result<(), SessionTransactionControlError> {
        if !self.plan.prediction_loan
            || matches!(self.state, CursorState::Invalid | CursorState::Finished)
            || (success && (self.state != CursorState::LoanFinish || self.failed))
        {
            return self.reject(SessionTransactionControlError::Completion);
        }
        self.failed |= !success;
        self.state = CursorState::Finished;
        Ok(())
    }
    /// Whether the transaction and any requested return have been resolved.
    pub fn is_complete(&self) -> bool {
        self.state == CursorState::Finished
    }
    fn reject<T>(
        &mut self,
        error: SessionTransactionControlError,
    ) -> Result<T, SessionTransactionControlError> {
        self.state = CursorState::Invalid;
        Err(error)
    }
}

/// Visits actual callback types without constructing or executing a callback.
/// Backends use these types to quote their own wrapping closure/error transport.
pub trait ParallelControlCallbackVisitor<B: NeuralBackend> {
    /// Typed query failure retained by the caller's own source funding.
    type Error;
    /// One actual phase/commit callback and its exact output and error types.
    fn visit<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Self::Error>
    where
        F: FnOnce(
            Option<(
                &B::ParallelContext,
                &eredu_nn::workspace::HostMetadataFunding,
            )>,
        ) -> Result<T, E>;
    /// The composed strategy callback actually passed to the native group loan.
    fn visit_group<T, E, F>(
        &mut self,
        occurrence: SessionTransactionControlOccurrence,
    ) -> Result<(), Self::Error>
    where
        B: crate::CommunicationBackend,
        F: FnOnce(Option<&B::CommunicationGroup>) -> Result<T, E>;
}

fn visit_factory<B, I, T, E, F, V>(
    _factory: impl FnOnce(I) -> F,
    occurrence: SessionTransactionControlOccurrence,
    visitor: &mut V,
) -> Result<(), V::Error>
where
    B: NeuralBackend,
    V: ParallelControlCallbackVisitor<B>,
    F: FnOnce(
        Option<(
            &B::ParallelContext,
            &eredu_nn::workspace::HostMetadataFunding,
        )>,
    ) -> Result<T, E>,
{
    visitor.visit::<T, E, F>(occurrence)
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    A: LayeredArchitecture<B, M::State>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Describes the existing output transaction using actual captured observer
    /// facts, requested demand, and this session's selected control strategy.
    pub fn transaction_control_plan(
        &self,
        demand: eredu_core::OutputDemand,
        transactional: bool,
        requires_sequence: bool,
        prediction_loan: bool,
    ) -> SessionTransactionControlPlan {
        SessionTransactionControlPlan::prepare(
            demand,
            transactional,
            requires_sequence,
            prediction_loan,
            |event| D::parallel_control_operation(&self.execution, event),
        )
    }
    /// Quotes the exact named callback types used by the ordinary shared driver.
    pub fn visit_transaction_control_callbacks<'a, V>(
        &'a self,
        plan: &SessionTransactionControlPlan,
        visitor: &mut V,
    ) -> Result<bool, V::Error>
    where
        B: crate::CommunicationBackend,
        V: ParallelControlCallbackVisitor<B>,
    {
        self.visit_parallel_control_occurrences(plan.occurrences(), visitor)
    }
    pub(in crate::replicated_session) fn visit_parallel_control_occurrences<'a, V>(
        &'a self,
        occurrences: impl IntoIterator<Item = SessionTransactionControlOccurrence>,
        visitor: &mut V,
    ) -> Result<bool, V::Error>
    where
        B: crate::CommunicationBackend,
        V: ParallelControlCallbackVisitor<B>,
    {
        for occurrence in occurrences {
            match occurrence.event {
                ParallelControlEvent::Phase(phase) => visit_factory::<B, _, _, _, _, V>(
                    |(runtime, context): (
                        &'a mut D::Runtime,
                        &'a <B::Tensor as Tensor>::Context,
                    )| {
                        Self::phase_control_callback(runtime, phase, false, context)
                    },
                    occurrence,
                    visitor,
                )?,
                ParallelControlEvent::Commit => visit_factory::<B, _, _, _, _, V>(
                    |(runtime, epoch, context): (
                        &'a mut D::Runtime,
                        DistributedCommitEpoch,
                        &'a <B::Tensor as Tensor>::Context,
                    )| Self::commit_control_callback(runtime, epoch, context),
                    occurrence,
                    visitor,
                )?,
            }
            if !D::visit_parallel_control_callback(&self.execution, occurrence, visitor)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    pub(in crate::replicated_session) fn phase_control_callback<'a>(
        runtime: &'a mut D::Runtime,
        phase: Phase,
        success: bool,
        context: &'a <B::Tensor as Tensor>::Context,
    ) -> impl FnOnce(
        Option<(
            &B::ParallelContext,
            &eredu_nn::workspace::HostMetadataFunding,
        )>,
    ) -> Result<
        bool,
        ReplicatedTextSessionError<A::Error, M::PolicyError, std::convert::Infallible>,
    > + use<'a, A, B, M, D> {
        move |prepared| {
            D::agree_distributed_phase_with_parallel(runtime, phase, success, context, prepared)
        }
    }
    pub(in crate::replicated_session) fn commit_control_callback<'a>(
        runtime: &'a mut D::Runtime,
        epoch: DistributedCommitEpoch,
        context: &'a <B::Tensor as Tensor>::Context,
    ) -> impl FnOnce(
        Option<(
            &B::ParallelContext,
            &eredu_nn::workspace::HostMetadataFunding,
        )>,
    ) -> Result<
        DistributedCommitOutcome,
        ReplicatedTextSessionError<A::Error, M::PolicyError, std::convert::Infallible>,
    > + use<'a, A, B, M, D> {
        move |prepared| D::commit_after_completion_with_parallel(runtime, epoch, context, prepared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan(transactional: bool, loan: bool) -> SessionTransactionControlPlan {
        SessionTransactionControlPlan::prepare(
            eredu_core::OutputDemand::LastPosition,
            transactional,
            false,
            loan,
            |_| Some(CommunicationOperation::FailureAgreement),
        )
    }
    #[test]
    fn transaction_cursor_preserves_conditional_paths_and_consumes_loan_return() {
        for transactional in [false, true] {
            for selective in [false, true] {
                let plan = plan(transactional, true);
                let mut cursor = plan.cursor();
                for row in plan.occurrences() {
                    if row.ordinal == plan.occurrences().count() - 1 {
                        break;
                    }
                    if (!selective
                        && row.event == ParallelControlEvent::Phase(Phase::StateOnlyReadout))
                        || (!transactional
                            && row.event
                                == ParallelControlEvent::Phase(
                                    Phase::ObservationParticipationRequired,
                                ))
                    {
                        continue;
                    }
                    assert_eq!(cursor.claim(row.event).unwrap(), row.ordinal);
                }
                cursor.finish_transaction(true).unwrap();
                assert!(!cursor.is_complete());
                let returned = plan.occurrences().last().unwrap();
                assert_eq!(cursor.claim(returned.event).unwrap(), returned.ordinal);
                assert!(!cursor.is_complete());
                cursor.finish_loan(true).unwrap();
                assert!(cursor.is_complete());
                assert!(cursor.finish_loan(true).is_err());
            }
        }
    }
    #[test]
    fn transaction_cursor_rejects_replay_successful_prefix_and_false_continuation() {
        let plan = plan(true, true);
        let rows: Vec<_> = plan.occurrences().collect();
        for boundary in 1..rows.len() {
            let mut cursor = plan.cursor();
            for row in &rows[..boundary] {
                cursor.claim(row.event).unwrap();
            }
            if boundary < rows.len() - 1 {
                assert!(cursor.finish_transaction(true).is_err());
            }
            let mut cursor = plan.cursor();
            for row in &rows[..boundary] {
                cursor.claim(row.event).unwrap();
            }
            cursor.finish_transaction(false).unwrap();
            cursor.claim(rows.last().unwrap().event).unwrap();
            assert!(
                cursor.finish_loan(true).is_err(),
                "failure cannot become success"
            );
        }
        let mut cursor = plan.cursor();
        cursor.claim(rows[0].event).unwrap();
        assert!(
            cursor.claim(rows[0].event).is_err(),
            "entry cannot be replayed as return"
        );
        assert!(
            cursor.finish_loan(false).is_err(),
            "rejected claims remain poisoned"
        );
        let inactive = self::plan(false, false);
        let mut cursor = inactive.cursor();
        for row in inactive.occurrences() {
            cursor.claim(row.event).unwrap();
            if row.event == ParallelControlEvent::Phase(Phase::ObservationParticipationRequired) {
                break;
            }
        }
        assert!(
            cursor
                .claim(ParallelControlEvent::Phase(Phase::StateCheckpoint))
                .is_err()
        );
        let local = SessionTransactionControlPlan::prepare(
            eredu_core::OutputDemand::Sequence,
            false,
            false,
            false,
            |_| None,
        );
        assert_eq!(local.occurrences().count(), 0);
        let mut cursor = local.cursor();
        cursor.finish_transaction(true).unwrap();
        assert!(cursor.is_complete());
        assert!(cursor.finish_transaction(true).is_err());
    }
}
