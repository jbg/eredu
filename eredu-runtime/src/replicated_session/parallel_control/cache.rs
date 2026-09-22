//! Descriptive control source for the existing distributed prompt-cache driver.
use super::{
    super::*, ParallelControlEvent, SessionTransactionControlError,
    SessionTransactionControlOccurrence,
};
use crate::{CommunicationOperation, DistributedExecutionPhase as Phase};
use std::mem::{size_of, size_of_val};

const SLOTS: usize = 3;

/// Existing distributed prompt-cache operation whose controls are described.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionCacheControlOperation {
    /// Preflight, provisional preparation, and reversible publication agreement.
    Save,
    /// Preflight and provisional preparation agreement before state replacement.
    Load,
}

/// Actual selected control occurrences of one distributed prompt-cache call.
/// This describes source geometry and grants no memory or execution authority.
#[derive(Clone, Debug)]
pub struct SessionCacheControlPlan {
    operation: SessionCacheControlOperation,
    rows: [Option<SessionTransactionControlOccurrence>; SLOTS],
}
impl SessionCacheControlPlan {
    fn prepare(
        operation: SessionCacheControlOperation,
        mut selected: impl FnMut(ParallelControlEvent) -> Option<CommunicationOperation>,
    ) -> Self {
        let phases = match operation {
            SessionCacheControlOperation::Save => [
                Some(Phase::PromptCacheSavePreflight),
                Some(Phase::PromptCacheSavePreparation),
                Some(Phase::PromptCacheSavePublication),
            ],
            SessionCacheControlOperation::Load => [
                Some(Phase::PromptCacheLoadPreflight),
                Some(Phase::PromptCacheLoadPreparation),
                None,
            ],
        };
        let mut ordinal = 0;
        let rows = phases.map(|phase| {
            let event = ParallelControlEvent::Phase(phase?);
            let operation = selected(event)?;
            let row = SessionTransactionControlOccurrence::new(ordinal, event, operation);
            ordinal += 1; // This fixed population contains at most SLOTS entries.
            Some(row)
        });
        Self { operation, rows }
    }
    /// Shared prompt-cache operation described by this source.
    pub const fn operation(&self) -> SessionCacheControlOperation {
        self.operation
    }
    /// Exact possible native calls, omitting phases selected as local-only.
    pub fn occurrences(&self) -> impl Iterator<Item = SessionTransactionControlOccurrence> + '_ {
        self.rows.iter().flatten().copied()
    }
    /// Starts a move-only ordered validator; failures cannot become success.
    pub fn cursor(&self) -> SessionCacheControlCursor {
        SessionCacheControlCursor {
            plan: self.clone(),
            next: 0,
            state: CursorState::Active,
        }
    }
    /// Fixed source, cursor, and validation transports, with no dynamic table.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<SessionCacheControlCursor>(),
            size_of::<[Option<Phase>; SLOTS]>(),
            size_of::<Result<usize, SessionTransactionControlError>>(),
            size_of::<Result<(), SessionTransactionControlError>>(),
            size_of::<(&Self, ParallelControlEvent, bool, usize)>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CursorState {
    Active,
    Finished,
    Invalid,
}

/// One-use validation of actual prompt-cache controls and their final result.
#[derive(Debug)]
pub struct SessionCacheControlCursor {
    plan: SessionCacheControlPlan,
    next: usize,
    state: CursorState,
}
impl SessionCacheControlCursor {
    /// Consumes the next exact native occurrence and returns its source ordinal.
    pub fn claim(
        &mut self,
        event: ParallelControlEvent,
    ) -> Result<usize, SessionTransactionControlError> {
        if self.state == CursorState::Active {
            if let Some(row) = self.plan.occurrences().nth(self.next) {
                if row.event() == event {
                    self.next += 1;
                    return Ok(row.ordinal());
                }
            }
        }
        self.reject(SessionTransactionControlError::Occurrence)
    }
    /// Records the actual shared-driver result. An error may terminate any
    /// reached prefix; success requires every selected occurrence exactly once.
    pub fn finish(&mut self, success: bool) -> Result<(), SessionTransactionControlError> {
        if self.state != CursorState::Active
            || (success && self.next != self.plan.occurrences().count())
        {
            return self.reject(SessionTransactionControlError::Completion);
        }
        self.state = CursorState::Finished;
        Ok(())
    }
    /// Whether the operation's real final result has been recorded.
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
    /// Describes the existing save/load protocol using this session's actual
    /// selected control strategy. Local-only phases create no native occurrence.
    pub fn cache_control_plan(
        &self,
        operation: SessionCacheControlOperation,
    ) -> SessionCacheControlPlan {
        SessionCacheControlPlan::prepare(operation, |event| {
            D::parallel_control_operation(&self.execution, event)
        })
    }
    /// Visits the exact outer phase and required-group callback types shared
    /// with the cache driver, without constructing or executing a callback.
    pub fn visit_cache_control_callbacks<'a, V>(
        &'a self,
        plan: &SessionCacheControlPlan,
        visitor: &mut V,
    ) -> Result<bool, V::Error>
    where
        B: crate::CommunicationBackend,
        V: ParallelControlCallbackVisitor<B>,
    {
        self.visit_parallel_control_occurrences(plan.occurrences(), visitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_control_cursor_preserves_failed_prefixes_and_rejects_replay() {
        for operation in [
            SessionCacheControlOperation::Save,
            SessionCacheControlOperation::Load,
        ] {
            let plan = SessionCacheControlPlan::prepare(operation, |_| {
                Some(CommunicationOperation::FailureAgreement)
            });
            let rows: Vec<_> = plan.occurrences().collect();
            for boundary in 0..=rows.len() {
                let mut cursor = plan.cursor();
                for row in &rows[..boundary] {
                    assert_eq!(cursor.claim(row.event()).unwrap(), row.ordinal());
                }
                cursor.finish(false).unwrap();
                assert!(cursor.is_complete());
                assert!(cursor.finish(true).is_err());
                let mut cursor = plan.cursor();
                for row in &rows[..boundary] {
                    cursor.claim(row.event()).unwrap();
                }
                assert_eq!(cursor.finish(true).is_ok(), boundary == rows.len());
            }
            let mut cursor = plan.cursor();
            cursor.claim(rows[0].event()).unwrap();
            assert!(cursor.claim(rows[0].event()).is_err());
            assert!(
                cursor.finish(false).is_err(),
                "rejection poisons the cursor"
            );
            let mut cursor = plan.cursor();
            assert!(
                cursor.claim(rows[1].event()).is_err(),
                "ordering remains exact"
            );
            let local = SessionCacheControlPlan::prepare(operation, |_| None);
            assert_eq!(local.occurrences().count(), 0);
            local.cursor().finish(true).unwrap();
        }
    }
}
