//! Exact model-internal occurrences from the ordinary partition trace.
use super::{
    ParallelControlCallbackVisitor, ParallelControlEvent, SessionTransactionControlError,
    SessionTransactionControlOccurrence,
};
use crate::replicated_session::*;
use crate::{CommunicationOperation, CommunicationRouteId, DistributedExecutionPhase};
use eredu_nn::workspace::{
    HostMetadataFunding, WorkspaceMetadataAllocation, WorkspaceMetadataError,
    WorkspaceModelControl, WorkspaceModelControlPhase, WorkspaceOperation, WorkspaceOperationKind,
};
use std::mem::{size_of, size_of_val};

/// A control source at its actual operation ordinal in the shared equation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionModelControlOccurrence {
    operation_ordinal: usize,
    declaration: WorkspaceModelControl,
}
impl SessionModelControlOccurrence {
    /// Exact operation position, including intervening tensor operations.
    pub const fn operation_ordinal(self) -> usize {
        self.operation_ordinal
    }
    /// Selected group and model phase from the shared driver.
    pub const fn declaration(self) -> WorkspaceModelControl {
        self.declaration
    }
    /// Runtime event corresponding to this descriptive source.
    pub fn event(self) -> ParallelControlEvent {
        ParallelControlEvent::Phase(match self.declaration.phase {
            WorkspaceModelControlPhase::Execution => DistributedExecutionPhase::Execution,
            WorkspaceModelControlPhase::BoundarySourceCompletion { route } => {
                DistributedExecutionPhase::BoundarySourceCompletion(CommunicationRouteId::new(
                    route,
                ))
            }
            WorkspaceModelControlPhase::BoundarySourceReady { route } => {
                DistributedExecutionPhase::BoundarySourceReady(CommunicationRouteId::new(route))
            }
        })
    }
    pub(crate) fn callback(self) -> SessionTransactionControlOccurrence {
        SessionTransactionControlOccurrence::new(
            self.operation_ordinal,
            self.event(),
            CommunicationOperation::FailureAgreement,
        )
    }
}
/// Paid descriptions of the actual model trace. This table owns no reservation
/// or execution authority and never estimates a separate partition schedule.
#[derive(Debug)]
pub struct SessionModelControlPlan {
    rows: Option<std::sync::Arc<Vec<SessionModelControlOccurrence>>>,
    // Drop the backing before its exact constructor payer.
    funding: HostMetadataFunding,
}
impl Clone for SessionModelControlPlan {
    fn clone(&self) -> Self {
        Self {
            rows: self.rows.clone(),
            funding: self.funding.clone(),
        }
    }
}
impl Drop for SessionModelControlPlan {
    fn drop(&mut self) {
        if let Some(rows) = self.rows.take() {
            drop(std::sync::Arc::into_inner(rows));
        }
    }
}
impl SessionModelControlPlan {
    /// Retains only markers emitted by the shared driver. As with NN metadata
    /// workers, the caller retains funding through any escaping constructor error.
    pub fn from_operations(
        operations: &[WorkspaceOperation],
        funding: &HostMetadataFunding,
    ) -> Result<Self, eredu_nn::Error> {
        let frames = [
            size_of::<Self>(),
            size_of::<Result<Self, eredu_nn::Error>>(),
            size_of::<(&[WorkspaceOperation], &HostMetadataFunding)>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, WorkspaceOperation>>>(),
            eredu_nn::workspace::WorkspaceContext::metadata_arc_bytes::<
                Vec<SessionModelControlOccurrence>,
            >()
            .ok_or(WorkspaceMetadataError::Overflow)?,
        ];
        funding
            .reserve_metadata(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(WorkspaceMetadataError::Overflow)?,
            )
            .map_err(WorkspaceMetadataError::from)?;
        let count = operations
            .iter()
            .filter(|row| matches!(row.kind, WorkspaceOperationKind::CommunicationControl(_)))
            .count();
        let mut rows = funding.metadata_vec(count)?;
        for (operation_ordinal, operation) in operations.iter().enumerate() {
            if let WorkspaceOperationKind::CommunicationControl(declaration) = operation.kind {
                if !operation.inputs.is_empty() || !operation.outputs.is_empty() {
                    return Err(WorkspaceMetadataError::Unqualified.into());
                }
                rows.push(SessionModelControlOccurrence {
                    operation_ordinal,
                    declaration,
                });
            }
        }
        Ok(Self {
            rows: Some(std::sync::Arc::new(rows)),
            funding: funding.clone(),
        })
    }
    /// Exact ordered source population; this does not claim an occurrence.
    pub fn occurrences(&self) -> &[SessionModelControlOccurrence] {
        self.rows
            .as_deref()
            .expect("live model-control source table")
            .as_slice()
    }
    /// Moves the retained source into one independently validated execution cursor.
    pub fn into_cursor(self) -> SessionModelControlCursor {
        SessionModelControlCursor {
            plan: self,
            next: 0,
            closed: false,
        }
    }
    /// Fixed cursor and claim transports, separately quoted from the dynamic table.
    pub fn control_bytes() -> Option<usize> {
        let frames = [
            size_of::<SessionModelControlCursor>(),
            size_of::<(&mut SessionModelControlCursor, WorkspaceModelControl, bool)>(),
            size_of::<Result<usize, SessionTransactionControlError>>(),
            size_of::<Result<(), SessionTransactionControlError>>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }
}
/// One exact attempted model-control sequence, separate from outer lifecycle phases.
#[derive(Debug)]
pub struct SessionModelControlCursor {
    plan: SessionModelControlPlan,
    next: usize,
    closed: bool,
}
impl SessionModelControlCursor {
    /// Consume the matching source before any fallible native preparation.
    /// A mismatch permanently closes this cursor; failed attempts never rewind.
    pub fn claim(
        &mut self,
        declaration: WorkspaceModelControl,
    ) -> Result<usize, SessionTransactionControlError> {
        if self.closed {
            return Err(SessionTransactionControlError::Occurrence);
        }
        let Some(row) = self
            .plan
            .occurrences()
            .get(self.next)
            .filter(|row| row.declaration == declaration)
        else {
            self.closed = true;
            return Err(SessionTransactionControlError::Occurrence);
        };
        let ordinal = row.operation_ordinal;
        self.next += 1; // An existing Vec row proves next is below usize::MAX.
        Ok(ordinal)
    }
    /// Close successful full execution or an unsuccessful prefix exactly once.
    /// Closing a failed prefix never constitutes successful completion.
    pub fn finish(&mut self, success: bool) -> Result<(), SessionTransactionControlError> {
        let complete = !self.closed && (!success || self.next == self.plan.occurrences().len());
        self.closed = true;
        if complete {
            Ok(())
        } else {
            Err(SessionTransactionControlError::Completion)
        }
    }
    /// Actual table payer, retained after the enclosing cold source is dropped.
    pub fn funding(&self) -> &HostMetadataFunding {
        &self.plan.funding
    }
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: crate::SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
        + crate::CommunicationBackend,
    A: LayeredArchitecture<B, M::State>,
    M: ReplicatedTextSessionMechanisms<A, B>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Visits the exact outer context and required-group callback types used by
    /// each retained model occurrence. False preserves unavailable source evidence.
    pub fn visit_model_control_callbacks<V>(
        &self,
        plan: &SessionModelControlPlan,
        visitor: &mut V,
    ) -> Result<bool, V::Error>
    where
        V: ParallelControlCallbackVisitor<B>,
    {
        for &occurrence in plan.occurrences() {
            if !D::visit_model_control_callback(&self.execution, occurrence, visitor)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}
