//! Exact source views and placement receipts for the shared typed exchange.
use super::*;
use crate::working_memory::{
    CopiedMediaStateBinding, InferenceRetention, InferenceStateAdmission, InferenceStateRevision,
    MediaSessionBinding, PendingTextBranchExchange, WorkingMemoryError,
};

/// Immutable source read from one actual typed state at a resolved boundary.
/// It copies only existing identity/admission handles, never charge directories.
#[derive(Debug, Clone)]
pub struct ControlBranchSource {
    revision: InferenceStateRevision,
    admission: Option<InferenceStateAdmission>,
    execution: crate::working_memory::InferenceExecutionIdentity,
    control: crate::replicated_session::ParameterControlIdentity,
    frontier: Option<u64>,
}
impl ControlBranchSource {
    /// Actual established state revision; this never initializes a lazy identity.
    pub fn revision(&self) -> &InferenceStateRevision {
        &self.revision
    }
    /// Actual retained request and logical position, if the branch has executed.
    pub fn admission(&self) -> Option<&InferenceStateAdmission> {
        self.admission.as_ref()
    }
    /// Checks prepared media against this actual state's exact source identity.
    pub fn validate_media(&self, source: &MediaSessionBinding) -> Result<(), WorkingMemoryError> {
        if source.matches_snapshot(
            &self.execution,
            &self.revision,
            self.frontier.ok_or(WorkingMemoryError::UnknownBound)?,
        ) && source.control.matches(&self.control)
        {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Checks the actual native frontier; explicit stateless ranks use admission.
    pub fn validate_frontier(&self, expected: u64) -> Result<(), WorkingMemoryError> {
        if self
            .frontier
            .or_else(|| self.admission.as_ref().map(|a| a.position()))
            .is_some_and(|actual| actual != expected)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    fn validate_state(
        &self,
        retained: &InferenceRetention,
        frontier: Option<u64>,
    ) -> Result<(), WorkingMemoryError> {
        if retained.established_revision() != Some(&self.revision) || frontier != self.frontier {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        match (self.admission.as_ref(), retained.admission()) {
            (None, None) => Ok(()),
            (Some(a), Some(b)) if a.position() == b.position() => {
                a.request().validate_same_request(b.request())
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    fn validate_run(
        &self,
        pending: &PendingTextBranchExchange,
        incoming: bool,
    ) -> Result<(), WorkingMemoryError> {
        let (request, context) = pending.source_request(incoming);
        request.validate(&self.execution, request.geometry())?;
        if context.attempt() != 0 {
            self.admission
                .as_ref()
                .ok_or(WorkingMemoryError::IdentityMismatch)?
                .request()
                .validate_same_request(request)?;
        }
        Ok(())
    }
}

/// Source-to-destination identity receipt minted only after the typed state move.
#[derive(Debug)]
pub struct ControlBranchPlacement {
    source: ControlBranchSource,
    destination: InferenceStateRevision,
}
impl ControlBranchPlacement {
    /// Exact pre-move source; retained only for deferred outgoing-machine pairing.
    pub fn source(&self) -> &ControlBranchSource {
        &self.source
    }
    /// Validates a displaced slot which has not executed since this transition.
    pub fn validate_destination(
        &self,
        actual: &ControlBranchSource,
    ) -> Result<(), WorkingMemoryError> {
        if self.destination != actual.revision
            || self.source.frontier != actual.frontier
            || !self.source.control.matches(&actual.control)
            || !self.source.execution.same_execution(&actual.execution)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        match (self.source.admission.as_ref(), actual.admission.as_ref()) {
            (None, None) => Ok(()),
            (Some(a), Some(b)) if a.position() == b.position() => {
                a.request().validate_same_request(b.request())
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    /// Composes successive actual moves before the displaced machine is paired.
    pub fn through(self, previous: &Self) -> Result<Self, WorkingMemoryError> {
        previous.validate_destination(&self.source)?;
        Ok(Self {
            source: previous.source.clone(),
            destination: self.destination,
        })
    }
    /// Authenticates an actual pending token/opening against its prior placement.
    pub fn matches_source(&self, revision: &InferenceStateRevision) -> bool {
        self.source.revision == *revision
    }
    /// Fresh funded revision of this same moved branch, never a new run allowance.
    pub fn revision(&self) -> &InferenceStateRevision {
        &self.destination
    }
    /// Rebinds exact prepared media through this completed placement transition.
    pub fn media_transition(
        &self,
        source: &MediaSessionBinding,
    ) -> Result<CopiedMediaStateBinding, WorkingMemoryError> {
        if !source.matches_snapshot(
            &self.source.execution,
            &self.source.revision,
            self.source
                .frontier
                .ok_or(WorkingMemoryError::UnknownBound)?,
        ) || !source.control.matches(&self.source.control)
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(CopiedMediaStateBinding::new(
            source,
            MediaSessionBinding {
                execution: self.source.execution.clone(),
                revision: self.destination.clone(),
                control: self.source.control.clone(),
                frontier: source.frontier,
            },
        ))
    }
}

/// Results of the one shared exchange worker. Ordinary exchange has no receipt;
/// resume may bind media, and branch exchange retains both exact placements.
#[derive(Debug)]
pub struct ControlExchangeResult {
    pub media: Option<CopiedMediaStateBinding>,
    pub placements: Option<[ControlBranchPlacement; 2]>,
    pub displaced: Option<ControlBranchPlacement>,
}

impl<A, B, M, D> ReplicatedTextSession<A, B, M, D>
where
    B: SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>,
    M: ReplicatedTextSnapshotMechanisms<A, B>,
    A: LayeredArchitecture<B, M::State>,
    D: ReplicatedTextExecutionStrategy<A, B, M::State, M::ResidentPolicy, M::BoundedPolicy>,
    A::Error: std::fmt::Display,
    M::PolicyError: std::fmt::Display,
    M::Error: std::fmt::Display,
{
    /// Borrows exact installed/incoming placement facts, without cloning storage.
    pub fn original_control_branch_sources(
        &self,
        slot: &ReplicatedTextControlState<M::State>,
    ) -> Result<[ControlBranchSource; 2], PreparedControlExchangeError<A::Error, M::PolicyError>>
    {
        self.validate_control_state_fixed(slot)?;
        Ok([
            self.control_branch_source(&self.state)?,
            self.control_branch_source(&slot.state)?,
        ])
    }
    pub(super) fn control_branch_source(
        &self,
        state: &M::State,
    ) -> Result<ControlBranchSource, WorkingMemoryError> {
        let retained = state.inference_retention();
        Ok(ControlBranchSource {
            revision: retained
                .established_revision()
                .ok_or(WorkingMemoryError::IdentityMismatch)?
                .clone(),
            admission: retained.admission().cloned(),
            execution: self.prefill_identity.clone(),
            control: self.control_identity.clone(),
            frontier: self.mechanisms.original_prefill_state_frontier(state)?,
        })
    }
    pub(super) fn validate_branch_sources(
        &self,
        slot: &ReplicatedTextControlState<M::State>,
        pending: &PendingTextBranchExchange,
        sources: &[ControlBranchSource; 2],
    ) -> Result<(), WorkingMemoryError> {
        pending.validate()?;
        for (index, state) in [&self.state, &slot.state].into_iter().enumerate() {
            let source = &sources[index];
            if !source.control.matches(&self.control_identity)
                || !source.execution.same_execution(&self.prefill_identity)
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            source.validate_state(
                state.inference_retention(),
                self.mechanisms.original_prefill_state_frontier(state)?,
            )?;
            source.validate_run(pending, index == 1)?;
        }
        Ok(())
    }
}

pub(super) fn placements(
    sources: &[ControlBranchSource; 2],
    installed: InferenceStateRevision,
    incoming: InferenceStateRevision,
) -> [ControlBranchPlacement; 2] {
    [
        ControlBranchPlacement {
            source: sources[0].clone(),
            destination: installed,
        },
        ControlBranchPlacement {
            source: sources[1].clone(),
            destination: incoming,
        },
    ]
}

pub(super) fn displaced(
    source: ControlBranchSource,
    destination: InferenceStateRevision,
) -> ControlBranchPlacement {
    ControlBranchPlacement {
        source,
        destination,
    }
}
