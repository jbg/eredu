//! Exact outer bank identity, separate from account-only slot storage.
use super::*;
use eredu_runtime::working_memory::{
    OriginalEmbeddedSpeculativeRole, OriginalExternalSpeculativeRole, OriginalHostSourceBank,
    OriginalSpeculativeBudgetCustody, OriginalSpeculativeRole,
};

#[derive(Debug, Clone)]
pub(super) enum OperationControls {
    Text(OriginalTextControlGuard),
    Speculative(SpeculativeOperationRole),
    Realtime(eredu_runtime::working_memory::OriginalRealtimeBudgetCustody),
}
impl OperationControls {
    fn validate_registry(&self, registry: &Registry) -> Result<(), Error> {
        match (self, &registry.request) {
            (Self::Text(controls), OperationRequest::Text(request)) => controls
                .validate_reservation(request.memory_reservation())
                .map_err(memory),
            (Self::Speculative(role), OperationRequest::Speculative(actual))
                if role.same_role(actual) =>
            {
                Ok(())
            }
            (Self::Realtime(custody), OperationRequest::Realtime(actual))
                if custody.same_account(actual) =>
            {
                Ok(())
            }
            _ => Err(identity()),
        }
    }
    pub(super) fn validate(&self, registry: &Registry) -> Result<(), Error> {
        self.validate_registry(registry)
    }
}

pub(super) enum OperationRequest {
    Text(InferenceRequest),
    Speculative(SpeculativeOperationRole),
    Realtime(eredu_runtime::working_memory::OriginalRealtimeBudgetCustody),
}
impl OperationRequest {
    pub(super) fn memory_reservation(
        &self,
    ) -> Option<&eredu_runtime::working_memory::WorkingMemoryReservation> {
        match self {
            Self::Text(request) => Some(request.memory_reservation()),
            Self::Speculative(_) | Self::Realtime(_) => None,
        }
    }
    pub(super) fn validate_same_request(
        &self,
        request: &InferenceRequest,
    ) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Text(actual) => actual.validate_same_request(request),
            Self::Speculative(_) | Self::Realtime(_) => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
}
impl From<InferenceRequest> for OperationRequest {
    fn from(request: InferenceRequest) -> Self {
        Self::Text(request)
    }
}

/// Exact native tag around existing role custody. Tags never compare equal to
/// each other and this carrier cannot manufacture a role or invocation claim.
#[derive(Debug, Clone)]
pub(in crate::backend::runtime::execution::generic) enum SpeculativeOperationRole {
    Autoregressive(OriginalSpeculativeRole),
    Embedded(OriginalEmbeddedSpeculativeRole),
    External(OriginalExternalSpeculativeRole),
}
impl SpeculativeOperationRole {
    pub(super) fn same_role(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Autoregressive(a), Self::Autoregressive(b)) => a.same_role(b),
            (Self::Embedded(a), Self::Embedded(b)) => a.same_role(b),
            (Self::External(a), Self::External(b)) => a.same_role(b),
            _ => false,
        }
    }
    pub(super) fn budget_custody(&self) -> OriginalSpeculativeBudgetCustody {
        match self {
            Self::Autoregressive(role) => role.budget_custody(),
            Self::Embedded(role) => role.budget_custody(),
            Self::External(role) => role.budget_custody(),
        }
    }
    pub(super) fn claim_neural_bank(&self, controls: u64) -> Result<(), WorkingMemoryError> {
        match self {
            Self::Autoregressive(role) => role.claim_neural_bank(controls),
            Self::Embedded(role) => role.claim_neural_bank(controls),
            Self::External(role) => role.claim_neural_bank(controls),
        }
    }
    pub(super) fn take_host_source_constructions(
        &self,
    ) -> Result<Option<OriginalHostSourceBank>, WorkingMemoryError> {
        match self {
            Self::Autoregressive(role) => role.take_host_source_constructions(),
            Self::Embedded(role) => role.take_host_source_constructions(),
            Self::External(role) => role.take_host_source_constructions(),
        }
    }
}
