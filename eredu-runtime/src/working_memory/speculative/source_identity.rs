//! Retained identity only: no request, issue permission, role list or native owner.
use super::*;

/// Identifies the original issuance owner, independently of metadata funding.
/// Cloning retains only its pool identity; it cannot issue an invocation.
#[derive(Clone, Debug)]
pub struct OriginalSpeculativeSourceIdentity {
    pub(super) schedule: ScheduleIdentity,
    pub(super) account: u64,
    pub(super) pool: MemoryLedger,
}
impl OriginalSpeculativeRequest {
    /// Compares the exact retained executable without granting any invocation.
    /// An equal geometry or capacity from another execution is insufficient.
    pub fn validate_execution(
        &self,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        if std::sync::Arc::ptr_eq(&self.execution.0, &execution.0) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    /// Retain exact source identity for operations that outlive a callback loan.
    /// Consumers still require a separate accepted operation and source proof.
    pub fn source_identity(&self) -> OriginalSpeculativeSourceIdentity {
        OriginalSpeculativeSourceIdentity {
            schedule: self.identity,
            account: self.ticket.id(),
            pool: self.ticket.pool().clone(),
        }
    }
}
impl OriginalSpeculativeSourceIdentity {
    /// Exact retained issuance identity; this comparison grants no operation.
    pub fn same_identity(&self, other: &Self) -> bool {
        self.schedule == other.schedule
            && self.account == other.account
            && self.pool.same_ledger(&other.pool)
    }

    /// Same schedule, issuance account and pool as the actual request. A shared
    /// header or metadata account alone cannot authorize another request's copy.
    /// This reads immutable identities and allocates no owner or diagnostic.
    pub fn belongs_to_request(&self, request: &OriginalSpeculativeRequest) -> bool {
        self.schedule == request.identity
            && self.account == request.ticket.id()
            && self.pool.same_ledger(request.ticket.pool())
    }

    /// Pool identity alone grants no source provenance or native permission.
    pub fn pool(&self) -> &MemoryLedger {
        &self.pool
    }
}
