//! Exact accepted source account, separate from finite construction authority.
use super::{
    MemoryLedger, OriginalHostMetadataCustody, OriginalOperationMetadataCustody,
    OriginalRealtimeBudgetCustody, OriginalSpeculativeBudgetCustody, OriginalTextControlGuard,
    Usage, WorkingMemoryError, WorkingMemoryFundingScope, WorkingMemoryReservation,
};

/// Retains the actual source publisher. This alias cannot create a bank, debit
/// bytes, publish storage or turn metadata custody into a source receipt.
#[derive(Debug, Clone)]
pub struct OriginalHostSourceCustody(Custody);
#[derive(Debug, Clone)]
enum Custody {
    Text(OriginalTextControlGuard),
    Speculative(OriginalSpeculativeBudgetCustody),
    Realtime(OriginalRealtimeBudgetCustody),
    Numerical(super::OriginalNumericalBudgetCustody),
}
impl From<OriginalTextControlGuard> for OriginalHostSourceCustody {
    fn from(value: OriginalTextControlGuard) -> Self {
        Self(Custody::Text(value))
    }
}
impl From<OriginalSpeculativeBudgetCustody> for OriginalHostSourceCustody {
    fn from(value: OriginalSpeculativeBudgetCustody) -> Self {
        Self(Custody::Speculative(value))
    }
}
impl From<OriginalRealtimeBudgetCustody> for OriginalHostSourceCustody {
    fn from(value: OriginalRealtimeBudgetCustody) -> Self {
        Self(Custody::Realtime(value))
    }
}
impl From<super::OriginalNumericalBudgetCustody> for OriginalHostSourceCustody {
    fn from(value: super::OriginalNumericalBudgetCustody) -> Self {
        Self(Custody::Numerical(value))
    }
}
impl OriginalHostSourceCustody {
    /// Borrow the already accepted origin; callers still validate under Usage.
    pub(in crate::working_memory) fn origin(
        &self,
    ) -> (&MemoryLedger, &super::InferenceExecutionIdentity, u64) {
        match &self.0 {
            Custody::Text(value) => {
                let raw = value.custody.raw();
                (raw.pool(), raw.execution(), raw.account())
            }
            Custody::Speculative(value) => (value.pool(), value.execution(), value.account_id()),
            Custody::Realtime(value) => (value.pool(), value.execution(), value.account_id()),
            Custody::Numerical(value) => (value.pool(), value.execution(), value.account_id()),
        }
    }

    /// Exact accepted owner equality, not equal capacity, geometry or pool.
    pub fn same_source(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Custody::Text(a), Custody::Text(b)) => a.custody.same(&b.custody),
            (Custody::Speculative(a), Custody::Speculative(b)) => a.same_account(b),
            (Custody::Realtime(a), Custody::Realtime(b)) => a.same_account(b),
            (Custody::Numerical(a), Custody::Numerical(b)) => a.same_account(b),
            _ => false,
        }
    }
    /// Metadata retention only, with no source/native payload backedge.
    pub fn metadata_custody(&self) -> OriginalOperationMetadataCustody {
        match &self.0 {
            Custody::Text(value) => value.metadata_custody().into(),
            Custody::Speculative(value) => value.clone().into(),
            Custody::Realtime(value) => value.clone().into(),
            Custody::Numerical(value) => value.clone().into(),
        }
    }
    pub(in crate::working_memory) fn accounting(&self) -> OriginalHostMetadataCustody {
        match &self.0 {
            Custody::Text(value) => OriginalHostMetadataCustody::from_guard(value),
            Custody::Speculative(value) => OriginalHostMetadataCustody::from_budget(value.clone()),
            Custody::Realtime(value) => OriginalHostMetadataCustody::from_realtime(value.clone()),
            Custody::Numerical(value) => OriginalHostMetadataCustody::from_numerical(value.clone()),
        }
    }
    /// Revalidate the original accepted account. Text requires its exact
    /// reservation; speculative source custody accepts only its own role ticket.
    /// This health check issues neither a bank nor publication permission.
    pub fn validate_account(
        &self,
        reservation: Option<&WorkingMemoryReservation>,
    ) -> Result<(), WorkingMemoryError> {
        match (&self.0, reservation) {
            (Custody::Text(value), Some(reservation)) => value.validate_reservation(reservation),
            (Custody::Speculative(value), None) => {
                let usage = value
                    .pool()
                    .0
                    .usage
                    .lock()
                    .map_err(|_| WorkingMemoryError::Poisoned)?;
                value.validate_copy_source(value.pool(), &usage)
            }
            (Custody::Realtime(value), None) => {
                let usage = value
                    .pool()
                    .0
                    .usage
                    .lock()
                    .map_err(|_| WorkingMemoryError::Poisoned)?;
                value.validate_copy_source(value.pool(), &usage)
            }
            (Custody::Numerical(value), None) => {
                let usage = value
                    .pool()
                    .0
                    .usage
                    .lock()
                    .map_err(|_| WorkingMemoryError::Poisoned)?;
                value.validate_copy_source(value.pool(), &usage)
            }
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    pub(in crate::working_memory) fn validate_publication_locked(
        &self,
        reservation: Option<&WorkingMemoryReservation>,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        match (&self.0, reservation) {
            (Custody::Text(value), Some(reservation)) => {
                value.custody.raw().validate_origin_locked(pool, usage)?;
                value
                    .custody
                    .validate_source_publication_locked(usage, reservation)
            }
            (Custody::Speculative(value), None) => value.validate_copy_source(pool, usage),
            (Custody::Realtime(value), None) => value.validate_copy_source(pool, usage),
            (Custody::Numerical(value), None) => value.validate_copy_source(pool, usage),
            _ => Err(WorkingMemoryError::IdentityMismatch),
        }
    }
    /// Funded text scopes account registration bookkeeping independently of
    /// their source hold. A speculative ticket remains Unfunded; its exact
    /// immutable origin retains the whole accepted account through the final
    /// canonical row, so it must never enter funded allocation retirement.
    pub(in crate::working_memory) fn funded_registration_account(&self) -> Option<u64> {
        match &self.0 {
            Custody::Text(value) => Some(value.custody.raw().account()),
            Custody::Speculative(_) | Custody::Realtime(_) | Custody::Numerical(_) => None,
        }
    }
    pub(in crate::working_memory) fn text_scope(
        &self,
    ) -> Result<Option<&WorkingMemoryFundingScope>, WorkingMemoryError> {
        match &self.0 {
            Custody::Text(value) => value.custody.source_scope().map(Some),
            Custody::Speculative(_) | Custody::Realtime(_) | Custody::Numerical(_) => Ok(None),
        }
    }
}
