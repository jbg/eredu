//! Original reset metadata shares the ordinary physical-domain account.
use super::*;

/// Original reset preparation's host funding and accounting identity.
/// Every native communication producer retains its own admitted storage custody.
#[derive(Clone, Debug)]
pub struct SessionResetPreparationFunding {
    funding: HostMetadataFunding,
    pool: MemoryLedger,
}
impl SessionResetPreparationFunding {
    /// The ordinary account used by source and transport metadata constructors.
    pub fn metadata(&self) -> &HostMetadataFunding {
        &self.funding
    }
    /// Confirms the same coordinator before a separately admitted producer.
    pub fn validate_ledger(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        if self.pool.same_ledger(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}
impl MemoryLedger {
    /// Starts host preparation with the actual reset's per-domain constraints.
    /// State construction later reserves its complete attributed requirements.
    pub fn prepare_reset_metadata<T>(
        &self,
        session: &T,
        claim: &eredu_core::SessionResetClaim<'_>,
        execution: &InferenceExecutionIdentity,
        _state_bytes: u64,
    ) -> Result<SessionResetPreparationFunding, HostMetadataFundingError> {
        claim
            .validate_session(session)
            .map_err(|_| HostMetadataFundingError::Unavailable)?;
        let limits = claim
            .limits()
            .memory_limits
            .resolve(self.topology())
            .map_err(|_| HostMetadataFundingError::Unavailable)?;
        let funding = self.prepare_workspace_metadata(execution, limits)?;
        funding.reserve_metadata(size_of::<SessionResetPreparationFunding>())?;
        Ok(SessionResetPreparationFunding {
            funding,
            pool: self.clone(),
        })
    }
}
