//! Host planning participates in the same ledger before each allocation.
use super::{
    InferenceExecutionIdentity, MemoryLedger, PreparedAccountCommit, WorkingMemoryError,
    funding::{AccountNode, AccountTicket, PendingAccount, PendingOriginal},
};
use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
use std::mem::size_of;

// The closed NN funding owner destroys its erasure and shared shells before
// this ticket retires the numeric node. No native scope can be minted from it.
#[derive(Debug)]
struct PlanningAccount {
    ticket: AccountTicket,
}

/// Move-only construction authority for host metadata. Sealing closes growth
/// through every funding alias while preserving the charge and its custody.
/// This owner grants no native execution permission. Closed storage constructors
/// independently reserve and authenticate the backing they publish.
#[derive(Debug)]
pub struct PreparedConstructionMetadata {
    funding: HostMetadataFunding,
    pool: MemoryLedger,
    id: u64,
}

impl PreparedConstructionMetadata {
    /// Loans the existing account to constructors before their allocations.
    pub fn funding(&self) -> &HostMetadataFunding {
        &self.funding
    }

    /// Retains this genuine ledger-owned metadata source for closed storage
    /// constructors. Payload reservation remains separate and precedes birth;
    /// neither this loan nor publication grants native execution permission.
    pub fn storage_funding(
        &self,
    ) -> Result<super::StorageMetadataFunding, HostMetadataFundingError> {
        self.funding.reserve_metadata(size_of::<(
            &Self,
            super::StorageMetadataFunding,
            Result<super::StorageMetadataFunding, HostMetadataFundingError>,
        )>())?;
        Ok(super::StorageMetadataFunding::from_construction(
            &self.pool,
            &self.funding,
        ))
    }

    /// Retains the complete metadata charge and closes this producer's active
    /// construction exclusion. Keep the returned funding with the immutable
    /// payload until its final backing retires. Existing aliases retain custody
    /// but cannot fund further allocations, including zero-byte requests.
    pub fn seal(self) -> Result<HostMetadataFunding, HostMetadataFundingError> {
        {
            let mut usage = self
                .pool
                .0
                .usage
                .lock()
                .map_err(|_| HostMetadataFundingError::Poisoned)?;
            let reservations = usage
                .reservations
                .checked_sub(1)
                .ok_or(HostMetadataFundingError::Unavailable)?;
            usage
                .funding
                .seal_construction_metadata(self.id, self.pool.construction_identity())
                .map_err(failure)?;
            usage.reservations = reservations;
        }
        Ok(self.funding)
    }
}

fn failure(cause: WorkingMemoryError) -> HostMetadataFundingError {
    match cause {
        WorkingMemoryError::Domain(cause) => HostMetadataFundingError::Domain(cause),
        WorkingMemoryError::Poisoned => HostMetadataFundingError::Poisoned,
        WorkingMemoryError::Overflow => HostMetadataFundingError::Overflow,
        _ => HostMetadataFundingError::Unavailable,
    }
}

impl HostMetadataAccount for PlanningAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let bytes = u64::try_from(bytes).map_err(|_| HostMetadataFundingError::Overflow)?;
        let pool = self.ticket.pool();
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| HostMetadataFundingError::Poisoned)?;
        // Pending publication and unquoted owners obey the same exclusion as
        // an ordinary request. There is no optimistic available-then-reserve gap.
        if usage.unquoted_owners != 0 || usage.pending_original.is_some() {
            return Err(HostMetadataFundingError::Unavailable);
        }
        usage
            .funding
            .validate_constructing(self.ticket.id())
            .map_err(failure)?;
        pool.0
            .check_host_increment(&usage, bytes)
            .map_err(failure)?;
        // Compute every fallible scalar before mutating either the node or pool.
        let reserved = usage
            .reserved
            .checked_add(bytes)
            .ok_or(HostMetadataFundingError::Overflow)?;
        let used = pool
            .0
            .existing
            .checked_add(usage.registered)
            .and_then(|value| value.checked_add(reserved))
            .ok_or(HostMetadataFundingError::Overflow)?;
        usage
            .funding
            .grow_planning(self.ticket.id(), bytes)
            .map_err(failure)?;
        usage.reserved = reserved;
        usage.peak = usage.peak.max(used);
        Ok(())
    }
}

impl MemoryLedger {
    /// Starts host-only workspace planning under the real shared domain ceiling.
    /// Each participating constructor reserves its actual request before it
    /// allocates. Retired spans do not refund the cumulative planning charge.
    ///
    /// Retain the returned owner through all constructed metadata, including
    /// escaped reports, plans and errors. This grants no tensor storage,
    /// submission permission or completeness certificate for native execution.
    pub fn prepare_workspace_metadata(
        &self,
        execution: &InferenceExecutionIdentity,
        capacity: eredu_core::MemoryLimits,
    ) -> Result<HostMetadataFunding, HostMetadataFundingError> {
        self.prepare_metadata_account(execution, Some(capacity))
            .map(|(funding, _)| funding)
    }

    /// Funds cold constructor metadata before its allocations. Configured and
    /// live account limits apply to each allocation; this owner contributes no
    /// additional request ceiling. It grants neither native allocation nor
    /// submission authority. Seal completed immutable metadata and retain its
    /// funding through the payload. Partial results and errors retain funding
    /// from this same account until their allocations retire.
    pub fn prepare_construction_metadata(
        &self,
    ) -> Result<PreparedConstructionMetadata, HostMetadataFundingError> {
        let (funding, id) = self.prepare_metadata_account(self.construction_identity(), None)?;
        Ok(PreparedConstructionMetadata {
            funding,
            pool: self.clone(),
            id,
        })
    }

    fn prepare_metadata_account(
        &self,
        execution: &InferenceExecutionIdentity,
        capacity: Option<eredu_core::MemoryLimits>,
    ) -> Result<(HostMetadataFunding, u64), HostMetadataFundingError> {
        let controls = [
            size_of::<PlanningAccount>(),
            size_of::<AccountNode>(),
            size_of::<AccountTicket>(),
            size_of::<PendingAccount>(),
            size_of::<PendingOriginal>(),
            size_of::<PreparedAccountCommit<'_>>(),
            size_of::<Option<eredu_core::MemoryLimits>>(),
            size_of::<Result<PendingAccount, WorkingMemoryError>>(),
            size_of::<HostMetadataFundingError>(),
            size_of::<Result<HostMetadataFunding, HostMetadataFundingError>>(),
            size_of::<Result<(HostMetadataFunding, u64), HostMetadataFundingError>>(),
            size_of::<PreparedConstructionMetadata>(),
            size_of::<Result<PreparedConstructionMetadata, HostMetadataFundingError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(HostMetadataFundingError::Overflow)?;
        let pending = self.prepare_planning_account(execution, capacity, bytes)?;
        let account = PlanningAccount {
            ticket: pending.publish(),
        };
        account.ticket.status().map_err(failure)?;
        let id = account.ticket.id();
        // The shared erasure constructor debits this same account before either
        // of its allocations. A refused constructor has published no metadata.
        HostMetadataFunding::new(account).map(|funding| (funding, id))
    }
    fn prepare_planning_account(
        &self,
        execution: &InferenceExecutionIdentity,
        capacity: Option<eredu_core::MemoryLimits>,
        bytes: u64,
    ) -> Result<PendingAccount, HostMetadataFundingError> {
        let domain_bytes =
            super::funding::domain_balance_bytes(self.topology()).map_err(failure)?;
        let requirements =
            eredu_core::DomainMemoryRequirements::construction_backing_bytes(self.topology(), 0)
                .map_err(|_| HostMetadataFundingError::Overflow)?;
        let limits = capacity
            .as_ref()
            .map(eredu_core::MemoryLimits::backing_bytes)
            .transpose()
            .map_err(|_| HostMetadataFundingError::Overflow)?
            .unwrap_or(0);
        let bytes = bytes
            .checked_add(domain_bytes)
            .and_then(|bytes| bytes.checked_add(requirements))
            .and_then(|bytes| bytes.checked_add(limits))
            .ok_or(HostMetadataFundingError::Overflow)?;
        let mut usage = self
            .0
            .usage
            .lock()
            .map_err(|_| HostMetadataFundingError::Unavailable)?;
        PendingAccount::accept_planning(self, execution, &mut usage, bytes, capacity)
            .map_err(failure)
    }
}

mod reset;
pub use reset::SessionResetPreparationFunding;

#[cfg(test)]
mod tests;
