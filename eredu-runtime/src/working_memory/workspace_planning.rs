//! Host planning participates in the same domain before each allocation.
use super::{
    InferenceExecutionIdentity, PreparedAccountCommit, WorkingMemoryError, WorkingMemoryPool,
    funding::{AccountNode, AccountTicket, PendingAccount, PendingOriginal},
};
use eredu_nn::workspace::{
    WorkspaceMetadataAccount, WorkspaceMetadataFunding, WorkspaceMetadataFundingError,
};
use std::mem::size_of;

// The closed NN funding owner destroys its erasure and shared shells before
// this ticket retires the numeric node. No native scope can be minted from it.
#[derive(Debug)]
struct PlanningAccount {
    ticket: AccountTicket,
}

fn failure(cause: WorkingMemoryError) -> WorkspaceMetadataFundingError {
    match cause {
        WorkingMemoryError::BudgetExceeded {
            required_bytes,
            available_bytes,
        } => WorkspaceMetadataFundingError::Capacity {
            required: required_bytes,
            available: available_bytes,
        },
        WorkingMemoryError::CapacityBelowUsage { .. } => WorkspaceMetadataFundingError::Unavailable,
        WorkingMemoryError::Overflow => WorkspaceMetadataFundingError::Overflow,
        _ => WorkspaceMetadataFundingError::Unavailable,
    }
}

impl WorkspaceMetadataAccount for PlanningAccount {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), WorkspaceMetadataFundingError> {
        let bytes = u64::try_from(bytes).map_err(|_| WorkspaceMetadataFundingError::Overflow)?;
        let pool = self.ticket.pool();
        let mut usage = pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkspaceMetadataFundingError::Unavailable)?;
        // Pending publication and unquoted owners obey the same exclusion as
        // an ordinary request. There is no optimistic available-then-reserve gap.
        if usage.unquoted_owners != 0 || usage.pending_original.is_some() {
            return Err(WorkspaceMetadataFundingError::Unavailable);
        }
        usage
            .funding
            .validate_constructing(self.ticket.id())
            .map_err(failure)?;
        let available = pool.0.available(&usage, None).map_err(failure)?;
        if bytes > available {
            return Err(WorkspaceMetadataFundingError::Capacity {
                required: bytes,
                available,
            });
        }
        // Compute every fallible scalar before mutating either the node or pool.
        let reserved = usage
            .reserved
            .checked_add(bytes)
            .ok_or(WorkspaceMetadataFundingError::Overflow)?;
        let used = pool
            .0
            .existing
            .checked_add(usage.registered)
            .and_then(|value| value.checked_add(reserved))
            .ok_or(WorkspaceMetadataFundingError::Overflow)?;
        usage
            .funding
            .grow_planning(self.ticket.id(), bytes)
            .map_err(failure)?;
        usage.reserved = reserved;
        usage.peak = usage.peak.max(used);
        Ok(())
    }
}

impl WorkingMemoryPool {
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
        capacity: u64,
    ) -> Result<WorkspaceMetadataFunding, WorkspaceMetadataFundingError> {
        let controls = [
            size_of::<PlanningAccount>(),
            size_of::<AccountNode>(),
            size_of::<AccountTicket>(),
            size_of::<PendingAccount>(),
            size_of::<PendingOriginal>(),
            size_of::<PreparedAccountCommit<'_>>(),
            size_of::<WorkspaceMetadataFundingError>(),
            size_of::<Result<WorkspaceMetadataFunding, WorkspaceMetadataFundingError>>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(std::mem::size_of_val(&controls), usize::checked_add)
            .and_then(|value| u64::try_from(value).ok())
            .ok_or(WorkspaceMetadataFundingError::Overflow)?;
        let pending = {
            let mut usage = self
                .0
                .usage
                .lock()
                .map_err(|_| WorkspaceMetadataFundingError::Unavailable)?;
            let commit =
                PreparedAccountCommit::prepare(self, execution, &usage, bytes, Some(capacity), &[])
                    .map_err(failure)?;
            PendingAccount::accept(
                self,
                execution,
                &mut usage,
                commit,
                bytes,
                Some(capacity),
                bytes,
            )
            .map_err(failure)?
        };
        let account = PlanningAccount {
            ticket: pending.publish(),
        };
        account.ticket.status().map_err(failure)?;
        // The shared erasure constructor debits this same account before either
        // of its allocations. A refused constructor has published no metadata.
        WorkspaceMetadataFunding::new(account)
    }
}

#[cfg(test)]
mod tests;
