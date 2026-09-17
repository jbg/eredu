//! The caller's ceiling is live before stop compilation and prompt encoding.
use super::{OriginalTextSourceError, OriginalTokenizer};
use crate::working_memory::{
    funding::{AccountNode, AccountTicket, PendingAccount, PendingOriginal},
    InferenceExecutionIdentity, PreparedAccountCommit, WorkingMemoryError,
};
use std::mem::size_of;

/// Retains a domain ceiling while originally admitted text sources are prepared.
/// This owns only its fixed accounting storage; it authorizes no native work,
/// source allocation, or inference. Each source still reserves its own full cost.
/// Successful request admission must establish the same ceiling before this retires.
#[derive(Debug)]
pub struct OriginalTextSourceBudget(AccountTicket);

impl OriginalTextSourceBudget {
    /// Fixed managed storage retained by this ceiling, excluding source allocations.
    pub fn storage_bytes() -> Result<u64, WorkingMemoryError> {
        [
            size_of::<AccountNode>(),
            size_of::<PendingOriginal>(),
            size_of::<PendingAccount>(),
            size_of::<AccountTicket>(),
            size_of::<PreparedAccountCommit<'_>>(),
            size_of::<OriginalTextSourceBudget>(),
            size_of::<OriginalTextSourceBudgetError>(),
            size_of::<OriginalTextSourceError>(),
            size_of::<Result<OriginalTextSourceBudget, OriginalTextSourceBudgetError>>(),
            size_of::<Result<OriginalTextSourceBudget, OriginalTextSourceError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or(WorkingMemoryError::Overflow)
    }
}

/// A fixed refusal or a failed publication retaining its real accounting owner.
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
pub struct OriginalTextSourceBudgetError {
    #[source]
    cause: WorkingMemoryError,
    // Last: the cause and any publication controls retire before their account.
    _budget: Option<OriginalTextSourceBudget>,
}
impl OriginalTextSourceBudgetError {
    fn rejected(cause: WorkingMemoryError) -> Self {
        Self {
            cause,
            _budget: None,
        }
    }
}

impl OriginalTokenizer {
    /// Installs an originally charged ceiling in the existing domain account
    /// ledger before request source construction. The execution identity is the
    /// actual selected session's identity, not an execution or submission grant.
    pub fn prepare_text_source_budget(
        &self,
        execution: &InferenceExecutionIdentity,
        capacity: u64,
    ) -> Result<OriginalTextSourceBudget, OriginalTextSourceBudgetError> {
        let pool = self.pool();
        // The node is the only new heap allocation. Its complete constructor,
        // pending publication, by-value error and return controls share the hold.
        let bytes = OriginalTextSourceBudget::storage_bytes()
            .map_err(OriginalTextSourceBudgetError::rejected)?;
        let pending = {
            let mut usage = pool.0.usage.lock().map_err(|_| {
                OriginalTextSourceBudgetError::rejected(WorkingMemoryError::Poisoned)
            })?;
            let commit =
                PreparedAccountCommit::prepare(pool, execution, &usage, bytes, Some(capacity), &[])
                    .map_err(OriginalTextSourceBudgetError::rejected)?;
            PendingAccount::accept(
                pool,
                execution,
                &mut usage,
                commit,
                bytes,
                Some(capacity),
                bytes,
            )
            .map_err(OriginalTextSourceBudgetError::rejected)?
        };
        // Existing pending publication holds the ceiling before Box::new; the
        // ledger then retains it until the physical node has been destroyed.
        let budget = OriginalTextSourceBudget(pending.publish());
        match budget.0.status() {
            Ok(()) => Ok(budget),
            Err(cause) => Err(OriginalTextSourceBudgetError {
                cause,
                _budget: Some(budget),
            }),
        }
    }
}

#[cfg(test)]
mod tests;
