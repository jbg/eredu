//! Explicit successor policy for the ceilings of completed funding accounts.
//! Numeric accounting and storage origins remain in the original pool ledger.

use super::*;
use crate::working_memory::{InferenceExecutionIdentity, Pool};

/// Move-only delegation from one unique funding owner. This token authorizes
/// explicitly selected successor policy to raise this account's ceiling only
/// after the pool proves complete closure and certification. It grants no work,
/// submission, storage credit or permission to close an active run.
///
/// Weak pool/execution identities and a never-reused funding id keep this token
/// payload-free. It does not keep an otherwise retired account or pool alive.
/// Dropping it changes no ceiling. Historical reservation diagnostics remain
/// unchanged after a successful handoff; live descendants retain the adopted
/// ceiling even if the successor retires first.
#[derive(Debug)]
pub struct WorkingMemoryCapacityHandoff {
    pool: Weak<Pool>,
    execution: Weak<()>,
    id: u64,
}

impl WorkingMemoryCapacityHandoff {
    pub(super) fn account_id(&self) -> u64 {
        self.id
    }
    /// Cold accounting-only pruning hint. Retirement cannot be reversed because
    /// funding ids never repeat. Poison remains an error, not retirement proof.
    pub fn is_retired(&self) -> Result<bool, WorkingMemoryError> {
        let Some(pool) = self.pool.upgrade() else {
            return Ok(true);
        };
        let usage = pool
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if self.id >= usage.next_funding {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let identity = usage.funding.live_identity(self.id).or_else(|| {
            usage
                .account_retiring
                .as_ref()
                .and_then(|retiring| retiring.identity(self.id))
        });
        let identity = identity.or_else(|| {
            usage
                .pending_original
                .as_ref()
                .and_then(|node| node.identity(self.id))
        });
        match identity {
            Some(actual) if actual != self.execution.as_ptr() as usize => {
                Err(WorkingMemoryError::IdentityMismatch)
            }
            Some(_) => Ok(false),
            None => Ok(true),
        }
    }
}

impl WorkingMemoryFundingRun {
    /// Delegates ceiling succession once from this move-only funding owner.
    /// Minting changes no charges, ceiling or completion state. The recipient
    /// must opt into explicit successor policy; committing a raise revalidates
    /// the exact execution and fully settled account under the pool lock.
    pub fn take_capacity_handoff(
        &mut self,
    ) -> Result<WorkingMemoryCapacityHandoff, WorkingMemoryError> {
        if self.handoff_taken {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let state = usage
            .funding
            .get(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let token = WorkingMemoryCapacityHandoff {
            pool: Arc::downgrade(&self.pool.0),
            execution: state.execution.clone(),
            id: self.id,
        };
        self.handoff_taken = true;
        Ok(token)
    }
}

impl WorkingMemoryPool {
    /// Identity-only preflight for candidate planners. Eligibility and exact
    /// live account identity are still checked atomically during reservation.
    pub(in super::super) fn validate_capacity_handoff_identities(
        &self,
        execution: &InferenceExecutionIdentity,
        handoffs: &[WorkingMemoryCapacityHandoff],
    ) -> Result<(), WorkingMemoryError> {
        let pool = Arc::downgrade(&self.0);
        let execution = Arc::downgrade(&execution.0);
        for (index, handoff) in handoffs.iter().enumerate() {
            if !Weak::ptr_eq(&pool, &handoff.pool)
                || !Weak::ptr_eq(&execution, &handoff.execution)
                || handoffs[..index]
                    .iter()
                    .any(|earlier| earlier.id == handoff.id)
            {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
        }
        Ok(())
    }
}

/// Borrowed checked numeric transaction. No map clone, raise vector or source
/// callback is needed; the exact same lock covers validation and scalar commit.
pub(in super::super) struct CapacityHandoffPlan<'h> {
    capacity: u64,
    requested: Option<u64>,
    handoffs: &'h [WorkingMemoryCapacityHandoff],
}
impl<'h> CapacityHandoffPlan<'h> {
    pub(in super::super) fn prepare(
        pool: &WorkingMemoryPool,
        execution: &InferenceExecutionIdentity,
        usage: &Usage,
        requested: Option<u64>,
        handoffs: &'h [WorkingMemoryCapacityHandoff],
    ) -> Result<Self, WorkingMemoryError> {
        pool.validate_capacity_handoff_identities(execution, handoffs)?;
        for handoff in handoffs {
            if handoff.id >= usage.next_funding {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            let Some(state) = usage.funding.get(&handoff.id) else {
                let retiring = usage.funding.live_identity(handoff.id).or_else(|| {
                    usage
                        .account_retiring
                        .as_ref()
                        .and_then(|node| node.identity(handoff.id))
                });
                let retiring = retiring.or_else(|| {
                    usage
                        .pending_original
                        .as_ref()
                        .and_then(|node| node.identity(handoff.id))
                });
                if let Some(identity) = retiring {
                    if identity != handoff.execution.as_ptr() as usize {
                        return Err(WorkingMemoryError::IdentityMismatch);
                    }
                    return Err(WorkingMemoryError::ExecutionFenced);
                }
                continue;
            };
            if !Weak::ptr_eq(&state.execution, &handoff.execution) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            let (Some(current), Some(next)) = (state.capacity, requested) else {
                continue;
            };
            if current >= next {
                continue;
            }
            if state.run_open
                || state.scopes != 0
                || state.quarantined
                || state.native_held.is_some()
                || state.remaining != state.control_floor
            {
                return Err(WorkingMemoryError::ExecutionFenced);
            }
        }
        Ok(Self {
            capacity: crate::working_memory::resident_reset::capacity(usage)
                .min(usage.funding.capacity_after(requested, handoffs))
                .min(
                    usage
                        .account_retiring
                        .as_ref()
                        .map_or(u64::MAX, RetiringAccount::capacity),
                ),
            requested,
            handoffs,
        })
    }
    pub(in super::super) fn effective_capacity(
        &self,
        configured: u64,
        requested: Option<u64>,
    ) -> u64 {
        configured
            .min(self.capacity)
            .min(requested.unwrap_or(u64::MAX))
    }
    pub(in super::super) fn commit(self, usage: &mut Usage) {
        usage.funding.commit_handoffs(self.requested, self.handoffs);
    }
}

#[cfg(test)]
mod tests;
