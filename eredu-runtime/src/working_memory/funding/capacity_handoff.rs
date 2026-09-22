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

impl MemoryLedger {
    /// Rejects an ineligible ceiling succession before constructing planning
    /// metadata. This read-only observation grants no reservation or execution
    /// authority; reservation repeats the same validation under its commit lock.
    pub fn preflight_capacity_handoff(
        &self,
        execution: &InferenceExecutionIdentity,
        requested: &MemoryLimits,
        handoffs: &[WorkingMemoryCapacityHandoff],
    ) -> Result<(), WorkingMemoryError> {
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        CapacityHandoffPlan::prepare(self, execution, &usage, Some(requested), handoffs)?;
        Ok(())
    }

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
    requested: Option<&'h MemoryLimits>,
    handoffs: &'h [WorkingMemoryCapacityHandoff],
}
impl<'h> CapacityHandoffPlan<'h> {
    pub(in super::super) fn prepare(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &Usage,
        requested: Option<&'h MemoryLimits>,
        handoffs: &'h [WorkingMemoryCapacityHandoff],
    ) -> Result<Self, WorkingMemoryError> {
        if let Some(limits) = requested {
            limits.validate(pool.topology())?;
        }
        pool.validate_capacity_handoff_identities(execution, handoffs)?;
        for handoff in handoffs {
            if handoff.id >= usage.next_funding {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            let Some(state) = usage.funding.get(&handoff.id) else {
                let retiring = usage
                    .funding
                    .live_identity(handoff.id)
                    .or_else(|| {
                        usage
                            .account_retiring
                            .as_ref()
                            .and_then(|node| node.identity(handoff.id))
                    })
                    .or_else(|| {
                        usage
                            .pending_original
                            .as_ref()
                            .and_then(|node| node.identity(handoff.id))
                    });
                if let Some(identity) = retiring {
                    return Err(if identity != handoff.execution.as_ptr() as usize {
                        WorkingMemoryError::IdentityMismatch
                    } else {
                        WorkingMemoryError::ExecutionFenced
                    });
                }
                continue;
            };
            if !Weak::ptr_eq(&state.execution, &handoff.execution) {
                return Err(WorkingMemoryError::IdentityMismatch);
            }
            let mut raises = false;
            for (domain, _) in pool.topology().domains() {
                let next = requested
                    .map(|c| c.get(domain))
                    .transpose()?
                    .unwrap_or(MemoryLimit::Unlimited);
                raises |= state.limit(domain)?.is_raised_by(next);
            }
            if raises
                && (state.run_open
                    || state.scopes != 0
                    || state.quarantined
                    || state
                        .domains
                        .iter()
                        .any(|balance| balance.native_held.is_some())
                    || state.domains.iter().enumerate().any(|(slot, balance)| {
                        balance.remaining
                            != if slot == state.host_slot {
                                state.control_floor
                            } else {
                                0
                            }
                    }))
            {
                return Err(WorkingMemoryError::ExecutionFenced);
            }
        }
        Ok(Self {
            requested,
            handoffs,
        })
    }
    pub(in super::super) fn effective_capacity(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
        domain: MemoryDomainId,
    ) -> Result<MemoryLimit, WorkingMemoryError> {
        let mut limit = pool
            .0
            .limits
            .get(domain)?
            .minimum(
                usage
                    .funding
                    .capacity_after(domain, self.requested, self.handoffs)?,
            );
        if usage.account_retiring.is_some() {
            limit = limit.minimum(usage.domains[pool.topology().slot(domain)?].retiring_limit);
        }
        if let Some(account) = &usage.pending_original {
            limit = limit.minimum(account.capacity(domain)?);
        }
        limit = limit.minimum(crate::working_memory::resident_reset::capacity(
            usage, domain,
        )?);
        if let Some(requested) = self.requested {
            limit = limit.minimum(requested.get(domain)?);
        }
        Ok(limit)
    }
    pub(in super::super) fn commit(self, usage: &mut Usage) {
        usage.funding.commit_handoffs(self.requested, self.handoffs);
    }
}

#[cfg(test)]
mod tests;
