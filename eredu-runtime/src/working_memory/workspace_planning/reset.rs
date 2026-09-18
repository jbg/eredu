//! One genuine reset's cumulative preparation policy, separate from storage authority.
use super::*;
use std::{alloc::Layout, sync::{Arc, Mutex}, mem::size_of_val};

#[derive(Debug)]
struct Usage { used: u64, limit: u64 }
impl Usage {
    fn next(&self, bytes: u64) -> Result<u64, HostMetadataFundingError> {
        let next = self.used.checked_add(bytes).ok_or(HostMetadataFundingError::Overflow)?;
        if next > self.limit {
            return Err(HostMetadataFundingError::Capacity { required: bytes, available: self.limit.saturating_sub(self.used) });
        }
        Ok(next)
    }
}
#[derive(Debug)]
struct ResetAccount { usage: Mutex<Usage>, account: PlanningAccount }
// The final Arc shell retires before its original planning ticket. No Weak or
// raw Arc is exposed, so a policy handle cannot escape its own paid allocation.
#[derive(Debug)]
struct Shared(Option<Arc<ResetAccount>>);
impl Clone for Shared { fn clone(&self) -> Self { Self(self.0.clone()) } }
impl Drop for Shared { fn drop(&mut self) { if let Some(value) = self.0.take() { drop(Arc::into_inner(value)); } } }
impl Shared { fn account(&self) -> &ResetAccount { self.0.as_deref().expect("live reset preparation") } }
impl HostMetadataAccount for Shared {
    fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
        let account = self.account();
        let mut usage = account.usage.lock().map_err(|_| HostMetadataFundingError::Unavailable)?;
        let next = usage.next(u64::try_from(bytes).map_err(|_| HostMetadataFundingError::Overflow)?)?;
        account.account.reserve_metadata(bytes)?;
        usage.used = next;
        Ok(())
    }
}
/// Original reset preparation's host account and cumulative application policy.
/// Native producers still need their own exact domain admission. This handle
/// can refuse such a producer; it grants no native storage, execution or refill.
#[derive(Clone, Debug)]
pub struct SessionResetPreparationFunding { funding: HostMetadataFunding, account: Shared }
impl SessionResetPreparationFunding {
    /// Borrow the same account for actual source/transport metadata producers.
    pub fn metadata(&self) -> &HostMetadataFunding { &self.funding }
    /// Check an independently admitted communication producer before construction.
    /// Its full requirement remains spent in this operation's policy even if a
    /// later domain admission fails; no physical storage is reserved here.
    pub fn charge_communication(&self, pool: &WorkingMemoryPool, bytes: u64) -> Result<(), WorkingMemoryError> {
        let account = self.account.account();
        if !account.account.ticket.pool().same_domain(pool) { return Err(WorkingMemoryError::IdentityMismatch); }
        let mut usage = account.usage.lock().map_err(|_| WorkingMemoryError::Poisoned)?;
        usage.used = usage.next(bytes).map_err(|cause| match cause {
            HostMetadataFundingError::Capacity { required, available } => WorkingMemoryError::BudgetExceeded { required_bytes: required, available_bytes: available },
            HostMetadataFundingError::Overflow => WorkingMemoryError::Overflow,
            _ => WorkingMemoryError::UnknownBound,
        })?;
        Ok(())
    }
}
impl WorkingMemoryPool {
    /// Starts preparation for the actual core-issued reset. The exact planned
    /// state/publication and safety bytes participate in policy immediately,
    /// but are reserved only by that unchanged source-bound state constructor.
    pub fn prepare_reset_metadata<T>(&self, session: &T, claim: &eredu_core::SessionResetClaim<'_>,
        execution: &InferenceExecutionIdentity, state_bytes: u64)
        -> Result<SessionResetPreparationFunding, HostMetadataFundingError> {
        claim.validate_session(session).map_err(|_| HostMetadataFundingError::Unavailable)?;
        let controls = [size_of::<ResetAccount>(), size_of::<Shared>(), size_of::<SessionResetPreparationFunding>(),
            size_of::<AccountNode>(), size_of::<AccountTicket>(), size_of::<PendingAccount>(),
            size_of::<PendingOriginal>(), size_of::<PreparedAccountCommit<'_>>(), size_of::<Usage>(),
            size_of::<Result<SessionResetPreparationFunding, HostMetadataFundingError>>(),
            Layout::new::<[usize; 2]>().extend(Layout::new::<ResetAccount>())
                .map_err(|_| HostMetadataFundingError::Overflow)?.0.pad_to_align().size()];
        let bytes = controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .and_then(|n| u64::try_from(n).ok()).ok_or(HostMetadataFundingError::Overflow)?;
        let limits = claim.limits();
        let mut usage = Usage { used: state_bytes.checked_add(limits.safety_reserve_bytes)
            .ok_or(HostMetadataFundingError::Overflow)?, limit: limits.application_memory_budget_bytes.unwrap_or(u64::MAX) };
        usage.used = usage.next(bytes)?;
        let pending = self.prepare_planning_account(execution, limits.capacity_bytes, bytes)?;
        let account = PlanningAccount { ticket: pending.publish() };
        account.ticket.status().map_err(failure)?;
        let shared = Shared(Some(Arc::new(ResetAccount { usage: Mutex::new(usage), account })));
        let funding = HostMetadataFunding::new(shared.clone())?;
        Ok(SessionResetPreparationFunding { funding, account: shared })
    }
}
