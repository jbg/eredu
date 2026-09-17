//! Fixed numeric nodes. Generic source/provider payloads never enter this list.
//! Existing quarantined accounting pins remain in FundingState and may retain
//! the pool indirectly; such conservative nodes never enter terminal retirement.
use super::*;
mod original;
pub(in crate::working_memory) use original::{AccountTicket, PendingAccount, PendingOriginal};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Phase {
    Unfunded,
    Funded,
    Terminal,
}

#[derive(Debug)]
pub(in crate::working_memory) struct AccountNode {
    id: u64,
    phase: Phase,
    state: FundingState,
    next: Option<Box<AccountNode>>,
    planning_metadata: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
}
impl AccountNode {
    // Ordinary paths use their existing owner. Original requests use an accepted
    // pending slot; prepared copies use their enclosing admitted host plan.
    pub(in crate::working_memory) fn empty() -> Box<Self> {
        Self::empty_with_planning(None)
    }
    pub(in crate::working_memory) fn empty_with_planning(
        planning_metadata: Option<eredu_nn::workspace::WorkspaceMetadataFunding>,
    ) -> Box<Self> {
        Box::new(Self {
            id: 0,
            phase: Phase::Unfunded,
            state: FundingState::empty(),
            next: None,
            planning_metadata,
        })
    }

    // The closed ledger is the sole owner. Take custody out before dropping the
    // actual Box so its last token cannot refund a still-live shell or Weak.
    fn retire(mut node: Box<Self>) {
        let preparation = node.state.preparation.take();
        let planning_metadata = node.planning_metadata.take();
        drop(node);
        drop(preparation);
        drop(planning_metadata);
    }
}

#[derive(Debug, Default)]
pub(in crate::working_memory) struct AccountLedger {
    head: Option<Box<AccountNode>>,
}
impl AccountLedger {
    // Temporary reached-failure attribution; removed after the ownership diagnosis.
    pub(in crate::working_memory) fn trace_accounts(&self) {
        for node in self.nodes() {
            let state = &node.state;
            if state.remaining >= (1 << 20) {
                eprintln!("WORKSPACE_LEDGER_ACCOUNT id={} phase={:?} remaining={} host={} native={:?} scopes={} native_scopes={} run_open={} metadata_live={} quarantined={}",
                    node.id,node.phase,state.remaining,state.host_held,state.native_held,
                    state.scopes,state.native_scopes,state.run_open,state.metadata_live,state.quarantined);
            }
        }
    }
    fn nodes(&self) -> impl Iterator<Item = &AccountNode> {
        let mut next = self.head.as_deref();
        std::iter::from_fn(move || {
            let node = next?;
            next = node.next.as_deref();
            Some(node)
        })
    }
    #[cfg(test)]
    pub(in crate::working_memory) fn iter(&self) -> impl Iterator<Item = (&u64, &FundingState)> {
        self.nodes()
            .filter(|node| node.phase == Phase::Funded)
            .map(|node| (&node.id, &node.state))
    }
    pub(in crate::working_memory) fn values(&self) -> impl Iterator<Item = &FundingState> {
        self.nodes()
            .filter(|node| node.phase == Phase::Funded)
            .map(|node| &node.state)
    }
    pub(in crate::working_memory) fn values_mut(
        &mut self,
    ) -> impl Iterator<Item = &mut FundingState> {
        let mut next = self.head.as_deref_mut();
        std::iter::from_fn(move || {
            while let Some(node) = next.take() {
                next = node.next.as_deref_mut();
                if node.phase == Phase::Funded {
                    return Some(&mut node.state);
                }
            }
            None
        })
    }
    fn quarantine_all(&mut self) {
        let mut next = self.head.as_deref_mut();
        while let Some(node) = next {
            // Legacy unfunded reservation retirement owns its whole exact charge.
            // Original control floors and every funded account remain fenced.
            if node.phase == Phase::Funded || node.state.control_floor != 0 {
                node.state.quarantined = true;
            }
            next = node.next.as_deref_mut();
        }
    }
    pub(in crate::working_memory) fn get(&self, id: &u64) -> Option<&FundingState> {
        self.nodes()
            .find(|node| node.id == *id && node.phase == Phase::Funded)
            .map(|node| &node.state)
    }
    // A validated original source may retain its accepted account before native
    // funding starts. This reads only that live account's ceiling; callers must
    // authenticate the source role under the same Usage lock. General getters
    // remain Funded-only and no phase or construction authority changes here.
    pub(in crate::working_memory) fn accepted_source_capacity(
        &self,
        id: u64,
        execution: &InferenceExecutionIdentity,
    ) -> Result<u64, WorkingMemoryError> {
        self.validate_metadata(id, execution)?;
        self.nodes()
            .find(|node| node.id == id)
            .expect("validated source account")
            .state
            .accepted_capacity()
            .ok_or(WorkingMemoryError::UnknownBound)
    }
    pub(in crate::working_memory) fn get_mut(&mut self, id: &u64) -> Option<&mut FundingState> {
        let mut next = self.head.as_deref_mut();
        while let Some(node) = next {
            if node.id == *id && node.phase == Phase::Funded {
                return Some(&mut node.state);
            }
            next = node.next.as_deref_mut();
        }
        None
    }
    pub(in crate::working_memory) fn len(&self) -> usize {
        self.values().count()
    }
    #[cfg(test)]
    pub(in crate::working_memory) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(in crate::working_memory) fn capacity(&self) -> u64 {
        self.nodes()
            .filter_map(|node| node.state.capacity)
            .min()
            .unwrap_or(u64::MAX)
    }
    pub(in crate::working_memory) fn validate_metadata(
        &self,
        id: u64,
        execution: &InferenceExecutionIdentity,
    ) -> Result<(), WorkingMemoryError> {
        let node = self
            .nodes()
            .find(|node| node.id == id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if node.state.execution.as_ptr() != Arc::as_ptr(&execution.0) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if node.phase == Phase::Terminal || !node.state.metadata_live || node.state.quarantined {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }
    // Only a still-owning AccountTicket calls this. Its accepted account has
    // not entered WorkingMemoryFundingScope, so the general Funded getters
    // deliberately cannot reach it. Quarantine preserves phase and all charges.
    fn quarantine_original(&mut self, id: u64) {
        let mut next = self.head.as_deref_mut();
        while let Some(node) = next {
            if node.id == id {
                assert_eq!(node.phase, Phase::Unfunded);
                assert!(node.state.metadata_live);
                node.state.quarantined = true;
                return;
            }
            next = node.next.as_deref_mut();
        }
        panic!("live original source account");
    }

    pub(in crate::working_memory) fn validate_constructing(
        &self,
        id: u64,
    ) -> Result<(), WorkingMemoryError> {
        let node = self
            .nodes()
            .find(|node| node.id == id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if node.phase != Phase::Unfunded || !node.state.metadata_live || node.state.quarantined {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        Ok(())
    }
    // A host-only planning account never enters native funding. Its complete
    // cumulative charge remains protected through final metadata retirement.
    // The caller checks the domain ceiling and prepares the shared counters
    // under the same Usage lock before committing this allocation-free change.
    pub(in crate::working_memory) fn grow_planning(
        &mut self,
        id: u64,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        let mut next = self.head.as_deref_mut();
        while let Some(node) = next {
            if node.id == id {
                let state = &mut node.state;
                if node.phase != Phase::Unfunded
                    || !state.metadata_live
                    || state.quarantined
                    || state.remaining != state.control_floor
                    || state.host_held != state.control_floor
                {
                    return Err(WorkingMemoryError::ExecutionFenced);
                }
                let total = state
                    .remaining
                    .checked_add(bytes)
                    .ok_or(WorkingMemoryError::Overflow)?;
                state.remaining = total;
                state.control_floor = total;
                state.host_held = total;
                return Ok(());
            }
            next = node.next.as_deref_mut();
        }
        Err(WorkingMemoryError::IdentityMismatch)
    }
    pub(in crate::working_memory) fn live_identity(&self, id: u64) -> Option<usize> {
        self.nodes()
            .find(|node| node.id == id)
            .map(|node| node.state.execution.as_ptr() as usize)
    }
    pub(in crate::working_memory) fn publish(
        &mut self,
        mut node: Box<AccountNode>,
        id: u64,
        state: FundingState,
        funded: bool,
    ) {
        debug_assert!(self.nodes().all(|existing| existing.id != id));
        node.id = id;
        node.state = state; // Only the empty allocation-free placeholder retires here.
        node.phase = if funded {
            Phase::Funded
        } else {
            Phase::Unfunded
        };
        node.next = self.head.take();
        self.head = Some(node);
    }
    pub(in crate::working_memory) fn start(&mut self, id: u64) -> Result<(), WorkingMemoryError> {
        let mut next = self.head.as_deref_mut();
        while let Some(node) = next {
            if node.id == id {
                if node.phase != Phase::Unfunded {
                    return Err(WorkingMemoryError::IdentityMismatch);
                }
                node.phase = Phase::Funded;
                return Ok(());
            }
            next = node.next.as_deref_mut();
        }
        Err(WorkingMemoryError::IdentityMismatch)
    }
    pub(in crate::working_memory) fn retire_unfunded(&mut self, id: u64) {
        let mut next = self.head.as_deref_mut();
        while let Some(node) = next {
            if node.id == id {
                assert_eq!(node.phase, Phase::Unfunded);
                // Settlement is shared with a funded reservation whose run has closed.
                node.phase = Phase::Funded;
                node.state.run_open = false;
                node.state.metadata_live = false;
                return;
            }
            next = node.next.as_deref_mut();
        }
        panic!("live reservation account");
    }
    pub(in crate::working_memory) fn mark_terminal(&mut self, id: u64) {
        let mut next = self.head.as_deref_mut();
        while let Some(node) = next {
            if node.id == id {
                assert_eq!(node.phase, Phase::Funded);
                assert!(node.state.quarantined_borrowed.is_none());
                node.phase = Phase::Terminal;
                return;
            }
            next = node.next.as_deref_mut();
        }
        panic!("live terminal account");
    }
    fn take_terminal(&mut self, poisoned: bool) -> Option<Box<AccountNode>> {
        let mut link = &mut self.head;
        loop {
            if link.as_ref()?.phase == Phase::Terminal
                && (!poisoned || link.as_ref()?.state.control_floor == 0)
            {
                let mut node = link.take().expect("checked terminal node");
                *link = node.next.take();
                return Some(node);
            }
            link = &mut link.as_mut().expect("checked live link").next;
        }
    }
    // Original and ordinary capacity come from the same actual nodes. This
    // borrowed scan has no container growth or per-concurrent-request multiplier.
    pub(in crate::working_memory) fn capacity_after(
        &self,
        requested: Option<u64>,
        handoffs: &[WorkingMemoryCapacityHandoff],
    ) -> u64 {
        self.nodes()
            .filter_map(|node| {
                node.state.capacity.map(|current| {
                    if node.phase == Phase::Funded
                        && handoffs
                            .iter()
                            .any(|handoff| handoff.account_id() == node.id)
                    {
                        current.max(requested.unwrap_or(current))
                    } else {
                        current
                    }
                })
            })
            .min()
            .unwrap_or(u64::MAX)
    }
    pub(in crate::working_memory) fn commit_handoffs(
        &mut self,
        requested: Option<u64>,
        handoffs: &[WorkingMemoryCapacityHandoff],
    ) {
        let Some(next) = requested else {
            return;
        };
        for handoff in handoffs {
            if let Some(state) = self.get_mut(&handoff.account_id()) {
                if let Some(current) = &mut state.capacity {
                    *current = (*current).max(next);
                }
            }
        }
    }
    #[cfg(test)]
    pub(in crate::working_memory) fn capacity_counts(&self) -> BTreeMap<u64, usize> {
        let mut counts = BTreeMap::new();
        for cap in self.nodes().filter_map(|node| node.state.capacity) {
            *counts.entry(cap).or_default() += 1;
        }
        counts
    }
}
impl std::ops::Index<&u64> for AccountLedger {
    type Output = FundingState;
    fn index(&self, id: &u64) -> &FundingState {
        self.get(id).expect("live funding account")
    }
}
impl Drop for AccountLedger {
    fn drop(&mut self) {
        // Pool destruction is outside Usage. Avoid recursive linked-list drop.
        while let Some(mut node) = self.head.take() {
            self.head = node.next.take();
            AccountNode::retire(node);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(in crate::working_memory) struct RetiringAccount {
    id: u64,
    execution: usize,
    capacity: Option<u64>,
    floor: u64,
}
impl RetiringAccount {
    pub(in crate::working_memory) fn capacity(&self) -> u64 {
        self.capacity.unwrap_or(u64::MAX)
    }
    pub(in crate::working_memory) fn identity(&self, id: u64) -> Option<usize> {
        (id == self.id).then_some(self.execution)
    }
}

// Every cleanup caller uses this same guard. It unlocks before draining terminal
// node shells; old source/provider retirement keeps its existing detached order.
pub(in crate::working_memory) struct RetirementGuard<'a> {
    pool: &'a WorkingMemoryPool,
    usage: Option<std::sync::MutexGuard<'a, Usage>>,
}
impl std::ops::Deref for RetirementGuard<'_> {
    type Target = Usage;
    fn deref(&self) -> &Usage {
        self.usage.as_deref().expect("live retirement guard")
    }
}
impl std::ops::DerefMut for RetirementGuard<'_> {
    fn deref_mut(&mut self) -> &mut Usage {
        self.usage.as_deref_mut().expect("live retirement guard")
    }
}
impl Drop for RetirementGuard<'_> {
    fn drop(&mut self) {
        drop(self.usage.take());
        drain(self.pool);
    }
}
pub(in crate::working_memory) fn lock(pool: &WorkingMemoryPool) -> RetirementGuard<'_> {
    let usage = match pool.0.usage.lock() {
        Ok(usage) => usage,
        Err(poison) => {
            let mut usage = poison.into_inner();
            usage.funding.quarantine_all();
            usage
        }
    };
    RetirementGuard {
        pool,
        usage: Some(usage),
    }
}
pub(in crate::working_memory) fn drain(pool: &WorkingMemoryPool) {
    loop {
        let node = {
            let (mut usage, poisoned) = match pool.0.usage.lock() {
                Ok(usage) => (usage, false),
                Err(poison) => (poison.into_inner(), true),
            };
            if usage.account_retiring.is_some() {
                return;
            }
            let Some(node) = usage.funding.take_terminal(poisoned) else {
                return;
            };
            usage.account_retiring = Some(RetiringAccount {
                id: node.id,
                execution: node.state.execution.as_ptr() as usize,
                capacity: node.state.capacity,
                floor: node.state.control_floor,
            });
            node
        };
        // Both the actual Box and all weak/account controls retire before credit.
        // Terminal predicates exclude quarantine chains and any active span.
        // A preparation-token callback can retire another account, but the
        // existing account_retiring slot prevents recursive node draining.
        AccountNode::retire(node);
        #[cfg(test)]
        AFTER_NODE_RETIRE.with(|hook| {
            let hook = hook.borrow_mut().take();
            if let Some(hook) = hook {
                hook(pool);
            }
        });
        let mut usage = match pool.0.usage.lock() {
            Ok(usage) => usage,
            Err(poison) => {
                let usage = poison.into_inner();
                if usage
                    .account_retiring
                    .as_ref()
                    .is_some_and(|node| node.floor != 0)
                {
                    return;
                }
                usage
            }
        };
        let retiring = usage
            .account_retiring
            .take()
            .expect("one active account drainer");
        usage.reserved = usage
            .reserved
            .checked_sub(retiring.floor)
            .expect("retained control floor");
        usage.reservations = usage
            .reservations
            .checked_sub(1)
            .expect("retained account exclusion");
    }
}

#[cfg(test)]
thread_local! {
    static AFTER_NODE_RETIRE: std::cell::RefCell<Option<Box<dyn FnOnce(&WorkingMemoryPool)>>> = const { std::cell::RefCell::new(None) };
}
#[cfg(test)]
mod tests;
