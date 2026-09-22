//! Publication of the original numeric account before any owned diagnostics.
//! Only the closed D request compiler may supply a complete measured recipe.
use super::*;

#[derive(Debug, Clone)]
pub(in crate::working_memory) struct PendingOriginal {
    id: u64,
    execution: usize,
    capacity: Option<eredu_core::MemoryLimits>,
    bytes: u64,
    floor: u64,
}
impl PendingOriginal {
    #[cfg(test)]
    pub(in crate::working_memory) fn id(&self) -> u64 {
        self.id
    }
    pub(in crate::working_memory) fn capacity(
        &self,
        domain: MemoryDomainId,
    ) -> Result<MemoryLimit, WorkingMemoryError> {
        Ok(self
            .capacity
            .as_ref()
            .map(|c| c.get(domain))
            .transpose()?
            .unwrap_or(MemoryLimit::Unlimited))
    }
    pub(in crate::working_memory) fn identity(&self, id: u64) -> Option<usize> {
        (id == self.id).then_some(self.execution)
    }
}

// Stack construction owner, created only after the existing atomic commit.
// It cannot escape as a retryable admission or be cloned from a source alias.
pub(in crate::working_memory) struct PendingAccount {
    pool: MemoryLedger,
    execution: InferenceExecutionIdentity,
    id: u64,
    bytes: u64,
    active: bool,
    requirements: Option<eredu_core::DomainMemoryRequirements>,
    charges: Option<Vec<DomainMemoryCharge>>,
}
impl PendingAccount {
    /// Retains the caller's complete limits without allocating another domain
    /// vector. Comparison borrows those limits until the same commit publishes
    /// them in the pending slot; no observer can see an unconstrained account.
    pub(in crate::working_memory) fn accept_planning(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &mut Usage,
        bytes: u64,
        capacity: Option<MemoryLimits>,
    ) -> Result<Self, WorkingMemoryError> {
        let commit =
            PreparedAccountCommit::prepare(pool, execution, usage, bytes, capacity.as_ref(), &[])?;
        let pending = Self::accept(pool, execution, usage, commit, bytes, None, bytes)?;
        usage
            .pending_original
            .as_mut()
            .expect("accepted planning account")
            .capacity = capacity;
        Ok(pending)
    }

    pub(in crate::working_memory) fn accept(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &mut Usage,
        commit: PreparedAccountCommit<'_>,
        bytes: u64,
        capacity: Option<eredu_core::MemoryLimits>,
        floor: u64,
    ) -> Result<Self, WorkingMemoryError> {
        if floor > bytes {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if usage.pending_original.is_some() {
            return Err(WorkingMemoryError::AccountConstructionBusy);
        }
        let id = usage.next_funding;
        let next = id.checked_add(1).ok_or(WorkingMemoryError::Overflow)?;
        commit.commit(pool, usage);
        usage.next_funding = next;
        usage.pending_original = Some(PendingOriginal {
            id,
            execution: Arc::as_ptr(&execution.0) as usize,
            capacity,
            bytes,
            floor,
        });
        Ok(Self {
            pool: pool.clone(),
            execution: execution.clone(),
            id,
            bytes,
            active: true,
            requirements: None,
            charges: None,
        })
    }

    pub(in crate::working_memory) fn after_projected_commit(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &mut Usage,
        charges: Vec<DomainMemoryCharge>,
        capacity: MemoryLimits,
        floor: u64,
    ) -> Self {
        let id = usage.next_funding;
        usage.next_funding += 1;
        let bytes = charges[usage.host_slot]
            .total()
            .expect("validated host charge");
        usage.pending_original = Some(PendingOriginal {
            id,
            execution: Arc::as_ptr(&execution.0) as usize,
            capacity: Some(capacity),
            bytes,
            floor,
        });
        Self {
            pool: pool.clone(),
            execution: execution.clone(),
            id,
            bytes,
            active: true,
            requirements: None,
            charges: Some(charges),
        }
    }

    pub(in crate::working_memory) fn accept_domains(
        pool: &MemoryLedger,
        execution: &InferenceExecutionIdentity,
        usage: &mut Usage,
        commit: PreparedAccountCommit<'_>,
        requirements: eredu_core::DomainMemoryRequirements,
        capacity: Option<eredu_core::MemoryLimits>,
        floor: u64,
    ) -> Result<Self, WorkingMemoryError> {
        let host = requirements.get(pool.topology().host_domain())?.total()?;
        if floor > host {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let mut pending = Self::accept(pool, execution, usage, commit, host, capacity, floor)?;
        pending.requirements = Some(requirements);
        Ok(pending)
    }

    // Publication always yields its metadata ticket, including sticky poison.
    // The caller inspects status while retaining that ticket on owning failure.
    // No arbitrary-lived failure keeps the one pending construction slot Busy.
    pub(in crate::working_memory) fn publish(mut self) -> AccountTicket {
        let node = AccountNode::empty();
        let mut state = match (self.requirements.as_ref(), self.charges.as_ref()) {
            (Some(requirements), _) => FundingState::new(
                self.pool.topology(),
                requirements,
                None,
                &self.execution,
                true,
                0,
                0,
            ),
            (None, Some(charges)) => FundingState::from_charges(
                self.pool.topology(),
                charges.iter().copied(),
                None,
                &self.execution,
                true,
                0,
                0,
            ),
            (None, None) => FundingState::host(
                self.pool.topology(),
                self.bytes,
                None,
                &self.execution,
                true,
                0,
                0,
            ),
        }
        .expect("accepted account geometry");
        let (mut usage, poisoned) = match self.pool.0.usage.lock() {
            Ok(usage) => (usage, false),
            Err(poison) => (poison.into_inner(), true),
        };
        let pending = usage
            .pending_original
            .take()
            .expect("original pending account");
        assert_eq!(pending.id, self.id);
        state.capacity = pending.capacity;
        state.control_floor = pending.floor;
        state.host_held = pending.floor;
        state.quarantined = poisoned;
        usage.funding.publish(node, self.id, state, false);
        self.active = false;
        drop(usage);
        self.pool.0.construction_ready.notify_all();
        AccountTicket {
            pool: self.pool.clone(),
            id: self.id,
            status: if poisoned {
                Err(WorkingMemoryError::Poisoned)
            } else {
                Ok(())
            },
            active: true,
        }
    }
}
impl Drop for PendingAccount {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Only synchronous construction unwind reaches this branch. No target
        // node/diagnostic allocation has escaped; source arguments outlive it.
        let Ok(mut usage) = self.pool.0.usage.lock() else {
            self.pool.0.construction_ready.notify_all();
            return;
        };
        let pending = usage
            .pending_original
            .take()
            .expect("owned original pending account");
        assert_eq!(pending.id, self.id);
        if let Some(charges) = &self.charges {
            for (domain, charge) in usage.domains.iter_mut().zip(charges) {
                domain.reserved = domain
                    .reserved
                    .checked_sub(charge.total().expect("validated charge"))
                    .expect("pending charge");
                domain.placement_allowances -= charge.placement_allowance_bytes;
                domain.estimates -= charge.estimated_overhead_bytes;
                domain.headroom -= charge.headroom_bytes;
            }
        } else if let Some(requirements) = &self.requirements {
            for (slot, (_, charge)) in requirements.iter().enumerate() {
                let domain = &mut usage.domains[slot];
                domain.reserved = domain
                    .reserved
                    .checked_sub(charge.total().expect("validated charge"))
                    .expect("pending charge");
                domain.placement_allowances -= charge.placement_allowance_bytes;
                domain.estimates -= charge.estimated_overhead_bytes;
                domain.headroom -= charge.headroom_bytes;
            }
        } else {
            usage.reserved = usage
                .reserved
                .checked_sub(pending.bytes)
                .expect("pending charge");
        }
        usage.reservations = usage
            .reservations
            .checked_sub(1)
            .expect("pending exclusion");
        drop(usage);
        self.pool.0.construction_ready.notify_all();
        drop(pending);
        // Issuance and accepted predecessor raises never rewind.
    }
}

// One stack ticket spans provisional diagnostic storage and the final Arc
// allocation. Conversion to Reservation transfers this exact original account;
// an error must own the ticket until all of its partial payloads retire.
#[derive(Debug)]
pub(in crate::working_memory) struct AccountTicket {
    pool: MemoryLedger,
    id: u64,
    status: Result<(), WorkingMemoryError>,
    active: bool,
}
impl AccountTicket {
    /// Transfers an accepted construction account to the ordinary copy lifecycle.
    /// The caller constructs all allocating descriptors while this ticket still
    /// owns their grant. Counter and floor geometry are checked before accepting
    /// the pending account; this publication performs no allocation or callback.
    pub(in crate::working_memory) fn into_copy(
        mut self,
        controls: super::super::copy_controls::CopyAccountState,
        execution: &InferenceExecutionIdentity,
    ) -> Result<u64, WorkingMemoryError> {
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.validate_in(&usage)?;
        usage
            .funding
            .start(self.id)
            .expect("validated construction account");
        let state = usage
            .funding
            .get_mut(&self.id)
            .expect("started construction account");
        state.execution = Arc::downgrade(&execution.0);
        state.control_floor = controls.control_floor;
        state.report_control_bytes = controls.control_floor;
        state.host_held = controls.host_held;
        state.scopes = controls.scopes;
        state.native_scopes = controls.native_scopes;
        state.run_open = controls.run_open;
        state.metadata_live = false;
        self.active = false;
        Ok(self.id)
    }
    pub(in crate::working_memory) fn retire_completed_native_occupancy(
        &self,
        capacity: u64,
        occupied: u64,
        placement: &eredu_core::MemoryPlacement,
        released: &std::sync::atomic::AtomicU64,
    ) {
        use std::sync::atomic::Ordering;
        let mut usage = super::super::lock_for_retirement(&self.pool);
        let before = released.load(Ordering::Relaxed);
        let validation = (|| -> Result<Option<(u64, u64)>, WorkingMemoryError> {
            self.validate_in(&usage)?;
            if std::thread::panicking() {
                return Err(WorkingMemoryError::Poisoned);
            }
            placement.validate(self.pool.topology())?;
            let total = capacity
                .checked_sub(occupied)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if total <= before {
                return Ok(None);
            }
            let decrement = total
                .checked_sub(before)
                .ok_or(WorkingMemoryError::Poisoned)?;
            let state = &usage
                .funding
                .nodes()
                .find(|node| node.id == self.id)
                .ok_or(WorkingMemoryError::IdentityMismatch)?
                .state;
            for (slot, (domain, _)) in self.pool.topology().domains().enumerate() {
                if !placement.domains().contains(&domain) {
                    continue;
                }
                let balance = &state.domains[slot];
                let remaining = balance
                    .remaining
                    .checked_sub(decrement)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                if slot == state.host_slot && remaining < state.control_floor {
                    return Err(WorkingMemoryError::Poisoned);
                }
                usage.domains[slot]
                    .reserved
                    .checked_sub(decrement)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                match placement.kind() {
                    eredu_core::MemoryPlacementKind::Possible { .. } => {
                        balance
                            .remaining_charge
                            .placement_allowance_bytes
                            .checked_sub(decrement)
                            .ok_or(WorkingMemoryError::Poisoned)?;
                        usage.domains[slot]
                            .placement_allowances
                            .checked_sub(decrement)
                            .ok_or(WorkingMemoryError::Poisoned)?;
                    }
                    eredu_core::MemoryPlacementKind::Fixed(_) => {
                        balance
                            .remaining_charge
                            .accounted_bytes
                            .checked_sub(decrement)
                            .ok_or(WorkingMemoryError::Poisoned)?;
                    }
                }
            }
            Ok(Some((total, decrement)))
        })();
        let (total, decrement) = match validation {
            Ok(Some(value)) => value,
            Ok(None) => return,
            Err(_) => {
                usage.funding.quarantine_original(self.id);
                return;
            }
        };
        let Usage {
            funding, domains, ..
        } = &mut *usage;
        let mut next = funding.head.as_deref_mut();
        let state = loop {
            let node = next.expect("validated original account");
            if node.id == self.id {
                break &mut node.state;
            }
            next = node.next.as_deref_mut();
        };
        for (slot, (domain, _)) in self.pool.topology().domains().enumerate() {
            if !placement.domains().contains(&domain) {
                continue;
            }
            let balance = &mut state.domains[slot];
            balance.remaining -= decrement;
            domains[slot].reserved -= decrement;
            match placement.kind() {
                eredu_core::MemoryPlacementKind::Possible { .. } => {
                    balance.remaining_charge.placement_allowance_bytes -= decrement;
                    domains[slot].placement_allowances -= decrement;
                }
                eredu_core::MemoryPlacementKind::Fixed(_) => {
                    balance.remaining_charge.accounted_bytes -= decrement
                }
            }
        }
        released.store(total, Ordering::Release);
    }
    pub(in crate::working_memory) fn quarantine(&self) {
        let mut usage = super::super::lock_for_retirement(&self.pool);
        usage.funding.quarantine_original(self.id);
    }
    pub(in crate::working_memory) fn status(&self) -> Result<(), WorkingMemoryError> {
        self.status.clone()?;
        let usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        usage.funding.validate_constructing(self.id)
    }
    pub(in crate::working_memory) fn validate_in(
        &self,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.status.clone()?;
        usage.funding.validate_constructing(self.id)
    }
    pub(in crate::working_memory) fn pool(&self) -> &MemoryLedger {
        &self.pool
    }
    pub(in crate::working_memory) fn id(&self) -> u64 {
        self.id
    }
    pub(in crate::working_memory) fn into_reservation(
        mut self,
        mut value: crate::working_memory::Reservation,
    ) -> crate::working_memory::WorkingMemoryReservation {
        assert_eq!(value.account_id, self.id);
        assert!(value.pool.same_ledger(&self.pool));
        value.funding = None;
        // The ticket still owns the account while Arc allocation may unwind.
        let owner = crate::working_memory::ReservationOwner::new(value);
        self.active = false;
        crate::working_memory::WorkingMemoryReservation(owner)
    }
}
impl Drop for AccountTicket {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut usage = lock(&self.pool);
        usage.funding.retire_unfunded(self.id);
        retire_metadata(&mut usage, self.id);
    }
}
