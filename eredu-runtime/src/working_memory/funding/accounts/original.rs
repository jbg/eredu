//! Publication of the original numeric account before any owned diagnostics.
//! Only the closed D request compiler may supply a complete measured recipe.
use super::*;

#[derive(Debug, Clone, Copy)]
pub(in crate::working_memory) struct PendingOriginal {
    id: u64,
    execution: usize,
    capacity: Option<u64>,
    bytes: u64,
    floor: u64,
}
impl PendingOriginal {
    #[cfg(test)]
    pub(in crate::working_memory) fn id(&self) -> u64 {
        self.id
    }
    pub(in crate::working_memory) fn capacity(&self) -> u64 {
        self.capacity.unwrap_or(u64::MAX)
    }
    pub(in crate::working_memory) fn identity(&self, id: u64) -> Option<usize> {
        (id == self.id).then_some(self.execution)
    }
}

// Stack construction owner, created only after the existing atomic commit.
// It cannot escape as a retryable admission or be cloned from a source alias.
pub(in crate::working_memory) struct PendingAccount {
    pool: WorkingMemoryPool,
    execution: InferenceExecutionIdentity,
    id: u64,
    active: bool,
}
impl PendingAccount {
    pub(in crate::working_memory) fn accept(
        pool: &WorkingMemoryPool,
        execution: &InferenceExecutionIdentity,
        usage: &mut Usage,
        commit: PreparedAccountCommit<'_>,
        bytes: u64,
        capacity: Option<u64>,
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
        commit.commit(usage);
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
            active: true,
        })
    }

    // Publication always yields its metadata ticket, including sticky poison.
    // The caller inspects status while retaining that ticket on owning failure.
    // No arbitrary-lived failure keeps the one pending construction slot Busy.
    pub(in crate::working_memory) fn publish(mut self) -> AccountTicket {
        let node = AccountNode::empty();
        let (mut usage, poisoned) = match self.pool.0.usage.lock() {
            Ok(usage) => (usage, false),
            Err(poison) => (poison.into_inner(), true),
        };
        let pending = usage
            .pending_original
            .take()
            .expect("original pending account");
        assert_eq!(pending.id, self.id);
        let mut state =
            FundingState::new(pending.bytes, pending.capacity, &self.execution, true, 0, 0);
        state.control_floor = pending.floor;
        state.host_held = pending.floor;
        state.quarantined = poisoned;
        usage.funding.publish(node, self.id, state, false);
        self.active = false;
        drop(usage);
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
            return;
        };
        let pending = usage
            .pending_original
            .take()
            .expect("owned original pending account");
        assert_eq!(pending.id, self.id);
        usage.reserved = usage
            .reserved
            .checked_sub(pending.bytes)
            .expect("pending charge");
        usage.reservations = usage
            .reservations
            .checked_sub(1)
            .expect("pending exclusion");
        // Issuance and accepted predecessor raises never rewind.
    }
}

// One stack ticket spans provisional diagnostic storage and the final Arc
// allocation. Conversion to Reservation transfers this exact original account;
// an error must own the ticket until all of its partial payloads retire.
#[derive(Debug)]
pub(in crate::working_memory) struct AccountTicket {
    pool: WorkingMemoryPool,
    id: u64,
    status: Result<(), WorkingMemoryError>,
    active: bool,
}
impl AccountTicket {
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
    pub(in crate::working_memory) fn validate_in(&self, usage: &Usage) -> Result<(), WorkingMemoryError> {
        self.status.clone()?;
        usage.funding.validate_constructing(self.id)
    }
    pub(in crate::working_memory) fn pool(&self) -> &WorkingMemoryPool {
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
        assert!(value.pool.same_domain(&self.pool));
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
