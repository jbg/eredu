//! Private protected partition from the exact accepted selected span.
//! Native authentication remains a truthful backend mechanism contract; neither
//! a public control guard nor a bare byte count can issue partition coverage.

use super::*;

fn remaining_bound(total: u64, registered: u64) -> u64 {
    if total > registered {
        total - registered
    } else {
        0
    }
}

/// Exact opt-in Work association. The Arc allocation retires before its raw
/// metadata custody; neither field can reach a Work/native owner or full bank.
#[derive(Clone, Debug)]
pub(in crate::working_memory) struct NativePublicationScopeIdentity {
    identity: Arc<()>,
    _raw: super::RawSpanHostOwner,
}
impl NativePublicationScopeIdentity {
    pub(in crate::working_memory) fn new(raw: super::RawSpanHostOwner) -> Self {
        Self {
            identity: Arc::new(()),
            _raw: raw,
        }
    }
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }
}
pub(in crate::working_memory) fn requested_control_bytes() -> Result<u64, WorkingMemoryError> {
    [
        std::mem::size_of::<ReservedNativePartition>(),
        std::mem::size_of::<PartitionOwner>(),
        std::mem::size_of::<NativePartition>(),
        std::mem::size_of::<NativePublicationScopeIdentity>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|n| u64::try_from(n).ok())
    .ok_or(WorkingMemoryError::Overflow)
}

pub(in crate::working_memory) fn qualified_header_bytes(
    maximum_publications: usize,
) -> Result<u64, WorkingMemoryError> {
    use crate::working_memory::qualified_storage as storage;
    let per_scope = storage::shared_bytes::<()>()?
        // The scope's own wrapper survives independently of the publication's
        // wrapper. At most one fresh shell per finite claim can be constructed.
        .checked_add(std::mem::size_of::<NativePublicationScopeIdentity>() as u64)
        .and_then(|n| n.checked_mul(u64::try_from(maximum_publications).ok()?))
        .ok_or(WorkingMemoryError::Overflow)?;
    storage::shared_header_bytes::<PartitionOwner>()?
        .checked_add(per_scope)
        .ok_or(WorkingMemoryError::Overflow)
}

pub(in crate::working_memory) struct ReservedNativePartition {
    pool: MemoryLedger,
    account: u64,
    execution: Weak<()>,
    bytes: u64,
    placement: Arc<eredu_core::MemoryPlacement>,
    raw: Option<super::RawSpanHostOwner>,
    // Transient full source validation only; never moved into PartitionOwner.
    validation: Option<(
        crate::working_memory::OriginalTextControlGuard,
        WorkingMemoryReservation,
    )>,
}

#[derive(Debug)]
struct PartitionOwner {
    pool: MemoryLedger,
    account: u64,
    bytes: u64,
    current_bytes: std::sync::atomic::AtomicU64,
    placement: Arc<eredu_core::MemoryPlacement>,
    active: bool,
    // Last: accounted metadata survives native births and stale canonical rows.
    // This owner contains no controls/source table/native pair/Array backedge.
    _raw: Option<super::RawSpanHostOwner>,
}

/// Accounting-only custody. It cannot retain an Array, native birth or Scope.
/// No Weak or raw Arc escapes; the last Arc shell retires before its pool hold.
#[derive(Debug)]
pub(in crate::working_memory) struct NativePartition(Option<Arc<PartitionOwner>>);

impl Clone for NativePartition {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live partition"))))
    }
}

impl Drop for NativePartition {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            if owner.active && std::thread::panicking() {
                // Every active custody retires outside Usage: prepared inputs
                // and rows precede the caller's partition; detached canonical
                // entries/batches leave the lock before provider destruction.
                // Mark the origin on ANY unwinding alias, not just the last:
                // another native/registry alias can survive catch_unwind.
                let mut usage = lock_for_retirement(&owner.pool);
                usage
                    .funding
                    .get_mut(&owner.account)
                    .expect("live partition custody")
                    .quarantined = true;
            }
            drop(Arc::into_inner(owner));
        }
    }
}

impl Drop for PartitionOwner {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let mut usage = lock_for_retirement(&self.pool);
        let state = usage
            .funding
            .get_mut(&self.account)
            .expect("live partition account");
        if std::thread::panicking() {
            // Match ordinary publication: a provider destructor unwind is not
            // evidence that its complete retained payload population retired.
            state.quarantined = true;
        }
        if !state.native_issued
            || self
                .pool
                .topology()
                .domains()
                .enumerate()
                .any(|(slot, (domain, _))| {
                    state.domains[slot].native_held
                        != Some(if self.placement.domains().contains(&domain) {
                            remaining_bound(
                                self.current_bytes
                                    .load(std::sync::atomic::Ordering::Relaxed),
                                state.domains[slot].native_registered,
                            )
                        } else {
                            0
                        })
                })
        {
            // Preserve the account rather than inventing credit after corruption.
            state.quarantined = true;
            return;
        }
        if state
            .domains
            .iter()
            .any(|balance| balance.native_registered != 0)
        {
            state.quarantined = true;
            return;
        }
        for balance in &mut state.domains {
            balance.native_held = None;
        }
        settle(&mut usage, self.account);
    }
}

impl NativePartition {
    pub(in crate::working_memory) fn pool(&self) -> &MemoryLedger {
        &self.owner().pool
    }
    fn owner(&self) -> &PartitionOwner {
        self.0.as_deref().expect("live partition")
    }

    // Only the metadata hold, never the native/P partition itself, may own an
    // outer canonical namespace. Production issuance always supplies this raw
    // original Host-purpose custody; private legacy ledger fixtures may not.
    pub(in crate::working_memory) fn namespace_metadata(&self) -> Option<super::RawSpanHostOwner> {
        self.owner()._raw.clone()
    }

    pub(in crate::working_memory) fn placement(&self) -> &Arc<eredu_core::MemoryPlacement> {
        &self.owner().placement
    }

    pub(in crate::working_memory) fn capacity(&self) -> u64 {
        self.owner().bytes
    }

    /// A native producer-closure observation certifies that the remaining
    /// occupancy can only decrease. Stale concurrent observations are harmless.
    pub(in crate::working_memory) fn retire_completed_occupancy(&self, bytes: u64) {
        let owner = self.owner();
        let mut usage = lock_for_retirement(&owner.pool);
        let current = owner
            .current_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        if bytes >= current {
            return;
        }
        let valid = (|| -> Result<(), WorkingMemoryError> {
            self.validate_origin(&usage)?;
            let state = usage
                .funding
                .get(&owner.account)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if state.quarantined || std::thread::panicking() {
                return Err(WorkingMemoryError::Poisoned);
            }
            for (slot, (domain, _)) in owner.pool.topology().domains().enumerate() {
                if !owner.placement.domains().contains(&domain) {
                    continue;
                }
                let balance = &state.domains[slot];
                let release = remaining_bound(current, balance.native_registered)
                    - remaining_bound(bytes, balance.native_registered);
                balance
                    .remaining
                    .checked_sub(release)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                usage.domains[slot]
                    .reserved
                    .checked_sub(release)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                let allowance = match owner.placement.kind() {
                    eredu_core::MemoryPlacementKind::Possible { .. } => release,
                    eredu_core::MemoryPlacementKind::Fixed(_) => {
                        balance.native_held_allowance.min(release)
                    }
                };
                balance
                    .remaining_charge
                    .placement_allowance_bytes
                    .checked_sub(allowance)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                usage.domains[slot]
                    .placement_allowances
                    .checked_sub(allowance)
                    .ok_or(WorkingMemoryError::Poisoned)?;
            }
            Ok(())
        })();
        if valid.is_err() {
            if let Some(state) = usage.funding.get_mut(&owner.account) {
                state.quarantined = true;
            }
            return;
        }
        for (slot, (domain, _)) in owner.pool.topology().domains().enumerate() {
            if !owner.placement.domains().contains(&domain) {
                continue;
            }
            let state = usage
                .funding
                .get_mut(&owner.account)
                .expect("validated partition");
            let balance = &mut state.domains[slot];
            let release = remaining_bound(current, balance.native_registered)
                - remaining_bound(bytes, balance.native_registered);
            let allowance = match owner.placement.kind() {
                eredu_core::MemoryPlacementKind::Possible { .. } => release,
                eredu_core::MemoryPlacementKind::Fixed(_) => {
                    balance.native_held_allowance.min(release)
                }
            };
            balance.remaining -= release;
            balance.remaining_charge.placement_allowance_bytes -= allowance;
            balance.native_held_allowance -= allowance;
            balance.native_held = Some(remaining_bound(bytes, balance.native_registered));
            usage.domains[slot].reserved -= release;
            usage.domains[slot].placement_allowances -= allowance;
        }
        owner
            .current_bytes
            .store(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Retire one canonical backing after its keys/native sidecars leave the
    /// coordinator. Live producers regain their protected allocation allowance;
    /// a closed producer population retains only certified remaining occupancy.
    pub(in crate::working_memory) fn retire_registered(
        &self,
        usage: &mut Usage,
        bytes: u64,
        placement: &eredu_core::MemoryPlacement,
        converted_allowance: u64,
    ) {
        let owner = self.owner();
        let current = owner
            .current_bytes
            .load(std::sync::atomic::Ordering::Relaxed);
        let possible = matches!(
            placement.kind(),
            eredu_core::MemoryPlacementKind::Possible { .. }
        );
        let validate = (|| -> Result<(), WorkingMemoryError> {
            self.validate_origin(usage)?;
            let state = usage
                .funding
                .get(&owner.account)
                .ok_or(WorkingMemoryError::IdentityMismatch)?;
            if state.quarantined || std::thread::panicking() {
                return Err(WorkingMemoryError::Poisoned);
            }
            for domain in placement.domains() {
                let slot = owner.pool.topology().slot(*domain)?;
                let balance = &state.domains[slot];
                let registered = balance
                    .native_registered
                    .checked_sub(bytes)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                let refill = remaining_bound(current, registered)
                    .checked_sub(balance.native_held.ok_or(WorkingMemoryError::Poisoned)?)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                let restored = if possible {
                    refill
                } else {
                    converted_allowance.min(refill)
                };
                balance
                    .remaining
                    .checked_add(refill)
                    .ok_or(WorkingMemoryError::Overflow)?;
                balance
                    .remaining_charge
                    .placement_allowance_bytes
                    .checked_add(restored)
                    .ok_or(WorkingMemoryError::Overflow)?;
                usage.domains[slot]
                    .registered
                    .checked_sub(bytes)
                    .ok_or(WorkingMemoryError::Poisoned)?;
                usage.domains[slot]
                    .reserved
                    .checked_add(refill)
                    .ok_or(WorkingMemoryError::Overflow)?;
                usage.domains[slot]
                    .placement_allowances
                    .checked_sub(if possible { bytes } else { 0 })
                    .and_then(|amount| amount.checked_add(restored))
                    .ok_or(WorkingMemoryError::Poisoned)?;
            }
            Ok(())
        })();
        if validate.is_err() {
            if let Some(state) = usage.funding.get_mut(&owner.account) {
                state.quarantined = true;
            }
            return;
        }
        for domain in placement.domains() {
            let slot = owner
                .pool
                .topology()
                .slot(*domain)
                .expect("validated placement");
            let balance = &mut usage
                .funding
                .get_mut(&owner.account)
                .expect("validated origin")
                .domains[slot];
            balance.native_registered -= bytes;
            let held = remaining_bound(current, balance.native_registered);
            let refill = held - balance.native_held.expect("validated partition");
            let restored = if possible {
                refill
            } else {
                converted_allowance.min(refill)
            };
            balance.native_held = Some(held);
            balance.remaining += refill;
            balance.remaining_charge.placement_allowance_bytes += restored;
            balance.native_held_allowance += restored;
            usage.domains[slot].registered -= bytes;
            usage.domains[slot].reserved += refill;
            usage.domains[slot].placement_allowances -= if possible { bytes } else { 0 };
            usage.domains[slot].placement_allowances += restored;
        }
    }

    pub(in crate::working_memory) fn same_origin(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live partition"),
            other.0.as_ref().expect("live partition"),
        )
    }

    pub(in crate::working_memory) fn validate_origin(
        &self,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        let owner = self.owner();
        let state = usage
            .funding
            .get(&owner.account)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !owner.active
            || !state.native_issued
            || owner
                .pool
                .topology()
                .domains()
                .enumerate()
                .any(|(slot, (domain, _))| {
                    state.domains[slot].native_held
                        != Some(if owner.placement.domains().contains(&domain) {
                            remaining_bound(
                                owner
                                    .current_bytes
                                    .load(std::sync::atomic::Ordering::Relaxed),
                                state.domains[slot].native_registered,
                            )
                        } else {
                            0
                        })
                })
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        state.validate_registered_copy_origin()
    }

    pub(in crate::working_memory) fn validate_pool(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.owner().pool.same_ledger(pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        self.validate_origin(usage)
    }

    pub(in crate::working_memory) fn validate_publisher(
        &self,
        scope: &WorkingMemoryFundingScope,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_pool(scope.pool(), usage)?;
        if self.owner().account != scope.id {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}

impl ReservedNativePartition {
    pub(in crate::working_memory) fn from_selected_span(
        run: &WorkingMemoryFundingRun,
        span: &crate::working_memory::OwnedTextSpanWorkspace,
        capacity: u64,
        placement: Arc<eredu_core::MemoryPlacement>,
    ) -> Result<Self, WorkingMemoryError> {
        let reservation = span.reservation();
        if reservation.0.funding != Some(run.id) || !run.pool.same_ledger(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        placement.validate(run.pool.topology())?;
        let controls = span.control_guard();
        controls.validate_reservation(reservation)?;
        Ok(Self {
            pool: run.pool.clone(),
            account: run.id,
            execution: Arc::downgrade(&reservation.0.execution.0),
            bytes: capacity,
            placement,
            raw: Some(controls.custody.raw().clone()),
            validation: Some((controls, reservation.clone())),
        })
    }
}

impl WorkingMemoryFundingRun {
    // No production caller can create ReservedNativePartition in this slice.
    pub(in crate::working_memory) fn take_native_partition(
        &self,
        receipt: ReservedNativePartition,
    ) -> Result<NativePartition, WorkingMemoryError> {
        if !self.pool.same_ledger(&receipt.pool) || self.id != receipt.account {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Prepare the exact owner shell before borrowing Usage. Failure leaves
        // it inactive, so its Drop cannot acquire the retirement lock here.
        let mut partition = NativePartition(Some(Arc::new(PartitionOwner {
            pool: self.pool.clone(),
            account: self.id,
            bytes: receipt.bytes,
            current_bytes: std::sync::atomic::AtomicU64::new(receipt.bytes),
            placement: receipt.placement,
            active: false,
            _raw: receipt.raw,
        })));
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.pool.0.check_host_increment(&usage, 0)?;
        if let Some((controls, reservation)) = &receipt.validation {
            controls
                .custody
                .validate_issuance_locked(&self.pool, &usage, reservation)?;
        }
        let state = usage
            .funding
            .get_mut(&self.id)
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if !Weak::ptr_eq(&state.execution, &receipt.execution) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if !self.open || !state.run_open || !state.metadata_live || state.quarantined {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(None)?;
        if state.native_issued {
            return Err(WorkingMemoryError::PreparationAlreadyStarted);
        }
        for (slot, (domain, _)) in self.pool.topology().domains().enumerate() {
            let required = if partition.owner().placement.domains().contains(&domain) {
                partition.owner().bytes
            } else {
                0
            };
            let available = if slot == self.pool.0.host_slot {
                state.spendable_remaining()?
            } else {
                state.domains[slot]
                    .remaining
                    .checked_sub(state.domains[slot].native_held.unwrap_or(0))
                    .ok_or(WorkingMemoryError::Poisoned)?
            };
            if required > available {
                return Err(WorkingMemoryError::DomainAllowanceExceeded {
                    domain,
                    required_bytes: required,
                    available_bytes: available,
                });
            }
            match partition.owner().placement.kind() {
                eredu_core::MemoryPlacementKind::Possible { .. } => {
                    state.domains[slot]
                        .remaining_charge
                        .placement_allowance_bytes
                        .checked_sub(required)
                        .ok_or(WorkingMemoryError::IdentityMismatch)?;
                }
                eredu_core::MemoryPlacementKind::Fixed(_) => {
                    state.allocation_allowance(slot, required, 0, 0)?;
                }
            }
        }
        for (slot, (domain, _)) in self.pool.topology().domains().enumerate() {
            let required = if partition.owner().placement.domains().contains(&domain) {
                partition.owner().bytes
            } else {
                0
            };
            let allowance = match partition.owner().placement.kind() {
                eredu_core::MemoryPlacementKind::Possible { .. } => required,
                eredu_core::MemoryPlacementKind::Fixed(_) => state
                    .allocation_allowance(slot, required, 0, 0)
                    .expect("validated partition category"),
            };
            state.domains[slot].native_held = Some(required);
            state.domains[slot].native_held_allowance = allowance;
        }
        state.native_issued = true;
        Arc::get_mut(partition.0.as_mut().expect("prepared owner"))
            .expect("unpublished partition")
            .active = true;
        drop(usage);
        Ok(partition)
    }
}

// This fixture realizes a neutral ledger receipt, never a native proof or a
// Complete workspace report. Production issuance remains deliberately absent.
#[cfg(test)]
pub(in crate::working_memory) fn test_receipt(
    run: &WorkingMemoryFundingRun,
    bytes: u64,
) -> ReservedNativePartition {
    let usage = run.pool.0.usage.lock().unwrap();
    let placement = run.pool.host_placement_handle();
    ReservedNativePartition {
        pool: run.pool.clone(),
        account: run.id,
        execution: usage.funding.get(&run.id).unwrap().execution.clone(),
        bytes,
        placement,
        raw: None,
        validation: None,
    }
}

#[cfg(test)]
mod tests;
