//! Private protected partition from the exact accepted selected span.
//! Native authentication remains a truthful backend mechanism contract; neither
//! a public control guard nor a bare byte count can issue partition coverage.

use super::*;

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
    pool: WorkingMemoryPool,
    account: u64,
    execution: Weak<()>,
    bytes: u64,
    raw: Option<super::RawSpanHostOwner>,
    // Transient full source validation only; never moved into PartitionOwner.
    validation: Option<(
        crate::working_memory::OriginalTextControlGuard,
        WorkingMemoryReservation,
    )>,
}

#[derive(Debug)]
struct PartitionOwner {
    pool: WorkingMemoryPool,
    account: u64,
    bytes: u64,
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
        if state.native_held != Some(self.bytes) || !state.native_issued {
            // Preserve the account rather than inventing credit after corruption.
            state.quarantined = true;
            return;
        }
        state.native_held = None;
        settle(&mut usage, self.account);
    }
}

impl NativePartition {
    fn owner(&self) -> &PartitionOwner {
        self.0.as_deref().expect("live partition")
    }

    // Only the metadata hold, never the native/P partition itself, may own an
    // outer canonical namespace. Production issuance always supplies this raw
    // original Host-purpose custody; private legacy ledger fixtures may not.
    pub(in crate::working_memory) fn namespace_metadata(&self) -> Option<super::RawSpanHostOwner> {
        self.owner()._raw.clone()
    }

    pub(in crate::working_memory) fn capacity(&self) -> u64 {
        self.owner().bytes
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
        if !owner.active || !state.native_issued || state.native_held != Some(owner.bytes) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        state.validate_registered_copy_origin()
    }

    pub(in crate::working_memory) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !self.owner().pool.same_domain(pool) {
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
    ) -> Result<Self, WorkingMemoryError> {
        let reservation = span.reservation();
        if reservation.0.funding != Some(run.id) || !run.pool.same_domain(&reservation.0.pool) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let controls = span.control_guard();
        controls.validate_reservation(reservation)?;
        Ok(Self {
            pool: run.pool.clone(),
            account: run.id,
            execution: Arc::downgrade(&reservation.0.execution.0),
            bytes: capacity,
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
        if !self.pool.same_domain(&receipt.pool) || self.id != receipt.account {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        // Prepare the exact owner shell before borrowing Usage. Failure leaves
        // it inactive, so its Drop cannot acquire the retirement lock here.
        let mut partition = NativePartition(Some(Arc::new(PartitionOwner {
            pool: self.pool.clone(),
            account: self.id,
            bytes: receipt.bytes,
            active: false,
            _raw: receipt.raw,
        })));
        let mut usage = self
            .pool
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        self.pool.0.available(&usage, None)?;
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
        let available = state.spendable_remaining()?;
        if receipt.bytes > available {
            return Err(WorkingMemoryError::BudgetExceeded {
                required_bytes: receipt.bytes,
                available_bytes: available,
            });
        }
        state.native_held = Some(receipt.bytes);
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
    ReservedNativePartition {
        pool: run.pool.clone(),
        account: run.id,
        execution: usage.funding.get(&run.id).unwrap().execution.clone(),
        bytes,
        raw: None,
        validation: None,
    }
}

#[cfg(test)]
mod tests;
