//! Assigned allocation allowance transported with one actual native scope.
use super::*;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug)]
struct AllocationFunding {
    pool: MemoryLedger,
    account: u64,
    active: AtomicBool,
    // Last, and released after the shared allocation itself is deallocated.
    _host: eredu_core::HostPreparationAuthority,
}

/// Accounting-only access to an existing native scope's assigned allowance.
/// Clones share its exact lifetime marker and paid host controls. This handle
/// grants no submission or execution authority and cannot create another scope.
/// Certification, abandonment or quarantine prevents subsequent allocation.
#[derive(Debug)]
pub struct WorkingMemoryAllocationFunding(Option<Arc<AllocationFunding>>);

impl Clone for WorkingMemoryAllocationFunding {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(
            self.0.as_ref().expect("live allocation funding"),
        )))
    }
}
impl Drop for WorkingMemoryAllocationFunding {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl WorkingMemoryAllocationFunding {
    fn inner(&self) -> &AllocationFunding {
        self.0.as_deref().expect("live allocation funding")
    }
    /// The ledger that already reserved this allocation allowance.
    pub fn pool(&self) -> &MemoryLedger {
        &self.inner().pool
    }
    pub(in crate::working_memory) fn account(&self) -> u64 {
        self.inner().account
    }
    pub(in crate::working_memory) fn validate_locked(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        if !pool.same_ledger(self.pool()) {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        if !self.inner().active.load(Ordering::Acquire) {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        let state = usage
            .funding
            .get(&self.account())
            .ok_or(WorkingMemoryError::ExecutionFenced)?;
        if state.quarantined || state.native_scopes == 0 || !state.retains_workspace() {
            return Err(WorkingMemoryError::ExecutionFenced);
        }
        state.validate_span_spend(None)
    }
    pub(super) fn close_locked(&self) {
        self.inner().active.store(false, Ordering::Release);
    }
    /// Funds allocation-publication metadata from the same assigned host
    /// allowance. The actual allocation transaction rechecks scope liveness.
    pub fn prepare_storage_metadata(
        &self,
    ) -> Result<super::super::StorageMetadataFunding, eredu_core::HostMetadataFundingError> {
        {
            let usage = self
                .pool()
                .0
                .usage
                .lock()
                .map_err(|_| eredu_core::HostMetadataFundingError::Poisoned)?;
            self.validate_locked(self.pool(), &usage)
                .map_err(super::super::storage_metadata::failure)?;
        }
        super::super::StorageMetadataFunding::from_account(self.pool(), self.account())
    }
}

impl WorkingMemoryFundingScope {
    /// Complete host quotation for one shared allocation-funding marker.
    /// Reusing or cloning that marker requires no additional allocation.
    pub fn allocation_funding_control_bytes() -> Result<u64, WorkingMemoryError> {
        let bytes = allocation_owner_bytes()?;
        let host = super::super::StorageMetadataFunding::host_owner_bytes(bytes)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        MemoryLedger::storage_metadata_control_bytes()?
            .checked_add(u64::try_from(host).map_err(|_| WorkingMemoryError::Overflow)?)
            .ok_or(WorkingMemoryError::Overflow)
    }
    /// Retains a thread-safe accounting handle for this exact native scope.
    /// Its host owner is funded before constructing the shared marker.
    pub fn allocation_funding(
        &mut self,
    ) -> Result<WorkingMemoryAllocationFunding, WorkingMemoryError> {
        self.validate_native_purpose()?;
        if let Some(funding) = &self.allocation_funding {
            let usage = self
                .pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            funding.validate_locked(&self.pool, &usage)?;
            return Ok(funding.clone());
        }
        let metadata = self
            .prepare_storage_metadata()
            .map_err(super::super::reservation_metadata::funding_error)?;
        let host = metadata
            .prepare_host_owner(allocation_owner_bytes()?)
            .map_err(super::super::reservation_metadata::funding_error)?;
        let funding = WorkingMemoryAllocationFunding(Some(Arc::new(AllocationFunding {
            pool: self.pool.clone(),
            account: self.id,
            active: AtomicBool::new(true),
            _host: host,
        })));
        {
            let usage = self
                .pool
                .0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?;
            funding.validate_locked(&self.pool, &usage)?;
        }
        self.allocation_funding = Some(funding.clone());
        Ok(funding)
    }
}

fn allocation_owner_bytes() -> Result<usize, WorkingMemoryError> {
    use std::mem::size_of;
    let frames = [
        size_of::<AllocationFunding>(),
        size_of::<WorkingMemoryAllocationFunding>(),
        size_of::<Result<WorkingMemoryAllocationFunding, WorkingMemoryError>>(),
    ];
    frames.into_iter().try_fold(
        usize::try_from(super::super::qualified_storage::shared_bytes::<
            AllocationFunding,
        >()?)
        .ok()
        .and_then(|n| n.checked_add(std::mem::size_of_val(&frames)))
        .ok_or(WorkingMemoryError::Overflow)?,
        |sum, value| sum.checked_add(value).ok_or(WorkingMemoryError::Overflow),
    )
}
