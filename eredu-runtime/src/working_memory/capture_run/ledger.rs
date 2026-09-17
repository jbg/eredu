//! Cumulative same-run source only; no capture claims, native roots or grant.
use super::*;
use std::{
    alloc::Layout,
    sync::{Arc, Mutex, TryLockError, atomic::AtomicUsize},
};

#[derive(Debug)]
struct State {
    usage: CaptureUsage,
    active: bool,
}
#[derive(Debug)]
enum Custody {
    Capture { _custody: CaptureTensorCustody },
    Request { _ticket: crate::working_memory::funding::AccountTicket },
}
#[derive(Debug)]
struct Owner {
    state: Mutex<State>,
    // Only the account which paid this fixed owner, never a source run/bank.
    _custody: Custody,
}
/// Closed original cumulative ledger source. Weak or raw Arc aliases cannot
/// escape. The final Arc allocation retires before its paying host custody.
#[derive(Debug)]
pub(crate) struct CaptureRunLedger(Option<Arc<Owner>>);
impl Clone for CaptureRunLedger {
    fn clone(&self) -> Self {
        Self(Some(self.0.as_ref().expect("live ledger").clone()))
    }
}
impl Drop for CaptureRunLedger {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl CaptureRunLedger {
    pub(in crate::working_memory) fn new_numerical(
        custody: crate::working_memory::OriginalSpeculativeNumericalBudgetCustody,
    ) -> Self {
        Self::new(
            CaptureTensorCustody::Speculative(custody),
            CaptureUsage::default(),
        )
    }
    pub(in crate::working_memory) fn new_model(
        custody: crate::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Self {
        Self::new(
            CaptureTensorCustody::Model(custody),
            CaptureUsage::default(),
        )
    }
    pub(super) fn new(custody: CaptureTensorCustody, usage: CaptureUsage) -> Self {
        Self::with_custody(Custody::Capture { _custody: custody }, usage)
    }
    pub(in crate::working_memory) fn new_request(
        ticket: crate::working_memory::funding::AccountTicket,
    ) -> Self {
        Self::with_custody(Custody::Request { _ticket: ticket }, CaptureUsage::default())
    }
    fn with_custody(custody: Custody, usage: CaptureUsage) -> Self {
        Self(Some(Arc::new(Owner {
            state: Mutex::new(State {
                usage,
                active: false,
            }),
            _custody: custody,
        })))
    }
    pub(crate) fn same_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(self.0.as_ref().expect("live ledger"), other.0.as_ref().expect("live ledger"))
    }
    pub(crate) fn inspection_control_bytes() -> Option<usize> {
        Some(
            size_of::<(&Self, CaptureUsage)>()
                .checked_add(size_of::<std::sync::MutexGuard<'_, State>>())?
                .checked_add(size_of::<TryLockError<std::sync::MutexGuard<'_, State>>>())?
                .checked_add(size_of::<Result<CaptureUsage, WorkingMemoryError>>())?,
        )
    }
    /// Reads the settled cumulative value without starting a ledger loan or
    /// changing its active state. An in-progress callback cannot be inspected.
    pub(crate) fn inspect_usage(&self) -> Result<CaptureUsage, WorkingMemoryError> {
        let state = self
            .0
            .as_ref()
            .expect("live ledger")
            .state
            .try_lock()
            .map_err(|error| match error {
                TryLockError::WouldBlock => WorkingMemoryError::AccountConstructionBusy,
                TryLockError::Poisoned(_) => WorkingMemoryError::Poisoned,
            })?;
        if state.active {
            return Err(WorkingMemoryError::AccountConstructionBusy);
        }
        Ok(state.usage)
    }
    pub(crate) fn borrow(
        &self,
    ) -> Result<CaptureRunLedgerGuard<'_>, crate::capture::CaptureProtocolError> {
        let owner = self.0.as_ref().expect("live ledger");
        let mut state = owner.state.try_lock().map_err(|error| match error {
            TryLockError::WouldBlock => crate::capture::CaptureProtocolError::CumulativeInUse,
            TryLockError::Poisoned(_) => crate::capture::CaptureProtocolError::CumulativePoisoned,
        })?;
        if state.active {
            return Err(crate::capture::CaptureProtocolError::CumulativeInUse);
        }
        state.active = true;
        let usage = state.usage;
        drop(state);
        // No mutex is held through a backend callback, including panic/reentry.
        Ok(CaptureRunLedgerGuard { owner, usage })
    }
    pub(crate) fn control_bytes() -> Result<u64, WorkingMemoryError> {
        let allocation = Layout::new::<[AtomicUsize; 2]>()
            .extend(Layout::new::<Owner>())
            .map_err(|_| WorkingMemoryError::Overflow)?
            .0
            .pad_to_align()
            .size();
        [
            allocation,
            size_of::<Owner>(),
            size_of::<CaptureRunLedger>(),
            size_of::<CaptureRunLedgerGuard<'_>>(),
            size_of::<Option<Owner>>(),
            size_of::<CaptureUsage>(),
            size_of::<Result<CaptureRunLedgerGuard<'_>, crate::capture::CaptureProtocolError>>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .and_then(|n| u64::try_from(n).ok())
        .ok_or(WorkingMemoryError::Overflow)
    }
}
#[derive(Debug)]
pub(crate) struct CaptureRunLedgerGuard<'a> {
    owner: &'a Owner,
    usage: CaptureUsage,
}
impl CaptureRunLedgerGuard<'_> {
    pub(crate) fn usage(&self) -> CaptureUsage {
        self.usage
    }
    pub(crate) fn record(&mut self, usage: CaptureUsage) {
        self.usage.captures = self.usage.captures.max(usage.captures);
        self.usage.retained_bytes = self.usage.retained_bytes.max(usage.retained_bytes);
        self.usage.host_bytes = self.usage.host_bytes.max(usage.host_bytes);
        self.usage.encoded_bytes = self.usage.encoded_bytes.max(usage.encoded_bytes);
    }
}
impl Drop for CaptureRunLedgerGuard<'_> {
    fn drop(&mut self) {
        // No caller code runs under this fixed lock. Acquiring it during unwind
        // does not poison it merely because the outer callback panicked. If a
        // separate internal panic already poisoned it, retain the spent totals
        // before future operations report the existing poison condition.
        let mut state = self.owner.state.lock().unwrap_or_else(|p| p.into_inner());
        state.usage = self.usage;
        state.active = false;
    }
}
