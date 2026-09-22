//! Metadata retention without a source/manager/native ownership backedge.
use super::*;
use crate::working_memory::funding::RawSpanHostOwner;

/// Accounting-only alias of the original Host-purpose hold. It cannot issue
/// work, expose capacity, refund storage or retain the selected source bank.
/// Every enclosing allocation must deallocate before its last custody alias.
#[derive(Clone, Debug)]
pub struct OriginalHostMetadataCustody {
    _raw: Accounting,
}
#[derive(Clone, Debug)]
enum Accounting {
    Text(RawSpanHostOwner),
    Speculative(crate::working_memory::OriginalSpeculativeBudgetCustody),
    Realtime(crate::working_memory::OriginalRealtimeBudgetCustody),
    Numerical(crate::working_memory::OriginalNumericalBudgetCustody),
}
impl From<RawSpanHostOwner> for OriginalHostMetadataCustody {
    fn from(value: RawSpanHostOwner) -> Self {
        Self {
            _raw: Accounting::Text(value),
        }
    }
}
impl OriginalHostMetadataCustody {
    pub(in crate::working_memory) fn matches_operation(
        &self,
        other: &crate::working_memory::operation_custody::Custody,
    ) -> bool {
        use crate::working_memory::operation_custody::Custody;
        match (&self._raw, other) {
            (Accounting::Text(a), Custody::Text(b)) => b.same_raw_account(a),
            (Accounting::Speculative(a), Custody::Speculative(b)) => a.same_account(b),
            (Accounting::Realtime(a), Custody::Realtime(b)) => a.same_account(b),
            (Accounting::Numerical(a), Custody::Numerical(b)) => a.same_account(b),
            _ => false,
        }
    }

    pub(in crate::working_memory) fn operation_origin_control_bytes() -> Option<usize> {
        use crate::working_memory::{
            OriginalRealtimeBudgetCustody, OriginalSpeculativeBudgetCustody,
            OriginalTextMetadataCustody, operation_custody::Custody,
        };
        use std::mem::{size_of, size_of_val};
        let frames = [
            size_of::<(&Self, &Custody)>(),
            size_of::<(&Accounting, &Custody)>(),
            size_of::<(&OriginalTextMetadataCustody, &RawSpanHostOwner)>(),
            size_of::<(&RawSpanHostOwner, &RawSpanHostOwner)>(),
            size_of::<(
                &OriginalSpeculativeBudgetCustody,
                &OriginalSpeculativeBudgetCustody,
            )>(),
            size_of::<(
                &OriginalRealtimeBudgetCustody,
                &OriginalRealtimeBudgetCustody,
            )>(),
            size_of::<(
                &crate::working_memory::OriginalNumericalBudgetCustody,
                &crate::working_memory::OriginalNumericalBudgetCustody,
            )>(),
            size_of::<bool>(),
        ];
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add)
    }

    pub(in crate::working_memory) fn from_realtime(
        value: crate::working_memory::OriginalRealtimeBudgetCustody,
    ) -> Self {
        Self {
            _raw: Accounting::Realtime(value),
        }
    }
    pub(in crate::working_memory) fn from_numerical(
        value: crate::working_memory::OriginalNumericalBudgetCustody,
    ) -> Self {
        Self {
            _raw: Accounting::Numerical(value),
        }
    }
    pub(in crate::working_memory) fn from_budget(
        value: crate::working_memory::OriginalSpeculativeBudgetCustody,
    ) -> Self {
        Self {
            _raw: Accounting::Speculative(value),
        }
    }
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        match (&self._raw, &other._raw) {
            (Accounting::Text(a), Accounting::Text(b)) => a.same(b),
            (Accounting::Speculative(a), Accounting::Speculative(b)) => a.same_account(b),
            (Accounting::Realtime(a), Accounting::Realtime(b)) => a.same_account(b),
            (Accounting::Numerical(a), Accounting::Numerical(b)) => a.same_account(b),
            _ => false,
        }
    }
    pub(in crate::working_memory) fn pool(&self) -> &crate::working_memory::MemoryLedger {
        match &self._raw {
            Accounting::Text(v) => v.pool(),
            Accounting::Speculative(v) => v.pool(),
            Accounting::Realtime(v) => v.pool(),
            Accounting::Numerical(v) => v.pool(),
        }
    }
    pub(in crate::working_memory) fn account(&self) -> u64 {
        match &self._raw {
            Accounting::Text(v) => v.account(),
            Accounting::Speculative(v) => v.account_id(),
            Accounting::Realtime(v) => v.account_id(),
            Accounting::Numerical(v) => v.account_id(),
        }
    }
    pub(in crate::working_memory) fn quarantine(&self) {
        match &self._raw {
            Accounting::Text(v) => v.quarantine(),
            Accounting::Speculative(v) => v.quarantine(),
            Accounting::Realtime(v) => v.quarantine(),
            Accounting::Numerical(v) => v.quarantine(),
        }
    }
    pub(in crate::working_memory) fn validate_origin_locked(
        &self,
        pool: &crate::working_memory::MemoryLedger,
        usage: &crate::working_memory::Usage,
    ) -> Result<(), WorkingMemoryError> {
        match &self._raw {
            Accounting::Text(v) => v.validate_origin_locked(pool, usage),
            Accounting::Speculative(v) => v.validate_copy_source(pool, usage),
            Accounting::Realtime(v) => v.validate_copy_source(pool, usage),
            Accounting::Numerical(v) => v.validate_copy_source(pool, usage),
        }
    }
    pub(in crate::working_memory) fn from_guard(guard: &OriginalTextControlGuard) -> Self {
        Self {
            _raw: Accounting::Text(guard.custody.raw().clone()),
        }
    }
    /// Qualified managed Arc/Rc extent for an owning producer's actual payload.
    /// This is a layout fact only, never an independent allocation grant.
    pub fn shared_storage_bytes(payload: Layout) -> Result<u64, WorkingMemoryError> {
        qualified_storage::shared_layout_bytes(payload)
    }
    /// Managed extent of a fresh concrete scalar-buffer clone on the pinned
    /// compiler/liballoc/libstd producer. Covers Vec<u8/usize/u64>, String and
    /// this target's PathBuf byte storage, not generic Clone or clone_from.
    /// The caller supplies the actual owning source's checked requested layout
    /// and must admit it before calling that clone; this grants no capacity.
    pub fn fresh_clone_storage_bytes(payload: Layout) -> Result<u64, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        u64::try_from(payload.size()).map_err(|_| WorkingMemoryError::Overflow)
    }
    /// Qualified Box extent, excluding allocator-private bookkeeping. The
    /// caller still needs admission before constructing that exact payload.
    pub fn boxed_storage_bytes(payload: Layout) -> Result<u64, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        u64::try_from(payload.size()).map_err(|_| WorkingMemoryError::Overflow)
    }
    /// Additional PAL storage of one standard Condvar initialized before
    /// publication. The caller must prevent competing lazy constructors.
    pub fn initialized_condvar_bytes() -> Result<u64, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
        {
            u64::try_from(std::mem::size_of::<libc::pthread_cond_t>())
                .map_err(|_| WorkingMemoryError::Overflow)
        }
        #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
        {
            Err(WorkingMemoryError::UnknownBound)
        }
    }
    /// Additional PAL storage for an explicitly initialized, unshared standard
    /// Mutex. Competing lazy initialization candidates are not covered.
    pub fn initialized_mutex_bytes() -> Result<u64, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        crate::working_memory::fixed_baseline::pal_mutex_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)
    }
}

/// The same admitted Copy Vec backing, retaining raw accounting after the
/// allocation. Conversion never extracts a Vec, reallocates or gains capacity.
#[derive(Debug)]
pub struct OriginalHostMetadataVec<T: Copy> {
    values: Vec<T>,
    _custody: OriginalHostMetadataCustody,
}
impl<T: Copy> OriginalHostMetadataVec<T> {
    /// Exact initialized extent of this intact owner.
    pub fn as_slice(&self) -> &[T] {
        &self.values
    }
    /// Mutate existing initialized elements without growing the allocation.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.values
    }
    /// Actual exposed capacity, unchanged from the admitted Vec.
    pub fn capacity(&self) -> usize {
        self.values.capacity()
    }
}
impl<T: Copy> OriginalHostVec<T> {
    /// Preserve the exact backing while replacing the full source guard by its
    /// accounting-only retention alias. This supplies no new allocation right.
    pub fn into_metadata(self) -> OriginalHostMetadataVec<T> {
        let Self {
            values, _receipt, ..
        } = self;
        let metadata = OriginalHostMetadataVec {
            values,
            _custody: OriginalHostMetadataCustody::from_guard(&_receipt._controls),
        };
        // Full source custody can reach a provider key destructor. Assemble the
        // values-first owner before that retirement, so its unwind also frees
        // the allocation before its last raw accounting alias.
        drop(_receipt);
        metadata
    }
}
