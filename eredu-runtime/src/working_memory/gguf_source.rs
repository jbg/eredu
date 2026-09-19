//! Finite source reader/materializer construction, separate from request fit.
use super::{qualified_storage, WorkingMemoryError, WorkingMemoryPool};
use eredu_checkpoint::gguf_store::{
    GgufSourceStorageRequest, GgufWeightStore, PreparedGgufCatalog, PreparedGgufSourceFailure,
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Weak,
    },
};

// A canonical SourceStorageIdentity can live in this pool's registry. This
// account must therefore never own the pool strongly. Its one prepaid ledger
// contribution remains until all source/metadata/weak-key owners retire; if
// the pool itself retired first, there is no surviving ledger to credit.
#[derive(Debug)]
struct SourceAccountInner {
    pool: Weak<super::Pool>,
    bytes: u64,
    active: AtomicBool,
    compiling: AtomicBool,
}
impl Drop for SourceAccountInner {
    fn drop(&mut self) {
        if !*self.active.get_mut() {
            return;
        }
        let Some(pool) = self.pool.upgrade() else {
            return;
        };
        if let Ok(mut usage) = pool.usage.lock() {
            if *self.compiling.get_mut() {
                usage.reservations -= 1;
            }
            usage.reserved -= self.bytes;
        };
    }
}
#[derive(Debug)]
pub(super) struct SourceAccount(Option<Arc<SourceAccountInner>>);
impl Drop for SourceAccount {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            drop(Arc::into_inner(owner));
        }
    }
}
impl SourceAccount {
    pub(super) fn storage_bytes() -> Result<u64, WorkingMemoryError> {
        qualified_storage::shared_bytes::<SourceAccountInner>()
    }
    pub(super) fn new_unarmed(pool: &WorkingMemoryPool, bytes: u64) -> Self {
        Self(Some(Arc::new(SourceAccountInner {
            pool: Arc::downgrade(&pool.0),
            bytes,
            active: AtomicBool::new(false),
            compiling: AtomicBool::new(true),
        })))
    }
    fn inner(&self) -> &SourceAccountInner {
        self.0.as_deref().expect("source account")
    }
    pub(super) fn activate(&self) {
        self.inner().active.store(true, Ordering::Relaxed);
    }
    pub(super) fn share(&self) -> Self {
        Self(self.0.clone())
    }
    pub(super) fn matches_pool(&self, pool: &WorkingMemoryPool) -> bool {
        Weak::ptr_eq(&self.inner().pool, &Arc::downgrade(&pool.0))
    }
    pub(super) fn finish(&self) -> Result<(), WorkingMemoryError> {
        let pool = self
            .inner()
            .pool
            .upgrade()
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let mut usage = pool
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        if self.inner().compiling.swap(false, Ordering::Relaxed) {
            usage.reservations -= 1;
        }
        Ok(())
    }
}
// Shared only by runtime-owned constructors that reserve storage before birth.
// Providers cannot construct or extract this physical-inventory credit.
#[derive(Debug)]
pub(super) struct SourcePayloadCustody {
    account: SourceAccount,
    physical_inventory: u64,
    prepaid_inventory: u64,
}

impl SourcePayloadCustody {
    pub(super) fn new(
        account: SourceAccount,
        physical_inventory: u64,
        prepaid_inventory: u64,
    ) -> Self {
        Self {
            account,
            physical_inventory,
            prepaid_inventory,
        }
    }
}

/// Key projection for checkpoint-owned encoded payload or reader inventories.
/// This returns only the genuine opaque identity already held in that key;
/// it provides no amount, coverage callback or grant. Key equality must retain
/// the ordinary physical-identity contract of register_storage.
pub trait GgufSourceStorageKey {
    /// Actual source identity, or None for any other physical namespace.
    fn gguf_source_identity(&self) -> Option<&eredu_checkpoint::store::SourceStorageIdentity>;
}
impl GgufSourceStorageKey for eredu_checkpoint::store::SourceStorageIdentity {
    fn gguf_source_identity(&self) -> Option<&Self> {
        Some(self)
    }
}
#[derive(Debug)]
pub(in crate::working_memory) struct SourceInventoryOrigin {
    account: SourceAccount,
    total: u64,
    prepaid: u64,
}
impl Clone for SourceInventoryOrigin {
    fn clone(&self) -> Self {
        Self {
            account: self.account.share(),
            total: self.total,
            prepaid: self.prepaid,
        }
    }
}
impl SourceInventoryOrigin {
    pub(in crate::working_memory) fn has_original_constructor(
        identity: &eredu_checkpoint::store::SourceStorageIdentity,
    ) -> bool {
        identity.constructor_control_owner::<SourcePayloadCustody>().is_some()
    }

    pub(in crate::working_memory) fn inspect(
        identity: &eredu_checkpoint::store::SourceStorageIdentity,
        bytes: u64,
        pool: &WorkingMemoryPool,
    ) -> Result<Option<Self>, WorkingMemoryError> {
        let Some(custody) = identity.constructor_control_owner::<SourcePayloadCustody>() else {
            return Ok(None);
        };
        if bytes != custody.physical_inventory {
            return Err(WorkingMemoryError::StorageCapacityMismatch {
                expected_bytes: custody.physical_inventory,
                actual_bytes: bytes,
            });
        }
        // Independent pools register the complete physical inventory. They may
        // retain A's original source, but cannot borrow A's prepaid coverage.
        if !custody.account.matches_pool(pool) {
            return Ok(None);
        }
        let origin = Self {
            account: custody.account.share(),
            total: bytes,
            prepaid: custody.prepaid_inventory,
        };
        origin.validate()?;
        Ok(Some(origin))
    }
    pub(in crate::working_memory) fn validate(&self) -> Result<(), WorkingMemoryError> {
        let account = self.account.inner();
        if !account.active.load(Ordering::Relaxed)
            || account.compiling.load(Ordering::Relaxed)
            || self.prepaid > self.total
            || self.prepaid > account.bytes
        {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
    pub(in crate::working_memory) fn validate_pool(
        &self,
        pool: &WorkingMemoryPool,
    ) -> Result<(), WorkingMemoryError> {
        self.validate()?;
        if self.account.matches_pool(pool) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
    pub(in crate::working_memory) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.account.0.as_ref().unwrap(),
            other.account.0.as_ref().unwrap(),
        ) && self.total == other.total
            && self.prepaid == other.prepaid
    }
    pub(in crate::working_memory) fn total(&self) -> u64 {
        self.total
    }
    pub(in crate::working_memory) fn residual(&self) -> u64 {
        self.total - self.prepaid
    }
}
/// Fixed source refusal. Original catalog and every constructor prefix remain
/// owned, and their storage retires before the independent source charge.
#[derive(Debug)]
pub struct OriginalGgufSourceError {
    accounting: Option<WorkingMemoryError>,
    construction: Option<PreparedGgufSourceFailure>,
    input: Option<PreparedGgufCatalog>,
    completed: Option<GgufWeightStore>,
    account: Option<SourceAccount>,
}
impl OriginalGgufSourceError {
    /// Typed pre-admission/qualification/ledger refusal, if any.
    pub fn accounting_failure(&self) -> Option<&WorkingMemoryError> {
        self.accounting.as_ref()
    }
    /// Actual original constructor failure, including successful materializers.
    pub fn construction_failure(&self) -> Option<&PreparedGgufSourceFailure> {
        self.construction.as_ref()
    }
    /// Uncalled original catalog, retained on any pre-admission refusal.
    pub fn rejected_catalog(&self) -> Option<&PreparedGgufCatalog> {
        self.input.as_ref()
    }
}
impl fmt::Display for OriginalGgufSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(cause) = &self.construction {
            cause.fmt(f)
        } else {
            self.accounting.as_ref().expect("source failure").fmt(f)
        }
    }
}
impl std::error::Error for OriginalGgufSourceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.construction {
            Some(cause) => Some(cause),
            None => self.accounting.as_ref().map(|cause| cause as _),
        }
    }
}
struct GgufSourceQuoteParts {
    source: u64,
    custody: u64,
    account: u64,
    checkpoints: u64,
    mutexes: u64,
    controls: u64,
}
impl WorkingMemoryPool {
    /// Validate this exact existing source's already admitted constructor
    /// custody in this pool. No registration, clone, credit or allowance is
    /// created; ordinary and foreign-pool sources remain unqualified.
    pub fn validate_original_source_inventory(
        &self,
        identity: &eredu_checkpoint::store::SourceStorageIdentity,
        bytes: u64,
    ) -> Result<(), WorkingMemoryError> {
        SourceInventoryOrigin::inspect(identity, bytes, self)?
            .ok_or(WorkingMemoryError::UnknownBound)?;
        Ok(())
    }

    /// New finite reader bank, materializer/scratch/coordinate storage, touched
    /// rows, initial empty recipe/PAL controls and source/Checkpoint shared shells.
    /// Original checkpoint/header payload, future recipe entries, OS file handles
    /// and selected context/thread initialization are not certified by this quote.
    pub fn gguf_source_required_bytes(
        catalog: &PreparedGgufCatalog,
    ) -> Result<u64, WorkingMemoryError> {
        if !qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        let request = catalog
            .source_storage_request::<SourcePayloadCustody>()
            .ok_or(WorkingMemoryError::Overflow)?;
        let mutex =
            super::fixed_baseline::pal_mutex_bytes().ok_or(WorkingMemoryError::UnknownBound)?;
        let (count, checkpoint) = request.checkpoint_bodies();
        let controls = [
            size_of::<GgufSourceStorageRequest>(),
            size_of::<Option<GgufSourceStorageRequest>>(),
            size_of::<super::loaded_decode_source::Allowance>(),
            size_of::<SourceAccount>(),
            size_of::<SourcePayloadCustody>(),
            size_of::<OriginalGgufSourceError>(),
            size_of::<Result<GgufWeightStore, OriginalGgufSourceError>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<GgufSourceQuoteParts>(),
            size_of::<usize>(),              // materializer count
            size_of::<std::alloc::Layout>(), // checkpoint body
            size_of::<u64>(),                // total requested bytes
            size_of::<Option<u64>>(),
        ];
        let controls = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or(WorkingMemoryError::Overflow)?;
        let mut bytes =
            u64::try_from(request.requested_bytes()).map_err(|_| WorkingMemoryError::Overflow)?;
        let parts = GgufSourceQuoteParts {
            source: qualified_storage::shared_layout_bytes(request.source_body())?,
            custody: qualified_storage::shared_layout_bytes(request.custody_body())?,
            account: qualified_storage::shared_bytes::<SourceAccountInner>()?,
            checkpoints: qualified_storage::shared_layout_bytes(checkpoint)?
                .checked_mul(u64::try_from(count).map_err(|_| WorkingMemoryError::Overflow)?)
                .ok_or(WorkingMemoryError::Overflow)?,
            mutexes: mutex
                .checked_mul(request.mutex_count() as u64)
                .ok_or(WorkingMemoryError::Overflow)?,
            controls: controls as u64,
        };
        bytes = bytes
            .checked_add(parts.source)
            .and_then(|n| n.checked_add(parts.custody))
            .and_then(|n| n.checked_add(parts.account))
            .and_then(|n| n.checked_add(parts.checkpoints))
            .and_then(|n| n.checked_add(parts.mutexes))
            .and_then(|n| n.checked_add(parts.controls))
            .ok_or(WorkingMemoryError::Overflow)?;
        Ok(bytes)
    }
    /// Continue only this exact admitted immutable catalog into its finite source
    /// constructor. Qualification and the complete comparison precede all new
    /// owning allocations. The original catalog must belong to this same pool.
    pub fn compile_gguf_source(
        &self,
        catalog: PreparedGgufCatalog,
    ) -> Result<GgufWeightStore, OriginalGgufSourceError> {
        let refused = |catalog, cause| OriginalGgufSourceError {
            accounting: Some(cause),
            construction: None,
            input: Some(catalog),
            completed: None,
            account: None,
        };
        if let Err(cause) = self.validate_prepared_gguf_catalog(&catalog) {
            return Err(refused(catalog, cause));
        }
        let bytes = match Self::gguf_source_required_bytes(&catalog) {
            Ok(n) => n,
            Err(e) => return Err(refused(catalog, e)),
        };
        let allowance = match self.admit_source_compiler(bytes) {
            Ok(a) => a,
            Err(e) => return Err(refused(catalog, e)),
        };
        let inventory = catalog
            .source_storage_request::<SourcePayloadCustody>()
            .expect("same qualified immutable input")
            .inventory_bytes();
        let account = allowance.into_source_account();
        match catalog.build_with_source_custody(SourcePayloadCustody {
            account: account.share(),
            physical_inventory: inventory.0,
            prepaid_inventory: inventory.1,
        }) {
            Err(construction) => Err(OriginalGgufSourceError {
                accounting: None,
                construction: Some(construction),
                input: None,
                completed: None,
                account: Some(account),
            }),
            Ok(source) => {
                if let Err(cause) = account.finish() {
                    return Err(OriginalGgufSourceError {
                        accounting: Some(cause),
                        construction: None,
                        input: None,
                        completed: Some(source),
                        account: Some(account),
                    });
                }
                Ok(source)
            }
        }
    }
    /// Read-only origin check from the exact closed source constructor. Mere
    /// prepared storage or ordinary construction never supplies this proof.
    pub fn validate_gguf_source_controls(
        &self,
        source: &GgufWeightStore,
    ) -> Result<(), WorkingMemoryError> {
        let custody = source
            .source_control_owner::<SourcePayloadCustody>()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        if custody.account.matches_pool(self) {
            Ok(())
        } else {
            Err(WorkingMemoryError::IdentityMismatch)
        }
    }
}

#[cfg(test)]
mod tests;
