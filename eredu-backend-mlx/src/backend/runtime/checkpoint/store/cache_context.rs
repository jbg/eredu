//! One actual shared cache owner; stream handles and cache rows are separate.
use super::cache::Index;
use eredu_runtime::working_memory::{
    MemoryLedger, OriginalHostMetadataCustody, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
    WorkingMemoryReservation,
};
use std::{
    alloc::Layout,
    fmt,
    mem::{size_of, size_of_val},
    sync::{Arc, LockResult, Mutex, MutexGuard, TryLockResult},
};

#[derive(Debug)]
struct Body {
    // All rows and the PAL mutex retire before raw accounting. The handle's
    // custom Drop deallocates the Arc block before either field is destroyed.
    index: Mutex<Index>,
    origin: Option<SharedNativeInitializationCustody>,
}

/// Only this wrapper can own the Arc. No Weak or bare Arc escapes, including
/// from ordinary contexts, so final into_inner retires the allocation first.
#[derive(Debug)]
pub(crate) struct CacheHandle(Option<Arc<Body>>);
impl Clone for CacheHandle {
    fn clone(&self) -> Self {
        Self(Some(Arc::clone(self.0.as_ref().expect("live cache"))))
    }
}
impl Drop for CacheHandle {
    fn drop(&mut self) {
        if let Some(body) = self.0.take() {
            drop(Arc::into_inner(body));
        }
    }
}
impl CacheHandle {
    fn body(&self) -> &Body {
        self.0.as_deref().expect("live cache")
    }
    pub(super) fn lock(&self) -> LockResult<MutexGuard<'_, Index>> {
        self.body().index.lock()
    }
    pub(super) fn try_lock(&self) -> TryLockResult<MutexGuard<'_, Index>> {
        self.body().index.try_lock()
    }
    pub(super) fn ordinary() -> Self {
        Self::construct(None).expect("unpublished cache mutex")
    }
    fn construct(
        origin: Option<SharedNativeInitializationCustody>,
    ) -> Result<Self, ConstructionError> {
        // Install the retirement wrapper before the first PAL initialization.
        let out = Self(Some(Arc::new(Body {
            index: Mutex::new(Index::default()),
            origin,
        })));
        let status = match out.lock() {
            Ok(guard) => {
                drop(guard);
                Ok(())
            }
            Err(poisoned) => {
                drop(poisoned);
                Err(WorkingMemoryError::Poisoned)
            }
        };
        match status {
            Ok(()) => Ok(out),
            Err(cause) => Err(ConstructionError {
                cause,
                _prefix: out,
            }),
        }
    }
    pub(crate) fn prepare(pool: &MemoryLedger) -> Result<Self, CacheInitializationError> {
        let initialized = pool
            .initialize_shared_native(Initializer)
            .map_err(CacheInitializationError)?;
        // The concrete body holds the incoming raw account independently of
        // this constructor-result wrapper. This alias is allocation-free.
        Ok(initialized.output().clone())
    }
    pub(crate) fn required_storage_bytes() -> Result<u64, WorkingMemoryError> {
        MemoryLedger::shared_native_initialization_required_bytes(&Initializer)
    }
    /// Fixed identity/validation transports. The caller prices this beside its
    /// lookup/install frames, not again in the cache's shared birth allowance.
    pub(crate) fn identity_control_bytes() -> Option<usize> {
        let controls = [
            size_of::<[&CacheHandle; 2]>(),
            size_of::<[&Body; 2]>(),
            size_of::<[&Arc<Body>; 2]>(),
            size_of::<Arc<Body>>(),
            size_of::<Option<Arc<Body>>>(),
            size_of::<Option<&SharedNativeInitializationCustody>>(),
            size_of::<&SharedNativeInitializationCustody>(),
            size_of::<&WorkingMemoryReservation>(),
            size_of::<&MemoryLedger>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<WorkingMemoryError>(),
            size_of::<bool>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(
            self.0.as_ref().expect("live cache"),
            other.0.as_ref().expect("live cache"),
        )
    }
    pub(crate) fn validate_pool(&self, pool: &MemoryLedger) -> Result<(), WorkingMemoryError> {
        self.body()
            .origin
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?
            .validate_pool(pool)
    }
    pub(crate) fn validate_reservation(
        &self,
        reservation: &WorkingMemoryReservation,
    ) -> Result<(), WorkingMemoryError> {
        self.body()
            .origin
            .as_ref()
            .ok_or(WorkingMemoryError::UnknownBound)?
            .validate_reservation(reservation)
    }
    /// Retained body/header and one eagerly initialized PAL mutex. Dynamic
    /// rows and construction frames have separate owners/contributions.
    pub(super) fn retained_storage_bytes() -> Result<u64, WorkingMemoryError> {
        OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<Body>())?
            .checked_add(OriginalHostMetadataCustody::initialized_mutex_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)
    }
}

#[derive(Debug)]
struct Initializer;
impl SharedNativeInitializer for Initializer {
    type Output = CacheHandle;
    type Error = ConstructionError;
    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let controls = [
            size_of::<Body>(),
            size_of::<Option<SharedNativeInitializationCustody>>(),
            size_of::<Arc<Body>>(),
            size_of::<CacheHandle>(),
            size_of::<&MemoryLedger>(),
            size_of::<LockResult<MutexGuard<'static, Index>>>(),
            size_of::<MutexGuard<'static, Index>>(),
            size_of::<std::sync::PoisonError<MutexGuard<'static, Index>>>(),
            size_of::<Option<Body>>(),
            size_of::<&CacheHandle>(),
            size_of::<&Mutex<Index>>(),
            size_of::<Result<(), WorkingMemoryError>>(),
            size_of::<CacheInitializationError>(),
            size_of::<Result<CacheHandle, CacheInitializationError>>(),
        ];
        let retained = usize::try_from(CacheHandle::retained_storage_bytes()?)
            .map_err(|_| WorkingMemoryError::Overflow)?;
        controls
            .into_iter()
            .try_fold(retained, usize::checked_add)
            .and_then(|n| n.checked_add(size_of_val(&controls)))
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        CacheHandle::construct(Some(custody))
    }
}
#[derive(Debug)]
struct ConstructionError {
    cause: WorkingMemoryError,
    _prefix: CacheHandle,
}
impl fmt::Display for ConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.cause.fmt(f)
    }
}
impl std::error::Error for ConstructionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// The original typed rejection or actual cache/PAL prefix and account.
#[derive(Debug)]
pub(crate) struct CacheInitializationError(SharedNativeInitializationError<Initializer>);
impl fmt::Display for CacheInitializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for CacheInitializationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[cfg(test)]
pub(super) mod tests;
