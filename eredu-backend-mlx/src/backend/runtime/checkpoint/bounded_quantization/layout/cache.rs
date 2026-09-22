use super::super::preflight::quantization_error;
use super::super::{memory, Error, BOUNDED_QUANTIZATION_MAX_CACHE_BYTES};
use crate::backend::ordinary_retirement::OrdinaryRetirement;
use eredu_runtime::working_memory::{
    InitializedSharedNative, MemoryLedger, SharedNativeInitializationCustody,
    SharedNativeInitializationError, SharedNativeInitializer, WorkingMemoryError,
};
use std::{cell::Cell, convert::Infallible};

pub(in super::super) struct BoundedAllocatorCache {
    working_set_limit_bytes: u64,
    retained_limit_bytes: u64,
    cleanup: CleanupOwner,
}

enum CleanupOwner {
    Ordinary(OrdinaryRetirement<AllocatorCacheCleanup>),
    Admitted(InitializedSharedNative<OrdinaryRetirement<AllocatorCacheCleanup>>),
}

struct AllocatorCacheCleanup {
    finished: Cell<bool>,
    // This custody follows the actual cleanup node into deferred retirement.
    // Dropping the enclosing pipeline cannot refund a queued cleanup owner.
    _custody: Option<SharedNativeInitializationCustody>,
}

struct Initializer {
    window_control_bytes: usize,
}

impl SharedNativeInitializer for Initializer {
    type Output = OrdinaryRetirement<AllocatorCacheCleanup>;
    type Error = Infallible;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        let retirement = OrdinaryRetirement::<AllocatorCacheCleanup>::control_bytes()
            .ok_or(WorkingMemoryError::UnknownBound)?;
        usize::try_from(retirement)
            .ok()
            .and_then(|bytes| bytes.checked_add(self.window_control_bytes))
            .and_then(|bytes| bytes.checked_add(size_of::<BoundedAllocatorCache>()))
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Infallible> {
        Ok(OrdinaryRetirement::new(AllocatorCacheCleanup {
            finished: Cell::new(false),
            _custody: Some(custody),
        }))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("allocator cache controls: {0}")]
pub(in super::super) struct CacheAdmissionError(
    #[source] SharedNativeInitializationError<Initializer>,
);

impl CacheAdmissionError {
    pub(in super::super) fn into_backend_failure(self) -> eredu_core::BackendFailure {
        eredu_core::BackendFailure::from_error(
            self.0
                .into_parts()
                .1
                .retire_output_and_map_error(|never| -> Infallible { match never {} }),
        )
    }
}

impl BoundedAllocatorCache {
    pub(in super::super) fn new(working_set_limit_bytes: u64) -> Self {
        Self::with_cleanup(
            working_set_limit_bytes,
            CleanupOwner::Ordinary(OrdinaryRetirement::new(AllocatorCacheCleanup {
                finished: Cell::new(false),
                _custody: None,
            })),
        )
    }

    pub(in super::super) fn prepare_original(
        pool: &MemoryLedger,
        working_set_limit_bytes: u64,
        window_control_bytes: usize,
    ) -> Result<Self, CacheAdmissionError> {
        let cleanup = pool
            .initialize_shared_native(Initializer {
                window_control_bytes,
            })
            .map_err(CacheAdmissionError)?;
        Ok(Self::with_cleanup(
            working_set_limit_bytes,
            CleanupOwner::Admitted(cleanup),
        ))
    }

    #[cfg(test)]
    pub(in super::super) fn required_original_bytes(
        window_control_bytes: usize,
    ) -> Result<u64, WorkingMemoryError> {
        MemoryLedger::shared_native_initialization_required_bytes(&Initializer {
            window_control_bytes,
        })
    }

    fn with_cleanup(working_set_limit_bytes: u64, cleanup: CleanupOwner) -> Self {
        Self {
            working_set_limit_bytes,
            retained_limit_bytes: (working_set_limit_bytes / 4)
                .min(BOUNDED_QUANTIZATION_MAX_CACHE_BYTES),
            cleanup,
        }
    }

    pub(in super::super) fn begin(&mut self) -> Result<(), Error> {
        memory::clear_cache()?;
        Ok(())
    }

    pub(in super::super) fn prepare_submission(
        &mut self,
        queued_working_set_bytes: u64,
        incoming_tile_bytes: u64,
    ) -> Result<(), Error> {
        let planned_bytes = queued_working_set_bytes
            .checked_add(incoming_tile_bytes)
            .ok_or_else(|| quantization_error("conversion submission working-set overflow"))?;
        if planned_bytes > self.working_set_limit_bytes {
            return Err(quantization_error(format!(
                "conversion submission requires {planned_bytes} working-set bytes, but the plan permits {}",
                self.working_set_limit_bytes
            )));
        }
        self.clear_if_needed(planned_bytes)
    }

    pub(in super::super) fn tile_completed(
        &mut self,
        queued_working_set_bytes: u64,
    ) -> Result<(), Error> {
        if queued_working_set_bytes > self.working_set_limit_bytes {
            return Err(quantization_error(format!(
                "queued conversion tiles require {queued_working_set_bytes} working-set bytes, but the plan permits {}",
                self.working_set_limit_bytes
            )));
        }
        self.clear_if_needed(queued_working_set_bytes)
    }

    fn clear_if_needed(&mut self, active_working_set_bytes: u64) -> Result<(), Error> {
        let cached_bytes = u64::try_from(memory::cache_memory()?)
            .map_err(|_| quantization_error("allocator-cache bytes are not representable"))?;
        let available_cache_bytes = self
            .working_set_limit_bytes
            .checked_sub(active_working_set_bytes)
            .expect("validated active conversion working set");
        if allocator_cache_requires_clear(
            cached_bytes,
            self.retained_limit_bytes,
            available_cache_bytes,
        ) {
            memory::clear_cache()?;
        }
        Ok(())
    }

    pub(in super::super) fn finish(&mut self) -> Result<(), Error> {
        memory::clear_cache()?;
        let cleanup = match &self.cleanup {
            CleanupOwner::Ordinary(cleanup) => &**cleanup,
            CleanupOwner::Admitted(cleanup) => &**cleanup.output(),
        };
        cleanup.finished.set(true);
        Ok(())
    }
}

pub(in super::super) const fn allocator_cache_requires_clear(
    cached_bytes: u64,
    retained_limit_bytes: u64,
    available_cache_bytes: u64,
) -> bool {
    cached_bytes > retained_limit_bytes || cached_bytes > available_cache_bytes
}

impl Drop for AllocatorCacheCleanup {
    fn drop(&mut self) {
        if !self.finished.get() {
            let _ = memory::clear_cache();
        }
    }
}

#[cfg(test)]
mod tests;
