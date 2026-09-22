//! Unenforced custody for the same finite native families used after admission.
use super::NativeMemoryOwner;
use safemlx::{
    error::Exception,
    fast::{KernelDefinitionError, PreparedMetalKernelFamily},
    OriginalScopeObserver,
};
use std::sync::{Mutex, OnceLock};

/// Serializes source construction without caching a transient allocation error.
/// An admitted call must borrow its paid owner and cannot initialize this cache.
pub(crate) struct UnenforcedFamilyCache {
    ready: OnceLock<PreparedMetalKernelFamily<NativeMemoryOwner>>,
    initialization: Mutex<()>,
}

impl UnenforcedFamilyCache {
    pub(crate) const fn new() -> Self {
        Self {
            ready: OnceLock::new(),
            initialization: Mutex::new(()),
        }
    }

    pub(crate) fn get_or_try_init(
        &self,
        construct: impl FnOnce(
            NativeMemoryOwner,
        ) -> Result<
            PreparedMetalKernelFamily<NativeMemoryOwner>,
            KernelDefinitionError<NativeMemoryOwner>,
        >,
    ) -> Result<&PreparedMetalKernelFamily<NativeMemoryOwner>, Exception> {
        if let Some(observer) = OriginalScopeObserver::try_current()? {
            return Err(observer.capacity_error());
        }
        if let Some(family) = self.ready.get() {
            return Ok(family);
        }
        // A cold ordinary source remains unquoted through cache/error lifetime.
        // It cannot disappear from admission just because the invoking arrays
        // retired, nor can it be promoted into a later paid source family.
        let owner =
            NativeMemoryOwner::acquire_typed(&super::ledger()).map_err(Exception::from_source)?;
        let _guard = self.initialization.lock().map_err(|_| {
            Exception::from_source(InitializationPoisoned {
                owner: owner.clone(),
            })
        })?;
        if let Some(family) = self.ready.get() {
            return Ok(family);
        }
        let family = construct(owner).map_err(Exception::from_source)?;
        self.ready
            .set(family)
            .expect("exclusive kernel source initializer");
        Ok(self.ready.get().expect("initialized kernel source"))
    }
}

#[derive(Debug, thiserror::Error)]
#[error("kernel source initialization failed")]
struct InitializationPoisoned {
    owner: NativeMemoryOwner,
}

#[cfg(all(test, feature = "metal", not(feature = "cuda")))]
mod tests {
    use super::*;
    use safemlx::fast::MetalKernelFamilyPlan;
    use std::sync::Arc;

    #[test]
    fn cold_source_failure_and_cache_keep_actual_unquoted_custody() {
        let cache = UnenforcedFamilyCache::new();
        let family = &super::super::pointwise_kernel::FAMILY;
        let invalid = MetalKernelFamilyPlan {
            definition: family.definition,
            specializations: [],
        };
        let mut failed_owner = None;
        let error = cache
            .get_or_try_init(|owner| {
                failed_owner = Some(Arc::downgrade(owner.shared()));
                invalid.realize(owner)
            })
            .unwrap_err();
        let failed_owner = failed_owner.unwrap();
        assert!(failed_owner.upgrade().is_some());
        assert!(std::error::Error::source(&error).is_some());
        drop(error);
        assert!(failed_owner.upgrade().is_none());

        let mut cached_owner = None;
        cache
            .get_or_try_init(|owner| {
                cached_owner = Some(Arc::downgrade(owner.shared()));
                family.realize(owner)
            })
            .unwrap();
        let cached_owner = cached_owner.unwrap();
        assert!(cached_owner.upgrade().is_some());
        cache
            .get_or_try_init(|_| panic!("completed source constructed twice"))
            .unwrap();
        drop(cache);
        // Native family destruction queues its Rust custody for ordinary host
        // reclamation; dropping the handle alone must not pretend it retired.
        assert!(cached_owner.upgrade().is_some());
        safemlx::reclaim_allocation_owners();
        assert!(cached_owner.upgrade().is_none());
    }
}
