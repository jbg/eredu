//! Fixed allocations owned by an explicitly initialized pool, not its requests.
use super::{qualified_storage, Pool, WorkingMemoryError, WorkingMemoryPool};

impl WorkingMemoryPool {
    /// Managed heap extent retained by one explicitly initialized empty pool.
    /// Includes its shared Pool, separate shared domain identity and the pinned
    /// standard mutex's PAL allocation. Cloned handles allocate none of these.
    /// Dynamic storage registrations/accounts and thread initialization are not
    /// included. An unsupported compiler/target returns `UnknownBound`.
    pub fn fixed_owner_bytes() -> Result<u64, WorkingMemoryError> {
        let pool = qualified_storage::shared_bytes::<Pool>()?;
        let domain = qualified_storage::shared_layout_bytes(
            eredu_core::SharedStorageDomain::shared_payload_layout(),
        )?;
        let mutex = pal_mutex_bytes().ok_or(WorkingMemoryError::UnknownBound)?;
        pool.checked_add(domain)
            .and_then(|bytes| bytes.checked_add(mutex))
            .ok_or(WorkingMemoryError::Overflow)
    }

    /// Creates a pool whose fixed allocations are charged once in `existing`.
    /// `existing` supplies only the caller's disjoint lifetime-long baseline;
    /// this method adds `fixed_owner_bytes` and initializes the private mutex
    /// before returning any handle. One exclusive initializer therefore creates
    /// one PAL mutex, with no competing first-lock candidates. Ordinary `new`
    /// retains its existing behavior and does not add this owner contribution.
    ///
    /// This is accounting for existing storage, not a native/request grant or a
    /// bound on thread/selected-context initialization. Callers must account for
    /// their containing static/owner representation separately.
    pub fn new_with_fixed_owner_baseline(
        capacity: u64,
        existing: u64,
    ) -> Result<Self, WorkingMemoryError> {
        let existing = existing
            .checked_add(Self::fixed_owner_bytes()?)
            .ok_or(WorkingMemoryError::Overflow)?;
        let pool = Self::new(capacity, existing)?;
        // The only handle remains local. On pinned Darwin this initializes the
        // already charged Box<pal::Mutex>; no competing constructor can run.
        drop(
            pool.0
                .usage
                .lock()
                .map_err(|_| WorkingMemoryError::Poisoned)?,
        );
        Ok(pool)
    }
}

pub(super) fn pal_mutex_bytes() -> Option<u64> {
    // Existing qualification pins the actual Darwin libstd Mutex selector,
    // OnceBox producer and PAL UnsafeCell<pthread_mutex_t> backing. Use the
    // target host type, not an invented private std layout or observed malloc.
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    {
        u64::try_from(std::mem::size_of::<libc::pthread_mutex_t>()).ok()
    }
    #[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialized_pool_owner_baseline_is_exact_and_shared() {
        let required = WorkingMemoryPool::fixed_owner_bytes();
        if std::env::var_os("EREDU_REQUIRE_STATIC_BASELINE_QUALIFICATION").is_some() {
            assert!(required.is_ok());
        }
        let bytes = match required {
            Ok(bytes) => bytes,
            Err(WorkingMemoryError::UnknownBound) => {
                assert!(matches!(
                    WorkingMemoryPool::new_with_fixed_owner_baseline(u64::MAX, 7),
                    Err(WorkingMemoryError::UnknownBound)
                ));
                return;
            }
            Err(error) => panic!("unexpected owning layout refusal: {error}"),
        };
        assert!(bytes > 0);
        assert!(
            matches!(WorkingMemoryPool::new_with_fixed_owner_baseline(bytes + 6, 7),
            Err(WorkingMemoryError::BudgetExceeded {
                required_bytes, available_bytes,
            }) if required_bytes == bytes + 7 && available_bytes == bytes + 6)
        );
        let pool = WorkingMemoryPool::new_with_fixed_owner_baseline(bytes + 7, 7).unwrap();
        let alias = pool.clone();
        assert!(pool.same_domain(&alias));
        assert_eq!(pool.used_bytes().unwrap(), bytes + 7);
        drop(pool);
        assert_eq!(alias.used_bytes().unwrap(), bytes + 7);
        assert_eq!(alias.unquoted_owner_count().unwrap(), 0);
        assert!(matches!(
            WorkingMemoryPool::new_with_fixed_owner_baseline(u64::MAX, u64::MAX),
            Err(WorkingMemoryError::Overflow)
        ));
    }
}
