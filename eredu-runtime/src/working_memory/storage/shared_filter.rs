//! Read-only source authentication; no source adoption, registration or pin birth.
use super::*;
use eredu_core::{SharedStorageAttachmentError, SharedStorageIdentity, SharedTokenFilter, SharedControllerSource};
impl WorkingMemoryPool {
    /// Named borrowed registration/attachment query transports; no allocation.
    pub fn shared_controller_source_validation_control_bytes() -> Option<usize> {
        use std::mem::{size_of, size_of_val};
        let parts = [size_of::<SharedControllerSource<'_>>(),
            size_of::<(&Self, SharedControllerSource<'_>)>(),
            size_of::<std::sync::MutexGuard<'_, super::super::Usage>>(),
            size_of::<Result<bool, SharedStorageAttachmentError<std::convert::Infallible>>>(),
            size_of::<Result<(), WorkingMemoryError>>(), size_of::<TypeId>(),
            size_of::<Option<&Registry<SharedStorageIdentity>>>(),
            size_of::<Option<(EntryLocator, &Entry)>>(), size_of::<(u64, bool)>()];
        parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Verifies this borrowed immutable filter has its own retained accounting
    /// attachment and an exact canonical registration in this pool. A retained
    /// alias keeps that attachment alive after this check. Empty numerical
    /// payloads require no registration; no capacity is discounted or granted.
    pub fn validate_shared_token_filter_source(
        &self,
        source: &SharedTokenFilter,
    ) -> Result<(), WorkingMemoryError> {
        self.validate_shared_controller_source(SharedControllerSource::Filter(source))
    }
    /// Verifies the actual immutable source's attachment and exact registration.
    /// This is the same worker for filters, recipe bytes and declarations; it
    /// neither creates custody nor discounts independently paid source storage.
    pub fn validate_shared_controller_source(&self, source: SharedControllerSource<'_>)
        -> Result<(), WorkingMemoryError> {
        let bytes = source
            .capacity_bytes()
            .ok_or(WorkingMemoryError::Overflow)?;
        if bytes == 0 {
            return Ok(());
        }
        let attached = source
            .has_accounting_custody(self.shared_storage_domain())
            .map_err(|error| match error {
                SharedStorageAttachmentError::Poisoned => WorkingMemoryError::Poisoned,
                _ => WorkingMemoryError::IdentityMismatch,
            })?;
        if !attached {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        let usage = self
            .0
            .usage
            .lock()
            .map_err(|_| WorkingMemoryError::Poisoned)?;
        let registry = usage
            .storage
            .get(&TypeId::of::<SharedStorageIdentity>())
            .and_then(|r| r.downcast_ref::<Registry<SharedStorageIdentity>>())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        let (_, entry) = registry
            .locate(source.identity())
            .ok_or(WorkingMemoryError::IdentityMismatch)?;
        if entry.bytes != bytes || entry.owners == 0 {
            return Err(WorkingMemoryError::IdentityMismatch);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::TokenFilter;
    #[test]
    fn attached_filter_source_requires_exact_pool_owner_and_registration() {
        let pool = WorkingMemoryPool::new(4096, 0).unwrap();
        let other = WorkingMemoryPool::new(4096, 0).unwrap();
        let source = pool
            .prepare_shared_token_filter(|| TokenFilter::allowed(vec![true, false, true]).unwrap())
            .unwrap();
        let before = pool.used_bytes().unwrap();
        pool.validate_shared_token_filter_source(&source).unwrap();
        pool.validate_shared_token_filter_source(&source.clone())
            .unwrap();
        assert!(matches!(
            other.validate_shared_token_filter_source(&source),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        let same_contents = SharedTokenFilter::new(source.as_ref().clone());
        let registration = pool
            .register_storage([(
                same_contents.identity().clone(),
                same_contents.capacity_bytes().unwrap(),
            )])
            .unwrap();
        // An external registration alone does not give surviving source aliases custody.
        assert!(matches!(
            pool.validate_shared_token_filter_source(&same_contents),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        drop(registration);
        assert_eq!(pool.used_bytes().unwrap(), before);
        pool.validate_shared_token_filter_source(&SharedTokenFilter::new(TokenFilter::All))
            .unwrap();
        drop(source);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
