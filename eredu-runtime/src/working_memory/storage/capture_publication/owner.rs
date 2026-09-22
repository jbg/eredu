//! Same closed C owner through the completed source-witness population.
use super::*;
use eredu_core::ErasedSharedStorageOwner;

#[derive(Clone, Debug)]
pub(in crate::working_memory) struct CaptureSourceOwner {
    source: eredu_core::SharedStorageIdentity,
    // The identity alias retires before the actual source's original custody.
    owner: ErasedSharedStorageOwner,
    validate:
        fn(&ErasedSharedStorageOwner, &MemoryLedger, &Usage) -> Result<(), WorkingMemoryError>,
}
impl CaptureSourceOwner {
    pub(in crate::working_memory) fn new<K: CapturePlanStorageKey>(
        owner: SharedStorageOwner<PublishedCaptureStorage<K>>,
    ) -> Self {
        Self {
            source: owner.source.clone(),
            owner: owner.erase(),
            validate: validate::<K>,
        }
    }
    pub(in crate::working_memory) fn same_source(
        &self,
        source: &eredu_core::SharedStorageIdentity,
    ) -> bool {
        &self.source == source
    }
    pub(in crate::working_memory) fn validate(
        &self,
        pool: &MemoryLedger,
        usage: &Usage,
    ) -> Result<(), WorkingMemoryError> {
        (self.validate)(&self.owner, pool, usage)
    }
}
fn validate<K: CapturePlanStorageKey>(
    owner: &ErasedSharedStorageOwner,
    pool: &MemoryLedger,
    usage: &Usage,
) -> Result<(), WorkingMemoryError> {
    owner
        .downcast_ref::<PublishedCaptureStorage<K>>()
        .ok_or(WorkingMemoryError::IdentityMismatch)?
        .validate(pool, usage)
}
