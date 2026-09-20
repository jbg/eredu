//! Admission for original memory-backed checkpoint read metadata.
use super::*;
use eredu_checkpoint::store::{
    MemoryEncodedReadBuildError, MemoryEncodedReadPlan, MemoryEncodedReadPlanError,
    MemoryWeightStore, PreparedMemoryEncodedRead,
};

/// Binds the actual source and ordered occurrences before the cold pool compares
/// their constructor storage. Existing source payloads and keys need their own
/// retained admission; this initializer neither registers nor copies them.
pub struct MemoryEncodedReadInitializer<'a>(MemoryEncodedReadPlan<'a>);
impl<'a> MemoryEncodedReadInitializer<'a> {
    /// Inspect the actual immutable source without constructing read metadata.
    pub fn new(
        source: &'a MemoryWeightStore,
        keys: &'a [String],
    ) -> Result<Self, MemoryEncodedReadPlanError> {
        MemoryEncodedReadPlan::new(source, keys).map(Self)
    }

    /// Total managed constructor contribution including the retained account.
    pub fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        WorkingMemoryPool::shared_native_initialization_required_bytes(self)
    }

    /// Compare before allocation and retain the original account through the
    /// successful batch or failed constructor prefix. No payload is read here.
    pub fn prepare(
        self,
        pool: &WorkingMemoryPool,
    ) -> Result<
        InitializedSharedNative<PreparedMemoryEncodedRead<SharedNativeInitializationCustody>>,
        SharedNativeInitializationError<Self>,
    > {
        pool.initialize_shared_native(self)
    }
}
impl SharedNativeInitializer for MemoryEncodedReadInitializer<'_> {
    type Output = PreparedMemoryEncodedRead<SharedNativeInitializationCustody>;
    type Error = MemoryEncodedReadBuildError<SharedNativeInitializationCustody>;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        if !super::super::qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        self.0
            .required_bytes::<SharedNativeInitializationCustody>()
            .ok_or(WorkingMemoryError::Overflow)
    }
    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        self.0.construct(custody)
    }
}

#[cfg(test)]
mod tests;
