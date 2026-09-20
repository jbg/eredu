//! Admission for the checkpoint-owned borrowed metadata index.
use super::*;
use eredu_checkpoint::recipe::{
    ReadBatchCatalog, ReadBatchCatalogBuildError, ReadBatchCatalogPlan,
};

impl<'a> SharedNativeInitializer for ReadBatchCatalogPlan<'a> {
    type Output = ReadBatchCatalog<'a, SharedNativeInitializationCustody>;
    type Error = ReadBatchCatalogBuildError<SharedNativeInitializationCustody>;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        if !super::super::qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        self.required_bytes::<SharedNativeInitializationCustody>()
            .ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        self.construct(custody)
    }
}

#[cfg(test)]
mod tests;
