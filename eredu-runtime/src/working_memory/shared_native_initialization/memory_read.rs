//! Admission directly on the checkpoint-owned read constructor.
use super::*;
use eredu_checkpoint::store::{
    MemoryEncodedReadBuildError, MemoryEncodedReadPlan, PreparedEncodedRead,
};

impl SharedNativeInitializer for MemoryEncodedReadPlan<'_> {
    type Output = PreparedEncodedRead<SharedNativeInitializationCustody>;
    type Error = MemoryEncodedReadBuildError<SharedNativeInitializationCustody>;

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
