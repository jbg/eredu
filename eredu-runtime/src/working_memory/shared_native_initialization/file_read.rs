//! Original file-read metadata admission, separate from source/header ownership.
use super::*;
use eredu_checkpoint::store::{
    PreparedSafetensorsEncodedRead, SafetensorsEncodedReadBuildError, SafetensorsEncodedReadPlan,
    SafetensorsEncodedReadPlanError, SafetensorsWeightStore,
};

/// Binds the actual source's prepared headers before comparing the original
/// read-metadata constructor. Source/header/diagnostic and key storage retain
/// separate admission. The plan never opens a source or initializes a header.
pub struct SafetensorsEncodedReadInitializer<'a>(SafetensorsEncodedReadPlan<'a>);
impl<'a> SafetensorsEncodedReadInitializer<'a> {
    /// Inspect only metadata already retained by this exact immutable source.
    pub fn new(
        source: &'a SafetensorsWeightStore,
        keys: &'a [String],
    ) -> Result<Self, SafetensorsEncodedReadPlanError<'a>> {
        SafetensorsEncodedReadPlan::new(source, keys).map(Self)
    }
    /// Complete original constructor contribution including the retained account.
    pub fn required_bytes(&self) -> Result<u64, WorkingMemoryError> {
        WorkingMemoryPool::shared_native_initialization_required_bytes(self)
    }
    /// Compare before allocation and keep the original account through the
    /// completed file batch or its actual failed prefix. Later read scratch and
    /// output destinations remain separate from constructor admission.
    pub fn prepare(
        self,
        pool: &WorkingMemoryPool,
    ) -> Result<
        InitializedSharedNative<PreparedSafetensorsEncodedRead<SharedNativeInitializationCustody>>,
        SharedNativeInitializationError<Self>,
    > {
        pool.initialize_shared_native(self)
    }
}
impl SharedNativeInitializer for SafetensorsEncodedReadInitializer<'_> {
    type Output = PreparedSafetensorsEncodedRead<SharedNativeInitializationCustody>;
    type Error = SafetensorsEncodedReadBuildError<SharedNativeInitializationCustody>;
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
