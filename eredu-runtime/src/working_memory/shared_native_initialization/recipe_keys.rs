//! Source-key admission through the existing shared constructor account.
use super::*;
use eredu_checkpoint::recipe::{
    EncodedRecipeKeysBuildError, EncodedRecipeKeysPlan, PreparedEncodedRecipeKeys,
};

impl SharedNativeInitializer for EncodedRecipeKeysPlan<'_> {
    type Output = PreparedEncodedRecipeKeys<SharedNativeInitializationCustody>;
    type Error = EncodedRecipeKeysBuildError<SharedNativeInitializationCustody>;

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
