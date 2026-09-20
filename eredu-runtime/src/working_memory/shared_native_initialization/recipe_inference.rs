//! Admission for checkpoint-owned finite inference over borrowed metadata.
use super::*;
use eredu_checkpoint::recipe::{
    RecipeCatalog, RecipeInferenceError, RecipeInferencePlan, RecipeMetadata,
};

impl<C: RecipeCatalog + ?Sized> SharedNativeInitializer for RecipeInferencePlan<'_, C> {
    type Output = RecipeMetadata;
    type Error = RecipeInferenceError;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        if !super::super::qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        Ok(self.layout().required_bytes())
    }

    fn initialize(
        self,
        _custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        // All scratch retires synchronously. The existing shared initialization
        // result retains its account alongside the owned metadata or error;
        // this constructor publishes no independent alias or native work.
        self.infer()
    }
}

#[cfg(test)]
mod tests;
