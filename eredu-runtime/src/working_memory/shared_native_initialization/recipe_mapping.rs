//! Admission for checkpoint-owned encoded byte-coordinate mappings.
use super::*;
use eredu_checkpoint::recipe::{EncodedRecipeMapping, EncodedRecipeMappingPlan, RecipeError};

impl SharedNativeInitializer for EncodedRecipeMappingPlan<'_> {
    type Output = EncodedRecipeMapping;
    type Error = RecipeError;

    fn required_storage_bytes(&self) -> Result<usize, WorkingMemoryError> {
        if !super::super::qualified_storage::qualified() {
            return Err(WorkingMemoryError::UnknownBound);
        }
        self.required_bytes().ok_or(WorkingMemoryError::Overflow)
    }

    fn initialize(
        self,
        _custody: SharedNativeInitializationCustody,
    ) -> Result<Self::Output, Self::Error> {
        // Output and errors remain in the shared initialization owner. Temporary
        // construction storage retires before return; no independent alias escapes.
        self.build()
    }
}

#[cfg(test)]
mod tests;
