//! Pool-backed construction policy for checkpoint's shared recipe traversal.
use super::*;
use eredu_checkpoint::{
    recipe::{
        DerivedWeightRecipe, EncodedRecipeChildren, EncodedRecipeChildrenPlan,
        EncodedRecipeConstruction, EncodedRecipeMapping, EncodedRecipeMappingPlan, RecipeCatalog,
        RecipeError, RecipeInferenceError, RecipeInferenceInput, RecipeInferencePlan,
        RecipeMetadata,
    },
    store::{SelectionReadDestinationPlan, SelectionReadRanges},
};
use std::collections::TryReserveError;

impl<M> SharedNativeInitializer for EncodedRecipeChildrenPlan<M> {
    type Output = EncodedRecipeChildren<M, SharedNativeInitializationCustody>;
    type Error = TryReserveError;
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

/// Recursive encoded-recipe storage admitted through one original pool. Recipe
/// branches, byte mappings and selection semantics remain checkpoint-owned.
/// Source birth, recipe declarations, read scratch and output are prerequisites.
pub struct AdmittedRecipeConstruction<'a> {
    pool: &'a MemoryLedger,
}
impl<'a> AdmittedRecipeConstruction<'a> {
    /// Borrow the pool used for each actual constructor and retained output.
    pub fn new(pool: &'a MemoryLedger) -> Self {
        Self { pool }
    }
}

/// Typed compilation refusal with the actual failed constructor's output/error
/// and accounting. Temporary successful child owners retire synchronously.
#[derive(Debug, thiserror::Error)]
pub enum EncodedRecipeConstructionError {
    /// A fixed semantic or checked-layout error outside an allocation.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Metadata inference retains owned diagnostics and original admission.
    #[error(transparent)]
    Inference(SharedNativeInitializationFailure<RecipeMetadata, RecipeInferenceError>),
    /// Byte-coordinate construction or admission failed.
    #[error(transparent)]
    Mapping(SharedNativeInitializationFailure<EncodedRecipeMapping, RecipeError>),
    /// Selection interval construction or admission failed.
    #[error(transparent)]
    Ranges(SharedNativeInitializationFailure<SelectionReadRanges, TryReserveError>),
    /// Child-array construction or admission failed before any children were added.
    #[error(transparent)]
    Children(
        SharedNativeInitializationFailure<
            EncodedRecipeChildren<
                InitializedSharedNative<EncodedRecipeMapping>,
                SharedNativeInitializationCustody,
            >,
            TryReserveError,
        >,
    ),
}

impl EncodedRecipeConstruction for AdmittedRecipeConstruction<'_> {
    type Error = EncodedRecipeConstructionError;
    type Metadata = InitializedSharedNative<RecipeMetadata>;
    type Mapping = InitializedSharedNative<EncodedRecipeMapping>;
    type Ranges = InitializedSharedNative<SelectionReadRanges>;
    type ChildCustody = (
        SharedNativeInitializationCustody,
        SharedNativeInitializationCustody,
    );

    fn infer<C: RecipeCatalog + ?Sized>(
        &mut self,
        recipe: &DerivedWeightRecipe,
        catalog: &C,
    ) -> Result<Self::Metadata, Self::Error> {
        let plan = RecipeInferencePlan::new(RecipeInferenceInput::Derived(recipe), catalog)
            .ok_or(RecipeError::InferenceUnavailable)?;
        self.pool
            .initialize_shared_native(plan)
            .map_err(|error| Self::Error::Inference(error.into_parts().1))
    }
    fn mapping(
        &mut self,
        plan: EncodedRecipeMappingPlan<'_>,
    ) -> Result<Self::Mapping, Self::Error> {
        self.pool
            .initialize_shared_native(plan)
            .map_err(|error| Self::Error::Mapping(error.into_parts().1))
    }
    fn ranges(
        &mut self,
        plan: SelectionReadDestinationPlan<'_, '_>,
    ) -> Result<Self::Ranges, Self::Error> {
        self.pool
            .initialize_shared_native(plan)
            .map_err(|error| Self::Error::Ranges(error.into_parts().1))
    }
    fn children(
        &mut self,
        plan: EncodedRecipeChildrenPlan<Self::Mapping>,
    ) -> Result<EncodedRecipeChildren<Self::Mapping, Self::ChildCustody>, Self::Error> {
        let InitializedSharedNative { output, account } = self
            .pool
            .initialize_shared_native(plan)
            .map_err(|error| Self::Error::Children(error.into_parts().1))?;
        Ok(output.with_custody(SharedNativeInitializationCustody(account, None)))
    }
}

#[cfg(test)]
mod tests;
