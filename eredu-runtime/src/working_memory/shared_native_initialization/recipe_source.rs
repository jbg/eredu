//! Original key, source-read and recursive compiler composition over retained roots.
use super::*;
use eredu_checkpoint::{
    recipe::{
        DerivedWeightRecipe, EncodedRecipeKeysBuildError, EncodedRecipeKeysPlan, EncodedRecipeRead,
        PreparedEncodedRecipeKeys, RecipeError,
    },
    store::{
        MemoryEncodedReadBuildError, MemoryEncodedReadPlan, MemoryEncodedReadRouteError,
        PreparedEncodedRead, RetainedCheckpointSource, SafetensorsEncodedReadBuildError,
        SafetensorsEncodedReadPlan, SafetensorsEncodedReadPlanError,
    },
};

/// An owned source/compiler failure. File inspection diagnostics keep
/// their actual source owners; initialized prefixes keep original pool custody.
#[derive(Debug, thiserror::Error)]
pub enum EncodedRecipeSourceError {
    /// Fixed key-plan geometry failure.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
    /// Key admission or construction, including any completed string prefix.
    #[error(transparent)]
    Keys(
        SharedNativeInitializationFailure<
            PreparedEncodedRecipeKeys<SharedNativeInitializationCustody>,
            EncodedRecipeKeysBuildError<SharedNativeInitializationCustody>,
        >,
    ),
    /// Closed memory routing/geometry refusal.
    #[error(transparent)]
    MemoryRoute(#[from] MemoryEncodedReadRouteError),
    /// Closed file routing/geometry refusal retaining source diagnostics.
    #[error(transparent)]
    FileRoute(#[from] SafetensorsEncodedReadPlanError),
    /// Actual memory read construction or settlement failure.
    #[error(transparent)]
    Memory(
        SharedNativeInitializationFailure<
            PreparedEncodedRead<SharedNativeInitializationCustody>,
            MemoryEncodedReadBuildError<SharedNativeInitializationCustody>,
        >,
    ),
    /// Actual file read construction or settlement failure.
    #[error(transparent)]
    File(
        SharedNativeInitializationFailure<
            PreparedEncodedRead<SharedNativeInitializationCustody>,
            SafetensorsEncodedReadBuildError<SharedNativeInitializationCustody>,
        >,
    ),
    /// Recursive compilation and projected-read construction.
    #[error(transparent)]
    Compilation(#[from] EncodedRecipeReadPreparationError<SharedNativeInitializationCustody>),
}

impl MemoryLedger {
    /// Prepare encoded recipe metadata and projected source records under this
    /// pool before native input construction. The retained source supplies its
    /// exact closed route; no ordinary read or lease callback is used as fallback.
    /// Unsupported source/recipe routes return `None`; admission/inspection errors
    /// propagate with their owners. Source/header birth and recipe storage remain
    /// separately admitted prerequisites. No payload is read here.
    pub fn prepare_encoded_recipe(
        &self,
        source: &RetainedCheckpointSource,
        recipe: &DerivedWeightRecipe,
    ) -> Result<
        Option<EncodedRecipeRead<CompiledRecipeCustody<SharedNativeInitializationCustody>>>,
        EncodedRecipeSourceError,
    > {
        use EncodedRecipeSourceError as E;
        let Some(plan) = EncodedRecipeKeysPlan::new(recipe)? else {
            return Ok(None);
        };
        let keys = self
            .initialize_shared_native(plan)
            .map_err(|error| E::Keys(error.into_parts().1))?;
        let read =
            if let Some(plan) = MemoryEncodedReadPlan::from_source(source, keys.output().keys())? {
                self.initialize_shared_native(plan)
                    .map_err(|error| E::Memory(error.into_parts().1))?
            } else if let Some(plan) =
                SafetensorsEncodedReadPlan::from_source(source, keys.output().keys())?
            {
                self.initialize_shared_native(plan)
                    .map_err(|error| E::File(error.into_parts().1))?
            } else {
                return Ok(None);
            };
        drop(keys);
        read.compile_recipe(recipe, self).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests;
