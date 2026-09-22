//! Compile and assemble an already admitted encoded batch for native consumers.
use super::*;
use eredu_checkpoint::{
    recipe::{
        DerivedWeightRecipe, EncodedRecipeRead, EncodedRecipeReadAssemblyError,
        ReadBatchCatalogBuildError, ReadBatchCatalogPlan,
    },
    store::{EncodedProjectionBuildError, PreparedEncodedRead},
};

/// Original read ownership plus its initialization result's account.
type ReadCustody<C> = (C, SharedNativeInitializationCustody);
/// A projection retains the original read and its own constructor custody.
type ProjectionCustody<C> = (ReadCustody<C>, SharedNativeInitializationCustody);
/// Complete read, projection and inferred-output ownership. Each tuple member
/// carries an existing account; assembling it creates no new allowance.
pub type CompiledRecipeCustody<C> = (
    (ProjectionCustody<C>, SharedNativeInitializationCustody),
    SharedNativeInitializationCustody,
);

/// Compilation or assembly failure retaining actual owned diagnostics/prefixes.
/// Catalog scratch and successful temporary mappings retire synchronously.
#[derive(Debug, thiserror::Error)]
pub enum EncodedRecipeReadPreparationError<C: fmt::Debug + 'static> {
    /// Original pool identity or fixed geometry refusal.
    #[error(transparent)]
    Accounting(#[from] WorkingMemoryError),
    /// Fixed recipe or catalog-plan geometry failure.
    #[error(transparent)]
    Recipe(#[from] eredu_checkpoint::recipe::RecipeError),
    /// The borrowed catalog is retired locally; its error keeps the original hold.
    #[error(transparent)]
    Catalog(
        SharedNativeInitializationFailure<
            (),
            ReadBatchCatalogBuildError<SharedNativeInitializationCustody>,
        >,
    ),
    /// Actual recursive constructor error with its retained admission.
    #[error(transparent)]
    Compilation(#[from] EncodedRecipeConstructionError),
    /// Projection planning retains the original read on geometry refusal.
    #[error(transparent)]
    ProjectionPlan(EncodedProjectionBuildError<ReadCustody<C>>),
    /// Projection construction or settlement retains its actual read prefix.
    #[error(transparent)]
    Projection(
        SharedNativeInitializationFailure<
            PreparedEncodedRead<ProjectionCustody<C>>,
            EncodedProjectionBuildError<ProjectionCustody<C>>,
        >,
    ),
    /// Output metadata and projected source lengths disagree; both remain owned.
    #[error(transparent)]
    Assembly(EncodedRecipeReadAssemblyError<CompiledRecipeCustody<C>>),
}

impl<C: fmt::Debug + 'static> InitializedSharedNative<PreparedEncodedRead<C>> {
    /// Compile this exact batch through checkpoint's shared traversal, then move
    /// its projected records and inferred metadata into one recipe-read owner.
    /// All constructors use the original pool; ordinary read preparation is
    /// never used as a fallback. Numerical recipes return `None` and retire the
    /// consumed batch. Source birth, keys and recipe declarations are separate.
    pub fn compile_recipe(
        self,
        recipe: &DerivedWeightRecipe,
        pool: &MemoryLedger,
    ) -> Result<
        Option<EncodedRecipeRead<CompiledRecipeCustody<C>>>,
        EncodedRecipeReadPreparationError<C>,
    > {
        use EncodedRecipeReadPreparationError as E;
        self.validate_pool(pool)?;
        let catalog = pool
            .initialize_shared_native(ReadBatchCatalogPlan::new(self.output.tensors())?)
            .map_err(|error| {
                E::Catalog(
                    error
                        .into_parts()
                        .1
                        .retire_output_and_map_error(|cause| cause),
                )
            })?;
        let compiled = catalog
            .output()
            .compile_recipe(recipe, &mut AdmittedRecipeConstruction::new(pool))?;
        drop(catalog);
        let Some((metadata, mapping)) = compiled else {
            return Ok(None);
        };
        let plan = self
            .into_owned_read()
            .project(mapping.output())
            .map_err(E::ProjectionPlan)?;
        let projected = pool.initialize_shared_native(plan).map_err(|error| {
            let (uncalled, failure) = error.into_parts();
            // A rejected plan owns only synchronously destructible host reads.
            // Retire it locally; any actual failed prefix remains in failure.
            drop(uncalled);
            E::Projection(failure)
        })?;
        drop(mapping);
        let InitializedSharedNative { output, account } = metadata;
        let read = projected
            .into_owned_read()
            .with_custody(SharedNativeInitializationCustody(account, None));
        EncodedRecipeRead::from_prepared(output, read)
            .map(Some)
            .map_err(E::Assembly)
    }
}

#[cfg(test)]
mod tests;
