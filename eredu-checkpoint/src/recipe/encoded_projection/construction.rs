//! Allocation policy for the single encoded-recipe traversal.
use super::*;
use crate::store::{SelectionReadDestinationPlan, SelectionReadRanges};
use std::borrow::Borrow;

/// Constructs the storage used by encoded recipe compilation. Implementations
/// must preserve the supplied plans and infer the recipe against the supplied
/// immutable catalog. Recipe semantics and traversal remain checkpoint-owned.
pub trait EncodedRecipeConstruction {
    /// Typed semantic or construction failure, retaining any owned diagnostics.
    type Error: From<RecipeError>;
    /// Inferred metadata owner, including any caller admission.
    type Metadata: Borrow<RecipeMetadata>;
    /// Byte-coordinate owner, including any caller admission.
    type Mapping: Borrow<EncodedRecipeMapping>;
    /// Selection-coordinate owner, including any caller admission.
    type Ranges: Borrow<SelectionReadRanges>;
    /// Custody retained after the actual child array and its mapping owners.
    type ChildCustody;

    /// Validate the complete recipe using the shared metadata inference worker.
    fn infer<C: RecipeCatalog + ?Sized>(
        &mut self,
        recipe: &DerivedWeightRecipe,
        catalog: &C,
    ) -> Result<Self::Metadata, Self::Error>;
    /// Construct one counted byte mapping.
    fn mapping(&mut self, plan: EncodedRecipeMappingPlan<'_>)
    -> Result<Self::Mapping, Self::Error>;
    /// Construct the validated selection intervals.
    fn ranges(
        &mut self,
        plan: SelectionReadDestinationPlan<'_, '_>,
    ) -> Result<Self::Ranges, Self::Error>;
    /// Reserve the actual child-owner array before recursive construction.
    fn children(
        &mut self,
        plan: EncodedRecipeChildrenPlan<Self::Mapping>,
    ) -> Result<EncodedRecipeChildren<Self::Mapping, Self::ChildCustody>, Self::Error>;
}

pub(super) struct OrdinaryConstruction;
impl EncodedRecipeConstruction for OrdinaryConstruction {
    type Error = RecipeError;
    type Metadata = RecipeMetadata;
    type Mapping = EncodedRecipeMapping;
    type Ranges = SelectionReadRanges;
    type ChildCustody = ();
    fn infer<C: RecipeCatalog + ?Sized>(
        &mut self,
        recipe: &DerivedWeightRecipe,
        catalog: &C,
    ) -> Result<Self::Metadata, Self::Error> {
        infer_read_metadata(recipe, catalog)
    }
    fn mapping(
        &mut self,
        plan: EncodedRecipeMappingPlan<'_>,
    ) -> Result<Self::Mapping, Self::Error> {
        plan.build()
    }
    fn ranges(
        &mut self,
        plan: SelectionReadDestinationPlan<'_, '_>,
    ) -> Result<Self::Ranges, Self::Error> {
        plan.build().map_err(RecipeError::ProjectionReserve)
    }
    fn children(
        &mut self,
        plan: EncodedRecipeChildrenPlan<Self::Mapping>,
    ) -> Result<EncodedRecipeChildren<Self::Mapping, ()>, Self::Error> {
        plan.construct(()).map_err(RecipeError::ProjectionReserve)
    }
}
