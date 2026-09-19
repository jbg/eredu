//! Metadata-only source resolution for selected format transformations.

use super::*;
use eredu_checkpoint::recipe::RecipeMetadata;

/// A selected transform's source recipe or rank-local placement is invalid.
#[derive(Debug, thiserror::Error)]
pub enum TransformSourceError {
    /// The selected task disagrees with its source or admitted derived output.
    #[error("{details}")]
    Task {
        /// The exact task or source mismatch.
        details: String,
    },
    /// The retained placement cannot select the resolved source geometry.
    #[error(transparent)]
    Placement(#[from] crate::BindingPlacementError),
    /// Source recipe inference or bounded selection failed.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
}

/// Resolves the exact floating source of an architecture-selected transform.
///
/// The complete derived output is checked before applying the retained rank's
/// placements, in their declared order. Native construction can compare the
/// returned metadata with its source slots; cold planning needs no native
/// tensor to perform the same recipe and placement checks. This operation
/// allocates planning metadata but neither reads payloads nor grants admission.
pub fn resolve_replicated_text_transform_source(
    source: &dyn CheckpointSource,
    task: &ReplicatedTextMaterializationTask,
    local_layout: Option<&crate::LocalModelLayout>,
) -> Result<(DerivedWeightRecipe, RecipeMetadata), TransformSourceError> {
    let invalid = |details| TransformSourceError::Task { details };
    if !matches!(
        task.lowering(),
        WeightLoweringKind::Transform | WeightLoweringKind::DerivedTransform
    ) {
        return Err(invalid(format!(
            "selected materialization task {:?} did not select a transform lowering",
            task.name()
        )));
    }
    let mut recipe = task
        .source_recipe()
        .map_err(|error| invalid(error.to_string()))?;
    let mut metadata = recipe.infer(source)?;
    if task
        .derived_output()
        .is_some_and(|expected| expected != &metadata)
    {
        return Err(invalid(format!(
            "selected materialization task {:?} differs from its admitted derived output",
            task.name()
        )));
    }
    if let Some(layout) = local_layout {
        let tensor = layout.tensor(task.name()).ok_or_else(|| {
            invalid(format!(
                "selected materialization task {:?} has no source local placement",
                task.name()
            ))
        })?;
        for placement in tensor
            .additional_placements()
            .iter()
            .chain(std::iter::once(tensor.placement()))
        {
            let selection = crate::placement_selection(tensor, placement, metadata.shape())?;
            if selection != TensorSelection::Full {
                recipe = recipe.select_bounded(source, selection)?;
                metadata = recipe.infer(source)?;
            }
        }
    }
    if !matches!(
        metadata.dtype(),
        RecipeDtype::F16 | RecipeDtype::BF16 | RecipeDtype::F32
    ) || metadata.shape().len() < 2
    {
        return Err(invalid(format!(
            "selected materialization task {:?} does not resolve to a floating matrix",
            task.name()
        )));
    }
    Ok((recipe, metadata))
}
