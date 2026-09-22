//! Actual owned declaration cloning under the constructor's metadata account.
use super::*;
use eredu_checkpoint::recipe::{RecipeDtype, RecipeInferenceInput, RecipeInferencePlan};
use std::mem::{size_of, size_of_val};

pub(super) fn infer(
    recipe: &DerivedWeightRecipe,
    source: &dyn CheckpointSource,
    context: &WorkspaceContext,
) -> Result<RecipeMetadata, WeightRecipeError> {
    let plan = RecipeInferencePlan::new(RecipeInferenceInput::Derived(recipe), source)
        .ok_or_else(|| WeightRecipeError::Workspace(WorkspaceMetadataError::Unqualified.into()))?;
    context
        .charge_metadata(plan.layout().required_bytes())
        .map_err(|cause| WeightRecipeError::Workspace(cause.into()))?;
    plan.infer()
        .map_err(|cause| WeightRecipeError::Workspace(context.metadata_source(cause)))
}

pub(super) fn clone_recipe(
    recipe: &DerivedWeightRecipe,
    context: &WorkspaceContext,
) -> Result<DerivedWeightRecipe, WeightRecipeError> {
    use DerivedWeightRecipe as R;
    let frames = [
        size_of::<R>(),
        size_of::<Result<R, WeightRecipeError>>(),
        size_of::<(&R, &WorkspaceContext)>(),
        size_of::<std::slice::Iter<'_, R>>(),
    ];
    context
        .charge_metadata(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or_else(|| {
                    WeightRecipeError::Workspace(WorkspaceMetadataError::Overflow.into())
                })?,
        )
        .map_err(|cause| WeightRecipeError::Workspace(cause.into()))?;
    Ok(match recipe {
        R::Source { key, selection } => R::Source {
            key: context.metadata_string(format_args!("{key}"))?,
            selection: clone_selection(selection, context)?,
        },
        R::Select { input, selection } => R::Select {
            input: child(input, context)?,
            selection: clone_selection(selection, context)?,
        },
        R::Concatenate { inputs, axis } => R::Concatenate {
            inputs: children(inputs, context)?,
            axis: *axis,
        },
        R::Stack { inputs, axis } => R::Stack {
            inputs: children(inputs, context)?,
            axis: *axis,
        },
        R::Reshape { input, shape } => R::Reshape {
            input: child(input, context)?,
            shape: indices(shape, context)?,
        },
        R::Transpose { input, axes } => R::Transpose {
            input: child(input, context)?,
            axes: indices(axes, context)?,
        },
        R::Cast {
            input,
            dtype: value,
        } => R::Cast {
            input: child(input, context)?,
            dtype: dtype(value, context)?,
        },
        R::View {
            input,
            dtype: value,
            shape,
        } => R::View {
            input: child(input, context)?,
            dtype: dtype(value, context)?,
            shape: indices(shape, context)?,
        },
        R::NegLog { input } => R::NegLog {
            input: child(input, context)?,
        },
        R::SubtractOne { input } => R::SubtractOne {
            input: child(input, context)?,
        },
    })
}
fn child(
    recipe: &DerivedWeightRecipe,
    context: &WorkspaceContext,
) -> Result<Box<DerivedWeightRecipe>, WeightRecipeError> {
    context
        .charge_metadata(size_of::<DerivedWeightRecipe>())
        .map_err(|cause| WeightRecipeError::Workspace(cause.into()))?;
    Ok(Box::new(clone_recipe(recipe, context)?))
}
fn children(
    recipes: &[DerivedWeightRecipe],
    context: &WorkspaceContext,
) -> Result<Vec<DerivedWeightRecipe>, WeightRecipeError> {
    let mut values = context.metadata_vec(recipes.len())?;
    for recipe in recipes {
        values.push(clone_recipe(recipe, context)?);
    }
    Ok(values)
}
fn indices(source: &[usize], context: &WorkspaceContext) -> Result<Vec<usize>, WeightRecipeError> {
    let mut values = context.metadata_vec(source.len())?;
    values.extend_from_slice(source);
    Ok(values)
}
fn dtype(
    source: &RecipeDtype,
    context: &WorkspaceContext,
) -> Result<RecipeDtype, WeightRecipeError> {
    Ok(match source {
        RecipeDtype::Other(value) => {
            RecipeDtype::Other(context.metadata_string(format_args!("{value}"))?)
        }
        value => value.clone(),
    })
}
