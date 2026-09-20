//! Compile byte-preserving row reordering into the existing admitted read spans.
//! Runtime reads use the same final destinations, source validation and detached
//! source/working-storage census as contiguous encoded recipes.
use super::*;
use crate::store::{EncodedRange, encoded_selection_plan};
use std::borrow::Borrow;
mod children;
mod construction;
pub use children::{EncodedRecipeChildren, EncodedRecipeChildrenPlan};
pub use construction::EncodedRecipeConstruction;
use construction::OrdinaryConstruction;

fn overflow() -> RecipeError {
    RecipeError::ArithmeticOverflow("encoded recipe range projection")
}
fn bytes(value: u64) -> Result<usize, RecipeError> {
    usize::try_from(value).map_err(|_| overflow())
}
fn product(shape: &[usize]) -> Result<usize, RecipeError> {
    shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d).ok_or_else(overflow))
}

mod mapping;
use mapping::{Children, EncodedRecipeMappingInput as MappingInput};
pub use mapping::{EncodedRecipeMapping, EncodedRecipeMappingPlan};
use {EncodedRecipeMapping as Mapping, EncodedRecipeMappingPlan as MappingPlan};

struct Compiler<'a, C: ?Sized, P> {
    construction: &'a mut P,
    catalog: &'a C,
    tensors: &'a [TensorMetadata],
    source_index: usize,
    source_offset: usize,
}
impl<C: RecipeCatalog + ?Sized, P: EncodedRecipeConstruction> Compiler<'_, C, P> {
    fn select(
        &mut self,
        input: P::Mapping,
        shape: &[usize],
        bits: usize,
        selection: &TensorSelection,
        output: &RecipeMetadata,
    ) -> Result<Option<P::Mapping>, P::Error> {
        let plan = match encoded_selection_plan(
            "encoded recipe",
            bits,
            shape,
            input.borrow().length,
            selection,
            &output.shape,
        ) {
            Ok(plan) => plan,
            Err(error) if error.is_bounded_unavailable() => return Ok(None),
            Err(error) => {
                return Err(RecipeError::EncodedSelection(error.with_key("encoded recipe")).into());
            }
        };
        let ranges = self.construction.ranges(plan)?;
        let mapped = self.construction.mapping(MappingPlan::selected(
            input.borrow(),
            ranges.borrow().ranges(),
        )?)?;
        Ok(Some(mapped))
    }
    fn compile(&mut self, recipe: &DerivedWeightRecipe) -> Result<Option<P::Mapping>, P::Error> {
        let output_owner = self.construction.infer(recipe, self.catalog)?;
        let output = output_owner.borrow();
        let mapped = match recipe {
            DerivedWeightRecipe::Source { key, selection } => {
                let source = self.tensors.get(self.source_index).ok_or_else(overflow)?;
                if source.name != *key {
                    return Err(overflow().into());
                }
                let length = bytes(source.encoded_byte_len)?;
                let end = self
                    .source_offset
                    .checked_add(length)
                    .ok_or_else(overflow)?;
                let mapped = self
                    .construction
                    .mapping(MappingPlan::source(self.source_offset..end)?)?;
                self.source_index = self.source_index.checked_add(1).ok_or_else(overflow)?;
                self.source_offset = end;
                let bits = bytes(output.dtype.bit_width()?)?;
                let Some(mapped) =
                    self.select(mapped, &source.logical_shape, bits, selection, output)?
                else {
                    return Ok(None);
                };
                mapped
            }
            DerivedWeightRecipe::Concatenate { axis, inputs }
            | DerivedWeightRecipe::Stack { axis, inputs } => {
                let outer = product(&output.shape[..*axis])?;
                let mut children = self
                    .construction
                    .children(EncodedRecipeChildrenPlan::new(inputs.len())?)?;
                for input in inputs {
                    let metadata_owner = self.construction.infer(input, self.catalog)?;
                    let metadata = metadata_owner.borrow();
                    let tail = &metadata.shape[*axis..];
                    let bits = product(tail)?
                        .checked_mul(bytes(metadata.dtype.bit_width()?)?)
                        .ok_or_else(overflow)?;
                    // A row crossing a packed scalar's byte boundary requires
                    // a numerical transform and remains on the ordinary path.
                    if !bits.is_multiple_of(8) {
                        return Ok(None);
                    }
                    let Some(child) = self.compile(input)? else {
                        return Ok(None);
                    };
                    let chunk = bits / 8;
                    if outer.checked_mul(chunk).ok_or_else(overflow)? != child.borrow().length {
                        return Err(overflow().into());
                    }
                    children.push(child, chunk)?;
                }
                self.construction
                    .mapping(MappingPlan::new(MappingInput::Interleaved {
                        children: Children::Constructed(&children),
                        outer,
                    })?)?
            }
            DerivedWeightRecipe::Select { input, selection } => {
                let metadata_owner = self.construction.infer(input, self.catalog)?;
                let metadata = metadata_owner.borrow();
                let Some(input) = self.compile(input)? else {
                    return Ok(None);
                };
                let Some(mapped) = self.select(
                    input,
                    &metadata.shape,
                    bytes(metadata.dtype.bit_width()?)?,
                    selection,
                    output,
                )?
                else {
                    return Ok(None);
                };
                mapped
            }
            DerivedWeightRecipe::Reshape { input, .. }
            | DerivedWeightRecipe::View { input, .. } => {
                let Some(mapped) = self.compile(input)? else {
                    return Ok(None);
                };
                mapped
            }
            DerivedWeightRecipe::Cast { input, dtype } => {
                if self.construction.infer(input, self.catalog)?.borrow().dtype != *dtype {
                    return Ok(None);
                }
                let Some(mapped) = self.compile(input)? else {
                    return Ok(None);
                };
                mapped
            }
            DerivedWeightRecipe::Transpose { input, axes } => {
                let metadata_owner = self.construction.infer(input, self.catalog)?;
                let metadata = metadata_owner.borrow();
                // Ordinary recipe inference already validated the permutation.
                if !metadata.shape.contains(&0)
                    && !axes
                        .iter()
                        .copied()
                        .filter(|axis| metadata.shape[*axis] > 1)
                        .eq((0..metadata.shape.len()).filter(|axis| metadata.shape[*axis] > 1))
                {
                    return Ok(None);
                }
                let Some(mapped) = self.compile(input)? else {
                    return Ok(None);
                };
                mapped
            }
            _ => return Ok(None),
        };
        if mapped.borrow().length != bytes(output.byte_len)? {
            return Ok(None);
        }
        Ok(Some(mapped))
    }
}

pub(super) fn compile<C: RecipeCatalog + ?Sized, P: EncodedRecipeConstruction>(
    recipe: &DerivedWeightRecipe,
    catalog: &C,
    tensors: &[TensorMetadata],
    length: usize,
    construction: &mut P,
) -> Result<Option<(P::Metadata, P::Mapping)>, P::Error> {
    // Validate the whole recipe before deciding whether it is byte-readable.
    let output = construction.infer(recipe, catalog)?;
    let mut compiler = Compiler {
        catalog,
        tensors,
        construction,
        source_index: 0,
        source_offset: 0,
    };
    let Some(mapping) = compiler.compile(recipe)? else {
        return Ok(None);
    };
    if compiler.source_index != tensors.len() || compiler.source_offset != length {
        return Err(overflow().into());
    }
    Ok(Some((output, mapping)))
}

pub(super) fn prepare(
    recipe: &DerivedWeightRecipe,
    source: &dyn CheckpointSource,
    keys: &[String],
    use_source_cache: bool,
) -> Result<Option<EncodedRecipeRead>, RecipeError> {
    let Some(batch) = source.prepare_encoded_read(keys)? else {
        return Ok(None);
    };
    let compiled = if use_source_cache && source.recipe_cache().is_some() {
        compile(
            recipe,
            source,
            batch.tensors(),
            batch.byte_len(),
            &mut OrdinaryConstruction,
        )?
    } else {
        let catalog = ReadBatchCatalogPlan::new(batch.tensors())?.construct(())?;
        catalog.compile_recipe(recipe, &mut OrdinaryConstruction)?
    };
    let Some((output, mapping)) = compiled else {
        return Ok(None);
    };
    let batch = batch.project_ranges(&mapping.ranges, mapping.length)?;
    Ok(Some(EncodedRecipeRead { output, batch, _custody: () }))
}

#[cfg(test)]
mod tests;
