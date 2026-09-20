//! Compile byte-preserving row reordering into the existing admitted read spans.
//! Runtime reads use the same final destinations, source validation and detached
//! source/working-storage census as contiguous encoded recipes.
use super::*;
use crate::store::{EncodedRange, encoded_selection_plan};

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
use mapping::{Mapping, MappingInput, MappingPlan};

struct Compiler<'a, C: ?Sized> {
    catalog: &'a C,
    tensors: &'a [TensorMetadata],
    source_index: usize,
    source_offset: usize,
}
impl<C: RecipeCatalog + ?Sized> Compiler<'_, C> {
    fn select(
        &self,
        input: Mapping,
        metadata: &RecipeMetadata,
        selection: &TensorSelection,
        output: &RecipeMetadata,
    ) -> Result<Option<Mapping>, RecipeError> {
        let bits = bytes(metadata.dtype.bit_width()?)?;
        let plan = match encoded_selection_plan(
            "encoded recipe",
            bits,
            &metadata.shape,
            input.length,
            selection,
            &output.shape,
        ) {
            Ok(plan) => plan,
            Err(error) => match StoreError::from(error) {
                StoreError::BoundedSelectionUnavailable { .. } => return Ok(None),
                error => return Err(error.into()),
            },
        };
        let mut ranges = vec![0..0; plan.range_count()];
        plan.fill_into(&mut ranges).map_err(StoreError::from)?;
        let mapped = MappingPlan::new(MappingInput::Selected {
            input: &input,
            ranges: &ranges,
        })?
        .build()?;
        Ok(Some(mapped))
    }
    fn compile(&mut self, recipe: &DerivedWeightRecipe) -> Result<Option<Mapping>, RecipeError> {
        let output = infer_read_metadata(recipe, self.catalog)?;
        let mapped = match recipe {
            DerivedWeightRecipe::Source { key, selection } => {
                let source = self.tensors.get(self.source_index).ok_or_else(overflow)?;
                if source.name != *key {
                    return Err(overflow());
                }
                let length = bytes(source.encoded_byte_len)?;
                let end = self
                    .source_offset
                    .checked_add(length)
                    .ok_or_else(overflow)?;
                let mapped =
                    MappingPlan::new(MappingInput::Source(self.source_offset..end))?.build()?;
                self.source_index = self.source_index.checked_add(1).ok_or_else(overflow)?;
                self.source_offset = end;
                let metadata = RecipeMetadata {
                    shape: source.logical_shape.clone(),
                    dtype: source.stored_dtype.clone().into(),
                    byte_len: source.encoded_byte_len,
                };
                let Some(mapped) = self.select(mapped, &metadata, selection, &output)? else {
                    return Ok(None);
                };
                mapped
            }
            DerivedWeightRecipe::Concatenate { axis, inputs }
            | DerivedWeightRecipe::Stack { axis, inputs } => {
                let outer = product(&output.shape[..*axis])?;
                let mut children = Vec::with_capacity(inputs.len());
                let mut chunks = Vec::with_capacity(inputs.len());
                for input in inputs {
                    let metadata = infer_read_metadata(input, self.catalog)?;
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
                    if outer.checked_mul(chunk).ok_or_else(overflow)? != child.length {
                        return Err(overflow());
                    }
                    chunks.push(chunk);
                    children.push(child);
                }
                MappingPlan::new(MappingInput::Interleaved {
                    children: &children,
                    chunks: &chunks,
                    outer,
                })?
                .build()?
            }
            DerivedWeightRecipe::Select { input, selection } => {
                let metadata = infer_read_metadata(input, self.catalog)?;
                let Some(input) = self.compile(input)? else {
                    return Ok(None);
                };
                let Some(mapped) = self.select(input, &metadata, selection, &output)? else {
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
                if infer_read_metadata(input, self.catalog)?.dtype != *dtype {
                    return Ok(None);
                }
                let Some(mapped) = self.compile(input)? else {
                    return Ok(None);
                };
                mapped
            }
            DerivedWeightRecipe::Transpose { input, axes } => {
                let metadata = infer_read_metadata(input, self.catalog)?;
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
        if mapped.length != bytes(output.byte_len)? {
            return Ok(None);
        }
        Ok(Some(mapped))
    }
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
    fn compile<C: RecipeCatalog + ?Sized>(
        recipe: &DerivedWeightRecipe,
        catalog: &C,
        tensors: &[TensorMetadata],
        length: usize,
    ) -> Result<Option<(RecipeMetadata, Mapping)>, RecipeError> {
        // Preserve ordinary left-to-right geometry/error precedence before
        // deciding whether the fully validated recipe is byte-readable.
        let output = infer_read_metadata(recipe, catalog)?;
        let mut compiler = Compiler {
            catalog,
            tensors,
            source_index: 0,
            source_offset: 0,
        };
        let Some(mapping) = compiler.compile(recipe)? else {
            return Ok(None);
        };
        if compiler.source_index != tensors.len() || compiler.source_offset != length {
            return Err(overflow());
        }
        Ok(Some((output, mapping)))
    }
    let compiled = if use_source_cache && source.recipe_cache().is_some() {
        compile(recipe, source, batch.tensors(), batch.byte_len())?
    } else {
        let catalog = ReadBatchCatalogPlan::new(batch.tensors())?.construct(())?;
        compile(recipe, &catalog, batch.tensors(), batch.byte_len())?
    };
    let Some((output, mapping)) = compiled else {
        return Ok(None);
    };
    let batch = batch.project_ranges(&mapping.ranges, mapping.length)?;
    Ok(Some(EncodedRecipeRead { output, batch }))
}

#[cfg(test)]
mod tests;
