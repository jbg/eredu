//! Compile byte-preserving row reordering into the existing admitted read spans.
//! Runtime reads use the same final destinations, source validation and detached
//! source/working-storage census as contiguous encoded recipes.
use super::*;
use crate::store::{EncodedRange, encoded_selection_ranges};
use std::ops::Range;

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
fn collect(recipe: &DerivedWeightRecipe, keys: &mut Vec<String>) -> bool {
    match recipe {
        DerivedWeightRecipe::Source { key, .. } => {
            keys.push(key.clone());
            true
        }
        DerivedWeightRecipe::Concatenate { inputs, .. }
        | DerivedWeightRecipe::Stack { inputs, .. } => {
            inputs.iter().all(|input| collect(input, keys))
        }
        DerivedWeightRecipe::Select { input, .. }
        | DerivedWeightRecipe::Reshape { input, .. }
        | DerivedWeightRecipe::View { input, .. }
        | DerivedWeightRecipe::Transpose { input, .. }
        | DerivedWeightRecipe::Cast { input, .. } => collect(input, keys),
        _ => false,
    }
}

struct Mapping {
    ranges: Vec<EncodedRange>,
    length: usize,
}
impl Mapping {
    fn new() -> Self {
        Self {
            ranges: Vec::new(),
            length: 0,
        }
    }
    fn push(&mut self, source: Range<usize>) -> Result<(), RecipeError> {
        let length = source.end.checked_sub(source.start).ok_or_else(overflow)?;
        if length == 0 {
            return Ok(());
        }
        let end = self.length.checked_add(length).ok_or_else(overflow)?;
        if let Some(last) = self
            .ranges
            .last_mut()
            .filter(|last| last.source.end == source.start)
        {
            last.source.end = source.end;
            last.destination.end = end;
        } else {
            self.ranges.push(EncodedRange {
                source,
                destination: self.length..end,
            });
        }
        self.length = end;
        Ok(())
    }
    fn append_slice(&mut self, input: &Self, selected: Range<usize>) -> Result<(), RecipeError> {
        if selected.start > selected.end || selected.end > input.length {
            return Err(overflow());
        }
        let first = input
            .ranges
            .partition_point(|row| row.destination.end <= selected.start);
        for row in &input.ranges[first..] {
            if row.destination.start >= selected.end {
                break;
            }
            let start = row.destination.start.max(selected.start);
            let end = row.destination.end.min(selected.end);
            if start < end {
                let source_start = row
                    .source
                    .start
                    .checked_add(start - row.destination.start)
                    .ok_or_else(overflow)?;
                self.push(
                    source_start..source_start.checked_add(end - start).ok_or_else(overflow)?,
                )?;
            }
        }
        Ok(())
    }
}

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
        let ranges = match encoded_selection_ranges(
            "encoded recipe",
            bits,
            &metadata.shape,
            input.length,
            selection,
            &output.shape,
        ) {
            Ok(ranges) => ranges,
            Err(StoreError::BoundedSelectionUnavailable { .. }) => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut mapped = Mapping::new();
        for range in ranges {
            mapped.append_slice(&input, range)?;
        }
        Ok(Some(mapped))
    }
    fn compile(&mut self, recipe: &DerivedWeightRecipe) -> Result<Option<Mapping>, RecipeError> {
        let output = recipe.infer(self.catalog)?;
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
                let mut mapped = Mapping::new();
                mapped.push(self.source_offset..end)?;
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
                    let metadata = input.infer(self.catalog)?;
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
                let mut mapped = Mapping::new();
                for index in 0..outer {
                    for (child, chunk) in children.iter().zip(chunks.iter().copied()) {
                        let start = index.checked_mul(chunk).ok_or_else(overflow)?;
                        mapped.append_slice(
                            child,
                            start..start.checked_add(chunk).ok_or_else(overflow)?,
                        )?;
                    }
                }
                mapped
            }
            DerivedWeightRecipe::Select { input, selection } => {
                let metadata = input.infer(self.catalog)?;
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
                if input.infer(self.catalog)?.dtype != *dtype {
                    return Ok(None);
                }
                let Some(mapped) = self.compile(input)? else {
                    return Ok(None);
                };
                mapped
            }
            DerivedWeightRecipe::Transpose { input, axes } => {
                let metadata = input.infer(self.catalog)?;
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
) -> Result<Option<EncodedRecipeRead>, RecipeError> {
    let mut keys = Vec::new();
    if !collect(recipe, &mut keys) {
        return Ok(None);
    }
    let Some(batch) = source.prepare_encoded_read(&keys)? else {
        return Ok(None);
    };
    struct Catalog<'a>(BTreeMap<&'a str, &'a TensorMetadata>);
    impl RecipeCatalog for Catalog<'_> {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            self.0
                .get(key)
                .map(|value| (*value).clone())
                .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })
        }
    }
    fn compile<C: RecipeCatalog + ?Sized>(
        recipe: &DerivedWeightRecipe,
        catalog: &C,
        tensors: &[TensorMetadata],
        length: usize,
    ) -> Result<Option<(RecipeMetadata, Mapping)>, RecipeError> {
        // Preserve ordinary left-to-right geometry/error precedence before
        // deciding whether the fully validated recipe is byte-readable.
        let output = recipe.infer(catalog)?;
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
    let compiled = if source.recipe_cache().is_some() {
        compile(recipe, source, batch.tensors(), batch.byte_len())?
    } else {
        let catalog = Catalog(
            batch
                .tensors()
                .iter()
                .map(|value| (value.name.as_str(), value))
                .collect(),
        );
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
