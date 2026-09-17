//! Source-borrowed, uncached recipe inference with finite constructor storage.
use super::*;
use std::{alloc::Layout, collections::TryReserveError, mem::size_of};

/// Exact borrowed recipe declaration, including direct bindings without a temporary recipe clone.
#[derive(Clone, Copy, Debug)]
pub enum RecipeInferenceInput<'a> {
    /// Existing complete derived recipe.
    Derived(&'a DerivedWeightRecipe),
    /// Existing direct source name and selection.
    Source {
        key: &'a str,
        selection: &'a TensorSelection,
    },
}
/// Source-derived finite inference allocations and fixed controls.
#[derive(Clone, Copy, Debug)]
pub struct RecipeInferenceLayout {
    nodes: usize,
    rank: usize,
    bytes: usize,
}
impl RecipeInferenceLayout {
    /// Requested storage, including possible typed error-owned arrays/text.
    pub const fn required_bytes(self) -> usize {
        self.bytes
    }
    /// Inspect only borrowed metadata. Missing borrowed catalog support stays unknown.
    pub fn inspect<C: RecipeCatalog + ?Sized>(
        input: RecipeInferenceInput<'_>,
        catalog: &C,
    ) -> Option<Self> {
        let mut population = Population::default();
        population.visit(input, catalog)?;
        let rank = population.rank.checked_add(population.stacks)?;
        let actions = population.nodes.checked_mul(2)?;
        let mut bytes = Layout::array::<Action<'static>>(actions)
            .ok()?
            .size()
            .checked_add(
                Layout::array::<RecipeMetadata>(population.nodes)
                    .ok()?
                    .size(),
            )?
            .checked_add(Layout::array::<usize>(population.nodes).ok()?.size())?;
        bytes = bytes
            .checked_add(
                Layout::array::<usize>(rank)
                    .ok()?
                    .size()
                    .checked_mul(population.nodes)?,
            )?
            .checked_add(
                population
                    .dtype_bytes
                    .checked_mul(population.nodes.checked_add(1)?)?,
            )?
            .checked_add(Layout::array::<usize>(population.axes).ok()?.size())?;
        for n in [
            size_of::<Self>(),
            size_of::<Population>(),
            size_of::<Worker<'static>>(),
            size_of::<RecipeMetadata>(),
            size_of::<RecipeError>(),
            size_of::<RecipeInferenceError>(),
            size_of::<Action<'static>>(),
            size_of::<Option<Action<'static>>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<RecipeInferenceInput<'static>>(),
        ] {
            bytes = bytes.checked_add(n)?;
        }
        // Named input/iterator transports for the source walk, bounded by the
        // actual tree population. This is not the compiler's machine stack-frame
        // size or a whole-thread stack guarantee. Cold planning borrows its
        // caller's storage and receives no retrospective constructor authority;
        // the funded inference worker below uses explicit action destinations.
        bytes = bytes.checked_add(population.nodes.checked_mul(
            size_of::<RecipeInferenceInput<'static>>()
                + size_of::<std::slice::Iter<'static, DerivedWeightRecipe>>(),
        )?)?;
        Some(Self {
            nodes: population.nodes,
            rank,
            bytes,
        })
    }
}
#[derive(Default)]
struct Population {
    nodes: usize,
    rank: usize,
    stacks: usize,
    axes: usize,
    dtype_bytes: usize,
}
impl Population {
    fn visit<C: RecipeCatalog + ?Sized>(
        &mut self,
        input: RecipeInferenceInput<'_>,
        catalog: &C,
    ) -> Option<()> {
        self.nodes = self.nodes.checked_add(1)?;
        match input {
            RecipeInferenceInput::Source { key, selection } => {
                let metadata = catalog.tensor_metadata_borrowed(key)?;
                self.rank = self.rank.max(metadata.logical_shape.len());
                if let StoredDtype::Other(name) = &metadata.stored_dtype {
                    self.dtype_bytes = self.dtype_bytes.max(name.len());
                }
                self.selection(selection);
            }
            RecipeInferenceInput::Derived(recipe) => match recipe {
                DerivedWeightRecipe::Source { key, selection } => {
                    let metadata = catalog.tensor_metadata_borrowed(key)?;
                    self.rank = self.rank.max(metadata.logical_shape.len());
                    if let StoredDtype::Other(name) = &metadata.stored_dtype {
                        self.dtype_bytes = self.dtype_bytes.max(name.len());
                    }
                    self.selection(selection);
                }
                DerivedWeightRecipe::Concatenate { inputs, .. }
                | DerivedWeightRecipe::Stack { inputs, .. } => {
                    if matches!(recipe, DerivedWeightRecipe::Stack { .. }) {
                        self.stacks = self.stacks.checked_add(1)?;
                    }
                    for child in inputs {
                        self.visit(RecipeInferenceInput::Derived(child), catalog)?;
                    }
                }
                DerivedWeightRecipe::Select { input, selection } => {
                    self.selection(selection);
                    self.visit(RecipeInferenceInput::Derived(input), catalog)?;
                }
                DerivedWeightRecipe::Reshape { input, shape } => {
                    self.rank = self.rank.max(shape.len());
                    self.visit(RecipeInferenceInput::Derived(input), catalog)?;
                }
                DerivedWeightRecipe::Transpose { input, axes } => {
                    self.axes = self.axes.max(axes.len());
                    self.visit(RecipeInferenceInput::Derived(input), catalog)?;
                }
                DerivedWeightRecipe::Cast { input, dtype } => {
                    self.dtype(dtype);
                    self.visit(RecipeInferenceInput::Derived(input), catalog)?;
                }
                DerivedWeightRecipe::View {
                    input,
                    dtype,
                    shape,
                } => {
                    self.rank = self.rank.max(shape.len());
                    self.dtype(dtype);
                    self.visit(RecipeInferenceInput::Derived(input), catalog)?;
                }
                DerivedWeightRecipe::NegLog { input }
                | DerivedWeightRecipe::SubtractOne { input } => {
                    self.visit(RecipeInferenceInput::Derived(input), catalog)?
                }
            },
        }
        Some(())
    }
    fn selection(&mut self, selection: &TensorSelection) {
        if let TensorSelection::Contiguous { shape, .. } = selection {
            self.rank = self.rank.max(shape.len());
        }
    }
    fn dtype(&mut self, dtype: &RecipeDtype) {
        if let RecipeDtype::Other(name) = dtype {
            self.dtype_bytes = self.dtype_bytes.max(name.len());
        }
    }
}
/// Actual finite-constructor or ordinary recipe validation failure.
#[derive(Debug, thiserror::Error)]
pub enum RecipeInferenceError {
    /// Catalog cannot lend its retained metadata, or source geometry changed.
    #[error("finite recipe metadata construction is unavailable")]
    Unavailable,
    /// Exact finite destination reserve failed.
    #[error("finite recipe inference reserve failed")]
    Reserve(#[source] TryReserveError),
    /// Same recipe validation failure as ordinary inference.
    #[error(transparent)]
    Recipe(#[from] RecipeError),
}
#[derive(Clone, Copy)]
enum Action<'a> {
    Visit(RecipeInferenceInput<'a>),
    Apply(&'a DerivedWeightRecipe, usize),
}
struct Worker<'a> {
    actions: Vec<Action<'a>>,
    values: Vec<RecipeMetadata>,
    roots: Vec<usize>,
    layout: RecipeInferenceLayout,
}
impl Worker<'_> {
    fn shape(&self) -> Result<Vec<usize>, RecipeInferenceError> {
        let mut shape = Vec::new();
        shape
            .try_reserve_exact(self.layout.rank)
            .map_err(RecipeInferenceError::Reserve)?;
        Ok(shape)
    }
    fn append(&mut self, metadata: RecipeMetadata) -> Result<(), RecipeInferenceError> {
        if self.values.len() == self.layout.nodes || self.roots.len() == self.layout.nodes {
            return Err(RecipeInferenceError::Unavailable);
        }
        self.roots.push(self.values.len());
        self.values.push(metadata);
        Ok(())
    }
    fn source<C: RecipeCatalog + ?Sized>(
        &mut self,
        key: &str,
        selection: &TensorSelection,
        catalog: &C,
    ) -> Result<(), RecipeInferenceError> {
        if key.trim().is_empty() {
            return Err(RecipeError::EmptySourceKey.into());
        }
        let metadata = catalog
            .tensor_metadata_borrowed(key)
            .ok_or(RecipeInferenceError::Unavailable)?;
        let selected = SelectedRecipeShape::new(&metadata.logical_shape, selection)?;
        let mut shape = self.shape()?;
        if selected.iter().len() > self.layout.rank {
            return Err(RecipeInferenceError::Unavailable);
        }
        shape.extend(selected.iter());
        self.append(metadata_for(shape, metadata.stored_dtype.clone().into())?)
    }
    fn apply(
        &mut self,
        recipe: &DerivedWeightRecipe,
        count: usize,
    ) -> Result<(), RecipeInferenceError> {
        let start = self
            .roots
            .len()
            .checked_sub(count)
            .ok_or(RecipeInferenceError::Unavailable)?;
        let roots = &self.roots[start..];
        let first = roots.first().map(|&i| &self.values[i]);
        if let DerivedWeightRecipe::Transpose { axes, .. } = recipe {
            validate_permutation(
                axes,
                first.ok_or(RecipeInferenceError::Unavailable)?.shape.len(),
            )?;
        }
        let output_rank = match recipe {
            DerivedWeightRecipe::Concatenate { .. } => first.map_or(0, |v| v.shape.len()),
            DerivedWeightRecipe::Stack { .. } => first
                .map_or(0, |v| v.shape.len())
                .checked_add(1)
                .ok_or(RecipeInferenceError::Unavailable)?,
            DerivedWeightRecipe::Select { selection, .. } => match selection {
                TensorSelection::Contiguous { shape, .. } => shape.len(),
                _ => first.map_or(0, |v| v.shape.len()),
            },
            DerivedWeightRecipe::Reshape { shape, .. }
            | DerivedWeightRecipe::View { shape, .. } => shape.len(),
            DerivedWeightRecipe::Transpose { axes, .. } => axes.len(),
            _ => first.map_or(0, |v| v.shape.len()),
        };
        if output_rank > self.layout.rank {
            return Err(RecipeInferenceError::Unavailable);
        }
        let mut shape = self.shape()?;
        let dtype = match recipe {
            DerivedWeightRecipe::Concatenate { axis, .. }
            | DerivedWeightRecipe::Stack { axis, .. } => {
                let stack = matches!(recipe, DerivedWeightRecipe::Stack { .. });
                fill_join_shape(
                    *axis,
                    roots.iter().map(|&i| &self.values[i]),
                    stack,
                    &mut shape,
                )?;
                first.ok_or(RecipeError::EmptyInputs)?.dtype.clone()
            }
            DerivedWeightRecipe::Select { selection, .. } => {
                let first = first.ok_or(RecipeInferenceError::Unavailable)?;
                shape.extend(SelectedRecipeShape::new(&first.shape, selection)?.iter());
                first.dtype.clone()
            }
            DerivedWeightRecipe::Reshape { shape: target, .. } => {
                let first = first.ok_or(RecipeInferenceError::Unavailable)?;
                validate_reshape(&first.shape, target)?;
                shape.extend_from_slice(target);
                first.dtype.clone()
            }
            DerivedWeightRecipe::Transpose { axes, .. } => {
                let first = first.ok_or(RecipeInferenceError::Unavailable)?;
                validate_permutation(axes, first.shape.len())?;
                shape.extend(axes.iter().map(|&axis| first.shape[axis]));
                first.dtype.clone()
            }
            DerivedWeightRecipe::Cast { dtype, .. } => {
                shape.extend_from_slice(&first.ok_or(RecipeInferenceError::Unavailable)?.shape);
                dtype.clone()
            }
            DerivedWeightRecipe::View {
                dtype,
                shape: target,
                ..
            } => {
                shape.extend_from_slice(target);
                dtype.clone()
            }
            DerivedWeightRecipe::NegLog { .. } | DerivedWeightRecipe::SubtractOne { .. } => {
                let first = first.ok_or(RecipeInferenceError::Unavailable)?;
                shape.extend_from_slice(&first.shape);
                first.dtype.clone()
            }
            DerivedWeightRecipe::Source { .. } => return Err(RecipeInferenceError::Unavailable),
        };
        let metadata = metadata_for(shape, dtype)?;
        if matches!(recipe, DerivedWeightRecipe::View { .. }) {
            let input = first.ok_or(RecipeInferenceError::Unavailable)?.byte_len;
            if input != metadata.byte_len {
                return Err(RecipeError::ByteCountMismatch {
                    input,
                    output: metadata.byte_len,
                }
                .into());
            }
        }
        self.roots.truncate(start);
        self.append(metadata)
    }
}
/// Executes the actual uncached inference once, without constructing a recipe or
/// consulting/inserting an inference cache. All temporary owners retire before return.
pub fn infer_recipe_bytes<C: RecipeCatalog + ?Sized>(
    input: RecipeInferenceInput<'_>,
    catalog: &C,
) -> Result<u64, RecipeInferenceError> {
    let layout =
        RecipeInferenceLayout::inspect(input, catalog).ok_or(RecipeInferenceError::Unavailable)?;
    let mut worker = Worker {
        actions: Vec::new(),
        values: Vec::new(),
        roots: Vec::new(),
        layout,
    };
    worker
        .actions
        .try_reserve_exact(
            layout
                .nodes
                .checked_mul(2)
                .ok_or(RecipeInferenceError::Unavailable)?,
        )
        .map_err(RecipeInferenceError::Reserve)?;
    worker
        .values
        .try_reserve_exact(layout.nodes)
        .map_err(RecipeInferenceError::Reserve)?;
    worker
        .roots
        .try_reserve_exact(layout.nodes)
        .map_err(RecipeInferenceError::Reserve)?;
    worker.actions.push(Action::Visit(input));
    while let Some(action) = worker.actions.pop() {
        match action {
            Action::Visit(RecipeInferenceInput::Source { key, selection }) => {
                worker.source(key, selection, catalog)?
            }
            Action::Visit(RecipeInferenceInput::Derived(recipe)) => match recipe {
                DerivedWeightRecipe::Source { key, selection } => {
                    worker.source(key, selection, catalog)?
                }
                DerivedWeightRecipe::Concatenate { inputs, .. }
                | DerivedWeightRecipe::Stack { inputs, .. } => {
                    worker.actions.push(Action::Apply(recipe, inputs.len()));
                    for child in inputs.iter().rev() {
                        worker
                            .actions
                            .push(Action::Visit(RecipeInferenceInput::Derived(child)));
                    }
                }
                DerivedWeightRecipe::Select { input, .. }
                | DerivedWeightRecipe::Reshape { input, .. }
                | DerivedWeightRecipe::Transpose { input, .. }
                | DerivedWeightRecipe::Cast { input, .. }
                | DerivedWeightRecipe::View { input, .. }
                | DerivedWeightRecipe::NegLog { input }
                | DerivedWeightRecipe::SubtractOne { input } => {
                    worker.actions.push(Action::Apply(recipe, 1));
                    worker
                        .actions
                        .push(Action::Visit(RecipeInferenceInput::Derived(input)));
                }
            },
            Action::Apply(recipe, count) => worker.apply(recipe, count)?,
        }
    }
    if worker.roots.len() != 1 {
        return Err(RecipeInferenceError::Unavailable);
    }
    Ok(worker.values[worker.roots[0]].byte_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Catalog(TensorMetadata);
    impl RecipeCatalog for Catalog {
        fn tensor_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            Ok(self.0.clone())
        }
        fn tensor_metadata_borrowed(&self, _: &str) -> Option<&TensorMetadata> {
            Some(&self.0)
        }
    }
    #[test]
    fn finite_inference_preserves_join_selection_and_invalid_permutation_results() {
        let catalog = Catalog(TensorMetadata {
            name: "w".into(),
            logical_shape: vec![2, 3],
            physical_shape: vec![2, 3],
            stored_dtype: StoredDtype::F32,
            encoded_byte_len: 24,
            backing_shard: None,
        });
        let source = || DerivedWeightRecipe::source("w", TensorSelection::Full);
        let joined = DerivedWeightRecipe::Stack {
            axis: 1,
            inputs: vec![source(), source()],
        };
        let selected = DerivedWeightRecipe::Select {
            input: Box::new(joined),
            selection: TensorSelection::Range {
                axis: 2,
                start: 1,
                end: 3,
            },
        };
        let recipe = DerivedWeightRecipe::View {
            input: Box::new(selected),
            dtype: RecipeDtype::U8,
            shape: vec![32],
        };
        assert_eq!(
            infer_recipe_bytes(RecipeInferenceInput::Derived(&recipe), &catalog).unwrap(),
            recipe.infer(&catalog).unwrap().byte_len()
        );
        let invalid = DerivedWeightRecipe::Transpose {
            input: Box::new(source()),
            axes: vec![0, 0],
        };
        assert!(matches!(
            invalid.infer(&catalog),
            Err(RecipeError::InvalidPermutation { rank: 2, .. })
        ));
        assert!(matches!(
            infer_recipe_bytes(RecipeInferenceInput::Derived(&invalid), &catalog),
            Err(RecipeInferenceError::Recipe(
                RecipeError::InvalidPermutation { rank: 2, .. }
            ))
        ));
        assert_eq!(
            infer_recipe_bytes(
                RecipeInferenceInput::Source {
                    key: "w",
                    selection: &TensorSelection::Range {
                        axis: 1,
                        start: 1,
                        end: 3
                    }
                },
                &catalog
            )
            .unwrap(),
            16
        );
    }
}
