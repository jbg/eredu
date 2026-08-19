//! Backend-neutral derived-weight recipes and shape inference.

use std::collections::BTreeSet;

use crate::store::{CheckpointSource, StoreError, TensorMetadata, TensorSelection, WeightStore};
use crate::StoredDtype;

/// Metadata-only catalog used to validate a derived-weight recipe.
pub trait RecipeCatalog {
    /// Returns source tensor metadata without reading its payload.
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError>;
}

/// Cold-path capability for proving that every recipe source can be read with
/// its declared physical bound.
pub trait BoundedRecipeSource: RecipeCatalog {
    /// Acquires and immediately releases one source under bounded-read policy.
    fn verify_bounded_source(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<(), StoreError>;
}

impl<T: WeightStore> BoundedRecipeSource for T {
    fn verify_bounded_source(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<(), StoreError> {
        drop(self.acquire(crate::store::TensorReadRequest {
            key: key.to_owned(),
            selection,
            policy: crate::store::ReadPolicy::RequireBounded,
        })?);
        Ok(())
    }
}

impl<T: WeightStore> RecipeCatalog for T {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.metadata(key)
    }
}

impl RecipeCatalog for dyn CheckpointSource + '_ {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.source_metadata(key)
    }
}

impl BoundedRecipeSource for dyn CheckpointSource + '_ {
    fn verify_bounded_source(
        &self,
        key: &str,
        selection: TensorSelection,
    ) -> Result<(), StoreError> {
        drop(self.acquire_lease(crate::store::TensorReadRequest {
            key: key.to_owned(),
            selection,
            policy: crate::store::ReadPolicy::RequireBounded,
        })?);
        Ok(())
    }
}

/// Scalar representation produced by a recipe operation.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
#[allow(missing_docs)]
pub enum RecipeDtype {
    Bool,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
    C64,
    F8E4M3,
    F8E5M2,
    F4,
    F8E8M0,
    Other(String),
}

impl RecipeDtype {
    /// Returns the exact scalar representation width in bits.
    pub fn bit_width(&self) -> Result<u64, RecipeError> {
        match self {
            Self::F4 => Ok(4),
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 => Ok(8),
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => Ok(16),
            Self::I32 | Self::U32 | Self::F32 => Ok(32),
            Self::F64 | Self::I64 | Self::U64 | Self::C64 => Ok(64),
            Self::Other(dtype) => Err(RecipeError::UnsupportedDtype {
                dtype: dtype.clone(),
            }),
        }
    }
}

impl From<StoredDtype> for RecipeDtype {
    fn from(value: StoredDtype) -> Self {
        match value {
            StoredDtype::Bool => Self::Bool,
            StoredDtype::U8 => Self::U8,
            StoredDtype::I8 => Self::I8,
            StoredDtype::I16 => Self::I16,
            StoredDtype::U16 => Self::U16,
            StoredDtype::F16 => Self::F16,
            StoredDtype::BF16 => Self::BF16,
            StoredDtype::I32 => Self::I32,
            StoredDtype::U32 => Self::U32,
            StoredDtype::F32 => Self::F32,
            StoredDtype::F64 => Self::F64,
            StoredDtype::I64 => Self::I64,
            StoredDtype::U64 => Self::U64,
            StoredDtype::C64 => Self::C64,
            StoredDtype::F8E4M3 => Self::F8E4M3,
            StoredDtype::F8E5M2 => Self::F8E5M2,
            StoredDtype::F4 => Self::F4,
            StoredDtype::F8E8M0 => Self::F8E8M0,
            StoredDtype::Other(dtype) => Self::Other(dtype),
        }
    }
}

/// Shape, representation, and byte size inferred for a recipe.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecipeMetadata {
    /// Inferred logical output shape.
    pub shape: Vec<usize>,
    /// Inferred output scalar representation.
    pub dtype: RecipeDtype,
    /// Exact encoded or materialized output byte count.
    pub byte_len: u64,
}

impl RecipeMetadata {
    /// Returns the inferred output shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the inferred scalar representation.
    pub const fn dtype(&self) -> &RecipeDtype {
        &self.dtype
    }

    /// Returns the exact inferred byte count.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// Typed operations needed to derive a runtime parameter from checkpoint tensors.
#[derive(Debug, Clone, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum DerivedWeightRecipe {
    Source {
        key: String,
        selection: TensorSelection,
    },
    Select {
        input: Box<Self>,
        selection: TensorSelection,
    },
    Concatenate {
        axis: usize,
        inputs: Vec<Self>,
    },
    Stack {
        axis: usize,
        inputs: Vec<Self>,
    },
    Reshape {
        input: Box<Self>,
        shape: Vec<usize>,
    },
    Transpose {
        input: Box<Self>,
        axes: Vec<usize>,
    },
    Cast {
        input: Box<Self>,
        dtype: RecipeDtype,
    },
    View {
        input: Box<Self>,
        dtype: RecipeDtype,
        shape: Vec<usize>,
    },
    NegLog {
        input: Box<Self>,
    },
    SubtractOne {
        input: Box<Self>,
    },
}

impl DerivedWeightRecipe {
    /// Creates a recipe reading one selected checkpoint tensor.
    pub fn source(key: impl Into<String>, selection: TensorSelection) -> Self {
        Self::Source {
            key: key.into(),
            selection,
        }
    }

    /// Proves that every physical source can honor its declared bounded read.
    pub fn preflight_bounded<S: BoundedRecipeSource + ?Sized>(
        &self,
        source: &S,
    ) -> Result<(), RecipeError> {
        match self {
            Self::Source { key, selection } => {
                source.verify_bounded_source(key, selection.clone())?;
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                for input in inputs {
                    input.preflight_bounded(source)?;
                }
            }
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => input.preflight_bounded(source)?,
        }
        Ok(())
    }

    /// Rewrites an output selection toward physically bounded sources.
    pub fn select_bounded<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
        selection: TensorSelection,
    ) -> Result<Self, RecipeError> {
        let metadata = self.infer(catalog)?;
        let selection = normalize_selection(selection, &metadata.shape)?;
        let expected_shape = selected_shape(metadata.shape.clone(), &selection)?;
        let expanded = expand_indexed_sources(self.clone());
        let rewritten = normalize_bounded_source_ranges(
            expand_indexed_sources(push_selection(&expanded, catalog, selection)?),
            catalog,
        )?;
        let actual = rewritten.infer(catalog)?;
        if actual.shape != expected_shape || actual.dtype != metadata.dtype {
            return Err(RecipeError::SelectionPushdownUnsupported {
                operation: "recipe",
                reason: format!(
                    "rewrite produced shape {:?} and dtype {:?}, expected {:?} and {:?}",
                    actual.shape, actual.dtype, expected_shape, metadata.dtype
                ),
            });
        }
        Ok(rewritten)
    }

    /// Selects rows from one matrix while retaining leading singleton axes.
    pub fn select_bounded_matrix_rows<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
        leading_index: usize,
        start: usize,
        end: usize,
    ) -> Result<Self, RecipeError> {
        let metadata = self.infer(catalog)?;
        if metadata.shape.len() < 2 {
            return Err(RecipeError::SelectionPushdownUnsupported {
                operation: "matrix row selection",
                reason: format!("rank {} has no matrix row axis", metadata.shape.len()),
            });
        }
        let row_axis = metadata.shape.len() - 2;
        let leading = usize::try_from(element_count(
            &metadata.shape[..row_axis],
            "leading matrix dimensions",
        )?)
        .map_err(|_| RecipeError::ArithmeticOverflow("leading matrix dimensions"))?;
        if leading_index >= leading {
            return Err(RecipeError::InvalidIndices {
                axis: 0,
                dimension: leading,
            });
        }
        let mut coordinates = vec![0usize; row_axis];
        let mut remainder = leading_index;
        for axis in (0..row_axis).rev() {
            let dimension = metadata.shape[axis];
            coordinates[axis] = remainder % dimension;
            remainder /= dimension;
        }
        let mut selected = self.clone();
        for (axis, coordinate) in coordinates.into_iter().enumerate() {
            selected = selected.select_bounded(
                catalog,
                TensorSelection::Range {
                    axis,
                    start: coordinate,
                    end: coordinate + 1,
                },
            )?;
        }
        selected.select_bounded(
            catalog,
            TensorSelection::Range {
                axis: row_axis,
                start,
                end,
            },
        )
    }

    /// Returns a conservative bound for simultaneously live recipe values.
    pub fn peak_materialization_bytes<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
    ) -> Result<u64, RecipeError> {
        let output_bytes = self.infer(catalog)?.byte_len();
        match self {
            Self::Source { .. } => Ok(output_bytes),
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => {
                let input_bytes = input.infer(catalog)?.byte_len();
                let child_peak = input.peak_materialization_bytes(catalog)?;
                Ok(child_peak.max(input_bytes.checked_add(output_bytes).ok_or(
                    RecipeError::ArithmeticOverflow("unary recipe peak materialization bytes"),
                )?))
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                let mut retained = 0u64;
                let mut peak = 0u64;
                for input in inputs {
                    let child_peak = input.peak_materialization_bytes(catalog)?;
                    peak = peak.max(retained.checked_add(child_peak).ok_or(
                        RecipeError::ArithmeticOverflow("joined recipe child peak bytes"),
                    )?);
                    retained = retained
                        .checked_add(input.infer(catalog)?.byte_len())
                        .ok_or(RecipeError::ArithmeticOverflow(
                            "joined recipe retained input bytes",
                        ))?;
                }
                Ok(peak.max(retained.checked_add(output_bytes).ok_or(
                    RecipeError::ArithmeticOverflow("joined recipe output peak bytes"),
                )?))
            }
        }
    }

    /// Returns every source checkpoint key in deterministic order.
    pub fn source_keys(&self) -> Vec<&str> {
        let mut keys = BTreeSet::new();
        self.collect_source_keys(&mut keys);
        keys.into_iter().collect()
    }

    fn collect_source_keys<'a>(&'a self, keys: &mut BTreeSet<&'a str>) {
        match self {
            Self::Source { key, .. } => {
                keys.insert(key);
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                for input in inputs {
                    input.collect_source_keys(keys);
                }
            }
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => input.collect_source_keys(keys),
        }
    }

    /// Validates every operation and infers its exact output metadata.
    pub fn infer<C: RecipeCatalog + ?Sized>(
        &self,
        catalog: &C,
    ) -> Result<RecipeMetadata, RecipeError> {
        match self {
            Self::Source { key, selection } => {
                if key.trim().is_empty() {
                    return Err(RecipeError::EmptySourceKey);
                }
                let metadata = catalog.tensor_metadata(key)?;
                metadata_for(
                    selected_shape(metadata.logical_shape, selection)?,
                    metadata.stored_dtype.into(),
                )
            }
            Self::Select { input, selection } => {
                let metadata = input.infer(catalog)?;
                metadata_for(selected_shape(metadata.shape, selection)?, metadata.dtype)
            }
            Self::Concatenate { axis, inputs } => infer_join(catalog, *axis, inputs, false),
            Self::Stack { axis, inputs } => infer_join(catalog, *axis, inputs, true),
            Self::Reshape { input, shape } => {
                let metadata = input.infer(catalog)?;
                let old_count = element_count(&metadata.shape, "reshape input")?;
                let new_count = element_count(shape, "reshape output")?;
                if old_count != new_count {
                    return Err(RecipeError::ElementCountMismatch {
                        input: old_count,
                        output: new_count,
                    });
                }
                metadata_for(shape.clone(), metadata.dtype)
            }
            Self::Transpose { input, axes } => {
                let metadata = input.infer(catalog)?;
                let unique = axes.iter().copied().collect::<BTreeSet<_>>();
                if axes.len() != metadata.shape.len()
                    || unique.len() != axes.len()
                    || axes.iter().any(|axis| *axis >= axes.len())
                {
                    return Err(RecipeError::InvalidPermutation {
                        axes: axes.clone(),
                        rank: metadata.shape.len(),
                    });
                }
                metadata_for(
                    axes.iter().map(|axis| metadata.shape[*axis]).collect(),
                    metadata.dtype,
                )
            }
            Self::Cast { input, dtype } => metadata_for(input.infer(catalog)?.shape, dtype.clone()),
            Self::View {
                input,
                dtype,
                shape,
            } => {
                let input = input.infer(catalog)?;
                let output = metadata_for(shape.clone(), dtype.clone())?;
                if input.byte_len != output.byte_len {
                    return Err(RecipeError::ByteCountMismatch {
                        input: input.byte_len,
                        output: output.byte_len,
                    });
                }
                Ok(output)
            }
            Self::NegLog { input } | Self::SubtractOne { input } => input.infer(catalog),
        }
    }
}

fn selected_shape(
    mut shape: Vec<usize>,
    selection: &TensorSelection,
) -> Result<Vec<usize>, RecipeError> {
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let rank = shape.len();
            let dimension = shape
                .get_mut(*axis)
                .ok_or(RecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
            if start >= end || *end > *dimension {
                return Err(RecipeError::InvalidRange {
                    axis: *axis,
                    start: *start,
                    end: *end,
                    dimension: *dimension,
                });
            }
            *dimension = end - start;
        }
        TensorSelection::Indices { axis, indices } => {
            let rank = shape.len();
            let dimension = shape
                .get_mut(*axis)
                .ok_or(RecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
            if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                return Err(RecipeError::InvalidIndices {
                    axis: *axis,
                    dimension: *dimension,
                });
            }
            *dimension = indices.len();
        }
        TensorSelection::Contiguous {
            offset_elements,
            shape: selected,
        } => {
            if selected.is_empty() || selected.contains(&0) {
                return Err(RecipeError::InvalidContiguousSelection);
            }
            let full = element_count(&shape, "contiguous source")?;
            let count = element_count(selected, "contiguous selection")?;
            let end = u64::try_from(*offset_elements)
                .map_err(|_| RecipeError::ArithmeticOverflow("contiguous offset"))?
                .checked_add(count)
                .ok_or(RecipeError::ArithmeticOverflow("contiguous end"))?;
            if end > full {
                return Err(RecipeError::InvalidContiguousSelection);
            }
            shape = selected.clone();
        }
    }
    Ok(shape)
}

fn expand_indexed_sources(recipe: DerivedWeightRecipe) -> DerivedWeightRecipe {
    match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: TensorSelection::Indices { axis, indices },
        } => {
            let mut runs = Vec::<(usize, usize)>::new();
            for index in indices {
                if let Some((_, end)) = runs.last_mut() {
                    if *end == index {
                        *end += 1;
                        continue;
                    }
                }
                runs.push((index, index + 1));
            }
            let mut inputs = runs
                .into_iter()
                .map(|(start, end)| {
                    DerivedWeightRecipe::source(
                        key.clone(),
                        TensorSelection::Range { axis, start, end },
                    )
                })
                .collect::<Vec<_>>();
            if inputs.len() == 1 {
                inputs.pop().unwrap()
            } else {
                DerivedWeightRecipe::Concatenate { axis, inputs }
            }
        }
        DerivedWeightRecipe::Source { .. } => recipe,
        DerivedWeightRecipe::Select { input, selection } => DerivedWeightRecipe::Select {
            input: Box::new(expand_indexed_sources(*input)),
            selection,
        },
        DerivedWeightRecipe::Concatenate { axis, inputs } => DerivedWeightRecipe::Concatenate {
            axis,
            inputs: inputs.into_iter().map(expand_indexed_sources).collect(),
        },
        DerivedWeightRecipe::Stack { axis, inputs } => DerivedWeightRecipe::Stack {
            axis,
            inputs: inputs.into_iter().map(expand_indexed_sources).collect(),
        },
        DerivedWeightRecipe::Reshape { input, shape } => DerivedWeightRecipe::Reshape {
            input: Box::new(expand_indexed_sources(*input)),
            shape,
        },
        DerivedWeightRecipe::Transpose { input, axes } => DerivedWeightRecipe::Transpose {
            input: Box::new(expand_indexed_sources(*input)),
            axes,
        },
        DerivedWeightRecipe::Cast { input, dtype } => DerivedWeightRecipe::Cast {
            input: Box::new(expand_indexed_sources(*input)),
            dtype,
        },
        DerivedWeightRecipe::View {
            input,
            dtype,
            shape,
        } => DerivedWeightRecipe::View {
            input: Box::new(expand_indexed_sources(*input)),
            dtype,
            shape,
        },
        DerivedWeightRecipe::NegLog { input } => DerivedWeightRecipe::NegLog {
            input: Box::new(expand_indexed_sources(*input)),
        },
        DerivedWeightRecipe::SubtractOne { input } => DerivedWeightRecipe::SubtractOne {
            input: Box::new(expand_indexed_sources(*input)),
        },
    }
}

fn normalize_bounded_source_ranges<C: RecipeCatalog + ?Sized>(
    recipe: DerivedWeightRecipe,
    store: &C,
) -> Result<DerivedWeightRecipe, RecipeError> {
    Ok(match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: TensorSelection::Range { axis, start, end },
        } if axis > 0 => {
            let shape = store.tensor_metadata(&key)?.logical_shape;
            if shape[..axis].iter().product::<usize>() == 1 {
                let trailing = shape[axis + 1..]
                    .iter()
                    .try_fold(1usize, |count, dimension| {
                        count
                            .checked_mul(*dimension)
                            .ok_or(RecipeError::ArithmeticOverflow(
                                "bounded source range trailing span",
                            ))
                    })?;
                let offset_elements =
                    start
                        .checked_mul(trailing)
                        .ok_or(RecipeError::ArithmeticOverflow(
                            "bounded source range offset",
                        ))?;
                let mut selected_shape = shape;
                selected_shape[axis] = end - start;
                DerivedWeightRecipe::source(
                    key,
                    TensorSelection::Contiguous {
                        offset_elements,
                        shape: selected_shape,
                    },
                )
            } else {
                DerivedWeightRecipe::source(key, TensorSelection::Range { axis, start, end })
            }
        }
        DerivedWeightRecipe::Source { .. } => recipe,
        DerivedWeightRecipe::Select { input, selection } => DerivedWeightRecipe::Select {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            selection,
        },
        DerivedWeightRecipe::Concatenate { axis, inputs } => DerivedWeightRecipe::Concatenate {
            axis,
            inputs: inputs
                .into_iter()
                .map(|input| normalize_bounded_source_ranges(input, store))
                .collect::<Result<Vec<_>, _>>()?,
        },
        DerivedWeightRecipe::Stack { axis, inputs } => DerivedWeightRecipe::Stack {
            axis,
            inputs: inputs
                .into_iter()
                .map(|input| normalize_bounded_source_ranges(input, store))
                .collect::<Result<Vec<_>, _>>()?,
        },
        DerivedWeightRecipe::Reshape { input, shape } => DerivedWeightRecipe::Reshape {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            shape,
        },
        DerivedWeightRecipe::Transpose { input, axes } => DerivedWeightRecipe::Transpose {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            axes,
        },
        DerivedWeightRecipe::Cast { input, dtype } => DerivedWeightRecipe::Cast {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            dtype,
        },
        DerivedWeightRecipe::View {
            input,
            dtype,
            shape,
        } => DerivedWeightRecipe::View {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
            dtype,
            shape,
        },
        DerivedWeightRecipe::NegLog { input } => DerivedWeightRecipe::NegLog {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
        },
        DerivedWeightRecipe::SubtractOne { input } => DerivedWeightRecipe::SubtractOne {
            input: Box::new(normalize_bounded_source_ranges(*input, store)?),
        },
    })
}

fn push_selection<C: RecipeCatalog + ?Sized>(
    recipe: &DerivedWeightRecipe,
    store: &C,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, RecipeError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok(recipe.clone());
    }
    match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: source_selection,
        } => {
            let source_shape = store.tensor_metadata(key)?.logical_shape;
            let source_selection = normalize_selection(source_selection.clone(), &source_shape)?;
            let selected_source_shape = selected_shape(source_shape.clone(), &source_selection)?;
            let selection = normalize_selection(selection, &selected_source_shape)?;
            if matches!(selection, TensorSelection::Full) {
                return Ok(DerivedWeightRecipe::source(key.clone(), source_selection));
            }
            if matches!(source_selection, TensorSelection::Full) {
                return Ok(DerivedWeightRecipe::source(key.clone(), selection));
            }
            if let Some(selection) = select_from_contiguous_span(&source_selection, &selection)? {
                return Ok(DerivedWeightRecipe::source(key.clone(), selection));
            }
            if selection_axis(&source_selection) == selection_axis(&selection) {
                return Ok(DerivedWeightRecipe::source(
                    key.clone(),
                    compose_same_axis_selection(&source_selection, &selection)?,
                ));
            }
            if let Some(contiguous) =
                combine_independent_ranges(&source_shape, &source_selection, &selection)?
            {
                return Ok(DerivedWeightRecipe::source(key.clone(), contiguous));
            }
            Ok(DerivedWeightRecipe::Select {
                input: Box::new(DerivedWeightRecipe::source(key.clone(), source_selection)),
                selection,
            })
        }
        DerivedWeightRecipe::Select {
            input,
            selection: existing,
        } => {
            if matches!(existing, TensorSelection::Full) {
                return push_selection(input, store, selection);
            }
            if selection_axis(existing) == selection_axis(&selection) {
                return push_selection(
                    input,
                    store,
                    compose_same_axis_selection(existing, &selection)?,
                );
            }
            // Independent axis selections commute. Push the newest selection
            // toward the source first so a dynamic row tile can combine with
            // an expert range into one contiguous checkpoint span before the
            // pre-existing TP column selection is reapplied.
            let selected_input = push_selection(input, store, selection)?;
            push_selection(&selected_input, store, existing.clone())
        }
        DerivedWeightRecipe::Concatenate { axis, inputs } => {
            push_concatenate_selection(*axis, inputs, store, selection)
        }
        DerivedWeightRecipe::Stack { axis, inputs } => {
            push_stack_selection(*axis, inputs, store, selection)
        }
        DerivedWeightRecipe::Reshape { input, shape } => {
            let input_metadata = input.infer(store)?;
            let output_metadata = recipe.infer(store)?;
            let input_selection = map_reinterpret_selection(
                &input_metadata,
                &output_metadata,
                &selection,
                "reshape",
            )?;
            Ok(DerivedWeightRecipe::Reshape {
                input: Box::new(push_selection(input, store, input_selection)?),
                shape: selected_shape(shape.clone(), &selection)?,
            })
        }
        DerivedWeightRecipe::Transpose { input, axes } => {
            let output_axis = selection_axis(&selection).expect("non-full selection");
            let input_axis = *axes
                .get(output_axis)
                .ok_or(RecipeError::InvalidSelectionAxis {
                    axis: output_axis,
                    rank: axes.len(),
                })?;
            Ok(DerivedWeightRecipe::Transpose {
                input: Box::new(push_selection(
                    input,
                    store,
                    selection_with_axis(selection, input_axis),
                )?),
                axes: axes.clone(),
            })
        }
        DerivedWeightRecipe::Cast { input, dtype } => Ok(DerivedWeightRecipe::Cast {
            input: Box::new(push_selection(input, store, selection)?),
            dtype: dtype.clone(),
        }),
        DerivedWeightRecipe::View {
            input,
            dtype,
            shape,
        } => {
            let input_metadata = input.infer(store)?;
            let output_metadata = recipe.infer(store)?;
            let input_selection =
                map_reinterpret_selection(&input_metadata, &output_metadata, &selection, "view")?;
            Ok(DerivedWeightRecipe::View {
                input: Box::new(push_selection(input, store, input_selection)?),
                dtype: dtype.clone(),
                shape: selected_shape(shape.clone(), &selection)?,
            })
        }
        DerivedWeightRecipe::NegLog { input } => Ok(DerivedWeightRecipe::NegLog {
            input: Box::new(push_selection(input, store, selection)?),
        }),
        DerivedWeightRecipe::SubtractOne { input } => Ok(DerivedWeightRecipe::SubtractOne {
            input: Box::new(push_selection(input, store, selection)?),
        }),
    }
}

fn push_concatenate_selection<C: RecipeCatalog + ?Sized>(
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    store: &C,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, RecipeError> {
    let selected_axis = selection_axis(&selection).expect("non-full selection");
    if selected_axis != axis {
        let inputs = inputs
            .iter()
            .map(|input| push_selection(input, store, selection.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(DerivedWeightRecipe::Concatenate { axis, inputs });
    }
    let metadata = inputs
        .iter()
        .map(|input| input.infer(store))
        .collect::<Result<Vec<_>, _>>()?;
    let dimensions = metadata
        .iter()
        .map(|item| item.shape[axis])
        .collect::<Vec<_>>();
    let mut rewritten = Vec::new();
    match selection {
        TensorSelection::Range { start, end, .. } => {
            let mut offset = 0usize;
            for (input, dimension) in inputs.iter().zip(dimensions) {
                let child_end =
                    offset
                        .checked_add(dimension)
                        .ok_or(RecipeError::ArithmeticOverflow(
                            "concatenate selection offset",
                        ))?;
                let overlap_start = start.max(offset);
                let overlap_end = end.min(child_end);
                if overlap_start < overlap_end {
                    let child_selection = normalize_selection(
                        TensorSelection::Range {
                            axis,
                            start: overlap_start - offset,
                            end: overlap_end - offset,
                        },
                        &input.infer(store)?.shape,
                    )?;
                    rewritten.push(push_selection(input, store, child_selection)?);
                }
                offset = child_end;
            }
        }
        TensorSelection::Indices { indices, .. } => {
            let mut offsets = Vec::with_capacity(dimensions.len() + 1);
            offsets.push(0usize);
            for dimension in dimensions {
                let next = offsets
                    .last()
                    .copied()
                    .unwrap()
                    .checked_add(dimension)
                    .ok_or(RecipeError::ArithmeticOverflow(
                        "concatenate selection offset",
                    ))?;
                offsets.push(next);
            }
            let mut runs = Vec::<(usize, Vec<usize>)>::new();
            for index in indices {
                let child = offsets
                    .windows(2)
                    .position(|bounds| index >= bounds[0] && index < bounds[1])
                    .ok_or(RecipeError::InvalidIndices {
                        axis,
                        dimension: *offsets.last().unwrap(),
                    })?;
                let local = index - offsets[child];
                if let Some((last_child, local_indices)) = runs.last_mut() {
                    if *last_child == child {
                        local_indices.push(local);
                        continue;
                    }
                }
                runs.push((child, vec![local]));
            }
            for (child, indices) in runs {
                rewritten.push(push_selection(
                    &inputs[child],
                    store,
                    TensorSelection::Indices { axis, indices },
                )?);
            }
        }
        TensorSelection::Full => unreachable!(),
        TensorSelection::Contiguous { .. } => {
            return Err(RecipeError::SelectionPushdownUnsupported {
                operation: "concatenate",
                reason: "a storage-contiguous span has no concatenate-axis semantics".into(),
            })
        }
    }
    match rewritten.len() {
        0 => Err(RecipeError::SelectionPushdownUnsupported {
            operation: "concatenate",
            reason: "selection did not intersect any child".into(),
        }),
        1 => Ok(rewritten.pop().unwrap()),
        _ => Ok(DerivedWeightRecipe::Concatenate {
            axis,
            inputs: rewritten,
        }),
    }
}

fn push_stack_selection<C: RecipeCatalog + ?Sized>(
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    store: &C,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, RecipeError> {
    let selected_axis = selection_axis(&selection).expect("non-full selection");
    if selected_axis == axis {
        let selected = match selection {
            TensorSelection::Range { start, end, .. } => inputs[start..end].to_vec(),
            TensorSelection::Indices { indices, .. } => indices
                .into_iter()
                .map(|index| inputs[index].clone())
                .collect(),
            TensorSelection::Full => unreachable!(),
            TensorSelection::Contiguous { .. } => {
                return Err(RecipeError::SelectionPushdownUnsupported {
                    operation: "stack",
                    reason: "a storage-contiguous span has no stack-axis semantics".into(),
                })
            }
        };
        return Ok(DerivedWeightRecipe::Stack {
            axis,
            inputs: selected,
        });
    }
    let input_axis = if selected_axis < axis {
        selected_axis
    } else {
        selected_axis - 1
    };
    let input_selection = selection_with_axis(selection, input_axis);
    let inputs = inputs
        .iter()
        .map(|input| push_selection(input, store, input_selection.clone()))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DerivedWeightRecipe::Stack { axis, inputs })
}

fn map_reinterpret_selection(
    input: &RecipeMetadata,
    output: &RecipeMetadata,
    selection: &TensorSelection,
    operation: &'static str,
) -> Result<TensorSelection, RecipeError> {
    let output_axis = selection_axis(selection).expect("non-full selection");
    let output_unit = axis_unit_bytes(&output.shape, output.dtype.bit_width()?, output_axis)?;
    let output_cycle = output_unit
        .checked_mul(output.shape[output_axis] as u64)
        .ok_or(RecipeError::ArithmeticOverflow("selection output cycle"))?;
    for input_axis in 0..input.shape.len() {
        let input_unit = axis_unit_bytes(&input.shape, input.dtype.bit_width()?, input_axis)?;
        let input_cycle = input_unit
            .checked_mul(input.shape[input_axis] as u64)
            .ok_or(RecipeError::ArithmeticOverflow("selection input cycle"))?;
        if input_cycle != output_cycle {
            continue;
        }
        if let Some(mapped) = map_selection_units(
            selection,
            input_axis,
            input.shape[input_axis],
            output_unit,
            input_unit,
        )? {
            return normalize_selection(mapped, &input.shape);
        }
    }
    Err(RecipeError::SelectionPushdownUnsupported {
        operation,
        reason: format!(
            "axis {output_axis} selection cannot be expressed as a single-axis bounded selection from shape {:?} to {:?}",
            input.shape, output.shape
        ),
    })
}

fn map_selection_units(
    selection: &TensorSelection,
    input_axis: usize,
    input_dimension: usize,
    output_unit: u64,
    input_unit: u64,
) -> Result<Option<TensorSelection>, RecipeError> {
    let map_interval = |start: usize, end: usize| -> Result<Option<(usize, usize)>, RecipeError> {
        let start_bytes = (start as u64)
            .checked_mul(output_unit)
            .ok_or(RecipeError::ArithmeticOverflow("selection interval start"))?;
        let end_bytes = (end as u64)
            .checked_mul(output_unit)
            .ok_or(RecipeError::ArithmeticOverflow("selection interval end"))?;
        if start_bytes % input_unit != 0 || end_bytes % input_unit != 0 {
            return Ok(None);
        }
        let start = usize::try_from(start_bytes / input_unit)
            .map_err(|_| RecipeError::ArithmeticOverflow("mapped selection start"))?;
        let end = usize::try_from(end_bytes / input_unit)
            .map_err(|_| RecipeError::ArithmeticOverflow("mapped selection end"))?;
        Ok((end <= input_dimension).then_some((start, end)))
    };
    match selection {
        TensorSelection::Range { start, end, .. } => Ok(map_interval(*start, *end)?.map(
            |(start, end)| TensorSelection::Range {
                axis: input_axis,
                start,
                end,
            },
        )),
        TensorSelection::Indices { indices, .. } => {
            let mut mapped = Vec::new();
            let mut run_start = indices[0];
            let mut run_end = run_start + 1;
            for index in indices.iter().copied().skip(1) {
                if index == run_end {
                    run_end += 1;
                    continue;
                }
                let Some((start, end)) = map_interval(run_start, run_end)? else {
                    return Ok(None);
                };
                mapped.extend(start..end);
                run_start = index;
                run_end = index + 1;
            }
            let Some((start, end)) = map_interval(run_start, run_end)? else {
                return Ok(None);
            };
            mapped.extend(start..end);
            Ok(Some(TensorSelection::Indices {
                axis: input_axis,
                indices: mapped,
            }))
        }
        TensorSelection::Full => Ok(Some(TensorSelection::Full)),
        TensorSelection::Contiguous { .. } => Ok(None),
    }
}

fn axis_unit_bytes(shape: &[usize], dtype_width: u64, axis: usize) -> Result<u64, RecipeError> {
    shape[axis + 1..]
        .iter()
        .try_fold(dtype_width, |bytes, dimension| {
            bytes
                .checked_mul(*dimension as u64)
                .ok_or(RecipeError::ArithmeticOverflow("selection axis unit"))
        })
}

fn selection_axis(selection: &TensorSelection) -> Option<usize> {
    match selection {
        TensorSelection::Full => None,
        TensorSelection::Range { axis, .. } | TensorSelection::Indices { axis, .. } => Some(*axis),
        TensorSelection::Contiguous { .. } => None,
    }
}

fn selection_with_axis(selection: TensorSelection, axis: usize) -> TensorSelection {
    match selection {
        TensorSelection::Full => TensorSelection::Full,
        TensorSelection::Range { start, end, .. } => TensorSelection::Range { axis, start, end },
        TensorSelection::Indices { indices, .. } => TensorSelection::Indices { axis, indices },
        selection @ TensorSelection::Contiguous { .. } => selection,
    }
}

fn normalize_selection(
    selection: TensorSelection,
    shape: &[usize],
) -> Result<TensorSelection, RecipeError> {
    selected_shape(shape.to_vec(), &selection)?;
    match selection {
        TensorSelection::Range {
            axis,
            start: 0,
            end,
        } if end == shape[axis] => Ok(TensorSelection::Full),
        TensorSelection::Indices { axis, indices }
            if indices.windows(2).all(|pair| pair[1] == pair[0] + 1) =>
        {
            let start = indices[0];
            let end = indices[indices.len() - 1] + 1;
            if start == 0 && end == shape[axis] {
                Ok(TensorSelection::Full)
            } else {
                Ok(TensorSelection::Range { axis, start, end })
            }
        }
        selection => Ok(selection),
    }
}

fn compose_same_axis_selection(
    existing: &TensorSelection,
    requested: &TensorSelection,
) -> Result<TensorSelection, RecipeError> {
    debug_assert_eq!(selection_axis(existing), selection_axis(requested));
    let axis = selection_axis(existing).expect("non-full selections");
    match (existing, requested) {
        (
            TensorSelection::Range { start, .. },
            TensorSelection::Range {
                start: requested_start,
                end: requested_end,
                ..
            },
        ) => Ok(TensorSelection::Range {
            axis,
            start: start + requested_start,
            end: start + requested_end,
        }),
        (TensorSelection::Range { start, .. }, TensorSelection::Indices { indices, .. }) => {
            Ok(TensorSelection::Indices {
                axis,
                indices: indices.iter().map(|index| start + index).collect(),
            })
        }
        (TensorSelection::Indices { indices, .. }, TensorSelection::Range { start, end, .. }) => {
            Ok(TensorSelection::Indices {
                axis,
                indices: indices[*start..*end].to_vec(),
            })
        }
        (
            TensorSelection::Indices {
                indices: existing, ..
            },
            TensorSelection::Indices { indices, .. },
        ) => Ok(TensorSelection::Indices {
            axis,
            indices: indices.iter().map(|index| existing[*index]).collect(),
        }),
        _ => Err(RecipeError::SelectionPushdownUnsupported {
            operation: "selection composition",
            reason: "full selections must be normalized before composition".into(),
        }),
    }
}

fn combine_independent_ranges(
    source_shape: &[usize],
    existing: &TensorSelection,
    requested: &TensorSelection,
) -> Result<Option<TensorSelection>, RecipeError> {
    let (
        TensorSelection::Range {
            axis: existing_axis,
            start: existing_start,
            end: existing_end,
        },
        TensorSelection::Range {
            axis: requested_axis,
            start: requested_start,
            end: requested_end,
        },
    ) = (existing, requested)
    else {
        return Ok(None);
    };
    if existing_axis == requested_axis {
        return Ok(None);
    }
    let mut starts = vec![0usize; source_shape.len()];
    let mut ends = source_shape.to_vec();
    starts[*existing_axis] = *existing_start;
    ends[*existing_axis] = *existing_end;
    starts[*requested_axis] = *requested_start;
    ends[*requested_axis] = *requested_end;
    let selected_shape = starts
        .iter()
        .zip(&ends)
        .map(|(start, end)| end - start)
        .collect::<Vec<_>>();
    let Some(last_partial) = (0..source_shape.len())
        .rev()
        .find(|axis| starts[*axis] != 0 || ends[*axis] != source_shape[*axis])
    else {
        return Ok(Some(TensorSelection::Full));
    };
    if selected_shape[..last_partial]
        .iter()
        .any(|dimension| *dimension != 1)
        || (last_partial + 1..source_shape.len())
            .any(|axis| starts[axis] != 0 || ends[axis] != source_shape[axis])
    {
        return Ok(None);
    }
    let mut offset_elements = 0usize;
    let mut stride = 1usize;
    for axis in (0..source_shape.len()).rev() {
        offset_elements = offset_elements
            .checked_add(starts[axis].checked_mul(stride).ok_or(
                RecipeError::ArithmeticOverflow("contiguous selection offset"),
            )?)
            .ok_or(RecipeError::ArithmeticOverflow(
                "contiguous selection offset",
            ))?;
        stride = stride
            .checked_mul(source_shape[axis])
            .ok_or(RecipeError::ArithmeticOverflow(
                "contiguous selection stride",
            ))?;
    }
    Ok(Some(TensorSelection::Contiguous {
        offset_elements,
        shape: selected_shape,
    }))
}

fn select_from_contiguous_span(
    existing: &TensorSelection,
    requested: &TensorSelection,
) -> Result<Option<TensorSelection>, RecipeError> {
    let TensorSelection::Contiguous {
        offset_elements,
        shape,
    } = existing
    else {
        return Ok(None);
    };
    let (axis, start, end) = match requested {
        TensorSelection::Range { axis, start, end } => (*axis, *start, *end),
        TensorSelection::Indices { axis, indices }
            if indices.windows(2).all(|pair| pair[1] == pair[0] + 1) =>
        {
            (*axis, indices[0], indices[indices.len() - 1] + 1)
        }
        _ => return Ok(None),
    };
    if shape[..axis].iter().product::<usize>() != 1 {
        return Ok(None);
    }
    let trailing = shape[axis + 1..]
        .iter()
        .try_fold(1usize, |count, dimension| {
            count
                .checked_mul(*dimension)
                .ok_or(RecipeError::ArithmeticOverflow(
                    "contiguous selection trailing span",
                ))
        })?;
    let offset_elements = offset_elements
        .checked_add(
            start
                .checked_mul(trailing)
                .ok_or(RecipeError::ArithmeticOverflow(
                    "contiguous selection offset",
                ))?,
        )
        .ok_or(RecipeError::ArithmeticOverflow(
            "contiguous selection offset",
        ))?;
    let mut selected_shape = shape.clone();
    selected_shape[axis] = end - start;
    Ok(Some(TensorSelection::Contiguous {
        offset_elements,
        shape: selected_shape,
    }))
}

fn infer_join<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    stack: bool,
) -> Result<RecipeMetadata, RecipeError> {
    if inputs.is_empty() {
        return Err(RecipeError::EmptyInputs);
    }
    let metadata = inputs
        .iter()
        .map(|input| input.infer(catalog))
        .collect::<Result<Vec<_>, _>>()?;
    let first = &metadata[0];
    if metadata.iter().any(|item| item.dtype != first.dtype) {
        return Err(RecipeError::DtypeMismatch);
    }
    let rank = first.shape.len();
    if axis > rank || (!stack && axis == rank) {
        return Err(RecipeError::InvalidJoinAxis { axis, rank, stack });
    }
    if stack {
        if metadata.iter().any(|item| item.shape != first.shape) {
            return Err(RecipeError::ShapeMismatch);
        }
        let mut shape = first.shape.clone();
        shape.insert(axis, metadata.len());
        metadata_for(shape, first.dtype.clone())
    } else {
        let mut shape = first.shape.clone();
        shape[axis] = 0;
        for item in &metadata {
            if item.shape.len() != rank
                || item
                    .shape
                    .iter()
                    .enumerate()
                    .any(|(index, dimension)| index != axis && *dimension != first.shape[index])
            {
                return Err(RecipeError::ShapeMismatch);
            }
            shape[axis] = shape[axis]
                .checked_add(item.shape[axis])
                .ok_or(RecipeError::ArithmeticOverflow("concatenate dimension"))?;
        }
        metadata_for(shape, first.dtype.clone())
    }
}

fn metadata_for(shape: Vec<usize>, dtype: RecipeDtype) -> Result<RecipeMetadata, RecipeError> {
    let bits = element_count(&shape, "recipe output")?
        .checked_mul(dtype.bit_width()?)
        .ok_or(RecipeError::ArithmeticOverflow("recipe output bits"))?;
    let byte_len = bits
        .checked_add(7)
        .ok_or(RecipeError::ArithmeticOverflow("recipe output bytes"))?
        / 8;
    if byte_len == 0 {
        return Err(RecipeError::ZeroSizedOutput);
    }
    Ok(RecipeMetadata {
        shape,
        dtype,
        byte_len,
    })
}

fn element_count(shape: &[usize], context: &'static str) -> Result<u64, RecipeError> {
    shape.iter().try_fold(1u64, |count, dimension| {
        count
            .checked_mul(
                u64::try_from(*dimension).map_err(|_| RecipeError::ArithmeticOverflow(context))?,
            )
            .ok_or(RecipeError::ArithmeticOverflow(context))
    })
}

/// Structured neutral recipe validation failures.
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum RecipeError {
    #[error("derived-weight source key must not be empty")]
    EmptySourceKey,
    #[error("selection axis {axis} is outside rank {rank}")]
    InvalidSelectionAxis { axis: usize, rank: usize },
    #[error("range {start}..{end} is invalid for axis {axis} dimension {dimension}")]
    InvalidRange {
        axis: usize,
        start: usize,
        end: usize,
        dimension: usize,
    },
    #[error("ordered indices are empty or outside axis {axis} dimension {dimension}")]
    InvalidIndices { axis: usize, dimension: usize },
    #[error("contiguous selection is empty or outside its source tensor")]
    InvalidContiguousSelection,
    #[error("concatenate and stack recipes require at least one input")]
    EmptyInputs,
    #[error("derived-weight inputs have different dtypes")]
    DtypeMismatch,
    #[error("derived-weight inputs have incompatible shapes")]
    ShapeMismatch,
    #[error("axis {axis} is invalid for rank {rank} (stack={stack})")]
    InvalidJoinAxis {
        axis: usize,
        rank: usize,
        stack: bool,
    },
    #[error("reshape changes element count from {input} to {output}")]
    ElementCountMismatch { input: u64, output: u64 },
    #[error("bitwise view changes byte count from {input} to {output}")]
    ByteCountMismatch { input: u64, output: u64 },
    #[error("axes {axes:?} are not a permutation of rank {rank}")]
    InvalidPermutation { axes: Vec<usize>, rank: usize },
    #[error("derived-weight output must contain at least one byte")]
    ZeroSizedOutput,
    #[error("derived-weight dtype {dtype} is unsupported")]
    UnsupportedDtype { dtype: String },
    #[error("derived-weight arithmetic overflow: {0}")]
    ArithmeticOverflow(&'static str),
    #[error("cannot push selection through {operation}: {reason}")]
    SelectionPushdownUnsupported {
        operation: &'static str,
        reason: String,
    },
    #[error(transparent)]
    Store(#[from] StoreError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{EncodedTensorLease, TensorReadRequest, WeightStoreDiagnostics};
    use std::path::Path;
    use std::sync::Mutex;

    struct Catalog;
    struct Lease;

    #[derive(Default)]
    struct BoundedCatalog {
        requests: Mutex<Vec<(String, TensorSelection)>>,
    }

    impl RecipeCatalog for BoundedCatalog {
        fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            Catalog.metadata(key)
        }
    }

    impl BoundedRecipeSource for BoundedCatalog {
        fn verify_bounded_source(
            &self,
            key: &str,
            selection: TensorSelection,
        ) -> Result<(), StoreError> {
            self.requests
                .lock()
                .unwrap()
                .push((key.to_owned(), selection));
            Ok(())
        }
    }

    impl EncodedTensorLease for Lease {
        fn metadata(&self) -> &TensorMetadata {
            unreachable!()
        }
        fn selection(&self) -> &TensorSelection {
            unreachable!()
        }
        fn output_shape(&self) -> &[usize] {
            unreachable!()
        }
        fn bounded_read_proof(&self) -> &crate::store::BoundedReadProof {
            unreachable!()
        }
        fn backing_path(&self) -> Option<&Path> {
            None
        }
        fn encoded_bytes(&self) -> Option<&[u8]> {
            None
        }
    }

    impl WeightStore for Catalog {
        type Lease = Lease;

        fn keys(&self) -> Vec<String> {
            vec!["left".into(), "right".into()]
        }
        fn metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            if !self.keys().iter().any(|candidate| candidate == key) {
                return Err(StoreError::UnknownTensor { key: key.into() });
            }
            Ok(TensorMetadata {
                name: key.into(),
                logical_shape: vec![2, 3],
                physical_shape: vec![2, 3],
                stored_dtype: StoredDtype::F16,
                encoded_byte_len: 12,
                backing_shard: None,
            })
        }
        fn acquire(&self, _: TensorReadRequest) -> Result<Self::Lease, StoreError> {
            unreachable!()
        }
        fn diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            unreachable!()
        }
    }

    #[test]
    fn nested_recipe_inference_is_backend_independent() {
        let recipe = DerivedWeightRecipe::Transpose {
            input: Box::new(DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source("left", TensorSelection::Full),
                    DerivedWeightRecipe::source(
                        "right",
                        TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 1,
                        },
                    ),
                ],
            }),
            axes: vec![1, 0],
        };
        let metadata = recipe.infer(&Catalog).unwrap();
        assert_eq!(metadata.shape(), &[3, 3]);
        assert_eq!(metadata.dtype(), &RecipeDtype::F16);
        assert_eq!(metadata.byte_len(), 18);
        assert_eq!(recipe.source_keys(), ["left", "right"]);
    }

    #[test]
    fn bounded_preflight_walks_exact_physical_source_selections() {
        let catalog = BoundedCatalog::default();
        let recipe = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source(
                    "left",
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                ),
                DerivedWeightRecipe::Reshape {
                    input: Box::new(DerivedWeightRecipe::source("right", TensorSelection::Full)),
                    shape: vec![2, 3],
                },
            ],
        };

        recipe.preflight_bounded(&catalog).unwrap();
        assert_eq!(
            *catalog.requests.lock().unwrap(),
            vec![
                (
                    "left".into(),
                    TensorSelection::Range {
                        axis: 0,
                        start: 0,
                        end: 1,
                    },
                ),
                ("right".into(), TensorSelection::Full),
            ]
        );
    }

    #[test]
    fn bounded_selection_pushdown_is_backend_independent() {
        let recipe = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source("left", TensorSelection::Full),
                DerivedWeightRecipe::source("right", TensorSelection::Full),
            ],
        };
        let selected = recipe
            .select_bounded(
                &Catalog,
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 3,
                },
            )
            .unwrap();
        assert_eq!(selected.infer(&Catalog).unwrap().shape(), &[2, 3]);
        assert_eq!(
            selected,
            DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source(
                        "left",
                        TensorSelection::Range {
                            axis: 0,
                            start: 1,
                            end: 2,
                        },
                    ),
                    DerivedWeightRecipe::source(
                        "right",
                        TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 1,
                        },
                    ),
                ],
            }
        );
    }
}
