//! Backend-neutral derived-weight recipes and shape inference.

use std::collections::BTreeSet;

use crate::store::{StoreError, TensorMetadata, TensorSelection, WeightStore};
use crate::StoredDtype;

/// Metadata-only catalog used to validate a derived-weight recipe.
pub trait RecipeCatalog {
    /// Returns source tensor metadata without reading its payload.
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError>;
}

impl<T: WeightStore> RecipeCatalog for T {
    fn tensor_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.metadata(key)
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
    #[error(transparent)]
    Store(#[from] StoreError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{EncodedTensorLease, TensorReadRequest, WeightStoreDiagnostics};
    use std::path::Path;

    struct Catalog;
    struct Lease;

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
}
