//! Deterministic checkpoint-derived weight recipes.

//!
//! Recipes describe the runtime representation of a parameter without tying it
//! to a single checkpoint key. They are validated from checkpoint metadata and
//! materialized on the residency source stream before device promotion.

pub use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype};
use eredu_checkpoint::store::{CheckpointSource, ReadPolicy, TensorReadRequest};

use safemlx::{
    ops::{concatenate_axis, contiguous, stack_axis},
    transforms::async_eval_with_event,
    Array, Dtype, Stream,
};

use crate::backend::mlx::runtime::checkpoint::store::{
    MlxParameterMaterializationContext, PendingWeightMaterialization, TensorSelection,
    WeightStoreError,
};

/// Converts an MLX scalar type into the backend-neutral recipe representation.
pub fn recipe_dtype_from_mlx(value: Dtype) -> RecipeDtype {
    match value {
        Dtype::Bool => RecipeDtype::Bool,
        Dtype::Uint8 => RecipeDtype::U8,
        Dtype::Uint16 => RecipeDtype::U16,
        Dtype::Uint32 => RecipeDtype::U32,
        Dtype::Uint64 => RecipeDtype::U64,
        Dtype::Int8 => RecipeDtype::I8,
        Dtype::Int16 => RecipeDtype::I16,
        Dtype::Int32 => RecipeDtype::I32,
        Dtype::Int64 => RecipeDtype::I64,
        Dtype::Float16 => RecipeDtype::F16,
        Dtype::Float32 => RecipeDtype::F32,
        Dtype::Float64 => RecipeDtype::F64,
        Dtype::Bfloat16 => RecipeDtype::BF16,
        Dtype::Complex64 => RecipeDtype::C64,
    }
}

fn mlx_dtype(value: &RecipeDtype) -> Result<Dtype, WeightRecipeError> {
    match value {
        RecipeDtype::Bool => Ok(Dtype::Bool),
        RecipeDtype::U8
        | RecipeDtype::F8E4M3
        | RecipeDtype::F8E5M2
        | RecipeDtype::F4
        | RecipeDtype::F8E8M0 => Ok(Dtype::Uint8),
        RecipeDtype::I8 => Ok(Dtype::Int8),
        RecipeDtype::I16 => Ok(Dtype::Int16),
        RecipeDtype::U16 => Ok(Dtype::Uint16),
        RecipeDtype::F16 => Ok(Dtype::Float16),
        RecipeDtype::BF16 => Ok(Dtype::Bfloat16),
        RecipeDtype::I32 => Ok(Dtype::Int32),
        RecipeDtype::U32 => Ok(Dtype::Uint32),
        RecipeDtype::F32 => Ok(Dtype::Float32),
        RecipeDtype::F64 => Ok(Dtype::Float64),
        RecipeDtype::I64 => Ok(Dtype::Int64),
        RecipeDtype::U64 => Ok(Dtype::Uint64),
        RecipeDtype::C64 => Ok(Dtype::Complex64),
        RecipeDtype::Other(dtype) => Err(WeightRecipeError::UnsupportedDtype {
            dtype: dtype.clone(),
        }),
        _ => Err(WeightRecipeError::UnsupportedDtype {
            dtype: format!("{value:?}"),
        }),
    }
}

/// Lowers a terminal logical MXFP4 value recipe to MLX's packed U32 storage.
///
/// Neutral checkpoint recipes describe the represented F4 values. MLX affine
/// kernels instead address eight packed F4 values through each U32 unit, so
/// composition must apply this lowering before it constructs runtime bindings.
pub fn lower_mxfp4_recipe(
    recipe: DerivedWeightRecipe,
    store: &dyn CheckpointSource,
) -> Result<DerivedWeightRecipe, WeightRecipeError> {
    let metadata = recipe.infer(store)?;
    let DerivedWeightRecipe::View {
        input,
        dtype: RecipeDtype::F4,
        shape,
    } = recipe
    else {
        return Err(WeightRecipeError::ExpectedLogicalMxFp4 {
            dtype: metadata.dtype().clone(),
        });
    };
    debug_assert_eq!(shape, metadata.shape());
    let mut packed_shape = shape;
    let logical_width = packed_shape
        .last_mut()
        .ok_or(WeightRecipeError::InvalidMxFp4LogicalShape)?;
    if *logical_width == 0 || !logical_width.is_multiple_of(8) {
        return Err(WeightRecipeError::InvalidMxFp4LogicalShape);
    }
    *logical_width /= 8;
    let lowered = DerivedWeightRecipe::View {
        input,
        dtype: RecipeDtype::U32,
        shape: packed_shape,
    };
    lowered.infer(store)?;
    Ok(lowered)
}

/// MLX lowering operations for a backend-neutral recipe.
pub trait MlxWeightRecipeExt {
    #[cfg(test)]
    fn materialize(
        &self,
        store: &dyn CheckpointSource,
        source_stream: &Stream,
    ) -> Result<Array, WeightRecipeError>;
    fn prepare_materialization(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
    ) -> Result<PendingWeightRecipe, WeightRecipeError>;
    fn prepare_borrowed_materialization(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
    ) -> Result<PendingWeightRecipe, WeightRecipeError>;
    fn prepare_materialization_mode(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
        borrow_sources: bool,
    ) -> Result<PendingWeightRecipe, WeightRecipeError>;
    fn materialize_inner(
        &self,
        store: &dyn CheckpointSource,
        stream: &Stream,
        sources: &mut Vec<PendingWeightMaterialization>,
        borrow_sources: bool,
        context: &MlxParameterMaterializationContext,
    ) -> Result<Array, WeightRecipeError>;
}

impl MlxWeightRecipeExt for DerivedWeightRecipe {
    ///
    /// Source leases remain live until their dependent output has been
    /// evaluated. If a multi-input join reaches the mapping bound, completed
    /// children are detached before retrying so cross-shard recipes can honor
    /// a one-mapping limit without serializing the normal batched path.
    #[cfg(test)]
    fn materialize(
        &self,
        store: &dyn CheckpointSource,
        source_stream: &Stream,
    ) -> Result<Array, WeightRecipeError> {
        let context = MlxParameterMaterializationContext::new(source_stream, source_stream);
        self.prepare_materialization(store, &context)?.finish()
    }

    /// Schedules a recipe while retaining all mmap-backed source selections.
    fn prepare_materialization(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
    ) -> Result<PendingWeightRecipe, WeightRecipeError> {
        self.prepare_materialization_mode(store, context, false)
    }

    /// Schedules a recipe whose bounded checkpoint sources may remain borrowed
    /// until the containing output is evaluated.
    fn prepare_borrowed_materialization(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
    ) -> Result<PendingWeightRecipe, WeightRecipeError> {
        self.prepare_materialization_mode(store, context, true)
    }

    fn prepare_materialization_mode(
        &self,
        store: &dyn eredu_checkpoint::store::CheckpointSource,
        context: &MlxParameterMaterializationContext,
        borrow_sources: bool,
    ) -> Result<PendingWeightRecipe, WeightRecipeError> {
        self.infer(store)?;
        let mut sources = Vec::new();
        let source_stream = context.source_stream();
        let output =
            self.materialize_inner(store, source_stream, &mut sources, borrow_sources, context)?;
        // Derived recipe outputs are immutable and may be reused across forwards.
        // Detach gathers/transposes into their final row-major representation
        // once here so consumers do not silently repack a full weight on every
        // kernel invocation.
        let output = contiguous(output, false, source_stream)?;
        Ok(PendingWeightRecipe { output, sources })
    }

    fn materialize_inner(
        &self,
        store: &dyn CheckpointSource,
        stream: &Stream,
        sources: &mut Vec<PendingWeightMaterialization>,
        borrow_sources: bool,
        context: &MlxParameterMaterializationContext,
    ) -> Result<Array, WeightRecipeError> {
        match self {
            Self::Source { key, selection } => {
                let lease = store
                    .acquire_lease(TensorReadRequest {
                        key: key.clone(),
                        selection: selection.clone(),
                        policy: ReadPolicy::RequireBounded,
                    })
                    .map_err(
                        crate::backend::mlx::runtime::checkpoint::store::neutral_store_error,
                    )?;
                let lease = context.weight_lease(lease)?;
                let pending = if borrow_sources {
                    lease.prepare_borrowed_materialization(stream)?
                } else {
                    lease.prepare_materialization(stream, stream)?
                };
                let array = pending.output().clone();
                sources.push(pending);
                Ok(array)
            }
            Self::Select { input, selection } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                match selection {
                    TensorSelection::Full => Ok(array),
                    TensorSelection::Range { axis, start, end } => {
                        let indices = (*start..*end)
                            .map(|index| usize_to_i32(index, "selection index"))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(array.take_axis(
                            Array::from_slice(&indices, &[indices.len() as i32]),
                            usize_to_i32(*axis, "selection axis")?,
                            stream,
                        )?)
                    }
                    TensorSelection::Indices { axis, indices } => {
                        let indices = indices
                            .iter()
                            .map(|index| usize_to_i32(*index, "selection index"))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(array.take_axis(
                            Array::from_slice(&indices, &[indices.len() as i32]),
                            usize_to_i32(*axis, "selection axis")?,
                            stream,
                        )?)
                    }
                    TensorSelection::Contiguous {
                        offset_elements,
                        shape,
                    } => {
                        let elements = shape.iter().try_fold(1usize, |count, dimension| {
                            count.checked_mul(*dimension).ok_or(
                                WeightRecipeError::ArithmeticOverflow(
                                    "contiguous recipe selection size",
                                ),
                            )
                        })?;
                        let indices = (*offset_elements..offset_elements + elements)
                            .map(|index| usize_to_i32(index, "contiguous selection index"))
                            .collect::<Result<Vec<_>, _>>()?;
                        let flattened = array.reshape(&[-1], stream)?;
                        let selected = flattened.take_axis(
                            Array::from_slice(&indices, &[indices.len() as i32]),
                            0,
                            stream,
                        )?;
                        let shape = shape
                            .iter()
                            .map(|dimension| usize_to_i32(*dimension, "contiguous selection shape"))
                            .collect::<Result<Vec<_>, _>>()?;
                        Ok(selected.reshape(&shape, stream)?)
                    }
                }
            }
            Self::Concatenate { axis, inputs } => {
                let arrays =
                    materialize_inputs(inputs, store, stream, sources, borrow_sources, context)?;
                let references = arrays.iter().collect::<Vec<_>>();
                Ok(concatenate_axis(
                    &references,
                    usize_to_i32(*axis, "concatenate axis")?,
                    stream,
                )?)
            }
            Self::Stack { axis, inputs } => {
                let arrays =
                    materialize_inputs(inputs, store, stream, sources, borrow_sources, context)?;
                let references = arrays.iter().collect::<Vec<_>>();
                Ok(stack_axis(
                    &references,
                    usize_to_i32(*axis, "stack axis")?,
                    stream,
                )?)
            }
            Self::Reshape { input, shape } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                let shape = shape
                    .iter()
                    .map(|dimension| usize_to_i32(*dimension, "reshape dimension"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.reshape(&shape, stream)?)
            }
            Self::Transpose { input, axes } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                let axes = axes
                    .iter()
                    .map(|axis| usize_to_i32(*axis, "transpose axis"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.transpose_axes(&axes, stream)?)
            }
            Self::Cast { input, dtype } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                Ok(array.as_dtype(mlx_dtype(dtype)?, stream)?)
            }
            Self::View {
                input,
                dtype,
                shape,
            } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                let shape = shape
                    .iter()
                    .map(|dimension| usize_to_i32(*dimension, "view dimension"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array
                    .view_dtype(mlx_dtype(dtype)?, stream)?
                    .reshape(&shape, stream)?)
            }
            Self::NegLog { input } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                let all_negative = array
                    .lt(Array::from_f32(0.0), stream)?
                    .all(false, stream)?
                    .item::<bool>(stream);
                if !all_negative {
                    return Err(WeightRecipeError::NonNegativeNegLogInput);
                }
                Ok(array.multiply(Array::from_f32(-1.0), stream)?.log(stream)?)
            }
            Self::SubtractOne { input } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                Ok(array.subtract(Array::from_f32(1.0), stream)?)
            }
        }
    }
}

pub struct PendingWeightRecipe {
    output: Array,
    sources: Vec<PendingWeightMaterialization>,
}

impl PendingWeightRecipe {
    pub fn into_parts(self) -> (Array, Vec<PendingWeightMaterialization>) {
        (self.output, self.sources)
    }

    #[cfg(test)]
    fn finish(self) -> Result<Array, WeightRecipeError> {
        async_eval_with_event([&self.output])?.synchronize()?;
        for source in self.sources {
            source.complete();
        }
        Ok(self.output)
    }
}

fn materialize_inputs(
    inputs: &[DerivedWeightRecipe],
    store: &dyn CheckpointSource,
    stream: &Stream,
    sources: &mut Vec<PendingWeightMaterialization>,
    borrow_sources: bool,
    context: &MlxParameterMaterializationContext,
) -> Result<Vec<Array>, WeightRecipeError> {
    let mut pending =
        Vec::<(Array, Vec<PendingWeightMaterialization>)>::with_capacity(inputs.len());
    let mut detach_remaining = false;
    for input in inputs {
        loop {
            let mut input_sources = Vec::new();
            match input.materialize_inner(
                store,
                stream,
                &mut input_sources,
                borrow_sources,
                context,
            ) {
                Ok(array) => {
                    if detach_remaining && !input_sources.is_empty() {
                        async_eval_with_event([&array])?.synchronize()?;
                        for source in input_sources.drain(..) {
                            source.complete();
                        }
                    }
                    pending.push((array, input_sources));
                    break;
                }
                Err(error)
                    if !borrow_sources
                        && !detach_remaining
                        && !pending.is_empty()
                        && matches!(
                            &error,
                            WeightRecipeError::WeightStore(
                                WeightStoreError::CapacityExhausted { .. }
                            )
                        ) =>
                {
                    // The current child could not acquire another shard while
                    // earlier children pinned the mapping cache. Their arrays
                    // are sufficient evaluation roots, so detach them and retry.
                    drop(input_sources);
                    for (array, child_sources) in &mut pending {
                        if child_sources.is_empty() {
                            continue;
                        }
                        async_eval_with_event([&*array])?.synchronize()?;
                        for source in child_sources.drain(..) {
                            source.complete();
                        }
                    }
                    detach_remaining = true;
                }
                Err(error) => return Err(error),
            }
        }
    }
    let mut arrays = Vec::with_capacity(pending.len());
    for (array, input_sources) in pending {
        arrays.push(array);
        sources.extend(input_sources);
    }
    Ok(arrays)
}

fn usize_to_i32(value: usize, context: &'static str) -> Result<i32, WeightRecipeError> {
    i32::try_from(value).map_err(|_| WeightRecipeError::ArithmeticOverflow(context))
}

/// Structured validation and materialization failures for derived weights.
#[derive(Debug, thiserror::Error)]
pub enum WeightRecipeError {
    /// Backend-neutral recipe validation or shape inference failed.
    #[error(transparent)]
    Neutral(#[from] eredu_checkpoint::recipe::RecipeError),
    /// A source key was empty.
    #[error("derived-weight source key must not be empty")]
    EmptySourceKey,
    /// A source selection axis was outside the tensor rank.
    #[error("selection axis {axis} is outside rank {rank}")]
    InvalidSelectionAxis {
        /// Requested axis.
        axis: usize,
        /// Source rank.
        rank: usize,
    },
    /// A source range was invalid.
    #[error("range {start}..{end} is invalid for axis {axis} dimension {dimension}")]
    InvalidRange {
        /// Requested axis.
        axis: usize,
        /// Inclusive start.
        start: usize,
        /// Exclusive end.
        end: usize,
        /// Source dimension.
        dimension: usize,
    },
    /// An ordered-index selection was empty or out of bounds.
    #[error("ordered indices are empty or outside axis {axis} dimension {dimension}")]
    InvalidIndices {
        /// Requested axis.
        axis: usize,
        /// Source dimension.
        dimension: usize,
    },
    /// Concatenate or stack had no children.
    #[error("concatenate and stack recipes require at least one input")]
    EmptyInputs,
    /// Child dtypes did not agree.
    #[error("derived-weight inputs have different dtypes")]
    DtypeMismatch,
    /// Child shapes were incompatible.
    #[error("derived-weight inputs have incompatible shapes")]
    ShapeMismatch,
    /// A concatenate or stack axis was outside the accepted range.
    #[error("axis {axis} is invalid for rank {rank} (stack={stack})")]
    InvalidJoinAxis {
        /// Requested axis.
        axis: usize,
        /// Child rank.
        rank: usize,
        /// Whether the operation was stack instead of concatenate.
        stack: bool,
    },
    /// A reshape changed the element count.
    #[error("reshape changes element count from {input} to {output}")]
    ElementCountMismatch {
        /// Input element count.
        input: u64,
        /// Requested output element count.
        output: u64,
    },
    /// A bitwise view changed the number of represented bytes.
    #[error("bitwise view changes byte count from {input} to {output}")]
    ByteCountMismatch {
        /// Input byte count.
        input: u64,
        /// Requested output byte count.
        output: u64,
    },
    /// A transpose was not a rank-sized permutation.
    #[error("axes {axes:?} are not a permutation of rank {rank}")]
    InvalidPermutation {
        /// Requested axis order.
        axes: Vec<usize>,
        /// Child rank.
        rank: usize,
    },
    /// The inferred output contains no bytes.
    #[error("derived-weight output must contain at least one byte")]
    ZeroSizedOutput,
    /// A stored encoding has no known runtime byte width.
    #[error("derived-weight dtype {dtype} is unsupported")]
    UnsupportedDtype {
        /// Debug name of the unsupported encoding.
        dtype: String,
    },
    /// MLX MXFP4 lowering received a recipe other than a logical F4 view.
    #[error("MLX MXFP4 lowering requires a terminal logical F4 view, got {dtype:?}")]
    ExpectedLogicalMxFp4 {
        /// Actual recipe dtype.
        dtype: RecipeDtype,
    },
    /// A logical MXFP4 value shape cannot be represented by packed U32 units.
    #[error("logical MXFP4 shape must have a nonzero final dimension divisible by 8")]
    InvalidMxFp4LogicalShape,
    /// Checked shape or byte arithmetic overflowed.
    #[error("derived-weight arithmetic overflow: {0}")]
    ArithmeticOverflow(&'static str),
    /// A selected derived output could not be represented by bounded source reads.
    #[error("cannot push bounded selection through {operation}: {reason}")]
    SelectionPushdownUnsupported {
        /// Recipe operation that could not preserve the selection.
        operation: &'static str,
        /// Checked geometry that prevented the rewrite.
        reason: String,
    },
    /// A transition-rate normalization contained zero or a positive value.
    #[error("log(-x) derived-weight input must contain only negative values")]
    NonNegativeNegLogInput,
    /// Checkpoint storage failed.
    #[error(transparent)]
    WeightStore(#[from] crate::backend::mlx::runtime::checkpoint::store::WeightStoreError),
    /// MLX transformation or synchronization failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use safemlx::{Device, DeviceType};
    use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};

    use super::*;
    use crate::backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore;

    fn fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        let left = [1i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let right = [5i32, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let cube = [0i32; 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                (
                    "left",
                    TensorView::new(SafeDtype::I32, vec![2, 2], &left).unwrap(),
                ),
                (
                    "right",
                    TensorView::new(SafeDtype::I32, vec![2, 2], &right).unwrap(),
                ),
                (
                    "cube",
                    TensorView::new(SafeDtype::I32, vec![2, 2, 2], &cube).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
        (dir, store)
    }

    #[test]
    fn bitwise_view_preserves_checkpoint_bytes() {
        let (_dir, store) = fixture();
        let context =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let recipe = DerivedWeightRecipe::View {
            input: Box::new(DerivedWeightRecipe::source("left", TensorSelection::Full)),
            dtype: RecipeDtype::U8,
            shape: vec![2, 8],
        };
        let metadata = recipe.infer(store.as_ref()).unwrap();
        assert_eq!(metadata.shape(), &[2, 8]);
        assert_eq!(metadata.dtype(), &RecipeDtype::U8);
        let output = recipe
            .materialize(store.as_ref(), context.stream())
            .unwrap();
        assert_eq!(
            output.evaluated().unwrap().as_slice::<u8>(),
            &[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0]
        );
    }

    #[test]
    fn lowers_logical_mxfp4_values_to_mlx_u32_storage() {
        let (_dir, store) = fixture();
        let logical = DerivedWeightRecipe::View {
            input: Box::new(DerivedWeightRecipe::source("left", TensorSelection::Full)),
            dtype: RecipeDtype::F4,
            shape: vec![1, 32],
        };

        let lowered = lower_mxfp4_recipe(logical, store.as_ref()).unwrap();
        let metadata = lowered.infer(store.as_ref()).unwrap();
        assert_eq!(metadata.shape(), &[1, 4]);
        assert_eq!(metadata.dtype(), &RecipeDtype::U32);
        assert!(matches!(
            lowered,
            DerivedWeightRecipe::View {
                dtype: RecipeDtype::U32,
                ref shape,
                ..
            } if shape == &[1, 4]
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "requires local MLX Metal execution"]
    fn mlx_lowered_mxfp4_recipe_materializes_u32_storage() {
        let (_dir, store) = fixture();
        let context =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let logical = DerivedWeightRecipe::View {
            input: Box::new(DerivedWeightRecipe::source("left", TensorSelection::Full)),
            dtype: RecipeDtype::F4,
            shape: vec![1, 32],
        };
        let lowered = lower_mxfp4_recipe(logical, store.as_ref()).unwrap();

        let output = lowered
            .materialize(store.as_ref(), context.stream())
            .unwrap();
        assert_eq!(output.shape(), &[1, 4]);
        assert_eq!(output.dtype(), Dtype::Uint32);
        assert_eq!(output.evaluated().unwrap().as_slice::<u32>(), &[1, 2, 3, 4]);
    }

    fn one_mapping_cross_shard_fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
        let dir = tempfile::tempdir().unwrap();
        let left = [1i32, 2, 3, 4]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let right = [5i32, 6, 7, 8]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [(
                "left",
                TensorView::new(SafeDtype::I32, vec![2, 2], &left).unwrap(),
            )],
            None,
            &dir.path().join("model-00001-of-00002.safetensors"),
        )
        .unwrap();
        serialize_to_file(
            [(
                "right",
                TensorView::new(SafeDtype::I32, vec![2, 2], &right).unwrap(),
            )],
            None,
            &dir.path().join("model-00002-of-00002.safetensors"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({
                "weight_map": {
                    "left": "model-00001-of-00002.safetensors",
                    "right": "model-00002-of-00002.safetensors"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let store =
            Arc::new(SafetensorsWeightStore::open_with_max_mapped_shards(dir.path(), 1).unwrap());
        (dir, store)
    }

    fn source(key: &str) -> DerivedWeightRecipe {
        DerivedWeightRecipe::source(key, TensorSelection::Full)
    }

    #[test]
    fn conservative_peak_counts_retained_inputs_and_outputs() {
        let (_directory, store) = fixture();
        let reshape = DerivedWeightRecipe::Reshape {
            input: Box::new(source("left")),
            shape: vec![4],
        };
        assert_eq!(
            reshape.peak_materialization_bytes(store.as_ref()).unwrap(),
            32
        );

        let concatenate = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![source("left"), source("right")],
        };
        assert_eq!(
            concatenate
                .peak_materialization_bytes(store.as_ref())
                .unwrap(),
            64
        );
    }

    #[test]
    fn infers_nested_stack_concatenate_slice_and_cast() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Cast {
            input: Box::new(DerivedWeightRecipe::Stack {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![source("left"), source("right")],
                    },
                    DerivedWeightRecipe::Concatenate {
                        axis: 0,
                        inputs: vec![
                            DerivedWeightRecipe::source(
                                "left",
                                TensorSelection::Indices {
                                    axis: 0,
                                    indices: vec![1, 0],
                                },
                            ),
                            source("right"),
                        ],
                    },
                ],
            }),
            dtype: RecipeDtype::F32,
        };
        let metadata = recipe.infer(store.as_ref()).unwrap();
        assert_eq!(metadata.shape(), &[2, 4, 2]);
        assert_eq!(metadata.dtype(), &RecipeDtype::F32);
        assert_eq!(metadata.byte_len(), 64);
        assert_eq!(recipe.source_keys(), vec!["left", "right"]);
    }

    #[test]
    fn materializes_ordered_expert_stack_on_cpu() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![source("right"), source("left")],
        };
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = recipe.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(output.shape(), &[2, 2, 2]);
        assert_eq!(output.nbytes(), 32);
        assert_eq!(
            output.evaluated().unwrap().as_slice::<i32>(),
            &[5, 6, 7, 8, 1, 2, 3, 4]
        );
    }

    #[test]
    fn materializes_cross_shard_join_with_one_mapping() {
        let (_dir, store) = one_mapping_cross_shard_fixture();
        let recipe = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![source("left"), source("right")],
            }],
        };
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = recipe.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(output.shape(), &[1, 4, 2]);
        assert_eq!(
            output.evaluated().unwrap().as_slice::<i32>(),
            &[1, 2, 3, 4, 5, 6, 7, 8]
        );
        let diagnostics = store.source_diagnostics().unwrap();
        assert_eq!(diagnostics.currently_mapped_shards, 1);
        assert_eq!(diagnostics.touched_shard_paths.len(), 2);
        assert!(diagnostics.evictions >= 1);
    }

    #[test]
    fn selects_an_axis_from_a_derived_source_selection() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Select {
            input: Box::new(DerivedWeightRecipe::source(
                "left",
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )),
            selection: TensorSelection::Indices {
                axis: 1,
                indices: vec![1],
            },
        };
        let metadata = recipe.infer(store.as_ref()).unwrap();
        assert_eq!(metadata.shape(), &[1, 1]);
        assert_eq!(recipe.source_keys(), vec!["left"]);
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = recipe.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(output.evaluated().unwrap().as_slice::<i32>(), &[2]);
    }

    #[test]
    fn materialized_recipe_selection_has_row_major_storage() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Select {
            input: Box::new(source("cube")),
            selection: TensorSelection::Indices {
                axis: 1,
                indices: vec![1, 0],
            },
        };
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = recipe.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(output.shape(), &[2, 2, 2]);
        assert_eq!(output.strides(), &[4, 2, 1]);
        output.evaluated().unwrap();
    }

    #[test]
    fn bounded_selection_composes_at_the_source() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::source(
            "left",
            TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0],
            },
        );
        let rewritten = recipe
            .select_bounded(
                store.as_ref(),
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )
            .unwrap();
        assert_eq!(
            rewritten,
            DerivedWeightRecipe::source(
                "left",
                TensorSelection::Range {
                    axis: 0,
                    start: 1,
                    end: 2,
                }
            )
        );
    }

    #[test]
    fn independent_axis_selections_collapse_to_one_contiguous_source_span() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Select {
            input: Box::new(DerivedWeightRecipe::source(
                "cube",
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )),
            selection: TensorSelection::Range {
                axis: 1,
                start: 0,
                end: 1,
            },
        };
        let rewritten = recipe
            .select_bounded(
                store.as_ref(),
                TensorSelection::Range {
                    axis: 2,
                    start: 0,
                    end: 1,
                },
            )
            .unwrap();
        assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[1, 1, 1]);
        assert_eq!(
            rewritten,
            DerivedWeightRecipe::source(
                "cube",
                TensorSelection::Contiguous {
                    offset_elements: 0,
                    shape: vec![1, 1, 1],
                }
            )
        );
    }

    #[test]
    fn bounded_selection_prunes_and_reorders_join_children() {
        let (_dir, store) = fixture();
        let concatenate = DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![source("left"), source("right")],
        };
        let rewritten = concatenate
            .select_bounded(
                store.as_ref(),
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![3, 0, 2],
                },
            )
            .unwrap();
        assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[3, 2]);
        assert_eq!(
            rewritten,
            DerivedWeightRecipe::Concatenate {
                axis: 0,
                inputs: vec![
                    DerivedWeightRecipe::source(
                        "right",
                        TensorSelection::Range {
                            axis: 0,
                            start: 1,
                            end: 2,
                        },
                    ),
                    DerivedWeightRecipe::source(
                        "left",
                        TensorSelection::Range {
                            axis: 0,
                            start: 0,
                            end: 1,
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
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let output = rewritten.materialize(store.as_ref(), &stream).unwrap();
        assert_eq!(
            output.evaluated().unwrap().as_slice::<i32>(),
            &[7, 8, 1, 2, 5, 6]
        );

        let stack = DerivedWeightRecipe::Stack {
            axis: 0,
            inputs: vec![source("left"), source("right")],
        };
        let rewritten = stack
            .select_bounded(
                store.as_ref(),
                TensorSelection::Indices {
                    axis: 0,
                    indices: vec![1, 0, 1],
                },
            )
            .unwrap();
        assert_eq!(rewritten.source_keys(), vec!["left", "right"]);
        assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[3, 2, 2]);
        assert!(matches!(
            rewritten,
            DerivedWeightRecipe::Stack { inputs, .. }
                if inputs == vec![source("right"), source("left"), source("right")]
        ));
    }

    #[test]
    fn bounded_selection_crosses_transpose_reshape_and_view() {
        let (_dir, store) = fixture();
        let transpose = DerivedWeightRecipe::Transpose {
            input: Box::new(source("left")),
            axes: vec![1, 0],
        };
        let rewritten = transpose
            .select_bounded(
                store.as_ref(),
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 1,
                },
            )
            .unwrap();
        assert!(matches!(
            rewritten,
            DerivedWeightRecipe::Transpose { input, .. }
                if matches!(
                    *input,
                    DerivedWeightRecipe::Source {
                        selection: TensorSelection::Range {
                            axis: 1,
                            start: 0,
                            end: 1,
                        },
                        ..
                    }
                )
        ));

        let reshape = DerivedWeightRecipe::Reshape {
            input: Box::new(source("left")),
            shape: vec![4],
        };
        let rewritten = reshape
            .select_bounded(
                store.as_ref(),
                TensorSelection::Range {
                    axis: 0,
                    start: 0,
                    end: 2,
                },
            )
            .unwrap();
        assert!(matches!(
            rewritten,
            DerivedWeightRecipe::Reshape { input, shape }
                if shape == vec![2]
                    && matches!(
                        *input,
                        DerivedWeightRecipe::Source {
                            selection: TensorSelection::Range {
                                axis: 0,
                                start: 0,
                                end: 1,
                            },
                            ..
                        }
                    )
        ));

        let view = DerivedWeightRecipe::View {
            input: Box::new(source("left")),
            dtype: RecipeDtype::U8,
            shape: vec![2, 8],
        };
        let rewritten = view
            .select_bounded(
                store.as_ref(),
                TensorSelection::Range {
                    axis: 1,
                    start: 0,
                    end: 4,
                },
            )
            .unwrap();
        assert!(matches!(
            rewritten,
            DerivedWeightRecipe::View { input, shape, .. }
                if shape == vec![2, 4]
                    && matches!(
                        *input,
                        DerivedWeightRecipe::Source {
                            selection: TensorSelection::Range {
                                axis: 1,
                                start: 0,
                                end: 1,
                            },
                            ..
                        }
                    )
        ));
    }

    #[test]
    fn bounded_selection_splits_a_stacked_expert_reshape_at_sources() {
        let (_dir, store) = fixture();
        let recipe = DerivedWeightRecipe::Reshape {
            input: Box::new(DerivedWeightRecipe::Stack {
                axis: 2,
                inputs: vec![source("cube"), source("cube")],
            }),
            shape: vec![2, 4, 2],
        };
        let rewritten = recipe
            .select_bounded(
                store.as_ref(),
                TensorSelection::Range {
                    axis: 1,
                    start: 2,
                    end: 4,
                },
            )
            .unwrap();
        assert_eq!(rewritten.infer(store.as_ref()).unwrap().shape(), &[2, 2, 2]);
        assert!(matches!(
            rewritten,
            DerivedWeightRecipe::Reshape { input, .. }
                if matches!(
                    &*input,
                    DerivedWeightRecipe::Stack { inputs, .. }
                        if inputs.iter().all(|input| matches!(
                            input,
                            DerivedWeightRecipe::Source {
                                selection: TensorSelection::Range {
                                    axis: 1,
                                    start: 1,
                                    end: 2,
                                },
                                ..
                            }
                        ))
                )
        ));
    }

    #[test]
    fn bounded_selection_rejects_unaligned_view_geometry() {
        let (_dir, store) = fixture();
        let view = DerivedWeightRecipe::View {
            input: Box::new(source("left")),
            dtype: RecipeDtype::U8,
            shape: vec![2, 8],
        };
        assert!(matches!(
            view.select_bounded(
                store.as_ref(),
                TensorSelection::Range {
                    axis: 1,
                    start: 1,
                    end: 2,
                },
            ),
            Err(
                eredu_checkpoint::recipe::RecipeError::SelectionPushdownUnsupported {
                    operation: "view",
                    ..
                }
            )
        ));
    }

    #[test]
    fn rejects_shape_and_permutation_errors_before_materialization() {
        let (_dir, store) = fixture();
        let reshape = DerivedWeightRecipe::Reshape {
            input: Box::new(source("left")),
            shape: vec![3],
        };
        assert!(matches!(
            reshape.infer(store.as_ref()),
            Err(eredu_checkpoint::recipe::RecipeError::ElementCountMismatch { .. })
        ));
        let transpose = DerivedWeightRecipe::Transpose {
            input: Box::new(source("left")),
            axes: vec![0, 0],
        };
        assert!(matches!(
            transpose.infer(store.as_ref()),
            Err(eredu_checkpoint::recipe::RecipeError::InvalidPermutation { .. })
        ));
    }

    #[test]
    fn neg_log_materializes_negative_rates_and_rejects_nonnegative_values() {
        let dir = tempfile::tempdir().unwrap();
        let negative = [-1.0f32, -4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let invalid = [-1.0f32, 0.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                (
                    "negative",
                    TensorView::new(SafeDtype::F32, vec![2], &negative).unwrap(),
                ),
                (
                    "invalid",
                    TensorView::new(SafeDtype::F32, vec![2], &invalid).unwrap(),
                ),
            ],
            None,
            &dir.path().join("model.safetensors"),
        )
        .unwrap();
        let store = SafetensorsWeightStore::open(dir.path()).unwrap();
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        let recipe = |key| DerivedWeightRecipe::NegLog {
            input: Box::new(source(key)),
        };

        let output = recipe("negative").materialize(&store, &stream).unwrap();
        let output = output.evaluated().unwrap();
        assert_eq!(output.as_slice::<f32>()[0], 0.0);
        assert!((output.as_slice::<f32>()[1] - 4.0f32.ln()).abs() < 1e-6);
        assert!(matches!(
            recipe("invalid").materialize(&store, &stream),
            Err(WeightRecipeError::NonNegativeNegLogInput)
        ));
    }
}
