//! Deterministic checkpoint-derived weight recipes.

//!
//! Recipes describe the runtime representation of a parameter without tying it
//! to a single checkpoint key. They are validated from checkpoint metadata and
//! materialized on the residency source stream before device promotion.

use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype};
use eredu_checkpoint::store::{CheckpointSource, ReadPolicy, TensorReadRequest, TensorSelection};

use safemlx::{
    ops::{concatenate_axis, contiguous, stack_axis},
    Array, Dtype, Stream,
};

use crate::backend::runtime::checkpoint::store::{
    MlxParameterMaterializationContext, PendingWeightMaterialization, WeightMaterialization,
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

/// Validates that MLX can represent every source and intermediate value in a
/// neutral recipe without acquiring payloads or constructing native arrays.
pub(crate) fn preflight_mlx_recipe(
    recipe: &DerivedWeightRecipe,
    source: &dyn CheckpointSource,
) -> Result<(), WeightRecipeError> {
    fn validate_selection(selection: &TensorSelection) -> Result<(), WeightRecipeError> {
        match selection {
            TensorSelection::Full => {}
            TensorSelection::Range { axis, start, end } => {
                usize_to_i32(*axis, "selection axis")?;
                usize_to_i32(*start, "selection start")?;
                usize_to_i32(end.saturating_sub(1), "selection end")?;
            }
            TensorSelection::Indices { axis, indices } => {
                usize_to_i32(*axis, "selection axis")?;
                for index in indices {
                    usize_to_i32(*index, "selection index")?;
                }
            }
            TensorSelection::Contiguous {
                offset_elements,
                shape,
            } => {
                let elements = shape.iter().try_fold(1usize, |count, dimension| {
                    count
                        .checked_mul(*dimension)
                        .ok_or(WeightRecipeError::ArithmeticOverflow(
                            "contiguous recipe selection size",
                        ))
                })?;
                let last = offset_elements
                    .checked_add(elements.saturating_sub(1))
                    .ok_or(WeightRecipeError::ArithmeticOverflow(
                        "contiguous recipe selection end",
                    ))?;
                usize_to_i32(last, "contiguous selection index")?;
            }
        }
        Ok(())
    }

    // MLX representation limits here are constant and contain no device facts.
    struct MlxRecipeRepresentation;
    fn visit(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<(), WeightRecipeError> {
        if let Some(cache) = source.recipe_cache() {
            cache
                .validate::<MlxRecipeRepresentation>(recipe, || {
                    visit_uncached(recipe, source).map_err(|error| error.to_string())
                })
                .map_err(WeightRecipeError::Preflight)
        } else {
            visit_uncached(recipe, source)
        }
    }

    fn visit_uncached(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<(), WeightRecipeError> {
        let metadata = recipe.infer(source)?;
        for dimension in metadata.shape() {
            usize_to_i32(*dimension, "recipe dimension")?;
        }
        mlx_dtype(metadata.dtype())?;

        match recipe {
            DerivedWeightRecipe::Source { key, selection } => {
                validate_selection(selection)?;
                let source_metadata = source.source_metadata(key)?;
                for dimension in source_metadata
                    .logical_shape
                    .iter()
                    .chain(&source_metadata.physical_shape)
                {
                    usize_to_i32(*dimension, "checkpoint source dimension")?;
                }
                usize::try_from(source_metadata.encoded_byte_len).map_err(|_| {
                    WeightRecipeError::ArithmeticOverflow("checkpoint source byte length")
                })?;
                let provenance = source.source_provenance(key)?;
                match provenance.source_encoding {
                    eredu_checkpoint::SourceTensorEncoding::Safetensors(dtype) => {
                        super::store::safetensors_dtype(key, &dtype)?;
                    }
                    eredu_checkpoint::SourceTensorEncoding::RecipeOutput(dtype) => {
                        mlx_dtype(&dtype.into())?;
                    }
                    eredu_checkpoint::SourceTensorEncoding::Gguf { .. } => {}
                    encoding => {
                        return Err(WeightRecipeError::UnsupportedDtype {
                            dtype: format!("{encoding:?}"),
                        });
                    }
                }
            }
            DerivedWeightRecipe::Concatenate { axis, inputs }
            | DerivedWeightRecipe::Stack { axis, inputs } => {
                usize_to_i32(*axis, "recipe join axis")?;
                for input in inputs {
                    visit(input, source)?;
                }
            }
            DerivedWeightRecipe::Select { input, selection } => {
                validate_selection(selection)?;
                visit(input, source)?;
            }
            DerivedWeightRecipe::Transpose { input, axes } => {
                for axis in axes {
                    usize_to_i32(*axis, "transpose axis")?;
                }
                visit(input, source)?;
            }
            DerivedWeightRecipe::Cast { input, dtype }
            | DerivedWeightRecipe::View { input, dtype, .. } => {
                mlx_dtype(dtype)?;
                visit(input, source)?;
            }
            DerivedWeightRecipe::Reshape { input, .. }
            | DerivedWeightRecipe::NegLog { input }
            | DerivedWeightRecipe::SubtractOne { input } => visit(input, source)?,
        }
        Ok(())
    }

    visit(recipe, source)
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
    /// Materializes the recipe synchronously for tests.
    fn materialize(
        &self,
        store: &dyn CheckpointSource,
        source_stream: &Stream,
    ) -> Result<Array, WeightRecipeError>;
    /// Prepares owned checkpoint sources and submits the recipe transformations.
    fn prepare_materialization(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
    ) -> Result<PendingWeightRecipe, WeightRecipeError>;
    /// Prepares borrowed checkpoint sources and submits the recipe transformations.
    fn prepare_borrowed_materialization(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
    ) -> Result<PendingWeightRecipe, WeightRecipeError>;
    /// Prepares a recipe using the selected source ownership mode.
    fn prepare_materialization_mode(
        &self,
        store: &dyn CheckpointSource,
        context: &MlxParameterMaterializationContext,
        borrow_sources: bool,
    ) -> Result<PendingWeightRecipe, WeightRecipeError>;
    /// Recursively lowers a recipe and retains all pending source operations.
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
    /// evaluated. If a multi-input join reaches the shard-cache bound, completed
    /// children are detached before retrying so cross-shard recipes can honor
    /// a one-shard cache limit without serializing the normal batched path.
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
        // On CPU and unified-memory Metal, the allocation filled by checkpoint
        // I/O is already suitable for execution. Recipes own the layout plan;
        // the native wrapper owns allocation and publication of immutable data.
        #[cfg(not(feature = "cuda"))]
        if let Some(read) = self.prepare_encoded_read(store)? {
            // Byte-preserving joins already proved a single common dtype.
            if let Some(source) = read.sources().first() {
                super::store::safetensors_dtype(&source.name, &source.stored_dtype)?;
            }
            let mut shape = read
                .output()
                .shape
                .iter()
                .map(|dimension| usize_to_i32(*dimension, "direct recipe output shape"))
                .collect::<Result<Vec<_>, _>>()?;
            let dtype = mlx_dtype(&read.output().dtype)?;
            if read.output().dtype == RecipeDtype::F4 {
                let last = shape
                    .last_mut()
                    .filter(|last| **last % 2 == 0)
                    .ok_or(WeightRecipeError::InvalidMxFp4LogicalShape)?;
                *last /= 2;
            }
            let output = Array::try_init_with(&shape, dtype, |bytes| {
                read.read_into(bytes)
                    .map_err(WeightRecipeError::CheckpointStore)
            })?;
            return Ok(PendingWeightRecipe {
                output,
                sources: Vec::new(),
            });
        }
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
                let lease = store.acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: selection.clone(),
                    policy: ReadPolicy::RequireBounded,
                })?;
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
                let prepared =
                    WeightMaterialization::prepare_retained(vec![array], std::mem::take(sources))?;
                let all_negative = prepared.inputs()[0]
                    .lt(Array::from_f32(0.0), stream)?
                    .all(false, stream)?
                    .try_item::<bool>(stream)?;
                if !all_negative {
                    prepared.finish()?;
                    return Err(WeightRecipeError::NonNegativeNegLogInput);
                }
                let output = prepared.inputs()[0]
                    .multiply(Array::from_f32(-1.0), stream)?
                    .log(stream)?;
                sources.extend(prepared.finish_preparation()?);
                Ok(output)
            }
            Self::SubtractOne { input } => {
                let array =
                    input.materialize_inner(store, stream, sources, borrow_sources, context)?;
                Ok(array.subtract(Array::from_f32(1.0), stream)?)
            }
        }
    }
}

/// Submitted recipe output and the source operations it retains.
pub struct PendingWeightRecipe {
    output: Array,
    sources: Vec<PendingWeightMaterialization>,
}

impl PendingWeightRecipe {
    /// Separates the output array from its retained source operations.
    pub fn into_parts(self) -> (Array, Vec<PendingWeightMaterialization>) {
        (self.output, self.sources)
    }

    #[cfg(test)]
    fn finish(self) -> Result<Array, WeightRecipeError> {
        Ok(WeightMaterialization::submit_retained(self.output, self.sources)?.synchronize()?)
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
                        WeightMaterialization::submit_retained(
                            array.clone(),
                            std::mem::take(&mut input_sources),
                        )?
                        .synchronize()?;
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
                            WeightRecipeError::CheckpointStore(
                                eredu_checkpoint::store::StoreError::CapacityExhausted { .. }
                            )
                        ) =>
                {
                    // The current child could not acquire another shard while
                    // earlier children pinned the shard cache. Their arrays
                    // are sufficient evaluation roots, so detach them and retry.
                    drop(input_sources);
                    for (array, child_sources) in &mut pending {
                        if child_sources.is_empty() {
                            continue;
                        }
                        WeightMaterialization::submit_retained(
                            array.clone(),
                            std::mem::take(child_sources),
                        )?
                        .synchronize()?;
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
    /// Retained metadata-only MLX representation validation failure.
    #[error("{0}")]
    Preflight(String),
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
    /// Backend-neutral checkpoint inspection or lease acquisition failed.
    #[error(transparent)]
    CheckpointStore(#[from] eredu_checkpoint::store::StoreError),
    /// MLX checkpoint materialization failed.
    #[error(transparent)]
    CheckpointMaterialization(
        #[from] crate::backend::runtime::checkpoint::store::CheckpointMaterializationError,
    ),
    /// MLX transformation or synchronization failed.
    #[error(transparent)]
    Mlx(#[from] safemlx::error::Exception),
}

#[cfg(test)]
mod tests;
