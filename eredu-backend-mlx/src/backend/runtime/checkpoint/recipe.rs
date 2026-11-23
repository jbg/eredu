//! Deterministic checkpoint-derived weight recipes.

//!
//! Recipes describe the runtime representation of a parameter without tying it
//! to a single checkpoint key. They are validated from checkpoint metadata and
//! materialized on the residency source stream before device promotion.

use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeCatalog, RecipeDtype};
use eredu_checkpoint::store::{CheckpointLease, CheckpointSource, StoreError, TensorSelection};
use eredu_core::checkpoint_mechanisms::recipe_execution::{self, RecipeExecution};

use safemlx::{
    ops::{concatenate_axis, contiguous, stack_axis},
    Array, Dtype, Stream,
};

use crate::backend::runtime::checkpoint::store::{
    MlxParameterMaterializationContext, PendingWeightMaterialization, WeightMaterialization,
};

/// A byte-preserving recipe admitted for direct initialization of native storage.
#[doc(hidden)]
pub struct DirectRecipeRead {
    shape: Vec<i32>,
    dtype: Dtype,
    read: eredu_checkpoint::recipe::EncodedRecipeRead,
    transposed: bool,
    _prepared_source: Option<eredu_checkpoint::store::SharedCheckpointSource>,
}

impl DirectRecipeRead {
    pub(crate) fn for_binding(
        source: &dyn CheckpointSource,
        binding: &eredu_core::residency::parameter_binding::WeightBinding,
    ) -> Result<Option<Self>, WeightRecipeError> {
        Self::prepare_with_layout(&binding.recipe_view(), source, binding.storage_layout())
    }

    pub(crate) fn prepare(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
    ) -> Result<Option<Self>, WeightRecipeError> {
        #[cfg(feature = "cuda")]
        {
            let _ = (recipe, source);
            Ok(None)
        }
        #[cfg(not(feature = "cuda"))]
        {
            let Some(read) = recipe.prepare_encoded_read(source)? else {
                return Ok(None);
            };
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
            Ok(Some(Self {
                shape,
                dtype,
                read,
                transposed: false,
                _prepared_source: None,
            }))
        }
    }

    pub(crate) fn prepare_with_layout(
        recipe: &DerivedWeightRecipe,
        source: &dyn CheckpointSource,
        layout: eredu_nn::ParameterStorageLayout,
    ) -> Result<Option<Self>, WeightRecipeError> {
        if layout == eredu_nn::ParameterStorageLayout::TransposedMatrix && !cfg!(feature = "cuda") {
            if let Some(policy) = source.prepared_cache_policy() {
                if let Some(prepared) = eredu_checkpoint::prepared_cache::prepare_transposed_matrix(
                    source,
                    recipe,
                    policy,
                    64 * 1024 * 1024,
                )? {
                    let storage = DerivedWeightRecipe::source(
                        "matrix",
                        eredu_checkpoint::store::TensorSelection::Full,
                    );
                    if let Some(mut read) = Self::prepare(&storage, prepared.as_ref())? {
                        read.transposed = true;
                        read._prepared_source = Some(prepared);
                        return Ok(Some(read));
                    }
                }
            }
        }
        Self::prepare(recipe, source)
    }

    pub(crate) fn materialize_many(
        reads: Vec<Self>,
        stream: &Stream,
    ) -> Result<Vec<Array>, WeightRecipeError> {
        let layouts: Vec<_> = reads.iter().map(|r| r.transposed).collect();
        // Pins remain live through the direct encoded reads, including unwinding.
        let _pins: Vec<_> = reads
            .iter()
            .filter_map(|r| r._prepared_source.clone())
            .collect();
        let (specifications, reads): (Vec<_>, Vec<_>) = reads
            .into_iter()
            .map(|value| ((value.shape, value.dtype), value.read))
            .unzip();
        let specifications = specifications
            .iter()
            .map(|(shape, dtype)| (shape.as_slice(), *dtype))
            .collect::<Vec<_>>();
        let arrays = Array::try_init_many_with(&specifications, |outputs| {
            eredu_checkpoint::recipe::EncodedRecipeRead::read_many_into(reads, outputs)
                .map_err(WeightRecipeError::CheckpointStore)
        })?;
        arrays
            .into_iter()
            .zip(layouts)
            .map(|(array, transposed)| {
                if transposed {
                    Ok(array.transpose(stream)?)
                } else {
                    Ok(array)
                }
            })
            .collect()
    }
}

/// Materializes a recipe already classified as requiring the ordinary path.
/// Keeping that decision avoids preparing its encoded read a second time.
pub(crate) fn prepare_ordinary_recipe(
    recipe: &DerivedWeightRecipe,
    store: &dyn CheckpointSource,
    context: &MlxParameterMaterializationContext,
    borrow_sources: bool,
) -> Result<PendingWeightRecipe, WeightRecipeError> {
    recipe.infer(store)?;
    let mut sources = Vec::new();
    let source_stream = context.source_stream();
    let output =
        recipe.materialize_inner(store, source_stream, &mut sources, borrow_sources, context)?;
    // Derived recipe outputs are immutable and may be reused across forwards.
    // Detach gathers/transposes into their final row-major representation
    // once here so consumers do not silently repack a full weight on every
    // kernel invocation.
    let output = contiguous(output, false, source_stream)?;
    Ok(PendingWeightRecipe { output, sources })
}

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

pub(crate) fn mlx_dtype(value: &RecipeDtype) -> Result<Dtype, WeightRecipeError> {
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
    preflight_mlx_recipe_metadata(recipe, source, &|key| {
        Ok(source.source_provenance(key)?.source_encoding)
    })
}

/// Validates recipe representations using immutable headers, without a readable source.
pub(crate) fn preflight_mlx_recipe_metadata<C: RecipeCatalog + ?Sized>(
    recipe: &DerivedWeightRecipe,
    source: &C,
    encoding: &impl Fn(
        &str,
    ) -> Result<
        eredu_checkpoint::SourceTensorEncoding,
        eredu_checkpoint::store::StoreError,
    >,
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
    recipe_execution::preflight(
        recipe,
        source,
        &|node, metadata| {
            for dimension in metadata.shape() {
                usize_to_i32(*dimension, "recipe dimension")?;
            }
            mlx_dtype(metadata.dtype())?;

            match node {
                DerivedWeightRecipe::Source { key, selection } => {
                    validate_selection(selection)?;
                    let source_metadata = source.tensor_metadata(key)?;
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
                    match encoding(key)? {
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
                DerivedWeightRecipe::Concatenate { axis, .. }
                | DerivedWeightRecipe::Stack { axis, .. } => {
                    usize_to_i32(*axis, "recipe join axis")?;
                }
                DerivedWeightRecipe::Select { selection, .. } => {
                    validate_selection(selection)?;
                }
                DerivedWeightRecipe::Transpose { axes, .. } => {
                    for axis in axes {
                        usize_to_i32(*axis, "transpose axis")?;
                    }
                }
                DerivedWeightRecipe::Cast { dtype, .. }
                | DerivedWeightRecipe::View { dtype, .. } => {
                    mlx_dtype(dtype)?;
                }
                DerivedWeightRecipe::Reshape { .. }
                | DerivedWeightRecipe::NegLog { .. }
                | DerivedWeightRecipe::SubtractOne { .. }
                | DerivedWeightRecipe::DecodeTensorFp8 { .. }
                | DerivedWeightRecipe::ScaleF32 { .. } => {}
            }
            Ok(())
        },
        &WeightRecipeError::from,
        &|node, validate| match source.recipe_cache() {
            Some(cache) => cache
                .validate::<MlxRecipeRepresentation>(node, || {
                    validate().map_err(|error| error.to_string())
                })
                .map_err(WeightRecipeError::Preflight),
            None => validate(),
        },
    )
}

/// MLX lowering operations for a backend-neutral recipe.
pub trait MlxWeightRecipeExt {
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
    /// Realizes an ordinary recipe into the caller's native retention scope.
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
        if let Some(read) = DirectRecipeRead::prepare(self, store)? {
            let output = DirectRecipeRead::materialize_many(vec![read], context.source_stream())?
                .pop()
                .expect("one output");
            return Ok(PendingWeightRecipe {
                output,
                sources: Vec::new(),
            });
        }
        prepare_ordinary_recipe(self, store, context, borrow_sources)
    }

    fn materialize_inner(
        &self,
        store: &dyn CheckpointSource,
        stream: &Stream,
        sources: &mut Vec<PendingWeightMaterialization>,
        borrow_sources: bool,
        context: &MlxParameterMaterializationContext,
    ) -> Result<Array, WeightRecipeError> {
        let execution = MlxRecipeExecution {
            context,
            stream,
            borrow_sources,
        };
        recipe_execution::execute(self, store, &execution, sources)
    }
}

struct MlxRecipeExecution<'a> {
    context: &'a MlxParameterMaterializationContext,
    stream: &'a Stream,
    borrow_sources: bool,
}

impl RecipeExecution for MlxRecipeExecution<'_> {
    type Value = Array;
    type Retained = PendingWeightMaterialization;
    type Error = WeightRecipeError;

    fn import(
        &self,
        lease: CheckpointLease,
        sources: &mut Vec<PendingWeightMaterialization>,
    ) -> Result<Array, WeightRecipeError> {
        let stream = self.stream;
        let lease = self.context.weight_lease(lease)?;
        let pending = if self.borrow_sources {
            lease.prepare_borrowed_materialization(stream)?
        } else {
            lease.prepare_materialization(stream, stream)?
        };
        let array = pending.output().clone();
        sources.push(pending);
        Ok(array)
    }

    fn apply(
        &self,
        recipe: &DerivedWeightRecipe,
        operands: Vec<Array>,
        sources: &mut Vec<PendingWeightMaterialization>,
    ) -> Result<Array, WeightRecipeError> {
        use DerivedWeightRecipe::*;
        let stream = self.stream;
        let mut operands = operands.into_iter();
        let mut next = || operands.next().expect("validated recipe operand");
        match recipe {
            Source { .. } => unreachable!("source nodes use the import callback"),
            Select { selection, .. } => {
                let array = next();
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
            Concatenate { axis, .. } => {
                let references = operands.as_slice().iter().collect::<Vec<_>>();
                Ok(concatenate_axis(
                    &references,
                    usize_to_i32(*axis, "concatenate axis")?,
                    stream,
                )?)
            }
            Stack { axis, .. } => {
                let references = operands.as_slice().iter().collect::<Vec<_>>();
                Ok(stack_axis(
                    &references,
                    usize_to_i32(*axis, "stack axis")?,
                    stream,
                )?)
            }
            Reshape { shape, .. } => {
                let array = next();
                let shape = shape
                    .iter()
                    .map(|dimension| usize_to_i32(*dimension, "reshape dimension"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.reshape(&shape, stream)?)
            }
            Transpose { axes, .. } => {
                let array = next();
                let axes = axes
                    .iter()
                    .map(|axis| usize_to_i32(*axis, "transpose axis"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.transpose_axes(&axes, stream)?)
            }
            DecodeTensorFp8 { .. } | ScaleF32 { .. } => {
                let weight = next();
                let scale = next();
                let prepared = WeightMaterialization::prepare_retained(
                    vec![weight, scale],
                    std::mem::take(sources),
                )?;
                let value = prepared.inputs()[1].clone().try_item::<f32>(stream)?;
                if !value.is_finite() || value <= 0.0 {
                    prepared.finish()?;
                    return Err(if matches!(recipe, DecodeTensorFp8 { .. }) {
                        WeightRecipeError::InvalidTensorFp8Scale
                    } else {
                        WeightRecipeError::InvalidF32Scale
                    });
                }
                let weight = &prepared.inputs()[0];
                let scale = &prepared.inputs()[1];
                let output = if matches!(recipe, DecodeTensorFp8 { .. }) {
                    weight
                        .from_fp8(Dtype::Float32, stream)?
                        .multiply(scale, stream)?
                } else {
                    weight.multiply(scale, stream)?
                };
                sources.extend(prepared.finish_preparation()?);
                Ok(output)
            }
            Cast { dtype, .. } => {
                let array = next();
                Ok(array.as_dtype(mlx_dtype(dtype)?, stream)?)
            }
            View { dtype, shape, .. } => {
                let array = next();
                let shape = shape
                    .iter()
                    .map(|dimension| usize_to_i32(*dimension, "view dimension"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array
                    .view_dtype(mlx_dtype(dtype)?, stream)?
                    .reshape(&shape, stream)?)
            }
            NegLog { .. } => {
                let array = next();
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
            SubtractOne { .. } => {
                let array = next();
                let one = Array::from_f32(1.0).as_dtype(array.dtype(), stream)?;
                Ok(array.subtract(one, stream)?)
            }
        }
    }

    fn store_error(&self, error: StoreError) -> WeightRecipeError {
        error.into()
    }

    fn can_retry_join_operand(&self, error: &WeightRecipeError) -> bool {
        !self.borrow_sources
            && matches!(
                error,
                WeightRecipeError::CheckpointStore(StoreError::CapacityExhausted { .. })
            )
    }

    fn detach_join_operand(
        &self,
        value: &Array,
        sources: &mut Vec<PendingWeightMaterialization>,
    ) -> Result<(), WeightRecipeError> {
        WeightMaterialization::submit_retained(value.clone(), std::mem::take(sources))?
            .synchronize()?;
        Ok(())
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
    /// A weight-only FP8 multiplier cannot define finite, positively scaled weights.
    #[error("tensor FP8 scale must be finite and positive")]
    InvalidTensorFp8Scale,
    /// A checkpoint multiplier is outside the admitted positive finite range.
    #[error("F32 scaling requires a finite positive scalar")]
    InvalidF32Scale,
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

/// Copies final storage on the destination stream while preserving a selected
/// physical matrix layout. Only the returned logical view survives publication.
pub(crate) fn copy_parameter_storage(
    input: &Array,
    layout: eredu_nn::ParameterStorageLayout,
    stream: &Stream,
) -> Result<Array, safemlx::error::Exception> {
    if layout == eredu_nn::ParameterStorageLayout::TransposedMatrix && input.ndim() == 2 {
        input
            .transpose(stream)?
            .ensure_device_storage(stream)?
            .transpose(stream)
    } else {
        input.ensure_device_storage(stream)
    }
}
