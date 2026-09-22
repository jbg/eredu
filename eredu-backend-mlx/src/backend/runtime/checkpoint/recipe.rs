//! Deterministic checkpoint-derived weight recipes.

//!
//! Recipes describe the runtime representation of a parameter without tying it
//! to a single checkpoint key. They are validated from checkpoint metadata and
//! materialized on the residency source stream before device promotion.

use crate::backend::runtime::checkpoint::store::OriginalMaterializationSlots;
use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype};
use eredu_checkpoint::store::{CheckpointSource, ReadPolicy, TensorReadRequest, TensorSelection};
use safemlx::OriginalScopeObserver;

use safemlx::{
    Array, Dtype, HostTransferBuffer, HostTransferPolicy, Stream,
    ops::{concatenate_axis, contiguous, stack_axis},
};

use crate::backend::runtime::checkpoint::store::{
    MlxParameterMaterializationContext, PendingWeightMaterialization, WeightMaterialization,
};

mod direct_plan;
mod equation;
mod native;
pub(crate) use native::RecipeLeafProducer;
mod trace;
pub(crate) use trace::{RecipeEquationTrace, trace_ordinary_recipe};
mod workspace;
pub(crate) use direct_plan::{PreparedDirectReadError, PreparedDirectReadPlan};

/// Logical tensor and host-index sizes of the recipe materializer.
/// Direct encoded initialization needs only its output. The ordinary path can
/// retain every source until completion; count all intermediates, source copies,
/// and the final contiguous output even when the allocator can recycle them.
/// This diagnostic predates native allocator-capacity accounting. It cannot
/// authorize working-memory admission: allocation rounding, reuse, source
/// staging and some primitive scratch are absent. Use the selected batch's
/// workspace bound where available; other paths must remain unknown.
pub(crate) fn native_recipe_workspace(
    recipe: &DerivedWeightRecipe,
    source: &dyn CheckpointSource,
) -> Result<u64, String> {
    if DirectRecipeRead::prepare(recipe, source)
        .map_err(|error| error.to_string())?
        .is_some()
    {
        return recipe
            .infer(source)
            .map(|metadata| metadata.byte_len())
            .map_err(|error| error.to_string());
    }
    fn add(left: u64, right: u64) -> Result<u64, String> {
        left.checked_add(right)
            .ok_or_else(|| "native recipe workspace overflow".into())
    }
    fn buffers(recipe: &DerivedWeightRecipe, source: &dyn CheckpointSource) -> Result<u64, String> {
        let output = recipe
            .infer(source)
            .map_err(|error| error.to_string())?
            .byte_len();
        match recipe {
            DerivedWeightRecipe::Source { .. } => add(output, output),
            DerivedWeightRecipe::Stack { inputs, .. }
            | DerivedWeightRecipe::Concatenate { inputs, .. } => inputs
                .iter()
                .try_fold(output, |bytes, input| add(bytes, buffers(input, source)?)),
            DerivedWeightRecipe::Select { input, selection } => {
                let indices = match selection {
                    TensorSelection::Full => 0,
                    TensorSelection::Range { start, end, .. } => end - start,
                    TensorSelection::Indices { indices, .. } => indices.len(),
                    TensorSelection::Contiguous { shape, .. } => shape
                        .iter()
                        .try_fold(1usize, |n, d| n.checked_mul(*d))
                        .ok_or("native index count overflow")?,
                };
                let index_bytes = (indices as u64)
                    .checked_mul(8)
                    .ok_or("native index buffers overflow")?;
                add(add(buffers(input, source)?, output)?, index_bytes)
            }
            DerivedWeightRecipe::NegLog { input } => {
                // Negation output plus the sign-validation boolean tensor.
                let metadata = input.infer(source).map_err(|error| error.to_string())?;
                let elements = metadata
                    .shape()
                    .iter()
                    .try_fold(1u64, |n, d| n.checked_mul(*d as u64))
                    .ok_or("native validation size overflow")?;
                add(
                    add(
                        buffers(input, source)?,
                        output
                            .checked_mul(2)
                            .ok_or("native neg-log size overflow")?,
                    )?,
                    elements,
                )
            }
            DerivedWeightRecipe::Reshape { input, .. }
            | DerivedWeightRecipe::Transpose { input, .. }
            | DerivedWeightRecipe::Cast { input, .. }
            | DerivedWeightRecipe::View { input, .. }
            | DerivedWeightRecipe::SubtractOne { input } => add(buffers(input, source)?, output),
        }
    }
    add(
        buffers(recipe, source)?,
        recipe
            .infer(source)
            .map_err(|error| error.to_string())?
            .byte_len(),
    )
}

/// A byte-preserving recipe admitted for direct initialization of native storage.
#[derive(Clone)]
pub(crate) struct DirectRecipeRead {
    shape: Vec<i32>,
    dtype: Dtype,
    read: eredu_checkpoint::recipe::EncodedRecipeRead,
}

impl DirectRecipeRead {
    pub(crate) fn encoded(&self) -> &eredu_checkpoint::recipe::EncodedRecipeRead {
        &self.read
    }
    pub(crate) fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub(crate) fn dtype(&self) -> Dtype {
        self.dtype
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
            Ok(Some(Self { shape, dtype, read }))
        }
    }

    pub(crate) fn materialize_many(reads: Vec<Self>) -> Result<Vec<Array>, WeightRecipeError> {
        let (specifications, reads): (Vec<_>, Vec<_>) = reads
            .into_iter()
            .map(|value| ((value.shape, value.dtype), value.read))
            .unzip();
        let specifications = specifications
            .iter()
            .map(|(shape, dtype)| (shape.as_slice(), *dtype))
            .collect::<Vec<_>>();
        Array::try_init_many_with(&specifications, |outputs| {
            eredu_checkpoint::recipe::EncodedRecipeRead::read_many_into(reads, outputs)
                .map_err(WeightRecipeError::CheckpointStore)
        })
    }

    /// Fill the final host-transfer allocations through the same encoded read
    /// worker. No intermediate array, stream submission or completion is needed.
    /// A failed synchronous read drops every unpublished destination, including
    /// any already written prefix. Metadata and native buffer controls here are
    /// ordinary allocations; this method does not itself grant strict admission.
    pub(crate) fn materialize_host_many(
        reads: Vec<Self>,
    ) -> Result<Vec<HostTransferBuffer>, WeightRecipeError> {
        let mut buffers = Vec::with_capacity(reads.len());
        let mut encoded = Vec::with_capacity(reads.len());
        for read in reads {
            buffers.push(HostTransferBuffer::new(
                &read.shape,
                read.dtype,
                HostTransferPolicy::Transfer,
            )?);
            encoded.push(read.read);
        }
        let mut destinations = buffers
            .iter_mut()
            .map(HostTransferBuffer::as_bytes_mut)
            .collect::<Result<Vec<_>, _>>()?;
        eredu_checkpoint::recipe::EncodedRecipeRead::read_many_into(encoded, &mut destinations)?;
        Ok(buffers)
    }
}

/// One compatible group, or one binding requiring the ordinary materializer.
pub(crate) enum BindingReadBatch<'a> {
    Direct {
        bindings: Vec<&'a eredu_runtime::WeightBinding>,
        reads: Vec<DirectRecipeRead>,
    },
    Ordinary(&'a eredu_runtime::WeightBinding),
}

/// Plans all reads without allocating weights; allocation is bounded per group.
pub(crate) fn plan_binding_reads<'a>(
    source: &dyn CheckpointSource,
    bindings: impl IntoIterator<Item = &'a eredu_runtime::WeightBinding>,
) -> Result<Vec<BindingReadBatch<'a>>, WeightRecipeError> {
    let mut batches = Vec::new();
    let mut group = Vec::new();
    let mut reads = Vec::new();
    let mut budget = eredu_runtime::ParameterBatchBudget::default();
    for binding in bindings {
        let direct_source;
        let recipe = if let Some(recipe) = binding.recipe() {
            recipe
        } else {
            direct_source = binding.source_recipe();
            &direct_source
        };
        let read = DirectRecipeRead::prepare(recipe, source)?;
        if read.is_none() || !budget.try_push(binding.expected_bytes()) {
            if !group.is_empty() {
                batches.push(BindingReadBatch::Direct {
                    bindings: std::mem::take(&mut group),
                    reads: std::mem::take(&mut reads),
                });
            }
            budget = Default::default();
            if read.is_some() {
                let admitted = budget.try_push(binding.expected_bytes());
                debug_assert!(admitted);
            }
        }
        if let Some(read) = read {
            group.push(binding);
            reads.push(read);
        } else {
            batches.push(BindingReadBatch::Ordinary(binding));
        }
    }
    if !group.is_empty() {
        batches.push(BindingReadBatch::Direct {
            bindings: group,
            reads,
        });
    }
    Ok(batches)
}

/// Materializes a recipe already classified as requiring the ordinary path.
/// Keeping that decision avoids preparing its encoded read a second time.
pub(crate) fn prepare_ordinary_recipe<'context>(
    recipe: &DerivedWeightRecipe,
    store: &dyn CheckpointSource,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    borrow_sources: bool,
) -> Result<PendingWeightRecipe, WeightRecipeError> {
    prepare_ordinary_recipe_with_operations(recipe, store, context, borrow_sources, None)
}
pub(crate) fn prepare_ordinary_recipe_with_operations<'context>(
    recipe: &DerivedWeightRecipe,
    store: &dyn CheckpointSource,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    borrow_sources: bool,
    original: Option<(
        &mut OriginalMaterializationSlots<'_>,
        &OriginalScopeObserver,
    )>,
) -> Result<PendingWeightRecipe, WeightRecipeError> {
    let context = context.into();
    recipe.infer(store)?;
    let mut sources = Vec::new();
    let source_stream = context.source_stream();
    // The shared equation includes the final row-major detachment.
    let output = equation::materialize(
        recipe,
        &mut native::NativeRecipe {
            source: native::NativeRecipeSource::Checkpoint(store),
            stream: source_stream,
            sources: &mut sources,
            borrow_sources,
            context: &context,
            original,
        },
    )?;
    Ok(PendingWeightRecipe { output, sources })
}

/// Executes the shared equation with its actual prepared source producer.
/// The caller validates the recipe and ordered leaf declarations beforehand,
/// admits their complete read/copy/equation/detachment requirements, and keeps
/// the leaf buffers and operation custody in recovery through final detachment.
/// This entry creates no source admission or original execution authority.
pub(crate) fn prepare_ordinary_recipe_from_leaves<'source, 'context>(
    recipe: &DerivedWeightRecipe,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    leaves: &'source mut RecipeLeafProducer<'source>,
) -> Result<PendingWeightRecipe, WeightRecipeError> {
    prepare_recipe_from_leaves_impl(recipe, context.into(), leaves, None, None)
}

/// Uses the same prepared equation and validation guards, retaining the actual
/// admitted constructor's Host payer through every nested recovery scope.
pub(crate) fn prepare_ordinary_recipe_from_funded_leaves<'source, 'context>(
    recipe: &DerivedWeightRecipe,
    context: impl Into<crate::backend::runtime::checkpoint::store::MaterializationView<'context>>,
    leaves: &'source mut RecipeLeafProducer<'source>,
    host: &'source eredu_core::HostPreparationAuthority,
) -> Result<PendingWeightRecipe, WeightRecipeError> {
    prepare_recipe_from_leaves_impl(recipe, context.into(), leaves, Some(host), None)
}

/// The same prepared-leaf equation inside a genuine Original native role.
/// Its validation operations consume the caller's accepted slots; the source
/// callback retains each actual prepared Host copy through role completion.
pub(crate) fn prepare_original_recipe_from_leaves<'source>(
    recipe: &DerivedWeightRecipe,
    context: crate::backend::runtime::checkpoint::store::MaterializationView<'_>,
    leaves: &'source mut RecipeLeafProducer<'source>,
    slots: &mut OriginalMaterializationSlots<'_>,
    observer: &OriginalScopeObserver,
) -> Result<PendingWeightRecipe, WeightRecipeError> {
    prepare_recipe_from_leaves_impl(recipe, context, leaves, None, Some((slots, observer)))
}

fn prepare_recipe_from_leaves_impl<'source>(
    recipe: &DerivedWeightRecipe,
    context: crate::backend::runtime::checkpoint::store::MaterializationView<'_>,
    leaves: &'source mut RecipeLeafProducer<'source>,
    host: Option<&'source eredu_core::HostPreparationAuthority>,
    original: Option<(
        &mut OriginalMaterializationSlots<'_>,
        &OriginalScopeObserver,
    )>,
) -> Result<PendingWeightRecipe, WeightRecipeError> {
    let mut sources = Vec::new();
    let output = equation::materialize(
        recipe,
        &mut native::NativeRecipe {
            source: native::NativeRecipeSource::Prepared { leaves, host },
            stream: context.source_stream(),
            sources: &mut sources,
            borrow_sources: false,
            context: &context,
            original,
        },
    )?;
    Ok(PendingWeightRecipe { output, sources })
}

pub(crate) fn prepared_recipe_entry_control_bytes() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [
        size_of::<native::NativeRecipe<'_, '_, '_, '_, '_, '_, '_, '_>>(),
        size_of::<native::NativeRecipeSource<'_>>(),
        size_of::<&mut RecipeLeafProducer<'_>>(),
        size_of::<Option<&eredu_core::HostPreparationAuthority>>(),
        size_of::<
            Option<(
                &mut OriginalMaterializationSlots<'_>,
                &OriginalScopeObserver,
            )>,
        >(),
        size_of::<crate::backend::runtime::checkpoint::store::MaterializationView<'_>>(),
        size_of::<Vec<PendingWeightMaterialization>>(),
        size_of::<Array>(),
        size_of::<PendingWeightRecipe>(),
        size_of::<Result<PendingWeightRecipe, WeightRecipeError>>(),
        size_of::<(&DerivedWeightRecipe, &Stream)>(),
    ];
    frames
        .into_iter()
        .try_fold(size_of_val(&frames), usize::checked_add)
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
    mlx_dtype_if_supported(value).ok_or_else(|| WeightRecipeError::UnsupportedDtype {
        dtype: match value {
            RecipeDtype::Other(dtype) => dtype.clone(),
            _ => format!("{value:?}"),
        },
    })
}
pub(crate) fn mlx_dtype_if_supported(value: &RecipeDtype) -> Option<Dtype> {
    match value {
        RecipeDtype::Bool => Some(Dtype::Bool),
        RecipeDtype::U8
        | RecipeDtype::F8E4M3
        | RecipeDtype::F8E5M2
        | RecipeDtype::F4
        | RecipeDtype::F8E8M0 => Some(Dtype::Uint8),
        RecipeDtype::I8 => Some(Dtype::Int8),
        RecipeDtype::I16 => Some(Dtype::Int16),
        RecipeDtype::U16 => Some(Dtype::Uint16),
        RecipeDtype::F16 => Some(Dtype::Float16),
        RecipeDtype::BF16 => Some(Dtype::Bfloat16),
        RecipeDtype::I32 => Some(Dtype::Int32),
        RecipeDtype::U32 => Some(Dtype::Uint32),
        RecipeDtype::F32 => Some(Dtype::Float32),
        RecipeDtype::F64 => Some(Dtype::Float64),
        RecipeDtype::I64 => Some(Dtype::Int64),
        RecipeDtype::U64 => Some(Dtype::Uint64),
        RecipeDtype::C64 => Some(Dtype::Complex64),
        _ => None,
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
        if let Some(read) = DirectRecipeRead::prepare(self, store)? {
            let output = DirectRecipeRead::materialize_many(vec![read])?
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
        let context =
            crate::backend::runtime::checkpoint::store::MaterializationView::from(context);
        materialize_inner_with_operations(
            self,
            store,
            stream,
            sources,
            borrow_sources,
            &context,
            None,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn materialize_inner_with_operations<'context>(
    recipe: &DerivedWeightRecipe,
    store: &dyn CheckpointSource,
    stream: &Stream,
    sources: &mut Vec<PendingWeightMaterialization>,
    borrow_sources: bool,
    context: &crate::backend::runtime::checkpoint::store::MaterializationView<'context>,
    original: Option<(
        &mut OriginalMaterializationSlots<'_>,
        &OriginalScopeObserver,
    )>,
) -> Result<Array, WeightRecipeError> {
    equation::execute(
        recipe,
        &mut native::NativeRecipe {
            source: native::NativeRecipeSource::Checkpoint(store),
            stream,
            sources,
            borrow_sources,
            context,
            original,
        },
    )
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

fn submit_materialized_child(
    array: Array,
    sources: Vec<PendingWeightMaterialization>,
    original: Option<(
        &mut OriginalMaterializationSlots<'_>,
        &OriginalScopeObserver,
    )>,
) -> Result<WeightMaterialization, WeightRecipeError> {
    Ok(match original {
        Some((slots, observer)) => {
            WeightMaterialization::submit_retained_with_operations(array, sources, slots, observer)?
        }
        None => WeightMaterialization::submit_retained(array, sources)?,
    })
}

fn materialize_inputs<'context>(
    inputs: &[DerivedWeightRecipe],
    store: &dyn CheckpointSource,
    stream: &Stream,
    sources: &mut Vec<PendingWeightMaterialization>,
    borrow_sources: bool,
    context: &crate::backend::runtime::checkpoint::store::MaterializationView<'context>,
    mut original: Option<(
        &mut OriginalMaterializationSlots<'_>,
        &OriginalScopeObserver,
    )>,
) -> Result<Vec<Array>, WeightRecipeError> {
    let mut pending =
        Vec::<(Array, Vec<PendingWeightMaterialization>)>::with_capacity(inputs.len());
    let mut detach_remaining = false;
    for input in inputs {
        loop {
            let mut input_sources = Vec::new();
            match materialize_inner_with_operations(
                input,
                store,
                stream,
                &mut input_sources,
                borrow_sources,
                context,
                original
                    .as_mut()
                    .map(|(slots, observer)| (&mut **slots, *observer)),
            ) {
                Ok(array) => {
                    if detach_remaining && !input_sources.is_empty() {
                        submit_materialized_child(
                            array.clone(),
                            std::mem::take(&mut input_sources),
                            original
                                .as_mut()
                                .map(|(slots, observer)| (&mut **slots, *observer)),
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
                        && recipe_cache_capacity(&error) =>
                {
                    // Prepared failures may retain an actual rejected lease.
                    // It cannot keep the cache pinned across this retry.
                    let _ordinary_error = original.is_none().then_some(error);
                    // The current child could not acquire another shard while
                    // earlier children pinned the shard cache. Their arrays
                    // are sufficient evaluation roots, so detach them and retry.
                    drop(input_sources);
                    for (array, child_sources) in &mut pending {
                        if child_sources.is_empty() {
                            continue;
                        }
                        submit_materialized_child(
                            array.clone(),
                            std::mem::take(child_sources),
                            original
                                .as_mut()
                                .map(|(slots, observer)| (&mut **slots, *observer)),
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
    /// Descriptive equation construction preserves its neutral funding cause.
    #[error(transparent)]
    Workspace(#[from] eredu_nn::Error),
    /// The shared equation could not allocate its checked metadata vector.
    #[error("checkpoint recipe metadata allocation: {0}")]
    MetadataAllocation(#[source] std::collections::TryReserveError),
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
    /// Keep source-less admission failures in the public error chain.
    #[error("{0}")]
    CheckpointStore(#[from] eredu_checkpoint::store::StoreError),
    /// MLX checkpoint materialization failed.
    #[error(transparent)]
    CheckpointMaterialization(
        #[from] crate::backend::runtime::checkpoint::store::CheckpointMaterializationError,
    ),
    /// MLX transformation or synchronization failed.
    #[error("{0}")]
    Mlx(#[from] safemlx::error::Exception),
}

#[cfg(test)]
mod tests;

#[cfg(all(
    test,
    target_vendor = "apple",
    feature = "metal",
    not(feature = "cuda")
))]
mod funded_tests;

fn recipe_cache_capacity(error: &WeightRecipeError) -> bool {
    use crate::backend::runtime::checkpoint::store::CheckpointMaterializationError;
    match error {
        WeightRecipeError::CheckpointStore(
            eredu_checkpoint::store::StoreError::CapacityExhausted { .. },
        ) => true,
        WeightRecipeError::CheckpointMaterialization(
            CheckpointMaterializationError::PreparedAcquisition(error),
        ) => matches!(
            error.store_error(),
            Some(eredu_checkpoint::store::StoreError::CapacityExhausted { .. })
        ),
        _ => false,
    }
}
