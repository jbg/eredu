//! Deterministic checkpoint-derived weight recipes.
//!
//! Recipes describe the runtime representation of a parameter without tying it
//! to a single checkpoint key. They are validated from checkpoint metadata and
//! materialized on the residency source stream before device promotion.

use std::collections::BTreeSet;

use safemlx::{
    ops::{concatenate_axis, contiguous, stack_axis},
    transforms::async_eval_with_event,
    Array, Dtype, Stream,
};

use crate::runtime::checkpoint::store::{
    PendingWeightMaterialization, StoredDtype, TensorSelection, WeightReadPolicy, WeightStore,
    WeightStoreError,
};

/// Scalar encoding produced by a derived-weight recipe.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum RecipeDtype {
    /// Boolean values.
    Bool,
    /// Unsigned 8-bit integers.
    U8,
    /// Signed 8-bit integers.
    I8,
    /// Signed 16-bit integers.
    I16,
    /// Unsigned 16-bit integers.
    U16,
    /// IEEE half-precision floating point.
    F16,
    /// Brain floating point.
    BF16,
    /// Signed 32-bit integers.
    I32,
    /// Unsigned 32-bit integers.
    U32,
    /// IEEE single-precision floating point.
    F32,
    /// IEEE double-precision floating point.
    F64,
    /// Signed 64-bit integers.
    I64,
    /// Unsigned 64-bit integers.
    U64,
    /// Complex values with two 32-bit floating-point components.
    C64,
    /// Encoded FP8 E4M3 bytes.
    F8E4M3,
    /// Encoded FP8 E5M2 bytes.
    F8E5M2,
    /// A checkpoint encoding unknown to this runtime.
    Other(String),
}

impl RecipeDtype {
    fn byte_width(&self) -> Result<u64, WeightRecipeError> {
        match self {
            Self::Bool | Self::U8 | Self::I8 | Self::F8E4M3 | Self::F8E5M2 => Ok(1),
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => Ok(2),
            Self::I32 | Self::U32 | Self::F32 => Ok(4),
            Self::F64 | Self::I64 | Self::U64 | Self::C64 => Ok(8),
            Self::Other(dtype) => Err(WeightRecipeError::UnsupportedDtype {
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
            StoredDtype::Other(dtype) => Self::Other(dtype),
        }
    }
}

impl From<Dtype> for RecipeDtype {
    fn from(value: Dtype) -> Self {
        match value {
            Dtype::Bool => Self::Bool,
            Dtype::Uint8 => Self::U8,
            Dtype::Uint16 => Self::U16,
            Dtype::Uint32 => Self::U32,
            Dtype::Uint64 => Self::U64,
            Dtype::Int8 => Self::I8,
            Dtype::Int16 => Self::I16,
            Dtype::Int32 => Self::I32,
            Dtype::Int64 => Self::I64,
            Dtype::Float16 => Self::F16,
            Dtype::Float32 => Self::F32,
            Dtype::Float64 => Self::F64,
            Dtype::Bfloat16 => Self::BF16,
            Dtype::Complex64 => Self::C64,
        }
    }
}

/// Shape, encoding, and materialized size inferred for a recipe.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightRecipeMetadata {
    shape: Vec<usize>,
    dtype: RecipeDtype,
    byte_len: u64,
}

impl WeightRecipeMetadata {
    /// Returns the inferred output shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the inferred output scalar encoding.
    pub const fn dtype(&self) -> &RecipeDtype {
        &self.dtype
    }

    /// Returns the checked materialized output size.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// A composable operation that derives one runtime weight from checkpoint tensors.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum DerivedWeightRecipe {
    /// Selects a complete tensor, contiguous range, or ordered indices.
    Source {
        /// Checkpoint tensor key. This also represents checkpoint-key renaming.
        key: String,
        /// Selection applied before any derived operations.
        selection: TensorSelection,
    },
    /// Applies another selection to a derived array.
    Select {
        /// Child recipe.
        input: Box<Self>,
        /// Selection applied to the child output.
        selection: TensorSelection,
    },
    /// Concatenates child outputs along an existing axis.
    Concatenate {
        /// Concatenation axis.
        axis: usize,
        /// Ordered child recipes.
        inputs: Vec<Self>,
    },
    /// Stacks child outputs along a new axis.
    Stack {
        /// Inserted axis.
        axis: usize,
        /// Ordered child recipes.
        inputs: Vec<Self>,
    },
    /// Reshapes a child while preserving its element count.
    Reshape {
        /// Child recipe.
        input: Box<Self>,
        /// Exact output shape.
        shape: Vec<usize>,
    },
    /// Reorders axes. Axis movement is represented by its resulting permutation.
    Transpose {
        /// Child recipe.
        input: Box<Self>,
        /// Output-axis to input-axis permutation.
        axes: Vec<usize>,
    },
    /// Casts a child to an MLX execution dtype.
    Cast {
        /// Child recipe.
        input: Box<Self>,
        /// Output execution dtype.
        dtype: Dtype,
    },
    /// Reinterprets the backing bytes as another scalar dtype and exact shape.
    ///
    /// Unlike [`Self::Cast`], this does not numerically convert values. It is
    /// intended for checkpoint formats whose packed byte representation is
    /// already the representation consumed by the runtime kernel.
    View {
        /// Child recipe.
        input: Box<Self>,
        /// Output execution dtype.
        dtype: Dtype,
        /// Exact output shape.
        shape: Vec<usize>,
    },
    /// Applies `log(-x)` elementwise.
    NegLog {
        /// Child recipe.
        input: Box<Self>,
    },
    /// Subtracts one elementwise for GGUF offset-normalization weights.
    SubtractOne {
        /// Child recipe.
        input: Box<Self>,
    },
}

impl DerivedWeightRecipe {
    /// Creates a direct checkpoint source recipe.
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

    /// Rewrites an output selection toward checkpoint sources.
    ///
    /// The returned recipe is metadata-equivalent to selecting this recipe's
    /// output, but joins and shape transforms are rewritten so source stores
    /// can perform bounded reads. Geometry that cannot be represented by the
    /// store's single-axis selection contract is rejected during planning.
    pub fn select_bounded(
        &self,
        store: &dyn WeightStore,
        selection: TensorSelection,
    ) -> Result<Self, WeightRecipeError> {
        let metadata = self.infer(store)?;
        let selection = normalize_selection(selection, &metadata.shape)?;
        let expected_shape = selected_shape(metadata.shape.clone(), &selection)?;
        // Expand strided source indices before pushing an independent leading
        // selection. This lets each physical row range combine with (for
        // example) one expert coordinate into a contiguous scalar span. If
        // expansion happens afterwards, the leading range remains wrapped
        // around an indexed source and the store cannot bound either axis.
        let expanded = expand_indexed_sources(self.clone());
        let rewritten = normalize_bounded_source_ranges(
            expand_indexed_sources(push_selection(&expanded, store, selection)?),
            store,
        )?;
        let actual = rewritten.infer(store)?;
        if actual.shape != expected_shape || actual.dtype != metadata.dtype {
            return Err(WeightRecipeError::SelectionPushdownUnsupported {
                operation: "recipe",
                reason: format!(
                    "rewrite produced shape {:?} and dtype {:?}, expected {:?} and {:?}",
                    actual.shape, actual.dtype, expected_shape, metadata.dtype
                ),
            });
        }
        Ok(rewritten)
    }

    /// Selects rows from one matrix in a rank-two-or-higher tensor recipe.
    ///
    /// Leading dimensions are retained as singleton dimensions. This lets a
    /// bounded writer visit an expert-major bank one matrix at a time without
    /// flattening away the runtime bank geometry or reading adjacent experts.
    pub(crate) fn select_bounded_matrix_rows(
        &self,
        store: &dyn WeightStore,
        leading_index: usize,
        start: usize,
        end: usize,
    ) -> Result<Self, WeightRecipeError> {
        let metadata = self.infer(store)?;
        if metadata.shape.len() < 2 {
            return Err(WeightRecipeError::SelectionPushdownUnsupported {
                operation: "matrix row selection",
                reason: format!("rank {} has no matrix row axis", metadata.shape.len()),
            });
        }
        let row_axis = metadata.shape.len() - 2;
        let leading = usize::try_from(element_count(
            &metadata.shape[..row_axis],
            "leading matrix dimensions",
        )?)
        .map_err(|_| WeightRecipeError::ArithmeticOverflow("leading matrix dimensions"))?;
        if leading_index >= leading {
            return Err(WeightRecipeError::InvalidIndices {
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
                store,
                TensorSelection::Range {
                    axis,
                    start: coordinate,
                    end: coordinate + 1,
                },
            )?;
        }
        selected.select_bounded(
            store,
            TensorSelection::Range {
                axis: row_axis,
                start,
                end,
            },
        )
    }

    /// Validates that every checkpoint source can be acquired without a
    /// complete-tensor fallback.
    pub fn preflight_bounded(&self, store: &dyn WeightStore) -> Result<(), WeightRecipeError> {
        match self {
            Self::Source { key, selection } => {
                drop(store.acquire_with_policy(
                    key,
                    selection.clone(),
                    WeightReadPolicy::RequireBounded,
                )?);
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                for input in inputs {
                    input.preflight_bounded(store)?;
                }
            }
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => input.preflight_bounded(store)?,
        }
        Ok(())
    }

    /// Returns a conservative bound for simultaneously live materialized
    /// arrays while evaluating this recipe.
    ///
    /// The bound includes retained inputs and the operation output. It is
    /// intentionally conservative for view-like operations so callers can use
    /// it as a hard admission input for out-of-core transformations without
    /// relying on MLX aliasing optimizations.
    pub(crate) fn peak_materialization_bytes(
        &self,
        store: &dyn WeightStore,
    ) -> Result<u64, WeightRecipeError> {
        let output_bytes = self.infer(store)?.byte_len();
        match self {
            Self::Source { .. } => Ok(output_bytes),
            Self::Select { input, .. }
            | Self::Reshape { input, .. }
            | Self::Transpose { input, .. }
            | Self::Cast { input, .. }
            | Self::View { input, .. }
            | Self::NegLog { input }
            | Self::SubtractOne { input } => {
                let input_bytes = input.infer(store)?.byte_len();
                let child_peak = input.peak_materialization_bytes(store)?;
                Ok(child_peak.max(input_bytes.checked_add(output_bytes).ok_or(
                    WeightRecipeError::ArithmeticOverflow(
                        "unary recipe peak materialization bytes",
                    ),
                )?))
            }
            Self::Concatenate { inputs, .. } | Self::Stack { inputs, .. } => {
                let mut retained = 0u64;
                let mut peak = 0u64;
                for input in inputs {
                    let child_peak = input.peak_materialization_bytes(store)?;
                    peak = peak.max(retained.checked_add(child_peak).ok_or(
                        WeightRecipeError::ArithmeticOverflow("joined recipe child peak bytes"),
                    )?);
                    retained = retained.checked_add(input.infer(store)?.byte_len()).ok_or(
                        WeightRecipeError::ArithmeticOverflow("joined recipe retained input bytes"),
                    )?;
                }
                Ok(peak.max(retained.checked_add(output_bytes).ok_or(
                    WeightRecipeError::ArithmeticOverflow("joined recipe output peak bytes"),
                )?))
            }
        }
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

    /// Validates the complete recipe and infers its output metadata.
    pub fn infer(
        &self,
        store: &dyn WeightStore,
    ) -> Result<WeightRecipeMetadata, WeightRecipeError> {
        match self {
            Self::Source { key, selection } => infer_source(store, key, selection),
            Self::Select { input, selection } => {
                let metadata = input.infer(store)?;
                let shape = selected_shape(metadata.shape, selection)?;
                metadata_for(shape, metadata.dtype)
            }
            Self::Concatenate { axis, inputs } => infer_join(store, *axis, inputs, false),
            Self::Stack { axis, inputs } => infer_join(store, *axis, inputs, true),
            Self::Reshape { input, shape } => {
                let metadata = input.infer(store)?;
                let old_count = element_count(&metadata.shape, "reshape input")?;
                let new_count = element_count(shape, "reshape output")?;
                if old_count != new_count {
                    return Err(WeightRecipeError::ElementCountMismatch {
                        input: old_count,
                        output: new_count,
                    });
                }
                metadata_for(shape.clone(), metadata.dtype)
            }
            Self::Transpose { input, axes } => {
                let metadata = input.infer(store)?;
                if axes.len() != metadata.shape.len() {
                    return Err(WeightRecipeError::InvalidPermutation {
                        axes: axes.clone(),
                        rank: metadata.shape.len(),
                    });
                }
                let unique = axes.iter().copied().collect::<BTreeSet<_>>();
                if unique.len() != axes.len() || axes.iter().any(|axis| *axis >= axes.len()) {
                    return Err(WeightRecipeError::InvalidPermutation {
                        axes: axes.clone(),
                        rank: metadata.shape.len(),
                    });
                }
                let shape = axes.iter().map(|axis| metadata.shape[*axis]).collect();
                metadata_for(shape, metadata.dtype)
            }
            Self::Cast { input, dtype } => {
                let metadata = input.infer(store)?;
                metadata_for(metadata.shape, (*dtype).into())
            }
            Self::View {
                input,
                dtype,
                shape,
            } => {
                let input = input.infer(store)?;
                let output = metadata_for(shape.clone(), (*dtype).into())?;
                if input.byte_len != output.byte_len {
                    return Err(WeightRecipeError::ByteCountMismatch {
                        input: input.byte_len,
                        output: output.byte_len,
                    });
                }
                Ok(output)
            }
            Self::NegLog { input } => input.infer(store),
            Self::SubtractOne { input } => input.infer(store),
        }
    }

    /// Materializes this recipe on the host source stream.
    ///
    /// Source leases remain live until their dependent output has been
    /// evaluated. If a multi-input join reaches the mapping bound, completed
    /// children are detached before retrying so cross-shard recipes can honor
    /// a one-mapping limit without serializing the normal batched path.
    pub fn materialize(
        &self,
        store: &dyn WeightStore,
        source_stream: &Stream,
    ) -> Result<Array, WeightRecipeError> {
        self.prepare_materialization(store, source_stream)?.finish()
    }

    /// Schedules a recipe while retaining all mmap-backed source selections.
    pub(crate) fn prepare_materialization(
        &self,
        store: &dyn WeightStore,
        source_stream: &Stream,
    ) -> Result<PendingWeightRecipe, WeightRecipeError> {
        self.prepare_materialization_mode(store, source_stream, false)
    }

    /// Schedules a recipe whose bounded checkpoint sources may remain borrowed
    /// until the containing output is evaluated.
    pub(crate) fn prepare_borrowed_materialization(
        &self,
        store: &dyn WeightStore,
        source_stream: &Stream,
    ) -> Result<PendingWeightRecipe, WeightRecipeError> {
        self.prepare_materialization_mode(store, source_stream, true)
    }

    fn prepare_materialization_mode(
        &self,
        store: &dyn WeightStore,
        source_stream: &Stream,
        borrow_sources: bool,
    ) -> Result<PendingWeightRecipe, WeightRecipeError> {
        self.infer(store)?;
        let mut sources = Vec::new();
        let output = self.materialize_inner(store, source_stream, &mut sources, borrow_sources)?;
        // Derived recipe outputs are immutable and may be reused across forwards.
        // Detach gathers/transposes into their final row-major representation
        // once here so consumers do not silently repack a full weight on every
        // kernel invocation.
        let output = contiguous(output, false, source_stream)?;
        Ok(PendingWeightRecipe { output, sources })
    }

    fn materialize_inner(
        &self,
        store: &dyn WeightStore,
        stream: &Stream,
        sources: &mut Vec<PendingWeightMaterialization>,
        borrow_sources: bool,
    ) -> Result<Array, WeightRecipeError> {
        match self {
            Self::Source { key, selection } => {
                let lease = store.acquire_with_policy(
                    key,
                    selection.clone(),
                    WeightReadPolicy::RequireBounded,
                )?;
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
                let array = input.materialize_inner(store, stream, sources, borrow_sources)?;
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
                let arrays = materialize_inputs(inputs, store, stream, sources, borrow_sources)?;
                let references = arrays.iter().collect::<Vec<_>>();
                Ok(concatenate_axis(
                    &references,
                    usize_to_i32(*axis, "concatenate axis")?,
                    stream,
                )?)
            }
            Self::Stack { axis, inputs } => {
                let arrays = materialize_inputs(inputs, store, stream, sources, borrow_sources)?;
                let references = arrays.iter().collect::<Vec<_>>();
                Ok(stack_axis(
                    &references,
                    usize_to_i32(*axis, "stack axis")?,
                    stream,
                )?)
            }
            Self::Reshape { input, shape } => {
                let array = input.materialize_inner(store, stream, sources, borrow_sources)?;
                let shape = shape
                    .iter()
                    .map(|dimension| usize_to_i32(*dimension, "reshape dimension"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.reshape(&shape, stream)?)
            }
            Self::Transpose { input, axes } => {
                let array = input.materialize_inner(store, stream, sources, borrow_sources)?;
                let axes = axes
                    .iter()
                    .map(|axis| usize_to_i32(*axis, "transpose axis"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.transpose_axes(&axes, stream)?)
            }
            Self::Cast { input, dtype } => {
                let array = input.materialize_inner(store, stream, sources, borrow_sources)?;
                Ok(array.as_dtype(*dtype, stream)?)
            }
            Self::View {
                input,
                dtype,
                shape,
            } => {
                let array = input.materialize_inner(store, stream, sources, borrow_sources)?;
                let shape = shape
                    .iter()
                    .map(|dimension| usize_to_i32(*dimension, "view dimension"))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(array.view_dtype(*dtype, stream)?.reshape(&shape, stream)?)
            }
            Self::NegLog { input } => {
                let array = input.materialize_inner(store, stream, sources, borrow_sources)?;
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
                let array = input.materialize_inner(store, stream, sources, borrow_sources)?;
                Ok(array.subtract(Array::from_f32(1.0), stream)?)
            }
        }
    }
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

fn normalize_bounded_source_ranges(
    recipe: DerivedWeightRecipe,
    store: &dyn WeightStore,
) -> Result<DerivedWeightRecipe, WeightRecipeError> {
    Ok(match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: TensorSelection::Range { axis, start, end },
        } if axis > 0 => {
            let shape = store.metadata(&key)?.shape;
            if shape[..axis].iter().product::<usize>() == 1 {
                let trailing = shape[axis + 1..]
                    .iter()
                    .try_fold(1usize, |count, dimension| {
                        count
                            .checked_mul(*dimension)
                            .ok_or(WeightRecipeError::ArithmeticOverflow(
                                "bounded source range trailing span",
                            ))
                    })?;
                let offset_elements =
                    start
                        .checked_mul(trailing)
                        .ok_or(WeightRecipeError::ArithmeticOverflow(
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

fn push_selection(
    recipe: &DerivedWeightRecipe,
    store: &dyn WeightStore,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, WeightRecipeError> {
    if matches!(selection, TensorSelection::Full) {
        return Ok(recipe.clone());
    }
    match recipe {
        DerivedWeightRecipe::Source {
            key,
            selection: source_selection,
        } => {
            let source_shape = store.metadata(key)?.shape;
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
            let input_axis =
                *axes
                    .get(output_axis)
                    .ok_or(WeightRecipeError::InvalidSelectionAxis {
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
            dtype: *dtype,
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
                dtype: *dtype,
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

fn push_concatenate_selection(
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    store: &dyn WeightStore,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, WeightRecipeError> {
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
                        .ok_or(WeightRecipeError::ArithmeticOverflow(
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
                    .ok_or(WeightRecipeError::ArithmeticOverflow(
                        "concatenate selection offset",
                    ))?;
                offsets.push(next);
            }
            let mut runs = Vec::<(usize, Vec<usize>)>::new();
            for index in indices {
                let child = offsets
                    .windows(2)
                    .position(|bounds| index >= bounds[0] && index < bounds[1])
                    .ok_or(WeightRecipeError::InvalidIndices {
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
            return Err(WeightRecipeError::SelectionPushdownUnsupported {
                operation: "concatenate",
                reason: "a storage-contiguous span has no concatenate-axis semantics".into(),
            })
        }
    }
    match rewritten.len() {
        0 => Err(WeightRecipeError::SelectionPushdownUnsupported {
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

fn push_stack_selection(
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    store: &dyn WeightStore,
    selection: TensorSelection,
) -> Result<DerivedWeightRecipe, WeightRecipeError> {
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
                return Err(WeightRecipeError::SelectionPushdownUnsupported {
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
    input: &WeightRecipeMetadata,
    output: &WeightRecipeMetadata,
    selection: &TensorSelection,
    operation: &'static str,
) -> Result<TensorSelection, WeightRecipeError> {
    let output_axis = selection_axis(selection).expect("non-full selection");
    let output_unit = axis_unit_bytes(&output.shape, output.dtype.byte_width()?, output_axis)?;
    let output_cycle = output_unit
        .checked_mul(output.shape[output_axis] as u64)
        .ok_or(WeightRecipeError::ArithmeticOverflow(
            "selection output cycle",
        ))?;
    for input_axis in 0..input.shape.len() {
        let input_unit = axis_unit_bytes(&input.shape, input.dtype.byte_width()?, input_axis)?;
        let input_cycle = input_unit
            .checked_mul(input.shape[input_axis] as u64)
            .ok_or(WeightRecipeError::ArithmeticOverflow(
                "selection input cycle",
            ))?;
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
    Err(WeightRecipeError::SelectionPushdownUnsupported {
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
) -> Result<Option<TensorSelection>, WeightRecipeError> {
    let map_interval =
        |start: usize, end: usize| -> Result<Option<(usize, usize)>, WeightRecipeError> {
            let start_bytes = (start as u64).checked_mul(output_unit).ok_or(
                WeightRecipeError::ArithmeticOverflow("selection interval start"),
            )?;
            let end_bytes = (end as u64).checked_mul(output_unit).ok_or(
                WeightRecipeError::ArithmeticOverflow("selection interval end"),
            )?;
            if start_bytes % input_unit != 0 || end_bytes % input_unit != 0 {
                return Ok(None);
            }
            let start = usize::try_from(start_bytes / input_unit)
                .map_err(|_| WeightRecipeError::ArithmeticOverflow("mapped selection start"))?;
            let end = usize::try_from(end_bytes / input_unit)
                .map_err(|_| WeightRecipeError::ArithmeticOverflow("mapped selection end"))?;
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

fn axis_unit_bytes(
    shape: &[usize],
    dtype_width: u64,
    axis: usize,
) -> Result<u64, WeightRecipeError> {
    shape[axis + 1..]
        .iter()
        .try_fold(dtype_width, |bytes, dimension| {
            bytes
                .checked_mul(*dimension as u64)
                .ok_or(WeightRecipeError::ArithmeticOverflow("selection axis unit"))
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
) -> Result<TensorSelection, WeightRecipeError> {
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
) -> Result<TensorSelection, WeightRecipeError> {
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
        _ => Err(WeightRecipeError::SelectionPushdownUnsupported {
            operation: "selection composition",
            reason: "full selections must be normalized before composition".into(),
        }),
    }
}

fn combine_independent_ranges(
    source_shape: &[usize],
    existing: &TensorSelection,
    requested: &TensorSelection,
) -> Result<Option<TensorSelection>, WeightRecipeError> {
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
                WeightRecipeError::ArithmeticOverflow("contiguous selection offset"),
            )?)
            .ok_or(WeightRecipeError::ArithmeticOverflow(
                "contiguous selection offset",
            ))?;
        stride =
            stride
                .checked_mul(source_shape[axis])
                .ok_or(WeightRecipeError::ArithmeticOverflow(
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
) -> Result<Option<TensorSelection>, WeightRecipeError> {
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
                .ok_or(WeightRecipeError::ArithmeticOverflow(
                    "contiguous selection trailing span",
                ))
        })?;
    let offset_elements =
        offset_elements
            .checked_add(start.checked_mul(trailing).ok_or(
                WeightRecipeError::ArithmeticOverflow("contiguous selection offset"),
            )?)
            .ok_or(WeightRecipeError::ArithmeticOverflow(
                "contiguous selection offset",
            ))?;
    let mut selected_shape = shape.clone();
    selected_shape[axis] = end - start;
    Ok(Some(TensorSelection::Contiguous {
        offset_elements,
        shape: selected_shape,
    }))
}

/// Lazy recipe output and the source mappings required to evaluate it safely.
pub(crate) struct PendingWeightRecipe {
    output: Array,
    sources: Vec<PendingWeightMaterialization>,
}

impl PendingWeightRecipe {
    pub(crate) fn into_parts(self) -> (Array, Vec<PendingWeightMaterialization>) {
        (self.output, self.sources)
    }

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
    store: &dyn WeightStore,
    stream: &Stream,
    sources: &mut Vec<PendingWeightMaterialization>,
    borrow_sources: bool,
) -> Result<Vec<Array>, WeightRecipeError> {
    let mut pending =
        Vec::<(Array, Vec<PendingWeightMaterialization>)>::with_capacity(inputs.len());
    let mut detach_remaining = false;
    for input in inputs {
        loop {
            let mut input_sources = Vec::new();
            match input.materialize_inner(store, stream, &mut input_sources, borrow_sources) {
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

fn infer_source(
    store: &dyn WeightStore,
    key: &str,
    selection: &TensorSelection,
) -> Result<WeightRecipeMetadata, WeightRecipeError> {
    if key.trim().is_empty() {
        return Err(WeightRecipeError::EmptySourceKey);
    }
    let metadata = store.metadata(key)?;
    let shape = selected_shape(metadata.shape, selection)?;
    metadata_for(shape, metadata.stored_dtype.into())
}

fn selected_shape(
    mut shape: Vec<usize>,
    selection: &TensorSelection,
) -> Result<Vec<usize>, WeightRecipeError> {
    match selection {
        TensorSelection::Full => {}
        TensorSelection::Range { axis, start, end } => {
            let rank = shape.len();
            let dimension = shape
                .get_mut(*axis)
                .ok_or(WeightRecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
            if start >= end || *end > *dimension {
                return Err(WeightRecipeError::InvalidRange {
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
                .ok_or(WeightRecipeError::InvalidSelectionAxis { axis: *axis, rank })?;
            if indices.is_empty() || indices.iter().any(|index| *index >= *dimension) {
                return Err(WeightRecipeError::InvalidIndices {
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
                return Err(WeightRecipeError::SelectionPushdownUnsupported {
                    operation: "contiguous source selection",
                    reason: "output shape must be non-empty and nonzero".into(),
                });
            }
            let full_elements = shape.iter().try_fold(1usize, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or(WeightRecipeError::ArithmeticOverflow(
                        "contiguous source element count",
                    ))
            })?;
            let selected_elements = selected.iter().try_fold(1usize, |count, dimension| {
                count
                    .checked_mul(*dimension)
                    .ok_or(WeightRecipeError::ArithmeticOverflow(
                        "contiguous selected element count",
                    ))
            })?;
            let end = offset_elements.checked_add(selected_elements).ok_or(
                WeightRecipeError::ArithmeticOverflow("contiguous selection end"),
            )?;
            if end > full_elements {
                return Err(WeightRecipeError::SelectionPushdownUnsupported {
                    operation: "contiguous source selection",
                    reason: format!(
                        "scalar span {offset_elements}..{end} exceeds {full_elements} elements"
                    ),
                });
            }
            shape = selected.clone();
        }
    }
    Ok(shape)
}

fn infer_join(
    store: &dyn WeightStore,
    axis: usize,
    inputs: &[DerivedWeightRecipe],
    stack: bool,
) -> Result<WeightRecipeMetadata, WeightRecipeError> {
    if inputs.is_empty() {
        return Err(WeightRecipeError::EmptyInputs);
    }
    let metadata = inputs
        .iter()
        .map(|input| input.infer(store))
        .collect::<Result<Vec<_>, _>>()?;
    let first = &metadata[0];
    if metadata.iter().any(|item| item.dtype != first.dtype) {
        return Err(WeightRecipeError::DtypeMismatch);
    }
    let rank = first.shape.len();
    if axis > rank || (!stack && axis == rank) {
        return Err(WeightRecipeError::InvalidJoinAxis { axis, rank, stack });
    }
    if stack {
        if metadata.iter().any(|item| item.shape != first.shape) {
            return Err(WeightRecipeError::ShapeMismatch);
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
                return Err(WeightRecipeError::ShapeMismatch);
            }
            shape[axis] = shape[axis].checked_add(item.shape[axis]).ok_or(
                WeightRecipeError::ArithmeticOverflow("concatenate dimension"),
            )?;
        }
        metadata_for(shape, first.dtype.clone())
    }
}

fn metadata_for(
    shape: Vec<usize>,
    dtype: RecipeDtype,
) -> Result<WeightRecipeMetadata, WeightRecipeError> {
    let elements = element_count(&shape, "recipe output")?;
    let byte_len = elements
        .checked_mul(dtype.byte_width()?)
        .ok_or(WeightRecipeError::ArithmeticOverflow("recipe output bytes"))?;
    if byte_len == 0 {
        return Err(WeightRecipeError::ZeroSizedOutput);
    }
    Ok(WeightRecipeMetadata {
        shape,
        dtype,
        byte_len,
    })
}

fn element_count(shape: &[usize], context: &'static str) -> Result<u64, WeightRecipeError> {
    shape.iter().try_fold(1u64, |count, dimension| {
        let dimension = u64::try_from(*dimension)
            .map_err(|_| WeightRecipeError::ArithmeticOverflow(context))?;
        count
            .checked_mul(dimension)
            .ok_or(WeightRecipeError::ArithmeticOverflow(context))
    })
}

fn usize_to_i32(value: usize, context: &'static str) -> Result<i32, WeightRecipeError> {
    i32::try_from(value).map_err(|_| WeightRecipeError::ArithmeticOverflow(context))
}

/// Structured validation and materialization failures for derived weights.
#[derive(Debug, thiserror::Error)]
pub enum WeightRecipeError {
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
    WeightStore(#[from] crate::runtime::checkpoint::store::WeightStoreError),
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
    use crate::runtime::checkpoint::store::SafetensorsWeightStore;

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
            dtype: Dtype::Uint8,
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
            dtype: Dtype::Float32,
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
        let diagnostics = store.diagnostics().unwrap();
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
            dtype: Dtype::Uint8,
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
            dtype: Dtype::Uint8,
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
            Err(WeightRecipeError::SelectionPushdownUnsupported {
                operation: "view",
                ..
            })
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
            Err(WeightRecipeError::ElementCountMismatch { .. })
        ));
        let transpose = DerivedWeightRecipe::Transpose {
            input: Box::new(source("left")),
            axes: vec![0, 0],
        };
        assert!(matches!(
            transpose.infer(store.as_ref()),
            Err(WeightRecipeError::InvalidPermutation { .. })
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
