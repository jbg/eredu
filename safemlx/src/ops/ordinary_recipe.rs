//! Ordinary wrapper sources for explicitly recorded scalar/array calls.
use crate::error::Result;
use crate::utils::{VectorArray, guard::Guarded, runtime_lock::RuntimeLockGuard};
use crate::{Array, Dtype, EvaluatedArray, OperationEvent, OrdinaryControlPopulation, Stream};
use std::mem::{size_of, size_of_val};

/// One actual call in a shared recipe traversal, not a native primitive census.
/// Unary and binary variants describe the common C/Rust argument layouts.
/// The selected numerical source independently qualifies each actual operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OrdinaryRecipeCall {
    /// `Array::try_from_f32`, through the typed scalar/slice constructor.
    ScalarF32,
    /// `Array::try_from_int`, using the same typed four-byte scalar producer.
    ScalarI32,
    /// `Array::try_from_scalar::<u32>`, through the typed four-byte constructor.
    ScalarU32,
    /// Borrowed one-byte scalar producer; the common fixed scalar frame uses
    /// the conservative four-byte transport layout.
    ScalarU8,
    /// `Array::try_from_bool`, using the same typed one-byte scalar producer.
    /// The shared four-byte scalar transport conservatively covers its frames.
    ScalarBool,
    /// `Array::arange::<i32, i32>`, with integer endpoints and optional step.
    ArangeI32,
    /// A rank-one `Array::try_from_slice::<i32>` with borrowed input cells.
    IndicesI32 {
        /// The actual borrowed slice length, representable in the i32 shape.
        elements: usize,
    },
    /// `Array::try_from_slice::<f32>` with a borrowed payload and shape.
    SliceF32 {
        /// Number of borrowed f32 cells; tensor payload is quoted separately.
        elements: usize,
        /// Number of dimensions in the borrowed shape.
        rank: usize,
    },
    /// `Array::try_from_slice::<i32>` with a borrowed payload and shape.
    SliceI32 {
        /// Number of borrowed i32 cells; tensor payload is quoted separately.
        elements: usize,
        /// Number of dimensions in the borrowed shape.
        rank: usize,
    },
    /// `Array::try_from_slice::<u8>` with borrowed payload and shape. The
    /// common four-byte slice transport conservatively covers the byte caller.
    SliceU8 {
        /// Number of borrowed bytes; tensor payload is quoted separately.
        elements: usize,
        /// Number of dimensions in the borrowed shape.
        rank: usize,
    },
    /// `Array::try_from_slice::<bool>` with the real borrowed expanded mask.
    /// The common four-byte slice transport covers this one-byte caller.
    SliceBool {
        /// Number of borrowed Boolean cells; backing is a separate source.
        elements: usize,
        /// Number of entries in the actual borrowed shape.
        rank: usize,
    },
    /// One array/stream unary call such as log, exp, square, tanh or sigmoid.
    Unary,
    /// One two-array/stream call, including arithmetic, comparison and matmul.
    Binary,
    /// `Array::reshape` with a caller-owned shape vector.
    Reshape {
        /// Number of entries in the borrowed shape.
        rank: usize,
    },
    /// `broadcast_to` with the actual borrowed destination shape.
    Broadcast {
        /// Number of entries in the borrowed destination shape.
        rank: usize,
    },
    /// `topk_axis`, with explicit count and axis.
    TopKAxis,
    /// `argpartition_axis`, with explicit partition index and axis.
    ArgPartitionAxis,
    /// One-array explicit-axis call, such as argsort.
    UnaryAxis,
    /// Cumulative sum with an explicit axis and default reverse/inclusive flags.
    CumulativeSumAxis,
    /// A typed full constructor borrowing an actual scalar array.
    Full {
        /// Number of dimensions in the borrowed destination shape.
        rank: usize,
    },
    /// `Array::swap_axes`, forwarding two explicit axes to the same free worker.
    SwapAxes,
    /// Flattened take with two array arguments and no explicit axis.
    FlatTake,
    /// Flattened argsort with one array argument and no explicit axis.
    FlatArgsort,
    /// The actual optional-index gather-matmul wrapper, with its sorted flag.
    GatherMm,
    /// Clip with actual f32 scalar limits, constructing one scalar per present bound.
    Clip {
        /// The actual minimum bound is a supplied f32 scalar.
        minimum: bool,
        /// The actual maximum bound is a supplied f32 scalar.
        maximum: bool,
    },
    /// Three-array scatter with one explicit axis.
    ScatterSingle,
    /// Zeros with the input's shape and dtype, through its one-array wrapper.
    ZerosLike,
    /// Basic tuple indexing without array indices or ellipsis. Inline controls
    /// cover up to four input/output axes and four index declarations.
    BasicIndex {
        /// Number of input coordinate axes passed to the shared Slice.
        input_rank: usize,
        /// Number of output dimensions after optional new-axis/removal reshape.
        output_rank: usize,
        /// Number of borrowed tuple index declarations, between one and four.
        operations: usize,
        /// Whether the real basic indexing branch performs its final reshape.
        reshape: bool,
    },
    /// `Array::transpose_axes`, including the C wrapper's copied axis vector.
    Transpose {
        /// Number of entries in the borrowed axis permutation.
        rank: usize,
    },
    /// `Array::as_dtype`, including calls that preserve the dtype.
    Cast,
    /// The default axis-reversal transpose with no caller axis vector.
    TransposeDefault,
    /// One explicit-axis expand-dimensions call.
    ExpandDims,
    /// Squeeze over a caller-owned vector of axes.
    SqueezeAxes {
        /// Number of actual axes copied by the C wrapper.
        axes: usize,
    },
    /// One sum/mean/argmin-style axis reduction with explicit keep-dimensions.
    ReduceAxis,
    /// Three-array element selection through the ordinary where wrapper.
    Select,
    /// Two-array RMS normalization with its explicit epsilon.
    RmsNorm,
    /// Three-array add-matmul with alpha/beta defaults converted by the wrapper.
    AddMm,
    /// `Array::view_dtype` with an explicit destination dtype.
    View,
    /// `Array::take_axis` with an owned indices array and explicit axis.
    Take,
    /// `stack_axis` or `concatenate_axis` over a borrowed array-reference slice.
    Join {
        /// Number of actual input references; must be nonzero.
        inputs: usize,
        /// Select stack; false selects concatenation and its singleton shortcut.
        stack: bool,
    },
    /// `contiguous` with explicit row/column-order selection.
    Contiguous,
    /// `Array::all` over every axis with explicit keep-dimensions selection.
    All,
    /// `Array::any` over every axis with explicit keep-dimensions selection.
    Any,
    /// An actual Bool scalar: a differing dtype needs its explicit Cast call.
    BoolScalarRead,
    /// An actual U32 scalar read, without a dtype conversion or array clone.
    U32ScalarRead,
    /// Synchronous `eval` of a fixed one-, two- or three-element borrowed root array.
    Evaluate {
        /// Exact fixed root-array length; other iterator producers need their own census.
        inputs: usize,
    },
    /// Borrow an already completed array through `Array::evaluated`.
    BorrowedEvaluation,
    /// Ordinary zeros/ones constructor with one borrowed shape and explicit dtype.
    Fill {
        /// Number of dimensions in the borrowed shape.
        rank: usize,
    },
    /// Sum over every axis with an explicit keep-dimensions argument.
    SumAll,
    /// Three-array scatter addition along one explicit axis.
    ScatterAddAxis,
    /// Direct borrowed static Slice with positive coordinate strides.
    StaticSlice {
        /// The actual input rank shared by all three coordinate vectors.
        rank: usize,
    },
    /// Direct borrowed static SliceUpdate with matching input/update dtype and rank.
    StaticSliceUpdate {
        /// The actual input rank shared by all three coordinate vectors.
        rank: usize,
    },
}
/// Fixed C/Rust wrapper metadata and separately observed argument-vector storage.
/// Tensor backing, ArrayDesc/Data/primitive storage, Eval/backend work and opaque
/// native error-reporting allocations remain separate source contributions.
#[derive(Clone, Copy, Debug)]
pub struct OrdinaryRecipeWrapperControls {
    metadata: usize,
    observed: Option<OrdinaryControlPopulation>,
}
impl OrdinaryRecipeWrapperControls {
    /// C-owned shells and fixed safe/native call controls, excluding observed storage.
    pub fn metadata_bytes(self) -> usize {
        self.metadata
    }
    /// Only the C caller's append-built ArrayVector, if this call constructs it.
    /// The enclosing equation separately counts primitive and Eval containers.
    pub fn observed_controls(self) -> Option<OrdinaryControlPopulation> {
        self.observed
    }
}
fn sum<const N: usize>(parts: [usize; N]) -> Option<usize> {
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
fn guard<T: Guarded>() -> Option<usize> {
    sum([
        size_of::<<T as Guarded>::Guard>(),
        size_of::<T>(),
        size_of::<Result<T>>(),
        size_of::<Result<T>>(),
        size_of::<i32>(),
        size_of::<Option<crate::OriginalScopeObserver>>(),
        // Both ordinary guard entries perform the TLS-only try_current call.
        // It returns None: no Original observer/recovery owner is constructed.
        ordinary_current_scope_controls()?,
        ordinary_current_scope_controls()?,
    ])
}
/// Fixed ordinary Array result-guard frames, including the checked ambient
/// scope lookup. This describes caller metadata and grants no native scope.
pub fn ordinary_array_result_guard_control_bytes() -> Option<usize> {
    guard::<Array>()
}
fn ordinary_current_scope_controls() -> Option<usize> {
    sum([
        size_of::<safemlx_sys::mlx_submission_observer>(),
        size_of::<*mut safemlx_sys::mlx_submission_observer>(),
        size_of::<Result<Option<crate::OriginalScopeObserver>>>(),
        size_of::<Option<crate::OriginalScopeObserver>>(),
        size_of::<*const ()>(),
        size_of::<i32>(),
    ])
}
impl OrdinaryRecipeCall {
    /// Quote the linked ordinary wrapper without constructing any native value.
    /// Caller-owned Vec<Array>, Vec<&Array>, index and rank-vector cells are
    /// excluded. Transpose's own C std::vector<int> is included. Native Shape
    /// arguments are qualified only within their linked inline capacity.
    pub fn control_bytes(self) -> Option<OrdinaryRecipeWrapperControls> {
        if let Self::BasicIndex {
            input_rank,
            output_rank,
            operations,
            reshape,
        } = self
        {
            if input_rank > 4
                || output_rank > 4
                || !(1..=4).contains(&operations)
                || (!reshape && input_rank != output_rank)
            {
                return None;
            }
            // Both entries use slice_device's actual C signature. StaticSlice's
            // additional checked-coordinate transports are retained as unused
            // framing allowance; the concrete basic worker stays inline.
            let mut result = Self::StaticSlice { rank: input_rank }.control_bytes()?;
            result.metadata = result
                .metadata
                .checked_add(super::indexing::inline_basic_index_control_bytes()?)?;
            if reshape {
                result.metadata = result.metadata.checked_add(
                    Self::Reshape { rank: output_rank }
                        .control_bytes()?
                        .metadata,
                )?;
            }
            return Some(result);
        }
        let (kind, rank) = match self {
            Self::ScalarF32
            | Self::ScalarI32
            | Self::ScalarU32
            | Self::ScalarU8
            | Self::ScalarBool => (0, 0),
            Self::ArangeI32 => (27, 0),
            Self::IndicesI32 { elements } => {
                i32::try_from(elements).ok()?;
                elements.checked_mul(size_of::<i32>())?;
                (0, 1)
            }
            Self::SliceF32 { elements, rank }
            | Self::SliceI32 { elements, rank }
            | Self::SliceU8 { elements, rank }
            | Self::SliceBool { elements, rank } => {
                elements.checked_mul(size_of::<f32>())?;
                (0, rank)
            }
            Self::Unary | Self::TransposeDefault | Self::FlatArgsort | Self::ZerosLike => (1, 0),
            Self::Binary | Self::FlatTake => (2, 0),
            Self::Reshape { rank } | Self::Broadcast { rank } => (3, rank),
            Self::TopKAxis | Self::ArgPartitionAxis | Self::SwapAxes => (22, 0),
            Self::GatherMm => (25, 0),
            Self::Clip { minimum, maximum } if minimum || maximum => (26, 0),
            Self::Clip { .. } => return None,
            Self::UnaryAxis => (11, 0),
            Self::CumulativeSumAxis => (23, 0),
            Self::Full { rank } => (24, rank),
            Self::Transpose { rank } => (4, rank),
            Self::SqueezeAxes { axes } => (4, axes),
            Self::ExpandDims => (11, 0),
            Self::ReduceAxis => (12, 0),
            Self::Select => (13, 0),
            Self::RmsNorm => (14, 0),
            Self::AddMm => (15, 0),
            Self::Cast | Self::View => (5, 0),
            Self::Take => (6, 0),
            Self::Join { inputs, .. } => {
                if inputs == 0 {
                    return None;
                }
                (7, 0)
            }
            Self::Contiguous => (8, 0),
            Self::All | Self::Any => (9, 0),
            Self::BoolScalarRead => (10, 0),
            Self::U32ScalarRead => (16, 0),
            Self::Fill { rank } => (17, rank),
            Self::SumAll => (9, 0),
            Self::ScatterAddAxis | Self::ScatterSingle => (18, 0),
            Self::BasicIndex { .. } => return None,
            Self::StaticSlice { rank } => (19, rank),
            Self::StaticSliceUpdate { rank } => (28, rank),
            Self::Evaluate { inputs: 1 | 2 | 3 } => (20, 0),
            Self::Evaluate { .. } => return None,
            Self::BorrowedEvaluation => (21, 0),
        };
        let mut native = 0usize;
        // SAFETY: checked sizeof/range query writes only the initialized scalar.
        if !unsafe { safemlx_sys::mlx_ordinary_array_wrapper_controls(&mut native, kind, rank) } {
            return None;
        }
        let mut observed = None;
        let wrapper = match self {
            Self::ArangeI32 => sum([
                guard::<Array>()?,
                size_of::<(Option<i32>, i32, Option<i32>, &Stream)>(),
                size_of::<Option<i32>>().checked_mul(2)?,
                size_of::<Option<f64>>().checked_mul(3)?,
                size_of::<[f64; 3]>(),
                size_of::<Dtype>(),
                size_of::<(&f64, &f64, &f64, &Dtype, &&Stream)>(),
            ])?,
            Self::ScalarF32
            | Self::ScalarI32
            | Self::ScalarU32
            | Self::ScalarU8
            | Self::ScalarBool => sum([
                guard::<Array>()?,
                size_of::<f32>(),
                size_of::<&f32>(),
                size_of::<&[f32]>(),
                size_of::<&[i32]>(),
                size_of::<std::slice::Iter<'static, i32>>(),
                size_of::<(&[f32], &[i32], i32)>(),
                size_of::<usize>(),
                size_of::<i32>(),
                size_of::<Option<usize>>(),
                size_of::<Dtype>(),
            ])?,
            Self::IndicesI32 { .. }
            | Self::SliceF32 { .. }
            | Self::SliceI32 { .. }
            | Self::SliceU8 { .. }
            | Self::SliceBool { .. } => sum([
                guard::<Array>()?,
                size_of::<&[i32]>(),
                size_of::<&[i32]>(),
                size_of::<std::slice::Iter<'static, i32>>(),
                size_of::<(&[i32], &[i32], i32)>(),
                size_of::<usize>(),
                size_of::<i32>(),
                size_of::<Option<usize>>(),
                size_of::<Dtype>(),
            ])?,
            Self::Unary | Self::TransposeDefault | Self::FlatArgsort | Self::ZerosLike => sum([
                guard::<Array>()?,
                size_of::<(&Array, &Stream)>(),
                size_of::<(&Array, &&Stream)>(),
            ])?,
            Self::Binary | Self::FlatTake => sum([
                guard::<Array>()?,
                size_of::<(&Array, Array, &Stream)>(),
                size_of::<(&Array, &Array, &&Stream)>(),
            ])?,
            Self::Reshape { .. }
            | Self::Broadcast { .. }
            | Self::Transpose { .. }
            | Self::SqueezeAxes { .. } => sum([
                guard::<Array>()?,
                size_of::<(&Array, &[i32], &Stream)>(),
                size_of::<(&Array, &[i32], &&Stream)>(),
                size_of::<usize>(),
            ])?,
            Self::TopKAxis | Self::ArgPartitionAxis | Self::SwapAxes => sum([
                guard::<Array>()?,
                size_of::<(&Array, i32, i32, &Stream)>(),
                size_of::<(&Array, &i32, &i32, &&Stream)>(),
            ])?,
            Self::GatherMm => sum([
                guard::<Array>()?,
                size_of::<(
                    &Array,
                    &Array,
                    Option<&Array>,
                    Option<&Array>,
                    Option<bool>,
                    &Stream,
                )>(),
                size_of::<safemlx_sys::mlx_array>().checked_mul(6)?,
                size_of::<Option<&Array>>().checked_mul(2)?,
                size_of::<Option<bool>>(),
                size_of::<bool>(),
                size_of::<(
                    &safemlx_sys::mlx_array,
                    &safemlx_sys::mlx_array,
                    &safemlx_sys::mlx_array,
                    &safemlx_sys::mlx_array,
                    &bool,
                    &&Stream,
                )>(),
            ])?,
            Self::Clip { minimum, maximum } => sum([
                guard::<Array>()?,
                Self::ScalarF32
                    .control_bytes()?
                    .metadata
                    .checked_mul(usize::from(minimum) + usize::from(maximum))?,
                size_of::<(&Array, (f32, f32), &Stream)>(),
                size_of::<(Option<f32>, Option<f32>)>(),
                size_of::<(Option<Array>, Option<Array>)>(),
                size_of::<safemlx_sys::mlx_array>().checked_mul(2)?,
                size_of::<(
                    &Array,
                    &safemlx_sys::mlx_array,
                    &safemlx_sys::mlx_array,
                    &&Stream,
                )>(),
            ])?,
            Self::BasicIndex { .. } => return None,
            Self::CumulativeSumAxis => sum([
                guard::<Array>()?,
                size_of::<(&Array, i32, Option<bool>, Option<bool>, &Stream)>(),
                size_of::<Option<i32>>(),
                size_of::<(&Array, i32, bool, bool, &Stream)>(),
                size_of::<&Stream>(),
            ])?,
            Self::Full { .. } => sum([
                guard::<Array>()?,
                size_of::<(&[i32], Array, &Stream)>(),
                size_of::<(&[i32], &Array, &&Stream)>(),
                size_of::<Dtype>(),
            ])?,
            Self::ExpandDims | Self::UnaryAxis => sum([
                guard::<Array>()?,
                size_of::<(&Array, i32, &Stream)>(),
                size_of::<(&Array, &i32, &&Stream)>(),
            ])?,
            Self::ReduceAxis => sum([
                guard::<Array>()?,
                size_of::<(&Array, i32, bool, &Stream)>(),
                size_of::<(&Array, &i32, &bool, &&Stream)>(),
                size_of::<Option<bool>>(),
            ])?,
            Self::Select => sum([
                guard::<Array>()?,
                size_of::<(Array, Array, Array, &Stream)>(),
                size_of::<(&Array, &Array, &Array, &&Stream)>(),
            ])?,
            Self::RmsNorm => sum([
                guard::<Array>()?,
                size_of::<(&Array, &Array, f32, &Stream)>(),
                size_of::<(&Array, &Array, &f32, &&Stream)>(),
            ])?,
            Self::AddMm => sum([
                guard::<Array>()?,
                size_of::<(&Array, &Array, &Array, Option<f32>, Option<f32>, &Stream)>(),
                size_of::<(&Array, &Array, &Array, &Option<f32>, &Option<f32>, &&Stream)>(),
                size_of::<f32>() * 2,
                size_of::<safemlx_sys::mlx_array>() * 3,
            ])?,
            Self::Cast => Array::as_dtype_control_bytes()?.checked_add(guard::<Array>()?)?,
            Self::View => Array::view_dtype_control_bytes()?.checked_add(guard::<Array>()?)?,
            Self::Take => sum([
                guard::<Array>()?,
                size_of::<(&Array, Array, i32, &Stream)>(),
                size_of::<(&Array, &Array, &i32, &&Stream)>(),
            ])?,
            Self::Join {
                inputs: 1,
                stack: false,
            } => {
                // concatenate_axis takes its real singleton clone shortcut.
                native = 0;
                sum([
                    Array::ordinary_clone_control_bytes()?,
                    ordinary_current_scope_controls()?,
                    ordinary_current_scope_controls()?,
                    size_of::<(&[&Array], i32, &Stream)>(),
                ])?
            }
            Self::Join { inputs, .. } => {
                observed = Some(OperationEvent::ordinary_array_vector_control_layout(
                    inputs,
                )?);
                // The shared vector query includes its C call and core result
                // temporary. This is only the distinct final C Array owner.
                native = Array::inspection_clone_handle_bytes();
                // The same safe controls cover stack's identical input-vector
                // construction. Element buffers are observed, never debited here.
                sum([
                    super::shapes::concatenate_axis_wrapper_control_bytes()?,
                    guard::<Array>()?,
                    guard::<VectorArray>()?,
                    size_of::<std::slice::Iter<'static, &'static Array>>(),
                    size_of::<(&[&Array], i32, &Stream)>(),
                    size_of::<(&VectorArray, &i32, &&Stream)>(),
                    size_of::<i32>(),
                ])?
            }
            Self::Contiguous | Self::All | Self::Any | Self::SumAll => sum([
                guard::<Array>()?,
                size_of::<(&Array, bool, &Stream)>(),
                size_of::<(&Array, &bool, &&Stream)>(),
                size_of::<Option<bool>>(),
            ])?,
            Self::BoolScalarRead => sum([
                ordinary_current_scope_controls()?,
                guard::<()>()?,
                guard::<bool>()?,
                size_of::<(Array, &Stream)>(),
                size_of::<EvaluatedArray<'static>>(),
                size_of::<Result<EvaluatedArray<'static>>>(),
                size_of::<&EvaluatedArray<'static>>(),
                size_of::<&Array>(),
                size_of::<RuntimeLockGuard>(),
                size_of::<Dtype>(),
                size_of::<bool>(),
                size_of::<Result<bool>>(),
                size_of::<safemlx_sys::mlx_array>(),
            ])?,
            Self::Fill { .. } => sum([
                guard::<Array>()?,
                size_of::<(&[i32], &Stream)>(),
                size_of::<(&[i32], Dtype, &Stream)>(),
                size_of::<(&[i32], &Dtype, &&Stream)>(),
                size_of::<Dtype>(),
            ])?,
            Self::ScatterAddAxis | Self::ScatterSingle => sum([
                guard::<Array>()?,
                size_of::<(Array, &Array, &Array, i32, &Stream)>(),
                size_of::<(&Array, &Array, &Array, i32, &Stream)>(),
                size_of::<(&Array, &Array, &Array, &i32, &&Stream)>(),
            ])?,
            Self::StaticSlice { .. } => {
                type Axes = std::iter::Zip<
                    std::iter::Zip<
                        std::iter::Zip<
                            std::slice::Iter<'static, i32>,
                            std::slice::Iter<'static, i32>,
                        >,
                        std::slice::Iter<'static, i32>,
                    >,
                    std::slice::Iter<'static, i32>,
                >;
                sum([
                    guard::<Array>()?,
                    size_of::<(&Array, &[i32], &[i32], &[i32], &Stream)>() * 2,
                    size_of::<Axes>(),
                    size_of::<[&i32; 4]>(),
                    size_of::<[usize; 4]>(),
                    size_of::<[i32; 4]>(),
                    size_of::<Option<i32>>() * 2,
                    size_of::<Result<Array>>(),
                    size_of::<bool>(),
                ])?
            }
            Self::StaticSliceUpdate { .. } => sum([
                guard::<Array>()?,
                Array::static_slice_update_control_bytes()?,
            ])?,
            Self::Evaluate { inputs } => {
                observed = Some(OperationEvent::ordinary_array_vector_control_layout(
                    inputs,
                )?);
                let iterator = match inputs {
                    1 => size_of::<std::array::IntoIter<&Array, 1>>(),
                    2 => size_of::<std::array::IntoIter<&Array, 2>>(),
                    3 => size_of::<std::array::IntoIter<&Array, 3>>(),
                    _ => return None,
                };
                sum([
                    guard::<VectorArray>()?,
                    guard::<()>()?,
                    iterator,
                    size_of::<&Array>(),
                    size_of::<i32>(),
                    size_of::<RuntimeLockGuard>(),
                    size_of::<&VectorArray>(),
                    inputs.checked_mul(size_of::<&Array>())?,
                ])?
            }
            Self::BorrowedEvaluation => sum([
                ordinary_current_scope_controls()?,
                guard::<()>()?,
                size_of::<&Array>(),
                size_of::<EvaluatedArray<'static>>(),
                size_of::<Result<EvaluatedArray<'static>>>(),
                size_of::<RuntimeLockGuard>(),
                size_of::<safemlx_sys::mlx_array>(),
            ])?,
            Self::U32ScalarRead => sum([
                ordinary_current_scope_controls()?,
                guard::<()>()?,
                guard::<u32>()?,
                size_of::<(Array, &Stream)>(),
                size_of::<EvaluatedArray<'static>>(),
                size_of::<Result<EvaluatedArray<'static>>>(),
                size_of::<&EvaluatedArray<'static>>(),
                size_of::<&Array>(),
                size_of::<RuntimeLockGuard>(),
                size_of::<Dtype>(),
                size_of::<u32>(),
                size_of::<Result<u32>>(),
                size_of::<safemlx_sys::mlx_array>(),
            ])?,
        };
        Some(OrdinaryRecipeWrapperControls {
            metadata: native.checked_add(wrapper)?,
            observed,
        })
    }
}
