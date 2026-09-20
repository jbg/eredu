//! Bounds derived from the vendored MLX GPU copy/slicing and Metal unary,
//! binary, ternary and custom-kernel implementations. Ordinary pointwise
//! kernels read strided inputs directly. Custom kernels can copy each input
//! once to row-contiguous storage. All intermediate buffers are charged until
//! completion, even when MLX can release or donate them sooner.

use super::facts::{self, Aliases, Emitter, FactResult, Output, add, buffer_capacity, mul};
use super::*;

/// The architecture's exact prepared text-input placeholder. Its producer
/// already owns borrowed token data and native copy/source controls separately;
/// this equation accounts only the one resulting integer matrix backing.
/// Generic Initialize may mean a fill/copy graph and is deliberately unrelated.
pub(super) fn is_prepared_token_input(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(operation.kind, WorkspaceOperationKindView::Elementwise("prepared_token_input"))
        && operation.inputs.is_empty() && operation.outputs.len() == 1
        && operation.outputs.get(0).is_some_and(|output|
            matches!(output.dtype(), WorkspaceDtype::Int32 | WorkspaceDtype::Uint32)
                && matches!(output.shape(), [batch, positions] if *batch > 0 && *positions > 0))
}

/// The eager host scalar has one exact F32 value and no lazy fill/copy operator.
/// This describes Array::try_from_f32; generic Initialize retains its own bound.
pub(super) fn is_scalar_f32(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(operation.kind, WorkspaceOperationKindView::Elementwise("scalar_f32"))
        && operation.inputs.is_empty() && operation.outputs.len() == 1
        && operation.outputs.get(0).is_some_and(|output|
            output.dtype() == WorkspaceDtype::Float32 && output.shape().is_empty())
}

/// Closed eager byte seed used by the shared boundary alignment worker.
pub(super) fn is_scalar_u8(operation: WorkspaceOperationView<'_>) -> bool {
    matches!(operation.kind, WorkspaceOperationKindView::Elementwise("scalar_u8"))
        && operation.inputs.is_empty() && operation.outputs.len() == 1
        && operation.outputs.get(0).is_some_and(|output|
            output.dtype() == WorkspaceDtype::Uint8 && output.shape().is_empty())
}

/// Fixed Rust transports for the actual borrowed scalar constructor. Native
/// Array/Data/C-wrapper controls remain in the resident seed-source layout.
pub(super) fn scalar_f32_control_bytes() -> Option<usize> { scalar_control_bytes::<f32>() }
pub(super) fn scalar_u8_control_bytes() -> Option<usize> { scalar_control_bytes::<u8>() }
fn scalar_control_bytes<T>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let frames = [size_of::<T>() * 3, size_of::<&[T]>() * 2,
        size_of::<&[i32]>() * 2, size_of::<safemlx::Array>() * 2,
        size_of::<Result<safemlx::Array, safemlx::error::Exception>>() * 2,
        size_of::<(&[T], &[i32], i32)>(), size_of::<Option<usize>>(),
        size_of::<usize>() * 2, size_of::<i32>(),
        size_of::<std::slice::Iter<'_, i32>>(),
        size_of::<Result<i32, std::num::TryFromIntError>>(),size_of::<Option<usize>>()];
    frames.into_iter().try_fold(size_of_val(&frames), usize::checked_add)
}

/// The closed cast worker has already made one packed logical text-matrix copy.
/// Both tensor and host facts require this same exact descriptor contract.
pub(super) fn is_isolated_text_u32_cast(operation: &WorkspaceOperation) -> bool {
    is_isolated_text_u32_cast_view(operation.as_view())
}
pub(super) fn is_isolated_text_u32_cast_view(operation: WorkspaceOperationView<'_>) -> bool {
    let (Some([input]), Some([output])) = (operation.inputs.array(), operation.outputs.array())
    else {
        return false;
    };
    matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("cast_u32")
    ) && matches!(input.shape(), [1, positions] if *positions > 0)
        && output.shape() == input.shape()
        && input.dtype() == WorkspaceDtype::Int32
        && output.dtype() == WorkspaceDtype::Uint32
}

/// The closed nonempty capture selection emits this operation for actual half
/// sources and for future floating sources whose precision is erased. The latter
/// conservatively covers both F32's no-op and native F16/BF16 conversion. Nonempty
/// is essential: identity Slice need not repair an old empty descriptor. The
/// capture worker and future program both skip that conversion.
pub(super) fn is_selected_capture_f32_cast(operation: &WorkspaceOperation) -> bool {
    is_selected_capture_f32_cast_view(operation.as_view())
}
pub(super) fn is_selected_capture_f32_cast_view(operation: WorkspaceOperationView<'_>) -> bool {
    let (Some([input]), Some([output])) = (operation.inputs.array(), operation.outputs.array())
    else {
        return false;
    };
    matches!(
        operation.kind,
        WorkspaceOperationKindView::Elementwise("capture_cast_f32" | "cast_f32")
    ) && input.dtype() == WorkspaceDtype::Float32
        && output.dtype() == WorkspaceDtype::Float32
        && input.shape() == output.shape()
        && output.shape().len() <= 32
        && output.elements().is_ok_and(|n| n > 0)
}

/// Geometry shared by native seed/copy pricing and the fixed host buffer fact.
pub(super) fn generated_f32_plan(
    operation: &WorkspaceOperation,
) -> Result<eredu_nn::F32InitializationPlan<'_>, Error> {
    generated_f32_plan_view(operation.as_view()).map_err(MlxWorkspaceFactError::ordinary)
}
pub(super) fn generated_f32_plan_view(
    operation: WorkspaceOperationView<'_>,
) -> FactResult<eredu_nn::F32InitializationPlan<'_>> {
    let output = one_output(operation)?;
    if !operation.inputs.is_empty() || output.dtype() != WorkspaceDtype::Float32 {
        return invalid();
    }
    eredu_nn::F32InitializationPlan::new(output.shape()).map_err(MlxWorkspaceFactError::from)
}

/// The transfer descriptors keep the actual floating representation separate
/// from the conservative Float32 workspace extent class. Stored Host handles
/// have no Tensor source/output alias; the native source bank authenticates
/// the matching completed store occurrence separately.
pub(super) fn host_transfer_dtype(operation: WorkspaceOperationView<'_>) -> Option<safemlx::Dtype> {
    use WorkspaceOperationKindView as K;
    let (dtype, layout) = match operation.kind {
        K::HostTransferFloating(dtype) => {
            let (Some([input]), Some([output])) =
                (operation.inputs.array(), operation.outputs.array())
            else {
                return None;
            };
            if input.dtype() != output.dtype() || input.shape() != output.shape() {
                return None;
            }
            (dtype, input)
        }
        K::HostStoreFloating(_, dtype) => {
            let Some([input]) = operation.inputs.array() else {
                return None;
            };
            if !operation.outputs.is_empty() {
                return None;
            }
            (dtype, input)
        }
        K::HostLoadStoredFloating(_, dtype) => {
            let Some([output]) = operation.outputs.array() else {
                return None;
            };
            if !operation.inputs.is_empty() {
                return None;
            }
            (dtype, output)
        }
        _ => return None,
    };
    if layout.dtype() != WorkspaceDtype::Float32
        || layout.shape().is_empty()
        || layout.shape().len() > 4
        || layout.shape().iter().any(|n| *n <= 0)
        || layout.representation().is_some_and(|r| r.dtype() != dtype)
    {
        return None;
    }
    Some(match dtype {
        WorkspaceFloatingType::Float32 => safemlx::Dtype::Float32,
        WorkspaceFloatingType::Float16 => safemlx::Dtype::Float16,
        WorkspaceFloatingType::Bfloat16 => safemlx::Dtype::Bfloat16,
    })
}

pub(super) fn operation_bound(
    operation: &WorkspaceOperation,
    allocation: NativeAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary_with(
        |sink| emit(operation.as_view(), allocation, sink),
        |error| super::ordinary_error(operation, error),
    )
}

pub(super) fn emit(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    use WorkspaceOperationKindView as Kind;
    if matches!(operation.kind, Kind::Elementwise("masked_scatter")) {
        return super::masked_scatter::emit(operation, allocation, sink);
    }
    let bound = match &operation.kind {
        Kind::IndexedElementSelect | Kind::IndexedElementUpdate => {
            if !indexed_elements(operation) { return invalid(); }
            let output = operation.outputs.get(0).expect("validated element output");
            // The one-index Gather reads source strides directly. Exact-dtype
            // Scatter copies its complete base once, then overwrites sparse
            // cells; its rank-one broadcast/reshape preparation only aliases.
            one(sink, allocation, Output::Allocate(buffer_capacity(allocation, output.bytes()?)?), 0,
                format_args!("single-index Gather or general Scatter overwrite; retained I32 indices, equal physical update precision, and no dense expert tensor"))?
        }
        Kind::IndexedRowAdd => {
            if !indexed_row_add(operation) { return invalid(); }
            let base=operation.inputs.get(0).expect("validated indexed base");
            let updates=operation.inputs.get(2).expect("validated indexed updates");
            let output=buffer_capacity(allocation,base.bytes()?)?;
            // ops.cpp::scatter_axis casts updates once, broadcasts both indices
            // and updates, broadcasts all three inputs ignoring axis0, then
            // emits ScatterAxis::Sum. Retain every possible copy independently.
            // Integral normalized IDs and floating updates are at most4 bytes.
            let route_bytes=mul(updates.elements()?,4)?;
            let scratch=add(mul(buffer_capacity(allocation,route_bytes)?,5)?,output)?;
            one(sink,allocation,Output::AllocateOrAliasInputs{bytes:output,inputs:Aliases::Slice(&[0])},scratch,
                format_args!("rank-two ScatterAxis Sum with actual integer source; update cast, five broadcasts and output copy; repeated row destinations accumulate"))?
        }
        Kind::Elementwise("prepared_token_input") => {
            if !is_prepared_token_input(operation) { return invalid(); }
            one(sink, allocation, Output::Allocate(buffer_capacity(allocation, one_output(operation)?.bytes()?)?),
                0, format_args!("one integer matrix supplied by the separately admitted prepared input; no model fill/copy primitive"))?
        }
        Kind::ValueCompletion | Kind::ValueRetention => {
            if operation.inputs.is_empty() || !operation.outputs.is_empty() {
                return invalid();
            }
            sink.finish(0, format_args!("completion preserves supplied value storage; exact native traversal and source admission are separate"))?
        }
        Kind::ParameterPlaceholder => emit_parameter_placeholder(operation, allocation, sink)?,
        Kind::Pad(mode) => {
            let output = one_output(operation)?;
            let Some([input]) = operation.inputs.array() else {
                return invalid();
            };
            if input.dtype() != output.dtype()
                || input.shape().len() != output.shape().len()
                || input.shape().iter().zip(output.shape()).any(|(a, b)| a > b)
            {
                return invalid();
            }
            let changed_axes = input
                .shape()
                .iter()
                .zip(output.shape())
                .filter(|(a, b)| a != b)
                .count() as u64;
            let bytes = buffer_capacity(allocation, output.bytes()?.max(4))?;
            let scalar = buffer_capacity(allocation, 4)?;
            let scratch = match mode {
                // pad_gpu fills its output and copies the strided input into a
                // shared output slice. safemlx constructs an I32 zero and casts
                // it to the input dtype; neither copy requires another array.
                eredu_nn::PadMode::Constant => mul(2, scalar)?,
                eredu_nn::PadMode::Edge => {
                    if input
                        .shape()
                        .iter()
                        .zip(output.shape())
                        .any(|(&a, &b)| a == 0 && b > 0)
                    {
                        return invalid();
                    }
                    // edge_pad creates zeros, copies the input with one full
                    // slice_update, then performs at most two further updates
                    // for each padded axis. Edge slices/broadcasts alias their
                    // preceding result. Retain every full copy, without relying
                    // on donation. Width splits are unnecessary for this bound.
                    add(mul(add(1, mul(2, changed_axes)?)?, bytes)?, mul(4, scalar)?)?
                }
            };
            one(
                sink,
                allocation,
                Output::AllocateOrAliasInputs {
                    bytes,
                    inputs: Aliases::Slice(&[0]),
                },
                scratch,
                format_args!(
                    "{}",
                    "constant pad fills/copies one output; edge pad retains zeros, input update and up to two full updates per padded axis; fill scalar casts included; no host payload vectors"
                ),
            )?
        }
        Kind::Elementwise(_) if super::zero_fill::dtype(operation).is_some() => {
            let (_,_,seed_bytes)=super::zero_fill::dtype(operation).expect("qualified zero fill");
            let output=one_output(operation)?;
            one(sink,allocation,Output::Allocate(buffer_capacity(allocation,output.bytes()?)?),
                buffer_capacity(allocation,seed_bytes)?,format_args!("typed eager zero plus optional Broadcast and Full"))?
        }
        Kind::Elementwise("scalar_u8") => {
            if !is_scalar_u8(operation) { return invalid(); }
            one(sink, allocation, Output::Allocate(buffer_capacity(allocation, 1)?), 0,
                format_args!("one eager U8 scalar copied directly to native storage"))?
        }
        Kind::Elementwise("scalar_f32") => {
            if !is_scalar_f32(operation) { return invalid(); }
            one(sink, allocation, Output::Allocate(buffer_capacity(allocation, 4)?), 0,
                format_args!("one eager F32 scalar copied directly to native storage"))?
        }
        Kind::Initialize | Kind::InitializeFloating(_) | Kind::GeneratedF32Initialization => {
            let output = one_output(operation)?;
            if operation.inputs.len() > 1 {
                return invalid();
            }
            if matches!(operation.kind, Kind::InitializeFloating(_))
                && (!operation.inputs.is_empty() || output.dtype() != WorkspaceDtype::Float32)
            {
                return invalid();
            }
            if matches!(operation.kind, Kind::GeneratedF32Initialization) {
                generated_f32_plan_view(operation)?;
            }
            // Host input, scalar fill and zeros_like share this envelope.
            // Charge a complete source plus a possible copy and fill scalar.
            // Even an empty broadcast can retain its scalar backing buffer.
            let bytes = buffer_capacity(allocation, output.bytes()?.max(4))?;
            one(
                sink,
                allocation,
                Output::Allocate(bytes),
                add(bytes, buffer_capacity(allocation, 4)?)?,
                format_args!("{}", "host/scalar initialization and copy"),
            )?
        }
        Kind::View(name) if super::byte_view::selected(name).is_some() => {
            let Some((bytes,_))=super::byte_view::inspect(operation) else{return Ok(None);};
            // View's exact GPU worker shares storage or creates one contiguous
            // temporary of the same physical byte size and shares that result.
            one(sink,allocation,Output::AllocateOrAliasInputs{
                bytes:buffer_capacity(allocation,bytes.max(one_output(operation)?.bytes()?))?,inputs:Aliases::Slice(&[0]),
            },0,format_args!("byte reinterpretation with possible strided copy"))?
        }
        Kind::Transpose(axes) => {
            let output = one_output(operation)?;
            let Some(input) = operation.inputs.get(0) else {return invalid();};
            if operation.inputs.len()!=1 || input.dtype()!=output.dtype() ||
                axes.len()!=input.shape().len() || axes.len()!=output.shape().len() ||
                axes.iter().enumerate().any(|(i,&axis)| axis>=axes.len() ||
                    axes[..i].contains(&axis) || output.shape()[i]!=input.shape()[axis]) {
                return invalid();
            }
            one(sink,allocation,Output::AliasInput(0),0,
                format_args!("exact axis permutation shares the retained source buffer"))?
        }
        Kind::View(name) => {
            if !matches!(
                *name,
                "broadcast" | "squeeze" | "expand_dims" | "reshape"
            ) {
                return Ok(None);
            }
            let output = one_output(operation)?;
            if operation.inputs.len() != 1
                || operation.inputs.get(0).unwrap().dtype() != output.dtype()
            {
                return invalid();
            }
            let storage = if *name == "reshape" {
                if operation.inputs.get(0).unwrap().elements()? != output.elements()? {
                    return invalid();
                }
                // reshape_gpu copies non-view-compatible strides; its only
                // additional tensor buffer is the result of that copy.
                Output::AllocateOrAliasInputs {
                    bytes: buffer_capacity(allocation, output.bytes()?)?,
                    inputs: Aliases::Slice(&[0]),
                }
            } else {
                Output::AliasInput(0)
            };
            one(
                sink,
                allocation,
                storage,
                0,
                format_args!("{}", "shared-buffer views; reshape may copy"),
            )?
        }
        Kind::StaticSlice {..} => {
            if !is_static_slice(operation) {return invalid();}
            one(sink,allocation,Output::AliasInput(0),0,
                format_args!("exact rank-preserving static Slice aliases the full source backing"))?
        }
        Kind::Index { selected_axes } => {
            let output = one_output(operation)?;
            if operation.inputs.len() != 1
                || operation.inputs.get(0).unwrap().dtype() != output.dtype()
            {
                return invalid();
            }
            if operation
                .inputs
                .get(0)
                .unwrap()
                .shape()
                .len()
                .checked_sub(*selected_axes)
                != Some(output.shape().len())
            {
                return invalid();
            }
            // Static slices share their input, but the native indexing wrapper
            // implements integer-axis removal with reshape. MLX may copy even
            // that reshape for strided slices. Dynamic gathers are separate.
            let storage = if *selected_axes == 0 {
                Output::AliasInput(0)
            } else {
                Output::AllocateOrAliasInputs {
                    bytes: buffer_capacity(allocation, output.bytes()?)?,
                    inputs: Aliases::Slice(&[0]),
                }
            };
            one(
                sink,
                allocation,
                storage,
                0,
                format_args!(
                    "{}",
                    "static slices alias; integer-axis removal may reshape-copy"
                ),
            )?
        }
        Kind::Concatenate => {
            let output = one_output(operation)?;
            if operation.inputs.is_empty()
                || operation
                    .inputs
                    .iter()
                    .any(|input| input.dtype() != output.dtype())
            {
                return invalid();
            }
            let elements = operation
                .inputs
                .iter()
                .try_fold(0, |sum, input| add(sum, input.elements()?))?;
            if elements != output.elements()? {
                return invalid();
            }
            // Metal concatenate allocates its result and copies directly into
            // aliased slices. The floating descriptor can represent mixed
            // F16/BF16/F32, so allow one dtype cast per input as well.
            let scratch = operation.inputs.iter().try_fold(0, |sum, input| {
                add(sum, buffer_capacity(allocation, input.bytes()?)?)
            })?;
            one(
                sink,
                allocation,
                Output::AllocateOrAliasInputs {
                    bytes: buffer_capacity(allocation, output.bytes()?)?,
                    inputs: Aliases::Range {
                        start: 0,
                        end: operation.inputs.len(),
                    },
                },
                scratch,
                format_args!("{}", "concatenate output and each possible input cast"),
            )?
        }
        Kind::HostStoreFloating(..) => {
            if host_transfer_dtype(operation).is_none() {
                return Ok(None);
            }
            let input = operation.inputs.get(0).expect("validated store input");
            sink.finish(
                buffer_capacity(allocation, input.bytes()?)?,
                format_args!("the existing Host store may compact its Device input once; its internal output aliases the separately accepted immutable Host destination, with no portable Tensor output"),
            )?
        }
        Kind::HostTransferFloating(_) | Kind::HostLoadStoredFloating(..) => {
            if host_transfer_dtype(operation).is_none() {
                return Ok(None);
            }
            one(
                sink,
                allocation,
                Output::Allocate(buffer_capacity(
                    allocation,
                    one_output(operation)?.bytes()?,
                )?),
                0,
                format_args!(
                    "retained immutable host transfer creates independent device storage; exact scalar type stays within the floating envelope"
                ),
            )?
        }
        Kind::CastFloating(_) => {
            let Some([input]) = operation.inputs.array() else {
                return invalid();
            };
            let output = one_output(operation)?;
            if input.dtype() != WorkspaceDtype::Float32
                || output.dtype() != WorkspaceDtype::Float32
                || input.shape() != output.shape()
                || output.shape().len() > 32
                || output.elements()? == 0
            {
                return Ok(None);
            }
            // ops.cpp::astype aliases equal types. Otherwise AsType dispatches
            // copy_gpu Vector (data_size <= elements) or General (logical size).
            // F16/BF16 conversions fit the retained four-byte float envelope;
            // the alias alternative preserves the entire actual source backing.
            one(
                sink,
                allocation,
                Output::AllocateOrAliasInputs {
                    bytes: buffer_capacity(allocation, output.bytes()?)?,
                    inputs: Aliases::Slice(&[0]),
                },
                0,
                format_args!(
                    "exact floating AsType: full logical conversion or unchanged backing alias; no host numerical staging"
                ),
            )?
        }
        Kind::Elementwise("capture_cast_f32" | "cast_f32") => {
            if !is_selected_capture_f32_cast_view(operation) {
                return Ok(None);
            }
            // AsType Vector copies the nonbroadcast contiguous span (<= N);
            // irregular strides use General and allocate N*4. Actual source
            // and all selected intermediates remain retained; no donation or
            // dtype-equality alias is assumed. A future F32 source may use less.
            one(
                sink,
                allocation,
                Output::Allocate(buffer_capacity(
                    allocation,
                    operation.outputs.get(0).unwrap().bytes()?,
                )?),
                0,
                format_args!(
                    "{}",
                    "closed nonempty static capture selection covers native F16/BF16 AsType(F32) and a future F32 no-op; Vector data_size <= logical elements, General allocates logical N*4; explicit cast bound despite erased precision; no numerical host staging"
                ),
            )?
        }
        Kind::Elementwise("is_finite") => {
            let Some([input]) = operation.inputs.array() else {
                return invalid();
            };
            let output = one_output(operation)?;
            if input.dtype() != WorkspaceDtype::Float32
                || output.dtype() != WorkspaceDtype::Bool
                || input.shape() != output.shape()
                || input.elements()? == 0
            {
                return invalid();
            }
            // Native floating isfinite = !(isposinf | isneginf | isnan).
            // Equal(+/-infinity), NotEqual(a,a), two Ors and Not create six
            // Bool outputs; both infinity seeds use the actual input dtype.
            // Equal and logical operations keep their same input dtypes, so
            // native AsType candidates alias. Strided pointwise kernels read
            // their sources directly. All six outputs remain counted.
            let boolean = buffer_capacity(allocation, output.bytes()?)?;
            one(
                sink,
                allocation,
                Output::Allocate(boolean),
                add(mul(5, boolean)?, mul(2, buffer_capacity(allocation, 4)?)?)?,
                format_args!(
                    "floating native isfinite: six actual Bool outputs and two at-most-F32 infinity seeds; full population retained; no host numeric staging"
                ),
            )?
        }
        Kind::Elementwise(name @ ("is_nan" | "is_positive_infinity" | "is_negative_infinity")) => {
            let Some([input]) = operation.inputs.array() else {
                return invalid();
            };
            let output = one_output(operation)?;
            if input.dtype() != WorkspaceDtype::Float32
                || output.dtype() != WorkspaceDtype::Bool
                || input.shape() != output.shape()
                || input.elements()? == 0
            {
                return invalid();
            }
            // Floating IsNaN is NotEqual(a,a); +/-Inf is Equal(a, scalar).
            // Equal input casts/broadcasts alias the same F32 storage. The
            // scalar is an actual eager four-byte source in the infinity case.
            let scalar = if *name == "is_nan" {
                0
            } else {
                buffer_capacity(allocation, 4)?
            };
            one(
                sink,
                allocation,
                Output::Allocate(buffer_capacity(allocation, output.bytes()?)?),
                scalar,
                format_args!(
                    "floating {name}: one Bool comparison output and actual F32 infinity seed when present; native casts/broadcasts alias exact F32 inputs"
                ),
            )?
        }
        Kind::Elementwise("bool_to_u32") => {
            let Some([input]) = operation.inputs.array() else {
                return invalid();
            };
            let output = one_output(operation)?;
            if input.dtype() != WorkspaceDtype::Bool
                || output.dtype() != WorkspaceDtype::Uint32
                || input.shape() != output.shape()
                || input.elements()? == 0
            {
                return invalid();
            }
            one(
                sink,
                allocation,
                Output::Allocate(buffer_capacity(allocation, output.bytes()?)?),
                0,
                format_args!(
                    "native Bool-to-U32 AsType writes one logical-size output; arbitrary strided input remains separately retained"
                ),
            )?
        }
        Kind::Elementwise("cast_u32") => {
            // This closed operation receives the packed text-matrix result of
            // isolated_copy followed by reshape [1, N]. It is not a bound for
            // arbitrary strided/padded dtype conversion. MLX AsType uses the
            // GPU copy kernel; same-width donation can alias its input.
            if !is_isolated_text_u32_cast_view(operation) {
                return Ok(None);
            }
            one(
                sink,
                allocation,
                Output::AllocateOrAliasInputs {
                    bytes: buffer_capacity(allocation, operation.outputs.get(0).unwrap().bytes()?)?,
                    inputs: Aliases::Slice(&[0]),
                },
                0,
                format_args!(
                    "{}",
                    "MLX AsType on the isolated packed I32 text matrix writes its logical U32 elements through copy_gpu, or donates the same-width input; no additional numerical staging"
                ),
            )?
        }
        Kind::Elementwise(name) => {
            // One unary: possible cast + output. One binary: two possible
            // casts + output. One ternary: three casts + output. These upper
            // counts deliberately do not depend on fusion or donation.
            let (arity, buffers, scalars) = match *name {
                "add" | "subtract" | "multiply" | "divide" | "logical_or" | "greater" | "less"
                | "greater_equal" | "less_equal" | "logical_and" => (2, 3, 0),
                "logical_not" => (1, 2, 0),
                "square" | "tanh" | "exp" | "log" => (1, 2, 0),
                "multiply_scalar" | "maximum_scalar" | "maximum_i32" | "equal_i32" => (1, 3, 1),
                "clip" => (3, 6, 0),
                "where" => (3, 4, 0),
                // SiLU fallback: widen + Negative + Exp/cast + Add's three
                // buffers + Divide's three + final cast = eleven. This also
                // covers the custom branch's widen/copy/output/final cast.
                "silu" => (1, 11, 1),
                // Widen, custom input copy, custom output, final cast.
                "sigmoid" => (1, 4, 1),
                // Widen + mul/exp/log1p/div/gt/where + final cast.
                "softplus" => (1, 19, 3),
                // Four binary operations and erf.
                "gelu" => (1, 14, 3),
                // Seven binary operations and tanh; five scalar inputs,
                // plus the cast/result of the scalar sqrt(2/pi).
                "gelu_approximate" => (1, 23, 7),
                // gt/exp/sub/mul/where.
                "elu" => (1, 15, 3),
                _ => return Ok(None),
            };
            if operation.inputs.len() != arity {
                return invalid();
            }
            if *name == "equal_i32" && one_output(operation)?.dtype() != WorkspaceDtype::Bool {
                return invalid();
            }
            if *name == "maximum_i32"
                && operation.inputs.get(0).unwrap().dtype() == WorkspaceDtype::Uint32
            {
                return Ok(None);
            }
            // A mixed signed/unsigned integer promotion can exceed four
            // bytes; that execution has no bound under this descriptor.
            if operation
                .inputs
                .iter()
                .any(|v| v.dtype() == WorkspaceDtype::Int32)
                && operation
                    .inputs
                    .iter()
                    .any(|v| v.dtype() == WorkspaceDtype::Uint32)
            {
                return Ok(None);
            }
            if matches!(
                *name,
                "silu" | "sigmoid" | "softplus" | "gelu" | "gelu_approximate" | "elu"
            ) && operation.inputs.get(0).unwrap().dtype() != WorkspaceDtype::Float32
            {
                return Ok(None);
            }
            if matches!(*name, "greater" | "less" | "greater_equal" | "less_equal")
                && (one_output(operation)?.dtype() != WorkspaceDtype::Bool
                    || operation
                        .inputs
                        .iter()
                        .any(|input| input.dtype() != WorkspaceDtype::Float32))
            {
                return invalid();
            }
            if matches!(*name, "logical_and" | "logical_not")
                && (one_output(operation)?.dtype() != WorkspaceDtype::Bool
                    || operation
                        .inputs
                        .iter()
                        .any(|input| input.dtype() != WorkspaceDtype::Bool))
            {
                return invalid();
            }
            check_pointwise_shape(operation)?;
            pointwise(operation, allocation, buffers, scalars, sink)?
        }
        Kind::GatedProduct(policy) => {
            policy.validate_fixed()?;
            let output = one_output(operation)?;
            if operation.inputs.len() != 2
                || operation.inputs.iter().any(|input| {
                    input.shape() != output.shape() || input.dtype() != WorkspaceDtype::Float32
                })
                || output.dtype() != WorkspaceDtype::Float32
            {
                return invalid();
            }
            let (mut buffers, mut scalars) = match policy.activation() {
                eredu_nn::GatedProductActivation::Silu if policy.sigmoid_multiplier() == 1.0 => {
                    // Same shared SiLU fused/fallback population as above;
                    // the final multiplication is added below.
                    (11, 1)
                }
                // Scale, native unary sigmoid, multiply original gate.
                eredu_nn::GatedProductActivation::Silu => (8, 1),
                eredu_nn::GatedProductActivation::GeluApproximate => (23, 7),
                _ => return Ok(None),
            };
            if policy.gate_upper_bound().is_some() {
                buffers += 3;
                scalars += 1;
            }
            if policy.up_absolute_bound().is_some() {
                buffers += 6;
                scalars += 2;
            }
            if policy.up_offset() != 0.0 {
                buffers += 3;
                scalars += 1;
            }
            pointwise(operation, allocation, buffers + 3, scalars, sink)?
        }
        Kind::PoolingMask(geometry) => {
            let output = one_output(operation)?;
            if !operation.inputs.is_empty()
                || output.shape() != [geometry.queries(), geometry.pooled()]
                || output.dtype() != WorkspaceDtype::Bool
            {
                return invalid();
            }
            let queries = buffer_capacity(allocation, mul(geometry.queries() as u64, 4)?.max(4))?;
            let pooled = buffer_capacity(allocation, mul(geometry.pooled() as u64, 4)?.max(4))?;
            one(
                sink,
                allocation,
                Output::Allocate(buffer_capacity(allocation, output.bytes()?.max(4))?),
                add(
                    add(mul(2, queries)?, pooled)?,
                    buffer_capacity(allocation, 4)?,
                )?,
                format_args!(
                    "{}",
                    "I32 pooled/query aranges, integer quotient vector and ratio scalar, contiguous reshape aliases and boolean comparison output; no host payload vector"
                ),
            )?
        }
        Kind::CausalMask(geometry) => {
            let output = one_output(operation)?;
            if !operation.inputs.is_empty()
                || output.shape() != [geometry.sequence(), geometry.keys()]
                || output.dtype() != WorkspaceDtype::Bool
            {
                return invalid();
            }
            let queries = buffer_capacity(allocation, mul(geometry.sequence() as u64, 4)?)?;
            let keys = buffer_capacity(allocation, mul(geometry.keys() as u64, 4)?)?;
            let mask = buffer_capacity(allocation, output.bytes()?)?;
            let mut scratch = add(queries, keys)?;
            if geometry.max_past().is_some() {
                // Earliest-key subtraction and scalar, second comparison,
                // and the first mask retained until the final logical_and.
                scratch = add(
                    scratch,
                    add(
                        add(queries, buffer_capacity(allocation, 4)?)?,
                        mul(2, mask)?,
                    )?,
                )?;
            }
            one(
                sink,
                allocation,
                Output::AllocateOrAliasInputs {
                    bytes: mask,
                    inputs: Aliases::Slice(&[]),
                },
                scratch,
                format_args!(
                    "{}",
                    "I32 coordinate aranges, optional window subtraction and boolean mask intermediates; no lengths input"
                ),
            )?
        }
        _ => return Ok(None),
    };
    Ok(Some(bound))
}

fn invalid<T>() -> FactResult<T> {
    Err(MlxWorkspaceFactError::descriptor(
        "invalid Metal basic-operation workspace descriptor",
    ))
}

fn parameter_placeholder_output(
    operation: WorkspaceOperationView<'_>,
) -> FactResult<WorkspaceLayoutView<'_>> {
    if !matches!(operation.kind, WorkspaceOperationKindView::ParameterPlaceholder)
        || !operation.inputs.is_empty()
        || operation.outputs.len() != 1
        || !operation.outputs.get(0).unwrap().shape().is_empty()
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid unloaded parameter scalar source descriptor",
        ));
    }
    Ok(operation.outputs.get(0).unwrap())
}

/// The eager scalar constructor is shared by CPU and Metal. This describes its
/// backing only: a retained constructor inventory must still certify the lazy
/// Broadcast/Full graph, which strict parameter binding replaces before Eval.
pub(super) fn emit_parameter_placeholder(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<WorkspaceOperationFacts> {
    let output = parameter_placeholder_output(operation)?;
    // Pinned zeros(shape, dtype) constructs array(0, dtype) eagerly on either
    // device. Each constructor owns one seed; no donation sharing is assumed.
    sink.output(Output::Allocate(buffer_capacity(allocation, output.dtype().bytes())?))?;
    sink.finish(0, format_args!(
        "unloaded parameter construction eagerly copies one dtype-sized scalar into shared native storage; logical weight fill stays unevaluated until replaced by binding; page={} bytes with bounded oversized cache reuse",
        allocation.page_size(),
    ))
}

pub(super) fn emit_parameter_placeholder_host(
    operation: WorkspaceOperationView<'_>,
    sink: &mut facts::HostEmitter<'_>,
) -> FactResult<WorkspaceHostFacts> {
    parameter_placeholder_output(operation)?;
    sink.finish(0, format_args!(
        "the stack scalar is copied directly into the already-priced shared native allocation; lazy fill/shape descriptors have no disjoint host numerical payload",
    ))
}
fn one_output(operation: WorkspaceOperationView<'_>) -> FactResult<WorkspaceLayoutView<'_>> {
    if operation.outputs.len() != 1 {
        return invalid();
    }
    Ok(operation.outputs.get(0).unwrap())
}
pub(super) fn check_pointwise_shape(operation: WorkspaceOperationView<'_>) -> FactResult<()> {
    let output = one_output(operation)?;
    for input in operation.inputs.iter() {
        if input.shape().len() > output.shape().len()
            || input
                .shape()
                .iter()
                .rev()
                .zip(output.shape().iter().rev())
                .any(|(a, b)| a != b && *a != 1)
        {
            return invalid();
        }
    }
    Ok(())
}
fn pointwise(
    operation: WorkspaceOperationView<'_>,
    allocation: NativeAllocationFacts,
    buffers: u64,
    scalars: u64,
    sink: &mut Emitter<'_>,
) -> FactResult<WorkspaceOperationFacts> {
    let output = one_output(operation)?;
    // A cast precedes broadcasting. With empty result dimensions an input
    // can be larger than the result, so bound both, including scalar storage.
    let elements = operation
        .inputs
        .iter()
        .try_fold(output.elements()?, |largest, input| {
            Ok::<_, MlxWorkspaceFactError>(largest.max(input.elements()?))
        })?;
    // Vendored dtype.cpp promotes U32 plus an eager I32 scalar to I64.
    // ops.cpp equal casts both operands before broadcasting and produces Bool.
    // Preserve those eight-byte intermediate allocations even though the
    // neutral result is Bool; wider-result integer equations remain rejected.
    let element_bytes = if matches!(
        operation.kind, WorkspaceOperationKindView::Elementwise("equal_i32")
    ) && operation.inputs.get(0).is_some_and(|input| input.dtype() == WorkspaceDtype::Uint32) {
        8
    } else {
        4
    };
    // Empty broadcasts can still have a scalar backing allocation.
    let bytes = buffer_capacity(allocation, mul(elements.max(1), element_bytes)?)?;
    Ok(one(
        sink,
        allocation,
        Output::AllocateOrAliasInputs {
            bytes,
            inputs: Aliases::Range {
                start: 0,
                end: operation.inputs.len(),
            },
        },
        add(
            mul(buffers - 1, bytes)?,
            mul(scalars, buffer_capacity(allocation, element_bytes)?)?,
        )?,
        format_args!(
            "pointwise equation: at most {buffers} tensor buffers and {scalars} scalar buffers with {element_bytes}-byte elements; includes casts/contiguous copies; retains all intermediates"
        ),
    )?)
}
fn one(
    sink: &mut Emitter<'_>,
    allocation: NativeAllocationFacts,
    output: Output<'_>,
    scratch: u64,
    equation: std::fmt::Arguments<'_>,
) -> FactResult<WorkspaceOperationFacts> {
    sink.output(output)?;
    sink.finish(scratch, format_args!("vendored MLX Metal {equation}; page={} bytes with bounded oversized cache reuse; active tensor buffers only, excluding cache residency, heap/driver/JIT storage", allocation.page_size()))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod placeholder_tests;

#[cfg(test)]
mod split_tests;

/// Coordinate qualification shared by the CPU source and ordinary Metal facts.
pub(super) fn is_static_slice(operation:WorkspaceOperationView<'_>)->bool {
    let WorkspaceOperationKindView::StaticSlice {starts,ends,strides}=operation.kind else {return false;};
    let (Some([input]),Some([output]))=(operation.inputs.array(),operation.outputs.array()) else {return false;};
    let rank=input.shape().len();
    if output.shape().len()!=rank || input.dtype()!=output.dtype() || starts.len()!=rank || ends.len()!=rank || strides.len()!=rank {return false;}
    for axis in 0..rank {
        let (a,b,step,n)=(starts[axis],ends[axis],strides[axis],input.shape()[axis]);
        if a<0 || b<a || b>n || step<=0 || b.checked_sub(a).and_then(|d|d.checked_add(step-1)).map(|d|d/step)!=Some(output.shape()[axis]) {return false;}
    }
    true
}

/// Exact geometry of the shared rank-two ScatterAxis::Sum worker.
pub(super) fn indexed_row_add(operation:WorkspaceOperationView<'_>)->bool {
    if !matches!(operation.kind,WorkspaceOperationKindView::IndexedRowAdd){return false;}
    let (Some([base,indices,updates]),Some([output]))=(operation.inputs.array(),operation.outputs.array()) else{return false;};
    base.shape().len()==2 && indices.shape().len()==2 && updates.shape().len()==2
        && indices.shape()[1]==1 && indices.shape()[0]==updates.shape()[0]
        && base.shape()[1]==updates.shape()[1] && base.shape()==output.shape()
        && base.dtype()==WorkspaceDtype::Float32 && updates.dtype()==base.dtype() && output.dtype()==base.dtype()
        && matches!(indices.dtype(),WorkspaceDtype::Int32|WorkspaceDtype::Uint32)
}

/// Exact descriptor of the shared sparse native element worker. The owning
/// source establishes index values before the numerical operation is admitted.
pub(super) fn indexed_elements(operation: WorkspaceOperationView<'_>) -> bool {
    let update = match operation.kind {
        WorkspaceOperationKindView::IndexedElementSelect => false,
        WorkspaceOperationKindView::IndexedElementUpdate => true,
        _ => return false,
    };
    if operation.inputs.len() != if update { 3 } else { 2 } || operation.outputs.len() != 1 { return false; }
    let source = operation.inputs.get(0).expect("checked indexed source");
    let indices = operation.inputs.get(1).expect("checked indexed source");
    let output = operation.outputs.get(0).expect("checked indexed output");
    if source.shape().len() != 1 || source.shape()[0] <= 0
        || indices.shape().len() != 1 || indices.dtype() != WorkspaceDtype::Int32
        || source.dtype() != WorkspaceDtype::Float32 || output.dtype() != source.dtype()
        || output.shape() != if update { source.shape() } else { indices.shape() } {
        return false;
    }
    let Some(precision) = source.representation().map(|r| r.dtype()) else { return false; };
    !update || operation.inputs.get(2).is_some_and(|updates|
        updates.shape() == indices.shape() && updates.dtype() == source.dtype()
            && updates.representation().is_some_and(|r| r.dtype() == precision))
}
