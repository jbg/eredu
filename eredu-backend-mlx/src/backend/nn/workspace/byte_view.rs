//! Closed byte reinterpretation over exact scalar evidence, not numeric casts.
use super::*;
use eredu_nn::workspace::{WorkspaceFloatingType as F, WorkspaceLayoutView};
use eredu_nn::Tensor;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Dtype {
    U8,
    I32,
    U32,
    F16,
    Bf16,
    F32,
}
impl Dtype {
    pub(crate) fn native(self) -> safemlx::Dtype {
        match self {
            Self::U8 => safemlx::Dtype::Uint8,
            Self::I32 => safemlx::Dtype::Int32,
            Self::U32 => safemlx::Dtype::Uint32,
            Self::F16 => safemlx::Dtype::Float16,
            Self::Bf16 => safemlx::Dtype::Bfloat16,
            Self::F32 => safemlx::Dtype::Float32,
        }
    }
    pub(crate) fn from_native(value: safemlx::Dtype) -> Option<Self> {
        Some(match value {
            safemlx::Dtype::Uint8 => Self::U8,
            safemlx::Dtype::Int32 => Self::I32,
            safemlx::Dtype::Uint32 => Self::U32,
            safemlx::Dtype::Float16 => Self::F16,
            safemlx::Dtype::Bfloat16 => Self::Bf16,
            safemlx::Dtype::Float32 => Self::F32,
            _ => return None,
        })
    }
    pub(crate) fn bytes(self) -> u64 {
        match self {
            Self::U8 => 1,
            Self::F16 | Self::Bf16 => 2,
            _ => 4,
        }
    }
    pub(crate) fn logical(self) -> WorkspaceDtype {
        match self {
            Self::U8 => WorkspaceDtype::Uint8,
            Self::I32 => WorkspaceDtype::Int32,
            Self::U32 => WorkspaceDtype::Uint32,
            _ => WorkspaceDtype::Float32,
        }
    }
    pub(crate) fn floating(self) -> Option<F> {
        match self {
            Self::F16 => Some(F::Float16),
            Self::Bf16 => Some(F::Bfloat16),
            Self::F32 => Some(F::Float32),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::U8 => "bitcast_u8",
            Self::I32 => "bitcast_i32",
            Self::U32 => "bitcast_u32",
            Self::F16 => "bitcast_f16",
            Self::Bf16 => "bitcast_bf16",
            Self::F32 => "bitcast_f32",
        }
    }
    pub(crate) fn from_layout(value: WorkspaceLayoutView<'_>) -> Option<Self> {
        Some(match value.dtype() {
            WorkspaceDtype::Uint8 => Self::U8,
            WorkspaceDtype::Int32 => Self::I32,
            WorkspaceDtype::Uint32 => Self::U32,
            WorkspaceDtype::Float32 => match value.representation()?.dtype() {
                F::Float16 => Self::F16,
                F::Bfloat16 => Self::Bf16,
                F::Float32 => Self::F32,
            },
            _ => return None,
        })
    }
}
pub(crate) fn selected(name: &str) -> Option<Dtype> {
    Some(match name {
        "bitcast_u8" => Dtype::U8,
        "bitcast_i32" => Dtype::I32,
        "bitcast_u32" => Dtype::U32,
        "bitcast_f16" => Dtype::F16,
        "bitcast_bf16" => Dtype::Bf16,
        "bitcast_f32" => Dtype::F32,
        _ => return None,
    })
}
/// Require the same rank/prefix and exact last-axis byte ratio used by ops::view.
/// Floating logical geometry alone never supplies a physical byte width.
pub(crate) fn inspect(operation: WorkspaceOperationView<'_>) -> Option<(u64, usize)> {
    inspect_geometry(operation, true)
}
/// The output precision is being published by the mechanism on this pass.
/// Its shape and logical dtype still follow the same exact byte relation;
/// any already-present precision must agree, and source precision is required.
pub(crate) fn output_geometry(operation: WorkspaceOperationView<'_>) -> Option<(u64, usize)> {
    inspect_geometry(operation, false)
}
fn inspect_geometry(
    operation: WorkspaceOperationView<'_>,
    require_precision: bool,
) -> Option<(u64, usize)> {
    let WorkspaceOperationKindView::View(name) = operation.kind else {
        return None;
    };
    let target = selected(name)?;
    if operation.inputs.len() != 1 || operation.outputs.len() != 1 {
        return None;
    }
    let input = operation.inputs.get(0)?;
    let output = operation.outputs.get(0)?;
    let source = Dtype::from_layout(input)?;
    let precision = output.representation().map(|p| p.dtype());
    if output.dtype() != target.logical()
        || ((require_precision || precision.is_some()) && precision != target.floating())
        || input.shape().len() != output.shape().len()
    {
        return None;
    }
    let rank = input.shape().len();
    if rank == 0 {
        if source.bytes() != target.bytes() {
            return None;
        }
    } else {
        if input.shape()[..rank - 1] != output.shape()[..rank - 1] {
            return None;
        }
        let last = u64::try_from(input.shape()[rank - 1])
            .ok()?
            .checked_mul(source.bytes())?;
        if last % target.bytes() != 0
            || u64::try_from(output.shape()[rank - 1]).ok()? != last / target.bytes()
        {
            return None;
        }
    }
    let bytes = input.elements().ok()?.checked_mul(source.bytes())?;
    if output.elements().ok()?.checked_mul(target.bytes())? != bytes {
        return None;
    }
    Some((bytes, rank))
}
pub(crate) fn trace(
    value: &WorkspaceTensor,
    target: Dtype,
    context: &WorkspaceContext,
) -> Result<WorkspaceTensor, eredu_nn::Error> {
    let source = Dtype::from_layout(value.layout().as_view()).ok_or_else(|| {
        context.metadata_error(format_args!(
            "byte view lacks its actual source scalar representation"
        ))
    })?;
    let mut shape = context.metadata_vec(value.shape().len())?;
    shape.extend_from_slice(value.shape());
    match shape.last_mut() {
        Some(last) => {
            let bytes = u64::try_from(*last)
                .ok()
                .and_then(|n| n.checked_mul(source.bytes()))
                .ok_or_else(|| {
                    context.metadata_error(format_args!("byte-view dimension overflow"))
                })?;
            if bytes % target.bytes() != 0 {
                return Err(
                    context.metadata_error(format_args!("byte-view dimension is not divisible"))
                );
            }
            *last = i32::try_from(bytes / target.bytes()).map_err(|_| {
                context.metadata_error(format_args!("byte-view dimension overflow"))
            })?;
        }
        None if source.bytes() != target.bytes() => {
            return Err(context.metadata_error(format_args!("byte view changes scalar width")))
        }
        None => {}
    }
    let mut outputs = context.metadata_vec(1)?;
    outputs.push(context.layout(&shape, target.logical())?);
    let mut values = context.execute(
        WorkspaceOperationKind::View(target.name()),
        &[value],
        outputs,
    )?;
    Ok(values.pop().expect("one declared byte view"))
}
