//! Selected input provenance and the fixed compact E4M3 decode primitive.
use super::facts::{self, buffer_capacity, Emitter, FactResult, Output};
use super::*;
use eredu_nn::{LinearFormatSpec, LinearRowLayout, ProjectionInputObservationMechanism};

pub(super) fn selection(
    format: &LinearFormatSpec,
) -> Result<Option<ProjectionInputObservationMechanism>, Error> {
    format.validate()?;
    Ok(match format.encoding() {
        eredu_checkpoint::LinearFormat::E4M3BlockFp8(config) => {
            #[cfg(feature = "cuda")]
            {
                let _ = config;
                None
            }
            #[cfg(not(feature = "cuda"))]
            {
                (config.block_rows == 128
                    && config.block_columns == 128
                    && format.row_layout() == LinearRowLayout::Contiguous)
                    .then_some(ProjectionInputObservationMechanism::BlockFp8Gpu)
            }
        }
        _ => Some(ProjectionInputObservationMechanism::Borrowed),
    })
}

pub(super) fn operation_bound(
    op: &WorkspaceOperation,
    allocation: MetalAllocationFacts,
) -> Result<Option<WorkspaceOperationBound>, Error> {
    facts::ordinary(|sink| emit(op.as_view(), allocation, sink))
}

pub(super) fn emit(
    op: WorkspaceOperationView<'_>,
    allocation: MetalAllocationFacts,
    sink: &mut Emitter<'_>,
) -> FactResult<Option<WorkspaceOperationFacts>> {
    if !matches!(
        op.kind,
        WorkspaceOperationKindView::BlockFp8ActivationDecode
    ) {
        return Ok(None);
    }
    let (Some([input]), Some([output])) = (op.inputs.array(), op.outputs.array()) else {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid compact FP8 decode arity",
        ));
    };
    if input.dtype() != WorkspaceDtype::Uint8
        || output.dtype() != WorkspaceDtype::Float32
        || input.shape().len() != 2
        || input.shape() != output.shape()
        || input.shape().iter().any(|&n| n <= 0)
    {
        return Err(MlxWorkspaceFactError::descriptor(
            "invalid compact FP8 decode geometry",
        ));
    }
    i32::try_from(input.elements()?)?;
    sink.output(Output::Allocate(buffer_capacity(
        allocation,
        output.bytes()?,
    )?))?;
    sink.finish(0, format_args!("fixed quantizer-produced compact Uint8 values: MLX from_fp8 constructs ConvertFP8; Metal unary conversion owns one F32 output and reads the compact source directly; source roots and repeated scales are separately retained by the fixed reconstruction program")).map(Some)
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests;
