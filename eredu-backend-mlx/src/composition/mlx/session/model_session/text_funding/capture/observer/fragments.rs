//! Physical retained factory validation for an already claimed original fragment.
use super::*;
use eredu_nn::{
    BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole,
};
use eredu_runtime::{capture::CaptureProtocolError, working_memory::CapturePrefillFragmentClaim};
fn backend<E: std::error::Error + Send + Sync + 'static>(e: E) -> FundedCaptureError<Error> {
    FundedCaptureError::Backend(Error::Other(Box::new(e)))
}
impl NativeScheduledCapture<'_> {
    pub(super) fn prepare_generated_fragment(
        &mut self,
        prototype: &Array,
        source: &eredu_core::capture::GeneratedCaptureSource,
        program: GeneratedTensorProgram<'_>,
        construct: bool,
        claim: &CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        self.work
            .prepare_generated_capture(claim, construct)
            .map_err(FundedCaptureError::Backend)?;
        PreparedCaptureTensor::validate_stream(self.stream).map_err(backend)?;
        self.validate_prefill_source(prototype, claim.fragment())?;
        let GeneratedTensorProgram::BlockFp8Input(plan) = program;
        if plan.shape() != prototype.shape()
            || source.source_dtype != Some(TensorDtype::F32)
            || plan
                .logical_capture_source()
                .map_err(backend)?
                .creation_bytes
                != source.creation_bytes
        {
            return Err(CaptureProtocolError::GeneratedSource.into());
        }
        Ok(())
    }
    pub(super) fn retain_generated_fragment_operand(
        &mut self,
        prototype: &Array,
        role: GeneratedTensorSourceRole,
        value: &Array,
        claim: &CapturePrefillFragmentClaim<'_, '_, '_, '_>,
    ) -> Result<(), FundedCaptureError<Error>> {
        let plan = BlockFp8InputReconstructionPlan::new(prototype.shape()).map_err(backend)?;
        let (shape, dtype) = match role {
            GeneratedTensorSourceRole::CompactValues => {
                (plan.values_shape(), safemlx::Dtype::Uint8)
            }
            GeneratedTensorSourceRole::BlockScales => {
                (plan.scales_shape(), safemlx::Dtype::Float32)
            }
        };
        if value.shape() != shape || value.dtype() != dtype {
            return Err(backend(eredu_nn::ProjectionObservationError::Geometry));
        }
        self.work
            .retain_generated_capture_root(value, claim)
            .map_err(FundedCaptureError::Backend)
    }
}
