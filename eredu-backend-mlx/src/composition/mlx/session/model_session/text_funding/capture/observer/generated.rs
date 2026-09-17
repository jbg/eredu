//! Retention into the same original work; no new scope/admission/completion.
use super::*;
use eredu_core::capture::GeneratedCaptureSource;
use eredu_nn::{
    BlockFp8InputReconstructionPlan, GeneratedTensorProgram, GeneratedTensorSourceRole,
};
use eredu_runtime::capture::CaptureObservationStep;
use safemlx::Dtype;

type Result<T> = std::result::Result<T, FundedCaptureError<Error>>;
fn backend<E: std::error::Error + Send + Sync + 'static>(error: E) -> FundedCaptureError<Error> {
    FundedCaptureError::Backend(Error::Other(Box::new(error)))
}
impl NativeScheduledCapture<'_> {
    fn validate_generated_claim(&self, claim: &CaptureTensorClaim<'_, '_>) -> Result<()> {
        let scope = self.work.scope.try_borrow().map_err(backend)?;
        let scope = scope
            .as_ref()
            .ok_or_else(|| backend(WorkingMemoryError::ExecutionFenced))?;
        claim.validate_native_scope(scope).map_err(backend)
    }
    pub(super) fn prepare_generated(
        &mut self,
        prototype: &Array,
        source: &GeneratedCaptureSource,
        program: GeneratedTensorProgram<'_>,
        construct: bool,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<()> {
        self.validate_generated_claim(claim)?;
        PreparedCaptureTensor::validate_stream(self.stream).map_err(backend)?;
        PreparedCaptureTensor::validate_borrowed_source(prototype, claim.geometry())
            .map_err(backend)?;
        let geometry = claim.geometry();
        CaptureObservationStep::new(
            geometry.admission(),
            geometry.phase(),
            geometry.prediction(),
        )?
        .validate_generated(geometry.selection_index(), source, program)?;
        let GeneratedTensorProgram::BlockFp8Input(plan) = program;
        if plan.shape() != prototype.shape() {
            return Err(backend(eredu_nn::ProjectionObservationError::Geometry));
        }
        if construct {
            // This closed program has two actual inputs and seven outputs. Only
            // descriptor slots are reserved; no numerical capacity is minted.
            self.work
                .roots
                .try_borrow_mut()
                .map_err(backend)?
                .try_reserve_exact(9)
                .map_err(backend)?;
        }
        // Every guard above ends before the caller invokes the actual producer.
        Ok(())
    }
    pub(super) fn retain_generated_operand(
        &mut self,
        prototype: &Array,
        role: GeneratedTensorSourceRole,
        value: &Array,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<()> {
        let plan = BlockFp8InputReconstructionPlan::new(prototype.shape()).map_err(backend)?;
        let (shape, dtype) = match role {
            GeneratedTensorSourceRole::CompactValues => (plan.values_shape(), Dtype::Uint8),
            GeneratedTensorSourceRole::BlockScales => (plan.scales_shape(), Dtype::Float32),
        };
        if value.shape() != shape || value.dtype() != dtype {
            return Err(backend(eredu_nn::ProjectionObservationError::Geometry));
        }
        self.retain_generated_root(value, claim)
    }
    pub(super) fn retain_generated_root(
        &mut self,
        value: &Array,
        claim: &CaptureTensorClaim<'_, '_>,
    ) -> Result<()> {
        // Retain before the subsequent health check can reject. All successful
        // prior outputs remain in the original operation's recovery collector.
        let retained = value.clone();
        {
            let mut roots = self.work.roots.try_borrow_mut().map_err(backend)?;
            if roots.len() == roots.capacity() {
                return Err(backend(WorkingMemoryError::UnknownBound));
            }
            roots.push(retained);
        }
        self.work.published.set(false);
        self.validate_generated_claim(claim)
    }
}
