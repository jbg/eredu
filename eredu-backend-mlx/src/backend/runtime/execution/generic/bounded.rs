//! Native storage, transfer ordering and consumer completion for bounded units.

use super::*;
use eredu_core::residency::NativeBoundedUnits;
use eredu_core::residency::ResidencyTransferOrdering;
use eredu_core::{Completion, SubmissionBackend};

impl NativeBoundedUnits for MlxUnitOperations {
    type Tensor = MlxTensor;
    type Context = Stream;
    type Parameters = MlxNeuralBackend;
    type Lease<U: 'static> = MlxUnitLease<U>;
    type Transfer = ResidentTransfer;

    fn reclaim(&mut self, inspection: bool) {
        crate::backend::submission_recovery::reap();
        if inspection {
            crate::backend::ordinary_retirement::reclaim_all();
        } else {
            crate::backend::ordinary_retirement::reclaim();
        }
    }

    fn reclaim_retired(&mut self) {
        crate::backend::ordinary_retirement::reclaim_all();
    }

    fn token_width(&self, input: &MlxTensor) -> usize {
        input.as_array().dim(1) as usize
    }

    fn is_complete<U: 'static>(&self, lease: &Self::Lease<U>) -> Result<bool, Error> {
        lease.is_complete()
    }

    fn retire<U: 'static>(&mut self, lease: Self::Lease<U>) -> Result<(), Error> {
        lease.finish()
    }

    fn bind_lease<U: 'static>(
        &mut self,
        unit: U,
        transfer: ResidentTransfer,
        context: &Stream,
    ) -> Result<Self::Lease<U>, Error> {
        transfer.order_after(context)?;
        MlxUnitLease::new(MlxModule::new(unit), transfer)
    }

    fn unit_weights<U: 'static>(
        lease: &mut Self::Lease<U>,
    ) -> Result<Vec<(String, MlxTensor)>, Error> {
        resident_weights(lease.population_parts().1)
    }

    fn submit<'a, U: 'static>(
        &mut self,
        lease: &mut Self::Lease<U>,
        roots: impl Iterator<Item = &'a MlxTensor>,
    ) -> Result<(), Error> {
        lease.submitted(async_eval_with_event(roots.map(MlxTensor::as_array))?)
    }

    fn submit_inspection<U: 'static>(
        &mut self,
        lease: &mut Self::Lease<U>,
        context: &Stream,
    ) -> Result<(), Error> {
        let marker = safemlx::ops::zeros_dtype(&[1], safemlx::Dtype::Float32, context)?;
        lease.submitted(async_eval_with_event([&marker])?)
    }

    fn finish_output(&mut self, output: &MlxTensor, context: &Stream) -> Result<(), Error> {
        Ok(MlxNeuralBackend::submit(context, [output])?.wait()?)
    }
}
