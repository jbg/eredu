//! MLX realization of the backend-neutral layerwise unit policy.

use eredu_core::residency::{NativeResidentUnits, ResidencyTransferOrdering, UnitRetirement};

use safemlx::{transforms::async_eval_with_event, Event, Stream};

use crate::backend::{
    error::Error,
    nn::shared::{MlxModule, MlxNeuralBackend},
    ordinary_retirement::OrdinaryRetirement,
    runtime::residency::manager::{ResidentTransfer, ResidentUnitLease},
};
use crate::MlxTensor;

mod submission;
pub use submission::MlxUnitLease;

/// Native retention and completion for a leased prediction module's tensor roots.
pub struct MlxPredictionTransfer {
    loan: MlxUnitLease<Vec<MlxTensor>>,
    stream: Stream,
}

impl MlxPredictionTransfer {
    pub(crate) fn begin(transfer: ResidentTransfer, stream: &Stream) -> Result<Self, Error> {
        transfer.order_after(stream)?;
        Ok(Self {
            loan: MlxUnitLease::new(MlxModule::new(Vec::new()), transfer)?,
            stream: stream.clone(),
        })
    }

    pub(crate) fn weights(&mut self) -> Result<Vec<(String, MlxTensor)>, Error> {
        resident_weights(self.loan.population_parts().1)
    }

    pub(crate) fn finish(mut self, roots: Vec<MlxTensor>) -> Result<(), Error> {
        self.loan.population_parts().0.inner = roots;
        let validations = crate::backend::nn::tensor::active_token_validation_arrays();
        let marker = safemlx::ops::zeros_dtype(&[1], safemlx::Dtype::Float32, &self.stream)?;
        let event = async_eval_with_event(
            self.loan
                .iter()
                .map(MlxTensor::as_array)
                .chain(validations.iter())
                .chain([&marker]),
        )?;
        self.loan.submitted(event)?;
        self.loan.finish()?;
        crate::backend::ordinary_retirement::reclaim_all();
        crate::backend::nn::tensor::validate_active_token_validations()?;
        Ok(())
    }
}

/// Native transfer, submission and retirement for a selected parameter unit.
#[doc(hidden)]
pub struct MlxUnitOperations;

impl UnitRetirement for MlxUnitOperations {
    type Error = Error;
    type RetainedResource<T: 'static> = OrdinaryRetirement<T>;

    fn retain_resource<T: 'static>(resource: T) -> Self::RetainedResource<T> {
        OrdinaryRetirement::new(resource)
    }
}

impl NativeResidentUnits<MlxNeuralBackend, ResidentTransfer> for MlxUnitOperations {
    type Module<U: 'static> = MlxModule<U>;
    type Parameters = MlxNeuralBackend;

    fn order_resident_transfer(
        transfer: &ResidentTransfer,
        stream: &Stream,
    ) -> Result<usize, Error> {
        transfer.order_after(stream)?;
        Ok(transfer.leases().len())
    }

    fn resident_weights(
        transfer: &ResidentTransfer,
        ordinal: usize,
    ) -> Result<Vec<(String, MlxTensor)>, Error> {
        resident_weights(&transfer.leases()[ordinal])
    }

    fn retain_unit<U: 'static>(unit: U) -> Self::Module<U> {
        MlxModule::new(unit)
    }
}

fn resident_weights(lease: &ResidentUnitLease) -> Result<Vec<(String, MlxTensor)>, Error> {
    lease
        .binding_names()
        .map(|name| {
            let array = lease
                .device_value(name)
                .map_err(eredu_core::BackendFailure::from_error)?;
            Ok((name.to_owned(), MlxTensor::from_array(array.clone())))
        })
        .collect()
}

/// Cold preparation for bounded and resident policies.
mod preparation;
pub(crate) use preparation::parameter_residency_streams;

mod bounded;

mod selected;
pub(crate) use selected::MlxImmutableParameters;
