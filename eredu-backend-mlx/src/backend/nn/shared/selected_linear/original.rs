//! Closed error transport for the admitted grouped FP8 call prefix.
use super::*;
use safemlx::{OriginalScopeObserver, error::Exception};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure {
    #[source]
    cause: Exception,
    _custody: Exception,
}
#[derive(Debug)]
pub(in crate::backend::nn::shared) struct Transport {
    fp8: bool,
}
impl Transport {
    pub(in crate::backend::nn::shared) fn new(fp8: bool) -> Self {
        Self { fp8 }
    }
    fn carrier(&self) -> Option<Exception> {
        if !self.fp8 {
            return None;
        }
        match OriginalScopeObserver::try_current() {
            Ok(Some(scope)) => Some(scope.invalid_input_error()),
            Ok(None) => None,
            Err(cause) => Some(cause),
        }
    }
    pub(in crate::backend::nn::shared) fn compute<T>(
        &self,
        value: Result<T, Exception>,
    ) -> Result<T, ComputeError> {
        value.map_err(|cause| match self.carrier() {
            Some(custody) => ComputeError::backend_retained_source(Failure {
                cause,
                _custody: custody,
            }),
            None => ComputeError::backend_retained_source(cause),
        })
    }
    pub(in crate::backend::nn::shared) fn tensor(
        &self,
        value: Result<Array, Exception>,
    ) -> Result<MlxTensor, ComputeError> {
        self.compute(value).map(MlxTensor::from_array)
    }
    pub(in crate::backend::nn::shared) fn invalid(&self, message: &'static str) -> ComputeError {
        match self.carrier() {
            Some(cause) => ComputeError::backend_retained_source(cause),
            None => ComputeError::backend(message),
        }
    }
}
pub(crate) fn control_bytes() -> Option<usize> {
    let controls = [
        ComputeError::retained_source_construction_bytes::<Failure>()?,
        ComputeError::retained_source_construction_bytes::<Exception>()?,
        OriginalScopeObserver::control_bytes()?,
        size_of::<Transport>(),
        size_of::<Option<Exception>>(),
        size_of::<Result<Option<OriginalScopeObserver>, Exception>>(),
        size_of::<common::grouping::GroupedSelectionPlan>(),
        size_of::<Result<common::grouping::GroupedSelectionPlan, Exception>>(),
        size_of::<Result<common::grouping::GroupedSelectionPlan, ComputeError>>(),
        size_of::<Result<Option<Array>, Exception>>(),
        size_of::<Result<Option<Array>, ComputeError>>(),
        size_of::<Result<Array, Exception>>(),
        size_of::<Result<Array, ComputeError>>(),
        size_of::<Result<MlxTensor, ComputeError>>(),
        size_of::<Exception>(),
        5 * size_of::<Array>(),
        size_of::<Option<&Array>>(),
        size_of::<&GroupSelection<MlxTensor>>(),
    ];
    controls
        .into_iter()
        .try_fold(size_of_val(&controls), usize::checked_add)
}
