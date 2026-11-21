//! Transparent MLX tensor handle conversions for activation observation.

use crate::{backend::error::Error, MlxTensor};
use eredu_core::observation::ObserverTensorMapping;
use ref_cast::RefCast;
use safemlx::Array;

/// Converts native array views to the tensor representation used by observers.
pub struct ArrayToTensor;

impl ObserverTensorMapping for ArrayToTensor {
    type InvocationTensor = Array;
    type ObserverTensor = MlxTensor;
    type InvocationError = Error;
    type ObserverError = Error;

    fn borrow_tensor(value: &Array) -> &MlxTensor {
        MlxTensor::ref_cast(value)
    }

    fn borrow_invocation_tensor(value: &MlxTensor) -> &Array {
        value.as_array()
    }

    fn generated_tensor(value: Array) -> MlxTensor {
        MlxTensor::from_array(value)
    }

    fn replacement_tensor(value: MlxTensor) -> Array {
        value.into_array()
    }

    fn invocation_error(error: Error) -> Error {
        error
    }

    fn observer_error(error: Error) -> Error {
        error
    }
}

/// Maps neural operator observations to the submission tensor representation.
pub struct TensorToArray;

impl ObserverTensorMapping for TensorToArray {
    type InvocationTensor = MlxTensor;
    type ObserverTensor = Array;
    type InvocationError = eredu_nn::Error;
    type ObserverError = Error;

    fn borrow_tensor(value: &MlxTensor) -> &Array {
        value.as_array()
    }

    fn borrow_invocation_tensor(value: &Array) -> &MlxTensor {
        MlxTensor::ref_cast(value)
    }

    fn generated_tensor(value: MlxTensor) -> Array {
        value.into_array()
    }

    fn replacement_tensor(value: Array) -> MlxTensor {
        MlxTensor::from_array(value)
    }

    fn invocation_error(error: Error) -> eredu_nn::Error {
        eredu_nn::Error::backend_source(error)
    }

    fn observer_error(error: eredu_nn::Error) -> Error {
        error.into()
    }
}
