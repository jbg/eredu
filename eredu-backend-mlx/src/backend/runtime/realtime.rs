//! Native host readback of realtime tensors.

use eredu_core::{realtime_mechanisms::RealtimeTensorReadback, BackendFailure};
use eredu_nn::Tensor;
use safemlx::{Array, Dtype, Stream};

use crate::{backend::error::Error, MlxTensor};

fn array_i32_host(array: &Array) -> Result<Vec<i32>, Error> {
    if array.size() == 0 {
        return Ok(Vec::new());
    }
    let evaluated = array.evaluated()?;
    match array.dtype() {
        Dtype::Int32 => Ok(evaluated.as_slice::<i32>().to_vec()),
        Dtype::Uint32 => evaluated
            .as_slice::<u32>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Int64 => evaluated
            .as_slice::<i64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        Dtype::Uint64 => evaluated
            .as_slice::<u64>()
            .iter()
            .map(|value| i32::try_from(*value).map_err(|error| Error::Parallel(error.to_string())))
            .collect(),
        dtype => Err(Error::Parallel(format!(
            "realtime token observation expected integer values, got {dtype:?}"
        ))),
    }
}

/// Native tensor readback on the selected execution stream.
pub struct MlxRealtimeReadback {
    stream: Stream,
}

impl MlxRealtimeReadback {
    /// Retains the execution stream for host readback.
    pub fn new(stream: &Stream) -> Self {
        Self {
            stream: stream.clone(),
        }
    }
}

impl RealtimeTensorReadback for MlxRealtimeReadback {
    type Tensor = MlxTensor;

    fn read_i32(&self, value: &MlxTensor) -> Result<Vec<i32>, BackendFailure> {
        let array = value
            .as_array()
            .contiguous(false, &self.stream)
            .map_err(BackendFailure::from_error)?;
        array_i32_host(&array).map_err(BackendFailure::from_error)
    }

    fn read_f32(&self, value: &MlxTensor) -> Result<Vec<f32>, BackendFailure> {
        value
            .to_f32_vec(&self.stream)
            .map_err(BackendFailure::from_error)
    }
}
