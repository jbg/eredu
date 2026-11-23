//! Native reads of exact routed coordinates and selected units.
use super::*;

pub(in crate::composition::mlx) struct NativeRoutedReadout<'a>(pub &'a Stream);

impl From<CaptureError> for Error {
    fn from(error: CaptureError) -> Self {
        Self::observation(error)
    }
}

impl RoutedReadout<MlxTensor> for NativeRoutedReadout<'_> {
    type Error = Error;

    fn ids(&self, tensor: &MlxTensor, indices: Option<&[u64]>) -> Result<Vec<u32>, Error> {
        let array = if let Some(indices) = indices {
            tensor
                .as_array()
                .reshape(&[-1], self.0)?
                .take(&native_indices(indices)?, self.0)?
        } else {
            tensor.as_array().clone()
        };
        let array = array.as_dtype(Dtype::Uint32, self.0)?;
        Ok(array.evaluated()?.as_slice::<u32>().to_vec())
    }

    fn coefficients(&self, tensor: &MlxTensor, indices: &[u64]) -> Result<Vec<f32>, Error> {
        let array = tensor
            .as_array()
            .reshape(&[-1], self.0)?
            .take(&native_indices(indices)?, self.0)?
            .as_dtype(Dtype::Float32, self.0)?;
        Ok(array.evaluated()?.as_slice::<f32>().to_vec())
    }

    fn units(
        &self,
        tensor: &MlxTensor,
        row: usize,
        columns: &[u64],
    ) -> Result<TensorObservation, Error> {
        if columns.is_empty() {
            return TensorObservation::new(vec![0], eredu_core::TensorObservationData::F32(vec![]))
                .map_err(Error::observation);
        }
        let row = i32::try_from(row).map_err(|_| CaptureError::Overflow)?;
        let values = tensor
            .as_array()
            .try_index_device((row, ..), self.0)?
            .take(&native_indices(columns)?, self.0)?
            .as_dtype(Dtype::Float32, self.0)?
            .contiguous(false, self.0)?;
        super::super::observation::observe_tensor(&MlxTensor::from_array(values), self.0)
    }
}

pub(in crate::composition::mlx) fn native_indices(values: &[u64]) -> Result<Array, Error> {
    let values = values
        .iter()
        .map(|&index| i32::try_from(index).map_err(|_| CaptureError::Overflow))
        .collect::<Result<Vec<_>, _>>()?;
    let length = i32::try_from(values.len()).map_err(|_| CaptureError::Overflow)?;
    Ok(Array::from_slice(&values, &[length]))
}
