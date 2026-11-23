//! Eager native reads of exact routed coordinates and selected units.
use super::*;
use eredu_core::capture::RoutedReadout;

pub(crate) struct NativeRoutedReadout<'a>(pub &'a CandleContext);

impl RoutedReadout<CandleTensor> for NativeRoutedReadout<'_> {
    type Error = BackendFailure;

    fn ids(
        &self,
        tensor: &CandleTensor,
        indices: Option<&[u64]>,
    ) -> Result<Vec<u32>, BackendFailure> {
        let flat = tensor
            .value
            .flatten_all()
            .map_err(BackendFailure::from_error)?;
        let tensor = if let Some(indices) = indices {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            let flat = if flat.device().is_metal() && flat.dtype() == DType::I32 {
                crate::metal_f32::widen_i32(&flat).map_err(BackendFailure::from_error)?
            } else {
                flat
            };
            flat.index_select(&native_indices(indices, self.0)?, 0)
                .map_err(BackendFailure::from_error)?
        } else {
            flat
        };
        match tensor.dtype() {
            DType::U32 => tensor.to_vec1::<u32>().map_err(BackendFailure::from_error),
            DType::I32 => tensor
                .to_vec1::<i32>()
                .map_err(BackendFailure::from_error)?
                .into_iter()
                .map(|id| u32::try_from(id).map_err(BackendFailure::from_error))
                .collect(),
            DType::I64 => tensor
                .to_vec1::<i64>()
                .map_err(BackendFailure::from_error)?
                .into_iter()
                .map(|id| u32::try_from(id).map_err(BackendFailure::from_error))
                .collect(),
            _ => Err(BackendFailure::from_error(CaptureError::Invalid(
                "wrong routing index dtype".into(),
            ))),
        }
    }

    fn coefficients(
        &self,
        tensor: &CandleTensor,
        indices: &[u64],
    ) -> Result<Vec<f32>, BackendFailure> {
        select_elements(tensor, indices, self.0)?
            .to_dtype(DType::F32)
            .and_then(|values| values.to_vec1::<f32>())
            .map_err(BackendFailure::from_error)
    }

    fn units(
        &self,
        tensor: &CandleTensor,
        row: usize,
        columns: &[u64],
    ) -> Result<TensorObservation, BackendFailure> {
        if columns.is_empty() {
            return TensorObservation::new(vec![0], TensorObservationData::F32(vec![]))
                .map_err(BackendFailure::from_error);
        }
        let indices = native_indices(columns, self.0)?;
        let values = tensor
            .value
            .get(row)
            .and_then(|row| row.index_select(&indices, 0))
            .and_then(|row| row.to_dtype(DType::F32))
            .and_then(|row| row.to_vec1::<f32>())
            .map_err(BackendFailure::from_error)?;
        TensorObservation::new(vec![columns.len()], TensorObservationData::F32(values))
            .map_err(BackendFailure::from_error)
    }
}

pub(crate) fn native_indices(
    indices: &[u64],
    context: &CandleContext,
) -> Result<NativeTensor, BackendFailure> {
    let values = indices
        .iter()
        .map(|&index| u32::try_from(index).map_err(BackendFailure::from_error))
        .collect::<Result<Vec<_>, _>>()?;
    NativeTensor::new(values, &context.device).map_err(BackendFailure::from_error)
}

pub(crate) fn select_elements(
    tensor: &CandleTensor,
    indices: &[u64],
    context: &CandleContext,
) -> Result<NativeTensor, BackendFailure> {
    let indices = native_indices(indices, context)?;
    tensor
        .value
        .flatten_all()
        .and_then(|flat| flat.index_select(&indices, 0))
        .map_err(BackendFailure::from_error)
}
