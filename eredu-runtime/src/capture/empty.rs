//! Host-only identity results for an admitted empty selection, after reservation.
use super::*;
use eredu_core::{checkpoint::TensorDtype, TensorObservation, TensorObservationData};

pub(super) fn empty_payload(
    transform: &CaptureTransform,
    shape: &[u64],
    precision: Option<&TensorDtype>,
) -> Result<CapturePayload, CaptureError> {
    if elements(shape)? != 0 {
        return Err(CaptureError::Invalid(
            "empty capture requires an empty selection".into(),
        ));
    }
    Ok(match transform {
        CaptureTransform::FullTensor
        | CaptureTransform::Slice
        | CaptureTransform::Preview { .. } => {
            let data = match precision {
                Some(TensorDtype::Bool) => TensorObservationData::Bool(Vec::new()),
                Some(TensorDtype::I8 | TensorDtype::I16 | TensorDtype::I32 | TensorDtype::I64) => {
                    TensorObservationData::I64(Vec::new())
                }
                Some(TensorDtype::U8 | TensorDtype::U16 | TensorDtype::U32 | TensorDtype::U64) => {
                    TensorObservationData::U64(Vec::new())
                }
                Some(
                    TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::F64,
                ) => TensorObservationData::F32(Vec::new()),
                _ => {
                    return Err(CaptureError::Unsupported(
                        "empty raw capture requires a supported actual source precision".into(),
                    ))
                }
            };
            let shape = if matches!(transform, CaptureTransform::Preview { .. }) {
                vec![0]
            } else {
                shape
                    .iter()
                    .map(|value| usize::try_from(*value).map_err(|_| CaptureError::Overflow))
                    .collect::<Result<Vec<_>, _>>()?
            };
            CapturePayload::Tensor(
                TensorObservation::new(shape, data)
                    .map_err(|error| CaptureError::Invalid(error.to_string()))?,
            )
        }
        CaptureTransform::Summary => CapturePayload::Summary(CaptureSummary {
            elements: 0,
            finite: 0,
            non_finite: 0,
            nan: 0,
            positive_infinity: 0,
            negative_infinity: 0,
            min: None,
            max: None,
            mean: None,
            rms: None,
        }),
        CaptureTransform::Histogram { edges } => CapturePayload::Histogram(CaptureHistogram {
            edges: edges.clone(),
            counts: vec![
                0;
                edges
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| CaptureError::Invalid(
                        "histogram edges are absent".into()
                    ))?
            ],
            below: 0,
            above: 0,
            non_finite: 0,
        }),
        _ => {
            return Err(CaptureError::Unsupported(
                "empty capture requires a raw or reduction transform".into(),
            ))
        }
    })
}
