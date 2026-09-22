//! Windowed receipts use the same ordinary typed selection worker.
use super::*;
pub(in crate::capture::partition) fn source_slice(
    source: &AdmittedCapturePlan,
    context: &PartitionCaptureContext,
    shape: &[u64],
) -> Result<ResolvedCaptureSlice, CaptureError> {
    let index = context.selection_index;
    let selection = &source.plan().selections[index];
    if context.invocation_window.is_none() {
        return resolve_slice(&source.points()[index], selection, shape);
    }
    // Only the ordinary owning constructor uses this adapter. The paid
    // constructor supplies these same four destinations from its funding source.
    macro_rules! selected {
        ($source:expr) => {{
            let geometry = $source
                .map_err(|_| invalid("receipt interval differs from its original selection"))?;
            if !geometry
                .source_shape()
                .iter()
                .copied()
                .map(|n| n as u64)
                .eq(shape.iter().copied())
            {
                return Err(invalid(
                    "receipt physical axes differ from its original interval",
                ));
            }
            Ok(ResolvedCaptureSlice {
                starts: geometry.starts().to_vec(),
                ends: geometry.ends().to_vec(),
                strides: geometry.strides().to_vec(),
                shape: geometry
                    .starts()
                    .iter()
                    .zip(geometry.ends())
                    .zip(geometry.strides())
                    .map(|((&start, &end), &stride)| (end - start).div_ceil(stride))
                    .collect(),
            })
        }};
    }
    match selection.transform {
        CaptureTransform::Summary => {
            selected!(CaptureSummaryGeometry::prepare_receipt(source, context))
        }
        CaptureTransform::Histogram { .. } => {
            selected!(CaptureHistogramGeometry::prepare_receipt(source, context))
        }
        CaptureTransform::RoutedUnits => {
            selected!(CaptureRoutedUnitsGeometry::prepare_receipt(source, context))
        }
        _ => selected!(CaptureTensorGeometry::prepare_receipt(source, context)),
    }
}
