//! Complete selected floating terms, assembled before nonlinear observation transforms.
use super::*;
use eredu_core::{
    checkpoint::TensorDtype, ObservationDtype, ObservationPoint, ObservationValueType,
    TensorObservation, TensorObservationData,
};

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

pub(super) fn native_transform(transform: &CaptureTransform) -> CaptureTransform {
    match transform {
        CaptureTransform::Preview { max_elements } => CaptureTransform::Preview {
            max_elements: *max_elements,
        },
        _ => CaptureTransform::Slice,
    }
}

pub(super) fn native_selection(selection: &CaptureSelection) -> CaptureSelection {
    CaptureSelection {
        id: selection.id.clone(),
        path: selection.path.clone(),
        schedule: selection.schedule.clone(),
        slices: selection.slices.clone(),
        transform: native_transform(&selection.transform),
    }
}

pub(super) fn reserve_native_selection<'a>(
    plan: &'a AdmittedCapturePlan,
    index: usize,
    combination: PartitionCaptureCombination,
    world: usize,
    ledger: &mut dyn CaptureReservation,
) -> Result<std::borrow::Cow<'a, CaptureSelection>, CaptureError> {
    let selection = &plan.plan().selections[index];
    if combination == PartitionCaptureCombination::Disjoint {
        return Ok(std::borrow::Cow::Borrowed(selection));
    }
    let usage = CaptureUsage {
        host_bytes: mul(
            metadata_reservation(selection, &plan.points()[index])?.host_bytes,
            world as u64,
        )?,
        ..Default::default()
    };
    ledger.reserve_quota(usage)?;
    Ok(std::borrow::Cow::Owned(native_selection(selection)))
}

pub(super) fn validate_precision(dtype: Option<&TensorDtype>) -> Result<(), CaptureError> {
    if !matches!(
        dtype,
        None | Some(TensorDtype::F16 | TensorDtype::Bf16 | TensorDtype::F32 | TensorDtype::F64)
    ) {
        return Err(invalid(
            "summed activation has a non-floating native source",
        ));
    }
    Ok(())
}

pub(super) fn validate_projection(
    selection: &CaptureSelection,
    point: &ObservationPoint,
    projection: &CaptureSlicePartition,
) -> Result<(), CaptureError> {
    if point.value_type != ObservationValueType::Tensor
        || point.dtype != ObservationDtype::Floating
        || !matches!(
            selection.transform,
            CaptureTransform::FullTensor
                | CaptureTransform::Slice
                | CaptureTransform::Preview { .. }
                | CaptureTransform::Summary
                | CaptureTransform::Histogram { .. }
        )
    {
        return Err(CaptureError::Unsupported("summed activation receipts require floating tensor observations, not routed or vocabulary records".into()));
    }
    let selected = projection.global_slice();
    let empty = elements(&selected.shape)? == 0;
    if projection.local_shape() != projection.global_shape()
        || projection.fragments().len() != usize::from(!empty)
        || projection.fragments().iter().any(|fragment| {
            let local = fragment.local();
            let destination = fragment.destination();
            local.starts != selected.starts
                || local.strides != selected.strides
                || local.shape != selected.shape
                || destination.shape != selected.shape
                || destination.ends != selected.shape
                || destination.starts.iter().any(|start| *start != 0)
                || destination.strides.iter().any(|stride| *stride != 1)
        })
    {
        return Err(invalid(
            "each summed activation producer must supply the complete ordered selection",
        ));
    }
    Ok(())
}

pub(super) fn payload_usage(
    transform: &CaptureTransform,
    count: u64,
) -> Result<CaptureUsage, CaptureError> {
    let values = match transform {
        CaptureTransform::Preview { max_elements } => count.min(*max_elements),
        _ => count,
    };
    // One bounded F32 assembly buffer; the ordinary payload bound additionally
    // covers result ownership, histogram edges, metadata and serialized values.
    super::assembly::assembly_payload_usage(transform, count, count == 0)?.checked_add(
        CaptureUsage {
            host_bytes: mul(values, 4)?,
            ..Default::default()
        },
    )
}

#[derive(Default)]
struct Sum {
    value: f64,
    correction: f64,
}
impl Sum {
    fn add(&mut self, value: f64) {
        let next = self.value + value;
        if self.value.is_finite() && value.is_finite() {
            self.correction += if self.value.abs() >= value.abs() {
                (self.value - next) + value
            } else {
                (value - next) + self.value
            };
        } else {
            self.correction = 0.0;
        }
        self.value = next;
    }
    fn finish(self) -> f64 {
        self.value + self.correction
    }
}

pub(super) fn assemble(
    receipt: &PartitionCaptureReceiptPlan,
    fragments: Vec<CapturedPartitionFragment>,
    ledger: &mut dyn CaptureReservation,
) -> Result<AssembledPartitionCapture, PartitionCaptureMergeError> {
    let plan = &receipt.plan;
    let context = &receipt.context;
    let selection = &plan.plan().selections[context.selection_index];
    let point = &plan.points()[context.selection_index];
    let slice = resolve_slice(point, selection, &receipt.global_shape)?;
    let count = elements(&slice.shape)?;
    let emitted = match selection.transform {
        CaptureTransform::Preview { max_elements } => count.min(max_elements),
        _ => count,
    };
    if receipt.combination != PartitionCaptureCombination::SumF64ToF32
        || fragments.len() != receipt.producers().count()
        || fragments.is_empty()
    {
        return Err(invalid("summed activation requires every admitted producer").into());
    }
    let precision = fragments[0].record.source_dtype.clone();
    validate_precision(precision.as_ref())?;
    let mut charged = CaptureUsage::default();
    for (fragment, (rank, projection)) in fragments.iter().zip(receipt.producers()) {
        validate_projection(selection, point, projection)?;
        if fragment.combination != receipt.combination
            || fragment.producer_rank != rank
            || fragment.plan_identity != plan.identity()
            || fragment.selection_index != context.selection_index
            || fragment.phase != context.phase
            || fragment.prediction != context.prediction
            || fragment.invocation != context.invocation
            || fragment.global_shape != receipt.global_shape
            || fragment.global_slice != slice
            || fragment.axis != projection.axis()
            || fragment.geometry != projection.fragments()[0]
            || fragment.record.source_dtype != precision
        {
            return Err(
                invalid("summed activation term differs from retained receipt authority").into(),
            );
        }
        if fragment.record.outcome
            != completed_capture_outcome(&native_transform(&selection.transform), count)
        {
            return Err(PartitionCaptureMergeError::FragmentOutcome {
                producer_rank: rank,
                outcome: fragment.record.outcome.clone(),
            });
        }
        let Some(CapturePayload::Tensor(tensor)) = &fragment.record.payload else {
            return Err(invalid("summed activation term has no raw tensor payload").into());
        };
        if !matches!(tensor.data(), TensorObservationData::F32(_))
            || tensor.data().len() as u64 != emitted
        {
            return Err(
                invalid("summed activation term changed its floating payload or size").into(),
            );
        }
        charged = charged.checked_add(fragment.record.charged)?;
    }
    let usage = super::assembly::assembly_metadata_usage(
        selection,
        point,
        slice.shape.len(),
        fragments.len(),
    )?
    .checked_add(payload_usage(&selection.transform, count)?)?;
    ledger.reserve_quota(usage)?;
    charged = charged.checked_add(usage)?;
    let emitted = usize::try_from(emitted).map_err(|_| CaptureError::Overflow)?;
    let mut values = Vec::with_capacity(emitted);
    for index in 0..emitted {
        let mut sum = Sum::default();
        for fragment in &fragments {
            let Some(CapturePayload::Tensor(tensor)) = &fragment.record.payload else {
                unreachable!()
            };
            let TensorObservationData::F32(terms) = tensor.data() else {
                unreachable!()
            };
            sum.add(f64::from(terms[index]));
        }
        values.push(sum.finish() as f32);
    }
    let payload = transform(values, &slice.shape, &selection.transform)?;
    let contributions = fragments
        .into_iter()
        .map(|fragment| PartitionCaptureContribution {
            producer_rank: fragment.producer_rank,
            geometry: fragment.geometry,
            charged: fragment.record.charged,
            routed: None,
        })
        .collect();
    Ok(AssembledPartitionCapture {
        combination: receipt.combination,
        plan_identity: plan.identity().into(),
        phase: context.phase,
        prediction: context.prediction,
        record: CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: selection.id.clone(),
            path: selection.path.clone(),
            node_id: point.node_id.clone(),
            position: point.position,
            source_shape: Some(receipt.global_shape.clone()),
            source_dtype: precision,
            selected_shape: Some(slice.shape),
            outcome: completed_capture_outcome(&selection.transform, count),
            payload: Some(payload),
            charged,
        },
        contributions,
    })
}

fn transform(
    values: Vec<f32>,
    shape: &[u64],
    transform: &CaptureTransform,
) -> Result<CapturePayload, CaptureError> {
    Ok(match transform {
        CaptureTransform::Slice
        | CaptureTransform::FullTensor
        | CaptureTransform::Preview { .. } => {
            let shape = if matches!(transform, CaptureTransform::Preview { .. }) {
                vec![values.len()]
            } else {
                shape
                    .iter()
                    .map(|n| usize::try_from(*n).map_err(|_| CaptureError::Overflow))
                    .collect::<Result<_, _>>()?
            };
            CapturePayload::Tensor(
                TensorObservation::new(shape, TensorObservationData::F32(values))
                    .map_err(|error| invalid(&error.to_string()))?,
            )
        }
        CaptureTransform::Summary => {
            let mut summary = CaptureSummary {
                elements: values.len() as u64,
                finite: 0,
                non_finite: 0,
                nan: 0,
                positive_infinity: 0,
                negative_infinity: 0,
                min: None,
                max: None,
                mean: None,
                rms: None,
            };
            let mut sum = Sum::default();
            let mut squares = Sum::default();
            for value in values {
                if value.is_finite() {
                    let value = f64::from(value);
                    summary.finite += 1;
                    summary.min = Some(summary.min.map_or(value, |min| min.min(value)));
                    summary.max = Some(summary.max.map_or(value, |max| max.max(value)));
                    sum.add(value);
                    squares.add(value * value);
                } else {
                    summary.non_finite += 1;
                    summary.nan += u64::from(value.is_nan());
                    summary.positive_infinity += u64::from(value == f32::INFINITY);
                    summary.negative_infinity += u64::from(value == f32::NEG_INFINITY);
                }
            }
            if summary.finite != 0 {
                summary.mean = Some(sum.finish() / summary.finite as f64);
                summary.rms = Some((squares.finish() / summary.finite as f64).sqrt());
            }
            CapturePayload::Summary(summary)
        }
        CaptureTransform::Histogram { edges } => {
            let mut histogram = CaptureHistogram {
                edges: edges.clone(),
                counts: vec![0; edges.len() - 1],
                below: 0,
                above: 0,
                non_finite: 0,
            };
            for value in values {
                if !value.is_finite() {
                    histogram.non_finite += 1;
                } else if value < edges[0] {
                    histogram.below += 1;
                } else if value > *edges.last().expect("admitted histogram") {
                    histogram.above += 1;
                } else {
                    let index = edges
                        .partition_point(|edge| *edge <= value)
                        .saturating_sub(1)
                        .min(histogram.counts.len() - 1);
                    histogram.counts[index] += 1;
                }
            }
            CapturePayload::Histogram(histogram)
        }
        _ => {
            return Err(CaptureError::Unsupported(
                "summed activation transform".into(),
            ))
        }
    })
}
