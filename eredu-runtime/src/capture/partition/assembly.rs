//! Global tensor and finite-statistic assembly with exact bounded coverage checks.
use super::*;
use eredu_core::{
    checkpoint::TensorDtype, ObservationPoint, TensorObservation, TensorObservationData,
};

fn invalid(message: &str) -> PartitionCaptureMergeError {
    CaptureError::Invalid(message.into()).into()
}

pub(super) fn assembly_metadata_usage(
    selection: &CaptureSelection,
    point: &ObservationPoint,
    rank: usize,
    fragments: usize,
) -> Result<CaptureUsage, CaptureError> {
    metadata_reservation(selection, point)?.checked_add(CaptureUsage {
        host_bytes: mul(
            fragments as u64,
            std::mem::size_of::<PartitionCaptureContribution>() as u64,
        )?,
        encoded_bytes: mul(fragments as u64, add(512, mul(rank as u64, 320)?)?)?,
        ..Default::default()
    })
}

pub(super) fn assembly_payload_usage(
    transform: &CaptureTransform,
    elements: u64,
    empty: bool,
) -> Result<CaptureUsage, CaptureError> {
    let (host_bytes, encoded_bytes) = if empty {
        let extra = match transform {
            CaptureTransform::Histogram { edges } => mul(edges.len() as u64, 64)?,
            _ => 0,
        };
        (add(512, extra)?, add(1024, extra)?)
    } else {
        match transform {
            CaptureTransform::RoutedUnits => {
                return Err(CaptureError::Unsupported(
                    "routed-unit assembly requires retained expert/source ownership".into(),
                ))
            }
            CaptureTransform::FullTensor | CaptureTransform::Slice => {
                (mul(elements, 8)?, mul(elements, 32)?)
            }
            CaptureTransform::Preview { max_elements } => {
                let count = elements.min(*max_elements);
                (mul(count, 8)?, mul(count, 32)?)
            }
            CaptureTransform::Summary => (256, 1024),
            CaptureTransform::Histogram { edges } => (
                add(256, mul(edges.len() as u64, 16)?)?,
                add(1024, mul(edges.len() as u64, 64)?)?,
            ),
            CaptureTransform::TokenScores { token_ids } => (
                add(256, mul(token_ids.len() as u64, 128)?)?,
                add(1024, mul(token_ids.len() as u64, 512)?)?,
            ),
            CaptureTransform::TopCandidates { count } => {
                (add(256, mul(*count, 32)?)?, add(1024, mul(*count, 128)?)?)
            }
        }
    };
    Ok(CaptureUsage {
        host_bytes,
        encoded_bytes,
        ..Default::default()
    })
}

/// Moves a bounded reduction from the authoritative complete source, retaining
/// its global identity and producer provenance without exporting the logits.
pub(super) fn assemble_vocabulary_fragment(
    plan: &AdmittedCapturePlan,
    selection_index: usize,
    phase: CapturePhase,
    prediction: u64,
    global_shape: &[u64],
    mut fragments: Vec<CapturedPartitionFragment>,
    ledger: &mut dyn CaptureReservation,
) -> Result<AssembledPartitionCapture, PartitionCaptureMergeError> {
    let mut assembly = Assembly::validate(
        plan,
        selection_index,
        phase,
        prediction,
        global_shape,
        &fragments,
    )?;
    if fragments.len() != 1 {
        return Err(invalid(
            "vocabulary reduction requires a single complete fragment",
        ));
    }
    let payload = fragments[0]
        .record
        .payload
        .as_ref()
        .ok_or_else(|| invalid("missing vocabulary payload"))?;
    super::vocabulary::validate_payload(
        assembly.selection,
        assembly.point.position,
        global_shape,
        payload,
    )?;
    let usage = assembly_payload_usage(&assembly.selection.transform, assembly.elements, false)?;
    assembly.reserve(&fragments, usage, ledger)?;
    let payload = fragments[0]
        .record
        .payload
        .take()
        .expect("validated vocabulary payload");
    Ok(assembly.finish(fragments, payload))
}

struct Assembly<'a> {
    plan: &'a AdmittedCapturePlan,
    selection: &'a CaptureSelection,
    point: &'a ObservationPoint,
    phase: CapturePhase,
    prediction: u64,
    global_shape: &'a [u64],
    slice: ResolvedCaptureSlice,
    precision: Option<TensorDtype>,
    charged: CaptureUsage,
    elements: u64,
}

impl<'a> Assembly<'a> {
    fn validate(
        plan: &'a AdmittedCapturePlan,
        selection_index: usize,
        phase: CapturePhase,
        prediction: u64,
        global_shape: &'a [u64],
        fragments: &[CapturedPartitionFragment],
    ) -> Result<Self, PartitionCaptureMergeError> {
        let selection = plan
            .plan()
            .selections
            .get(selection_index)
            .ok_or_else(|| invalid("unknown global capture selection"))?;
        if prediction >= plan.request().max_predictions
            || !selection.schedule.includes(phase, prediction)
        {
            return Err(invalid(
                "global fragment assembly is outside its admitted schedule",
            ));
        }
        let point = &plan.points()[selection_index];
        let invocation = fragments.first().and_then(|fragment| fragment.invocation);
        plan.geometry_at(phase, prediction, invocation)?
            .validate_actual(point, global_shape)?;
        let slice = resolve_slice(point, selection, global_shape)?;
        let count = elements(&slice.shape)?;
        let Some(first) = fragments.first() else {
            return Err(PartitionCaptureMergeError::Incomplete {
                expected_elements: count,
                received_elements: 0,
            });
        };
        let axis = first.axis;
        let precision = first.record.source_dtype.clone();
        let mut charged = CaptureUsage::default();
        let mut coverage = 0u64;
        for (index, fragment) in fragments.iter().enumerate() {
            if fragment.combination != PartitionCaptureCombination::Disjoint
                || fragment.plan_identity != plan.identity()
                || fragment.selection_index != selection_index
                || fragment.phase != phase
                || fragment.prediction != prediction
                || fragment.invocation != invocation
                || fragment.axis != axis
                || fragment.global_shape != global_shape
                || fragment.global_slice != slice
            {
                return Err(invalid(
                    "partition fragment identity, geometry or precision disagrees",
                ));
            }
            if fragment.record.outcome
                != completed_capture_outcome(
                    &selection.transform,
                    elements(&fragment.geometry.local().shape)?,
                )
            {
                return Err(PartitionCaptureMergeError::FragmentOutcome {
                    producer_rank: fragment.producer_rank,
                    outcome: fragment.record.outcome.clone(),
                });
            }
            if fragment.record.source_dtype != precision {
                return Err(invalid("partition fragment source precision disagrees"));
            }
            // Geometry is constructed by CaptureSlicePartition. Every fragment
            // spans the same selected axes except its one partitioned axis.
            for other in &fragments[..index] {
                if fragment.geometry.overlaps_destination(&other.geometry)? {
                    return Err(PartitionCaptureMergeError::Overlap);
                }
            }
            coverage = add(coverage, fragment.geometry.destination().shape[axis])?;
            charged = charged.checked_add(fragment.record.charged)?;
        }
        if coverage != slice.shape[axis] {
            return Err(PartitionCaptureMergeError::Incomplete {
                expected_elements: count,
                received_elements: mul(count / slice.shape[axis], coverage)?,
            });
        }
        Ok(Self {
            plan,
            selection,
            point,
            phase,
            prediction,
            global_shape,
            slice,
            precision,
            charged,
            elements: count,
        })
    }

    fn reserve(
        &mut self,
        fragments: &[CapturedPartitionFragment],
        payload: CaptureUsage,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<(), PartitionCaptureMergeError> {
        let usage = assembly_metadata_usage(
            self.selection,
            self.point,
            self.slice.shape.len(),
            fragments.len(),
        )?
        .checked_add(payload)?;
        ledger.reserve_quota(usage)?;
        self.charged = self.charged.checked_add(usage)?;
        Ok(())
    }

    fn finish(
        self,
        fragments: Vec<CapturedPartitionFragment>,
        payload: CapturePayload,
    ) -> AssembledPartitionCapture {
        let contributions = fragments
            .into_iter()
            .map(|fragment| PartitionCaptureContribution {
                producer_rank: fragment.producer_rank,
                geometry: fragment.geometry,
                charged: fragment.record.charged,
                routed: None,
            })
            .collect();
        let record = CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: self.selection.id.clone(),
            path: self.selection.path.clone(),
            node_id: self.point.node_id.clone(),
            position: self.point.position,
            source_shape: Some(self.global_shape.to_vec()),
            source_dtype: self.precision,
            selected_shape: Some(self.slice.shape),
            outcome: completed_capture_outcome(&self.selection.transform, self.elements),
            payload: Some(payload),
            charged: self.charged,
        };
        AssembledPartitionCapture {
            combination: PartitionCaptureCombination::Disjoint,
            plan_identity: self.plan.identity().into(),
            phase: self.phase,
            prediction: self.prediction,
            record,
            contributions,
        }
    }
}

/// Reassembles raw tensor fragments after validating exact global coverage.
/// Allocation/copy space is reserved first. Producer evidence remains attached;
/// the enclosing driver must establish transport and completion before publication.
pub fn assemble_tensor_fragments(
    plan: &AdmittedCapturePlan,
    selection_index: usize,
    phase: CapturePhase,
    prediction: u64,
    global_shape: &[u64],
    fragments: Vec<CapturedPartitionFragment>,
    ledger: &mut dyn CaptureReservation,
) -> Result<AssembledPartitionCapture, PartitionCaptureMergeError> {
    let mut assembly = Assembly::validate(
        plan,
        selection_index,
        phase,
        prediction,
        global_shape,
        &fragments,
    )?;
    if !matches!(
        assembly.selection.transform,
        CaptureTransform::FullTensor | CaptureTransform::Slice | CaptureTransform::Preview { .. }
    ) {
        return Err(invalid(
            "tensor fragment assembly requires a raw tensor selection",
        ));
    }
    let mut payload_kind = None;
    for fragment in &fragments {
        let Some(CapturePayload::Tensor(value)) = &fragment.record.payload else {
            return Err(invalid("raw capture fragment has no tensor payload"));
        };
        let correct_shape =
            if let CaptureTransform::Preview { max_elements } = assembly.selection.transform {
                value.shape().len() == 1
                    && value.shape()[0] as u64
                        == max_elements.min(elements(&fragment.geometry.local().shape)?)
            } else {
                value.shape().len() == fragment.geometry.local().shape.len()
                    && !value
                        .shape()
                        .iter()
                        .zip(&fragment.geometry.local().shape)
                        .any(|(actual, expected)| *actual as u64 != *expected)
            };
        if !correct_shape {
            return Err(invalid(
                "raw capture fragment payload has incorrect geometry",
            ));
        }
        let kind = std::mem::discriminant(value.data());
        if payload_kind.is_some_and(|expected| expected != kind) {
            return Err(invalid("raw capture fragment payload dtypes disagree"));
        }
        payload_kind = Some(kind);
    }
    assembly.reserve(
        &fragments,
        assembly_payload_usage(&assembly.selection.transform, assembly.elements, false)?,
        ledger,
    )?;
    let emitted = if let CaptureTransform::Preview { max_elements } = assembly.selection.transform {
        assembly.elements.min(max_elements)
    } else {
        assembly.elements
    };
    let output_count = usize::try_from(emitted).map_err(|_| CaptureError::Overflow)?;
    let Some(CapturePayload::Tensor(first)) = &fragments[0].record.payload else {
        unreachable!()
    };
    let mut values = match first.data() {
        TensorObservationData::F32(_) => TensorObservationData::F32(vec![0.0; output_count]),
        TensorObservationData::I64(_) => TensorObservationData::I64(vec![0; output_count]),
        TensorObservationData::U64(_) => TensorObservationData::U64(vec![0; output_count]),
        TensorObservationData::Bool(_) => TensorObservationData::Bool(vec![false; output_count]),
    };
    let mut copied = 0u64;
    for fragment in &fragments {
        let destination = fragment.geometry.destination();
        let Some(CapturePayload::Tensor(value)) = &fragment.record.payload else {
            unreachable!()
        };
        for local in 0..value.data().len() {
            let mut remainder = local as u64;
            let mut output = 0u64;
            let mut output_stride = 1u64;
            for dimension in (0..assembly.slice.shape.len()).rev() {
                let coordinate = remainder % destination.shape[dimension];
                remainder /= destination.shape[dimension];
                output += (destination.starts[dimension]
                    + coordinate * destination.strides[dimension])
                    * output_stride;
                output_stride *= assembly.slice.shape[dimension];
            }
            // A fragment's destination ordinals strictly increase. Its local
            // prefix of length N therefore includes every destination below N.
            // Exact disjoint coverage was checked before allocating this prefix.
            if output >= emitted {
                continue;
            }
            copied += 1;
            match (&mut values, value.data()) {
                (TensorObservationData::F32(out), TensorObservationData::F32(input)) => {
                    out[output as usize] = input[local]
                }
                (TensorObservationData::I64(out), TensorObservationData::I64(input)) => {
                    out[output as usize] = input[local]
                }
                (TensorObservationData::U64(out), TensorObservationData::U64(input)) => {
                    out[output as usize] = input[local]
                }
                (TensorObservationData::Bool(out), TensorObservationData::Bool(input)) => {
                    out[output as usize] = input[local]
                }
                _ => unreachable!("payload dtypes validated before allocation"),
            }
        }
    }
    if copied != emitted {
        return Err(PartitionCaptureMergeError::Incomplete {
            expected_elements: emitted,
            received_elements: copied,
        });
    }
    let tensor_shape = if matches!(
        assembly.selection.transform,
        CaptureTransform::Preview { .. }
    ) {
        vec![output_count]
    } else {
        assembly
            .slice
            .shape
            .iter()
            .map(|extent| usize::try_from(*extent).map_err(|_| CaptureError::Overflow))
            .collect::<Result<Vec<_>, _>>()?
    };
    let payload = TensorObservation::new(tensor_shape, values)
        .map_err(|error| invalid(&error.to_string()))?;
    Ok(assembly.finish(fragments, CapturePayload::Tensor(payload)))
}

/// Merges bounded native summaries or fixed-edge histograms without exporting
/// source elements or allocating storage proportional to selected tensor width.
/// Counts and coverage are exact; finite means/RMS combine native F32-converted
/// statistics in F64 and can differ from another native reduction order.
pub fn assemble_reduced_fragments(
    plan: &AdmittedCapturePlan,
    selection_index: usize,
    phase: CapturePhase,
    prediction: u64,
    global_shape: &[u64],
    fragments: Vec<CapturedPartitionFragment>,
    ledger: &mut dyn CaptureReservation,
) -> Result<AssembledPartitionCapture, PartitionCaptureMergeError> {
    let mut assembly = Assembly::validate(
        plan,
        selection_index,
        phase,
        prediction,
        global_shape,
        &fragments,
    )?;
    match &assembly.selection.transform {
        CaptureTransform::Summary => {
            // No heap allocation while validating/combining scalar statistics.
            let summary = merge_summaries(&fragments)?;
            assembly.reserve(
                &fragments,
                assembly_payload_usage(&assembly.selection.transform, assembly.elements, false)?,
                ledger,
            )?;
            Ok(assembly.finish(fragments, CapturePayload::Summary(summary)))
        }
        CaptureTransform::Histogram { edges } => {
            for fragment in &fragments {
                let Some(CapturePayload::Histogram(value)) = &fragment.record.payload else {
                    return Err(invalid(
                        "histogram capture fragment has no histogram payload",
                    ));
                };
                if &value.edges != edges || value.counts.len() + 1 != edges.len() {
                    return Err(invalid("histogram fragment changed its admitted bin edges"));
                }
                let count = value.counts.iter().try_fold(
                    add(add(value.below, value.above)?, value.non_finite)?,
                    |sum, count| add(sum, *count),
                )?;
                if count != elements(&fragment.geometry.local().shape)? {
                    return Err(invalid(
                        "histogram fragment counts disagree with its selected geometry",
                    ));
                }
            }
            assembly.reserve(
                &fragments,
                assembly_payload_usage(&assembly.selection.transform, assembly.elements, false)?,
                ledger,
            )?;
            let mut output = CaptureHistogram {
                edges: edges.clone(),
                counts: vec![0; edges.len() - 1],
                below: 0,
                above: 0,
                non_finite: 0,
            };
            for fragment in &fragments {
                let Some(CapturePayload::Histogram(value)) = &fragment.record.payload else {
                    unreachable!()
                };
                output.below = add(output.below, value.below)?;
                output.above = add(output.above, value.above)?;
                output.non_finite = add(output.non_finite, value.non_finite)?;
                for (out, count) in output.counts.iter_mut().zip(&value.counts) {
                    *out = add(*out, *count)?;
                }
            }
            Ok(assembly.finish(fragments, CapturePayload::Histogram(output)))
        }
        _ => Err(invalid(
            "reduced fragment assembly requires a summary or histogram selection",
        )),
    }
}

#[derive(Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}
impl CompensatedSum {
    fn add(&mut self, value: f64) -> Result<(), PartitionCaptureMergeError> {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
        if !self.sum.is_finite() || !self.correction.is_finite() {
            return Err(invalid(
                "summary fragment aggregation overflowed finite statistics",
            ));
        }
        Ok(())
    }
    fn value(&self) -> f64 {
        self.sum + self.correction
    }
}

fn merge_summaries(
    fragments: &[CapturedPartitionFragment],
) -> Result<CaptureSummary, PartitionCaptureMergeError> {
    let mut output = CaptureSummary {
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
    };
    let mut sum = CompensatedSum::default();
    let mut squares = CompensatedSum::default();
    for fragment in fragments {
        let Some(CapturePayload::Summary(value)) = &fragment.record.payload else {
            return Err(invalid("summary capture fragment has no summary payload"));
        };
        if value.elements != elements(&fragment.geometry.local().shape)?
            || add(value.finite, value.non_finite)? != value.elements
            || add(
                add(value.nan, value.positive_infinity)?,
                value.negative_infinity,
            )? != value.non_finite
        {
            return Err(invalid(
                "summary fragment counts disagree with its selected geometry",
            ));
        }
        if value.finite == 0 {
            if [value.min, value.max, value.mean, value.rms]
                .iter()
                .any(Option::is_some)
            {
                return Err(invalid("summary fragment fabricated finite aggregates"));
            }
        } else {
            let (Some(min), Some(max), Some(mean), Some(rms)) =
                (value.min, value.max, value.mean, value.rms)
            else {
                return Err(invalid("summary fragment is missing finite aggregates"));
            };
            if ![min, max, mean, rms].iter().all(|v| v.is_finite()) || min > max || rms < 0.0 {
                return Err(invalid(
                    "summary fragment contains invalid finite aggregates",
                ));
            }
            output.min = Some(output.min.map_or(min, |current| current.min(min)));
            output.max = Some(output.max.map_or(max, |current| current.max(max)));
            sum.add(mean * value.finite as f64)?;
            squares.add(rms * rms * value.finite as f64)?;
        }
        output.elements = add(output.elements, value.elements)?;
        output.finite = add(output.finite, value.finite)?;
        output.non_finite = add(output.non_finite, value.non_finite)?;
        output.nan = add(output.nan, value.nan)?;
        output.positive_infinity = add(output.positive_infinity, value.positive_infinity)?;
        output.negative_infinity = add(output.negative_infinity, value.negative_infinity)?;
    }
    if output.finite != 0 {
        output.mean = Some(sum.value() / output.finite as f64);
        output.rms = Some((squares.value() / output.finite as f64).sqrt());
        if !output.mean.unwrap().is_finite() || !output.rms.unwrap().is_finite() {
            return Err(invalid(
                "summary fragment aggregation overflowed finite statistics",
            ));
        }
    }
    Ok(output)
}
