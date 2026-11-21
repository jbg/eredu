//! Dynamic expert-row receipts under the ordinary partition delivery protocol.
use super::*;
use eredu_core::{
    component::ComponentCoordinateMap, ObservationPoint, ObservationValueType, TensorObservation,
    TensorObservationData,
};
use std::collections::BTreeMap;

/// One expected sparse producer. The unit projection is rectangular; expert
/// ownership determines which participating token/slot rows it may contribute.
#[derive(Debug, Clone)]
pub struct PartitionRoutedCaptureProducer {
    /// World rank checked independently against the transport sender.
    pub rank: usize,
    /// Original global token/slot/unit selection projected onto local columns.
    pub projection: CaptureSlicePartition,
    /// Retained expert ownership and authoritative logical source peer.
    pub ownership: RoutedUnitCaptureOwnership,
}

fn invalid(message: &str) -> CaptureError {
    CaptureError::Invalid(message.into())
}

pub(super) fn ownership_bytes(owned: &RoutedUnitCaptureOwnership) -> Result<u64, CaptureError> {
    [owned.coordinates.experts(), owned.coordinates.units()]
        .into_iter()
        .try_fold(2048, |bytes, map| {
            add(
                bytes,
                if map.contiguous_range().is_some() {
                    64
                } else {
                    mul(map.local_count() as u64, 128)?
                },
            )
        })
}

/// Bounds copies of the retained ownership and native chunk evidence. A chunk
/// contains at least one native row; the receipt byte cap bounds it independently.
pub(super) fn provenance_bytes(plan: &PartitionCaptureReceiptPlan) -> Result<u64, CaptureError> {
    plan.producers().try_fold(0, |bytes, (rank, projection)| {
        let Some(owned) = plan.routed.get(&rank) else {
            return Ok(bytes);
        };
        let chunks = owned
            .maximum_source_rows(plan.global_shape[0], plan.global_shape[1])?
            .min(plan.max_record_bytes());
        add(
            bytes,
            mul(
                projection.fragments().len() as u64,
                add(ownership_bytes(owned)?, mul(chunks, 64)?)?,
            )?,
        )
    })
}

pub(super) fn payload_usage(
    plan: &PartitionCaptureReceiptPlan,
    slice: &ResolvedCaptureSlice,
) -> Result<CaptureUsage, CaptureError> {
    let values = elements(&slice.shape)?;
    let rows = if values == 0 {
        0
    } else {
        mul(slice.shape[0], slice.shape[1])?
    };
    let extra = provenance_bytes(plan)?;
    Ok(CaptureUsage {
        host_bytes: add(extra, add(mul(rows, 1024)?, mul(values, 32)?)?)?,
        encoded_bytes: add(extra, add(mul(rows, 768)?, mul(values, 32)?)?)?,
        ..Default::default()
    })
}

pub(in crate::capture::partition) fn maps_overlap(
    a: &ComponentCoordinateMap,
    b: &ComponentCoordinateMap,
) -> bool {
    if let (Some(a), Some(b)) = (a.contiguous_range(), b.contiguous_range()) {
        return a.start.max(b.start) < a.end.min(b.end);
    }
    let (small, large) = if a.local_count() <= b.local_count() {
        (a, b)
    } else {
        (b, a)
    };
    (0..small.local_count()).any(|index| {
        large
            .global_to_local(small.local_to_global(index).expect("checked map"))
            .is_some()
    })
}
pub(super) fn owners_overlap(
    owners: &BTreeMap<usize, RoutedUnitCaptureOwnership>,
    a: usize,
    b: usize,
) -> bool {
    match (owners.get(&a), owners.get(&b)) {
        (Some(a), Some(b)) => maps_overlap(a.coordinates.experts(), b.coordinates.experts()),
        _ => true,
    }
}

pub(super) fn validate_declarations(
    point: &ObservationPoint,
    selection: &CaptureSelection,
    producers: &[PartitionCaptureProducer],
    ownership: &BTreeMap<usize, RoutedUnitCaptureOwnership>,
    max_fragments: usize,
) -> Result<Option<RoutedUnitGeometry>, CaptureError> {
    if !matches!(selection.transform, CaptureTransform::RoutedUnits) {
        return if ownership.is_empty() {
            Ok(None)
        } else {
            Err(invalid("dense capture has sparse ownership"))
        };
    }
    let ObservationValueType::RoutedUnits { geometry, .. } = point.value_type else {
        return Err(invalid("sparse capture lacks bank geometry"));
    };
    if ownership.len() != producers.len() || ownership.is_empty() {
        return Err(invalid("sparse capture requires exact producer ownership"));
    }
    let source = ownership.values().next().expect("nonempty");
    for producer in producers {
        let owned = ownership
            .get(&producer.rank)
            .ok_or_else(|| invalid("sparse producer lacks ownership"))?;
        owned.validate(geometry)?;
        if owned.source_peer != source.source_peer
            || owned.source_peers != source.source_peers
            || producer.projection.axis() != 2
            || producer.projection.global_shape().len() != 3
            || producer.projection.global_shape()[1..]
                != [geometry.routes_per_token, geometry.units_per_expert]
        {
            return Err(invalid("sparse producer source or axes disagree"));
        }
        let projected = CaptureSlicePartition::new(
            producer.projection.global_shape(),
            producer.projection.global_slice(),
            2,
            owned.coordinates.units(),
            max_fragments.min(producer.projection.fragments().len()),
        )?;
        if projected != producer.projection {
            return Err(invalid(
                "sparse producer projection differs from its scalar map",
            ));
        }
    }
    Ok(Some(geometry))
}

/// Exact original coordinates addressed by this selected-result fragment.
pub(super) fn global_fragment_slice(
    projection: &CaptureSlicePartition,
    index: usize,
) -> Result<ResolvedCaptureSlice, CaptureError> {
    let selected = projection.global_slice();
    let destination = projection
        .fragments()
        .get(index)
        .ok_or_else(|| invalid("unknown sparse fragment"))?
        .destination();
    let mut result = ResolvedCaptureSlice {
        starts: vec![],
        ends: vec![],
        strides: vec![],
        shape: destination.shape.clone(),
    };
    for axis in 0..3 {
        let start = add(
            selected.starts[axis],
            mul(destination.starts[axis], selected.strides[axis])?,
        )?;
        let stride = mul(destination.strides[axis], selected.strides[axis])?;
        let count = destination.shape[axis];
        let end = if count == 0 {
            start
        } else {
            add(add(start, mul(count - 1, stride)?)?, 1)?
        };
        result.starts.push(start);
        result.ends.push(end);
        result.strides.push(stride);
    }
    Ok(result)
}

pub(super) fn validate_payload(
    plan: &PartitionCaptureReceiptPlan,
    rank: usize,
    projection: &CaptureSlicePartition,
    index: usize,
    payload: &RoutedUnitCapture,
) -> Result<(), CaptureError> {
    let point = &plan.plan.points()[plan.context.selection_index];
    let ObservationValueType::RoutedUnits { geometry, .. } = point.value_type else {
        return Err(invalid("missing routed bank geometry"));
    };
    let owned = plan
        .routed
        .get(&rank)
        .ok_or_else(|| invalid("missing routed producer ownership"))?;
    if payload.geometry != geometry {
        return Err(invalid("sparse receipt changed bank geometry"));
    }
    let slice = global_fragment_slice(projection, index)?;
    payload.validate_rows(&slice)?;
    if payload.rows.len() as u64 > mul(slice.shape[0], slice.shape[1])?
        || payload.rows.iter().any(|row| {
            row.source_peer != owned.source_peer
                || usize::try_from(row.expert)
                    .ok()
                    .and_then(|expert| owned.coordinates.experts().global_to_local(expert))
                    .is_none()
        })
    {
        return Err(invalid(
            "sparse receipt exceeds its expert/source ownership",
        ));
    }
    let bound = owned.maximum_source_rows(plan.global_shape[0], geometry.routes_per_token)?;
    let mut end = 0;
    for &[start, next] in &payload.source_token_ranges {
        if start != end || next <= start || next > bound {
            return Err(invalid("invalid sparse native chunk coverage"));
        }
        end = next;
    }
    if !payload.rows.is_empty()
        && (end == 0 || (owned.source_peer.is_none() && end != plan.global_shape[0]))
    {
        return Err(invalid("sparse rows have incomplete native chunk evidence"));
    }
    Ok(())
}

struct Row {
    expert: u64,
    coefficient: f32,
    values: Vec<f32>,
    seen: Vec<bool>,
    filled: u64,
}

pub(super) fn assemble(
    plan: &PartitionCaptureReceiptPlan,
    fragments: Vec<CapturedPartitionFragment>,
    ledger: &mut dyn CaptureReservation,
) -> Result<AssembledPartitionCapture, PartitionCaptureMergeError> {
    let selection = &plan.plan.plan().selections[plan.context.selection_index];
    let point = &plan.plan.points()[plan.context.selection_index];
    let ObservationValueType::RoutedUnits { geometry, .. } = point.value_type else {
        return Err(invalid("sparse assembly has no bank geometry").into());
    };
    let slice = resolve_slice(point, selection, &plan.global_shape)?;
    let values = elements(&slice.shape)?;
    let expected_rows = if values == 0 {
        0
    } else {
        mul(slice.shape[0], slice.shape[1])?
    };
    let units = usize::try_from(slice.shape[2]).map_err(|_| CaptureError::Overflow)?;
    let mut charged = CaptureUsage::default();
    for fragment in &fragments {
        if fragment.record.outcome != CaptureOutcome::Captured {
            return Err(PartitionCaptureMergeError::FragmentOutcome {
                producer_rank: fragment.producer_rank,
                outcome: fragment.record.outcome.clone(),
            });
        }
        let Some(CapturePayload::RoutedUnits(_)) = &fragment.record.payload else {
            return Err(invalid("sparse fragment has no routed payload").into());
        };
        charged = charged.checked_add(fragment.record.charged)?;
    }
    let usage = super::assembly::assembly_metadata_usage(selection, point, 3, fragments.len())?
        .checked_add(payload_usage(plan, &slice)?)?;
    ledger.reserve_quota(usage)?;
    charged = charged.checked_add(usage)?;
    let mut rows: BTreeMap<(Option<u64>, u64, u64), Row> = BTreeMap::new();
    let mut precision = None;
    let mut chunks = BTreeMap::new();
    for fragment in &fragments {
        let Some(CapturePayload::RoutedUnits(payload)) = &fragment.record.payload else {
            unreachable!()
        };
        if chunks
            .insert(fragment.producer_rank, &payload.source_token_ranges)
            .is_some_and(|previous| previous != &payload.source_token_ranges)
        {
            return Err(invalid("sparse producer unit fragments disagree on native chunks").into());
        }
        if !payload.rows.is_empty() {
            if precision
                .as_ref()
                .is_some_and(|previous| previous != &fragment.record.source_dtype)
            {
                return Err(invalid("sparse source precision disagrees").into());
            }
            precision = Some(fragment.record.source_dtype.clone());
        }
        let destination = fragment.geometry.destination();
        for row in &payload.rows {
            let key = (row.source_peer, row.token, row.slot);
            if !rows.contains_key(&key) && rows.len() as u64 >= expected_rows {
                return Err(invalid("excess sparse route rows").into());
            }
            let merged = rows.entry(key).or_insert_with(|| Row {
                expert: row.expert,
                coefficient: row.coefficient,
                values: vec![0.0; units],
                seen: vec![false; units],
                filled: 0,
            });
            if merged.expert != row.expert
                || merged.coefficient.to_bits() != row.coefficient.to_bits()
            {
                return Err(
                    invalid("sparse unit fragments disagree on expert or coefficient").into(),
                );
            }
            let TensorObservationData::F32(values) = row.values.data() else {
                return Err(invalid("sparse receipt has non-floating values").into());
            };
            for (index, &value) in values.iter().enumerate() {
                let destination = usize::try_from(add(
                    destination.starts[2],
                    mul(index as u64, destination.strides[2])?,
                )?)
                .map_err(|_| CaptureError::Overflow)?;
                let seen = merged
                    .seen
                    .get_mut(destination)
                    .ok_or_else(|| invalid("sparse unit destination exceeds selection"))?;
                if std::mem::replace(seen, true) {
                    return Err(PartitionCaptureMergeError::Overlap);
                }
                merged.values[destination] = value;
                merged.filled += 1;
            }
        }
    }
    let received = rows
        .values()
        .try_fold(0u64, |count, row| add(count, row.filled))?;
    if rows.len() as u64 != expected_rows || received != values {
        return Err(PartitionCaptureMergeError::Incomplete {
            expected_elements: values,
            received_elements: received,
        });
    }
    let rows = rows
        .into_iter()
        .map(|((source_peer, token, slot), row)| RoutedUnitCaptureRow {
            source_peer,
            token,
            slot,
            expert: row.expert,
            coefficient: row.coefficient,
            unit_start: slice.starts[2],
            unit_stride: slice.strides[2],
            values: TensorObservation::new(vec![units], TensorObservationData::F32(row.values))
                .expect("checked scalar extent"),
        })
        .collect();
    let payload = RoutedUnitCapture {
        geometry,
        source_token_ranges: vec![],
        rows,
    };
    payload.validate_rows(&slice)?;
    let contributions = fragments
        .into_iter()
        .map(|fragment| {
            let Some(CapturePayload::RoutedUnits(payload)) = fragment.record.payload else {
                unreachable!()
            };
            PartitionCaptureContribution {
                producer_rank: fragment.producer_rank,
                geometry: fragment.geometry,
                charged: fragment.record.charged,
                routed: Some(RoutedUnitCaptureProvenance {
                    ownership: plan.routed[&fragment.producer_rank].clone(),
                    source_token_ranges: payload.source_token_ranges,
                }),
            }
        })
        .collect();
    Ok(AssembledPartitionCapture {
        combination: PartitionCaptureCombination::Disjoint,
        plan_identity: plan.plan.identity().into(),
        phase: plan.context.phase,
        prediction: plan.context.prediction,
        record: CaptureRecord {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selection_id: selection.id.clone(),
            path: selection.path.clone(),
            node_id: point.node_id.clone(),
            position: point.position,
            source_shape: Some(plan.global_shape.clone()),
            source_dtype: precision.flatten(),
            selected_shape: Some(slice.shape),
            outcome: CaptureOutcome::Captured,
            payload: Some(CapturePayload::RoutedUnits(payload)),
            charged,
        },
        contributions,
    })
}
