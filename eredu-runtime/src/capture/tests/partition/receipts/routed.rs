use super::*;
use eredu_core::component::RoutedComponentCoordinateMap;

pub(super) const GEOMETRY: RoutedUnitGeometry = RoutedUnitGeometry {
    experts: 5,
    units_per_expert: 7,
    routes_per_token: 2,
};

pub(super) fn plan(empty: bool) -> AdmittedCapturePlan {
    plan_with_effective(empty, false)
}

pub(super) fn plan_with_effective(empty: bool, effective: bool) -> AdmittedCapturePlan {
    let (mut plan, mut catalog, mut support, mut capabilities) =
        fixture(CaptureTransform::RoutedUnits);
    capabilities
        .transformations
        .push(CaptureTransformKind::RoutedUnits);
    catalog.points[0].value_type = ObservationValueType::RoutedUnits {
        routing: "experts".into(),
        geometry: GEOMETRY,
    };
    catalog.points[0].axes = Some(vec![
        TensorAxis {
            name: "token".into(),
            dimension: SymbolicDimension::TokenRows,
        },
        TensorAxis {
            name: "route".into(),
            dimension: SymbolicDimension::Known(2),
        },
        TensorAxis {
            name: "component".into(),
            dimension: SymbolicDimension::Known(7),
        },
    ]);
    plan.selections[0].schedule.decode = false;
    plan.selections[0].slices = vec![
        CaptureSlice {
            axis: "token".into(),
            start: 0,
            end: 3,
            stride: 2,
        },
        CaptureSlice {
            axis: "component".into(),
            start: 1,
            end: if empty { 1 } else { 7 },
            stride: 2,
        },
    ];
    plan.limits.per_step = CaptureUsage {
        captures: 1000,
        retained_bytes: 1_000_000_000,
        host_bytes: 1_000_000_000,
        encoded_bytes: 1_000_000_000,
    };
    plan.limits.cumulative = plan.limits.per_step;
    if effective {
        let mut point = catalog.points[0].clone();
        point.path.push_str(".effective");
        point.position = ObservationPosition::AfterIntervention;
        let mut selection = plan.selections[0].clone();
        selection.path = point.path.clone();
        selection.id.push_str("-effective");
        let mut supported = support.points[0].clone();
        supported.path = point.path.clone();
        catalog.points.push(point);
        plan.selections.push(selection);
        support.points.push(supported);
    }
    admit(plan, &catalog, &support, &capabilities).unwrap()
}

pub(super) fn limits() -> PartitionCaptureReceiptLimits {
    PartitionCaptureReceiptLimits {
        max_producers: 8,
        max_fragments: 32,
        max_record_bytes: 16384,
    }
}

pub(super) fn producers(
    plan: &AdmittedCapturePlan,
    exchange: bool,
) -> Vec<PartitionRoutedCaptureProducer> {
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 2, 7]).unwrap();
    let mut producers = vec![];
    // Unequal EP ownership. Expert 4 has no selected routes; its acknowledgment
    // remains required. Each EP owner uses permuted TP scalar columns.
    for experts in [vec![4], vec![1], vec![0, 3, 2]] {
        for units in [vec![5, 0, 3], vec![6, 1, 2, 4]] {
            let units = ComponentCoordinateMap::indices(7, units).unwrap();
            producers.push(PartitionRoutedCaptureProducer {
                rank: producers.len(),
                projection: CaptureSlicePartition::new(&[3, 2, 7], &slice, 2, &units, 32).unwrap(),
                ownership: RoutedUnitCaptureOwnership {
                    coordinates: RoutedComponentCoordinateMap::new(
                        ComponentCoordinateMap::indices(5, experts.clone()).unwrap(),
                        units,
                    ),
                    source_peer: exchange.then_some(0),
                    source_peers: if exchange { 3 } else { 1 },
                },
            });
        }
    }
    let units = ComponentCoordinateMap::range(7, 0..1).unwrap();
    producers.push(PartitionRoutedCaptureProducer {
        rank: 6,
        projection: CaptureSlicePartition::new(&[3, 2, 7], &slice, 2, &units, 32).unwrap(),
        ownership: RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(5, 0..5).unwrap(),
                units,
            ),
            source_peer: exchange.then_some(0),
            source_peers: if exchange { 3 } else { 1 },
        },
    });
    producers
}

pub(super) fn receipt(
    plan: &AdmittedCapturePlan,
    exchange: bool,
    ledger: &mut CaptureLedger,
) -> PartitionCaptureReceiptPlan {
    PartitionCaptureReceiptPlan::new_routed(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(plan),
        producers(plan, exchange),
        8,
        limits(),
        ledger,
    )
    .unwrap()
}

fn expert(token: u64, slot: u64) -> u64 {
    if token == 0 {
        1
    } else {
        (token + slot) % 5
    }
}

fn scalar(token: u64, slot: u64, unit: u64) -> f32 {
    (token * 100 + slot * 10 + unit) as f32 * 0.125 - 3.25
}

/// Independent host fixture emulates actual sorted provider output. It uses
/// public wire evidence; delivery must revalidate it against retained admission.
pub(super) fn records(
    plan: &AdmittedCapturePlan,
    receipt: &PartitionCaptureReceiptPlan,
) -> Vec<PartitionCaptureProducerRecord> {
    receipt
        .producers()
        .map(|(rank, projection)| {
            let owned = receipt.routed_producer(rank).unwrap();
            let fragments = projection
                .fragments()
                .iter()
                .enumerate()
                .map(|(fragment_index, fragment)| {
                    let selection = &plan.plan().selections[0];
                    let point = &plan.points()[0];
                    let selected = projection.global_slice();
                    let destination = fragment.destination();
                    let unit_start =
                        selected.starts[2] + destination.starts[2] * selected.strides[2];
                    let unit_stride = destination.strides[2] * selected.strides[2];
                    let mut rows = vec![];
                    for token in [2, 0] {
                        // Native provider order is not logical token order.
                        for slot in [1, 0] {
                            let expert = expert(token, slot);
                            if owned
                                .coordinates
                                .experts()
                                .global_to_local(expert as usize)
                                .is_none()
                            {
                                continue;
                            }
                            let values = (0..destination.shape[2])
                                .map(|index| scalar(token, slot, unit_start + index * unit_stride))
                                .collect::<Vec<_>>();
                            rows.push(RoutedUnitCaptureRow {
                                source_peer: owned.source_peer,
                                token,
                                slot,
                                expert,
                                coefficient: if slot == 0 { 0.75 } else { 0.25 },
                                unit_start,
                                unit_stride,
                                values: TensorObservation::new(
                                    vec![values.len()],
                                    TensorObservationData::F32(values),
                                )
                                .unwrap(),
                            });
                        }
                    }
                    let source_token_ranges = if owned.source_peer.is_none() {
                        vec![[0, 1], [1, 3]]
                    } else if rank < 2 {
                        vec![]
                    }
                    // Empty EP owner.
                    else {
                        vec![[0, 2], [2, 4]]
                    }; // Receive order, not original token ranges.
                    PartitionCaptureFragmentRecord {
                        fragment_index,
                        record: CaptureRecord {
                            schema_version: CAPTURE_SCHEMA_VERSION,
                            selection_id: selection.id.clone(),
                            path: selection.path.clone(),
                            node_id: point.node_id.clone(),
                            position: point.position,
                            source_shape: Some(projection.local_shape().to_vec()),
                            source_dtype: Some(TensorDtype::F32),
                            selected_shape: Some(fragment.local().shape.clone()),
                            outcome: CaptureOutcome::Captured,
                            payload: Some(CapturePayload::RoutedUnits(RoutedUnitCapture {
                                geometry: GEOMETRY,
                                source_token_ranges,
                                rows,
                            })),
                            charged: CaptureUsage {
                                captures: 1,
                                retained_bytes: 4096,
                                host_bytes: 8192,
                                encoded_bytes: 16384,
                            },
                        },
                    }
                })
                .collect();
            PartitionCaptureProducerRecord {
                combination: PartitionCaptureCombination::Disjoint,
                schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
                receipt_plan_identity: receipt.identity().into(),
                context: context(plan),
                producer_rank: rank,
                source_dtype: Some(TensorDtype::F32),
                fragments,
            }
        })
        .collect()
}

pub(super) fn assert_values(result: &ReceivedPartitionCapture, exchange: bool) {
    assert_eq!(result.producers(), [0, 1, 2, 3, 4, 5, 6]);
    let record = result.capture().record();
    assert_eq!(record.source_shape.as_deref(), Some(&[3, 2, 7][..]));
    assert_eq!(record.selected_shape.as_deref(), Some(&[2, 2, 3][..]));
    let Some(CapturePayload::RoutedUnits(payload)) = &record.payload else {
        panic!("sparse payload")
    };
    assert!(
        payload.source_token_ranges.is_empty(),
        "global evidence must not invent native chunk coverage"
    );
    assert_eq!(payload.rows.len(), 4);
    for (row, (token, slot)) in payload.rows.iter().zip([(0, 0), (0, 1), (2, 0), (2, 1)]) {
        assert_eq!(
            (row.source_peer, row.token, row.slot, row.expert),
            (exchange.then_some(0), token, slot, expert(token, slot))
        );
        assert_eq!((row.unit_start, row.unit_stride), (1, 2));
        assert_eq!(
            row.values.data(),
            &TensorObservationData::F32(
                [1, 3, 5]
                    .into_iter()
                    .map(|unit| scalar(token, slot, unit))
                    .collect()
            )
        );
    }
    for contribution in result.capture().contributions() {
        let provenance = contribution.routed().unwrap();
        assert_eq!(provenance.ownership.source_peer, exchange.then_some(0));
        if exchange && contribution.producer_rank() >= 2 {
            assert_eq!(provenance.source_token_ranges, [[0, 2], [2, 4]]);
        }
    }
}

#[test]
fn routed_receipts_merge_tp_columns_and_uneven_experts_with_prepaid_delivery() {
    for exchange in [false, true] {
        let plan = plan(false);
        let mut ledger = CaptureLedger::new(&plan);
        let receipt = receipt(&plan, exchange, &mut ledger);
        let records = records(&plan, &receipt);
        assert!(!records[0].fragments.is_empty());
        assert!(records[6].fragments.is_empty());
        let mut quota = ledger
            .reserve_quota(receipt.delivery_usage().unwrap())
            .unwrap();
        let charged = ledger.total();
        let mut delivery = receipt.into_delivery();
        for record in records.into_iter().rev() {
            delivery
                .receive(
                    record.producer_rank,
                    &serde_json::to_vec(&record).unwrap(),
                    &mut quota,
                )
                .unwrap();
        }
        let result = delivery.finish(&mut quota).unwrap();
        assert_values(&result, exchange);
        assert_eq!(
            ledger.total(),
            charged,
            "prepaid delivery must not charge the parent twice"
        );
    }
}

#[test]
fn sparse_ownership_is_complete_disjoint_and_part_of_receipt_identity() {
    let plan = plan(false);
    let mut ledger = CaptureLedger::new(&plan);
    let expected = receipt(&plan, true, &mut ledger);
    for failure in [
        "overlap",
        "missing",
        "source",
        "peers",
        "projection",
        "expert_extent",
        "dense",
    ] {
        let mut owned = producers(&plan, true);
        match failure {
            "overlap" => {
                let mut duplicate = owned[2].clone();
                duplicate.rank = 7;
                owned.push(duplicate);
            }
            "missing" => {
                owned.remove(2);
            }
            "source" => owned[0].ownership.source_peer = Some(1),
            "peers" => owned[0].ownership.source_peers = 0,
            "projection" => owned[0].ownership.coordinates = owned[1].ownership.coordinates.clone(),
            "expert_extent" => {
                owned[0].ownership.coordinates = RoutedComponentCoordinateMap::new(
                    ComponentCoordinateMap::range(6, 0..1).unwrap(),
                    owned[0].ownership.coordinates.units().clone(),
                )
            }
            _ => {
                let dense = owned
                    .into_iter()
                    .map(|p| PartitionCaptureProducer {
                        rank: p.rank,
                        projection: p.projection,
                    })
                    .collect();
                assert!(PartitionCaptureReceiptPlan::new(
                    eredu_core::capture::SharedCapturePlan::new(plan.clone()),
                    context(&plan),
                    dense,
                    8,
                    limits(),
                    &mut ledger
                )
                .is_err());
                continue;
            }
        }
        assert!(
            PartitionCaptureReceiptPlan::new_routed(
                eredu_core::capture::SharedCapturePlan::new(plan.clone()),
                context(&plan),
                owned,
                8,
                limits(),
                &mut ledger
            )
            .is_err(),
            "{failure}"
        );
    }
    let mut owned = producers(&plan, true);
    owned.reverse();
    let reordered = PartitionCaptureReceiptPlan::new_routed(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(&plan),
        owned,
        8,
        limits(),
        &mut ledger,
    )
    .unwrap();
    assert_eq!(reordered.identity(), expected.identity());
    let mut owned = producers(&plan, true);
    for p in &mut owned {
        p.ownership.source_peer = Some(1);
    }
    let other_peer = PartitionCaptureReceiptPlan::new_routed(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(&plan),
        owned,
        8,
        limits(),
        &mut ledger,
    )
    .unwrap();
    assert_ne!(other_peer.identity(), expected.identity());
    let mut owned = producers(&plan, true);
    for p in &mut owned[4..6] {
        p.ownership.coordinates = RoutedComponentCoordinateMap::new(
            ComponentCoordinateMap::indices(5, vec![2, 3, 0]).unwrap(),
            p.ownership.coordinates.units().clone(),
        );
    }
    let permuted = PartitionCaptureReceiptPlan::new_routed(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(&plan),
        owned,
        8,
        limits(),
        &mut ledger,
    )
    .unwrap();
    assert_ne!(
        permuted.identity(),
        expected.identity(),
        "expert storage order is retained even when ownership sets agree"
    );
}

fn payload(record: &mut PartitionCaptureProducerRecord) -> &mut RoutedUnitCapture {
    let Some(CapturePayload::RoutedUnits(payload)) = &mut record.fragments[0].record.payload else {
        panic!("sparse")
    };
    payload
}

#[test]
fn sparse_preparation_respects_exact_prepaid_metadata_and_never_refunds_rejection() {
    let plan = plan(false);
    let mut ledger = CaptureLedger::new(&plan);
    let declared = producers(&plan, true);
    let bound = PartitionCaptureReceiptPlan::routed_preparation_usage(&declared).unwrap();
    let mut quota = ledger.reserve_quota(bound).unwrap();
    PartitionCaptureReceiptPlan::new_routed(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(&plan),
        declared,
        8,
        limits(),
        &mut quota,
    )
    .unwrap();
    assert_eq!(quota.used(), bound);
    let mut short = bound;
    short.host_bytes -= 1;
    let mut quota = ledger.reserve_quota(short).unwrap();
    let charged = ledger.total();
    let rejected = PartitionCaptureReceiptPlan::new_routed(
        eredu_core::capture::SharedCapturePlan::new(plan.clone()),
        context(&plan),
        producers(&plan, true),
        8,
        limits(),
        &mut quota,
    );
    assert!(matches!(
        rejected,
        Err(PartitionCaptureMergeError::Capture(CaptureError::Limit {
            budget: CaptureBudget::Host,
            ..
        }))
    ));
    assert!(quota.used().host_bytes > 0);
    assert_eq!(ledger.total(), charged);
}

#[test]
fn sparse_wire_rejects_changed_ownership_geometry_and_chunk_evidence() {
    let plan = plan(false);
    for failure in [
        "expert",
        "peer",
        "token",
        "slot",
        "unit_start",
        "unit_stride",
        "shape",
        "duplicate",
        "coefficient",
        "bank",
        "chunk_gap",
        "chunk_overlap",
        "chunk_bound",
        "chunk_missing",
        "dtype",
        "integer",
    ] {
        let mut ledger = CaptureLedger::new(&plan);
        let receipt = receipt(&plan, true, &mut ledger);
        let mut records = records(&plan, &receipt);
        let record = &mut records[2];
        let data = payload(record);
        match failure {
            "expert" => data.rows[0].expert = 0,
            "peer" => data.rows[0].source_peer = Some(2),
            "token" => data.rows[0].token = 1,
            "slot" => data.rows[0].slot = 2,
            "unit_start" => data.rows[0].unit_start += 1,
            "unit_stride" => data.rows[0].unit_stride += 1,
            "shape" => {
                data.rows[0].values =
                    TensorObservation::new(vec![0], TensorObservationData::F32(vec![])).unwrap()
            }
            "duplicate" => data.rows.push(data.rows[0].clone()),
            "coefficient" => data.rows[0].coefficient = f32::NAN,
            "bank" => data.geometry.experts += 1,
            "chunk_gap" => data.source_token_ranges[1][0] += 1,
            "chunk_overlap" => data.source_token_ranges[1][0] -= 1,
            "chunk_bound" => data.source_token_ranges[1][1] = 19,
            "chunk_missing" => data.source_token_ranges.clear(),
            "integer" => {
                record.source_dtype = Some(TensorDtype::I64);
                for fragment in &mut record.fragments {
                    fragment.record.source_dtype = Some(TensorDtype::I64);
                }
            }
            _ => record.fragments[0].record.source_dtype = Some(TensorDtype::Bf16),
        }
        let mut delivery = receipt.into_delivery();
        let before = ledger.total();
        assert!(
            delivery
                .receive(2, &serde_json::to_vec(record).unwrap(), &mut ledger)
                .is_err(),
            "{failure}"
        );
        assert!(delivery.missing_producers().any(|rank| rank == 2));
        assert!(
            ledger.total().host_bytes > before.host_bytes,
            "failed parsing remains charged"
        );
    }
}

#[test]
fn sparse_assembly_rejects_missing_routes_mismatched_tp_metadata_and_idle_receipts() {
    let plan = plan(false);
    for failure in ["missing_row", "coefficient", "chunks", "idle"] {
        let mut ledger = CaptureLedger::new(&plan);
        let receipt = receipt(&plan, true, &mut ledger);
        let mut records = records(&plan, &receipt);
        match failure {
            "missing_row" => {
                payload(&mut records[2]).rows.pop();
            }
            "coefficient" => payload(&mut records[2]).rows[0].coefficient = 0.5,
            "chunks" => {
                assert!(records[2].fragments.len() > 1);
                payload(&mut records[2]).source_token_ranges = vec![[0, 4]];
            }
            _ => {
                records.remove(6);
            }
        }
        let mut delivery = receipt.into_delivery();
        for record in records {
            delivery
                .receive(
                    record.producer_rank,
                    &serde_json::to_vec(&record).unwrap(),
                    &mut ledger,
                )
                .unwrap();
        }
        let error = delivery.finish(&mut ledger).unwrap_err();
        if failure == "idle" {
            assert!(matches!(
                error,
                PartitionCaptureMergeError::MissingProducer { producer_rank: 6 }
            ));
        }
        if failure == "missing_row" {
            assert!(matches!(
                error,
                PartitionCaptureMergeError::Incomplete { .. }
            ));
        }
    }
}

#[test]
fn empty_sparse_selection_requires_every_acknowledgment_without_fabricating_routes() {
    let plan = plan(true);
    let mut ledger = CaptureLedger::new(&plan);
    let receipt = receipt(&plan, true, &mut ledger);
    let records = records(&plan, &receipt);
    assert!(records.iter().all(|record| record.fragments.is_empty()));
    let mut quota = ledger
        .reserve_quota(receipt.delivery_usage().unwrap())
        .unwrap();
    let mut delivery = receipt.into_delivery();
    for record in records {
        delivery
            .receive(
                record.producer_rank,
                &serde_json::to_vec(&record).unwrap(),
                &mut quota,
            )
            .unwrap();
    }
    let result = delivery.finish(&mut quota).unwrap();
    assert_eq!(
        result.capture().record().source_dtype,
        Some(TensorDtype::F32)
    );
    let Some(CapturePayload::RoutedUnits(payload)) = &result.capture().record().payload else {
        panic!("sparse")
    };
    assert!(payload.rows.is_empty());
    assert!(result.capture().contributions().is_empty());
    assert_eq!(result.producers().len(), 7);
}

mod funded;
