use super::super::super::routed as fixture;
use super::*;

#[derive(Default)]
struct SparseBackend {
    calls: usize,
    corrupt: bool,
}
impl CaptureBackend for SparseBackend {
    type Tensor = Value;
    type Error = std::io::Error;
    fn shape(&self, value: &Value) -> Result<Vec<u64>, Self::Error> {
        Ok(value.shape.clone())
    }
    fn source_dtype(&self, value: &Value) -> Option<TensorDtype> {
        Backend::default().source_dtype(value)
    }
    fn estimate(
        &self,
        _: &Value,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        panic!("sparse work must use its own native recipe")
    }
    fn transform(
        &mut self,
        _: &Value,
        _: &CaptureSelection,
        _: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Self::Error> {
        panic!("sparse work must use the actual routed source")
    }
    fn estimate_partition_routed_units(
        &self,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        request.validate()?;
        Ok(CaptureUsage {
            captures: 1,
            retained_bytes: 4096,
            host_bytes: 16384,
            encoded_bytes: 16384,
        })
    }
    fn capture_partition_routed_units(
        &mut self,
        source: &PartitionRoutedUnitCaptureSource<'_, Value>,
        request: &PartitionRoutedUnitCaptureRequest<'_>,
    ) -> Option<Result<RoutedUnitCapture, Self::Error>> {
        self.calls += 1;
        let TensorObservationData::U64(groups) = &source.source.source_groups.data else {
            panic!("groups")
        };
        let TensorObservationData::F32(values) = &source.source.values.data else {
            panic!("values")
        };
        let count = source.source.coefficients.shape[0];
        let mut rows = Vec::new();
        for local in (0..count).rev() {
            let native = source.source.token_offset + local;
            let origin = source.origins.unwrap().resolve(native as usize).unwrap();
            let slice = request.slice;
            let token = origin.token as u64;
            let slot = origin.slot as u64;
            if origin.source_peer.map(|peer| peer as u64) != request.ownership.source_peer
                || token < slice.starts[0]
                || token >= slice.ends[0]
                || !(token - slice.starts[0]).is_multiple_of(slice.strides[0])
            {
                continue;
            }
            let data = (0..slice.shape[2])
                .map(|unit| {
                    let global = slice.starts[2] + unit * slice.strides[2];
                    let column = source
                        .unit_coordinates
                        .global_to_local(global as usize)
                        .unwrap();
                    values[local as usize * source.unit_coordinates.local_count() + column]
                })
                .collect::<Vec<_>>();
            rows.push(RoutedUnitCaptureRow {
                source_peer: Some(0),
                token,
                slot,
                expert: groups[native as usize],
                coefficient: if slot == 0 { 0.75 } else { 0.25 },
                unit_start: slice.starts[2],
                unit_stride: slice.strides[2],
                values: TensorObservation::new(vec![data.len()], TensorObservationData::F32(data))
                    .unwrap(),
            });
        }
        Some(Ok(RoutedUnitCapture {
            geometry: request.geometry,
            source_token_ranges: vec![[
                source.source.token_offset,
                source.source.token_offset + count + u64::from(self.corrupt),
            ]],
            rows,
        }))
    }
}

fn scalar(token: usize, slot: usize, unit: usize) -> f32 {
    (token * 100 + slot * 10 + unit) as f32 * 0.125 - 3.25
}

fn singleton(plan: &AdmittedCapturePlan) -> Vec<PartitionRoutedCaptureProducer> {
    let units = ComponentCoordinateMap::range(7, 0..7).unwrap();
    let slice = resolve_slice(&plan.points()[0], &plan.plan().selections[0], &[3, 2, 7]).unwrap();
    vec![PartitionRoutedCaptureProducer {
        rank: 0,
        projection: CaptureSlicePartition::new(&[3, 2, 7], &slice, 2, &units, 4).unwrap(),
        ownership: RoutedUnitCaptureOwnership {
            coordinates: eredu_core::component::RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(5, 0..5).unwrap(),
                units,
            ),
            source_peer: Some(0),
            source_peers: 3,
        },
    }]
}

#[test]
fn live_sparse_start_checks_owner_coordination_maps_rows_and_dtype_before_collection() {
    for invalid in ["units", "rows", "dtype", "origins"] {
        let plan = fixture::plan(false);
        let producers = singleton(&plan);
        let units = producers[0].ownership.coordinates.units().clone();
        let mut capture = configured(plan.clone(), 1);
        let mut foreign = configured(plan, 1);
        let epoch = DistributedCommitEpoch::FIRST;
        capture
            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
            .unwrap();
        foreign
            .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
            .unwrap();
        let transport = transport(world(1), 0, Fault::None);
        let backend = SparseBackend::default();
        let mut work = capture
            .prepare_partition_routed_capture(
                &transport,
                0,
                producers,
                fixture::limits(),
                |request| backend.estimate_partition_routed_units(request),
            )
            .unwrap();
        let origins =
            eredu_core::capture::RoutedUnitOrigins::new(&[6, 0, 0], &[0, 1, 2, 3, 4, 5], 2)
                .unwrap();
        assert!(
            capture
                .begin_partition_routed_capture(
                    &mut work,
                    6,
                    TensorDtype::F32,
                    &units,
                    Some(origins)
                )
                .is_err(),
            "uncoordinated authority"
        );
        assert!(
            foreign
                .begin_partition_routed_capture(
                    &mut work,
                    6,
                    TensorDtype::F32,
                    &units,
                    Some(origins)
                )
                .is_err(),
            "foreign owner"
        );
        let coordination = capture.prepare_partition_coordination(&transport).unwrap();
        capture.coordinate_partition_capture(coordination).unwrap();
        let prepaid = capture.cumulative_usage();
        let wrong_units = ComponentCoordinateMap::indices(7, vec![6, 5, 4, 3, 2, 1, 0]).unwrap();
        assert!(capture
            .begin_partition_routed_capture(
                &mut work,
                if invalid == "rows" { 5 } else { 6 },
                if invalid == "dtype" {
                    TensorDtype::I64
                } else {
                    TensorDtype::F32
                },
                if invalid == "units" {
                    &wrong_units
                } else {
                    &units
                },
                if invalid == "origins" {
                    None
                } else {
                    Some(origins)
                },
            )
            .is_err());
        assert!(capture
            .finish_partition_routed_capture(&mut work, true)
            .is_err());
        assert!(capture.complete_partition_capture(work).is_err());
        assert_eq!(capture.cumulative_usage(), prepaid);
        assert_eq!(backend.calls, 0);
        capture.finish_transaction(epoch, false);
        assert!(capture.take_step().unwrap().records[0].payload.is_none());
    }
}

#[test]
fn live_sparse_native_budget_exhaustion_rejects_before_invocation() {
    let plan = fixture::plan(false);
    let producers = singleton(&plan);
    let limit = plan.plan().limits.per_step.retained_bytes;
    let mut capture = configured(plan, 1);
    capture
        .prepare_step_transaction(DistributedCommitEpoch::FIRST, crate::ExpertPass::Prefill, 0)
        .unwrap();
    capture
        .ledger
        .reserve(CaptureUsage {
            retained_bytes: limit,
            ..Default::default()
        })
        .unwrap();
    let transport = transport(world(1), 0, Fault::None);
    let backend = SparseBackend::default();
    assert!(matches!(
        capture.prepare_partition_routed_capture(
            &transport,
            0,
            producers,
            fixture::limits(),
            |request| backend.estimate_partition_routed_units(request)
        ),
        Err(PartitionCaptureExchangeError::Capture(
            CaptureError::Limit {
                budget: CaptureBudget::Retention,
                ..
            }
        ))
    ));
    assert_eq!(backend.calls, 0);
    assert_eq!(transport.calls.load(Ordering::SeqCst), 0);
    assert_eq!(capture.cumulative_usage().retained_bytes, limit);
}

fn collect(
    session: &mut CaptureSession,
    work: &mut SessionPartitionCapture<'_, Transport>,
    owned: &RoutedUnitCaptureOwnership,
    fault: Option<&str>,
) -> Result<usize, CaptureExecutionError<std::io::Error>> {
    let mut tags = vec![];
    let mut experts = vec![];
    for token in 0..3usize {
        for slot in 0..2 {
            let expert = if token == 0 { 1 } else { (token + slot) % 5 };
            if owned
                .coordinates
                .experts()
                .global_to_local(expert)
                .is_some()
            {
                tags.push(token * 2 + slot);
                experts.push(expert as u64);
            }
        }
    }
    let count = tags.len();
    tags.extend_from_within(..);
    experts.extend_from_within(..);
    let counts = [count, 0, count];
    let origins = eredu_core::capture::RoutedUnitOrigins::new(&counts, &tags, 2).unwrap();
    let rows = tags.len() as u64;
    if fault == Some("missing invocation") {
        return Ok(0);
    }
    session.begin_partition_routed_capture(
        work,
        rows,
        TensorDtype::F32,
        owned.coordinates.units(),
        Some(origins),
    )?;
    let groups = Value {
        shape: vec![rows, 1],
        data: TensorObservationData::U64(experts),
    };
    let mut backend = SparseBackend {
        corrupt: fault == Some("corrupt chunk"),
        ..Default::default()
    };
    let units = owned.coordinates.units();
    for start in (0..rows).step_by(2) {
        let end = (start + 2).min(rows);
        if fault == Some("missing chunk") && end == rows {
            break;
        }
        let values = Value {
            shape: vec![end - start, units.local_count() as u64],
            data: TensorObservationData::F32(
                (start..end)
                    .flat_map(|row| {
                        let origin = origins.resolve(row as usize).unwrap();
                        (0..units.local_count()).map(move |unit| {
                            scalar(
                                origin.token,
                                origin.slot,
                                units.local_to_global(unit).unwrap(),
                            ) + origin.source_peer.unwrap() as f32 * 1000.
                        })
                    })
                    .collect(),
            ),
        };
        let coefficients = Value {
            shape: vec![end - start, 1],
            data: TensorObservationData::F32(vec![]),
        };
        let source = PartitionRoutedUnitCaptureSource {
            source: RoutedUnitCaptureSource {
                values: &values,
                token_indices: &groups,
                selection_indices: &groups,
                coefficients: &coefficients,
                source_groups: &groups,
                token_offset: start,
                global_groups: None,
            },
            origins: Some(origins),
            unit_coordinates: units,
        };
        session.observe_partition_routed_units(work, &mut backend, &source)?;
    }
    session.finish_partition_routed_capture(work, fault != Some("provider failure"))?;
    if fault == Some("duplicate completion") {
        session.finish_partition_routed_capture(work, true)?;
    }
    Ok(backend.calls)
}

#[test]
fn live_sparse_chunks_share_quotas_delivery_and_commit_with_idle_and_inactive_ranks() {
    for (empty, committed, fault) in [
        (false, true, None),
        (true, true, None),
        (false, false, None),
        (false, false, Some("missing invocation")),
        (false, false, Some("missing chunk")),
        (false, false, Some("corrupt chunk")),
        (false, false, Some("provider failure")),
        (false, false, Some("duplicate completion")),
    ] {
        let world = world(8);
        let threads = (0..8)
            .map(|rank| {
                let world = Arc::clone(&world);
                std::thread::spawn(move || {
                    let plan = fixture::plan(empty);
                    let declarations = fixture::producers(&plan, true);
                    let owned = declarations
                        .get(rank)
                        .map(|producer| producer.ownership.clone());
                    let mut session = configured(plan, 8);
                    let epoch = DistributedCommitEpoch::FIRST;
                    let transport = transport(world, rank, Fault::None);
                    session
                        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
                        .unwrap();
                    let backend = SparseBackend::default();
                    let mut work = session
                        .prepare_partition_routed_capture(
                            &transport,
                            0,
                            declarations,
                            fixture::limits(),
                            |request| backend.estimate_partition_routed_units(request),
                        )
                        .unwrap();
                    let reserved = work.global_reserved();
                    let coordination = session.prepare_partition_coordination(&transport).unwrap();
                    session.coordinate_partition_capture(coordination).unwrap();
                    let prepaid = session.cumulative_usage();
                    let collected = owned.as_ref().map(|owned| {
                        collect(
                            &mut session,
                            &mut work,
                            owned,
                            if rank == 2 { fault } else { None },
                        )
                    });
                    if fault.is_none() {
                        let calls = collected.transpose().unwrap().unwrap_or(0);
                        if empty || !(2..6).contains(&rank) {
                            assert_eq!(calls, 0);
                        }
                    }
                    assert_eq!(session.cumulative_usage(), prepaid);
                    assert!(session.take_step().is_none());
                    let delivered = session.complete_partition_capture(work);
                    assert_eq!(delivered.is_ok(), fault.is_none());
                    assert!(
                        session.take_step().is_none(),
                        "delivery does not commit the forward"
                    );
                    if delivered.is_ok() {
                        session.complete_transaction(epoch).unwrap();
                    }
                    session.finish_transaction(epoch, committed);
                    let step = session.take_step().unwrap();
                    assert_eq!(step.cumulative_usage, prepaid);
                    if committed {
                        assert_eq!(step.records[0].outcome, CaptureOutcome::Captured);
                        assert_eq!(step.partitions[0].producers, [0, 1, 2, 3, 4, 5, 6]);
                        let Some(CapturePayload::RoutedUnits(payload)) = &step.records[0].payload
                        else {
                            panic!("sparse")
                        };
                        assert_eq!(payload.rows.len(), if empty { 0 } else { 4 });
                        for row in &payload.rows {
                            assert_eq!(row.source_peer, Some(0));
                            let TensorObservationData::F32(values) = row.values.data() else {
                                panic!("values")
                            };
                            assert_eq!(
                                *values,
                                [1, 3, 5].map(|unit| scalar(
                                    row.token as usize,
                                    row.slot as usize,
                                    unit
                                ))
                            );
                        }
                        assert!(payload.source_token_ranges.is_empty());
                        for contribution in &step.partitions[0].contributions {
                            let provenance = contribution.routed.as_ref().unwrap();
                            if contribution.producer_rank < 2 {
                                assert!(provenance.source_token_ranges.is_empty());
                            }
                        }
                    } else {
                        assert!(step.records[0].payload.is_none());
                        assert!(step.partitions.is_empty());
                    }
                    (prepaid, reserved)
                })
            })
            .collect::<Vec<_>>();
        let result = threads
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>();
        assert!(result.iter().all(|value| *value == result[0]));
    }
}

mod observer;
