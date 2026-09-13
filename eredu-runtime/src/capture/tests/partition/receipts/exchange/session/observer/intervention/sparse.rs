use super::*;
use crate::RoutedUnitObserver;
mod preflight;
mod provider;
use crate::intervention::PartitionRoutedActivationMember;
use eredu_core::component::RoutedComponentCoordinateMap;
const GEOMETRY: RoutedUnitGeometry = RoutedUnitGeometry {
    experts: 5,
    units_per_expert: 7,
    routes_per_token: 2,
};

struct SparseLayout(Vec<RoutedUnitCaptureOwnership>);
impl SparseLayout {
    fn new() -> Self {
        let mut owners = vec![];
        for experts in [vec![4], vec![1], vec![0, 3, 2]] {
            for units in [vec![5, 0, 3], vec![6, 1, 2, 4]] {
                owners.push(RoutedUnitCaptureOwnership {
                    coordinates: RoutedComponentCoordinateMap::new(
                        ComponentCoordinateMap::indices(5, experts.clone()).unwrap(),
                        ComponentCoordinateMap::indices(7, units).unwrap(),
                    ),
                    source_peer: Some(0),
                    source_peers: 3,
                });
            }
        }
        owners.push(RoutedUnitCaptureOwnership {
            coordinates: RoutedComponentCoordinateMap::new(
                ComponentCoordinateMap::range(5, 0..0).unwrap(),
                ComponentCoordinateMap::range(7, 0..7).unwrap(),
            ),
            source_peer: Some(0),
            source_peers: 3,
        });
        owners.push(owners[2].clone());
        Self(owners)
    }
}
impl PartitionCaptureLayout for SparseLayout {
    fn capture_placement(
        &self,
        _: &AdmittedCapturePlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        panic!("no capture selection")
    }
}
impl PartitionActivationLayout for SparseLayout {
    fn activation_members<'a>(
        &'a self,
        _: &'a AdmittedInterventionPlan,
        _: usize,
        _: CapturePhase,
        _: u64,
        _: usize,
        _: usize,
    ) -> Result<Vec<PartitionActivationMember<'a>>, CaptureError> {
        panic!("sparse operation must not enter dense projection")
    }
    fn routed_activation_members<'a>(
        &'a self,
        _: &AdmittedInterventionPlan,
        _: usize,
        max: usize,
    ) -> Result<Vec<PartitionRoutedActivationMember<'a>>, CaptureError> {
        assert!(max >= self.0.len());
        Ok(self
            .0
            .iter()
            .enumerate()
            .map(|(rank, ownership)| PartitionRoutedActivationMember {
                rank,
                ownership,
                input_width: 4,
            })
            .collect())
    }
}
fn admission(rank: usize, all_keep: bool) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    admission_at(rank, all_keep, None)
}
fn admission_at(
    rank: usize,
    all_keep: bool,
    bounds: Option<CaptureInvocationBounds>,
) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    let original = plan_for(CaptureTransform::Slice, false);
    let catalog = discovery(&original);
    let mut capture = original.plan().clone();
    capture.selections.clear();
    capture.limits.per_step = CaptureUsage {
        captures: 1000,
        retained_bytes: 1 << 29,
        host_bytes: 1 << 29,
        encoded_bytes: 1 << 29,
    };
    capture.limits.cumulative = capture.limits.per_step;
    let capture = match bounds {
        Some(bounds) => capture.admit_invocations(
            &catalog.catalog,
            &catalog.support,
            &catalog.support.capture,
            bounds,
        ),
        None => capture.admit(
            &catalog.catalog,
            &catalog.support,
            &catalog.support.capture,
            original.request(),
        ),
    }
    .unwrap();
    let point = InterventionPoint {
        path: "experts.units".into(),
        node_id: "bank".into(),
        stage: InterventionStage::Activation,
        axes: vec![
            TensorAxis {
                name: "token".into(),
                dimension: SymbolicDimension::TokenRows,
            },
            TensorAxis {
                name: "component".into(),
                dimension: SymbolicDimension::Known(35),
            },
        ],
        dtypes: vec![InterventionDtype::Float32],
        operations: vec![InterventionKind::MaskComponents, InterventionKind::Scale],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routing: None,
        routed_units: Some(RoutedUnitInterventionPoint {
            routing: "experts".into(),
            geometry: GEOMETRY,
        }),
    };
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "artifact-exact".into(),
        session_identity: Some(format!("session-{rank}")),
        points: vec![point],
    };
    let actions = if all_keep {
        vec![InterventionAction::MaskComponents {
            dtype: InterventionDtype::Float32,
            indices: (0..35).collect(),
            keep_selected: true,
        }]
    } else {
        vec![
            InterventionAction::MaskComponents {
                dtype: InterventionDtype::Float32,
                indices: (0..5)
                    .flat_map(|expert| [expert * 7, expert * 7 + 5])
                    .collect(),
                keep_selected: false,
            },
            InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: 0.5,
            },
        ]
    };
    let plan = InterventionPlan {
        schema_version: 1,
        operations: actions
            .into_iter()
            .enumerate()
            .map(|(i, action)| InterventionOperation {
                id: format!("edit-{i}"),
                target: "experts.units".into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![CaptureSlice {
                    axis: "token".into(),
                    start: 0,
                    end: 3,
                    stride: 2,
                }],
                action,
                evidence: InterventionEvidence::None,
            })
            .collect(),
    };
    let plan = match bounds {
        Some(bounds) => plan.admit_invocations(&discovery, bounds, "trial"),
        None => plan.admit(&discovery, capture.request(), "trial"),
    }
    .unwrap();
    (capture, plan)
}
fn invoke(
    observer: &mut dyn crate::RoutedUnitObserver<Value>,
    owned: &RoutedUnitCaptureOwnership,
    all_keep: bool,
    fault: Option<&str>,
) -> Result<(), eredu_nn::Error> {
    let mut tags = vec![];
    let mut experts = vec![];
    for token in 0..3 {
        for slot in 0..2 {
            let expert = if token == 0 { 1 } else { (token + slot) % 4 };
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
    let origins = crate::RoutedUnitOrigins::new(&counts, &tags, 2).unwrap();
    let rows = tags.len() as u64;
    let input = value(
        vec![rows, if fault == Some("width") { 5 } else { 4 }],
        vec![],
    );
    let units = owned.coordinates.units();
    crate::with_routed_unit_invocation(
        Some(observer),
        crate::RoutedUnitInvocation {
            input: &input,
            origins: Some(origins),
            unit_coordinates: Some(units),
        },
        |observer| {
            let observer = observer.unwrap();
            let groups = Value {
                shape: vec![rows, 1],
                data: TensorObservationData::U64(experts),
            };
            for start in (0..rows).step_by(2) {
                let end = (start + 2).min(rows);
                if fault == Some("missing") && end == rows {
                    break;
                }
                let values = value(
                    vec![end - start, units.local_count() as u64],
                    (start..end)
                        .flat_map(|native| {
                            let origin = origins.resolve(native as usize).unwrap();
                            (0..units.local_count()).map(move |unit| {
                                3. + origin.source_peer.unwrap() as f32 * 1000.
                                    + origin.token as f32 * 10.
                                    + origin.slot as f32
                                    + units.local_to_global(unit).unwrap() as f32 * 0.125
                            })
                        })
                        .collect(),
                );
                let coefficients = value(vec![end - start, 1], vec![]);
                let batch = crate::RoutedUnitBatch {
                    units: eredu_nn::GroupedUnitBatch {
                        values: &values,
                        group_indices: &groups,
                        token_indices: &groups,
                        selection_indices: &groups,
                        coefficients: &coefficients,
                        token_offset: 0,
                        total_token_count: rows as usize,
                        group_count: 5,
                    },
                    source_groups: &groups,
                    global_groups: None,
                    provider_token_offset: if fault == Some("repeat") && start > 0 {
                        0
                    } else {
                        start as usize
                    },
                    origins: Some(origins),
                    unit_coordinates: Some(units),
                };
                let effective = observer.intervene(&batch)?;
                let actual = effective.as_ref().unwrap_or(&values);
                for (row, values) in data(actual).chunks(units.local_count()).enumerate() {
                    let origin = origins.resolve(start as usize + row).unwrap();
                    for (unit, actual) in values.iter().enumerate() {
                        let global = units.local_to_global(unit).unwrap();
                        let original = data(&batch.units.values)[row * units.local_count() + unit];
                        let expected = if all_keep || origin.token == 1 {
                            original
                        } else if [0, 5].contains(&global) {
                            0.
                        } else {
                            original * 0.5
                        };
                        assert_eq!(
                            *actual, expected,
                            "all peers, including source 2, use the actual edited values"
                        );
                    }
                }
            }
            Ok(())
        },
        |error| error,
    )
}

#[test]
fn shared_sparse_operations_cover_replicas_idle_owners_strides_counts_and_failures() {
    let mut usage = None;
    for (all_keep, committed, fault, failed_rank) in [
        (false, true, None, 2),
        (true, true, None, 2),
        (false, false, None, 2),
        (false, false, Some("width"), 0),
        (false, false, Some("width"), 2),
        (false, false, Some("native"), 2),
        (false, false, Some("missing"), 2),
        (false, false, Some("repeat"), 2),
    ] {
        let all = world(9);
        let active = world(8);
        let outputs = std::thread::scope(|scope| {
            (0..9).map(|rank| {
            let all = Arc::clone(&all); let active = Arc::clone(&active);
            scope.spawn(move || {
                let transport = HookTransport { transport: transport(all, rank, Fault::None), hook: (rank < 8).then(|| transport(active, rank, Fault::None)), members: (0..8).collect() };
                let (capture, plan) = admission(rank, all_keep);
                let mut session = configured(capture, 9);
                session.enable_interventions(plan, Arc::new(Estimates)).unwrap();
                let layout = SparseLayout::new();
                let epoch = DistributedCommitEpoch::FIRST;
                let limits = PartitionCaptureReceiptLimits { max_producers: 9, ..LIMITS };
                let mut observer = PartitionCaptureObserver::for_step(&mut session, Backend { fail: fault == Some("native") && rank == failed_rank, ..Default::default() }, &transport, &layout, 0, limits, estimate, |error| error).with_interventions();
                observer.prepare_transaction(epoch, crate::ExpertPass::Prefill).unwrap();
                observer.coordinate_transaction(epoch).unwrap();
                if rank < 8 {
                    let result = invoke(observer.routed_unit_observer("experts").unwrap().unwrap(), &layout.0[rank], all_keep, if rank == failed_rank { fault } else { None });
                    assert_eq!(result.is_ok(), fault.is_none(), "rank {rank} {fault:?}: {result:?}");
                    assert!(!observer.invocation_active());
                    assert_eq!(transport.hook.as_ref().unwrap().calls.load(Ordering::SeqCst), if all_keep { 2 } else { 4 }, "every source and final operation vote settles before reverse exchange");
                }
                let delivery = observer.complete_transaction(epoch);
                assert_eq!(delivery.is_ok(), fault.is_none(), "{fault:?}: {delivery:?}");
                observer.finish_transaction(epoch, committed);
                drop(observer);
                let step = session.take_step().unwrap();
                assert!(step.records.is_empty() && step.partitions.is_empty());
                if committed {
                    for (index, record) in step.interventions.iter().enumerate() {
                        assert_eq!(record.outcome, if all_keep { InterventionOutcome::Unmatched } else { InterventionOutcome::Applied });
                        let receipt = record.routed_units.unwrap();
                        assert_eq!((receipt.source_tokens, receipt.completed_tokens), (3, 3));
                        assert_eq!(receipt.affected_values, if all_keep { 0 } else if index == 0 { 8 } else { 28 });
                    }
                } else {
                    assert!(step.interventions.iter().all(|record| !matches!(record.outcome, InterventionOutcome::Applied | InterventionOutcome::Unmatched)));
                }
                step.cumulative_usage
            })
        }).collect::<Vec<_>>().into_iter().map(|worker| worker.join().unwrap()).collect::<Vec<_>>()
        });
        assert!(outputs.iter().all(|n| *n == outputs[0]));
        if !all_keep {
            if let Some(usage) = usage {
                assert_eq!(outputs[0], usage, "failure timing never refunds work");
            } else {
                usage = Some(outputs[0]);
            }
        }
    }
}
