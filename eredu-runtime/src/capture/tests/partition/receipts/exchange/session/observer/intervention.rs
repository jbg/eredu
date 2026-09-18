use super::*;
mod invocation;
mod sparse;
use crate::intervention::{
    PartitionActivationLayout, PartitionActivationMember, PartitionActivationProjection,
};
use eredu_core::intervention::*;

struct EditLayout(Vec<(usize, ComponentCoordinateMap)>);
impl PartitionActivationLayout for EditLayout {
    fn activation_members<'a>(
        &'a self,
        plan: &'a AdmittedInterventionPlan,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        max_members: usize,
        max_regions: usize,
    ) -> Result<Vec<PartitionActivationMember<'a>>, CaptureError> {
        self.activation_members_at(
            plan,
            operation,
            phase,
            prediction,
            None,
            max_members,
            max_regions,
        )
    }
    fn activation_members_at<'a>(
        &'a self,
        plan: &'a AdmittedInterventionPlan,
        operation: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        max_members: usize,
        max_regions: usize,
    ) -> Result<Vec<PartitionActivationMember<'a>>, CaptureError> {
        let shape = plan
            .geometry_at(phase, prediction, invocation)?
            .resolve(&plan.points()[operation].observation_geometry())?
            .unwrap();
        assert!(self.0.len() <= max_members);
        self.0
            .iter()
            .map(|(rank, coordinates)| {
                Ok(PartitionActivationMember {
                    rank: *rank,
                    projection: PartitionActivationProjection::new_at(
                        plan,
                        operation,
                        phase,
                        prediction,
                        invocation,
                        &shape,
                        1,
                        coordinates,
                        max_regions,
                    )?,
                })
            })
            .collect()
    }
}
impl PartitionCaptureLayout for EditLayout {
    fn capture_placement(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        self.capture_placement_at(plan, index, phase, prediction, None, limits)
    }
    fn capture_placement_at(
        &self,
        plan: &AdmittedCapturePlan,
        index: usize,
        phase: CapturePhase,
        prediction: u64,
        invocation: Option<CaptureInvocationShape>,
        limits: PartitionCaptureReceiptLimits,
    ) -> Result<PartitionCapturePlacement, CaptureError> {
        let shape = plan
            .geometry_at(phase, prediction, invocation)?
            .resolve(&plan.points()[index])?
            .unwrap();
        let slice = resolve_slice(
            &plan.points()[index],
            &plan.plan().selections[index],
            &shape,
        )?;
        Ok(PartitionCapturePlacement {
            // Rank 3 executes a replica. Rank 4 has an actual empty local axis.
            producers: self
                .0
                .iter()
                .filter(|(rank, _)| *rank != 3)
                .map(|(rank, map)| {
                    Ok(PartitionCaptureProducer {
                        rank: *rank,
                        projection: CaptureSlicePartition::new(
                            &shape,
                            &slice,
                            1,
                            map,
                            limits.max_fragments,
                        )?,
                    })
                })
                .collect::<Result<_, CaptureError>>()?,
            hook_members: self.0.iter().map(|(rank, _)| *rank).collect(),
            source_shapes: self
                .0
                .iter()
                .map(|(_, map)| vec![shape[0], map.local_count() as u64])
                .collect(),
        })
    }
}
fn layout() -> EditLayout {
    EditLayout(vec![
        (
            0,
            ComponentCoordinateMap::indices(20, vec![7, 1, 4, 2, 0, 3, 5, 6]).unwrap(),
        ),
        (2, ComponentCoordinateMap::range(20, 8..20).unwrap()),
        (
            3,
            ComponentCoordinateMap::indices(20, vec![7, 1, 4, 2, 0, 3, 5, 6]).unwrap(),
        ),
        (4, ComponentCoordinateMap::range(20, 20..20).unwrap()),
    ])
}
fn data(value: &Value) -> &[f32] {
    match &value.data {
        TensorObservationData::F32(data) => data,
        _ => panic!("float fixture"),
    }
}
fn value(shape: Vec<u64>, data: Vec<f32>) -> Value {
    Value {
        shape,
        data: TensorObservationData::F32(data),
    }
}
impl InterventionBackend for Backend {
    fn partition_routed_unit_locations(
        &mut self,
        source: &PartitionRoutedUnitCaptureSource<'_, Value>,
        _: RoutedUnitGeometry,
    ) -> Option<Result<RoutedUnitLocations, Self::Error>> {
        let TensorObservationData::U64(groups) = &source.source.source_groups.data else {
            panic!("groups")
        };
        let start = source.source.token_offset;
        let end = start + source.source.coefficients.shape[0];
        Some(Ok(RoutedUnitLocations {
            source_token_range: [start, end],
            rows: (start..end)
                .map(|native| {
                    let origin = source.origins.unwrap().resolve(native as usize).unwrap();
                    RoutedUnitLocation {
                        source_peer: origin.source_peer.map(|p| p as u64),
                        token: origin.token as u64,
                        slot: origin.slot as u64,
                        expert: groups[native as usize],
                    }
                })
                .collect(),
        }))
    }
    fn select_elements(
        &mut self,
        source: &Value,
        indices: &[u64],
    ) -> Option<Result<Value, Self::Error>> {
        if self.fail {
            return Some(Err(std::io::Error::other(
                "injected native sparse edit failure",
            )));
        }
        Some(Ok(value(
            vec![indices.len() as u64],
            indices.iter().map(|i| data(source)[*i as usize]).collect(),
        )))
    }
    fn update_elements(
        &mut self,
        source: &Value,
        indices: &[u64],
        replacement: &Value,
    ) -> Option<Result<Value, Self::Error>> {
        let mut output = data(source).to_vec();
        for (index, value) in indices.iter().zip(data(replacement)) {
            output[*index as usize] = *value;
        }
        Some(Ok(value(source.shape.clone(), output)))
    }

    fn intervention_dtype(&self, _: &Value) -> Result<InterventionDtype, Self::Error> {
        Ok(InterventionDtype::Float32)
    }
    fn validate_intervention_geometry(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        Ok(())
    }
    fn select_region(&mut self, v: &Value, s: &ResolvedCaptureSlice) -> Result<Value, Self::Error> {
        if self.fail {
            return Err(std::io::Error::other(
                "injected native intervention failure",
            ));
        }
        Ok(value(
            s.shape.clone(),
            selected_indices(&v.shape, s)
                .iter()
                .map(|i| data(v)[*i])
                .collect(),
        ))
    }
    fn update_region(
        &mut self,
        v: &Value,
        s: &ResolvedCaptureSlice,
        replacement: &Value,
    ) -> Result<Value, Self::Error> {
        let mut output = data(v).to_vec();
        for (index, replacement) in selected_indices(&v.shape, s)
            .into_iter()
            .zip(data(replacement))
        {
            output[index] = *replacement;
        }
        Ok(value(v.shape.clone(), output))
    }
    fn zeros(&mut self, shape: &[u64], _: InterventionDtype) -> Result<Value, Self::Error> {
        Ok(value(
            shape.to_vec(),
            vec![0.; elements(shape).unwrap() as usize],
        ))
    }
    fn scale(&mut self, v: &Value, factor: f32) -> Result<Value, Self::Error> {
        Ok(value(
            v.shape.clone(),
            data(v).iter().map(|v| v * factor).collect(),
        ))
    }
    fn fill_masked(&mut self, v: &Value, keep: &[bool], fill: f32) -> Result<Value, Self::Error> {
        Ok(value(
            v.shape.clone(),
            data(v)
                .iter()
                .zip(keep)
                .map(|(v, k)| if *k { *v } else { fill })
                .collect(),
        ))
    }
    fn mask_components(
        &mut self,
        v: &Value,
        ids: &[u32],
        keep: bool,
    ) -> Result<Value, Self::Error> {
        let width = *v.shape.last().unwrap() as usize;
        Ok(value(
            v.shape.clone(),
            data(v)
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if ids.contains(&((i % width) as u32)) == keep {
                        *v
                    } else {
                        0.
                    }
                })
                .collect(),
        ))
    }
    fn realize_tensor(&mut self, v: &InterventionTensor) -> Result<Value, Self::Error> {
        let InterventionValues::Float32(data) = &v.values else {
            panic!("float fixture")
        };
        Ok(value(v.shape.clone(), data.clone()))
    }
    fn add(&mut self, a: &Value, b: &Value) -> Result<Value, Self::Error> {
        Ok(value(
            a.shape.clone(),
            data(a).iter().zip(data(b)).map(|(a, b)| a + b).collect(),
        ))
    }
    fn fill_columns(&mut self, v: &Value, ids: &[u32], fill: f32) -> Result<Value, Self::Error> {
        let width = *v.shape.last().unwrap() as usize;
        Ok(value(
            v.shape.clone(),
            data(v)
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if ids.contains(&((i % width) as u32)) {
                        fill
                    } else {
                        *v
                    }
                })
                .collect(),
        ))
    }
}
struct Estimates;
impl InterventionEstimator for Estimates {
    fn partition_routed_unit_usage(
        &self,
        geometry: RoutedUnitGeometry,
        tokens: u64,
        owned: &RoutedUnitCaptureOwnership,
        _: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        let rows = owned.maximum_source_rows(tokens, geometry.routes_per_token)?;
        Ok(CaptureUsage {
            retained_bytes: rows * owned.coordinates.units().local_count() as u64 * 32 + 1024,
            host_bytes: rows * 256 + 1024,
            ..Default::default()
        })
    }
    fn routed_unit_usage(
        &self,
        _: RoutedUnitGeometry,
        _: &[u64],
        _: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: 16384,
            host_bytes: 16384,
            ..Default::default()
        })
    }

    fn validate_geometry(&self, _: &[u64], _: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        Ok(())
    }
    fn activation_usage(
        &self,
        shape: &[u64],
        slice: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: mul(add(elements(shape)?, mul(elements(&slice.shape)?, 3)?)?, 8)?,
            host_bytes: 256,
            ..Default::default()
        })
    }
    fn capture_usage(
        &self,
        shape: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(estimate(shape, selection, slice)?.capture)
    }
    fn original_route_usage(
        &self,
        _: &InterventionRoutingPolicy,
        _: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        panic!("activation only")
    }
}
fn plans(
    rank: usize,
    evidence: InterventionEvidence,
    different: bool,
) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    plans_at(rank, evidence, different, None)
}
fn plans_at(
    rank: usize,
    evidence: InterventionEvidence,
    different: bool,
    bounds: Option<CaptureInvocationBounds>,
) -> (AdmittedCapturePlan, AdmittedInterventionPlan) {
    let original = plan_for(CaptureTransform::Slice, false);
    let mut discovery = discovery(&original);
    discovery.catalog.points[0]
        .axes
        .as_mut()
        .unwrap()
        .last_mut()
        .unwrap()
        .name = "component".into();
    let mut capture = original.plan().clone();
    // Include ordinary pre-intervention capture at the same boundary.
    capture.selections[0].slices.clear();
    capture.selections[0].transform = CaptureTransform::FullTensor;
    capture.limits.per_step = CaptureUsage {
        captures: 1000,
        retained_bytes: 1 << 29,
        host_bytes: 2 << 30,
        encoded_bytes: 1 << 29,
    };
    capture.limits.cumulative = capture.limits.per_step.checked_mul(8).unwrap();
    let mut caps = discovery.support.capture.clone();
    caps.transformations.push(CaptureTransformKind::FullTensor);
    if bounds.is_some() {
        capture.selections[0].schedule = CaptureSchedule::default();
    }
    let capture = match bounds {
        Some(bounds) => {
            capture.admit_invocations(&discovery.catalog, &discovery.support, &caps, bounds)
        }
        None => capture.admit(
            &discovery.catalog,
            &discovery.support,
            &caps,
            original.request(),
        ),
    }
    .unwrap();
    let point = InterventionPoint {
        path: "block.output".into(),
        node_id: "block".into(),
        stage: InterventionStage::Activation,
        axes: capture.points()[0].axes.clone().unwrap(),
        dtypes: vec![InterventionDtype::Float32],
        operations: vec![InterventionKind::Scale, InterventionKind::MaskComponents],
        score_stages: vec![],
        prefill: ObservationSupportStatus::Supported,
        decode: ObservationSupportStatus::Supported,
        conditions: vec![],
        routed_units: None,
        routing: None,
    };
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "artifact-exact".into(),
        session_identity: Some(format!("backend-session-{rank}")),
        points: vec![point],
    };
    let mut operations = vec![
        InterventionOperation {
            id: "keep".into(),
            target: "block.output".into(),
            schedule: CaptureSchedule {
                decode: false,
                ..Default::default()
            },
            slices: vec![CaptureSlice {
                axis: capture.points()[0].axes.as_ref().unwrap()[0].name.clone(),
                start: 1,
                end: 3,
                stride: 1,
            }],
            action: InterventionAction::MaskComponents {
                dtype: InterventionDtype::Float32,
                indices: vec![2, 9, 17],
                keep_selected: true,
            },
            evidence: evidence.clone(),
        },
        InterventionOperation {
            id: "scale".into(),
            target: "block.output".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: if different { 3. } else { 2. },
            },
            evidence: evidence.clone(),
        },
    ];
    if bounds.is_some() {
        operations[0].schedule = CaptureSchedule::default();
        operations[0].slices[0].start = 0;
        operations[0].slices[0].end = 1;
    }
    let intervention = InterventionPlan {
        schema_version: 1,
        operations,
    };
    let intervention = match bounds {
        Some(bounds) => intervention.admit_invocations(&discovery, bounds, &format!("run-{rank}")),
        None => intervention.admit(&discovery, capture.request(), &format!("run-{rank}")),
    }
    .unwrap();
    (capture, intervention)
}

#[test]
fn global_interventions_preserve_order_evidence_replicas_and_inactive_pipeline_members() {
    for evidence in [
        InterventionEvidence::None,
        InterventionEvidence::Preview { max_elements: 60 },
        InterventionEvidence::Summary,
    ] {
        for committed in [false, true] {
            let all = world(5);
            let hooks = world(4);
            let results = std::thread::scope(|scope| {
                (0..5)
                    .map(|rank| {
                        let all = Arc::clone(&all);
                        let hooks = Arc::clone(&hooks);
                        let evidence = evidence.clone();
                        scope.spawn(move || {
                            let members = vec![0, 2, 3, 4];
                            let transport = HookTransport {
                                transport: transport(all, rank, Fault::None),
                                hook: members
                                    .iter()
                                    .position(|r| *r == rank)
                                    .map(|r| transport(hooks, r, Fault::None)),
                                members,
                            };
                            let layout = layout();
                            let (capture, plan) = plans(rank, evidence.clone(), false);
                            let mut session = configured(capture, 5);
                            session
                                .enable_interventions(plan, Arc::new(Estimates))
                                .unwrap();
                            let limits = PartitionCaptureReceiptLimits {
                                max_producers: 5,
                                max_fragments: 64,
                                max_record_bytes: 1 << 16,
                            };
                            let epoch = DistributedCommitEpoch::FIRST;
                            let mut observer = PartitionCaptureObserver::for_step(
                                &mut session,
                                Backend::default(),
                                &transport,
                                &layout,
                                0,
                                limits,
                                estimate,
                                |e| e,
                            )
                            .with_interventions();
                            observer
                                .prepare_transaction(epoch, crate::ExpertPass::Prefill)
                                .unwrap();
                            assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
                            observer.coordinate_transaction(epoch).unwrap();
                            if let Some((_, map)) = layout.0.iter().find(|(r, _)| *r == rank) {
                                let input = local_value(&global(), map);
                                observer.observe("block.output", &input).unwrap();
                                let output = observer.intervene("block.output", &input).unwrap();
                                let mut expected = global();
                                let TensorObservationData::F32(values) = &mut expected.data else {
                                    unreachable!()
                                };
                                for (i, value) in values.iter_mut().enumerate() {
                                    if i >= 20 && ![2, 9, 17].contains(&(i % 20)) {
                                        *value = 0.;
                                    }
                                    *value *= 2.;
                                }
                                assert_eq!(
                                    data(output.as_ref().unwrap_or(&input)),
                                    data(&local_value(&expected, map))
                                );
                                assert_eq!(
                                    data(&input),
                                    data(&local_value(&global(), map)),
                                    "source preserved"
                                );
                            }
                            observer.complete_transaction(epoch).unwrap();
                            // Completed results are still pending until final commit.
                            drop(observer);
                            assert!(session.take_step().is_none());
                            session.finish_transaction(epoch, committed);
                            let step = session.take_step().unwrap();
                            assert_eq!(
                                step.interventions
                                    .iter()
                                    .all(|r| r.outcome == InterventionOutcome::Applied),
                                committed
                            );
                            assert_eq!(
                                step.interventions
                                    .iter()
                                    .all(|r| r.evidence.iter().all(|e| e.payload.is_none())),
                                !committed || evidence == InterventionEvidence::None
                            );
                            assert!(!transport.transport.poison.load(Ordering::SeqCst));
                            step
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|worker| worker.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for step in &results {
                assert_eq!(step.cumulative_usage, results[0].cumulative_usage);
            }
            if committed {
                let (capture, plan) = plans(0, evidence.clone(), false);
                let mut ordinary = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
                ordinary
                    .enable_interventions(plan, Arc::new(Estimates))
                    .unwrap();
                ordinary.begin_step(CapturePhase::Prefill, 0).unwrap();
                ordinary
                    .observe(&mut Backend::default(), "block.output", &global())
                    .unwrap();
                ordinary
                    .intervene(&mut Backend::default(), "block.output", &global())
                    .unwrap();
                let ordinary = ordinary.take_step().unwrap();
                for step in &results {
                    assert_eq!(step.records[0].payload, ordinary.records[0].payload);
                    for (actual, expected) in step.interventions.iter().zip(&ordinary.interventions)
                    {
                        for (actual, expected) in actual.evidence.iter().zip(&expected.evidence) {
                            assert_eq!(actual.position, expected.position);
                            match (&actual.payload, &expected.payload) {
                                (
                                    Some(CapturePayload::Summary(actual)),
                                    Some(CapturePayload::Summary(expected)),
                                ) => {
                                    assert_eq!(actual.elements, expected.elements);
                                    assert_eq!(actual.finite, expected.finite);
                                    assert_eq!(actual.min, expected.min);
                                    assert_eq!(actual.max, expected.max);
                                    assert!(
                                        (actual.mean.unwrap() - expected.mean.unwrap()).abs()
                                            < 1e-12
                                    );
                                    assert!(
                                        (actual.rms.unwrap() - expected.rms.unwrap()).abs() < 1e-12
                                    );
                                }
                                _ => assert_eq!(actual.payload, expected.payload),
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn global_interventions_reject_mismatched_intent_missing_invocations_and_member_failures() {
    for mode in ["intent", "placement", "omitted", "geometry", "native"] {
        let all = world(5);
        let hooks = world(4);
        let boundary = Arc::new(Barrier::new(5));
        let next_model = Arc::new(AtomicUsize::new(0));
        std::thread::scope(|scope| {
            let workers = (0..5)
                .map(|rank| {
                    let all = Arc::clone(&all);
                    let hooks = Arc::clone(&hooks);
                    let boundary = Arc::clone(&boundary);
                    let next_model = Arc::clone(&next_model);
                    scope.spawn(move || {
                        let members = vec![0, 2, 3, 4];
                        let transport = HookTransport {
                            transport: transport(all, rank, Fault::None),
                            hook: members
                                .iter()
                                .position(|r| *r == rank)
                                .map(|r| transport(hooks, r, Fault::None)),
                            members,
                        };
                        let mut layout = layout();
                        if mode == "placement" && rank == 2 {
                            layout.0[1].1 =
                                ComponentCoordinateMap::indices(20, (8..20).rev().collect())
                                    .unwrap();
                        }
                        let (capture, plan) = plans(
                            rank,
                            InterventionEvidence::None,
                            mode == "intent" && rank == 2,
                        );
                        let discovery = discovery(&capture);
                        let mut empty = capture.plan().clone();
                        empty.selections.clear();
                        let empty = empty
                            .admit(
                                &discovery.catalog,
                                &discovery.support,
                                &discovery.support.capture,
                                capture.request(),
                            )
                            .unwrap();
                        let mut session = configured(empty, 5);
                        session
                            .enable_interventions(plan, Arc::new(Estimates))
                            .unwrap();
                        let limits = PartitionCaptureReceiptLimits {
                            max_producers: 5,
                            max_fragments: 64,
                            max_record_bytes: 1 << 16,
                        };
                        let epoch = DistributedCommitEpoch::FIRST;
                        let mut observer = PartitionCaptureObserver::for_step(
                            &mut session,
                            Backend {
                                fail: mode == "native" && rank == 2,
                                ..Default::default()
                            },
                            &transport,
                            &layout,
                            0,
                            limits,
                            estimate,
                            |e| e,
                        )
                        .with_interventions();
                        observer
                            .prepare_transaction(epoch, crate::ExpertPass::Prefill)
                            .unwrap();
                        let coordinated = observer.coordinate_transaction(epoch);
                        if mode == "intent" || mode == "placement" {
                            assert!(coordinated.is_err());
                            assert!(transport
                                .hook
                                .as_ref()
                                .is_none_or(|hook| hook.calls.load(Ordering::SeqCst) == 0));
                        } else {
                            coordinated.unwrap();
                            if mode == "omitted" {
                                assert!(observer.complete_transaction(epoch).is_err());
                            } else if let Some((_, map)) = layout.0.iter().find(|(r, _)| *r == rank)
                            {
                                let mut input = local_value(&global(), map);
                                if mode == "geometry" && rank == 2 {
                                    input.shape[1] += 1;
                                }
                                let result = observer.intervene("block.output", &input);
                                if result.is_ok() {
                                    next_model.fetch_add(1, Ordering::SeqCst);
                                }
                                assert!(result.is_err());
                                if mode == "native" && rank == 2 {
                                    assert!(result
                                        .err()
                                        .unwrap()
                                        .to_string()
                                        .contains("injected native intervention failure"));
                                }
                            }
                        }
                        boundary.wait();
                        assert_eq!(next_model.load(Ordering::SeqCst), 0);
                        observer.finish_transaction(epoch, false);
                        drop(observer);
                        assert!(session
                            .take_step()
                            .unwrap()
                            .interventions
                            .iter()
                            .all(|record| record.outcome != InterventionOutcome::Applied));
                        assert!(!transport.transport.poison.load(Ordering::SeqCst));
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
    }
}

#[test]
fn global_intervention_authority_rejects_foreign_owners_direct_bypass_and_restore_replay() {
    let (capture, plan) = plans(0, InterventionEvidence::None, false);
    let catalog = discovery(&capture);
    let new_session = || {
        let mut session = configured(capture.clone(), 1);
        session
            .enable_interventions(plan.clone(), Arc::new(Estimates))
            .unwrap();
        session
    };
    let mut session = new_session();
    let saved = session.checkpoint(&catalog).unwrap();
    let mut foreign = new_session();
    let transport = HookTransport {
        transport: transport(world(1), 0, Fault::None),
        hook: None,
        members: vec![0],
    };
    let layout = EditLayout(vec![(0, ComponentCoordinateMap::range(20, 0..20).unwrap())]);
    let limits = PartitionCaptureReceiptLimits {
        max_producers: 1,
        max_fragments: 8,
        max_record_bytes: 65536,
    };
    let epoch = DistributedCommitEpoch::FIRST;
    session
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    foreign
        .prepare_step_transaction(epoch, crate::ExpertPass::Prefill, 0)
        .unwrap();
    let mut backend = Backend::default();
    let mut work = session
        .prepare_partition_intervention(&transport, &layout, &backend, 0, limits)
        .unwrap();
    let used = session.cumulative_usage();
    assert!(foreign
        .apply_partition_intervention(&mut work, &mut backend, &global())
        .is_err());
    assert!(
        session
            .apply_partition_intervention(&mut work, &mut backend, &global())
            .is_err(),
        "cannot bypass common coordination"
    );
    assert!(
        session
            .intervene(&mut backend, "block.output", &global())
            .is_err(),
        "ordinary editing cannot bypass partition authority"
    );
    assert_eq!(backend.transforms, 0);
    session.finish_transaction(epoch, false);
    session.take_step().unwrap();
    session.restore(&saved).unwrap();
    assert_eq!(session.cumulative_usage(), used);
    session
        .prepare_step_transaction(epoch.next().unwrap(), crate::ExpertPass::Prefill, 0)
        .unwrap();
    assert!(session
        .apply_partition_intervention(&mut work, &mut backend, &global())
        .is_err());
    assert!(session.complete_partition_intervention(work).is_err());
    assert_eq!(transport.transport.calls.load(Ordering::SeqCst), 0);
    assert!(session.cumulative_usage().host_bytes > used.host_bytes);
}
