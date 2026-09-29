//! Actual dense tensors through shared admission, accounting and control restoration.
use super::*;
use eredu_core::{capture::*, intervention::*, speculative::*};
use eredu_runtime::capture::{
    CaptureBackendProvider, CaptureExecutionError, SpeculativeCaptureObserver,
};

#[path = "routing.rs"]
mod routing;
#[path = "sparse.rs"]
mod sparse;

#[path = "boundaries.rs"]
mod boundaries;

#[path = "graph.rs"]
mod graph;

struct Collector;
struct Provider;
impl CaptureBackendProvider for Provider {
    type Tensor = NumericTensor;
    type Error = Error;
    type Backend<'a> = Collector;
    fn backend(&mut self) -> Collector {
        Collector
    }
}
fn indices(value: &NumericTensor, slice: &ResolvedCaptureSlice) -> Vec<usize> {
    (0..slice.shape.iter().product::<u64>())
        .map(|mut ordinal| {
            let (mut index, mut stride) = (0, 1);
            for axis in (0..slice.shape.len()).rev() {
                index += (slice.starts[axis] + ordinal % slice.shape[axis] * slice.strides[axis])
                    as usize
                    * stride;
                ordinal /= slice.shape[axis];
                stride *= value.shape[axis] as usize;
            }
            index
        })
        .collect()
}
fn usage(source: &[u64], slice: &ResolvedCaptureSlice) -> CaptureUsage {
    let count = slice.shape.iter().product::<u64>();
    CaptureUsage {
        captures: 1,
        retained_bytes: source.iter().product::<u64>() * 4 + count * 16 + 4096,
        host_bytes: count * 16 + 4096,
        encoded_bytes: count * 32 + 4096,
    }
}
thread_local! { static TRANSFORMS: Cell<usize> = const { Cell::new(0) }; }
impl CaptureBackend for Collector {
    fn estimate_routed_units(
        &self,
        shape: &[u64],
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(sparse::cost(shape, slice))
    }
    fn capture_routed_units(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, NumericTensor>,
        geometry: RoutedUnitGeometry,
        slice: &ResolvedCaptureSlice,
    ) -> Option<Result<RoutedUnitCapture, Error>> {
        Some(Ok(sparse::capture(source, geometry, slice)))
    }
    type Tensor = NumericTensor;
    type Error = Error;
    fn shape(&self, value: &NumericTensor) -> Result<Vec<u64>, Error> {
        Ok(value.shape.iter().map(|d| *d as u64).collect())
    }
    fn source_dtype(&self, value: &NumericTensor) -> Option<eredu_core::checkpoint::TensorDtype> {
        Some(value.dtype.clone())
    }
    fn estimate(
        &self,
        value: &NumericTensor,
        _: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(usage(&self.shape(value).unwrap(), slice))
    }
    fn transform(
        &mut self,
        value: &NumericTensor,
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CapturePayload, Error> {
        assert!(matches!(
            selection.transform,
            CaptureTransform::Slice | CaptureTransform::Preview { .. }
        ));
        TRANSFORMS.with(|n| n.set(n.get() + 1));
        // Gather only requested scalars. Integer payloads retain their exact
        // backing rather than the numeric fixture's auxiliary floating view.
        let mut indices = indices(value, slice);
        let shape = if let CaptureTransform::Preview { max_elements } = selection.transform {
            indices.truncate(max_elements as usize);
            vec![indices.len()]
        } else {
            slice.shape.iter().map(|n| *n as usize).collect()
        };
        let data = match &value.dtype {
            eredu_core::checkpoint::TensorDtype::I32 => {
                let exact = value
                    .exact_i32
                    .as_ref()
                    .expect("integer capture requires exact backing");
                eredu_core::TensorObservationData::I64(
                    indices.into_iter().map(|i| i64::from(exact[i])).collect(),
                )
            }
            eredu_core::checkpoint::TensorDtype::F32 => eredu_core::TensorObservationData::F32(
                indices.into_iter().map(|i| value.data[i]).collect(),
            ),
            other => panic!("unadvertised fixture capture dtype {other:?}"),
        };
        Ok(CapturePayload::Tensor(
            eredu_core::TensorObservation::new(shape, data).unwrap(),
        ))
    }
}
impl InterventionBackend for Collector {
    fn routed_unit_locations(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, NumericTensor>,
        _: RoutedUnitGeometry,
    ) -> Option<Result<RoutedUnitLocations, Error>> {
        Some(Ok(sparse::unit_locations(source)))
    }
    fn select_elements(
        &mut self,
        source: &NumericTensor,
        indices: &[u64],
    ) -> Option<Result<NumericTensor, Error>> {
        Some(Ok(NumericTensor::new(
            vec![indices.len() as i32],
            indices.iter().map(|i| source.data[*i as usize]).collect(),
        )))
    }
    fn update_elements(
        &mut self,
        source: &NumericTensor,
        indices: &[u64],
        replacement: &NumericTensor,
    ) -> Option<Result<NumericTensor, Error>> {
        let mut result = source.clone();
        for (i, v) in indices.iter().zip(&replacement.data) {
            result.data[*i as usize] = *v;
        }
        Some(Ok(result))
    }
    fn intervention_dtype(&self, _: &NumericTensor) -> Result<InterventionDtype, Error> {
        Ok(InterventionDtype::Float32)
    }
    fn validate_intervention_geometry(
        &self,
        _: &[u64],
        _: &ResolvedCaptureSlice,
    ) -> Result<(), CaptureError> {
        Ok(())
    }
    fn select_region(
        &mut self,
        value: &NumericTensor,
        slice: &ResolvedCaptureSlice,
    ) -> Result<NumericTensor, Error> {
        Ok(NumericTensor::new(
            slice.shape.iter().map(|n| *n as i32).collect::<Vec<_>>(),
            indices(value, slice)
                .into_iter()
                .map(|i| value.data[i])
                .collect(),
        ))
    }
    fn update_region(
        &mut self,
        value: &NumericTensor,
        slice: &ResolvedCaptureSlice,
        replacement: &NumericTensor,
    ) -> Result<NumericTensor, Error> {
        let mut result = value.clone();
        for (i, v) in indices(value, slice).into_iter().zip(&replacement.data) {
            result.data[i] = *v;
        }
        Ok(result)
    }
    fn zeros(&mut self, shape: &[u64], _: InterventionDtype) -> Result<NumericTensor, Error> {
        Ok(NumericTensor::new(
            shape.iter().map(|n| *n as i32).collect::<Vec<_>>(),
            vec![0.; shape.iter().product::<u64>() as usize],
        ))
    }
    fn scale(&mut self, value: &NumericTensor, factor: f32) -> Result<NumericTensor, Error> {
        Ok(NumericTensor::new(
            value.shape.clone(),
            value.data.iter().map(|v| v * factor).collect(),
        ))
    }
    fn fill_masked(
        &mut self,
        _: &NumericTensor,
        _: &[bool],
        _: f32,
    ) -> Result<NumericTensor, Error> {
        unreachable!("unadvertised mask")
    }
    fn mask_components(
        &mut self,
        _: &NumericTensor,
        _: &[u32],
        _: bool,
    ) -> Result<NumericTensor, Error> {
        unreachable!("unadvertised mask")
    }
    fn realize_tensor(&mut self, _: &InterventionTensor) -> Result<NumericTensor, Error> {
        unreachable!("unadvertised upload")
    }
    fn add(&mut self, _: &NumericTensor, _: &NumericTensor) -> Result<NumericTensor, Error> {
        unreachable!("unadvertised addition")
    }
    fn fill_columns(
        &mut self,
        value: &NumericTensor,
        ids: &[u32],
        fill: f32,
    ) -> Result<NumericTensor, Error> {
        let mut result = value.clone();
        let width = *value.shape.last().unwrap() as usize;
        for row in result.data.chunks_mut(width) {
            for id in ids {
                row[*id as usize] = fill;
            }
        }
        Ok(result)
    }
}
impl InterventionEstimator for Collector {
    fn routed_unit_usage(
        &self,
        geometry: RoutedUnitGeometry,
        source: &[u64],
        _: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(sparse::edit_cost(geometry, source))
    }
    fn validate_geometry(&self, _: &[u64], _: &ResolvedCaptureSlice) -> Result<(), CaptureError> {
        Ok(())
    }
    fn activation_usage(
        &self,
        source: &[u64],
        slice: &ResolvedCaptureSlice,
        _: &InterventionAction,
    ) -> Result<CaptureUsage, CaptureError> {
        let mut cost = usage(source, slice);
        cost.captures = 0;
        cost.encoded_bytes = 0;
        Ok(cost)
    }
    fn capture_usage(
        &self,
        source: &[u64],
        selection: &CaptureSelection,
        slice: &ResolvedCaptureSlice,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(if selection.transform == CaptureTransform::RoutedUnits {
            sparse::cost(source, slice)
        } else {
            usage(source, slice)
        })
    }
    fn original_route_usage(
        &self,
        policy: &InterventionRoutingPolicy,
        rows: u64,
    ) -> Result<CaptureUsage, CaptureError> {
        Ok(CaptureUsage {
            retained_bytes: 4096 + rows * u64::from(policy.expert_count) * 32,
            host_bytes: 4096 + rows * u64::from(policy.top_k) * 16,
            ..Default::default()
        })
    }
}
pub(in super::super) fn observer(
    plan: &AdmittedSpeculativeActivations,
    request: eredu_core::SpeculativeRequestId,
) -> Result<
    Option<Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<NumericTensor, Error>>>,
    SpeculativeControlError,
> {
    let observer = SpeculativeCaptureObserver::from_admitted(
        plan,
        Provider,
        |e: &CaptureExecutionError<Error>| Error::backend(e.to_string()),
        request,
        std::sync::Arc::new(Collector),
    )?;
    Ok(observer.map(|o| Box::new(o) as _))
}
fn binding() -> eredu_architectures::qwen4_exp::prepared::PredictionObservationBinding<'static> {
    use eredu_core::artifact::{fingerprint_artifact, ArtifactMemberIdentity};
    eredu_architectures::qwen4_exp::prepared::PredictionObservationBinding {
        // Synthetic content authority: the fixture owns this in-memory source.
        artifact: fingerprint_artifact(
            "numeric-qwen4-pair",
            [ArtifactMemberIdentity::new("fixture", 1, [91; 32])],
        )
        .unwrap(),
        session: "numeric-qwen4-pair",
        overlay: None,
        observations: eredu_core::ObservationMechanisms {
            activation_tensors: true,
            routing_tensors: true,
            routed_unit_tensors: true,
            floating_to_f32: true,
        },
        captures: CaptureCapabilities {
            transformations: vec![
                CaptureTransformKind::Slice,
                CaptureTransformKind::RoutedUnits,
                CaptureTransformKind::Preview,
            ],
            ..Default::default()
        },
        interventions: InterventionMechanisms {
            routed_units: true,
            operations: vec![
                InterventionKind::Zero,
                InterventionKind::Scale,
                InterventionKind::MaskLogits,
                InterventionKind::ExcludeExperts,
                InterventionKind::ZeroExpertContribution,
                InterventionKind::BiasRoutingScores,
                InterventionKind::ForceExperts,
            ],
            score_stages: vec![
                RoutingScoreStage::RawLogits,
                RoutingScoreStage::TransformedScores,
                RoutingScoreStage::RankingScores,
            ],
            dtypes: vec![InterventionDtype::Float32],
            ..Default::default()
        },
    }
}
pub(super) fn discovery(
    selected: &Selected,
    extension: &Executor,
    speculative: &SelectedSpeculativeRealization,
) -> SpeculativeActivationDiscovery {
    let execution = <Executor as MaterializedPredictionExecutor<
        Target,
        NumericBackend,
        Materializer,
    >>::activation_execution(extension, speculative)
    .unwrap();
    selected
        .speculative_discovery(&execution, binding())
        .unwrap()
        .activations
}
fn plan(discovery: &SpeculativeActivationDiscovery) -> SpeculativeActivationPlan {
    // One finite MiB per selected point covers three unchanged continuations.
    // More declared points spend more allowance; budget-failure checks below
    // still constrain the actual shared cumulative ledger independently.
    let bytes = (discovery.captures.catalog.points.len() as u64) << 20;
    let limits = CaptureUsage {
        captures: 100_000,
        retained_bytes: bytes,
        host_bytes: bytes,
        encoded_bytes: bytes,
    };
    SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        captures: CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: discovery
                .captures
                .catalog
                .points
                .iter()
                .map(|p| CaptureSelection {
                    id: p.path.clone(),
                    path: p.path.clone(),
                    schedule: CaptureSchedule::default(),
                    slices: vec![CaptureSlice {
                        axis: p.axes.as_ref().unwrap()[0].name.clone(),
                        start: 0,
                        end: 1,
                        stride: 1,
                    }],
                    transform: if matches!(
                        p.value_type,
                        eredu_core::ObservationValueType::RoutedUnits { .. }
                    ) {
                        CaptureTransform::RoutedUnits
                    } else {
                        CaptureTransform::Slice
                    },
                })
                .collect(),
            limits: CaptureLimits {
                per_step: limits,
                cumulative: limits,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        },
        interventions: InterventionPlan::none(),
        bounds: CaptureInvocationBounds {
            batch: 1,
            max_sequence: 8,
            max_context: None,
            max_predictions: 12,
        },
    }
}
pub(super) fn check(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    plain: &Outcome,
) {
    let (_, extension) = paired(selected, prepared, ctx);
    let speculative = selection(selected);
    let reads = prepared
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    let discovery = discovery(selected, &extension, &speculative);
    let execution = <Executor as MaterializedPredictionExecutor<
        Target,
        NumericBackend,
        Materializer,
    >>::activation_execution(&extension, &speculative)
    .unwrap();
    let joined = selected
        .speculative_discovery(&execution, binding())
        .unwrap();
    graph::check(&joined, prepared);
    assert_eq!(
        joined.activations.captures.catalog,
        discovery.captures.catalog
    );
    assert_eq!(
        prepared
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        reads,
        "discovery must not reread checkpoint payloads"
    );
    assert!(discovery.captures.support.points.iter().all(|p| p.prefill
        == eredu_core::ObservationSupportStatus::Supported
        && p.decode == eredu_core::ObservationSupportStatus::Supported));
    integer_slice_check(&discovery, ctx);
    routing_support_check(selected, &extension, &speculative);
    sparse::checks(selected, prepared, ctx, &discovery, plain);
    routing::checks(selected, prepared, ctx, &discovery, plain);
    boundaries::checks(selected, prepared, ctx, &discovery, plain);
    let admitted = plan(&discovery).admit(&discovery).unwrap();
    let mut physical = plan(&discovery);
    physical.captures.limits.physical_native_bytes = Some(1024);
    assert!(physical.admit(&discovery).is_err());
    let mut changed = discovery.clone();
    changed.execution_identity.push('x');
    assert!(admitted.validate(&changed).is_err());
    changed = discovery.clone();
    changed.interventions.session_identity = Some("different".into());
    assert!(admitted.validate(&changed).is_err());
    let execution = <Executor as MaterializedPredictionExecutor<
        Target,
        NumericBackend,
        Materializer,
    >>::activation_execution(&extension, &speculative)
    .unwrap();
    let mut unavailable = binding();
    unavailable.observations.activation_tensors = false;
    let unavailable = selected
        .speculative_discovery(&execution, unavailable)
        .unwrap()
        .activations;
    assert!(plan(&unavailable).admit(&unavailable).is_err());
    let mut options = options();
    options.activations = Some(admitted.clone());
    let mut records = vec![];
    let captured = run(selected, prepared, ctx, options.clone(), |session| {
        while let Some(step) = session.step()? {
            records.extend(step.activations);
        }
        Ok(())
    });
    assert_eq!(captured.tokens, plain.tokens);
    exact(&captured.target, &plain.target, "captured target");
    exact(
        &captured.prediction,
        &plain.prediction,
        "captured prediction",
    );
    let mut observed = std::collections::BTreeSet::new();
    let mut nonzero = std::collections::BTreeSet::new();
    let mut phases = vec![];
    for invocation in &records {
        assert!(invocation.completed);
        assert_eq!(
            invocation.admission_identity.as_deref(),
            Some(admitted.identity())
        );
        phases.push(invocation.phase);
        routing_sums(invocation, &discovery);
        boundaries::validate(invocation);
        for record in &invocation.captures.records {
            let scope = discovery
                .bindings
                .iter()
                .find(|b| b.node_id == record.node_id)
                .unwrap()
                .scope;
            if scope.applies(invocation.phase) {
                assert_eq!(
                    record.outcome,
                    CaptureOutcome::Captured,
                    "{} {:?}",
                    record.path,
                    invocation.phase
                );
                if matches!(record.payload, Some(CapturePayload::RoutedUnits(_))) {
                    if sparse::validate(invocation, record) {
                        nonzero.insert(record.path.clone());
                    }
                    observed.insert(record.path.clone());
                    continue;
                }
                let CapturePayload::Tensor(tensor) = record.payload.as_ref().unwrap() else {
                    panic!()
                };
                let axes = discovery
                    .captures
                    .catalog
                    .get(&record.path)
                    .unwrap()
                    .axes
                    .as_ref()
                    .unwrap();
                assert_eq!(tensor.shape().len(), axes.len());
                for (size, axis) in tensor.shape().iter().zip(axes) {
                    if let eredu_core::SymbolicDimension::Known(n) = axis.dimension {
                        assert_eq!(*size, n, "{}", record.path);
                    }
                }
                let point = discovery.captures.catalog.get(&record.path).unwrap();
                let is_nonzero = match tensor.data() {
                    eredu_core::TensorObservationData::F32(values) => {
                        assert_eq!(point.dtype, eredu_core::ObservationDtype::Floating);
                        assert_eq!(
                            record.source_dtype,
                            Some(eredu_core::checkpoint::TensorDtype::F32)
                        );
                        assert!(values.iter().all(|v| v.is_finite()));
                        values.iter().any(|v| *v != 0.)
                    }
                    eredu_core::TensorObservationData::I64(values) => {
                        assert_eq!(point.dtype, eredu_core::ObservationDtype::Integer);
                        assert_eq!(
                            record.source_dtype,
                            Some(eredu_core::checkpoint::TensorDtype::I32)
                        );
                        assert_eq!(point.position, eredu_core::ObservationPosition::ReadOnly);
                        assert!(!discovery
                            .interventions
                            .points
                            .iter()
                            .any(|p| p.path == point.path));
                        if axes.last().unwrap().name == "selected_position" {
                            selected_positions(values, tensor.shape(), scope, invocation);
                        } else {
                            assert_eq!(axes.last().unwrap().name, "selected_expert");
                            assert!(values.iter().all(|id| (0..3).contains(id)));
                            assert!(values.chunks(2).all(|ids| ids[0] != ids[1]));
                        }
                        values.iter().any(|v| *v != 0)
                    }
                    data => panic!("unexpected capture {data:?}"),
                };
                if is_nonzero {
                    nonzero.insert(record.path.clone());
                }
                observed.insert(record.path.clone());
            } else {
                assert_eq!(
                    record.outcome,
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::NotInvoked
                    }
                );
            }
        }
    }
    assert_eq!(observed.len(), discovery.captures.catalog.points.len());
    assert_eq!(
        nonzero, observed,
        "every declared point needs nonzero coverage somewhere; EOS lexical writes may be zero"
    );
    for phase in [
        SpeculativeActivationPhase::TargetPrefill,
        SpeculativeActivationPhase::PredictionPrefill,
        SpeculativeActivationPhase::Proposal { depth: 0 },
        SpeculativeActivationPhase::Proposal { depth: 1 },
        SpeculativeActivationPhase::Verification,
        SpeculativeActivationPhase::PredictionReplay,
        SpeculativeActivationPhase::TargetReplay,
    ] {
        assert!(phases.contains(&phase), "{phase:?}");
    }
    assert!(records
        .windows(2)
        .all(|p| p[0].invocation < p[1].invocation));

    let mut replayed = vec![];
    let restored = run(selected, prepared, ctx, options, |session| {
        replayed.extend(session.step()?.unwrap().activations);
        let saved = session.snapshot()?;
        let branch = session.fork(&saved)?;
        for attempt in 0..2 {
            while let Some(step) = session.step()? {
                replayed.extend(step.activations);
            }
            if attempt == 0 {
                session.restore(&saved)?;
            }
        }
        session.exchange(&branch)?;
        assert_eq!(session.run_id(), 1);
        while let Some(step) = session.step()? {
            assert_eq!(step.run_id, 1);
            replayed.extend(step.activations);
        }
        session.exchange(&branch)?;
        session.release_branch(&branch)?;
        Ok(())
    });
    exact(&restored.target, &plain.target, "capture restore target");
    exact(
        &restored.prediction,
        &plain.prediction,
        "capture restore prediction",
    );
    assert!(replayed.len() > records.len());
    assert!(replayed
        .windows(2)
        .all(|p| p[0].invocation < p[1].invocation
            && p[0].captures.cumulative_usage.captures <= p[1].captures.cumulative_usage.captures));
    assert!(
        replayed.last().unwrap().captures.cumulative_usage.captures
            > records.last().unwrap().captures.cumulative_usage.captures
    );
    // Repeated unchanged branches must reproduce values and provenance while
    // retaining new invocation identities and cumulative reservations.
    let prefill = records
        .iter()
        .take_while(|r| {
            matches!(
                r.phase,
                SpeculativeActivationPhase::TargetPrefill
                    | SpeculativeActivationPhase::PredictionPrefill
            )
        })
        .count();
    let tail = &records[prefill..];
    assert_eq!(replayed.len(), prefill + 3 * tail.len());
    for replay in replayed[prefill..].chunks(tail.len()) {
        for (actual, expected) in replay.iter().zip(tail) {
            assert_eq!(actual.phase, expected.phase);
            assert_eq!(actual.origin, expected.origin);
            for (actual, expected) in actual
                .captures
                .records
                .iter()
                .zip(&expected.captures.records)
            {
                assert_eq!(actual.payload, expected.payload);
                assert_eq!(actual.source_shape, expected.source_shape);
            }
        }
    }
    budget_checks(selected, prepared, ctx, &discovery, &records);
    edit_checks(selected, prepared, ctx, &discovery, plain);
    eprintln!("Qwen4 portable capture: policy={:?}, points={}, invocations={}, restore/fork invocations={}, cumulative={:?}",
        selected.realization().text().residency(), discovery.captures.catalog.points.len(), records.len(), replayed.len(), replayed.last().unwrap().captures.cumulative_usage);
}

fn budget_checks(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    discovery: &SpeculativeActivationDiscovery,
    records: &[SpeculativeActivationCapture],
) {
    let mut request = plan(discovery);
    request.captures.limits.per_step.captures = 1;
    let mut options = super::options();
    options.activations = Some(request.admit(discovery).unwrap());
    let before = TRANSFORMS.with(Cell::get);
    let failed = run_result(selected, prepared, ctx, options, |_| {
        panic!("preflight must reject before advancement")
    });
    assert!(matches!(
        failed,
        Err(SpeculativeControlError::Capture(CaptureError::Limit {
            budget: CaptureBudget::Captures,
            cumulative: false
        }))
    ));
    assert_eq!(TRANSFORMS.with(Cell::get), before);

    let mut request = plan(discovery);
    request.captures.limits.cumulative.captures =
        records.last().unwrap().captures.cumulative_usage.captures
            + 2 * discovery.captures.catalog.points.len() as u64;
    let mut options = super::options();
    options.activations = Some(request.admit(discovery).unwrap());
    let exhausted = Cell::new(false);
    let failed = run_result(selected, prepared, ctx, options, |session| {
        session.step()?;
        let saved = session.snapshot()?;
        finish(session, &mut vec![])?;
        session.restore(&saved)?;
        loop {
            match session.step() {
                Ok(Some(_)) => {}
                Ok(None) => panic!("restore must not refund the capture allowance"),
                Err(error) => {
                    assert!(
                        matches!(
                            error,
                            SpeculativeControlError::Capture(CaptureError::Limit {
                                budget: CaptureBudget::Captures,
                                cumulative: true
                            })
                        ),
                        "{error}"
                    );
                    exhausted.set(true);
                    let before = TRANSFORMS.with(Cell::get);
                    let mut aborted = false;
                    while let Some(evidence) = session.take_activation_evidence()? {
                        aborted |= !evidence.activation.completed;
                        assert!(
                            evidence.activation.invocation > records.last().unwrap().invocation
                        );
                    }
                    assert!(
                        aborted,
                        "failed invocation remains available as bounded evidence"
                    );
                    assert!(session.step().is_err());
                    assert_eq!(TRANSFORMS.with(Cell::get), before);
                    return Err(error);
                }
            }
        }
    });
    assert!(exhausted.get());
    assert!(matches!(
        failed,
        Err(SpeculativeControlError::Capture(CaptureError::Limit {
            budget: CaptureBudget::Captures,
            cumulative: true
        }))
    ));
}
fn edit_checks(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    discovery: &SpeculativeActivationDiscovery,
    plain: &Outcome,
) {
    let mut request = plan(discovery);
    let operation = InterventionOperation {
        id: "identity-scale".into(),
        target: "mtp.layers.0.prediction.fusion".into(),
        schedule: CaptureSchedule::default(),
        slices: vec![],
        action: InterventionAction::Scale {
            dtype: InterventionDtype::Float32,
            factor: 1.,
        },
        evidence: InterventionEvidence::None,
    };
    request.interventions.operations.push(operation);
    for path in [
        "mtp.layers.0.prediction.readout.projection_input",
        "mtp.layers.0.prediction.attention.write_input",
        "mtp.layers.1.prediction.mixer.qkv.projected",
        "mtp.layers.1.prediction.mixer.write_input",
        "mtp.layers.0.prediction.attention.selected_positions",
        "mtp.layers.0.mlp.routing.selected_experts",
        "mtp.layers.0.mlp.routing.routed_output",
    ] {
        let mut readonly = request.clone();
        readonly.interventions.operations[0].target = path.into();
        assert!(
            readonly.admit(discovery).is_err(),
            "read-only boundary {path}"
        );
    }
    let admitted = request.clone().admit(discovery).unwrap();
    let mut options = super::options();
    options.activations = Some(admitted.clone());
    let mut edits = vec![];
    let edited = run(selected, prepared, ctx, options, |session| {
        edits.extend(session.step()?.unwrap().activations);
        // Loaded-session validation also guards prospective re-admission.
        let mut foreign = discovery.clone();
        foreign.execution_identity.push('x');
        let foreign = plan(&foreign).admit(&foreign).unwrap();
        assert!(session.readmit_activation_interventions(foreign).is_err());
        session.readmit_activation_interventions(admitted)?;
        while let Some(step) = session.step()? {
            edits.extend(step.activations);
        }
        Ok(())
    });
    let scope = SpeculativeCaptureScope::Prediction { depth: 0 };
    for invocation in &edits {
        assert_eq!(invocation.captures.interventions.len(), 1);
        assert_eq!(
            invocation.captures.interventions[0].outcome,
            if scope.applies(invocation.phase) {
                InterventionOutcome::Applied
            } else {
                InterventionOutcome::Inactive
            }
        );
    }
    assert_eq!(edited.tokens, plain.tokens);
    exact(&edited.target, &plain.target, "identity edit target");
    exact(
        &edited.prediction,
        &plain.prediction,
        "identity edit prediction",
    );
    for (target, consumed) in [
        (
            "mtp.layers.0.prediction.fusion",
            "mtp.layers.0.prediction.attention.input",
        ),
        (
            "mtp.layers.0.prediction.attention.channels",
            "mtp.layers.0.prediction.attention.write_input",
        ),
        (
            "mtp.layers.1.prediction.mixer.channels",
            "mtp.layers.1.prediction.mixer.write_input",
        ),
        (
            "mtp.layers.0.mlp.shared_expert.feed_forward.units",
            "mtp.layers.0.mlp.shared_expert.feed_forward.write_input",
        ),
    ] {
        let mut request = request.clone();
        request.interventions.operations[0].id = "zero-internal-value".into();
        request.interventions.operations[0].target = target.into();
        request.interventions.operations[0].action = InterventionAction::Zero {
            dtype: InterventionDtype::Float32,
        };
        let mut options = super::options();
        options.activations = Some(request.admit(discovery).unwrap());
        let edited = run(selected, prepared, ctx, options, |session| {
            let step = session.step()?.unwrap();
            let prediction = step
                .activations
                .iter()
                .find(|a| a.phase == SpeculativeActivationPhase::PredictionPrefill)
                .unwrap();
            let values = |path: &str| {
                let record = prediction
                    .captures
                    .records
                    .iter()
                    .find(|r| r.path == path)
                    .unwrap();
                let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
                    panic!("missing {path}")
                };
                let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                    panic!()
                };
                values.clone()
            };
            assert!(
                values(target).iter().any(|v| *v != 0.),
                "ordinary capture precedes edit at {target}"
            );
            assert!(
                values(consumed).iter().all(|v| *v == 0.),
                "{consumed} consumes the replacement"
            );
            assert_eq!(
                prediction.captures.interventions[0].outcome,
                InterventionOutcome::Applied
            );
            Ok(())
        });
        assert_eq!(edited.tokens, plain.tokens[..1]);
    }
}

// Independent structural oracle: all visible tokens fit before the budget;
// afterwards retain two complete three-token blocks plus the incomplete tail.
fn selected_positions(
    values: &[i64],
    shape: &[usize],
    scope: SpeculativeCaptureScope,
    invocation: &SpeculativeActivationCapture,
) {
    assert_eq!(shape[2], 8);
    for (row, slots) in values.chunks(8).enumerate() {
        let valid = slots
            .iter()
            .take_while(|p| **p >= 0)
            .copied()
            .collect::<Vec<_>>();
        assert!(slots[valid.len()..].iter().all(|p| *p == -1));
        assert!(valid.windows(2).all(|p| p[0] < p[1]));
        assert!(!valid.is_empty());
        let start = match (scope, invocation.phase) {
            (SpeculativeCaptureScope::Target, SpeculativeActivationPhase::TargetPrefill) => Some(0),
            (
                SpeculativeCaptureScope::Target,
                SpeculativeActivationPhase::Verification | SpeculativeActivationPhase::TargetReplay,
            ) => Some(2 + invocation.origin.committed_tokens),
            (_, SpeculativeActivationPhase::PredictionPrefill) => Some(0),
            _ => None,
        };
        if let Some(start) = start {
            let visible = start + row + 1;
            let complete = visible / 3;
            let tail = visible % 3;
            assert_eq!(valid.len(), complete.min(2) * 3 + tail);
            assert!(valid.iter().all(|p| *p < visible as i64));
            assert_eq!(
                &valid[valid.len() - tail..],
                &(visible - tail..visible)
                    .map(|n| n as i64)
                    .collect::<Vec<_>>()
            );
            for block in valid[..valid.len() - tail].chunks(3) {
                assert_eq!(block[0] % 3, 0);
                assert_eq!(block, [block[0], block[0] + 1, block[0] + 2]);
            }
            if complete <= 2 {
                assert_eq!(valid, (0..visible as i64).collect::<Vec<_>>());
            }
        }
    }
}
fn integer_slice_check(discovery: &SpeculativeActivationDiscovery, ctx: &NumericContext) {
    let path = "model.layers.1.attention.selected_positions";
    let point = discovery.captures.catalog.get(path).unwrap();
    let selection = CaptureSelection {
        id: "exact-integers".into(),
        path: path.into(),
        schedule: CaptureSchedule::default(),
        slices: vec![CaptureSlice {
            axis: "selected_position".into(),
            start: 1,
            end: 8,
            stride: 2,
        }],
        transform: CaptureTransform::Slice,
    };
    let slice = resolve_slice(point, &selection, &[1, 1, 8]).unwrap();
    let source = NumericTensor::from_i32_slice(
        &[0, 16_777_217, 1, i32::MAX, 2, i32::MIN, 3, -1],
        &[1, 1, 8],
        ctx,
    )
    .unwrap();
    let mut collector = Collector;
    let payload = collector.transform(&source, &selection, &slice).unwrap();
    let CapturePayload::Tensor(tensor) = payload else {
        panic!()
    };
    assert_eq!(tensor.shape(), [1, 1, 4]);
    assert_eq!(
        tensor.data(),
        &eredu_core::TensorObservationData::I64(vec![
            16_777_217,
            i32::MAX as i64,
            i32::MIN as i64,
            -1
        ])
    );
}
fn routing_support_check(
    selected: &Selected,
    extension: &Executor,
    speculative: &SelectedSpeculativeRealization,
) {
    let execution = <Executor as MaterializedPredictionExecutor<
        Target,
        NumericBackend,
        Materializer,
    >>::activation_execution(extension, speculative)
    .unwrap();
    let mut facts = binding();
    facts.observations.routing_tensors = false;
    let discovery = selected
        .speculative_discovery(&execution, facts)
        .unwrap()
        .activations;
    let mut request = plan(&discovery);
    request
        .captures
        .selections
        .retain(|s| s.path == "model.layers.0.mlp.routing.selected_experts");
    assert!(request.admit(&discovery).is_err());
    let mut facts = binding();
    facts.observations.routed_unit_tensors = false;
    let unavailable = selected
        .speculative_discovery(&execution, facts)
        .unwrap()
        .activations;
    let mut request = plan(&unavailable);
    request
        .captures
        .selections
        .retain(|s| s.path == "mtp.layers.0.mlp.units");
    assert!(request.admit(&unavailable).is_err());
    let mut facts = binding();
    facts.interventions.routed_units = false;
    let unavailable = selected
        .speculative_discovery(&execution, facts)
        .unwrap()
        .activations;
    let mut request = plan(&unavailable);
    request
        .interventions
        .operations
        .push(InterventionOperation {
            id: "missing-sparse-editor".into(),
            target: "mtp.layers.0.mlp.units".into(),
            schedule: CaptureSchedule::default(),
            slices: vec![],
            action: InterventionAction::Zero {
                dtype: InterventionDtype::Float32,
            },
            evidence: InterventionEvidence::None,
        });
    assert!(request.admit(&unavailable).is_err());
}

fn routing_sums(
    invocation: &SpeculativeActivationCapture,
    discovery: &SpeculativeActivationDiscovery,
) {
    use eredu_core::RoutingObservationField as F;
    for point in &discovery.captures.catalog.points {
        let Some(root) = point.path.strip_suffix(".routing.routed_output") else {
            continue;
        };
        let scope = discovery
            .bindings
            .iter()
            .find(|b| b.node_id == point.node_id)
            .unwrap()
            .scope;
        if !scope.applies(invocation.phase) {
            continue;
        }
        let values = |field: F| {
            let path = field.path(root);
            let record = invocation
                .captures
                .records
                .iter()
                .find(|r| r.path == path)
                .unwrap();
            let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
                panic!("missing {path}")
            };
            let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                panic!()
            };
            values
        };
        let routed = values(F::RoutedOutput);
        let shared = values(F::SharedOutput);
        let combined = values(F::CombinedOutput);
        assert_eq!(routed.len(), combined.len());
        assert_eq!(shared.len(), combined.len());
        for ((routed, shared), combined) in routed.iter().zip(shared).zip(combined) {
            assert_eq!(*routed + *shared, *combined, "{root} routing attribution");
        }
    }
}
