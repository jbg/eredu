//! Scheduled sparse edits cross the same original observer as native providers.
use super::*;
use crate::intervention::{InterventionPrefillWindow, PreparedRoutedInterventionRows};
use eredu_core::intervention::*;

struct Edits {
    inner: Backend,
    source: OriginalInterventionSource,
    funding: HostMetadataFunding,
    charges: std::cell::Cell<usize>,
    ranges: Vec<[u64; 2]>,
}
impl ScheduledCaptureBackend for Edits {
    type Tensor = Value;
    type Error = Native;
    fn validate_source(
        &self,
        tensor: &Value,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<TensorDtype, Native> {
        self.inner.validate_source(tensor, geometry)
    }
    fn estimate(
        &self,
        tensor: &Value,
        geometry: &CaptureTensorGeometry<'_>,
    ) -> Result<CaptureUsage, CaptureError> {
        self.inner.estimate(tensor, geometry)
    }
    fn transform(
        &mut self,
        tensor: &Value,
        claim: CaptureTensorClaim<'_, '_>,
    ) -> Result<ClaimedCaptureTensor, Native> {
        self.inner.transform(tensor, claim)
    }
    fn prefill_routed_intervention_range(
        &self,
        source: &RoutedUnitCaptureSource<'_, Value>,
        claim: &CaptureInterventionClaim<'_>,
        window: InterventionPrefillWindow,
    ) -> Result<[u64; 2], FundedCaptureError<Native>> {
        assert_eq!(
            source.source_groups.shape,
            [window.physical().sequence as i32, 3]
        );
        self.routed_intervention_range(source, claim)
    }
    fn routed_intervention_range(
        &self,
        source: &RoutedUnitCaptureSource<'_, Value>,
        claim: &CaptureInterventionClaim<'_>,
    ) -> Result<[u64; 2], FundedCaptureError<Native>> {
        claim.validate_source(&self.source).unwrap();
        claim.validate_native_custody(&self.inner.scope).unwrap();
        assert_eq!(source.values.shape, [3, 5]);
        Ok([source.token_offset, source.token_offset + 1])
    }
    fn routed_intervention_usage(
        &self,
        _: &RoutedUnitCaptureSource<'_, Value>,
        _: &CaptureInterventionClaim<'_>,
    ) -> Result<CaptureUsage, FundedCaptureError<Native>> {
        self.charges.set(self.charges.get() + 1);
        Ok(CaptureUsage {
            host_bytes: 2048,
            retained_bytes: 1024,
            ..Default::default()
        })
    }
    fn prefill_routed_intervention_usage(
        &self,
        source: &RoutedUnitCaptureSource<'_, Value>,
        claim: &CaptureInterventionClaim<'_>,
        _: InterventionPrefillWindow,
    ) -> Result<CaptureUsage, FundedCaptureError<Native>> {
        self.routed_intervention_usage(source, claim)
    }
    fn apply_routed_intervention(
        &mut self,
        source: &RoutedUnitCaptureSource<'_, Value>,
        batch: RoutedInterventionBatch<'_, '_>,
    ) -> Result<Option<Value>, FundedCaptureError<Native>> {
        let (tokens, range) = batch.source_chunk();
        self.ranges.push(range);
        let local = [source.token_offset, source.token_offset + 1];
        let mut rows = if let Some(window) = batch.prefill_window() {
            PreparedRoutedInterventionRows::prepare_scheduled_prefill(
                &self.source,
                batch.claim().index(),
                window,
                local,
                3,
                self.funding.clone(),
            )
            .unwrap()
        } else {
            let (phase, prediction) = batch.claim().coordinate();
            PreparedRoutedInterventionRows::prepare(
                &self.source,
                batch.claim().index(),
                phase,
                prediction,
                None,
                None,
                tokens,
                range,
                3,
                self.funding.clone(),
            )
            .unwrap()
        };
        for slot in 0..3 {
            rows.push_row(RoutedUnitLocation {
                source_peer: None,
                token: source.token_offset,
                slot,
                expert: source.source_groups.values
                    [source.token_offset as usize * 3 + slot as usize]
                    as u64,
            })
            .unwrap();
        }
        let prepared = rows.finish().unwrap();
        batch.finish(&prepared).unwrap();
        Ok(Some(scalar(
            &[3, 5],
            source
                .values
                .values
                .iter()
                .map(|value| -2.0 * value)
                .collect(),
        )))
    }
}

fn sources(schedule: CaptureSchedule) -> (SharedCapturePlan, AdmittedInterventionPlan) {
    let base = sparse_source();
    let mut plan = base.admission().plan().clone();
    plan.selections.clear();
    let capture = SharedCapturePlan::new(
        plan.admit_with_text_origin(
            &ObservationCatalog {
                schema_version: 1,
                points: vec![],
                completeness: DescriptionCompleteness::Complete,
            },
            &ObservationSupportReport {
                schema_version: 1,
                capture: Default::default(),
                points: vec![],
            },
            &CaptureCapabilities::default(),
            base.admission().request(),
            CaptureTextOrigin {
                cached_positions: 7,
            },
        )
        .unwrap(),
    );
    let discovery = InterventionDiscovery {
        schema_version: 1,
        artifact_identity: "sparse-observer".into(),
        session_identity: Some("session".into()),
        points: vec![InterventionPoint {
            path: "units".into(),
            node_id: "layer.0".into(),
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
            operations: vec![InterventionKind::Scale],
            score_stages: vec![],
            prefill: ObservationSupportStatus::Supported,
            decode: ObservationSupportStatus::Supported,
            conditions: vec![],
            routing: None,
            routed_units: Some(RoutedUnitInterventionPoint {
                routing: "route".into(),
                geometry: RoutedUnitGeometry {
                    experts: 7,
                    units_per_expert: 5,
                    routes_per_token: 3,
                },
            }),
        }],
    };
    let edits = InterventionPlan {
        schema_version: 1,
        operations: vec![InterventionOperation {
            id: "scale".into(),
            target: "units".into(),
            schedule,
            slices: vec![],
            action: InterventionAction::Scale {
                dtype: InterventionDtype::Float32,
                factor: -2.0,
            },
            evidence: InterventionEvidence::None,
        }],
    }
    .admit_with_text_origin(
        &discovery,
        capture.admission().request(),
        CaptureTextOrigin {
            cached_positions: 7,
        },
        "session",
    )
    .unwrap();
    (capture, edits)
}

#[test]
fn scheduled_sparse_observer_applies_prefill_chunks_and_decode_without_invocation_token() {
    scheduled_sparse_observer(Default::default());
}

#[test]
fn scheduled_sparse_observer_omits_inactive_prefill_and_decode_provider_callbacks() {
    scheduled_sparse_observer(CaptureSchedule {
        first_prediction: 2,
        prefill: false,
        ..Default::default()
    });
}

fn scheduled_sparse_observer(schedule: CaptureSchedule) {
    let (capture, edits) = sources(schedule.clone());
    let pool = crate::working_memory::memory_fixture::host_ledger(1 << 27, 0).unwrap();
    let source = pool
        .compile_intervention_source(PreparedInterventionPlanCopy::inspect(&edits).unwrap())
        .unwrap();
    let plan = CaptureRunHostPlan::prepare(&capture)
        .unwrap()
        .with_interventions(&source)
        .unwrap();
    let admission = crate::working_memory::memory_fixture::host_admission(
        &pool,
        plan.initialization_peak_bytes(),
    );
    let execution = InferenceExecutionIdentity::default();
    let (reservation, run) = pool
        .reserve_with_capacity(&execution, &admission, pool.configured_limits().clone())
        .unwrap()
        .into_funding()
        .unwrap();
    let mut bank = run
        .prepare_capture_run(&reservation, plan)
        .unwrap()
        .into_capture_session()
        .unwrap();
    let funding = pool
        .prepare_workspace_metadata(&execution, pool.configured_limits().clone())
        .unwrap();
    let mut backend = Edits {
        inner: backend(&run),
        source,
        funding,
        charges: Default::default(),
        ranges: vec![],
    };
    let paths = super::super::super::source();
    let selected = paths.prepare_capture_selection(&capture).unwrap();
    let bound = selected.bind_geometry(geometry()).unwrap();
    let invoke = |o: &mut dyn ActivationObserver<Value, Error>, tokens: u64, base: f32| {
        let groups = scalar(
            &[tokens as i32, 3],
            (0..tokens * 3).map(|i| (i % 7) as f32).collect(),
        );
        let input = scalar(&[tokens as i32, 5], vec![1.0; tokens as usize * 5]);
        let routed = o.routed_unit_observer("route").unwrap().unwrap();
        routed
            .begin_invocation(&crate::RoutedUnitInvocation {
                input: &input,
                origins: None,
                unit_coordinates: None,
            })
            .unwrap();
        for token in 0..tokens {
            let values = scalar(
                &[3, 5],
                (0..15)
                    .map(|i| base + token as f32 * 100.0 + i as f32 + 0.125)
                    .collect(),
            );
            let indices = scalar(&[3], vec![0.0; 3]);
            let slots = scalar(&[3], vec![0.0, 1.0, 2.0]);
            let coefficients = scalar(&[1, 3], vec![0.25, 0.5, 0.75]);
            let batch = crate::RoutedUnitBatch {
                units: eredu_nn::GroupedUnitBatch {
                    values: &values,
                    group_indices: &slots,
                    selection_indices: &slots,
                    token_indices: &indices,
                    coefficients: &coefficients,
                    token_offset: token as usize,
                    total_token_count: tokens as usize,
                    group_count: 7,
                },
                source_groups: &groups,
                global_groups: None,
                provider_token_offset: 0,
                origins: None,
                unit_coordinates: None,
            };
            let output = routed.intervene(&batch).unwrap().unwrap();
            assert_eq!(
                *output.values,
                values
                    .values
                    .iter()
                    .map(|value| value * -2.0)
                    .collect::<Vec<_>>()
            );
        }
        routed.finish_invocation(true).unwrap();
    };
    bank.with_prefill_observer(&mut backend, bound, &Error::Capture, |o| {
        for index in 0..2 {
            enter(o, index);
            if schedule.prefill && schedule.first_prediction == 0 {
                invoke(o, (3 - index * 2).min(2), index as f32 * 200.0);
            } else {
                // No provider callback means no grouped native graph, error slot,
                // scalar validation, or source completion is needed in this span.
                assert!(o.routed_unit_observer("route").unwrap().is_none());
            }
            commit(o, index);
        }
        o.finish_prefill(true);
    })
    .unwrap();
    let prefill = bank.take_shared_step().unwrap().unwrap();
    let prefill_active = schedule.prefill && schedule.first_prediction == 0;
    assert_eq!(
        prefill.interventions()[0].outcome,
        if prefill_active {
            InterventionOutcome::Applied
        } else {
            InterventionOutcome::Inactive
        }
    );
    if prefill_active {
        assert_eq!(
            prefill.interventions()[0]
                .routed_units
                .unwrap()
                .completed_tokens,
            3
        );
        assert_eq!(backend.ranges, [[0, 1], [1, 2], [2, 3]]);
    } else {
        assert!(backend.ranges.is_empty());
    }
    let mut expected_charges = usize::from(prefill_active);
    assert_eq!(backend.charges.get(), expected_charges);
    let mut deliveries = vec![prefill];
    for prediction in 1..3 {
        let active = prediction >= schedule.first_prediction;
        bank.with_observer(&mut backend, prediction, &Error::Capture, |o| {
            o.prepare_transaction(epoch(prediction + 1), ExpertPass::Decode)
                .unwrap();
            if active {
                invoke(o, 1, prediction as f32 * 400.0);
            } else {
                assert!(o.routed_unit_observer("route").unwrap().is_none());
            }
            o.complete_transaction(epoch(prediction + 1)).unwrap();
            o.finish_transaction(epoch(prediction + 1), true);
        })
        .unwrap();
        let decode = bank.take_shared_step().unwrap().unwrap();
        assert_eq!(
            decode.interventions()[0].outcome,
            if active {
                InterventionOutcome::Applied
            } else {
                InterventionOutcome::Inactive
            }
        );
        if active {
            assert_eq!(
                decode.interventions()[0]
                    .routed_units
                    .unwrap()
                    .completed_tokens,
                1
            );
            expected_charges += 1;
        }
        assert_eq!(backend.charges.get(), expected_charges);
        deliveries.push(decode);
    }
    backend.inner.scope.certify().unwrap();
    drop((backend.source, backend.funding, bank, run, reservation));
    assert!(pool.payload_used_bytes().unwrap() > 0);
    drop(deliveries);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}
