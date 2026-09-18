mod windows;
use super::*;
use crate::{
    capture::{
        CaptureBackendProvider, SpeculativeCaptureObserver, SpeculativeCaptureScope as Scope,
    },
    inspection::{with_speculative_activation, SpeculativeActivationObserver},
    ActivationObserver,
};
use eredu_core::speculative::{
    SpeculativeActivationOrigin, SpeculativeActivationPhase as Phase, SpeculativeControlError,
};

struct Provider {
    fail: bool,
}
impl CaptureBackendProvider for Provider {
    type Tensor = Value;
    type Error = std::io::Error;
    type Backend<'a> = Backend;
    fn backend(&mut self) -> Backend {
        Backend {
            fail_application: self.fail.then_some(1),
            ..Backend::default()
        }
    }
}
fn map(error: &CaptureExecutionError<std::io::Error>) -> String {
    error.to_string()
}
type Observer =
    SpeculativeCaptureObserver<Provider, fn(&CaptureExecutionError<std::io::Error>) -> String>;
fn collector(run: CaptureSession, scope: Scope) -> Observer {
    SpeculativeCaptureObserver::new(
        run,
        Provider { fail: false },
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        vec![scope],
        vec![scope],
    )
    .unwrap()
}
fn origin(prediction: usize) -> SpeculativeActivationOrigin {
    SpeculativeActivationOrigin {
        request: SpeculativeRequestId::new(0),
        committed_tokens: prediction,
        prediction,
        prefix_digest: [7; 32],
        optimistic: false,
    }
}
fn forward(
    observer: &mut Observer,
    phase: Phase,
    sequence: u64,
    transactional: bool,
) -> Result<Value, String> {
    forward_at(
        observer,
        phase,
        sequence,
        transactional,
        DistributedCommitEpoch::FIRST,
    )
}
fn forward_at(
    observer: &mut Observer,
    phase: Phase,
    sequence: u64,
    transactional: bool,
    epoch: DistributedCommitEpoch,
) -> Result<Value, String> {
    with_speculative_activation(Some(observer), phase, sequence as usize, |observer| {
        let observer = observer.unwrap();
        if transactional {
            let pass = match phase {
                Phase::TargetPrefill | Phase::PredictionPrefill => crate::ExpertPass::Prefill,
                _ => crate::ExpertPass::Decode,
            };
            observer.prepare_transaction(epoch, pass)?;
        }
        let value = input(sequence);
        let output = crate::observe_and_intervene(observer, "block.output", &value)?;
        if transactional {
            observer.finish()?;
            observer.complete_transaction(epoch)?;
            observer.finish_transaction(epoch, true);
        }
        Ok(output)
    })
}

#[test]
fn speculative_phases_share_value_evidence_budget_and_monotonic_restore_identity() {
    let (run, discovery, _) = setup(64, false);
    let mut observer = collector(run, Scope::Target);
    let checkpoint = observer.checkpoint(&discovery).unwrap();
    observer.set_activation_origin(Some(origin(0)));
    assert_eq!(
        forward(&mut observer, Phase::TargetPrefill, 3, true)
            .unwrap()
            .data,
        [2., 4., 6., 8., 10., 12.]
    );
    assert_eq!(
        forward(&mut observer, Phase::PredictionPrefill, 2, false)
            .unwrap()
            .data,
        [1., 2., 3., 4.]
    );
    observer.set_activation_origin(Some(origin(1)));
    assert_eq!(
        forward(&mut observer, Phase::Verification, 3, false)
            .unwrap()
            .data,
        [2., 4., 6., 8., 10., 12.]
    );
    observer.set_activation_origin(None);
    assert!(
        observer.checkpoint(&discovery).is_err(),
        "delivery must drain before restore"
    );
    let records: Vec<_> = std::iter::from_fn(|| observer.take_activation_capture()).collect();
    assert_eq!(records.len(), 3);
    for (index, record) in records.iter().enumerate() {
        assert_eq!(record.invocation, index as u64);
        assert!(record.completed);
        assert_eq!(
            record.captures.as_step().invocation.unwrap().sequence,
            [3, 2, 3][index]
        );
        assert_eq!(record.origin.prediction, usize::from(index == 2));
        assert!(record.captures.as_step().step_usage.encoded_bytes >= 1024);
    }
    assert_eq!(records[0].captures.as_step().outcome, CaptureStepOutcome::Committed);
    assert_eq!(
        values(&records[0].captures.as_step().interventions[0].evidence[1]),
        [2., 4., 6., 8., 10., 12.]
    );
    assert_eq!(
        records[1].captures.as_step().records[0].outcome,
        CaptureOutcome::Skipped {
            reason: CaptureSkipReason::NotInvoked
        }
    );
    let usage = observer.session().cumulative_usage();
    observer.restore(&checkpoint).unwrap();
    assert_eq!(observer.session().cumulative_usage(), usage);
    observer.set_activation_origin(Some(origin(1)));
    forward(&mut observer, Phase::Verification, 2, false).unwrap();
    let repeated = observer.take_activation_capture().unwrap();
    assert_eq!(repeated.invocation, 3);
    assert!(repeated.captures.as_step().cumulative_usage.host_bytes > usage.host_bytes);
}

#[test]
fn prediction_depth_scope_disables_unrelated_work_and_preserves_typed_failure() {
    let (run, _, _) = setup(3, false);
    let mut observer = collector(run, Scope::Prediction { depth: 1 });
    let invoked = Cell::new(false);
    assert!(with_speculative_activation(
        Some(&mut observer),
        Phase::Proposal { depth: 1 },
        1,
        |_| {
            invoked.set(true);
            Ok(())
        }
    )
    .is_err());
    assert!(!invoked.get());
    assert!(matches!(
        observer.take_activation_error(),
        Some(SpeculativeControlError::Capture(CaptureError::Invalid(_)))
    ));
    assert!(observer.take_activation_capture().is_none());
    observer.set_activation_origin(Some(origin(1)));
    assert_eq!(
        forward(&mut observer, Phase::Proposal { depth: 0 }, 1, false)
            .unwrap()
            .data,
        [1., 2.]
    );
    let skipped = observer.take_activation_capture().unwrap();
    assert_eq!(skipped.captures.as_step().step_usage.captures, 0);
    assert_eq!(
        forward(&mut observer, Phase::Proposal { depth: 1 }, 1, false)
            .unwrap()
            .data,
        [2., 4.]
    );
    let captured = observer.take_activation_capture().unwrap();
    assert_eq!(captured.captures.as_step().step_usage.captures, 3);
    assert!(forward(&mut observer, Phase::Proposal { depth: 1 }, 1, false).is_err());
    assert!(matches!(
        observer.take_activation_error(),
        Some(SpeculativeControlError::Capture(CaptureError::Limit {
            budget: CaptureBudget::Captures,
            cumulative: true
        }))
    ));
    let failed = observer.take_activation_capture().unwrap();
    assert!(!failed.completed);
    assert_eq!(failed.captures.as_step().outcome, CaptureStepOutcome::Aborted);
    assert_eq!(failed.captures.as_step().cumulative_usage.captures, 3);
    assert!(
        failed.captures.as_step().cumulative_usage.host_bytes > captured.captures.as_step().cumulative_usage.host_bytes
    );
}

#[test]
fn context_and_fused_edits_execute_only_in_their_phases_without_refunding_replay() {
    for scope in [Scope::PredictionContext, Scope::FusedProposal] {
        let (run, discovery, _) = setup(64, false);
        let mut observer = collector(run, scope);
        let checkpoint = observer.checkpoint(&discovery).unwrap();
        for (index, phase) in [
            Phase::PredictionPrefill,
            Phase::FusedProposal,
            Phase::PredictionReplay,
            Phase::Verification,
            Phase::Proposal { depth: 0 },
        ]
        .into_iter()
        .enumerate()
        {
            observer.set_activation_origin(Some(origin(1)));
            let output = forward_at(
                &mut observer,
                phase,
                2,
                true,
                DistributedCommitEpoch::new(index as u64 + 1).unwrap(),
            )
            .unwrap();
            let invoked = scope.applies(phase);
            assert_eq!(
                output.data,
                if invoked {
                    vec![2., 4., 6., 8.]
                } else {
                    vec![1., 2., 3., 4.]
                }
            );
            let record = observer.take_activation_capture().unwrap();
            assert!(record.completed);
            assert_eq!(
                record.captures.as_step().step_usage.captures,
                if invoked { 3 } else { 0 }
            );
            if invoked {
                assert_eq!(
                    values(&record.captures.as_step().interventions[0].evidence[1]),
                    output.data
                );
            } else {
                assert_eq!(
                    record.captures.as_step().records[0].outcome,
                    CaptureOutcome::Skipped {
                        reason: CaptureSkipReason::NotInvoked
                    }
                );
            }
        }
        let usage = observer.session().cumulative_usage();
        observer.set_activation_origin(None);
        observer.restore(&checkpoint).unwrap();
        assert_eq!(observer.session().cumulative_usage(), usage);
        observer.set_activation_origin(Some(origin(1)));
        let phase = if scope == Scope::PredictionContext {
            Phase::PredictionReplay
        } else {
            Phase::FusedProposal
        };
        assert_eq!(
            forward_at(
                &mut observer,
                phase,
                2,
                true,
                DistributedCommitEpoch::new(6).unwrap()
            )
            .unwrap()
            .data,
            [2., 4., 6., 8.]
        );
        let replay = observer.take_activation_capture().unwrap();
        assert!(replay.captures.as_step().cumulative_usage.host_bytes > usage.host_bytes);
    }
}

#[test]
fn missing_intervention_and_execution_failure_remain_aborted_evidence() {
    let (run, _, _) = setup(64, false);
    let mut observer = collector(run, Scope::Target);
    observer.set_activation_origin(Some(origin(1)));
    let _error =
        with_speculative_activation(Some(&mut observer), Phase::Verification, 2, |_| Ok(()))
            .unwrap_err();
    assert!(matches!(
        observer.take_activation_error(),
        Some(SpeculativeControlError::Capture(CaptureError::Invalid(_)))
    ));
    assert!(!observer.take_activation_capture().unwrap().completed);
    assert_eq!(
        with_speculative_activation(Some(&mut observer), Phase::Verification, 2, |o| {
            o.unwrap().observe("block.output", &input(2))?;
            Err::<(), _>("native forward failure".to_owned())
        }),
        Err("native forward failure".to_owned())
    );
    let failed = observer.take_activation_capture().unwrap();
    assert!(!failed.completed);
    assert_eq!(values(&failed.captures.as_step().records[0]), [1., 2., 3., 4.]);
    assert_eq!(failed.captures.as_step().outcome, CaptureStepOutcome::Aborted);
}

#[test]
fn native_transform_failure_keeps_its_original_source_across_string_propagation() {
    use std::error::Error;
    let (run, _, _) = setup(64, false);
    let mut observer = SpeculativeCaptureObserver::new(
        run,
        Provider { fail: true },
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        vec![Scope::Target],
        vec![Scope::Target],
    )
    .unwrap();
    observer.set_activation_origin(Some(origin(1)));
    assert!(forward(&mut observer, Phase::Verification, 2, false).is_err());
    let Some(SpeculativeControlError::Backend(error)) = observer.take_activation_error() else {
        panic!("native failure must preserve its source");
    };
    assert!(error.source().unwrap().is::<std::io::Error>());
    assert!(!observer.take_activation_capture().unwrap().completed);
}

#[test]
fn loaded_activation_authority_constructs_one_shared_run_and_tags_evidence() {
    use eredu_core::speculative::{
        SpeculativeActivationDiscovery, SpeculativeActivationPlan, SpeculativeCaptureBinding,
        SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
    };
    let (run, captures, interventions) = setup(64, false);
    let node = captures.catalog.points[0].node_id.clone();
    let discovery = SpeculativeActivationDiscovery {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        execution_identity: "loaded-execution".into(),
        captures,
        interventions,
        bindings: vec![SpeculativeCaptureBinding {
            node_id: node,
            scope: Scope::Target,
        }],
    };
    let mut raw = SpeculativeActivationPlan {
        schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
        captures: run.plan().plan().clone(),
        interventions: run.intervention_plan().unwrap().plan().clone(),
        bounds: bounds(),
    };
    let admitted = raw.clone().admit(&discovery).unwrap();
    let mut observer = SpeculativeCaptureObserver::from_admitted(
        &admitted,
        Provider { fail: false },
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        Arc::new(Estimates),
    )
    .unwrap()
    .unwrap();
    observer.set_activation_origin(Some(origin(1)));
    assert_eq!(
        forward(&mut observer, Phase::Verification, 2, false)
            .unwrap()
            .data,
        [2., 4., 6., 8.]
    );
    let record = observer.take_activation_capture().unwrap();
    assert_eq!(
        record.admission_identity.as_deref(),
        Some(admitted.identity())
    );
    assert_eq!(record.captures.as_step().step_usage.captures, 3);
    assert!(
        serde_json::to_vec(&record).unwrap().len() as u64
            <= record.captures.as_step().step_usage.encoded_bytes
    );
    observer.set_activation_origin(None);
    let saved = observer.activation_checkpoint().unwrap();
    let before = observer.session().cumulative_usage();
    let mut replacement = raw.clone();
    replacement.interventions.operations.clear();
    let replacement = replacement.admit(&discovery).unwrap();
    observer
        .readmit_activation_interventions(replacement.clone())
        .unwrap();
    assert!(observer.session().cumulative_usage().host_bytes > before.host_bytes);
    // Preparing and dropping a restore changes neither authority nor accounting.
    drop(observer.prepare_activation_restore(&saved).unwrap());
    observer.set_activation_origin(Some(origin(1)));
    assert_eq!(
        forward(&mut observer, Phase::Verification, 2, false)
            .unwrap()
            .data,
        [1., 2., 3., 4.]
    );
    assert!(observer.activation_checkpoint().is_err());
    observer.set_activation_origin(None);
    let unedited = observer.take_activation_capture().unwrap();
    assert_eq!(
        unedited.admission_identity.as_deref(),
        Some(replacement.identity())
    );
    assert!(unedited.captures.as_step().interventions.is_empty());
    observer
        .prepare_activation_restore(&saved)
        .unwrap()
        .commit();
    observer.set_activation_origin(Some(origin(1)));
    assert_eq!(
        forward(&mut observer, Phase::Verification, 2, false)
            .unwrap()
            .data,
        [2., 4., 6., 8.]
    );
    observer.set_activation_origin(None);
    let restored = observer.take_activation_capture().unwrap();
    assert_eq!(
        restored.admission_identity.as_deref(),
        Some(admitted.identity())
    );
    assert!(restored.invocation > unedited.invocation);
    assert!(
        restored.captures.as_step().cumulative_usage.encoded_bytes
            > unedited.captures.as_step().cumulative_usage.encoded_bytes
    );
    let mut foreign = SpeculativeCaptureObserver::from_admitted(
        &admitted,
        Provider { fail: false },
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        Arc::new(Estimates),
    )
    .unwrap()
    .unwrap();
    assert!(foreign.prepare_activation_restore(&saved).is_err());
    raw.captures.selections.clear();
    raw.interventions.operations.clear();
    let empty = raw.admit(&discovery).unwrap();
    let before = observer.session().cumulative_usage();
    assert!(observer
        .readmit_activation_interventions(empty.clone())
        .is_err());
    assert_eq!(observer.session().cumulative_usage(), before);
    assert!(SpeculativeCaptureObserver::from_admitted(
        &empty,
        Provider { fail: true },
        map as fn(&CaptureExecutionError<std::io::Error>) -> String,
        SpeculativeRequestId::new(0),
        Arc::new(Estimates),
    )
    .unwrap()
    .is_none());
}

#[test]
fn default_prefill_intervention_evidence_preserves_complete_target_and_shifted_seed_rows() {
    // A single full span and several spans use the same logical evidence owner.
    for (scope, phase, rows, token_start) in [
        (Scope::Target, Phase::TargetPrefill, 3, 0),
        (
            Scope::Prediction { depth: 0 },
            Phase::PredictionPrefill,
            2,
            1,
        ),
    ] {
        let (run, _, _) = setup(64, false);
        let mut observer = collector(run, scope);
        assert!(observer.requires_sequence_readout());
        observer.set_activation_origin(Some(origin(0)));
        if observer.supports_prefill_spans() {
            observer.set_prefill_reduction_geometry(
                eredu_core::speculative::SpeculativePrefillReductionGeometry {
                    target_sequence: 3,
                    prediction_sequence: 2,
                },
            );
            observer.set_prefill_span(Some(eredu_core::speculative::SpeculativePrefillSpan {
                prompt_tokens: 3,
                input_start: 0,
                input_end: 3,
                position: 0,
                hidden_start: 0,
                token_start,
                sequence: rows,
                seed_start: 0,
            }));
        }
        let output = forward(&mut observer, phase, rows, true).unwrap();
        observer.complete_prefill_reductions().unwrap();
        observer.finish_prefill_reductions(true);
        let expected: Vec<_> = input(rows).data.iter().map(|value| 2.0 * value).collect();
        assert_eq!(output.data, expected);
        let record = observer.take_activation_capture().unwrap();
        assert!(record.completed);
        assert_eq!(record.phase, phase);
        assert_eq!(record.captures.as_step().invocation.unwrap().sequence, rows);
        assert_eq!(values(&record.captures.as_step().records[0]), input(rows).data);
        let evidence = &record.captures.as_step().interventions[0].evidence;
        assert_eq!(values(&evidence[0]), input(rows).data);
        assert_eq!(values(&evidence[1]), expected);
        assert_eq!(record.captures.as_step().step_usage.captures, 3);
        assert!(observer.take_activation_capture().is_none());
        assert!(observer.take_activation_error().is_none());
    }
}

#[test]
fn speculative_span_capability_preserves_evidence_free_and_decode_only_operations() {
    for (evidence, prefill) in [
        (InterventionEvidence::None, true),
        (InterventionEvidence::Preview { max_elements: 64 }, false),
    ] {
        let (original, mut discovery, edits) = setup(64, false);
        let mut raw = original.plan().plan().clone();
        raw.selections[0].transform = CaptureTransform::FullTensor;
        discovery
            .support
            .capture
            .transformations
            .push(CaptureTransformKind::FullTensor);
        let capture = raw
            .admit_invocations(
                &discovery.catalog,
                &discovery.support,
                &discovery.support.capture,
                bounds(),
            )
            .unwrap();
        let mut raw = original.intervention_plan().unwrap().plan().clone();
        raw.operations[0].evidence = evidence;
        raw.operations[0].schedule.prefill = prefill;
        let plan = raw.admit_invocations(&edits, bounds(), "session").unwrap();
        let mut run = CaptureSession::new(eredu_core::capture::SharedCapturePlan::new(capture));
        run.enable_interventions(plan, Arc::new(Estimates)).unwrap();
        let mut observer = collector(run, Scope::Target);
        assert!(observer.supports_prefill_spans());
        observer.set_activation_origin(Some(origin(0)));
        for start in [0, 2] {
            observer.set_prefill_span(Some(eredu_core::speculative::SpeculativePrefillSpan {
                prompt_tokens: 4,
                input_start: start,
                input_end: start + 2,
                position: start,
                hidden_start: start,
                token_start: start,
                sequence: 2,
                seed_start: start,
            }));
            let output = forward_at(
                &mut observer,
                Phase::TargetPrefill,
                2,
                true,
                DistributedCommitEpoch::new(start + 1).unwrap(),
            )
            .unwrap();
            assert_eq!(
                output.data,
                if prefill {
                    vec![2., 4., 6., 8.]
                } else {
                    input(2).data
                }
            );
            let record = observer.take_activation_capture().unwrap();
            assert!(record.completed);
            assert_eq!(record.prefill_span.unwrap().input_start, start);
            assert!(record.captures.as_step().interventions[0]
                .evidence
                .iter()
                .all(|row| row.payload.is_none()));
        }
        assert!(observer.take_activation_error().is_none());
    }
}
