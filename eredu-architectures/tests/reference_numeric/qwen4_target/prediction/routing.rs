//! Pre-dispatch controls on the actual target/MTP pair with attributed decisions.
use super::*;
fn tensor<'a>(records: &'a [CaptureRecord], path: &str) -> &'a eredu_core::TensorObservation {
    let Some(CapturePayload::Tensor(value)) =
        &records.iter().find(|r| r.path == path).unwrap().payload
    else {
        panic!("missing {path}")
    };
    value
}
fn ids(value: &eredu_core::TensorObservation) -> &[i64] {
    let eredu_core::TensorObservationData::I64(values) = value.data() else {
        panic!()
    };
    values
}
fn floats(value: &eredu_core::TensorObservation) -> &[f32] {
    let eredu_core::TensorObservationData::F32(values) = value.data() else {
        panic!()
    };
    values
}
fn request(
    discovery: &SpeculativeActivationDiscovery,
    root: &str,
    action: InterventionAction,
) -> SpeculativeActivationPlan {
    let mut request = plan(discovery);
    request
        .interventions
        .operations
        .push(InterventionOperation {
            id: "route-edit".into(),
            target: root.into(),
            schedule: CaptureSchedule::default(),
            slices: vec![CaptureSlice {
                axis: "token".into(),
                start: 0,
                end: 1,
                stride: 1,
            }],
            action,
            evidence: InterventionEvidence::Preview { max_elements: 2 },
        });
    request
}
pub(super) fn checks(
    selected: &Selected,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    discovery: &SpeculativeActivationDiscovery,
    plain: &Outcome,
) {
    use eredu_core::RoutingObservationField as F;
    for root in ["model.layers.0.mlp", "mtp.layers.0.mlp", "mtp.layers.1.mlp"] {
        let point = discovery
            .interventions
            .points
            .iter()
            .find(|p| p.path == root)
            .unwrap();
        let policy = point.routing.as_ref().unwrap();
        assert_eq!(
            (
                policy.expert_count,
                policy.top_k,
                policy.shared_experts,
                policy.learned_coefficient_scale
            ),
            (3, 2, 1, false)
        );
        let scope = discovery
            .bindings
            .iter()
            .find(|b| b.node_id == point.node_id)
            .unwrap()
            .scope;
        let identity = InterventionAction::BiasRoutingScores {
            stage: RoutingScoreStage::RawLogits,
            expert_ids: vec![0],
            biases: vec![0.],
        };
        let mut unavailable = discovery.clone();
        unavailable
            .interventions
            .points
            .iter_mut()
            .find(|p| p.path == root)
            .unwrap()
            .operations
            .clear();
        assert!(request(&unavailable, root, identity.clone())
            .admit(&unavailable)
            .is_err());
        for action in [
            identity.clone(),
            InterventionAction::ZeroExpertContribution {
                expert_ids: vec![0, 1, 2],
            },
            InterventionAction::ExcludeExperts {
                expert_ids: vec![0],
            },
            InterventionAction::ForceExperts {
                shape: [1, 2],
                expert_ids: vec![2, 0],
            },
            InterventionAction::BiasRoutingScores {
                stage: RoutingScoreStage::RawLogits,
                expert_ids: vec![0],
                biases: vec![3.],
            },
            InterventionAction::BiasRoutingScores {
                stage: RoutingScoreStage::TransformedScores,
                expert_ids: vec![0],
                biases: vec![3.],
            },
            InterventionAction::BiasRoutingScores {
                stage: RoutingScoreStage::RankingScores,
                expert_ids: vec![0],
                biases: vec![3.],
            },
        ] {
            let is_identity = action == identity;
            let is_force = matches!(action, InterventionAction::ForceExperts { .. });
            let request = request(discovery, root, action.clone());
            let admitted = request.admit(discovery).unwrap();
            let mut options = super::super::options();
            options.activations = Some(admitted);
            let mut applied = 0;
            let result = run(selected, prepared, ctx, options, |session| {
                let mut saved = None;
                let mut initial_count = 0;
                let mut original = vec![];
                while let Some(step) = session.step()? {
                    for invocation in &step.activations {
                        let edit = &invocation.captures.interventions[0];
                        if !scope.applies(invocation.phase) {
                            assert_eq!(edit.outcome, InterventionOutcome::Inactive);
                            continue;
                        }
                        applied += 1;
                        assert_eq!(edit.outcome, InterventionOutcome::Applied);
                        assert_eq!(edit.evidence.len(), 4);
                        assert!(edit
                            .evidence
                            .iter()
                            .all(|r| r.outcome == CaptureOutcome::Captured));
                        let value = |index: usize| {
                            let Some(CapturePayload::Tensor(value)) = &edit.evidence[index].payload
                            else {
                                panic!()
                            };
                            value
                        };
                        let (original, coefficients) = (ids(value(0)), floats(value(1)));
                        let (effective, weights) = (ids(value(2)), floats(value(3)));
                        assert_eq!(
                            effective,
                            ids(tensor(
                                &invocation.captures.records,
                                &F::SelectedExperts.path(root)
                            ))
                        );
                        assert_eq!(
                            weights,
                            floats(tensor(
                                &invocation.captures.records,
                                &F::Coefficients.path(root)
                            ))
                        );
                        assert_eq!(value(0).shape(), &[2]);
                        assert!(original.iter().all(|id| (0..3).contains(id)));
                        let Some(CapturePayload::RoutedUnits(units)) = &invocation
                            .captures
                            .records
                            .iter()
                            .find(|r| r.path == format!("{root}.units"))
                            .unwrap()
                            .payload
                        else {
                            panic!()
                        };
                        for (slot, row) in units.rows.iter().enumerate() {
                            assert_eq!(row.expert as i64, effective[slot]);
                            assert_eq!(row.coefficient, weights[slot]);
                        }
                        if is_identity {
                            assert_eq!(original, effective);
                            assert_eq!(coefficients, weights);
                        } else {
                            match &action {
                                InterventionAction::ZeroExpertContribution { .. } => {
                                    assert_eq!(original, effective);
                                    assert!(coefficients.iter().all(|v| *v > 0.));
                                    assert_eq!(weights, &[0., 0.]);
                                    let out = floats(tensor(
                                        &invocation.captures.records,
                                        &F::RoutedOutput.path(root),
                                    ));
                                    assert!(out[..2].iter().all(|v| *v == 0.));
                                    // Shared path continues even when the routed contribution is zero.
                                    let shared = floats(tensor(
                                        &invocation.captures.records,
                                        &F::SharedOutput.path(root),
                                    ));
                                    assert!(shared[..2].iter().any(|v| *v != 0.));
                                }
                                InterventionAction::ExcludeExperts { .. } => {
                                    assert!(effective.iter().all(|id| *id != 0))
                                }
                                InterventionAction::ForceExperts { .. } => {
                                    assert_eq!(effective, &[2, 0])
                                }
                                InterventionAction::BiasRoutingScores { .. } => {
                                    assert_eq!(effective[0], 0)
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                    if is_force {
                        if saved.is_none() {
                            saved = Some(session.snapshot()?);
                            initial_count = step.activations.len();
                        }
                        original.extend(step.activations);
                    }
                    if !is_identity && !is_force {
                        break;
                    }
                }
                if let Some(saved) = saved {
                    session.restore(&saved)?;
                    let mut replayed = vec![];
                    while let Some(step) = session.step()? {
                        replayed.extend(step.activations);
                    }
                    let expected = &original[initial_count..];
                    assert_eq!(replayed.len(), expected.len());
                    assert!(replayed[0].invocation > original.last().unwrap().invocation);
                    for (a, b) in replayed.iter().zip(expected) {
                        assert_eq!(a.phase, b.phase);
                        assert!(
                            a.captures.cumulative_usage.captures
                                > b.captures.cumulative_usage.captures
                        );
                        for (a, b) in a.captures.records.iter().zip(&b.captures.records) {
                            assert_eq!(
                                (&a.path, &a.outcome, &a.payload),
                                (&b.path, &b.outcome, &b.payload)
                            );
                        }
                        for (a, b) in a
                            .captures
                            .interventions
                            .iter()
                            .zip(&b.captures.interventions)
                        {
                            assert_eq!(a.outcome, b.outcome);
                            for (a, b) in a.evidence.iter().zip(&b.evidence) {
                                assert_eq!(a.payload, b.payload);
                            }
                        }
                    }
                }
                Ok(())
            });
            assert!(applied > 0);
            if is_identity {
                assert_eq!(result.tokens, plain.tokens);
                exact(&result.target, &plain.target, "routing identity target");
                exact(
                    &result.prediction,
                    &plain.prediction,
                    "routing identity prediction",
                );
            }
        }
        for action in [
            InterventionAction::ForceExperts {
                shape: [1, 2],
                expert_ids: vec![0, 0],
            },
            InterventionAction::ForceExperts {
                shape: [1, 2],
                expert_ids: vec![0, 3],
            },
            InterventionAction::ExcludeExperts {
                expert_ids: vec![0, 1],
            },
        ] {
            assert!(request(discovery, root, action).admit(discovery).is_err());
        }
    }
}
