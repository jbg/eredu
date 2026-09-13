use super::*;
use eredu_core::{component::ComponentActivation, intervention::*, TensorObservationData};

fn public_internal_activations(device: LocalDevice, pooling: bool) {
    public_internal_activations_profile(device, pooling, false);
}

fn public_internal_activations_profile(device: LocalDevice, pooling: bool, fused: bool) {
    for residency in super::super::v3_components::residencies() {
        let root = if pooling {
            super::pooling::source(fused)
        } else {
            super::super::v3_components::source_with_prediction(true, true, 2)
        };
        let graph = inspect_architecture(&root.0).unwrap();
        let scope = &graph.component_scopes[0];
        let channel = scope
            .components
            .iter()
            .find(|group| {
                matches!(
                    group.activation_equation,
                    ComponentActivation::Attention { .. }
                )
            })
            .unwrap();
        let mut paths = vec![
            channel.activation.clone(),
            channel.effective_activation.clone(),
            scope.readout.normalized.clone(),
            scope.readout.logits.clone(),
            eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
        ];
        if fused {
            paths.extend([
                "dspark.context.normalized".into(),
                "dspark.context.normalized.effective".into(),
                scope.readout.score_writes[0].output.clone(),
                scope.readout.score_writes[0].effective_output.clone(),
            ]);
        }
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency.clone())
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
        let mut loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap();
        let generation = loaded.speculative_generation_options().unwrap().unwrap();
        let (model, _) = loaded.parts_mut();
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":"left right"})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(5),
                temperature: Some(0.0),
                ..Default::default()
            },
            ..Default::default()
        };
        let request = || PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, vec![1, 2, 5]),
            drafting: eredu_core::SpeculativeDraft::Embedded,
            settings,
            options: generation.clone(),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        };
        let baseline = model.generate_prepared_text_speculative(request()).unwrap();
        let mut ordinary_records = Vec::new();
        for mask in [false, true] {
            let per_step = CaptureUsage {
                captures: 128,
                retained_bytes: 8 << 20,
                host_bytes: 8 << 20,
                encoded_bytes: 8 << 20,
            };
            let admitted = model
                .prepare_speculative_activations(SpeculativeActivationPlan {
                    schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                    bounds: CaptureInvocationBounds {
                        batch: 1,
                        max_sequence: 3,
                        max_context: None,
                        max_predictions: 5,
                    },
                    captures: CapturePlan {
                        schema_version: CAPTURE_SCHEMA_VERSION,
                        selections: paths
                            .iter()
                            .enumerate()
                            .map(|(index, path)| CaptureSelection {
                                id: index.to_string(),
                                path: path.clone(),
                                schedule: Default::default(),
                                slices: vec![],
                                transform: CaptureTransform::Preview { max_elements: 2048 },
                            })
                            .collect(),
                        limits: CaptureLimits {
                            per_step,
                            cumulative: per_step.checked_mul(8).unwrap(),
                            physical_native_bytes: None,
                            on_limit: CaptureLimitPolicy::Fail,
                        },
                    },
                    interventions: InterventionPlan {
                        schema_version: INTERVENTION_SCHEMA_VERSION,
                        operations: if mask {
                            vec![InterventionOperation {
                                id: "mask-prediction-channel".into(),
                                target: channel.activation.clone(),
                                schedule: Default::default(),
                                slices: vec![],
                                action: InterventionAction::MaskComponents {
                                    dtype: InterventionDtype::Float32,
                                    indices: vec![0],
                                    keep_selected: false,
                                },
                                evidence: InterventionEvidence::Preview { max_elements: 2048 },
                            }]
                        } else {
                            vec![]
                        },
                    },
                })
                .unwrap();
            let identity = admitted.identity().to_owned();
            let mut alternate_edits = admitted.interventions().plan().clone();
            if mask {
                alternate_edits.operations.clear();
            } else {
                alternate_edits.operations.push(InterventionOperation {
                    id: "mask-prediction-channel".into(),
                    target: channel.activation.clone(),
                    schedule: Default::default(),
                    slices: vec![],
                    action: InterventionAction::MaskComponents {
                        dtype: InterventionDtype::Float32,
                        indices: vec![0],
                        keep_selected: false,
                    },
                    evidence: InterventionEvidence::Preview { max_elements: 2048 },
                });
            }
            let alternate = model
                .prepare_speculative_activations(SpeculativeActivationPlan {
                    schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                    captures: admitted.captures().plan().clone(),
                    interventions: alternate_edits,
                    bounds: admitted.captures().invocation_bounds().unwrap(),
                })
                .unwrap();
            let mut stale_discovery = model.speculative_activation_discovery().unwrap();
            stale_discovery
                .execution_identity
                .push_str("-different-execution");
            let stale = SpeculativeActivationPlan {
                schema_version: SPECULATIVE_ACTIVATION_SCHEMA_VERSION,
                captures: alternate.captures().plan().clone(),
                interventions: alternate.interventions().plan().clone(),
                bounds: alternate.captures().invocation_bounds().unwrap(),
            }
            .admit(&stale_discovery)
            .unwrap();
            let mut continuous = Vec::new();
            let output = model
                .generate_observed_text_speculative(
                    request(),
                    ControlledSpeculativeOptions {
                        activations: Some(admitted.clone()),
                        ..Default::default()
                    },
                    |step| {
                        continuous.extend(step.activations);
                        ControlFlow::Continue(())
                    },
                )
                .unwrap();
            assert_eq!(output.token_ids(), baseline.token_ids());
            assert!(continuous
                .iter()
                .any(|r| r.phase == SpeculativeActivationPhase::Verification));
            assert_eq!(continuous[0].captures.invocation.unwrap().sequence, 3);
            if fused {
                assert!(continuous
                    .iter()
                    .any(|record| record.phase == SpeculativeActivationPhase::FusedProposal));
                assert!(continuous
                    .iter()
                    .any(|record| record.phase == SpeculativeActivationPhase::PredictionPrefill));
                for record in &continuous {
                    for captured in &record.captures.records {
                        let context = captured.path == "dspark.context.normalized";
                        let proposal = captured.path == channel.activation;
                        if context || proposal {
                            let invoked = if context {
                                matches!(
                                    record.phase,
                                    SpeculativeActivationPhase::PredictionPrefill
                                        | SpeculativeActivationPhase::PredictionReplay
                                )
                            } else {
                                record.phase == SpeculativeActivationPhase::FusedProposal
                            };
                            assert_eq!(
                                captured.outcome == CaptureOutcome::Captured,
                                invoked,
                                "{} in {:?}",
                                captured.path,
                                record.phase
                            );
                        }
                    }
                }
            }
            let mut controlled = Vec::new();
            let output =
                model
                    .with_controlled_text_speculative(
                        request(),
                        ControlledSpeculativeOptions {
                            activations: Some(admitted),
                            snapshots: Some(SnapshotLimits {
                                max_snapshots: 2,
                                max_branches: 2,
                                retained_bytes: 64 << 20,
                                cumulative_copy_bytes: 512 << 20,
                            }),
                            ..Default::default()
                        },
                        |session| {
                            assert!(session
                                .readmit_activation_interventions(stale.clone())
                                .is_err());
                            controlled.extend(session.step()?.unwrap().activations);
                            assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                            let saved = session.snapshot()?;
                            let start = controlled.len();
                            while let Some(step) = session.step()? {
                                controlled.extend(step.activations);
                            }
                            let expected_tokens = session.token_ids().to_vec();
                            let mut last_invocation = controlled.last().unwrap().invocation;
                            for _ in 0..2 {
                                let usage = session.snapshot_usage();
                                session.restore(&saved)?;
                                assert!(
                                    session.snapshot_usage().cumulative_copy_bytes
                                        > usage.cumulative_copy_bytes
                                );
                                let mut replay = Vec::new();
                                while let Some(step) = session.step()? {
                                    replay.extend(step.activations);
                                }
                                assert_eq!(session.token_ids(), expected_tokens);
                                assert_eq!(replay.len(), controlled.len() - start);
                                for (a, b) in replay.iter().zip(&controlled[start..]) {
                                    assert!(a.invocation > last_invocation);
                                    last_invocation = a.invocation;
                                    assert_eq!(a.admission_identity, b.admission_identity);
                                    assert_eq!((a.origin, a.phase), (b.origin, b.phase));
                                    assert_eq!(a.captures.records, b.captures.records);
                                    assert_eq!(a.captures.interventions, b.captures.interventions);
                                    assert!(
                                        a.captures.cumulative_usage.encoded_bytes
                                            > b.captures.cumulative_usage.encoded_bytes
                                    );
                                }
                            }
                            let left = session.fork(&saved)?;
                            let right = session.fork(&saved)?;
                            session.exchange(&left)?;
                            session.readmit_activation_interventions(alternate.clone())?;
                            let different = (expected_tokens[session.token_ids().len()] + 1) % 64;
                            session.force_next_token(different)?;
                            let changed = session.snapshot()?;
                            let mut child_records = Vec::new();
                            while let Some(step) = session.step()? {
                                child_records.extend(step.activations);
                            }
                            assert_ne!(session.token_ids(), expected_tokens);
                            let child_tokens = session.token_ids().to_vec();
                            assert!(child_records
                                .iter()
                                .all(|r| r.admission_identity.as_deref()
                                    == Some(alternate.identity())));
                            if mask {
                                assert!(child_records
                                    .iter()
                                    .all(|r| r.captures.interventions.is_empty()));
                                for record in &child_records {
                                    let payload = |path: &str| {
                                        record
                                            .captures
                                            .records
                                            .iter()
                                            .find(|r| r.path == path)
                                            .and_then(|r| r.payload.as_ref())
                                    };
                                    if let Some(before) = payload(&channel.activation) {
                                        assert_eq!(
                                            Some(before),
                                            payload(&channel.effective_activation)
                                        );
                                    }
                                }
                            } else {
                                let applied: Vec<_> = child_records
                                    .iter()
                                    .flat_map(|r| &r.captures.interventions)
                                    .filter(|e| e.outcome == InterventionOutcome::Applied)
                                    .collect();
                                assert!(!applied.is_empty());
                                for edit in applied {
                                    let values = |record: &CaptureRecord| {
                                        let Some(CapturePayload::Tensor(tensor)) = &record.payload
                                        else {
                                            panic!("missing edited evidence")
                                        };
                                        let TensorObservationData::F32(values) = tensor.data()
                                        else {
                                            panic!("wrong dtype")
                                        };
                                        values.clone()
                                    };
                                    for (index, (before, after)) in values(&edit.evidence[0])
                                        .iter()
                                        .zip(values(&edit.evidence[1]))
                                        .enumerate()
                                    {
                                        assert_eq!(
                                            after,
                                            if index % channel.count == 0 {
                                                0.0
                                            } else {
                                                *before
                                            }
                                        );
                                    }
                                }
                            }
                            session.restore(&changed)?;
                            let mut replay_records = Vec::new();
                            while let Some(step) = session.step()? {
                                replay_records.extend(step.activations);
                            }
                            assert_eq!(session.token_ids(), child_tokens);
                            assert_eq!(child_records.len(), replay_records.len());
                            for (a, b) in child_records.iter().zip(&replay_records) {
                                assert_eq!(a.admission_identity, b.admission_identity);
                                assert_eq!(a.captures.records, b.captures.records);
                                assert_eq!(a.captures.interventions, b.captures.interventions);
                                assert!(b.invocation > a.invocation);
                            }
                            session.release_snapshot(&changed)?;
                            session.exchange(&left)?;
                            assert_eq!(session.token_ids(), expected_tokens);
                            session.exchange(&right)?;
                            while let Some(step) = session.step()? {
                                assert!(step
                                    .activations
                                    .iter()
                                    .all(|r| r.admission_identity.as_deref()
                                        == Some(identity.as_str())));
                            }
                            assert_eq!(session.token_ids(), expected_tokens);
                            session.exchange(&right)?;
                            assert_eq!(session.token_ids(), expected_tokens);
                            session.release_branch(&left)?;
                            session.release_branch(&right)?;
                            session.release_snapshot(&saved)?;
                            Ok(())
                        },
                    )
                    .unwrap();
            assert_eq!(output.token_ids(), baseline.token_ids());
            assert_eq!(continuous.len(), controlled.len());
            for (a, b) in continuous.iter().zip(&controlled) {
                assert_eq!(a.admission_identity.as_deref(), Some(identity.as_str()));
                assert_eq!((a.origin, a.phase), (b.origin, b.phase));
                assert_eq!(a.captures.records, b.captures.records);
                assert_eq!(a.captures.interventions, b.captures.interventions);
                assert!(a.completed && b.completed);
            }
            if mask {
                let logits = |records: &[SpeculativeActivationCapture]| -> Vec<_> {
                    records
                        .iter()
                        .flat_map(|r| &r.captures.records)
                        .filter(|r| r.path == scope.readout.logits)
                        .filter_map(|r| r.payload.clone())
                        .collect()
                };
                assert_ne!(logits(&continuous), logits(&ordinary_records));
                for edit in continuous
                    .iter()
                    .flat_map(|r| &r.captures.interventions)
                    .filter(|e| e.outcome == InterventionOutcome::Applied)
                {
                    let values = |record: &CaptureRecord| {
                        let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
                            panic!("missing evidence");
                        };
                        let TensorObservationData::F32(values) = tensor.data() else {
                            panic!("wrong dtype");
                        };
                        values.clone()
                    };
                    let before = values(&edit.evidence[0]);
                    let after = values(&edit.evidence[1]);
                    assert!(before.iter().any(|v| v.abs() > 1e-5));
                    for (index, (a, b)) in before.iter().zip(&after).enumerate() {
                        assert_eq!(*b, if index % channel.count == 0 { 0.0 } else { *a });
                    }
                }
            } else {
                ordinary_records = continuous;
            }
            let restored = model.generate_prepared_text_speculative(request()).unwrap();
            assert_eq!(restored.token_ids(), baseline.token_ids());
        }
    }
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn public_v3_internal_activations_match_continuous_and_controlled_cpu() {
    public_internal_activations(LocalDevice::Cpu, false);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn public_v3_internal_activations_match_continuous_and_controlled_metal() {
    public_internal_activations(LocalDevice::Accelerator(0), false);
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn public_v4_internal_activations_match_continuous_and_controlled_cpu() {
    public_internal_activations(LocalDevice::Cpu, true);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn public_v4_internal_activations_match_continuous_and_controlled_metal() {
    public_internal_activations(LocalDevice::Accelerator(0), true);
}

#[test]
#[ignore = "requires native MLX CPU execution; run explicitly"]
fn public_dspark_internal_activations_match_continuous_and_controlled_cpu() {
    public_internal_activations_profile(LocalDevice::Cpu, true, true);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "requires native MLX Metal execution; run explicitly"]
fn public_dspark_internal_activations_match_continuous_and_controlled_metal() {
    public_internal_activations_profile(LocalDevice::Accelerator(0), true, true);
}
