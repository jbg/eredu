use super::*;
use eredu_core::intervention::*;

fn sparse<'a>(step: &'a CapturedStep, path: &str) -> &'a RoutedUnitCapture {
    let record = step.records.iter().find(|r| r.path == path).unwrap();
    assert_eq!(record.outcome, CaptureOutcome::Captured);
    let Some(CapturePayload::RoutedUnits(payload)) = &record.payload else {
        panic!("sparse payload")
    };
    payload
}
fn row_values(row: &RoutedUnitCaptureRow) -> &[f32] {
    let eredu_core::TensorObservationData::F32(values) = row.values.data() else {
        panic!()
    };
    values
}
fn payloads(steps: &[CapturedStep]) -> Vec<Vec<Option<CapturePayload>>> {
    steps
        .iter()
        .map(|s| s.records.iter().map(|r| r.payload.clone()).collect())
        .collect()
}

fn verify(device: LocalDevice) {
    let root = routed_fixture();
    // A genuinely sparse top-k subset; parameter geometry is unchanged.
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.0.join("config.json")).unwrap()).unwrap();
    config["num_experts_per_tok"] = 2.into();
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let graph = inspect_architecture(&root.0).unwrap();
    let usage = CaptureUsage {
        captures: 1024,
        retained_bytes: 256 << 20,
        host_bytes: 64 << 20,
        encoded_bytes: 64 << 20,
    };
    for residency in [
        eredu_core::ResidencyPlan::FullyResident,
        eredu_core::ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        },
        eredu_core::ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 8 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency);
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let discovery = model.intervention_discovery().unwrap();
        for group in &graph.routed_components {
            let point = discovery
                .points
                .iter()
                .find(|p| p.path == group.activation)
                .unwrap();
            assert_eq!(
                point.prefill,
                eredu_core::ObservationSupportStatus::Supported
            );
            assert_eq!(
                point.decode,
                eredu_core::ObservationSupportStatus::Supported
            );
            assert_eq!(
                point
                    .routed_units
                    .as_ref()
                    .unwrap()
                    .geometry
                    .routes_per_token,
                2
            );
            assert_eq!(
                group
                    .component_index(&RoutedComponentId {
                        group: group.id.clone(),
                        expert: 2,
                        index: 3
                    })
                    .unwrap(),
                15
            );
        }
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user","content":"left right"})],
                add_generation_prompt: true,
                ..Default::default()
            })
            .unwrap();
        let prefix: Vec<_> = (0..67).map(|i| 1 + i % 7).collect();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(3),
                ..Default::default()
            },
            seed: 17,
            ..Default::default()
        };
        let mut selections = vec![];
        for group in &graph.routed_components {
            for path in [&group.activation, &group.effective_activation] {
                selections.push(CaptureSelection {
                    id: path.clone(),
                    path: path.clone(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::RoutedUnits,
                });
            }
        }
        selections.push(CaptureSelection {
            id: "logits".into(),
            path: "model.logits".into(),
            schedule: Default::default(),
            slices: vec![],
            transform: CaptureTransform::FullTensor,
        });
        let capture = CapturePlan {
            schema_version: 1,
            selections,
            limits: CaptureLimits {
                per_step: usage,
                cumulative: usage,
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        };
        let trace = TraceLimits {
            per_record_bytes: 1 << 20,
            total_bytes: 16 << 20,
        };
        let mut baseline = None;
        let mut keep_plan = None;
        let mut keep_payloads = None;
        for trial in 0..9 {
            model.reset().unwrap();
            let mut operations = vec![];
            if trial != 0 {
                for group in &graph.routed_components {
                    for prefill in [true, false] {
                        let mut schedule = CaptureSchedule::default();
                        schedule.prefill = prefill;
                        schedule.decode = !prefill;
                        let slices = if prefill {
                            vec![CaptureSlice {
                                axis: "token".into(),
                                start: 1,
                                end: 67,
                                stride: 3,
                            }]
                        } else {
                            vec![]
                        };
                        let count = if prefill { 22 } else { 1 };
                        let payload = || InterventionTensor {
                            shape: vec![count, 24],
                            values: InterventionValues::Float32(
                                (0..count * 24).map(|i| (i + 1) as f32 * 0.001).collect(),
                            ),
                        };
                        let action = if trial == 4 || trial == 8 {
                            InterventionAction::Scale {
                                dtype: InterventionDtype::Float32,
                                factor: -0.5,
                            }
                        } else if trial == 5 {
                            InterventionAction::Replace { tensor: payload() }
                        } else if trial == 6 {
                            InterventionAction::Add { tensor: payload() }
                        } else if trial == 7 {
                            InterventionAction::Mask {
                                dtype: InterventionDtype::Float32,
                                shape: vec![count, 24],
                                keep: (0..count * 24)
                                    .map(|i| (i / 24 + i % 24) % 3 != 0)
                                    .collect(),
                            }
                        } else {
                            InterventionAction::MaskComponents {
                                dtype: InterventionDtype::Float32,
                                indices: if trial == 1 {
                                    (0..24).collect()
                                } else {
                                    vec![8, 15]
                                },
                                keep_selected: trial != 2,
                            }
                        };
                        operations.push(InterventionOperation {
                            id: format!("{}:{prefill}", group.id),
                            target: group.activation.clone(),
                            schedule: schedule.clone(),
                            slices: slices.clone(),
                            action,
                            evidence: InterventionEvidence::None,
                        });
                        if trial == 8 {
                            operations.push(InterventionOperation {
                                id: format!("{}:{prefill}:add", group.id),
                                target: group.activation.clone(),
                                schedule,
                                slices,
                                action: InterventionAction::Add { tensor: payload() },
                                evidence: InterventionEvidence::None,
                            });
                        }
                    }
                }
            }
            let plan = InterventionPlan {
                schema_version: 1,
                operations,
            };
            let prepared = model
                .prepare_intervened_token_ids(
                    &chat,
                    prefix.clone(),
                    settings,
                    capture.clone(),
                    plan.clone(),
                    trace,
                )
                .unwrap();
            let mut steps = vec![];
            model
                .generate_observed_text(prepared, &[], Default::default(), |record| {
                    if let ObservedGenerationEvent::Token {
                        forced,
                        captures: Some(step),
                        ..
                    } = record.event
                    {
                        assert!(!forced);
                        steps.push(step);
                    }
                    ControlFlow::Continue(())
                })
                .unwrap();
            assert_eq!(steps.len(), 3);
            for step in &steps {
                for group in &graph.routed_components {
                    let before = sparse(step, &group.activation);
                    let after = sparse(step, &group.effective_activation);
                    assert_eq!(
                        before.rows.len(),
                        if step.prediction_index == 0 { 134 } else { 2 }
                    );
                    let mut affected = 0;
                    for (a, b) in before.rows.iter().zip(&after.rows) {
                        assert_eq!(
                            (a.token, a.slot, a.expert, a.coefficient),
                            (b.token, b.slot, b.expert, b.coefficient)
                        );
                        for (unit, (&original, &effective)) in
                            row_values(a).iter().zip(row_values(b)).enumerate()
                        {
                            let selected_token = step.prediction_index != 0
                                || (a.token >= 1 && (a.token - 1) % 3 == 0);
                            let component = a.expert * 6 + unit as u64;
                            let selected_row = if step.prediction_index == 0 {
                                a.token.saturating_sub(1) / 3
                            } else {
                                0
                            };
                            let delta = (selected_row * 24 + component + 1) as f32 * 0.001;
                            let selected_unit = [8, 15].contains(&component);
                            let removed = selected_token
                                && ((trial == 2 && selected_unit)
                                    || (trial == 3 && !selected_unit)
                                    || (trial == 7 && (selected_row + component) % 3 == 0));
                            let scaled = selected_token && (trial == 4 || trial == 8);
                            affected += u64::from(
                                removed || scaled || (selected_token && (trial == 5 || trial == 6)),
                            );
                            let expected = if removed {
                                0.0
                            } else if selected_token && trial == 5 {
                                delta
                            } else if selected_token && trial == 6 {
                                original + delta
                            } else if selected_token && trial == 8 {
                                original * -0.5 + delta
                            } else if scaled {
                                original * -0.5
                            } else {
                                original
                            };
                            assert_eq!(
                                effective, expected,
                                "trial={trial} prediction={} token={} expert={} unit={unit}",
                                step.prediction_index, a.token, a.expert
                            );
                        }
                    }
                    if trial != 0 {
                        let record = step
                            .interventions
                            .iter()
                            .find(|r| {
                                r.target == group.activation
                                    && r.outcome != InterventionOutcome::Inactive
                            })
                            .unwrap();
                        assert_eq!(record.routed_units.unwrap().affected_values, affected);
                        assert_eq!(
                            record.outcome,
                            if affected == 0 {
                                InterventionOutcome::Unmatched
                            } else {
                                InterventionOutcome::Applied
                            }
                        );
                    }
                }
            }
            if trial == 0 {
                baseline = Some(payloads(&steps));
            }
            if trial == 1 {
                assert_eq!(payloads(&steps), *baseline.as_ref().unwrap());
            }
            if trial == 3 {
                let baseline = baseline.as_ref().unwrap();
                assert_ne!(payloads(&steps)[0].last(), baseline[0].last());
                // Layer one's units are recomputed from the changed residual.
                assert_ne!(payloads(&steps)[0][2], baseline[0][2]);
                keep_payloads = Some(payloads(&steps));
                keep_plan = Some(plan);
            }
        }
        // Same admitted path under explicit advancement, replay and isolated fork.
        model.reset().unwrap();
        let prepared = model
            .prepare_intervened_token_ids(
                &chat,
                prefix,
                settings,
                capture.clone(),
                keep_plan.unwrap(),
                trace,
            )
            .unwrap();
        let mut steps = vec![];
        {
            let mut emit = |record: ControlledGenerationRecord| {
                if let ObservedGenerationEvent::Token {
                    forced,
                    captures: Some(step),
                    ..
                } = record.generation.event
                {
                    assert!(!forced);
                    steps.push(step);
                }
                ControlFlow::Continue(())
            };
            let mut run = model
                .start_controlled_text(prepared, &[], Default::default(), &mut emit)
                .unwrap();
            run.enable_snapshots(SnapshotLimits {
                max_snapshots: 2,
                max_branches: 1,
                retained_bytes: 512 << 20,
                cumulative_copy_bytes: 2 << 30,
            })
            .unwrap();
            let initial = run.snapshot(&mut emit).unwrap();
            run.run(&mut emit).unwrap();
            run.restore(&initial, &mut emit).unwrap();
            run.run(&mut emit).unwrap();
            let mut child = run
                .fork(
                    &initial,
                    GenerationBranchOptions {
                        trace_limits: TraceLimits {
                            per_record_bytes: 1 << 20,
                            total_bytes: 4 << 20,
                        },
                        capture_limits: Some(capture.limits),
                        sampling: None,
                        intervention: None,
                    },
                    &mut emit,
                )
                .unwrap();
            run.exchange(&mut child, &mut emit).unwrap();
            run.run(&mut emit).unwrap();
            run.exchange(&mut child, &mut emit).unwrap();
        }
        assert_eq!(steps.len(), 9);
        for trial in steps.chunks(3) {
            assert_eq!(payloads(trial), *keep_payloads.as_ref().unwrap());
        }
        assert!(steps[3].cumulative_usage.host_bytes > steps[0].cumulative_usage.host_bytes);
    }
}

#[test]
#[ignore = "requires native CPU execution"]
fn native_sparse_unit_interventions_cpu() {
    verify(LocalDevice::Cpu);
}
#[test]
#[cfg(feature = "metal")]
#[ignore = "requires Metal execution"]
fn native_sparse_unit_interventions_metal() {
    verify(LocalDevice::Accelerator(0));
}
