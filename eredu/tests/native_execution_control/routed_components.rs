use super::*;
use eredu_core::{component::*, parameters::*};

#[path = "routed_components/interventions.rs"]
mod interventions;

pub(super) fn routed_fixture() -> Fixture {
    let root = fixture(false);
    let config = serde_json::json!({
        "model_type":"qwen3_moe", "hidden_size":16, "num_hidden_layers":2,
        "intermediate_size":0, "moe_intermediate_size":6, "num_experts":4,
        "num_experts_per_tok":4, "num_attention_heads":4, "num_key_value_heads":2,
        "head_dim":4, "rms_norm_eps":0.00001, "vocab_size":64,
        "eos_token_id":63, "max_position_embeddings":1024,
        "tie_word_embeddings":false, "norm_topk_prob":true
    });
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    write_tensor_plan(&root.0, resolved.architecture.checkpoint());
    root
}

fn verify(device: LocalDevice) {
    let root = routed_fixture();
    let source = std::fs::read(root.0.join("model.safetensors")).unwrap();
    let graph = inspect_architecture(&root.0).unwrap();
    assert_eq!(graph.routed_components.len(), 2);
    let group = &graph.routed_components[0];
    let component = RoutedComponentId {
        group: group.id.clone(),
        expert: 2,
        index: 3,
    };
    let limits = CaptureUsage {
        captures: 1024,
        retained_bytes: 256 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
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
        let baseline = super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0;
        let other_baseline = super::parameters::parameter_logits(&mut model, &[7, 5, 2, 1]).0;
        let facts = model.parameter_discovery().unwrap();
        let gate = group
            .read_weight(&component, ComponentReadRole::Gate, &facts)
            .unwrap();
        let value = group
            .read_weight(&component, ComponentReadRole::Value, &facts)
            .unwrap();
        let write = group.write_column(&component, &facts).unwrap();
        assert_eq!(gate.parameter.id, value.parameter.id);
        assert_eq!(gate.region.starts, [2, 3, 0]);
        assert_eq!(value.region.starts, [2, 9, 0]);
        assert_eq!(write.region.starts, [2, 0, 3]);
        assert!(group.write_bias(&component, &facts).unwrap().is_none());
        let mut originals = std::collections::BTreeMap::new();
        for selection in [&gate, &value, &write] {
            let parameter = selection.parameter;
            let before = model.parameter_discovery().unwrap().usage;
            assert!(matches!(
                model.query_parameter(
                    &facts.identity,
                    &parameter.id,
                    selection.region.clone(),
                    before
                ),
                Err(ParameterError::Budget(_))
            ));
            assert_eq!(model.parameter_discovery().unwrap().usage, before);
            let full = model
                .query_parameter(
                    &facts.identity,
                    &parameter.id,
                    ParameterRegion {
                        starts: vec![0; parameter.shape.len()],
                        shape: parameter.shape.clone(),
                    },
                    limits,
                )
                .unwrap();
            let selected = model
                .query_parameter(
                    &facts.identity,
                    &parameter.id,
                    selection.region.clone(),
                    limits,
                )
                .unwrap();
            let mut expected = Vec::new();
            let [expert, row, column]: [u64; 3] =
                selection.region.starts.clone().try_into().unwrap();
            for r in row..row + selection.region.shape[1] {
                for c in column..column + selection.region.shape[2] {
                    expected.push(
                        full.values
                            [((expert * parameter.shape[1] + r) * parameter.shape[2] + c) as usize],
                    );
                }
            }
            assert_eq!(selected.values, expected);
            originals.insert(parameter.id.clone(), full.values);
        }
        let edits = [&gate, &write]
            .into_iter()
            .enumerate()
            .map(|(ordinal, selection)| ParameterEdit {
                id: format!("component-edit-{ordinal}"),
                parameter: selection.parameter.id.clone(),
                parameter_shape: selection.parameter.shape.clone(),
                dtype: selection.parameter.dtype.unwrap(),
                region: selection.region.clone(),
                update: ParameterUpdate::Add {
                    values: vec![0.75; selection.region.shape.iter().product::<u64>() as usize],
                },
            })
            .collect();
        let admitted = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity.clone(),
                provenance: "discovered expert read/write coordinate fixture".into(),
                edits,
            })
            .unwrap();
        let active = model.activate_parameter_overlay(&admitted, limits).unwrap();
        let edited = super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]);
        assert_ne!(edited.0, baseline);
        assert_eq!(edited.1, active.overlay_identity);
        assert_eq!(
            super::parameters::parameter_logits_mode(&mut model, &[1, 2, 5, 7], true).0,
            edited.0
        );
        assert_ne!(
            super::parameters::parameter_logits(&mut model, &[7, 5, 2, 1]).0,
            other_baseline
        );
        for selection in [&gate, &write] {
            let parameter = selection.parameter;
            let full = model
                .query_parameter(
                    &active.identity,
                    &parameter.id,
                    ParameterRegion {
                        starts: vec![0; 3],
                        shape: parameter.shape.clone(),
                    },
                    limits,
                )
                .unwrap();
            for (index, (&actual, &original)) in full
                .values
                .iter()
                .zip(&originals[&parameter.id])
                .enumerate()
            {
                let expert = index as u64 / (parameter.shape[1] * parameter.shape[2]);
                let row = index as u64 / parameter.shape[2] % parameter.shape[1];
                let column = index as u64 % parameter.shape[2];
                let touched = [expert, row, column]
                    .iter()
                    .enumerate()
                    .all(|(axis, value)| {
                        *value >= selection.region.starts[axis]
                            && *value < selection.region.starts[axis] + selection.region.shape[axis]
                    });
                assert_eq!(actual, if touched { original + 0.75 } else { original });
            }
        }
        model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(
            super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0,
            baseline
        );
        assert_eq!(
            super::parameters::parameter_logits(&mut model, &[7, 5, 2, 1]).0,
            other_baseline
        );
        assert_eq!(
            std::fs::read(root.0.join("model.safetensors")).unwrap(),
            source
        );
    }
}

#[test]
#[ignore = "requires native MLX CPU services"]
fn native_routed_component_parameters_cpu() {
    verify(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires native MLX Metal services"]
fn native_routed_component_parameters_metal() {
    verify(LocalDevice::Accelerator(0));
}

fn floats(record: &CaptureRecord) -> &[f32] {
    let Some(CapturePayload::Tensor(t)) = &record.payload else {
        panic!("tensor payload")
    };
    let eredu_core::TensorObservationData::F32(v) = t.data() else {
        panic!("floating payload")
    };
    v
}

fn capture_units(device: LocalDevice) {
    let root = routed_fixture();
    let graph = inspect_architecture(&root.0).unwrap();
    let group = &graph.routed_components[0];
    let output_path = eredu_core::RoutingObservationField::RoutedOutput.path(&group.routing);
    let usage = CaptureUsage {
        captures: 128,
        retained_bytes: 256 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
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
        let facts = model.parameter_discovery().unwrap();
        let id = RoutedComponentId {
            group: group.id.clone(),
            expert: 0,
            index: 0,
        };
        let mut weights = vec![];
        for selection in [
            group
                .read_weight(&id, ComponentReadRole::Gate, &facts)
                .unwrap(),
            group.write_column(&id, &facts).unwrap(),
        ] {
            weights.push(
                model
                    .query_parameter(
                        &facts.identity,
                        &selection.parameter.id,
                        ParameterRegion {
                            starts: vec![0; 3],
                            shape: selection.parameter.shape.clone(),
                        },
                        usage,
                    )
                    .unwrap()
                    .values,
            );
        }
        let discovery = model.capture_discovery().unwrap();
        for path in [&group.activation, &group.effective_activation] {
            let supported = discovery
                .support
                .points
                .iter()
                .find(|p| &p.path == path)
                .unwrap();
            assert_eq!(
                supported.prefill,
                eredu_core::ObservationSupportStatus::Supported
            );
            assert_eq!(
                supported.decode,
                eredu_core::ObservationSupportStatus::Supported
            );
        }
        let chat = model
            .source_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user","content":"left"})],
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
        let paths = [
            group.activation.clone(),
            group.effective_activation.clone(),
            group.input.clone().unwrap(),
            output_path.clone(),
            "model.logits".into(),
        ];
        let capture = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: paths
                .iter()
                .enumerate()
                .map(|(i, path)| CaptureSelection {
                    id: format!("s{i}"),
                    path: path.clone(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: if i < 2 {
                        CaptureTransform::RoutedUnits
                    } else {
                        CaptureTransform::FullTensor
                    },
                })
                .collect(),
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
        for controlled in [false, true] {
            model.reset().unwrap();
            let prepared_prefix = prefix.clone();
            let prepared_capture = capture.clone();
            let prepared_trace = trace;
            let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings));
            prepared.input = PreparedChatPrompt::TokenIds(&prepared_prefix);
            prepared.output_mode = PreparedChatOutputMode::Text;
            prepared.capture = Some(&prepared_capture);
            let mut events = vec![];
            if controlled {
                let mut emit = |r: ControlledGenerationRecord| {
                    events.push(r);
                    ControlFlow::Continue(())
                };
                let mut run = model
                    .start_controlled_chat(prepared, prepared_trace, Default::default(), &mut emit)
                    .unwrap()
                    .unwrap();
                run.enable_snapshots(
                    SnapshotLimits {
                        max_snapshots: 2,
                        max_branches: 1,
                        // Fork admission includes worst-case trace delivery storage.
                        retained_bytes: 512 << 20,
                        cumulative_copy_bytes: 2 << 30,
                    },
                    ORIGINAL_CAPACITY,
                    copy_limits(),
                )
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
                            capture_limits: Some(capture.limits.clone()),
                            sampling: None,
                            intervention: None,
                        },
                        &mut emit,
                    )
                    .unwrap();
                run.exchange(&mut child, &mut emit).unwrap();
                run.run(&mut emit).unwrap();
                run.exchange(&mut child, &mut emit).unwrap();
            } else {
                (|| -> Result<_, ControlledGenerationError> {
                    let mut emit = |r| {
                        events.push(r);
                        ControlFlow::Continue(())
                    };
                    let mut run = model
                        .start_controlled_chat(
                            prepared,
                            prepared_trace,
                            GenerationControlHandle::new(Default::default()),
                            &mut emit,
                        )?
                        .expect("live fixture control");
                    run.run(&mut emit)
                })()
                .unwrap();
            }
            let steps: Vec<_> = events
                .iter()
                .filter_map(|r| match r.event.progress() {
                    Some(ObservedGenerationEvent::Token {
                        forced,
                        captures: Some(step),
                        ..
                    }) => {
                        assert!(!forced);
                        Some(step)
                    }
                    _ => None,
                })
                .collect();
            assert_eq!(steps.len(), if controlled { 9 } else { 3 });
            for (index, step) in steps.iter().enumerate() {
                assert_eq!(step.prediction_index, index as u64 % 3);
                assert!(
                    step.records
                        .iter()
                        .all(|r| r.outcome == CaptureOutcome::Captured),
                    "{:?}",
                    step.records
                        .iter()
                        .map(|r| (&r.path, &r.outcome))
                        .collect::<Vec<_>>()
                );
                let Some(CapturePayload::RoutedUnits(units)) = &step.records[0].payload else {
                    panic!("sparse units")
                };
                assert_eq!(step.records[0].payload, step.records[1].payload);
                let tokens = if index % 3 == 0 { 67 } else { 1 };
                assert_eq!(units.rows.len(), tokens * 4);
                let input = floats(&step.records[2]);
                let expected_write = floats(&step.records[3]);
                let mut reconstructed = vec![0.0f64; tokens * 16];
                for row in &units.rows {
                    let eredu_core::TensorObservationData::F32(values) = row.values.data() else {
                        panic!("units")
                    };
                    assert_eq!(values.len(), 6);
                    for (unit, &actual) in values.iter().enumerate() {
                        let mut gate = 0.0f64;
                        let mut up = 0.0f64;
                        for k in 0..16 {
                            let x = input[row.token as usize * 16 + k] as f64;
                            gate +=
                                x * weights[0][(row.expert as usize * 12 + unit) * 16 + k] as f64;
                            up += x * weights[0][(row.expert as usize * 12 + 6 + unit) * 16 + k]
                                as f64;
                        }
                        let expected = gate / (1.0 + (-gate).exp()) * up;
                        assert!((actual as f64 - expected).abs() < 2e-5 + 2e-5 * expected.abs());
                        for k in 0..16 {
                            reconstructed[row.token as usize * 16 + k] += actual as f64
                                * row.coefficient as f64
                                * weights[1][(row.expert as usize * 16 + k) * 6 + unit] as f64;
                        }
                    }
                }
                for (a, &b) in reconstructed.iter().zip(expected_write) {
                    assert!((*a - b as f64).abs() < 2e-5 + 2e-5 * a.abs());
                }
            }
            let payloads: Vec<_> = steps
                .iter()
                .map(|step| {
                    step.records
                        .iter()
                        .map(|r| r.payload.clone())
                        .collect::<Vec<_>>()
                })
                .collect();
            if let Some(expected) = &baseline {
                assert_eq!(&payloads[..3], expected);
                assert_eq!(&payloads[3..6], expected);
                assert_eq!(&payloads[6..], expected);
                assert!(
                    steps[3].cumulative_usage.host_bytes > steps[0].cumulative_usage.host_bytes
                );
            } else {
                baseline = Some(payloads);
            }
        }
        for empty in [false, true] {
            model.reset().unwrap();
            let mut sliced = capture.clone();
            sliced.selections.truncate(1);
            sliced.selections[0].schedule.decode = false;
            sliced.selections[0].slices = vec![
                CaptureSlice {
                    axis: "token".into(),
                    start: 1,
                    end: 67,
                    stride: 3,
                },
                CaptureSlice {
                    axis: "route".into(),
                    start: 1,
                    end: if empty { 1 } else { 3 },
                    stride: 1,
                },
                CaptureSlice {
                    axis: "component".into(),
                    start: 1,
                    end: 6,
                    stride: 2,
                },
            ];
            let prepared_prefix = prefix.clone();
            let prepared_capture = sliced;
            let prepared_trace = trace;
            let mut prepared = PreparedChatRequest::new(&chat, original_settings(settings));
            prepared.input = PreparedChatPrompt::TokenIds(&prepared_prefix);
            prepared.output_mode = PreparedChatOutputMode::Text;
            prepared.capture = Some(&prepared_capture);
            let mut events = vec![];
            (|| -> Result<_, ControlledGenerationError> {
                let mut emit = |r| {
                    events.push(r);
                    ControlFlow::Continue(())
                };
                let mut run = model
                    .start_controlled_chat(
                        prepared,
                        prepared_trace,
                        GenerationControlHandle::new(Default::default()),
                        &mut emit,
                    )?
                    .expect("live fixture control");
                run.run(&mut emit)
            })()
            .unwrap();
            let steps: Vec<_> = events
                .iter()
                .filter_map(|r| match r.event.progress() {
                    Some(ObservedGenerationEvent::Token {
                        captures: Some(s), ..
                    }) => Some(s),
                    _ => None,
                })
                .collect();
            let record = &steps[0].records[0];
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            let Some(CapturePayload::RoutedUnits(sparse)) = &record.payload else {
                panic!("sparse subset")
            };
            let Some(CapturePayload::RoutedUnits(full)) = &baseline.as_ref().unwrap()[0][0] else {
                panic!("full sparse")
            };
            if empty {
                assert!(sparse.rows.is_empty());
                assert_eq!(record.charged.retained_bytes, 0);
            } else {
                let expected: Vec<_> = full
                    .rows
                    .iter()
                    .filter(|r| {
                        r.token >= 1 && (r.token - 1).is_multiple_of(3) && (1..3).contains(&r.slot)
                    })
                    .collect();
                assert_eq!(sparse.rows.len(), expected.len());
                for (actual, reference) in sparse.rows.iter().zip(expected) {
                    assert_eq!(
                        (actual.token, actual.slot, actual.expert, actual.coefficient),
                        (
                            reference.token,
                            reference.slot,
                            reference.expert,
                            reference.coefficient
                        )
                    );
                    let eredu_core::TensorObservationData::F32(all) = reference.values.data()
                    else {
                        panic!("full")
                    };
                    assert_eq!(
                        actual.values.data(),
                        &eredu_core::TensorObservationData::F32(vec![all[1], all[3], all[5]])
                    );
                }
            }
            assert!(matches!(
                steps[1].records[0].outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            ));
        }
    }
}

#[test]
#[ignore = "requires native MLX CPU services"]
fn native_routed_component_capture_cpu() {
    capture_units(LocalDevice::Cpu);
}

#[test]
#[cfg(feature = "metal")]
#[ignore = "requires native MLX Metal services"]
fn native_routed_component_capture_metal() {
    capture_units(LocalDevice::Accelerator(0));
}
