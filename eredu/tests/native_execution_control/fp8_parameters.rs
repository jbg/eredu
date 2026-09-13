//! Loaded FP8 arithmetic and reversible edits through public portable APIs.
use super::*;
use eredu_core::{intervention::*, parameters::*, ArchitectureDescriptor, ResidencyPlan};
use std::collections::BTreeMap;
#[path = "../../examples/component_reference_analysis.rs"]
pub(super) mod analysis;
#[path = "fp8_parameters/grouped.rs"]
mod grouped;

type Tensors = BTreeMap<String, (String, Vec<usize>, Vec<u8>)>;
fn decode(code: u8) -> f32 {
    let exponent = ((code & 127) >> 3) as i32;
    let mantissa = (code & 7) as f32;
    let v = if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    };
    if code & 128 == 0 {
        v
    } else {
        -v
    }
}
fn fp8_fixture() -> (Fixture, serde_json::Value, Tensors, Tensors) {
    fp8_fixture_dimensions(false)
}
fn fp8_fixture_dimensions(partial_blocks: bool) -> (Fixture, serde_json::Value, Tensors, Tensors) {
    let root = fixture(false);
    let source: serde_json::Value = serde_json::from_str(include_str!(
        "../../../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    ))
    .unwrap();
    let mut config = source["dense"]["config"].clone();
    for (key, value) in [
        ("hidden_size", if partial_blocks { 130 } else { 32 }),
        ("intermediate_size", if partial_blocks { 130 } else { 384 }),
        ("head_dim", if partial_blocks { 4 } else { 64 }),
        ("vocab_size", if partial_blocks { 129 } else { 128 }),
    ] {
        config[key] = value.into();
    }
    // Cached-reference trials stop at the explicit prediction budget.
    config["eos_token_id"] = serde_json::json!([]);
    config["quantization_config"] = serde_json::json!({"quant_method":"fp8",
        "activation_scheme":"dynamic","weight_block_size":[128,128],"ignored_layers":[]});
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let plan = resolved.architecture.checkpoint();
    let mut packed = BTreeMap::new();
    let mut dense = BTreeMap::new();
    let scale = |row: usize, col: usize| (1 + (row + col * 2) % 5) as f32 / 64.0;
    for tensor in plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .filter_map(|g| g.variants.first())
            .flat_map(|v| &v.tensors),
    ) {
        if tensor.requirement != eredu_checkpoint::schema::TensorRequirement::Required {
            continue;
        }
        let count = tensor.shape.iter().product::<usize>();
        let shape = tensor.shape.clone();
        if matches!(
            tensor.dtype,
            eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                eredu_checkpoint::StoredDtype::F8E4M3
            )
        ) {
            let codes: Vec<_> = (0..count)
                .map(|i| {
                    (0x28 + (i * 13 + tensor.key.len() * 7) % 40) as u8
                        | if (i + tensor.key.len()) % 3 == 0 {
                            128
                        } else {
                            0
                        }
                })
                .collect();
            let values: Vec<_> = codes
                .iter()
                .enumerate()
                .map(|(i, c)| decode(*c) * scale(i / shape[1] / 128, i % shape[1] / 128))
                .collect();
            packed.insert(tensor.key.clone(), ("F8_E4M3".into(), shape.clone(), codes));
            dense.insert(
                tensor.key.clone(),
                (
                    "F32".into(),
                    shape,
                    values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                ),
            );
        } else {
            let values: Vec<_> = (0..count)
                .map(|i| {
                    if tensor.key.ends_with("weight_scale_inv") {
                        scale(i / shape[1], i % shape[1])
                    } else if tensor.key.contains("norm") {
                        0.9 + (i % 7) as f32 * 0.03
                    } else {
                        (((i * 17 + tensor.key.len() * 7) % 101) as f32 - 50.0) * 0.003
                    }
                })
                .collect();
            let data = (
                "F32".into(),
                shape,
                values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            );
            packed.insert(tensor.key.clone(), data.clone());
            if tensor.role != eredu_checkpoint::schema::TensorRole::Companion {
                dense.insert(tensor.key.clone(), data);
            }
        }
    }
    assert!(packed.values().any(|(dtype, _, _)| dtype == "F8_E4M3"));
    std::fs::write(
        root.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    super::quantized_parameters::write_tensors(&root.0, &packed);
    (root, config, packed, dense)
}

pub(super) fn capture<
    B: eredu_runtime::execution_control::TextSnapshotBackend
        + eredu_runtime::execution_control::TextSamplingControlBackend,
>(
    model: &mut LoadedModel<B>,
    architecture: &ArchitectureDescriptor,
    masked: bool,
    controlled: bool,
) -> BTreeMap<String, Vec<f32>> {
    capture_with_precision(
        model,
        architecture,
        masked.then_some(InterventionDtype::Float32),
        controlled,
    )
    .0
}

fn capture_with_precision<
    B: eredu_runtime::execution_control::TextSnapshotBackend
        + eredu_runtime::execution_control::TextSamplingControlBackend,
>(
    model: &mut LoadedModel<B>,
    architecture: &ArchitectureDescriptor,
    masked: Option<InterventionDtype>,
    controlled: bool,
) -> (
    BTreeMap<String, Vec<f32>>,
    BTreeMap<String, eredu_core::checkpoint::TensorDtype>,
) {
    capture_with_precision_and_selection(model, architecture, masked, controlled, false)
}

fn capture_with_precision_and_selection<
    B: eredu_runtime::execution_control::TextSnapshotBackend
        + eredu_runtime::execution_control::TextSamplingControlBackend,
>(
    model: &mut LoadedModel<B>,
    architecture: &ArchitectureDescriptor,
    masked: Option<InterventionDtype>,
    controlled: bool,
    empty_projection_inputs: bool,
) -> (
    BTreeMap<String, Vec<f32>>,
    BTreeMap<String, eredu_core::checkpoint::TensorDtype>,
) {
    model.reset().unwrap();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"left"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let readout = architecture.component_readout.as_ref().unwrap();
    let paths: Vec<_> = architecture
        .components
        .iter()
        .flat_map(|g| {
            [
                g.effective_activation.clone(),
                g.write_input.clone().unwrap(),
            ]
        })
        .chain(
            [
                &readout.embedding,
                &readout.residual,
                &readout.normalized,
                &readout.linear_scores,
            ]
            .into_iter()
            .map(|p| format!("{p}.effective")),
        )
        .chain(readout.projection_input.clone())
        .chain(
            readout
                .other_writes
                .iter()
                .map(|write| write.effective_output.clone()),
        )
        .chain(["model.logits".into()])
        .filter(|path| {
            !empty_projection_inputs
                || path == "model.logits"
                || architecture
                    .components
                    .iter()
                    .any(|group| group.write_input.as_ref() == Some(path))
                || readout.projection_input.as_ref() == Some(path)
        })
        .collect();
    let usage = CaptureUsage {
        captures: 256,
        retained_bytes: 128 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let slice = CaptureSlice {
        axis: "sequence".into(),
        start: 3,
        end: 4,
        stride: 1,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: paths
            .iter()
            .enumerate()
            .map(|(i, p)| CaptureSelection {
                id: format!("c{i}"),
                path: p.clone(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![if empty_projection_inputs && p != "model.logits" {
                    CaptureSlice {
                        end: slice.start,
                        ..slice.clone()
                    }
                } else {
                    slice.clone()
                }],
                transform: CaptureTransform::Slice,
            })
            .collect(),
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let intervention = InterventionPlan {
        schema_version: INTERVENTION_SCHEMA_VERSION,
        operations: if let Some(dtype) = masked {
            architecture
                .components
                .iter()
                .take(2)
                .enumerate()
                .map(|(i, g)| InterventionOperation {
                    id: format!("keep{i}"),
                    target: g.activation.clone(),
                    schedule: Default::default(),
                    slices: vec![slice.clone()],
                    action: InterventionAction::MaskComponents {
                        dtype,
                        indices: vec![1, 3],
                        keep_selected: true,
                    },
                    evidence: InterventionEvidence::None,
                })
                .collect()
        } else {
            vec![]
        },
    };
    let prepared = model
        .prepare_intervened_token_ids(
            &chat,
            vec![1, 2, 5, 7],
            PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    temperature: Some(0.0),
                    max_new_tokens: Some(1),
                    ..Default::default()
                },
                seed: 17,
                ..Default::default()
            },
            capture,
            intervention,
            TraceLimits {
                per_record_bytes: 4 << 20,
                total_bytes: 16 << 20,
            },
        )
        .unwrap();
    let mut records = vec![];
    if controlled {
        let mut run = model
            .start_controlled_text(prepared, &[], Default::default(), |e| {
                records.push(e.generation);
                ControlFlow::Continue(())
            })
            .unwrap();
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 1,
            retained_bytes: 64 << 20,
            cumulative_copy_bytes: 256 << 20,
        })
        .unwrap();
        let snap = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
        run.run(|e| {
            records.push(e.generation);
            ControlFlow::Continue(())
        })
        .unwrap();
        let first = super::components::tensors(&records);
        let first_precision = source_precisions(&records);
        records.clear();
        run.restore(&snap, |_| ControlFlow::Continue(())).unwrap();
        run.run(|e| {
            records.push(e.generation);
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(first, super::components::tensors(&records));
        assert_eq!(first_precision, source_precisions(&records));
    } else {
        model
            .generate_observed_text(prepared, &[], Default::default(), |e| {
                records.push(e);
                ControlFlow::Continue(())
            })
            .unwrap();
    }
    if empty_projection_inputs {
        let mut empty_records = 0;
        for record in records
            .iter()
            .filter_map(|event| match &event.event {
                ObservedGenerationEvent::Token {
                    captures: Some(step),
                    ..
                } => Some(step),
                _ => None,
            })
            .flat_map(|step| &step.records)
            .filter(|record| record.path != "model.logits")
        {
            assert_eq!(record.outcome, CaptureOutcome::Captured);
            assert_eq!(
                record.source_dtype,
                Some(eredu_core::checkpoint::TensorDtype::F32)
            );
            assert!(record.selected_shape.as_ref().unwrap().contains(&0));
            let Some(CapturePayload::Tensor(value)) = &record.payload else {
                panic!("empty raw evidence")
            };
            assert_eq!(
                value.data(),
                &eredu_core::TensorObservationData::F32(vec![])
            );
            empty_records += 1;
        }
        assert_eq!(empty_records, paths.len() - 1);
    }
    (
        super::components::tensors(&records),
        source_precisions(&records),
    )
}
fn source_precisions(
    records: &[ObservedGenerationRecord],
) -> BTreeMap<String, eredu_core::checkpoint::TensorDtype> {
    records
        .iter()
        .filter_map(|e| match &e.event {
            ObservedGenerationEvent::Token {
                captures: Some(step),
                ..
            } => Some(step),
            _ => None,
        })
        .flat_map(|step| &step.records)
        .map(|record| {
            (
                record.path.clone(),
                record
                    .source_dtype
                    .clone()
                    .expect("native source precision"),
            )
        })
        .collect()
}
fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() <= 3e-5 + 3e-5 * b.abs(), "{a} != {b}");
    }
}
fn check_evidence<B: ParameterBackend>(
    model: &mut LoadedModel<B>,
    architecture: &ArchitectureDescriptor,
    evidence: &BTreeMap<String, Vec<f32>>,
    expect_fp8: bool,
) {
    let result = analysis::reconstruct(model, architecture, evidence, 1, 2).unwrap();
    let correction = result["signed_reconstruction"][0]["projection_input_correction"]
        .as_f64()
        .unwrap();
    if expect_fp8 {
        assert!(
            correction.abs() > 1e-8,
            "nonzero head quantization correction"
        );
    } else {
        assert_eq!(correction, 0.0);
    }
    for g in &architecture.components {
        let actual = &evidence[g.write_input.as_ref().unwrap()];
        let supplied = &evidence[&g.effective_activation];
        assert_eq!(actual.len(), supplied.len());
        if expect_fp8 {
            assert_ne!(actual, supplied);
        } else {
            assert_eq!(actual, supplied);
        }
    }
}
#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_fp8_inputs_reconstruct_components_and_overlay_transitions_across_residencies() {
    verify_fp8_inputs_and_overlay_transitions(LocalDevice::Cpu, false);
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_fp8_partial_blocks_preserve_components_and_overlays_across_residencies() {
    verify_fp8_inputs_and_overlay_transitions(LocalDevice::Cpu, true);
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
#[ignore = "requires an accessible Metal device; run explicitly on a GPU worker"]
fn native_fp8_metal_inputs_and_overlay_transitions_across_residencies() {
    verify_fp8_inputs_and_overlay_transitions(LocalDevice::Accelerator(0), false);
}

#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
#[ignore = "requires an accessible Metal device; run explicitly on a GPU worker"]
fn native_fp8_partial_blocks_metal_components_and_overlays_across_residencies() {
    verify_fp8_inputs_and_overlay_transitions(LocalDevice::Accelerator(0), true);
}

fn verify_fp8_inputs_and_overlay_transitions(device: LocalDevice, partial_blocks: bool) {
    let limits = analysis::parameter_limits();
    for residency in [
        ResidencyPlan::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        },
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 8 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        let (root, mut config, mut packed, dense) = fp8_fixture_dimensions(partial_blocks);
        let original_file = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let architecture = inspect_architecture(&root.0).unwrap();
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency)
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let facts = model.parameter_discovery().unwrap();
        let intervention_points = model.intervention_discovery().unwrap();
        for path in architecture
            .components
            .iter()
            .filter_map(|g| g.write_input.as_ref())
            .chain(
                architecture
                    .component_readout
                    .as_ref()
                    .unwrap()
                    .projection_input
                    .as_ref(),
            )
        {
            assert!(!intervention_points
                .points
                .iter()
                .any(|point| &point.path == path));
        }
        for p in facts.parameters.iter().filter(|p| p.supported) {
            let expected = &dense[&p.id];
            let values: Vec<f32> = expected
                .2
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(
                model
                    .query_parameter(
                        &facts.identity,
                        &p.id,
                        ParameterRegion {
                            starts: vec![0; p.shape.len()],
                            shape: p.shape.clone()
                        },
                        limits
                    )
                    .unwrap()
                    .values,
                values
            );
            assert_eq!(
                matches!(
                    p.input_transform,
                    ProjectionInputTransform::BlockFp8E4m3 { .. }
                ),
                packed[&p.id].0 == "F8_E4M3"
            );
        }
        let baseline = capture(&mut model, &architecture, false, false);
        assert_eq!(
            baseline["model.logits"],
            super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0
        );
        assert_eq!(baseline, capture(&mut model, &architecture, false, true));
        let empty =
            capture_with_precision_and_selection(&mut model, &architecture, None, false, true);
        assert_eq!(empty.0["model.logits"], baseline["model.logits"]);
        assert_eq!(
            empty,
            capture_with_precision_and_selection(&mut model, &architecture, None, true, true)
        );
        check_evidence(&mut model, &architecture, &baseline, true);
        let masked = capture(&mut model, &architecture, true, false);
        for group in architecture.components.iter().take(2) {
            for (i, value) in masked[&group.effective_activation].iter().enumerate() {
                if ![1, 3].contains(&i) {
                    assert_eq!(*value, 0.0);
                }
            }
        }
        assert_ne!(
            masked[&architecture.components[1].effective_activation],
            baseline[&architecture.components[1].effective_activation]
        );
        assert_ne!(masked["model.logits"], baseline["model.logits"]);
        assert_eq!(masked, capture(&mut model, &architecture, true, true));
        check_evidence(&mut model, &architecture, &masked, true);
        // Every attention Q/K/V/O and gated FFN matrix, plus the output head,
        // transitions atomically. Initially all numerical deltas are zero.
        let edits: Vec<_> = facts
            .parameters
            .iter()
            .filter(|p| {
                matches!(
                    p.input_transform,
                    ProjectionInputTransform::BlockFp8E4m3 { .. }
                )
            })
            .enumerate()
            .map(|(i, p)| ParameterEdit {
                id: format!("zero{i}"),
                parameter: p.id.clone(),
                parameter_shape: p.shape.clone(),
                dtype: InterventionDtype::Float32,
                region: ParameterRegion {
                    starts: vec![0, 0],
                    shape: vec![1, 1],
                },
                update: ParameterUpdate::Add { values: vec![0.0] },
            })
            .collect();
        assert!(edits.len() > 7);
        for edit in &edits {
            packed.insert(edit.parameter.clone(), dense[&edit.parameter].clone());
            packed.remove(&format!("{}_scale_inv", edit.parameter));
        }
        config["quantization_config"]["ignored_layers"] = serde_json::json!(edits
            .iter()
            .map(|e| e.parameter.trim_end_matches(".weight"))
            .collect::<Vec<_>>());
        let reference = fixture(false);
        std::fs::write(
            reference.0.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        super::quantized_parameters::write_tensors(&reference.0, &packed);
        let plan = ParameterOverlayPlan {
            schema_version: PARAMETER_SCHEMA_VERSION,
            base_identity: facts.identity.clone(),
            provenance:
                "independent FP8 host weight decode; zero deltas retain F32 execution semantics"
                    .into(),
            edits: edits.clone(),
        };
        let roundtrip = serde_json::from_slice(&serde_json::to_vec(&plan).unwrap()).unwrap();
        let overlay = model.admit_parameter_overlay(roundtrip).unwrap();
        let before = model.parameter_discovery().unwrap().usage;
        assert!(matches!(
            model.activate_parameter_overlay(&overlay, before),
            Err(ParameterError::Budget(_))
        ));
        assert_eq!(model.parameter_discovery().unwrap().usage, before);
        model.activate_parameter_overlay(&overlay, limits).unwrap();
        let active = model.parameter_discovery().unwrap();
        assert!(active
            .parameters
            .iter()
            .filter(|p| p.supported)
            .all(|p| p.input_transform == ProjectionInputTransform::Identity));
        let changed = capture(&mut model, &architecture, false, true);
        assert_ne!(changed["model.logits"], baseline["model.logits"]);
        check_evidence(&mut model, &architecture, &changed, false);
        let (mut expected, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            &reference.0,
            &execution,
        )
        .unwrap()
        .into_parts();
        let expected_evidence = capture(&mut expected, &architecture, false, false);
        for (path, values) in &changed {
            close(values, &expected_evidence[path]);
        }
        model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(baseline, capture(&mut model, &architecture, false, false));
        assert_eq!(
            model.parameter_discovery().unwrap().parameters,
            facts.parameters
        );
        // A second transaction includes nonzero read-row and write-column edits.
        let mut edits = edits;
        for (i, e) in edits.iter_mut().enumerate() {
            e.region.shape = if e.parameter.ends_with("down_proj.weight")
                || e.parameter.ends_with("o_proj.weight")
            {
                vec![e.parameter_shape[0], 1]
            } else {
                vec![1, e.parameter_shape[1]]
            };
            e.update = ParameterUpdate::Add {
                values: (0..e.region.shape.iter().product::<u64>())
                    .map(|j| ((j + i as u64) % 7) as f32 * 0.017 - 0.041)
                    .collect(),
            };
        }
        super::parameters::edit_reference(&reference.0, &edits);
        let current = model.parameter_discovery().unwrap();
        let overlay = model
            .admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: current.identity,
                provenance: "coordinated FP8 attention and FFN read/write edits".into(),
                edits,
            })
            .unwrap();
        model.activate_parameter_overlay(&overlay, limits).unwrap();
        drop(expected);
        let (mut expected, _) = LoadedModel::load_execution_plan(
            &MlxBackendFactory::default(),
            &reference.0,
            &execution,
        )
        .unwrap()
        .into_parts();
        for prefix in [[1, 2, 5, 7], [7, 5, 2, 1]] {
            let actual = super::parameters::parameter_logits_mode(&mut model, &prefix, true);
            assert_eq!(actual.1.as_deref(), Some(overlay.identity()));
            close(
                &actual.0,
                &super::parameters::parameter_logits(&mut expected, &prefix).0,
            );
        }
        let active = model.parameter_discovery().unwrap();
        model.remove_parameter_overlay(&active.identity).unwrap();
        assert_eq!(baseline, capture(&mut model, &architecture, false, false));
        assert_eq!(
            std::fs::read(root.0.join("model.safetensors")).unwrap(),
            original_file
        );
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_fp8_single_overlay_preserves_other_projection_input_transforms() {
    let (root, mut config, mut packed, dense) = fp8_fixture();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let facts = model.parameter_discovery().unwrap();
    let head = facts
        .parameters
        .iter()
        .find(|p| p.id == "lm_head.weight")
        .unwrap();
    let baseline = super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0;
    let overlay = model
        .admit_parameter_overlay(ParameterOverlayPlan {
            schema_version: PARAMETER_SCHEMA_VERSION,
            base_identity: facts.identity.clone(),
            provenance: "isolated zero-delta FP8 head transition".into(),
            edits: vec![ParameterEdit {
                id: "head".into(),
                parameter: head.id.clone(),
                parameter_shape: head.shape.clone(),
                dtype: InterventionDtype::Float32,
                region: ParameterRegion {
                    starts: vec![0, 0],
                    shape: vec![1, 1],
                },
                update: ParameterUpdate::Add { values: vec![0.0] },
            }],
        })
        .unwrap();
    model
        .activate_parameter_overlay(&overlay, analysis::parameter_limits())
        .unwrap();
    let active = model.parameter_discovery().unwrap();
    for (before, after) in facts.parameters.iter().zip(&active.parameters) {
        assert_eq!(before.id, after.id);
        assert_eq!(
            after.input_transform,
            if before.id == head.id {
                ProjectionInputTransform::Identity
            } else {
                before.input_transform.clone()
            }
        );
    }
    packed.insert(head.id.clone(), dense[&head.id].clone());
    packed.remove("lm_head.weight_scale_inv");
    config["quantization_config"]["ignored_layers"] = serde_json::json!(["lm_head"]);
    let reference = fixture(false);
    std::fs::write(
        reference.0.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    super::quantized_parameters::write_tensors(&reference.0, &packed);
    let (mut expected, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &reference.0, &execution)
            .unwrap()
            .into_parts();
    let changed = super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0;
    assert_ne!(changed, baseline);
    close(
        &changed,
        &super::parameters::parameter_logits(&mut expected, &[1, 2, 5, 7]).0,
    );
    model.remove_parameter_overlay(&active.identity).unwrap();
    assert_eq!(
        facts.parameters,
        model.parameter_discovery().unwrap().parameters
    );
    assert_eq!(
        baseline,
        super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0
    );
}

fn verify_fp8_bfloat16_overlay_precision(device: LocalDevice) {
    for residency in [
        ResidencyPlan::FullyResident,
        ResidencyPlan::LayerwiseHost {
            device_layer_window: 1,
            device_budget_bytes: Some(8 << 20),
            host_budget_bytes: Some(8 << 20),
        },
        ResidencyPlan::DenseDiskStream {
            device_budget_bytes: 8 << 20,
            host_budget_bytes: 8 << 20,
            host_lookahead: 1,
            background_queue: 1,
        },
    ] {
        let (root, config, mut packed, dense) = fp8_fixture();
        // Choose exact BF16 fixture values for ordinary embeddings and norms.
        // FP8 weights and their F32 block scales retain their original encoding.
        for (name, (dtype, _, bytes)) in &mut packed {
            if dtype == "F32" && !name.ends_with("weight_scale_inv") {
                *bytes = bytes
                    .chunks_exact(4)
                    .flat_map(|word| {
                        let bits = u32::from_le_bytes(word.try_into().unwrap());
                        ((bits >> 16) as u16).to_le_bytes()
                    })
                    .collect();
                *dtype = "BF16".into();
            }
        }
        super::quantized_parameters::write_tensors(&root.0, &packed);
        let source = std::fs::read(root.0.join("model.safetensors")).unwrap();
        let execution = ExecutionPlan::fully_resident(local_device_plan(device).unwrap())
            .with_residency(residency);
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
                .unwrap()
                .into_parts();
        let architecture = inspect_architecture(&root.0).unwrap();
        let estimate = |model: &LoadedModel<_>| {
            model
                .estimate_runtime_state(eredu_core::InputTokenCount::text(4), 3, 1)
                .unwrap()
        };
        let baseline_estimate = estimate(&model);
        assert_eq!(
            baseline_estimate
                .assumptions
                .floating_state_dtype_bytes
                .get(),
            2
        );
        let (_, baseline_precision) = capture_with_precision(&mut model, &architecture, None, true);
        assert_eq!(
            baseline_precision["model.logits"],
            eredu_core::checkpoint::TensorDtype::Bf16
        );
        for group in &architecture.components {
            assert_eq!(
                baseline_precision[&group.effective_activation],
                eredu_core::checkpoint::TensorDtype::Bf16
            );
            assert_eq!(
                baseline_precision[group.write_input.as_ref().unwrap()],
                eredu_core::checkpoint::TensorDtype::F32
            );
        }
        let ordinary_mask = capture_with_precision(
            &mut model,
            &architecture,
            Some(InterventionDtype::Bfloat16),
            false,
        );
        assert_eq!(
            ordinary_mask,
            capture_with_precision(
                &mut model,
                &architecture,
                Some(InterventionDtype::Bfloat16),
                true
            )
        );
        let original = model.parameter_discovery().unwrap();
        let norm = original
            .parameters
            .iter()
            .find(|p| p.id == "model.norm.weight")
            .unwrap();
        assert_eq!(norm.dtype, Some(InterventionDtype::Bfloat16));
        let norm_values = model
            .query_parameter(
                &original.identity,
                &norm.id,
                ParameterRegion {
                    starts: vec![0],
                    shape: norm.shape.clone(),
                },
                analysis::parameter_limits(),
            )
            .unwrap()
            .values;
        assert!(norm_values.iter().all(|v| v.to_bits() & 65535 == 0));
        let baseline = super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0;
        assert!(
            baseline.iter().all(|v| v.to_bits() & 65535 == 0),
            "baseline head returns BF16 values"
        );
        let baseline_decode =
            super::parameters::parameter_decode_logits(&mut model, &[1, 2, 5, 7], false);
        // Isolate Q/K/V promotion, including differently typed cache arrays.
        for case in 0..5 {
            let head_only = case == 0;
            let current = model.parameter_discovery().unwrap();
            let edits: Vec<_> = current
                .parameters
                .iter()
                .filter(|p| {
                    matches!(
                        p.input_transform,
                        ProjectionInputTransform::BlockFp8E4m3 { .. }
                    ) && if head_only {
                        p.id == "lm_head.weight"
                    } else if case >= 2 {
                        p.id == [
                            "model.layers.0.self_attn.q_proj.weight",
                            "model.layers.0.self_attn.k_proj.weight",
                            "model.layers.0.self_attn.v_proj.weight",
                        ][case - 2]
                    } else {
                        p.id.starts_with("model.layers.0.")
                    }
                })
                .enumerate()
                .map(|(i, p)| {
                    let region = if head_only {
                        ParameterRegion {
                            starts: vec![0, 0],
                            shape: vec![1, 1],
                        }
                    } else if p.id.ends_with("down_proj.weight") || p.id.ends_with("o_proj.weight")
                    {
                        ParameterRegion {
                            starts: vec![0, 1],
                            shape: vec![p.shape[0], 1],
                        }
                    } else {
                        ParameterRegion {
                            starts: vec![1, 0],
                            shape: vec![1, p.shape[1]],
                        }
                    };
                    ParameterEdit {
                        id: format!("mixed{i}"),
                        parameter: p.id.clone(),
                        parameter_shape: p.shape.clone(),
                        dtype: InterventionDtype::Float32,
                        update: ParameterUpdate::Add {
                            values: (0..region.shape.iter().product::<u64>())
                                .map(|j| {
                                    if head_only {
                                        0.0
                                    } else {
                                        ((j + i as u64) % 7) as f32 * 0.017 - 0.041
                                    }
                                })
                                .collect(),
                        },
                        region,
                    }
                })
                .collect();
            assert_eq!(edits.len(), if case == 1 { 7 } else { 1 });
            let mut reference_tensors = packed.clone();
            for edit in &edits {
                reference_tensors.insert(edit.parameter.clone(), dense[&edit.parameter].clone());
                reference_tensors.remove(&format!("{}_scale_inv", edit.parameter));
            }
            let mut reference_config = config.clone();
            reference_config["quantization_config"]["ignored_layers"] = serde_json::json!(edits
                .iter()
                .map(|e| e.parameter.trim_end_matches(".weight"))
                .collect::<Vec<_>>());
            let reference = fixture(false);
            std::fs::write(
                reference.0.join("config.json"),
                serde_json::to_vec(&reference_config).unwrap(),
            )
            .unwrap();
            super::quantized_parameters::write_tensors(&reference.0, &reference_tensors);
            super::parameters::edit_reference(&reference.0, &edits);
            let overlay = model
                .admit_parameter_overlay(ParameterOverlayPlan {
                    schema_version: PARAMETER_SCHEMA_VERSION,
                    base_identity: current.identity,
                    provenance:
                        "FP8 with BF16 embeddings/norms; independent F32 matrix replacement".into(),
                    edits,
                })
                .unwrap();
            let published = model
                .activate_parameter_overlay(&overlay, analysis::parameter_limits())
                .unwrap();
            assert_eq!(published, model.parameter_discovery().unwrap());
            let active_estimate = estimate(&model);
            assert_eq!(
                active_estimate.assumptions.floating_state_dtype_bytes.get(),
                4
            );
            assert_eq!(
                active_estimate.context_state_bytes,
                baseline_estimate.context_state_bytes * 2
            );
            let (mut expected, _) = LoadedModel::load_execution_plan(
                &MlxBackendFactory::default(),
                &reference.0,
                &execution,
            )
            .unwrap()
            .into_parts();
            let (values, precision) = capture_with_precision(&mut model, &architecture, None, true);
            let (expected_values, expected_precision) =
                capture_with_precision(&mut expected, &architecture, None, false);
            assert_eq!(precision, expected_precision, "replacement case {case}");
            for (path, values) in &values {
                close(values, &expected_values[path]);
            }
            // An isolated K/V replacement need not promote the attention output:
            // the attention/cache operators determine its effective precision.
            if case <= 2 {
                assert_eq!(
                    precision["model.logits"],
                    eredu_core::checkpoint::TensorDtype::F32
                );
            }
            let dtype = match precision[&architecture.components[0].effective_activation] {
                eredu_core::checkpoint::TensorDtype::Bf16 => InterventionDtype::Bfloat16,
                eredu_core::checkpoint::TensorDtype::F32 => InterventionDtype::Float32,
                ref other => panic!("unexpected component precision {other:?}"),
            };
            if case <= 2 {
                assert_eq!(
                    dtype,
                    if head_only {
                        InterventionDtype::Bfloat16
                    } else {
                        InterventionDtype::Float32
                    }
                );
            }
            let ordinary_mask =
                capture_with_precision(&mut model, &architecture, Some(dtype), false);
            assert_eq!(
                ordinary_mask,
                capture_with_precision(&mut model, &architecture, Some(dtype), true)
            );
            for prefix in [[1, 2, 5, 7], [7, 5, 2, 1]] {
                let actual = super::parameters::parameter_logits_mode(&mut model, &prefix, true);
                assert_eq!(actual.1.as_deref(), Some(overlay.identity()));
                assert_eq!(
                    actual.0.iter().any(|v| v.to_bits() & 65535 != 0),
                    precision["model.logits"] == eredu_core::checkpoint::TensorDtype::F32,
                    "exported logits retain measured source precision for case {case}"
                );
                close(
                    &actual.0,
                    &super::parameters::parameter_logits(&mut expected, &prefix).0,
                );
            }
            super::parameters::compare_parameter_decodes(
                &super::parameters::parameter_decode_logits(&mut model, &[1, 2, 5, 7], true),
                &super::parameters::parameter_decode_logits(&mut expected, &[1, 2, 5, 7], false),
                Some(overlay.identity()),
            );
            let active = model.parameter_discovery().unwrap();
            assert_eq!(
                active
                    .parameters
                    .iter()
                    .find(|p| p.id == norm.id)
                    .unwrap()
                    .dtype,
                Some(InterventionDtype::Bfloat16)
            );
            let restored = model.remove_parameter_overlay(&active.identity).unwrap();
            assert_eq!(restored, model.parameter_discovery().unwrap());
            assert_eq!(estimate(&model), baseline_estimate);
            assert_eq!(
                baseline_decode,
                super::parameters::parameter_decode_logits(&mut model, &[1, 2, 5, 7], false)
            );
            assert_eq!(
                capture_with_precision(&mut model, &architecture, None, false).1,
                baseline_precision
            );
            assert_eq!(
                model.parameter_discovery().unwrap().parameters,
                original.parameters
            );
            assert_eq!(
                super::parameters::parameter_logits(&mut model, &[1, 2, 5, 7]).0,
                baseline
            );
        }
        assert_eq!(
            std::fs::read(root.0.join("model.safetensors")).unwrap(),
            source
        );
    }
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_fp8_bfloat16_overlay_precision_cpu() {
    verify_fp8_bfloat16_overlay_precision(LocalDevice::Cpu);
}
#[cfg(all(feature = "metal", target_vendor = "apple"))]
#[test]
#[ignore = "requires an accessible Metal device; run explicitly on a GPU worker"]
fn native_fp8_bfloat16_overlay_precision_metal() {
    verify_fp8_bfloat16_overlay_precision(LocalDevice::Accelerator(0));
}
