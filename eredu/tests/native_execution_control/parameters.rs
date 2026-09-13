use super::*;
use eredu_core::{intervention::InterventionDtype, parameters::*};

pub(super) fn verify_native_projections<B: ParameterBackend>(
    model: &mut LoadedModel<B>,
    limits: CaptureUsage,
) {
    let facts = model.parameter_discovery().unwrap();
    let parameter = facts
        .parameters
        .iter()
        .find(|p| p.id == "model.layers.0.mlp.down_proj.weight")
        .unwrap();
    let region = ParameterRegion {
        starts: vec![0, 0],
        shape: parameter.shape.clone(),
    };
    let full = model
        .query_parameter(&facts.identity, &parameter.id, region.clone(), limits)
        .unwrap();
    for axis in 0..2 {
        let width = parameter.shape[axis] as usize;
        let coefficients: Vec<_> = (0..2 * width).map(|i| (i % 7) as f32 * 0.3 - 0.7).collect();
        let request = ParameterProjection {
            region: region.clone(),
            axis,
            directions: 2,
            coefficients: coefficients.clone(),
        };
        let prior = model.parameter_discovery().unwrap().usage;
        assert!(matches!(
            model.project_parameter(&facts.identity, &parameter.id, request.clone(), prior),
            Err(ParameterError::Budget(_))
        ));
        assert_eq!(model.parameter_discovery().unwrap().usage, prior);
        let projected = model
            .project_parameter(&facts.identity, &parameter.id, request, limits)
            .unwrap();
        let other = parameter.shape[1 - axis] as usize;
        assert_eq!(projected.shape, [other as u64, 2]);
        for row in 0..other {
            for direction in 0..2 {
                let expected: f64 = (0..width)
                    .map(|k| {
                        let index = if axis == 0 {
                            k * other + row
                        } else {
                            row * width + k
                        };
                        full.values[index] as f64 * coefficients[direction * width + k] as f64
                    })
                    .sum();
                let actual = projected.values[row * 2 + direction] as f64;
                assert!((actual - expected).abs() <= 2e-6 + 2e-6 * expected.abs());
            }
        }
    }
}

pub(super) fn parameter_logits<
    B: eredu_runtime::execution_control::TextSnapshotBackend
        + eredu_runtime::execution_control::TextSamplingControlBackend,
>(
    model: &mut LoadedModel<B>,
    prefix: &[u32],
) -> (Vec<f32>, Option<String>) {
    parameter_logits_mode(model, prefix, false)
}

pub(super) fn parameter_logits_mode<
    B: eredu_runtime::execution_control::TextSnapshotBackend
        + eredu_runtime::execution_control::TextSamplingControlBackend,
>(
    model: &mut LoadedModel<B>,
    prefix: &[u32],
    controlled: bool,
) -> (Vec<f32>, Option<String>) {
    model.reset().unwrap();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"left"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    let usage = CaptureUsage {
        captures: 16,
        retained_bytes: 64 << 20,
        host_bytes: 4 << 20,
        encoded_bytes: 4 << 20,
    };
    let plan = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: vec![
            CaptureSelection {
                id: "score".into(),
                path: "model.logits".into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: prefix.len() as u64 - 1,
                    end: prefix.len() as u64,
                    stride: 1,
                }],
                transform: CaptureTransform::Slice,
            },
            CaptureSelection {
                id: "full-scores".into(),
                path: "model.logits".into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![],
                transform: CaptureTransform::TokenScores {
                    token_ids: vec![1, 2, 3],
                },
            },
        ],
        limits: CaptureLimits {
            per_step: usage,
            cumulative: usage,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let branch_limits = plan.limits.clone();
    let prepared = model
        .prepare_observed_token_ids(
            &chat,
            prefix.to_vec(),
            settings,
            plan,
            TraceLimits {
                per_record_bytes: 1 << 20,
                total_bytes: 4 << 20,
            },
        )
        .unwrap();
    let mut records = vec![];
    if controlled {
        let mut run = model
            .start_controlled_text(prepared, &[], Default::default(), |record| {
                records.push(record.generation);
                ControlFlow::Continue(())
            })
            .unwrap();
        let mut emit = |record: ControlledGenerationRecord| {
            records.push(record.generation);
            ControlFlow::Continue(())
        };
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 1,
            retained_bytes: 64 << 20,
            cumulative_copy_bytes: 256 << 20,
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
                        // Fork retention reserves worst-case semantic-event storage
                        // from the trace limit. This branch emits one short prediction.
                        per_record_bytes: 16 << 10,
                        total_bytes: 64 << 10,
                    },
                    capture_limits: Some(branch_limits),
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
        model
            .generate_observed_text(prepared, &[], Default::default(), |record| {
                records.push(record);
                ControlFlow::Continue(())
            })
            .unwrap();
    }
    let mut result = None;
    let mut count = 0;
    for record in records {
        if let ObservedGenerationEvent::Token {
            forced,
            captures: Some(step),
            ..
        } = record.event
        {
            assert!(!forced);
            let Some(CapturePayload::Tensor(tensor)) = &step.records[0].payload else {
                panic!("missing logits")
            };
            let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                panic!("dtype")
            };
            let Some(CapturePayload::TokenScores(scores)) = &step.records[1].payload else {
                panic!("missing full-vocabulary scores")
            };
            let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let log_mass = values
                .iter()
                .map(|v| (*v as f64 - maximum).exp())
                .sum::<f64>()
                .ln();
            for score in &scores.scores {
                let expected = values[score.target.token_id as usize];
                assert_eq!(score.target.score, expected);
                assert_eq!(
                    score.rank,
                    1 + values.iter().filter(|v| **v > expected).count() as u64
                );
                assert!(
                    (score.log_probability - (expected as f64 - maximum - log_mass)).abs() < 2e-7
                );
            }
            let value = (values.clone(), record.parameter_overlay_id);
            if let Some(prior) = &result {
                assert_eq!(&value, prior, "snapshot and sibling parameter isolation");
            }
            result = Some(value);
            count += 1;
        }
    }
    assert_eq!(count, if controlled { 3 } else { 1 });
    result.unwrap()
}

// Independent reference: edit only the test checkpoint's selected F32 bytes.
// The implementation under test never sees this path or mutates either artifact.
pub(super) fn edit_reference(root: &Path, edits: &[ParameterEdit]) {
    let path = root.join("model.safetensors");
    let mut bytes = std::fs::read(&path).unwrap();
    let header_length = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_length]).unwrap();
    for edit in edits {
        let tensor = &header[&edit.parameter];
        assert_eq!(tensor["dtype"], "F32", "source tensor {}", edit.parameter);
        let offset = 8 + header_length + tensor["data_offsets"][0].as_u64().unwrap() as usize;
        let shape: Vec<u64> = serde_json::from_value(tensor["shape"].clone()).unwrap();
        assert_eq!(shape, edit.parameter_shape);
        assert_eq!(edit.region.shape.len(), shape.len());
        assert_eq!(edit.region.starts.len(), shape.len());
        let count = edit.region.shape.iter().product::<u64>() as usize;
        assert_eq!(edit.update.values().len(), count);
        for linear in 0..count {
            let mut coordinate = linear;
            let (mut index, mut stride) = (0, 1);
            for axis in (0..shape.len()).rev() {
                let extent = edit.region.shape[axis] as usize;
                let component = coordinate % extent;
                coordinate /= extent;
                let source = edit.region.starts[axis] as usize + component;
                assert!(source < shape[axis] as usize);
                index += source * stride;
                stride *= shape[axis] as usize;
            }
            let start = offset + index * 4;
            let old = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
            let value = edit.update.values()[linear];
            let new = match edit.update {
                ParameterUpdate::Replace { .. } => value,
                ParameterUpdate::Add { .. } => old + value,
            };
            bytes[start..start + 4].copy_from_slice(&new.to_le_bytes());
        }
    }
    std::fs::write(path, bytes).unwrap();
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_parameter_overlays_are_atomic_input_dependent_and_reversible() {
    let root = fixture(false);
    let reference = fixture(false);
    let original_bytes = std::fs::read(root.0.join("model.safetensors")).unwrap();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let prefix = [1, 2, 5, 7];
    let other = [7, 5, 2, 1];
    let baseline = parameter_logits(&mut model, &prefix).0;
    let other_baseline = parameter_logits(&mut model, &other).0;
    // Independent experimental owners can share the same source artifact.
    let (mut peer, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    assert_eq!(parameter_logits_mode(&mut peer, &prefix, true).0, baseline);
    let peer_facts = peer.parameter_discovery().unwrap();
    let facts = model.parameter_discovery().unwrap();
    let limits = CaptureUsage {
        captures: 128,
        retained_bytes: 256 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    verify_native_projections(&mut model, limits);
    let specs = [
        ("self_attn.q_proj.weight", false),
        ("self_attn.k_proj.weight", false),
        ("self_attn.v_proj.weight", false),
        ("self_attn.o_proj.weight", true),
        ("mlp.gate_proj.weight", false),
        ("mlp.up_proj.weight", false),
        ("mlp.down_proj.weight", true),
    ];
    let edits: Vec<_> = specs
        .into_iter()
        .enumerate()
        .map(|(i, (suffix, column))| {
            let parameter = format!("model.layers.0.{suffix}");
            let descriptor = facts.parameters.iter().find(|p| p.id == parameter).unwrap();
            assert!(descriptor.supported);
            let region = if column {
                ParameterRegion {
                    starts: vec![0, 1],
                    shape: vec![descriptor.shape[0], 1],
                }
            } else {
                ParameterRegion {
                    starts: vec![1, 0],
                    shape: vec![1, descriptor.shape[1]],
                }
            };
            let count = region.shape.iter().product::<u64>() as usize;
            ParameterEdit {
                id: format!("e{i}"),
                parameter,
                parameter_shape: descriptor.shape.clone(),
                dtype: InterventionDtype::Float32,
                region,
                update: ParameterUpdate::Add {
                    values: (0..count)
                        .map(|j| ((j + i) % 5) as f32 * 0.12 - 0.19)
                        .collect(),
                },
            }
        })
        .collect();
    let before = model
        .query_parameter(
            &facts.identity,
            &edits[0].parameter,
            edits[0].region.clone(),
            limits,
        )
        .unwrap();
    let admitted = model
        .admit_parameter_overlay(ParameterOverlayPlan {
            schema_version: PARAMETER_SCHEMA_VERSION,
            base_identity: facts.identity.clone(),
            provenance: "independent F32 checkpoint byte edits".into(),
            edits: edits.clone(),
        })
        .unwrap();
    assert!(matches!(
        model.activate_parameter_overlay(&admitted, CaptureUsage::default()),
        Err(ParameterError::Budget(_))
    ));
    assert_eq!(parameter_logits(&mut model, &prefix).0, baseline);
    let active = model.activate_parameter_overlay(&admitted, limits).unwrap();
    assert!(matches!(
        peer.activate_parameter_overlay(&admitted, limits),
        Err(ParameterError::StaleIdentity)
    ));
    assert_eq!(peer.parameter_discovery().unwrap(), peer_facts);
    assert_eq!(
        parameter_logits_mode(&mut peer, &prefix, true),
        (baseline.clone(), None)
    );
    assert_eq!(parameter_logits(&mut peer, &other).0, other_baseline);
    verify_native_projections(&mut model, limits);
    assert_eq!(
        active.overlay_identity.as_deref(),
        Some(admitted.identity())
    );
    assert!(matches!(
        model.query_parameter(
            &facts.identity,
            &edits[0].parameter,
            edits[0].region.clone(),
            limits
        ),
        Err(ParameterError::StaleIdentity)
    ));
    let after = model
        .query_parameter(
            &active.identity,
            &edits[0].parameter,
            edits[0].region.clone(),
            limits,
        )
        .unwrap();
    for ((old, delta), actual) in before
        .values
        .iter()
        .zip(edits[0].update.values())
        .zip(&after.values)
    {
        assert_eq!(old + delta, *actual);
    }
    let (modified, identity) = parameter_logits(&mut model, &prefix);
    assert_eq!(identity.as_deref(), Some(admitted.identity()));
    let (other_modified, _) = parameter_logits(&mut model, &other);
    assert_eq!(
        parameter_logits_mode(&mut model, &prefix, true),
        (modified.clone(), identity)
    );
    assert_ne!(modified, baseline);
    assert_ne!(other_modified, other_baseline);
    assert_ne!(modified, other_modified);
    edit_reference(&reference.0, &edits);
    let (mut oracle, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &reference.0, &execution)
            .unwrap()
            .into_parts();
    assert_eq!(modified, parameter_logits(&mut oracle, &prefix).0);
    assert_eq!(other_modified, parameter_logits(&mut oracle, &other).0);
    let restored = model.remove_parameter_overlay(&active.identity).unwrap();
    verify_native_projections(&mut model, limits);
    assert!(restored.overlay_identity.is_none());
    assert!(restored.usage.host_bytes >= after.usage.host_bytes);
    assert_eq!(parameter_logits(&mut model, &prefix), (baseline, None));
    assert_eq!(parameter_logits(&mut model, &other), (other_baseline, None));
    assert_eq!(
        std::fs::read(root.0.join("model.safetensors")).unwrap(),
        original_bytes
    );

    // Completed native validation rejects the second candidate without publishing
    // the first, fencing a healthy session, or refunding the completed work.
    let overflow_fixture = fixture(false);
    let mut overflowing = edits[1].clone();
    overflowing.update = ParameterUpdate::Replace {
        values: vec![f32::MAX; overflowing.update.values().len()],
    };
    edit_reference(&overflow_fixture.0, &[overflowing.clone()]);
    let (mut overflow_model, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default(),
        &overflow_fixture.0,
        &execution,
    )
    .unwrap()
    .into_parts();
    let overflow_facts = overflow_model.parameter_discovery().unwrap();
    let before = overflow_model
        .query_parameter(
            &overflow_facts.identity,
            &edits[0].parameter,
            edits[0].region.clone(),
            limits,
        )
        .unwrap()
        .values;
    overflowing.update = ParameterUpdate::Add {
        values: vec![f32::MAX; overflowing.update.values().len()],
    };
    let authority = overflow_model
        .admit_parameter_overlay(ParameterOverlayPlan {
            schema_version: PARAMETER_SCHEMA_VERSION,
            base_identity: overflow_facts.identity.clone(),
            provenance: "atomic publication under native F32 overflow".into(),
            edits: vec![edits[0].clone(), overflowing],
        })
        .unwrap();
    assert!(matches!(
        overflow_model.activate_parameter_overlay(&authority, limits),
        Err(ParameterError::Invalid(_))
    ));
    let overflow_projection = ParameterProjection {
        region: edits[1].region.clone(),
        axis: 1,
        directions: 1,
        coefficients: vec![2.0; edits[1].region.shape[1] as usize],
    };
    assert!(matches!(
        overflow_model.project_parameter(
            &overflow_facts.identity,
            &edits[1].parameter,
            overflow_projection,
            limits
        ),
        Err(ParameterError::Invalid(_))
    ));
    let unchanged = overflow_model.parameter_discovery().unwrap();
    assert_eq!(unchanged.identity, overflow_facts.identity);
    assert!(unchanged.overlay_identity.is_none());
    assert!(unchanged.usage.retained_bytes > overflow_facts.usage.retained_bytes);
    assert_eq!(
        overflow_model
            .query_parameter(
                &unchanged.identity,
                &edits[0].parameter,
                edits[0].region.clone(),
                limits
            )
            .unwrap()
            .values,
        before
    );
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx"
)]
fn native_shared_invocation_edits_apply_to_every_parameter_replica_and_restore() {
    verify_shared_parameter_edits(eredu_core::ResidencyPlan::FullyResident);
}

pub(super) fn verify_shared_parameter_edits(residency: eredu_core::ResidencyPlan) {
    let root = fixture(false);
    let reference = fixture(false);
    use_nanbeige_weights(&root.0);
    use_nanbeige_weights(&reference.0);
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_residency(residency);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let limits = CaptureUsage {
        captures: 256,
        retained_bytes: 512 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    let facts = model.parameter_discovery().unwrap();
    let first = facts
        .parameters
        .iter()
        .find(|p| p.id == "model.layers.0.mlp.down_proj.weight")
        .unwrap();
    let repeated = facts
        .parameters
        .iter()
        .find(|p| p.id == "model.layers.2.mlp.down_proj.weight")
        .unwrap();
    assert_eq!(first.shared_id, repeated.shared_id);
    let region = ParameterRegion {
        starts: vec![0, 1],
        shape: vec![first.shape[0], 1],
    };
    let query = |model: &mut LoadedModel<_>, id: &str| {
        model
            .query_parameter(&facts.identity, id, region.clone(), limits)
            .unwrap()
            .values
    };
    assert_eq!(
        query(&mut model, &first.id),
        query(&mut model, &repeated.id)
    );
    let prefix = [1, 2, 5, 7];
    let baseline = parameter_logits(&mut model, &prefix).0;
    let edit = ParameterEdit {
        id: "shared-write".into(),
        parameter: first.id.clone(),
        parameter_shape: first.shape.clone(),
        dtype: InterventionDtype::Float32,
        update: ParameterUpdate::Add {
            values: (0..first.shape[0])
                .map(|i| (i % 5) as f32 * 0.1 - 0.17)
                .collect(),
        },
        region: region.clone(),
    };
    let plan = ParameterOverlayPlan {
        schema_version: PARAMETER_SCHEMA_VERSION,
        base_identity: facts.identity.clone(),
        provenance: "shared invocation replica reference".into(),
        edits: vec![edit.clone()],
    };
    let mut conflict = plan.clone();
    let mut alias = edit.clone();
    alias.id = "alias-overlap".into();
    alias.parameter = repeated.id.clone();
    conflict.edits.push(alias);
    assert!(matches!(
        model.admit_parameter_overlay(conflict),
        Err(ParameterError::Conflict(_))
    ));
    let overlay = model.admit_parameter_overlay(plan).unwrap();
    let active = model.activate_parameter_overlay(&overlay, limits).unwrap();
    let one = model
        .query_parameter(&active.identity, &first.id, region.clone(), limits)
        .unwrap()
        .values;
    let two = model
        .query_parameter(&active.identity, &repeated.id, region, limits)
        .unwrap()
        .values;
    assert_eq!(one, two);
    edit_reference(&reference.0, &[edit]);
    let (mut expected, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &reference.0, &execution)
            .unwrap()
            .into_parts();
    let changed = parameter_logits_mode(&mut model, &prefix, true).0;
    assert_ne!(changed, baseline);
    assert_eq!(changed, parameter_logits(&mut expected, &prefix).0);
    model.remove_parameter_overlay(&active.identity).unwrap();
    assert_eq!(parameter_logits(&mut model, &prefix).0, baseline);
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct ParameterDecodeStep {
    pub token: u32,
    pub values: Vec<f32>,
    pub dtype: eredu_core::checkpoint::TensorDtype,
    pub overlay: Option<String>,
}

/// Three unforced predictions, including two cached decode submissions. The
/// controlled path also restores the post-prefill cache and replays both steps.
pub(super) fn parameter_decode_logits<
    B: eredu_runtime::execution_control::TextSnapshotBackend
        + eredu_runtime::execution_control::TextSamplingControlBackend,
>(
    model: &mut LoadedModel<B>,
    prefix: &[u32],
    controlled: bool,
) -> Vec<ParameterDecodeStep> {
    model.reset().unwrap();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user","content":"left"})],
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let usage = CaptureUsage {
        captures: 16,
        retained_bytes: 64 << 20,
        host_bytes: 4 << 20,
        encoded_bytes: 4 << 20,
    };
    let prepared = model
        .prepare_observed_token_ids(
            &chat,
            prefix.to_vec(),
            PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    temperature: Some(0.0),
                    max_new_tokens: Some(3),
                    ..Default::default()
                },
                seed: 17,
                ..Default::default()
            },
            CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: vec![CaptureSelection {
                    id: "cached-logits".into(),
                    path: "model.logits".into(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::FullTensor,
                }],
                limits: CaptureLimits {
                    per_step: usage,
                    cumulative: usage,
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            },
            TraceLimits {
                per_record_bytes: 1 << 20,
                total_bytes: 8 << 20,
            },
        )
        .unwrap();
    let extract = |records: &[ObservedGenerationRecord], first: usize| {
        let result: Vec<_> = records
            .iter()
            .filter_map(|record| {
                let ObservedGenerationEvent::Token {
                    token_id,
                    forced,
                    prediction_index,
                    input_range,
                    captures: Some(step),
                    ..
                } = &record.event
                else {
                    return None;
                };
                assert!(!forced);
                let expected_range = if *prediction_index == 0 {
                    [0, prefix.len() as u64]
                } else {
                    [
                        prefix.len() as u64 + prediction_index - 1,
                        prefix.len() as u64 + prediction_index,
                    ]
                };
                assert_eq!(*input_range, expected_range);
                assert_eq!(step.records.len(), 1);
                let capture = &step.records[0];
                assert_eq!(capture.outcome, CaptureOutcome::Captured);
                let Some(CapturePayload::Tensor(tensor)) = &capture.payload else {
                    panic!("missing cached logits");
                };
                let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                    panic!("cached logits host dtype");
                };
                Some((
                    *prediction_index as usize,
                    ParameterDecodeStep {
                        token: *token_id,
                        values: values.clone(),
                        dtype: capture.source_dtype.clone().unwrap(),
                        overlay: record.parameter_overlay_id.clone(),
                    },
                ))
            })
            .collect();
        assert_eq!(
            result.len(),
            3 - first,
            "fixture must execute both cached decode steps"
        );
        for (i, (prediction, _)) in result.iter().enumerate() {
            assert_eq!(*prediction, first + i);
        }
        result.into_iter().map(|(_, step)| step).collect::<Vec<_>>()
    };
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
        run.step(|e| {
            records.push(e.generation);
            ControlFlow::Continue(())
        })
        .unwrap();
        let snapshot = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
        run.run(|e| {
            records.push(e.generation);
            ControlFlow::Continue(())
        })
        .unwrap();
        let original = extract(&records, 0);
        records.clear();
        run.restore(&snapshot, |_| ControlFlow::Continue(()))
            .unwrap();
        run.run(|e| {
            records.push(e.generation);
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(original[1..], extract(&records, 1));
        original
    } else {
        model
            .generate_observed_text(prepared, &[], Default::default(), |e| {
                records.push(e);
                ControlFlow::Continue(())
            })
            .unwrap();
        extract(&records, 0)
    }
}

pub(super) fn compare_parameter_decodes(
    actual: &[ParameterDecodeStep],
    expected: &[ParameterDecodeStep],
    overlay: Option<&str>,
) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.token, expected.token);
        assert_eq!(actual.dtype, expected.dtype);
        assert_eq!(actual.overlay.as_deref(), overlay);
        assert_eq!(actual.values.len(), expected.values.len());
        for (actual, expected) in actual.values.iter().zip(&expected.values) {
            assert!(
                (actual - expected).abs() <= 3e-5 + 3e-5 * expected.abs(),
                "cached logits {actual} != {expected}"
            );
        }
    }
}
