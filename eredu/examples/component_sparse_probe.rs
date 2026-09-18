//! Released sparse-checkpoint evidence through public portable contracts.
//! Pair with eredu-evaluation/scripts/component_sparse_reference.py.
mod component_sparse_analysis;
mod component_sparse_scores;
use anyhow::{Context, ensure};
use eredu::api::*;
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    ArchitectureDescriptor, ExecutionPlan, GenerationConfigOverrides, capture::*, component::*,
    execution_control::SnapshotLimits, intervention::*, parameters::*,
};
use serde_json::{Value, json};
use std::{collections::BTreeMap, ops::ControlFlow};

fn dense_group<'a>(
    graph: &'a ArchitectureDescriptor,
    kind: &str,
    layer: usize,
) -> anyhow::Result<&'a ComponentGroup> {
    graph
        .components
        .iter()
        .find(|g| {
            g.layer_index == layer
                && matches!(g.activation_equation, ComponentActivation::Attention { .. })
                    == (kind == "attention")
        })
        .context("declared scalar component group")
}

fn contains_node(graph: &ArchitectureDescriptor, ancestor: &str, descendant: &str) -> bool {
    let mut current = Some(descendant);
    for _ in 0..=graph.nodes.len() {
        let Some(node) = current else { return false };
        if node == ancestor {
            return true;
        }
        current = graph.node(node).and_then(|node| node.parent.as_deref());
    }
    false
}

fn sequence_capture(
    selections: &mut Vec<CaptureSelection>,
    path: &str,
    label: &str,
    prefix_len: usize,
    all_positions: bool,
) {
    for prefill in [true, false] {
        let row = if prefill { prefix_len as u64 - 1 } else { 0 };
        selections.push(CaptureSelection {
            id: format!("{prefill}:{label}"),
            path: path.into(),
            schedule: CaptureSchedule {
                prefill,
                decode: !prefill,
                ..Default::default()
            },
            slices: vec![CaptureSlice {
                axis: "sequence".into(),
                start: if all_positions { 0 } else { row },
                end: row + 1,
                stride: 1,
            }],
            transform: CaptureTransform::Slice,
        });
    }
}

fn parameter_selection<'a>(
    graph: &ArchitectureDescriptor,
    facts: &'a ParameterDiscovery,
    selector: &Value,
) -> anyhow::Result<(&'a LoadedParameter, ParameterRegion)> {
    let kind = selector["kind"].as_str().context("component kind")?;
    let layer = selector["layer"].as_u64().context("component layer")? as usize;
    let unit = selector["unit"].as_u64().context("component unit")? as usize;
    let role = selector["role"].as_str().context("component role")?;
    let read_role = match role {
        "gate" => ComponentReadRole::Gate,
        "value" => ComponentReadRole::Value,
        "query" => ComponentReadRole::Query,
        "key" => ComponentReadRole::Key,
        "write" => ComponentReadRole::Input,
        _ => anyhow::bail!("unknown component role {role}"),
    };
    if kind == "routed" {
        let group = graph
            .routed_components
            .iter()
            .find(|g| g.layer_index == layer)
            .context("routed group")?;
        let id = RoutedComponentId {
            group: group.id.clone(),
            expert: selector["expert"].as_u64().context("expert")? as usize,
            index: unit,
        };
        let selected = if role == "write" {
            group.write_column(&id, facts)?
        } else {
            group.read_weight(&id, read_role, facts)?
        };
        return Ok((selected.parameter, selected.region));
    }
    let group = dense_group(graph, kind, layer)?;
    ensure!(unit < group.count, "component outside group");
    let (name, row) = if role == "write" {
        (&group.write_weight, None)
    } else {
        let read = group
            .reads
            .iter()
            .find(|r| r.role == read_role)
            .context("declared read role")?;
        let rows = read.rows.row_range(unit).context("component read rows")?;
        // The reference selects one row from the declared head dependency.
        let row = if rows.len() > 1 {
            rows.start + unit % rows.len()
        } else {
            rows.start
        };
        (&read.weight, Some(row))
    };
    let parameter = facts
        .parameters
        .iter()
        .find(|p| &p.id == name)
        .context("loaded parameter")?;
    ensure!(parameter.shape.len() == 2, "scalar component matrix rank");
    let region = if let Some(row) = row {
        ParameterRegion {
            starts: vec![row as u64, 0],
            shape: vec![1, parameter.shape[1]],
        }
    } else {
        ParameterRegion {
            starts: vec![0, unit as u64],
            shape: vec![parameter.shape[0], 1],
        }
    };
    Ok((parameter, region))
}

fn record_step(
    event: ControlledGenerationRecord,
    paths: &BTreeMap<String, (String, &str)>,
    precision: &mut BTreeMap<String, eredu_core::checkpoint::TensorDtype>,
    steps: &mut BTreeMap<u64, Value>,
    measure_precision: bool,
) -> ControlFlow<()> {
    if let Some(ObservedGenerationEvent::Token {
        prediction_index,
        token_id,
        forced,
        captures: Some(step),
        ..
    }) = event.event.progress()
    {
        assert!(!forced);
        let entry = steps.entry(*prediction_index).or_insert_with(
            || json!({"prediction":prediction_index,"token":token_id,"captures":{}}),
        );
        for record in &step.records {
            if matches!(
                record.outcome,
                CaptureOutcome::Skipped {
                    reason: CaptureSkipReason::Schedule
                }
            ) {
                continue;
            }
            assert_eq!(record.outcome, CaptureOutcome::Captured, "{record:?}");
            if measure_precision {
                if let Some(dtype) = &record.source_dtype {
                    precision.insert(record.path.clone(), dtype.clone());
                }
            }
            match &record.payload {
                Some(CapturePayload::Tensor(tensor)) => {
                    let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                        panic!("host float values")
                    };
                    if record.path == "model.logits" {
                        entry["logits"] = json!(values);
                    } else {
                        let (key, mode) = &paths[&record.path];
                        if *mode == "write" {
                            entry["writes"][key] = json!(values);
                        } else if *mode == "boundary" {
                            entry["boundaries"][key] = json!(values);
                        } else if *mode == "readout" {
                            entry["readout"][key] = json!(values);
                        } else if matches!(
                            key.as_str(),
                            "head_input" | "convolution_input" | "routed_write"
                        ) {
                            entry[key] = json!(values);
                        } else {
                            entry["captures"][key][*mode] = json!(values);
                        }
                    }
                }
                Some(CapturePayload::RoutedUnits(payload)) => {
                    let (key, mode) = &paths[&record.path];
                    let rows = payload.rows.iter().map(|row| {
                                let eredu_core::TensorObservationData::F32(values) = row.values.data() else { panic!("routed host values") };
                                json!({"token":row.token,"slot":row.slot,"expert":row.expert,"coefficient":row.coefficient,"values":values})
                            }).collect::<Vec<_>>();
                    entry["captures"][key][*mode] = json!(rows);
                }
                other => panic!("unexpected capture {other:?}"),
            }
        }
    }
    ControlFlow::Continue(())
}

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let reference: Value = serde_json::from_slice(&std::fs::read(args.first().context(
        "usage: component_sparse_probe <reference.json> <output.json> [cpu|gpu] [controlled] [writes|scores]",
    )?)?)?;
    let score_mode = args.iter().skip(3).any(|arg| arg == "scores");
    let write_mode = score_mode || args.iter().skip(3).any(|arg| arg == "writes");
    ensure!(
        !score_mode || reference["capture_full_readout"].as_bool() == Some(true),
        "scores require a reference with full readout captures"
    );
    let path = reference["provenance"]["path"]
        .as_str()
        .context("checkpoint path")?;
    let prefix: Vec<u32> = serde_json::from_value(reference["prefix_ids"].clone())?;
    let graph = inspect_architecture(path)?;
    let device = match args.get(2).map(String::as_str).unwrap_or("cpu") {
        "cpu" => LocalDevice::Cpu,
        "gpu" => LocalDevice::Accelerator(0),
        other => anyhow::bail!("unknown device {other}"),
    };
    let execution = ExecutionPlan::fully_resident(local_device_plan(device)?);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &execution)?
            .into_parts();
    eprintln!("loaded sparse execution");
    let facts = model.parameter_discovery()?;
    let mut parameter_budget = facts.usage.checked_add(CaptureUsage {
        captures: 100_000,
        retained_bytes: 128 << 30,
        host_bytes: 2 << 30,
        encoded_bytes: 2 << 30,
    })?;
    if write_mode {
        parameter_budget.retained_bytes = parameter_budget
            .retained_bytes
            .checked_add(component_sparse_analysis::retention_allowance(
                &graph,
                &facts,
                &reference,
                prefix.len() as u64 - 1,
                score_mode,
            )?)
            .context("parameter reconstruction budget overflow")?;
    }
    let mut queries = vec![];
    let mut edits = vec![];
    for (index, query) in reference["queries"]
        .as_array()
        .context("queries")?
        .iter()
        .enumerate()
    {
        let (parameter, region) = parameter_selection(&graph, &facts, &query["selector"])?;
        let actual = model.query_parameter(
            &facts.identity,
            &parameter.id,
            region.clone(),
            parameter_budget,
        )?;
        let expected: Vec<f32> = serde_json::from_value(query["original"].clone())?;
        ensure!(
            actual.values == expected,
            "effective read differs for {}",
            query["selector"]
        );
        let axis = if query["selector"]["role"] == "write" {
            region.shape.len() - 2
        } else {
            region.shape.len() - 1
        };
        let coefficients = (0..region.shape[axis])
            .map(|i| (i as i32 % 7 - 3) as f32 / 8.)
            .collect();
        let projected = model.project_parameter(
            &facts.identity,
            &parameter.id,
            ParameterProjection {
                region: region.clone(),
                axis,
                directions: 1,
                coefficients,
            },
            parameter_budget,
        )?;
        let expected_projection = query["projection"].as_f64().context("projection")?;
        ensure!(
            projected.values.len() == 1
                && (projected.values[0] as f64 - expected_projection).abs()
                    <= 2e-5 + 2e-5 * expected_projection.abs(),
            "signed parameter projection"
        );
        queries.push(
            json!({"selector":query["selector"], "parameter":parameter.id,
            "projection":projected.values[0], "values":actual.values}),
        );
        edits.push(ParameterEdit {
            id: format!("edit-{index}"),
            parameter: parameter.id.clone(),
            parameter_shape: parameter.shape.clone(),
            dtype: parameter.dtype.context("parameter dtype")?,
            region,
            update: ParameterUpdate::Replace {
                values: serde_json::from_value(query["replacement"].clone())?,
            },
        });
    }
    let mut paths = BTreeMap::new();
    let mut selections = vec![];
    if let Some(boundaries) = reference["diagnostic_boundaries"].as_object() {
        for (key, path) in boundaries {
            let path = path.as_str().context("diagnostic boundary path")?;
            ensure!(
                graph.observations.get(path).is_some(),
                "undeclared boundary {path}"
            );
            paths.insert(path.into(), (key.clone(), "boundary"));
            sequence_capture(
                &mut selections,
                path,
                &format!("boundary:{key}"),
                prefix.len(),
                reference["capture_all_positions"].as_bool() == Some(true),
            );
        }
    }
    for selected in reference["captured"]
        .as_array()
        .context("capture selectors")?
    {
        let kind = selected["kind"].as_str().context("kind")?;
        let layer = selected["layer"].as_u64().context("layer")? as usize;
        let (original, effective) = if kind == "routed" {
            let g = graph
                .routed_components
                .iter()
                .find(|g| g.layer_index == layer)
                .context("routed capture")?;
            (&g.activation, &g.effective_activation)
        } else {
            let g = dense_group(&graph, kind, layer)?;
            (&g.activation, &g.effective_activation)
        };
        for (mode, path) in [("original", original), ("effective", effective)] {
            paths.insert(path.clone(), (format!("{kind}:{layer}"), mode));
            for prefill in [true, false] {
                let row = if prefill { prefix.len() as u64 - 1 } else { 0 };
                selections.push(CaptureSelection {
                    id: format!("{prefill}:{path}"),
                    path: path.clone(),
                    schedule: CaptureSchedule {
                        prefill,
                        decode: !prefill,
                        ..Default::default()
                    },
                    slices: vec![CaptureSlice {
                        axis: if kind == "routed" {
                            "token"
                        } else {
                            "sequence"
                        }
                        .into(),
                        start: if reference["capture_all_positions"].as_bool() == Some(true) {
                            0
                        } else {
                            row
                        },
                        end: row + 1,
                        stride: 1,
                    }],
                    transform: if kind == "routed" {
                        CaptureTransform::RoutedUnits
                    } else {
                        CaptureTransform::Slice
                    },
                });
            }
        }
    }
    if reference["capture_writes"].as_bool() == Some(true) {
        for selected in reference["captured"]
            .as_array()
            .context("write selectors")?
        {
            let kind = selected["kind"].as_str().context("write kind")?;
            let layer = selected["layer"].as_u64().context("write layer")? as usize;
            let path = if kind == "routed" {
                let group = graph
                    .routed_components
                    .iter()
                    .find(|g| g.layer_index == layer)
                    .context("routed write group")?;
                &graph
                    .component_readout
                    .as_ref()
                    .context("readout")?
                    .other_writes
                    .iter()
                    .find(|write| contains_node(&graph, &write.node_id, &group.node_id))
                    .context("whole routed write")?
                    .effective_output
            } else {
                dense_group(&graph, kind, layer)?
                    .write_output
                    .as_ref()
                    .context("component affine write")?
            };
            let key = format!("{kind}:{layer}");
            paths.insert(path.clone(), (key.clone(), "write"));
            sequence_capture(
                &mut selections,
                path,
                &format!("write:{key}"),
                prefix.len(),
                reference["capture_all_positions"].as_bool() == Some(true),
            );
        }
    }
    for prefill in [true, false] {
        let row = if prefill { prefix.len() as u64 - 1 } else { 0 };
        selections.push(CaptureSelection {
            id: format!("{prefill}:logits"),
            path: "model.logits".into(),
            schedule: CaptureSchedule {
                prefill,
                decode: !prefill,
                ..Default::default()
            },
            slices: vec![CaptureSlice {
                axis: "sequence".into(),
                start: row,
                end: row + 1,
                stride: 1,
            }],
            transform: CaptureTransform::Slice,
        });
    }
    if reference["capture_head_input"].as_bool() == Some(true) {
        let readout = graph
            .component_readout
            .as_ref()
            .context("declared readout")?;
        let path = readout
            .projection_input
            .as_ref()
            .unwrap_or(&readout.normalized);
        paths.insert(path.clone(), ("head_input".into(), "effective"));
        for prefill in [true, false] {
            let row = if prefill { prefix.len() as u64 - 1 } else { 0 };
            selections.push(CaptureSelection {
                id: format!("{prefill}:head_input"),
                path: path.clone(),
                schedule: CaptureSchedule {
                    prefill,
                    decode: !prefill,
                    ..Default::default()
                },
                slices: vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: row,
                    end: row + 1,
                    stride: 1,
                }],
                transform: CaptureTransform::Slice,
            });
        }
    }
    if let Some(layer) = reference["convolution_input_layer"].as_u64() {
        let routed = graph
            .routed_components
            .iter()
            .find(|group| group.layer_index == layer as usize)
            .context("routed invocation following the convolution")?;
        let readout = graph
            .component_readout
            .as_ref()
            .context("declared readout")?;
        let whole = readout
            .other_writes
            .iter()
            .find(|write| {
                write.layer_index == layer as usize
                    && !contains_node(&graph, &write.node_id, &routed.node_id)
            })
            .context("declared whole token-mixer write")?;
        let path = whole.input.as_ref().context("declared token-mixer input")?;
        if reference["capture_writes"].as_bool() == Some(true) {
            let key = format!("convolution:{layer}");
            paths.insert(whole.effective_output.clone(), (key.clone(), "write"));
            sequence_capture(
                &mut selections,
                &whole.effective_output,
                &format!("write:{key}"),
                prefix.len(),
                reference["capture_all_positions"].as_bool() == Some(true),
            );
        }
        paths.insert(path.clone(), ("convolution_input".into(), "effective"));
        for prefill in [true, false] {
            let row = if prefill { prefix.len() as u64 - 1 } else { 0 };
            selections.push(CaptureSelection {
                id: format!("{prefill}:convolution_input"),
                path: path.clone(),
                schedule: CaptureSchedule {
                    prefill,
                    decode: !prefill,
                    ..Default::default()
                },
                slices: vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: if reference["capture_all_positions"].as_bool() == Some(true) {
                        0
                    } else {
                        row
                    },
                    end: row + 1,
                    stride: 1,
                }],
                transform: CaptureTransform::Slice,
            });
        }
    }
    if let Some(layer) = reference["routed_write_layer"].as_u64() {
        let routed = graph
            .routed_components
            .iter()
            .find(|group| group.layer_index == layer as usize)
            .context("routed write invocation")?;
        let readout = graph
            .component_readout
            .as_ref()
            .context("declared readout")?;
        let whole = readout
            .other_writes
            .iter()
            .find(|write| contains_node(&graph, &write.node_id, &routed.node_id))
            .context("declared routed residual write")?;
        let path = &whole.effective_output;
        paths.insert(path.clone(), ("routed_write".into(), "effective"));
        for prefill in [true, false] {
            let row = if prefill { prefix.len() as u64 - 1 } else { 0 };
            selections.push(CaptureSelection {
                id: format!("{prefill}:routed_write"),
                path: path.clone(),
                schedule: CaptureSchedule {
                    prefill,
                    decode: !prefill,
                    ..Default::default()
                },
                slices: vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: if reference["capture_all_positions"].as_bool() == Some(true) {
                        0
                    } else {
                        row
                    },
                    end: row + 1,
                    stride: 1,
                }],
                transform: CaptureTransform::Slice,
            });
        }
    }
    if reference["capture_full_readout"].as_bool() == Some(true) {
        let readout = graph
            .component_readout
            .as_ref()
            .context("complete readout declaration")?;
        for (key, path) in [
            ("embedding", &readout.embedding),
            ("residual", &readout.residual),
        ] {
            let path = format!("{path}.effective");
            ensure!(
                graph.observations.get(&path).is_some(),
                "missing effective readout {path}"
            );
            paths.insert(path.clone(), (key.into(), "readout"));
            sequence_capture(
                &mut selections,
                &path,
                &format!("readout:{key}"),
                prefix.len(),
                false,
            );
        }
        for write in &readout.other_writes {
            if graph
                .routed_components
                .iter()
                .any(|group| contains_node(&graph, &write.node_id, &group.node_id))
            {
                continue;
            }
            if paths.contains_key(&write.effective_output) {
                continue;
            }
            ensure!(
                graph
                    .node(&write.node_id)
                    .is_some_and(|node| node.kind == eredu_core::ArchitectureNodeKind::Mixer),
                "fixture expects complete mixer writes"
            );
            let key = format!("convolution:{}", write.layer_index);
            paths.insert(write.effective_output.clone(), (key.clone(), "write"));
            sequence_capture(
                &mut selections,
                &write.effective_output,
                &format!("write:{key}"),
                prefix.len(),
                reference["capture_all_positions"].as_bool() == Some(true),
            );
        }
    }
    let budget = CaptureUsage {
        captures: 4096,
        retained_bytes: 8 << 30,
        host_bytes: 256 << 20,
        encoded_bytes: 256 << 20,
    };
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections,
        limits: CaptureLimits {
            per_step: budget,
            cumulative: budget.checked_mul(16)?,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let trace = if reference["capture_all_positions"].as_bool() == Some(true)
        || reference["capture_full_readout"].as_bool() == Some(true)
    {
        // The full released prefill includes 32 MiB of component JSON before
        // record metadata and logits, so reserve a bounded larger record.
        TraceLimits {
            per_record_bytes: 64 << 20,
            total_bytes: 256 << 20,
        }
    } else {
        TraceLimits {
            per_record_bytes: 16 << 20,
            total_bytes: 128 << 20,
        }
    };
    const CAPACITY: u64 = 64 << 30;
    let cancellation = eredu_core::GenerationCancellationToken::new();
    let tokenizer =
        model.compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)?;
    let source = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancellation,
        )?
        .context("cancelled before source compilation")?;
    let policy = ChatTemplateRequest {
        messages: vec![json!({"role":"user","content":"reference output contract"})],
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = model
        .prepare_chat(&source, &policy, CAPACITY, &cancellation)?
        .context("cancelled before chat preparation")?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.),
            max_new_tokens: Some(4),
            ..Default::default()
        },
        inference: eredu_core::TextInferencePolicy {
            managed_memory_capacity_bytes: Some(CAPACITY),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    let mut output = json!({"provenance":reference["provenance"], "queries":queries, "trials":{}});
    let mut precision = BTreeMap::new();
    let mut active = None;
    for trial in ["baseline", "deletion", "keep_only", "overlay", "restored"] {
        if reference["trials"].get(trial).is_none() {
            continue;
        }
        model.reset()?;
        if trial == "overlay" {
            let facts = model.parameter_discovery()?;
            let admitted = model.admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity,
                provenance: "pinned sparse independent coordinated edit".into(),
                edits: edits.clone(),
            })?;
            let activated = model.activate_parameter_overlay(&admitted, parameter_budget)?;
            for edit in &edits {
                let actual = model.query_parameter(
                    &activated.identity,
                    &edit.parameter,
                    edit.region.clone(),
                    parameter_budget,
                )?;
                ensure!(
                    actual.values == edit.update.values(),
                    "effective replacement differs"
                );
            }
            active = Some(activated.identity);
        } else if trial == "restored" {
            model.remove_parameter_overlay(active.as_deref().context("active overlay")?)?;
        }
        let mut operations = vec![];
        if matches!(trial, "deletion" | "keep_only") {
            for selected in reference["interventions"]
                .as_array()
                .context("interventions")?
            {
                let kind = selected["kind"].as_str().context("kind")?;
                let layer = selected["layer"].as_u64().context("layer")? as usize;
                let (target, indices, axis) = if kind == "routed" {
                    let g = graph
                        .routed_components
                        .iter()
                        .find(|g| g.layer_index == layer)
                        .context("routed mask")?;
                    let indices = (0..g.expert_count)
                        .flat_map(|expert| {
                            [1, 3].map(|index| RoutedComponentId {
                                group: g.id.clone(),
                                expert,
                                index,
                            })
                        })
                        .map(|id| g.component_index(&id))
                        .collect::<Result<Vec<_>, _>>()?;
                    (&g.activation, indices, "token")
                } else {
                    let g = dense_group(&graph, kind, layer)?;
                    (&g.activation, vec![1, 3], "sequence")
                };
                let dtype = match precision.get(target) {
                    Some(eredu_core::checkpoint::TensorDtype::F32) => InterventionDtype::Float32,
                    Some(eredu_core::checkpoint::TensorDtype::Bf16) => InterventionDtype::Bfloat16,
                    Some(eredu_core::checkpoint::TensorDtype::F16) => InterventionDtype::Float16,
                    other => anyhow::bail!("unmeasured precision {target}: {other:?}"),
                };
                operations.push(InterventionOperation {
                    id: target.clone(),
                    target: target.clone(),
                    schedule: CaptureSchedule {
                        decode: false,
                        ..Default::default()
                    },
                    slices: vec![CaptureSlice {
                        axis: axis.into(),
                        start: prefix.len() as u64 - 1,
                        end: prefix.len() as u64,
                        stride: 1,
                    }],
                    action: InterventionAction::MaskComponents {
                        dtype,
                        indices,
                        keep_selected: trial == "keep_only",
                    },
                    evidence: InterventionEvidence::None,
                });
            }
        }
        let intervention = InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations,
        };
        let mut request = PreparedChatRequest::new(&chat, settings);
        request.input = PreparedChatPrompt::TokenIds(&prefix);
        request.output_mode = PreparedChatOutputMode::Text;
        request.capture = Some(&capture);
        request.intervention = Some(&intervention);
        let mut steps = BTreeMap::<u64, Value>::new();
        let mut collect = |event| {
            record_step(
                event,
                &paths,
                &mut precision,
                &mut steps,
                trial == "baseline",
            )
        };
        let mut run = model
            .start_controlled_chat(request, trace, Default::default(), &mut collect)?
            .context("cancelled before sparse reference trial")?;
        run.run(&mut collect)?;
        drop(run);
        output["trials"][trial] = json!(steps.into_values().collect::<Vec<_>>());
        if score_mode {
            let mut reports = vec![];
            for step in output["trials"][trial]
                .as_array()
                .context("captured predictions")?
            {
                let prediction = step["prediction"].as_u64().context("prediction index")? as usize;
                let targets: [u32; 2] =
                    serde_json::from_value(reference["score_targets"][prediction].clone())?;
                reports.push(component_sparse_analysis::reconstruct(
                    &mut model,
                    &graph,
                    reference["captured"]
                        .as_array()
                        .context("captured groups")?,
                    step,
                    if prediction == 0 {
                        prefix.len() as u64 - 1
                    } else {
                        0
                    },
                    parameter_budget,
                    Some(targets),
                )?);
            }
            output["write_reconstruction"][trial] =
                reports.first().context("prefill score report")?.clone();
            output["score_reconstruction"][trial] = json!(reports);
        } else if write_mode {
            output["write_reconstruction"][trial] = component_sparse_analysis::reconstruct(
                &mut model,
                &graph,
                reference["captured"]
                    .as_array()
                    .context("captured groups")?,
                &output["trials"][trial][0],
                prefix.len() as u64 - 1,
                parameter_budget,
                None,
            )?;
        }
        std::fs::write(
            args.get(1).context("output path")?,
            serde_json::to_vec(&output)?,
        )?;
        if args.iter().skip(3).any(|arg| arg == "controlled") {
            model.reset()?;
            let mut request = PreparedChatRequest::new(&chat, settings);
            request.input = PreparedChatPrompt::TokenIds(&prefix);
            request.output_mode = PreparedChatOutputMode::Text;
            request.capture = Some(&capture);
            request.intervention = Some(&intervention);
            let mut steps = BTreeMap::new();
            let mut run = model
                .start_controlled_chat(request, trace, Default::default(), |event| {
                    record_step(event, &paths, &mut precision, &mut steps, false)
                })?
                .context("cancelled before controlled sparse trial")?;
            let child_trace = TraceLimits {
                per_record_bytes: trace.per_record_bytes,
                total_bytes: trace.total_bytes / 4,
            };
            // Forks reserve worst-case semantic-event storage from their trace
            // allowance. Add bounded room for both small cache snapshots and
            // the copied capture plans; this is a logical reservation.
            let snapshot_retention = child_trace
                .total_bytes
                .checked_mul(std::mem::size_of::<eredu_core::SemanticEvent>() as u64 + 1)
                .and_then(|bytes| bytes.checked_add(1 << 30))
                .context("snapshot reservation overflow")?;
            run.enable_snapshots(
                SnapshotLimits {
                    max_snapshots: 2,
                    max_branches: 1,
                    retained_bytes: snapshot_retention,
                    cumulative_copy_bytes: 4 << 30,
                },
                CAPACITY,
                eredu_runtime::working_memory::WorkspaceCopyLimits::new(CAPACITY),
            )?;
            let initial = run
                .snapshot(|_| ControlFlow::Continue(()))
                .context("initial snapshot")?;
            run.step(|event| record_step(event, &paths, &mut precision, &mut steps, false))?;
            let cached = run
                .snapshot(|_| ControlFlow::Continue(()))
                .context("cached snapshot")?;
            run.run(|event| record_step(event, &paths, &mut precision, &mut steps, false))?;
            let expected = output["trials"][trial]
                .as_array()
                .context("ordinary evidence")?;
            ensure!(
                steps.into_values().collect::<Vec<_>>() == *expected,
                "controlled {trial} differs"
            );
            let mut replay = BTreeMap::new();
            run.restore(&cached, |_| ControlFlow::Continue(()))?;
            run.run(|event| record_step(event, &paths, &mut precision, &mut replay, false))?;
            ensure!(
                replay.into_values().collect::<Vec<_>>() == expected[1..],
                "cached replay {trial} differs"
            );
            let mut child = run
                .fork(
                    &initial,
                    GenerationBranchOptions {
                        trace_limits: child_trace,
                        capture_limits: Some(capture.limits.clone()),
                        sampling: None,
                        intervention: None,
                    },
                    |_| ControlFlow::Continue(()),
                )
                .context("fork initial snapshot")?;
            run.exchange(&mut child, |_| ControlFlow::Continue(()))?;
            let mut sibling = BTreeMap::new();
            run.run(|event| record_step(event, &paths, &mut precision, &mut sibling, false))?;
            ensure!(
                sibling.into_values().collect::<Vec<_>>() == *expected,
                "sibling {trial} differs"
            );
            run.exchange(&mut child, |_| ControlFlow::Continue(()))?;
            run.restore(&cached, |_| ControlFlow::Continue(()))?;
            let mut parent = BTreeMap::new();
            run.run(|event| record_step(event, &paths, &mut precision, &mut parent, false))?;
            ensure!(
                parent.into_values().collect::<Vec<_>>() == expected[1..],
                "parent after sibling {trial} differs"
            );
            output["controlled"][trial] = json!({
                "ordinary_equal": true, "cached_restore_equal": true,
                "sibling_equal": true, "parent_after_sibling_equal": true,
                "predictions_checked": 14,
                "snapshot_retained_limit": snapshot_retention,
            });
        }
        if trial == "restored" {
            ensure!(
                output["trials"][trial] == output["trials"]["baseline"],
                "native restoration differs"
            );
        }
        std::fs::write(
            args.get(1).context("output path")?,
            serde_json::to_vec(&output)?,
        )?;
        eprintln!("completed {trial}");
    }
    Ok(())
}
