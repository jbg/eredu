//! Released-checkpoint capture/intervention parity through public APIs.
//! Generate the oracle with eredu-evaluation/scripts/component_reference.py.
use anyhow::{Context, ensure};
use eredu::api::*;
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::parameters::*;
use eredu_core::{ExecutionPlan, GenerationConfigOverrides, capture::*, intervention::*};
use std::ops::ControlFlow;

#[path = "component_reference_analysis.rs"]
mod analysis;
#[path = "component_reference_association.rs"]
mod association;

fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let reference: serde_json::Value = serde_json::from_slice(&std::fs::read(
        args.first()
            .context("usage: component_reference_probe <reference.json>")?,
    )?)?;
    let path = reference["provenance"]["path"]
        .as_str()
        .context("checkpoint path")?;
    let prefix: Vec<u32> = serde_json::from_value(reference["prefix_ids"].clone())?;
    let architecture = inspect_architecture(path)?;
    let last_layer = architecture
        .components
        .iter()
        .map(|g| g.layer_index)
        .max()
        .context("component declarations")?;
    let captured_layers: Vec<usize> = reference
        .get("captured_layers")
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()?
        .unwrap_or_else(|| vec![0, last_layer]);
    let intervention_layers: Vec<usize> = reference
        .get("intervention_layers")
        .map(|value| serde_json::from_value(value.clone()))
        .transpose()?
        .unwrap_or_else(|| vec![0]);
    let paths: Vec<_> = architecture
        .components
        .iter()
        .filter(|g| captured_layers.contains(&g.layer_index))
        .flat_map(|g| [g.activation.clone(), g.effective_activation.clone()])
        .collect();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu)?);
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &execution)?
            .into_parts();
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
        messages: vec![serde_json::json!({"role":"user", "content":"reference output contract"})],
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = model
        .prepare_chat(&source, &policy, CAPACITY, &cancellation)?
        .context("cancelled before chat preparation")?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
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
    let budget = CaptureUsage {
        captures: 1024,
        retained_bytes: 1 << 30,
        host_bytes: 32 << 20,
        encoded_bytes: 64 << 20,
    };
    let trace = TraceLimits {
        per_record_bytes: 8 << 20,
        total_bytes: 64 << 20,
    };
    let mut selections = vec![];
    for (phase, row) in [("prefill", prefix.len() as u64 - 1), ("decode", 0)] {
        for path in paths
            .iter()
            .chain(std::iter::once(&"model.logits".to_string()))
        {
            selections.push(CaptureSelection {
                id: format!("{phase}:{path}"),
                path: path.clone(),
                schedule: CaptureSchedule {
                    prefill: phase == "prefill",
                    decode: phase == "decode",
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
    let readout = architecture
        .component_readout
        .as_ref()
        .context("readout declaration")?;
    let analysis_paths = architecture
        .components
        .iter()
        .map(|g| g.effective_activation.clone())
        .chain(
            architecture
                .components
                .iter()
                .filter_map(|g| g.write_input.clone()),
        )
        .chain(readout.projection_input.clone())
        .chain(
            readout
                .other_writes
                .iter()
                .map(|term| term.effective_output.clone()),
        )
        .chain(
            [
                &readout.embedding,
                &readout.residual,
                &readout.normalized,
                &readout.linear_scores,
            ]
            .into_iter()
            .map(|path| format!("{path}.effective")),
        );
    for path in analysis_paths {
        if selections
            .iter()
            .any(|selection| selection.path == path && selection.schedule.prefill)
        {
            continue;
        }
        selections.push(CaptureSelection {
            id: format!("analysis:{path}"),
            path,
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
        });
    }
    let selected_scores = ["baseline", "deletion", "keep_only", "overlay"]
        .into_iter()
        .flat_map(|trial| {
            [
                reference[trial]["prefill"]["target"].as_u64().unwrap() as u32,
                reference[trial]["prefill"]["competitor"].as_u64().unwrap() as u32,
            ]
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();
    selections.push(CaptureSelection {
        id: "full-vocabulary-scores".into(),
        path: "model.logits".into(),
        schedule: CaptureSchedule {
            decode: false,
            ..Default::default()
        },
        slices: vec![],
        transform: CaptureTransform::TokenScores {
            token_ids: selected_scores,
        },
    });
    let capture = CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections,
        limits: CaptureLimits {
            per_step: budget,
            cumulative: budget,
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    };
    let mut active_overlay = None;
    let mut baseline_evidence = None;
    let mut baseline_precision = std::collections::BTreeMap::new();
    let parameter_budget = analysis::parameter_limits();
    for trial in ["baseline", "deletion", "keep_only", "overlay", "restored"] {
        if trial == "overlay" {
            association::run(&mut model, &architecture, &reference["association"])?;
            let discovery = model.parameter_discovery()?;
            let edits: Vec<ParameterEdit> =
                serde_json::from_value(reference["parameter_edits"].clone())?;
            let mut original = vec![];
            for edit in &edits {
                original.push(
                    model
                        .query_parameter(
                            &discovery.identity,
                            &edit.parameter,
                            edit.region.clone(),
                            parameter_budget,
                        )?
                        .values,
                );
            }
            let admitted = model.admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: discovery.identity,
                provenance: format!(
                    "pinned {} {} independent Transformers coordinated parameter trial",
                    reference["provenance"]["repository"], reference["provenance"]["revision"]
                ),
                edits: edits.clone(),
            })?;
            let active = model.activate_parameter_overlay(&admitted, parameter_budget)?;
            for (edit, before) in edits.iter().zip(original) {
                let values = model
                    .query_parameter(
                        &active.identity,
                        &edit.parameter,
                        edit.region.clone(),
                        parameter_budget,
                    )?
                    .values;
                for ((before, delta), actual) in before.iter().zip(edit.update.values()).zip(values)
                {
                    ensure!(
                        before + delta == actual,
                        "effective query differs from installed update"
                    );
                }
            }
            active_overlay = Some(active.identity);
        } else if trial == "restored" {
            model.remove_parameter_overlay(
                active_overlay.as_deref().context("active transaction")?,
            )?;
        }
        let expected_trial = if trial == "restored" {
            "baseline"
        } else {
            trial
        };
        model.reset()?;
        let operations = architecture
            .components
            .iter()
            .filter(|g| {
                matches!(trial, "keep_only" | "deletion")
                    && intervention_layers.contains(&g.layer_index)
            })
            .map(|g| -> anyhow::Result<_> {
                let dtype = match baseline_precision.get(&g.effective_activation) {
                    Some(eredu_core::checkpoint::TensorDtype::F32) => InterventionDtype::Float32,
                    Some(eredu_core::checkpoint::TensorDtype::F16) => InterventionDtype::Float16,
                    Some(eredu_core::checkpoint::TensorDtype::Bf16) => InterventionDtype::Bfloat16,
                    other => anyhow::bail!(
                        "{} has no supported measured source precision: {other:?}",
                        g.activation
                    ),
                };
                Ok(InterventionOperation {
                    id: g.id.clone(),
                    target: g.activation.clone(),
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
                    action: InterventionAction::MaskComponents {
                        dtype,
                        indices: vec![1, 3],
                        keep_selected: trial == "keep_only",
                    },
                    evidence: InterventionEvidence::None,
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        let interventions = InterventionPlan {
            schema_version: INTERVENTION_SCHEMA_VERSION,
            operations,
        };
        let mut request = PreparedChatRequest::new(&chat, settings);
        request.input = PreparedChatPrompt::TokenIds(&prefix);
        request.output_mode = PreparedChatOutputMode::Text;
        request.capture = Some(&capture);
        request.intervention = Some(&interventions);
        let mut records = vec![];
        let mut collect = |record| {
            records.push(record);
            ControlFlow::Continue(())
        };
        let mut run = model
            .start_controlled_chat(request, trace, Default::default(), &mut collect)?
            .context("cancelled before reference trial")?;
        run.run(&mut collect)?;
        let output_ids = run.token_ids().to_vec();
        drop(run);
        let expected_ids: Vec<u32> =
            serde_json::from_value(reference[expected_trial]["generated"].clone())?;
        ensure!(
            output_ids == expected_ids,
            "{trial}: generated {:?}, expected {expected_ids:?}",
            output_ids
        );
        let mut worst = 0.0f64;
        let mut compared = 0;
        let mut evidence = std::collections::BTreeMap::new();
        let mut precision = std::collections::BTreeMap::new();
        for event in &records {
            let Some(ObservedGenerationEvent::Token {
                prediction_index,
                forced,
                captures: Some(step),
                ..
            }) = event.event.progress()
            else {
                continue;
            };
            ensure!(
                !forced,
                "reference trials must not force their tested predictions"
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
                ensure!(
                    record.outcome == CaptureOutcome::Captured,
                    "missing capture: {record:?}"
                );
                if let Some(CapturePayload::TokenScores(scores)) = &record.payload {
                    let native = step
                        .records
                        .iter()
                        .find_map(|r| match &r.payload {
                            Some(CapturePayload::Tensor(t)) if r.path == "model.logits" => Some(t),
                            _ => None,
                        })
                        .context("validation logits")?;
                    let eredu_core::TensorObservationData::F32(native) = native.data() else {
                        anyhow::bail!("native score dtype")
                    };
                    let maximum = native.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let log_mass = native
                        .iter()
                        .map(|v| (*v as f64 - maximum).exp())
                        .sum::<f64>()
                        .ln();
                    let oracle = &reference[expected_trial]["prefill"];
                    let oracle_partition = oracle["target_score"].as_f64().unwrap()
                        - oracle["target_log_probability"].as_f64().unwrap();
                    for selected in &scores.scores {
                        let id = selected.target.token_id as usize;
                        ensure!(
                            selected.target.score == native[id],
                            "selected score differs from native logits"
                        );
                        ensure!(
                            selected.rank
                                == 1 + native.iter().filter(|v| **v > native[id]).count() as u64,
                            "rank differs from full native vocabulary"
                        );
                        ensure!(
                            (selected.log_probability - ((native[id] as f64 - maximum) - log_mass))
                                .abs()
                                < 2e-7,
                            "native full log probability reduction"
                        );
                        let oracle_probability =
                            oracle["logits"][id].as_f64().unwrap() - oracle_partition;
                        ensure!(
                            (selected.log_probability - oracle_probability).abs() < 5e-4,
                            "independent full log probability differs"
                        );
                        let alternative = selected
                            .strongest_alternative
                            .as_ref()
                            .context("strongest alternative")?;
                        ensure!(
                            alternative.token_id != selected.target.token_id
                                && alternative.score
                                    == native
                                        .iter()
                                        .enumerate()
                                        .filter(|(i, _)| *i != id)
                                        .map(|(_, v)| *v)
                                        .fold(f32::NEG_INFINITY, f32::max),
                            "strongest alternative differs"
                        );
                        compared += 1;
                    }
                    continue;
                }
                let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
                    anyhow::bail!("non-tensor capture")
                };
                let eredu_core::TensorObservationData::F32(values) = tensor.data() else {
                    anyhow::bail!("non-f32 capture")
                };
                if *prediction_index == 0 {
                    evidence.insert(record.path.clone(), values.clone());
                    precision.insert(
                        record.path.clone(),
                        record
                            .source_dtype
                            .clone()
                            .context("native source precision")?,
                    );
                }
                if record.selection_id.starts_with("analysis:") {
                    continue;
                }
                let expected = if record.path == "model.logits" {
                    if *prediction_index != 0 {
                        continue;
                    }
                    &reference[expected_trial]["prefill"]["logits"]
                } else {
                    &reference[expected_trial]["captures"][*prediction_index as usize][&record.path]
                };
                let expected: Vec<f64> = serde_json::from_value(expected.clone())?;
                ensure!(
                    values.len() == expected.len(),
                    "{} shape mismatch",
                    record.path
                );
                for (&actual, expected) in values.iter().zip(expected) {
                    let error = (actual as f64 - expected).abs();
                    worst = worst.max(error);
                    ensure!(
                        error <= 2e-4 + 3e-4 * expected.abs(),
                        "{trial} prediction {prediction_index} {}: {actual} != {expected} (error {error})",
                        record.path
                    );
                    compared += 1;
                }
            }
        }
        ensure!(compared > 0, "no reference values compared");
        if trial == "baseline" {
            baseline_evidence = Some(evidence.clone());
            baseline_precision = precision.clone();
        }
        if trial == "restored" {
            ensure!(
                baseline_evidence.as_ref() == Some(&evidence) && baseline_precision == precision,
                "restored native prefill evidence differs from baseline"
            );
        }
        let reconstruction = analysis::reconstruct(
            &mut model,
            &architecture,
            &evidence,
            reference[expected_trial]["prefill"]["target"]
                .as_u64()
                .unwrap() as u32,
            reference[expected_trial]["prefill"]["competitor"]
                .as_u64()
                .unwrap() as u32,
        )?;
        println!(
            "{}",
            serde_json::json!({"trial":trial,"generated":output_ids,"compared_values":compared,"max_absolute_error":worst,"absolute_tolerance":2e-4,"relative_tolerance":3e-4,"source_dtypes":precision,"analysis":reconstruction})
        );
    }
    Ok(())
}
