//! Three-prefix public parameter-edit consumer, checked against the independent oracle.
use anyhow::{Context, ensure};
use eredu::{api::*, runtime::chat::ChatTemplateRequest};
use eredu_core::{
    ArchitectureDescriptor, GenerationConfigOverrides, capture::*, component::ComponentReadRole,
    parameters::*,
};
use std::{collections::BTreeMap, ops::ControlFlow};

pub fn run<B: ParameterBackend + eredu_runtime::working_memory::OriginalChatBackend>(
    model: &mut LoadedModel<B>,
    architecture: &ArchitectureDescriptor,
    reference: &serde_json::Value,
) -> anyhow::Result<()> {
    let layer = reference["layer"].as_u64().context("association layer")? as usize;
    let unit = reference["component"]
        .as_u64()
        .context("association unit")? as usize;
    let target = reference["target"].as_u64().context("association token")? as u32;
    let group = architecture
        .components
        .iter()
        .find(|group| {
            group.layer_index == layer
                && group
                    .reads
                    .iter()
                    .any(|read| read.role == ComponentReadRole::Gate)
        })
        .context("gated component")?;
    ensure!(unit < group.count, "component index");
    let edits: Vec<ParameterEdit> = serde_json::from_value(reference["parameter_edits"].clone())?;
    // Join the supplied reference edit to architecture declarations, never infer
    // semantic roles by parsing a checkpoint parameter's name.
    ensure!(
        edits.iter().all(|edit| edit.parameter == group.write_weight
            || group.reads.iter().any(|read| read.weight == edit.parameter)),
        "association topology mismatch"
    );
    let cases = reference["cases"].as_array().context("association cases")?;
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
        messages: vec![serde_json::json!({"role":"user","content":"association output contract"})],
        add_generation_prompt: true,
        ..Default::default()
    };
    let chat = model
        .prepare_chat(&source, &policy, CAPACITY, &cancellation)?
        .context("cancelled before chat preparation")?;
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(1),
            ..Default::default()
        },
        inference: eredu_core::TextInferencePolicy {
            managed_memory_capacity_bytes: Some(CAPACITY),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    let usage = CaptureUsage {
        captures: 32,
        retained_bytes: 256 << 20,
        host_bytes: 16 << 20,
        encoded_bytes: 16 << 20,
    };
    let mut active = None;
    let mut original = BTreeMap::new();
    for phase in ["baseline", "edited", "restored"] {
        if phase == "edited" {
            let facts = model.parameter_discovery()?;
            let admitted = model.admit_parameter_overlay(ParameterOverlayPlan {
                schema_version: PARAMETER_SCHEMA_VERSION,
                base_identity: facts.identity,
                provenance: reference["method"]
                    .as_str()
                    .context("association provenance")?
                    .into(),
                edits: edits.clone(),
            })?;
            active = Some(
                model
                    .activate_parameter_overlay(&admitted, super::analysis::parameter_limits())?
                    .identity,
            );
        } else if phase == "restored" {
            model.remove_parameter_overlay(active.as_deref().context("active association")?)?;
        }
        for (index, case) in cases.iter().enumerate() {
            model.reset()?;
            let prefix: Vec<u32> = serde_json::from_value(case["prefix_ids"].clone())?;
            let expected = &case[if phase == "restored" {
                "baseline"
            } else {
                phase
            }];
            let mut selections: Vec<_> = [
                &group.input,
                &group.effective_activation,
                &"model.logits".to_string(),
            ]
            .into_iter()
            .map(|path| CaptureSelection {
                id: path.clone(),
                path: path.clone(),
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
            })
            .collect();
            selections.push(CaptureSelection {
                id: "scores".into(),
                path: "model.logits".into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![],
                transform: CaptureTransform::TokenScores {
                    token_ids: vec![target],
                },
            });
            let capture = CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections,
                limits: CaptureLimits {
                    per_step: usage,
                    cumulative: usage,
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            };
            let mut request = PreparedChatRequest::new(&chat, settings);
            request.input = PreparedChatPrompt::TokenIds(&prefix);
            request.output_mode = PreparedChatOutputMode::Text;
            request.capture = Some(&capture);
            let trace = TraceLimits {
                per_record_bytes: 8 << 20,
                total_bytes: 16 << 20,
            };
            let mut records = vec![];
            let mut collect = |record| {
                records.push(record);
                ControlFlow::Continue(())
            };
            let mut run = model
                .start_controlled_chat(request, trace, Default::default(), &mut collect)?
                .context("cancelled before reference association")?;
            run.run(&mut collect)?;
            ensure!(
                run.token_ids() == [expected["winner"].as_u64().context("winner")? as u32],
                "association winner differs"
            );
            let winner = run.token_ids()[0];
            drop(run);
            let mut logits = None;
            let mut score = None;
            let mut activation = None;
            for record in records {
                if let Some(ObservedGenerationEvent::Token {
                    forced,
                    captures: Some(step),
                    ..
                }) = record.event.progress()
                {
                    ensure!(!forced, "association token must not be forced");
                    ensure!(
                        record.parameter_overlay_id.is_some() == (phase == "edited"),
                        "association edit provenance"
                    );
                    for capture in &step.records {
                        ensure!(
                            capture.outcome == CaptureOutcome::Captured,
                            "association missing capture"
                        );
                        match &capture.payload {
                            Some(CapturePayload::Tensor(tensor)) => {
                                let eredu_core::TensorObservationData::F32(values) = tensor.data()
                                else {
                                    anyhow::bail!("dtype")
                                };
                                let field = if capture.path == "model.logits" {
                                    "logits"
                                } else if capture.path == group.input {
                                    "input"
                                } else {
                                    "units"
                                };
                                let expected_values: Vec<f64> =
                                    serde_json::from_value(expected[field].clone())?;
                                ensure!(
                                    values.len() == expected_values.len(),
                                    "association geometry"
                                );
                                for (actual, expected) in values.iter().zip(expected_values) {
                                    ensure!(
                                        (*actual as f64 - expected).abs()
                                            <= 2e-4 + 3e-4 * expected.abs(),
                                        "association {phase} {field} differs from reference"
                                    );
                                }
                                if field == "logits" {
                                    logits = Some(values.clone());
                                }
                                if field == "units" {
                                    activation = Some(values[unit]);
                                }
                            }
                            Some(CapturePayload::TokenScores(scores)) => {
                                score = Some(scores.scores[0].clone())
                            }
                            _ => anyhow::bail!("association payload"),
                        }
                    }
                }
            }
            let logits = logits.context("association logits")?;
            if phase == "baseline" {
                original.insert(index, logits.clone());
            }
            if phase == "restored" {
                ensure!(
                    original.get(&index) == Some(&logits),
                    "association restoration differs"
                );
            }
            let score = score.context("target score")?;
            println!(
                "{}",
                serde_json::json!({"association_phase":phase,"case":case["kind"],
                "prefix":case["text"],"winner":winner,"target":target,"target_score":score.target.score,
                "target_log_probability":score.log_probability,"rank":score.rank,"unit_value":activation,
                "claim":"Illustrative read/write construction; held-out behavior is measured, not guaranteed"})
            );
        }
    }
    Ok(())
}
