//! Released-checkpoint A/B validation of shared projection integration.
//! Captures invalidate performance measurements; use projection_baseline for those.
use anyhow::{ensure, Context};
use eredu::{api::*, runtime::chat::ChatTemplateRequest};
use eredu_backend_mlx::{
    backend::nn::projection_profile::{CastGemmReference, ProjectionCapture},
    MlxBackendFactory,
};
use eredu_core::{
    capture::*, residency::ParameterConversionRetentionPolicy, DraftPlacementPlan, DraftingPlan,
    ExecutionPlan, GenerationConfigOverrides, TensorObservationData, TextGenerationConfig,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{ops::ControlFlow, path::PathBuf};

fn capture_plan(positions: usize, speculative: bool) -> CapturePlan {
    let selection = |id: &str, prefill, decode, start| CaptureSelection {
        id: id.into(),
        path: "model.logits".into(),
        schedule: CaptureSchedule {
            prefill,
            decode,
            ..Default::default()
        },
        slices: vec![CaptureSlice {
            axis: "sequence".into(),
            start,
            end: start + 1,
            stride: 1,
        }],
        transform: CaptureTransform::Slice,
    };
    CapturePlan {
        schema_version: CAPTURE_SCHEMA_VERSION,
        selections: if speculative {
            vec![selection("logits", true, true, 0)]
        } else {
            vec![
                selection("prefill", true, false, positions as u64 - 1),
                selection("decode", false, true, 0),
            ]
        },
        limits: CaptureLimits {
            per_step: CaptureUsage {
                captures: 128,
                retained_bytes: 4 << 30,
                host_bytes: 64 << 20,
                encoded_bytes: 64 << 20,
            },
            cumulative: CaptureUsage {
                captures: 512,
                retained_bytes: 64 << 30,
                host_bytes: 256 << 20,
                encoded_bytes: 256 << 20,
            },
            physical_native_bytes: None,
            on_limit: CaptureLimitPolicy::Fail,
        },
    }
}
fn logit_bits(capture: &CapturedStep) -> Vec<u32> {
    let records: Vec<_> = capture
        .records
        .iter()
        .filter(|record| record.outcome == CaptureOutcome::Captured)
        .collect();
    assert_eq!(records.len(), 1);
    let record = records[0];
    let Some(CapturePayload::Tensor(tensor)) = &record.payload else {
        panic!("missing full logits")
    };
    let TensorObservationData::F32(values) = tensor.data() else {
        panic!("non-F32 logits")
    };
    assert_eq!(values.len(), 65536, "pinned LFM2.5 vocabulary");
    assert!(values.iter().all(|v| v.is_finite()));
    values.iter().map(|x| x.to_bits()).collect()
}
fn fingerprint(bits: &[u32]) -> String {
    let mut hash = Sha256::new();
    for x in bits {
        hash.update(x.to_le_bytes());
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    ensure!(
        args.len() == 4,
        "usage: projection_parity MODEL OUTPUT_JSON BYTES|unlimited POSITIONS"
    );
    let path = PathBuf::from(&args[0]);
    let positions: usize = args[3].parse()?;
    ensure!(positions > 0, "empty prompt");
    let policy = if args[2] == "unlimited" {
        ParameterConversionRetentionPolicy::Unlimited
    } else {
        ParameterConversionRetentionPolicy::Bounded {
            max_bytes: args[2].parse()?,
        }
    };
    set_local_allocator_cache_limit(0)?;
    let plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Accelerator(0))?)
        .with_parameter_conversion_retention(Some(policy));
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.0),
            max_new_tokens: Some(8),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut baseline = None;
    let mut results = Vec::new();
    for reference in [true, false] {
        eprintln!("ordinary parity: reference={reference}");
        let _guard = reference.then(CastGemmReference::begin);
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &plan)?
                .into_parts();
        let chat = model.prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role":"user", "content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })?;
        let config = model.resolve_generation_config(settings.overrides)?;
        let raw: Vec<_> = model
            .generate_tokens(vec![1; positions], TextGenerationConfig::new(config))?
            .map(|t| t.and_then(|t| t.token_id()))
            .collect::<Result<_, _>>()?;
        ensure!(raw.len() == 8, "need eight cached predictions");
        for controlled in [false, true] {
            model.reset()?;
            let prepared = model.prepare_observed_token_ids(
                &chat,
                vec![1; positions],
                settings,
                capture_plan(positions, false),
                TraceLimits {
                    per_record_bytes: 8 << 20,
                    total_bytes: 64 << 20,
                },
            )?;
            let mut decisions = Vec::new();
            let mut collect = |record: ObservedGenerationRecord| {
                if let ObservedGenerationEvent::Token {
                    token_id,
                    prediction_index,
                    captures,
                    ..
                } = record.event
                {
                    decisions.push((token_id, prediction_index, logit_bits(&captures.unwrap())));
                }
                ControlFlow::Continue(())
            };
            if controlled {
                let mut run =
                    model.start_controlled_text(prepared, &[], Default::default(), |r| {
                        collect(r.generation)
                    })?;
                while run.finish_reason().is_none() {
                    run.step(|r| collect(r.generation))?;
                }
                ensure!(run.token_ids() == raw, "controlled tokens differ");
            } else {
                model.generate_observed_text(prepared, &[], Default::default(), collect)?;
            }
            ensure!(decisions.len() == 8, "missing captured predictions");
            ensure!(
                decisions.iter().map(|d| d.0).collect::<Vec<_>>() == raw,
                "observed tokens differ"
            );
            if let Some(baseline) = &baseline {
                ensure!(
                    &decisions == baseline,
                    "exact ordinary/controlled/reference/mixed logits differ"
                );
            } else {
                baseline = Some(decisions.clone());
            }
            results.push(
                json!({"reference": reference, "controlled": controlled, "tokens": raw,
                "logit_sha256": decisions.iter().map(|d| fingerprint(&d.2)).collect::<Vec<_>>() }),
            );
        }
    }
    let expected_tokens: Vec<_> = baseline.as_ref().unwrap().iter().map(|d| d.0).collect();
    let spec_plan = plan.with_drafting(DraftingPlan::External {
        model: path.display().to_string(),
        placement: DraftPlacementPlan::Target,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let mut spec_baseline = None;
    let mut speculative = Vec::new();
    for reference in [true, false] {
        eprintln!("speculative parity: reference={reference}");
        let _guard = reference.then(CastGemmReference::begin);
        let mut loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &path, &spec_plan)?;
        let options = loaded
            .speculative_generation_options()?
            .context("missing speculation")?;
        let (model, drafting) = loaded.parts_mut();
        let chat = model.prepare_chat(ChatTemplateRequest {
            messages: vec![json!({"role":"user", "content":"hello"})],
            add_generation_prompt: true,
            ..Default::default()
        })?;
        let output =
            model.generate_prepared_text_speculative(PreparedChatSpeculativeGenerationRequest {
                input: PreparedChatInput::token_ids(&chat, vec![1; positions]),
                drafting: drafting.as_speculative_draft().unwrap(),
                settings,
                options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            })?;
        ensure!(
            output.token_ids() == expected_tokens,
            "ordinary speculative tokens differ"
        );
        let capture = model
            .prepare_speculative_capture(settings, capture_plan(positions, true))
            .context("admit speculative capture")?;
        let mut captures = Vec::new();
        let mut verifications = 0;
        let profile = ProjectionCapture::begin(4096)?;
        let output = model
            .with_controlled_text_speculative(
                PreparedChatSpeculativeGenerationRequest {
                    input: PreparedChatInput::token_ids(&chat, vec![1; positions]),
                    drafting: drafting.as_speculative_draft().unwrap(),
                    settings,
                    options,
                    caller_stop_sequences: &[],
                    cancellation: Default::default(),
                    on_event: |_| {},
                },
                ControlledSpeculativeOptions {
                    trace_limits: TraceLimits {
                        per_record_bytes: 16 << 20,
                        total_bytes: 256 << 20,
                    },
                    capture: Some(capture),
                    ..Default::default()
                },
                |session| {
                    while let Some(step) = session.step()? {
                        if step.verification.is_some() {
                            verifications += 1;
                        }
                        for item in step.captures {
                            captures.push((item.role, item.position, logit_bits(&item.capture)));
                        }
                    }
                    Ok(())
                },
            )
            .context("execute controlled speculative capture")?;
        ensure!(
            output.token_ids() == expected_tokens,
            "controlled speculative tokens differ"
        );
        ensure!(
            verifications > 0 && !captures.is_empty(),
            "missing verification evidence"
        );
        if let Some(baseline) = &spec_baseline {
            ensure!(
                &captures == baseline,
                "exact speculative reference/mixed logits differ"
            );
        } else {
            spec_baseline = Some(captures.clone());
        }
        let classes: Vec<Value> = profile
            .finish()?
            .iter()
            .map(|c| c.describe())
            .collect::<Result<_, _>>()?;
        if !reference {
            ensure!(
                classes
                    .iter()
                    .any(|c| c["observed_path"] == "mixed_gemm" && {
                        let shape = c["input_shape"].as_array().unwrap();
                        let rows: u64 = shape[..shape.len() - 1]
                            .iter()
                            .map(|v| v.as_u64().unwrap())
                            .product();
                        rows > 1 && rows <= 3
                    }),
                "no multi-row mixed verification projections"
            );
        }
        speculative.push(json!({"reference": reference, "tokens": output.token_ids(), "verifications": verifications,
            "captures": captures.iter().map(|(role, position, bits)| json!({"role": role, "position": position, "logit_sha256": fingerprint(bits)})).collect::<Vec<_>>(),
            "classes": classes }));
    }
    std::fs::write(
        &args[1],
        serde_json::to_vec_pretty(&json!({"positions": positions, "retention": args[2],
        "ordinary_and_controlled": results, "speculative": speculative, "exact_bits_compared": true }))?,
    )?;
    Ok(())
}
