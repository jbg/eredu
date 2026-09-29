//! Ordinary application loading must not require family workspace arithmetic.
use eredu::api::*;
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_backend_mlx::MlxBackendFactory;
use eredu_core::{
    capture::CapturePlan, execution_control::SnapshotLimits, GenerationConfigOverrides,
    InspectionReadiness, PrefillChunkPolicy,
};
use eredu_evaluation::qwen4_exp::{metadata, Fixture, PreparedFixtures};
use eredu_gguf::MetadataValue;
use std::{num::NonZeroUsize, ops::ControlFlow};
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::whitespace::Whitespace, Tokenizer};

const PROMPT: [u32; 19] = [3, 4, 0, 5, 8, 9, 11, 12, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
const TEMPLATE: &str = "{% for m in messages %}{{ m.content }}{% endfor %}";

fn settings(prefill: PrefillChunkPolicy) -> PreparedChatGenerationSettings {
    PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            do_sample: Some(false),
            max_new_tokens: Some(4),
            ..Default::default()
        },
        prefill,
        ..Default::default()
    }
}

fn write_tokenizer(path: &std::path::Path) -> Tokenizer {
    let vocabulary = (0..16).map(|id| (format!("word{id}"), id)).collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("word0".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer.save(path.join("tokenizer.json"), false).unwrap();
    std::fs::write(path.join("chat_template.jinja"), TEMPLATE).unwrap();
    tokenizer
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx for CPU-only native initialization"
)]
fn qwen4_default_inspection_and_public_loading_generate_and_restore_both_formats() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = PreparedFixtures::write(directory.path()).unwrap();
    let tokenizer = write_tokenizer(&fixture.safetensors_path);
    let mut gguf_metadata = metadata();
    gguf_metadata.insert(
        "tokenizer.huggingface.json".into(),
        MetadataValue::String(tokenizer.to_string(false).unwrap()),
    );
    gguf_metadata.insert(
        "tokenizer.chat_template".into(),
        MetadataValue::String(TEMPLATE.into()),
    );
    gguf_metadata.insert(
        "tokenizer.ggml.eos_token_id".into(),
        MetadataValue::Uint32(0),
    );
    // Use a new path: prepared fixture sources retain their original provenance.
    let gguf_path = directory.path().join("with-tokenizer.gguf");
    Fixture::new().write_metadata(&gguf_path, &gguf_metadata);

    let mut outputs = Vec::new();
    for path in [&fixture.safetensors_path, &gguf_path] {
        let report = inspect_local_model(path, LocalInspectionOptions::default()).unwrap();
        assert_eq!(
            report.requested_load,
            InspectionReadiness::Ready,
            "{report:#?}"
        );
        // Native inspection admits construction; the facade validates text and
        // tokenizer behavior through the actual load/generation below.
        // The public plan supplies placement alone, without bounded-execution,
        // table-row, QSA, or recurrent workspace settings.
        let plan =
            eredu_core::ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
        let (mut model, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &plan)
                .unwrap()
                .into_parts();
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role": "user", "content": "word3 word4"})],
                add_generation_prompt: false,
                ..Default::default()
            })
            .unwrap();
        let ordinary = model
            .generate_prepared_text(PreparedChatGenerationRequest {
                input: PreparedChatInput::token_ids(&chat, PROMPT.to_vec()),
                settings: settings(PrefillChunkPolicy::Unchunked),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            })
            .unwrap();
        // The first independent fixture prediction is token 7. At least two
        // predictions ensure this exercises cached decode as well as prefill.
        assert_eq!(ordinary.token_ids.first(), Some(&7));
        assert!(ordinary.token_ids.len() >= 2);
        model.reset().unwrap();
        let chunked = model
            .generate_prepared_text(PreparedChatGenerationRequest {
                input: PreparedChatInput::token_ids(&chat, PROMPT.to_vec()),
                settings: settings(PrefillChunkPolicy::Bounded(NonZeroUsize::new(3).unwrap())),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            })
            .unwrap();
        assert_eq!(chunked.token_ids, ordinary.token_ids);
        assert_eq!(chunked.finish_reason, ordinary.finish_reason);
        model.reset().unwrap();
        let prepared = model
            .prepare_observed_token_ids(
                &chat,
                PROMPT.to_vec(),
                settings(PrefillChunkPolicy::Unchunked),
                CapturePlan::none(),
                TraceLimits {
                    per_record_bytes: 16384,
                    total_bytes: 65536,
                },
            )
            .unwrap();
        let mut run = model
            .start_controlled_text(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 8 << 20,
            cumulative_copy_bytes: 32 << 20,
        })
        .unwrap();
        run.step(|_| ControlFlow::Continue(())).unwrap();
        let snapshot = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
        run.run(|_| ControlFlow::Continue(())).unwrap();
        assert_eq!(run.token_ids(), ordinary.token_ids);
        assert_eq!(run.finish_reason(), Some(ordinary.finish_reason));
        run.restore(&snapshot, |_| ControlFlow::Continue(()))
            .unwrap();
        assert_eq!(run.token_ids(), snapshot.token_ids());
        run.run(|_| ControlFlow::Continue(())).unwrap();
        assert_eq!(run.token_ids(), ordinary.token_ids);
        drop(run);
        model.synchronize().unwrap();
        outputs.push(ordinary.token_ids);
    }
    assert_eq!(outputs[0], outputs[1]);
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx for CPU-only native initialization"
)]
fn qwen4_public_embedded_generation_matches_target_and_controlled_snapshot_replay() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixture.safetensors_path).unwrap();
    eredu_evaluation::qwen4_exp::set_context_length(&fixture.safetensors_path, 1024).unwrap();
    // Cross the selected 512-token invocation cap, the QSA block budget and an
    // incomplete final micro-block. Original IDs include n-gram reset tokens.
    let prompt: Vec<_> = PROMPT.into_iter().cycle().take(515).collect();
    let tokenizer = write_tokenizer(&fixture.safetensors_path);
    let mut gguf_metadata = metadata();
    gguf_metadata.insert(
        "qwen4exp.context_length".into(),
        MetadataValue::Uint32(1024),
    );
    gguf_metadata.insert(
        "tokenizer.huggingface.json".into(),
        MetadataValue::String(tokenizer.to_string(false).unwrap()),
    );
    gguf_metadata.insert(
        "tokenizer.chat_template".into(),
        MetadataValue::String(TEMPLATE.into()),
    );
    gguf_metadata.insert(
        "tokenizer.ggml.eos_token_id".into(),
        MetadataValue::Uint32(0),
    );
    let gguf_path = directory.path().join("prediction-with-tokenizer.gguf");
    Fixture::new().write_metadata(&gguf_path, &gguf_metadata);
    let plan =
        eredu_core::ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let chat_request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"word3 word4"})],
        add_generation_prompt: false,
        ..Default::default()
    };
    let (mut target, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default(),
        &fixture.safetensors_path,
        &plan,
    )
    .unwrap()
    .into_parts();
    let chat = target.prepare_chat(chat_request()).unwrap();
    let ordinary = target
        .generate_prepared_text(PreparedChatGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, prompt.clone()),
            settings: settings(PrefillChunkPolicy::Bounded(NonZeroUsize::new(17).unwrap())),
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        })
        .unwrap();
    assert!(ordinary.token_ids.len() >= 2);
    drop(target);

    let plan = plan.with_drafting(eredu_core::DraftingPlan::Embedded {
        max_draft_tokens: 1,
        lookahead: false,
        adaptive_lookahead: false,
    });
    for (path, plan) in [
        (&fixture.safetensors_path, plan.clone()),
        (
            &gguf_path,
            plan.with_prediction_source(fixture.safetensors_path.clone()),
        ),
    ] {
        let loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &plan).unwrap();
        let generation = loaded.speculative_generation_options().unwrap().unwrap();
        let (mut model, _) = loaded.into_parts();
        let chat = model.prepare_chat(chat_request()).unwrap();
        let request = || PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::token_ids(&chat, prompt.clone()),
            drafting: eredu_core::SpeculativeDraft::Embedded,
            settings: settings(PrefillChunkPolicy::Unchunked),
            options: generation,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        };
        let speculative = model.generate_prepared_text_speculative(request()).unwrap();
        assert_eq!(speculative.token_ids(), ordinary.token_ids);
        assert_eq!(speculative.finish_reason(), ordinary.finish_reason);
        assert!(speculative.stats().draft_tokens() > 0);
        model.reset().unwrap();
        let controlled = model
            .with_controlled_text_speculative(
                request(),
                ControlledSpeculativeOptions {
                    snapshots: Some(SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 1,
                        retained_bytes: 32 << 20,
                        cumulative_copy_bytes: 128 << 20,
                    }),
                    ..Default::default()
                },
                |session| {
                    assert!(session.step()?.is_some());
                    assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
                    let saved = session.snapshot()?;
                    while session.step()?.is_some() {}
                    let expected = session.token_ids().to_vec();
                    let before = session.snapshot_usage().cumulative_copy_bytes;
                    session.restore(&saved)?;
                    assert!(session.snapshot_usage().cumulative_copy_bytes > before);
                    while session.step()?.is_some() {}
                    assert_eq!(session.token_ids(), expected);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(controlled.token_ids(), ordinary.token_ids);
        assert_eq!(controlled.finish_reason(), ordinary.finish_reason);
        assert!(controlled.stats().draft_tokens() > 0);
        model.synchronize().unwrap();
    }
}

#[test]
#[cfg(feature = "image")]
#[cfg_attr(
    feature = "metal",
    ignore = "run with --no-default-features --features mlx,image for CPU-only native initialization"
)]
fn qwen4_public_image_video_embedded_generation_and_controlled_replay() {
    use eredu_core::{
        Media, MultimodalRequest, MultimodalSegment as Segment, RgbImage, Video, VideoSampling,
    };
    use eredu_evaluation::qwen4_exp::{
        add_prediction_weights, add_vision_weights, set_context_length, write_vision_projector,
    };
    let directory = tempfile::tempdir().unwrap();
    let fixture = PreparedFixtures::write(directory.path()).unwrap();
    add_prediction_weights(&fixture.safetensors_path).unwrap();
    set_context_length(&fixture.safetensors_path, 1024).unwrap();
    add_vision_weights(&fixture.safetensors_path, 2);
    write_vision_projector(&directory.path().join("mmproj.gguf"), 2);
    let names = [
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ];
    let vocabulary = (0..16)
        .map(|id| {
            (
                if id < 12 {
                    format!("word{id}")
                } else {
                    names[(id - 12) as usize].into()
                },
                id,
            )
        })
        .collect();
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("word0".into())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(Whitespace));
    tokenizer
        .add_special_tokens(names.map(|name| tokenizers::AddedToken::from(name, true)))
        .unwrap();
    tokenizer
        .save(fixture.safetensors_path.join("tokenizer.json"), false)
        .unwrap();
    std::fs::write(
        fixture.safetensors_path.join("chat_template.jinja"),
        TEMPLATE,
    )
    .unwrap();
    let mut gguf_metadata = metadata();
    gguf_metadata.insert(
        "qwen4exp.context_length".into(),
        MetadataValue::Uint32(1024),
    );
    gguf_metadata.insert(
        "tokenizer.huggingface.json".into(),
        MetadataValue::String(tokenizer.to_string(false).unwrap()),
    );
    gguf_metadata.insert(
        "tokenizer.chat_template".into(),
        MetadataValue::String(TEMPLATE.into()),
    );
    gguf_metadata.insert(
        "tokenizer.ggml.eos_token_id".into(),
        MetadataValue::Uint32(0),
    );
    let gguf_path = directory.path().join("media-prediction.gguf");
    Fixture::new().write_metadata(&gguf_path, &gguf_metadata);

    let image = |width, height, seed| {
        RgbImage::new(
            (0..width * height * 3)
                .map(|i| ((i * 7 + seed) % 251) as u8)
                .collect(),
            width,
            height,
        )
        .unwrap()
    };
    // The image span crosses the default invocation boundary. Original token IDs
    // include EOS-separated n-grams and remain attached when pixels become embeddings.
    let media = MultimodalRequest::new(vec![
        Segment::TokenIds(
            [3, 4, 0, 5, 8, 9, 11]
                .into_iter()
                .cycle()
                .take(509)
                .collect(),
        ),
        Segment::Media(Media::Image(image(8, 8, 9))),
        Segment::Media(Media::Video(
            Video::new(
                (0..4).map(|i| image(8, 4, i * 17 + 1)).collect(),
                Some(1.0),
                VideoSampling::All,
            )
            .unwrap(),
        )),
        Segment::TokenIds(vec![3, 0, 5, 8, 9]),
    ])
    .unwrap();
    let chat_request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"word3 word4"})],
        add_generation_prompt: false,
        ..Default::default()
    };
    let target_plan =
        eredu_core::ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    for path in [&fixture.safetensors_path, &gguf_path] {
        let (mut target, _) =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &target_plan)
                .unwrap()
                .into_parts();
        let chat = target.prepare_chat(chat_request()).unwrap();
        let prompt = target.prepare_multimodal_input(&media).unwrap();
        let ordinary = target
            .generate_prepared_text(PreparedChatGenerationRequest {
                input: PreparedChatInput::prepared_backend_input(&chat, prompt),
                settings: settings(PrefillChunkPolicy::Unchunked),
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event: |_| {},
            })
            .unwrap();
        assert!(ordinary.token_ids.len() >= 2);
        target.reset().unwrap();
        let prompt = target.prepare_multimodal_input(&media).unwrap();
        // The fixture's spatial merge is two. Keep explicit original IDs for
        // public trace alignment, including the processor's media delimiters.
        let original_ids = prompt.with_borrowed(|input| {
            use eredu_runtime::PreparedInputPayload;
            let mut ids = Vec::new();
            for part in input.parts {
                if let PreparedInputPayload::TokenIds(tokens) = part.payload() {
                    let values = tokens.evaluated().unwrap();
                    if let Ok(values) = values.try_as_slice::<u32>() {
                        ids.extend_from_slice(values);
                    } else {
                        ids.extend(
                            values
                                .try_as_slice::<i32>()
                                .unwrap()
                                .iter()
                                .map(|v| *v as u32),
                        );
                    }
                } else {
                    let grid = part
                        .metadata_value(eredu_core::InputMetadataKey::PatchGrid)
                        .unwrap()
                        .evaluated()
                        .unwrap();
                    let grid = grid.try_as_slice::<i32>().unwrap();
                    let positions = grid
                        .chunks_exact(3)
                        .map(|row| (row[0] * row[1] * row[2]) as usize / 4)
                        .sum();
                    let id = if part.modality() == eredu_core::InputModality::Image {
                        12
                    } else {
                        13
                    };
                    ids.extend(std::iter::repeat_n(id, positions));
                }
            }
            ids
        });
        use eredu_core::capture::*;
        let allowance = CaptureUsage {
            captures: 64,
            retained_bytes: 8 << 20,
            host_bytes: 8 << 20,
            encoded_bytes: 8 << 20,
        };
        let captures = CapturePlan {
            schema_version: CAPTURE_SCHEMA_VERSION,
            selections: vec![CaptureSelection {
                id: "across-image-boundary".into(),
                path: "model.layers.0.input".into(),
                schedule: CaptureSchedule {
                    decode: false,
                    ..Default::default()
                },
                slices: vec![CaptureSlice {
                    axis: "sequence".into(),
                    start: 510,
                    end: 514,
                    stride: 1,
                }],
                transform: CaptureTransform::Slice,
            }],
            limits: CaptureLimits {
                per_step: allowance,
                cumulative: allowance.checked_mul(8).unwrap(),
                physical_native_bytes: None,
                on_limit: CaptureLimitPolicy::Fail,
            },
        };
        let observed = target
            .prepare_observed_input(
                &chat,
                prompt,
                original_ids,
                settings(PrefillChunkPolicy::Bounded(NonZeroUsize::new(512).unwrap())),
                captures,
                TraceLimits {
                    per_record_bytes: 1 << 20,
                    total_bytes: 16 << 20,
                },
            )
            .unwrap();
        let mut run = target
            .start_controlled_text(observed, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 1,
            retained_bytes: 32 << 20,
            cumulative_copy_bytes: 128 << 20,
        })
        .unwrap();
        let initial = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
        let mut first_captures = Vec::new();
        run.step(|record| {
            if let ObservedGenerationEvent::PrefillProgress {
                captures: Some(step),
                ..
            } = record.generation.event
            {
                first_captures.push(step);
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        assert!(run.token_ids().is_empty(), "a prefix step must not sample");
        assert_eq!(run.next_prediction(), 0);
        assert_eq!(first_captures.len(), 1);
        assert_eq!(first_captures[0].prefill_span.unwrap().start, 0);
        assert_eq!(first_captures[0].prefill_span.unwrap().end, 512);
        assert_eq!(
            first_captures[0].records[0].outcome,
            CaptureOutcome::Captured
        );
        let saved = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
        let mut later = Vec::new();
        run.run(|record| {
            if let ObservedGenerationEvent::Token {
                captures: Some(step),
                ..
            } = record.generation.event
            {
                if step.prefill_span.is_some() {
                    later.push(step);
                }
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(run.token_ids(), ordinary.token_ids);
        assert_eq!(later.len(), 1);
        assert_eq!(later[0].prefill_span.unwrap().start, 512);
        assert_eq!(later[0].records[0].outcome, CaptureOutcome::Captured);
        let expected = later[0].records.clone();
        run.restore(&saved, |_| ControlFlow::Continue(())).unwrap();
        assert!(run.token_ids().is_empty());
        run.run(|record| {
            if let ObservedGenerationEvent::Token {
                captures: Some(step),
                ..
            } = record.generation.event
            {
                if step.prefill_span.is_some() {
                    assert_eq!(step.records, expected);
                }
            }
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(run.token_ids(), ordinary.token_ids);
        let mut branch = run
            .fork(
                &saved,
                GenerationBranchOptions {
                    trace_limits: TraceLimits {
                        per_record_bytes: 1 << 20,
                        total_bytes: 16 << 20,
                    },
                    capture_limits: Some(CaptureLimits {
                        per_step: allowance,
                        cumulative: allowance.checked_mul(8).unwrap(),
                        physical_native_bytes: None,
                        on_limit: CaptureLimitPolicy::Fail,
                    }),
                    sampling: None,
                    intervention: None,
                },
                |_| ControlFlow::Continue(()),
            )
            .unwrap();
        run.exchange(&mut branch, |_| ControlFlow::Continue(()))
            .unwrap();
        assert!(run.token_ids().is_empty());
        run.run(|_| ControlFlow::Continue(())).unwrap();
        assert_eq!(run.token_ids(), ordinary.token_ids);
        run.exchange(&mut branch, |_| ControlFlow::Continue(()))
            .unwrap();
        drop(branch);
        run.restore(&initial, |_| ControlFlow::Continue(()))
            .unwrap();
        run.run(|_| ControlFlow::Continue(())).unwrap();
        assert_eq!(run.token_ids(), ordinary.token_ids);
        drop(run);
        drop(target);

        let mut plan = target_plan
            .clone()
            .with_drafting(eredu_core::DraftingPlan::Embedded {
                max_draft_tokens: 1,
                lookahead: false,
                adaptive_lookahead: false,
            });
        if path == &gguf_path {
            plan = plan.with_prediction_source(fixture.safetensors_path.clone());
        }
        let loaded =
            LoadedModel::load_execution_plan(&MlxBackendFactory::default(), path, &plan).unwrap();
        let generation = loaded.speculative_generation_options().unwrap().unwrap();
        let (mut model, _) = loaded.into_parts();
        let chat = model.prepare_chat(chat_request()).unwrap();
        let prompt = model.prepare_multimodal_input(&media).unwrap();
        let request = || PreparedChatSpeculativeGenerationRequest {
            input: PreparedChatInput::prepared_backend_input(&chat, prompt.clone()),
            drafting: eredu_core::SpeculativeDraft::Embedded,
            settings: settings(PrefillChunkPolicy::Unchunked),
            options: generation,
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |_| {},
        };
        let speculative = model.generate_prepared_text_speculative(request()).unwrap();
        assert_eq!(speculative.token_ids(), ordinary.token_ids);
        assert_eq!(speculative.finish_reason(), ordinary.finish_reason);
        assert!(speculative.stats().draft_tokens() > 0);
        model.reset().unwrap();
        let controlled = model
            .with_controlled_text_speculative(
                request(),
                ControlledSpeculativeOptions {
                    snapshots: Some(SnapshotLimits {
                        max_snapshots: 1,
                        max_branches: 1,
                        retained_bytes: 32 << 20,
                        cumulative_copy_bytes: 128 << 20,
                    }),
                    ..Default::default()
                },
                |session| {
                    assert!(session.step()?.is_some());
                    let saved = session.snapshot()?;
                    while session.step()?.is_some() {}
                    let expected = session.token_ids().to_vec();
                    session.restore(&saved)?;
                    while session.step()?.is_some() {}
                    assert_eq!(session.token_ids(), expected);
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(controlled.token_ids(), ordinary.token_ids);
        assert_eq!(controlled.finish_reason(), ordinary.finish_reason);
        assert!(controlled.stats().draft_tokens() > 0);
        model.synchronize().unwrap();
    }
}
