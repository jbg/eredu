//! Actual public V2 pending-media snapshots; existing selected native driver only.
use super::*;
use eredu_core::{InputExtent, InputMetadataKey, InputModality, InputPayloadKind};
use eredu_runtime::input::host::{HostInputPart, HostTensorValues, HostTensorView};

pub(super) const CHAT_TEXT: &str = "left right <|image_pad|> <|image_pad|> word4 word5 word3";

pub(super) fn media_fixture() -> Fixture {
    let root = components::qwen_vl_component_fixture(false);
    let path = root.0.join("tokenizer.json");
    let mut json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let vocabulary = json["model"]["vocab"].as_object_mut().unwrap();
    assert_eq!(vocabulary.remove("word42"), Some(42.into()));
    vocabulary.insert("<|image_pad|>".into(), 42.into());
    std::fs::write(&path, serde_json::to_vec(&json).unwrap()).unwrap();
    let mut tokenizer = Tokenizer::from_file(&path).unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("<|image_pad|>", true).normalized(false)])
        .unwrap();
    tokenizer.save(path, false).unwrap();
    std::fs::write(
        root.0.join("chat_template.jinja"),
        "{% for message in messages %}{{ message.content }}{% endfor %}",
    )
    .unwrap();
    assert_eq!(
        tokenizer.encode(CHAT_TEXT, false).unwrap().get_ids(),
        [1, 2, 42, 42, 4, 5, 3]
    );
    root
}

pub(super) fn with_parts<R>(run: impl FnOnce(&[HostInputPart<'_>]) -> R) -> R {
    with_text_parts(false, run)
}

fn with_text_parts<R>(projected_text: bool, run: impl FnOnce(&[HostInputPart<'_>]) -> R) -> R {
    let image0 = std::array::from_fn::<_, 192, _>(|i| (i as f32 - 93.) / 193.);
    let image1 = std::array::from_fn::<_, 192, _>(|i| (i as f32 - 93.) / 193. + 0.125);
    let projected = std::array::from_fn::<_, 32, _>(|i| (i as f32 - 15.) / 33.);
    let metadata = [(
        InputMetadataKey::PatchGrid,
        HostTensorView {
            shape: &[1, 3],
            values: HostTensorValues::I32(&[1, 4, 4]),
        },
    )];
    let extents = [InputExtent::PatchGrid {
        time: 1,
        height: 4,
        width: 4,
    }];
    run(&[
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 2],
                values: HostTensorValues::U32(&[1, 2]),
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Image,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[16, 12],
                values: HostTensorValues::F32(&image0),
            },
            metadata: &metadata,
            extents: &extents,
        },
        HostInputPart {
            modality: InputModality::Image,
            kind: InputPayloadKind::Tensor,
            payload: HostTensorView {
                shape: &[16, 12],
                values: HostTensorValues::F32(&image1),
            },
            metadata: &metadata,
            extents: &extents,
        },
        HostInputPart {
            modality: InputModality::Text,
            kind: if projected_text {
                InputPayloadKind::Embeddings
            } else {
                InputPayloadKind::TokenIds
            },
            payload: if projected_text {
                HostTensorView {
                    shape: &[1, 2, 16],
                    values: HostTensorValues::F32(&projected),
                }
            } else {
                HostTensorView {
                    shape: &[1, 2],
                    values: HostTensorValues::U32(&[4, 5]),
                }
            },
            metadata: &[],
            extents: &[],
        },
        HostInputPart {
            modality: InputModality::Text,
            kind: InputPayloadKind::TokenIds,
            payload: HostTensorView {
                shape: &[1, 1],
                values: HostTensorValues::U32(&[3]),
            },
            metadata: &[],
            extents: &[],
        },
    ])
}

#[test]
fn public_media_chat_rejects_projected_text_without_rendered_token_identity() {
    let root = media_fixture();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap());
    let (model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let chat = model
        .source_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":CHAT_TEXT})],
            ..Default::default()
        })
        .unwrap();
    let error = with_text_parts(true, |parts| {
        model.prepare_chat_input(
            &chat,
            parts,
            &eredu_core::GenerationCancellationToken::new(),
        )
    })
    .err()
    .expect("unidentified projected text cannot authenticate the rendered chat");
    assert_eq!(
        error.chat_input_rejection(),
        Some(eredu_runtime::input::PreparedChatInputRejection::ProjectionUnavailable { part: 3 })
    );
}

fn ranges(records: &[ControlledGenerationRecord]) -> Vec<[u64; 2]> {
    records
        .iter()
        .filter_map(|record| match &record.event {
            ControlledGenerationEvent::Progress {
                event:
                    ObservedGenerationEvent::Token {
                        input_range,
                        captures,
                        ..
                    },
            } => {
                assert!(captures.is_none());
                Some(*input_range)
            }
            _ => None,
        })
        .collect()
}
fn run(residency: eredu_core::ResidencyPlan, snapshots: bool) -> Vec<u32> {
    let root = media_fixture();
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_residency(residency)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let chat = model
        .source_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":CHAT_TEXT})],
            tools: vec![],
            tool_choice: ToolChoice::None,
            add_generation_prompt: true,
            ..Default::default()
        })
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            temperature: Some(0.7),
            max_new_tokens: Some(4),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    let trace = TraceLimits {
        per_record_bytes: 1 << 20,
        total_bytes: 64 << 20,
    };
    let cancellation = eredu_core::GenerationCancellationToken::new();
    let input = with_parts(|parts| model.prepare_chat_input(&chat, parts, &cancellation))
        .unwrap()
        .expect("live media preparation");
    let mut settings = original_settings(settings.clone());
    settings.inference.prefill_chunk_positions = std::num::NonZeroU64::new(2);
    let mut prepared = PreparedChatRequest::new(&chat, settings.clone());
    prepared.input = PreparedChatPrompt::Media(input);
    prepared.output_mode = PreparedChatOutputMode::Text;
    let mut records = Vec::new();
    let mut session = model
        .start_controlled_chat(prepared, trace, Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap()
        .expect("live control");
    assert_eq!(session.prompt_attribution().decoder_positions, 13);
    assert_eq!(
        session.prompt_attribution().canonical_token_ids,
        [1, 2, 4, 5, 3]
    );
    let attribution = session.prompt_attribution().clone();
    assert_eq!(session.status(), GenerationStatus::Prepared);
    assert!(session.token_ids().is_empty());
    if !snapshots {
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(
            session.token_ids().len(),
            4,
            "one media prefill and three cached decodes"
        );
        assert_eq!(ranges(&records), [[0, 13], [13, 14], [14, 15], [15, 16]]);
        return session.token_ids().to_vec();
    }
    let snapshot_limits = SnapshotLimits {
        max_snapshots: 3,
        max_branches: 1,
        retained_bytes: 128 << 20,
        cumulative_copy_bytes: 1 << 30,
    };
    session
        .enable_snapshots(
            snapshot_limits,
            native_limits(ORIGINAL_CAPACITY),
            copy_limits(),
        )
        .unwrap();
    // This is the pending-media boundary, before any encoder or decoder work.
    let initial = session.snapshot(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(initial.prompt_attribution(), &attribution);
    assert_eq!(
        initial.complete_token_ids(),
        None,
        "image/projected rows are not invented IDs"
    );
    let after_initial = session.snapshot_usage().unwrap();
    assert!(after_initial.retained_bytes > 0 && after_initial.cumulative_copy_bytes > 0);
    session
        .step(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.status(), GenerationStatus::Paused);
    assert_eq!(session.token_ids().len(), 1);
    let partial = session.snapshot(|_| ControlFlow::Continue(())).unwrap();
    while session.status() == GenerationStatus::Paused {
        session
            .step(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
    }
    let expected = session.token_ids().to_vec();
    assert_eq!(expected.len(), 4);
    assert_eq!(ranges(&records), [[0, 13], [13, 14], [14, 15], [15, 16]]);
    let complete = session.snapshot(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(complete.prompt_attribution(), &attribution);
    let mut copied = session.snapshot_usage().unwrap().cumulative_copy_bytes;
    for _ in 0..2 {
        session
            .restore(&initial, |_| ControlFlow::Continue(()))
            .unwrap();
        assert_eq!(session.status(), GenerationStatus::Prepared);
        assert!(session.token_ids().is_empty());
        assert_eq!(session.prompt_attribution(), &attribution);
        assert!(session.snapshot_usage().unwrap().cumulative_copy_bytes > copied);
        copied = session.snapshot_usage().unwrap().cumulative_copy_bytes;
        let before = records.len();
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.token_ids(), expected);
        assert_eq!(
            ranges(&records[before..]),
            [[0, 13], [13, 14], [14, 15], [15, 16]]
        );
    }
    session
        .restore(&partial, |_| ControlFlow::Continue(()))
        .unwrap();
    let before = records.len();
    session
        .run(|r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(session.token_ids(), expected);
    assert_eq!(ranges(&records[before..]), [[13, 14], [14, 15], [15, 16]]);
    session
        .restore(&complete, |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(session.status(), GenerationStatus::Completed);
    session
        .run(|_| panic!("completed restore cannot predict"))
        .unwrap();
    assert_eq!(session.token_ids(), expected);
    // JSON transport limits do not allocate future semantic journal entries.
    // One full-trace child fits; a second simultaneous child exceeds the
    // admitted branch population before any native copy can start.
    let before_full_trace = session.snapshot_usage().unwrap();
    let mut full_trace_branch = session
        .fork(
            &initial,
            GenerationBranchOptions {
                trace_limits: trace,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            |_| ControlFlow::Continue(()),
        )
        .unwrap();
    let before_fork = session.snapshot_usage().unwrap();
    assert_eq!(before_fork.branches, snapshot_limits.max_branches);
    assert!(before_fork.retained_bytes > before_full_trace.retained_bytes);
    assert!(before_fork.cumulative_copy_bytes > before_full_trace.cumulative_copy_bytes);
    let before_status = session.status();
    let mut rejected_emission = false;
    let error = session
        .fork(
            &initial,
            GenerationBranchOptions {
                trace_limits: trace,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            |_| {
                rejected_emission = true;
                ControlFlow::Continue(())
            },
        )
        .err()
        .expect("second live child rejects before native copy");
    assert!(
        matches!(
            error
                .session_failure()
                .and_then(|session| session.snapshot_failure()),
            Some(
                eredu_runtime::execution_control::TextSnapshotError::Control(
                    ExecutionControlError::Limit("branch count")
                )
            )
        ),
        "typed simultaneous-branch refusal: {error:?}"
    );
    assert!(!rejected_emission);
    assert_eq!(
        session.snapshot_usage().unwrap(),
        before_fork,
        "all reservation/cumulative-copy counters stay unchanged before native copy"
    );
    assert_eq!(session.status(), before_status);
    assert_eq!(session.token_ids(), expected);
    assert_eq!(session.prompt_attribution(), &attribution);
    drop(error);
    session
        .exchange(&mut full_trace_branch, |_| ControlFlow::Continue(()))
        .unwrap();
    session.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.token_ids(), expected);
    assert_eq!(session.prompt_attribution(), &attribution);
    assert!(session.emitted_bytes() > 0 && session.emitted_bytes() <= trace.total_bytes);
    session
        .exchange(&mut full_trace_branch, |_| ControlFlow::Continue(()))
        .unwrap();
    drop(full_trace_branch);
    let before_fork = session.snapshot_usage().unwrap();
    assert_eq!(before_fork.branches, before_full_trace.branches);
    // Four token records and lifecycle/lineage metadata must fit this finite trace.
    // This changes only the child's request; parent trace/snapshot limits remain.
    let child_trace = TraceLimits {
        per_record_bytes: 64 << 10,
        total_bytes: 64 << 10,
    };
    let mut branch_attribution = None;
    let mut branch = session
        .fork(
            &initial,
            GenerationBranchOptions {
                trace_limits: child_trace,
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            |record| {
                if let ControlledGenerationEvent::BranchStarted {
                    prompt_attribution, ..
                } = &record.event
                {
                    branch_attribution = Some(prompt_attribution.attribution().clone());
                }
                ControlFlow::Continue(())
            },
        )
        .unwrap();
    assert_eq!(branch_attribution.as_ref(), Some(&attribution));
    let after_fork = session.snapshot_usage().unwrap();
    assert_eq!(after_fork.branches, before_fork.branches + 1);
    assert!(after_fork.retained_bytes > before_fork.retained_bytes);
    assert!(after_fork.cumulative_copy_bytes > before_fork.cumulative_copy_bytes);
    session
        .exchange(&mut branch, |_| ControlFlow::Continue(()))
        .unwrap();
    assert!(session.token_ids().is_empty());
    session.run(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.token_ids(), expected);
    assert_eq!(session.prompt_attribution(), &attribution);
    assert!(session.emitted_bytes() > 0 && session.emitted_bytes() <= child_trace.total_bytes);
    assert!(
        session.snapshot_usage().unwrap().cumulative_copy_bytes >= after_fork.cumulative_copy_bytes
    );
    session
        .exchange(&mut branch, |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(session.token_ids(), expected);
    let before_drop = session.snapshot_usage().unwrap();
    drop((branch, initial, partial, complete));
    let after_drop = session.snapshot_usage().unwrap();
    assert_eq!(after_drop.snapshots, 0);
    assert_eq!(after_drop.branches, 0);
    assert_eq!(
        after_drop.cumulative_copy_bytes,
        before_drop.cumulative_copy_bytes
    );
    assert!(after_drop.cumulative_copy_bytes > copied);
    expected
}

#[test]
#[cfg_attr(
    feature = "metal",
    ignore = "run CPU fixtures with --no-default-features --features mlx"
)]
fn native_v2_media_prepared_paused_complete_snapshots_restore_twice_and_fork_in_all_residencies() {
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
        assert_eq!(run(residency.clone(), false), run(residency, true));
    }
}
