//! Actual public V2 pending-media snapshots; existing selected native driver only.
use super::*;
use eredu_backend_mlx::{backend::runtime::media::input, native::MlxModelInput};
use eredu_core::{InputExtent, InputMetadataKey, InputModality};
use safemlx::Array;

pub(super) fn prompt() -> MlxModelInput {
    let image = |offset: f32| {
        input::input_part(
            InputModality::Image,
            input::InputPayload::Tensor(Array::from_slice(
                &(0..192)
                    .map(|i| (i as f32 - 93.) / 193. + offset)
                    .collect::<Vec<_>>(),
                &[16, 12],
            )),
            [(
                InputMetadataKey::PatchGrid,
                Array::from_slice(&[1_i32, 4, 4], &[1, 3]),
            )],
            [InputExtent::PatchGrid {
                time: 1,
                height: 4,
                width: 4,
            }],
        )
        .unwrap()
    };
    let parts = [
        input::token_ids_part(&Array::from_slice(&[1_u32, 2], &[1, 2])).unwrap(),
        image(0.),
        image(0.125),
        input::input_part(
            InputModality::Text,
            input::InputPayload::Embeddings(Array::from_slice(
                &(0..32).map(|i| (i as f32 - 15.) / 33.).collect::<Vec<_>>(),
                &[1, 2, 16],
            )),
            [],
            [],
        )
        .unwrap(),
        input::token_ids_part(&Array::from_slice(&[3_u32], &[1, 1])).unwrap(),
    ];
    MlxModelInput::from(input::ModelInput::new(&parts))
        .with_semantic_content_fingerprint("public pending multi-image and projected-text source")
        .unwrap()
        .with_prefill_chunk_positions(2.try_into().unwrap())
}
fn ranges(records: &[PreparedControlledGenerationRecord]) -> Vec<[u64; 2]> {
    records
        .iter()
        .filter_map(|record| match &record.event {
            PreparedControlledGenerationEvent::Progress {
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
    let root = components::qwen_vl_component_fixture(false);
    let execution = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_residency(residency)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true));
    let (mut model, _) =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &root.0, &execution)
            .unwrap()
            .into_parts();
    let chat = model
        .prepare_chat(ChatTemplateRequest {
            messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
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
    let prepared = model
        .prepare_controlled_input(
            &chat,
            prompt(),
            settings,
            PreparedInputInstrumentation::Unobserved,
            trace,
        )
        .unwrap();
    assert_eq!(prepared.prompt_attribution().decoder_positions, 13);
    assert_eq!(prepared.prompt_attribution().canonical_token_ids, [1, 2, 3]);
    let attribution = prepared.prompt_attribution().clone();
    let mut records = Vec::new();
    let mut session = model
        .start_controlled_prepared_text(prepared, &[], Default::default(), |r| {
            records.push(r);
            ControlFlow::Continue(())
        })
        .unwrap();
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
    session.enable_snapshots(snapshot_limits).unwrap();
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
    // The original child request reserves every possible future semantic slot.
    // Its trace contribution alone exceeds the unchanged retained-state ceiling.
    assert!(
        trace
            .total_bytes
            .checked_mul(std::mem::size_of::<SemanticEvent>() as u64 + 1)
            .unwrap()
            > snapshot_limits.retained_bytes
    );
    let before_fork = session.snapshot_usage().unwrap();
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
        .expect("oversized child trace rejects before native copy");
    let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut retained_limit = false;
    while let Some(error) = cause {
        if let Some(ControlledGenerationError::Snapshot(
            eredu_runtime::execution_control::TextSnapshotError::Control(
                ExecutionControlError::Limit("retained bytes"),
            ),
        )) = error.downcast_ref::<ControlledGenerationError>()
        {
            retained_limit = true;
            break;
        }
        cause = error.source();
    }
    assert!(retained_limit, "typed retained-byte refusal");
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
                if let PreparedControlledGenerationEvent::BranchStarted {
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
