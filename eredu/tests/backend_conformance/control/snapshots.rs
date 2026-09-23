use super::*;
use eredu_core::PendingTextInput;
use eredu_runtime::{capture::CaptureSession, execution_control::TextSnapshotBackend};

// This semantic fixture has no persistent model tensors: its model continuation
// retains only the consumed-input position and pending canonical input. Native storage conformance runs separately
// against actual dense, convolution/recurrent and MoE fixtures.
pub(crate) struct Native(String, u64);
impl NativeTextStateBackend for MockBackend {
    type NativeTextState = Native;
    fn estimate_native_text_growth(
        runtime: &ModelRuntime<Self>,
        saved: &Native,
        _: u64,
    ) -> Result<Option<u64>, MockError> {
        Self::validate_native_text_state(runtime, saved)?;
        Ok(Some(0))
    }
    fn native_text_state_support(_: &ModelRuntime<Self>) -> ControlSupport {
        ControlSupport::Supported
    }
    fn estimate_native_text_state(
        runtime: &ModelRuntime<Self>,
        saved: Option<&Native>,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        runtime.session().authority.require_idle()?;
        if let Some(saved) = saved {
            Self::validate_native_text_state(runtime, saved)?;
        }
        let bytes = 64 + runtime.session().intervention_identity.len() as u64;
        Ok(Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        }))
    }
    fn capture_native_text_state(runtime: &mut ModelRuntime<Self>) -> Result<Native, MockError> {
        runtime.session().authority.require_idle()?;
        Ok(Native(
            runtime.session().intervention_identity.clone(),
            runtime.session().cache_positions,
        ))
    }
    fn copy_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        saved: &Native,
    ) -> Result<Native, MockError> {
        Self::validate_native_text_state(runtime, saved)?;
        Ok(Native(saved.0.clone(), saved.1))
    }
    fn validate_native_text_state(
        runtime: &ModelRuntime<Self>,
        saved: &Native,
    ) -> Result<(), MockError> {
        runtime.session().authority.require_idle()?;
        if saved.0 != runtime.session().intervention_identity {
            return Err(MockError::Capture("foreign snapshot".into()));
        }
        Ok(())
    }
    fn exchange_native_text_state(
        runtime: &mut ModelRuntime<Self>,
        saved: &mut Native,
    ) -> Result<(), MockError> {
        Self::validate_native_text_state(runtime, saved)?;
        std::mem::swap(&mut runtime.session_mut().cache_positions, &mut saved.1);
        Ok(())
    }
}
use observed_mock::Sampling;
impl TextSnapshotBackend for MockBackend {
    type SamplingState = Sampling;
    fn continuation_input_tokens(
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
        predictions: u64,
    ) -> Option<u64> {
        if predictions == 0 {
            return Some(0);
        }
        match input? {
            PendingTextInput::Prefill(prompt) => (prompt.len() as u64).checked_add(predictions - 1),
            PendingTextInput::Decode(_) => Some(predictions),
        }
    }
    fn estimate_sampling_growth(
        _: &ModelRuntime<Self>,
        _: &Sampling,
        _: u64,
    ) -> Result<Option<u64>, MockError> {
        Ok(Some(0))
    }
    fn sampling_state(state: &observed_mock::State) -> &Sampling {
        &state.sampling
    }
    fn install_sampling_state(state: &mut observed_mock::State, sampling: Sampling) {
        state.sampling = sampling;
    }
    fn assemble_generation_state(
        sampling: Sampling,
        capture: Option<CaptureSession>,
    ) -> observed_mock::State {
        observed_mock::State { sampling, capture }
    }
    fn sampling_prediction(sampling: &Sampling) -> u64 {
        sampling.prediction
    }
    fn estimate_sampling_state(
        _: &ModelRuntime<Self>,
        _: &Sampling,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        Ok(Some(SnapshotEstimate {
            retained_bytes: 64,
            copy_bytes: 64,
        }))
    }
    fn copy_sampling_state(
        _: &mut ModelRuntime<Self>,
        sampling: &Sampling,
    ) -> Result<Sampling, MockError> {
        Ok(sampling.clone())
    }
    fn estimate_pending_input(
        _: &ModelRuntime<Self>,
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
    ) -> Result<Option<SnapshotEstimate>, MockError> {
        let bytes = match input {
            Some(PendingTextInput::Prefill(ids)) => 24 + ids.len() as u64 * 4,
            Some(PendingTextInput::Decode(_)) => 4,
            None => 0,
        };
        Ok(Some(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        }))
    }
    fn copy_pending_input(
        _: &mut ModelRuntime<Self>,
        input: Option<PendingTextInput<&Vec<u32>, &MockToken>>,
    ) -> Result<Option<PendingTextInput<Vec<u32>, MockToken>>, MockError> {
        Ok(input.map(|input| match input {
            PendingTextInput::Prefill(ids) => PendingTextInput::Prefill(ids.clone()),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(token.clone()),
        }))
    }
    fn capture_run(state: &observed_mock::State) -> Option<&CaptureSession> {
        state.capture.as_ref()
    }
    fn capture_run_mut(state: &mut observed_mock::State) -> Option<&mut CaptureSession> {
        state.capture.as_mut()
    }
    fn estimate_child_capture(
        _: &ModelRuntime<Self>,
        shape: &[u64],
        _: &eredu_core::capture::CaptureSelection,
        _: &eredu_core::capture::ResolvedCaptureSlice,
    ) -> Result<eredu_core::capture::CaptureUsage, eredu_core::capture::CaptureError> {
        observed_mock::cost(shape)
    }
    fn child_intervention_estimator(
        _: &ModelRuntime<Self>,
    ) -> Result<
        std::sync::Arc<dyn eredu_core::intervention::InterventionEstimator>,
        eredu_core::capture::CaptureError,
    > {
        Ok(std::sync::Arc::new(observed_mock::Estimates))
    }
}

fn snapshot_setup() -> (
    LoadedModel<MockBackend>,
    PreparedChat,
    PreparedChatGenerationSettings,
    u32,
) {
    let request = || ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
        tools: vec![serde_json::json!({"type":"function", "function":{
            "name":"weather", "description":"Weather", "parameters":{"type":"object", "properties":{}}
        }})],
        tool_choice: eredu::runtime::chat::ToolChoice::None,
        add_generation_prompt: true,
        ..Default::default()
    };
    let mut probe = unicode_model_with_vocabulary(None, 512);
    let chat = probe.prepare_chat(request()).unwrap();
    let first = probe.encode(chat.rendered_prompt(), false).unwrap().len() as u32;
    assert!(first + 3 < 512);
    let mut model = unicode_model_with_vocabulary(Some(first), 512);
    let chat = model.prepare_chat(request()).unwrap();
    (
        model,
        chat,
        PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(8),
                ..Default::default()
            },
            seed: 17,
            ..Default::default()
        },
        first,
    )
}
fn snapshot_limits() -> SnapshotLimits {
    SnapshotLimits {
        max_snapshots: 3,
        max_branches: 0,
        retained_bytes: 4_000_000,
        cumulative_copy_bytes: 32_000_000,
    }
}

#[test]
fn facade_complete_snapshots_restore_partial_unicode_terminal_state_and_cumulative_budgets() {
    for mode in 0..4 {
        let (mut model, chat, settings, first) = snapshot_setup();
        let mut plan = if mode & 1 == 0 {
            CapturePlan::none()
        } else {
            observed_mock::plan()
        };
        plan.limits = observed_mock::plan().limits;
        let prepared = if mode & 2 == 0 {
            model
                .prepare_observed_chat(&chat, settings, plan, limits())
                .unwrap()
        } else {
            model
                .prepare_intervened_chat(
                    &chat,
                    settings,
                    plan,
                    observed_mock::intervention_plan(1.0),
                    limits(),
                )
                .unwrap()
        };
        let mut records = vec![];
        let mut session = model
            .start_controlled_chat(prepared, &[], Default::default(), |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        session.enable_snapshots(snapshot_limits()).unwrap();
        assert_eq!(session.capabilities().snapshot, ControlSupport::Supported);
        let initial = session
            .snapshot(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        session
            .step(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        let saved = session
            .snapshot(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(saved.token_ids(), [first]);
        assert_eq!(saved.metadata().output.next_prediction, 1);
        let from = records.len();
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        let baseline = semantic(&records[from..]);
        assert_eq!(session.status(), GenerationStatus::Completed);
        let history = session.token_ids().to_vec();
        let bytes = session.emitted_bytes();
        let usage = session.snapshot_usage().unwrap();
        session
            .restore(&saved, |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.output_checkpoint().epoch, 1);
        assert_eq!(session.next_prediction(), 1);
        assert!(session.emitted_bytes() > bytes);
        assert!(
            session.snapshot_usage().unwrap().cumulative_copy_bytes > usage.cumulative_copy_bytes
        );
        let from = records.len();
        session
            .run(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.token_ids(), history);
        assert_eq!(semantic(&records[from..]), baseline);
        let terminal = session
            .snapshot(|r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert!(session
            .snapshot(|_| panic!("limited snapshot emitted"))
            .is_err());
        assert!(session.enable_snapshots(snapshot_limits()).is_err());
        session
            .restore(&initial, |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.status(), GenerationStatus::Prepared);
        assert!(session.token_ids().is_empty());
        session
            .restore(&terminal, |r| {
                records.push(r);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(session.status(), GenerationStatus::Completed);
        assert_eq!(session.token_ids(), history);
        assert!(session
            .step(|_| panic!("terminal snapshot revived"))
            .is_err());
        for (index, record) in records.iter().enumerate() {
            assert_eq!(record.sequence, index as u64);
        }
        assert!(records.iter().any(|record| matches!(&record.generation.event,
            ObservedGenerationEvent::Restored { output, .. } if output == &saved.metadata().output)));
        drop((initial, saved, terminal));
        assert_eq!(session.snapshot_usage().unwrap().retained_bytes, 0);
    }
}

#[test]
fn opaque_grammar_estimates_fail_before_snapshot_retention() {
    let (mut model, chat, settings, _) = setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut session = model
        .start_controlled_chat(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    assert!(session.enable_snapshots(snapshot_limits()).is_err());
    assert!(session.snapshot_usage().is_none());
    assert!(matches!(
        session.capabilities().snapshot,
        ControlSupport::Unsupported { .. }
    ));
    session.step(|_| ControlFlow::Continue(())).unwrap();
    assert_eq!(session.next_prediction(), 1);
}

fn collect(
    records: &mut Vec<ControlledGenerationRecord>,
) -> impl FnMut(ControlledGenerationRecord) -> ControlFlow<()> + '_ {
    |record| {
        records.push(record);
        ControlFlow::Continue(())
    }
}

#[test]
fn facade_branches_keep_partial_text_identity_siblings_and_budgets_isolated() {
    use eredu::api::{GenerationBranchOptions, SamplingOverride};
    for mode in 0..4 {
        let (mut model, chat, settings, first) = snapshot_setup();
        let mut plan = if mode & 1 == 0 {
            CapturePlan::none()
        } else {
            observed_mock::plan()
        };
        plan.limits = observed_mock::plan().limits;
        let prepared = if mode & 2 == 0 {
            model
                .prepare_observed_chat(&chat, settings, plan, limits())
                .unwrap()
        } else {
            model
                .prepare_intervened_chat(
                    &chat,
                    settings,
                    plan,
                    observed_mock::intervention_plan(1.0),
                    limits(),
                )
                .unwrap()
        };
        let options = || GenerationBranchOptions {
            trace_limits: TraceLimits {
                per_record_bytes: 16384,
                total_bytes: 65536,
            },
            capture_limits: Some(observed_mock::plan().limits),
            sampling: None,
            intervention: None,
        };
        let mut records = vec![];
        let mut run = model
            .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
            .unwrap();
        let parent = run.output_checkpoint().run_id;
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 3,
            max_branches: 2,
            retained_bytes: 64_000_000,
            cumulative_copy_bytes: 256_000_000,
        })
        .unwrap();
        let initial = run.snapshot(collect(&mut records)).unwrap();
        assert_eq!(run.capabilities().fork, ControlSupport::Supported);
        run.step(collect(&mut records)).unwrap();
        let partial = run.snapshot(collect(&mut records)).unwrap();
        let mut left = run
            .fork(&partial, options(), collect(&mut records))
            .unwrap();
        let mut right = run
            .fork(&partial, options(), collect(&mut records))
            .unwrap();
        let left_id = left.run_id().to_owned();
        let right_id = right.run_id().to_owned();
        assert_ne!(left_id, right_id);
        assert_ne!(parent, left_id);
        assert_eq!(left.token_ids(), [first]);
        assert_eq!(run.token_ids(), [first]);
        assert_eq!(run.snapshot_usage().unwrap().branches, 2);
        assert!(run
            .fork(&partial, options(), collect(&mut records))
            .is_err());
        let before = records.len();
        run.run(collect(&mut records)).unwrap();
        let baseline_text = semantic(&records[before..]);
        let baseline_ids = run.token_ids().to_vec();
        let parent_bytes = run.emitted_bytes();
        run.exchange(&mut left, collect(&mut records)).unwrap();
        assert_eq!(left.run_id(), parent);
        assert_eq!(run.output_checkpoint().run_id, left_id);
        run.step(collect(&mut records)).unwrap();
        // Swap to the sibling while both the parent and left continuation remain stable.
        run.exchange(&mut right, collect(&mut records)).unwrap();
        assert_eq!(run.output_checkpoint().run_id, right_id);
        assert_eq!(run.token_ids(), [first]);
        let before = records.len();
        run.run(collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), baseline_ids);
        assert_eq!(semantic(&records[before..]), baseline_text);
        assert!(run.restore(&partial, collect(&mut records)).is_err());
        run.exchange(&mut left, collect(&mut records)).unwrap();
        assert_eq!(run.output_checkpoint().run_id, parent);
        assert_eq!(run.token_ids(), baseline_ids);
        assert!(run.emitted_bytes() > parent_bytes);
        assert_eq!(partial.token_ids(), [first]);
        drop(left);
        drop(right);
        assert_eq!(run.snapshot_usage().unwrap().branches, 0);
        // A modified child starts before the replaced decision, keeping the
        // parent's exact continuation intact. Reseeding is explicit provenance.
        let mut changed = options();
        changed.sampling = Some(SamplingOverride {
            temperature: Some(0.5),
            reseed: Some(711),
        });
        if mode & 2 != 0 {
            changed.intervention = Some(observed_mock::intervention_plan(2.0));
        }
        let mut slot = run.fork(&initial, changed, collect(&mut records)).unwrap();
        let changed_id = slot.run_id().to_owned();
        run.exchange(&mut slot, collect(&mut records)).unwrap();
        assert_eq!(run.sampling_state().unwrap().temperature, 0.5);
        run.force_next_token(first + 3).unwrap();
        run.step(collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), [first + 3]);
        assert!(records.iter().any(|r| r.generation.run_id == changed_id
            && matches!(
                r.generation.event,
                ObservedGenerationEvent::Token { forced: true, .. }
            )));
        run.cancel(collect(&mut records)).unwrap();
        // Cancelling a child does not cancel or strand the parent in its slot.
        run.exchange(&mut slot, collect(&mut records)).unwrap();
        assert_eq!(run.output_checkpoint().run_id, parent);
        assert_eq!(run.token_ids(), baseline_ids);
        drop(slot);
        assert_eq!(run.snapshot_usage().unwrap().branches, 0);
        let mut sequences = std::collections::HashMap::new();
        for record in &records {
            let previous = sequences.insert(&record.generation.run_id, record.sequence);
            assert!(previous.is_none_or(|previous| record.sequence > previous));
            if let ObservedGenerationEvent::BranchStarted {
                lineage,
                prompt_token_ids,
                inherited_token_ids,
                inherited_semantics,
            } = &record.generation.event
            {
                assert_eq!(record.sequence, 0);
                assert_eq!(record.generation.run_id, lineage.run_id);
                assert_eq!(lineage.parent.output.run_id, parent);
                assert_eq!(prompt_token_ids, initial.prompt_token_ids());
                assert!(inherited_token_ids.is_empty() || inherited_token_ids == &[first]);
                assert!(inherited_semantics.is_empty()); // Incomplete UTF-8 was never flushed.
                if let Some(parent_plan) = &lineage.parent.intervention_plan_id {
                    assert_ne!(
                        record.generation.intervention_plan_id.as_ref(),
                        Some(parent_plan)
                    );
                }
            }
        }
    }
}

#[test]
fn facade_snapshot_preserves_pending_choice_and_child_stream_includes_semantic_prefix() {
    use eredu::api::GenerationBranchOptions;
    let (mut model, chat, settings, first) = snapshot_setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut records = vec![];
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 3,
        max_branches: 1,
        retained_bytes: 32_000_000,
        cumulative_copy_bytes: 128_000_000,
    })
    .unwrap();
    run.force_next_token(first).unwrap();
    let forced = run.snapshot(collect(&mut records)).unwrap();
    assert_eq!(forced.metadata().pending_forced_token, Some(first));
    run.step(collect(&mut records)).unwrap();
    assert_eq!(run.pending_forced_token(), None);
    run.restore(&forced, collect(&mut records)).unwrap();
    assert_eq!(run.pending_forced_token(), Some(first));
    run.step(collect(&mut records)).unwrap();
    run.step(collect(&mut records)).unwrap();
    let text = run.snapshot(collect(&mut records)).unwrap();
    let mut branch = run
        .fork(
            &text,
            GenerationBranchOptions {
                trace_limits: TraceLimits {
                    per_record_bytes: 16384,
                    total_bytes: 32768,
                },
                capture_limits: None,
                sampling: None,
                intervention: None,
            },
            collect(&mut records),
        )
        .unwrap();
    let branch_id = branch.run_id().to_owned();
    let inherited = records
        .iter()
        .find_map(|record| {
            if record.generation.run_id != branch_id {
                return None;
            }
            match &record.generation.event {
                ObservedGenerationEvent::BranchStarted {
                    inherited_semantics,
                    ..
                } => Some(inherited_semantics.clone()),
                _ => None,
            }
        })
        .unwrap();
    assert_eq!(
        inherited,
        vec![eredu_core::generation::SemanticEvent::TextDelta("é".into())]
    );
    let before = records.len();
    run.run(collect(&mut records)).unwrap();
    let baseline = semantic(&records[before..]);
    run.exchange(&mut branch, collect(&mut records)).unwrap();
    let before = records.len();
    run.run(collect(&mut records)).unwrap();
    assert_eq!(semantic(&records[before..]), baseline);
    assert_eq!(text.token_ids(), [first, first + 1]);
}

#[test]
fn facade_fork_rejects_snapshot_from_an_earlier_driver_on_the_same_loaded_model() {
    use eredu::api::GenerationBranchOptions;
    let (mut model, chat, settings, _) = snapshot_setup();
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let snapshot = {
        let mut run = model
            .start_controlled_chat(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        run.enable_snapshots(snapshot_limits()).unwrap();
        run.snapshot(|_| ControlFlow::Continue(())).unwrap()
    };
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_branches: 1,
        ..snapshot_limits()
    })
    .unwrap();
    let usage = run.snapshot_usage().unwrap();
    let result = run.fork(
        &snapshot,
        GenerationBranchOptions {
            trace_limits: TraceLimits {
                per_record_bytes: 4096,
                total_bytes: 8192,
            },
            capture_limits: None,
            sampling: None,
            intervention: None,
        },
        |_| ControlFlow::Continue(()),
    );
    assert!(matches!(
        result,
        Err(eredu::api::ControlledGenerationError::Snapshot(
            eredu_runtime::execution_control::TextSnapshotError::IncompatibleRun
        ))
    ));
    assert_eq!(run.snapshot_usage().unwrap(), usage);
    assert_eq!(run.status(), GenerationStatus::Prepared);
    run.step(|_| ControlFlow::Continue(())).unwrap();
}

#[test]
fn facade_restore_preserves_stop_lookbehind_across_a_saved_token_boundary() {
    let (mut model, chat, settings, first) = snapshot_setup();
    let alternate = first + 3;
    let stop = format!("é{}", model.decode(&[alternate], false).unwrap());
    let prepared = model
        .prepare_observed_chat(&chat, settings, CapturePlan::none(), limits())
        .unwrap();
    let mut records = vec![];
    let mut run = model
        .start_controlled_chat(prepared, &[stop], Default::default(), collect(&mut records))
        .unwrap();
    run.enable_snapshots(snapshot_limits()).unwrap();
    run.step(collect(&mut records)).unwrap();
    run.step(collect(&mut records)).unwrap();
    assert!(
        semantic(&records).is_empty(),
        "stop-prefix text should still be buffered"
    );
    let saved = run.snapshot(collect(&mut records)).unwrap();
    let mut expected = None;
    for _ in 0..2 {
        run.force_next_token(alternate).unwrap();
        let before = records.len();
        run.step(collect(&mut records)).unwrap();
        assert_eq!(run.finish_reason(), Some(FinishReason::StopSequence));
        let current = semantic(&records[before..]);
        if let Some(expected) = &expected {
            assert_eq!(&current, expected);
        } else {
            expected = Some(current);
        }
        run.restore(&saved, collect(&mut records)).unwrap();
        assert_eq!(run.token_ids(), [first, first + 1]);
    }
}

#[test]
fn explicit_intervention_removal_retains_its_empty_override_provenance() {
    use eredu::api::GenerationBranchOptions;
    let (mut model, chat, settings, _) = snapshot_setup();
    let prepared = model
        .prepare_intervened_chat(
            &chat,
            settings,
            observed_mock::plan(),
            observed_mock::intervention_plan(1.0),
            limits(),
        )
        .unwrap();
    let mut records = vec![];
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), collect(&mut records))
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 32_000_000,
        cumulative_copy_bytes: 128_000_000,
    })
    .unwrap();
    let saved = run.snapshot(collect(&mut records)).unwrap();
    let empty = eredu_core::intervention::InterventionPlan {
        schema_version: eredu_core::intervention::INTERVENTION_SCHEMA_VERSION,
        operations: vec![],
    };
    let mut branch = run
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: TraceLimits {
                    per_record_bytes: 16384,
                    total_bytes: 65536,
                },
                capture_limits: Some(observed_mock::plan().limits),
                sampling: None,
                intervention: Some(empty.clone()),
            },
            collect(&mut records),
        )
        .unwrap();
    let start = records
        .iter()
        .find(|record| {
            matches!(
                record.generation.event,
                ObservedGenerationEvent::BranchStarted { .. }
            )
        })
        .unwrap();
    let ObservedGenerationEvent::BranchStarted { lineage, .. } = &start.generation.event else {
        unreachable!()
    };
    assert_eq!(lineage.intervention_override.as_ref(), Some(&empty));
    assert!(lineage.parent.intervention_plan_id.is_some());
    assert!(start.generation.intervention_plan_id.is_none());
    run.exchange(&mut branch, collect(&mut records)).unwrap();
    run.step(collect(&mut records)).unwrap();
}

#[test]
fn continuation_forecasts_follow_restore_and_branch_without_charging_budgets() {
    use eredu::api::{GenerationBranchOptions, MemoryFit};
    let (mut model, chat, settings, first) = snapshot_setup();
    let trace = eredu::api::TraceLimits {
        per_record_bytes: 16384,
        total_bytes: 65536,
    };
    let prepared = model
        .prepare_observed_chat(&chat, settings, observed_mock::plan(), trace)
        .unwrap();
    let mut run = model
        .start_controlled_chat(prepared, &[], Default::default(), |_| {
            ControlFlow::Continue(())
        })
        .unwrap();
    run.enable_snapshots(SnapshotLimits {
        max_branches: 1,
        retained_bytes: 128 << 20,
        cumulative_copy_bytes: 512 << 20,
        ..snapshot_limits()
    })
    .unwrap();
    assert!(run
        .forecast_remaining_generation(3, &Default::default())
        .is_err());
    run.step(|_| ControlFlow::Continue(())).unwrap();
    let before_snapshot_emitted = run.emitted_bytes();
    let before_snapshot = run
        .forecast_remaining_generation(3, &Default::default())
        .unwrap();
    assert_eq!(
        before_snapshot.continuation.current_positions,
        u64::from(first)
    );
    let saved = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
    let checkpoint = run.output_checkpoint();
    let usage = run.snapshot_usage().unwrap();
    let transport = run.emitted_bytes();
    let forecast = run
        .forecast_remaining_generation(3, &Default::default())
        .unwrap();
    assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(forecast.estimate.requested_positions, u64::from(first) + 3);
    assert_eq!(run.output_checkpoint(), checkpoint);
    assert_eq!(run.snapshot_usage().unwrap(), usage);
    assert_eq!(run.emitted_bytes(), transport);
    // Snapshot metadata consumes trace without retaining another semantic event.
    // Only unspent trace remains available for future history.
    assert_eq!(
        forecast.request.domains[0]
            .retained_input
            .upper_bytes
            .unwrap()
            + transport,
        before_snapshot.request.domains[0]
            .retained_input
            .upper_bytes
            .unwrap()
            + before_snapshot_emitted
            + saved.metadata().retained_bytes,
    );
    // The forecast allows hypothetical horizons beyond the run's configured limit.
    let long = run
        .forecast_remaining_generation(100, &Default::default())
        .unwrap();
    assert_eq!(long.continuation.additional_input_tokens, 100);
    run.step(|_| ControlFlow::Continue(())).unwrap();
    let advanced = run
        .forecast_remaining_generation(2, &Default::default())
        .unwrap();
    assert_eq!(
        advanced.continuation.current_positions,
        u64::from(first) + 1
    );
    run.restore(&saved, |_| ControlFlow::Continue(())).unwrap();
    let restored = run
        .forecast_remaining_generation(3, &Default::default())
        .unwrap();
    assert_eq!(restored.continuation, forecast.continuation);
    assert_eq!(
        restored.request.domains[0]
            .retained_input
            .upper_bytes
            .unwrap()
            + run.emitted_bytes(),
        forecast.request.domains[0]
            .retained_input
            .upper_bytes
            .unwrap()
            + transport
    );

    let mut branch = run
        .fork(
            &saved,
            GenerationBranchOptions {
                trace_limits: trace,
                capture_limits: Some(observed_mock::plan().limits),
                sampling: None,
                intervention: None,
            },
            |_| ControlFlow::Continue(()),
        )
        .unwrap();
    run.exchange(&mut branch, |_| ControlFlow::Continue(()))
        .unwrap();
    let usage = run.snapshot_usage().unwrap();
    let child = run
        .forecast_remaining_generation(3, &Default::default())
        .unwrap();
    assert_eq!(child.estimate.fit, MemoryFit::LikelyFit);
    assert_eq!(child.continuation, forecast.continuation);
    assert_eq!(run.snapshot_usage().unwrap(), usage);
    let wire = serde_json::to_value(child).unwrap();
    assert_eq!(
        wire["estimate"]["domains"][0]["phases"][0]["phase"],
        "continuation_start"
    );
}

#[test]
fn large_trace_budget_bounds_semantic_history_and_branch_reservations_without_header_multiplier() {
    use eredu::api::{GenerationBranchOptions, GenerationForecastOptions, MemoryBudget, MemoryFit};
    let mut retained_with_consumed_trace = Vec::new();
    for trace_bytes in [32 << 20, 64 << 20] {
        let (mut model, chat, settings, _) = snapshot_setup();
        let trace = TraceLimits {
            per_record_bytes: 16384,
            total_bytes: trace_bytes,
        };
        let prepared = model
            .prepare_observed_chat(&chat, settings, observed_mock::plan(), trace)
            .unwrap();
        let mut run = model
            .start_controlled_chat(prepared, &[], Default::default(), |_| {
                ControlFlow::Continue(())
            })
            .unwrap();
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 128 << 20,
            cumulative_copy_bytes: 512 << 20,
        })
        .unwrap();
        run.step(|_| ControlFlow::Continue(())).unwrap();
        let options = GenerationForecastOptions {
            budget: MemoryBudget {
                application_limit_bytes: Some(256 << 20),
                ..Default::default()
            },
            ..Default::default()
        };
        let checkpoint = run.output_checkpoint();
        let transport = run.emitted_bytes();
        let usage = run.snapshot_usage();
        let forecast = run.forecast_remaining_generation(3, &options).unwrap();
        assert_eq!(forecast.estimate.fit, MemoryFit::LikelyFit);
        assert_eq!(run.output_checkpoint(), checkpoint);
        assert_eq!(run.emitted_bytes(), transport);
        assert_eq!(run.snapshot_usage(), usage);
        let retained = forecast.request.domains[0]
            .retained_input
            .upper_bytes
            .unwrap();
        // Two independent copies: encoded trace and logical semantic history.
        // Admitted native/host captures, sampler and decoder add less than 4 MiB here.
        assert!(
            retained < 2 * trace_bytes + (4 << 20),
            "retained upper: {retained}"
        );
        retained_with_consumed_trace.push(retained + transport);
        let saved = run.snapshot(|_| ControlFlow::Continue(())).unwrap();
        let before_fork = run.snapshot_usage().unwrap().retained_bytes;
        let mut branch = run
            .fork(
                &saved,
                GenerationBranchOptions {
                    trace_limits: trace,
                    capture_limits: Some(observed_mock::plan().limits),
                    sampling: None,
                    intervention: None,
                },
                |_| ControlFlow::Continue(()),
            )
            .unwrap();
        let branch_retained = run.snapshot_usage().unwrap().retained_bytes - before_fork;
        // The branch's future semantic history is reserved once, alongside its
        // existing source state, native growth and parser/sampler allowances.
        assert!(branch_retained >= trace_bytes);
        assert!(
            branch_retained < trace_bytes + (4 << 20),
            "branch reservation: {branch_retained}"
        );
        run.exchange(&mut branch, |_| ControlFlow::Continue(()))
            .unwrap();
        let before = run.snapshot_usage();
        let child = run
            .forecast_remaining_generation(3, &Default::default())
            .unwrap();
        assert_eq!(child.estimate.fit, MemoryFit::LikelyFit);
        assert_eq!(run.snapshot_usage(), before);
        run.step(|_| ControlFlow::Continue(())).unwrap();
        run.exchange(&mut branch, |_| ControlFlow::Continue(()))
            .unwrap();
        assert_eq!(run.token_ids(), saved.token_ids());
        drop(branch);
        assert_eq!(run.snapshot_usage().unwrap().retained_bytes, before_fork);
    }
    // Doubling the budget adds one budget's worth to each of the two copies,
    // not one SemanticEvent header per trace byte. Identity lengths may vary.
    let delta = retained_with_consumed_trace[1] - retained_with_consumed_trace[0];
    assert!(
        delta.abs_diff(2 * (32 << 20)) < 1024,
        "retained delta: {delta}"
    );
}
