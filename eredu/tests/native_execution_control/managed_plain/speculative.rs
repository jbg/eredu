//! Public independent speculation shares one source-funded ordinary/controlled driver.
use super::*;
use eredu_core::{DraftPlacementPlan, DraftingPlan};

const SPEC_CASE: &str = "managed_plain::speculative::native_original_independent_speculation_matches_plain_and_controlled";
const SPEC_MODE: &str = "EREDU_PUBLIC_MANAGED_SPECULATIVE_MODE";
const SPEC_RESULT: &str = "PUBLIC_MANAGED_SPECULATIVE_RESULT:";

fn run(mode: &str) -> serde_json::Value {
    if mode == "ordinary" {
        return run_mode("ordinary", fixture(false), 0.0);
    }
    run_artifacts(mode, fixture(false), fixture(false))
}

fn run_artifacts(mode: &str, target: Fixture, draft: Fixture) -> serde_json::Value {
    run_artifacts_at(mode, target, draft, DraftPlacementPlan::Target, 0.0)
}
fn run_artifacts_at(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    placement: DraftPlacementPlan,
    temperature: f32,
) -> serde_json::Value {
    run_artifacts_configured(mode, target, draft, placement, settings(temperature))
}
fn run_artifacts_configured(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    placement: DraftPlacementPlan,
    generation: PreparedChatGenerationSettings,
) -> serde_json::Value {
    run_artifacts_on(
        mode,
        target,
        draft,
        eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
        placement,
        generation,
    )
}
fn run_artifacts_on(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    target_device: eredu_core::DevicePlan,
    placement: DraftPlacementPlan,
    generation: PreparedChatGenerationSettings,
) -> serde_json::Value {
    run_artifacts_on_with_rounds(mode, target, draft, target_device, placement, generation).0
}
// Share the exact driver while exposing completed rounds to cached-dtype coverage.
fn run_artifacts_on_with_rounds(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    target_device: eredu_core::DevicePlan,
    placement: DraftPlacementPlan,
    generation: PreparedChatGenerationSettings,
) -> (serde_json::Value, usize) {
    run_artifacts_on_inspected(
        mode,
        target,
        draft,
        target_device,
        placement,
        generation,
        |plan| plan,
        None,
    )
}
fn run_artifacts_on_inspected(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    target_device: eredu_core::DevicePlan,
    placement: DraftPlacementPlan,
    generation: PreparedChatGenerationSettings,
    configure: impl FnOnce(ExecutionPlan) -> ExecutionPlan,
    inspect: Option<fn(&LoadedModel<eredu_backend_mlx::backend::MlxBackend<'_>>)>,
) -> (serde_json::Value, usize) {
    let target = managed_fixture(target);
    let draft = managed_fixture(draft);
    let execution = ExecutionPlan::fully_resident(target_device)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
        .with_drafting(DraftingPlan::External {
            model: draft.0.display().to_string(),
            placement,
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        });
    let execution = configure(execution);
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let mut visible = String::new();
    let mut observed = Vec::new();
    let on_event = |event| {
        if let SemanticEvent::TextDelta(text) = event {
            visible.push_str(text.as_str());
        }
    };
    if mode == "ordinary" {
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user", "content":PROMPT})],
                    add_generation_prompt: false,
                    tool_choice: ToolChoice::None,
                    ..Default::default()
                },
                8 * 1024 * 1024 * 1024,
            )
            .unwrap();
        let ids = model.encode(PROMPT, false).unwrap();
        assert_eq!(ids, [0, 1, 2, 3, 4]);
        let mut ordinary = generation;
        ordinary.inference.memory_limits = eredu_core::MemoryLimitDeclarations::unlimited();
        let output = model
            .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&ids),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: drafting.as_speculative_draft().unwrap(),
                settings: chat_settings(&chat, ordinary),
                options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event,
            })
            .unwrap_or_else(report_failure);
        assert_eq!(output.token_ids().len(), 4);
        assert!(output.stats().rounds() >= 1);
        let rounds = output.stats().rounds();
        // Ordinary and admitted paths both retain escaped output after the
        // loaded target, assistant and their fixture sources leave the driver.
        let address = output.token_ids().as_ptr();
        if let Some(inspect) = inspect {
            inspect(model);
        }
        drop(loaded);
        assert_eq!(output.token_ids().as_ptr(), address);
        return (
            serde_json::json!({"ids": output.token_ids(), "text": visible}),
            rounds,
        );
    }
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let request = ManagedPlainTextSpeculativeRequest {
        text: ManagedPlainTextRequest::new(PROMPT, generation),
        drafting: drafting.as_speculative_draft().unwrap(),
        options,
        cancellation: GenerationCancellationToken::new(),
        on_event,
    };
    let output = if mode == "controlled" {
        model.with_controlled_managed_plain_text_speculative(
            &source,
            request,
            ControlledSpeculativeOptions::default(),
            |session| {
                assert!(session.token_ids().is_empty());
                while let Some(step) = session.step()? {
                    observed.extend(step.committed_token_ids);
                }
                assert_eq!(observed, session.token_ids());
                Ok(())
            },
        )
    } else {
        assert_eq!(mode, "managed");
        model.generate_managed_plain_text_speculative(&source, request)
    }
    .unwrap_or_else(report_failure);
    assert_eq!(output.token_ids().len(), 4);
    assert!(output.stats().rounds() >= 1);
    assert!(output.timing().time_to_first_token().is_some());
    if mode == "controlled" {
        assert_eq!(observed, output.token_ids());
    }
    // Escaped canonical output remains valid after both source and native models retire.
    let address = output.token_ids().as_ptr();
    if let Some(inspect) = inspect {
        inspect(model);
    }
    drop((source, loaded));
    assert_eq!(output.token_ids().as_ptr(), address);
    (
        serde_json::json!({"ids": output.token_ids(), "text": visible}),
        output.stats().rounds(),
    )
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_independent_speculation_matches_plain_and_controlled() {
    if let Ok(mode) = std::env::var(SPEC_MODE) {
        println!("\n{SPEC_RESULT}{}", run(&mode));
        return;
    }
    let mut expected = None;
    for mode in ["ordinary", "managed", "controlled"] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", SPEC_CASE, "--ignored", "--nocapture"])
            .env(SPEC_MODE, mode)
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(SPEC_RESULT))
                .unwrap_or_else(|| panic!("missing positive execution marker: {mode}: {stdout}")),
        )
        .unwrap();
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected, "{mode}");
        } else {
            expected = Some(actual);
        }
    }
}

const CONTINUATION_CASE: &str = "managed_plain::speculative::native_original_stochastic_speculation_restores_forks_and_preserves_limits";
const CONTINUATION_MODE: &str = "EREDU_PUBLIC_SPECULATIVE_CONTINUATION_MODE";
const COPY_LIMIT: &str = "EREDU_PUBLIC_SPECULATIVE_COPY_LIMIT";
const TRACE_LIMIT: &str = "EREDU_PUBLIC_SPECULATIVE_TRACE_LIMIT";
const TRACE_RECORD: &str = "EREDU_PUBLIC_SPECULATIVE_TRACE_RECORD";

#[derive(Default)]
struct Delivery {
    next: u64,
    bytes: u64,
    maximum: u64,
}
impl Delivery {
    fn observe(&mut self, step: &ControlledSpeculativeStep) {
        assert_eq!(
            step.sequence, self.next,
            "delivery must not rewind on restore or exchange"
        );
        self.next += 1;
        let bytes = serde_json::to_vec(step).unwrap().len() as u64;
        self.bytes += bytes;
        self.maximum = self.maximum.max(bytes);
    }
    fn drain(
        &mut self,
        session: &mut dyn ControlledSpeculativeSession,
    ) -> Result<(), eredu_core::speculative::SpeculativeControlError> {
        while let Some(step) = session.step()? {
            self.observe(&step);
        }
        Ok(())
    }
}
fn copy_refusal(error: &eredu_core::speculative::SpeculativeControlError) -> bool {
    matches!(
        error,
        eredu_core::speculative::SpeculativeControlError::Control(ExecutionControlError::Limit(
            "cumulative copy bytes"
        ))
    )
}
fn run_continuation(mode: &str) -> serde_json::Value {
    run_continuation_artifacts(
        mode,
        fixture(false),
        fixture(false),
        DraftPlacementPlan::Target,
    )
}
fn run_continuation_artifacts(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    placement: DraftPlacementPlan,
) -> serde_json::Value {
    run_continuation_artifacts_configured(mode, target, draft, placement, settings(0.7))
}
fn run_continuation_artifacts_configured(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    placement: DraftPlacementPlan,
    generation: PreparedChatGenerationSettings,
) -> serde_json::Value {
    run_continuation_artifacts_on(
        mode,
        target,
        draft,
        eredu_core::DevicePlan::new("mlx", "metal:0").unwrap(),
        placement,
        generation,
    )
}
fn run_continuation_artifacts_on(
    mode: &str,
    target: Fixture,
    draft: Fixture,
    target_device: eredu_core::DevicePlan,
    placement: DraftPlacementPlan,
    generation: PreparedChatGenerationSettings,
) -> serde_json::Value {
    use eredu_core::speculative::SpeculativeControlError;
    use std::{cell::RefCell, rc::Rc};
    let target = managed_fixture(target);
    let draft = managed_fixture(draft);
    let execution = ExecutionPlan::fully_resident(target_device)
        .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
        .with_drafting(DraftingPlan::External {
            model: draft.0.display().to_string(),
            placement,
            max_draft_tokens: 1,
            lookahead: false,
            adaptive_lookahead: false,
        });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let visible = Rc::new(RefCell::new(String::new()));
    let event_text = visible.clone();
    let on_event = move |event| {
        if let SemanticEvent::TextDelta(text) = event {
            event_text.borrow_mut().push_str(text.as_str());
        }
    };
    if mode == "ordinary" {
        // This is the actual independent speculative baseline, including its
        // target/draft key split and acceptance draws, not autoregressive RNG.
        let chat = model
            .source_chat_with_capacity(
                ChatTemplateRequest {
                    messages: vec![serde_json::json!({"role":"user", "content":PROMPT})],
                    add_generation_prompt: false,
                    tool_choice: ToolChoice::None,
                    ..Default::default()
                },
                8 * 1024 * 1024 * 1024,
            )
            .unwrap();
        let ids = model.encode(PROMPT, false).unwrap();
        assert_eq!(ids, [0, 1, 2, 3, 4]);
        let mut ordinary = generation;
        ordinary.inference.memory_limits = eredu_core::MemoryLimitDeclarations::unlimited();
        let output = model
            .generate_prepared_chat_speculative(PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&ids),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: drafting.as_speculative_draft().unwrap(),
                settings: chat_settings(&chat, ordinary),
                options,
                caller_stop_sequences: &[],
                cancellation: Default::default(),
                on_event,
            })
            .unwrap_or_else(report_failure);
        assert_eq!(output.token_ids().len(), 4);
        return serde_json::json!({"ids":output.token_ids(), "text":*visible.borrow()});
    }
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let request = ManagedPlainTextSpeculativeRequest {
        text: ManagedPlainTextRequest::new(PROMPT, generation),
        drafting: drafting.as_speculative_draft().unwrap(),
        options,
        cancellation: Default::default(),
        on_event,
    };
    if mode == "managed" {
        let output = model
            .generate_managed_plain_text_speculative(&source, request)
            .unwrap_or_else(report_failure);
        assert_eq!(output.token_ids().len(), 4);
        drop((source, loaded));
        return serde_json::json!({"ids":output.token_ids(), "text":*visible.borrow()});
    }
    let copy_limit = std::env::var(COPY_LIMIT)
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1 << 30);
    let trace_limit = std::env::var(TRACE_LIMIT)
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(64 << 20);
    let trace_record = std::env::var(TRACE_RECORD)
        .ok()
        .map(|s| s.parse().unwrap())
        .unwrap_or(1 << 20);
    let control = ControlledSpeculativeOptions {
        snapshots: Some(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 128 << 20,
            cumulative_copy_bytes: copy_limit,
        }),
        trace_limits: TraceLimits {
            per_record_bytes: trace_record,
            total_bytes: trace_limit,
        },
        ..Default::default()
    };
    let mut initial_copy = 0;
    let mut first_trace = 0;
    let mut maximum_record = 0;
    let mut refused_trace = false;
    let output = model.with_controlled_managed_plain_text_speculative(
        &source,
        request,
        control,
        |session| {
            let mut delivery = Delivery::default();
            let first = session.step()?.expect("first target commitment");
            delivery.observe(&first);
            assert_eq!(session.token_ids().len(), 1);
            assert_eq!(first.committed_token_ids.as_ref(), session.token_ids());
            assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
            let prefix = session.token_ids().to_vec();
            let prefix_text = visible.borrow().clone();
            let saved = session.snapshot()?;
            let after_save = session.snapshot_usage();
            initial_copy = after_save.cumulative_copy_bytes;
            assert!(initial_copy > 0);
            assert_eq!(after_save.snapshots, 1);
            assert!(matches!(
                session.snapshot(),
                Err(SpeculativeControlError::Control(
                    ExecutionControlError::Limit("snapshot count")
                ))
            ));
            assert_eq!(session.snapshot_usage(), after_save);
            if mode == "copy-limit" {
                assert_eq!(
                    initial_copy, copy_limit,
                    "calibration uses this same first saved state"
                );
                let before_epoch = session.epoch();
                let error = session.restore(&saved).unwrap_err();
                assert!(copy_refusal(&error), "{error:?}");
                assert_eq!(session.epoch(), before_epoch);
                assert_eq!(session.token_ids(), prefix);
                assert_eq!(session.snapshot_usage(), after_save);
                session.release_snapshot(&saved)?;
                let released = session.snapshot_usage();
                assert_eq!(released.snapshots, 0);
                assert_eq!(released.cumulative_copy_bytes, initial_copy);
                let error = session.snapshot().unwrap_err();
                assert!(
                    copy_refusal(&error),
                    "released storage must not refund copying: {error:?}"
                );
                assert_eq!(session.snapshot_usage(), released);
                delivery.drain(session)?;
                return Ok(());
            }
            delivery.drain(session)?;
            let expected = session.token_ids().to_vec();
            let expected_text = visible.borrow().clone();
            assert_eq!(expected.len(), 4);
            first_trace = delivery.bytes;
            maximum_record = delivery.maximum;
            if mode == "trace-limit" {
                let mut restored = 0;
                loop {
                    session.restore(&saved)?;
                    restored += 1;
                    assert_eq!(session.token_ids(), prefix);
                    *visible.borrow_mut() = prefix_text.clone();
                    loop {
                        match session.step() {
                            Ok(Some(step)) => {
                                delivery.observe(&step);
                                assert!(delivery.bytes <= trace_limit);
                            }
                            Ok(None) => {
                                assert_eq!(session.token_ids(), expected);
                                break;
                            }
                            Err(error) => {
                                assert!(
                                    matches!(
                                        error,
                                        SpeculativeControlError::Capture(CaptureError::Limit {
                                            budget: CaptureBudget::Encoded,
                                            cumulative: true
                                        })
                                    ),
                                    "{error:?}"
                                );
                                assert!(restored > 0);
                                assert!(
                                    delivery.bytes > first_trace,
                                    "at least one restored delivery was admitted"
                                );
                                let used = session.snapshot_usage().cumulative_copy_bytes;
                                session.release_snapshot(&saved)?;
                                assert_eq!(session.snapshot_usage().cumulative_copy_bytes, used);
                                refused_trace = true;
                                return Err(error);
                            }
                        }
                    }
                }
            }
            assert_eq!(mode, "controlled");
            let child = session.fork(&saved)?;
            let forked = session.snapshot_usage();
            assert_eq!(forked.branches, 1);
            assert!(matches!(
                session.fork(&saved),
                Err(SpeculativeControlError::Control(
                    ExecutionControlError::Limit("branch count")
                ))
            ));
            assert_eq!(session.snapshot_usage(), forked);
            let before_restore = session.snapshot_usage();
            session.restore(&saved)?;
            assert_eq!(session.epoch(), 1);
            assert_eq!(session.token_ids(), prefix);
            assert!(
                session.snapshot_usage().cumulative_copy_bytes
                    > before_restore.cumulative_copy_bytes
            );
            *visible.borrow_mut() = prefix_text.clone();
            delivery.drain(session)?;
            assert_eq!(session.token_ids(), expected);
            assert_eq!(
                *visible.borrow(),
                expected_text,
                "semantic decoder and RNG replay together"
            );
            let parent_sampling = session.sampling_state();
            let before_exchange = session.snapshot_usage().cumulative_copy_bytes;
            let incoming = session.exchange(&child)?;
            assert_ne!(incoming.run_id, 0);
            assert_eq!(incoming.token_ids.as_ref(), prefix);
            assert_eq!(session.token_ids(), prefix);
            assert!(session.snapshot_usage().cumulative_copy_bytes > before_exchange);
            assert_eq!(session.branch_info(&child)?.token_ids.as_ref(), expected);
            let changed = (expected[1] + 1) % 64;
            assert!(matches!(
                session.force_next_token(64),
                Err(SpeculativeControlError::InvalidToken(64))
            ));
            assert_eq!(session.token_ids(), prefix);
            session.force_next_token(changed)?;
            assert!(matches!(
                session.force_next_token(64),
                Err(SpeculativeControlError::PendingToken)
            ));
            assert!(session.clear_forced_token()?);
            assert!(!session.clear_forced_token()?);
            assert_eq!(session.token_ids(), prefix);
            session.force_next_token(changed)?;
            *visible.borrow_mut() = prefix_text;
            delivery.drain(session)?;
            assert_eq!(session.token_ids()[1], changed);
            assert_ne!(session.token_ids(), expected);
            assert_eq!(
                session.branch_info(&child)?.token_ids.as_ref(),
                expected,
                "child mutation cannot alter inactive parent"
            );
            session.exchange(&child)?;
            assert_eq!(session.run_id(), 0);
            assert_eq!(session.token_ids(), expected);
            assert_eq!(session.sampling_state(), parent_sampling);
            assert_eq!(session.branch_info(&child)?.token_ids[1], changed);
            *visible.borrow_mut() = expected_text;
            let consumed = session.snapshot_usage().cumulative_copy_bytes;
            session.release_branch(&child)?;
            session.release_snapshot(&saved)?;
            let released = session.snapshot_usage();
            assert_eq!(released.snapshots, 0);
            assert_eq!(released.branches, 0);
            assert_eq!(released.retained_bytes, 0);
            assert_eq!(released.cumulative_copy_bytes, consumed);
            Ok(())
        },
    );
    if mode == "trace-limit" {
        let error = output.unwrap_err();
        assert!(refused_trace, "{error:?}");
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&error);
        let mut typed = false;
        while let Some(error) = cause {
            typed |= error
                .downcast_ref::<SpeculativeControlError>()
                .is_some_and(|cause| {
                    matches!(
                        cause,
                        SpeculativeControlError::Capture(CaptureError::Limit {
                            budget: CaptureBudget::Encoded,
                            cumulative: true
                        })
                    )
                });
            typed |= error.downcast_ref::<CaptureError>().is_some_and(|cause| {
                matches!(
                    cause,
                    CaptureError::Limit {
                        budget: CaptureBudget::Encoded,
                        cumulative: true
                    }
                )
            });
            cause = error.source();
        }
        assert!(
            typed,
            "public error must retain the actual trace refusal: {error:?}"
        );
        drop((error, source, loaded));
        return serde_json::json!({"trace_refused":true});
    }
    let output = output.unwrap_or_else(report_failure);
    assert_eq!(output.token_ids().len(), 4);
    drop((source, loaded));
    serde_json::json!({"ids":output.token_ids(), "text":*visible.borrow(),
        "initial_copy":initial_copy, "first_trace":first_trace, "maximum_record":maximum_record})
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_stochastic_speculation_restores_forks_and_preserves_limits() {
    if let Ok(mode) = std::env::var(CONTINUATION_MODE) {
        println!("\n{SPEC_RESULT}{}", run_continuation(&mode));
        return;
    }
    compare_continuation_modes(CONTINUATION_CASE, CONTINUATION_MODE, SPEC_RESULT);
}
fn compare_continuation_modes(case: &str, mode_env: &str, marker: &str) {
    let mut expected = None;
    let mut measured: Option<serde_json::Value> = None;
    for mode in [
        "ordinary",
        "managed",
        "controlled",
        "copy-limit",
        "trace-limit",
    ] {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", case, "--ignored", "--nocapture"])
            .env(mode_env, mode);
        if mode == "copy-limit" {
            command.env(
                COPY_LIMIT,
                measured.as_ref().unwrap()["initial_copy"]
                    .as_u64()
                    .unwrap()
                    .to_string(),
            );
        }
        if mode == "trace-limit" {
            let measured = measured.as_ref().unwrap();
            let record = measured["maximum_record"].as_u64().unwrap();
            command.env(TRACE_RECORD, (4 * record).to_string()).env(
                TRACE_LIMIT,
                (measured["first_trace"].as_u64().unwrap() + 2 * record).to_string(),
            );
        }
        let result = command.output().unwrap();
        let stdout = String::from_utf8_lossy(&result.stdout);
        assert!(
            result.status.success(),
            "{mode}: {stdout}\n{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let actual: serde_json::Value = serde_json::from_str(
            stdout
                .lines()
                .find_map(|line| line.strip_prefix(marker))
                .unwrap_or_else(|| panic!("missing positive marker: {mode}: {stdout}")),
        )
        .unwrap();
        if mode == "trace-limit" {
            assert_eq!(actual["trace_refused"], true);
            continue;
        }
        let canonical = serde_json::json!({"ids":actual["ids"], "text":actual["text"]});
        if let Some(expected) = &expected {
            assert_eq!(&canonical, expected, "{mode}");
        } else {
            expected = Some(canonical);
        }
        if mode == "controlled" {
            measured = Some(actual);
        }
    }
}

#[path = "speculative/residency.rs"]
mod residency;

#[path = "speculative/capture.rs"]
mod capture;

#[path = "speculative/interventions.rs"]
mod interventions;

#[path = "speculative/batch.rs"]
mod batch;

#[path = "speculative/assistant.rs"]
mod assistant;

#[path = "speculative/cpu_assistant.rs"]
mod cpu_assistant;

#[path = "speculative/cpu_target.rs"]
mod cpu_target;

#[path = "speculative/expert_cache.rs"]
mod expert_cache;
