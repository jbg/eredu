//! One active original Embedded request replays actual mutable caches and RNG.
use super::*;
use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotLimits},
    speculative::SpeculativeControlError,
};
use std::{cell::RefCell, rc::Rc};

// Temporary ledger attribution must run after unwinding: native cleanup is
// deliberately unavailable in a Drop that runs while panicking.
fn run(kind: &str) {
    // This lifecycle retains the original branch, a snapshot, restored state
    // and the exchanged branch simultaneously. Its finite success budget is
    // distinct from the 8 GiB rejection case below.
    run_with_budget(kind, 12 << 30, false);
}

fn run_with_budget(kind: &str, capacity: u64, expect_refusal: bool) {
    if std::env::var_os("EREDU_WORKSPACE_LEDGER_TRACE").is_none() {
        return run_inner(kind, capacity, expect_refusal);
    }
    let result = std::panic::catch_unwind(|| run_inner(kind, capacity, expect_refusal));
    // Reclamation consumes one detached snapshot. Dropping that snapshot can
    // enqueue predecessor owners; record successive ordinary host passes.
    for pass in 0..16 {
        eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        eprintln!("WORKSPACE_LEDGER_RECLAIM_PASS={pass}");
    }
    eprintln!("WORKSPACE_LEDGER_FIXTURE_RETIRED");
    if let Err(payload) = result {
        std::panic::resume_unwind(payload);
    }
}

fn run_inner(kind: &str, capacity: u64, expect_refusal: bool) {
    let target = managed_fixture(super::super::super::speculative::captured_prefill::source(
        kind,
    ));
    let execution =
        ExecutionPlan::fully_resident(eredu_core::DevicePlan::new("mlx", "metal:0").unwrap())
            .with_required_session_capabilities(SessionCapabilities::new(true, true, true))
            .with_drafting(DraftingPlan::Embedded {
                max_draft_tokens: 2,
                lookahead: false,
                adaptive_lookahead: false,
            });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &execution)
            .unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let source = model
        .compile_managed_plain_text_source(
            std::fs::File::open(target.0.join("tokenizer.json")).unwrap(),
        )
        .unwrap_or_else(report_failure);
    let visible = Rc::new(RefCell::new(String::new()));
    let events = visible.clone();
    let mut settings = settings(0.7);
    settings.inference.memory_limits = eredu_core::MemoryLimitDeclarations::new([(
        "host".into(),
        eredu_core::MemoryLimit::Finite(capacity),
    )]);
    let request = ManagedPlainTextSpeculativeRequest {
        text: ManagedPlainTextRequest::new(PROMPT, settings),
        drafting: drafting.as_speculative_draft().unwrap(),
        options,
        cancellation: Default::default(),
        on_event: move |event| {
            if let SemanticEvent::TextDelta(text) = event {
                events.borrow_mut().push_str(text.as_str());
            }
        },
    };
    let control = ControlledSpeculativeOptions {
        snapshots: Some(SnapshotLimits {
            max_snapshots: 1,
            max_branches: 1,
            retained_bytes: 128 << 20,
            cumulative_copy_bytes: 1 << 30,
        }),
        ..Default::default()
    };
    macro_rules! at_stage {
        ($label:literal,$operation:expr) => {{
            let result = $operation;
            if result.is_err() {
                eprintln!("CONTROL_STATE_STAGE={}", $label);
            }
            result?
        }};
    }
    let mut expected = Vec::new();
    let mut expected_text = String::new();
    let mut reached_exchanged_continuation = false;
    let output = model.with_controlled_managed_plain_text_speculative(
        &source,
        request,
        control,
        |session| {
            let first =
                at_stage!("first commitment", session.step()).expect("first target commitment");
            assert_eq!(session.token_ids().len(), 1);
            assert_eq!(first.committed_token_ids.as_ref(), session.token_ids());
            assert!(session.can_snapshot(), "{:?}", session.snapshot_support());
            let prefix = session.token_ids().to_vec();
            let prefix_text = visible.borrow().clone();
            let saved = at_stage!("snapshot", session.snapshot());
            let after_save = session.snapshot_usage();
            assert_eq!(after_save.snapshots, 1);
            assert!(after_save.cumulative_copy_bytes > 0);
            assert!(matches!(
                session.snapshot(),
                Err(SpeculativeControlError::Control(
                    ExecutionControlError::Limit("snapshot count")
                ))
            ));
            assert_eq!(session.snapshot_usage(), after_save);
            while at_stage!("finish original", session.step()).is_some() {}
            expected = session.token_ids().to_vec();
            expected_text = visible.borrow().clone();
            assert_eq!(expected.len(), 4);
            let child = at_stage!("fork", session.fork(&saved));
            let after_fork = session.snapshot_usage();
            assert_eq!(after_fork.branches, 1);
            assert_eq!(
                after_fork.cumulative_copy_bytes, after_save.cumulative_copy_bytes,
                "inactive fork shares immutable saved state; activation pays the copy"
            );
            assert!(matches!(
                session.fork(&saved),
                Err(SpeculativeControlError::Control(
                    ExecutionControlError::Limit("branch count")
                ))
            ));
            assert_eq!(session.snapshot_usage(), after_fork);
            let epoch = session.epoch();
            at_stage!("restore", session.restore(&saved));
            assert_eq!(session.epoch(), epoch + 1);
            assert_eq!(session.token_ids(), prefix);
            assert!(
                session.snapshot_usage().cumulative_copy_bytes > after_fork.cumulative_copy_bytes
            );
            *visible.borrow_mut() = prefix_text.clone();
            while at_stage!("finish restored", session.step()).is_some() {}
            assert_eq!(session.token_ids(), expected);
            assert_eq!(
                *visible.borrow(),
                expected_text,
                "decoder and RNG restore with the exact caches"
            );
            let spent = session.snapshot_usage().cumulative_copy_bytes;
            let incoming = at_stage!("exchange", session.exchange(&child));
            assert_eq!(incoming.token_ids.as_ref(), prefix);
            assert_eq!(session.branch_info(&child)?.token_ids.as_ref(), expected);
            assert_eq!(session.token_ids(), prefix);
            assert!(session.snapshot_usage().cumulative_copy_bytes > spent);
            *visible.borrow_mut() = prefix_text;
            reached_exchanged_continuation = true;
            while at_stage!("finish exchanged", session.step()).is_some() {}
            assert_eq!(session.token_ids(), expected);
            assert_eq!(
                *visible.borrow(),
                expected_text,
                "fork isolates target and prediction mutation"
            );
            let spent = session.snapshot_usage().cumulative_copy_bytes;
            session.release_branch(&child)?;
            session.release_snapshot(&saved)?;
            let released = session.snapshot_usage();
            assert_eq!((released.snapshots, released.branches), (0, 0));
            assert_eq!(
                released.cumulative_copy_bytes, spent,
                "retirement cannot refund prior copies"
            );
            Ok(())
        },
    );
    if expect_refusal {
        let error = match output {
            Err(error) => error,
            Ok(_) => panic!("the retained replay fixture must exceed its 8 GiB capacity"),
        };
        assert!(
            reached_exchanged_continuation,
            "refusal must exercise the final exchanged continuation"
        );
        assert!(
            budget_failure(&error),
            "the original neutral budget refusal must survive error custody: {error}"
        );
        return;
    }
    let output = output.unwrap_or_else(report_failure);
    assert_eq!(output.token_ids(), expected);
    let address = output.token_ids().as_ptr();
    drop((source, loaded));
    assert_eq!(output.token_ids().as_ptr(), address);
    assert_eq!(output.token_ids(), expected);
    assert_eq!(*visible.borrow(), expected_text);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_qwen_embedded_snapshot_restore_and_fork_preserve_state_and_spending() {
    run("qwen");
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_inkling_embedded_snapshot_restore_and_fork_preserve_state_and_spending() {
    run("inkling");
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_v3_embedded_snapshot_restore_and_fork_preserve_state_and_spending() {
    run("v3");
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_v4_embedded_snapshot_restore_and_fork_preserve_state_and_spending() {
    run("v4");
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_dspark_embedded_snapshot_restore_and_fork_preserve_state_and_spending() {
    run("dspark");
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_nemotron_embedded_snapshot_restore_and_fork_preserve_state_and_spending() {
    run("nemotron");
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_v4_embedded_replay_budget_refusal_preserves_cause() {
    run_with_budget("v4", 8 << 30, true);
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn native_original_dspark_embedded_replay_budget_refusal_preserves_cause() {
    run_with_budget("dspark", 8 << 30, true);
}
