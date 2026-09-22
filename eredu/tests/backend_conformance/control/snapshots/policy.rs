#[path = "policy/cold_estimate.rs"]
pub(super) mod cold_estimate;

use super::*;
use crate::host_authority::Guard;
use eredu_core::{TextGeneration, TextGenerationDriver, TokenFilterController};
use eredu_runtime::{
    execution_control::{
        ManagedTextContinuation, SnapshotBudget, SnapshotTokenController, TextContinuationSnapshot,
        TextSnapshotError,
    },
    working_memory::MemoryLedger,
};

#[derive(Clone, Default)]
struct Controller(Vec<u32>);

impl TokenFilterController for Controller {
    type Error = std::convert::Infallible;

    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(TokenFilter::All)
    }

    fn commit_token(&mut self, token: u32) -> Result<(), Self::Error> {
        self.0.push(token);
        Ok(())
    }

    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}

impl SnapshotTokenController for Controller {
    fn snapshot_storage_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64).checked_add((self.0.capacity() as u64).checked_mul(4)?)
    }

    fn original_snapshot_storage_bytes(&self) -> Option<u64> {
        self.snapshot_storage_bytes()
    }

    fn fork_snapshot(&self) -> Result<Self, String> {
        Ok(self.clone())
    }
}

fn config() -> TextGenerationConfig {
    TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            GenerationConfigOverrides {
                max_new_tokens: Some(4),
                ..Default::default()
            },
        )
        .unwrap(),
    )
}

fn ordinary() -> Vec<u32> {
    let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    TextGeneration::from_prompt(&mut runtime, vec![11, 7, 3].into(), config())
        .unwrap()
        .map(|token| token.unwrap().0)
        .collect()
}

fn budget() -> SnapshotBudget {
    SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 4,
        max_branches: 2,
        retained_bytes: 4_000_000,
        cumulative_copy_bytes: 32_000_000,
    })
}

fn advance(
    state: &mut ManagedTextContinuation<MockBackend, Controller>,
    driver: &mut TextGenerationDriver<'_, MockBackend>,
) -> Option<u32> {
    let token = state.advance(driver).unwrap()?.into_output().0;
    state.take_completed_delivery(driver).unwrap();
    Some(token)
}

fn ignore(_: ControlledGenerationRecord) -> ControlFlow<()> {
    ControlFlow::Continue(())
}
fn branch_options() -> eredu::api::GenerationBranchOptions {
    eredu::api::GenerationBranchOptions {
        trace_limits: limits(),
        capture_limits: None,
        sampling: None,
        intervention: None,
    }
}
fn enable_snapshots(run: &mut eredu::api::ControlledGenerationSession<'_, MockBackend>) {
    run.enable_snapshots(
        SnapshotLimits {
            max_snapshots: 4,
            max_branches: 2,
            retained_bytes: 64_000_000,
            cumulative_copy_bytes: 256_000_000,
        },
        crate::memory::limits(original_sources::CAPACITY),
        eredu_runtime::working_memory::WorkspaceCopyLimits::new(crate::memory::limits(
            original_sources::CAPACITY,
        )),
    )
    .unwrap();
}
#[test]
fn capture_and_fork_preserve_parent_policy_and_match_ordinary_outputs() {
    let (mut model, chat, settings, first_token) = snapshot_setup();
    let pool = model.original_pool().clone();
    let probe = Guard::new(&pool);
    let request = eredu::api::PreparedChatRequest::new(&chat, original_sources::settings(settings));
    let mut run = model
        .start_controlled_chat(request, limits(), Default::default(), ignore)
        .unwrap()
        .unwrap();
    enable_snapshots(&mut run);
    run.step(ignore).unwrap();
    assert_eq!(run.token_ids(), [first_token]);
    let first = probe.update(|p| p.steps[0].clone());
    assert_eq!(first.attempt(), 0);
    let saved = run.snapshot(ignore).unwrap();
    let mut child = run.fork(&saved, branch_options(), ignore).unwrap();
    assert_eq!(probe.update(|p| p.steps.len()), 1);
    run.run(ignore).unwrap();
    let expected = run.token_ids().to_vec();
    assert!(expected.len() > 1);
    let parent_steps = probe.update(|p| p.steps.clone());
    assert_eq!(parent_steps.len(), expected.len());
    for (ordinal, context) in parent_steps.iter().enumerate() {
        assert_eq!(context.run_identity(), first.run_identity());
        assert_eq!(context.policy_identity(), first.policy_identity());
        assert_eq!(context.attempt(), ordinal as u64);
    }
    run.exchange(&mut child, ignore).unwrap();
    run.run(ignore).unwrap();
    assert_eq!(run.token_ids(), expected);
    let steps = probe.update(|p| p.steps[expected.len()..].to_vec());
    assert_eq!(steps.len(), expected.len() - 1);
    assert_ne!(steps[0].run_identity(), first.run_identity());
    for (ordinal, context) in steps.iter().enumerate() {
        assert_eq!(context.run_identity(), steps[0].run_identity());
        assert_eq!(context.policy_identity(), steps[0].policy_identity());
        assert_eq!(context.attempt(), ordinal as u64);
    }
}
#[test]
fn failed_copy_staging_preserves_policy_and_restore_installs_fresh_funded_run() {
    for operation in ["capture", "restore", "fork"] {
        let (mut model, chat, settings, first_token) = snapshot_setup();
        let pool = model.original_pool().clone();
        let probe = Guard::new(&pool);
        let request =
            eredu::api::PreparedChatRequest::new(&chat, original_sources::settings(settings));
        let mut run = model
            .start_controlled_chat(request, limits(), Default::default(), ignore)
            .unwrap()
            .unwrap();
        enable_snapshots(&mut run);
        run.step(ignore).unwrap();
        let first = probe.update(|p| p.steps[0].clone());
        let saved = run.snapshot(ignore).unwrap();
        let before = run.snapshot_usage().unwrap();
        let failure = super::super::provider_errors::Armed::new(if operation == "capture" {
            "capture"
        } else {
            "copy"
        });
        let error = match operation {
            "capture" => run.snapshot(ignore).map(|_| ()),
            "restore" => run.restore(&saved, ignore),
            "fork" => run.fork(&saved, branch_options(), ignore).map(|_| ()),
            _ => unreachable!(),
        }
        .unwrap_err();
        failure.assert_error(&error);
        drop(error);
        drop(failure);
        assert_eq!(probe.update(|p| p.steps.len()), 1);
        assert_eq!(run.token_ids(), [first_token]);
        assert!(run.snapshot_usage().unwrap().cumulative_copy_bytes > before.cumulative_copy_bytes);
        run.step(ignore).unwrap();
        let after_copy = probe.update(|p| p.steps[1].clone());
        assert_eq!(after_copy.run_identity(), first.run_identity());
        assert_eq!(after_copy.policy_identity(), first.policy_identity());
        assert_eq!(after_copy.attempt(), 1);
        run.restore(&saved, ignore).unwrap();
        assert_eq!(run.token_ids(), [first_token]);
        run.step(ignore).unwrap();
        let after_restore = probe.update(|p| p.steps[2].clone());
        assert_ne!(after_restore.run_identity(), first.run_identity());
        assert_ne!(after_restore.policy_identity(), first.policy_identity());
        assert_eq!(after_restore.attempt(), 0);
        run.run(ignore).unwrap();
        assert!(run.token_ids().len() > 1);
        for (ordinal, context) in probe.update(|p| p.steps.clone()).iter().enumerate().skip(2) {
            assert_eq!(context.run_identity(), after_restore.run_identity());
            assert_eq!(context.policy_identity(), after_restore.policy_identity());
            assert_eq!(context.attempt(), (ordinal - 2) as u64);
        }
    }
}
