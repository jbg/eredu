#[path = "policy/cold_estimate.rs"]
pub(super) mod cold_estimate;

use super::*;
use crate::host_authority::Guard;
use eredu_core::{TextGeneration, TextGenerationDriver, TokenFilterController};
use eredu_runtime::{
    execution_control::{
        ManagedTextContinuation, SnapshotBudget, SnapshotTokenController, TextBranchRequest,
        TextContinuationSnapshot, TextSnapshotError,
    },
    working_memory::WorkingMemoryPool,
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

fn branch_request() -> TextBranchRequest<'static> {
    TextBranchRequest {
        session_id: "snapshot-policy-child",
        max_predictions: 4,
        capture_limits: None,
        intervention: None,
        host_bytes: Some(64),
        continuation_growth_bytes: Some(64),
    }
}

fn advance(
    state: &mut ManagedTextContinuation<MockBackend, Controller>,
    driver: &mut TextGenerationDriver<'_, MockBackend>,
) -> Option<u32> {
    let token = state.advance(driver).unwrap()?.into_output().0;
    state.take_completed_delivery(driver).unwrap();
    Some(token)
}

#[test]
fn capture_and_fork_preserve_parent_policy_and_match_ordinary_outputs() {
    let expected = ordinary();
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let probe = Guard::new(&pool);
    let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = ManagedTextContinuation::root(
        driver
            .start(vec![11, 7, 3].into(), config(), Controller::default())
            .unwrap(),
    );
    let budget = budget();
    assert_eq!(advance(&mut state, &mut driver), Some(expected[0]));
    let first = probe.update(|p| p.steps[0].clone());
    assert_eq!(first.attempt(), 0);
    let saved = TextContinuationSnapshot::capture(
        &mut state.boundary(&mut driver).unwrap(),
        &budget,
        Some(64),
    )
    .unwrap();
    let mut child = saved
        .fork(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            branch_request(),
        )
        .unwrap();
    assert_eq!(
        probe.update(|p| p.steps.len()),
        1,
        "copying must not issue a prediction"
    );
    while advance(&mut state, &mut driver).is_some() {}
    assert_eq!(state.controller().0, expected);
    let parent_steps = probe.update(|p| p.steps.clone());
    assert_eq!(parent_steps.len(), expected.len());
    for (ordinal, context) in parent_steps.iter().enumerate() {
        assert_eq!(context.run_identity(), first.run_identity());
        assert_eq!(context.policy_identity(), first.policy_identity());
        assert_eq!(context.attempt(), ordinal as u64);
    }

    child.exchange(&mut driver, &mut state).unwrap();
    while advance(&mut state, &mut driver).is_some() {}
    assert_eq!(state.controller().0, expected);
    let child_steps = probe.update(|p| p.steps[expected.len()..].to_vec());
    assert_eq!(child_steps.len(), expected.len() - 1);
    assert_ne!(child_steps[0].run_identity(), first.run_identity());
    for (ordinal, context) in child_steps.iter().enumerate() {
        assert_eq!(context.run_identity(), child_steps[0].run_identity());
        assert_eq!(context.policy_identity(), child_steps[0].policy_identity());
        assert_eq!(context.attempt(), ordinal as u64);
    }
}

#[test]
fn failed_copy_staging_preserves_policy_and_installed_restore_revises_without_refund() {
    let expected = ordinary();
    for operation in ["capture", "restore", "fork"] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let probe = Guard::new(&pool);
        let mut runtime = ModelRuntime::prepare(MockBackend, ()).unwrap();
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = ManagedTextContinuation::root(
            driver
                .start(vec![11, 7, 3].into(), config(), Controller::default())
                .unwrap(),
        );
        let budget = budget();
        assert_eq!(advance(&mut state, &mut driver), Some(expected[0]));
        let first = probe.update(|p| p.steps[0].clone());
        let saved = TextContinuationSnapshot::capture(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            Some(64),
        )
        .unwrap();
        probe.update(|p| p.fail_copy = Some("sampling copy"));
        let failed = {
            let mut boundary = state.boundary(&mut driver).unwrap();
            match operation {
                "capture" => {
                    TextContinuationSnapshot::capture(&mut boundary, &budget, Some(64)).map(|_| ())
                }
                "restore" => saved.restore(&mut boundary, &budget),
                "fork" => saved
                    .fork(&mut boundary, &budget, branch_request())
                    .map(|_| ()),
                _ => unreachable!(),
            }
        };
        assert!(
            matches!(failed, Err(TextSnapshotError::Backend(MockError::Capture(message))) if message == "host snapshot copy failed")
        );
        assert_eq!(probe.update(|p| p.steps.len()), 1);
        assert_eq!(state.controller().0, expected[..1]);
        probe.update(|p| p.fail_copy = None);

        assert_eq!(advance(&mut state, &mut driver), Some(expected[1]));
        let after_copy = probe.update(|p| p.steps[1].clone());
        assert_eq!(
            after_copy.run_identity(),
            first.run_identity(),
            "{operation}"
        );
        assert_eq!(
            after_copy.policy_identity(),
            first.policy_identity(),
            "{operation}"
        );
        assert_eq!(after_copy.attempt(), 1);

        saved
            .restore(&mut state.boundary(&mut driver).unwrap(), &budget)
            .unwrap();
        assert_eq!(state.controller().0, expected[..1]);
        assert_eq!(advance(&mut state, &mut driver), Some(expected[1]));
        let after_restore = probe.update(|p| p.steps[2].clone());
        assert_eq!(after_restore.run_identity(), first.run_identity());
        assert_ne!(after_restore.policy_identity(), first.policy_identity());
        assert_eq!(
            after_restore.attempt(),
            2,
            "restoration never refunds issued attempts"
        );
        while advance(&mut state, &mut driver).is_some() {}
        assert_eq!(state.controller().0, expected);
        let steps = probe.update(|p| p.steps.clone());
        for (ordinal, context) in steps.iter().enumerate().skip(2) {
            assert_eq!(context.run_identity(), first.run_identity());
            assert_eq!(context.policy_identity(), after_restore.policy_identity());
            assert_eq!(context.attempt(), ordinal as u64);
        }
    }
}
