//! Reusable continuation checks over the production portable snapshot driver.
//!
//! Fixtures supply actual backend mechanisms, an ordinary prepared continuation,
//! resource bounds and a probe for forward execution, resets and weight loading.
//! Run with ordinary, captured, intervened and combined admissions. These checks
//! concern native/sampler/controller/record state; facade semantic conformance is
//! additional, and must exercise the facade's incremental decoding pipeline.

use eredu_core::{capture::*, execution_control::*, TextGenerationDriver, TokenFilterController};
use eredu_runtime::execution_control::{
    apply_sampling_override, ManagedTextContinuation, SamplingOverride, SnapshotBudget,
    SnapshotTokenController, TextBranchRequest, TextContinuationSnapshot,
    TextSamplingControlBackend, TextSnapshotBackend, TextSnapshotError, TokenChoiceController,
};
use std::fmt::Debug;

#[cfg(test)]
mod tests;

/// Known fixture storage bounds and explicitly admitted child limits.
pub struct ContinuationFixtureLimits {
    /// Controller/host storage through the fixture's complete continuation.
    pub host_bytes: u64,
    /// Additional native and host growth through the absolute prediction limit.
    pub growth_bytes: u64,
    /// Absolute prediction limit, at least eight for this scenario.
    pub max_predictions: u64,
    /// Explicit child capture limits, including inherited consumption.
    pub capture: Option<CaptureLimits>,
}

fn step<B: TextSnapshotBackend, C: TokenFilterController>(
    driver: &mut TextGenerationDriver<'_, B>,
    state: &mut ManagedTextContinuation<B, C>,
) -> (u32, Option<CapturedStep>) {
    let token = state.advance(driver).unwrap().expect("fixture ended early");
    assert!(matches!(
        state.boundary(driver),
        Err(eredu_core::TextContinuationError::NotQuiescent)
    ));
    let records = state.take_completed_step(driver).unwrap().map(|mut step| {
        step.capture_seconds = 0.0;
        step.cumulative_usage = Default::default();
        step
    });
    (token.token_id(), records)
}

fn branch_values(mut value: (u32, Option<CapturedStep>)) -> (u32, Option<CapturedStep>) {
    if let Some(step) = &mut value.1 {
        for record in &mut step.interventions {
            // Child plans are intentionally re-admitted with fresh identities.
            record.plan_id.clear();
        }
    }
    value
}

/// Checks read-only rejection, exact inherited randomness across temperature
/// changes, and reproducible explicit reseeding in an isolated child. Restores
/// the initial parent afterward without model replay.
pub fn sampling_override_conformance<B, C, P>(
    driver: &mut TextGenerationDriver<'_, B>,
    state: &mut ManagedTextContinuation<B, C>,
    limits: &ContinuationFixtureLimits,
    probe: impl Fn() -> P,
) where
    B: TextSnapshotBackend + TextSamplingControlBackend,
    C: SnapshotTokenController + Clone + PartialEq + Debug,
    P: PartialEq + Debug,
{
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 1,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 512_000_000,
    });
    let initial = TextContinuationSnapshot::capture(
        &mut state.boundary(driver).unwrap(),
        &budget,
        Some(limits.host_bytes),
    )
    .unwrap();
    let facts = B::sampling_control_facts(state.boundary(driver).unwrap().parts().1);
    let baseline: Vec<_> = (0..3).map(|_| step(driver, state)).collect();
    initial
        .restore(&mut state.boundary(driver).unwrap(), &budget)
        .unwrap();
    let before = probe();
    assert!(apply_sampling_override(
        &mut state.boundary(driver).unwrap(),
        SamplingOverride {
            temperature: Some(f32::NAN),
            reseed: Some(123),
        }
    )
    .is_err());
    let greedy = apply_sampling_override(
        &mut state.boundary(driver).unwrap(),
        SamplingOverride {
            temperature: Some(0.0),
            reseed: None,
        },
    );
    if facts.requires_positive_temperature {
        assert!(greedy.is_err());
    } else {
        assert!(
            greedy.unwrap().has_rng,
            "temporary greedy selection discarded inherited RNG"
        );
    }
    apply_sampling_override(
        &mut state.boundary(driver).unwrap(),
        SamplingOverride {
            temperature: Some(facts.temperature),
            reseed: None,
        },
    )
    .unwrap();
    assert_eq!(probe(), before);
    let actual: Vec<_> = (0..3).map(|_| step(driver, state)).collect();
    assert_eq!(
        actual, baseline,
        "invalid/no-draw changes lost RNG or sampler state"
    );
    initial
        .restore(&mut state.boundary(driver).unwrap(), &budget)
        .unwrap();
    let mut child = initial
        .fork(
            &mut state.boundary(driver).unwrap(),
            &budget,
            TextBranchRequest {
                session_id: "conformance-sampling-child",
                max_predictions: limits.max_predictions,
                capture_limits: limits.capture.clone(),
                intervention: None,
                host_bytes: Some(limits.host_bytes),
                continuation_growth_bytes: Some(limits.growth_bytes),
            },
        )
        .unwrap();
    child.exchange(driver, state).unwrap();
    let before = probe();
    let updated = apply_sampling_override(
        &mut state.boundary(driver).unwrap(),
        SamplingOverride {
            temperature: Some(0.35),
            reseed: Some(711),
        },
    )
    .unwrap();
    assert_eq!(updated.temperature, 0.35);
    assert!(updated.has_rng);
    let modified = TextContinuationSnapshot::capture(
        &mut state.boundary(driver).unwrap(),
        &budget,
        Some(limits.host_bytes),
    )
    .unwrap();
    assert_eq!(probe(), before, "sampling change executed model work");
    let changed: Vec<_> = (0..3).map(|_| step(driver, state)).collect();
    modified
        .restore(&mut state.boundary(driver).unwrap(), &budget)
        .unwrap();
    assert_eq!(
        B::sampling_control_facts(state.boundary(driver).unwrap().parts().1),
        updated
    );
    let again: Vec<_> = (0..3).map(|_| step(driver, state)).collect();
    assert_eq!(
        again, changed,
        "snapshot lost explicit reseed or temperature change"
    );
    child.exchange(driver, state).unwrap();
    assert_eq!(
        B::sampling_control_facts(state.boundary(driver).unwrap().parts().1),
        facts
    );
    let parent: Vec<_> = (0..3).map(|_| step(driver, state)).collect();
    assert_eq!(parent, baseline, "sampling override changed the parent");
    initial
        .restore(&mut state.boundary(driver).unwrap(), &budget)
        .unwrap();
}

/// Exercises a genuinely alternative canonical choice, isolated branch commitment,
/// and a reusable snapshot with a still-pending choice. The fixture must allow at
/// least two tokens at its initial decision. Restores the initial state afterward,
/// so the same run can enter `continuation_conformance` without replay.
pub fn forced_choice_conformance<B, C, P>(
    driver: &mut TextGenerationDriver<'_, B>,
    state: &mut ManagedTextContinuation<B, TokenChoiceController<C>>,
    limits: &ContinuationFixtureLimits,
    vocabulary: usize,
    probe: impl Fn() -> P,
) where
    B: TextSnapshotBackend,
    C: SnapshotTokenController + Clone + PartialEq + Debug,
    P: PartialEq + Debug,
{
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 1,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 512_000_000,
    });
    let initial = TextContinuationSnapshot::capture(
        &mut state.boundary(driver).unwrap(),
        &budget,
        Some(limits.host_bytes),
    )
    .unwrap();
    let baseline = step(driver, state).0;
    initial
        .restore(&mut state.boundary(driver).unwrap(), &budget)
        .unwrap();
    let filter = state.controller_mut().current_filter().unwrap();
    let alternative = (0..vocabulary as u32)
        .find(|&token| {
            token != baseline
                && filter
                    .allowed_mask()
                    .is_none_or(|mask| mask[token as usize])
        })
        .expect("fixture must admit an alternative canonical token");
    let mut expected = state.controller().inner().clone();
    expected.commit_token(alternative).unwrap();
    let before = probe();
    let mut child = initial
        .fork(
            &mut state.boundary(driver).unwrap(),
            &budget,
            TextBranchRequest {
                session_id: "conformance-forced-child",
                max_predictions: limits.max_predictions,
                capture_limits: limits.capture.clone(),
                intervention: None,
                host_bytes: Some(limits.host_bytes),
                continuation_growth_bytes: Some(limits.growth_bytes),
            },
        )
        .unwrap();
    child.exchange(driver, state).unwrap();
    state.controller_mut().force_next(alternative).unwrap();
    let pending = TextContinuationSnapshot::capture(
        &mut state.boundary(driver).unwrap(),
        &budget,
        Some(limits.host_bytes),
    )
    .unwrap();
    assert_eq!(probe(), before, "choice/fork/capture executed the model");
    assert_eq!(step(driver, state).0, alternative);
    assert!(state.controller().last_committed_was_forced());
    assert_eq!(state.controller().inner(), &expected);
    let continuation: Vec<_> = (0..3).map(|_| step(driver, state)).collect();
    assert!(!state.controller().last_committed_was_forced());
    for _ in 0..2 {
        let before = probe();
        pending
            .restore(&mut state.boundary(driver).unwrap(), &budget)
            .unwrap();
        assert_eq!(probe(), before, "restore replayed the forced prefix");
        assert_eq!(state.controller().pending_forced(), Some(alternative));
        assert_eq!(step(driver, state).0, alternative);
        assert_eq!(state.controller().inner(), &expected);
        let actual: Vec<_> = (0..3).map(|_| step(driver, state)).collect();
        assert_eq!(actual, continuation);
    }
    child.exchange(driver, state).unwrap();
    assert_eq!(state.controller().pending_forced(), None);
    assert_eq!(step(driver, state).0, baseline, "child changed the parent");
    initial
        .restore(&mut state.boundary(driver).unwrap(), &budget)
        .unwrap();
}

/// Checks reusable initial/decode snapshots, pending-input position, exact sampled
/// continuations, controller restoration, isolated interleaved siblings, fresh
/// child admissions, cumulative capture accounting and retention-lease ownership.
/// `probe` must change on model forward, reset, artifact reopen or weight reload;
/// it must not count copying. Nonzero-temperature/adaptive fixtures detect lost
/// RNG/history state. This function consumes the initial continuation for cleanup
/// checks; its driver remains usable only for inspecting the fixture afterward.
pub fn continuation_conformance<B, C, P>(
    driver: &mut TextGenerationDriver<'_, B>,
    mut state: ManagedTextContinuation<B, C>,
    limits: ContinuationFixtureLimits,
    probe: impl Fn() -> P,
) where
    B: TextSnapshotBackend,
    C: SnapshotTokenController + Clone + PartialEq + Debug,
    P: PartialEq + Debug,
{
    assert!(limits.max_predictions >= 8);
    let budget = SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 2,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 512_000_000,
    });
    let before = probe();
    assert!(matches!(
        TextContinuationSnapshot::capture(&mut state.boundary(driver).unwrap(), &budget, None),
        Err(TextSnapshotError::Control(
            ExecutionControlError::UnknownEstimate
        ))
    ));
    assert_eq!(budget.usage(), SnapshotUsage::default());
    assert_eq!(probe(), before);
    let initial = TextContinuationSnapshot::capture(
        &mut state.boundary(driver).unwrap(),
        &budget,
        Some(limits.host_bytes),
    )
    .unwrap();
    assert_eq!(initial.next_prediction(), 0);
    let initial_native = B::estimate_native_text_state(driver.runtime(), None)
        .unwrap()
        .unwrap()
        .retained_bytes;
    let initial_growth = initial
        .native_continuation_growth(driver.runtime(), limits.max_predictions)
        .unwrap();
    assert_eq!(
        probe(),
        before,
        "growth estimation executed or rebuilt the model"
    );
    let prefix: Vec<_> = (0..3).map(|_| step(driver, &mut state)).collect();
    let prefix_controller = state.controller().clone();
    let saved = TextContinuationSnapshot::capture(
        &mut state.boundary(driver).unwrap(),
        &budget,
        Some(limits.host_bytes),
    )
    .unwrap();
    assert_eq!(saved.next_prediction(), 3);
    let usage = budget.usage();
    let before = probe();
    assert!(matches!(
        TextContinuationSnapshot::capture(
            &mut state.boundary(driver).unwrap(),
            &budget,
            Some(limits.host_bytes),
        ),
        Err(TextSnapshotError::Control(ExecutionControlError::Limit(
            "snapshot count"
        )))
    ));
    assert_eq!(budget.usage(), usage);
    let native_growth = saved
        .native_continuation_growth(driver.runtime(), limits.max_predictions)
        .unwrap();
    let child = |session_id| TextBranchRequest {
        session_id,
        max_predictions: limits.max_predictions,
        capture_limits: limits.capture.clone(),
        intervention: None,
        host_bytes: Some(limits.host_bytes),
        continuation_growth_bytes: Some(limits.growth_bytes.checked_add(native_growth).unwrap()),
    };
    let mut left = saved
        .fork(
            &mut state.boundary(driver).unwrap(),
            &budget,
            child("conformance-left"),
        )
        .unwrap();
    let mut right = saved
        .fork(
            &mut state.boundary(driver).unwrap(),
            &budget,
            child("conformance-right"),
        )
        .unwrap();
    assert_eq!(probe(), before, "copy/fork executed or rebuilt the model");
    let baseline: Vec<_> = (0..5).map(|_| step(driver, &mut state)).collect();
    let native_after = B::estimate_native_text_state(driver.runtime(), None)
        .unwrap()
        .unwrap()
        .retained_bytes;
    assert!(
        native_after <= initial_native.checked_add(initial_growth).unwrap(),
        "retained native state exceeded the pre-generation growth allowance"
    );
    left.exchange(driver, &mut state).unwrap();
    assert!(matches!(
        saved.restore(&mut state.boundary(driver).unwrap(), &budget),
        Err(TextSnapshotError::IncompatibleRun)
    ));
    if let Some(checkpoint) = saved.capture_checkpoint() {
        let boundary = state.boundary(driver).unwrap();
        assert_eq!(
            B::capture_run(boundary.parts().1)
                .unwrap()
                .cumulative_usage(),
            checkpoint.inherited_usage()
        );
    }
    let first_left = step(driver, &mut state);
    if let (Some(child), Some(parent)) = (&first_left.1, &baseline[0].1) {
        for (child, parent) in child.interventions.iter().zip(&parent.interventions) {
            assert_ne!(
                child.plan_id, parent.plan_id,
                "child reused a session-bound admission"
            );
        }
    }
    assert_eq!(
        branch_values(first_left),
        branch_values(baseline[0].clone())
    );
    left.exchange(driver, &mut state).unwrap();
    right.exchange(driver, &mut state).unwrap();
    let other: Vec<_> = (0..5)
        .map(|_| branch_values(step(driver, &mut state)))
        .collect();
    assert_eq!(
        other,
        baseline
            .iter()
            .cloned()
            .map(branch_values)
            .collect::<Vec<_>>()
    );
    right.exchange(driver, &mut state).unwrap();
    left.exchange(driver, &mut state).unwrap();
    let rest: Vec<_> = (0..4)
        .map(|_| branch_values(step(driver, &mut state)))
        .collect();
    assert_eq!(
        rest,
        baseline[1..]
            .iter()
            .cloned()
            .map(branch_values)
            .collect::<Vec<_>>()
    );
    left.exchange(driver, &mut state).unwrap();
    let capture_usage = {
        let boundary = state.boundary(driver).unwrap();
        B::capture_run(boundary.parts().1).map(|run| run.cumulative_usage())
    };
    for _ in 0..2 {
        let before = probe();
        saved
            .restore(&mut state.boundary(driver).unwrap(), &budget)
            .unwrap();
        assert_eq!(probe(), before, "restore executed or rebuilt the model");
        assert_eq!(state.controller(), &prefix_controller);
        let actual: Vec<_> = (0..5).map(|_| step(driver, &mut state)).collect();
        assert_eq!(actual, baseline);
    }
    if let Some(before) = capture_usage {
        let boundary = state.boundary(driver).unwrap();
        let after = B::capture_run(boundary.parts().1)
            .unwrap()
            .cumulative_usage();
        assert!(after.captures >= before.captures);
        assert!(after.encoded_bytes >= before.encoded_bytes);
        if before.encoded_bytes != 0 {
            assert!(after.encoded_bytes > before.encoded_bytes);
        }
    }
    initial
        .restore(&mut state.boundary(driver).unwrap(), &budget)
        .unwrap();
    let actual: Vec<_> = (0..3).map(|_| step(driver, &mut state)).collect();
    assert_eq!(actual, prefix);
    drop(right);
    assert_eq!(budget.usage().branches, 1);
    left.exchange(driver, &mut state).unwrap();
    assert!(state.retained_branch_bytes().is_some());
    let usage = budget.usage();
    drop(left); // This slot now holds the original parent, not the charged child.
    assert_eq!(
        budget.usage(),
        usage,
        "dropping parent released active child retention"
    );
    drop(state);
    assert_eq!(budget.usage().branches, 0);
    drop((initial, saved));
    assert_eq!(budget.usage().retained_bytes, 0);
    assert_eq!(budget.usage().snapshots, 0);
    assert_eq!(
        budget.usage().cumulative_copy_bytes,
        usage.cumulative_copy_bytes
    );
}
