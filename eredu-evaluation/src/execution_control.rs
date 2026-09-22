//! Reusable continuation checks over the funded portable snapshot driver.
//!
//! Fixtures supply actual original input, host-copy, native-copy and resume
//! mechanisms. The same controlled machine advances ordinary and resumed work.
use eredu_core::{
    capture::*, execution_control::*, ControlledTextGeneration, HostMetadataFunding, ModelRuntime,
    OriginalTextResumeKind, OriginalTextResumeOptions, TextGenerationBranch, TextResumeBackend,
    TextSnapshotSource,
};
use eredu_runtime::execution_control::{
    SnapshotBudget, SnapshotTokenController, TextContinuationSnapshot, TextSnapshotBackend,
    TextSnapshotError, TokenChoiceController,
};
use std::fmt::Debug;

/// Prepared source fixtures shared by native and portable conformance tests.
pub mod fixture;

/// Known fixture storage bounds and explicitly admitted child limits.
pub struct ContinuationFixtureLimits {
    /// Complete host-copy payload bound.
    pub host_bytes: u64,
    /// Complete future native and host growth bound.
    pub growth_bytes: u64,
    /// Absolute prediction bound, at least eight for the scenario.
    pub max_predictions: u64,
    /// Child capture limits including inherited consumption.
    pub capture: Option<CaptureLimits>,
}

/// Actual funded mechanisms supplied by a conformance fixture.
///
/// Capture calls `capture_original_host`; resume calls
/// `resume_original_host_with_displaced`. Their independently admitted host
/// providers retain copy controls, output aliases and logical budget custody.
/// Implementations do not create execution authority from diagnostic bounds.
pub trait ContinuationSnapshotProvider<B, C>
where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = <B as TextSnapshotBackend>::SavedTextComponents,
            DisplacedState = <B as NativeTextStateBackend>::NativeTextState,
        >,
    C: SnapshotTokenController,
{
    /// Independently owned copied host state.
    type Host;
    /// Copies the exact completed source through ordinary physical admission.
    fn capture(
        &mut self,
        source: &mut TextSnapshotSource<'_, B, C>,
        budget: &SnapshotBudget,
        host_bytes: Option<u64>,
    ) -> Result<(TextContinuationSnapshot<B, C>, Self::Host), TextSnapshotError<B::Error>>;
    /// Admits a fresh execution and installs the actual immutable saved state.
    fn resume<'a>(
        &mut self,
        runtime: &'a mut ModelRuntime<B>,
        saved: &TextContinuationSnapshot<B, C>,
        host: &Self::Host,
        options: &OriginalTextResumeOptions<'_>,
    ) -> Result<
        Option<(
            ControlledTextGeneration<'a, B, C>,
            B::NativeTextState,
            Self::Host,
        )>,
        TextSnapshotError<B::Error>,
    >;
    /// Applies one successful model commitment to the active host sequence.
    fn observe_token(&mut self, token: u32);
    /// Exchanges complete active host state alongside the core/native machine.
    fn exchange_host(&mut self, host: &mut Self::Host);
    /// Actual cumulative usage of the installed funded capture collector.
    fn capture_usage(&self, source: &TextSnapshotSource<'_, B, C>) -> CaptureUsage;
    /// Reaps already completed native owners until the supplied retirement
    /// observation holds. Portable owners retire synchronously. A native fixture
    /// must use actual completion evidence and fail if its bounded wait expires.
    fn settle_retirement(mut complete: impl FnMut() -> bool) {
        assert!(complete(), "completed fixture owners remain retained");
    }
    /// Original finite allowance for a prospective restriction mask.
    fn choice_funding(&self) -> HostMetadataFunding;
}

fn step<B, C, H>(
    state: &mut ControlledTextGeneration<'_, B, C>,
    provider: &mut H,
) -> (u32, Option<CapturedStep>)
where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController,
    H: ContinuationSnapshotProvider<B, C>,
{
    let token = state.next().expect("fixture ended early").unwrap();
    if state.capture_pending() {
        assert!(matches!(
            state.snapshot_source(),
            Err(eredu_core::TextContinuationError::NotQuiescent)
        ));
    }
    let records = state.take_captured_delivery().unwrap().map(|step| {
        let mut step = step.as_step().clone();
        step.capture_seconds = 0.0;
        step.cumulative_usage = Default::default();
        step
    });
    state.snapshot_source().unwrap();
    provider.observe_token(token.token_id());
    (token.token_id(), records)
}
fn branch_values(mut value: (u32, Option<CapturedStep>)) -> (u32, Option<CapturedStep>) {
    if let Some(step) = &mut value.1 {
        for record in &mut step.interventions {
            record.plan_id.clear();
        }
    }
    value
}
fn budget(branches: u64) -> SnapshotBudget {
    SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: branches,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 512_000_000,
    })
}
fn capture<B, C, H>(
    state: &mut ControlledTextGeneration<'_, B, C>,
    provider: &mut H,
    budget: &SnapshotBudget,
    limits: &ContinuationFixtureLimits,
) -> (TextContinuationSnapshot<B, C>, H::Host)
where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController,
    H: ContinuationSnapshotProvider<B, C>,
{
    provider
        .capture(
            &mut state.snapshot_source().unwrap(),
            budget,
            Some(limits.host_bytes),
        )
        .unwrap()
}
fn restore<B, C, H>(
    state: &mut ControlledTextGeneration<'_, B, C>,
    provider: &mut H,
    saved: &(TextContinuationSnapshot<B, C>, H::Host),
) where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController,
    H: ContinuationSnapshotProvider<B, C>,
{
    let mut host = state
        .replace_completed(
            |runtime| {
                provider
                    .resume(
                        runtime,
                        &saved.0,
                        &saved.1,
                        &OriginalTextResumeOptions::new(OriginalTextResumeKind::Restore),
                    )
                    .map(|value| {
                        value.map(|(state, displaced, host)| {
                            drop(displaced);
                            (state, host)
                        })
                    })
            },
            |error| panic!("completed restore boundary: {error:?}"),
        )
        .unwrap()
        .expect("nonterminal saved source");
    provider.exchange_host(&mut host);
}

fn fork<B, C, H>(
    state: &mut ControlledTextGeneration<'_, B, C>,
    provider: &mut H,
    saved: &(TextContinuationSnapshot<B, C>, H::Host),
    limits: &ContinuationFixtureLimits,
    session: &str,
) -> (TextGenerationBranch<B, C>, H::Host)
where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController,
    H: ContinuationSnapshotProvider<B, C>,
{
    let mut options = OriginalTextResumeOptions::new(OriginalTextResumeKind::Branch);
    options.session_id = Some(session);
    options.capture_limits = limits.capture.as_ref();
    state
        .fork_completed(
            |runtime| provider.resume(runtime, &saved.0, &saved.1, &options),
            |error| panic!("completed branch boundary: {error:?}"),
        )
        .unwrap()
        .expect("nonterminal saved source")
}

/// Checks read-only rejection, exact inherited randomness, temperature changes,
/// reproducible reseeding and isolation from the parent.
pub fn sampling_override_conformance<B, C, H, P>(
    state: &mut ControlledTextGeneration<'_, B, C>,
    provider: &mut H,
    limits: &ContinuationFixtureLimits,
    probe: impl Fn() -> P,
) where
    B: TextSnapshotBackend
        + TextSamplingControlBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController + Clone + PartialEq + Debug,
    H: ContinuationSnapshotProvider<B, C>,
    P: PartialEq + Debug,
{
    let budget = budget(1);
    let initial = capture(state, provider, &budget, limits);
    let facts = state.sampling_boundary().unwrap().facts();
    let baseline: Vec<_> = (0..3).map(|_| step(state, provider)).collect();
    restore(state, provider, &initial);
    let before = probe();
    assert!(state
        .sampling_boundary()
        .unwrap()
        .apply(SamplingOverride {
            temperature: Some(f32::NAN),
            reseed: Some(123)
        })
        .is_err());
    assert_eq!(state.sampling_boundary().unwrap().facts(), facts);
    assert_eq!(probe(), before);
    let greedy = state.sampling_boundary().unwrap().apply(SamplingOverride {
        temperature: Some(0.0),
        reseed: None,
    });
    if facts.requires_positive_temperature {
        assert!(matches!(
            greedy,
            Err(eredu_core::SamplingOverrideError::Invalid(_))
        ));
        assert_eq!(state.sampling_boundary().unwrap().facts(), facts);
        assert_eq!(probe(), before);
    } else {
        assert_eq!(greedy.unwrap().temperature, 0.0);
    }
    let restored = state
        .sampling_boundary()
        .unwrap()
        .apply(SamplingOverride {
            temperature: Some(facts.temperature),
            reseed: None,
        })
        .unwrap();
    assert_eq!(restored, facts);
    assert_eq!(
        (0..3).map(|_| step(state, provider)).collect::<Vec<_>>(),
        baseline
    );
    restore(state, provider, &initial);
    let mut child = fork(
        state,
        provider,
        &initial,
        limits,
        "conformance-sampling-child",
    );
    state.exchange_branch(&mut child.0).unwrap();
    provider.exchange_host(&mut child.1);
    let before = probe();
    let updated = state
        .sampling_boundary()
        .unwrap()
        .apply(SamplingOverride {
            temperature: Some(0.35),
            reseed: Some(711),
        })
        .unwrap();
    assert_eq!(updated.temperature, 0.35);
    assert!(updated.has_rng);
    let modified = capture(state, provider, &budget, limits);
    assert_eq!(probe(), before, "sampling change executed model work");
    let changed: Vec<_> = (0..3).map(|_| step(state, provider)).collect();
    restore(state, provider, &modified);
    assert_eq!(state.sampling_boundary().unwrap().facts(), updated);
    assert_eq!(
        (0..3).map(|_| step(state, provider)).collect::<Vec<_>>(),
        changed
    );
    state.exchange_branch(&mut child.0).unwrap();
    provider.exchange_host(&mut child.1);
    assert_eq!(state.sampling_boundary().unwrap().facts(), facts);
    assert_eq!(
        (0..3).map(|_| step(state, provider)).collect::<Vec<_>>(),
        baseline
    );
    restore(state, provider, &initial);
}

/// Checks a genuinely alternative canonical choice, isolated commitment and a
/// reusable saved source containing a still-pending prospective restriction.
pub fn forced_choice_conformance<B, C, H, P>(
    state: &mut ControlledTextGeneration<'_, B, TokenChoiceController<C>>,
    provider: &mut H,
    limits: &ContinuationFixtureLimits,
    vocabulary: usize,
    probe: impl Fn() -> P,
) where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController
        + eredu_core::SpeculativeTokenFilterController
        + Clone
        + PartialEq
        + Debug,
    H: ContinuationSnapshotProvider<B, TokenChoiceController<C>>,
    P: PartialEq + Debug,
{
    let budget = budget(1);
    let initial = capture(state, provider, &budget, limits);
    let baseline = step(state, provider).0;
    restore(state, provider, &initial);
    let filter = state.controller().inner().clone().current_filter().unwrap();
    let alternative = (0..vocabulary as u32)
        .find(|&token| {
            token != baseline
                && filter
                    .allowed_mask()
                    .is_none_or(|mask| mask[token as usize])
        })
        .expect("alternative canonical token");
    let mut expected = state.controller().inner().clone();
    expected.commit_token(alternative).unwrap();
    let before = probe();
    let mut child = fork(
        state,
        provider,
        &initial,
        limits,
        "conformance-forced-child",
    );
    state.exchange_branch(&mut child.0).unwrap();
    provider.exchange_host(&mut child.1);
    state
        .token_choice_boundary()
        .unwrap()
        .force_next(alternative, &provider.choice_funding())
        .unwrap();
    let pending = capture(state, provider, &budget, limits);
    assert_eq!(probe(), before, "choice/fork/capture executed the model");
    assert_eq!(step(state, provider).0, alternative);
    assert!(state.controller().last_committed_was_forced());
    assert_eq!(state.controller().inner(), &expected);
    let continuation: Vec<_> = (0..3).map(|_| step(state, provider)).collect();
    assert!(!state.controller().last_committed_was_forced());
    for _ in 0..2 {
        let before = probe();
        restore(state, provider, &pending);
        assert_eq!(probe(), before, "restore replayed the forced prefix");
        assert_eq!(state.controller().pending_forced(), Some(alternative));
        assert_eq!(step(state, provider).0, alternative);
        assert_eq!(state.controller().inner(), &expected);
        assert_eq!(
            (0..3).map(|_| step(state, provider)).collect::<Vec<_>>(),
            continuation
        );
    }
    state.exchange_branch(&mut child.0).unwrap();
    provider.exchange_host(&mut child.1);
    assert_eq!(state.controller().pending_forced(), None);
    assert_eq!(step(state, provider).0, baseline);
    restore(state, provider, &initial);
}

/// Checks exact snapshot/restore output, independent child state, non-replayed
/// native work, inherited capture usage and completion-safe branch retirement.
///
/// The final child remains installed in the runtime. After this function returns,
/// the caller retires that state through its normal reset or teardown mechanism,
/// then finishes the returned retirement check.
pub fn continuation_conformance<B, C, H, P>(
    mut state: ControlledTextGeneration<'_, B, C>,
    mut provider: H,
    limits: ContinuationFixtureLimits,
    probe: impl Fn() -> P,
) -> ContinuationRetirement<B, C, H>
where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController + Clone + PartialEq + Debug,
    H: ContinuationSnapshotProvider<B, C>,
    P: PartialEq + Debug,
{
    assert!(limits.max_predictions >= 8);
    let budget = budget(2);
    let before = probe();
    assert!(matches!(
        provider.capture(&mut state.snapshot_source().unwrap(), &budget, None),
        Err(TextSnapshotError::Control(
            ExecutionControlError::UnknownEstimate
        ))
    ));
    assert_eq!(budget.usage(), SnapshotUsage::default());
    assert_eq!(probe(), before);
    let initial = capture(&mut state, &mut provider, &budget, &limits);
    let initial_position = initial.0.next_prediction();
    let initial_native = B::estimate_native_text_state(state.runtime(), None)
        .unwrap()
        .unwrap()
        .retained_bytes;
    let initial_growth = initial
        .0
        .native_continuation_growth(state.runtime(), limits.max_predictions)
        .unwrap();
    assert_eq!(
        probe(),
        before,
        "growth estimation executed or rebuilt the model"
    );
    let prefix: Vec<_> = (0..3).map(|_| step(&mut state, &mut provider)).collect();
    let prefix_controller = state.controller().clone();
    let saved = capture(&mut state, &mut provider, &budget, &limits);
    assert_eq!(saved.0.next_prediction(), initial_position + 3);
    let usage = budget.usage();
    let before = probe();
    assert!(matches!(
        provider.capture(
            &mut state.snapshot_source().unwrap(),
            &budget,
            Some(limits.host_bytes)
        ),
        Err(TextSnapshotError::Control(ExecutionControlError::Limit(
            "snapshot count"
        )))
    ));
    assert_eq!(budget.usage(), usage);
    let snapshots_retained = usage.retained_bytes;
    let mut left = fork(
        &mut state,
        &mut provider,
        &saved,
        &limits,
        "conformance-left",
    );
    let active_child_bytes = budget.usage().retained_bytes - usage.retained_bytes;
    let mut right = fork(
        &mut state,
        &mut provider,
        &saved,
        &limits,
        "conformance-right",
    );
    assert_eq!(probe(), before, "copy/fork executed or rebuilt the model");
    let baseline: Vec<_> = (0..5).map(|_| step(&mut state, &mut provider)).collect();
    let native_after = B::estimate_native_text_state(state.runtime(), None)
        .unwrap()
        .unwrap()
        .retained_bytes;
    assert!(native_after <= initial_native.checked_add(initial_growth).unwrap());
    state.exchange_branch(&mut left.0).unwrap();
    provider.exchange_host(&mut left.1);
    if let Some(checkpoint) = saved.0.capture_checkpoint() {
        assert_eq!(
            provider.capture_usage(&state.snapshot_source().unwrap()),
            checkpoint.inherited_usage()
        );
    }
    let first_left = step(&mut state, &mut provider);
    if let (Some(child), Some(parent)) = (&first_left.1, &baseline[0].1) {
        for (child, parent) in child.interventions.iter().zip(&parent.interventions) {
            assert_ne!(child.plan_id, parent.plan_id);
        }
    }
    assert_eq!(
        branch_values(first_left),
        branch_values(baseline[0].clone())
    );
    state.exchange_branch(&mut left.0).unwrap();
    provider.exchange_host(&mut left.1);
    state.exchange_branch(&mut right.0).unwrap();
    provider.exchange_host(&mut right.1);
    assert_eq!(
        (0..5)
            .map(|_| branch_values(step(&mut state, &mut provider)))
            .collect::<Vec<_>>(),
        baseline
            .iter()
            .cloned()
            .map(branch_values)
            .collect::<Vec<_>>()
    );
    state.exchange_branch(&mut right.0).unwrap();
    provider.exchange_host(&mut right.1);
    state.exchange_branch(&mut left.0).unwrap();
    provider.exchange_host(&mut left.1);
    assert_eq!(
        (0..4)
            .map(|_| branch_values(step(&mut state, &mut provider)))
            .collect::<Vec<_>>(),
        baseline[1..]
            .iter()
            .cloned()
            .map(branch_values)
            .collect::<Vec<_>>()
    );
    state.exchange_branch(&mut left.0).unwrap();
    provider.exchange_host(&mut left.1);
    let before_usage = provider.capture_usage(&state.snapshot_source().unwrap());
    for _ in 0..2 {
        let before = probe();
        restore(&mut state, &mut provider, &saved);
        assert_eq!(probe(), before, "restore executed or rebuilt the model");
        assert_eq!(state.controller(), &prefix_controller);
        assert_eq!(
            (0..5)
                .map(|_| step(&mut state, &mut provider))
                .collect::<Vec<_>>(),
            baseline
        );
    }
    let after = provider.capture_usage(&state.snapshot_source().unwrap());
    assert!(after.captures >= before_usage.captures);
    assert!(after.encoded_bytes >= before_usage.encoded_bytes);
    restore(&mut state, &mut provider, &initial);
    assert_eq!(
        (0..3)
            .map(|_| step(&mut state, &mut provider))
            .collect::<Vec<_>>(),
        prefix
    );
    drop(right);
    assert_eq!(budget.usage().branches, 1);
    state.exchange_branch(&mut left.0).unwrap();
    provider.exchange_host(&mut left.1);
    let usage = budget.usage();
    drop(left);
    assert_eq!(
        budget.usage().branches,
        usage.branches,
        "dropping parent released active child retention"
    );
    assert_eq!(budget.usage().snapshots, usage.snapshots);
    assert_eq!(
        budget.usage().cumulative_copy_bytes,
        usage.cumulative_copy_bytes
    );
    assert_eq!(
        budget.usage().retained_bytes,
        snapshots_retained + active_child_bytes
    );
    drop(state);
    // The active host cursor is an independent owner of the same branch lease.
    assert_eq!(budget.usage().branches, 1);
    drop(provider);
    ContinuationRetirement {
        snapshots: [initial, saved],
        budget,
        cumulative_copy_bytes: usage.cumulative_copy_bytes,
    }
}

/// Final ownership checks after the caller retires the installed native state.
///
/// The logical branch lease follows its copied host and native owners, including
/// buffers still installed in the runtime after the controlled driver is dropped.
/// Completion polling alone cannot release an installed owner's lease.
#[must_use = "retire the installed runtime state, then finish the ownership checks"]
pub struct ContinuationRetirement<B, C, H>
where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController,
    H: ContinuationSnapshotProvider<B, C>,
{
    snapshots: [(TextContinuationSnapshot<B, C>, H::Host); 2],
    budget: SnapshotBudget,
    cumulative_copy_bytes: u64,
}

impl<B, C, H> ContinuationRetirement<B, C, H>
where
    B: TextSnapshotBackend
        + TextResumeBackend<
            ResumeSource = B::SavedTextComponents,
            DisplacedState = B::NativeTextState,
        >,
    C: SnapshotTokenController,
    H: ContinuationSnapshotProvider<B, C>,
{
    /// Verifies branch retirement, then releases snapshots and checks that all
    /// retained charges retire without refunding cumulative copy consumption.
    pub fn finish(self) {
        H::settle_retirement(|| self.budget.usage().branches == 0);
        assert_eq!(self.budget.usage().branches, 0);
        drop(self.snapshots);
        H::settle_retirement(|| self.budget.usage().retained_bytes == 0);
        assert_eq!(self.budget.usage().retained_bytes, 0);
        assert_eq!(self.budget.usage().snapshots, 0);
        assert_eq!(
            self.budget.usage().cumulative_copy_bytes,
            self.cumulative_copy_bytes
        );
    }
}
