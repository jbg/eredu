use super::*;
use eredu_runtime::working_memory::WorkspaceCopyLimits;

fn budget() -> SnapshotBudget {
    SnapshotBudget::new(SnapshotLimits {
        max_snapshots: 2,
        max_branches: 2,
        retained_bytes: 64_000_000,
        cumulative_copy_bytes: 512_000_000,
    })
}

#[test]
fn full_snapshot_uses_one_aggregate_capture_and_one_resume_per_installation() {
    let host = Host::default();
    let mut runtime = ModelRuntime::prepare(host.clone(), ()).unwrap();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, 0);
    for _ in 0..3 {
        step(&mut driver, &mut state);
    }
    let budget = budget();
    let forwards = host.forwards.get();
    let copies = host.copies.get();
    let saved = TextContinuationSnapshot::capture(
        &mut state.boundary(&mut driver).unwrap(),
        &budget,
        Some(4096),
    )
    .unwrap();
    assert_eq!(saved.next_prediction(), 3);
    assert_eq!(host.forwards.get(), forwards);
    assert_eq!(host.sampling_operations.get(), [1, 0, 0]);
    assert_eq!(host.paired_operations.get(), [1, 0, 0]);
    // One pending input, one sampler and one model state, with no staging copy
    // hidden behind a second aggregate operation.
    assert_eq!(host.copies.get() - copies, 3);
    let baseline: Vec<_> = (0..3).map(|_| step(&mut driver, &mut state)).collect();

    let copies = host.copies.get();
    saved
        .restore(&mut state.boundary(&mut driver).unwrap(), &budget)
        .unwrap();
    assert_eq!(host.sampling_operations.get(), [1, 0, 1]);
    assert_eq!(host.paired_operations.get(), [1, 0, 1]);
    assert_eq!(host.copies.get() - copies, 3);
    let restored: Vec<_> = (0..3).map(|_| step(&mut driver, &mut state)).collect();
    assert_eq!(restored, baseline);

    let copies = host.copies.get();
    let mut child = saved
        .fork(
            &mut state.boundary(&mut driver).unwrap(),
            &budget,
            TextBranchRequest {
                session_id: "aggregate-saved-child",
                max_predictions: 20,
                capture_limits: Some(limits()),
                intervention: None,
                host_bytes: Some(4096),
                continuation_growth_bytes: Some(4096),
            },
        )
        .unwrap();
    assert_eq!(host.sampling_operations.get(), [1, 0, 2]);
    assert_eq!(host.paired_operations.get(), [1, 0, 2]);
    assert_eq!(host.copies.get() - copies, 3);
    child.exchange(&mut driver, &mut state).unwrap();
    let forked: Vec<_> = (0..3)
        .map(|_| branch_values(step(&mut driver, &mut state)))
        .collect();
    assert_eq!(
        forked,
        baseline.into_iter().map(branch_values).collect::<Vec<_>>()
    );
}

#[test]
fn immutable_saved_sampling_remains_an_independent_source_after_live_progress() {
    let host = Host::default();
    let mut runtime = ModelRuntime::prepare(host.clone(), ()).unwrap();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, 0);
    for _ in 0..3 {
        step(&mut driver, &mut state);
    }
    let saved = {
        let mut boundary = state.boundary(&mut driver).unwrap();
        let (runtime, generation, pending) = boundary.copy_mechanism_parts();
        Host::capture_saved_sampling(
            runtime,
            Host::sampling_state(generation),
            pending,
            SamplingCopyPolicy::Unquoted,
        )
        .unwrap()
    };
    assert!(!saved.sampling.history.is_empty());
    let expected = saved.sampling.clone();
    let expected_pending = match saved.pending.as_ref().unwrap() {
        PendingTextInput::Decode(token) => token.0,
        PendingTextInput::Prefill(_) => panic!("committed source must retain its last token"),
    };
    for _ in 0..2 {
        step(&mut driver, &mut state);
    }
    let forwards = host.forwards.get();
    let mut boundary = state.boundary(&mut driver).unwrap();
    let (runtime, generation, pending) = boundary.copy_mechanism_parts();
    let live = Host::sampling_state(generation).clone();
    assert_ne!(live, expected);
    let copy = Host::copy_saved_sampling(runtime, &saved, SamplingCopyPolicy::Unquoted).unwrap();
    assert_eq!(copy.sampling, expected);
    assert_ne!(
        copy.sampling.history.as_ptr(),
        saved.sampling.history.as_ptr()
    );
    assert_eq!(Host::saved_sampling_prediction(&copy), 3);
    assert_eq!(Host::saved_input_tokens(&copy, 4), Some(4));
    assert_eq!(
        Host::estimate_saved_sampling_growth(runtime, &copy, 4).unwrap(),
        Some(16)
    );
    assert_eq!(
        Host::estimate_saved_sampling(runtime, &copy).unwrap(),
        estimate_copy(256 + 4 * expected.history.len() as u64)
    );
    let (mut prepared, prepared_input) =
        Host::prepare_saved_sampling_resume(runtime, &copy).unwrap();
    assert_eq!(prepared, expected);
    assert_ne!(prepared.history.as_ptr(), copy.sampling.history.as_ptr());
    assert!(
        matches!(prepared_input, Some(PendingTextInput::Decode(Token(id))) if id == expected_pending)
    );
    prepared.history.push(999);
    prepared.rng = 0;
    assert_eq!(copy.sampling, expected);
    assert_eq!(saved.sampling, expected);
    assert_eq!(Host::sampling_state(generation), &live);
    assert_eq!(host.forwards.get(), forwards);
    assert_eq!(host.sampling_operations.get(), [1, 1, 1]);

    let policy = SamplingCopyPolicy::Bounded(WorkspaceCopyLimits {
        capacity_bytes: u64::MAX,
        application_memory_budget_bytes: None,
        safety_reserve_bytes: 0,
    });
    let copies = host.copies.get();
    assert!(matches!(
        Host::capture_saved_sampling(runtime, Host::sampling_state(generation), pending, policy),
        Err(error) if error.kind() == io::ErrorKind::Unsupported
    ));
    assert!(matches!(
        Host::copy_saved_sampling(runtime, &copy, policy),
        Err(error) if error.kind() == io::ErrorKind::Unsupported
    ));
    assert_eq!(host.copies.get(), copies);
    assert_eq!(host.sampling_operations.get(), [1, 1, 1]);
    assert_eq!(saved.sampling, expected);
    assert_eq!(Host::sampling_state(generation), &live);
}
