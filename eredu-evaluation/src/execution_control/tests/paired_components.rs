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
fn saved_pair_copy_preserves_both_components_after_live_run_progress_and_retirement() {
    let host = Host::default();
    let mut runtime = ModelRuntime::prepare(host.clone(), ()).unwrap();
    let saved = {
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = start(&mut driver, 0);
        for _ in 0..3 {
            step(&mut driver, &mut state);
        }
        let saved = {
            let mut boundary = state.boundary(&mut driver).unwrap();
            let (runtime, generation, pending) = boundary.copy_mechanism_parts();
            Host::capture_saved_components(
                runtime,
                Host::sampling_state(generation),
                pending,
                SamplingCopyPolicy::Unquoted,
            )
            .unwrap()
        };
        for _ in 0..2 {
            step(&mut driver, &mut state);
        }
        saved
    };
    assert!(!saved.native.native.ring.is_empty());
    assert!(!saved.sampling.sampling.history.is_empty());
    let live_native = runtime.session().native.clone();
    assert_ne!(live_native, saved.native.native);
    let forwards = host.forwards.get();
    let copies = host.copies.get();
    let copy =
        Host::copy_saved_components(&mut runtime, &saved, SamplingCopyPolicy::Unquoted).unwrap();
    assert_eq!(host.copies.get() - copies, 3);
    assert_eq!(host.paired_operations.get(), [1, 1, 0]);
    assert_eq!(copy.native.native, saved.native.native);
    assert_eq!(copy.sampling.sampling, saved.sampling.sampling);
    assert_ne!(
        copy.native.native.ring.as_slices().0.as_ptr(),
        saved.native.native.ring.as_slices().0.as_ptr()
    );
    assert_ne!(
        copy.sampling.sampling.history.as_ptr(),
        saved.sampling.sampling.history.as_ptr()
    );
    let sampling = Host::saved_sampling(&copy);
    assert_eq!(Host::saved_sampling_prediction(sampling), 3);
    assert_eq!(Host::saved_input_tokens(sampling, 4), Some(4));
    assert_eq!(
        Host::estimate_saved_native_growth(&runtime, &copy, 4).unwrap(),
        Some(256)
    );
    assert_eq!(
        Host::estimate_saved_components(&runtime, &copy).unwrap(),
        estimate_copy(512 + 4 * sampling.sampling.history.len() as u64)
    );
    let (mut native, mut sampling, pending) =
        Host::prepare_saved_components_resume(&mut runtime, &copy).unwrap();
    assert_eq!(native.native, saved.native.native);
    assert_eq!(sampling, saved.sampling.sampling);
    assert!(matches!(pending, Some(PendingTextInput::Decode(_))));
    native.native.ring[0] = 999;
    sampling.history.push(999);
    sampling.rng = 0;
    assert_eq!(copy.native.native, saved.native.native);
    assert_eq!(copy.sampling.sampling, saved.sampling.sampling);
    assert_eq!(runtime.session().native, live_native);
    assert_eq!(host.forwards.get(), forwards);
    assert_eq!(host.paired_operations.get(), [1, 1, 1]);
}

#[test]
fn unpriced_paired_policy_and_foreign_pair_reject_before_any_component_copy() {
    let host = Host::default();
    let mut runtime = ModelRuntime::prepare(host.clone(), ()).unwrap();
    let policy = SamplingCopyPolicy::Bounded(WorkspaceCopyLimits {
        capacity_bytes: u64::MAX,
        application_memory_budget_bytes: None,
        safety_reserve_bytes: 0,
    });
    let saved = {
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = start(&mut driver, 0);
        for _ in 0..3 {
            step(&mut driver, &mut state);
        }
        let mut boundary = state.boundary(&mut driver).unwrap();
        let (runtime, generation, pending) = boundary.copy_mechanism_parts();
        let before = (
            host.copies.get(),
            host.paired_operations.get(),
            host.forwards.get(),
        );
        let check_input = pending.as_ref().map(|input| match input {
            PendingTextInput::Prefill(prompt) => PendingTextInput::Prefill(*prompt),
            PendingTextInput::Decode(token) => PendingTextInput::Decode(*token),
        });
        assert!(matches!(
            Host::capture_saved_components(runtime, Host::sampling_state(generation), check_input, policy),
            Err(error) if error.kind() == io::ErrorKind::Unsupported
        ));
        assert_eq!(
            (
                host.copies.get(),
                host.paired_operations.get(),
                host.forwards.get()
            ),
            before
        );
        Host::capture_saved_components(
            runtime,
            Host::sampling_state(generation),
            pending,
            SamplingCopyPolicy::Unquoted,
        )
        .unwrap()
    };
    let before = (
        host.copies.get(),
        host.paired_operations.get(),
        host.forwards.get(),
    );
    assert!(matches!(
        Host::copy_saved_components(&mut runtime, &saved, policy),
        Err(error) if error.kind() == io::ErrorKind::Unsupported
    ));
    let mut foreign = ModelRuntime::prepare(host.clone(), ()).unwrap();
    assert!(Host::validate_saved_components(&foreign, &saved).is_err());
    assert!(Host::estimate_saved_components(&foreign, &saved).is_err());
    assert!(Host::estimate_saved_native_growth(&foreign, &saved, 4).is_err());
    assert!(
        Host::copy_saved_components(&mut foreign, &saved, SamplingCopyPolicy::Unquoted).is_err()
    );
    assert!(Host::prepare_saved_components_resume(&mut foreign, &saved).is_err());
    assert_eq!(
        (
            host.copies.get(),
            host.paired_operations.get(),
            host.forwards.get()
        ),
        before
    );
    assert_eq!(foreign.session().native, NativeState::default());
    assert_eq!(
        Host::saved_sampling_prediction(Host::saved_sampling(&saved)),
        3
    );
}

#[test]
fn paired_resume_failure_after_sampler_copy_never_installs_partial_state() {
    let host = Host::default();
    let mut runtime = ModelRuntime::prepare(host.clone(), ()).unwrap();
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start(&mut driver, 0);
    for _ in 0..3 {
        step(&mut driver, &mut state);
    }
    let budget = budget();
    let saved = TextContinuationSnapshot::capture(
        &mut state.boundary(&mut driver).unwrap(),
        &budget,
        Some(4096),
    )
    .unwrap();
    let expected = step(&mut driver, &mut state);
    let before_native = driver.runtime().session().native.clone();
    let before_controller = state.controller().clone();
    let before_sampling = {
        let boundary = state.boundary(&mut driver).unwrap();
        Host::sampling_state(boundary.parts().1).clone()
    };
    let before_usage = budget.usage();
    let copies = host.copies.get();
    host.fault.set(Some("native"));
    assert!(matches!(
        saved.restore(&mut state.boundary(&mut driver).unwrap(), &budget),
        Err(TextSnapshotError::Backend(error)) if error.to_string() == "native"
    ));
    assert_eq!(host.copies.get() - copies, 3, "one failed paired resume");
    assert_eq!(host.paired_operations.get(), [1, 0, 1]);
    assert_eq!(driver.runtime().session().native, before_native);
    assert_eq!(state.controller(), &before_controller);
    let boundary = state.boundary(&mut driver).unwrap();
    assert_eq!(Host::sampling_state(boundary.parts().1), &before_sampling);
    drop(boundary);
    assert_eq!(saved.next_prediction(), 3);
    assert_eq!(budget.usage().retained_bytes, before_usage.retained_bytes);
    assert!(budget.usage().cumulative_copy_bytes > before_usage.cumulative_copy_bytes);
    host.fault.set(None);
    saved
        .restore(&mut state.boundary(&mut driver).unwrap(), &budget)
        .unwrap();
    assert_eq!(host.paired_operations.get(), [1, 0, 2]);
    assert_eq!(step(&mut driver, &mut state), expected);
}
