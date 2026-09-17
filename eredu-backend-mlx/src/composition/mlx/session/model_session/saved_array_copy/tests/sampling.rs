use super::*;
use eredu_core::PendingTextInput;
use eredu_runtime::execution_control::{SamplingCopyPolicy, TextSnapshotBackend};
use eredu_runtime::working_memory::SamplerCopyLimits;

fn history(sampler: &eredu_runtime::ConfiguredTextSampler) -> &[u32] {
    match sampler {
        MlxTextSampler::Standard(sampler) => sampler.generated_tokens(),
        MlxTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}

fn start_sampling(
    driver: &mut TextGenerationDriver<'_, MlxBackend<'static>>,
    adaptive: bool,
) -> State {
    let config = config(Some(u64::MAX));
    let config = if adaptive {
        config.with_mirostat_v2(5.0, 0.3).unwrap()
    } else {
        config
    };
    driver
        .start_input(TextGenerationInput::TokenIds(tokens()), config, AllTokens)
        .unwrap()
}

fn saved_values(saved: &CopiedTextSampling) -> (String, Vec<u32>, Vec<u32>, f32, u64, Option<u64>) {
    (
        format!("{:?}", saved.sampler.as_sampler()),
        words(saved.arrays.key.as_ref().unwrap()),
        words(saved.arrays.pending.as_ref().unwrap()),
        saved.temperature,
        saved.next_prediction,
        saved.parameter_epoch,
    )
}

fn assert_independent(left: &CopiedTextSampling, right: &CopiedTextSampling) {
    assert_eq!(saved_values(left), saved_values(right));
    assert_ne!(
        history(left.sampler.as_sampler()).as_ptr(),
        history(right.sampler.as_sampler()).as_ptr()
    );
    assert_ne!(
        identity(left.arrays.key.as_ref().unwrap()),
        identity(right.arrays.key.as_ref().unwrap())
    );
    assert_ne!(
        identity(left.arrays.pending.as_ref().unwrap()),
        identity(right.arrays.pending.as_ref().unwrap())
    );
}

#[test]
fn aggregate_standard_and_mirostat_copy_admit_exact_combined_capacity() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = start_sampling(&mut driver, adaptive);
        let outputs = vec![
            advance(&mut driver, &mut state),
            advance(&mut driver, &mut state),
        ];
        let (saved, original) = {
            let mut boundary = driver.quiescent(&mut state).unwrap();
            let (runtime, generation, _) = boundary.copy_mechanism_parts();
            let sampling = &generation.sampling;
            let original = source_state(sampling);
            let original_debug = format!("{:?}", sampling.sampler.as_sampler());
            let pending = words(&outputs[1].value);
            assert_eq!(original.1.len(), 2);
            assert!(original.0.iter().any(|word| *word != 0));
            assert!(original.1.iter().any(|token| *token != 0));
            assert!(sampling.sampler.as_sampler().history_capacity() > original.1.len());
            let host = sampling
                .sampler
                .as_sampler()
                .prepare_copy()
                .unwrap()
                .retained_bytes();
            let frontier = runtime.session().payload.model.erased().state_snapshot();
            let revision = runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap()
                .revision()
                .clone();
            let before = (accounting(&pool), paths::snapshot(), copies());
            let plan = PreparedTextArrayCopy::prepare(runtime, sampling, outputs.last()).unwrap();
            let required = plan.required_sampling_bytes().unwrap();
            assert_eq!(required, host + plan.required_bytes());
            assert!(host > 0 && required > host);
            let error = plan
                .copy_sampling(
                    runtime,
                    WorkspaceCopyLimits::new(before.0 .0 + required - 1),
                )
                .err()
                .unwrap();
            assert_eq!(
                cause::<WorkingMemoryError>(&error),
                Some(&WorkingMemoryError::BudgetExceeded {
                    required_bytes: required,
                    available_bytes: required - 1,
                })
            );
            assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
            assert_eq!(
                runtime.session().payload.model.erased().state_snapshot(),
                frontier
            );
            assert_eq!(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .retained_inference_authority()
                    .unwrap()
                    .revision(),
                &revision
            );
            assert_eq!(source_state(sampling), original);
            assert_eq!(
                format!("{:?}", sampling.sampler.as_sampler()),
                original_debug
            );
            let saved = PreparedTextArrayCopy::prepare(runtime, sampling, outputs.last())
                .unwrap()
                .copy_sampling(runtime, WorkspaceCopyLimits::new(before.0 .0 + required))
                .unwrap();
            assert_eq!(copies(), before.2 + 1);
            assert_eq!(pool.used_bytes().unwrap(), before.0 .0 + required);
            assert_eq!(saved.arrays.custody.bytes(), required);
            assert_eq!(saved.sampler.bytes(), host);
            assert_eq!(format!("{:?}", saved.sampler.as_sampler()), original_debug);
            assert_eq!(history(saved.sampler.as_sampler()), original.1);
            assert_eq!(
                saved.sampler.as_sampler().history_capacity(),
                sampling.sampler.as_sampler().history_capacity()
            );
            assert_ne!(
                history(saved.sampler.as_sampler()).as_ptr(),
                history(sampling.sampler.as_sampler()).as_ptr()
            );
            assert_eq!(words(saved.arrays.key.as_ref().unwrap()), original.0);
            assert_eq!(words(saved.arrays.pending.as_ref().unwrap()), pending);
            assert_ne!(
                identity(saved.arrays.key.as_ref().unwrap()),
                identity(sampling.prng.as_ref().unwrap().as_array())
            );
            assert_ne!(
                identity(saved.arrays.pending.as_ref().unwrap()),
                identity(&outputs[1].value)
            );
            assert_eq!(
                (
                    saved.temperature,
                    saved.next_prediction,
                    saved.parameter_epoch
                ),
                (
                    sampling.temperature,
                    sampling.next_prediction,
                    sampling.parameter_epoch
                )
            );
            assert_eq!(source_state(sampling), original);
            (saved, original)
        };
        let third = advance(&mut driver, &mut state);
        {
            let boundary = driver.quiescent(&mut state).unwrap();
            let changed = source_state(&boundary.parts().1.sampling);
            assert_eq!(changed.2, 3);
            assert_ne!(changed.0, original.0);
        }
        assert_eq!(history(saved.sampler.as_sampler()), original.1);
        assert_eq!(words(saved.arrays.key.as_ref().unwrap()), original.0);
        drop((saved, third, outputs, state));
        drop(driver);
        drop((runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn saved_sampling_copies_survive_source_advancement_and_session_retirement() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    // Both ordinary model loads happen before finite funding becomes live.
    let (mut destination_runtime, destination_artifact) = runtime(&pool);
    let destination_baseline = pool.used_bytes().unwrap();
    let (mut source_runtime, source_artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut source_runtime);
    let mut state = start_sampling(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy_sampling(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    let original = saved_values(&saved);
    let third = advance(&mut driver, &mut state);
    let after_advance = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        assert_ne!(generation.sampling.next_prediction, saved.next_prediction);
        assert_ne!(source_state(&generation.sampling).0, original.1);
        PreparedTextArrayCopy::prepare_saved(runtime, &saved)
            .unwrap()
            .copy_sampling(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    assert_independent(&saved, &after_advance);
    drop((third, outputs, state));
    drop(driver);
    drop((source_runtime, source_artifact));
    settle(
        &pool,
        destination_baseline + saved.arrays.custody.bytes() + after_advance.arrays.custody.bytes(),
    );
    // A saved immutable data copy requires the same pool, not its retired live
    // source's session, parameter epoch, frontier or last-token receipt.
    let after_retirement = PreparedTextArrayCopy::prepare_saved(&destination_runtime, &saved)
        .unwrap()
        .copy_sampling(&mut destination_runtime, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    assert_independent(&saved, &after_retirement);
    assert_independent(&after_advance, &after_retirement);
    assert_eq!(saved_values(&saved), original);
    drop((saved, after_advance, after_retirement));
    settle(&pool, destination_baseline);
    drop((destination_runtime, destination_artifact));
    settle(&pool, 0);
}

#[test]
fn aggregate_sampler_and_array_owners_retire_independently_with_raw_aliases() {
    for sampler_first in [true, false] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = start_sampling(&mut driver, false);
        let outputs = vec![
            advance(&mut driver, &mut state),
            advance(&mut driver, &mut state),
        ];
        let saved = {
            let mut boundary = driver.quiescent(&mut state).unwrap();
            let (runtime, generation, _) = boundary.copy_mechanism_parts();
            PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last())
                .unwrap()
                .copy_sampling(runtime, WorkspaceCopyLimits::new(u64::MAX))
                .unwrap()
        };
        let required = saved.arrays.custody.bytes();
        let host = saved.sampler.bytes();
        let key = saved.arrays.key.as_ref().unwrap().clone();
        let key_alias = key.clone();
        let pending = saved.arrays.pending.as_ref().unwrap().clone();
        let values = (words(&key), words(&pending));
        let original_history = history(saved.sampler.as_sampler()).to_vec();
        let native = physical_bytes([&key, &key_alias, &pending]);
        let pending_bytes = physical_bytes([&pending]);
        assert!(native > pending_bytes && pending_bytes > 0);
        assert!(required >= host + native);
        drop((outputs, state));
        drop(driver);
        drop((runtime, artifact));
        settle(&pool, required);
        let CopiedTextSampling {
            sampler, arrays, ..
        } = saved;
        if sampler_first {
            drop(sampler);
            // The native custody still holds the common account's remainder.
            settle(&pool, required);
            assert_eq!((words(&key), words(&pending)), values);
            drop(arrays);
            settle(&pool, native);
        } else {
            drop(arrays);
            // Native work was certified before returning the saved copy.
            // Closing destination custody leaves exactly the live sampler host
            // hold and the independently published native aliases.
            settle(&pool, host + native);
            assert_eq!(history(sampler.as_sampler()), original_history);
            let copied_host = pool
                .copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(u64::MAX))
                .unwrap();
            assert_eq!(copied_host.bytes(), host);
            assert_eq!(history(copied_host.as_sampler()), original_history);
            assert_ne!(
                history(copied_host.as_sampler()).as_ptr(),
                history(sampler.as_sampler()).as_ptr()
            );
            drop(sampler);
            settle(&pool, native + host);
            drop(copied_host);
            settle(&pool, native);
        }
        drop(key);
        settle(&pool, native);
        drop(key_alias);
        settle(&pool, pending_bytes);
        assert_eq!(words(&pending), values.1);
        drop(pending);
        settle(&pool, 0);
    }
}

#[test]
fn aggregate_post_key_failure_keeps_native_quarantine_after_host_cleanup() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start_sampling(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let (quarantined, pinned_sources) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let original = source_state(&generation.sampling);
        let original_debug = format!("{:?}", generation.sampling.sampler.as_sampler());
        let frontier = runtime.session().payload.model.erased().state_snapshot();
        let pinned = physical_bytes([
            generation.sampling.prng.as_ref().unwrap().as_array(),
            &outputs[1].value,
        ]);
        let before = (accounting(&pool), copies());
        let plan =
            PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last()).unwrap();
        let required = plan.required_sampling_bytes().unwrap();
        assert!(required > plan.required_bytes());
        FAIL_AFTER_KEY.with(|flag| assert!(!flag.replace(true)));
        let error = plan
            .copy_sampling(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .err()
            .unwrap();
        assert!(cause::<InjectedCopyFailure>(&error).is_some());
        assert_eq!(copies(), before.1 + 1);
        assert_eq!(source_state(&generation.sampling), original);
        assert_eq!(
            format!("{:?}", generation.sampling.sampler.as_sampler()),
            original_debug
        );
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            frontier
        );
        runtime.session().ensure_no_submission_in_flight().unwrap();
        settle(&pool, before.0 .0 + required);
        (required, pinned)
    };
    drop(advance(&mut driver, &mut state));
    drop((outputs, state));
    drop(driver);
    drop((runtime, artifact));
    // Host cleanup is not certification of the failed native scope. The exact
    // source accounting pin and joint destination envelope stay quarantined;
    // neither retains source model weights nor unfinished native payloads.
    settle(&pool, quarantined + pinned_sources);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn production_saved_sampling_hooks_copy_data_without_issuing_resume_authority() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start_sampling(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let (saved, duplicate) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let sampling = &generation.sampling;
        let original = source_state(sampling);
        let required = PreparedTextArrayCopy::prepare(runtime, sampling, outputs.last())
            .unwrap()
            .required_sampling_bytes()
            .unwrap();
        let saved = MlxBackend::capture_saved_sampling(
            runtime,
            sampling,
            Some(PendingTextInput::Decode(outputs.last().unwrap())),
            SamplingCopyPolicy::Bounded(WorkspaceCopyLimits::new(u64::MAX)),
        )
        .unwrap();
        assert_eq!(MlxBackend::saved_sampling_prediction(&saved), original.2);
        assert_eq!(MlxBackend::saved_input_tokens(&saved, 3), Some(3));
        let estimate = MlxBackend::estimate_saved_sampling(runtime, &saved)
            .unwrap()
            .unwrap();
        assert_eq!(estimate.retained_bytes, required);
        assert_eq!(estimate.copy_bytes, required);
        let before = (accounting(&pool), paths::snapshot(), copies());
        let error = MlxBackend::copy_saved_sampling(
            runtime,
            &saved,
            SamplingCopyPolicy::Bounded(WorkspaceCopyLimits::new(before.0 .0 + required - 1)),
        )
        .err()
        .unwrap();
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::BudgetExceeded {
                required_bytes: required,
                available_bytes: required - 1,
            })
        );
        assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
        let duplicate = MlxBackend::copy_saved_sampling(
            runtime,
            &saved,
            SamplingCopyPolicy::Bounded(WorkspaceCopyLimits::new(before.0 .0 + required)),
        )
        .unwrap();
        assert_eq!(pool.used_bytes().unwrap(), before.0 .0 + required);
        assert_eq!(
            MlxBackend::saved_sampling_prediction(&duplicate),
            original.2
        );
        assert_eq!(MlxBackend::saved_input_tokens(&duplicate, 3), Some(3));
        let frontier = runtime.session().payload.model.erased().state_snapshot();
        let revision = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .revision()
            .clone();
        let before = (accounting(&pool), paths::snapshot(), copies());
        for saved in [&saved, &duplicate] {
            let error = MlxBackend::prepare_saved_sampling_resume(runtime, saved)
                .err()
                .unwrap();
            assert_eq!(
                cause::<WorkingMemoryError>(&error),
                Some(&WorkingMemoryError::UnknownBound)
            );
            let error =
                MlxBackend::copy_saved_sampling(runtime, saved, SamplingCopyPolicy::Unquoted)
                    .err()
                    .unwrap();
            assert_eq!(
                cause::<WorkingMemoryError>(&error),
                Some(&WorkingMemoryError::UnknownBound)
            );
        }
        assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            frontier
        );
        assert_eq!(
            runtime
                .session()
                .payload
                .model
                .erased()
                .retained_inference_authority()
                .unwrap()
                .revision(),
            &revision
        );
        assert_eq!(source_state(sampling), original);
        (saved, duplicate)
    };
    drop(advance(&mut driver, &mut state));
    drop((outputs, state));
    drop(driver);
    drop((runtime, artifact));
    drop(saved);
    drop(duplicate);
    settle(&pool, 0);
}

#[test]
fn saved_copy_preparation_binds_the_destination_lease_before_admission() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    // Load both destinations before the authentic managed run acquires funding.
    let (mut destination, destination_artifact) = runtime(&pool);
    let (mut source, source_artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut source);
    let mut state = start_sampling(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        PreparedTextArrayCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy_sampling(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    drop(driver);
    let values = saved_values(&saved);
    assert_eq!(history(saved.sampler.as_sampler()).len(), 2);
    assert!(values.1.iter().any(|word| *word != 0));
    let source_state = source.session().test_state_presence();
    let destination_state = destination.session().test_state_presence();

    for aggregate in [false, true] {
        let before = (accounting(&pool), paths::snapshot(), copies());
        let plan = PreparedTextArrayCopy::prepare_saved(&source, &saved).unwrap();
        assert!(source.session().authority.borrow().require_idle().is_err());
        assert!(destination
            .session()
            .authority
            .borrow()
            .require_idle()
            .is_ok());
        // A zero limit would fail admission. Identity must reject first, before
        // that admission or an aggregate sampler-history/native-array copy.
        let error = if aggregate {
            plan.copy_sampling(&mut destination, WorkspaceCopyLimits::new(0))
                .err()
                .unwrap()
        } else {
            plan.copy(&mut destination, WorkspaceCopyLimits::new(0))
                .err()
                .unwrap()
        };
        assert_eq!(
            cause::<WorkingMemoryError>(&error),
            Some(&WorkingMemoryError::IdentityMismatch)
        );
        assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
        assert_eq!(source.session().test_state_presence(), source_state);
        assert_eq!(
            destination.session().test_state_presence(),
            destination_state
        );
        assert_eq!(saved_values(&saved), values);
        assert!(source.session().authority.borrow().require_idle().is_ok());
        assert!(destination
            .session()
            .authority
            .borrow()
            .require_idle()
            .is_ok());

        // The immutable saved source itself remains portable within this pool.
        // A fresh preparation on B owns B's lease and copies successfully.
        let plan = PreparedTextArrayCopy::prepare_saved(&destination, &saved).unwrap();
        if aggregate {
            let duplicate = plan
                .copy_sampling(&mut destination, WorkspaceCopyLimits::new(u64::MAX))
                .unwrap();
            assert_independent(&saved, &duplicate);
            drop(duplicate);
        } else {
            let duplicate = plan
                .copy(&mut destination, WorkspaceCopyLimits::new(u64::MAX))
                .unwrap();
            assert_eq!(words(duplicate.key.as_ref().unwrap()), values.1);
            assert_eq!(words(duplicate.pending.as_ref().unwrap()), values.2);
            assert_ne!(
                identity(duplicate.key.as_ref().unwrap()),
                identity(saved.arrays.key.as_ref().unwrap())
            );
            assert_ne!(
                identity(duplicate.pending.as_ref().unwrap()),
                identity(saved.arrays.pending.as_ref().unwrap())
            );
            drop(duplicate);
        }
        assert_eq!(copies(), before.2 + 1);
        settle(&pool, before.0 .0);
        assert_eq!(source.session().test_state_presence(), source_state);
        assert_eq!(
            destination.session().test_state_presence(),
            destination_state
        );
        assert_eq!(saved_values(&saved), values);
    }

    drop((saved, outputs, state));
    drop((source, source_artifact));
    drop((destination, destination_artifact));
    settle(&pool, 0);
}
