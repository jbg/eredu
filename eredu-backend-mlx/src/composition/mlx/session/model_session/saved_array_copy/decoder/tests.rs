use super::super::tests::{
    accounting, advance, cause, config, identity, reclaim, runtime, settle, source_state, tokens,
    words, AllTokens, State,
};
use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{TextGenerationDriver, TextGenerationInput};
use eredu_runtime::working_memory::{DecoderCopyAdmissionError, WorkingMemoryPool};

thread_local! {
    static COPIES: Cell<usize> = const { Cell::new(0) };
    static FAIL_AFTER_KEY: Cell<bool> = const { Cell::new(false) };
}

#[derive(Debug, thiserror::Error)]
#[error("injected paired decoder failure after the real key copy")]
struct InjectedPairedCopyFailure;

pub(super) fn after_key_copy() -> Result<(), Error> {
    COPIES.with(|copies| copies.set(copies.get() + 1));
    if FAIL_AFTER_KEY.with(|flag| flag.replace(false)) {
        return Err(Error::Other(Box::new(InjectedPairedCopyFailure)));
    }
    Ok(())
}

fn copies() -> usize {
    COPIES.with(Cell::get)
}

fn start_variant(
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

fn decoder_values(plan: &PreparedResidentDecoderCopy<'_>) -> Vec<Vec<f32>> {
    let mut values = Vec::new();
    plan.visit_operands(&mut |array| {
        values.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap());
    });
    values
}

fn decoder_ids(plan: &PreparedResidentDecoderCopy<'_>) -> Vec<safemlx::AllocationIdentity> {
    let mut ids = Vec::new();
    plan.visit_operands(&mut |array| ids.push(identity(array)));
    ids
}

fn history(sampler: &eredu_runtime::ConfiguredTextSampler) -> &[u32] {
    match sampler {
        MlxTextSampler::Standard(sampler) => sampler.generated_tokens(),
        MlxTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}

#[derive(Debug, PartialEq)]
struct Values {
    decoder: Vec<Vec<f32>>,
    sampler: String,
    key: Vec<u32>,
    pending: Vec<u32>,
    temperature: f32,
    next_prediction: u64,
    parameter_epoch: Option<u64>,
}

fn saved_values(saved: &CopiedTextComponents) -> Values {
    Values {
        decoder: decoder_values(&saved.decoder.native.prepare_copy().unwrap()),
        sampler: format!("{:?}", saved.sampling.sampler.as_sampler()),
        key: words(saved.sampling.arrays.key.as_ref().unwrap()),
        pending: words(saved.sampling.arrays.pending.as_ref().unwrap()),
        temperature: saved.sampling.temperature,
        next_prediction: saved.sampling.next_prediction,
        parameter_epoch: saved.sampling.parameter_epoch,
    }
}

fn shared_metadata_bytes(saved: &CopiedTextComponents) -> u64 {
    let plan = saved.decoder.native.prepare_copy().unwrap();
    let mut storage = RetainedStorage::default();
    storage
        .include_metadata(SharedHostMetadata::Layout(
            plan.shared_layout().unwrap().clone(),
        ))
        .unwrap();
    if let Some(input) = &saved.decoder.input {
        storage
            .include_metadata(SharedHostMetadata::Input(input.clone()))
            .unwrap();
    }
    storage.byte_bound().unwrap().unwrap()
}

fn assert_independent(left: &CopiedTextComponents, right: &CopiedTextComponents) {
    assert_eq!(saved_values(left), saved_values(right));
    let left_ids = decoder_ids(&left.decoder.native.prepare_copy().unwrap());
    let right_ids = decoder_ids(&right.decoder.native.prepare_copy().unwrap());
    assert!(!left_ids.is_empty());
    assert_eq!(left_ids.len(), right_ids.len());
    assert!(right_ids.iter().all(|id| !left_ids.contains(id)));
    assert_eq!(
        right_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        right_ids.len()
    );
    assert_ne!(
        history(left.sampling.sampler.as_sampler()).as_ptr(),
        history(right.sampling.sampler.as_sampler()).as_ptr()
    );
    assert_ne!(
        identity(left.sampling.arrays.key.as_ref().unwrap()),
        identity(right.sampling.arrays.key.as_ref().unwrap())
    );
    assert_ne!(
        identity(left.sampling.arrays.pending.as_ref().unwrap()),
        identity(right.sampling.arrays.pending.as_ref().unwrap())
    );
    assert!(left
        .decoder
        .native
        .prepare_copy()
        .unwrap()
        .shared_layout()
        .unwrap()
        .same_storage(
            right
                .decoder
                .native
                .prepare_copy()
                .unwrap()
                .shared_layout()
                .unwrap()
        ));
    assert!(left
        .decoder
        .input
        .as_ref()
        .unwrap()
        .same_storage(right.decoder.input.as_ref().unwrap()));
}

#[test]
fn paired_copy_admits_exact_managed_account_and_rejects_short_before_copy() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = start_variant(&mut driver, adaptive);
        let outputs = vec![
            advance(&mut driver, &mut state),
            advance(&mut driver, &mut state),
        ];
        let saved = {
            let mut boundary = driver.quiescent(&mut state).unwrap();
            let (runtime, generation, _) = boundary.copy_mechanism_parts();
            let sampling = &generation.sampling;
            let original = source_state(sampling);
            assert_eq!(original.1.len(), 2);
            assert!(sampling.sampler.as_sampler().history_capacity() > original.1.len());
            let decoder = runtime
                .session()
                .payload
                .model
                .erased()
                .prepare_resident_decoder_copy()
                .unwrap();
            let values = decoder_values(&decoder);
            let ids = decoder_ids(&decoder);
            assert!(values.len() >= 2);
            assert!(values.iter().flatten().any(|value| *value != 0.0));
            drop(decoder);
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
            let plan =
                PreparedTextComponentsCopy::prepare(runtime, sampling, outputs.last()).unwrap();
            let required = plan.required_bytes();
            assert!(
                required
                    > sampling
                        .sampler
                        .as_sampler()
                        .prepare_copy()
                        .unwrap()
                        .retained_bytes()
            );
            let error = plan
                .copy(
                    runtime,
                    WorkspaceCopyLimits {
                        application_memory_budget_bytes: Some(required - 1),
                        ..WorkspaceCopyLimits::new(u64::MAX)
                    },
                )
                .err()
                .unwrap();
            assert!(matches!(cause::<DecoderCopyAdmissionError>(&error),
                Some(DecoderCopyAdmissionError::ApplicationBudgetExceeded { required_bytes, budget_bytes })
                if *required_bytes == required && *budget_bytes == required - 1));
            let error = PreparedTextComponentsCopy::prepare(runtime, sampling, outputs.last())
                .unwrap()
                .copy(
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
            assert_eq!(source_state(sampling), original);
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
            let saved = PreparedTextComponentsCopy::prepare(runtime, sampling, outputs.last())
                .unwrap()
                .copy(runtime, WorkspaceCopyLimits::new(before.0 .0 + required))
                .unwrap();
            assert_eq!(saved.bytes(), required);
            assert_eq!(pool.used_bytes().unwrap(), before.0 .0 + required);
            assert_eq!(copies(), before.2 + 1);
            let actual = saved_values(&saved);
            assert_eq!(actual.decoder, values);
            assert_eq!(actual.key, original.0);
            assert_eq!(actual.pending, words(&outputs[1].value));
            assert_eq!(history(saved.sampling.sampler.as_sampler()), original.1);
            let saved_ids = decoder_ids(&saved.decoder.native.prepare_copy().unwrap());
            assert_eq!(saved_ids.len(), ids.len());
            assert!(saved_ids.iter().all(|id| !ids.contains(id)));
            assert_ne!(
                identity(saved.sampling.arrays.key.as_ref().unwrap()),
                identity(sampling.prng.as_ref().unwrap().as_array())
            );
            assert_ne!(
                identity(saved.sampling.arrays.pending.as_ref().unwrap()),
                identity(&outputs[1].value)
            );
            assert_eq!(source_state(sampling), original);
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
            saved
        };
        let frozen = saved_values(&saved);
        drop(advance(&mut driver, &mut state));
        assert_eq!(saved_values(&saved), frozen);
        drop((saved, outputs, state));
        drop(driver);
        drop((runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn paired_saved_copy_outlives_original_session_without_retaining_its_payload() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut destination, destination_artifact) = runtime(&pool);
    let destination_baseline = pool.used_bytes().unwrap();
    let (mut source, source_artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut source);
    let mut state = start_variant(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        PreparedTextComponentsCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    let frozen = saved_values(&saved);
    let metadata = shared_metadata_bytes(&saved);
    assert!(metadata > 0);
    drop(advance(&mut driver, &mut state));
    assert_eq!(saved_values(&saved), frozen);
    drop((outputs, state));
    drop(driver);
    // This observes active-owner retirement without retaining the live payload
    // or interfering with its checked exclusive access.
    let payload = source.session().payload.retirement_probe();
    drop((source, source_artifact));
    settle(&pool, destination_baseline + saved.bytes() + metadata);
    assert!(
        payload(),
        "saved output retained the source SessionPayload owner"
    );
    let duplicate = PreparedTextComponentsCopy::prepare_saved(&destination, &saved)
        .unwrap()
        .copy(&mut destination, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    assert_independent(&saved, &duplicate);
    drop(saved);
    settle(&pool, destination_baseline + duplicate.bytes() + metadata);
    assert_eq!(saved_values(&duplicate), frozen);
    drop((destination, destination_artifact));
    settle(&pool, duplicate.bytes() + metadata);
    assert_eq!(saved_values(&duplicate), frozen);
    drop(duplicate);
    settle(&pool, 0);
}

#[test]
fn paired_destination_mismatch_rejects_before_account_or_source_copy() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut destination, destination_artifact) = runtime(&pool);
    let (mut source, source_artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut source);
    let mut state = start_variant(&mut driver, false);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        PreparedTextComponentsCopy::prepare(runtime, &generation.sampling, outputs.last())
            .unwrap()
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .unwrap()
    };
    drop(driver);
    let before = (accounting(&pool), paths::snapshot(), copies());
    let source_frontier = source.session().payload.model.erased().state_snapshot();
    let destination_frontier = destination
        .session()
        .payload
        .model
        .erased()
        .state_snapshot();
    let values = saved_values(&saved);
    let plan = PreparedTextComponentsCopy::prepare_saved(&source, &saved).unwrap();
    assert!(source.session().authority.borrow().require_idle().is_err());
    let error = plan
        .copy(&mut destination, WorkspaceCopyLimits::new(0))
        .err()
        .unwrap();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::IdentityMismatch)
    );
    assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
    assert_eq!(
        source.session().payload.model.erased().state_snapshot(),
        source_frontier
    );
    assert_eq!(
        destination
            .session()
            .payload
            .model
            .erased()
            .state_snapshot(),
        destination_frontier
    );
    assert_eq!(saved_values(&saved), values);
    assert!(source.session().authority.borrow().require_idle().is_ok());
    assert!(destination
        .session()
        .authority
        .borrow()
        .require_idle()
        .is_ok());
    let duplicate = PreparedTextComponentsCopy::prepare_saved(&destination, &saved)
        .unwrap()
        .copy(&mut destination, WorkspaceCopyLimits::new(u64::MAX))
        .unwrap();
    assert_independent(&saved, &duplicate);
    assert_eq!(
        destination
            .session()
            .payload
            .model
            .erased()
            .state_snapshot(),
        destination_frontier
    );
    drop((saved, duplicate, outputs, state));
    drop((source, source_artifact, destination, destination_artifact));
    settle(&pool, 0);
}

#[test]
fn paired_late_failure_preserves_cause_source_state_and_complete_pin_quarantine() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start_variant(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let (required, pinned) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let original = source_state(&generation.sampling);
        let frontier = runtime.session().payload.model.erased().state_snapshot();
        let decoder = decoder_values(
            &runtime
                .session()
                .payload
                .model
                .erased()
                .prepare_resident_decoder_copy()
                .unwrap(),
        );
        let before = (accounting(&pool), copies());
        let plan =
            PreparedTextComponentsCopy::prepare(runtime, &generation.sampling, outputs.last())
                .unwrap();
        let mut storage = plan.decoder.complete_storage().unwrap();
        storage
            .include_array(generation.sampling.prng.as_ref().unwrap().as_array())
            .unwrap();
        storage.include_array(&outputs[1].value).unwrap();
        let pinned = storage.byte_bound().unwrap().unwrap();
        drop(storage);
        let required = plan.required_bytes();
        FAIL_AFTER_KEY.with(|flag| assert!(!flag.replace(true)));
        let error = plan
            .copy(runtime, WorkspaceCopyLimits::new(u64::MAX))
            .err()
            .unwrap();
        assert!(cause::<InjectedPairedCopyFailure>(&error).is_some());
        assert_eq!(copies(), before.1 + 1);
        assert_eq!(source_state(&generation.sampling), original);
        assert_eq!(
            runtime.session().payload.model.erased().state_snapshot(),
            frontier
        );
        assert_eq!(
            decoder_values(
                &runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .prepare_resident_decoder_copy()
                    .unwrap()
            ),
            decoder
        );
        runtime.session().ensure_no_submission_in_flight().unwrap();
        settle(&pool, before.0 .0 + required);
        (required, pinned)
    };
    // This is a failed read-only copy; ordinary source advancement still works.
    drop(advance(&mut driver, &mut state));
    drop((outputs, state));
    drop(driver);
    let payload = runtime.session().payload.retirement_probe();
    drop((runtime, artifact));
    settle(&pool, required + pinned);
    assert!(
        payload(),
        "quarantine retained the actual source executable"
    );
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
    reclaim();
    assert_eq!(pool.used_bytes().unwrap(), required + pinned);
}

#[test]
fn public_paired_hooks_admit_once_report_once_and_reject_ungranted_resume() {
    use eredu_core::PendingTextInput;
    use eredu_runtime::execution_control::{SamplingCopyPolicy, TextSnapshotBackend};

    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let mut state = start_variant(&mut driver, true);
    let outputs = vec![
        advance(&mut driver, &mut state),
        advance(&mut driver, &mut state),
    ];
    let (saved, duplicate) = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, _) = boundary.copy_mechanism_parts();
        let original = source_state(&generation.sampling);
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
        let required =
            PreparedTextComponentsCopy::prepare(runtime, &generation.sampling, outputs.last())
                .unwrap()
                .required_bytes();
        let before = (accounting(&pool), copies());
        // Exact application allowance leaves the global domain ceiling roomy
        // for the later simultaneously retained copy's exact capacity check.
        let saved = MlxBackend::capture_saved_components(
            runtime,
            &generation.sampling,
            Some(PendingTextInput::Decode(outputs.last().unwrap())),
            SamplingCopyPolicy::Bounded(WorkspaceCopyLimits {
                application_memory_budget_bytes: Some(required),
                ..WorkspaceCopyLimits::new(u64::MAX)
            }),
        )
        .unwrap();
        assert_eq!(pool.used_bytes().unwrap(), before.0 .0 + required);
        assert_eq!(copies(), before.1 + 1);
        let sampling = MlxBackend::saved_sampling(&saved);
        assert_eq!(MlxBackend::saved_sampling_prediction(sampling), original.2);
        assert_eq!(MlxBackend::saved_input_tokens(sampling, 3), Some(3));
        let estimate = MlxBackend::estimate_saved_components(runtime, &saved)
            .unwrap()
            .unwrap();
        // Both fields represent the one complete selected managed account;
        // decoder slot/native charge must not be added to it a second time.
        assert_eq!(estimate.retained_bytes, required);
        assert_eq!(estimate.copy_bytes, required);
        let sampling_estimate = MlxBackend::estimate_saved_sampling(runtime, sampling)
            .unwrap()
            .unwrap();
        assert_eq!(sampling_estimate.retained_bytes, required);
        let before = (accounting(&pool), paths::snapshot(), copies());
        let error = MlxBackend::copy_saved_components(
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
        let duplicate = MlxBackend::copy_saved_components(
            runtime,
            &saved,
            SamplingCopyPolicy::Bounded(WorkspaceCopyLimits::new(before.0 .0 + required)),
        )
        .unwrap();
        assert_eq!(pool.used_bytes().unwrap(), before.0 .0 + required);
        assert_eq!(copies(), before.2 + 1);
        for value in [&saved, &duplicate] {
            MlxBackend::validate_saved_components(runtime, value).unwrap();
            assert_eq!(
                MlxBackend::saved_sampling_prediction(MlxBackend::saved_sampling(value)),
                2
            );
            assert_eq!(
                MlxBackend::saved_input_tokens(MlxBackend::saved_sampling(value), 3),
                Some(3)
            );
            let estimate = MlxBackend::estimate_saved_components(runtime, value)
                .unwrap()
                .unwrap();
            assert_eq!(
                (estimate.retained_bytes, estimate.copy_bytes),
                (required, required)
            );
            let before = (accounting(&pool), paths::snapshot(), copies());
            let error = MlxBackend::prepare_saved_components_resume(runtime, value)
                .err()
                .unwrap();
            assert_eq!(
                cause::<WorkingMemoryError>(&error),
                Some(&WorkingMemoryError::UnknownBound)
            );
            let error =
                MlxBackend::copy_saved_components(runtime, value, SamplingCopyPolicy::Unquoted)
                    .err()
                    .unwrap();
            assert_eq!(
                cause::<WorkingMemoryError>(&error),
                Some(&WorkingMemoryError::UnknownBound)
            );
            assert_eq!((accounting(&pool), paths::snapshot(), copies()), before);
        }
        assert_eq!(source_state(&generation.sampling), original);
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
        (saved, duplicate)
    };
    drop(advance(&mut driver, &mut state));
    drop((outputs, state));
    drop(driver);
    drop((runtime, artifact));
    drop((saved, duplicate));
    settle(&pool, 0);
}

mod origin;
