//! Actual shared-driver resume from a public funded saved pair.
use super::super::super::tests::{
    AllTokens, Runtime, accounting, advance, cause, config as fixture_config, reclaim, runtime,
    settle, tokens, words,
};
use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    AdmissionRequest, AdmissionResult, ControlledTextGeneration, GenerationCancellationToken,
    InputTokenCount, TextControllerWorkspace, TextGeneration, TextGenerationDriver,
    TextGenerationInput, TextInferencePolicy, TokenFilter, TokenOutput,
};
use eredu_runtime::{
    ConfiguredTextSampler,
    execution_control::{SamplingCopyPolicy, TextSnapshotBackend},
    working_memory::{InferenceRequest, InferenceTextStepReceipt, WorkingMemoryPool},
};

pub(super) fn source_config(adaptive: bool, outputs: usize, capacity: u64) -> TextGenerationConfig {
    let mut sampling = fixture_config(Some(capacity)).sampling();
    sampling.max_new_tokens = Some(outputs);
    let config = TextGenerationConfig::new(sampling)
        .with_seed(19)
        .with_inference_policy(TextInferencePolicy {
            prefill_chunk_positions: std::num::NonZeroU64::new(1),
            managed_memory_capacity_bytes: Some(capacity),
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        });
    if adaptive {
        config.with_mirostat_v2(5.0, 0.3).unwrap()
    } else {
        config
    }
}

/// Save F6/A2, then finish the old five-output request to obtain next3 reference.
/// Original run/outputs retire; the currently installed decoder is later at F9.
pub(super) fn saved_source(
    runtime: &mut Runtime,
    adaptive: bool,
) -> (MlxSavedTextComponents, Vec<u32>) {
    let mut driver = TextGenerationDriver::new(runtime);
    let mut state = driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            source_config(adaptive, 5, u64::MAX),
            AllTokens,
        )
        .unwrap();
    let first = advance(&mut driver, &mut state);
    let second = advance(&mut driver, &mut state);
    let saved = {
        let mut boundary = driver.quiescent(&mut state).unwrap();
        let (runtime, generation, pending) = boundary.copy_mechanism_parts();
        MlxBackend::capture_saved_components(
            runtime,
            &generation.sampling,
            pending,
            SamplingCopyPolicy::Bounded(WorkspaceCopyLimits::new(u64::MAX)),
        )
        .unwrap()
    };
    let pair = saved.funded_source().unwrap();
    assert_eq!(
        (pair.sampling.frontier(), pair.sampling.next_prediction),
        (6, 2)
    );
    assert_eq!(history(pair.sampling.sampler.as_sampler()).len(), 2);
    assert!(
        history(pair.sampling.sampler.as_sampler())
            .iter()
            .any(|token| *token != 0)
    );
    let mut reference = Vec::new();
    for _ in 0..3 {
        let output = advance(&mut driver, &mut state);
        reference.push(output.token_id().unwrap());
    }
    assert!(driver.advance(&mut state).unwrap().is_none());
    drop((first, second, state, driver));
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        runtime.session().payload.active_owner_count() == 1
    });
    assert_eq!(
        runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .admission()
            .unwrap()
            .position(),
        9
    );
    (saved, reference)
}

/// Exactly the full source-inclusive reservation selected by the native route,
/// plus already charged domain occupancy. This does not perform admission.
pub(super) fn required_capacity(
    runtime: &Runtime,
    saved: &MlxSavedTextComponents,
    config: TextGenerationConfig,
) -> (u64, u64) {
    let quote = PreparedSavedTextResumeQuote::prepare(
        runtime,
        saved.funded_source().unwrap(),
        config,
        eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        },
    )
    .unwrap();
    let geometry = quote.geometry();
    let capabilities = runtime.session().capability_estimate().unwrap();
    let admission = eredu_core::apply_admission_policy_with_incremental(
        capabilities.capabilities(),
        AdmissionRequest {
            input: InputTokenCount::text(geometry.cached_positions + geometry.input_positions),
            max_output_tokens: geometry.max_output_tokens,
            batch_size: 1,
            safety_reserve_bytes: 0,
            application_memory_budget_bytes: None,
            require_complete_estimate: true,
        },
        quote.full().clone(),
        &eredu_core::WorkspaceBound::bounded(
            quote.incremental().incremental_bytes(),
            "sealed copy source credit",
        ),
        None,
    )
    .unwrap();
    let AdmissionResult::Admitted(admission) = admission else {
        panic!("actual saved quote rejected")
    };
    let required = admission.incremental_required_bytes;
    let capacity = runtime
        .backend()
        .memory_pool()
        .used_bytes()
        .unwrap()
        .checked_add(required)
        .unwrap();
    (required, capacity)
}

fn history(sampler: &ConfiguredTextSampler) -> &[u32] {
    match sampler {
        ConfiguredTextSampler::Standard(sampler) => sampler.generated_tokens(),
        ConfiguredTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}

#[derive(Debug, PartialEq)]
struct FrozenValues {
    sampler: String,
    history_pointer: usize,
    capacity: usize,
    key: Vec<u32>,
    pending: Vec<u32>,
    decoder: Vec<Vec<f32>>,
}
fn frozen(saved: &MlxSavedTextComponents) -> FrozenValues {
    let source = saved.funded_source().unwrap();
    let mut decoder = Vec::new();
    source
        .decoder
        .native
        .prepare_copy()
        .unwrap()
        .visit_operands(&mut |array| {
            decoder.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap());
        });
    FrozenValues {
        sampler: format!("{:?}", source.sampling.sampler.as_sampler()),
        history_pointer: history(source.sampling.sampler.as_sampler()).as_ptr() as usize,
        capacity: source.sampling.sampler.as_sampler().history_capacity(),
        key: words(source.sampling.arrays.key.as_ref().unwrap()),
        pending: words(source.sampling.arrays.pending.as_ref().unwrap()),
        decoder,
    }
}

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct ColdGuard;
impl ColdGuard {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
    fn assert_cold(&self) {
        assert_eq!(HOUSEKEEPING.get(), 0, "resume entered native work");
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

struct CountingController(Rc<Cell<usize>>);
impl TokenFilterController for CountingController {
    type Error = std::convert::Infallible;
    fn inference_workspace_is_run_owned(&self) -> bool {
        true
    }
    fn inference_workspace(&self, _: u64) -> Option<eredu_core::TextControllerWorkspace<'_>> {
        Some(eredu_core::TextControllerWorkspace {
            filter: (&TokenFilter::All).into(),
            additional_host_bytes: 0,
        })
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        self.0.set(self.0.get() + 1);
        Ok(TokenFilter::All)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        self.0.set(self.0.get() + 1);
        Ok(false)
    }
}

#[test]
fn actual_ordinary_and_controlled_resume_match_uninterrupted_stochastic_future() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let (saved, expected) = saved_source(&mut runtime, adaptive);
        let original = frozen(&saved);
        assert!(original.decoder.iter().flatten().any(|value| *value != 0.0));
        let (required, exact) =
            required_capacity(&runtime, &saved, source_config(adaptive, 3, u64::MAX));
        // Both successive runs share a sufficient finite ceiling. Their exact
        // cold rejection threshold is checked independently below.
        let capacity = exact.checked_add(required.checked_mul(3).unwrap()).unwrap();
        let config = source_config(adaptive, 3, capacity);
        let cancellation = GenerationCancellationToken::new();
        let ordinary = {
            let mut run = TextGeneration::resume_saved(&mut runtime, &saved, config, &cancellation)
                .unwrap()
                .unwrap();
            run.by_ref()
                .map(|token| token.unwrap().token_id().unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(ordinary, expected);
        reclaim();
        let controlled = {
            let mut run = ControlledTextGeneration::resume_saved(
                &mut runtime,
                &saved,
                config,
                AllTokens,
                &cancellation,
            )
            .unwrap()
            .unwrap();
            let mut tokens = Vec::new();
            while let Some(token) = run.next_cancellable(&cancellation) {
                tokens.push(token.unwrap().token_id());
            }
            tokens
        };
        assert_eq!(controlled, expected);
        assert_eq!(frozen(&saved), original);
        assert_eq!(pool.effective_capacity().unwrap(), capacity);
        drop((saved, runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn repeated_driver_resume_preserves_prefix_and_issues_fresh_local_receipts() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let (saved, expected) = saved_source(&mut runtime, true);
    let original = frozen(&saved);
    let prefix = saved
        .funded_source()
        .unwrap()
        .decoder
        .input
        .as_ref()
        .unwrap()
        .clone();
    let mut earlier: Option<(InferenceRequest, InferenceTextStepReceipt)> = None;
    for _ in 0..2 {
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .resume_saved(
                &saved,
                source_config(true, 3, u64::MAX),
                AllTokens,
                &GenerationCancellationToken::new(),
            )
            .unwrap()
            .unwrap();
        let request = {
            let boundary = driver.quiescent(&mut state).unwrap();
            let (runtime, generation, pending) = boundary.parts();
            let quote = generation.sampling.quote.as_ref().unwrap();
            assert_eq!(quote.local_prediction(2).unwrap(), 0);
            assert_eq!(generation.sampling.next_prediction, 2);
            assert_eq!(
                history(generation.sampling.sampler.as_sampler()),
                history(saved.funded_source().unwrap().sampling.sampler.as_sampler())
            );
            assert_eq!(
                format!("{:?}", generation.sampling.sampler.as_sampler()),
                original.sampler
            );
            assert_ne!(
                history(generation.sampling.sampler.as_sampler()).as_ptr() as usize,
                original.history_pointer
            );
            assert_eq!(
                words(generation.sampling.prng.as_ref().unwrap().as_array()),
                original.key
            );
            let Some(eredu_core::PendingTextInput::Prefill(prompt)) = pending else {
                panic!("fresh pending input must be prefill")
            };
            assert!(prompt.cache_identity.is_none());
            assert_eq!(words(prompt.parts[0].payload().value()), original.pending);
            assert!(
                runtime
                    .session()
                    .payload
                    .model
                    .erased()
                    .resident_copy_input_identity()
                    .unwrap()
                    .unwrap()
                    .same_storage(&prefix)
            );
            quote.request().clone()
        };
        if let Some((old_request, old_receipt)) = &earlier {
            assert!(old_request.validate_same_request(&request).is_err());
            assert!(matches!(
                old_receipt.validate_completed_source(&request, 3),
                Err(WorkingMemoryError::IdentityMismatch)
            ));
        }
        let mut actual = Vec::new();
        let mut last = None;
        for ordinal in 0..3 {
            let output = advance(&mut driver, &mut state);
            actual.push(output.token_id().unwrap());
            let receipt = output.step_receipt().unwrap().clone();
            assert_eq!(receipt.attempt(), ordinal);
            receipt
                .validate_completed_source(&request, ordinal + 1)
                .unwrap();
            last = Some(receipt);
        }
        assert_eq!(actual, expected);
        assert!(driver.advance(&mut state).unwrap().is_none());
        earlier = Some((request, last.unwrap()));
        drop((state, driver));
        reclaim();
        assert_eq!(frozen(&saved), original);
    }
    drop((earlier, prefix, saved, runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn zero_output_and_initial_cancellation_are_cold_and_leave_target_untouched() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let (saved, _) = saved_source(&mut runtime, false);
    let original = frozen(&saved);
    let revision = runtime
        .session()
        .payload
        .model
        .erased()
        .retained_inference_authority()
        .unwrap()
        .revision()
        .clone();
    let before = (accounting(&pool), paths::snapshot());
    let callbacks = Rc::new(Cell::new(0));
    let cold = ColdGuard::new();
    assert!(
        ControlledTextGeneration::resume_saved(
            &mut runtime,
            &saved,
            source_config(false, 0, u64::MAX),
            CountingController(Rc::clone(&callbacks)),
            &GenerationCancellationToken::new(),
        )
        .unwrap()
        .is_none()
    );
    let cancelled = GenerationCancellationToken::new();
    cancelled.cancel();
    assert!(
        TextGeneration::resume_saved(
            &mut runtime,
            &saved,
            source_config(false, 3, u64::MAX),
            &cancelled,
        )
        .unwrap()
        .is_none()
    );
    let mut driver = TextGenerationDriver::new(&mut runtime);
    assert!(
        driver
            .resume_saved(
                &saved,
                source_config(false, 3, u64::MAX),
                CountingController(Rc::clone(&callbacks)),
                &cancelled,
            )
            .unwrap()
            .is_none()
    );
    drop(driver);
    cold.assert_cold();
    drop(cold);
    assert_eq!(callbacks.get(), 0);
    assert_eq!((accounting(&pool), paths::snapshot()), before);
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
    assert_eq!(frozen(&saved), original);
    drop((saved, runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn exact_incremental_resume_capacity_accepts_and_one_short_rejects_before_work() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = runtime(&pool);
        let (saved, expected) = saved_source(&mut runtime, adaptive);
        let (required, capacity) =
            required_capacity(&runtime, &saved, source_config(adaptive, 3, u64::MAX));
        // Full diagnostics retain the old sources. The only reduction is the
        // exact union of registered decoder/key operands; pending, host and
        // future execution remain fully priced and the source is untouched.
        let quote = PreparedSavedTextResumeQuote::prepare(
            &runtime,
            saved.funded_source().unwrap(),
            source_config(adaptive, 3, u64::MAX),
            TextControllerWorkspace {
                filter: (&TokenFilter::All).into(),
                additional_host_bytes: 0,
            },
        )
        .unwrap();
        let full = quote.full().requested_state_bytes
            + quote
                .full()
                .execution_workspace
                .as_ref()
                .unwrap()
                .peak_bytes()
                .unwrap()
                .unwrap();
        let source = saved.funded_source().unwrap();
        let mut roots = std::collections::BTreeMap::new();
        let mut visit = |array: &safemlx::Array| {
            let allocation = array.try_metadata_snapshot().unwrap().allocation().unwrap();
            roots.insert(allocation.identity(), allocation.bytes() as u64);
        };
        source
            .decoder
            .native
            .prepare_copy()
            .unwrap()
            .visit_operands(&mut visit);
        if let Some(key) = &source.sampling.arrays.key {
            visit(key);
        }
        let source_bytes = roots.values().sum::<u64>();
        assert!(source_bytes > 0);
        assert_eq!(full - required, source_bytes);
        assert_eq!(quote.incremental().incremental_bytes(), required);
        drop(quote);
        let original = frozen(&saved);
        let revision = runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .revision()
            .clone();
        let before = (accounting(&pool), paths::snapshot());
        let callbacks = Rc::new(Cell::new(0));
        let cancellation = GenerationCancellationToken::new();
        {
            let cold = ColdGuard::new();
            let mut driver = TextGenerationDriver::new(&mut runtime);
            let error = driver
                .resume_saved(
                    &saved,
                    source_config(adaptive, 3, capacity - 1),
                    CountingController(Rc::clone(&callbacks)),
                    &cancellation,
                )
                .err()
                .expect("one-byte-short incremental resume must reject");
            assert!(
                matches!(cause::<WorkingMemoryError>(&error), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })
                if *required_bytes == required && *available_bytes == required - 1)
            );
            drop(driver);
            cold.assert_cold();
        }
        assert_eq!(callbacks.get(), 0);
        assert_eq!((accounting(&pool), paths::snapshot()), before);
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
        let mut driver = TextGenerationDriver::new(&mut runtime);
        let mut state = driver
            .resume_saved(
                &saved,
                source_config(adaptive, 3, capacity),
                CountingController(Rc::clone(&callbacks)),
                &cancellation,
            )
            .unwrap()
            .unwrap();
        assert_eq!(callbacks.get(), 0);
        assert_eq!(pool.peak_bytes().unwrap(), before.0.1.max(capacity));
        assert_eq!(pool.effective_capacity().unwrap(), capacity);
        let mut actual = Vec::new();
        for _ in 0..3 {
            let output = driver.advance(&mut state).unwrap().unwrap().into_output();
            driver.take_completed_delivery(&mut state).unwrap();
            actual.push(output.token_id().unwrap());
        }
        assert_eq!(actual, expected);
        assert!(callbacks.get() >= 6);
        assert!(driver.advance(&mut state).unwrap().is_none());
        drop((state, driver));
        assert_eq!(frozen(&saved), original);
        drop((saved, runtime, artifact));
        settle(&pool, 0);
    }
}

#[test]
fn independently_saved_duplicate_resumes_after_original_saved_account_retires() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = runtime(&pool);
    let (saved, expected) = saved_source(&mut runtime, true);
    let original = frozen(&saved);
    let original_bytes = saved.funded_source().unwrap().bytes();
    let source_retired = saved.funded_source().unwrap().retirement_probe();
    let duplicate = MlxBackend::copy_saved_components(
        &mut runtime,
        &saved,
        SamplingCopyPolicy::Bounded(WorkspaceCopyLimits::new(u64::MAX)),
    )
    .unwrap();
    let before = pool.used_bytes().unwrap();
    drop(saved);
    // The certified copy scope retires its source pins before returning saved
    // custody; the destination run/host holds contain no inherited source pin.
    // Shared layout/input metadata still belongs to the duplicate and target.
    settle(&pool, before - original_bytes);
    assert!(source_retired());
    let duplicated = frozen(&duplicate);
    assert_eq!(duplicated.sampler, original.sampler);
    assert_eq!(duplicated.key, original.key);
    assert_eq!(duplicated.pending, original.pending);
    assert_eq!(duplicated.decoder, original.decoder);
    let cancellation = GenerationCancellationToken::new();
    let actual = {
        let mut run = TextGeneration::resume_saved(
            &mut runtime,
            &duplicate,
            source_config(true, 3, u64::MAX),
            &cancellation,
        )
        .unwrap()
        .unwrap();
        run.by_ref()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(actual, expected);
    assert_eq!(frozen(&duplicate), duplicated);
    drop((duplicate, runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn saved_resume_logical_estimate_uses_saved_origin_and_excludes_future_growth() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut source_runtime, source_artifact) = runtime(&pool);
    // Model loading requires no active inference reservation in this domain.
    let (foreign_runtime, foreign_artifact) = runtime(&pool);
    let (saved, _) = saved_source(&mut source_runtime, true);
    // Same domain and equal numerical parameter epochs cannot identify an
    // executable. The saved F6 state also differs from the live F9 frontier.
    let values = frozen(&saved);
    let before = (accounting(&pool), paths::snapshot());
    let estimate = MlxBackend::original_saved_components_resume_estimate(
        &source_runtime,
        &saved,
        source_config(true, 3, u64::MAX),
        &eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Restore),
    )
    .expect("complete saved dense source has a logical destination estimate");
    assert!(estimate.retained_bytes > 0 && estimate.copy_bytes >= estimate.retained_bytes);
    let longer = MlxBackend::original_saved_components_resume_estimate(
        &source_runtime,
        &saved,
        source_config(true, 11, u64::MAX),
        &eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Restore),
    )
    .unwrap();
    assert_eq!(estimate.retained_bytes, longer.retained_bytes);
    assert_eq!(estimate.copy_bytes, longer.copy_bytes);
    assert!(
        MlxBackend::original_saved_components_resume_estimate(
            &foreign_runtime,
            &saved,
            source_config(true, 3, u64::MAX),
        &eredu_core::OriginalTextResumeOptions::new(eredu_core::OriginalTextResumeKind::Restore),
        )
        .is_none()
    );
    assert_eq!((accounting(&pool), paths::snapshot()), before);
    assert_eq!(frozen(&saved), values);
    drop((
        saved,
        source_runtime,
        source_artifact,
        foreign_runtime,
        foreign_artifact,
    ));
    settle(&pool, 0);
}
