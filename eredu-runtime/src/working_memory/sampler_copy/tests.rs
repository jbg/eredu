use super::*;
use crate::working_memory::{InferenceRequest, InferenceTextPreparation, WorkingMemoryReservation};
use crate::{PenaltyConfig, TokenDomain};
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig,
    StateMemoryLayout, TokenFilter, WorkspaceBound,
};
use std::cell::Cell;

mod resume;

thread_local! {
    static COPY_ATTEMPTS: Cell<usize> = const { Cell::new(0) };
    static FAIL_COPY: Cell<bool> = const { Cell::new(false) };
}

pub(super) fn before_copy() {
    COPY_ATTEMPTS.with(|count| count.set(count.get() + 1));
    assert!(
        !FAIL_COPY.with(|flag| flag.replace(false)),
        "injected copy failure"
    );
}

fn attempts() -> usize {
    COPY_ATTEMPTS.with(Cell::get)
}

const SOURCE_BYTES: u64 = 1024;

fn config(outputs: usize, adaptive: bool) -> TextGenerationConfig {
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: true,
        temperature: 0.7,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(outputs),
    });
    if adaptive {
        config.with_mirostat_v2(3.7, 0.23).unwrap()
    } else {
        config
    }
}

// A scalar portable sampling backend has no native arrays. Its real boxed
// history is constructed by the one-time stage and grown by the shared sampler.
// The generous source envelope covers this fixture's complete host work; it is
// not a synthetic assertion about native model workspace or snapshot support.
fn prepared_request(
    pool: &WorkingMemoryPool,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    prepared_request_with_bytes(pool, capacity, outputs, adaptive, SOURCE_BYTES)
}

fn prepared_request_with_bytes(
    pool: &WorkingMemoryPool,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
    source_bytes: u64,
) -> (
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
    TextGenerationConfig,
) {
    let execution = InferenceExecutionIdentity::default();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: outputs as u64,
        prefill_chunk_positions: 1,
        output: OutputDemand::LastPosition,
    };
    let layout = StateMemoryLayout::new(
        LayerSchedule::new(1, vec![LayerCachePolicy::NoState]).unwrap(),
        vec![0],
        1,
        1,
        EstimationCompleteness::Complete,
    )
    .unwrap();
    let state = eredu_core::estimate_runtime_state(
        &layout,
        InputTokenCount::text(1),
        outputs as u64,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "portable scalar sampler host envelope");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(source_bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let reservation: WorkingMemoryReservation = pool
        .reserve_with_capacity(
            &execution,
            &Admission {
                requested_positions: 1 + outputs as u64,
                state,
                incremental_required_bytes: source_bytes,
                available_memory_bytes: None,
            },
            capacity,
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let request = InferenceRequest::from(reservation);
    let config = config(outputs, adaptive);
    let preparation = request.prepare_text(&execution, geometry, config).unwrap();
    (preparation, run, config)
}

fn source(
    pool: &WorkingMemoryPool,
    capacity: u64,
    outputs: usize,
    adaptive: bool,
) -> (
    RunOwnedTextSampler,
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
) {
    let (preparation, run, config) = prepared_request(pool, capacity, outputs, adaptive);
    let (sampler, completion) = preparation
        .claim_sampling(config)
        .unwrap()
        .construct_sampler(run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    (sampler, preparation, run)
}

#[derive(Default)]
struct Context {
    callbacks: Cell<usize>,
}

struct Scalar;
impl SamplingBackend for Scalar {
    type Logits = u32;
    type Token = u32;
    type RandomState = ();
    type Context = Context;
    type Error = String;

    fn error(message: String) -> String {
        message
    }
    fn validate_token(token: &u32, _: TokenDomain, _: &Context) -> Result<u32, String> {
        Ok(*token)
    }
    fn scale_temperature(logits: &u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_penalties(
        logits: &u32,
        _: &[u32],
        _: PenaltyConfig,
        context: &Context,
    ) -> Result<u32, String> {
        context.callbacks.set(context.callbacks.get() + 1);
        Ok(*logits)
    }
    fn apply_top_k(logits: u32, _: i32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_top_p(logits: u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_min_p(logits: u32, _: f32, _: &Context) -> Result<u32, String> {
        Ok(logits)
    }
    fn apply_token_filter(logits: &u32, _: &TokenFilter, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn apply_mirostat(
        logits: &u32,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        context: &Context,
    ) -> Result<u32, String> {
        context.callbacks.set(context.callbacks.get() + 1);
        Ok(*logits)
    }
    fn sample_raw(logits: &u32, _: f32, _: Option<&mut ()>, _: &Context) -> Result<u32, String> {
        Ok(*logits)
    }
    fn sample_processed(
        logits: &u32,
        _: f32,
        _: Option<&mut ()>,
        _: &Context,
    ) -> Result<u32, String> {
        Ok(*logits)
    }
    fn token_id(token: &u32, _: &Context) -> Result<u32, String> {
        Ok(*token)
    }
    fn token_probability(_: &u32, _: u32, _: &Context) -> Result<f32, String> {
        Ok(0.125)
    }
}

fn grow(sampler: &mut RunOwnedTextSampler, tokens: &[u32]) {
    let context = Context::default();
    for token in tokens {
        assert_eq!(
            sampler
                .prepare_sample()
                .unwrap()
                .sample::<Scalar>(token, 0.7, None, &context)
                .unwrap(),
            *token
        );
    }
    assert_eq!(context.callbacks.get(), tokens.len());
}

fn history(sampler: &ConfiguredTextSampler) -> &[u32] {
    match sampler {
        ConfiguredTextSampler::Standard(sampler) => sampler.generated_tokens(),
        ConfiguredTextSampler::MirostatV2(sampler) => sampler.generated_tokens(),
    }
}

fn usage(pool: &WorkingMemoryPool) -> (u64, u64, u64) {
    (
        pool.used_bytes().unwrap(),
        pool.peak_bytes().unwrap(),
        pool.effective_capacity().unwrap(),
    )
}

#[test]
fn authenticated_sources_copy_spare_history_and_adaptive_state_at_exact_capacity() {
    for adaptive in [false, true] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (mut sampler, preparation, run) = source(&pool, 8192, 8, adaptive);
        grow(&mut sampler, &[3, 11, 7, 19, 5]);
        let plan = sampler.as_sampler().prepare_copy().unwrap();
        assert_eq!(
            (
                plan.history_len(),
                plan.history_capacity(),
                plan.history_bytes()
            ),
            (5, 8, 32)
        );
        let bytes = plan.retained_bytes();
        drop(plan);
        let before = attempts();
        let copy = pool
            .copy_sampler(
                sampler.borrow_funded(),
                SamplerCopyLimits::new(SOURCE_BYTES + bytes),
            )
            .unwrap();
        assert_eq!(attempts(), before + 1);
        assert_eq!(copy.bytes(), bytes);
        assert_eq!(
            usage(&pool),
            (
                SOURCE_BYTES + bytes,
                SOURCE_BYTES + bytes,
                SOURCE_BYTES + bytes
            )
        );
        assert_eq!(history(copy.as_sampler()), &[3, 11, 7, 19, 5]);
        assert_ne!(
            history(sampler.as_sampler()).as_ptr(),
            history(copy.as_sampler()).as_ptr()
        );
        if let (
            ConfiguredTextSampler::MirostatV2(original),
            ConfiguredTextSampler::MirostatV2(copied),
        ) = (sampler.as_sampler(), copy.as_sampler())
        {
            assert_ne!(original.mu(), 2.0 * original.tau());
            assert_eq!(
                (copied.mu(), copied.tau(), copied.eta()),
                (original.mu(), original.tau(), original.eta())
            );
        }
        grow(&mut sampler, &[29]);
        assert_eq!(history(copy.as_sampler()), &[3, 11, 7, 19, 5]);
        drop((preparation, run, sampler));
        assert_eq!(pool.used_bytes().unwrap(), bytes);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
        drop(copy);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        assert_eq!(pool.effective_capacity().unwrap(), 8192);
        drop(pool.acquire_unquoted().unwrap());
    }
}

#[test]
fn one_byte_short_and_application_rejection_precede_the_copy_worker() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (mut sampler, _preparation, _run) = source(&pool, 8192, 8, false);
    grow(&mut sampler, &[3, 11, 7, 19, 5]);
    let bytes = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    let original = usage(&pool);
    let before = attempts();
    assert!(matches!(
        pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(SOURCE_BYTES + bytes - 1)),
        Err(SamplerCopyAdmissionError::Memory(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes })) if required_bytes == bytes && available_bytes == bytes - 1
    ));
    assert_eq!(usage(&pool), original);
    assert_eq!(attempts(), before);
    let limits = SamplerCopyLimits {
        capacity_bytes: 8192,
        application_memory_budget_bytes: Some(bytes + 6),
        safety_reserve_bytes: 7,
    };
    assert!(
        matches!(pool.copy_sampler(sampler.borrow_funded(), limits), Err(SamplerCopyAdmissionError::ApplicationBudgetExceeded { required_bytes, budget_bytes }) if required_bytes == bytes + 7 && budget_bytes == bytes + 6)
    );
    assert_eq!(usage(&pool), original);
    assert_eq!(attempts(), before);
    let copy = pool
        .copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits {
                application_memory_budget_bytes: Some(bytes + 7),
                ..limits
            },
        )
        .unwrap();
    assert_eq!(copy.bytes(), bytes + 7);
}

#[test]
fn source_scope_outlives_parent_and_copy_can_be_an_independent_source() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, 8192, 8, false);
    grow(&mut sampler, &[2, 3, 5, 7, 11]);
    drop((preparation, run));
    // Original sampler plus the quoted 4-to-8 history replacement overlap.
    assert_eq!(
        pool.used_bytes().unwrap(),
        std::mem::size_of::<ConfiguredTextSampler>() as u64 + 48
    );
    let first = pool
        .copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192))
        .unwrap();
    let bytes = first.bytes();
    drop(sampler);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let second = pool
        .copy_sampler(first.borrow_funded(), SamplerCopyLimits::new(8192))
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), bytes * 2);
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert_eq!(history(second.as_sampler()), &[2, 3, 5, 7, 11]);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_ceiling_and_foreign_pool_are_not_replaced_by_copy_policy() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, _preparation, _run) = source(&pool, SOURCE_BYTES, 8, false);
    let original = usage(&pool);
    let before = attempts();
    assert!(matches!(
        pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192)),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::BudgetExceeded {
                available_bytes: 0,
                ..
            }
        ))
    ));
    assert_eq!(usage(&pool), original);
    let foreign = WorkingMemoryPool::new(8192, 0).unwrap();
    assert!(matches!(
        foreign.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192)),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(usage(&foreign), (0, 0, 8192));
    assert_eq!(attempts(), before);
}

#[test]
fn construction_is_once_and_rejects_wrong_stage_or_scope_before_binding() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (preparation, run, config) = prepared_request(&pool, 8192, 8, false);
    let prompt = preparation.claim_prompt().unwrap();
    assert!(matches!(
        prompt.construct_sampler(run.sampler_scope().unwrap()),
        Err(WorkingMemoryError::InvocationPhaseMismatch)
    ));
    let foreign = WorkingMemoryPool::new(8192, 0).unwrap();
    let (_foreign_preparation, foreign_run, _) = prepared_request(&foreign, 8192, 8, false);
    let stage = preparation.claim_sampling(config).unwrap();
    assert!(matches!(
        stage.construct_sampler(foreign_run.sampler_scope().unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        preparation.claim_sampling(config),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    // Rejected host-only custody certifies no native work and leaves other
    // fresh scope construction usable in both accounts.
    run.scope().unwrap().certify().unwrap();
    foreign_run.scope().unwrap().certify().unwrap();
}

#[test]
fn prepared_sampling_attempts_are_monotone_and_reject_before_callbacks() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, 8192, 2, false);
    let context = Context::default();
    drop(sampler.prepare_sample().unwrap());
    sampler
        .prepare_sample()
        .unwrap()
        .sample::<Scalar>(&41, 0.7, None, &context)
        .unwrap();
    assert_eq!(context.callbacks.get(), 1);
    assert!(matches!(
        sampler.prepare_sample(),
        Err(WorkingMemoryError::TextOutputAllowanceExceeded {
            issued: 2,
            limit: 2
        })
    ));
    assert_eq!(context.callbacks.get(), 1);
    assert_eq!(history(sampler.as_sampler()), &[41]);
    drop((preparation, run));
    // Two samples use the original four-slot host history bound.
    assert_eq!(
        pool.used_bytes().unwrap(),
        std::mem::size_of::<ConfiguredTextSampler>() as u64 + 16
    );
    drop(sampler);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn concurrent_copy_admission_reserves_before_either_result_can_escape() {
    use std::sync::{Arc, Barrier};
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, _preparation, _run) = source(&pool, 8192, 8, false);
    let bytes = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    let sampler = Arc::new(sampler);
    let barrier = Arc::new(Barrier::new(2));
    let handles: Vec<_> = (0..2)
        .map(|_| {
            let pool = pool.clone();
            let source = Arc::clone(&sampler);
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let result = pool.copy_sampler(
                    source.borrow_funded(),
                    SamplerCopyLimits::new(SOURCE_BYTES + bytes),
                );
                barrier.wait();
                result
            })
        })
        .collect();
    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + bytes);
    assert_eq!(pool.peak_bytes().unwrap(), SOURCE_BYTES + bytes);
    assert!(results.iter().any(|result| matches!(
        result,
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::BudgetExceeded { .. }
        ))
    )));
    drop(results);
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES);
}

#[test]
fn checked_failures_and_poison_do_not_enter_the_copy_worker() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, _preparation, _run) = source(&pool, 8192, 8, false);
    let before = attempts();
    let original = usage(&pool);
    assert!(matches!(
        pool.copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits {
                safety_reserve_bytes: u64::MAX,
                ..SamplerCopyLimits::new(8192)
            }
        ),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
        ))
    ));
    assert_eq!(usage(&pool), original);
    let next = {
        let mut usage = pool.0.usage.lock().unwrap();
        std::mem::replace(&mut usage.next_funding, u64::MAX)
    };
    assert!(matches!(
        pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192)),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
        ))
    ));
    assert_eq!(usage(&pool), original);
    pool.0.usage.lock().unwrap().next_funding = next;
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison accounting");
    }));
    assert!(matches!(
        pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192)),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Poisoned
        ))
    ));
    assert_eq!(attempts(), before);
}

#[test]
fn unexpected_copy_failure_quarantines_only_the_destination() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, 8192, 8, false);
    let bytes = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    FAIL_COPY.with(|flag| flag.set(true));
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192));
    }));
    assert!(failed.is_err());
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES + bytes);
    grow(&mut sampler, &[13]);
    drop((sampler, preparation, run));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn quarantined_source_rejects_copy_and_sampling_before_work() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (mut sampler, _preparation, run) = source(&pool, 8192, 8, false);
    drop(run.scope().unwrap());
    let original = usage(&pool);
    let before = attempts();
    assert!(matches!(
        pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(8192)),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::ExecutionFenced
        ))
    ));
    assert!(matches!(
        sampler.prepare_sample(),
        Err(WorkingMemoryError::ExecutionFenced)
    ));
    assert_eq!(usage(&pool), original);
    assert_eq!(attempts(), before);
}

#[test]
fn bootstrap_requires_the_complete_canonical_host_envelope() {
    let expected = std::mem::size_of::<ConfiguredTextSampler>() as u64 + (4 + 8) * 4;
    for available in [0, expected - 1, expected] {
        let pool = WorkingMemoryPool::new(8192, 0).unwrap();
        let (preparation, run, config) =
            prepared_request_with_bytes(&pool, 8192, 8, false, available);
        let original = usage(&pool);
        let result = preparation
            .claim_sampling(config)
            .unwrap()
            .construct_sampler(run.sampler_scope().unwrap());
        if available < expected {
            assert!(
                matches!(result, Err(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if required_bytes == expected && available_bytes == available)
            );
            assert_eq!(usage(&pool), original);
            assert_eq!(
                pool.0
                    .usage
                    .lock()
                    .unwrap()
                    .funding
                    .values()
                    .map(|state| state.host_held)
                    .sum::<u64>(),
                0
            );
            run.scope().unwrap().certify().unwrap();
        } else {
            let (mut sampler, completion) = result.unwrap();
            completion.finish().unwrap();
            grow(&mut sampler, &[2, 3, 5, 7, 11, 13, 17, 19]);
            assert_eq!(sampler.as_sampler().history_capacity(), 8);
            assert!(matches!(
                sampler.prepare_sample(),
                Err(WorkingMemoryError::TextOutputAllowanceExceeded { .. })
            ));
            drop(sampler);
        }
        drop((preparation, run));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn native_adoption_cannot_spend_the_source_sampler_hold() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, 8192, 8, false);
    let held = std::mem::size_of::<ConfiguredTextSampler>() as u64 + 48;
    assert_eq!(
        pool.0
            .usage
            .lock()
            .unwrap()
            .funding
            .values()
            .map(|state| state.host_held)
            .sum::<u64>(),
        held
    );
    let work = run.scope().unwrap();
    let original = usage(&pool);
    assert!(
        matches!(work.adopt_storage_individually([(17_u64, SOURCE_BYTES - held + 1)]), Err(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if required_bytes == SOURCE_BYTES - held + 1 && available_bytes == SOURCE_BYTES - held)
    );
    assert_eq!(usage(&pool), original);
    let registered = work
        .adopt_storage_individually([(17_u64, SOURCE_BYTES - held)])
        .unwrap();
    // Exact aliases consume no extra funding even while only the host hold is
    // left, and their retirement returns credit to the same original account.
    let alias = work
        .adopt_storage_individually([(17_u64, SOURCE_BYTES - held)])
        .unwrap();
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES);
    grow(&mut sampler, &[2, 3, 5, 7, 11]);
    drop(registered);
    assert!(matches!(
        work.adopt_storage_individually([(19_u64, 1)]),
        Err(WorkingMemoryError::BudgetExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(alias);
    let second = work
        .adopt_storage_individually([(19_u64, SOURCE_BYTES - held)])
        .unwrap();
    drop(second);
    drop(sampler);
    assert_eq!(
        pool.0
            .usage
            .lock()
            .unwrap()
            .funding
            .values()
            .map(|state| state.host_held)
            .sum::<u64>(),
        0
    );
    let final_storage = work
        .adopt_storage_individually([(23_u64, SOURCE_BYTES)])
        .unwrap();
    drop((preparation, run));
    work.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES);
    drop(final_storage);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn host_scope_cleanup_does_not_certify_an_independent_native_scope() {
    let pool = WorkingMemoryPool::new(8192, 0).unwrap();
    let (sampler, preparation, run) = source(&pool, 8192, 8, false);
    let native_scope = run.scope().unwrap();
    drop((sampler, preparation, run));
    assert_eq!(pool.used_bytes().unwrap(), SOURCE_BYTES);
    assert_eq!(
        pool.0
            .usage
            .lock()
            .unwrap()
            .funding
            .values()
            .map(|state| state.host_held)
            .sum::<u64>(),
        0
    );
    native_scope.certify().unwrap();
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
