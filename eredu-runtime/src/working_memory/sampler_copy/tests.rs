use super::*;
use crate::working_memory::{InferenceRequest, InferenceTextPreparation, WorkingMemoryReservation};
use crate::{PenaltyConfig, TokenDomain};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig, StateMemoryLayout,
    TokenFilter, WorkspaceBound, cache::LayerCachePolicy,
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
const ROOM: u64 = 1 << 20;

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
    pool: &MemoryLedger,
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
    pool: &MemoryLedger,
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
    let mut state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            physical_domains: Some(crate::working_memory::memory_fixture::host_workspace(
                pool,
                geometry,
                source_bytes,
            )),
            geometry,
            activations: bound(source_bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    state.physical_domains = Some(crate::working_memory::memory_fixture::empty_state(
        pool, geometry,
    ));
    let mut admission = Admission {
        memory_limits: crate::working_memory::memory_fixture::host_limits(capacity),
        additional_headroom: eredu_core::MemoryHeadroomDeclarations::none(),
        requested_positions: 1 + outputs as u64,
        state,
        incremental_required_bytes: Some(source_bytes),
    };
    let capacity = if capacity == SOURCE_BYTES {
        pool.reservation_requirements(&admission, None)
            .unwrap()
            .get(pool.topology().host_domain())
            .unwrap()
            .total()
            .unwrap()
    } else {
        capacity
    };
    admission.memory_limits = crate::working_memory::memory_fixture::host_limits(capacity);
    let reservation: WorkingMemoryReservation = pool
        .reserve_with_capacity(
            &execution,
            &admission,
            crate::working_memory::memory_fixture::resolved_host_limits(pool, capacity),
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let request = InferenceRequest::from(reservation);
    let config = config(outputs, adaptive);
    let preparation = request
        .prepare_text(&execution, geometry, config.clone())
        .unwrap();
    (preparation, run, config)
}

fn source(
    pool: &MemoryLedger,
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
        .claim_sampling(config.clone())
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

fn usage(pool: &MemoryLedger) -> (u64, u64, u64) {
    (
        pool.payload_used_bytes().unwrap(),
        pool.payload_peak_bytes().unwrap(),
        pool.payload_effective_capacity().unwrap(),
    )
}

fn current(pool: &MemoryLedger) -> u64 {
    let snapshot = pool.snapshot().unwrap();
    let domain = snapshot
        .domains
        .iter()
        .find(|d| d.domain == pool.topology().host_domain())
        .unwrap();
    domain.current_charge_bytes - domain.fixed_baseline.total().unwrap()
}
fn copy_increment(pool: &MemoryLedger, sampler: &BorrowedFundedSampler<'_>) -> u64 {
    pool.sampler_copy_requirements(sampler, &SamplerCopyLimits::default())
        .unwrap()
        .get(pool.topology().host_domain())
        .unwrap()
        .total()
        .unwrap()
}

#[test]
fn authenticated_sources_copy_spare_history_and_adaptive_state_at_exact_capacity() {
    for adaptive in [false, true] {
        let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
        let (mut sampler, preparation, run) = source(&pool, ROOM, 8, adaptive);
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
        let total = current(&pool) + copy_increment(&pool, &sampler.borrow_funded());
        let before = attempts();
        let copy = pool
            .copy_sampler(
                sampler.borrow_funded(),
                SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(total)),
            )
            .unwrap();
        assert_eq!(attempts(), before + 1);
        assert_eq!(copy.bytes(), bytes);
        assert_eq!(usage(&pool), (SOURCE_BYTES + bytes, total, total));
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
        assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
        drop(copy);
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
        assert_eq!(pool.payload_effective_capacity().unwrap(), ROOM);
        crate::working_memory::memory_fixture::assert_unquoted_idle(&pool);
    }
}

#[test]
fn one_byte_short_and_headroom_rejection_precede_the_copy_worker() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (mut sampler, _preparation, _run) = source(&pool, ROOM, 8, false);
    grow(&mut sampler, &[3, 11, 7, 19, 5]);
    let bytes = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    let increment = copy_increment(&pool, &sampler.borrow_funded());
    let total = current(&pool) + increment;
    let original = usage(&pool);
    let before = attempts();
    assert!(matches!(
        pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(total - 1))),
        Err(SamplerCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))) if required_bytes == increment && limit_bytes - existing_bytes == increment - 1
    ));
    assert_eq!(usage(&pool), original);
    assert_eq!(attempts(), before);
    let limits = SamplerCopyLimits {
        memory_limits: crate::working_memory::memory_fixture::host_limits(total + 6),
        additional_headroom: eredu_core::MemoryHeadroomDeclarations::new([("host".into(), 7)]),
    };
    assert!(
        matches!(pool.copy_sampler(sampler.borrow_funded(), limits.clone()), Err(SamplerCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. }))) if required_bytes == increment + 7 && limit_bytes - existing_bytes == increment + 6)
    );
    assert_eq!(usage(&pool), original);
    assert_eq!(attempts(), before);
    let copy = pool
        .copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits {
                memory_limits: crate::working_memory::memory_fixture::host_limits(total + 7),
                ..limits
            },
        )
        .unwrap();
    assert_eq!(copy.bytes(), bytes);
}

#[test]
fn source_scope_outlives_parent_and_copy_can_be_an_independent_source() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, ROOM, 8, false);
    grow(&mut sampler, &[2, 3, 5, 7, 11]);
    drop((preparation, run));
    // Original sampler plus the quoted 4-to-8 history replacement overlap.
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        std::mem::size_of::<ConfiguredTextSampler>() as u64 + 48
    );
    let first = pool
        .copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM)),
        )
        .unwrap();
    let bytes = first.bytes();
    drop(sampler);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    let second = pool
        .copy_sampler(
            first.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM)),
        )
        .unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes * 2);
    drop(first);
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    assert_eq!(history(second.as_sampler()), &[2, 3, 5, 7, 11]);
    drop(second);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn source_ceiling_and_foreign_pool_are_not_replaced_by_copy_policy() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (sampler, _preparation, _run) = source(&pool, SOURCE_BYTES, 8, false);
    let original = usage(&pool);
    let before = attempts();
    assert!(matches!(
        pool.copy_sampler(sampler.borrow_funded(), SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM))),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { limit_bytes, existing_bytes, .. })
        ))
     if limit_bytes - existing_bytes < copy_increment(&pool, &sampler.borrow_funded())));
    assert_eq!(usage(&pool), original);
    let foreign = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    assert!(matches!(
        foreign.copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM))
        ),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    assert_eq!(usage(&foreign), (0, 0, ROOM));
    assert_eq!(attempts(), before);
}

#[test]
fn construction_is_once_and_rejects_wrong_stage_or_scope_before_binding() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (preparation, run, config) = prepared_request(&pool, ROOM, 8, false);
    let prompt = preparation.claim_prompt().unwrap();
    assert!(matches!(
        prompt.construct_sampler(run.sampler_scope().unwrap()),
        Err(WorkingMemoryError::InvocationPhaseMismatch)
    ));
    let foreign = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (_foreign_preparation, foreign_run, _) = prepared_request(&foreign, ROOM, 8, false);
    let stage = preparation.claim_sampling(config.clone()).unwrap();
    assert!(matches!(
        stage.construct_sampler(foreign_run.sampler_scope().unwrap()),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(matches!(
        preparation.claim_sampling(config.clone()),
        Err(WorkingMemoryError::PreparationAlreadyStarted)
    ));
    // Rejected host-only custody certifies no native work and leaves other
    // fresh scope construction usable in both accounts.
    run.scope().unwrap().certify().unwrap();
    foreign_run.scope().unwrap().certify().unwrap();
}

#[test]
fn prepared_sampling_attempts_are_monotone_and_reject_before_callbacks() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, ROOM, 2, false);
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
        pool.payload_used_bytes().unwrap(),
        std::mem::size_of::<ConfiguredTextSampler>() as u64 + 16
    );
    drop(sampler);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn concurrent_copy_admission_reserves_before_either_result_can_escape() {
    use std::sync::{Arc, Barrier};
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (sampler, _preparation, _run) = source(&pool, ROOM, 8, false);
    let bytes = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    let total = current(&pool) + copy_increment(&pool, &sampler.borrow_funded());
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
                    SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(
                        total,
                    )),
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
    assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES + bytes);
    assert_eq!(pool.payload_peak_bytes().unwrap(), total);
    assert!(results.iter().any(|result| matches!(
        result,
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { .. })
        ))
    )));
    drop(results);
    assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES);
}

#[test]
fn checked_failures_and_poison_do_not_enter_the_copy_worker() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (sampler, _preparation, _run) = source(&pool, ROOM, 8, false);
    let before = attempts();
    let original = usage(&pool);
    assert!(matches!(
        pool.copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits {
                additional_headroom: eredu_core::MemoryHeadroomDeclarations::new([(
                    "host".into(),
                    u64::MAX
                )]),
                ..SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM))
            }
        ),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
                | WorkingMemoryError::Domain(eredu_core::MemoryDomainError::Overflow)
        ))
    ));
    assert_eq!(usage(&pool), original);
    let next = {
        let mut usage = pool.0.usage.lock().unwrap();
        std::mem::replace(&mut usage.next_funding, u64::MAX)
    };
    assert!(matches!(
        pool.copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM))
        ),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Overflow
                | WorkingMemoryError::Domain(eredu_core::MemoryDomainError::Overflow)
        ))
    ));
    assert_eq!(usage(&pool), original);
    pool.0.usage.lock().unwrap().next_funding = next;
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = pool.0.usage.lock().unwrap();
        panic!("poison accounting");
    }));
    assert!(matches!(
        pool.copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM))
        ),
        Err(SamplerCopyAdmissionError::Memory(
            WorkingMemoryError::Poisoned
        ))
    ));
    assert_eq!(attempts(), before);
}

#[test]
fn unexpected_copy_failure_quarantines_only_the_destination() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, ROOM, 8, false);
    let bytes = sampler
        .as_sampler()
        .prepare_copy()
        .unwrap()
        .retained_bytes();
    FAIL_COPY.with(|flag| flag.set(true));
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = pool.copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM)),
        );
    }));
    assert!(failed.is_err());
    assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES + bytes);
    grow(&mut sampler, &[13]);
    drop((sampler, preparation, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), bytes);
    assert!(matches!(
        pool.acquire_unquoted(),
        Err(WorkingMemoryError::ReservedWorkActive)
    ));
}

#[test]
fn quarantined_source_rejects_copy_and_sampling_before_work() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (mut sampler, _preparation, run) = source(&pool, ROOM, 8, false);
    drop(run.scope().unwrap());
    let original = usage(&pool);
    let before = attempts();
    assert!(matches!(
        pool.copy_sampler(
            sampler.borrow_funded(),
            SamplerCopyLimits::new(crate::working_memory::memory_fixture::host_limits(ROOM))
        ),
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
        let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
        let (preparation, run, config) =
            prepared_request_with_bytes(&pool, ROOM, 8, false, available);
        let original = usage(&pool);
        let result = preparation
            .claim_sampling(config.clone())
            .unwrap()
            .construct_sampler(run.sampler_scope().unwrap());
        if available < expected {
            assert!(
                matches!(result, Err(WorkingMemoryError::DomainAllowanceExceeded { required_bytes, available_bytes, .. }) if required_bytes == expected && available_bytes == available)
            );
            assert_eq!(usage(&pool), original);
            assert_eq!(
                pool.0
                    .usage
                    .lock()
                    .unwrap()
                    .funding
                    .values()
                    .map(|state| state.host_held.checked_sub(state.control_floor).unwrap())
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
        assert_eq!(pool.payload_used_bytes().unwrap(), 0);
    }
}

fn publication<K: Clone + Ord + Send + 'static>(
    pool: &MemoryLedger,
) -> crate::working_memory::PreparedStoragePublication<K> {
    crate::working_memory::StoragePublicationLayout::<K>::new(1)
        .unwrap()
        .fund(pool)
        .unwrap()
}
fn adopt<K: Clone + Ord + Send + 'static>(
    scope: &super::super::WorkingMemoryFundingScope,
    publication: crate::working_memory::PreparedStoragePublication<K>,
    entries: [(K, u64); 1],
) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError> {
    let placement = scope.pool().host_placement_handle();
    publication.adopt_storage_individually(
        scope,
        entries.map(|(key, bytes)| {
            (
                key,
                crate::working_memory::StorageAllocation::new(bytes, placement.clone()),
            )
        }),
    )
}

#[test]
fn native_adoption_cannot_spend_the_source_sampler_hold() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (mut sampler, preparation, run) = source(&pool, ROOM, 8, false);
    let held = std::mem::size_of::<ConfiguredTextSampler>() as u64 + 48;
    assert_eq!(
        pool.0
            .usage
            .lock()
            .unwrap()
            .funding
            .values()
            .map(|state| state.host_held.checked_sub(state.control_floor).unwrap())
            .sum::<u64>(),
        held
    );
    let work = run.scope().unwrap();
    let rejected_publication = publication::<u64>(&pool);
    let original = usage(&pool);
    assert!(
        matches!(adopt(&work, rejected_publication, [(17_u64, SOURCE_BYTES - held + 1)]), Err(WorkingMemoryError::DomainAllowanceExceeded { required_bytes, available_bytes, .. }) if required_bytes == SOURCE_BYTES - held + 1 && available_bytes == SOURCE_BYTES - held)
    );
    assert_eq!(pool.payload_peak_bytes().unwrap(), original.1);
    let registered = adopt(&work, publication(&pool), [(17_u64, SOURCE_BYTES - held)]).unwrap();
    // Exact aliases consume no extra funding even while only the host hold is
    // left, and their retirement returns credit to the same original account.
    let alias = adopt(&work, publication(&pool), [(17_u64, SOURCE_BYTES - held)]).unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES);
    grow(&mut sampler, &[2, 3, 5, 7, 11]);
    drop(registered);
    assert!(matches!(
        adopt(&work, publication(&pool), [(19_u64, 1)]),
        Err(WorkingMemoryError::DomainAllowanceExceeded {
            available_bytes: 0,
            ..
        })
    ));
    drop(alias);
    let second = adopt(&work, publication(&pool), [(19_u64, SOURCE_BYTES - held)]).unwrap();
    drop(second);
    drop(sampler);
    assert_eq!(
        pool.0
            .usage
            .lock()
            .unwrap()
            .funding
            .values()
            .map(|state| state.host_held.checked_sub(state.control_floor).unwrap())
            .sum::<u64>(),
        0
    );
    let final_storage = adopt(&work, publication(&pool), [(23_u64, SOURCE_BYTES)]).unwrap();
    drop((preparation, run));
    work.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES);
    drop(final_storage);
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

#[test]
fn host_scope_cleanup_does_not_certify_an_independent_native_scope() {
    let pool = crate::working_memory::memory_fixture::host_ledger(ROOM, 0).unwrap();
    let (sampler, preparation, run) = source(&pool, ROOM, 8, false);
    let native_scope = run.scope().unwrap();
    drop((sampler, preparation, run));
    assert_eq!(pool.payload_used_bytes().unwrap(), SOURCE_BYTES);
    assert_eq!(
        pool.0
            .usage
            .lock()
            .unwrap()
            .funding
            .values()
            .map(|state| state.host_held.checked_sub(state.control_floor).unwrap())
            .sum::<u64>(),
        0
    );
    native_scope.certify().unwrap();
    assert_eq!(pool.payload_used_bytes().unwrap(), 0);
}

impl crate::working_memory::WorkingMemoryFundingScope {
    /// A distinct, explicitly admitted host publisher keeps payload-only fixture
    /// allowances independent from the metadata needed to register their results.
    pub(crate) fn publish_host_storage_fixture<K: Clone + Ord + Send + 'static, const N: usize>(
        &self,
        entries: [(K, u64); N],
    ) -> Result<crate::working_memory::StorageRegistrations<K>, WorkingMemoryError> {
        let publisher =
            crate::working_memory::StoragePublicationLayout::<K>::new(N)?.fund(self.pool())?;
        let placement = self.pool().host_placement_handle();
        publisher.adopt_storage_individually(
            self,
            entries.map(|(key, bytes)| {
                (
                    key,
                    crate::working_memory::StorageAllocation::new(bytes, placement.clone()),
                )
            }),
        )
    }
}
