//! Explicit successor capacity changes preserve completed native storage charges.

use super::disk_layerwise_tests::{
    cause, config, evidence, live_bytes, reclaim, report, token_ids, tokens, Controller,
};
use super::*;
use crate::memory_fixture::LedgerFixture as _;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, InferenceGeometry, MemoryLimit, OutputDemand, TextGeneration,
    TextGenerationDriver, TextGenerationInput, TokenFilterController,
};
use eredu_runtime::working_memory::{MemoryDomainSnapshot, MemoryLedger, WorkingMemoryError};

fn host(pool: &MemoryLedger) -> MemoryDomainSnapshot {
    pool.snapshot()
        .unwrap()
        .domains
        .into_iter()
        .find(|domain| domain.domain == pool.topology().host_domain())
        .unwrap()
}
fn limit(pool: &MemoryLedger) -> u64 {
    match host(pool).effective_limit {
        MemoryLimit::Finite(bytes) => bytes,
        MemoryLimit::Unlimited => panic!("a live request retains its finite host ceiling"),
    }
}
#[track_caller]
fn settle(pool: &MemoryLedger, expected: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    loop {
        reclaim();
        let actual = pool.fixture_host_charge().unwrap();
        if actual == expected && pool.unquoted_owner_count().unwrap() == 0 {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "terminal charge did not retire: actual={actual}, expected={expected}, snapshot={:?}",
            pool.snapshot().unwrap(),
        );
        std::thread::yield_now();
    }
}
fn exact_capacity(
    runtime: &ModelRuntime<MlxBackend<'_>>,
    pool: &MemoryLedger,
    ids: &Vec<u32>,
    temperature: f32,
    chunk: u64,
) -> u64 {
    let before = pool.fixture_host_current().unwrap();
    let preparation = MlxBackend::admit_text_preparation(
        runtime,
        &evidence(ids),
        config(temperature, chunk, u64::MAX),
        &Controller::default(),
    )
    .unwrap();
    let capacity = pool.fixture_host_current().unwrap();
    assert!(capacity > before);
    drop(preparation);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.fixture_host_current().unwrap() == before
    });
    capacity
}
fn outputs(
    runtime: &mut ModelRuntime<MlxBackend<'_>>,
    ids: Vec<u32>,
    generation: TextGenerationConfig,
    controlled: bool,
) -> Vec<MlxTextToken> {
    let controller = Controller::default();
    let outputs = if controlled {
        let outputs = ControlledTextGeneration::from_input(
            runtime,
            TextGenerationInput::TokenIds(ids),
            generation,
            controller.clone(),
        )
        .unwrap()
        .map(|token| token.unwrap().into_output())
        .collect::<Vec<_>>();
        assert_eq!(controller.0.get(), (4, 4));
        outputs
    } else {
        TextGeneration::new(runtime, ids, generation)
            .unwrap()
            .map(Result::unwrap)
            .collect::<Vec<_>>()
    };
    assert_eq!(outputs.len(), 4);
    outputs
}

fn retained_token_request(token: &MlxTextToken) -> eredu_runtime::working_memory::InferenceRequest {
    token
        .owner
        .inference_retention
        .borrow()
        .requests()
        .find(|request| request.geometry().cached_positions == 0)
        .expect("first run's original request remains attached to its output")
        .clone()
}
fn retained_token_bytes(value: &Array) -> u64 {
    let backing = value.allocation_info().unwrap().unwrap();
    u64::try_from(backing.bytes())
        .unwrap()
        .checked_add(u64::try_from(backing.host_control_bytes()).unwrap())
        .unwrap()
}

#[test]
fn completed_disk_run_hands_off_larger_capacity_with_old_outputs_in_both_drivers() {
    if !crate::tests::support::native_process::enter("capacity-handoff") {
        return;
    }
    let fixture = text_quote::PreparedResidencyFixture::new();
    let pool = fixture.pool.clone();
    for controlled in [false, true] {
        let (mut runtime, artifact, source_baseline) = fixture.load(2, None);
        let capacity_a = exact_capacity(&runtime, &pool, &tokens(), 0.0, 2);
        let output_a = outputs(
            &mut runtime,
            tokens(),
            config(0.0, 2, capacity_a),
            controlled,
        );
        let ids_a = token_ids(&output_a);
        let request_a = retained_token_request(&output_a[0]);
        let admission_a = request_a.memory_reservation().admission().clone();
        assert!(
            matches!(
                request_a.memory_reservation().clone().into_funding(),
                Err(WorkingMemoryError::IdentityMismatch)
            ),
            "historical request metadata cannot recreate the unique funding owner"
        );
        let owner_a = output_a[0].owner.clone();
        let alias = output_a[0]
            .value
            .as_strided(&[1][..], &[1][..], 0, fixture.stream())
            .unwrap();
        alias.evaluated().unwrap();
        let allocation_a = alias.allocation_info().unwrap().unwrap();
        assert_eq!(
            Some(allocation_a),
            output_a[0].value.allocation_info().unwrap()
        );
        assert!(owner_a.resources_releasable());
        assert!(owner_a.direct_route.borrow().is_none());

        reclaim();
        assert_eq!(limit(&pool), capacity_a);
        assert!(pool.fixture_host_current().unwrap() <= capacity_a);

        // A's final sampled token is still pending decode. The new request
        // explicitly supplies a larger finite ceiling; no MAX admission probe
        // may change the predecessor's policy while measuring this request.
        let continuation = vec![ids_a[3], 6, 7];
        let capacity_b = capacity_a.checked_mul(2).unwrap();
        let before = report(&runtime);
        let reads_before = text_quote::PreparedResidencyFixture::actual_disk_read_bytes(&runtime);
        let paths_before = paths::snapshot();
        let used = pool.fixture_host_current().unwrap();
        let peak = host(&pool).historical_peak_bytes;
        let controller = Controller::default();
        let (diagnostic, width) = text_quote::quote(
            runtime.session(),
            InferenceGeometry {
                batch_size: 1,
                cached_positions: 8,
                input_positions: 3,
                max_output_tokens: 4,
                prefill_chunk_positions: 2,
                output: OutputDemand::LastPosition,
            },
            (continuation.capacity() * std::mem::size_of::<u32>()) as u64,
            config(0.0, 2, capacity_b),
            controller.inference_workspace(4).unwrap(),
        )
        .unwrap();
        assert_eq!(width, 64);
        assert!(diagnostic.execution_workspace.is_some());
        assert_eq!(report(&runtime), before);
        assert_eq!(paths::snapshot(), paths_before);
        assert_eq!(pool.fixture_host_current().unwrap(), used);
        assert_eq!(host(&pool).historical_peak_bytes, peak);
        assert_eq!(limit(&pool), capacity_a);

        let output_b = outputs(
            &mut runtime,
            continuation,
            config(0.0, 2, capacity_b),
            controlled,
        );
        let ids_b = token_ids(&output_b);
        fixture.assert_disk_read_progress(&runtime, 2, Some(reads_before));
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(14)
            .unwrap();
        assert!(
            text_quote::PreparedResidencyFixture::actual_disk_read_bytes(&runtime) > reads_before
        );
        assert_eq!(limit(&pool), capacity_b);
        assert!(pool.fixture_host_current().unwrap() <= capacity_b);
        assert_eq!(token_ids(&output_a), ids_a);
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation_a));
        assert_eq!(request_a.memory_reservation().admission(), &admission_a);
        assert!(owner_a.resources_releasable());
        assert!(owner_a.direct_route.borrow().is_none());

        reclaim();
        let token_bytes = retained_token_bytes(&alias);
        assert!(token_bytes > 0);
        drop((output_b, runtime, artifact));
        assert_eq!(limit(&pool), capacity_b);
        drop((output_a, owner_a, request_a));
        settle(&pool, source_baseline.checked_add(token_bytes).unwrap());
        // Only A's physical alias remains. B's retirement cannot remove the
        // adopted ceiling or charge before this last old allocation retires.
        assert_eq!(limit(&pool), capacity_b);
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation_a));
        assert_eq!(alias.evaluated().unwrap().item::<u32>(), ids_a[0]);
        assert_eq!(
            pool.acquire_unquoted().unwrap_err(),
            WorkingMemoryError::ReservedWorkActive
        );
        drop(alias);
        settle(&pool, source_baseline);
        assert_eq!(host(&pool).effective_limit, MemoryLimit::Unlimited);
        let (mut resident, resident_artifact, reference_baseline) = fixture.load(0, None);
        let mut full = tokens();
        full.extend_from_slice(&ids_a);
        full.extend([6, 7]);
        let reference_capacity = exact_capacity(&resident, &pool, &full, 0.0, 2);
        let reference = outputs(
            &mut resident,
            full,
            config(0.0, 2, reference_capacity),
            controlled,
        );
        assert_eq!(ids_b, token_ids(&reference));
        drop((reference, resident, resident_artifact));
        settle(&pool, reference_baseline);
    }
}

#[test]
fn successor_one_byte_short_rejects_without_changing_predecessor_ceiling() {
    if !crate::tests::support::native_process::enter("capacity-handoff") {
        return;
    }
    let fixture = text_quote::PreparedResidencyFixture::new();
    let pool = fixture.pool.clone();
    let (mut runtime, artifact, source_baseline) = fixture.load(2, Some(128));
    let capacity_a = exact_capacity(&runtime, &pool, &tokens(), 0.7, 1);
    let output_a = outputs(&mut runtime, tokens(), config(0.7, 1, capacity_a), true);
    let ids_a = token_ids(&output_a);

    reclaim();
    // A longer continuation requires additional simultaneous state and work;
    // a shorter continuation can reuse enough storage to fit A's ceiling.
    let mut continuation = vec![ids_a[3]];
    continuation.extend((0..63).map(|position| 6 + position % 58));
    let baseline = pool.fixture_host_current().unwrap();
    let before = report(&runtime);
    let paths_before = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let controller = Controller::default();
    let probe_capacity = capacity_a.checked_add(1).unwrap();
    let error = MlxBackend::admit_text_preparation(
        &runtime,
        &evidence(&continuation),
        config(0.7, 1, probe_capacity),
        &controller,
    )
    .unwrap_err();
    let Some(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded {
        requested_bytes: required_bytes,
        limit_bytes,
        existing_bytes,
        ..
    })) = cause::<WorkingMemoryError>(&error)
    else {
        panic!(
            "a valid but insufficient successor capacity must preserve its budget error: {error:?}"
        );
    };
    let required_bytes = *required_bytes;
    assert_eq!(*limit_bytes, probe_capacity);
    assert!(*existing_bytes >= baseline);
    let exact_b = existing_bytes.checked_add(required_bytes).unwrap();
    assert!(exact_b > probe_capacity);
    drop(error);

    let fault = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "rejected successor must not start sampler".into(),
    ));
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(continuation.clone()),
        config(0.7, 1, exact_b - 1),
        controller.clone(),
    )
    .err()
    .expect("one-byte-short minimum-chunk successor is rejected");
    assert_eq!(
        cause::<WorkingMemoryError>(&error)
            .and_then(crate::tests::support::memory_error::host_budget_numbers),
        Some((required_bytes, required_bytes - 1))
    );
    drop(error);
    assert_eq!(limit(&pool), capacity_a);
    assert_eq!(pool.fixture_host_current().unwrap(), baseline);
    assert_eq!(report(&runtime), before);
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.0.get(), (0, 0));
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    drop(fault);
    let output_b = outputs(&mut runtime, continuation, config(0.7, 1, exact_b), true);
    assert_eq!(limit(&pool), exact_b);
    assert!(pool.fixture_host_current().unwrap() <= exact_b);
    assert_eq!(token_ids(&output_a), ids_a);
    drop((output_a, output_b, runtime, artifact));
    settle(&pool, source_baseline);
}

#[test]
fn live_prepared_continuation_cannot_handoff_its_capacity() {
    if !crate::tests::support::native_process::enter("capacity-handoff") {
        return;
    }
    let fixture = text_quote::PreparedResidencyFixture::new();
    let pool = fixture.pool.clone();
    let (mut runtime, artifact, source_baseline) = fixture.load(2, None);
    // Leave room for cold descriptors so this case reaches the live execution
    // fence. Exact capacity refusal is exercised by the successor case.
    let capacity_a = u64::MAX - 1;
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let live = driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            config(0.7, 1, capacity_a),
            Controller::default(),
        )
        .unwrap();
    let baseline = pool.fixture_host_current().unwrap();
    let peak = host(&pool).historical_peak_bytes;
    let before = report(driver.runtime());
    let paths_before = paths::snapshot();
    let inputs = paths::session_input_creation_attempts();
    let controller = Controller::default();
    let fault = MlxBackend::fail_next_sampling_for_test(Error::ArchitectureModel(
        "live continuation must reject before another sampler".into(),
    ));
    let error = MlxBackend::admit_text_preparation(
        driver.runtime(),
        &evidence(&tokens()),
        config(0.7, 1, u64::MAX),
        &controller,
    )
    .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(limit(&pool), capacity_a);
    assert_eq!(pool.fixture_host_current().unwrap(), baseline);
    assert_eq!(host(&pool).historical_peak_bytes, peak);
    assert_eq!(report(driver.runtime()), before);
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.0.get(), (0, 0));
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    drop((error, fault, live, driver));
    reclaim();
    // Retirement, rather than historical request metadata, permits later work.
    let later = outputs(&mut runtime, tokens(), config(0.7, 1, capacity_a), true);
    drop((later, runtime, artifact));
    settle(&pool, source_baseline);
}

#[derive(Debug)]
struct HandoffSamplingFailure;

impl std::fmt::Display for HandoffSamplingFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("sampler preparation failed after successor admission")
    }
}
impl std::error::Error for HandoffSamplingFailure {}

#[test]
fn admitted_capacity_handoff_persists_after_settled_sampling_failure() {
    if !crate::tests::support::native_process::enter("capacity-handoff") {
        return;
    }
    let fixture = text_quote::PreparedResidencyFixture::new();
    let pool = fixture.pool.clone();
    let (mut runtime, artifact, source_baseline) = fixture.load(2, None);
    let capacity_a = exact_capacity(&runtime, &pool, &tokens(), 0.7, 2);
    let output_a = outputs(&mut runtime, tokens(), config(0.7, 2, capacity_a), true);
    let ids_a = token_ids(&output_a);
    let alias = output_a[0].value.clone();

    reclaim();
    let capacity_b = capacity_a.checked_mul(2).unwrap();
    let physical_a = pool.fixture_host_current().unwrap();
    let inputs = paths::session_input_creation_attempts();
    let reads = report(&runtime).weight_store().physical_reads;
    let controller = Controller::default();
    let fault =
        MlxBackend::fail_next_sampling_for_test(Error::Other(Box::new(HandoffSamplingFailure)));
    let error = ControlledTextGeneration::from_input(
        &mut runtime,
        TextGenerationInput::TokenIds(vec![ids_a[3], 6, 7]),
        config(0.7, 2, capacity_b),
        controller.clone(),
    )
    .err()
    .expect("admission succeeds before the native sampler setup fails");
    assert!(
        cause::<HandoffSamplingFailure>(&error).is_some(),
        "{error:?}"
    );
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_none()));
    // This counter tracks submitted decode-input construction, not prepared
    // prompt allocation. The sampler hook fails before model execution.
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.0.get(), (0, 0));
    assert_eq!(report(&runtime).weight_store().physical_reads, reads);
    assert_eq!(limit(&pool), capacity_b);
    drop((error, fault));
    // The hook returns before sampler work. Its empty native scope is fully
    // settled and certified, so only A's unchanged physical storage remains.
    crate::backend::submission_recovery::wait_for_retirement(|| {
        reclaim();
        pool.fixture_host_current().unwrap() == physical_a
    });
    let arrays = output_a
        .iter()
        .map(|token| &token.value)
        .collect::<Vec<_>>();
    assert!(live_bytes(Some(&runtime), &arrays) <= physical_a);
    drop(arrays);
    assert_eq!(limit(&pool), capacity_b);
    assert_eq!(token_ids(&output_a), ids_a);
    let token_bytes = retained_token_bytes(&alias);
    assert!(token_bytes > 0);
    drop((output_a, runtime, artifact));
    settle(&pool, source_baseline.checked_add(token_bytes).unwrap());
    // Successful admission commits the policy change even when a later
    // preparation stage fails safely. A's final alias retains that new limit.
    assert_eq!(limit(&pool), capacity_b);
    assert_eq!(alias.evaluated().unwrap().item::<u32>(), ids_a[0]);
    drop(alias);
    settle(&pool, source_baseline);
    assert_eq!(host(&pool).effective_limit, MemoryLimit::Unlimited);
}
