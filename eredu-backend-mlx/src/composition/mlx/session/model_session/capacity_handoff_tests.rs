//! Explicit successor capacity changes preserve completed native storage charges.

use super::disk_layerwise_tests::{
    assert_disk_window, cause, config, evidence, exact_capacity, live_bytes, load_runtime, outputs,
    report, settle, token_ids, tokens, Controller,
};
use super::*;
use crate::tests::support::path_instrumentation as paths;
use eredu_core::{
    ControlledTextGeneration, InferenceGeometry, OutputDemand, TextGenerationDriver,
    TextGenerationInput, TokenFilterController,
};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};

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

#[test]
fn completed_disk_run_hands_off_larger_capacity_with_old_outputs_in_both_drivers() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    for controlled in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut runtime, artifact) = load_runtime(&stream, &pool, true);
        let (capacity_a, _) = exact_capacity(&runtime, &pool, &tokens(), 0.0, 2);
        let output_a = outputs(
            &mut runtime,
            tokens(),
            config(0.0, 2, capacity_a),
            controlled,
        );
        let ids_a = token_ids(&output_a);
        let request_a = retained_token_request(&output_a[0]);
        let admission_a = request_a.memory_reservation().unwrap().admission().clone();
        assert!(
            matches!(
                request_a
                    .memory_reservation()
                    .unwrap()
                    .clone()
                    .into_funding(),
                Err(WorkingMemoryError::IdentityMismatch)
            ),
            "historical request metadata cannot recreate the unique funding owner"
        );
        let owner_a = output_a[0].owner.clone();
        let alias = output_a[0]
            .value
            .as_strided(&[1][..], &[1][..], 0, &stream)
            .unwrap();
        alias.evaluated().unwrap();
        let allocation_a = alias.allocation_info().unwrap().unwrap();
        assert_eq!(
            Some(allocation_a),
            output_a[0].value.allocation_info().unwrap()
        );
        assert!(owner_a.resources_releasable());
        assert!(owner_a.direct_route.borrow().is_none());
        let arrays_a = output_a
            .iter()
            .map(|token| &token.value)
            .collect::<Vec<_>>();
        settle(&pool, live_bytes(Some(&runtime), &arrays_a));
        drop(arrays_a);
        assert_eq!(pool.effective_capacity().unwrap(), capacity_a);
        assert!(pool.peak_bytes().unwrap() <= capacity_a);

        // A's final sampled token is still pending decode. The new request
        // explicitly supplies a larger finite ceiling; no MAX admission probe
        // may change the predecessor's policy while measuring this request.
        let continuation = vec![ids_a[3], 6, 7];
        let capacity_b = capacity_a.checked_mul(2).unwrap();
        let before = report(&runtime);
        let paths_before = paths::snapshot();
        let used = pool.used_bytes().unwrap();
        let peak = pool.peak_bytes().unwrap();
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
        assert_eq!(pool.used_bytes().unwrap(), used);
        assert_eq!(pool.peak_bytes().unwrap(), peak);
        assert_eq!(pool.effective_capacity().unwrap(), capacity_a);

        let output_b = outputs(
            &mut runtime,
            continuation,
            config(0.0, 2, capacity_b),
            controlled,
        );
        let ids_b = token_ids(&output_b);
        assert_disk_window(&runtime);
        runtime
            .session()
            .payload
            .model
            .erased()
            .validate_text_frontier(14)
            .unwrap();
        assert!(
            report(&runtime).weight_store().physical_reads > before.weight_store().physical_reads
        );
        assert_eq!(pool.effective_capacity().unwrap(), capacity_b);
        assert!(pool.peak_bytes().unwrap() <= capacity_b);
        assert_eq!(token_ids(&output_a), ids_a);
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation_a));
        assert_eq!(
            request_a.memory_reservation().unwrap().admission(),
            &admission_a
        );
        assert!(owner_a.resources_releasable());
        assert!(owner_a.direct_route.borrow().is_none());

        let reference_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (mut resident, resident_artifact) = load_runtime(&stream, &reference_pool, false);
        let mut full = tokens();
        full.extend_from_slice(&ids_a);
        full.extend([6, 7]);
        let (reference_capacity, _) = exact_capacity(&resident, &reference_pool, &full, 0.0, 2);
        let reference = outputs(
            &mut resident,
            full,
            config(0.0, 2, reference_capacity),
            controlled,
        );
        assert_eq!(ids_b, token_ids(&reference));
        drop((reference, resident, resident_artifact));
        settle(&reference_pool, 0);

        let all = output_a
            .iter()
            .chain(output_b.iter())
            .map(|token| &token.value)
            .collect::<Vec<_>>();
        settle(&pool, live_bytes(Some(&runtime), &all));
        drop(all);
        let token_bytes = live_bytes(None, &[&alias]);
        assert!(token_bytes > 0);
        drop((output_b, runtime, artifact));
        assert_eq!(pool.effective_capacity().unwrap(), capacity_b);
        drop((output_a, owner_a, request_a));
        settle(&pool, token_bytes);
        // Only A's physical alias remains. B's retirement cannot remove the
        // adopted ceiling or charge before this last old allocation retires.
        assert_eq!(pool.effective_capacity().unwrap(), capacity_b);
        assert_eq!(alias.allocation_info().unwrap(), Some(allocation_a));
        assert_eq!(alias.evaluated().unwrap().item::<u32>(), ids_a[0]);
        assert_eq!(
            pool.acquire_unquoted().unwrap_err(),
            WorkingMemoryError::ReservedWorkActive
        );
        drop(alias);
        settle(&pool, 0);
        assert_eq!(pool.effective_capacity().unwrap(), u64::MAX);
    }
}

#[test]
fn successor_one_byte_short_rejects_without_changing_predecessor_ceiling() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = load_runtime(&stream, &pool, true);
    let (capacity_a, _) = exact_capacity(&runtime, &pool, &tokens(), 0.7, 1);
    let output_a = outputs(&mut runtime, tokens(), config(0.7, 1, capacity_a), true);
    let ids_a = token_ids(&output_a);
    let arrays = output_a
        .iter()
        .map(|token| &token.value)
        .collect::<Vec<_>>();
    settle(&pool, live_bytes(Some(&runtime), &arrays));
    drop(arrays);
    let continuation = vec![ids_a[3], 6, 7];
    let baseline = pool.used_bytes().unwrap();
    let peak = pool.peak_bytes().unwrap();
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
    let Some(WorkingMemoryError::BudgetExceeded {
        required_bytes,
        available_bytes,
    }) = cause::<WorkingMemoryError>(&error)
    else {
        panic!(
            "a valid but insufficient successor capacity must preserve its budget error: {error:?}"
        );
    };
    let required_bytes = *required_bytes;
    assert_eq!(*available_bytes, probe_capacity - baseline);
    let exact_b = baseline.checked_add(required_bytes).unwrap();
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
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::BudgetExceeded {
            required_bytes,
            available_bytes: required_bytes - 1,
        })
    );
    assert_eq!(pool.effective_capacity().unwrap(), capacity_a);
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    assert_eq!(pool.peak_bytes().unwrap(), peak);
    assert_eq!(report(&runtime), before);
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.0.get(), (0, 0));
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    drop((error, fault));
    let output_b = outputs(&mut runtime, continuation, config(0.7, 1, exact_b), true);
    assert_eq!(pool.effective_capacity().unwrap(), exact_b);
    assert!(pool.peak_bytes().unwrap() <= exact_b);
    assert_eq!(token_ids(&output_a), ids_a);
    drop((output_a, output_b, runtime, artifact));
    settle(&pool, 0);
}

#[test]
fn live_prepared_continuation_cannot_handoff_its_capacity() {
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = load_runtime(&stream, &pool, true);
    let (capacity_a, _) = exact_capacity(&runtime, &pool, &tokens(), 0.7, 1);
    let mut driver = TextGenerationDriver::new(&mut runtime);
    let live = driver
        .start_input(
            TextGenerationInput::TokenIds(tokens()),
            config(0.7, 1, capacity_a),
            Controller::default(),
        )
        .unwrap();
    let baseline = pool.used_bytes().unwrap();
    let peak = pool.peak_bytes().unwrap();
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
        config(0.7, 1, capacity_a.checked_mul(2).unwrap()),
        &controller,
    )
    .unwrap_err();
    assert_eq!(
        cause::<WorkingMemoryError>(&error),
        Some(&WorkingMemoryError::ExecutionFenced)
    );
    assert_eq!(pool.effective_capacity().unwrap(), capacity_a);
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    assert_eq!(pool.peak_bytes().unwrap(), peak);
    assert_eq!(report(driver.runtime()), before);
    assert_eq!(paths::snapshot(), paths_before);
    assert_eq!(paths::session_input_creation_attempts(), inputs);
    assert_eq!(controller.0.get(), (0, 0));
    assert!(TEST_SAMPLING_FAILURE.with(|slot| slot.borrow().is_some()));
    drop((error, fault, live, driver));
    settle(&pool, live_bytes(Some(&runtime), &[]));
    // Retirement, rather than historical request metadata, permits later work.
    let later = outputs(&mut runtime, tokens(), config(0.7, 1, capacity_a), true);
    drop((later, runtime, artifact));
    settle(&pool, 0);
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
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, artifact) = load_runtime(&stream, &pool, true);
    let (capacity_a, _) = exact_capacity(&runtime, &pool, &tokens(), 0.7, 2);
    let output_a = outputs(&mut runtime, tokens(), config(0.7, 2, capacity_a), true);
    let ids_a = token_ids(&output_a);
    let alias = output_a[0].value.clone();
    let arrays = output_a
        .iter()
        .map(|token| &token.value)
        .collect::<Vec<_>>();
    settle(&pool, live_bytes(Some(&runtime), &arrays));
    drop(arrays);
    let capacity_b = capacity_a.checked_mul(2).unwrap();
    let physical_a = pool.used_bytes().unwrap();
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
    assert_eq!(pool.effective_capacity().unwrap(), capacity_b);
    drop((error, fault));
    // The hook returns before sampler work. Its empty native scope is fully
    // settled and certified, so only A's unchanged physical storage remains.
    settle(&pool, physical_a);
    let arrays = output_a
        .iter()
        .map(|token| &token.value)
        .collect::<Vec<_>>();
    assert_eq!(live_bytes(Some(&runtime), &arrays), physical_a);
    drop(arrays);
    assert_eq!(pool.effective_capacity().unwrap(), capacity_b);
    assert_eq!(token_ids(&output_a), ids_a);
    let token_bytes = live_bytes(None, &[&alias]);
    assert!(token_bytes > 0);
    drop((output_a, runtime, artifact));
    settle(&pool, token_bytes);
    // Successful admission commits the policy change even when a later
    // preparation stage fails safely. A's final alias retains that new limit.
    assert_eq!(pool.effective_capacity().unwrap(), capacity_b);
    assert_eq!(alias.evaluated().unwrap().item::<u32>(), ids_a[0]);
    drop(alias);
    settle(&pool, 0);
    assert_eq!(pool.effective_capacity().unwrap(), u64::MAX);
}
