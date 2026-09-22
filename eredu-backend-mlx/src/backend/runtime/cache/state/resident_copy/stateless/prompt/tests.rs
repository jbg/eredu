//! Native mechanism fixture; selected-NoState control/quote integration is separate.
use super::*;
use crate::backend::{
    array_copy::IsolatedArrayCopy,
    managed_memory::NativeMemoryOwner,
    nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms},
    runtime::residency::storage::RetainedStorage,
};
use crate::memory_fixture::LedgerFixture;
use eredu_core::{
    cache::LayerCachePolicy, Admission, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig,
    StateMemoryLayout, TextGenerationConfig, WorkspaceBound,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceIsolatedCopyPlan};
use eredu_runtime::working_memory::{
    InferenceExecutionIdentity, InferenceRequest, InferenceStateRetention,
    InferenceTextPreparation, RegisteredWorkspaceStorage,
};
use safemlx::{Device, DeviceType};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn fresh_admission(bytes: u64) -> Admission {
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 0,
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
        0,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |n| {
        WorkspaceBound::bounded(
            n,
            "actual isolated native key/pending copy; absent decoder needs no table payload",
        )
    };
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            physical_domains: None,
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    crate::memory_fixture::admission(Admission {
        requested_positions: 1,
        state,
        incremental_required_bytes: Some(bytes),
        memory_limits: Default::default(),
        additional_headroom: Default::default(),
    })
}

fn fresh_requirements(pool: &MemoryLedger, bytes: u64) -> u64 {
    crate::memory_fixture::host_total(
        &pool
            .reservation_requirements(&fresh_admission(bytes), None)
            .unwrap(),
    )
}

fn fresh(
    pool: &MemoryLedger,
    bytes: u64,
    capacity: u64,
) -> Result<(InferenceTextPreparation, WorkingMemoryFundingRun), WorkingMemoryError> {
    let execution = InferenceExecutionIdentity::default();
    let admission = fresh_admission(bytes);
    let geometry = admission
        .state
        .execution_workspace
        .as_ref()
        .unwrap()
        .geometry;
    let reservation = pool.reserve_with_capacity(
        &execution,
        &admission,
        crate::memory_fixture::physical_host_limits(pool, capacity),
    )?;
    let (reservation, run) = reservation.into_funding()?;
    // The low-level preparation contract accepts zero outputs; no application
    // generation configuration or forward loop is being constructed here.
    let config = TextGenerationConfig::new(ResolvedGenerationConfig {
        do_sample: false,
        temperature: 0.0,
        top_k: 17,
        top_p: 0.83,
        min_p: 0.07,
        repetition_penalty: 1.13,
        repeat_last_n: 23,
        frequency_penalty: 0.17,
        presence_penalty: 0.29,
        max_new_tokens: Some(0),
    });
    let preparation =
        InferenceRequest::from(reservation).prepare_text(&execution, geometry, config)?;
    Ok((preparation, run))
}

fn metal() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Gpu, 0))
}
fn settle(pool: &MemoryLedger, bytes: u64) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(9);
    let mut reported = false;
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::memory::clear_cache().unwrap();
        safemlx::reclaim_allocation_owners();
        if !reported && std::time::Instant::now() >= deadline {
            eprintln!(
                "retirement expected funded={bytes}, actual={}, snapshot={:?}",
                pool.fixture_funded_charge().unwrap(),
                pool.snapshot().unwrap()
            );
            reported = true;
        }
        pool.fixture_funded_charge().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}
fn arrays(pool: &MemoryLedger) -> [Array; 2] {
    let owner = NativeMemoryOwner::acquire(pool).unwrap();
    let arrays = [
        Array::from_slice(&[0x1020_3040u32, 0x5060_7080], &[2]),
        Array::from_slice(&[23u32], &[]),
    ];
    let mut storage = RetainedStorage::default();
    for array in &arrays {
        array.evaluated().unwrap();
        storage.include_array(array).unwrap();
    }
    drop(storage.publish_unquoted(&owner).unwrap());
    drop(owner);
    arrays
}
fn quote(pool: &MemoryLedger, arrays: &[Array; 2]) -> (u64, WorkingMemoryStorage<StorageIdentity>) {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::new(&context);
    let inputs = arrays
        .iter()
        .map(|array| projection.project(array).unwrap())
        .collect::<Vec<_>>();
    let storage = projection.into_storage();
    assert!(storage.is_complete());
    let registered = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        storage
            .iter()
            .map(|(id, _, root)| crate::backend::nn::workspace::registered_storage_row(id, root)),
    )
    .unwrap();
    let plan = WorkspaceIsolatedCopyPlan::prepare(&context, registered.borrowed_storage(), &inputs)
        .unwrap();
    let bytes = plan
        .incremental_bytes()
        .unwrap()
        .checked_add(crate::memory_fixture::publication_control_bytes(
            inputs.len().checked_mul(2).unwrap(),
        ))
        .unwrap();
    let mut complete = RetainedStorage::default();
    for array in arrays {
        complete.include_array(array).unwrap();
    }
    (bytes, complete.pin_registered(pool).unwrap())
}
fn absent(state: &MlxPoolingAttentionState) {
    assert!(state.layer_slot_metadata().is_none());
    assert!(state.prepare_layer_copy_slots().unwrap().is_none());
    assert!(state.optional_layout().is_none());
    assert!(state.shared_layout().is_none());
    assert!(state.as_ref().is_empty());
}
fn words(array: &Array) -> Vec<u32> {
    array.evaluated().unwrap().try_to_vec::<u32>().unwrap()
}
fn memory(error: &Error) -> &WorkingMemoryError {
    let mut cause: &(dyn std::error::Error + 'static) = error;
    loop {
        if let Some(found) = cause.downcast_ref::<WorkingMemoryError>() {
            return found;
        }
        cause = cause.source().expect("typed memory cause");
    }
}

#[test]
fn absent_prompt_exact_native_capacity_preserves_none_and_fresh_retention_from_live_or_saved() {
    for use_saved in [false, true] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let stream = metal();
        let source_arrays = arrays(&pool);
        let mut original = Some(MlxPoolingAttentionState::stateless());
        let old = fresh(&pool, 0, u64::MAX).unwrap();
        original
            .as_mut()
            .unwrap()
            .inference_retention_mut()
            .admit(old.0.request());
        assert!(!original.as_ref().unwrap().inference_retention().is_empty());
        let old_revision = original
            .as_ref()
            .unwrap()
            .inference_retention()
            .revision()
            .clone();
        let p = PreparedStatelessPoolingCopy::prepare(original.as_ref().unwrap()).unwrap();
        let saved = p.copy(p).unwrap();
        if use_saved {
            drop(original.take());
        }
        drop(old);
        let plan = if use_saved {
            saved.prepare_copy()
        } else {
            PreparedStatelessPoolingCopy::prepare(original.as_ref().unwrap()).unwrap()
        };
        let (required, complete) = quote(&pool, &source_arrays);
        assert!(required > 0);
        let before = (
            pool.fixture_funded_charge().unwrap(),
            pool.fixture_host_peak().unwrap(),
        );
        let capacity = pool
            .fixture_host_current()
            .unwrap()
            .checked_add(fresh_requirements(&pool, required))
            .unwrap();
        assert!(matches!(
            fresh(&pool, required, capacity - 1),
            Err(WorkingMemoryError::Domain(
                eredu_core::MemoryDomainError::BudgetExceeded { .. }
            ))
        ));
        assert_eq!(
            (
                pool.fixture_funded_charge().unwrap(),
                pool.fixture_host_peak().unwrap()
            ),
            before
        );
        let (preparation, run) = fresh(&pool, required, capacity).unwrap();
        let (initialized, native) = plan
            .construct_prompt(preparation.claim_prompt().unwrap(), &run, complete)
            .unwrap();
        let (destination, completion) = plan.copy_prompt(initialized).unwrap().into_parts();
        absent(&destination);
        assert!(destination.inference_retention().is_empty());
        assert_ne!(destination.inference_retention().revision(), &old_revision);
        assert!(matches!(
            preparation.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        let roots = RefCell::new(Vec::with_capacity(4));
        let copied = source_arrays
            .iter()
            .map(|array| {
                IsolatedArrayCopy::new(array)
                    .copy_retained(&stream, &roots)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let mut outputs = RetainedStorage::default();
        for (source, output) in source_arrays.iter().zip(&copied) {
            output.evaluated().unwrap();
            assert_eq!(words(output), words(source));
            assert_ne!(
                output.allocation_info().unwrap().unwrap().identity(),
                source.allocation_info().unwrap().unwrap().identity()
            );
            outputs.include_array(output).unwrap();
        }
        assert_eq!(roots.borrow().len(), 4);
        drop(outputs.publish_funded(&native).unwrap());
        roots.borrow_mut().clear();
        native.certify().unwrap(); // every surviving output is independently published
        completion.finish().unwrap();
        preparation.bind_prompt().unwrap();
        if let Some(source) = original.as_ref() {
            absent(source);
            assert_eq!(source.inference_retention().revision(), &old_revision);
        }
        let escaped = copied[0].clone();
        let bytes = {
            let info = escaped.allocation_info().unwrap().unwrap();
            (info.bytes() as u64)
                .checked_add(info.host_control_bytes() as u64)
                .unwrap()
        };
        let account_controls = crate::memory_fixture::request_control_bytes(preparation.request());
        drop((
            copied,
            destination,
            source_arrays,
            saved,
            original,
            preparation,
            run,
        ));
        settle(&pool, bytes + account_controls);
        assert_eq!(words(&escaped), vec![0x1020_3040, 0x5060_7080]);
        drop(escaped);
        settle(&pool, 0);
    }
}

#[test]
fn absent_prompt_wrong_actual_source_and_cancelled_claim_leave_no_native_roots() {
    for cancel in [false, true] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let source_arrays = arrays(&pool);
        let source = MlxPoolingAttentionState::stateless();
        let other = MlxPoolingAttentionState::stateless();
        let plan = PreparedStatelessPoolingCopy::prepare(&source).unwrap();
        let (required, complete) = quote(&pool, &source_arrays);
        let (preparation, run) = fresh(
            &pool,
            required,
            pool.fixture_host_current()
                .unwrap()
                .checked_add(fresh_requirements(&pool, required))
                .unwrap(),
        )
        .unwrap();
        let (initialized, native) = plan
            .construct_prompt(preparation.claim_prompt().unwrap(), &run, complete)
            .unwrap();
        let before = (
            pool.fixture_funded_charge().unwrap(),
            pool.fixture_host_peak().unwrap(),
        );
        if cancel {
            drop(initialized);
        } else {
            let other = PreparedStatelessPoolingCopy::prepare(&other).unwrap();
            let error = other.copy_prompt(initialized).err().unwrap();
            assert_eq!(memory(&error), &WorkingMemoryError::IdentityMismatch);
        }
        assert_eq!(
            (
                pool.fixture_funded_charge().unwrap(),
                pool.fixture_host_peak().unwrap()
            ),
            before
        );
        assert!(matches!(
            preparation.claim_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        assert!(matches!(
            preparation.bind_prompt(),
            Err(WorkingMemoryError::PreparationAlreadyStarted)
        ));
        absent(&source);
        absent(&other);
        native.certify().unwrap(); // no numerical worker was reached
        drop((source_arrays, source, other, preparation, run));
        settle(&pool, 0);
    }
}

#[derive(Debug, thiserror::Error)]
#[error("injected failure after absent-prompt key copy")]
struct AfterKey;

#[test]
fn absent_prompt_failure_or_unwind_retains_real_partials_until_explicit_settlement() {
    for unwind in [false, true] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let stream = metal();
        let source_arrays = arrays(&pool);
        let source = MlxPoolingAttentionState::stateless();
        let plan = PreparedStatelessPoolingCopy::prepare(&source).unwrap();
        let (required, complete) = quote(&pool, &source_arrays);
        let (preparation, run) = fresh(
            &pool,
            required,
            pool.fixture_host_current()
                .unwrap()
                .checked_add(fresh_requirements(&pool, required))
                .unwrap(),
        )
        .unwrap();
        let (initialized, native) = plan
            .construct_prompt(preparation.claim_prompt().unwrap(), &run, complete)
            .unwrap();
        let prepared = plan.copy_prompt(initialized).unwrap();
        let roots = RefCell::new(Vec::with_capacity(4));
        let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), Error> {
            let _key = IsolatedArrayCopy::new(&source_arrays[0]).copy_retained(&stream, &roots)?;
            if unwind {
                panic!("after absent-prompt key copy");
            }
            Err(Error::Other(Box::new(AfterKey)))
        }));
        if unwind {
            assert!(result.is_err());
        } else {
            let error = result.unwrap().unwrap_err();
            assert!(std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<AfterKey>()
                .is_some());
        }
        assert_eq!(roots.borrow().len(), 2);
        assert_eq!(words(&source_arrays[1]), vec![23]);
        let charged = pool.fixture_funded_charge().unwrap();
        drop((prepared, preparation, run));
        assert_eq!(pool.fixture_funded_charge().unwrap(), charged);
        for root in roots.borrow().iter() {
            root.evaluated().unwrap();
        }
        assert_eq!(words(&roots.borrow()[1]), words(&source_arrays[0]));
        roots.borrow_mut().clear();
        native.certify().unwrap(); // all partial output handles settled and retired
        drop((source_arrays, source));
        settle(&pool, 0);
    }
}
