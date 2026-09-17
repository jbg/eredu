use super::*;
use crate::backend::{
    nn::workspace::MlxMetalWorkspaceMechanisms, runtime::residency::storage::RetainedStorage,
};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, LayerSchedule, OutputDemand, ResolvedGenerationConfig, StateMemoryLayout,
    TextGenerationConfig, WorkspaceBound, cache::LayerCachePolicy,
};
use eredu_nn::workspace::WorkspaceContext;
use eredu_runtime::working_memory::{
    HostSlotStorageKey, InferenceExecutionIdentity, InferenceRequest, InferenceTextPreparation,
    RegisteredDecoderHostCopy, WorkingMemoryError, WorkingMemoryPool,
};
use eredu_runtime::{HostMetadataKey, HostSlotTable};
use safemlx::{Device, DeviceType, Dtype, ops::indexing::TryIndexOp};
use std::{cell::Cell, num::NonZeroU8};
const CAPACITY: u64 = 1048576;
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Key(HostMetadataKey);
impl HostSlotStorageKey for Key {
    fn host_slot_identity(&self) -> Option<&HostMetadataKey> {
        Some(&self.0)
    }
}
thread_local! {static HOUSEKEEPING:Cell<usize>=const{Cell::new(0)};}
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct Cold;
impl Cold {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
    fn check(&self) {
        assert_eq!(HOUSEKEEPING.get(), 0);
    }
}
impl Drop for Cold {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}
fn reclaim(pool: &WorkingMemoryPool, expected: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        pool.used_bytes().unwrap() == expected
    });
}
fn source(stream: &Stream, signed: bool) -> Array {
    let root = if signed {
        Array::from_slice(&[11i32, 23, 37, 53], &[4])
    } else {
        Array::from_slice(&[11u32, 23, 37, 53], &[4])
    };
    let source = root.try_index_device(2..3, stream).unwrap();
    source.evaluated().unwrap();
    source
}
fn fresh(
    pool: &WorkingMemoryPool,
    bytes: u64,
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
        NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "closed scalar host preparation fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(bytes),
            attention: bound(0),
            vocabulary: bound(0),
            state_update: bound(0),
            materialization: bound(0),
            retained: bound(0),
        })
        .unwrap();
    let reservation = pool
        .reserve_with_capacity(
            &execution,
            &Admission {
                requested_positions: 1,
                state,
                incremental_required_bytes: bytes,
                available_memory_bytes: None,
            },
            CAPACITY,
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let request = InferenceRequest::from(reservation);
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
    (
        request.prepare_text(&execution, geometry, config).unwrap(),
        run,
        config,
    )
}
fn completion(
    preparation: &InferenceTextPreparation,
    run: &WorkingMemoryFundingRun,
) -> InferencePromptCompletion {
    let source = HostSlotTable::new(Vec::<u8>::new().into_boxed_slice());
    let key = Key(source.metadata().identity().registry_key().clone());
    let charge = run.pool().register_storage([(key.clone(), 0)]).unwrap();
    let plan =
        RegisteredDecoderHostCopy::bind(run.pool(), source.prepare_copy_slots().unwrap(), key)
            .unwrap()
            .for_dense_destination::<u8>()
            .unwrap();
    let (slots, native) = preparation
        .claim_prompt()
        .unwrap()
        .construct_dense_decoder(plan, run, charge)
        .unwrap();
    let finished = slots.finish().unwrap();
    let destination = Key(finished.metadata().identity().registry_key().clone());
    let (table, completion) = finished.publish(destination).unwrap();
    native.certify().unwrap();
    drop(table);
    completion
}
#[test]
fn exact_host_and_native_pending_plan_share_one_part_and_preserve_incremental_identity() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for signed in [false, true] {
        let source = source(&stream, signed);
        let original = source.try_metadata_snapshot().unwrap();
        let facts = MlxMetalWorkspaceMechanisms::current_host().unwrap();
        let context = WorkspaceContext::new(facts);
        let cold = Cold::new();
        let plan = PreparedPendingPrompt::new(&source).unwrap();
        let mut projection = ExistingArrayProjection::new(&context);
        let input = projection.project(&source).unwrap();
        context.begin_state_span(&[input.clone()]).unwrap();
        let output = plan.trace(&mut projection).unwrap();
        let report = context.report(&[input, output]).unwrap();
        let native_bytes = report.total_bytes.unwrap();
        let host_bytes = plan.host_plan().initialization_peak_bytes();
        assert_eq!(
            plan.host_plan().retained_bytes(),
            std::mem::size_of::<input::InputPart>() as u64
        );
        assert!(report.unpriced_operations.is_empty());
        assert!(report.unpriced_host_operations.is_empty());
        cold.check();
        drop(cold);
        drop(projection);
        let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
        let mut source_inventory = RetainedStorage::default();
        source_inventory.include_array(&source).unwrap();
        let source_storage = source_inventory.register(&pool).unwrap();
        let source_bytes = source_storage.bytes();
        let total = host_bytes.checked_add(native_bytes).unwrap();
        // This low-level zero-output request prices only the actual component
        // workers; it makes no model-forward or resumed-run claim.
        let (preparation, run, _) = fresh(&pool, total);
        let complete = completion(&preparation, &run);
        let host = plan.prepare_host(complete, &run).unwrap();
        let native = run.scope().unwrap();
        let roots = RefCell::new(Vec::with_capacity(plan.retained_descriptor_count()));
        let (prompt, complete) = plan.copy_into_prompt(host, &stream, &roots).unwrap();
        assert!(prompt.quote.is_none());
        assert!(prompt.shared_cache_identity().is_none());
        assert!(prompt.cache_identity().is_none());
        let alias = prompt.clone();
        prompt.with_borrowed(|left| {
            alias.with_borrowed(|right| {
                assert!(std::ptr::eq(left.parts, right.parts));
                assert!(left.cache_identity().is_none());
                assert!(right.cache_identity().is_none());
                assert!(left.parts[0].metadata().is_empty());
                assert!(left.parts[0].extents().is_empty());
            })
        });
        prompt
            .inference_request
            .as_ref()
            .unwrap()
            .validate_same_request(preparation.request())
            .unwrap();
        let raw = prompt.with_borrowed(|input| input.parts[0].payload().value().clone());
        assert_eq!(raw.shape(), [1, 1]);
        assert_eq!(raw.dtype(), Dtype::Uint32);
        for root in roots.borrow().iter() {
            root.evaluated().unwrap();
        }
        assert_eq!(raw.evaluated().unwrap().as_slice::<u32>(), [37]);
        let actual = raw.try_metadata_snapshot().unwrap().allocation().unwrap();
        assert_ne!(actual.identity(), original.allocation().unwrap().identity());
        assert_eq!(source.try_metadata_snapshot().unwrap(), original);
        let mut destination = RetainedStorage::default();
        // Publish every live intermediate before certifying the native scope.
        for root in roots.borrow().iter() {
            destination.include_array(root).unwrap();
        }
        destination.include_array(&raw).unwrap();
        drop(destination.publish_funded(&native).unwrap());
        native.certify().unwrap();
        complete.finish().unwrap();
        preparation.bind_prompt().unwrap();
        drop((preparation, roots));
        run.close().unwrap();
        drop(prompt);
        assert!(pool.used_bytes().unwrap() >= source_bytes + host_bytes + actual.bytes() as u64);
        assert!(matches!(
            pool.acquire_unquoted(),
            Err(WorkingMemoryError::ReservedWorkActive)
        ));
        drop(alias);
        reclaim(&pool, source_bytes + actual.bytes() as u64);
        drop((source_storage, source));
        reclaim(&pool, actual.bytes() as u64);
        drop(raw);
        reclaim(&pool, 0);
    }
}

#[test]
fn short_host_hold_rejects_before_pending_numerical_construction() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = source(&stream, true);
    let original = source.try_metadata_snapshot().unwrap();
    let plan = PreparedPendingPrompt::new(&source).unwrap();
    let h = plan.host_plan().initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let mut inventory = RetainedStorage::default();
    inventory.include_array(&source).unwrap();
    let retained = inventory.register(&pool).unwrap();
    let (preparation, run, _) = fresh(&pool, h - 1);
    let complete = completion(&preparation, &run);
    let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
    let cold = Cold::new();
    let error = plan.prepare_host(complete, &run).unwrap_err();
    assert!(std::error::Error::source(&error).unwrap().downcast_ref::<WorkingMemoryError>().is_some_and(|error|
        matches!(error,WorkingMemoryError::BudgetExceeded{required_bytes,available_bytes} if *required_bytes==h && *available_bytes==h-1)));
    assert_eq!(
        (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
        before
    );
    assert_eq!(source.try_metadata_snapshot().unwrap(), original);
    cold.check();
    drop(cold);
    drop((plan, preparation));
    run.close().unwrap();
    reclaim(&pool, retained.bytes());
    drop((retained, source));
    reclaim(&pool, 0);
}

#[test]
fn closed_run_rejects_prepared_host_before_any_native_root_is_created() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let source = source(&stream, false);
    let original = source.try_metadata_snapshot().unwrap();
    let plan = PreparedPendingPrompt::new(&source).unwrap();
    let h = plan.host_plan().initialization_peak_bytes();
    let pool = WorkingMemoryPool::new(CAPACITY, 0).unwrap();
    let (preparation, run, _) = fresh(&pool, h);
    let complete = completion(&preparation, &run);
    let host = plan.prepare_host(complete, &run).unwrap();
    run.close().unwrap();
    let roots = RefCell::new(Vec::new());
    let cold = Cold::new();
    let error = plan.copy_into_prompt(host, &stream, &roots).unwrap_err();
    assert!(
        std::error::Error::source(&error)
            .unwrap()
            .downcast_ref::<WorkingMemoryError>()
            .is_some_and(|error| matches!(error, WorkingMemoryError::ExecutionFenced))
    );
    assert!(roots.borrow().is_empty());
    assert_eq!(source.try_metadata_snapshot().unwrap(), original);
    cold.check();
    drop(cold);
    drop(preparation);
    reclaim(&pool, 0);
}
