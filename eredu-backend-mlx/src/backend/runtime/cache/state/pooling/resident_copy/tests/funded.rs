use super::*;
use crate::backend::{
    managed_memory::NativeMemoryOwner,
    nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms},
    runtime::{cache::state::PreparedResidentDecoderCopy, residency::storage::RetainedStorage},
};
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, OutputDemand, StateMemoryLayout, TextGenerationConfig, WorkspaceBound,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceIsolatedCopyPlan};
use eredu_runtime::{
    working_memory::{
        BorrowedFundedSampler, InferenceExecutionIdentity, InferenceRequest,
        InferenceTextPreparation, RegisteredSamplingCopy, RegisteredWorkspaceCopy,
        RegisteredWorkspaceStorage, RunOwnedTextSampler, WorkingMemoryFundingRun,
        WorkspaceCopyLimits,
    },
    SharedHostMetadata,
};
pub(in super::super) fn metal() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Gpu, 0))
}
pub(in super::super) fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::reclaim_allocation_owners();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}
pub(in super::super) fn publish_source(
    source: &MlxPoolingAttentionState,
    owner: &NativeMemoryOwner,
) {
    let plan = PreparedResidentPoolingCopy::prepare(source).unwrap();
    let mut storage = RetainedStorage::default();
    plan.visit_retained_arrays(&mut |array| {
        array.evaluated().unwrap();
        storage.include_array(array).unwrap();
    });
    storage
        .include_slot_metadata(source.layer_slot_metadata().unwrap().clone())
        .unwrap();
    storage
        .include_metadata(SharedHostMetadata::Layout(plan.shared_layout().clone()))
        .unwrap();
    drop(storage.publish_unquoted(owner).unwrap());
}
// Same closed bootstrap as the portable runtime account fixture: this explicit
// 1024-byte source envelope covers only a scalar sampler's host construction.
// Native decoder copying below uses the selected Metal facts, never this number.
pub(in super::super) fn sampler(
    pool: &WorkingMemoryPool,
) -> (
    RunOwnedTextSampler,
    InferenceTextPreparation,
    WorkingMemoryFundingRun,
) {
    let execution = InferenceExecutionIdentity::default();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 1,
        max_output_tokens: 2,
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
        2,
        1,
        std::num::NonZeroU8::new(4).unwrap(),
    )
    .unwrap();
    let bound = |bytes| WorkspaceBound::bounded(bytes, "portable scalar sampler host fixture");
    let state = state
        .with_execution_workspace(ExecutionWorkspaceEstimate {
            geometry,
            activations: bound(1024),
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
                requested_positions: 3,
                state,
                incremental_required_bytes: 1024,
                available_memory_bytes: None,
            },
            u64::MAX,
        )
        .unwrap();
    let (reservation, run) = reservation.into_funding().unwrap();
    let config = TextGenerationConfig::new(
        eredu_core::resolve_generation_config(
            None,
            eredu_core::GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let preparation = InferenceRequest::from(reservation)
        .prepare_text(&execution, geometry, config)
        .unwrap();
    let (sampler, completion) = preparation
        .claim_sampling(config)
        .unwrap()
        .construct_sampler(run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    (sampler, preparation, run)
}

pub(in super::super) fn sampling_plan<'a>(
    plan: &PreparedResidentPoolingCopy<'_>,
    sampler: BorrowedFundedSampler<'a>,
    pool: &WorkingMemoryPool,
) -> (
    RegisteredSamplingCopy<'a, StorageIdentity>,
    eredu_runtime::working_memory::WorkingMemoryStorage<StorageIdentity>,
) {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::new(&context);
    let inputs = operands(plan)
        .into_iter()
        .map(|array| projection.project(array).unwrap())
        .collect::<Vec<_>>();
    let native = projection.into_storage();
    assert!(native.is_complete());
    let source = RegisteredWorkspaceStorage::bind(
        pool,
        &context,
        native
            .iter()
            .map(|(id, _, root)| (StorageIdentity::Native(id), root.clone())),
    )
    .unwrap();
    let program =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &inputs).unwrap();
    let arrays = RegisteredWorkspaceCopy::bind(program, source).unwrap();
    let mut complete = RetainedStorage::default();
    plan.visit_retained_arrays(&mut |array| complete.include_array(array).unwrap());
    complete
        .include_metadata(SharedHostMetadata::Layout(plan.shared_layout().clone()))
        .unwrap();
    let pin = complete.pin_registered(pool).unwrap();
    (
        RegisteredSamplingCopy::prepare(sampler, arrays).unwrap(),
        pin,
    )
}
pub(in super::super) fn finish_native(
    saved: &SavedResidentPoolingCopy,
    scope: eredu_runtime::working_memory::WorkingMemoryFundingScope,
    roots: &RefCell<Vec<Array>>,
) {
    for array in roots.borrow().iter() {
        array.evaluated().unwrap();
    }
    let mut storage = RetainedStorage::default();
    saved
        .prepare_copy()
        .unwrap()
        .visit_retained_arrays(&mut |array| {
            array.evaluated().unwrap();
            storage.include_array(array).unwrap();
        });
    drop(storage.publish_funded(&scope).unwrap());
    roots.borrow_mut().clear();
    scope.certify().unwrap();
}

#[test]
fn funded_pooling_copy_matches_local_compressed_sparse_and_saved_copy_after_source_drop() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = metal();
    let source = state(&stream, true);
    // The ordinary baseline also allocates, under this fixture's loading owner.
    let ordinary =
        super::super::super::MlxPoolingAttentionStateFactory::isolated_snapshot(&source, &stream)
            .unwrap();
    let expected_ordinary = values(&PreparedResidentPoolingCopy::prepare(&ordinary).unwrap());
    let controls_ordinary = controls(&PreparedResidentPoolingCopy::prepare(&ordinary).unwrap());
    drop(ordinary);
    publish_source(&source, &loading);
    drop(loading);
    let baseline = pool.used_bytes().unwrap();
    settle(&pool, baseline);
    let (sampler, preparation, run) = sampler(&pool);
    let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
    let expected = values(&plan);
    let expected_controls = controls(&plan);
    assert!(expected.iter().flatten().any(|v| *v != 0.));
    let ids = operands(&plan)
        .iter()
        .map(|a| a.allocation_info().unwrap().unwrap().identity())
        .collect::<Vec<_>>();
    assert!(ids.iter().collect::<std::collections::BTreeSet<_>>().len() < ids.len());
    assert_eq!(expected_ordinary, expected);
    assert_eq!(controls_ordinary, expected_controls);
    let (sampling, complete) = sampling_plan(&plan, sampler.borrow_funded(), &pool);
    let joined = sampling
        .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
        .unwrap();
    let required = joined.required_bytes();
    let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
    let error = pool
        .copy_text_components(joined, WorkspaceCopyLimits::new(before.0 + required - 1))
        .err()
        .unwrap();
    assert!(
        matches!(error,DecoderCopyAdmissionError::Memory(WorkingMemoryError::BudgetExceeded {required_bytes, available_bytes}) if required_bytes==required && available_bytes==required-1)
    );
    assert_eq!(
        (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
        before
    );
    let (sampling, complete) = sampling_plan(&plan, sampler.borrow_funded(), &pool);
    let joined = sampling
        .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
        .unwrap();
    let (copied_sampler, slots, native) = pool
        .copy_text_components(
            joined,
            WorkspaceCopyLimits {
                application_memory_budget_bytes: Some(required),
                ..WorkspaceCopyLimits::new(u64::MAX)
            },
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    let roots = RefCell::new(Vec::with_capacity(ids.len() * 2));
    let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
    assert_eq!(roots.borrow().len(), ids.len() * 2);
    finish_native(&saved, scope, &roots);
    assert_eq!(pool.used_bytes().unwrap(), before.0 + required);
    assert_eq!(values(&saved.prepare_copy().unwrap()), expected);
    assert_eq!(controls(&saved.prepare_copy().unwrap()), expected_controls);
    assert!(saved
        .shared_layout()
        .same_storage(source.shared_layout().unwrap()));
    let copied_ids = operands(&saved.prepare_copy().unwrap())
        .iter()
        .map(|a| a.allocation_info().unwrap().unwrap().identity())
        .collect::<Vec<_>>();
    assert!(copied_ids.iter().all(|id| !ids.contains(id)));
    assert_eq!(
        copied_ids
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        copied_ids.len()
    );
    drop((source, sampler, preparation, run));
    settle(
        &pool,
        custody.bytes() + saved.shared_layout().capacity_bytes().unwrap(),
    );
    let plan = saved.prepare_copy().unwrap();
    let (sampling, complete) = sampling_plan(&plan, copied_sampler.borrow_funded(), &pool);
    let joined = sampling
        .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
        .unwrap();
    let required = joined.required_bytes();
    let before = pool.used_bytes().unwrap();
    let (next_sampler, slots, next_native) = pool
        .copy_text_components(joined, WorkspaceCopyLimits::new(before + required))
        .unwrap();
    let (next_custody, scope) = next_native.into_parts();
    let duplicate = plan.copy_retained(slots, &stream, &roots).unwrap();
    finish_native(&duplicate, scope, &roots);
    assert_eq!(pool.used_bytes().unwrap(), before + required);
    drop((saved, copied_sampler, custody));
    assert_eq!(values(&duplicate.prepare_copy().unwrap()), expected);
    assert_eq!(
        controls(&duplicate.prepare_copy().unwrap()),
        expected_controls
    );
    let escaped = operands(&duplicate.prepare_copy().unwrap())[0].clone();
    let escaped_bytes = escaped.allocation_info().unwrap().unwrap().bytes() as u64;
    drop((duplicate, next_sampler, next_custody));
    settle(&pool, escaped_bytes);
    assert!(escaped
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .any(|v| *v != 0.));
    drop(escaped);
    settle(&pool, 0);
}

#[test]
fn pooling_builder_identity_rejects_before_work_and_late_failure_keeps_roots() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = metal();
    let source = state(&stream, true);
    let other = state(&stream, true);
    publish_source(&source, &loading);
    publish_source(&other, &loading);
    drop(loading);
    let (sampler, preparation, run) = sampler(&pool);
    let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
    let expected = values(&plan);
    let roots = RefCell::new(Vec::new());
    let (sampling, complete) = sampling_plan(&plan, sampler.borrow_funded(), &pool);
    let (copy_sampler, slots, native) = pool
        .copy_text_components(
            sampling
                .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
                .unwrap(),
            WorkspaceCopyLimits::new(u64::MAX),
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    let error = PreparedResidentPoolingCopy::prepare(&other)
        .unwrap()
        .copy_retained(slots, &stream, &roots)
        .err()
        .unwrap();
    assert!(matches!(
        error,
        ResidentPoolingCopyError::Memory(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(roots.borrow().is_empty());
    scope.certify().unwrap();
    drop((copy_sampler, custody));
    let (sampling, complete) = sampling_plan(&plan, sampler.borrow_funded(), &pool);
    let (copy_sampler, slots, native) = pool
        .copy_text_components(
            sampling
                .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
                .unwrap(),
            WorkspaceCopyLimits::new(u64::MAX),
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    FAIL_AFTER_LAYER.with(|fail| fail.set(Some(1)));
    let error = plan.copy_retained(slots, &stream, &roots).err().unwrap();
    assert!(
        matches!(&error,ResidentPoolingCopyError::Native(error) if std::error::Error::source(error).is_some_and(|source| source.downcast_ref::<InjectedPoolingCopyFailure>().is_some()))
    );
    // Two local slots, then two local plus three non-overlapping pooled slots;
    // each completed operand retains its contiguous and final copy roots.
    assert_eq!(roots.borrow().len(), 14);
    assert_eq!(
        values(&PreparedResidentPoolingCopy::prepare(&source).unwrap()),
        expected
    );
    drop((source, other, sampler, preparation, run));
    for array in roots.borrow().iter() {
        array.evaluated().unwrap();
    }
    assert!(roots.borrow()[1]
        .evaluated()
        .unwrap()
        .try_to_vec::<f32>()
        .unwrap()
        .iter()
        .any(|v| *v != 0.));
    roots.borrow_mut().clear();
    scope.certify().unwrap();
    drop((copy_sampler, custody));
    settle(&pool, 0);
}

#[test]
fn native_dispatch_uses_same_single_account_for_pooling_slots() {
    for populated in [false, true] {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = metal();
        let source = state(&stream, populated);
        publish_source(&source, &loading);
        drop(loading);
        let (sampler, preparation, run) = sampler(&pool);
        let plan = PreparedResidentPoolingCopy::prepare(&source).unwrap();
        let expected = values(&plan);
        let (sampling, complete) = sampling_plan(&plan, sampler.borrow_funded(), &pool);
        let native_plan = PreparedResidentDecoderCopy::pooling(&source).unwrap();
        assert_eq!(native_plan.global_layer_start(), None);
        let (copy_sampler, slots, native) = native_plan
            .host_copy(&pool)
            .unwrap()
            .admit(
                &pool,
                sampling,
                complete,
                WorkspaceCopyLimits::new(u64::MAX),
            )
            .unwrap();
        let aggregate = native.bytes();
        let (custody, scope) = native.into_parts();
        let roots = RefCell::new(Vec::new());
        let saved = native_plan.copy_retained(slots, &stream, &roots).unwrap();
        for array in roots.borrow().iter() {
            array.evaluated().unwrap();
        }
        let mut storage = RetainedStorage::default();
        let mut actual = Vec::new();
        saved
            .prepare_copy()
            .unwrap()
            .visit_retained_arrays(&mut |array| {
                array.evaluated().unwrap();
                storage.include_array(array).unwrap();
                actual.push(array.evaluated().unwrap().try_to_vec::<f32>().unwrap());
            });
        drop(storage.publish_funded(&scope).unwrap());
        roots.borrow_mut().clear();
        scope.certify().unwrap();
        assert_eq!(actual, expected);
        assert!(aggregate > saved.protected_slot_bytes());
        assert_eq!(saved.global_layer_start(), None);
        drop((
            source,
            sampler,
            preparation,
            run,
            saved,
            copy_sampler,
            custody,
        ));
        settle(&pool, 0);
    }
}

mod stateless;
