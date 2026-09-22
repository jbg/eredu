use super::*;
use crate::backend::{
    managed_memory::NativeMemoryOwner,
    nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms},
    runtime::residency::storage::RetainedStorage,
};
use crate::memory_fixture::LedgerFixture;
use eredu_core::{
    Admission, EstimationCompleteness, ExecutionWorkspaceEstimate, InferenceGeometry,
    InputTokenCount, OutputDemand, StateMemoryLayout, TextGenerationConfig, WorkspaceBound,
};
use eredu_nn::workspace::{WorkspaceContext, WorkspaceIsolatedCopyPlan};
use eredu_runtime::{
    working_memory::{
        AdmittedWorkspaceCopy, BorrowedFundedSampler, FundedSamplerCopy,
        InferenceExecutionIdentity, InferenceRequest, InferenceTextPreparation,
        RegisteredSamplingCopy, RegisteredWorkspaceCopy, RegisteredWorkspaceStorage,
        RunOwnedTextSampler, WorkingMemoryFundingRun, WorkspaceCopyLimits,
    },
    SharedHostMetadata,
};

pub(in super::super) fn metal() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Gpu, 0))
}
pub(in super::super) fn settle(pool: &MemoryLedger, bytes: u64) {
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
pub(in super::super) fn publish_source(source: &MlxKeyValueState, owner: &NativeMemoryOwner) {
    let plan = source.prepare_resident_copy().unwrap();
    let mut storage = RetainedStorage::default();
    for array in operands(&plan) {
        array.evaluated().unwrap();
        storage.include_array(array).unwrap();
    }
    storage
        .include_slot_metadata(source.layer_slot_metadata().clone())
        .unwrap();
    storage
        .include_metadata(SharedHostMetadata::Layout(source.layout.clone()))
        .unwrap();
    drop(storage.publish_unquoted(owner).unwrap());
}

// Same closed bootstrap as the portable runtime account fixture: this explicit
// 1024-byte source envelope covers only a scalar sampler's host construction.
// Native decoder copying below uses the selected Metal facts, never this number.
pub(in super::super) fn sampler(
    pool: &MemoryLedger,
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
            physical_domains: None,
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
            &crate::memory_fixture::host_admission(Admission {
                requested_positions: 3,
                state,
                incremental_required_bytes: Some(1024),
                memory_limits: Default::default(),
                additional_headroom: Default::default(),
            }),
            crate::memory_fixture::resolved_limits(u64::MAX),
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
        .prepare_text(&execution, geometry, config.clone())
        .unwrap();
    let (sampler, completion) = preparation
        .claim_sampling(config)
        .unwrap()
        .construct_sampler(run.sampler_scope().unwrap())
        .unwrap();
    completion.finish().unwrap();
    (sampler, preparation, run)
}

pub(in super::super) fn admit(
    plan: &PreparedResidentKvCopy<'_>,
    sampler: BorrowedFundedSampler<'_>,
    pool: &MemoryLedger,
) -> (
    FundedSamplerCopy,
    InitializedDecoderSlots<MlxKeyValueLayerState>,
    AdmittedWorkspaceCopy,
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
            .map(|(id, _, root)| crate::backend::nn::workspace::registered_storage_row(id, root)),
    )
    .unwrap();
    let program =
        WorkspaceIsolatedCopyPlan::prepare(&context, source.borrowed_storage(), &inputs).unwrap();
    let arrays = RegisteredWorkspaceCopy::bind(program, source).unwrap();
    let complete = pool
        .pin_registered_storage(
            native
                .iter()
                .map(|(id, bytes, _)| (StorageIdentity::Native(id), bytes))
                .chain(std::iter::once((
                    StorageIdentity::HostMetadata(
                        plan.shared_layout().identity().registry_key().clone(),
                    ),
                    plan.shared_layout().capacity_bytes().unwrap(),
                ))),
        )
        .unwrap();
    let joined = RegisteredSamplingCopy::prepare(sampler, arrays)
        .unwrap()
        .with_decoder_slots(plan.host_copy(pool).unwrap(), complete)
        .unwrap();
    pool.copy_text_components(
        joined,
        crate::memory_fixture::publication_copy_limits(pool, inputs.len(), u64::MAX),
    )
    .unwrap()
}

pub(in super::super) fn finish_native(
    saved: &SavedResidentKvCopy,
    scope: eredu_runtime::working_memory::WorkingMemoryFundingScope,
    roots: &RefCell<Vec<Array>>,
) {
    for array in roots.borrow().iter() {
        array.evaluated().unwrap();
    }
    let mut storage = RetainedStorage::default();
    for array in operands(&saved.prepare_copy().unwrap()) {
        array.evaluated().unwrap();
        storage.include_array(array).unwrap();
    }
    drop(storage.publish_funded(&scope).unwrap());
    roots.borrow_mut().clear();
    scope.certify().unwrap();
}

#[test]
fn funded_resident_copy_preserves_controls_and_aliases_without_runnable_state() {
    for aliases in [false, true] {
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
        let loading = NativeMemoryOwner::acquire(&pool).unwrap();
        let stream = metal();
        let mut source = state(&stream);
        if aliases {
            let alias = source.layers.slots()[0].clone();
            source.layers.slots_mut()[2] = alias;
        }
        publish_source(&source, &loading);
        drop(loading);
        let baseline = pool.fixture_funded_charge().unwrap();
        settle(&pool, baseline);
        let (sampler, preparation, run) = sampler(&pool);
        let plan = source.prepare_resident_copy().unwrap();
        let expected = values(&plan);
        let expected_controls = controls(&plan);
        let original_ids = operands(&plan)
            .iter()
            .map(|a| a.allocation_info().unwrap().unwrap().identity())
            .collect::<Vec<_>>();
        if aliases {
            assert_eq!(original_ids[0], original_ids[2]);
        }
        let (copied_sampler, slots, native) = admit(&plan, sampler.borrow_funded(), &pool);
        let slot_bytes = slots.retained_bytes();
        let protected = slots.protected_bytes();
        let (custody, scope) = native.into_parts();
        let roots = RefCell::new(Vec::with_capacity(original_ids.len() * 2));
        let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
        assert_eq!(roots.borrow().len(), original_ids.len() * 2);
        finish_native(&saved, scope, &roots);
        assert_eq!(saved.retained_slot_bytes(), slot_bytes);
        assert_eq!(saved.protected_slot_bytes(), protected);
        assert_eq!(saved.global_layer_start(), 13);
        assert!(saved.shared_layout().same_storage(&source.layout));
        let copy_plan = saved.prepare_copy().unwrap();
        assert_eq!(values(&copy_plan), expected);
        assert_eq!(controls(&copy_plan), expected_controls);
        let copy_ids = operands(&copy_plan)
            .iter()
            .map(|a| a.allocation_info().unwrap().unwrap().identity())
            .collect::<Vec<_>>();
        assert!(copy_ids.iter().all(|id| !original_ids.contains(id)));
        assert_eq!(
            copy_ids
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            copy_ids.len()
        );
        drop((source, sampler, preparation, run));
        // Saved-to-saved preparation borrows the funded slots; the old live
        // table registration and any inference request are no longer required.
        let plan = saved.prepare_copy().unwrap();
        let (second_sampler, slots, native) = admit(&plan, copied_sampler.borrow_funded(), &pool);
        let (second_custody, scope) = native.into_parts();
        let duplicate = plan.copy_retained(slots, &stream, &roots).unwrap();
        finish_native(&duplicate, scope, &roots);
        assert_eq!(values(&duplicate.prepare_copy().unwrap()), expected);
        assert_eq!(
            controls(&duplicate.prepare_copy().unwrap()),
            expected_controls
        );
        drop((saved, copied_sampler, custody));
        assert_eq!(values(&duplicate.prepare_copy().unwrap()), expected);
        drop((duplicate, second_sampler, second_custody));
        settle(&pool, 0);
    }
}

#[test]
fn funded_builder_mismatch_rejects_before_copy_and_late_failure_retains_partial_roots() {
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = metal();
    let source = state(&stream);
    let other = state(&stream);
    publish_source(&source, &loading);
    publish_source(&other, &loading);
    drop(loading);
    let baseline = pool.fixture_funded_charge().unwrap();
    settle(&pool, baseline);
    let (sampler, preparation, run) = sampler(&pool);
    let expected = values(&source.prepare_resident_copy().unwrap());
    let roots = RefCell::new(Vec::with_capacity(6));
    let (copied_sampler, slots, native) = admit(
        &source.prepare_resident_copy().unwrap(),
        sampler.borrow_funded(),
        &pool,
    );
    let (custody, scope) = native.into_parts();
    let error = other
        .prepare_resident_copy()
        .unwrap()
        .copy_retained(slots, &stream, &roots)
        .err()
        .unwrap();
    assert!(matches!(
        error,
        ResidentKvCopyError::Memory(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(roots.borrow().is_empty());
    scope.certify().unwrap(); // Explicitly no native entry occurred.
    drop((copied_sampler, custody));

    let (copied_sampler, slots, native) = admit(
        &source.prepare_resident_copy().unwrap(),
        sampler.borrow_funded(),
        &pool,
    );
    let (custody, scope) = native.into_parts();
    FAIL_AFTER_LAYER.with(|fail| fail.set(Some(0)));
    let error = source
        .prepare_resident_copy()
        .unwrap()
        .copy_retained(slots, &stream, &roots)
        .err()
        .unwrap();
    assert!(
        matches!(&error, ResidentKvCopyError::Native(error) if error.to_string().contains("injected resident decoder layer-copy failure"))
    );
    assert_eq!(roots.borrow().len(), 4);
    assert_eq!(values(&source.prepare_resident_copy().unwrap()), expected);
    drop((source, other, sampler, preparation, run));
    // The partial table is gone, but each temporary/final descriptor remains
    // available to the caller's exact settlement/recovery path.
    for array in roots.borrow().iter() {
        array.evaluated().unwrap();
    }
    assert!(!roots.borrow()[1]
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .is_empty());
    roots.borrow_mut().clear();
    scope.certify().unwrap(); // Test caller has now settled every submitted root.
    drop((copied_sampler, custody));
    settle(&pool, 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::{FundingFixture as _, StorageFixture as _};
