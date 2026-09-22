use super::*;
use crate::backend::{
    managed_memory::NativeMemoryOwner,
    nn::workspace::{ExistingArrayProjection, MlxMetalWorkspaceMechanisms},
    runtime::{cache::state::PreparedResidentDecoderCopy, residency::storage::RetainedStorage},
};
use crate::memory_fixture::LedgerFixture;
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

pub(in super::super) fn sampling_plan<'a>(
    plan: &PreparedResidentPoolingCopy<'_>,
    sampler: BorrowedFundedSampler<'a>,
    pool: &MemoryLedger,
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
            .map(|(id, _, root)| crate::backend::nn::workspace::registered_storage_row(id, root)),
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
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
    let baseline = pool.fixture_funded_charge().unwrap();
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
    let copy_limits =
        crate::memory_fixture::publication_copy_limits(&pool, operands(&plan).len(), u64::MAX);
    let payload = joined
        .required_bytes()
        .unwrap()
        .checked_add(copy_limits.additional_host_metadata_bytes)
        .unwrap();
    let required = crate::memory_fixture::host_total(
        &pool
            .text_components_copy_requirements(&joined, &copy_limits)
            .unwrap(),
    );
    let physical_before = pool.fixture_host_current().unwrap();
    let before = (
        pool.fixture_funded_charge().unwrap(),
        pool.fixture_host_peak().unwrap(),
    );
    let error = pool
        .copy_text_components(
            joined,
            crate::memory_fixture::publication_copy_limits(
                &pool,
                operands(&plan).len(),
                physical_before.checked_add(required).unwrap() - 1,
            ),
        )
        .err()
        .unwrap();
    assert!(
        matches!(error,DecoderCopyAdmissionError::Memory(WorkingMemoryError::Domain(eredu_core::MemoryDomainError::BudgetExceeded { requested_bytes: required_bytes, limit_bytes, existing_bytes, .. })) if required_bytes==required && limit_bytes.checked_sub(existing_bytes).unwrap()==required-1)
    );
    assert_eq!(
        (
            pool.fixture_funded_charge().unwrap(),
            pool.fixture_host_peak().unwrap()
        ),
        before
    );
    let (sampling, complete) = sampling_plan(&plan, sampler.borrow_funded(), &pool);
    let joined = sampling
        .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
        .unwrap();
    let (copied_sampler, slots, native) = pool
        .copy_text_components(
            joined,
            crate::memory_fixture::publication_copy_limits(&pool, operands(&plan).len(), u64::MAX),
        )
        .unwrap();
    let (custody, scope) = native.into_parts();
    let roots = RefCell::new(Vec::with_capacity(ids.len() * 2));
    let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
    assert_eq!(roots.borrow().len(), ids.len() * 2);
    finish_native(&saved, scope, &roots);
    assert!(
        pool.fixture_host_current().unwrap() <= physical_before + required,
        "completion retires temporary preparation charges within the admitted peak"
    );
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
        custody
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap()
            + saved.shared_layout().capacity_bytes().unwrap()
            - crate::memory_fixture::publication_control_bytes(
                operands(&saved.prepare_copy().unwrap())
                    .len()
                    .checked_mul(2)
                    .unwrap(),
            ),
    );
    let plan = saved.prepare_copy().unwrap();
    let (sampling, complete) = sampling_plan(&plan, copied_sampler.borrow_funded(), &pool);
    let joined = sampling
        .with_decoder_slots(plan.host_copy(&pool).unwrap(), complete)
        .unwrap();
    let copy_limits =
        crate::memory_fixture::publication_copy_limits(&pool, operands(&plan).len(), u64::MAX);
    let payload = joined
        .required_bytes()
        .unwrap()
        .checked_add(copy_limits.additional_host_metadata_bytes)
        .unwrap();
    let required = crate::memory_fixture::host_total(
        &pool
            .text_components_copy_requirements(&joined, &copy_limits)
            .unwrap(),
    );
    let physical_before = pool.fixture_host_current().unwrap();
    let before = pool.fixture_funded_charge().unwrap();
    let (next_sampler, slots, next_native) = pool
        .copy_text_components(
            joined,
            crate::memory_fixture::publication_copy_limits(
                &pool,
                operands(&plan).len(),
                physical_before.checked_add(required).unwrap(),
            ),
        )
        .unwrap();
    let (next_custody, scope) = next_native.into_parts();
    let duplicate = plan.copy_retained(slots, &stream, &roots).unwrap();
    finish_native(&duplicate, scope, &roots);
    assert!(
        pool.fixture_host_current().unwrap() <= physical_before + required,
        "completion retires temporary preparation charges within the admitted peak"
    );
    drop((saved, copied_sampler, custody));
    assert_eq!(values(&duplicate.prepare_copy().unwrap()), expected);
    assert_eq!(
        controls(&duplicate.prepare_copy().unwrap()),
        expected_controls
    );
    let escaped = operands(&duplicate.prepare_copy().unwrap())[0].clone();
    let info = escaped.allocation_info().unwrap().unwrap();
    let escaped_bytes = (info.bytes() as u64)
        .checked_add(info.host_control_bytes() as u64)
        .unwrap();
    let account_controls = required.checked_sub(payload).unwrap();
    drop((duplicate, next_sampler, next_custody));
    settle(&pool, escaped_bytes + account_controls);
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
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
            crate::memory_fixture::publication_copy_limits(&pool, operands(&plan).len(), u64::MAX),
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
            crate::memory_fixture::publication_copy_limits(&pool, operands(&plan).len(), u64::MAX),
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
        let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
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
                crate::memory_fixture::publication_copy_limits(
                    &pool,
                    operands(&plan).len(),
                    u64::MAX,
                ),
            )
            .unwrap();
        let aggregate = native
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap();
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
            })
            .unwrap();
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
