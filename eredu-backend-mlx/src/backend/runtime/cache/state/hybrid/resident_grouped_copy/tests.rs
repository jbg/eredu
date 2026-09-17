use super::*;
use crate::backend::runtime::cache::kv::KeyValueCache;
use crate::backend::{
    managed_memory::NativeMemoryOwner, nn::workspace::MlxMetalWorkspaceMechanisms,
    runtime::residency::storage::RetainedStorage,
};
use eredu_core::{
    Admission, AttentionPolicy, EstimationCompleteness, ExecutionWorkspaceEstimate,
    InferenceGeometry, InputTokenCount, LayerSchedule, OutputDemand, StateMemoryLayout,
    TextGenerationConfig, WorkspaceBound,
    cache::{MutableStateResidency, StateTensorDtype, StateTensorPolicy},
};
use eredu_nn::workspace::WorkspaceIsolatedCopyPlan;
use eredu_nn::{CompressedAttentionCache, CompressedAttentionState};
use eredu_runtime::{
    SharedHostMetadata,
    working_memory::{
        AdmittedWorkspaceCopy, BorrowedFundedSampler, FundedSamplerCopy,
        InferenceExecutionIdentity, InferenceRequest, InferenceTextPreparation,
        RegisteredSamplingCopy, RegisteredWorkspaceCopy, RegisteredWorkspaceStorage,
        RunOwnedTextSampler, WorkingMemoryFundingRun, WorkingMemoryFundingScope,
        WorkspaceCopyLimits,
    },
};
use safemlx::{Device, DeviceType, Dtype};
use std::{cell::Cell, collections::BTreeSet};
thread_local! {static FAIL_FIXED:Cell<bool>=const{Cell::new(false)};}
#[derive(Debug, thiserror::Error)]
#[error("injected error after actual grouped fixed copy")]
struct FixedCopyFailure;
pub(super) fn after_fixed_copy() -> Result<(), Error> {
    if FAIL_FIXED.replace(false) {
        Err(other(FixedCopyFailure))
    } else {
        Ok(())
    }
}
fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Gpu, 0))
}
fn settle(pool: &WorkingMemoryPool, bytes: u64) {
    crate::backend::submission_recovery::wait_for_retirement(|| {
        crate::backend::ordinary_retirement::reclaim_all();
        safemlx::reclaim_allocation_owners();
        pool.used_bytes().unwrap() == bytes && pool.unquoted_owner_count().unwrap() == 0
    });
}
fn tensor(role: StateTensorRole, dtype: StateTensorDtype) -> StateTensorPolicy {
    StateTensorPolicy::new(
        role,
        vec![StateTensorDimension::fixed(2).unwrap()],
        dtype,
        if role == StateTensorRole::Recurrent {
            MutableStateResidency::LayerScopedOffloadable
        } else {
            MutableStateResidency::AlwaysDeviceMutable
        },
    )
    .unwrap()
}
fn source(stream: &Stream) -> MlxHybridState {
    let layout = StateLayout::new(
        LayerSchedule::new(
            5,
            vec![
                LayerCachePolicy::key_value_with_fixed_state(
                    AttentionPolicy::Full,
                    1,
                    2,
                    vec![
                        tensor(StateTensorRole::Recurrent, StateTensorDtype::Float32),
                        tensor(StateTensorRole::PrefixEmbedding, StateTensorDtype::Float32)
                            .optional(),
                    ],
                )
                .unwrap(),
                LayerCachePolicy::fixed_only(vec![
                    tensor(
                        StateTensorRole::Convolution { slot: 1 },
                        StateTensorDtype::Float32,
                    ),
                    tensor(StateTensorRole::PositionDelta, StateTensorDtype::Int32),
                ])
                .unwrap(),
                LayerCachePolicy::NoState,
                LayerCachePolicy::key_only(AttentionPolicy::sliding(8).unwrap(), 1, 2).unwrap(),
                LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 8, 4).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let mut state = MlxHybridState::device_with_global_layer_start(layout, 7).unwrap();
    let values = Array::from_slice(&[1_f32, 3., 5., 7., 9., 11.], &[1, 1, 3, 2]);
    for i in [0, 3] {
        let Some(MlxHybridAttentionState::KeyValue(MlxKeyValueLayerState::Device(cache))) =
            &mut state.layers.slots_mut()[i].attention
        else {
            unreachable!()
        };
        cache
            .update_and_fetch(values.clone(), values.clone(), stream)
            .unwrap();
    }
    let fixed = MlxTensor::from_array(Array::from_slice(&[2.5_f32, 8.25], &[2]));
    *state.layers.slots_mut()[0]
        .fixed
        .get_mut(&StateTensorRole::Recurrent)
        .unwrap() = Some(fixed.clone());
    *state.layers.slots_mut()[1]
        .fixed
        .get_mut(&StateTensorRole::Convolution { slot: 1 })
        .unwrap() = Some(fixed);
    *state.layers.slots_mut()[1]
        .fixed
        .get_mut(&StateTensorRole::PositionDelta)
        .unwrap() = Some(MlxTensor::from_array(Array::from_slice(
        &[11_i32, -3],
        &[2],
    )));
    state.layers.slots_mut()[1].fixed_offset = 3;
    state.layers.slots_mut()[2].fixed_offset = 3;
    let Some(MlxHybridAttentionState::Compressed(cache)) =
        &mut state.layers.slots_mut()[4].attention
    else {
        unreachable!()
    };
    cache
        .update_and_fetch(
            Array::from_slice(&(1..=24).map(|n| n as f32).collect::<Vec<_>>(), &[1, 3, 8]),
            Array::from_slice(&(31..=42).map(|n| n as f32).collect::<Vec<_>>(), &[1, 3, 4]),
            stream,
        )
        .unwrap();
    *cache = cache.deep_clone_state().unwrap();
    for array in state.retained_arrays() {
        array.evaluated().unwrap();
    }
    state
}
fn publish(source: &MlxHybridState, loading: &NativeMemoryOwner) {
    let mut storage = RetainedStorage::default();
    for array in source.retained_arrays() {
        storage.include_array(array).unwrap();
    }
    storage
        .include_metadata(SharedHostMetadata::Layout(source.layout.clone()))
        .unwrap();
    storage
        .include_slot_metadata(source.layers.metadata().clone())
        .unwrap();
    for layer in source.layers.slots() {
        storage
            .include_slot_metadata(layer.fixed.metadata().clone())
            .unwrap();
    }
    drop(storage.publish_unquoted(loading).unwrap());
}
#[derive(Debug, PartialEq)]
enum Values {
    Float(Vec<f32>),
    Int(Vec<i32>),
}
fn values(plan: &PreparedHybridGroupedCopy<'_>) -> Vec<Values> {
    let mut out = Vec::new();
    plan.visit_operands(&mut |a| {
        out.push(match a.dtype() {
            Dtype::Float32 => Values::Float(a.evaluated().unwrap().try_to_vec::<f32>().unwrap()),
            Dtype::Int32 => Values::Int(a.evaluated().unwrap().try_to_vec::<i32>().unwrap()),
            _ => panic!("fixture dtype"),
        })
    });
    out
}
fn controls(plan: &PreparedHybridGroupedCopy<'_>) -> Vec<(i32, Vec<(StateTensorRole, bool)>)> {
    (0..plan.len())
        .map(|i| {
            let l = plan.layer(i).unwrap();
            (
                l.attention
                    .map_or(l.fixed_offset, MlxHybridAttentionState::offset),
                l.fixed.iter().map(|(r, v)| (*r, v.is_some())).collect(),
            )
        })
        .collect()
}
// Same closed bootstrap as the portable runtime account fixture: this explicit
// 1024-byte source envelope covers only a scalar sampler's host construction.
// Native decoder copying below uses the selected Metal facts, never this number.
fn sampler(
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

fn prepared<'a>(
    plan: &PreparedHybridGroupedCopy<'a>,
    sampler: BorrowedFundedSampler<'a>,
    pool: &WorkingMemoryPool,
) -> (
    RegisteredSamplingCopy<'a, StorageIdentity>,
    PreparedHybridGroupHostCopy<'a>,
    eredu_runtime::working_memory::WorkingMemoryStorage<StorageIdentity>,
    u64,
) {
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let mut projection = ExistingArrayProjection::new(&context);
    let mut inputs = Vec::new();
    plan.visit_operands(&mut |a| inputs.push(projection.project(a).unwrap()));
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
    let numerical = program.incremental_bytes().unwrap();
    let proof = RegisteredWorkspaceCopy::bind(program, source).unwrap();
    let mut complete = RetainedStorage::default();
    plan.visit_retained_arrays(&mut |a| complete.include_array(a).unwrap());
    complete
        .include_metadata(SharedHostMetadata::Layout(plan.shared_layout().clone()))
        .unwrap();
    let complete = complete.pin_registered(pool).unwrap();
    let host = plan.host_copy(pool).unwrap();
    let sampling = RegisteredSamplingCopy::prepare(sampler, proof).unwrap();
    let required = host
        .initialization_peak_bytes()
        .checked_add(sampling.required_bytes())
        .unwrap();
    assert!(required >= numerical + host.initialization_peak_bytes());
    (sampling, host, complete, required)
}
fn finish(
    saved: &SavedHybridGroupedCopy,
    scope: WorkingMemoryFundingScope,
    roots: &RefCell<Vec<Array>>,
) {
    for a in roots.borrow().iter() {
        a.evaluated().unwrap();
    }
    let mut storage = RetainedStorage::default();
    saved
        .prepare_copy()
        .unwrap()
        .visit_retained_arrays(&mut |a| storage.include_array(a).unwrap());
    drop(storage.publish_funded(&scope).unwrap());
    roots.borrow_mut().clear();
    scope.certify().unwrap();
}

#[test]
fn mixed_fixed_roles_and_absence_have_real_child_extents_and_copied_workspace() {
    let stream = stream();
    let source = source(&stream);
    let plan = PreparedHybridGroupedCopy::prepare(&source).unwrap();
    assert!(PreparedHybridGroupedCopy::requires_group(&source));
    assert_eq!(
        (0..plan.len())
            .map(|i| plan.layer(i).unwrap().fixed.len())
            .collect::<Vec<_>>(),
        [2, 2, 0, 0, 0]
    );
    assert_eq!(
        controls(&plan).iter().map(|x| x.0).collect::<Vec<_>>(),
        [3, 3, 3, 3, 3]
    );
    assert!(
        plan.layer(0)
            .unwrap()
            .fixed
            .iter()
            .any(|(r, v)| *r == StateTensorRole::PrefixEmbedding && v.is_none())
    );
    for i in 0..2 {
        let Fixed::Live(f) = plan.layer(i).unwrap().fixed else {
            unreachable!()
        };
        assert_eq!(
            f.payload_bytes(),
            Some(2 * std::mem::size_of::<Slot>() as u64)
        );
        assert!(
            f.prepare_slots().unwrap().initialization_peak_bytes() > f.payload_bytes().unwrap()
        );
    }
    let mut retained_ids = BTreeSet::new();
    plan.visit_retained_arrays(&mut |a| {
        retained_ids.insert(a.allocation_info().unwrap().unwrap().identity());
    });
    let mut operand_ids = Vec::new();
    plan.visit_operands(&mut |a| {
        operand_ids.push(a.allocation_info().unwrap().unwrap().identity())
    });
    assert!(
        retained_ids.len() > operand_ids.iter().collect::<BTreeSet<_>>().len(),
        "uncopied compressed stores remain real sources"
    );
    let context = WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap());
    let copied = plan
        .project_dense_workspace(NonZeroU32::new(1).unwrap(), &context)
        .unwrap();
    assert_eq!(copied.source_storage.iter().len(), retained_ids.len());
    assert!(copied.copy.total_bytes.is_some());
    let ordinary = source
        .project_resident_workspace(
            NonZeroU32::new(1).unwrap(),
            &WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap()),
        )
        .unwrap();
    assert_eq!(copied.state.layout(), ordinary.layout());
    assert_eq!(
        controls(&plan).iter().map(|x| x.0).collect::<Vec<_>>(),
        [3, 3, 3, 3, 3]
    );
}

#[test]
fn exact_group_copy_preserves_fixed_integer_aliases_and_recopies_after_original_retirement() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = stream();
    let source = source(&stream);
    publish(&source, &loading);
    drop(loading);
    settle(&pool, pool.used_bytes().unwrap());
    let (sampler, preparation, run) = sampler(&pool);
    let plan = PreparedHybridGroupedCopy::prepare(&source).unwrap();
    let original = values(&plan);
    let original_controls = controls(&plan);
    let (mut sampling, mut host, mut complete, required) =
        prepared(&plan, sampler.borrow_funded(), &pool);
    let before = (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap());
    assert!(
        host.admit(
            &pool,
            sampling,
            complete,
            WorkspaceCopyLimits::new(pool.used_bytes().unwrap().checked_add(required).unwrap() - 1)
        )
        .is_err()
    );
    assert_eq!(
        (pool.used_bytes().unwrap(), pool.peak_bytes().unwrap()),
        before
    );
    assert_eq!(values(&plan), original);
    (sampling, host, complete, _) = prepared(&plan, sampler.borrow_funded(), &pool);
    let (copied_sampler, slots, account) = host
        .admit(
            &pool,
            sampling,
            complete,
            WorkspaceCopyLimits::new(pool.used_bytes().unwrap().checked_add(required).unwrap()),
        )
        .unwrap();
    let protected = slots.protected_bytes();
    let retained = slots.retained_bytes();
    let (custody, scope) = account.into_parts();
    let roots = RefCell::new(Vec::new());
    let saved = plan.copy_retained(slots, &stream, &roots).unwrap();
    finish(&saved, scope, &roots);
    assert_eq!(
        (saved.protected_slot_bytes(), saved.retained_slot_bytes()),
        (protected, retained)
    );
    assert!(protected > retained);
    assert_eq!(values(&saved.prepare_copy().unwrap()), original);
    assert_eq!(controls(&saved.prepare_copy().unwrap()), original_controls);
    let mut destination_ids = Vec::new();
    saved.prepare_copy().unwrap().visit_operands(&mut |a| {
        destination_ids.push(a.allocation_info().unwrap().unwrap().identity())
    });
    assert_eq!(
        destination_ids.iter().collect::<BTreeSet<_>>().len(),
        destination_ids.len()
    );
    drop((source, sampler, preparation, run));
    // Certified native work released original registration pins. The saved
    // account holds its whole destination envelope; only shared layout remains
    // from original independently registered source metadata.
    settle(
        &pool,
        custody.bytes() + saved.shared_layout().capacity_bytes().unwrap(),
    );
    let plan = saved.prepare_copy().unwrap();
    let (sampling, host, complete, required) =
        prepared(&plan, copied_sampler.borrow_funded(), &pool);
    let (duplicate_sampler, slots, account) = host
        .admit(
            &pool,
            sampling,
            complete,
            WorkspaceCopyLimits::new(pool.used_bytes().unwrap().checked_add(required).unwrap()),
        )
        .unwrap();
    let (duplicate_custody, scope) = account.into_parts();
    let duplicate = plan.copy_retained(slots, &stream, &roots).unwrap();
    finish(&duplicate, scope, &roots);
    assert_eq!(values(&duplicate.prepare_copy().unwrap()), original);
    assert_eq!(
        controls(&duplicate.prepare_copy().unwrap()),
        original_controls
    );
    assert_eq!(duplicate.global_layer_start(), 7);
    drop((saved, copied_sampler, custody));
    assert_eq!(values(&duplicate.prepare_copy().unwrap()), original);
    drop((duplicate, duplicate_sampler, duplicate_custody, roots));
    settle(&pool, 0);
}

#[test]
fn exact_outer_mismatch_rejects_before_copy_and_late_fixed_error_keeps_native_recovery_roots() {
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let stream = stream();
    let source = source(&stream);
    let other_source = source.clone();
    publish(&source, &loading);
    publish(&other_source, &loading);
    drop(loading);
    settle(&pool, pool.used_bytes().unwrap());
    let (sampler, preparation, run) = sampler(&pool);
    let plan = PreparedHybridGroupedCopy::prepare(&source).unwrap();
    let (sampling, host, complete, required) = prepared(&plan, sampler.borrow_funded(), &pool);
    let (copied_sampler, slots, account) = host
        .admit(
            &pool,
            sampling,
            complete,
            WorkspaceCopyLimits::new(pool.used_bytes().unwrap().checked_add(required).unwrap()),
        )
        .unwrap();
    let (custody, scope) = account.into_parts();
    let roots = RefCell::new(Vec::new());
    let wrong = PreparedHybridGroupedCopy::prepare(&other_source)
        .unwrap()
        .copy_retained(slots, &stream, &roots);
    assert!(
        matches!(wrong,Err(Error::Other(ref e)) if e.downcast_ref::<WorkingMemoryError>()==Some(&WorkingMemoryError::IdentityMismatch))
    );
    assert!(roots.borrow().is_empty());
    scope.certify().unwrap();
    drop((copied_sampler, custody));
    let plan = PreparedHybridGroupedCopy::prepare(&source).unwrap();
    let expected = values(&plan);
    let (sampling, host, complete, required) = prepared(&plan, sampler.borrow_funded(), &pool);
    let (copied_sampler, slots, account) = host
        .admit(
            &pool,
            sampling,
            complete,
            WorkspaceCopyLimits::new(pool.used_bytes().unwrap().checked_add(required).unwrap()),
        )
        .unwrap();
    let (custody, scope) = account.into_parts();
    FAIL_FIXED.set(true);
    let failed = plan.copy_retained(slots, &stream, &roots);
    assert!(
        matches!(failed,Err(Error::Other(ref e)) if e.downcast_ref::<FixedCopyFailure>().is_some())
    );
    assert!(!FAIL_FIXED.get());
    assert!(
        roots.borrow().len() >= 6,
        "two KV roles and one real fixed numerical copy survived"
    );
    assert_eq!(
        values(&PreparedHybridGroupedCopy::prepare(&source).unwrap()),
        expected
    );
    for a in roots.borrow().iter() {
        a.evaluated().unwrap();
    }
    roots.borrow_mut().clear();
    scope.certify().unwrap();
    drop((
        copied_sampler,
        custody,
        source,
        other_source,
        sampler,
        preparation,
        run,
        roots,
    ));
    settle(&pool, 0);
}

#[test]
fn borrowed_hybrid_whole_visit_covers_fixed_shared_key_only_and_compressed_slots() {
    thread_local! { static CALLS:Cell<usize>=const {Cell::new(0)}; }
    fn housekeeping() {
        CALLS.set(CALLS.get() + 1);
    }
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            safemlx::unregister_thread_runtime_housekeeping(housekeeping);
        }
    }
    let stream = stream();
    let state = source(&stream);
    let expected = state.retained_arrays();
    assert_eq!(expected.len(), 10);
    let before = state.semantic_snapshot();
    safemlx::register_thread_runtime_housekeeping(housekeeping);
    let guard = Guard;
    CALLS.set(0);
    let mut total = 0;
    state
        .visit_all_retained_values(&mut |value| {
            assert!(std::ptr::eq(value.as_array(), expected[total]));
            total += 1;
        })
        .unwrap();
    let mut per_layer = [0; 5];
    for (index, layer) in state.layers.slots().iter().enumerate() {
        RuntimeLayerState::<MlxNeuralBackend>::visit_retained_values(layer, &mut |_| {
            per_layer[index] += 1
        });
    }
    assert_eq!(total, 10);
    assert_eq!(per_layer, [3, 2, 0, 1, 4]);
    assert_eq!(CALLS.get(), 0);
    drop(guard);
    assert_eq!(state.semantic_snapshot(), before);
}

pub(super) mod original_discard;
