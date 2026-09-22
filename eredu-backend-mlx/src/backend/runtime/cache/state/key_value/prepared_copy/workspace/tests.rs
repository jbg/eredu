use super::super::tests::{
    funded::{admit, finish_native, metal, publish_source, sampler, settle},
    operands, values,
};
use super::*;
use crate::backend::{
    managed_memory::NativeMemoryOwner,
    nn::workspace::MlxMetalWorkspaceMechanisms,
    runtime::cache::{
        kv::{ConcatKeyValueCache, KeyValueCache},
        state::{MlxKeyValueLayerState, MlxKeyValueState},
    },
};
use crate::memory_fixture::LedgerFixture;
use eredu_core::{cache::LayerCachePolicy, AttentionPolicy, LayerSchedule};
use eredu_nn::{
    workspace::{
        WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound, WorkspaceOperationKind,
        WorkspaceTensor,
    },
    AttentionCache, Tensor,
};
use eredu_runtime::{
    working_memory::MemoryLedger, RuntimeLayerState, RuntimeState, RuntimeStateComponents,
    StateLayout,
};
use safemlx::{ops::indexing::TryIndexOp, Array, Stream};
use std::{
    cell::{Cell, RefCell},
    collections::BTreeMap,
};

type ProjectedState = DeviceState<WorkspaceBackend, WorkspaceResidentLayerState>;
fn nz(n: u32) -> NonZeroU32 {
    NonZeroU32::new(n).unwrap()
}
fn new_context() -> WorkspaceContext {
    WorkspaceContext::new(MlxMetalWorkspaceMechanisms::current_host().unwrap())
}
fn input(positions: i32, seed: f32) -> Array {
    Array::from_slice(
        &(0..4 * positions)
            .map(|n| seed + n as f32 * 0.25)
            .collect::<Vec<_>>(),
        &[2, 1, positions, 2],
    )
}
fn layout(attention: AttentionPolicy) -> StateLayout {
    StateLayout::new(
        LayerSchedule::new(
            3,
            vec![
                LayerCachePolicy::key_value(attention.clone(), 1, 2).unwrap(),
                LayerCachePolicy::NoState,
                LayerCachePolicy::key_value(attention, 1, 2).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap()
}
fn aliased_source(stream: &Stream) -> MlxKeyValueState {
    let mut source =
        MlxKeyValueState::device_with_global_layer_start(layout(AttentionPolicy::Full), 17)
            .unwrap();
    let backing = input(4096, 0.5);
    let keys = backing
        .try_index_device((.., .., 3..6, ..), stream)
        .unwrap();
    let vals = backing
        .try_index_device((.., .., 7..10, ..), stream)
        .unwrap();
    keys.evaluated().unwrap();
    vals.evaluated().unwrap();
    let mut cache = ConcatKeyValueCache::new();
    cache.restore_resident(keys, vals, 3).unwrap();
    source.layers.slots_mut()[0] = MlxKeyValueLayerState::Device(cache.clone());
    source.layers.slots_mut()[2] = MlxKeyValueLayerState::Device(cache);
    source
}
fn roots(state: &ProjectedState) -> Vec<WorkspaceTensor> {
    state
        .as_ref()
        .iter()
        .flat_map(RuntimeLayerState::<WorkspaceBackend>::retained_values)
        .cloned()
        .collect()
}
fn native_facts(state: &MlxKeyValueState) -> BTreeMap<safemlx::AllocationIdentity, u64> {
    state
        .retained_arrays()
        .into_iter()
        .map(|array| {
            array.evaluated().unwrap();
            let allocation = array.allocation_info().unwrap().unwrap();
            (allocation.identity(), allocation.bytes() as u64)
        })
        .collect()
}
fn positions(state: &MlxKeyValueState) -> Vec<i32> {
    state
        .layers
        .slots()
        .iter()
        .map(KeyValueCache::offset)
        .collect()
}
fn check_shapes(state: &ProjectedState, native: &MlxKeyValueState) {
    let actual = native
        .retained_arrays()
        .into_iter()
        .map(|a| a.shape().to_vec())
        .collect::<Vec<_>>();
    let projected = roots(state)
        .iter()
        .map(|a| a.shape().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(projected, actual);
    assert_eq!(
        state
            .as_ref()
            .iter()
            .map(|layer| layer.position())
            .collect::<Vec<_>>(),
        positions(native)
    );
}

thread_local! { static HOUSEKEEPING: Cell<usize> = const { Cell::new(0) }; }
fn housekeeping() {
    HOUSEKEEPING.set(HOUSEKEEPING.get() + 1);
}
struct ColdGuard;
impl ColdGuard {
    fn new() -> Self {
        safemlx::register_thread_runtime_housekeeping(housekeeping);
        HOUSEKEEPING.set(0);
        Self
    }
    fn assert_cold(&self) {
        assert_eq!(
            HOUSEKEEPING.get(),
            0,
            "projection must not enter native housekeeping/evaluation"
        );
    }
}
impl Drop for ColdGuard {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(housekeeping);
    }
}

#[test]
fn large_aliased_source_projects_independent_bounded_destinations_without_native_work() {
    let stream = metal();
    let source = aliased_source(&stream);
    let plan = source.prepare_resident_copy().unwrap();
    let expected = values(&plan);
    let old_positions = positions(&source);
    let old_facts = native_facts(&source);
    assert_eq!(old_facts.len(), 1);
    let source_bytes = *old_facts.values().next().unwrap();
    let source_controls = u64::try_from(
        operands(&plan)[0]
            .allocation_info()
            .unwrap()
            .unwrap()
            .host_control_bytes(),
    )
    .unwrap();
    assert!(source_bytes >= 65536);
    let context = new_context();
    let guard = ColdGuard::new();
    let projected = plan.project_dense_workspace(nz(2), &context).unwrap();
    assert!(projected.source_storage.is_complete());
    assert_eq!(
        projected
            .source_storage
            .iter()
            .map(|(id, bytes, _)| (id, bytes))
            .collect::<BTreeMap<_, _>>(),
        old_facts
    );
    assert_eq!(projected.copy.operations.len(), 8);
    for pair in projected.copy.operations.chunks_exact(2) {
        assert!(matches!(pair[0].kind, WorkspaceOperationKind::Contiguous));
        assert!(matches!(pair[1].kind, WorkspaceOperationKind::DeepCopy));
    }
    assert!(projected.copy.total_bytes.is_some());
    assert!(projected.copy.unpriced_operations.is_empty());
    assert!(projected.copy.unpriced_host_operations.is_empty());
    assert_eq!(projected.state.layout(), source.layout.layout());
    let destination = roots(&projected.state);
    assert_eq!(destination.len(), 4);
    context.begin_state_span(&destination).unwrap();
    let future = context.report(&destination).unwrap();
    assert!(future.operations.is_empty());
    let future_bytes = future.state.unwrap().retained_bytes.unwrap();
    let individual_bytes: u64 = destination
        .iter()
        .map(|root| {
            context
                .report(std::slice::from_ref(root))
                .unwrap()
                .state
                .unwrap()
                .retained_bytes
                .unwrap()
        })
        .sum();
    assert_eq!(
        future_bytes, individual_bytes,
        "each logical copy owns a distinct output root"
    );
    assert!(
        future_bytes < source_bytes,
        "future state must use actual copy bounds, not the original huge backing"
    );
    assert_eq!(
        projected.copy.state.as_ref().unwrap().retained_bytes,
        Some(
            source_bytes
                .checked_add(future_bytes)
                .unwrap()
                .checked_add(source_controls)
                .unwrap()
        )
    );
    guard.assert_cold();
    drop(guard);
    let copied = source.isolated_snapshot(&stream).unwrap();
    assert_eq!(values(&copied.prepare_resident_copy().unwrap()), expected);
    let copied_facts = native_facts(&copied);
    assert_eq!(copied_facts.len(), 4);
    assert!(copied_facts.keys().all(|id| !old_facts.contains_key(id)));
    assert!(future_bytes >= copied_facts.values().sum());
    check_shapes(&projected.state, &copied);
    assert_eq!(positions(&source), old_positions);
    assert_eq!(native_facts(&source), old_facts);
    assert_eq!(values(&source.prepare_resident_copy().unwrap()), expected);
}

#[test]
fn projected_fresh_cache_appends_match_actual_isolated_copy_across_multiple_steps() {
    let stream = metal();
    for window in [None, Some(4), Some(1)] {
        let attention = window.map_or(AttentionPolicy::Full, |n| {
            AttentionPolicy::sliding(n).unwrap()
        });
        let mut source = MlxKeyValueState::device(layout(attention)).unwrap();
        for index in [0, 2] {
            KeyValueCache::update_for_attention(
                &mut source.layers.slots_mut()[index],
                input(3, 1.0),
                input(3, 101.0),
                &stream,
            )
            .unwrap();
        }
        for value in source.retained_arrays() {
            value.evaluated().unwrap();
        }
        let original = values(&source.prepare_resident_copy().unwrap());
        let context = new_context();
        let plan = source.prepare_resident_copy().unwrap();
        let mut projected = plan.project_dense_workspace(nz(2), &context).unwrap();
        let mut copied = source.isolated_snapshot(&stream).unwrap();
        check_shapes(&projected.state, &copied);
        for (step, count) in [1, 2, 1].into_iter().enumerate() {
            let opening = roots(&projected.state);
            context.begin_state_span(&opening).unwrap();
            for index in [0, 2] {
                let keys = input(count, 20.0 + step as f32);
                let vals = input(count, 120.0 + step as f32);
                keys.evaluated().unwrap();
                vals.evaluated().unwrap();
                let mut import = ExistingArrayProjection::new(&context);
                let projected_keys = import.project(&keys).unwrap();
                let projected_vals = import.project(&vals).unwrap();
                let (visible_keys, visible_vals) = projected.state.as_mut()[index]
                    .update_for_attention(projected_keys, projected_vals, &context)
                    .unwrap();
                let (native_keys, native_vals) = KeyValueCache::update_for_attention(
                    &mut copied.layers.slots_mut()[index],
                    keys,
                    vals,
                    &stream,
                )
                .unwrap();
                native_keys.evaluated().unwrap();
                native_vals.evaluated().unwrap();
                assert_eq!(visible_keys.shape(), native_keys.shape());
                assert_eq!(visible_vals.shape(), native_vals.shape());
            }
            check_shapes(&projected.state, &copied);
            let report = context.report(&roots(&projected.state)).unwrap();
            assert!(report.total_bytes.is_some());
            assert!(
                report.state.as_ref().unwrap().retained_bytes.unwrap()
                    >= native_facts(&copied).values().sum()
            );
            assert_eq!(positions(&source), vec![3, 0, 3]);
            assert_eq!(values(&source.prepare_resident_copy().unwrap()), original);
        }
        assert_eq!(positions(&copied), vec![7, 0, 7]);
    }
}

#[derive(Debug)]
struct Unpriced;
impl WorkspaceMechanisms for Unpriced {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, eredu_nn::Error> {
        Ok(None)
    }
}

#[test]
fn invalid_geometry_and_lazy_or_unpriced_sources_do_not_become_complete_copies() {
    let stream = metal();
    let source = aliased_source(&stream);
    let expected = values(&source.prepare_resident_copy().unwrap());
    for batch in [1, 3] {
        let context = new_context();
        let guard = ColdGuard::new();
        assert!(source
            .prepare_resident_copy()
            .unwrap()
            .project_dense_workspace(nz(batch), &context)
            .is_err());
        guard.assert_cold();
    }
    let mut padded = MlxKeyValueState::device(layout(AttentionPolicy::Full)).unwrap();
    let mut cache = ConcatKeyValueCache::new_with_max_size_and_step(16, 8);
    cache
        .update_and_fetch(input(3, 1.0), input(3, 101.0), &stream)
        .unwrap();
    padded.layers.slots_mut()[0] = MlxKeyValueLayerState::Device(cache);
    let context = new_context();
    let guard = ColdGuard::new();
    assert!(padded
        .prepare_resident_copy()
        .unwrap()
        .project_dense_workspace(nz(2), &context)
        .is_err());
    assert!(
        context.report(&[]).unwrap().operations.is_empty(),
        "selected non-exact cache rejects before the copy trace"
    );
    guard.assert_cold();
    drop(guard);
    let mut wrong_shape = MlxKeyValueState::device(layout(AttentionPolicy::Full)).unwrap();
    let mut cache = ConcatKeyValueCache::new();
    cache
        .restore_resident(
            input(3, 1.0).reshape(&[2, 2, 3, 1], &stream).unwrap(),
            input(3, 101.0).reshape(&[2, 2, 3, 1], &stream).unwrap(),
            3,
        )
        .unwrap();
    wrong_shape.layers.slots_mut()[0] = MlxKeyValueLayerState::Device(cache);
    assert!(wrong_shape
        .prepare_resident_copy()
        .unwrap()
        .project_dense_workspace(nz(2), &context)
        .is_err());
    let mut wrong_policy = MlxKeyValueState::device(layout(AttentionPolicy::Full)).unwrap();
    let mut cache = ConcatKeyValueCache::new_for_sliding_attention(4);
    cache
        .restore_resident(input(3, 1.0), input(3, 101.0), 3)
        .unwrap();
    wrong_policy.layers.slots_mut()[0] = MlxKeyValueLayerState::Device(cache);
    assert!(wrong_policy
        .prepare_resident_copy()
        .unwrap()
        .project_dense_workspace(nz(2), &context)
        .is_err());
    let unpriced = WorkspaceContext::new(Unpriced);
    let incomplete = source
        .prepare_resident_copy()
        .unwrap()
        .project_dense_workspace(nz(2), &unpriced)
        .unwrap();
    assert!(incomplete.source_storage.is_complete());
    assert!(incomplete.copy.total_bytes.is_none());
    assert!(!incomplete.copy.unpriced_operations.is_empty());
    assert!(!incomplete.copy.unpriced_host_operations.is_empty());
    let mut lazy = MlxKeyValueState::device(layout(AttentionPolicy::Full)).unwrap();
    let pending = input(3, 3.0).square(&stream).unwrap();
    assert!(pending
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none());
    let mut cache = ConcatKeyValueCache::new();
    cache
        .restore_resident(pending.clone(), pending.clone(), 3)
        .unwrap();
    lazy.layers.slots_mut()[0] = MlxKeyValueLayerState::Device(cache);
    let context = new_context();
    let guard = ColdGuard::new();
    let projected = lazy
        .prepare_resident_copy()
        .unwrap()
        .project_dense_workspace(nz(2), &context)
        .unwrap();
    assert!(!projected.source_storage.is_complete());
    assert!(projected
        .copy
        .state
        .as_ref()
        .unwrap()
        .retained_bytes
        .is_none());
    assert!(projected.copy.inference_transient_bytes().is_none());
    assert!(pending
        .try_metadata_snapshot()
        .unwrap()
        .allocation()
        .is_none());
    guard.assert_cold();
    drop(guard);
    // Known copy-output geometry can still be priced, but it cannot erase the
    // unknown actual source overlap from this complete copy report.
    assert_eq!(positions(&source), vec![3, 0, 3]);
    assert_eq!(values(&source.prepare_resident_copy().unwrap()), expected);
}

#[test]
fn funded_saved_source_projects_after_original_request_and_live_table_retire() {
    let stream = metal();
    let pool = crate::memory_fixture::ledger(u64::MAX, 0).unwrap();
    let loading = NativeMemoryOwner::acquire(&pool).unwrap();
    let source = aliased_source(&stream);
    publish_source(&source, &loading);
    drop(loading);
    settle(&pool, pool.fixture_funded_charge().unwrap());
    let (sampler, preparation, run) = sampler(&pool);
    let plan = source.prepare_resident_copy().unwrap();
    let expected = values(&plan);
    let (copied_sampler, slots, native) = admit(&plan, sampler.borrow_funded(), &pool);
    let (custody, scope) = native.into_parts();
    let collector = RefCell::new(Vec::new());
    let saved = plan.copy_retained(slots, &stream, &collector).unwrap();
    finish_native(&saved, scope, &collector);
    drop((source, sampler, preparation, run));
    settle(
        &pool,
        custody
            .requirements()
            .get(crate::memory_fixture::topology().host_domain())
            .unwrap()
            .total()
            .unwrap()
            .checked_sub(crate::memory_fixture::publication_control_bytes(
                operands(&saved.prepare_copy().unwrap())
                    .len()
                    .checked_mul(2)
                    .unwrap(),
            ))
            .unwrap()
            + saved.shared_layout().capacity_bytes().unwrap(),
    );
    let context = new_context();
    let before = (
        pool.fixture_funded_charge().unwrap(),
        pool.fixture_host_peak().unwrap(),
    );
    let guard = ColdGuard::new();
    let projected = saved
        .prepare_copy()
        .unwrap()
        .project_dense_workspace(nz(2), &context)
        .unwrap();
    assert!(projected.source_storage.is_complete());
    assert_eq!(
        projected.source_storage.iter().len(),
        4,
        "the first saved copy already split all aliases"
    );
    assert!(projected.copy.total_bytes.is_some());
    assert_eq!(
        (
            pool.fixture_funded_charge().unwrap(),
            pool.fixture_host_peak().unwrap()
        ),
        before
    );
    guard.assert_cold();
    drop(guard);
    assert_eq!(values(&saved.prepare_copy().unwrap()), expected);
    assert_eq!(roots(&projected.state).len(), 4);
    assert_eq!(
        projected
            .state
            .as_ref()
            .iter()
            .map(|layer| layer.position())
            .collect::<Vec<_>>(),
        vec![3, 0, 3]
    );
    drop((saved, copied_sampler, custody));
    // Inspection witnesses and the shared layout still own the old physical
    // source; no new allocation or funding account was created by projection.
    assert!(pool.fixture_funded_charge().unwrap() > 0);
    drop(projected);
    settle(&pool, 0);
}
