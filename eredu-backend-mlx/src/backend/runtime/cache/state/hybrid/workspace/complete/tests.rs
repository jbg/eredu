use super::*;
use crate::backend::runtime::cache::residency::CacheResidencyError;
use eredu_core::{
    cache::{MutableStateResidency, StateTensorDtype, StateTensorPolicy},
    AttentionPolicy, LayerSchedule,
};
use eredu_nn::{workspace::*, Error, Tensor};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger},
    CacheLifecycleError, PagedCacheOptions,
};
use safemlx::{Device, DeviceType};
use std::{cell::Cell, collections::BTreeMap};

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("projection must not execute equations")
    }
}
impl WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("projection cannot quote equations")
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Self::Error> {
        panic!("projection cannot emit equations")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("projection cannot quote host equations")
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Self::Error> {
        panic!("projection cannot emit host equations")
    }
}
thread_local! { static HOOKS: Cell<usize> = const { Cell::new(0) }; }
fn hook() {
    HOOKS.with(|n| n.set(n.get() + 1));
}
struct Hooks;
impl Drop for Hooks {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(hook);
    }
}
fn cold<T>(f: impl FnOnce() -> T) -> T {
    safemlx::register_thread_runtime_housekeeping(hook);
    let _hooks = Hooks;
    HOOKS.with(|n| n.set(0));
    let result = f();
    assert_eq!(HOOKS.with(Cell::get), 0);
    result
}

#[test]
#[ignore = "requires native CPU mixed cache sources"]
fn hybrid_paged_projection_preserves_fixed_aliases_integer_roles_and_compressed_capacity() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixed_role = StateTensorRole::Convolution { slot: 3 };
    let integer_role = StateTensorRole::PositionDelta;
    let fixed = StateTensorPolicy::new(
        fixed_role,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(5).unwrap(),
        ],
        StateTensorDtype::Float32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let integer = StateTensorPolicy::new(
        integer_role,
        vec![
            StateTensorDimension::Batch,
            StateTensorDimension::fixed(2).unwrap(),
        ],
        StateTensorDtype::Int32,
        MutableStateResidency::AlwaysDeviceMutable,
    )
    .unwrap();
    let layout = StateLayout::new(
        LayerSchedule::new(
            3,
            vec![
                LayerCachePolicy::key_value_with_fixed_state(
                    AttentionPolicy::Full,
                    1,
                    1,
                    vec![fixed],
                )
                .unwrap(),
                LayerCachePolicy::fixed_only(vec![integer]).unwrap(),
                LayerCachePolicy::compressed_latent_rotary(AttentionPolicy::Full, 2, 2).unwrap(),
            ],
        )
        .unwrap(),
    )
    .unwrap();
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut native =
        MlxHybridState::paged_with_global_layer_start(layout, manager.clone(), None, 7).unwrap();
    native.layers.slots_mut()[2].attention = Some(MlxHybridAttentionState::Compressed(
        CompressedLatentCache::new(),
    ));
    let input = Array::from_slice(&[0.25f32, 0.5, 0.75, 1.0, 1.25], &[1, 1, 5, 1]);
    drop(
        AttentionCache::update_for_attention(
            &mut native.layers.slots_mut()[0],
            MlxTensor::from(input.clone()),
            MlxTensor::from(input.clone()),
            &stream,
        )
        .unwrap(),
    );
    let fixed_alias = input.reshape(&[1, 5], &stream).unwrap();
    *native.layers.slots_mut()[0]
        .fixed_component(fixed_role)
        .unwrap() = Some(MlxTensor::from(fixed_alias));
    *native.layers.slots_mut()[1]
        .fixed_component(integer_role)
        .unwrap() = Some(MlxTensor::from(Array::from_slice(&[-3i32, 7], &[1, 2])));
    native.layers.slots_mut()[1].advance_fixed(5).unwrap();
    let Some(MlxHybridAttentionState::Compressed(cache)) =
        &mut native.layers.slots_mut()[2].attention
    else {
        unreachable!()
    };
    cache
        .update_and_fetch(
            Array::from_slice(&[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10.], &[1, 5, 2]),
            Array::from_slice(
                &[-1f32, -2., -3., -4., -5., -6., -7., -8., -9., -10.],
                &[1, 5, 2],
            ),
            &stream,
        )
        .unwrap();
    let expected_capacity = cache.capacity();
    for layer in native.layers.slots() {
        layer.visit_borrowed_values(&mut |value| {
            value.as_array().evaluated().unwrap();
        });
    }
    let expected = native.retained_storage().unwrap().array_allocation_facts();
    let capacity = 1 << 25;
    let pool = crate::memory_fixture::ledger(capacity, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(capacity),
        )
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let mut projected = cold(|| {
        native.project_resident_workspace_with_storage(NonZeroU32::new(1).unwrap(), &context)
    })
    .unwrap();
    assert!(projected.storage.is_complete());
    assert_eq!(
        projected
            .storage
            .iter()
            .map(|(id, bytes, _)| (id, bytes))
            .collect::<BTreeMap<_, _>>(),
        expected
    );
    assert_eq!(projected.storage.paged_sources().len(), 1);
    assert!(projected
        .state
        .as_ref()
        .iter()
        .all(|layer| layer.position() == 5));
    assert!(matches!(
        projected.state.as_ref()[0],
        WorkspaceResidentLayerState::Paged(_)
    ));
    assert!(matches!(
        projected.state.as_ref()[1],
        WorkspaceResidentLayerState::Ordinary(_)
    ));
    let WorkspaceResidentLayerState::Compressed(compressed) = &projected.state.as_ref()[2] else {
        panic!("compressed mechanism remains distinct")
    };
    assert_eq!(compressed.capacity(), expected_capacity);
    let fixed = projected.state.as_mut()[0]
        .fixed_component(fixed_role)
        .unwrap()
        .as_ref()
        .unwrap();
    assert_eq!(fixed.shape(), [1, 5]);
    let integer = projected.state.as_mut()[1]
        .fixed_component(integer_role)
        .unwrap()
        .as_ref()
        .unwrap();
    assert_eq!(integer.layout().dtype(), WorkspaceDtype::Int32);
    assert_eq!(integer.shape(), [1, 2]);
    let id = projected.storage.paged_sources()[0].geometry().blocks[0]
        .id
        .clone();
    // The wrong native fixed dtype refuses after accepting the actual pager;
    // its escaped failure must retain that prefix, without mutating live state.
    *native.layers.slots_mut()[1]
        .fixed_component(integer_role)
        .unwrap() = Some(MlxTensor::from(Array::from_slice(&[-3f32, 7.], &[1, 2])));
    let failure = match cold(|| {
        native.project_complete_workspace_with_storage(NonZeroU32::new(1).unwrap(), &context)
    }) {
        Err(failure) => failure,
        Ok(_) => panic!("declared integer role must reject floating source"),
    };
    assert_eq!(failure.retained_storage().unwrap().paged_sources().len(), 1);
    drop((native, input, context, funding));
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(projected);
    assert!(matches!(
        manager.remove_block(&id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    drop(failure);
    manager.remove_block(&id).unwrap();
    drop(manager);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
