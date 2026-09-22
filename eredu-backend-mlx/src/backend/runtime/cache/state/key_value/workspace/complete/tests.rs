use super::*;
use crate::backend::runtime::cache::residency::{
    CacheBlockArrays, CacheResidencyError, CacheSourceFailureCause,
};
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_nn::{workspace::*, Error, Tensor};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger},
    CacheLifecycleError, PagedCacheOptions,
};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
use std::{cell::Cell, collections::BTreeSet};

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
#[ignore = "requires native CPU cache sources"]
fn complete_paged_state_shares_dense_aliases_and_retires_failed_source_prefixes() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap();
    let layout = StateLayout::new(LayerSchedule::new(3, vec![policy; 3]).unwrap()).unwrap();
    let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let rank = Some(CacheRankIdentity::new(Some(1), Some(2), Some(3)));
    let mut native =
        MlxKeyValueState::paged_with_global_layer_start(layout, manager.clone(), rank, 7).unwrap();
    native.layers.slots_mut()[2] = MlxKeyValueLayerState::Device(ConcatKeyValueCache::new());
    // The same nonzero actual backing is both K and V, across two pagers and
    // the ordinary dense cache. No synthetic byte/identity witness is supplied.
    let input = Array::from_slice(&[0.25f32, 0.5, 0.75, 1.0, 1.25], &[1, 1, 5, 1]);
    for layer in native.layers.slots_mut() {
        drop(
            KeyValueCache::update_for_attention(layer, input.clone(), input.clone(), &stream)
                .unwrap(),
        );
        for value in RuntimeLayerState::<MlxNeuralBackend>::retained_values(layer) {
            value.as_array().evaluated().unwrap();
        }
    }
    let report = manager.report().unwrap();
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
    let mut expected = BTreeSet::new();
    let mut operands = 0;
    for layer in native.layers.slots() {
        match layer {
            MlxKeyValueLayerState::Paged(cache) => cache
                .with_workspace_source(&context, |source| {
                    for block in source.manager_source().blocks() {
                        for array in block.device().unwrap() {
                            expected
                                .insert(array.try_allocation_info().unwrap().unwrap().identity());
                            operands += 1;
                        }
                    }
                    for arrays in source.tail_arrays() {
                        for array in arrays {
                            expected
                                .insert(array.try_allocation_info().unwrap().unwrap().identity());
                            operands += 1;
                        }
                    }
                    Ok(())
                })
                .unwrap(),
            _ => {
                for value in RuntimeLayerState::<MlxNeuralBackend>::retained_values(layer) {
                    expected.insert(
                        value
                            .as_array()
                            .try_allocation_info()
                            .unwrap()
                            .unwrap()
                            .identity(),
                    );
                    operands += 1;
                }
            }
        }
    }
    let mut projected = cold(|| {
        native.project_resident_workspace_with_storage(NonZeroU32::new(1).unwrap(), &context)
    })
    .unwrap();
    assert!(projected.storage.is_complete());
    assert_eq!(
        projected
            .storage
            .iter()
            .map(|(id, _, _)| id)
            .collect::<BTreeSet<_>>(),
        expected
    );
    assert!(
        expected.len() < operands,
        "actual cross-role aliases share roots"
    );
    assert_eq!(projected.storage.paged_sources().len(), 2);
    for (index, source) in projected.storage.paged_sources().iter().enumerate() {
        let MlxKeyValueLayerState::Paged(cache) = &native.layers.slots()[index] else {
            panic!("paged source")
        };
        cold(|| source.validate_source(cache, &projected.storage, &context)).unwrap();
        assert_eq!(source.geometry().global_layer, index + 7);
        assert_eq!(source.geometry().rank, rank);
        assert_eq!(
            (source.geometry().offset, source.geometry().tail_start),
            (5, 4)
        );
        let WorkspaceResidentLayerState::Paged(state) = &projected.state.as_ref()[index] else {
            panic!("distinct paged equation")
        };
        assert_eq!(state.geometry().offset, 5);
    }
    assert!(matches!(
        projected.state.as_ref()[2],
        WorkspaceResidentLayerState::Ordinary(_)
    ));
    assert_eq!(manager.report().unwrap().demand_hits, report.demand_hits);
    assert_eq!(
        manager.report().unwrap().demand_misses,
        report.demand_misses
    );
    let id = projected.storage.paged_sources()[0].geometry().blocks[0]
        .id
        .clone();
    let unrelated = CacheResidencyManager::new(options.clone()).unwrap();
    let MlxKeyValueLayerState::Paged(cache) = &native.layers.slots()[0] else {
        unreachable!()
    };
    let mut foreign = cache.clone();
    foreign.rebind_paging_manager(unrelated.clone());
    let foreign_error = cold(|| {
        projected.storage.paged_sources()[0].validate_source(&foreign, &projected.storage, &context)
    })
    .unwrap_err();
    assert!(matches!(
        foreign_error.cause(),
        CacheSourceFailureCause::Source(CacheSourceError::Identity)
    ));
    drop((foreign_error, foreign, unrelated));

    // A source-valid Int64 second layer refuses the tensor representation only
    // after the first layer and both canonical source pins have been accepted.
    let unsupported_manager = CacheResidencyManager::new(options).unwrap();
    let unsupported_id = unsupported_manager
        .seal_block(
            8,
            0,
            2,
            rank,
            CacheBlockArrays::KeyValue {
                keys: Array::from_slice(&[7i64, 11], &[1, 1, 2, 1]),
                values: Array::from_slice(&[13i64, 17], &[1, 1, 2, 1]),
            },
            false,
        )
        .unwrap();
    native.layers.slots_mut()[1] = MlxKeyValueLayerState::Paged(
        PagedKeyValueCache::new_with_layout(unsupported_manager.clone(), 8, None, 0, rank).unwrap(),
    );
    let failure = match cold(|| {
        native.project_complete_workspace_with_storage(NonZeroU32::new(1).unwrap(), &context)
    }) {
        Err(failure) => failure,
        Ok(_) => panic!("unsupported source must refuse"),
    };
    assert!(matches!(
        failure.cause().cause(),
        CacheSourceFailureCause::Projection(
            crate::backend::nn::workspace::ProjectionSourceError::Dtype(Dtype::Int64)
        )
    ));
    assert_eq!(failure.retained_storage().unwrap().paged_sources().len(), 2);
    assert!(failure.retained_storage().unwrap().iter().len() > 0);
    let legacy_error = match cold(|| {
        native.project_resident_workspace_with_storage(NonZeroU32::new(1).unwrap(), &context)
    }) {
        Err(error) => error,
        Ok(_) => panic!("legacy entry must preserve refusal"),
    };
    // Both manager locks are released even when the accepted prefix escapes.
    let MlxKeyValueLayerState::Paged(cache) = &native.layers.slots()[1] else {
        unreachable!()
    };
    cache.with_workspace_source(&context, |_| Ok(())).unwrap();
    let accepted_sources = cold(|| projected.storage.take_paged_sources(&context))
        .unwrap()
        .unwrap();
    assert_eq!(accepted_sources.sources().len(), 2);
    assert!(projected.storage.paged_sources().is_empty());
    drop((native, input, context, funding));
    assert!(pool.fixture_host_charge().unwrap() > 0);
    for (identity, _, _) in projected.storage.iter() {
        let values = projected
            .storage
            .native_array(identity)
            .unwrap()
            .evaluated()
            .unwrap();
        assert!(values
            .as_slice::<f32>()
            .iter()
            .all(|value| [0.25, 0.5, 0.75, 1.0, 1.25].contains(value)));
    }
    drop(projected);
    assert!(matches!(
        manager.remove_block(&id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    assert!(matches!(
        unsupported_manager.remove_block(&unsupported_id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    drop(failure);
    assert!(matches!(
        manager.remove_block(&id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    let retained_role_alias = accepted_sources.clone();
    drop(accepted_sources);
    assert!(matches!(
        manager.remove_block(&id),
        Err(CacheResidencyError::Lifecycle(
            CacheLifecycleError::BlockLeased(_)
        ))
    ));
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(retained_role_alias);
    manager.remove_block(&id).unwrap();
    unsupported_manager.remove_block(&unsupported_id).unwrap();
    // The legacy error keeps H but no longer pins unsubmitted native storage.
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(legacy_error);
    drop((manager, unsupported_manager));
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
