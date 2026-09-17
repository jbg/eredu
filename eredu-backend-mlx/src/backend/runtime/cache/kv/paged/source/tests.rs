use super::*;
use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
use safemlx::{Device, DeviceType};
use std::cell::Cell;

#[derive(Debug)]
struct NoEquations;
impl WorkspaceMechanisms for NoEquations {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("source projection cannot execute equations")
    }
}
impl eredu_nn::workspace::WorkspaceFactMechanisms for NoEquations {
    type Error = std::convert::Infallible;
    fn operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("source projection cannot quote equations")
    }
    fn write_operation_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceEffectDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceOperationFacts>, Self::Error> {
        panic!("source projection cannot emit equations")
    }
    fn host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("source projection cannot quote host equations")
    }
    fn write_host_facts(
        &self,
        _: eredu_nn::workspace::WorkspaceOperationView<'_>,
        _: eredu_nn::workspace::WorkspaceHostDestination<'_>,
    ) -> Result<Option<eredu_nn::workspace::WorkspaceHostFacts>, Self::Error> {
        panic!("source projection cannot emit host equations")
    }
}
thread_local! {static HOOKS:Cell<usize>=const{Cell::new(0)};}
fn hook() {
    HOOKS.with(|value| value.set(value.get() + 1));
}
struct Hooks;
impl Drop for Hooks {
    fn drop(&mut self) {
        safemlx::unregister_thread_runtime_housekeeping(hook);
    }
}
fn cold<T>(run: impl FnOnce() -> T) -> T {
    safemlx::register_thread_runtime_housekeeping(hook);
    let _hooks = Hooks;
    HOOKS.with(|value| value.set(0));
    let value = run();
    assert_eq!(HOOKS.with(Cell::get), 0);
    value
}
fn settle(cache: &PagedKeyValueCache) {
    for array in cache.tail_keys.iter().chain(cache.tail_values.iter()) {
        array.evaluated().unwrap();
    }
}
#[test]
#[ignore = "requires native CPU cache producers"]
fn paged_source_keeps_exact_sealed_blocks_tail_rank_and_paid_geometry_after_retirement() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let manager = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let rank = Some(CacheRankIdentity::new(Some(1), Some(2), Some(3)));
    let mut cache = PagedKeyValueCache::new_with_layout(manager.clone(), 7, None, 2, rank).unwrap();
    let keys = Array::from_slice(
        &[0.5f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[1, 1, 5, 2],
    );
    let values = Array::from_slice(
        &[
            -0.5f32, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0,
        ],
        &[1, 1, 5, 2],
    );
    drop(cache.update_and_fetch(keys, values, &stream).unwrap());
    settle(&cache);
    let ids = manager
        .layer_block_ids(7, CacheRepresentation::KeyValue, 0, i64::MAX, 0)
        .unwrap();
    assert_eq!(
        ids.iter().map(|id| (id.start, id.end)).collect::<Vec<_>>(),
        [(0, 2), (2, 4)]
    );
    let capacity = 1 << 22;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), capacity)
        .unwrap();
    let context =
        WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let geometry = cold(|| {
        let source_use = Cell::new(0);
        let start = context.metadata_census().unwrap().context_bytes();
        let consume = |source: PagedKeyValueSource<'_>| {
            source_use.set(context.metadata_census().unwrap().context_bytes() - start);
            assert_eq!(source.block_count(), ids.len());
            assert_eq!(source.manager_source().session_id(), manager.session_id());
            assert_eq!(source.manager_source().pool().id(), manager.pool().id());
            for (block, id) in source.manager_source().blocks().zip(&ids) {
                assert_eq!(block.id(), id);
                assert!(block.device().is_some());
                assert!(block.host().is_none());
                assert!(block.disk().is_none());
            }
            let tail = source.tail_arrays().unwrap();
            assert!(std::ptr::eq(tail[0], cache.tail_keys.as_ref().unwrap()));
            let before = context.metadata_census().unwrap().context_bytes();
            let bound = source.geometry_control_bytes().unwrap();
            let geometry = source.prepare_geometry(&context)?;
            assert!(context.metadata_census().unwrap().context_bytes() - before <= bound);
            Ok(geometry)
        };
        let source_bound = PagedKeyValueCache::workspace_source_control_bytes::<
            PagedCacheSourceGeometry,
            _,
        >(&consume)
        .unwrap();
        let result = cache.with_workspace_source(&context, consume).unwrap();
        assert!(source_use.get() > 0 && source_use.get() <= source_bound);
        result
    });
    assert_eq!(geometry.session_id, manager.session_id());
    assert_eq!(geometry.pool_id, manager.pool().id());
    assert_eq!(geometry.global_layer, 7);
    assert_eq!(geometry.rank, rank);
    assert_eq!((geometry.offset, geometry.tail_start), (5, 4));
    assert_eq!(geometry.prefix_tokens, 2);
    assert_eq!(geometry.sliding_window, None);
    assert!(!geometry.key_only);
    assert_eq!(geometry.blocks.len(), 2);
    assert_eq!(geometry.tail.unwrap()[0].shape, [1, 1, 1, 2]);
    assert_eq!(geometry.blocks[0].arrays[0].logical_bytes, 16);
    assert_eq!(
        geometry
            .blocks
            .iter()
            .map(|block| &block.id)
            .collect::<Vec<_>>(),
        ids.iter().collect::<Vec<_>>()
    );
    // Rebinding a nonempty local tail to an unrelated manager cannot synthesize
    // canonical history or an account from its byte-compatible shape.
    let foreign = CacheResidencyManager::new(
        PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap();
    let mut stale = cache.clone();
    stale.rebind_paging_manager(foreign.clone());
    let error = cold(|| stale.with_workspace_source(&context, |_| Ok(()))).unwrap_err();
    assert!(matches!(
        error.cause(),
        crate::backend::runtime::cache::residency::CacheSourceFailureCause::Source(
            CacheSourceError::Identity
        )
    ));
    drop((context, funding, cache, stale, manager, foreign, ids));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(geometry.blocks[1].id.start, 2);
    drop(geometry);
    assert!(
        pool.used_bytes().unwrap() > 0,
        "escaped source failure retains its independently paid host owner"
    );
    drop(error);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}


#[test]
#[ignore = "requires native CPU cache producers"]
fn pinned_paged_projection_keeps_aliases_canonical_leases_and_failed_source_custody() {
    use crate::backend::runtime::cache::residency::{CacheBlockArrays, CacheResidencyError, CacheSourceFailureCause};
    use eredu_runtime::CacheLifecycleError;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1).unwrap().with_full_attention(true);
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let mut cache = PagedKeyValueCache::new_with_layout(manager.clone(), 3, None, 0, None).unwrap();
    let keys = Array::from_slice(&[1.25f32, 2.5, 3.75, 5.0, 6.25], &[1, 1, 5, 1]);
    let values = Array::from_slice(&[-1.25f32, -2.5, -3.75, -5.0, -6.25], &[1, 1, 5, 1]);
    drop(cache.update_and_fetch(keys, values, &stream).unwrap());
    settle(&cache);
    let report = manager.report().unwrap();
    let capacity = 1 << 23;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let funding = pool.prepare_workspace_metadata(&InferenceExecutionIdentity::default(), capacity).unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(NoEquations, funding.clone()).unwrap();
    let bound = cold(|| cache.with_workspace_source(&context, |source| {
        source.device_projection_control_bytes(&context)
    })).unwrap() + PagedKeyValueCache::device_projection_loan_control_bytes().unwrap();
    let before = context.metadata_census().unwrap().context_bytes();
    let projected = cold(|| cache.project_device_workspace(&context)).unwrap();
    assert!(context.metadata_census().unwrap().context_bytes() - before <= bound);
    assert!(projected.storage().is_complete());
    assert_eq!(projected.blocks().len(), 2);
    assert_eq!(projected.blocks()[0][0].layout().shape(), &[1, 1, 2, 1]);
    assert_eq!(projected.tail().unwrap()[0].layout().shape(), &[1, 1, 1, 1]);
    let second = cold(|| cache.project_device_workspace(&context)).unwrap();
    cold(|| cache.with_workspace_source(&context, |source| {
        projected.validate_source(&source, &context)?;
        second.validate_source(&source, &context)
    })).unwrap();
    for (identity, bytes, _) in projected.storage().iter() {
        let other = second.storage().native_array(identity).unwrap();
        assert_eq!(other.try_allocation_info().unwrap().unwrap().bytes() as u64, bytes);
    }
    assert_eq!(manager.report().unwrap().demand_hits, report.demand_hits);
    assert_eq!(manager.report().unwrap().demand_misses, report.demand_misses);
    let id = projected.geometry().blocks[0].id.clone();
    assert!(matches!(manager.remove_block(&id), Err(CacheResidencyError::Lifecycle(CacheLifecycleError::BlockLeased(_)))));
    drop(second);
    assert!(matches!(manager.remove_block(&id), Err(CacheResidencyError::Lifecycle(CacheLifecycleError::BlockLeased(_)))));

    // A real sealed native integer cache is a valid source, but the selected
    // workspace tensor representation refuses Int64. Pins accepted before the
    // import remain in the typed failure until the caller retires it.
    let unsupported_manager = CacheResidencyManager::new(options).unwrap();
    let unsupported_id = unsupported_manager.seal_block(3, 0, 2, None, CacheBlockArrays::KeyValue {
        keys: Array::from_slice(&[7i64, 13], &[1, 1, 2, 1]),
        values: Array::from_slice(&[17i64, 19], &[1, 1, 2, 1]),
    }, false).unwrap();
    let unsupported = PagedKeyValueCache::new_with_layout(unsupported_manager.clone(), 3, None, 0, None).unwrap();
    let failure = match cold(|| unsupported.project_device_workspace(&context)) {
        Err(error) => error,
        Ok(_) => panic!("unsupported projection must refuse"),
    };
    assert!(matches!(failure.cause().cause(), CacheSourceFailureCause::Projection(
        crate::backend::nn::workspace::ProjectionSourceError::Dtype(Dtype::Int64)
    )));
    assert!(matches!(unsupported_manager.remove_block(&unsupported_id), Err(CacheResidencyError::Lifecycle(CacheLifecycleError::BlockLeased(_)))));
    // The manager is unlocked even while an escaped failure retains its pin.
    unsupported.with_workspace_source(&context, |_| Ok(())).unwrap();
    drop((cache, unsupported, context, funding));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(projected);
    manager.remove_block(&id).unwrap();
    assert!(pool.used_bytes().unwrap() > 0);
    drop(failure);
    unsupported_manager.remove_block(&unsupported_id).unwrap();
    drop((manager, unsupported_manager));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[path = "tests/append.rs"]
mod append;
