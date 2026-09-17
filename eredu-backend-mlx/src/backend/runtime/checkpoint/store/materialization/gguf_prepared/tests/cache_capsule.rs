//! Actual cache consumers with a cache-only account. Fixture streams, catalogs,
//! and ordinary numerical outputs remain separate preexisting/test owners.
use super::*;
use crate::backend::runtime::checkpoint::store::{
    cache_context::tests::qualified_bytes, CacheHandle,
};
use crate::backend::runtime::residency::manager::{ResidencyError, ResidencyManager};
use eredu_core::residency::{
    MemoryTier, OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy,
};
use eredu_runtime::residency::{OffloadUnit, WeightBinding};
use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};

fn manager(
    fixture: &OriginalGgufMissFixture,
    cache: CacheHandle,
    pool: &WorkingMemoryPool,
) -> Result<ResidencyManager, ResidencyError> {
    let id = eredu_core::residency::OffloadUnitId::new("cache.window").unwrap();
    let binding = WeightBinding::new("weight", "bank.weight", TensorSelection::Full, 32).unwrap();
    let unit = OffloadUnit::new(id.clone(), vec![binding]).unwrap();
    let plan = OffloadPlan::new(
        OffloadConfig::default(),
        [OffloadUnitSpec::new(id, 32, ResidencyPolicy::Windowed, MemoryTier::Disk).unwrap()],
    )
    .unwrap();
    let source: eredu_checkpoint::store::SharedCheckpointSource = Arc::new(fixture.source.clone());
    ResidencyManager::new_shared_sources_with_cache(
        source,
        BTreeMap::new(),
        plan,
        [unit],
        fixture.stream.clone(),
        fixture.stream.clone(),
        cache,
        pool,
    )
}

#[test]
fn cache_capsule_manager_and_real_lease_preserve_the_original_birth() {
    let Some(bytes) = qualified_bytes() else {
        return;
    };
    let fixture = OriginalGgufMissFixture::new();
    let stream_bytes =
        crate::backend::runtime::checkpoint::store::PreparedMaterializationStreams::required_bytes(
            &fixture.stream,
            &fixture.stream,
        )
        .unwrap();
    let pool = WorkingMemoryPool::new(bytes.checked_add(stream_bytes).unwrap(), 0).unwrap();
    let foreign = WorkingMemoryPool::new(bytes, 0).unwrap();
    let cache = CacheHandle::prepare(&pool).unwrap();
    let selected = manager(&fixture, cache.clone(), &pool).unwrap();
    assert_eq!(pool.used_bytes().unwrap(), bytes + stream_bytes);
    assert!(selected.gguf_cache_handle().unwrap().same(&cache));
    assert!(matches!(
        manager(&fixture, CacheHandle::ordinary(), &pool),
        Err(ResidencyError::OriginalCache(
            WorkingMemoryError::UnknownBound
        ))
    ));
    assert!(matches!(
        manager(&fixture, cache.clone(), &foreign),
        Err(ResidencyError::OriginalCache(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    let context = MlxParameterMaterializationContext::with_cache(
        fixture.stream.clone(),
        fixture.stream.clone(),
        selected.gguf_cache_handle().unwrap(),
    );
    let lease = context
        .weight_lease(
            fixture
                .source
                .acquire_lease(OriginalGgufMissFixture::request())
                .unwrap(),
        )
        .unwrap();
    assert_eq!(lease.output_shape(), &[2, 4]);
    drop((context, selected, cache));
    safemlx::reclaim_allocation_owners(); // final private stream pair retired before its accounts
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(lease);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn cache_capsule_populated_cache_retires_after_pending_alias_and_before_final_credit() {
    let Some(bytes) = qualified_bytes() else {
        return;
    };
    let fixture = OriginalGgufMissFixture::new();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let cache = CacheHandle::prepare(&pool).unwrap();
    let context = MlxParameterMaterializationContext::with_cache(
        fixture.stream.clone(),
        fixture.stream.clone(),
        cache.clone(),
    );
    let lease = context
        .weight_lease(
            fixture
                .source
                .acquire_lease(OriginalGgufMissFixture::request())
                .unwrap(),
        )
        .unwrap();
    let pending = lease
        .prepare_materialization(&fixture.stream, &fixture.stream)
        .unwrap();
    assert_eq!(cache.try_lock().unwrap().len(), 1);
    drop((context, cache));
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    let output = pending.finish().unwrap();
    assert_eq!(values(&output, None), fixture.expected["bank.weight"]);
    // The output's ordinary native backing is a separate owner. Finishing the
    // last pending cache alias drops its now-stale row, mutex and shared shell.
    assert_eq!(pool.used_bytes().unwrap(), 0);
    drop(output);
}
