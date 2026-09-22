use super::*;
use eredu_core::{AttentionPolicy, LayerSchedule};
use eredu_runtime::{
    working_memory::{InferenceExecutionIdentity, MemoryLedger},
    PagedCacheOptions,
};
use safemlx::{Array, Device, DeviceType, Stream};

#[test]
#[ignore = "requires native paged cache sources"]
fn populated_reset_source_builds_independent_empty_manager_and_preserves_old_backing() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap();
    let layout =
        eredu_runtime::StateLayout::new(LayerSchedule::new(1, vec![policy.clone()]).unwrap())
            .unwrap();
    let options = PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let mut source =
        MlxKeyValueState::paged_with_global_layer_start(layout.clone(), manager.clone(), None, 7)
            .unwrap();
    let input = Array::from_slice(&[1.25f32, -2.5, 3.75], &[1, 1, 3, 1]);
    drop(
        KeyValueCache::update_for_attention(
            &mut source.layers.slots_mut()[0],
            input.clone(),
            input,
            &stream,
        )
        .unwrap(),
    );
    for value in RuntimeLayerState::<MlxNeuralBackend>::retained_values(&source.layers.slots()[0]) {
        value.as_array().evaluated().unwrap();
    }
    let prior = manager.report().unwrap();
    let (plan, bytes) = source.original_reset_plan().unwrap();
    assert!(bytes > 0);
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(1 << 24),
        )
        .unwrap();
    let mut context = source
        .prepare_original_reset(&plan, Some(&funding))
        .unwrap();
    let empty = MlxKeyValueState::empty_original_reset_layer(
        &mut context,
        &source.layers.slots()[0],
        &policy,
        None,
    )
    .unwrap();
    let MlxKeyValueLayerState::Paged(empty) = empty else {
        panic!("same paged representation")
    };
    assert_eq!(KeyValueCache::offset(&empty), 0);
    assert_ne!(empty.manager().session_id(), manager.session_id());
    assert_eq!(empty.manager().pool(), manager.pool());
    assert_eq!(KeyValueCache::offset(&source.layers.slots()[0]), 3);
    assert_eq!(manager.report().unwrap(), prior);
    let other_manager = CacheResidencyManager::new(options).unwrap();
    let other =
        MlxKeyValueState::paged_with_global_layer_start(layout, other_manager, None, 7).unwrap();
    assert!(other.prepare_original_reset(&plan, Some(&funding)).is_err());
    drop((context, funding, source, manager, other));
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(empty);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[test]
#[ignore = "requires native paged cache sources"]
fn host_reset_preserves_actual_sources_without_promotion_or_copy_authority() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let policy = LayerCachePolicy::key_value(AttentionPolicy::Full, 1, 1).unwrap();
    let layout =
        eredu_runtime::StateLayout::new(LayerSchedule::new(1, vec![policy.clone()]).unwrap())
            .unwrap();
    // Two sealed 16-byte pairs and an eight-byte tail force a real Host page.
    // Reserve the actual Transfer allocator extent for the four source buffers
    // that these two pages can retain, including native page rounding. Their
    // in-flight transfer reservations cover those same actual destinations.
    let host_buffer_bytes = safemlx::host_transfer_capacity_upper_bound(
        2 * std::mem::size_of::<f32>(),
        safemlx::HostTransferPolicy::Transfer,
    )
    .unwrap();
    let host_bytes = u64::try_from(host_buffer_bytes)
        .unwrap()
        .checked_mul(4)
        .unwrap();
    let physical = eredu_runtime::CacheResidencyPool::new(
        eredu_runtime::CachePoolLimits::new(4096, host_bytes, host_bytes, 0).unwrap(),
    );
    let options = PagedCacheOptions::new(2, 32, host_bytes, 1)
        .unwrap()
        .with_full_attention(true)
        .with_pool(physical)
        .unwrap();
    let manager = CacheResidencyManager::new(options.clone()).unwrap();
    let mut source =
        MlxKeyValueState::paged_with_global_layer_start(layout.clone(), manager.clone(), None, 7)
            .unwrap();
    // Each complete two-token ingress fits its real 32-byte publication
    // overlap. The final 40-byte history still cannot fit canonically, so the
    // existing manager must actually demote a sealed page to Host.
    for values in [&[1.25f32, -2.5][..], &[3.75f32, 4.5][..], &[-5.25f32][..]] {
        let input = Array::from_slice(values, &[1, 1, values.len() as i32, 1]);
        drop(
            KeyValueCache::update_for_attention(
                &mut source.layers.slots_mut()[0],
                input.clone(),
                input,
                &stream,
            )
            .unwrap(),
        );
    }
    stream.synchronize().unwrap();
    // Register the foreign manager before quoting the shared pool's actual
    // registration destination. A later registration legitimately changes that
    // source-derived layout and must invalidate a prior reset plan.
    let other = MlxKeyValueState::paged_with_global_layer_start(
        layout,
        CacheResidencyManager::new(options).unwrap(),
        None,
        7,
    )
    .unwrap();
    let prior = manager.report().unwrap();
    assert!(prior.current_host_bytes > 0);
    assert!(prior.host_demotions > 0);
    let (plan, bytes) = source.original_reset_plan().unwrap();
    assert!(bytes > 0);
    let pool = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    let funding = pool
        .prepare_workspace_metadata(
            &InferenceExecutionIdentity::default(),
            crate::memory_fixture::resolved_limits(1 << 24),
        )
        .unwrap();
    // Cold copy inspection describes the same two Host/Device pages. It
    // allocates no destination, promotes nothing and grants no numerical copy.
    let copy_layout = manager.inspect_paged_array_copy().unwrap();
    assert_eq!(copy_layout.page_operands(), Some(4));
    assert_eq!(manager.report().unwrap(), prior);
    assert!(other.prepare_original_reset(&plan, Some(&funding)).is_err());
    let mut context = source
        .prepare_original_reset(&plan, Some(&funding))
        .unwrap();
    let empty = MlxKeyValueState::empty_original_reset_layer(
        &mut context,
        &source.layers.slots()[0],
        &policy,
        None,
    )
    .unwrap();
    let MlxKeyValueLayerState::Paged(empty) = empty else {
        panic!("same paged representation")
    };
    assert_eq!(KeyValueCache::offset(&empty), 0);
    assert_ne!(empty.manager().session_id(), manager.session_id());
    assert_eq!(empty.manager().pool(), manager.pool());
    assert_eq!(empty.manager().options().device_budget_bytes(), 32);
    assert_eq!(empty.manager().options().host_budget_bytes(), host_bytes);
    assert_eq!(empty.manager().report().unwrap().current_host_bytes, 0);
    assert_eq!(KeyValueCache::offset(&source.layers.slots()[0]), 5);
    assert_eq!(manager.report().unwrap(), prior);
    drop((context, funding, source, manager, other));
    assert!(pool.fixture_host_charge().unwrap() > 0);
    drop(empty);
    assert_eq!(pool.fixture_host_charge().unwrap(), 0);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
