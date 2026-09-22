use super::*;
use crate::backend::runtime::checkpoint::store::cache;
use eredu_runtime::working_memory::OriginalHostMetadataCustody;
use std::alloc::Layout;

pub(super) fn cache_baseline(fixture: &OriginalGgufMissFixture) -> u64 {
    // The reference setup left ordinary stale rows. Retire them before this
    // scoped component accounts the empty shared context and begins work.
    let retired = {
        let mut cache = fixture.context.converted_groups.lock().unwrap();
        let retired = cache.extract_if(|_, row| row.stale());
        assert!(cache.is_empty());
        retired
    };
    drop(retired);
    let plan = fixture
        .source
        .conversion_plan(&OriginalGgufMissFixture::request())
        .unwrap();
    MlxParameterMaterializationContext::cache_context_storage_bytes().unwrap()
        + OriginalHostMetadataCustody::shared_storage_bytes(
            plan.identity().source_owner_payload_layout(),
        )
        .unwrap()
        + OriginalHostMetadataCustody::shared_storage_bytes(Layout::new::<
            safemlx::PreparedInputRuntime,
        >())
        .unwrap()
}

#[test]
fn prepared_cache_custody_survives_strong_group_or_stale_weak_row_without_double_charge() {
    for keep_group in [false, true] {
        let fixture = OriginalGgufMissFixture::new();
        let existing = cache_baseline(&fixture);
        let plan = fixture
            .source
            .conversion_plan(&OriginalGgufMissFixture::request())
            .unwrap();
        let destinations = PreparedPendingWeight::host_destination_requests(&plan).unwrap();
        let layouts = fixture.source_storage_layouts();
        let bytes = layouts.iter().sum::<u64>() + cache::control_bytes().unwrap();
        let (group, weak, context, pool) = component_destinations_with_baseline(
            bytes,
            layouts.len() + 1,
            0,
            Some((destinations.bytes() as u64, destinations.calls(), 0)),
            existing,
            |_| 0,
            |controls, observer, pool, bank| {
                let pending = fixture
                    .lease()
                    .prepare_original_gguf_fixture(
                        &fixture.stream,
                        fixture.admitted_ready(controls, bank),
                        observer.clone(),
                    )
                    .unwrap();
                fixture.settle(&pending, observer);
                let group = pending.retained.retention().group.as_ref().unwrap().clone();
                let weak = group.downgrade();
                assert_eq!(
                    super::super::values(group.arrays.get("bank.weight").unwrap(), Some(observer)),
                    fixture.expected["bank.weight"]
                );
                let context = fixture.context.clone();
                assert_eq!(context.converted_groups.lock().unwrap().len(), 1);
                let before_clone = pool.fixture_host_charge().unwrap();
                let second = context.clone();
                assert_eq!(pool.fixture_host_charge().unwrap(), before_clone);
                drop(second);
                OriginalGgufMissFixture::finish(pending);
                (group, weak, context, pool.clone())
            },
        );
        drop((plan, fixture));
        let held = pool.fixture_host_charge().unwrap();
        assert!(held > existing);
        if keep_group {
            drop((context, weak));
            safemlx::reclaim_allocation_owners();
            assert_eq!(pool.fixture_host_charge().unwrap(), held);
            drop(group);
        } else {
            drop(group);
            safemlx::reclaim_allocation_owners();
            assert!(weak.upgrade().is_none());
            assert_eq!(
                pool.fixture_host_charge().unwrap(),
                held,
                "stale row and external Weak retain metadata"
            );
            let retired = {
                let mut rows = context.converted_groups.lock().unwrap();
                let retired = rows.extract_if(|_, row| row.stale());
                assert!(rows.is_empty());
                retired
            };
            assert!(context.converted_groups.try_lock().is_ok());
            drop((retired, context));
            assert_eq!(
                pool.fixture_host_charge().unwrap(),
                held,
                "external Weak still pins the Arc shell"
            );
            drop(weak);
        }
        settle_pool_at(&pool, existing);
    }
}

#[test]
fn prepared_cache_prefix_refusal_keeps_same_source_and_never_reads_or_falls_back() {
    for refuse_controls in [false, true] {
        let fixture = OriginalGgufMissFixture::new();
        let existing = cache_baseline(&fixture);
        let mut request = OriginalGgufMissFixture::request();
        request.selection = TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        };
        let plan = fixture.source.conversion_plan(&request).unwrap();
        let controls_bytes = cache::control_bytes().unwrap();
        // The name fits exactly; the second, actual selection buffer cannot.
        let name_bytes = "bank.weight".len() as u64;
        let before = fixture.source_physical_reads();
        let (error, source, pool) = component_destinations_with_baseline(
            controls_bytes - u64::from(refuse_controls),
            1,
            0,
            Some((name_bytes, 2, 0)),
            existing,
            |_| 0,
            |controls, observer, pool, bank| {
                let lease = fixture
                    .context
                    .weight_lease(fixture.source.acquire_lease(request).unwrap())
                    .unwrap();
                let address = OriginalGgufMissFixture::address(&lease);
                let mut pending = PendingWeightMaterialization::begin_with_original(
                    lease,
                    &fixture.stream,
                    &fixture.stream,
                    Some((fixture.admitted_ready(controls, bank), observer.clone())),
                )
                .unwrap();
                assert!(pending.has_gguf_cache_source_bank());
                let error = pending.prepare_gguf_cache_key().unwrap_err();
                let host = pending.retained.retention().gguf_host.as_ref().unwrap();
                assert!(!host.raw_attempted && host.facts().iter().all(Option::is_none));
                let source = host.source_owner();
                assert_eq!(
                    std::ptr::from_ref(source.upgrade().unwrap().lease()),
                    address
                );
                assert_eq!(OriginalGgufMissFixture::address(pending.lease()), address);
                let bank = host.host_destinations.as_ref().unwrap();
                if refuse_controls {
                    assert!(
                        matches!(error.cause(), GgufHostCopyCause::SourceFunding(cause)
                        if matches!(cause.cause(), HostDestinationCause::Capacity {required,remaining}
                            if *required == controls_bytes && *remaining == controls_bytes - 1))
                    );
                    assert_eq!(
                        (bank.remaining_bytes(), bank.remaining_attempts()),
                        (name_bytes, 2)
                    );
                } else {
                    assert!(matches!(
                        error.cause(),
                        GgufHostCopyCause::CacheStorage(
                            eredu_gguf::SuppliedStorageError::Provider(
                                HostDestinationCause::Capacity { remaining: 0, .. }
                            )
                        )
                    ));
                    assert_eq!((bank.remaining_bytes(), bank.remaining_attempts()), (0, 0));
                }
                assert_eq!(fixture.source_physical_reads(), before);
                assert!(fixture
                    .context
                    .converted_groups
                    .try_lock()
                    .unwrap()
                    .is_empty());
                OriginalGgufMissFixture::finish(pending);
                assert!(
                    source.upgrade().is_some(),
                    "public typed failure keeps actual source after recovery"
                );
                (error, source, pool.clone())
            },
        );
        drop((plan, fixture));
        assert!(source.upgrade().is_some());
        assert!(pool.fixture_host_charge().unwrap() > existing);
        drop(error);
        assert!(source.upgrade().is_none());
        drop(source);
        settle_pool_at(&pool, existing);
    }
}

#[test]
fn prepared_cache_replaces_weak_row_that_expires_after_sweep_before_upgrade() {
    let fixture = OriginalGgufMissFixture::new();
    let existing = cache_baseline(&fixture);
    let plan = fixture
        .source
        .conversion_plan(&OriginalGgufMissFixture::request())
        .unwrap();
    let destinations = PreparedPendingWeight::host_destination_requests(&plan).unwrap();
    let layouts = fixture.source_storage_layouts();
    let bytes = layouts.iter().sum::<u64>() + cache::control_bytes().unwrap();
    let attempts = layouts.len() + 1;
    let before = fixture.source_physical_reads();
    let (array, pool) = component_destinations_with_baseline(
        bytes * 2,
        attempts * 2,
        2,
        Some((destinations.bytes() as u64 * 2, destinations.calls() * 2, 2)),
        existing,
        |_| 0,
        |controls, observer, pool, mut bank| {
            let first = fixture
                .lease()
                .prepare_original_gguf_fixture(
                    &fixture.stream,
                    fixture.admitted_ready(
                        controls,
                        bank.split(
                            destinations.bytes() as u64,
                            destinations.calls(),
                            Some((bytes, attempts)),
                        )
                        .unwrap(),
                    ),
                    observer.clone(),
                )
                .unwrap();
            fixture.settle(&first, observer);
            let group = first.retained.retention().group.as_ref().unwrap().clone();
            let weak = group.downgrade();
            OriginalGgufMissFixture::finish(first);
            assert!(weak.upgrade().is_some());
            // The sweep must see this last strong group. The actual lookup
            // then sees the expired Weak while its equal-key node still exists.
            cache::after_sweep_once(move || drop(group));
            let second = fixture
                .lease()
                .prepare_original_gguf_fixture(
                    &fixture.stream,
                    fixture.admitted_ready(
                        controls,
                        bank.split(
                            destinations.bytes() as u64,
                            destinations.calls(),
                            Some((bytes, attempts)),
                        )
                        .unwrap(),
                    ),
                    observer.clone(),
                )
                .unwrap();
            fixture.settle(&second, observer);
            assert!(weak.upgrade().is_none());
            assert_eq!(fixture.source_physical_reads(), before + 2);
            assert_eq!(
                fixture.context.converted_groups.try_lock().unwrap().len(),
                1
            );
            let output = second.output().clone();
            OriginalGgufMissFixture::finish(second);
            (output, pool.clone())
        },
    );
    drop((plan, fixture, array));
    settle_pool_at(&pool, existing);
}

#[test]
fn prepared_cache_live_row_refusal_retains_converted_group_and_key_until_recovery() {
    let fixture = OriginalGgufMissFixture::new();
    let existing = cache_baseline(&fixture);
    let plan = fixture
        .source
        .conversion_plan(&OriginalGgufMissFixture::request())
        .unwrap();
    let destinations = PreparedPendingWeight::host_destination_requests(&plan).unwrap();
    let layouts = fixture.source_storage_layouts();
    let bytes = layouts.iter().sum::<u64>() + cache::control_bytes().unwrap();
    let attempts = layouts.len() + 1;
    let (error, pool) = component_destinations_with_baseline(
        bytes * 2,
        attempts * 2,
        2,
        Some((destinations.bytes() as u64 * 2, destinations.calls() * 2, 2)),
        existing,
        |_| 0,
        |controls, observer, pool, mut bank| {
            let first = fixture
                .lease()
                .prepare_original_gguf_fixture(
                    &fixture.stream,
                    fixture.admitted_ready(
                        controls,
                        bank.split(
                            destinations.bytes() as u64,
                            destinations.calls(),
                            Some((bytes, attempts)),
                        )
                        .unwrap(),
                    ),
                    observer.clone(),
                )
                .unwrap();
            fixture.settle(&first, observer);
            let first_group = first.retained.retention().group.as_ref().unwrap().clone();
            // Exercise the actual commit's refusal after a complete distinct
            // attempted group, while the canonical row is still strongly owned.
            let mut pending = PendingWeightMaterialization::begin_with_original(
                fixture.lease(),
                &fixture.stream,
                &fixture.stream,
                Some((
                    fixture.admitted_ready(
                        controls,
                        bank.split(
                            destinations.bytes() as u64,
                            destinations.calls(),
                            Some((bytes, attempts)),
                        )
                        .unwrap(),
                    ),
                    observer.clone(),
                )),
            )
            .unwrap();
            pending.prepare_gguf_cache_key().unwrap();
            let portable = pending.materialize_gguf_admitted().unwrap();
            let arrays = pending.convert_original_stored_gguf(portable).unwrap();
            let error = {
                let mut rows = fixture.context.converted_groups.lock().unwrap();
                let error = pending
                    .complete_gguf_cache_entry(arrays, &mut rows)
                    .unwrap_err();
                assert_eq!(rows.len(), 1);
                error
            };
            assert!(matches!(error.cause(), GgufHostCopyCause::TypedBinding));
            let host = pending.retained.retention().gguf_host.as_ref().unwrap();
            let refused = host
                .retained_cache_group()
                .expect("successful conversion retained on commit refusal");
            assert!(!refused.same(&first_group));
            assert_eq!(
                super::super::values(refused.arrays.get("bank.weight").unwrap(), Some(observer)),
                fixture.expected["bank.weight"]
            );
            let weak = refused.downgrade();
            let held = pool.fixture_host_charge().unwrap();
            drop(first_group);
            OriginalGgufMissFixture::finish(first);
            assert!(weak.upgrade().is_some());
            assert_eq!(pool.fixture_host_charge().unwrap(), held);
            OriginalGgufMissFixture::finish(pending);
            assert!(weak.upgrade().is_none());
            assert_eq!(
                pool.fixture_host_charge().unwrap(),
                held,
                "failure/remaining raw aliases cannot return credit early"
            );
            (error, pool.clone())
        },
    );
    drop((plan, fixture));
    assert!(pool.fixture_host_charge().unwrap() > existing);
    drop(error);
    settle_pool_at(&pool, existing);
}

#[test]
fn funded_cache_capsule_and_stale_supplied_row_keep_separate_accounts() {
    use crate::backend::runtime::checkpoint::store::{
        cache_context::tests::qualified_bytes, CacheHandle,
    };
    let Some(cache_bytes) = qualified_bytes() else {
        return;
    };
    let fixture = OriginalGgufMissFixture::new();
    let existing = cache_baseline(&fixture);
    let plan = fixture
        .source
        .conversion_plan(&OriginalGgufMissFixture::request())
        .unwrap();
    let destinations = PreparedPendingWeight::host_destination_requests(&plan).unwrap();
    let layouts = fixture.source_storage_layouts();
    let bytes = layouts.iter().sum::<u64>() + cache::control_bytes().unwrap();
    // These stream handles and the neutral input are constructed outside the
    // original component. Only the new cache body is initialized below.
    let source_stream = fixture.stream.clone();
    let execution_stream = fixture.stream.clone();
    let input = fixture
        .source
        .acquire_lease(OriginalGgufMissFixture::request())
        .unwrap();
    let (group, weak, cache, pool) = component_destinations_with_retained_account(
        bytes,
        layouts.len() + 1,
        0,
        Some((destinations.bytes() as u64, destinations.calls(), 0)),
        existing,
        // A separate genuine initializer debits these exact quoted bytes. This
        // enlarges the common limit; it does not mint a receipt from headroom.
        |_| cache_bytes,
        cache_bytes,
        |controls, observer, pool, bank| {
            let before = pool.fixture_host_charge().unwrap();
            let cache = CacheHandle::prepare(pool).unwrap();
            assert_eq!(pool.fixture_host_charge().unwrap(), before + cache_bytes);
            let context = MlxParameterMaterializationContext::with_cache(
                source_stream,
                execution_stream,
                cache.clone(),
            );
            let pending = context
                .weight_lease(input)
                .unwrap()
                .prepare_original_gguf_fixture(
                    &fixture.stream,
                    fixture.admitted_ready(controls, bank),
                    observer.clone(),
                )
                .unwrap();
            fixture.settle(&pending, observer);
            let group = pending.retained.retention().group.as_ref().unwrap().clone();
            let weak = group.downgrade();
            assert_eq!(cache.try_lock().unwrap().len(), 1);
            assert_eq!(
                super::super::values(group.arrays.get("bank.weight").unwrap(), Some(observer)),
                fixture.expected["bank.weight"]
            );
            OriginalGgufMissFixture::finish(pending);
            (group, weak, cache, pool.clone())
        },
    );
    drop((fixture, plan, group));
    safemlx::reclaim_allocation_owners();
    assert!(weak.upgrade().is_none());
    assert_eq!(cache.try_lock().unwrap().len(), 1);
    let held = pool.fixture_host_charge().unwrap();
    assert!(held > existing + cache_bytes);
    // The stale supplied row is destroyed with the fixed capsule. An external
    // weak group still pins only the independent supplied metadata account H.
    drop(cache);
    assert_eq!(pool.fixture_host_charge().unwrap(), held - cache_bytes);
    drop(weak);
    settle_pool_at(&pool, existing);
}

#[cfg(test)]
#[allow(unused_imports)]
use crate::memory_fixture::LedgerFixture;
