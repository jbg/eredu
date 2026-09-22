use super::*;
use crate::memory_fixture::LedgerFixture;

#[test]
fn admitted_cache_transfer_stream_shares_ownership_and_rejects_foreign_ledger() {
    if !crate::tests::support::native_process::enter("cache-transfer-stream-owner") {
        return;
    }
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let ledger = crate::backend::managed_memory::try_ledger().unwrap();
    let execution = PreparedExecutionStreams::for_device_factory(&ledger, DeviceType::Cpu)
        .unwrap()
        .expect("actual admitted CPU stream factory");
    let manager = super::super::CacheResidencyManager::new(
        eredu_runtime::PagedCacheOptions::new(2, 4096, 4096, 1).unwrap(),
    )
    .unwrap();
    let before = ledger.fixture_host_charge().unwrap();
    manager
        .prepare_transfer_stream(&ledger, execution.execution())
        .unwrap();
    let source = manager.prepared_transfer_stream().unwrap();
    let alias = source.clone();
    let held = ledger.fixture_host_charge().unwrap();
    let retiring = {
        let owner = source.0.as_ref().unwrap();
        owner
            .original_bytes()
            .checked_add(
                owner
                    .output()
                    .lock()
                    .unwrap()
                    .retiring_wrapper_control_bytes(),
            )
            .unwrap()
    };
    assert!(held > before);
    assert!(held.checked_sub(before).unwrap() > retiring);
    assert_eq!(
        source.registry_owner_account().unwrap().1,
        held.checked_sub(before)
            .unwrap()
            .checked_sub(retiring)
            .unwrap()
    );
    manager
        .prepare_transfer_stream(&ledger, execution.execution())
        .unwrap();
    assert_eq!(ledger.fixture_host_charge().unwrap(), held);
    let transfer = source
        .with_stream(&ledger, execution.execution(), |stream| {
            stream.get_index().unwrap()
        })
        .unwrap();
    assert_ne!(transfer, execution.execution().get_index().unwrap());

    let foreign = crate::memory_fixture::ledger(1 << 24, 0).unwrap();
    assert!(matches!(
        source.with_stream(&foreign, execution.execution(), |_| ()),
        Err(CacheTransferStreamError::Accounting(
            WorkingMemoryError::IdentityMismatch
        ))
    ));
    drop((manager, source));
    safemlx::reclaim_allocation_owners();
    assert_eq!(ledger.fixture_host_charge().unwrap(), held);
    alias
        .with_stream(&ledger, execution.execution(), |_| ())
        .unwrap();
    drop(alias);
    safemlx::reclaim_allocation_owners();
    assert_eq!(
        ledger.fixture_host_charge().unwrap(),
        held.checked_sub(retiring).unwrap(),
        "the final alias frees its closed host wrappers; the native CPU registry still owns its streams and workers"
    );
}

#[test]
fn prepared_cache_transfer_handoff_preserves_live_loading_guard_and_installed_owner() {
    if !crate::tests::support::native_process::enter("cache-transfer-cold-handoff") {
        return;
    }
    let _sources = crate::tests::support::test_utils::initialize_original_sources();
    let ledger = crate::backend::managed_memory::try_ledger().unwrap();
    let execution = PreparedExecutionStreams::for_device_factory(&ledger, DeviceType::Cpu)
        .unwrap()
        .expect("actual admitted CPU stream factory");
    let selected = PreparedCacheTransferStream::prepare(&ledger, execution.execution()).unwrap();
    let other = PreparedCacheTransferStream::prepare(&ledger, execution.execution()).unwrap();
    let manager = super::super::CacheResidencyManager::new(
        eredu_runtime::PagedCacheOptions::new(2, 4096, 4096, 1).unwrap(),
    )
    .unwrap();
    let loading = crate::backend::managed_memory::NativeMemoryOwner::acquire(&ledger).unwrap();
    assert_eq!(ledger.unquoted_owner_count().unwrap(), 1);
    let charge = ledger.fixture_host_charge().unwrap();
    manager
        .install_transfer_stream(&selected, &ledger, execution.execution())
        .unwrap();
    manager
        .install_transfer_stream(&selected, &ledger, execution.execution())
        .unwrap();
    assert_eq!(ledger.fixture_host_charge().unwrap(), charge);
    assert!(matches!(
        manager.install_transfer_stream(&other, &ledger, execution.execution()),
        Err(CacheTransferStreamError::ForeignSource)
    ));
    let retained = manager.prepared_transfer_stream().unwrap();
    assert!(Arc::ptr_eq(
        retained.0.as_ref().unwrap(),
        selected.0.as_ref().unwrap()
    ));
    assert_eq!(ledger.fixture_host_charge().unwrap(), charge);
    assert_eq!(ledger.unquoted_owner_count().unwrap(), 1);
    drop(loading);
    assert_eq!(ledger.unquoted_owner_count().unwrap(), 0);
}
