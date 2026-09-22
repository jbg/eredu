use crate::memory_fixture::LedgerFixture;
use std::{collections::BTreeMap, sync::Arc};

use eredu_checkpoint::store::{CheckpointSource, SafetensorsWeightStore, TensorSelection};
use eredu_core::residency::{OffloadConfig, OffloadPlan, OffloadUnitSpec, ResidencyPolicy};
use safemlx::{Device, DeviceType, HostTransferBuffer, Stream};
use safetensors::tensor::{serialize_to_file, TensorView};

use super::*;
use crate::backend::runtime::residency::manager::ResidentHostBuffers;

fn id(name: &str) -> OffloadUnitId {
    OffloadUnitId::new(name).unwrap()
}

fn fixture(
    aliases: bool,
    device: DeviceType,
) -> (
    tempfile::TempDir,
    Arc<SafetensorsWeightStore>,
    ResidencyManager,
) {
    // These fixtures share the native process with the paid source test.
    // Establish its genuine allocator owner before any ordinary device use.
    crate::tests::support::test_utils::initialize_original_sources();
    let dir = tempfile::tempdir().unwrap();
    let a = [1_i32, 2]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let b = [3_i32, 4]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "a",
                TensorView::new(safetensors::Dtype::I32, vec![2], &a).unwrap(),
            ),
            (
                "b",
                TensorView::new(safetensors::Dtype::I32, vec![2], &b).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let owner = WeightBinding::new("weight", "a", TensorSelection::Full, 8)
        .unwrap()
        .with_logical_target("shared.weight")
        .unwrap();
    let mut units = vec![OffloadUnit::new(id("owner"), [owner]).unwrap()];
    if aliases {
        units.push(
            OffloadUnit::new(
                id("alias"),
                [
                    WeightBinding::alias("shared", "shared.weight", 8).unwrap(),
                    WeightBinding::new("local", "b", TensorSelection::Full, 8).unwrap(),
                ],
            )
            .unwrap(),
        );
    }
    let specs = units.iter().map(|unit| {
        OffloadUnitSpec::new(
            unit.id().clone(),
            8,
            ResidencyPolicy::Cacheable,
            MemoryTier::Disk,
        )
        .unwrap()
    });
    let plan = OffloadPlan::new(OffloadConfig::new(None, None, 1).unwrap(), specs).unwrap();
    let manager = ResidencyManager::new(
        source.clone(),
        plan,
        units,
        Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)),
        Stream::new_with_device(&Device::new(device, 0)),
    )
    .unwrap();
    manager.initialize().unwrap();
    (dir, source, manager)
}

fn facts() -> NativeAllocationFacts {
    NativeAllocationFacts::current_host().unwrap()
}

fn assert_unknown(result: Result<HostCopyWorkspace, HostCopyWorkspaceError>) {
    assert!(
        matches!(
            result,
            Err(HostCopyWorkspaceError::Unproved {
                source: WorkingMemoryError::UnknownBound,
                ..
            })
        ),
        "{result:?}"
    );
}

fn assert_mismatch(result: Result<HostCopyWorkspace, HostCopyWorkspaceError>) {
    assert!(
        matches!(
            result,
            Err(HostCopyWorkspaceError::Unproved {
                source: WorkingMemoryError::IdentityMismatch,
                ..
            })
        ),
        "{result:?}"
    );
}

#[test]
fn host_copy_snapshot_counts_alias_dispatches_but_deduplicates_sources() {
    let (_dir, store, manager) = fixture(true, DeviceType::Gpu);
    drop(manager.acquire(&id("alias"), MemoryTier::Host).unwrap());
    let ids = [id("owner"), id("alias")];
    let before = store.source_diagnostics().unwrap();
    let snapshot = manager.host_copy_workspace(&ids, facts()).unwrap();
    snapshot.validate_sources(&manager).unwrap();
    let repeated = manager.host_copy_workspace(&ids, facts()).unwrap();
    assert_eq!(
        store.source_diagnostics().unwrap(),
        before,
        "inspection never reads checkpoint payloads"
    );
    assert_eq!(snapshot.units().len(), 2);
    assert_eq!(snapshot.copies(&snapshot.units()[0]).len(), 1);
    assert_eq!(
        snapshot
            .copies(&snapshot.units()[1])
            .iter()
            .map(HostCopyBinding::name)
            .collect::<Vec<_>>(),
        ["local", "shared"]
    );
    let owner = &snapshot.copies(&snapshot.units()[0])[0];
    let local = &snapshot.copies(&snapshot.units()[1])[0];
    let alias = &snapshot.copies(&snapshot.units()[1])[1];
    assert_eq!(owner.source_allocation(), alias.source_allocation());
    assert_ne!(
        owner.source_allocation().identity(),
        local.source_allocation().identity()
    );
    assert_eq!(
        snapshot.unique_source_bytes(),
        (owner.source_allocation().bytes() + local.source_allocation().bytes()) as u64
    );
    assert_eq!(
        snapshot.fresh_capacity_bytes(),
        3 * facts().buffer_capacity(8).unwrap()
    );
    assert_eq!(
        snapshot.units()[1].fresh_capacity_bytes(),
        2 * facts().buffer_capacity(8).unwrap()
    );
    assert_eq!(
        snapshot.fresh_capacity_bytes(),
        repeated.fresh_capacity_bytes()
    );
    assert_eq!(snapshot.host_staging_bytes(), 0);
    let reversed = manager
        .host_copy_workspace(&[id("alias"), id("owner")], facts())
        .unwrap();
    assert_eq!(reversed.units()[0].id(), &id("alias"));
    assert_eq!(reversed.units()[1].id(), &id("owner"));
    assert_eq!(
        reversed
            .copies(&reversed.units()[0])
            .iter()
            .map(HostCopyBinding::name)
            .collect::<Vec<_>>(),
        ["local", "shared"]
    );
    assert_eq!(
        reversed.unique_source_bytes(),
        snapshot.unique_source_bytes()
    );
    assert_eq!(
        reversed.fresh_capacity_bytes(),
        snapshot.fresh_capacity_bytes()
    );
    let duplicate = manager.host_copy_workspace(&[id("owner"), id("alias"), id("owner")], facts());
    assert!(matches!(
        duplicate,
        Err(HostCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));

    // Native copies are deliberate validation outside the cold snapshot.
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for unit in snapshot.units() {
        assert_eq!(unit.id(), unit.definition().id());
        for copy in snapshot.copies(unit) {
            assert_eq!(copy.shape(), [2]);
            assert_eq!(copy.dtype(), Dtype::Int32);
            assert_eq!(copy.logical_bytes(), 8);
            assert_eq!(copy.binding().name(), copy.name());
            let output = copy
                .source()
                .copy_to_array(&stream)
                .unwrap()
                .synchronize()
                .unwrap();
            let ready = output.evaluated().unwrap();
            let allocation = output.allocation_info().unwrap().unwrap();
            assert!(allocation.bytes() as u64 <= copy.output_capacity_bytes());
            if allocation.identity() != copy.source_allocation().identity() {
                assert!(allocation.bytes() as u64 <= copy.fresh_capacity_bytes());
            }
            assert_eq!(
                ready.as_slice::<i32>(),
                if copy.name() == "local" {
                    &[3, 4]
                } else {
                    &[1, 2]
                }
            );
        }
    }
    drop(manager);
    assert_eq!(
        owner.source().as_bytes().unwrap().len(),
        8,
        "snapshot retains immutable sources after manager retirement"
    );
}

#[test]
fn missing_or_inflight_sources_reject_without_materializing_or_settling() {
    let (_dir, store, manager) = fixture(false, DeviceType::Gpu);
    let before = store.source_diagnostics().unwrap();
    assert_unknown(manager.host_copy_workspace(&[id("owner")], facts()));
    assert_eq!(store.source_diagnostics().unwrap(), before);
    drop(manager.acquire(&id("owner"), MemoryTier::Host).unwrap());
    let snapshot = manager
        .host_copy_workspace(&[id("owner")], facts())
        .unwrap();
    let before_pins = host_pin_counts(&manager, &[id("owner")]);
    let mut transfer = manager
        .acquire_many_with_transfer(&[(id("owner"), 1)], MemoryTier::Device)
        .unwrap();
    let before = store.source_diagnostics().unwrap();
    assert_unknown(manager.host_copy_workspace(&[id("owner")], facts()));
    assert!(matches!(
        snapshot.pin_sources(&manager),
        Err(HostCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::UnknownBound,
            ..
        })
    ));
    assert_eq!(host_pin_counts(&manager, &[id("owner")]), before_pins);
    assert!(manager
        .inner
        .state
        .lock()
        .unwrap()
        .control
        .ledger()
        .copy_status(&id("owner"), MemoryTier::Device)
        .unwrap()
        .unwrap()
        .in_flight()
        .is_some());
    assert_eq!(store.source_diagnostics().unwrap(), before);
    transfer.synchronize().unwrap();
    manager
        .host_copy_workspace(&[id("owner")], facts())
        .unwrap();
}

#[test]
fn failed_manager_does_not_receive_copy_facts() {
    let (_dir, _store, manager) = fixture(false, DeviceType::Gpu);
    drop(manager.acquire(&id("owner"), MemoryTier::Host).unwrap());
    let snapshot = manager
        .host_copy_workspace(&[id("owner")], facts())
        .unwrap();
    let before_pins = host_pin_counts(&manager, &[id("owner")]);
    manager.inner.failed_transfer.store(true, Ordering::Release);
    assert_unknown(manager.host_copy_workspace(&[id("owner")], facts()));
    assert!(matches!(
        snapshot.pin_sources(&manager),
        Err(HostCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::UnknownBound,
            ..
        })
    ));
    assert_eq!(host_pin_counts(&manager, &[id("owner")]), before_pins);
    manager
        .inner
        .failed_transfer
        .store(false, Ordering::Release);
    manager
        .host_copy_workspace(&[id("owner")], facts())
        .unwrap();
}

#[test]
fn host_copy_snapshot_retains_cpu_destination_and_rejects_a_changed_stream() {
    let (_dir, _store, manager) = fixture(true, DeviceType::Cpu);
    drop(manager.acquire(&id("alias"), MemoryTier::Host).unwrap());
    let snapshot = manager
        .host_copy_workspace(&[id("owner"), id("alias")], facts())
        .unwrap();
    assert_eq!(snapshot.destination_device_type(), DeviceType::Cpu);
    snapshot.validate_sources(&manager).unwrap();
    let before_pins = host_pin_counts(&manager, &[id("owner"), id("alias")]);
    let pins = snapshot.pin_sources(&manager).unwrap();
    for unit in snapshot.units() {
        for copy in snapshot.copies(unit) {
            let stream = manager.inner.state.lock().unwrap().device_stream.clone();
            let array = copy
                .source()
                .copy_to_array(&stream)
                .unwrap()
                .synchronize()
                .unwrap();
            assert_eq!(
                array.evaluated().unwrap().as_slice::<i32>(),
                if copy.name() == "local" {
                    &[3, 4]
                } else {
                    &[1, 2]
                }
            );
        }
    }
    drop(pins);
    crate::backend::ordinary_retirement::reclaim();
    assert_eq!(
        host_pin_counts(&manager, &[id("owner"), id("alias")]),
        before_pins
    );
    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        let replacement = super::super::owner::ManagerStream::Ordinary(Stream::new_with_device(
            &Device::new(device, 0),
        ));
        let old = {
            let mut state = manager.inner.state.lock().unwrap();
            std::mem::replace(&mut state.device_stream, replacement)
        };
        assert!(matches!(
            snapshot.validate_sources(&manager),
            Err(HostCopyWorkspaceError::Unproved {
                source: WorkingMemoryError::IdentityMismatch,
                ..
            })
        ));
        assert!(matches!(
            snapshot.pin_sources(&manager),
            Err(HostCopyWorkspaceError::Unproved {
                source: WorkingMemoryError::IdentityMismatch,
                ..
            })
        ));
        assert_eq!(
            host_pin_counts(&manager, &[id("owner"), id("alias")]),
            before_pins
        );
        let replacement = {
            let mut state = manager.inner.state.lock().unwrap();
            std::mem::replace(&mut state.device_stream, old)
        };
        drop(replacement);
    }
    snapshot.validate_sources(&manager).unwrap();
}

#[test]
fn cpu_host_copy_source_shape_matches_native_signed_index_geometry() {
    assert!(copy_shape_is_supported(DeviceType::Cpu, &[2, 3, 4]));
    assert!(copy_shape_is_supported(DeviceType::Cpu, &[i32::MAX]));
    for shape in [&[0][..], &[-1][..], &[i32::MAX, 2][..]] {
        assert!(!copy_shape_is_supported(DeviceType::Cpu, shape));
    }
}

#[test]
fn source_replacement_and_geometry_mismatch_fail_without_repair() {
    let (_dir, store, manager) = fixture(false, DeviceType::Gpu);
    drop(manager.acquire(&id("owner"), MemoryTier::Host).unwrap());
    let snapshot = manager
        .host_copy_workspace(&[id("owner")], facts())
        .unwrap();
    for (shape, dtype) in [
        (&[1, 2][..], Dtype::Int32),
        (&[2][..], Dtype::Uint32),
        (&[2][..], Dtype::Int32),
    ] {
        let mut replacement =
            HostTransferBuffer::new(shape, dtype, HostTransferPolicy::Transfer).unwrap();
        replacement.as_bytes_mut().unwrap().copy_from_slice(
            &[1_i32, 2]
                .into_iter()
                .flat_map(i32::to_ne_bytes)
                .collect::<Vec<_>>(),
        );
        let replacement = Arc::new(replacement.freeze()).into();
        let previous = manager
            .inner
            .state
            .lock()
            .unwrap()
            .storage
            .get_mut(&id("owner"))
            .unwrap()
            .host
            .replace(
                Arc::new(ResidentHostBuffers {
                    buffers: [("weight".to_owned(), replacement)].into_iter().collect(),
                })
                .into(),
            );
        drop(previous);
        let before = store.source_diagnostics().unwrap();
        if shape.len() == 2 || dtype != Dtype::Int32 {
            assert_mismatch(manager.host_copy_workspace(&[id("owner")], facts()));
        } else {
            manager
                .host_copy_workspace(&[id("owner")], facts())
                .unwrap();
            assert!(matches!(
                snapshot.validate_sources(&manager),
                Err(HostCopyWorkspaceError::Unproved {
                    source: WorkingMemoryError::IdentityMismatch,
                    ..
                })
            ));
        }
        assert_eq!(store.source_diagnostics().unwrap(), before);
    }
    assert_mismatch(manager.host_copy_workspace(&[id("owner"), id("owner")], facts()));
    assert_mismatch(manager.host_copy_workspace(&[id("missing")], facts()));
}

#[test]
fn poisoned_manager_lock_remains_an_error() {
    let (_dir, _store, manager) = fixture(false, DeviceType::Gpu);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _state = manager.inner.state.lock().unwrap();
        panic!("poison fixture");
    }));
    assert!(matches!(
        manager.host_copy_workspace(&[id("owner")], facts()),
        Err(HostCopyWorkspaceError::Residency(
            ResidencyError::StatePoisoned
        ))
    ));
}

fn host_pin_counts(manager: &ResidencyManager, ids: &[OffloadUnitId]) -> Vec<u64> {
    let state = manager.inner.state.lock().unwrap();
    ids.iter()
        .map(|id| {
            state
                .control
                .ledger()
                .copy_status(id, MemoryTier::Host)
                .unwrap()
                .unwrap()
                .pins()
        })
        .collect()
}

#[test]
fn ready_source_guard_blocks_eviction_until_ordinary_retirement() {
    let (_dir, store, manager) = fixture(true, DeviceType::Gpu);
    drop(manager.acquire(&id("alias"), MemoryTier::Host).unwrap());
    let ids = [id("owner"), id("alias")];
    let snapshot = manager.host_copy_workspace(&ids, facts()).unwrap();
    let before_pins = host_pin_counts(&manager, &ids);
    let before_reads = store.source_diagnostics().unwrap();
    let pins = snapshot.pin_sources(&manager).unwrap();
    assert_eq!(
        host_pin_counts(&manager, &ids),
        before_pins
            .iter()
            .map(|count| count + 1)
            .collect::<Vec<_>>()
    );
    assert_eq!(store.source_diagnostics().unwrap(), before_reads);
    snapshot.validate_sources(&manager).unwrap();
    assert!(matches!(
        manager.evict(&id("alias"), MemoryTier::Host),
        Err(ResidencyError::Ledger(
            eredu_core::residency::ResidencyLedgerError::InUseEviction { .. }
        ))
    ));
    // Drop under the state lock must only enqueue custody, not try to unpin.
    {
        let state = manager.inner.state.lock().unwrap();
        drop(pins);
        assert_eq!(
            state
                .control
                .ledger()
                .copy_status(&id("alias"), MemoryTier::Host)
                .unwrap()
                .unwrap()
                .pins(),
            before_pins[1] + 1
        );
    }
    // A cold query must not use manager.lock(), which would reclaim these
    // queued leases and silently change custody while inspecting readiness.
    manager.host_copy_workspace(&ids, facts()).unwrap();
    snapshot.validate_sources(&manager).unwrap();
    assert_eq!(
        host_pin_counts(&manager, &ids),
        before_pins
            .iter()
            .map(|count| count + 1)
            .collect::<Vec<_>>()
    );
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(host_pin_counts(&manager, &ids), before_pins);
    assert!(manager.evict(&id("alias"), MemoryTier::Host).unwrap());
    // The cold descriptor owns sources but intentionally grants no eviction protection.
    assert_eq!(
        snapshot.copies(&snapshot.units()[1])[0]
            .source()
            .nbytes()
            .unwrap(),
        8
    );
}

#[test]
fn source_guard_rejects_missing_and_replaced_units_before_any_pin() {
    let (_dir, store, manager) = fixture(true, DeviceType::Gpu);
    drop(manager.acquire(&id("alias"), MemoryTier::Host).unwrap());
    let ids = [id("owner"), id("alias")];
    let snapshot = manager.host_copy_workspace(&ids, facts()).unwrap();
    let before_pins = host_pin_counts(&manager, &ids);
    let before_reads = store.source_diagnostics().unwrap();
    let original = manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("alias"))
        .unwrap()
        .host
        .take()
        .unwrap();
    // The snapshot already authenticated this exact immutable owner. Its
    // removal is a changed source identity, not an unproved initial source.
    let missing = snapshot.pin_sources(&manager);
    assert!(
        matches!(
            missing,
            Err(HostCopyWorkspaceError::Unproved {
                source: WorkingMemoryError::IdentityMismatch,
                ..
            })
        ),
        "{missing:?}"
    );
    assert!(matches!(
        snapshot.validate_sources(&manager),
        Err(HostCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    assert_eq!(host_pin_counts(&manager, &ids), before_pins);
    assert_eq!(store.source_diagnostics().unwrap(), before_reads);

    let mut replacement =
        HostTransferBuffer::new(&[2], Dtype::Int32, HostTransferPolicy::Transfer).unwrap();
    replacement.as_bytes_mut().unwrap().copy_from_slice(
        &[3_i32, 4]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    let mut replaced_buffers = original.buffers.clone();
    replaced_buffers.insert("local".to_owned(), Arc::new(replacement.freeze()).into());
    manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("alias"))
        .unwrap()
        .host = Some(
        Arc::new(ResidentHostBuffers {
            buffers: replaced_buffers,
        })
        .into(),
    );
    assert!(matches!(
        snapshot.pin_sources(&manager),
        Err(HostCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    assert_eq!(host_pin_counts(&manager, &ids), before_pins);
    assert_eq!(store.source_diagnostics().unwrap(), before_reads);
    let replaced = manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&id("alias"))
        .unwrap()
        .host
        .replace(original);
    drop(replaced);
    let pins = snapshot.pin_sources(&manager).unwrap();
    drop(pins);
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(host_pin_counts(&manager, &ids), before_pins);
}

#[test]
fn retained_source_guard_keeps_exact_owners_and_defers_final_unpin() {
    let (_dir, store, manager) = fixture(true, DeviceType::Gpu);
    drop(manager.acquire(&id("alias"), MemoryTier::Host).unwrap());
    let ids = [id("owner"), id("alias")];
    let snapshot = manager.host_copy_workspace(&ids, facts()).unwrap();
    let before = host_pin_counts(&manager, &ids);
    let reads = store.source_diagnostics().unwrap();
    let pins = snapshot.pin_sources_impl(&manager, None).unwrap();
    assert_eq!(
        host_pin_counts(&manager, &ids),
        before.iter().map(|n| n + 1).collect::<Vec<_>>()
    );
    assert_eq!(store.source_diagnostics().unwrap(), reads);
    {
        let state = manager.inner.state.lock().unwrap();
        drop(pins);
        assert_eq!(
            state
                .control
                .ledger()
                .copy_status(&ids[0], MemoryTier::Host)
                .unwrap()
                .unwrap()
                .pins(),
            before[0] + 1
        );
    }
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(host_pin_counts(&manager, &ids), before);

    // Identical shape and numerical bytes are not the retained immutable owner.
    let mut replacement =
        HostTransferBuffer::new(&[2], Dtype::Int32, HostTransferPolicy::Transfer).unwrap();
    replacement
        .as_bytes_mut()
        .unwrap()
        .copy_from_slice(&[3_i32.to_ne_bytes(), 4_i32.to_ne_bytes()].concat());
    let original = {
        let mut state = manager.inner.state.lock().unwrap();
        let row = state.storage.get_mut(&ids[1]).unwrap();
        let original = row.host.take().unwrap();
        let mut buffers = original.buffers.clone();
        buffers.insert("local".into(), Arc::new(replacement.freeze()).into());
        row.host = Some(Arc::new(ResidentHostBuffers { buffers }).into());
        original
    };
    assert!(matches!(
        snapshot.pin_sources_impl(&manager, None),
        Err(HostCopyWorkspaceError::Unproved {
            source: WorkingMemoryError::IdentityMismatch,
            ..
        })
    ));
    assert_eq!(host_pin_counts(&manager, &ids), before);
    let replaced = manager
        .inner
        .state
        .lock()
        .unwrap()
        .storage
        .get_mut(&ids[1])
        .unwrap()
        .host
        .replace(original);
    drop(replaced);
    assert_eq!(store.source_diagnostics().unwrap(), reads);
    drop(snapshot.pin_sources_impl(&manager, None).unwrap());
    crate::backend::ordinary_retirement::reclaim_all();
    assert_eq!(host_pin_counts(&manager, &ids), before);
}

#[test]
fn paid_cpu_host_and_disk_sources_retain_the_selected_copy_destination() {
    use eredu_core::residency::OffloadUnitSpec;
    use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec, ExecutionUnitLayout};
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let dir = tempfile::tempdir().unwrap();
    let values = [1.25_f32, -3.5, 2.0, 0.125];
    let bytes = values
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [(
            "weight",
            TensorView::new(safetensors::Dtype::F32, vec![2, 2], &bytes).unwrap(),
        )],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let source = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let source_stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let destination = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let units = [OffloadUnit::new(
        id("owner"),
        [WeightBinding::new("weight", "weight", TensorSelection::Full, 16).unwrap()],
    )
    .unwrap()];
    let ids = [id("owner")];
    let groups = ["layers".to_owned()];
    let graph = ExecutionGraph::new(vec![ExecutionGroupSpec::root("layers")], "layers").unwrap();
    let layout = ExecutionUnitLayout::new(&graph, [1]).unwrap();
    crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
    safemlx::memory::clear_cache().unwrap();
    safemlx::reclaim_allocation_owners();
    let baseline = pool.fixture_host_charge().unwrap();
    for tier in [MemoryTier::Host, MemoryTier::Disk] {
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
        let plan = OffloadPlan::new(
            OffloadConfig::new(None, None, 1).unwrap(),
            [OffloadUnitSpec::new(ids[0].clone(), 16, ResidencyPolicy::Cacheable, tier).unwrap()],
        )
        .unwrap();
        let prepare = if tier == MemoryTier::Host {
            ResidencyManager::prepare_original_host
        } else {
            ResidencyManager::prepare_original_foreground_disk
        };
        let manager = prepare(
            source.clone().into(),
            BTreeMap::new(),
            &plan,
            &units,
            &groups,
            &ids,
            &layout,
            1,
            &std::collections::BTreeSet::new(),
            None,
            &source_stream,
            &destination,
            &pool,
        )
        .unwrap()
        .expect("CPU must retain the genuine paid source manager");
        assert!(manager.original_source_custody().is_some());
        assert!(
            pool.fixture_host_charge().unwrap() > baseline,
            "this manager must retain its actual source charge"
        );
        if tier == MemoryTier::Host {
            let copies = manager.prepared_host_copy_workspace(&ids, facts()).unwrap();
            assert_eq!(copies.destination_device_type(), DeviceType::Cpu);
            assert!(copies.prepared_identity().is_some());
            copies.validate_sources(&manager).unwrap();
            let copy = &copies.copies(&copies.units()[0])[0];
            assert_eq!(copy.dtype(), Dtype::Float32);
            assert_eq!(copy.shape(), [2, 2]);
            let output = copy
                .source()
                .copy_to_array(&destination)
                .unwrap()
                .synchronize()
                .unwrap();
            assert_eq!(output.evaluated().unwrap().as_slice::<f32>(), values);
        } else {
            let identity = manager
                .original_foreground_workspace()
                .expect("CPU disk keeps the source-owned foreground worker");
            assert_eq!(identity.destination_device_type(), DeviceType::Cpu);
            assert!(identity.matches_selection(&ids, &layout, 1));
            let (_, shape, dtype, _, logical, _, _) = identity.row(0, 0).unwrap();
            assert_eq!(shape, [2, 2]);
            assert_eq!(dtype, Dtype::Float32);
            assert_eq!(logical, 16);
            assert!(identity.materialization().bytes().unwrap() >= logical);
            assert_eq!(manager.detached_physical_read_bytes(0), Some(0));
        }
        drop(manager);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            crate::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
            safemlx::memory::clear_cache().unwrap();
            safemlx::reclaim_allocation_owners();
            pool.fixture_host_charge().unwrap() == baseline
        });
        assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
    }
}
