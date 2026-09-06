use std::{
    sync::{mpsc, Arc, Barrier},
    time::Duration,
};

use eredu_checkpoint::store::{CheckpointSource, SafetensorsWeightStore, TensorSelection};
use eredu_runtime::{DeviceLayerWindow, ResidentLayerGroup};
use safemlx::{
    host_transfer_capacity_upper_bound, transforms::eval, Device, DeviceType, HostTransferPolicy,
    HostTransferStorageKind,
};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

use super::*;
use eredu_core::residency::{
    OffloadConfig, OffloadUnitSpec, ResidencyLedgerError, ResidencyPolicy,
};

fn cpu_stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn write_fixture(path: &std::path::Path) {
    let a = [1i32, 2]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let b = [3i32, 4]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let c = [5i32, 6]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    let matrix = [10i32, 11, 12, 13, 14, 15]
        .into_iter()
        .flat_map(i32::to_le_bytes)
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            ("a", TensorView::new(Dtype::I32, vec![2], &a).unwrap()),
            ("b", TensorView::new(Dtype::I32, vec![2], &b).unwrap()),
            ("c", TensorView::new(Dtype::I32, vec![2], &c).unwrap()),
            (
                "matrix",
                TensorView::new(Dtype::I32, vec![3, 2], &matrix).unwrap(),
            ),
        ],
        None,
        path,
    )
    .unwrap();
}

fn fixture_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
    let dir = tempfile::tempdir().unwrap();
    write_fixture(&dir.path().join("model.safetensors"));
    let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    (dir, store)
}

fn cross_shard_store() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
    let dir = tempfile::tempdir().unwrap();
    for (file, key, values) in [
        ("model-00001-of-00002.safetensors", "left", [1i32, 2]),
        ("model-00002-of-00002.safetensors", "right", [3i32, 4]),
    ] {
        let bytes = values
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        serialize_to_file(
            [(key, TensorView::new(Dtype::I32, vec![2], &bytes).unwrap())],
            None,
            &dir.path().join(file),
        )
        .unwrap();
    }
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "weight_map": {
                "left": "model-00001-of-00002.safetensors",
                "right": "model-00002-of-00002.safetensors"
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let store =
        Arc::new(SafetensorsWeightStore::open_with_max_cached_shards(dir.path(), 1).unwrap());
    (dir, store)
}

fn id(value: &str) -> OffloadUnitId {
    OffloadUnitId::new(value).unwrap()
}

fn binding(name: &str, key: &str, selection: TensorSelection, bytes: u64) -> WeightBinding {
    WeightBinding::new(name, key, selection, bytes).unwrap()
}

fn unit(name: &str, bindings: impl IntoIterator<Item = WeightBinding>) -> OffloadUnit {
    OffloadUnit::new(id(name), bindings).unwrap()
}

fn spec(name: &str, bytes: u64, policy: ResidencyPolicy, tier: MemoryTier) -> OffloadUnitSpec {
    OffloadUnitSpec::new(id(name), bytes, policy, tier).unwrap()
}

fn manager(
    store: Arc<SafetensorsWeightStore>,
    config: OffloadConfig,
    specs: impl IntoIterator<Item = OffloadUnitSpec>,
    units: impl IntoIterator<Item = OffloadUnit>,
) -> ResidencyManager {
    // Most fixtures below use one independently allocated eight-byte
    // binding as their accounting unit. Preserve those unit-count tests
    // while exercising the production contract, which charges the full
    // backing extent of every host-transfer allocation.
    let host_budget = config.host_budget_bytes().map(fixture_physical_host_budget);
    let config = OffloadConfig::new(
        config.device_budget_bytes(),
        host_budget,
        config.prefetch_depth(),
    )
    .unwrap()
    .with_eviction_policy(config.eviction_policy());
    ResidencyManager::new(
        store,
        OffloadPlan::new(config, specs).unwrap(),
        units,
        cpu_stream(),
        cpu_stream(),
    )
    .unwrap()
}

fn fixture_physical_host_budget(logical_bytes: u64) -> u64 {
    if logical_bytes == 0 {
        return 0;
    }
    let minimum_capacity =
        host_transfer_capacity_upper_bound(1, HostTransferPolicy::Transfer).unwrap() as u64;
    if logical_bytes >= minimum_capacity {
        return logical_bytes;
    }
    let complete_bindings = logical_bytes / 8;
    let remainder = logical_bytes % 8;
    let binding_capacity = fixture_binding_capacity();
    complete_bindings
        .checked_mul(binding_capacity)
        .and_then(|bytes| {
            (remainder != 0)
                .then(|| {
                    host_transfer_capacity_upper_bound(
                        remainder as usize,
                        HostTransferPolicy::Transfer,
                    )
                    .unwrap() as u64
                })
                .map_or(Some(bytes), |tail| bytes.checked_add(tail))
        })
        .unwrap()
}

fn fixture_binding_capacity() -> u64 {
    host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64
}

fn fixture_host_capacity(bindings: u64) -> u64 {
    bindings.checked_mul(fixture_binding_capacity()).unwrap()
}

fn single(name: &str, key: &str) -> OffloadUnit {
    unit(name, [binding("weight", key, TensorSelection::Full, 8)])
}

fn host_i32(lease: &ResidentUnitLease, name: &str) -> Vec<i32> {
    lease
        .host_value(name)
        .unwrap()
        .as_bytes()
        .unwrap()
        .as_chunks::<{ size_of::<i32>() }>()
        .0
        .iter()
        .map(|bytes| i32::from_ne_bytes(*bytes))
        .collect()
}

include!("tests/lifecycle.rs");
include!("tests/transfer.rs");
include!("tests/materialization.rs");
include!("tests/accounting.rs");
