use std::sync::Arc;

use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointSource, SafetensorsWeightStore},
};
use safemlx::{
    host_transfer_capacity_upper_bound, Device, DeviceType, HostTransferPolicy,
    HostTransferStorageKind,
};
use safetensors::tensor::{serialize_to_file, Dtype as StoredDtype, TensorView};

use super::acquisition::preflight_selected_entry_bindings;
use super::*;
use crate::tests::support::grouped_provider::ParameterBankSelection;
use eredu_core::residency::CacheEvictionPolicy;
use eredu_runtime::WeightBinding;

fn stream() -> Stream {
    Stream::new_with_device(&Device::new(DeviceType::Cpu, 0))
}

fn fixture() -> (tempfile::TempDir, Arc<SafetensorsWeightStore>) {
    let dir = tempfile::tempdir().unwrap();
    let values = [[1i32, 2], [3, 4], [5, 6]]
        .into_iter()
        .map(|values| {
            values
                .into_iter()
                .flat_map(i32::to_le_bytes)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    serialize_to_file(
        values.iter().enumerate().map(|(entry, bytes)| {
            (
                format!("entry.{entry}"),
                TensorView::new(StoredDtype::I32, vec![1, 2], bytes).unwrap(),
            )
        }),
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    (dir, store)
}

fn entries() -> Vec<ParameterBankEntry> {
    (0..3)
        .map(|entry| {
            let identity = ParameterBankKey::new(0, 2, entry);
            let bindings = [
                WeightBinding::new("weight", format!("entry.{entry}"), TensorSelection::Full, 8)
                    .unwrap(),
                WeightBinding::new("scale", format!("entry.{entry}"), TensorSelection::Full, 8)
                    .unwrap(),
            ];
            let unit = OffloadUnit::new(identity.unit_id(), bindings).unwrap();
            ParameterBankEntry::new(identity, unit, 16).unwrap()
        })
        .collect()
}

#[test]
fn independent_bank_scopes_share_one_budget_and_reacquire_exact_companions() {
    let (_dir, store) = fixture();
    let execution = stream();
    let entries = (0..2)
        .flat_map(|bank| {
            entries().into_iter().map(move |entry| {
                let key =
                    ParameterBankKey::new(bank, entry.identity.unit(), entry.identity.member());
                ParameterBankEntry::new(
                    key,
                    OffloadUnit::new(key.unit_id(), entry.unit.bindings().to_vec()).unwrap(),
                    entry.bytes,
                )
                .unwrap()
            })
        })
        .collect::<Vec<_>>();
    let pool = SharedAddressableParameterBank::new(
        AddressableParameterBank::new(
            store,
            entries,
            ParameterBankOptions::new(OffloadConfig::new(Some(16), Some(32), 1).unwrap(), 16, 16)
                .unwrap(),
            stream(),
            execution.clone(),
        )
        .unwrap(),
    );
    let mut scopes = [pool.scoped(0).unwrap(), pool.scoped(1).unwrap()];
    for (bank, member) in [(0, 0), (1, 2), (0, 0), (1, 1)] {
        let key = ParameterBankKey::new(bank, 2, member);
        assert_eq!(scopes[1 - bank].member_bytes(key), None);
        let entries = [(key, 1)];
        let request = ParameterBankAcquisition::new(&entries, ParameterBankAccess::Incremental);
        let acquired = scopes[bank].acquire(request, &execution).unwrap();
        if bank == 0 && member == 0 {
            assert!(scopes[1].acquire(request, &execution).is_err());
            let competing = [(ParameterBankKey::new(1, 2, 2), 1)];
            let request =
                ParameterBankAcquisition::new(&competing, ParameterBankAccess::Incremental);
            assert!(
                scopes[1].acquire(request, &execution).is_err(),
                "live leases must prevent cross-bank overcommit"
            );
        }
        let weights = acquired
            .compact_binding("weight", &execution)
            .unwrap()
            .into_evaluated()
            .unwrap();
        let scales = acquired
            .compact_binding("scale", &execution)
            .unwrap()
            .into_evaluated()
            .unwrap();
        assert_eq!(
            weights.as_slice::<i32>(),
            &[(member * 2 + 1) as i32, (member * 2 + 2) as i32]
        );
        assert_eq!(weights.as_slice::<i32>(), scales.as_slice::<i32>());
        scopes[bank]
            .complete(
                acquired,
                &MlxTensor::from_array(weights.into_array().unwrap()),
                &execution,
            )
            .unwrap();
    }
    let report = ParameterBanksResidencyReport::new(
        scopes
            .iter()
            .enumerate()
            .map(|(bank, scope)| {
                (
                    eredu_runtime::RoutedBankId::new(bank as u32),
                    scope.report().unwrap(),
                )
            })
            .collect(),
    );
    assert_eq!(report.owned_entries(), 6);
    assert_eq!(report.owned_bytes(), 96);
    assert_eq!(report.peak_device_resident_bytes(), 16);
    assert!(report.peak_host_resident_bytes() <= 32);
    assert_eq!(report.device_resident_entries(), 1);
    assert_eq!(report.incremental().device().misses(), 4);
    assert_eq!(report.incremental().device().evictions(), 3);
    for bank in report.banks().values() {
        assert_eq!(bank.owned_entries(), 3);
        assert_eq!(bank.incremental().distinct_entries(), 2);
    }
}

#[test]
fn bank_preflight_rejects_every_entry_before_payload_reads_or_transforms() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = [0u8; 2];
    serialize_to_file(
        [(
            "unsupported",
            TensorView::new(StoredDtype::F8_E5M2, vec![2], &bytes).unwrap(),
        )],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store = SafetensorsWeightStore::open(dir.path()).unwrap();
    let key = ParameterBankKey::new(0, 0, 0);
    let unit = OffloadUnit::new(
        key.unit_id(),
        [WeightBinding::new("weight", "unsupported", TensorSelection::Full, 2).unwrap()],
    )
    .unwrap();
    let entry = ParameterBankEntry::new(key, unit, 2).unwrap();
    assert!(matches!(
        preflight_selected_entry_bindings(&store, &[entry]),
        Err(AddressableParameterBankError::Transformation { .. })
    ));
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, 0);
}

#[test]
fn selected_entry_byte_corruption_fails_before_checkpoint_work() {
    let (_dir, store) = fixture();
    let entry = entries().into_iter().next().unwrap();
    let identity = entry.identity;
    let selected = SelectedAddressableEntries {
        entries: vec![entry],
        transformations: BTreeMap::new(),
        expected_bytes: BTreeMap::from([(identity, 15)]),
        placements: BTreeMap::from([(
            identity,
            eredu_runtime::AddressableBankMemberPlacement::new(
                eredu_runtime::ExecutionGroupId::new("decoder").unwrap(),
                identity.unit(),
                "decoder.unit",
                eredu_runtime::AddressableBankDistribution::Replicated,
            )
            .unwrap(),
        )]),
    };
    let before = store.source_diagnostics().unwrap().physical_reads;
    let error = AddressableParameterBank::new_selected_shared(
        store.clone(),
        selected,
        ParameterBankOptions::default(),
        stream(),
        stream(),
    )
    .err()
    .expect("corrupt selected bytes must fail");
    assert!(error.to_string().contains("bytes differ"));
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, before);
}

#[test]
fn selected_entry_placement_coverage_fails_before_checkpoint_work() {
    let (_dir, store) = fixture();
    let entry = entries().into_iter().next().unwrap();
    let identity = entry.identity;
    let selected = SelectedAddressableEntries {
        entries: vec![entry],
        transformations: BTreeMap::new(),
        expected_bytes: BTreeMap::from([(identity, 16)]),
        placements: BTreeMap::new(),
    };
    let before = store.source_diagnostics().unwrap().physical_reads;
    let error = AddressableParameterBank::new_selected_shared(
        store.clone(),
        selected,
        ParameterBankOptions::default(),
        stream(),
        stream(),
    )
    .err()
    .expect("incomplete selected placement must fail");
    assert!(error.to_string().contains("placements do not cover"));
    assert_eq!(store.source_diagnostics().unwrap().physical_reads, before);
}

#[test]
fn residency_report_retains_exact_selected_entry_placement() {
    let (_dir, store) = fixture();
    let entry = entries().into_iter().next().unwrap();
    let identity = entry.identity;
    let placement = eredu_runtime::AddressableBankMemberPlacement::new(
        eredu_runtime::ExecutionGroupId::new("decoder").unwrap(),
        identity.unit(),
        "decoder.unit",
        eredu_runtime::AddressableBankDistribution::Replicated,
    )
    .unwrap();
    let selected = SelectedAddressableEntries {
        entries: vec![entry],
        transformations: BTreeMap::new(),
        expected_bytes: BTreeMap::from([(identity, 16)]),
        placements: BTreeMap::from([(identity, placement.clone())]),
    };
    let bank = AddressableParameterBank::new_selected_shared(
        store,
        selected,
        ParameterBankOptions::default(),
        stream(),
        stream(),
    )
    .unwrap();
    assert_eq!(
        bank.report().unwrap().placements(),
        &[(identity, placement)]
    );
}

#[test]
fn affine_selected_bytes_use_the_exact_non_f32_companion_dtype() {
    let affine =
        WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(64, 4).unwrap());
    assert_eq!(
        packed_projection_bytes(
            &[2, 64],
            affine,
            &eredu_checkpoint::recipe::RecipeDtype::F16,
        )
        .unwrap(),
        72
    );
    assert_eq!(
        packed_projection_bytes(
            &[2, 64],
            affine,
            &eredu_checkpoint::recipe::RecipeDtype::F32,
        )
        .unwrap(),
        80
    );
}

#[test]
fn mixed_selected_transforms_remain_explicit_in_telemetry() {
    let affine =
        WeightQuantization::Affine(eredu_checkpoint::AffineQuantization::new(64, 4).unwrap());
    let transformations = BTreeMap::from([
        (
            (ParameterBankKey::new(0, 0, 0), "weight".into()),
            SelectedBindingTransform {
                quantization: affine,
                companion_dtype: eredu_checkpoint::recipe::RecipeDtype::F16,
            },
        ),
        (
            (ParameterBankKey::new(0, 0, 1), "weight".into()),
            SelectedBindingTransform {
                quantization: WeightQuantization::MxFp4,
                companion_dtype: eredu_checkpoint::recipe::RecipeDtype::F16,
            },
        ),
    ]);
    assert_eq!(
        selected_transformation_formats(&transformations),
        [affine, WeightQuantization::MxFp4]
    );
}

fn cache(
    store: Arc<SafetensorsWeightStore>,
    device: u64,
    host: u64,
    scratch: u64,
    eviction: CacheEvictionPolicy,
) -> AddressableParameterBank {
    cache_with_target(store, device, host, scratch, scratch, eviction)
}

fn cache_with_target(
    store: Arc<SafetensorsWeightStore>,
    device: u64,
    host: u64,
    scratch: u64,
    bulk_target: u64,
    eviction: CacheEvictionPolicy,
) -> AddressableParameterBank {
    let binding_capacity =
        host_transfer_capacity_upper_bound(8, HostTransferPolicy::Transfer).unwrap() as u64;
    let physical_host = (host / 8).checked_mul(binding_capacity).unwrap();
    let storage = OffloadConfig::new(Some(device), Some(physical_host), 1)
        .unwrap()
        .with_eviction_policy(eviction);
    AddressableParameterBank::new(
        store,
        entries(),
        ParameterBankOptions::new(storage, scratch, bulk_target).unwrap(),
        stream(),
        stream(),
    )
    .unwrap()
}

#[test]
fn quantized_cache_materializes_only_rank_local_entry_and_tp_recipes() {
    let dir = tempfile::tempdir().unwrap();
    let gate = (0..2 * 4 * 64)
        .map(|index| (index as f32 - 127.0) / 32.0)
        .collect::<Vec<_>>();
    let down = gate.iter().map(|value| value * 0.5).collect::<Vec<_>>();
    let gate_bytes = gate
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let down_bytes = down
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "entries.gate",
                TensorView::new(StoredDtype::F32, vec![2, 4, 64], &gate_bytes).unwrap(),
            ),
            (
                "entries.down",
                TensorView::new(StoredDtype::F32, vec![2, 4, 64], &down_bytes).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store: Arc<dyn eredu_checkpoint::store::CheckpointSource> =
        Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let entries = (0..2)
        .map(|entry| {
            let identity = ParameterBankKey::new(0, 3, entry);
            let bindings = ["gate", "down"].map(|projection| {
                let owned = DerivedWeightRecipe::source(
                    format!("entries.{projection}"),
                    TensorSelection::Range {
                        axis: 0,
                        start: entry,
                        end: entry + 1,
                    },
                );
                let tp = DerivedWeightRecipe::Select {
                    input: Box::new(owned),
                    selection: TensorSelection::Range {
                        axis: 1,
                        start: entry * 2,
                        end: entry * 2 + 2,
                    },
                };
                WeightBinding::from_recipe(format!("{projection}_proj"), tp, 512)
                    .unwrap()
                    .with_quantization_companions(
                        format!("{projection}_proj_scales"),
                        Some(format!("{projection}_proj_biases")),
                    )
                    .unwrap()
            });
            let unit = OffloadUnit::new(identity.unit_id(), bindings).unwrap();
            ParameterBankEntry::new(identity, unit, 1_024).unwrap()
        })
        .collect::<Vec<_>>();
    let options =
        ParameterBankOptions::new(OffloadConfig::new(None, None, 1).unwrap(), 1_024, 1_024)
            .unwrap();
    let source_stream = stream();
    let quantization = WeightQuantization::Affine(Default::default());
    let transformed = quantize_entry_catalog(
        store,
        entries,
        quantization,
        options.compact_bank_scratch_bytes,
        &source_stream,
    )
    .unwrap();
    let cache = AddressableParameterBank::new_shared_with_policy(
        transformed.store,
        transformed.entries,
        options,
        ResidencyPolicy::Cacheable,
        MemoryTier::Disk,
        source_stream,
        stream(),
        vec![quantization],
        BTreeMap::new(),
        Some(transformed.report),
    )
    .unwrap();

    assert_eq!(
        cache.weight_quantizations(),
        [WeightQuantization::Affine(Default::default())]
    );
    let report = cache.report().unwrap();
    assert_eq!(report.owned_entries, 2);
    assert_eq!(report.owned_bytes, 320);
    assert_eq!(
        report.materialization,
        Some(WeightMaterializationReport {
            admitted_working_set_bytes: 320,
            transformed_weights: 4,
            source_tiles: 8,
            peak_in_flight_tiles: 1,
            source_bytes_read: 2_048,
            output_bytes: 320,
            peak_planned_working_set_bytes: 296,
            largest_source_tile_bytes: 256,
            largest_output_tile_bytes: 40,
        })
    );
    let acquired = cache
        .acquire_selection_slice(3, &[1], &[1, 1], BankAccessClass::Incremental, &stream())
        .unwrap();
    assert_eq!(
        acquired
            .compact_binding("gate_proj", &stream())
            .unwrap()
            .shape(),
        &[1, 2, 8]
    );
    assert_eq!(
        acquired
            .compact_binding("gate_proj_scales", &stream())
            .unwrap()
            .shape(),
        &[1, 2, 1]
    );
}

#[test]
fn entry_quantization_uses_only_declared_roles_and_companion_names() {
    let dir = tempfile::tempdir().unwrap();
    let values = (0..8 * 64)
        .map(|index| index as f32 / 16.0)
        .collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    serialize_to_file(
        [
            (
                "legitimate.bias_scale_blocks",
                TensorView::new(StoredDtype::F32, vec![8, 64], &bytes).unwrap(),
            ),
            (
                "preserved.matrix",
                TensorView::new(StoredDtype::F32, vec![8, 64], &bytes).unwrap(),
            ),
        ],
        None,
        &dir.path().join("model.safetensors"),
    )
    .unwrap();
    let store: Arc<dyn CheckpointSource> =
        Arc::new(SafetensorsWeightStore::open(dir.path()).unwrap());
    let identity = ParameterBankKey::new(0, 0, 0);
    let quantized = WeightBinding::from_recipe(
        "legitimate.bias_scale_blocks",
        DerivedWeightRecipe::source("legitimate.bias_scale_blocks", TensorSelection::Full),
        bytes.len() as u64,
    )
    .unwrap()
    .with_quantization_companions("declared.scale", Some("declared.bias".into()))
    .unwrap();
    let preserved = WeightBinding::from_recipe(
        "ordinary_matrix",
        DerivedWeightRecipe::source("preserved.matrix", TensorSelection::Full),
        bytes.len() as u64,
    )
    .unwrap();
    let entry = ParameterBankEntry::new(
        identity,
        OffloadUnit::new(identity.unit_id(), [quantized, preserved]).unwrap(),
        (bytes.len() * 2) as u64,
    )
    .unwrap();

    let transformed = quantize_entry_catalog(
        store,
        vec![entry],
        WeightQuantization::Affine(Default::default()),
        1_024,
        &stream(),
    )
    .unwrap();

    assert_eq!(transformed.report.transformed_weights, 1);
    assert_eq!(
        transformed.entries[0]
            .unit
            .bindings()
            .iter()
            .map(WeightBinding::name)
            .collect::<Vec<_>>(),
        [
            "declared.bias",
            "declared.scale",
            "legitimate.bias_scale_blocks",
            "ordinary_matrix",
        ]
    );
}

#[test]
fn coalesces_selections_in_global_order_and_separates_pass_counters() {
    let (_dir, store) = fixture();
    let cache = cache(store, 32, 32, 32, CacheEvictionPolicy::LeastRecentlyUsed);
    let first = cache
        .acquire_selection_slice(2, &[2, 0, 2, 0], &[2, 2], BankAccessClass::Bulk, &stream())
        .unwrap();
    assert_eq!(
        first.identities(),
        &[
            ParameterBankKey::new(0, 2, 0),
            ParameterBankKey::new(0, 2, 2)
        ]
    );
    assert_eq!(first.demand(), &[2, 2]);
    drop(first);

    let host = cache
        .manager
        .acquire(&ParameterBankKey::new(0, 2, 0).unit_id(), MemoryTier::Host)
        .unwrap();
    assert!(matches!(
        host.host_value("weight").unwrap().storage_kind().unwrap(),
        HostTransferStorageKind::Cpu
            | HostTransferStorageKind::MetalShared
            | HostTransferStorageKind::CudaPinned
    ));
    assert!(matches!(
        host.device_value("weight"),
        Err(ResidencyError::HostBindingIsNotArray { .. })
    ));
    drop(host);

    let second = cache
        .acquire_selection_slice(2, &[0, 2], &[1, 2], BankAccessClass::Incremental, &stream())
        .unwrap();
    drop(second);
    let report = cache.report().unwrap();
    assert_eq!(report.bulk.requested_selections, 4);
    assert_eq!(report.bulk.distinct_entries, 2);
    assert_eq!(report.bulk.coalesced_duplicates, 2);
    assert_eq!(report.bulk.device.misses, 2);
    assert_eq!(report.incremental.requested_selections, 2);
    assert_eq!(report.incremental.device.hits, 2);
    assert_eq!(report.owned_entries, 3);
    assert_eq!(report.owned_bytes, 48);
}

#[test]
fn resident_store_pins_every_entry_and_never_rereads_checkpoint_weights() {
    let (_dir, store) = fixture();
    let cache =
        AddressableParameterBank::new_resident_shared(store.clone(), entries(), stream(), stream())
            .unwrap();
    let initialized = cache.report().unwrap();
    assert_eq!(initialized.owned_entries, 3);
    assert_eq!(initialized.device_resident_entries, 3);
    assert_eq!(initialized.device_resident_bytes, initialized.owned_bytes);
    assert_eq!(initialized.host_resident_entries, 0);

    let reads_after_load = store.source_diagnostics().unwrap().physical_reads;
    let acquired = cache
        .acquire_selection_slice(
            2,
            &[2, 0, 2],
            &[3, 1],
            BankAccessClass::Incremental,
            &stream(),
        )
        .unwrap();
    let compact = acquired.compact_binding("weight", &stream()).unwrap();
    eval([&compact]).unwrap();

    assert_eq!(
        store.source_diagnostics().unwrap().physical_reads,
        reads_after_load
    );
    let executed = cache.report().unwrap();
    assert_eq!(executed.incremental.device.hits, 2);
    assert_eq!(executed.incremental.device.misses, 0);
    assert_eq!(executed.incremental.device.evictions, 0);
}

#[test]
fn bulk_target_is_required_and_cannot_exceed_scratch() {
    let storage = OffloadConfig::new(Some(48), Some(0), 1).unwrap();
    assert!(matches!(
        ParameterBankOptions::new(storage, 64, 0),
        Err(ParameterBankOptionsError::ZeroBulkBankTarget)
    ));
    assert!(matches!(
        ParameterBankOptions::new(storage, 64, 65),
        Err(ParameterBankOptionsError::BulkBankTargetExceedsScratch { .. })
    ));
}

#[test]
fn bounded_execution_chunks_bulk_but_not_incremental_and_preserves_row_order() {
    let (_dir, store) = fixture();
    let cache = cache_with_target(store, 48, 0, 48, 32, CacheEvictionPolicy::LeastRecentlyUsed);
    let execution = stream();
    let hidden = Array::from_slice(&[1f32, 2., 3., 4., 5., 6.], &[3, 2]);
    let selections = Array::from_slice(&[0i32, 1, 1, 2, 2, 0], &[3, 2]);
    let weights = Array::from_slice(&[0.5f32; 6], &[3, 2]);
    let mut bulk_banks = 0;
    let output = crate::tests::support::grouped_provider::execute_selections_bounded(
        &cache,
        ParameterBankSelection::new(2, &hidden, &selections, &weights, BankAccessClass::Bulk),
        &execution,
        |hidden, _acquired, _compact, _weights, _stream| {
            bulk_banks += 1;
            Ok(hidden.clone())
        },
    )
    .unwrap();
    assert_eq!(bulk_banks, 3);
    assert_eq!(
        output.evaluated().unwrap().as_slice::<f32>(),
        hidden.evaluated().unwrap().as_slice::<f32>()
    );

    let mut incremental_banks = 0;
    crate::tests::support::grouped_provider::execute_selections_bounded(
        &cache,
        ParameterBankSelection::new(
            2,
            &hidden,
            &selections,
            &weights,
            BankAccessClass::Incremental,
        ),
        &execution,
        |hidden, _acquired, _compact, _weights, _stream| {
            incremental_banks += 1;
            Ok(hidden.clone())
        },
    )
    .unwrap();
    assert_eq!(incremental_banks, 1);

    let distributed_selections = Array::from_slice(&[0i32, 1, 2], &[3]);
    let distributed_weights = Array::from_slice(&[1f32; 3], &[3]);
    let mut distributed_banks = 0;
    crate::tests::support::grouped_provider::execute_selections_bounded(
        &cache,
        ParameterBankSelection::new(
            2,
            &hidden,
            &distributed_selections,
            &distributed_weights,
            BankAccessClass::Bulk,
        ),
        &execution,
        |hidden, _acquired, _compact, _weights, _stream| {
            distributed_banks += 1;
            Ok(hidden.clone())
        },
    )
    .unwrap();
    assert_eq!(distributed_banks, 2);
}

#[test]
fn indexed_movement_validates_before_loading_and_bank_coalesces_demand() {
    let (_dir, store) = fixture();
    let cache = cache(store, 32, 32, 32, CacheEvictionPolicy::LeastRecentlyUsed);
    let execution = stream();
    let acquired = cache
        .acquire_selection_slice(2, &[2, 0, 2, 0], &[2, 2], BankAccessClass::Bulk, &execution)
        .unwrap();
    assert_eq!(
        acquired.identities(),
        &[
            ParameterBankKey::new(0, 2, 0),
            ParameterBankKey::new(0, 2, 2)
        ]
    );
    assert_eq!(acquired.demand(), &[2, 2]);
    drop(acquired);

    let mut movement = MlxIndexedMovement;
    let invalid = MlxTensor::from_array(Array::from_slice(&[-1i32, 0], &[2]));
    assert!(matches!(
        movement.index_demands(&invalid, 3, &execution),
        Err(Error::AddressableParameterBank(
            AddressableParameterBankError::InvalidSelectionSet {
                invalid_count: 1,
                ..
            }
        ))
    ));
    let report = cache.report().unwrap();
    assert_eq!(report.incremental.requested_selections, 0);

    let narrowing_alias = MlxTensor::from_array(Array::from_slice(&[1u64 << 32], &[1]));
    assert!(matches!(
        movement.index_demands(&narrowing_alias, 3, &execution),
        Err(Error::AddressableParameterBank(
            AddressableParameterBankError::InvalidSelectionSet {
                invalid_count: 1,
                ..
            }
        ))
    ));
}

#[test]
fn rejects_invalid_missing_and_over_scratch_selections_before_loading() {
    let (_dir, store) = fixture();
    let cache = cache(store, 48, 0, 16, CacheEvictionPolicy::LeastRecentlyUsed);
    assert!(matches!(
        cache.acquire_selection_slice(2, &[-1], &[1], BankAccessClass::Incremental, &stream()),
        Err(AddressableParameterBankError::InvalidEntryId { .. })
    ));
    assert!(matches!(
        cache.acquire_selection_slice(2, &[3], &[1], BankAccessClass::Incremental, &stream()),
        Err(AddressableParameterBankError::MissingOwnedEntry { .. })
    ));
    assert!(matches!(
        cache.acquire_selection_slice(2, &[0, 1], &[2], BankAccessClass::Bulk, &stream()),
        Err(AddressableParameterBankError::ScratchLimitExceeded { .. })
    ));
    let report = cache.report().unwrap();
    assert_eq!(report.device_resident_entries, 0);
    assert_eq!(report.bulk.requested_selections, 0);
    assert_eq!(report.incremental.requested_selections, 0);
}

#[test]
fn empty_selections_do_not_materialize_or_build_a_bank() {
    let (_dir, store) = fixture();
    let cache = cache(store, 16, 16, 16, CacheEvictionPolicy::LeastRecentlyUsed);
    let acquired = cache
        .acquire_selection_slice(2, &[], &[0, 2], BankAccessClass::Incremental, &stream())
        .unwrap();
    assert!(acquired.is_empty());
    assert_eq!(acquired.scratch_bytes(), 0);
    drop(acquired);
    let report = cache.report().unwrap();
    assert_eq!(report.host_resident_entries, 0);
    assert_eq!(report.device_resident_entries, 0);
    assert_eq!(report.incremental.compact_banks, 0);
}

#[test]
fn lfu_uses_duplicate_selection_demand_and_deterministic_recency_ties() {
    let (_dir, store) = fixture();
    let cache = cache(store, 32, 0, 32, CacheEvictionPolicy::LeastFrequentlyUsed);
    for selection in [&[0, 0, 0][..], &[1][..], &[2][..]] {
        let mut acquired = cache
            .acquire_selection_slice(
                2,
                selection,
                &[selection.len() as i32],
                BankAccessClass::Incremental,
                &stream(),
            )
            .unwrap();
        acquired.transfer.synchronize().unwrap();
        drop(acquired);
        crate::backend::ordinary_retirement::reclaim_all();
    }
    let report = cache.report().unwrap();
    let resident = report
        .residency
        .units()
        .iter()
        .filter(|unit| unit.device_resident())
        .map(|unit| unit.id().as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        resident,
        vec![
            ParameterBankKey::new(0, 2, 0).unit_id().as_str(),
            ParameterBankKey::new(0, 2, 2).unit_id().as_str()
        ]
    );
    assert_eq!(report.incremental.device.evictions, 1);
}
