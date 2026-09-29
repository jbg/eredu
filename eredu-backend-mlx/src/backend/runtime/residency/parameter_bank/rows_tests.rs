use super::*;
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    rows::PreparedRowSource,
    store::{MemoryWeightStore, SharedCheckpointSource},
};
use eredu_core::residency::OffloadUnitRange;
use eredu_nn::{ParameterId, TensorElementType};
use eredu_runtime::{
    BoundedRowLookup, ParameterBank, PreparedRowLookup, PreparedRowScale, RowEncoding,
    RowLookupLimits, RowLookupProvider, RowLookupSpec, RowResidencyRange,
};
use safetensors::Dtype as StoredDtype;
mod qwen4_gguf;

fn row_bank(
    dtype: StoredDtype,
    width: usize,
    bytes: Vec<u8>,
    spec: RowLookupSpec,
    scale: Option<(ParameterId, MlxTensor)>,
    stream: &Stream,
) -> (ResidencyManager, Result<MlxRowBank, Error>) {
    let row_bytes = bytes.len() / 8;
    let source: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([
            (
                "first".into(),
                dtype,
                vec![4, width],
                bytes[..4 * row_bytes].to_vec(),
            ),
            (
                "second".into(),
                dtype,
                vec![4, width],
                bytes[4 * row_bytes..].to_vec(),
            ),
        ])
        .unwrap(),
    );
    let table = PreparedRowSource::new(
        source.clone(),
        DerivedWeightRecipe::Concatenate {
            axis: 0,
            inputs: vec![
                DerivedWeightRecipe::source("first", TensorSelection::Full),
                DerivedWeightRecipe::source("second", TensorSelection::Full),
            ],
        },
    )
    .unwrap();
    let range = OffloadUnitRange::new(
        OffloadUnitId::new("test.rows").unwrap(),
        100,
        108,
        row_bytes as u64,
        ResidencyPolicy::Cacheable,
    )
    .unwrap();
    let rows = RowResidencyRange::new(range.clone(), table, "value").unwrap();
    let plan = OffloadPlan::with_ranges(
        OffloadConfig::new(Some(row_bytes as u64 * 2), Some(131072), 1).unwrap(),
        [],
        [range.clone()],
    )
    .unwrap();
    let row_binding = rows.clone();
    let manager = ResidencyManager::new_shared_row_ranges(
        source,
        BTreeMap::new(),
        plan,
        [],
        vec![rows],
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    manager.initialize().unwrap();
    let scalar = match &spec.encoding {
        RowEncoding::ScalarE4M3 { scale } => Some(PreparedRowScale {
            parameter: scale.clone(),
            source: Arc::new(
                MemoryWeightStore::from_safetensors([(
                    "retained.scalar".into(),
                    StoredDtype::F32,
                    vec![1],
                    1f32.to_le_bytes().to_vec(),
                )])
                .unwrap(),
            ),
            recipe: DerivedWeightRecipe::source("retained.scalar", TensorSelection::Full),
        }),
        _ => None,
    };
    let bank = PreparedRowLookup::new(row_binding, spec, scalar, limits())
        .map_err(Error::from)
        .and_then(|prepared| {
            check_row_memory(&prepared);
            MlxRowBank::new(manager.clone(), &prepared, scale, 65536)
        });
    (manager, bank)
}
fn declaration(encoding: RowEncoding, width: i32, output_type: TensorElementType) -> RowLookupSpec {
    RowLookupSpec {
        parameter: ParameterId::new("test.table").unwrap(),
        bank: 2,
        unit: 7,
        rows: 8,
        dimensions: width,
        encoding,
        output_type,
    }
}
fn limits() -> RowLookupLimits {
    RowLookupLimits {
        requests: 16,
        rows_per_acquisition: 2,
        acquisition_bytes: 4096,
        host_bytes: 2048,
        output_bytes: 65536,
    }
}
fn stream() -> Stream {
    Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0))
}

#[test]
fn row_lookup_dense_bf16_and_scalar_fp8_share_bounded_residency() {
    let stream = stream();
    let scale_id = ParameterId::new("shared.scalar").unwrap();
    for fp8 in [false, true] {
        let bytes = if fp8 {
            (0..16)
                .map(|i| {
                    if i % 2 == 0 {
                        0x38 + i / 2
                    } else {
                        0xb8 + i / 2
                    }
                })
                .collect()
        } else {
            (0..16)
                .flat_map(|i| half::bf16::from_f32(i as f32 * 0.25 - 1.).to_le_bytes())
                .collect()
        };
        let encoding = if fp8 {
            RowEncoding::ScalarE4M3 {
                scale: scale_id.clone(),
            }
        } else {
            RowEncoding::Dense
        };
        let spec = declaration(encoding, 2, TensorElementType::Bf16);
        let scale = fp8.then(|| {
            (
                scale_id.clone(),
                MlxTensor::from_array(Array::from_slice(&[0.375f32], &[1])),
            )
        });
        let (manager, bank) = row_bank(
            if fp8 {
                StoredDtype::F8_E4M3
            } else {
                StoredDtype::BF16
            },
            2,
            bytes,
            spec.clone(),
            scale,
            &stream,
        );
        let mut provider = BoundedRowLookup::new(bank.unwrap(), spec.clone(), limits()).unwrap();
        let ids = [7u64, 4, 3, 4, 0, 5];
        for _ in 0..4 {
            let result = provider
                .lookup_rows(&spec, &ids, ParameterBankAccess::Bulk, &stream)
                .unwrap();
            let array = result.as_array();
            eval([array]).unwrap();
            assert_eq!(array.dtype(), Dtype::Bfloat16);
            let actual = array
                .evaluated()
                .unwrap()
                .as_slice::<half::bf16>()
                .iter()
                .map(|v| v.to_f32())
                .collect::<Vec<_>>();
            let expected = ids
                .iter()
                .flat_map(|row| {
                    if fp8 {
                        let x = (1. + *row as f32 / 8.) * 0.375;
                        [
                            half::bf16::from_f32(x).to_f32(),
                            half::bf16::from_f32(-x).to_f32(),
                        ]
                    } else {
                        [*row as f32 * 0.5 - 1., *row as f32 * 0.5 - 0.75]
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
            let report = provider.bank().report().unwrap();
            assert!(report.units().len() <= 2);
            assert!(
                report.offload().resident_bytes().get(MemoryTier::Device)
                    <= if fp8 { 4 } else { 8 }
            );
        }
        // A returned tensor must no longer pin its encoded source rows.
        let result = provider
            .lookup_rows(&spec, &[0, 1], ParameterBankAccess::Incremental, &stream)
            .unwrap();
        for row in [100, 101] {
            assert!(manager
                .evict(
                    &OffloadUnitId::new(format!("test.rows.{row}")).unwrap(),
                    MemoryTier::Device
                )
                .unwrap());
        }
        eval([result.as_array()]).unwrap();
    }
}

#[test]
fn row_lookup_gguf_q8_decodes_only_selected_blocks() {
    let stream = stream();
    let width = 32;
    let mut bytes = Vec::new();
    for row in 0..8 {
        bytes.extend(half::f16::from_f32(0.25).to_le_bytes());
        bytes.extend((0..32).map(|i| (i - 16 + row) as i8 as u8));
    }
    let spec = declaration(
        RowEncoding::Gguf {
            encoding: eredu_gguf::GgmlType::Q8_0,
            endian: eredu_gguf::Endian::Little,
        },
        width,
        TensorElementType::F32,
    );
    let (_, bank) = row_bank(StoredDtype::U8, 34, bytes, spec.clone(), None, &stream);
    let mut provider = BoundedRowLookup::new(bank.unwrap(), spec.clone(), limits()).unwrap();
    let ids = [7u64, 4, 3, 4, 0];
    let result = provider
        .lookup_rows(&spec, &ids, ParameterBankAccess::Bulk, &stream)
        .unwrap();
    eval([result.as_array()]).unwrap();
    let expected = ids
        .iter()
        .flat_map(|row| (0..32).map(move |i| (i - 16 + *row as i32) as f32 * 0.25))
        .collect::<Vec<_>>();
    assert_eq!(
        result.as_array().evaluated().unwrap().as_slice::<f32>(),
        expected
    );
    assert!(
        provider
            .bank()
            .report()
            .unwrap()
            .offload()
            .peak_resident_bytes()
            .get(MemoryTier::Device)
            <= 68
    );
}

#[test]
fn row_lookup_rejects_missing_or_mismatched_scalar_before_reading_rows() {
    let stream = stream();
    let spec = declaration(
        RowEncoding::ScalarE4M3 {
            scale: ParameterId::new("exact.scale").unwrap(),
        },
        2,
        TensorElementType::F32,
    );
    for scale in [
        None,
        Some((
            ParameterId::new("wrong").unwrap(),
            MlxTensor::from_array(Array::from_slice(&[1f32], &[1])),
        )),
        Some((
            ParameterId::new("exact.scale").unwrap(),
            MlxTensor::from_array(Array::from_slice(&[1f32, 2.], &[2])),
        )),
        Some((
            ParameterId::new("exact.scale").unwrap(),
            MlxTensor::from_array(Array::from_slice(&[half::f16::from_f32(1.)], &[1])),
        )),
    ] {
        let (manager, bank) = row_bank(
            StoredDtype::F8_E4M3,
            2,
            vec![0x38; 16],
            spec.clone(),
            scale,
            &stream,
        );
        assert!(bank.is_err());
        assert!(manager.report().unwrap().units().is_empty());
    }
}

#[test]
fn row_lookup_disk_sources_coalesce_reads_and_warm_cache_has_no_io() {
    use eredu_checkpoint::{
        gguf_store::GgufWeightStore,
        schema::{
            CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
            TensorOperation,
        },
        store::SafetensorsWeightStore,
    };
    let stream = stream();
    for gguf in [false, true] {
        let directory = tempfile::tempdir().unwrap();
        let mut raw = Vec::new();
        for row in 0..8 {
            raw.extend(half::f16::from_f32(0.25).to_le_bytes());
            raw.extend((0..32).map(|i| (i - 16 + row) as i8 as u8));
        }
        let source: SharedCheckpointSource = if gguf {
            let path = directory.path().join("rows.gguf");
            eredu_gguf::Writer::default()
                .write(
                    std::fs::File::create(&path).unwrap(),
                    &BTreeMap::new(),
                    &[eredu_gguf::TensorInput {
                        name: "table.weight",
                        dimensions: &[32, 8],
                        ggml_type: eredu_gguf::GgmlType::Q8_0,
                        data: &raw,
                    }],
                )
                .unwrap();
            let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
            let plan = GgufCheckpointPlan::new(
                "row fixture",
                vec![GgufTensorConstraint::required(
                    "table.weight",
                    vec![8, 32],
                    GgufTypeConstraint::OperationClass(TensorOperation::Matrix),
                )],
                vec![],
                CatalogPolicy::strict(),
            )
            .unwrap();
            let mapping = checkpoint.translated_outputs(str::to_owned).unwrap();
            Arc::new(
                GgufWeightStore::builder()
                    .add_checkpoint(checkpoint, &plan, &mapping)
                    .unwrap()
                    .build()
                    .unwrap(),
            )
        } else {
            safetensors::tensor::serialize_to_file(
                [(
                    "table.weight",
                    safetensors::tensor::TensorView::new(StoredDtype::U8, vec![8, 34], &raw)
                        .unwrap(),
                )],
                None,
                &directory.path().join("model.safetensors"),
            )
            .unwrap();
            Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap())
        };
        let table = PreparedRowSource::new(
            source.clone(),
            DerivedWeightRecipe::source("table.weight", TensorSelection::Full),
        )
        .unwrap();
        let range = OffloadUnitRange::new(
            OffloadUnitId::new("disk.rows").unwrap(),
            0,
            8,
            34,
            ResidencyPolicy::Cacheable,
        )
        .unwrap();
        let rows = RowResidencyRange::new(range.clone(), table, "value").unwrap();
        let plan = OffloadPlan::with_ranges(
            OffloadConfig::new(Some(68), Some(131072), 1).unwrap(),
            [],
            [range.clone()],
        )
        .unwrap();
        let row_binding = rows.clone();
        let manager = ResidencyManager::new_shared_row_ranges(
            source.clone(),
            BTreeMap::new(),
            plan,
            [],
            vec![rows],
            stream.clone(),
            stream.clone(),
        )
        .unwrap();
        manager.initialize().unwrap();
        let spec = declaration(
            RowEncoding::Gguf {
                encoding: eredu_gguf::GgmlType::Q8_0,
                endian: eredu_gguf::Endian::Little,
            },
            32,
            TensorElementType::F32,
        );
        let prepared = PreparedRowLookup::new(row_binding, spec.clone(), None, limits()).unwrap();
        let bank = MlxRowBank::new(manager, &prepared, None, 65536).unwrap();
        let mut provider = prepared.bind(bank).unwrap();
        assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 0);
        let cold = std::time::Instant::now();
        let output = provider
            .lookup_rows(&spec, &[3, 2, 3], ParameterBankAccess::Bulk, &stream)
            .unwrap();
        eval([output.as_array()]).unwrap();
        let elapsed = cold.elapsed();
        let before = source.source_diagnostics().unwrap();
        assert_eq!(
            before.physical_read_bytes, 68,
            "only two unique encoded rows"
        );
        assert_eq!(
            before.physical_reads, 1,
            "adjacent row units share one physical read"
        );
        let warm = std::time::Instant::now();
        for _ in 0..32 {
            let result = provider
                .lookup_rows(&spec, &[2, 3, 2], ParameterBankAccess::Incremental, &stream)
                .unwrap();
            eval([result.as_array()]).unwrap();
            let expected = [2, 3, 2]
                .iter()
                .flat_map(|row| (0..32).map(move |i| (i - 16 + row) as f32 * 0.25))
                .collect::<Vec<_>>();
            assert_eq!(
                result.as_array().evaluated().unwrap().as_slice::<f32>(),
                expected
            );
        }
        assert_eq!(
            source.source_diagnostics().unwrap().physical_read_bytes,
            before.physical_read_bytes
        );
        eprintln!("row lookup {:?}: cold {:?}, warm mean {:?}, read {} bytes in {} operation, cache <=68 bytes",if gguf {"GGUF"} else {"SafeTensors"},elapsed,warm.elapsed()/32,before.physical_read_bytes,before.physical_reads);
    }
}

#[test]
fn row_lookup_prepared_set_shares_grouped_budget_and_loads_exact_scalars() {
    for budget in [16, 32] {
        for parallel in [false, true] {
            verify_prepared_row_pool(budget, parallel);
        }
    }
}
fn verify_prepared_row_pool(budget: u64, parallel: bool) {
    use eredu_checkpoint::store::RestrictedCheckpointSource;
    use eredu_runtime::{PreparedRowLookups, RowLookupProvider};
    let stream = Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let values: Vec<(String, StoredDtype, Vec<usize>, Vec<u8>)> = vec![
        (
            "dense".into(),
            StoredDtype::F32,
            vec![4, 2],
            (1..=8).flat_map(|i| (i as f32).to_le_bytes()).collect(),
        ),
        (
            "encoded".into(),
            StoredDtype::F8_E4M3,
            vec![4, 2],
            vec![0x38, 0xb8, 0x40, 0xc0, 0x44, 0xc4, 0x48, 0xc8],
        ),
        (
            "scalar".into(),
            StoredDtype::F32,
            vec![1],
            2.5f32.to_le_bytes().to_vec(),
        ),
        (
            "group".into(),
            StoredDtype::F32,
            vec![1, 4],
            [0.5f32, -1., 2., 3.]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        ),
    ];
    let directory = tempfile::tempdir().unwrap();
    safetensors::tensor::serialize_to_file(
        values.iter().map(|(name, dtype, shape, bytes)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        }),
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source: SharedCheckpointSource =
        Arc::new(eredu_checkpoint::store::SafetensorsWeightStore::open(directory.path()).unwrap());
    let entries = [("dense", 1, 8), ("encoded", 2, 2)].map(|(name, bank, bytes)| {
        let scalar = (bank == 2).then(|| PreparedRowScale {
            parameter: ParameterId::new("authoritative.scalar").unwrap(),
            source: Arc::new(
                RestrictedCheckpointSource::including(
                    source.clone(),
                    "scalar owner",
                    std::collections::BTreeSet::from(["scalar".into()]),
                )
                .unwrap(),
            ),
            recipe: DerivedWeightRecipe::source("scalar", TensorSelection::Full),
        });
        let table = PreparedRowSource::new(
            source.clone(),
            DerivedWeightRecipe::source(name, TensorSelection::Full),
        )
        .unwrap();
        let range = RowResidencyRange::new(
            OffloadUnitRange::new(
                OffloadUnitId::new(format!("table.{bank}")).unwrap(),
                0,
                4,
                bytes,
                ResidencyPolicy::Cacheable,
            )
            .unwrap(),
            table,
            "rows",
        )
        .unwrap();
        PreparedRowLookup::new(
            range,
            RowLookupSpec {
                parameter: ParameterId::new(format!("logical.{name}")).unwrap(),
                bank,
                unit: bank,
                rows: 4,
                dimensions: 2,
                encoding: scalar
                    .as_ref()
                    .map_or(RowEncoding::Dense, |s| RowEncoding::ScalarE4M3 {
                        scale: s.parameter.clone(),
                    }),
                output_type: TensorElementType::F32,
            },
            scalar,
            limits(),
        )
        .unwrap()
    });
    let prepared = PreparedRowLookups::new(entries, 3).unwrap();
    assert_eq!(prepared.requirements().source_bytes, 40);
    assert_eq!(prepared.requirements().scalar_bytes, 4);
    let key = ParameterBankKey::new(0, 0, 0);
    let binding = WeightBinding::new("weight", "group", TensorSelection::Full, 16).unwrap();
    let selected = || SelectedAddressableEntries {
        parameter_targets: BTreeMap::from([((key, "weight".into()), "logical.group".into())]),
        entries: vec![ParameterBankEntry::new(
            key,
            OffloadUnit::new(key.unit_id(), [binding.clone()]).unwrap(),
            16,
        )
        .unwrap()],
        transformations: BTreeMap::new(),
        expected_bytes: BTreeMap::from([(key, 16)]),
        placements: BTreeMap::from([(
            key,
            eredu_runtime::AddressableBankMemberPlacement::new(
                eredu_runtime::ExecutionGroupId::new("decoder").unwrap(),
                0,
                "unit",
                eredu_runtime::AddressableBankDistribution::Replicated,
            )
            .unwrap(),
        )]),
    };
    let first = prepared.entries().values().next().unwrap();
    let mut conflicting_spec = first.spec().clone();
    conflicting_spec.bank = 0;
    let conflicting = PreparedRowLookups::new(
        [PreparedRowLookup::new(
            first.range().clone(),
            conflicting_spec,
            first.scale().cloned(),
            first.limits(),
        )
        .unwrap()],
        3,
    )
    .unwrap();
    assert!(AddressableParameterBank::new_selected_shared(
        source.clone(),
        selected(),
        &eredu_runtime::SelectedRowLookups::select(
            conflicting,
            eredu_runtime::ParameterBankLoadOptions::default(),
            0,
            &MlxRowLookupSupport
        )
        .unwrap(),
        stream.clone(),
        stream.clone()
    )
    .is_err());
    assert!(
        eredu_runtime::SelectedRowLookups::select(
            prepared.clone(),
            eredu_runtime::ParameterBankLoadOptions::new(
                OffloadConfig::new(Some(15), Some(131072), 1).unwrap(),
                131072,
                65536,
            )
            .unwrap(),
            4,
            &MlxRowLookupSupport,
        )
        .is_err(),
        "cold pool admission must hold a complete row acquisition"
    );
    // Resident grouped operators can still share the common mechanism with a
    // row-only independent pool; no fake grouped catalog entry is required.
    let empty = SelectedAddressableEntries {
        parameter_targets: BTreeMap::new(),
        entries: vec![],
        transformations: BTreeMap::new(),
        expected_bytes: BTreeMap::new(),
        placements: BTreeMap::new(),
    };
    let row_only = AddressableParameterBank::new_selected_shared(
        source.clone(),
        empty,
        &eredu_runtime::SelectedRowLookups::select(
            prepared.clone(),
            eredu_runtime::ParameterBankLoadOptions::default(),
            4,
            &MlxRowLookupSupport,
        )
        .unwrap(),
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    assert_eq!(row_only.report().unwrap().owned_entries(), 0);
    drop(row_only);
    let select = |scratch, scalars| {
        eredu_runtime::SelectedRowLookups::select(
            prepared.clone(),
            eredu_runtime::ParameterBankLoadOptions::new(
                OffloadConfig::new(Some(budget), Some(131072), 1).unwrap(),
                scratch,
                scratch,
            )
            .unwrap(),
            scalars,
            &MlxRowLookupSupport,
        )
    };
    assert!(select(131072, 3).is_err());
    assert!(select(1, 4).is_err());
    // The former separate 64 KiB allowances hid simultaneous planning/output
    // and native decode buffers. One shared 64 KiB workspace is insufficient.
    assert!(select(65536, 4).is_err());
    let selected_rows = select(131072, 4).unwrap();
    assert_eq!(
        selected_rows.requirements().invocation_bytes,
        2048 + 65536 + 88
    );
    assert_eq!(selected_rows.requirements().decode_bytes, 88);
    // Header-only selection cannot assume direct encoded initialization. Reserve
    // source/copy and final output for the ordinary scalar recipe path.
    assert_eq!(selected_rows.requirements().scalar_preparation_bytes, 12);
    let mut grouped = SharedAddressableParameterBank::new(
        AddressableParameterBank::new_selected_shared(
            source.clone(),
            selected(),
            &selected_rows,
            stream.clone(),
            stream.clone(),
        )
        .unwrap(),
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 0);
    let wrong_pool = eredu_runtime::SelectedRowLookups::select(
        prepared.clone(),
        eredu_runtime::ParameterBankLoadOptions::default(),
        4,
        &MlxRowLookupSupport,
    )
    .unwrap();
    assert!(matches!(
        MlxRowLookups::bind_shared(&grouped, &wrong_pool, &stream, &stream),
        Err(Error::RowLookupSelection(
            eredu_runtime::RowLookupSelectionError::PoolMismatch
        ))
    ));
    assert_eq!(
        source.source_diagnostics().unwrap().physical_read_bytes,
        0,
        "all cold checks precede scalar reads"
    );
    let mut rows = if parallel {
        let native =
            safemlx::distributed::init(false, safemlx::distributed::Backend::Ring).unwrap();
        let parallel = crate::backend::runtime::distributed::Group::uncontracted(&native);
        assert!(MlxRowLookups::bind_tensor_parallel(
            None,
            &selected_rows,
            parallel.clone(),
            parallel.clone(),
            0,
            0,
            &stream,
            &stream
        )
        .is_err());
        assert!(MlxRowLookups::bind_tensor_parallel(
            Some(&grouped),
            &selected_rows,
            parallel.clone(),
            parallel.clone(),
            1,
            0,
            &stream,
            &stream
        )
        .is_err());
        assert_eq!(
            source.source_diagnostics().unwrap().physical_read_bytes,
            0,
            "invalid ownership rejects before scalar reads"
        );
        MlxRowLookups::bind_tensor_parallel(
            Some(&grouped),
            &selected_rows,
            parallel.clone(),
            parallel,
            0,
            0,
            &stream,
            &stream,
        )
        .unwrap()
    } else {
        MlxRowLookups::bind_shared(&grouped, &selected_rows, &stream, &stream).unwrap()
    };
    assert_eq!(
        source.source_diagnostics().unwrap().physical_read_bytes,
        4,
        "only the retained scalar is loaded"
    );
    let specs: Vec<_> = prepared
        .entries()
        .values()
        .map(|v| v.spec().clone())
        .collect();
    let mut held = Vec::new();
    for _ in 0..4 {
        let demands = [(key, 1)];
        let acquired = grouped
            .acquire(
                ParameterBankAcquisition::new(&demands, ParameterBankAccess::Incremental),
                &stream,
            )
            .unwrap();
        let competing =
            rows.lookup_rows(&specs[0], &[0], ParameterBankAccess::Incremental, &stream);
        if budget == 16 {
            assert!(competing.is_err(), "grouped leases prevent row overcommit");
        } else {
            assert!(competing.is_ok(), "the shared budget may hold both owners");
        }
        let weight = acquired
            .compact_binding("weight", &stream)
            .unwrap()
            .into_evaluated()
            .unwrap();
        assert_eq!(weight.as_slice::<f32>(), [0.5, -1., 2., 3.]);
        grouped
            .complete(
                acquired,
                &MlxTensor::from_array(weight.into_array().unwrap()),
                &stream,
            )
            .unwrap();
        for spec in &specs {
            let value = rows
                .lookup_rows(spec, &[3, 0, 3], ParameterBankAccess::Bulk, &stream)
                .unwrap();
            let expected = if spec.bank == 1 {
                vec![7., 8., 1., 2., 7., 8.]
            } else {
                vec![10., -10., 2.5, -2.5, 10., -10.]
            };
            assert_eq!(
                value.as_array().evaluated().unwrap().as_slice::<f32>(),
                expected
            );
            held.push((value, expected));
        }
        let report = rows.report().unwrap().unwrap();
        assert!(report.offload().resident_bytes().get(MemoryTier::Device) <= budget);
        assert!(
            report
                .offload()
                .peak_resident_bytes()
                .get(MemoryTier::Device)
                <= budget
        );
    }
    let aggregate = ParameterBanksResidencyReport::new(BTreeMap::from([(
        eredu_runtime::RoutedBankId::new(0),
        grouped.report().unwrap(),
    )]))
    .with_rows(rows.pool_report().unwrap().unwrap());
    assert_eq!(
        aggregate.device_resident_bytes(),
        rows.report()
            .unwrap()
            .unwrap()
            .offload()
            .resident_bytes()
            .get(MemoryTier::Device)
    );
    assert_eq!(
        aggregate.peak_device_resident_bytes(),
        rows.report()
            .unwrap()
            .unwrap()
            .offload()
            .peak_resident_bytes()
            .get(MemoryTier::Device)
    );
    assert_eq!(
        aggregate.rows().unwrap().requirements(),
        selected_rows.requirements()
    );
    assert_eq!(grouped.report().unwrap().owned_entries(), 1);
    assert_eq!(grouped.report().unwrap().owned_bytes(), 16);
    for (value, expected) in held {
        assert_eq!(
            value.as_array().evaluated().unwrap().as_slice::<f32>(),
            expected,
            "eviction cannot invalidate completed lookup outputs"
        );
    }
}

fn check_row_memory(prepared: &PreparedRowLookup) {
    use eredu_nn::mechanism_memory::*;
    use eredu_runtime::RowLookupMechanismSupport;
    let memory = MlxRowLookupSupport
        .decode_memory(prepared.descriptor())
        .unwrap();
    memory.validate().unwrap();
    assert_eq!(
        memory.values,
        prepared.descriptor().decode_memory().unwrap().values
    );
    let output = memory.storage.iter().find(|s| s.name == "output").unwrap();
    assert_eq!(
        output.payload,
        MechanismBytes::exact(
            prepared.maximum_acquisition_rows()
                * prepared.spec().dimensions as u64
                * element_bytes(prepared.spec().output_type)
        )
    );
    assert_eq!(output.retention, StorageRetention::Returned);
    assert_eq!(output.backing, MechanismBacking::Invocation);
    assert_eq!(output.capacity.upper, None);
    assert!(memory
        .storage
        .iter()
        .filter(|s| s.name != "output")
        .all(|s| s.retention == StorageRetention::NativeCompletion));
    assert!(!memory.missing.is_empty());
    let workspace = MlxRowLookupSupport
        .workspace(prepared.descriptor())
        .unwrap()
        .unwrap();
    assert!(
        memory
            .storage
            .iter()
            .map(|s| s.payload.upper.unwrap())
            .sum::<u64>()
            <= workspace.decode_bytes
    );
}

#[test]
fn row_decode_single_dense_result_has_distinct_completed_backing() {
    use eredu_runtime::RowLookupBank;
    let stream = stream();
    let spec = declaration(RowEncoding::Dense, 2, TensorElementType::F32);
    let (manager, bank) = row_bank(
        StoredDtype::F32,
        2,
        (0..16)
            .flat_map(|n| ((n as f32 + 1.) / 8.).to_le_bytes())
            .collect(),
        spec.clone(),
        None,
        &stream,
    );
    let mut bank = bank.unwrap();
    let entries = [(ParameterBankKey::new(2, 7, 3), 1)];
    let acquired = bank
        .acquire(
            ParameterBankAcquisition::new(&entries, ParameterBankAccess::Incremental),
            &stream,
        )
        .unwrap();
    let source = acquired.leases()[0].device_value("value").unwrap().clone();
    eval([&source]).unwrap();
    let output = bank.rows(acquired, &spec, &stream).unwrap();
    assert!(
        output.as_array().is_available().unwrap(),
        "row decoder completes before returning"
    );
    let original = source.evaluated().unwrap();
    let completed = output.as_array().evaluated().unwrap();
    assert_ne!(
        original.as_slice::<f32>().as_ptr(),
        completed.as_slice::<f32>().as_ptr(),
        "one contiguous source row must not escape as an alias"
    );
    assert_eq!(completed.as_slice::<f32>(), &[0.875, 1.]);
    assert!(manager
        .evict(
            &OffloadUnitId::new("test.rows.103").unwrap(),
            MemoryTier::Device
        )
        .unwrap());
    assert_eq!(
        output.as_array().evaluated().unwrap().as_slice::<f32>(),
        &[0.875, 1.]
    );
}

#[test]
fn row_decode_half_scalar_memory_is_cold_and_uses_completion_lifetimes() {
    use eredu_core::{resources::*, Observed};
    use eredu_runtime::resource_lifetimes::ResourceLifetime;
    use eredu_runtime::RowLookupMechanismSupport;
    let source: SharedCheckpointSource = Arc::new(
        MemoryWeightStore::from_safetensors([
            (
                "rows".into(),
                StoredDtype::F8_E4M3,
                vec![2, 1],
                vec![0x38, 0x40],
            ),
            (
                "scale".into(),
                StoredDtype::F16,
                vec![1],
                half::f16::from_f32(0.5).to_le_bytes().to_vec(),
            ),
        ])
        .unwrap(),
    );
    let row_source = PreparedRowSource::new(
        source.clone(),
        DerivedWeightRecipe::source("rows", TensorSelection::Full),
    )
    .unwrap();
    let range = RowResidencyRange::new(
        OffloadUnitRange::new(
            OffloadUnitId::new("cold.rows").unwrap(),
            0,
            2,
            1,
            ResidencyPolicy::Cacheable,
        )
        .unwrap(),
        row_source,
        "value",
    )
    .unwrap();
    let scale = PreparedRowScale {
        parameter: ParameterId::new("scale").unwrap(),
        source: Arc::new(
            MemoryWeightStore::from_safetensors([(
                "scale".into(),
                StoredDtype::F16,
                vec![1],
                half::f16::from_f32(0.5).to_le_bytes().to_vec(),
            )])
            .unwrap(),
        ),
        recipe: DerivedWeightRecipe::source("scale", TensorSelection::Full),
    };
    let mut spec = declaration(
        RowEncoding::ScalarE4M3 {
            scale: scale.parameter.clone(),
        },
        1,
        TensorElementType::Bf16,
    );
    spec.rows = 2;
    let prepared = PreparedRowLookup::new(
        range,
        spec,
        Some(scale),
        RowLookupLimits {
            rows_per_acquisition: 1,
            ..limits()
        },
    )
    .unwrap();
    check_row_memory(&prepared);
    let memory = MlxRowLookupSupport
        .decode_memory(prepared.descriptor())
        .unwrap();
    assert_eq!(
        memory
            .storage
            .iter()
            .find(|s| s.name == "scale_f32")
            .unwrap()
            .payload
            .upper,
        Some(4)
    );
    let query = eredu_runtime::MechanismResourceQuery {
        invocation: ResourceIdentity {
            scope: "fixture".into(),
            key: "decode".into(),
        },
        execution_pool: Observed::Unavailable {
            reason: "cold query has no device".into(),
        },
        host_pool: Observed::Unavailable {
            reason: "cold query has no physical host pool".into(),
        },
        owner_backings: BTreeMap::new(),
    };
    let resources = eredu_runtime::describe_mechanism_resources(memory, &query).unwrap();
    let completion = ResourceIdentity {
        scope: "fixture".into(),
        key: "completed".into(),
    };
    let owner = ResourceIdentity {
        scope: "fixture".into(),
        key: "consumer".into(),
    };
    let output = resources.storage_bindings["output"].clone();
    let lifetimes = eredu_runtime::resource_lifetimes::describe_mechanism_lifetimes(
        &resources,
        &eredu_runtime::resource_lifetimes::MechanismLifetimeBindings {
            completion: completion.clone(),
            evaluation: None,
            owners: BTreeMap::from([(output.clone(), owner.clone())]),
        },
    )
    .unwrap();
    assert_eq!(
        lifetimes.lifetimes[&output],
        [ResourceLifetime::Owner(owner)]
    );
    for (name, storage) in &resources.storage_bindings {
        if name != "output" {
            assert_eq!(
                lifetimes.lifetimes[storage],
                [ResourceLifetime::NativeCompletion(completion.clone())]
            );
        }
    }
    assert!(matches!(
        lifetimes.resources.coverage,
        ResourceCoverage::Partial { .. }
    ));
}
