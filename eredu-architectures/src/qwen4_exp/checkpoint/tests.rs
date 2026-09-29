use super::*;
use eredu_checkpoint::store::{
    CheckpointLease, MemoryWeightStore, TensorMetadata, WeightStoreDiagnostics,
};
use std::sync::Mutex;
mod released;
struct Source {
    inner: MemoryWeightStore,
    reads: Mutex<Vec<String>>,
}
impl CheckpointSource for Source {
    fn source_keys(&self) -> Vec<String> {
        self.inner.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        self.inner.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        assert!(
            !request.key.contains(".shard_"),
            "preparation must not read table payloads"
        );
        self.reads.lock().unwrap().push(request.key.clone());
        self.inner.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.inner.source_diagnostics()
    }
}
fn config() -> Config {
    let mut root: serde_json::Value =
        serde_json::from_str(include_str!("../config/released.json")).unwrap();
    root["text_config"]["heads_per_ngram"] = 1.into();
    root["text_config"]["ple_embed_dim"] = 4.into();
    root["text_config"]["make_ngram_vocab_size_divisible_by"] = 4.into();
    root["text_config"]["split_ngram_parts"] = 2.into();
    Config::from_json(&root).unwrap()
}
fn source(
    root: &str,
    fp8: bool,
    mutate: impl FnOnce(&mut Vec<(String, safetensors::Dtype, Vec<usize>, Vec<u8>)>),
) -> Arc<Source> {
    use safetensors::Dtype as D;
    let mut values = vec![
        (
            format!("{root}.layer_multipliers"),
            D::I64,
            vec![3],
            [23703573157769i64, 20109073645365, 8052911324071]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect(),
        ),
        (
            format!("{root}.ngram_heads_vocab_sizes"),
            D::I64,
            vec![2],
            [11i64, 13].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
        (
            format!("{root}.ngram_heads_offsets"),
            D::I64,
            vec![2],
            [0i64, 11].iter().flat_map(|v| v.to_le_bytes()).collect(),
        ),
    ];
    for part in 0..2 {
        values.push((
            format!("{root}.ngram_embedding.shard_{part}.weight"),
            if fp8 { D::F8_E4M3 } else { D::BF16 },
            vec![12, 2],
            vec![0; 24 * if fp8 { 1 } else { 2 }],
        ));
    }
    if fp8 {
        values.push((
            format!("{root}.ngram_embedding.weight_scale"),
            D::BF16,
            vec![1],
            vec![0x51, 0x39],
        ));
    }
    mutate(&mut values);
    Arc::new(Source {
        inner: MemoryWeightStore::from_safetensors(values).unwrap(),
        reads: Mutex::new(vec![]),
    })
}
#[test]
fn table_preparation_rejects_missing_malformed_or_ambiguous_companions() {
    let root = "model.language_model.layers.1.ple.ple_embedding";
    for case in 0..10 {
        let source = source(root, true, |values| match case {
            0 => {
                values.retain(|(name, ..)| !name.ends_with("weight_scale"));
            }
            1 => {
                let scale = values.last_mut().unwrap();
                scale.3 = vec![0, 0];
            }
            2 => {
                let scale = values.last_mut().unwrap();
                scale.2 = vec![2];
                scale.3 = vec![0; 4];
            }
            3 => {
                values[0].1 = safetensors::Dtype::F32;
                values[0].3 = vec![0; 12];
            }
            4 => {
                values[2].3 = [0i64, 10].iter().flat_map(|v| v.to_le_bytes()).collect();
            }
            5 => {
                values.remove(3);
            }
            6 => {
                let mut duplicate = values[0].clone();
                duplicate.0 = "model.layers.1.ple.ple_embedding.layer_multipliers".into();
                values.push(duplicate);
            }
            7 => {
                values[2].2 = vec![1];
                values[2].3.truncate(8);
            }
            8 => {
                let mut duplicate = values[3].clone();
                duplicate.0 = "layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight".into();
                values.push(duplicate);
            }
            9 => {
                let scale = values.last_mut().unwrap();
                scale.1 = safetensors::Dtype::F8_E4M3;
                scale.3 = vec![0x38];
            }
            _ => unreachable!(),
        });
        let header = SafetensorsTableSourcePlan::prepare(
            source.as_ref() as &dyn CheckpointSource,
            &config(),
            1,
        );
        assert!(source.reads.lock().unwrap().is_empty(), "case {case}");
        if matches!(case, 1 | 4) {
            assert!(
                header
                    .unwrap()
                    .bind(source, 2, 1, TensorElementType::Bf16)
                    .is_err(),
                "case {case}"
            );
        } else {
            assert!(header.is_err(), "case {case}");
        }
    }
}

#[test]
fn table_header_plan_accepts_unreadable_catalog_and_binding_checks_all_physical_headers_first() {
    use eredu_checkpoint::validation::{CatalogTensorMetadata, SafetensorsCatalog};
    struct Headers(BTreeMap<String, CatalogTensorMetadata>);
    impl SafetensorsCatalog for Headers {
        fn keys(&self) -> Vec<String> {
            self.0.keys().cloned().collect()
        }
        fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
            self.0
                .get(key)
                .cloned()
                .ok_or_else(|| format!("missing {key}"))
        }
    }
    struct PhysicalOverride {
        source: Arc<Source>,
        case: usize,
        calls: std::sync::atomic::AtomicUsize,
    }
    impl CheckpointSource for PhysicalOverride {
        fn source_keys(&self) -> Vec<String> {
            self.source.source_keys()
        }
        fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
            let mut meta = self.source.source_metadata(key)?;
            if key.ends_with("weight_scale") {
                match self.case {
                    0 => meta.encoded_byte_len += 1,
                    1 => meta.physical_shape = vec![2],
                    2 => meta.stored_dtype = StoredDtype::F32,
                    3 => {
                        if self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) > 0 {
                            meta.logical_shape = vec![2];
                            meta.physical_shape = vec![2];
                            meta.encoded_byte_len = 4;
                        }
                    }
                    _ => unreachable!(),
                }
            }
            Ok(meta)
        }
        fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            self.source.acquire_lease(request)
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            self.source.source_diagnostics()
        }
    }
    let root = "model.layers.1.ple.ple_embedding";
    let source = source(root, true, |_| {});
    let catalog = Headers(
        source
            .source_keys()
            .into_iter()
            .map(|key| {
                let meta = source.source_metadata(&key).unwrap();
                (
                    key,
                    CatalogTensorMetadata {
                        shape: meta.logical_shape,
                        stored_dtype: meta.stored_dtype,
                    },
                )
            })
            .collect(),
    );
    let header = SafetensorsTableSourcePlan::prepare(&catalog, &config(), 1).unwrap();
    assert!(source.reads.lock().unwrap().is_empty());
    for case in 0..4 {
        assert!(header
            .bind(
                Arc::new(PhysicalOverride {
                    source: source.clone(),
                    case,
                    calls: std::sync::atomic::AtomicUsize::new(0),
                }),
                2,
                1,
                TensorElementType::Bf16
            )
            .is_err());
        assert!(
            source.reads.lock().unwrap().is_empty(),
            "physical mismatch {case} read controls"
        );
    }
    assert!(header
        .bind(source.clone(), 2, 1, TensorElementType::I32)
        .is_err());
    assert!(source.reads.lock().unwrap().is_empty());
    header
        .bind(source.clone(), 2, 1, TensorElementType::Bf16)
        .unwrap();
    assert_eq!(source.reads.lock().unwrap().len(), 4);
}

#[test]
fn deferred_literals_preserve_integers_above_float_precision_and_scalar_rank() {
    let root = "layers.1.ple.ple_embedding";
    let constants = [9_007_199_254_740_993i64, i64::MAX - 2, i64::MIN + 1];
    let source = source(root, true, |values| {
        values[0].3 = constants.iter().flat_map(|v| v.to_le_bytes()).collect();
        values.last_mut().unwrap().2.clear();
    });
    let config = config();
    let header =
        SafetensorsTableSourcePlan::prepare(source.as_ref() as &dyn CheckpointSource, &config, 1)
            .unwrap();
    assert!(source.reads.lock().unwrap().is_empty());
    let table = header
        .bind(source.clone(), 2, 1, TensorElementType::Bf16)
        .unwrap();
    assert_eq!(
        table.hash,
        NGramHashSpec::new(
            config.vocabulary as u64,
            config.eos[0] as u64,
            3,
            1,
            constants.to_vec(),
            vec![11, 13],
            vec![0, 11],
            24
        )
        .unwrap()
    );
    assert_eq!(
        table
            .hash
            .select(Some(&[3, 7]), 1, 2, &[5, 9], 4)
            .unwrap()
            .rows
            .len(),
        4
    );
    assert_eq!(source.reads.lock().unwrap().len(), 4);
}

#[test]
fn header_row_selection_binds_exact_table_without_repeating_cold_queries() {
    use eredu_core::residency::{OffloadConfig, ResidencyPolicy};
    use eredu_runtime::{
        AddressableStorageCapabilities, ParameterBankLoadOptions, PreparedRowLookup,
        PreparedRowLookups, RowLookupDescriptor, RowLookupDescriptors, RowLookupError,
        RowLookupLimits, RowLookupMechanismSupport, RowLookupWorkspace, RowResidencyRange,
        SelectedRowLookupPlans,
    };
    use std::cell::Cell;
    struct Support {
        storage: Cell<usize>,
        workspace: Cell<usize>,
    }
    impl RowLookupMechanismSupport for Support {
        fn storage(&self) -> Option<AddressableStorageCapabilities> {
            self.storage.set(self.storage.get() + 1);
            Some(AddressableStorageCapabilities::new(true, true, true, 4096))
        }
        fn workspace(
            &self,
            entry: &RowLookupDescriptor,
        ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
            self.workspace.set(self.workspace.get() + 1);
            Ok(Some(RowLookupWorkspace {
                decode_bytes: 64,
                scalar_bytes: if entry.scale().is_some() { 2 } else { 0 },
            }))
        }
    }
    let limits = RowLookupLimits {
        requests: 8,
        rows_per_acquisition: 2,
        acquisition_bytes: 16,
        host_bytes: 1024,
        output_bytes: 192,
    };
    for root in [
        "model.language_model.layers.1.ple.ple_embedding",
        "model.layers.1.ple.ple_embedding",
        "layers.1.ple.ple_embedding",
    ] {
        for fp8 in [false, true] {
            let source = source(root, fp8, |values| {
                if fp8 && root.starts_with("layers.") {
                    values.last_mut().unwrap().2.clear();
                }
            });
            let header = SafetensorsTableSourcePlan::prepare(
                source.as_ref() as &dyn CheckpointSource,
                &config(),
                1,
            )
            .unwrap();
            let descriptor = header
                .row_descriptor(
                    2,
                    1,
                    TensorElementType::Bf16,
                    limits,
                    ResidencyPolicy::Cacheable,
                )
                .unwrap();
            assert!(matches!(
                header.row_descriptor(
                    2,
                    1,
                    TensorElementType::Bf16,
                    RowLookupLimits {
                        host_bytes: 1023,
                        ..limits
                    },
                    ResidencyPolicy::Cacheable
                ),
                Err(NGramArtifactError::Lookup(RowLookupError::Budget {
                    resource: "host planning bytes",
                    ..
                }))
            ));
            assert_eq!(
                descriptor.range().prefix().as_str(),
                "parameter_bank.2.1.rows"
            );
            assert_eq!(descriptor.metadata().shape, [24, 2]);
            assert_eq!(descriptor.metadata().byte_len, if fp8 { 48 } else { 96 });
            if let Some(scale) = descriptor.scale() {
                assert_eq!(
                    scale.catalog().keys().cloned().collect::<Vec<_>>(),
                    [format!("{root}.ngram_embedding.weight_scale")]
                );
                assert!(scale
                    .catalog()
                    .values()
                    .all(|entry| entry.metadata.backing_shard.is_none()));
            }
            let support = Support {
                storage: Cell::new(0),
                workspace: Cell::new(0),
            };
            let selected = SelectedRowLookupPlans::select(
                RowLookupDescriptors::new([descriptor.clone()], 2).unwrap(),
                ParameterBankLoadOptions::new(
                    OffloadConfig::new(Some(64), Some(0), 1).unwrap(),
                    4096,
                    4096,
                )
                .unwrap(),
                4,
                &support,
            )
            .unwrap();
            assert!(source.reads.lock().unwrap().is_empty());
            assert_eq!((support.storage.get(), support.workspace.get()), (1, 1));
            let table = header
                .bind(source.clone(), 2, 1, TensorElementType::Bf16)
                .unwrap();
            let range = RowResidencyRange::new(
                table_row_range(
                    &table.lookup,
                    table.rows.metadata(),
                    ResidencyPolicy::Cacheable,
                )
                .unwrap(),
                table.rows,
                table.lookup.parameter.as_str(),
            )
            .unwrap();
            let bound = PreparedRowLookup::new(range, table.lookup, table.scale, limits).unwrap();
            assert_eq!(bound.descriptor(), &descriptor);
            selected
                .bind(PreparedRowLookups::new([bound], 2).unwrap())
                .unwrap();
            assert_eq!((support.storage.get(), support.workspace.get()), (1, 1));
            assert_eq!(source.reads.lock().unwrap().len(), if fp8 { 4 } else { 3 });
        }
    }
}
