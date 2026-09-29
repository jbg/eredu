use super::*;
use eredu_checkpoint::{rows::RowReadLimits, store::CheckpointLease};
use eredu_gguf::{
    ConvertedTensor, Endian, GgmlType, MetadataArray, TensorInput, Writer, WriterOptions,
};

const MULTIPLIERS: [u64; 3] = [23703573157769, 9007199254740993, 8052911324071];
fn config() -> Config {
    let mut json: serde_json::Value =
        serde_json::from_str(include_str!("../../config/released.json")).unwrap();
    let text = &mut json["text_config"];
    text["heads_per_ngram"] = 1.into();
    text["ple_embed_dim"] = 64.into();
    text["make_ngram_vocab_size_divisible_by"] = 4.into();
    text["split_ngram_parts"] = 2.into();
    Config::from_json(&json).unwrap()
}
fn metadata(config: &Config) -> BTreeMap<String, MetadataValue> {
    let mut result = BTreeMap::from([(
        "general.architecture".into(),
        MetadataValue::String("qwen4exp".into()),
    )]);
    for (key, value) in [
        ("ple.ngram_size", 3),
        ("ple.heads_per_ngram", 1),
        ("ple.conv_kernel", config.ngram.kernel as u32),
        ("ple.eos_token_id", config.eos[0]),
        ("ple.image_token_id", config.media.as_ref().unwrap().image),
        ("embedding_length_per_layer_input", 32),
    ] {
        result.insert(format!("qwen4exp.{key}"), MetadataValue::Uint32(value));
    }
    result.insert(
        "qwen4exp.ple.layers".into(),
        MetadataValue::Array(MetadataArray::Int32(vec![1])),
    );
    for (key, values) in [
        ("ple.layer_multipliers", MULTIPLIERS.to_vec()),
        ("ple.head_vocab_sizes", vec![11, 13]),
        ("ple.head_offsets", vec![0, 11]),
    ] {
        result.insert(
            format!("qwen4exp.{key}"),
            MetadataValue::Array(MetadataArray::Uint64(values)),
        );
    }
    result
}
fn checkpoint(
    metadata: &BTreeMap<String, MetadataValue>,
    ty: GgmlType,
    endian: Endian,
    rows: u64,
    width: u64,
    name: &str,
) -> (tempfile::TempDir, Checkpoint) {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table.gguf");
    let mut bytes = vec![];
    for row in 0..rows as i32 {
        if ty == GgmlType::Q8_0 {
            assert_eq!(width, 32);
            let scale = half::f16::from_f32(0.25).to_bits();
            bytes.extend(if endian == Endian::Little {
                scale.to_le_bytes()
            } else {
                scale.to_be_bytes()
            });
            bytes.extend((0..32).map(|i| (i - 16 + row) as i8 as u8));
        } else if ty == GgmlType::Q4_0 {
            let scale = half::f16::from_f32(0.25).to_bits();
            bytes.extend(if endian == Endian::Little {
                scale.to_le_bytes()
            } else {
                scale.to_be_bytes()
            });
            bytes.extend((0..16).map(|i| ((i + row) % 16) as u8 * 17));
        } else if ty == GgmlType::MxFp4 {
            bytes.push(127);
            bytes.extend((0..16).map(|i| ((i + row) % 16) as u8 * 17));
        } else {
            for i in 0..width as i32 {
                let value = (i - 16 + row) as f32 * 0.25;
                if ty == GgmlType::F32 {
                    bytes.extend(if endian == Endian::Little {
                        value.to_le_bytes()
                    } else {
                        value.to_be_bytes()
                    });
                } else {
                    let bits = if ty == GgmlType::F16 {
                        half::f16::from_f32(value).to_bits()
                    } else {
                        half::bf16::from_f32(value).to_bits()
                    };
                    bytes.extend(if endian == Endian::Little {
                        bits.to_le_bytes()
                    } else {
                        bits.to_be_bytes()
                    });
                }
            }
        }
    }
    Writer::new(WriterOptions {
        endian,
        ..WriterOptions::default()
    })
    .unwrap()
    .write(
        std::fs::File::create(&path).unwrap(),
        metadata,
        &[
            TensorInput {
                name,
                dimensions: &[width, rows],
                ggml_type: ty,
                data: &bytes,
            },
            // A separate owner is deliberately present and inaccessible to rows.
            TensorInput {
                name: "output.weight",
                dimensions: &[1, 1],
                ggml_type: GgmlType::F32,
                data: &[0; 4],
            },
        ],
    )
    .unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    (directory, checkpoint)
}

#[test]
fn gguf_cold_row_selection_needs_no_file_and_binds_exact_encoded_descriptor() {
    use eredu_core::residency::{OffloadConfig, ResidencyPolicy};
    use eredu_runtime::{
        AddressableStorageCapabilities, ParameterBankLoadOptions, PreparedRowLookup,
        PreparedRowLookups, RowLookupDescriptor, RowLookupDescriptors, RowLookupError,
        RowLookupLimits, RowLookupMechanismSupport, RowLookupWorkspace, RowResidencyRange,
        SelectedRowLookupPlans,
    };
    use std::cell::Cell;
    struct Support(Cell<usize>);
    impl RowLookupMechanismSupport for Support {
        fn storage(&self) -> Option<AddressableStorageCapabilities> {
            self.0.set(self.0.get() + 1);
            Some(AddressableStorageCapabilities::new(true, true, true, 16384))
        }
        fn workspace(
            &self,
            _: &RowLookupDescriptor,
        ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
            self.0.set(self.0.get() + 1);
            Ok(Some(RowLookupWorkspace {
                decode_bytes: 512,
                scalar_bytes: 0,
            }))
        }
    }
    let config = config();
    let limits = RowLookupLimits {
        requests: 8,
        rows_per_acquisition: 2,
        acquisition_bytes: 256,
        host_bytes: 4096,
        output_bytes: 4096,
    };
    for ty in [
        GgmlType::Q8_0,
        GgmlType::Q4_0,
        GgmlType::MxFp4,
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Bf16,
    ] {
        for endian in [Endian::Little, Endian::Big] {
            let (directory, checkpoint) = checkpoint(&metadata(&config), ty, endian, 24, 32, TABLE);
            // Header plans and cold selection remain usable while the artifact
            // is inaccessible. There is no readable store or cached reader yet.
            let path = directory.path().join("table.gguf");
            let hidden = directory.path().join("table.unavailable");
            std::fs::rename(&path, &hidden).unwrap();
            let plan = GgufTableSourcePlan::prepare(&checkpoint, &config).unwrap();
            let descriptor = plan
                .row_descriptor(
                    1,
                    2,
                    1,
                    TensorElementType::F32,
                    limits,
                    ResidencyPolicy::Cacheable,
                )
                .unwrap();
            assert_eq!(descriptor.spec().rows, 24);
            assert_eq!(descriptor.spec().dimensions, 32);
            assert_eq!(
                descriptor.range().prefix().as_str(),
                "parameter_bank.2.1.rows"
            );
            assert!(descriptor.scale().is_none());
            assert!(plan.tensor_metadata(TABLE).unwrap().backing_shard.is_none());
            assert!(plan.tensor_metadata("output.weight").is_err());
            assert!(plan.lookup_spec(2, 2, 1, TensorElementType::F32).is_err());
            let row_bytes = match ty {
                GgmlType::Q8_0 => 34,
                GgmlType::Q4_0 => 18,
                GgmlType::MxFp4 => 17,
                GgmlType::F32 => 128,
                _ => 64,
            };
            assert_eq!(descriptor.metadata().byte_len, 24 * row_bytes);
            assert_eq!(
                descriptor.spec().encoding,
                if ty.block_and_bytes().unwrap().0 > 1 {
                    RowEncoding::Gguf {
                        encoding: ty,
                        endian,
                    }
                } else {
                    RowEncoding::Dense
                }
            );
            assert!(matches!(
                plan.row_descriptor(
                    1,
                    2,
                    1,
                    TensorElementType::F32,
                    RowLookupLimits {
                        acquisition_bytes: row_bytes - 1,
                        ..limits
                    },
                    ResidencyPolicy::Cacheable,
                ),
                Err(NGramArtifactError::Lookup(RowLookupError::Budget { .. }))
            ));
            let support = Support(Cell::new(0));
            let selected = SelectedRowLookupPlans::select(
                RowLookupDescriptors::new([descriptor.clone()], 2).unwrap(),
                ParameterBankLoadOptions::new(
                    OffloadConfig::new(Some(512), Some(0), 1).unwrap(),
                    16384,
                    16384,
                )
                .unwrap(),
                0,
                &support,
            )
            .unwrap();
            let cold_queries = support.0.get();
            assert_eq!(cold_queries, 2);
            std::fs::rename(hidden, path).unwrap();
            let source = source(&plan).unwrap();
            let table = plan
                .bind(source.clone(), 1, 2, 1, TensorElementType::F32)
                .unwrap();
            assert_eq!(table.lookup, *descriptor.spec());
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
            assert_eq!(support.0.get(), cold_queries);
            assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
        }
    }
}
fn prepare(
    checkpoint: &Checkpoint,
    config: &Config,
) -> Result<PreparedNGramTable, NGramArtifactError> {
    let plan = GgufTableSourcePlan::prepare(checkpoint, config)?;
    let source = source(&plan)?;
    plan.bind(source, 1, 2, 1, TensorElementType::F32)
}

fn source(plan: &GgufTableSourcePlan) -> Result<SharedCheckpointSource, NGramArtifactError> {
    let contract = eredu_checkpoint::schema::GgufCheckpointPlan::new(
        "table fixture",
        vec![plan.constraint().clone()],
        vec![],
        eredu_checkpoint::schema::CatalogPolicy::non_strict(),
    )
    .map_err(|e| invalid(e.to_string()))?;
    let mapping = plan
        .checkpoint()
        .translated_outputs(str::to_owned)
        .map_err(|e| invalid(e.to_string()))?;
    Ok(Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(1)?
            .add_checkpoint(plan.checkpoint().clone(), &contract, &mapping)?
            .build()?,
    ))
}

#[test]
fn gguf_table_retains_exact_controls_and_reads_only_selected_nonzero_rows() {
    let config = config();
    for (ty, endian) in [
        (GgmlType::Q8_0, Endian::Little),
        (GgmlType::Q4_0, Endian::Little),
        (GgmlType::Q4_0, Endian::Big),
        (GgmlType::MxFp4, Endian::Little),
        (GgmlType::MxFp4, Endian::Big),
        (GgmlType::F32, Endian::Little),
        (GgmlType::Bf16, Endian::Little),
        (GgmlType::F32, Endian::Big),
        (GgmlType::Bf16, Endian::Big),
    ] {
        let metadata = metadata(&config);
        let (_directory, checkpoint) = checkpoint(&metadata, ty, endian, 24, 32, TABLE);
        let table = prepare(&checkpoint, &config).unwrap();
        // The source remains usable after the caller drops the header object.
        drop(checkpoint);
        assert_eq!(table.lookup.rows, 24);
        assert_eq!(table.lookup.dimensions, 32);
        assert!(table.scale.is_none());
        let NGramControls::Gguf {
            metadata: retained,
            table: provenance,
        } = &table.controls
        else {
            panic!("GGUF provenance")
        };
        assert_eq!(retained, &metadata);
        assert_eq!(provenance.physical_tensor, TABLE);
        assert_eq!(
            provenance.source_encoding,
            eredu_checkpoint::SourceTensorEncoding::Gguf {
                ggml_type: ty,
                endian
            }
        );
        assert_eq!(table.rows.source_keys(), [TABLE]);
        assert!(table.rows.source_metadata("output.weight").is_err());
        assert_eq!(
            table.rows.source_diagnostics().unwrap().physical_read_bytes,
            0
        );
        let requests = [23, 12, 11, 12, 0];
        let plan = table
            .rows
            .plan(
                &requests,
                RowReadLimits {
                    requests: 5,
                    rows_per_read: 2,
                },
            )
            .unwrap();
        assert_eq!(plan.unique_rows(), [0, 11, 12, 23]);
        assert_eq!(plan.restore_order(), [3, 2, 1, 2, 0]);
        assert_eq!(plan.reads().len(), 3);
        let mut compact = vec![];
        for read in plan.reads() {
            let DerivedWeightRecipe::Source { key, selection } = read.recipe() else {
                panic!("one bounded physical range")
            };
            let lease = plan
                .source()
                .acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: selection.clone(),
                    policy: ReadPolicy::RequireBounded,
                })
                .unwrap();
            assert!(lease.bounded_read_proof().physically_bounded);
            assert_eq!(
                lease.bounded_read_proof().length_bytes,
                read.destinations().len() as u64
                    * match ty {
                        GgmlType::Q8_0 => 34,
                        GgmlType::Q4_0 => 18,
                        GgmlType::MxFp4 => 17,
                        GgmlType::Bf16 => 64,
                        _ => 128,
                    }
            );
            let CheckpointLease::Gguf(lease) = lease else {
                panic!("retained GGUF source")
            };
            match lease.materialize_portable().unwrap().into_converted() {
                ConvertedTensor::IQuant(t) => compact.extend(t.dequantize_f32().unwrap()),
                ConvertedTensor::Dense(t) if ty == GgmlType::F32 => compact.extend(
                    t.data
                        .chunks_exact(4)
                        .map(|v| f32::from_ne_bytes(v.try_into().unwrap())),
                ),
                ConvertedTensor::Dense(t) => compact.extend(t.data.chunks_exact(2).map(|v| {
                    half::bf16::from_bits(u16::from_ne_bytes(v.try_into().unwrap())).to_f32()
                })),
                _ => panic!("unexpected converted layout"),
            }
        }
        for (&row, &slot) in requests.iter().zip(plan.restore_order()) {
            let expected = (0..32)
                .map(|i| match ty {
                    GgmlType::Q4_0 => ((i + row as i32) % 16 - 8) as f32 * 0.25,
                    GgmlType::MxFp4 => [
                        0., 0.5, 1., 1.5, 2., 3., 4., 6., 0., -0.5, -1., -1.5, -2., -3., -4., -6.,
                    ][((i + row as i32) % 16) as usize],
                    _ => (i - 16 + row as i32) as f32 * 0.25,
                })
                .collect::<Vec<_>>();
            assert_eq!(&compact[slot * 32..(slot + 1) * 32], expected);
        }
        let diagnostics = table.rows.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 3);
        assert_eq!(
            diagnostics.physical_read_bytes,
            4 * match ty {
                GgmlType::Q8_0 => 34,
                GgmlType::Q4_0 => 18,
                GgmlType::MxFp4 => 17,
                GgmlType::Bf16 => 64,
                _ => 128,
            }
        );
        // This is the same literal integer contract as the original I64 buffers.
        let independent = NGramHashSpec::new(
            config.vocabulary as u64,
            config.eos[0] as u64,
            3,
            1,
            MULTIPLIERS.map(|v| v as i64).to_vec(),
            vec![11, 13],
            vec![0, 11],
            24,
        )
        .unwrap();
        assert_eq!(table.hash, independent);
        let ids = [3, config.eos[0] as u64, 7, 4];
        let history = table.hash.initial_history(1).unwrap();
        let full = table.hash.select(Some(&ids), 1, 4, &history, 8).unwrap();
        let first = table
            .hash
            .select(Some(&ids[..1]), 1, 1, &history, 2)
            .unwrap();
        let second = table
            .hash
            .select(Some(&ids[1..]), 1, 3, &first.next_history, 6)
            .unwrap();
        assert_eq!(full.rows, [first.rows, second.rows].concat());
        assert_eq!(full.next_history, second.next_history);
    }
}

#[test]
fn gguf_table_rejects_missing_rounded_overflowed_and_disagreeing_metadata() {
    let config = config();
    let baseline = metadata(&config);
    for key in baseline.keys().filter(|k| {
        !k.ends_with("image_token_id") && !k.ends_with("embedding_length_per_layer_input")
    }) {
        let mut missing = baseline.clone();
        missing.remove(key);
        let (_directory, checkpoint) =
            checkpoint(&missing, GgmlType::Q8_0, Endian::Little, 24, 32, TABLE);
        assert!(prepare(&checkpoint, &config).is_err(), "missing {key}");
    }
    for (key, value) in [
        (
            "general.architecture",
            MetadataValue::String("qwen35".into()),
        ),
        (
            "qwen4exp.ple.layer_multipliers",
            MetadataValue::Array(MetadataArray::Float64(
                MULTIPLIERS.map(|n| n as f64).to_vec(),
            )),
        ),
        (
            "qwen4exp.ple.layer_multipliers",
            MetadataValue::Array(MetadataArray::Uint64(vec![u64::MAX; 3])),
        ),
        (
            "qwen4exp.ple.layer_multipliers",
            MetadataValue::Array(MetadataArray::Uint64(vec![1; 4097])),
        ),
        (
            "qwen4exp.ple.head_offsets",
            MetadataValue::Array(MetadataArray::Uint64(vec![0, 10])),
        ),
        (
            "qwen4exp.ple.head_vocab_sizes",
            MetadataValue::Array(MetadataArray::Int64(vec![11, -13])),
        ),
        (
            "qwen4exp.ple.layers",
            MetadataValue::Array(MetadataArray::Int32(vec![2])),
        ),
        (
            "qwen4exp.ple.eos_token_id",
            MetadataValue::Uint32(config.eos[0] + 1),
        ),
        ("qwen4exp.ple.ngram_size", MetadataValue::Float32(3.)),
        (
            "qwen4exp.embedding_length_per_layer_input",
            MetadataValue::Uint32(64),
        ),
        (
            "qwen4exp.ple.image_token_id",
            MetadataValue::Uint64(u64::MAX),
        ),
    ] {
        let mut malformed = baseline.clone();
        malformed.insert(key.into(), value);
        let (_directory, checkpoint) =
            checkpoint(&malformed, GgmlType::Q8_0, Endian::Little, 24, 32, TABLE);
        assert!(prepare(&checkpoint, &config).is_err(), "malformed {key}");
    }
    for (rows, width, name) in [
        (28, 32, TABLE),
        (24, 16, TABLE),
        (24, 32, "unrelated.weight"),
    ] {
        let (_directory, checkpoint) =
            checkpoint(&baseline, GgmlType::F32, Endian::Little, rows, width, name);
        assert!(prepare(&checkpoint, &config).is_err());
    }
}

#[test]
fn gguf_table_optional_metadata_uses_physical_width_without_inventing_original_ids() {
    let config = config();
    let mut metadata = metadata(&config);
    metadata.remove("qwen4exp.embedding_length_per_layer_input");
    metadata.remove("qwen4exp.ple.image_token_id");
    let (_directory, checkpoint) =
        checkpoint(&metadata, GgmlType::Q8_0, Endian::Little, 24, 32, TABLE);
    let table = prepare(&checkpoint, &config).unwrap();
    assert_eq!(table.lookup.dimensions, 32);
    let history = table.hash.initial_history(1).unwrap();
    assert_eq!(
        table.hash.select(None, 1, 1, &history, 2),
        Err(NGramError::MissingTokenIds)
    );
    let NGramControls::Gguf {
        metadata: retained, ..
    } = table.controls
    else {
        panic!("GGUF controls")
    };
    assert_eq!(retained, metadata);
}

#[test]
fn retained_table_source_shares_cache_between_injection_owners_and_rejects_substitution() {
    let mut config = config();
    config.ngram.layers = vec![1, 3];
    let mut metadata = metadata(&config);
    metadata.insert(
        "qwen4exp.ple.layers".into(),
        MetadataValue::Array(MetadataArray::Int32(vec![1, 3])),
    );
    let (_directory, checkpoint) =
        checkpoint(&metadata, GgmlType::Q8_0, Endian::Little, 24, 32, TABLE);
    let plan = GgufTableSourcePlan::prepare(&checkpoint, &config).unwrap();
    let source = source(&plan).unwrap();
    let first = plan
        .bind(source.clone(), 1, 1, 1, TensorElementType::F32)
        .unwrap();
    let second = plan
        .bind(source.clone(), 3, 2, 4, TensorElementType::F32)
        .unwrap();
    assert_ne!(first.lookup.parameter, second.lookup.parameter);
    assert_ne!(first.lookup.bank, second.lookup.bank);
    assert_ne!(first.lookup.unit, second.lookup.unit);
    assert_eq!(
        first.rows.source_provenance(TABLE).unwrap(),
        second.rows.source_provenance(TABLE).unwrap()
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    for (owner, row) in [(&first, 2), (&second, 7)] {
        let read = owner
            .rows
            .plan(
                &[row],
                RowReadLimits {
                    requests: 1,
                    rows_per_read: 1,
                },
            )
            .unwrap();
        let DerivedWeightRecipe::Source { key, selection } = read.reads()[0].recipe() else {
            panic!("row range")
        };
        let lease = read
            .source()
            .acquire_lease(TensorReadRequest {
                key: key.clone(),
                selection: selection.clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert_eq!(lease.bounded_read_proof().length_bytes, 34);
        let CheckpointLease::Gguf(lease) = lease else {
            panic!("GGUF lease")
        };
        lease.materialize_portable().unwrap();
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(first.rows.source_diagnostics().unwrap(), diagnostics);
        assert_eq!(second.rows.source_diagnostics().unwrap(), diagnostics);
    }
    let diagnostics = source.source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 2);
    assert_eq!(diagnostics.physical_read_bytes, 68);
    assert_eq!(diagnostics.cache_misses, 1);
    assert!(diagnostics.cache_hits >= 1);
    assert_eq!(diagnostics.currently_cached_shards, 1);
    assert!(plan
        .bind(source.clone(), 2, 3, 5, TensorElementType::F32)
        .is_err());

    let (_other_directory, other_checkpoint) =
        self::checkpoint(&metadata, GgmlType::Q8_0, Endian::Little, 24, 32, TABLE);
    let other_plan = GgufTableSourcePlan::prepare(&other_checkpoint, &config).unwrap();
    let other_source = self::source(&other_plan).unwrap();
    assert!(plan
        .bind(other_source.clone(), 1, 1, 1, TensorElementType::F32)
        .is_err());
    assert_eq!(other_source.source_diagnostics().unwrap().physical_reads, 0);
}
