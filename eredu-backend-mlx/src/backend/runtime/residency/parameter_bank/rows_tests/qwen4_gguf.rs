//! Architecture-prepared GGUF rows consumed without family logic in native binding.
use super::*;
use eredu_architectures::qwen4_exp::{checkpoint::GgufTableSourcePlan, config::Config};
use eredu_gguf::{Endian, GgmlType, MetadataArray as A, MetadataValue as V};

#[test]
fn architecture_prepared_gguf_table_uses_bounded_native_rows_and_preserves_evicted_output() {
    for endian in [Endian::Little, Endian::Big] {
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::MxFp4,
        ] {
            run_table(ty, endian);
        }
    }
}
fn run_table(ty: GgmlType, endian: Endian) {
    let stream = stream();
    let (width, row_bytes) = ty.block_and_bytes().unwrap();
    let half = |v: f32| {
        let bits = half::f16::from_f32(v).to_bits();
        if endian == Endian::Little {
            bits.to_le_bytes()
        } else {
            bits.to_be_bytes()
        }
    };
    let mut raw = vec![];
    let mut reference = vec![];
    for row in 0..24i32 {
        let d = (row + 1) as f32 * 0.25;
        match ty {
            GgmlType::Q8_0 => {
                raw.extend(half(d));
                raw.extend((0..32).map(|i| (i - 16) as i8 as u8));
                reference.extend((0..32).map(|i| (i - 16) as f32 * d));
            }
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 => {
                raw.extend(half(d));
                let q5 = ty == GgmlType::Q5_0;
                let asymmetric = ty == GgmlType::Q4_1;
                if asymmetric {
                    raw.extend(half(-0.5));
                }
                let codes: Vec<_> = (0..32)
                    .map(|i| ((i * 7 + row) % if q5 { 32 } else { 16 }) as u8)
                    .collect();
                if q5 {
                    let high = codes
                        .iter()
                        .enumerate()
                        .fold(0u32, |n, (i, v)| n | (u32::from(*v >> 4) << i));
                    raw.extend(if endian == Endian::Little {
                        high.to_le_bytes()
                    } else {
                        high.to_be_bytes()
                    });
                }
                raw.extend((0..16).map(|i| (codes[i] & 15) | ((codes[i + 16] & 15) << 4)));
                reference.extend(codes.iter().map(|&c| {
                    if asymmetric {
                        d * c as f32 - 0.5
                    } else {
                        d * (i32::from(c) - if q5 { 16 } else { 8 }) as f32
                    }
                }));
            }
            GgmlType::Q2K => {
                raw.extend([0x21; 16]);
                raw.extend([0xe4; 64]);
                raw.extend(half(d));
                raw.extend(half(0.125));
                reference.extend((0..256).map(|i| d * ((i / 32) % 4) as f32 - 0.25));
            }
            GgmlType::Q3K => {
                raw.extend([0xa5; 32]);
                raw.extend([0xe4; 64]);
                raw.extend([0x33; 8]);
                raw.extend([0xaa; 4]);
                raw.extend(half(d));
                reference.extend((0..256).map(|i| {
                    d * 3.
                        * (((i / 32) % 4) as f32
                            - if [1, 0, 1, 0, 0, 1, 0, 1][i / 32] == 0 {
                                4.
                            } else {
                                0.
                            })
                }));
            }
            GgmlType::MxFp4 => {
                raw.push(127 + (row % 3) as u8);
                raw.extend((0..16).map(|i| ((i + row) % 16) as u8 * 17));
                let values = [
                    0., 0.5, 1., 1.5, 2., 3., 4., 6., 0., -0.5, -1., -1.5, -2., -3., -4., -6.,
                ];
                reference.extend(
                    (0..32).map(|i| values[((i + row) % 16) as usize] * (1 << (row % 3)) as f32),
                );
            }
            _ => unreachable!(),
        }
    }
    let mut config = Config::from_json(
        &serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../eredu-architectures/src/qwen4_exp/config/released.json"
        )))
        .unwrap(),
    )
    .unwrap();
    config.ngram.heads = 1;
    config.ngram.embedding_dim = width as i32 * 2;
    if let eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors {
        shards,
        vocabulary_alignment,
        ..
    } = &mut config.ngram.source
    {
        *shards = 2;
        *vocabulary_alignment = 4;
    }
    let mut metadata = BTreeMap::from([
        ("general.architecture".into(), V::String("qwen4exp".into())),
        ("qwen4exp.ple.layers".into(), V::Array(A::Int32(vec![1]))),
        (
            "qwen4exp.ple.layer_multipliers".into(),
            V::Array(A::Uint64(vec![
                23703573157769,
                20109073645365,
                8052911324071,
            ])),
        ),
        (
            "qwen4exp.ple.head_offsets".into(),
            V::Array(A::Uint64(vec![0, 11])),
        ),
        (
            "qwen4exp.ple.head_vocab_sizes".into(),
            V::Array(A::Uint64(vec![11, 13])),
        ),
    ]);
    for (suffix, value) in [
        ("ple.ngram_size", 3),
        ("ple.heads_per_ngram", 1),
        ("ple.conv_kernel", config.ngram.kernel as u32),
        ("ple.eos_token_id", config.eos[0]),
        ("embedding_length_per_layer_input", width as u32),
    ] {
        metadata.insert(format!("qwen4exp.{suffix}"), V::Uint32(value));
    }
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("table.gguf");
    eredu_gguf::Writer::new(eredu_gguf::WriterOptions {
        endian,
        ..Default::default()
    })
    .unwrap()
    .write(
        std::fs::File::create(&path).unwrap(),
        &metadata,
        &[eredu_gguf::TensorInput {
            name: "per_layer_token_embd.weight",
            dimensions: &[width, 24],
            ggml_type: ty,
            data: &raw,
        }],
    )
    .unwrap();
    let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
    let table_plan = GgufTableSourcePlan::prepare(&checkpoint, &config).unwrap();
    let contract = eredu_checkpoint::schema::GgufCheckpointPlan::new(
        "native table fixture",
        vec![table_plan.constraint().clone()],
        vec![],
        eredu_checkpoint::schema::CatalogPolicy::non_strict(),
    )
    .unwrap();
    let mapping = table_plan
        .checkpoint()
        .translated_outputs(str::to_owned)
        .unwrap();
    let source = Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .max_cached_readers(1)
            .unwrap()
            .add_checkpoint(table_plan.checkpoint().clone(), &contract, &mapping)
            .unwrap()
            .build()
            .unwrap(),
    );
    let table = table_plan
        .bind(source, 1, 2, 7, TensorElementType::F32)
        .unwrap();
    drop(checkpoint);
    let source: SharedCheckpointSource = table.rows.clone();
    let range = OffloadUnitRange::new(
        OffloadUnitId::new("prepared.rows").unwrap(),
        0,
        24,
        row_bytes,
        ResidencyPolicy::Cacheable,
    )
    .unwrap();
    let rows = RowResidencyRange::new(range.clone(), table.rows.clone(), "value").unwrap();
    let plan = OffloadPlan::with_ranges(
        OffloadConfig::new(Some(2 * row_bytes), Some(131072), 1).unwrap(),
        [],
        [range.clone()],
    )
    .unwrap();
    let manager = ResidencyManager::new_shared_row_ranges(
        source.clone(),
        BTreeMap::new(),
        plan,
        [],
        vec![rows.clone()],
        stream.clone(),
        stream.clone(),
    )
    .unwrap();
    manager.initialize().unwrap();
    let prepared =
        PreparedRowLookup::new(rows, table.lookup.clone(), table.scale, limits()).unwrap();
    let workspace = eredu_runtime::RowLookupMechanismSupport::workspace(
        &MlxRowLookupSupport,
        prepared.descriptor(),
    )
    .unwrap()
    .unwrap();
    assert!(MlxRowBank::new(manager.clone(), &prepared, None, workspace.decode_bytes - 1).is_err());
    if ty != GgmlType::Q8_0 {
        assert_eq!(workspace.decode_bytes, 2 * (row_bytes + 24 * width + 4));
        let facts = eredu_runtime::RowLookupMechanismSupport::decode_memory(
            &MlxRowLookupSupport,
            prepared.descriptor(),
        )
        .unwrap();
        let host = facts
            .storage
            .iter()
            .find(|s| s.name == "decoded_host")
            .unwrap();
        assert_eq!(
            host.placement,
            eredu_nn::mechanism_memory::MechanismPlacement::Host
        );
        assert_eq!(host.payload.upper, Some(2 * width * 4));
    }
    let bank = MlxRowBank::new(manager.clone(), &prepared, None, workspace.decode_bytes).unwrap();
    let mut provider = prepared.bind(bank).unwrap();
    assert_eq!(source.source_diagnostics().unwrap().physical_read_bytes, 0);
    let expected = |ids: &[i32]| {
        ids.iter()
            .flat_map(|&row| {
                reference[row as usize * width as usize..(row as usize + 1) * width as usize]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>()
    };
    let cold = std::time::Instant::now();
    let output = provider
        .lookup_rows(
            &table.lookup,
            &[12, 11, 12],
            ParameterBankAccess::Bulk,
            &stream,
        )
        .unwrap();
    eval([output.as_array()]).unwrap();
    let cold_elapsed = cold.elapsed();
    assert_eq!(
        output.as_array().evaluated().unwrap().as_slice::<f32>(),
        expected(&[12, 11, 12])
    );
    let before = source.source_diagnostics().unwrap();
    assert_eq!(before.physical_read_bytes, 2 * row_bytes);
    assert_eq!(before.physical_reads, 1);
    let warm = std::time::Instant::now();
    for _ in 0..32 {
        let result = provider
            .lookup_rows(
                &table.lookup,
                &[11, 12, 11],
                ParameterBankAccess::Incremental,
                &stream,
            )
            .unwrap();
        eval([result.as_array()]).unwrap();
        assert_eq!(
            result.as_array().evaluated().unwrap().as_slice::<f32>(),
            expected(&[11, 12, 11])
        );
        let report = provider.bank().report().unwrap();
        assert!(report.units().len() <= 2);
        assert!(report.offload().resident_bytes().get(MemoryTier::Device) <= 2 * row_bytes);
    }
    let warm_elapsed = warm.elapsed();
    assert_eq!(
        source.source_diagnostics().unwrap().physical_read_bytes,
        2 * row_bytes
    );
    for row in [11, 12] {
        assert!(manager
            .evict(
                &OffloadUnitId::new(format!("prepared.rows.{row}")).unwrap(),
                MemoryTier::Device
            )
            .unwrap());
    }
    // Returned output owns its storage after every encoded source row is evicted.
    eval([output.as_array()]).unwrap();
    assert_eq!(
        output.as_array().evaluated().unwrap().as_slice::<f32>(),
        expected(&[12, 11, 12])
    );
    let again = provider
        .lookup_rows(
            &table.lookup,
            &[12, 11, 12],
            ParameterBankAccess::Bulk,
            &stream,
        )
        .unwrap();
    eval([again.as_array()]).unwrap();
    assert_eq!(
        again.as_array().evaluated().unwrap().as_slice::<f32>(),
        expected(&[12, 11, 12])
    );
    let reads = source.source_diagnostics().unwrap();
    assert_eq!(reads.physical_read_bytes, 4 * row_bytes);
    assert_eq!(reads.physical_reads, 2);
    eprintln!("architecture-prepared GGUF {ty:?}/{endian:?}: cold {cold_elapsed:?}, warm mean {:?}, {}-byte cold read, zero warm I/O, cache <= {} bytes", warm_elapsed / 32, 2 * row_bytes, 2 * row_bytes);
}
