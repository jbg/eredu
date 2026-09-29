use super::*;
use crate::qwen4_exp::checkpoint::{GgufTableSourcePlan, SafetensorsTableSourcePlan};
use eredu_checkpoint::{
    rows::RowReadLimits,
    store::{CheckpointLease, EncodedTensorLease},
};
use eredu_gguf::{
    CheckpointHeader, Endian, GgmlType, Limits, Reader, TensorInput, Writer, WriterOptions,
};
use std::io::Cursor;

// Literal output of the pinned converter's released configuration. These values
// are independent of the parser's metadata-to-config translation.
fn metadata() -> BTreeMap<String, MetadataValue> {
    let mut m = BTreeMap::from([(
        "general.architecture".into(),
        MetadataValue::String("qwen4exp".into()),
    )]);
    for (key, v) in [
        ("block_count", 48),
        ("context_length", 262144),
        ("embedding_length", 2560),
        ("attention.head_count", 24),
        ("attention.head_count_kv", 2),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("rope.dimension_count", 64),
        ("full_attention_interval", 4),
        ("ssm.conv_kernel", 4),
        ("ssm.inner_size", 6144),
        ("ssm.state_size", 128),
        ("ssm.time_step_rank", 48),
        ("ssm.group_count", 16),
        ("hyper_connection.count", 4),
        ("hyper_connection.low_rank", 320),
        ("attention.indexer.head_count", 4),
        ("attention.indexer.key_length", 128),
        ("attention.indexer.top_k", 2048),
        ("expert_count", 512),
        ("expert_used_count", 10),
        ("expert_feed_forward_length", 640),
        ("expert_shared_feed_forward_length", 640),
        ("ple.ngram_size", 3),
        ("ple.heads_per_ngram", 8),
        ("ple.conv_kernel", 4),
        ("ple.eos_token_id", 248044),
        ("ple.image_token_id", 248056),
    ] {
        m.insert(format!("qwen4exp.{key}"), MetadataValue::Uint32(v));
    }
    for (key, v) in [
        ("rope.freq_base", 10000000.),
        ("attention.layer_norm_rms_epsilon", 1e-6),
    ] {
        m.insert(format!("qwen4exp.{key}"), MetadataValue::Float32(v));
    }
    m.insert(
        "qwen4exp.attention.recurrent_layers".into(),
        MetadataValue::Array(MetadataArray::Bool((0..48).map(|i| i % 4 != 3).collect())),
    );
    m.insert(
        "qwen4exp.attention.compress_ratios".into(),
        MetadataValue::Array(MetadataArray::Uint32(
            (0..48).map(|i| if i % 4 == 3 { 4 } else { 0 }).collect(),
        )),
    );
    for (key, values) in [
        ("rope.dimension_sections", vec![11, 11, 10, 0]),
        ("ple.layers", vec![1]),
    ] {
        m.insert(
            format!("qwen4exp.{key}"),
            MetadataValue::Array(MetadataArray::Uint32(values)),
        );
    }
    // Official integer fixtures contain independently range-read payload bytes.
    let catalog: Value =
        serde_json::from_str(include_str!("../../checkpoint/fixtures/bf16.json")).unwrap();
    // Controls are filled below from the compact source fixture, not generated
    // from vocabulary size/seed or rounded through floating metadata.
    for (key, suffix) in [
        ("ple.layer_multipliers", "layer_multipliers"),
        ("ple.head_vocab_sizes", "ngram_heads_vocab_sizes"),
        ("ple.head_offsets", "ngram_heads_offsets"),
    ] {
        let constant = catalog["constants"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["tensor"].as_str().unwrap().ends_with(suffix))
            .unwrap();
        let hex = constant["bytes"].as_str().unwrap();
        let bytes: Vec<_> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let values = bytes
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
            .collect();
        m.insert(
            format!("qwen4exp.{key}"),
            MetadataValue::Array(MetadataArray::Int64(values)),
        );
    }
    m
}

// Writes metadata using the container writer, then only descriptors. The 100 GB
// table is declared, never allocated or written; from_headers forbids payload use.
fn header(
    m: &BTreeMap<String, MetadataValue>,
    endian: Endian,
    shapes: &[(&str, Vec<u64>)],
) -> Checkpoint {
    let mut cursor = Cursor::new(vec![]);
    Writer::new(WriterOptions {
        endian,
        ..Default::default()
    })
    .unwrap()
    .write(&mut cursor, m, &[])
    .unwrap();
    cursor.set_position(0);
    drop(Reader::new(&mut cursor).unwrap());
    let end = cursor.position() as usize;
    let mut bytes = cursor.into_inner();
    bytes.truncate(end);
    let u64_bytes = |v: u64| {
        if endian == Endian::Little {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        }
    };
    let u32_bytes = |v: u32| {
        if endian == Endian::Little {
            v.to_le_bytes()
        } else {
            v.to_be_bytes()
        }
    };
    bytes[8..16].copy_from_slice(&u64_bytes(shapes.len() as u64));
    let mut offset = 0u64;
    for (name, shape) in shapes {
        offset = offset.div_ceil(32) * 32;
        bytes.extend(u64_bytes(name.len() as u64));
        bytes.extend(name.as_bytes());
        bytes.extend(u32_bytes(shape.len() as u32));
        for &d in shape.iter().rev() {
            bytes.extend(u64_bytes(d));
        }
        bytes.extend(u32_bytes(GgmlType::Bf16.code()));
        bytes.extend(u64_bytes(offset));
        offset += shape.iter().product::<u64>() * 2;
    }
    let file_len = (bytes.len() as u64).div_ceil(32) * 32 + offset;
    Checkpoint::from_headers(
        &[CheckpointHeader {
            member: "model.gguf".into(),
            bytes,
            file_len,
        }],
        Limits::default(),
    )
    .unwrap()
}
fn shapes() -> Vec<(&'static str, Vec<u64>)> {
    vec![
        ("token_embd.weight", vec![248320, 2560]),
        ("output.weight", vec![248320, 2560]),
        ("per_layer_token_embd.weight", vec![320001536, 160]),
    ]
}

#[test]
fn released_gguf_headers_normalize_without_sidecars_or_payloads() {
    let expected = Config::from_json(
        &serde_json::from_str::<Value>(include_str!("../released.json")).unwrap(),
    )
    .unwrap();
    for endian in [Endian::Little, Endian::Big] {
        let catalog = header(&metadata(), endian, &shapes());
        let actual = Config::from_gguf(&catalog).unwrap();
        assert_eq!(actual.layers, expected.layers);
        assert_eq!(
            (actual.hidden_size, actual.vocabulary, actual.max_positions),
            (2560, 248320, 262144)
        );
        assert_eq!(
            (
                actual.attention.heads,
                actual.attention.kv_heads,
                actual.attention.head_dim
            ),
            (24, 2, 256)
        );
        assert_eq!(
            (
                actual.attention.rotary_dim,
                actual.attention.budget,
                actual.attention.ratio
            ),
            (64, 2048, 4)
        );
        assert_eq!(actual.attention.rotary, expected.attention.rotary);
        assert_eq!(
            actual.attention.rope["mrope_section"],
            expected.attention.rope["mrope_section"]
        );
        assert_eq!(actual.attention.rope["mrope_interleaved"], json!(true));
        assert_eq!((actual.residual.streams, actual.residual.rank), (4, 320));
        assert_eq!(
            (
                actual.recurrent.key_heads,
                actual.recurrent.value_heads,
                actual.recurrent.key_dim,
                actual.recurrent.value_dim
            ),
            (16, 48, 128, 128)
        );
        assert_eq!(actual.recurrent.gate, OutputGateActivation::Sigmoid);
        assert_eq!(
            (
                actual.experts.count,
                actual.experts.selected,
                actual.experts.intermediate,
                actual.experts.shared_intermediate
            ),
            (512, 10, 640, 640)
        );
        assert_eq!(actual.ngram.layers, [1]);
        assert_eq!(
            (
                actual.ngram.order,
                actual.ngram.heads,
                actual.ngram.embedding_dim,
                actual.ngram.kernel
            ),
            (3, 8, 2560, 4)
        );
        assert_eq!(
            actual.ngram.source,
            NGramSourceLayout::Gguf { rows: 320001536 }
        );
        assert_eq!(actual.eos, [248044]);
        assert!(!actual.tied_embeddings);
        assert!(actual.prediction.is_none() && actual.vision.is_none() && actual.media.is_none());
        assert!(expected.prediction.is_some() && expected.vision.is_some());
    }
}

#[test]
fn explicit_schedule_ties_and_reset_semantics_survive_metadata_normalization() {
    let mut m = metadata();
    // Explicit nonperiodic schedule takes precedence over the retained interval.
    let mut schedule = vec![true; 48];
    schedule[2] = false;
    schedule[47] = false;
    m.insert(
        "qwen4exp.attention.recurrent_layers".into(),
        MetadataValue::Array(MetadataArray::Bool(schedule.clone())),
    );
    m.insert(
        "qwen4exp.attention.compress_ratios".into(),
        MetadataValue::Array(MetadataArray::Uint32(
            schedule.iter().map(|r| if *r { 0 } else { 4 }).collect(),
        )),
    );
    m.insert(
        "tokenizer.ggml.eos_token_id".into(),
        MetadataValue::Uint32(248045),
    );
    let mut tensors = shapes();
    tensors.remove(1);
    let config = Config::from_gguf(&header(&m, Endian::Little, &tensors)).unwrap();
    assert!(config.tied_embeddings);
    assert_eq!(config.eos[0], 248044); // generation termination must not replace hash reset
    assert_eq!(config.layers.get(2), Some(&LayerKind::Indexed));
    assert_eq!(config.layers.get(3), Some(&LayerKind::Recurrent));
    m = metadata();
    m.remove("qwen4exp.attention.recurrent_layers");
    assert_eq!(
        Config::from_gguf(&header(&m, Endian::Big, &shapes()))
            .unwrap()
            .layers
            .get(3),
        Some(&LayerKind::Indexed)
    );
    let mut tensors = shapes();
    tensors.push(("unrelated.attn_q.bias", vec![1]));
    assert!(
        !Config::from_gguf(&header(&metadata(), Endian::Little, &tensors))
            .unwrap()
            .attention
            .bias
    );
    tensors.push(("blk.3.attn_q.bias", vec![12288]));
    assert!(
        Config::from_gguf(&header(&metadata(), Endian::Little, &tensors))
            .unwrap()
            .attention
            .bias
    );
}

#[test]
fn malformed_or_unrepresented_metadata_is_rejected_before_preparation() {
    let base = metadata();
    for (key, value) in [
        ("attention.causal", MetadataValue::Bool(false)),
        ("expert_gating_func", MetadataValue::Uint32(2)),
        ("block_count", MetadataValue::Uint32(4097)),
        ("block_count", MetadataValue::Float32(48.)),
        ("ssm.inner_size", MetadataValue::Uint32(6145)),
        ("attention.head_count_kv", MetadataValue::Uint32(5)),
        ("attention.value_length", MetadataValue::Uint32(128)),
        (
            "attention.recurrent_layers",
            MetadataValue::Array(MetadataArray::Bool(vec![true])),
        ),
        (
            "attention.compress_ratios",
            MetadataValue::Array(MetadataArray::Uint32(vec![4; 48])),
        ),
        (
            "rope.dimension_sections",
            MetadataValue::Array(MetadataArray::Uint32(vec![11, 11, 10, 1])),
        ),
        ("rope.freq_base", MetadataValue::Float64(1e100)),
        (
            "ple.layers",
            MetadataValue::Array(MetadataArray::Uint32(vec![3])),
        ),
        (
            "ple.layer_multipliers",
            MetadataValue::Array(MetadataArray::Float64(vec![1.; 3])),
        ),
        (
            "ple.head_offsets",
            MetadataValue::Array(MetadataArray::Uint64(vec![u64::MAX; 16])),
        ),
        ("ple.image_token_id", MetadataValue::Uint32(248320)),
        ("vocab_size", MetadataValue::Uint32(248319)),
        ("nextn_predict_layers", MetadataValue::Uint32(1)),
        ("rope.scaling.type", MetadataValue::String("yarn".into())),
    ] {
        let mut m = base.clone();
        m.insert(format!("qwen4exp.{key}"), value);
        assert!(
            Config::from_gguf(&header(&m, Endian::Little, &shapes())).is_err(),
            "accepted {key}"
        );
    }
    for key in [
        "hyper_connection.low_rank",
        "attention.indexer.top_k",
        "ple.head_vocab_sizes",
        "ssm.group_count",
    ] {
        let mut m = base.clone();
        m.remove(&format!("qwen4exp.{key}"));
        assert!(
            Config::from_gguf(&header(&m, Endian::Little, &shapes())).is_err(),
            "missing {key}"
        );
    }
    for tensors in [
        vec![
            ("token_embd.weight", vec![248320, 2559]),
            ("per_layer_token_embd.weight", vec![320001536, 160]),
        ],
        vec![
            ("token_embd.weight", vec![248320, 2560]),
            ("output.weight", vec![248319, 2560]),
            ("per_layer_token_embd.weight", vec![320001536, 160]),
        ],
        vec![
            ("token_embd.weight", vec![248320, 2560]),
            ("per_layer_token_embd.weight", vec![319999999, 160]),
        ],
    ] {
        assert!(Config::from_gguf(&header(&base, Endian::Little, &tensors)).is_err());
    }
    let mut differing = base.clone();
    let MetadataValue::Array(MetadataArray::Uint32(ratios)) = differing
        .get_mut("qwen4exp.attention.compress_ratios")
        .unwrap()
    else {
        unreachable!()
    };
    ratios[7] = 2;
    assert!(Config::from_gguf(&header(&differing, Endian::Little, &shapes())).is_err());
}

#[test]
fn standalone_gguf_config_prepares_nonzero_rows_and_preserves_literal_padding() {
    use eredu_checkpoint::{
        recipe::DerivedWeightRecipe,
        store::{CheckpointSource, ReadPolicy, TensorReadRequest, TensorSelection},
    };
    let mut m = metadata();
    for (key, v) in [
        ("embedding_length", 32),
        ("ple.heads_per_ngram", 1),
        ("ple.eos_token_id", 7),
        ("ple.image_token_id", 8),
    ] {
        m.insert(format!("qwen4exp.{key}"), MetadataValue::Uint32(v));
    }
    for (key, values) in [
        (
            "ple.layer_multipliers",
            vec![23703573157769, 9007199254740993, 8052911324071],
        ),
        ("ple.head_vocab_sizes", vec![11, 13]),
        ("ple.head_offsets", vec![0, 11]),
    ] {
        m.insert(
            format!("qwen4exp.{key}"),
            MetadataValue::Array(MetadataArray::Int64(values)),
        );
    }
    m.insert(
        "tokenizer.ggml.eos_token_id".into(),
        MetadataValue::Uint32(9),
    );
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tiny.gguf");
    let embedding = vec![0; 64 * 32 * 4];
    let table: Vec<_> = (0..27)
        .flat_map(|r| (0..32).flat_map(move |c| ((r * 32 + c) as f32 * 0.125 - 4.).to_le_bytes()))
        .collect();
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &m,
            &[
                TensorInput {
                    name: "token_embd.weight",
                    dimensions: &[32, 64],
                    ggml_type: GgmlType::F32,
                    data: &embedding,
                },
                TensorInput {
                    name: "per_layer_token_embd.weight",
                    dimensions: &[32, 27],
                    ggml_type: GgmlType::F32,
                    data: &table,
                },
            ],
        )
        .unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    let config = Config::from_gguf(&checkpoint).unwrap();
    assert_eq!(config.ngram.source, NGramSourceLayout::Gguf { rows: 27 });
    assert_eq!(config.eos, [7]);
    let table_plan = GgufTableSourcePlan::prepare(&checkpoint, &config).unwrap();
    let contract = eredu_checkpoint::schema::GgufCheckpointPlan::new(
        "table fixture",
        vec![table_plan.constraint().clone()],
        vec![],
        eredu_checkpoint::schema::CatalogPolicy::non_strict(),
    )
    .unwrap();
    let mapping = table_plan
        .checkpoint()
        .translated_outputs(str::to_owned)
        .unwrap();
    let source = std::sync::Arc::new(
        eredu_checkpoint::gguf_store::GgufWeightStore::builder()
            .add_checkpoint(table_plan.checkpoint().clone(), &contract, &mapping)
            .unwrap()
            .build()
            .unwrap(),
    );
    let prepared = table_plan
        .bind(source, 1, 0, 1, eredu_nn::TensorElementType::F32)
        .unwrap();
    assert_eq!(prepared.lookup.dimensions, 32);
    assert_eq!(
        prepared
            .rows
            .source_diagnostics()
            .unwrap()
            .physical_read_bytes,
        0
    );
    let limits = RowReadLimits {
        requests: 4,
        rows_per_read: 2,
    };
    let plan = prepared.rows.plan(&[23, 11, 12, 11], limits).unwrap();
    assert_eq!(plan.restore_order(), [2, 0, 1, 0]);
    let mut decoded = vec![];
    for read in plan.reads() {
        let DerivedWeightRecipe::Source { key, selection } = read.recipe() else {
            panic!("physical range")
        };
        assert!(matches!(selection, TensorSelection::Range { .. }));
        let lease = plan
            .source()
            .acquire_lease(TensorReadRequest {
                key: key.clone(),
                selection: selection.clone(),
                policy: ReadPolicy::RequireBounded,
            })
            .unwrap();
        assert!(lease.bounded_read_proof().physically_bounded);
        let CheckpointLease::Gguf(lease) = lease else {
            panic!("GGUF lease")
        };
        let eredu_gguf::ConvertedTensor::Dense(values) =
            lease.materialize_portable().unwrap().into_converted()
        else {
            panic!("dense rows")
        };
        decoded.extend(
            values
                .data
                .chunks_exact(4)
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap())),
        );
    }
    for (slot, row) in [11, 12, 23].into_iter().enumerate() {
        for column in 0..32 {
            assert_eq!(
                decoded[slot * 32 + column],
                (row * 32 + column) as f32 * 0.125 - 4.
            );
        }
    }
    assert_eq!(
        prepared
            .rows
            .source_diagnostics()
            .unwrap()
            .physical_read_bytes,
        384
    );
    let table_source: &dyn eredu_checkpoint::store::CheckpointSource = prepared.rows.as_ref();
    assert!(SafetensorsTableSourcePlan::prepare(table_source, &config, 1).is_err());
}
