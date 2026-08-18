#![cfg(unix)]

use std::{
    collections::{BTreeMap, HashMap},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    module::ModuleParameters,
    ops::{indexing::TryIndexOp, GgufMetadataArray, GgufMetadataValue},
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_gguf::{GgmlType, TensorInput, Writer};
use safemlx_lm::{
    architectures::{
        distributed::pipeline::{load_pipeline_model_with_options, PipelineStep},
        gemma4::model::{self as gemma4, Cache, Model},
    },
    core::residency::OffloadConfig,
    nn::generation::CausalLm,
    runtime::media::{
        input::{InputMetadata, InputPart, ModelInput},
        PreparedModelInput,
    },
    DenseDiskStreamLoadOptions, DeviceAssignment, LayerwiseLoadOptions, MlxBackend,
    MlxParallelContext, ModelLoadOptions, PagedCacheOptions, PromptCacheDescriptor,
    PromptCacheOptions, PromptCacheTopology, WeightResidency,
};

const WORKER: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_WORKER";
const CHECKPOINT: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_CHECKPOINT";
const TENSOR_PARALLEL: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_TP";
const DENSE_STREAM: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_STREAM";
const LAYERWISE_HOST: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_HOST";
const AXES: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_AXES";
const CACHE_ROOT: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_CACHE_ROOT";

fn config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "gemma4",
        "tie_word_embeddings": false,
        "image_token_id": 20,
        "video_token_id": 21,
        "audio_token_id": 22,
        "text_config": {
            "model_type": "gemma4",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": 16,
            "num_attention_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "pad_token_id": 0,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "head_dim": 4,
            "attention_bias": false,
            "hidden_size_per_layer_input": 4,
            "vocab_size_per_layer_input": 32,
            "num_kv_shared_layers": 0,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 8,
            "final_logit_softcapping": 4.0
        },
        "vision_config": {
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "position_embedding_size": 4,
            "rms_norm_eps": 1e-6,
            "hidden_activation": "gelu_pytorch_tanh",
            "standardize": false
        },
        "audio_config": {
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "output_proj_dims": 8,
            "conv_kernel_size": 3,
            "attention_chunk_size": 4,
            "attention_context_left": 4,
            "attention_context_right": 0,
            "attention_invalid_logits_value": -1.0e9,
            "attention_logit_cap": 10.0,
            "residual_weight": 0.5,
            "rms_norm_eps": 1e-6,
            "subsampling_conv_channels": [2, 2]
        }
    })
}

fn moe_config() -> serde_json::Value {
    let mut config = config();
    let text = config["text_config"].as_object_mut().unwrap();
    text.insert("enable_moe_block".into(), serde_json::json!(true));
    text.insert("num_experts".into(), serde_json::json!(4));
    text.insert("top_k_experts".into(), serde_json::json!(2));
    text.insert("moe_intermediate_size".into(), serde_json::json!(8));
    config
}

fn write_fixture(directory: &Path) {
    write_safetensors_fixture(directory, config());
}

fn write_moe_fixture(directory: &Path) {
    write_safetensors_fixture(directory, moe_config());
}

fn write_safetensors_fixture(directory: &Path, config: serde_json::Value) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut model = Model::new_from_config_value(&config, stream).unwrap();
    let mut names = model
        .parameters()
        .flatten()
        .keys()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    names.sort();
    let mut parameters = model.parameters_mut().flatten();
    for (ordinal, name) in names.iter().enumerate() {
        let parameter = parameters.get_mut(name.as_str()).unwrap();
        let shape = parameter.shape().to_vec();
        **parameter = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else if name.ends_with(".bias") {
            Array::zeros::<f32>(&shape, stream).unwrap()
        } else {
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.0002 * (ordinal + 1) as f32),
                stream,
            )
            .unwrap()
        };
    }
    let arrays = model
        .parameters()
        .flatten()
        .iter()
        .map(|(name, value)| {
            let canonical = safemlx_lm::runtime::checkpoint::binding::canonical_checkpoint_name(
                name,
            )
            .replacen("model.language_model.", "language_model.model.", 1);
            (canonical, *value)
        })
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), *value)),
        None,
        directory.join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec_pretty(&config).unwrap(),
    )
    .unwrap();
}

struct OwnedGgufTensor {
    name: String,
    dimensions: Vec<u64>,
    data: Vec<u8>,
}

fn write_dense_gguf(
    path: &Path,
    arrays: &HashMap<String, Array>,
    metadata: HashMap<String, GgufMetadataValue>,
) {
    let mut names = arrays.keys().collect::<Vec<_>>();
    names.sort_unstable();
    let tensors = names
        .into_iter()
        .map(|name| {
            let evaluated = arrays[name].evaluated().unwrap();
            OwnedGgufTensor {
                name: name.clone(),
                dimensions: evaluated
                    .as_array()
                    .shape()
                    .iter()
                    .rev()
                    .map(|&dimension| u64::try_from(dimension).unwrap())
                    .collect(),
                data: evaluated
                    .as_slice::<f32>()
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let inputs = tensors
        .iter()
        .map(|tensor| TensorInput {
            name: &tensor.name,
            dimensions: &tensor.dimensions,
            ggml_type: GgmlType::F32,
            data: &tensor.data,
        })
        .collect::<Vec<_>>();
    Writer::default()
        .write(
            std::fs::File::create(path).unwrap(),
            &metadata.into_iter().collect::<BTreeMap<_, _>>(),
            &inputs,
        )
        .unwrap();
}

fn text_gguf_name(name: &str) -> String {
    for (target, source) in [
        (
            "model.language_model.embed_tokens_per_layer",
            "per_layer_token_embd",
        ),
        (
            "model.language_model.per_layer_model_projection",
            "per_layer_model_proj",
        ),
        (
            "model.language_model.per_layer_projection_norm",
            "per_layer_proj_norm",
        ),
        ("model.language_model.embed_tokens", "token_embd"),
        ("model.language_model.norm", "output_norm"),
        ("lm_head", "output"),
    ] {
        if name == target || name.starts_with(&format!("{target}.")) {
            return name.replacen(target, source, 1);
        }
    }
    let rest = name
        .strip_prefix("model.language_model.layers.")
        .unwrap_or_else(|| panic!("unmapped Gemma 4 text tensor {name}"));
    let (layer, parameter) = rest.split_once('.').unwrap();
    if parameter == "layer_scalar" {
        return format!("blk.{layer}.layer_output_scale.weight");
    }
    if parameter == "router.per_expert_scale" {
        return format!("blk.{layer}.ffn_down_exps.scale");
    }
    if parameter == "router.scale" {
        return format!("blk.{layer}.ffn_gate_inp.scale");
    }
    for (target, source) in [
        ("self_attn.q_norm", "attn_q_norm"),
        ("self_attn.k_norm", "attn_k_norm"),
        ("self_attn.q_proj", "attn_q"),
        ("self_attn.k_proj", "attn_k"),
        ("self_attn.v_proj", "attn_v"),
        ("self_attn.o_proj", "attn_output"),
        ("input_layernorm", "attn_norm"),
        ("post_attention_layernorm", "post_attention_norm"),
        ("pre_feedforward_layernorm", "ffn_norm"),
        ("post_feedforward_layernorm", "post_ffw_norm"),
        ("mlp.gate_proj", "ffn_gate"),
        ("mlp.down_proj", "ffn_down"),
        ("mlp.up_proj", "ffn_up"),
        ("router.proj", "ffn_gate_inp"),
        ("experts.switch_glu.gate_proj", "ffn_gate_exps"),
        ("experts.switch_glu.up_proj", "ffn_up_exps"),
        ("experts.switch_glu.down_proj", "ffn_down_exps"),
        ("pre_feedforward_layernorm_2", "pre_ffw_norm_2"),
        ("post_feedforward_layernorm_1", "post_ffw_norm_1"),
        ("post_feedforward_layernorm_2", "post_ffw_norm_2"),
        ("per_layer_input_gate", "inp_gate"),
        ("per_layer_projection", "proj"),
        ("post_per_layer_input_norm", "post_norm"),
    ] {
        if parameter == target || parameter.starts_with(&format!("{target}.")) {
            return format!("blk.{layer}.{}", parameter.replacen(target, source, 1));
        }
    }
    panic!("unmapped Gemma 4 text layer tensor {name}")
}

fn projector_metadata() -> HashMap<String, GgufMetadataValue> {
    HashMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("clip".into()),
        ),
        (
            "clip.has_vision_encoder".into(),
            GgufMetadataValue::Bool(true),
        ),
        (
            "clip.has_audio_encoder".into(),
            GgufMetadataValue::Bool(true),
        ),
        (
            "clip.vision.projector_type".into(),
            GgufMetadataValue::String("gemma4".into()),
        ),
        (
            "clip.audio.projector_type".into(),
            GgufMetadataValue::String("gemma4".into()),
        ),
        (
            "clip.vision.embedding_length".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "clip.vision.feed_forward_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "clip.vision.block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "clip.vision.attention.head_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.attention.key_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "clip.vision.patch_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.pooling_kernel_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.position_embedding_size".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "clip.vision.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-6),
        ),
        (
            "clip.vision.hidden_activation".into(),
            GgufMetadataValue::String("gelu_pytorch_tanh".into()),
        ),
        (
            "clip.vision.standardize".into(),
            GgufMetadataValue::Bool(false),
        ),
        (
            "clip.vision.rope.freq_base".into(),
            GgufMetadataValue::Float32(100.0),
        ),
        (
            "clip.audio.embedding_length".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "clip.audio.block_count".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "clip.audio.attention.head_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.audio.projection_dim".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "clip.audio.conv_kernel_size".into(),
            GgufMetadataValue::Uint32(3),
        ),
        (
            "clip.audio.attention.chunk_size".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "clip.audio.attention.context_left".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "clip.audio.attention.context_right".into(),
            GgufMetadataValue::Uint32(0),
        ),
        (
            "clip.audio.attention.invalid_logits_value".into(),
            GgufMetadataValue::Float32(-1.0e9),
        ),
        (
            "clip.audio.attention.logit_cap".into(),
            GgufMetadataValue::Float32(10.0),
        ),
        (
            "clip.audio.residual_weight".into(),
            GgufMetadataValue::Float32(0.5),
        ),
        (
            "clip.audio.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-6),
        ),
        (
            "clip.audio.subsampling_conv_channels".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![2, 2])),
        ),
    ])
}

fn write_gguf_fixture(directory: &Path) -> PathBuf {
    write_gguf_fixture_kind(directory, false)
}

fn write_moe_gguf_fixture(directory: &Path) -> PathBuf {
    write_gguf_fixture_kind(directory, true)
}

fn write_gguf_fixture_kind(directory: &Path, moe: bool) -> PathBuf {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let source_config = if moe { moe_config() } else { config() };
    let mut model = Model::new_from_config_value(&source_config, stream).unwrap();
    let mut names = model
        .parameters()
        .flatten()
        .keys()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    names.sort();
    let mut parameters = model.parameters_mut().flatten();
    for (ordinal, name) in names.iter().enumerate() {
        let parameter = parameters.get_mut(name.as_str()).unwrap();
        let shape = parameter.shape().to_vec();
        **parameter = if name.ends_with("norm.weight") || name.ends_with("layernorm.weight") {
            Array::ones::<f32>(&shape, stream).unwrap()
        } else if name.ends_with(".bias") {
            Array::zeros::<f32>(&shape, stream).unwrap()
        } else {
            Array::full::<f32>(
                &shape,
                Array::from_f32(0.0002 * (ordinal + 1) as f32),
                stream,
            )
            .unwrap()
        };
    }
    let mut text = HashMap::new();
    let mut projector = HashMap::new();
    for (name, value) in model.parameters().flatten() {
        let name = safemlx_lm::runtime::checkpoint::binding::canonical_checkpoint_name(&name);
        if name.starts_with("model.language_model.") || name.starts_with("lm_head.") {
            text.insert(text_gguf_name(&name), value.clone());
        } else {
            let physical = name.strip_prefix("model.").unwrap_or(&name).to_string();
            projector.insert(physical, value.clone());
        }
    }
    let mut text_metadata = HashMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("gemma4".into()),
        ),
        ("gemma4.block_count".into(), GgufMetadataValue::Uint32(2)),
        (
            "gemma4.embedding_length".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "gemma4.feed_forward_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "gemma4.attention.head_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "gemma4.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "gemma4.attention.key_length".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "gemma4.attention.key_length_swa".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "gemma4.attention.sliding_window_pattern".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true, false])),
        ),
        (
            "gemma4.attention.shared_kv_layers".into(),
            GgufMetadataValue::Uint32(0),
        ),
        (
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-6),
        ),
        (
            "gemma4.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "gemma4.context_length".into(),
            GgufMetadataValue::Uint32(128),
        ),
        ("gemma4.vocab_size".into(), GgufMetadataValue::Uint32(32)),
        (
            "gemma4.embedding_length_per_layer_input".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "gemma4.final_logit_softcapping".into(),
            GgufMetadataValue::Float32(4.0),
        ),
        (
            "gemma4.image_token_id".into(),
            GgufMetadataValue::Uint32(20),
        ),
        (
            "gemma4.video_token_id".into(),
            GgufMetadataValue::Uint32(21),
        ),
        (
            "gemma4.audio_token_id".into(),
            GgufMetadataValue::Uint32(22),
        ),
    ]);
    if moe {
        text_metadata.insert("gemma4.expert_count".into(), GgufMetadataValue::Uint32(4));
        text_metadata.insert(
            "gemma4.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        );
        text_metadata.insert(
            "gemma4.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(8),
        );
    }
    let model_path = directory.join(if moe {
        "gemma4-moe-f32.gguf"
    } else {
        "gemma4-f32.gguf"
    });
    write_dense_gguf(&model_path, &text, text_metadata);
    write_dense_gguf(
        &directory.join("mmproj-gemma4-f32.gguf"),
        &projector,
        projector_metadata(),
    );
    model_path
}

fn typed_input<'a>(
    text: &'a Array,
    pixels: &'a Array,
    positions: &'a Array,
    audio: &'a Array,
    audio_mask: &'a Array,
    parts: &'a mut [InputPart<'a>; 3],
) -> ModelInput<'a> {
    *parts = [
        InputPart::text_token_ids(text),
        InputPart::image_tensor(pixels, InputMetadata::patch_position_ids(positions)),
        InputPart::audio_tensor(audio, InputMetadata::audio_mask(audio_mask)),
    ];
    ModelInput::new(parts)
}

#[test]
fn gemma4_gguf_mmproj_rejects_wrong_identity_before_materialization() {
    let directory = tempfile::tempdir().unwrap();
    let model_path = write_gguf_fixture(directory.path());
    let projector_path = directory.path().join("wrong-projector.gguf");
    let stream = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let array = Array::zeros::<f32>(&[1], stream.stream()).unwrap();
    let mut metadata = projector_metadata();
    metadata.insert(
        "clip.vision.projector_type".into(),
        GgufMetadataValue::String("other".into()),
    );
    write_dense_gguf(
        &projector_path,
        &HashMap::from([("dummy".into(), array)]),
        metadata,
    );
    let error = gemma4::load_gemma4_gguf_with_mmproj(
        model_path,
        projector_path,
        stream.stream(),
        stream.stream(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("projector type \"gemma4\""));
}

#[test]
fn gemma4_gguf_mmproj_rejects_incomplete_catalog_before_materialization() {
    let directory = tempfile::tempdir().unwrap();
    let model_path = write_gguf_fixture(directory.path());
    let projector_path = directory.path().join("incomplete-projector.gguf");
    let stream = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let array = Array::zeros::<f32>(&[1], stream.stream()).unwrap();
    write_dense_gguf(
        &projector_path,
        &HashMap::from([("vision_tower.unexpected".into(), array)]),
        projector_metadata(),
    );
    let error = gemma4::load_gemma4_gguf_with_mmproj(
        model_path,
        projector_path,
        stream.stream(),
        stream.stream(),
    )
    .unwrap_err();
    let detail = error.to_string();
    assert!(
        detail.contains("missing") || detail.contains("Missing"),
        "{detail}"
    );
    assert!(detail.contains("audio_tower.layers.0"), "{detail}");
}

fn assert_close(actual: &Array, expected: &Array, stream: &Stream) {
    let actual = if actual.shape() != expected.shape() && actual.dim(1) > 1 {
        actual.try_index_device((.., -1, ..), stream).unwrap()
    } else {
        actual.clone()
    };
    let actual = actual.evaluated().unwrap();
    let expected = expected.evaluated().unwrap();
    let actual = actual.as_slice::<f32>();
    let expected = expected.as_slice::<f32>();
    assert!(actual.len() >= expected.len());
    let actual = &actual[actual.len() - expected.len()..];
    let maximum_error = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max);
    assert!(
        maximum_error <= 4e-3,
        "maximum absolute error {maximum_error} exceeded 4e-3"
    );
}

#[test]
fn gemma4_multimodal_pipeline_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT).unwrap());
    let cache_root = PathBuf::from(std::env::var_os(CACHE_ROOT).unwrap());
    let axes = std::env::var(AXES).ok();
    let tp = usize::from(
        std::env::var_os(TENSOR_PARALLEL).is_some() || axes.as_deref() == Some("tp-pp-ep"),
    ) + 1;
    let ep = if axes.is_some() { 2 } else { 1 };
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        MlxParallelContext::for_group(&group, tp, 2, ep, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let residency = if std::env::var_os(DENSE_STREAM).is_some() {
        WeightResidency::dense_disk_stream(
            DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
        )
    } else if std::env::var_os(LAYERWISE_HOST).is_some() {
        WeightResidency::layerwise_host(LayerwiseLoadOptions::new(
            OffloadConfig::new(None, None, 1).unwrap(),
        ))
    } else {
        WeightResidency::fully_resident()
    };
    let mut model = load_pipeline_model_with_options(
        &checkpoint,
        ModelLoadOptions::with_parallel(topology).with_weight_residency(residency),
        &stream,
        &stream,
    )
    .unwrap();
    let execution = MlxBackend::new(&stream)
        .communication_for_topology(topology, &group)
        .unwrap();
    assert_eq!(
        model.stage_info().global_layer_range,
        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
    );
    if ep > 1 {
        assert_eq!(model.stage_info().global_expert_count, Some(4));
        assert_eq!(model.stage_info().local_expert_ids.len(), 2);
    }
    if std::env::var_os(DENSE_STREAM).is_some() {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert_eq!(
            report.planned_layer_count(),
            if topology.pipeline_parallel_rank == 0 {
                3
            } else {
                1
            }
        );
    }
    if std::env::var_os(LAYERWISE_HOST).is_some() {
        let report = model.parameter_residency_report().unwrap().unwrap();
        assert_eq!(
            report.units().len(),
            if topology.pipeline_parallel_rank == 0 {
                3
            } else {
                1
            }
        );
    }

    let text = Array::from_slice(&[1u32, 2], &[1, 2]);
    let pixels = Array::zeros::<f32>(&[1, 4, 12], &stream).unwrap();
    let positions = Array::from_slice(&[0i32, 0, 0, 1, 1, 0, 1, 1], &[1, 4, 2]);
    let audio = Array::zeros::<f32>(&[1, 8, 128], &stream).unwrap();
    let audio_mask = Array::from_slice(&[true; 8], &[1, 8]);
    let mut parts = [InputPart::text_token_ids(&text); 3];
    let input = typed_input(&text, &pixels, &positions, &audio, &audio_mask, &mut parts);
    let prepared = PreparedModelInput::from_model_input(input).unwrap();
    let paged = PagedCacheOptions::new(1, 1 << 20, 1 << 20, 1)
        .unwrap()
        .with_full_attention(true);
    let mut cache = model
        .new_cache_with_options(safemlx_lm::CacheResidencyPolicy::Paged(paged.clone()))
        .unwrap();
    // Two text tokens, one pooled image token, and two audio tokens.
    let step = PipelineStep::new(1, 5).unwrap();
    let completion = prepared
        .with_model_input(|input| {
            model.prefill_distributed(
                model.stage_info().is_first.then_some(input),
                step,
                None,
                &mut cache,
                &execution,
            )
        })
        .unwrap();
    let schedule = model.placed_ingress_schedule_report();
    let bounded =
        std::env::var_os(DENSE_STREAM).is_some() || std::env::var_os(LAYERWISE_HOST).is_some();
    if bounded || tp > 1 {
        assert_eq!(schedule.maximum_in_flight_groups, 1);
        assert!(!schedule.serial_fallbacks.is_empty());
    } else {
        assert_eq!(schedule.maximum_in_flight_groups, 2);
        assert!(schedule.ready_batches.iter().any(|batch| {
            batch == &["vision_encoder".to_string(), "audio_encoder".to_string()]
        }));
    }
    if topology.pipeline_parallel_rank == 0 {
        assert!(!schedule.routed_transfers.is_empty());
    } else {
        assert!(schedule.routed_transfers.iter().all(|route| {
            route.from_group != "vision_encoder" && route.from_group != "audio_encoder"
        }));
    }
    let logits = completion.into_logits().unwrap();
    assert_eq!(logits.is_some(), topology.pipeline_parallel_rank == 1);

    let mut resident_cache = Cache::default();
    let mut resident = (topology.pipeline_parallel_rank == 1).then(|| {
        if checkpoint.extension().is_some_and(|value| value == "gguf") {
            gemma4::load_gemma4_gguf(&checkpoint, &stream, &stream).unwrap()
        } else {
            gemma4::load_gemma4_model(&checkpoint, &stream, &stream).unwrap()
        }
    });
    if let (Some(logits), Some(resident)) = (&logits, &mut resident) {
        let expected = resident
            .prefill_input_logits(input, &mut resident_cache, &stream)
            .unwrap();
        assert_close(logits, &expected, &stream);
    }

    let descriptor = PromptCacheDescriptor {
        model_family: "gemma4".into(),
        effective_model_type: "gemma4".into(),
        checkpoint_fingerprint: "gemma4-moe-pipeline-ring".into(),
        prefix_content_fingerprint: "typed:text+image+audio".into(),
        architecture_fingerprint: model.prompt_cache_architecture_fingerprint().unwrap(),
        layer_count: 2,
        global_layer_start: topology.pipeline_parallel_rank,
        global_layer_end: topology.pipeline_parallel_rank + 1,
        batch_size: 1,
        layer_prefix_offsets: vec![0],
        layer_layout: model.prompt_cache_layer_layout().unwrap(),
        sink_tokens: 0,
        topology: PromptCacheTopology {
            pipeline: Some((2, topology.pipeline_parallel_rank)),
            tensor_parallel: (tp > 1).then_some((tp, topology.tensor_parallel_rank)),
            expert_parallel: (ep > 1).then_some((ep, topology.expert_parallel_rank)),
            expert_parallel_cache_replicated: true,
        },
    };
    let prefix_ids = [1u32, 2, 20, 22, 22];
    model
        .save_prompt_cache(
            &mut cache,
            &cache_root,
            descriptor.clone(),
            &prefix_ids,
            &PromptCacheOptions::default(),
            &stream,
        )
        .unwrap();

    let token = Array::from_slice(&[3u32], &[1, 1]);
    let decoded = model
        .forward_distributed(
            model.stage_info().is_first.then_some(&token),
            PipelineStep::new(1, 1).unwrap(),
            None,
            &mut cache,
            &execution,
        )
        .unwrap()
        .into_logits()
        .unwrap();
    let (mut restored_cache, _) = model
        .load_prompt_cache(&cache_root, &descriptor, &prefix_ids, paged, &stream)
        .unwrap();
    let restored = model
        .forward_distributed(
            model.stage_info().is_first.then_some(&token),
            PipelineStep::new(1, 1).unwrap(),
            None,
            &mut restored_cache,
            &execution,
        )
        .unwrap()
        .into_logits()
        .unwrap();
    match (&decoded, &restored) {
        (Some(decoded), Some(restored)) => assert_close(decoded, restored, &stream),
        (None, None) => {}
        _ => panic!("prompt-cache reload changed Gemma 4 pipeline output ownership"),
    }
    if let (Some(decoded), Some(resident)) = (&decoded, &mut resident) {
        let expected = resident
            .decode_logits(&token, &mut resident_cache, &stream)
            .unwrap();
        assert_close(decoded, &expected, &stream);
    }
}

struct ChildGuard(Vec<Child>);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn failure(rank: usize, output: &Output) -> String {
    format!(
        "Gemma multimodal rank {rank} exited with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn run_ring(tp: bool, dense: bool, host: bool, gguf: bool) {
    run_ring_axes(tp, dense, host, gguf, None);
}

fn run_ring_axes(tp: bool, dense: bool, host: bool, gguf: bool, axes: Option<&str>) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if gguf && axes.is_some() {
        write_moe_gguf_fixture(checkpoint.path())
    } else if gguf {
        write_gguf_fixture(checkpoint.path())
    } else if axes.is_some() {
        write_moe_fixture(checkpoint.path());
        checkpoint.path().to_path_buf()
    } else {
        write_fixture(checkpoint.path());
        checkpoint.path().to_path_buf()
    };
    let world = if axes == Some("tp-pp-ep") {
        8
    } else if axes.is_some() || tp {
        4
    } else {
        2
    };
    let sockets = (0..world)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let ring = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    let hostfile = ring.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let executable = std::env::current_exe().unwrap();
    let mut children = ChildGuard(Vec::new());
    for rank in 0..world {
        let mut command = Command::new(&executable);
        command
            .args([
                "--exact",
                "distributed_gemma4_multimodal_pipeline_ring::gemma4_multimodal_pipeline_ring_worker",
                "--nocapture",
            ])
            .env(WORKER, rank.to_string())
            .env(CHECKPOINT, &checkpoint_path)
            .env(CACHE_ROOT, cache.path())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .env_remove("MLX_RING_VERBOSE")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if tp {
            command.env(TENSOR_PARALLEL, "1");
        }
        if dense {
            command.env(DENSE_STREAM, "1");
        }
        if host {
            command.env(LAYERWISE_HOST, "1");
        }
        if let Some(axes) = axes {
            command.env(AXES, axes);
        }
        children.0.push(command.spawn().unwrap());
    }
    let deadline = Instant::now() + Duration::from_secs(if world > 4 { 180 } else { 120 });
    let mut timed_out = false;
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        timed_out = Instant::now() >= deadline;
        if timed_out || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                if child.try_wait().unwrap().is_none() {
                    let _ = child.kill();
                }
            }
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let outputs = children
        .0
        .drain(..)
        .map(|child| child.wait_with_output().unwrap())
        .collect::<Vec<_>>();
    let failures = outputs
        .iter()
        .enumerate()
        .filter(|(_, output)| !output.status.success())
        .map(|(rank, output)| failure(rank, output))
        .collect::<Vec<_>>();
    assert!(
        failures.is_empty() && !timed_out,
        "{}{}",
        if timed_out {
            "Ring workers timed out\n"
        } else {
            ""
        },
        failures.join("\n\n")
    );
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_pipeline() {
    run_ring(false, false, false, false);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_dense_stream_pipeline() {
    run_ring(false, true, false, false);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_host_tensor_pipeline() {
    run_ring(true, false, true, false);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_gguf_pipeline() {
    run_ring(false, false, false, true);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_gguf_dense_stream_pipeline() {
    run_ring(false, true, false, true);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_gguf_host_tensor_pipeline() {
    run_ring(true, false, true, true);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_moe_pipeline_expert_multimodal() {
    run_ring_axes(false, false, false, false, Some("pp-ep"));
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_moe_triple_axis_multimodal() {
    run_ring_axes(false, false, false, false, Some("tp-pp-ep"));
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_moe_streamed_pipeline_expert_multimodal() {
    run_ring_axes(false, true, false, false, Some("pp-ep"));
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_moe_host_triple_axis_multimodal() {
    run_ring_axes(false, false, true, false, Some("tp-pp-ep"));
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_moe_gguf_triple_axis_multimodal() {
    run_ring_axes(false, true, false, true, Some("tp-pp-ep"));
}
