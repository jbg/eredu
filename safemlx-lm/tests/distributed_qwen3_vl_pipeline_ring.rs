#![cfg(unix)]

use std::{
    collections::{BTreeMap, HashMap},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
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
use safemlx_lm::runtime::residency::policy::OffloadConfig;
use safemlx_lm::{
    architectures::{
        distributed::pipeline::{
            load_pipeline_model_with_options, PipelineInferencePhase, PipelineInferenceScheduler,
            PipelineMicrobatchInput, PipelineStep,
        },
        qwen::vl::model as qwen3_vl,
    },
    nn::generation::CausalLm,
    runtime::generation::sampler::DefaultSampler,
    runtime::media::input::{InputMetadata, InputPart, ModelInput},
    runtime::media::PreparedModelInput,
    runtime::scheduler::{RequestId, RequestStatus, SchedulerLimits},
    CartesianExecution, DenseDiskStreamLoadOptions, DeviceAssignment, ExpertCacheLoadOptions,
    LayerwiseLoadOptions, ModelLoadOptions, NonExpertWeightResidency, PagedCacheOptions,
    ParallelTopology, PromptCacheDescriptor, PromptCacheOptions, PromptCacheTopology,
    WeightResidency,
};

const WORKER: &str = "SAFEMLX_QWEN3_VL_PIPELINE_WORKER";
const CHECKPOINT: &str = "SAFEMLX_QWEN3_VL_PIPELINE_CHECKPOINT";
const CACHE_ROOT: &str = "SAFEMLX_QWEN3_VL_PIPELINE_CACHE";
const AXES: &str = "SAFEMLX_QWEN3_VL_PIPELINE_AXES";
const MOE: &str = "SAFEMLX_QWEN3_VL_PIPELINE_MOE";
const STREAMED: &str = "SAFEMLX_QWEN3_VL_PIPELINE_STREAMED";
const LAYERWISE_HOST: &str = "SAFEMLX_QWEN3_VL_PIPELINE_LAYERWISE_HOST";
const EXPERT_CACHE: &str = "SAFEMLX_QWEN3_VL_PIPELINE_EXPERT_CACHE";
const SCHEDULE_MISMATCH: &str = "SAFEMLX_QWEN3_VL_PIPELINE_SCHEDULE_MISMATCH";

fn config(moe: bool) -> serde_json::Value {
    let model_type = if moe { "qwen3_vl_moe" } else { "qwen3_vl" };
    serde_json::json!({
        "model_type": model_type,
        "image_token_id": 30,
        "video_token_id": 31,
        "tie_word_embeddings": !moe,
        "text_config": {
            "model_type": format!("{model_type}_text"),
            "hidden_size": 12,
            "num_hidden_layers": 2,
            "intermediate_size": 24,
            "num_attention_heads": 2,
            "rms_norm_eps": 1e-6,
            "vocab_size": 32,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "rope_theta": 10000.0,
            "head_dim": 6,
            "tie_word_embeddings": !moe,
            "moe_intermediate_size": if moe { 8 } else { 0 },
            "num_experts": if moe { 4 } else { 0 },
            "num_experts_per_tok": if moe { 2 } else { 0 },
            "norm_topk_prob": moe,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_interleaved": true,
                "mrope_section": [1, 1, 1]
            }
        },
        "vision_config": {
            "depth": 2,
            "hidden_size": 8,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 16,
            "num_heads": 2,
            "num_position_embeddings": 16,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 12,
            "deepstack_visual_indexes": [0]
        }
    })
}

fn write_fixture(directory: &Path, moe: bool) {
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config(moe)).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = qwen3_vl::get_qwen3_vl_model_args(directory).unwrap();
    let mut model = qwen3_vl::Model::new(args, stream).unwrap();
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
            (
                safemlx_lm::runtime::checkpoint::binding::canonical_checkpoint_name(name),
                *value,
            )
        })
        .collect::<Vec<_>>();
    Array::save_safetensors(
        arrays.iter().map(|(name, value)| (name.as_str(), *value)),
        None,
        directory.join("model.safetensors"),
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

fn write_gguf_fixture(root: &Path) -> PathBuf {
    let source = root.join("source");
    std::fs::create_dir_all(&source).unwrap();
    let mut value = config(true);
    value["tie_word_embeddings"] = serde_json::Value::Bool(true);
    value["text_config"]["tie_word_embeddings"] = serde_json::Value::Bool(true);
    std::fs::write(
        source.join("config.json"),
        serde_json::to_vec(&value).unwrap(),
    )
    .unwrap();
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let args = qwen3_vl::get_qwen3_vl_model_args(&source).unwrap();
    let mut model = qwen3_vl::Model::new(args, stream).unwrap();
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
    let mut vision = HashMap::new();
    for (name, value) in model.parameters().flatten() {
        if let Some(name) = name.strip_prefix("model.language_model.") {
            if let Some(rest) = name.strip_prefix("layers.") {
                let (layer, suffix) = rest.split_once('.').unwrap();
                if suffix == "mlp.experts.gate_up_proj" {
                    text.insert(
                        format!("blk.{layer}.ffn_gate_exps.weight"),
                        value.try_index_device((.., ..8, ..), stream).unwrap(),
                    );
                    text.insert(
                        format!("blk.{layer}.ffn_up_exps.weight"),
                        value.try_index_device((.., 8.., ..), stream).unwrap(),
                    );
                    continue;
                }
                if suffix == "mlp.experts.down_proj" {
                    text.insert(format!("blk.{layer}.ffn_down_exps.weight"), value.clone());
                    continue;
                }
            }
            let name = name
                .replace("layers.", "blk.")
                .replace("self_attn.q_norm", "attn_q_norm")
                .replace("self_attn.k_norm", "attn_k_norm")
                .replace("self_attn.q_proj", "attn_q")
                .replace("self_attn.k_proj", "attn_k")
                .replace("self_attn.v_proj", "attn_v")
                .replace("self_attn.o_proj", "attn_output")
                .replace("input_layernorm", "attn_norm")
                .replace("post_attention_layernorm", "ffn_norm")
                .replace("mlp.gate.weight", "ffn_gate_inp.weight");
            let name = match name.as_str() {
                "embed_tokens.weight" => "token_embd.weight".into(),
                "norm.weight" => "output_norm.weight".into(),
                _ => name,
            };
            text.insert(name, value.clone());
            continue;
        }
        if name.as_ref() == "lm_head.weight" {
            text.insert("output.weight".into(), value.clone());
            continue;
        }
        let name = name.strip_prefix("model.visual.").unwrap();
        if name == "patch_embed.proj.weight" {
            vision.insert(
                "v.patch_embd.weight".into(),
                value.try_index_device((.., .., 0, .., ..), stream).unwrap(),
            );
            vision.insert(
                "v.patch_embd.weight.1".into(),
                value.try_index_device((.., .., 1, .., ..), stream).unwrap(),
            );
            continue;
        }
        let name = name
            .replace("pos_embed", "v.position_embd")
            .replace("patch_embed.proj", "v.patch_embd")
            .replace("blocks.", "v.blk.")
            .replace(".attn.qkv.", ".attn_qkv.")
            .replace(".attn.proj.", ".attn_out.")
            .replace(".mlp.linear_fc1.", ".ffn_up.")
            .replace(".mlp.linear_fc2.", ".ffn_down.")
            .replace(".norm1.", ".ln1.")
            .replace(".norm2.", ".ln2.")
            .replace("merger.norm", "v.post_ln")
            .replace("merger.linear_fc1", "mm.0")
            .replace("merger.linear_fc2", "mm.2")
            .replace("deepstack_merger_list.0.norm", "v.deepstack.0.norm")
            .replace("deepstack_merger_list.0.linear_fc1", "v.deepstack.0.fc1")
            .replace("deepstack_merger_list.0.linear_fc2", "v.deepstack.0.fc2");
        vision.insert(name, value.clone());
    }

    let mut tokens = (0..30)
        .map(|index| format!("token-{index}"))
        .collect::<Vec<_>>();
    tokens.extend(["<|image_pad|>".into(), "<|video_pad|>".into()]);
    let text_metadata = HashMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("qwen3vlmoe".into()),
        ),
        (
            "qwen3vlmoe.embedding_length".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "qwen3vlmoe.block_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "qwen3vlmoe.feed_forward_length".into(),
            GgufMetadataValue::Uint32(24),
        ),
        (
            "qwen3vlmoe.expert_feed_forward_length".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "qwen3vlmoe.expert_count".into(),
            GgufMetadataValue::Uint32(4),
        ),
        (
            "qwen3vlmoe.expert_used_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "qwen3vlmoe.attention.head_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "qwen3vlmoe.attention.head_count_kv".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "qwen3vlmoe.attention.key_length".into(),
            GgufMetadataValue::Uint32(6),
        ),
        (
            "qwen3vlmoe.attention.value_length".into(),
            GgufMetadataValue::Uint32(6),
        ),
        (
            "qwen3vlmoe.attention.layer_norm_rms_epsilon".into(),
            GgufMetadataValue::Float32(1e-6),
        ),
        (
            "qwen3vlmoe.context_length".into(),
            GgufMetadataValue::Uint32(128),
        ),
        (
            "qwen3vlmoe.rope.freq_base".into(),
            GgufMetadataValue::Float32(10_000.0),
        ),
        (
            "qwen3vlmoe.rope.dimension_sections".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![1, 1, 1, 0])),
        ),
        (
            "qwen3vlmoe.n_deepstack_layers".into(),
            GgufMetadataValue::Uint32(1),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufMetadataValue::Array(GgufMetadataArray::String(tokens)),
        ),
        (
            "tokenizer.ggml.eos_token_id".into(),
            GgufMetadataValue::Uint32(2),
        ),
    ]);
    let vision_metadata = HashMap::from([
        (
            "general.architecture".into(),
            GgufMetadataValue::String("clip".into()),
        ),
        (
            "clip.projector_type".into(),
            GgufMetadataValue::String("qwen3vl_merger".into()),
        ),
        (
            "clip.vision.embedding_length".into(),
            GgufMetadataValue::Uint32(8),
        ),
        (
            "clip.vision.block_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.feed_forward_length".into(),
            GgufMetadataValue::Uint32(16),
        ),
        (
            "clip.vision.attention.head_count".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.patch_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.spatial_merge_size".into(),
            GgufMetadataValue::Uint32(2),
        ),
        (
            "clip.vision.projection_dim".into(),
            GgufMetadataValue::Uint32(12),
        ),
        (
            "clip.vision.is_deepstack_layers".into(),
            GgufMetadataValue::Array(GgufMetadataArray::Bool(vec![true, false])),
        ),
    ]);
    let model_path = root.join("qwen3vlmoe-f32.gguf");
    let mmproj_path = root.join("mmproj-qwen3vlmoe-f32.gguf");
    write_dense_gguf(&model_path, &text, text_metadata);
    write_dense_gguf(&mmproj_path, &vision, vision_metadata);
    model_path
}

fn multimodal_input<'a>(
    before: &'a Array,
    pixels: &'a Array,
    grid: &'a Array,
    after: &'a Array,
    parts: &'a mut [InputPart<'a>; 3],
) -> ModelInput<'a> {
    *parts = [
        InputPart::text_token_ids(before),
        InputPart::image_tensor(pixels, InputMetadata::qwen_grid_thw(grid)),
        InputPart::text_token_ids(after),
    ];
    ModelInput::new(parts)
}

#[test]
fn qwen3_vl_pipeline_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER) else {
        return;
    };
    let expected_rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT).unwrap());
    let cache_root = PathBuf::from(std::env::var_os(CACHE_ROOT).unwrap());
    let moe = std::env::var_os(MOE).is_some();
    let axes = std::env::var(AXES).ok();
    let (tp, ep) = match axes.as_deref() {
        None => (1, 1),
        Some("tp-pp") => (2, 1),
        Some("pp-ep") => (1, 2),
        Some("tp-pp-ep") => (2, 2),
        other => panic!("unexpected axes {other:?}"),
    };
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        ParallelTopology::from_group(&group, tp, 2, ep, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, expected_rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let cartesian = axes
        .as_ref()
        .map(|_| CartesianExecution::new(topology, Some(2), moe.then_some(4), &group).unwrap());
    let streamed = std::env::var_os(STREAMED).is_some();
    let layerwise_host = std::env::var_os(LAYERWISE_HOST).is_some();
    let expert_cache = std::env::var_os(EXPERT_CACHE).is_some();
    assert!(!(streamed && layerwise_host));
    let residency = if expert_cache {
        let non_expert = if streamed {
            NonExpertWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
            )
        } else if layerwise_host {
            NonExpertWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(None, None, 1).unwrap(),
            ))
        } else {
            NonExpertWeightResidency::FullyResident
        };
        WeightResidency::with_expert_cache(non_expert, ExpertCacheLoadOptions::default())
    } else if streamed {
        WeightResidency::dense_disk_stream(
            DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1).unwrap(),
        )
    } else if layerwise_host {
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
    if expert_cache {
        let report = model.expert_cache_report().unwrap().unwrap();
        assert_eq!(
            report.owned_experts,
            model.stage_info().local_expert_ids.len()
        );
        assert!(report.owned_bytes > 0);
    }
    if std::env::var_os(SCHEDULE_MISMATCH).is_some() {
        let request = RequestId::new(101);
        let mut scheduler =
            PipelineInferenceScheduler::new(&model, SchedulerLimits::new(1, 1).unwrap()).unwrap();
        scheduler.register_request(&model, request).unwrap();
        let before = Array::from_slice(&[1u32], &[1, 1]);
        let after = Array::from_slice(&[2u32], &[1, 1]);
        let pixel_shape = if expected_rank == 0 { [4, 24] } else { [2, 48] };
        let pixels = Array::from_slice(&[0.01f32; 96], &pixel_shape);
        let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
        let mut parts = [InputPart::text_token_ids(&before); 3];
        let prepared = PreparedModelInput::from_model_input(multimodal_input(
            &before, &pixels, &grid, &after, &mut parts,
        ))
        .unwrap();
        let identity = prepared.identity();
        let work = PipelineMicrobatchInput::new(
            request,
            PipelineInferencePhase::Prefill,
            PipelineStep::new(1, 3).unwrap(),
        );
        let work = if model.stage_info().is_first {
            work.with_prepared_input(prepared)
        } else {
            work.with_prepared_input_identity(identity)
        };
        scheduler.enqueue(work).unwrap();
        let error = match &cartesian {
            Some(cartesian) => scheduler
                .run_queued_cartesian(&mut model, cartesian, &stream)
                .unwrap_err(),
            None => scheduler
                .run_queued(&mut model, &group, &stream)
                .unwrap_err(),
        };
        assert!(error.to_string().contains("work descriptors differ"));
        assert!(scheduler.report().poisoned);
        assert_eq!(
            scheduler.request_status(request),
            Some(RequestStatus::Failed)
        );
        return;
    }
    assert_eq!(
        model.stage_info().global_layer_range,
        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
    );
    let paged = PagedCacheOptions::new(1, 32768, 32768, 1)
        .unwrap()
        .with_full_attention(true);
    let before = Array::from_slice(&[1u32], &[1, 1]);
    let after = Array::from_slice(&[2u32], &[1, 1]);
    let pixels = Array::from_slice(&[0.01f32; 96], &[4, 24]);
    let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
    let mut parts = [InputPart::text_token_ids(&before); 3];
    let input = multimodal_input(&before, &pixels, &grid, &after, &mut parts);
    let prepared = PreparedModelInput::from_model_input(input).unwrap();
    let identity = prepared.identity();
    let request = RequestId::new(7);
    let mut scheduler =
        PipelineInferenceScheduler::new(&model, SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler
        .register_request_with_options(
            &model,
            request,
            safemlx_lm::CacheResidencyPolicy::Paged(paged.clone()),
        )
        .unwrap();
    let work = PipelineMicrobatchInput::new(
        request,
        PipelineInferencePhase::Prefill,
        PipelineStep::new(1, 3).unwrap(),
    );
    let work = if model.stage_info().is_first {
        work.with_prepared_input(prepared)
    } else {
        work.with_prepared_input_identity(identity)
    };
    scheduler.enqueue(work).unwrap();
    let mut completed = Vec::new();
    for _ in 0..64 {
        completed.extend(match &cartesian {
            Some(cartesian) => scheduler
                .run_queued_cartesian(&mut model, cartesian, &stream)
                .unwrap(),
            None => scheduler.run_queued(&mut model, &group, &stream).unwrap(),
        });
        if !completed.is_empty() {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(completed.len(), 1);
    let logits = completed.pop().unwrap().into_logits().unwrap();
    let mut cache = scheduler.release_request_cache(request).unwrap();
    assert_eq!(logits.is_some(), topology.pipeline_parallel_rank == 1);
    let synchronized = model
        .sample_and_synchronize(
            logits.as_ref(),
            PipelineStep::new(1, 3).unwrap(),
            &mut DefaultSampler,
            0.0,
            None,
            false,
            &group,
            &stream,
        )
        .unwrap();
    assert_eq!(synchronized.token.shape(), &[1, 1]);

    if topology.pipeline_parallel_rank == 1 && topology.tensor_parallel_rank == 0 {
        let mut resident = if checkpoint.extension().is_some_and(|value| value == "gguf") {
            qwen3_vl::load_qwen3_vl_gguf(
                &checkpoint,
                checkpoint
                    .parent()
                    .unwrap()
                    .join("mmproj-qwen3vlmoe-f32.gguf"),
                &stream,
                &stream,
            )
            .unwrap()
        } else {
            qwen3_vl::load_qwen3_vl_model(&checkpoint, &stream, &stream).unwrap()
        };
        let mut resident_cache = resident.new_cache();
        let expected = resident
            .prefill_input_logits(input, &mut resident_cache, &stream)
            .unwrap();
        assert_close(
            &logits
                .as_ref()
                .unwrap()
                .try_index_device((.., -1, ..), &stream)
                .unwrap(),
            &expected,
        );
    }

    let descriptor = PromptCacheDescriptor {
        model_family: "qwen3_vl".into(),
        effective_model_type: if moe { "qwen3_vl_moe" } else { "qwen3_vl" }.into(),
        checkpoint_fingerprint: "qwen3-vl-pipeline-ring".into(),
        prefix_content_fingerprint: "tokens:1,30,2".into(),
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
    model
        .save_prompt_cache(
            &mut cache,
            &cache_root,
            descriptor.clone(),
            &[1, 30, 2],
            &PromptCacheOptions::default(),
            &stream,
        )
        .unwrap();
    let token = Array::from_slice(&[3u32], &[1, 1]);
    let decode =
        |model: &mut safemlx_lm::architectures::distributed::pipeline::PipelineModel,
         cache: &mut safemlx_lm::architectures::distributed::pipeline::PipelineCache| {
            match &cartesian {
                Some(cartesian) => model.forward_cartesian(
                    (topology.pipeline_parallel_rank == 0).then_some(&token),
                    PipelineStep::new(1, 1).unwrap(),
                    None,
                    cache,
                    cartesian,
                    &stream,
                ),
                None => model.forward_pipeline(
                    (topology.pipeline_parallel_rank == 0).then_some(&token),
                    PipelineStep::new(1, 1).unwrap(),
                    None,
                    cache,
                    &group,
                    &stream,
                ),
            }
        };
    let uninterrupted = decode(&mut model, &mut cache)
        .unwrap()
        .into_logits()
        .unwrap();
    let (mut restored_cache, _) = model
        .load_prompt_cache(&cache_root, &descriptor, &[1, 30, 2], paged, &stream)
        .unwrap();
    let restored = decode(&mut model, &mut restored_cache)
        .unwrap()
        .into_logits()
        .unwrap();
    match (uninterrupted, restored) {
        (Some(left), Some(right)) => assert_close(&left, &right),
        (None, None) => {}
        _ => panic!("prompt-cache reload changed Qwen3-VL output ownership"),
    }
    if streamed {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert_eq!(report.planned_layer_count(), 2);
        assert!(report.prefill_forwards() >= 1);
        assert!(report.decode_forwards() >= 1);
        if checkpoint.extension().is_some_and(|value| value == "gguf") {
            let diagnostics = model.checkpoint_diagnostics().unwrap().unwrap();
            let total = std::fs::metadata(&checkpoint).unwrap().len()
                + std::fs::metadata(
                    checkpoint
                        .parent()
                        .unwrap()
                        .join("mmproj-qwen3vlmoe-f32.gguf"),
                )
                .unwrap()
                .len();
            assert!(diagnostics.physical_reads > 0);
            assert!(diagnostics.physical_read_bytes < total);
        }
    }
    if layerwise_host {
        let report = model.parameter_residency_report().unwrap().unwrap();
        assert!(report.initialized());
        assert_eq!(report.units().len(), 2);
        assert!(report.units().iter().all(|unit| unit.host_resident()));
    }
    if expert_cache {
        let report = model.expert_cache_report().unwrap().unwrap();
        let requests = report.prefill.device.requests + report.decode.device.requests;
        if requests == 0 {
            assert_eq!(report.device_resident_experts, 0);
        } else {
            assert!(report.device_resident_experts > 0);
        }
        let total_requests = distributed::all_sum(
            &Array::from_slice(&[requests as f32], &[1]),
            &group,
            &stream,
        )
        .unwrap()
        .try_item::<f32>(&stream)
        .unwrap();
        assert!(total_requests > 0.0);
    }
}

fn assert_close(left: &Array, right: &Array) {
    let left = left.evaluated().unwrap();
    let right = right.evaluated().unwrap();
    assert_eq!(left.as_array().shape(), right.as_array().shape());
    let maximum_error = left
        .as_slice::<f32>()
        .iter()
        .zip(right.as_slice::<f32>())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0f32, f32::max);
    assert!(
        maximum_error <= 2e-3,
        "maximum absolute error {maximum_error} exceeded 2e-3"
    );
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_pipeline_multimodal() {
    run(false, false, false, false, false, None);
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_dense_stream_pipeline_multimodal() {
    run(false, true, false, false, false, None);
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_tensor_pipeline_multimodal() {
    run(false, true, false, false, false, Some("tp-pp"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_pipeline_expert_multimodal() {
    run(true, true, false, false, false, Some("pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_resident_pipeline_expert_multimodal() {
    run(true, false, false, false, false, Some("pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_triple_axis_multimodal() {
    run(true, false, false, false, false, Some("tp-pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_triple_axis_resident_nonexpert_cache() {
    run(true, false, false, true, false, Some("tp-pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_triple_axis_streamed_nonexpert_cache() {
    run(true, true, false, true, false, Some("tp-pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_triple_axis_layerwise_host_nonexpert_cache() {
    run(true, false, true, true, false, Some("tp-pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_tensor_pipeline_expert_cache_without_ep() {
    run(true, true, false, true, false, Some("tp-pp"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_gguf_triple_axis_streamed_nonexpert_cache() {
    run(true, true, false, true, true, Some("tp-pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_triple_axis_expert_cache_mismatch_consensus() {
    run_mode(true, false, false, true, false, Some("tp-pp-ep"), true);
}

struct Children(Vec<Child>);

impl Drop for Children {
    fn drop(&mut self) {
        for child in &mut self.0 {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

fn run(
    moe: bool,
    streamed: bool,
    layerwise_host: bool,
    expert_cache: bool,
    gguf: bool,
    axes: Option<&str>,
) {
    run_mode(
        moe,
        streamed,
        layerwise_host,
        expert_cache,
        gguf,
        axes,
        false,
    );
}

#[allow(clippy::too_many_arguments)]
fn run_mode(
    moe: bool,
    streamed: bool,
    layerwise_host: bool,
    expert_cache: bool,
    gguf: bool,
    axes: Option<&str>,
    schedule_mismatch: bool,
) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    let checkpoint_path = if gguf {
        write_gguf_fixture(checkpoint.path())
    } else {
        write_fixture(checkpoint.path(), moe);
        checkpoint.path().to_path_buf()
    };
    let cache = tempfile::tempdir().unwrap();
    let world = if axes == Some("tp-pp-ep") {
        8
    } else if axes.is_some() {
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
    let hostfile = ring.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let executable = std::env::current_exe().unwrap();
    let mut children = Children(Vec::new());
    for rank in 0..world {
        let mut command = Command::new(&executable);
        command
            .args(["--exact", "qwen3_vl_pipeline_ring_worker", "--nocapture"])
            .env(WORKER, rank.to_string())
            .env(CHECKPOINT, &checkpoint_path)
            .env(CACHE_ROOT, cache.path())
            .env("MLX_RANK", rank.to_string())
            .env("MLX_HOSTFILE", &hostfile)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if moe {
            command.env(MOE, "1");
        }
        if streamed {
            command.env(STREAMED, "1");
        }
        if layerwise_host {
            command.env(LAYERWISE_HOST, "1");
        }
        if expert_cache {
            command.env(EXPERT_CACHE, "1");
        }
        if schedule_mismatch {
            command.env(SCHEDULE_MISMATCH, "1");
        }
        if let Some(axes) = axes {
            command.env(AXES, axes);
        }
        children.0.push(command.spawn().unwrap());
    }
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let states = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if states.iter().all(Option::is_some) {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for Qwen3-VL Ring workers"
        );
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
        .map(|(rank, output)| {
            format!(
                "rank {rank}:\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>();
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
