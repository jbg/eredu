#![cfg(unix)]

use std::{
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Backend},
    module::ModuleParameters,
    ops::indexing::TryIndexOp,
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_lm::{
    architectures::{
        distributed::pipeline::{load_pipeline_model_with_options, PipelineStep},
        qwen::vl::model as qwen3_vl,
    },
    nn::generation::CausalLm,
    runtime::generation::sampler::DefaultSampler,
    runtime::media::input::{InputMetadata, InputPart, ModelInput},
    CartesianExecution, DenseDiskStreamLoadOptions, DeviceAssignment, ModelLoadOptions,
    PagedCacheOptions, ParallelTopology, PromptCacheDescriptor, PromptCacheOptions,
    PromptCacheTopology, WeightResidency,
};

const WORKER: &str = "SAFEMLX_QWEN3_VL_PIPELINE_WORKER";
const CHECKPOINT: &str = "SAFEMLX_QWEN3_VL_PIPELINE_CHECKPOINT";
const CACHE_ROOT: &str = "SAFEMLX_QWEN3_VL_PIPELINE_CACHE";
const AXES: &str = "SAFEMLX_QWEN3_VL_PIPELINE_AXES";
const MOE: &str = "SAFEMLX_QWEN3_VL_PIPELINE_MOE";
const STREAMED: &str = "SAFEMLX_QWEN3_VL_PIPELINE_STREAMED";

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
    let residency = if std::env::var_os(STREAMED).is_some() {
        WeightResidency::dense_disk_stream(
            DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1, 1).unwrap(),
        )
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
    assert_eq!(
        model.stage_info().global_layer_range,
        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
    );
    let paged = PagedCacheOptions::new(1, 4096, 4096, 1)
        .unwrap()
        .with_full_attention(true);
    let mut cache = model
        .new_cache_with_options(safemlx_lm::CacheResidencyPolicy::Paged(paged.clone()))
        .unwrap();
    let before = Array::from_slice(&[1u32], &[1, 1]);
    let after = Array::from_slice(&[2u32], &[1, 1]);
    let pixels = Array::from_slice(&[0.01f32; 96], &[4, 24]);
    let grid = Array::from_slice(&[1i32, 2, 2], &[1, 3]);
    let mut parts = [InputPart::text_token_ids(&before); 3];
    let input = multimodal_input(&before, &pixels, &grid, &after, &mut parts);
    let logits = match &cartesian {
        Some(cartesian) => model
            .prefill_cartesian(
                (topology.pipeline_parallel_rank == 0).then_some(input),
                PipelineStep::new(1, 3).unwrap(),
                None,
                &mut cache,
                cartesian,
                &stream,
            )
            .unwrap(),
        None => model
            .prefill_pipeline(
                (topology.pipeline_parallel_rank == 0).then_some(input),
                PipelineStep::new(1, 3).unwrap(),
                None,
                &mut cache,
                &group,
                &stream,
            )
            .unwrap(),
    };
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
        let mut resident = qwen3_vl::load_qwen3_vl_model(&checkpoint, &stream, &stream).unwrap();
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
    let uninterrupted = decode(&mut model, &mut cache).unwrap();
    let (mut restored_cache, _) = model
        .load_prompt_cache(&cache_root, &descriptor, &[1, 30, 2], paged, &stream)
        .unwrap();
    let restored = decode(&mut model, &mut restored_cache).unwrap();
    match (uninterrupted, restored) {
        (Some(left), Some(right)) => assert_close(&left, &right),
        (None, None) => {}
        _ => panic!("prompt-cache reload changed Qwen3-VL output ownership"),
    }
    if std::env::var_os(STREAMED).is_some() {
        let report = model.dense_stream_report().unwrap().unwrap();
        assert_eq!(report.planned_layer_count(), 1);
        assert!(report.prefill_forwards() >= 1);
        assert!(report.decode_forwards() >= 1);
    }
}

fn assert_close(left: &Array, right: &Array) {
    let left = left.evaluated().unwrap();
    let right = right.evaluated().unwrap();
    assert_eq!(left.as_array().shape(), right.as_array().shape());
    assert!(left
        .as_slice::<f32>()
        .iter()
        .zip(right.as_slice::<f32>())
        .all(|(left, right)| (left - right).abs() <= 8e-5));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_pipeline_multimodal() {
    run(false, false, None);
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_dense_stream_pipeline_multimodal() {
    run(false, true, None);
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_tensor_pipeline_multimodal() {
    run(false, true, Some("tp-pp"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_pipeline_expert_multimodal() {
    run(true, true, Some("pp-ep"));
}

#[test]
#[ignore = "spawns local processes and opens loopback sockets; run explicitly"]
fn ring_qwen3_vl_moe_resident_pipeline_expert_multimodal() {
    run(true, false, Some("pp-ep"));
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

fn run(moe: bool, streamed: bool, axes: Option<&str>) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path(), moe);
    let cache = tempfile::tempdir().unwrap();
    let world = if axes.is_some() { 4 } else { 2 };
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
            .env(CHECKPOINT, checkpoint.path())
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
