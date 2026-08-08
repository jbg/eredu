#![cfg(unix)]

use std::{
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
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
        distributed::pipeline::{
            load_pipeline_model_with_options, PipelineInferencePhase, PipelineInferenceScheduler,
            PipelineMicrobatchInput, PipelineStep,
        },
        gemma4::model::{self as gemma4, Cache, Model},
    },
    nn::generation::CausalLm,
    runtime::{
        media::{
            input::{InputMetadata, InputPart, ModelInput},
            PreparedModelInput,
        },
        residency::policy::OffloadConfig,
        scheduler::{RequestId, SchedulerLimits},
    },
    CartesianExecution, DenseDiskStreamLoadOptions, DeviceAssignment, LayerwiseLoadOptions,
    ModelLoadOptions, ParallelTopology, WeightResidency,
};

const WORKER: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_WORKER";
const CHECKPOINT: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_CHECKPOINT";
const TENSOR_PARALLEL: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_TP";
const DENSE_STREAM: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_STREAM";
const LAYERWISE_HOST: &str = "SAFEMLX_GEMMA4_MM_PIPELINE_HOST";

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

fn write_fixture(directory: &Path) {
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut model = Model::new_from_config_value(&config(), stream).unwrap();
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
        serde_json::to_vec_pretty(&config()).unwrap(),
    )
    .unwrap();
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
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(actual, expected)| (actual - expected).abs() <= 1e-4));
}

#[test]
fn gemma4_multimodal_pipeline_ring_worker() {
    let Some(rank) = std::env::var_os(WORKER) else {
        return;
    };
    let rank = rank.to_string_lossy().parse::<usize>().unwrap();
    let checkpoint = PathBuf::from(std::env::var_os(CHECKPOINT).unwrap());
    let tp = if std::env::var_os(TENSOR_PARALLEL).is_some() {
        2
    } else {
        1
    };
    let group = distributed::init(true, Backend::Ring).unwrap();
    let topology =
        ParallelTopology::from_group(&group, tp, 2, 1, DeviceAssignment::new(DeviceType::Cpu, 0))
            .unwrap();
    assert_eq!(topology.global_rank, rank);
    let stream = Stream::new_with_device(&topology.device.device().unwrap());
    let residency = if std::env::var_os(DENSE_STREAM).is_some() {
        WeightResidency::dense_disk_stream(
            DenseDiskStreamLoadOptions::new(u64::MAX, u64::MAX, 1, 1, 1).unwrap(),
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
    let cartesian =
        (tp > 1).then(|| CartesianExecution::new(topology, Some(2), None, &group).unwrap());
    assert_eq!(
        model.stage_info().global_layer_range,
        topology.pipeline_parallel_rank..topology.pipeline_parallel_rank + 1
    );
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
    let identity = prepared.identity();
    let request = RequestId::new(44);
    let mut scheduler =
        PipelineInferenceScheduler::new(&model, SchedulerLimits::new(1, 1).unwrap()).unwrap();
    scheduler.register_request(&model, request).unwrap();
    // Two text tokens, one pooled image token, and two audio tokens.
    let step = PipelineStep::new(1, 5).unwrap();
    let work = PipelineMicrobatchInput::new(request, PipelineInferencePhase::Prefill, step);
    scheduler
        .enqueue(if model.stage_info().is_first {
            work.with_prepared_input(prepared)
        } else {
            work.with_prepared_input_identity(identity)
        })
        .unwrap();
    let mut completed = match &cartesian {
        Some(cartesian) => scheduler
            .run_queued_cartesian(&mut model, cartesian, &stream)
            .unwrap(),
        None => scheduler.run_queued(&mut model, &group, &stream).unwrap(),
    };
    let logits = completed.pop().unwrap().into_logits();
    let mut cache = scheduler.release_request_cache(request).unwrap();
    assert_eq!(logits.is_some(), topology.pipeline_parallel_rank == 1);

    let mut resident_cache = Cache::default();
    let mut resident = (topology.pipeline_parallel_rank == 1)
        .then(|| gemma4::load_gemma4_model(&checkpoint, &stream, &stream).unwrap());
    if let (Some(logits), Some(resident)) = (&logits, &mut resident) {
        let expected = resident
            .prefill_input_logits(input, &mut resident_cache, &stream)
            .unwrap();
        assert_close(logits, &expected, &stream);
    }

    let token = Array::from_slice(&[3u32], &[1, 1]);
    let decoded = match &cartesian {
        Some(cartesian) => model
            .forward_cartesian(
                model.stage_info().is_first.then_some(&token),
                PipelineStep::new(1, 1).unwrap(),
                None,
                &mut cache,
                cartesian,
                &stream,
            )
            .unwrap(),
        None => model
            .forward_pipeline(
                model.stage_info().is_first.then_some(&token),
                PipelineStep::new(1, 1).unwrap(),
                None,
                &mut cache,
                &group,
                &stream,
            )
            .unwrap(),
    };
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

fn run_ring(tp: bool, dense: bool, host: bool) {
    assert!(distributed::is_available(Backend::Ring));
    let checkpoint = tempfile::tempdir().unwrap();
    write_fixture(checkpoint.path());
    let world = if tp { 4 } else { 2 };
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
    let mut children = ChildGuard(Vec::new());
    for rank in 0..world {
        let mut command = Command::new(&executable);
        command
            .args([
                "--exact",
                "gemma4_multimodal_pipeline_ring_worker",
                "--nocapture",
            ])
            .env(WORKER, rank.to_string())
            .env(CHECKPOINT, checkpoint.path())
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
        children.0.push(command.spawn().unwrap());
    }
    let deadline = Instant::now() + Duration::from_secs(if tp { 90 } else { 60 });
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
    run_ring(false, false, false);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_dense_stream_pipeline() {
    run_ring(false, true, false);
}

#[test]
#[ignore = "spawns local Ring processes"]
fn ring_gemma4_multimodal_host_tensor_pipeline() {
    run_ring(true, false, true);
}
