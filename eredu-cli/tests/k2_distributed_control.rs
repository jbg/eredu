//! Application-level validation of the same facade driver on every native rank.
#![cfg(feature = "mlx")]
use eredu::api::*;
use eredu::runtime::chat::ChatTemplateRequest;
use eredu_core::{capture::*, execution_control::*, GenerationConfigOverrides};
use safemlx::{distributed, Device, DeviceType, Stream};
use std::{
    io::Write,
    ops::ControlFlow,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

fn fixture(root: &Path, family: &str) {
    let fixtures: serde_json::Value = serde_json::from_str(include_str!(
        "../../eredu-architectures/tests/fixtures/k2_horizon/reference.json"
    ))
    .unwrap();
    let mut config = fixtures[family]["config"].clone();
    config["num_hidden_layers"] = 3.into();
    config["vocab_size"] = 64.into();
    config.as_object_mut().unwrap().remove("eos_token_id");
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let args = eredu_architectures::k2_horizon::model_args_from_config_value(&config).unwrap();
    let plan = eredu_architectures::k2_horizon::safetensors_plan(&args).unwrap();
    let mut data = vec![];
    let mut header = serde_json::Map::new();
    for tensor in plan.common_tensors.iter().chain(
        plan.layout_groups
            .iter()
            .filter_map(|group| group.variants.first())
            .flat_map(|variant| &variant.tensors),
    ) {
        let start = data.len();
        for index in 0..tensor.shape.iter().product::<usize>() {
            let value = if tensor.key.contains("norm") {
                1.0f32
            } else {
                (((index * 17 + tensor.key.len() * 7) % 101) as f32 - 50.0) * 0.003
            };
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(tensor.key.clone(),serde_json::json!({"dtype":"F32","shape":tensor.shape,"data_offsets":[start,data.len()]}));
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = std::fs::File::create(root.join("model.safetensors")).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(&data).unwrap();
    let vocab = (0..64)
        .map(|id| (format!("word{id}"), serde_json::json!(id)))
        .collect::<serde_json::Map<_, _>>();
    let tokenizer = serde_json::json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":vocab,"unk_token":"word0"}});
    std::fs::write(
        root.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    std::fs::write(
        root.join("chat_template.jinja"),
        "{% for m in messages %}{{ m.content }}{% endfor %}",
    )
    .unwrap();
}

#[test]
#[ignore = "spawns local CPU ranks and opens loopback sockets; run explicitly"]
fn k2_distributed_facade_control_matrix() {
    for family in ["dense", "mova"] {
        for residency in ["resident", "host", "disk"] {
            let root = tempfile::tempdir().unwrap();
            fixture(root.path(), family);
            let world = if family == "dense" { 4 } else { 8 };
            let sockets = (0..world)
                .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
                .collect::<Vec<_>>();
            let hosts = sockets
                .iter()
                .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
                .collect::<Vec<_>>();
            let hostfile = root.path().join("hosts.json");
            std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
            drop(sockets);
            let mut children = vec![];
            for rank in 0..world {
                let log =
                    std::fs::File::create(root.path().join(format!("rank-{rank}.log"))).unwrap();
                children.push(
                    Command::new(std::env::current_exe().unwrap())
                        .args(["--exact", "k2_distributed_facade_worker", "--nocapture"])
                        .env("K2_CONTROL_ARTIFACT", root.path())
                        .env("K2_CONTROL_FAMILY", family)
                        .env("K2_CONTROL_RESIDENCY", residency)
                        .env("MLX_RANK", rank.to_string())
                        .env("MLX_HOSTFILE", &hostfile)
                        .stdout(Stdio::from(log.try_clone().unwrap()))
                        .stderr(Stdio::from(log))
                        .spawn()
                        .unwrap(),
                );
            }
            let deadline = Instant::now() + Duration::from_secs(120);
            while children
                .iter_mut()
                .any(|child| child.try_wait().unwrap().is_none())
                && Instant::now() < deadline
            {
                std::thread::sleep(Duration::from_millis(25));
            }
            let mut failures = vec![];
            for (rank, mut child) in children.into_iter().enumerate() {
                let status = match child.try_wait().unwrap() {
                    Some(status) => status,
                    None => {
                        child.kill().unwrap();
                        child.wait().unwrap()
                    }
                };
                if !status.success() {
                    failures.push(format!(
                        "rank {rank}: {}",
                        std::fs::read_to_string(root.path().join(format!("rank-{rank}.log")))
                            .unwrap()
                    ));
                }
            }
            assert!(
                failures.is_empty(),
                "{family}/{residency}: {}",
                failures.join("\n")
            );
        }
    }
}

#[test]
fn k2_distributed_facade_worker() {
    let Some(root) = std::env::var_os("K2_CONTROL_ARTIFACT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let family = std::env::var("K2_CONTROL_FAMILY").unwrap();
    let residency = std::env::var("K2_CONTROL_RESIDENCY").unwrap();
    let rank = std::env::var("MLX_RANK").unwrap().parse().unwrap();
    let world = distributed::init(true, distributed::Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let backend = eredu_backend_mlx::native::distributed_backend(&stream, &stream, &world);
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(2, 2, if family == "mova" { 2 } else { 1 }, 1).unwrap(),
        rank,
    )
    .unwrap();
    let ordinary = match residency.as_str() {
        "resident" => eredu_runtime::OrdinaryWeightResidency::FullyResident,
        "host" => eredu_runtime::OrdinaryWeightResidency::LayerwiseHost(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
            ),
        ),
        _ => eredu_runtime::OrdinaryWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 1 << 20, 1, 1).unwrap(),
        ),
    };
    let weights = if family == "mova" {
        eredu_runtime::WeightResidency::with_independent_parameter_banks(
            ordinary,
            eredu_runtime::ParameterBankLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1152), Some(1 << 20), 1).unwrap(),
                1152,
                1152,
            )
            .unwrap(),
        )
    } else {
        eredu_runtime::WeightResidency::with_layers(ordinary.layers())
    };
    let options = eredu_backend_mlx::MlxLoadRequest::from_normalized(
        eredu_runtime::NormalizedLoadRequest::default()
            .with_weight_residency(weights)
            .with_state_residency(eredu_runtime::CacheResidencyPolicy::Paged(
                eredu_runtime::PagedCacheOptions::new(2, 32768, 1 << 20, 1)
                    .unwrap()
                    .with_full_attention(true),
            )),
    )
    .with_parallel_topology(
        topology,
        eredu_backend_mlx::native::DeviceAssignment::new(DeviceType::Cpu, 0),
        eredu_runtime::PipelineWireContract::new(eredu_runtime::PipelineActivationDtype::Float32),
        1,
        128,
        eredu_runtime::CommunicationCompletionPolicy::new(
            Duration::from_secs(30),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap(),
    )
    .unwrap();
    let prepared = eredu_core::load_model(&backend, &root, options).unwrap();
    let runtime = eredu_core::ModelRuntime::from_prepared(backend, prepared).unwrap();
    let tokenizer =
        eredu_text::tokenizer::Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        tokenizer,
        LoadedTextModelConfig {
            model_family: eredu_architectures::configuration::ModelKind::K2Horizon,
            effective_model_type: "k2_horizon".into(),
            model_id: "k2-control-fixture".into(),
            chat_template: Some("{% for m in messages %}{{ m.content }}{% endfor %}".into()),
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    );
    for temperature in [0.0, 0.8] {
        let chat = model
            .prepare_chat(ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user","content":"word1 word2"})],
                ..Default::default()
            })
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(temperature),
                max_new_tokens: Some(8),
                ..Default::default()
            },
            seed: 827,
            ..Default::default()
        };
        let trace = TraceLimits {
            per_record_bytes: 16384,
            total_bytes: 1 << 20,
        };
        let prepared = model
            .prepare_observed_chat(&chat, settings, CapturePlan::none(), trace)
            .unwrap();
        let baseline = model
            .generate_tokens(
                prepared.prompt_token_ids().to_vec(),
                eredu_core::TextGenerationConfig::new(prepared.generation_config())
                    .with_seed(settings.seed),
            )
            .unwrap()
            .map(|token| token.unwrap().token_id().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(baseline.len(), 8);
        model.reset().unwrap();
        let emit = |_: ControlledGenerationRecord| ControlFlow::Continue(());
        let mut run = model
            .start_controlled_text(prepared, &[], Default::default(), emit)
            .unwrap();
        run.enable_snapshots(SnapshotLimits {
            max_snapshots: 2,
            max_branches: 1,
            retained_bytes: 64 << 20,
            cumulative_copy_bytes: 256 << 20,
        })
        .unwrap();
        run.step(emit).unwrap();
        run.step(emit).unwrap();
        let saved = run.snapshot(emit).unwrap();
        run.run(emit).unwrap();
        assert_eq!(run.token_ids(), baseline);
        let before = run.snapshot_usage().unwrap().cumulative_copy_bytes;
        run.restore(&saved, emit).unwrap();
        assert!(run.snapshot_usage().unwrap().cumulative_copy_bytes > before);
        run.run(emit).unwrap();
        assert_eq!(run.token_ids(), baseline);
        let mut child = run
            .fork(
                &saved,
                GenerationBranchOptions {
                    trace_limits: trace,
                    capture_limits: None,
                    sampling: None,
                    intervention: None,
                },
                emit,
            )
            .unwrap();
        run.exchange(&mut child, emit).unwrap();
        run.run(emit).unwrap();
        assert_eq!(run.token_ids(), baseline);
        let consumed = run.snapshot_usage().unwrap().cumulative_copy_bytes;
        drop(child);
        assert_eq!(
            run.snapshot_usage().unwrap().cumulative_copy_bytes,
            consumed
        );
        drop(run);
        model.reset().unwrap();
    }
}
