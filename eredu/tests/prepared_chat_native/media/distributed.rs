//! The same public image/tool request through actual tensor and pipeline ranks.
use super::*;
use safemlx::{DeviceType, distributed};
use std::{
    ops::ControlFlow,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[test]
#[ignore = "spawns two native Metal ranks and opens loopback sockets; run explicitly"]
fn two_rank_image_tools_preserve_ordinary_manual_and_recorded_output() {
    image_tools_topology(2, 1);
}

#[test]
#[ignore = "spawns two native Metal ranks and opens loopback sockets; run explicitly"]
fn two_rank_pipeline_image_tools_preserve_ordinary_manual_and_recorded_output() {
    image_tools_topology(1, 2);
}

#[test]
#[ignore = "spawns four native Metal ranks and opens loopback sockets; run explicitly"]
fn four_rank_tensor_pipeline_image_tools_preserve_ordinary_manual_and_recorded_output() {
    image_tools_topology(2, 2);
}

fn image_tools_topology(tensor: usize, pipeline: usize) {
    let ranks = tensor * pipeline;
    let (root, _, _) = fixture(true);
    let sockets = (0..ranks)
        .map(|_| std::net::TcpListener::bind("127.0.0.1:0").unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|s| vec![format!("127.0.0.1:{}", s.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let hostfile = root.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut children = (0..ranks)
        .map(|rank| {
            let log = std::fs::File::create(root.path().join(format!("rank-{rank}.log"))).unwrap();
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "media::distributed::image_tools_rank",
                    "--nocapture",
                ])
                .env("EREDU_IMAGE_TOOLS_ARTIFACT", root.path())
                .env("EREDU_IMAGE_TOOLS_TENSOR", tensor.to_string())
                .env("EREDU_IMAGE_TOOLS_PIPELINE", pipeline.to_string())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .stdout(Stdio::from(log.try_clone().unwrap()))
                .stderr(Stdio::from(log))
                .spawn()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(300);
    while children.iter_mut().any(|c| c.try_wait().unwrap().is_none()) && Instant::now() < deadline
    {
        std::thread::sleep(Duration::from_millis(25));
    }
    let mut failures = Vec::new();
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
                "TP{tensor}/PP{pipeline} rank {rank}: {}",
                std::fs::read_to_string(root.path().join(format!("rank-{rank}.log"))).unwrap()
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    let rank0: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.path().join("output-0.json")).unwrap()).unwrap();
    for rank in 1..ranks {
        let result: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.path().join(format!("output-{rank}.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(
            rank0, result,
            "TP{tensor}/PP{pipeline}: rank {rank} must expose the same committed semantic result"
        );
    }
}

#[test]
fn image_tools_rank() {
    let Some(root) = std::env::var_os("EREDU_IMAGE_TOOLS_ARTIFACT") else {
        return;
    };
    let root = std::path::PathBuf::from(root);
    let rank: usize = std::env::var("MLX_RANK").unwrap().parse().unwrap();
    let tensor: usize = std::env::var("EREDU_IMAGE_TOOLS_TENSOR")
        .unwrap()
        .parse()
        .unwrap();
    let pipeline: usize = std::env::var("EREDU_IMAGE_TOOLS_PIPELINE")
        .unwrap()
        .parse()
        .unwrap();
    let world = distributed::init(true, distributed::Backend::Ring).unwrap();
    let backend =
        eredu_backend_mlx::native::prepared_distributed_backend_on(&world, DeviceType::Gpu)
            .unwrap_or_else(fail)
            .expect("qualified native stream constructor");
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(tensor, pipeline, 1, 1).unwrap(),
        rank,
    )
    .unwrap();
    let options = eredu_backend_mlx::MlxLoadRequest::from_normalized(Default::default())
        .with_parallel_topology(
            topology,
            eredu_backend_mlx::native::DeviceAssignment::new(DeviceType::Gpu, 0),
            eredu_runtime::PipelineWireContract::new(
                eredu_runtime::PipelineActivationDtype::Float32,
            ),
            1,
            2048,
            eredu_runtime::CommunicationCompletionPolicy::new(
                Duration::from_secs(30),
                eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        )
        .unwrap_or_else(fail);
    eprintln!("DISTRIBUTED_MEDIA TP{tensor}/PP{pipeline} rank={rank} load");
    let prepared = eredu_core::load_model(&backend, &root, options).unwrap_or_else(fail);
    let runtime = eredu_core::ModelRuntime::from_prepared(backend, prepared).unwrap_or_else(fail);
    let tokenizer = Tokenizer::from_file(root.join("tokenizer.json")).unwrap();
    let script = PIECES.map(|piece| tokenizer.token_to_id(piece).unwrap());
    let eos = tokenizer.token_to_id("<|im_end|>").unwrap();
    let mut model = LoadedModel::from_runtime(
        runtime,
        eredu_text::tokenizer::Tokenizer::from_file(root.join("tokenizer.json")).unwrap(),
        LoadedTextModelConfig {
            model_family: eredu_architectures::configuration::ModelKind::Qwen3Vl,
            effective_model_type: "qwen3_vl".into(),
            model_id: "image-tools-fixture".into(),
            chat_template: Some(
                include_str!("../../fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja")
                    .into(),
            ),
            eos_token_ids: vec![eos],
            checkpoint_generation_config: None,
        },
    )
    .unwrap_or_else(fail);
    let cancel = GenerationCancellationToken::new();
    let tokens = model
        .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
        .unwrap_or_else(fail);
    let source = model
        .compile_managed_chat_source(
            &tokens,
            ChatSourceInput::RetainedConfiguration,
            true,
            &cancel,
        )
        .unwrap_or_else(fail)
        .unwrap();
    let mut results = Vec::new();
    for choice in [ToolChoice::Required, ToolChoice::Auto] {
        eprintln!(
            "DISTRIBUTED_MEDIA TP{tensor}/PP{pipeline} rank={rank} choice={choice:?} source policy"
        );
        let chat = model
            .prepare_chat(&source, &image_policy(choice), CAPACITY, &cancel)
            .unwrap_or_else(fail)
            .unwrap();
        let encoded = tokenizer.encode(chat.rendered_prompt(), false).unwrap();
        let image_id = tokenizer.token_to_id("<|image_pad|>").unwrap();
        let marker = encoded
            .get_ids()
            .iter()
            .position(|&id| id == image_id)
            .unwrap();
        assert_eq!(
            encoded
                .get_ids()
                .iter()
                .filter(|&&id| id == image_id)
                .count(),
            1
        );
        let image = ImagePrompt::new(
            &encoded.get_ids()[..marker],
            &encoded.get_ids()[marker + 1..],
        );
        let parts = image.parts();
        let positions = encoded.len() as u64 + 3;
        let chunk = if positions % 17 == 0 { 19 } else { 17 };
        assert!(positions > chunk && positions % chunk != 0);
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                temperature: Some(0.0),
                max_new_tokens: Some(8),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                managed_memory_capacity_bytes: Some(CAPACITY),
                prefill_chunk_positions: NonZeroU64::new(chunk),
                ..Default::default()
            },
            seed: 37,
            ..Default::default()
        };
        let mut expected = None;
        for mode in 0..3 {
            eprintln!(
                "DISTRIBUTED_MEDIA TP{tensor}/PP{pipeline} rank={rank} choice={choice:?} mode={mode} prepare/reset"
            );
            if !results.is_empty() || mode != 0 {
                model
                    .prepare_reset_ordinary()
                    .unwrap_or_else(fail)
                    .reset_admitted(eredu_core::SessionResetLimits::new(CAPACITY))
                    .unwrap_or_else(fail);
                model.synchronize().unwrap_or_else(fail);
            }
            let input = model
                .prepare_chat_input(&chat, &parts, &cancel)
                .unwrap_or_else(fail)
                .unwrap();
            let mut request = PreparedChatRequest::new(&chat, settings.clone());
            request.input = PreparedChatPrompt::Media(input);
            eprintln!(
                "DISTRIBUTED_MEDIA TP{tensor}/PP{pipeline} rank={rank} choice={choice:?} mode={mode} start"
            );
            let mut events = Vec::new();
            let (ids, finish) = if mode == 2 {
                let mut emit = |record: ControlledGenerationRecord| {
                    if let Some(ObservedGenerationEvent::Semantic { event, .. }) =
                        record.event.progress()
                    {
                        events.push(event.clone());
                    }
                    ControlFlow::Continue(())
                };
                let mut session = model
                    .start_controlled_chat(
                        request,
                        TraceLimits {
                            per_record_bytes: 65536,
                            total_bytes: 1 << 20,
                        },
                        eredu_core::execution_control::GenerationControlHandle::default(),
                        &mut emit,
                    )
                    .unwrap_or_else(fail)
                    .unwrap();
                session.run(&mut emit).unwrap_or_else(fail);
                (
                    session.token_ids().to_vec(),
                    session.finish_reason().unwrap(),
                )
            } else {
                let mut session = model
                    .start_prepared_chat(request, &cancel)
                    .unwrap_or_else(fail)
                    .unwrap();
                let attribution = session
                    .prompt_attribution()
                    .expect("original image attribution");
                assert_eq!(attribution.attribution().decoder_positions, positions);
                assert_eq!(
                    attribution.attribution().segments[1].plan.decoder_range,
                    [marker as u64, marker as u64 + 4]
                );
                assert!(attribution.attribution().complete_token_ids().is_none());
                let output = if mode == 0 {
                    session
                        .run(&cancel, &mut |event| events.push(event))
                        .unwrap_or_else(fail)
                } else {
                    while session.finish_reason().is_none() {
                        session = session
                            .advance(&cancel, &mut |event| events.push(event))
                            .unwrap_or_else(fail);
                    }
                    session
                        .into_output()
                        .unwrap_or_else(|_| panic!("terminal image session"))
                };
                (output.token_ids.to_vec(), output.finish_reason)
            };
            assert!(ids.starts_with(&script));
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                    .count(),
                1
            );
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
                    .count(),
                1
            );
            let result = serde_json::json!({"ids":ids,"finish":finish,"events":events});
            if let Some(expected) = &expected {
                assert_eq!(&result, expected, "rank {rank}, {choice:?}, mode {mode}");
            } else {
                expected = Some(result);
            }
        }
        results.push(expected.unwrap());
    }
    std::fs::write(
        root.join(format!("output-{rank}.json")),
        serde_json::to_vec(&results).unwrap(),
    )
    .unwrap();
    drop((source, tokens, model));
    eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend::reclaim_retired_resources();
}
