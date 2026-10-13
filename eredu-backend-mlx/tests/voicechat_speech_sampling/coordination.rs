//! One native rank owns refinement draws; every participant retains the same codes.
use super::*;
use eredu_architectures::voicechat::{
    speech_sampling::SpeechSamplingCoordination, tts_transformer::TransformerConfig,
};
use eredu_backend_mlx::native::MlxReplicatedCoordinator;
use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode};
use eredu_runtime::{
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationTensorLimits,
};

const WORKER: &str = "EREDU_SPEECH_SAMPLING_WORLD_SIZE";

fn coordinator(world: &safemlx::distributed::Group, stream: &Stream) -> MlxReplicatedCoordinator {
    let id = CollectiveGroupId::new(97);
    let group = CommunicationGroupDescriptor::new(
        id,
        0,
        (0..world.size()).collect(),
        Some(world.rank()),
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::failure_agreement(true),
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::I32, TensorDtype::F32],
                CommunicationTensorLimits::new(1, 3, 4, None).unwrap(),
                true,
            )
            .unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(world.size(), world.rank(), vec![group], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(10),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    // A nonzero owner exposes accidental assumptions that the first rank draws.
    MlxReplicatedCoordinator::from_manifest(manifest, id, 1, world, stream).unwrap()
}

#[test]
fn native_speech_sampling_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut agreement = coordinator(&world, &stream);
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON).unwrap();
    let mut config = VoiceChatConfig::from_json(eredu_evaluation::voicechat::OFFICIAL_CONFIG_JSON)
        .unwrap()
        .tts;
    config.backbone = geometry(&config.backbone, &fixture["transformer_config"]);
    let transformer: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    config.character_encoder = geometry(
        &config.character_encoder,
        &transformer["t5gemma"]["config"]["encoder"],
    );
    config.latent_size = 4;
    config.codebook_size = 5;
    config.mog.intermediate_size = 12;
    config.mog.num_layers = 3;
    config.mog.low_rank = 2;
    config.mog.num_predictions = 4;
    let mut model = TtsGenerator::<MlxNeuralBackend>::new(&config, 5, 5, &stream).unwrap();
    let mut bind = Bind(&fixture["parameters"], &stream);
    model.conditioning.visit_parameters_mut(&mut bind);
    model.transformer.visit_parameters_mut(&mut bind);
    model.refinement.visit_parameters_mut(&mut bind);
    model.mog.visit_parameters_mut(&mut bind);
    let mut caches = MlxKeyValueState::device(
        eredu_architectures::decoder::state_layout(
            &TransformerConfig::new(&config.backbone).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let first = &fixture["calls"][0];
    let mask: Vec<_> = first["audio_mask"]["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_bool().unwrap())
        .collect();
    model
        .prefill(
            &codes(&first["input_codes"], &stream),
            &PromptLayout::new(1, 5, &mask).unwrap(),
            SpeakerPrompt::Aria,
            Some(&tensor(&first["supplied_subwords"], &stream)),
            caches.as_mut(),
            &stream,
        )
        .unwrap();
    let call = &fixture["calls"][1];
    let frame = model
        .begin_frame(
            &codes(&call["input_codes"], &stream),
            Some(&tensor(&call["supplied_subwords"], &stream)),
            caches.as_mut(),
            &stream,
        )
        .unwrap();
    let state = |owner: bool| {
        Generation::new(
            Host,
            RealtimeGenerationContract::new((), 0),
            Host,
            (),
            vec![],
            owner.then(|| RandomState::with_seed(317).unwrap()),
        )
        .unwrap()
    };
    let mut expected_random = state(true).branch().unwrap();
    let expected = model
        .generate_sampled_frame::<MlxSamplingBackend, _, _, _, _, _>(
            &frame,
            &mut expected_random,
            None,
            &stream,
        )
        .unwrap();
    let expected_codes = expected.codes().unwrap().to_i32_vec(&stream).unwrap();
    let expected_next = expected_random
        .draw_standard_normal::<MlxSamplingBackend>(&[7], &stream)
        .unwrap()
        .to_f32_vec(&stream)
        .unwrap();
    let owner = world.rank() == 1;
    let mut controlled_random = state(owner).branch().unwrap();
    let mut uninterrupted_random = state(owner).branch().unwrap();
    let mut controlled = frame.clone();
    while let Some(slot) = controlled.refinement().next_slot() {
        let mut authority = SpeechSamplingCoordination::new(&mut agreement, [31, 7, slot as u64]);
        controlled = model
            .advance_sampled_frame::<MlxSamplingBackend, _, _, _, _, _>(
                &controlled,
                &mut controlled_random,
                Some(&mut authority),
                &stream,
            )
            .unwrap();
    }
    let mut authority = SpeechSamplingCoordination::new(&mut agreement, [31, 7, 8]);
    let uninterrupted = model
        .generate_sampled_frame::<MlxSamplingBackend, _, _, _, _, _>(
            &frame,
            &mut uninterrupted_random,
            Some(&mut authority),
            &stream,
        )
        .unwrap();
    for output in [&controlled, &uninterrupted] {
        assert_eq!(
            output.codes().unwrap().to_i32_vec(&stream).unwrap(),
            expected_codes
        );
    }
    for branch in [&mut controlled_random, &mut uninterrupted_random] {
        let next = branch.draw_standard_normal::<MlxSamplingBackend>(&[7], &stream);
        if owner {
            assert_eq!(next.unwrap().to_f32_vec(&stream).unwrap(), expected_next);
        } else {
            assert!(next.is_err(), "follower retains no random stream");
        }
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_speech_sampling_owner_draws_match_controlled_and_uninterrupted() {
    use std::{
        net::TcpListener,
        process::{Child, Command, Stdio},
        time::{Duration, Instant},
    };
    struct Children(Vec<Child>);
    impl Drop for Children {
        fn drop(&mut self) {
            for child in &mut self.0 {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
    let sockets = (0..2)
        .map(|_| TcpListener::bind(("127.0.0.1", 0)).unwrap())
        .collect::<Vec<_>>();
    let hosts = sockets
        .iter()
        .map(|socket| vec![format!("127.0.0.1:{}", socket.local_addr().unwrap().port())])
        .collect::<Vec<_>>();
    let directory = tempfile::tempdir().unwrap();
    let hostfile = directory.path().join("hosts.json");
    std::fs::write(&hostfile, serde_json::to_vec(&hosts).unwrap()).unwrap();
    drop(sockets);
    let mut children = Children(Vec::new());
    for rank in 0..2 {
        children.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "coordination::native_speech_sampling_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(WORKER, "2")
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(90);
    loop {
        let statuses = children
            .0
            .iter_mut()
            .map(|child| child.try_wait().unwrap())
            .collect::<Vec<_>>();
        if statuses.iter().all(Option::is_some) {
            break;
        }
        if Instant::now() >= deadline || statuses.iter().flatten().any(|status| !status.success()) {
            for child in &mut children.0 {
                let _ = child.kill();
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let outputs = children
        .0
        .drain(..)
        .map(Child::wait_with_output)
        .collect::<Vec<_>>();
    assert!(
        outputs
            .iter()
            .all(|output| output.as_ref().is_ok_and(|output| output.status.success())),
        "speech sampling rank outputs:\n{}",
        outputs
            .iter()
            .enumerate()
            .map(|(rank, output)| match output {
                Ok(output) => format!(
                    "rank {rank}: {}\n{}\n{}",
                    output.status,
                    String::from_utf8_lossy(&output.stdout),
                    String::from_utf8_lossy(&output.stderr)
                ),
                Err(error) => format!("rank {rank}: failed to reap worker: {error}"),
            })
            .collect::<Vec<_>>()
            .join("\n")
    );
}
