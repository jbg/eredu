//! Native vocabulary shards preserve RNNT neural phases and replicated recurrence.
use super::*;
use eredu_backend_mlx::native::{materialize_voicechat_rnnt_parallel, MlxReplicatedCoordinator};
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_RNNT_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_RNNT_TP_DIRECTORY";

fn uneven_fixture(original: &Value) -> Value {
    let mut fixture = original.clone();
    fixture["geometry"]["vocab_size"] = 4.into();
    // Insert an unused, low-score symbol immediately before blank. The altered
    // checkpoint is compared with ordinary execution over these exact values.
    for (name, width, added) in [
        ("stt_model.rnnt_decoder.prediction.embed.weight", 2, 0.),
        ("stt_model.rnnt_joint.joint_net.2.weight", 3, 0.),
        ("stt_model.rnnt_joint.joint_net.2.bias", 1, -100.),
    ] {
        let parameter = &mut fixture["parameters"][name];
        parameter["shape"][0] = 5.into();
        let values = parameter["values"].as_array_mut().unwrap();
        values.splice(3 * width..3 * width, vec![serde_json::json!(added); width]);
    }
    fixture
}

fn compare_trace(actual: &Trace, expected: &Trace, fixture: &Value) {
    assert_eq!(actual.decisions, expected.decisions);
    assert_eq!(actual.states.len(), expected.states.len());
    let compare_values = |actual: &[f32], expected: &[f32]| {
        assert_eq!(actual.len(), expected.len());
        let abs = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
        let rel = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= abs + rel * e.abs(),
                "{index}: {a} versus {e}"
            );
        }
    };
    for ((actual, ah, af), (expected, eh, ef)) in actual.states.iter().zip(&expected.states) {
        assert_eq!((ah, af), (eh, ef));
        assert_eq!(actual.len(), expected.len());
        for ((shape, dtype, a), (eshape, edtype, e)) in actual.iter().zip(expected) {
            assert_eq!((shape, dtype), (eshape, edtype));
            compare_values(a, e);
        }
    }
    assert_eq!(actual.logits.len(), expected.logits.len());
    for (a, e) in actual.logits.iter().zip(&expected.logits) {
        compare_values(a, e);
    }
}

fn coordinator(world: &safemlx::distributed::Group, stream: &Stream) -> MlxReplicatedCoordinator {
    use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode};
    use eredu_runtime::{
        CommunicationCompletionPolicy, CommunicationGroupDescriptor,
        CommunicationGroupRequirements, CommunicationManifest, CommunicationOperation,
        CommunicationOperationRequirement, CommunicationTensorLimits,
    };
    let tensor_requirement = |operation| {
        CommunicationOperationRequirement::tensors(
            operation,
            [TensorDtype::F32],
            CommunicationTensorLimits::new(1, 3, 64, None).unwrap(),
            true,
        )
        .unwrap()
    };
    let id = CollectiveGroupId::new(71);
    let group = CommunicationGroupDescriptor::new(
        id,
        0,
        vec![0, 1],
        Some(world.rank()),
        CommunicationGroupRequirements::new([
            CommunicationOperationRequirement::failure_agreement(true),
            tensor_requirement(CommunicationOperation::AllReduceSum),
            tensor_requirement(CommunicationOperation::AllGatherEven),
            tensor_requirement(CommunicationOperation::AllGatherUneven),
            CommunicationOperationRequirement::tensors(
                CommunicationOperation::Broadcast,
                [TensorDtype::I32],
                CommunicationTensorLimits::new(1, 1, 4, None).unwrap(),
                true,
            )
            .unwrap(),
        ])
        .unwrap(),
    )
    .unwrap();
    let manifest = CommunicationManifest::new(2, world.rank(), vec![group], vec![])
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(5),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    MlxReplicatedCoordinator::from_manifest(manifest, id, 1, world, stream).unwrap()
}

fn coordinated_frames(
    model: &mut MlxRnnt,
    coordinator: &mut MlxReplicatedCoordinator,
    fixture: &Value,
    world: &safemlx::distributed::Group,
    stream: &Stream,
) {
    use eredu_architectures::voicechat::rnnt_coordination::decode_frame;
    use eredu_backend_mlx::backend::nn::shared::MlxNeuralBackend;
    let limits = RnntLimits {
        max_symbols_per_frame: 3,
        hypothesis_capacity: 32,
    };
    let mut state = RnntState::default();
    for (index, frame) in fixture["streaming_frames"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let encoder = tensor(&frame["encoder"], model.dimensions().encoder_width);
        let before = retained(&state, stream);
        let candidate = decode_frame::<MlxNeuralBackend, _, _>(
            model,
            coordinator,
            [81, 9, index as u64],
            &encoder,
            &state,
            limits,
            stream,
        )
        .unwrap();
        assert_eq!(retained(&state, stream), before);
        let expected = frame["emitted"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        assert_eq!(candidate.emitted, expected);
        assert_eq!(
            candidate.first_blank,
            frame["first_blank"].as_bool().unwrap()
        );
        assert_eq!(
            candidate.ended_blank,
            frame["ended_blank"].as_bool().unwrap()
        );
        assert_eq!(
            candidate.hit_symbol_limit,
            frame["hit_symbol_limit"].as_bool().unwrap()
        );
        compare_state(&candidate.state, frame, fixture, stream);
        let replay = decode_frame::<MlxNeuralBackend, _, _>(
            model,
            coordinator,
            [81, 10, index as u64],
            &encoder,
            &state.clone(),
            limits,
            stream,
        )
        .unwrap();
        assert_eq!(replay.emitted, candidate.emitted);
        assert_eq!(
            retained(&replay.state, stream),
            retained(&candidate.state, stream)
        );
        state = candidate.state;
    }
    let reads = model
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let encoder = tensor(
        &fixture["streaming_frames"][0]["encoder"],
        model.dimensions().encoder_width,
    );
    let before = retained(&state, stream);
    assert!(decode_frame::<MlxNeuralBackend, _, _>(
        model,
        coordinator,
        [81, 11, world.rank() as u64],
        &encoder,
        &state,
        limits,
        stream
    )
    .is_err());
    assert_eq!(
        model
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        reads,
        "descriptor mismatch precedes every RNNT parameter lease"
    );
    assert_eq!(retained(&state, stream), before);
    // Each rank claims authority over the same selected group. The initial
    // coordinator handshake must reject this before either can advance RNNT.
    let mut conflicting = MlxReplicatedCoordinator::from_manifest(
        coordinator.manifest().clone(),
        coordinator.group(),
        world.rank(),
        world,
        stream,
    )
    .unwrap();
    assert!(decode_frame::<MlxNeuralBackend, _, _>(
        model,
        &mut conflicting,
        [81, 12, 0],
        &encoder,
        &state,
        limits,
        stream,
    )
    .is_err());
    assert_eq!(
        model
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        reads,
        "conflicting owners cannot acquire RNNT parameters"
    );
    assert_eq!(retained(&state, stream), before);
}

#[test]
fn native_rnnt_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(group.size(), 2);
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut coordinator = coordinator(&group, &stream);
    let original: Value =
        serde_json::from_str(eredu_evaluation::voicechat::RNNT_REFERENCE_JSON).unwrap();
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    for (uneven, f16) in [(true, false), (true, true), (false, false)] {
        let fixture = if uneven {
            uneven_fixture(&original)
        } else {
            original.clone()
        };
        let dimensions = dimensions(&fixture);
        let source = checkpoint(
            &fixture,
            f16,
            &directory
                .path()
                .join(format!("rnnt_{uneven}_{f16}.safetensors")),
        );
        let (mut ordinary, _) = materialize_voicechat_rnnt(
            prepared(dimensions, source.clone()),
            LayerWeightResidency::FullyResident,
            &weights,
            &stream,
        )
        .unwrap();
        let baseline = run(&mut ordinary, &fixture, f16 || uneven, &stream);
        let (selected, _) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap();
        // Embedding/output rows shard; both LSTM layers and encoder/predictor
        // affines remain complete replicas. Uneven ownership is [3, 2].
        let rows = if uneven && group.rank() == 0 { 3 } else { 2 };
        let bytes = selected
            .unit_bindings()
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(bytes, [rows * 8, 192, 192, 48, 36, rows * 16]);
        assert!(matches!(
            selected.source().source_metadata("unrelated.weight"),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(materialize_voicechat_rnnt(
            selected,
            LayerWeightResidency::FullyResident,
            &weights,
            &stream
        )
        .is_err());
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        let wrong =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1 - group.rank())
                .unwrap();
        let (wrong, _) = prepared(dimensions, source.clone())
            .select_parallel(wrong)
            .unwrap();
        assert!(materialize_voicechat_rnnt_parallel(
            wrong,
            &coordinator,
            LayerWeightResidency::FullyResident,
            &weights,
            &stream
        )
        .is_err());
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            reads,
            "wrong rank is rejected before acquisition"
        );
        let modes = if f16 {
            vec![2]
        } else if uneven {
            vec![0]
        } else {
            vec![0, 1, 2]
        };
        for mode in modes {
            // Disk prefetch overlaps the two replicated predictor layers.
            let budget = if mode == 2 { 384 } else { 192 };
            let residency = match mode {
                0 => LayerWeightResidency::FullyResident,
                1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(budget), Some(1 << 20), 1).unwrap(),
                )),
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(budget, 0, 0, 0).unwrap(),
                ),
            };
            let bindings = prepared(dimensions, source.clone());
            let bindings = if mode == 2 {
                bindings.select_parallel(topology).unwrap().0
            } else {
                bindings
            };
            let (mut model, metadata) = materialize_voicechat_rnnt_parallel(
                bindings,
                &coordinator,
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.layer_parameter_bytes(), bytes.iter().sum::<u64>());
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(model.static_lease_count(), 0);
            // Identical rank-local inputs exercise primitive parity here; the
            // shared coordinated driver separately owns distributed decisions.
            let actual = run(&mut model, &fixture, f16 || uneven, &stream);
            compare_trace(&actual, &baseline, &fixture);
            if !uneven && mode == 2 {
                coordinated_frames(&mut model, &mut coordinator, &fixture, &group, &stream);
            }
            let report = model.residency_report().unwrap();
            if mode > 0 {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= budget
                );
            }
            if let Some(report) = model.dense_stream_report().unwrap() {
                assert!(report
                    .execution_groups()
                    .iter()
                    .all(|group| group.completed_executions() > 0));
            }
        }
    }
}

fn launch() {
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
                    "parallel::native_rnnt_parallel_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(WORKER, "2")
                .env(DIRECTORY, directory.path())
                .env("MLX_RANK", rank.to_string())
                .env("MLX_HOSTFILE", &hostfile)
                .env_remove("MLX_RING_VERBOSE")
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap(),
        );
    }
    let deadline = Instant::now() + Duration::from_secs(120);
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
    // Reap every child before inspecting any failure; a panic cannot detach the
    // remaining ranks or leave their temporary checkpoints alive.
    let outputs = children
        .0
        .drain(..)
        .map(Child::wait_with_output)
        .collect::<Vec<_>>();
    for (rank, output) in outputs.into_iter().enumerate() {
        let output = output.unwrap();
        assert!(
            output.status.success(),
            "RNNT rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_rnnt_tp2_residency_matches_nvidia_and_uneven_vocabulary() {
    launch();
}
