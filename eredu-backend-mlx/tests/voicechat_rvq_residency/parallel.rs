//! Latent-sharded RVQ executes the same explicit refinement transition on native ranks.
use super::*;
use eredu_backend_mlx::native::materialize_voicechat_rvq_parallel;
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_RVQ_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_RVQ_TP_DIRECTORY";

#[test]
fn native_rvq_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(group.size(), 2);
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let coordinator = component_coordinator::coordinator(&group, &stream);
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let dimensions = dimensions(&fixture["dimensions"]);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    let case = &fixture["released_policy"];
    let hidden = tensor(&case["hidden"]);
    let conditional = hidden.index(&[Index::Range(0, 1)], &stream).unwrap();
    let unconditional = hidden.index(&[Index::Range(1, 2)], &stream).unwrap();
    let policy = RefinementPolicy {
        guidance_scale: 0.2,
        noise_scale: 0.001,
        top_p: 0.95,
    };
    for f16 in [false, true] {
        let source = checkpoint(
            &fixture,
            f16,
            &directory.path().join(format!("rvq_{f16}.safetensors")),
        );
        let (selected, _) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap();
        let (_, _, _, _, units) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap()
            .0
            .into_parts();
        let local_bytes: Vec<u64> = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect();
        // Each rank retains two latent coordinates of all 31 five-entry
        // codebooks and the matching two columns of the four-output projection.
        assert_eq!(local_bytes, [1240, 32]);
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(materialize_voicechat_rvq(
            selected,
            LayerWeightResidency::FullyResident,
            &weights,
            &stream
        )
        .is_err());
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, reads);
        let wrong_topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1 - group.rank())
                .unwrap();
        let wrong = prepared(dimensions, source.clone())
            .select_parallel(wrong_topology)
            .unwrap()
            .0;
        assert!(materialize_voicechat_rvq_parallel(
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
            "rank mismatch reads no native payload"
        );
        let mut ordinary = f16.then(|| {
            materialize_voicechat_rvq(
                prepared(dimensions, source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap()
            .0
        });
        for mode in if f16 { 2..3 } else { 0..3 } {
            let residency = match mode {
                0 => LayerWeightResidency::FullyResident,
                1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(1240), Some(1 << 20), 1).unwrap(),
                )),
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(1240, 0, 0, 0).unwrap(),
                ),
            };
            let bindings = prepared(dimensions, source.clone());
            let bindings = if mode == 2 {
                bindings.select_parallel(topology).unwrap().0
            } else {
                bindings
            };
            let (mut model, metadata) = materialize_voicechat_rvq_parallel(
                bindings,
                &coordinator,
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), 1272);
            assert_eq!(model.static_lease_count(), 0);
            let mut state = model.start(1, 1, &stream).unwrap();
            let mut ordinary_state = ordinary.as_ref().map(|m| m.start(1, 1, &stream).unwrap());
            let reads = source.source_diagnostics().unwrap().physical_reads;
            let invalid_latent =
                MlxTensor::full_f32(0., &[1, 1, dimensions.latent - 1], &stream).unwrap();
            assert!(matches!(
                model.advance(&state, &invalid_latent, &stream),
                Err(TtsError::Shape(_))
            ));
            let invalid_codes =
                MlxTensor::full_i32(dimensions.cardinality, &[1, 1, 30], &stream).unwrap();
            assert!(matches!(
                model.depthsum(&invalid_codes, &stream),
                Err(TtsError::Shape(_))
            ));
            assert!(matches!(
                model.embed_codes(&invalid_codes, &stream),
                Err(TtsError::Shape(_))
            ));
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "invalid geometry acquires no parameter lease"
            );
            assert_eq!(
                model
                    .depthsum(state.codes(), &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap(),
                vec![0.; dimensions.latent as usize]
            );
            for step in case["steps"].as_array().unwrap() {
                let parent = state.clone();
                let prefix = parent.codes().to_i32_vec(&stream).unwrap();
                let embedded = model
                    .embed_codes(state.codes(), &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap();
                let input = model
                    .input(&state, &conditional, Some(&unconditional), policy, &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap();
                state = model
                    .advance(&state, &tensor(&step["latent"]), &stream)
                    .unwrap();
                let after_codes = state.codes().to_i32_vec(&stream).unwrap();
                let sum = model
                    .depthsum(state.codes(), &stream)
                    .unwrap()
                    .to_f32_vec(&stream)
                    .unwrap();
                assert_eq!(
                    parent.codes().to_i32_vec(&stream).unwrap(),
                    prefix,
                    "parent remains unchanged"
                );
                assert_eq!(
                    &after_codes[..parent.completed_depth()],
                    &prefix[..parent.completed_depth()]
                );
                if let Some(ordinary) = ordinary.as_mut() {
                    let baseline = ordinary_state.as_mut().unwrap();
                    let expected_embedded = ordinary
                        .embed_codes(baseline.codes(), &stream)
                        .unwrap()
                        .to_f32_vec(&stream)
                        .unwrap();
                    let expected_input = ordinary
                        .input(
                            baseline,
                            &conditional,
                            Some(&unconditional),
                            policy,
                            &stream,
                        )
                        .unwrap()
                        .to_f32_vec(&stream)
                        .unwrap();
                    *baseline = ordinary
                        .advance(baseline, &tensor(&step["latent"]), &stream)
                        .unwrap();
                    let expected_sum = ordinary
                        .depthsum(baseline.codes(), &stream)
                        .unwrap()
                        .to_f32_vec(&stream)
                        .unwrap();
                    assert_eq!(after_codes, baseline.codes().to_i32_vec(&stream).unwrap());
                    for (actual, expected) in [
                        (&embedded, expected_embedded),
                        (&input, expected_input),
                        (&sum, expected_sum),
                    ] {
                        compare(actual, &serde_json::json!({"values": expected}), &fixture);
                    }
                } else {
                    assert_eq!(after_codes, codes(&step["codes_after"]));
                    compare(&sum, &step["summed_feedback"], &fixture);
                    compare(&input, &step["input"], &fixture);
                    // The reference's conditional input is its hidden state plus
                    // code feedback, so subtraction exposes an independent target.
                    let expected_embedding: Vec<f32> = floats(&step["input"])
                        .into_iter()
                        .zip(floats(&case["hidden"]))
                        .take(dimensions.hidden as usize)
                        .map(|(input, hidden)| input - hidden)
                        .collect();
                    compare(
                        &embedded,
                        &serde_json::json!({"values": expected_embedding}),
                        &fixture,
                    );
                }
            }
            assert_eq!(state.completed_depth(), 31);
            assert_eq!(state.next_slot(), None);
            let report = model.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if mode > 0 {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= 1240
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

fn launch(size: usize) {
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
    let sockets = (0..size)
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
    for rank in 0..size {
        children.0.push(
            Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "parallel::native_rvq_parallel_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(WORKER, size.to_string())
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
    for (rank, child) in children.0.drain(..).enumerate() {
        let output = child.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "RVQ world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_rvq_tp2_residency_matches_nvidia_prefixes_and_feedback() {
    launch(2);
}
