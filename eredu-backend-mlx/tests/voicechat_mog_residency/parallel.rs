//! Native MoG tensor parallelism preserves explicit stochastic decisions and bounded leases.
use super::*;
use eredu_backend_mlx::native::materialize_voicechat_mog_parallel;
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_MOG_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_MOG_TP_DIRECTORY";

#[test]
fn native_mog_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(group.size(), 2);
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let coordinator = super::component_coordinator::coordinator(&group, &stream);
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let dimensions = dimensions(&fixture["dimensions"]);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    for f16 in [false, true] {
        let source = checkpoint(
            &fixture,
            f16,
            &directory.path().join(format!("mog_{f16}.safetensors")),
        );
        let (selected, _) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap();
        let (_, _, layout, _, units) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap()
            .0
            .into_parts();
        let local_bytes: Vec<u64> = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect();
        // Three blocks retain two full four-wide norms and three 4x4 local
        // projections each. Prediction is replicated (20 floats); the bank
        // retains two 16-float shards and 20 replicated residual/log-std floats.
        assert_eq!(local_bytes, [224, 224, 224, 80, 208]);
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(materialize_voicechat_mog(
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
        assert!(materialize_voicechat_mog_parallel(
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
            "placement mismatch is rejected before native payload reads"
        );
        let disk_depth = LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(u64::MAX, 0, 0, 0).unwrap(),
        )
        .device_depth(units.len());
        let disk_budget = (0..layout.group_count())
            .flat_map(|group| {
                let bytes = &local_bytes[layout.group_range(group).unwrap()];
                bytes
                    .windows(disk_depth.min(bytes.len()))
                    .map(|window| window.iter().sum::<u64>())
                    .collect::<Vec<_>>()
            })
            .max()
            .unwrap();
        let mut ordinary = f16.then(|| {
            materialize_voicechat_mog(
                prepared(dimensions, source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap()
            .0
        });
        // Conversion is checked once under disk streaming against ordinary
        // execution over exactly the same F16 source values.
        for mode in if f16 { 2..3 } else { 0..3 } {
            let residency = match mode {
                0 => LayerWeightResidency::FullyResident,
                1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(224), Some(1 << 20), 1).unwrap(),
                )),
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(disk_budget, 0, 0, 0).unwrap(),
                ),
            };
            let bindings = prepared(dimensions, source.clone());
            let bindings = if mode == 2 {
                bindings.select_parallel(topology).unwrap().0
            } else {
                bindings
            };
            let (mut model, metadata) = materialize_voicechat_mog_parallel(
                bindings,
                &coordinator,
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), 960);
            assert_eq!(model.static_lease_count(), 0);
            let reads = source.source_diagnostics().unwrap().physical_reads;
            let invalid = MlxTensor::full_f32(0., &[1, 1, dimensions.hidden + 1], &stream).unwrap();
            assert!(model.predict(&invalid, 0., &stream).is_err());
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "invalid predictor input acquires no parameter lease"
            );
            for case_name in ["released_policy", "top_k_without_guidance"] {
                let case = &fixture[case_name];
                let noise = case["policy"]["noise_scale"].as_f64().unwrap() as f32;
                for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
                    let hidden = tensor(&step["input"]);
                    let guidance = step["guidance_scale"].as_f64().unwrap() as f32;
                    let gaussian = tensor(&step["gaussian_draw"]);
                    let prediction = model.predict(&hidden, guidance, &stream).unwrap();
                    let mixture = indices(
                        &step["mixture_indices"],
                        &prediction.hidden.shape()[..2],
                        &stream,
                    );
                    if index == 0 {
                        let reads = source.source_diagnostics().unwrap().physical_reads;
                        let mut shape = prediction.hidden.shape().to_vec();
                        shape[2] = dimensions.latent + 1;
                        let invalid = MlxTensor::full_f32(0., &shape, &stream).unwrap();
                        let draw = if group.rank() == 0 {
                            &invalid
                        } else {
                            &gaussian
                        };
                        assert!(model
                            .selected(&prediction, &mixture, draw, noise, &stream)
                            .is_err());
                        assert_eq!(
                            source.source_diagnostics().unwrap().physical_reads,
                            reads,
                            "invalid Gaussian draw acquires no selected-bank lease"
                        );
                    }
                    let sample = model
                        .selected(&prediction, &mixture, &gaussian, noise, &stream)
                        .unwrap();
                    let actual = [
                        &prediction.hidden,
                        &prediction.logits,
                        &sample.mean,
                        &sample.log_std,
                        &sample.latent,
                    ]
                    .map(|value| value.to_f32_vec(&stream).unwrap());
                    if let Some(ordinary) = ordinary.as_mut() {
                        let prediction = ordinary.predict(&hidden, guidance, &stream).unwrap();
                        let sample = ordinary
                            .selected(&prediction, &mixture, &gaussian, noise, &stream)
                            .unwrap();
                        let expected = [
                            &prediction.hidden,
                            &prediction.logits,
                            &sample.mean,
                            &sample.log_std,
                            &sample.latent,
                        ]
                        .map(|value| value.to_f32_vec(&stream).unwrap());
                        for (actual, expected) in actual.iter().zip(expected) {
                            compare(actual, &serde_json::json!({"values": expected}), &fixture);
                        }
                    } else {
                        for (actual, key) in
                            actual
                                .iter()
                                .zip(["guided_hidden", "logits", "mu", "logs", "latent"])
                        {
                            compare(actual, &step[key], &fixture);
                        }
                    }
                }
            }
            let report = model.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if mode > 0 {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= if mode == 1 { 224 } else { disk_budget }
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
                    "parallel::native_mog_parallel_worker",
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
            "MoG world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_mog_tp2_residency_matches_nvidia_guidance_and_selected_draws() {
    launch(2);
}
