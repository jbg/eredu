//! Codec channel shards preserve global residual decisions and selected-depth acquisition.
use super::*;
#[path = "../support/component_coordinator.rs"]
mod component_coordinator;
use eredu_backend_mlx::native::materialize_prvq_parallel;
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_PRVQ_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_PRVQ_TP_DIRECTORY";

#[test]
fn native_prvq_parallel_worker() {
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
        serde_json::from_str(eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON).unwrap();
    let geometry = Geometry {
        depth: 3,
        cardinality: 5,
        channels: 4,
    };
    let latent = tensor(&fixture["input"]);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    for f16 in [false, true] {
        let source = checkpoint(
            &fixture,
            f16,
            &directory.path().join(format!("prvq_{f16}.safetensors")),
        );
        let (selected, _) = prepared(geometry, source.clone())
            .select_parallel(topology)
            .unwrap();
        let (_, _, _, _, _, units) = prepared(geometry, source.clone())
            .select_parallel(topology)
            .unwrap()
            .0
            .into_parts();
        let local_bytes: Vec<u64> = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect();
        // Five two-channel means and a replicated scalar variance per selected depth.
        assert_eq!(local_bytes, [44, 44, 44]);
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(materialize_prvq(
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
        let wrong = prepared(geometry, source.clone())
            .select_parallel(wrong_topology)
            .unwrap()
            .0;
        assert!(materialize_prvq_parallel(
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
            materialize_prvq(
                prepared(geometry, source.clone()),
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
                    OffloadConfig::new(Some(44), Some(1 << 20), 1).unwrap(),
                )),
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(44, 0, 0, 0).unwrap(),
                ),
            };
            let bindings = prepared(geometry, source.clone());
            let bindings = if mode == 2 {
                bindings.select_parallel(topology).unwrap().0
            } else {
                bindings
            };
            let (mut model, metadata) =
                materialize_prvq_parallel(bindings, &coordinator, residency, &weights, &stream)
                    .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), 132);
            assert_eq!(model.static_lease_count(), 0);
            let reads = source.source_diagnostics().unwrap().physical_reads;
            assert!(matches!(
                model.quantize_range(&latent, 1..4, DistanceMetric::SquaredEuclidean, &stream),
                Err(eredu_codec::Error::InvalidShape(_))
            ));
            let invalid_latent = MlxTensor::full_f32(0., &[1, 1, 3], &stream).unwrap();
            assert!(matches!(
                model.quantize(&invalid_latent, &stream),
                Err(eredu_codec::Error::InvalidShape(_))
            ));
            let invalid_codes = MlxTensor::full_i32(0, &[1, 1, 2], &stream).unwrap();
            assert!(matches!(
                model.decode_range(&invalid_codes, 1..2, &stream),
                Err(eredu_codec::Error::InvalidShape(_))
            ));
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "invalid ingress acquires no codebook"
            );
            let middle = model
                .quantize_range(
                    &latent,
                    1..2,
                    DistanceMetric::CodebookNormMinusTwiceDot,
                    &stream,
                )
                .unwrap();
            let middle_decoded = model.decode_range(&middle.codes, 1..2, &stream).unwrap();
            if let Some(report) = model.dense_stream_report().unwrap() {
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.completed_executions() > 0)
                        .count(),
                    1
                );
                assert_eq!(
                    report
                        .execution_groups()
                        .iter()
                        .filter(|group| group.peak_device_bytes() == 0)
                        .count(),
                    2,
                    "unselected depths remain unloaded"
                );
            }
            let full = model.quantize(&latent, &stream).unwrap();
            let decoded = model.decode(&full.codes, &stream).unwrap();
            let prefix_codes = full
                .codes
                .index(&[Index::Full, Index::Full, Index::Range(0, 2)], &stream)
                .unwrap();
            let prefix = model.decode_range(&prefix_codes, 0..2, &stream).unwrap();
            if let Some(ordinary) = ordinary.as_mut() {
                let expected_middle = ordinary
                    .quantize_range(
                        &latent,
                        1..2,
                        DistanceMetric::CodebookNormMinusTwiceDot,
                        &stream,
                    )
                    .unwrap();
                let expected_middle_decoded = ordinary
                    .decode_range(&expected_middle.codes, 1..2, &stream)
                    .unwrap();
                let expected_full = ordinary.quantize(&latent, &stream).unwrap();
                let expected_decoded = ordinary.decode(&expected_full.codes, &stream).unwrap();
                let expected_prefix = ordinary.decode_range(&prefix_codes, 0..2, &stream).unwrap();
                assert_eq!(
                    middle.codes.to_i32_vec(&stream).unwrap(),
                    expected_middle.codes.to_i32_vec(&stream).unwrap()
                );
                assert_eq!(
                    full.codes.to_i32_vec(&stream).unwrap(),
                    expected_full.codes.to_i32_vec(&stream).unwrap()
                );
                for (actual, expected) in [
                    (&middle.residual, &expected_middle.residual),
                    (&middle.quantized, &expected_middle.quantized),
                    (&middle_decoded, &expected_middle_decoded),
                    (&full.residual, &expected_full.residual),
                    (&full.quantized, &expected_full.quantized),
                    (&decoded, &expected_decoded),
                    (&prefix, &expected_prefix),
                ] {
                    compare(
                        &actual.to_f32_vec(&stream).unwrap(),
                        &serde_json::json!({ "values": expected.to_f32_vec(&stream).unwrap() }),
                        &fixture,
                    );
                }
            } else {
                assert_eq!(
                    full.codes.to_i32_vec(&stream).unwrap(),
                    codes(&fixture["codes"])
                );
                for (actual, key) in [
                    (&full.residual, "residual"),
                    (&full.quantized, "quantized"),
                    (&decoded, "decoded"),
                    (&prefix, "prefix_decoded"),
                ] {
                    compare(
                        &actual.to_f32_vec(&stream).unwrap(),
                        &fixture[key],
                        &fixture,
                    );
                }
                compare(
                    &middle.quantized.to_f32_vec(&stream).unwrap(),
                    &serde_json::json!({ "values": middle_decoded.to_f32_vec(&stream).unwrap() }),
                    &fixture,
                );
            }
            let report = model.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if mode > 0 {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= 44
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
                    "parallel::native_prvq_parallel_worker",
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
    // Reap every worker before asserting so one failed rank cannot abandon its peers.
    let outputs = children
        .0
        .drain(..)
        .map(Child::wait_with_output)
        .collect::<Vec<_>>();
    let succeeded = outputs
        .iter()
        .all(|output| output.as_ref().is_ok_and(|output| output.status.success()));
    assert!(
        succeeded,
        "PRVQ world {size}:\n{}",
        outputs
            .iter()
            .enumerate()
            .map(|(rank, output)| {
                match output {
                    Ok(output) => format!(
                        "rank {rank}: {}\n{}\n{}",
                        output.status,
                        String::from_utf8_lossy(&output.stdout),
                        String::from_utf8_lossy(&output.stderr)
                    ),
                    Err(error) => format!("rank {rank}: failed to reap worker: {error}"),
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    );
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_prvq_tp2_residency_matches_nvidia_quantization_and_ranges() {
    launch(2);
}
