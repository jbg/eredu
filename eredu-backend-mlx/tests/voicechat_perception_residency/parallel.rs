//! Actual Ring TP2 with bounded parameter providers and released perception equations.
use super::*;
use eredu_backend_mlx::native::materialize_voicechat_perception_parallel;
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_PERCEPTION_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_PERCEPTION_TP_DIRECTORY";

pub(super) fn compare(actual: &MlxTensor, reference: &Value, fixture: &Value, stream: &Stream) {
    let shape = reference["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap() as i32)
        .collect::<Vec<_>>();
    assert_eq!(actual.shape(), shape);
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual
        .to_f32_vec(stream)
        .unwrap()
        .into_iter()
        .zip(floats(reference))
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "perception element {index}: {actual} != NVIDIA {expected}"
        );
    }
}

fn compare_state(
    state: &PerceptionState<MlxTensor>,
    reference: &Value,
    frames: u64,
    fixture: &Value,
    stream: &Stream,
) {
    assert_eq!(state.frames(), frames);
    assert_eq!(state.blocks().len(), 1);
    let cache = &state.blocks()[0];
    assert_eq!(
        cache.valid_channel(),
        reference["valid_cache"].as_u64().unwrap() as usize
    );
    compare(
        cache.channel().unwrap(),
        &reference["channel_cache"],
        fixture,
        stream,
    );
    compare(
        &cache
            .convolution()
            .unwrap()
            .transpose_axes(&[0, 2, 1], stream)
            .unwrap(),
        &reference["time_cache"],
        fixture,
        stream,
    );
}

#[test]
fn native_perception_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(group.size(), 2);
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let dimensions = dimensions(&fixture);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let source = checkpoint(
        &fixture,
        false,
        &directory.path().join("perception.safetensors"),
    );
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    for mode in 0..3 {
        let (bindings, _) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap();
        let sizes = bindings
            .unit_bindings()
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect::<Vec<_>>();
        let largest = *sizes.iter().max().unwrap();
        let total: u64 = sizes.iter().sum();
        let residency = match mode {
            0 => LayerWeightResidency::FullyResident,
            1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(largest), Some(1 << 20), 1).unwrap(),
            )),
            _ => LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(largest * 2, 0, 0, 0).unwrap(),
            ),
        };
        let (mut model, metadata) = materialize_voicechat_perception_parallel(
            prepared(dimensions, source.clone()),
            &group,
            residency,
            &weights,
            &stream,
        )
        .unwrap();
        assert_eq!(metadata.static_device_bytes(), 0);
        assert_eq!(metadata.layer_parameter_bytes(), total);
        assert_eq!(metadata.residency(), residency.execution_residency());
        let references = fixture["perception"]["steps"].as_array().unwrap();
        let full = model
            .forward(
                &tensor(&fixture["subsampling"]["input"]),
                0,
                &PerceptionState::default(),
                &stream,
            )
            .unwrap();
        compare(
            &full.encoder,
            &fixture["perception"]["encoder_output"],
            &fixture,
            &stream,
        );
        compare(
            &full.projected,
            &fixture["perception"]["projected_output"],
            &fixture,
            &stream,
        );
        compare_state(
            &full.state,
            references.last().unwrap(),
            6,
            &fixture,
            &stream,
        );
        let mut previous = PerceptionState::default();
        for (index, step) in fixture["subsampling"]["steps"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let input = tensor(&step["input"]);
            if index == 2 {
                let before = state_values(&previous, &stream);
                assert!(matches!(
                    model.forward(&input, 100, &previous, &stream),
                    Err(PerceptionError::State(_))
                ));
                assert_eq!(state_values(&previous, &stream), before);
                compare_state(
                    &previous,
                    &references[index - 1],
                    index as u64,
                    &fixture,
                    &stream,
                );
            }
            let output = model
                .forward(
                    &input,
                    step["drop"].as_u64().unwrap() as usize,
                    &previous,
                    &stream,
                )
                .unwrap();
            compare(
                &output.encoder,
                &references[index]["encoder_output"],
                &fixture,
                &stream,
            );
            compare(
                &output.projected,
                &references[index]["projected_output"],
                &fixture,
                &stream,
            );
            compare_state(
                &output.state,
                &references[index],
                index as u64 + 1,
                &fixture,
                &stream,
            );
            assert_eq!(previous.frames(), index as u64);
            previous = output.state;
        }
        let report = model.residency_report().unwrap();
        assert!(report.weight_store().physical_reads > 0);
        if mode > 0 {
            let budget = if mode == 1 { largest } else { largest * 2 };
            assert!(
                report
                    .offload()
                    .peak_resident_bytes()
                    .get(MemoryTier::Device)
                    <= budget
            );
        }
        if mode == 2 {
            assert!(
                model
                    .dense_stream_report()
                    .unwrap()
                    .unwrap()
                    .decode_forwards()
                    > 0
            );
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
                    "parallel::native_perception_parallel_worker",
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
            "perception world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_perception_tp2_residency_matches_nvidia_outputs_and_cache() {
    launch(2);
}
