//! Native Ring perception stages with exact encoded-frame transport.
use super::parallel::compare;
use super::*;
use eredu_architectures::voicechat::perception_layered::{PerceptionInput, PerceptionRuntimeState};
use eredu_core::{DistributedCommitEpoch, DistributedCommitOutcome};

const WORKER: &str = "EREDU_PERCEPTION_PIPELINE_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_PERCEPTION_PIPELINE_DIRECTORY";

fn compare_cache(
    state: &PerceptionRuntimeState<MlxTensor>,
    expected: &Value,
    frames: u64,
    owns_block: bool,
    fixture: &Value,
    stream: &Stream,
) {
    assert_eq!(state.frames(), frames);
    assert_eq!(state.blocks().len(), usize::from(owns_block));
    if let Some(cache) = state.blocks().first() {
        assert_eq!(
            cache.valid_channel(),
            expected["valid_cache"].as_u64().unwrap() as usize
        );
        compare(
            cache.channel().unwrap(),
            &expected["channel_cache"],
            fixture,
            stream,
        );
        compare(
            &cache
                .convolution()
                .unwrap()
                .transpose_axes(&[0, 2, 1], stream)
                .unwrap(),
            &expected["time_cache"],
            fixture,
            stream,
        );
    }
}

#[test]
fn native_perception_pipeline_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    let size = size.parse::<usize>().unwrap();
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), size);
    let rank = world.rank();
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
    for cut in 0..2 {
        for mode in 0..3 {
            let super::fixture_support::PipelineFixture {
                mut model,
                mut state,
                largest,
                owns_block,
                ..
            } = super::fixture_support::pipeline(
                dimensions,
                source.clone(),
                cut,
                mode,
                &world,
                &weights,
                &stream,
            );
            let windows = fixture["subsampling"]["steps"].as_array().unwrap();
            let references = fixture["perception"]["steps"].as_array().unwrap();
            for step in 0..=windows.len() {
                let (input, drop, reference, frames) = if step == 0 {
                    (
                        tensor(&fixture["subsampling"]["input"]),
                        0,
                        &fixture["perception"],
                        6,
                    )
                } else {
                    (
                        tensor(&windows[step - 1]["input"]),
                        windows[step - 1]["drop"].as_u64().unwrap() as usize,
                        &references[step - 1],
                        step as u64,
                    )
                };
                if step == 1 {
                    state = model.initial_state();
                }
                let descriptor = [
                    cut as u64,
                    mode,
                    step as u64,
                    drop as u64,
                    input.shape()[0] as u64,
                    input.shape()[1] as u64,
                    0,
                    0,
                ];
                let epoch = DistributedCommitEpoch::new(step as u64 + 1).unwrap();
                if step == 2 && cut == 0 && mode == 0 {
                    let mut disagreement = descriptor;
                    disagreement[7] = rank as u64;
                    assert!(model
                        .forward(
                            PerceptionInput {
                                mel: &input,
                                drop_preencoded: drop
                            },
                            &mut state,
                            disagreement,
                            epoch,
                            &stream,
                        )
                        .is_err());
                    compare_cache(&state, &references[0], 1, owns_block, &fixture, &stream);
                }
                let (encoder, projected, outcome) = model
                    .forward(
                        PerceptionInput {
                            mel: &input,
                            drop_preencoded: drop,
                        },
                        &mut state,
                        descriptor,
                        epoch,
                        &stream,
                    )
                    .unwrap();
                assert_eq!(outcome, DistributedCommitOutcome::Committed(epoch));
                compare(&encoder, &reference["encoder_output"], &fixture, &stream);
                compare(
                    &projected,
                    &reference["projected_output"],
                    &fixture,
                    &stream,
                );
                let cache_reference = if step == 0 {
                    references.last().unwrap()
                } else {
                    reference
                };
                compare_cache(
                    &state,
                    cache_reference,
                    frames,
                    owns_block,
                    &fixture,
                    &stream,
                );
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
                    "pipeline::native_perception_pipeline_worker",
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
    let outputs = children
        .0
        .drain(..)
        .map(Child::wait_with_output)
        .collect::<Vec<_>>();
    for (rank, output) in outputs.into_iter().enumerate() {
        let output = output.unwrap();
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
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_perception_pipeline_pp2_and_tp2_pp2_match_nvidia_outputs_and_cache() {
    launch(2);
    launch(4);
}
