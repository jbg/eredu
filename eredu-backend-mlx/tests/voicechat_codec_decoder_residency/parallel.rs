//! Neural decoder shards preserve replicated causal carry and the composed waveform lifecycle.
use super::*;
#[path = "../support/component_coordinator.rs"]
mod component_coordinator;
use eredu_backend_mlx::native::{
    materialize_ear_decoder_parallel, materialize_prvq_parallel, MlxEarDecoder, MlxPrvq,
};
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_EAR_DECODER_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_EAR_DECODER_TP_DIRECTORY";

type Model = AudioDecoder<Backend, MlxEarDecoder, MlxPrvq>;

fn run(
    model: &mut Model,
    case: &Value,
    fixture: &Value,
    reference: bool,
    stream: &Stream,
) -> Vec<Vec<f32>> {
    let all = code_tensor(&case["codes"], stream);
    let raw = model
        .neural
        .forward(&tensor(&case["latent"]), stream)
        .unwrap();
    let full = model.forward(&all, true, stream).unwrap();
    let mut outputs = vec![
        raw.to_f32_vec(stream).unwrap(),
        full.to_f32_vec(stream).unwrap(),
    ];
    if reference {
        compare(&outputs[0], &case["full"][0]["raw"], fixture);
        compare(&outputs[1], &case["full"][0]["waveform"], fixture);
    }
    let initial = model.initial_state(1).unwrap();
    let reads = model
        .neural
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let invalid =
        MlxTensor::full_f32(0., &[1, 1, model.neural.config().latent_size + 1], stream).unwrap();
    assert!(matches!(
        model.neural.step(&invalid, initial.neural(), stream),
        Err(eredu_codec::Error::InvalidShape(_))
    ));
    assert_eq!(
        model
            .neural
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        reads,
        "invalid latent acquires no decoder unit"
    );
    let mut state = initial.clone();
    let mut frames = 0;
    let mut samples = 0;
    for step in case["streaming_unflushed"].as_array().unwrap() {
        let count = step["input"]["shape"][1].as_i64().unwrap() as i32;
        let input = all
            .index(
                &[
                    Index::Full,
                    Index::Range(frames, frames + count),
                    Index::Full,
                ],
                stream,
            )
            .unwrap();
        let before = retained(&state, stream);
        let (pcm, next) = model.step(&input, &state, true, stream).unwrap();
        assert_eq!(
            retained(&state, stream),
            before,
            "candidate execution preserves parent caches"
        );
        let (replay, replay_state) = model.step(&input, &state.clone(), true, stream).unwrap();
        assert_eq!(
            pcm.to_f32_vec(stream).unwrap(),
            replay.to_f32_vec(stream).unwrap()
        );
        assert_eq!(
            retained(&next, stream),
            retained(&replay_state, stream),
            "snapshot replay includes neural and spectral carry"
        );
        let pcm = pcm.to_f32_vec(stream).unwrap();
        if reference {
            compare(&pcm, &step["waveform"], fixture);
            for (index, carry) in next
                .neural()
                .units()
                .iter()
                .filter_map(|unit| unit.carry())
                .enumerate()
            {
                compare(
                    &carry.to_f32_vec(stream).unwrap(),
                    &step["cache"][index.to_string()],
                    fixture,
                );
            }
        }
        samples += pcm.len() as u64;
        outputs.push(pcm);
        outputs.extend(retained(&next, stream).into_iter().map(|(_, value)| value));
        frames += count;
        assert_eq!(next.frames_seen(), frames as u64);
        state = next;
    }
    let reads = model
        .neural
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let quantizer_reads = model
        .quantizer
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let (tail, finished) = model.finish(&state, true, stream).unwrap();
    assert_eq!(
        model
            .neural
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        reads
    );
    assert_eq!(
        model
            .quantizer
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        quantizer_reads
    );
    let tail = tail.unwrap().to_f32_vec(stream).unwrap();
    if reference {
        compare(&tail, &case["finish_tail"], fixture);
    }
    samples += tail.len() as u64;
    outputs.push(tail);
    outputs.extend(
        retained(&finished, stream)
            .into_iter()
            .map(|(_, value)| value),
    );
    assert!(finished.is_finished());
    assert_eq!(finished.frames_seen(), frames as u64);
    assert_eq!(finished.samples_emitted(), samples);
    assert!(model.finish(&finished, true, stream).unwrap().0.is_none());
    assert!(matches!(
        model.step(&all, &finished, true, stream),
        Err(eredu_codec::Error::StreamFinished)
    ));
    assert_eq!(initial.frames_seen(), 0);
    assert!(retained(&initial, stream).is_empty());
    outputs
}

#[test]
fn native_ear_decoder_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let coordinator = component_coordinator::coordinator(&group, &stream);
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let case = &fixture["composition"];
    let config = AudioDecoderConfig {
        quantizer: Geometry {
            depth: 3,
            cardinality: 5,
            channels: 4,
        },
        neural: geometry(&case["geometry"]),
        spectral: spectral(&case["geometry"]),
    };
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    for f16 in [false, true] {
        let source = checkpoint(
            case,
            f16,
            &directory.path().join(format!("decoder_{f16}.safetensors")),
        );
        let (_, _, _, layout, _, units) = prepared(config.neural.clone(), source.clone())
            .select_parallel(topology)
            .unwrap()
            .0
            .into_parts();
        let bytes: Vec<_> = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect();
        // Upsample/spectral projections and MLP intermediates are halved. Depthwise,
        // normalization and output-bias replicas total 336 of 1,240 bytes per rank.
        assert_eq!(bytes, [96, 400, 400, 32, 136, 136, 40]);
        let largest = *bytes.iter().max().unwrap();
        let depth = LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(u64::MAX, 0, 0, 0).unwrap(),
        )
        .device_depth(units.len());
        let disk_budget = (0..layout.group_count())
            .flat_map(|group| {
                let range = layout.group_range(group).unwrap();
                bytes[range.clone()]
                    .windows(depth.min(range.len()))
                    .map(|window| window.iter().sum::<u64>())
                    .collect::<Vec<_>>()
            })
            .max()
            .unwrap();
        let reads = source.source_diagnostics().unwrap().physical_reads;
        let selected = prepared(config.neural.clone(), source.clone())
            .select_parallel(topology)
            .unwrap()
            .0;
        assert!(materialize_ear_decoder(
            selected,
            LayerWeightResidency::FullyResident,
            &weights,
            &stream
        )
        .is_err());
        let wrong_topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1 - group.rank())
                .unwrap();
        let wrong = prepared(config.neural.clone(), source.clone())
            .select_parallel(wrong_topology)
            .unwrap()
            .0;
        assert!(materialize_ear_decoder_parallel(
            wrong,
            &group,
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
        let quantizer_prepared = || {
            let recipes = source
                .source_keys()
                .into_iter()
                .map(|name| {
                    (
                        name.clone(),
                        DerivedWeightRecipe::Source {
                            key: name,
                            selection: TensorSelection::Full,
                        },
                    )
                })
                .collect();
            PreparedQuantizerBindings::from_recipes(
                config.quantizer,
                "tts_model.audio_codec.prvq",
                source.clone(),
                &recipes,
            )
            .unwrap()
        };
        let expected = if f16 {
            let neural = materialize_ear_decoder(
                prepared(config.neural.clone(), source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap()
            .0;
            let quantizer = materialize_prvq(
                quantizer_prepared(),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap()
            .0;
            let mut ordinary =
                Model::from_parts(config.clone(), quantizer, neural, &stream).unwrap();
            Some(run(&mut ordinary, case, &fixture, false, &stream))
        } else {
            None
        };
        for mode in if f16 { 2..3 } else { 0..3 } {
            let policy = |budget| match mode {
                0 => LayerWeightResidency::FullyResident,
                1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(budget), Some(1 << 20), 1).unwrap(),
                )),
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(budget, 0, 0, 0).unwrap(),
                ),
            };
            let budget = if mode == 2 { disk_budget } else { largest };
            let bindings = prepared(config.neural.clone(), source.clone());
            let bindings = if mode == 2 {
                bindings.select_parallel(topology).unwrap().0
            } else {
                bindings
            };
            let (neural, metadata) = materialize_ear_decoder_parallel(
                bindings,
                &group,
                policy(budget),
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), 1240);
            assert_eq!(neural.static_lease_count(), 0);
            let (quantizer, quantizer_metadata) = materialize_prvq_parallel(
                quantizer_prepared(),
                &coordinator,
                policy(44),
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(quantizer_metadata.layer_parameter_bytes(), 132);
            let mut model = Model::from_parts(config.clone(), quantizer, neural, &stream).unwrap();
            let actual = run(&mut model, case, &fixture, !f16, &stream);
            if let Some(expected) = &expected {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    compare(actual, &serde_json::json!({"values": expected}), &fixture);
                }
            }
            let report = model.neural.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if mode > 0 {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= budget
                );
                assert!(
                    model
                        .quantizer
                        .residency_report()
                        .unwrap()
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= 44
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
                    "parallel::native_ear_decoder_parallel_worker",
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
    for (rank, output) in outputs.into_iter().enumerate() {
        let output = output.unwrap();
        assert!(
            output.status.success(),
            "EAR decoder world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_ear_decoder_tp2_residency_matches_nvidia_caches_waveforms_and_drain() {
    launch(2);
}
