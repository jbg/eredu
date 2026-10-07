//! Finite waveform encoding through native encoder and borrowed PRVQ channel shards.
use super::*;
#[path = "../support/component_coordinator.rs"]
mod component_coordinator;
use eredu_backend_mlx::native::{
    materialize_ear_encoder_parallel, materialize_prvq_parallel, MlxEarEncoder, MlxPrvq,
};
use eredu_codec::ear_encoder_layered::EncoderExecution;
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_EAR_ENCODER_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_EAR_ENCODER_TP_DIRECTORY";

type Output = (Vec<f32>, Vec<i32>, Vec<f32>);
fn run(
    model: &mut AudioEncoder<Backend, MlxEarEncoder>,
    quantizer: &mut MlxPrvq,
    config: &AudioEncoderConfig,
    case: &Value,
    stream: &Stream,
) -> Output {
    let reads = model
        .neural
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let quantizer_reads = quantizer
        .residency_report()
        .unwrap()
        .weight_store()
        .physical_reads;
    let invalid =
        MlxTensor::full_f32(0., &[1, 1, config.samples_per_frame().unwrap() - 1], stream).unwrap();
    assert!(matches!(
        model.encode(&invalid, quantizer, stream),
        Err(eredu_codec::Error::InvalidShape(_))
    ));
    let invalid_spectrum =
        MlxTensor::full_f32(0., &[1, config.neural.fft_size + 3, 6], stream).unwrap();
    assert!(matches!(
        model.neural.forward(&invalid_spectrum, stream),
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
        "invalid PCM and spectrum acquire no encoder parameters"
    );
    assert_eq!(
        quantizer
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads,
        quantizer_reads
    );
    let pcm = tensor(&case["pcm"]);
    let latent = model.latent(&pcm, stream).unwrap();
    let encoded = model.encode(&pcm, quantizer, stream).unwrap();
    let decoded = quantizer.decode(&encoded, stream).unwrap();
    (
        latent.to_f32_vec(stream).unwrap(),
        encoded.to_i32_vec(stream).unwrap(),
        decoded.to_f32_vec(stream).unwrap(),
    )
}

// Give each convolution group its own contiguous input block in canonical OIK
// storage. The ordinary model supplies the expected grouped-convolution equation;
// the selected model must materialize scattered output rows in both groups.
fn grouped_case(original: &Value) -> Value {
    let mut case = original.clone();
    case["geometry"]["groups"] = serde_json::json!(2);
    for layer in [3, 6] {
        let key = format!("tts_model.audio_codec.encoder.layers.{layer}.weight");
        let tensor = &mut case["parameters"][&key];
        let shape = tensor["shape"].as_array().unwrap();
        let output = shape[0].as_u64().unwrap() as usize;
        let input = shape[1].as_u64().unwrap() as usize;
        let kernel = shape[2].as_u64().unwrap() as usize;
        let values = floats(tensor);
        let mut grouped = Vec::new();
        for row in 0..output {
            let group = row / (output / 2);
            let start = (row * input + group * input / 2) * kernel;
            grouped.extend_from_slice(&values[start..start + input / 2 * kernel]);
        }
        *tensor = serde_json::json!({"shape": [output,input/2,kernel], "values": grouped});
    }
    case
}

#[test]
fn native_ear_encoder_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let coordinator = component_coordinator::coordinator(&group, &stream);
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let original = &fixture["encoders"]["compact"];
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    for (f16, grouped) in [(false, false), (true, false), (false, true)] {
        let case = if grouped {
            grouped_case(original)
        } else {
            original.clone()
        };
        let config = AudioEncoderConfig {
            quantizer: Geometry {
                depth: 3,
                cardinality: 5,
                channels: 4,
            },
            neural: geometry(&case["geometry"]),
            spectral: spectral(&case["geometry"]),
        };
        let source = checkpoint(
            &case,
            f16,
            &directory
                .path()
                .join(format!("encoder_{f16}_{grouped}.safetensors")),
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
        // Input and downsampling convolutions shard output channels. ConvNeXt depthwise
        // convolution, normalization and output bias total 336 replicated bytes.
        let expected_bytes = if grouped {
            [40, 136, 136, 16, 400, 400, 48]
        } else {
            [40, 136, 136, 32, 400, 400, 96]
        };
        assert_eq!(bytes, expected_bytes);
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
        assert!(materialize_ear_encoder(
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
        assert!(materialize_ear_encoder_parallel(
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
            "rank mismatch reads no payload"
        );
        let quantizer_prepared = || {
            PreparedQuantizerBindings::from_recipes(
                config.quantizer,
                "tts_model.audio_codec.prvq",
                source.clone(),
                &recipes(&source),
            )
            .unwrap()
        };
        let expected = if f16 || grouped {
            let neural = materialize_ear_encoder(
                prepared(config.neural.clone(), source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap()
            .0;
            let mut quantizer = materialize_prvq(
                quantizer_prepared(),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap()
            .0;
            let mut ordinary =
                AudioEncoder::<Backend, _>::from_parts(config.clone(), neural, &stream).unwrap();
            Some(run(&mut ordinary, &mut quantizer, &config, &case, &stream))
        } else {
            None
        };
        for mode in if grouped {
            0..1
        } else if f16 {
            2..3
        } else {
            0..3
        } {
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
            let (neural, metadata) = materialize_ear_encoder_parallel(
                bindings,
                &group,
                policy(budget),
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(
                metadata.layer_parameter_bytes(),
                if grouped { 1176 } else { 1240 }
            );
            assert_eq!(neural.static_lease_count(), 0);
            let (mut quantizer, quantizer_metadata) = materialize_prvq_parallel(
                quantizer_prepared(),
                &coordinator,
                policy(44),
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(quantizer_metadata.layer_parameter_bytes(), 132);
            let mut model =
                AudioEncoder::<Backend, _>::from_parts(config.clone(), neural, &stream).unwrap();
            let actual = run(&mut model, &mut quantizer, &config, &case, &stream);
            if let Some(expected) = &expected {
                compare(
                    &actual.0,
                    &serde_json::json!({"values":expected.0}),
                    &fixture,
                );
                assert_eq!(actual.1, expected.1);
                compare(
                    &actual.2,
                    &serde_json::json!({"values":expected.2}),
                    &fixture,
                );
            } else {
                compare(&actual.0, &case["latent"], &fixture);
                assert_eq!(actual.1, codes(&case["codes"]));
                compare(&actual.2, &case["decoded"], &fixture);
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
                    quantizer
                        .residency_report()
                        .unwrap()
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= 44
                );
            }
            if let Some(report) = model.neural.dense_stream_report().unwrap() {
                assert_eq!(report.prefill_forwards() + report.decode_forwards(), 2);
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
                    "parallel::native_ear_encoder_parallel_worker",
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
            "EAR encoder world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_ear_encoder_tp2_residency_matches_nvidia_finite_latent_and_codes() {
    launch(2);
}
