//! Conditioning shards retain replicated character outputs and both guidance lanes.
use super::*;
use eredu_backend_mlx::native::{
    materialize_voicechat_conditioning_parallel, MlxConditioning, MlxRvq,
};
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_CONDITIONING_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_CONDITIONING_TP_DIRECTORY";

fn outputs(
    model: &mut MlxConditioning,
    feedback: &mut MlxRvq,
    batch: &CharacterBatch,
    character: &Value,
    ingress: &Value,
    reference: bool,
    stream: &Stream,
) -> Vec<Vec<f32>> {
    let embedded = model.character_embed(batch, stream).unwrap();
    let output = model
        .character_finish(batch, &tensor(&character["hidden"]), stream)
        .unwrap();
    let fused = model
        .fuse(&tensor(&character["audio"]), &output, stream)
        .unwrap();
    if reference {
        let a = character["tolerance"]["absolute"].as_f64().unwrap() as f32;
        let r = character["tolerance"]["relative"].as_f64().unwrap() as f32;
        compare(&embedded, &character["embedded"], a, r, stream);
        compare(&output, &character["character_output"], a, r, stream);
        compare(&fused, &character["fusion_output"], a, r, stream);
    }
    let mut result = vec![embedded, output, fused];
    for case in ingress["cases"].as_array().unwrap() {
        let code = code_tensor(&case["code"], stream);
        let condition = tensor(&case["supplied_subwords"]);
        let any = case["subword_mask"]["values"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_bool().unwrap());
        let condition = any.then_some(&condition);
        let lanes = if case["audio_mask"].is_null() {
            model
                .decode(feedback, &code, condition, true, stream)
                .unwrap()
        } else {
            let mask: Vec<bool> = case["audio_mask"]["values"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_bool().unwrap())
                .collect();
            let layout = PromptLayout::new(code.dim(0), code.dim(1), &mask).unwrap();
            let speaker = if case["supplied_speaker"].is_null() {
                SpeakerPrompt::ProjectedCodes
            } else {
                SpeakerPrompt::Aria
            };
            model
                .prefill(feedback, &code, &layout, speaker, condition, true, stream)
                .unwrap()
        };
        if reference {
            let a = ingress["comparison"]["absolute_tolerance"]
                .as_f64()
                .unwrap() as f32;
            let r = ingress["comparison"]["relative_tolerance"]
                .as_f64()
                .unwrap() as f32;
            compare(&lanes.audio, &case["audio_embedding"], a, r, stream);
            compare(&lanes.text, &case["conditioning"], a, r, stream);
        }
        let fused = model.fuse(&lanes.audio, &lanes.text, stream).unwrap();
        result.extend([lanes.audio, lanes.text, fused]);
    }
    result
        .into_iter()
        .map(|t| t.to_f32_vec(stream).unwrap())
        .collect()
}

#[test]
fn native_conditioning_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(group.size(), 2);
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let (character, ingress, combined, batch, dimensions) = fixture();
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    for f16 in [false, true] {
        let source = checkpoint(
            &combined,
            f16,
            &directory
                .path()
                .join(format!("conditioning_{f16}.safetensors")),
        );
        let (selected, _) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap();
        let (_, _, _, _, units) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap()
            .0
            .into_parts();
        let bytes: Vec<u64> = units
            .iter()
            .map(|unit| binding_bytes(unit).unwrap())
            .collect();
        // Rank-local character columns/projection rows; prompt projection columns;
        // fusion projection rows and biases. Flags, speaker and control weights replicate.
        let character_elements = (dimensions.character.characters as u64 + 1) * 2 + 8 + 20;
        assert_eq!(bytes, [character_elements * 4, 144, 116]);
        let budget = *bytes.iter().max().unwrap();
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(materialize_voicechat_conditioning(
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
        assert!(materialize_voicechat_conditioning_parallel(
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
        let (mut feedback, _) = materialize_voicechat_rvq(
            PreparedRvqBindings::from_parameters(
                RvqDimensions {
                    cardinality: 5,
                    latent: 4,
                    hidden: 4,
                },
                source.clone(),
                &parameters(&source),
            )
            .unwrap(),
            LayerWeightResidency::FullyResident,
            &weights,
            &stream,
        )
        .unwrap();
        let expected = f16.then(|| {
            let (mut ordinary, _) = materialize_voicechat_conditioning(
                prepared(dimensions, source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap();
            outputs(
                &mut ordinary,
                &mut feedback,
                &batch,
                &character,
                &ingress,
                false,
                &stream,
            )
        });
        for mode in if f16 { 2..3 } else { 0..3 } {
            let bindings = prepared(dimensions, source.clone());
            let bindings = if mode == 2 {
                bindings.select_parallel(topology).unwrap().0
            } else {
                bindings
            };
            let (mut model, metadata) = materialize_voicechat_conditioning_parallel(
                bindings,
                &group,
                residency(mode, budget),
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), bytes.iter().sum::<u64>());
            assert_eq!(model.static_lease_count(), 0);
            let reads = source.source_diagnostics().unwrap().physical_reads;
            let invalid = tensor(&character["audio"])
                .cast_float(TensorElementType::F16, &stream)
                .unwrap();
            assert!(matches!(
                model.fuse(&invalid, &tensor(&character["character_output"]), &stream),
                Err(TtsError::Shape(_))
            ));
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "invalid ingress acquires no parameter lease"
            );
            let actual = outputs(
                &mut model,
                &mut feedback,
                &batch,
                &character,
                &ingress,
                !f16,
                &stream,
            );
            if let Some(expected) = &expected {
                assert_eq!(actual.len(), expected.len());
                for (actual, expected) in actual.iter().zip(expected) {
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(expected) {
                        let a = character["tolerance"]["absolute"].as_f64().unwrap() as f32;
                        let r = character["tolerance"]["relative"].as_f64().unwrap() as f32;
                        assert!(
                            (actual - expected).abs() <= a + r * expected.abs(),
                            "F16 native shard parity: {actual} versus {expected}"
                        );
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
                    "parallel::native_conditioning_parallel_worker",
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
            "conditioning world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_conditioning_tp2_residency_matches_nvidia_character_ingress_and_fusion() {
    launch(2);
}
