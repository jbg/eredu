//! Native vocabulary gathering retains independently bounded function-head weights.
use super::*;
use eredu_backend_mlx::native::materialize_voicechat_function_head_parallel;
use eredu_core::{ParallelRankTopology, ParallelTopology};

const WORKER: &str = "EREDU_FUNCTION_HEAD_TP_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_FUNCTION_HEAD_TP_DIRECTORY";

#[test]
fn native_function_head_parallel_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    assert_eq!(size, "2");
    let group = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(group.size(), 2);
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let coordinator = component_coordinator::coordinator_with_gather(&group, &stream);
    let original: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let hidden = tensor(&original["f32"]["normalized_hidden"]);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let topology =
        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), group.rank())
            .unwrap();
    // Removing the final vocabulary row leaves the other NVIDIA reference
    // logits unchanged, while requiring uneven shard padding and gathering.
    for (vocabulary, f16) in [(24, false), (23, false), (23, true)] {
        let mut fixture = original.clone();
        fixture["parameters"]["stt_model.function_head.weight"]["shape"][0] = vocabulary.into();
        fixture["parameters"]["stt_model.function_head.weight"]["values"]
            .as_array_mut()
            .unwrap()
            .truncate(vocabulary as usize * 4);
        let expected = floats(&original["f32"]["function_logits"])
            .as_chunks::<24>()
            .0
            .iter()
            .flat_map(|row| row[..vocabulary as usize].iter().copied())
            .collect::<Vec<_>>();
        let mut expected = serde_json::json!({"values": expected});
        let dimensions = FunctionHeadDimensions::new(4, vocabulary).unwrap();
        let source = checkpoint(
            &fixture,
            f16,
            &directory
                .path()
                .join(format!("function_{vocabulary}_{f16}.safetensors")),
        );
        if f16 {
            // The conversion-and-placement path is compared with ordinary
            // execution over exactly the same F16 values, independently of TP.
            let (mut ordinary, _) = materialize_voicechat_function_head(
                prepared(dimensions, source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap();
            expected = serde_json::json!({"values": ordinary.project(&hidden, &stream)
                .unwrap().to_f32_vec(&stream).unwrap()});
        }
        let (selected, _) = prepared(dimensions, source.clone())
            .select_parallel(topology)
            .unwrap();
        let local_bytes = binding_bytes(&selected.unit_bindings()[0]).unwrap();
        let expected_rows = if group.rank() == 0 {
            12
        } else {
            vocabulary - 12
        };
        assert_eq!(local_bytes, expected_rows as u64 * 4 * 4);
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(materialize_voicechat_function_head(
            selected,
            LayerWeightResidency::FullyResident,
            &weights,
            &stream
        )
        .is_err());
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            reads,
            "ordinary realization cannot read rank-local weights"
        );
        let wrong_topology =
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), 1 - group.rank())
                .unwrap();
        let (wrong_rank, _) = prepared(dimensions, source.clone())
            .select_parallel(wrong_topology)
            .unwrap();
        let reads = source.source_diagnostics().unwrap().physical_reads;
        assert!(materialize_voicechat_function_head_parallel(
            wrong_rank,
            &coordinator,
            LayerWeightResidency::FullyResident,
            &weights,
            &stream,
        )
        .is_err());
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            reads,
            "selected placement must match the communicator before payload reads"
        );
        // One F16 disk case covers conversion followed by slicing without
        // repeating the F32 residency matrix.
        for mode in if f16 { 2..3 } else { 0..3 } {
            let residency = match mode {
                0 => LayerWeightResidency::FullyResident,
                1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                    OffloadConfig::new(Some(local_bytes), Some(1 << 20), 1).unwrap(),
                )),
                _ => LayerWeightResidency::DenseDiskStream(
                    DenseDiskStreamLoadOptions::new(local_bytes, 0, 0, 0).unwrap(),
                ),
            };
            let bindings = prepared(dimensions, source.clone());
            let bindings = if mode == 2 {
                bindings.select_parallel(topology).unwrap().0
            } else {
                bindings
            };
            let (mut head, metadata) = materialize_voicechat_function_head_parallel(
                bindings,
                &coordinator,
                residency,
                &weights,
                &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), local_bytes);
            assert_eq!(head.static_lease_count(), 0);
            let reads = source.source_diagnostics().unwrap().physical_reads;
            // A single participant rejects geometry before any learned lease.
            let invalid = if group.rank() == 0 {
                MlxTensor::full_f32(0., &[1, 1, 5], &stream).unwrap()
            } else {
                hidden.clone()
            };
            assert!(head.project(&invalid, &stream).is_err());
            assert_eq!(
                source.source_diagnostics().unwrap().physical_reads,
                reads,
                "a peer's invalid input prevents every head acquisition"
            );
            // Repeat acquisition to expose leases retained across a completed gather.
            for _ in 0..2 {
                let actual = head.project(&hidden, &stream).unwrap();
                assert_eq!(actual.shape(), &[1, 3, vocabulary]);
                compare(&actual.to_f32_vec(&stream).unwrap(), &expected, &original);
            }
            let report = head.residency_report().unwrap();
            assert!(report.weight_store().physical_reads > 0);
            if mode > 0 {
                assert!(
                    report
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= local_bytes
                );
            }
            if let Some(report) = head.dense_stream_report().unwrap() {
                assert_eq!(report.prefill_forwards() + report.decode_forwards(), 2);
            }
        }
    }
    acquisition_failure(
        &group,
        &coordinator,
        &original,
        directory.path(),
        &weights,
        &stream,
    );
}

/// Retains the exact catalog and backing file while rejecting one payload acquisition.
struct FailingSource {
    source: SharedCheckpointSource,
    reject: std::sync::atomic::AtomicBool,
}
impl eredu_checkpoint::store::CheckpointSource for FailingSource {
    fn source_keys(&self) -> Vec<String> {
        self.source.source_keys()
    }
    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, StoreError> {
        self.source.source_metadata(key)
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, StoreError> {
        self.source.source_provenance(key)
    }
    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }
    fn acquire_lease(
        &self,
        request: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, StoreError> {
        self.check()?;
        self.source.acquire_lease(request)
    }
    fn prepare_encoded_read(
        &self,
        requests: &[eredu_checkpoint::store::TensorReadRequest],
    ) -> Result<Option<eredu_checkpoint::store::EncodedReadBatch>, StoreError> {
        self.check()?;
        self.source.prepare_encoded_read(requests)
    }
}
impl FailingSource {
    fn check(&self) -> Result<(), StoreError> {
        if self.reject.load(std::sync::atomic::Ordering::SeqCst) {
            Err(StoreError::Internal(
                "injected function-head payload acquisition failure".into(),
            ))
        } else {
            Ok(())
        }
    }
}

fn acquisition_failure(
    world: &safemlx::distributed::Group,
    coordinator: &eredu_backend_mlx::native::MlxReplicatedCoordinator,
    fixture: &Value,
    directory: &std::path::Path,
    weights: &Stream,
    stream: &Stream,
) {
    let dimensions = FunctionHeadDimensions::new(4, 24).unwrap();
    let source = checkpoint(
        fixture,
        false,
        &directory.join("head_acquisition.safetensors"),
    );
    let faulty = Arc::new(FailingSource {
        source,
        reject: std::sync::atomic::AtomicBool::new(false),
    });
    let locality = prepared(dimensions, faulty.clone())
        .select_parallel(
            ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), world.rank())
                .unwrap(),
        )
        .unwrap()
        .0;
    let local_bytes = binding_bytes(&locality.unit_bindings()[0]).unwrap();
    let policy = LayerWeightResidency::DenseDiskStream(
        DenseDiskStreamLoadOptions::new(local_bytes, 0, 0, 0).unwrap(),
    );
    let (mut head, _) = materialize_voicechat_function_head_parallel(
        locality,
        coordinator,
        policy,
        weights,
        stream,
    )
    .unwrap();
    faulty
        .reject
        .store(world.rank() == 0, std::sync::atomic::Ordering::SeqCst);
    let hidden = tensor(&fixture["f32"]["normalized_hidden"]);
    let failed = head.project(&hidden, stream);
    assert!(
        failed.is_err(),
        "one acquisition failure must reject every rank"
    );
    if world.rank() == 0 {
        let error = failed.unwrap_err();
        let mut source: &dyn std::error::Error = &error;
        let mut retained = false;
        loop {
            retained |= source
                .to_string()
                .contains("injected function-head payload acquisition failure");
            match source.source() {
                Some(next) => source = next,
                None => break,
            }
        }
        assert!(retained, "the failing rank retains its source error");
    }
    faulty
        .reject
        .store(false, std::sync::atomic::Ordering::SeqCst);
    // A new selected provider retries the immutable source after completed failure.
    let (mut retry, _) = materialize_voicechat_function_head_parallel(
        prepared(dimensions, faulty),
        coordinator,
        policy,
        weights,
        stream,
    )
    .unwrap();
    let actual = retry.project(&hidden, stream).unwrap();
    compare(
        &actual.to_f32_vec(stream).unwrap(),
        &fixture["f32"]["function_logits"],
        fixture,
    );
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
                    "parallel::native_function_head_parallel_worker",
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
            "function head world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_function_head_tp2_residency_matches_nvidia_logits_and_uneven_vocabulary() {
    launch(2);
}
