//! Native streaming decoder cuts preserve local carry and codec-owned channel/cadence geometry.
use super::*;
use eredu_backend_mlx::native::{materialize_ear_decoder_pipeline, MlxEarDecoderPipelineState};
use eredu_checkpoint::store::StoreError;
use eredu_codec::{
    ear_decoder_distribution::{DecoderPipelinePreparationError, PreparedDecoderPipelineBindings},
    ear_decoder_layered::{decoder_parameter_description, DecoderArchitecture},
    ear_decoder_partition::{DecoderBoundarySchema, DecoderPipelineCuts},
    streaming::StreamPhase,
};
use eredu_core::{
    checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode, DistributedCommitEpoch,
    DistributedCommitOutcome, ParallelRankTopology, ParallelTopology,
};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitectureParameters, ArchitecturePartition,
    ArchitectureStatePartitionPlan, ArchitectureStatePartitionRule, BoundaryDimensionContract,
    BoundaryRoleContract, BoundaryTensorDimension, BoundaryTensorDtype,
    CommunicationCompletionPolicy, CommunicationGroupDescriptor, CommunicationGroupRequirements,
    CommunicationManifest, CommunicationOperation, CommunicationOperationRequirement,
    CommunicationRouteDescriptor, CommunicationRouteId, CommunicationTensorLimits,
    LayeredPartitionDriver, PartitionBoundaryRoute, PartitionOutputPublication, PartitionOwnership,
    PartitionedExecutionPlan, PipelineActivationDtype, PipelineWireContract,
    RoleExactBoundaryContract,
};

const WORKER: &str = "EREDU_EAR_DECODER_PIPELINE_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_EAR_DECODER_PIPELINE_DIRECTORY";

pub(super) fn retained_pipeline(
    state: &MlxEarDecoderPipelineState,
    stream: &Stream,
) -> Vec<(Vec<i32>, Vec<f32>)> {
    state
        .units()
        .iter()
        .filter_map(|unit| unit.carry())
        .map(|tensor| (tensor.shape().to_vec(), tensor.to_f32_vec(stream).unwrap()))
        .collect()
}
pub(super) fn compare_cache(
    state: &MlxEarDecoderPipelineState,
    expected: &Value,
    range: std::ops::Range<usize>,
    fixture: &Value,
    stream: &Stream,
) {
    assert_eq!(state.global_units(), range);
    assert_eq!(state.units().len(), range.len());
    for (global, unit) in range.zip(state.units()) {
        let cache = match global {
            1 => Some(0),
            2 => Some(1),
            4 => Some(2),
            5 => Some(3),
            _ => None,
        };
        if let Some(cache) = cache {
            compare(
                &unit.carry().unwrap().to_f32_vec(stream).unwrap(),
                &expected["cache"][cache.to_string()],
                fixture,
            );
        } else {
            assert!(unit.carry().is_none());
        }
    }
}

#[test]
fn native_ear_decoder_pipeline_worker() {
    run_worker(false);
}

#[test]
fn native_ear_waveform_pipeline_worker() {
    run_worker(true);
}

fn run_worker(waveform: bool) {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    let size = size.parse::<usize>().unwrap();
    let tensor_ranks = size / 2;
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), size);
    let rank = world.rank();
    let stage = rank / tensor_ranks;
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let case = if waveform {
        &fixture["composition"]
    } else {
        &fixture["decoders"]["compact"]
    };
    let config = geometry(&case["geometry"]);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let source = checkpoint(case, false, &directory.path().join("decoder.safetensors"));
    // Cut3 carries [B,4,3T]; cut4 carries [B,2,6T]. The spectral fixture
    // isolates both wire geometries; waveform composition also crosses cut4
    // through its host-resident units.
    let selections = if waveform {
        vec![(3, 0), (4, 1), (3, 2)]
    } else {
        vec![(3, 0), (3, 1), (3, 2), (4, 0)]
    };
    for (cut, mode) in selections {
        let reads = source.source_diagnostics().unwrap().physical_reads;
        let global = prepared(config.clone(), source.clone());
        let (global, local) = if tensor_ranks > 1 {
            let topology = ParallelRankTopology::new(
                ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
                rank % tensor_ranks,
            )
            .unwrap();
            let (bindings, layout) = global.select_parallel(topology).unwrap();
            (bindings, Some(layout))
        } else {
            (global, None)
        };
        let prefix = "tts_model.audio_codec.decoder";
        let description = decoder_parameter_description(&config, prefix).unwrap();
        let complete = if let Some(layout) = &local {
            DecoderArchitecture::<Backend>::new_parallel(config.clone(), prefix, layout).unwrap()
        } else {
            DecoderArchitecture::<Backend>::new(config.clone(), prefix).unwrap()
        };
        let range = if stage == 0 { 0..cut } else { cut..7 };
        let partition = ArchitecturePartition::from_description(
            &description,
            [(description.graph().groups()[0].id(), range.clone())],
            PartitionOwnership::new(stage == 0, stage == 1, std::iter::empty::<String>()).unwrap(),
            &complete.state_layout().unwrap(),
            &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                0,
                0..7,
            )]),
            (),
            DecoderBoundarySchema::new(&config).unwrap(),
        )
        .unwrap();
        let selected = global.into_partition(&partition).unwrap();
        assert!(selected.source().source_keys().iter().all(|key| range
            .clone()
            .any(|unit| key.starts_with(&format!("{prefix}.layers.{unit}.")))));
        let neighbor = if stage == 0 {
            format!(
                "{prefix}.layers.{cut}.{}",
                if cut == 3 { "weight" } else { "dwconv.weight" }
            )
        } else {
            format!("{prefix}.layers.0.weight")
        };
        assert!(matches!(
            selected.source().source_metadata(&neighbor),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
        let sizes = selected
            .unit_bindings()
            .iter()
            .map(|(_, unit)| binding_bytes(unit).unwrap())
            .collect::<Vec<_>>();
        let expected = if tensor_ranks == 1 {
            [192, 688, 688, 64, 216, 216, 80]
        } else {
            [96, 400, 400, 32, 136, 136, 40]
        };
        assert_eq!(sizes, expected[range.clone()]);
        let largest = *sizes.iter().max().unwrap();
        let total = sizes.iter().sum::<u64>();
        let budget = if mode == 2 { largest * 2 } else { largest };
        let residency = match mode {
            0 => LayerWeightResidency::FullyResident,
            1 => LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
                OffloadConfig::new(Some(budget), Some(1 << 20), 1).unwrap(),
            )),
            _ => LayerWeightResidency::DenseDiskStream(
                DenseDiskStreamLoadOptions::new(budget, 0, 0, 0).unwrap(),
            ),
        };
        let (manifest, routes, session, tensor_group) =
            communication(&config, tensor_ranks, rank, cut);
        let cuts = DecoderPipelineCuts::new(&config, routes.iter().map(|route| (route.route, cut)))
            .unwrap();
        let driver =
            LayeredPartitionDriver::new_with_state_ownership(&partition, 0, range.clone(), true)
                .unwrap();
        let plan = PartitionedExecutionPlan::new(
            description.graph().clone(),
            vec![(ArchitectureGroupKind::AudioDecoder, false)],
            vec![Some(driver)],
            routes,
            Some(PartitionOutputPublication {
                group: session,
                owner_rank: tensor_ranks,
            }),
            Some(session),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        if !waveform && cut == 3 && mode == 0 {
            let stage_bindings = || {
                let bindings = prepared(config.clone(), source.clone());
                let bindings = match &local {
                    Some(layout) => bindings.into_parallel(layout).unwrap(),
                    None => bindings,
                };
                bindings.into_partition(&partition).unwrap()
            };
            let wrong = DecoderPipelineCuts::new(
                &config,
                plan.routes().iter().map(|route| (route.route, 4)),
            )
            .unwrap();
            assert!(matches!(
                PreparedDecoderPipelineBindings::new(
                    stage_bindings(),
                    partition.clone(),
                    wrong,
                    manifest.clone(),
                    plan.clone(),
                    tensor_group,
                    PipelineActivationDtype::Float32
                ),
                Err(DecoderPipelinePreparationError::Contract(_))
            ));
            if tensor_ranks == 2 {
                let other = CommunicationRouteId::new(100 + (1 - rank % 2) as u64);
                let skewed = DecoderPipelineCuts::new(
                    &config,
                    plan.routes()
                        .iter()
                        .map(|route| (route.route, if route.route == other { 4 } else { 3 })),
                )
                .unwrap();
                assert!(matches!(
                    PreparedDecoderPipelineBindings::new(
                        stage_bindings(),
                        partition.clone(),
                        skewed,
                        manifest.clone(),
                        plan.clone(),
                        tensor_group,
                        PipelineActivationDtype::Float32
                    ),
                    Err(DecoderPipelinePreparationError::Contract(_))
                ));
            }
        }
        let prepared = PreparedDecoderPipelineBindings::new(
            selected,
            partition,
            cuts,
            manifest,
            plan,
            tensor_group,
            PipelineActivationDtype::Float32,
        )
        .unwrap();
        assert_eq!(
            source.source_diagnostics().unwrap().physical_reads,
            reads,
            "complete stage admission reads no payload"
        );
        let (mut model, initial, metadata) =
            materialize_ear_decoder_pipeline(prepared, &world, residency, &weights, &stream)
                .unwrap();
        assert_eq!(metadata.static_device_bytes(), 0);
        assert_eq!(metadata.layer_parameter_bytes(), total);
        assert_eq!(metadata.residency(), residency.execution_residency());
        assert_eq!(initial.global_units(), range);
        assert!(retained_pipeline(&initial, &stream).is_empty());
        if waveform {
            super::waveform::exercise(
                &mut model,
                source.clone(),
                case,
                &fixture,
                mode,
                &world,
                &weights,
                &stream,
                budget,
            );
            continue;
        }
        let mut epoch_value = 0;
        let mut state = initial.clone();
        for (index, reference) in std::iter::once(&case["full"][0])
            .chain(case["streaming"].as_array().unwrap().iter())
            .enumerate()
        {
            if index == 1 {
                state = model.initial_state();
            }
            let input = tensor(&reference["input"]);
            let descriptor = [
                cut as u64,
                mode,
                index as u64,
                input.shape()[0] as u64,
                input.shape()[1] as u64,
                0,
                0,
                0,
            ];
            if index == 0 && cut == 3 && mode == 2 {
                let before = model
                    .residency_report()
                    .unwrap()
                    .weight_store()
                    .physical_reads;
                let mut disagreement = descriptor;
                disagreement[7] = rank as u64;
                assert!(model
                    .forward(
                        &input,
                        &mut state,
                        disagreement,
                        DistributedCommitEpoch::new(1).unwrap(),
                        &stream
                    )
                    .is_err());
                assert_eq!(
                    model
                        .residency_report()
                        .unwrap()
                        .weight_store()
                        .physical_reads,
                    before
                );
                assert!(retained_pipeline(&state, &stream).is_empty());
            }
            let parent = state.clone();
            let before = retained_pipeline(&parent, &stream);
            epoch_value += 1;
            let epoch = DistributedCommitEpoch::new(epoch_value).unwrap();
            let (raw, outcome) = model
                .forward(&input, &mut state, descriptor, epoch, &stream)
                .unwrap();
            assert_eq!(outcome, DistributedCommitOutcome::Committed(epoch));
            compare(
                &raw.to_f32_vec(&stream).unwrap(),
                &reference["raw"],
                &fixture,
            );
            if index > 0 && !reference["flush"].as_bool().unwrap() {
                compare_cache(&state, reference, range.clone(), &fixture, &stream);
            } else {
                // The offline oracle retains no caches, and the streaming wrapper
                // clears them on flush. Neural carry is released in finish below.
                assert_eq!(state.global_units(), range);
                assert_eq!(state.units().len(), range.len());
            }
            assert_eq!(
                retained_pipeline(&parent, &stream),
                before,
                "candidate advances only its local cache branch"
            );
            if index > 0 {
                let mut replay = parent.clone();
                epoch_value += 1;
                let epoch = DistributedCommitEpoch::new(epoch_value).unwrap();
                let (replayed, outcome) = model
                    .forward(&input, &mut replay, descriptor, epoch, &stream)
                    .unwrap();
                assert_eq!(outcome, DistributedCommitOutcome::Committed(epoch));
                assert_eq!(
                    raw.to_f32_vec(&stream).unwrap(),
                    replayed.to_f32_vec(&stream).unwrap()
                );
                assert_eq!(
                    retained_pipeline(&state, &stream),
                    retained_pipeline(&replay, &stream)
                );
            }
        }
        let before = retained_pipeline(&state, &stream);
        let reads = model
            .residency_report()
            .unwrap()
            .weight_store()
            .physical_reads;
        let finished = model.finish(&state).unwrap();
        assert_eq!(finished.phase(), StreamPhase::Finished);
        assert!(retained_pipeline(&finished, &stream).is_empty());
        assert_eq!(retained_pipeline(&state, &stream), before);
        assert!(model
            .finish(&finished)
            .unwrap()
            .units()
            .iter()
            .all(|unit| unit.carry().is_none()));
        assert_eq!(
            model
                .residency_report()
                .unwrap()
                .weight_store()
                .physical_reads,
            reads,
            "neural finish acquires no learned parameters"
        );
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
            assert_eq!(report.prefill_forwards() + report.decode_forwards(), 9);
        }
    }
}

fn boundary_contract(config: &DecoderConfig, cut: usize) -> RoleExactBoundaryContract {
    let geometry = DecoderBoundarySchema::new(config)
        .unwrap()
        .at_cut(cut)
        .unwrap();
    let resolved = geometry.resolve(2, 6).unwrap();
    let schema = geometry.wire_schema().unwrap();
    let roles = std::iter::once(schema.primary())
        .chain(schema.auxiliary())
        .zip(std::iter::once(resolved.primary()).chain(resolved.auxiliary()))
        .map(|(symbolic, resolved)| {
            let dtype = match symbolic.dtype() {
                BoundaryTensorDtype::Activation | BoundaryTensorDtype::Float32 => TensorDtype::F32,
                BoundaryTensorDtype::Int32 => TensorDtype::I32,
                other => panic!("unexpected codec wire dtype {other:?}"),
            };
            let shape = symbolic
                .shape()
                .iter()
                .zip(resolved.shape())
                .map(|(dim, extent)| match dim {
                    BoundaryTensorDimension::Batch | BoundaryTensorDimension::Sequence => {
                        BoundaryDimensionContract::Variable {
                            maximum: *extent as usize,
                        }
                    }
                    BoundaryTensorDimension::Fixed(value) => {
                        BoundaryDimensionContract::Fixed(*value as usize)
                    }
                    other => panic!("unexpected codec wire dimension {other:?}"),
                })
                .collect();
            BoundaryRoleContract::symbolic(symbolic.role(), dtype, shape).unwrap()
        });
    RoleExactBoundaryContract::new(schema.identity(), roles).unwrap()
}

fn communication(
    config: &DecoderConfig,
    tensor_ranks: usize,
    rank: usize,
    cut: usize,
) -> (
    CommunicationManifest,
    Vec<PartitionBoundaryRoute>,
    CollectiveGroupId,
    Option<CollectiveGroupId>,
) {
    let world = tensor_ranks * 2;
    let session = CollectiveGroupId::new(1);
    let tensor_requirement = |operation, dtypes| {
        let count = if operation == CommunicationOperation::SendReceive {
            2
        } else {
            1
        };
        let limits = CommunicationTensorLimits::new(count, 3, 8192, None).unwrap();
        CommunicationOperationRequirement::tensors(operation, dtypes, limits, true).unwrap()
    };
    let mut groups = vec![CommunicationGroupDescriptor::new(
        session,
        0,
        (0..world).collect(),
        Some(rank),
        CommunicationGroupRequirements::new([
            tensor_requirement(CommunicationOperation::Broadcast, vec![TensorDtype::F32]),
            CommunicationOperationRequirement::failure_agreement(true),
        ])
        .unwrap(),
    )
    .unwrap()];
    let mut local_tensor = None;
    if tensor_ranks > 1 {
        let stage = rank / tensor_ranks;
        let id = CollectiveGroupId::new(10 + stage as u32);
        groups.push(
            CommunicationGroupDescriptor::new(
                id,
                1,
                (stage * tensor_ranks..(stage + 1) * tensor_ranks).collect(),
                Some(rank % tensor_ranks),
                CommunicationGroupRequirements::new([
                    tensor_requirement(
                        CommunicationOperation::AllReduceSum,
                        vec![TensorDtype::F32],
                    ),
                    CommunicationOperationRequirement::failure_agreement(true),
                ])
                .unwrap(),
            )
            .unwrap(),
        );
        local_tensor = Some(id);
    }
    let mut routes = Vec::new();
    let mut descriptors = Vec::new();
    for lane in 0..tensor_ranks {
        let route = CommunicationRouteId::new(100 + lane as u64);
        routes.push(PartitionBoundaryRoute {
            source_group: 0,
            destination_group: 0,
            source_rank: lane,
            destination_rank: tensor_ranks + lane,
            route,
        });
        descriptors.push(
            CommunicationRouteDescriptor::new(
                route,
                lane,
                lane,
                tensor_ranks + lane,
                tensor_requirement(
                    CommunicationOperation::SendReceive,
                    vec![TensorDtype::F32, TensorDtype::I32],
                ),
            )
            .unwrap()
            .with_boundary_contract(boundary_contract(config, cut))
            .unwrap(),
        );
    }
    let manifest = CommunicationManifest::new(world, rank, groups, descriptors)
        .unwrap()
        .with_completion_policy(
            CommunicationCompletionPolicy::new(
                std::time::Duration::from_secs(30),
                CompletionCancellationMode::QuarantineUntilComplete,
            )
            .unwrap(),
        );
    (manifest, routes, session, local_tensor)
}

pub(super) fn launch(size: usize, worker: &str) {
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
                .args(["--exact", worker, "--nocapture", "--test-threads=1"])
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
    let outputs = outputs.into_iter().map(Result::unwrap).collect::<Vec<_>>();
    let diagnostics = outputs
        .iter()
        .enumerate()
        .map(|(rank, output)| {
            format!(
                "world {size} rank {rank}: {}\n{}\n{}",
                output.status,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        outputs.iter().all(|output| output.status.success()),
        "EAR decoder pipeline worker failure:\n{diagnostics}"
    );
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_ear_decoder_pipeline_pp2_and_tp2_pp2_match_nvidia_streaming_spectra_and_caches() {
    launch(2, "pipeline::native_ear_decoder_pipeline_worker");
    launch(4, "pipeline::native_ear_decoder_pipeline_worker");
}
