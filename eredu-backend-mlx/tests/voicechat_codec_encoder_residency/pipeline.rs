//! Native finite encoder cuts preserve codec-owned channel/time geometry.
use super::*;
use eredu_backend_mlx::native::materialize_ear_encoder_pipeline;
use eredu_codec::{
    ear_encoder_distribution::PreparedEncoderPipelineBindings,
    ear_encoder_layered::{encoder_parameter_description, EncoderArchitecture},
    ear_encoder_partition::{EncoderBoundarySchema, EncoderPipelineCuts},
};
use eredu_core::{
    checkpoint::TensorDtype, CollectiveGroupId, CompletionCancellationMode, DistributedCommitEpoch,
    DistributedCommitOutcome, ParallelRankTopology, ParallelTopology,
};
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitectureParameters, ArchitecturePartition,
    ArchitectureStatePartitionPlan, BoundaryDimensionContract, BoundaryRoleContract,
    BoundaryTensorDimension, BoundaryTensorDtype, CommunicationCompletionPolicy,
    CommunicationGroupDescriptor, CommunicationGroupRequirements, CommunicationManifest,
    CommunicationOperation, CommunicationOperationRequirement, CommunicationRouteDescriptor,
    CommunicationRouteId, CommunicationTensorLimits, LayeredPartitionDriver,
    PartitionBoundaryRoute, PartitionOutputPublication, PartitionOwnership,
    PartitionedExecutionPlan, PipelineActivationDtype, PipelineWireContract,
    RoleExactBoundaryContract,
};

const WORKER: &str = "EREDU_EAR_ENCODER_PIPELINE_WORLD_SIZE";
const DIRECTORY: &str = "EREDU_EAR_ENCODER_PIPELINE_DIRECTORY";

#[test]
fn native_ear_encoder_pipeline_worker() {
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
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let case = &fixture["encoders"]["compact"];
    let config = geometry(&case["geometry"]);
    let spectrum = tensor(&case["packed"]);
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let source = checkpoint(case, false, &directory.path().join("encoder.safetensors"));
    // Cut3 carries [B,2,T]; cut4 carries [B,4,T/2]. Exercise every
    // residency policy at cut3, then the distinct downsample boundary once.
    for (cut, mode) in [(3, 0), (3, 1), (3, 2), (4, 0)] {
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
        let prefix = "tts_model.audio_codec.encoder";
        let description = encoder_parameter_description(&config, prefix).unwrap();
        let complete = if let Some(layout) = &local {
            EncoderArchitecture::<Backend>::new_parallel(config.clone(), prefix, layout).unwrap()
        } else {
            EncoderArchitecture::<Backend>::new(config.clone(), prefix).unwrap()
        };
        let range = if stage == 0 { 0..cut } else { cut..7 };
        let partition = ArchitecturePartition::from_description(
            &description,
            [(description.graph().groups()[0].id(), range.clone())],
            PartitionOwnership::new(stage == 0, stage == 1, std::iter::empty::<String>()).unwrap(),
            &complete.state_layout().unwrap(),
            &ArchitectureStatePartitionPlan::new([]),
            (),
            EncoderBoundarySchema::new(&config).unwrap(),
        )
        .unwrap();
        assert!(
            partition.state().is_none(),
            "finite encoder owns no streaming cache"
        );
        let selected = global.into_partition(&partition).unwrap();
        let keys = selected.source().source_keys();
        assert!(keys.iter().all(|key| range
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
            [80, 216, 216, 64, 688, 688, 192]
        } else {
            [40, 136, 136, 32, 400, 400, 96]
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
        let cuts = EncoderPipelineCuts::new(&config, routes.iter().map(|route| (route.route, cut)))
            .unwrap();
        let driver =
            LayeredPartitionDriver::new_with_state_ownership(&partition, 0, range, false).unwrap();
        let plan = PartitionedExecutionPlan::new(
            description.graph().clone(),
            vec![(ArchitectureGroupKind::AudioEncoder, false)],
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
        if cut == 3 && mode == 0 {
            let stage_bindings = || {
                let bindings = prepared(config.clone(), source.clone());
                let bindings = match &local {
                    Some(layout) => bindings.into_parallel(layout).unwrap(),
                    None => bindings,
                };
                bindings.into_partition(&partition).unwrap()
            };
            let wrong_cut = EncoderPipelineCuts::new(
                &config,
                plan.routes().iter().map(|route| (route.route, 4)),
            )
            .unwrap();
            assert!(matches!(PreparedEncoderPipelineBindings::new(
                stage_bindings(), partition.clone(), wrong_cut, manifest.clone(), plan.clone(),
                tensor_group, PipelineActivationDtype::Float32,
            ), Err(eredu_codec::ear_encoder_distribution::EncoderPipelinePreparationError::Contract(_))));
            if tensor_ranks == 2 {
                // The local lane's cut still matches, but its TP peer owns a
                // different range and would enter a different collective order.
                let other = CommunicationRouteId::new(100 + (1 - rank % 2) as u64);
                let skewed = EncoderPipelineCuts::new(
                    &config,
                    plan.routes()
                        .iter()
                        .map(|route| (route.route, if route.route == other { 4 } else { 3 })),
                )
                .unwrap();
                assert!(matches!(PreparedEncoderPipelineBindings::new(
                    stage_bindings(), partition.clone(), skewed, manifest.clone(), plan.clone(),
                    tensor_group, PipelineActivationDtype::Float32,
                ), Err(eredu_codec::ear_encoder_distribution::EncoderPipelinePreparationError::Contract(_))));
            }
        }
        let prepared = PreparedEncoderPipelineBindings::new(
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
        let (mut model, metadata) =
            materialize_ear_encoder_pipeline(prepared, &world, residency, &weights, &stream)
                .unwrap();
        assert_eq!(metadata.static_device_bytes(), 0);
        assert_eq!(metadata.layer_parameter_bytes(), total);
        assert_eq!(metadata.residency(), residency.execution_residency());
        let descriptor = [
            cut as u64,
            mode,
            spectrum.shape()[0] as u64,
            spectrum.shape()[2] as u64,
            0,
            0,
            0,
            0,
        ];
        if cut == 3 && mode == 2 {
            let before = model
                .residency_report()
                .unwrap()
                .weight_store()
                .physical_reads;
            let mut disagreement = descriptor;
            disagreement[7] = rank as u64;
            assert!(model
                .forward(
                    &spectrum,
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
                before,
                "descriptor disagreement precedes native parameter acquisition"
            );
        }
        for invocation in 1..=2 {
            let epoch = DistributedCommitEpoch::new(invocation).unwrap();
            let (latent, outcome) = model
                .forward(&spectrum, descriptor, epoch, &stream)
                .unwrap();
            assert_eq!(outcome, DistributedCommitOutcome::Committed(epoch));
            compare(
                &latent.to_f32_vec(&stream).unwrap(),
                &case["latent"],
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
                    <= budget
            );
        }
        if let Some(report) = model.dense_stream_report().unwrap() {
            assert_eq!(report.prefill_forwards() + report.decode_forwards(), 2);
        }
    }
}

fn boundary_contract(config: &EncoderConfig, cut: usize) -> RoleExactBoundaryContract {
    let schema = EncoderBoundarySchema::new(config)
        .unwrap()
        .at_cut(cut)
        .unwrap()
        .wire_schema()
        .unwrap();
    let resolved = schema.resolve(2, 24).unwrap();
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
    config: &EncoderConfig,
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
                    "pipeline::native_ear_encoder_pipeline_worker",
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
            "EAR encoder pipeline world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_ear_encoder_pipeline_pp2_and_tp2_pp2_match_nvidia_latents() {
    launch(2);
    launch(4);
}
