//! Native Ring transport between separately materialized transformer stages.
use super::*;
use eredu_architectures::voicechat::{
    tts_transformer_distribution::PreparedTtsTransformerPipelineBindings,
    tts_transformer_layered::{
        tts_transformer_parameter_description, TransformerArchitecture, TransformerBoundarySchema,
        TransformerInput,
    },
};
use eredu_backend_mlx::native::materialize_voicechat_tts_transformer_pipeline;
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
    RoleExactBoundaryContract, RuntimeLayerState,
};

const WORKER: &str = "EREDU_TTS_PIPELINE_WORLD_SIZE";
const GENERATOR: &str = "EREDU_TTS_GENERATOR_PIPELINE";

#[path = "generator.rs"]
mod generator;

const DIRECTORY: &str = "EREDU_TTS_PIPELINE_DIRECTORY";

fn boundary_contract(config: &TransformerConfig) -> RoleExactBoundaryContract {
    let schema = TransformerBoundarySchema::new(config)
        .wire_schema()
        .unwrap();
    let resolved = schema.resolve(2, 16).unwrap();
    let roles = std::iter::once(schema.primary())
        .chain(schema.auxiliary())
        .zip(std::iter::once(resolved.primary()).chain(resolved.auxiliary()))
        .map(|(symbolic, resolved)| {
            let dtype = match symbolic.dtype() {
                BoundaryTensorDtype::Activation | BoundaryTensorDtype::Float32 => TensorDtype::F32,
                BoundaryTensorDtype::Int32 => TensorDtype::I32,
                other => panic!("unexpected fixture wire type {other:?}"),
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
                    other => panic!("unexpected fixture wire dimension {other:?}"),
                })
                .collect();
            BoundaryRoleContract::symbolic(symbolic.role(), dtype, shape).unwrap()
        });
    RoleExactBoundaryContract::new(schema.identity(), roles).unwrap()
}

fn communication(
    config: &TransformerConfig,
    tensor_ranks: usize,
    rank: usize,
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
            3
        } else {
            1
        };
        let limits = CommunicationTensorLimits::new(count, 3, 512, None).unwrap();
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
        for stage in 0..2 {
            let members = (stage * tensor_ranks..(stage + 1) * tensor_ranks).collect::<Vec<_>>();
            let local = members.iter().position(|&member| member == rank);
            let group = CollectiveGroupId::new(10 + stage as u32);
            if local.is_none() {
                continue;
            }
            local_tensor = Some(group);
            groups.push(
                CommunicationGroupDescriptor::new(
                    group,
                    1,
                    members,
                    local,
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
        }
    }
    let (source_group, destination_group) = if tensor_ranks == 1 { (0, 1) } else { (0, 0) };
    let mut routes = Vec::new();
    let mut descriptors = Vec::new();
    for lane in 0..tensor_ranks {
        let route = CommunicationRouteId::new(100 + lane as u64);
        routes.push(PartitionBoundaryRoute {
            source_group,
            destination_group,
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
            .with_boundary_contract(boundary_contract(config))
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

fn compare_local_cache(
    actual: &[MlxKeyValueLayerState],
    ordinary: &MlxKeyValueState,
    blocks: std::ops::Range<usize>,
    fixture: &Value,
    stream: &Stream,
) {
    let ordinary: &[MlxKeyValueLayerState] = ordinary.as_ref();
    assert_eq!(actual.len(), blocks.len());
    for (actual, expected) in actual.iter().zip(&ordinary[blocks]) {
        assert_eq!(actual.offset(), expected.offset());
        let actual =
            RuntimeLayerState::<MlxNeuralBackend>::retained_values(actual).collect::<Vec<_>>();
        let expected =
            RuntimeLayerState::<MlxNeuralBackend>::retained_values(expected).collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_eq!(actual.shape(), expected.shape());
            compare(
                &actual.to_f32_vec(stream).unwrap(),
                &serde_json::json!({"values": expected.to_f32_vec(stream).unwrap()}),
                fixture,
            );
        }
    }
}

#[test]
fn native_tts_transformer_pipeline_worker() {
    let Ok(size) = std::env::var(WORKER) else {
        return;
    };
    let size: usize = size.parse().unwrap();
    let world = safemlx::distributed::init(true, safemlx::distributed::Backend::Ring).unwrap();
    assert_eq!(world.size(), size);
    let rank = world.rank();
    let tensor_ranks = size / 2;
    let stage = rank / tensor_ranks;
    let tensor_rank = rank % tensor_ranks;
    let weights = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mut fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    let generator_fixture = std::env::var(GENERATOR)
        .ok()
        .filter(|value| value == "1")
        .map(|_| {
            let transition: Value =
                serde_json::from_str(eredu_evaluation::voicechat::TTS_TRANSITION_REFERENCE_JSON)
                    .unwrap();
            fixture["gemma3"]["config"] = transition["transformer_config"].clone();
            fixture["gemma3"]["parameters"] = transition["parameters"].clone();
            transition
        });
    let selected_fixture = if tensor_ranks > 1 {
        duplicated_kv_fixture(&fixture)
    } else {
        fixture.clone()
    };
    let directory = tempfile::tempdir_in(std::env::var_os(DIRECTORY).unwrap()).unwrap();
    let original_source = checkpoint(
        &fixture,
        false,
        &directory.path().join("original.safetensors"),
    );
    let selected_source = checkpoint(
        &selected_fixture,
        false,
        &directory.path().join("selected.safetensors"),
    );
    for causal in [true, false] {
        if !causal && generator_fixture.is_some() {
            continue;
        }
        let family = if causal { "gemma3" } else { "t5gemma" };
        let data = &fixture[family];
        let configuration = |part: &Value| {
            TransformerConfig::new(&geometry(
                if causal {
                    &part["config"]
                } else {
                    &part["config"]["encoder"]
                },
                causal,
            ))
            .unwrap()
        };
        let config = configuration(&selected_fixture[family]);
        let original_config = configuration(data);
        for mode in 0..3 {
            let global = prepared(config.clone(), selected_source.clone());
            let (global, local) = if tensor_ranks > 1 {
                let topology = ParallelRankTopology::new(
                    ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
                    tensor_rank,
                )
                .unwrap();
                let (global, local) = global.select_parallel(topology).unwrap();
                (global, Some(local))
            } else {
                (global, None)
            };
            let complete = if let Some(local) = &local {
                TransformerArchitecture::<MlxNeuralBackend>::new_parallel(config.clone(), local)
                    .unwrap()
            } else {
                TransformerArchitecture::<MlxNeuralBackend>::new(config.clone()).unwrap()
            };
            let (blocks, norm) = if tensor_ranks == 1 {
                if stage == 0 {
                    (0..2, false)
                } else {
                    (2..2, true)
                }
            } else if stage == 0 {
                (0..1, false)
            } else {
                (1..2, true)
            };
            let description = tts_transformer_parameter_description(&config).unwrap();
            let mut groups = Vec::new();
            if !blocks.is_empty() {
                groups.push((description.graph().groups()[0].id(), blocks.clone()));
            }
            if norm {
                groups.push((description.graph().groups()[1].id(), 0..1));
            }
            let partition = ArchitecturePartition::from_description(
                &description,
                groups,
                PartitionOwnership::new(stage == 0, norm, std::iter::empty::<String>()).unwrap(),
                &complete.state_layout().unwrap(),
                &ArchitectureStatePartitionPlan::new([
                    ArchitectureStatePartitionRule::group_units(0, 0..2),
                ]),
                (),
                TransformerBoundarySchema::new(&config),
            )
            .unwrap();
            let selected = global.into_partition(&partition).unwrap();
            let largest = selected
                .unit_bindings()
                .iter()
                .map(|(_, unit)| binding_bytes(unit).unwrap())
                .max()
                .unwrap();
            let total: u64 = selected
                .unit_bindings()
                .iter()
                .map(|(_, unit)| binding_bytes(unit).unwrap())
                .sum();
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
                communication(&config, tensor_ranks, rank);
            let mut drivers = vec![None, None];
            for owned in partition.groups() {
                drivers[owned.group_index()] = Some(
                    LayeredPartitionDriver::new_with_state_ownership(
                        &partition,
                        owned.group_index(),
                        owned.global_units(),
                        owned.group_index() == 0,
                    )
                    .unwrap(),
                );
            }
            let plan = PartitionedExecutionPlan::new(
                description.graph().clone(),
                vec![
                    (
                        if causal {
                            ArchitectureGroupKind::Decoder
                        } else {
                            ArchitectureGroupKind::AudioEncoder
                        },
                        false,
                    ),
                    (ArchitectureGroupKind::Projector, false),
                ],
                drivers,
                routes,
                Some(PartitionOutputPublication {
                    group: session,
                    owner_rank: tensor_ranks,
                }),
                Some(session),
                PipelineWireContract::new(PipelineActivationDtype::Float32),
            )
            .unwrap();
            let prepared = PreparedTtsTransformerPipelineBindings::new(
                selected,
                partition,
                manifest,
                plan,
                tensor_group,
                PipelineActivationDtype::Float32,
            )
            .unwrap();
            let (mut model, mut state, metadata) = materialize_voicechat_tts_transformer_pipeline(
                prepared, &world, residency, &weights, &stream,
            )
            .unwrap();
            assert_eq!(metadata.static_device_bytes(), 0);
            assert_eq!(metadata.layer_parameter_bytes(), total);
            if blocks.is_empty() {
                assert!(model.state_layout().is_none());
            }
            if !causal {
                assert!(state.caches().is_empty());
            }
            if let Some(transition) = &generator_fixture {
                generator::run(
                    &world,
                    &mut model,
                    &mut state,
                    &config,
                    blocks.clone(),
                    transition,
                    mode,
                    &stream,
                );
                if !residency.is_fully_resident() {
                    assert!(
                        model
                            .residency_report()
                            .unwrap()
                            .offload()
                            .peak_resident_bytes()
                            .get(MemoryTier::Device)
                            <= budget
                    );
                }
                continue;
            }
            let (mut ordinary, _) = materialize_voicechat_tts_transformer(
                super::prepared(original_config.clone(), original_source.clone()),
                LayerWeightResidency::FullyResident,
                &weights,
                &stream,
            )
            .unwrap();
            let mut original_cache = causal.then(|| {
                MlxKeyValueState::device(
                    eredu_architectures::decoder::state_layout(&original_config).unwrap(),
                )
                .unwrap()
            });
            let calls = if causal {
                data["chunks"].as_array().unwrap()
            } else {
                data["batches"].as_array().unwrap()
            };
            let mask = (!causal).then(|| {
                tensor(&data["attention_mask"])
                    .equal_i32(1, &stream)
                    .unwrap()
            });
            for (step, call) in calls.iter().enumerate() {
                let input = tensor(&call["input"]);
                let expected = if let Some(cache) = &mut original_cache {
                    ordinary
                        .forward_causal(&input, cache.as_mut(), &stream)
                        .unwrap()
                } else {
                    ordinary
                        .forward_encoder(&input, mask.as_ref().unwrap(), &stream)
                        .unwrap()
                };
                compare(
                    &expected.to_f32_vec(&stream).unwrap(),
                    &call["output"],
                    &fixture,
                );
                let epoch = DistributedCommitEpoch::new(step as u64 + 1).unwrap();
                let descriptor = [
                    u64::from(causal),
                    mode,
                    step as u64,
                    input.shape()[0] as u64,
                    input.shape()[1] as u64,
                    8,
                    0,
                    0,
                ];
                let (output, outcome) = model
                    .forward(
                        TransformerInput {
                            embedded: &input,
                            key_mask: mask.as_ref(),
                        },
                        &mut state,
                        descriptor,
                        epoch,
                        &stream,
                    )
                    .unwrap();
                assert_eq!(outcome, DistributedCommitOutcome::Committed(epoch));
                compare(
                    &output.to_f32_vec(&stream).unwrap(),
                    &call["output"],
                    &fixture,
                );
                if let Some(cache) = &original_cache {
                    compare_local_cache(state.caches(), cache, blocks.clone(), &fixture, &stream);
                }
            }
            if !residency.is_fully_resident() {
                assert!(
                    model
                        .residency_report()
                        .unwrap()
                        .offload()
                        .peak_resident_bytes()
                        .get(MemoryTier::Device)
                        <= budget
                );
            }
            if causal && mode == 0 {
                let input = tensor(&calls.last().unwrap()["input"]);
                let epoch = DistributedCommitEpoch::new(calls.len() as u64 + 1).unwrap();
                let mismatched = [9, 0, 0, 0, 0, 0, 0, rank as u64];
                assert!(model
                    .forward(
                        TransformerInput {
                            embedded: &input,
                            key_mask: None
                        },
                        &mut state,
                        mismatched,
                        epoch,
                        &stream
                    )
                    .is_err());
                compare_local_cache(
                    state.caches(),
                    original_cache.as_ref().unwrap(),
                    blocks.clone(),
                    &fixture,
                    &stream,
                );
                let expected = ordinary
                    .forward_causal(&input, original_cache.as_mut().unwrap().as_mut(), &stream)
                    .unwrap();
                let (output, outcome) = model
                    .forward(
                        TransformerInput {
                            embedded: &input,
                            key_mask: None,
                        },
                        &mut state,
                        [9; 8],
                        epoch,
                        &stream,
                    )
                    .unwrap();
                assert_eq!(outcome, DistributedCommitOutcome::Committed(epoch));
                compare(
                    &output.to_f32_vec(&stream).unwrap(),
                    &serde_json::json!({"values": expected.to_f32_vec(&stream).unwrap()}),
                    &fixture,
                );
                compare_local_cache(
                    state.caches(),
                    original_cache.as_ref().unwrap(),
                    blocks.clone(),
                    &fixture,
                    &stream,
                );
            }
        }
    }
}

fn launch(size: usize, generator: bool) {
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
                    "pipeline::native_tts_transformer_pipeline_worker",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env(WORKER, size.to_string())
                .env(GENERATOR, if generator { "1" } else { "" })
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
            "pipeline world {size} rank {rank}: {}\n{}\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_tts_transformer_pipeline_pp2_and_tp2_pp2_preserves_hf_outputs_and_cache() {
    launch(2, false);
    launch(4, false);
}

#[test]
#[ignore = "spawns two/four local MLX Ring ranks and opens loopback sockets; run explicitly"]
fn native_tts_generator_pipeline_pp2_tp2pp2_matches_cached_refinement() {
    launch(2, true);
    launch(4, true);
}
