//! Real neutral pipeline transport, output publication and failure agreement.
use super::voicechat_transformer::{geometry, CausalCache};
use super::voicechat_tts_generator::{compare, compare_cache, tensor};
use super::*;
use eredu_architectures::voicechat::{
    tts_transformer::TransformerConfig,
    tts_transformer_layered::{
        tts_transformer_parameter_description, OwnedTransformerRuntimeState,
        TransformerArchitecture, TransformerBoundarySchema, TransformerInput, TransformerUnit,
    },
};
use eredu_core::{checkpoint::TensorDtype, DistributedCommitEpoch};
use eredu_runtime::PipelinePartitionExecutor;
use eredu_runtime::{
    ArchitectureBoundary, ArchitectureGroupKind, BoundaryDimensionContract, BoundaryRoleContract,
    BoundaryTensorDimension, BoundaryTensorDtype, CommunicationRouteDescriptor,
    CommunicationRouteId, ExecutionResidency, OpaqueBoundaryTransport, OpaqueFailureAgreement,
    OpaqueOutputPublisher, PartitionBoundaryRoute, PartitionOutputPublication,
    PartitionedExecutionPlan, PartitionedTextExecution, PartitionedTextRuntime,
    PipelineActivationDtype, PipelineWireContract, ReplicatedTextExecutionStrategy,
    RoleExactBoundaryContract,
};
use serde_json::Value;

type Arch = TransformerArchitecture<NumericBackend>;
type State = OwnedTransformerRuntimeState<NumericBackend, CausalCache>;
type Unit = TransformerUnit<NumericBackend>;
type Policy = NumericReplicatedPolicy<Unit>;
type Executor =
    PipelinePartitionExecutor<Arch, NumericBackend, State, Policy, NumericPartitionTensorAllocator>;
type Runtime = PartitionedTextRuntime<
    Arch,
    NumericBackend,
    State,
    Policy,
    Executor,
    u64,
    u64,
    NumericCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;
type Engine = PartitionedTextExecution<
    Executor,
    u64,
    u64,
    NumericCommunicationMetadata,
    OpaqueBoundaryTransport,
    OpaqueOutputPublisher,
    OpaqueFailureAgreement,
>;

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
                    CommunicationGroupRequirements::new([tensor_requirement(
                        CommunicationOperation::AllReduceSum,
                        vec![TensorDtype::F32],
                    )])
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

fn selected_weights(
    data: &Value,
    layout: &LocalModelLayout,
    duplicate_kv: bool,
) -> BTreeMap<String, NumericTensor> {
    data["parameters"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, value)| {
            let mut value = tensor(value);
            if duplicate_kv
                && (name.ends_with(".k_proj.weight") || name.ends_with(".v_proj.weight"))
            {
                value = NumericTensor::concatenate(
                    &[value.clone(), value],
                    0,
                    &NumericContext::default(),
                )
                .unwrap();
            }
            (
                name.clone(),
                select_parameter(&value, layout.tensor(name).unwrap().placement()).unwrap(),
            )
        })
        .collect()
}

fn advance(
    runtime: &mut Runtime,
    engine: &mut Engine,
    state: &mut State,
    input: &NumericTensor,
    mask: Option<&NumericTensor>,
    epoch: u64,
    context: &NumericContext,
) -> Result<NumericTensor, String> {
    let mut observer = eredu_runtime::NoopObserver;
    let (output, _) = <Engine as ReplicatedTextExecutionStrategy<
        Arch,
        NumericBackend,
        State,
        Policy,
        Policy,
    >>::forward_with_observer(
        engine,
        runtime,
        TransformerInput {
            embedded: input,
            key_mask: mask,
        },
        state,
        if epoch == 1 {
            ExpertPass::Prefill
        } else {
            ExpertPass::Decode
        },
        context,
        &mut observer,
    )
    .map_err(|error| error.to_string())?;
    let output = <Engine as ReplicatedTextExecutionStrategy<
        Arch,
        NumericBackend,
        State,
        Policy,
        Policy,
    >>::observe_output(runtime, &output, &mut observer, context)
    .map_err(|error| error.to_string())?;
    let output = <Engine as ReplicatedTextExecutionStrategy<
        Arch,
        NumericBackend,
        State,
        Policy,
        Policy,
    >>::publish_observed_output(runtime, output, context)
    .map_err(|error| error.to_string())?;
    let epoch = DistributedCommitEpoch::new(epoch).unwrap();
    assert_eq!(
        <Engine as ReplicatedTextExecutionStrategy<Arch, NumericBackend, State, Policy, Policy>>::commit_after_completion(runtime, epoch, context),
        DistributedCommitOutcome::Committed(epoch),
    );
    Ok(output)
}

#[test]
fn voicechat_transformer_distributed_pp2_and_tp2_pp2_transport_cached_state_and_failure() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TRANSFORMER_REFERENCE_JSON).unwrap();
    for causal in [true, false] {
        let data = &fixture[if causal { "gemma3" } else { "t5gemma" }];
        for tensor_ranks in [1, 2] {
            let mut dimensions = geometry(
                if causal {
                    &data["config"]
                } else {
                    &data["config"]["encoder"]
                },
                causal,
            );
            // Identical fixture KV heads exercise the ordinary grouped-head TP plan.
            if tensor_ranks == 2 {
                dimensions.num_key_value_heads = 2;
            }
            let config = TransformerConfig::new(&dimensions).unwrap();
            let world = Arc::new(NumericPartitionWorld::default());
            let barrier = Arc::new(std::sync::Barrier::new(tensor_ranks * 2));
            std::thread::scope(|scope| {
                let handles = (0..tensor_ranks * 2)
                    .map(|rank| {
                        let config = config.clone();
                        let world = Arc::clone(&world);
                        let barrier = Arc::clone(&barrier);
                        let fixture = &fixture;
                        scope.spawn(move || {
                            let stage = rank / tensor_ranks;
                            let tensor_rank = rank % tensor_ranks;
                            let description =
                                tts_transformer_parameter_description(&config).unwrap();
                            let groups = description
                                .groups()
                                .iter()
                                .map(|owned| owned.group().clone())
                                .collect::<Vec<_>>();
                            let layout =
                                numeric_local_layout(&groups, tensor_ranks, tensor_rank).unwrap();
                            let context = NumericContext::with_partition(
                                layout.clone(),
                                rank,
                                Arc::clone(&world),
                            );
                            let complete = if tensor_ranks > 1 {
                                Arch::new_parallel(config.clone(), &layout).unwrap()
                            } else {
                                Arch::new(config.clone()).unwrap()
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
                            let mut owned = Vec::new();
                            if !blocks.is_empty() {
                                owned.push((description.graph().groups()[0].id(), blocks.clone()));
                            }
                            if norm {
                                owned.push((description.graph().groups()[1].id(), 0..1));
                            }
                            let partition = ArchitecturePartition::from_description(
                                &description,
                                owned,
                                PartitionOwnership::new(
                                    stage == 0,
                                    norm,
                                    std::iter::empty::<String>(),
                                )
                                .unwrap(),
                                &complete.state_layout().unwrap(),
                                &ArchitectureStatePartitionPlan::new([
                                    ArchitectureStatePartitionRule::group_units(0, 0..2),
                                ]),
                                (),
                                TransformerBoundarySchema::new(&config),
                            )
                            .unwrap();
                            let architecture = Arch::from_partition(
                                config.clone(),
                                (tensor_ranks > 1).then_some(&layout),
                                &partition,
                            )
                            .unwrap();
                            let state_arch = Arch::from_partition(
                                config.clone(),
                                (tensor_ranks > 1).then_some(&layout),
                                &partition,
                            )
                            .unwrap();
                            let cache = if causal {
                                blocks
                                    .clone()
                                    .map(|index| {
                                        CausalCache(NumericCache::new((index == 0).then_some(3)))
                                    })
                                    .collect()
                            } else {
                                vec![]
                            };
                            let mut committed = State::for_partition(&state_arch, cache).unwrap();
                            let values = selected_weights(data, &layout, tensor_ranks > 1);
                            let units = partition
                                .units()
                                .map(|address| {
                                    let mut unit = architecture
                                        .build_unit(address.group(), address.index(), &context)
                                        .unwrap();
                                    replace_numeric_parameters(&mut unit, &values);
                                    unit
                                })
                                .collect();
                            let policy = Policy::resident(units);
                            let (manifest, routes, session, tensor_group) =
                                communication(&config, tensor_ranks, rank);
                            world.realize_manifest(&manifest).unwrap();
                            let parallel = tensor_group.map(|group| {
                                numeric_parallel_context(&world, &manifest, group).unwrap()
                            });
                            let mut drivers = vec![None, None];
                            for group in partition.groups() {
                                drivers[group.group_index()] = Some(
                                    LayeredPartitionDriver::new_with_state_ownership(
                                        &partition,
                                        group.group_index(),
                                        group.global_units(),
                                        group.group_index() == 0,
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
                            let executor = Executor::new(
                                architecture,
                                policy,
                                partition.units().collect(),
                                parallel,
                                NumericPartitionTensorAllocator,
                                PipelineActivationDtype::Float32,
                            )
                            .unwrap();
                            let communication =
                                numeric_partition_communication(&world, manifest).unwrap();
                            let mut runtime = Runtime::new(
                                plan,
                                executor,
                                communication,
                                context.clone(),
                                OpaqueBoundaryTransport,
                                OpaqueOutputPublisher,
                                OpaqueFailureAgreement,
                                ExecutionResidency::FullyResident,
                                None,
                            )
                            .unwrap();
                            let mut engine = Engine::new();
                            let calls = if causal {
                                data["chunks"]
                                    .as_array()
                                    .unwrap()
                                    .iter()
                                    .collect::<Vec<_>>()
                            } else {
                                vec![&data["batches"][0], &data["batches"][0]]
                            };
                            let mask = (!causal).then(|| {
                                tensor(&data["attention_mask"])
                                    .equal_i32(1, &context)
                                    .unwrap()
                            });
                            for (step, call) in calls.iter().enumerate() {
                                let input = tensor(&call["input"]);
                                if step == 1 {
                                    barrier.wait();
                                    let before = world.lifecycle_counts();
                                    if rank == 0 {
                                        world.arm_fault_one_per_agreement_group(
                                            NumericPartitionFault::LocalExecution,
                                        );
                                    }
                                    barrier.wait();
                                    let mut branch = committed
                                        .try_map_caches(|caches| {
                                            Ok::<_, std::convert::Infallible>(caches.to_vec())
                                        })
                                        .unwrap();
                                    assert!(advance(
                                        &mut runtime,
                                        &mut engine,
                                        &mut branch,
                                        &input,
                                        mask.as_ref(),
                                        step as u64 + 1,
                                        &context
                                    )
                                    .is_err());
                                    barrier.wait();
                                    let after = world.lifecycle_counts();
                                    assert_eq!(
                                        before.0.get(&NumericOpaqueOperation::Broadcast),
                                        after.0.get(&NumericOpaqueOperation::Broadcast)
                                    );
                                    assert_eq!(
                                        before.3, after.3,
                                        "rejected execution publishes no committed state"
                                    );
                                    if causal {
                                        let expected = Value::Array(
                                            calls[0]["cache"].as_array().unwrap()[blocks.clone()]
                                                .to_vec(),
                                        );
                                        compare_cache(committed.caches(), &expected, fixture);
                                    }
                                    barrier.wait();
                                }
                                let mut branch = committed
                                    .try_map_caches(|caches| {
                                        Ok::<_, std::convert::Infallible>(caches.to_vec())
                                    })
                                    .unwrap();
                                let output = advance(
                                    &mut runtime,
                                    &mut engine,
                                    &mut branch,
                                    &input,
                                    mask.as_ref(),
                                    step as u64 + 1,
                                    &context,
                                )
                                .unwrap();
                                compare(&output, &call["output"], fixture);
                                if causal {
                                    let expected_position: i32 = calls[..=step]
                                        .iter()
                                        .map(|call| tensor(&call["input"]).dim(1))
                                        .sum();
                                    assert_eq!(
                                        state_arch.validate_causal_state(&branch).unwrap(),
                                        expected_position
                                    );
                                    assert_eq!(
                                        committed.causal_position().unwrap(),
                                        expected_position - input.dim(1)
                                    );
                                    let expected = Value::Array(
                                        call["cache"].as_array().unwrap()[blocks.clone()].to_vec(),
                                    );
                                    compare_cache(branch.caches(), &expected, fixture);
                                }
                                committed = branch;
                                world.record_commit(rank);
                            }
                        })
                    })
                    .collect::<Vec<_>>();
                for handle in handles {
                    handle.join().unwrap();
                }
            });
            assert!(world
                .trace()
                .iter()
                .any(|event| event.operation == NumericOpaqueOperation::PointToPoint));
            assert!(world
                .trace()
                .iter()
                .any(|event| event.operation == NumericOpaqueOperation::Broadcast));
        }
    }
}
