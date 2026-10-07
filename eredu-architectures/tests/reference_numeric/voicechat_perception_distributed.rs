//! Perception cache ownership and packed outputs over shared opaque pipeline transport.
use super::voicechat_perception_residency::{
    assert_state, assert_tensor, dimensions, parameters, tensor,
};
use super::*;
use eredu_architectures::voicechat::{
    perception::{Perception, PerceptionDimensions, PerceptionState},
    perception_layered::{
        perception_parameter_description, unpack_perception_pipeline_output,
        PerceptionArchitecture, PerceptionBoundarySchema, PerceptionInput, PerceptionRuntimeState,
        PerceptionUnit,
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

type Arch = PerceptionArchitecture<NumericBackend>;
type State = PerceptionRuntimeState<NumericTensor>;
type Policy = NumericReplicatedPolicy<PerceptionUnit<NumericBackend>>;
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

fn expected_value(value: &NumericTensor) -> Value {
    serde_json::json!({"shape": value.shape(), "values": value.data})
}

fn compare_state(
    state: &State,
    expected: &PerceptionState<NumericTensor>,
    blocks: std::ops::Range<usize>,
    fixture: &Value,
) {
    assert_eq!(state.frames(), expected.frames());
    assert_eq!(state.blocks().len(), blocks.len());
    for (actual, expected) in state.blocks().iter().zip(&expected.blocks()[blocks]) {
        assert_eq!(actual.valid_channel(), expected.valid_channel());
        for (actual, expected) in [
            (actual.channel(), expected.channel()),
            (actual.convolution(), expected.convolution()),
        ] {
            match (actual, expected) {
                (Some(actual), Some(expected)) => {
                    assert_tensor(actual, &expected_value(expected), fixture)
                }
                (None, None) => {}
                _ => panic!("partition changed cache presence"),
            }
        }
    }
}

#[test]
fn voicechat_perception_partition_rejects_stale_frontier_and_stateless_owner_aliasing() {
    use eredu_architectures::voicechat::{
        perception::PerceptionError, perception_layered::PerceptionBoundary,
    };
    use eredu_runtime::{
        LayeredPartitionBeginError, LayeredPartitionInput, LayeredPartitionOutput,
        LayerwisePartitionRuntimeError,
    };
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let dimensions = dimensions(&fixture);
    let description = perception_parameter_description(dimensions).unwrap();
    let complete = Arch::new(dimensions).unwrap();
    let partition = |groups: &[usize], input, output| {
        ArchitecturePartition::from_description(
            &description,
            groups
                .iter()
                .map(|&index| (description.graph().groups()[index].id(), 0..1)),
            PartitionOwnership::new(input, output, std::iter::empty::<String>()).unwrap(),
            &complete.state_layout().unwrap(),
            &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                1,
                0..1,
            )]),
            (),
            PerceptionBoundarySchema::new(dimensions),
        )
        .unwrap()
    };

    // These owners both have no Conformer tensors and a zero frontier. Their
    // different stage roles still prevent a state branch from changing owner.
    let subsampling = Arch::new_partition(dimensions, None, &partition(&[0], true, false)).unwrap();
    let projection = Arch::new_partition(dimensions, None, &partition(&[2], false, true)).unwrap();
    let subsampling_state = State::for_partition(&subsampling, None).unwrap();
    assert!(subsampling_state.blocks().is_empty());
    assert!(matches!(
        State::for_partition(&projection, Some(&subsampling_state)),
        Err(PerceptionError::State(_)),
    ));

    let context = NumericContext::default();
    let selected = partition(&[1, 2], false, true);
    let architecture = Arch::new_partition(dimensions, None, &selected).unwrap();
    let mut state = State::for_partition(&architecture, None).unwrap();
    let values = parameters(&fixture);
    let units = selected
        .units()
        .map(|address| {
            let mut unit = architecture
                .build_unit(address.group(), address.index(), &context)
                .unwrap();
            replace_numeric_parameters(&mut unit, &values);
            unit
        })
        .collect();
    let mut runtime = architecture
        .into_partition_runtime(Policy::resident(units), &selected)
        .unwrap();
    let execute = |runtime: &mut eredu_runtime::LayerwisePartitionRuntime<
        Arch,
        NumericBackend,
        State,
        Policy,
    >,
                   state: &mut State,
                   input: NumericTensor,
                   frontier: NumericTensor| {
        runtime.forward_with_preparation(state, None, &context, |architecture, state, expected| {
            architecture.begin_prepared_partition(
                LayeredPartitionInput::Hidden {
                    hidden: input,
                    auxiliary: PerceptionBoundary {
                        frame_frontier: frontier,
                    },
                },
                0,
                Some(1),
                state,
                expected,
                None,
                &context,
            )
        })
    };
    let integer_header = |words: &[i32]| {
        NumericTensor::from_i32_slice(words, &[words.len() as i32], &context).unwrap()
    };
    // Literal upstream ABI: u64 start/end low/high words followed by the
    // encoded count. Features come directly from the independent subsampler.
    let output = execute(
        &mut runtime,
        &mut state,
        tensor(&fixture["subsampling"]["steps"][0]["output"]),
        integer_header(&[0, 0, 1, 0, 1]),
    )
    .unwrap();
    let LayeredPartitionOutput::Final { output: packed, .. } = output else {
        panic!("output owner returned a boundary");
    };
    let (_, projected) = unpack_perception_pipeline_output(dimensions, &packed, &context).unwrap();
    assert_tensor(
        &projected,
        &fixture["perception"]["steps"][0]["projected_output"],
        &fixture,
    );
    let before = state.clone();
    let input = tensor(&fixture["subsampling"]["steps"][1]["output"]);
    let malformed = [
        integer_header(&[0, 0, 1, 0, 1]),                      // stale start
        integer_header(&[1, 0, 3, 0, 1]),                      // inconsistent end
        integer_header(&[1, 0, 2, 0]),                         // missing count
        NumericTensor::new(vec![5], vec![1., 0., 2., 0., 1.]), // wrong dtype
    ];
    for frontier in malformed {
        assert!(matches!(
            execute(&mut runtime, &mut state, input.clone(), frontier),
            Err(LayerwisePartitionRuntimeError::Begin(
                LayeredPartitionBeginError::Architecture(PerceptionError::State(_)),
            )),
        ));
        assert_eq!(state.frames(), before.frames());
        for (actual, expected) in state.blocks().iter().zip(before.blocks()) {
            assert_eq!(actual.valid_channel(), expected.valid_channel());
            for (actual, expected) in [
                (actual.channel().unwrap(), expected.channel().unwrap()),
                (
                    actual.convolution().unwrap(),
                    expected.convolution().unwrap(),
                ),
            ] {
                assert_eq!(actual.shape(), expected.shape());
                assert_eq!(actual.data, expected.data);
            }
        }
    }
    let output = execute(
        &mut runtime,
        &mut state,
        input,
        integer_header(&[1, 0, 2, 0, 1]),
    )
    .unwrap();
    let LayeredPartitionOutput::Final { output: packed, .. } = output else {
        panic!("output owner returned a boundary");
    };
    let (encoder, projected) =
        unpack_perception_pipeline_output(dimensions, &packed, &context).unwrap();
    assert_tensor(
        &encoder,
        &fixture["perception"]["steps"][1]["encoder_output"],
        &fixture,
    );
    assert_tensor(
        &projected,
        &fixture["perception"]["steps"][1]["projected_output"],
        &fixture,
    );
    assert_eq!(state.frames(), 2);
}

#[test]
fn voicechat_perception_distributed_pp2_and_tp2_pp2_preserve_cache_and_publication() {
    let original: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    // Both component boundaries use the independent one-block oracle. The
    // within-encoder cut adds a distinct nonzero block and compares the same
    // equations against ordinary execution, without inventing NVIDIA outputs.
    for cut in 0..3 {
        let mut dimensions = dimensions(&original);
        let mut values = parameters(&original);
        if cut == 2 {
            dimensions.layers = 2;
            let second = values
                .iter()
                .filter(|(name, _)| name.contains(".layers.0."))
                .map(|(name, value)| {
                    let mut value = value.clone();
                    for (index, item) in value.data.iter_mut().enumerate() {
                        *item = *item * 0.83 + (index % 5) as f32 * 0.003;
                    }
                    (name.replace(".layers.0.", ".layers.1."), value)
                })
                .collect::<Vec<_>>();
            values.extend(second);
        }
        for tensor_ranks in [1, 2] {
            for resident in [true, false] {
                let world = Arc::new(NumericPartitionWorld::default());
                let barrier = Arc::new(std::sync::Barrier::new(tensor_ranks * 2));
                std::thread::scope(|scope| {
                    let workers = (0..tensor_ranks * 2)
                        .map(|rank| {
                            let fixture = &original;
                            let values = &values;
                            let world = world.clone();
                            let barrier = barrier.clone();
                            scope.spawn(move || {
                                let stage = rank / tensor_ranks;
                                let description =
                                    perception_parameter_description(dimensions).unwrap();
                                let topology = ParallelRankTopology::new(
                                    ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
                                    rank % tensor_ranks,
                                )
                                .unwrap();
                                let layout = eredu_runtime::derive_partitioned_local_layout(
                                    &description,
                                    topology,
                                )
                                .unwrap();
                                let context = NumericContext::with_partition(
                                    layout.clone(),
                                    rank,
                                    world.clone(),
                                );
                                let complete = if tensor_ranks > 1 {
                                    Arch::new_parallel(dimensions, &layout).unwrap()
                                } else {
                                    Arch::new(dimensions).unwrap()
                                };
                                let (
                                    blocks,
                                    subsampling,
                                    projection,
                                    source_group,
                                    destination_group,
                                ) = match (cut, stage) {
                                    (0, 0) => (0..0, true, false, 0, 1),
                                    (0, _) => (0..1, false, true, 0, 1),
                                    (1, 0) => (0..1, true, false, 1, 2),
                                    (1, _) => (1..1, false, true, 1, 2),
                                    (2, 0) => (0..1, true, false, 1, 1),
                                    (2, _) => (1..2, false, true, 1, 1),
                                    _ => unreachable!(),
                                };
                                let mut owned = Vec::new();
                                if subsampling {
                                    owned.push((description.graph().groups()[0].id(), 0..1));
                                }
                                if !blocks.is_empty() {
                                    owned.push((
                                        description.graph().groups()[1].id(),
                                        blocks.clone(),
                                    ));
                                }
                                if projection {
                                    owned.push((description.graph().groups()[2].id(), 0..1));
                                }
                                let partition = ArchitecturePartition::from_description(
                                    &description,
                                    owned,
                                    PartitionOwnership::new(
                                        subsampling,
                                        projection,
                                        std::iter::empty::<String>(),
                                    )
                                    .unwrap(),
                                    &complete.state_layout().unwrap(),
                                    &ArchitectureStatePartitionPlan::new([
                                        ArchitectureStatePartitionRule::group_units(
                                            1,
                                            0..dimensions.layers,
                                        ),
                                    ]),
                                    (),
                                    PerceptionBoundarySchema::new(dimensions),
                                )
                                .unwrap();
                                let architecture = Arch::new_partition(
                                    dimensions,
                                    (tensor_ranks > 1).then_some(&layout),
                                    &partition,
                                )
                                .unwrap();
                                let state_arch = Arch::new_partition(
                                    dimensions,
                                    (tensor_ranks > 1).then_some(&layout),
                                    &partition,
                                )
                                .unwrap();
                                let mut committed =
                                    State::for_partition(&state_arch, None).unwrap();
                                let selected: BTreeMap<_, _> = values
                                    .iter()
                                    .map(|(name, value)| {
                                        (
                                            name.clone(),
                                            select_parameter(
                                                value,
                                                layout.tensor(name).unwrap().placement(),
                                            )
                                            .unwrap(),
                                        )
                                    })
                                    .collect();
                                let policy = if resident {
                                    let units = partition
                                        .units()
                                        .map(|address| {
                                            let mut unit = architecture
                                                .build_unit(
                                                    address.group(),
                                                    address.index(),
                                                    &context,
                                                )
                                                .unwrap();
                                            replace_numeric_parameters(&mut unit, &selected);
                                            unit
                                        })
                                        .collect();
                                    Policy::resident(units)
                                } else {
                                    let mut policy = Policy::bounded();
                                    policy.parameter_overrides = selected;
                                    policy
                                };
                                let (manifest, routes, session, tensor_group) = communication(
                                    dimensions,
                                    tensor_ranks,
                                    rank,
                                    source_group,
                                    destination_group,
                                );
                                world.realize_manifest(&manifest).unwrap();
                                let parallel = tensor_group.map(|id| {
                                    numeric_parallel_context(&world, &manifest, id).unwrap()
                                });
                                let mut drivers = vec![None, None, None];
                                for group in partition.groups() {
                                    drivers[group.group_index()] = Some(
                                        LayeredPartitionDriver::new_with_state_ownership(
                                            &partition,
                                            group.group_index(),
                                            group.global_units(),
                                            group.group_index() == 1,
                                        )
                                        .unwrap(),
                                    );
                                }
                                let plan = PartitionedExecutionPlan::new(
                                    description.graph().clone(),
                                    vec![
                                        (ArchitectureGroupKind::AudioEncoder, false),
                                        (ArchitectureGroupKind::AudioEncoder, false),
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
                                let bounded_policy = (!resident).then(|| policy.clone());
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
                                    if resident {
                                        ExecutionResidency::FullyResident
                                    } else {
                                        ExecutionResidency::LayerwiseHost
                                    },
                                    bounded_policy,
                                )
                                .unwrap();
                                let mut engine = Engine::new();
                                let ordinary_context = NumericContext::default();
                                let mut ordinary = Perception::<NumericBackend>::new(
                                    dimensions,
                                    &ordinary_context,
                                )
                                .unwrap();
                                replace_numeric_parameters(&mut ordinary, values);
                                let mut ordinary_state = PerceptionState::default();
                                let windows = fixture["subsampling"]["steps"].as_array().unwrap();
                                for step in 0..=windows.len() {
                                    let (input, drop) = if step == 0 {
                                        (tensor(&fixture["subsampling"]["input"]), 0)
                                    } else {
                                        (
                                            tensor(&windows[step - 1]["input"]),
                                            windows[step - 1]["drop"].as_u64().unwrap() as usize,
                                        )
                                    };
                                    if step == 1 {
                                        // A separate initial branch compares segmented streaming to
                                        // the full-input invocation; epochs remain increasing.
                                        committed =
                                            State::for_partition(&state_arch, None).unwrap();
                                        ordinary_state = PerceptionState::default();
                                    }
                                    if step == 2 && !resident && cut == 0 && tensor_ranks == 2 {
                                        barrier.wait();
                                        let before = world.lifecycle_counts();
                                        if rank == 0 {
                                            world.arm_fault_one_per_agreement_group(
                                                NumericPartitionFault::LocalExecution,
                                            );
                                        }
                                        barrier.wait();
                                        let mut branch =
                                            State::for_partition(&state_arch, Some(&committed))
                                                .unwrap();
                                        assert!(advance(
                                            &mut runtime,
                                            &mut engine,
                                            &mut branch,
                                            &input,
                                            drop,
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
                                        assert_eq!(before.3, after.3);
                                        compare_state(
                                            &committed,
                                            &ordinary_state,
                                            blocks.clone(),
                                            fixture,
                                        );
                                        barrier.wait();
                                    }
                                    let expected = ordinary
                                        .forward(&input, drop, &ordinary_state, &ordinary_context)
                                        .unwrap();
                                    if cut != 2 {
                                        let reference = if step == 0 {
                                            &fixture["perception"]
                                        } else {
                                            &fixture["perception"]["steps"][step - 1]
                                        };
                                        assert_tensor(
                                            &expected.encoder,
                                            &reference["encoder_output"],
                                            fixture,
                                        );
                                        assert_tensor(
                                            &expected.projected,
                                            &reference["projected_output"],
                                            fixture,
                                        );
                                        let cache_reference = if step == 0 {
                                            fixture["perception"]["steps"]
                                                .as_array()
                                                .unwrap()
                                                .last()
                                                .unwrap()
                                        } else {
                                            reference
                                        };
                                        assert_state(
                                            &expected.state,
                                            cache_reference,
                                            expected.state.frames(),
                                            fixture,
                                            &ordinary_context,
                                        );
                                    }
                                    let mut branch =
                                        State::for_partition(&state_arch, Some(&committed))
                                            .unwrap();
                                    let packed = advance(
                                        &mut runtime,
                                        &mut engine,
                                        &mut branch,
                                        &input,
                                        drop,
                                        step as u64 + 1,
                                        &context,
                                    )
                                    .unwrap();
                                    let (encoder, projected) = unpack_perception_pipeline_output(
                                        dimensions, &packed, &context,
                                    )
                                    .unwrap();
                                    assert_tensor(
                                        &encoder,
                                        &expected_value(&expected.encoder),
                                        fixture,
                                    );
                                    assert_tensor(
                                        &projected,
                                        &expected_value(&expected.projected),
                                        fixture,
                                    );
                                    compare_state(
                                        &branch,
                                        &expected.state,
                                        blocks.clone(),
                                        fixture,
                                    );
                                    committed = branch;
                                    ordinary_state = expected.state;
                                    world.record_commit(rank);
                                }
                            })
                        })
                        .collect::<Vec<_>>();
                    for worker in workers {
                        worker.join().unwrap();
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
}

fn boundary_contract(dimensions: PerceptionDimensions) -> RoleExactBoundaryContract {
    let schema = PerceptionBoundarySchema::new(dimensions)
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

pub(super) fn communication(
    dimensions: PerceptionDimensions,
    tensor_ranks: usize,
    rank: usize,
    source_group: usize,
    destination_group: usize,
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
        let limits = CommunicationTensorLimits::new(count, 3, 2048, None).unwrap();
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
            .with_boundary_contract(boundary_contract(dimensions))
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

fn advance(
    runtime: &mut Runtime,
    engine: &mut Engine,
    state: &mut State,
    input: &NumericTensor,
    drop_preencoded: usize,
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
        PerceptionInput {
            mel: input,
            drop_preencoded,
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

/// Shared numerical component pipeline reused by paired perception/RNNT coverage.
pub(super) struct PipelinePerception {
    runtime: Runtime,
    engine: Engine,
    architecture: Arch,
    context: NumericContext,
    pub(super) calls: usize,
    pub(super) reject_commit: bool,
}
impl PipelinePerception {
    pub(super) fn new(fixture: &Value, rank: usize, world: Arc<NumericPartitionWorld>) -> Self {
        let dimensions = dimensions(fixture);
        let description = perception_parameter_description(dimensions).unwrap();
        let complete = Arch::new(dimensions).unwrap();
        let owned = if rank == 0 {
            vec![
                (description.graph().groups()[0].id(), 0..1),
                (description.graph().groups()[1].id(), 0..dimensions.layers),
            ]
        } else {
            vec![(description.graph().groups()[2].id(), 0..1)]
        };
        let partition = ArchitecturePartition::from_description(
            &description,
            owned,
            PartitionOwnership::new(rank == 0, rank == 1, std::iter::empty::<String>()).unwrap(),
            &complete.state_layout().unwrap(),
            &ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
                1,
                0..dimensions.layers,
            )]),
            (),
            PerceptionBoundarySchema::new(dimensions),
        )
        .unwrap();
        let architecture = Arch::new_partition(dimensions, None, &partition).unwrap();
        let state_arch = Arch::new_partition(dimensions, None, &partition).unwrap();
        let layout = eredu_runtime::derive_partitioned_local_layout(
            &description,
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 1, 1).unwrap(), 0).unwrap(),
        )
        .unwrap();
        let context = NumericContext::with_partition(layout, rank, world.clone());
        let mut policy = Policy::bounded();
        policy.parameter_overrides = parameters(fixture);
        let (manifest, routes, session, _) = communication(dimensions, 1, rank, 1, 2);
        world.realize_manifest(&manifest).unwrap();
        let mut drivers = vec![None, None, None];
        for group in partition.groups() {
            drivers[group.group_index()] = Some(
                LayeredPartitionDriver::new_with_state_ownership(
                    &partition,
                    group.group_index(),
                    group.global_units(),
                    group.group_index() == 1,
                )
                .unwrap(),
            );
        }
        let plan = PartitionedExecutionPlan::new(
            description.graph().clone(),
            vec![
                (ArchitectureGroupKind::AudioEncoder, false),
                (ArchitectureGroupKind::AudioEncoder, false),
                (ArchitectureGroupKind::Projector, false),
            ],
            drivers,
            routes,
            Some(PartitionOutputPublication {
                group: session,
                owner_rank: 1,
            }),
            Some(session),
            PipelineWireContract::new(PipelineActivationDtype::Float32),
        )
        .unwrap();
        let executor = Executor::new(
            architecture,
            policy.clone(),
            partition.units().collect(),
            None,
            NumericPartitionTensorAllocator,
            PipelineActivationDtype::Float32,
        )
        .unwrap();
        let communication = numeric_partition_communication(&world, manifest).unwrap();
        let runtime = Runtime::new(
            plan,
            executor,
            communication,
            context.clone(),
            OpaqueBoundaryTransport,
            OpaqueOutputPublisher,
            OpaqueFailureAgreement,
            ExecutionResidency::LayerwiseHost,
            Some(policy),
        )
        .unwrap();
        Self {
            runtime,
            engine: Engine::new(),
            architecture: state_arch,
            context,
            calls: 0,
            reject_commit: false,
        }
    }
    pub(super) fn initial_state(&self) -> State {
        State::for_partition(&self.architecture, None).unwrap()
    }
}
impl
    eredu_architectures::voicechat::perception_rnnt_pipeline::PerceptionPipelineExecution<
        NumericBackend,
    > for PipelinePerception
{
    type State = State;
    fn dimensions(&self) -> PerceptionDimensions {
        self.architecture.dimensions()
    }
    fn validate_parent(
        &self,
        parent: &State,
        batch: i32,
    ) -> Result<(), eredu_architectures::voicechat::perception::PerceptionError> {
        self.architecture.validate_parent_for_batch(parent, batch)
    }
    fn forward(
        &mut self,
        input: PerceptionInput<'_, NumericTensor>,
        parent: &State,
        _descriptor: [u64; 8],
        epoch: DistributedCommitEpoch,
        _context: &NumericContext,
    ) -> Result<
        (
            NumericTensor,
            NumericTensor,
            State,
            DistributedCommitOutcome,
        ),
        eredu_architectures::voicechat::perception::PerceptionError,
    > {
        self.calls += 1;
        let mut state = State::for_partition(&self.architecture, Some(parent))?;
        let packed = advance(
            &mut self.runtime,
            &mut self.engine,
            &mut state,
            input.mel,
            input.drop_preencoded,
            epoch.value(),
            &self.context,
        )
        .map_err(|error| {
            eredu_architectures::voicechat::perception::PerceptionError::Backend(
                eredu_core::BackendFailure::from_error(std::io::Error::other(error)),
            )
        })?;
        let (raw, projected) = unpack_perception_pipeline_output(
            self.architecture.dimensions(),
            &packed,
            &self.context,
        )?;
        let outcome = if self.reject_commit {
            DistributedCommitOutcome::Aborted(epoch)
        } else {
            DistributedCommitOutcome::Committed(epoch)
        };
        Ok((raw, projected, state, outcome))
    }
}
