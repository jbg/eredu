//! Finite EAR encoder partitions share pipeline transport, residency and output commitment.
use super::voicechat_mog_residency::weights;
use super::voicechat_tts_generator::{compare, tensor};
use super::*;
use eredu_codec::{
    ear_encoder::{EncoderConfig, EncoderUnit},
    ear_encoder_layered::{
        encoder_parameter_description, EncoderArchitecture, EncoderRuntimeState,
    },
    ear_encoder_partition::{EncoderBoundarySchema, EncoderPipelineCuts},
};
use eredu_core::{checkpoint::TensorDtype, DistributedCommitEpoch};
use eredu_runtime::*;
use serde_json::Value;

const PREFIX: &str = "tts_model.audio_codec.encoder";
type Arch = EncoderArchitecture<NumericBackend>;
type State = EncoderRuntimeState;
type Unit = EncoderUnit<NumericTensor>;

#[derive(Clone)]
struct Policy {
    inner: NumericReplicatedPolicy<Unit>,
    acquisitions: Arc<AtomicUsize>,
}
impl LayerwisePolicy<NumericBackend, Unit> for Policy {
    type Lease = NumericReplicatedLease<Unit>;
    type Error = Error;
    fn begin(&mut self, input: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        self.inner.begin(input, context)
    }
    fn acquire<E, F>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        build: F,
        context: &NumericContext,
    ) -> Result<Self::Lease, LayerwiseAcquireError<E, Error>>
    where
        F: FnOnce(&NumericContext) -> Result<Unit, E>,
    {
        self.acquisitions.fetch_add(1, Ordering::Relaxed);
        self.inner.acquire(ordinal, address, build, context)
    }
    fn complete<'a, S, C>(
        &mut self,
        ordinal: usize,
        address: ExecutionUnitAddress,
        lease: Self::Lease,
        output: &'a NumericTensor,
        state: S,
        retained: C,
        context: &NumericContext,
    ) -> Result<(), Error>
    where
        NumericTensor: 'a,
        S: Iterator<Item = &'a NumericTensor>,
        C: Iterator<Item = &'a NumericTensor>,
    {
        self.inner
            .complete(ordinal, address, lease, output, state, retained, context)
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        self.inner.finish(output, context)
    }
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &NumericContext,
    ) {
        self.inner.abort(active, context)
    }
}
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
fn config(value: &Value) -> EncoderConfig {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    EncoderConfig {
        latent_size: i("latent_size"),
        base_hidden_size: i("base_hidden_size"),
        channel_mult: serde_json::from_value(value["channel_mult"].clone()).unwrap(),
        rates: serde_json::from_value(value["rates"].clone()).unwrap(),
        blocks_per_stage: i("num_blocks") as usize,
        kernel_size: i("kernel_size"),
        fft_size: i("n_fft"),
        groups: i("groups"),
    }
}

fn boundary_contract(schema: &BoundaryWireSchema) -> RoleExactBoundaryContract {
    let resolved = schema.resolve(2, 24).unwrap();
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
    schema: &BoundaryWireSchema,
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
            .with_boundary_contract(boundary_contract(schema))
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
        input,
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

fn partition(
    config: &EncoderConfig,
    cut: usize,
    stage: usize,
) -> ArchitecturePartition<(), EncoderBoundarySchema> {
    let complete = Arch::new(config.clone(), PREFIX).unwrap();
    let description = encoder_parameter_description(config, PREFIX).unwrap();
    let count = config.unit_specs().unwrap().len();
    let range = if stage == 0 { 0..cut } else { cut..count };
    let state = complete.state_layout().unwrap();
    ArchitecturePartition::from_description(
        &description,
        [(description.graph().groups()[0].id(), range)],
        PartitionOwnership::new(stage == 0, stage == 1, std::iter::empty::<String>()).unwrap(),
        &state,
        &complete.state_partition_plan(&state),
        (),
        EncoderBoundarySchema::new(config).unwrap(),
    )
    .unwrap()
}

fn run_rank(
    fixture: &Value,
    cut: usize,
    tensor_ranks: usize,
    rank: usize,
    resident: bool,
    world: Arc<NumericPartitionWorld>,
    barrier: Arc<std::sync::Barrier>,
) {
    let case = &fixture["encoders"]["compact"];
    let config = config(&case["geometry"]);
    let stage = rank / tensor_ranks;
    let description = encoder_parameter_description(&config, PREFIX).unwrap();
    let topology = ParallelRankTopology::new(
        ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
        rank % tensor_ranks,
    )
    .unwrap();
    let layout = derive_partitioned_local_layout(&description, topology).unwrap();
    let context = NumericContext::with_partition(layout.clone(), rank, world.clone());
    let partition = partition(&config, cut, stage);
    let schema = EncoderBoundarySchema::new(&config)
        .unwrap()
        .at_cut(cut)
        .unwrap()
        .wire_schema()
        .unwrap();
    let (manifest, routes, session, tensor_group) = communication(&schema, tensor_ranks, rank);
    let cuts =
        EncoderPipelineCuts::new(&config, routes.iter().map(|route| (route.route, cut))).unwrap();
    let architecture = Arch::new_partition(
        config.clone(),
        PREFIX,
        (tensor_ranks > 1).then_some(&layout),
        &partition,
        &cuts,
    )
    .unwrap();
    let selected: BTreeMap<_, _> = weights(&case["parameters"])
        .into_iter()
        .filter_map(|(name, value)| {
            let tensor = layout.tensor(&name)?;
            Some((name, select_parameter(&value, tensor.placement()).unwrap()))
        })
        .collect();
    let inner = if resident {
        let units = partition
            .units()
            .map(|address| {
                let mut unit = architecture
                    .build_unit(address.group(), address.index(), &context)
                    .unwrap();
                replace_numeric_parameters(&mut unit, &selected);
                unit
            })
            .collect();
        NumericReplicatedPolicy::resident(units)
    } else {
        let mut policy = NumericReplicatedPolicy::bounded();
        policy.parameter_overrides = selected;
        policy
    };
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let policy = Policy {
        inner,
        acquisitions: acquisitions.clone(),
    };
    let bounded_policy = (!resident).then(|| policy.clone());
    world.realize_manifest(&manifest).unwrap();
    let parallel = tensor_group.map(|id| numeric_parallel_context(&world, &manifest, id).unwrap());
    let driver = LayeredPartitionDriver::new_with_state_ownership(
        &partition,
        0,
        partition.groups()[0].global_units(),
        false,
    )
    .unwrap();
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
    let executor = Executor::new(
        architecture,
        policy,
        partition.units().collect(),
        parallel,
        NumericPartitionTensorAllocator,
        PipelineActivationDtype::Float32,
    )
    .unwrap();
    let communication = numeric_partition_communication(&world, manifest).unwrap();
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
    let mut state = State::new().unwrap();
    let packed = tensor(&case["packed"]);
    let invalid = NumericTensor::zeros([2, config.fft_size + 1, packed.dim(2)]);
    assert!(advance(&mut runtime, &mut engine, &mut state, &invalid, 1, &context).is_err());
    assert_eq!(
        acquisitions.load(Ordering::Relaxed),
        0,
        "invalid spectral geometry acquires no local parameters"
    );
    for (step, input) in [packed.clone(), packed.axis_slice(0, 1, 2), packed.clone()]
        .into_iter()
        .enumerate()
    {
        if step == 1 && !resident && cut == 4 && tensor_ranks == 2 {
            barrier.wait();
            let before = world.lifecycle_counts();
            if rank == 0 {
                world.arm_fault_one_per_agreement_group(NumericPartitionFault::LocalExecution);
            }
            barrier.wait();
            assert!(advance(
                &mut runtime,
                &mut engine,
                &mut state,
                &input,
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
                "failed finite work cannot commit or publish"
            );
            barrier.wait();
        }
        let output = advance(
            &mut runtime,
            &mut engine,
            &mut state,
            &input,
            step as u64 + 1,
            &context,
        )
        .unwrap();
        let expected = if step == 1 {
            tensor(&case["latent"]).axis_slice(0, 1, 2)
        } else {
            tensor(&case["latent"])
        };
        compare(
            &output,
            &serde_json::json!({"shape": expected.shape, "values": expected.data}),
            fixture,
        );
        assert_eq!(
            input.data,
            if step == 1 {
                packed.axis_slice(0, 1, 2).data
            } else {
                packed.data.clone()
            }
        );
        world.record_commit(rank);
    }
}

#[test]
fn voicechat_codec_encoder_distributed_pp2_and_tp2_pp2_match_pinned_latents_and_retry() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    // The first downsampler changes both channel count and cadence; cuts on
    // either side use its independently recorded NVIDIA stage products.
    for cut in [3, 4] {
        for tensor_ranks in [1, 2] {
            for resident in [true, false] {
                let world = Arc::new(NumericPartitionWorld::default());
                let barrier = Arc::new(std::sync::Barrier::new(tensor_ranks * 2));
                std::thread::scope(|scope| {
                    let workers = (0..tensor_ranks * 2)
                        .map(|rank| {
                            let world = world.clone();
                            let barrier = barrier.clone();
                            let fixture = &fixture;
                            scope.spawn(move || {
                                run_rank(fixture, cut, tensor_ranks, rank, resident, world, barrier)
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
                assert_eq!(
                    world.lifecycle_counts().3.values().sum::<usize>(),
                    tensor_ranks * 2 * 3
                );
            }
        }
    }
}

#[test]
fn voicechat_codec_encoder_partition_rejects_malformed_route_cut_and_frame_metadata_before_acquisition(
) {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let case = &fixture["encoders"]["compact"];
    let config = config(&case["geometry"]);
    let context = NumericContext::default();
    let selected = partition(&config, 3, 1);
    let route = PartitionBoundaryRoute {
        source_group: 0,
        destination_group: 0,
        source_rank: 0,
        destination_rank: 1,
        route: CommunicationRouteId::new(100),
    };
    let cuts = EncoderPipelineCuts::new(&config, [(route.route, 3)]).unwrap();
    let architecture = Arch::new_partition(config, PREFIX, None, &selected, &cuts).unwrap();
    let mut inner = NumericReplicatedPolicy::bounded();
    inner.parameter_overrides = weights(&case["parameters"]);
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let policy = Policy {
        inner,
        acquisitions: acquisitions.clone(),
    };
    let driver = LayeredPartitionDriver::new_with_state_ownership(
        &selected,
        0,
        selected.groups()[0].global_units(),
        false,
    )
    .unwrap();
    let mut executor = Executor::new(
        architecture,
        policy,
        selected.units().collect(),
        None,
        NumericPartitionTensorAllocator,
        PipelineActivationDtype::Float32,
    )
    .unwrap();
    let communication = PartitionCommunication::new(
        CommunicationManifest::new(1, 0, vec![], vec![]).unwrap(),
        Vec::<RealizedCommunicationGroup<u64>>::new(),
        Vec::<RealizedCommunicationRoute<u64>>::new(),
        NumericCommunicationMetadata(None),
    )
    .unwrap();
    let input = tensor(&case["packed"]);
    let hidden = tensor(&case["stages"][2]);
    let header = |words: &[i32]| {
        NumericTensor::from_i32_slice(words, &[words.len() as i32], &context).unwrap()
    };
    let mut state = State::new().unwrap();
    for (incoming, metadata, selected_route) in [
        (hidden.clone(), header(&[3, 2]), route.clone()),
        (hidden.clone(), header(&[4, 2, 24]), route.clone()),
        (hidden.clone(), header(&[3, 1, 24]), route.clone()),
        (hidden.clone(), header(&[3, 2, 12]), route.clone()),
        (
            NumericTensor::zeros([2, 3, 24]),
            header(&[3, 2, 24]),
            route.clone(),
        ),
        (
            hidden.clone(),
            NumericTensor::new([3], vec![3., 2., 24.]),
            route.clone(),
        ),
        (
            hidden.clone(),
            header(&[3, 2, 24]),
            PartitionBoundaryRoute {
                route: CommunicationRouteId::new(999),
                ..route.clone()
            },
        ),
    ] {
        let mut pass = <Executor as PartitionedGroupExecutor<
            Arch,
            NumericBackend,
            State,
            u64,
            u64,
            NumericCommunicationMetadata,
        >>::begin(
            &mut executor,
            &input,
            &mut state,
            ExpertPass::Prefill,
            &context,
        )
        .unwrap();
        let accepted = <Executor as PartitionedGroupExecutor<
            Arch,
            NumericBackend,
            State,
            u64,
            u64,
            NumericCommunicationMetadata,
        >>::accept_boundary(
            &mut executor,
            &mut pass,
            &selected_route,
            vec![incoming, metadata],
        );
        if accepted.is_ok() {
            assert!(executor
                .execute_group(
                    &mut pass,
                    &driver,
                    &mut state,
                    &communication,
                    &context,
                    &context,
                    &mut NoopObserver
                )
                .is_err());
        }
        assert_eq!(
            acquisitions.load(Ordering::Relaxed),
            0,
            "invalid boundary must not lease even the first local block"
        );
    }
    let mut pass = <Executor as PartitionedGroupExecutor<
        Arch,
        NumericBackend,
        State,
        u64,
        u64,
        NumericCommunicationMetadata,
    >>::begin(
        &mut executor,
        &input,
        &mut state,
        ExpertPass::Prefill,
        &context,
    )
    .unwrap();
    <Executor as PartitionedGroupExecutor<
        Arch,
        NumericBackend,
        State,
        u64,
        u64,
        NumericCommunicationMetadata,
    >>::accept_boundary(
        &mut executor,
        &mut pass,
        &route,
        vec![hidden, header(&[3, 2, 24])],
    )
    .unwrap();
    executor
        .execute_group(
            &mut pass,
            &driver,
            &mut state,
            &communication,
            &context,
            &context,
            &mut NoopObserver,
        )
        .unwrap();
    let (output, _) = <Executor as PartitionedGroupExecutor<
        Arch,
        NumericBackend,
        State,
        u64,
        u64,
        NumericCommunicationMetadata,
    >>::finish(&mut executor, pass, &mut state, &context)
    .unwrap();
    compare(&output, &case["latent"], &fixture);
    assert_eq!(
        acquisitions.load(Ordering::Relaxed),
        4,
        "retry leases exactly the selected four remaining units"
    );
}
