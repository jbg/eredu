//! Streaming EAR decoder partitions preserve local carry, publication and replay.
use super::voicechat_codec_encoder_distributed::communication;
use super::voicechat_mog_residency::weights;
use super::voicechat_tts_generator::{compare, tensor};
use super::*;
use eredu_codec::{
    ear_decoder::{DecoderConfig, DecoderUnit},
    ear_decoder_layered::{
        decoder_parameter_description, DecoderArchitecture, DecoderRuntimeState,
    },
    ear_decoder_partition::{DecoderBoundarySchema, DecoderPipelineCuts},
    ear_spectral::{EarSpectralDecoder, SpectralConfig},
    streaming::StreamPhase,
};
use eredu_core::DistributedCommitEpoch;
use eredu_runtime::*;
use serde_json::Value;

const PREFIX: &str = "tts_model.audio_codec.decoder";
type Arch = DecoderArchitecture<NumericBackend>;
type State = DecoderRuntimeState<NumericTensor>;
type Unit = DecoderUnit<NumericTensor>;

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
fn config(value: &Value) -> DecoderConfig {
    let i = |name: &str| value[name].as_i64().unwrap() as i32;
    DecoderConfig {
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
    config: &DecoderConfig,
    cut: usize,
    stage: usize,
) -> ArchitecturePartition<(), DecoderBoundarySchema> {
    let complete = Arch::new(config.clone(), PREFIX).unwrap();
    let description = decoder_parameter_description(config, PREFIX).unwrap();
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
        DecoderBoundarySchema::new(config).unwrap(),
    )
    .unwrap()
}

fn compare_carry(state: &State, cache: &Value, fixture: &Value) {
    assert_eq!(state.phase(), StreamPhase::AcceptingInput);
    for (global, unit) in state.global_units().zip(state.units()) {
        let index = match global {
            1 => Some(0),
            2 => Some(1),
            4 => Some(2),
            5 => Some(3),
            _ => None,
        };
        match (index, unit.carry()) {
            (Some(index), Some(carry)) => compare(carry, &cache[index.to_string()], fixture),
            (None, None) => {}
            _ => panic!("owned decoder unit {global} has wrong convolution carry presence"),
        }
    }
}
fn same_carry(actual: &State, expected: &State) {
    assert_eq!(actual.global_units(), expected.global_units());
    assert_eq!(actual.phase(), expected.phase());
    for (actual, expected) in actual.units().iter().zip(expected.units()) {
        match (actual.carry(), expected.carry()) {
            (Some(actual), Some(expected)) => {
                assert_eq!(actual.shape, expected.shape);
                assert_eq!(actual.data, expected.data);
            }
            (None, None) => {}
            _ => panic!("candidate changed saved parent carry presence"),
        }
    }
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
    let case = &fixture["decoders"]["compact"];
    let config = config(&case["geometry"]);
    let stage = rank / tensor_ranks;
    let description = decoder_parameter_description(&config, PREFIX).unwrap();
    let topology = ParallelRankTopology::new(
        ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
        rank % tensor_ranks,
    )
    .unwrap();
    let layout = derive_partitioned_local_layout(&description, topology).unwrap();
    let context = NumericContext::with_partition(layout.clone(), rank, world.clone());
    let partition = partition(&config, cut, stage);
    let schema = DecoderBoundarySchema::new(&config)
        .unwrap()
        .at_cut(cut)
        .unwrap()
        .wire_schema()
        .unwrap();
    let (manifest, routes, session, tensor_group) = communication(&schema, tensor_ranks, rank);
    let cuts =
        DecoderPipelineCuts::new(&config, routes.iter().map(|route| (route.route, cut))).unwrap();
    let build = || {
        Arch::new_partition(
            config.clone(),
            PREFIX,
            (tensor_ranks > 1).then_some(&layout),
            &partition,
            &cuts,
        )
        .unwrap()
    };
    let architecture = build();
    let state_arch = build();
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
    let driver =
        LayeredPartitionDriver::new(&partition, 0, partition.groups()[0].global_units()).unwrap();
    let plan = PartitionedExecutionPlan::new(
        description.graph().clone(),
        vec![(architecture.group_transport(0).kind, false)],
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
    let mut committed = State::for_partition(&state_arch, None).unwrap();
    let mut branch = State::for_partition(&state_arch, Some(&committed)).unwrap();
    let invalid = NumericTensor::zeros([1, 1, config.latent_size + 1]);
    assert!(advance(
        &mut runtime,
        &mut engine,
        &mut branch,
        &invalid,
        1,
        &context
    )
    .is_err());
    assert_eq!(acquisitions.load(Ordering::Relaxed), 0);
    same_carry(&branch, &committed);
    let spectral = EarSpectralDecoder::<NumericBackend>::new(
        SpectralConfig {
            fft_size: config.fft_size,
            hop_length: case["geometry"]["hop_length"].as_i64().unwrap() as i32,
        },
        &context,
    )
    .unwrap();
    let mut spectral_state = spectral.initial_state(1).unwrap();
    let mut epoch = 1;
    for (step_index, step) in case["streaming"].as_array().unwrap().iter().enumerate() {
        let input = tensor(&step["input"]);
        let parent = State::for_partition(&state_arch, Some(&committed)).unwrap();
        let spectral_parent = spectral_state.clone();
        if step_index == 1 && !resident && cut == 4 && tensor_ranks == 2 {
            barrier.wait();
            let before = world.lifecycle_counts();
            if rank == 0 {
                world.arm_fault_one_per_agreement_group(NumericPartitionFault::LocalExecution);
            }
            barrier.wait();
            let mut failed = State::for_partition(&state_arch, Some(&parent)).unwrap();
            assert!(advance(
                &mut runtime,
                &mut engine,
                &mut failed,
                &input,
                epoch,
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
            compare_carry(&parent, &case["streaming"][0]["cache"], fixture);
            same_carry(&parent, &committed);
            barrier.wait();
        }
        let mut candidate = State::for_partition(&state_arch, Some(&parent)).unwrap();
        let raw = advance(
            &mut runtime,
            &mut engine,
            &mut candidate,
            &input,
            epoch,
            &context,
        )
        .unwrap();
        epoch += 1;
        world.record_commit(rank);
        compare(&raw, &step["raw"], fixture);
        same_carry(&parent, &committed);
        let flush = step["flush"].as_bool().unwrap();
        let (pcm, next_spectral) = spectral
            .step(&raw, &spectral_parent, flush, true, &context)
            .unwrap();
        compare(&pcm, &step["waveform"], fixture);
        if !flush {
            compare_carry(&candidate, &step["cache"], fixture);
            let (real, imaginary) = next_spectral.history().unwrap();
            compare(real, &step["cache"]["4"]["real"], fixture);
            compare(imaginary, &step["cache"]["4"]["imag"], fixture);
        }
        if step_index == 1 {
            let mut fork = State::for_partition(&state_arch, Some(&parent)).unwrap();
            let replay = advance(
                &mut runtime,
                &mut engine,
                &mut fork,
                &input,
                epoch,
                &context,
            )
            .unwrap();
            epoch += 1;
            world.record_commit(rank);
            compare(&replay, &step["raw"], fixture);
            same_carry(&fork, &candidate);
            same_carry(&parent, &committed);
            let (replayed_pcm, _) = spectral
                .step(&replay, &spectral_parent, false, true, &context)
                .unwrap();
            compare(&replayed_pcm, &step["waveform"], fixture);
        }
        committed = candidate;
        spectral_state = next_spectral;
    }
    assert!(spectral_state.is_finished());
    assert!(spectral_state.history().is_none());
    let before = acquisitions.load(Ordering::Relaxed);
    let mut finished = committed.finish_partition(&state_arch).unwrap();
    assert_eq!(finished.phase(), StreamPhase::Finished);
    assert!(finished.units().iter().all(|unit| unit.carry().is_none()));
    assert!(advance(
        &mut runtime,
        &mut engine,
        &mut finished,
        &tensor(&case["streaming"][0]["input"]),
        epoch,
        &context
    )
    .is_err());
    assert_eq!(
        acquisitions.load(Ordering::Relaxed),
        before,
        "drain and finished rejection cannot acquire learned units"
    );
}

#[test]
fn voicechat_codec_decoder_distributed_pp2_and_tp2_pp2_preserve_carry_replay_and_flush() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
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
                    tensor_ranks * 2 * 5
                );
            }
        }
    }
}

#[test]
fn voicechat_codec_decoder_partition_rejects_malformed_boundary_and_foreign_carry_before_acquisition(
) {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let case = &fixture["decoders"]["compact"];
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
    let cuts = DecoderPipelineCuts::new(&config, [(route.route, 3)]).unwrap();
    let build = || Arch::new_partition(config.clone(), PREFIX, None, &selected, &cuts).unwrap();
    let state_arch = build();
    let complete_identity = Arch::new(config.clone(), PREFIX)
        .unwrap()
        .state_identity(selected.state().unwrap(), PromptCacheTopology::default())
        .unwrap();
    assert_eq!(
        complete_identity,
        state_arch
            .state_identity(selected.state().unwrap(), PromptCacheTopology::default())
            .unwrap(),
        "cold complete construction and selected execution describe the same stage"
    );
    let architecture = build();
    let mut inner = NumericReplicatedPolicy::bounded();
    inner.parameter_overrides = weights(&case["parameters"]);
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let policy = Policy {
        inner,
        acquisitions: acquisitions.clone(),
    };
    let driver =
        LayeredPartitionDriver::new(&selected, 0, selected.groups()[0].global_units()).unwrap();
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
    let input = tensor(&case["streaming"][0]["input"]);
    // Build valid upstream activation once; the suffix's result is still
    // compared with NVIDIA's independently recorded complete decoder output.
    let complete = Arch::new(config.clone(), PREFIX).unwrap();
    let mut hidden = input.swap_axes(1, 2, &context).unwrap();
    for index in 0..3 {
        let mut unit = complete.build_unit(0, index, &context).unwrap();
        replace_numeric_parameters(&mut unit, &weights(&case["parameters"]));
        hidden = unit
            .step(&hidden, &unit.initial_state().unwrap(), &context)
            .unwrap()
            .0;
    }
    let header = |words: &[i32]| {
        NumericTensor::from_i32_slice(words, &[words.len() as i32], &context).unwrap()
    };
    let mut state = State::for_partition(&state_arch, None).unwrap();
    for (incoming, metadata, selected_route) in [
        (hidden.clone(), header(&[3, 1]), route.clone()),
        (hidden.clone(), header(&[4, 1, 1]), route.clone()),
        (hidden.clone(), header(&[3, 2, 1]), route.clone()),
        (hidden.clone(), header(&[3, 1, 2]), route.clone()),
        (
            NumericTensor::zeros([1, 3, 3]),
            header(&[3, 1, 1]),
            route.clone(),
        ),
        (
            hidden.clone(),
            NumericTensor::new([3], vec![3., 1., 1.]),
            route.clone(),
        ),
        (
            hidden.clone(),
            header(&[3, 1, 1]),
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
            &context,
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
        assert_eq!(acquisitions.load(Ordering::Relaxed), 0);
        assert!(state.units().iter().all(|unit| unit.carry().is_none()));
    }
    let other_partition = partition(&config, 3, 0);
    assert!(state_arch
        .state_identity(
            other_partition.state().unwrap(),
            PromptCacheTopology::default()
        )
        .is_err());
    let other = Arch::new_partition(config, PREFIX, None, &other_partition, &cuts).unwrap();
    let mut foreign = State::for_partition(&other, None).unwrap();
    assert!(State::for_partition(&state_arch, Some(&foreign)).is_err());
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
        &mut foreign,
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
        vec![hidden.clone(), header(&[3, 1, 1])],
        &context,
    )
    .unwrap();
    assert!(executor
        .execute_group(
            &mut pass,
            &driver,
            &mut foreign,
            &communication,
            &context,
            &context,
            &mut NoopObserver
        )
        .is_err());
    assert_eq!(
        acquisitions.load(Ordering::Relaxed),
        0,
        "a different stage cannot acquire units under another stage's carry owner"
    );
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
        vec![hidden, header(&[3, 1, 1])],
        &context,
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
    compare(&output, &case["streaming"][0]["raw"], &fixture);
    compare_carry(&state, &case["streaming"][0]["cache"], &fixture);
    assert_eq!(acquisitions.load(Ordering::Relaxed), 4);
}
