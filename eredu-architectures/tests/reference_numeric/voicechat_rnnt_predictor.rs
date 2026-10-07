//! Recurrent pipeline phases preserve selected carry and explicit request identity.
use super::voicechat_codec_encoder_distributed::communication;
use super::voicechat_mog_residency::weights;
use super::voicechat_rnnt::{assert_values, dimensions, fixture};
use super::*;
use eredu_architectures::voicechat::{
    rnnt::{RnntDimensions, RnntPredictionState},
    rnnt_predictor::{
        rnnt_predictor_parameter_description, RnntPredictorArchitecture, RnntPredictorInput,
        RnntPredictorRuntimeState, RnntPredictorUnit,
    },
    rnnt_predictor_partition::{RnntPredictorBoundarySchema, RnntPredictorPipelineCuts},
};
use eredu_core::DistributedCommitEpoch;
use eredu_nn::recurrent::{LstmLayerState, LstmState};
use eredu_runtime::*;
use serde_json::Value;
type Arch = RnntPredictorArchitecture<NumericBackend>;
type State = RnntPredictorRuntimeState<NumericTensor>;
type Unit = RnntPredictorUnit<NumericBackend>;
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
fn advance(
    runtime: &mut Runtime,
    engine: &mut Engine,
    state: &mut State,
    input: &NumericTensor,
    step: u64,
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
        RnntPredictorInput { input, step },
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
    dim: RnntDimensions,
    stage: usize,
) -> ArchitecturePartition<(), RnntPredictorBoundarySchema> {
    let arch = Arch::new(dim).unwrap();
    let description = rnnt_predictor_parameter_description(dim).unwrap();
    let layout = arch.state_layout().unwrap();
    ArchitecturePartition::from_description(
        &description,
        [(description.graph().groups()[0].id(), stage..stage + 1)],
        PartitionOwnership::new(stage == 0, stage == 1, std::iter::empty::<String>()).unwrap(),
        &layout,
        &arch.state_partition_plan(&layout),
        (),
        RnntPredictorBoundarySchema::new(dim).unwrap(),
    )
    .unwrap()
}
pub(super) fn compare_state(
    state: &RnntPredictionState<NumericTensor>,
    stage: usize,
    expected: &Value,
    fixture: &Value,
) {
    assert_eq!(state.global_layers(), stage..stage + 1);
    let width = state.dimensions().predictor_width as usize;
    let slice = |name: &str, index: usize| {
        Value::Array(
            expected[name].as_array().unwrap()[index * width..(index + 1) * width].to_vec(),
        )
    };
    assert_values(
        &state.local_state().layers[0].hidden.data,
        &slice("h", stage),
        fixture,
    );
    assert_values(
        &state.local_state().layers[0].cell.data,
        &slice("c", stage),
        fixture,
    );
    assert_values(&state.output().data, &slice("h", 1), fixture);
}
fn same_state(a: &RnntPredictionState<NumericTensor>, b: &RnntPredictionState<NumericTensor>) {
    assert_eq!(a.global_layers(), b.global_layers());
    assert_eq!(a.output().data, b.output().data);
    for (a, b) in a.local_state().layers.iter().zip(&b.local_state().layers) {
        assert_eq!(a.hidden.data, b.hidden.data);
        assert_eq!(a.cell.data, b.cell.data);
    }
}
pub(super) struct PipelinePredictor {
    runtime: Runtime,
    engine: Engine,
    architecture: Arch,
    context: NumericContext,
    pub(super) acquisitions: Arc<AtomicUsize>,
    pub(super) calls: Vec<(u64, u64)>,
    pub(super) reject_commit: bool,
}
impl PipelinePredictor {
    pub(super) fn new(
        fixture: &Value,
        tensor_ranks: usize,
        rank: usize,
        world: Arc<NumericPartitionWorld>,
    ) -> Self {
        let dim = dimensions(fixture);
        let stage = rank / tensor_ranks;
        let description = rnnt_predictor_parameter_description(dim).unwrap();
        let layout = derive_partitioned_local_layout(
            &description,
            ParallelRankTopology::new(
                ParallelTopology::new(tensor_ranks, 1, 1, 1).unwrap(),
                rank % tensor_ranks,
            )
            .unwrap(),
        )
        .unwrap();
        let context = NumericContext::with_partition(layout, rank, world.clone());
        let partition = partition(dim, stage);
        let schema = RnntPredictorBoundarySchema::new(dim)
            .unwrap()
            .at_cut(1)
            .unwrap()
            .wire_schema()
            .unwrap();
        let (manifest, routes, session, tensor_group) = communication(&schema, tensor_ranks, rank);
        let cuts =
            RnntPredictorPipelineCuts::new(dim, routes.iter().map(|r| (r.route, 1))).unwrap();
        let build = || Arch::new_partition(dim, &partition, &cuts).unwrap();
        let architecture = build();
        let state_arch = build();
        let mut inner = NumericReplicatedPolicy::bounded();
        inner.parameter_overrides = weights(&fixture["parameters"]);
        let acquisitions = Arc::new(AtomicUsize::new(0));
        let policy = Policy {
            inner,
            acquisitions: acquisitions.clone(),
        };
        world.realize_manifest(&manifest).unwrap();
        let parallel =
            tensor_group.map(|id| numeric_parallel_context(&world, &manifest, id).unwrap());
        let driver =
            LayeredPartitionDriver::new(&partition, 0, partition.groups()[0].global_units())
                .unwrap();
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
            policy.clone(),
            partition.units().collect(),
            parallel,
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
            acquisitions,
            calls: vec![],
            reject_commit: false,
        }
    }
}
impl eredu_architectures::voicechat::rnnt_pipeline::RnntPredictorExecution<NumericBackend>
    for PipelinePredictor
{
    fn dimensions(&self) -> RnntDimensions {
        self.architecture.dimensions()
    }
    fn predictor_ownership(&self) -> std::ops::Range<usize> {
        self.architecture.predictor_ownership()
    }
    fn predictor_step(
        &mut self,
        input: &NumericTensor,
        parent: Option<&RnntPredictionState<NumericTensor>>,
        step: u64,
        _descriptor: [u64; 7],
        epoch: DistributedCommitEpoch,
        _context: &NumericContext,
    ) -> Result<
        (
            RnntPredictionState<NumericTensor>,
            eredu_core::DistributedCommitOutcome,
        ),
        eredu_architectures::voicechat::rnnt::RnntError,
    > {
        self.calls.push((step, epoch.value()));
        let mut state = State::for_partition(&self.architecture, parent)?;
        let output = advance(
            &mut self.runtime,
            &mut self.engine,
            &mut state,
            input,
            step,
            epoch.value(),
            &self.context,
        )
        .map_err(|error| {
            eredu_architectures::voicechat::rnnt::RnntError::Backend(
                eredu_core::BackendFailure::from_error(std::io::Error::other(error)),
            )
        })?;
        let outcome = if self.reject_commit {
            eredu_core::DistributedCommitOutcome::Aborted(epoch)
        } else {
            eredu_core::DistributedCommitOutcome::Committed(epoch)
        };
        Ok((state.into_prediction(output)?, outcome))
    }
}

fn run_rank(
    fixture: &Value,
    tensor_ranks: usize,
    rank: usize,
    world: Arc<NumericPartitionWorld>,
    barrier: Arc<std::sync::Barrier>,
) {
    let dim = dimensions(fixture);
    let stage = rank / tensor_ranks;
    let PipelinePredictor {
        mut runtime,
        mut engine,
        architecture: state_arch,
        context,
        acquisitions,
        ..
    } = PipelinePredictor::new(fixture, tensor_ranks, rank, world.clone());
    let mut committed: Option<RnntPredictionState<NumericTensor>> = None;
    let mut epoch = 1;
    let base_step = 0xfedc_ba98_8765_4321u64;
    let steps = fixture["predictor_steps"].as_array().unwrap();
    // The released prime executes two zero-input recurrent passes.
    for call in 0..steps.len() + 1 {
        let input = if call < 2 {
            NumericTensor::zeros([1, dim.predictor_width])
        } else {
            let token = steps[call - 1]["token"].as_u64().unwrap() as usize;
            let embedding =
                &fixture["parameters"]["stt_model.rnnt_decoder.prediction.embed.weight"]["values"];
            NumericTensor::new(
                [1, dim.predictor_width],
                embedding.as_array().unwrap()[token * 2..token * 2 + 2]
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect(),
            )
        };
        let parent = committed.clone();
        if call == 2 {
            barrier.wait();
            if rank == 0 {
                world.arm_fault_one_per_agreement_group(NumericPartitionFault::LocalExecution);
            }
            barrier.wait();
            let mut failed = State::for_partition(&state_arch, parent.as_ref()).unwrap();
            assert!(advance(
                &mut runtime,
                &mut engine,
                &mut failed,
                &input,
                base_step + call as u64,
                epoch,
                &context
            )
            .is_err());
            barrier.wait();
            same_state(parent.as_ref().unwrap(), committed.as_ref().unwrap());
        }
        let mut candidate = State::for_partition(&state_arch, parent.as_ref()).unwrap();
        let output = advance(
            &mut runtime,
            &mut engine,
            &mut candidate,
            &input,
            base_step + call as u64,
            epoch,
            &context,
        )
        .unwrap();
        epoch += 1;
        let candidate = candidate.into_prediction(output).unwrap();
        if call == 0 {
            assert_values(
                &candidate.output().data,
                &Value::Array(steps[0]["output"].as_array().unwrap()[..2].to_vec()),
                fixture,
            );
        } else {
            compare_state(&candidate, stage, &steps[call - 1], fixture);
        }
        if call == 2 {
            let mut replay = State::for_partition(&state_arch, parent.as_ref()).unwrap();
            let replay_output = advance(
                &mut runtime,
                &mut engine,
                &mut replay,
                &input,
                base_step + call as u64,
                epoch,
                &context,
            )
            .unwrap();
            epoch += 1;
            same_state(&candidate, &replay.into_prediction(replay_output).unwrap());
            same_state(parent.as_ref().unwrap(), committed.as_ref().unwrap());
        }
        committed = Some(candidate);
    }
    assert!(acquisitions.load(Ordering::Relaxed) > 0);
}
#[test]
fn voicechat_rnnt_predictor_pp_and_tp_pp_match_pinned_carry_and_replay() {
    let fixture = fixture();
    for tensor_ranks in [1, 2] {
        let world = Arc::new(NumericPartitionWorld::default());
        let barrier = Arc::new(std::sync::Barrier::new(2 * tensor_ranks));
        std::thread::scope(|scope| {
            let workers = (0..2 * tensor_ranks)
                .map(|rank| {
                    let world = world.clone();
                    let barrier = barrier.clone();
                    let fixture = &fixture;
                    scope.spawn(move || run_rank(fixture, tensor_ranks, rank, world, barrier))
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
    }
}

#[test]
fn voicechat_rnnt_predictor_rejects_cut_step_and_foreign_or_non_f32_parent_before_acquisition() {
    let fixture = fixture();
    let dim = dimensions(&fixture);
    let context = NumericContext::default();
    let selected = partition(dim, 1);
    let route = PartitionBoundaryRoute {
        source_group: 0,
        destination_group: 0,
        source_rank: 0,
        destination_rank: 1,
        route: CommunicationRouteId::new(100),
    };
    let cuts = RnntPredictorPipelineCuts::new(dim, [(route.route, 1)]).unwrap();
    let build = || Arch::new_partition(dim, &selected, &cuts).unwrap();
    let state_arch = build();
    let mut inner = NumericReplicatedPolicy::bounded();
    inner.parameter_overrides = weights(&fixture["parameters"]);
    let acquisitions = Arc::new(AtomicUsize::new(0));
    let policy = Policy {
        inner,
        acquisitions: acquisitions.clone(),
    };
    let mut executor = Executor::new(
        build(),
        policy,
        selected.units().collect(),
        None,
        NumericPartitionTensorAllocator,
        PipelineActivationDtype::Float32,
    )
    .unwrap();
    let driver =
        LayeredPartitionDriver::new(&selected, 0, selected.groups()[0].global_units()).unwrap();
    let communication = PartitionCommunication::new(
        CommunicationManifest::new(1, 0, vec![], vec![]).unwrap(),
        Vec::<RealizedCommunicationGroup<u64>>::new(),
        Vec::<RealizedCommunicationRoute<u64>>::new(),
        NumericCommunicationMetadata(None),
    )
    .unwrap();
    let input = NumericTensor::zeros([1, dim.predictor_width]);
    let step = 0xfedc_ba98_8765_4321u64;
    let words = [1, step as u32 as i32, (step >> 32) as u32 as i32];
    let header = |v: &[i32]| NumericTensor::from_i32_slice(v, &[v.len() as i32], &context).unwrap();
    let full = RnntPredictionState::full(
        dim,
        LstmState {
            layers: (0..2)
                .map(|_| LstmLayerState {
                    hidden: input.clone(),
                    cell: input.clone(),
                })
                .collect(),
        },
    )
    .unwrap();
    let half = input
        .clone()
        .with_dtype(eredu_core::checkpoint::TensorDtype::F16);
    let non_f32 = RnntPredictionState::new(
        dim,
        1..2,
        LstmState {
            layers: vec![LstmLayerState {
                hidden: half.clone(),
                cell: half,
            }],
        },
        None,
    )
    .unwrap();
    let valid = RnntPredictionState::new(
        dim,
        1..2,
        LstmState {
            layers: vec![LstmLayerState {
                hidden: input.clone(),
                cell: input.clone(),
            }],
        },
        None,
    )
    .unwrap();
    for (parent, metadata, incoming) in [
        (None, header(&[0, words[1], words[2]]), input.clone()),
        (None, header(&[1, words[1] ^ 1, words[2]]), input.clone()),
        (None, header(&[1, words[1], words[2] ^ 1]), input.clone()),
        (None, header(&words[..2]), input.clone()),
        (
            None,
            NumericTensor::new([3], vec![1., 0., 0.]),
            input.clone(),
        ),
        (
            None,
            header(&words),
            NumericTensor::zeros([1, dim.predictor_width + 1]),
        ),
        (Some(&full), header(&words), input.clone()),
        (Some(&non_f32), header(&words), input.clone()),
    ] {
        let mut state = State::for_partition(&state_arch, parent).unwrap();
        let mut pass = <Executor as PartitionedGroupExecutor<
            Arch,
            NumericBackend,
            State,
            u64,
            u64,
            NumericCommunicationMetadata,
        >>::begin(
            &mut executor,
            RnntPredictorInput {
                input: &input,
                step,
            },
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
            &route,
            vec![incoming, input.clone(), metadata],
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
        assert!(state.into_prediction(input.clone()).is_err());
    }
    // The same full-width u64 identity is accepted after rejected requests.
    let mut state = State::for_partition(&state_arch, Some(&valid)).unwrap();
    let mut pass = <Executor as PartitionedGroupExecutor<
        Arch,
        NumericBackend,
        State,
        u64,
        u64,
        NumericCommunicationMetadata,
    >>::begin(
        &mut executor,
        RnntPredictorInput {
            input: &input,
            step,
        },
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
        vec![input.clone(), input.clone(), header(&words)],
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
    assert_eq!(acquisitions.load(Ordering::Relaxed), 1);
    assert_eq!(
        valid.local_state().layers[0].hidden.data,
        vec![0.; dim.predictor_width as usize]
    );
    assert_eq!(
        valid.local_state().layers[0].cell.data,
        vec![0.; dim.predictor_width as usize]
    );
}
