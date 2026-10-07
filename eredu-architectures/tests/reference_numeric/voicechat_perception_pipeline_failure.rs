//! Pipeline-local acquisition failure stops the active TP cohort before neural collectives.
use super::voicechat_perception_distributed::communication;
use super::voicechat_perception_residency::{assert_tensor, dimensions, parameters, tensor};
use super::*;
use eredu_architectures::voicechat::perception_layered::{
    perception_parameter_description, PerceptionArchitecture, PerceptionBoundarySchema,
    PerceptionInput, PerceptionRuntimeState, PerceptionUnit,
};
use eredu_runtime::*;
use std::sync::atomic::AtomicBool;
type Arch = PerceptionArchitecture<NumericBackend>;
type State = PerceptionRuntimeState<NumericTensor>;
type Unit = PerceptionUnit<NumericBackend>;
struct Policy {
    inner: NumericReplicatedPolicy<Unit>,
    fail: Arc<AtomicBool>,
    aborts: Arc<AtomicUsize>,
    completed: Arc<AtomicUsize>,
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
        if address.group() == 1 && self.fail.load(Ordering::Relaxed) {
            return Err(LayerwiseAcquireError::Policy(Error::backend(
                "injected stage-local parameter failure",
            )));
        }
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
            .complete(ordinal, address, lease, output, state, retained, context)?;
        self.completed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    fn finish(&mut self, output: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        self.inner.finish(output, context)
    }
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        context: &NumericContext,
    ) {
        self.aborts.fetch_add(1, Ordering::Relaxed);
        self.inner.abort(active, context)
    }
}
type Executor =
    PipelinePartitionExecutor<Arch, NumericBackend, State, Policy, NumericPartitionTensorAllocator>;
#[test]
fn voicechat_perception_pipeline_tp_acquisition_failure_excludes_inactive_stages() {
    let fixture: serde_json::Value =
        serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap();
    let dimensions = dimensions(&fixture);
    let world = Arc::new(NumericPartitionWorld::default());
    // Only the first stage's two TP ranks execute. An accidental allworld fence
    // would wait for the deliberately absent second stage.
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let fixture = &fixture;
                let world = world.clone();
                scope.spawn(move || {
                    let description = perception_parameter_description(dimensions).unwrap();
                    let layout = derive_partitioned_local_layout(
                        &description,
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap(),
                    )
                    .unwrap();
                    let context =
                        NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let complete = Arch::new_parallel(dimensions, &layout).unwrap();
                    let partition = ArchitecturePartition::from_description(
                        &description,
                        [
                            (description.graph().groups()[0].id(), 0..1),
                            (description.graph().groups()[1].id(), 0..dimensions.layers),
                        ],
                        PartitionOwnership::new(true, false, std::iter::empty::<String>()).unwrap(),
                        &complete.state_layout().unwrap(),
                        &ArchitectureStatePartitionPlan::new([
                            ArchitectureStatePartitionRule::group_units(1, 0..dimensions.layers),
                        ]),
                        (),
                        PerceptionBoundarySchema::new(dimensions),
                    )
                    .unwrap();
                    let build =
                        || Arch::new_partition(dimensions, Some(&layout), &partition).unwrap();
                    let state_arch = build();
                    let parent = State::for_partition(&state_arch, None).unwrap();
                    let (manifest, _, _, cohort) = communication(dimensions, 2, rank, 1, 2);
                    let cohort = cohort.unwrap();
                    validate_pipeline_local_agreement(&manifest, cohort).unwrap();
                    world.realize_manifest(&manifest).unwrap();
                    let parallel = numeric_parallel_context(&world, &manifest, cohort).unwrap();
                    let mut inner = NumericReplicatedPolicy::bounded();
                    inner.parameter_overrides = parameters(fixture)
                        .into_iter()
                        .map(|(name, value)| {
                            let selected =
                                select_parameter(&value, layout.tensor(&name).unwrap().placement())
                                    .unwrap();
                            (name, selected)
                        })
                        .collect();
                    let fail = Arc::new(AtomicBool::new(rank == 0));
                    let aborts = Arc::new(AtomicUsize::new(0));
                    let completed = Arc::new(AtomicUsize::new(0));
                    let policy = Policy {
                        inner,
                        fail: fail.clone(),
                        aborts: aborts.clone(),
                        completed: completed.clone(),
                    };
                    let mut executor = Executor::new(
                        build(),
                        policy,
                        partition.units().collect(),
                        Some(parallel.clone()),
                        NumericPartitionTensorAllocator,
                        PipelineActivationDtype::Float32,
                    )
                    .unwrap()
                    .with_local_agreement(Some(cohort));
                    let communication = numeric_partition_communication(&world, manifest).unwrap();
                    let driver0 = LayeredPartitionDriver::new_with_state_ownership(
                        &partition,
                        0,
                        0..1,
                        false,
                    )
                    .unwrap();
                    let driver1 = LayeredPartitionDriver::new_with_state_ownership(
                        &partition,
                        1,
                        0..dimensions.layers,
                        true,
                    )
                    .unwrap();
                    let window = &fixture["subsampling"]["steps"][0];
                    let input = tensor(&window["input"]);
                    let drop = window["drop"].as_u64().unwrap() as usize;
                    for retry in [false, true] {
                        if retry {
                            fail.store(false, Ordering::Relaxed);
                        }
                        let mut state = State::for_partition(&state_arch, Some(&parent)).unwrap();
                        let mut pass = <Executor as PartitionedGroupExecutor<
                            Arch,
                            NumericBackend,
                            State,
                            u64,
                            u64,
                            NumericCommunicationMetadata,
                        >>::begin(
                            &mut executor,
                            PerceptionInput {
                                mel: &input,
                                drop_preencoded: drop,
                            },
                            &mut state,
                            ExpertPass::Prefill,
                            &context,
                        )
                        .unwrap();
                        executor
                            .execute_group(
                                &mut pass,
                                &driver0,
                                &mut state,
                                &communication,
                                &context,
                                &context,
                                &mut NoopObserver,
                            )
                            .unwrap();
                        let before = parallel.trace().len();
                        let result = executor.execute_group(
                            &mut pass,
                            &driver1,
                            &mut state,
                            &communication,
                            &context,
                            &context,
                            &mut NoopObserver,
                        );
                        if !retry {
                            assert!(result.is_err());
                            assert_eq!(
                                parallel.trace().len(),
                                before,
                                "peer cannot enter Conformer collectives after failed acquisition"
                            );
                            assert_eq!(completed.load(Ordering::Relaxed), 1);
                            assert!(aborts.load(Ordering::Relaxed) > 0);
                            assert_eq!(state.frames(), 0);
                        } else {
                            result.unwrap();
                            assert!(parallel.trace().len() > before);
                            assert_eq!(state.frames(), 1);
                            assert_tensor(
                                state.blocks()[0].channel().unwrap(),
                                &fixture["perception"]["steps"][0]["channel_cache"],
                                fixture,
                            );
                        }
                        assert_eq!(parent.frames(), 0);
                        assert!(parent.blocks()[0].channel().is_none());
                        assert!(parent.blocks()[0].convolution().is_none());
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}
