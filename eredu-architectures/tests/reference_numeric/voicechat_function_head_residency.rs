//! Pinned normalized-hidden readout through independently leased function heads.
use super::voicechat_tts_generator::tensor;
use super::*;
use eredu_architectures::voicechat::{
    channels::FunctionHead,
    function_head_layered::{
        FunctionHeadArchitecture, FunctionHeadDimensions, FunctionHeadExecution,
        LayerwiseFunctionHead,
    },
};
use serde_json::Value;

type Unit = FunctionHead<NumericBackend>;
pub(super) struct BoundFixturePolicy {
    pub(super) inner: RebuildingUnitPolicy,
    parameters: BTreeMap<String, NumericTensor>,
    aborted: Vec<bool>,
    malformed_unit: bool,
}
impl LayerwisePolicy<NumericBackend, Unit> for BoundFixturePolicy {
    type Lease = RebuiltUnitLease<Unit>;
    type Error = Error;
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        _: &NumericContext,
    ) {
        self.aborted.push(active.is_some());
        drop(active);
    }
    fn begin(&mut self, input: &NumericTensor, context: &NumericContext) -> Result<(), Error> {
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::begin(
            &mut self.inner,
            input,
            context,
        )
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
        let parameters = &self.parameters;
        let malformed_unit = self.malformed_unit;
        self.inner.acquire(
            ordinal,
            address,
            |context| {
                let mut unit = build(context)?;
                if malformed_unit {
                    unit = FunctionHead::new(4, 24, dense_linear_format(), context).unwrap();
                }
                replace_numeric_parameters(&mut unit, parameters);
                Ok(unit)
            },
            context,
        )
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
        <RebuildingUnitPolicy as LayerwisePolicy<NumericBackend, Unit>>::finish(
            &mut self.inner,
            output,
            context,
        )
    }
}

pub(super) fn bounded(
    fixture: &Value,
) -> LayerwiseFunctionHead<NumericBackend, BoundFixturePolicy> {
    let dimensions = FunctionHeadDimensions::new(4, 24).unwrap();
    LayerwiseFunctionHead::new(
        FunctionHeadArchitecture::new(dimensions, dense_linear_format()).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            aborted: Vec::new(),
            malformed_unit: false,
            parameters: BTreeMap::from([(
                "stt_model.function_head.weight".into(),
                tensor(&fixture["parameters"]["stt_model.function_head.weight"]),
            )]),
        },
    )
}

fn compare(actual: &NumericTensor, fixture: &Value) {
    super::voicechat_conditioning::compare(actual, &fixture["f32"]["function_logits"], fixture);
}

#[test]
fn voicechat_function_head_residency_matches_pinned_normalized_readout() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let hidden = tensor(&fixture["f32"]["normalized_hidden"]);
    let mut ordinary = super::voicechat_backbone_execution::function(&fixture, &context);
    let architecture = FunctionHeadArchitecture::new(
        FunctionHeadDimensions::new(4, 24).unwrap(),
        dense_linear_format(),
    )
    .unwrap();
    let unit = super::voicechat_backbone_execution::function(&fixture, &context);
    let mut resident =
        LayerwiseFunctionHead::new(architecture, NumericReplicatedPolicy::resident(vec![unit]));
    let mut bounded = bounded(&fixture);
    compare(&ordinary.project(&hidden, &context).unwrap(), &fixture);
    compare(&resident.project(&hidden, &context).unwrap(), &fixture);
    compare(&bounded.project(&hidden, &context).unwrap(), &fixture);
    assert_eq!(
        bounded.runtime().policy().inner.events,
        expected_stream_events(&[(0, 0)])
    );
}

#[test]
fn voicechat_function_head_residency_acquisition_failure_retries_and_invalid_hidden_is_preflighted()
{
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let hidden = tensor(&fixture["f32"]["normalized_hidden"]);
    let original = hidden.data.clone();
    let mut bounded = bounded(&fixture);
    for malformed in [
        NumericTensor::zeros(vec![1, 3, 5]),
        NumericTensor::zeros(vec![3, 4]),
        NumericTensor::zeros(vec![1, 0, 4]),
        NumericTensor::from_i32_slice(&[1; 4], &[1, 1, 4], &context).unwrap(),
    ] {
        assert!(bounded.project(&malformed, &context).is_err());
        assert!(bounded.runtime().policy().inner.events.is_empty());
    }
    bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(0);
    assert!(bounded.project(&hidden, &context).is_err());
    assert_eq!(hidden.data, original);
    assert_eq!(
        bounded.runtime().policy().inner.events,
        vec![
            StreamPolicyEvent::Begin,
            StreamPolicyEvent::Acquire(0, 0, 0)
        ]
    );
    bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
    bounded.runtime_mut().policy_mut().inner.events.clear();
    compare(&bounded.project(&hidden, &context).unwrap(), &fixture);
    assert_eq!(hidden.data, original);
    assert_eq!(
        bounded.runtime().policy().inner.events,
        expected_stream_events(&[(0, 0)])
    );
}

#[test]
fn voicechat_function_head_parallel_resident_and_bounded_gather_pinned_logits() {
    use eredu_architectures::voicechat::function_head_layered::function_head_parameter_description;

    let original: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    // Removing one output row preserves the independent NVIDIA projection oracle
    // while requiring an uneven vocabulary gather (12 rows plus 11 rows).
    for vocabulary in [24usize, 23] {
        let mut fixture = original.clone();
        let weight = &mut fixture["parameters"]["stt_model.function_head.weight"];
        weight["shape"][0] = vocabulary.into();
        weight["values"]
            .as_array_mut()
            .unwrap()
            .truncate(vocabulary * 4);
        let logits = &mut fixture["f32"]["function_logits"];
        logits["shape"][2] = vocabulary.into();
        logits["values"] = Value::Array(
            logits["values"]
                .as_array()
                .unwrap()
                .chunks_exact(24)
                .flat_map(|row| row[..vocabulary].iter().cloned())
                .collect(),
        );
        let dimensions = FunctionHeadDimensions::new(4, vocabulary as i32).unwrap();
        let description =
            function_head_parameter_description(dimensions, &dense_linear_format()).unwrap();
        let world = Arc::new(NumericPartitionWorld::default());
        let group = NumericParallelGroup::new(2);
        std::thread::scope(|scope| {
            let workers = (0..2)
                .map(|rank| {
                    let world = world.clone();
                    let group = group.clone();
                    let fixture = &fixture;
                    let description = &description;
                    scope.spawn(move || {
                        let topology = ParallelRankTopology::new(
                            ParallelTopology::new(2, 1, 1, 1).unwrap(),
                            rank,
                        )
                        .unwrap();
                        let layout =
                            eredu_runtime::derive_partitioned_local_layout(description, topology)
                                .unwrap();
                        let context =
                            NumericContext::with_partition(layout.clone(), rank, world.clone());
                        let parallel =
                            NumericParallelContext::new_partitioned(rank, rank, group, world);
                        let name = "stt_model.function_head.weight";
                        let parameters = BTreeMap::from([(
                            name.into(),
                            select_parameter(
                                &tensor(&fixture["parameters"][name]),
                                layout.tensor(name).unwrap().placement(),
                            )
                            .unwrap(),
                        )]);
                        let architecture = || {
                            FunctionHeadArchitecture::<NumericBackend>::new_parallel(
                                dimensions,
                                dense_linear_format(),
                                &layout,
                            )
                            .unwrap()
                        };
                        let hidden = tensor(&fixture["f32"]["normalized_hidden"]);
                        let mut unit = architecture().build_unit(0, 0, &context).unwrap();
                        replace_numeric_parameters(&mut unit, &parameters);
                        let mut resident = LayerwiseFunctionHead::new(
                            architecture(),
                            NumericReplicatedPolicy::resident(vec![unit]),
                        );
                        compare(
                            &resident
                                .project_parallel(&hidden, &parallel, &context)
                                .unwrap(),
                            fixture,
                        );
                        let mut bounded = LayerwiseFunctionHead::new(
                            architecture(),
                            BoundFixturePolicy {
                                inner: RebuildingUnitPolicy::default(),
                                aborted: Vec::new(),
                                malformed_unit: false,
                                parameters,
                            },
                        );
                        // A collective head cannot fall back to ordinary projection,
                        // and communicator drift must fail before acquiring its matrix.
                        assert!(bounded.project(&hidden, &context).is_err());
                        for wrong in [
                            NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                            NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                        ] {
                            assert!(bounded.project_parallel(&hidden, &wrong, &context).is_err());
                        }
                        assert!(bounded.runtime().policy().inner.events.is_empty());
                        bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(0);
                        assert!(
                            bounded
                                .project_parallel(&hidden, &parallel, &context)
                                .is_err()
                        );
                        assert_eq!(
                            bounded.runtime().policy().inner.events,
                            vec![
                                StreamPolicyEvent::Begin,
                                StreamPolicyEvent::Acquire(0, 0, 0)
                            ]
                        );
                        bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
                        bounded.runtime_mut().policy_mut().inner.events.clear();
                        compare(
                            &bounded
                                .project_parallel(&hidden, &parallel, &context)
                                .unwrap(),
                            fixture,
                        );
                        assert_eq!(
                            bounded.runtime().policy().inner.events,
                            expected_stream_events(&[(0, 0)])
                        );
                    })
                })
                .collect::<Vec<_>>();
            for worker in workers {
                worker.join().unwrap();
            }
        });
        assert_eq!(world.tensor_collective_calls(), 4);
    }
}

#[test]
fn voicechat_function_head_parallel_local_failures_agree_before_gather_and_retry() {
    use super::voicechat_rnnt_coordination::{Coordinator, coordinator_pair};
    use eredu_architectures::voicechat::{
        channels::ChannelError, function_head_layered::function_head_parameter_description,
    };
    use eredu_runtime::{
        LayeredLocalPhase, LayeredTraversalHook, replicated_execution::ReplicatedCoordinator,
    };
    use std::{cell::RefCell, rc::Rc, sync::atomic::Ordering};

    struct Hook(Rc<RefCell<Coordinator>>);
    impl LayeredTraversalHook<NumericBackend, (), ChannelError> for Hook {
        fn local_phase(
            &mut self,
            phase: LayeredLocalPhase,
            success: bool,
            context: &NumericContext,
        ) -> Result<(), ChannelError> {
            let (phase, group, index) = match phase {
                LayeredLocalPhase::Initialization => (0, 0, 0),
                LayeredLocalPhase::PolicyBegin => (1, 0, 0),
                LayeredLocalPhase::GroupIngress { group } => (2, group, 0),
                LayeredLocalPhase::BeforeUnit { group, index } => (3, group, index),
                LayeredLocalPhase::Acquisition { group, index } => (4, group, index),
                LayeredLocalPhase::UnitExecution { group, index } => (5, group, index),
                LayeredLocalPhase::UnitCompletion { group, index } => (6, group, index),
                LayeredLocalPhase::GroupCompletion { group } => (7, group, 0),
                LayeredLocalPhase::Finish => (8, 0, 0),
            };
            if !self.0.borrow_mut().agree(
                [
                    0x46554e43484f4f4b,
                    phase,
                    group as u64,
                    index as u64,
                    0,
                    0,
                    0,
                    0,
                ],
                success,
                context,
            )? {
                return Err(ChannelError::Configuration(
                    "function-head peer failed local traversal phase",
                ));
            }
            Ok(())
        }
    }

    let (coordinators, timeouts) = coordinator_pair();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, coordinator)| {
                let world = world.clone();
                let group = group.clone();
                scope.spawn(move || {
                    let fixture: Value =
                        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON)
                            .unwrap();
                    let dimensions = FunctionHeadDimensions::new(4, 24).unwrap();
                    let description =
                        function_head_parameter_description(dimensions, &dense_linear_format())
                            .unwrap();
                    let topology =
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap();
                    let layout =
                        eredu_runtime::derive_partitioned_local_layout(&description, topology)
                            .unwrap();
                    let context =
                        NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let parallel =
                        NumericParallelContext::new_partitioned(rank, rank, group, world.clone());
                    let name = "stt_model.function_head.weight";
                    let parameters = BTreeMap::from([(
                        name.into(),
                        select_parameter(
                            &tensor(&fixture["parameters"][name]),
                            layout.tensor(name).unwrap().placement(),
                        )
                        .unwrap(),
                    )]);
                    let mut head = LayerwiseFunctionHead::new(
                        FunctionHeadArchitecture::<NumericBackend>::new_parallel(
                            dimensions,
                            dense_linear_format(),
                            &layout,
                        )
                        .unwrap(),
                        BoundFixturePolicy {
                            inner: RebuildingUnitPolicy::default(),
                            parameters,
                            aborted: Vec::new(),
                            malformed_unit: false,
                        },
                    );
                    let hidden = tensor(&fixture["f32"]["normalized_hidden"]);
                    let coordinator = Rc::new(RefCell::new(coordinator));
                    let mut hook = Hook(coordinator.clone());
                    let agree = |descriptor, success, context: &NumericContext| {
                        coordinator
                            .borrow_mut()
                            .agree(descriptor, success, context)
                            .map_err(ChannelError::from)
                    };

                    let malformed = NumericTensor::zeros([1, 3, 5]);
                    let error = head
                        .project_parallel_guarded(
                            if rank == 0 { &malformed } else { &hidden },
                            &parallel,
                            &context,
                            &mut hook,
                            agree,
                        )
                        .unwrap_err();
                    if rank == 0 {
                        assert!(matches!(error, ChannelError::Shape(_)));
                    }
                    assert!(head.runtime().policy().inner.events.is_empty());
                    assert_eq!(world.tensor_collective_calls(), 0);

                    // Both payloads satisfy the local head geometry, but their
                    // different batch extents cannot share one collective call.
                    let larger_batch = NumericTensor::zeros([
                        hidden.shape()[0] + 1,
                        hidden.shape()[1],
                        hidden.shape()[2],
                    ]);
                    assert!(
                        head.project_parallel_guarded(
                            if rank == 0 { &hidden } else { &larger_batch },
                            &parallel,
                            &context,
                            &mut hook,
                            agree,
                        )
                        .is_err()
                    );
                    assert!(head.runtime().policy().inner.events.is_empty());
                    assert_eq!(world.tensor_collective_calls(), 0);

                    head.runtime_mut().policy_mut().inner.fail_acquire = (rank == 0).then_some(0);
                    let error = head
                        .project_parallel_guarded(&hidden, &parallel, &context, &mut hook, agree)
                        .unwrap_err();
                    if rank == 0 {
                        let mut source: &(dyn std::error::Error + 'static) = &error;
                        let mut found_original = false;
                        loop {
                            if source.downcast_ref::<Error>().is_some_and(|error| {
                                error
                                    .to_string()
                                    .contains("injected dense-stream acquisition failure")
                            }) {
                                found_original = true;
                            }
                            match source.source() {
                                Some(next) => source = next,
                                None => break,
                            }
                        }
                        assert!(
                            found_original,
                            "the original acquisition error remains in the source chain"
                        );
                    }
                    assert_eq!(world.tensor_collective_calls(), 0);
                    assert_eq!(
                        head.runtime().policy().aborted.last(),
                        Some(&(rank == 1)),
                        "the successful peer's acquired lease is handed to abort"
                    );
                    assert_eq!(
                        head.runtime().policy().inner.events,
                        vec![
                            StreamPolicyEvent::Begin,
                            StreamPolicyEvent::Acquire(0, 0, 0)
                        ]
                    );

                    head.runtime_mut().policy_mut().inner.fail_acquire = None;
                    head.runtime_mut().policy_mut().inner.events.clear();
                    head.runtime_mut().policy_mut().malformed_unit = rank == 0;
                    assert!(
                        head.project_parallel_guarded(
                            &hidden, &parallel, &context, &mut hook, agree
                        )
                        .is_err()
                    );
                    assert_eq!(world.tensor_collective_calls(), 0);
                    assert_eq!(head.runtime().policy().aborted.last(), Some(&true));
                    assert_eq!(
                        head.runtime().policy().inner.events,
                        vec![
                            StreamPolicyEvent::Begin,
                            StreamPolicyEvent::Acquire(0, 0, 0)
                        ]
                    );

                    head.runtime_mut().policy_mut().malformed_unit = false;
                    head.runtime_mut().policy_mut().inner.events.clear();
                    compare(
                        &head
                            .project_parallel_guarded(
                                &hidden, &parallel, &context, &mut hook, agree,
                            )
                            .unwrap(),
                        &fixture,
                    );
                    assert_eq!(
                        head.runtime().policy().inner.events,
                        expected_stream_events(&[(0, 0)])
                    );
                    assert_eq!(head.runtime().policy().aborted.len(), 2);
                    assert_eq!(
                        hidden.data,
                        tensor(&fixture["f32"]["normalized_hidden"]).data
                    );
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(
        world.tensor_collective_calls(),
        2,
        "only the successful retry gathers rows"
    );
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
