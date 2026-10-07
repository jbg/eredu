//! RNNT prediction and greedy transitions through independently leased units.
use super::voicechat_mog_residency::weights;
use super::voicechat_rnnt::{assert_state, assert_values, dimensions, fixture, frame, limits};
use super::*;
use eredu_architectures::voicechat::{
    rnnt::{RnntExecution, RnntState},
    rnnt_layered::{LayerwiseRnnt, RnntArchitecture, RnntUnit},
};
use eredu_runtime::derive_partitioned_local_layout;

type Unit = RnntUnit<NumericBackend>;
pub(super) struct Policy {
    inner: NumericReplicatedPolicy<Unit>,
    pub(super) acquired: Vec<usize>,
    pub(super) completed: Vec<usize>,
    pub(super) aborted: usize,
    pub(super) fail_ordinal: Option<usize>,
}
impl Policy {
    pub(super) fn bounded(parameters: BTreeMap<String, NumericTensor>) -> Self {
        let mut inner = NumericReplicatedPolicy::bounded();
        inner.parameter_overrides = parameters;
        Self {
            inner,
            acquired: vec![],
            completed: vec![],
            aborted: 0,
            fail_ordinal: None,
        }
    }
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
        self.acquired.push(ordinal);
        if self.fail_ordinal == Some(ordinal) {
            return Err(LayerwiseAcquireError::Policy(Error::backend(
                "injected RNNT acquisition failure",
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
        self.completed.push(ordinal);
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
        self.aborted += 1;
        self.inner.abort(active, context);
    }
}
pub(super) fn provider(
    fixture: &serde_json::Value,
    resident: bool,
    context: &NumericContext,
) -> LayerwiseRnnt<NumericBackend, Policy> {
    let architecture = RnntArchitecture::new(dimensions(fixture)).unwrap();
    let parameters = weights(&fixture["parameters"]);
    let inner = if resident {
        let units = architecture
            .execution_graph()
            .unwrap()
            .groups()
            .iter()
            .enumerate()
            .flat_map(|(group, _)| {
                let architecture = &architecture;
                let parameters = &parameters;
                (0..architecture.group_unit_count(group).unwrap()).map(move |index| {
                    let mut unit = architecture.build_unit(group, index, context).unwrap();
                    replace_numeric_parameters(&mut unit, parameters);
                    unit
                })
            })
            .collect();
        NumericReplicatedPolicy::resident(units)
    } else {
        let mut policy = NumericReplicatedPolicy::bounded();
        policy.parameter_overrides = parameters;
        policy
    };
    LayerwiseRnnt::new(
        architecture,
        Policy {
            inner,
            acquired: vec![],
            completed: vec![],
            aborted: 0,
            fail_ordinal: None,
        },
    )
}
pub(super) fn take_acquisitions(model: &mut LayerwiseRnnt<NumericBackend, Policy>) -> Vec<usize> {
    let policy = model.runtime_mut().policy_mut();
    assert_eq!(
        policy.acquired, policy.completed,
        "every successful lease completed"
    );
    policy.completed.clear();
    std::mem::take(&mut policy.acquired)
}
pub(super) fn same_state(a: &RnntState<NumericTensor>, b: &RnntState<NumericTensor>) {
    assert_eq!(a.hypothesis(), b.hypothesis());
    assert_eq!(a.frames(), b.frames());
    let a = a.predictor().unwrap();
    let b = b.predictor().unwrap();
    assert_eq!(a.dimensions(), b.dimensions());
    assert_eq!(a.global_layers(), b.global_layers());
    assert_eq!(a.output().data, b.output().data);
    let a = a.local_state();
    let b = b.local_state();
    assert_eq!(a.layers.len(), b.layers.len());
    for (a, b) in a.layers.iter().zip(&b.layers) {
        assert_eq!(a.hidden.shape(), b.hidden.shape());
        assert_eq!(a.hidden.data, b.hidden.data);
        assert_eq!(a.cell.data, b.cell.data);
    }
}

#[test]
fn voicechat_rnnt_resident_and_bounded_phases_match_nvidia_without_unneeded_units() {
    let fixture = fixture();
    let context = NumericContext::default();
    for resident in [true, false] {
        let mut model = provider(&fixture, resident, &context);
        let initial = RnntState::default();
        let mut prediction = model.prime(&initial, &context).unwrap();
        assert!(initial.predictor().is_none());
        assert_state(
            prediction.predictor().unwrap(),
            &fixture["predictor_steps"][0],
            &fixture,
        );
        assert_eq!(
            take_acquisitions(&mut model),
            [0, 1, 2, 1, 2],
            "priming resolves embedding dtype once and leases no joint weights"
        );
        for step in fixture["predictor_steps"]
            .as_array()
            .unwrap()
            .iter()
            .skip(1)
        {
            let parent = prediction.clone();
            prediction = model
                .predict_token(
                    step["token"].as_u64().unwrap() as usize,
                    &parent,
                    32,
                    &context,
                )
                .unwrap();
            assert_state(prediction.predictor().unwrap(), step, &fixture);
            assert_eq!(prediction.frames(), 0);
            assert_eq!(
                take_acquisitions(&mut model),
                [0, 1, 2],
                "one token only leases embedding and recurrent layers"
            );
        }
        for case in ["streaming_frames", "feedback_terminates_frame"] {
            let mut state = model.prime(&initial, &context).unwrap();
            take_acquisitions(&mut model);
            let mut emitted = Vec::new();
            for (index, expected) in fixture[case].as_array().unwrap().iter().enumerate() {
                let encoder = frame(expected);
                let logits = model.joint_logits(&encoder, &state, &context).unwrap();
                assert_values(
                    &logits.data,
                    &expected["joint_calls"][0]["logits"],
                    &fixture,
                );
                assert_eq!(
                    take_acquisitions(&mut model),
                    [3, 4, 5],
                    "joint evaluation does not acquire predictor weights"
                );
                let parent = state.clone();
                let result = model
                    .decode_frame(&encoder, &state, limits(32), &context)
                    .unwrap();
                same_state(&state, &parent);
                let tokens = expected["emitted"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect::<Vec<_>>();
                assert_eq!(result.emitted, tokens);
                assert_eq!(
                    result.first_blank,
                    expected["first_blank"].as_bool().unwrap()
                );
                assert_eq!(
                    result.ended_blank,
                    expected["ended_blank"].as_bool().unwrap()
                );
                assert_eq!(
                    result.hit_symbol_limit,
                    expected["hit_symbol_limit"].as_bool().unwrap()
                );
                emitted.extend(tokens);
                assert_eq!(result.state.hypothesis(), emitted);
                assert_eq!(result.state.frames(), index as u64 + 1);
                assert_state(result.state.predictor().unwrap(), expected, &fixture);
                take_acquisitions(&mut model);
                state = result.state;
            }
        }
    }
}

#[test]
fn voicechat_rnnt_late_lease_failure_replay_and_preflight_preserve_parent() {
    let fixture = fixture();
    let context = NumericContext::default();
    for resident in [true, false] {
        let mut model = provider(&fixture, resident, &context);
        let parent = model
            .decode_frame(
                &frame(&fixture["streaming_frames"][0]),
                &RnntState::default(),
                limits(32),
                &context,
            )
            .unwrap()
            .state;
        let saved = parent.clone();
        take_acquisitions(&mut model);
        model.runtime_mut().policy_mut().fail_ordinal = Some(2);
        let expected = &fixture["streaming_frames"][1];
        assert!(model
            .decode_frame(&frame(expected), &parent, limits(32), &context)
            .is_err());
        same_state(&parent, &saved);
        let policy = model.runtime_mut().policy_mut();
        assert_eq!(policy.acquired.last(), Some(&2));
        assert!(
            policy.completed.contains(&1),
            "failure follows a tentative first-layer update"
        );
        assert!(
            policy.completed.contains(&5),
            "joint logits preceded the tentative token update"
        );
        assert!(policy.aborted > 0);
        policy.fail_ordinal = None;
        policy.acquired.clear();
        policy.completed.clear();
        let retried = model
            .decode_frame(&frame(expected), &parent, limits(32), &context)
            .unwrap();
        assert_state(retried.state.predictor().unwrap(), expected, &fixture);
        assert_eq!(retried.state.frames(), 2);
        take_acquisitions(&mut model);
        let replayed = model
            .decode_frame(&frame(expected), &saved, limits(32), &context)
            .unwrap();
        same_state(&retried.state, &replayed.state);
        assert_eq!(retried.emitted, replayed.emitted);
        same_state(&parent, &saved);
        take_acquisitions(&mut model);
        assert!(model
            .decode_frame(&NumericTensor::zeros([1, 4]), &parent, limits(32), &context)
            .is_err());
        assert!(model
            .predict_token(usize::MAX, &parent, 32, &context)
            .is_err());
        assert!(model
            .joint_logits(&frame(expected), &RnntState::default(), &context)
            .is_err());
        assert!(
            model.runtime().policy().acquired.is_empty(),
            "invalid ingress, tokens and unprimed joint state fail before parameter acquisition"
        );
        same_state(&parent, &saved);
    }
}

fn parallel_provider(
    dimensions: eredu_architectures::voicechat::rnnt::RnntDimensions,
    parameters: BTreeMap<String, NumericTensor>,
    layout: &LocalModelLayout,
) -> LayerwiseRnnt<NumericBackend, Policy> {
    let architecture = RnntArchitecture::new_parallel(dimensions, layout).unwrap();
    let mut inner = NumericReplicatedPolicy::bounded();
    inner.parameter_overrides = parameters
        .into_iter()
        .map(|(name, value)| {
            let selected =
                select_parameter(&value, layout.tensor(&name).unwrap().placement()).unwrap();
            (name, selected)
        })
        .collect();
    LayerwiseRnnt::new(
        architecture,
        Policy {
            inner,
            acquired: vec![],
            completed: vec![],
            aborted: 0,
            fail_ordinal: None,
        },
    )
}

#[test]
fn voicechat_rnnt_tp2_leased_prediction_and_joint_phases_match_pinned_nvidia() {
    use eredu_architectures::voicechat::rnnt_layered::rnnt_parameter_description;
    let fixture = fixture();
    let dimensions = dimensions(&fixture);
    let description = rnnt_parameter_description(dimensions).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let fixture = &fixture;
                let description = &description;
                let world = world.clone();
                let group = group.clone();
                scope.spawn(move || {
                    let topology =
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap();
                    let layout = derive_partitioned_local_layout(description, topology).unwrap();
                    let context =
                        NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let parallel =
                        NumericParallelContext::new_partitioned(rank, rank, group, world);
                    let mut model =
                        parallel_provider(dimensions, weights(&fixture["parameters"]), &layout);
                    for wrong in [
                        NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                        NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                    ] {
                        assert!(model.embed_token_parallel(0, &wrong, &context).is_err());
                    }
                    assert!(model.embed_token(0, &context).is_err());
                    assert!(model.runtime().policy().acquired.is_empty());
                    let zero = model
                        .embed_token_parallel(dimensions.vocab_size as usize, &parallel, &context)
                        .unwrap()
                        .zeros_like(&context)
                        .unwrap();
                    let first = model
                        .predictor_step_parallel(&zero, None, &parallel, &context)
                        .unwrap();
                    let primed = model
                        .predictor_step_parallel(&zero, Some(&first), &parallel, &context)
                        .unwrap();
                    assert_state(&primed, &fixture["predictor_steps"][0], fixture);
                    assert_eq!(take_acquisitions(&mut model), [0, 1, 2, 1, 2]);
                    let mut prediction = primed.clone();
                    for step in fixture["predictor_steps"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .skip(1)
                    {
                        let token = step["token"].as_u64().unwrap() as usize;
                        let embedded = model
                            .embed_token_parallel(token, &parallel, &context)
                            .unwrap();
                        prediction = model
                            .predictor_step_parallel(
                                &embedded,
                                Some(&prediction),
                                &parallel,
                                &context,
                            )
                            .unwrap();
                        assert_state(&prediction, step, fixture);
                        assert_eq!(take_acquisitions(&mut model), [0, 1, 2]);
                    }
                    // Decisions are fixed by NVIDIA's fixture. These assertions isolate
                    // neural TP numerics from the shared coordinator's decision policy.
                    for case in ["streaming_frames", "feedback_terminates_frame"] {
                        let mut prediction = primed.clone();
                        for expected in fixture[case].as_array().unwrap() {
                            let encoded = model
                                .encode_frame_parallel(&frame(expected), &parallel, &context)
                                .unwrap();
                            let emissions = expected["emitted"].as_array().unwrap();
                            for (index, call) in expected["joint_calls"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .enumerate()
                            {
                                let projected = model
                                    .project_prediction_parallel(
                                        prediction.output(),
                                        &parallel,
                                        &context,
                                    )
                                    .unwrap();
                                let sum = encoded.add(&projected, &context).unwrap();
                                assert_values(&sum.data, &call["hidden_sum"], fixture);
                                let logits = model
                                    .project_output_parallel(
                                        &sum.maximum_scalar(0., &context).unwrap(),
                                        &parallel,
                                        &context,
                                    )
                                    .unwrap();
                                assert_values(&logits.data, &call["logits"], fixture);
                                if let Some(token) = emissions.get(index) {
                                    let embedded = model
                                        .embed_token_parallel(
                                            token.as_u64().unwrap() as usize,
                                            &parallel,
                                            &context,
                                        )
                                        .unwrap();
                                    prediction = model
                                        .predictor_step_parallel(
                                            &embedded,
                                            Some(&prediction),
                                            &parallel,
                                            &context,
                                        )
                                        .unwrap();
                                }
                            }
                            assert_state(&prediction, expected, fixture);
                            take_acquisitions(&mut model);
                        }
                    }
                    assert_state(&primed, &fixture["predictor_steps"][0], fixture);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert!(world.tensor_collective_calls() > 0);
}

#[test]
fn voicechat_rnnt_tp2_uneven_vocabulary_preserves_blank_owner_row_order_and_bias() {
    use eredu_architectures::voicechat::{
        rnnt::RnntDimensions, rnnt_layered::rnnt_parameter_description,
    };
    let dimensions = RnntDimensions {
        vocab_size: 4,
        predictor_width: 2,
        predictor_layers: 1,
        encoder_width: 2,
        joint_width: 2,
    };
    let description = rnnt_parameter_description(dimensions).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let world = world.clone();
                let group = group.clone();
                let description = &description;
                scope.spawn(move || {
                    let topology =
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap();
                    let layout = derive_partitioned_local_layout(description, topology).unwrap();
                    let context =
                        NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let parallel =
                        NumericParallelContext::new_partitioned(rank, rank, group, world);
                    let parameters = BTreeMap::from([
                        (
                            "stt_model.rnnt_decoder.prediction.embed.weight".into(),
                            NumericTensor::new(
                                vec![5, 2],
                                vec![1., -2., 3., -4., 5., -6., 7., -8., 0.25, -0.75],
                            ),
                        ),
                        (
                            "stt_model.rnnt_joint.joint_net.2.weight".into(),
                            NumericTensor::new(
                                vec![5, 2],
                                vec![1., 2., -3., 4., 5., -6., -7., 8., 9., 10.],
                            ),
                        ),
                        (
                            "stt_model.rnnt_joint.joint_net.2.bias".into(),
                            NumericTensor::new(vec![5], vec![0.25, -0.5, 0.75, -1., 1.25]),
                        ),
                    ]);
                    let mut model = parallel_provider(dimensions, parameters, &layout);
                    // The first rank holds rows 0..3 and the second rows 3..5.
                    // An explicit blank embedding must retain row 4, before the
                    // priming policy replaces its values with typed zeros.
                    for (token, expected) in [
                        (0, [1., -2.]),
                        (2, [5., -6.]),
                        (3, [7., -8.]),
                        (4, [0.25, -0.75]),
                    ] {
                        let embedding = model
                            .embed_token_parallel(token, &parallel, &context)
                            .unwrap();
                        assert_eq!(embedding.shape(), [1, 2]);
                        assert_eq!(embedding.data, expected);
                    }
                    let logits = model
                        .project_output_parallel(
                            &NumericTensor::new(vec![1, 2], vec![2., 3.]),
                            &parallel,
                            &context,
                        )
                        .unwrap();
                    assert_eq!(logits.shape(), [1, 5]);
                    assert_eq!(logits.data, [8.25, 5.5, -7.25, 9., 49.25]);
                    assert_eq!(take_acquisitions(&mut model), [0, 0, 0, 0, 4]);
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert!(world.tensor_collective_calls() > 0);
}

#[test]
fn voicechat_rnnt_prediction_state_preserves_local_carry_and_global_readout() {
    use eredu_architectures::voicechat::rnnt::RnntPredictionState;
    use eredu_nn::{recurrent::LstmState, RetainedTensorRoots};
    let fixture = fixture();
    let context = NumericContext::default();
    let mut model = provider(&fixture, false, &context);
    let primed = model.prime(&RnntState::default(), &context).unwrap();
    take_acquisitions(&mut model);
    let full = primed.predictor().unwrap();
    let dimensions = full.dimensions();
    assert_eq!(dimensions.predictor_layers, 2);
    let prefix = RnntPredictionState::new(
        dimensions,
        0..1,
        LstmState {
            layers: vec![full.local_state().layers[0].clone()],
        },
        Some(full.output().clone()),
    )
    .unwrap();
    let suffix = RnntPredictionState::new(
        dimensions,
        1..2,
        LstmState {
            layers: vec![full.local_state().layers[1].clone()],
        },
        None,
    )
    .unwrap();
    assert_eq!(prefix.output().data, full.output().data);
    assert_eq!(suffix.output().data, full.output().data);
    assert_ne!(
        prefix.local_state().layers[0].hidden.data,
        prefix.output().data
    );
    let roots = |state: &RnntPredictionState<NumericTensor>| {
        let mut values = Vec::new();
        state.visit_retained_tensors(&mut |value| values.push(value.data.clone()));
        values
    };
    assert_eq!(
        roots(&prefix),
        [
            full.local_state().layers[0].hidden.data.clone(),
            full.local_state().layers[0].cell.data.clone(),
            full.output().data.clone()
        ]
    );
    assert_eq!(
        roots(&suffix),
        [
            full.local_state().layers[1].hidden.data.clone(),
            full.local_state().layers[1].cell.data.clone()
        ]
    );
    assert_eq!(
        roots(full).len(),
        4,
        "complete ownership retains no duplicate readout root"
    );
    assert!(prefix.validate(dimensions, &(1..2)).is_err());
    assert!(
        RnntPredictionState::new(dimensions, 0..1, prefix.local_state().clone(), None).is_err()
    );
    assert!(RnntPredictionState::new(
        dimensions,
        1..2,
        suffix.local_state().clone(),
        Some(full.output().clone())
    )
    .is_err());
    let empty = RnntPredictionState::new(
        dimensions,
        0..0,
        LstmState { layers: vec![] },
        Some(full.output().clone()),
    )
    .unwrap();
    assert_eq!(roots(&empty), [full.output().data.clone()]);
    assert!(
        RnntPredictionState::new(
            dimensions,
            0..1,
            prefix.local_state().clone(),
            Some(NumericTensor::from_i32_slice(&[0, 0], &[1, 2], &context).unwrap()),
        )
        .is_err(),
        "integer restored readout must fail before execution"
    );
    let saved = roots(&prefix);
    assert!(model
        .predictor_step(
            &NumericTensor::zeros([1, dimensions.predictor_width]),
            Some(&prefix),
            &context
        )
        .is_err());
    assert!(
        model.runtime().policy().acquired.is_empty(),
        "wrong predictor ownership fails before a parameter lease"
    );
    assert_eq!(roots(&prefix), saved);
}
