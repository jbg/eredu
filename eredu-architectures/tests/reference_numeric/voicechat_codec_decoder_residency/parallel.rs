//! Decoder tensor partitions retain global causal carry and publish complete candidate frames.
use super::*;
use eredu_codec::ear_decoder_layered::decoder_parameter_description;
use eredu_runtime::derive_partitioned_local_layout;

fn models(
    config: DecoderConfig,
    parameters: BTreeMap<String, NumericTensor>,
    layout: &LocalModelLayout,
    context: &NumericContext,
) -> (
    LayerwiseDecoder<NumericBackend, NumericReplicatedPolicy<Unit>>,
    LayerwiseDecoder<NumericBackend, BoundFixturePolicy>,
) {
    let parameters: BTreeMap<_, _> = parameters
        .into_iter()
        .map(|(name, value)| {
            let selected =
                select_parameter(&value, layout.tensor(&name).unwrap().placement()).unwrap();
            (name, selected)
        })
        .collect();
    let architecture =
        || DecoderArchitecture::new_parallel(config.clone(), PREFIX, layout).unwrap();
    let units: Vec<_> = (0..config.unit_specs().unwrap().len())
        .map(|index| {
            let mut unit = architecture().build_unit(0, index, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            unit
        })
        .collect();
    assert!(
        NeuralDecoder::<NumericBackend>::from_units(config.clone(), units.clone()).is_err(),
        "rank-local units cannot be adopted as an ordinary decoder"
    );
    (
        LayerwiseDecoder::new(architecture(), NumericReplicatedPolicy::resident(units)),
        LayerwiseDecoder::new(
            architecture(),
            BoundFixturePolicy {
                inner: RebuildingUnitPolicy::default(),
                parameters,
            },
        ),
    )
}

fn same_state(actual: &DecoderState<NumericTensor>, expected: &DecoderState<NumericTensor>) {
    assert_eq!(actual.phase(), expected.phase());
    assert_eq!(actual.units().len(), expected.units().len());
    for (actual, expected) in actual.units().iter().zip(expected.units()) {
        match (actual.carry(), expected.carry()) {
            (Some(actual), Some(expected)) => {
                assert_eq!(actual.shape, expected.shape);
                assert_eq!(actual.data, expected.data);
            }
            (None, None) => {}
            _ => panic!("replicated decoder carry presence differs"),
        }
    }
}

#[test]
fn voicechat_codec_decoder_tp2_matches_pinned_streaming_carry_replay_and_late_retry() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_REFERENCE_JSON).unwrap();
    let case = &fixture["decoders"]["compact"];
    let config = geometry(&case["geometry"]);
    let count = config.unit_specs().unwrap().len();
    let description = decoder_parameter_description(&config, PREFIX).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let world = world.clone();
                let group = group.clone();
                let config = config.clone();
                let fixture = &fixture;
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
                    let (mut resident, mut bounded) =
                        models(config, weights(&case["parameters"]), &layout, &context);
                    let mut resident_state = resident.initial_state().unwrap();
                    let mut bounded_state = bounded.initial_state().unwrap();
                    let full = tensor(&case["full"][0]["input"]);
                    for wrong in [
                        NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                        NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                    ] {
                        assert!(bounded
                            .step_parallel(&full, &bounded_state, &wrong, &context)
                            .is_err());
                    }
                    assert!(bounded.step(&full, &bounded_state, &context).is_err());
                    assert!(bounded
                        .step_parallel(
                            &NumericTensor::zeros([1, 3, 1]),
                            &bounded_state,
                            &parallel,
                            &context
                        )
                        .is_err());
                    assert!(bounded.runtime().policy().inner.events.is_empty());
                    for output in [
                        resident
                            .step_parallel(&full, &resident_state, &parallel, &context)
                            .unwrap()
                            .0,
                        bounded
                            .step_parallel(&full, &bounded_state, &parallel, &context)
                            .unwrap()
                            .0,
                    ] {
                        compare(&output, &case["full"][0]["raw"], fixture);
                    }
                    let steps = case["streaming"].as_array().unwrap();
                    for (index, step) in steps.iter().enumerate() {
                        let input = tensor(&step["input"]);
                        bounded.runtime_mut().policy_mut().inner.events.clear();
                        let parent = bounded_state.clone();
                        if index == 1 {
                            bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(count - 1);
                            let error = bounded
                                .step_parallel(&input, &parent, &parallel, &context)
                                .unwrap_err();
                            assert!(error
                                .to_string()
                                .contains("injected dense-stream acquisition failure"));
                            compare_neural(&parent, &steps[0]["cache"], fixture);
                            assert_eq!(
                                bounded.runtime().policy().inner.events,
                                events(count)[..2 * count]
                            );
                            bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
                            bounded.runtime_mut().policy_mut().inner.events.clear();
                        }
                        let (leased, leased_state) = resident
                            .step_parallel(&input, &resident_state, &parallel, &context)
                            .unwrap();
                        let (streamed, streamed_state) = bounded
                            .step_parallel(&input, &parent, &parallel, &context)
                            .unwrap();
                        compare(&leased, &step["raw"], fixture);
                        compare(&streamed, &step["raw"], fixture);
                        assert_eq!(bounded.runtime().policy().inner.events, events(count));
                        if !step["flush"].as_bool().unwrap() {
                            compare_neural(&leased_state, &step["cache"], fixture);
                            compare_neural(&streamed_state, &step["cache"], fixture);
                        }
                        same_state(&leased_state, &streamed_state);
                        same_state(&parent, &bounded_state);
                        if index == 1 {
                            let (replayed, fork) = bounded
                                .step_parallel(&input, &parent, &parallel, &context)
                                .unwrap();
                            compare(&replayed, &step["raw"], fixture);
                            same_state(&fork, &streamed_state);
                            same_state(&parent, &bounded_state);
                        }
                        resident_state = leased_state;
                        bounded_state = streamed_state;
                    }
                    bounded.runtime_mut().policy_mut().inner.events.clear();
                    let finished = bounded.finish(&bounded_state).unwrap();
                    assert_eq!(finished.phase(), StreamPhase::Finished);
                    assert!(finished.units().iter().all(|unit| unit.carry().is_none()));
                    assert!(bounded
                        .step_parallel(&full, &finished, &parallel, &context)
                        .is_err());
                    assert!(bounded.runtime().policy().inner.events.is_empty());
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
fn voicechat_codec_decoder_uneven_grouped_tp2_preserves_channel_order_and_single_bias() {
    // Three outputs per convolution group split 2/1 across ranks. A zero MLP
    // with nonzero output bias exposes bias replication independently of GELU.
    let config = DecoderConfig {
        latent_size: 2,
        base_hidden_size: 6,
        channel_mult: vec![1],
        rates: vec![2],
        blocks_per_stage: 1,
        kernel_size: 2,
        fft_size: 4,
        groups: 2,
    };
    let description = decoder_parameter_description(&config, PREFIX).unwrap();
    let mut parameters = BTreeMap::new();
    for group in description.groups() {
        for member in group.members() {
            parameters.insert(
                member.target().to_owned(),
                NumericTensor::zeros(
                    member
                        .global_shape()
                        .iter()
                        .map(|n| *n as i32)
                        .collect::<Vec<_>>(),
                ),
            );
        }
    }
    parameters.insert(
        format!("{PREFIX}.layers.0.weight"),
        NumericTensor::new([2, 3, 2], (1..=12).map(|n| n as f32).collect()),
    );
    parameters.insert(
        format!("{PREFIX}.layers.1.norm.weight"),
        NumericTensor::new([6], vec![1.; 6]),
    );
    parameters.insert(
        format!("{PREFIX}.layers.1.pwconv2.bias"),
        NumericTensor::new([6], vec![1., 2., 3., 4., 5., 6.]),
    );
    parameters.insert(
        format!("{PREFIX}.layers.2.weight"),
        NumericTensor::new(
            [6, 6, 1],
            (0..36)
                .map(|i| if i / 6 == i % 6 { 1. } else { 0. })
                .collect(),
        ),
    );
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let world = world.clone();
                let group = group.clone();
                let config = config.clone();
                let parameters = parameters.clone();
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
                    let (mut resident, mut bounded) = models(config, parameters, &layout, &context);
                    let mut resident_state = resident.initial_state().unwrap();
                    let mut bounded_state = bounded.initial_state().unwrap();
                    for (input, expected, carry) in [
                        (
                            vec![1., 10.],
                            vec![2., 3., 5., 6., 8., 9., 74., 84., 95., 105., 116., 126.],
                            vec![2., 4., 6., 80., 100., 120.],
                        ),
                        (
                            vec![2., 20.],
                            vec![
                                3., 5., 8., 10., 13., 15., 144., 164., 185., 205., 226., 246.,
                            ],
                            vec![4., 8., 12., 160., 200., 240.],
                        ),
                    ] {
                        let input = NumericTensor::new([1, 1, 2], input);
                        let (leased, leased_state) = resident
                            .step_parallel(&input, &resident_state, &parallel, &context)
                            .unwrap();
                        let (streamed, streamed_state) = bounded
                            .step_parallel(&input, &bounded_state, &parallel, &context)
                            .unwrap();
                        for output in [leased, streamed] {
                            assert_eq!(output.shape, [1, 6, 2]);
                            assert_eq!(output.data, expected);
                        }
                        for state in [&leased_state, &streamed_state] {
                            let actual = state.units()[1].carry().unwrap();
                            assert_eq!(actual.shape, [1, 6, 1]);
                            assert_eq!(actual.data, carry);
                        }
                        resident_state = leased_state;
                        bounded_state = streamed_state;
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert!(world.tensor_collective_calls() > 0);
}
