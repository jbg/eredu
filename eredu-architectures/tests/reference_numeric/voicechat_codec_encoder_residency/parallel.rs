//! Finite tensor-parallel encoding retains the existing audio-to-code transaction.
use super::*;
use eredu_codec::ear_encoder_layered::{encoder_parameter_description, EncoderExecution};
use eredu_runtime::derive_partitioned_local_layout;

fn models(
    config: EncoderConfig,
    parameters: BTreeMap<String, NumericTensor>,
    layout: &LocalModelLayout,
    context: &NumericContext,
) -> (
    LayerwiseEncoder<NumericBackend, NumericReplicatedPolicy<Unit>>,
    LayerwiseEncoder<NumericBackend, BoundFixturePolicy>,
) {
    let parameters: BTreeMap<_, _> = parameters
        .into_iter()
        .filter_map(|(name, value)| {
            // The pinned encoder fixture also carries the borrowed PRVQ tables.
            let selected = select_parameter(&value, layout.tensor(&name)?.placement()).unwrap();
            Some((name, selected))
        })
        .collect();
    let architecture =
        || EncoderArchitecture::new_parallel(config.clone(), PREFIX, layout).unwrap();
    let units: Vec<_> = (0..config.unit_specs().unwrap().len())
        .map(|index| {
            let mut unit = architecture().build_unit(0, index, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            unit
        })
        .collect();
    assert!(
        NeuralEncoder::<NumericBackend>::from_units(config.clone(), units.clone()).is_err(),
        "rank-local units cannot be adopted as an ordinary encoder"
    );
    (
        LayerwiseEncoder::new(architecture(), NumericReplicatedPolicy::resident(units)),
        LayerwiseEncoder::new(
            architecture(),
            BoundFixturePolicy {
                inner: RebuildingUnitPolicy::default(),
                parameters,
            },
        ),
    )
}

struct ParallelEncoder<P: LayerwisePolicy<NumericBackend, Unit>> {
    inner: LayerwiseEncoder<NumericBackend, P>,
    parallel: NumericParallelContext,
}
impl<P> EncoderExecution<NumericBackend> for ParallelEncoder<P>
where
    P: LayerwisePolicy<NumericBackend, Unit>,
    P::Error: std::error::Error + Send + Sync + 'static,
{
    fn config(&self) -> &EncoderConfig {
        self.inner.config()
    }
    fn forward(
        &mut self,
        spectrum: &NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, eredu_codec::Error> {
        self.inner
            .forward_parallel(spectrum, &self.parallel, context)
    }
}

#[test]
fn voicechat_codec_encoder_tp2_matches_pinned_latents_codes_and_late_retry_without_quantizer_work()
{
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::CODEC_ENCODER_REFERENCE_JSON).unwrap();
    let case = &fixture["encoders"]["compact"];
    let config = config(&case["geometry"]);
    let count = config.neural.unit_specs().unwrap().len();
    let description = encoder_parameter_description(&config.neural, PREFIX).unwrap();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2).map(|rank| {
            let world = world.clone();
            let group = group.clone();
            let config = config.clone();
            let fixture = &fixture;
            let description = &description;
            scope.spawn(move || {
                let topology = ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank).unwrap();
                let layout = derive_partitioned_local_layout(description, topology).unwrap();
                let context = NumericContext::with_partition(layout.clone(), rank, world.clone());
                let parallel = NumericParallelContext::new_partitioned(rank, rank, group, world);
                let (resident, mut bounded) = models(config.neural.clone(), weights(&case["parameters"]), &layout, &context);
                let spectrum = tensor(&case["packed"]);
                for wrong in [
                    NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                    NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                ] {
                    assert!(bounded.forward_parallel(&spectrum, &wrong, &context).is_err());
                }
                assert!(bounded.forward(&spectrum, &context).is_err());
                assert!(bounded.forward_parallel(&NumericTensor::zeros([1, 3, 12]), &parallel, &context).is_err());
                assert!(bounded.runtime().policy().inner.events.is_empty());
                let mut resident = AudioEncoder::<NumericBackend, _>::from_parts(config.clone(), ParallelEncoder { inner: resident, parallel: parallel.clone() }, &context).unwrap();
                let mut bounded = AudioEncoder::<NumericBackend, _>::from_parts(config.clone(), ParallelEncoder { inner: bounded, parallel }, &context).unwrap();
                let mut quantizer = voicechat_codec_quantizer_residency::bounded(config.quantizer, weights(&case["parameters"]));
                let pcm = tensor(&case["pcm"]);
                for latent in [resident.latent(&pcm, &context).unwrap(), bounded.latent(&pcm, &context).unwrap()] {
                    compare(&latent, &case["latent"], fixture);
                }
                assert_eq!(bounded.neural.inner.runtime().policy().inner.events, events(count));
                for encoded in [resident.encode(&pcm, &mut quantizer, &context).unwrap(), bounded.encode(&pcm, &mut quantizer, &context).unwrap()] {
                    assert_eq!(encoded.data, codes(&case["codes"], &context).data);
                    compare(&quantizer.decode(&encoded, &context).unwrap(), &case["decoded"], fixture);
                }
                bounded.neural.inner.runtime_mut().policy_mut().inner.events.clear();
                quantizer.runtime_mut().policy_mut().inner.events.clear();
                bounded.neural.inner.runtime_mut().policy_mut().inner.fail_acquire = Some(count - 1);
                let error = bounded.encode(&pcm, &mut quantizer, &context).unwrap_err();
                assert!(error.to_string().contains("injected dense-stream acquisition failure"));
                assert_eq!(bounded.neural.inner.runtime().policy().inner.events, events(count)[..2 * count]);
                assert!(quantizer.runtime().policy().inner.events.is_empty(), "failed parallel latent construction cannot start quantization");
                bounded.neural.inner.runtime_mut().policy_mut().inner.fail_acquire = None;
                bounded.neural.inner.runtime_mut().policy_mut().inner.events.clear();
                let encoded = bounded.encode(&pcm, &mut quantizer, &context).unwrap();
                assert_eq!(encoded.data, codes(&case["codes"], &context).data);
                assert_eq!(bounded.neural.inner.runtime().policy().inner.events, events(count));
                let signal = pcm.axis_slice(0, 1, 2);
                let expected_signal = tensor(&case["latent"]).axis_slice(0, 1, 2);
                let actual_signal = bounded.latent(&signal, &context).unwrap();
                compare(&actual_signal, &serde_json::json!({"shape": expected_signal.shape, "values": expected_signal.data}), fixture);
                assert_eq!(bounded.latent(&signal, &context).unwrap().data, actual_signal.data,
                    "finite replay begins with empty convolution carry");
                bounded.neural.inner.runtime_mut().policy_mut().inner.events.clear();
                quantizer.runtime_mut().policy_mut().inner.events.clear();
                assert!(bounded.encode(&signal.axis_slice(2, 0, 47), &mut quantizer, &context).is_err());
                assert!(bounded.neural.inner.runtime().policy().inner.events.is_empty());
                assert!(quantizer.runtime().policy().inner.events.is_empty());
                assert_eq!(pcm.data, tensor(&case["pcm"]).data);
            })
        }).collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert!(world.tensor_collective_calls() > 0);
}

#[test]
fn voicechat_codec_encoder_uneven_grouped_tp2_preserves_rows_frame_order_and_single_bias() {
    // Input projects spectral channels 0 and 3 into separate convolution groups.
    // Three output rows per group split 2/1; all kernels and FF output biases differ.
    let config = EncoderConfig {
        latent_size: 6,
        base_hidden_size: 2,
        channel_mult: vec![1],
        rates: vec![2],
        blocks_per_stage: 1,
        kernel_size: 2,
        fft_size: 4,
        groups: 2,
    };
    let description = encoder_parameter_description(&config, PREFIX).unwrap();
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
        NumericTensor::new(
            [2, 6, 1],
            vec![1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        ),
    );
    parameters.insert(
        format!("{PREFIX}.layers.1.norm.weight"),
        NumericTensor::new([2], vec![1.; 2]),
    );
    parameters.insert(
        format!("{PREFIX}.layers.1.pwconv2.bias"),
        NumericTensor::new([2], vec![1., 10.]),
    );
    parameters.insert(
        format!("{PREFIX}.layers.2.weight"),
        NumericTensor::new([6, 1, 2], (1..=12).map(|n| n as f32).collect()),
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
                    let input = NumericTensor::new(
                        [1, 6, 4],
                        vec![
                            1., 2., 3., 4., 101., 102., 103., 104., 201., 202., 203., 204., 10.,
                            20., 30., 40., 301., 302., 303., 304., 401., 402., 403., 404.,
                        ],
                    );
                    for output in [
                        resident
                            .forward_parallel(&input, &parallel, &context)
                            .unwrap(),
                        bounded
                            .forward_parallel(&input, &parallel, &context)
                            .unwrap(),
                        bounded
                            .forward_parallel(&input, &parallel, &context)
                            .unwrap(),
                    ] {
                        assert_eq!(output.shape, [1, 2, 6]);
                        assert_eq!(
                            output.data,
                            [8., 18., 28., 380., 480., 580., 14., 32., 50., 680., 860., 1040.]
                        );
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
