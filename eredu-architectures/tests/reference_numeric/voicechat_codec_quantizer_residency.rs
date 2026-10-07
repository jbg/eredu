//! Pinned PRVQ residual equations through independent codebook leases.
use super::voicechat_conditioning::compare as prvq_compare;
use super::voicechat_mog_residency::weights;
use super::voicechat_tts_generator::{codes, compare, tensor};
use super::*;
use eredu_codec::{
    prvq::{Codebook, DistanceMetric, Geometry, Quantized, ResidualQuantizer},
    prvq_layered::{LayerwiseQuantizer, QuantizerArchitecture, QuantizerExecution},
};
use serde_json::Value;
use std::ops::Range;

const PREFIX: &str = "tts_model.audio_codec.prvq";
type Unit = Codebook<NumericTensor>;
pub(super) struct BoundFixturePolicy {
    pub(super) inner: RebuildingUnitPolicy,
    pub(super) parameters: BTreeMap<String, NumericTensor>,
    pub(super) aborted: usize,
}
impl LayerwisePolicy<NumericBackend, Unit> for BoundFixturePolicy {
    type Lease = RebuiltUnitLease<Unit>;
    type Error = Error;
    fn abort(
        &mut self,
        active: Option<(usize, ExecutionUnitAddress, Self::Lease)>,
        _: &NumericContext,
    ) {
        self.aborted += 1;
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
        self.inner.acquire(
            ordinal,
            address,
            |context| {
                let mut unit = build(context)?;
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
    geometry: Geometry,
    parameters: BTreeMap<String, NumericTensor>,
) -> LayerwiseQuantizer<NumericBackend, BoundFixturePolicy> {
    LayerwiseQuantizer::new(
        QuantizerArchitecture::new(geometry, PREFIX).unwrap(),
        BoundFixturePolicy {
            inner: RebuildingUnitPolicy::default(),
            aborted: 0,
            parameters,
        },
    )
}
fn resident(
    geometry: Geometry,
    parameters: &Value,
    context: &NumericContext,
) -> LayerwiseQuantizer<NumericBackend, NumericReplicatedPolicy<Unit>> {
    let architecture = QuantizerArchitecture::new(geometry, PREFIX).unwrap();
    let parameters = weights(parameters);
    let units = (0..geometry.depth)
        .map(|depth| {
            let mut unit = architecture.build_unit(depth, 0, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            unit
        })
        .collect();
    LayerwiseQuantizer::new(architecture, NumericReplicatedPolicy::resident(units))
}
fn events(range: Range<usize>) -> Vec<StreamPolicyEvent> {
    let mut events = vec![StreamPolicyEvent::Begin];
    for depth in range {
        events.extend([
            StreamPolicyEvent::Acquire(depth, depth, 0),
            StreamPolicyEvent::Complete(depth, depth, 0),
        ]);
    }
    events.push(StreamPolicyEvent::Finish);
    events
}
fn compare_quantized(result: &Quantized<NumericTensor>, fixture: &Value, context: &NumericContext) {
    assert_eq!(result.codes.data, codes(&fixture["codes"], context).data);
    prvq_compare(&result.quantized, &fixture["quantized"], fixture);
    prvq_compare(&result.residual, &fixture["residual"], fixture);
}

#[test]
fn voicechat_codec_quantizer_resident_and_bounded_match_pinned_residuals_and_selected_prefixes() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let geometry = Geometry {
        depth: 3,
        cardinality: 5,
        channels: 4,
    };
    let mut ordinary =
        ResidualQuantizer::<NumericBackend>::new(geometry, PREFIX, &context).unwrap();
    replace_numeric_parameters(&mut ordinary, &weights(&fixture["parameters"]));
    let mut resident = resident(geometry, &fixture["parameters"], &context);
    let mut bounded = bounded(geometry, weights(&fixture["parameters"]));
    let latent = tensor(&fixture["input"]);
    for result in [
        ordinary.quantize(&latent, &context).unwrap(),
        resident.quantize(&latent, &context).unwrap(),
        bounded.quantize(&latent, &context).unwrap(),
    ] {
        compare_quantized(&result, &fixture, &context);
    }
    assert_eq!(bounded.runtime().policy().inner.events, events(0..3));
    let recorded = codes(&fixture["codes"], &context);
    for decoded in [
        ordinary.decode(&recorded, &context).unwrap(),
        resident.decode(&recorded, &context).unwrap(),
        bounded.decode(&recorded, &context).unwrap(),
    ] {
        prvq_compare(&decoded, &fixture["decoded"], &fixture);
    }
    bounded.runtime_mut().policy_mut().inner.events.clear();
    let prefix = recorded.axis_slice(2, 0, 2);
    prvq_compare(
        &bounded.decode_range(&prefix, 0..2, &context).unwrap(),
        &fixture["prefix_decoded"],
        &fixture,
    );
    assert_eq!(bounded.runtime().policy().inner.events, events(0..2));
    bounded.runtime_mut().policy_mut().inner.events.clear();
    bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(2);
    let error = bounded.quantize(&latent, &context).unwrap_err();
    assert!(error
        .to_string()
        .contains("injected dense-stream acquisition failure"));
    assert_eq!(bounded.runtime().policy().inner.events, events(0..3)[..6]);
    assert_eq!(latent.data, tensor(&fixture["input"]).data);
    assert_eq!(recorded.data, codes(&fixture["codes"], &context).data);
    bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
    bounded.runtime_mut().policy_mut().inner.events.clear();
    compare_quantized(
        &bounded.quantize(&latent, &context).unwrap(),
        &fixture,
        &context,
    );
    assert_eq!(bounded.runtime().policy().inner.events, events(0..3));
}

#[test]
fn voicechat_codec_quantizer_bounded_suffix_starts_from_fresh_residual_and_skips_prefix_codebook() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::MOG_REFERENCE_JSON).unwrap();
    let context = NumericContext::default();
    let case = &fixture["depthsum_cases"];
    let tables = tensor(&case["embeddings"]);
    let mut parameters = BTreeMap::new();
    for depth in 0..3 {
        parameters.insert(
            format!("{PREFIX}.mus_list.{depth}"),
            tables
                .axis_slice(0, depth, depth + 1)
                .reshape(&[4, 2], &context)
                .unwrap(),
        );
        parameters.insert(
            format!("{PREFIX}._variance_list.{depth}.variance"),
            NumericTensor::new(vec![], vec![0.1]),
        );
    }
    let mut bounded = bounded(
        Geometry {
            depth: 3,
            cardinality: 4,
            channels: 2,
        },
        parameters,
    );
    let residual = tensor(&case["residual"]);
    let metric = DistanceMetric::CodebookNormMinusTwiceDot;
    let full = bounded
        .quantize_range(&residual, 0..3, metric, &context)
        .unwrap();
    assert_eq!(full.codes.data, codes(&case["full_codes"], &context).data);
    compare(
        &bounded.decode(&full.codes, &context).unwrap(),
        &case["full_sum"],
        &fixture,
    );
    bounded.runtime_mut().policy_mut().inner.events.clear();
    let suffix = bounded
        .quantize_range(&residual, 1..3, metric, &context)
        .unwrap();
    let recorded = codes(&case["suffix_codes"], &context);
    assert_eq!(suffix.codes.data, recorded.axis_slice(2, 1, 3).data);
    assert_eq!(bounded.runtime().policy().inner.events, events(1..3));
    bounded.runtime_mut().policy_mut().inner.events.clear();
    let decoded_suffix = bounded.decode_range(&suffix.codes, 1..3, &context).unwrap();
    assert_eq!(bounded.runtime().policy().inner.events, events(1..3));
    // NVIDIA's suffix_sum includes its retained, independently supplied first
    // depth. Reattach that prefix to compare the recorded ordered decode.
    let prefix = recorded.axis_slice(2, 0, 1);
    let decoded_prefix = bounded.decode_range(&prefix, 0..1, &context).unwrap();
    compare(
        &decoded_prefix.add(&decoded_suffix, &context).unwrap(),
        &case["suffix_sum"],
        &fixture,
    );
    assert_eq!(residual.data, tensor(&case["residual"]).data);
}

fn parallel_models(
    geometry: Geometry,
    parameters: BTreeMap<String, NumericTensor>,
    layout: &LocalModelLayout,
    context: &NumericContext,
) -> (
    LayerwiseQuantizer<NumericBackend, NumericReplicatedPolicy<Unit>>,
    LayerwiseQuantizer<NumericBackend, BoundFixturePolicy>,
) {
    let parameters: BTreeMap<_, _> = parameters
        .into_iter()
        .map(|(name, value)| {
            let tensor = layout.tensor(&name).unwrap();
            let selected = select_parameter(&value, tensor.placement()).unwrap();
            (name, selected)
        })
        .collect();
    let architecture = || QuantizerArchitecture::new_parallel(geometry, PREFIX, layout).unwrap();
    let units = (0..geometry.depth)
        .map(|depth| {
            let mut unit = architecture().build_unit(depth, 0, context).unwrap();
            replace_numeric_parameters(&mut unit, &parameters);
            unit
        })
        .collect();
    (
        LayerwiseQuantizer::new(architecture(), NumericReplicatedPolicy::resident(units)),
        LayerwiseQuantizer::new(
            architecture(),
            BoundFixturePolicy {
                inner: RebuildingUnitPolicy::default(),
                aborted: 0,
                parameters,
            },
        ),
    )
}

#[test]
fn voicechat_codec_quantizer_tp2_matches_pinned_residuals_prefixes_and_retry() {
    use eredu_codec::prvq_layered::quantizer_parameter_description;
    use eredu_runtime::derive_partitioned_local_layout;
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::PRVQ_REFERENCE_JSON).unwrap();
    let geometry = Geometry {
        depth: 3,
        cardinality: 5,
        channels: 4,
    };
    let description = quantizer_parameter_description(geometry, PREFIX).unwrap();
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
                    let topology =
                        ParallelRankTopology::new(ParallelTopology::new(2, 1, 1, 1).unwrap(), rank)
                            .unwrap();
                    let layout = derive_partitioned_local_layout(description, topology).unwrap();
                    let context =
                        NumericContext::with_partition(layout.clone(), rank, world.clone());
                    let parallel =
                        NumericParallelContext::new_partitioned(rank, rank, group, world);
                    let (mut resident, mut bounded) = parallel_models(
                        geometry,
                        weights(&fixture["parameters"]),
                        &layout,
                        &context,
                    );
                    let latent = tensor(&fixture["input"]);
                    let recorded = codes(&fixture["codes"], &context);
                    for wrong in [
                        NumericParallelContext::new(0, NumericParallelGroup::new(1)),
                        NumericParallelContext::new(1 - rank, NumericParallelGroup::new(2)),
                    ] {
                        assert!(bounded
                            .quantize_range_parallel(
                                &latent,
                                0..3,
                                DistanceMetric::SquaredEuclidean,
                                &wrong,
                                &context
                            )
                            .is_err());
                        assert!(bounded
                            .decode_range_parallel(&recorded, 0..3, &wrong, &context)
                            .is_err());
                    }
                    assert!(bounded.quantize(&latent, &context).is_err());
                    assert!(bounded.decode(&recorded, &context).is_err());
                    assert!(bounded
                        .quantize_range_parallel(
                            &NumericTensor::zeros([1, 1, 3]),
                            0..3,
                            DistanceMetric::SquaredEuclidean,
                            &parallel,
                            &context
                        )
                        .is_err());
                    assert!(bounded
                        .decode_range_parallel(&recorded, 1..3, &parallel, &context)
                        .is_err());
                    assert!(bounded.runtime().policy().inner.events.is_empty());
                    for result in [
                        resident
                            .quantize_range_parallel(
                                &latent,
                                0..3,
                                DistanceMetric::SquaredEuclidean,
                                &parallel,
                                &context,
                            )
                            .unwrap(),
                        bounded
                            .quantize_range_parallel(
                                &latent,
                                0..3,
                                DistanceMetric::SquaredEuclidean,
                                &parallel,
                                &context,
                            )
                            .unwrap(),
                    ] {
                        compare_quantized(&result, fixture, &context);
                    }
                    for decoded in [
                        resident
                            .decode_range_parallel(&recorded, 0..3, &parallel, &context)
                            .unwrap(),
                        bounded
                            .decode_range_parallel(&recorded, 0..3, &parallel, &context)
                            .unwrap(),
                    ] {
                        prvq_compare(&decoded, &fixture["decoded"], fixture);
                    }
                    bounded.runtime_mut().policy_mut().inner.events.clear();
                    let prefix = recorded.axis_slice(2, 0, 2);
                    prvq_compare(
                        &bounded
                            .decode_range_parallel(&prefix, 0..2, &parallel, &context)
                            .unwrap(),
                        &fixture["prefix_decoded"],
                        fixture,
                    );
                    assert_eq!(bounded.runtime().policy().inner.events, events(0..2));
                    bounded.runtime_mut().policy_mut().inner.events.clear();
                    // Fail after two completed collective decisions; retry retains the original ingress.
                    bounded.runtime_mut().policy_mut().inner.fail_acquire = Some(2);
                    let error = bounded
                        .quantize_range_parallel(
                            &latent,
                            0..3,
                            DistanceMetric::SquaredEuclidean,
                            &parallel,
                            &context,
                        )
                        .unwrap_err();
                    assert!(error
                        .to_string()
                        .contains("injected dense-stream acquisition failure"));
                    assert_eq!(bounded.runtime().policy().inner.events, events(0..3)[..6]);
                    assert_eq!(latent.data, tensor(&fixture["input"]).data);
                    bounded.runtime_mut().policy_mut().inner.fail_acquire = None;
                    bounded.runtime_mut().policy_mut().inner.events.clear();
                    compare_quantized(
                        &bounded
                            .quantize_range_parallel(
                                &latent,
                                0..3,
                                DistanceMetric::SquaredEuclidean,
                                &parallel,
                                &context,
                            )
                            .unwrap(),
                        fixture,
                        &context,
                    );
                    assert_eq!(bounded.runtime().policy().inner.events, events(0..3));
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
fn voicechat_codec_quantizer_uneven_tp2_reduces_global_scores_preserves_ties_and_suffix_residual() {
    use eredu_codec::prvq_layered::quantizer_parameter_description;
    use eredu_runtime::derive_partitioned_local_layout;
    let geometry = Geometry {
        depth: 3,
        cardinality: 3,
        channels: 5,
    };
    let description = quantizer_parameter_description(geometry, PREFIX).unwrap();
    // With zero input each shard prefers a different mean (0 and 1), while full
    // squared distances [16, 9, 8] select 2. Depth 1 then ties means 0 and 1.
    let means = [
        vec![0., 0., 0., 4., 0., 3., 0., 0., 0., 0., 2., 0., 0., 2., 0.],
        vec![-1., 0., 0., 1., 0., 1., 0., 0., -1., 0., 2., 0., 0., 2., 0.],
        vec![0.; 15],
    ];
    let parameters: BTreeMap<_, _> = means
        .into_iter()
        .enumerate()
        .flat_map(|(depth, data)| {
            [
                (
                    format!("{PREFIX}.mus_list.{depth}"),
                    NumericTensor::new([3, 5], data),
                ),
                (
                    format!("{PREFIX}._variance_list.{depth}.variance"),
                    NumericTensor::new(vec![], vec![0.1]),
                ),
            ]
        })
        .collect();
    let world = Arc::new(NumericPartitionWorld::default());
    let group = NumericParallelGroup::new(2);
    std::thread::scope(|scope| {
        let workers = (0..2)
            .map(|rank| {
                let world = world.clone();
                let group = group.clone();
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
                    let (_, mut bounded) = parallel_models(geometry, parameters, &layout, &context);
                    let zero = NumericTensor::zeros([1, 1, 5]);
                    for metric in [
                        DistanceMetric::SquaredEuclidean,
                        DistanceMetric::CodebookNormMinusTwiceDot,
                    ] {
                        let result = bounded
                            .quantize_range_parallel(&zero, 0..3, metric, &parallel, &context)
                            .unwrap();
                        assert_eq!(result.codes.data, vec![2., 0., 0.]);
                        assert_eq!(result.residual.data, vec![-1., 0., 0., -3., 0.]);
                        assert_eq!(result.quantized.data, vec![1., 0., 0., 3., 0.]);
                        assert_eq!(
                            bounded
                                .decode_range_parallel(&result.codes, 0..3, &parallel, &context)
                                .unwrap()
                                .data,
                            vec![1., 0., 0., 3., 0.]
                        );
                        bounded.runtime_mut().policy_mut().inner.events.clear();
                        let suffix = bounded
                            .quantize_range_parallel(&zero, 1..3, metric, &parallel, &context)
                            .unwrap();
                        assert_eq!(suffix.codes.data, vec![0., 0.]);
                        assert_eq!(suffix.residual.data, vec![1., 0., 0., -1., 0.]);
                        assert_eq!(suffix.quantized.data, vec![-1., 0., 0., 1., 0.]);
                        assert_eq!(bounded.runtime().policy().inner.events, events(1..3));
                        assert_eq!(
                            bounded
                                .decode_range_parallel(&suffix.codes, 1..3, &parallel, &context)
                                .unwrap()
                                .data,
                            vec![-1., 0., 0., 1., 0.]
                        );
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
}
