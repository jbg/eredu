//! Ordinary and admitted execution consume the same selector and its quote.
use super::*;
use crate::backend::{
    managed_memory::gpu_stream::PreparedExecutionStreams,
    nn::{
        grouped::{
            GroupSelectionOutput, TopKGroupScoring, TopKGroupSelector, TopKGroupSelectorConfig,
        },
        shared::MlxNeuralBackend,
    },
    MlxBackend, MlxDeviceIdentity,
};
use safemlx::{
    Array, Device, DeviceType, OriginalBufferBudget, OriginalScopeObserver, PrefillRoots,
    PrefillRootsRuntime, PreparedOriginalBufferBudget, PreparedPrefillFailure,
    PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
    SubmissionScope,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
#[derive(Debug)]
struct Lifetime(Arc<AtomicBool>);
impl Drop for Lifetime {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
fn values(output: &GroupSelectionOutput) -> (Vec<i32>, Vec<f32>, Vec<f32>) {
    output.indices.evaluated().unwrap();
    let ids = if output.indices.dtype() == Dtype::Int32 {
        output
            .indices
            .evaluated()
            .unwrap()
            .try_to_vec::<i32>()
            .unwrap()
    } else {
        output
            .indices
            .evaluated()
            .unwrap()
            .try_to_vec::<u32>()
            .unwrap()
            .into_iter()
            .map(|n| n as i32)
            .collect()
    };
    (
        ids,
        output
            .scores
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        output
            .weights
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
    )
}
fn numeric(
    actual: &(Vec<i32>, Vec<f32>, Vec<f32>),
    hidden: &[f32],
    weight: &[f32],
    scoring: GroupScoring,
    extended: bool,
    supplied: bool,
) {
    let bias = [0.125f64, -0.2, 0.3];
    let correction = [0.45f64, -0.25, 0.1];
    for row in 0..2 {
        let input = &hidden[row * 8..row * 8 + 8];
        let denominator =
            (input.iter().map(|&n| f64::from(n).powi(2)).sum::<f64>() / 8.0 + 1e-5).sqrt();
        let logits = (0..3)
            .map(|g| {
                input
                    .iter()
                    .enumerate()
                    .map(|(c, &n)| {
                        let x = if extended {
                            f64::from(n) / denominator * (0.75 + c as f64 / 16.0) / 8f64.sqrt()
                        } else {
                            f64::from(n)
                        };
                        x * f64::from(weight[g * 8 + c])
                    })
                    .sum::<f64>()
                    + if extended { bias[g] } else { 0.0 }
            })
            .collect::<Vec<_>>();
        let mut scores = logits
            .iter()
            .map(|&x| match scoring {
                GroupScoring::Sigmoid => 1.0 / (1.0 + (-x).exp()),
                GroupScoring::SqrtSoftplus => x.exp().ln_1p().sqrt(),
                GroupScoring::Softmax => x.exp(),
                _ => x,
            })
            .collect::<Vec<_>>();
        if scoring == GroupScoring::Softmax {
            let total = scores.iter().sum::<f64>();
            for x in &mut scores {
                *x /= total;
            }
        }
        let ids = &actual.0[row * 2..row * 2 + 2];
        if supplied {
            assert_eq!(ids, &[[2, 0], [1, 2]][row]);
        } else {
            let mut expected = [0, 1, 2];
            expected.sort_by(|&a, &b| {
                (scores[b] + correction[b]).total_cmp(&(scores[a] + correction[a]))
            });
            let mut chosen = ids.iter().map(|&n| n as usize).collect::<Vec<_>>();
            chosen.sort();
            let mut expected = expected[..2].to_vec();
            expected.sort();
            assert_eq!(chosen, expected);
        }
        let mut chosen = ids
            .iter()
            .map(|&id| scores[id as usize])
            .collect::<Vec<_>>();
        if scoring == GroupScoring::SelectedSoftmax {
            let z = chosen.iter().map(|x| x.exp()).sum::<f64>();
            for x in &mut chosen {
                *x = x.exp() / z;
            }
        }
        let total = chosen.iter().sum::<f64>() + 1e-6;
        for (slot, (&id, &score)) in ids.iter().zip(&chosen).enumerate() {
            let coef = score / total
                * 1.7
                * if extended {
                    [0.8f64, 1.2, 0.9][id as usize]
                } else {
                    1.0
                };
            for (value, reference) in [
                (actual.1[row * 2 + slot], score),
                (actual.2[row * 2 + slot], coef),
            ] {
                assert!(
                    (f64::from(value) - reference).abs() < 3e-5 * reference.abs().max(1.0),
                    "{scoring:?}, extended={extended}, supplied={supplied}: {value} != {reference}"
                );
            }
        }
    }
}
#[test]
#[ignore = "requires qualified original CPU streams and native allocator"]
fn original_cpu_selector_preserves_ranking_transforms_and_escaped_custody() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    let backend = MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    );
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    let runtime = PrefillRootsRuntime::prepare_for_stream(stream, stream).unwrap();
    let allocator = environment.input_runtime().unwrap();
    let (ordinary, cpu) = mechanisms();
    for (scoring, native_scoring) in [
        (GroupScoring::Softmax, TopKGroupScoring::Softmax),
        (
            GroupScoring::SelectedSoftmax,
            TopKGroupScoring::SelectedSoftmax,
        ),
        (GroupScoring::Sigmoid, TopKGroupScoring::Sigmoid),
        (GroupScoring::SqrtSoftplus, TopKGroupScoring::SqrtSoftplus),
    ] {
        for extended in [false, true] {
            for supplied in [false, true] {
                let specification = if extended {
                    extended_spec(scoring)
                } else {
                    spec(scoring, 2, true, 1.7)
                };
                let config = TopKGroupSelectorConfig::new(
                    2,
                    3,
                    8,
                    native_scoring,
                    true,
                    1e-6,
                    1.7,
                    1,
                    1,
                    extended,
                    true,
                    extended.then_some(1e-5),
                    extended,
                    extended,
                )
                .unwrap()
                .with_arithmetic(specification.arithmetic());
                let mut selector =
                    TopKGroupSelector::new_with_quantization(config, None, stream).unwrap();
                let hidden = (0..16)
                    .map(|n| ((n * 7) % 19) as f32 / 8.0 - 1.0)
                    .collect::<Vec<_>>();
                let weight = (0..24)
                    .map(|n| ((n * 5) % 17) as f32 / 16.0 - 0.5)
                    .collect::<Vec<_>>();
                let input = Array::from_slice(&hidden, &[1, 2, 8]);
                let ids = Array::from_slice(&[2i32, 0, 1, 2], &[2, 2]);
                selector.weight.value = Array::from_slice(&weight, &[3, 8]);
                selector.e_score_correction_bias.value =
                    Some(Array::from_slice(&[0.45f32, -0.25, 0.1], &[3]));
                if extended {
                    selector.bias.value = Some(Array::from_slice(&[0.125f32, -0.2, 0.3], &[3]));
                    selector.input_scale.value = Some(Array::from_slice(
                        &(0..8).map(|n| 0.75 + n as f32 / 16.0).collect::<Vec<_>>(),
                        &[8],
                    ));
                    selector.learned_coefficient_scale.value =
                        Some(Array::from_slice(&[0.8f32, 1.2, 0.9], &[3]));
                }
                let run = |selector: &mut TopKGroupSelector| {
                    if supplied {
                        selector.select_indices(&input, &ids, stream)
                    } else {
                        selector.select_with_selection_bias(&input, None, stream)
                    }
                };
                let ordinary_output = run(&mut selector).unwrap();
                let expected = values(&ordinary_output);
                numeric(&expected, &hidden, &weight, scoring, extended, supplied);
                drop(ordinary_output);
                let leaves = [
                    Some(&input),
                    Some(&ids),
                    Some(&selector.weight.value),
                    selector.e_score_correction_bias.value.as_ref(),
                    selector.bias.value.as_ref(),
                    selector.input_scale.value.as_ref(),
                    selector.learned_coefficient_scale.value.as_ref(),
                ];
                for leaf in leaves.into_iter().flatten() {
                    leaf.evaluated().unwrap();
                }
                let (context, report) =
                    trace(specification, 2, supplied, WorkspaceDtype::Int32, cpu);
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_outputs(
                    &report, 3, ordinary, cpu, &context,
                )
                .unwrap();
                let completion = recipe.completion;
                let physical = OriginalBufferBudget::population_layout(
                    &allocator,
                    recipe.storage.mutable_bytes().try_into().unwrap(),
                    recipe.storage.maximum_births(),
                )
                .unwrap()
                .capacity();
                let released = Arc::new(AtomicBool::new(false));
                let owner = Arc::new(Lifetime(released.clone()));
                let graph =
                    PreparedSubmissionGraphQuota::try_new(recipe.graph_capacity, owner.clone())
                        .unwrap()
                        .try_allocate()
                        .unwrap();
                let records =
                    PreparedSubmissionRecordQuota::try_new(recipe.record_capacity, owner.clone())
                        .unwrap()
                        .try_allocate()
                        .unwrap();
                let budget =
                    PreparedOriginalBufferBudget::try_new(&allocator, physical, owner.clone())
                        .unwrap()
                        .try_allocate()
                        .unwrap();
                let failure = PreparedPrefillFailure::try_new(owner.clone())
                    .unwrap()
                    .try_allocate()
                    .unwrap();
                let mut roots = PrefillRoots::new_retained(&runtime, 3, &graph, &failure).unwrap();
                let mut scope = SubmissionScope::try_begin_retaining(
                    PreparedSubmissionScopeOwner::try_new(owner.clone())
                        .unwrap()
                        .with_graph_quota(graph.clone())
                        .with_record_quota(records.clone()),
                )
                .unwrap();
                scope.enable_scoped_observation().unwrap();
                scope.require_original_native_controls().unwrap();
                roots.bind_scope(&scope).unwrap();
                scope.enable_original_native_controls().unwrap();
                scope.bind_original_buffer_budget(&budget).unwrap();
                let observer = OriginalScopeObserver::require_current().unwrap();
                for leaf in leaves.into_iter().flatten() {
                    OperationEvent::validate_traversal_leaf(leaf, &observer).unwrap();
                }
                let bank =
                    OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
                let output = run(&mut selector).unwrap();
                drop(bank);
                for value in [&output.indices, &output.scores, &output.weights] {
                    roots.append(value).unwrap();
                }
                roots
                    .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
                    .unwrap_or_else(|e| {
                        panic!("{scoring:?}, extended={extended}, supplied={supplied}: {e:?}")
                    });
                assert!(budget.occupied_bytes() > 0 && budget.occupied_bytes() <= physical);
                if supplied {
                    assert_eq!(
                        output.indices.allocation_info().unwrap(),
                        ids.allocation_info().unwrap()
                    );
                } else {
                    assert!(
                        output.indices.allocation_info().unwrap().unwrap().bytes() >= 2 * 3 * 4
                    );
                }
                scope.seal();
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    assert!(std::time::Instant::now() < deadline);
                    let (progress, status) = observer.progress().unwrap();
                    assert_eq!(progress, safemlx::ScopedSubmissionProgress::Observed);
                    assert!(!status.failed() && !status.blocked());
                    status.is_settled()
                });
                assert_eq!(
                    observer.retire_completed_records().unwrap(),
                    safemlx::SubmissionRetirement::CompleteSnapshot
                );
                safemlx::try_with_submission_retirement(|| {
                    drop((roots, scope, observer, failure, records, graph, budget))
                })
                .unwrap();
                let actual = values(&output);
                assert_eq!(actual, expected);
                numeric(&actual, &hidden, &weight, scoring, extended, supplied);
                let GroupSelectionOutput {
                    indices,
                    scores,
                    weights,
                } = output;
                drop((indices, scores));
                drop(owner);
                safemlx::reclaim_allocation_owners();
                assert!(!released.load(Ordering::SeqCst));
                assert_eq!(
                    weights.evaluated().unwrap().try_to_vec::<f32>().unwrap(),
                    expected.2
                );
                drop(weights);
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    assert!(std::time::Instant::now() < deadline);
                    safemlx::try_retire_completed_submissions().unwrap();
                    MlxNeuralBackend::reclaim_retired_resources();
                    safemlx::reclaim_allocation_owners();
                    released.load(Ordering::SeqCst)
                });
            }
        }
    }
}
