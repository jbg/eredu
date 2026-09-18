//! Joint routing's actual CPU graph, quoted by the same cold operation source.
use super::*;
use crate::{
    MlxTensor,
    backend::{
        MlxBackend, MlxDeviceIdentity, managed_memory::gpu_stream::PreparedExecutionStreams,
        nn::shared::MlxNeuralBackend,
    },
};
use eredu_nn::{JointGroupSelection, JointGroupSelectionInput, JointGroupSelectionSpec};
use safemlx::{
    Array, Device, DeviceType, OriginalBufferBudget, OriginalScopeObserver, PrefillRoots,
    PrefillRootsRuntime, PreparedOriginalBufferBudget, PreparedPrefillFailure,
    PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
    SubmissionScope,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Debug)]
struct Lifetime(Arc<AtomicBool>);
impl Drop for Lifetime {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn outputs<T>(selection: &JointGroupSelection<T>) -> [&T; 3] {
    [
        selection.primary_indices(),
        selection.primary_coefficients(),
        selection.always_on_coefficients(),
    ]
}

fn quote(
    inputs: &[MlxTensor; 4],
    spec: JointGroupSelectionSpec,
    ordinary: MlxMetalWorkspaceMechanisms,
    cpu: MlxCpuWorkspaceMechanisms,
) -> SpeculativeNumericalRecipe {
    let context = WorkspaceContext::new(cpu);
    let inputs = inputs
        .iter()
        .map(|value| {
            WorkspaceTensor::existing(
                context
                    .layout(value.as_array().shape(), WorkspaceDtype::Float32)
                    .unwrap()
                    .with_representation(Some(WorkspaceRepresentation::new(
                        WorkspaceFloatingType::Float32,
                        true,
                    ))),
                &context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    context.begin_state_span(inputs.iter()).unwrap();
    let result = WorkspaceBackend::joint_group_selection(
        JointGroupSelectionInput::new(&inputs[0], &inputs[1], &inputs[2], &inputs[3], spec)
            .unwrap(),
        &context,
    )
    .unwrap();
    let report = context
        .finish_report(&outputs(&result).map(Clone::clone))
        .unwrap();
    assert!(report.unpriced_operations.is_empty());
    assert!(report.unpriced_host_operations.is_empty());
    let rows = result.primary_indices().shape()[0] as u64;
    // The selected index view keeps the complete partition backing; both
    // coefficient views keep one joined allocation, charged only once.
    let retained = ordinary
        .allocation()
        .fixed_buffer_capacity(rows * spec.selectable_groups() as u64 * 4)
        .unwrap()
        + ordinary
            .allocation()
            .fixed_buffer_capacity(rows * (spec.top_k() + spec.always_on_groups()) as u64 * 4)
            .unwrap();
    assert_eq!(report.tensor_buffers.retained_bytes, Some(retained));
    let recipe =
        SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 3, ordinary, cpu, &context)
            .unwrap();
    assert_eq!(recipe.kernels, 0);
    assert_eq!(recipe.completion.validation_roots, 0);
    recipe
}

fn values(output: &JointGroupSelection<MlxTensor>) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    (
        output
            .primary_indices()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<u32>()
            .unwrap(),
        output
            .primary_coefficients()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        output
            .always_on_coefficients()
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
    )
}

fn check_numeric(
    actual: &(Vec<u32>, Vec<f32>, Vec<f32>),
    hidden: &[f32],
    weight: &[f32],
    correction: &[f32],
    spec: JointGroupSelectionSpec,
    global: f32,
) {
    let k = spec.top_k() as usize;
    let shared = spec.always_on_groups() as usize;
    let primary = spec.selectable_groups() as usize;
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    for (row, x) in hidden.chunks_exact(8).enumerate() {
        let logits = weight
            .chunks_exact(8)
            .map(|w| {
                x.iter()
                    .zip(w)
                    .map(|(&x, &w)| f64::from(x) * f64::from(w))
                    .sum::<f64>()
            })
            .collect::<Vec<_>>();
        let scores = (0..primary)
            .map(|i| sigmoid(logits[i]) + f64::from(correction[i]))
            .collect::<Vec<_>>();
        let mut expected_ids = (0..primary).collect::<Vec<_>>();
        expected_ids.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        if k < primary {
            assert!(scores[expected_ids[k - 1]] - scores[expected_ids[k]] > 1e-3);
        }
        expected_ids.truncate(k);
        let ids = &actual.0[row * k..(row + 1) * k];
        let mut sorted_ids = ids.iter().map(|&id| id as usize).collect::<Vec<_>>();
        sorted_ids.sort_unstable();
        expected_ids.sort_unstable();
        assert_eq!(sorted_ids, expected_ids, "row={row}, k={k}");
        // Selection uses the correction; normalized coefficients use unbiased
        // logits. The independent oracle normalizes sigmoid directly, without
        // reproducing the native log-sigmoid/LogAddExp/softmax pipeline.
        if k == 1 {
            assert_eq!(ids, &[0]);
            assert!(logits[0] < logits[3]);
        }
        let denominator = ids
            .iter()
            .map(|&id| sigmoid(logits[id as usize]))
            .sum::<f64>()
            + logits[primary..].iter().map(|&x| sigmoid(x)).sum::<f64>();
        let scale = f64::from(spec.coefficient_scale()) * f64::from(global);
        for (column, &id) in ids.iter().enumerate() {
            let expected = sigmoid(logits[id as usize]) / denominator * scale;
            assert!(
                (f64::from(actual.1[row * k + column]) - expected).abs() < 3e-6,
                "selected coefficient row={row}, column={column}, k={k}"
            );
        }
        for column in 0..shared {
            let expected = sigmoid(logits[primary + column]) / denominator * scale;
            assert!(
                (f64::from(actual.2[row * shared + column]) - expected).abs() < 3e-6,
                "shared coefficient row={row}, column={column}, k={k}"
            );
        }
        let sum = actual.1[row * k..(row + 1) * k]
            .iter()
            .chain(&actual.2[row * shared..(row + 1) * shared])
            .map(|&x| f64::from(x))
            .sum::<f64>();
        assert!((sum - scale).abs() < 3e-6);
    }
}

#[test]
#[ignore = "requires qualified original CPU streams and native allocator"]
fn original_cpu_joint_selector_preserves_unbiased_weights_strided_selection_and_custody() {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (ordinary, cpu) = mechanisms();
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
    for (rows, k) in [(1usize, 1), (2, 2), (3, 4)] {
        let spec = JointGroupSelectionSpec::new(4, 2, k, 1.7).unwrap();
        let hidden = (0..rows)
            .flat_map(|r| {
                [
                    1.0 + r as f32 * 0.5,
                    -0.75 + r as f32 * 0.125,
                    0.5,
                    -0.25,
                    0.125,
                    -0.375,
                    0.25,
                    0.625,
                ]
            })
            .collect::<Vec<_>>();
        let weight = (0..6)
            .flat_map(|g| (0..8).map(move |c| ((g * 11 + c * 5) % 17 - 8) as f32 / 16.0))
            .collect::<Vec<_>>();
        let correction = [0.55f32, -0.3, 0.1, -0.1];
        let global = 0.75f32;
        let inputs = [
            MlxTensor::from_array(Array::from_slice(&hidden, &[1, rows as i32, 8])),
            MlxTensor::from_array(Array::from_slice(&weight, &[6, 8])),
            MlxTensor::from_array(Array::from_slice(&correction, &[4])),
            MlxTensor::from_array(Array::from_slice(&[global], &[1])),
        ];
        let run = || {
            MlxNeuralBackend::joint_group_selection(
                JointGroupSelectionInput::new(&inputs[0], &inputs[1], &inputs[2], &inputs[3], spec)
                    .unwrap(),
                stream,
            )
            .unwrap()
        };
        let ordinary_output = run();
        let expected = values(&ordinary_output);
        check_numeric(&expected, &hidden, &weight, &correction, spec, global);
        drop(ordinary_output);
        for input in &inputs {
            input.as_array().evaluated().unwrap();
        }
        let recipe = quote(&inputs, spec, ordinary, cpu);
        let completion = recipe.completion;
        let physical = OriginalBufferBudget::metal_population_layout(
            &allocator,
            recipe.storage.mutable_bytes().try_into().unwrap(),
            recipe.storage.maximum_births(),
        )
        .unwrap()
        .capacity();
        let released = Arc::new(AtomicBool::new(false));
        let owner = Arc::new(Lifetime(released.clone()));
        let graph = PreparedSubmissionGraphQuota::try_new(recipe.graph_capacity, owner.clone())
            .unwrap()
            .try_allocate()
            .unwrap();
        let records = PreparedSubmissionRecordQuota::try_new(recipe.record_capacity, owner.clone())
            .unwrap()
            .try_allocate()
            .unwrap();
        let budget = PreparedOriginalBufferBudget::try_new(&allocator, physical, owner.clone())
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
        for input in &inputs {
            OperationEvent::validate_traversal_leaf(input.as_array(), &observer).unwrap();
        }
        let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
        let output = run();
        drop(bank);
        for value in outputs(&output) {
            roots.append(value.as_array()).unwrap();
        }
        roots
            .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
            .unwrap_or_else(|cause| panic!("rows={rows}, k={k}: {cause:?}"));
        let actual = values(&output);
        assert_eq!(actual, expected);
        check_numeric(&actual, &hidden, &weight, &correction, spec, global);
        let [ids, primary, shared] =
            outputs(&output).map(|value| value.as_array().allocation_info().unwrap().unwrap());
        assert_eq!(primary, shared);
        assert_ne!(ids.identity(), primary.identity());
        assert!(ids.bytes() >= rows * 4 * 4);
        assert!(shared.bytes() >= rows * (k as usize + 2) * 4);
        assert!(budget.occupied_bytes() > 0 && budget.occupied_bytes() <= physical);
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
        // Keep only the narrow always-on view. It must retain the whole joined
        // allocation and its original payer after every graph/scope owner ends.
        let escaped = output.always_on_coefficients().clone();
        safemlx::try_with_submission_retirement(|| {
            drop((
                output, roots, scope, observer, failure, records, graph, budget,
            ))
        })
        .unwrap();
        drop(owner);
        safemlx::reclaim_allocation_owners();
        assert!(!released.load(Ordering::SeqCst));
        assert_eq!(escaped.as_array().allocation_info().unwrap(), Some(shared));
        assert_eq!(
            escaped
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap(),
            expected.2
        );
        drop(escaped);
        crate::backend::submission_recovery::wait_for_retirement(|| {
            assert!(std::time::Instant::now() < deadline);
            safemlx::try_retire_completed_submissions().unwrap();
            MlxNeuralBackend::reclaim_retired_resources();
            safemlx::reclaim_allocation_owners();
            released.load(Ordering::SeqCst)
        });
    }
}
