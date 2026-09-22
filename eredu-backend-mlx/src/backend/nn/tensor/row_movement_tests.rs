//! Native source conformance. The finite assertion destination is allocated
//! before entry; public request admission is covered by the distributed tests.
use super::*;
use crate::{
    MlxTensor,
    backend::{
        MlxBackend, MlxDeviceIdentity,
        managed_memory::gpu_stream::PreparedExecutionStreams,
        nn::{
            expert_movement as movement,
            shared::MlxNeuralBackend,
            workspace::{
                MlxCpuWorkspaceMechanisms, MlxMetalWorkspaceMechanisms,
                ResidentExecutionMechanisms, SpeculativeNumericalRecipe,
            },
        },
    },
};
use eredu_nn::Tensor;
use eredu_nn::workspace::{
    WorkspaceContext, WorkspaceDtype, WorkspaceFloatingType, WorkspaceRepresentation,
    WorkspaceTensor,
};
use safemlx::{
    Device, DeviceType, OperationEvent, OriginalBufferBudget, OriginalScopeObserver, PrefillRoots,
    PrefillRootsRuntime, PreparedOriginalBufferBudget, PreparedPrefillFailure,
    PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner,
    SubmissionScope,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug)]
enum Operation {
    Gather(bool),
    Add,
}
#[derive(Debug)]
struct Lifetime(Arc<AtomicBool>);
impl Drop for Lifetime {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

fn trace(operation: Operation, inputs: &[&MlxTensor]) -> SpeculativeNumericalRecipe {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected = crate::backend::nn::workspace::MlxCpuMatmulMechanism::select(
        eredu_nn::CpuMatmulImplementation::Float32Tiles,
    )
    .unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    let context = WorkspaceContext::new(cpu);
    let inputs = inputs
        .iter()
        .map(|input| {
            let (dtype, representation) = match input.as_array().dtype() {
                Dtype::Int32 => (WorkspaceDtype::Int32, None),
                Dtype::Uint32 => (WorkspaceDtype::Uint32, None),
                dtype => (
                    WorkspaceDtype::Float32,
                    Some(WorkspaceRepresentation::new(
                        match dtype {
                            Dtype::Float32 => WorkspaceFloatingType::Float32,
                            Dtype::Float16 => WorkspaceFloatingType::Float16,
                            Dtype::Bfloat16 => WorkspaceFloatingType::Bfloat16,
                            _ => panic!("unqualified fixture dtype"),
                        },
                        true,
                    )),
                ),
            };
            WorkspaceTensor::existing(
                context
                    .layout(input.as_array().shape(), dtype)
                    .unwrap()
                    .with_representation(representation),
                &context,
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    context.begin_span();
    let ops = movement::Workspace(&context);
    let output = match operation {
        Operation::Gather(route_values) => {
            movement::gather(&ops, &inputs[0], &inputs[1], route_values)
        }
        Operation::Add => movement::add(&ops, &inputs[0], &inputs[1], &inputs[2]),
    }
    .unwrap();
    let report = context.finish_report(&[output]).unwrap();
    SpeculativeNumericalRecipe::inspect_owned_child(
        &report,
        1,
        ResidentExecutionMechanisms::Cpu { ordinary, cpu },
        &context,
    )
    .unwrap()
}

fn run_case(
    operation: Operation,
    inputs: &[&MlxTensor],
    expected: &[f32],
    invalid: bool,
    backend: &MlxBackend<'_>,
) {
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    let runtime = PrefillRootsRuntime::prepare_for_stream(stream, stream).unwrap();
    let allocator = environment.input_runtime().unwrap();
    for input in inputs {
        input.as_array().evaluated().unwrap();
        // The first call may submit a setup cast; the second settles that
        // exact retained descriptor before its independent original role.
        input.as_array().evaluated().unwrap();
    }
    let expected_shape = match operation {
        Operation::Gather(false) => {
            let mut shape = inputs[0].shape().to_vec();
            shape[0] = inputs[1].shape()[0];
            shape
        }
        Operation::Gather(true) => vec![inputs[1].shape()[0], 1],
        Operation::Add => inputs[0].shape().to_vec(),
    };
    let recipe = trace(operation, inputs);
    let completion = recipe.completion;
    // This component fixture supplies the actual production collector with its
    // exact, prepaid slot count. It claims native quotas, not facade admission.
    let mut validations = Vec::new();
    validations
        .try_reserve_exact(completion.validation_roots)
        .unwrap();
    assert_eq!(validations.capacity(), completion.validation_roots);
    let prepared = PreparedTokenValidations(
        TokenValidationBatch {
            validations,
            _original: None,
        },
        PreparedGroupedOutputs::default(),
    );
    let physical = OriginalBufferBudget::population_layout(
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
    let mut roots =
        PrefillRoots::new_retained(&runtime, completion.traversal.roots(), &graph, &failure)
            .unwrap();
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
    for input in inputs {
        OperationEvent::validate_traversal_leaf(input.as_array(), &observer).unwrap();
    }
    let collector = TokenValidationScope::begin_prepared(prepared).unwrap();
    let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
    let ops = movement::Native(stream);
    let output = match operation {
        Operation::Gather(route_values) => {
            movement::gather(&ops, inputs[0], inputs[1], route_values)
        }
        Operation::Add => movement::add(&ops, inputs[0], inputs[1], inputs[2]),
    }
    .unwrap_or_else(|cause| {
        panic!(
            "{operation:?} construction, rows {:?}: {cause:?}",
            inputs[1].as_array().shape()
        )
    });
    drop(bank);
    assert_eq!(output.shape(), expected_shape);
    let batch = collector.finish();
    assert_eq!(batch.validations.len(), completion.validation_roots);
    roots.append(output.as_array()).unwrap();
    for value in batch.arrays() {
        roots.append(value).unwrap();
    }
    roots
        .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
        .unwrap_or_else(|cause| {
            panic!(
                "{operation:?} completion, rows {:?}: {cause:?}",
                inputs[1].as_array().shape()
            )
        });
    for value in batch.arrays() {
        assert_eq!(
            value
                .completed_in_original_scope(&observer)
                .unwrap()
                .try_as_slice::<bool>()
                .unwrap(),
            &[invalid]
        );
    }
    assert_eq!(batch.validate_completed().is_err(), invalid);
    assert!(budget.occupied_bytes() <= physical);
    if !expected.is_empty() {
        assert!(budget.occupied_bytes() > 0);
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
        drop((
            batch, roots, scope, observer, failure, records, graph, budget,
        ))
    })
    .unwrap();
    drop(owner);
    safemlx::reclaim_allocation_owners();
    if !expected.is_empty() {
        assert!(
            !released.load(Ordering::SeqCst),
            "escaped output keeps original custody"
        );
    }
    assert_eq!(
        output
            .as_array()
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        expected
    );
    drop(output);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        assert!(std::time::Instant::now() < deadline);
        safemlx::try_retire_completed_submissions().unwrap();
        MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        released.load(Ordering::SeqCst)
    });
}

fn backend() -> MlxBackend<'static> {
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let selected = crate::backend::nn::workspace::MlxCpuMatmulMechanism::select(
        eredu_nn::CpuMatmulImplementation::Float32Tiles,
    )
    .unwrap();
    let streams = PreparedExecutionStreams::for_cpu_factory_with_matmul(&pool, selected)
        .unwrap()
        .unwrap();
    MlxBackend::for_prepared_execution_plan(
        streams,
        MlxDeviceIdentity::from_realized_device(&Device::new(DeviceType::Cpu, 0), None).unwrap(),
    )
}

#[test]
#[ignore = "requires qualified original CPU streams and native allocator"]
fn original_cpu_checked_row_gather_preserves_signed_indices_validation_and_custody() {
    let backend = backend();
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    let values = (0..21)
        .map(|i| (i as f32 - 9.0) * 0.375)
        .collect::<Vec<_>>();
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        let input = MlxTensor::from_array(
            Array::from_slice(&values, &[7, 3])
                .as_dtype(dtype, stream)
                .unwrap(),
        );
        for route_values in [false, true] {
            for count in [0, 1, 7] {
                let extent: i32 = if route_values { 21 } else { 7 };
                let ids = [-1, 0, -extent, 3, 3, 1, -2];
                let indices =
                    MlxTensor::from_array(Array::from_slice(&ids[..count], &[count as i32]));
                let expected = ids[..count]
                    .iter()
                    .flat_map(|&id| {
                        let row = id.rem_euclid(extent) as usize;
                        let width = if route_values { 1 } else { 3 };
                        values[row * width..(row + 1) * width].iter().copied()
                    })
                    .collect::<Vec<_>>();
                run_case(
                    Operation::Gather(route_values),
                    &[&input, &indices],
                    &expected,
                    false,
                    &backend,
                );
            }
        }
        let indices = MlxTensor::from_array(Array::from_slice(&[7i32], &[1]));
        // Invalid indices are masked before the kernel; the independently
        // completed assertion still rejects them before output publication.
        run_case(
            Operation::Gather(false),
            &[&input, &indices],
            &values[..3],
            true,
            &backend,
        );
    }
}

#[test]
#[ignore = "requires qualified original CPU streams and native allocator"]
fn original_cpu_additive_rows_preserve_duplicates_and_constructor_census() {
    let backend = backend();
    for width in [1usize, 3] {
        let base = (0..7 * width)
            .map(|i| i as f32 * 0.25 - 2.0)
            .collect::<Vec<_>>();
        let input = MlxTensor::from_array(Array::from_slice(&base, &[7, width as i32]));
        for count in [0usize, 1, 7] {
            let updates = (0..count * width)
                .map(|i| 0.125 + i as f32 * 0.25)
                .collect::<Vec<_>>();
            let update =
                MlxTensor::from_array(Array::from_slice(&updates, &[count as i32, width as i32]));
            for signed in [false, true] {
                let ids = [6i32, 0, 6, 3, 3, 1, 5];
                let mut expected = base.clone();
                for (row, &id) in ids[..count].iter().enumerate() {
                    for col in 0..width {
                        expected[id as usize * width + col] += updates[row * width + col];
                    }
                }
                let indices = MlxTensor::from_array(if signed {
                    let ids = ids[..count]
                        .iter()
                        .map(|&n| if n == 6 { -1 } else { n })
                        .collect::<Vec<_>>();
                    Array::from_slice(&ids, &[count as i32, 1])
                } else {
                    let ids = ids[..count].iter().map(|&n| n as u32).collect::<Vec<_>>();
                    Array::from_slice(&ids, &[count as i32, 1])
                });
                run_case(
                    Operation::Add,
                    &[&input, &indices, &update],
                    &expected,
                    false,
                    &backend,
                );
            }
        }
    }
}

#[test]
fn cpu_ranked_axis_zero_gather_preserves_full_slices_and_original_custody() {
    if !crate::tests::support::native_process::enter("cpu-ranked-axis-zero-gather") {
        return;
    }
    let backend = backend();
    for shape in [&[8, 4, 8][..], &[3, 2, 2, 3][..]] {
        let count = shape.iter().map(|&n| n as usize).product::<usize>();
        let width = shape[1..].iter().map(|&n| n as usize).product::<usize>();
        let values = (0..count)
            .map(|n| (n as f32 - 91.0) * 0.125)
            .collect::<Vec<_>>();
        let picks = [2, -1, 0, 2];
        let input = MlxTensor::from_array(Array::from_slice(&values, shape));
        let indices = MlxTensor::from_array(Array::from_slice(&picks, &[4]));
        let expected = picks
            .iter()
            .flat_map(|&index| {
                let index = if index < 0 { shape[0] + index } else { index } as usize;
                values[index * width..(index + 1) * width].iter().copied()
            })
            .collect::<Vec<_>>();
        // Signed indices require the same prepaid validation collector and
        // completed assertion roots as the ordinary checked-gather worker.
        run_case(
            Operation::Gather(false),
            &[&input, &indices],
            &expected,
            false,
            &backend,
        );
    }
}
