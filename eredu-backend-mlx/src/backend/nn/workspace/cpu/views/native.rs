use super::*;
use crate::backend::{
    managed_memory::gpu_stream::PreparedExecutionStreams, nn::shared::MlxNeuralBackend, MlxBackend,
    MlxDeviceIdentity,
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

#[test]
fn original_strided_transpose_unit_reshape_preserves_values_and_retires() {
    if !crate::tests::support::native_process::enter("cpu-strided-transpose-reshape") {
        return;
    }
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
    let values = (0..768)
        .map(|i| ((i % 31) as f32 - 15.) / 8.)
        .collect::<Vec<_>>();
    let mut expected = Vec::new();
    for head in 0..2 {
        for row in 8..24 {
            for lane in 0..4 {
                expected.push(values[row * 24 + 8 + head * 4 + lane]);
            }
        }
    }
    for dtype in [
        WorkspaceFloatingType::Float32,
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Bfloat16,
    ] {
        let array = match dtype {
            WorkspaceFloatingType::Float32 => Array::from_slice(&values, &[32, 3, 2, 4]),
            WorkspaceFloatingType::Float16 => Array::from_slice(
                &values
                    .iter()
                    .copied()
                    .map(half::f16::from_f32)
                    .collect::<Vec<_>>(),
                &[32, 3, 2, 4],
            ),
            WorkspaceFloatingType::Bfloat16 => Array::from_slice(
                &values
                    .iter()
                    .copied()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
                &[32, 3, 2, 4],
            ),
        };
        let input = crate::MlxTensor::from_array(array);
        input.as_array().evaluated().unwrap();
        let context = WorkspaceContext::new(cpu);
        let source = WorkspaceTensor::existing(
            context
                .layout(&[32, 3, 2, 4], WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
            &context,
        )
        .unwrap();
        context.begin_state_span([&source]).unwrap();
        let (expanded, transposed) = transposed_unit_views(&source, &context);
        let roundtrip = expanded.reshape(&[2, 16, 4], &context).unwrap();
        let report = context
            .finish_report(&[expanded, transposed, roundtrip])
            .unwrap();
        let recipe =
            SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 3, ordinary, cpu, &context)
                .unwrap();
        assert_eq!(recipe.storage.maximum_births(), 1);
        super::super::super::test_execution::run_many(
            recipe,
            &backend,
            &[&input],
            |stream| {
                let segment = input
                    .narrow_axis(1, 1, 2, stream)
                    .unwrap()
                    .squeeze_axes(&[1], stream)
                    .unwrap()
                    .narrow_axis(0, 8, 24, stream)
                    .unwrap()
                    .transpose_axes(&[1, 0, 2], stream)
                    .unwrap();
                let expanded = segment.reshape(&[1, 2, 16, 4], stream).unwrap();
                let roundtrip = expanded.reshape(&[2, 16, 4], stream).unwrap();
                [expanded, segment, roundtrip]
            },
            |actual| {
                assert_eq!(actual[0].shape(), &[1, 2, 16, 4]);
                assert_eq!(actual[1].shape(), &[2, 16, 4]);
                assert_eq!(actual[2].shape(), &[2, 16, 4]);
                for output in actual {
                    let evaluated = output.as_array().evaluated().unwrap();
                    let values = match dtype {
                        WorkspaceFloatingType::Float32 => evaluated.try_to_vec::<f32>().unwrap(),
                        WorkspaceFloatingType::Float16 => evaluated
                            .try_to_vec::<half::f16>()
                            .unwrap()
                            .into_iter()
                            .map(half::f16::to_f32)
                            .collect(),
                        WorkspaceFloatingType::Bfloat16 => evaluated
                            .try_to_vec::<half::bf16>()
                            .unwrap()
                            .into_iter()
                            .map(half::bf16::to_f32)
                            .collect(),
                    };
                    assert_eq!(values, expected);
                }
            },
        );
    }
}

#[test]
fn original_strided_unit_axis_views_preserve_values_and_source_custody() {
    if !crate::tests::support::native_process::enter("cpu-strided-unit-axis") {
        return;
    }
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
    let values = (0..768)
        .map(|i| (i as f32 - 319.0) / 8.0)
        .collect::<Vec<_>>();
    let input = crate::MlxTensor::from_array(Array::from_slice(&values, &[32, 3, 2, 4]));
    input.as_array().evaluated().unwrap();
    let expected = (0..32)
        .flat_map(|row| values[row * 24 + 8..row * 24 + 16].iter().copied())
        .collect::<Vec<_>>();
    let context = WorkspaceContext::new(cpu);
    let source = WorkspaceTensor::existing(
        context
            .layout(&[32, 3, 2, 4], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            ))),
        &context,
    )
    .unwrap();
    context.begin_state_span([&source]).unwrap();
    let selected = source.narrow_axis(1, 1, 2, &context).unwrap();
    let squeezed = selected.squeeze_axes(&[1], &context).unwrap();
    let expanded = squeezed.expand_dims(1, &context).unwrap();
    let flat = expanded.reshape(&[256], &context).unwrap();
    let report = context.finish_report(&[squeezed, expanded, flat]).unwrap();
    let recipe =
        SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 3, ordinary, cpu, &context)
            .unwrap();
    assert_eq!(recipe.storage.maximum_births(), 1);
    super::super::super::test_execution::run_many(
        recipe,
        &backend,
        &[&input],
        |stream| {
            let selected = input.narrow_axis(1, 1, 2, stream).unwrap();
            let squeezed = selected.squeeze_axes(&[1], stream).unwrap();
            let expanded = squeezed.expand_dims(1, stream).unwrap();
            let flat = expanded.reshape(&[256], stream).unwrap();
            [squeezed, expanded, flat]
        },
        |actual| {
            assert_eq!(actual[0].shape(), &[32, 2, 4]);
            assert_eq!(actual[1].shape(), &[32, 1, 2, 4]);
            assert_eq!(actual[2].shape(), &[256]);
            for output in actual {
                assert_eq!(
                    output
                        .as_array()
                        .evaluated()
                        .unwrap()
                        .try_to_vec::<f32>()
                        .unwrap(),
                    expected
                );
            }
        },
    );
}

#[test]
fn original_ranked_contiguous_reshape_preserves_values_and_full_source_custody() {
    if !crate::tests::support::native_process::enter("cpu-ranked-reshape") {
        return;
    }
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
    for shape in [
        &[8, 3, 1, 2, 2][..],
        &[2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 48][..],
    ] {
        let values = std::array::from_fn::<_, 96, _>(|i| (i as f32 - 47.0) / 8.0);
        let input = crate::MlxTensor::from_array(Array::from_slice(&values, shape));
        input.as_array().evaluated().unwrap();
        let context = WorkspaceContext::new(cpu);
        let source = WorkspaceTensor::existing(
            context
                .layout(shape, WorkspaceDtype::Float32)
                .unwrap()
                .with_representation(Some(WorkspaceRepresentation::new(
                    WorkspaceFloatingType::Float32,
                    true,
                ))),
            &context,
        )
        .unwrap();
        context.begin_state_span([&source]).unwrap();
        let view = source.reshape(&[8, 12], &context).unwrap();
        let squared = view.multiply(&view, &context).unwrap();
        let report = context.finish_report(&[view, squared]).unwrap();
        let recipe =
            SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 2, ordinary, cpu, &context)
                .unwrap();
        assert_eq!(recipe.storage.maximum_births(), 1);
        super::super::super::test_execution::run_many(
            recipe,
            &backend,
            &[&input],
            |stream| {
                let view = input.reshape(&[8, 12], stream).unwrap();
                let squared = view.multiply(&view, stream).unwrap();
                [view, squared]
            },
            |actual| {
                assert_eq!(actual[0].shape(), &[8, 12]);
                assert_eq!(
                    actual[0].as_array().evaluated().unwrap().as_slice::<f32>(),
                    &values
                );
                assert_eq!(
                    actual[1].as_array().evaluated().unwrap().as_slice::<f32>(),
                    &values.map(|v| v * v)
                );
            },
        );
    }
}

#[test]
#[ignore = "requires qualified original CPU streams and native allocator"]
fn original_cpu_integer_gather_and_strided_reshape_preserve_values_and_custody() {
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
    for strided in [false, true] {
        let shape = if strided { [2, 3] } else { [3, 18] };
        let count = shape.iter().map(|&n| n as usize).product::<usize>();
        let values = (0..count)
            .map(|n| 16_777_217 + n as i32)
            .collect::<Vec<_>>();
        let input = Array::from_slice(&values, &shape);
        input.evaluated().unwrap();
        let run = || -> Result<Array, safemlx::error::Exception> {
            if strided {
                input.swap_axes(0, 1, stream)?.reshape(&[6], stream)
            } else {
                let ops = logical_collective::Native(stream);
                let stacked = packed::gather_stacked(&ops, &input, &[2, 1, 0])?;
                packed::flatten(&ops, &stacked, &[18], 3)
            }
        };
        let expected = if strided {
            [0usize, 3, 1, 4, 2, 5].map(|i| values[i]).to_vec()
        } else {
            values
                .chunks_exact(18)
                .rev()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        };
        let ordinary_output = run().unwrap();
        assert_eq!(
            ordinary_output
                .evaluated()
                .unwrap()
                .try_to_vec::<i32>()
                .unwrap(),
            expected
        );
        drop(ordinary_output);
        input.evaluated().unwrap();
        let recipe = if strided {
            let context = WorkspaceContext::new(cpu);
            let source = WorkspaceTensor::existing(
                context.layout(&shape, WorkspaceDtype::Int32).unwrap(),
                &context,
            )
            .unwrap();
            context.begin_state_span([&source]).unwrap();
            let output = source
                .transpose_axes(&[1, 0], &context)
                .unwrap()
                .reshape(&[6], &context)
                .unwrap();
            let report = context.finish_report(&[output]).unwrap();
            SpeculativeNumericalRecipe::inspect_cpu_outputs(&report, 1, ordinary, cpu, &context)
                .unwrap()
        } else {
            trace_gather(&[18], 3, true, ordinary, cpu)
        };
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
        let mut roots = PrefillRoots::new_retained(&runtime, 1, &graph, &failure).unwrap();
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
        for input in [&input] {
            OperationEvent::validate_traversal_leaf(input, &observer).unwrap();
        }
        let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
        let output = run().unwrap();
        drop(bank);
        roots.append(&output).unwrap();
        roots
            .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
            .unwrap();
        assert_eq!(
            output.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
            expected
        );
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
        safemlx::try_with_submission_retirement(|| {
            drop((roots, scope, observer, failure, records, graph, budget))
        })
        .unwrap();
        drop(owner);
        safemlx::reclaim_allocation_owners();
        assert!(!released.load(Ordering::SeqCst));
        assert_eq!(
            output.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
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
        assert_eq!(
            input.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
            values
        );
    }
}
