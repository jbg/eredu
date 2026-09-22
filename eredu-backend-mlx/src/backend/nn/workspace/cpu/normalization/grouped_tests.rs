use super::*;
use eredu_nn::{
    NeuralBackend, NormalizationConstructionSpec, NormalizationOperator, ParameterSpec,
    ParameterVisitorMut, Parameterized, Tensor,
};

fn spec(groups: i32, scale: u8) -> NormalizationConstructionSpec {
    NormalizationConstructionSpec {
        dimensions: 8,
        epsilon: 1e-5,
        groups: Some(groups),
        scale: match scale {
            0 => NormalizationScale::Unit,
            1 => NormalizationScale::Learned(ParameterSpec::trainable("norm.weight").unwrap()),
            _ => NormalizationScale::LearnedOffset {
                weight: ParameterSpec::trainable("norm.weight").unwrap(),
                offset: 1.0,
            },
        },
    }
}
struct Bind<'a, T>(&'a T);
impl<'a, T: Tensor + 'a> ParameterVisitorMut<'a, T> for Bind<'_, T> {
    fn visit_mut(&mut self, _: eredu_nn::ParameterMetadataView<'_>, value: &'a mut T) {
        *value = self.0.clone();
    }
}
fn quote(
    cpu: MlxCpuWorkspaceMechanisms,
    groups: i32,
    scale: u8,
    dtype: WorkspaceFloatingType,
    context: &WorkspaceContext,
) -> WorkspaceTraceReport {
    let input = WorkspaceTensor::existing(
        context
            .layout(&[2, 3, 8], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(dtype, true))),
        context,
    )
    .unwrap();
    let weight = WorkspaceTensor::existing(
        context
            .layout(&[8], WorkspaceDtype::Float32)
            .unwrap()
            .with_representation(Some(WorkspaceRepresentation::new(
                WorkspaceFloatingType::Float32,
                true,
            ))),
        context,
    )
    .unwrap();
    let mut norm = WorkspaceBackend::normalization(spec(groups, scale), context).unwrap();
    norm.visit_parameters_mut(&mut Bind(&weight));
    context.begin_span();
    let output = norm.forward(&input, context).unwrap();
    assert_eq!(output.layout().representation().unwrap().dtype(), dtype);
    let report = context.finish_report(&[output]).unwrap();
    let plan = cpu.plan(report.operations[0].as_view()).unwrap().unwrap();
    assert_eq!(plan.rank, 4);
    assert_eq!(plan.seeds, 2 + usize::from(scale == 2));
    assert_eq!(plan.parameter_shells, usize::from(scale != 0));
    report
}
#[test]
fn cpu_grouped_rms_counts_real_casts_groups_and_scale_sources() {
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
    for groups in [1, 2, 8] {
        for scale in 0..3 {
            for dtype in [
                WorkspaceFloatingType::Float32,
                WorkspaceFloatingType::Float16,
                WorkspaceFloatingType::Bfloat16,
            ] {
                let context = WorkspaceContext::new(cpu);
                let report = quote(cpu, groups, scale, dtype, &context);
                let operation = report.operations[0].as_view();
                let plan = cpu.plan(operation).unwrap().unwrap();
                assert!(
                    cpu.ordinary_report_call_controls(&report)
                        .unwrap()
                        .is_some()
                );
                let recipe = SpeculativeNumericalRecipe::inspect_cpu_equations(
                    &report, ordinary, cpu, &context,
                )
                .unwrap();
                assert_eq!(
                    recipe.storage.maximum_births(),
                    plan.population.births + plan.seeds
                );
                let unknown = [operation.inputs.get(0).unwrap().with_representation(None)];
                let mut layouts = vec![unknown[0]];
                layouts.extend(operation.inputs.iter().skip(1));
                assert!(
                    cpu.plan(WorkspaceOperationView {
                        inputs: WorkspaceLayoutList::Views(&layouts),
                        ..operation
                    })
                    .unwrap()
                    .is_none()
                );
            }
        }
    }
}

#[test]
#[ignore = "requires qualified native allocator and selected CPU execution"]
fn original_cpu_grouped_rms_matches_scalar_and_preserves_independent_sources() {
    use crate::{
        MlxTensor,
        backend::{
            MlxBackend, MlxDeviceIdentity, managed_memory::gpu_stream::PreparedExecutionStreams,
            nn::shared::MlxNeuralBackend,
        },
    };
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
    // Each native destination keeps the same original lifetime. These weak
    // probes reveal which actual carrier survives without adding a native
    // alias or preventing the required final release.
    #[derive(Debug)]
    struct Carrier(#[allow(dead_code)] Arc<Lifetime>);
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let ordinary = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    let selected =
        MlxCpuMatmulMechanism::select(eredu_nn::CpuMatmulImplementation::Float32Tiles).unwrap();
    let cpu = MlxCpuWorkspaceMechanisms::new(ordinary.allocation(), selected);
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
    let data = (0..48)
        .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.125)
        .collect::<Vec<_>>();
    let gains = (0..8).map(|i| 0.75 + i as f32 / 16.0).collect::<Vec<_>>();
    let input = MlxTensor::from_array(Array::from_slice(&data, &[2, 3, 8]));
    let weight = MlxTensor::from_array(Array::from_slice(&gains, &[8]));
    input.as_array().evaluated().unwrap();
    weight.as_array().evaluated().unwrap();
    for groups in [1, 2, 8] {
        for scale in 0..3 {
            let mut module = MlxNeuralBackend::normalization(spec(groups, scale), stream).unwrap();
            module.visit_parameters_mut(&mut Bind(&weight));
            let reference = NormalizationOperator::forward(&mut module, &input, stream).unwrap();
            let expected = reference
                .as_array()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            drop(reference);
            let width = 8 / groups as usize;
            for row in 0..6 {
                for group in 0..groups as usize {
                    let start = row * 8 + group * width;
                    let mean = data[start..start + width]
                        .iter()
                        .map(|x| x * x)
                        .sum::<f32>()
                        / width as f32;
                    for lane in 0..width {
                        let gain = if scale == 0 {
                            1.0
                        } else {
                            gains[group * width + lane] + if scale == 2 { 1.0 } else { 0.0 }
                        };
                        let value = (data[start + lane] * (mean + 1e-5).sqrt().recip()) * gain;
                        assert!(
                            (expected[start + lane] - value).abs() < 3e-6,
                            "groups={groups} scale={scale}"
                        );
                    }
                }
            }
            let context = WorkspaceContext::new(cpu);
            let report = quote(cpu, groups, scale, WorkspaceFloatingType::Float32, &context);
            let recipe =
                SpeculativeNumericalRecipe::inspect_cpu_equations(&report, ordinary, cpu, &context)
                    .unwrap();
            let completion = recipe.completion;
            let physical = OriginalBufferBudget::population_layout(
                &allocator,
                usize::try_from(recipe.storage.mutable_bytes()).unwrap(),
                recipe.storage.maximum_births(),
            )
            .unwrap()
            .capacity();
            let released = Arc::new(AtomicBool::new(false));
            let owner = Arc::new(Lifetime(released.clone()));
            let retained_owner = Arc::downgrade(&owner);
            let carrier = || {
                let value = Arc::new(Carrier(owner.clone()));
                let weak = Arc::downgrade(&value);
                (value, weak)
            };
            let (graph_owner, graph_probe) = carrier();
            let (record_owner, record_probe) = carrier();
            let (buffer_owner, buffer_probe) = carrier();
            let (failure_owner, failure_probe) = carrier();
            let (scope_owner, scope_probe) = carrier();
            let graph = PreparedSubmissionGraphQuota::try_new(recipe.graph_capacity, graph_owner)
                .unwrap()
                .try_allocate()
                .unwrap();
            let records =
                PreparedSubmissionRecordQuota::try_new(recipe.record_capacity, record_owner)
                    .unwrap()
                    .try_allocate()
                    .unwrap();
            let budget = PreparedOriginalBufferBudget::try_new(&allocator, physical, buffer_owner)
                .unwrap()
                .try_allocate()
                .unwrap();
            let failure = PreparedPrefillFailure::try_new(failure_owner)
                .unwrap()
                .try_allocate()
                .unwrap();
            let mut roots = PrefillRoots::new_retained(&runtime, 1, &graph, &failure).unwrap();
            let mut scope = SubmissionScope::try_begin_retaining(
                PreparedSubmissionScopeOwner::try_new(scope_owner)
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
            OperationEvent::validate_traversal_leaf(input.as_array(), &observer).unwrap();
            if scale != 0 {
                OperationEvent::validate_traversal_leaf(weight.as_array(), &observer).unwrap();
            }
            let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
            let actual = NormalizationOperator::forward(&mut module, &input, stream).unwrap();
            drop(bank);
            roots.append(actual.as_array()).unwrap();
            roots
                .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
                .unwrap_or_else(|error| {
                    panic!("grouped RMS completion groups={groups} scale={scale}: {error}")
                });
            assert!(!observer.status().failed());
            assert!(budget.occupied_bytes() > 0 && budget.occupied_bytes() <= physical);
            assert_eq!(
                actual
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap(),
                expected
            );
            scope.seal();
            // Output readiness can precede the CPU signal task's final descriptor
            // destruction and accepted-frontier update. Publish settlement through
            // this exact retained observer before discarding it; retirement itself
            // deliberately does not poll unfinished records or flush a stream.
            // Both settlement and final alias retirement share the original cap.
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            crate::backend::submission_recovery::wait_for_retirement(|| {
                let (progress, status) = observer.progress().unwrap();
                assert_eq!(progress, safemlx::ScopedSubmissionProgress::Observed);
                assert!(
                    !status.failed() && !status.blocked(),
                    "groups={groups} scale={scale}"
                );
                if status.is_settled() {
                    return true;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "grouped RMS record settlement groups={groups} scale={scale}"
                );
                false
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
            drop(actual);
            let retirement = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::backend::submission_recovery::wait_for_retirement(|| {
                    safemlx::try_retire_completed_submissions().unwrap();
                    MlxNeuralBackend::reclaim_retired_resources();
                    safemlx::reclaim_allocation_owners();
                    if released.load(Ordering::SeqCst) {
                        return true;
                    }
                    assert!(
                        std::time::Instant::now() < deadline,
                        "grouped RMS final alias retirement groups={groups} scale={scale}"
                    );
                    false
                })
            }));
            assert!(
                retirement.is_ok(),
                "grouped RMS retirement groups={groups} scale={scale}, remaining native owners={}, graph={}, record={}, buffer={}, failure={}, scope={}",
                retained_owner.strong_count(),
                graph_probe.strong_count(),
                record_probe.strong_count(),
                buffer_probe.strong_count(),
                failure_probe.strong_count(),
                scope_probe.strong_count()
            );
            assert_eq!(
                input
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap(),
                data
            );
            assert_eq!(
                weight
                    .as_array()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap(),
                gains
            );
        }
    }
}
