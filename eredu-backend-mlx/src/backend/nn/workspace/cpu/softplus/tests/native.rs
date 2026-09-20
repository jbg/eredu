use super::*;
use crate::backend::{
    MlxBackend, MlxDeviceIdentity, managed_memory::gpu_stream::PreparedExecutionStreams,
    nn::shared::MlxNeuralBackend,
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

#[test]
#[ignore = "requires qualified original CPU streams and native allocator"]
fn original_cpu_softplus_preserves_precision_numerics_and_escaped_custody() {
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
    for shape in [&[][..], &[7][..], &[1, 2, 8][..], &[1, 2, 2, 2][..]] {
        for (dtype, floating) in [
            (Dtype::Float32, WorkspaceFloatingType::Float32),
            (Dtype::Bfloat16, WorkspaceFloatingType::Bfloat16),
            (Dtype::Float16, WorkspaceFloatingType::Float16),
        ] {
            let count = shape.iter().map(|&n| n as usize).product::<usize>();
            let values = (0..count)
                .map(|n| [-30.0f32, -6.0, -2.0, -0.5, 0.0, 0.5, 4.0, 40.0][n % 8])
                .collect::<Vec<_>>();
            let input = Array::from_slice(&values, shape)
                .as_dtype(dtype, stream)
                .unwrap();
            input.evaluated().unwrap();
            let real_input = input
                .as_dtype(Dtype::Float32, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            let beta = std::f32::consts::LN_2;
            let run = || {
                MlxNeuralBackend::softplus(
                    crate::MlxTensor::from_array(input.clone()),
                    beta,
                    stream,
                )
                .map(crate::MlxTensor::into_array)
            };
            let ordinary = run().unwrap();
            let expected = ordinary
                .as_dtype(Dtype::Float32, stream)
                .unwrap()
                .evaluated()
                .unwrap()
                .try_to_vec::<f32>()
                .unwrap();
            drop(ordinary);
            for (&x, &actual) in real_input.iter().zip(&expected) {
                let scaled = f64::from(x) * f64::from(beta);
                let reference = if scaled > 20.0 {
                    f64::from(x)
                } else {
                    scaled.exp().ln_1p() / f64::from(beta)
                };
                let tolerance = if dtype == Dtype::Bfloat16 {
                    0.006
                } else if dtype == Dtype::Float16 {
                    0.001
                } else {
                    2e-6
                };
                assert!(
                    (f64::from(actual) - reference).abs() <= tolerance * reference.abs().max(1.0),
                    "shape={shape:?}, dtype={dtype:?}, input={x}, actual={actual}, reference={reference}"
                );
            }
            // The first ordinary evaluation submits a half-precision cast;
            // reading its F32 reference completes a different Array. Settle
            // this retained input itself before the new role: its wait path
            // promotes the completed cast to an available leaf.
            input.evaluated().unwrap();
            let recipe = trace(shape, floating);
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
            let records =
                PreparedSubmissionRecordQuota::try_new(recipe.record_capacity, owner.clone())
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
            OperationEvent::validate_traversal_leaf(&input, &observer).unwrap_or_else(|error| {
                panic!("softplus input leaf shape={shape:?} dtype={dtype:?}: {error:?}")
            });
            let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
            let output = run().unwrap();
            drop(bank);
            assert_eq!(output.dtype(), dtype);
            roots.append(&output).unwrap();
            roots
                .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
                .unwrap_or_else(|error| {
                    panic!("softplus submission shape={shape:?} dtype={dtype:?}: {error:?}")
                });
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
            assert!(
                !released.load(Ordering::SeqCst),
                "escaped output retains its original source"
            );
            assert_eq!(
                output
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
            assert_eq!(
                input
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
                    .evaluated()
                    .unwrap()
                    .try_to_vec::<f32>()
                    .unwrap(),
                real_input
            );
        }
    }
}
