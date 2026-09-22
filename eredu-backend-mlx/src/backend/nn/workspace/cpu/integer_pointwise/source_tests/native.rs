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
#[ignore = "requires qualified original CPU streams and native allocator"]
fn original_cpu_integer_peer_preserves_values_and_escaped_output_custody() {
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
    for shape in [&[][..], &[18][..], &[2, 3][..]] {
        let count = shape.iter().map(|&n| n as usize).product::<usize>();
        let sent_values = (0..count).map(|n| n as i32 * 7 - 19).collect::<Vec<_>>();
        let received_values = (0..count)
            .map(|n| 16_777_217 + n as i32)
            .collect::<Vec<_>>();
        let sent = Array::from_slice(&sent_values, shape);
        let received = Array::from_slice(&received_values, shape);
        let factor = Array::from_slice(&[3i32], &[]);
        let offset = Array::from_slice(&[7i32], &[]);
        for input in [&sent, &received, &factor, &offset] {
            input.evaluated().unwrap();
        }
        let run = || {
            let ops = logical_collective::Native(stream);
            let zero = ops.zero(&sent)?;
            logical_collective::peer(&ops, &sent, &received, &zero)?
                .multiply(&factor, stream)?
                .subtract(&offset, stream)
        };
        let expected = received_values
            .iter()
            .map(|&n| n * 3 - 7)
            .collect::<Vec<_>>();
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
        let recipe = trace_peer(shape, ordinary, cpu);
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
        for input in [&sent, &received, &factor, &offset] {
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
            sent.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
            sent_values
        );
        assert_eq!(
            received.evaluated().unwrap().try_to_vec::<i32>().unwrap(),
            received_values
        );
    }
}
