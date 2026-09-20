//! Shared native admission, completion and custody assertions for CPU equations.
use super::*;
use crate::{
    MlxTensor,
    backend::{MlxBackend, nn::shared::MlxNeuralBackend},
};
use eredu_nn::NeuralBackend;
use safemlx::{
    OriginalBufferBudget, OriginalScopeObserver, PrefillRoots, PrefillRootsRuntime,
    PreparedOriginalBufferBudget, PreparedPrefillFailure, PreparedSubmissionGraphQuota,
    PreparedSubmissionRecordQuota, PreparedSubmissionScopeOwner, SubmissionScope,
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

pub(super) fn run(
    recipe: SpeculativeNumericalRecipe,
    backend: &MlxBackend<'_>,
    inputs: &[&MlxTensor],
    construct: impl FnOnce(&safemlx::Stream) -> MlxTensor,
    verify: impl FnOnce(&MlxTensor),
) {
    let environment = backend.original_copy_environment().unwrap();
    let stream = environment.stream();
    let runtime = PrefillRootsRuntime::prepare_for_stream(stream, stream).unwrap();
    let allocator = environment.input_runtime().unwrap();
    let completion = recipe.completion;
    let physical = OriginalBufferBudget::population_layout(
        &allocator,
        usize::try_from(recipe.storage.mutable_bytes()).unwrap(),
        recipe.storage.maximum_births(),
    )
    .unwrap()
    .capacity();
    let retired = Arc::new(AtomicBool::new(false));
    let owner = Arc::new(Lifetime(retired.clone()));
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
    for input in inputs {
        OperationEvent::validate_traversal_leaf(input.as_array(), &observer).unwrap();
    }
    let bank = OperationEvent::prepare_resident_graph(completion.graph, &observer).unwrap();
    let actual = construct(stream);
    drop(bank);
    roots.append(actual.as_array()).unwrap();
    roots
        .complete_current_scope_on_stream_prepared(stream, &completion.traversal)
        .unwrap();
    assert!(!observer.status().failed());
    assert!(budget.occupied_bytes() > 0 && budget.occupied_bytes() <= physical);
    verify(&actual);
    scope.seal();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        let (progress, status) = observer.progress().unwrap();
        assert_eq!(progress, safemlx::ScopedSubmissionProgress::Observed);
        assert!(!status.failed() && !status.blocked());
        assert!(std::time::Instant::now() < deadline);
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
    assert!(!retired.load(Ordering::SeqCst));
    drop(actual);
    crate::backend::submission_recovery::wait_for_retirement(|| {
        safemlx::try_retire_completed_submissions().unwrap();
        MlxNeuralBackend::reclaim_retired_resources();
        safemlx::reclaim_allocation_owners();
        assert!(std::time::Instant::now() < deadline);
        retired.load(Ordering::SeqCst)
    });
}
