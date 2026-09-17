use super::*;
use crate::{
    reclaim_allocation_owners, OrdinarySubmissionEntry, PreparedOriginalBufferBudget,
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    PreparedSubmissionScopeOwner, SubmissionScope,
};

fn facts(runtime: &PreparedInputRuntime) -> Option<OriginalPromptInputFacts> {
    match OriginalPromptInputFacts::inspect(runtime, 3) {
        Ok(facts) => Some(facts),
        Err(cause) => {
            assert_eq!(cause, OriginalPromptInputCause::Unqualified);
            assert_ne!(
                std::env::var_os("EREDU_REQUIRE_QUALIFIED_PROMPT_INPUT"),
                Some("1".into()),
                "pinned prompt validation requires positive layout qualification"
            );
            None
        }
    }
}

#[test]
fn original_prompt_eager_final_shape_keeps_exact_birth_through_scope_and_handle_aliases() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let Some(facts) = facts(&runtime) else {
        return;
    };
    let input = [17, 0xf123_4567, 29];
    let budget = PreparedOriginalBufferBudget::try_new(&runtime, facts.mutable_bytes(), ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let ordinary = OrdinarySubmissionEntry::enter().unwrap();
    let graph = PreparedSubmissionGraphQuota::try_new(facts.graph_bytes(), ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(
        PreparedSubmissionRecordQuota::<()>::minimum_layout()
            .unwrap()
            .capacity,
        (),
    )
    .unwrap()
    .try_allocate()
    .unwrap();
    let failure = PreparedPrefillFailure::try_new(())
        .unwrap()
        .try_allocate()
        .unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_graph_quota(graph.clone())
            .with_record_quota(records.clone()),
    )
    .unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    drop(ordinary);
    let array = Array::try_from_original_prompt_ids(&input).unwrap();
    assert_eq!(facts.maximum_births(), 1);
    assert_eq!(budget.occupied_bytes(), facts.mutable_bytes());
    assert_eq!(
        scope.accepted_records().activity(),
        crate::SubmissionRecordActivity::Quiescent
    );
    scope.seal();
    let allocation = budget.inspect_array(&array).unwrap().unwrap().allocation();
    assert_eq!(allocation.bytes(), facts.mutable_bytes());
    assert_eq!(array.shape(), &[1, 3]);
    assert_eq!(
        array.evaluated().unwrap().try_as_slice::<u32>().unwrap(),
        &input
    );
    let alias = array.clone();
    drop((array, scope, failure, records));
    assert_eq!(budget.occupied_bytes(), facts.mutable_bytes());
    assert_eq!(
        alias
            .inspect_original_buffer_alias()
            .unwrap()
            .unwrap()
            .allocation()
            .identity(),
        allocation.identity()
    );
    drop(alias);
    reclaim_allocation_owners();
    assert_eq!(budget.occupied_bytes(), 0);
    assert_eq!(graph.occupied_bytes(), 0);
}

#[test]
fn original_prompt_one_short_keeps_native_cause_and_never_falls_back_to_ordinary_storage() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let Some(facts) = facts(&runtime) else {
        return;
    };
    let input = [31, 37, 41];
    assert_eq!(
        Array::try_from_original_prompt_ids(&[])
            .unwrap_err()
            .scoped_evaluation_cause(),
        Some(ScopedEvaluationCause::Invalid)
    );
    assert_eq!(
        Array::try_from_original_prompt_ids(&input)
            .unwrap_err()
            .scoped_evaluation_cause(),
        Some(ScopedEvaluationCause::Domain)
    );
    let budget = PreparedOriginalBufferBudget::try_new(&runtime, facts.mutable_bytes() - 1, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let ordinary = OrdinarySubmissionEntry::enter().unwrap();
    let graph = PreparedSubmissionGraphQuota::try_new(facts.graph_bytes(), ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(
        PreparedSubmissionRecordQuota::<()>::minimum_layout()
            .unwrap()
            .capacity,
        (),
    )
    .unwrap()
    .try_allocate()
    .unwrap();
    let failure = PreparedPrefillFailure::try_new(())
        .unwrap()
        .try_allocate()
        .unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_graph_quota(graph.clone())
            .with_record_quota(records.clone()),
    )
    .unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    drop(ordinary);
    let error = Array::try_from_original_prompt_ids(&input).unwrap_err();
    assert_eq!(
        error.scoped_evaluation_cause(),
        Some(ScopedEvaluationCause::Failed)
    );
    assert!(std::error::Error::source(&error).is_some());
    assert!(failure.error().is_some());
    assert_eq!(budget.occupied_bytes(), 0);
    assert_eq!(graph.occupied_bytes(), 0);
    assert_eq!(input, [31, 37, 41]);
    scope.seal();
    drop((scope, failure, records, graph, budget));
    reclaim_allocation_owners();
    assert!(
        std::error::Error::source(&error).is_some(),
        "returned error preserves the native source after recovery owners retire"
    );
    drop(error);
    reclaim_allocation_owners();
}

#[test]
fn original_prediction_signed_source_and_prepared_alias_keep_exact_values_and_birth() {
    let runtime = PreparedInputRuntime::prepare().unwrap();
    let Some(facts) = facts(&runtime) else { return; };
    let input = [i32::MIN, -17, i32::MAX];
    let mut alias_slot = crate::PreparedArrayClone::try_prepare_for_inspection().unwrap();
    let budget = PreparedOriginalBufferBudget::try_new(&runtime, facts.mutable_bytes(), ())
        .unwrap().try_allocate().unwrap();
    let ordinary = OrdinarySubmissionEntry::enter().unwrap();
    let graph = PreparedSubmissionGraphQuota::try_new(facts.graph_bytes(), ())
        .unwrap().try_allocate().unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(
        PreparedSubmissionRecordQuota::<()>::minimum_layout().unwrap().capacity, (),
    ).unwrap().try_allocate().unwrap();
    let failure = PreparedPrefillFailure::try_new(()).unwrap().try_allocate().unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(()).unwrap()
            .with_graph_quota(graph.clone()).with_record_quota(records.clone()),
    ).unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    failure.bind_original_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    scope.bind_original_buffer_budget(&budget).unwrap();
    drop(ordinary);
    let array = Array::try_from_original_prediction_ids(&input).unwrap();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let alias = alias_slot.fill_in_original_scope(&array, &observer).unwrap();
    assert_eq!(budget.occupied_bytes(), facts.mutable_bytes());
    assert_eq!(scope.accepted_records().activity(), crate::SubmissionRecordActivity::Quiescent);
    scope.seal();
    let allocation = budget.inspect_array(&array).unwrap().unwrap().allocation();
    assert_eq!(array.shape(), &[1, 3]);
    assert_eq!(array.dtype(), crate::Dtype::Int32);
    assert_eq!(array.evaluated().unwrap().try_as_slice::<i32>().unwrap(), &input);
    drop((array, scope, failure, records, observer, alias_slot));
    assert_eq!(budget.occupied_bytes(), facts.mutable_bytes());
    assert_eq!(alias.inspect_original_buffer_alias().unwrap().unwrap().allocation().identity(), allocation.identity());
    assert_eq!(alias.evaluated().unwrap().try_as_slice::<i32>().unwrap(), &input);
    drop(alias);
    reclaim_allocation_owners();
    assert_eq!(budget.occupied_bytes(), 0);
    assert_eq!(graph.occupied_bytes(), 0);
}
