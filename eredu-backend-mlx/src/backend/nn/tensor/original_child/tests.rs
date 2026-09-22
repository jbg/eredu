use super::*;
use safemlx::{
    PreparedPrefillFailure, PreparedSubmissionGraphQuota, PreparedSubmissionRecordQuota,
    PreparedSubmissionScopeOwner, RetainedPrefillFailure, SubmissionGraphQuota,
    SubmissionRecordQuota, SubmissionScope,
};

struct Original {
    scope: SubmissionScope,
    _graph: SubmissionGraphQuota,
    _records: SubmissionRecordQuota,
    _failure: RetainedPrefillFailure,
}
impl Original {
    fn new() -> Self {
        let graph = PreparedSubmissionGraphQuota::try_new(1 << 20, ())
            .unwrap()
            .try_allocate()
            .unwrap();
        let records = PreparedSubmissionRecordQuota::try_new(1 << 16, ())
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
        Self {
            scope,
            _graph: graph,
            _records: records,
            _failure: failure,
        }
    }
}
#[test]
fn retained_parent_authentication_rejects_stale_and_unrelated_current_scopes() {
    let mut first = Original::new();
    let parent = OriginalScopeObserver::require_current().unwrap();
    assert!(
        TokenValidationScope::capture_observer_for(&parent)
            .unwrap()
            .same_scope(&parent)
    );
    first.scope.seal();
    assert!(TokenValidationScope::capture_observer_for(&parent).is_err());
    {
        let _second = Original::new();
        let current = OriginalScopeObserver::require_current().unwrap();
        assert!(!parent.same_scope(&current));
        assert!(TokenValidationScope::capture_observer_for(&parent).is_err());
        assert!(
            TokenValidationScope::capture_observer_for(&current)
                .unwrap()
                .same_scope(&current)
        );
    }
    assert!(TokenValidationScope::capture_observer_for(&parent).is_err());
}
