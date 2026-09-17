use super::*;
use crate::preparation_selection::{
    select_preparation,
    tests::{inspected_config, prediction_config, BoundedIndependentAdapter},
};
use eredu_nn::workspace::{WorkspaceMechanisms, WorkspaceOperation, WorkspaceOperationBound};
use eredu_runtime::NormalizedLoadRequest;

#[derive(Debug)]
struct NoPayload;
impl WorkspaceMechanisms for NoPayload {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(None)
    }
}

#[test]
fn cold_prediction_destinations_keep_source_and_do_not_publish_executable_placement() {
    let (_directory, inspection) = inspected_config(prediction_config());
    let request = NormalizedLoadRequest::default();
    let selected =
        select_preparation(&inspection, &request, &BoundedIndependentAdapter::default()).unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        request.preparation_policy().unwrap(),
        selected.session_capabilities(),
    )
    .unwrap();
    let sources = crate::prepared_sources::prepare_model_sources(plan, selected).unwrap();
    let context = WorkspaceContext::new(NoPayload);
    assert!(sources.graph().prediction_placement.get().is_none());
    let first = project_replicated_text_binding_destinations(&sources, &context).unwrap();
    assert!(!first.prediction_modules().is_empty());
    assert!(sources.graph().prediction_placement.get().is_none());
    let second = project_replicated_text_binding_destinations(&sources, &context).unwrap();
    assert!(sources.graph().prediction_placement.get().is_none());
    assert_eq!(
        first.prediction_modules().len(),
        second.prediction_modules().len()
    );
    for (index, (left, right)) in first
        .prediction_modules()
        .iter()
        .zip(second.prediction_modules())
        .enumerate()
    {
        assert!(!left.parameters.is_empty());
        assert!(!left.tasks.is_empty());
        assert_eq!(left.ordinal, right.ordinal);
        assert_eq!(left.parameters, right.parameters);
        assert_eq!(left.tasks, right.tasks);
        assert_eq!(left.shared, right.shared);
        assert!(first.prediction_modules()[..index]
            .iter()
            .all(|prior| prior.ordinal != left.ordinal));
    }
    // The real executable still owns the single placement publication.
    let prepared =
        crate::prediction_extension::prepare_replicated_prediction_extension::<WorkspaceBackend>(
            sources.prediction_extension().unwrap(),
            sources
                .selected()
                .text_realization()
                .auxiliary_materialization_tasks(),
            &context,
            &context,
        )
        .unwrap();
    let topology = eredu_core::ParallelRankTopology::new(
        eredu_core::ParallelTopology::new(1, 1, 1, 1).unwrap(),
        0,
    )
    .unwrap();
    assert!(sources
        .graph()
        .prediction_placement
        .set(prepared.retained_placement(topology))
        .is_ok());
}
