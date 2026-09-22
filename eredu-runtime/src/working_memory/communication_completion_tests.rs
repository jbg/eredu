use super::*;

#[derive(Debug)]
struct CompletionFacts(WorkspaceCompletionStrategy);
impl WorkspaceMechanisms for CompletionFacts {
    fn completion_strategy(&self) -> WorkspaceCompletionStrategy {
        self.0
    }
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        assert!(matches!(
            operation.kind.as_view(),
            WorkspaceOperationKindView::ValueCompletion
        ));
        assert!(!operation.inputs.is_empty());
        assert!(operation.outputs.is_empty());
        Ok(Some(WorkspaceOperationBound {
            outputs: Vec::new(),
            scratch_bytes: 0,
            assumptions: "mock completion owns no tensor backing".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 0,
            assumptions: "mock completion owns no host payload".into(),
        }))
    }
}

#[test]
fn shared_dependency_driver_records_the_selected_completion_worker_and_keeps_alias_custody() {
    use WorkspaceCompletionStrategy::{EnclosingSubmission, OperationSubmissions};
    for strategy in [OperationSubmissions, EnclosingSubmission] {
        let context = WorkspaceContext::new(CompletionFacts(strategy));
        let value = WorkspaceTensor::existing(
            context.layout(&[2, 3], WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        context.begin_state_span([&value]).unwrap();
        if strategy == EnclosingSubmission {
            assert!(
                WorkspaceBackend::submit_local_dependencies(std::iter::once(&value), &context)
                    .is_err()
            );
            assert_eq!(context.operation_count(), 0);
        }
        let policy = CommunicationCompletionPolicy::new(
            std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::QuarantineUntilComplete,
        )
        .unwrap();
        let manifest = CommunicationManifest::new(1, 0, Vec::new(), Vec::new())
            .unwrap()
            .with_completion_policy(policy);
        let communication = Communication::new(
            manifest,
            Vec::new(),
            Vec::new(),
            WorkspaceCommunicationMetadata,
        )
        .unwrap();
        let parallel = WorkspaceParallelContext::new(0, 1).unwrap();
        communication
            .complete_execution_dependencies_with_parallel(
                &[&value, &value],
                &context,
                Some(&parallel),
            )
            .unwrap();
        let report = context.report(&[value]).unwrap();
        assert_eq!(report.operations.len(), 1);
        assert!(match strategy {
            OperationSubmissions => matches!(
                report.operations[0].kind,
                WorkspaceOperationKind::CommunicationDependencies
            ),
            EnclosingSubmission => matches!(
                report.operations[0].kind,
                WorkspaceOperationKind::ValueCompletion
            ),
        });
        assert_eq!(report.operations[0].inputs.len(), 2);
        assert_eq!(report.tensor_buffers.total_bytes, Some(0));
        assert_eq!(report.state.unwrap().retained_bytes, Some(24));
        communication.authority().ensure_active().unwrap();
    }
}
