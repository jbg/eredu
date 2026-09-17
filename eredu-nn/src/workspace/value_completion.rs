//! Actual value-completion boundary recorded by a shared mechanism.
use super::*;
impl WorkspaceContext {
    /// Records completion of existing values without creating output storage.
    /// Empty or foreign roots reject before changing the trace. Native source,
    /// stream, finite traversal and completion permission remain independent.
    pub fn complete_values(&self, values: &[&WorkspaceTensor]) -> Result<(), Error> {
        self.record_values(WorkspaceOperationKind::ValueCompletion, values)
    }
    /// Records actual roots kept until the enclosing model completion. This
    /// grants no source or submission authority and performs no completion.
    pub fn retain_values(&self, values: &[&WorkspaceTensor]) -> Result<(), Error> {
        self.record_values(WorkspaceOperationKind::ValueRetention, values)
    }
    fn record_values(&self, kind: WorkspaceOperationKind, values: &[&WorkspaceTensor]) -> Result<(), Error> {
        if values.is_empty() {
            return Err(WorkspaceMetadataError::Unqualified.into());
        }
        self.validate_values(values.iter().copied())?;
        self.charge_metadata(std::mem::size_of::<(
            &Self,
            &[&WorkspaceTensor],
            (&Self, &[&WorkspaceTensor]),
            WorkspaceOperationKind,
            Vec<WorkspaceLayout>,
            Vec<WorkspaceTensor>,
            Result<(), Error>,
        )>())?;
        let outputs = self.execute(kind, values, Vec::new())?;
        debug_assert!(outputs.is_empty());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Debug)]
    struct CompletionFacts;
    impl WorkspaceMechanisms for CompletionFacts {
        fn operation_bound(
            &self,
            operation: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            assert!(matches!(
                operation.kind,
                WorkspaceOperationKind::ValueCompletion | WorkspaceOperationKind::ValueRetention
            ));
            assert!(operation.outputs.is_empty());
            Ok(Some(WorkspaceOperationBound {
                outputs: Vec::new(),
                scratch_bytes: 0,
                assumptions: "existing-value completion creates no output storage".into(),
            }))
        }
        fn host_workspace_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceHostBound>, Error> {
            Ok(Some(WorkspaceHostBound {
                bytes: 0,
                assumptions: "mock completion has no host numerical payload".into(),
            }))
        }
    }
    #[test]
    fn value_completion_preserves_storage_and_rejects_foreign_or_empty_before_trace() {
        let context = WorkspaceContext::new(CompletionFacts);
        let other = WorkspaceContext::new(CompletionFacts);
        let layout = context.layout(&[2, 3], WorkspaceDtype::Float32).unwrap();
        let value = WorkspaceTensor::existing(layout, &context).unwrap();
        let foreign = WorkspaceTensor::existing(
            other.layout(&[2, 3], WorkspaceDtype::Float32).unwrap(),
            &other,
        )
        .unwrap();
        context.begin_state_span([&value]).unwrap();
        assert!(context.complete_values(&[]).is_err());
        assert!(context.complete_values(&[&value, &foreign]).is_err());
        assert_eq!(context.operation_count(), 0);
        context.complete_values(&[&value, &value]).unwrap();
        context.retain_values(&[&value, &value]).unwrap();
        let report = context.report(&[value]).unwrap();
        assert_eq!(report.operations.len(), 2);
        assert_eq!(report.operations[0].inputs.len(), 2);
        assert!(report.operations[0].outputs.is_empty());
        assert_eq!(report.tensor_buffers.total_bytes, Some(0));
        assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(24));
        assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
        assert_eq!(report.inference_transient_bytes(), Some(0));
    }
}
