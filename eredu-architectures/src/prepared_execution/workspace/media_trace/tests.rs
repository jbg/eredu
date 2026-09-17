use super::*;
use eredu_nn::{Tensor, workspace::*};
use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
use std::convert::Infallible;

#[derive(Debug)]
struct MissingFacts;
impl WorkspaceMechanisms for MissingFacts {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        panic!("checked media trace entered ordinary facts")
    }
}
impl WorkspaceFactMechanisms for MissingFacts {
    type Error = Infallible;
    fn operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        Ok(None)
    }
    fn write_operation_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceEffectDestination<'_>,
    ) -> Result<Option<WorkspaceOperationFacts>, Infallible> {
        panic!("missing tensor facts must not emit")
    }
    fn host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        Ok(None)
    }
    fn write_host_facts(
        &self,
        _: WorkspaceOperationView<'_>,
        _: WorkspaceHostDestination<'_>,
    ) -> Result<Option<WorkspaceHostFacts>, Infallible> {
        panic!("missing host facts must not emit")
    }
}

#[test]
fn checked_cut_keeps_later_operations_and_report_alias_keeps_planning_custody() {
    let capacity = 1 << 24;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let funding = pool
        .prepare_workspace_metadata(&execution, capacity)
        .unwrap();
    let context = WorkspaceContext::new_with_metadata_funding(MissingFacts, funding).unwrap();
    let geometry = InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 2,
        max_output_tokens: 1,
        prefill_chunk_positions: 1,
        output: eredu_core::OutputDemand::LastPosition,
    };
    let mut intervals = Vec::new();
    let report = quote_inference_workspace_with_report_owner(geometry, &context, |_| {
        context.begin_state_span([])?;
        let input = WorkspaceTensor::full_i32(3, &[1], &context)?;
        let cut_operations = context.operation_count();
        let cut = context.report_scalars(std::slice::from_ref(&input))?;
        assert!(cut.state.is_some());
        let output = input.square(&context)?;
        let complete = context.finish_report(std::slice::from_ref(&output))?;
        assert!(complete.operations.len() > cut_operations);
        let owner = TraceOwner::new(complete, &context)?;
        intervals.push(owner.clone());
        Ok::<_, Error>(owner)
    })
    .unwrap();
    assert_eq!(report.completed_spans(), 3);
    assert_eq!(intervals.len(), 3);
    assert!(report.first_gap().is_some(), "missing facts became a bound");
    assert!(
        intervals
            .iter()
            .all(|interval| interval.operations.len() == 2)
    );
    let charged = pool.used_bytes().unwrap();
    assert!(charged > 0);
    drop(context);
    drop(report);
    assert_eq!(pool.used_bytes().unwrap(), charged);
    let last = intervals[0].clone();
    drop(intervals);
    assert_eq!(pool.used_bytes().unwrap(), charged);
    assert_eq!(last.operations.len(), 2);
    drop(last);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
