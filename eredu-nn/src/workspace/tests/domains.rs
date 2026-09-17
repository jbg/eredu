use super::*;

#[derive(Debug)]
struct TensorOnly;
impl WorkspaceMechanisms for TensorOnly {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        AllocatingMechanism.operation_bound(op)
    }
}

#[test]
fn known_tensor_buffers_do_not_supply_a_missing_host_bound_even_for_aliases() {
    let context = WorkspaceContext::new(TensorOnly);
    let input = existing_f32(&[2, 3], &context).unwrap();
    assert_eq!(context.report(&[]).unwrap().total_bytes, Some(0));
    let output = input.square(&context).unwrap();
    let view = output.reshape(&[6], &context).unwrap();
    let report = context.report(&[view]).unwrap();
    assert_eq!(report.tensor_buffers.total_bytes, Some(31));
    assert_eq!(report.tensor_buffers.retained_bytes, Some(24));
    assert_eq!(report.tensor_buffers.transient_bytes, Some(7));
    assert_eq!(report.retained_bytes, Some(24));
    assert_eq!(
        (
            report.total_bytes,
            report.transient_bytes,
            report.host_workspace_bytes
        ),
        (None, None, None)
    );
    assert!(report.unpriced_operations.is_empty());
    assert_eq!(report.unpriced_host_operations, [0, 1]);
    context.begin_span();
    let alias = input.reshape(&[6], &context).unwrap();
    let report = context.report(&[alias]).unwrap();
    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
    assert_eq!(report.total_bytes, None);
    assert_eq!(report.unpriced_host_operations, [0]);
}

#[derive(Debug)]
struct Both {
    missing_tensor: bool,
}
impl WorkspaceMechanisms for Both {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if self.missing_tensor {
            Ok(None)
        } else {
            AllocatingMechanism.operation_bound(op)
        }
    }
    fn host_workspace_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {bytes:if matches!(op.kind,WorkspaceOperationKind::View(_) | WorkspaceOperationKind::Transpose(_)) {3} else {5},assumptions:"test mechanism: five-byte operation staging or three-byte view metadata; tensor backing excluded".into()}))
    }
}
#[test]
fn disjoint_host_workspace_is_added_once_and_cloned_tensor_roots_stay_deduplicated() {
    let context = WorkspaceContext::new(Both {
        missing_tensor: false,
    });
    let input = existing_f32(&[2, 3], &context).unwrap();
    let output = input.square(&context).unwrap();
    let view = output.reshape(&[6], &context.clone()).unwrap();
    let report = context
        .report(&[output.clone(), view.clone(), view])
        .unwrap();
    assert_eq!(report.tensor_buffers.total_bytes, Some(31));
    assert_eq!(report.host_workspace_bytes, Some(8));
    assert_eq!(report.total_bytes, Some(39));
    assert_eq!(report.retained_bytes, Some(24));
    assert_eq!(report.transient_bytes, Some(15));
    assert!(report.unpriced_host_operations.is_empty());
    assert!(report
        .assumptions
        .iter()
        .any(|s| s.contains("tensor backing excluded")));
    drop(output);
    assert_eq!(context.report(&[]).unwrap().total_bytes, Some(39));
    context.begin_span();
    let empty = context.report(&[]).unwrap();
    assert_eq!(
        (empty.total_bytes, empty.host_workspace_bytes),
        (Some(0), Some(0))
    );
}
#[test]
fn known_host_workspace_cannot_fill_missing_tensor_storage() {
    let context = WorkspaceContext::new(Both {
        missing_tensor: true,
    });
    let input = existing_f32(&[2, 3], &context).unwrap();
    let output = input.square(&context).unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(report.host_workspace_bytes, Some(5));
    assert_eq!(
        (
            report.total_bytes,
            report.tensor_buffers.total_bytes,
            report.retained_bytes
        ),
        (None, None, None)
    );
    assert_eq!(report.unpriced_operations, [0]);
    assert!(report.unpriced_host_operations.is_empty());
}

#[derive(Debug)]
struct InvalidHost {
    empty: bool,
    fail: bool,
}
impl WorkspaceMechanisms for InvalidHost {
    fn operation_bound(
        &self,
        op: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        AllocatingMechanism.operation_bound(op)
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        if self.fail {
            return Err(Error::backend("host workspace fact failed"));
        }
        Ok(Some(WorkspaceHostBound {
            bytes: u64::MAX,
            assumptions: if self.empty {
                " \n"
            } else {
                "overflow fixture"
            }
            .into(),
        }))
    }
}
#[test]
fn invalid_host_facts_and_overflow_cannot_publish_a_small_or_partial_quote() {
    for (empty, fail) in [(true, false), (false, true)] {
        let context = WorkspaceContext::new(InvalidHost { empty, fail });
        let input = existing_f32(&[1], &context).unwrap();
        assert!(input.square(&context).is_err());
        assert!(context.report(&[]).unwrap().operations.is_empty());
    }
    let context = WorkspaceContext::new(InvalidHost {
        empty: false,
        fail: false,
    });
    let input = existing_f32(&[1], &context).unwrap();
    let _ = input.square(&context).unwrap();
    assert!(context.report(&[]).is_err()); // tensor + host domain sum overflow
    assert!(input.square(&context).is_err()); // host accumulation overflow
    let trace = context.trace.borrow();
    assert_eq!(trace.operations.len(), 1);
    assert_eq!(trace.allocations.len(), 1); // failed operation published no output
}
