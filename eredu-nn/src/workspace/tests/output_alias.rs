use super::*;

#[derive(Debug)]
struct SharedOutputs {
    first: WorkspaceOutputStorage,
    second: usize,
}
impl WorkspaceMechanisms for SharedOutputs {
    fn operation_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        Ok(Some(WorkspaceOperationBound {
            outputs: vec![
                self.first.clone(),
                WorkspaceOutputStorage::AliasOutput(self.second),
                WorkspaceOutputStorage::AliasOutput(1),
            ],
            scratch_bytes: 13,
            assumptions: "test outputs share one backing across different logical views".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(Some(WorkspaceHostBound {
            bytes: 7,
            assumptions: "test host scratch".into(),
        }))
    }
}
fn layouts() -> Vec<WorkspaceLayout> {
    [vec![2, 3], vec![2, 5], vec![0]]
        .into_iter()
        .map(|s| WorkspaceLayout::new(&s, WorkspaceDtype::Float32).unwrap())
        .collect()
}

#[test]
fn output_aliases_retain_one_complete_backing_after_sibling_views_are_dropped() {
    let context = WorkspaceContext::new(SharedOutputs {
        first: WorkspaceOutputStorage::Allocate(64),
        second: 0,
    });
    let mut output = context
        .execute(WorkspaceOperationKind::Initialize, &[], layouts())
        .unwrap();
    let report = context.report(&output).unwrap();
    assert_eq!(report.tensor_buffers.retained_bytes, Some(64));
    assert_eq!(report.total_bytes, Some(84));
    let empty = output.pop().unwrap();
    drop(output);
    let report = context.report(&[empty.clone()]).unwrap();
    assert_eq!(report.tensor_buffers.retained_bytes, Some(64));
    context.begin_span();
    let report = context.report(&[empty]).unwrap();
    assert_eq!(report.tensor_buffers.total_bytes, Some(0));
}

#[test]
fn output_aliases_preserve_existing_and_possible_input_roots_without_duplicate_charges() {
    for uncertain in [false, true] {
        let context = WorkspaceContext::new(SharedOutputs {
            first: if uncertain {
                WorkspaceOutputStorage::AllocateOrAliasInputs {
                    bytes: 64,
                    inputs: vec![0],
                }
            } else {
                WorkspaceOutputStorage::AliasInput(0)
            },
            second: 0,
        });
        let storage = WorkspaceExistingStorage::new(Some(128), &context);
        let input = WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[32], WorkspaceDtype::Float32).unwrap(),
            &storage,
            &context,
        )
        .unwrap();
        context.begin_state_span([&input]).unwrap();
        let output = context
            .execute(WorkspaceOperationKind::Initialize, &[&input], layouts())
            .unwrap();
        let report = context.report(&output).unwrap();
        assert_eq!(
            report.state.as_ref().unwrap().retained_bytes,
            Some(if uncertain { 192 } else { 128 })
        );
        assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(0));
        assert_eq!(
            report.tensor_buffers.total_bytes,
            Some(if uncertain { 77 } else { 13 })
        );
        assert_eq!(
            report.tensor_buffers.retained_bytes,
            Some(if uncertain { 64 } else { 0 })
        );
        let last = output.last().unwrap().clone();
        drop(output);
        drop(input);
        drop(storage);
        assert_eq!(
            context
                .report(&[last])
                .unwrap()
                .tensor_buffers
                .retained_bytes,
            report.tensor_buffers.retained_bytes
        );
    }
}

#[test]
fn output_aliases_do_not_hide_unknown_retained_input_capacity() {
    let context = WorkspaceContext::new(SharedOutputs {
        first: WorkspaceOutputStorage::AllocateOrAliasInputs {
            bytes: 64,
            inputs: vec![0],
        },
        second: 0,
    });
    let storage = WorkspaceExistingStorage::new(None, &context);
    let input = WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[32], WorkspaceDtype::Float32).unwrap(),
        &storage,
        &context,
    )
    .unwrap();
    context.begin_state_span([&input]).unwrap();
    let mut output = context
        .execute(WorkspaceOperationKind::Initialize, &[&input], layouts())
        .unwrap();
    let last = output.pop().unwrap();
    drop(output);
    let report = context.report(&[last]).unwrap();
    assert_eq!(report.state.as_ref().unwrap().retained_bytes, None);
    assert_eq!(report.inference_transient_bytes(), None);
}

#[test]
fn invalid_output_aliases_are_rejected_before_trace_mutation() {
    for target in [1, 2, usize::MAX] {
        let context = WorkspaceContext::new(SharedOutputs {
            first: WorkspaceOutputStorage::Allocate(64),
            second: target,
        });
        assert!(context
            .execute(WorkspaceOperationKind::Initialize, &[], layouts())
            .is_err());
        let report = context.report(&[]).unwrap();
        assert!(report.operations.is_empty());
        assert_eq!(report.tensor_buffers.total_bytes, Some(0));
    }
}
