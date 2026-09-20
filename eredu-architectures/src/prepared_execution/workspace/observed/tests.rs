use super::*;
use eredu_nn::workspace::*;

#[derive(Debug)]
struct Facts {
    known: bool,
    alias: bool,
    host: bool,
}
impl WorkspaceMechanisms for Facts {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if !self.known {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| {
                    if self.alias {
                        Ok(WorkspaceOutputStorage::AliasInput(0))
                    } else {
                        layout.bytes().map(WorkspaceOutputStorage::Allocate)
                    }
                })
                .collect::<Result<_, _>>()?,
            scratch_bytes: 7,
            assumptions: "test exact alias/allocation with seven temporary bytes".into(),
        }))
    }
    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok(self.host.then(|| WorkspaceHostBound {
            bytes: 3,
            assumptions: "test three-byte host staging".into(),
        }))
    }
}

#[test]
fn final_row_selection_keeps_full_alias_backing_and_unknown_mechanism_facts() {
    for known in [false, true] {
        for alias in [false, true] {
            let context = WorkspaceContext::new(Facts {
                known,
                alias,
                host: true,
            });
            let backing = WorkspaceExistingStorage::new(Some(8192), &context);
            let scores = WorkspaceTensor::existing_with_storage(
                WorkspaceLayout::new(&[1, 5, 17], WorkspaceDtype::Float32).unwrap(),
                &backing,
                &context,
            )
            .unwrap();
            context.begin_state_span([&scores]).unwrap();
            let row = sampling_row(&scores, &context).unwrap();
            assert_eq!(row.shape(), [1, 17]);
            assert!(row.same_context(&scores));
            let report = context.report(&[row]).unwrap();
            assert_eq!(report.operations.len(), 2);
            let WorkspaceOperationKind::StaticSlice { starts, ends, strides } =
                &report.operations[0].kind else { panic!("missing selected interval") };
            assert_eq!(starts, &[0, 4, 0]);
            assert_eq!(ends, &[1, 5, 17]);
            assert_eq!(strides, &[1, 1, 1]);
            assert!(matches!(report.operations[1].kind, WorkspaceOperationKind::View("squeeze")));
            assert_eq!(
                report.state.unwrap().retained_bytes,
                if !known {
                    None
                } else if alias {
                    Some(8192)
                } else {
                    Some(68)
                }
            );
            assert_eq!(report.total_bytes.is_some(), known);
        }
    }
    let context = WorkspaceContext::new(Facts {
        known: true,
        alias: false,
        host: true,
    });
    for shape in [&[1, 17][..], &[1, 0, 17][..], &[1, 1, 1, 17][..]] {
        let scores = WorkspaceTensor::existing(
            WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap(),
            &context,
        )
        .unwrap();
        assert!(sampling_row(&scores, &context).is_err());
    }
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn prepared_hook_overlap_preserves_residual_identity_unknowns_and_decoder_only_state() {
    for host in [false, true] {
        let context = WorkspaceContext::new(Facts {
            known: true,
            alias: false,
            host,
        });
        let source = WorkspaceExistingStorage::new(Some(128), &context);
        let borrowed = WorkspaceBorrowedStorage::new(&context, [&source]).unwrap();
        context.set_borrowed_storage(borrowed.clone()).unwrap();
        let input = WorkspaceTensor::existing_with_storage(
            WorkspaceLayout::new(&[1, 4], WorkspaceDtype::Float32).unwrap(),
            &source,
            &context,
        )
        .unwrap();
        context.begin_state_span([&input]).unwrap();
        let output = input.tanh(&context).unwrap();
        let before = context.report(&[output]).unwrap();
        // The production extent comes only from the prepared borrowed hook.
        // This unit isolates arithmetic by feeding the concrete inline layout.
        let extra = (std::mem::size_of::<&mut dyn InferenceWorkspaceObserver>()
            + std::mem::size_of::<&SharedLayeredObservationPaths>()) as u64
            * 2;
        let report = with_hook_workspace(before.clone(), extra, &context).unwrap();
        assert_eq!(
            report.state.as_ref().unwrap().retained_bytes,
            before.state.as_ref().unwrap().retained_bytes
        );
        assert_eq!(
            report.tensor_buffers.total_bytes,
            before.tensor_buffers.total_bytes
        );
        assert_eq!(
            report.host_workspace_bytes,
            if host { Some(3 + extra) } else { None }
        );
        assert_eq!(
            report.inference_transient_bytes(),
            before
                .inference_transient_bytes()
                .map(|bytes| bytes + extra)
        );
        let residual = report.residual.unwrap();
        assert!(residual.borrowed_storage.same_identity(&borrowed));
        assert_eq!(
            residual.total_bytes,
            before
                .residual
                .as_ref()
                .unwrap()
                .total_bytes
                .map(|bytes| bytes + extra)
        );
        if host {
            assert!(with_hook_workspace(before, u64::MAX, &context).is_err());
        }
    }
}
