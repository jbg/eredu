use super::*;
use crate::Tensor;

#[derive(Debug, Default)]
struct Mechanism {
    missing_native: bool,
    missing_host: bool,
}

impl WorkspaceMechanisms for Mechanism {
    fn operation_bound(
        &self,
        operation: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceOperationBound>, Error> {
        if self.missing_native {
            return Ok(None);
        }
        Ok(Some(WorkspaceOperationBound {
            outputs: operation
                .outputs
                .iter()
                .map(|layout| {
                    Ok(match operation.kind {
                        WorkspaceOperationKind::View(_) | WorkspaceOperationKind::Transpose(_) => {
                            WorkspaceOutputStorage::AliasInput(0)
                        }
                        WorkspaceOperationKind::Contiguous => {
                            WorkspaceOutputStorage::AllocateOrAliasInputs {
                                bytes: layout.bytes()?,
                                inputs: vec![0],
                            }
                        }
                        _ => WorkspaceOutputStorage::Allocate(layout.bytes()?),
                    })
                })
                .collect::<Result<_, Error>>()?,
            scratch_bytes: 5,
            assumptions: "test mechanism: five scratch bytes, explicit alias candidates".into(),
        }))
    }

    fn host_workspace_bound(
        &self,
        _: &WorkspaceOperation,
    ) -> Result<Option<WorkspaceHostBound>, Error> {
        Ok((!self.missing_host).then(|| WorkspaceHostBound {
            bytes: 3,
            assumptions: "test mechanism: three disjoint host bytes per operation".into(),
        }))
    }
}

fn view(
    context: &WorkspaceContext,
    root: &WorkspaceExistingStorage,
    elements: i32,
) -> WorkspaceTensor {
    WorkspaceTensor::existing_with_storage(
        WorkspaceLayout::new(&[elements], WorkspaceDtype::Float32).unwrap(),
        root,
        context,
    )
    .unwrap()
}

#[test]
fn borrowed_selection_validates_atomically_and_pins_exact_metadata_roots() {
    let context = WorkspaceContext::new(Mechanism::default());
    let other = WorkspaceContext::new(Mechanism::default());
    let first = WorkspaceExistingStorage::new(Some(64), &context);
    let alias = first.clone();
    let second = WorkspaceExistingStorage::new(Some(64), &context);
    let unknown = WorkspaceExistingStorage::new(None, &context);
    let foreign = WorkspaceExistingStorage::new(Some(64), &other);
    let huge = WorkspaceExistingStorage::new(Some(u64::MAX), &context);
    assert!(first.same_storage(&alias));
    assert!(!first.same_storage(&second));
    assert_eq!(first.capacity_bytes(), Some(64));
    assert_eq!(unknown.capacity_bytes(), None);
    for invalid in [&unknown, &foreign, &huge] {
        assert!(WorkspaceBorrowedStorage::new(&context, [&first, invalid]).is_err());
        assert!(context.report(&[]).unwrap().residual.is_none());
    }
    let selected = WorkspaceBorrowedStorage::new(&context, [&first, &alias, &second]).unwrap();
    assert_eq!(selected.roots().len(), 2);
    assert_eq!(selected.total_bytes(), Some(128));
    assert!(selected.roots()[0].same_storage(&first));
    assert!(selected.same_identity(&selected.clone()));
    assert!(!selected
        .same_identity(&WorkspaceBorrowedStorage::new(&context, [&first, &second]).unwrap()));
    let wrong_context = WorkspaceBorrowedStorage::new(&other, [&foreign]).unwrap();
    assert!(context.set_borrowed_storage(wrong_context).is_err());
    context.set_borrowed_storage(selected.clone()).unwrap();
    assert!(context.set_borrowed_storage(selected.clone()).is_err());

    let first_view = view(&context, &first, 4);
    let weak = Rc::downgrade(&first_view.storage);
    drop((first, alias, second, first_view, context));
    assert!(weak.upgrade().is_some());
    assert_eq!(selected.roots()[0].capacity_bytes(), Some(64));
    drop(selected);
    assert!(weak.upgrade().is_none());
}

#[test]
fn residual_union_traverses_possible_aliases_and_keeps_equal_capacity_roots_distinct() {
    for borrow_actual in [false, true] {
        let context = WorkspaceContext::new(Mechanism::default());
        let old = WorkspaceExistingStorage::new(Some(128), &context);
        let unrelated = WorkspaceExistingStorage::new(Some(128), &context);
        let fixed = WorkspaceExistingStorage::new(Some(64), &context);
        let selected = WorkspaceBorrowedStorage::new(
            &context,
            [if borrow_actual { &old } else { &unrelated }],
        )
        .unwrap();
        context.set_borrowed_storage(selected.clone()).unwrap();
        let old = view(&context, &old, 8);
        let fixed = view(&context, &fixed, 4);
        context.begin_state_span([&old, &fixed]).unwrap();
        let candidate = old.contiguous(&context).unwrap();
        let alias = candidate.reshape(&[2, 4], &context).unwrap();
        let copy = alias.deep_copy(&context).unwrap();
        let report = context.report(&[copy.clone(), copy, alias, fixed]).unwrap();
        // Full reports are identical with either selection.
        assert_eq!(report.total_bytes, Some(88));
        assert_eq!(report.retained_bytes, Some(64));
        assert_eq!(report.inference_transient_bytes(), Some(24));
        assert_eq!(report.state.as_ref().unwrap().retained_bytes, Some(256));
        let residual = report.residual.unwrap();
        assert!(residual.borrowed_storage.same_identity(&selected));
        assert_eq!(
            residual.total_bytes,
            Some(if borrow_actual { 152 } else { 280 })
        );
        assert_eq!(
            residual.retained_bytes,
            Some(if borrow_actual { 128 } else { 256 })
        );
        assert_eq!(residual.displaced_bytes, Some(0));
        assert_eq!(residual.transient_bytes, Some(24));
        assert_eq!(
            residual.opening_storage,
            Some(WorkspaceStoragePopulation {
                bytes: Some(if borrow_actual { 64 } else { 192 }),
                maximum_allocations: if borrow_actual { 1 } else { 2 },
            })
        );
        assert_eq!(
            residual.closing_storage,
            Some(WorkspaceStoragePopulation {
                bytes: Some(if borrow_actual { 128 } else { 256 }),
                maximum_allocations: if borrow_actual { 3 } else { 4 },
            })
        );
    }
}

#[test]
fn replacement_checkpoint_and_rollback_copies_stay_incremental_across_span_resets() {
    let context = WorkspaceContext::new(Mechanism::default());
    let root = WorkspaceExistingStorage::new(Some(128), &context);
    let selected = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
    context.set_borrowed_storage(selected.clone()).unwrap();
    let old = view(&context, &root, 8);
    context.begin_state_span([&old]).unwrap();
    let replacement = old.square(&context).unwrap();
    let checkpoint = old.deep_copy(&context).unwrap();
    let _rollback = checkpoint.deep_copy(&context).unwrap();
    let report = context.report(&[replacement.clone()]).unwrap();
    assert_eq!(report.state.as_ref().unwrap().displaced_bytes, Some(128));
    assert_eq!(report.inference_transient_bytes(), Some(216));
    let residual = report.residual.unwrap();
    assert_eq!(residual.total_bytes, Some(120));
    assert_eq!(residual.retained_bytes, Some(32));
    assert_eq!(residual.displaced_bytes, Some(0));
    assert_eq!(residual.transient_bytes, Some(88));

    let lane = context.clone();
    lane.begin_state_span([&replacement]).unwrap();
    let next = replacement.square(&lane).unwrap();
    let residual = context.report(&[next]).unwrap().residual.unwrap();
    assert!(residual.borrowed_storage.same_identity(&selected));
    // The original 128-byte root is no longer in this union. It cannot be
    // subtracted from the 72 bytes owned by this later span.
    assert_eq!(residual.total_bytes, Some(72));
    assert_eq!(residual.retained_bytes, Some(32));
    assert_eq!(residual.displaced_bytes, Some(32));
    assert_eq!(residual.transient_bytes, Some(40));
    lane.begin_span();
    let residual = context.report(&[]).unwrap().residual.unwrap();
    assert!(residual.borrowed_storage.same_identity(&selected));
    assert_eq!(residual.total_bytes, None, "state was not seeded");
    assert_eq!(residual.opening_storage, None);
    assert_eq!(residual.closing_storage, None);
    context.begin_state_span([]).unwrap();
    let residual = context.report(&[]).unwrap().residual.unwrap();
    assert!(residual.borrowed_storage.same_identity(&selected));
    assert_eq!(residual.total_bytes, Some(0));
    assert_eq!(
        residual.opening_storage,
        Some(WorkspaceStoragePopulation::EMPTY)
    );
    assert_eq!(
        residual.closing_storage,
        Some(WorkspaceStoragePopulation::EMPTY)
    );
}

#[test]
fn explicit_empty_selection_reports_full_union_and_missing_selection_stays_absent() {
    let context = WorkspaceContext::new(Mechanism::default());
    let existing = super::tests::existing_f32(&[8], &context).unwrap();
    assert!(context
        .report(&[existing.clone()])
        .unwrap()
        .residual
        .is_none());
    let empty = WorkspaceBorrowedStorage::new(&context, []).unwrap();
    context.set_borrowed_storage(empty.clone()).unwrap();
    assert_eq!(empty.total_bytes(), Some(0));
    context.begin_state_span([&existing]).unwrap();
    let closing_only = super::tests::existing_f32(&[4], &context).unwrap();
    let report = context.report(&[existing, closing_only]).unwrap();
    assert_eq!(report.total_bytes, Some(0));
    assert_eq!(report.state.unwrap().retained_bytes, Some(48));
    let residual = report.residual.unwrap();
    assert_eq!(residual.total_bytes, Some(48));
    assert_eq!(residual.retained_bytes, Some(48));
    assert_eq!(residual.transient_bytes, Some(0));
}

#[test]
fn residual_diagnostics_preserve_missing_native_host_state_and_capacity_facts() {
    for (missing_native, missing_host) in [(true, false), (false, true)] {
        let context = WorkspaceContext::new(Mechanism {
            missing_native,
            missing_host,
        });
        let root = WorkspaceExistingStorage::new(Some(32), &context);
        context
            .set_borrowed_storage(WorkspaceBorrowedStorage::new(&context, [&root]).unwrap())
            .unwrap();
        let old = view(&context, &root, 8);
        context.begin_state_span([&old]).unwrap();
        let next = old.square(&context).unwrap();
        let report = context.report(&[next]).unwrap();
        assert_eq!(
            report.unpriced_operations,
            if missing_native { vec![0] } else { vec![] }
        );
        assert_eq!(
            report.unpriced_host_operations,
            if missing_host { vec![0] } else { vec![] }
        );
        let residual = report.residual.unwrap();
        assert_eq!(residual.total_bytes, None);
        assert_eq!(residual.transient_bytes, None);
    }
    let context = WorkspaceContext::new(Mechanism::default());
    let unknown = WorkspaceExistingStorage::new(None, &context);
    context
        .set_borrowed_storage(WorkspaceBorrowedStorage::new(&context, []).unwrap())
        .unwrap();
    let old = view(&context, &unknown, 8);
    context.begin_state_span([&old]).unwrap();
    let residual = context.report(&[old]).unwrap().residual.unwrap();
    assert_eq!(residual.total_bytes, None);
    assert_eq!(residual.retained_bytes, None);
    let unknown = Some(WorkspaceStoragePopulation {
        bytes: None,
        maximum_allocations: 1,
    });
    assert_eq!(residual.opening_storage, unknown);
    assert_eq!(residual.closing_storage, unknown);
}

#[test]
fn borrowed_selection_cannot_change_after_tracing_and_failed_validation_is_atomic() {
    for begin_state in [false, true] {
        let context = WorkspaceContext::new(Mechanism::default());
        let root = WorkspaceExistingStorage::new(Some(32), &context);
        let token = WorkspaceBorrowedStorage::new(&context, [&root]).unwrap();
        if begin_state {
            context.begin_state_span([]).unwrap();
        } else {
            WorkspaceTensor::full_f32(1.0, &[1], &context).unwrap();
        }
        let before = context.report(&[]).unwrap();
        assert!(context.set_borrowed_storage(token).is_err());
        let after = context.report(&[]).unwrap();
        assert!(after.residual.is_none());
        assert_eq!(after.total_bytes, before.total_bytes);
        assert_eq!(after.operations.len(), before.operations.len());
    }
    let context = WorkspaceContext::new(Mechanism::default());
    let foreign = WorkspaceContext::new(Mechanism::default());
    let foreign_value = super::tests::existing_f32(&[1], &foreign).unwrap();
    assert!(context.begin_state_span([&foreign_value]).is_err());
    assert!(foreign_value.square(&context).is_err());
    context
        .set_borrowed_storage(WorkspaceBorrowedStorage::new(&context, []).unwrap())
        .unwrap();
    assert!(context.report(&[]).unwrap().operations.is_empty());
}
