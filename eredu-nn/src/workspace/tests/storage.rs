use super::*;

#[test]
fn rectangular_update_prices_destination_and_retains_checked_absolute_starts() {
    let context = context();
    let destination = existing_f32(&[2, 256, 8], &context).unwrap();
    let update = existing_f32(&[2, 3, 8], &context).unwrap();
    let result = destination
        .update_slice(&update, &[0, 251, 0], &context)
        .unwrap();
    let report = context.report(&[result.clone()]).unwrap();
    assert_eq!(result.shape(), [2, 256, 8]);
    assert_eq!(report.retained_bytes, Some(2 * 256 * 8 * 4));
    assert_eq!(report.total_bytes, Some(2 * 256 * 8 * 4 + 7));
    assert!(
        matches!(&report.operations[0].kind, WorkspaceOperationKind::SliceUpdate { starts } if starts == &[0,251,0])
    );
    assert_eq!(report.operations[0].inputs[1].shape(), [2, 3, 8]);
}

#[test]
fn invalid_update_geometry_and_foreign_storage_fail_before_recording_work() {
    let context = context();
    let foreign = WorkspaceContext::new(AllocatingMechanism);
    let destination = existing_f32(&[2, 256, 8], &context).unwrap();
    let update = existing_f32(&[2, 3, 8], &context).unwrap();
    for starts in [
        vec![0, 254, 0],
        vec![0, -1, 0],
        vec![0, i32::MAX, 0],
        vec![0, 0],
    ] {
        assert!(destination
            .update_slice(&update, &starts, &context)
            .is_err());
    }
    let wrong_type = existing_i32(&[2, 3, 8], &context).unwrap();
    assert!(destination
        .update_slice(&wrong_type, &[0, 0, 0], &context)
        .is_err());
    let wrong_rank = existing_f32(&[3, 8], &context).unwrap();
    assert!(destination
        .update_slice(&wrong_rank, &[0, 0, 0], &context)
        .is_err());
    let wrong_context = existing_f32(&[2, 3, 8], &foreign).unwrap();
    assert!(destination
        .update_slice(&wrong_context, &[0, 0, 0], &context)
        .is_err());
    assert!(context.report(&[]).unwrap().operations.is_empty());
}

#[test]
fn compact_and_independent_copies_have_separate_operations_and_retained_storage() {
    let context = context();
    let input = existing_f32(&[2, 256, 8], &context).unwrap();
    let logical = input
        .index(&[Index::Full, Index::Range(0, 3)], &context)
        .unwrap();
    let contiguous = logical.contiguous(&context).unwrap();
    let copy = contiguous.deep_copy(&context).unwrap();
    let report = context.report(&[contiguous, copy.clone(), copy]).unwrap();
    assert_eq!(report.retained_bytes, Some(2 * 2 * 3 * 8 * 4));
    assert!(matches!(
        report.operations[1].kind,
        WorkspaceOperationKind::Contiguous
    ));
    assert!(matches!(
        report.operations[2].kind,
        WorkspaceOperationKind::DeepCopy
    ));
    assert_eq!(report.transient_bytes, Some(14));
}

#[test]
fn storage_descriptors_cannot_substitute_known_logical_sizes_for_missing_facts() {
    #[derive(Debug)]
    struct Unknown;
    impl WorkspaceMechanisms for Unknown {
        fn operation_bound(
            &self,
            _: &WorkspaceOperation,
        ) -> Result<Option<WorkspaceOperationBound>, Error> {
            Ok(None)
        }
    }
    let context = WorkspaceContext::new(Unknown);
    let input = existing_f32(&[2, 256, 8], &context).unwrap();
    let update = existing_f32(&[2, 3, 8], &context).unwrap();
    let copied = input
        .update_slice(&update, &[0, 7, 0], &context)
        .unwrap()
        .contiguous(&context)
        .unwrap()
        .deep_copy(&context)
        .unwrap();
    let report = context.report(&[copied]).unwrap();
    assert_eq!(report.total_bytes, None);
    assert_eq!(report.unpriced_operations, [0, 1, 2]);
    assert_eq!(report.unpriced_host_operations, [0, 1, 2]);
}
