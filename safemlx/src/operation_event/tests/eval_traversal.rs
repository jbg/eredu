use super::*;

fn limits() -> OperationEvalTraversalLimits {
    OperationEvalTraversalLimits {
        roots: 1,
        arrays: 5,
        tape_entries: 3,
        input_edges: 5,
        output_slots: 3,
        streams: 1,
        captures: 8,
    }
}

#[test]
fn prepared_eval_layout_is_qualified_or_explicitly_unknown() {
    let layout = OperationEvent::eval_traversal_layout(limits());
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(
            layout.is_some(),
            "pinned validation requires the actual prepared producer"
        );
    }
    let Some(layout) = layout else {
        assert!(OperationEvent::eval_record_layout(3, 1, 3).is_none());
        assert!(
            OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
                roots: 0,
                arrays: 1,
                tape_entries: 1,
                input_edges: 0,
                output_slots: 1,
                streams: 1,
                captures: 8,
            })
            .is_none()
        );
        return;
    };
    assert_eq!(layout.roots(), 1);
    assert_eq!(layout.record_allocations(), 11);
    assert_eq!(layout.requests().len(), 11);
    assert_eq!(
        layout.requests().map(|(bytes, _)| bytes).sum::<usize>(),
        layout.record_requested_bytes()
    );
    assert!(
        layout
            .requests()
            .all(|(bytes, align)| bytes > 0 && align.is_power_of_two())
    );
    assert!(layout.query_control_bytes().unwrap() > 0);
    assert!(
        OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
            arrays: usize::MAX,
            ..limits()
        })
        .is_none()
    );
}

#[test]
fn prepared_eval_safe_consumer_keeps_real_roots_leaf_provenance_and_completion() {
    let Some(layout) = OperationEvent::eval_traversal_layout(limits()) else {
        assert_ne!(
            std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref(),
            Ok("1")
        );
        return;
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let left = Array::from_slice(&[2.0f32, 3., 5.], &[3]);
    let right = Array::from_slice(&[7.0f32, 11., 13.], &[3]);
    left.evaluated().unwrap();
    right.evaluated().unwrap();
    let _original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    OperationEvent::validate_traversal_leaf(&left, &observer).unwrap();
    OperationEvent::validate_traversal_leaf(&right, &observer).unwrap();
    let value = left.add(&right, &stream).unwrap();
    let value = value.multiply(&left, &stream).unwrap();
    assert!(OperationEvent::validate_traversal_leaf(&value, &observer).is_err());
    let completion = crate::transforms::async_eval_with_original_prepared_traversal(
        std::slice::from_ref(&value),
        &observer,
        &stream,
        &layout,
    )
    .unwrap();
    completion.synchronize().unwrap();
    OperationEvent::validate_traversal_leaf(&value, &observer).unwrap();
    assert_eq!(
        value.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[18., 42., 90.]
    );
    crate::try_with_submission_retirement(|| drop((value, completion))).unwrap();
    settle(&observer);
}

#[test]
fn resident_root_owner_refuses_short_recipe_then_completes_same_nonzero_roots() {
    let Some(short) = OperationEvent::eval_traversal_layout(limits()) else {
        assert_ne!(
            std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref(),
            Ok("1")
        );
        return;
    };
    let full = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots: 3,
        arrays: 6,
        input_edges: 7,
        ..limits()
    })
    .unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let source = Array::from_slice(&[2.0f32, -3.0, 7.0], &[3]);
    let offset = Array::from_slice(&[4.0f32], &[1]);
    source.evaluated().unwrap();
    offset.evaluated().unwrap();
    // Root storage is prepared before its carrier is bound, matching model
    // execution setup. A carrier already bound to a scope cannot be reused to
    // construct another independent root owner.
    let graph = PreparedSubmissionGraphQuota::try_new(GRAPH_CAPACITY, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let records = PreparedSubmissionRecordQuota::try_new(1 << 20, ())
        .unwrap()
        .try_allocate()
        .unwrap();
    let failure = PreparedPrefillFailure::try_new(())
        .unwrap()
        .try_allocate()
        .unwrap();
    let mut roots = crate::PrefillRoots::new_retained(&runtime, 2, &graph, &failure).unwrap();
    let mut scope = SubmissionScope::try_begin_retaining(
        PreparedSubmissionScopeOwner::try_new(())
            .unwrap()
            .with_graph_quota(graph.clone())
            .with_record_quota(records.clone()),
    )
    .unwrap();
    scope.enable_scoped_observation().unwrap();
    scope.require_original_native_controls().unwrap();
    roots.bind_scope(&scope).unwrap();
    scope.enable_original_native_controls().unwrap();
    let original = Original {
        scope,
        graph,
        _records: records,
        _failure: failure,
    };
    let observer = OriginalScopeObserver::require_current().unwrap();
    let graph_layout = OperationEvent::resident_graph_layout(5, 0, 1).unwrap();
    let graph_before = original.graph.occupied_bytes();
    let construction = OperationEvent::prepare_resident_graph(graph_layout, &observer).unwrap();
    assert!(original.graph.occupied_bytes() > graph_before);
    let value = source.add(&offset, &stream).unwrap();
    let reserved = original.graph.occupied_bytes();
    drop(construction);
    // Unused physical destinations retire, while the produced graph and C
    // handle retain their actual allocations until final value retirement.
    assert!(original.graph.occupied_bytes() < reserved);
    assert!(original.graph.occupied_bytes() > graph_before);
    roots.append(&value).unwrap();
    roots.append(&source).unwrap();
    let before = original._records.occupied_bytes();
    let failure = roots
        .complete_current_scope_on_stream_prepared(&stream, &short)
        .unwrap_err();
    assert!(matches!(
        failure,
        crate::PrefillRootsError::Refused(crate::PrefillRootsCause::Invalid)
    ));
    assert_eq!(roots.len(), 2);
    assert_eq!(original._records.occupied_bytes(), before);
    assert!(!observer.status().failed());
    // The cold maximum can exceed actual roots after descriptor deduplication.
    // Native tightens only this count, preserving all other finite capacities.
    roots
        .complete_current_scope_on_stream_prepared(&stream, &full)
        .unwrap();
    assert_eq!(
        value.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[6., 1., 11.]
    );
    assert!(matches!(
        roots.complete_current_scope_on_stream_prepared(&stream, &full),
        Err(crate::PrefillRootsError::Refused(
            crate::PrefillRootsCause::Spent
        ))
    ));
    crate::try_with_submission_retirement(|| drop((roots, value))).unwrap();
    settle(&observer);
}

#[test]
fn prepared_traversal_borrowed_iterator_checks_count_before_submission_and_keeps_values() {
    let Some(traversal) = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots: 1,
        arrays: 4,
        tape_entries: 2,
        input_edges: 3,
        output_slots: 2,
        streams: 1,
        captures: 8,
    }) else {
        assert_ne!(
            std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref(),
            Ok("1")
        );
        return;
    };
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let source = Array::from_slice(&[3.0f32, 7.0, 11.0], &[3]);
    source.evaluated().unwrap();
    let _original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let output = source.add(&source, &stream).unwrap();
    assert!(
        crate::transforms::async_eval_with_original_prepared_traversal(
            std::iter::empty::<&Array>(),
            &observer,
            &stream,
            &traversal,
        )
        .is_err()
    );
    let event = crate::transforms::async_eval_with_original_prepared_traversal(
        std::iter::once(&output),
        &observer,
        &stream,
        &traversal,
    )
    .unwrap();
    event.synchronize().unwrap();
    assert_eq!(
        output
            .completed_in_original_scope(&observer)
            .unwrap()
            .try_as_slice::<f32>()
            .unwrap(),
        &[6.0, 14.0, 22.0]
    );
    crate::try_with_submission_retirement(|| drop((event, output))).unwrap();
    settle(&observer);
}

#[test]
fn resident_graph_semantic_shells_consume_exact_bank_and_keep_independent_handles() {
    let Some(layout) = OperationEvent::resident_graph_layout_with_shells(0, 0, 1, 4, 3) else {
        assert_ne!(
            std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref(),
            Ok("1")
        );
        return;
    };
    assert_eq!(layout.additional_shells(), 3);
    assert_eq!(layout.blocks(), 3);
    let source = Array::from_slice(&[2.0f32, -3.0, 7.0], &[3]);
    source.evaluated().unwrap();
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let baseline = original.graph.occupied_bytes();
    let bank = OperationEvent::prepare_resident_graph(layout, &observer).unwrap();
    let reserved = original.graph.occupied_bytes();
    let first = source.clone();
    let second = source.clone();
    let third = first.clone();
    assert_ne!(first.as_ptr().ctx, second.as_ptr().ctx);
    assert_ne!(first.as_ptr().ctx, third.as_ptr().ctx);
    // Every successful alias consumes a preowned shell: no refill or arena
    // growth occurs, and a fourth alias cannot steal another constructor class.
    assert_eq!(original.graph.occupied_bytes(), reserved);
    use crate::utils::guard::Guarded;
    // Same fallible setter used by Array::clone, without turning its expected
    // capacity refusal into a panic in this failure-path fixture.
    let fourth = Array::try_from_op(|destination| unsafe {
        safemlx_sys::mlx_array_set(destination, source.as_ptr())
    });
    assert!(fourth.is_err());
    assert_eq!(original.graph.occupied_bytes(), reserved);
    drop(bank);
    assert!(original.graph.occupied_bytes() > baseline);
    crate::try_with_submission_retirement(|| drop((first, second, third))).unwrap();
    assert_eq!(original.graph.occupied_bytes(), baseline);
}
