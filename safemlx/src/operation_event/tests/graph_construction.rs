use super::*;

#[test]
fn pointwise_graph_query_is_qualified_or_explicitly_unknown() {
    let layout = OperationEvent::pointwise_graph_layout(2, 11);
    if std::env::var("EREDU_REQUIRE_QUALIFIED_RECORD_LAYOUT").as_deref() == Ok("1") {
        assert!(layout.is_some());
    }
    let Some(layout) = layout else {
        assert!(OperationEvent::pointwise_graph_layout(0, 0).is_none());
        return;
    };
    assert_eq!(layout.operations(), 2);
    assert_eq!(layout.maximum_rank(), 11);
    assert_eq!(layout.blocks(), 74);
    assert_eq!(layout.classes().len(), 10);
    assert_eq!(layout.classes().map(|(_, _, n)| n).sum::<usize>(), 74);
    assert_eq!(
        layout.requested_bytes(),
        layout.classes().map(|(b, _, n)| b * n).sum::<usize>()
            + layout
                .owner_requests()
                .into_iter()
                .map(|(b, _)| b)
                .sum::<usize>()
    );
    assert!(layout.control_bytes().unwrap() > 0);
    assert!(OperationEvent::pointwise_graph_layout(usize::MAX, 11).is_none());
}

#[test]
fn pointwise_graph_safe_bank_refuses_nested_then_retires_before_real_eval() {
    let Some(layout) = OperationEvent::pointwise_graph_layout(2, 1) else {
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
    let bank = OperationEvent::prepare_pointwise_graph(layout, &observer).unwrap();
    assert!(OperationEvent::prepare_pointwise_graph(layout, &observer).is_err());
    let value = left
        .add(&right, &stream)
        .unwrap()
        .multiply(&left, &stream)
        .unwrap();
    drop(bank);
    let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
        roots: 1,
        arrays: 5,
        tape_entries: 3,
        input_edges: 5,
        output_slots: 3,
        streams: 1,
        captures: 8,
    })
    .unwrap();
    let completion = crate::transforms::async_eval_with_original_prepared_traversal(
        std::slice::from_ref(&value),
        &observer,
        &stream,
        &traversal,
    )
    .unwrap();
    completion.synchronize().unwrap();
    assert_eq!(
        value.evaluated().unwrap().try_as_slice::<f32>().unwrap(),
        &[18., 42., 90.]
    );
    crate::try_with_submission_retirement(|| drop((value, completion))).unwrap();
    settle(&observer);
}
