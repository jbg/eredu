use super::*;

#[test]
fn original_argmax_preserves_lazy_values_and_reclaims_its_graph_handle() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    // Ties choose the first maximum; neither row has a zero-valued maximum.
    let logits = Array::from_slice(&[-5.0f32, -1., -1., 8., 3., -2.], &[2, 3]);
    logits.evaluated().unwrap();
    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    OperationEvent::validate_traversal_leaf(&logits, &observer).unwrap();
    let baseline = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    let handler = crate::error::mlx_error_handler_state_for_test();
    crate::register_thread_runtime_housekeeping(hook);
    let hooks = Hook;
    HOOKS.with(|value| value.set(0));

    for keep_dims in [false, true] {
        let value = crate::ops::indexing::argmax_axis(&logits, -1, keep_dims, &stream).unwrap();
        assert_eq!(
            value.shape(),
            if keep_dims { &[2, 1][..] } else { &[2][..] }
        );
        assert!(original.graph.occupied_bytes() > baseline);
        assert_eq!(
            original._records.occupied_bytes(),
            records,
            "construction stays lazy"
        );
        let traversal = OperationEvent::eval_traversal_layout(OperationEvalTraversalLimits {
            roots: 1,
            arrays: 4,
            tape_entries: 3,
            input_edges: 3,
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
            value
                .completed_in_original_scope(&observer)
                .unwrap()
                .try_as_slice::<u32>()
                .unwrap(),
            &[1, 0]
        );
        crate::try_with_submission_retirement(|| drop((value, completion))).unwrap();
        settle(&observer);
        assert_eq!(original.graph.occupied_bytes(), baseline);
    }
    let invalid = crate::ops::indexing::argmax_axis(&logits, i32::MIN, false, &stream).unwrap_err();
    assert_eq!(
        invalid.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Failed)
    );
    assert_eq!(original.graph.occupied_bytes(), baseline);
    assert_eq!(original._records.occupied_bytes(), records);
    assert!(std::error::Error::source(&invalid).is_some());
    assert_eq!(HOOKS.with(Cell::get), 0);
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), handler);
    drop(hooks);
    drop((observer, original));
    assert!(
        invalid.to_string().contains("Invalid axis"),
        "the native exception survives the original scope"
    );
}

#[test]
fn original_non_array_status_preserves_source_unavailable_without_text_handler() {
    use crate::utils::guard::Guarded;
    unsafe extern "C" {
        fn _mlx_error(
            file: *const std::ffi::c_char,
            line: std::ffi::c_int,
            fmt: *const std::ffi::c_char,
            ...
        );
    }
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let original = Original::new();
    let handler = crate::error::mlx_error_handler_state_for_test();
    let error = <() as Guarded>::try_from_op(|_| {
        // SAFETY: static nul-terminated arguments; deliberately no active C++
        // exception, exercising a C status failure rather than a thrown source.
        unsafe { _mlx_error(c"fixture".as_ptr(), 1, c"fixed status".as_ptr()) };
        1
    })
    .unwrap_err();
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), handler);
    assert_eq!(
        error.scoped_evaluation_cause(),
        Some(crate::error::ScopedEvaluationCause::Failed)
    );
    drop(original);
    let source = std::error::Error::source(&error).unwrap();
    let native = std::error::Error::source(source)
        .unwrap()
        .downcast_ref::<crate::PrefillNativeError>()
        .unwrap();
    assert_eq!(
        native.kind(),
        crate::PrefillNativeFailureKind::SourceUnavailable
    );
    assert!(native.message_bytes().is_none());
}

#[test]
fn repeated_i32_input_fills_final_storage_and_retires_after_last_alias() {
    use crate::RepeatedI32InputPlan;
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let _runtime = PrefillRootsRuntime::prepare_for_stream(&stream, &stream).unwrap();
    let plan = RepeatedI32InputPlan::new(3, 4).unwrap();
    let expected = [0i32, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
    assert_eq!(plan.elements(), expected.len());
    assert_eq!(plan.requested_bytes(), size_of_val(&expected));
    assert!(RepeatedI32InputPlan::control_bytes().unwrap() > 0);
    let ordinary = plan.create().unwrap();
    assert_eq!(ordinary.dtype(), Dtype::Int32);
    assert_eq!(ordinary.shape(), &[12]);
    assert_eq!(ordinary.evaluated().unwrap().as_slice::<i32>(), &expected);
    drop(ordinary);
    for (groups, repeats) in [(0, 4), (3, 0)] {
        let empty = RepeatedI32InputPlan::new(groups, repeats)
            .unwrap()
            .create()
            .unwrap();
        assert_eq!(empty.shape(), &[0]);
        assert_eq!(empty.size(), 0);
    }

    let original = Original::new();
    let observer = OriginalScopeObserver::require_current().unwrap();
    let baseline = original.graph.occupied_bytes();
    let records = original._records.occupied_bytes();
    // All overflow refusals are cold; none may construct a native prefix.
    assert!(RepeatedI32InputPlan::new(usize::MAX, 2).is_none());
    assert!(RepeatedI32InputPlan::new(i32::MAX as usize, 2).is_none());
    assert!(RepeatedI32InputPlan::new(1, i32::MAX as usize).is_some());
    assert_eq!(original.graph.occupied_bytes(), baseline);
    let handler = crate::error::mlx_error_handler_state_for_test();
    crate::register_thread_runtime_housekeeping(hook);
    let hooks = Hook;
    HOOKS.with(|value| value.set(0));
    let value = plan.create().unwrap();
    assert_eq!(
        value
            .completed_in_original_scope(&observer)
            .unwrap()
            .try_as_slice::<i32>()
            .unwrap(),
        &expected
    );
    let alias = value.clone();
    crate::try_with_submission_retirement(|| drop(value)).unwrap();
    assert!(original.graph.occupied_bytes() > baseline);
    assert_eq!(
        alias
            .completed_in_original_scope(&observer)
            .unwrap()
            .try_as_slice::<i32>()
            .unwrap(),
        &expected
    );
    crate::try_with_submission_retirement(|| drop(alias)).unwrap();
    settle(&observer);
    assert_eq!(original.graph.occupied_bytes(), baseline);
    assert_eq!(original._records.occupied_bytes(), records);
    assert_eq!(HOOKS.with(Cell::get), 0);
    assert_eq!(crate::error::mlx_error_handler_state_for_test(), handler);
    drop(hooks);
}
