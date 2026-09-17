use super::*;

#[test]
fn prepared_invocation_keeps_physical_width_and_shared_slice_payload_validation() {
    let action = InterventionAction::Scale {
        dtype: InterventionDtype::Float32,
        factor: -0.5,
    };
    let mut op = operation(action);
    op.slices.push(CaptureSlice {
        axis: "hidden".into(),
        start: 0,
        end: 2,
        stride: 2,
    });
    let admitted = plan(vec![op])
        .admit_invocations(
            &discovery(false),
            CaptureInvocationBounds {
                batch: 1,
                max_sequence: 3,
                max_context: None,
                max_predictions: 4,
            },
            "session-a",
        )
        .unwrap();
    // Decode logical index 2 still has three actual verification rows.
    let invocation = CaptureInvocationShape {
        batch: 1,
        sequence: 3,
        context: None,
    };
    let shape = [1, 3, 2];
    let expected = admitted
        .validate_at(
            0,
            CapturePhase::Decode,
            2,
            Some(invocation),
            &shape,
            Some(InterventionDtype::Float32),
        )
        .unwrap();
    let mut actual = ResolvedCaptureSlice {
        starts: vec![0; 3],
        ends: vec![0; 3],
        strides: vec![0; 3],
        shape: vec![0; 3],
    };
    admitted
        .resolve_prepared_invocation_at(
            0,
            CapturePhase::Decode,
            2,
            invocation,
            &shape,
            InterventionDtype::Float32,
            &mut actual,
        )
        .unwrap();
    assert_eq!(actual, expected);
    assert!(
        admitted
            .resolve_prepared_at(
                0,
                CapturePhase::Decode,
                2,
                &shape,
                InterventionDtype::Float32,
                &mut actual
            )
            .is_err()
    );
    assert!(
        admitted
            .resolve_prepared_invocation_at(
                0,
                CapturePhase::Decode,
                2,
                CaptureInvocationShape {
                    sequence: 2,
                    ..invocation
                },
                &shape,
                InterventionDtype::Float32,
                &mut actual
            )
            .is_err()
    );
    assert!(
        admitted
            .resolve_prepared_invocation_at(
                0,
                CapturePhase::Decode,
                2,
                invocation,
                &shape,
                InterventionDtype::Float16,
                &mut actual
            )
            .is_err()
    );
    let ordinary = admit(operation(InterventionAction::Scale {
        dtype: InterventionDtype::Float32,
        factor: -0.5,
    }))
    .unwrap();
    assert!(
        ordinary
            .resolve_prepared_invocation_at(
                0,
                CapturePhase::Prefill,
                0,
                invocation,
                &shape,
                InterventionDtype::Float32,
                &mut actual
            )
            .is_err()
    );
}
