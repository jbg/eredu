use super::*;
use crate::working_memory::{InferenceExecutionIdentity, MemoryLedger};
use eredu_nn::workspace::WorkspaceContext;

#[test]
fn window_payloads_share_stride_bits_and_keep_success_and_partial_error_funding() {
    let existing = 17;
    let capacity = 1 << 20;
    let pool = crate::working_memory::memory_fixture::host_ledger(capacity, existing).unwrap();
    let execution = InferenceExecutionIdentity::default();
    let funding = pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(&pool, capacity),
        )
        .unwrap();
    let initial = pool.payload_used_bytes().unwrap();
    let destination = ResolvedCaptureSlice {
        starts: vec![0, 1, 0],
        ends: vec![1, 3, 3],
        strides: vec![1, 1, 2],
        shape: vec![1, 2, 2],
    };
    let indices = [3, 5, 6, 8];
    let float = [-0.0f32, 1.25, -2.0, 3.5, 4.0, -5.25, 6.0, 7.5, -8.0];
    let half = [
        0x8000u16, 0x3c01, 0x0001, 0x7bff, 0xbc00, 0x3555, 0x0400, 0x3bff, 0x1234,
    ];
    let bfloat = [
        0x8000u16, 0x3f81, 0x0001, 0x7f7f, 0xbf80, 0x3eab, 0x0080, 0x3f7f, 0x1234,
    ];
    let mask = [true, false, true, false, true, false, true, false, true];
    let actions = [
        InterventionAction::Add {
            tensor: InterventionTensor {
                shape: vec![1, 3, 3],
                values: InterventionValues::Float32(float.to_vec()),
            },
        },
        InterventionAction::Replace {
            tensor: InterventionTensor {
                shape: vec![1, 3, 3],
                values: InterventionValues::Float16(half.to_vec()),
            },
        },
        InterventionAction::Add {
            tensor: InterventionTensor {
                shape: vec![1, 3, 3],
                values: InterventionValues::Bfloat16(bfloat.to_vec()),
            },
        },
        InterventionAction::Mask {
            dtype: InterventionDtype::Float32,
            shape: vec![1, 3, 3],
            keep: mask.to_vec(),
        },
    ];
    let mut escaped = Vec::new();
    for action in &actions {
        let ordinary = super::super::project_action(action, &destination).unwrap();
        let owner =
            PreparedWindowInterventionPayload::prepare(action, &destination, funding.clone())
                .unwrap();
        let projected = owner.projected_action().unwrap();
        assert_eq!(projected, &ordinary);
        match projected {
            InterventionAction::Replace { tensor } | InterventionAction::Add { tensor } => {
                assert_eq!(tensor.shape, [1, 2, 2]);
                match &tensor.values {
                    InterventionValues::Float32(values) => assert_eq!(
                        values
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        indices.map(|i| float[i].to_bits())
                    ),
                    InterventionValues::Float16(values) => {
                        assert_eq!(values.as_slice(), indices.map(|i| half[i]).as_slice())
                    }
                    InterventionValues::Bfloat16(values) => {
                        assert_eq!(values.as_slice(), indices.map(|i| bfloat[i]).as_slice())
                    }
                }
            }
            InterventionAction::Mask { keep, .. } => {
                assert_eq!(keep.as_slice(), indices.map(|i| mask[i]).as_slice())
            }
            _ => panic!("payload action changed"),
        }
        escaped.push(owner);
    }
    let mut bad = destination.clone();
    bad.ends[1] = 4;
    let rejected =
        PreparedWindowInterventionPayload::prepare(&actions[0], &bad, funding.clone()).unwrap_err();
    assert!(matches!(
        rejected.cause,
        Cause::Geometry(WindowInterventionPayloadError::Destination)
    ));
    assert!(pool.payload_used_bytes().unwrap() > initial);
    let charged = pool.payload_used_bytes().unwrap();
    drop(actions);
    drop(funding);
    drop(escaped);
    assert_eq!(
        pool.payload_used_bytes().unwrap(),
        charged,
        "failure keeps the same cumulative account after payload retirement"
    );
    drop(rejected);
    assert_eq!(pool.payload_used_bytes().unwrap(), existing);

    // Admit exactly the queried controls and the first (shape) vector. The
    // second (bool payload) constructor must refuse after that actual allocation.
    let partial_capacity = initial
        + PreparedWindowInterventionPayload::control_bytes().unwrap() as u64
        + WorkspaceContext::metadata_vec_bytes::<u64>(3).unwrap() as u64;
    let partial_pool =
        crate::working_memory::memory_fixture::host_ledger(partial_capacity, existing).unwrap();
    let funding = partial_pool
        .prepare_workspace_metadata(
            &execution,
            crate::working_memory::memory_fixture::resolved_host_limits(
                &partial_pool,
                partial_capacity,
            ),
        )
        .unwrap();
    assert_eq!(partial_pool.payload_used_bytes().unwrap(), initial);
    let action = InterventionAction::Mask {
        dtype: InterventionDtype::Float32,
        shape: vec![1, 3, 3],
        keep: mask.to_vec(),
    };
    let error =
        PreparedWindowInterventionPayload::prepare(&action, &destination, funding).unwrap_err();
    assert!(matches!(error.cause, Cause::Metadata(_)));
    assert_eq!(partial_pool.payload_used_bytes().unwrap(), partial_capacity);
    drop(action);
    assert_eq!(partial_pool.payload_used_bytes().unwrap(), partial_capacity);
    drop(error);
    assert_eq!(partial_pool.payload_used_bytes().unwrap(), existing);
}
