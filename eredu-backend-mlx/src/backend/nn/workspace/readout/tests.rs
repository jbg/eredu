use super::*;
use eredu_nn::{multimodal::MaskedOutputProjectionInput, Tensor};

fn mechanisms() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 16384,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    }
}
fn operation([b, s, h, v, c, top]: [i32; 6], unsigned: bool) -> WorkspaceOperation {
    let f = |shape: &[i32]| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
    WorkspaceOperation {
        kind: WorkspaceOperationKind::MaskedOutputProjection {
            top_centroids: top,
            mask_margin: 1.,
        },
        inputs: vec![
            f(&[b, s, h]),
            f(&[v, h]),
            f(&[b, s, c]),
            WorkspaceLayout::new(
                &[v],
                if unsigned {
                    WorkspaceDtype::Uint32
                } else {
                    WorkspaceDtype::Int32
                },
            )
            .unwrap(),
        ],
        outputs: vec![f(&[b, s, v])],
    }
}
fn total(op: &WorkspaceOperation, m: MlxMetalWorkspaceMechanisms) -> u64 {
    let bound = m.operation_bound(op).unwrap().unwrap();
    let WorkspaceOutputStorage::Allocate(out) = bound.outputs[0] else {
        panic!("readout must own its output");
    };
    bound.scratch_bytes + out
}

#[test]
fn masked_readout_trace_prices_selected_replicas_and_retains_only_scores() {
    for dims in [
        [2, 7, 65, 24, 4, 2],
        [1, 1, 1025, 16, 4, 1],
        [2, 0, 7, 8, 2, 2],
        [0, 3, 7, 8, 2, 1],
    ] {
        for unsigned in [false, true] {
            let op = operation(dims, unsigned);
            let context = WorkspaceContext::new(mechanisms());
            let inputs = op
                .inputs
                .iter()
                .map(|l| WorkspaceTensor::existing(l.clone(), &context).unwrap())
                .collect::<Vec<_>>();
            let output = WorkspaceTensor::masked_output_projection(
                MaskedOutputProjectionInput {
                    hidden: &inputs[0],
                    output_weight: &inputs[1],
                    centroid_logits: &inputs[2],
                    token_ordering: &inputs[3],
                    top_centroids: dims[5],
                    mask_margin: 1.,
                },
                &context,
            )
            .unwrap();
            assert_eq!(output.layout(), &op.outputs[0]);
            let report = context.report(&[output]).unwrap();
            assert_eq!(
                report.tensor_buffers.total_bytes,
                Some(total(&op, mechanisms()))
            );
            assert_eq!(
                report.tensor_buffers.retained_bytes,
                Some(capacity(mechanisms().allocation, op.outputs[0].elements().unwrap()).unwrap())
            );
            assert_eq!(report.host_workspace_bytes, Some(0));
        }
    }
    let small = operation([2, 7, 65, 24, 4, 1], false);
    let large = operation([2, 7, 65, 24, 4, 4], false);
    assert!(total(&large, mechanisms()) > total(&small, mechanisms()));
    assert!(
        total(&large, mechanisms()) >= capacity(mechanisms().allocation, 2 * 7 * 24 * 65).unwrap()
    );
}

#[test]
fn masked_readout_rejects_invalid_geometry_and_leaves_unsupported_dtypes_unknown() {
    let valid = operation([2, 3, 7, 8, 2, 1], false);
    for (top, margin) in [(0, 1.), (3, 1.), (1, 0.), (1, f32::NAN)] {
        let mut op = valid.clone();
        op.kind = WorkspaceOperationKind::MaskedOutputProjection {
            top_centroids: top,
            mask_margin: margin,
        };
        assert!(mechanisms().operation_bound(&op).is_err());
    }
    for shape in [vec![], vec![2, 3], vec![2, 3, 0], vec![2, 4, 7]] {
        let mut op = valid.clone();
        op.inputs[0] = WorkspaceLayout::new(&shape, WorkspaceDtype::Float32).unwrap();
        assert!(mechanisms().operation_bound(&op).is_err());
    }
    for index in 0..4 {
        let mut op = valid.clone();
        op.inputs[index] =
            WorkspaceLayout::new(op.inputs[index].shape(), WorkspaceDtype::Bool).unwrap();
        assert!(mechanisms().operation_bound(&op).unwrap().is_none());
        assert!(mechanisms().host_workspace_bound(&op).unwrap().is_none());
    }
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
mod native;
