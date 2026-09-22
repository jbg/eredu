use super::*;
use eredu_nn::Tensor;

fn operation(input: &[i32], mask: &[i32], source: &[i32]) -> WorkspaceOperation {
    let f = |shape| WorkspaceLayout::new(shape, WorkspaceDtype::Float32).unwrap();
    WorkspaceOperation {
        kind: WorkspaceOperationKind::Elementwise("masked_scatter"),
        inputs: vec![
            f(input),
            WorkspaceLayout::new(mask, WorkspaceDtype::Bool).unwrap(),
            f(source),
        ],
        outputs: vec![f(input)],
    }
}

#[test]
fn masked_scatter_prices_copies_offsets_and_full_source_cast() {
    let mechanism = MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 4096,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    };
    let op = operation(&[1, 17, 16], &[1, 17], &[4, 16]);
    let bound = mechanism.operation_bound(&op).unwrap().unwrap();
    // Below one page, the actual allocator's possible reused capacity is
    // 2*request-1: output1088, cast256, mask272, offsets1088 bytes.
    assert_eq!(bound.outputs, [WorkspaceOutputStorage::Allocate(2175)]);
    assert_eq!(bound.scratch_bytes, 511 + 543 + 2175);
    assert_eq!(
        mechanism.host_workspace_bound(&op).unwrap().unwrap().bytes,
        0
    );
    let context = WorkspaceContext::new(mechanism);
    let tensors = op
        .inputs
        .iter()
        .map(|layout| WorkspaceTensor::existing(layout.clone(), &context).unwrap())
        .collect::<Vec<_>>();
    let output = tensors[0]
        .masked_scatter(&tensors[1], &tensors[2], &context)
        .unwrap();
    let report = context.report(&[output]).unwrap();
    assert_eq!(
        report.total_bytes,
        Some(
            2175 + 511
                + 543
                + 2175
                + crate::backend::nn::workspace::test_backing_controls(&mechanism, &report)
        )
    );
    assert_eq!(report.host_workspace_bytes, Some(0));
    // Unused source rows remain part of the actual conversion allocation.
    let larger = operation(&[1, 17, 16], &[1, 17], &[64, 16]);
    assert!(
        mechanism
            .operation_bound(&larger)
            .unwrap()
            .unwrap()
            .scratch_bytes
            > bound.scratch_bytes
    );
}

#[test]
fn masked_scatter_rejects_bad_shapes_and_unrepresentable_metal_scan_domains() {
    let allocation = NativeAllocationFacts {
        page_size: 4096,
        cpu_header: false,
        original_storage: false,
    };
    for op in [
        operation(&[1, 17, 16], &[17, 16], &[4, 16]),
        operation(&[1, 17, 16], &[1, 17], &[4, 15]),
    ] {
        assert!(emit(op.as_view(), allocation, &mut Emitter::count()).is_err());
    }
    let op = operation(&[65536, 65536], &[65536], &[1, 65536]);
    assert!(emit(op.as_view(), allocation, &mut Emitter::count())
        .unwrap()
        .is_none());
    let op = operation(&[0, 16], &[0], &[0, 16]);
    let bound = emit(op.as_view(), allocation, &mut Emitter::count())
        .unwrap()
        .unwrap();
    assert_eq!(bound.scratch_bytes, 0);
}

#[test]
fn masked_scatter_scalar_follows_destination_cast_without_inventing_evidence() {
    let mechanism = MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 4096,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    };
    for destination in [
        WorkspaceFloatingType::Float32,
        WorkspaceFloatingType::Float16,
        WorkspaceFloatingType::Bfloat16,
    ] {
        for source in [
            None,
            Some(WorkspaceFloatingType::Float32),
            Some(WorkspaceFloatingType::Float16),
            Some(WorkspaceFloatingType::Bfloat16),
        ] {
            let mut op = operation(&[1, 3, 2], &[1, 3], &[2, 2]);
            op.inputs[0] = op.inputs[0]
                .clone()
                .with_representation(Some(WorkspaceRepresentation::new(destination, false)));
            op.inputs[2] = op.inputs[2].clone().with_representation(
                source.map(|scalar| WorkspaceRepresentation::new(scalar, false)),
            );
            assert_eq!(
                mechanism.output_representation(op.as_view(), 0),
                Some(WorkspaceRepresentation::new(destination, false))
            );
            op.inputs[0] = op.inputs[0].clone().with_representation(None);
            assert!(mechanism.output_representation(op.as_view(), 0).is_none());
        }
    }
    let mut op = operation(&[1, 3, 2], &[1, 3], &[2, 3]);
    op.inputs[0] = op.inputs[0]
        .clone()
        .with_representation(Some(WorkspaceRepresentation::new(
            WorkspaceFloatingType::Float32,
            false,
        )));
    assert!(mechanism.output_representation(op.as_view(), 0).is_none());
}

#[test]
#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[ignore = "native CPU masked-scatter dtype parity; run in serialized native suite"]
fn native_masked_scatter_source_cast_matches_all_reported_destination_types() {
    use safemlx::{Array, Device, DeviceType, Dtype, Stream};
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let mechanism = MlxMetalWorkspaceMechanisms {
        allocation: NativeAllocationFacts {
            page_size: 4096,
            cpu_header: false,
            original_storage: false,
        },
        sdpa_blocks: None,
    };
    let formats = [
        (WorkspaceFloatingType::Float32, Dtype::Float32),
        (WorkspaceFloatingType::Float16, Dtype::Float16),
        (WorkspaceFloatingType::Bfloat16, Dtype::Bfloat16),
    ];
    for (destination, dtype) in formats {
        let input = Array::from_slice(&[1_f32, 2., 3., 4., 5., 6.], &[1, 3, 2])
            .as_dtype(dtype, &stream)
            .unwrap();
        let mask = Array::from_slice(&[false, true, true], &[1, 3]);
        for (source, source_dtype) in formats {
            let rows = Array::from_slice(&[-1_f32, -2., 7., 8.], &[2, 2])
                .as_dtype(source_dtype, &stream)
                .unwrap();
            let output =
                safemlx::ops::indexing::masked_scatter(&input, &mask, &rows, &stream).unwrap();
            assert_eq!(output.dtype(), dtype);
            let op = operation(&[1, 3, 2], &[1, 3], &[2, 2]);
            let mut op = op;
            op.inputs[0] = op.inputs[0]
                .clone()
                .with_representation(Some(WorkspaceRepresentation::new(destination, false)));
            op.inputs[2] = op.inputs[2]
                .clone()
                .with_representation(Some(WorkspaceRepresentation::new(source, false)));
            assert_eq!(
                mechanism
                    .output_representation(op.as_view(), 0)
                    .unwrap()
                    .dtype(),
                destination
            );
            let values = output.as_dtype(Dtype::Float32, &stream).unwrap();
            assert_eq!(
                values.evaluated().unwrap().as_slice::<f32>(),
                &[1., 2., -1., -2., 7., 8.]
            );
        }
    }
}
