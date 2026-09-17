use super::*;
use eredu_nn::{F32InitializationPlan, Tensor};

fn selected() -> MlxMetalWorkspaceMechanisms {
    MlxMetalWorkspaceMechanisms {
        allocation: MetalAllocationFacts { page_size: 16_384 },
        sdpa_blocks: None,
    }
}

#[test]
fn generated_f32_prices_fixed_host_buffer_and_native_copy_in_separate_domains() {
    for shape in [&[3, 5][..], &[][..], &[0, 7][..]] {
        let context = WorkspaceContext::new(selected());
        let first =
            WorkspaceTensor::from_f32_fn(shape, |_| panic!("cold scalar executed"), &context)
                .unwrap();
        let second =
            WorkspaceTensor::from_f32_fn(shape, |_| panic!("cold scalar executed"), &context)
                .unwrap();
        let report = context.report(&[first, second]).unwrap();
        let bytes = F32InitializationPlan::new(shape)
            .unwrap()
            .host_buffer_bytes();
        assert_eq!(report.host_workspace_bytes, Some(2 * bytes));
        assert!(report.tensor_buffers.total_bytes.unwrap() >= 2 * bytes);
        assert_eq!(
            report.total_bytes,
            report
                .tensor_buffers
                .total_bytes
                .map(|native| native + 2 * bytes)
        );
        assert_eq!(report.operations.len(), 2);
        for operation in report.operations {
            assert!(matches!(
                operation.kind,
                WorkspaceOperationKind::GeneratedF32Initialization
            ));
            assert!(operation.inputs.is_empty());
            assert_eq!(operation.outputs.len(), 1);
            let bound = selected().operation_bound(&operation).unwrap().unwrap();
            // The same native seed/copy envelope as existing F32 from_slice;
            // its separately-owned source buffer is the new host fact.
            let mut borrowed = operation.clone();
            borrowed.kind = WorkspaceOperationKind::Initialize;
            let previous = selected().operation_bound(&borrowed).unwrap().unwrap();
            assert_eq!(bound.outputs, previous.outputs);
            assert_eq!(bound.scratch_bytes, previous.scratch_bytes);
            assert_eq!(
                selected()
                    .host_workspace_bound(&borrowed)
                    .unwrap()
                    .unwrap()
                    .bytes,
                0
            );
            assert_eq!(
                selected()
                    .host_workspace_bound(&operation)
                    .unwrap()
                    .unwrap()
                    .bytes,
                bytes
            );
        }
        assert_eq!(
            context.report(&[]).unwrap().host_workspace_bytes,
            Some(2 * bytes)
        );
    }
}

#[test]
fn generated_f32_fact_rejects_wrong_dtype_inputs_and_multiple_outputs() {
    let layout = WorkspaceLayout::new(&[3], WorkspaceDtype::Float32).unwrap();
    let valid = WorkspaceOperation {
        kind: WorkspaceOperationKind::GeneratedF32Initialization,
        inputs: vec![],
        outputs: vec![layout.clone()],
    };
    let mut wrong_dtype = valid.clone();
    wrong_dtype.outputs[0] = WorkspaceLayout::new(&[3], WorkspaceDtype::Int32).unwrap();
    let mut input = valid.clone();
    input.inputs.push(layout.clone());
    let mut outputs = valid;
    outputs.outputs.push(layout);
    for operation in [wrong_dtype, input, outputs] {
        assert!(selected().operation_bound(&operation).is_err());
        assert!(selected().host_workspace_bound(&operation).is_err());
    }
}

#[test]
fn native_generated_f32_default_worker_preserves_nonzero_values_and_shape() {
    let stream =
        safemlx::Stream::new_with_device(&safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
    let calls = std::cell::Cell::new(0);
    let value = crate::MlxTensor::from_f32_fn(
        &[2, 3],
        |index| {
            assert_eq!(calls.get(), index);
            calls.set(index + 1);
            (index as f32 - 2.0) * 1.75
        },
        &stream,
    )
    .unwrap();
    assert_eq!(calls.get(), 6);
    assert_eq!(value.shape(), [2, 3]);
    assert_eq!(
        value
            .as_array()
            .evaluated()
            .unwrap()
            .try_to_vec::<f32>()
            .unwrap(),
        [-3.5, -1.75, 0.0, 1.75, 3.5, 5.25]
    );
}
