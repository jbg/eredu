//! Compare the cold compact-row layout with the actual concatenate branches.
use super::*;
use eredu_nn::Tensor;
use safemlx::{Array, Device, DeviceType, Dtype, Stream};

#[test]
#[ignore = "requires Metal; native singleton alias and canonical concatenate copy"]
fn native_concatenate_preserves_alias_layout_and_materializes_canonical_rows() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    let mechanism = MlxMetalWorkspaceMechanisms::current_host().unwrap();
    for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
        for empty in [false, true] {
            for transposed in [false, true] {
                let data: &[f32] = if empty { &[] } else { &[1., 2., 3., 4., 5., 6.] };
                let mut input = Array::from_slice(data, &[if empty { 0 } else { 2 }, 3])
                    .as_dtype(dtype, &stream).unwrap();
                if transposed { input = input.transpose_axes(&[1, 0], &stream).unwrap(); }
                input.evaluated().unwrap();
                for count in [1, 2] {
                    let arrays = vec![input.clone(); count];
                    let actual = safemlx::ops::concatenate_axis(&arrays, 0, &stream).unwrap();
                    actual.evaluated().unwrap();
                    let context = WorkspaceContext::new(mechanism);
                    let mut projection = ExistingArrayProjection::with_source_count(&context, 2).unwrap();
                    let source = projection.project(&input).unwrap();
                    let expected = projection.project(&actual).unwrap();
                    let symbolic = WorkspaceTensor::concatenate(&vec![source.clone(); count], 0, &context).unwrap();
                    assert_eq!(symbolic.shape(), actual.shape());
                    let result = symbolic.layout().representation().unwrap();
                    let native = expected.layout().representation().unwrap();
                    assert_eq!(result.dtype(), native.dtype());
                    assert_eq!(result.row_contiguous(), native.row_contiguous());
                    assert_eq!(result.last_axis_contiguous(), native.last_axis_contiguous());
                    if count == 1 {
                        assert_eq!(symbolic.layout().representation(), source.layout().representation());
                    } else {
                        assert!(result.row_contiguous());
                    }
                    // The source carries no numerical replacement: the native
                    // worker must still concatenate the original nonzero rows.
                    let original = crate::MlxTensor::from_array(input.clone()).to_f32_vec(&stream).unwrap();
                    let values = crate::MlxTensor::from_array(actual).to_f32_vec(&stream).unwrap();
                    assert_eq!(values, original.repeat(count));
                }
            }
        }
    }
    // No actual floating dtype or physical source means no representation,
    // even when the selected copying worker would create canonical storage.
    let context = WorkspaceContext::new(mechanism);
    let unknown = WorkspaceTensor::existing(context.layout(&[1, 3], WorkspaceDtype::Float32).unwrap(), &context).unwrap();
    for count in [1, 2] {
        let output = WorkspaceTensor::concatenate(&vec![unknown.clone(); count], 0, &context).unwrap();
        assert!(output.layout().representation().is_none());
    }
}
