use super::*;
use crate::backend::ExecutionContext;
use eredu_nn::multimodal::{
    masked_output_projection, multi_axis_rotary_embeddings, project_flattened_patches,
    reference_flattened_patch_projection, reference_masked_output_projection,
    reference_multi_axis_rotary_embeddings, FlattenedPatchSpec, MaskedOutputProjectionInput,
    MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
};
use safemlx::{Device, DeviceType};

fn close(actual: &Array, expected: &[f32]) {
    let actual = actual.evaluated().unwrap();
    assert_eq!(actual.as_slice::<f32>().len(), expected.len());
    assert!(actual
        .as_slice::<f32>()
        .iter()
        .zip(expected)
        .all(|(left, right)| (left - right).abs() < 1e-5));
}

fn arrays_close(actual: &Array, expected: &Array) {
    assert_eq!(actual.shape(), expected.shape());
    assert_eq!(actual.dtype(), expected.dtype());
    let actual = actual.evaluated().unwrap();
    let expected = expected.evaluated().unwrap();
    assert_eq!(
        actual.as_slice::<f32>().len(),
        expected.as_slice::<f32>().len()
    );
    assert!(actual
        .as_slice::<f32>()
        .iter()
        .zip(expected.as_slice::<f32>())
        .all(|(left, right)| (left - right).abs() < 1e-5));
}

#[test]
fn wrapper_is_one_transparent_native_handle() {
    fn assert_contract<T: Tensor + AsRef<Array> + From<Array>>() {}
    fn assert_native_conversion<T: Into<Array>>() {}
    fn assert_parameter_traversal<T: eredu_nn::Parameterized<MlxTensor>>() {}

    assert_contract::<MlxTensor>();
    assert_native_conversion::<MlxTensor>();
    assert_parameter_traversal::<eredu_nn::Parameter<MlxTensor>>();
    assert_eq!(
        std::mem::size_of::<MlxTensor>(),
        std::mem::size_of::<Array>()
    );
    assert_eq!(
        std::mem::align_of::<MlxTensor>(),
        std::mem::align_of::<Array>()
    );
}

#[test]
#[ignore = "requires an MLX execution device; run with --ignored on an MLX-capable host"]
fn wrapping_and_unwrapping_preserve_the_native_handle() {
    let native = Array::from_slice(&[1.0_f32, 2.0], &[2]);
    let native_handle = native.as_ptr().ctx;
    let wrapped = MlxTensor::from_array(native);
    assert_eq!(wrapped.as_array().as_ptr().ctx, native_handle);
    let native = wrapped.into_array();
    assert_eq!(native.as_ptr().ctx, native_handle);
}

#[test]
#[ignore = "requires an MLX execution device; run with --ignored on an MLX-capable host"]
fn arithmetic_shape_and_indexing_match_native_operations() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let left = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let right = Array::from_slice(&[6.0_f32, 5.0, 4.0, 3.0, 2.0, 1.0], &[2, 3]);
    let wrapped_left = MlxTensor::from_array(left.clone());
    let wrapped_right = MlxTensor::from_array(right.clone());

    let actual = wrapped_left.add(&wrapped_right, stream).unwrap();
    let expected = Array::add(&left, &right, stream).unwrap();
    arrays_close(actual.as_array(), &expected);

    let actual = actual.reshape(&[3, 2], stream).unwrap();
    let expected = expected.reshape(&[3, 2], stream).unwrap();
    arrays_close(actual.as_array(), &expected);

    let actual = wrapped_left
        .index(&[Index::Range(0, 2), Index::At(1)], stream)
        .unwrap();
    let expected = left.try_index_device((0..2, 1), stream).unwrap();
    arrays_close(actual.as_array(), &expected);
}

#[test]
#[ignore = "requires an MLX execution device; run with --ignored on an MLX-capable host"]
fn convolution_matches_native_operation() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let input = Array::from_slice(
        &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        &[1, 5, 2],
    );
    let weight = Array::from_slice(
        &[
            0.5_f32, 0.0, -0.5, 1.0, 0.0, 1.5, 2.0, 0.0, -2.0, 1.5, 0.0, 1.0,
        ],
        &[2, 3, 2],
    );
    let actual = MlxTensor::conv1d(
        &MlxTensor::from_array(input.clone()),
        &MlxTensor::from_array(weight.clone()),
        1,
        0,
        1,
        1,
        stream,
    )
    .unwrap();
    let expected = safemlx::ops::conv1d(&input, &weight, 1, 0, 1, 1, stream).unwrap();
    arrays_close(actual.as_array(), &expected);
}

#[test]
#[ignore = "requires an MLX execution device; run with --ignored on an MLX-capable host"]
fn scaled_attention_matches_native_operation() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let queries = Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
    let keys = Array::from_slice(&[1.0_f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2]);
    let values = Array::from_slice(&[2.0_f32, 1.0, 4.0, 3.0], &[1, 1, 2, 2]);
    let scale = 1.0 / 2.0_f32.sqrt();
    let actual = MlxTensor::scaled_dot_product_attention(
        &MlxTensor::from_array(queries.clone()),
        &MlxTensor::from_array(keys.clone()),
        &MlxTensor::from_array(values.clone()),
        scale,
        AttentionMask::Causal,
        stream,
    )
    .unwrap();
    let expected = safemlx::fast::scaled_dot_product_attention(
        &queries,
        &keys,
        &values,
        scale,
        ScaledDotProductAttentionMask::Causal,
        None,
        stream,
    )
    .unwrap();
    arrays_close(actual.as_array(), &expected);
}

#[test]
#[ignore = "requires an MLX execution device; run with --ignored on an MLX-capable host"]
fn invalid_shapes_return_backend_errors() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let native = Array::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]);
    let wrapped = MlxTensor::from_array(native.clone());

    assert!(native.reshape(&[3], stream).is_err());
    assert!(wrapped.reshape(&[3], stream).is_err());
}

#[test]
#[ignore = "explicit MLX patch-projection parity; run outside the sandbox"]
fn mlx_flattened_patch_projection_matches_scalar_reference() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let input_values = [1.0, 2.0, 3.0, 4.0];
    let weight_values = [1.0, 0.5, -1.0, 2.0];
    let bias_values = [0.25, -0.5];
    let input = MlxTensor::from_array(Array::from_slice(&input_values, &[2, 2]));
    let weight = MlxTensor::from_array(Array::from_slice(&weight_values, &[2, 1, 1, 1, 2]));
    let bias = MlxTensor::from_array(Array::from_slice(&bias_values, &[2]));
    let actual = project_flattened_patches(
        &input,
        &weight,
        Some(&bias),
        FlattenedPatchSpec {
            channels: 1,
            temporal: 1,
            height: 1,
            width: 2,
            output: 2,
        },
        stream,
    )
    .unwrap();
    let expected = reference_flattened_patch_projection(
        &input_values,
        2,
        &weight_values,
        2,
        Some(&bias_values),
    )
    .unwrap();
    close(actual.as_array(), &expected);
}

#[test]
#[ignore = "explicit MLX multi-axis rotary parity; run outside the sandbox"]
fn mlx_multi_axis_rotary_matches_scalar_reference() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let position_values = [-1, 2, 3, 4];
    let positions = MlxTensor::from_array(Array::from_slice(&position_values, &[2, 2]));
    let spec = MultiAxisRotarySpec {
        axes: vec![
            RotaryAxisSpec {
                dimensions: 4,
                position_offset: 0,
            },
            RotaryAxisSpec {
                dimensions: 4,
                position_offset: 1,
            },
        ],
        base: 100.0,
        minimum_position: 0,
        layout: MultiAxisRotaryLayout::RoundRobinSections,
    };
    let (actual_cosine, actual_sine) =
        multi_axis_rotary_embeddings(&positions, &spec, stream).unwrap();
    let (expected_cosine, expected_sine) =
        reference_multi_axis_rotary_embeddings(&position_values, 2, &spec).unwrap();
    close(actual_cosine.as_array(), &expected_cosine);
    close(actual_sine.as_array(), &expected_sine);
}

#[test]
#[ignore = "explicit MLX masked-output parity; run outside the sandbox"]
fn mlx_masked_output_projection_matches_scalar_reference() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let hidden_values = [2.0, 1.0];
    let weight_values = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0];
    let centroid_values = [0.1, 0.9];
    let ordering_values = [2, 0, 3, 1];
    let hidden = MlxTensor::from_array(Array::from_slice(&hidden_values, &[1, 1, 2]));
    let weight = MlxTensor::from_array(Array::from_slice(&weight_values, &[4, 2]));
    let centroids = MlxTensor::from_array(Array::from_slice(&centroid_values, &[1, 1, 2]));
    let ordering = MlxTensor::from_array(Array::from_slice(&ordering_values, &[4]));
    let actual = masked_output_projection(
        MaskedOutputProjectionInput {
            hidden: &hidden,
            output_weight: &weight,
            centroid_logits: &centroids,
            token_ordering: &ordering,
            top_centroids: 1,
            mask_margin: 1.0,
        },
        stream,
    )
    .unwrap();
    let expected = reference_masked_output_projection(
        &hidden_values,
        1,
        2,
        &weight_values,
        4,
        &centroid_values,
        2,
        &[2, 0, 3, 1],
        1,
        1.0,
    )
    .unwrap();
    close(actual.as_array(), &expected);
}
