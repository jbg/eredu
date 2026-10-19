use super::*;
use crate::backend::ExecutionContext;
use eredu_nn::multimodal::{
    masked_output_projection, multi_axis_rotary_embeddings, project_flattened_patches,
    reference_flattened_patch_projection, reference_masked_output_projection,
    reference_multi_axis_rotary_embeddings, FlattenedPatchSpec, MaskedOutputProjectionInput,
    MultiAxisRotaryLayout, MultiAxisRotarySpec, RotaryAxisSpec,
};
use safemlx::{Device, DeviceType};

#[test]
fn strict_score_selection_and_scatter_preserve_rank_ties_and_i64_validation() {
    use crate::backend::nn::tensor::TokenValidationScope;
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let scores = MlxTensor::from_f32_slice(
        &[3., 5., 5., -2., 5., -1., -1., -3., 4., 4.],
        &[2, 5],
        stream,
    )
    .unwrap();
    let (values, ids) = scores.ranked_topk(3, stream).unwrap();
    assert_eq!(ids.to_i32_vec(stream).unwrap(), [1, 2, 4, 3, 4, 0]);
    assert_eq!(
        values.to_f32_vec(stream).unwrap(),
        [5., 5., 5., 4., 4., -1.]
    );
    let base = MlxTensor::full_f32(f32::NEG_INFINITY, &[2, 5], stream).unwrap();
    let scattered = base
        .scatter_last_axis(&ids, &values, stream)
        .unwrap()
        .to_f32_vec(stream)
        .unwrap();
    assert_eq!(
        scattered,
        [
            f32::NEG_INFINITY,
            5.,
            5.,
            f32::NEG_INFINITY,
            5.,
            -1.,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            4.,
            4.
        ]
    );
    for invalid in [vec![0_i64, 0], vec![0, 1_i64 << 40], vec![0, -1_i64 << 40]] {
        let indices = MlxTensor::from_array(Array::from_slice(&invalid, &[1, 2]));
        let base = MlxTensor::full_f32(0., &[1, 5], stream).unwrap();
        let values = MlxTensor::from_f32_slice(&[2., 7.], &[1, 2], stream).unwrap();
        let scope = TokenValidationScope::begin().unwrap();
        let output = base.scatter_last_axis(&indices, &values, stream).unwrap();
        let validation = scope.finish();
        safemlx::transforms::eval(
            validation
                .arrays()
                .chain(std::iter::once(output.as_array())),
        )
        .unwrap();
        assert!(validation.validate_completed().is_err());
    }
}

#[test]
fn i64_parameter_binding_preserves_offsets_beyond_floating_and_i32_precision() {
    use eredu_nn::{Parameter, ParameterSpec, Tensor, TensorElementType};
    let execution = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let mut value = Parameter::<MlxTensor>::unloaded_i64(
        ParameterSpec::trainable("offsets").unwrap(),
        &[2],
        stream,
    )
    .unwrap();
    assert_eq!(value.as_ref().element_type(), Some(TensorElementType::I64));
    let expected = [(1_i64 << 40) + 3, -(1_i64 << 40) + 5];
    value.replace(MlxTensor::from_array(Array::from_slice(&expected, &[2])));
    let selected = value
        .as_ref()
        .take_axis(
            &MlxTensor::from_array(Array::from_slice(&[1_i32, 0], &[2])),
            0,
            stream,
        )
        .unwrap();
    let selected = selected.as_array().evaluated().unwrap();
    assert_eq!(selected.as_slice::<i64>(), &[expected[1], expected[0]]);
}

fn close(actual: &Array, expected: &[f32]) {
    let actual = actual.evaluated().unwrap();
    assert_eq!(actual.as_slice::<f32>().len(), expected.len());
    assert!(actual
        .as_slice::<f32>()
        .iter()
        .zip(expected)
        .all(|(left, right)| (left - right).abs() < 1e-5));
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
#[ignore = "requires local MLX CPU and Metal execution"]
fn gather_bounds_settle_before_rejection_and_preserve_valid_indices() {
    use crate::backend::nn::tensor::TokenValidationScope;
    use safemlx::transforms::async_eval_with_event;

    for device in [DeviceType::Cpu, DeviceType::Gpu] {
        let execution = ExecutionContext::new(Device::new(device, 0));
        let stream = execution.stream();
        let table = MlxTensor::from_array(Array::from_slice(&[0_i32, 2, 1], &[3]));
        let values = MlxTensor::from_array(Array::from_slice(&[0.25_f32, -2.0, 3.5], &[3]));
        let valid = MlxTensor::from_array(Array::from_slice(&[-3_i32, -1, 1], &[3]));
        let selected = table.take_axis(&valid, -1, stream).unwrap();
        close(
            values.take_axis(&selected, 0, stream).unwrap().as_array(),
            &[0.25, -2.0, 3.5],
        );
        for invalid in [
            Array::from_slice(&[-4_i32, 3, i32::MAX], &[3]),
            Array::from_slice(&[3_u32, u32::MAX, 4], &[3]),
            Array::from_slice(&[3_i64, i64::MAX, i64::MIN], &[3]),
            Array::from_slice(&[3_u64, u64::MAX, 4], &[3]),
        ] {
            let invalid = MlxTensor::from_array(invalid);
            assert!(table.take_axis(&invalid, 0, stream).is_err());
            let scope = TokenValidationScope::begin().unwrap();
            let selected = table.take_axis(&invalid, 0, stream).unwrap();
            // A second gather must remain safe even before the first one's
            // deferred rejection reaches the submission completion boundary.
            let output = values.take_axis(&selected, 0, stream).unwrap();
            let validations = scope.finish();
            async_eval_with_event(std::iter::once(output.as_array()).chain(validations.arrays()))
                .unwrap()
                .synchronize()
                .unwrap();
            assert!(validations
                .validate_completed()
                .unwrap_err()
                .to_string()
                .contains("gather index is outside"));
            close(output.as_array(), &[0.25; 3]);
        }
        assert!(table.take_axis(&valid, 1, stream).is_err());
        assert!(table.take_axis(&valid, i32::MIN, stream).is_err());
    }
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
