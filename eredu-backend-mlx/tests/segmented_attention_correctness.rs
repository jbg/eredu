//! Mixed native source dtypes agree with an independent segmented-attention oracle.
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{
    reference_segmented_attention, NeuralBackend, SegmentedAttentionInput, Tensor,
    TensorElementType,
};
use safemlx::{Array, Device, DeviceType, Dtype, Stream};
fn tensor(data: &[f32], shape: &[i32], dtype: Dtype, stream: &Stream) -> MlxTensor {
    MlxTensor::from_array(
        Array::from_slice(data, shape)
            .as_dtype(dtype, stream)
            .unwrap(),
    )
}

fn floats(tensor: &MlxTensor, stream: &Stream) -> Vec<f32> {
    tensor
        .as_array()
        .as_dtype(Dtype::Float32, stream)
        .unwrap()
        .into_evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec()
}

fn exercise_native(device: DeviceType) {
    let stream = Stream::new_with_device(&Device::new(device, 0));
    // Dyadic values survive every source cast exactly; the independent oracle
    // therefore isolates promotion/segmentation from input quantization error.
    let q: Vec<f32> = (0..40).map(|i| (i as i32 % 11 - 5) as f32 / 8.).collect();
    let k: Vec<f32> = (0..40)
        .map(|i| ((i * 7) as i32 % 13 - 6) as f32 / 8.)
        .collect();
    let v: Vec<f32> = (0..30)
        .map(|i| ((i * 3) as i32 % 17 - 8) as f32 / 4.)
        .collect();
    let segments = [2, 1, 2];
    let expected = reference_segmented_attention(5, 2, 4, 3, &q, &k, &v, &segments, 0.5).unwrap();
    for types in [
        [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16],
        [Dtype::Float16, Dtype::Bfloat16, Dtype::Float16],
        [Dtype::Float32; 3],
    ] {
        let queries = tensor(&q, &[5, 2, 4], types[0], &stream);
        let keys = tensor(&k, &[5, 2, 4], types[1], &stream);
        let values = tensor(&v, &[5, 2, 3], types[2], &stream);
        let input = SegmentedAttentionInput {
            queries: &queries,
            keys: &keys,
            values: &values,
            segment_lengths: &segments,
            scale: 0.5,
        };
        let output = Backend::segmented_attention(input, &stream).unwrap();
        assert_eq!(output.shape(), &[5, 2, 3]);
        assert_eq!(output.element_type(), Some(TensorElementType::F32));
        let actual = floats(&output, &stream);
        for (actual, expected) in actual.iter().zip(&expected) {
            assert!(
                (actual - expected).abs() <= 2e-5,
                "scalar segmented oracle: {actual} versus {expected}"
            );
        }
    }
}
#[test]
fn cpu_segmented_attention_matches_mixed_dtype_scalar_oracle() {
    exercise_native(DeviceType::Cpu);
}
#[test]
fn malformed_segments_reject_before_input_evaluation() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let value = tensor(&[0.25; 64], &[32, 1, 2], Dtype::Bfloat16, &stream);
    assert!(!value.as_array().is_available().unwrap());
    for lengths in [vec![], vec![31], vec![32, 1], vec![16, 0, 16], vec![33, -1]] {
        let input = SegmentedAttentionInput {
            queries: &value,
            keys: &value,
            values: &value,
            segment_lengths: &lengths,
            scale: 0.5,
        };
        assert!(Backend::segmented_attention(input, &stream).is_err());
        assert!(!value.as_array().is_available().unwrap());
    }
}
#[cfg(all(feature = "metal", not(feature = "cuda")))]
#[test]
#[ignore = "requires a local Metal device; set MLX_ENABLE_TF32=0"]
fn metal_segmented_attention_matches_mixed_dtype_scalar_oracle() {
    exercise_native(DeviceType::Gpu);
}
