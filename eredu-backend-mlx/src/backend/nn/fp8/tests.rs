#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
use super::quantize_activations;
use super::{decode_scale, grouped_linear, linear, segmented_linear, segmented_transposed_linear};

#[test]
#[ignore = "requires MLX runtime execution"]
fn decodes_native_e8m0_scale_bytes() {
    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let encoded = Array::from_slice(&[127u8, 128, 126], &[3]);
    let decoded = decode_scale(&encoded, stream).unwrap();
    let decoded = decoded.evaluated().unwrap();
    assert_eq!(decoded.as_slice::<f32>(), &[1.0, 2.0, 0.5]);
}
use crate::backend::ExecutionContext;
use safemlx::{ops::indexing::TryIndexOp, Array, Device, DeviceType, Dtype};

fn assert_block_fp8_dense_and_grouped_projections(device_type: DeviceType) {
    let context = ExecutionContext::new(Device::new(device_type, 0));
    let stream = context.stream();
    // E4M3 0x38 represents 1.0; inverse block scale 2.0 makes
    // every effective weight 2.0.
    let input = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 4]);
    let weight = Array::from_slice(&[0x38u8; 16], &[4, 4]);
    let scale = Array::from_slice(&[2.0f32], &[1, 1]);
    let output = linear(&input, &weight, &scale, stream).unwrap();
    assert_eq!(output.shape(), &[1, 4]);
    assert_eq!(
        output
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        8.0
    );

    // Some block-FP8 checkpoints store inverse block scales as
    // BF16. Keep them in their checkpoint dtype through the custom kernel
    // instead of requiring a separate conversion at every projection.
    let bf16_scale = scale.as_dtype(Dtype::Bfloat16, stream).unwrap();
    let bf16_scale_output = linear(&input, &weight, &bf16_scale, stream).unwrap();
    assert_eq!(
        bf16_scale_output
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        8.0
    );

    let bf16_input = input.as_dtype(Dtype::Bfloat16, stream).unwrap();
    let bf16_output = linear(&bf16_input, &weight, &scale, stream).unwrap();
    assert_eq!(bf16_output.dtype(), Dtype::Bfloat16);
    assert_eq!(
        bf16_output
            .as_dtype(Dtype::Float32, stream)
            .unwrap()
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        8.0
    );

    let grouped_weight = Array::from_slice(&[0x38u8; 32], &[2, 4, 4]);
    let grouped_scale = Array::from_slice(&[2.0f32, 3.0], &[2, 1, 1])
        .as_dtype(Dtype::Bfloat16, stream)
        .unwrap();
    let rows = Array::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[2, 4]);
    let groups = Array::from_slice(&[0u32, 1], &[2]);
    let output = grouped_linear(&rows, &grouped_weight, &grouped_scale, &groups, stream).unwrap();
    assert_eq!(output.shape(), &[2, 4]);
    assert_eq!(
        output
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        8.0
    );
    assert_eq!(
        output
            .try_index_device((1, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        12.0
    );

    // Exercise grouped row segments at real 128-row block boundaries,
    // including the transposed path used by absorbed MLA queries.
    let segmented_weight = Array::from_slice(&vec![0x38u8; 512 * 128], &[512, 128]);
    let segmented_scale = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4, 1])
        .as_dtype(Dtype::Bfloat16, stream)
        .unwrap();
    let segmented_groups = Array::from_slice(&[0u32, 1], &[2]);
    let query = Array::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]);
    let absorbed_query = segmented_transposed_linear(
        &query,
        &segmented_weight,
        &segmented_scale,
        &segmented_groups,
        256,
        0,
        stream,
    )
    .unwrap();
    assert_eq!(absorbed_query.shape(), &[2, 128]);
    assert_eq!(
        absorbed_query
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        128.0
    );
    assert_eq!(
        absorbed_query
            .try_index_device((1, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        384.0
    );

    let context = Array::from_slice(&vec![1.0f32; 2 * 128], &[2, 128]);
    let absorbed_output = segmented_linear(
        &context,
        &segmented_weight,
        &segmented_scale,
        &segmented_groups,
        256,
        128,
        128,
        stream,
    )
    .unwrap();
    assert_eq!(absorbed_output.shape(), &[2, 128]);
    assert_eq!(
        absorbed_output
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        256.0
    );
    assert_eq!(
        absorbed_output
            .try_index_device((1, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        512.0
    );
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn block_fp8_dense_and_grouped_projections() {
    assert_block_fp8_dense_and_grouped_projections(DeviceType::Gpu);
}

#[test]
fn block_fp8_cpu_reference_projections() {
    assert_block_fp8_dense_and_grouped_projections(DeviceType::Cpu);
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn dynamic_activation_quantization_matches_e4m3_and_clamped_scale() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let mut input = vec![0.0f32; 256];
    for (index, value) in [448.0, -448.0, 1.0, 0.5, 0.015625, 0.001953125]
        .into_iter()
        .enumerate()
    {
        input[index] = value;
    }
    let input = Array::from_slice(&input, &[1, 256]);
    let quantized = quantize_activations(&input, 1, 256, stream).unwrap();

    for (index, expected) in [0x7eu8, 0xfe, 0x38, 0x30, 0x08, 0x01]
        .into_iter()
        .enumerate()
    {
        assert_eq!(
            quantized
                .values
                .try_index_device((0, index as i32), stream)
                .unwrap()
                .item::<u8>(stream),
            expected
        );
    }
    assert_eq!(
        quantized
            .scales
            .try_index_device((0, 0), stream)
            .unwrap()
            .item::<f32>(stream),
        1.0
    );
    let zero_scale = quantized
        .scales
        .try_index_device((0, 1), stream)
        .unwrap()
        .item::<f32>(stream);
    assert!((zero_scale - 1.0e-4 / 448.0).abs() < 1.0e-12);

    // Compare every finite positive and negative E4M3FN encoding against
    // MLX's native conversion. Each 128-value block includes magnitude
    // 448, so the dynamic scale is exactly one.
    let encoded = (0u8..=126)
        .chain(std::iter::once(126))
        .chain(128u8..=254)
        .chain(std::iter::once(254))
        .collect::<Vec<_>>();
    let decoded = Array::from_slice(&encoded, &[2, 128])
        .from_fp8(Dtype::Float32, stream)
        .unwrap();
    let quantized = quantize_activations(&decoded, 2, 128, stream).unwrap();
    for (index, expected) in encoded.into_iter().enumerate() {
        assert_eq!(
            quantized
                .values
                .try_index_device((index as i32 / 128, index as i32 % 128), stream)
                .unwrap()
                .item::<u8>(stream),
            expected
        );
    }
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
#[test]
fn block_fp8_dense_and_grouped_scale_block_boundaries() {
    let context = ExecutionContext::new(Device::new(DeviceType::Gpu, 0));
    let stream = context.stream();
    let in_dim = 129;
    let out_dim = 130;

    // Every packed weight is 1.0, making the expected result depend only
    // on the four independently scaled 128x128 weight blocks.
    let input = Array::from_slice(
        &[vec![1.0f32; in_dim as usize], vec![2.0f32; in_dim as usize]].concat(),
        &[2, in_dim],
    );
    let weight = Array::from_slice(
        &vec![0x38u8; (out_dim * in_dim) as usize],
        &[out_dim, in_dim],
    );
    let scale = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
    let output = linear(&input, &weight, &scale, stream).unwrap();
    for (row, low, high) in [(0, 130.0f32, 388.0f32), (1, 260.0, 776.0)] {
        assert_eq!(
            output
                .try_index_device((row, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            low
        );
        assert_eq!(
            output
                .try_index_device((row, 129), stream)
                .unwrap()
                .item::<f32>(stream),
            high
        );
    }

    let grouped_weight = Array::from_slice(
        &vec![0x38u8; (2 * out_dim * in_dim) as usize],
        &[2, out_dim, in_dim],
    );
    let grouped_scale = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
    let grouped_input = Array::from_slice(&vec![1.0f32; (2 * in_dim) as usize], &[2, in_dim]);
    let groups = Array::from_slice(&[1u32, 0], &[2]);
    let output = grouped_linear(
        &grouped_input,
        &grouped_weight,
        &grouped_scale,
        &groups,
        stream,
    )
    .unwrap();
    for (route, low, high) in [(0, 646.0f32, 904.0f32), (1, 130.0, 388.0)] {
        assert_eq!(
            output
                .try_index_device((route, 0), stream)
                .unwrap()
                .item::<f32>(stream),
            low
        );
        assert_eq!(
            output
                .try_index_device((route, 129), stream)
                .unwrap()
                .item::<f32>(stream),
            high
        );
    }
}
