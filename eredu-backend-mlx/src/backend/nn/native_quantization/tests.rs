use super::*;
use crate::ops::indexing::TryIndexOp;

#[test]
fn k_quant_linear_dispatch_uses_shape_specific_kernels() {
    assert_eq!(
        k_quant_linear_kernel_class(1),
        KQuantLinearKernelClass::MatrixVector
    );
    for rows in 2..=8 {
        assert_eq!(
            k_quant_linear_kernel_class(rows),
            KQuantLinearKernelClass::SmallBatch
        );
    }
    for rows in [9, 16, 32, 33] {
        assert_eq!(
            k_quant_linear_kernel_class(rows),
            KQuantLinearKernelClass::MatrixMatrix
        );
    }
}

fn unhex(value: &str) -> Vec<u8> {
    value
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn every_iq_format_executes_direct_packed_linear_embedding_and_grouped() {
    let stream = crate::test_stream();
    for line in include_str!("../fixtures/llama-c0bc8591-iq.oracle").lines() {
        let mut fields = line.split('|');
        let ty = GgmlType::from_code(fields.next().unwrap().parse().unwrap());
        let all_raw = unhex(fields.next().unwrap());
        let _oracle_f16 = fields.next().unwrap();
        let (block_values, block_bytes) = ty.block_and_bytes().unwrap();
        let raw = &all_raw[..block_bytes as usize];
        let canonical = eredu_gguf::IQuantTensor {
            shape: vec![1, block_values],
            ggml_type: ty,
            endian: GgufEndian::Little,
            data: raw.to_vec(),
        }
        .dequantize_f32()
        .unwrap();
        let packed = Array::from_slice(raw, &[1, block_bytes as i32])
            .copy(stream)
            .unwrap();
        let native = NativeQuantizedTensor::from_iq_array(
            packed,
            &[1, block_values as i32],
            ty,
            GgufEndian::Little,
        )
        .unwrap();

        let input = (0..block_values)
            .map(|index| ((index % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let expected = input
            .iter()
            .zip(&canonical)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>();
        let actual = native
            .linear(
                &Array::from_slice(&input, &[1, block_values as i32]),
                true,
                stream,
            )
            .unwrap();
        eval([&actual]).unwrap();
        let actual = actual.evaluated().unwrap().as_slice::<f32>()[0];
        let tolerance = 2e-4 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{ty:?}: packed linear {actual} != {expected}"
        );

        let prefill_values = (0..9)
            .flat_map(|row| input.iter().map(move |value| value + row as f32 * 0.01))
            .collect::<Vec<_>>();
        let prefill = Array::from_slice(&prefill_values, &[9, block_values as i32]);
        let actual = native.linear(&prefill, true, stream).unwrap();
        let dense = Array::from_slice(&canonical, &[1, block_values as i32]);
        let expected = matmul(&prefill, dense.transpose(stream).unwrap(), stream).unwrap();
        assert!(actual
            .all_close(&expected, Some(3e-4), Some(3e-4), None, stream)
            .unwrap()
            .item::<bool>(stream));

        let embedded = native
            .embedding(&Array::from_slice(&[0i32], &[1]), stream)
            .unwrap();
        eval([&embedded]).unwrap();
        let evaluated = embedded.evaluated().unwrap();
        let actual = evaluated.as_slice::<f32>();
        for (index, (&actual, &expected)) in actual.iter().zip(&canonical).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{ty:?} embedding element {index}"
            );
        }

        let bank = Array::from_slice(&all_raw, &[2, 1, block_bytes as i32])
            .copy(stream)
            .unwrap();
        let bank = NativeQuantizedTensor::from_iq_array(
            bank,
            &[2, 1, block_values as i32],
            ty,
            GgufEndian::Little,
        )
        .unwrap();
        let grouped_input = Array::from_slice(
            &[input.as_slice(), input.as_slice()].concat(),
            &[2, block_values as i32],
        );
        let grouped = native_grouped_linear(
            &grouped_input,
            &bank,
            &Array::from_slice(&[1i32, 0], &[2]),
            stream,
        )
        .unwrap();
        let dense = bank.dequantize(stream).unwrap();
        let selected = dense
            .try_index_device(&Array::from_slice(&[1i32, 0], &[2]), stream)
            .unwrap();
        let expected = matmul(
            grouped_input
                .reshape(&[2, 1, block_values as i32], stream)
                .unwrap(),
            selected.swap_axes(-1, -2, stream).unwrap(),
            stream,
        )
        .unwrap()
        .reshape(&[2, 1], stream)
        .unwrap();
        assert!(grouped
            .all_close(&expected, Some(2e-4), Some(2e-4), None, stream)
            .unwrap()
            .item::<bool>(stream));
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn every_iq_format_executes_batched_metal_kernels() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
    for line in include_str!("../fixtures/llama-c0bc8591-iq.oracle").lines() {
        let mut fields = line.split('|');
        let ty = GgmlType::from_code(fields.next().unwrap().parse().unwrap());
        let all_raw = unhex(fields.next().unwrap());
        let _oracle_f16 = fields.next().unwrap();
        let (block_values, block_bytes) = ty.block_and_bytes().unwrap();
        let raw = &all_raw[..block_bytes as usize];
        let input_values = (0..9 * block_values as usize)
            .map(|index| ((index % 17) as f32 - 8.0) / 16.0)
            .collect::<Vec<_>>();
        let input = Array::from_slice(&input_values, &[9, block_values as i32])
            .copy(&stream)
            .unwrap();
        let native = NativeQuantizedTensor::from_iq_array(
            Array::from_slice(raw, &[1, block_bytes as i32])
                .copy(&stream)
                .unwrap(),
            &[1, block_values as i32],
            ty,
            GgufEndian::Little,
        )
        .unwrap();
        let dense = native.dequantize(&stream).unwrap();
        let actual = native.linear(&input, true, &stream).unwrap();
        let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
        assert!(
            actual
                .all_close(&expected, Some(4e-4), Some(4e-4), None, &stream)
                .unwrap()
                .item::<bool>(&stream),
            "{ty:?} batched Metal linear"
        );
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn iq_metal_execution_honors_big_endian_block_fields() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
    for ty in [GgmlType::IQ4NL, GgmlType::IQ2XS] {
        let (values, bytes) = ty.block_and_bytes().unwrap();
        let mut little = (0..bytes)
            .map(|index| (index * 29 + 7) as u8)
            .collect::<Vec<_>>();
        little[..2].copy_from_slice(&half::f16::from_f32(0.75).to_bits().to_le_bytes());
        let mut big = little.clone();
        match ty {
            GgmlType::IQ4NL => big[..2].reverse(),
            GgmlType::IQ2XS => {
                for pair in big[..66].as_chunks_mut::<2>().0 {
                    pair.reverse();
                }
            }
            _ => unreachable!(),
        }
        let input = Array::from_slice(&vec![1.0f32; values as usize], &[1, values as i32])
            .copy(&stream)
            .unwrap();
        let execute = |raw: &[u8], endian| {
            let packed = Array::from_slice(raw, &[1, bytes as i32])
                .copy(&stream)
                .unwrap();
            let native =
                NativeQuantizedTensor::from_iq_array(packed, &[1, values as i32], ty, endian)
                    .unwrap();
            let output = native.linear(&input, true, &stream).unwrap();
            eval([&output]).unwrap();
            output.evaluated().unwrap().as_slice::<f32>()[0]
        };
        assert_eq!(
            execute(&little, GgufEndian::Little).to_bits(),
            execute(&big, GgufEndian::Big).to_bits(),
            "{ty:?}"
        );
    }
}

fn sample_block() -> Vec<u8> {
    let mut block = vec![0u8; 144];
    block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
    for (index, value) in block[4..16].iter_mut().enumerate() {
        *value = (index as u8).wrapping_mul(17).wrapping_add(3);
    }
    for (index, value) in block[16..].iter_mut().enumerate() {
        *value = (index as u8).wrapping_mul(29).wrapping_add(11);
    }
    block
}

fn sample_q5_1_block() -> Vec<u8> {
    let mut block = vec![0u8; 24];
    block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
    block[2..4].copy_from_slice(&half::f16::from_f32(-0.75).to_bits().to_le_bytes());
    block[4..8].copy_from_slice(&0xa5c3_781fu32.to_le_bytes());
    for (index, value) in block[8..].iter_mut().enumerate() {
        *value = (index as u8).wrapping_mul(23).wrapping_add(5);
    }
    block
}

fn sample_q8_0_block() -> Vec<u8> {
    let mut block = vec![0u8; 34];
    block[0..2].copy_from_slice(&half::f16::from_f32(0.125).to_bits().to_le_bytes());
    for (index, value) in block[2..].iter_mut().enumerate() {
        *value = (index as i8).wrapping_mul(11).wrapping_sub(97) as u8;
    }
    block
}

#[test]
fn q4k_views_share_storage_and_decode_segments() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
    let mut raw = Vec::new();
    for _ in 0..8 {
        raw.extend(sample_block());
    }
    let fused = NativeQuantizedTensor::from_q4k_bytes(&raw, &[2, 4, 256], &stream).unwrap();
    let gate = fused.row_view(0, 2).unwrap();
    let up = fused.row_view(2, 2).unwrap();
    assert!(Arc::ptr_eq(gate.storage(), up.storage()));
    assert_eq!(gate.shape(), vec![2, 2, 256]);
    assert_eq!(up.row_start(), 2);
    drop(fused);
    // Logical views retain the backing allocation after the physical
    // parent object is gone.
    assert_eq!(gate.dequantize(&stream).unwrap().shape(), &[2, 2, 256]);
}

#[test]
fn native_view_copy_preserves_logical_slice() {
    let source = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
    let destination = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
    let mut raw = Vec::new();
    for _ in 0..8 {
        raw.extend(sample_block());
    }
    let source_view = NativeQuantizedTensor::from_q4k_bytes(&raw, &[2, 4, 256], &source)
        .unwrap()
        .row_view(1, 2)
        .unwrap();
    let copied = source_view.copy_to_stream(&destination).unwrap();

    assert!(!Arc::ptr_eq(source_view.storage(), copied.storage()));
    assert_eq!(copied.shape(), source_view.shape());
    assert_eq!(copied.row_start(), source_view.row_start());
    assert_eq!(copied.physical_rows(), source_view.physical_rows());
    assert_eq!(
        copied
            .dequantize(&destination)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>(),
        source_view
            .dequantize(&source)
            .unwrap()
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
    );
}

#[test]
fn q4k_decode_matches_affine_reference() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
    let raw = sample_block();
    let native = NativeQuantizedTensor::from_q4k_bytes(&raw, &[1, 256], &stream).unwrap();
    let actual = native.dequantize(&stream).unwrap();
    let actual = actual.evaluated().unwrap();
    let actual = actual.as_slice::<f32>();

    let descriptor = eredu_gguf::TensorDescriptor {
        name: "test.weight".into(),
        dimensions: vec![256, 1],
        ggml_type: eredu_gguf::GgmlType::Q4K,
        relative_offset: 0,
        data_offset: 0,
        byte_len: 144,
    };
    let reference =
        eredu_gguf::convert_affine(&descriptor, &raw, eredu_gguf::Endian::Little).unwrap();
    let expected = reference.dequantize();
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn q4k_cpu_linear_embedding_and_grouped_fallbacks() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
    let mut raw = Vec::new();
    raw.extend(sample_block());
    raw.extend(sample_block());
    let matrix = NativeQuantizedTensor::from_q4k_bytes(&raw, &[2, 256], &stream).unwrap();
    let dense = matrix.dequantize(&stream).unwrap();
    let input = Array::from_slice(&vec![0.5f32; 512], &[2, 256]);
    let output = matrix.linear(&input, true, &stream).unwrap();
    assert_eq!(output.shape(), &[2, 2]);
    let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
    assert!(output
        .all_close(&expected, Some(1e-5), Some(1e-5), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
    let untransposed_input = Array::from_slice(&[0.25f32, -0.5], &[1, 2]);
    let untransposed = matrix.linear(&untransposed_input, false, &stream).unwrap();
    assert_eq!(untransposed.shape(), &[1, 256]);
    let expected = matmul(&untransposed_input, &dense, &stream).unwrap();
    assert!(untransposed
        .all_close(&expected, Some(1e-5), Some(1e-5), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
    let ids = Array::from_slice(&[1i32, 0], &[2]);
    let embedded = matrix.embedding(&ids, &stream).unwrap();
    assert_eq!(embedded.shape(), &[2, 256]);
    let expected = dense.try_index_device(&ids, &stream).unwrap();
    assert!(embedded
        .all_close(&expected, Some(1e-6), Some(1e-6), None, &stream)
        .unwrap()
        .item::<bool>(&stream));

    let mut group_raw = Vec::new();
    for _ in 0..4 {
        group_raw.extend(sample_block());
    }
    let groups = NativeQuantizedTensor::from_q4k_bytes(&group_raw, &[2, 2, 256], &stream).unwrap();
    let group_ids = Array::from_slice(&[1i32, 0], &[2]);
    let grouped = native_grouped_linear(&input, &groups, &group_ids, &stream).unwrap();
    assert_eq!(grouped.shape(), &[2, 2]);
    let selected = groups
        .dequantize(&stream)
        .unwrap()
        .try_index_device(&group_ids, &stream)
        .unwrap();
    let expected = matmul(
        input.reshape(&[2, 1, 256], &stream).unwrap(),
        selected.swap_axes(-1, -2, &stream).unwrap(),
        &stream,
    )
    .unwrap()
    .reshape(&[2, 2], &stream)
    .unwrap();
    assert!(grouped
        .all_close(&expected, Some(1e-5), Some(1e-5), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
}

#[test]
fn q5_1_decode_matches_affine_reference() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
    let raw = sample_q5_1_block();
    let native = NativeQuantizedTensor::from_q5_1_bytes(&raw, &[1, 32], &stream).unwrap();
    let actual = native.dequantize(&stream).unwrap();
    let actual = actual.evaluated().unwrap();

    let descriptor = eredu_gguf::TensorDescriptor {
        name: "test.weight".into(),
        dimensions: vec![32, 1],
        ggml_type: eredu_gguf::GgmlType::Q5_1,
        relative_offset: 0,
        data_offset: 0,
        byte_len: raw.len() as u64,
    };
    let reference =
        eredu_gguf::convert_affine(&descriptor, &raw, eredu_gguf::Endian::Little).unwrap();
    for (actual, expected) in actual.as_slice::<f32>().iter().zip(reference.dequantize()) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }
}

#[test]
fn q5_1_direct_linear_prefill_and_embedding_match_dequantized_reference() {
    let stream = crate::test_stream();
    let mut raw = sample_q5_1_block();
    let mut second = sample_q5_1_block();
    second[8] ^= 0x5a;
    raw.extend(second);
    let native = NativeQuantizedTensor::from_q5_1_bytes(&raw, &[2, 32], stream).unwrap();
    let input_values = (0..9 * 32)
        .map(|index| (index as f32 % 29.0 - 14.0) / 20.0)
        .collect::<Vec<_>>();
    let input = Array::from_slice(&input_values, &[9, 32]);
    let actual = native.linear(&input, true, stream).unwrap();
    let dense = native.dequantize(stream).unwrap();
    let expected = matmul(&input, dense.transpose(stream).unwrap(), stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-4), Some(2e-4), None, stream)
        .unwrap()
        .item::<bool>(stream));

    let ids = Array::from_slice(&[1i32, 0, 1], &[3]);
    let actual = native.embedding(&ids, stream).unwrap();
    let expected = dense.try_index_device(&ids, stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(1e-6), Some(1e-6), None, stream)
        .unwrap()
        .item::<bool>(stream));
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn q5_1_metal_linear_prefill_and_embedding_match_dequantized_reference() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
    let mut raw = sample_q5_1_block();
    let mut second = sample_q5_1_block();
    second[8] ^= 0x5a;
    raw.extend(second);
    let native = NativeQuantizedTensor::from_q5_1_bytes(&raw, &[2, 32], &stream).unwrap();
    let input = Array::from_slice(
        &(0..9 * 32)
            .map(|index| (index as f32 % 29.0 - 14.0) / 20.0)
            .collect::<Vec<_>>(),
        &[9, 32],
    )
    .copy(&stream)
    .unwrap();
    let dense = native.dequantize(&stream).unwrap();
    let actual = native.linear(&input, true, &stream).unwrap();
    let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-4), Some(2e-4), None, &stream)
        .unwrap()
        .item::<bool>(&stream));

    let ids = Array::from_slice(&[1i32, 0, 1], &[3])
        .copy(&stream)
        .unwrap();
    let actual = native.embedding(&ids, &stream).unwrap();
    let expected = dense.try_index_device(&ids, &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(1e-6), Some(1e-6), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
}

#[test]
fn q8_0_decode_and_cpu_operations_match_affine_reference() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Cpu, 0));
    let mut raw = Vec::new();
    raw.extend(sample_q8_0_block());
    raw.extend(sample_q8_0_block());
    let native = NativeQuantizedTensor::from_q8_0_bytes(&raw, &[2, 32], &stream).unwrap();
    let actual = native.dequantize(&stream).unwrap();
    let actual = actual.evaluated().unwrap();

    let descriptor = eredu_gguf::TensorDescriptor {
        name: "test.weight".into(),
        dimensions: vec![32, 2],
        ggml_type: eredu_gguf::GgmlType::Q8_0,
        relative_offset: 0,
        data_offset: 0,
        byte_len: raw.len() as u64,
    };
    let reference =
        eredu_gguf::convert_affine(&descriptor, &raw, eredu_gguf::Endian::Little).unwrap();
    for (actual, expected) in actual.as_slice::<f32>().iter().zip(reference.dequantize()) {
        assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
    }

    let input = Array::from_slice(&vec![0.01f32; 3 * 32], &[3, 32]);
    assert_eq!(
        native.linear(&input, true, &stream).unwrap().shape(),
        &[3, 2]
    );
    let ids = Array::from_slice(&[1i32, 0], &[2]);
    assert_eq!(native.embedding(&ids, &stream).unwrap().shape(), &[2, 32]);
}

fn repeated_blocks(count: usize) -> Vec<u8> {
    let block = sample_block();
    let mut raw = Vec::with_capacity(count * block.len());
    for index in 0..count {
        let mut block = block.clone();
        block[16] = block[16].wrapping_add(index as u8);
        raw.extend(block);
    }
    raw
}

fn repeated_qk_blocks(block_bytes: usize, count: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity(block_bytes * count);
    for block_index in 0..count {
        let mut block = (0..block_bytes)
            .map(|index| {
                (index as u8)
                    .wrapping_mul(29)
                    .wrapping_add(block_index as u8)
                    .wrapping_add(11)
            })
            .collect::<Vec<_>>();
        let d = half::f16::from_f32(0.0078125).to_bits().to_le_bytes();
        if block_bytes == Q5_K_BLOCK_BYTES as usize {
            block[0..2].copy_from_slice(&d);
            block[2..4].copy_from_slice(&half::f16::from_f32(0.00390625).to_bits().to_le_bytes());
        } else {
            block[208..210].copy_from_slice(&d);
        }
        raw.extend(block);
    }
    raw
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn q5k_and_q6k_metal_decode_prefill_and_embedding_match_float() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
    for format in [
        NativeQuantizationFormat::GgufQ5K,
        NativeQuantizationFormat::GgufQ6K,
    ] {
        let (_, block_bytes) = format.block_geometry();
        let raw = repeated_qk_blocks(block_bytes as usize, 67 * 2);
        let native = match format {
            NativeQuantizationFormat::GgufQ5K => {
                NativeQuantizedTensor::from_q5k_bytes(&raw, &[67, 512], &stream).unwrap()
            }
            NativeQuantizationFormat::GgufQ6K => {
                NativeQuantizedTensor::from_q6k_bytes(&raw, &[67, 512], &stream).unwrap()
            }
            _ => unreachable!(),
        };
        let dense = native.dequantize(&stream).unwrap();
        for batch_rows in [1, 3, 8, 9, 15, 16, 17, 32, 33] {
            let input = Array::from_slice(
                &(0..batch_rows * 512)
                    .map(|index| (index as f32 % 37.0 - 18.0) / 19.0)
                    .collect::<Vec<_>>(),
                &[batch_rows, 512],
            )
            .copy(&stream)
            .unwrap();
            let actual = native.linear(&input, true, &stream).unwrap();
            let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
            assert!(
                actual
                    .all_close(
                        &expected,
                        Some(if batch_rows > 8 { 3e-2 } else { 3e-3 }),
                        Some(if batch_rows > 8 { 1.0 } else { 3e-3 }),
                        None,
                        &stream,
                    )
                    .unwrap()
                    .item::<bool>(&stream),
                "{format:?} batch-{batch_rows} linear mismatch"
            );
        }
        for dtype in [Dtype::Float16, Dtype::Bfloat16] {
            let input = Array::from_slice(
                &(0..32 * 512)
                    .map(|index| (index as f32 % 37.0 - 18.0) / 19.0)
                    .collect::<Vec<_>>(),
                &[32, 512],
            )
            .as_dtype(dtype, &stream)
            .unwrap();
            let dense = dense.as_dtype(dtype, &stream).unwrap();
            let actual = native.linear(&input, true, &stream).unwrap();
            let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
            assert!(
                actual
                    .all_close(&expected, Some(4e-2), Some(2.0), None, &stream)
                    .unwrap()
                    .item::<bool>(&stream),
                "{format:?} {dtype:?} tiled linear mismatch"
            );
        }
        let ids = Array::from_slice(&[66u32, 1, 4], &[3])
            .copy(&stream)
            .unwrap();
        let actual = native.embedding(&ids, &stream).unwrap();
        let expected = dense.try_index_device(&ids, &stream).unwrap();
        assert!(
            actual
                .all_close(&expected, Some(1e-6), Some(1e-6), None, &stream)
                .unwrap()
                .item::<bool>(&stream),
            "{format:?} embedding mismatch"
        );
    }
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn q4k_metal_linear_prefill_embedding_and_partial_tiles_match_float() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
    let raw = repeated_blocks(67 * 2);
    let native = NativeQuantizedTensor::from_q4k_bytes(&raw, &[67, 512], &stream).unwrap();
    let input = Array::from_slice(
        &(0..3 * 512)
            .map(|index| (index as f32 % 37.0 - 18.0) / 19.0)
            .collect::<Vec<_>>(),
        &[3, 512],
    )
    .copy(&stream)
    .unwrap();
    let actual = native.linear(&input, true, &stream).unwrap();
    let dense = native.dequantize(&stream).unwrap();
    let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
        .unwrap()
        .item::<bool>(&stream));

    for batch_rows in [8, 9, 15, 16, 17, 32, 33] {
        let tiled_input = Array::from_slice(
            &(0..batch_rows * 512)
                .map(|index| (index as f32 % 37.0 - 18.0) / 19.0)
                .collect::<Vec<_>>(),
            &[batch_rows, 512],
        )
        .copy(&stream)
        .unwrap();
        let actual = native.linear(&tiled_input, true, &stream).unwrap();
        let expected = matmul(&tiled_input, dense.transpose(&stream).unwrap(), &stream).unwrap();
        assert!(
            actual
                .all_close(&expected, Some(3e-2), Some(1.0), None, &stream)
                .unwrap()
                .item::<bool>(&stream),
            "Q4_K batch-{batch_rows} linear mismatch"
        );
    }
    for dtype in [Dtype::Float16, Dtype::Bfloat16] {
        let input = Array::from_slice(
            &(0..32 * 512)
                .map(|index| (index as f32 % 37.0 - 18.0) / 19.0)
                .collect::<Vec<_>>(),
            &[32, 512],
        )
        .as_dtype(dtype, &stream)
        .unwrap();
        let dense = dense.as_dtype(dtype, &stream).unwrap();
        let actual = native.linear(&input, true, &stream).unwrap();
        let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
        assert!(
            actual
                .all_close(&expected, Some(4e-2), Some(2.0), None, &stream)
                .unwrap()
                .item::<bool>(&stream),
            "Q4_K {dtype:?} tiled linear mismatch"
        );
    }

    // Decode uses the two-output-row SIMD kernel rather than the batched
    // prefill path. Keep an odd output width here to cover its partial
    // final row pair.
    let decode_input = input.try_index_device(0, &stream).unwrap();
    let actual = native.linear(&decode_input, true, &stream).unwrap();
    let expected = matmul(&decode_input, dense.transpose(&stream).unwrap(), &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
        .unwrap()
        .item::<bool>(&stream));

    let group_bank = NativeQuantizedTensor::from_q4k_bytes(&raw, &[67, 1, 512], &stream).unwrap();
    let group_ids = Array::from_slice(&[66i32, 1, 4], &[3])
        .copy(&stream)
        .unwrap();
    let actual = native_grouped_linear(&input, &group_bank, &group_ids, &stream).unwrap();
    let selected = group_bank
        .dequantize(&stream)
        .unwrap()
        .try_index_device(&group_ids, &stream)
        .unwrap();
    let expected = matmul(
        input.reshape(&[3, 1, 512], &stream).unwrap(),
        selected.swap_axes(-1, -2, &stream).unwrap(),
        &stream,
    )
    .unwrap()
    .reshape(&[3, 1], &stream)
    .unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
        .unwrap()
        .item::<bool>(&stream));

    let ids = Array::from_slice(&[66u32, 1, 4], &[3])
        .copy(&stream)
        .unwrap();
    let actual = native.embedding(&ids, &stream).unwrap();
    let expected = dense.try_index_device(&ids, &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(1e-6), Some(1e-6), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn q8_0_metal_linear_prefill_embedding_and_partial_tiles_match_float() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
    let mut raw = Vec::new();
    for index in 0..(5 * 3) {
        let mut block = sample_q8_0_block();
        block[2] = block[2].wrapping_add(index as u8);
        raw.extend(block);
    }
    let native = NativeQuantizedTensor::from_q8_0_bytes(&raw, &[5, 96], &stream).unwrap();
    let input = Array::from_slice(
        &(0..9 * 96)
            .map(|index| (index as f32 % 37.0 - 18.0) / 190.0)
            .collect::<Vec<_>>(),
        &[9, 96],
    )
    .copy(&stream)
    .unwrap();
    let actual = native.linear(&input, true, &stream).unwrap();
    let dense = native.dequantize(&stream).unwrap();
    let expected = matmul(&input, dense.transpose(&stream).unwrap(), &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
    for dtype in [Dtype::Float16, Dtype::Bfloat16] {
        let typed_input = input.as_dtype(dtype, &stream).unwrap();
        let typed_dense = dense.as_dtype(dtype, &stream).unwrap();
        let actual = native.linear(&typed_input, true, &stream).unwrap();
        let expected = matmul(
            &typed_input,
            typed_dense.transpose(&stream).unwrap(),
            &stream,
        )
        .unwrap();
        assert!(
            actual
                .all_close(&expected, Some(2e-2), Some(2e-2), None, &stream)
                .unwrap()
                .item::<bool>(&stream),
            "Q8_0 {dtype:?} linear disagrees with float reference"
        );
    }

    let decode = input.try_index_device((0..1, ..), &stream).unwrap();
    let actual = native.linear(&decode, true, &stream).unwrap();
    let expected = matmul(&decode, dense.transpose(&stream).unwrap(), &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
        .unwrap()
        .item::<bool>(&stream));

    let ids = Array::from_slice(&[4u32, 1, 4], &[3])
        .copy(&stream)
        .unwrap();
    let actual = native.embedding(&ids, &stream).unwrap();
    let expected = dense.try_index_device(&ids, &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(1e-6), Some(1e-6), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
}

#[test]
#[ignore = "requires an accessible Metal device"]
fn q8_0_metal_grouped_matches_float_with_repeated_ids() {
    let stream = crate::Stream::new_with_device(&crate::Device::new(DeviceType::Gpu, 0));
    let groups = 3;
    let output = 5;
    let input_dim = 64;
    let routes = 4;
    let mut raw = Vec::new();
    for index in 0..(groups * output * input_dim / 32) {
        let mut block = sample_q8_0_block();
        block[2] = block[2].wrapping_add(index as u8);
        raw.extend(block);
    }
    let native =
        NativeQuantizedTensor::from_q8_0_bytes(&raw, &[groups, output, input_dim], &stream)
            .unwrap();
    let input = Array::from_slice(
        &(0..routes * input_dim)
            .map(|index| (index as f32 % 37.0 - 18.0) / 190.0)
            .collect::<Vec<_>>(),
        &[routes, input_dim],
    )
    .copy(&stream)
    .unwrap();
    let ids = Array::from_slice(&[2i32, 0, 2, 1], &[routes])
        .copy(&stream)
        .unwrap();
    let selected = native
        .dequantize(&stream)
        .unwrap()
        .try_index_device(&ids, &stream)
        .unwrap();
    let expected = matmul(
        input.reshape(&[-1, 1, input_dim], &stream).unwrap(),
        selected.swap_axes(-1, -2, &stream).unwrap(),
        &stream,
    )
    .unwrap()
    .reshape(&[routes, output], &stream)
    .unwrap();
    let actual = native_grouped_linear(&input, &native, &ids, &stream).unwrap();
    assert!(actual
        .all_close(&expected, Some(2e-3), Some(2e-3), None, &stream)
        .unwrap()
        .item::<bool>(&stream));
}
