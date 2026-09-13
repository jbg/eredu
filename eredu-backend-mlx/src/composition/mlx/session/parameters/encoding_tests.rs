use super::*;
use crate::backend::nn::linear::PhysicalLinear;
use crate::module::Module;
use eredu_checkpoint::{BlockFp8Format, BlockFp8ScaleEncoding};
use eredu_gguf::{Endian, GgmlType};
use safemlx::{Device, DeviceType};
#[path = "grouped_encoding_tests.rs"]
mod grouped;

fn host(array: &Array) -> Vec<f32> {
    array.evaluated().unwrap().as_slice::<f32>().to_vec()
}

fn verify_encoding(
    format: LinearFormat,
    rows: i32,
    columns: i32,
    packed: Array,
    scale: Option<Array>,
    expected: Vec<f32>,
    stream: &Stream,
) {
    let mut layout = EffectiveLayout::new(format, vec![rows as u64, columns as u64]);
    let mut values = BTreeMap::from([("weight".into(), MlxTensor::from_array(packed.clone()))]);
    if let Some(scale) = &scale {
        layout.bind_companion(LinearCompanionRole::Scale, "scale".into());
        values.insert("scale".into(), MlxTensor::from_array(scale.clone()));
    }
    assert!(layout.supported(
        &packed.shape().iter().map(|v| *v as u64).collect::<Vec<_>>(),
        packed.dtype()
    ));
    let effective = layout.effective("weight", &values, stream).unwrap();
    assert_eq!(host(&effective), expected);
    let mut linear = PhysicalLinear::unloaded(columns, rows, false, format, stream).unwrap();
    linear.weight.value = packed.clone();
    if matches!(format, LinearFormat::E4M3BlockFp8(_)) {
        linear.weight_scale_inv.value = scale;
        assert!(layout.conversion_and_loan_bytes().unwrap() >= 128 * 128 * 8);
    } else {
        linear.scales.value = scale;
    }
    let input_values: Vec<f32> = (0..columns).map(|c| (c % 5 - 2) as f32 * 0.125).collect();
    let input = Array::from_slice(&input_values, &[1, columns]);
    let original = host(&linear.forward(&input, stream).unwrap());
    let reference: Vec<f32> = expected
        .chunks(columns as usize)
        .map(|row| row.iter().zip(&input_values).map(|(w, x)| w * x).sum())
        .collect();
    assert_eq!(original, reference);
    // A completed copy-on-write publication replaces only the affected weight.
    // The packed format and companions remain unchanged for exact restoration.
    let mut edited = expected.clone();
    edited[columns as usize + 1] += 0.375;
    linear.weight.value = Array::from_slice(&edited, &[rows, columns]);
    values.insert(
        "weight".into(),
        MlxTensor::from_array(linear.weight.value.clone()),
    );
    assert_eq!(
        host(&layout.effective("weight", &values, stream).unwrap()),
        edited
    );
    let edited_reference: Vec<f32> = edited
        .chunks(columns as usize)
        .map(|row| row.iter().zip(&input_values).map(|(w, x)| w * x).sum())
        .collect();
    assert_eq!(
        host(&linear.forward(&input, stream).unwrap()),
        edited_reference
    );
    assert_ne!(edited_reference, original);
    linear.weight.value = packed;
    assert_eq!(host(&linear.forward(&input, stream).unwrap()), original);
    if let Some(quantization) = format.weight_quantization() {
        let crate::backend::nn::linear::PhysicalEmbedding::Quantized(mut embedding) =
            crate::backend::nn::linear::unloaded_embedding(
                rows,
                columns,
                Some(quantization),
                stream,
            )
            .unwrap()
        else {
            panic!("packed embedding");
        };
        let original_embedding = linear
            .weight
            .value
            .reshape(embedding.inner.weight.value.shape(), stream)
            .unwrap();
        embedding.inner.weight.value = original_embedding.clone();
        embedding.scales.value = linear.scales.value.clone();
        embedding.biases.value = linear.biases.value.clone();
        let indices = Array::from_slice(&[1u32, 2], &[2]);
        let selected = |values: &[f32]| values[columns as usize..3 * columns as usize].to_vec();
        assert_eq!(
            host(
                &embedding
                    .forward(&indices, stream)
                    .unwrap()
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
            ),
            selected(&expected)
        );
        assert_eq!(
            host(&embedding.as_linear(&input, stream).unwrap()),
            original
        );
        embedding.inner.weight.value = Array::from_slice(&edited, &[rows, columns]);
        assert_eq!(
            host(
                &embedding
                    .forward(&indices, stream)
                    .unwrap()
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
            ),
            selected(&edited)
        );
        assert_eq!(
            host(&embedding.as_linear(&input, stream).unwrap()),
            edited_reference
        );
        embedding.inner.weight.value = original_embedding;
        assert_eq!(
            host(
                &embedding
                    .forward(&indices, stream)
                    .unwrap()
                    .as_dtype(Dtype::Float32, stream)
                    .unwrap()
            ),
            selected(&expected)
        );
        assert_eq!(
            host(&embedding.as_linear(&input, stream).unwrap()),
            original
        );
    }
}

#[test]
fn packed_parameter_formats_decode_edit_and_restore_their_effective_values() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let table = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let mut codes = vec![0u32; 4 * 4];
    let mut expected = Vec::new();
    for row in 0..4 {
        for col in 0..32 {
            let code = (row * 5 + col * 3) % 16;
            codes[row * 4 + col / 8] |= (code as u32) << (col % 8 * 4);
            expected.push(
                table[code % 8] * if code < 8 { 1.0 } else { -1.0 } * 2.0f32.powi(row as i32 - 1),
            );
        }
    }
    verify_encoding(
        LinearFormat::MxFp4,
        4,
        32,
        Array::from_slice(&codes, &[4, 4]),
        Some(Array::from_slice(&[126u8, 127, 128, 129], &[4, 1])),
        expected,
        &stream,
    );
    for scale_encoding in [
        BlockFp8ScaleEncoding::FloatingPoint,
        BlockFp8ScaleEncoding::Ue8m0,
    ] {
        let codes: Vec<u8> = (0..15)
            .map(|i| [0x38, 0xb8, 0x30, 0xb0, 0x40][i % 5])
            .collect();
        let expected: Vec<f32> = (0..15)
            .map(|i| [2.0, -2.0, 1.0, -1.0, 4.0][i % 5])
            .collect();
        let scale = match scale_encoding {
            BlockFp8ScaleEncoding::FloatingPoint => Array::from_slice(&[2.0f32], &[1, 1]),
            BlockFp8ScaleEncoding::Ue8m0 => Array::from_slice(&[128u8], &[1, 1]),
        };
        verify_encoding(
            LinearFormat::E4M3BlockFp8(BlockFp8Format::new(128, 128, scale_encoding).unwrap()),
            3,
            5,
            Array::from_slice(&codes, &[3, 5]),
            Some(scale),
            expected,
            &stream,
        );
    }
    for endian in [Endian::Little, Endian::Big] {
        for ggml_type in [GgmlType::Q4K, GgmlType::Q5_1] {
            let columns = if ggml_type == GgmlType::Q4K { 256 } else { 32 };
            let mut bytes = Vec::new();
            let mut expected = Vec::new();
            for row in 0..3 {
                let d = [0.125f32, 0.25, 0.5][row];
                let scale = [0x3000u16, 0x3400, 0x3800][row];
                bytes.extend(match endian {
                    Endian::Little => scale.to_le_bytes(),
                    Endian::Big => scale.to_be_bytes(),
                });
                let offset = if ggml_type == GgmlType::Q4K {
                    0x2c00u16
                } else {
                    0x3800u16
                };
                bytes.extend(match endian {
                    Endian::Little => offset.to_le_bytes(),
                    Endian::Big => offset.to_be_bytes(),
                });
                if ggml_type == GgmlType::Q4K {
                    let mut scales = [0u8; 12];
                    for g in 0..4 {
                        scales[g] = (g + 1) as u8;
                        scales[g + 4] = (g % 3) as u8;
                        scales[g + 8] = (g + 5) as u8 | (((g + 4) % 3) as u8) << 4;
                    }
                    bytes.extend(scales);
                    let code = |g: usize, i: usize| ((row * 7 + g * 3 + i * 5) % 16) as u8;
                    for pair in 0..4 {
                        for i in 0..32 {
                            bytes.push(code(pair * 2, i) | code(pair * 2 + 1, i) << 4);
                        }
                    }
                    for g in 0..8 {
                        for i in 0..32 {
                            expected.push(
                                d * (g + 1) as f32 * code(g, i) as f32 - 0.0625 * (g % 3) as f32,
                            );
                        }
                    }
                } else {
                    let code = |i: usize| ((row * 7 + i * 3) % 32) as u8;
                    let mut high = 0u32;
                    for i in 0..32 {
                        high |= ((code(i) >> 4) as u32) << i;
                        expected.push(d * code(i) as f32 + 0.5);
                    }
                    bytes.extend(match endian {
                        Endian::Little => high.to_le_bytes(),
                        Endian::Big => high.to_be_bytes(),
                    });
                    for i in 0..16 {
                        bytes.push((code(i) & 15) | (code(i + 16) & 15) << 4);
                    }
                }
            }
            verify_encoding(
                LinearFormat::GgufIQuant { ggml_type, endian },
                3,
                columns,
                Array::from_slice(&bytes, &[bytes.len() as i32]),
                None,
                expected,
                &stream,
            );
        }
        let mut bytes = Vec::new();
        let mut expected = Vec::new();
        for row in 0..3 {
            // Exact FP16 0.125, 0.25, 0.5.
            let bits = [0x3000u16, 0x3400, 0x3800][row];
            bytes.extend(match endian {
                Endian::Little => bits.to_le_bytes(),
                Endian::Big => bits.to_be_bytes(),
            });
            for col in 0..32 {
                let code = ((col * 7 + row * 3) % 31) as i8 - 15;
                bytes.push(code as u8);
                expected.push(code as f32 * [0.125, 0.25, 0.5][row]);
            }
        }
        let packed = Array::from_slice(&bytes, &[bytes.len() as i32]);
        let format = LinearFormat::GgufIQuant {
            ggml_type: GgmlType::Q8_0,
            endian,
        };
        assert_eq!(
            EffectiveLayout::new(format, vec![3, 32])
                .host_conversion_bytes()
                .unwrap(),
            3 * 32 * 32
        );
        verify_encoding(format, 3, 32, packed, None, expected, &stream);
    }
}

#[test]
fn packed_parameter_padding_accounting_rejects_overflow() {
    let format = LinearFormat::E4M3BlockFp8(
        BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint).unwrap(),
    );
    assert!(EffectiveLayout::new(format, vec![u64::MAX, 1])
        .conversion_and_loan_bytes()
        .is_err());
    assert!(EffectiveLayout::new(format, vec![1 << 40, 1 << 40])
        .conversion_and_loan_bytes()
        .is_err());
    assert!(EffectiveLayout::new(format, vec![u64::MAX, 128, 128])
        .conversion_and_loan_bytes()
        .is_err());
    for shape in [vec![], vec![128], vec![1, 1, 128, 128]] {
        assert!(EffectiveLayout::new(format, shape)
            .conversion_and_loan_bytes()
            .is_err());
    }
    let unsupported = EffectiveLayout::new(
        LinearFormat::GgufIQuant {
            ggml_type: GgmlType::F32,
            endian: Endian::Little,
        },
        vec![3, 32],
    );
    assert!(!unsupported.supported(&[3 * 32 * 4], Dtype::Uint8));
}
