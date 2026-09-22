use super::*;

fn verify_grouped(stream: &Stream, affine: bool) {
    const GROUPS: usize = 3;
    const ROWS: usize = 4;
    const WIDTH: usize = 64;
    let mut packed = vec![0u32; GROUPS * ROWS * WIDTH / 8];
    let mut expected = Vec::new();
    let mut scales = Vec::new();
    let mut exponents = Vec::new();
    let mut biases = Vec::new();
    let table = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    for group in 0..GROUPS {
        for row in 0..ROWS {
            for block in 0..WIDTH / 32 {
                let scale = (1 + group + row + block) as f32 * 0.125;
                let bias = (group as f32 - row as f32) * 0.25;
                let exponent = 126 + ((group + row + block) % 3) as u8;
                scales.push(scale);
                biases.push(bias);
                exponents.push(exponent);
                for column in 0..32 {
                    let code = (group * 7 + row * 5 + block * 3 + column) % 16;
                    let index = (group * ROWS + row) * WIDTH + block * 32 + column;
                    packed[index / 8] |= (code as u32) << ((index % 8) * 4);
                    expected.push(if affine {
                        code as f32 * scale + bias
                    } else {
                        table[code % 8]
                            * if code < 8 { 1.0 } else { -1.0 }
                            * 2.0f32.powi(exponent as i32 - 127)
                    });
                }
            }
        }
    }
    let format = if affine {
        LinearFormat::Affine(eredu_checkpoint::AffineQuantization::new(32, 4).unwrap())
    } else {
        LinearFormat::MxFp4
    };
    let packed = Array::from_slice(&packed, &[3, 4, 8]);
    let scale = if affine {
        Array::from_slice(&scales, &[3, 4, 2])
    } else {
        Array::from_slice(&exponents, &[3, 4, 2])
    };
    let bias = affine.then(|| Array::from_slice(&biases, &[3, 4, 2]));
    let mut layout = EffectiveLayout::new(format, vec![3, 4, 64]);
    layout.bind_companion(LinearCompanionRole::Scale, "scale".into());
    let mut values = BTreeMap::from([
        ("weight".into(), MlxTensor::from_array(packed.clone())),
        ("scale".into(), MlxTensor::from_array(scale.clone())),
    ]);
    if let Some(bias) = &bias {
        layout.bind_companion(LinearCompanionRole::AffineBias, "bias".into());
        values.insert("bias".into(), MlxTensor::from_array(bias.clone()));
    }
    assert!(layout.supported(&[3, 4, 8], Dtype::Uint32));
    assert!(!layout.supported(&[2, 4, 8], Dtype::Uint32));
    assert!(!layout.supported(&[3, 4, 7], Dtype::Uint32));
    let mut effective = layout.effective("weight", &values, stream).unwrap();
    assert_eq!(effective.shape(), &[3, 4, 64]);
    assert_eq!(host(&effective), expected);
    let region = ParameterRegion {
        starts: vec![1, 1, 2],
        shape: vec![2, 2, 60],
    };
    let selected: Vec<_> = (1..3)
        .flat_map(|g| (1..3).flat_map(move |r| (2..62).map(move |c| (g * ROWS + r) * WIDTH + c)))
        .map(|i| expected[i])
        .collect();
    assert_eq!(
        read_effective(&effective, &region, stream).unwrap(),
        selected
    );
    for axis in 0..3 {
        let projection = ParameterProjection {
            region: region.clone(),
            axis,
            directions: 2,
            coefficients: (0..region.shape[axis] * 2)
                .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
                .collect(),
        };
        let output = project_effective(&effective, &projection, stream).unwrap();
        let other: Vec<_> = (0..3).filter(|a| *a != axis).collect();
        let expected_projection: Vec<_> = (0..region.shape[other[0]])
            .flat_map(|a| {
                (0..region.shape[other[1]]).flat_map(move |b| (0..2).map(move |d| (a, b, d)))
            })
            .map(|(a, b, d)| {
                (0..region.shape[axis])
                    .map(|k| {
                        let mut at = [0; 3];
                        at[other[0]] = a;
                        at[other[1]] = b;
                        at[axis] = k;
                        selected[((at[0] * 2 + at[1]) * 60 + at[2]) as usize] as f64
                            * projection.coefficients[(d * region.shape[axis] + k) as usize] as f64
                    })
                    .sum::<f64>() as f32
            })
            .collect();
        assert_eq!(output, expected_projection);
    }
    let inputs: Vec<_> = (0..6 * WIDTH)
        .map(|i| ((i % 9) as f32 - 4.0) * 0.0625)
        .collect();
    let input = Array::from_slice(&inputs, &[6, WIDTH as i32]);
    let group_ids = Array::from_slice(&[0u32, 0, 1, 1, 2, 2], &[6]);
    let project = |weight: &Array| {
        host(
            &crate::backend::nn::grouped::packed_grouped_linear(
                &input,
                weight,
                &scale,
                bias.as_ref(),
                &group_ids,
                format.weight_quantization().unwrap(),
                stream,
            )
            .unwrap(),
        )
    };
    let reference = |weights: &[f32]| -> Vec<f32> {
        (0..6)
            .flat_map(|selection| (0..ROWS).map(move |row| (selection, row)))
            .map(|(selection, row)| {
                let offset = ((selection / 2) * ROWS + row) * WIDTH;
                inputs[selection * WIDTH..(selection + 1) * WIDTH]
                    .iter()
                    .zip(&weights[offset..offset + WIDTH])
                    .map(|(x, w)| *x as f64 * *w as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    };
    let baseline = project(&packed);
    assert_eq!(baseline, reference(&expected));
    let edit = ParameterRegion {
        starts: vec![1, 2, 3],
        shape: vec![1, 1, 2],
    };
    effective = numerical::update_for_test(
        &effective,
        &edit,
        &ParameterUpdate::Add {
            values: vec![0.375, -0.625],
        },
        stream,
    )
    .unwrap();
    let mut changed = expected.clone();
    changed[(ROWS + 2) * WIDTH + 3] += 0.375;
    changed[(ROWS + 2) * WIDTH + 4] -= 0.625;
    assert_eq!(host(&effective), changed);
    assert_eq!(project(&effective), reference(&changed));
    assert_ne!(project(&effective), baseline);
    assert_eq!(project(&packed), baseline);
}

#[test]
#[ignore = "requires local native MLX execution"]
fn grouped_affine_mxfp4_parameters_project_edit_and_restore_cpu() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    for affine in [false, true] {
        verify_grouped(&stream, affine);
    }
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn grouped_affine_mxfp4_parameters_project_edit_and_restore_metal() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Gpu, 0));
    for affine in [false, true] {
        verify_grouped(&stream, affine);
    }
}

fn verify_native_bank(
    stream: &Stream,
    layout: EffectiveLayout,
    packed: Array,
    scale: Option<Array>,
    expected: Vec<f32>,
) {
    let shape: Vec<_> = layout.shape.iter().map(|v| *v as i32).collect();
    let groups = shape[0] as usize;
    let rows = shape[1] as usize;
    let width = shape[2] as usize;
    let physical: Vec<_> = packed.shape().iter().map(|v| *v as u64).collect();
    assert!(layout.supported(&physical, packed.dtype()));
    let mut values = BTreeMap::from([("weight".into(), MlxTensor::from_array(packed.clone()))]);
    if let Some(scale) = &scale {
        values.insert("scale".into(), MlxTensor::from_array(scale.clone()));
    }
    let mut effective = layout.effective("weight", &values, stream).unwrap();
    // A singleton group must retain its leading axis through GGUF decoding.
    assert_eq!(effective.shape(), shape);
    assert_eq!(host(&effective), expected);
    let region = ParameterRegion {
        starts: vec![0, 1, 2],
        shape: vec![groups as u64, 2, (width - 4) as u64],
    };
    let selected: Vec<_> = (0..groups)
        .flat_map(|g| {
            (1..3).flat_map(move |r| (2..width - 2).map(move |c| (g * rows + r) * width + c))
        })
        .map(|i| expected[i])
        .collect();
    assert_eq!(
        read_effective(&effective, &region, stream).unwrap(),
        selected
    );
    for axis in 0..3 {
        let projection = ParameterProjection {
            region: region.clone(),
            axis,
            directions: 1,
            coefficients: (0..region.shape[axis])
                .map(|i| if i % 2 == 0 { 0.5 } else { -0.25 })
                .collect(),
        };
        let output = project_effective(&effective, &projection, stream).unwrap();
        let other: Vec<_> = (0..3).filter(|a| *a != axis).collect();
        let mut reference = Vec::new();
        for a in 0..region.shape[other[0]] {
            for b in 0..region.shape[other[1]] {
                let mut sum = 0.0f64;
                for k in 0..region.shape[axis] {
                    let mut at = [0; 3];
                    at[other[0]] = a;
                    at[other[1]] = b;
                    at[axis] = k;
                    let i = ((at[0] * 2 + at[1]) * (width as u64 - 4) + at[2]) as usize;
                    sum += selected[i] as f64 * projection.coefficients[k as usize] as f64;
                }
                reference.push(sum as f32);
            }
        }
        assert_eq!(output, reference);
    }
    // Every 128-wide input block has magnitude 448, so FP8's scale is exactly
    // one. Other entries are signed exactly representable E4M3 values.
    let inputs: Vec<_> = (0..groups * 2 * width)
        .map(|i| {
            if (i % width) % 128 == 0 {
                448.0
            } else {
                [0.5, -1.0, 2.0, -2.0][i % 4]
            }
        })
        .collect();
    let input = Array::from_slice(&inputs, &[groups as i32 * 2, width as i32]);
    let ids = Array::from_slice(
        &(0..groups * 2).map(|i| (i / 2) as u32).collect::<Vec<_>>(),
        &[groups as i32 * 2],
    );
    let project = |weight: &Array| {
        let output = match layout.format {
            LinearFormat::E4M3BlockFp8(_) => crate::backend::nn::fp8::grouped_linear(
                &input,
                weight,
                scale.as_ref().unwrap(),
                &ids,
                stream,
            ),
            LinearFormat::GgufIQuant { ggml_type, endian } => {
                crate::native_quantization::native_grouped_linear_from_array(
                    &input,
                    weight,
                    &[shape[0], shape[1], shape[2]],
                    ggml_type,
                    endian,
                    &ids,
                    stream,
                )
            }
            _ => unreachable!(),
        }
        .unwrap();
        host(&output)
    };
    let reference = |weights: &[f32]| -> Vec<f32> {
        (0..groups * 2)
            .flat_map(|route| (0..rows).map(move |row| (route, row)))
            .map(|(route, row)| {
                (0..width)
                    .map(|c| {
                        inputs[route * width + c] as f64
                            * weights[((route / 2) * rows + row) * width + c] as f64
                    })
                    .sum::<f64>() as f32
            })
            .collect()
    };
    let baseline = project(&packed);
    assert_eq!(baseline, reference(&expected));
    let edit = ParameterRegion {
        starts: vec![(groups - 1) as u64, 2, 3],
        shape: vec![1, 1, 2],
    };
    effective = numerical::update_for_test(
        &effective,
        &edit,
        &ParameterUpdate::Add {
            values: vec![0.375, -0.625],
        },
        stream,
    )
    .unwrap();
    let mut changed = expected.clone();
    changed[((groups - 1) * rows + 2) * width + 3] += 0.375;
    changed[((groups - 1) * rows + 2) * width + 4] -= 0.625;
    values.insert("weight".into(), MlxTensor::from_array(effective.clone()));
    assert_eq!(
        host(&layout.effective("weight", &values, stream).unwrap()),
        changed
    );
    assert_eq!(project(&effective), reference(&changed));
    assert_ne!(project(&effective), baseline);
    assert_eq!(project(&packed), baseline);
    if matches!(layout.format, LinearFormat::E4M3BlockFp8(_)) {
        // A floating overlay also changes the projection's input arithmetic.
        // Inexact new inputs distinguish this from retaining FP8 quantization.
        let varied: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(i, value)| {
                if (i % width) % 128 == 0 {
                    *value
                } else {
                    *value + 0.037
                }
            })
            .collect();
        let input = Array::from_slice(&varied, &[groups as i32 * 2, width as i32]);
        let apply = |weight: &Array| {
            host(
                &crate::backend::nn::fp8::grouped_linear(
                    &input,
                    weight,
                    scale.as_ref().unwrap(),
                    &ids,
                    stream,
                )
                .unwrap(),
            )
        };
        let dense = |weights: &[f32]| -> Vec<f32> {
            (0..groups * 2)
                .flat_map(|route| (0..rows).map(move |row| (route, row)))
                .map(|(route, row)| {
                    (0..width)
                        .map(|c| {
                            varied[route * width + c] as f64
                                * weights[((route / 2) * rows + row) * width + c] as f64
                        })
                        .sum::<f64>() as f32
                })
                .collect()
        };
        for (actual, expected) in apply(&effective).iter().zip(dense(&changed)) {
            assert!(
                (actual - expected).abs() < 1e-4 + 1e-6 * expected.abs(),
                "{actual} != {expected}"
            );
        }
        assert!(apply(&packed)
            .iter()
            .zip(dense(&expected))
            .any(|(a, b)| (a - b).abs() > 0.01));
    }
}

fn verify_fp8_gguf_banks(stream: &Stream) {
    for groups in [1usize, 3] {
        for exponent_scales in [false, true] {
            let shape = [groups as i32, 130, 129];
            let mut bytes = Vec::new();
            let mut expected = Vec::new();
            let mut exponents = Vec::new();
            let mut scales = Vec::new();
            for g in 0..groups {
                for r in 0..2 {
                    for c in 0..2 {
                        exponents.push(125 + ((g + r + c) % 3) as u8);
                        scales.push(2.0f32.powi(-2 + ((g + r + c) % 3) as i32));
                    }
                }
                for r in 0..130 {
                    for c in 0..129 {
                        let code = (g * 3 + r * 5 + c) % 6;
                        bytes.push([0x30u8, 0x38, 0x40, 0xb0, 0xb8, 0xc0][code]);
                        expected.push(
                            [0.5, 1.0, 2.0, -0.5, -1.0, -2.0][code]
                                * scales[g * 4 + (r / 128) * 2 + c / 128],
                        );
                    }
                }
            }
            let format = LinearFormat::E4M3BlockFp8(
                BlockFp8Format::new(
                    128,
                    128,
                    if exponent_scales {
                        BlockFp8ScaleEncoding::Ue8m0
                    } else {
                        BlockFp8ScaleEncoding::FloatingPoint
                    },
                )
                .unwrap(),
            );
            let mut layout =
                EffectiveLayout::new(format, shape.iter().map(|a| *a as u64).collect());
            layout.bind_companion(LinearCompanionRole::Scale, "scale".into());
            assert_eq!(
                layout.conversion_and_loan_bytes().unwrap(),
                groups as u64 * 256 * 256 * 8
            );
            assert!(!layout.supported(&[groups as u64, 129, 130], Dtype::Uint8));
            let scale = if exponent_scales {
                Array::from_slice(&exponents, &[groups as i32, 2, 2])
            } else {
                Array::from_slice(&scales, &[groups as i32, 2, 2])
            };
            verify_native_bank(
                stream,
                layout,
                Array::from_slice(&bytes, &shape),
                Some(scale),
                expected,
            );
        }
        for endian in [Endian::Little, Endian::Big] {
            for ggml_type in [GgmlType::Q8_0, GgmlType::IQ4NL] {
                let mut bytes = Vec::new();
                let mut expected = Vec::new();
                for g in 0..groups {
                    for r in 0..4 {
                        for block in 0..2 {
                            let exponent = (g + r + block) % 3;
                            let bits = [0x3000u16, 0x3400, 0x3800][exponent];
                            let scale = [0.125f32, 0.25, 0.5][exponent];
                            bytes.extend(match endian {
                                Endian::Little => bits.to_le_bytes(),
                                Endian::Big => bits.to_be_bytes(),
                            });
                            if ggml_type == GgmlType::Q8_0 {
                                for c in 0..32 {
                                    let code = ((g * 7 + r * 3 + block * 5 + c) % 31) as i8 - 15;
                                    bytes.push(code as u8);
                                    expected.push(code as f32 * scale);
                                }
                            } else {
                                let table = [
                                    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53,
                                    69, 89, 113,
                                ];
                                let code = |c: usize| (g * 7 + r * 3 + block * 5 + c * 7) % 16;
                                for c in 0..16 {
                                    bytes.push(code(c) as u8 | (code(c + 16) as u8) << 4);
                                }
                                for c in 0..32 {
                                    expected.push(table[code(c)] as f32 * scale);
                                }
                            }
                        }
                    }
                }
                let layout = EffectiveLayout::new(
                    LinearFormat::GgufIQuant { ggml_type, endian },
                    vec![groups as u64, 4, 64],
                );
                assert!(!layout.supported(&[bytes.len() as u64 - 1], Dtype::Uint8));
                assert_eq!(
                    layout.host_conversion_bytes().unwrap(),
                    groups as u64 * 4 * 64 * 32
                );
                verify_native_bank(
                    stream,
                    layout,
                    Array::from_slice(&bytes, &[bytes.len() as i32]),
                    None,
                    expected,
                );
            }
        }
    }
}

#[test]
#[ignore = "requires local native MLX execution"]
fn grouped_fp8_gguf_parameters_project_edit_and_restore_cpu() {
    verify_fp8_gguf_banks(&Stream::new_with_device(&Device::new(DeviceType::Cpu, 0)));
}

#[test]
#[ignore = "requires local MLX Metal execution"]
fn grouped_fp8_gguf_parameters_project_edit_and_restore_metal() {
    verify_fp8_gguf_banks(&Stream::new_with_device(&Device::new(DeviceType::Gpu, 0)));
}
