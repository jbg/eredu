use eredu_gguf::{
    BlockDecoder, Checkpoint, ConvertedTensor, Endian, GgmlType, LogicalDtype, TensorInput,
    TensorSelection, Writer, WriterOptions,
};
use std::collections::BTreeMap;

fn half(value: f32, endian: Endian) -> [u8; 2] {
    let bits = half::f16::from_f32(value).to_bits();
    match endian {
        Endian::Little => bits.to_le_bytes(),
        Endian::Big => bits.to_be_bytes(),
    }
}

#[test]
fn block_decoding_matches_hash_pinned_upstream_c_f32_bits() {
    let hex = |text: &str| {
        text.as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect::<Vec<_>>()
    };
    for line in include_str!("fixtures/scalar-block-rows.txt")
        .lines()
        .filter(|l| !l.starts_with('#'))
    {
        let columns: Vec<_> = line.split('|').collect();
        let ty = GgmlType::from_code(columns[0].parse().unwrap());
        let original = hex(columns[1]);
        let expected = hex(columns[2]);
        for endian in [Endian::Little, Endian::Big] {
            let mut encoded = original.clone();
            if endian == Endian::Big {
                for block in encoded.chunks_exact_mut(ty.block_and_bytes().unwrap().1 as usize) {
                    match ty {
                        GgmlType::Q4_0 | GgmlType::Q8_0 => block[..2].reverse(),
                        GgmlType::Q4K | GgmlType::Q5K => {
                            block[..2].reverse();
                            block[2..4].reverse();
                        }
                        GgmlType::Q6K => block[208..210].reverse(),
                        GgmlType::Q5_1 => {
                            block[..2].reverse();
                            block[2..4].reverse();
                            block[4..8].reverse();
                        }
                        GgmlType::Q4_1 => {
                            block[..2].reverse();
                            block[2..4].reverse();
                        }
                        GgmlType::Q5_0 => {
                            block[..2].reverse();
                            block[2..6].reverse();
                        }
                        GgmlType::Q2K => {
                            block[80..82].reverse();
                            block[82..84].reverse();
                        }
                        GgmlType::Q3K => block[108..110].reverse(),
                        GgmlType::MxFp4 => {}
                        _ => panic!("unexpected fixture encoding"),
                    }
                }
            }
            let mut values = vec![0.; expected.len() / 4];
            BlockDecoder::new(ty, endian)
                .unwrap()
                .decode_into(&encoded, &mut values)
                .unwrap();
            let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(
                bytes, expected,
                "{ty:?}/{endian:?}: exact upstream F32 bits"
            );
        }
    }
}

#[test]
fn scalar_rows_match_independent_codes_without_half_rounded_affine_intermediates() {
    for endian in [Endian::Little, Endian::Big] {
        for ty in [GgmlType::Q4_0, GgmlType::Q4_1, GgmlType::Q5_0] {
            let asymmetric = ty == GgmlType::Q4_1;
            let q5 = ty == GgmlType::Q5_0;
            // Large finite scales expose the overflow introduced by synthesizing
            // a F16 bias for a symmetric raw block.
            let scale = if asymmetric { 0.25 } else { 16384. };
            let mut data = half(scale, endian).to_vec();
            if asymmetric {
                data.extend(half(-1.5, endian));
            }
            let codes: Vec<u8> = (0..32)
                .map(|i| ((i * 7 + 3) % if q5 { 32 } else { 16 }) as u8)
                .collect();
            if q5 {
                let high = codes
                    .iter()
                    .enumerate()
                    .fold(0u32, |n, (i, v)| n | (u32::from(*v >> 4) << i));
                data.extend(match endian {
                    Endian::Little => high.to_le_bytes(),
                    Endian::Big => high.to_be_bytes(),
                });
            }
            data.extend((0..16).map(|i| (codes[i] & 15) | ((codes[i + 16] & 15) << 4)));
            let decoder = BlockDecoder::new(ty, endian).unwrap();
            let mut actual = vec![0.; 64];
            decoder
                .decode_into(&[data.clone(), data].concat(), &mut actual)
                .unwrap();
            let expected: Vec<_> = codes
                .iter()
                .cycle()
                .take(64)
                .map(|&code| {
                    if asymmetric {
                        code as f32 / 4. - 1.5
                    } else {
                        (i32::from(code) - if q5 { 16 } else { 8 }) as f32 * 16384.
                    }
                })
                .collect();
            assert_eq!(actual, expected, "{ty:?} {endian:?}");
            assert!(actual.iter().all(|v| v.is_finite()));
        }
        for ty in [GgmlType::Q2K, GgmlType::Q3K] {
            let q3 = ty == GgmlType::Q3K;
            let mut data = vec![0; if q3 { 110 } else { 84 }];
            if q3 {
                data[..32].fill(0b10100101);
                data[32..96].fill(0b11100100);
                for g in 0..16 {
                    let scale = (g + 24) as u8; // signed scale -8..7
                    data[96 + g % 8] |= (scale & 15) << (4 * (g / 8));
                    data[104 + g % 4] |= (scale >> 4) << (2 * (g / 4));
                }
                data[108..].copy_from_slice(&half(0.375, endian));
            } else {
                for g in 0..16 {
                    data[g] = ((15 - g) << 4 | g) as u8;
                }
                data[16..80].fill(0b11100100);
                data[80..82].copy_from_slice(&half(0.375, endian));
                data[82..84].copy_from_slice(&half(0.125, endian));
            }
            let mut actual = vec![0.; 256];
            BlockDecoder::new(ty, endian)
                .unwrap()
                .decode_into(&data, &mut actual)
                .unwrap();
            let expected: Vec<_> = (0..16)
                .flat_map(|g| {
                    let code = ((g / 2) % 4) as f32;
                    let value = if q3 {
                        0.375
                            * (g as f32 - 8.)
                            * (code
                                - if [1, 0, 1, 0, 0, 1, 0, 1][g / 2] == 0 {
                                    4.
                                } else {
                                    0.
                                })
                    } else {
                        0.375 * g as f32 * code - 0.125 * (15 - g) as f32
                    };
                    [value; 16]
                })
                .collect();
            assert_eq!(actual, expected, "{ty:?} {endian:?}");
        }
    }
}

#[test]
fn mxfp4_extreme_exponents_and_invalid_lengths_are_exact_and_bounded() {
    let codes = [
        0f32, 0.5, 1., 1.5, 2., 3., 4., 6., 0., -0.5, -1., -1.5, -2., -3., -4., -6.,
    ];
    for endian in [Endian::Little, Endian::Big] {
        let decoder = BlockDecoder::new(GgmlType::MxFp4, endian).unwrap();
        for e in [0u8, 1, 126, 127, 128, 254, 255] {
            let mut encoded = vec![e];
            encoded.extend((0..16).map(|i| i | ((15 - i) << 4)));
            let mut actual = [0.; 32];
            decoder.decode_into(&encoded, &mut actual).unwrap();
            // Use F64 exponentiation as an independent arithmetic oracle, then
            // round at the F32 output boundary. The pinned converter treats 255
            // numerically (rather than substituting an E8M0 NaN).
            let expected: Vec<_> = codes
                .iter()
                .chain(codes.iter().rev())
                .map(|&v| (v as f64 * 2f64.powi(i32::from(e) - 127)) as f32)
                .collect();
            assert_eq!(actual.as_slice(), expected);
        }
        let mut untouched = [42.; 32];
        assert!(decoder.decode_into(&[0; 16], &mut untouched).is_err());
        assert!(decoder.decode_into(&[0; 34], &mut untouched).is_err());
        assert_eq!(untouched, [42.; 32]);
        assert!(decoder.output_len(usize::MAX / 17 * 17).is_err());
        assert_eq!(decoder.output_len(0).unwrap(), 0);
    }
    assert!(BlockDecoder::new(GgmlType::F32, Endian::Little).is_err());
}

#[test]
fn encoded_catalog_projection_controls_selected_full_and_iterator_materialization() {
    for endian in [Endian::Little, Endian::Big] {
        for ty in [GgmlType::Q4_0, GgmlType::MxFp4] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("blocks.gguf");
            let row_bytes = ty.block_and_bytes().unwrap().1 as usize;
            let raw: Vec<_> = (0..4 * row_bytes).map(|i| (i * 17 + 3) as u8).collect();
            Writer::new(WriterOptions {
                endian,
                ..WriterOptions::default()
            })
            .unwrap()
            .write(
                std::fs::File::create(&path).unwrap(),
                &BTreeMap::new(),
                &[
                    TensorInput {
                        name: "table.weight",
                        dimensions: &[32, 4],
                        ggml_type: ty,
                        data: &raw,
                    },
                    TensorInput {
                        name: "other",
                        dimensions: &[2],
                        ggml_type: GgmlType::F32,
                        data: &[0; 8],
                    },
                ],
            )
            .unwrap();
            let original = Checkpoint::open(&path).unwrap();
            assert!(!original.tensors().next().unwrap().is_encoded());
            assert!(original.logical_outputs().count() > 2);
            let encoded = original
                .clone()
                .into_tensor_representation(
                    ["table.weight".into()],
                    eredu_gguf::QuantizedTensorRepresentation::Encoded,
                )
                .unwrap();
            assert_eq!(encoded.logical_outputs().count(), 2);
            let tensor = encoded.tensors().next().unwrap();
            assert!(tensor.is_encoded());
            assert!(!tensor.is_mxfp4());
            assert_eq!(tensor.affine(), None);
            assert_eq!(tensor.outputs()[0].dtype, LogicalDtype::U8);
            assert_eq!(tensor.outputs()[0].shape, [4, row_bytes as u64]);
            let mut materializer = encoded.materializer();
            let selected = materializer
                .converted_tensor_selected(
                    "table.weight",
                    &TensorSelection::Range {
                        axis: 0,
                        start: 1,
                        end: 3,
                    },
                )
                .unwrap();
            let ConvertedTensor::IQuant(selected) = selected.into_converted() else {
                panic!("encoded selected blocks")
            };
            assert_eq!(selected.data, raw[row_bytes..3 * row_bytes]);
            assert_eq!(selected.shape, [2, 32]);
            assert_eq!(selected.endian, endian);
            for full in [
                materializer.converted_tensor("table.weight").unwrap(),
                encoded.converted_tensors().next().unwrap().unwrap(),
            ] {
                let ConvertedTensor::IQuant(full) = full.into_converted() else {
                    panic!("encoded full blocks")
                };
                assert_eq!(full.data, raw);
            }
            assert!(original
                .clone()
                .into_tensor_representation(
                    ["missing".into()],
                    eredu_gguf::QuantizedTensorRepresentation::Encoded
                )
                .is_err());
            assert!(original
                .into_tensor_representation(
                    ["other".into()],
                    eredu_gguf::QuantizedTensorRepresentation::Encoded
                )
                .is_err());
        }
    }
}

#[test]
fn descriptor_extent_does_not_authorize_an_unbounded_payload_allocation() {
    use eredu_gguf::{Error, Limits, Reader};
    use std::io::Cursor;
    let bytes = vec![0u8; 4 * 8 * 32];
    let mut cursor = Cursor::new(vec![]);
    Writer::default()
        .write(
            &mut cursor,
            &BTreeMap::new(),
            &[TensorInput {
                name: "large.weight",
                dimensions: &[32, 8],
                ggml_type: GgmlType::F32,
                data: &bytes,
            }],
        )
        .unwrap();
    cursor.set_position(0);
    // A file can declare more rows than any one operation may allocate.
    let mut reader = Reader::with_limits(
        cursor,
        Limits {
            max_allocation_bytes: 128,
            ..Default::default()
        },
    )
    .unwrap();
    let descriptor = reader.tensors()[0].clone();
    assert_eq!(descriptor.byte_len, 1024);
    assert!(matches!(
        reader.read_raw(&descriptor),
        Err(Error::Limit {
            resource: "tensor allocation",
            actual: 1024,
            limit: 128
        })
    ));
    let selected = eredu_gguf::TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Range {
            axis: 0,
            start: 3,
            end: 4,
        },
    )
    .unwrap();
    assert_eq!(reader.read_raw_plan(&selected).unwrap().len(), 128);
    let too_many = eredu_gguf::TensorSelectionPlan::new(
        &descriptor,
        TensorSelection::Range {
            axis: 0,
            start: 3,
            end: 5,
        },
    )
    .unwrap();
    assert!(matches!(
        reader.read_tensor_plan(&too_many),
        Err(Error::Limit {
            actual: 256,
            limit: 128,
            ..
        })
    ));
}

#[test]
fn decoded_catalog_view_preserves_exact_blocks_and_bounds_expanded_allocations() {
    use eredu_gguf::{DenseDtype, Error, Limits, QuantizedTensorRepresentation};
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("decoded.gguf");
    let mut block = half(0.375, Endian::Little).to_vec();
    block.extend((0..16).map(|i| i | ((15 - i) << 4)));
    let data = block.repeat(8);
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &BTreeMap::new(),
            &[TensorInput {
                name: "matrix.weight",
                dimensions: &[64, 4],
                ggml_type: GgmlType::Q4_0,
                data: &data,
            }],
        )
        .unwrap();
    let cp = Checkpoint::open_with_limits(
        &path,
        Limits {
            max_allocation_bytes: 256,
            ..Default::default()
        },
    )
    .unwrap()
    .into_tensor_representation(
        ["matrix.weight".into()],
        QuantizedTensorRepresentation::DecodedF32,
    )
    .unwrap();
    let t = cp.tensors().next().unwrap();
    assert!(t.is_decoded_f32());
    assert!(!t.is_encoded());
    assert_eq!(t.outputs().len(), 1);
    assert_eq!(t.outputs()[0].shape, [4, 64]);
    assert_eq!(t.outputs()[0].dtype, LogicalDtype::F32);
    let mut materializer = cp.materializer();
    let selected = materializer
        .converted_tensor_selected(
            "matrix.weight",
            &TensorSelection::Range {
                axis: 0,
                start: 2,
                end: 3,
            },
        )
        .unwrap();
    let ConvertedTensor::Dense(selected) = selected.into_converted() else {
        panic!("dense view")
    };
    assert_eq!(selected.dtype, DenseDtype::F32);
    assert_eq!(selected.shape, [1, 64]);
    let expected: Vec<_> = (0..64)
        .map(|i| ((if i % 32 < 16 { i % 16 } else { 15 - i % 16 }) as f32 - 8.) * 0.375)
        .flat_map(f32::to_ne_bytes)
        .collect();
    assert_eq!(selected.data, expected);
    for error in [
        materializer.converted_tensor("matrix.weight").unwrap_err(),
        cp.converted_tensors().next().unwrap().unwrap_err(),
    ] {
        let error = match error {
            Error::Shard { source, .. } => *source,
            other => other,
        };
        assert!(
            matches!(
                error,
                Error::Limit {
                    actual: 1024,
                    limit: 256,
                    ..
                }
            ),
            "{error:?}"
        );
    }
    let cp = Checkpoint::open(&path)
        .unwrap()
        .into_tensor_representation(
            ["matrix.weight".into()],
            QuantizedTensorRepresentation::DecodedF32,
        )
        .unwrap();
    for full in [
        cp.materializer().converted_tensor("matrix.weight").unwrap(),
        cp.converted_tensors().next().unwrap().unwrap(),
    ] {
        let ConvertedTensor::Dense(full) = full.into_converted() else {
            panic!("dense view")
        };
        assert_eq!(full.data, expected.repeat(4));
    }
}
