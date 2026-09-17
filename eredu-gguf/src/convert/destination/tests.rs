use super::*;
#[allow(dead_code)]
#[path = "old.rs"]
mod old;

fn desc(ty: GgmlType, dims: Vec<u64>, bytes: u64) -> TensorDescriptor {
    TensorDescriptor {
        name: "actual.tensor.weight".into(),
        dimensions: dims,
        ggml_type: ty,
        relative_offset: 32,
        data_offset: 128,
        byte_len: bytes,
    }
}
fn equal(actual: &ConvertedTensor, expected: &old::ConvertedTensor) {
    // The untouched converter defines separate output types. Their complete
    // structural Debug representation includes every dtype/shape/packed bit.
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}
fn payload(ty: GgmlType, blocks: usize) -> (TensorDescriptor, Vec<u8>) {
    let (values, bytes) = ty.block_and_bytes().unwrap();
    let raw = (0..bytes as usize * blocks)
        .map(|i| (i.wrapping_mul(73).wrapping_add(29) % 256) as u8)
        .collect();
    (
        desc(ty, vec![values, blocks as u64], bytes * blocks as u64),
        raw,
    )
}
#[test]
fn all_dispatch_branches_match_untouched_converter_and_reuse_final_vectors() {
    for ty in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::Bf16,
        GgmlType::I8,
        GgmlType::I16,
        GgmlType::I32,
        GgmlType::I64,
        GgmlType::F64,
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::IQ2XXS,
        GgmlType::IQ2XS,
        GgmlType::IQ3XXS,
        GgmlType::IQ1S,
        GgmlType::IQ4NL,
        GgmlType::IQ3S,
        GgmlType::IQ2S,
        GgmlType::IQ4XS,
        GgmlType::IQ1M,
        GgmlType::MxFp4,
    ] {
        for endian in [Endian::Little, Endian::Big] {
            let (d, raw) = payload(ty, 3);
            let expected = old::convert(&d, &raw, endian).unwrap();
            equal(&convert(&d, &raw, endian).unwrap(), &expected);
            let mut prepared = PreparedConversion::prepare(d.clone(), endian).unwrap();
            let capacities = prepared.capacities();
            let pointers = (
                prepared.shape1.as_ptr(),
                prepared.shape2.as_ptr(),
                prepared.bytes.as_ptr(),
                prepared.words.as_ptr(),
                prepared.scales.as_ptr(),
                prepared.biases.as_ptr(),
                prepared.e8m0.as_ptr(),
            );
            let out = prepared.fill(&raw, &d, endian).unwrap();
            equal(&out, &expected);
            match &out {
                ConvertedTensor::Dense(t) => {
                    assert_eq!(t.shape.as_ptr(), pointers.0);
                    assert_eq!(t.data.as_ptr(), pointers.2);
                    assert_eq!(t.data.capacity(), capacities[2]);
                }
                ConvertedTensor::IQuant(t) => {
                    assert_eq!(t.shape.as_ptr(), pointers.0);
                    assert_eq!(t.data.as_ptr(), pointers.2);
                }
                ConvertedTensor::Affine(t) => {
                    assert_eq!(t.weight_shape.as_ptr(), pointers.0);
                    assert_eq!(t.scale_shape.as_ptr(), pointers.1);
                    assert_eq!(t.weights.as_ptr(), pointers.3);
                    assert_eq!(t.scales.as_ptr(), pointers.4);
                    assert_eq!(t.biases.as_ptr(), pointers.5);
                }
                ConvertedTensor::MxFp4(t) => {
                    assert_eq!(t.weights.as_ptr(), pointers.3);
                    assert_eq!(t.scales.as_ptr(), pointers.6);
                }
            }
        }
    }
}
#[test]
fn explicit_affine_and_q8_inline_scratch_match_old_bits_and_rounding() {
    for ty in [
        GgmlType::Q4_0,
        GgmlType::Q4_1,
        GgmlType::Q5_0,
        GgmlType::Q5_1,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ] {
        for endian in [Endian::Little, Endian::Big] {
            let (d, raw) = payload(ty, 5);
            let expected = old::convert_affine(&d, &raw, endian).unwrap();
            assert_eq!(
                format!("{:?}", convert_affine(&d, &raw, endian).unwrap()),
                format!("{expected:?}")
            );
            let mut prepared = PreparedConversion::prepare_affine(d.clone(), endian).unwrap();
            let scratch = prepared.scratch.as_ptr();
            let out = prepared.run(&raw, None).unwrap();
            assert_eq!(scratch, prepared.scratch.as_ptr());
            let ConvertedTensor::Affine(actual) = out else {
                panic!("affine")
            };
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
            if ty == GgmlType::Q8_0 {
                assert_eq!(
                    prepared.scratch.as_slice(),
                    raw[raw.len() - 32..]
                        .iter()
                        .map(|x| x ^ 128)
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}
#[test]
fn dense_bit_patterns_keep_signed_zero_nan_and_partial_word_endian_behavior() {
    for ty in [GgmlType::F16, GgmlType::Bf16, GgmlType::F32, GgmlType::F64] {
        for endian in [Endian::Little, Endian::Big] {
            let raw = [
                0, 0, 0, 128, 0, 0, 128, 127, 1, 0, 192, 127, 255, 255, 255, 255, 73,
            ];
            let d = desc(ty, vec![3], raw.len() as u64);
            let out = PreparedConversion::prepare(d.clone(), endian)
                .unwrap()
                .convert(&raw)
                .unwrap();
            equal(&out, &old::convert(&d, &raw, endian).unwrap());
        }
    }
}
#[test]
fn malformed_affine_emission_reaches_original_final_shape_error_without_growth() {
    for ty in [
        GgmlType::Q4_0,
        GgmlType::Q5_0,
        GgmlType::Q8_0,
        GgmlType::Q2K,
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
    ] {
        let (mut d, raw) = payload(ty, 3);
        d.dimensions[1] = 1;
        let expected = old::convert_affine(&d, &raw, Endian::Big).unwrap_err();
        let prepared = PreparedConversion::prepare_affine(d, Endian::Big).unwrap();
        let cap = prepared.capacities();
        let error = prepared.convert(&raw).unwrap_err();
        assert_eq!(error.to_string(), expected.to_string());
        assert!(matches!(error.cause(), ConversionDestinationError::Gguf(_)));
        assert_eq!(error.destination().capacities(), cap);
        assert!(!error.destination().words.is_empty());
        assert!(!error.destination().scales.is_empty());
    }
    let d = desc(GgmlType::MxFp4, vec![32], 18);
    let raw = [3; 18];
    equal(
        &PreparedConversion::prepare(d.clone(), Endian::Little)
            .unwrap()
            .convert(&raw)
            .unwrap(),
        &old::convert(&d, &raw, Endian::Little).unwrap(),
    );
}
#[test]
fn binding_capacity_and_repeated_use_refusals_retain_partial_final_storage() {
    let (native, native_raw) = payload(GgmlType::Q8_0, 2);
    let mut wrong_mode =
        PreparedConversion::prepare_affine(native.clone(), Endian::Little).unwrap();
    assert!(matches!(
        wrong_mode.fill(&native_raw, &native, Endian::Little),
        Err(ConversionDestinationError::Binding)
    ));
    assert!(matches!(
        convert(&native, &native_raw, Endian::Little).unwrap(),
        ConvertedTensor::IQuant(_)
    ));
    let mut wrong_mode =
        PreparedConversion::prepare_affine(native.clone(), Endian::Little).unwrap();
    let expected = old::convert(&native, &native_raw[..3], Endian::Little).unwrap_err();
    assert_eq!(
        wrong_mode
            .fill(&native_raw[..3], &native, Endian::Little)
            .unwrap_err()
            .to_string(),
        expected.to_string()
    );
    let (d, raw) = payload(GgmlType::Q4_0, 2);
    let mut destination = PreparedConversion::prepare(d.clone(), Endian::Little).unwrap();
    let mut foreign = d.clone();
    foreign.name.push_str(".foreign");
    let cap = destination.capacities();
    assert!(matches!(
        destination.fill(&raw, &foreign, Endian::Little),
        Err(ConversionDestinationError::Binding)
    ));
    assert_eq!(destination.capacities(), cap);
    assert!(matches!(
        destination.fill(&raw, &d, Endian::Little),
        Err(ConversionDestinationError::Consumed)
    ));
    let mut destination = PreparedConversion::prepare(d.clone(), Endian::Little).unwrap();
    destination.limits[3] = 0; // Private corruption: no public caller may choose a smaller accepted bound.
    let cap = destination.capacities();
    let e = destination.convert(&raw).unwrap_err();
    assert!(matches!(
        e.cause(),
        ConversionDestinationError::Capacity { .. }
    ));
    assert_eq!(e.destination().capacities(), cap);
    assert!(!e.destination().shape1.is_empty());
    let mut destination = PreparedConversion::prepare(d.clone(), Endian::Little).unwrap();
    assert!(matches!(
        destination.fill(&raw, &d, Endian::Big),
        Err(ConversionDestinationError::Binding)
    ));
}
#[test]
fn invalid_shapes_and_layout_overflow_keep_original_causes_and_prepared_prefixes() {
    for d in [
        desc(GgmlType::Q4_0, vec![], 18),
        desc(GgmlType::MxFp4, vec![31], 17),
        desc(GgmlType::Q4_0, vec![31], 18),
        desc(GgmlType::Unknown(999), vec![1], 1),
    ] {
        let raw = vec![1; d.byte_len as usize];
        let expected = old::convert(&d, &raw, Endian::Big).unwrap_err();
        let failure = PreparedConversion::prepare(d, Endian::Big).unwrap_err();
        assert_eq!(failure.to_string(), expected.to_string());
    }
    let d = desc(GgmlType::F32, vec![1], u64::MAX);
    let failure = PreparedConversion::prepare(d, Endian::Little).unwrap_err();
    assert!(matches!(
        failure.cause(),
        ConversionDestinationError::Layout
    ));
    assert_eq!(failure.destination().shape1, [1]);
    assert!(failure.destination().bytes.is_empty());
    assert_eq!(
        failure.destination().descriptor.name,
        "actual.tensor.weight"
    );
}
