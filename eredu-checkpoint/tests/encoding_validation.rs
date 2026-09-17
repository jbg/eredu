use eredu_checkpoint::{
    AffineQuantization, AffineQuantizationMode, BlockFp8Format, BlockFp8ScaleEncoding,
    EncodingValidationError, LinearFormat, WeightQuantization,
};
use eredu_gguf::{Endian, GgmlType};

#[test]
fn affine_fixed_causes_preserve_group_before_bits_and_ordinary_diagnostics() {
    for group_size in [16, 32, 64, 128, i32::MAX - 31] {
        for bits in [2, 3, 4, 5, 6, 8] {
            let value = AffineQuantization {
                group_size,
                bits,
                mode: AffineQuantizationMode::Affine,
            };
            assert_eq!(value.validate_fixed(), Ok(()));
            assert_eq!(AffineQuantization::new(group_size, bits).unwrap(), value);
        }
    }
    for group_size in [i32::MIN, -32, 0, 1, 17, 48, i32::MAX] {
        let value = AffineQuantization {
            group_size,
            bits: 1,
            mode: AffineQuantizationMode::Affine,
        };
        assert_eq!(
            value.validate_fixed(),
            Err(EncodingValidationError::AffineGroupSize(group_size))
        );
        assert_eq!(
            value.validate().unwrap_err().to_string(),
            format!("group_size must be 16 or a positive multiple of 32, got {group_size}")
        );
    }
    for bits in [i32::MIN, -1, 0, 1, 7, 9, i32::MAX] {
        let value = AffineQuantization {
            bits,
            ..Default::default()
        };
        assert_eq!(
            value.validate_fixed(),
            Err(EncodingValidationError::AffineBits(bits))
        );
        assert_eq!(
            value.validate().unwrap_err().to_string(),
            format!("bits must be one of 2, 3, 4, 5, 6, or 8, got {bits}")
        );
    }
}

#[test]
fn fixed_fp8_failures_retain_both_original_dimensions_for_both_scale_encodings() {
    for scale_encoding in [
        BlockFp8ScaleEncoding::FloatingPoint,
        BlockFp8ScaleEncoding::Ue8m0,
    ] {
        for (rows, columns) in [(1, 1), (128, 64), (i32::MAX, i32::MAX)] {
            let format = BlockFp8Format::new(rows, columns, scale_encoding).unwrap();
            assert_eq!(format.validate_fixed(), Ok(()));
        }
        for (rows, columns) in [(0, 64), (128, 0), (-1, -2), (i32::MIN, i32::MAX)] {
            let format = BlockFp8Format {
                block_rows: rows,
                block_columns: columns,
                scale_encoding,
            };
            let cause = EncodingValidationError::BlockFp8Geometry { rows, columns };
            assert_eq!(format.validate_fixed(), Err(cause));
            assert_eq!(
                LinearFormat::E4M3BlockFp8(format).validate_fixed(),
                Err(cause)
            );
            assert_eq!(
                format.validate().unwrap_err().to_string(),
                format!("block-FP8 geometry must be positive, got [{rows}, {columns}]")
            );
        }
    }
}

#[test]
fn fixed_format_dispatch_keeps_exact_ggml_causes_and_container_byte_order_policy() {
    assert_eq!(LinearFormat::Dense.validate_fixed(), Ok(()));
    assert_eq!(LinearFormat::MxFp4.validate_fixed(), Ok(()));
    assert_eq!(WeightQuantization::MxFp4.validate_fixed(), Ok(()));
    let invalid = AffineQuantization {
        group_size: 0,
        bits: 0,
        ..Default::default()
    };
    for result in [
        LinearFormat::Affine(invalid).validate_fixed(),
        WeightQuantization::Affine(invalid).validate_fixed(),
    ] {
        assert_eq!(result, Err(EncodingValidationError::AffineGroupSize(0)));
    }
    for endian in [Endian::Little, Endian::Big] {
        for ggml_type in [GgmlType::F16, GgmlType::IQ2XS, GgmlType::MxFp4] {
            let weight = WeightQuantization::GgufIQuant { ggml_type, endian };
            assert_eq!(weight.validate_fixed(), Ok(()));
            assert_eq!(LinearFormat::from(weight).validate_fixed(), Ok(()));
        }
        for ggml_type in [
            GgmlType::RemovedIQ4NL4_4,
            GgmlType::Unknown(0),
            GgmlType::Unknown(u32::MAX),
        ] {
            let weight = WeightQuantization::GgufIQuant { ggml_type, endian };
            let fixed = weight.validate_fixed().unwrap_err();
            let EncodingValidationError::UnsupportedGgmlType(cause) = fixed else {
                panic!("wrong encoding cause")
            };
            assert_eq!(cause.code(), ggml_type.code());
            assert_eq!(LinearFormat::from(weight).validate_fixed(), Err(fixed));
            assert_eq!(
                weight.validate().unwrap_err().to_string(),
                format!("unsupported GGML tensor type {}", ggml_type.code())
            );
            assert_eq!(LinearFormat::from(weight).validate(), weight.validate());
        }
    }
}
