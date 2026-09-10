use super::*;

pub(crate) fn supports_direct(descriptor: &WeightLoweringDescriptor) -> bool {
    let source = descriptor.source();
    let executable = descriptor.executable();
    let supported = match (source, executable) {
        (
            SourceTensorEncoding::Safetensors(
                StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32,
            ),
            LinearFormat::Dense,
        ) => true,
        (
            SourceTensorEncoding::RecipeOutput(
                StoredDtype::F16
                | StoredDtype::BF16
                | StoredDtype::F32
                | StoredDtype::U8
                | StoredDtype::I32
                | StoredDtype::F8E8M0,
            ),
            LinearFormat::Dense,
        ) => true,
        (
            SourceTensorEncoding::Safetensors(
                StoredDtype::U8 | StoredDtype::I32 | StoredDtype::F8E8M0,
            ),
            LinearFormat::Dense,
        ) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U8), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::F4), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::RecipeOutput(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (SourceTensorEncoding::RecipeOutput(StoredDtype::U32), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::RecipeOutput(StoredDtype::U8), LinearFormat::MxFp4) => true,
        (
            SourceTensorEncoding::RecipeOutput(StoredDtype::U8),
            LinearFormat::GgufIQuant { ggml_type, .. },
        ) => NativeQuantizationFormat::from_ggml_type(ggml_type).is_some(),
        (SourceTensorEncoding::RecipeOutput(StoredDtype::F4), LinearFormat::MxFp4) => true,
        (
            SourceTensorEncoding::Safetensors(StoredDtype::F8E4M3)
            | SourceTensorEncoding::RecipeOutput(StoredDtype::F8E4M3),
            LinearFormat::E4M3BlockFp8(format),
        ) => format.validate().is_ok() && format.block_rows == 128 && format.block_columns == 128,
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Dense) => {
            matches!(
                ggml_type,
                eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
            ) || gguf_affine(*ggml_type).is_some()
                || *ggml_type == eredu_gguf::GgmlType::MxFp4
                || NativeQuantizationFormat::from_ggml_type(*ggml_type).is_some()
        }
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::Affine(format)) => {
            gguf_affine(*ggml_type).is_some_and(|native| native == format)
        }
        (SourceTensorEncoding::Gguf { ggml_type, .. }, LinearFormat::MxFp4) => {
            *ggml_type == eredu_gguf::GgmlType::MxFp4
        }
        (
            SourceTensorEncoding::Gguf { ggml_type, endian },
            LinearFormat::GgufIQuant {
                ggml_type: executable,
                endian: executable_endian,
            },
        ) => {
            *ggml_type == executable
                && *endian == executable_endian
                && NativeQuantizationFormat::from_ggml_type(executable).is_some()
        }
        _ => false,
    };
    supported && descriptor.has_valid_direct_geometry()
}

pub(crate) fn supports_transform(descriptor: &WeightLoweringDescriptor) -> bool {
    let source = descriptor.source();
    let executable = descriptor.executable();
    let decodable = match source {
        SourceTensorEncoding::Safetensors(dtype) => matches!(
            dtype,
            StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32 | StoredDtype::F64
        ),
        SourceTensorEncoding::RecipeOutput(dtype) => matches!(
            dtype,
            StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32 | StoredDtype::F64
        ),
        SourceTensorEncoding::Gguf { ggml_type, .. } => matches!(
            ggml_type,
            eredu_gguf::GgmlType::F16 | eredu_gguf::GgmlType::F32 | eredu_gguf::GgmlType::Bf16
        ),
        _ => false,
    };
    decodable
        && descriptor.has_valid_transform_geometry()
        && match executable {
            LinearFormat::Affine(format) => format.validate().is_ok(),
            LinearFormat::MxFp4 => true,
            LinearFormat::Dense
            | LinearFormat::GgufIQuant { .. }
            | LinearFormat::E4M3BlockFp8(_) => false,
        }
}

pub(super) fn gguf_affine(
    ggml_type: eredu_gguf::GgmlType,
) -> Option<eredu_checkpoint::AffineQuantization> {
    let (bits, group_size) = match ggml_type {
        eredu_gguf::GgmlType::Q2K => (2, 16),
        eredu_gguf::GgmlType::Q3K => (3, 16),
        eredu_gguf::GgmlType::Q4_0 | eredu_gguf::GgmlType::Q4_1 | eredu_gguf::GgmlType::Q4K => {
            (4, 32)
        }
        eredu_gguf::GgmlType::Q5_0 | eredu_gguf::GgmlType::Q5_1 | eredu_gguf::GgmlType::Q5K => {
            (5, 32)
        }
        eredu_gguf::GgmlType::Q6K => (6, 16),
        eredu_gguf::GgmlType::Q8_0 => (8, 32),
        _ => return None,
    };
    eredu_checkpoint::AffineQuantization::new(group_size, bits).ok()
}
