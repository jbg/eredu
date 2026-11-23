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
                | StoredDtype::I64
                | StoredDtype::F8E8M0,
            ),
            LinearFormat::Dense,
        ) => true,
        (
            SourceTensorEncoding::Safetensors(
                StoredDtype::U8 | StoredDtype::I32 | StoredDtype::I64 | StoredDtype::F8E8M0,
            ),
            LinearFormat::Dense,
        ) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (
            SourceTensorEncoding::Safetensors(StoredDtype::U32),
            LinearFormat::MxFp4 | LinearFormat::NvFp4,
        ) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::U8), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::Safetensors(StoredDtype::F4), LinearFormat::MxFp4) => true,
        (SourceTensorEncoding::RecipeOutput(StoredDtype::U32), LinearFormat::Affine(format)) => {
            format.validate().is_ok()
        }
        (
            SourceTensorEncoding::RecipeOutput(StoredDtype::U32),
            LinearFormat::MxFp4 | LinearFormat::NvFp4,
        ) => true,
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

pub(super) fn gguf_affine(
    ggml_type: eredu_gguf::GgmlType,
) -> Option<eredu_checkpoint::AffineQuantization> {
    let (bits, group_size) = eredu_gguf::affine_layout(ggml_type)?;
    eredu_checkpoint::AffineQuantization::new(group_size as i32, i32::from(bits)).ok()
}
