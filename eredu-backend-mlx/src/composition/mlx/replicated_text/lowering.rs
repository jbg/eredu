use super::*;

pub(super) fn valid_packed_geometry(descriptor: &WeightLoweringDescriptor) -> bool {
    if descriptor.executable() != LinearFormat::Dense
        && descriptor.packed_axis() != descriptor.logical_shape().len().checked_sub(1)
    {
        return false;
    }
    let Some(extent) = descriptor.packed_extent() else {
        return false;
    };
    match descriptor.executable() {
        LinearFormat::Affine(format) => usize::try_from(format.group_size)
            .ok()
            .is_some_and(|group| group != 0 && group <= extent && extent.is_multiple_of(group)),
        LinearFormat::MxFp4 => extent.is_multiple_of(32),
        LinearFormat::GgufIQuant { ggml_type, .. } => ggml_type
            .block_and_bytes()
            .ok()
            .and_then(|(block, _)| usize::try_from(block).ok())
            .is_some_and(|block| extent.is_multiple_of(block)),
        LinearFormat::E4M3BlockFp8(_) => true,
        LinearFormat::Dense => true,
    }
}

pub(super) fn valid_direct_source_geometry(descriptor: &WeightLoweringDescriptor) -> bool {
    let same_unpacked_dimensions = |packed_axis: usize| {
        descriptor
            .physical_shape()
            .iter()
            .zip(descriptor.logical_shape())
            .enumerate()
            .all(|(axis, (physical, logical))| axis == packed_axis || physical == logical)
    };
    match descriptor.source() {
        SourceTensorEncoding::Gguf { ggml_type, .. } => ggml_type
            .block_and_bytes()
            .ok()
            .and_then(|(block, _)| usize::try_from(block).ok())
            .is_some_and(|block| match descriptor.packed_axis() {
                Some(axis) if same_unpacked_dimensions(axis) => {
                    let physical = descriptor.physical_shape()[axis];
                    let logical = descriptor.logical_shape()[axis];
                    physical >= logical
                        && physical.is_multiple_of(block)
                        && physical - logical < block
                }
                Some(_) => false,
                None => descriptor.physical_shape() == descriptor.logical_shape(),
            }),
        SourceTensorEncoding::Safetensors(StoredDtype::U8)
            if descriptor.executable() == LinearFormat::MxFp4 =>
        {
            descriptor.physical_shape() == descriptor.logical_shape()
                || descriptor.packed_axis().is_some_and(|axis| {
                    let physical = descriptor.physical_shape();
                    let logical = descriptor.logical_shape();
                    physical.len() == logical.len() + 1
                        && physical.last() == Some(&16)
                        && physical[..axis] == logical[..axis]
                        && physical[axis].checked_mul(32) == Some(logical[axis])
                        && physical[axis + 1..physical.len() - 1] == logical[axis + 1..]
                })
        }
        SourceTensorEncoding::Safetensors(StoredDtype::U32) => {
            let Some(axis) = descriptor.packed_axis() else {
                return false;
            };
            if !same_unpacked_dimensions(axis) {
                return false;
            }
            let bits = match descriptor.executable() {
                LinearFormat::Affine(format) => usize::try_from(format.bits).ok(),
                LinearFormat::MxFp4 => Some(4),
                _ => None,
            };
            bits.is_some_and(|bits| {
                descriptor.physical_shape()[axis].checked_mul(32)
                    == descriptor.logical_shape()[axis].checked_mul(bits)
            }) && valid_packed_geometry(descriptor)
        }
        SourceTensorEncoding::RecipeOutput(dtype) => {
            let equivalent = WeightLoweringDescriptor::new(
                SourceTensorEncoding::Safetensors(dtype.clone()),
                descriptor.executable(),
                descriptor.physical_shape().to_vec(),
                descriptor.logical_shape().to_vec(),
                descriptor.packed_axis(),
            )
            .expect("validated recipe-output descriptor remains valid");
            valid_direct_source_geometry(&equivalent)
        }
        _ => {
            descriptor.physical_shape() == descriptor.logical_shape()
                && (descriptor.executable() == LinearFormat::Dense
                    || valid_packed_geometry(descriptor))
        }
    }
}

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
        (SourceTensorEncoding::RecipeOutput(StoredDtype::F4), LinearFormat::MxFp4) => true,
        (
            SourceTensorEncoding::Safetensors(StoredDtype::F8E4M3),
            LinearFormat::E4M3BlockFp8(format),
        ) => format.validate().is_ok(),
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
    supported && valid_direct_source_geometry(descriptor)
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
        && descriptor.physical_shape() == descriptor.logical_shape()
        && descriptor.packed_axis() == descriptor.logical_shape().len().checked_sub(1)
        && valid_packed_geometry(descriptor)
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
