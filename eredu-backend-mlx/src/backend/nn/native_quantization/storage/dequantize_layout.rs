//! Actual host requests of a complete standalone GGML matrix decode.
use super::*;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};
impl NativeQuantizedTensor {
    /// Quotes the same full-view constructor and decoder used by parameter
    /// inspection. No source handle, payload, or execution permission is made.
    pub(crate) fn standalone_dequantization_control_bytes(
        shape: &[i32],
        ty: GgmlType,
        endian: GgufEndian,
    ) -> Option<usize> {
        let rank = shape.len();
        if !(2..=3).contains(&rank) || shape.iter().any(|&n| n <= 0) {
            return None;
        }
        let format = NativeQuantizationFormat::from_ggml_type(ty)?;
        let (block_values, block_bytes) = format.block_geometry();
        if shape.last().copied()? % block_values != 0 {
            return None;
        }
        let count = shape
            .iter()
            .try_fold(1usize, |n, &axis| n.checked_mul(axis as usize))?;
        let raw = count
            .checked_div(block_values as usize)?
            .checked_mul(block_bytes as usize)?;
        if raw > i32::MAX as usize {
            return None;
        }
        let native_owner = usize::try_from(
            eredu_runtime::working_memory::OriginalHostMetadataCustody::shared_storage_bytes(
                Layout::new::<NativeStorage>(),
            )
            .ok()?,
        )
        .ok()?;
        let parts = [
            native_owner,
            size_of::<Self>(),
            size_of::<Result<Self, Exception>>(),
            size_of::<Array>(),
            size_of::<Result<Array, Exception>>(),
            size_of::<safemlx::EvaluatedArray<'_>>(),
            safemlx::EvaluatedArray::completed_readback_control_bytes::<u8>()?,
            size_of::<Vec<i32>>(),
            size_of::<Vec<f32>>(),
            size_of::<Result<Vec<f32>, Exception>>(),
            size_of::<(&Self, &Stream)>(),
            Layout::array::<i32>(rank).ok()?.size(),
            size_of::<usize>().checked_mul(12)?,
            size_of::<i32>().checked_mul(5)?,
            size_of::<std::ops::Range<usize>>().checked_mul(3)?,
            size_of::<&[u8]>().checked_mul(3)?,
            size_of::<f32>().checked_mul(4)?,
            size_of::<u32>(),
            size_of::<u8>().checked_mul(6)?,
        ];
        let bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?;
        let specialized = endian == GgufEndian::Little
            && matches!(
                format,
                NativeQuantizationFormat::GgufQ4K
                    | NativeQuantizationFormat::GgufQ5_1
                    | NativeQuantizationFormat::GgufQ8_0
            );
        if specialized {
            return bytes.checked_add(Layout::array::<f32>(count).ok()?.size());
        }
        let mut logical = [0u64; 3];
        for (out, &axis) in logical.iter_mut().zip(shape) {
            *out = axis as u64;
        }
        let decoder = eredu_gguf::IQuantTensor::dequantize_f32_control_bytes(&logical[..rank], ty)?;
        let controls = [
            size_of::<eredu_gguf::IQuantTensor>(),
            size_of::<[u64; 3]>(),
            Layout::array::<u64>(rank).ok()?.size(),
            raw,
            decoder,
        ];
        controls.into_iter().try_fold(
            bytes.checked_add(size_of_val(&controls))?,
            usize::checked_add,
        )
    }
}
