//! Controlled temporary storage of the existing scalar native-block decoder.
use super::*;
use std::{
    alloc::Layout,
    mem::{size_of, size_of_val},
};
impl IQuantTensor {
    /// Host payload and fixed-control requests made by `dequantize_f32` for
    /// validated complete blocks. The borrowed input tensor is excluded.
    /// This does not include allocator overhead, admission or native storage.
    pub fn dequantize_f32_control_bytes(shape: &[u64], ty: GgmlType) -> Option<usize> {
        let (block, _) = ty.block_and_bytes().ok()?;
        let count = shape
            .iter()
            .try_fold(1u64, |n, &axis| n.checked_mul(axis))?;
        if shape.is_empty() || shape.iter().any(|&n| n == 0) || count % block != 0 {
            return None;
        }
        let count = usize::try_from(count).ok()?;
        let parts = [
            size_of::<(&Self, &[u64], GgmlType)>(),
            size_of::<Vec<f32>>(),
            size_of::<Result<Vec<f32>>>(),
            size_of::<Error>(),
            size_of::<Endian>(),
            size_of::<std::slice::ChunksExact<'_, u8>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<usize>().checked_mul(8)?,
            size_of::<f32>().checked_mul(6)?,
            Layout::array::<f32>(count).ok()?.size(),
        ];
        let mut bytes = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)?;
        if ty.is_iq() {
            // Scalar IQ decoding reads static codebooks, reuses one block's
            // inline grids/scales, and appends into the single final Vec.
            let controls = [
                size_of::<[u16; 4]>(),
                size_of::<[u8; 8]>(),
                size_of::<[i8; 8]>(),
                size_of::<[u8; 4]>(),
                size_of::<[f32; 2]>(),
                size_of::<u32>().checked_mul(4)?,
                size_of::<&[u8]>().checked_mul(5)?,
                size_of::<[u8; 4]>(),
                size_of::<std::array::IntoIter<u8, 8>>(),
                size_of::<std::array::IntoIter<i8, 8>>(),
            ];
            return controls.into_iter().try_fold(
                bytes.checked_add(size_of_val(&controls))?,
                usize::checked_add,
            );
        }
        let (bits, group) = affine_config(ty)?;
        if count % group as usize != 0 || count.checked_mul(bits as usize)? % 32 != 0 {
            return None;
        }
        let words = count.checked_mul(bits as usize)? / 32;
        let groups = count / group as usize;
        let scratch = match ty {
            GgmlType::Q2K => 256,
            GgmlType::Q3K => 272,
            GgmlType::Q6K => 128,
            _ => 32,
        };
        // IQuantTensor builds one descriptor. The same affine conversion
        // allocates two shapes, packed words, scales and biases before the
        // final F32 conversion. Q8's codes allocation is one reused block.
        let controls = [
            size_of::<TensorDescriptor>(),
            size_of::<AffineTensor>(),
            size_of::<AffineOutput<'_>>(),
            size_of::<Storage<'_>>(),
            size_of::<Result<AffineTensor>>(),
            size_of::<CResult<AffineOutput<'_>>>(),
            "<native blocks>".len(),
            Layout::array::<u64>(shape.len())
                .ok()?
                .size()
                .checked_mul(3)?,
            Layout::array::<u32>(words).ok()?.size(),
            Layout::array::<u16>(groups).ok()?.size().checked_mul(2)?,
            scratch,
            size_of::<destination::Codes<'_>>(),
            size_of::<&[u8]>().checked_mul(5)?,
            size_of::<Option<&[u8]>>(),
            size_of::<u32>().checked_mul(4)?,
            size_of::<usize>().checked_mul(10)?,
            size_of::<std::ops::Range<usize>>().checked_mul(3)?,
        ];
        bytes = controls.into_iter().try_fold(
            bytes.checked_add(size_of_val(&controls))?,
            usize::checked_add,
        )?;
        Some(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_blocks_have_finite_scalar_decode_storage_and_malformed_extents_refuse() {
        for ty in [
            GgmlType::IQ2XXS,
            GgmlType::IQ4NL,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q8_0,
        ] {
            let (block, bytes) = ty.block_and_bytes().unwrap();
            let shape = vec![2, block];
            let tensor = IQuantTensor {
                shape: shape.clone(),
                ggml_type: ty,
                endian: Endian::Little,
                data: vec![0; bytes as usize * 2],
            };
            let request = IQuantTensor::dequantize_f32_control_bytes(&shape, ty).unwrap();
            let values = tensor.dequantize_f32().unwrap();
            assert_eq!(values.len(), block as usize * 2);
            assert!(request >= values.capacity() * std::mem::size_of::<f32>());
            assert!(values.iter().all(|v| v.is_finite()));
            assert!(IQuantTensor::dequantize_f32_control_bytes(&[block + 1], ty).is_none());
        }
        assert!(
            IQuantTensor::dequantize_f32_control_bytes(&[u64::MAX, 2], GgmlType::Q6K).is_none()
        );
    }
}
