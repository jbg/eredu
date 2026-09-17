//! Storage-independent transport of already converted representations.
use super::*;
/// The existing conversion result with its owning storage types preserved.
/// This transport selects no GGML conversion policy and performs no allocation.
#[derive(Debug)]
pub enum ConvertedParts<S, B8, B16, B32> {
    /// Dense native-endian bytes.
    Dense {
        shape: S,
        dtype: DenseDtype,
        data: B8,
    },
    /// Original nonlinear GGML encoding.
    IQuant {
        shape: S,
        ggml_type: GgmlType,
        endian: Endian,
        data: B8,
    },
    /// Affine packed weights and half-bit companions.
    Affine {
        weight_shape: S,
        scale_shape: S,
        bits: u8,
        group_size: u32,
        weights: B32,
        scales: B16,
        biases: B16,
    },
    /// MXFP4 packed weights and E8M0 scales.
    MxFp4 {
        weight_shape: S,
        scale_shape: S,
        weights: B32,
        scales: B8,
    },
}
impl ConvertedTensor {
    /// Move existing owners into the common representation; no copy or transform.
    pub fn into_storage_parts(self) -> ConvertedParts<Vec<u64>, Vec<u8>, Vec<u16>, Vec<u32>> {
        match self {
            Self::Dense(v) => ConvertedParts::Dense {
                shape: v.shape,
                dtype: v.dtype,
                data: v.data,
            },
            Self::IQuant(v) => ConvertedParts::IQuant {
                shape: v.shape,
                ggml_type: v.ggml_type,
                endian: v.endian,
                data: v.data,
            },
            Self::Affine(v) => ConvertedParts::Affine {
                weight_shape: v.weight_shape,
                scale_shape: v.scale_shape,
                bits: v.bits,
                group_size: v.group_size,
                weights: v.weights,
                scales: v.scales,
                biases: v.biases,
            },
            Self::MxFp4(v) => ConvertedParts::MxFp4 {
                weight_shape: v.weight_shape,
                scale_shape: v.scale_shape,
                weights: v.weights,
                scales: v.scales,
            },
        }
    }
}
impl<F: crate::StorageFamily> crate::StoredConvertedTensor<F> {
    /// Move supplied owners into the same common representation intact.
    pub fn into_storage_parts(
        self,
    ) -> ConvertedParts<
        crate::StoredBuffer<F, u64>,
        crate::StoredBuffer<F, u8>,
        crate::StoredBuffer<F, u16>,
        crate::StoredBuffer<F, u32>,
    > {
        match self {
            Self::Dense { shape, dtype, data } => ConvertedParts::Dense { shape, dtype, data },
            Self::IQuant {
                shape,
                ggml_type,
                endian,
                data,
            } => ConvertedParts::IQuant {
                shape,
                ggml_type,
                endian,
                data,
            },
            Self::Affine {
                weight_shape,
                scale_shape,
                bits,
                group_size,
                weights,
                scales,
                biases,
            } => ConvertedParts::Affine {
                weight_shape,
                scale_shape,
                bits,
                group_size,
                weights,
                scales,
                biases,
            },
            Self::MxFp4 {
                weight_shape,
                scale_shape,
                weights,
                scales,
            } => ConvertedParts::MxFp4 {
                weight_shape,
                scale_shape,
                weights,
                scales,
            },
        }
    }
}
/// Same checked packed-shape equation used by the owning `IQuantTensor` adapter.
pub fn packed_iquant_shape(shape: &[u64], ggml_type: GgmlType) -> Result<Vec<u64>> {
    super::iquant_packed_shape(shape, ggml_type)
}
