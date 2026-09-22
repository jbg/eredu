//! Effective checkpoint decoding shared by native and metadata realizations.
use crate::{LinearFormat, LinearRowLayout};
use eredu_checkpoint::AffineQuantization;

/// One checkpoint decoder occurrence. Inputs carry the actual weight followed
/// by its selected scale and affine-bias companions; output geometry is exact.
/// This description grants no checkpoint, allocation or execution authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParameterDecoding {
    /// Encoding retained by the architecture's loaded parameter declaration.
    pub format: LinearFormat,
    /// Independent row-block origins of the selected physical weight layout.
    pub row_layout: LinearRowLayout,
}

/// Scalar-independent operations of the existing effective-value decoder.
/// Sources are borrowed; every returned value follows the realization's ordinary
/// ownership and completion rules.
pub trait ParameterDecodingMechanism {
    /// Numerical or metadata tensor.
    type Value;
    /// Realization's typed failure.
    type Error;
    /// Actual loaded primary slot, including a previously published dense edit.
    fn weight(&self) -> Result<&Self::Value, Self::Error>;
    /// Whether this actual primary slot already contains floating values.
    fn is_floating(&self, value: &Self::Value) -> bool;
    /// Alias the actual source without decoding a published dense replacement.
    fn alias(&self, value: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Borrow the exact declared scale companion.
    fn scale(&self) -> Result<&Self::Value, Self::Error>;
    /// Borrow the exact declared affine-bias companion.
    fn affine_bias(&self) -> Result<&Self::Value, Self::Error>;
    /// Convert through the selected realization, including an identity alias.
    fn cast_f32(&self, value: &Self::Value) -> Result<Self::Value, Self::Error>;
    /// Decode the actual affine packed matrix and F32 companions.
    fn affine(
        &self,
        weight: &Self::Value,
        scale: &Self::Value,
        bias: &Self::Value,
        config: AffineQuantization,
    ) -> Result<Self::Value, Self::Error>;
    /// Decode E2M1 packed values with their actual microscaling source.
    fn mx_fp4(&self, weight: &Self::Value, scale: &Self::Value)
        -> Result<Self::Value, Self::Error>;
    /// Decode the retained block-scaled FP8 layout.
    fn block_fp8(
        &self,
        weight: &Self::Value,
        scale: &Self::Value,
        decoding: ParameterDecoding,
    ) -> Result<Self::Value, Self::Error>;
    /// Decode the actual checkpoint-native GGML blocks in their declared order.
    fn gguf(
        &self,
        weight: &Self::Value,
        decoding: ParameterDecoding,
    ) -> Result<Self::Value, Self::Error>;
    /// Restore the checked logical tensor geometry after flat GGML decoding.
    fn restore_shape(&self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    /// Typed malformed-source failure for a nonfloating dense primary slot.
    fn invalid_dense(&self) -> Self::Error;
}

/// Decodes the same selected sources with the same conversion order as native
/// parameter access. In particular affine companions convert before decoding;
/// dense edited slots bypass the retained packed declaration.
pub fn decode_parameter<M: ParameterDecodingMechanism>(
    mechanism: M,
    decoding: ParameterDecoding,
) -> Result<M::Value, M::Error> {
    let weight = mechanism.weight()?;
    if mechanism.is_floating(weight) {
        return mechanism.alias(weight);
    }
    let decoded = match decoding.format {
        LinearFormat::Dense => return Err(mechanism.invalid_dense()),
        LinearFormat::Affine(config) => {
            let scale = mechanism.cast_f32(mechanism.scale()?)?;
            let bias = mechanism.cast_f32(mechanism.affine_bias()?)?;
            mechanism.affine(weight, &scale, &bias, config)?
        }
        LinearFormat::MxFp4 => mechanism.mx_fp4(weight, mechanism.scale()?)?,
        LinearFormat::E4M3BlockFp8(_) => {
            mechanism.block_fp8(weight, mechanism.scale()?, decoding)?
        }
        LinearFormat::GgufIQuant { .. } => {
            let decoded = mechanism.gguf(weight, decoding)?;
            mechanism.restore_shape(decoded)?
        }
    };
    mechanism.cast_f32(&decoded)
}
