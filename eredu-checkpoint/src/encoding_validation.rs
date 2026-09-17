//! Shared physical-encoding validation with scalar, allocation-free causes.

use crate::{
    AffineQuantization, AffineQuantizationMode, BlockFp8Format, Error, LinearFormat,
    WeightQuantization,
};

/// Invalid physical encoding or geometry, without allocated diagnostic storage.
///
/// Fixed validation returns these causes directly. Ordinary [`Error`] conversion
/// formats the same diagnostic as the existing checkpoint validation API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum EncodingValidationError {
    /// Integer quantization requires affine mode.
    #[error("only affine integer quantization is supported")]
    AffineMode,
    /// An affine group is neither 16 nor a positive multiple of 32.
    #[error("group_size must be 16 or a positive multiple of 32, got {0}")]
    AffineGroupSize(i32),
    /// The packed affine width is unsupported.
    #[error("bits must be one of 2, 3, 4, 5, 6, or 8, got {0}")]
    AffineBits(i32),
    /// At least one block-FP8 dimension is nonpositive.
    #[error("block-FP8 geometry must be positive, got [{rows}, {columns}]")]
    BlockFp8Geometry {
        /// Original output-row extent.
        rows: i32,
        /// Original input-column extent.
        columns: i32,
    },
    /// The shared GGUF geometry worker rejected this exact encoding.
    #[error(transparent)]
    UnsupportedGgmlType(#[from] eredu_gguf::UnsupportedGgmlType),
}

impl From<EncodingValidationError> for Error {
    fn from(value: EncodingValidationError) -> Self {
        Self::invalid(value.to_string())
    }
}

impl AffineQuantization {
    /// Checks mode, group geometry, then packed width without allocating.
    pub fn validate_fixed(self) -> Result<(), EncodingValidationError> {
        if self.mode != AffineQuantizationMode::Affine {
            return Err(EncodingValidationError::AffineMode);
        }
        if self.group_size != 16 && (self.group_size <= 0 || self.group_size % 32 != 0) {
            return Err(EncodingValidationError::AffineGroupSize(self.group_size));
        }
        if !matches!(self.bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(EncodingValidationError::AffineBits(self.bits));
        }
        Ok(())
    }
}

impl BlockFp8Format {
    /// Checks both positive block dimensions without allocating.
    pub fn validate_fixed(self) -> Result<(), EncodingValidationError> {
        if self.block_rows <= 0 || self.block_columns <= 0 {
            return Err(EncodingValidationError::BlockFp8Geometry {
                rows: self.block_rows,
                columns: self.block_columns,
            });
        }
        Ok(())
    }
}

impl WeightQuantization {
    /// Checks the selected packed format using fixed encoding causes.
    pub fn validate_fixed(self) -> Result<(), EncodingValidationError> {
        match self {
            Self::Affine(config) => config.validate_fixed(),
            Self::MxFp4 => Ok(()),
            Self::GgufIQuant { ggml_type, .. } => ggml_type
                .block_and_bytes_fixed()
                .map(|_| ())
                .map_err(EncodingValidationError::from),
        }
    }
}

impl LinearFormat {
    /// Checks the selected encoding and geometry without diagnostic allocation.
    pub fn validate_fixed(self) -> Result<(), EncodingValidationError> {
        match self {
            Self::Dense => Ok(()),
            Self::Affine(config) => config.validate_fixed(),
            Self::MxFp4 => WeightQuantization::MxFp4.validate_fixed(),
            Self::GgufIQuant { ggml_type, endian } => {
                WeightQuantization::GgufIQuant { ggml_type, endian }.validate_fixed()
            }
            Self::E4M3BlockFp8(format) => format.validate_fixed(),
        }
    }
}
