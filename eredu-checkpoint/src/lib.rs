//! Backend-neutral checkpoint contracts.
//!
//! This crate describes stored tensor encodings and load-time intent. It does
//! not allocate backend tensors or execute accelerator operations.

#![warn(missing_docs)]

use eredu_gguf::{Endian, GgmlType};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

pub mod schema;
/// Backend-neutral checkpoint stores, selections, and encoded leases.
pub mod store;
/// Header-only validation of declarative SafeTensors and GGUF plans.
pub mod validation;

/// Backend-neutral description of a checkpoint's stored scalar encoding.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum StoredDtype {
    /// Boolean values.
    Bool,
    /// Unsigned 8-bit integers.
    U8,
    /// Signed 8-bit integers.
    I8,
    /// Signed 16-bit integers.
    I16,
    /// Unsigned 16-bit integers.
    U16,
    /// IEEE half-precision floating point.
    F16,
    /// Brain floating point.
    BF16,
    /// Signed 32-bit integers.
    I32,
    /// Unsigned 32-bit integers.
    U32,
    /// IEEE single-precision floating point.
    F32,
    /// IEEE double-precision floating point.
    F64,
    /// Signed 64-bit integers.
    I64,
    /// Unsigned 64-bit integers.
    U64,
    /// Complex values with two 32-bit floating-point components.
    C64,
    /// Encoded FP8 E4M3 bytes.
    F8E4M3,
    /// Packed FP4 E2M1 values.
    F4,
    /// Unsigned E8M0 scale bytes used by MX formats.
    F8E8M0,
    /// Encoded FP8 E5M2 bytes.
    F8E5M2,
    /// Another storage encoding not represented by a named variant.
    Other(String),
}

/// Invalid backend-neutral checkpoint metadata.
#[derive(Debug, Clone, thiserror::Error, Eq, PartialEq)]
#[error("{0}")]
pub struct Error(String);

impl Error {
    /// Creates a checkpoint metadata error.
    pub fn invalid(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

/// Per-group affine integer quantization stored alongside a checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct AffineQuantization {
    /// Number of adjacent input values sharing one scale and bias.
    pub group_size: i32,
    /// Packed bit width for each weight value.
    pub bits: i32,
    /// Quantization mode.
    #[serde(default = "default_affine_mode")]
    pub mode: AffineQuantizationMode,
}

impl Default for AffineQuantization {
    fn default() -> Self {
        Self {
            group_size: 64,
            bits: 4,
            mode: AffineQuantizationMode::Affine,
        }
    }
}

impl AffineQuantization {
    /// Creates and validates an affine encoding.
    pub fn new(group_size: i32, bits: i32) -> Result<Self, Error> {
        let value = Self {
            group_size,
            bits,
            mode: AffineQuantizationMode::Affine,
        };
        value.validate()?;
        Ok(value)
    }

    /// Validates the portable affine storage geometry.
    pub fn validate(self) -> Result<(), Error> {
        if self.mode != AffineQuantizationMode::Affine {
            return Err(Error::invalid(
                "only affine integer quantization is supported",
            ));
        }
        if self.group_size != 16 && (self.group_size <= 0 || self.group_size % 32 != 0) {
            return Err(Error::invalid(format!(
                "group_size must be 16 or a positive multiple of 32, got {}",
                self.group_size
            )));
        }
        if !matches!(self.bits, 2 | 3 | 4 | 5 | 6 | 8) {
            return Err(Error::invalid(format!(
                "bits must be one of 2, 3, 4, 5, 6, or 8, got {}",
                self.bits
            )));
        }
        Ok(())
    }
}

const fn default_affine_mode() -> AffineQuantizationMode {
    AffineQuantizationMode::Affine
}

/// Portable affine quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AffineQuantizationMode {
    /// Per-group scale-and-bias affine quantization.
    Affine,
}

/// Packed physical encoding of a model weight. Dense storage is represented
/// by `None` at the use site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightQuantization {
    /// Per-group affine integer storage.
    Affine(AffineQuantization),
    /// Microscaling FP4 with E2M1 values and E8M0 scales.
    MxFp4,
    /// Checkpoint-native GGML blocks.
    GgufIQuant {
        /// Native GGML tensor encoding.
        ggml_type: GgmlType,
        /// Byte order declared by the GGUF container.
        endian: Endian,
    },
}

impl WeightQuantization {
    /// MXFP4 group size fixed by the format.
    pub const MXFP4_GROUP_SIZE: i32 = 32;
    /// MXFP4 packed value width fixed by the format.
    pub const MXFP4_BITS: i32 = 4;

    /// Returns the grouping used by packed execution.
    pub fn group_size(self) -> i32 {
        match self {
            Self::Affine(config) => config.group_size,
            Self::MxFp4 => Self::MXFP4_GROUP_SIZE,
            Self::GgufIQuant { ggml_type, .. } => {
                ggml_type.block_and_bytes().expect("validated GGML type").0 as i32
            }
        }
    }

    /// Returns the packed storage width.
    pub fn bits(self) -> i32 {
        match self {
            Self::Affine(config) => config.bits,
            Self::MxFp4 => Self::MXFP4_BITS,
            Self::GgufIQuant { ggml_type, .. } => {
                ggml_type.block_and_bytes().expect("validated GGML type").1 as i32
            }
        }
    }

    /// Returns whether the encoding stores affine bias companions.
    pub const fn has_biases(self) -> bool {
        matches!(self, Self::Affine(_))
    }

    /// Returns checkpoint-native GGML metadata, when present.
    pub const fn gguf_iquant(self) -> Option<(GgmlType, Endian)> {
        match self {
            Self::GgufIQuant { ggml_type, endian } => Some((ggml_type, endian)),
            _ => None,
        }
    }

    /// Validates portable storage geometry.
    pub fn validate(self) -> Result<(), Error> {
        match self {
            Self::Affine(config) => config.validate(),
            Self::MxFp4 => Ok(()),
            Self::GgufIQuant { ggml_type, .. } => ggml_type
                .block_and_bytes()
                .map(|_| ())
                .map_err(|error| Error::invalid(error.to_string())),
        }
    }
}

impl From<AffineQuantization> for WeightQuantization {
    fn from(value: AffineQuantization) -> Self {
        Self::Affine(value)
    }
}

#[derive(Serialize, Deserialize)]
struct WeightQuantizationMetadata {
    group_size: i32,
    bits: i32,
    #[serde(default = "default_quantization_mode")]
    mode: String,
}

fn default_quantization_mode() -> String {
    "affine".into()
}

impl Serialize for WeightQuantization {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mode = match self {
            Self::Affine(_) => "affine",
            Self::MxFp4 => "mxfp4",
            Self::GgufIQuant { .. } => {
                return Err(serde::ser::Error::custom(
                    "checkpoint-native GGML block metadata is not serializable",
                ))
            }
        };
        WeightQuantizationMetadata {
            group_size: self.group_size(),
            bits: self.bits(),
            mode: mode.into(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for WeightQuantization {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let metadata = WeightQuantizationMetadata::deserialize(deserializer)?;
        match metadata.mode.as_str() {
            "affine" => AffineQuantization::new(metadata.group_size, metadata.bits)
                .map(Self::Affine)
                .map_err(de::Error::custom),
            "mxfp4"
                if metadata.group_size == Self::MXFP4_GROUP_SIZE
                    && metadata.bits == Self::MXFP4_BITS =>
            {
                Ok(Self::MxFp4)
            }
            "mxfp4" => Err(de::Error::custom(format!(
                "MXFP4 requires group_size=32 and bits=4, got group_size={} bits={}",
                metadata.group_size, metadata.bits
            ))),
            mode => Err(de::Error::custom(format!(
                "unsupported quantization mode {mode:?}"
            ))),
        }
    }
}
