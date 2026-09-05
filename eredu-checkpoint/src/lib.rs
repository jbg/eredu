//! Backend-neutral checkpoint contracts.
//!
//! This crate describes stored tensor encodings and load-time intent. It does
//! not allocate backend tensors or execute accelerator operations.

#![warn(missing_docs)]

use eredu_gguf::{Endian, GgmlType};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};

/// Composite model-artifact and component schemas.
pub mod composite;
/// Backend-neutral logical GGUF storage and portable encoded leases.
pub mod expert;
pub mod gguf_store;
pub mod recipe;
/// Canonical SafeTensors index parsing and shard-path admission.
pub mod safetensors;
pub mod schema;
/// Backend-neutral checkpoint stores, selections, and encoded leases.
pub mod store;
/// Header-only validation of declarative SafeTensors and GGUF plans.
pub mod validation;

pub use recipe::{AtomicMatrixRecipeFamily, MatrixRecipeMember, RecipeAlias};

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

/// Physical tensor encoding recorded by an admitted artifact catalog.
///
/// This remains distinct from [`LinearFormat`], which describes the format
/// selected for an executable neural operator.
#[derive(Debug, Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum SourceTensorEncoding {
    /// Scalar storage in a SafeTensors payload.
    Safetensors(StoredDtype),
    /// One physical GGML block encoding in a GGUF shard.
    Gguf {
        /// Exact GGML tensor encoding.
        ggml_type: GgmlType,
        /// Byte order declared by the containing shard.
        endian: Endian,
    },
    /// Scalar tensor produced by an admitted architecture recipe before an
    /// optional executable-format lowering.
    RecipeOutput(StoredDtype),
}

impl SourceTensorEncoding {
    /// Returns the scalar dtype when this encoding is an unpacked tensor.
    pub fn scalar_dtype(&self) -> Option<StoredDtype> {
        match self {
            Self::Safetensors(dtype) | Self::RecipeOutput(dtype) => Some(dtype.clone()),
            Self::Gguf { ggml_type, .. } => match ggml_type {
                GgmlType::F16 => Some(StoredDtype::F16),
                GgmlType::Bf16 => Some(StoredDtype::BF16),
                GgmlType::F32 => Some(StoredDtype::F32),
                _ => None,
            },
        }
    }
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

/// Physical encoding of the scale companion for an E4M3 block-FP8 matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockFp8ScaleEncoding {
    /// Floating-point inverse scales (F16, BF16, or F32 in an artifact).
    FloatingPoint,
    /// Unsigned exponent-only E8M0 inverse scales.
    Ue8m0,
}

/// Geometry and companion encoding for an E4M3 block-FP8 matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockFp8Format {
    /// Number of output rows represented by one scale.
    pub block_rows: i32,
    /// Number of input columns represented by one scale.
    pub block_columns: i32,
    /// Physical encoding of each inverse scale.
    pub scale_encoding: BlockFp8ScaleEncoding,
}

impl BlockFp8Format {
    /// Creates a validated block-FP8 format.
    pub fn new(
        block_rows: i32,
        block_columns: i32,
        scale_encoding: BlockFp8ScaleEncoding,
    ) -> Result<Self, Error> {
        let format = Self {
            block_rows,
            block_columns,
            scale_encoding,
        };
        format.validate()?;
        Ok(format)
    }

    /// Validates positive two-dimensional block geometry.
    pub fn validate(self) -> Result<(), Error> {
        if self.block_rows <= 0 || self.block_columns <= 0 {
            return Err(Error::invalid(format!(
                "block-FP8 geometry must be positive, got [{}, {}]",
                self.block_rows, self.block_columns
            )));
        }
        Ok(())
    }
}

/// Complete physical encoding selected for one linear matrix.
///
/// Unlike [`WeightQuantization`], this type includes dense and block-FP8
/// storage, so a neural-layer specification never needs an architecture-owned
/// format enum or an out-of-band quantization flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearFormat {
    /// Ordinary floating-point matrix storage.
    Dense,
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
    /// E4M3 values with one inverse scale per two-dimensional block.
    E4M3BlockFp8(BlockFp8Format),
}

impl LinearFormat {
    /// Validates the selected physical encoding and its geometry.
    pub fn validate(self) -> Result<(), Error> {
        match self {
            Self::Dense => Ok(()),
            Self::Affine(config) => config.validate(),
            Self::MxFp4 => WeightQuantization::MxFp4.validate(),
            Self::GgufIQuant { ggml_type, endian } => {
                WeightQuantization::GgufIQuant { ggml_type, endian }.validate()
            }
            Self::E4M3BlockFp8(format) => format.validate(),
        }
    }

    /// Returns the packed-quantization descriptor when this format is
    /// represented by the standard affine/GGUF materializer.
    pub const fn weight_quantization(self) -> Option<WeightQuantization> {
        match self {
            Self::Dense | Self::E4M3BlockFp8(_) => None,
            Self::Affine(config) => Some(WeightQuantization::Affine(config)),
            Self::MxFp4 => Some(WeightQuantization::MxFp4),
            Self::GgufIQuant { ggml_type, endian } => {
                Some(WeightQuantization::GgufIQuant { ggml_type, endian })
            }
        }
    }
}

impl From<WeightQuantization> for LinearFormat {
    fn from(value: WeightQuantization) -> Self {
        match value {
            WeightQuantization::Affine(config) => Self::Affine(config),
            WeightQuantization::MxFp4 => Self::MxFp4,
            WeightQuantization::GgufIQuant { ggml_type, endian } => {
                Self::GgufIQuant { ggml_type, endian }
            }
        }
    }
}

impl From<Option<WeightQuantization>> for LinearFormat {
    fn from(value: Option<WeightQuantization>) -> Self {
        value.map_or(Self::Dense, Into::into)
    }
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
