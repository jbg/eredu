//! Exact source-to-executable storage geometry for native support queries.

use crate::{LinearFormat, SourceTensorEncoding, StoredDtype};

/// Invalid shape or packing axis in an exact lowering request.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct WeightLoweringGeometryError(&'static str);

/// Exact source-to-executable lowering query presented to a backend.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightLoweringDescriptor {
    source: SourceTensorEncoding,
    executable: LinearFormat,
    physical_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    packed_axis: Option<usize>,
}

impl WeightLoweringDescriptor {
    /// Creates a geometry-bearing lowering query.
    pub fn new(
        source: SourceTensorEncoding,
        executable: LinearFormat,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
        packed_axis: Option<usize>,
    ) -> Result<Self, WeightLoweringGeometryError> {
        if physical_shape.contains(&0)
            || logical_shape.contains(&0)
            || physical_shape.len() != logical_shape.len()
        {
            return Err(WeightLoweringGeometryError(
                "weight lowering requires positive extents and equal physical and logical ranks",
            ));
        }
        if packed_axis.is_some_and(|axis| axis >= logical_shape.len()) {
            return Err(WeightLoweringGeometryError(
                "weight lowering packed axis is outside the logical shape",
            ));
        }
        Ok(Self {
            source,
            executable,
            physical_shape,
            logical_shape,
            packed_axis,
        })
    }

    /// Returns the admitted source encoding.
    pub const fn source(&self) -> &SourceTensorEncoding {
        &self.source
    }

    /// Returns the selected executable format.
    pub const fn executable(&self) -> LinearFormat {
        self.executable
    }

    /// Returns the admitted physical source shape.
    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    /// Returns the architecture-declared logical shape.
    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    /// Returns the executable packing axis, when the parameter is packable.
    pub const fn packed_axis(&self) -> Option<usize> {
        self.packed_axis
    }

    /// Returns the exact extent along the packing axis.
    pub fn packed_extent(&self) -> Option<usize> {
        self.packed_axis.map(|axis| self.logical_shape[axis])
    }
}

impl WeightLoweringDescriptor {
    /// Validates source encoding geometry without choosing a native implementation.
    pub fn has_valid_direct_geometry(&self) -> bool {
        if self.executable().validate().is_err() {
            return false;
        }
        if self.executable() != LinearFormat::Dense
            && (self.packed_axis().is_none()
                || self.packed_axis() != self.logical_shape().len().checked_sub(1))
        {
            return false;
        }
        let same_unpacked_dimensions = |packed_axis: usize| {
            self.physical_shape()
                .iter()
                .zip(self.logical_shape())
                .enumerate()
                .all(|(axis, (physical, logical))| axis == packed_axis || physical == logical)
        };
        match self.source() {
            SourceTensorEncoding::Gguf { ggml_type, .. } => ggml_type
                .block_and_bytes()
                .ok()
                .and_then(|(block, _)| usize::try_from(block).ok())
                .is_some_and(|block| match self.packed_axis() {
                    Some(axis) if same_unpacked_dimensions(axis) => {
                        let physical = self.physical_shape()[axis];
                        let logical = self.logical_shape()[axis];
                        physical >= logical
                            && physical.is_multiple_of(block)
                            && physical - logical < block
                    }
                    Some(_) => false,
                    None => self.physical_shape() == self.logical_shape(),
                }),
            SourceTensorEncoding::Safetensors(StoredDtype::U32)
            | SourceTensorEncoding::RecipeOutput(StoredDtype::U32) => {
                let Some(axis) = self.packed_axis() else {
                    return false;
                };
                if !same_unpacked_dimensions(axis) {
                    return false;
                }
                let bits = match self.executable() {
                    LinearFormat::Affine(format) => usize::try_from(format.bits).ok(),
                    LinearFormat::MxFp4 | LinearFormat::NvFp4 => Some(4),
                    _ => None,
                };
                bits.is_some_and(|bits| {
                    self.physical_shape()[axis]
                        .checked_mul(32)
                        .zip(self.logical_shape()[axis].checked_mul(bits))
                        .is_some_and(|(physical, logical)| physical == logical)
                }) && self.has_valid_packed_geometry()
            }
            SourceTensorEncoding::RecipeOutput(StoredDtype::U8)
                if matches!(self.executable(), LinearFormat::GgufIQuant { .. }) =>
            {
                let Some(axis) = self.packed_axis() else {
                    return false;
                };
                let LinearFormat::GgufIQuant { ggml_type, .. } = self.executable() else {
                    unreachable!()
                };
                same_unpacked_dimensions(axis)
                    && self.has_valid_packed_geometry()
                    && ggml_type
                        .block_and_bytes()
                        .ok()
                        .is_some_and(|(block, bytes)| {
                            let logical = self.logical_shape()[axis] as u64;
                            logical
                                .checked_div(block)
                                .and_then(|n| n.checked_mul(bytes))
                                == Some(self.physical_shape()[axis] as u64)
                        })
            }
            SourceTensorEncoding::Safetensors(StoredDtype::U8)
            | SourceTensorEncoding::RecipeOutput(StoredDtype::U8)
                if self.executable() == LinearFormat::MxFp4 =>
            {
                self.physical_shape() == self.logical_shape()
            }
            _ => {
                self.physical_shape() == self.logical_shape()
                    && (self.executable() == LinearFormat::Dense
                        || self.has_valid_packed_geometry())
            }
        }
    }

    /// Validates an unpacked source and the selected transform's packing geometry.
    pub fn has_valid_transform_geometry(&self) -> bool {
        self.physical_shape() == self.logical_shape()
            && self.packed_axis() == self.logical_shape().len().checked_sub(1)
            && self.has_valid_packed_geometry()
    }

    fn has_valid_packed_geometry(&self) -> bool {
        if self.executable() != LinearFormat::Dense
            && (self.packed_axis().is_none()
                || self.packed_axis() != self.logical_shape().len().checked_sub(1))
        {
            return false;
        }
        let Some(extent) = self.packed_extent() else {
            return false;
        };
        match self.executable() {
            LinearFormat::Affine(format) => {
                format.validate().is_ok()
                    && usize::try_from(format.group_size)
                        .ok()
                        .is_some_and(|group| extent.is_multiple_of(group))
            }
            LinearFormat::MxFp4 => extent.is_multiple_of(32),
            LinearFormat::NvFp4 => extent.is_multiple_of(16),
            LinearFormat::GgufIQuant { ggml_type, .. } => ggml_type
                .block_and_bytes()
                .ok()
                .and_then(|(block, _)| usize::try_from(block).ok())
                .is_some_and(|block| extent.is_multiple_of(block)),
            LinearFormat::E4M3BlockFp8(format) => format.validate().is_ok(),
            LinearFormat::Dense => true,
        }
    }
}
