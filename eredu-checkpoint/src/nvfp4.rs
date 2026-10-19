//! Bounded ModelOpt NVFP4 weight decoding, independent of native execution.

use std::ops::Range;

/// Logical geometry of a packed E2M1 matrix with block-16 E4M3 scales.
/// Global F32 scales apply to contiguous, equally sized output-row partitions.
/// A fused gate/up matrix uses two partitions; an ordinary matrix uses one.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct NvFp4MatrixFormat {
    rows: usize,
    columns: usize,
    scale_partitions: usize,
}

/// Failure before or during bounded NVFP4 decoding.
#[derive(Debug, Clone, thiserror::Error, Eq, PartialEq)]
pub enum NvFp4Error {
    /// Shapes, selected rows or packed companion lengths disagree.
    #[error("invalid NVFP4 matrix geometry")]
    Geometry,
    /// A block/global scale is nonfinite, negative or overflows F32 decoding.
    #[error("invalid NVFP4 scale at output row {row}, block {block}")]
    Scale {
        /// Logical output row.
        row: usize,
        /// Input block within the row.
        block: usize,
    },
    /// The selected output exceeds the caller's explicit byte budget.
    #[error("NVFP4 decoding needs {required} bytes, exceeding limit {limit}")]
    Budget {
        /// Exact F32 output size.
        required: usize,
        /// Caller's conversion budget.
        limit: usize,
    },
}

impl NvFp4MatrixFormat {
    /// Validates matrix and fused-projection scale geometry without reading values.
    pub fn new(rows: usize, columns: usize, scale_partitions: usize) -> Result<Self, NvFp4Error> {
        if rows == 0
            || columns == 0
            || columns % 16 != 0
            || scale_partitions == 0
            || rows % scale_partitions != 0
            || rows.checked_mul(columns).is_none()
        {
            return Err(NvFp4Error::Geometry);
        }
        Ok(Self {
            rows,
            columns,
            scale_partitions,
        })
    }

    /// Logical output width.
    pub const fn rows(self) -> usize {
        self.rows
    }

    /// Logical input width.
    pub const fn columns(self) -> usize {
        self.columns
    }

    /// Number of contiguous row partitions with independent global scales.
    pub const fn scale_partitions(self) -> usize {
        self.scale_partitions
    }

    /// Exact U8 shape, with the earlier column in each byte's low nibble.
    pub const fn packed_shape(self) -> [usize; 2] {
        [self.rows, self.columns / 2]
    }

    /// Exact E4M3 block-scale shape.
    pub const fn block_scale_shape(self) -> [usize; 2] {
        [self.rows, self.columns / 16]
    }

    /// Decodes only the selected contiguous output rows into bounded F32 storage.
    /// Packed bytes and companions remain borrowed. Input-quantizer scales are
    /// not weight-dequantization factors and are not multiplied into weights.
    pub fn decode_rows(
        self,
        packed: &[u8],
        block_scales: &[u8],
        global_scales: &[f32],
        rows: Range<usize>,
        output_byte_limit: usize,
    ) -> Result<Vec<f32>, NvFp4Error> {
        if packed.len() != self.rows * (self.columns / 2)
            || block_scales.len() != self.rows * (self.columns / 16)
            || global_scales.len() != self.scale_partitions
            || rows.start > rows.end
            || rows.end > self.rows
        {
            return Err(NvFp4Error::Geometry);
        }
        let elements = (rows.end - rows.start)
            .checked_mul(self.columns)
            .ok_or(NvFp4Error::Geometry)?;
        let required = elements.checked_mul(4).ok_or(NvFp4Error::Geometry)?;
        if required > output_byte_limit {
            return Err(NvFp4Error::Budget {
                required,
                limit: output_byte_limit,
            });
        }
        let blocks = self.columns / 16;
        let partition_rows = self.rows / self.scale_partitions;
        let mut result = Vec::with_capacity(elements);
        for row in rows {
            let global = global_scales[row / partition_rows];
            if !global.is_finite() || global <= 0.0 {
                return Err(NvFp4Error::Scale { row, block: 0 });
            }
            for block in 0..blocks {
                let scale_byte = block_scales[row * blocks + block];
                let scale = decode_scale(scale_byte).ok_or(NvFp4Error::Scale { row, block })?;
                let multiplier = scale * global;
                if !multiplier.is_finite() {
                    return Err(NvFp4Error::Scale { row, block });
                }
                let first_byte = row * (self.columns / 2) + block * 8;
                for byte in &packed[first_byte..first_byte + 8] {
                    for bits in [byte & 15, byte >> 4] {
                        let value = decode_value(bits) * multiplier;
                        if !value.is_finite() {
                            return Err(NvFp4Error::Scale { row, block });
                        }
                        result.push(value);
                    }
                }
            }
        }
        Ok(result)
    }
}

fn decode_value(bits: u8) -> f32 {
    let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][usize::from(bits & 7)];
    if bits & 8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn decode_scale(bits: u8) -> Option<f32> {
    if bits & 128 != 0 || bits == 127 {
        return None;
    }
    let exponent = (bits >> 3) & 15;
    let fraction = f32::from(bits & 7);
    Some(if exponent == 0 {
        fraction * (1.0 / 512.0)
    } else {
        (1.0 + fraction / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    })
}
