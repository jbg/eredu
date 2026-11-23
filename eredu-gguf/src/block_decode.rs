//! Bounded, allocation-free scalar decoding of packed GGML blocks.
//! Equations: ggml-quants.c at 2145525a4081d66ff1a87cf43ef809f95a85ac0c.
use std::ops::Range;

use crate::{packed_words::FP4_CODES, Endian, Error, GgmlType, Result};

/// A checked block decoder writing into caller-owned F32 storage. It never expands
/// a table, allocates conversion companions or rounds scales/biases through F16.
#[derive(Debug, Clone, Copy)]
pub struct BlockDecoder {
    encoding: GgmlType,
    endian: Endian,
    values: usize,
    bytes: usize,
}
impl BlockDecoder {
    /// Selects the scalar decoder for an encoding and byte order.
    pub fn new(encoding: GgmlType, endian: Endian) -> Result<Self> {
        if !matches!(
            encoding,
            GgmlType::Q4_0
                | GgmlType::Q4_1
                | GgmlType::Q5_0
                | GgmlType::Q5_1
                | GgmlType::Q4K
                | GgmlType::Q5K
                | GgmlType::Q6K
                | GgmlType::Q8_0
                | GgmlType::Q2K
                | GgmlType::Q3K
                | GgmlType::MxFp4
                | GgmlType::NvFp4
        ) && !encoding.is_iq()
        {
            return Err(Error::UnsupportedTensorType(encoding.code()));
        }
        let (values, bytes) = encoding.block_and_bytes()?;
        Ok(Self {
            encoding,
            endian,
            values: values as usize,
            bytes: bytes as usize,
        })
    }

    /// Exact output length. Truncated blocks and integer overflow reject before
    /// allocation or output mutation.
    pub fn output_len(self, encoded_bytes: usize) -> Result<usize> {
        if !encoded_bytes.is_multiple_of(self.bytes) {
            return Err(Error::tensor("<blocks>", "partial encoded block"));
        }
        (encoded_bytes / self.bytes)
            .checked_mul(self.values)
            .ok_or(Error::Overflow("block decoder output"))
    }

    /// Decodes only the supplied complete blocks. Scratch contains scalar locals;
    /// all decoded storage belongs to the caller and is checked before writes.
    pub fn decode_into(self, encoded: &[u8], output: &mut [f32]) -> Result<()> {
        if self.output_len(encoded.len())? != output.len() {
            return Err(Error::tensor(
                "<blocks>",
                "decoded output length differs from encoded blocks",
            ));
        }
        if self.encoding.is_iq() {
            return crate::iquant::decode_into(self.encoding, encoded, self.endian, output);
        }
        if self.encoding == GgmlType::NvFp4
            && encoded
                .chunks_exact(self.bytes)
                .any(|block| block[..4].iter().any(|&scale| scale >= 0x7f))
        {
            return Err(Error::tensor("<blocks>", "invalid NVFP4 UE4M3 scale"));
        }
        for (block, output) in encoded
            .chunks_exact(self.bytes)
            .zip(output.chunks_exact_mut(self.values))
        {
            self.block(block, output);
        }
        Ok(())
    }
    /// Borrows packed storage as a flat sequence of physical rows. Each row
    /// contains whole blocks and the supplied extent must match the bytes exactly.
    pub fn packed_rows(
        self,
        encoded: &[u8],
        physical_rows: usize,
        columns: usize,
    ) -> Result<PackedRows<'_>> {
        if columns == 0 || !columns.is_multiple_of(self.values) {
            return Err(Error::tensor(
                "<blocks>",
                "packed row requires whole blocks",
            ));
        }
        let row_bytes = (columns / self.values)
            .checked_mul(self.bytes)
            .ok_or(Error::Overflow("packed row bytes"))?;
        let expected = physical_rows
            .checked_mul(row_bytes)
            .ok_or(Error::Overflow("packed storage bytes"))?;
        if encoded.len() != expected {
            return Err(Error::tensor(
                "<blocks>",
                "packed storage differs from row geometry",
            ));
        }
        Ok(PackedRows {
            decoder: self,
            encoded,
            physical_rows,
            row_bytes,
        })
    }

    fn half(self, bytes: &[u8]) -> f32 {
        half::f16::from_bits(self.endian.u16([bytes[0], bytes[1]])).to_f32()
    }
    fn block(self, b: &[u8], y: &mut [f32]) {
        match self.encoding {
            GgmlType::Q4_0 | GgmlType::Q4_1 | GgmlType::Q5_0 | GgmlType::Q5_1 => {
                let d = self.half(b);
                let asymmetric = matches!(self.encoding, GgmlType::Q4_1 | GgmlType::Q5_1);
                let q5 = matches!(self.encoding, GgmlType::Q5_0 | GgmlType::Q5_1);
                let high_offset = if asymmetric { 4 } else { 2 };
                let offset = high_offset + if q5 { 4 } else { 0 };
                let high = if q5 {
                    self.endian.u32(
                        b[high_offset..high_offset + 4]
                            .try_into()
                            .expect("block length"),
                    )
                } else {
                    0
                };
                let bias = if asymmetric { self.half(&b[2..]) } else { 0. };
                for j in 0..32 {
                    let code = if j < 16 {
                        b[offset + j] & 15
                    } else {
                        b[offset + j - 16] >> 4
                    };
                    let code = i32::from(code) | if q5 { ((high >> j) & 1) as i32 * 16 } else { 0 };
                    y[j] = if asymmetric {
                        d * code as f32 + bias
                    } else {
                        d * (code - if q5 { 16 } else { 8 }) as f32
                    };
                }
            }
            GgmlType::Q8_0 => {
                let d = self.half(b);
                for i in 0..32 {
                    y[i] = d * (b[2 + i] as i8) as f32;
                }
            }
            GgmlType::Q4K | GgmlType::Q5K => {
                let q5 = self.encoding == GgmlType::Q5K;
                let d = self.half(b);
                let dm = self.half(&b[2..]);
                let scales = unpack_k_scale_min(b[4..16].try_into().expect("K-block scale table"));
                let qs = &b[if q5 { 48 } else { 16 }..];
                for (g, (sc, min)) in scales.into_iter().enumerate() {
                    let scale = d * sc as f32;
                    let minimum = dm * min as f32;
                    for i in 0..32 {
                        let byte = qs[(g / 2) * 32 + i];
                        let code = (if g % 2 == 0 { byte & 15 } else { byte >> 4 })
                            | if q5 && b[16 + i] & (1 << g) != 0 {
                                16
                            } else {
                                0
                            };
                        y[g * 32 + i] = scale * code as f32 - minimum;
                    }
                }
            }
            GgmlType::Q6K => {
                let d = self.half(&b[208..]);
                for section in 0..2 {
                    for part in 0..4 {
                        for i in 0..32 {
                            let low = b[section * 64 + (part % 2) * 32 + i];
                            let code = (if part < 2 { low & 15 } else { low >> 4 })
                                | (((b[128 + section * 32 + i] >> (part * 2)) & 3) << 4);
                            let sc = b[192 + section * 8 + part * 2 + i / 16] as i8;
                            y[section * 128 + part * 32 + i] =
                                d * sc as f32 * (i32::from(code) - 32) as f32;
                        }
                    }
                }
            }
            GgmlType::Q2K | GgmlType::Q3K => {
                let q3 = self.encoding == GgmlType::Q3K;
                let d = self.half(&b[if q3 { 108 } else { 80 }..]);
                let dm = if q3 { 0. } else { self.half(&b[82..]) };
                for g in 0..16 {
                    let low = if q3 {
                        (b[96 + g % 8] >> (4 * (g / 8))) & 15
                    } else {
                        b[g] & 15
                    };
                    let scale = if q3 {
                        i32::from(low | (((b[104 + g % 4] >> (2 * (g / 4))) & 3) << 4)) - 32
                    } else {
                        i32::from(low)
                    };
                    let scaled = d * scale as f32;
                    let minimum = if q3 { 0. } else { dm * (b[g] >> 4) as f32 };
                    let offset = (g / 8) * 32 + (g % 2) * 16;
                    let shift = ((g % 8) / 2) * 2;
                    for i in 0..16 {
                        let code = ((b[if q3 { 32 } else { 16 } + offset + i] >> shift) & 3) as i32;
                        let code = code
                            - if q3 && b[(g % 2) * 16 + i] & (1 << (g / 2)) == 0 {
                                4
                            } else {
                                0
                            };
                        y[g * 16 + i] = scaled * code as f32 - minimum;
                    }
                }
            }
            GgmlType::MxFp4 => {
                // Pinned GGML uses doubled FP4 integers and a halved E8M0
                // scale, including exponents 0/1/255. Byte order is irrelevant.
                for i in 0..16 {
                    y[i] = crate::packed_words::mxfp4_value(b[i + 1] & 15, b[0]);
                    y[i + 16] = crate::packed_words::mxfp4_value(b[i + 1] >> 4, b[0]);
                }
            }
            GgmlType::NvFp4 => {
                // Each 16-value group has its own unsigned E4M3 scale.
                // FP4_CODES doubles the code values, so halve the scale.
                for (group, &scale) in b[..4].iter().enumerate() {
                    let d = nvfp4_half_scale(scale);
                    for (i, &codes) in b[4 + group * 8..][..8].iter().enumerate() {
                        y[group * 16 + i] = d * f32::from(FP4_CODES[usize::from(codes & 15)]);
                        y[group * 16 + i + 8] = d * f32::from(FP4_CODES[usize::from(codes >> 4)]);
                    }
                }
            }
            _ => unreachable!("checked decoder encoding"),
        }
    }
}

/// Checked borrowed storage for bounded packed row decoding.
pub struct PackedRows<'a> {
    decoder: BlockDecoder,
    encoded: &'a [u8],
    physical_rows: usize,
    row_bytes: usize,
}

impl PackedRows<'_> {
    /// Decodes one physical row into an exactly sized FP32 destination.
    pub fn decode_row(&self, row: usize, output: &mut [f32]) -> Result<()> {
        let end = row
            .checked_add(1)
            .ok_or(Error::Overflow("packed row index"))?;
        self.decode_rows(row..end, output)
    }

    /// Decodes a contiguous physical row range into an exactly sized destination.
    /// Bytes outside the selected range are left encoded.
    pub fn decode_rows(&self, rows: Range<usize>, output: &mut [f32]) -> Result<()> {
        if rows.start > rows.end || rows.end > self.physical_rows {
            return Err(Error::tensor(
                "<blocks>",
                "packed row range exceeds storage",
            ));
        }
        // The constructor checked the entire physical extent.
        let encoded = &self.encoded[rows.start * self.row_bytes..rows.end * self.row_bytes];
        self.decoder.decode_into(encoded, output)
    }
}

/// Extracts the eight six-bit scale/min pairs shared by Q4_K and Q5_K blocks.
pub fn unpack_k_scale_min(table: &[u8; 12]) -> [(u8, u8); 8] {
    std::array::from_fn(|group| {
        if group < 4 {
            (table[group] & 63, table[group + 4] & 63)
        } else {
            let packed = table[group + 4];
            (
                (packed & 15) | ((table[group - 4] >> 6) << 4),
                (packed >> 4) | ((table[group] >> 6) << 4),
            )
        }
    })
}

pub(crate) fn nvfp4_half_scale(scale: u8) -> f32 {
    let exponent = u32::from(scale >> 3);
    let mantissa = f32::from(scale & 7);
    if exponent == 0 {
        mantissa / 1024.0
    } else {
        let half_scale_exponent = exponent + 127 - 7 - 1;
        (1.0 + mantissa / 8.0) * f32::from_bits(half_scale_exponent << 23)
    }
}
