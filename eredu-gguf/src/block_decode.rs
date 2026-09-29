//! Allocation-free scalar decoding for block layouts without native row kernels.
//! Equations: ggml-quants.c at 2145525a4081d66ff1a87cf43ef809f95a85ac0c.
use crate::{Endian, Error, GgmlType, Result};

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
    /// Supported generic scalar mechanisms. Other encodings retain their existing
    /// native/codebook decoders; this is not a model-family support predicate.
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
        ) {
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
        if encoded_bytes % self.bytes != 0 {
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
        for (block, output) in encoded
            .chunks_exact(self.bytes)
            .zip(output.chunks_exact_mut(self.values))
        {
            self.block(block, output);
        }
        Ok(())
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
                let s = &b[4..16];
                let qs = &b[if q5 { 48 } else { 16 }..];
                for g in 0..8 {
                    let (sc, min) = if g < 4 {
                        (s[g] & 63, s[g + 4] & 63)
                    } else {
                        (
                            (s[g + 4] & 15) | ((s[g - 4] >> 6) << 4),
                            (s[g + 4] >> 4) | ((s[g] >> 6) << 4),
                        )
                    };
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
                const CODES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
                let exponent = u32::from(b[0]);
                let d = f32::from_bits(if exponent < 2 {
                    0x0020_0000 << exponent
                } else {
                    (exponent - 1) << 23
                });
                for i in 0..16 {
                    y[i] = f32::from(CODES[(b[i + 1] & 15) as usize]) * d;
                    y[i + 16] = f32::from(CODES[(b[i + 1] >> 4) as usize]) * d;
                }
            }
            _ => unreachable!("checked decoder encoding"),
        }
    }
}
