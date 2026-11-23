//! Deterministic scalar encoding of bounded affine and MXFP4 checkpoint tiles.

use crate::{recipe::RecipeDtype, Error, PackedWordGeometry, StoredDtype, WeightQuantization};

/// One little-endian packed output from a prepared quantization tile.
pub struct WordTileOutput {
    /// Storage scalar represented by the byte buffer.
    pub dtype: StoredDtype,
    /// Exact row-major packed data.
    pub bytes: Vec<u8>,
}

/// Returns the common complete-row geometry for load-time encoding.
pub(crate) fn geometry(
    encoding: WeightQuantization,
    columns: usize,
) -> Result<PackedWordGeometry, Error> {
    if !matches!(
        encoding,
        WeightQuantization::Affine(_) | WeightQuantization::MxFp4
    ) || columns == 0
        || !columns.is_multiple_of(32)
    {
        return Err(Error::invalid(
            "unsupported format or unaligned checkpoint row",
        ));
    }
    encoding.packed_word_geometry(columns)
}

/// Encodes complete rows. The scalar accessor reads the original prepared tile
/// in row-major order so no widened dense tile or per-group array is required.
pub fn encode(
    rows: usize,
    columns: usize,
    encoding: WeightQuantization,
    companion_dtype: RecipeDtype,
    value: impl Fn(usize) -> f32,
) -> Result<Vec<WordTileOutput>, Error> {
    if rows == 0 {
        return Err(Error::invalid(
            "unsupported word quantization format or empty tile",
        ));
    }
    let geometry = geometry(encoding, columns)?;
    let bits = encoding.bits() as usize;
    let group = encoding.group_size() as usize;
    let microscaling = encoding == WeightQuantization::MxFp4;
    let float_dtype = match companion_dtype {
        RecipeDtype::F16 => StoredDtype::F16,
        RecipeDtype::BF16 => StoredDtype::BF16,
        RecipeDtype::F32 => StoredDtype::F32,
        _ => {
            return Err(Error::invalid(
                "affine companions require F16, BF16, or F32",
            ))
        }
    };
    let capacity = |width: usize, bytes: usize| {
        rows.checked_mul(width)
            .and_then(|n| n.checked_mul(bytes))
            .ok_or_else(|| Error::invalid("word quantization tile overflows"))
    };
    capacity(columns, 1)?;
    let mut words = Vec::with_capacity(capacity(geometry.word_columns(), 4)?);
    let scalar_width = if float_dtype == StoredDtype::F32 {
        4
    } else {
        2
    };
    let companion_width = if microscaling { 1 } else { scalar_width };
    let mut scales = Vec::with_capacity(capacity(geometry.group_columns(), companion_width)?);
    let mut offsets = if microscaling {
        Vec::new()
    } else {
        Vec::with_capacity(scales.capacity())
    };
    for row in 0..rows {
        let mut word = 0u64;
        let mut occupied = 0;
        for range_start in (row * columns..(row + 1) * columns).step_by(group) {
            let range = range_start..range_start + group;
            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for index in range.clone() {
                let scalar = value(index);
                if !scalar.is_finite() {
                    return Err(Error::invalid("weight quantization requires finite input"));
                }
                min = min.min(scalar);
                max = max.max(scalar);
            }
            let levels = (1u32 << bits) - 1;
            let (exponent, affine_scale) = if microscaling {
                let magnitude = min.abs().max(max.abs());
                let exponent = if magnitude == 0. {
                    0
                } else {
                    // The shared E8M0 exponent places the maximum exponent
                    // in the E2M1 range. Codes saturate at ±6.
                    (magnitude.log2().floor() as i32 - 2 + 127).clamp(0, 254) as u8
                };
                scales.push(exponent);
                (exponent, 0.)
            } else {
                let scale = ((f64::from(max) - f64::from(min)) / f64::from(levels)) as f32;
                append_float(&mut scales, scale, &float_dtype)?;
                append_float(&mut offsets, min, &float_dtype)?;
                (0, scale)
            };
            let mx_scale = f64::from(eredu_gguf::packed_words::mxfp4_value(2, exponent));
            for index in range {
                let scalar = value(index);
                let code = if microscaling {
                    u32::from(nearest_fp4(scalar, mx_scale))
                } else if affine_scale == 0. {
                    0
                } else {
                    ((f64::from(scalar) - f64::from(min)) / f64::from(affine_scale))
                        .round_ties_even()
                        .clamp(0., f64::from(levels)) as u32
                };
                word |= u64::from(code) << occupied;
                occupied += bits;
                if occupied >= 32 {
                    words.extend_from_slice(&(word as u32).to_le_bytes());
                    word >>= 32;
                    occupied -= 32;
                }
            }
        }
        debug_assert_eq!(occupied, 0);
    }
    let mut outputs = vec![
        WordTileOutput {
            dtype: StoredDtype::U32,
            bytes: words,
        },
        WordTileOutput {
            dtype: if microscaling {
                StoredDtype::U8
            } else {
                float_dtype.clone()
            },
            bytes: scales,
        },
    ];
    if !microscaling {
        outputs.push(WordTileOutput {
            dtype: float_dtype,
            bytes: offsets,
        });
    }
    Ok(outputs)
}

fn nearest_fp4(value: f32, scale: f64) -> u8 {
    const MIDPOINTS: [f64; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.];
    let magnitude = f64::from(value.abs()) / scale;
    let index = MIDPOINTS.partition_point(|&point| point < magnitude);
    let code = if index % 2 == 1 && MIDPOINTS.get(index) == Some(&magnitude) {
        index + 1
    } else {
        index
    };
    code as u8 | (u8::from(value.is_sign_negative()) << 3)
}

fn append_float(bytes: &mut Vec<u8>, value: f32, dtype: &StoredDtype) -> Result<(), Error> {
    match dtype {
        StoredDtype::F16 => {
            let value = half::f16::from_f32(value);
            if !value.is_finite() {
                return Err(Error::invalid(
                    "affine companion exceeds its storage precision",
                ));
            }
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        StoredDtype::BF16 => {
            let value = half::bf16::from_f32(value);
            if !value.is_finite() {
                return Err(Error::invalid(
                    "affine companion exceeds its storage precision",
                ));
            }
            bytes.extend_from_slice(&value.to_bits().to_le_bytes());
        }
        StoredDtype::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
        _ => unreachable!("validated affine companion"),
    }
    Ok(())
}
