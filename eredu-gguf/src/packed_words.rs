//! Canonical packed-word scalar decoding shared by GGUF and checkpoint consumers.

/// Extracts one field from adjacent little-endian words, including fields that
/// span a word boundary. The caller validates the packed extent first.
pub fn unpack_code(words: &[u32], index: usize, bits: usize) -> u32 {
    let bit = index * bits;
    let word = bit / 32;
    let shift = bit % 32;
    let mut code = words[word] >> shift;
    if shift + bits > 32 {
        code |= words[word + 1] << (32 - shift);
    }
    code & ((1 << bits) - 1)
}

/// E2M1 values as integers after doubling their magnitude.
pub(crate) const FP4_CODES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Decodes an E2M1 value with the pinned GGML E8M0 scaling convention.
pub fn mxfp4_value(code: u8, exponent: u8) -> f32 {
    let exponent = u32::from(exponent);
    let half_scale = f32::from_bits(if exponent < 2 {
        0x0020_0000 << exponent
    } else {
        (exponent - 1) << 23
    });
    f32::from(FP4_CODES[usize::from(code & 15)]) * half_scale
}
