//! GPT-2's fixed byte alphabet, with no runtime allocation or initialization.

pub(crate) static BYTE_TO_CHAR: [char; 256] = make_alphabet();

const fn make_alphabet() -> [char; 256] {
    let mut alphabet = ['\0'; 256];
    let mut byte = 0u32;
    while byte < 256 {
        let scalar = match byte {
            0..=32 => byte + 256,
            127..=160 => byte + 162,
            173 => 323,
            _ => byte,
        };
        alphabet[byte as usize] = match char::from_u32(scalar) {
            Some(value) => value,
            None => panic!("byte alphabet contains an invalid scalar"),
        };
        byte += 1;
    }
    alphabet
}

/// Only alphabet characters decode to bytes. Unknown characters cause the
/// existing decoder to fall back to the original UTF-8 of the entire token.
pub(crate) const fn char_to_byte(c: char) -> Option<u8> {
    match c as u32 {
        scalar @ (33..=126 | 161..=172 | 174..=255) => Some(scalar as u8),
        scalar @ 256..=288 => Some((scalar - 256) as u8),
        scalar @ 289..=322 => Some((scalar - 162) as u8),
        323 => Some(173),
        _ => None,
    }
}
