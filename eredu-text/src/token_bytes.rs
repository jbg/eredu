//! Lexical bytes for token constraints, shared with ordinary TokTrie input.
//! This is deliberately distinct from streaming decode: it preserves arbitrary
//! bytes, performs no lossy UTF-8 repair and never applies generation prefixes.
mod packed;
pub use packed::{PackedTokenBytePlan, PackedTokenByteView};
#[cfg(test)]
mod tests;
use std::mem::{size_of, size_of_val};
use tokenizers::{decoders::DecoderWrapper, normalizers::replace::ReplacePattern};

/// Fixed errors from source inspection or an exact destination.
#[derive(Debug, thiserror::Error)]
pub enum TokenByteError {
    /// No ByteLevel or ByteFallback component was present.
    #[error("token decoder has no lexical byte encoding")]
    Encoding,
    /// Malformed byte-fallback spelling; the original integer cause is retained.
    #[error("invalid byte-fallback token")]
    Fallback(#[source] Option<std::num::ParseIntError>),
    /// Sparse ID extent or packed byte arithmetic overflowed.
    #[error("token-byte source extent overflow")]
    Overflow,
    /// The exact output slice does not match the planned population.
    #[error("token-byte destination geometry changed")]
    Destination,
    /// The source has no canonical, bidirectionally consistent token ID.
    #[error("token-byte vocabulary has no canonical ID")]
    Empty,
}
#[derive(Debug, Clone, Copy)]
enum Kind {
    ByteLevel,
    Fallback(char),
}
#[derive(Default)]
struct Parts {
    byte_level: bool,
    fallback: bool,
    marker: Option<char>,
    depth: usize,
}
/// Immutable encoding facts from the actual decoder. ByteFallback has the same
/// precedence over ByteLevel as ordinary semantic vocabulary construction.
#[derive(Debug, Clone, Copy)]
pub struct TokenByteEncoding {
    kind: Kind,
    depth: usize,
}
impl TokenByteEncoding {
    /// Borrows the decoder without serde, regex construction, tables or cloning.
    /// Ordered sequence traversal preserves the last single-scalar space marker.
    pub fn inspect(decoder: Option<&DecoderWrapper>) -> Result<Self, TokenByteError> {
        fn visit(
            decoder: &DecoderWrapper,
            parts: &mut Parts,
            depth: usize,
        ) -> Result<(), TokenByteError> {
            parts.depth = parts.depth.max(depth);
            match decoder {
                DecoderWrapper::ByteLevel(_) => parts.byte_level = true,
                DecoderWrapper::ByteFallback(_) => parts.fallback = true,
                DecoderWrapper::Replace(replace) if replace.content == " " => {
                    if let ReplacePattern::String(pattern) = replace.pattern() {
                        let mut chars = pattern.chars();
                        if let (Some(marker), None) = (chars.next(), chars.next()) {
                            parts.marker = Some(marker);
                        }
                    }
                }
                DecoderWrapper::Sequence(sequence) => {
                    for member in sequence.get_decoders() {
                        visit(
                            member,
                            parts,
                            depth.checked_add(1).ok_or(TokenByteError::Overflow)?,
                        )?;
                    }
                }
                _ => {}
            }
            Ok(())
        }
        let mut parts = Parts::default();
        if let Some(decoder) = decoder {
            visit(decoder, &mut parts, 1)?;
        }
        let kind = if parts.fallback {
            Kind::Fallback(parts.marker.unwrap_or(' '))
        } else if parts.byte_level {
            Kind::ByteLevel
        } else {
            return Err(TokenByteError::Encoding);
        };
        Ok(Self {
            kind,
            depth: parts.depth,
        })
    }
    // Same byte spelling used by ordinary token-trie construction. The callback
    // cannot allocate internally; concrete count/fill destinations are below.
    fn visit(self, token: &str, mut emit: impl FnMut(u8)) -> Result<(), TokenByteError> {
        match self.kind {
            Kind::ByteLevel => {
                // ByteLevel decodes a whole token as raw UTF-8 when any scalar
                // lies outside its alphabet. Added tokens commonly contain raw
                // whitespace or Unicode; a per-scalar fallback would change
                // mixed spellings such as a mapped space followed by an emoji.
                if token
                    .chars()
                    .all(|c| crate::decoder_storage::byte_for_char(c).is_some())
                {
                    for character in token.chars() {
                        emit(
                            crate::decoder_storage::byte_for_char(character)
                                .expect("checked complete byte-level alphabet"),
                        );
                    }
                } else {
                    for &byte in token.as_bytes() {
                        emit(byte);
                    }
                }
            }
            Kind::Fallback(marker) => {
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    emit(
                        u8::from_str_radix(&token[3..5], 16)
                            .map_err(|cause| TokenByteError::Fallback(Some(cause)))?,
                    );
                } else if token.starts_with("<0x") {
                    return Err(TokenByteError::Fallback(None));
                } else {
                    for character in token.chars() {
                        if character == marker {
                            emit(b' ');
                        } else {
                            let mut scalar = [0; 4];
                            for &byte in character.encode_utf8(&mut scalar).as_bytes() {
                                emit(byte);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
    // One token-to-trie-byte worker for ordinary environment and original source
    // construction. Special membership is an actual source fact; its marker is
    // distinct from lexical bytes such as a literal fallback 0xFF token.
    fn visit_trie(
        self,
        token: &str,
        special: bool,
        mut emit: impl FnMut(u8),
    ) -> Result<(), TokenByteError> {
        if special {
            emit(0xff);
            for &byte in token.as_bytes() {
                emit(byte);
            }
            Ok(())
        } else {
            self.visit(token, emit)
        }
    }
    /// Exact trie-input bytes, including the marker only for a special token.
    pub fn trie_token_len(self, token: &str, special: bool) -> Result<usize, TokenByteError> {
        let mut bytes = 0usize;
        let mut overflow = false;
        self.visit_trie(token, special, |_| match bytes.checked_add(1) {
            Some(next) => bytes = next,
            None => overflow = true,
        })?;
        if overflow {
            Err(TokenByteError::Overflow)
        } else {
            Ok(bytes)
        }
    }
    /// Writes the same exact ordinary trie token bytes without temporary storage.
    /// A malformed spelling or wrong destination length leaves it unchanged.
    pub fn write_trie_token(
        self,
        token: &str,
        special: bool,
        destination: &mut [u8],
    ) -> Result<(), TokenByteError> {
        if self.trie_token_len(token, special)? != destination.len() {
            return Err(TokenByteError::Destination);
        }
        let mut position = 0;
        self.visit_trie(token, special, |byte| {
            destination[position] = byte;
            position += 1;
        })
    }
    /// Exact bytes, validating every scalar before a destination is touched.
    pub fn encoded_len(self, token: &str) -> Result<usize, TokenByteError> {
        let mut bytes = 0usize;
        let mut overflow = false;
        self.visit(token, |_| match bytes.checked_add(1) {
            Some(next) => bytes = next,
            None => overflow = true,
        })?;
        if overflow {
            Err(TokenByteError::Overflow)
        } else {
            Ok(bytes)
        }
    }
    /// Writes exactly the planned slice; malformed inputs and wrong lengths leave
    /// the destination unchanged. No temporary token Vec or map is constructed.
    pub fn write(self, token: &str, destination: &mut [u8]) -> Result<(), TokenByteError> {
        if self.encoded_len(token)? != destination.len() {
            return Err(TokenByteError::Destination);
        }
        let mut index = 0;
        self.visit(token, |byte| {
            destination[index] = byte;
            index += 1;
        })
    }
    /// Named source/byte traversal frames, including actual nested decoder depth.
    /// This describes host work and grants no source or allocation authority.
    pub fn control_bytes(self) -> Option<usize> {
        let recursive = size_of::<(&DecoderWrapper, &mut Parts, usize)>()
            .checked_add(size_of::<std::slice::Iter<'_, DecoderWrapper>>())?
            .checked_add(size_of::<std::str::Chars<'_>>())?
            .checked_add(size_of::<Result<(), TokenByteError>>())?;
        let parts = [
            recursive.checked_mul(self.depth)?,
            size_of::<Parts>(),
            size_of::<Self>(),
            size_of::<Result<Self, TokenByteError>>(),
            size_of::<TokenByteError>(),
            size_of::<(Self, &str, &mut [u8], usize)>(),
            size_of::<(Self, &str, bool, &mut [u8], usize)>(),
            size_of::<std::str::Chars<'_>>(),
            size_of::<std::slice::Iter<'_, u8>>(),
            size_of::<[u8; 4]>(),
            size_of::<Result<usize, TokenByteError>>(),
            size_of::<Result<(), TokenByteError>>(),
            size_of::<(usize, bool)>(),
            size_of::<(&mut usize, &mut bool)>(),
            size_of::<(&mut [u8], &mut usize)>(),
            size_of::<Result<u8, std::num::ParseIntError>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
