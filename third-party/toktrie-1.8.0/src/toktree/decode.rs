//! Shared token expansion and exact raw-capture decode destinations.
use super::{parse_numeric_token, TokTrie, TokenId};
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

// The ordinary '<[u32]>' spelling, including all ten decimal digits.
fn absent_token(token: TokenId, buffer: &mut [u8; 14]) -> &[u8] {
    let mut token = token;
    let mut cursor = 12;
    buffer[12] = b']';
    buffer[13] = b'>';
    loop {
        cursor -= 1;
        buffer[cursor] = b'0' + (token % 10) as u8;
        token /= 10;
        if token == 0 {
            break;
        }
    }
    cursor -= 2;
    buffer[cursor] = b'<';
    buffer[cursor + 1] = b'[';
    &buffer[cursor..]
}

impl TokTrie {
    /// Compares ordinary decoded token bytes without allocating an output Vec.
    /// Special spellings and absent-token numeric spellings use the same decoder.
    pub fn decoded_tokens_match(&self, tokens: &[TokenId], mut bytes: &[u8]) -> bool {
        for &token in tokens {
            if self.decode_token_with(token, true, &mut |part| {
                bytes = bytes.strip_prefix(part).ok_or(())?;
                Ok::<(), ()>(())
            }).is_err() { return false; }
        }
        bytes.is_empty()
    }
    /// Fixed borrowed comparison and existing per-token decoder frames.
    pub fn decoded_tokens_match_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<(&Self, &[TokenId], &[u8])>(),
            size_of::<std::slice::Iter<'_, TokenId>>(),
            size_of::<(&mut &[u8],)>(),
            size_of::<(&Self, TokenId, bool)>(),
            size_of::<Option<&[u8]>>(),
            size_of::<Result<(), ()>>(),
            size_of::<[u8; 14]>(),
            size_of::<(&mut [u8; 14], TokenId, usize)>(),
            size_of::<bool>(),
        ];
        parts.iter().copied().try_fold(size_of_val(&parts), usize::checked_add)
    }
    pub(super) fn decode_token_with<E>(
        &self,
        token: TokenId,
        include_special: bool,
        emit: &mut impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        let bytes = self.token(token);
        if bytes.is_empty() {
            if include_special {
                emit(absent_token(token, &mut [0; 14]))?;
            }
        } else if bytes[0] == TokTrie::SPECIAL_TOKEN_MARKER {
            if include_special {
                emit(&bytes[1..])?;
            }
        } else {
            emit(bytes)?;
        }
        Ok(())
    }
    pub(super) fn decode_raw_with<E>(
        &self,
        bytes: &[u8],
        emit: &mut impl FnMut(&[u8]) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == TokTrie::SPECIAL_TOKEN_MARKER {
                if let Some((length, token)) = parse_numeric_token(&bytes[index + 1..]) {
                    self.decode_token_with(token, true, emit)?;
                    index += length + 1;
                    continue;
                }
            }
            emit(&bytes[index..index + 1])?;
            index += 1;
        }
        Ok(())
    }
    /// Borrows this exact trie and raw parser bytes, computes their decoded
    /// extent with the same worker, and supplies one independent destination.
    pub fn raw_decode_plan<'a>(
        &'a self,
        bytes: &'a [u8],
    ) -> Result<RawTokenDecodePlan<'a>, RawTokenDecodeError> {
        let mut output_bytes = 0usize;
        self.decode_raw_with(bytes, &mut |part| {
            output_bytes = output_bytes
                .checked_add(part.len())
                .ok_or(RawTokenDecodeError::Overflow)?;
            Ok(())
        })?;
        let layout =
            Layout::array::<u8>(output_bytes).map_err(|_| RawTokenDecodeError::Overflow)?;
        let controls =
            RawTokenDecodePlan::inspection_control_bytes().ok_or(RawTokenDecodeError::Overflow)?;
        let required = controls
            .checked_add(layout.size())
            .ok_or(RawTokenDecodeError::Overflow)?;
        Ok(RawTokenDecodePlan {
            trie: self,
            bytes,
            output_bytes,
            required,
        })
    }
}
/// A typed source or actual output allocation refusal.
#[derive(Debug)]
pub enum RawTokenDecodeError {
    /// The reached output extent or fixed frame sum cannot be represented.
    Overflow,
    /// The actual destination differs from its inspected source population.
    Capacity,
    /// Allocating the exact reached output failed.
    Allocation(TryReserveError),
}
impl fmt::Display for RawTokenDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow => f.write_str("raw token decode extent overflow"),
            Self::Capacity => f.write_str("raw token decode destination differs from source"),
            Self::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for RawTokenDecodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Allocation(e) => Some(e),
            _ => None,
        }
    }
}
/// A failed decoder retains the actual output allocation. Its enclosing caller
/// retains the funding owner until this prefix is dropped.
#[derive(Debug)]
pub struct RawTokenDecodeFailure {
    cause: RawTokenDecodeError,
    output: Vec<u8>,
}
impl fmt::Display for RawTokenDecodeFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for RawTokenDecodeFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Exact immutable source loan and complete independent output quote.
pub struct RawTokenDecodePlan<'a> {
    trie: &'a TokTrie,
    bytes: &'a [u8],
    output_bytes: usize,
    required: usize,
}
impl fmt::Debug for RawTokenDecodePlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RawTokenDecodePlan")
            .field("input_bytes", &self.bytes.len())
            .field("output_bytes", &self.output_bytes)
            .field("required_bytes", &self.required)
            .finish()
    }
}
impl RawTokenDecodePlan<'_> {
    /// Fixed source/worker/count/write/failure frames, to price before inspection.
    pub fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<RawTokenDecodeFailure>(),
            size_of::<RawTokenDecodeError>(),
            size_of::<[u8; 14]>(),
            size_of::<(usize, usize, TokenId, bool)>(),
            size_of::<(&TokTrie, &[u8], &mut usize)>(),
            size_of::<(&mut Vec<u8>, usize)>(),
            size_of::<Layout>(),
            size_of::<Result<Self, RawTokenDecodeError>>(),
            size_of::<Result<Vec<u8>, RawTokenDecodeFailure>>(),
            size_of::<Result<(), RawTokenDecodeError>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<Option<(usize, TokenId)>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Exact decoded payload bytes, including ordinary absent-token spelling.
    pub fn output_bytes(&self) -> usize {
        self.output_bytes
    }
    /// Complete output allocation and fixed local worker frames.
    pub fn required_bytes(&self) -> usize {
        self.required
    }
    /// Runs the same decoder after the caller reserves this plan's complete
    /// requirement. No nested per-token buffer or formatting allocation occurs.
    pub fn compile(self) -> Result<Vec<u8>, RawTokenDecodeFailure> {
        let mut output = Vec::new();
        let result = (|| {
            output
                .try_reserve_exact(self.output_bytes)
                .map_err(RawTokenDecodeError::Allocation)?;
            if output.capacity() != self.output_bytes {
                return Err(RawTokenDecodeError::Capacity);
            }
            self.trie.decode_raw_with(self.bytes, &mut |part| {
                let end = output
                    .len()
                    .checked_add(part.len())
                    .ok_or(RawTokenDecodeError::Overflow)?;
                if end > self.output_bytes {
                    return Err(RawTokenDecodeError::Capacity);
                }
                output.extend_from_slice(part);
                Ok(())
            })?;
            if output.len() != self.output_bytes {
                return Err(RawTokenDecodeError::Capacity);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(output),
            Err(cause) => Err(RawTokenDecodeFailure { cause, output }),
        }
    }
}
