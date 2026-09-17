//! Exact borrowed canonical vocabulary for trie inputs or marker-stripped constraints.
use super::*;
use crate::tokenizer_storage::PreparedTokenizer;

/// One immutable tokenizer loan and finite packed output geometry. This does not
/// retain a trie, reconstruct a tokenizer, or certify a runtime source account.
pub struct PackedTokenBytePlan<'a> {
    source: &'a PreparedTokenizer,
    encoding: TokenByteEncoding,
    tokens: usize,
    bytes: usize,
    maximum: usize,
    preserve_markers: bool,
}
impl<'a> PackedTokenBytePlan<'a> {
    pub(crate) fn prepare(
        source: &'a PreparedTokenizer,
        encoding: TokenByteEncoding,
    ) -> Result<Self, TokenByteError> {
        Self::prepare_with_markers(source, encoding, false)
    }
    pub(crate) fn prepare_trie(
        source: &'a PreparedTokenizer,
        encoding: TokenByteEncoding,
    ) -> Result<Self, TokenByteError> {
        Self::prepare_with_markers(source, encoding, true)
    }
    fn prepare_with_markers(
        source: &'a PreparedTokenizer,
        encoding: TokenByteEncoding,
        preserve_markers: bool,
    ) -> Result<Self, TokenByteError> {
        let mut last = None;
        for id in source.ids() {
            if source.spelling(id).and_then(|s| source.token_id(s)) == Some(id) {
                last = Some(last.map_or(id, |previous: u32| previous.max(id)));
            }
        }
        // Same TokRxInfo u32 extent used by ordinary vocabulary construction.
        let tokens = usize::try_from(
            last.ok_or(TokenByteError::Empty)?
                .checked_add(1)
                .ok_or(TokenByteError::Overflow)?,
        )
        .map_err(|_| TokenByteError::Overflow)?;
        let bytes = tokens
            .checked_add(1)
            .and_then(|n| n.checked_mul(size_of::<u64>()))
            .ok_or(TokenByteError::Overflow)?;
        let mut plan = Self {
            source,
            encoding,
            tokens,
            bytes,
            maximum: 0,
            preserve_markers,
        };
        for token in 0..tokens {
            let length = plan.token_len(token)?;
            plan.bytes = plan
                .bytes
                .checked_add(length)
                .ok_or(TokenByteError::Overflow)?;
            plan.maximum = plan.maximum.max(length);
        }
        if plan.bytes > isize::MAX as usize || u64::try_from(plan.bytes).is_err() {
            return Err(TokenByteError::Overflow);
        }
        Ok(plan)
    }
    fn token(&self, token: usize) -> Option<(&str, bool)> {
        let id = u32::try_from(token).ok()?;
        let spelling = self.source.spelling(id)?;
        if self.source.token_id(spelling) != Some(id) {
            return None;
        }
        Some((spelling, self.source.is_special(spelling)))
    }
    fn visit(&self, token: usize, mut emit: impl FnMut(u8)) -> Result<(), TokenByteError> {
        let Some((spelling, special)) = self.token(token) else {
            return Ok(());
        };
        // The constraint worker retains its exact existing strip_prefix rule,
        // including a lexical fallback 0xFF. Trie inputs preserve all bytes.
        let mut first = true;
        self.encoding.visit_trie(spelling, special, |byte| {
            if self.preserve_markers || !first || byte != 0xff {
                emit(byte);
            }
            first = false;
        })
    }

    fn token_len(&self, token: usize) -> Result<usize, TokenByteError> {
        let mut length = 0usize;
        let mut overflow = false;
        self.visit(token, |_| match length.checked_add(1) {
            Some(next) => length = next,
            None => overflow = true,
        })?;
        if overflow {
            Err(TokenByteError::Overflow)
        } else {
            Ok(length)
        }
    }
    /// Dense domain including actual sparse holes, matching ordinary TokTrie.
    pub fn token_count(&self) -> usize {
        self.tokens
    }
    /// Offset table plus exact lexical token bytes.
    pub fn packed_bytes(&self) -> usize {
        self.bytes
    }
    /// Maximum token length after this plan's selected marker rule.
    pub fn maximum_token_bytes(&self) -> usize {
        self.maximum
    }
    /// Writes the same little-endian absolute offsets and token byte order into
    /// one exact caller-owned destination. There is no hidden scratch allocation.
    pub fn write(&self, destination: &mut [u8]) -> Result<(), TokenByteError> {
        if destination.len() != self.bytes {
            return Err(TokenByteError::Destination);
        }
        let mut offset = (self.tokens + 1) * size_of::<u64>();
        destination[..8].copy_from_slice(&(offset as u64).to_le_bytes());
        for token in 0..self.tokens {
            self.visit(token, |byte| {
                destination[offset] = byte;
                offset += 1;
            })?;
            let index = (token + 1) * 8;
            destination[index..index + 8].copy_from_slice(&(offset as u64).to_le_bytes());
        }
        debug_assert_eq!(offset, self.bytes);
        Ok(())
    }
    /// Writes and lends closed per-token slices from this source. No caller
    /// offsets are adopted, and neither the source nor destination can change
    /// while the returned lexical view is borrowed.
    pub fn write_tokens<'b>(
        &'b self,
        destination: &'b mut [u8],
    ) -> Result<PackedTokenByteView<'b>, TokenByteError> {
        self.write(destination)?;
        Ok(PackedTokenByteView {
            bytes: destination,
            tokens: self.tokens,
            maximum: self.maximum,
            _source: self.source,
        })
    }
    /// Exact named plan/iteration/byte controls; payload is `packed_bytes`.
    pub fn control_bytes(&self) -> Option<usize> {
        let parts = [
            self.encoding.control_bytes()?,
            size_of::<Self>(),
            size_of::<PackedTokenByteView<'_>>(),
            size_of::<Result<PackedTokenByteView<'_>, TokenByteError>>(),
            size_of::<Result<Self, TokenByteError>>(),
            size_of::<(&Self, &mut [u8])>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<Option<(&str, bool)>>(),
            size_of::<Option<u32>>(),
            size_of::<[u8; 8]>(),
            size_of::<(usize, usize, bool)>(),
            size_of::<(&mut usize, &mut bool)>(),
            size_of::<(&mut [u8], &mut usize)>(),
            size_of::<(&mut bool, &mut usize)>(),
            size_of_val(&self.source.ids()),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}

/// Read-only token slices just written by an exact original tokenizer plan.
/// This is a lexical borrow, not allocation or grammar-execution authority.
pub struct PackedTokenByteView<'a> {
    bytes: &'a [u8],
    tokens: usize,
    maximum: usize,
    _source: &'a PreparedTokenizer,
}
impl PackedTokenByteView<'_> {
    /// Dense vocabulary size, preserving empty holes.
    pub fn token_count(&self) -> usize {
        self.tokens
    }
    /// Maximum byte length from the retained source and selected marker policy.
    pub fn maximum_token_bytes(&self) -> usize {
        self.maximum
    }
    /// Borrows one actual token byte slice, including empty sparse slots.
    pub fn token(&self, id: usize) -> Option<&[u8]> {
        if id >= self.tokens {
            return None;
        }
        let position = id * 8;
        let start = u64::from_le_bytes(
            self.bytes[position..position + 8]
                .try_into()
                .expect("written offset"),
        ) as usize;
        let end = u64::from_le_bytes(
            self.bytes[position + 8..position + 16]
                .try_into()
                .expect("written offset"),
        ) as usize;
        Some(&self.bytes[start..end])
    }
}
