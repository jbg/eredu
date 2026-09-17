//! Fixed raw-token spelling and allocation-free token-byte traversal.
use super::{Recognizer, TokTrie, TokenId, TrieNode};
use std::mem::{size_of, size_of_val};
/// The trie's exact numeric-token spelling: marker, '[', u32 decimal digits, ']'.
#[derive(Clone)]
pub struct MarkedTokenBytes {
    bytes: [u8; 13],
    start: usize,
}
impl MarkedTokenBytes {
    /// Formats one actual token ID without allocating.
    pub fn new(mut token: TokenId) -> Self {
        let mut bytes = [0; 13];
        bytes[12] = b']';
        let mut start = 12;
        loop {
            start -= 1;
            bytes[start] = b'0' + (token % 10) as u8;
            token /= 10;
            if token == 0 {
                break;
            }
        }
        start -= 1;
        bytes[start] = b'[';
        start -= 1;
        bytes[start] = TokTrie::SPECIAL_TOKEN_MARKER;
        Self { bytes, start }
    }
    /// Borrowed exact spelling, including the internal marker byte.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[self.start..]
    }
}
#[derive(Clone)]
enum Source<'a> {
    Text(&'a [u8]),
    Marked(MarkedTokenBytes),
}
impl Source<'_> {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Text(b) => b,
            Self::Marked(b) => b.bytes(),
        }
    }
}
/// Borrowed token IDs and at most one fixed numeric spelling, in raw decode order.
#[derive(Clone)]
pub struct RawTokenBytes<'a> {
    trie: &'a TokTrie,
    tokens: &'a [TokenId],
    current: Source<'a>,
    offset: usize,
}
impl Iterator for RawTokenBytes<'_> {
    type Item = u8;
    fn next(&mut self) -> Option<u8> {
        loop {
            if let Some(&byte) = self.current.bytes().get(self.offset) {
                self.offset += 1;
                return Some(byte);
            }
            let (&token, rest) = self.tokens.split_first()?;
            self.tokens = rest;
            self.current = self.trie.raw_token_source(token);
            self.offset = 0;
        }
    }
}
impl TokTrie {
    fn raw_token_source(&self, token: TokenId) -> Source<'_> {
        let bytes = self.token(token);
        if bytes.is_empty() || bytes[0] == Self::SPECIAL_TOKEN_MARKER {
            Source::Marked(MarkedTokenBytes::new(token))
        } else {
            Source::Text(bytes)
        }
    }
    /// Same raw decoding as `decode_raw`, borrowing source storage and using
    /// fixed spelling storage instead of allocating a temporary byte vector.
    pub fn raw_token_bytes<'a>(&'a self, tokens: &'a [TokenId]) -> RawTokenBytes<'a> {
        RawTokenBytes {
            trie: self,
            tokens,
            current: Source::Text(&[]),
            offset: 0,
        }
    }
    /// Exact raw extent from token metadata, without visiting every output byte.
    pub fn raw_token_bytes_len(&self, tokens: &[TokenId]) -> Option<usize> {
        tokens.iter().try_fold(0usize, |total, &id| {
            total.checked_add(self.raw_token_source(id).bytes().len())
        })
    }
    /// Fixed local frames for the exact suffix and extension worker. The
    /// caller's recognizer state and any reached parser growth remain separate.
    pub fn chop_tokens_control_bytes<R: Recognizer>() -> Option<usize> {
        let parts = [
            size_of::<RawTokenBytes<'_>>() * 2,
            size_of::<std::iter::Skip<RawTokenBytes<'_>>>(),
            size_of::<Source<'_>>(),
            size_of::<MarkedTokenBytes>(),
            size_of::<Option<&TrieNode>>(),
            size_of::<std::ops::Range<usize>>(),
            size_of::<std::ops::RangeInclusive<usize>>(),
            size_of::<(&TokTrie, &mut R, &[TokenId])>(),
            size_of::<(&TokTrie, &mut R, &TrieNode)>(),
            size_of::<(&TokTrie, &[TokenId])>(),
            size_of::<std::slice::Iter<'_, TokenId>>(),
            size_of::<(
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                usize,
                u8,
                TokenId,
                bool,
            )>(),
            size_of::<(usize, usize)>(),
            size_of::<Option<usize>>(),
            size_of::<Option<u8>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
}
