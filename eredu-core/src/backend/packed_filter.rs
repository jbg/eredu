//! Borrowed packed grammar-mask geometry; no execution or allocation authority.
use super::{TokenFilter, TokenFilterError};
/// A packed semantic mask has no complete finite representation.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PackedTokenFilterError {
    /// The word slice does not exactly cover the declared canonical vocabulary.
    #[error("packed token filter geometry differs from its declared vocabulary")]
    Geometry,
    /// Preserves the ordinary empty-domain and executable-prefix refusals.
    #[error(transparent)]
    Filter(#[from] TokenFilterError),
}
/// A lexical loan of actual LSB-first u32 mask words and tokenizer validity.
/// It owns no payload and cannot authenticate the parser, source, account or
/// constructor. Native consumers still require their original source contract.
#[derive(Debug, Clone, Copy)]
pub struct PackedTokenFilter<'a> {
    words: &'a [u32],
    vocabulary: usize,
    validity: &'a TokenFilter,
}
impl<'a> PackedTokenFilter<'a> {
    /// Validates the actual mask extent without allocating or copying it.
    /// The final word's padding is ignored, exactly like the ordinary bitset.
    pub fn new(
        words: &'a [u32],
        vocabulary: usize,
        validity: &'a TokenFilter,
    ) -> Result<Self, PackedTokenFilterError> {
        if vocabulary == 0 {
            return Err(TokenFilterError::EmptyVocabulary.into());
        }
        if words.len() != vocabulary.div_ceil(32) || u32::try_from(vocabulary - 1).is_err() {
            return Err(PackedTokenFilterError::Geometry);
        }
        let result = Self {
            words,
            vocabulary,
            validity,
        };
        if !(0..vocabulary).any(|token| result.allows(token as u32)) {
            return Err(TokenFilterError::NoAllowedToken.into());
        }
        Ok(result)
    }
    /// Exact mask words, borrowed from the same immutable decision loan.
    pub fn words(self) -> &'a [u32] {
        self.words
    }
    /// Declared canonical bit count, excluding final-word padding.
    pub fn vocabulary(self) -> usize {
        self.vocabulary
    }
    /// Tokenizer baseline used by both sampling and observation.
    pub fn tokenizer_validity(self) -> &'a TokenFilter {
        self.validity
    }
    /// Exact semantic intersection; IDs beyond the source extent are forbidden.
    pub fn allows(self, token: u32) -> bool {
        let index = token as usize;
        index < self.vocabulary
            && self.validity.allows(token)
            && self.words[index / 32] & (1 << (index % 32)) != 0
    }
    /// Same truncation/padding validation as the ordinary concrete filter.
    pub fn validate_output_width(self, output_width: usize) -> Result<(), TokenFilterError> {
        if output_width == 0 {
            return Err(TokenFilterError::EmptyVocabulary);
        }
        if !(0..output_width.min(self.vocabulary)).any(|token| self.allows(token as u32)) {
            return Err(TokenFilterError::NoExecutableToken { output_width });
        }
        Ok(())
    }
    /// Fixed source/validation frames; loops reuse their bounded scalar state.
    pub fn control_bytes() -> usize {
        use std::mem::size_of;
        size_of::<Self>()
            + size_of::<(&[u32], usize, &TokenFilter)>()
            + size_of::<std::ops::Range<usize>>()
            + size_of::<(usize, u32, bool)>()
            + size_of::<Result<Self, PackedTokenFilterError>>()
            + size_of::<Result<(), TokenFilterError>>()
    }
}
