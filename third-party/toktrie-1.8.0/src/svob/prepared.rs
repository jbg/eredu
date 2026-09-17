//! Source-derived masks used by the ordinary tokenizer slicer and trie walker.
use super::SimpleVob;
use crate::TokTrie;
use std::{
    alloc::Layout,
    collections::TryReserveError,
    fmt,
    mem::{size_of, size_of_val},
};

/// Fixed mask destination failure, without formatted diagnostic storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenMaskSourceError {
    /// Actual source geometry cannot be represented by a vector allocation.
    Overflow,
    /// An allocator returned capacity above the exact requested source extent.
    Capacity,
}
impl fmt::Display for TokenMaskSourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "token mask source {self:?}")
    }
}
impl std::error::Error for TokenMaskSourceError {}
#[derive(Clone, Copy)]
enum Source<'a> {
    Copy(&'a SimpleVob),
    Empty(&'a TokTrie),
    Zeroed(usize),
}
/// Exact retained words and fixed constructor/error representations.
#[derive(Debug, Clone, Copy)]
pub struct TokenMaskConstructionRequirements {
    words: usize,
    buffers: usize,
    controls: usize,
    total: usize,
}
impl TokenMaskConstructionRequirements {
    /// Physical words, including the trie walker's existing sentinel backing.
    pub fn word_capacity(&self) -> usize {
        self.words
    }
    /// One destination vector's exact payload extent.
    pub fn buffer_bytes(&self) -> usize {
        self.buffers
    }
    /// Fixed source, constructor, error and result controls.
    pub fn control_bytes(&self) -> usize {
        self.controls
    }
    /// Complete checked constructor quote, with no runtime funding authority.
    pub fn required_bytes(&self) -> usize {
        self.total
    }
}
/// One immutable source loan. Raw capacities or unrelated storage are not accepted.
pub struct TokenMaskConstructionPlan<'a> {
    source: Source<'a>,
    requirements: TokenMaskConstructionRequirements,
}
impl fmt::Debug for TokenMaskConstructionPlan<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TokenMaskConstructionPlan")
            .field("requirements", &self.requirements)
            .finish_non_exhaustive()
    }
}
impl<'a> TokenMaskConstructionPlan<'a> {
    /// Copies every actual backing word and the logical size, exactly as ordinary
    /// Vec cloning does; spare capacity beyond word length is not invented.
    pub fn copy(source: &'a SimpleVob) -> Result<Self, TokenMaskSourceError> {
        Self::prepare(Source::Copy(source), source.data.len())
    }
    /// Creates the trie walker's actual empty mask with one extra sentinel bit.
    pub fn for_trie(source: &'a TokTrie) -> Result<Self, TokenMaskSourceError> {
        let bits = source
            .vocab_size()
            .checked_add(1)
            .ok_or(TokenMaskSourceError::Overflow)?;
        Self::prepare(Source::Empty(source), bits.div_ceil(u32::BITS as usize))
    }
    /// Quotes the same zero-filled logical bit vector as `SimpleVob::alloc`.
    /// The caller must bind this shape to its actual declaration and reserve the
    /// returned byte population before construction; no execution authority is
    /// supplied by a count or by this storage plan.
    pub fn zeroed(bits: usize) -> Result<Self, TokenMaskSourceError> {
        Self::prepare(Source::Zeroed(bits), bits.div_ceil(u32::BITS as usize))
    }
    /// Exact fixed planning/constructor frames, available before inspecting or
    /// quoting a source shape. This does not allocate or provide authority.
    pub fn inspection_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<Self, TokenMaskSourceError>>(),
            size_of::<TokenMaskConstructionRequirements>(),
            size_of::<Source<'_>>(),
            size_of::<SimpleVob>(),
            size_of::<TokenMaskConstructionFailure>(),
            size_of::<Result<SimpleVob, TokenMaskConstructionFailure>>(),
            size_of::<Result<(), TryReserveError>>(),
            size_of::<Cause>(),
            size_of::<Vec<u32>>(),
            size_of::<(&SimpleVob, usize)>(),
            size_of::<(&TokTrie, usize)>(),
            size_of::<(Source<'_>, usize)>(),
            size_of::<Option<usize>>(),
            size_of::<Result<Layout, std::alloc::LayoutError>>(),
            size_of::<std::slice::Iter<'_, u32>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    fn prepare(source: Source<'a>, words: usize) -> Result<Self, TokenMaskSourceError> {
        let buffers = Layout::array::<u32>(words)
            .map_err(|_| TokenMaskSourceError::Overflow)?
            .size();
        let controls = Self::inspection_control_bytes().ok_or(TokenMaskSourceError::Overflow)?;
        let total = buffers
            .checked_add(controls)
            .ok_or(TokenMaskSourceError::Overflow)?;
        Ok(Self {
            source,
            requirements: TokenMaskConstructionRequirements {
                words,
                buffers,
                controls,
                total,
            },
        })
    }
    /// Exact geometry of this same immutable source loan.
    pub fn requirements(&self) -> TokenMaskConstructionRequirements {
        self.requirements
    }
    /// Performs one fallible vector reserve, then the same copy/zero fill as the
    /// ordinary worker; an error retains any actual allocation prefix.
    pub fn compile(self) -> Result<SimpleVob, TokenMaskConstructionFailure> {
        let mut data = Vec::new();
        if let Err(cause) = data.try_reserve_exact(self.requirements.words) {
            return Err(TokenMaskConstructionFailure {
                cause: Cause::Allocation(cause),
                data,
            });
        }
        if data.capacity() > self.requirements.words {
            return Err(TokenMaskConstructionFailure {
                cause: Cause::Source(TokenMaskSourceError::Capacity),
                data,
            });
        }
        let size = match self.source {
            Source::Copy(source) => {
                data.extend_from_slice(&source.data);
                source.size
            }
            Source::Empty(source) => {
                data.resize(self.requirements.words, 0);
                source.vocab_size()
            }
            Source::Zeroed(bits) => {
                data.resize(self.requirements.words, 0);
                bits
            }
        };
        Ok(SimpleVob { data, size })
    }
}
#[derive(Debug)]
enum Cause {
    Source(TokenMaskSourceError),
    Allocation(TryReserveError),
}
/// Actual failed allocation/capacity result; it offers no prefix extraction.
#[derive(Debug)]
pub struct TokenMaskConstructionFailure {
    cause: Cause,
    data: Vec<u32>,
}
impl TokenMaskConstructionFailure {
    /// Actual retained allocation capacity, in words.
    pub fn word_capacity(&self) -> usize {
        self.data.capacity()
    }
}
impl fmt::Display for TokenMaskConstructionFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.cause {
            Cause::Source(e) => fmt::Display::fmt(e, f),
            Cause::Allocation(e) => fmt::Display::fmt(e, f),
        }
    }
}
impl std::error::Error for TokenMaskConstructionFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match &self.cause {
            Cause::Source(e) => e,
            Cause::Allocation(e) => e,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_mask_copy_preserves_logical_extent_sentinel_backing_and_trimmed_words() {
        let words = (0..32)
            .map(|i| vec![b'a' + (i % 26) as u8])
            .collect::<Vec<_>>();
        let trie = TokTrie::from(&crate::TokRxInfo::new(32, 0), &words);
        let plan = TokenMaskConstructionPlan::for_trie(&trie).unwrap();
        assert_eq!(plan.requirements().word_capacity(), 2);
        let mut mask = plan.compile().unwrap();
        assert_eq!(mask.len(), 32);
        assert_eq!(mask.as_slice(), &[0, 0]);
        mask.allow_token(2);
        mask.allow_token(32); // Existing private walker sentinel is physical backing.
        let copied = TokenMaskConstructionPlan::copy(&mask)
            .unwrap()
            .compile()
            .unwrap();
        assert_eq!(copied, mask);
        assert_eq!(copied.as_slice(), &[4, 1]);
        assert_eq!(mask.clone(), copied);
        mask.disallow_token(32);
        mask.trim_trailing_zeros();
        let trimmed_plan = TokenMaskConstructionPlan::copy(&mask).unwrap();
        assert_eq!(trimmed_plan.requirements().word_capacity(), 1);
        let trimmed = trimmed_plan.compile().unwrap();
        assert_eq!(trimmed.len(), 32);
        assert_eq!(trimmed.as_slice(), &[4]);
        drop((trie, words, mask));
        assert_eq!(copied.as_slice(), &[4, 1]);
        assert!(trimmed.is_allowed(2));
        let empty = SimpleVob::new();
        let empty_plan = TokenMaskConstructionPlan::copy(&empty).unwrap();
        assert_eq!(empty_plan.requirements().buffer_bytes(), 0);
        assert!(empty_plan.compile().unwrap().is_empty());
    }
}
