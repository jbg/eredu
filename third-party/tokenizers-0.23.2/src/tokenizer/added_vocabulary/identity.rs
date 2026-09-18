//! Shared ordered literal phases when the actual normalizer is absent.
use super::*;
use crate::normalizers::NormalizerWrapper;
use std::mem::{size_of, size_of_val};

/// Borrowed source facts, never a caller-supplied identity or byte allowance.
pub(crate) struct IdentityMatching<'a> {
    source: &'a AddedVocabulary,
    normalized: bool,
}
impl<'a> IdentityMatching<'a> {
    pub(super) fn ordinary(source: &'a AddedVocabulary) -> Self {
        let normalized = source.tokens().any(|(_, token)| token.normalized);
        Self { source, normalized }
    }
    pub(crate) fn checked(
        source: &'a AddedVocabulary,
        normalizer: Option<&NormalizerWrapper>,
    ) -> Option<Self> {
        if normalizer.is_some()
            || !source
                .tokens()
                .all(|(_, t)| !t.single_word && !t.lstrip && !t.rstrip)
        {
            return None;
        }
        Some(Self::ordinary(source))
    }
    pub(crate) fn visit<E>(
        &self,
        sentence: &str,
        mut visit: impl FnMut(Option<u32>, Offsets) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        if !self.normalized {
            return self.source.visit_matches(sentence, false, visit);
        }
        let mut raw = RawPhase {
            source: self.source,
            sentence,
            visit: &mut visit,
        };
        self.source
            .visit_matches(sentence, false, |id, offsets| raw.accept(id, offsets))
    }
    pub(crate) fn control_bytes<E>(&self) -> Option<usize> {
        // Only references to V are stored, so its concrete closure layout does
        // not affect these two types. E's concrete callback state is priced by E.
        type Visit = fn(Option<u32>, Offsets) -> std::result::Result<(), ()>;
        let one = size_of_val(&self.source.raw_matches("", false))
            .checked_add(size_of::<(Option<u32>, Offsets)>())?
            .checked_add(size_of::<AddedTokenRef<'_>>())?
            .checked_add(size_of::<std::result::Result<(), E>>())?;
        let mut bytes = size_of::<Self>()
            .checked_add(size_of::<Option<Self>>())?
            .checked_add(size_of_val(&self.source.tokens()))?
            .checked_add(one)?;
        if self.normalized {
            bytes = bytes
                .checked_add(one)?
                .checked_add(size_of::<RawPhase<'_, Visit>>())?
                .checked_add(size_of::<OffsetPhase<'_, Visit>>())?
                .checked_add(size_of::<&mut RawPhase<'_, Visit>>())?
                .checked_add(size_of::<&mut OffsetPhase<'_, Visit>>())?;
        }
        Some(bytes)
    }
}
// Existing filtered raw spans are authoritative. A skipped special consumes
// shorter raw matches but remains in a coalesced None span for the next phase.
struct RawPhase<'a, V> {
    source: &'a AddedVocabulary,
    sentence: &'a str,
    visit: &'a mut V,
}
impl<V> RawPhase<'_, V> {
    fn accept<E>(&mut self, id: Option<u32>, (start, end): Offsets) -> std::result::Result<(), E>
    where
        V: FnMut(Option<u32>, Offsets) -> std::result::Result<(), E>,
    {
        if id.is_some() {
            return (self.visit)(id, (start, end));
        }
        let mut local = OffsetPhase {
            base: start,
            visit: &mut *self.visit,
        };
        self.source
            .visit_matches(&self.sentence[start..end], true, |id, offsets| {
                local.accept(id, offsets)
            })
    }
}
struct OffsetPhase<'a, V> {
    base: usize,
    visit: &'a mut V,
}
impl<V> OffsetPhase<'_, V> {
    fn accept<E>(&mut self, id: Option<u32>, (start, end): Offsets) -> std::result::Result<(), E>
    where
        V: FnMut(Option<u32>, Offsets) -> std::result::Result<(), E>,
    {
        // The inner matcher only yields ranges within the borrowed outer None
        // slice, so each sum is <= the original input length.
        (self.visit)(id, (self.base + start, self.base + end))
    }
}
#[cfg(test)]
mod tests;
