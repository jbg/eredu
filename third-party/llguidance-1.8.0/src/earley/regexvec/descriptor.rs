//! Shared DFA state descriptor equations; destinations are owned by the caller.
use super::{
    ExprRef, ExprSet, LexemeIdx, LexemeSet, MatchingLexemes, NextByte, NextByteCache, StateDesc,
    StateID,
};
pub(super) trait Context {
    type Error;
    fn source(&self) -> &ExprSet;
    fn next_byte(&mut self, root: ExprRef) -> Result<NextByte, Self::Error>;
    fn prepare_matches(
        &mut self,
        values: &mut Vec<LexemeIdx>,
        total: usize,
    ) -> Result<(), Self::Error>;
}
pub(super) struct Selection {
    pub all_eoi: bool,
    pub eois: MatchingLexemes,
    pub lazies: MatchingLexemes,
    pub hidden_len: u32,
}
impl Selection {
    pub fn new() -> Self {
        Self {
            all_eoi: true,
            eois: MatchingLexemes::None,
            lazies: MatchingLexemes::None,
            hidden_len: 0,
        }
    }
}
pub(super) fn pairs(words: &[u32]) -> impl Iterator<Item = (LexemeIdx, ExprRef)> + '_ {
    words
        .chunks_exact(2)
        .map(|pair| (LexemeIdx::new(pair[0] as usize), ExprRef::new(pair[1])))
}
pub(super) fn possible<C: Context>(
    context: &mut C,
    words: &[u32],
    desc: &mut StateDesc,
) -> Result<(), C::Error> {
    for (index, root) in pairs(words) {
        desc.possible.add(index);
        if context.source().is_nullable(root) {
            desc.greedy_accepting.add_with(index, |values, total| {
                context.prepare_matches(values, total)
            })?;
        }
    }
    if desc.possible.is_empty() {
        assert!(desc.state == StateID::DEAD);
    }
    Ok(())
}
pub(super) fn lowest<C: Context>(
    context: &mut C,
    words: &[u32],
    roots: &[ExprRef],
    special: Option<ExprRef>,
    lazy: &LexemeSet,
    desc: &mut StateDesc,
    selection: &mut Selection,
) -> Result<(), C::Error> {
    for (index, root) in pairs(words) {
        if !context.source().is_nullable(root) {
            selection.all_eoi = false;
            continue;
        }
        if Some(roots[index.as_usize()]) == special {
            desc.lazy_accepting = MatchingLexemes::One(index);
            desc.has_special_token = true;
            return Ok(());
        }
        if lazy.contains(index) {
            if selection.lazies.is_none() {
                selection.all_eoi = false;
                selection.hidden_len = context.source().possible_lookahead_len(root) as u32;
            }
            selection.lazies.add_with(index, |values, total| {
                context.prepare_matches(values, total)
            })?;
            continue;
        }
        if selection.all_eoi {
            if context.next_byte(root)? == NextByte::ForcedEOI {
                selection.eois.add_with(index, |values, total| {
                    context.prepare_matches(values, total)
                })?;
            } else {
                selection.all_eoi = false;
            }
        }
    }
    if selection.lazies.is_some() {
        desc.lazy_accepting = std::mem::replace(&mut selection.lazies, MatchingLexemes::None);
        desc.lazy_hidden_len = selection.hidden_len;
    } else if selection.all_eoi {
        desc.lazy_accepting = std::mem::replace(&mut selection.eois, MatchingLexemes::None);
    }
    Ok(())
}
pub(super) struct Ordinary<'a> {
    pub source: &'a ExprSet,
    pub next: &'a mut NextByteCache,
}
impl Context for Ordinary<'_> {
    type Error = std::convert::Infallible;
    fn source(&self) -> &ExprSet {
        self.source
    }
    fn next_byte(&mut self, root: ExprRef) -> Result<NextByte, Self::Error> {
        Ok(self.next.next_byte(self.source, root))
    }
    fn prepare_matches(
        &mut self,
        values: &mut Vec<LexemeIdx>,
        total: usize,
    ) -> Result<(), Self::Error> {
        if values.capacity() == 0 {
            values.reserve_exact(total);
        } else {
            values.reserve(total - values.len());
        }
        Ok(())
    }
}

/// Same hidden-length source scan used by the ordinary descriptor cache.
pub(super) fn possible_hidden(source: &ExprSet, words: &[u32]) -> usize {
    pairs(words)
        .map(|(_, root)| source.possible_lookahead_len(root))
        .max()
        .unwrap_or(0)
}
pub(super) fn lookahead(
    source: &ExprSet,
    words: &[u32],
    greedy: &MatchingLexemes,
) -> Option<usize> {
    let mut result = None;
    for (index, root) in pairs(words) {
        if result.is_none() && source.is_nullable(root) {
            assert!(greedy.contains(index));
            result = Some(source.lookahead_len(root).unwrap_or(0));
        }
    }
    result
}
pub(super) fn next_bytes<C: Context>(context: &mut C, words: &[u32]) -> Result<NextByte, C::Error> {
    let mut next = NextByte::Dead;
    for (_, root) in pairs(words) {
        next = next | context.next_byte(root)?;
        if next.is_some_bytes() {
            break;
        }
    }
    Ok(next)
}
pub(super) fn restricted<E>(
    words: &[u32],
    allowed: &LexemeSet,
    mut emit: impl FnMut(LexemeIdx, ExprRef) -> Result<(), E>,
) -> Result<(), E> {
    for (index, root) in pairs(words) {
        if allowed.contains(index) {
            emit(index, root)?;
        }
    }
    Ok(())
}
