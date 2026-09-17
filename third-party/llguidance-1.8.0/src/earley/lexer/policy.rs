//! Allocation-free lexical decisions shared by ordinary and prepared workers.
use super::{LexerResult, MatchingLexemesIdx, NextByte, PreLexeme, StateDesc, StateID};
use toktrie::SimpleVob;

pub(super) fn warm_first_bytes<E>(
    allowed: &mut SimpleVob,
    mut transition: impl FnMut(u8) -> Result<StateID, E>,
) -> Result<(), E> {
    for byte in 0..=255 {
        if !transition(byte)?.is_dead() {
            allowed.allow_token(byte as u32);
        }
    }
    Ok(())
}

pub(super) fn advance(
    prev: StateID,
    byte: u8,
    state: StateID,
    first: &SimpleVob,
    previous: &StateDesc,
    current: &StateDesc,
) -> LexerResult {
    if state.is_dead() {
        if !first.is_allowed(byte as u32) {
            return LexerResult::Error;
        }
        if previous.greedy_accepting.is_some() {
            LexerResult::Lexeme(PreLexeme {
                idx: MatchingLexemesIdx::GreedyAccepting(prev),
                byte: Some(byte),
                byte_next_row: true,
            })
        } else {
            LexerResult::Error
        }
    } else if state.has_lowest_match() {
        assert!(current.lazy_accepting.is_some());
        if current.has_special_token {
            return LexerResult::SpecialToken(state);
        }
        LexerResult::Lexeme(PreLexeme {
            idx: MatchingLexemesIdx::LazyAccepting(state),
            byte: Some(byte),
            byte_next_row: false,
        })
    } else {
        LexerResult::State(state, byte)
    }
}

pub(super) fn force_end(info: &StateDesc) -> LexerResult {
    match info.possible.first() {
        Some(idx) => LexerResult::Lexeme(PreLexeme::just_idx(MatchingLexemesIdx::Single(idx))),
        None => LexerResult::Error,
    }
}
pub(super) fn try_end(state: StateID, info: &StateDesc) -> LexerResult {
    if info.greedy_accepting.is_some() {
        LexerResult::Lexeme(PreLexeme::just_idx(MatchingLexemesIdx::GreedyAccepting(
            state,
        )))
    } else {
        LexerResult::Error
    }
}
pub(super) fn single_byte(state: StateID, byte: u8, next: NextByte) -> Option<PreLexeme> {
    (next == NextByte::ForcedEOI).then_some(PreLexeme {
        idx: MatchingLexemesIdx::GreedyAccepting(state),
        byte: Some(byte),
        byte_next_row: false,
    })
}
pub(super) fn next_byte(next: NextByte, info: &StateDesc) -> NextByte {
    if info.greedy_accepting.is_some() {
        next.make_fuzzy()
    } else {
        next
    }
}

// One typed interpretation of lexical match handles. The immutable spec supplies
// singletons and the actual state owner supplies greedy/lazy descriptor rows.
pub(crate) fn matching<'a>(
    spec: &'a super::LexerSpec,
    idx: MatchingLexemesIdx,
    state: impl FnOnce(StateID) -> Option<&'a StateDesc>,
) -> Option<&'a super::MatchingLexemes> {
    match idx {
        MatchingLexemesIdx::Single(id) => spec
            .lexemes
            .get(id.as_usize())
            .map(|lexeme| &lexeme.single_set),
        MatchingLexemesIdx::GreedyAccepting(id) => state(id).map(|info| &info.greedy_accepting),
        MatchingLexemesIdx::LazyAccepting(id) => state(id).map(|info| &info.lazy_accepting),
    }
}

pub(crate) fn properties<'a>(
    spec: &super::LexerSpec,
    idx: MatchingLexemesIdx,
    state: impl FnOnce(StateID) -> Option<&'a StateDesc>,
) -> Option<(u32, bool)> {
    match idx {
        MatchingLexemesIdx::Single(_) | MatchingLexemesIdx::GreedyAccepting(_) => Some((0, false)),
        MatchingLexemesIdx::LazyAccepting(id) => {
            let info = state(id)?;
            if info.lazy_hidden_len > 0 {
                let lexeme = spec.lexemes.get(info.lazy_accepting.first()?.as_usize())?;
                Some((info.lazy_hidden_len, lexeme.is_suffix))
            } else {
                Some((0, false))
            }
        }
    }
}

pub(crate) fn allows_eos(spec: &super::LexerSpec, state: &StateDesc) -> bool {
    state
        .greedy_accepting
        .as_slice()
        .iter()
        .any(|index| spec.lexeme_ends_at_eos(*index))
}
