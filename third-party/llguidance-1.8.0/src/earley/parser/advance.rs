//! One lexical-row handoff, including greedy transition and second-byte lexemes.
use super::{Lexeme, LexerState, MatchingLexemesIdx, PreLexeme, ITEM_TRACE};
use derivre::StateID;

pub(super) trait Context {
    type Error;
    type Frame;
    fn enter(&mut self) -> Result<Self::Frame, Self::Error>;
    fn within_items(&self) -> bool;
    fn make_lexeme(&mut self, byte: Option<u8>, pre: PreLexeme) -> Result<Lexeme, Self::Error>;
    fn scan_or_cached(&mut self, lexeme: &Lexeme) -> Result<bool, Self::Error>;
    fn row_state(&mut self, lexeme: Lexeme, byte: Option<u8>) -> Result<LexerState, Self::Error>;
    fn properties(&self, index: MatchingLexemesIdx) -> Result<(u32, bool), Self::Error>;
    fn hidden(
        &mut self,
        state: LexerState,
        byte: Option<u8>,
        pre: PreLexeme,
    ) -> Result<bool, Self::Error>;
    fn trim_failed_row(&mut self, row: usize);
    fn single_byte(&mut self, state: StateID, byte: u8) -> Result<Option<PreLexeme>, Self::Error>;
    fn push_state(&mut self, state: LexerState) -> Result<(), Self::Error>;
    fn pop_state(&mut self) -> LexerState;
    fn replace_top(&mut self, state: LexerState);
    fn assert_finished(&self);
    fn hidden_lexeme(&mut self, byte: Option<u8>, pre: PreLexeme) -> Result<Lexeme, Self::Error>;
    fn added_start(&self) -> StateID;
    fn forced_hidden(&self, state: StateID, bytes: &[u8]) -> Result<bool, Self::Error>;
    fn pop_states(&mut self, count: usize);
    fn lex_advance(&mut self, state: StateID, byte: u8) -> Result<super::LexerResult, Self::Error>;
    fn definitive(&self) -> bool;
    fn set_backtrack(&mut self, bytes: usize);
    fn current(&self) -> LexerState;
    fn force_end(&mut self, state: StateID) -> Result<super::LexerResult, Self::Error>;
    fn observe_byte(&mut self);
    fn backtrack_count(&self) -> usize;
    fn append_byte(&mut self, byte: u8) -> Result<(), Self::Error>;
    fn take_backtrack(&mut self) -> usize;
    fn finish_backtrack(&mut self, count: usize);
    fn special_token(&mut self, state: StateID) -> Result<bool, Self::Error>;
}

#[inline(never)]
pub(super) fn run<C: Context>(context: &mut C, pre: PreLexeme) -> Result<bool, C::Error> {
    let _frame = context.enter()?;
    if !context.within_items() {
        return Ok(false);
    }
    let transition = if pre.byte_next_row { pre.byte } else { None };
    let final_byte = if pre.byte_next_row { None } else { pre.byte };
    let lexeme = context.make_lexeme(final_byte, pre)?;
    if !context.scan_or_cached(&lexeme)? {
        return Ok(false);
    }
    let mut state = context.row_state(lexeme, transition)?;
    let (hidden, suffix) = context.properties(pre.idx)?;
    if hidden > 0 && !suffix {
        return context.hidden(state, final_byte, pre);
    }
    if pre.byte_next_row && state.lexer_state.is_dead() {
        context.trim_failed_row(state.row_idx as usize);
        return Ok(false);
    }
    if let Some(byte) = transition {
        if let Some(second) = context.single_byte(state.lexer_state, byte)? {
            state.byte = None;
            context.push_state(state)?;
            assert!(pre.byte_next_row);
            assert!(!second.byte_next_row);
            let result = run(context, second)?;
            if result {
                let top = context.pop_state();
                context.replace_top(top);
            } else {
                context.pop_state();
            }
            return Ok(result);
        }
    }
    context.push_state(state)?;
    context.assert_finished();
    Ok(true)
}

pub(super) fn hidden<C: Context>(
    context: &mut C,
    no_hidden: LexerState,
    last: Option<u8>,
    pre: PreLexeme,
) -> Result<bool, C::Error> {
    let _frame = context.enter()?;
    let start = context.added_start();
    let lexeme = context.hidden_lexeme(last, pre)?;
    let bytes = lexeme.hidden_bytes();
    if context.forced_hidden(start, bytes)? {
        let mut state = start;
        context.pop_states(bytes.len() - 1);
        for (index, &byte) in bytes.iter().enumerate() {
            match context.lex_advance(state, byte)? {
                super::LexerResult::State(next, _) => state = next,
                super::LexerResult::SpecialToken(_) => {
                    panic!("hidden byte resulted in special token")
                }
                super::LexerResult::Error => panic!("hidden byte failed"),
                super::LexerResult::Lexeme(second) => {
                    assert!(
                        index == bytes.len() - 1,
                        "lexeme in the middle of hidden bytes"
                    );
                    context.push_state(LexerState {
                        lexer_state: state,
                        byte: None,
                        ..no_hidden
                    })?;
                    let result = run(context, second)?;
                    if result {
                        let top = context.pop_state();
                        context.replace_top(top);
                    } else {
                        context.pop_state();
                    }
                    return Ok(result);
                }
            }
            context.push_state(LexerState {
                lexer_state: state,
                byte: Some(byte),
                ..no_hidden
            })?;
        }
        context.assert_finished();
    } else if context.definitive() {
        context.push_state(LexerState {
            lexer_state: start,
            byte: None,
            ..no_hidden
        })?;
        context.assert_finished();
        context.set_backtrack(bytes.len());
    } else {
        context.push_state(LexerState {
            lexer_state: StateID::DEAD,
            byte: None,
            ..no_hidden
        })?;
    }
    Ok(true)
}

pub(super) fn route<C: Context>(
    context: &mut C,
    result: super::LexerResult,
    current: LexerState,
) -> Result<bool, C::Error> {
    match result {
        super::LexerResult::State(next, byte) => {
            context.push_state(LexerState {
                row_idx: current.row_idx,
                lexer_state: next,
                byte: Some(byte),
            })?;
            Ok(true)
        }
        super::LexerResult::Error => Ok(false),
        super::LexerResult::Lexeme(pre) => run(context, pre),
        super::LexerResult::SpecialToken(state) => context.special_token(state),
    }
}

pub(super) fn definitive<C: Context>(
    context: &mut C,
    byte: Option<u8>,
) -> Result<(bool, usize), C::Error> {
    assert!(context.definitive());
    let current = context.current();
    let result = if let Some(byte) = byte {
        context.observe_byte();
        context.lex_advance(current.lexer_state, byte)?
    } else {
        context.force_end(current.lexer_state)?
    };
    assert!(context.backtrack_count() == 0);
    if route(context, result, current)? {
        if let Some(byte) = byte {
            context.append_byte(byte)?;
        }
        let backtrack = context.take_backtrack();
        if backtrack > 0 {
            context.finish_backtrack(backtrack);
        }
        Ok((true, backtrack))
    } else {
        Ok((false, 0))
    }
}

impl Context for super::ParserState {
    type Error = anyhow::Error;
    type Frame = ();
    fn enter(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
    fn within_items(&self) -> bool {
        self.stats.all_items <= self.max_all_items
    }
    fn make_lexeme(&mut self, byte: Option<u8>, pre: PreLexeme) -> Result<Lexeme, Self::Error> {
        Ok(if self.scratch.definitive {
            self.mk_lexeme(byte, pre)
        } else {
            Lexeme::just_idx(pre.idx)
        })
    }
    fn scan_or_cached(&mut self, lexeme: &Lexeme) -> Result<bool, Self::Error> {
        if super::speculation::cached(
            self.scratch.definitive,
            self.num_rows(),
            self.rows_valid_end,
            &self.rows,
            lexeme.idx,
        ) {
            self.stats.cached_rows += 1;
            return Ok(true);
        }
        let result = self.scan(lexeme);
        if result && super::ITEM_TRACE {
            let added = self.num_rows();
            let row = &self.rows[added];
            item_trace!(
                "  row: {:?} -> {}",
                self.lexer_stack_top(),
                row.item_indices().len()
            );
            if self.stats.all_items > self.max_all_items {
                panic!("max items exceeded");
            }
        }
        Ok(result)
    }
    fn row_state(&mut self, lexeme: Lexeme, byte: Option<u8>) -> Result<LexerState, Self::Error> {
        Ok(self.lexer_state_for_added_row(lexeme, byte))
    }
    fn properties(&self, index: MatchingLexemesIdx) -> Result<(u32, bool), Self::Error> {
        Ok(self.lexer().lexeme_props(index))
    }
    fn hidden(
        &mut self,
        state: LexerState,
        byte: Option<u8>,
        pre: PreLexeme,
    ) -> Result<bool, Self::Error> {
        Ok(self.handle_hidden_bytes(state, byte, pre))
    }
    fn trim_failed_row(&mut self, row: usize) {
        if self.scratch.definitive {
            self.row_infos.drain(row..);
        }
    }
    fn single_byte(&mut self, state: StateID, byte: u8) -> Result<Option<PreLexeme>, Self::Error> {
        Ok(self.lexer_mut().check_for_single_byte_lexeme(state, byte))
    }
    fn push_state(&mut self, state: LexerState) -> Result<(), Self::Error> {
        self.lexer_stack.push(state);
        Ok(())
    }
    fn pop_state(&mut self) -> LexerState {
        self.lexer_stack.pop().unwrap()
    }
    fn replace_top(&mut self, state: LexerState) {
        *self.lexer_stack.last_mut().unwrap() = state;
    }
    fn assert_finished(&self) {
        if self.scratch.definitive {
            self.assert_definitive_inner();
        }
    }
    fn hidden_lexeme(&mut self, byte: Option<u8>, pre: PreLexeme) -> Result<Lexeme, Self::Error> {
        Ok(self.mk_lexeme(byte, pre))
    }
    fn added_start(&self) -> StateID {
        self.rows[self.num_rows()].lexer_start_state
    }
    fn forced_hidden(&self, state: StateID, bytes: &[u8]) -> Result<bool, Self::Error> {
        Ok(self.has_forced_bytes(self.lexer().possible_lexemes(state), bytes))
    }
    fn pop_states(&mut self, count: usize) {
        self.pop_lexer_states(count);
    }
    fn lex_advance(&mut self, state: StateID, byte: u8) -> Result<super::LexerResult, Self::Error> {
        Ok(self.lexer_mut().advance(state, byte, false))
    }
    fn definitive(&self) -> bool {
        self.scratch.definitive
    }
    fn set_backtrack(&mut self, bytes: usize) {
        self.backtrack_byte_count = bytes;
    }
    fn current(&self) -> LexerState {
        self.lexer_state()
    }
    fn force_end(&mut self, state: StateID) -> Result<super::LexerResult, Self::Error> {
        Ok(self.lexer_mut().force_lexeme_end(state))
    }
    fn observe_byte(&mut self) {
        self.stats.definitive_bytes += 1;
    }
    fn backtrack_count(&self) -> usize {
        self.backtrack_byte_count
    }
    fn append_byte(&mut self, byte: u8) -> Result<(), Self::Error> {
        self.bytes.push(byte);
        Ok(())
    }
    fn take_backtrack(&mut self) -> usize {
        std::mem::take(&mut self.backtrack_byte_count)
    }
    fn finish_backtrack(&mut self, count: usize) {
        assert!(self.lexer_spec().has_stop);
        self.last_force_bytes_len = usize::MAX;
        self.bytes.truncate(self.bytes.len() - count);
    }
    fn special_token(&mut self, state: StateID) -> Result<bool, Self::Error> {
        Ok(self.special_pre_lexeme(state))
    }
}
