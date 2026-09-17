//! Original definitive byte/history owner consuming the shared handoff worker.
use super::super::{advance, scan, Lexeme, LexerResult, LexerState, MatchingLexemesIdx, PreLexeme};
use super::{reserve, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use crate::earley::{PreparedLexer, PreparedLexerOperationError};
use derivre::StateID;
use std::mem::{size_of, size_of_val};
use toktrie::TokTrie;

pub(super) struct Context<'a, F> {
    pub(super) owner: &'a mut PreparedEarleySeed,
    pub(super) lexer: &'a mut PreparedLexer,
    pub(super) trie: &'a TokTrie,
    pub(super) funding: &'a F,
}
impl<F: Fn(usize) -> Result<(), E>, E> Context<'_, F> {
    fn row_bytes(&mut self, last: Option<u8>) -> Result<Vec<u8>, Cause<E>> {
        let row = self.owner.lexer_stack.last().ok_or(Cause::Source)?.row_idx;
        let source = scan::row_bytes(&self.owner.lexer_stack, row);
        let controls = [
            size_of_val(&source),
            size_of::<Option<u8>>(),
            size_of::<Vec<u8>>(),
            size_of::<(&mut Self, usize)>(),
            size_of::<Result<Vec<u8>, Cause<E>>>(),
        ];
        (self.funding)(
            controls
                .into_iter()
                .try_fold(size_of_val(&controls), usize::checked_add)
                .ok_or(Cause::Overflow)?,
        )
        .map_err(Cause::Funding)?;
        let count = source
            .count()
            .checked_add(usize::from(last.is_some()))
            .ok_or(Cause::Overflow)?;
        self.owner.lexeme_bytes.clear();
        reserve(&mut self.owner.lexeme_bytes, count, self.funding)?;
        self.owner
            .lexeme_bytes
            .extend(scan::row_bytes(&self.owner.lexer_stack, row));
        self.owner.lexeme_bytes.reverse();
        if let Some(byte) = last {
            self.owner.lexeme_bytes.push(byte);
        }
        Ok(std::mem::take(&mut self.owner.lexeme_bytes))
    }
    fn lexeme(&mut self, last: Option<u8>, pre: PreLexeme) -> Result<Lexeme, Cause<E>> {
        let (hidden, suffix) = self.props(pre.idx)?;
        let bytes = self.row_bytes(last)?;
        if hidden as usize > bytes.len() {
            return Err(Cause::Source);
        }
        Ok(Lexeme::new(pre.idx, bytes, hidden, suffix))
    }
    fn props(&self, index: MatchingLexemesIdx) -> Result<(u32, bool), Cause<E>> {
        crate::earley::lexer::properties(self.owner.grammar.lexer_spec(), index, |state| {
            self.lexer.vector().state_desc(state)
        })
        .ok_or(Cause::Source)
    }
}
impl<F: Fn(usize) -> Result<(), E>, E> advance::Context for Context<'_, F> {
    type Error = Cause<E>;
    fn enter(&mut self) -> Result<(), Self::Error> {
        let frames = [
            size_of::<&mut Self>(),
            size_of::<Lexeme>(),
            size_of::<(LexerState, LexerState, PreLexeme, PreLexeme)>(),
            size_of::<(Option<u8>, Option<u8>, u32, bool, usize)>(),
            size_of::<Result<bool, Cause<E>>>(),
            size_of::<Result<Lexeme, Cause<E>>>(),
            size_of::<Result<LexerState, Cause<E>>>(),
            size_of::<Result<Option<PreLexeme>, Cause<E>>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, u8>>>(),
        ];
        (self.funding)(
            frames
                .into_iter()
                .try_fold(size_of_val(&frames), usize::checked_add)
                .ok_or(Cause::Overflow)?,
        )
        .map_err(Cause::Funding)
    }
    fn within_items(&self) -> bool {
        self.owner.stats.all_items <= self.owner.max_all_items
    }
    fn make_lexeme(&mut self, byte: Option<u8>, pre: PreLexeme) -> Result<Lexeme, Self::Error> {
        if self.definitive() {
            self.lexeme(byte, pre)
        } else {
            Ok(Lexeme::just_idx(pre.idx))
        }
    }
    fn scan_or_cached(&mut self, lexeme: &Lexeme) -> Result<bool, Self::Error> {
        let next = self.owner.lexer_stack.last().ok_or(Cause::Source)?.row_idx as usize + 1;
        if super::super::speculation::cached(
            self.definitive(),
            next,
            self.owner.rows_valid_end,
            &self.owner.rows,
            lexeme.idx,
        ) {
            self.owner.stats.cached_rows += 1;
            return Ok(true);
        }
        let token = self.owner.token_idx;
        let bytes = self.owner.bytes.len();
        self.owner
            .scan_lexeme(self.lexer, self.trie, lexeme, token, bytes, self.funding)
    }
    fn row_state(&mut self, lexeme: Lexeme, byte: Option<u8>) -> Result<LexerState, Self::Error> {
        let next = self.owner.lexer_stack.last().ok_or(Cause::Source)?.row_idx as usize + 1;
        let start = self
            .owner
            .rows
            .get(next)
            .ok_or(Cause::Source)?
            .lexer_start_state;
        let state = self
            .lexer
            .transition_start_state(start, byte, self.funding)
            .map_err(Cause::Lexer)?;
        if self.definitive() {
            self.owner.row_infos[next - 1].lexeme = lexeme;
            if byte.is_some() {
                let adjust = self.owner.row_infos[next - 1]
                    .start_byte_idx
                    .saturating_sub(1);
                self.owner.row_infos[next].start_byte_idx = self.owner.row_infos[next]
                    .start_byte_idx
                    .checked_sub(adjust)
                    .ok_or(Cause::Source)?;
            }
        }
        Ok(LexerState {
            row_idx: next.try_into().map_err(|_| Cause::Overflow)?,
            lexer_state: state,
            byte,
        })
    }
    fn properties(&self, index: MatchingLexemesIdx) -> Result<(u32, bool), Self::Error> {
        self.props(index)
    }
    fn hidden(
        &mut self,
        state: LexerState,
        byte: Option<u8>,
        pre: PreLexeme,
    ) -> Result<bool, Self::Error> {
        advance::hidden(self, state, byte, pre)
    }
    fn trim_failed_row(&mut self, row: usize) {
        if self.definitive() {
            self.owner.row_infos.drain(row..);
        }
    }
    fn single_byte(&mut self, state: StateID, byte: u8) -> Result<Option<PreLexeme>, Self::Error> {
        self.lexer
            .check_for_single_byte_lexeme(state, byte, self.funding)
            .map_err(Cause::Lexer)
    }
    fn push_state(&mut self, state: LexerState) -> Result<(), Self::Error> {
        let count = self
            .owner
            .lexer_stack
            .len()
            .checked_add(1)
            .ok_or(Cause::Overflow)?;
        reserve(&mut self.owner.lexer_stack, count, self.funding)?;
        self.owner.lexer_stack.push(state);
        Ok(())
    }
    fn pop_state(&mut self) -> LexerState {
        self.owner.lexer_stack.pop().expect("lexical handoff stack")
    }
    fn replace_top(&mut self, state: LexerState) {
        *self
            .owner
            .lexer_stack
            .last_mut()
            .expect("lexical handoff parent") = state;
    }
    fn assert_finished(&self) {
        if self.definitive() {
            assert!(self.owner.backtrack_bytes == 0);
            assert_eq!(
                self.owner
                    .lexer_stack
                    .last()
                    .expect("lexical history")
                    .row_idx as usize
                    + 1,
                self.owner.row_infos.len()
            );
        }
    }
    fn hidden_lexeme(&mut self, byte: Option<u8>, pre: PreLexeme) -> Result<Lexeme, Self::Error> {
        self.lexeme(byte, pre)
    }
    fn added_start(&self) -> StateID {
        self.owner.rows[self.owner.lexer_stack.last().expect("history").row_idx as usize + 1]
            .lexer_start_state
    }
    fn forced_hidden(&self, state: StateID, bytes: &[u8]) -> Result<bool, Self::Error> {
        let allowed = &self
            .lexer
            .vector()
            .state_desc(state)
            .ok_or(Cause::Source)?
            .possible;
        Ok(scan::forced(
            self.owner.grammar.lexer_spec(),
            allowed,
            bytes,
        ))
    }
    fn pop_states(&mut self, count: usize) {
        self.owner
            .lexer_stack
            .truncate(self.owner.lexer_stack.len().saturating_sub(count));
    }
    fn lex_advance(&mut self, state: StateID, byte: u8) -> Result<LexerResult, Self::Error> {
        self.lexer
            .advance(state, byte, self.funding)
            .map_err(Cause::Lexer)
    }
    fn definitive(&self) -> bool {
        self.owner
            .scratch
            .as_ref()
            .is_some_and(|scratch| scratch.definitive)
    }
    fn set_backtrack(&mut self, bytes: usize) {
        self.owner.backtrack_bytes = bytes;
    }
    fn current(&self) -> LexerState {
        *self
            .owner
            .lexer_stack
            .last()
            .expect("active lexical history")
    }
    fn force_end(&mut self, state: StateID) -> Result<LexerResult, Self::Error> {
        self.lexer
            .force_lexeme_end(state, self.funding)
            .map_err(Cause::Lexer)
    }
    fn observe_byte(&mut self) {
        self.owner.stats.definitive_bytes += 1;
    }
    fn backtrack_count(&self) -> usize {
        self.owner.backtrack_bytes
    }
    fn append_byte(&mut self, byte: u8) -> Result<(), Self::Error> {
        let count = self
            .owner
            .bytes
            .len()
            .checked_add(1)
            .ok_or(Cause::Overflow)?;
        reserve(&mut self.owner.bytes, count, self.funding)?;
        self.owner.bytes.push(byte);
        Ok(())
    }
    fn take_backtrack(&mut self) -> usize {
        std::mem::take(&mut self.owner.backtrack_bytes)
    }
    fn finish_backtrack(&mut self, count: usize) {
        assert!(self.owner.grammar.lexer_spec().has_stop);
        self.owner.last_force_bytes_len = usize::MAX;
        self.owner.bytes.truncate(self.owner.bytes.len() - count);
    }
    fn special_token(&mut self, state: StateID) -> Result<bool, Self::Error> {
        let bytes = self.row_bytes(None)?;
        let possible = &self
            .lexer
            .vector()
            .state_desc(state)
            .ok_or(Cause::Source)?
            .possible;
        let Some(index) = scan::special(self.owner.grammar.lexer_spec(), possible, &bytes) else {
            return Ok(false);
        };
        advance::run(
            self,
            PreLexeme {
                idx: MatchingLexemesIdx::Single(index),
                byte: Some(b']'),
                byte_next_row: false,
            },
        )
    }
}

impl PreparedEarleySeed {
    /// Advances actual definitive byte history using the same ordinary parser
    /// routing, scan, capture, hidden-byte and greedy-handoff workers. Composition
    /// supplies the lexer/trie from this chart's retained source chain.
    /// Storage errors own the partial chart; ordinary byte rejection returns it.
    pub fn push_byte<F: Fn(usize) -> Result<(), E>, E>(
        mut self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        byte: Option<u8>,
        funding: &F,
    ) -> Result<(Self, bool, usize), PreparedEarleySeedError<E>> {
        let result = (|| {
            let controls = [
                Self::controls::<F, E>().ok_or(Cause::Overflow)?,
                size_of::<Context<'_, F>>(),
                size_of::<PreparedLexerOperationError<E>>(),
                size_of::<Lexeme>(),
                size_of::<LexerState>(),
                size_of::<PreLexeme>(),
                size_of::<LexerResult>(),
                size_of::<(bool, usize, Option<u8>, StateID, usize)>(),
                size_of::<Result<(Self, bool, usize), PreparedEarleySeedError<E>>>(),
                size_of::<Result<(bool, usize), Cause<E>>>(),
                size_of::<Result<Lexeme, Cause<E>>>(),
                size_of::<Result<LexerState, Cause<E>>>(),
                size_of::<Result<(u32, bool), Cause<E>>>(),
                size_of::<Result<bool, Cause<E>>>(),
                size_of::<Result<Option<PreLexeme>, PreparedLexerOperationError<E>>>(),
                size_of::<Result<LexerResult, PreparedLexerOperationError<E>>>(),
                size_of::<Result<StateID, PreparedLexerOperationError<E>>>(),
                size_of::<std::iter::Enumerate<std::slice::Iter<'_, u8>>>(),
            ];
            funding(
                controls
                    .into_iter()
                    .try_fold(size_of_val(&controls), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            if !self.row_complete
                || self.lexer_stack.is_empty()
                || self.backtrack_bytes != 0
                || !self
                    .scratch
                    .as_ref()
                    .is_some_and(|scratch| scratch.definitive)
            {
                return Err(Cause::Source);
            }
            advance::definitive(
                &mut Context {
                    owner: &mut self,
                    lexer,
                    trie,
                    funding,
                },
                byte,
            )
        })();
        match result {
            Ok((accepted, backtrack)) => Ok((self, accepted, backtrack)),
            Err(cause) => Err(PreparedEarleySeedError {
                cause,
                prefix: self,
            }),
        }
    }
    /// Actual definitive parser byte history, after hidden-byte backtracking.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Current lexer state in the actual byte-history stack.
    pub fn lexer_state(&self) -> Option<StateID> {
        self.row_complete
            .then(|| self.lexer_stack.last().expect("history").lexer_state)
    }
}
