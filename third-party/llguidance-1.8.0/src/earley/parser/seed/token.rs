//! Paid token history using the ordinary token/numeric/EOS worker.
use crate::earley::PreparedFunding as _;
use super::super::{advance, speculation, token, LexemeIdx, LexemeSet, LexerResult, PreLexeme};
use super::{advance::Context, reserve, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use crate::{api::ParserLimits, earley::PreparedLexer};
use advance::Context as _;
use std::{
    fmt,
    mem::{size_of, size_of_val},
};
use toktrie::{TokTrie, TokenId, TokenMaskConstructionPlan};

#[derive(Debug)]
pub(super) enum TokenApplicationFailure {
    Byte {
        applied: usize,
        byte: u8,
        expected: Option<u8>,
    },
    Numeric,
    Items {
        actual: usize,
        maximum: usize,
    },
}
impl fmt::Display for TokenApplicationFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Byte {
                applied,
                byte,
                expected: Some(expected),
            } => write!(
                f,
                "token byte {byte} differs from forced byte {expected} at {applied}"
            ),
            Self::Byte {
                applied,
                byte,
                expected: None,
            } => write!(f, "token byte {byte} fails grammar at {applied}"),
            Self::Numeric => {
                f.write_str("failed to advance parser after adding bytes ignoring lexer")
            }
            Self::Items { actual, maximum } => {
                write!(f, "current row has {actual} items; maximum is {maximum}")
            }
        }
    }
}
impl std::error::Error for TokenApplicationFailure {}

impl<F: crate::earley::PreparedFunding<Error = E>, E> Context<'_, F> {
    pub(super) fn speculative<T>(
        &mut self,
        run: impl FnOnce(&mut Self) -> Result<T, Cause<E>>,
    ) -> Result<T, Cause<E>> {
        let saved = speculation::begin(
            self.owner.scratch.as_mut().expect("token scratch"),
            &self.owner.lexer_stack,
        );
        self.owner.rows_valid_end = self.current().row_idx as usize + 1;
        let result = run(self);
        if result.is_ok() {
            speculation::finish(
                self.owner.scratch.as_mut().expect("token scratch"),
                &mut self.owner.lexer_stack,
                saved,
            );
            self.owner.rows_valid_end = self.current().row_idx as usize + 1;
            self.owner.lexer_stack_flush_position = 0;
        }
        result
    }
}
impl<F: crate::earley::PreparedFunding<Error = E>, E> token::Context for Context<'_, F> {
    fn rows(&self) -> usize {
        self.current().row_idx as usize + 1
    }
    fn row_start(&self, row: usize) -> usize {
        self.owner.row_infos[row].start_byte_idx
    }
    fn apply_row(&mut self, row: usize, reset: bool) {
        if reset {
            self.owner.row_infos[row].set_token_idx(self.owner.token_idx);
        } else {
            self.owner.row_infos[row].apply_token_idx(self.owner.token_idx);
        }
    }
    fn bytes(&self) -> &[u8] {
        &self.owner.bytes
    }
    fn applied(&self) -> usize {
        self.owner.byte_to_token_idx.len()
    }
    fn append_applied(&mut self, count: usize) -> Result<(), Self::Error> {
        let total = self
            .owner
            .byte_to_token_idx
            .len()
            .checked_add(count)
            .ok_or(Cause::Overflow)?;
        let token = u32::try_from(self.owner.token_idx).map_err(|_| Cause::Overflow)?;
        reserve(&mut self.owner.byte_to_token_idx, total, self.funding)?;
        self.owner.byte_to_token_idx.resize(total, token);
        Ok(())
    }
    fn truncate_applied(&mut self, count: usize) {
        self.owner.byte_to_token_idx.truncate(count);
    }
    fn append_numeric(&mut self, bytes: &[u8]) -> Result<(), Self::Error> {
        let total = self
            .owner
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or(Cause::Overflow)?;
        reserve(&mut self.owner.bytes, total, self.funding)?;
        self.owner.bytes.extend_from_slice(bytes);
        self.append_applied(bytes.len())
    }
    fn trie(&self) -> &TokTrie {
        self.trie
    }
    fn numeric_possible(&mut self, token: TokenId) -> Result<bool, Self::Error> {
        self.speculative(|context| token::numeric(context, token))
            .map(|index| index.is_some())
    }
    fn numeric_index(&self, token: TokenId) -> Option<LexemeIdx> {
        let possible = &self
            .lexer
            .vector()
            .state_desc(self.current().lexer_state)
            .expect("current lexical state")
            .possible;
        self.owner
            .grammar
            .lexer_spec()
            .iter_token_range_lexemes(possible)
            .find(|spec| spec.contains_token(token))
            .map(|spec| spec.idx)
    }
    fn pending(&self) -> bool {
        speculation::pending(&self.owner.lexer_stack)
    }
    fn try_end(&mut self) -> Result<LexerResult, Self::Error> {
        self.lexer
            .try_lexeme_end(self.current().lexer_state, self.funding)
            .map_err(Cause::Lexer)
    }
    fn stack_len(&self) -> usize {
        self.owner.lexer_stack.len()
    }
    fn set_flush_position(&mut self, position: usize) {
        self.owner.lexer_stack_flush_position = position;
    }
    fn flush_position(&self) -> usize {
        self.owner.lexer_stack_flush_position
    }
    fn remove_state(&mut self, position: usize) {
        self.owner.lexer_stack.remove(position);
    }
    fn allows_eos(&self) -> bool {
        self.pending()
            && crate::earley::lexer::allows_eos(
                self.owner.grammar.lexer_spec(),
                self.lexer
                    .vector()
                    .state_desc(self.current().lexer_state)
                    .expect("current lexical state"),
            )
    }
    fn set_top_eos(&mut self) {
        self.owner.lexer_stack_top_eos = true;
    }
    fn limit_tokens(&mut self) -> Result<bool, Self::Error> {
        let row = self.current().row_idx as usize;
        let state = self.current().lexer_state;
        if self.owner.initial_selection.is_some() {
            return Err(Cause::Source);
        }
        let plan = TokenMaskConstructionPlan::zeroed(self.owner.grammar.lexer_spec().lexemes.len())
            .map_err(Cause::MaskSource)?;
        self.funding.reserve(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
        self.owner.initial_selection = Some(LexemeSet::from_owned_vob(
            plan.compile().map_err(Cause::Mask)?,
        ));
        let removed = token::limit(
            self.owner.grammar.lexer_spec(),
            self.owner.scratch.as_ref().expect("token scratch"),
            self.owner.rows[row].grammar_stack_ptr,
            &self
                .lexer
                .vector()
                .state_desc(state)
                .ok_or(Cause::Source)?
                .possible,
            self.owner.token_idx,
            self.owner.row_infos[row].token_idx_start,
            self.owner
                .initial_selection
                .as_mut()
                .expect("token selection"),
        );
        if removed > 0 {
            let next = self
                .lexer
                .limit_state_to(
                    state,
                    self.owner
                        .initial_selection
                        .as_ref()
                        .expect("token selection"),
                    self.funding,
                )
                .map_err(Cause::Lexer)?;
            self.owner.initial_selection = None;
            if next.is_dead() {
                let (accepted, backtrack) = advance::definitive(self, None)?;
                assert_eq!(backtrack, 0);
                if !accepted {
                    return Ok(false);
                }
            } else {
                self.owner
                    .lexer_stack
                    .last_mut()
                    .expect("token stack")
                    .lexer_state = next;
            }
        } else {
            self.owner.initial_selection = None;
        }
        Ok(true)
    }
    fn check_row_items(&self) -> Result<(), Self::Error> {
        let count = self.owner.rows[self.current().row_idx as usize]
            .item_indices()
            .count();
        if count > self.owner.max_items_in_row {
            Err(Cause::Token(TokenApplicationFailure::Items {
                actual: count,
                maximum: self.owner.max_items_in_row,
            }))
        } else {
            Ok(())
        }
    }
    fn rejected_byte(&self, _token: &[u8], byte: u8, expected: Option<u8>) -> Self::Error {
        Cause::Token(TokenApplicationFailure::Byte {
            applied: self.owner.byte_to_token_idx.len(),
            byte,
            expected,
        })
    }
    fn rejected_numeric(&self) -> Self::Error {
        Cause::Token(TokenApplicationFailure::Numeric)
    }
}

impl PreparedEarleySeed {
    pub(super) fn token_operation<T, F, E, G>(
        mut self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        advance_token: bool,
        funding: &F,
        run: G,
    ) -> Result<(Self, T), PreparedEarleySeedError<E>>
    where
        F: crate::earley::PreparedFunding<Error = E>,
        G: FnOnce(&mut Context<'_, F>) -> Result<T, Cause<E>>,
    {
        let result = (|| {
            let parts = [
                Self::controls::<F, E>().ok_or(Cause::Overflow)?,
                size_of::<Context<'_, F>>(),
                size_of::<G>(),
                size_of::<T>(),
                size_of::<speculation::Snapshot>(),
                size_of::<TokenApplicationFailure>(),
                size_of::<TokenMaskConstructionPlan<'_>>(),
                size_of::<Result<(Self, T), PreparedEarleySeedError<E>>>(),
                size_of::<Result<T, Cause<E>>>(),
                size_of::<Result<usize, Cause<E>>>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<Result<bool, Cause<E>>>(),
                size_of::<Result<Option<LexemeIdx>, Cause<E>>>(),
                size_of::<(&mut Self, &mut PreparedLexer, &TokTrie, &ParserLimits, &F)>(),
                size_of::<(usize, usize, usize, u8, TokenId, bool)>(),
                size_of::<PreLexeme>(),
                size_of::<super::super::Lexeme>(),
                size_of::<super::super::LexerState>(),
                size_of::<LexerResult>(),
                size_of::<crate::earley::PreparedLexerOperationError<E>>(),
                size_of::<Result<LexerResult, crate::earley::PreparedLexerOperationError<E>>>(),
                size_of::<
                    Result<super::super::StateID, crate::earley::PreparedLexerOperationError<E>>,
                >(),
                size_of::<Result<Option<PreLexeme>, crate::earley::PreparedLexerOperationError<E>>>(
                ),
                size_of::<Result<super::super::Lexeme, Cause<E>>>(),
                size_of::<Result<super::super::LexerState, Cause<E>>>(),
                size_of::<Result<(u32, bool), Cause<E>>>(),
                size_of::<std::iter::Enumerate<std::slice::Iter<'_, u8>>>(),
                size_of::<std::ops::Range<usize>>(),
                size_of::<toktrie::SimpleVobIter<'_>>(),
                size_of::<super::super::GrammarStackPtr>(),
            ];
            funding.reserve(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            if !self.row_complete
                || self.backtrack_bytes != 0
                || !self
                    .scratch
                    .as_ref()
                    .is_some_and(|scratch| scratch.definitive)
            {
                return Err(Cause::Source);
            }
            u32::try_from(self.token_idx.checked_add(1).ok_or(Cause::Overflow)?)
                .map_err(|_| Cause::Overflow)?;
            self.max_items_in_row = limits.max_items_in_row;
            let scope = derivre::prepared_funding::Scope::new(funding).map_err(Cause::frame)?;
            let result = run(&mut Context {
                owner: &mut self,
                lexer,
                trie,
                funding: &scope,
            });
            if advance_token {
                self.token_idx += 1;
            }
            result
        })();
        match result {
            Ok(value) => Ok((self, value)),
            Err(cause) => Err(PreparedEarleySeedError {
                cause,
                prefix: self,
            }),
        }
    }
    /// Applies an actual token through the shared numeric/forced-byte/hidden-stop
    /// and max-token worker. Failures retain partial token and parser history.
    pub fn apply_token<F: crate::earley::PreparedFunding<Error = E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        bytes: &[u8],
        token: TokenId,
        funding: &F,
    ) -> Result<(Self, usize), PreparedEarleySeedError<E>> {
        self.token_operation(lexer, trie, limits, true, funding, |context| {
            token::apply(context, bytes, token)
        })
    }
    /// Consumes EOS through the same pending-lexeme flush and terminal rules.
    pub fn scan_eos<F: crate::earley::PreparedFunding<Error = E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        funding: &F,
    ) -> Result<(Self, bool), PreparedEarleySeedError<E>> {
        self.token_operation(lexer, trie, limits, false, funding, |context| {
            token::eos(context)
        })
    }
    /// Checks actual completed-start-rule acceptance without committing the
    /// speculative flush or changing captures/history.
    pub fn is_accepting<F: crate::earley::PreparedFunding<Error = E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        funding: &F,
    ) -> Result<(Self, bool), PreparedEarleySeedError<E>> {
        self.token_operation(lexer, trie, limits, false, funding, |context| {
            context.speculative(|context| {
                if !token::flush(context)? {
                    return Ok(false);
                }
                let row = context.current().row_idx as usize;
                Ok(speculation::accepting(
                    &context.owner.grammar,
                    context.owner.scratch.as_ref().expect("accepting scratch"),
                    &context.owner.rows[row],
                ))
            })
        })
    }
    /// Applied bytes' actual token ordinals, after any hidden-stop backtrack.
    pub fn byte_token_indices(&self) -> &[u32] {
        &self.byte_to_token_idx
    }
}
