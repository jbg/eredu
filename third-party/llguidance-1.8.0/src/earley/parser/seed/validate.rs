//! Paid speculative token validation, retaining failed lexical/chart prefixes.
use super::super::{advance, force::Numeric, speculation, token, validate};
use super::{advance::Context, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use crate::{api::ParserLimits, earley::PreparedLexer};
use advance::Context as _;
use std::mem::{size_of, size_of_val};
use toktrie::{TokTrie, TokenId};
impl<F: crate::earley::PreparedFunding<Error = E>, E> validate::Context for Context<'_, F> {
    fn accepting_inner(&mut self) -> Result<bool, Self::Error> {
        if !token::flush(self)? {
            return Ok(false);
        }
        Ok(speculation::accepting(
            &self.owner.grammar,
            self.owner.scratch.as_ref().expect("validation scratch"),
            &self.owner.rows[self.current().row_idx as usize],
        ))
    }
    fn restore_stack(&mut self, length: usize) {
        self.owner.lexer_stack.truncate(length);
    }
    fn probe(&mut self, byte: u8) -> Result<bool, Self::Error> {
        let state = self.current();
        let next = self.lex_advance(state.lexer_state, byte)?;
        advance::route(self, next, state)
    }
}
impl PreparedEarleySeed {
    /// Validates actual token IDs without committing rows, captures or token
    /// ordinals. Like the ordinary multi-token query, this can ignore a
    /// per-lexeme max-token limit across the supplied token sequence.
    /// The supplied lexer/trie must belong to this owner's retained source.
    pub fn validate_tokens<F: crate::earley::PreparedFunding<Error = E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        tokens: &[TokenId],
        funding: &F,
    ) -> Result<(Self, usize), PreparedEarleySeedError<E>> {
        self.token_operation(lexer, trie, limits, false, funding, |context| {
            let parts = [
                size_of::<Numeric>(),
                size_of::<(usize, usize, usize, u8, TokenId, bool)>(),
                size_of::<(&mut Context<'_, F>, &[TokenId])>(),
                size_of::<std::iter::Enumerate<std::slice::Iter<'_, TokenId>>>(),
                size_of::<std::ops::Range<usize>>(),
                size_of::<Result<usize, Cause<E>>>(),
                size_of::<Result<bool, Cause<E>>>(),
                size_of::<Result<(), E>>(),
            ];
            funding.reserve(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            if tokens.iter().any(|&id| id as usize >= trie.vocab_size()) {
                return Err(Cause::Source);
            }
            context.speculative(|context| {
                context
                    .owner
                    .scratch
                    .as_mut()
                    .expect("validation scratch")
                    .log_override = true;
                validate::run(context, tokens)
            })
        })
    }
    /// Whether the actual committed row can consume another byte, including a
    /// partially consumed lexeme. None means the initial row is not published.
    pub fn can_advance(&self) -> Option<bool> {
        if !self.row_complete {
            return None;
        }
        let state = self.lexer_stack.last()?;
        Some(
            speculation::pending(&self.lexer_stack)
                || speculation::can_advance(
                    &self.grammar,
                    self.scratch.as_ref()?,
                    self.rows.get(state.row_idx as usize)?,
                ),
        )
    }
}
