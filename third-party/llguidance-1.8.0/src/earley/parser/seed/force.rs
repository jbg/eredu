//! Original deterministic-byte history through the shared forcing worker.
use crate::earley::PreparedFunding as _;
use super::super::{advance, force as shared, speculation, token};
use super::{advance::Context, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use crate::{api::ParserLimits, earley::PreparedLexer};
use advance::Context as _;
use derivre::NextByte;
use std::mem::{size_of, size_of_val};
use toktrie::{TokTrie, TokenId};
impl<F: crate::earley::PreparedFunding<Error = E>, E> shared::Context for Context<'_, F> {
    fn accepting(&mut self) -> Result<bool, Self::Error> {
        self.speculative(|context| {
            if !token::flush(context)? {
                return Ok(false);
            }
            Ok(speculation::accepting(
                &context.owner.grammar,
                context.owner.scratch.as_ref().expect("forced-byte scratch"),
                &context.owner.rows[context.current().row_idx as usize],
            ))
        })
    }
    fn hint(&mut self) -> Result<NextByte, Self::Error> {
        self.lexer
            .next_byte(self.current().lexer_state, self.funding)
            .map_err(Cause::Lexer)
    }
    fn probe(&mut self, hint: NextByte) -> Result<Option<u8>, Self::Error> {
        self.speculative(|context| {
            shared::unique(hint, |byte| {
                let state = context.current();
                let result = context.lex_advance(state.lexer_state, byte)?;
                let accepted = advance::route(context, result, state)?;
                if accepted {
                    context.pop_states(1);
                }
                Ok(accepted)
            })
        })
    }
    fn unique_numeric(&self) -> Option<TokenId> {
        let state = self
            .lexer
            .vector()
            .state_desc(self.current().lexer_state)
            .expect("current forced-byte state");
        shared::unique_numeric(self.owner.grammar.lexer_spec(), &state.possible)
    }
}
impl PreparedEarleySeed {
    /// Appends exactly the deterministic bytes found by the ordinary quick hint
    /// and speculative unique-byte probe. Numeric token spellings use the same
    /// fixed decimal worker as ordinary parsing. Applied token history is kept.
    pub fn force_bytes<F: crate::earley::PreparedFunding<Error = E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        funding: &F,
    ) -> Result<Self, PreparedEarleySeedError<E>> {
        self.token_operation(lexer, trie, limits, false, funding, |context| {
            let parts = [
                size_of::<NextByte>(),
                size_of::<shared::Numeric>(),
                size_of::<(usize, usize, TokenId, Option<TokenId>, u8, bool)>(),
                size_of::<Context<'_, F>>(),
                size_of::<speculation::Snapshot>(),
                size_of::<Result<Option<u8>, Cause<E>>>(),
                size_of::<Result<NextByte, Cause<E>>>(),
                size_of::<Result<(bool, usize), Cause<E>>>(),
                size_of::<std::slice::Iter<'_, u8>>(),
                size_of::<std::ops::RangeInclusive<u32>>(),
            ];
            context.funding.reserve(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            if context.owner.last_force_bytes_len == context.owner.bytes.len() {
                return Ok(());
            }
            let initial = context.owner.stats.all_items;
            let maximum = initial
                .checked_add(limits.step_max_items)
                .ok_or(Cause::Overflow)?;
            context.owner.max_all_items = maximum;
            shared::run(context)?;
            context.owner.max_all_items = usize::MAX;
            if context.owner.stats.all_items > maximum {
                return Err(Cause::Step {
                    limit: limits.step_max_items,
                    actual: context.owner.stats.all_items.saturating_sub(initial),
                });
            }
            context.assert_finished();
            context.owner.last_force_bytes_len = context.owner.bytes.len();
            Ok(())
        })
        .map(|(owner, ())| owner)
    }
    /// Borrowed deterministic bytes not yet covered by an applied token. Numeric
    /// spellings retain the actual TokTrie marked representation for tokenization.
    pub fn pending_token_bytes(&self) -> &[u8] {
        &self.bytes[self.byte_to_token_idx.len()..]
    }
}
