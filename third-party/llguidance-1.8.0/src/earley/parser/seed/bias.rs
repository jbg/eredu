//! Complete byte/numeric/EOS mask through the shared ordinary finalizer.
use super::super::{advance::Context as _, bias as shared, speculation, token};
use super::{advance::Context, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use crate::{api::ParserLimits, earley::PreparedLexer};
use std::mem::{size_of, size_of_val};
use toktrie::{SimpleVob, TokTrie, TokenId};
impl<F: crate::earley::PreparedFunding<Error = E>, E> shared::Context for Context<'_, F> {
    type Error = Cause<E>;
    fn marker(&self) -> TokenId {
        shared::marker_token(self.trie)
    }
    fn eos(&self) -> TokenId {
        self.trie.eos_token()
    }
    fn numeric_ranges(&mut self, mask: &mut SimpleVob) -> Result<(), Self::Error> {
        self.speculative(|context| {
            if token::flush(context)? {
                let state = context
                    .lexer
                    .vector()
                    .state_desc(context.current().lexer_state)
                    .ok_or(Cause::Source)?;
                shared::ranges(context.owner.grammar.lexer_spec(), &state.possible, mask);
            }
            Ok(())
        })
    }
    fn allows_eos(&self) -> bool {
        token::Context::allows_eos(self)
    }
}
impl PreparedEarleySeed {
    fn mask_key(&self) -> Result<shared::Key, ()> {
        let current = self.lexer_stack.last().copied().ok_or(())?;
        Ok(shared::Key::new(
            current,
            speculation::pending(&self.lexer_stack),
        ))
    }
    /// Computes the complete ordinary grammar mask, including numeric ranges
    /// and EOS, under the selected step fuel/state/item policy. Cached results
    /// remain borrowed from this owner; no independent output copy is implied.
    pub fn compute_token_mask<F: crate::earley::PreparedFunding<Error = E>, E>(
        mut self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        start: &[u8],
        funding: &F,
    ) -> Result<Self, PreparedEarleySeedError<E>> {
        let initial_items = self.stats.all_items;
        let initial = (|| {
            let frames = [
                Self::controls::<F, E>().ok_or(Cause::Overflow)?,
                size_of::<Context<'_, F>>(),
                size_of::<shared::Key>(),
                size_of::<Option<shared::Key>>(),
                size_of::<SimpleVob>(),
                size_of::<Option<SimpleVob>>(),
                size_of::<speculation::Snapshot>(),
                size_of::<toktrie::GreedyTokens<'_>>(),
                size_of::<[u8; 1]>(),
                size_of::<std::ops::RangeInclusive<u32>>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<Result<bool, Cause<E>>>(),
                size_of::<Result<Self, PreparedEarleySeedError<E>>>(),
                size_of::<crate::earley::PreparedLexerOperationError<E>>(),
                size_of::<Result<(), crate::earley::PreparedLexerOperationError<E>>>(),
                size_of::<(
                    &mut Self,
                    &mut PreparedLexer,
                    &TokTrie,
                    &ParserLimits,
                    &[u8],
                    &F,
                )>(),
                size_of::<(usize, usize, usize, TokenId, bool)>(),
                size_of::<toktrie::SimpleVobIter<'_>>(),
                size_of::<super::super::LexerResult>(),
                size_of::<Result<super::super::LexerResult, Cause<E>>>(),
                size_of::<Result<bool, Cause<E>>>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<Result<shared::Key, ()>>(),
                size_of::<TokenId>() * 2,
            ];
            funding.reserve(
                frames
                    .into_iter()
                    .try_fold(size_of_val(&frames), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            if !self.row_complete
                || self.backtrack_bytes != 0
                || !self.scratch.as_ref().is_some_and(|s| s.definitive)
            {
                return Err(Cause::Source);
            }
            let key = self.mask_key().map_err(|_| Cause::Source)?;
            if start.is_empty() && self.mask_key == Some(key) && self.token_mask.is_some() {
                return Ok(true);
            }
            lexer.begin_mask(limits, funding).map_err(Cause::Lexer)?;
            self.max_all_items = initial_items
                .checked_add(limits.step_max_items)
                .ok_or(Cause::Overflow)?;
            Ok(false)
        })();
        match initial {
            Ok(true) => return Ok(self),
            Err(cause) => {
                return Err(PreparedEarleySeedError {
                    cause,
                    prefix: self,
                })
            }
            Ok(false) => {}
        }
        self = self.scan_token_mask(lexer, trie, start, funding)?;
        let finish = (|| {
            let maximum = self.max_all_items;
            self.max_all_items = usize::MAX;
            self.stats.lexer_cost = lexer.vector().expressions().cost();
            if self.stats.all_items > maximum {
                return Err(Cause::Step {
                    limit: limits.step_max_items,
                    actual: self.stats.all_items.saturating_sub(initial_items),
                });
            }
            let mut mask = self.token_mask.take().ok_or(Cause::Source)?;
            let scope = derivre::prepared_funding::Scope::new(funding).map_err(Cause::frame)?;
            let result = shared::finish(
                &mut Context {
                    owner: &mut self,
                    lexer,
                    trie,
                    funding: &scope,
                },
                &mut mask,
                start,
            );
            self.token_mask = Some(mask);
            result?;
            self.mask_key = if start.is_empty() {
                Some(self.mask_key().map_err(|_| Cause::Source)?)
            } else {
                None
            };
            Ok(())
        })();
        match finish {
            Ok(()) => Ok(self),
            Err(cause) => Err(PreparedEarleySeedError {
                cause,
                prefix: self,
            }),
        }
    }
}
