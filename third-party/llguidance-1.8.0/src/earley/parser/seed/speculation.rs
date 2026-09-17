//! The actual trie walker over paid speculative rows, with typed failure custody.
use super::super::{advance, speculation as shared};
use super::{advance::Context, Cause, PreparedEarleySeed, PreparedEarleySeedError};
use crate::earley::PreparedLexer;
use advance::Context as _;
use std::mem::{size_of, size_of_val};
use toktrie::{Recognizer, SimpleVob, TokTrie, TokenMaskConstructionPlan};

struct Walker<'a, F, E> {
    context: Context<'a, F>,
    saved: Option<shared::Snapshot>,
    failure: Option<Cause<E>>,
}
impl<F: Fn(usize) -> Result<(), E>, E> Walker<'_, F, E> {
    fn finish(&mut self) {
        if let Some(saved) = self.saved.take() {
            // A failed recursive handoff may own a partially rewritten lexical
            // suffix. Preserve it in the terminal error rather than publishing
            // a reconstructed committed history or refunding its destinations.
            if self.failure.is_none() {
                shared::finish(
                    self.context.owner.scratch.as_mut().expect("trie scratch"),
                    &mut self.context.owner.lexer_stack,
                    saved,
                );
                self.context.owner.rows_valid_end = self.context.current().row_idx as usize + 1;
            }
        }
    }
}
impl<F: Fn(usize) -> Result<(), E>, E> Recognizer for Walker<'_, F, E> {
    fn collapse(&mut self) {}
    fn trie_started(&mut self, _label: &str) {
        if self.failure.is_some() {
            return;
        }
        assert!(self.saved.is_none());
        self.saved = Some(shared::begin(
            self.context.owner.scratch.as_mut().expect("trie scratch"),
            &self.context.owner.lexer_stack,
        ));
        self.context.owner.rows_valid_end = self.context.current().row_idx as usize + 1;
    }
    fn trie_finished(&mut self) {
        self.finish();
    }
    fn pop_bytes(&mut self, count: usize) {
        if self.failure.is_none() {
            self.context.pop_states(count);
        }
    }
    fn try_push_byte(&mut self, byte: u8) -> bool {
        if self.failure.is_some() {
            return false;
        }
        let current = self.context.current();
        let result = self
            .context
            .lex_advance(current.lexer_state, byte)
            .and_then(|result| advance::route(&mut self.context, result, current));
        match result {
            Ok(accepted) => accepted,
            Err(cause) => {
                self.failure = Some(cause);
                false
            }
        }
    }
    fn save_stats(&mut self, nodes_walked: usize) {
        self.context.owner.stats.trie_nodes_walked += nodes_walked;
    }
}
impl PreparedEarleySeed {
    fn trie_controls<F: Fn(usize) -> Result<(), E>, E>(start: &[u8]) -> Option<usize> {
        let parts = [
            Self::controls::<F, E>()?,
            TokTrie::add_bias_control_bytes::<Walker<'_, F, E>>(start)?,
            size_of::<Walker<'_, F, E>>(),
            size_of::<Context<'_, F>>(),
            size_of::<shared::Snapshot>(),
            size_of::<Option<shared::Snapshot>>(),
            size_of::<Option<Cause<E>>>(),
            size_of::<Option<SimpleVob>>(),
            size_of::<SimpleVob>(),
            size_of::<TokenMaskConstructionPlan<'_>>(),
            size_of::<Result<Self, PreparedEarleySeedError<E>>>(),
            size_of::<Result<(), Cause<E>>>(),
            size_of::<Result<bool, Cause<E>>>(),
            size_of::<Result<super::super::LexerResult, Cause<E>>>(),
            size_of::<(&mut Self, &mut PreparedLexer, &TokTrie, &[u8], &F)>(),
            size_of::<(&TokTrie, &mut Walker<'_, F, E>, &mut SimpleVob, &[u8])>(),
            size_of::<(usize, bool, super::super::LexerState)>(),
            size_of::<Option<usize>>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }
    /// Walks the same actual token trie as ordinary `add_bias`, retaining the
    /// resulting byte-compatible token mask. This is the trie stage of bias
    /// computation; numeric ranges, EOS and controller policy are added by the
    /// enclosing parser driver. `start` is an already forced byte prefix.
    /// The supplied lexer and trie must belong to this chart's retained source.
    pub fn scan_token_mask<F: Fn(usize) -> Result<(), E>, E>(
        mut self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        start: &[u8],
        funding: &F,
    ) -> Result<Self, PreparedEarleySeedError<E>> {
        let result = (|| {
            funding(Self::trie_controls::<F, E>(start).ok_or(Cause::Overflow)?)
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
            self.mask_key = None;
            // Existing masks remain owned until this independently funded mask
            // has been constructed. No observation/copy budget is refunded.
            let plan = TokenMaskConstructionPlan::for_trie(trie).map_err(Cause::MaskSource)?;
            funding(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
            self.token_mask = Some(plan.compile().map_err(Cause::Mask)?);
            let mut mask = self.token_mask.take().expect("trie mask destination");
            let mut walker = Walker {
                context: Context {
                    owner: &mut self,
                    lexer,
                    trie,
                    funding,
                },
                saved: None,
                failure: None,
            };
            trie.add_bias(&mut walker, &mut mask, start);
            walker.finish();
            let failure = walker.failure.take();
            drop(walker);
            self.token_mask = Some(mask);
            match failure {
                Some(cause) => Err(cause),
                None => Ok(()),
            }
        })();
        match result {
            Ok(()) => Ok(self),
            Err(cause) => Err(PreparedEarleySeedError {
                cause,
                prefix: self,
            }),
        }
    }
    /// Borrowed byte-compatible mask from the completed actual trie traversal.
    pub fn scanned_token_mask(&self) -> Option<&SimpleVob> {
        self.token_mask.as_ref()
    }
}

impl PreparedEarleySeed {
    /// Runs the ordinary token-suffix chop and extension traversal over this
    /// chart. The source token IDs and trie remain borrowed for the complete
    /// traversal; failures retain the actual speculative chart prefix.
    pub fn chop_tokens<F: Fn(usize) -> Result<(), E>, E>(
        mut self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        tokens: &[toktrie::TokenId],
        funding: &F,
    ) -> Result<(Self, usize, usize), PreparedEarleySeedError<E>> {
        let result = (|| {
            let parts = [
                Self::controls::<F, E>().ok_or(Cause::Overflow)?,
                TokTrie::chop_tokens_control_bytes::<Walker<'_, F, E>>().ok_or(Cause::Overflow)?,
                size_of::<Walker<'_, F, E>>(),
                size_of::<Context<'_, F>>(),
                size_of::<shared::Snapshot>(),
                size_of::<Option<shared::Snapshot>>(),
                size_of::<Option<Cause<E>>>(),
                size_of::<(usize, usize)>(),
                size_of::<Result<(usize, usize), Cause<E>>>(),
                size_of::<Result<(Self, usize, usize), PreparedEarleySeedError<E>>>(),
                size_of::<Result<bool, Cause<E>>>(),
                size_of::<Result<super::super::LexerResult, Cause<E>>>(),
                size_of::<(
                    &mut Self,
                    &mut PreparedLexer,
                    &TokTrie,
                    &[toktrie::TokenId],
                    &F,
                )>(),
            ];
            funding(
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
                || tokens
                    .iter()
                    .any(|&token| token as usize >= trie.vocab_size())
            {
                return Err(Cause::Source);
            }
            trie.raw_token_bytes_len(&tokens[tokens.len().saturating_sub(4)..])
                .ok_or(Cause::Overflow)?;
            let mut walker = Walker {
                context: Context {
                    owner: &mut self,
                    lexer,
                    trie,
                    funding,
                },
                saved: None,
                failure: None,
            };
            let result = trie.chop_tokens(&mut walker, tokens);
            walker.finish();
            match walker.failure.take() {
                Some(cause) => Err(cause),
                None => Ok(result),
            }
        })();
        match result {
            Ok((tokens, bytes)) => Ok((self, tokens, bytes)),
            Err(cause) => Err(PreparedEarleySeedError {
                cause,
                prefix: self,
            }),
        }
    }
}
