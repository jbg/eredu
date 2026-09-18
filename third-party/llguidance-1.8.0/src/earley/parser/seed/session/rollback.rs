//! Same ordinary rollback worker over the paid prepared chart/history.
use super::{Cause, Context, Failure, PreparedTokenParser, PreparedTokenParserError};
use super::super::super::rollback as chart;
use crate::{api::{ParserLimits, StopReason}, earley::PreparedLexer, tokenparser::{progress, rollback}};
use std::mem::{size_of, size_of_val};
use toktrie::{TokTrie, TokenId};
impl<F: crate::earley::PreparedFunding<Error = E>, E> rollback::Context for Context<'_, '_, F> {
    type Error = Cause<E>;
    fn tokens(&self) -> &[TokenId] { &self.state.tokens }
    fn bytes_len(&self) -> usize { self.state.bytes.len() }
    fn eos(&self, token: TokenId) -> bool { self.chart.trie.eos_tokens().contains(&token) }
    fn token_len(&self, token: TokenId) -> usize { self.chart.trie.token_len(token) }
    fn reopen_normal_stop(&mut self) {
        if self.state.stop.is_ok() { self.state.stop = StopReason::NotStopped; }
    }
    fn initialized(&self) -> Result<(), Self::Error> {
        progress::Context::initialized(self, "rollback")
    }
    fn mark_rollback(&mut self) { self.state.had_rollback = true; }
    fn rollback_bytes(&mut self, removed: usize) -> Result<(), Self::Error> {
        let owner = &mut self.chart.owner;
        if !owner.grammar.lexer_spec().can_rollback() {
            return Err(Cause::Session(Failure::Rollback(rollback::Failure::UnsupportedLexemes)));
        }
        chart::run(chart::Fields {
            byte_to_token_idx: &mut owner.byte_to_token_idx,
            bytes: &mut owner.bytes,
            lexer_stack: &mut owner.lexer_stack,
            row_infos: &mut owner.row_infos,
            token_idx: &mut owner.token_idx,
            last_force_bytes_len: &mut owner.last_force_bytes_len,
            lexer_stack_top_eos: &mut owner.lexer_stack_top_eos,
            rows_valid_end: &mut owner.rows_valid_end,
        }, removed).map_err(|cause| Cause::Session(Failure::ChartRollback(cause)))
    }
    fn finish(&mut self, tokens: usize, bytes: usize, removed: usize) {
        self.state.remaining = self.state.remaining.saturating_add(removed);
        self.state.tokens.truncate(tokens);
        self.state.bytes.truncate(self.state.bytes.len() - bytes);
        self.state.accepting = None;
        self.state.mask = None;
    }
    fn failure(&self, cause: rollback::Failure) -> Self::Error {
        Cause::Session(Failure::Rollback(cause))
    }
}
impl PreparedTokenParser {
    /// Rolls back committed tokens through the same ordinary history/chart
    /// workers. Normal terminal states reopen; failures retain the reached state.
    /// Token limits return by the ordinary rule; funding is never refunded.
    pub fn rollback<F: crate::earley::PreparedFunding<Error = E>, E>(
        self, lexer: &mut PreparedLexer, trie: &TokTrie, limits: &ParserLimits,
        tokens: usize, funding: &F,
    ) -> Result<Self, PreparedTokenParserError<E>> {
        self.operation(lexer, trie, limits, funding, |context| {
            let parts = [
                rollback::controls::<Context<'_, '_, F>>().ok_or(Cause::Overflow)?,
                chart::controls().ok_or(Cause::Overflow)?,
                size_of::<Self>(),
                size_of::<Result<Self, PreparedTokenParserError<E>>>(),
                size_of::<rollback::Failure>(), size_of::<chart::Failure>(),
                size_of::<Result<(), chart::Failure>>(), size_of::<Result<(), Cause<E>>>(),
                size_of::<(&mut Context<'_, '_, F>, usize)>(),
                size_of::<(usize, usize)>(),
            ];
            funding.reserve(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
                .ok_or(Cause::Overflow)?).map_err(Cause::Funding)?;
            rollback::run(context, tokens)
        }).map(|(owner, ())| owner)
    }
    /// Resets all committed token history using the same rollback operation.
    pub fn reset<F: crate::earley::PreparedFunding<Error = E>, E>(
        self, lexer: &mut PreparedLexer, trie: &TokTrie, limits: &ParserLimits, funding: &F,
    ) -> Result<Self, PreparedTokenParserError<E>> {
        let tokens = self.state.tokens.len();
        self.rollback(lexer, trie, limits, tokens, funding)
    }
}
