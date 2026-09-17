//! Owning no-prompt token session over the shared ordinary Matcher driver.
mod rollback;
mod copy;
use super::super::{force::Numeric, token, validate};
use super::{
    advance::Context as Chart, reserve, Cause, PreparedEarleySeed, PreparedEarleySeedError,
};
use crate::{
    api::{ParserLimits, StopReason},
    earley::PreparedLexer,
    tokenparser::progress,
};
use std::{
    fmt,
    mem::{size_of, size_of_val},
};
use toktrie::{SimpleVob, TokTrie, TokenId, TokenMaskConstructionPlan};

#[derive(Debug)]
pub(super) enum Failure {
    Stopped(StopReason),
    TokenLimit,
    InvalidToken(TokenId),
    Backtrack,
    NoExtension,
    Rollback(crate::tokenparser::rollback::Failure),
    ChartRollback(super::super::rollback::Failure),
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stopped(reason) => write!(f, "token parser is stopped ({reason:?})"),
            Self::TokenLimit => f.write_str("max_tokens_total reached"),
            Self::InvalidToken(token) => write!(f, "token id {token} out of range"),
            Self::Backtrack => f.write_str("unexpected backtracking"),
            Self::NoExtension => f.write_str("NoExtensionBias"),
            Self::Rollback(cause) => fmt::Display::fmt(cause, f),
            Self::ChartRollback(cause) => fmt::Display::fmt(cause, f),
        }
    }
}
impl std::error::Error for Failure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Rollback(cause) => Some(cause),
            Self::ChartRollback(cause) => Some(cause),
            _ => None,
        }
    }
}
#[derive(Debug)]
struct State {
    tokens: Vec<TokenId>,
    bytes: Vec<u8>,
    mask: Option<SimpleVob>,
    remaining: usize,
    stop: StopReason,
    accepting: Option<bool>,
    had_backtrack: bool,
    had_rollback: bool,
}
/// Actual mutable chart and token histories for the no-prompt Matcher contract.
/// The selected lexer and immutable trie remain with the enclosing source owner.
/// Masks consume the exact canonical forcing result from the enclosing source.
/// Snapshot copies require a separate mutable-state construction source.
#[derive(Debug)]
pub struct PreparedTokenParser {
    chart: PreparedEarleySeed,
    state: State,
}
/// A failed token operation retains the changed chart and actual token history.
/// Its enclosing source owner keeps the lexer and funding through this failure.
pub struct PreparedTokenParserError<E> {
    cause: PreparedEarleySeedError<E>,
    state: State,
}
impl<E: fmt::Debug> fmt::Debug for PreparedTokenParserError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedTokenParserError")
            .field("cause", &self.cause)
            .field("tokens", &self.state.tokens.len())
            .field("bytes", &self.state.bytes.len())
            .finish()
    }
}
impl<E: fmt::Display> fmt::Display for PreparedTokenParserError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.cause, f)
    }
}
impl<E: std::error::Error + 'static> std::error::Error for PreparedTokenParserError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
struct Context<'a, 'b, F> {
    chart: &'a mut Chart<'b, F>,
    state: &'a mut State,
}
impl<F: Fn(usize) -> Result<(), E>, E> Context<'_, '_, F> {
    fn append_token(&mut self, token: TokenId) -> Result<(), Cause<E>> {
        let total = self
            .state
            .tokens
            .len()
            .checked_add(1)
            .ok_or(Cause::Overflow)?;
        reserve(&mut self.state.tokens, total, self.chart.funding)?;
        self.state.tokens.push(token);
        Ok(())
    }
}
impl<F: Fn(usize) -> Result<(), E>, E> progress::Context for Context<'_, '_, F> {
    type Error = Cause<E>;
    fn initialized(&self, _operation: &'static str) -> Result<(), Self::Error> {
        if self.state.stop == StopReason::NotStopped {
            Ok(())
        } else {
            Err(Cause::Session(Failure::Stopped(self.state.stop)))
        }
    }
    fn take_token(&mut self) -> Result<(), Self::Error> {
        if self.state.remaining == 0 {
            self.state.stop = StopReason::MaxTokensTotal;
            return Err(Cause::Session(Failure::TokenLimit));
        }
        self.state.remaining -= 1;
        self.state.mask = None;
        Ok(())
    }
    fn eos(&self, token: TokenId) -> bool {
        self.chart.trie.eos_tokens().contains(&token)
    }
    fn scan_eos(&mut self) -> Result<bool, Self::Error> {
        token::eos(self.chart)
    }
    fn accepting(&mut self) -> Result<bool, Self::Error> {
        if let Some(value) = self.state.accepting {
            return Ok(value);
        }
        let value = self.chart.owner.pending_token_bytes().is_empty()
            && self
                .chart
                .speculative(|chart| validate::Context::accepting_inner(chart))?;
        self.state.accepting = Some(value);
        Ok(value)
    }
    fn record_eos(&mut self, token: TokenId) -> Result<(), Self::Error> {
        self.append_token(token)
    }
    fn apply(&mut self, token: TokenId) -> Result<usize, Self::Error> {
        self.state.accepting = None;
        let trie = self.chart.trie;
        if token as usize >= trie.vocab_size() {
            self.state.stop = StopReason::InternalError;
            return Err(Cause::Session(Failure::InvalidToken(token)));
        }
        self.append_token(token)?;
        let source = trie.token(token);
        let spelling = Numeric::new(token);
        let bytes = if source.is_empty() || source[0] == TokTrie::SPECIAL_TOKEN_MARKER {
            spelling.bytes()
        } else {
            source
        };
        let next = self
            .chart
            .owner
            .token_idx
            .checked_add(1)
            .ok_or(Cause::Overflow)?;
        u32::try_from(next).map_err(|_| Cause::Overflow)?;
        let result = token::apply(self.chart, bytes, token);
        self.chart.owner.token_idx = next;
        let backtrack = result?;
        let total = self
            .state
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or(Cause::Overflow)?;
        reserve(&mut self.state.bytes, total, self.chart.funding)?;
        self.state.bytes.extend_from_slice(bytes);
        if backtrack != 0 {
            self.state.had_backtrack = true;
            let span =
                progress::backtrack(trie, &self.state.tokens, backtrack).ok_or(Cause::Source)?;
            let bytes = self
                .state
                .bytes
                .len()
                .checked_sub(span.full_bytes)
                .ok_or(Cause::Source)?;
            let tokens = self
                .state
                .tokens
                .len()
                .checked_sub(span.tokens)
                .ok_or(Cause::Source)?;
            self.state.bytes.truncate(bytes);
            self.state.tokens.truncate(tokens);
            // Matcher has no backtracking capability: ordinary history is
            // removed, but zero is reported to the enclosing generation loop.
        }
        Ok(0)
    }
    fn pending_eos(&self) -> bool {
        self.state
            .tokens
            .last()
            .is_some_and(|token| self.chart.trie.eos_tokens().contains(token))
    }
    fn can_advance(&self) -> bool {
        self.chart
            .owner
            .can_advance()
            .expect("published token chart")
    }
    fn stop(&mut self, reason: StopReason) {
        self.state.stop = reason;
    }
    fn validate(&mut self, token: TokenId) -> Result<bool, Self::Error> {
        if self.state.stop != StopReason::NotStopped {
            return Ok(false);
        }
        if token as usize >= self.chart.trie.vocab_size() {
            self.state.stop = StopReason::InternalError;
            return Err(Cause::Session(Failure::InvalidToken(token)));
        }
        self.chart.speculative(|chart| {
            chart
                .owner
                .scratch
                .as_mut()
                .expect("token validation scratch")
                .log_override = true;
            validate::run(chart, &[token]).map(|count| count > 0)
        })
    }
    fn unexpected_backtrack(&self) -> Self::Error {
        Cause::Session(Failure::Backtrack)
    }
}
impl PreparedTokenParser {
    fn operation<T, F, E, G>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        funding: &F,
        run: G,
    ) -> Result<(Self, T), PreparedTokenParserError<E>>
    where
        F: Fn(usize) -> Result<(), E>,
        G: FnOnce(&mut Context<'_, '_, F>) -> Result<T, Cause<E>>,
    {
        let Self { chart, mut state } = self;
        let result = chart.token_operation(lexer, trie, limits, false, funding, |chart| {
            let parts = [
                size_of::<Self>(),
                size_of::<PreparedTokenParserError<E>>(),
                size_of::<State>(),
                size_of::<Context<'_, '_, F>>(),
                size_of::<G>(),
                size_of::<T>(),
                size_of::<Failure>(),
                size_of::<Numeric>(),
                size_of::<progress::Backtrack>(),
                size_of::<Option<progress::Backtrack>>(),
                size_of::<(&TokTrie, &[TokenId], usize)>(),
                size_of::<(usize, usize, usize, isize, TokenId, bool, bool, bool)>(),
                size_of::<Result<(Self, T), PreparedTokenParserError<E>>>(),
                size_of::<Result<T, Cause<E>>>(),
                size_of::<Result<usize, Cause<E>>>(),
                size_of::<Result<bool, Cause<E>>>(),
                size_of::<Result<(), Cause<E>>>(),
                size_of::<Result<(), E>>(),
                size_of::<std::iter::Enumerate<std::slice::Iter<'_, TokenId>>>(),
                size_of::<std::ops::Range<usize>>(),
            ];
            funding(
                parts
                    .into_iter()
                    .try_fold(size_of_val(&parts), usize::checked_add)
                    .ok_or(Cause::Overflow)?,
            )
            .map_err(Cause::Funding)?;
            run(&mut Context {
                chart,
                state: &mut state,
            })
        });
        match result {
            Ok((chart, value)) => Ok((Self { chart, state }, value)),
            Err(cause) => Err(PreparedTokenParserError { cause, state }),
        }
    }
    fn map_chart<T, F, E, G>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        funding: &F,
        run: G,
    ) -> Result<(Self, T), PreparedTokenParserError<E>>
    where
        F: Fn(usize) -> Result<(), E>,
        G: FnOnce(
            PreparedEarleySeed,
            &mut PreparedLexer,
            &TokTrie,
            &ParserLimits,
            &F,
        ) -> Result<(PreparedEarleySeed, T), PreparedEarleySeedError<E>>,
    {
        let Self { chart, state } = self;
        let parts = [
            size_of::<Self>(),
            size_of::<PreparedTokenParserError<E>>(),
            size_of::<State>(),
            size_of::<G>(),
            size_of::<T>(),
            size_of::<Result<(Self, T), PreparedTokenParserError<E>>>(),
            size_of::<Result<(PreparedEarleySeed, T), PreparedEarleySeedError<E>>>(),
            size_of::<Result<(), E>>(),
            size_of::<Cause<E>>(),
            size_of::<(&mut PreparedLexer, &TokTrie, &ParserLimits, &F)>(),
        ];
        let controls = parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add);
        let cause = match controls {
            Some(bytes) => funding(bytes).err().map(Cause::Funding),
            None => Some(Cause::Overflow),
        };
        if let Some(cause) = cause {
            return Err(PreparedTokenParserError {
                cause: PreparedEarleySeedError {
                    cause,
                    prefix: chart,
                },
                state,
            });
        }
        match run(chart, lexer, trie, limits, funding) {
            Ok((chart, value)) => Ok((Self { chart, state }, value)),
            Err(cause) => Err(PreparedTokenParserError { cause, state }),
        }
    }
    /// Commits deterministic bytes using the same underlying force worker,
    /// retaining token history separately until those bytes are consumed.
    pub fn force_bytes<F: Fn(usize) -> Result<(), E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        funding: &F,
    ) -> Result<Self, PreparedTokenParserError<E>> {
        self.map_chart(
            lexer,
            trie,
            limits,
            funding,
            |chart, lexer, trie, limits, funding| {
                chart
                    .force_bytes(lexer, trie, limits, funding)
                    .map(|chart| (chart, ()))
            },
        )
        .map(|(owner, ())| owner)
    }
    /// Uses the actual speculative token-suffix chop without committing tokens.
    pub fn chop_tokens<F: Fn(usize) -> Result<(), E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        tokens: &[TokenId],
        funding: &F,
    ) -> Result<(Self, usize, usize), PreparedTokenParserError<E>> {
        self.map_chart(
            lexer,
            trie,
            limits,
            funding,
            |chart, lexer, trie, _limits, funding| {
                chart
                    .chop_tokens(lexer, trie, tokens, funding)
                    .map(|(chart, count, bytes)| (chart, (count, bytes)))
            },
        )
        .map(|(owner, (count, bytes))| (owner, count, bytes))
    }
    /// Publishes the complete no-prompt token-session mask. `first_forced` and
    /// `prefix` are the enclosing exact canonical forcing driver's result.
    /// Stopped sessions produce EOS aliases; live forced tokens produce a
    /// singleton, otherwise the actual Earley mask is copied then finalized.
    pub fn compute_mask<F: Fn(usize) -> Result<(), E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        first_forced: Option<TokenId>,
        prefix: &[u8],
        funding: &F,
    ) -> Result<Self, PreparedTokenParserError<E>> {
        let stopped = self.state.stop != StopReason::NotStopped;
        let owner = if !stopped && first_forced.is_none() {
            self.map_chart(
                lexer,
                trie,
                limits,
                funding,
                |chart, lexer, trie, limits, funding| {
                    chart
                        .compute_token_mask(lexer, trie, limits, prefix, funding)
                        .map(|chart| (chart, ()))
                },
            )?
            .0
        } else {
            self
        };
        owner
            .operation(lexer, trie, limits, funding, |context| {
                let parts = [
                    TokenMaskConstructionPlan::inspection_control_bytes().ok_or(Cause::Overflow)?,
                    size_of::<TokenMaskConstructionPlan<'_>>(),
                    size_of::<SimpleVob>(),
                    size_of::<Option<SimpleVob>>(),
                    size_of::<Result<TokenMaskConstructionPlan<'_>, toktrie::TokenMaskSourceError>>(
                    ),
                    size_of::<Result<SimpleVob, toktrie::TokenMaskConstructionFailure>>(),
                    size_of::<Result<(), Cause<E>>>(),
                    size_of::<Option<TokenId>>(),
                    size_of::<std::slice::Iter<'_, TokenId>>(),
                    size_of::<(bool, bool, TokenId)>(),
                ];
                funding(
                    parts
                        .into_iter()
                        .try_fold(size_of_val(&parts), usize::checked_add)
                        .ok_or(Cause::Overflow)?,
                )
                .map_err(Cause::Funding)?;
                if !stopped && first_forced.is_some_and(|token| token as usize >= trie.vocab_size())
                {
                    return Err(Cause::Source);
                }
                let accepting = if stopped {
                    true
                } else if first_forced.is_some() {
                    false
                } else {
                    progress::Context::accepting(context)?
                };
                let plan = if stopped || first_forced.is_some() {
                    TokenMaskConstructionPlan::for_trie(trie)
                } else {
                    TokenMaskConstructionPlan::copy(
                        context
                            .chart
                            .owner
                            .scanned_token_mask()
                            .ok_or(Cause::Source)?,
                    )
                }
                .map_err(Cause::MaskSource)?;
                funding(plan.requirements().required_bytes()).map_err(Cause::Funding)?;
                let mut mask = plan.compile().map_err(Cause::Mask)?;
                if !stopped {
                    if let Some(token) = first_forced {
                        mask.allow_token(token);
                    }
                }
                progress::finish_mask(&mut mask, trie.eos_tokens(), accepting);
                let empty = mask.is_zero();
                context.state.mask = Some(mask);
                if empty {
                    context.state.stop = StopReason::NoExtensionBias;
                    return Err(Cause::Session(Failure::NoExtension));
                }
                Ok(())
            })
            .map(|(owner, ())| owner)
    }
    /// Completed independent token-session mask, distinct from its Earley cache.
    /// A committed token invalidates this borrowed publication before mutation.
    pub fn token_mask(&self) -> Option<&SimpleVob> {
        self.state.mask.as_ref()
    }
    /// Starts the ordinary no-prompt, no-backtracking Matcher state from an
    /// actual untouched initial chart. `max_tokens` is the exact grammar
    /// request's token limit, not a storage allowance or grammar capability.
    pub fn prepare<F: Fn(usize) -> Result<(), E>, E>(
        chart: PreparedEarleySeed,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        max_tokens: Option<usize>,
        funding: &F,
    ) -> Result<Self, PreparedTokenParserError<E>> {
        let owner = Self {
            chart,
            state: State {
                tokens: Vec::new(),
                bytes: Vec::new(),
                mask: None,
                remaining: max_tokens.unwrap_or(usize::MAX),
                stop: StopReason::NotStopped,
                accepting: None,
                had_backtrack: false,
                had_rollback: false,
            },
        };
        owner
            .operation(lexer, trie, limits, funding, |context| {
                if !context.chart.owner.bytes.is_empty()
                    || !context.chart.owner.byte_to_token_idx.is_empty()
                    || context.chart.owner.token_idx != 0
                {
                    return Err(Cause::Source);
                }
                Ok(())
            })
            .map(|(owner, ())| owner)
    }
    /// Uses the same Matcher validation/consume/stop loop, returning the first
    /// invalid token's index without committing that token.
    pub fn try_consume_tokens<F: Fn(usize) -> Result<(), E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        tokens: &[TokenId],
        funding: &F,
    ) -> Result<(Self, usize), PreparedTokenParserError<E>> {
        self.operation(lexer, trie, limits, funding, |context| {
            progress::try_consume(context, tokens)
        })
    }
    /// Same token-level acceptance predicate, excluding unapplied forced bytes.
    pub fn is_accepting<F: Fn(usize) -> Result<(), E>, E>(
        self,
        lexer: &mut PreparedLexer,
        trie: &TokTrie,
        limits: &ParserLimits,
        funding: &F,
    ) -> Result<(Self, bool), PreparedTokenParserError<E>> {
        self.operation(lexer, trie, limits, funding, |context| {
            progress::Context::accepting(context)
        })
    }
    /// Exact committed parser and capture owner.
    pub fn chart(&self) -> &PreparedEarleySeed {
        &self.chart
    }
    /// Token IDs retained by the actual no-prompt session, including EOS.
    pub fn tokens(&self) -> &[TokenId] {
        &self.state.tokens
    }
    /// Actual visible raw session bytes after any hidden-stop backtrack.
    pub fn bytes(&self) -> &[u8] {
        &self.state.bytes
    }
    /// Actual terminal reason from the shared consume/check-stop driver.
    pub fn stop_reason(&self) -> StopReason {
        self.state.stop
    }
}
