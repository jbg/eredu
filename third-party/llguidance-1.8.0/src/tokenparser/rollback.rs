//! Same token-history rollback driver for ordinary and prepared sessions.
use super::TokenParser;
use crate::api::StopReason;
use std::{fmt, mem::{size_of, size_of_val}};
use toktrie::TokenId;
#[derive(Debug)]
pub(crate) enum Failure {
    Tokens { requested: usize, available: usize },
    Bytes { requested: usize, available: usize },
    Overflow,
    UnsupportedLexemes,
}
impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokens { requested, available } => write!(f, "rollback: {requested} > {available}"),
            Self::Bytes { requested, available } => write!(f, "rollback bytes: {requested} > {available}"),
            Self::Overflow => f.write_str("rollback byte count overflow"),
            Self::UnsupportedLexemes => f.write_str("rollback not supported with max_tokens=... or stop=... lexemes; suffix=... is OK"),
        }
    }
}
impl std::error::Error for Failure {}
pub(crate) trait Context {
    type Error;
    fn tokens(&self) -> &[TokenId];
    fn bytes_len(&self) -> usize;
    fn eos(&self, token: TokenId) -> bool;
    fn token_len(&self, token: TokenId) -> usize;
    fn reopen_normal_stop(&mut self);
    fn initialized(&self) -> Result<(), Self::Error>;
    fn mark_rollback(&mut self);
    fn rollback_bytes(&mut self, bytes: usize) -> Result<(), Self::Error>;
    fn finish(&mut self, tokens: usize, bytes: usize, removed: usize);
    fn failure(&self, cause: Failure) -> Self::Error;
}
pub(crate) fn controls<C: Context>() -> Option<usize> {
    let parts = [
        size_of::<&mut C>(), size_of::<Failure>(),
        size_of::<Result<(), C::Error>>(), size_of::<std::slice::Iter<'_, TokenId>>(),
        size_of::<(usize, usize, usize, usize, TokenId, bool)>(),
        size_of::<Option<usize>>(),
    ];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}
pub(crate) fn run<C: Context>(context: &mut C, removed: usize) -> Result<(), C::Error> {
    if removed == 0 { return Ok(()); }
    let len = context.tokens().len();
    if removed > len {
        return Err(context.failure(Failure::Tokens { requested: removed, available: len }));
    }
    context.reopen_normal_stop();
    context.initialized()?;
    context.mark_rollback();
    let new_len = len - removed;
    let mut bytes = 0usize;
    for &token in &context.tokens()[new_len..] {
        if !context.eos(token) {
            bytes = bytes.checked_add(context.token_len(token))
                .ok_or_else(|| context.failure(Failure::Overflow))?;
        }
    }
    if bytes > context.bytes_len() {
        return Err(context.failure(Failure::Bytes { requested: bytes, available: context.bytes_len() }));
    }
    context.rollback_bytes(bytes)?;
    context.finish(new_len, bytes, removed);
    Ok(())
}
impl Context for TokenParser {
    type Error = anyhow::Error;
    fn tokens(&self) -> &[TokenId] { &self.llm_tokens }
    fn bytes_len(&self) -> usize { self.llm_bytes.len() }
    fn eos(&self, token: TokenId) -> bool { self.eos_tokens.contains(&token) }
    fn token_len(&self, token: TokenId) -> usize { self.tok_trie().token_len(token) }
    fn reopen_normal_stop(&mut self) {
        if self.stop_reason.is_ok() { self.stop_reason = StopReason::NotStopped; }
    }
    fn initialized(&self) -> anyhow::Result<()> { self.check_initialized("rollback") }
    fn mark_rollback(&mut self) { self.had_rollback = true; }
    fn rollback_bytes(&mut self, bytes: usize) -> anyhow::Result<()> { self.parser.rollback(bytes) }
    fn finish(&mut self, tokens: usize, bytes: usize, removed: usize) {
        self.max_tokens_total = self.max_tokens_total.saturating_add(removed);
        self.llm_tokens.truncate(tokens);
        self.llm_bytes.truncate(self.llm_bytes.len() - bytes);
        self.clear_caches();
    }
    fn failure(&self, cause: Failure) -> anyhow::Error { cause.into() }
}
