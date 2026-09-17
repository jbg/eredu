//! Shared token-session consumption and terminal decisions.
use crate::{api::StopReason, TokenParser};
use toktrie::{TokTrie, TokenId};
pub(crate) trait Context {
    type Error;
    fn initialized(&self, operation: &'static str) -> Result<(), Self::Error>;
    fn take_token(&mut self) -> Result<(), Self::Error>;
    fn eos(&self, token: TokenId) -> bool;
    fn scan_eos(&mut self) -> Result<bool, Self::Error>;
    fn accepting(&mut self) -> Result<bool, Self::Error>;
    fn record_eos(&mut self, token: TokenId) -> Result<(), Self::Error>;
    fn apply(&mut self, token: TokenId) -> Result<usize, Self::Error>;
    fn pending_eos(&self) -> bool;
    fn can_advance(&self) -> bool;
    fn stop(&mut self, reason: StopReason);
    fn validate(&mut self, token: TokenId) -> Result<bool, Self::Error>;
    fn unexpected_backtrack(&self) -> Self::Error;
    fn observe_terminal(&mut self, _accepting: bool, _pending_eos: bool, _can_advance: bool) {}
}
pub(crate) fn consume<C: Context>(context: &mut C, token: TokenId) -> Result<usize, C::Error> {
    context.initialized("consume_token")?;
    context.take_token()?;
    if context.eos(token) {
        if context.scan_eos()? {
            return Ok(0);
        }
        if context.accepting()? {
            context.record_eos(token)?;
            return Ok(0);
        }
    }
    context.apply(token)
}
pub(crate) fn check_stop<C: Context>(context: &mut C) -> Result<bool, C::Error> {
    let pending_eos = context.pending_eos();
    let accepting = context.accepting()?;
    let can_advance = context.can_advance();
    context.observe_terminal(accepting, pending_eos, can_advance);
    if accepting && (!can_advance || pending_eos) {
        context.stop(if pending_eos {
            StopReason::EndOfSentence
        } else {
            StopReason::NoExtension
        });
        Ok(true)
    } else {
        Ok(false)
    }
}
pub(crate) fn try_consume<C: Context>(
    context: &mut C,
    tokens: &[TokenId],
) -> Result<usize, C::Error> {
    for (index, &token) in tokens.iter().enumerate() {
        if !context.validate(token)? {
            return Ok(index);
        }
        let backtrack = consume(context, token)?;
        check_stop(context)?;
        if backtrack != 0 {
            return Err(context.unexpected_backtrack());
        }
    }
    Ok(tokens.len())
}
#[derive(Clone, Copy)]
pub(crate) struct Backtrack {
    pub(crate) tokens: usize,
    pub(crate) additional_bytes: usize,
    pub(crate) full_bytes: usize,
}
pub(crate) fn backtrack(trie: &TokTrie, tokens: &[TokenId], bytes: usize) -> Option<Backtrack> {
    let mut remaining = isize::try_from(bytes).ok()?;
    let mut count = 0;
    while remaining > 0 {
        let offset = tokens.len().checked_sub(count)?;
        if offset == 0 {
            break;
        }
        remaining =
            remaining.checked_sub(isize::try_from(trie.token_len(tokens[offset - 1])).ok()?)?;
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let additional = usize::try_from(remaining.checked_neg()?).ok()?;
    Some(Backtrack {
        tokens: count,
        additional_bytes: additional,
        full_bytes: bytes.checked_add(additional)?,
    })
}
impl Context for TokenParser {
    type Error = anyhow::Error;
    fn initialized(&self, operation: &'static str) -> anyhow::Result<()> {
        self.check_initialized(operation)
    }
    fn take_token(&mut self) -> anyhow::Result<()> {
        if self.max_tokens_total == 0 {
            return Err(self.stop("max_tokens_total reached", StopReason::MaxTokensTotal));
        }
        self.max_tokens_total -= 1;
        Ok(())
    }
    fn eos(&self, token: TokenId) -> bool {
        self.eos_tokens.contains(&token)
    }
    fn scan_eos(&mut self) -> anyhow::Result<bool> {
        Ok(self.parser.scan_eos())
    }
    fn accepting(&mut self) -> anyhow::Result<bool> {
        Ok(self.is_accepting())
    }
    fn record_eos(&mut self, token: TokenId) -> anyhow::Result<()> {
        self.llm_tokens.push(token);
        Ok(())
    }
    fn apply(&mut self, token: TokenId) -> anyhow::Result<usize> {
        let result = self.apply_token(token);
        self.parser.log_row_infos("post-apply");
        result.map_err(|_| self.anyhow_error())
    }
    fn pending_eos(&self) -> bool {
        self.llm_tokens
            .last()
            .is_some_and(|token| self.eos_tokens.contains(token))
    }
    fn can_advance(&self) -> bool {
        self.parser.can_advance()
    }
    fn stop(&mut self, reason: StopReason) {
        self.stop("", reason);
    }
    fn validate(&mut self, token: TokenId) -> anyhow::Result<bool> {
        self.validate_token(token)
    }
    fn unexpected_backtrack(&self) -> anyhow::Error {
        anyhow::anyhow!("unexpected backtracking")
    }
    fn observe_terminal(&mut self, accepting: bool, pending_eos: bool, can_advance: bool) {
        let empty_token_prefix = !self.has_ff_bytes();
        let lexer_bytes = self.parser.has_pending_lexeme_bytes();
        let parser_done = accepting && (!can_advance || pending_eos);
        crate::infoln!(self, "parser_done: {parser_done}; lexer_bytes: {lexer_bytes}; can_advance: {can_advance} (eos:{pending_eos}); accept: {accepting}; empty_token_prefix: {empty_token_prefix}");
        assert!(!accepting || empty_token_prefix);
    }
}

// Same final token-session EOS aliases, independent of the Earley bias cache.
pub(crate) fn finish_mask(mask: &mut toktrie::SimpleVob, eos: &[TokenId], accepting: bool) {
    if accepting {
        for &token in eos {
            if token != toktrie::INVALID_TOKEN {
                mask.allow_token(token);
            }
        }
    }
}
