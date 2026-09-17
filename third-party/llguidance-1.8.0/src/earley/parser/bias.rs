//! Shared final mask rules and exact current-state cache identity.
use super::{token, LexerState, ParserState};
use token::Context as _;
use toktrie::{SimpleVob, TokTrie, TokenId, INVALID_TOKEN};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Key {
    state: derivre::StateID,
    row: u32,
    pending: bool,
}
impl Key {
    pub(super) fn new(state: LexerState, pending: bool) -> Self {
        Self {
            state: state.lexer_state,
            row: state.row_idx,
            pending,
        }
    }
    pub(super) fn matches(self, state: derivre::StateID, row: u32, pending: bool) -> bool {
        self.state == state && self.row == row && self.pending == pending
    }
}
pub(super) fn marker_token(trie: &TokTrie) -> TokenId {
    let byte = [TokTrie::SPECIAL_TOKEN_MARKER];
    let mut tokens = trie.greedy_tokens(&byte);
    let first = tokens.next().unwrap_or(INVALID_TOKEN);
    if tokens.next().is_none() {
        first
    } else {
        INVALID_TOKEN
    }
}
pub(super) trait Context {
    type Error;
    fn marker(&self) -> TokenId;
    fn eos(&self) -> TokenId;
    fn numeric_ranges(&mut self, mask: &mut SimpleVob) -> Result<(), Self::Error>;
    fn allows_eos(&self) -> bool;
}
pub(super) fn finish<C: Context>(
    context: &mut C,
    mask: &mut SimpleVob,
    start: &[u8],
) -> Result<(), C::Error> {
    let marker = context.marker();
    if marker != INVALID_TOKEN {
        mask.disallow_token(marker);
    }
    if start.is_empty() {
        context.numeric_ranges(mask)?;
    }
    let eos = context.eos();
    if eos != INVALID_TOKEN && start.is_empty() && context.allows_eos() {
        mask.allow_token(eos);
    }
    Ok(())
}
pub(super) fn ranges(spec: &super::LexerSpec, possible: &super::LexemeSet, mask: &mut SimpleVob) {
    for lexeme in spec.iter_token_range_lexemes(possible) {
        for range in &lexeme.token_ranges {
            mask.allow_range(range.clone());
        }
    }
}
impl Context for ParserState {
    type Error = anyhow::Error;
    fn marker(&self) -> TokenId {
        self.special_token_marker_token
    }
    fn eos(&self) -> TokenId {
        self.tok_env.tok_trie().eos_token()
    }
    fn numeric_ranges(&mut self, mask: &mut SimpleVob) -> Result<(), Self::Error> {
        self.run_speculative("token_ranges", |state| {
            if token::flush(state)? {
                ranges(
                    state.lexer_spec(),
                    state
                        .lexer()
                        .possible_lexemes(state.lexer_state().lexer_state),
                    mask,
                );
            }
            Ok(())
        })
    }
    fn allows_eos(&self) -> bool {
        token::Context::allows_eos(self)
    }
}
