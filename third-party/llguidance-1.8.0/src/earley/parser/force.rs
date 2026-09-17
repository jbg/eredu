//! Same unique-byte probe and deterministic-byte loop for both parser owners.
use super::{advance, token, LexemeSet, LexerSpec, ParserState};
use derivre::NextByte;
use toktrie::{Recognizer, TokTrie, TokenId};
pub(super) trait Context: token::Context {
    fn accepting(&mut self) -> Result<bool, Self::Error>;
    fn hint(&mut self) -> Result<NextByte, Self::Error>;
    fn probe(&mut self, hint: NextByte) -> Result<Option<u8>, Self::Error>;
    fn unique_numeric(&self) -> Option<TokenId>;
}
pub(super) fn unique<E>(
    hint: NextByte,
    mut visit: impl FnMut(u8) -> Result<bool, E>,
) -> Result<Option<u8>, E> {
    if let NextByte::SomeBytes2([a, b]) = hint {
        if visit(a)? && visit(b)? {
            return Ok(None);
        }
    }
    let first = hint.some_bytes().first().copied().unwrap_or(b' ');
    let mut byte = first;
    let mut selected = None;
    loop {
        if visit(byte)? {
            if selected.is_some() {
                return Ok(None);
            }
            selected = Some(byte);
        }
        byte = byte.wrapping_add(1);
        if byte == first {
            return Ok(selected);
        }
    }
}
pub(super) fn next<C: Context>(context: &mut C) -> Result<Option<u8>, C::Error> {
    if context.accepting()? {
        return Ok(None);
    }
    let hint = context.hint()?;
    if let NextByte::ForcedByte(byte) = hint {
        return Ok(Some(byte));
    }
    context.probe(hint)
}
pub(super) fn unique_numeric(spec: &LexerSpec, possible: &LexemeSet) -> Option<TokenId> {
    let mut selected = None;
    for lexeme in spec.iter_token_range_lexemes(possible) {
        for range in &lexeme.token_ranges {
            if range.start() != range.end() {
                return None;
            }
            let id = *range.start();
            if selected.is_some_and(|prior| prior != id) {
                return None;
            }
            selected = Some(id);
        }
    }
    selected
}
// Shared exact raw spelling also used by TokTrie::decode_raw and token chop.
pub(super) use toktrie::MarkedTokenBytes as Numeric;
pub(super) fn run<C: Context>(context: &mut C) -> Result<(), C::Error> {
    while let Some(byte) = next(context)? {
        if byte == TokTrie::SPECIAL_TOKEN_MARKER {
            assert!(!context.pending());
            let Some(token) = context.unique_numeric() else {
                break;
            };
            let spelling = Numeric::new(token);
            let mut all = true;
            for &byte in spelling.bytes() {
                let (accepted, backtrack) = advance::definitive(context, Some(byte))?;
                assert_eq!(backtrack, 0);
                if !accepted {
                    all = false;
                    break;
                }
            }
            if !all {
                break;
            }
        } else {
            let (accepted, backtrack) = advance::definitive(context, Some(byte))?;
            assert_eq!(backtrack, 0);
            if !accepted {
                break;
            }
        }
    }
    Ok(())
}
impl Context for ParserState {
    fn accepting(&mut self) -> Result<bool, Self::Error> {
        Ok(self.is_accepting())
    }
    fn hint(&mut self) -> Result<NextByte, Self::Error> {
        let state = self.lexer_state().lexer_state;
        Ok(self.lexer_mut().next_byte(state))
    }
    fn probe(&mut self, hint: NextByte) -> Result<Option<u8>, Self::Error> {
        self.run_speculative("forced_byte", |state| {
            let mut recognizer = super::ParserRecognizer { state };
            unique(hint, |byte| {
                let accepted = recognizer.try_push_byte(byte);
                if accepted {
                    recognizer.pop_bytes(1);
                }
                Ok(accepted)
            })
        })
    }
    fn unique_numeric(&self) -> Option<TokenId> {
        unique_numeric(
            self.lexer_spec(),
            self.lexer()
                .possible_lexemes(self.lexer_state().lexer_state),
        )
    }
}
