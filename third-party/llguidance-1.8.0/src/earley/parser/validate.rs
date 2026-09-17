//! Shared noncommitting token traversal over the actual parser history.
use super::{force::Numeric, token, ParserRecognizer, ParserState};
use toktrie::{Recognizer, TokTrie, TokenId};
pub(super) trait Context: token::Context {
    fn accepting_inner(&mut self) -> Result<bool, Self::Error>;
    fn restore_stack(&mut self, length: usize);
    fn probe(&mut self, byte: u8) -> Result<bool, Self::Error>;
}
pub(super) fn run<C: Context>(context: &mut C, tokens: &[TokenId]) -> Result<usize, C::Error> {
    let mut applied = context.applied();
    for (index, &id) in tokens.iter().enumerate() {
        if context.trie().eos_tokens().contains(&id) {
            return Ok(
                if applied == context.bytes().len() && context.accepting_inner()? {
                    index + 1
                } else {
                    index
                },
            );
        }
        if applied >= context.bytes().len() {
            let saved = context.stack_len();
            if let Some(lexeme) = token::numeric(context, id)? {
                let spelling = Numeric::new(id);
                token::add_numeric(context, lexeme, spelling.bytes())?;
                continue;
            }
            context.restore_stack(saved);
        }
        // Same raw token expansion as TokTrie::decode_raw, with the existing
        // marked numeric spelling when matching a previously forced token.
        // Borrow individual source bytes across calls instead of allocating a
        // temporary Vec that has no independent lifetime or output contract.
        let source = context.trie().token(id);
        let marked = source.is_empty()
            || source[0] == TokTrie::SPECIAL_TOKEN_MARKER
            || context.bytes().get(applied) == Some(&TokTrie::SPECIAL_TOKEN_MARKER);
        let spelling = Numeric::new(id);
        let count = if marked {
            spelling.bytes().len()
        } else {
            source.len()
        };
        for offset in 0..count {
            let byte = if marked {
                spelling.bytes()[offset]
            } else {
                context.trie().token(id)[offset]
            };
            if applied < context.bytes().len() {
                if context.bytes()[applied] != byte {
                    return Ok(index);
                }
                applied += 1;
            } else if byte == TokTrie::SPECIAL_TOKEN_MARKER || !context.probe(byte)? {
                return Ok(index);
            }
        }
    }
    Ok(tokens.len())
}
impl Context for ParserState {
    fn accepting_inner(&mut self) -> Result<bool, Self::Error> {
        Ok(self.is_accepting_inner())
    }
    fn restore_stack(&mut self, length: usize) {
        self.lexer_stack.truncate(length);
    }
    fn probe(&mut self, byte: u8) -> Result<bool, Self::Error> {
        Ok(ParserRecognizer { state: self }.try_push_byte(byte))
    }
}
