#![forbid(unsafe_code)]
use std::borrow::Cow;

use crate::compiler::tokens::{Span, Token};
use crate::error::Error;
use crate::syntax::SyntaxConfig;
use crate::utils::unescape;

pub(crate) mod literal;
pub(crate) mod scanner;
use literal::{parse_number, Lexeme, Number};
use scanner::{Scanner, Syntax};

/// Internal config struct to control whitespace in the engine.
#[derive(Copy, Clone, Debug, Default)]
pub struct WhitespaceConfig {
    pub keep_trailing_newline: bool,
    pub lstrip_blocks: bool,
    pub trim_blocks: bool,
}

/// Tokenizes jinja templates through the shared scanner and ordinary literal storage.
pub struct Tokenizer<'s> {
    scanner: Scanner<'s>,
}
impl<'s> Tokenizer<'s> {
    /// Creates a new tokenizer.
    pub fn new(
        input: &'s str,
        filename: &'s str,
        in_expr: bool,
        syntax_config: SyntaxConfig,
        whitespace_config: WhitespaceConfig,
    ) -> Self {
        Self {
            scanner: Scanner::new(
                input,
                filename,
                in_expr,
                Syntax::Owned(syntax_config),
                whitespace_config,
            ),
        }
    }
    /// Returns the current filename.
    pub fn filename(&self) -> &str {
        self.scanner.filename()
    }
    /// Returns the source.
    pub fn source(&self) -> &'s str {
        self.scanner.source()
    }
    /// Produces the next token using the ordinary owned Token representation.
    pub fn next_token(&mut self) -> Result<Option<(Token<'s>, Span)>, Error> {
        let Some((token, span)) = self
            .scanner
            .next_token()
            .map_err(|error| error.ordinary(self.filename()))?
        else {
            return Ok(None);
        };
        let token = match token {
            Lexeme::TemplateData(x) => Token::TemplateData(x),
            Lexeme::VariableStart => Token::VariableStart,
            Lexeme::VariableEnd => Token::VariableEnd,
            Lexeme::BlockStart => Token::BlockStart,
            Lexeme::BlockEnd => Token::BlockEnd,
            Lexeme::Ident(x) => Token::Ident(x),
            Lexeme::Str(x) => Token::Str(x),
            Lexeme::Plus => Token::Plus,
            Lexeme::Minus => Token::Minus,
            Lexeme::Mul => Token::Mul,
            Lexeme::Div => Token::Div,
            Lexeme::FloorDiv => Token::FloorDiv,
            Lexeme::Pow => Token::Pow,
            Lexeme::Mod => Token::Mod,
            Lexeme::Dot => Token::Dot,
            Lexeme::Comma => Token::Comma,
            Lexeme::Colon => Token::Colon,
            Lexeme::Tilde => Token::Tilde,
            Lexeme::Assign => Token::Assign,
            Lexeme::Pipe => Token::Pipe,
            Lexeme::Eq => Token::Eq,
            Lexeme::Ne => Token::Ne,
            Lexeme::Gt => Token::Gt,
            Lexeme::Gte => Token::Gte,
            Lexeme::Lt => Token::Lt,
            Lexeme::Lte => Token::Lte,
            Lexeme::BracketOpen => Token::BracketOpen,
            Lexeme::BracketClose => Token::BracketClose,
            Lexeme::ParenOpen => Token::ParenOpen,
            Lexeme::ParenClose => Token::ParenClose,
            Lexeme::BraceOpen => Token::BraceOpen,
            Lexeme::BraceClose => Token::BraceClose,
            Lexeme::Escaped(raw) => Token::String(unescape(raw)?.into_boxed_str()),
            Lexeme::Number(literal) => {
                let raw = if literal.has_underscore {
                    Cow::Owned(literal.raw.replace('_', ""))
                } else {
                    Cow::Borrowed(literal.raw)
                };
                match parse_number(&raw, literal)
                    .map_err(|detail| self.scanner.syntax_error(detail).ordinary(self.filename()))?
                {
                    Number::Int(n) => Token::Int(n),
                    Number::Int128(n) => Token::Int128(Box::new(n)),
                    Number::Float(n) => Token::Float(n),
                }
            }
        };
        Ok(Some((token, span)))
    }
}

/// Utility function to quickly tokenize into an iterator.
#[cfg(any(test, feature = "unstable_machinery"))]
pub fn tokenize(
    input: &str,
    in_expr: bool,
    syntax_config: SyntaxConfig,
    whitespace_config: WhitespaceConfig,
) -> impl Iterator<Item = Result<(Token<'_>, Span), Error>> {
    // This function is unused in minijinja itself, it's only used in tests and in the
    // unstable machinery as a convenient alternative to the tokenizer.
    let mut tokenizer =
        Tokenizer::new(input, "<string>", in_expr, syntax_config, whitespace_config);
    std::iter::from_fn(move || tokenizer.next_token().transpose())
}

#[cfg(test)]
mod tests {
    use super::scanner::{skip_basic_tag, Whitespace};
    use super::*;

    use similar_asserts::assert_eq;

    #[test]
    fn test_is_basic_tag() {
        assert_eq!(
            skip_basic_tag(" raw %}", "raw", "%}", false),
            Some((7, Whitespace::Default))
        );
        assert_eq!(skip_basic_tag(" raw %}", "endraw", "%}", false), None);
        assert_eq!(
            skip_basic_tag("  raw  %}", "raw", "%}", false),
            Some((9, Whitespace::Default))
        );
        assert_eq!(
            skip_basic_tag("  raw  -%}", "raw", "%}", false),
            Some((10, Whitespace::Remove))
        );
        assert_eq!(
            skip_basic_tag("  raw  +%}", "raw", "%}", false),
            Some((10, Whitespace::Preserve))
        );
    }

    #[test]
    fn test_basic_identifiers() {
        fn assert_ident(s: &str) {
            match tokenize(s, true, Default::default(), Default::default()).next() {
                Some(Ok((Token::Ident(ident), _))) if ident == s => {}
                _ => panic!("did not get a matching token result: {s:?}"),
            }
        }

        fn assert_not_ident(s: &str) {
            let res = tokenize(s, true, Default::default(), Default::default())
                .collect::<Result<Vec<_>, _>>();
            if let Ok(tokens) = res {
                if let &[(Token::Ident(_), _)] = &tokens[..] {
                    panic!("got a single ident for {s:?}")
                }
            }
        }

        assert_ident("foo_bar_baz");
        assert_ident("_foo_bar_baz");
        assert_ident("_42world");
        assert_ident("_world42");
        assert_ident("world42");
        assert_not_ident("42world");

        #[cfg(feature = "unicode")]
        {
            assert_ident("foo");
            assert_ident("föö");
            assert_ident("き");
            assert_ident("_");
            assert_not_ident("1a");
            assert_not_ident("a-");
            assert_not_ident("🐍a");
            assert_not_ident("a🐍🐍");
            assert_ident("ᢅ");
            assert_ident("ᢆ");
            assert_ident("℘");
            assert_ident("℮");
            assert_not_ident("·");
            assert_ident("a·");
        }
    }
}
