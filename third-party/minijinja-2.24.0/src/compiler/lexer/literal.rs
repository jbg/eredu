#![forbid(unsafe_code)]

use crate::compiler::tokens::Span;
use crate::error::{Error, ErrorKind};

#[derive(Clone, Copy, Debug)]
pub(crate) enum Lexeme<'s> {
    TemplateData(&'s str),
    VariableStart,
    VariableEnd,
    BlockStart,
    BlockEnd,
    Ident(&'s str),
    Str(&'s str),
    Plus,
    Minus,
    Mul,
    Div,
    FloorDiv,
    Pow,
    Mod,
    Dot,
    Comma,
    Colon,
    Tilde,
    Assign,
    Pipe,
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    BracketOpen,
    BracketClose,
    ParenOpen,
    ParenClose,
    BraceOpen,
    BraceClose,
    Escaped(&'s str),
    Number(NumberLiteral<'s>),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct NumberLiteral<'s> {
    pub raw: &'s str,
    pub radix: u32,
    pub is_float: bool,
    pub has_underscore: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Number {
    Int(u64),
    Int128(u128),
    Float(f64),
}

// Both materializers call the same standard conversions after removing
// underscores. In particular, float rounding/overflow are not reimplemented.
pub(crate) fn parse_number(raw: &str, literal: NumberLiteral<'_>) -> Result<Number, &'static str> {
    if literal.is_float {
        raw.parse().map(Number::Float).map_err(|_| "invalid float")
    } else if let Ok(n) = u64::from_str_radix(raw, literal.radix) {
        Ok(Number::Int(n))
    } else {
        u128::from_str_radix(raw, literal.radix)
            .map(Number::Int128)
            .map_err(|_| "invalid integer (too large)")
    }
}

/// A nonallocating cause. Syntax positions are extended only by the ordinary
/// Error adapter, preserving its existing overflow and filename behavior.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LexError {
    pub empty_stack: bool,
    pub kind: ErrorKind,
    pub detail: Option<&'static str>,
    pub span: Option<Span>,
}
impl LexError {
    pub fn syntax(detail: &'static str, span: Span) -> Self {
        Self {
            empty_stack: false,
            kind: ErrorKind::SyntaxError,
            detail: Some(detail),
            span: Some(span),
        }
    }
    pub fn empty_stack() -> Self {
        Self {
            empty_stack: true,
            kind: ErrorKind::SyntaxError,
            detail: None,
            span: None,
        }
    }
    pub fn ordinary(self, filename: &str) -> Error {
        if self.empty_stack {
            panic!("empty lexer stack");
        }
        let span = self.span.map(|mut span| {
            if span.start_col == span.end_col {
                span.end_col += 1;
                span.end_offset += 1;
            }
            span
        });
        let mut error = match self.detail {
            Some(detail) => Error::new(self.kind, detail),
            None => self.kind.into(),
        };
        if let Some(span) = span {
            error.set_filename_and_span(filename, span);
        }
        error
    }
}
