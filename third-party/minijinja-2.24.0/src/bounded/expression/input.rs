//! Lazy lookahead, with every real fetch using the shared scanner/literal worker.
#![forbid(unsafe_code)]

use super::store::Bytes;
use super::{Buffer, Cause, PlanError};
use crate::compiler::lexer::literal::{parse_number, LexError, Lexeme, Number};
use crate::compiler::lexer::scanner::{Scanner, Syntax};
use crate::compiler::lexer::WhitespaceConfig;
use crate::compiler::parser::storage::{Input, Item, Kind, TextRef};
use crate::compiler::tokens::Span;
use crate::utils::literal::{decode, Failure, Output};
use crate::ErrorKind;
use std::mem;

pub(in crate::bounded) fn inspect(
    source: &str,
    filename: &str,
) -> Result<(usize, usize, usize), PlanError> {
    inspect_config(source, filename, true, WhitespaceConfig::default())
}
pub(in crate::bounded) fn inspect_config(
    source: &str,
    filename: &str,
    in_expr: bool,
    whitespace: WhitespaceConfig,
) -> Result<(usize, usize, usize), PlanError> {
    let mut scanner = Scanner::new(source, filename, in_expr, Syntax::Default, whitespace);
    let (mut events, mut bytes, mut numeric) = (0usize, 0usize, 0usize);
    loop {
        let before = scanner.progress();
        match scanner.next_token() {
            Ok(None) => break,
            Ok(Some((token, _))) => {
                events = events.checked_add(1).ok_or(PlanError::Geometry)?;
                match token {
                    Lexeme::Escaped(raw) => match decode(raw, Output::Count(&mut bytes)) {
                        Ok(()) | Err(Failure::BadEscape) => (),
                        Err(Failure::Capacity) => return Err(PlanError::Geometry),
                    },
                    Lexeme::Number(value) if value.has_underscore => {
                        numeric = numeric.max(value.raw.bytes().filter(|&b| b != b'_').count());
                    }
                    _ => (),
                }
            }
            Err(_) => {
                events = events.checked_add(1).ok_or(PlanError::Geometry)?;
                // All mutable scanner fields are compared. In this unchanged
                // state a later fetch deterministically repeats the same error,
                // and cannot create another literal or successful token.
                if scanner.progress() == before {
                    break;
                }
            }
        }
    }
    Ok((events, bytes, numeric))
}

pub(in crate::bounded) struct Buffers {
    pub(in crate::bounded) literals: Vec<u8>,
    pub(in crate::bounded) numeric: Vec<u8>,
    pub(in crate::bounded) literal_limit: usize,
    pub(in crate::bounded) numeric_limit: usize,
}
impl Buffers {
    pub(in crate::bounded) fn new(literal_limit: usize, numeric_limit: usize) -> Self {
        Self {
            literals: Vec::new(),
            numeric: Vec::new(),
            literal_limit,
            numeric_limit,
        }
    }
}

pub(in crate::bounded) struct ParserInput<'a, 's> {
    scanner: Scanner<'s>,
    buffers: &'a mut Buffers,
    current: Result<Option<(Item<'s, Bytes>, Span)>, Cause<'s>>,
    last_span: Span,
    terminal: Option<Terminal>,
}

#[derive(Clone, Copy)]
enum Terminal {
    Capacity(Buffer),
    ExpressionEnd,
}
impl Terminal {
    fn cause<'s>(self) -> Cause<'s> {
        match self {
            Self::Capacity(buffer) => Cause::Capacity(buffer),
            Self::ExpressionEnd => Cause::ExpressionEnd,
        }
    }
}
impl<'a, 's> ParserInput<'a, 's> {
    pub(in crate::bounded) fn new(
        source: &'s str,
        filename: &'s str,
        buffers: &'a mut Buffers,
    ) -> Self {
        Self::configured(source, filename, buffers, true, WhitespaceConfig::default())
    }
    pub(in crate::bounded) fn configured(
        source: &'s str,
        filename: &'s str,
        buffers: &'a mut Buffers,
        in_expr: bool,
        whitespace: WhitespaceConfig,
    ) -> Self {
        let mut input = Self {
            scanner: Scanner::new(source, filename, in_expr, Syntax::Default, whitespace),
            buffers,
            current: Ok(None),
            last_span: Span::default(),
            terminal: None,
        };
        input.current = input.fetch();
        input
    }
    fn fetch(&mut self) -> Result<Option<(Item<'s, Bytes>, Span)>, Cause<'s>> {
        let result = self.fetch_raw();
        self.terminal = match &result {
            Err(Cause::Capacity(buffer)) => Some(Terminal::Capacity(*buffer)),
            Err(Cause::ExpressionEnd) => Some(Terminal::ExpressionEnd),
            _ => self.terminal,
        };
        result
    }
    fn fetch_raw(&mut self) -> Result<Option<(Item<'s, Bytes>, Span)>, Cause<'s>> {
        let Some((token, span)) = self.scanner.next_token().map_err(Cause::lexical)? else {
            return Ok(None);
        };
        let item = match token {
            Lexeme::Escaped(raw) => {
                let start = self.buffers.literals.len();
                decode(
                    raw,
                    Output::Packed {
                        bytes: &mut self.buffers.literals,
                        limit: self.buffers.literal_limit,
                    },
                )
                .map_err(|error| match error {
                    Failure::BadEscape => Cause::lexical(LexError {
                        empty_stack: false,
                        kind: ErrorKind::BadEscape,
                        detail: None,
                        span: None,
                    }),
                    Failure::Capacity => Cause::Capacity(Buffer::Literals),
                })?;
                Item::String(Bytes {
                    start,
                    end: self.buffers.literals.len(),
                })
            }
            Lexeme::Number(number) => {
                self.buffers.numeric.clear();
                let raw = if number.has_underscore {
                    for byte in number.raw.bytes().filter(|&b| b != b'_') {
                        if self.buffers.numeric.len() >= self.buffers.numeric_limit
                            || self.buffers.numeric.len() == self.buffers.numeric.capacity()
                        {
                            return Err(Cause::Capacity(Buffer::NumericScratch));
                        }
                        self.buffers.numeric.push(byte);
                    }
                    std::str::from_utf8(&self.buffers.numeric).expect("ASCII numeric scanner")
                } else {
                    number.raw
                };
                match parse_number(raw, number)
                    .map_err(|detail| Cause::lexical(self.scanner.syntax_error(detail)))?
                {
                    Number::Int(value) => Item::Int(value),
                    Number::Int128(value) => Item::Int128(value),
                    Number::Float(value) => Item::Float(value),
                }
            }
            Lexeme::Str(value) => Item::Str(value),
            Lexeme::Ident(value) => Item::Simple(Kind::Ident(value)),
            Lexeme::TemplateData(raw) => Item::TemplateData(raw),
            Lexeme::VariableStart => Item::Simple(Kind::VariableStart),
            Lexeme::VariableEnd => Item::Simple(Kind::VariableEnd),
            Lexeme::BlockStart => Item::Simple(Kind::BlockStart),
            Lexeme::BlockEnd => Item::Simple(Kind::BlockEnd),
            Lexeme::Plus => Item::Simple(Kind::Plus),
            Lexeme::Minus => Item::Simple(Kind::Minus),
            Lexeme::Mul => Item::Simple(Kind::Mul),
            Lexeme::Div => Item::Simple(Kind::Div),
            Lexeme::FloorDiv => Item::Simple(Kind::FloorDiv),
            Lexeme::Pow => Item::Simple(Kind::Pow),
            Lexeme::Mod => Item::Simple(Kind::Mod),
            Lexeme::Dot => Item::Simple(Kind::Dot),
            Lexeme::Comma => Item::Simple(Kind::Comma),
            Lexeme::Colon => Item::Simple(Kind::Colon),
            Lexeme::Tilde => Item::Simple(Kind::Tilde),
            Lexeme::Assign => Item::Simple(Kind::Assign),
            Lexeme::Pipe => Item::Simple(Kind::Pipe),
            Lexeme::Eq => Item::Simple(Kind::Eq),
            Lexeme::Ne => Item::Simple(Kind::Ne),
            Lexeme::Gt => Item::Simple(Kind::Gt),
            Lexeme::Gte => Item::Simple(Kind::Gte),
            Lexeme::Lt => Item::Simple(Kind::Lt),
            Lexeme::Lte => Item::Simple(Kind::Lte),
            Lexeme::BracketOpen => Item::Simple(Kind::BracketOpen),
            Lexeme::BracketClose => Item::Simple(Kind::BracketClose),
            Lexeme::ParenOpen => Item::Simple(Kind::ParenOpen),
            Lexeme::ParenClose => Item::Simple(Kind::ParenClose),
            Lexeme::BraceOpen => Item::Simple(Kind::BraceOpen),
            Lexeme::BraceClose => Item::Simple(Kind::BraceClose),
        };
        Ok(Some((item, span)))
    }
    fn take_error(&mut self) -> Result<(), Cause<'s>> {
        if let Some(terminal) = self.terminal {
            return Err(terminal.cause());
        }
        if self.current.is_err() {
            match mem::replace(&mut self.current, Ok(None)) {
                Err(error) => Err(error),
                _ => unreachable!(),
            }
        } else {
            Ok(())
        }
    }
}

impl<'s> Input<'s> for ParserInput<'_, 's> {
    type Text = Bytes;
    type Error = Cause<'s>;
    fn next(&mut self) -> Result<Option<(Item<'s, Bytes>, Span)>, Cause<'s>> {
        if let Some(terminal) = self.terminal {
            return Err(terminal.cause());
        }
        // Preserve prefetch before returning current, including when current
        // already contains an error. The owner retains both materialized prefixes.
        let fetched = self.fetch();
        if let Some(terminal) = self.terminal {
            return Err(terminal.cause());
        }
        let value = mem::replace(&mut self.current, fetched)?;
        if let Some((_, span)) = value.as_ref() {
            self.last_span = *span;
        }
        Ok(value)
    }
    fn current(&mut self) -> Result<Option<(Kind<'s>, Span)>, Cause<'s>> {
        self.take_error()?;
        Ok(self
            .current
            .as_ref()
            .unwrap()
            .as_ref()
            .map(|(item, span)| (item.kind(), *span)))
    }
    fn current_text(&mut self) -> Result<Option<TextRef<'_, 's, Bytes>>, Cause<'s>> {
        self.take_error()?;
        Ok(match self.current.as_ref().unwrap() {
            Some((Item::Str(value), _)) => Some(TextRef::Source(value)),
            Some((Item::String(value), _)) => Some(TextRef::Decoded(value)),
            _ => None,
        })
    }
    fn current_span(&self) -> Span {
        match &self.current {
            Ok(Some((_, span))) => *span,
            _ => self.last_span,
        }
    }
    fn last_span(&self) -> Span {
        self.last_span
    }
    fn source(&self) -> &'s str {
        self.scanner.source()
    }
}
