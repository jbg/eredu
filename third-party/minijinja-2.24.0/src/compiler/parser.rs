#![forbid(unsafe_code)]

pub(crate) mod ordinary;
pub(crate) mod shared;
pub(crate) mod statements;
pub(crate) mod storage;

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::mem;

use crate::compiler::ast;
use crate::compiler::lexer::{Tokenizer, WhitespaceConfig};
use crate::compiler::tokens::{Span, Token};
use crate::error::{Error, ErrorKind};
use crate::syntax::SyntaxConfig;

const MAX_RECURSION: usize = 150;
const RESERVED_NAMES: [&str; 8] = [
    "true", "True", "false", "False", "none", "None", "loop", "self",
];

fn syntax_error(msg: Cow<'static, str>) -> Error {
    Error::new(ErrorKind::SyntaxError, msg)
}

macro_rules! syntax_error {
    ($msg:expr) => {{
        return Err(syntax_error(Cow::Borrowed($msg)));
    }};
    ($msg:expr, $($tt:tt)*) => {{
        return Err(syntax_error(Cow::Owned(format!($msg, $($tt)*))));
    }};
}

struct TokenStream<'a> {
    tokenizer: Tokenizer<'a>,
    current: Result<Option<(Token<'a>, Span)>, Error>,
    last_span: Span,
}

impl<'a> TokenStream<'a> {
    /// Tokenize a template
    pub fn new(
        source: &'a str,
        filename: &'a str,
        in_expr: bool,
        syntax_config: SyntaxConfig,
        whitespace_config: WhitespaceConfig,
    ) -> TokenStream<'a> {
        let mut tokenizer =
            Tokenizer::new(source, filename, in_expr, syntax_config, whitespace_config);
        let current = tokenizer.next_token();
        TokenStream {
            tokenizer,
            current,
            last_span: Span::default(),
        }
    }

    /// Advance the stream.
    #[inline(always)]
    pub fn next(&mut self) -> Result<Option<(Token<'a>, Span)>, Error> {
        let rv = mem::replace(&mut self.current, self.tokenizer.next_token());
        match rv {
            Ok(Some((token, span))) => {
                self.last_span = span;
                Ok(Some((token, span)))
            }
            Ok(None) => Ok(None),
            Err(err) => Err(err),
        }
    }

    /// Look at the current token
    #[inline(always)]
    pub fn current(&mut self) -> Result<Option<(&Token<'a>, Span)>, Error> {
        if self.current.is_err() {
            return match mem::replace(&mut self.current, Ok(None)) {
                Err(err) => Err(err),
                _ => unreachable!(),
            };
        }

        match self.current {
            Ok(Some((ref token, span))) => Ok(Some((token, span))),
            Ok(None) => Ok(None),
            Err(_) => unreachable!(),
        }
    }

    /// Expands the span
    #[inline(always)]
    pub fn expand_span(&self, mut span: Span) -> Span {
        span.end_line = self.last_span.end_line;
        span.end_col = self.last_span.end_col;
        span.end_offset = self.last_span.end_offset;
        span
    }

    /// Returns the current span.
    #[inline(always)]
    pub fn current_span(&self) -> Span {
        match self.current {
            Ok(Some((_, span))) => span,
            _ => self.last_span,
        }
    }

    /// Returns the last seen span.
    #[inline(always)]
    pub fn last_span(&self) -> Span {
        self.last_span
    }
}

struct Parser<'a> {
    stream: TokenStream<'a>,
    #[allow(unused)]
    in_macro: bool,
    #[allow(unused)]
    in_loop: bool,
    #[allow(unused)]
    blocks: BTreeSet<&'a str>,
    depth: usize,
}

impl<'a> Parser<'a> {
    /// Creates a new parser.
    ///
    /// `in_expr` is necessary to parse within an expression context.  Otherwise
    /// the parser starts out in template context.  This means that when
    /// [`parse`](Self::parse) is to be called, the `in_expr` argument must be
    /// `false` and for [`parse_standalone_expr`](Self::parse_standalone_expr)
    /// it must be `true`.
    pub fn new(
        source: &'a str,
        filename: &'a str,
        in_expr: bool,
        syntax_config: SyntaxConfig,
        whitespace_config: WhitespaceConfig,
    ) -> Parser<'a> {
        Parser {
            stream: TokenStream::new(source, filename, in_expr, syntax_config, whitespace_config),
            in_macro: false,
            in_loop: false,
            blocks: BTreeSet::new(),
            depth: 0,
        }
    }

    /// Parses a template.
    pub fn parse(&mut self) -> Result<ast::Stmt<'a>, Error> {
        let mut statements = Vec::new();
        let mut expressions = Vec::new();
        statements::run(
            &mut self.stream,
            &mut ordinary::Ordinary,
            &mut expressions,
            &mut self.blocks,
            &mut statements,
            statements::State {
                depth: &mut self.depth,
                in_loop: &mut self.in_loop,
                in_macro: &mut self.in_macro,
            },
        )
        .map_err(|err| self.attach_location_to_error(err))
    }

    /// Parses an expression and asserts that there is no more input after it.
    pub fn parse_standalone_expr(&mut self) -> Result<ast::Expr<'a>, Error> {
        self.parse_expr()
            .and_then(|result| {
                if ok!(self.stream.next()).is_some() {
                    syntax_error!("unexpected input after expression")
                } else {
                    Ok(result)
                }
            })
            .map_err(|err| self.attach_location_to_error(err))
    }

    /// Returns the current filename.
    pub fn filename(&self) -> &str {
        self.stream.tokenizer.filename()
    }

    fn parse_expr(&mut self) -> Result<ast::Expr<'a>, Error> {
        self.shared_expression(shared::Method::Expr)
    }

    fn shared_expression(&mut self, method: shared::Method) -> Result<ast::Expr<'a>, Error> {
        let mut frames = Vec::new();
        match shared::run(
            method,
            &mut self.stream,
            &mut ordinary::Ordinary,
            &mut frames,
            &mut self.depth,
        )? {
            shared::Output::Expr(expr) => Ok(expr),
            _ => unreachable!("expression parser output"),
        }
    }

    #[inline]
    fn attach_location_to_error(&mut self, mut err: Error) -> Error {
        if err.line().is_none() {
            err.set_filename_and_span(self.filename(), self.stream.last_span())
        }
        err
    }
}

/// Parses a template.
pub fn parse<'source>(
    source: &'source str,
    filename: &'source str,
    syntax_config: SyntaxConfig,
    whitespace_config: WhitespaceConfig,
) -> Result<ast::Stmt<'source>, Error> {
    Parser::new(source, filename, false, syntax_config, whitespace_config).parse()
}

/// Parses a standalone expression.
pub fn parse_expr(source: &str) -> Result<ast::Expr<'_>, Error> {
    Parser::new(
        source,
        "<expression>",
        true,
        Default::default(),
        Default::default(),
    )
    .parse_standalone_expr()
}

#[cfg(test)]
pub(crate) fn expression_depth_case(
    source: &str,
    initial: usize,
) -> (Result<ast::Expr<'_>, Error>, usize) {
    let mut parser = Parser::new(
        source,
        "<expression>",
        true,
        Default::default(),
        Default::default(),
    );
    parser.depth = initial;
    let value = parser.parse_expr();
    (value, parser.depth)
}

#[cfg(test)]
pub(crate) fn statement_reuse_case<'s>(
    source: &'s str,
) -> Vec<(
    Result<Result<ast::Stmt<'s>, Error>, Box<dyn std::any::Any + Send>>,
    bool,
    bool,
    usize,
    Vec<&'s str>,
)> {
    let mut parser = Parser::new(
        source,
        "template",
        false,
        Default::default(),
        Default::default(),
    );
    (0..3)
        .map(|_| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| parser.parse()));
            (
                result,
                parser.in_loop,
                parser.in_macro,
                parser.depth,
                parser.blocks.iter().copied().collect(),
            )
        })
        .collect()
}
