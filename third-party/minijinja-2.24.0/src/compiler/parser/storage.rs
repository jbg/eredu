//! Private storage boundary for the shared expression grammar.
#![forbid(unsafe_code)]

use crate::compiler::ast::{BinOpKind, CompareOpKind, UnaryOpKind};
use crate::compiler::tokens::Span;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind<'s> {
    TemplateData,
    VariableStart,
    VariableEnd,
    BlockStart,
    BlockEnd,
    Ident(&'s str),
    Str,
    String,
    Int,
    Int128,
    Float,
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
}

impl Kind<'_> {
    pub(crate) fn is_string(self) -> bool {
        matches!(self, Self::Str | Self::String)
    }
}

impl fmt::Display for Kind<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::TemplateData => "template-data",
            Self::VariableStart => "start of variable block",
            Self::VariableEnd => "end of variable block",
            Self::BlockStart => "start of block",
            Self::BlockEnd => "end of block",
            Self::Ident(_) => "identifier",
            Self::Str | Self::String => "string",
            Self::Int | Self::Int128 => "integer",
            Self::Float => "float",
            Self::Plus => "`+`",
            Self::Minus => "`-`",
            Self::Mul => "`*`",
            Self::Div => "`/`",
            Self::FloorDiv => "`//`",
            Self::Pow => "`**`",
            Self::Mod => "`%`",
            Self::Dot => "`.`",
            Self::Comma => "`,`",
            Self::Colon => "`:`",
            Self::Tilde => "`~`",
            Self::Assign => "`=`",
            Self::Pipe => "`|`",
            Self::Eq => "`==`",
            Self::Ne => "`!=`",
            Self::Gt => "`>`",
            Self::Gte => "`>=`",
            Self::Lt => "`<`",
            Self::Lte => "`<=`",
            Self::BracketOpen => "`[`",
            Self::BracketClose => "`]`",
            Self::ParenOpen => "`(`",
            Self::ParenClose => "`)`",
            Self::BraceOpen => "`{`",
            Self::BraceClose => "`}`",
        })
    }
}

pub(crate) enum Item<'s, T> {
    Simple(Kind<'s>),
    TemplateData(&'s str),
    Str(&'s str),
    String(T),
    Int(u64),
    Int128(u128),
    Float(f64),
}

impl<'s, T> Item<'s, T> {
    pub(crate) fn kind(&self) -> Kind<'s> {
        match self {
            Self::Simple(kind) => *kind,
            Self::TemplateData(_) => Kind::TemplateData,
            Self::Str(_) => Kind::Str,
            Self::String(_) => Kind::String,
            Self::Int(_) => Kind::Int,
            Self::Int128(_) => Kind::Int128,
            Self::Float(_) => Kind::Float,
        }
    }
}

pub(crate) enum Text<'s, T> {
    Source(&'s str),
    Decoded(T),
}

pub(crate) enum TextRef<'a, 's, T> {
    Source(&'s str),
    Decoded(&'a T),
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum SyntaxFailure<'s> {
    Unexpected(Kind<'s>, &'static str),
    UnexpectedOnly(Kind<'s>),
    Eof(&'static str),
    Static(&'static str),
    UnknownStatement(&'s str),
    UnknownStatementToken(Kind<'s>),
    ReservedName(&'s str),
    DuplicateBlock(&'s str),
    BlockName { actual: &'s str, expected: &'s str },
    ExpectedCall(&'static str),
}

impl fmt::Display for SyntaxFailure<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unexpected(found, expected) => {
                write!(f, "unexpected {found}, expected {expected}")
            }
            Self::UnexpectedOnly(found) => write!(f, "unexpected {found}"),
            Self::Eof(expected) => write!(f, "unexpected end of input, expected {expected}"),
            Self::Static(detail) => f.write_str(detail),
            Self::UnknownStatement(name) => write!(f, "unknown statement {name}"),
            Self::UnknownStatementToken(kind) => write!(f, "unknown {kind}, expected statement"),
            Self::ReservedName(name) => write!(f, "cannot assign to reserved variable name {name}"),
            Self::DuplicateBlock(name) => write!(f, "block '{name}' defined twice"),
            Self::BlockName { actual, expected } => write!(
                f,
                "mismatching name on block. Got `{actual}`, expected `{expected}`"
            ),
            Self::ExpectedCall(description) => write!(
                f,
                "expected call expression in call block, got {description}"
            ),
        }
    }
}

pub(crate) trait Input<'s> {
    type Text;
    type Error;

    fn next(&mut self) -> Result<Option<(Item<'s, Self::Text>, Span)>, Self::Error>;
    fn current(&mut self) -> Result<Option<(Kind<'s>, Span)>, Self::Error>;
    fn current_text(&mut self) -> Result<Option<TextRef<'_, 's, Self::Text>>, Self::Error>;
    fn current_span(&self) -> Span;
    fn last_span(&self) -> Span;
    fn source(&self) -> &'s str;

    fn expand_span(&self, mut span: Span) -> Span {
        let last = self.last_span();
        span.end_line = last.end_line;
        span.end_col = last.end_col;
        span.end_offset = last.end_offset;
        span
    }
}

pub(crate) enum Literal<'s, C> {
    None,
    Bool(bool),
    Int(u64),
    Int128(u128),
    Float(f64),
    Plain(&'s str),
    Joined(C),
}

pub(crate) enum Arg<'s, E> {
    Pos(E),
    Kwarg(&'s str, E),
    PosSplat(E),
    KwargSplat(E),
}

pub(crate) enum Node<'s, S: Store<'s>> {
    Var(&'s str),
    Const(Literal<'s, S::Concat>),
    Slice {
        expr: S::Expr,
        start: Option<S::Expr>,
        stop: Option<S::Expr>,
        step: Option<S::Expr>,
    },
    Unary(UnaryOpKind, S::Expr),
    Binary(BinOpKind, S::Expr, S::Expr),
    Compare(S::Expr, S::Comparisons),
    If(S::Expr, S::Expr, Option<S::Expr>),
    Filter(&'s str, Option<S::Expr>, S::Args),
    Test(&'s str, S::Expr, S::Args),
    Attr(S::Expr, &'s str),
    Item(S::Expr, S::Expr),
    Call(S::Expr, S::Args),
    List(S::Exprs),
    Map(S::Exprs, S::Exprs),
}

pub(crate) trait Store<'s>: Sized {
    type Expr;
    type Exprs;
    type Args;
    type Comparisons;
    type Text;
    type Concat;
    type Error;

    fn node(&mut self, node: Node<'s, Self>, span: Span) -> Result<Self::Expr, Self::Error>;
    fn variable(&self, expr: &Self::Expr) -> Option<(&'s str, Span)>;
    fn exprs(&mut self, initial_hint: usize) -> Self::Exprs;
    fn exprs_len(&self, values: &Self::Exprs) -> usize;
    fn push_expr(&mut self, values: &mut Self::Exprs, expr: Self::Expr) -> Result<(), Self::Error>;
    fn args(&mut self) -> Self::Args;
    fn args_len(&self, args: &Self::Args) -> usize;
    fn push_arg(
        &mut self,
        args: &mut Self::Args,
        arg: Arg<'s, Self::Expr>,
    ) -> Result<(), Self::Error>;
    fn comparisons(&mut self) -> Self::Comparisons;
    fn comparisons_len(&self, values: &Self::Comparisons) -> usize;
    fn push_comparison(
        &mut self,
        values: &mut Self::Comparisons,
        op: CompareOpKind,
        expr: Self::Expr,
    ) -> Result<(), Self::Error>;
    fn pop_comparison(&mut self, values: &mut Self::Comparisons) -> (CompareOpKind, Self::Expr);
    fn begin_text(&mut self, text: Text<'s, Self::Text>) -> Result<Self::Concat, Self::Error>;
    fn append_text(
        &mut self,
        joined: &mut Self::Concat,
        text: TextRef<'_, 's, Self::Text>,
    ) -> Result<(), Self::Error>;
    fn syntax(&self, failure: SyntaxFailure<'s>) -> Self::Error;
}
