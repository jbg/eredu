//! Private statement payload/storage operations, with no public capacity authority.
#![forbid(unsafe_code)]

use super::super::storage::Store;
use crate::compiler::tokens::Span;

pub(crate) enum Statement<'s, S: StatementStore<'s>> {
    Template(S::Body),
    EmitExpr(S::Expr),
    EmitRaw(&'s str),
    For {
        target: S::Expr,
        iter: S::Expr,
        filter: Option<S::Expr>,
        recursive: bool,
        body: S::Body,
        otherwise: S::Body,
    },
    If {
        test: S::Expr,
        yes: S::Body,
        no: S::Body,
    },
    With(S::Bindings, S::Body),
    Set(S::Expr, S::Expr),
    SetBlock(S::Expr, Option<S::Expr>, S::Body),
    AutoEscape(S::Expr, S::Body),
    FilterBlock(S::Expr, S::Body),
    #[cfg(feature = "multi_template")]
    Block {
        name: &'s str,
        required: bool,
        body: S::Body,
    },
    #[cfg(feature = "multi_template")]
    Extends(S::Expr),
    #[cfg(feature = "multi_template")]
    Include(S::Expr, bool),
    #[cfg(feature = "multi_template")]
    Import(S::Expr, S::Expr),
    #[cfg(feature = "multi_template")]
    FromImport(S::Expr, S::Bindings),
    #[cfg(feature = "macros")]
    Macro {
        name: &'s str,
        args: S::Exprs,
        defaults: S::Exprs,
        body: S::Body,
    },
    #[cfg(feature = "macros")]
    CallBlock {
        call: S::Call,
        args: S::Exprs,
        defaults: S::Exprs,
        body: S::Body,
        macro_span: Span,
    },
    #[cfg(feature = "loop_controls")]
    Continue,
    #[cfg(feature = "loop_controls")]
    Break,
    Do(S::Call),
}

pub(crate) trait StatementStore<'s>: Store<'s> {
    type Stmt;
    type Body;
    type Bindings;
    type Call;

    fn statement(
        &mut self,
        value: Statement<'s, Self>,
        span: Span,
    ) -> Result<Self::Stmt, Self::Error>;
    fn body(&mut self, hint: usize) -> Self::Body;
    fn push_statement(
        &mut self,
        body: &mut Self::Body,
        stmt: Self::Stmt,
    ) -> Result<(), Self::Error>;
    fn body_is_whitespace(&self, body: &Self::Body) -> bool;
    fn first_expr(&mut self, exprs: Self::Exprs) -> Self::Expr;
    fn bindings(&mut self, hint: usize, optional: bool) -> Self::Bindings;
    fn bindings_len(&self, bindings: &Self::Bindings) -> usize;
    fn push_binding(
        &mut self,
        bindings: &mut Self::Bindings,
        target: Self::Expr,
        value: Option<Self::Expr>,
    ) -> Result<(), Self::Error>;
    fn call(&mut self, expr: Self::Expr) -> Result<Self::Call, Self::Error>;
}

pub(crate) trait Names<'s, E> {
    fn insert(&mut self, name: &'s str) -> Result<bool, E>;
}
impl<'s, E> Names<'s, E> for std::collections::BTreeSet<&'s str> {
    fn insert(&mut self, name: &'s str) -> Result<bool, E> {
        Ok(std::collections::BTreeSet::insert(self, name))
    }
}
