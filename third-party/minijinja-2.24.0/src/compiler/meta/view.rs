//! Private borrowed syntax views and capture storage operations.
#![forbid(unsafe_code)]

#[derive(Clone, Copy)]
pub(crate) enum Expression<'s, V: View<'s>> {
    Var(&'s str),
    Const,
    Unary(V::Expr),
    Binary(V::Expr, V::Expr),
    Compare(V::Expr, V::Cursor),
    If(V::Expr, V::Expr, Option<V::Expr>),
    Filter(Option<V::Expr>, V::Cursor),
    Test(V::Expr, V::Cursor),
    Attr(V::Expr, &'s str),
    Item(V::Expr, V::Expr),
    Slice(Option<V::Expr>, Option<V::Expr>, Option<V::Expr>),
    Call(V::Expr, V::Cursor),
    List(V::Cursor),
    Map(V::Cursor),
}

#[derive(Clone, Copy)]
pub(crate) enum Statement<'s, V: View<'s>> {
    Template(V::Cursor),
    EmitExpr(V::Expr),
    EmitRaw,
    For {
        target: V::Expr,
        iter: V::Expr,
        filter: Option<V::Expr>,
        body: V::Cursor,
        otherwise: V::Cursor,
    },
    If(V::Expr, V::Cursor, V::Cursor),
    With(V::Cursor, V::Cursor),
    Set(V::Expr, V::Expr),
    SetBlock(V::Expr, V::Cursor),
    AutoEscape(V::Cursor),
    FilterBlock(V::Cursor),
    #[cfg(feature = "multi_template")]
    Block(V::Cursor),
    #[cfg(feature = "multi_template")]
    Extends,
    #[cfg(feature = "multi_template")]
    Include,
    #[cfg(feature = "multi_template")]
    Import(V::Expr),
    #[cfg(feature = "multi_template")]
    FromImport(V::Cursor),
    #[cfg(feature = "macros")]
    Macro(&'s str, V::Macro),
    #[cfg(feature = "macros")]
    CallBlock(V::Expr, V::Cursor, V::Macro),
    #[cfg(feature = "loop_controls")]
    Continue,
    #[cfg(feature = "loop_controls")]
    Break,
    Do(V::Expr, V::Cursor),
}

#[derive(Clone, Copy)]
pub(crate) enum Item<'s, V: View<'s>> {
    Stmt(V::Stmt),
    Expr(V::Expr),
    Pair(V::Expr, Option<V::Expr>),
}

#[cfg(feature = "macros")]
pub(crate) struct MacroParts<'s, V: View<'s>> {
    pub(crate) args: V::Cursor,
    pub(crate) defaults: V::Cursor,
    pub(crate) body: V::Cursor,
}

pub(crate) trait View<'s>: Copy {
    type Expr: Copy;
    type Stmt: Copy;
    #[cfg(feature = "macros")]
    type Macro: Copy;
    type Cursor: Copy;
    fn expression(self, expr: Self::Expr) -> Expression<'s, Self>;
    fn statement(self, stmt: Self::Stmt) -> Statement<'s, Self>;
    #[cfg(feature = "macros")]
    fn macro_parts(self, value: Self::Macro) -> MacroParts<'s, Self>;
    fn next(self, cursor: Self::Cursor) -> Option<(Item<'s, Self>, Self::Cursor)>;
}

#[derive(Clone, Copy)]
pub(crate) enum SequenceKind {
    Body,
    Lookup,
    Assign,
    Map,
    Binding,
    #[cfg(feature = "multi_template")]
    Import,
}

#[derive(Clone, Copy)]
pub(crate) enum Task<'s, V: View<'s>> {
    Stmt(V::Stmt),
    Expr(V::Expr),
    Assign(V::Expr),
    #[cfg(feature = "macros")]
    Macro(V::Macro, bool),
    Sequence(SequenceKind, V::Cursor),
    PushScope,
    PopScope,
    Name(&'s str),
}

pub(crate) trait Storage<'s, V: View<'s>> {
    type Error;
    type Nested;
    fn push_task(&mut self, task: Task<'s, V>) -> Result<(), Self::Error>;
    fn pop_task(&mut self) -> Option<Task<'s, V>>;
    fn is_assigned(&self, name: &str) -> bool;
    fn assign(&mut self, name: &'s str) -> Result<(), Self::Error>;
    fn capture(&mut self, name: &'s str) -> Result<(), Self::Error>;
    fn push_scope(&mut self) -> Result<(), Self::Error>;
    fn pop_scope(&mut self) -> Result<(), Self::Error>;
    fn tracks_nested(&self) -> bool;
    fn nested_start(&mut self, attr: &'s str) -> Result<Self::Nested, Self::Error>;
    fn nested_attr(&mut self, attrs: &mut Self::Nested, attr: &'s str) -> Result<(), Self::Error>;
    fn nested_finish(&mut self, attrs: Self::Nested, root: &'s str) -> Result<(), Self::Error>;
    fn nested_variable(&mut self, name: &'s str) -> Result<(), Self::Error>;
}
