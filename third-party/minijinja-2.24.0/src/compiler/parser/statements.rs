//! Shared full-template syntax decisions with explicit recursive continuations.
#![forbid(unsafe_code)]

pub(crate) mod storage;
use self::storage::{Names, Statement, StatementStore};
use super::shared;
use super::storage::{Input, Item, Kind, Node, SyntaxFailure};
use crate::compiler::tokens::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Root,
    Subparse,
    Stmt,
    Assignment,
    For,
    If,
    With,
    Set,
    AutoEscape,
    FilterBlock,
    #[cfg(feature = "multi_template")]
    Block,
    #[cfg(feature = "macros")]
    Macro,
    #[cfg(feature = "macros")]
    CallBlock,
}
#[derive(Clone, Copy)]
pub(crate) enum End {
    Never,
    For,
    If,
    Named(&'static str),
}
impl End {
    fn matches(self, kind: Kind<'_>) -> bool {
        match self {
            Self::Never => false,
            Self::For => matches!(kind, Kind::Ident("endfor" | "else")),
            Self::If => matches!(kind, Kind::Ident("endif" | "else" | "elif")),
            Self::Named(name) => kind == Kind::Ident(name),
        }
    }
}

// Actual zero-consumption Call edges: Root -> Subparse, For -> Assignment,
// Set -> Assignment. All other recursive Calls follow a consumed token.
// Each zero-edge segment therefore contains at most two frames.
pub(crate) const ZERO_RUN_FRAMES: usize = 2;

pub(crate) struct Frame<'s, S: StatementStore<'s>> {
    method: Method,
    phase: u8,
    span: Span,
    inner_span: Span,
    end: End,
    expr: Option<S::Expr>,
    other: Option<S::Expr>,
    filter: Option<S::Expr>,
    exprs: Option<S::Exprs>,
    defaults: Option<S::Exprs>,
    body: Option<S::Body>,
    otherwise: Option<S::Body>,
    bindings: Option<S::Bindings>,
    call: Option<S::Call>,
    name: Option<&'s str>,
    old_loop: bool,
    old_macro: bool,
    flag: bool,
    dotted: bool,
}
impl<'s, S: StatementStore<'s>> Frame<'s, S> {
    fn new(method: Method) -> Self {
        Self {
            method,
            phase: 0,
            span: Span::default(),
            inner_span: Span::default(),
            end: End::Never,
            expr: None,
            other: None,
            filter: None,
            exprs: None,
            defaults: None,
            body: None,
            otherwise: None,
            bindings: None,
            call: None,
            name: None,
            old_loop: false,
            old_macro: false,
            flag: false,
            dotted: false,
        }
    }
    fn body(end: End) -> Self {
        let mut f = Self::new(Method::Subparse);
        f.end = end;
        f
    }
    fn assignment(dotted: bool) -> Self {
        let mut f = Self::new(Method::Assignment);
        f.dotted = dotted;
        f
    }
}

pub(crate) trait Stack<'s, S: StatementStore<'s>> {
    fn push(&mut self, frame: Frame<'s, S>) -> Result<(), S::Error>;
    fn pop(&mut self) -> Option<Frame<'s, S>>;
}
impl<'s, S: StatementStore<'s>> Stack<'s, S> for Vec<Frame<'s, S>> {
    fn push(&mut self, frame: Frame<'s, S>) -> Result<(), S::Error> {
        Vec::push(self, frame);
        Ok(())
    }
    fn pop(&mut self) -> Option<Frame<'s, S>> {
        Vec::pop(self)
    }
}
enum Output<'s, S: StatementStore<'s>> {
    Stmt(S::Stmt),
    Value(Statement<'s, S>),
    Body(S::Body),
    Expr(S::Expr),
}
enum Action<'s, S: StatementStore<'s>> {
    Continue(Frame<'s, S>),
    Call(Frame<'s, S>, Frame<'s, S>),
    Return(Output<'s, S>),
}
pub(crate) struct State<'a> {
    pub(crate) depth: &'a mut usize,
    pub(crate) in_loop: &'a mut bool,
    pub(crate) in_macro: &'a mut bool,
}
struct Engine<'a, 's, S: StatementStore<'s>, I, X, N> {
    input: &'a mut I,
    store: &'a mut S,
    expressions: &'a mut X,
    names: &'a mut N,
    state: State<'a>,
    active_guards: usize,
    returned: Option<Output<'s, S>>,
}
pub(crate) fn control_bytes<'s, S: StatementStore<'s>, I, X, N>() -> usize {
    std::mem::size_of::<Engine<'_, 's, S, I, X, N>>()
        + std::mem::size_of::<Frame<'s, S>>()
        + std::mem::size_of::<Action<'s, S>>()
        + std::mem::size_of::<Result<S::Stmt, S::Error>>()
}
pub(crate) fn run<'s, S, I, X, N, W>(
    input: &mut I,
    store: &mut S,
    expressions: &mut X,
    names: &mut N,
    stack: &mut W,
    state: State<'_>,
) -> Result<S::Stmt, S::Error>
where
    S: StatementStore<'s>,
    I: Input<'s, Text = S::Text, Error = S::Error>,
    X: shared::Stack<'s, S>,
    N: Names<'s, S::Error>,
    W: Stack<'s, S>,
{
    stack.push(Frame::new(Method::Root))?;
    let mut engine = Engine {
        input,
        store,
        expressions,
        names,
        state,
        active_guards: 0,
        returned: None,
    };
    let result = (|| {
        while let Some(frame) = stack.pop() {
            match engine.step(frame)? {
                Action::Continue(frame) => stack.push(frame)?,
                Action::Call(parent, child) => {
                    stack.push(parent)?;
                    stack.push(child)?;
                }
                Action::Return(value) => engine.returned = Some(value),
            }
        }
        match engine.returned.take().expect("template return") {
            Output::Stmt(value) => Ok(value),
            _ => unreachable!("template return kind"),
        }
    })();
    if result.is_err() {
        *engine.state.depth -= engine.active_guards;
        // Deliberately do not restore loop/macro flags or remove block names.
        // The ordinary parser restores each only at its specific success point.
        while stack.pop().is_some() {}
    }
    result
}
impl<'s, S, I, X, N> Engine<'_, 's, S, I, X, N>
where
    S: StatementStore<'s>,
    I: Input<'s, Text = S::Text, Error = S::Error>,
    X: shared::Stack<'s, S>,
    N: Names<'s, S::Error>,
{
    fn body(&mut self) -> S::Body {
        match self.returned.take().expect("body return") {
            Output::Body(value) => value,
            _ => unreachable!("body return kind"),
        }
    }
    fn expr_return(&mut self) -> S::Expr {
        match self.returned.take().expect("assignment return") {
            Output::Expr(value) => value,
            _ => unreachable!("assignment return kind"),
        }
    }
    fn value(&mut self) -> Statement<'s, S> {
        match self.returned.take().expect("statement value") {
            Output::Value(value) => value,
            _ => unreachable!("statement value kind"),
        }
    }
    fn stmt(&mut self) -> S::Stmt {
        match self.returned.take().expect("statement return") {
            Output::Stmt(value) => value,
            _ => unreachable!("statement return kind"),
        }
    }
    fn expression(&mut self, method: shared::Method) -> Result<S::Expr, S::Error> {
        match shared::run(
            method,
            self.input,
            self.store,
            self.expressions,
            self.state.depth,
        )? {
            shared::Output::Expr(value) => Ok(value),
            _ => unreachable!("expression kind"),
        }
    }
    fn args(&mut self) -> Result<S::Args, S::Error> {
        match shared::run(
            shared::Method::Args,
            self.input,
            self.store,
            self.expressions,
            self.state.depth,
        )? {
            shared::Output::Args(value) => Ok(value),
            _ => unreachable!("args kind"),
        }
    }
    fn peek(&mut self) -> Result<Option<Kind<'s>>, S::Error> {
        self.input.current().map(|v| v.map(|(kind, _)| kind))
    }
    fn skip(&mut self, kind: Kind<'s>) -> Result<bool, S::Error> {
        if self.peek()? == Some(kind) {
            self.input.next()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn expect(&mut self, kind: Kind<'s>, expected: &'static str) -> Result<Span, S::Error> {
        match self.input.next()? {
            Some((item, span)) if item.kind() == kind => Ok(span),
            Some((item, _)) => Err(self
                .store
                .syntax(SyntaxFailure::Unexpected(item.kind(), expected))),
            None => Err(self.store.syntax(SyntaxFailure::Eof(expected))),
        }
    }
    fn ident(&mut self) -> Result<(&'s str, Span), S::Error> {
        match self.input.next()? {
            Some((Item::Simple(Kind::Ident(name)), span)) => Ok((name, span)),
            Some((item, _)) => Err(self
                .store
                .syntax(SyntaxFailure::Unexpected(item.kind(), "identifier"))),
            None => Err(self.store.syntax(SyntaxFailure::Eof("identifier"))),
        }
    }
    fn assign_name(&mut self, dotted: bool) -> Result<S::Expr, S::Error> {
        let (name, span) = self.ident()?;
        if super::RESERVED_NAMES.contains(&name) {
            return Err(self.store.syntax(SyntaxFailure::ReservedName(name)));
        }
        let mut value = self.store.node(Node::Var(name), span)?;
        if dotted {
            while self.skip(Kind::Dot)? {
                let (name, span) = self.ident()?;
                value = self.store.node(Node::Attr(value, name), span)?;
            }
        }
        Ok(value)
    }
    fn filter_chain(&mut self) -> Result<S::Expr, S::Error> {
        let mut filter = None;
        while self.peek()? != Some(Kind::BlockEnd) {
            if filter.is_some() {
                self.expect(Kind::Pipe, "`|`")?;
            }
            let (name, span) = shared::filter_name(self.input, self.store)?;
            let args = if self.peek()? == Some(Kind::ParenOpen) {
                self.args()?
            } else {
                self.store.args()
            };
            filter = Some(self.store.node(
                Node::Filter(name, filter, args),
                self.input.expand_span(span),
            )?);
        }
        filter.ok_or_else(|| {
            self.store
                .syntax(SyntaxFailure::Static("expected a filter"))
        })
    }
    #[cfg(feature = "multi_template")]
    fn context_marker(&mut self) -> Result<bool, S::Error> {
        if matches!(self.peek()?, Some(Kind::Ident("with" | "without"))) {
            self.input.next()?;
            self.expect(Kind::Ident("context"), "context")?;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    #[cfg(feature = "macros")]
    fn macro_args(&mut self, args: &mut S::Exprs, defaults: &mut S::Exprs) -> Result<(), S::Error> {
        loop {
            if self.skip(Kind::ParenClose)? {
                break;
            }
            if self.store.exprs_len(args) != 0 {
                self.expect(Kind::Comma, "`,`")?;
                if self.skip(Kind::ParenClose)? {
                    break;
                }
            }
            let arg = self.assign_name(false)?;
            self.store.push_expr(args, arg)?;
            if self.skip(Kind::Assign)? {
                let value = self.expression(shared::Method::Expr)?;
                self.store.push_expr(defaults, value)?;
            } else if self.store.exprs_len(defaults) != 0 {
                self.expect(Kind::Assign, "`=`")?;
            }
        }
        Ok(())
    }
    fn finish_stmt(
        &mut self,
        value: Statement<'s, S>,
        span: Span,
    ) -> Result<Action<'s, S>, S::Error> {
        let stmt = self.store.statement(value, self.input.expand_span(span))?;
        *self.state.depth -= 1;
        self.active_guards -= 1;
        Ok(Action::Return(Output::Stmt(stmt)))
    }
    fn step(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        use Method::*;
        match f.method {
            Root => {
                if f.phase == 0 {
                    f.span = self.input.last_span();
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::body(End::Never)));
                }
                let body = self.body();
                let stmt = self
                    .store
                    .statement(Statement::Template(body), self.input.expand_span(f.span))?;
                Ok(Action::Return(Output::Stmt(stmt)))
            }
            Subparse => {
                if f.phase == 0 {
                    f.body = Some(self.store.body(16));
                    f.phase = 1;
                }
                if f.phase == 2 {
                    let stmt = self.stmt();
                    self.store.push_statement(f.body.as_mut().unwrap(), stmt)?;
                    self.expect(Kind::BlockEnd, "end of block")?;
                    f.phase = 1;
                }
                match self.input.next()? {
                    None => Ok(Action::Return(Output::Body(f.body.take().unwrap()))),
                    Some((Item::TemplateData(raw), span)) => {
                        let stmt = self.store.statement(Statement::EmitRaw(raw), span)?;
                        self.store.push_statement(f.body.as_mut().unwrap(), stmt)?;
                        Ok(Action::Continue(f))
                    }
                    Some((Item::Simple(Kind::VariableStart), span)) => {
                        let expr = self.expression(shared::Method::Expr)?;
                        let stmt = self
                            .store
                            .statement(Statement::EmitExpr(expr), self.input.expand_span(span))?;
                        self.store.push_statement(f.body.as_mut().unwrap(), stmt)?;
                        self.expect(Kind::VariableEnd, "end of variable block")?;
                        Ok(Action::Continue(f))
                    }
                    Some((Item::Simple(Kind::BlockStart), _)) => {
                        let kind = self
                            .peek()?
                            .ok_or_else(|| self.store.syntax(SyntaxFailure::Eof("keyword")))?;
                        if f.end.matches(kind) {
                            return Ok(Action::Return(Output::Body(f.body.take().unwrap())));
                        }
                        f.phase = 2;
                        Ok(Action::Call(f, Frame::new(Stmt)))
                    }
                    _ => unreachable!("lexer produced garbage"),
                }
            }
            Stmt => {
                if f.phase == 1 {
                    let value = self.value();
                    return self.finish_stmt(value, f.span);
                }
                *self.state.depth += 1;
                if *self.state.depth > super::MAX_RECURSION {
                    return Err(self.store.syntax(SyntaxFailure::Static(
                        "template exceeds maximum recursion limits",
                    )));
                }
                self.active_guards += 1;
                let (item, span) = self
                    .input
                    .next()?
                    .ok_or_else(|| self.store.syntax(SyntaxFailure::Eof("block keyword")))?;
                f.span = span;
                let name = match item.kind() {
                    Kind::Ident(name) => name,
                    kind => {
                        return Err(self
                            .store
                            .syntax(SyntaxFailure::UnknownStatementToken(kind)))
                    }
                };
                let child = match name {
                    "for" => Some(For),
                    "if" => Some(If),
                    "with" => Some(With),
                    "set" => Some(Set),
                    "autoescape" => Some(AutoEscape),
                    "filter" => Some(FilterBlock),
                    #[cfg(feature = "multi_template")]
                    "block" => Some(Block),
                    #[cfg(feature = "macros")]
                    "macro" => Some(Macro),
                    #[cfg(feature = "macros")]
                    "call" => Some(CallBlock),
                    _ => None,
                };
                if let Some(method) = child {
                    f.phase = 1;
                    let mut child = Frame::new(method);
                    child.span = span;
                    return Ok(Action::Call(f, child));
                }
                let value = match name {
                    #[cfg(feature = "loop_controls")]
                    "continue" => {
                        if !*self.state.in_loop {
                            return Err(self.store.syntax(SyntaxFailure::Static(
                                "'continue' must be placed inside a loop",
                            )));
                        }
                        Statement::Continue
                    }
                    #[cfg(feature = "loop_controls")]
                    "break" => {
                        if !*self.state.in_loop {
                            return Err(self.store.syntax(SyntaxFailure::Static(
                                "'break' must be placed inside a loop",
                            )));
                        }
                        Statement::Break
                    }
                    "do" => {
                        let expr = self.expression(shared::Method::Expr)?;
                        Statement::Do(self.store.call(expr)?)
                    }
                    #[cfg(feature = "multi_template")]
                    "extends" => Statement::Extends(self.expression(shared::Method::Expr)?),
                    #[cfg(feature = "multi_template")]
                    "include" => {
                        let name = self.expression(shared::Method::Expr)?;
                        let skipped = self.context_marker()?;
                        let ignore = if self.skip(Kind::Ident("ignore"))? {
                            self.expect(Kind::Ident("missing"), "missing keyword")?;
                            if !skipped {
                                self.context_marker()?;
                            }
                            true
                        } else {
                            false
                        };
                        Statement::Include(name, ignore)
                    }
                    #[cfg(feature = "multi_template")]
                    "import" => {
                        let expr = self.expression(shared::Method::Expr)?;
                        self.expect(Kind::Ident("as"), "as")?;
                        let name = self.expression(shared::Method::Expr)?;
                        self.context_marker()?;
                        Statement::Import(expr, name)
                    }
                    #[cfg(feature = "multi_template")]
                    "from" => {
                        let expr = self.expression(shared::Method::Expr)?;
                        let mut names = self.store.bindings(4, true);
                        self.expect(Kind::Ident("import"), "import")?;
                        loop {
                            if self.context_marker()? || self.peek()? == Some(Kind::BlockEnd) {
                                break;
                            }
                            if self.store.bindings_len(&names) != 0 {
                                self.expect(Kind::Comma, "`,`")?;
                            }
                            if self.context_marker()? || self.peek()? == Some(Kind::BlockEnd) {
                                break;
                            }
                            let name = self.assign_name(false)?;
                            let alias = if self.skip(Kind::Ident("as"))? {
                                Some(self.assign_name(false)?)
                            } else {
                                None
                            };
                            self.store.push_binding(&mut names, name, alias)?;
                        }
                        Statement::FromImport(expr, names)
                    }
                    name => return Err(self.store.syntax(SyntaxFailure::UnknownStatement(name))),
                };
                self.finish_stmt(value, span)
            }
            Assignment => {
                if f.phase == 0 {
                    f.span = self.input.current_span();
                    f.exprs = Some(self.store.exprs(2));
                    f.phase = 1;
                }
                if f.phase == 2 {
                    let expr = self.expr_return();
                    self.expect(Kind::ParenClose, "`)`")?;
                    self.store.push_expr(f.exprs.as_mut().unwrap(), expr)?;
                    f.phase = 3;
                }
                if f.phase == 3 {
                    if self.peek()? == Some(Kind::Comma) {
                        f.flag = true;
                        f.phase = 1;
                    } else {
                        return self.finish_assignment(f);
                    }
                }
                if self.store.exprs_len(f.exprs.as_ref().unwrap()) != 0 {
                    self.expect(Kind::Comma, "`,`")?;
                }
                if matches!(
                    self.peek()?,
                    Some(Kind::ParenClose | Kind::VariableEnd | Kind::BlockEnd | Kind::Ident("in"))
                ) {
                    return self.finish_assignment(f);
                }
                if self.skip(Kind::ParenOpen)? {
                    f.phase = 2;
                    let child = Frame::assignment(f.dotted);
                    return Ok(Action::Call(f, child));
                }
                let expr = self.assign_name(f.dotted)?;
                self.store.push_expr(f.exprs.as_mut().unwrap(), expr)?;
                f.phase = 3;
                Ok(Action::Continue(f))
            }
            For => {
                if f.phase == 0 {
                    f.old_loop = std::mem::replace(self.state.in_loop, true);
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::assignment(false)));
                }
                if f.phase == 1 {
                    f.expr = Some(self.expr_return());
                    self.expect(Kind::Ident("in"), "in")?;
                    f.other = Some(self.expression(shared::Method::Or)?);
                    if self.skip(Kind::Ident("if"))? {
                        f.filter = Some(self.expression(shared::Method::Expr)?);
                    }
                    f.flag = self.skip(Kind::Ident("recursive"))?;
                    self.expect(Kind::BlockEnd, "end of block")?;
                    f.phase = 2;
                    return Ok(Action::Call(f, Frame::body(End::For)));
                }
                if f.phase == 2 {
                    f.body = Some(self.body());
                    if self.skip(Kind::Ident("else"))? {
                        self.expect(Kind::BlockEnd, "end of block")?;
                        f.phase = 3;
                        return Ok(Action::Call(f, Frame::body(End::Named("endfor"))));
                    }
                    f.otherwise = Some(self.store.body(0));
                } else {
                    f.otherwise = Some(self.body());
                }
                self.input.next()?;
                *self.state.in_loop = f.old_loop;
                Ok(Action::Return(Output::Value(Statement::For {
                    target: f.expr.take().unwrap(),
                    iter: f.other.take().unwrap(),
                    filter: f.filter.take(),
                    recursive: f.flag,
                    body: f.body.take().unwrap(),
                    otherwise: f.otherwise.take().unwrap(),
                })))
            }
            If => {
                if f.phase == 0 {
                    f.expr = Some(self.expression(shared::Method::Or)?);
                    self.expect(Kind::BlockEnd, "end of block")?;
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::body(End::If)));
                }
                if f.phase == 1 {
                    f.body = Some(self.body());
                    match self.input.next()? {
                        Some((Item::Simple(Kind::Ident("else")), _)) => {
                            self.expect(Kind::BlockEnd, "end of block")?;
                            f.phase = 2;
                            return Ok(Action::Call(f, Frame::body(End::Named("endif"))));
                        }
                        Some((Item::Simple(Kind::Ident("elif")), span)) => {
                            f.inner_span = span;
                            f.phase = 3;
                            return Ok(Action::Call(f, Frame::new(If)));
                        }
                        _ => f.otherwise = Some(self.store.body(0)),
                    }
                } else if f.phase == 2 {
                    f.otherwise = Some(self.body());
                    self.input.next()?;
                } else {
                    let value = self.value();
                    let stmt = self
                        .store
                        .statement(value, self.input.expand_span(f.inner_span))?;
                    let mut body = self.store.body(1);
                    self.store.push_statement(&mut body, stmt)?;
                    f.otherwise = Some(body);
                }
                Ok(Action::Return(Output::Value(Statement::If {
                    test: f.expr.take().unwrap(),
                    yes: f.body.take().unwrap(),
                    no: f.otherwise.take().unwrap(),
                })))
            }
            With => {
                if f.phase == 0 {
                    f.bindings = Some(self.store.bindings(2, false));
                    f.phase = 1;
                }
                if f.phase == 3 {
                    let body = self.body();
                    self.input.next()?;
                    return Ok(Action::Return(Output::Value(Statement::With(
                        f.bindings.take().unwrap(),
                        body,
                    ))));
                }
                if f.phase == 2 {
                    f.expr = Some(self.expr_return());
                    self.expect(Kind::ParenClose, "`)`")?;
                    self.with_binding(&mut f)?;
                    f.phase = 1;
                }
                if self.peek()? == Some(Kind::BlockEnd) {
                    self.expect(Kind::BlockEnd, "end of block")?;
                    f.phase = 3;
                    return Ok(Action::Call(f, Frame::body(End::Named("endwith"))));
                }
                if self.store.bindings_len(f.bindings.as_ref().unwrap()) != 0 {
                    self.expect(Kind::Comma, "comma")?;
                }
                if self.skip(Kind::ParenOpen)? {
                    f.phase = 2;
                    return Ok(Action::Call(f, Frame::assignment(false)));
                }
                f.expr = Some(self.assign_name(false)?);
                self.with_binding(&mut f)?;
                Ok(Action::Continue(f))
            }
            Set => {
                if f.phase == 0 {
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::assignment(true)));
                }
                if f.phase == 2 {
                    let body = self.body();
                    self.input.next()?;
                    return Ok(Action::Return(Output::Value(Statement::SetBlock(
                        f.expr.take().unwrap(),
                        f.filter.take(),
                        body,
                    ))));
                }
                f.expr = Some(self.expr_return());
                if matches!(self.peek()?, Some(Kind::BlockEnd | Kind::Pipe)) {
                    if self.skip(Kind::Pipe)? {
                        f.filter = Some(self.filter_chain()?);
                    }
                    self.expect(Kind::BlockEnd, "end of block")?;
                    f.phase = 2;
                    return Ok(Action::Call(f, Frame::body(End::Named("endset"))));
                }
                self.expect(Kind::Assign, "assignment operator")?;
                let mut expr = self.expression(shared::Method::Expr)?;
                if self.skip(Kind::Comma)? {
                    let span = self.input.current_span();
                    let mut items = self.store.exprs(1);
                    self.store.push_expr(&mut items, expr)?;
                    loop {
                        if self.peek()? == Some(Kind::BlockEnd) {
                            break;
                        }
                        let item = self.expression(shared::Method::Expr)?;
                        self.store.push_expr(&mut items, item)?;
                        if !self.skip(Kind::Comma)? {
                            break;
                        }
                    }
                    expr = self
                        .store
                        .node(Node::List(items), self.input.expand_span(span))?;
                }
                Ok(Action::Return(Output::Value(Statement::Set(
                    f.expr.take().unwrap(),
                    expr,
                ))))
            }
            AutoEscape | FilterBlock => {
                if f.phase == 0 {
                    f.expr = Some(if f.method == AutoEscape {
                        self.expression(shared::Method::Expr)?
                    } else {
                        self.filter_chain()?
                    });
                    self.expect(Kind::BlockEnd, "end of block")?;
                    let end = if f.method == AutoEscape {
                        "endautoescape"
                    } else {
                        "endfilter"
                    };
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::body(End::Named(end))));
                }
                let body = self.body();
                self.input.next()?;
                let expr = f.expr.take().unwrap();
                Ok(Action::Return(Output::Value(if f.method == AutoEscape {
                    Statement::AutoEscape(expr, body)
                } else {
                    Statement::FilterBlock(expr, body)
                })))
            }
            #[cfg(feature = "multi_template")]
            Block => {
                if f.phase == 0 {
                    if *self.state.in_macro {
                        return Err(self.store.syntax(SyntaxFailure::Static(
                            "block tags in macros are not allowed",
                        )));
                    }
                    f.old_loop = std::mem::replace(self.state.in_loop, false);
                    let (name, _) = self.ident()?;
                    f.name = Some(name);
                    self.skip(Kind::Ident("scoped"))?;
                    f.flag = self.skip(Kind::Ident("required"))?;
                    if !self.names.insert(name)? {
                        return Err(self.store.syntax(SyntaxFailure::DuplicateBlock(name)));
                    }
                    self.expect(Kind::BlockEnd, "end of block")?;
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::body(End::Named("endblock"))));
                }
                let body = self.body();
                self.input.next()?;
                if f.flag && !self.store.body_is_whitespace(&body) {
                    return Err(self.store.syntax(SyntaxFailure::Static(
                        "Required blocks can only contain comments or whitespace",
                    )));
                }
                let name = f.name.unwrap();
                if let Some(Kind::Ident(actual)) = self.peek()? {
                    if actual != name {
                        return Err(self.store.syntax(SyntaxFailure::BlockName {
                            actual,
                            expected: name,
                        }));
                    }
                    self.input.next()?;
                }
                *self.state.in_loop = f.old_loop;
                Ok(Action::Return(Output::Value(Statement::Block {
                    name,
                    required: f.flag,
                    body,
                })))
            }
            #[cfg(feature = "macros")]
            Macro | CallBlock => {
                if f.phase == 0 {
                    let mut args = self.store.exprs(4);
                    let mut defaults = self.store.exprs(4);
                    if f.method == Macro {
                        let (name, _) = self.ident()?;
                        f.name = Some(name);
                        self.expect(Kind::ParenOpen, "`(`")?;
                        self.macro_args(&mut args, &mut defaults)?;
                    } else {
                        if self.skip(Kind::ParenOpen)? {
                            self.macro_args(&mut args, &mut defaults)?;
                        }
                        let expr = self.expression(shared::Method::Expr)?;
                        f.call = Some(self.store.call(expr)?);
                    }
                    f.exprs = Some(args);
                    f.defaults = Some(defaults);
                    self.expect(Kind::BlockEnd, "end of block")?;
                    f.old_loop = std::mem::replace(self.state.in_loop, false);
                    f.old_macro = std::mem::replace(self.state.in_macro, true);
                    let end = if f.method == Macro {
                        "endmacro"
                    } else {
                        "endcall"
                    };
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::body(End::Named(end))));
                }
                let body = self.body();
                *self.state.in_macro = f.old_macro;
                *self.state.in_loop = f.old_loop;
                self.input.next()?;
                let args = f.exprs.take().unwrap();
                let defaults = f.defaults.take().unwrap();
                Ok(Action::Return(Output::Value(if f.method == Macro {
                    Statement::Macro {
                        name: f.name.unwrap(),
                        args,
                        defaults,
                        body,
                    }
                } else {
                    Statement::CallBlock {
                        call: f.call.take().unwrap(),
                        args,
                        defaults,
                        body,
                        macro_span: self.input.expand_span(f.span),
                    }
                })))
            }
        }
    }
    fn finish_assignment(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        let items = f.exprs.take().unwrap();
        let expr = if !f.flag && self.store.exprs_len(&items) == 1 {
            self.store.first_expr(items)
        } else {
            self.store
                .node(Node::List(items), self.input.expand_span(f.span))?
        };
        Ok(Action::Return(Output::Expr(expr)))
    }
    fn with_binding(&mut self, f: &mut Frame<'s, S>) -> Result<(), S::Error> {
        self.expect(Kind::Assign, "assignment operator")?;
        let expr = self.expression(shared::Method::Expr)?;
        self.store.push_binding(
            f.bindings.as_mut().unwrap(),
            f.expr.take().unwrap(),
            Some(expr),
        )
    }
}
