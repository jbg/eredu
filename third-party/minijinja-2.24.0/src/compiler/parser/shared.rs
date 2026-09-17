//! The ordinary expression control graph, expressed as explicit continuations.
#![forbid(unsafe_code)]

use super::storage::*;
use crate::compiler::ast::{BinOpKind as B, CompareOpKind as C, UnaryOpKind as U};
use crate::compiler::tokens::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Expr,
    If,
    Or,
    And,
    Not,
    Compare,
    Math1,
    Concat,
    Math2,
    Pow,
    Unary,
    UnaryOnly,
    Primary,
    List,
    Map,
    Tuple,
    Postfix,
    Filter,
    Args,
}

/// Longest path in the enumerated zero-consumption Call graph. This is not
/// a caller limit; each consumed-edge segment is separately charged to a token.
pub(crate) const ZERO_RUN_FRAMES: usize = 15;

/// Concrete movable worker shells, separate from the frame Vec and input.
/// This inventories typed controls, not compiler-generated machine stack bytes.
pub(crate) fn control_bytes<'s, S: Store<'s>, I>() -> usize {
    std::mem::size_of::<Engine<'_, 's, S, I>>()
        + std::mem::size_of::<Frame<'s, S>>()
        + std::mem::size_of::<Action<'s, S>>()
        + std::mem::size_of::<Result<Output<'s, S>, S::Error>>()
}

pub(crate) struct Frame<'s, S: Store<'s>> {
    pub(crate) method: Method,
    phase: u8,
    span: Span,
    next_span: Span,
    expr: Option<S::Expr>,
    other: Option<S::Expr>,
    start: Option<S::Expr>,
    stop: Option<S::Expr>,
    step: Option<S::Expr>,
    exprs: Option<S::Exprs>,
    values: Option<S::Exprs>,
    args: Option<S::Args>,
    comparisons: Option<S::Comparisons>,
    binary: Option<B>,
    comparison: Option<C>,
    name: Option<&'s str>,
    flag: bool,
    flag2: bool,
    arg_kind: u8,
}

impl<'s, S: Store<'s>> Frame<'s, S> {
    pub(crate) fn new(method: Method) -> Self {
        Self {
            method,
            phase: 0,
            span: Span::default(),
            next_span: Span::default(),
            expr: None,
            other: None,
            start: None,
            stop: None,
            step: None,
            exprs: None,
            values: None,
            args: None,
            comparisons: None,
            binary: None,
            comparison: None,
            name: None,
            flag: false,
            flag2: false,
            arg_kind: 0,
        }
    }
    fn located(method: Method, span: Span) -> Self {
        let mut frame = Self::new(method);
        frame.span = span;
        frame
    }
    fn with_expr(method: Method, expr: S::Expr, span: Span) -> Self {
        let mut frame = Self::located(method, span);
        frame.expr = Some(expr);
        frame
    }
}

pub(crate) enum Output<'s, S: Store<'s>> {
    Expr(S::Expr),
    Args(S::Args),
}

pub(crate) trait Stack<'s, S: Store<'s>> {
    fn push(&mut self, frame: Frame<'s, S>) -> Result<(), S::Error>;
    fn pop(&mut self) -> Option<Frame<'s, S>>;
}

impl<'s, S: Store<'s>> Stack<'s, S> for Vec<Frame<'s, S>> {
    fn push(&mut self, frame: Frame<'s, S>) -> Result<(), S::Error> {
        Vec::push(self, frame);
        Ok(())
    }
    fn pop(&mut self) -> Option<Frame<'s, S>> {
        Vec::pop(self)
    }
}

enum Action<'s, S: Store<'s>> {
    Continue(Frame<'s, S>),
    Call(Frame<'s, S>, Frame<'s, S>),
    Return(Output<'s, S>),
}

struct Engine<'a, 's, S: Store<'s>, I> {
    input: &'a mut I,
    store: &'a mut S,
    depth: &'a mut usize,
    active_guards: usize,
    returned: Option<Output<'s, S>>,
}

pub(crate) fn run<'s, S, I, W>(
    method: Method,
    input: &mut I,
    store: &mut S,
    stack: &mut W,
    depth: &mut usize,
) -> Result<Output<'s, S>, S::Error>
where
    S: Store<'s>,
    I: Input<'s, Text = S::Text, Error = S::Error>,
    W: Stack<'s, S>,
{
    stack.push(Frame::new(method))?;
    let mut engine = Engine {
        input,
        store,
        depth,
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
        Ok(engine.returned.take().expect("expression worker return"))
    })();
    if result.is_err() {
        // Successful guard entries decrement while errors propagate in the
        // ordinary macro. Its overflowing entry increments before returning,
        // without decrementing; that entry is deliberately not active here.
        *engine.depth -= engine.active_guards;
        while stack.pop().is_some() {}
    }
    result
}

pub(crate) fn filter_name<'s, S, I>(
    input: &mut I,
    store: &mut S,
) -> Result<(&'s str, Span), S::Error>
where
    S: Store<'s>,
    I: Input<'s, Text = S::Text, Error = S::Error>,
{
    let mut depth = 0;
    Engine {
        input,
        store,
        depth: &mut depth,
        active_guards: 0,
        returned: None,
    }
    .filter_name()
}

impl<'s, S, I> Engine<'_, 's, S, I>
where
    S: Store<'s>,
    I: Input<'s, Text = S::Text, Error = S::Error>,
{
    fn expr(&mut self) -> S::Expr {
        match self.returned.take().expect("expression return") {
            Output::Expr(expr) => expr,
            _ => unreachable!("expression return kind"),
        }
    }
    fn args(&mut self) -> S::Args {
        match self.returned.take().expect("argument return") {
            Output::Args(args) => args,
            _ => unreachable!("argument return kind"),
        }
    }
    fn enter_guard(&mut self) -> Result<(), S::Error> {
        *self.depth += 1;
        if *self.depth > super::MAX_RECURSION {
            return Err(self.store.syntax(SyntaxFailure::Static(
                "template exceeds maximum recursion limits",
            )));
        }
        self.active_guards += 1;
        Ok(())
    }
    fn leave_guard(&mut self) {
        *self.depth -= 1;
        self.active_guards -= 1;
    }
    fn peek(&mut self) -> Result<Option<Kind<'s>>, S::Error> {
        self.input
            .current()
            .map(|value| value.map(|(kind, _)| kind))
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
    fn filter_name(&mut self) -> Result<(&'s str, Span), S::Error> {
        let (first, span) = self.ident()?;
        let mut end = span.end_offset as usize;
        let mut dotted = false;
        while self.skip(Kind::Dot)? {
            let (_, segment) = self.ident()?;
            end = segment.end_offset as usize;
            dotted = true;
        }
        Ok(if dotted {
            (
                &self.input.source()[span.start_offset as usize..end],
                self.input.expand_span(span),
            )
        } else {
            (first, span)
        })
    }
    fn finish(&mut self, node: Node<'s, S>, span: Span) -> Result<Action<'s, S>, S::Error> {
        self.store
            .node(node, span)
            .map(|expr| Action::Return(Output::Expr(expr)))
    }
    fn binary_method(method: Method) -> Option<Method> {
        Some(match method {
            Method::Or => Method::And,
            Method::And => Method::Not,
            Method::Math1 => Method::Concat,
            Method::Concat => Method::Math2,
            Method::Math2 => Method::Pow,
            Method::Pow => Method::Unary,
            _ => return None,
        })
    }
    fn binary_op(method: Method, token: Option<Kind<'s>>) -> Option<B> {
        match (method, token) {
            (Method::Or, Some(Kind::Ident("or"))) => Some(B::ScOr),
            (Method::And, Some(Kind::Ident("and"))) => Some(B::ScAnd),
            (Method::Math1, Some(Kind::Plus)) => Some(B::Add),
            (Method::Math1, Some(Kind::Minus)) => Some(B::Sub),
            (Method::Concat, Some(Kind::Tilde)) => Some(B::Concat),
            (Method::Math2, Some(Kind::Mul)) => Some(B::Mul),
            (Method::Math2, Some(Kind::Div)) => Some(B::Div),
            (Method::Math2, Some(Kind::FloorDiv)) => Some(B::FloorDiv),
            (Method::Math2, Some(Kind::Mod)) => Some(B::Rem),
            (Method::Pow, Some(Kind::Pow)) => Some(B::Pow),
            _ => None,
        }
    }
    fn step(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        if let Some(child) = Self::binary_method(f.method) {
            if f.phase == 0 {
                f.span = self.input.current_span();
                f.phase = 1;
                return Ok(Action::Call(f, Frame::new(child)));
            }
            let right = self.expr();
            f.expr = Some(if f.phase == 2 {
                self.store.node(
                    Node::Binary(f.binary.take().unwrap(), f.expr.take().unwrap(), right),
                    self.input.expand_span(f.span),
                )?
            } else {
                right
            });
            if let Some(op) = Self::binary_op(f.method, self.peek()?) {
                self.input.next()?;
                f.binary = Some(op);
                f.phase = 2;
                return Ok(Action::Call(f, Frame::new(child)));
            }
            return Ok(Action::Return(Output::Expr(f.expr.take().unwrap())));
        }
        match f.method {
            Method::Expr => {
                if f.phase == 0 {
                    self.enter_guard()?;
                    f.phase = 1;
                    Ok(Action::Call(f, Frame::new(Method::If)))
                } else {
                    self.leave_guard();
                    Ok(Action::Return(Output::Expr(self.expr())))
                }
            }
            Method::Not | Method::UnaryOnly => {
                if f.phase == 0 {
                    f.span = self.input.current_span();
                    let token = if f.method == Method::Not {
                        Kind::Ident("not")
                    } else {
                        Kind::Minus
                    };
                    if self.skip(token)? {
                        f.phase = 1;
                        let child = Frame::new(f.method);
                        Ok(Action::Call(f, child))
                    } else {
                        f.phase = 2;
                        let child = if f.method == Method::Not {
                            Method::Compare
                        } else {
                            Method::Primary
                        };
                        Ok(Action::Call(f, Frame::new(child)))
                    }
                } else {
                    let expr = self.expr();
                    if f.phase == 1 {
                        let op = if f.method == Method::Not {
                            U::Not
                        } else {
                            U::Neg
                        };
                        self.finish(Node::Unary(op, expr), self.input.expand_span(f.span))
                    } else {
                        Ok(Action::Return(Output::Expr(expr)))
                    }
                }
            }
            Method::Unary => match f.phase {
                0 => {
                    f.span = self.input.current_span();
                    f.phase = 1;
                    Ok(Action::Call(f, Frame::new(Method::UnaryOnly)))
                }
                1 => {
                    let child = Frame::with_expr(Method::Postfix, self.expr(), f.span);
                    f.phase = 2;
                    Ok(Action::Call(f, child))
                }
                2 => {
                    let child = Frame::with_expr(Method::Filter, self.expr(), Span::default());
                    f.phase = 3;
                    Ok(Action::Call(f, child))
                }
                _ => Ok(Action::Return(Output::Expr(self.expr()))),
            },
            Method::If => self.if_expr(f),
            Method::Compare => self.compare(f),
            Method::Primary => self.primary(f),
            Method::List | Method::Map | Method::Tuple => self.collection(f),
            Method::Postfix => self.postfix(f),
            Method::Filter => self.filter(f),
            Method::Args => self.arguments(f),
            _ => unreachable!("binary method handled above"),
        }
    }

    fn if_expr(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        match f.phase {
            0 => {
                f.span = self.input.last_span();
                f.phase = 1;
                return Ok(Action::Call(f, Frame::new(Method::Or)));
            }
            1 => f.expr = Some(self.expr()),
            2 => {
                f.other = Some(self.expr());
                if self.skip(Kind::Ident("else"))? {
                    f.phase = 3;
                    return Ok(Action::Call(f, Frame::new(Method::If)));
                }
                let node = Node::If(f.other.take().unwrap(), f.expr.take().unwrap(), None);
                f.expr = Some(self.store.node(node, self.input.expand_span(f.span))?);
                f.span = self.input.last_span();
            }
            3 => {
                let node = Node::If(
                    f.other.take().unwrap(),
                    f.expr.take().unwrap(),
                    Some(self.expr()),
                );
                f.expr = Some(self.store.node(node, self.input.expand_span(f.span))?);
                f.span = self.input.last_span();
            }
            _ => unreachable!(),
        }
        if self.skip(Kind::Ident("if"))? {
            f.phase = 2;
            Ok(Action::Call(f, Frame::new(Method::Or)))
        } else {
            Ok(Action::Return(Output::Expr(f.expr.take().unwrap())))
        }
    }

    fn compare(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        if f.phase == 0 {
            f.span = self.input.last_span();
            f.phase = 1;
            return Ok(Action::Call(f, Frame::new(Method::Math1)));
        }
        let expr = self.expr();
        if f.phase == 1 {
            f.expr = Some(expr);
            f.comparisons = Some(self.store.comparisons());
        } else {
            self.store.push_comparison(
                f.comparisons.as_mut().unwrap(),
                f.comparison.take().unwrap(),
                expr,
            )?;
        }
        let op = match self.peek()? {
            Some(Kind::Eq) => Some(C::Eq),
            Some(Kind::Ne) => Some(C::Ne),
            Some(Kind::Lt) => Some(C::Lt),
            Some(Kind::Lte) => Some(C::Lte),
            Some(Kind::Gt) => Some(C::Gt),
            Some(Kind::Gte) => Some(C::Gte),
            Some(Kind::Ident("in")) => Some(C::In),
            Some(Kind::Ident("not")) => {
                self.input.next()?;
                self.expect(Kind::Ident("in"), "in")?;
                Some(C::NotIn)
            }
            _ => None,
        };
        if let Some(op) = op {
            if !matches!(op, C::NotIn) {
                self.input.next()?;
            }
            f.comparison = Some(op);
            f.phase = 2;
            return Ok(Action::Call(f, Frame::new(Method::Math1)));
        }
        let mut ops = f.comparisons.take().unwrap();
        let expr = f.expr.take().unwrap();
        let span = self.input.expand_span(f.span);
        match self.store.comparisons_len(&ops) {
            0 => Ok(Action::Return(Output::Expr(expr))),
            1 => {
                let (op, right) = self.store.pop_comparison(&mut ops);
                let (binary, negated) = match op {
                    C::Eq => (B::Eq, false),
                    C::Ne => (B::Ne, false),
                    C::Lt => (B::Lt, false),
                    C::Lte => (B::Lte, false),
                    C::Gt => (B::Gt, false),
                    C::Gte => (B::Gte, false),
                    C::In => (B::In, false),
                    C::NotIn => (B::In, true),
                };
                let expr = self.store.node(Node::Binary(binary, expr, right), span)?;
                if negated {
                    self.finish(Node::Unary(U::Not, expr), span)
                } else {
                    Ok(Action::Return(Output::Expr(expr)))
                }
            }
            _ => self.finish(Node::Compare(expr, ops), span),
        }
    }

    fn primary(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        if f.phase != 0 {
            self.leave_guard();
            return Ok(Action::Return(Output::Expr(self.expr())));
        }
        self.enter_guard()?;
        let (token, span) = self
            .input
            .next()?
            .ok_or_else(|| self.store.syntax(SyntaxFailure::Eof("expression")))?;
        f.span = span;
        let value = match token {
            Item::Simple(Kind::Ident("true" | "True")) => Literal::Bool(true),
            Item::Simple(Kind::Ident("false" | "False")) => Literal::Bool(false),
            Item::Simple(Kind::Ident("none" | "None")) => Literal::None,
            Item::Simple(Kind::Ident(name)) => {
                let result = self.finish(Node::Var(name), span)?;
                self.leave_guard();
                return Ok(result);
            }
            Item::Str(value) if !matches!(self.peek(), Ok(Some(kind)) if kind.is_string()) => {
                Literal::Plain(value)
            }
            Item::Str(_) | Item::String(_) => {
                let first = match token {
                    Item::Str(value) => Text::Source(value),
                    Item::String(value) => Text::Decoded(value),
                    _ => unreachable!(),
                };
                let mut joined = self.store.begin_text(first)?;
                while let Some(text) = self.input.current_text()? {
                    self.store.append_text(&mut joined, text)?;
                    self.input.next()?;
                }
                Literal::Joined(joined)
            }
            Item::Int(value) => Literal::Int(value),
            Item::Int128(value) => Literal::Int128(value),
            Item::Float(value) => Literal::Float(value),
            Item::Simple(Kind::ParenOpen | Kind::BracketOpen | Kind::BraceOpen) => {
                let method = match token.kind() {
                    Kind::ParenOpen => Method::Tuple,
                    Kind::BracketOpen => Method::List,
                    _ => Method::Map,
                };
                f.phase = 1;
                return Ok(Action::Call(f, Frame::located(method, span)));
            }
            value => {
                return Err(self
                    .store
                    .syntax(SyntaxFailure::UnexpectedOnly(value.kind())))
            }
        };
        let result = self.finish(Node::Const(value), self.input.expand_span(span))?;
        self.leave_guard();
        Ok(result)
    }

    fn collection(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        if f.method == Method::Tuple {
            match f.phase {
                0 => {
                    if self.skip(Kind::ParenClose)? {
                        let items = self.store.exprs(0);
                        return self.finish(Node::List(items), self.input.expand_span(f.span));
                    }
                    f.phase = 1;
                    return Ok(Action::Call(f, Frame::new(Method::Expr)));
                }
                1 => {
                    let expr = self.expr();
                    if self.peek()? != Some(Kind::Comma) {
                        self.expect(Kind::ParenClose, "`)`")?;
                        return Ok(Action::Return(Output::Expr(expr)));
                    }
                    let mut items = self.store.exprs(1);
                    self.store.push_expr(&mut items, expr)?;
                    f.exprs = Some(items);
                }
                2 => {
                    let expr = self.expr();
                    self.store.push_expr(f.exprs.as_mut().unwrap(), expr)?;
                }
                _ => unreachable!(),
            }
            if self.skip(Kind::ParenClose)? {
                return self.finish(
                    Node::List(f.exprs.take().unwrap()),
                    self.input.expand_span(f.span),
                );
            }
            self.expect(Kind::Comma, "`,`")?;
            if self.skip(Kind::ParenClose)? {
                return self.finish(
                    Node::List(f.exprs.take().unwrap()),
                    self.input.expand_span(f.span),
                );
            }
            f.phase = 2;
            return Ok(Action::Call(f, Frame::new(Method::Expr)));
        }
        if f.phase == 0 {
            f.exprs = Some(self.store.exprs(4));
            if f.method == Method::Map {
                f.values = Some(self.store.exprs(4));
            }
        } else if f.phase == 1 {
            let expr = self.expr();
            self.store.push_expr(f.exprs.as_mut().unwrap(), expr)?;
            if f.method == Method::Map {
                self.expect(Kind::Colon, "`:`")?;
                f.phase = 2;
                return Ok(Action::Call(f, Frame::new(Method::Expr)));
            }
        } else {
            let expr = self.expr();
            self.store.push_expr(f.values.as_mut().unwrap(), expr)?;
        }
        let end = if f.method == Method::List {
            Kind::BracketClose
        } else {
            Kind::BraceClose
        };
        let mut done = self.skip(end)?;
        if !done && self.store.exprs_len(f.exprs.as_ref().unwrap()) != 0 {
            self.expect(Kind::Comma, "`,`")?;
            done = self.skip(end)?;
        }
        if done {
            let node = if f.method == Method::List {
                Node::List(f.exprs.take().unwrap())
            } else {
                Node::Map(f.exprs.take().unwrap(), f.values.take().unwrap())
            };
            return self.finish(node, self.input.expand_span(f.span));
        }
        f.phase = 1;
        Ok(Action::Call(f, Frame::new(Method::Expr)))
    }

    fn postfix(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        // 1=start, 2=stop, 3=step, 4=call arguments; 5 resumes a slice
        // after its opening bracket was consumed, with no initial expression.
        match f.phase {
            1 => f.start = Some(self.expr()),
            2 => f.stop = Some(self.expr()),
            3 => f.step = Some(self.expr()),
            4 => {
                let args = self.args();
                f.expr = Some(self.store.node(
                    Node::Call(f.expr.take().unwrap(), args),
                    self.input.expand_span(f.span),
                )?);
                f.span = f.next_span;
                f.phase = 0;
            }
            _ => (),
        }
        if matches!(f.phase, 1 | 2 | 3 | 5) {
            if f.phase == 1 || f.phase == 5 {
                if self.skip(Kind::Colon)? {
                    f.flag = true;
                    if !matches!(self.peek()?, Some(Kind::BracketClose | Kind::Colon)) {
                        f.phase = 2;
                        return Ok(Action::Call(f, Frame::new(Method::Expr)));
                    }
                }
            }
            if f.flag
                && f.phase != 3
                && self.skip(Kind::Colon)?
                && self.peek()? != Some(Kind::BracketClose)
            {
                f.phase = 3;
                return Ok(Action::Call(f, Frame::new(Method::Expr)));
            }
            self.expect(Kind::BracketClose, "`]`")?;
            let node = if f.flag {
                Node::Slice {
                    expr: f.expr.take().unwrap(),
                    start: f.start.take(),
                    stop: f.stop.take(),
                    step: f.step.take(),
                }
            } else {
                let subscript = f
                    .start
                    .take()
                    .ok_or_else(|| self.store.syntax(SyntaxFailure::Static("empty subscript")))?;
                Node::Item(f.expr.take().unwrap(), subscript)
            };
            f.expr = Some(self.store.node(node, self.input.expand_span(f.span))?);
            f.span = f.next_span;
            f.phase = 0;
        }
        f.next_span = self.input.current_span();
        match self.peek()? {
            Some(Kind::Dot) => {
                self.input.next()?;
                let (item, span) = self.input.next()?.ok_or_else(|| {
                    self.store
                        .syntax(SyntaxFailure::Eof("identifier or integer"))
                })?;
                let node = match item {
                    Item::Simple(Kind::Ident(name)) => Node::Attr(f.expr.take().unwrap(), name),
                    Item::Int(value) => {
                        let subscript = self.store.node(Node::Const(Literal::Int(value)), span)?;
                        Node::Item(f.expr.take().unwrap(), subscript)
                    }
                    Item::Int128(value) => {
                        let subscript =
                            self.store.node(Node::Const(Literal::Int128(value)), span)?;
                        Node::Item(f.expr.take().unwrap(), subscript)
                    }
                    item => {
                        return Err(self.store.syntax(SyntaxFailure::Unexpected(
                            item.kind(),
                            "identifier or integer",
                        )))
                    }
                };
                f.expr = Some(self.store.node(node, self.input.expand_span(f.span))?);
                f.span = f.next_span;
                Ok(Action::Continue(f))
            }
            Some(Kind::BracketOpen) => {
                self.input.next()?;
                f.start = None;
                f.stop = None;
                f.step = None;
                f.flag = false;
                if self.peek()? != Some(Kind::Colon) {
                    f.phase = 1;
                    Ok(Action::Call(f, Frame::new(Method::Expr)))
                } else {
                    f.phase = 5;
                    Ok(Action::Continue(f))
                }
            }
            Some(Kind::ParenOpen) => {
                f.phase = 4;
                Ok(Action::Call(f, Frame::new(Method::Args)))
            }
            _ => Ok(Action::Return(Output::Expr(f.expr.take().unwrap()))),
        }
    }

    fn filter_finish(&mut self, f: &mut Frame<'s, S>, args: S::Args) -> Result<(), S::Error> {
        let span = self.input.expand_span(f.span);
        let name = f.name.take().unwrap();
        let expr = f.expr.take().unwrap();
        let node = if f.flag {
            Node::Test(name, expr, args)
        } else {
            Node::Filter(name, Some(expr), args)
        };
        let mut expr = self.store.node(node, span)?;
        if f.flag2 {
            expr = self.store.node(Node::Unary(U::Not, expr), span)?;
        }
        f.expr = Some(expr);
        f.phase = 0;
        Ok(())
    }

    fn filter(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        match f.phase {
            1 => {
                let args = self.args();
                self.filter_finish(&mut f, args)?;
            }
            2 => {
                let child = Frame::with_expr(Method::Postfix, self.expr(), f.next_span);
                f.phase = 3;
                return Ok(Action::Call(f, child));
            }
            3 => {
                let expr = self.expr();
                let mut args = self.store.args();
                self.store.push_arg(&mut args, Arg::Pos(expr))?;
                self.filter_finish(&mut f, args)?;
            }
            _ => (),
        }
        match self.peek()? {
            Some(Kind::Pipe) => {
                self.input.next()?;
                f.flag = false;
                f.flag2 = false;
            }
            Some(Kind::Ident("is")) => {
                self.input.next()?;
                f.flag = true;
                f.flag2 = self.skip(Kind::Ident("not"))?;
            }
            _ => return Ok(Action::Return(Output::Expr(f.expr.take().unwrap()))),
        }
        let (name, span) = self.filter_name()?;
        f.name = Some(name);
        f.span = span;
        if self.peek()? == Some(Kind::ParenOpen) {
            f.phase = 1;
            return Ok(Action::Call(f, Frame::new(Method::Args)));
        }
        if f.flag
            && matches!(
                self.peek()?,
                Some(
                    Kind::Ident(_)
                        | Kind::Str
                        | Kind::String
                        | Kind::Int
                        | Kind::Int128
                        | Kind::Float
                        | Kind::Plus
                        | Kind::Minus
                        | Kind::BracketOpen
                        | Kind::BraceOpen
                )
            )
            && !matches!(
                self.peek()?,
                Some(Kind::Ident("and" | "or" | "else" | "is"))
            )
        {
            f.next_span = self.input.current_span();
            f.phase = 2;
            return Ok(Action::Call(f, Frame::new(Method::UnaryOnly)));
        }
        let args = self.store.args();
        self.filter_finish(&mut f, args)?;
        Ok(Action::Continue(f))
    }

    fn arguments(&mut self, mut f: Frame<'s, S>) -> Result<Action<'s, S>, S::Error> {
        if f.phase == 0 {
            f.args = Some(self.store.args());
            self.expect(Kind::ParenOpen, "`(`")?;
        } else if f.phase == 1 {
            let expr = self.expr();
            let arg = match f.arg_kind {
                0 => {
                    if let Some((name, _span)) = self.store.variable(&expr) {
                        if self.skip(Kind::Assign)? {
                            f.flag = true;
                            f.name = Some(name);
                            f.phase = 2;
                            return Ok(Action::Call(f, Frame::new(Method::Expr)));
                        }
                    }
                    if f.flag {
                        return Err(self
                            .store
                            .syntax(SyntaxFailure::Static("non-keyword arg after keyword arg")));
                    }
                    Arg::Pos(expr)
                }
                1 => Arg::PosSplat(expr),
                _ => {
                    f.flag = true;
                    Arg::KwargSplat(expr)
                }
            };
            self.store.push_arg(f.args.as_mut().unwrap(), arg)?;
        } else {
            let expr = self.expr();
            self.store.push_arg(
                f.args.as_mut().unwrap(),
                Arg::Kwarg(f.name.take().unwrap(), expr),
            )?;
        }
        if self.store.args_len(f.args.as_ref().unwrap()) > 2000 {
            return Err(self
                .store
                .syntax(SyntaxFailure::Static("Too many arguments in function call")));
        }
        if self.skip(Kind::ParenClose)? {
            return Ok(Action::Return(Output::Args(f.args.take().unwrap())));
        }
        if self.store.args_len(f.args.as_ref().unwrap()) != 0 || f.flag {
            self.expect(Kind::Comma, "`,`")?;
            if self.skip(Kind::ParenClose)? {
                return Ok(Action::Return(Output::Args(f.args.take().unwrap())));
            }
        }
        f.arg_kind = if self.skip(Kind::Pow)? {
            2
        } else if self.skip(Kind::Mul)? {
            1
        } else {
            0
        };
        f.phase = 1;
        Ok(Action::Call(f, Frame::new(Method::Expr)))
    }
}
