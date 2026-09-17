#![forbid(unsafe_code)]
use super::{Buffer, Cause};
use crate::bounded::expression::store::{self as expr, NodeId, OperandKind, RecordKind, Sequence};
use crate::compiler::ast::CompareOpKind;
use crate::compiler::parser::statements::storage::{Names, Statement, StatementStore};
use crate::compiler::parser::storage::{Arg, Node, Store, SyntaxFailure, Text, TextRef};
use crate::compiler::tokens::Span;

#[derive(Clone, Copy)]
pub(super) struct StmtId(pub(super) usize);
pub(super) struct Record<'s> {
    pub(super) value: Statement<'s, Storage<'s>>,
    pub(super) span: Span,
}
pub(super) struct Link {
    pub(super) stmt: StmtId,
    pub(super) next: Option<usize>,
}
pub(super) struct Storage<'s> {
    pub(super) expressions: expr::Storage<'s>,
    pub(super) statements: Vec<Record<'s>>,
    pub(super) links: Vec<Link>,
    pub(super) statement_limit: usize,
    pub(super) link_limit: usize,
}
impl<'s> Storage<'s> {
    pub(super) fn new(events: usize, statements: usize) -> Self {
        Self {
            expressions: expr::Storage::new(events),
            statements: Vec::new(),
            links: Vec::new(),
            statement_limit: statements,
            link_limit: events,
        }
    }
}
impl<'s> Store<'s> for Storage<'s> {
    type Expr = NodeId;
    type Exprs = Sequence;
    type Args = Sequence;
    type Comparisons = Sequence;
    type Text = expr::Bytes;
    type Concat = expr::Joined;
    type Error = Cause<'s>;
    fn node(&mut self, value: Node<'s, Self>, span: Span) -> Result<NodeId, Cause<'s>> {
        // Associated payload types are identical; this changes only the private
        // storage type parameter and always calls the existing node worker.
        let value = match value {
            Node::Var(v) => Node::Var(v),
            Node::Const(v) => Node::Const(v),
            Node::Slice {
                expr,
                start,
                stop,
                step,
            } => Node::Slice {
                expr,
                start,
                stop,
                step,
            },
            Node::Unary(a, b) => Node::Unary(a, b),
            Node::Binary(a, b, c) => Node::Binary(a, b, c),
            Node::Compare(a, b) => Node::Compare(a, b),
            Node::If(a, b, c) => Node::If(a, b, c),
            Node::Filter(a, b, c) => Node::Filter(a, b, c),
            Node::Test(a, b, c) => Node::Test(a, b, c),
            Node::Attr(a, b) => Node::Attr(a, b),
            Node::Item(a, b) => Node::Item(a, b),
            Node::Call(a, b) => Node::Call(a, b),
            Node::List(v) => Node::List(v),
            Node::Map(a, b) => Node::Map(a, b),
        };
        self.expressions.node(value, span)
    }
    fn variable(&self, value: &NodeId) -> Option<(&'s str, Span)> {
        self.expressions.variable(value)
    }
    fn exprs(&mut self, hint: usize) -> Sequence {
        self.expressions.exprs(hint)
    }
    fn exprs_len(&self, values: &Sequence) -> usize {
        self.expressions.exprs_len(values)
    }
    fn push_expr(&mut self, values: &mut Sequence, expr: NodeId) -> Result<(), Cause<'s>> {
        self.expressions.push_expr(values, expr)
    }
    fn args(&mut self) -> Sequence {
        self.expressions.args()
    }
    fn args_len(&self, args: &Sequence) -> usize {
        self.expressions.args_len(args)
    }
    fn push_arg(&mut self, args: &mut Sequence, arg: Arg<'s, NodeId>) -> Result<(), Cause<'s>> {
        self.expressions.push_arg(args, arg)
    }
    fn comparisons(&mut self) -> Sequence {
        self.expressions.comparisons()
    }
    fn comparisons_len(&self, values: &Sequence) -> usize {
        self.expressions.comparisons_len(values)
    }
    fn push_comparison(
        &mut self,
        values: &mut Sequence,
        op: CompareOpKind,
        expr: NodeId,
    ) -> Result<(), Cause<'s>> {
        self.expressions.push_comparison(values, op, expr)
    }
    fn pop_comparison(&mut self, values: &mut Sequence) -> (CompareOpKind, NodeId) {
        self.expressions.pop_comparison(values)
    }
    fn begin_text(&mut self, text: Text<'s, expr::Bytes>) -> Result<expr::Joined, Cause<'s>> {
        self.expressions.begin_text(text)
    }
    fn append_text(
        &mut self,
        joined: &mut expr::Joined,
        text: TextRef<'_, 's, expr::Bytes>,
    ) -> Result<(), Cause<'s>> {
        self.expressions.append_text(joined, text)
    }
    fn syntax(&self, failure: SyntaxFailure<'s>) -> Cause<'s> {
        self.expressions.syntax(failure)
    }
}
impl<'s> StatementStore<'s> for Storage<'s> {
    type Stmt = StmtId;
    type Body = Sequence;
    type Bindings = Sequence;
    type Call = NodeId;
    fn statement(&mut self, value: Statement<'s, Self>, span: Span) -> Result<StmtId, Cause<'s>> {
        if self.statements.len() >= self.statement_limit
            || self.statements.len() == self.statements.capacity()
        {
            return Err(Cause::Capacity(Buffer::Statements));
        }
        let id = StmtId(self.statements.len());
        self.statements.push(Record { value, span });
        Ok(id)
    }
    fn body(&mut self, _hint: usize) -> Sequence {
        Sequence::default()
    }
    fn push_statement(&mut self, body: &mut Sequence, stmt: StmtId) -> Result<(), Cause<'s>> {
        if self.links.len() >= self.link_limit || self.links.len() == self.links.capacity() {
            return Err(Cause::Capacity(Buffer::Bodies));
        }
        let id = self.links.len();
        self.links.push(Link { stmt, next: None });
        if let Some(tail) = body.tail {
            self.links[tail].next = Some(id);
        } else {
            body.head = Some(id);
        }
        body.tail = Some(id);
        body.len += 1;
        Ok(())
    }
    fn body_is_whitespace(&self, body: &Sequence) -> bool {
        let mut index = body.head;
        while let Some(i) = index {
            let link = &self.links[i];
            if !matches!(self.statements[link.stmt.0].value,Statement::EmitRaw(raw) if raw.trim().is_empty())
            {
                return false;
            }
            index = link.next;
        }
        true
    }
    fn first_expr(&mut self, values: Sequence) -> NodeId {
        assert_eq!(values.len, 1);
        match self.expressions.operands[values.head.expect("assignment item")].kind {
            OperandKind::Expr(id) => id,
            _ => unreachable!("assignment operand"),
        }
    }
    fn bindings(&mut self, _hint: usize, _optional: bool) -> Sequence {
        Sequence::default()
    }
    fn bindings_len(&self, values: &Sequence) -> usize {
        values.len
    }
    fn push_binding(
        &mut self,
        values: &mut Sequence,
        target: NodeId,
        value: Option<NodeId>,
    ) -> Result<(), Cause<'s>> {
        self.expressions
            .operand(values, OperandKind::Binding(target, value))
    }
    fn call(&mut self, expr: NodeId) -> Result<NodeId, Cause<'s>> {
        let description = match &self.expressions.nodes[expr.0].kind {
            RecordKind::Call(..) => return Ok(expr),
            RecordKind::Var(..) => "variable",
            RecordKind::Const(..) => "constant",
            RecordKind::List(..) => "list literal",
            RecordKind::Map(..) => "map literal",
            RecordKind::Test(..) => "test expression",
            RecordKind::Filter(..) => "filter expression",
            _ => "expression",
        };
        Err(self.syntax(SyntaxFailure::ExpectedCall(description)))
    }
}

pub(super) struct BlockNames<'s> {
    pub(super) values: Vec<&'s str>,
    pub(super) limit: usize,
}
impl<'s> Names<'s, Cause<'s>> for BlockNames<'s> {
    fn insert(&mut self, name: &'s str) -> Result<bool, Cause<'s>> {
        if self.values.contains(&name) {
            return Ok(false);
        }
        if self.values.len() >= self.limit || self.values.len() == self.values.capacity() {
            return Err(Cause::Capacity(Buffer::BlockNames));
        }
        self.values.push(name);
        Ok(true)
    }
}
