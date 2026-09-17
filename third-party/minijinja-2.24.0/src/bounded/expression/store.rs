#![forbid(unsafe_code)]

use super::{Buffer, Cause, SyntaxError};
use crate::compiler::ast::{BinOpKind, CompareOpKind, UnaryOpKind};
use crate::compiler::parser::storage::{Arg, Literal, Node, Store, SyntaxFailure, Text, TextRef};
use crate::compiler::tokens::Span;

#[derive(Clone, Copy)]
pub(in crate::bounded) struct NodeId(pub(in crate::bounded) usize);

#[derive(Clone, Copy)]
pub(in crate::bounded) struct Bytes {
    pub(in crate::bounded) start: usize,
    pub(in crate::bounded) end: usize,
}

#[derive(Clone, Copy, Default)]
pub(in crate::bounded) struct Sequence {
    pub(in crate::bounded) head: Option<usize>,
    pub(in crate::bounded) tail: Option<usize>,
    pub(in crate::bounded) len: usize,
}

#[derive(Clone, Copy, Default)]
pub(in crate::bounded) struct Joined {
    pub(in crate::bounded) sequence: Sequence,
    pub(in crate::bounded) bytes: usize,
}

// Retained syntax payload; the production codegen consumer is a later unit.
#[allow(dead_code)]
pub(in crate::bounded) enum Scalar {
    None,
    Bool(bool),
    Int(u64),
    Int128(u128),
    Float(f64),
    Text(Joined),
}

// Retained syntax payload; the production codegen consumer is a later unit.
#[allow(dead_code)]
pub(in crate::bounded) enum RecordKind<'s> {
    Var(&'s str),
    Const(Scalar),
    Slice {
        expr: NodeId,
        start: Option<NodeId>,
        stop: Option<NodeId>,
        step: Option<NodeId>,
    },
    Unary(UnaryOpKind, NodeId),
    Binary(BinOpKind, NodeId, NodeId),
    Compare(NodeId, Sequence),
    If(NodeId, NodeId, Option<NodeId>),
    Filter(&'s str, Option<NodeId>, Sequence),
    Test(&'s str, NodeId, Sequence),
    Attr(NodeId, &'s str),
    Item(NodeId, NodeId),
    Call(NodeId, Sequence),
    List(Sequence),
    Map(Sequence, Sequence),
}

pub(in crate::bounded) struct Record<'s> {
    pub(in crate::bounded) kind: RecordKind<'s>,
    pub(in crate::bounded) span: Span,
}

// Retained syntax payload; the production codegen consumer is a later unit.
#[allow(dead_code)]
pub(in crate::bounded) enum OperandKind<'s> {
    Expr(NodeId),
    Binding(NodeId, Option<NodeId>),
    Pos(NodeId),
    Kwarg(&'s str, NodeId),
    PosSplat(NodeId),
    KwargSplat(NodeId),
    Compare(CompareOpKind, NodeId),
}

pub(in crate::bounded) struct Operand<'s> {
    pub(in crate::bounded) kind: OperandKind<'s>,
    pub(in crate::bounded) next: Option<usize>,
}

// Retained syntax payload; the production codegen consumer is a later unit.
#[allow(dead_code)]
pub(in crate::bounded) enum SegmentKind<'s> {
    Source(&'s str),
    Decoded(Bytes),
}

#[allow(dead_code)] // Segment payload is exercised by the independent syntax visitor.
pub(in crate::bounded) struct Segment<'s> {
    pub(in crate::bounded) kind: SegmentKind<'s>,
    pub(in crate::bounded) next: Option<usize>,
}

pub(in crate::bounded) struct Storage<'s> {
    pub(in crate::bounded) nodes: Vec<Record<'s>>,
    pub(in crate::bounded) operands: Vec<Operand<'s>>,
    pub(in crate::bounded) segments: Vec<Segment<'s>>,
    pub(in crate::bounded) node_limit: usize,
    pub(in crate::bounded) operand_limit: usize,
    pub(in crate::bounded) segment_limit: usize,
}

impl<'s> Storage<'s> {
    pub(in crate::bounded) fn new(limit: usize) -> Self {
        Self {
            nodes: Vec::new(),
            operands: Vec::new(),
            segments: Vec::new(),
            node_limit: limit,
            operand_limit: limit,
            segment_limit: limit,
        }
    }
    pub(in crate::bounded) fn operand(
        &mut self,
        values: &mut Sequence,
        kind: OperandKind<'s>,
    ) -> Result<(), Cause<'s>> {
        if self.operands.len() >= self.operand_limit
            || self.operands.len() == self.operands.capacity()
        {
            return Err(Cause::Capacity(Buffer::Operands));
        }
        let index = self.operands.len();
        self.operands.push(Operand { kind, next: None });
        if let Some(tail) = values.tail {
            self.operands[tail].next = Some(index);
        } else {
            values.head = Some(index);
        }
        values.tail = Some(index);
        values.len += 1;
        Ok(())
    }
    fn segment(&mut self, joined: &mut Joined, kind: SegmentKind<'s>) -> Result<(), Cause<'s>> {
        let bytes = match &kind {
            SegmentKind::Source(value) => value.len(),
            SegmentKind::Decoded(range) => range.end - range.start,
        };
        let total = joined
            .bytes
            .checked_add(bytes)
            .ok_or(Cause::Capacity(Buffer::Segments))?;
        if self.segments.len() >= self.segment_limit
            || self.segments.len() == self.segments.capacity()
        {
            return Err(Cause::Capacity(Buffer::Segments));
        }
        let index = self.segments.len();
        self.segments.push(Segment { kind, next: None });
        if let Some(tail) = joined.sequence.tail {
            self.segments[tail].next = Some(index);
        } else {
            joined.sequence.head = Some(index);
        }
        joined.sequence.tail = Some(index);
        joined.sequence.len += 1;
        joined.bytes = total;
        Ok(())
    }
}

impl<'s> Store<'s> for Storage<'s> {
    type Expr = NodeId;
    type Exprs = Sequence;
    type Args = Sequence;
    type Comparisons = Sequence;
    type Text = Bytes;
    type Concat = Joined;
    type Error = Cause<'s>;

    fn node(&mut self, node: Node<'s, Self>, span: Span) -> Result<NodeId, Cause<'s>> {
        let kind = match node {
            Node::Var(name) => RecordKind::Var(name),
            Node::Const(value) => RecordKind::Const(match value {
                Literal::None => Scalar::None,
                Literal::Bool(value) => Scalar::Bool(value),
                Literal::Int(value) => Scalar::Int(value),
                Literal::Int128(value) => Scalar::Int128(value),
                Literal::Float(value) => Scalar::Float(value),
                Literal::Plain(value) => {
                    let mut joined = Joined::default();
                    self.segment(&mut joined, SegmentKind::Source(value))?;
                    Scalar::Text(joined)
                }
                Literal::Joined(value) => Scalar::Text(value),
            }),
            Node::Slice {
                expr,
                start,
                stop,
                step,
            } => RecordKind::Slice {
                expr,
                start,
                stop,
                step,
            },
            Node::Unary(op, expr) => RecordKind::Unary(op, expr),
            Node::Binary(op, left, right) => RecordKind::Binary(op, left, right),
            Node::Compare(expr, ops) => RecordKind::Compare(expr, ops),
            Node::If(test, yes, no) => RecordKind::If(test, yes, no),
            Node::Filter(name, expr, args) => RecordKind::Filter(name, expr, args),
            Node::Test(name, expr, args) => RecordKind::Test(name, expr, args),
            Node::Attr(expr, name) => RecordKind::Attr(expr, name),
            Node::Item(expr, subscript) => RecordKind::Item(expr, subscript),
            Node::Call(expr, args) => RecordKind::Call(expr, args),
            Node::List(values) => RecordKind::List(values),
            Node::Map(keys, values) => RecordKind::Map(keys, values),
        };
        if self.nodes.len() >= self.node_limit || self.nodes.len() == self.nodes.capacity() {
            return Err(Cause::Capacity(Buffer::Nodes));
        }
        let id = NodeId(self.nodes.len());
        self.nodes.push(Record { kind, span });
        Ok(id)
    }

    fn variable(&self, expr: &NodeId) -> Option<(&'s str, Span)> {
        match self.nodes[expr.0].kind {
            RecordKind::Var(name) => Some((name, self.nodes[expr.0].span)),
            _ => None,
        }
    }
    fn exprs(&mut self, _hint: usize) -> Sequence {
        Sequence::default()
    }
    fn exprs_len(&self, values: &Sequence) -> usize {
        values.len
    }
    fn push_expr(&mut self, values: &mut Sequence, expr: NodeId) -> Result<(), Cause<'s>> {
        self.operand(values, OperandKind::Expr(expr))
    }
    fn args(&mut self) -> Sequence {
        Sequence::default()
    }
    fn args_len(&self, values: &Sequence) -> usize {
        values.len
    }
    fn push_arg(&mut self, values: &mut Sequence, arg: Arg<'s, NodeId>) -> Result<(), Cause<'s>> {
        self.operand(
            values,
            match arg {
                Arg::Pos(expr) => OperandKind::Pos(expr),
                Arg::Kwarg(name, expr) => OperandKind::Kwarg(name, expr),
                Arg::PosSplat(expr) => OperandKind::PosSplat(expr),
                Arg::KwargSplat(expr) => OperandKind::KwargSplat(expr),
            },
        )
    }
    fn comparisons(&mut self) -> Sequence {
        Sequence::default()
    }
    fn comparisons_len(&self, values: &Sequence) -> usize {
        values.len
    }
    fn push_comparison(
        &mut self,
        values: &mut Sequence,
        op: CompareOpKind,
        expr: NodeId,
    ) -> Result<(), Cause<'s>> {
        self.operand(values, OperandKind::Compare(op, expr))
    }
    fn pop_comparison(&mut self, values: &mut Sequence) -> (CompareOpKind, NodeId) {
        assert_eq!(values.len, 1);
        let result = match self.operands[values.head.unwrap()].kind {
            OperandKind::Compare(op, expr) => (op, expr),
            _ => unreachable!("comparison list"),
        };
        // The actual record remains retained. This grammar simplification
        // cannot refund storage or create a second copy of the operand.
        *values = Sequence::default();
        result
    }
    fn begin_text(&mut self, text: Text<'s, Bytes>) -> Result<Joined, Cause<'s>> {
        let mut joined = Joined::default();
        self.segment(
            &mut joined,
            match text {
                Text::Source(value) => SegmentKind::Source(value),
                Text::Decoded(range) => SegmentKind::Decoded(range),
            },
        )?;
        Ok(joined)
    }
    fn append_text(
        &mut self,
        joined: &mut Joined,
        text: TextRef<'_, 's, Bytes>,
    ) -> Result<(), Cause<'s>> {
        self.segment(
            joined,
            match text {
                TextRef::Source(value) => SegmentKind::Source(value),
                TextRef::Decoded(range) => SegmentKind::Decoded(*range),
            },
        )
    }
    fn syntax(&self, detail: SyntaxFailure<'s>) -> Cause<'s> {
        Cause::Syntax(SyntaxError { detail, span: None })
    }
}
