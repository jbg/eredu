//! Private syntax projection; no caller-selected values or IDs are admitted.
#![forbid(unsafe_code)]
use super::super::ParsedTemplate;
use crate::bounded::expression::store::{NodeId, OperandKind, RecordKind, Sequence};
use crate::compiler::{
    ast::CompareOpKind,
    fold::{Node, View},
};
#[derive(Clone, Copy)]
pub(super) struct Packed<'o, 's>(pub(super) &'o ParsedTemplate<'s>);
impl<'o, 's: 'o> View for Packed<'o, 's> {
    type Expr = NodeId;
    type Items = Sequence;
    type Comparisons = Sequence;
    fn node(self, expr: NodeId) -> Node<NodeId, Sequence, Sequence> {
        match &self.0.storage.expressions.nodes[expr.0].kind {
            RecordKind::Const(_) => Node::Constant(expr),
            RecordKind::Unary(op, child) => Node::Unary(
                match op {
                    crate::compiler::ast::UnaryOpKind::Not => crate::compiler::fold::Unary::Not,
                    crate::compiler::ast::UnaryOpKind::Neg => crate::compiler::fold::Unary::Neg,
                },
                *child,
            ),
            RecordKind::Binary(op, left, right) => Node::Binary(*op, *left, *right),
            RecordKind::Compare(left, rest) => Node::Compare(*left, *rest),
            RecordKind::List(items) => Node::List(*items),
            RecordKind::Map(keys, values) => Node::Map(*keys, *values),
            _ => Node::Other,
        }
    }
    fn next_item(self, mut items: Sequence) -> Option<(NodeId, Sequence)> {
        let item = &self.0.storage.expressions.operands[items.head?];
        let OperandKind::Expr(expr) = item.kind else {
            unreachable!("private parsed collection")
        };
        items.head = item.next;
        Some((expr, items))
    }
    fn next_comparison(self, mut rest: Sequence) -> Option<(CompareOpKind, NodeId, Sequence)> {
        let item = &self.0.storage.expressions.operands[rest.head?];
        let OperandKind::Compare(op, expr) = item.kind else {
            unreachable!("private parsed comparison")
        };
        rest.head = item.next;
        Some((op, expr, rest))
    }
}
