//! Growable ordinary storage and the existing Value operation/materialization boundary.
#![forbid(unsafe_code)]
use super::{Continuation, Node, Store, View};
use crate::compiler::ast::{self, BinOpKind, CompareOpKind, Expr};
use crate::value::{ops, value_map_with_capacity, Value};
use std::convert::Infallible;

impl<'a, 's: 'a> View for OrdinaryView<'a, 's> {
    type Expr = &'a Expr<'s>;
    type Items = &'a [Expr<'s>];
    type Comparisons = &'a [ast::CompareOp<'s>];
    fn node(self, expr: Self::Expr) -> Node<Self::Expr, Self::Items, Self::Comparisons> {
        match expr {
            Expr::Const(_) => Node::Constant(expr),
            Expr::UnaryOp(c) => Node::Unary(
                match c.op {
                    ast::UnaryOpKind::Not => super::Unary::Not,
                    ast::UnaryOpKind::Neg => super::Unary::Neg,
                },
                &c.expr,
            ),
            Expr::BinOp(c) => Node::Binary(c.op, &c.left, &c.right),
            Expr::Compare(c) => Node::Compare(&c.expr, &c.ops),
            Expr::List(l) => Node::List(&l.items),
            Expr::Map(m) => Node::Map(&m.keys, &m.values),
            _ => Node::Other,
        }
    }
    fn next_item(self, items: Self::Items) -> Option<(Self::Expr, Self::Items)> {
        items.split_first()
    }
    fn next_comparison(
        self,
        ops: Self::Comparisons,
    ) -> Option<(CompareOpKind, Self::Expr, Self::Comparisons)> {
        ops.split_first()
            .map(|(first, tail)| (first.op, &first.expr, tail))
    }
}
#[derive(Clone, Copy)]
struct OrdinaryView<'a, 's>(std::marker::PhantomData<&'a Expr<'s>>);
struct Storage<'a, 's> {
    frames: Vec<Continuation<OrdinaryView<'a, 's>, Value>>,
}
impl Drop for Storage<'_, '_> {
    fn drop(&mut self) {
        // Match recursive ordinary unwinding: inner active parents retire first.
        while let Some(frame) = self.frames.pop() {
            drop(frame);
        }
    }
}
impl<'a, 's: 'a> Store<OrdinaryView<'a, 's>> for Storage<'a, 's> {
    type Value = Value;
    type Error = Infallible;
    fn push(&mut self, frame: Continuation<OrdinaryView<'a, 's>, Value>) -> Result<(), Infallible> {
        self.frames.push(frame);
        Ok(())
    }
    fn pop(&mut self) -> Option<Continuation<OrdinaryView<'a, 's>, Value>> {
        self.frames.pop()
    }
    fn constant(&mut self, expr: &'a Expr<'s>) -> Value {
        match expr {
            Expr::Const(c) => c.value.clone(),
            _ => unreachable!("shared direct-constant selection"),
        }
    }
    fn boolean(&mut self, value: bool) -> Value {
        Value::from(value)
    }
    fn not(&mut self, value: &Value) -> Result<Value, Infallible> {
        Ok(Value::from(!value.is_true()))
    }
    fn neg(&mut self, value: &Value) -> Result<Option<Value>, Infallible> {
        Ok(ops::neg(value).ok())
    }
    fn binary(
        &mut self,
        op: BinOpKind,
        left: &Value,
        right: &Value,
    ) -> Result<Option<Value>, Infallible> {
        Ok(ast::eval_binop(op, left, right))
    }
    fn compare(
        &mut self,
        op: CompareOpKind,
        left: &Value,
        right: &Value,
    ) -> Result<Option<bool>, Infallible> {
        Ok(ast::eval_compare(op, left, right).map(|v| v.is_true()))
    }
    fn list(&mut self, items: &'a [Expr<'s>]) -> Result<Value, Infallible> {
        // Preserve the existing iterator, cloning and Vec/Value construction.
        let sequence = items.iter().filter_map(|expr| match expr {
            Expr::Const(v) => Some(v.value.clone()),
            _ => None,
        });
        Ok(Value::from(sequence.collect::<Vec<_>>()))
    }
    fn map(&mut self, keys: &'a [Expr<'s>], values: &'a [Expr<'s>]) -> Result<Value, Infallible> {
        let mut rv = value_map_with_capacity(keys.len());
        for (key, value) in keys.iter().zip(values.iter()) {
            if let (Expr::Const(maybe_key), Expr::Const(value)) = (key, value) {
                rv.insert(maybe_key.value.clone(), value.value.clone());
            }
        }
        Ok(Value::from_object(rv))
    }
}
fn run<'a, 's: 'a>(
    node: Node<&'a Expr<'s>, &'a [Expr<'s>], &'a [ast::CompareOp<'s>]>,
) -> Option<Value> {
    let mut store = Storage { frames: Vec::new() };
    match super::run(OrdinaryView(std::marker::PhantomData), &mut store, node) {
        Ok(value) => value,
        Err(error) => match error {},
    }
}
pub(crate) fn expression(expr: &Expr<'_>) -> Option<Value> {
    let view = OrdinaryView(std::marker::PhantomData);
    run(view.node(expr))
}
pub(crate) fn list(list: &ast::List<'_>) -> Option<Value> {
    run(Node::List(&list.items))
}
pub(crate) fn map(map: &ast::Map<'_>) -> Option<Value> {
    run(Node::Map(&map.keys, &map.values))
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Object;
    use std::sync::{Arc, Mutex};
    #[derive(Debug)]
    struct Dropped(&'static str, Arc<Mutex<Vec<&'static str>>>);
    impl Object for Dropped {}
    impl Drop for Dropped {
        fn drop(&mut self) {
            self.1.lock().unwrap().push(self.0)
        }
    }
    #[test]
    fn ordinary_parent_value_custody_unwinds_in_recursive_lifo_order() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut store = Storage::<'static, 'static> { frames: Vec::new() };
            store
                .push(super::super::Frame::BinaryRight(
                    BinOpKind::Add,
                    Some(Value::from_object(Dropped("outer", events.clone()))),
                ))
                .unwrap();
            store
                .push(super::super::Frame::CompareRight(
                    CompareOpKind::Eq,
                    &[],
                    Value::from_object(Dropped("inner", events.clone())),
                ))
                .unwrap();
            panic!("ordinary callback unwind");
        }));
        assert!(result.is_err());
        assert_eq!(*events.lock().unwrap(), vec!["inner", "outer"]);
    }
}
