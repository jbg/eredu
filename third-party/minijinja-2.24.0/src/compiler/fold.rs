//! The ordinary constant-fold decisions over borrowed syntax and explicit storage.
#![forbid(unsafe_code)]
use super::ast::{BinOpKind, CompareOpKind};
#[derive(Clone, Copy)]
pub(crate) enum Unary {
    Not,
    Neg,
}
pub(crate) mod ordinary;

#[derive(Clone, Copy)]
pub(crate) enum Node<E, I, C> {
    Constant(E),
    Unary(Unary, E),
    Binary(BinOpKind, E, E),
    Compare(E, C),
    List(I),
    Map(I, I),
    Other,
}
pub(crate) trait View: Copy {
    type Expr: Copy;
    type Items: Copy;
    type Comparisons: Copy;
    fn node(self, expr: Self::Expr) -> Node<Self::Expr, Self::Items, Self::Comparisons>;
    fn next_item(self, items: Self::Items) -> Option<(Self::Expr, Self::Items)>;
    fn next_comparison(
        self,
        comparisons: Self::Comparisons,
    ) -> Option<(CompareOpKind, Self::Expr, Self::Comparisons)>;
}
pub(crate) enum Frame<E, C, V> {
    Unary(Unary),
    BinaryLeft(BinOpKind, E),
    BinaryRight(BinOpKind, Option<V>),
    CompareStart(C),
    CompareRight(CompareOpKind, C, V),
}
pub(crate) type Continuation<S, V> = Frame<<S as View>::Expr, <S as View>::Comparisons, V>;
pub(crate) trait Store<S: View> {
    type Value;
    type Error;
    fn push(&mut self, frame: Continuation<S, Self::Value>) -> Result<(), Self::Error>;
    fn pop(&mut self) -> Option<Continuation<S, Self::Value>>;
    fn constant(&mut self, expr: S::Expr) -> Self::Value;
    fn boolean(&mut self, value: bool) -> Self::Value;
    fn not(&mut self, value: &Self::Value) -> Result<Self::Value, Self::Error>;
    fn neg(&mut self, value: &Self::Value) -> Result<Option<Self::Value>, Self::Error>;
    fn binary(
        &mut self,
        op: BinOpKind,
        left: &Self::Value,
        right: &Self::Value,
    ) -> Result<Option<Self::Value>, Self::Error>;
    fn compare(
        &mut self,
        op: CompareOpKind,
        left: &Self::Value,
        right: &Self::Value,
    ) -> Result<Option<bool>, Self::Error>;
    fn list(&mut self, items: S::Items) -> Result<Self::Value, Self::Error>;
    fn map(&mut self, keys: S::Items, values: S::Items) -> Result<Self::Value, Self::Error>;
}
fn all_constants<S: View>(source: S, mut items: S::Items) -> bool {
    while let Some((expr, rest)) = source.next_item(items) {
        if !matches!(source.node(expr), Node::Constant(_)) {
            return false;
        }
        items = rest;
    }
    true
}

/// One parent continuation per active child descent; item eligibility does not recurse.
pub(crate) fn run<S: View, T: Store<S>>(
    source: S,
    store: &mut T,
    root: Node<S::Expr, S::Items, S::Comparisons>,
) -> Result<Option<T::Value>, T::Error> {
    let mut next = Some(root);
    let mut value = None;
    loop {
        if let Some(node) = next.take() {
            value = match node {
                Node::Constant(expr) => Some(store.constant(expr)),
                Node::Other => None,
                Node::List(items) => {
                    if all_constants(source, items) {
                        Some(store.list(items)?)
                    } else {
                        None
                    }
                }
                Node::Map(keys, values) => {
                    // Preserve the ordinary keys-first, short-circuit eligibility scan.
                    if all_constants(source, keys) && all_constants(source, values) {
                        Some(store.map(keys, values)?)
                    } else {
                        None
                    }
                }
                Node::Unary(op, expr) => {
                    store.push(Frame::Unary(op))?;
                    next = Some(source.node(expr));
                    continue;
                }
                Node::Binary(op, left, right) => {
                    store.push(Frame::BinaryLeft(op, right))?;
                    next = Some(source.node(left));
                    continue;
                }
                Node::Compare(expr, rest) => {
                    store.push(Frame::CompareStart(rest))?;
                    next = Some(source.node(expr));
                    continue;
                }
            };
        }
        match store.pop() {
            None => return Ok(value),
            Some(Frame::Unary(op)) => {
                value = match value {
                    Some(v) => match op {
                        Unary::Not => Some(store.not(&v)?),
                        Unary::Neg => store.neg(&v)?,
                    },
                    None => None,
                };
            }
            Some(Frame::BinaryLeft(op, right)) => {
                // Unlike Compare, BinOp always folds the right, even if the left is None.
                store.push(Frame::BinaryRight(op, value.take()))?;
                next = Some(source.node(right));
            }
            Some(Frame::BinaryRight(op, left)) => {
                value = match (left, value) {
                    (Some(left), Some(right)) => {
                        let result = store.binary(op, &left, &right)?;
                        drop(right);
                        drop(left);
                        result
                    }
                    _ => None,
                };
            }
            Some(Frame::CompareStart(rest)) => {
                if let Some(left) = value.take() {
                    if let Some((op, right, tail)) = source.next_comparison(rest) {
                        store.push(Frame::CompareRight(op, tail, left))?;
                        next = Some(source.node(right));
                    } else {
                        value = Some(store.boolean(true));
                        drop(left);
                    }
                }
            }
            Some(Frame::CompareRight(op, rest, left)) => {
                value = match value {
                    None => None,
                    Some(right) => match store.compare(op, &left, &right)? {
                        None => {
                            drop(right);
                            None
                        }
                        Some(false) => {
                            let result = Some(store.boolean(false));
                            drop(right);
                            result
                        }
                        Some(true) => {
                            // Ordinary `left = right` drops old left before the next RHS.
                            drop(left);
                            if let Some((op, expr, tail)) = source.next_comparison(rest) {
                                value = None;
                                store.push(Frame::CompareRight(op, tail, right))?;
                                next = Some(source.node(expr));
                            } else {
                                value = Some(store.boolean(true));
                                drop(right);
                            }
                            continue;
                        }
                    },
                };
                drop(left);
            }
        }
    }
}

pub(crate) fn control_bytes<S: View, T: Store<S>>() -> usize {
    use std::mem::size_of;
    size_of::<S>()
        + size_of::<Option<Node<S::Expr, S::Items, S::Comparisons>>>()
        + size_of::<Option<Continuation<S, T::Value>>>()
        + size_of::<Option<T::Value>>()
        + 2 * size_of::<T::Value>()
        + size_of::<Option<bool>>()
        + size_of::<Result<Option<T::Value>, T::Error>>()
        + size_of::<S::Items>()
        + size_of::<Option<(S::Expr, S::Items)>>()
        + size_of::<Option<(CompareOpKind, S::Expr, S::Comparisons)>>()
}

#[cfg(test)]
pub(crate) mod reference;
#[cfg(test)]
pub(crate) mod tests;

#[cfg(test)]
pub(crate) mod scalar_reference;

#[cfg(test)]
pub(crate) mod materialization_reference;
