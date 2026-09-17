//! Read-only source views. IDs never cross this private owner boundary.
#![forbid(unsafe_code)]
use super::super::{store::StmtId, ParsedTemplate};
use crate::bounded::expression::store::{NodeId, OperandKind as O, RecordKind as R, Sequence};
use crate::compiler::meta::view::{Expression as E, Item, MacroParts, Statement as S, View};
use crate::compiler::parser::statements::storage::Statement;

#[derive(Clone, Copy)]
pub(super) struct Packed<'o, 's>(pub(super) &'o ParsedTemplate<'s>);
#[derive(Clone, Copy)]
pub(super) enum Cursor {
    Body(Option<usize>),
    Operand(Option<usize>),
    Map(Option<usize>, Option<usize>),
}
fn body(v: Sequence) -> Cursor {
    Cursor::Body(v.head)
}
fn operands(v: Sequence) -> Cursor {
    Cursor::Operand(v.head)
}

impl<'o, 's: 'o> Packed<'o, 's> {
    fn call(self, id: NodeId) -> (NodeId, Cursor) {
        match self.0.storage.expressions.nodes[id.0].kind {
            R::Call(receiver, args) => (receiver, operands(args)),
            _ => unreachable!("source parser admitted a call"),
        }
    }
}
impl<'o, 's: 'o> View<'s> for Packed<'o, 's> {
    type Expr = NodeId;
    type Stmt = StmtId;
    type Macro = StmtId;
    type Cursor = Cursor;
    fn expression(self, id: NodeId) -> E<'s, Self> {
        match &self.0.storage.expressions.nodes[id.0].kind {
            R::Var(name) => E::Var(name),
            R::Const(_) => E::Const,
            R::Unary(_, expr) => E::Unary(*expr),
            R::Binary(_, left, right) => E::Binary(*left, *right),
            R::Compare(root, rest) => E::Compare(*root, operands(*rest)),
            R::If(test, yes, no) => E::If(*test, *yes, *no),
            R::Filter(_, root, args) => E::Filter(*root, operands(*args)),
            R::Test(_, root, args) => E::Test(*root, operands(*args)),
            R::Attr(root, name) => E::Attr(*root, name),
            R::Item(root, index) => E::Item(*root, *index),
            R::Slice {
                start, stop, step, ..
            } => E::Slice(*start, *stop, *step),
            R::Call(root, args) => E::Call(*root, operands(*args)),
            R::List(items) => E::List(operands(*items)),
            R::Map(keys, values) => E::Map(Cursor::Map(keys.head, values.head)),
        }
    }
    fn statement(self, id: StmtId) -> S<'s, Self> {
        match &self.0.storage.statements[id.0].value {
            Statement::Template(items) => S::Template(body(*items)),
            Statement::EmitExpr(expr) => S::EmitExpr(*expr),
            Statement::EmitRaw(_) => S::EmitRaw,
            Statement::For {
                target,
                iter,
                filter,
                body: items,
                otherwise,
                ..
            } => S::For {
                target: *target,
                iter: *iter,
                filter: *filter,
                body: body(*items),
                otherwise: body(*otherwise),
            },
            Statement::If { test, yes, no } => S::If(*test, body(*yes), body(*no)),
            Statement::With(bindings, items) => S::With(operands(*bindings), body(*items)),
            Statement::Set(target, value) => S::Set(*target, *value),
            Statement::SetBlock(target, _, items) => S::SetBlock(*target, body(*items)),
            Statement::AutoEscape(_, items) => S::AutoEscape(body(*items)),
            Statement::FilterBlock(_, items) => S::FilterBlock(body(*items)),
            #[cfg(feature = "multi_template")]
            Statement::Block { body: items, .. } => S::Block(body(*items)),
            #[cfg(feature = "multi_template")]
            Statement::Extends(_) => S::Extends,
            #[cfg(feature = "multi_template")]
            Statement::Include(..) => S::Include,
            #[cfg(feature = "multi_template")]
            Statement::Import(_, name) => S::Import(*name),
            #[cfg(feature = "multi_template")]
            Statement::FromImport(_, names) => S::FromImport(operands(*names)),
            Statement::Macro { name, .. } => S::Macro(name, id),
            Statement::CallBlock { call, .. } => {
                let (receiver, args) = self.call(*call);
                S::CallBlock(receiver, args, id)
            }
            #[cfg(feature = "loop_controls")]
            Statement::Continue => S::Continue,
            #[cfg(feature = "loop_controls")]
            Statement::Break => S::Break,
            Statement::Do(call) => {
                let (receiver, args) = self.call(*call);
                S::Do(receiver, args)
            }
        }
    }
    fn macro_parts(self, id: StmtId) -> MacroParts<'s, Self> {
        match &self.0.storage.statements[id.0].value {
            Statement::Macro {
                args,
                defaults,
                body: items,
                ..
            }
            | Statement::CallBlock {
                args,
                defaults,
                body: items,
                ..
            } => MacroParts {
                args: operands(*args),
                defaults: operands(*defaults),
                body: body(*items),
            },
            _ => unreachable!("private source macro handle"),
        }
    }
    fn next(self, cursor: Cursor) -> Option<(Item<'s, Self>, Cursor)> {
        Some(match cursor {
            Cursor::Body(index) => {
                let entry = &self.0.storage.links[index?];
                (Item::Stmt(entry.stmt), Cursor::Body(entry.next))
            }
            Cursor::Operand(index) => {
                let entry = &self.0.storage.expressions.operands[index?];
                let item = match entry.kind {
                    O::Expr(id)
                    | O::Pos(id)
                    | O::Kwarg(_, id)
                    | O::PosSplat(id)
                    | O::KwargSplat(id)
                    | O::Compare(_, id) => Item::Expr(id),
                    O::Binding(target, value) => Item::Pair(target, value),
                };
                (item, Cursor::Operand(entry.next))
            }
            Cursor::Map(key, value) => {
                let key = &self.0.storage.expressions.operands[key?];
                let value = &self.0.storage.expressions.operands[value?];
                let (O::Expr(k), O::Expr(v)) = (&key.kind, &value.kind) else {
                    unreachable!("source map operands")
                };
                (Item::Pair(*k, Some(*v)), Cursor::Map(key.next, value.next))
            }
        })
    }
}
