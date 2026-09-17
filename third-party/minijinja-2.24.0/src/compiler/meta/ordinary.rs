//! Ordinary AST/HashSet adapter. Hash-state construction and insertion stay here.
#![forbid(unsafe_code)]
use super::view::{self, Expression as E, Item, Statement as S, Task, View};
use crate::compiler::ast;
use std::{collections::HashSet, convert::Infallible, fmt::Write, marker::PhantomData};

#[derive(Clone, Copy)]
pub(crate) struct Ordinary<'t, 's>(PhantomData<(&'t (), &'s ())>);
impl Ordinary<'_, '_> {
    pub(crate) fn new() -> Self {
        Self(PhantomData)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Cursor<'t, 's> {
    Body(&'t [ast::Stmt<'s>]),
    Exprs(&'t [ast::Expr<'s>]),
    Args(&'t [ast::CallArg<'s>]),
    Comparisons(&'t [ast::CompareOp<'s>]),
    Map(&'t [ast::Expr<'s>], &'t [ast::Expr<'s>]),
    Bindings(&'t [(ast::Expr<'s>, ast::Expr<'s>)]),
    #[cfg(feature = "multi_template")]
    Imports(&'t [(ast::Expr<'s>, Option<ast::Expr<'s>>)]),
}

impl<'t, 's: 't> View<'s> for Ordinary<'t, 's> {
    type Expr = &'t ast::Expr<'s>;
    type Stmt = &'t ast::Stmt<'s>;
    #[cfg(feature = "macros")]
    type Macro = &'t ast::Macro<'s>;
    type Cursor = Cursor<'t, 's>;
    fn expression(self, expr: Self::Expr) -> E<'s, Self> {
        match expr {
            ast::Expr::Var(v) => E::Var(v.id),
            ast::Expr::Const(_) => E::Const,
            ast::Expr::UnaryOp(v) => E::Unary(&v.expr),
            ast::Expr::BinOp(v) => E::Binary(&v.left, &v.right),
            ast::Expr::Compare(v) => E::Compare(&v.expr, Cursor::Comparisons(&v.ops)),
            ast::Expr::IfExpr(v) => E::If(&v.test_expr, &v.true_expr, v.false_expr.as_ref()),
            ast::Expr::Filter(v) => E::Filter(v.expr.as_ref(), Cursor::Args(&v.args)),
            ast::Expr::Test(v) => E::Test(&v.expr, Cursor::Args(&v.args)),
            ast::Expr::GetAttr(v) => E::Attr(&v.expr, v.name),
            ast::Expr::GetItem(v) => E::Item(&v.expr, &v.subscript_expr),
            ast::Expr::Slice(v) => E::Slice(v.start.as_ref(), v.stop.as_ref(), v.step.as_ref()),
            ast::Expr::Call(v) => E::Call(&v.expr, Cursor::Args(&v.args)),
            ast::Expr::List(v) => E::List(Cursor::Exprs(&v.items)),
            ast::Expr::Map(v) => E::Map(Cursor::Map(&v.keys, &v.values)),
        }
    }
    fn statement(self, stmt: Self::Stmt) -> S<'s, Self> {
        match stmt {
            ast::Stmt::Template(v) => S::Template(Cursor::Body(&v.children)),
            ast::Stmt::EmitExpr(v) => S::EmitExpr(&v.expr),
            ast::Stmt::EmitRaw(_) => S::EmitRaw,
            ast::Stmt::ForLoop(v) => S::For {
                target: &v.target,
                iter: &v.iter,
                filter: v.filter_expr.as_ref(),
                body: Cursor::Body(&v.body),
                otherwise: Cursor::Body(&v.else_body),
            },
            ast::Stmt::IfCond(v) => S::If(
                &v.expr,
                Cursor::Body(&v.true_body),
                Cursor::Body(&v.false_body),
            ),
            ast::Stmt::WithBlock(v) => {
                S::With(Cursor::Bindings(&v.assignments), Cursor::Body(&v.body))
            }
            ast::Stmt::Set(v) => S::Set(&v.target, &v.expr),
            ast::Stmt::AutoEscape(v) => S::AutoEscape(Cursor::Body(&v.body)),
            ast::Stmt::FilterBlock(v) => S::FilterBlock(Cursor::Body(&v.body)),
            ast::Stmt::SetBlock(v) => S::SetBlock(&v.target, Cursor::Body(&v.body)),
            #[cfg(feature = "multi_template")]
            ast::Stmt::Block(v) => S::Block(Cursor::Body(&v.body)),
            #[cfg(feature = "multi_template")]
            ast::Stmt::Extends(_) => S::Extends,
            #[cfg(feature = "multi_template")]
            ast::Stmt::Include(_) => S::Include,
            #[cfg(feature = "multi_template")]
            ast::Stmt::Import(v) => S::Import(&v.name),
            #[cfg(feature = "multi_template")]
            ast::Stmt::FromImport(v) => S::FromImport(Cursor::Imports(&v.names)),
            #[cfg(feature = "macros")]
            ast::Stmt::Macro(v) => S::Macro(v.name, &**v),
            #[cfg(feature = "macros")]
            ast::Stmt::CallBlock(v) => {
                S::CallBlock(&v.call.expr, Cursor::Args(&v.call.args), &*v.macro_decl)
            }
            #[cfg(feature = "loop_controls")]
            ast::Stmt::Continue(_) => S::Continue,
            #[cfg(feature = "loop_controls")]
            ast::Stmt::Break(_) => S::Break,
            ast::Stmt::Do(v) => S::Do(&v.call.expr, Cursor::Args(&v.call.args)),
        }
    }
    #[cfg(feature = "macros")]
    fn macro_parts(self, value: Self::Macro) -> view::MacroParts<'s, Self> {
        view::MacroParts {
            args: Cursor::Exprs(&value.args),
            defaults: Cursor::Exprs(&value.defaults),
            body: Cursor::Body(&value.body),
        }
    }
    fn next(self, cursor: Self::Cursor) -> Option<(Item<'s, Self>, Self::Cursor)> {
        Some(match cursor {
            Cursor::Body(values) => {
                let (first, rest) = values.split_first()?;
                (Item::Stmt(first), Cursor::Body(rest))
            }
            Cursor::Exprs(values) => {
                let (first, rest) = values.split_first()?;
                (Item::Expr(first), Cursor::Exprs(rest))
            }
            Cursor::Args(values) => {
                let (first, rest) = values.split_first()?;
                let (ast::CallArg::Pos(v)
                | ast::CallArg::Kwarg(_, v)
                | ast::CallArg::PosSplat(v)
                | ast::CallArg::KwargSplat(v)) = first;
                (Item::Expr(v), Cursor::Args(rest))
            }
            Cursor::Comparisons(values) => {
                let (first, rest) = values.split_first()?;
                (Item::Expr(&first.expr), Cursor::Comparisons(rest))
            }
            Cursor::Map(keys, values) => {
                let (key, keys) = keys.split_first()?;
                let (value, values) = values.split_first()?;
                (Item::Pair(key, Some(value)), Cursor::Map(keys, values))
            }
            Cursor::Bindings(values) => {
                let ((key, value), rest) = values.split_first()?;
                (Item::Pair(key, Some(value)), Cursor::Bindings(rest))
            }
            #[cfg(feature = "multi_template")]
            Cursor::Imports(values) => {
                let ((key, value), rest) = values.split_first()?;
                (Item::Pair(key, value.as_ref()), Cursor::Imports(rest))
            }
        })
    }
}

pub(super) struct State<'t, 's: 't> {
    pub(super) out: HashSet<&'s str>,
    pub(super) nested_out: Option<HashSet<String>>,
    assigned: Vec<HashSet<&'s str>>,
    tasks: Vec<Task<'s, Ordinary<'t, 's>>>,
}
impl<'t, 's: 't> State<'t, 's> {
    pub(super) fn new(nested: bool) -> Self {
        // Preserve the original HashSet creation order. The explicit traversal
        // vector creates no hash state and begins empty.
        Self {
            out: HashSet::new(),
            nested_out: if nested { Some(HashSet::new()) } else { None },
            assigned: vec![Default::default()],
            tasks: Vec::new(),
        }
    }
    fn assign_nested(&mut self, name: String) {
        if let Some(ref mut nested_out) = self.nested_out {
            if !nested_out.contains(&name) {
                nested_out.insert(name);
            }
        }
    }
}
impl<'t, 's: 't> view::Storage<'s, Ordinary<'t, 's>> for State<'t, 's> {
    type Error = Infallible;
    type Nested = Vec<&'s str>;
    fn push_task(&mut self, task: Task<'s, Ordinary<'t, 's>>) -> Result<(), Infallible> {
        self.tasks.push(task);
        Ok(())
    }
    fn pop_task(&mut self) -> Option<Task<'s, Ordinary<'t, 's>>> {
        self.tasks.pop()
    }
    fn is_assigned(&self, name: &str) -> bool {
        self.assigned.iter().any(|x| x.contains(name))
    }
    fn assign(&mut self, name: &'s str) -> Result<(), Infallible> {
        self.assigned.last_mut().unwrap().insert(name);
        Ok(())
    }
    fn capture(&mut self, name: &'s str) -> Result<(), Infallible> {
        self.out.insert(name);
        Ok(())
    }
    fn push_scope(&mut self) -> Result<(), Infallible> {
        self.assigned.push(Default::default());
        Ok(())
    }
    fn pop_scope(&mut self) -> Result<(), Infallible> {
        self.assigned.pop();
        Ok(())
    }
    fn tracks_nested(&self) -> bool {
        self.nested_out.is_some()
    }
    fn nested_start(&mut self, attr: &'s str) -> Result<Self::Nested, Infallible> {
        Ok(vec![attr])
    }
    fn nested_attr(&mut self, attrs: &mut Self::Nested, attr: &'s str) -> Result<(), Infallible> {
        attrs.push(attr);
        Ok(())
    }
    fn nested_finish(&mut self, attrs: Self::Nested, root: &'s str) -> Result<(), Infallible> {
        let mut name = root.to_string();
        for attr in attrs.iter().rev() {
            write!(name, ".{attr}").ok();
        }
        self.assign_nested(name);
        Ok(())
    }
    fn nested_variable(&mut self, name: &'s str) -> Result<(), Infallible> {
        self.assign_nested(name.to_string());
        Ok(())
    }
}
