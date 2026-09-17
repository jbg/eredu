//! Test-only complete AST traversal; it does not parse or compile.
#![forbid(unsafe_code)]
use crate::compiler::ast::*;
fn args(values: &[CallArg<'_>], f: &mut impl FnMut(&Expr<'_>)) {
    for arg in values {
        let e = match arg {
            CallArg::Pos(e)
            | CallArg::Kwarg(_, e)
            | CallArg::PosSplat(e)
            | CallArg::KwargSplat(e) => e,
        };
        expr(e, f)
    }
}
fn expr(value: &Expr<'_>, f: &mut impl FnMut(&Expr<'_>)) {
    f(value);
    match value {
        Expr::Const(_) | Expr::Var(_) => (),
        Expr::UnaryOp(s) => expr(&s.expr, f),
        Expr::BinOp(s) => {
            expr(&s.left, f);
            expr(&s.right, f)
        }
        Expr::Compare(s) => {
            expr(&s.expr, f);
            for v in &s.ops {
                expr(&v.expr, f)
            }
        }
        Expr::IfExpr(s) => {
            expr(&s.test_expr, f);
            expr(&s.true_expr, f);
            if let Some(e) = &s.false_expr {
                expr(e, f)
            }
        }
        Expr::Slice(s) => {
            expr(&s.expr, f);
            for v in [&s.start, &s.stop, &s.step].into_iter().flatten() {
                expr(v, f)
            }
        }
        Expr::Filter(s) => {
            if let Some(e) = &s.expr {
                expr(e, f)
            }
            args(&s.args, f)
        }
        Expr::Test(s) => {
            expr(&s.expr, f);
            args(&s.args, f)
        }
        Expr::GetAttr(s) => expr(&s.expr, f),
        Expr::GetItem(s) => {
            expr(&s.expr, f);
            expr(&s.subscript_expr, f)
        }
        Expr::Call(s) => {
            expr(&s.expr, f);
            args(&s.args, f)
        }
        Expr::List(s) => {
            for e in &s.items {
                expr(e, f)
            }
        }
        Expr::Map(s) => {
            for e in &s.keys {
                expr(e, f)
            }
            for e in &s.values {
                expr(e, f)
            }
        }
    }
}
fn body(values: &[Stmt<'_>], f: &mut impl FnMut(&Expr<'_>)) {
    for s in values {
        statement(s, f)
    }
}
#[cfg(feature = "macros")]
fn mac(value: &Macro<'_>, f: &mut impl FnMut(&Expr<'_>)) {
    for e in &value.args {
        expr(e, f)
    }
    for e in &value.defaults {
        expr(e, f)
    }
    body(&value.body, f)
}
pub(super) fn statement(value: &Stmt<'_>, f: &mut impl FnMut(&Expr<'_>)) {
    match value {
        Stmt::Template(s) => body(&s.children, f),
        Stmt::EmitExpr(s) => expr(&s.expr, f),
        Stmt::EmitRaw(_) => (),
        Stmt::ForLoop(s) => {
            expr(&s.target, f);
            expr(&s.iter, f);
            if let Some(e) = &s.filter_expr {
                expr(e, f)
            }
            body(&s.body, f);
            body(&s.else_body, f)
        }
        Stmt::IfCond(s) => {
            expr(&s.expr, f);
            body(&s.true_body, f);
            body(&s.false_body, f)
        }
        Stmt::WithBlock(s) => {
            for (t, e) in &s.assignments {
                expr(t, f);
                expr(e, f)
            }
            body(&s.body, f)
        }
        Stmt::Set(s) => {
            expr(&s.target, f);
            expr(&s.expr, f)
        }
        Stmt::SetBlock(s) => {
            expr(&s.target, f);
            if let Some(e) = &s.filter {
                expr(e, f)
            }
            body(&s.body, f)
        }
        Stmt::AutoEscape(s) => {
            expr(&s.enabled, f);
            body(&s.body, f)
        }
        Stmt::FilterBlock(s) => {
            expr(&s.filter, f);
            body(&s.body, f)
        }
        #[cfg(feature = "multi_template")]
        Stmt::Block(s) => body(&s.body, f),
        #[cfg(feature = "multi_template")]
        Stmt::Extends(s) => expr(&s.name, f),
        #[cfg(feature = "multi_template")]
        Stmt::Include(s) => expr(&s.name, f),
        #[cfg(feature = "multi_template")]
        Stmt::Import(s) => {
            expr(&s.expr, f);
            expr(&s.name, f)
        }
        #[cfg(feature = "multi_template")]
        Stmt::FromImport(s) => {
            expr(&s.expr, f);
            for (n, a) in &s.names {
                expr(n, f);
                if let Some(e) = a {
                    expr(e, f)
                }
            }
        }
        #[cfg(feature = "macros")]
        Stmt::Macro(s) => mac(s, f),
        #[cfg(feature = "macros")]
        Stmt::CallBlock(s) => {
            expr(&s.call.expr, f);
            args(&s.call.args, f);
            mac(&s.macro_decl, f)
        }
        #[cfg(feature = "loop_controls")]
        Stmt::Continue(_) | Stmt::Break(_) => (),
        Stmt::Do(s) => {
            expr(&s.call.expr, f);
            args(&s.call.args, f)
        }
    }
}
