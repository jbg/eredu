//! One metadata walk for ordinary AST and source-bound flat syntax.
#![forbid(unsafe_code)]
use super::view::{Expression as E, Item, SequenceKind as K, Statement as S, Storage, Task, View};

pub(crate) fn run<'s, V: View<'s>, T: Storage<'s, V>>(
    view: V,
    state: &mut T,
    root: Task<'s, V>,
) -> Result<(), T::Error> {
    state.push_task(root)?;
    while let Some(task) = state.pop_task() {
        match task {
            Task::Name(name) => state.assign(name)?,
            Task::PushScope => state.push_scope()?,
            Task::PopScope => state.pop_scope()?,
            Task::Assign(expr) => match view.expression(expr) {
                E::Var(name) => state.assign(name)?,
                E::List(items) => state.push_task(Task::Sequence(K::Assign, items))?,
                _ => {}
            },
            Task::Expr(expr) => match view.expression(expr) {
                E::Var(name) => {
                    if !state.is_assigned(name) {
                        state.capture(name)?;
                        if state.tracks_nested() {
                            state.nested_variable(name)?;
                        } else {
                            state.assign(name)?;
                        }
                    }
                }
                E::Const => {}
                E::Unary(child) => state.push_task(Task::Expr(child))?,
                E::Binary(left, right) | E::Item(left, right) => {
                    state.push_task(Task::Expr(right))?;
                    state.push_task(Task::Expr(left))?;
                }
                E::Compare(root, rest) | E::Test(root, rest) | E::Call(root, rest) => {
                    state.push_task(Task::Sequence(K::Lookup, rest))?;
                    state.push_task(Task::Expr(root))?;
                }
                E::If(test, yes, no) => {
                    if let Some(no) = no {
                        state.push_task(Task::Expr(no))?;
                    }
                    state.push_task(Task::Expr(yes))?;
                    state.push_task(Task::Expr(test))?;
                }
                E::Filter(root, rest) => {
                    state.push_task(Task::Sequence(K::Lookup, rest))?;
                    if let Some(root) = root {
                        state.push_task(Task::Expr(root))?;
                    }
                }
                E::Attr(base, name) => {
                    let mut recorded = false;
                    if state.tracks_nested() {
                        let mut attrs = state.nested_start(name)?;
                        let mut ptr = base;
                        loop {
                            match view.expression(ptr) {
                                E::Var(name) => {
                                    if !state.is_assigned(name) {
                                        state.nested_finish(attrs, name)?;
                                        recorded = true;
                                    }
                                    break;
                                }
                                E::Attr(child, name) => {
                                    state.nested_attr(&mut attrs, name)?;
                                    ptr = child;
                                }
                                _ => break,
                            }
                        }
                    }
                    if !recorded {
                        state.push_task(Task::Expr(base))?;
                    }
                }
                E::Slice(start, stop, step) => {
                    // Preserve metadata's existing omission of the slice base.
                    if let Some(step) = step {
                        state.push_task(Task::Expr(step))?;
                    }
                    if let Some(stop) = stop {
                        state.push_task(Task::Expr(stop))?;
                    }
                    if let Some(start) = start {
                        state.push_task(Task::Expr(start))?;
                    }
                }
                E::List(items) => state.push_task(Task::Sequence(K::Lookup, items))?,
                E::Map(items) => state.push_task(Task::Sequence(K::Map, items))?,
            },
            #[cfg(feature = "macros")]
            Task::Macro(value, declare_caller) => {
                let parts = view.macro_parts(value);
                state.push_task(Task::Sequence(K::Body, parts.body))?;
                state.push_task(Task::Sequence(K::Lookup, parts.defaults))?;
                state.push_task(Task::Sequence(K::Assign, parts.args))?;
                if declare_caller {
                    state.push_task(Task::Name("caller"))?;
                }
            }
            Task::Sequence(kind, cursor) => {
                if let Some((item, rest)) = view.next(cursor) {
                    state.push_task(Task::Sequence(kind, rest))?;
                    match (kind, item) {
                        (K::Body, Item::Stmt(stmt)) => state.push_task(Task::Stmt(stmt))?,
                        (K::Lookup, Item::Expr(expr)) => state.push_task(Task::Expr(expr))?,
                        (K::Assign, Item::Expr(expr)) => state.push_task(Task::Assign(expr))?,
                        (K::Map, Item::Pair(key, Some(value))) => {
                            state.push_task(Task::Expr(value))?;
                            state.push_task(Task::Expr(key))?;
                        }
                        (K::Binding, Item::Pair(target, Some(value))) => {
                            state.push_task(Task::Expr(value))?;
                            state.push_task(Task::Assign(target))?;
                        }
                        #[cfg(feature = "multi_template")]
                        (K::Import, Item::Pair(name, alias)) => {
                            state.push_task(Task::Assign(alias.unwrap_or(name)))?;
                        }
                        _ => unreachable!("private metadata cursor kind"),
                    }
                }
            }
            Task::Stmt(stmt) => match view.statement(stmt) {
                S::Template(children) => {
                    state.push_task(Task::Sequence(K::Body, children))?;
                    state.push_task(Task::Name("self"))?;
                }
                S::EmitExpr(expr) => state.push_task(Task::Expr(expr))?,
                S::EmitRaw => {}
                S::For {
                    target,
                    iter,
                    filter,
                    body,
                    otherwise,
                } => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, otherwise))?;
                    state.push_task(Task::PushScope)?;
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, body))?;
                    if let Some(filter) = filter {
                        state.push_task(Task::Expr(filter))?;
                    }
                    state.push_task(Task::Assign(target))?;
                    state.push_task(Task::Expr(iter))?;
                    state.push_task(Task::Name("loop"))?;
                    state.push_task(Task::PushScope)?;
                }
                S::If(test, yes, no) => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, no))?;
                    state.push_task(Task::PushScope)?;
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, yes))?;
                    state.push_task(Task::PushScope)?;
                    state.push_task(Task::Expr(test))?;
                }
                S::With(bindings, body) => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, body))?;
                    state.push_task(Task::Sequence(K::Binding, bindings))?;
                    state.push_task(Task::PushScope)?;
                }
                S::Set(target, value) => {
                    state.push_task(Task::Expr(value))?;
                    state.push_task(Task::Assign(target))?;
                }
                S::AutoEscape(body) | S::FilterBlock(body) => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, body))?;
                    state.push_task(Task::PushScope)?;
                }
                S::SetBlock(target, body) => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, body))?;
                    state.push_task(Task::PushScope)?;
                    state.push_task(Task::Assign(target))?;
                }
                #[cfg(feature = "multi_template")]
                S::Block(body) => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Sequence(K::Body, body))?;
                    state.push_task(Task::Name("super"))?;
                    state.push_task(Task::PushScope)?;
                }
                #[cfg(feature = "multi_template")]
                S::Extends | S::Include => {}
                #[cfg(feature = "multi_template")]
                S::Import(name) => state.push_task(Task::Assign(name))?,
                #[cfg(feature = "multi_template")]
                S::FromImport(names) => state.push_task(Task::Sequence(K::Import, names))?,
                #[cfg(feature = "macros")]
                S::Macro(name, value) => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Macro(value, true))?;
                    state.push_task(Task::PushScope)?;
                    state.push_task(Task::Name(name))?;
                }
                #[cfg(feature = "macros")]
                S::CallBlock(receiver, args, value) => {
                    state.push_task(Task::PopScope)?;
                    state.push_task(Task::Macro(value, true))?;
                    state.push_task(Task::PushScope)?;
                    state.push_task(Task::Sequence(K::Lookup, args))?;
                    state.push_task(Task::Expr(receiver))?;
                }
                #[cfg(feature = "loop_controls")]
                S::Continue | S::Break => {}
                S::Do(receiver, args) => {
                    state.push_task(Task::Sequence(K::Lookup, args))?;
                    state.push_task(Task::Expr(receiver))?;
                }
            },
        }
    }
    Ok(())
}
