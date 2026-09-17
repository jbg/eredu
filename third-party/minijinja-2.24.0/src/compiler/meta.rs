//! Shared ordinary and checked metadata traversal.
#![forbid(unsafe_code)]
use crate::compiler::ast;
use std::collections::HashSet;

mod ordinary;
pub(crate) mod shared;
pub(crate) mod view;

/// Finds all variables that need to be captured as closure for a macro.
#[cfg(feature = "macros")]
pub fn find_macro_closure<'a>(m: &ast::Macro<'a>) -> HashSet<&'a str> {
    let mut state = ordinary::State::new(false);
    shared::run(
        ordinary::Ordinary::new(),
        &mut state,
        view::Task::Macro(m, false),
    )
    .unwrap();
    state.out
}

/// Finds all variables that are undeclared in a template.
pub fn find_undeclared(t: &ast::Stmt<'_>, track_nested: bool) -> HashSet<String> {
    let mut state = ordinary::State::new(track_nested);
    shared::run(ordinary::Ordinary::new(), &mut state, view::Task::Stmt(t)).unwrap();
    if let Some(nested) = state.nested_out {
        nested
    } else {
        state.out.into_iter().map(|x| x.to_string()).collect()
    }
}

#[cfg(test)]
#[path = "meta/reference.rs"]
pub(crate) mod reference;
#[cfg(test)]
#[path = "meta/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "meta/trace_reference.rs"]
mod trace_reference;
