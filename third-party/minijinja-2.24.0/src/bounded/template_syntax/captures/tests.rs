#![forbid(unsafe_code)]
use super::*;
use crate::bounded::template_syntax::{Plan as SyntaxPlan, WhitespaceConfig};
use crate::compiler::{ast, meta, parser};
use std::collections::HashSet;

include!("../../expression/tests/fixtures.rs");

pub(super) fn fixtures() -> &'static [(&'static str, &'static str)] {
    FIXTURES
}

pub(super) const SOURCE: &str = "{% macro outer(p, q=default) %}{{ first }}{{ second }}{% if condition %}{{ branch }}{% else %}{{ other }}{% endif %}{% macro inner(x) %}{{ caller }}{{ nested }}{{ p }}{% endmacro %}{{ inner(p) }}{% endmacro %}";
pub(super) fn syntax(source: &str) -> ParsedTemplate<'_> {
    SyntaxPlan::inspect(source, "capture", WhitespaceConfig::default())
        .unwrap()
        .construct()
        .unwrap()
}
fn pointers(work: &Workspace<'_, '_>) -> [usize; 4] {
    [
        work.scratch.assigned.as_ptr() as usize,
        work.scratch.scopes.as_ptr() as usize,
        work.scratch.captures.as_ptr() as usize,
        work.scratch.tasks.as_ptr() as usize,
    ]
}
fn ordinary_macros<'t, 's>(root: &'t ast::Stmt<'s>) -> Vec<(&'t ast::Macro<'s>, Span)> {
    let mut stack = vec![root];
    let mut out = Vec::new();
    while let Some(stmt) = stack.pop() {
        let mut body = |items: &'t [ast::Stmt<'s>]| {
            stack.extend(items.iter().rev());
        };
        match stmt {
            ast::Stmt::Template(v) => body(&v.children),
            ast::Stmt::ForLoop(v) => {
                body(&v.else_body);
                body(&v.body);
            }
            ast::Stmt::IfCond(v) => {
                body(&v.false_body);
                body(&v.true_body);
            }
            ast::Stmt::WithBlock(v) => body(&v.body),
            ast::Stmt::AutoEscape(v) => body(&v.body),
            ast::Stmt::FilterBlock(v) => body(&v.body),
            ast::Stmt::SetBlock(v) => body(&v.body),
            #[cfg(feature = "multi_template")]
            ast::Stmt::Block(v) => body(&v.body),
            ast::Stmt::Macro(v) => {
                out.push((&**v, v.span()));
                body(&v.body);
            }
            ast::Stmt::CallBlock(v) => {
                out.push((&*v.macro_decl, v.macro_decl.span()));
                body(&v.macro_decl.body);
            }
            _ => {}
        }
    }
    out
}
fn compare(source: &str, whitespace: WhitespaceConfig) -> usize {
    let ordinary = parser::parse(source, "capture", Default::default(), whitespace);
    let closed = SyntaxPlan::inspect(source, "capture", whitespace)
        .unwrap()
        .construct();
    match (ordinary, closed) {
        (Ok(ast), Ok(owner)) => {
            for nested in [false, true] {
                assert_eq!(
                    meta::find_undeclared(&ast, nested),
                    meta::reference::find_undeclared(&ast, nested)
                );
            }
            let original = ordinary_macros(&ast);
            let handles: Vec<_> = owner.macros().collect();
            assert_eq!(handles.len(), original.len());
            let Some(first) = handles.first().copied() else {
                return 0;
            };
            let mut workspace = Plan::for_macro(first).unwrap().construct().unwrap();
            let capacities = workspace.capacities();
            let ptrs = pointers(&workspace);
            for handle in handles {
                let values: Vec<_> = original
                    .iter()
                    .filter(|(_, span)| *span == handle.span())
                    .collect();
                assert_eq!(values.len(), 1, "exact original macro span");
                let expected = meta::reference::find_macro_closure(values[0].0);
                assert_eq!(meta::find_macro_closure(values[0].0), expected);
                let result = workspace.analyze(handle).unwrap();
                let actual: HashSet<_> = result.names().collect();
                assert_eq!(actual.len(), result.names().len());
                assert_eq!(actual, expected, "{source}");
                assert_eq!(result.contains("caller"), expected.contains("caller"));
                workspace = result.into_workspace();
                assert_eq!(workspace.capacities(), capacities);
                assert_eq!(pointers(&workspace), ptrs);
            }
            original.len()
        }
        (Err(_), Err(_)) => 0, // Existing syntax oracle separately checks exact diagnostics.
        _ => panic!("ordinary and closed source parsing differ"),
    }
}

#[test]
fn all_real_template_macro_and_dotted_memberships_match_untouched_reference() {
    assert_eq!(FIXTURES.len(), 32);
    let mut macros = 0;
    let mut attempts = 0;
    for (_, source) in FIXTURES {
        for trim_blocks in [false, true] {
            for lstrip_blocks in [false, true] {
                for keep_trailing_newline in [false, true] {
                    macros += compare(
                        source,
                        WhitespaceConfig {
                            trim_blocks,
                            lstrip_blocks,
                            keep_trailing_newline,
                        },
                    );
                    attempts += 1;
                }
            }
        }
    }
    assert_eq!(attempts, 256);
    assert!(macros > 0);
    println!("checked macro corpus: templates=32 whitespace=8 attempts={attempts} actual_macro_comparisons={macros}");
}

#[test]
fn nested_scopes_defaults_callers_and_ignored_children_match_actual_macro_worker() {
    for source in [
        SOURCE,
        "{% macro m(x=default) %}{% set a = a %}{% with b=b, c=b %}{{ b }}{{ c }}{{ free }}{% endwith %}{{ free }}{% for t in t %}{{ loop.index }}{{ t }}{{ value }}{% else %}{{ t }}{{ value }}{% endfor %}{% endmacro %}",
        "{% macro m() %}{{ ignored[a:b:c] }}{% autoescape ignored %}{{ kept }}{% endautoescape %}{% filter trim(ignored) %}{{ kept }}{% endfilter %}{% set x|trim(ignored) %}{{ body }}{% endset %}{{ x }}{% endmacro %}",
        "{% macro m() %}{{ f(a, named=b, *args, **kwargs) }}{{ {key: value} }}{{ [first, second] }}{{ var if cond else other }}{% endmacro %}{% call(x) m() %}{{ x }}{{ external }}{% endcall %}",
    ] { assert!(compare(source, WhitespaceConfig::default()) > 0); }
}

#[test]
fn foreign_equal_source_owner_is_rejected_before_clearing_completed_membership() {
    let one = syntax(SOURCE);
    let two = syntax(SOURCE);
    let handle = one.macros().last().unwrap();
    let work = Plan::for_macro(handle).unwrap().construct().unwrap();
    let work = work.analyze(handle).unwrap().into_workspace();
    let old = work.scratch.captures.clone();
    let ptrs = pointers(&work);
    let capacities = work.capacities();
    let failure = work.analyze(two.macros().last().unwrap()).unwrap_err();
    assert!(matches!(failure.cause(), Cause::SourceMismatch));
    assert_eq!(failure.workspace.scratch.captures, old);
    assert_eq!(failure.capacities(), capacities);
    assert_eq!(pointers(&failure.workspace), ptrs);
    let work = failure.into_workspace();
    assert!(work.analyze(handle).is_ok());
}

#[cfg(feature = "development-closed-chat")]
#[test]
fn every_real_capture_reserve_failure_retains_exact_completed_prefix() {
    use std::error::Error;
    let owner = syntax(SOURCE);
    let handle = owner.macros().last().unwrap();
    for buffer in [
        Buffer::Assigned,
        Buffer::Scopes,
        Buffer::Captures,
        Buffer::Tasks,
    ] {
        let plan = Plan::for_macro(handle).unwrap();
        let requirements = plan.requirements();
        let failure = plan.fail_reservation(buffer).construct().unwrap_err();
        let Cause::Reserve {
            buffer: actual,
            error,
        } = failure.cause()
        else {
            panic!("actual reserve error required")
        };
        assert_eq!(*actual, buffer);
        assert!(std::ptr::eq(
            error,
            failure
                .cause()
                .source()
                .unwrap()
                .downcast_ref::<TryReserveError>()
                .unwrap()
        ));
        let retained = failure.capacities();
        for (index, capacity) in retained.into_iter().enumerate() {
            if index < buffer.index() {
                assert!(capacity >= requirements.counts[index] && requirements.counts[index] > 0);
            } else {
                assert_eq!(capacity, 0);
            }
        }
    }
}

#[test]
fn actual_late_append_refusals_hold_all_allocations_and_real_prefixes() {
    let owner = syntax(SOURCE);
    let handle = owner.macros().last().unwrap();
    for (buffer, limit) in [
        (Buffer::Assigned, 1),
        (Buffer::Scopes, 1),
        (Buffer::Captures, 1),
        (Buffer::Tasks, 2),
    ] {
        let mut work = Plan::for_macro(handle).unwrap().construct().unwrap();
        let capacities = work.capacities();
        let ptrs = pointers(&work);
        work.scratch.set_limit(buffer, limit);
        let failure = work.analyze(handle).unwrap_err();
        assert!(matches!(failure.cause(), Cause::Capacity(b) if *b==buffer));
        assert_eq!(failure.capacities(), capacities);
        assert_eq!(pointers(&failure.workspace), ptrs);
        match buffer {
            Buffer::Assigned => assert_eq!(failure.workspace.scratch.assigned.len(), 1),
            Buffer::Scopes => {
                assert_eq!(failure.workspace.scratch.scopes.len(), 1);
                assert!(!failure.workspace.scratch.captures.is_empty());
            }
            Buffer::Captures => assert_eq!(failure.workspace.scratch.captures.len(), 1),
            Buffer::Tasks => assert_eq!(failure.workspace.scratch.tasks.len(), 2),
        }
    }
}

#[test]
fn independent_workspaces_share_only_the_immutable_source_and_survive_unwind() {
    let owner = syntax(SOURCE);
    std::thread::scope(|threads| {
        let owner_ref = &owner;
        let left = threads.spawn(move || {
            let h = owner_ref.macros().last().unwrap();
            Plan::for_macro(h)
                .unwrap()
                .construct()
                .unwrap()
                .analyze(h)
                .unwrap()
                .names()
                .count()
        });
        let right = threads.spawn(move || {
            let h = owner_ref.macros().next().unwrap();
            Plan::for_macro(h)
                .unwrap()
                .construct()
                .unwrap()
                .analyze(h)
                .unwrap()
                .names()
                .count()
        });
        assert!(left.join().unwrap() > 0);
        assert!(right.join().unwrap() > 0);
    });
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let h = owner.macros().last().unwrap();
        let _result = Plan::for_macro(h)
            .unwrap()
            .construct()
            .unwrap()
            .analyze(h)
            .unwrap();
        panic!("external unwind");
    }));
    assert!(caught.is_err());
    let h = owner.macros().last().unwrap();
    assert!(Plan::for_macro(h)
        .unwrap()
        .construct()
        .unwrap()
        .analyze(h)
        .is_ok());
}

#[test]
fn checked_geometry_rejects_each_layout_overflow_before_reserving() {
    for index in 0..4 {
        let mut counts = [0; 4];
        counts[index] = usize::MAX;
        assert_eq!(Requirements::checked(counts), Err(PlanError::Overflow));
    }
    // Each individual layout fits isize, while their checked sum does not.
    let refs = isize::MAX as usize / size_of::<&str>();
    let marks = isize::MAX as usize / size_of::<usize>();
    assert_eq!(
        Requirements::checked([refs, marks, refs, 0]),
        Err(PlanError::Overflow)
    );
    let mut count = usize::MAX;
    assert_eq!(add(&mut count, 1), Err(PlanError::Overflow));
    assert_eq!(count, usize::MAX);
    let counts = [2, 3, 4, 5];
    let requirements = Requirements::checked(counts).unwrap();
    assert_eq!(requirements.capacity(Buffer::Tasks), 5);
    assert_eq!(
        requirements.buffer_bytes(Buffer::Assigned),
        2 * size_of::<&str>()
    );
    assert_eq!(
        requirements.buffer_bytes(Buffer::Tasks),
        5 * size_of::<walk::Task<'static, Packed<'static, 'static>>>()
    );
    assert_eq!(
        requirements.requested_bytes(),
        requirements.bytes.iter().sum::<usize>()
    );
    assert!(control_bytes().unwrap() >= size_of::<Workspace<'static, 'static>>());
}

#[test]
fn templates_without_macros_need_no_workspace_and_empty_capture_retains_no_name_buffer() {
    let plain = syntax("plain UTF-8 café 🦀");
    assert!(plain.macros().next().is_none());
    let owner = syntax("{% macro empty() %}UTF-8 café 🦀{% endmacro %}");
    let handle = owner.macros().next().unwrap();
    let plan = Plan::for_macro(handle).unwrap();
    assert_eq!(plan.requirements().capacity(Buffer::Captures), 0);
    let result = plan.construct().unwrap().analyze(handle).unwrap();
    assert_eq!(result.names().len(), 0);
    assert_eq!(
        result.into_workspace().capacities()[Buffer::Captures.index()],
        0
    );
}

#[cfg(feature = "unicode")]
#[test]
fn unicode_macro_parameters_and_capture_names_keep_exact_borrowed_text() {
    assert!(
        compare(
            "{% macro café(α) %}{{ α }}{{ β }}{{ café }}{% endmacro %}",
            WhitespaceConfig::default()
        ) > 0
    );
}
