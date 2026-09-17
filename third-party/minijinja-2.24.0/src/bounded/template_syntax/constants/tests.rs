#![forbid(unsafe_code)]
use super::*;
use crate::bounded::expression::store::{OperandKind as O, RecordKind as R, Scalar as L};
use crate::compiler::fold::tests::exact;
use crate::compiler::{ast, fold::materialization_reference as reference, parser};
use crate::value::Value as OrdinaryValue;
include!("../../expression/tests/fixtures.rs");
// A fold-only projection, not an AST/parser equivalence oracle. Other variants all
// have the original immediate-None fold rule; inherited syntax tests compare full ASTs.
fn text(source: &ParsedTemplate<'_>, joined: Joined) -> String {
    Segments {
        source,
        next: joined.sequence.head,
        generated: None,
    }
    .collect()
}
fn expressions(source: &ParsedTemplate<'_>, mut seq: Sequence) -> Vec<ast::Expr<'static>> {
    let mut out = Vec::new();
    while let Some(i) = seq.head {
        let item = &source.storage.expressions.operands[i];
        seq.head = item.next;
        let O::Expr(id) = item.kind else {
            panic!("collection expression")
        };
        out.push(project(source, id));
    }
    out
}
fn project(source: &ParsedTemplate<'_>, id: NodeId) -> ast::Expr<'static> {
    let rec = &source.storage.expressions.nodes[id.0];
    let span = rec.span;
    match &rec.kind {
        R::Const(x) => ast::Expr::Const(ast::Spanned::new(
            ast::Const {
                value: match x {
                    L::None => OrdinaryValue::from(()),
                    L::Bool(x) => OrdinaryValue::from(*x),
                    L::Int(x) => OrdinaryValue::from(*x),
                    L::Int128(x) => OrdinaryValue::from(*x),
                    L::Float(x) => OrdinaryValue::from(*x),
                    L::Text(x) => OrdinaryValue::from(text(source, *x)),
                },
            },
            span,
        )),
        R::Unary(op, x) => ast::Expr::UnaryOp(ast::Spanned::new(
            ast::UnaryOp {
                op: match op {
                    ast::UnaryOpKind::Not => ast::UnaryOpKind::Not,
                    ast::UnaryOpKind::Neg => ast::UnaryOpKind::Neg,
                },
                expr: project(source, *x),
            },
            span,
        )),
        R::Binary(op, l, r) => ast::Expr::BinOp(ast::Spanned::new(
            ast::BinOp {
                op: *op,
                left: project(source, *l),
                right: project(source, *r),
            },
            span,
        )),
        R::Compare(left, ops) => {
            let mut cur = ops.head;
            let mut out = Vec::new();
            while let Some(i) = cur {
                let e = &source.storage.expressions.operands[i];
                cur = e.next;
                let O::Compare(op, id) = e.kind else {
                    panic!("comparison")
                };
                out.push(ast::CompareOp {
                    op,
                    expr: project(source, id),
                });
            }
            ast::Expr::Compare(ast::Spanned::new(
                ast::Compare {
                    expr: project(source, *left),
                    ops: out,
                },
                span,
            ))
        }
        R::List(items) => ast::Expr::List(ast::Spanned::new(
            ast::List {
                items: expressions(source, *items),
            },
            span,
        )),
        R::Map(keys, values) => ast::Expr::Map(ast::Spanned::new(
            ast::Map {
                keys: expressions(source, *keys),
                values: expressions(source, *values),
            },
            span,
        )),
        _ => ast::Expr::Var(ast::Spanned::new(
            ast::Var {
                id: "__immediate_nonconstant",
            },
            span,
        )),
    }
}
fn materialize(value: Option<Value<'_, '_>>) -> Option<OrdinaryValue> {
    value.map(|x| match x {
        Value::Text(t) => OrdinaryValue::from(t.segments().collect::<String>()),
        Value::List(list) => OrdinaryValue::from(
            (0..list.len())
                .map(|i| materialize(list.get(i)).unwrap())
                .collect::<Vec<_>>(),
        ),
        Value::Scalar(s) => match s {
            Scalar::None => OrdinaryValue::from(()),
            Scalar::Bool(v) => v.into(),
            Scalar::U64(v) => v.into(),
            Scalar::U128(v) => v.into(),
            Scalar::I64(v) => v.into(),
            Scalar::I128(v) => v.into(),
            Scalar::F64(v) => v.into(),
        },
    })
}
fn syntax(source: &str) -> ParsedTemplate<'_> {
    super::super::Plan::inspect(source, "fold", Default::default())
        .unwrap()
        .construct()
        .unwrap()
}
fn compare(source: &str) {
    let input = format!("{{{{ {source} }}}}");
    let owner = syntax(&input);
    let expr = owner.expressions().last().unwrap();
    let ordinary = parser::parse_expr(source).unwrap();
    exact(ordinary.as_const(), reference::expression(&ordinary));
    let result = Plan::for_expression(expr)
        .unwrap()
        .construct()
        .unwrap()
        .fold(expr)
        .unwrap();
    exact(
        materialize(result.value()),
        reference::expression(&ordinary),
    );
}
#[test]
fn source_literals_joined_utf8_and_unary_results_preserve_exact_reference_values() {
    for s in [
        "none",
        "true",
        "false",
        "0",
        "18446744073709551615",
        "18446744073709551616",
        "170141183460469231731687303715884105728",
        "170141183460469231731687303715884105729",
        "1e999",
        "0.0",
        "-0.0",
        "-9223372036854775808",
        "-9223372036854775809",
        "-170141183460469231731687303715884105728",
        "-170141183460469231731687303715884105729",
        "--170141183460469231731687303715884105728",
        "-true",
        "-none",
        "-'text'",
        "''",
        "'λ' '\u{4E2D}'",
        "'a\\n' 'b'",
        "not ''",
        "not ('λ' '\\n')",
        "not none",
        "not -0.0",
        "not not true",
    ] {
        compare(s)
    }
}
#[test]
fn reached_unfinished_operations_are_distinct_from_real_nonconstancy() {
    let malformed = "{{ x + not '' }}";
    let old = parser::parse(malformed, "fold", Default::default(), Default::default())
        .err().expect("original parser must reject malformed input");
    let closed = super::super::Plan::inspect(malformed, "fold", Default::default())
        .unwrap().construct().unwrap_err();
    let crate::bounded::expression::Cause::Syntax(error) = closed.cause() else {
        panic!("expected original malformed-expression diagnostic");
    };
    assert_eq!(old.kind(), crate::ErrorKind::SyntaxError);
    assert_eq!(old.detail(), Some(error.to_string().as_str()));
    assert_eq!(old.name(), Some("fold"));
    assert_eq!(old.line(), error.span().map(|span| span.start_line as usize).filter(|line| *line > 0));
    #[cfg(feature = "debug")]
    assert_eq!(old.span(), error.span());
    for s in ["[1]+[2]", "[1]*2", "2*[1]"] {
        let input = format!("{{{{ {s} }}}}");
        let source = syntax(&input);
        let handle = source.expressions().last().unwrap();
        let error = Plan::for_expression(handle)
            .unwrap()
            .construct()
            .unwrap()
            .fold(handle)
            .unwrap_err();
        assert!(matches!(
            error.cause(),
            Cause::NeedsMaterialization(Collection::List)
        ));
        assert!(error.owner.operation.is_some());
        assert!(error.capacity() > 0);
    }
    for s in [
        "x",
        "x+y",
        "x + (not '')",
        "x < [1]",
        "x.foo",
        "x[1]",
        "f(1)",
        "1 if true else 2",
    ] {
        compare(s)
    }
}
#[test]
fn eager_binary_right_refuses_materialization_but_comparison_stops_at_nonconstant_left() {
    let source = syntax("{{ x + {} }}{{ x < {} }}{{ x < {} < 1 }}");
    let handles: Vec<_> = source
        .expressions()
        .filter(|h| {
            matches!(
                source.storage.expressions.nodes[h.id.0].kind,
                R::Binary(..) | R::Compare(..)
            )
        })
        .collect();
    assert_eq!(handles.len(), 3);
    assert!(matches!(source.storage.expressions.nodes[handles[1].id.0].kind, R::Binary(..)));
    assert!(matches!(source.storage.expressions.nodes[handles[2].id.0].kind, R::Compare(..)));
    let error = Plan::for_expression(handles[0])
        .unwrap()
        .construct()
        .unwrap()
        .fold(handles[0])
        .unwrap_err();
    assert!(matches!(
        error.cause(),
        Cause::NeedsMaterialization(Collection::Map)
    ));
    assert!(error
        .owner
        .frames
        .iter()
        .any(|f| matches!(f, fold::Frame::BinaryRight(_, None))));
    let error = error.into_workspace().fold(handles[1]).unwrap_err();
    assert!(matches!(error.cause(), Cause::NeedsMaterialization(Collection::Map)));
    assert!(error.owner.frames.iter().any(|f| matches!(f, fold::Frame::BinaryRight(_, None))));
    let result = error.into_workspace().fold(handles[2]).unwrap();
    assert!(result.value().is_none());
}
#[test]
fn collection_eligibility_does_not_recursively_fold_children_and_refuses_empty_values() {
    for s in ["{}", "{'a':1,'a':2}"] {
        let input = format!("{{{{ {s} }}}}");
        let source = syntax(&input);
        let handle = source.expressions().last().unwrap();
        let failure = Plan::for_expression(handle)
            .unwrap()
            .construct()
            .unwrap()
            .fold(handle)
            .unwrap_err();
        assert!(matches!(failure.cause(), Cause::NeedsMaterialization(_)));
        assert!(failure.owner.collection.is_some());
    }
    for s in ["[1+2]", "[x]", "{'x': 1+2}", "{x: 1}", "[not true]"] {
        compare(s)
    }
}
#[test]
fn exact_owner_mismatch_preserves_prior_value_storage_and_allows_same_owner_reuse() {
    let source = syntax("{{ 'a' 'b' }}{{ not false }}");
    let foreign = syntax("{{ 'a' 'b' }}{{ not false }}");
    let first = source
        .expressions()
        .find(|h| {
            matches!(
                source.storage.expressions.nodes[h.id.0].kind,
                R::Const(L::Text(_))
            )
        })
        .unwrap();
    let later = source.expressions().last().unwrap();
    let handle = foreign.expressions().last().unwrap();
    let workspace = Plan::for_expression(first).unwrap().construct().unwrap();
    let pointer = workspace.frames.as_ptr();
    let capacity = workspace.capacity();
    let result = workspace.fold(first).unwrap();
    assert_eq!(materialize(result.value()).unwrap().as_str(), Some("ab"));
    let failure = result.into_workspace().fold(handle).unwrap_err();
    assert!(matches!(failure.cause(), Cause::Source));
    assert_eq!(failure.owner.frames.as_ptr(), pointer);
    assert_eq!(failure.capacity(), capacity);
    assert!(matches!(failure.owner.value, Some(Descriptor::Text(_))));
    let result = failure.into_workspace().fold(later).unwrap();
    exact(materialize(result.value()), Some(OrdinaryValue::from(true)));
    assert_eq!(result.owner.frames.as_ptr(), pointer);
}
#[test]
fn one_real_reserve_error_retains_cause_and_source_without_an_unrelated_allocation() {
    let source = syntax("{{ not false }}");
    let handle = source.expressions().last().unwrap();
    let failure = Plan::for_expression(handle)
        .unwrap()
        .construct_inner(Some(Buffer::Frames))
        .unwrap_err();
    assert!(matches!(failure.cause(), Cause::Reserve(_)));
    assert!(std::error::Error::source(&failure).is_some());
    assert_eq!(failure.capacity(), 0);
    assert!(std::ptr::eq(failure.owner.source, &source));
}
#[test]
fn private_push_limit_keeps_real_parent_prefix_and_failed_current_frame() {
    let source = syntax("{{ not not not false }}");
    let handle = source.expressions().last().unwrap();
    let mut workspace = Plan::for_expression(handle).unwrap().construct().unwrap();
    let pointer = workspace.frames.as_ptr();
    workspace.limit = 1;
    let error = workspace.fold(handle).unwrap_err();
    assert!(matches!(error.cause(), Cause::Capacity));
    assert_eq!(error.owner.frames.len(), 1);
    assert!(error.owner.pending.is_some());
    assert_eq!(error.owner.frames.as_ptr(), pointer);
    let mut workspace = error.into_workspace();
    workspace.limit = workspace.requirements.frames;
    let result = workspace.fold(handle).unwrap();
    exact(materialize(result.value()), Some(OrdinaryValue::from(true)));
}
#[test]
fn checked_layout_overflow_and_source_derived_peak_are_not_allocator_capacity_claims() {
    assert!(matches!(
        requirements(usize::MAX, 0),
        Err(PlanError::Overflow)
    ));
    assert!(matches!(
        requirements(isize::MAX as usize, 0),
        Err(PlanError::Overflow)
    ));
    let input = format!("{{{{ {}false }}}}", "not ".repeat(20));
    let source = syntax(&input);
    let handle = source.expressions().last().unwrap();
    let plan = Plan::for_expression(handle).unwrap();
    let q = plan.requirements();
    assert_eq!(q.frames(), source.expression_count() + 1);
    assert_eq!(
        q.requested_bytes(),
        Layout::array::<Continuation<Packed<'static, 'static>, Descriptor>>(q.frames())
            .unwrap()
            .size()
            + Layout::array::<Recipe>(q.recipes()).unwrap().size()
    );
    let result = plan.construct().unwrap().fold(handle).unwrap();
    assert_eq!(result.owner.peak, 20);
    assert_eq!(result.owner.pushes, 20);
    assert!(result.owner.peak < q.frames());
}
#[test]
fn independent_workspaces_share_only_immutable_syntax_and_survive_external_unwind() {
    let source = syntax("{{ not false }}");
    let handle = source.expressions().last().unwrap();
    std::thread::scope(|scope| {
        let mut threads = Vec::new();
        for _ in 0..4 {
            threads.push(scope.spawn(|| {
                let result = Plan::for_expression(handle)
                    .unwrap()
                    .construct()
                    .unwrap()
                    .fold(handle)
                    .unwrap();
                exact(materialize(result.value()), Some(OrdinaryValue::from(true)));
            }));
        }
        for t in threads {
            t.join().unwrap()
        }
    });
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let result = Plan::for_expression(handle)
            .unwrap()
            .construct()
            .unwrap()
            .fold(handle)
            .unwrap();
        assert!(result.value().is_some());
        panic!("external caller unwind")
    }));
    assert!(panic.is_err());
    compare("not false");
}
#[test]
fn every_closed_expression_in_32_real_templates_matches_the_reference_or_explicit_refusal() {
    let (mut parsed, mut errors, mut values, mut nonconstant, mut operations, mut materializers) =
        (0, 0, 0, 0, 0, 0);
    for (name, input) in FIXTURES {
        let ordinary = parser::parse(input, name, Default::default(), Default::default());
        let source = super::super::Plan::inspect(input, name, Default::default())
            .unwrap()
            .construct();
        let source = match source {
            Ok(s) => {
                assert!(ordinary.is_ok(), "{name}");
                s
            }
            Err(e) => {
                let old = match ordinary {
                    Err(e) => e,
                    Ok(_) => panic!("closed parser rejected {name}"),
                };
                let kind = match e.cause() {
                    crate::bounded::expression::Cause::Syntax(_) => crate::ErrorKind::SyntaxError,
                    crate::bounded::expression::Cause::Lexical(e) => e.kind(),
                    other => panic!("unexpected closed parse cause {other:?}"),
                };
                assert_eq!(old.kind(), kind);
                errors += 1;
                continue;
            }
        };
        parsed += 1;
        let Some(first) = source.expressions().next() else {
            continue;
        };
        let mut workspace = Plan::for_expression(first).unwrap().construct().unwrap();
        let pointer = workspace.frames.as_ptr();
        for expr in source.expressions() {
            let reference_ast = project(&source, expr.id);
            let expected = reference::expression(&reference_ast);
            exact(
                reference_ast.as_const(),
                reference::expression(&reference_ast),
            );
            match workspace.fold(expr) {
                Ok(result) => {
                    if result.value().is_some() {
                        values += 1
                    } else {
                        nonconstant += 1
                    };
                    exact(materialize(result.value()), expected);
                    assert!(result.owner.peak <= result.owner.requirements.frames);
                    workspace = result.into_workspace()
                }
                Err(error) => {
                    match error.cause() {
                        Cause::NeedsOperation(_) => operations += 1,
                        Cause::NeedsMaterialization(_) => materializers += 1,
                        other => panic!("{name}: {other:?}"),
                    };
                    workspace = error.into_workspace()
                }
            }
            assert_eq!(workspace.frames.as_ptr(), pointer);
        }
    }
    assert_eq!(parsed + errors, 32);
    println!("constant_fold_corpus templates={parsed} parse_errors={errors} values={values} nonconstant={nonconstant} needs_operation={operations} needs_materialization={materializers}");
    assert!(parsed > 0 && values > 0 && nonconstant > 0);
    // The macro-free accepted subset contains no immediate constant maps.
    #[cfg(feature = "macros")]
    assert!(materializers > 0);
    // Keep an actual materialization refusal control in every feature profile.
    let control = syntax("{{ {} }}");
    let expr = control.expressions().last().unwrap();
    let failure = Plan::for_expression(expr).unwrap().construct().unwrap().fold(expr).unwrap_err();
    assert!(matches!(failure.cause(), Cause::NeedsMaterialization(Collection::Map)));
    assert!(failure.owner.collection.is_some());
}

#[test]
fn scalar_source_operations_and_actual_semantic_errors_match_the_old_reference() {
    for input in [
        "1+2",
        "3-5",
        "true+false",
        "3*7",
        "3/2",
        "-3//2",
        "-3%2",
        "-3.0%2.0",
        "2**8",
        "2**-1",
        "none+1",
        "1//0",
        "1%0",
        "1 in 2",
        "1 not in 2",
        "1<2",
        "1<2<3",
        "false==0",
        "false<0",
        "0.0/0.0==0.0/0.0",
        "0.0/0.0<=0.0/0.0",
        "0.0/0.0",
        "1.0/0.0",
        "0 and 3",
        "false or 1",
        "'λ' and 2",
        "true and ''",
        "'' or 'λ'",
        "'a' 'λ' or 'z'",
        "2.0**0.5",
        "340282366920938463463374607431768211455+170141183460469231731687303715884105728",
    ] {
        compare(input);
    }
}
#[test]
fn scalar_chain_progress_keeps_later_materialization_refusal_and_actual_prefix() {
    for (input, refuses) in [
        ("{{ 1<2<{} }}", true),
        ("{{ 2<1<{} }}", false),
        ("{{ 1<none+1<{} }}", false),
        ("{{ false and {} }}", true),
        ("{{ x + {} }}", true),
    ] {
        let source = syntax(input);
        let h = source.expressions().last().unwrap();
        let q = Plan::for_expression(h).unwrap();
        let w = q.construct().unwrap();
        let pointer = w.frames.as_ptr();
        match w.fold(h) {
            Err(error) => {
                assert!(refuses);
                assert!(matches!(error.cause(), Cause::NeedsMaterialization(_)));
                assert!(error.owner.collection.is_some());
                let w = error.into_workspace();
                assert_eq!(w.frames.as_ptr(), pointer);
            }
            Ok(result) => {
                assert!(!refuses);
                assert_eq!(result.owner.frames.as_ptr(), pointer);
            }
        }
    }
}
#[test]
fn scalar_result_and_remaining_operation_error_keep_source_identity_and_reuse() {
    let source = syntax("{{ 2+3 }}{{ [1]+[2] }}");
    let handles: Vec<_> = source
        .expressions()
        .filter(|h| matches!(source.storage.expressions.nodes[h.id.0].kind, R::Binary(..)))
        .collect();
    assert_eq!(handles.len(), 2);
    let result = Plan::for_expression(handles[0])
        .unwrap()
        .construct()
        .unwrap()
        .fold(handles[0])
        .unwrap();
    let pointer = result.owner.frames.as_ptr();
    exact(materialize(result.value()), Some(OrdinaryValue::from(5i64)));
    let error = result.into_workspace().fold(handles[1]).unwrap_err();
    assert!(matches!(
        error.cause(),
        Cause::NeedsMaterialization(Collection::List)
    ));
    assert!(error.owner.operation.is_some());
    let result = error.into_workspace().fold(handles[0]).unwrap();
    assert_eq!(result.owner.frames.as_ptr(), pointer);
    exact(materialize(result.value()), Some(OrdinaryValue::from(5i64)));
}

#[path = "materialization_tests.rs"]
mod materialization;
