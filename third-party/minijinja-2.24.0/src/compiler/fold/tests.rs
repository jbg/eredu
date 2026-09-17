#![forbid(unsafe_code)]
use super::reference;
use crate::compiler::tokens::Span;
use crate::compiler::{ast::*, parser};
use crate::value::{ops, DynObject, Enumerator, Object, ObjectRepr, Value, ValueRepr};
use std::{
    cmp::Ordering,
    fmt,
    sync::{Arc, Mutex},
};
fn constant(value: Value) -> Expr<'static> {
    Expr::Const(Spanned::new(Const { value }, Span::default()))
}
fn binary(op: BinOpKind, left: Expr<'static>, right: Expr<'static>) -> Expr<'static> {
    Expr::BinOp(Spanned::new(BinOp { op, left, right }, Span::default()))
}
fn unary(op: UnaryOpKind, expr: Expr<'static>) -> Expr<'static> {
    Expr::UnaryOp(Spanned::new(UnaryOp { op, expr }, Span::default()))
}
pub(crate) fn exact(a: Option<Value>, b: Option<Value>) {
    match (a, b) {
        (None, None) => (),
        (Some(a), Some(b)) => match (&a.0, &b.0) {
            (ValueRepr::None, ValueRepr::None) => (),
            (ValueRepr::Bool(a), ValueRepr::Bool(b)) => assert_eq!(a, b),
            (ValueRepr::U64(a), ValueRepr::U64(b)) => assert_eq!(a, b),
            (ValueRepr::I64(a), ValueRepr::I64(b)) => assert_eq!(a, b),
            (ValueRepr::U128(a), ValueRepr::U128(b)) => assert_eq!({ a.0 }, { b.0 }),
            (ValueRepr::I128(a), ValueRepr::I128(b)) => assert_eq!({ a.0 }, { b.0 }),
            (ValueRepr::F64(a), ValueRepr::F64(b)) => assert_eq!(a.to_bits(), b.to_bits()),
            (ValueRepr::Object(_), ValueRepr::Object(_)) => {
                match (a.downcast_object_ref::<Vec<Value>>(), b.downcast_object_ref::<Vec<Value>>()) {
                    (Some(a), Some(b)) => {
                        assert_eq!(a.len(), b.len());
                        for (a, b) in a.iter().zip(b) {
                            exact(Some(a.clone()), Some(b.clone()));
                        }
                    }
                    (None, None) => assert_eq!(a, b),
                    _ => panic!("different concrete sequence representations: {a:?} {b:?}"),
                }
            }
            (ValueRepr::Undefined(_), ValueRepr::Undefined(_)) => {
                assert!(a.is_undefined() && b.is_undefined())
            }
            _ if a.as_str().is_some() && b.as_str().is_some() => {
                assert_eq!(a.as_str(), b.as_str());
                assert_eq!(a.is_safe(), b.is_safe())
            }
            _ => panic!("different exact representations: {a:?}, {b:?}"),
        },
        (a, b) => panic!("different fold results {a:?} {b:?}"),
    }
}
#[test]
fn literal_and_unary_ordinary_types_float_bits_and_conversion_errors_match_reference() {
    let mut values = vec![
        Value::from(()),
        Value::from(true),
        Value::from(false),
        Value::from(""),
        Value::from("λ"),
        Value::from(0u64),
        Value::from(u64::MAX),
        Value::from(i64::MIN),
        Value::from(i128::MIN),
        Value::from(i128::MAX),
        Value::from((1u128 << 127) - 1),
        Value::from(1u128 << 127),
        Value::from((1u128 << 127) + 1),
        Value::from(u128::MAX),
        Value::UNDEFINED,
    ];
    values.extend(
        [
            0.0f64,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::from_bits(0x7ff8_0000_0000_0137),
            f64::from_bits(0xfff8_0000_0000_0421),
        ]
        .map(Value::from),
    );
    for value in values {
        exact(
            constant(value.clone()).as_const(),
            reference::expression(&constant(value.clone())),
        );
        assert_eq!(value.is_true(), reference::is_true(&value));
        exact(
            unary(UnaryOpKind::Not, constant(value.clone())).as_const(),
            reference::expression(&unary(UnaryOpKind::Not, constant(value.clone()))),
        );
        exact(
            unary(UnaryOpKind::Neg, constant(value.clone())).as_const(),
            reference::expression(&unary(UnaryOpKind::Neg, constant(value.clone()))),
        );
        match (ops::neg(&value), reference::neg(&value)) {
            (Ok(a), Ok(b)) => exact(Some(a), Some(b)),
            (Err(a), Err(b)) => {
                assert_eq!(a.kind(), b.kind());
                assert_eq!(a.to_string(), b.to_string());
                assert_eq!(
                    std::error::Error::source(&a).is_some(),
                    std::error::Error::source(&b).is_some()
                )
            }
            _ => panic!("negation semantic failure changed"),
        }
    }
}
#[test]
fn ordinary_binary_comparison_and_direct_collection_eligibility_keep_exact_semantics() {
    for source in [
        "1+2",
        "1/0",
        "2 ** 1000",
        "'x'~1",
        "0 and 3",
        "'' or 4",
        "true and 4",
        "0 or 4",
        "x + [1, 2]",
        "x < [1, 2]",
        "2 < 1 < ('x'~1)",
        "1 == 1 < 2",
        "[1, 2, 3]",
        "[1+2]",
        "{'x':1,'x':2}",
        "{'x':1+2}",
        "[]",
        "{}",
        "(1 if true else 2)",
    ] {
        let expr = parser::parse_expr(source).unwrap();
        exact(expr.as_const(), reference::expression(&expr));
    }
    let list = List {
        items: vec![
            constant(Value::from(1)),
            unary(UnaryOpKind::Neg, constant(Value::from(2))),
        ],
    };
    exact(list.as_const(), reference::list(&list));
    assert!(list.as_const().is_none());
    // Ordinary public AST permits unequal map lengths; preserve keys/values scan and zip.
    for extra in [
        constant(Value::from(2)),
        Expr::Var(Spanned::new(Var { id: "x" }, Span::default())),
    ] {
        let map = Map {
            keys: vec![constant(Value::from("x")), extra],
            values: vec![constant(Value::from(1))],
        };
        exact(map.as_const(), reference::map(&map));
    }
    let expr = parser::parse_expr("0 and 3").unwrap();
    exact(expr.as_const(), Some(Value::from(false)));
}
#[derive(Debug)]
struct Probe {
    events: Arc<Mutex<Vec<&'static str>>>,
    truth: bool,
}
impl Object for Probe {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Plain
    }
    fn is_true(self: &Arc<Self>) -> bool {
        self.events.lock().unwrap().push("truth");
        self.truth
    }
    fn custom_cmp(self: &Arc<Self>, _: &DynObject) -> Option<Ordering> {
        self.events.lock().unwrap().push("compare");
        Some(Ordering::Less)
    }
    fn enumerate(self: &Arc<Self>) -> Enumerator {
        self.events.lock().unwrap().push("enumerate");
        Enumerator::Values(vec![Value::from(1)])
    }
    fn render(self: &Arc<Self>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.events.lock().unwrap().push("render");
        f.write_str("probe")
    }
}
#[test]
fn ordinary_callbacks_preserve_eager_binary_and_short_comparison_order() {
    let events = Arc::new(Mutex::new(Vec::new()));
    let probe = || {
        constant(Value::from_object(Probe {
            events: events.clone(),
            truth: true,
        }))
    };
    let missing = || Expr::Var(Spanned::new(Var { id: "missing" }, Span::default()));
    let exprs = [
        binary(
            BinOpKind::Add,
            missing(),
            binary(BinOpKind::Concat, probe(), constant(Value::from("!"))),
        ),
        binary(
            BinOpKind::ScAnd,
            unary(UnaryOpKind::Not, probe()),
            unary(UnaryOpKind::Not, probe()),
        ),
        Expr::Compare(Spanned::new(
            Compare {
                expr: constant(Value::from(2)),
                ops: vec![
                    CompareOp {
                        op: CompareOpKind::Lt,
                        expr: constant(Value::from(1)),
                    },
                    CompareOp {
                        op: CompareOpKind::Lt,
                        expr: probe(),
                    },
                ],
            },
            Span::default(),
        )),
        binary(BinOpKind::Lt, probe(), probe()),
        binary(BinOpKind::In, constant(Value::from(1)), probe()),
    ];
    for expr in &exprs {
        events.lock().unwrap().clear();
        let expected = reference::expression(expr);
        let old = events.lock().unwrap().clone();
        events.lock().unwrap().clear();
        let actual = expr.as_const();
        let new = events.lock().unwrap().clone();
        assert_eq!(new, old);
        exact(actual, expected);
    }
    events.lock().unwrap().clear();
    assert!(exprs[0].as_const().is_none());
    assert_eq!(*events.lock().unwrap(), vec!["render"]);
    events.lock().unwrap().clear();
    exact(exprs[2].as_const(), Some(Value::from(false)));
    assert!(events.lock().unwrap().is_empty());
}
#[path = "walk_tests.rs"]
mod walk;
include!("../../bounded/expression/tests/fixtures.rs");
#[test]
fn all_released_ordinary_ast_expressions_keep_pre_refactor_fold_results() {
    let (mut parsed, mut rejected, mut expressions) = (0, 0, 0);
    for (name, source) in FIXTURES {
        match parser::parse(source, name, Default::default(), Default::default()) {
            Ok(ast) => {
                parsed += 1;
                walk::statement(&ast, &mut |expr| {
                    exact(expr.as_const(), reference::expression(expr));
                    expressions += 1
                })
            }
            Err(_) => rejected += 1,
        }
    }
    assert_eq!(parsed + rejected, 32);
    assert!(parsed > 0 && expressions > 0);
    println!("ordinary_constant_fold_corpus templates={parsed} rejected={rejected} expressions={expressions}");
}
