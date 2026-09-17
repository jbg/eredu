#![forbid(unsafe_code)]
use super::*;
use crate::compiler::fold::tests::exact;
use crate::compiler::fold::{reference, scalar_reference as old};
use crate::value::ops;
use crate::Error;
use std::error::Error as _;
fn corpus() -> [Scalar; 50] {
    [
        Scalar::None,
        Scalar::Bool(false),
        Scalar::Bool(true),
        Scalar::U64(0u64),
        Scalar::U64(1u64),
        Scalar::U64(4294967295u64),
        Scalar::U64(4294967296u64),
        Scalar::U64(9007199254740991u64),
        Scalar::U64(9007199254740992u64),
        Scalar::U64(9007199254740993u64),
        Scalar::U64(18446744073709551615u64),
        Scalar::U128(0u128),
        Scalar::U128(1u128),
        Scalar::U128(18446744073709551615u128),
        Scalar::U128(18446744073709551616u128),
        Scalar::U128(170141183460469231731687303715884105727u128),
        Scalar::U128(170141183460469231731687303715884105728u128),
        Scalar::U128(170141183460469231731687303715884105729u128),
        Scalar::U128(340282366920938463463374607431768211455u128),
        Scalar::I64(-9223372036854775808i64),
        Scalar::I64(-1i64),
        Scalar::I64(0i64),
        Scalar::I64(1i64),
        Scalar::I64(9223372036854775807i64),
        Scalar::I128(-170141183460469231731687303715884105728i128),
        Scalar::I128(-170141183460469231731687303715884105727i128),
        Scalar::I128(-18446744073709551616i128),
        Scalar::I128(-1i128),
        Scalar::I128(0i128),
        Scalar::I128(1i128),
        Scalar::I128(170141183460469231731687303715884105727i128),
        Scalar::F64(f64::from_bits(0x0000000000000000)),
        Scalar::F64(f64::from_bits(0x8000000000000000)),
        Scalar::F64(f64::from_bits(0x3ff0000000000000)),
        Scalar::F64(f64::from_bits(0xbff0000000000000)),
        Scalar::F64(f64::from_bits(0x3fe0000000000000)),
        Scalar::F64(f64::from_bits(0xbfe0000000000000)),
        Scalar::F64(f64::from_bits(0x0000000000000001)),
        Scalar::F64(f64::from_bits(0x8000000000000001)),
        Scalar::F64(f64::from_bits(0x000fffffffffffff)),
        Scalar::F64(f64::from_bits(0x0010000000000000)),
        Scalar::F64(f64::from_bits(0x7fefffffffffffff)),
        Scalar::F64(f64::from_bits(0xffefffffffffffff)),
        Scalar::F64(f64::from_bits(0x7ff0000000000000)),
        Scalar::F64(f64::from_bits(0xfff0000000000000)),
        Scalar::F64(f64::from_bits(0x7ff8000000000000)),
        Scalar::F64(f64::from_bits(0x7ff8000000000001)),
        Scalar::F64(f64::from_bits(0xfff8000000000001)),
        Scalar::F64(f64::from_bits(0x7ff0000000000001)),
        Scalar::F64(f64::from_bits(0xfff0000000000001)),
    ]
}
fn result(actual: Result<Value, Error>, expected: Result<Value, Error>) {
    match (actual, expected) {
        (Ok(a), Ok(b)) => exact(Some(a), Some(b)),
        (Err(a), Err(b)) => {
            assert_eq!(a.kind(), b.kind());
            assert_eq!(a.detail(), b.detail());
            assert_eq!(a.to_string(), b.to_string());
            assert_eq!(a.name(), b.name());
            assert_eq!(a.line(), b.line());
            assert!(a.source().is_none() && b.source().is_none());
        }
        (a, b) => panic!("different operation results {a:?} {b:?}"),
    }
}
fn pair(actual: Option<Pair>, expected: Option<old::ops::CoerceResult<'_>>) {
    match (actual, expected) {
        (None, None) => (),
        (Some(Pair::I128(a, b)), Some(old::ops::CoerceResult::I128(x, y))) => {
            assert_eq!((a, b), (x, y))
        }
        (Some(Pair::F64(a, b)), Some(old::ops::CoerceResult::F64(x, y))) => {
            assert_eq!((a.to_bits(), b.to_bits()), (x.to_bits(), y.to_bits()))
        }
        _ => panic!("different coercion representation"),
    }
}
#[test]
fn scalar_coercion_uses_exact_original_round_trips_and_representations() {
    let mut comparisons = 0;
    for a in corpus() {
        for b in corpus() {
            let (av, bv) = (value(a), value(b));
            for lossy in [false, true] {
                pair(
                    coerce(a, b, lossy, |side| {
                        signed(if side == Side::Left { a } else { b })
                    }),
                    old::ops::coerce(&av, &bv, lossy),
                );
                match ops::coerce(&av, &bv, lossy) {
                    Some(ops::CoerceResult::I128(x, y)) => {
                        pair(Some(Pair::I128(x, y)), old::ops::coerce(&av, &bv, lossy))
                    }
                    Some(ops::CoerceResult::F64(x, y)) => {
                        pair(Some(Pair::F64(x, y)), old::ops::coerce(&av, &bv, lossy))
                    }
                    None => pair(None, old::ops::coerce(&av, &bv, lossy)),
                    _ => panic!("scalar became text"),
                }
                comparisons += 1;
            }
        }
    }
    assert_eq!(comparisons, 50 * 50 * 2);
    println!("scalar_coercion_reference comparisons={comparisons}");
}
#[test]
fn scalar_conversion_callback_is_lazy_ordered_and_keeps_real_conversion_failure() {
    let cases = [
        (Scalar::U128(u128::MAX), Scalar::U128(0), vec![]),
        (Scalar::F64(1.0), Scalar::None, vec![]),
        (Scalar::None, Scalar::Bool(true), vec![Side::Left]),
        (
            Scalar::Bool(true),
            Scalar::None,
            vec![Side::Left, Side::Right],
        ),
        (Scalar::U128(u128::MAX), Scalar::I64(0), vec![Side::Left]),
        (
            Scalar::I64(1),
            Scalar::U64(2),
            vec![Side::Left, Side::Right],
        ),
    ];
    for (a, b, expected) in cases {
        let mut seen = Vec::new();
        let (av, bv) = (value(a), value(b));
        let actual = coerce(a, b, false, |side| {
            seen.push(side);
            i128::try_from(if side == Side::Left {
                av.clone()
            } else {
                bv.clone()
            })
            .ok()
        });
        pair(actual, old::ops::coerce(&av, &bv, false));
        assert_eq!(seen, expected);
    }
}
#[test]
fn scalar_arithmetic_cartesian_results_errors_and_float_bits_match_untouched_operations() {
    type Op = fn(&Value, &Value) -> Result<Value, Error>;
    let operations: [(Arithmetic, Op, Op); 7] = [
        (Arithmetic::Add, ops::add, old::ops::add),
        (Arithmetic::Sub, ops::sub, old::ops::sub),
        (Arithmetic::Mul, ops::mul, old::ops::mul),
        (Arithmetic::Div, ops::div, old::ops::div),
        (Arithmetic::FloorDiv, ops::int_div, old::ops::int_div),
        (Arithmetic::Rem, ops::rem, old::ops::rem),
        (Arithmetic::Pow, ops::pow, old::ops::pow),
    ];
    let mut comparisons = 0;
    for a in corpus() {
        for b in corpus() {
            let (av, bv) = (value(a), value(b));
            for (op, current, previous) in operations {
                result(current(&av, &bv), previous(&av, &bv));
                exact(
                    arithmetic(op, a, b, |side| {
                        signed(if side == Side::Left { a } else { b })
                    })
                    .ok()
                    .map(value),
                    previous(&av, &bv).ok(),
                );
                comparisons += 1;
            }
        }
    }
    assert_eq!(comparisons, 50 * 50 * 7);
    println!("scalar_arithmetic_reference comparisons={comparisons}");
}
#[test]
fn scalar_equality_and_total_order_remain_distinct_against_untouched_comparisons() {
    let mut comparisons = 0;
    for a in corpus() {
        for b in corpus() {
            let (av, bv) = (value(a), value(b));
            let conversion = |side| signed(if side == Side::Left { a } else { b });
            assert_eq!(av == bv, old::equal(&av, &bv));
            assert_eq!(equal(a, b, conversion), old::equal(&av, &bv));
            assert_eq!(av.cmp(&bv), old::compare(&av, &bv));
            assert_eq!(compare(a, b, conversion), old::compare(&av, &bv));
            assert_eq!(av.partial_cmp(&bv), Some(old::compare(&av, &bv)));
            comparisons += 1;
        }
    }
    assert_eq!(comparisons, 50 * 50);
    println!("scalar_comparison_reference pairs={comparisons}");
}
#[test]
fn scalar_unsigned_boolean_nan_and_euclidean_quirks_are_not_normalized() {
    assert!(Value::from(false) == Value::from(0u64));
    assert_eq!(Value::from(false).cmp(&Value::from(0u64)), Ordering::Less);
    let n = Value::from(f64::from_bits(0xfff8000000000001));
    assert!(!(n == n));
    assert_eq!(n.cmp(&n), Ordering::Equal);
    assert_eq!(Value::from(-0.0).cmp(&Value::from(0.0)), Ordering::Equal);
    exact(
        ops::add(&Value::from(u128::MAX), &Value::from(0u128)).ok(),
        Some(Value::from(-1i64)),
    );
    assert!(ops::add(&Value::from(u128::MAX), &Value::from(0u64)).is_err());
    assert!(Value::from(u128::MAX) > Value::from(0u128));
    exact(
        ops::rem(&Value::from(-3i64), &Value::from(2i64)).ok(),
        Some(Value::from(1i64)),
    );
    exact(
        ops::rem(&Value::from(-3.0), &Value::from(2.0)).ok(),
        Some(Value::from(-1.0)),
    );
    assert!(ops::rem(&Value::from(i128::MIN), &Value::from(-1i128)).is_err());
    assert!(ops::int_div(&Value::from(i128::MIN), &Value::from(-1i128)).is_err());
}
#[test]
fn ordinary_string_bytes_and_sequence_prebranches_match_old_materialization() {
    let pairs = [
        (Value::from("λ"), Value::from("x")),
        (Value::from("ab"), Value::from(2u64)),
        (Value::from_bytes(b"ab".to_vec()), Value::from(2u64)),
        (Value::from("a"), Value::from(-1i64)),
        (Value::from(vec![1, 2]), Value::from(vec![3])),
        (Value::from(vec![1, 2]), Value::from(2u64)),
    ];
    for (a, b) in pairs {
        result(ops::add(&a, &b), old::ops::add(&a, &b));
        result(ops::mul(&a, &b), old::ops::mul(&a, &b));
        result(ops::contains(&a, &b), old::ops::contains(&a, &b));
        assert_eq!(a == b, old::equal(&a, &b));
        assert_eq!(a.cmp(&b), old::compare(&a, &b));
    }
}
#[test]
fn shared_logical_selector_preserves_lazy_truth_and_exact_operand_selection() {
    use std::cell::RefCell;
    for op in [Logical::And, Logical::Or] {
        for a in [false, true] {
            for b in [false, true] {
                let calls = RefCell::new(Vec::new());
                let selected = logical(
                    op,
                    || {
                        calls.borrow_mut().push("left");
                        a
                    },
                    || {
                        calls.borrow_mut().push("right");
                        b
                    },
                );
                let expected = match op {
                    Logical::And => {
                        if a && b {
                            Selection::Right
                        } else {
                            Selection::False
                        }
                    }
                    Logical::Or => {
                        if a {
                            Selection::Left
                        } else {
                            Selection::Right
                        }
                    }
                };
                assert_eq!(selected, expected);
                assert_eq!(
                    *calls.borrow(),
                    if matches!(op, Logical::And) && a {
                        vec!["left", "right"]
                    } else {
                        vec!["left"]
                    }
                );
            }
        }
    }
    for source in [
        "0 and 3",
        "false or 1",
        "'λ' and 2",
        "true and ''",
        "'' or 'λ'",
        "1 in 2",
        "1 not in 2",
    ] {
        let expr = crate::compiler::parser::parse_expr(source).unwrap();
        exact(expr.as_const(), reference::expression(&expr));
    }
}

#[derive(Debug)]
struct Probe {
    events: std::sync::Arc<std::sync::Mutex<Vec<&'static str>>>,
    repr: crate::value::ObjectRepr,
    truth: bool,
    custom: bool,
    panic_render: bool,
    dropped: &'static str,
}
impl Drop for Probe {
    fn drop(&mut self) {
        self.events.lock().unwrap().push(self.dropped);
    }
}
impl crate::value::Object for Probe {
    fn repr(self: &std::sync::Arc<Self>) -> crate::value::ObjectRepr {
        self.events.lock().unwrap().push("repr");
        self.repr
    }
    fn is_true(self: &std::sync::Arc<Self>) -> bool {
        self.events.lock().unwrap().push("truth");
        self.truth
    }
    fn custom_cmp(self: &std::sync::Arc<Self>, _: &crate::value::DynObject) -> Option<Ordering> {
        self.events.lock().unwrap().push("custom");
        self.custom.then_some(Ordering::Less)
    }
    fn enumerate(self: &std::sync::Arc<Self>) -> crate::value::Enumerator {
        self.events.lock().unwrap().push("enumerate");
        match self.repr {
            crate::value::ObjectRepr::Map => crate::value::Enumerator::Str(&["k"]),
            crate::value::ObjectRepr::Seq => crate::value::Enumerator::Seq(2),
            _ => crate::value::Enumerator::Empty,
        }
    }
    fn get_value(self: &std::sync::Arc<Self>, key: &Value) -> Option<Value> {
        self.events.lock().unwrap().push("get");
        if key.as_str() == Some("k") || key.as_usize().is_some_and(|x| x < 2) {
            Some(Value::from(1))
        } else {
            None
        }
    }
    fn render(self: &std::sync::Arc<Self>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.events.lock().unwrap().push("render");
        assert!(!self.panic_render, "actual ordinary formatter panic");
        f.write_str("probe")
    }
}
fn dynamic_trace(
    previous: bool,
    repr: crate::value::ObjectRepr,
    custom: bool,
) -> Vec<&'static str> {
    use std::sync::{Arc, Mutex};
    let events = Arc::new(Mutex::new(Vec::new()));
    let a = Value::from_object(Probe {
        events: events.clone(),
        repr,
        truth: true,
        custom,
        panic_render: false,
        dropped: "drop-left",
    });
    let b = Value::from_object(Probe {
        events: events.clone(),
        repr,
        truth: false,
        custom,
        panic_render: false,
        dropped: "drop-right",
    });
    let added = if previous {
        old::ops::add(&a, &b)
    } else {
        ops::add(&a, &b)
    };
    if let Ok(value) = &added {
        drop(value.to_string());
    }
    drop(added);
    let repeated = if previous {
        old::ops::mul(&a, &Value::from(2))
    } else {
        ops::mul(&a, &Value::from(2))
    };
    if let Ok(value) = &repeated {
        drop(value.to_string());
    }
    drop(repeated);
    if previous {
        let _ = old::equal(&a, &b);
        let _ = old::compare(&a, &b);
        drop(old::ops::contains(&a, &Value::from(1)));
    } else {
        let _ = a == b;
        let _ = a.cmp(&b);
        drop(ops::contains(&a, &Value::from(1)));
    }
    drop(b);
    drop(a);
    let result = events.lock().unwrap().clone();
    result
}
#[test]
fn ordinary_dynamic_prebranches_callbacks_and_last_value_drop_match_reference() {
    use crate::value::ObjectRepr;
    for repr in [ObjectRepr::Plain, ObjectRepr::Map, ObjectRepr::Seq] {
        for custom in [false, true] {
            let old = dynamic_trace(true, repr, custom);
            let actual = dynamic_trace(false, repr, custom);
            assert_eq!(actual, old);
            assert_eq!(&actual[actual.len() - 2..], ["drop-right", "drop-left"]);
        }
    }
}
#[test]
fn ordinary_logical_selected_value_and_formatter_unwind_keep_actual_arc_custody() {
    use crate::compiler::{ast::*, tokens::Span};
    use std::sync::{Arc, Mutex};
    fn expr(op: BinOpKind, left: Value, right: Value) -> Expr<'static> {
        let literal = |value| Expr::Const(Spanned::new(Const { value }, Span::default()));
        Expr::BinOp(Spanned::new(
            BinOp {
                op,
                left: literal(left),
                right: literal(right),
            },
            Span::default(),
        ))
    }
    for op in [BinOpKind::ScAnd, BinOpKind::ScOr, BinOpKind::Concat] {
        let run = |previous| {
            let events = Arc::new(Mutex::new(Vec::new()));
            let left = Arc::new(Probe {
                events: events.clone(),
                repr: crate::value::ObjectRepr::Plain,
                truth: true,
                custom: false,
                panic_render: matches!(op, BinOpKind::Concat),
                dropped: "left",
            });
            let weak = Arc::downgrade(&left);
            let left_value = Value::from_dyn_object(left.clone());
            drop(left);
            let right = Value::from_object(Probe {
                events: events.clone(),
                repr: crate::value::ObjectRepr::Plain,
                truth: true,
                custom: false,
                panic_render: false,
                dropped: "right",
            });
            let expression = expr(op, left_value, right);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if previous {
                    reference::expression(&expression)
                } else {
                    expression.as_const()
                }
            }));
            assert_eq!(result.is_err(), matches!(op, BinOpKind::Concat));
            assert!(weak.upgrade().is_some());
            drop(expression);
            if matches!(op, BinOpKind::ScOr) {
                assert!(weak.upgrade().is_some());
            } else {
                assert!(weak.upgrade().is_none());
            }
            drop(result);
            assert!(weak.upgrade().is_none());
            let trace = events.lock().unwrap().clone();
            trace
        };
        assert_eq!(run(false), run(true));
    }
}
