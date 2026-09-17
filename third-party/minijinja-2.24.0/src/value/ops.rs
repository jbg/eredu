use crate::error::{Error, ErrorKind};
use crate::value::merge_object::MergeSeq;
use crate::value::primitive::scalar::{self, Arithmetic, Pair, Side};
use crate::value::{DynObject, ObjectRepr, Value, ValueKind, ValueRepr};

use super::primitive::text::{repeat_length, RepeatFailure};

/// Iterator wrapper that provides exact size hints for iterators with known length.
pub(crate) struct LenIterWrap<I: Send + Sync>(pub(crate) usize, pub(crate) I);

impl<I: Iterator<Item = Value> + Send + Sync> Iterator for LenIterWrap<I> {
    type Item = Value;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        self.1.next()
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.0, Some(self.0))
    }
}

pub enum CoerceResult<'a> {
    I128(i128, i128),
    F64(f64, f64),
    Str(&'a str, &'a str),
}

pub(crate) fn as_f64(value: &Value, lossy: bool) -> Option<f64> {
    scalar::project(value).and_then(|value| scalar::as_f64(value, lossy))
}

pub fn coerce<'x>(a: &'x Value, b: &'x Value, lossy: bool) -> Option<CoerceResult<'x>> {
    if let (Some(left), Some(right)) = (scalar::project(a), scalar::project(b)) {
        return scalar::coerce(left, right, lossy, |side| {
            let value = match side {
                Side::Left => a,
                Side::Right => b,
            };
            i128::try_from(value.clone()).ok()
        })
        .map(|pair| match pair {
            Pair::I128(a, b) => CoerceResult::I128(a, b),
            Pair::F64(a, b) => CoerceResult::F64(a, b),
        });
    }
    match (&a.0, &b.0) {
        (ValueRepr::String(a, _), ValueRepr::String(b, _)) => Some(CoerceResult::Str(a, b)),
        (ValueRepr::SmallStr(a), ValueRepr::SmallStr(b)) => {
            Some(CoerceResult::Str(a.as_str(), b.as_str()))
        }
        (ValueRepr::SmallStr(a), ValueRepr::String(b, _)) => Some(CoerceResult::Str(a.as_str(), b)),
        (ValueRepr::String(a, _), ValueRepr::SmallStr(b)) => Some(CoerceResult::Str(a, b.as_str())),
        (ValueRepr::F64(a), _) => Some(CoerceResult::F64(*a, some!(as_f64(b, lossy)))),
        (_, ValueRepr::F64(b)) => Some(CoerceResult::F64(some!(as_f64(a, lossy)), *b)),
        _ => Some(CoerceResult::I128(
            some!(i128::try_from(a.clone()).ok()),
            some!(i128::try_from(b.clone()).ok()),
        )),
    }
}

fn get_offset_and_len<F: FnOnce() -> usize>(
    start: Option<i64>,
    stop: Option<i64>,
    end: F,
) -> (usize, usize) {
    let start = start.unwrap_or(0);
    if start < 0 || stop.map_or(true, |x| x < 0) {
        let end = end();
        let start = if start < 0 {
            std::cmp::max(0, end as i64 + start) as usize
        } else {
            start as usize
        };
        let stop = match stop {
            None => end,
            Some(x) if x < 0 => std::cmp::max(0, end as i64 + x) as usize,
            Some(x) => x as usize,
        };
        (start, stop.saturating_sub(start))
    } else {
        (
            start as usize,
            (stop.unwrap() as usize).saturating_sub(start as usize),
        )
    }
}

pub fn slice(value: Value, start: Value, stop: Value, step: Value) -> Result<Value, Error> {
    let start = if start.is_none() {
        None
    } else {
        Some(ok!(start.try_into()))
    };
    let stop = if stop.is_none() {
        None
    } else {
        Some(ok!(i64::try_from(stop)))
    };
    let step = if step.is_none() {
        1i64
    } else {
        ok!(i64::try_from(step))
    };
    if step == 0 {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "cannot slice by step size of 0",
        ));
    }

    let kind = value.kind();
    let error = Err(Error::new(
        ErrorKind::InvalidOperation,
        format!("value of type {kind} cannot be sliced"),
    ));

    let plan = |length| super::primitive::slice::Plan::new(length, start, stop, step)
        .expect("validated nonzero slice step");
    match value.0 {
        ValueRepr::String(..) | ValueRepr::SmallStr(_) => {
            let s = value.as_str().unwrap();
            let selected = plan(s.chars().count());
            let mut output = String::new();
            if step > 0 {
                output.extend(s.chars().skip(selected.first).step_by(step as usize).take(selected.length));
            } else {
                let skip = s.chars().count().saturating_sub(selected.first + 1);
                output.extend(s.chars().rev().skip(skip).step_by(step.unsigned_abs() as usize).take(selected.length));
            }
            Ok(Value::from(output))
        }
        ValueRepr::Bytes(ref bytes) => Ok(Value::from_bytes(plan(bytes.len()).indices().map(|i| bytes[i]).collect())),
        ValueRepr::Undefined(_) | ValueRepr::None => Ok(Value::from(Vec::<Value>::new())),
        ValueRepr::Object(obj) if matches!(obj.repr(), ObjectRepr::Seq | ObjectRepr::Iterable) => {
            if step > 0 {
                // Preserve laziness for ordinary objects without an exact length.
                let (first, count) = if let Some(length) = obj.enumerator_len() {
                    let selected = plan(length);
                    (selected.first, selected.length)
                } else {
                    let (first, span) = get_offset_and_len(start, stop, || 0);
                    (first, span / step as usize + usize::from(span % step as usize != 0))
                };
                Ok(Value::make_object_iterable(obj, move |obj| {
                    if let Some(iter) = obj.try_iter() {
                        Box::new(iter.skip(first).step_by(step as usize).take(count))
                    } else { Box::new(None.into_iter()) }
                }))
            } else {
                Ok(Value::make_object_iterable(obj.clone(), move |obj| {
                    if let Some(iter) = obj.try_iter() {
                        let values: Vec<Value> = iter.collect();
                        let selected = super::primitive::slice::Plan::new(values.len(), start, stop, step)
                            .expect("validated nonzero slice step");
                        Box::new(selected.indices().map(move |i| values[i].clone()))
                    } else { Box::new(None.into_iter()) }
                }))
            }
        }
        _ => error,
    }
}

fn primitive_integer_as_value(value: super::primitive::Integer) -> Value {
    match value {
        super::primitive::Integer::I64(value) => value.into(),
        super::primitive::Integer::I128(value) => value.into(),
    }
}

fn impossible_op(op: &str, lhs: &Value, rhs: &Value) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!(
            "tried to use {} operator on unsupported types {} and {}",
            op,
            lhs.kind(),
            rhs.kind()
        ),
    )
}

fn failed_op(op: &str, lhs: &Value, rhs: &Value) -> Error {
    Error::new(
        ErrorKind::InvalidOperation,
        format!("unable to calculate {lhs} {op} {rhs}"),
    )
}

fn numeric_result(op: Arithmetic, pair: Pair, lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    scalar::numeric(op, pair)
        .map(scalar::value)
        .map_err(|_| failed_op(op.symbol(), lhs, rhs))
}
macro_rules! math_binop {
    ($name:ident, $operation:ident, $float:tt) => {
        pub fn $name(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
            match coerce(lhs, rhs, true) {
                Some(CoerceResult::I128(a, b)) => {
                    numeric_result(Arithmetic::$operation, Pair::I128(a, b), lhs, rhs)
                }
                Some(CoerceResult::F64(a, b)) => {
                    numeric_result(Arithmetic::$operation, Pair::F64(a, b), lhs, rhs)
                }
                _ => Err(impossible_op(stringify!($float), lhs, rhs)),
            }
        }
    };
}

fn seq_concat_len(lhs: &Value, rhs: &Value) -> Option<usize> {
    lhs.len()?.checked_add(rhs.len()?)
}

fn materialize_seq_concat(lhs: &Value, rhs: &Value, len: usize) -> Result<Value, Error> {
    let mut rv = Vec::with_capacity(len);
    rv.extend(ok!(lhs.try_iter()));
    rv.extend(ok!(rhs.try_iter()));
    Ok(Value::from(rv))
}

pub fn add(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    if matches!(lhs.kind(), ValueKind::Seq | ValueKind::Iterable)
        && matches!(rhs.kind(), ValueKind::Seq | ValueKind::Iterable)
    {
        let values = vec![lhs.clone(), rhs.clone()];
        let depth = MergeSeq::depth_for_values(&values);

        // Keep sequence concatenation lazy by default.  The one case where we
        // materialize eagerly is when repeated `seq = seq + [x]` has built a
        // chain deep enough that later iteration or drop would risk one native
        // stack frame per concatenation.  Only do that for sized operands;
        // unsized iterables may represent streams and must not be consumed, so
        // `MergeSeq` flattens only its own lazy structure instead.
        if depth > MergeSeq::MAX_DEPTH {
            if let Some(len) = seq_concat_len(lhs, rhs) {
                return materialize_seq_concat(lhs, rhs, len);
            }
        }

        return Ok(Value::from_object(MergeSeq::new_iterable(values)));
    }
    match coerce(lhs, rhs, true) {
        Some(CoerceResult::I128(a, b)) => {
            numeric_result(Arithmetic::Add, Pair::I128(a, b), lhs, rhs)
        }
        Some(CoerceResult::F64(a, b)) => numeric_result(Arithmetic::Add, Pair::F64(a, b), lhs, rhs),
        Some(CoerceResult::Str(a, b)) => Ok(Value::from([a, b].concat())),
        _ => Err(impossible_op("+", lhs, rhs)),
    }
}

math_binop!(sub, Sub, -);
math_binop!(rem, Rem, %);

pub fn mul(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    if let Some((s, n)) = lhs
        .as_str()
        .map(|s| (s, rhs))
        .or_else(|| rhs.as_str().map(|s| (s, lhs)))
    {
        // Conversion still runs through the ordinary Value callback, including its discarded error.
        let (n, _) = ok!(repeat_length(s.len(), n.as_usize()).map_err(|cause| {
            Error::new(
                ErrorKind::InvalidOperation,
                match cause {
                    RepeatFailure::Integer => "strings can only be multiplied with integers",
                    RepeatFailure::TooLarge => "repeated string is too large",
                },
            )
        }));
        return Ok(Value::from(s.repeat(n)));
    } else if let Some((seq, n)) = lhs
        .as_object()
        .map(|s| (s, rhs))
        .or_else(|| rhs.as_object().map(|s| (s, lhs)))
        .filter(|x| matches!(x.0.repr(), ObjectRepr::Iterable | ObjectRepr::Seq))
    {
        return repeat_iterable(n, seq);
    }

    match coerce(lhs, rhs, true) {
        Some(CoerceResult::I128(a, b)) => {
            numeric_result(Arithmetic::Mul, Pair::I128(a, b), lhs, rhs)
        }
        Some(CoerceResult::F64(a, b)) => numeric_result(Arithmetic::Mul, Pair::F64(a, b), lhs, rhs),
        _ => Err(impossible_op(stringify!(*), lhs, rhs)),
    }
}

fn repeat_iterable(n: &Value, seq: &DynObject) -> Result<Value, Error> {
    let n = ok!(n.as_usize().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            "sequences and iterables can only be multiplied with integers",
        )
    }));

    let len = ok!(seq.enumerator_len().ok_or_else(|| {
        Error::new(
            ErrorKind::InvalidOperation,
            "cannot repeat unsized iterables",
        )
    }));

    // This is not optimal.  We only query the enumerator for the length once
    // but we support repeated iteration.  We could both lie about our length
    // here and we could actually deal with an object that changes how much
    // data it returns.  This is not really permissible so we won't try to
    // improve on this here.
    Ok(Value::make_object_iterable(seq.clone(), move |seq| {
        Box::new(LenIterWrap(
            len * n,
            (0..n).flat_map(move |_| {
                seq.try_iter().unwrap_or_else(|| {
                    Box::new(
                        std::iter::repeat(Value::from(Error::new(
                            ErrorKind::InvalidOperation,
                            "iterable did not iterate against expectations",
                        )))
                        .take(len),
                    )
                })
            }),
        ))
    }))
}

pub fn div(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    fn do_it(lhs: &Value, rhs: &Value) -> Option<Value> {
        let a = some!(as_f64(lhs, true));
        let b = some!(as_f64(rhs, true));
        Some(scalar::value(
            scalar::numeric(Arithmetic::Div, Pair::F64(a, b))
                .expect("floating division is infallible"),
        ))
    }
    do_it(lhs, rhs).ok_or_else(|| impossible_op("/", lhs, rhs))
}

pub fn int_div(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    match coerce(lhs, rhs, true) {
        Some(CoerceResult::I128(a, b)) => {
            numeric_result(Arithmetic::FloorDiv, Pair::I128(a, b), lhs, rhs)
        }
        Some(CoerceResult::F64(a, b)) => {
            numeric_result(Arithmetic::FloorDiv, Pair::F64(a, b), lhs, rhs)
        }
        _ => Err(impossible_op("//", lhs, rhs)),
    }
}

/// Implements a binary `pow` operation on values.
pub fn pow(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    match coerce(lhs, rhs, true) {
        Some(CoerceResult::I128(a, b)) => {
            numeric_result(Arithmetic::Pow, Pair::I128(a, b), lhs, rhs)
        }
        Some(CoerceResult::F64(a, b)) => numeric_result(Arithmetic::Pow, Pair::F64(a, b), lhs, rhs),
        _ => Err(impossible_op("**", lhs, rhs)),
    }
}

/// Implements an unary `neg` operation on value.
pub fn neg(val: &Value) -> Result<Value, Error> {
    if val.kind() == ValueKind::Number {
        match val.0 {
            ValueRepr::F64(x) => Ok(super::primitive::negative_float(x).into()),
            // Preserve the ordinary U128 special case before attempted conversion.
            ValueRepr::U128(x) if super::primitive::negative_special_u128(x.0).is_some() => Ok(
                Value::from(super::primitive::negative_special_u128(x.0).unwrap()),
            ),
            _ => {
                // This ordinary conversion can construct an Error before the replacement
                // InvalidOperation. Keep that boundary; checked primitives do not call it.
                if let Ok(x) = i128::try_from(val.clone()) {
                    super::primitive::negative_integer(x)
                        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "overflow"))
                        .map(primitive_integer_as_value)
                } else {
                    Err(Error::from(ErrorKind::InvalidOperation))
                }
            }
        }
    } else {
        Err(Error::from(ErrorKind::InvalidOperation))
    }
}

/// Attempts a string concatenation.
pub fn string_concat(left: Value, right: &Value) -> Value {
    Value::from(format!("{left}{right}"))
}

/// Implements a containment operation on values.
pub fn contains(container: &Value, value: &Value) -> Result<Value, Error> {
    // Special case where if the container is undefined, it cannot hold
    // values.  For strict containment checks the vm has a special case.
    if container.is_undefined() {
        return Ok(Value::from(false));
    }
    let rv = if let Some(s) = container.as_str() {
        if let Some(s2) = value.as_str() {
            super::primitive::membership::text_contains(s, s2)
        } else {
            super::primitive::membership::text_contains(s, &value.to_string())
        }
    } else if let ValueRepr::Object(ref obj) = container.0 {
        match obj.repr() {
            ObjectRepr::Plain => false,
            ObjectRepr::Map => obj.get_value(value).is_some(),
            ObjectRepr::Seq | ObjectRepr::Iterable => {
                super::primitive::membership::sequence_contains(
                    obj.try_iter().into_iter().flatten(),
                    |v| Ok::<_, std::convert::Infallible>(&v == value),
                )
                .unwrap_or_else(|never| match never {})
            }
        }
    } else {
        return Err(match scalar::invalid_container() {
            scalar::Failure::Containment => Error::new(
                ErrorKind::InvalidOperation,
                "cannot perform a containment check on this value",
            ),
            _ => unreachable!("fixed invalid-container cause"),
        });
    };
    Ok(Value::from(rv))
}

#[cfg(test)]
mod tests {
    use super::*;

    use similar_asserts::assert_eq;

    #[test]
    fn test_neg() {
        let err = neg(&Value::from(i128::MIN)).unwrap_err();
        assert_eq!(err.to_string(), "invalid operation: overflow");
    }

    #[test]
    fn test_string_repeat_size_limit() {
        let err = mul(&Value::from("ab"), &Value::from(50_000_001usize)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: repeated string is too large"
        );

        let err = mul(&Value::from("ab"), &Value::from(usize::MAX)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: repeated string is too large"
        );
    }

    #[test]
    fn test_adding() {
        let err = add(&Value::from("a"), &Value::from(42)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: tried to use + operator on unsupported types string and number"
        );

        assert_eq!(
            add(&Value::from(1), &Value::from(2)).unwrap(),
            Value::from(3)
        );
        assert_eq!(
            add(&Value::from("foo"), &Value::from("bar")).unwrap(),
            Value::from("foobar")
        );

        let err = add(&Value::from(i128::MAX), &Value::from(1)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: unable to calculate 170141183460469231731687303715884105727 + 1"
        );
    }

    #[test]
    #[cfg_attr(
        target_os = "wasi",
        ignore = "std::thread::Builder::spawn is unsupported on WASI"
    )]
    fn test_repeated_seq_add_does_not_overflow_stack() {
        // Regression test for repeated sequence concatenation, for example a
        // chat-template accumulator `messages = messages + [m]` applied for
        // many turns.  `+` may stay lazy, but must not build an unbounded chain
        // of lazy iterables, otherwise enumerating the result (for example via
        // `{{ messages | length }}` -> `Value::len()`) recurses one native frame
        // per concatenation and overflows the stack.  Run on a small (1 MiB)
        // stack so the regression aborts deterministically rather than
        // depending on the platform default stack size.
        let handle = std::thread::Builder::new()
            .stack_size(1024 * 1024)
            .spawn(|| {
                let n = 5000;
                let mut acc = Value::from(Vec::<Value>::new());
                for i in 0..n {
                    acc = add(&acc, &Value::from(vec![Value::from(i)])).unwrap();
                }
                // The crash path: `|length` -> `Value::len()`.
                assert_eq!(acc.len(), Some(n));
                // Iteration must also work and stay correct.
                assert_eq!(acc.try_iter().unwrap().count(), n);

                let mut acc = Value::make_iterable(|| (0i64..).take_while(|_| false));
                for i in 0..n {
                    acc = add(&acc, &Value::from(vec![Value::from(i)])).unwrap();
                }
                // Unsized iterables cannot be eagerly materialized to cut off
                // nesting, so iterating the lazy concat object must flatten its
                // own nested concat nodes without recursion.
                assert_eq!(acc.len(), None);
                assert_eq!(acc.try_iter().unwrap().count(), n);
            })
            .unwrap();
        handle.join().unwrap();
    }

    #[test]
    fn test_sized_iterable_add_stays_lazy() {
        struct CountingIter {
            next_count: std::sync::Arc<std::sync::atomic::AtomicUsize>,
            idx: usize,
            len: usize,
        }

        impl Iterator for CountingIter {
            type Item = Value;

            fn next(&mut self) -> Option<Self::Item> {
                if self.idx == self.len {
                    return None;
                }
                let idx = self.idx;
                self.idx += 1;
                self.next_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Some(Value::from(idx as i64))
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let remaining = self.len - self.idx;
                (remaining, Some(remaining))
            }
        }

        let next_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let lhs_next_count = next_count.clone();
        let lhs = Value::make_iterable(move || CountingIter {
            next_count: lhs_next_count.clone(),
            idx: 0,
            len: 2,
        });
        let rhs = Value::from(vec![Value::from(2), Value::from(3)]);

        let res = add(&lhs, &rhs).unwrap();
        assert_eq!(next_count.load(std::sync::atomic::Ordering::SeqCst), 0);

        // Length remains known without consuming either side.
        assert_eq!(res.len(), Some(4));
        assert_eq!(next_count.load(std::sync::atomic::Ordering::SeqCst), 0);

        let got: Vec<i64> = res
            .try_iter()
            .unwrap()
            .map(|v| i64::try_from(v).unwrap())
            .collect();
        assert_eq!(got, vec![0, 1, 2, 3]);
        assert_eq!(next_count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn test_unsized_iterable_add_stays_lazy() {
        // An unsized iterable (no exact size hint) must NOT be eagerly
        // materialized; the lazy-chaining fallback must still apply so adding
        // to a potentially-unbounded stream stays lazy.
        let unsized_iter = Value::make_iterable(|| (0i64..).take_while(|&x| x < 3));
        assert_eq!(unsized_iter.len(), None, "precondition: operand is unsized");
        let res = add(&unsized_iter, &Value::from(vec![Value::from(99)])).unwrap();
        // Result is still a lazy iterable with unknown length...
        assert_eq!(res.kind(), ValueKind::Iterable);
        assert_eq!(res.len(), None);
        // ...but iterates to the correct concatenated contents.
        let got: Vec<i64> = res
            .try_iter()
            .unwrap()
            .map(|v| i64::try_from(v).unwrap())
            .collect();
        assert_eq!(got, vec![0, 1, 2, 99]);
    }

    #[test]
    fn test_subtracting() {
        let err = sub(&Value::from("a"), &Value::from(42)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: tried to use - operator on unsupported types string and number"
        );

        let err = sub(&Value::from("foo"), &Value::from("bar")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: tried to use - operator on unsupported types string and string"
        );

        assert_eq!(
            sub(&Value::from(2), &Value::from(1)).unwrap(),
            Value::from(1)
        );
    }

    #[test]
    fn test_dividing() {
        let err = div(&Value::from("a"), &Value::from(42)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: tried to use / operator on unsupported types string and number"
        );

        let err = div(&Value::from("foo"), &Value::from("bar")).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: tried to use / operator on unsupported types string and string"
        );

        assert_eq!(
            div(&Value::from(100), &Value::from(2)).unwrap(),
            Value::from(50.0)
        );

        let err = int_div(&Value::from(i128::MIN), &Value::from(-1i128)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "invalid operation: unable to calculate -170141183460469231731687303715884105728 // -1"
        );
    }

    #[test]
    fn test_concat() {
        assert_eq!(
            string_concat(Value::from("foo"), &Value::from(42)),
            Value::from("foo42")
        );
        assert_eq!(
            string_concat(Value::from(23), &Value::from(42)),
            Value::from("2342")
        );
    }

    #[test]
    fn test_slicing() {
        let v = Value::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

        // [::] - full slice
        assert_eq!(
            slice(v.clone(), Value::from(()), Value::from(()), Value::from(())).unwrap(),
            Value::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        );

        // [::2] - every 2nd element
        assert_eq!(
            slice(v.clone(), Value::from(()), Value::from(()), Value::from(2)).unwrap(),
            Value::from(vec![0, 2, 4, 6, 8])
        );

        // [1:2:2] - slice with start, stop, step
        assert_eq!(
            slice(v.clone(), Value::from(1), Value::from(2), Value::from(2)).unwrap(),
            Value::from(vec![1])
        );

        // [::-2] - reverse with step of 2
        assert_eq!(
            slice(v.clone(), Value::from(()), Value::from(()), Value::from(-2)).unwrap(),
            Value::from(vec![9, 7, 5, 3, 1])
        );

        // [:-8:] - from index 0 to -8
        assert_eq!(
            slice(v.clone(), Value::from(()), Value::from(-8), Value::from(())).unwrap(),
            Value::from(vec![0, 1])
        );

        // [-8::] - from index -8 to the end
        assert_eq!(
            slice(v.clone(), Value::from(-8), Value::from(()), Value::from(())).unwrap(),
            Value::from(vec![2, 3, 4, 5, 6, 7, 8, 9])
        );

        // [-11::] - from index -11 to the end, which is the same as [::]
        // because the start index is before the start of the vector
        assert_eq!(
            slice(
                v.clone(),
                Value::from(-11),
                Value::from(()),
                Value::from(())
            )
            .unwrap(),
            Value::from(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        );

        // [:-11:] - from index -11 to the end, which is the same as [:0:]
        // because the end index is before the start of the vector
        assert_eq!(
            slice(
                v.clone(),
                Value::from(()),
                Value::from(-11),
                Value::from(())
            )
            .unwrap(),
            Value::from(Vec::<usize>::new())
        );

        // [2::-2] - from index 2 to start, reverse with step of 2
        assert_eq!(
            slice(v.clone(), Value::from(2), Value::from(()), Value::from(-2)).unwrap(),
            Value::from(vec![2, 0])
        );

        // [4:2:-2] - from index 4 to 2, reverse with step of 2
        assert_eq!(
            slice(v.clone(), Value::from(4), Value::from(2), Value::from(-2)).unwrap(),
            Value::from(vec![4])
        );

        // [8:3:-2] - from index 8 to 3, reverse with step of 2
        assert_eq!(
            slice(v.clone(), Value::from(8), Value::from(3), Value::from(-2)).unwrap(),
            Value::from(vec![8, 6, 4])
        );
    }

    #[test]
    fn test_string_slicing() {
        let s = Value::from("abcdefghij");

        // [::] - full slice
        assert_eq!(
            slice(s.clone(), Value::from(()), Value::from(()), Value::from(())).unwrap(),
            Value::from("abcdefghij")
        );

        // [::2] - every 2nd character
        assert_eq!(
            slice(s.clone(), Value::from(()), Value::from(()), Value::from(2)).unwrap(),
            Value::from("acegi")
        );

        // [1:2:2] - slice with start, stop, step
        assert_eq!(
            slice(s.clone(), Value::from(1), Value::from(2), Value::from(2)).unwrap(),
            Value::from("b")
        );

        // [::-2] - reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(()), Value::from(()), Value::from(-2)).unwrap(),
            Value::from("jhfdb")
        );

        // [2::-2] - from index 2 to start, reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(2), Value::from(()), Value::from(-2)).unwrap(),
            Value::from("ca")
        );

        // [4:2:-2] - from index 4 to 2, reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(4), Value::from(2), Value::from(-2)).unwrap(),
            Value::from("e")
        );

        // [8:3:-2] - from index 8 to 3, reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(8), Value::from(3), Value::from(-2)).unwrap(),
            Value::from("ige")
        );
    }

    #[test]
    fn test_bytes_slicing() {
        let s = Value::from_bytes(b"abcdefghij".to_vec());

        // [::] - full slice
        assert_eq!(
            slice(s.clone(), Value::from(()), Value::from(()), Value::from(())).unwrap(),
            Value::from_bytes(b"abcdefghij".to_vec())
        );

        // [::2] - every 2nd character
        assert_eq!(
            slice(s.clone(), Value::from(()), Value::from(()), Value::from(2)).unwrap(),
            Value::from_bytes(b"acegi".to_vec())
        );

        // [1:2:2] - slice with start, stop, step
        assert_eq!(
            slice(s.clone(), Value::from(1), Value::from(2), Value::from(2)).unwrap(),
            Value::from_bytes(b"b".to_vec())
        );

        // [::-2] - reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(()), Value::from(()), Value::from(-2)).unwrap(),
            Value::from_bytes(b"jhfdb".to_vec())
        );

        // [2::-2] - from index 2 to start, reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(2), Value::from(()), Value::from(-2)).unwrap(),
            Value::from_bytes(b"ca".to_vec())
        );

        // [4:2:-2] - from index 4 to 2, reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(4), Value::from(2), Value::from(-2)).unwrap(),
            Value::from_bytes(b"e".to_vec())
        );

        // [8:3:-2] - from index 8 to 3, reverse with step of 2
        assert_eq!(
            slice(s.clone(), Value::from(8), Value::from(3), Value::from(-2)).unwrap(),
            Value::from_bytes(b"ige".to_vec())
        );
    }
}
