//! Prechange operation bodies with only independent formatter/usize seams redirected.
#![forbid(unsafe_code)]
#![allow(dead_code)]
use crate::error::{Error, ErrorKind};
use crate::value::merge_object::MergeSeq;
use crate::value::{DynObject, ObjectRepr, Value, ValueKind, ValueRepr};

const MIN_I128_AS_POS_U128: u128 = 170141183460469231731687303715884105728;
const MAX_REPEATED_STRING_LEN: usize = 100_000_000;

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
    macro_rules! checked {
        ($expr:expr, $ty:ty) => {{
            let rv = $expr as f64;
            return if lossy || rv as $ty == $expr {
                Some(rv)
            } else {
                None
            };
        }};
    }

    Some(match value.0 {
        ValueRepr::Bool(x) => x as i64 as f64,
        ValueRepr::U64(x) => checked!(x, u64),
        ValueRepr::U128(x) => checked!(x.0, u128),
        ValueRepr::I64(x) => checked!(x, i64),
        ValueRepr::I128(x) => checked!(x.0, i128),
        ValueRepr::F64(x) => x,
        _ => return None,
    })
}

pub fn coerce<'x>(a: &'x Value, b: &'x Value, lossy: bool) -> Option<CoerceResult<'x>> {
    match (&a.0, &b.0) {
        // equal mappings are trivial
        (ValueRepr::U64(a), ValueRepr::U64(b)) => Some(CoerceResult::I128(*a as i128, *b as i128)),
        (ValueRepr::U128(a), ValueRepr::U128(b)) => {
            Some(CoerceResult::I128(a.0 as i128, b.0 as i128))
        }
        (ValueRepr::String(a, _), ValueRepr::String(b, _)) => Some(CoerceResult::Str(a, b)),
        (ValueRepr::SmallStr(a), ValueRepr::SmallStr(b)) => {
            Some(CoerceResult::Str(a.as_str(), b.as_str()))
        }
        (ValueRepr::SmallStr(a), ValueRepr::String(b, _)) => Some(CoerceResult::Str(a.as_str(), b)),
        (ValueRepr::String(a, _), ValueRepr::SmallStr(b)) => Some(CoerceResult::Str(a, b.as_str())),
        (ValueRepr::I64(a), ValueRepr::I64(b)) => Some(CoerceResult::I128(*a as i128, *b as i128)),
        (ValueRepr::I128(a), ValueRepr::I128(b)) => Some(CoerceResult::I128(a.0, b.0)),
        (ValueRepr::F64(a), ValueRepr::F64(b)) => Some(CoerceResult::F64(*a, *b)),

        // are floats involved?
        (ValueRepr::F64(a), _) => Some(CoerceResult::F64(*a, some!(as_f64(b, lossy)))),
        (_, ValueRepr::F64(b)) => Some(CoerceResult::F64(some!(as_f64(a, lossy)), *b)),

        // everything else goes up to i128
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

fn range_step_backwards(
    start: Option<i64>,
    stop: Option<i64>,
    step: usize,
    end: usize,
) -> impl Iterator<Item = usize> {
    let start = match start {
        None => end.saturating_sub(1),
        Some(start) if start >= end as i64 => end.saturating_sub(1),
        Some(start) if start >= 0 => start as usize,
        Some(start) => (end as i64 + start).max(0) as usize,
    };
    let stop = match stop {
        None => 0,
        Some(stop) if stop < 0 => (end as i64 + stop).max(0) as usize,
        Some(stop) => stop as usize,
    };
    let length = if stop == 0 {
        (start + step) / step
    } else {
        (start - stop + step - 1) / step
    };
    (stop..=start).rev().step_by(step).take(length)
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

    match value.0 {
        ValueRepr::String(..) | ValueRepr::SmallStr(_) => {
            let s = value.as_str().unwrap();
            if step > 0 {
                let (start, len) = get_offset_and_len(start, stop, || s.chars().count());
                Ok(Value::from(
                    s.chars()
                        .skip(start)
                        .take(len)
                        .step_by(step as usize)
                        .collect::<String>(),
                ))
            } else {
                let chars: Vec<char> = s.chars().collect();
                Ok(Value::from(
                    range_step_backwards(start, stop, -step as usize, chars.len())
                        .map(move |i| chars[i])
                        .collect::<String>(),
                ))
            }
        }
        ValueRepr::Bytes(ref b) => {
            if step > 0 {
                let (start, len) = get_offset_and_len(start, stop, || b.len());
                Ok(Value::from_bytes(
                    b.iter()
                        .skip(start)
                        .take(len)
                        .step_by(step as usize)
                        .copied()
                        .collect(),
                ))
            } else {
                Ok(Value::from_bytes(
                    range_step_backwards(start, stop, -step as usize, b.len())
                        .map(|i| b[i])
                        .collect::<Vec<u8>>(),
                ))
            }
        }
        ValueRepr::Undefined(_) | ValueRepr::None => Ok(Value::from(Vec::<Value>::new())),
        ValueRepr::Object(obj) if matches!(obj.repr(), ObjectRepr::Seq | ObjectRepr::Iterable) => {
            if step > 0 {
                let len = obj.enumerator_len().unwrap_or_default();
                let (start, len) = get_offset_and_len(start, stop, || len);
                Ok(Value::make_object_iterable(obj, move |obj| {
                    if let Some(iter) = obj.try_iter() {
                        Box::new(iter.skip(start).take(len).step_by(step as usize))
                    } else {
                        Box::new(None.into_iter())
                    }
                }))
            } else {
                Ok(Value::make_object_iterable(obj.clone(), move |obj| {
                    if let Some(iter) = obj.try_iter() {
                        let vec: Vec<Value> = iter.collect();
                        Box::new(
                            range_step_backwards(start, stop, -step as usize, vec.len())
                                .map(move |i| vec[i].clone()),
                        )
                    } else {
                        Box::new(None.into_iter())
                    }
                }))
            }
        }
        _ => error,
    }
}

fn int_as_value(val: i128) -> Value {
    if val as i64 as i128 == val {
        (val as i64).into()
    } else {
        val.into()
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

macro_rules! math_binop {
    ($name:ident, $int:ident, $float:tt) => {
        pub fn $name(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
            match coerce(lhs, rhs, true) {
                Some(CoerceResult::I128(a, b)) => match a.$int(b) {
                    Some(val) => Ok(int_as_value(val)),
                    None => Err(failed_op(stringify!($float), lhs, rhs))
                },
                Some(CoerceResult::F64(a, b)) => Ok((a $float b).into()),
                _ => Err(impossible_op(stringify!($float), lhs, rhs))
            }
        }
    }
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
        Some(CoerceResult::I128(a, b)) => a
            .checked_add(b)
            .ok_or_else(|| failed_op("+", lhs, rhs))
            .map(int_as_value),
        Some(CoerceResult::F64(a, b)) => Ok((a + b).into()),
        Some(CoerceResult::Str(a, b)) => Ok(Value::from([a, b].concat())),
        _ => Err(impossible_op("+", lhs, rhs)),
    }
}

math_binop!(sub, checked_sub, -);
math_binop!(rem, checked_rem_euclid, %);

pub fn mul(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    if let Some((s, n)) = lhs
        .as_str()
        .map(|s| (s, rhs))
        .or_else(|| rhs.as_str().map(|s| (s, lhs)))
    {
        let n = ok!(super::as_usize(n).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                "strings can only be multiplied with integers",
            )
        }));
        if !matches!(s.len().checked_mul(n), Some(len) if len <= MAX_REPEATED_STRING_LEN) {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                "repeated string is too large",
            ));
        }
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
        Some(CoerceResult::I128(a, b)) => match a.checked_mul(b) {
            Some(val) => Ok(int_as_value(val)),
            None => Err(failed_op(stringify!(*), lhs, rhs)),
        },
        Some(CoerceResult::F64(a, b)) => Ok((a * b).into()),
        _ => Err(impossible_op(stringify!(*), lhs, rhs)),
    }
}

fn repeat_iterable(n: &Value, seq: &DynObject) -> Result<Value, Error> {
    let n = ok!(super::as_usize(n).ok_or_else(|| {
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
        Some((a / b).into())
    }
    do_it(lhs, rhs).ok_or_else(|| impossible_op("/", lhs, rhs))
}

pub fn int_div(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    match coerce(lhs, rhs, true) {
        Some(CoerceResult::I128(a, b)) => {
            if b != 0 {
                a.checked_div_euclid(b)
                    .ok_or_else(|| failed_op("//", lhs, rhs))
                    .map(int_as_value)
            } else {
                Err(failed_op("//", lhs, rhs))
            }
        }
        Some(CoerceResult::F64(a, b)) => Ok(a.div_euclid(b).into()),
        _ => Err(impossible_op("//", lhs, rhs)),
    }
}

/// Implements a binary `pow` operation on values.
pub fn pow(lhs: &Value, rhs: &Value) -> Result<Value, Error> {
    match coerce(lhs, rhs, true) {
        Some(CoerceResult::I128(a, b)) => {
            match TryFrom::try_from(b).ok().and_then(|b| a.checked_pow(b)) {
                Some(val) => Ok(int_as_value(val)),
                None => Err(failed_op("**", lhs, rhs)),
            }
        }
        Some(CoerceResult::F64(a, b)) => Ok((a.powf(b)).into()),
        _ => Err(impossible_op("**", lhs, rhs)),
    }
}

/// Implements an unary `neg` operation on value.
pub fn neg(val: &Value) -> Result<Value, Error> {
    if val.kind() == ValueKind::Number {
        match val.0 {
            ValueRepr::F64(x) => Ok((-x).into()),
            // special case for the largest i128 that can still be
            // represented.
            ValueRepr::U128(x) if x.0 == MIN_I128_AS_POS_U128 => {
                Ok(Value::from(MIN_I128_AS_POS_U128))
            }
            _ => {
                if let Ok(x) = i128::try_from(val.clone()) {
                    x.checked_mul(-1)
                        .ok_or_else(|| Error::new(ErrorKind::InvalidOperation, "overflow"))
                        .map(int_as_value)
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
    Value::from(format!(
        "{}{}",
        super::Display(&left),
        super::Display(right)
    ))
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
            s.contains(s2)
        } else {
            s.contains(&super::Display(value).to_string())
        }
    } else if let ValueRepr::Object(ref obj) = container.0 {
        match obj.repr() {
            ObjectRepr::Plain => false,
            ObjectRepr::Map => obj.get_value(value).is_some(),
            ObjectRepr::Seq | ObjectRepr::Iterable => {
                obj.try_iter().into_iter().flatten().any(|v| &v == value)
            }
        }
    } else {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "cannot perform a containment check on this value",
        ));
    };
    Ok(Value::from(rv))
}
