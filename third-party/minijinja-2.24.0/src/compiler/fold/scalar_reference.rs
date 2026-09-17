//! Independent old scalar comparisons; dynamic recursive Value calls remain explicit dependencies.
#![forbid(unsafe_code)]
use crate::value::{ObjectRepr, Value, ValueRepr};
use std::cmp::Ordering;
pub(crate) mod ops;
fn f64_total_cmp(left: f64, right: f64) -> Ordering {
    // this is taken from f64::total_cmp on newer rust versions
    let mut left = left.to_bits() as i64;
    let mut right = right.to_bits() as i64;
    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;
    left.cmp(&right)
}

fn cmp_f64(left: f64, right: f64) -> Ordering {
    if left == right {
        Ordering::Equal
    } else {
        f64_total_cmp(left, right)
    }
}

#[derive(Copy, Clone)]
enum Number {
    I128(i128),
    U128(u128),
    F64(f64),
}

fn number(value: &Value) -> Option<Number> {
    Some(match &value.0 {
        ValueRepr::U64(x) => Number::U128(*x as u128),
        ValueRepr::U128(x) => Number::U128(x.0),
        ValueRepr::I64(x) => Number::I128(*x as i128),
        ValueRepr::I128(x) => Number::I128(x.0),
        ValueRepr::F64(x) => Number::F64(*x),
        _ => return None,
    })
}

fn cmp_i128_u128(left: i128, right: u128) -> Ordering {
    if left < 0 {
        Ordering::Less
    } else {
        (left as u128).cmp(&right)
    }
}

fn cmp_f64_i128(left: f64, right: i128) -> Ordering {
    match cmp_f64(left, right as f64) {
        Ordering::Equal if left.is_finite() => {
            if left >= i128::MAX as f64 {
                Ordering::Greater
            } else {
                let trunc = left.trunc();
                (trunc as i128)
                    .cmp(&right)
                    .then_with(|| left.partial_cmp(&trunc).unwrap())
            }
        }
        rv => rv,
    }
}

fn cmp_f64_u128(left: f64, right: u128) -> Ordering {
    match cmp_f64(left, right as f64) {
        Ordering::Equal if left.is_finite() => {
            if left < 0.0 {
                Ordering::Less
            } else if left >= u128::MAX as f64 {
                Ordering::Greater
            } else {
                (left as u128).cmp(&right)
            }
        }
        rv => rv,
    }
}

// This is only needed when `coerce` cannot find a common lossless numeric
// representation.  Keep it out of line so the rare fallback does not bloat the
// hot comparison path.
#[cold]
#[inline(never)]
fn cmp_uncoercible_numbers(left: &Value, right: &Value) -> Ordering {
    match (number(left).unwrap(), number(right).unwrap()) {
        (Number::F64(a), Number::F64(b)) => cmp_f64(a, b),
        (Number::F64(a), Number::I128(b)) => cmp_f64_i128(a, b),
        (Number::I128(a), Number::F64(b)) => cmp_f64_i128(b, a).reverse(),
        (Number::F64(a), Number::U128(b)) => cmp_f64_u128(a, b),
        (Number::U128(a), Number::F64(b)) => cmp_f64_u128(b, a).reverse(),
        (Number::I128(a), Number::I128(b)) => a.cmp(&b),
        (Number::U128(a), Number::U128(b)) => a.cmp(&b),
        (Number::I128(a), Number::U128(b)) => cmp_i128_u128(a, b),
        (Number::U128(a), Number::I128(b)) => cmp_i128_u128(b, a).reverse(),
    }
}

pub(crate) fn equal(left: &Value, right: &Value) -> bool {
    match (&left.0, &right.0) {
        (&ValueRepr::None, &ValueRepr::None) => true,
        (&ValueRepr::Undefined(_), &ValueRepr::Undefined(_)) => true,
        (&ValueRepr::String(ref a, _), &ValueRepr::String(ref b, _)) => a == b,
        (&ValueRepr::SmallStr(ref a), &ValueRepr::SmallStr(ref b)) => a.as_str() == b.as_str(),
        (&ValueRepr::Bytes(ref a), &ValueRepr::Bytes(ref b)) => a == b,
        _ => match ops::coerce(left, right, false) {
            Some(ops::CoerceResult::F64(a, b)) => a == b,
            Some(ops::CoerceResult::I128(a, b)) => a == b,
            Some(ops::CoerceResult::Str(a, b)) => a == b,
            None => {
                if let (Some(a), Some(b)) = (left.as_object(), right.as_object()) {
                    if a.is_same_object(b) {
                        return true;
                    } else if a.is_same_object_type(b) {
                        if let Some(rv) = a.custom_cmp(b) {
                            return rv == Ordering::Equal;
                        }
                    }
                    match (a.repr(), b.repr()) {
                        (ObjectRepr::Map, ObjectRepr::Map) => {
                            // only if we have known lengths can we compare the enumerators
                            // ahead of time.  This function has a fallback for when a
                            // map has an unknown length.  That's generally a bad idea, but
                            // it makes sense supporting regardless as silent failures are
                            // not a lot of fun.
                            let mut need_length_fallback = true;
                            if let (Some(a_len), Some(b_len)) =
                                (a.enumerator_len(), b.enumerator_len())
                            {
                                if a_len != b_len {
                                    return false;
                                }
                                need_length_fallback = false;
                            }
                            let mut a_count = 0;
                            if !a.try_iter_pairs().is_some_and(|mut ak| {
                                ak.all(|(k, v1)| {
                                    a_count += 1;
                                    b.get_value(&k) == Some(v1)
                                })
                            }) {
                                return false;
                            }
                            if !need_length_fallback {
                                true
                            } else {
                                a_count == b.try_iter().map_or(0, |x| x.count())
                            }
                        }
                        (
                            ObjectRepr::Seq | ObjectRepr::Iterable,
                            ObjectRepr::Seq | ObjectRepr::Iterable,
                        ) => {
                            if let (Some(ak), Some(bk)) = (a.try_iter(), b.try_iter()) {
                                ak.eq(bk)
                            } else {
                                false
                            }
                        }
                        // terrible fallback for plain objects
                        (ObjectRepr::Plain, ObjectRepr::Plain) => a.to_string() == b.to_string(),
                        // should not happen
                        (_, _) => false,
                    }
                } else {
                    false
                }
            }
        },
    }
}
pub(crate) fn compare(left: &Value, right: &Value) -> Ordering {
    let kind_ordering = left.kind().cmp(&right.kind());
    if matches!(kind_ordering, Ordering::Less | Ordering::Greater) {
        return kind_ordering;
    }
    match (&left.0, &right.0) {
        (&ValueRepr::None, &ValueRepr::None) => Ordering::Equal,
        (&ValueRepr::Undefined(_), &ValueRepr::Undefined(_)) => Ordering::Equal,
        (&ValueRepr::String(ref a, _), &ValueRepr::String(ref b, _)) => a.cmp(b),
        (&ValueRepr::SmallStr(ref a), &ValueRepr::SmallStr(ref b)) => a.as_str().cmp(b.as_str()),
        (&ValueRepr::Bytes(ref a), &ValueRepr::Bytes(ref b)) => a.cmp(b),
        // `coerce` represents two u128 values as i128, which reverses the
        // order if only one of them exceeds i128::MAX.
        (&ValueRepr::U128(a), &ValueRepr::U128(b)) => { a.0 }.cmp(&{ b.0 }),
        _ => match ops::coerce(left, right, false) {
            Some(ops::CoerceResult::F64(a, b)) => cmp_f64(a, b),
            Some(ops::CoerceResult::I128(a, b)) => a.cmp(&b),
            Some(ops::CoerceResult::Str(a, b)) => a.cmp(b),
            None => {
                if left.is_number() && right.is_number() {
                    return cmp_uncoercible_numbers(left, right);
                }

                let a = left.as_object().unwrap();
                let b = right.as_object().unwrap();

                if a.is_same_object(b) {
                    Ordering::Equal
                } else {
                    // if there is a custom comparison, run it.
                    if a.is_same_object_type(b) {
                        if let Some(rv) = a.custom_cmp(b) {
                            return rv;
                        }
                    }
                    match (a.repr(), b.repr()) {
                        (ObjectRepr::Map, ObjectRepr::Map) => {
                            // This is not really correct.  Because the keys can be in arbitrary
                            // order this could just sort really weirdly as a result.  However
                            // we don't want to pay the cost of actually sorting the keys for
                            // ordering so we just accept this for now.
                            match (a.try_iter_pairs(), b.try_iter_pairs()) {
                                (Some(a), Some(b)) => a.cmp(b),
                                _ => unreachable!(),
                            }
                        }
                        (
                            ObjectRepr::Seq | ObjectRepr::Iterable,
                            ObjectRepr::Seq | ObjectRepr::Iterable,
                        ) => match (a.try_iter(), b.try_iter()) {
                            (Some(a), Some(b)) => a.cmp(b),
                            _ => unreachable!(),
                        },
                        // terrible fallback for plain objects
                        (ObjectRepr::Plain, ObjectRepr::Plain) => a.to_string().cmp(&b.to_string()),
                        // should not happen
                        (_, _) => unreachable!(),
                    }
                }
            }
        },
    }
}
