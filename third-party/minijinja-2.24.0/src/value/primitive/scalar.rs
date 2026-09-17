//! Exact ordinary scalar decisions with explicit conversion and error adapters.
#![forbid(unsafe_code)]
use crate::value::{Value, ValueKind, ValueRepr};
use std::cmp::Ordering;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Scalar {
    None,
    Bool(bool),
    U64(u64),
    U128(u128),
    I64(i64),
    I128(i128),
    F64(f64),
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Pair {
    I128(i128, i128),
    F64(f64, f64),
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Side {
    Left,
    Right,
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Arithmetic {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Rem,
    Pow,
}
impl Arithmetic {
    pub(crate) fn symbol(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::FloorDiv => "//",
            Self::Rem => "%",
            Self::Pow => "**",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    Unsupported,
    Arithmetic,
    Containment,
}
#[derive(Clone, Copy)]
pub(crate) enum Logical {
    And,
    Or,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Selection {
    Left,
    Right,
    False,
}

pub(crate) fn project(value: &Value) -> Option<Scalar> {
    Some(match value.0 {
        ValueRepr::None => Scalar::None,
        ValueRepr::Bool(x) => Scalar::Bool(x),
        ValueRepr::U64(x) => Scalar::U64(x),
        ValueRepr::U128(x) => Scalar::U128(x.0),
        ValueRepr::I64(x) => Scalar::I64(x),
        ValueRepr::I128(x) => Scalar::I128(x.0),
        ValueRepr::F64(x) => Scalar::F64(x),
        _ => return None,
    })
}
pub(crate) fn value(value: Scalar) -> Value {
    match value {
        Scalar::None => Value::from(()),
        Scalar::Bool(x) => x.into(),
        Scalar::U64(x) => x.into(),
        Scalar::U128(x) => x.into(),
        Scalar::I64(x) => x.into(),
        Scalar::I128(x) => x.into(),
        Scalar::F64(x) => x.into(),
    }
}
pub(crate) fn kind(value: Scalar) -> ValueKind {
    match value {
        Scalar::None => ValueKind::None,
        Scalar::Bool(_) => ValueKind::Bool,
        _ => ValueKind::Number,
    }
}
pub(crate) fn signed(value: Scalar) -> Option<i128> {
    match value {
        Scalar::None => None,
        Scalar::Bool(x) => Some(x as usize as i128),
        Scalar::U64(x) => Some(x as i128),
        Scalar::U128(x) => i128::try_from(x).ok(),
        Scalar::I64(x) => Some(x as i128),
        Scalar::I128(x) => Some(x),
        Scalar::F64(x) if x as i64 as f64 == x => Some(x as i64 as i128),
        Scalar::F64(_) => None,
    }
}
pub(crate) fn fixed_conversion(a: Scalar, b: Scalar) -> impl FnMut(Side) -> Option<i128> {
    move |side| {
        signed(match side {
            Side::Left => a,
            Side::Right => b,
        })
    }
}
pub(crate) fn as_f64(value: Scalar, lossy: bool) -> Option<f64> {
    macro_rules! checked {
        ($x:expr,$ty:ty) => {{
            let rv = $x as f64;
            return if lossy || rv as $ty == $x {
                Some(rv)
            } else {
                None
            };
        }};
    }
    Some(match value {
        Scalar::Bool(x) => x as i64 as f64,
        Scalar::U64(x) => checked!(x, u64),
        Scalar::U128(x) => checked!(x, u128),
        Scalar::I64(x) => checked!(x, i64),
        Scalar::I128(x) => checked!(x, i128),
        Scalar::F64(x) => x,
        Scalar::None => return None,
    })
}
/// The callback is invoked only by the old fallback, left before right.
pub(crate) fn coerce(
    a: Scalar,
    b: Scalar,
    lossy: bool,
    mut convert: impl FnMut(Side) -> Option<i128>,
) -> Option<Pair> {
    Some(match (a, b) {
        (Scalar::U64(a), Scalar::U64(b)) => Pair::I128(a as i128, b as i128),
        (Scalar::U128(a), Scalar::U128(b)) => Pair::I128(a as i128, b as i128),
        (Scalar::I64(a), Scalar::I64(b)) => Pair::I128(a as i128, b as i128),
        (Scalar::I128(a), Scalar::I128(b)) => Pair::I128(a, b),
        (Scalar::F64(a), Scalar::F64(b)) => Pair::F64(a, b),
        (Scalar::F64(a), b) => Pair::F64(a, as_f64(b, lossy)?),
        (a, Scalar::F64(b)) => Pair::F64(as_f64(a, lossy)?, b),
        _ => Pair::I128(convert(Side::Left)?, convert(Side::Right)?),
    })
}
fn integer(x: i128) -> Scalar {
    match super::integer(x) {
        super::Integer::I64(x) => Scalar::I64(x),
        super::Integer::I128(x) => Scalar::I128(x),
    }
}
/// Numeric arithmetic only; ordinary dynamic prebranches remain with their owners.
pub(crate) fn numeric(op: Arithmetic, pair: Pair) -> Result<Scalar, Failure> {
    match pair {
        Pair::I128(a, b) => {
            let result = match op {
                Arithmetic::Add => a.checked_add(b),
                Arithmetic::Sub => a.checked_sub(b),
                Arithmetic::Mul => a.checked_mul(b),
                Arithmetic::Rem => a.checked_rem_euclid(b),
                Arithmetic::FloorDiv => {
                    if b != 0 {
                        a.checked_div_euclid(b)
                    } else {
                        None
                    }
                }
                Arithmetic::Pow => u32::try_from(b).ok().and_then(|b| a.checked_pow(b)),
                // True division always enters the floating pair via its own as_f64 path.
                Arithmetic::Div => unreachable!("true division uses floating coercion"),
            };
            result.map(integer).ok_or(Failure::Arithmetic)
        }
        Pair::F64(a, b) => Ok(Scalar::F64(match op {
            Arithmetic::Add => a + b,
            Arithmetic::Sub => a - b,
            Arithmetic::Mul => a * b,
            Arithmetic::Div => a / b,
            Arithmetic::FloorDiv => a.div_euclid(b),
            Arithmetic::Rem => a % b,
            Arithmetic::Pow => a.powf(b),
        })),
    }
}
pub(crate) fn arithmetic(
    op: Arithmetic,
    a: Scalar,
    b: Scalar,
    convert: impl FnMut(Side) -> Option<i128>,
) -> Result<Scalar, Failure> {
    let pair = if matches!(op, Arithmetic::Div) {
        Pair::F64(
            as_f64(a, true).ok_or(Failure::Unsupported)?,
            as_f64(b, true).ok_or(Failure::Unsupported)?,
        )
    } else {
        coerce(a, b, true, convert).ok_or(Failure::Unsupported)?
    };
    numeric(op, pair)
}
pub(crate) fn equal(a: Scalar, b: Scalar, convert: impl FnMut(Side) -> Option<i128>) -> bool {
    if matches!((a, b), (Scalar::None, Scalar::None)) {
        return true;
    }
    match coerce(a, b, false, convert) {
        Some(Pair::I128(a, b)) => a == b,
        Some(Pair::F64(a, b)) => a == b,
        None => false,
    }
}
pub(crate) fn compare(a: Scalar, b: Scalar, convert: impl FnMut(Side) -> Option<i128>) -> Ordering {
    let order = kind(a).cmp(&kind(b));
    if order != Ordering::Equal {
        return order;
    }
    match (a, b) {
        (Scalar::None, Scalar::None) => Ordering::Equal,
        (Scalar::U128(a), Scalar::U128(b)) => a.cmp(&b),
        _ => match coerce(a, b, false, convert) {
            Some(Pair::I128(a, b)) => a.cmp(&b),
            Some(Pair::F64(a, b)) => cmp_f64(a, b),
            None => cmp_uncoercible_numbers(a, b),
        },
    }
}
pub(crate) fn logical(
    op: Logical,
    left: impl FnOnce() -> bool,
    right: impl FnOnce() -> bool,
) -> Selection {
    match op {
        Logical::And => {
            if left() && right() {
                Selection::Right
            } else {
                Selection::False
            }
        }
        Logical::Or => {
            if left() {
                Selection::Left
            } else {
                Selection::Right
            }
        }
    }
}
pub(crate) fn invalid_container() -> Failure {
    Failure::Containment
}
#[derive(Clone, Copy)]
enum Number {
    I128(i128),
    U128(u128),
    F64(f64),
}
fn number(value: Scalar) -> Number {
    match value {
        Scalar::U64(x) => Number::U128(x as u128),
        Scalar::U128(x) => Number::U128(x),
        Scalar::I64(x) => Number::I128(x as i128),
        Scalar::I128(x) => Number::I128(x),
        Scalar::F64(x) => Number::F64(x),
        Scalar::None | Scalar::Bool(_) => unreachable!("same-kind scalar fallback is numeric"),
    }
}
pub(crate) fn f64_total_cmp(left: f64, right: f64) -> Ordering {
    // this is taken from f64::total_cmp on newer rust versions
    let mut left = left.to_bits() as i64;
    let mut right = right.to_bits() as i64;
    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;
    left.cmp(&right)
}

pub(crate) fn cmp_f64(left: f64, right: f64) -> Ordering {
    if left == right {
        Ordering::Equal
    } else {
        f64_total_cmp(left, right)
    }
}

pub(crate) fn cmp_i128_u128(left: i128, right: u128) -> Ordering {
    if left < 0 {
        Ordering::Less
    } else {
        (left as u128).cmp(&right)
    }
}

pub(crate) fn cmp_f64_i128(left: f64, right: i128) -> Ordering {
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

pub(crate) fn cmp_f64_u128(left: f64, right: u128) -> Ordering {
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

fn cmp_uncoercible_numbers(left: Scalar, right: Scalar) -> Ordering {
    match (number(left), number(right)) {
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
/// Concrete scalar worker shells; not a compiler/native whole-thread stack bound.
pub(crate) fn control_bytes() -> usize {
    use std::mem::size_of;
    std::mem::size_of_val(&fixed_conversion(Scalar::None, Scalar::None))
        + 2 * size_of::<Scalar>()
        + size_of::<Option<Pair>>()
        + size_of::<Arithmetic>()
        + size_of::<Side>()
        + size_of::<Result<Scalar, Failure>>()
        + size_of::<Option<i128>>()
        + size_of::<Ordering>()
        + size_of::<Selection>()
        + 2 * size_of::<Number>()
        + 4 * size_of::<f64>()
        + 2 * size_of::<i128>()
}
#[cfg(test)]
mod tests;

/// Ordinary float-filter numeric conversion; undefined handling stays with the VM.
pub(crate) fn filter_float(value:Scalar)->f64 {
    match value {Scalar::None=>0.0,other=>as_f64(other,true).expect("numeric scalar")}
}
/// Both admitted and ordinary text conversion call the same standard parser.
pub(crate) fn parse_float(value:&str)->Result<f64,std::num::ParseFloatError>{
    value.parse::<f64>()
}
