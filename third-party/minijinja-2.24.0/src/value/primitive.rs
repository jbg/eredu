//! Fixed primitive truth and checked negation, without Value/error construction.
#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug)]
pub(crate) enum Truth {
    Bool(bool),
    U64(u64),
    U128(u128),
    I64(i64),
    I128(i128),
    F64(f64),
    Length(usize),
    False,
}
impl Truth {
    pub(crate) fn get(self) -> bool {
        match self {
            Self::Bool(value) => value,
            Self::U64(value) => value != 0,
            Self::U128(value) => value != 0,
            Self::I64(value) => value != 0,
            Self::I128(value) => value != 0,
            Self::F64(value) => value != 0.0,
            Self::Length(value) => value != 0,
            Self::False => false,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub(crate) enum Integer {
    I64(i64),
    I128(i128),
}
pub(crate) fn integer(value: i128) -> Integer {
    if value as i64 as i128 == value {
        Integer::I64(value as i64)
    } else {
        Integer::I128(value)
    }
}
pub(crate) fn negative_integer(value: i128) -> Option<Integer> {
    value.checked_mul(-1).map(integer)
}
pub(crate) fn negative_float(value: f64) -> f64 {
    -value
}
/// Preserve the existing ordinary special case, including its U128 representation.
pub(crate) fn negative_special_u128(value: u128) -> Option<u128> {
    const MIN_I128_AS_POS_U128: u128 = 170141183460469231731687303715884105728;
    (value == MIN_I128_AS_POS_U128).then_some(MIN_I128_AS_POS_U128)
}

pub(crate) mod scalar;
pub(crate) mod slice;

pub(crate) mod text;

/// Ordinary signed indexing, shared by dynamic and checked borrowed sequences.
pub(crate) fn sequence_index(
    value: Option<i64>,
    len: impl FnOnce() -> Option<usize>,
) -> Option<usize> {
    match value.and_then(|v| isize::try_from(v).ok()) {
        Some(i) if i < 0 => len()?.checked_sub(i.unsigned_abs()),
        Some(i) => Some(i as usize),
        None => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Presence {
    Defined,
    Undefined,
    None,
}
impl Presence {
    pub(crate) fn check(self, undefined: bool, none: bool) -> bool {
        match self {
            Self::Defined => !undefined,
            Self::Undefined => undefined,
            Self::None => none,
        }
    }
}

pub(crate) mod type_tests;

pub(crate) mod length;

pub(crate) mod default;

pub(crate) mod join;

pub(crate) mod replace;
pub(crate) mod case;

pub(crate) mod membership;

pub(crate) mod range;
