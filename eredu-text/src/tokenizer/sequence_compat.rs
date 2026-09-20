//! Python-compatible signed ranges and Unicode slices for chat templates.
//!
//! The source adapter uses MiniJinja's public, version-pinned parser API. It
//! rewrites slice expressions only; upstream still compiles and executes the
//! complete template. Both ordinary and prepared chat install these helpers.
use minijinja::value::ValueKind;
use minijinja::{Environment, Error, ErrorKind, Value};

mod source;
pub(super) use source::normalize_slices;

pub(super) fn install(env: &mut Environment<'_>) {
    env.add_function("range", signed_range);
    env.add_filter("__eredu_slice", slice);
}

fn invalid(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidOperation, message)
}

fn signed_range(first: i64, stop: Option<i64>, step: Option<i64>) -> Result<Value, Error> {
    let (start, stop) = stop.map_or((0, first), |stop| (first, stop));
    let step = i128::from(step.unwrap_or(1));
    if step == 0 {
        return Err(invalid("cannot create range with step of 0"));
    }
    let distance = if step > 0 {
        i128::from(stop) - i128::from(start)
    } else {
        i128::from(start) - i128::from(stop)
    };
    let count = if distance <= 0 {
        0
    } else {
        1 + (distance - 1) / step.abs()
    };
    if count > 100_000 {
        return Err(invalid("range has too many elements"));
    }
    // Every yielded value lies between two i64 endpoints. Intermediates use
    // i128 so neither endpoint subtraction nor negating MIN can overflow.
    Ok(Value::make_iterable(move || {
        (0..count as usize).map(move |index| (i128::from(start) + index as i128 * step) as i64)
    }))
}

/// Coordinates follow Python's slice.indices convention, including the
/// distinction between omitted stop and explicit -1 for negative steps.
fn coordinates(
    len: usize,
    start: Option<i64>,
    stop: Option<i64>,
    step: i64,
) -> impl Iterator<Item = usize> {
    let len = len as i128;
    let reverse = step < 0;
    let (low, high) = if reverse { (-1, len - 1) } else { (0, len) };
    let bound = |value: i64| {
        let value = i128::from(value);
        (if value < 0 { len + value } else { value }).clamp(low, high)
    };
    let mut next = start.map_or(if reverse { len - 1 } else { 0 }, bound);
    let end = stop.map_or(if reverse { -1 } else { len }, bound);
    std::iter::from_fn(move || {
        if if reverse { next <= end } else { next >= end } {
            return None;
        }
        let current = next as usize;
        next += i128::from(step);
        Some(current)
    })
}

fn slice(
    value: Value,
    start: Option<i64>,
    stop: Option<i64>,
    step: Option<i64>,
) -> Result<Value, Error> {
    let step = step.unwrap_or(1);
    if step == 0 {
        return Err(invalid("slice step cannot be zero"));
    }
    if let Some(text) = value.as_str() {
        let chars: Vec<_> = text.chars().collect();
        let result: String = coordinates(chars.len(), start, stop, step)
            .map(|i| chars[i])
            .collect();
        return Ok(Value::from(result));
    }
    match value.kind() {
        ValueKind::Seq | ValueKind::Iterable => {
            let items: Vec<_> = value.try_iter()?.collect();
            Ok(Value::from(
                coordinates(items.len(), start, stop, step)
                    .map(|i| items[i].clone())
                    .collect::<Vec<_>>(),
            ))
        }
        ValueKind::Undefined | ValueKind::None => Ok(Value::from(Vec::<Value>::new())),
        _ => Err(invalid("value cannot be sliced")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_bounds_have_independent_python_coordinates() {
        let cases = [
            (None, None, -1, vec![3, 2, 1, 0]),
            (Some(0), Some(0), -1, vec![]),
            (Some(-9), None, -1, vec![]),
            (None, Some(-1), -1, vec![]),
            (None, Some(-9), -2, vec![3, 1]),
            (None, None, i64::MIN, vec![3]),
            (Some(i64::MIN), Some(i64::MAX), i64::MAX, vec![0]),
        ];
        for (start, stop, step, expected) in cases {
            assert_eq!(
                coordinates(4, start, stop, step).collect::<Vec<_>>(),
                expected
            );
            assert_eq!(coordinates(0, start, stop, step).next(), None);
        }
    }

    #[test]
    fn full_signed_range_uses_wide_intermediates_and_enforces_count() {
        let range = signed_range(i64::MIN, Some(i64::MAX), Some(i64::MAX)).unwrap();
        let values: Vec<_> = range
            .try_iter()
            .unwrap()
            .map(|v| i64::try_from(v).unwrap())
            .collect();
        assert_eq!(values, [i64::MIN, -1, i64::MAX - 1]);
        assert!(signed_range(i64::MIN, Some(i64::MAX), Some(1)).is_err());
        assert!(signed_range(1, Some(3), Some(0)).is_err());
        assert_eq!(
            signed_range(100_000, None, None).unwrap().len(),
            Some(100_000)
        );
    }
}
