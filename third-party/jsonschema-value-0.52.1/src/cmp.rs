use num_cmp::NumCmp;
use serde_json::{Map, Value};

macro_rules! num_cmp {
    ($left:expr, $right:expr) => {
        if let Some(b) = $right.as_u64() {
            NumCmp::num_eq($left, b)
        } else if let Some(b) = $right.as_i64() {
            NumCmp::num_eq($left, b)
        } else {
            #[cfg(feature = "arbitrary-precision")]
            {
                use crate::numeric::bignum;
                use fraction::BigFraction;

                let left_frac = BigFraction::from($left);

                // Check BigInt/BigFraction BEFORE f64 to avoid precision loss
                if let Some(right_bigint) = bignum::try_parse_bigint($right) {
                    let right_frac = BigFraction::from(right_bigint);
                    left_frac == right_frac
                } else if let Some(right_frac) = bignum::try_parse_bigfraction($right) {
                    left_frac == right_frac
                } else if let Some(b) = $right.as_f64() {
                    // Fallback to f64 for scientific notation or other cases
                    left_frac == BigFraction::from(b)
                } else {
                    // Can't parse right - not equal
                    false
                }
            }
            #[cfg(not(feature = "arbitrary-precision"))]
            {
                if let Some(b) = $right.as_f64() {
                    NumCmp::num_eq($left, b)
                } else {
                    unreachable!("Numbers always fit in u64/i64/f64 without arbitrary-precision")
                }
            }
        }
    };
}

/// Compare two JSON numbers for equality with arbitrary precision support
#[inline]
#[doc(hidden)]
#[must_use]
pub fn equal_numbers<L: crate::JsonNumber>(left: &L, right: &serde_json::Number) -> bool {
    #[cfg(feature = "arbitrary-precision")]
    {
        use crate::numeric::bignum;
        use fraction::BigFraction;

        // Check BigInt/BigFraction first to avoid precision loss from f64 conversion
        if let Some(left_bigint) = bignum::try_parse_bigint(&left.to_number()) {
            if let Some(right_bigint) = bignum::try_parse_bigint(right) {
                left_bigint == right_bigint
            } else if let Some(b) = right.as_u64() {
                left_bigint == num_bigint::BigInt::from(b)
            } else if let Some(b) = right.as_i64() {
                left_bigint == num_bigint::BigInt::from(b)
            } else if let Some(right_frac) = bignum::try_parse_bigfraction(right) {
                BigFraction::from(left_bigint) == right_frac
            } else if let Some(b) = right.as_f64() {
                BigFraction::from(left_bigint) == BigFraction::from(b)
            } else {
                unreachable!("Right is not parseable as any numeric type - should not happen for valid JSON numbers")
            }
        } else if let Some(left_frac) = bignum::try_parse_bigfraction(&left.to_number()) {
            if let Some(right_frac) = bignum::try_parse_bigfraction(right) {
                left_frac == right_frac
            } else if let Some(right_bigint) = bignum::try_parse_bigint(right) {
                left_frac == BigFraction::from(right_bigint)
            } else if let Some(b) = right.as_u64() {
                left_frac == BigFraction::from(b)
            } else if let Some(b) = right.as_i64() {
                left_frac == BigFraction::from(b)
            } else if let Some(b) = right.as_f64() {
                left_frac == BigFraction::from(b)
            } else {
                unreachable!("Right is not parseable as any numeric type - should not happen for valid JSON numbers")
            }
        } else if let Some(a) = left.as_u64() {
            num_cmp!(a, right)
        } else if let Some(a) = left.as_i64() {
            num_cmp!(a, right)
        } else if let Some(a) = left.as_f64() {
            num_cmp!(a, right)
        } else {
            // Left is a number in scientific notation that doesn't fit in f64
            // (e.g., 1e309, 1e400). With arbitrary-precision, these are stored as
            // strings but can't be converted to any numeric type we support.
            // Return false as we can't reliably compare them.
            false
        }
    }
    #[cfg(not(feature = "arbitrary-precision"))]
    { equal_node_numbers(left, right) }
}

/// The same primitive number equality for two borrowed input numbers. The
/// allocating arbitrary-precision branch remains separately qualified.
pub fn equal_node_numbers<L: crate::JsonNumber, R: crate::JsonNumber>(left: &L, right: &R) -> bool {
    #[cfg(feature = "arbitrary-precision")]
    { equal_numbers(left, &right.to_number()) }
    #[cfg(not(feature = "arbitrary-precision"))]
    {
        if let Some(a) = left.as_u64() {
            num_cmp!(a, right)
        } else if let Some(a) = left.as_i64() {
            num_cmp!(a, right)
        } else if let Some(a) = left.as_f64() {
            num_cmp!(a, right)
        } else {
            unreachable!("Numbers always fit in u64/i64/f64 without arbitrary-precision")
        }
    }
}

/// Tests for two JSON values to be equal using the JSON Schema semantic.
#[must_use]
#[allow(clippy::missing_panics_doc)]
pub fn equal(left: &Value, right: &Value) -> bool {
    equal_nodes_with::<crate::SerdeJson, _>(&left, &right, &mut |_| true)
}

#[inline]
#[must_use]
pub fn equal_arrays(left: &[Value], right: &[Value]) -> bool {
    left.len() == right.len() && {
        let mut idx = 0_usize;
        while idx < left.len() {
            if !equal(&left[idx], &right[idx]) {
                return false;
            }
            idx += 1;
        }
        true
    }
}

#[inline]
#[must_use]
pub fn equal_objects(left: &Map<String, Value>, right: &Map<String, Value>) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|((ka, va), (kb, vb))| ka == kb && equal(va, vb))
}

#[cfg(test)]
mod tests {
    use super::equal;
    use serde_json::{json, Value};
    use test_case::test_case;

    #[test_case(&json!(1), &json!(1.0))]
    #[test_case(&json!([2]), &json!([2.0]))]
    #[test_case(&json!([-3]), &json!([-3.0]))]
    #[test_case(&json!({"a": 1}), &json!({"a": 1.0}))]
    fn are_equal(left: &Value, right: &Value) {
        assert!(equal(left, right));
    }

    #[test_case(&json!(1), &json!(2.0))]
    #[test_case(&json!([]), &json!(["foo"]))]
    #[test_case(&json!([-3]), &json!([-4.0]))]
    #[test_case(&json!({"a": 1}), &json!({"a": 1.0, "b": 2}))]
    fn are_not_equal(left: &Value, right: &Value) {
        assert!(!equal(left, right));
    }

    #[cfg(feature = "arbitrary-precision")]
    mod arbitrary_precision {
        use super::equal;
        use serde_json::Value;
        use test_case::test_case;

        fn parse_json(s: &str) -> Value {
            serde_json::from_str(s).unwrap()
        }
        #[test_case("0.1", "0.1", true; "exact decimal match")]
        #[test_case("0.1", "0.10", true; "decimal with trailing zero")]
        #[test_case("0.1", "0.100000", true; "decimal with many trailing zeros")]
        #[test_case("0.1", "0.2", false; "different decimals")]
        #[test_case("0.3", "0.30", true; "another trailing zero case")]
        #[test_case("1.0", "1", true; "decimal vs integer")]
        #[test_case("1.00", "1.0", true; "decimals with different trailing zeros")]
        #[test_case(
            "99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999.5",
            "99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999.5",
            true;
            "huge decimal self equality"
        )]
        #[test_case(
            "99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999.5",
            "99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999.6",
            false;
            "huge decimals different"
        )]
        #[test_case(
            "99999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999.5",
            "100",
            false;
            "huge decimal vs small integer"
        )]
        #[test_case("18446744073709551616", "18446744073709551616", true; "large integer self equality")]
        #[test_case("18446744073709551616", "18446744073709551617", false; "large integers different")]
        #[test_case("99999999999999999999999999999999999999", "99999999999999999999999999999999999999", true; "very large integer equality")]
        #[test_case("99999999999999999999999999999999999999", "100", false; "very large vs small integer")]
        #[test_case("100", "100.0", true; "small integer vs decimal")]
        #[test_case("0.1", "1", false; "small decimal vs integer")]
        #[test_case("1.0", "1", true; "decimal one vs integer one")]
        #[test_case("18446744073709551616", "100.5", false; "large int vs decimal")]
        #[test_case("18446744073709551616.0", "18446744073709551616", true; "large int as decimal vs large int")]
        #[test_case("-0.1", "-0.1", true; "negative decimal equality")]
        #[test_case("-0.1", "-0.10", true; "negative decimal trailing zero")]
        #[test_case("-18446744073709551616", "-18446744073709551616", true; "negative large int")]
        #[test_case("-18446744073709551616", "18446744073709551616", false; "negative vs positive large")]
        #[test_case("-100.5", "-100.5", true; "negative decimal match")]
        #[test_case("-100.5", "100.5", false; "negative vs positive decimal")]
        #[test_case("0", "0.0", true; "zero integer vs decimal")]
        #[test_case("0.0", "0.00", true; "zero decimals with different precision")]
        #[test_case("-0.0", "0.0", true; "negative zero vs positive zero")]
        #[test_case("1e10", "10000000000", true; "scientific notation vs integer")]
        #[test_case("1e19", "10000000000000000000", true; "scientific integer beyond i64")]
        #[test_case("1e19", "10000000000000000001", false; "scientific integer mismatch")]
        #[test_case("1.5e2", "150", true; "decimal scientific vs integer")]
        #[test_case("1.5e2", "150.0", true; "decimal scientific vs decimal")]
        #[test_case(r"[0.1, 0.2, 0.3]", r"[0.1, 0.2, 0.3]", true; "array exact match")]
        #[test_case(r"[0.1, 0.2]", r"[0.10, 0.20]", true; "array with trailing zeros")]
        #[test_case(r"[18446744073709551616]", r"[18446744073709551616]", true; "array with large integer")]
        #[test_case(r"[0.1, 0.2]", r"[0.1, 0.3]", false; "array different values")]
        #[test_case(r#"{"value": 0.1}"#, r#"{"value": 0.1}"#, true; "object exact match")]
        #[test_case(r#"{"value": 0.1}"#, r#"{"value": 0.10}"#, true; "object with trailing zero")]
        #[test_case(r#"{"id": 18446744073709551616}"#, r#"{"id": 18446744073709551616}"#, true; "object with large integer")]
        #[test_case(r#"{"value": 0.1}"#, r#"{"value": 0.2}"#, false; "object different values")]
        #[test_case("18446744073709551616", "-1", false; "large positive bigint vs negative i64")]
        #[test_case("18446744073709551616", "-100", false; "large positive bigint vs negative i64 small")]
        #[test_case("-18446744073709551616", "-1", false; "large negative bigint vs small negative i64")]
        #[test_case("18446744073709551616", "1e10", false; "large bigint vs scientific notation f64")]
        #[test_case("10000000000", "1e10", true; "bigint vs scientific notation equal")]
        #[test_case("-18446744073709551616", "-1.5e3", false; "negative bigint vs scientific notation")]
        #[test_case("0.5", "5e-1", true; "bigfraction vs scientific notation equal")]
        #[test_case("0.3", "3e-1", true; "bigfraction vs scientific equal exact")]
        #[test_case("123.456", "1.23456e2", true; "bigfraction vs scientific notation")]
        #[test_case("0.1", "1e-2", false; "bigfraction vs scientific not equal")]
        #[test_case("1e309", "1e309", true; "huge scientific notation now handled")]
        #[test_case("1e400", "1e400", true; "extreme scientific notation now handled")]
        #[test_case("1e-400", "1e-400", true; "extreme small scientific notation self equality")]
        #[test_case("1e309", "1", false; "huge scientific notation vs integer")]
        fn arbitrary_precision_equality(left_str: &str, right_str: &str, should_equal: bool) {
            let left = parse_json(left_str);
            let right = parse_json(right_str);
            assert_eq!(equal(&left, &right), should_equal);
        }
    }
}

/// Fixed controls of this crate's selected primitive number equality worker.
/// Returns None when its compiled arbitrary-precision branch needs allocation.
/// No numbers are read and no comparison occurs. Borrowed accessors and the
/// caller's input/source backings remain separately qualified obligations.
#[must_use]
pub fn original_number_equality_control_bytes<N>() -> Option<usize> {
    use std::mem::{size_of, size_of_val};
    let parts = [
        crate::numeric::original_comparison_control_bytes::<N, u64>()?,
        crate::numeric::original_comparison_control_bytes::<N, i64>()?,
        crate::numeric::original_comparison_control_bytes::<N, f64>()?,
        size_of::<(&N, &serde_json::Number)>(), size_of::<bool>(),
    ];
    parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
}

/// A reached population in the shared borrowed JSON equality worker.
#[derive(Clone, Copy, Debug)]
pub enum EqualityCharge {
    /// Concrete recursive frame before any access or descent; None is overflow.
    Frame(Option<usize>),
    /// Actual selected number helper; None requires its allocating producer.
    Number(Option<usize>),
}

/// Compare a borrowed JSON tree against a schema value with a fallible per-step
/// control loan. Both containers must expose their JSON key order and unique
/// keys; the worker preserves the existing serde comparison's zip order.
/// No input conversion, prepared-key construction or owned temporary is used.
/// A false loan result stops before that frame/helper runs. The caller keeps
/// the exact first loan error; false alone conveys no diagnostic or permission.
pub fn equal_node_with<F, P>(left: &F::Node<'_>, right: &Value, permit: &mut P) -> bool
where F: crate::Json, P: FnMut(EqualityCharge) -> bool,
{
    let bytes = std::mem::size_of::<(&F::Node<'_>, &Value, &mut P)>()
        .checked_add(std::mem::size_of::<(Option<usize>, EqualityCharge, bool)>());
    if !permit(EqualityCharge::Frame(bytes)) { return false; }
    equal_expected_with::<F, Value, P>(left, right, permit)
}

/// Compare against a compiled immutable literal through the same borrowed
/// worker and per-step loans as equal_node_with.
pub fn equal_literal_with<F, P>(left: &F::Node<'_>, right: &crate::literal::Literal, permit: &mut P) -> bool
where F: crate::Json, P: FnMut(EqualityCharge) -> bool,
{
    let bytes = std::mem::size_of::<(&F::Node<'_>, &crate::literal::Literal, &mut P)>()
        .checked_add(std::mem::size_of::<(Option<usize>, EqualityCharge, bool)>());
    if !permit(EqualityCharge::Frame(bytes)) { return false; }
    equal_expected_with::<F, crate::literal::Literal, P>(left, right, permit)
}

fn equal_expected_with<F, E, P>(left: &F::Node<'_>, right: &E, permit: &mut P) -> bool
where F: crate::Json, E: crate::literal::Expected, P: FnMut(EqualityCharge) -> bool,
{
    use crate::{Array, Node, Object, literal::View};
    use std::mem::{size_of, size_of_val};
    let parts = [
        size_of::<(&F::Node<'_>, &E, &mut P)>(), size_of::<P>(),
        size_of::<<F::Node<'_> as Node<'_, F>>::Object>(),
        size_of::<<F::Node<'_> as Node<'_, F>>::Array>(),
        size_of::<<F::Node<'_> as Node<'_, F>>::Number>(),
        size_of::<<<F::Node<'_> as Node<'_, F>>::Object as Object<'_, F>>::MembersIter>(),
        size_of::<<<F::Node<'_> as Node<'_, F>>::Object as Object<'_, F>>::MemberName>(),
        size_of::<<<F::Node<'_> as Node<'_, F>>::Array as Array<'_, F>>::ElementsIter>(),
        size_of::<E::Members<'_>>(), size_of::<std::slice::Iter<'_, E>>(),
        size_of::<Option<<F::Node<'_> as Node<'_, F>>::Object>>(),
        size_of::<Option<<F::Node<'_> as Node<'_, F>>::Array>>(),
        size_of::<Option<<F::Node<'_> as Node<'_, F>>::Number>>(),
        size_of::<std::iter::Zip<<<F::Node<'_> as Node<'_, F>>::Array as Array<'_, F>>::ElementsIter, std::slice::Iter<'_, E>>>(),
        size_of::<std::iter::Zip<<<F::Node<'_> as Node<'_, F>>::Object as Object<'_, F>>::MembersIter, E::Members<'_>>>(),
        size_of::<(F::Node<'_>, &E)>(),
        size_of::<(<<F::Node<'_> as Node<'_, F>>::Object as Object<'_, F>>::MemberName, F::Node<'_>, &str, &E)>(),

        size_of::<Option<F::Node<'_>>>(), size_of::<Option<std::borrow::Cow<'_, str>>>(),
        size_of::<Option<bool>>(), size_of::<EqualityCharge>(),
        size_of::<View<'_, E>>(), size_of::<(&str, &str)>(), size_of::<(usize, usize)>(), size_of::<bool>(),
    ];
    let frame = parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add);
    if !permit(EqualityCharge::Frame(frame)) { return false; }
    match right.view() {
        View::Null => left.is_null(),
        View::Bool(expected) => left.as_boolean() == Some(expected),
        View::String(expected) => left.as_string().is_some_and(|value| value.as_ref() == expected),
        View::Number(expected) => {
            let Some(number) = left.as_number() else { return false; };
            if !permit(EqualityCharge::Number(original_number_equality_control_bytes::<<F::Node<'_> as Node<'_, F>>::Number>())) { return false; }
            equal_numbers(&number, expected)
        }
        View::Array(expected) => {
            let Some(array) = left.as_array() else { return false; };
            array.len() == expected.len() && array.elements().zip(expected).all(|(node, value)| equal_expected_with::<F, E, P>(&node, value, permit))
        }
        View::Object(expected) => {
            let Some(object) = left.as_object() else { return false; };
            object.len() == expected.len() && object.members().zip(expected).all(|((name, node), (key, value))| {
                name.as_ref() == key && equal_expected_with::<F, E, P>(&node, value, permit)
            })
        }
    }
}


/// Compare two borrowed input nodes through the ordinary recursive equality
/// worker. Original callers pay its frames and selected number helper; ordinary
/// Value equality uses this same function with an unrestricted call loan.
pub fn equal_nodes_with<F, P>(left: &F::Node<'_>, right: &F::Node<'_>, permit: &mut P) -> bool
where F: crate::Json, P: FnMut(EqualityCharge) -> bool,
{
    use crate::{Node, Array, Object, JsonType};
    use std::mem::{size_of, size_of_val};
    let parts = [size_of::<(&F::Node<'_>, &F::Node<'_>, &mut P)>(), size_of::<P>(),
        size_of::<JsonType>(), size_of::<Option<bool>>(), size_of::<Option<std::borrow::Cow<'_,str>>>(),
        size_of::<<F::Node<'_> as Node<'_,F>>::Number>(),
        size_of::<<F::Node<'_> as Node<'_,F>>::Array>(),
        size_of::<<F::Node<'_> as Node<'_,F>>::Object>(),
        size_of::<<<F::Node<'_> as Node<'_,F>>::Array as Array<'_,F>>::ElementsIter>(),
        size_of::<<<F::Node<'_> as Node<'_,F>>::Object as Object<'_,F>>::MembersIter>(),
        size_of::<<<F::Node<'_> as Node<'_,F>>::Object as Object<'_,F>>::MemberName>(),
        size_of::<(F::Node<'_>, F::Node<'_>)>(), size_of::<(usize,usize,bool)>(),
        size_of::<EqualityCharge>(), size_of::<Option<usize>>()];
    let frame = parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
        .and_then(|n| n.checked_mul(2));
    if !permit(EqualityCharge::Frame(frame)) { return false; }
    match right.json_type() {
        JsonType::Null => left.is_null(),
        JsonType::Boolean => left.as_boolean() == right.as_boolean(),
        JsonType::String => match (left.as_string(),right.as_string()) {
            (Some(a),Some(b)) => a.as_ref()==b.as_ref(), _ => false,
        },
        JsonType::Number | JsonType::Integer => {
            let (Some(a),Some(b)) = (left.as_number(),right.as_number()) else { return false; };
            if !permit(EqualityCharge::Number(original_number_equality_control_bytes::<<F::Node<'_> as Node<'_,F>>::Number>())) { return false; }
            equal_node_numbers(&a,&b)
        }
        JsonType::Array => {
            let (Some(a),Some(b)) = (left.as_array(),right.as_array()) else { return false; };
            a.len()==b.len() && a.elements().zip(b.elements()).all(|(a,b)| equal_nodes_with::<F,P>(&a,&b,permit))
        }
        JsonType::Object => {
            let (Some(a),Some(b)) = (left.as_object(),right.as_object()) else { return false; };
            a.len()==b.len() && a.members().zip(b.members()).all(|((ka,a),(kb,b))| {
                ka.as_ref()==kb.as_ref() && equal_nodes_with::<F,P>(&a,&b,permit)
            })
        }
    }
}
