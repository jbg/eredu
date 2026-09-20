//! Stock serde visitor delivering ordered receipt fields to the fixed reader.
//! Borrowed RawValue nodes preserve duplicate keys. Typed scalar dispatch also
//! preserves numeric events when a downstream enables serde_json/arbitrary_precision.
//! Syntax is validated before callbacks; no intermediate JSON tree is built.
use serde::{
    Deserialize,
    de::{DeserializeSeed, Deserializer, Error as _, MapAccess, SeqAccess, Visitor},
};
use serde_json::value::RawValue;

#[derive(Debug)]
pub(super) enum Event<'a> {
    Object,
    EndObject,
    Array,
    EndArray,
    Key(&'a str),
    String(&'a str),
    I64(i64),
    U64(u64),
    F64(f64),
    Bool(bool),
    Null,
}

pub(super) fn parse(
    bytes: &[u8],
    mut emit: impl for<'a> FnMut(Event<'a>),
) -> Result<(), serde_json::Error> {
    let raw: &RawValue = serde_json::from_slice(bytes)?;
    dispatch(raw, &mut emit, 0)
}

fn dispatch<F: for<'a> FnMut(Event<'a>)>(
    raw: &RawValue,
    emit: &mut F,
    depth: usize,
) -> Result<(), serde_json::Error> {
    if depth >= 128 {
        return Err(serde_json::Error::custom(
            "capture receipt exceeds maximum JSON depth",
        ));
    }
    let text = raw.get();
    let mut parser = serde_json::Deserializer::from_str(text);
    let visitor = Value(emit, depth);
    match text.as_bytes()[0] {
        b'{' => parser.deserialize_map(visitor)?,
        b'[' => parser.deserialize_seq(visitor)?,
        b'"' => parser.deserialize_str(visitor)?,
        b't' | b'f' => parser.deserialize_bool(visitor)?,
        b'n' => parser.deserialize_unit(visitor)?,
        _ => {
            // Preserve integer identity fields and signed floating-point zero.
            // Non-integral notation goes through serde's public f64 conversion,
            // whose overflow validation is independent of arbitrary_precision.
            if text != "-0" && !text.bytes().any(|byte| matches!(byte, b'.' | b'e' | b'E')) {
                if let Ok(value) = text.parse::<u64>() {
                    (visitor.0)(Event::U64(value));
                    return Ok(());
                }
                if let Ok(value) = text.parse::<i64>() {
                    (visitor.0)(Event::I64(value));
                    return Ok(());
                }
            }
            parser.deserialize_f64(visitor)?;
        }
    }
    parser.end()
}

struct Value<'a, F>(&'a mut F, usize);
impl<'de, F: for<'a> FnMut(Event<'a>)> DeserializeSeed<'de> for Value<'_, F> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        let raw = <&RawValue>::deserialize(deserializer)?;
        dispatch(raw, self.0, self.1).map_err(D::Error::custom)
    }
}
impl<'de, F: for<'a> FnMut(Event<'a>)> Visitor<'de> for Value<'_, F> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON capture receipt")
    }
    fn visit_bool<E>(self, value: bool) -> Result<(), E> {
        (self.0)(Event::Bool(value));
        Ok(())
    }
    fn visit_unit<E>(self) -> Result<(), E> {
        (self.0)(Event::Null);
        Ok(())
    }
    fn visit_i64<E>(self, value: i64) -> Result<(), E> {
        (self.0)(Event::I64(value));
        Ok(())
    }
    fn visit_u64<E>(self, value: u64) -> Result<(), E> {
        (self.0)(Event::U64(value));
        Ok(())
    }
    fn visit_f64<E>(self, value: f64) -> Result<(), E> {
        (self.0)(Event::F64(value));
        Ok(())
    }
    fn visit_str<E>(self, value: &str) -> Result<(), E> {
        (self.0)(Event::String(value));
        Ok(())
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut array: A) -> Result<(), A::Error> {
        (self.0)(Event::Array);
        while array
            .next_element_seed(Value(&mut *self.0, self.1 + 1))?
            .is_some()
        {}
        (self.0)(Event::EndArray);
        Ok(())
    }
    fn visit_map<A: MapAccess<'de>>(self, mut object: A) -> Result<(), A::Error> {
        (self.0)(Event::Object);
        while object.next_key_seed(Key(&mut *self.0))?.is_some() {
            object.next_value_seed(Value(&mut *self.0, self.1 + 1))?;
        }
        (self.0)(Event::EndObject);
        Ok(())
    }
}
struct Key<'a, F>(&'a mut F);
impl<'de, F: for<'a> FnMut(Event<'a>)> DeserializeSeed<'de> for Key<'_, F> {
    type Value = ();
    fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
        deserializer.deserialize_str(self)
    }
}
impl<F: for<'a> FnMut(Event<'a>)> Visitor<'_> for Key<'_, F> {
    type Value = ();
    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a receipt field name")
    }
    fn visit_str<E>(self, value: &str) -> Result<(), E> {
        (self.0)(Event::Key(value));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn duplicate_fields_and_unicode_escape_spelling_reach_the_reader_in_order() {
        let mut seen = Vec::new();
        parse(
            br#"{"a":1,"\u0061":[-7,1.25,true,null,"\u754c"]}"#,
            |event| seen.push(format!("{event:?}")),
        )
        .unwrap();
        assert_eq!(
            seen,
            [
                "Object",
                "Key(\"a\")",
                "U64(1)",
                "Key(\"a\")",
                "Array",
                "I64(-7)",
                "F64(1.25)",
                "Bool(true)",
                "Null",
                "String(\"界\")",
                "EndArray",
                "EndObject"
            ]
        );
    }
    #[test]
    fn complete_validation_rejects_nonfinite_numbers_bad_escapes_and_trailing_input() {
        for bytes in [
            b"1e400".as_slice(),
            br#"{"x":"\ud800"}"#,
            b"[1,]",
            b"{} false",
        ] {
            assert!(parse(bytes, |_| {}).is_err());
        }
        let mut last = String::new();
        assert!(parse(br#"{"a":7,"b":[11,"#, |event| last = format!("{event:?}")).is_err());
        assert!(
            last.is_empty(),
            "invalid syntax is rejected before callbacks"
        );
    }
}

#[cfg(test)]
mod numeric_tests {
    use super::*;
    #[test]
    fn numeric_kinds_and_signed_zero_do_not_depend_on_dependency_features() {
        let mut seen = Vec::new();
        parse(br#"[-0,0,-9223372036854775808,18446744073709551615,1.25,1e3,{"$serde_json::private::Number":"2.5"}]"#,
            |event| seen.push(format!("{event:?}"))).unwrap();
        assert_eq!(
            seen,
            [
                "Array",
                "F64(-0.0)",
                "U64(0)",
                "I64(-9223372036854775808)",
                "U64(18446744073709551615)",
                "F64(1.25)",
                "F64(1000.0)",
                "Object",
                "Key(\"$serde_json::private::Number\")",
                "String(\"2.5\")",
                "EndObject",
                "EndArray"
            ]
        );
    }
    #[test]
    fn nested_containers_have_an_explicit_depth_limit() {
        let input = format!("{}0{}", "[".repeat(129), "]".repeat(129));
        assert!(parse(input.as_bytes(), |_| {}).is_err());
    }
}
