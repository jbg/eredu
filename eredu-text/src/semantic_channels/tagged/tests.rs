use super::*;
use serde_json::json;
struct Schemas(Value);
impl TaggedSchemas for Schemas {
    type Error = TaggedValueError<std::convert::Infallible>;
    fn contains_tool(&self, name: &str) -> bool {
        name == "write"
    }
    fn missing_required(&self, _: &str, names: &TaggedParameters) -> bool {
        !names.iter().any(|n| n == "content") || !names.iter().any(|n| n == "count")
    }
    fn parse_parameter(
        &self,
        _: &str,
        name: &str,
        declared: Option<&str>,
        raw: &str,
    ) -> Result<Value, Self::Error> {
        parse_tagged_value(&self.0[name], declared, raw, |value| {
            Ok(match name {
                "content" => value.is_string(),
                "count" => value.as_i64().is_some_and(|n| n >= 2),
                _ => false,
            })
        })
    }
}
fn schema() -> Schemas {
    Schemas(json!({"content":{"type":"string"},"count":{"type":"integer","minimum":2}}))
}
fn encoding() -> TaggedEncoding<'static> {
    TaggedEncoding {
        function_prefix: "<fn=",
        function_name_suffix: ">",
        parameter_prefix: "<p=",
        parameter_name_suffix: ">",
        parameter_type: None,
        parameter_value_prefix: "",
        strip_value_framing: true,
        parameter_suffix: "</p>",
        function_suffix: "</fn>",
    }
}
fn consume(
    call: &mut TaggedCall,
    pending: &mut String,
    schemas: &Schemas,
) -> Result<Vec<TaggedEvent>, TaggedCallError<<Schemas as TaggedSchemas>::Error>> {
    let mut events = Vec::new();
    loop {
        let (consumed, wait, event) = call.advance(encoding(), pending, schemas)?;
        pending.drain(..consumed);
        if event != TaggedEvent::None {
            events.push(event);
        }
        if wait || event == TaggedEvent::Complete {
            return Ok(events);
        }
    }
}
#[test]
fn split_headers_values_and_independent_copies_preserve_exact_arguments() {
    let schema = schema();
    let wire = " \n<fn=write>\r\n<p=content>\n leading é\ntrailing\r\n</p>\t<p=count>2</p> </fn>";
    for split in wire.char_indices().map(|(n, _)| n).chain([wire.len()]) {
        let mut call = TaggedCall::default();
        let mut pending = wire[..split].to_owned();
        let mut events = consume(&mut call, &mut pending, &schema).unwrap();
        let mut copy = call.clone();
        pending.push_str(&wire[split..]);
        let mut copy_pending = pending.clone();
        if events.last() != Some(&TaggedEvent::Complete) {
            let following = consume(&mut call, &mut pending, &schema).unwrap();
            assert_eq!(
                following,
                consume(&mut copy, &mut copy_pending, &schema).unwrap()
            );
            events.extend(following);
        }
        assert_eq!(events, [TaggedEvent::Start, TaggedEvent::Complete]);
        assert_eq!(call.name(), "write");
        assert_eq!(
            call.arguments(),
            r#"{"content":" leading é\ntrailing\r","count":2}"#
        );
        assert_eq!(copy.arguments(), call.arguments());
        assert!(pending.is_empty());
    }
}
#[test]
fn duplicate_missing_unknown_and_invalid_value_stay_rejected() {
    for wire in [
        "<fn=unknown></fn>",
        "<fn=write></fn>",
        "<fn=write><p=content>x</p><p=content>y</p></fn>",
        "<fn=write><p=content>x</p><p=count>1</p></fn>",
    ] {
        assert!(consume(&mut TaggedCall::default(), &mut wire.to_owned(), &schema(),).is_err());
    }
}
#[test]
fn declared_nullable_and_array_types_preserve_values() {
    use std::convert::Infallible;
    let cases = [
        (
            json!({"type":["string","null"]}),
            Some("null"),
            "null",
            Value::Null,
        ),
        (
            json!({"type":["string","null"]}),
            Some("string"),
            "null",
            json!("null"),
        ),
        (json!({"enum":["null",null]}), None, "null", Value::Null),
        (
            json!({"type":"array","items":{"type":"integer"}}),
            Some("array[integer]"),
            "[1,2]",
            json!([1, 2]),
        ),
    ];
    for (schema, declared, raw, expected) in cases {
        let value =
            parse_tagged_value(&schema, declared, raw, |_| Ok::<_, Infallible>(true)).unwrap();
        assert_eq!(value, expected);
    }
}

#[test]
fn many_parameters_keep_insertion_order_and_indexed_duplicate_detection() {
    use std::convert::Infallible;
    struct Any;
    impl TaggedSchemas for Any {
        type Error = TaggedValueError<Infallible>;
        fn contains_tool(&self, _: &str) -> bool {
            true
        }
        fn missing_required(&self, _: &str, _: &TaggedParameters) -> bool {
            false
        }
        fn parse_parameter(
            &self,
            _: &str,
            _: &str,
            _: Option<&str>,
            raw: &str,
        ) -> Result<Value, Self::Error> {
            parse_tagged_value_policy(&TaggedValuePolicy::any(), None, raw, |_| Ok(true))
        }
    }
    let mut wire = String::from("<fn=write>");
    let mut expected = serde_json::Map::new();
    for index in (0..2048).rev() {
        use std::fmt::Write;
        write!(wire, "<p=field_{index}>{index}</p>").unwrap();
        expected.insert(format!("field_{index}"), json!(index));
    }
    let mut call = TaggedCall::default();
    let mut offset = 0;
    while offset < wire.len() {
        let (consumed, wait, _) = call.advance(encoding(), &wire[offset..], &Any).unwrap();
        offset += consumed;
        if wait {
            break;
        }
    }
    let mut copy = call.clone();
    assert!(matches!(
        copy.advance(encoding(), "<p=field_0>3</p>", &Any)
            .unwrap()
            .2,
        TaggedEvent::None
    ));
    assert!(matches!(
        copy.advance(encoding(), "field_0>3</p>", &Any),
        Err(TaggedCallError::Duplicate)
    ));
    assert_eq!(
        call.advance(encoding(), "</fn>", &Any).unwrap().2,
        TaggedEvent::Complete
    );
    assert_eq!(call.arguments(), Value::Object(expected).to_string());
}

#[test]
fn validation_failure_never_falls_back_to_raw_text() {
    #[derive(Debug, thiserror::Error)]
    #[error("validation resource refusal")]
    struct Refused;
    let result =
        parse_tagged_value_policy(&TaggedValuePolicy::any(), None, "true", |_| Err(Refused));
    assert!(matches!(result, Err(TaggedValueError::Validation(Refused))));
}

#[test]
fn mixed_enum_falls_back_only_when_json_is_invalid_for_the_schema() {
    use std::convert::Infallible;
    let schema = json!({"enum": ["1", 2, "plain"]});
    for (raw, expected) in [
        ("1", json!("1")),
        ("2", json!(2)),
        ("plain", json!("plain")),
    ] {
        let value = parse_tagged_value(&schema, None, raw, |value| {
            Ok::<_, Infallible>(schema["enum"].as_array().unwrap().contains(value))
        })
        .unwrap();
        assert_eq!(value, expected);
    }
    let result = parse_tagged_value(&schema, Some("integer"), "1", |value| {
        Ok::<_, Infallible>(schema["enum"].as_array().unwrap().contains(value))
    });
    assert!(matches!(result, Err(TaggedValueError::DeclaredType)));
}
