use super::*;
use serde_json::{
    allocation::{Allocation, AllocationError, Unenforced},
    json,
};
use std::cell::Cell;
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
        funding: &dyn Allocation,
    ) -> Result<Value, Self::Error> {
        parse_tagged_value(&self.0[name], declared, raw, funding, |value| {
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
    funding: &dyn Allocation,
) -> Result<Vec<TaggedEvent>, TaggedCallError<<Schemas as TaggedSchemas>::Error>> {
    let mut events = Vec::new();
    loop {
        let (consumed, wait, event) = call.advance(encoding(), pending, schemas, funding)?;
        pending.drain(..consumed);
        if event != TaggedEvent::None {
            events.push(event);
        }
        if wait || event == TaggedEvent::Complete {
            return Ok(events);
        }
    }
}
struct Budget {
    calls: Cell<usize>,
    bytes: Cell<usize>,
    fail: usize,
}
impl Allocation for Budget {
    fn reserve(&self, bytes: usize) -> Result<(), AllocationError> {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call == self.fail {
            return Err(AllocationError::Refused);
        }
        self.bytes.set(self.bytes.get() + bytes);
        Ok(())
    }
}
impl Budget {
    fn new(fail: usize) -> Self {
        Self {
            calls: Cell::new(0),
            bytes: Cell::new(0),
            fail,
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
        let mut events = consume(&mut call, &mut pending, &schema, &Unenforced).unwrap();
        let copy_budget = Budget::new(usize::MAX);
        let copied_bytes = call.copy_bytes().unwrap();
        let mut copy = call.try_clone_with_allocations(&copy_budget).unwrap();
        assert_eq!(copy_budget.bytes.get(), copied_bytes);
        pending.push_str(&wire[split..]);
        let mut copy_pending = pending.clone();
        if events.last() != Some(&TaggedEvent::Complete) {
            let following = consume(&mut call, &mut pending, &schema, &Unenforced).unwrap();
            assert_eq!(
                following,
                consume(&mut copy, &mut copy_pending, &schema, &Unenforced).unwrap()
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
fn every_reached_producer_refuses_without_completing_the_call() {
    let schema = schema();
    let wire = "<fn=write><p=content>text</p><p=count>2</p></fn>";
    let observed = Budget::new(usize::MAX);
    consume(
        &mut TaggedCall::default(),
        &mut wire.to_owned(),
        &schema,
        &observed,
    )
    .unwrap();
    assert!(observed.calls.get() > 20);
    for index in 0..observed.calls.get() {
        let budget = Budget::new(index);
        let result = consume(
            &mut TaggedCall::default(),
            &mut wire.to_owned(),
            &schema,
            &budget,
        );
        assert!(result.is_err(), "request {index} ignored");
        assert_eq!(
            budget.calls.get(),
            index + 1,
            "producer continued after refusal {index}"
        );
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
        assert!(
            consume(
                &mut TaggedCall::default(),
                &mut wire.to_owned(),
                &schema(),
                &Unenforced
            )
            .is_err()
        );
    }
}
#[test]
fn shared_serializer_matches_ordinary_and_refuses_every_reached_request() {
    let value = json!({"a":[1,true,null,"é\n\"",{"z":3.5}],"b":[]});
    let expected = r#"{"a":[1,true,null,"é\n\"",{"z":3.5}],"b":[]}"#;
    assert_eq!(serde_json::to_string(&value).unwrap(), expected);
    let observed = Budget::new(usize::MAX);
    assert_eq!(
        value.to_string_with_allocations(&observed).unwrap(),
        expected
    );
    for cut in 0..observed.calls.get() {
        let budget = Budget::new(cut);
        assert!(matches!(
            value.to_string_with_allocations(&budget),
            Err(serde_json::value::ValueWriteError::Allocation(
                AllocationError::Refused
            ))
        ));
        assert_eq!(budget.calls.get(), cut + 1);
    }
}

#[test]
fn declared_nullable_and_array_types_preserve_values_and_refusals() {
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
        let parse = |budget: &dyn Allocation| {
            parse_tagged_value(&schema, declared, raw, budget, |_| {
                Ok::<_, Infallible>(true)
            })
        };
        let paid = Budget::new(usize::MAX);
        assert_eq!(parse(&paid).unwrap(), expected);
        for cut in 0..paid.calls.get() {
            let budget = Budget::new(cut);
            assert!(parse(&budget).is_err());
            assert_eq!(budget.calls.get(), cut + 1);
        }
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
            funding: &dyn Allocation,
        ) -> Result<Value, Self::Error> {
            parse_tagged_value_policy(&TaggedValuePolicy::any(), None, raw, funding, |_| Ok(true))
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
        let (consumed, wait, _) = call
            .advance(encoding(), &wire[offset..], &Any, &Unenforced)
            .unwrap();
        offset += consumed;
        if wait {
            break;
        }
    }
    let paid = Budget::new(usize::MAX);
    let mut copy = call.try_clone_with_allocations(&paid).unwrap();
    assert_eq!(paid.bytes.get(), call.copy_bytes().unwrap());
    assert!(matches!(
        copy.advance(encoding(), "<p=field_0>3</p>", &Any, &Unenforced)
            .unwrap()
            .2,
        TaggedEvent::None
    ));
    assert!(matches!(
        copy.advance(encoding(), "field_0>3</p>", &Any, &Unenforced),
        Err(TaggedCallError::Duplicate)
    ));
    assert_eq!(
        call.advance(encoding(), "</fn>", &Any, &Unenforced)
            .unwrap()
            .2,
        TaggedEvent::Complete
    );
    assert_eq!(call.arguments(), Value::Object(expected).to_string());
}
