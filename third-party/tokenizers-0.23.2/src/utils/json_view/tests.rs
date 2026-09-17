use super::*;

#[test]
fn checked_json_views_preserve_decoded_keys_values_and_original_borrow() {
    let input =
        r#"{"same":"first","s\u0061me":"last","items":["\uD83D\uDE42",null,{"a":1,"\u0061":2}]}"#;
    let document = Document::parse(input.as_bytes()).unwrap();
    assert_eq!(document.source().as_ptr(), input.as_ptr());
    let object = document.root().object().unwrap();
    assert_eq!(object.unique_len(), 2);
    assert!(object.get("same").unwrap().string().unwrap().is("last"));
    let mut items = object.get("items").unwrap().array().unwrap();
    let scalar = items.next().unwrap().string().unwrap();
    assert_eq!(scalar.bytes().collect::<Vec<_>>(), "🙂".as_bytes());
    assert_eq!(scalar.len(), "🙂".len());
    assert!(items.next().unwrap().is_null());
    assert_eq!(items.next().unwrap().object().unwrap().unique_len(), 1);
    assert!(items.next().is_none());
    assert!(document.validate_integer_profile().is_ok());
}

#[test]
fn integer_profile_is_distinct_from_grammar_and_ignores_numeric_strings() {
    for input in [
        "0",
        "-0",
        "9223372036854775807",
        "-9223372036854775808",
        "18446744073709551615",
    ] {
        Document::parse(input.as_bytes())
            .unwrap()
            .validate_integer_profile()
            .unwrap();
        assert!(serde_json::from_str::<serde_json::Value>(input).is_ok());
    }
    for (input, kind) in [
        ("1e999", IntegerProfileErrorKind::NonInteger),
        ("1.25", IntegerProfileErrorKind::NonInteger),
        ("-2e-3", IntegerProfileErrorKind::NonInteger),
        ("18446744073709551616", IntegerProfileErrorKind::OutOfRange),
        ("-9223372036854775809", IntegerProfileErrorKind::OutOfRange),
    ] {
        let document = Document::parse(input.as_bytes()).unwrap();
        let error = document.validate_integer_profile().unwrap_err();
        assert_eq!(error.offset, 0);
        assert_eq!(error.kind, kind);
    }
    // Ordinary serde accepts finite fractional/exponent numbers, but this first
    // bounded constructor profile intentionally does not yet accept them.
    assert!(serde_json::from_str::<serde_json::Value>("1.25").is_ok());
    assert!(serde_json::from_str::<serde_json::Value>("-2e-3").is_ok());
    assert!(serde_json::from_str::<serde_json::Value>("1e999").is_err());
    let strings =
        r#"{"1e999":"-9223372036854775809", "quote":"\"1.25\"", "nested":["1e999",8192]}"#;
    Document::parse(strings.as_bytes())
        .unwrap()
        .validate_integer_profile()
        .unwrap();
}

#[test]
fn whole_document_validation_precedes_any_subvalue_or_numeric_profile() {
    for input in [
        b"{} trailing".as_slice(),
        br#"{"a":01}"#,
        br#"["\uD800"]"#,
        br#"["\uDC00"]"#,
        b"[\xff]",
        br#"{"a":1,}"#,
    ] {
        assert!(Document::parse(input).is_err());
    }
    let deep = format!("{}0{}", "[".repeat(129), "]".repeat(129));
    assert!(Document::parse(deep.as_bytes()).unwrap_err().depth_limit);
    assert!(Document::control_bytes().unwrap() > json::stack_bytes());
}
