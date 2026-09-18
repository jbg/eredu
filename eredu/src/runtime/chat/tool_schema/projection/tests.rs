use super::*;
use serde_json::json;

#[test]
fn fragment_relocation_visits_schema_positions_and_preserves_application_data() {
    let input = json!({
        "$defs": {"count": {"type":"integer", "minimum":17}},
        "properties": {"count": {"$ref":"#/$defs/count"}},
        "allOf": [{"if":{"$ref":"#"}, "then":{"not":{"$ref":"#/$defs/count"}}}],
        "dependencies": {"count":["other"], "other":{"$ref":"#"}},
        "items": [{"$ref":"#/$defs/count"}],
        "default": {"$ref":"#/application-data"},
        "enum": [{"$ref":"#/also-data"}],
        "examples": [{"properties":{"$ref":"#/still-data"}}]
    });
    let projected =
        serde_json::to_value(ArgumentsSchema::new(&input, "/payload").unwrap()).unwrap();
    let schema = &projected["allOf"][1];
    assert_eq!(projected["allOf"][0], json!({"type":"object"}));
    assert_eq!(
        schema["properties"]["count"]["$ref"],
        "#/payload/allOf/1/$defs/count"
    );
    assert_eq!(schema["allOf"][0]["if"]["$ref"], "#/payload/allOf/1");
    assert_eq!(
        schema["allOf"][0]["then"]["not"]["$ref"],
        "#/payload/allOf/1/$defs/count"
    );
    assert_eq!(schema["dependencies"]["count"], json!(["other"]));
    assert_eq!(schema["dependencies"]["other"]["$ref"], "#/payload/allOf/1");
    assert_eq!(schema["items"][0]["$ref"], "#/payload/allOf/1/$defs/count");
    for field in ["default", "enum", "examples", "$defs"] {
        assert_eq!(schema[field], input[field]);
    }
    assert_eq!(input["properties"]["count"]["$ref"], "#/$defs/count");
}

#[test]
fn exact_call_envelopes_validate_references_alternatives_and_unicode_ids() {
    let parameters = json!({"$defs":{"count":{"type":"integer","minimum":17}},
        "type":"object", "properties":{"count":{"$ref":"#/$defs/count"}},
        "required":["count"],"additionalProperties":false});
    let other = json!({"type":"object","properties":{"word":{"const":"é🙂"}},
        "required":["word"],"additionalProperties":false});
    let rows = [
        ToolDefinition {
            name: "measure",
            parameters: &parameters,
        },
        ToolDefinition {
            name: "word",
            parameters: &other,
        },
    ];
    for count in [1, 2] {
        let schema = ToolCallSchema::new(
            &rows[..count],
            "name",
            "arg~/box",
            Some(DeclarativeCallId {
                field: "id",
                length: Some(2),
            }),
        )
        .unwrap();
        let bytes = serde_json::to_vec(&schema).unwrap();
        let emitted: Value = serde_json::from_slice(&bytes).unwrap();
        let validator = jsonschema::validator_for(&emitted).unwrap();
        assert!(validator.is_valid(&json!({"name":"measure","arg~/box":{"count":19},"id":"é🙂"})));
        assert!(!validator.is_valid(&json!({"name":"measure","arg~/box":{"count":16},"id":"é🙂"})));
        assert!(!validator.is_valid(&json!({"name":"measure","arg~/box":{"count":19},"id":"é"})));
        assert!(
            !validator.is_valid(
                &json!({"name":"measure","arg~/box":{"count":19},"id":"é🙂", "extra":true})
            )
        );
        assert_eq!(
            validator.is_valid(&json!({"name":"word","arg~/box":{"word":"é🙂"},"id":"ab"})),
            count == 2
        );
    }
}

#[test]
fn unsupported_reference_scopes_relax_only_the_embedded_schema_node() {
    for schema in [
        json!({"$id":"https://example.invalid/schema", "type":"integer"}),
        json!({"$ref":"https://example.invalid/schema"}),
        json!({"$ref":"#anchor"}),
    ] {
        let input = json!({"type":"object","properties":{"value":schema}});
        let projected = serde_json::to_value(ArgumentsSchema::new(&input, "").unwrap()).unwrap();
        assert_eq!(
            projected,
            json!({"allOf":[{"type":"object"},
            {"type":"object","properties":{"value":true}}]})
        );
    }
    for input in [Value::Bool(false), Value::Bool(true)] {
        let projected = serde_json::to_value(ArgumentsSchema::new(&input, "").unwrap()).unwrap();
        let validator = jsonschema::validator_for(&projected).unwrap();
        assert_eq!(
            validator.is_valid(&json!({"count":19})),
            input == Value::Bool(true)
        );
        assert!(!validator.is_valid(&json!([19])));
    }
    assert!(ArgumentsSchema::new(&json!({"properties":{"n":{"$ref":17}}}), "").is_err());
}
