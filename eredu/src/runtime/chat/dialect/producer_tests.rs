use super::*;
use crate::runtime::chat::grammar_text::producer_tests::every_refusal;
use serde_json::json;

fn tagged_encoding() -> TaggedParametersEncoding {
    TaggedParametersEncoding {
        function_prefix: "<function=",
        function_name_suffix: ">",
        function_suffix: "</function>",
        parameter_prefix: "<parameter=",
        parameter_name_suffix: ">",
        parameter_suffix: "</parameter>",
        parameter_type: Some(ExactEnvelope {
            prefix: "<type>",
            suffix: "</type>",
        }),
        parameter_value_prefix: "<value>",
        strip_value_framing: true,
    }
}

#[test]
fn original_tagged_grammar_producers_refuse_every_reached_allocation() {
    let schema = json!({"type":"object","properties":{
        "a":{"type":"string","enum":["line\n水","\nframed\n","</parameter>"]},
        "b":{"type":"array","items":{"type":"integer"}},
        "c":{"enum":["mixed",17]},"d":{"type":"string"}},"required":["b"],"additionalProperties":false});
    let tools = [ToolDefinition {
        name: "dispatch",
        parameters: &schema,
    }];
    let grammar =
        every_refusal(|funding| tagged_parameters_grammar(&tools, tagged_encoding(), funding));
    assert!(grammar.contains("\"array[integer]\""));
    assert!(grammar.contains("tagged_value_1_json: %json {"));
    assert!(grammar.contains("(tagged_parameter_0_0 tagged_parameter_0_1 (tagged_parameter_0_2 (tagged_parameter_0_3 \"\")?)?)?"));
    assert!(grammar.ends_with("tagged_call: tagged_tool_0\n"));
}

#[test]
fn original_structural_grammar_producers_refuse_every_reached_allocation() {
    let schema = json!({"type":"object","properties":{
        "a":{"enum":[null,false,17,"水<x>",[1,"x"],{"nested":[true]}]},
        "b":{"type":"array","items":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]},"minItems":1,"maxItems":3}},"required":["b"]});
    let tools = [ToolDefinition {
        name: "dispatch",
        parameters: &schema,
    }];
    let encoding = StructuralObjectEncoding {
        name_prefix: "<name>",
        string_delimiter: "<x>",
    };
    let grammar = every_refusal(|funding| {
        structural_object_grammar(&tools, encoding, &["<name>", "<x>"], &[7, 11], funding)
    });
    assert!(grammar.starts_with("structural_call: <[7]> \"dispatch\" structural_object_0\n"));
    assert!(grammar.contains("<[11]> \"水\" <[11]> <[11]>"));
}

#[test]
fn compact_type_spelling_preserves_arrays_references_and_actual_union_values() {
    let cases = [
        (
            json!({"type":"array","items":{"type":"array","items":{"$ref":"#/defs/Thing"}}}),
            "array[array[Thing]]",
        ),
        (json!({"type":["integer"]}), "integer"),
        (json!({"items":{"properties":{}}}), "array[object]"),
    ];
    for (schema, expected) in cases {
        assert_eq!(tagged_type_name(&schema, None), expected);
    }
    assert_eq!(
        tagged_type_name(&json!({"type":["string","integer"]}), Some(&json!(7))),
        "integer"
    );
}

#[test]
fn tagged_local_reference_decoding_uses_the_original_funding_owner() {
    let schema = json!({"type":"object","properties":{"field":{"$ref":"#/$defs/a~1b~0"}},
        "$defs":{"a/b~":{"type":"string","enum":["accepted"]}},"required":["field"]});
    let tools = [ToolDefinition {
        name: "dispatch",
        parameters: &schema,
    }];
    let grammar =
        every_refusal(|funding| tagged_parameters_grammar(&tools, tagged_encoding(), funding));
    assert!(grammar.contains("tagged_value_0: \"\\naccepted\\n\""));
}
