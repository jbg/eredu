use super::*;
use crate::runtime::chat::grammar_text::producer_tests::every_refusal;
use serde_json::json;

#[test]
fn original_python_grammar_producers_refuse_every_reached_allocation() {
    let schema = json!({"type":"object","properties":{
        "a":{"enum":[null,true,false,17,"'\"\\\n水",[1,"x"],{"nested":[false]}]},
        "b":{"type":"array","items":{"type":"object","properties":{"value":{"type":"string"}},"required":["value"]},"minItems":1,"maxItems":3},
        "c":{"type":"string"}},"required":["b"],"additionalProperties":false});
    let tools = [ToolDefinition {
        name: "dispatch",
        parameters: &schema,
    }];
    let output = every_refusal(|funding| {
        Lfm2Dialect::grammar(
            &tools,
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            &[7, 11, 19],
            funding,
        )
    });
    assert!(output.starts_with("start: <[7]> \"[\" python_call \"]\" <[11]>\n"));
    assert!(output.contains("python_call_0: \"dispatch\" \"(\" python_arguments_0 \")\"\n"));
    assert!(output.contains("(\"None\" | \"null\")"));
}

#[test]
fn python_literal_spelling_preserves_nested_values_and_escapes() {
    let funding = PreparationFunding::unmanaged();
    let mut output = GrammarText::new(&funding).unwrap();
    python_value_literal(&mut output, &json!([17, true, null, {"x": false}])).unwrap();
    assert_eq!(
        output.finish(),
        "\"[\" \"17\" \", \" (\"True\" | \"true\") \", \" (\"None\" | \"null\") \", \" \"{\" \"\\\"x\\\"\" \": \" (\"False\" | \"false\") \"}\" \"]\""
    );
    assert_eq!(
        PythonString("a'\"\\\n\u{7}水", '\'').to_string(),
        "'a\\'\"\\\\\\n\\u0007水'"
    );
    assert_eq!(
        PythonString("a'\"\\\n\u{7}水", '"').to_string(),
        "\"a'\\\"\\\\\\n\\u0007水\""
    );
}

#[test]
fn invalid_python_name_diagnostic_is_funded_before_materializing() {
    let error = validate_function_name("bad-name").unwrap_err();
    assert_eq!(
        error.to_string(),
        "LFM2 tool function name \"bad-name\" is not a valid Python identifier"
    );
    every_refusal(|funding| {
        let error = validate_function_name("bad-name")
            .unwrap_err()
            .grammar(funding);
        match error {
            GrammarError::Policy(text) => Ok(text),
            other => Err(other),
        }
    });
}
