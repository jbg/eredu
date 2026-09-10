//! IFM envelopes expressed through the shared declarative parser and constraints.

use super::dialect::{
    DeclarativeDialectSpec, DeclarativePayloadShape, DelimitedChannel, ExactEnvelope,
    GenerationPromptBehavior, ParallelCallLayout, TaggedParametersEncoding,
};

pub(crate) const REASONING: [(&str, &str, &str); 3] = [
    ("high", "<ifm|think>\n", "</ifm|think>"),
    ("medium", "<ifm|think_fast>\n", "</ifm|think_fast>"),
    ("low", "<ifm|think_faster>\n", "</ifm|think_faster>"),
];

const TOKENS: &[&str] = &[
    "<|ifm|im_start|>",
    "<|ifm|im_end|>",
    "<ifm|think>",
    "</ifm|think>",
    "<ifm|think_fast>",
    "</ifm|think_fast>",
    "<ifm|think_faster>",
    "</ifm|think_faster>",
    "<ifm|tool_calls>",
    "</ifm|tool_calls>",
    "<ifm|tool_call>",
    "</ifm|tool_call>",
    "<ifm|arg_key>",
    "</ifm|arg_key>",
    "<ifm|arg_type>",
    "</ifm|arg_type>",
    "<ifm|arg_value>",
    "</ifm|arg_value>",
];

const fn make_spec(format: usize, reasoning: usize, prefilled: bool) -> DeclarativeDialectSpec {
    DeclarativeDialectSpec {
        generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
        reasoning_template_kwarg: "enable_thinking",
        supports_tool_reasoning: true,
        output: ExactEnvelope {
            prefix: "<ifm|tool_calls>\n",
            suffix: "</ifm|tool_calls>",
        },
        call: ExactEnvelope {
            prefix: "<ifm|tool_call>",
            suffix: if format == 0 {
                "</ifm|tool_call>\n"
            } else {
                ""
            },
        },
        payload_shape: if format == 0 {
            DeclarativePayloadShape::JsonObject
        } else {
            DeclarativePayloadShape::TaggedParameters(TaggedParametersEncoding {
                function_prefix: "",
                function_name_suffix: "\n",
                function_suffix: "</ifm|tool_call>",
                parameter_prefix: "<ifm|arg_key>",
                parameter_name_suffix: "</ifm|arg_key>",
                parameter_type: if format == 2 {
                    Some(ExactEnvelope {
                        prefix: "<ifm|arg_type>",
                        suffix: "</ifm|arg_type>",
                    })
                } else {
                    None
                },
                parameter_value_prefix: "<ifm|arg_value>",
                strip_value_framing: false,
                parameter_suffix: "</ifm|arg_value>",
            })
        },
        json_function: if format == 0 {
            Some(&super::NAME_ARGUMENTS_JSON_FUNCTION)
        } else {
            None
        },
        reasoning_channel: if reasoning == 3 {
            None
        } else {
            Some(DelimitedChannel {
                prefix: REASONING[reasoning].1,
                suffix: REASONING[reasoning].2,
                required: false,
                prefix_in_prompt: prefilled,
            })
        },
        text_channel: None,
        raw_text_before_calls: true,
        call_separator: if format == 0 { "" } else { "\n" },
        parallel_layout: ParallelCallLayout::RepeatedEnvelopes,
        protocol_max_tools: None,
        protocol_max_calls: None,
        auto_activation_trigger: Some("<ifm|tool_calls>\n"),
        required_structural_tokens: TOKENS,
        stop_sequences: &["<|ifm|im_end|>"],
    }
}

const fn specifications() -> [[[DeclarativeDialectSpec; 2]; 4]; 3] {
    let mut specs = [[[make_spec(0, 0, false); 2]; 4]; 3];
    let mut format = 0;
    while format < 3 {
        let mut reasoning = 0;
        while reasoning < 4 {
            specs[format][reasoning] = [
                make_spec(format, reasoning, false),
                make_spec(format, reasoning, true),
            ];
            reasoning += 1;
        }
        format += 1;
    }
    specs
}

static SPECIFICATIONS: [[[DeclarativeDialectSpec; 2]; 4]; 3] = specifications();

pub(crate) fn spec(
    format: &str,
    effort: &str,
    prefilled: bool,
    disabled: bool,
) -> Option<&'static DeclarativeDialectSpec> {
    let format = ["json", "xml", "xml_typed"]
        .iter()
        .position(|&name| name == format)?;
    let reasoning = if disabled {
        3
    } else {
        REASONING.iter().position(|&(name, _, _)| name == effort)?
    };
    Some(&SPECIFICATIONS[format][reasoning][usize::from(prefilled)])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::chat::{
        constraints::ConstraintCompiler,
        dialect::{DialectParameters, DECLARATIVE_DIALECT},
        ParallelToolCallPolicy, ToolChoice,
    };
    use eredu_core::generation::{FinishReason, SemanticEvent};
    use serde_json::{json, Value};

    // Byte-token grammar fixtures; facade tests bind the published special IDs.
    static BYTE_SPECS: [[[DeclarativeDialectSpec; 2]; 4]; 3] = {
        let mut specs = specifications();
        let mut f = 0;
        while f < 3 {
            let mut r = 0;
            while r < 4 {
                specs[f][r][0].required_structural_tokens = &[];
                specs[f][r][1].required_structural_tokens = &[];
                r += 1;
            }
            f += 1;
        }
        specs
    };

    fn tool() -> Value {
        json!({"type":"function", "function":{"name":"lookup", "parameters":{
            "type":"object", "properties":{
                "query":{"type":"string"},
                "count":{"type":"integer", "minimum":1},
                "mixed":{"anyOf":[{"type":"string"},{"type":"integer"}]},
                "filters":{"type":"array","items":{"type":"string"}},
            }, "required":["query","count","mixed","filters"], "additionalProperties":false
        }}})
    }

    fn call(format: usize) -> String {
        let args =
            json!({"query":"\nkeep framing\n", "count":2, "mixed":"007", "filters":["active"]});
        let body = if format == 0 {
            json!({"name":"lookup", "arguments":args}).to_string()
        } else {
            let mut body = "lookup\n".to_owned();
            for (name, kind) in [
                ("query", "string"),
                ("count", "integer"),
                ("mixed", "string"),
                ("filters", "array[string]"),
            ] {
                body.push_str(&format!("<ifm|arg_key>{name}</ifm|arg_key>\n"));
                if format == 2 {
                    body.push_str(&format!("<ifm|arg_type>{kind}</ifm|arg_type>\n"));
                }
                let value = args[name]
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| args[name].to_string());
                body.push_str(&format!("<ifm|arg_value>{value}</ifm|arg_value>\n"));
            }
            body
        };
        format!("<ifm|tool_call>{body}</ifm|tool_call>")
    }

    #[test]
    fn ifm_constraints_and_incremental_parsing_cover_formats_efforts_and_parallel_calls() {
        for (format, specs) in BYTE_SPECS.iter().enumerate() {
            let collection = format!(
                "<ifm|tool_calls>\n{}\n{}\n</ifm|tool_calls>",
                call(format),
                call(format)
            );
            for (effort, &(_, opening, closing)) in REASONING.iter().enumerate() {
                for prefilled in [false, true] {
                    let plan = ConstraintCompiler::synthetic_for_tests()
                        .compile_tool_plan(
                            &DECLARATIVE_DIALECT,
                            DialectParameters::Declarative(&specs[effort][usize::from(prefilled)]),
                            &[tool()],
                            ToolChoice::Required,
                            ParallelToolCallPolicy::Enabled { max_calls: None },
                            vec![],
                        )
                        .unwrap();
                    let output = format!(
                        "{}Check carefully.{closing}\n{collection}",
                        if prefilled { "" } else { opening }
                    );
                    let mut grammar = plan.generation_constraint().grammar_state();
                    for (index, byte) in output.bytes().enumerate() {
                        grammar.commit(u32::from(byte)).unwrap_or_else(|e| panic!("format {format}, effort {effort}, prefilled {prefilled}, index {index}, prefix {:?}: {e}", &output[..index]));
                    }
                    assert!(grammar.is_complete().unwrap());
                    for split in 0..=output.len() {
                        let mut parser = plan.create_parser().unwrap();
                        parser
                            .push(&output[..split])
                            .unwrap_or_else(|e| panic!("format {format}, split {split}: {e}"));
                        let mut branch = parser.fork().unwrap();
                        parser
                            .push(&output[split..])
                            .unwrap_or_else(|e| panic!("format {format}, split {split}: {e}"));
                        parser.finish(FinishReason::GrammarComplete).unwrap();
                        branch.push(&output[split..]).unwrap();
                        branch.finish(FinishReason::GrammarComplete).unwrap();
                        assert_eq!(branch.events(), parser.events());
                        let mut args = [String::new(), String::new()];
                        let mut reasoning = String::new();
                        for event in parser.events() {
                            match event {
                                SemanticEvent::ToolArgumentsDelta {
                                    index,
                                    json_fragment,
                                } => args[*index].push_str(json_fragment),
                                SemanticEvent::ReasoningDelta(value) => reasoning.push_str(value),
                                _ => {}
                            }
                        }
                        assert_eq!(reasoning, "Check carefully.");
                        assert_eq!(
                            parser
                                .events()
                                .iter()
                                .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                                .count(),
                            2
                        );
                        for args in args {
                            assert_eq!(
                                serde_json::from_str::<Value>(&args).unwrap(),
                                json!({"query":"\nkeep framing\n","count":2,"mixed":"007","filters":["active"]})
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn ifm_publisher_generated_weather_call_parses_at_every_byte_boundary() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../../eredu-text/tests/fixtures/k2_horizon/tool-generation.json"
        ))
        .unwrap();
        for reference in fixture["references"].as_array().unwrap() {
            let output = reference["generated_text"]
                .as_str()
                .unwrap()
                .split("<|ifm|im_end|>")
                .next()
                .unwrap();
            let plan = ConstraintCompiler::synthetic_for_tests()
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    DialectParameters::Declarative(&BYTE_SPECS[1][2][1]),
                    fixture["request"]["tools"].as_array().unwrap(),
                    ToolChoice::Auto,
                    ParallelToolCallPolicy::Disabled,
                    vec![],
                )
                .unwrap();
            for split in 0..=output.len() {
                let mut parser = plan.create_parser().unwrap();
                parser.push(&output[..split]).unwrap();
                parser.push(&output[split..]).unwrap();
                parser.finish(FinishReason::GrammarComplete).unwrap();
                let mut arguments = String::new();
                for event in parser.events() {
                    if let SemanticEvent::ToolArgumentsDelta { json_fragment, .. } = event {
                        arguments.push_str(json_fragment);
                    }
                }
                assert_eq!(
                    serde_json::from_str::<Value>(&arguments).unwrap(),
                    json!({"city":"Paris"})
                );
                assert_eq!(
                    parser
                        .events()
                        .iter()
                        .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn ifm_incomplete_calls_and_false_type_annotations_do_not_complete() {
        for (format, specs) in BYTE_SPECS.iter().enumerate() {
            let plan = ConstraintCompiler::synthetic_for_tests()
                .compile_tool_plan(
                    &DECLARATIVE_DIALECT,
                    DialectParameters::Declarative(&specs[3][0]),
                    &[tool()],
                    ToolChoice::Required,
                    ParallelToolCallPolicy::Disabled,
                    vec![],
                )
                .unwrap();
            let call = format!("<ifm|tool_calls>\n{}", call(format));
            for cut in 1..call.len() {
                let mut parser = plan.create_parser().unwrap();
                let _ = parser.push(&call[..cut]);
                let _ = parser.finish(FinishReason::MaxTokens);
                assert!(
                    !parser
                        .events()
                        .iter()
                        .any(|event| matches!(event, SemanticEvent::ToolCallEnd)),
                    "format {format}, cut {cut}"
                );
            }
            if format == 2 {
                let mut parser = plan.create_parser().unwrap();
                assert!(parser
                    .push(&call.replacen("<ifm|arg_type>string", "<ifm|arg_type>integer", 1))
                    .is_err());
                assert!(!parser
                    .events()
                    .iter()
                    .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
            }
        }
    }
}
