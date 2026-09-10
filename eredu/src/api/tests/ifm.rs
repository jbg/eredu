use super::*;

const TEMPLATES: [(&str, &str); 4] = [
    (
        "dense-safetensors",
        include_str!("../../../../eredu-text/tests/fixtures/k2_horizon/dense-safetensors.jinja"),
    ),
    (
        "dense-gguf",
        include_str!("../../../../eredu-text/tests/fixtures/k2_horizon/dense-gguf.jinja"),
    ),
    (
        "mova-safetensors",
        include_str!("../../../../eredu-text/tests/fixtures/k2_horizon/mova-safetensors.jinja"),
    ),
    (
        "mova-gguf",
        include_str!("../../../../eredu-text/tests/fixtures/k2_horizon/mova-gguf.jinja"),
    ),
];

fn tokenizer() -> ChatTokenizer {
    llama_chat_tokenizer(
        32,
        crate::runtime::chat::ifm::spec("xml", "high", true, false)
            .unwrap()
            .required_structural_tokens,
    )
}

#[test]
fn ifm_released_templates_prepare_exact_prompts_and_recognize_all_formats() {
    let cases: Vec<serde_json::Value> = serde_json::from_str(include_str!(
        "../../../../eredu-text/tests/fixtures/k2_horizon/reference.json"
    ))
    .unwrap();
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    for case in cases {
        let mut tokenizer = tokenizer();
        tokenizer.set_template_kwargs(case["template_defaults"].as_object().unwrap().clone());
        let template = TEMPLATES
            .iter()
            .find(|(name, _)| *name == case["template"].as_str().unwrap())
            .unwrap()
            .1;
        let prepared = prepare_chat_from_parts(
            &mut tokenizer,
            template.into(),
            "unrelated-model-name",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: case["messages"].as_array().unwrap().clone(),
                tools: case["tools"].as_array().unwrap().clone(),
                add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap(),
                extra_template_kwargs: case["kwargs"].as_object().unwrap().clone(),
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("{}: {error}", case["name"]));
        assert_eq!(
            prepared.format_profile_identity(),
            Some("ifm.tools.v1"),
            "{}",
            case["name"]
        );
        assert_eq!(
            prepared.rendered_prompt(),
            case["rendered"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        assert!(prepared.semantic_runtime_plan().is_some());
        assert!(prepared.native_tool_support().is_supported());
    }
}

#[test]
fn ifm_reasoning_disable_is_established_per_artifact_template() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    for (name, template) in TEMPLATES {
        let result = prepare_chat_from_parts(
            &mut tokenizer(),
            template.into(),
            "unrelated",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: vec![json!({"role": "user", "content": "Hello"})],
                enable_thinking: Some(false),
                add_generation_prompt: true,
                ..Default::default()
            },
        );
        if name.ends_with("gguf") {
            assert!(result
                .unwrap()
                .rendered_prompt()
                .ends_with("<ifm|think>\n</ifm|think>\n"));
        } else {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("reasoning-disable"));
        }
    }
}
