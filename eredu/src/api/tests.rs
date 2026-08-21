use crate::api::metadata::{
    eos_token_ids_from_sidecar_dir, gguf_eos_token_ids, merge_eos_token_id_sources,
    read_checkpoint_generation_config,
};
use crate::api::request::prepare_chat_from_parts;
use crate::api::tokenizer::{load_chat_template, load_tokenizer_template_kwargs};
use crate::api::{chat_template_kwargs, load_tokenizer, TextModelError};
use crate::{
    core::generation::{
        resolve_generation_config, FinishReason, GenerationConfigOverrides, SemanticEvent,
    },
    core::GgufArchitecture,
    runtime::chat::constraints::ConstraintCompiler,
    runtime::chat::{
        ChatTemplateRequest, NativeToolSupport, ParallelToolCallPolicy, ToolChoice,
        SYNTHETIC_STRUCTURAL_TOKEN, SYNTHETIC_TOOL_TEMPLATE,
    },
};
use eredu_text::tokenizer::chat_template_kwargs as inspect_chat_template_kwargs;
use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
use eredu_text::tokenizer::{ChatTemplateIdentity, ModelChatTemplate};
use serde_json::json;
use std::{
    collections::BTreeMap,
    fs,
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};
use tokenizers::{models::wordlevel::WordLevel, AddedToken, Tokenizer};

#[cfg(feature = "mlx")]
mod mlx;

static TEMP_DIR_COUNTER: AtomicUsize = AtomicUsize::new(0);

const QWEN25_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/qwen2.5-7b-instruct-acbd9653.jinja");
const QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/qwen3-0.6b-7e4ae267.jinja");
const QWEN3_OLDER_TOKENIZER_CONFIG: &str =
    include_str!("../../../eredu-text/tests/fixtures/qwen3/tokenizer_config.json");
const QWEN3_VL_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/qwen3-vl-2b-instruct-89644892.jinja");
const KIMI_LINEAR_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/kimi-linear-48b-a3b-instruct.jinja");
const HERMES2_PRO_TOOL_USE_FIXTURE: &str = include_str!(
    "../../tests/fixtures/chat_templates/hermes-2-pro-llama-3-8b-f798274b-tool-use.jinja"
);
const MISTRAL7_V03_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/mistral-7b-instruct-v0.3-c170c708.jinja");
const MINISTRAL8_2410_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/ministral-8b-instruct-2410-2f494a19.jinja");
const LLAMA31_33_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/llama-3.1-3.3-e10ca381.jinja");
const LLAMA32_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/llama-3.2-5816fce1.jinja");
const LLAMA4_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/llama-4-01a91bfb.jinja");
const NEMOTRON_NANO_FIXTURE_WITH_TERMINATOR: &str = include_str!(
    "../../tests/fixtures/chat_templates/llama-3.1-nemotron-nano-8b-v1-072b9ab4.jinja"
);
const NEMOTRON_NANO_V2_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/nemotron-nano-v2-6533e8de.jinja");
const GEMMA4_EDGE_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/gemma-4-e2b-it-3e22461f.jinja");
const UNSLOTH_GEMMA4_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/unsloth-gemma-4-26b-a4b-it-94899c0f.jinja");
const GPT_OSS_HARMONY_CURRENT_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/gpt-oss-harmony-a4c9919c.jinja");
const GPT_OSS_HARMONY_ESCAPED_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/gpt-oss-harmony-b474759b.jinja");
const GPT_OSS_HARMONY_INITIAL_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/gpt-oss-harmony-f8d92557.jinja");
const LFM2_CLASSIC_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/lfm2-classic-b3afba27.jinja");
const LFM25_8B_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/lfm2.5-8b-a1b-5673e0de.jinja");
const LFM25_VL_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/lfm2.5-vl-450m-fc6221ca.jinja");
const DEEPSEEK_V3_TOOL_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/deepseek-v3-tools-7e28c67d.jinja");
const DEEPSEEK_V31_TOOL_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/deepseek-v3.1-tools-ef1ab230.jinja");
const INKLING_SMALL_FIXTURE: &str =
    include_str!("../../tests/fixtures/chat_templates/inkling-small-8cc5877b.jinja");
const MUSE_GLIMMER_FIXTURE_WITH_TERMINATOR: &str =
    include_str!("../../tests/fixtures/chat_templates/muse-glimmer-30b-97c77dff.jinja");

fn temp_model_dir(config: &str) -> std::path::PathBuf {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = TEMP_DIR_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "model_metadata_test_{}_{}_{}",
        std::process::id(),
        id,
        counter
    ));
    fs::create_dir_all(&dir).unwrap();
    fs::write(dir.join("config.json"), config).unwrap();
    Tokenizer::new(WordLevel::default())
        .save(dir.join("tokenizer.json"), false)
        .unwrap();
    dir
}

fn synthetic_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    let ordinary = (0..preceding_tokens)
        .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
        .collect::<Vec<_>>();
    raw.add_tokens(ordinary).unwrap();
    assert_eq!(
        raw.add_special_tokens([
            AddedToken::from(SYNTHETIC_STRUCTURAL_TOKEN, true).normalized(false)
        ])
        .unwrap(),
        1
    );
    ChatTokenizer::from_tokenizer(raw)
}

fn production_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens([AddedToken::from("<|im_end|>", true).normalized(false)])
            .unwrap(),
        1
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([
        ("bos_token".into(), json!("<|begin_of_text|>")),
        ("add_vision_id".into(), json!(false)),
    ]));
    tokenizer
}

fn mistral_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens([
            AddedToken::from("[TOOL_CALLS]", true).normalized(false),
            AddedToken::from("</s>", true).normalized(false),
        ])
        .unwrap(),
        2
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([
        ("bos_token".into(), json!("<s>")),
        ("eos_token".into(), json!("</s>")),
    ]));
    tokenizer
}

fn llama_chat_tokenizer(preceding_tokens: usize, structural_tokens: &[&str]) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            structural_tokens
                .iter()
                .map(|token| AddedToken::from((*token).to_owned(), true).normalized(false))
                .collect::<Vec<_>>()
        )
        .unwrap(),
        structural_tokens.len()
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([
        ("bos_token".into(), json!("<|begin_of_text|>")),
        ("eos_token".into(), json!("<|eot_id|>")),
    ]));
    tokenizer
}

fn gemma4_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            [
                "<|channel>",
                "<channel|>",
                "<|tool_call>",
                "<tool_call|>",
                "<|\"|>",
                "<|tool_response>",
                "<turn|>",
            ]
            .map(|token| AddedToken::from(token, true).normalized(false))
        )
        .unwrap(),
        7
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([
        ("bos_token".into(), json!("<bos>")),
        ("eos_token".into(), json!("<eos>")),
    ]));
    tokenizer
}

fn gemma4_gguf_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_tokens(
            [
                "<|channel>",
                "<channel|>",
                "<|tool_call>",
                "<tool_call|>",
                "<|\"|>",
                "<|tool_response>",
            ]
            .map(|token| AddedToken::from(token, false).normalized(false))
        )
        .unwrap(),
        6
    );
    assert_eq!(
        raw.add_special_tokens([AddedToken::from("<turn|>", true).normalized(false)])
            .unwrap(),
        1
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([
        ("bos_token".into(), json!("<bos>")),
        ("eos_token".into(), json!("<eos>")),
    ]));
    tokenizer
}

fn gemma4_reasoning_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            ["<|channel>", "<channel|>", "<|tool_response>", "<turn|>"]
                .map(|token| AddedToken::from(token, true).normalized(false))
        )
        .unwrap(),
        4
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([
        ("bos_token".into(), json!("<bos>")),
        ("eos_token".into(), json!("<eos>")),
    ]));
    tokenizer
}

fn harmony_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            [
                "<|start|>",
                "<|end|>",
                "<|message|>",
                "<|channel|>",
                "<|constrain|>",
                "<|return|>",
                "<|call|>",
            ]
            .map(|token| AddedToken::from(token, true).normalized(false))
        )
        .unwrap(),
        7
    );
    ChatTokenizer::from_tokenizer(raw)
}

fn lfm2_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            ["<|tool_call_start|>", "<|tool_call_end|>", "<|im_end|>"]
                .map(|token| AddedToken::from(token, true).normalized(false))
        )
        .unwrap(),
        3
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([(
        "bos_token".into(),
        json!("<|startoftext|>"),
    )]));
    tokenizer
}

fn deepseek_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            [
                "<｜tool▁calls▁begin｜>",
                "<｜tool▁calls▁end｜>",
                "<｜tool▁call▁begin｜>",
                "<｜tool▁call▁end｜>",
                "<｜tool▁sep｜>",
                "<｜end▁of▁sentence｜>",
            ]
            .map(|token| AddedToken::from(token, true).normalized(false))
        )
        .unwrap(),
        6
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([(
        "bos_token".into(),
        json!("<｜begin▁of▁sentence｜>"),
    )]));
    tokenizer
}

fn inkling_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            [
                "<|message_model|>",
                "<|content_text|>",
                "<|content_thinking|>",
                "<|content_invoke_tool_json|>",
                "<|end_message|>",
                "<|content_model_end_sampling|>",
            ]
            .map(|token| AddedToken::from(token, true).normalized(false))
        )
        .unwrap(),
        6
    );
    ChatTokenizer::from_tokenizer(raw)
}

fn muse_atem_chat_tokenizer(preceding_tokens: usize) -> ChatTokenizer {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_tokens(
        (0..preceding_tokens)
            .map(|index| AddedToken::from(format!("ordinary_{index}"), false))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(
        raw.add_special_tokens(
            ["<|start|>", "<|message|>", "<|eom|>", "<|eot|>"]
                .map(|token| AddedToken::from(token, true).normalized(false))
        )
        .unwrap(),
        4
    );
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    tokenizer.set_template_kwargs(serde_json::Map::from_iter([(
        "bos_token".into(),
        json!(""),
    )]));
    tokenizer
}

fn production_tool(name: &str) -> serde_json::Value {
    json!({
        "type": "function",
        "function": {
            "name": name,
            "description": "Look up one integer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "integer",
                        "description": "The integer to look up."
                    }
                },
                "required": ["value"],
                "additionalProperties": false
            }
        }
    })
}

#[test]
fn production_kimi_template_renders_parallel_tools_and_selects_native_dialect() {
    let mut raw = Tokenizer::new(WordLevel::default());
    raw.add_special_tokens(
        [
            "<|tool_calls_section_begin|>",
            "<|tool_calls_section_end|>",
            "<|tool_call_begin|>",
            "<|tool_call_argument_begin|>",
            "<|tool_call_end|>",
            "<|im_end|>",
            "<|im_system|>",
            "<|im_user|>",
            "<|im_assistant|>",
            "<|im_middle|>",
        ]
        .map(|token| AddedToken::from(token, true).normalized(false)),
    )
    .unwrap();
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(KIMI_LINEAR_FIXTURE.into()),
        "moonshotai/Kimi-Linear-48B-A3B-Instruct",
        &[163586],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![
                json!({"role": "user", "content": "Look up two values."}),
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "functions.lookup:0",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"value": 1}}
                        },
                        {
                            "id": "functions.lookup:1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"value\":2}"}
                        }
                    ]
                }),
                json!({
                    "role": "tool",
                    "tool_call_id": "functions.lookup:0",
                    "content": "{\"result\":1}"
                }),
            ],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        prepared.format_profile_identity(),
        Some("kimi-k2.native-tools.v1")
    );
    assert!(matches!(
        prepared.native_tool_support(),
        NativeToolSupport::Supported
    ));
    assert!(prepared.rendered_prompt().contains(concat!(
        "<|tool_call_begin|>functions.lookup:0",
        "<|tool_call_argument_begin|>{\"value\":1}<|tool_call_end|>"
    )));
    assert!(prepared.rendered_prompt().contains(
        "<|tool_call_begin|>functions.lookup:1<|tool_call_argument_begin|>{\"value\":2}"
    ));
    assert!(prepared
        .rendered_prompt()
        .contains("## Return of functions.lookup:0\n{\"result\":1}"));
    assert_eq!(
        prepared.generation_prompt(),
        "<|im_assistant|>assistant<|im_middle|>"
    );
}

fn plan_accepts(plan: &crate::runtime::chat::GenerationRuntimePlan, output: &str) -> bool {
    let mut grammar = plan.generation_constraint().grammar_state();
    let structural_tokens = plan.structural_tokens().collect::<Vec<_>>();
    let mut offset = 0;
    while offset < output.len() {
        if let Some((token_id, spelling)) = output
            .is_char_boundary(offset)
            .then(|| {
                structural_tokens
                    .iter()
                    .find(|(_, spelling)| output[offset..].starts_with(*spelling))
            })
            .flatten()
        {
            if grammar.commit(*token_id).is_err() {
                return false;
            }
            offset += spelling.len();
            continue;
        }
        if grammar
            .commit(u32::from(output.as_bytes()[offset]))
            .is_err()
        {
            return false;
        }
        offset += 1;
    }
    grammar.is_complete().unwrap()
}

fn tool_argument_events(events: &[SemanticEvent]) -> Vec<String> {
    let mut arguments = Vec::<String>::new();
    for event in events {
        match event {
            SemanticEvent::ToolCallStart { index, .. } => {
                assert_eq!(*index, arguments.len());
                arguments.push(String::new());
            }
            SemanticEvent::ToolArgumentsDelta {
                index,
                json_fragment,
            } => arguments[*index].push_str(json_fragment),
            _ => {}
        }
    }
    arguments
}

#[test]
fn prepares_prompt_and_generation_contribution_separately() {
    let raw = Tokenizer::new(WordLevel::default());
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let template = ModelChatTemplate::Single(
        concat!(
            "{% for message in messages %}{{ message.role }}={{ message.content }};",
            "{% endfor %}tools={{ tools|length }};",
            "{% if enable_thinking %}thinking=on;{% else %}thinking=off;{% endif %}",
            "tone={{ tone }}",
            "{% if add_generation_prompt %}<assistant>{% endif %}",
        )
        .into(),
    );
    let request = ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "hello"})],
        tools: vec![json!({
            "type": "function",
            "function": {"name": "lookup", "parameters": {"type": "object"}}
        })],
        tool_choice: ToolChoice::Required,
        parallel_tool_calls: ParallelToolCallPolicy::Enabled {
            max_calls: std::num::NonZeroUsize::new(2),
        },
        enable_thinking: Some(false),
        reasoning_effort: None,
        allow_unparsed_reasoning: false,
        add_generation_prompt: false,
        extra_template_kwargs: serde_json::Map::from_iter([("tone".into(), json!("brief"))]),
    };

    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        template,
        "chat-preparation-test",
        &[7, 8],
        None,
        request,
    )
    .unwrap();

    assert_eq!(
        prepared.rendered_prompt(),
        "user=hello;tools=1;thinking=off;tone=brief"
    );
    assert_eq!(prepared.generation_prompt(), "<assistant>");
    assert_eq!(prepared.template_identity(), &ChatTemplateIdentity::Single);
    assert_eq!(prepared.format_profile_identity(), None);
    assert_eq!(prepared.eos_token_ids(), &[7, 8]);
    assert!(prepared.preserved_structural_token_ids().is_empty());
    assert!(prepared.profile_stop_sequences().is_empty());
    assert!(matches!(
        prepared.native_tool_support(),
        NativeToolSupport::Unsupported { reason }
            if reason.contains("no behavioral format recognizer")
    ));
}

#[test]
fn tool_choice_none_does_not_render_tool_definitions() {
    let raw = Tokenizer::new(WordLevel::default());
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let template = ModelChatTemplate::Single(
        "tools={{ tools|length }}{% for tool in tools %}:{{ tool.function.name }}{% endfor %}"
            .into(),
    );

    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        template,
        "none-hides-tools",
        &[],
        None,
        ChatTemplateRequest {
            tools: vec![json!({
                "type": "function",
                "function": {
                    "name": "must_not_be_rendered",
                    "parameters": {"type": "object"}
                }
            })],
            tool_choice: ToolChoice::None,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert_eq!(prepared.rendered_prompt(), "tools=0");
    assert!(!prepared.rendered_prompt().contains("must_not_be_rendered"));
}

#[test]
fn preparation_selects_named_tool_template_without_model_type_fallback() {
    let raw = Tokenizer::new(WordLevel::default());
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let template = ModelChatTemplate::Named(BTreeMap::from([
        ("default".into(), "default".into()),
        (
            "tool_use".into(),
            "tool-use{% if add_generation_prompt %}:generate{% endif %}".into(),
        ),
    ]));
    let request = ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "hello"})],
        tools: vec![json!({"type": "function"})],
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    };

    let prepared =
        prepare_chat_from_parts(&mut tokenizer, template, "llama", &[], None, request).unwrap();

    assert_eq!(prepared.rendered_prompt(), "tool-use:generate");
    assert_eq!(prepared.generation_prompt(), ":generate");
    assert_eq!(
        prepared.template_identity(),
        &ChatTemplateIdentity::Named("tool_use".into())
    );
    assert_eq!(prepared.format_profile_identity(), None);
}

#[test]
fn production_qwen_profile_renders_history_generation_prompt_and_dynamic_tokens() {
    let template = QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .unwrap();
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let messages = vec![
        json!({"role": "user", "content": "first"}),
        json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "inspect",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "lookup", "arguments": {"value": 1}}
            }]
        }),
        json!({"role": "tool", "name": "lookup", "content": "{\"result\":1}"}),
        json!({"role": "user", "content": "again"}),
    ];

    for (add_generation_prompt, expected_suffix) in [
        (false, ""),
        (true, "<|im_start|>assistant\n<think>\n\n</think>\n\n"),
    ] {
        let mut tokenizer = production_chat_tokenizer(9);
        let prepared = prepare_chat_from_parts(
            &mut tokenizer,
            ModelChatTemplate::Single(template.into()),
            "deliberately-not-a-qwen-model-type",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: messages.clone(),
                tools: vec![production_tool("lookup")],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                enable_thinking: Some(false),
                add_generation_prompt,
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();

        assert_eq!(
            prepared.format_profile_identity(),
            Some("qwen.xml-tools.reasoning.v1")
        );
        assert_eq!(
            prepared.generation_prompt(),
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        if add_generation_prompt {
            assert!(prepared.rendered_prompt().ends_with(expected_suffix));
        } else {
            assert!(!prepared
                .rendered_prompt()
                .ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        }
        assert!(prepared.rendered_prompt().contains(
            "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"value\":1}}\n</tool_call>"
        ));
        assert!(prepared
            .rendered_prompt()
            .contains("<tool_response>\n{\"result\":1}\n</tool_response>"));
        assert_eq!(prepared.preserved_structural_token_ids(), &[9]);
        assert_eq!(prepared.profile_stop_sequences(), ["<|im_end|>"]);
        let plan = prepared
            .tool_runtime_plan()
            .unwrap_or_else(|| panic!("registered Qwen profile must be supported"));
        assert_eq!(plan.auto_activation_trigger(), Some("<tool_call>\n"));
    }
}

#[test]
fn qwen25_instruct_template_renders_chat_tools_and_checkpoint_stops() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = production_chat_tokenizer(9);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(QWEN25_FIXTURE.into()),
        "qwen2",
        &[151_643, 151_645],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Use lookup."})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Auto,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert!(prepared
        .rendered_prompt()
        .contains("<|im_start|>user\nUse lookup.<|im_end|>"));
    assert!(prepared
        .rendered_prompt()
        .ends_with("<|im_start|>assistant\n"));
    assert_eq!(prepared.eos_token_ids(), &[151_643, 151_645]);
    assert_eq!(prepared.profile_stop_sequences(), ["<|im_end|>"]);
    assert_eq!(prepared.format_profile_identity(), Some("xml-tools.v1"));
    assert!(prepared.tool_runtime_plan().is_some());
}

#[test]
fn production_qwen_vl_and_named_hermes_templates_prepare_without_architecture_keys() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut qwen_vl_tokenizer = production_chat_tokenizer(3);
    let qwen_vl = prepare_chat_from_parts(
        &mut qwen_vl_tokenizer,
        ModelChatTemplate::Single(QWEN3_VL_FIXTURE.into()),
        "unrelated",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({
                "role": "user",
                "content": [
                    {"type": "image", "image": "fixture"},
                    {"type": "text", "text": "describe"}
                ]
            })],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(qwen_vl.format_profile_identity(), Some("xml-tools.v1"));
    assert_eq!(qwen_vl.generation_prompt(), "<|im_start|>assistant\n");
    assert!(qwen_vl
        .rendered_prompt()
        .contains("<|vision_start|><|image_pad|><|vision_end|>describe"));
    let qwen_vl_plan = qwen_vl
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered Qwen-VL profile must be supported"));
    assert_eq!(qwen_vl_plan.auto_activation_trigger(), None);

    let mut hermes_tokenizer = production_chat_tokenizer(5);
    let hermes = prepare_chat_from_parts(
        &mut hermes_tokenizer,
        ModelChatTemplate::Named(BTreeMap::from([
            ("default".into(), "default template".into()),
            ("tool_use".into(), HERMES2_PRO_TOOL_USE_FIXTURE.into()),
        ])),
        "qwen-would-be-a-misleading-model-type",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![
                json!({"role": "user", "content": "first"}),
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"value\":1}"
                        }
                    }]
                }),
                json!({"role": "tool", "content": "{\"result\":1}"}),
                json!({"role": "user", "content": "next"}),
            ],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        hermes.template_identity(),
        &ChatTemplateIdentity::Named("tool_use".into())
    );
    assert_eq!(hermes.format_profile_identity(), Some("xml-tools.v1"));
    assert_eq!(hermes.generation_prompt(), "<|im_start|>assistant\n");
    assert!(hermes.rendered_prompt().contains(
        "<tool_call>\n{\"name\": \"lookup\", \"arguments\": {\"value\":1}}\n</tool_call>"
    ));
    assert!(hermes
        .rendered_prompt()
        .contains("<tool_response>\n{\"result\":1}\n</tool_response>"));
}

#[test]
fn behavioral_recognition_survives_nonsemantic_template_refactors() {
    fn prepare(tokenizer: &mut ChatTokenizer, template: String, expected_identity: &str) {
        let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
        let prepared = prepare_chat_from_parts(
            tokenizer,
            ModelChatTemplate::Single(template),
            "architecture-metadata-is-not-a-recognition-key",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: vec![json!({"role": "user", "content": "Use a tool."})],
                tools: vec![production_tool("lookup")],
                tool_choice: ToolChoice::Auto,
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();
        assert_eq!(prepared.format_profile_identity(), Some(expected_identity));
        assert!(prepared.tool_runtime_plan().is_some());
    }

    let qwen3_current = QWEN3_CURRENT_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .unwrap();
    let qwen3_older_config: serde_json::Value =
        serde_json::from_str(QWEN3_OLDER_TOKENIZER_CONFIG).unwrap();
    let qwen3_older = qwen3_older_config["chat_template"].as_str().unwrap();
    for (template, identity) in [
        (QWEN25_FIXTURE, "xml-tools.v1"),
        (qwen3_older, "qwen.xml-tools.reasoning.v1"),
        (qwen3_current, "qwen.xml-tools.reasoning.v1"),
        (QWEN3_VL_FIXTURE, "xml-tools.v1"),
        (HERMES2_PRO_TOOL_USE_FIXTURE, "xml-tools.v1"),
    ] {
        prepare(
            &mut production_chat_tokenizer(7),
            format!("{template}\n{{# nonsemantic converter refactor #}}"),
            identity,
        );
    }

    for fixture in [
        GPT_OSS_HARMONY_CURRENT_FIXTURE_WITH_TERMINATOR,
        GPT_OSS_HARMONY_ESCAPED_FIXTURE_WITH_TERMINATOR,
        GPT_OSS_HARMONY_INITIAL_FIXTURE_WITH_TERMINATOR,
    ] {
        let template = fixture.strip_suffix('\n').unwrap();
        prepare(
            &mut harmony_chat_tokenizer(11),
            format!("{template}\n{{# nonsemantic converter refactor #}}"),
            "harmony.channels.v1",
        );
    }

    for fixture in [
        LFM25_8B_FIXTURE_WITH_TERMINATOR,
        LFM25_VL_FIXTURE_WITH_TERMINATOR,
    ] {
        let template = fixture.strip_suffix('\n').unwrap();
        prepare(
            &mut lfm2_chat_tokenizer(13),
            format!("{template}\n{{# nonsemantic converter refactor #}}"),
            "lfm2.python-tools.v1",
        );
    }

    for (template, identity) in [
        (MISTRAL7_V03_FIXTURE, "mistral.json-list-tools.v1"),
        (
            MINISTRAL8_2410_FIXTURE,
            "mistral.json-list-tools.compact.v1",
        ),
    ] {
        prepare(
            &mut mistral_chat_tokenizer(17),
            format!("{template}\n{{# nonsemantic converter refactor #}}"),
            identity,
        );
    }

    for (template, structural_tokens, identity) in [
        (
            LLAMA31_33_FIXTURE,
            &["<|eot_id|>"][..],
            "llama.json-tools.v1",
        ),
        (LLAMA32_FIXTURE, &["<|eot_id|>"][..], "llama.json-tools.v1"),
        (
            LLAMA4_FIXTURE,
            &["<|python_start|>", "<|python_end|>", "<|eot|>"][..],
            "llama.python-channel-tools.v1",
        ),
    ] {
        prepare(
            &mut llama_chat_tokenizer(19, structural_tokens),
            format!("{template}\n{{# nonsemantic converter refactor #}}"),
            identity,
        );
    }

    let nemotron = NEMOTRON_NANO_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .unwrap();
    prepare(
        &mut llama_chat_tokenizer(23, &["<|eot_id|>"]),
        format!("{nemotron}\n{{# nonsemantic converter refactor #}}"),
        "nemotron.json-list-tools.v1",
    );
    let nemotron_v2 = NEMOTRON_NANO_V2_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .unwrap();
    prepare(
        &mut llama_chat_tokenizer(29, &["<SPECIAL_12>"]),
        format!("{nemotron_v2}\n{{# nonsemantic converter refactor #}}"),
        "nemotron.json-list-tools.reasoning.v1",
    );

    for (template, identity) in [
        (
            DEEPSEEK_V3_TOOL_FIXTURE,
            "deepseek.structural-json-tools.v1",
        ),
        (
            DEEPSEEK_V31_TOOL_FIXTURE,
            "deepseek.structural-json-tools.v2",
        ),
    ] {
        prepare(
            &mut deepseek_chat_tokenizer(31),
            format!("{template}\n{{# nonsemantic converter refactor #}}"),
            identity,
        );
    }
}

#[test]
fn behavioral_recognition_rejects_changed_wire_envelopes() {
    let changed = QWEN25_FIXTURE.replace("<tool_call>", "<tool_invoke>");
    let prepared = prepare_chat_from_parts(
        &mut production_chat_tokenizer(5),
        ModelChatTemplate::Single(changed),
        "qwen",
        &[],
        Some(&Ok(ConstraintCompiler::synthetic_for_tests())),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Use a tool."})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Auto,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(prepared.format_profile_identity(), None);
    assert!(prepared.tool_runtime_plan().is_none());
}

#[test]
fn production_mistral_json_list_templates_render_golden_tool_history_and_prompts() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let messages = vec![
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "first"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "abc123456",
                "type": "function",
                "function": {"name": "lookup", "arguments": {"value": 1}}
            }]
        }),
        json!({
            "role": "tool",
            "tool_call_id": "abc123456",
            "content": "{\"result\":1}"
        }),
        json!({"role": "assistant", "content": "done"}),
        json!({"role": "user", "content": "again"}),
    ];
    let available_tools = concat!(
        r#"[{"type": "function", "function": {"description": "Look up one integer.", "#,
        r#""name": "lookup", "parameters": {"additionalProperties":false,"properties":{"value":"#,
        r#"{"description":"The integer to look up.","type":"integer"}},"required":["value"],"#,
        r#""type":"object"}}}]"#,
    );
    let cases = [
        (
            MISTRAL7_V03_FIXTURE,
            "mistral.json-list-tools.v1",
            concat!(
                "<s>[INST] first[/INST]",
                r#"[TOOL_CALLS] [{"arguments":{"value":1},"name":"lookup", "id": "abc123456"}]</s>"#,
                r#"[TOOL_RESULTS] {"content": {"result":1}, "call_id": "abc123456"}[/TOOL_RESULTS]"#,
                " done</s>",
                "[AVAILABLE_TOOLS] ",
            ),
            "[/AVAILABLE_TOOLS][INST] Be brief.\n\nagain[/INST]",
            Some("[TOOL_CALLS] "),
            ToolChoice::Auto,
        ),
        (
            MINISTRAL8_2410_FIXTURE,
            "mistral.json-list-tools.compact.v1",
            concat!(
                "<s>[INST]first[/INST]",
                r#"[TOOL_CALLS][{"arguments":{"value":1},"name":"lookup", "id": "abc123456"}]</s>"#,
                r#"[TOOL_RESULTS]{"content": {"result":1}, "call_id": "abc123456"}[/TOOL_RESULTS]"#,
                "done</s>",
                "[AVAILABLE_TOOLS]",
            ),
            "[/AVAILABLE_TOOLS][INST]Be brief.\n\nagain[/INST]",
            None,
            ToolChoice::Required,
        ),
    ];

    for (
        template,
        identity,
        expected_prefix,
        expected_suffix,
        expected_auto_trigger,
        tool_choice,
    ) in cases
    {
        for add_generation_prompt in [false, true] {
            let mut tokenizer = mistral_chat_tokenizer(4);
            let prepared = prepare_chat_from_parts(
                &mut tokenizer,
                ModelChatTemplate::Single(template.into()),
                "architecture-name-is-not-a-support-key",
                &[5],
                Some(&compiler),
                ChatTemplateRequest {
                    messages: messages.clone(),
                    tools: vec![production_tool("lookup")],
                    tool_choice,
                    parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                        max_calls: std::num::NonZeroUsize::new(2),
                    },
                    add_generation_prompt,
                    ..ChatTemplateRequest::default()
                },
            )
            .unwrap();

            assert_eq!(prepared.format_profile_identity(), Some(identity));
            assert_eq!(prepared.generation_prompt(), "");
            assert_eq!(
                prepared.rendered_prompt(),
                format!("{expected_prefix}{available_tools}{expected_suffix}")
            );
            assert_eq!(prepared.preserved_structural_token_ids(), &[4, 5]);
            assert_eq!(prepared.profile_stop_sequences(), ["</s>"]);
            let plan = prepared
                .tool_runtime_plan()
                .unwrap_or_else(|| panic!("registered Mistral profile must be supported"));
            assert_eq!(plan.auto_activation_trigger(), expected_auto_trigger);
        }
    }
}

#[test]
fn production_meta_llama_templates_render_golden_tool_history_and_prompts() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let messages = vec![
        json!({"role": "system", "content": "Be brief."}),
        json!({"role": "user", "content": "first"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {"name": "lookup", "arguments": {"value": 1}}
            }]
        }),
        json!({"role": "tool", "content": "{\"result\":\"Bogotá\"}"}),
        json!({"role": "user", "content": "again"}),
    ];
    let available_tool = concat!(
        "{\n",
        "    \"function\": {\n",
        "        \"description\": \"Look up one integer.\",\n",
        "        \"name\": \"lookup\",\n",
        "        \"parameters\": {\n",
        "            \"additionalProperties\": false,\n",
        "            \"properties\": {\n",
        "                \"value\": {\n",
        "                    \"description\": \"The integer to look up.\",\n",
        "                    \"type\": \"integer\"\n",
        "                }\n",
        "            },\n",
        "            \"required\": [\n",
        "                \"value\"\n",
        "            ],\n",
        "            \"type\": \"object\"\n",
        "        }\n",
        "    },\n",
        "    \"type\": \"function\"\n",
        "}",
    );
    let llama3_golden = format!(
        concat!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "Environment: ipython\nCutting Knowledge Date: December 2023\n",
            "Today Date: 27 Jul 2026\n\nBe brief.<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            "Given the following functions, please respond with a JSON for a function call ",
            "with its proper arguments that best answers the given prompt.\n\n",
            "Respond in the format {{\"name\": function name, \"parameters\": dictionary of ",
            "argument name and its value}}.Do not use variables.\n\n",
            "{}\n\nfirst<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "{{\"name\": \"lookup\", \"parameters\": {{\"value\":1}}}}<|eot_id|>",
            "<|start_header_id|>ipython<|end_header_id|>\n\n",
            "\"{{\\\"result\\\":\\\"Bogotá\\\"}}\"<|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\nagain<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ),
        available_tool
    );
    for (template, identity) in [
        (LLAMA31_33_FIXTURE, "llama.json-tools.v1"),
        (LLAMA32_FIXTURE, "llama.json-tools.v1"),
    ] {
        let mut tokenizer = llama_chat_tokenizer(7, &["<|eot_id|>"]);
        let prepared = prepare_chat_from_parts(
            &mut tokenizer,
            ModelChatTemplate::Single(template.into()),
            "unrelated",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: messages.clone(),
                tools: vec![production_tool("lookup")],
                tool_choice: ToolChoice::Auto,
                parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                    max_calls: std::num::NonZeroUsize::new(2),
                },
                add_generation_prompt: true,
                extra_template_kwargs: serde_json::Map::from_iter([(
                    "date_string".into(),
                    json!("27 Jul 2026"),
                )]),
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();
        assert_eq!(prepared.format_profile_identity(), Some(identity));
        assert_eq!(prepared.rendered_prompt(), llama3_golden);
        assert_eq!(
            prepared.generation_prompt(),
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        );
        assert_eq!(prepared.preserved_structural_token_ids(), &[7]);
        assert_eq!(prepared.profile_stop_sequences(), ["<|eot_id|>"]);
        let plan = prepared
            .tool_runtime_plan()
            .unwrap_or_else(|| panic!("registered Llama 3 profile must be supported"));
        assert_eq!(plan.auto_activation_trigger(), Some("{"));
        let output = r#"{"name":"lookup","parameters":{"value":1}}"#;
        assert!(plan_accepts(plan, output));
        assert!(!plan_accepts(
            plan,
            concat!(
                r#"{"name":"lookup","parameters":{"value":1}}"#,
                r#"{"name":"lookup","parameters":{"value":2}}"#
            )
        ));
        assert!(!plan_accepts(
            plan,
            r#"{"name":"missing","parameters":{"value":1}}"#
        ));
        assert!(!plan_accepts(
            plan,
            r#"{"name":"lookup","parameters":{"value":"one"}}"#
        ));

        let mut parser = plan.create_parser().unwrap();
        parser.push(&format!("{output}<|eot_id|>ignored")).unwrap();
        assert_eq!(tool_argument_events(parser.events()), [r#"{"value":1}"#]);
        assert!(parser.events().contains(&SemanticEvent::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "lookup".into(),
        }));
        assert_eq!(
            parser.events().last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::StopSequence,
            })
        );
        for split in 0..=output.len() {
            let mut parser = plan.create_parser().unwrap();
            parser.push(&output[..split]).unwrap();
            parser
                .push(&format!("{}<|eot_id|>", &output[split..]))
                .unwrap();
            assert_eq!(
                tool_argument_events(parser.events()),
                [r#"{"value":1}"#],
                "split {split}"
            );
        }
        for incomplete in [
            "{",
            r#"{"name":"lookup""#,
            r#"{"name":"lookup","parameters":{"value":1}"#,
        ] {
            let mut parser = plan.create_parser().unwrap();
            parser.push(incomplete).unwrap();
            parser.finish(FinishReason::MaxTokens).unwrap();
            assert!(!parser
                .events()
                .iter()
                .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
        }
        let mut malformed = plan.create_parser().unwrap();
        assert!(malformed
            .push(r#"{"name":"lookup","parameters":{"value":]}}"#)
            .is_err());
    }

    let mut tokenizer =
        llama_chat_tokenizer(11, &["<|python_start|>", "<|python_end|>", "<|eot|>"]);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(LLAMA4_FIXTURE.into()),
        "unrelated",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: messages.clone(),
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                max_calls: std::num::NonZeroUsize::new(2),
            },
            add_generation_prompt: true,
            extra_template_kwargs: serde_json::Map::from_iter([(
                "date_string".into(),
                json!("27 Jul 2026"),
            )]),
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    let llama4_golden = format!(
        concat!(
            "<|begin_of_text|><|header_start|>system<|header_end|>\n\n",
            "Environment: ipython\nBe brief.<|eot|>",
            "<|header_start|>user<|header_end|>\n\n",
            "Given the following functions, please respond with a JSON for a function call ",
            "with its proper arguments that best answers the given prompt.\n\n",
            "Respond in the format {{\"name\": function name, \"parameters\": dictionary of ",
            "argument name and its value}}.Do not use variables.\n\n",
            "{}\n\nfirst<|eot|>",
            "<|header_start|>assistant<|header_end|>\n\n",
            "<|python_start|><|python_end|>",
            "{{\"name\": \"lookup\", \"parameters\": {{\"value\":1}}}}<|eot|>",
            "<|header_start|>ipython<|header_end|>\n\n",
            "\"{{\\\"result\\\":\\\"Bogotá\\\"}}\"<|eot|>",
            "<|header_start|>user<|header_end|>\n\nagain<|eot|>",
            "<|header_start|>assistant<|header_end|>\n\n"
        ),
        available_tool
    );
    assert_eq!(
        prepared.format_profile_identity(),
        Some("llama.python-channel-tools.v1")
    );
    assert_eq!(prepared.rendered_prompt(), llama4_golden);
    assert_eq!(
        prepared.generation_prompt(),
        "<|header_start|>assistant<|header_end|>\n\n"
    );
    assert_eq!(prepared.preserved_structural_token_ids(), &[11, 12, 13]);
    assert_eq!(prepared.profile_stop_sequences(), ["<|eot|>"]);
    let plan = prepared
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered Llama 4 profile must be supported"));
    assert_eq!(plan.auto_activation_trigger(), Some("<|python_start|>"));
    let output = concat!(
        "<|python_start|>analysis 🦀<|python_end|>",
        r#"{"name":"lookup","parameters":{"value":1}}"#,
        r#"{"name":"lookup","parameters":{"value":2}}"#,
    );
    assert!(plan_accepts(plan, output));
    assert!(!plan_accepts(
        plan,
        concat!(
            "<|python_start|><|python_end|>",
            r#"{"name":"missing","parameters":{"value":1}}"#
        )
    ));
    let mut parser = plan.create_parser().unwrap();
    parser.push(&format!("{output}<|eot|>ignored")).unwrap();
    assert_eq!(
        tool_argument_events(parser.events()),
        [r#"{"value":1}"#, r#"{"value":2}"#]
    );
    assert!(parser
        .events()
        .contains(&SemanticEvent::TextDelta("analysis 🦀".into())));
    assert!(parser.events().contains(&SemanticEvent::ToolCallStart {
        index: 1,
        id: "call_1".into(),
        name: "lookup".into(),
    }));
    assert_eq!(
        parser.events().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence,
        })
    );
    for split in (0..=output.len()).filter(|index| output.is_char_boundary(*index)) {
        let mut parser = plan.create_parser().unwrap();
        parser.push(&output[..split]).unwrap();
        parser
            .push(&format!("{}<|eot|>", &output[split..]))
            .unwrap();
        assert_eq!(
            tool_argument_events(parser.events()),
            [r#"{"value":1}"#, r#"{"value":2}"#],
            "split {split}"
        );
    }
    let mut incomplete = plan.create_parser().unwrap();
    incomplete
        .push("<|python_start|>analysis<|python_end|>{\"name\":\"lookup\"")
        .unwrap();
    incomplete.finish(FinishReason::MaxTokens).unwrap();
    assert!(!incomplete
        .events()
        .iter()
        .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
}

#[test]
fn production_nemotron_renders_golden_parallel_history_and_prompt() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let template = NEMOTRON_NANO_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .unwrap();
    let mut tokenizer = llama_chat_tokenizer(5, &["<|eot_id|>"]);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "unrelated",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![
                json!({"role": "system", "content": "Be brief."}),
                json!({"role": "user", "content": "first"}),
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"value": 1}}
                        },
                        {
                            "type": "function",
                            "function": {"name": "lookup", "arguments": {"value": 2}}
                        }
                    ]
                }),
                json!({"role": "tool", "content": "{\"result\":\"Bogotá\"}"}),
            ],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                max_calls: std::num::NonZeroUsize::new(2),
            },
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        prepared.rendered_prompt(),
        concat!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
            "Be brief.\n\n<AVAILABLE_TOOLS>[",
            r#"{"description":"Look up one integer.","name":"lookup","parameters":{"additionalProperties":false,"properties":{"value":{"description":"The integer to look up.","type":"integer"}},"required":["value"],"type":"object"}}"#,
            "]</AVAILABLE_TOOLS><|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\nfirst<|eot_id|>",
            "<|start_header_id|>assistant<|end_header_id|>\n\n<TOOLCALL>[",
            r#"{"name": "lookup", "arguments": {"value":1}}, {"name": "lookup", "arguments": {"value":2}}"#,
            "]</TOOLCALL><|eot_id|>",
            "<|start_header_id|>user<|end_header_id|>\n\n",
            r#"<TOOL_RESPONSE>[{"result":"Bogotá"}]</TOOL_RESPONSE>"#,
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    );
    assert_eq!(
        prepared.generation_prompt(),
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    );
    assert_eq!(
        prepared.format_profile_identity(),
        Some("nemotron.json-list-tools.v1")
    );
    assert_eq!(prepared.preserved_structural_token_ids(), &[5]);
    assert_eq!(prepared.profile_stop_sequences(), ["<|eot_id|>"]);
    let plan = prepared
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered Nemotron profile must be supported"));
    assert_eq!(plan.auto_activation_trigger(), None);

    let output = concat!(
        "<TOOLCALL>[",
        r#"{"name":"lookup","arguments":{"value":1}}, "#,
        r#"{"name":"lookup","arguments":{"value":2}}"#,
        "]</TOOLCALL>"
    );
    assert!(plan_accepts(plan, output));
    assert!(!plan_accepts(
        plan,
        r#"<TOOLCALL>[{"name":"missing","arguments":{"value":1}}]</TOOLCALL>"#
    ));
    assert!(!plan_accepts(
        plan,
        r#"<TOOLCALL>[{"name":"lookup","arguments":{"value":"one"}}]</TOOLCALL>"#
    ));

    let mut parser = plan.create_parser().unwrap();
    parser.push(&format!("{output}<|eot_id|>ignored")).unwrap();
    assert_eq!(
        tool_argument_events(parser.events()),
        [r#"{"value":1}"#, r#"{"value":2}"#]
    );
    assert!(parser.events().contains(&SemanticEvent::ToolCallStart {
        index: 1,
        id: "call_1".into(),
        name: "lookup".into(),
    }));
    assert_eq!(
        parser.events().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence,
        })
    );
    for split in 0..=output.len() {
        let mut parser = plan.create_parser().unwrap();
        parser.push(&output[..split]).unwrap();
        parser
            .push(&format!("{}<|eot_id|>", &output[split..]))
            .unwrap();
        assert_eq!(
            tool_argument_events(parser.events()),
            [r#"{"value":1}"#, r#"{"value":2}"#],
            "split {split}"
        );
    }
    let mut incomplete = plan.create_parser().unwrap();
    incomplete
        .push(r#"<TOOLCALL>[{"name":"lookup","arguments":{"value":1}}"#)
        .unwrap();
    incomplete.finish(FinishReason::MaxTokens).unwrap();
    assert!(!incomplete
        .events()
        .iter()
        .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
    let mut malformed = plan.create_parser().unwrap();
    assert!(malformed
        .push(r#"<TOOLCALL>[{"name":"lookup","arguments":{"value":]}}]</TOOLCALL>"#)
        .is_err());
}

#[test]
fn production_nemotron_v2_covers_reasoning_constraints_and_streaming() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let template = NEMOTRON_NANO_V2_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    let tool = json!({
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up text.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": false
            }
        }
    });
    let messages = vec![
        json!({"role": "system", "content": "Be brief. /think"}),
        json!({"role": "user", "content": "first"}),
        json!({
            "role": "assistant",
            "content": "<think>\nstale\n</think>\n\nworking",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": {"query": "Bogotá"}
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": r#"{"query":"a \"quote\""}"#
                    }
                }
            ]
        }),
        json!({"role": "tool", "content": "{\"result\":\"uno\"}"}),
        json!({"role": "tool", "content": "{\"result\":\"dos\"}"}),
        json!({"role": "user", "content": "again"}),
    ];
    let mut tokenizer = llama_chat_tokenizer(19, &["<SPECIAL_12>"]);
    let required = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "unrelated-architecture-name",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages,
            tools: vec![tool.clone()],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                max_calls: std::num::NonZeroUsize::new(2),
            },
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
            required.rendered_prompt(),
            concat!(
                "<SPECIAL_10>System\nBe brief.\n\n",
                "You can use the following tools to assist the user if required:\n",
                "<AVAILABLE_TOOLS>[",
                r#"{"description":"Look up text.","name":"lookup","parameters":{"additionalProperties":false,"properties":{"query":{"type":"string"}},"required":["query"],"type":"object"}}"#,
                "]</AVAILABLE_TOOLS>\n\n",
                "If you decide to call any tool(s), use the following format:\n",
                r#"<TOOLCALL>[{{"name": "tool_name1", "arguments": "tool_args1"}}, {{"name": "tool_name2", "arguments": "tool_args2"}}]</TOOLCALL>"#,
                "\n\nThe user will execute tool-calls and return responses from tool(s) in this format:\n",
                r#"<TOOL_RESPONSE>[{{"tool_response1"}}, {{"tool_response2"}}]</TOOL_RESPONSE>"#,
                "\n\nBased on the tool responses, you can call additional tools if needed, ",
                "correct tool calls if any errors are found, or just respond to the user.\n",
                "<SPECIAL_11>User\nfirst\n",
                "<SPECIAL_11>Assistant\nworking\n\n<TOOLCALL>[",
                r#"{"name": "lookup", "arguments": {"query":"Bogotá"}}, "#,
                r#"{"name": "lookup", "arguments": {"query":"a \"quote\""}}"#,
                "]</TOOLCALL>\n<SPECIAL_12>\n",
                "<SPECIAL_11>User\n<TOOL_RESPONSE>[",
                r#"{"result":"uno"}, {"result":"dos"}"#,
                "]</TOOL_RESPONSE>\n",
                "<SPECIAL_11>User\nagain\n",
                "<SPECIAL_11>Assistant\n<think>\n",
            )
        );
    assert_eq!(
        required.generation_prompt(),
        "<SPECIAL_11>Assistant\n<think>\n"
    );
    assert_eq!(
        required.format_profile_identity(),
        Some("nemotron.json-list-tools.reasoning.v1")
    );
    assert_eq!(required.preserved_structural_token_ids(), &[19]);
    assert_eq!(required.profile_stop_sequences(), ["<SPECIAL_12>"]);
    let required_plan = required
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered Nemotron v2 profile must be supported"));
    assert_eq!(required_plan.auto_activation_trigger(), None);

    let output = concat!(
        "checking Bogotá 🦀\n</think>\n\n<TOOLCALL>[",
        r#"{"name":"lookup","arguments":{"query":"Bogotá 🦀 \"quoted\" \\ path\n東京"}}, "#,
        r#"{"name":"lookup","arguments":{"query":"second"}}"#,
        "]</TOOLCALL>",
    );
    assert!(plan_accepts(required_plan, output));
    assert!(!plan_accepts(
        required_plan,
        r#"<TOOLCALL>[{"name":"lookup","arguments":{"query":7}}]</TOOLCALL>"#
    ));
    assert!(!plan_accepts(
        required_plan,
        r#"<TOOLCALL>[{"name":"missing","arguments":{"query":"x"}}]</TOOLCALL>"#
    ));

    let mut parser = required_plan.create_parser().unwrap();
    parser
        .push(&format!("{output}<SPECIAL_12>ignored"))
        .unwrap();
    assert!(parser
        .events()
        .contains(&SemanticEvent::ReasoningDelta("checking Bogotá 🦀".into())));
    assert_eq!(
        tool_argument_events(parser.events()),
        [
            r#"{"query":"Bogotá 🦀 \"quoted\" \\ path\n東京"}"#,
            r#"{"query":"second"}"#,
        ]
    );
    assert_eq!(
        parser.events().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence,
        })
    );
    for split in (0..=output.len()).filter(|index| output.is_char_boundary(*index)) {
        let mut parser = required_plan.create_parser().unwrap();
        parser.push(&output[..split]).unwrap();
        parser
            .push(&format!("{}<SPECIAL_12>", &output[split..]))
            .unwrap();
        assert_eq!(
            tool_argument_events(parser.events()).len(),
            2,
            "split {split}"
        );
    }

    let mut incomplete = required_plan.create_parser().unwrap();
    incomplete
        .push("reasoning\n</think>\n\n<TOOLCALL>[{\"name\":\"lookup\"")
        .unwrap();
    incomplete.finish(FinishReason::MaxTokens).unwrap();
    assert!(!incomplete
        .events()
        .iter()
        .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
    let mut malformed = required_plan.create_parser().unwrap();
    assert!(malformed
        .push("reasoning\n</think>\n\n<TOOLCALL>[{\"name\":\"lookup\",\"arguments\":{\"query\":]}}")
        .is_err());

    let mut tokenizer = llama_chat_tokenizer(23, &["<SPECIAL_12>"]);
    let auto = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "nemotron_h",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![
                json!({"role": "system", "content": "/no_think"}),
                json!({"role": "user", "content": "lookup"}),
            ],
            tools: vec![tool],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: ParallelToolCallPolicy::Disabled,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert!(auto
        .rendered_prompt()
        .ends_with("<SPECIAL_11>Assistant\n<think></think>"));
    let auto_plan = auto
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered Nemotron v2 Auto profile must be supported"));
    assert_eq!(auto_plan.auto_activation_trigger(), Some("<TOOLCALL>["));
    assert!(plan_accepts(
        auto_plan,
        r#"<TOOLCALL>[{"name":"lookup","arguments":{"query":"one"}}]</TOOLCALL>"#
    ));
    assert!(!plan_accepts(
        auto_plan,
        concat!(
            r#"<TOOLCALL>[{"name":"lookup","arguments":{"query":"one"}}, "#,
            r#"{"name":"lookup","arguments":{"query":"two"}}]</TOOLCALL>"#
        )
    ));
}

#[test]
fn llama_auto_and_required_activation_are_exact_without_family_fallback() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let nemotron = NEMOTRON_NANO_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .unwrap();
    for (template, structural_tokens, expected_auto_trigger) in [
        (LLAMA31_33_FIXTURE, &["<|eot_id|>"][..], "{"),
        (LLAMA32_FIXTURE, &["<|eot_id|>"][..], "{"),
        (
            LLAMA4_FIXTURE,
            &["<|python_start|>", "<|python_end|>", "<|eot|>"][..],
            "<|python_start|>",
        ),
        (nemotron, &["<|eot_id|>"][..], "<TOOLCALL>["),
    ] {
        for (tool_choice, expected_trigger) in [
            (ToolChoice::Auto, Some(expected_auto_trigger)),
            (ToolChoice::Required, None),
        ] {
            let mut tokenizer = llama_chat_tokenizer(17, structural_tokens);
            let prepared = prepare_chat_from_parts(
                &mut tokenizer,
                ModelChatTemplate::Single(template.into()),
                "llama-model-family-must-not-select-a-profile",
                &[],
                Some(&compiler),
                ChatTemplateRequest {
                    messages: vec![json!({"role": "user", "content": "call lookup"})],
                    tools: vec![production_tool("lookup")],
                    tool_choice,
                    add_generation_prompt: true,
                    ..ChatTemplateRequest::default()
                },
            )
            .unwrap();
            let plan = prepared
                .tool_runtime_plan()
                .unwrap_or_else(|| panic!("exact registered fixture must be supported"));
            assert_eq!(plan.auto_activation_trigger(), expected_trigger);
        }
    }

    let mut tokenizer = llama_chat_tokenizer(2, &["<|eot_id|>"]);
    let unsupported = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(
            "{{ bos_token }} generic Llama template without a tool protocol".into(),
        ),
        "llama",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(unsupported.format_profile_identity(), None);
    assert!(matches!(
        unsupported.native_tool_support(),
        NativeToolSupport::Unsupported { reason }
            if reason.contains("no behavioral format recognizer")
    ));
}

#[test]
fn production_gpt_oss_template_renders_harmony_history_and_runtime_profile() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = harmony_chat_tokenizer(31);
    let template = GPT_OSS_HARMONY_CURRENT_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    let messages = vec![
        json!({"role": "system", "content": "Be exact."}),
        json!({"role": "user", "content": "Look it up."}),
        json!({
            "role": "assistant",
            "thinking": "Need the lookup result.",
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "arguments": {"value": 7}
                }
            }]
        }),
        json!({"role": "tool", "content": "{\"result\":\"Bogotá\"}"}),
        json!({"role": "user", "content": "Now summarize."}),
    ];
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "architecture-metadata-must-not-select-harmony",
        &[36],
        Some(&compiler),
        ChatTemplateRequest {
            messages,
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            extra_template_kwargs: serde_json::Map::from_iter([
                ("model_identity".into(), json!("Fixture identity.")),
                ("reasoning_effort".into(), json!("low")),
            ]),
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert!(prepared.rendered_prompt().starts_with(
        "<|start|>system<|message|>Fixture identity.\nKnowledge cutoff: 2024-06\nCurrent date: "
    ));
    assert!(prepared.rendered_prompt().contains(concat!(
            "\n\nReasoning: low\n\n",
            "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n",
            "Calls to these tools must go to the commentary channel: 'functions'.<|end|>",
            "<|start|>developer<|message|># Instructions\n\nBe exact.\n\n# Tools\n\n"
        )));
    assert!(prepared.rendered_prompt().contains(concat!(
        "<|start|>user<|message|>Look it up.<|end|>",
        "<|start|>assistant<|channel|>analysis<|message|>Need the lookup result.<|end|>",
        "<|start|>assistant to=functions.lookup<|channel|>commentary json",
        "<|message|>{\"value\":7}<|call|>",
        "<|start|>functions.lookup to=assistant<|channel|>commentary<|message|>",
        "\"{\\\"result\\\":\\\"Bogotá\\\"}\"<|end|>",
        "<|start|>user<|message|>Now summarize.<|end|>",
        "<|start|>assistant"
    )));
    assert_eq!(prepared.generation_prompt(), "<|start|>assistant");
    assert_eq!(
        prepared.format_profile_identity(),
        Some("harmony.channels.v1")
    );
    assert_eq!(
        prepared.preserved_structural_token_ids(),
        &[31, 32, 33, 34, 35, 36, 37]
    );
    assert_eq!(
        prepared.profile_stop_sequences(),
        ["<|return|>", "<|call|>"]
    );
    let plan = prepared
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("recognized Harmony protocol must prepare Harmony"));
    assert_eq!(plan.auto_activation_trigger(), None);
}

#[test]
fn production_lfm2_templates_render_tools_prior_calls_and_results() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let tools = vec![production_tool("lookup")];

    let mut current_tokenizer = lfm2_chat_tokenizer(41);
    let current_template = LFM25_8B_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    inspect_chat_template_kwargs(current_template, "lfm2.5-8b").unwrap();
    let vl_template = LFM25_VL_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    inspect_chat_template_kwargs(vl_template, "lfm2.5-vl").unwrap();
    let current = prepare_chat_from_parts(
        &mut current_tokenizer,
        ModelChatTemplate::Single(current_template.into()),
        "architecture-metadata-must-not-select-lfm2",
        &[43],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![
                json!({"role": "system", "content": "Be exact."}),
                json!({"role": "user", "content": "Look it up."}),
                json!({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": {"value": 7}
                        }
                    }]
                }),
                json!({"role": "tool", "content": "{\"result\":\"Bogotá\"}"}),
            ],
            tools: tools.clone(),
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert!(current.rendered_prompt().contains("List of tools: ["));
    assert!(current.rendered_prompt().contains("\"name\":\"lookup\""));
    assert!(current.rendered_prompt().contains(concat!(
        "<|im_start|>assistant\n",
        "<|tool_call_start|>[lookup(value=7)]<|tool_call_end|>",
        "<|im_end|>\n",
        "<|im_start|>tool\n",
        "{\"result\":\"Bogotá\"}<|im_end|>\n",
        "<|im_start|>assistant\n"
    )));
    assert_eq!(
        current.format_profile_identity(),
        Some("lfm2.python-tools.v1")
    );
    assert_eq!(current.preserved_structural_token_ids(), &[41, 42, 43]);
    assert_eq!(
        current.profile_stop_sequences(),
        ["<|tool_call_end|>", "<|im_end|>"]
    );
    let plan = current
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("recognized LFM2 protocol must prepare Python tools"));
    assert_eq!(plan.auto_activation_trigger(), None);

    let mut classic_tokenizer = lfm2_chat_tokenizer(51);
    let classic_template = LFM2_CLASSIC_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    let classic = prepare_chat_from_parts(
        &mut classic_tokenizer,
        ModelChatTemplate::Single(classic_template.into()),
        "architecture-metadata-must-not-select-lfm2",
        &[53],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![
                json!({"role": "user", "content": "Look it up."}),
                json!({
                    "role": "assistant",
                    "content": "<|tool_call_start|>[lookup(value=7)]<|tool_call_end|>"
                }),
                json!({"role": "tool", "content": "{\"result\":\"Bogotá\"}"}),
            ],
            tools,
            tool_choice: ToolChoice::Auto,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert!(classic
        .rendered_prompt()
        .contains("List of tools: <|tool_list_start|>["));
    assert!(classic.rendered_prompt().contains(concat!(
        "<|im_start|>assistant\n",
        "<|tool_call_start|>[lookup(value=7)]<|tool_call_end|><|im_end|>\n",
        "<|im_start|>tool\n",
        "<|tool_response_start|>{\"result\":\"Bogotá\"}<|tool_response_end|>",
        "<|im_end|>\n",
        "<|im_start|>assistant\n"
    )));
    assert_eq!(classic.format_profile_identity(), None);
    assert!(classic.tool_runtime_plan().is_none());
    assert!(matches!(
        classic.native_tool_support(),
        NativeToolSupport::Unsupported { .. }
    ));
}

#[test]
fn production_deepseek_templates_render_tools_history_and_exact_generation_prompts() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let tools = vec![production_tool("lookup")];
    let messages = vec![
        json!({"role": "system", "content": "Be exact."}),
        json!({"role": "user", "content": "Look it up."}),
        json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "type": "function",
                "function": {"name": "lookup", "arguments": {"value": 7}}
            }]
        }),
        json!({"role": "tool", "content": "{\"result\":\"Bogotá\"}"}),
    ];

    for (template, identity, preceding_tokens, golden_prompt_signature) in [
        (
            DEEPSEEK_V3_TOOL_FIXTURE,
            "deepseek.structural-json-tools.v1",
            31,
            [
                0xa0, 0x3b, 0xd6, 0x32, 0x50, 0xe4, 0xd0, 0x15, 0x32, 0x7d, 0xc8, 0x54, 0x68, 0x6b,
                0x34, 0x89, 0xcd, 0x1f, 0x3b, 0x38, 0x29, 0x57, 0xdd, 0x92, 0x99, 0xb7, 0x7f, 0x00,
                0xe7, 0xdd, 0x21, 0x3c,
            ],
        ),
        (
            DEEPSEEK_V31_TOOL_FIXTURE,
            "deepseek.structural-json-tools.v2",
            41,
            [
                0x76, 0x4c, 0x69, 0x2b, 0x69, 0x47, 0xc4, 0xcc, 0x9a, 0x15, 0xf8, 0x14, 0xc1, 0xc9,
                0x66, 0x18, 0xff, 0x91, 0x71, 0x29, 0xea, 0xdf, 0x7d, 0xf2, 0x2d, 0x95, 0xc7, 0x40,
                0x64, 0xee, 0xd5, 0x85,
            ],
        ),
    ] {
        let mut tokenizer = deepseek_chat_tokenizer(preceding_tokens);
        let prepared = prepare_chat_from_parts(
            &mut tokenizer,
            ModelChatTemplate::Single(template.into()),
            "deepseek architecture metadata is not a support key",
            &[99],
            Some(&compiler),
            ChatTemplateRequest {
                messages: messages.clone(),
                tools: tools.clone(),
                tool_choice: ToolChoice::Required,
                enable_thinking: Some(false),
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();
        assert_eq!(
            crate::runtime::chat::template_signature(prepared.rendered_prompt()),
            golden_prompt_signature,
            "exact rendered history golden for {identity}"
        );
        assert_eq!(prepared.format_profile_identity(), Some(identity));
        assert_eq!(
            prepared.preserved_structural_token_ids(),
            &[
                preceding_tokens as u32,
                preceding_tokens as u32 + 1,
                preceding_tokens as u32 + 2,
                preceding_tokens as u32 + 3,
                preceding_tokens as u32 + 4,
                preceding_tokens as u32 + 5,
            ]
        );
        assert_eq!(prepared.profile_stop_sequences(), ["<｜end▁of▁sentence｜>"]);
        assert!(prepared.rendered_prompt().contains("lookup"));
        assert!(prepared.rendered_prompt().contains("{\"value\":7}"));
        assert!(prepared.rendered_prompt().contains("Bogotá"));
    }

    for (template, identity, expected_generation_prompt, golden_prompt_signature) in [
        (
            DEEPSEEK_V3_TOOL_FIXTURE,
            "deepseek.structural-json-tools.v1",
            "",
            [
                0x46, 0xf0, 0xed, 0x06, 0x01, 0x25, 0x07, 0x5c, 0x56, 0x06, 0x9a, 0x84, 0x3b, 0x8b,
                0xd7, 0x35, 0x8f, 0x6e, 0xb9, 0xa4, 0x45, 0x2f, 0xb3, 0x93, 0x98, 0x5d, 0x4e, 0xad,
                0xba, 0xab, 0xf9, 0x56,
            ],
        ),
        (
            DEEPSEEK_V31_TOOL_FIXTURE,
            "deepseek.structural-json-tools.v2",
            "\n  <｜Assistant｜>\n    </think>\n",
            [
                0x74, 0x80, 0x75, 0x7f, 0x81, 0xa5, 0x3b, 0xae, 0x1f, 0x51, 0x44, 0x51, 0xe3, 0x90,
                0xb1, 0x39, 0x72, 0xb8, 0x44, 0xa3, 0xb1, 0x68, 0xf3, 0xf8, 0x21, 0x1a, 0xbf, 0x85,
                0xc0, 0xe4, 0x75, 0xf8,
            ],
        ),
    ] {
        let mut tokenizer = deepseek_chat_tokenizer(51);
        let prepared = prepare_chat_from_parts(
            &mut tokenizer,
            ModelChatTemplate::Single(template.into()),
            "unrelated",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: vec![json!({"role": "user", "content": "Look it up."})],
                tools: tools.clone(),
                tool_choice: ToolChoice::Auto,
                enable_thinking: Some(false),
                add_generation_prompt: true,
                extra_template_kwargs: serde_json::Map::from_iter([
                    ("enable_thinking".into(), json!(true)),
                    ("thinking".into(), json!(true)),
                ]),
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();
        assert_eq!(
            crate::runtime::chat::template_signature(prepared.rendered_prompt()),
            golden_prompt_signature,
            "exact fresh prompt golden for {identity}"
        );
        assert_eq!(prepared.generation_prompt(), expected_generation_prompt);
    }

    for template in [DEEPSEEK_V3_TOOL_FIXTURE, DEEPSEEK_V31_TOOL_FIXTURE] {
        let error = prepare_chat_from_parts(
            &mut deepseek_chat_tokenizer(61),
            ModelChatTemplate::Single(template.into()),
            "unrelated",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: vec![json!({"role": "user", "content": "Look it up."})],
                tools: tools.clone(),
                tool_choice: ToolChoice::Required,
                enable_thinking: Some(true),
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, TextModelError::ToolConstraint(_)));
    }
}

#[test]
fn inkling_reasoning_toggle_maps_to_named_effort() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    for (enabled, expected) in [(false, "0"), (true, "0.9")] {
        let mut tokenizer = inkling_chat_tokenizer(7);
        let prepared = prepare_chat_from_parts(
            &mut tokenizer,
            ModelChatTemplate::Single(INKLING_SMALL_FIXTURE.into()),
            "unrelated-model-id",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: vec![json!({"role": "user", "content": "hello"})],
                enable_thinking: Some(enabled),
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();
        assert!(prepared.rendered_prompt().contains(&format!(
            "<|content_text|>Thinking effort level: {expected}<|end_message|>"
        )));
    }
}

#[test]
fn inkling_recognition_is_behavioral_and_exposes_native_tools() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let refactored = format!("{{# source-only refactor #}}{INKLING_SMALL_FIXTURE}");
    let mut tokenizer = inkling_chat_tokenizer(11);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(refactored),
        "unrelated-model-id",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        prepared.format_profile_identity(),
        Some("inkling.messages.v1")
    );
    assert!(matches!(
        prepared.native_tool_support(),
        NativeToolSupport::Supported
    ));
    assert!(prepared.tool_runtime_plan().is_none());

    let marker_soup = concat!(
        "<|message_model|><|content_text|>",
        "{{ messages[0]['content'] }}<|end_message|>",
        "<|content_thinking|><|content_model_end_sampling|>"
    );
    let mut tokenizer = inkling_chat_tokenizer(17);
    let unsupported = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(marker_soup.into()),
        "inkling_mm_model",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(unsupported.format_profile_identity(), None);
    assert!(matches!(
        unsupported.semantic_support(),
        crate::runtime::chat::SemanticSupport::Unsupported { .. }
    ));
}

#[test]
fn inkling_native_tools_render_constrain_and_parse_protocol() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let tool = production_tool("lookup");
    let messages = vec![
        json!({"role": "user", "content": "Look up the first value."}),
        json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "inspect",
            "tool_calls": [{
                "id": "call_previous",
                "type": "function",
                "function": {"name": "lookup", "arguments": {"value": 3}}
            }]
        }),
        json!({
            "role": "tool",
            "name": "lookup",
            "tool_call_id": "call_previous",
            "content": "{\"result\":3}"
        }),
        json!({"role": "user", "content": "Now look up two more."}),
    ];
    let mut tokenizer = inkling_chat_tokenizer(29);
    let required = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(INKLING_SMALL_FIXTURE.into()),
        "unrelated-model-id",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages,
            tools: vec![tool.clone()],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: ParallelToolCallPolicy::Enabled {
                max_calls: std::num::NonZeroUsize::new(2),
            },
            enable_thinking: Some(true),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert!(required.rendered_prompt().contains(concat!(
        "<|message_system|>tool_declare<|content_xml|>",
        "[{\"description\":\"Look up one integer.\",\"name\":\"lookup\",\"parameters\":"
    )));
    assert!(required.rendered_prompt().contains(concat!(
        "<|message_model|>lookup<|content_invoke_tool_json|>",
        "{\"name\":\"lookup\",\"args\":{\"value\":3}}<|end_message|>"
    )));
    assert!(required.rendered_prompt().contains(concat!(
        "<|message_tool|>lookup<|content_text|>",
        "{\"result\":3}<|end_message|>"
    )));
    assert!(required.rendered_prompt().ends_with("<|message_model|>"));
    assert_eq!(required.preserved_structural_token_ids().len(), 6);

    let required_plan = required
        .tool_runtime_plan()
        .expect("recognized Inkling tools must compile a runtime plan");
    assert_eq!(required_plan.auto_activation_trigger(), None);
    let call = |value| {
        format!(
            "lookup<|content_invoke_tool_json|>{{\"name\":\"lookup\",\"args\":{{\"value\":{value}}}}}<|end_message|>"
        )
    };
    let output = format!(
        "<|content_thinking|>check both<|end_message|><|message_model|>\
         <|content_text|>I'll look.<|end_message|><|message_model|>{}{}{}",
        call(7),
        "<|message_model|>",
        call(11),
    );
    assert!(plan_accepts(required_plan, &output));
    assert!(!plan_accepts(
        required_plan,
        &format!("{}<|message_model|>{}", output, call(13))
    ));
    assert!(!plan_accepts(
        required_plan,
        "missing<|content_invoke_tool_json|>{\"name\":\"missing\",\"args\":{\"value\":7}}<|end_message|>"
    ));
    assert!(!plan_accepts(
        required_plan,
        "lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"value\":\"seven\"}}<|end_message|>"
    ));

    let mut parser = required_plan
        .create_parser_with_stops(std::iter::empty())
        .unwrap();
    let structural = required_plan.structural_tokens().collect::<Vec<_>>();
    let framed_output = format!("{output}<|content_model_end_sampling|>");
    let mut offset = 0;
    while offset < framed_output.len() {
        if let Some((token_id, spelling)) = structural
            .iter()
            .find(|(_, spelling)| framed_output[offset..].starts_with(*spelling))
        {
            parser.push_structural(*token_id, spelling).unwrap();
            offset += spelling.len();
        } else {
            let next = structural
                .iter()
                .filter_map(|(_, spelling)| framed_output[offset..].find(*spelling))
                .min()
                .map_or(framed_output.len(), |position| offset + position);
            parser.push(&framed_output[offset..next]).unwrap();
            offset = next;
        }
    }
    assert_eq!(
        tool_argument_events(parser.events()),
        ["{\"value\":7}", "{\"value\":11}"]
    );
    assert_eq!(
        parser
            .events()
            .iter()
            .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
            .count(),
        2
    );
    assert_eq!(
        parser.events().last(),
        Some(&SemanticEvent::Finished {
            reason: FinishReason::StopSequence,
        })
    );

    let mut tokenizer = inkling_chat_tokenizer(37);
    let auto = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(INKLING_SMALL_FIXTURE.into()),
        "unrelated-model-id",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Maybe use a tool."})],
            tools: vec![tool],
            tool_choice: ToolChoice::Auto,
            parallel_tool_calls: ParallelToolCallPolicy::Disabled,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    let auto_plan = auto.tool_runtime_plan().unwrap();
    assert_eq!(
        auto_plan.auto_activation_trigger(),
        Some("<|content_invoke_tool_json|>")
    );
    assert!(plan_accepts(
        auto_plan,
        "<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"value\":17}}<|end_message|>"
    ));
    assert!(!plan_accepts(
        auto_plan,
        "<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"value\":17}}<|end_message|><|message_model|>lookup<|content_invoke_tool_json|>{\"name\":\"lookup\",\"args\":{\"value\":19}}<|end_message|>"
    ));
}

#[test]
#[ignore = "requires SAFEMLX_INKLING_MODEL_DIR pointing to a local Inkling checkpoint"]
fn inkling_real_checkpoint_template_is_recognized() {
    let Ok(model_dir) = std::env::var("SAFEMLX_INKLING_MODEL_DIR") else {
        eprintln!("skipping real Inkling checkpoint test: SAFEMLX_INKLING_MODEL_DIR is not set");
        return;
    };
    let model_dir = std::path::PathBuf::from(model_dir);
    let template = load_chat_template(&model_dir)
        .unwrap()
        .expect("Inkling checkpoint must provide a chat template");
    let mut tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(&model_dir).unwrap());
    tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(&model_dir).unwrap());
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        template,
        "real-inkling-checkpoint",
        &[],
        None,
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        prepared.format_profile_identity(),
        Some("inkling.messages.v1")
    );
    assert!(prepared.capabilities().reasoning_parser.is_supported());
    assert_eq!(prepared.generation_prompt(), "<|message_model|>");
}

#[test]
fn gemma_recognition_accepts_source_refactors_and_splits_tool_capabilities() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = gemma4_chat_tokenizer(50);
    let refactored = format!("{GEMMA4_EDGE_FIXTURE}\n{{# converter-only comment #}}");
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(refactored),
        "converter-variant",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            enable_thinking: Some(true),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        prepared.format_profile_identity(),
        Some("gemma.channels.v1")
    );
    assert!(prepared.capabilities().reasoning_parser.is_supported());
    assert!(prepared.capabilities().visible_text_parser.is_supported());
    assert!(prepared.capabilities().tool_output_parser.is_supported());
    assert!(prepared.capabilities().tool_input_rendering.is_supported());
    assert!(prepared
        .capabilities()
        .mapping_tool_arguments
        .is_supported());
    assert!(prepared
        .capabilities()
        .constrained_tool_generation
        .is_supported());

    let mut reasoning_tokenizer = gemma4_reasoning_tokenizer(60);
    let reasoning_only = prepare_chat_from_parts(
        &mut reasoning_tokenizer,
        ModelChatTemplate::Single(GEMMA4_EDGE_FIXTURE.into()),
        "converter-variant",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            enable_thinking: Some(true),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert!(reasoning_only
        .capabilities()
        .reasoning_parser
        .is_supported());
    assert!(!reasoning_only
        .capabilities()
        .tool_output_parser
        .is_supported());
    assert!(!reasoning_only
        .capabilities()
        .tool_input_rendering
        .is_supported());
    assert!(reasoning_only.tool_runtime_plan().is_none());
}

#[test]
fn muse_atem_recognition_accepts_jinja_conditional_keyword_arguments() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = muse_atem_chat_tokenizer(80);
    let template = MUSE_GLIMMER_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "behavior-not-model-id-selects-muse",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            enable_thinking: Some(true),
            add_generation_prompt: true,
            extra_template_kwargs: serde_json::Map::from_iter([(
                "reasoning_strength".into(),
                json!("xhigh"),
            )]),
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert_eq!(
        prepared.format_profile_identity(),
        Some("muse-glimmer.atem.v1")
    );
    assert!(prepared
        .rendered_prompt()
        .contains("Reasoning strength: xhigh."));
    assert!(prepared.rendered_prompt().ends_with("<|start|>assistant"));
    assert!(prepared.capabilities().reasoning_parser.is_supported());
    assert!(prepared.capabilities().tool_input_rendering.is_supported());
    assert!(prepared
        .capabilities()
        .mapping_tool_arguments
        .is_supported());
    assert!(prepared.tool_runtime_plan().is_some());

    let direct = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "behavior-not-model-id-selects-muse",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert!(plan_accepts(
        direct.generation_runtime_plan().unwrap(),
        " to=user<|message|>direct answer<|eot|>"
    ));

    let low_effort = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "behavior-not-model-id-selects-muse",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            reasoning_effort: Some("low".into()),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert!(low_effort
        .rendered_prompt()
        .contains("Reasoning strength: low."));
}

#[test]
fn muse_atem_accepts_structural_eot_that_is_also_checkpoint_eos() {
    // Place ATEM's fourth structural token at ID 255, then configure that ID
    // as a secondary EOS alias behind a distinct canonical EOS.
    let mut tokenizer = muse_atem_chat_tokenizer(252);
    let eot = tokenizer
        .token_to_id("<|eot|>")
        .expect("synthetic Muse tokenizer has an EOT token");
    assert_eq!(eot, 255);
    let compiler = Ok(ConstraintCompiler::synthetic_with_eos_aliases_for_tests(&[
        254, eot,
    ]));
    let template = MUSE_GLIMMER_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template.into()),
        "behavior-not-model-id-selects-muse",
        &[eot],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert!(plan_accepts(
        prepared.generation_runtime_plan().unwrap(),
        " to=user<|message|>direct answer<|eot|>"
    ));
}

#[test]
fn gemma_recognition_accepts_self_defaulting_kwargs_and_gguf_added_tokens() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let template = format!(
        "{{%- set enable_thinking = enable_thinking | default(false) -%}}{}",
        GEMMA4_EDGE_FIXTURE
    );
    let mut tokenizer = gemma4_gguf_chat_tokenizer(50);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(template),
        "converter-variant",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            enable_thinking: Some(true),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();

    assert_eq!(
        prepared.format_profile_identity(),
        Some("gemma.channels.v1")
    );
    assert!(prepared.capabilities().reasoning_parser.is_supported());
    assert!(prepared.rendered_prompt().contains("<|think|>"));
}

#[test]
fn unsloth_gemma_variant_is_recognized_behaviorally_and_accepts_both_argument_forms() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let template = UNSLOTH_GEMMA4_FIXTURE_WITH_TERMINATOR
        .strip_suffix('\n')
        .expect("the fixture-only file terminator is documented");
    for arguments in [
        json!({"value": "mapping"}),
        json!("{\"value\":\"serialized\"}"),
    ] {
        let mut tokenizer = gemma4_chat_tokenizer(70);
        let prepared = prepare_chat_from_parts(
            &mut tokenizer,
            ModelChatTemplate::Single(template.into()),
            "converter-and-model-id-are-not-support-keys",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                messages: vec![
                    json!({"role": "user", "content": "first"}),
                    json!({
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "inspect",
                        "tool_calls": [{
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": arguments}
                        }]
                    }),
                    json!({
                        "role": "tool",
                        "tool_call_id": "call_a",
                        "content": "{\"result\":1}"
                    }),
                ],
                tools: vec![production_tool("lookup")],
                tool_choice: ToolChoice::Required,
                enable_thinking: Some(true),
                add_generation_prompt: true,
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();

        assert_eq!(
            prepared.format_profile_identity(),
            Some("gemma.channels.v1")
        );
        assert!(prepared
            .rendered_prompt()
            .contains("<|channel>thought\ninspect\n<channel|>"));
        assert!(prepared.capabilities().reasoning_parser.is_supported());
        assert!(prepared.capabilities().tool_output_parser.is_supported());
        assert!(prepared.capabilities().tool_input_rendering.is_supported());
        assert!(prepared
            .capabilities()
            .mapping_tool_arguments
            .is_supported());
        assert!(prepared.capabilities().string_tool_arguments.is_supported());
        assert!(prepared.tool_runtime_plan().is_some());
    }
}

#[test]
fn production_gemma4_semantic_events_survive_every_protocol_byte_split() {
    fn push_split_text(
        parser: &mut crate::runtime::generation::streaming::ToolRuntimeParser,
        text: &str,
        start: usize,
        split: usize,
    ) {
        let local_split = split.saturating_sub(start).min(text.len());
        let mut pending = Vec::new();
        for chunk in [
            &text.as_bytes()[..local_split],
            &text.as_bytes()[local_split..],
        ] {
            pending.extend_from_slice(chunk);
            loop {
                match std::str::from_utf8(&pending) {
                    Ok(text) => {
                        parser.push(text).unwrap();
                        pending.clear();
                        break;
                    }
                    Err(error) if error.error_len().is_none() => {
                        let valid_up_to = error.valid_up_to();
                        if valid_up_to == 0 {
                            break;
                        }
                        parser
                            .push(std::str::from_utf8(&pending[..valid_up_to]).unwrap())
                            .unwrap();
                        pending.drain(..valid_up_to);
                    }
                    Err(error) => panic!("golden output is invalid UTF-8: {error}"),
                }
            }
        }
        assert!(pending.is_empty(), "split {split}");
    }

    fn push_structural_output(
        parser: &mut crate::runtime::generation::streaming::ToolRuntimeParser,
        output: &str,
        split: usize,
        structural_spellings: &[&str],
    ) {
        let mut cursor = 0;
        while cursor < output.len() {
            let next = structural_spellings
                .iter()
                .enumerate()
                .filter_map(|(index, spelling)| {
                    output[cursor..]
                        .find(spelling)
                        .map(|offset| (cursor + offset, index, *spelling))
                })
                .min_by_key(|(position, index, _)| (*position, *index));
            let Some((position, structural_index, spelling)) = next else {
                push_split_text(parser, &output[cursor..], cursor, split);
                break;
            };
            push_split_text(parser, &output[cursor..position], cursor, split);
            parser
                .push_structural(23 + u32::try_from(structural_index).unwrap(), spelling)
                .unwrap();
            cursor = position + spelling.len();
        }
    }

    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let mut tokenizer = gemma4_chat_tokenizer(23);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(GEMMA4_EDGE_FIXTURE.into()),
        "unrelated-architecture",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "lookup"})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            parallel_tool_calls: ParallelToolCallPolicy::Disabled,
            enable_thinking: Some(true),
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    let plan = prepared
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("registered Gemma 4 profile must be supported"));
    let tool_output = concat!(
        "<|channel>thought\nNeed 東京 🦀\n<channel|>",
        "<|tool_call>call:lookup{value:7}<tool_call|><|tool_response>",
    );
    let text_output = "<|channel>thought\nDone 🦀\n<channel|>Visible Bogotá<turn|>";
    let structural_spellings = [
        "<|channel>",
        "<channel|>",
        "<|tool_call>",
        "<tool_call|>",
        "<|\"|>",
        "<|tool_response>",
        "<turn|>",
    ];
    let grammar_output = tool_output
        .strip_suffix("<|tool_response>")
        .expect("profile stop is outside the tool grammar");
    let mut grammar = plan.generation_constraint().grammar_state();
    let mut remaining = grammar_output;
    while !remaining.is_empty() {
        let next = structural_spellings
            .iter()
            .enumerate()
            .filter_map(|(index, spelling)| {
                remaining
                    .find(spelling)
                    .map(|position| (position, index, *spelling))
            })
            .min_by_key(|(position, index, _)| (*position, *index));
        let Some((position, structural_index, spelling)) = next else {
            for byte in remaining.bytes() {
                grammar.commit(u32::from(byte)).unwrap();
            }
            break;
        };
        for byte in remaining[..position].bytes() {
            grammar.commit(u32::from(byte)).unwrap();
        }
        grammar
            .commit(23 + u32::try_from(structural_index).unwrap())
            .unwrap();
        remaining = &remaining[position + spelling.len()..];
    }
    assert!(grammar.is_complete().unwrap());

    for split in 0..=tool_output.len() {
        let mut parser = plan.create_parser_with_stops(std::iter::empty()).unwrap();
        let visible_output = tool_output
            .strip_suffix("<|tool_response>")
            .expect("tool response is the structural stop");
        let split = split.min(visible_output.len());
        push_structural_output(&mut parser, visible_output, split, &structural_spellings);
        parser.push_structural(28, "<|tool_response>").unwrap();
        let reasoning = parser
            .events()
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ReasoningDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(reasoning, "Need 東京 🦀\n", "split {split}");
        assert!(parser.events().contains(&SemanticEvent::ToolCallStart {
            index: 0,
            id: "call_0".into(),
            name: "lookup".into(),
        }));
        assert_eq!(tool_argument_events(parser.events()), [r#"{"value":7}"#]);
        assert_eq!(
            parser
                .events()
                .iter()
                .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                .count(),
            1,
            "split {split}"
        );
        assert_eq!(
            parser.events().last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::StopSequence
            }),
            "split {split}"
        );
    }

    for split in 0..=text_output.len() {
        let mut parser = plan.create_parser_with_stops(std::iter::empty()).unwrap();
        let visible_output = text_output
            .strip_suffix("<turn|>")
            .expect("turn close is the structural stop");
        let split = split.min(visible_output.len());
        push_structural_output(&mut parser, visible_output, split, &structural_spellings);
        parser.push_structural(29, "<turn|>").unwrap();
        let reasoning = parser
            .events()
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ReasoningDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        let visible = parser
            .events()
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(reasoning, "Done 🦀\n", "split {split}");
        assert_eq!(visible, "Visible Bogotá", "split {split}");
        assert_eq!(
            parser.events().last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::StopSequence
            }),
            "split {split}"
        );
    }
}

#[test]
fn gemma4_model_type_does_not_grant_unregistered_template_support() {
    let raw = Tokenizer::new(WordLevel::default());
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single("unregistered Gemma 4 template".into()),
        "gemma4",
        &[],
        None,
        ChatTemplateRequest::default(),
    )
    .unwrap();

    assert_eq!(prepared.format_profile_identity(), None);
    assert!(matches!(
        prepared.native_tool_support(),
        NativeToolSupport::Unsupported { reason }
            if reason.contains("no behavioral format recognizer")
    ));
}

#[test]
fn explicit_thinking_requires_a_recognized_semantic_protocol_unless_opted_out() {
    let raw = Tokenizer::new(WordLevel::default());
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let template = ModelChatTemplate::Single(
        "{% for message in messages %}{{ message.content }}{% endfor %}".into(),
    );
    let request = ChatTemplateRequest {
        messages: vec![json!({"role": "user", "content": "hello"})],
        enable_thinking: Some(true),
        ..ChatTemplateRequest::default()
    };
    let error = prepare_chat_from_parts(
        &mut tokenizer,
        template.clone(),
        "unknown",
        &[],
        None,
        request,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("no semantic reasoning protocol was recognized"));

    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        template,
        "unknown",
        &[],
        None,
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            enable_thinking: Some(true),
            allow_unparsed_reasoning: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert!(matches!(
        prepared.semantic_support(),
        crate::runtime::chat::SemanticSupport::Unsupported { .. }
    ));
}

#[test]
fn mistral_architecture_name_does_not_grant_an_unregistered_template_support() {
    let raw = Tokenizer::new(WordLevel::default());
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let template = ModelChatTemplate::Single("unregistered mistral template".into());
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        template,
        "MistralForCausalLM",
        &[],
        None,
        ChatTemplateRequest::default(),
    )
    .unwrap();

    assert_eq!(prepared.format_profile_identity(), None);
    assert!(matches!(
        prepared.native_tool_support(),
        NativeToolSupport::Unsupported { reason }
            if reason.contains("no behavioral format recognizer")
    ));
}

#[test]
fn named_template_behavioral_recognition_uses_only_the_selected_body() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let templates = ModelChatTemplate::Named(BTreeMap::from([
        ("default".into(), "unregistered named default".into()),
        ("tool_use".into(), HERMES2_PRO_TOOL_USE_FIXTURE.to_owned()),
    ]));

    let mut tokenizer = production_chat_tokenizer(31);
    let default = prepare_chat_from_parts(
        &mut tokenizer,
        templates.clone(),
        "named-selection",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "hello"})],
            tool_choice: ToolChoice::None,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        default.template_identity(),
        &ChatTemplateIdentity::Named("default".into())
    );
    assert_eq!(default.format_profile_identity(), None);
    assert!(matches!(
        default.native_tool_support(),
        NativeToolSupport::Unsupported { reason }
            if reason.contains("no behavioral format recognizer")
    ));

    let selected_tool_use = prepare_chat_from_parts(
        &mut tokenizer,
        templates,
        "named-selection",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "call lookup"})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap();
    assert_eq!(
        selected_tool_use.template_identity(),
        &ChatTemplateIdentity::Named("tool_use".into())
    );
    assert_eq!(
        selected_tool_use.format_profile_identity(),
        Some("xml-tools.v1")
    );
    assert!(matches!(
        selected_tool_use.native_tool_support(),
        NativeToolSupport::Supported
    ));
}

#[test]
fn synthetic_profile_compiles_request_tools_before_rendering() {
    let mut tokenizer = synthetic_chat_tokenizer(0);
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let template = ModelChatTemplate::Single(SYNTHETIC_TOOL_TEMPLATE.into());
    let valid = ChatTemplateRequest {
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": false
                }
            }
        })],
        tool_choice: ToolChoice::Required,
        add_generation_prompt: true,
        ..ChatTemplateRequest::default()
    };
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        template.clone(),
        "synthetic",
        &[],
        Some(&compiler),
        valid,
    )
    .unwrap();
    assert_eq!(
        prepared.format_profile_identity(),
        Some("safemlx.synthetic-tools.v1")
    );
    let plan = prepared
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("synthetic profile must prepare an executable runtime plan"));
    let mut parser = plan.create_parser().unwrap();
    parser
        .push(r#"{"calls":[{"name":"lookup","arguments":{"query":"weather"}}]}"#)
        .unwrap();
    parser.finish(FinishReason::GrammarComplete).unwrap();
    assert!(parser.events().iter().any(|event| matches!(
        event,
        SemanticEvent::ToolCallStart { name, .. } if name == "lookup"
    )));

    let invalid = ChatTemplateRequest {
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "oneOf": [{"type": "string"}, {"type": "number"}]
                        }
                    }
                }
            }
        })],
        tool_choice: ToolChoice::Required,
        extra_template_kwargs: serde_json::Map::from_iter([("fail_render".into(), json!(true))]),
        ..ChatTemplateRequest::default()
    };
    let error = prepare_chat_from_parts(
        &mut tokenizer,
        template,
        "synthetic",
        &[],
        Some(&compiler),
        invalid,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        TextModelError::ToolConstraint(ref message) if message.contains("unsupported schema composition")
    ));
}

#[test]
fn structural_tokens_resolve_against_each_preparation_tokenizer() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let request = || ChatTemplateRequest {
        tools: vec![json!({
            "type": "function",
            "function": {
                "name": "ping",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": false
                }
            }
        })],
        tool_choice: ToolChoice::Required,
        ..ChatTemplateRequest::default()
    };

    let mut first_tokenizer = synthetic_chat_tokenizer(0);
    let first = prepare_chat_from_parts(
        &mut first_tokenizer,
        ModelChatTemplate::Single(SYNTHETIC_TOOL_TEMPLATE.into()),
        "synthetic-first",
        &[],
        Some(&compiler),
        request(),
    )
    .unwrap();
    let mut second_tokenizer = synthetic_chat_tokenizer(7);
    let second = prepare_chat_from_parts(
        &mut second_tokenizer,
        ModelChatTemplate::Single(SYNTHETIC_TOOL_TEMPLATE.into()),
        "synthetic-second",
        &[],
        Some(&compiler),
        request(),
    )
    .unwrap();

    assert_eq!(first.preserved_structural_token_ids(), &[0]);
    assert_eq!(second.preserved_structural_token_ids(), &[7]);
    let first_plan = first
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("synthetic profile must prepare a runtime plan"));
    let second_plan = second
        .tool_runtime_plan()
        .unwrap_or_else(|| panic!("synthetic profile must prepare a runtime plan"));
    assert_eq!(first_plan.structural_token_ids().collect::<Vec<_>>(), [0]);
    assert_eq!(second_plan.structural_token_ids().collect::<Vec<_>>(), [7]);
}

#[test]
fn tokenizer_analysis_is_model_scoped_while_schemas_compile_per_prepare_chat() {
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    assert_eq!(compiler.as_ref().unwrap().cache_analysis_counts(), (1, 0));
    let mut tokenizer = synthetic_chat_tokenizer(0);

    let prepare = |tokenizer: &mut ChatTokenizer, property: &str| {
        prepare_chat_from_parts(
            tokenizer,
            ModelChatTemplate::Single(SYNTHETIC_TOOL_TEMPLATE.into()),
            "one-loaded-model",
            &[],
            Some(&compiler),
            ChatTemplateRequest {
                tools: vec![json!({
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {(property): {"type": "string"}},
                            "required": [property],
                            "additionalProperties": false
                        }
                    }
                })],
                tool_choice: ToolChoice::Required,
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap()
    };

    let first = prepare(&mut tokenizer, "city");
    let second = prepare(&mut tokenizer, "country");
    assert_eq!(compiler.as_ref().unwrap().cache_analysis_counts(), (1, 2));
    assert_ne!(
        first
            .tool_runtime_plan()
            .unwrap()
            .generation_constraint()
            .fingerprint,
        second
            .tool_runtime_plan()
            .unwrap()
            .generation_constraint()
            .fingerprint,
        "request schemas must compile into independent grammars"
    );
}

#[test]
fn missing_structural_added_token_fails_before_prompt_rendering() {
    let raw = Tokenizer::new(WordLevel::default());
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw);
    let compiler = Ok(ConstraintCompiler::synthetic_for_tests());
    let error = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(SYNTHETIC_TOOL_TEMPLATE.into()),
        "synthetic-missing-structural-token",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            tool_choice: ToolChoice::None,
            extra_template_kwargs: serde_json::Map::from_iter([(
                "fail_render".into(),
                json!(true),
            )]),
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap_err();

    assert!(matches!(
        error,
        TextModelError::ToolConstraint(ref message)
            if message.contains(SYNTHETIC_STRUCTURAL_TOKEN)
                && message.contains("not registered as an added token")
    ));
}

#[test]
fn grammar_compiler_failure_is_reported_before_prompt_rendering() {
    let mut tokenizer = synthetic_chat_tokenizer(0);
    let compiler = Err("failed to compile tool grammar: synthetic compiler failure".into());
    let error = prepare_chat_from_parts(
        &mut tokenizer,
        ModelChatTemplate::Single(SYNTHETIC_TOOL_TEMPLATE.into()),
        "synthetic-grammar-failure",
        &[],
        Some(&compiler),
        ChatTemplateRequest {
            tool_choice: ToolChoice::Required,
            extra_template_kwargs: serde_json::Map::from_iter([(
                "fail_render".into(),
                json!(true),
            )]),
            ..ChatTemplateRequest::default()
        },
    )
    .unwrap_err();

    assert!(matches!(
        error,
        TextModelError::ToolConstraint(ref message)
            if message == "failed to compile tool grammar: synthetic compiler failure"
    ));
}

#[test]
fn prepared_prompt_matches_existing_json_renderer() {
    let template = ModelChatTemplate::Single(
        concat!(
            "{{ prefix }}",
            "{% for message in messages %}{{ message.role }}:{{ message.content }};",
            "{% endfor %}",
            "{% if tools %}tools={{ tools|length }};{% endif %}",
            "{% if add_generation_prompt %}assistant:{% endif %}",
        )
        .into(),
    );
    let messages = vec![json!({"role": "user", "content": "hello"})];
    let tools = vec![json!({"type": "function"})];
    let kwargs = serde_json::Map::from_iter([("prefix".into(), json!("<bos>"))]);

    for add_generation_prompt in [false, true] {
        let raw = Tokenizer::new(WordLevel::default());
        let mut existing_tokenizer = ChatTokenizer::from_tokenizer(raw.clone());
        let expected = existing_tokenizer
            .apply_chat_template_json(
                template.clone(),
                [messages.clone()],
                Some(&tools),
                "legacy-json-renderer",
                add_generation_prompt,
                Some(&kwargs),
            )
            .unwrap()
            .remove(0);
        let mut preparation_tokenizer = ChatTokenizer::from_tokenizer(raw);
        let prepared = prepare_chat_from_parts(
            &mut preparation_tokenizer,
            template.clone(),
            "prepared-json-renderer",
            &[],
            None,
            ChatTemplateRequest {
                messages: messages.clone(),
                tools: tools.clone(),
                add_generation_prompt,
                extra_template_kwargs: kwargs.clone(),
                ..ChatTemplateRequest::default()
            },
        )
        .unwrap();

        assert_eq!(prepared.rendered_prompt(), expected);
    }
}

#[test]
fn eos_sidecars_load_single_and_multiple_ids() {
    let dir = temp_model_dir(
        r#"{
              "model_type": "llama",
              "eos_token_id": 1,
              "text_config": { "eos_token_id": [2, 3] }
            }"#,
    );

    assert_eq!(eos_token_ids_from_sidecar_dir(&dir).unwrap(), [1, 2, 3]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn eos_sidecars_load_generation_config_only_ids() {
    let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
    fs::write(
        dir.join("generation_config.json"),
        r#"{"eos_token_id":[4,5]}"#,
    )
    .unwrap();

    assert_eq!(eos_token_ids_from_sidecar_dir(&dir).unwrap(), [4, 5]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn checkpoint_generation_config_resolves_declared_values_and_request_overrides() {
    let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
    fs::write(
        dir.join("generation_config.json"),
        r#"{
            "do_sample": true,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "max_new_tokens": 512
        }"#,
    )
    .unwrap();

    let checkpoint = read_checkpoint_generation_config(&dir).unwrap().unwrap();
    let resolved =
        resolve_generation_config(Some(&checkpoint), GenerationConfigOverrides::default()).unwrap();
    assert!(resolved.do_sample);
    assert_eq!(resolved.temperature, 1.0);
    assert_eq!(resolved.top_k, 64);
    assert_eq!(resolved.top_p, 0.95);
    assert_eq!(resolved.min_p, 0.0);
    assert_eq!(resolved.max_new_tokens, Some(512));

    let overridden = resolve_generation_config(
        Some(&checkpoint),
        GenerationConfigOverrides {
            temperature: Some(0.0),
            top_k: Some(12),
            max_new_tokens: Some(32),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!overridden.do_sample);
    assert_eq!(overridden.temperature, 0.0);
    assert_eq!(overridden.top_k, 12);
    assert_eq!(overridden.max_new_tokens, Some(32));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn checkpoint_generation_config_honors_do_sample_false() {
    let checkpoint = crate::CheckpointGenerationConfig {
        do_sample: Some(false),
        temperature: Some(0.8),
        top_k: Some(20),
        ..Default::default()
    };
    let resolved =
        resolve_generation_config(Some(&checkpoint), GenerationConfigOverrides::default()).unwrap();
    assert!(!resolved.do_sample);
    assert_eq!(resolved.temperature, 0.0);
    assert_eq!(resolved.top_k, 20);

    let temperature_override = resolve_generation_config(
        Some(&checkpoint),
        GenerationConfigOverrides {
            temperature: Some(0.6),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(temperature_override.do_sample);
    assert_eq!(temperature_override.temperature, 0.6);

    let greedy_override = resolve_generation_config(
        Some(&checkpoint),
        GenerationConfigOverrides {
            do_sample: Some(false),
            temperature: Some(0.6),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!greedy_override.do_sample);
    assert_eq!(greedy_override.temperature, 0.0);
}

#[test]
fn eos_sidecars_use_text_config_when_top_level_is_missing() {
    let dir = temp_model_dir(
        r#"{
              "model_type": "qwen3_vl",
              "text_config": { "eos_token_id": [7, 8] }
            }"#,
    );

    assert_eq!(eos_token_ids_from_sidecar_dir(&dir).unwrap(), [7, 8]);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn eos_sources_merge_stably_and_deduplicate_overlaps() {
    let merged = merge_eos_token_id_sources([vec![1, 2, 1], vec![2, 3], vec![3, 4, 2], vec![4, 5]]);

    assert_eq!(merged, [1, 2, 3, 4, 5]);
}

#[test]
fn eos_loading_allows_missing_sidecar_files_and_gguf_key() {
    let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
    fs::remove_file(dir.join("config.json")).unwrap();
    let metadata = std::collections::HashMap::new();

    assert!(eos_token_ids_from_sidecar_dir(&dir).unwrap().is_empty());
    assert!(gguf_eos_token_ids(&metadata).unwrap().is_empty());
    fs::remove_dir_all(dir).unwrap();
}

fn append_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn append_gguf_string_value(bytes: &mut Vec<u8>, key: &str, value: &str) {
    append_gguf_string(bytes, key);
    bytes.extend_from_slice(&8u32.to_le_bytes());
    append_gguf_string(bytes, value);
}

fn append_gguf_strings(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
    append_gguf_string(bytes, key);
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&8u32.to_le_bytes());
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for value in values {
        append_gguf_string(bytes, value);
    }
}

#[test]
fn loads_tokenizer_directly_from_gguf_metadata() {
    let dir = temp_model_dir(r#"{"model_type":"qwen3"}"#);
    fs::remove_file(dir.join("tokenizer.json")).unwrap();
    let file = dir.join("embedded-tokenizer.gguf");
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&6u64.to_le_bytes());
    append_gguf_string_value(&mut bytes, "general.architecture", "qwen3");
    append_gguf_string_value(&mut bytes, "tokenizer.ggml.model", "gpt2");
    append_gguf_strings(
        &mut bytes,
        "tokenizer.ggml.tokens",
        &["<eos>", "h", "e", "l", "o", "he", "ll", "hell", "hello"],
    );
    append_gguf_strings(
        &mut bytes,
        "tokenizer.ggml.merges",
        &["h e", "l l", "he ll", "hell o"],
    );
    append_gguf_string(&mut bytes, "tokenizer.ggml.eos_token_id");
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    append_gguf_string(&mut bytes, "tokenizer.ggml.add_eos_token");
    bytes.extend_from_slice(&7u32.to_le_bytes());
    bytes.push(1);
    fs::write(&file, bytes).unwrap();

    let tokenizer = load_tokenizer(&file).unwrap();
    assert_eq!(tokenizer.encode("hello", true).unwrap().get_ids(), &[8, 0]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn load_tokenizer_accepts_top_level_qwen3_5_moe_metadata() {
    let dir = temp_model_dir(
        r#"{
              "model_type": "qwen3_5_moe",
              "text_config": {
                "model_type": "qwen3_5_moe_text"
              }
            }"#,
    );
    let tokenizer = load_tokenizer(&dir).unwrap();
    assert_eq!(tokenizer.get_vocab_size(false), 0);
}

#[test]
fn load_chat_template_reads_standalone_jinja_file() {
    let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
    fs::write(
        dir.join("chat_template.jinja"),
        "hello {{ messages[0].role }}",
    )
    .unwrap();

    let template = load_chat_template(&dir).unwrap().unwrap();
    assert_eq!(
        template.select(None).unwrap().template(),
        "hello {{ messages[0].role }}"
    );
    assert!(matches!(template, ModelChatTemplate::Single(_)));
}

#[test]
fn gguf_sidecar_loads_named_chat_templates() {
    let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
    let gguf_path = dir.join("model.gguf");
    let file = fs::File::create(&gguf_path).unwrap();
    eredu_gguf::Writer::default()
        .write(file, &BTreeMap::new(), &[])
        .unwrap();
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{
              "chat_template": [
                {"name": "default", "template": "{{ default_kw }}"},
                {"name": "tool_use", "template": "{{ tool_kw }}"}
              ]
            }"#,
    )
    .unwrap();

    assert_eq!(
        chat_template_kwargs(&gguf_path).unwrap(),
        vec!["default_kw"]
    );
    let templates = load_chat_template(&dir).unwrap().unwrap();
    let tools = [json!({"type": "function"})];
    let selected = templates.select(Some(&tools)).unwrap();
    assert_eq!(selected.template(), "{{ tool_kw }}");
    assert_eq!(
        selected.identity(),
        &ChatTemplateIdentity::Named("tool_use".into())
    );
}

#[test]
fn load_tokenizer_template_kwargs_reads_special_tokens() {
    let dir = temp_model_dir(r#"{"model_type":"llama"}"#);
    fs::write(
        dir.join("tokenizer_config.json"),
        r#"{
              "bos_token": "<bos>",
              "eos_token": "<eos>",
              "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}{{ custom_flag }}",
              "model_max_length": 128
            }"#,
    )
    .unwrap();

    let kwargs = load_tokenizer_template_kwargs(&dir).unwrap();
    assert_eq!(kwargs.get("bos_token"), Some(&json!("<bos>")));
    assert_eq!(kwargs.get("eos_token"), Some(&json!("<eos>")));
    assert!(!kwargs.contains_key("chat_template"));
    assert!(!kwargs.contains_key("model_max_length"));
    assert_eq!(chat_template_kwargs(&dir).unwrap(), vec!["custom_flag"]);
}

#[test]
#[ignore = "requires local Nemotron-H model files and Python transformers"]
fn nemotron_chat_template_matches_transformers_on_small_prompts() {
    let model_dir = std::env::var("NEMOTRON_H_PARITY_MODEL_DIR")
        .expect("set NEMOTRON_H_PARITY_MODEL_DIR to a local Nemotron-H snapshot");
    let model_dir = std::path::PathBuf::from(model_dir);
    let template = load_chat_template(&model_dir).unwrap().unwrap();
    let mut tokenizer = ChatTokenizer::from_tokenizer(load_tokenizer(&model_dir).unwrap());
    let conversations = vec![vec![
        json!({"role": "system", "content": "You are concise."}),
        json!({"role": "user", "content": "What is 2+2?"}),
    ]];
    let prompts = ["Hello, world!", "What is 84 * 3 / 2?"];
    let local_prompt_ids = prompts
        .iter()
        .map(|prompt| tokenizer.encode(*prompt, false).unwrap().get_ids().to_vec())
        .collect::<Vec<_>>();

    let rendered = tokenizer
        .apply_chat_template_json(
            template.clone(),
            conversations.clone(),
            None,
            "nemotron_h",
            true,
            None,
        )
        .unwrap()
        .remove(0);
    let script = r#"
import json, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
messages = json.loads(sys.argv[2])
prompts = json.loads(sys.argv[3])
rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = [tok.encode(prompt, add_special_tokens=False) for prompt in prompts]
print(json.dumps({"rendered": rendered, "ids": ids}))
"#;
    let python =
        std::env::var("NEMOTRON_H_PARITY_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let output = Command::new(&python)
        .arg("-c")
        .arg(script)
        .arg(&model_dir)
        .arg(serde_json::to_string(&conversations[0]).unwrap())
        .arg(serde_json::to_string(&prompts).unwrap())
        .output()
        .expect("failed to run Python transformers parity check");
    assert!(
        output.status.success(),
        "transformers parity script failed with {python}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let expected: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(rendered, expected["rendered"].as_str().unwrap());
    let expected_ids: Vec<Vec<u32>> = serde_json::from_value(expected["ids"].clone()).unwrap();
    assert_eq!(local_prompt_ids, expected_ids);
    assert!(template
        .select(None)
        .unwrap()
        .template()
        .contains("<|im_start|>assistant"));
}

#[test]
fn gguf_architecture_resolution_recognizes_exact_qwen2_identity() {
    assert_eq!(
        GgufArchitecture::resolve("qwen2").unwrap(),
        GgufArchitecture::Qwen2
    );
    for nearby in ["qwen", "qwen2moe", "qwen2vl", "qwen2.5"] {
        assert!(GgufArchitecture::resolve(nearby).is_err());
    }
}
