use super::*;
use eredu::api::{
    PreparedChatGenerationRequest, PreparedChatGenerationSettings, PreparedChatInput,
};
use eredu::runtime::chat::{ChatTemplateRequest, ParallelToolCallPolicy, ToolChoice};
use eredu_core::{FinishReason, SemanticEvent};
use serde_json::{json, Value};

const TEMPLATE: &str = include_str!("../fixtures/chat_templates/nanbeige4.2-0e137298.jinja");
const TOKENIZER: &str =
    include_str!("../fixtures/chat_templates/nanbeige4.2-0e137298-tokenizer.json");

type Calls = std::rc::Rc<std::cell::RefCell<BackendCalls>>;

fn model() -> (LoadedModel<MockBackend>, Calls) {
    let tokenizer = Tokenizer::from_bytes(TOKENIZER).unwrap();
    let backend = MockBackend::default();
    let calls = backend.calls.clone();
    (
        LoadedModel::from_runtime(
            ModelRuntime::prepare(backend, ()).unwrap(),
            ChatTokenizer::from_tokenizer(tokenizer),
            LoadedTextModelConfig {
                model_family: ModelKind::Nanbeige,
                effective_model_type: "nanbeige".into(),
                model_id: "Nanbeige/Nanbeige4.2-3B".into(),
                chat_template: Some(TEMPLATE.into()),
                eos_token_ids: vec![166101],
                checkpoint_generation_config: None,
            },
        ),
        calls,
    )
}

fn tools() -> Vec<Value> {
    vec![json!({"type":"function", "function": {
        "name":"todo__todo_write", "parameters": {
            "type":"object", "properties":{"content":{"type":"string"}},
            "required":["content"], "additionalProperties":false
        }
    }})]
}

fn request(thinking: bool, tools: Vec<Value>) -> ChatTemplateRequest {
    ChatTemplateRequest {
        messages: vec![json!({"role":"user", "content":"Write the todo and report the result."})],
        tools,
        tool_choice: ToolChoice::Auto,
        enable_thinking: Some(thinking),
        add_generation_prompt: true,
        ..Default::default()
    }
}

fn generate(
    model: &mut LoadedModel<MockBackend>,
    prepared: &eredu::runtime::chat::PreparedChat,
    calls: &Calls,
    text: &str,
) -> (eredu_core::GenerationOutput, Vec<SemanticEvent>) {
    // No initial Metaspace prefix: these are continuation tokens, as emitted by
    // the model, including the checkpoint's non-special atomic XML markers.
    let mut tokenizer = Tokenizer::from_bytes(TOKENIZER).unwrap();
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::metaspace::Metaspace::new(
        '▁',
        tokenizers::pre_tokenizers::metaspace::PrependScheme::Never,
        false,
    )));
    let ids = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
    calls.borrow_mut().scripted_tokens = ids.into();
    let mut events = Vec::new();
    let output = model
        .generate_prepared_chat(PreparedChatGenerationRequest {
            input: PreparedChatInput::rendered_prompt(prepared),
            settings: PreparedChatGenerationSettings {
                overrides: GenerationConfigOverrides {
                    max_new_tokens: Some(1024),
                    ..Default::default()
                },
                ..Default::default()
            },
            caller_stop_sequences: &[],
            cancellation: Default::default(),
            on_event: |e| events.push(e),
        })
        .unwrap();
    (output, events)
}

#[test]
fn official_nanbeige_reasoning_closing_token_is_selectable() {
    let (mut model, calls) = model();
    let prepared = model.prepare_chat(request(true, vec![])).unwrap();
    assert!(prepared.rendered_prompt().ends_with("<think>\n"));
    let (_, events) = generate(
        &mut model,
        &prepared,
        &calls,
        "Follow the instruction.\n</think>\n\nhello<|im_end|>",
    );
    let reasoning = events
        .iter()
        .filter_map(|e| match e {
            SemanticEvent::ReasoningDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<String>();
    let visible = events
        .iter()
        .filter_map(|e| match e {
            SemanticEvent::TextDelta(t) => Some(t.as_str()),
            _ => None,
        })
        .collect::<String>();
    assert_eq!(reasoning, "Follow the instruction.");
    assert_eq!(visible.trim(), "hello");
}

#[test]
fn official_nanbeige_bare_tool_marker_at_budget_exhaustion_cannot_execute() {
    let (mut model, calls) = model();
    let prepared = model.prepare_chat(request(false, tools())).unwrap();
    calls.borrow_mut().scripted_tokens = [166105].into();
    let mut events = Vec::new();
    let result = model.generate_prepared_chat(PreparedChatGenerationRequest {
        input: PreparedChatInput::rendered_prompt(&prepared),
        settings: PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(1),
                ..Default::default()
            },
            ..Default::default()
        },
        caller_stop_sequences: &[],
        cancellation: Default::default(),
        on_event: |e| events.push(e),
    });
    assert!(result.is_err());
    assert!(!events
        .iter()
        .any(|e| matches!(e, SemanticEvent::ToolCallEnd | SemanticEvent::TextDelta(_))));
}

#[test]
fn official_nanbeige_replay_rejects_unescaped_parameter_delimiters() {
    let (mut model, _) = model();
    for content in ["before</parameter>after", "before\n</parameter>after"] {
        let mut request = request(false, tools());
        request.messages.push(json!({"role":"assistant", "content":"", "tool_calls":[{
            "id":"call_0", "type":"function", "function":{"name":"todo__todo_write", "arguments":{"content":content}}
        }]}));
        assert!(model.prepare_chat(request).is_err());
    }
}

#[test]
fn official_nanbeige_xml_whitespace_values_and_history() {
    for whitespace in ["\n", "  ", "", "\r\n\t "] {
        let (mut model, calls) = model();
        let mut req = request(true, tools());
        req.parallel_tool_calls = ParallelToolCallPolicy::Enabled {
            max_calls: std::num::NonZeroUsize::new(2),
        };
        let prepared = model.prepare_chat(req.clone()).unwrap();
        let value = "  first\n\n第二行\t \n";
        let call = format!("<tool_call>{whitespace}<function=todo__todo_write>{whitespace}<parameter=content>\n{value}\n</parameter>{whitespace}</function>{whitespace}</tool_call>");
        let (_, events) = generate(
            &mut model,
            &prepared,
            &calls,
            &format!("Use the tool.\n</think>\n{call}{whitespace}{call}"),
        );
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, SemanticEvent::ToolCallEnd))
                .count(),
            2
        );
        let args = events
            .iter()
            .filter_map(|e| match e {
                SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => {
                    Some(serde_json::from_str::<Value>(json_fragment).unwrap())
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            args,
            vec![json!({"content":value}), json!({"content":value})]
        );
        assert!(!events
            .iter()
            .any(|e| matches!(e, SemanticEvent::TextDelta(t) if !t.trim().is_empty())));
        req.messages.push(json!({"role":"assistant", "content":"", "reasoning_content":"Use the tool.", "tool_calls":[
            {"id":"call_0", "type":"function", "function":{"name":"todo__todo_write", "arguments":args[0]}},
            {"id":"call_1", "type":"function", "function":{"name":"todo__todo_write", "arguments":args[1]}}
        ]}));
        req.messages
            .push(json!({"role":"tool", "tool_call_id":"call_0", "content":"Saved two items."}));
        req.messages
            .push(json!({"role":"tool", "tool_call_id":"call_1", "content":"Saved two items."}));
        let replay = model.prepare_chat(req).unwrap();
        assert!(replay
            .rendered_prompt()
            .contains(&format!("<parameter=content>\n{value}\n</parameter>")));
        assert!(replay
            .rendered_prompt()
            .contains("<tool_response>\nSaved two items.\n</tool_response>"));
        let (_, answer) = generate(
            &mut model,
            &replay,
            &calls,
            "The tool succeeded.\n</think>\nSaved two items.<|im_end|>",
        );
        let visible = answer
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(visible.trim(), "Saved two items.");
        assert_eq!(
            answer.last(),
            Some(&SemanticEvent::Finished {
                reason: FinishReason::StopSequence
            })
        );
    }
}
