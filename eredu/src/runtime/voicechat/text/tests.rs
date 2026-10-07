use super::*;
use std::error::Error;
use tokenizers::{
    models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
    processors::template::TemplateProcessing, PaddingParams, PaddingStrategy, TruncationParams,
};

fn fixture(unknown: &str) -> (Tokenizer, TextControls) {
    let vocabulary = [
        ("[UNK]", 0),
        ("<s>", 1),
        ("</s>", 2),
        ("user-start", 3),
        ("user-end", 4),
        ("listen", 11),
        ("pad", 12),
        ("call-start", 20),
        ("call-end", 21),
        ("response-end", 22),
        ("hello", 30),
        ("world", 31),
        ("<TOOL_RESPONSE>", 32),
        ("</TOOL_RESPONSE>", 33),
        ("out-of-domain", 131_072),
    ]
    .into_iter()
    .map(|(word, id)| (word.to_owned(), id))
    .collect();
    let words = WordLevel::builder()
        .vocab(vocabulary)
        .unk_token(unknown.to_owned())
        .build()
        .unwrap();
    let mut tokenizer = tokenizers::Tokenizer::new(words);
    tokenizer.with_pre_tokenizer(Some(WhitespaceSplit));
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("<s> $A </s>")
            .unwrap()
            .special_tokens(vec![("<s>", 1), ("</s>", 2)])
            .build()
            .unwrap(),
    ));
    (
        Tokenizer::from_tokenizer(tokenizer),
        TextControls {
            bos: 1,
            eos: 2,
            pad: 12,
            listening: 11,
            user_bos: 3,
            user_eos: 4,
            tool_call_start: 20,
            tool_call_end: 21,
            tool_response_end: 22,
        },
    )
}

fn prepare(max_bytes: usize, max_tokens: usize) -> VoiceChatTextPreparation {
    let (tokenizer, controls) = fixture("[UNK]");
    VoiceChatTextPreparation::from_tokenizer(
        &tokenizer,
        controls,
        VoiceChatTextLimits {
            max_bytes,
            max_tokens,
        },
    )
    .unwrap()
}

#[test]
fn prompt_admits_framing_once_and_response_retains_only_supplied_markers() {
    let prepared = prepare(256, 16);
    let prompt = prepared
        .prepare_system_prompt(
            "hello world",
            PromptPolicy {
                repeat: 2,
                max_tokens: 8,
            },
        )
        .unwrap();
    let (tokens, policy) = prompt.invocation_prompt().unwrap();
    assert_eq!(tokens, [30, 31]);
    assert_eq!(policy.repeat, 2);
    assert!(matches!(
        prepared.prepare_system_prompt(
            "hello world",
            PromptPolicy {
                repeat: 2,
                max_tokens: 7
            }
        ),
        Err(VoiceChatTextError::Prompt(ChannelError::PromptCapacity {
            required: 8,
            capacity: 7
        }))
    ));
    let response = prepared
        .prepare_tool_response("<TOOL_RESPONSE> hello </TOOL_RESPONSE>")
        .unwrap();
    assert_eq!(response.tokens(), [32, 30, 33]);
    assert!(prepared
        .prepare_system_prompt(
            " \n\t",
            PromptPolicy {
                repeat: 1,
                max_tokens: 1
            }
        )
        .unwrap()
        .invocation_prompt()
        .is_none());
    assert!(matches!(
        prepared.prepare_tool_response(" \n\t"),
        Err(VoiceChatTextError::EmptyToolResponse)
    ));
}

#[test]
fn text_capacity_rejects_complete_encoding_without_checkpoint_truncation_or_padding() {
    let (mut tokenizer, controls) = fixture("[UNK]");
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: 3,
            ..Default::default()
        }))
        .unwrap();
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(8),
        pad_id: 12,
        pad_token: "pad".to_owned(),
        ..Default::default()
    }));
    let prepared = VoiceChatTextPreparation::from_tokenizer(
        &tokenizer,
        controls,
        VoiceChatTextLimits {
            max_bytes: 128,
            max_tokens: 4,
        },
    )
    .unwrap();
    assert_eq!(
        prepared
            .prepare_tool_response("hello world hello world")
            .unwrap()
            .tokens(),
        [30, 31, 30, 31]
    );
    assert_eq!(
        tokenizer
            .encode("hello world hello world", false)
            .unwrap()
            .get_ids(),
        [30, 31, 30, 12, 12, 12, 12, 12]
    );
    assert!(matches!(
        prepare(128, 3).prepare_tool_response("hello world hello world"),
        Err(VoiceChatTextError::TokenCapacity {
            kind: VoiceChatTextKind::ToolResponse,
            required: 4,
            capacity: 3
        })
    ));
    assert!(matches!(
        prepare(10, 8).prepare_system_prompt(
            "hello world",
            PromptPolicy {
                repeat: 1,
                max_tokens: 16
            }
        ),
        Err(VoiceChatTextError::ByteCapacity {
            kind: VoiceChatTextKind::SystemPrompt,
            required: 11,
            capacity: 10
        })
    ));
}

#[test]
fn ascii_errors_identify_original_input_offset_and_ids_stay_in_shared_domain() {
    let prepared = prepare(128, 8);
    assert!(matches!(
        prepared.prepare_system_prompt(
            "hi é",
            PromptPolicy {
                repeat: 1,
                max_tokens: 16
            }
        ),
        Err(VoiceChatTextError::NonAscii {
            kind: VoiceChatTextKind::SystemPrompt,
            byte_offset: 3
        })
    ));
    assert!(matches!(
        prepared.prepare_tool_response("a🙂"),
        Err(VoiceChatTextError::NonAscii {
            kind: VoiceChatTextKind::ToolResponse,
            byte_offset: 1
        })
    ));
    assert!(matches!(
        prepared.prepare_tool_response("out-of-domain"),
        Err(VoiceChatTextError::TokenDomain {
            kind: VoiceChatTextKind::ToolResponse,
            token: 131_072
        })
    ));
}

#[test]
fn system_tool_rendering_preserves_typed_input_and_expansion_bounds() {
    let prepared = prepare(256, 256);
    let policy = PromptPolicy {
        repeat: 1,
        max_tokens: 258,
    };
    assert!(matches!(
        prepared.prepare_system_prompt_with_tools("é", &[serde_json::json!(null)], policy),
        Err(VoiceChatTextError::NonAscii {
            kind: VoiceChatTextKind::SystemPrompt,
            byte_offset: 0
        })
    ));
    let tools = [serde_json::json!({
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "x".repeat(300),
            "parameters": {"type": "object", "properties": {}}
        }
    })];
    assert!(matches!(
        prepared.prepare_system_prompt_with_tools("hello", &tools, policy),
        Err(VoiceChatTextError::RenderedPromptCapacity { capacity: 256 })
    ));
}

#[test]
fn invalid_control_and_unbounded_text_configuration_fail_before_encoding() {
    let (tokenizer, mut controls) = fixture("[UNK]");
    let limits = VoiceChatTextLimits {
        max_bytes: 128,
        max_tokens: 8,
    };
    controls.bos = 2;
    assert!(matches!(
        VoiceChatTextPreparation::from_tokenizer(&tokenizer, controls, limits),
        Err(VoiceChatTextError::Control("BOS"))
    ));
    controls.bos = 1;
    controls.user_bos = 9;
    assert!(matches!(
        VoiceChatTextPreparation::from_tokenizer(&tokenizer, controls, limits),
        Err(VoiceChatTextError::Control("user BOS"))
    ));
    controls.user_bos = 3;
    for limits in [
        VoiceChatTextLimits {
            max_bytes: 0,
            max_tokens: 8,
        },
        VoiceChatTextLimits {
            max_bytes: 128,
            max_tokens: 0,
        },
    ] {
        assert!(matches!(
            VoiceChatTextPreparation::from_tokenizer(&tokenizer, controls, limits),
            Err(VoiceChatTextError::InvalidLimit(_))
        ));
    }
}

#[test]
fn tokenizer_failures_preserve_the_original_error_source() {
    let (tokenizer, controls) = fixture("missing-unknown-token");
    let original = tokenizer.encode("unlisted", false).unwrap_err();
    let prepared = VoiceChatTextPreparation::from_tokenizer(
        &tokenizer,
        controls,
        VoiceChatTextLimits {
            max_bytes: 128,
            max_tokens: 8,
        },
    )
    .unwrap();
    let failure = prepared.prepare_tool_response("unlisted").unwrap_err();
    let VoiceChatTextError::Tokenizer(backend) = failure else {
        panic!("expected preserved tokenizer failure")
    };
    assert_eq!(backend.operation(), "VoiceChat text preparation");
    let mut source = backend.source().unwrap();
    while let Some(next) = source.source() {
        source = next;
    }
    assert_eq!(source.to_string(), original.to_string());
}

#[test]
fn normalization_cannot_admit_an_empty_response_injection() {
    let (mut tokenizer, controls) = fixture("[UNK]");
    tokenizer
        .with_normalizer(Some(
            tokenizers::normalizers::replace::Replace::new("hello", "").unwrap(),
        ))
        .unwrap();
    let prepared = VoiceChatTextPreparation::from_tokenizer(
        &tokenizer,
        controls,
        VoiceChatTextLimits {
            max_bytes: 128,
            max_tokens: 8,
        },
    )
    .unwrap();
    assert!(matches!(
        prepared.prepare_tool_response("hello"),
        Err(VoiceChatTextError::EmptyToolResponse)
    ));
}
