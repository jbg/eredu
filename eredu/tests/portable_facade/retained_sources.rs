use super::*;
use eredu::api::{ChatSourceInput, TokenizerSourceInput};
use eredu::runtime::chat::{ChatTemplateRequest, ToolChoice};
use eredu_runtime::working_memory::MemoryLedger;
use eredu_text::tokenizer::{ChatTemplateIdentity, ModelChatTemplate};

#[test]
fn retained_metaspace_decoder_runs_public_plain_source_without_reconstruction() {
    use eredu::api::{
        ManagedPlainTextRequest, PreparedChatGenerationSettings, TextInferencePolicy,
    };
    for scheme in ["always", "first", "never"] {
        let pool = crate::memory::host_ledger(original_sources::CAPACITY, 0).unwrap();
        let config = serde_json::json!({"version":"1.0", "truncation":null, "padding":null,
            "added_tokens":[], "normalizer":null,
            "pre_tokenizer":{"type":"Metaspace", "replacement":"▁", "prepend_scheme":"first", "split":true},
            "post_processor":null,
            "decoder":{"type":"Metaspace", "replacement":"▁", "prepend_scheme":scheme, "split":true},
            "model":{"type":"WordLevel", "vocab":{"[UNK]":0,"▁hello":2,"▁world":7}, "unk_token":"[UNK]"}});
        let tokenizer = ChatTokenizer::from_bytes(config.to_string().as_bytes()).unwrap();
        let expected = tokenizer.decode(&[2, 7], true).unwrap();
        let backend = MockBackend::default();
        backend.calls.borrow_mut().pool = Some(pool.clone());
        backend.calls.borrow_mut().scripted_tokens.extend([2, 7]);
        let mut model = LoadedModel::from_runtime(
            ModelRuntime::prepare(backend, ()).unwrap(),
            tokenizer,
            LoadedTextModelConfig {
                model_family: ModelKind::Qwen2,
                effective_model_type: "qwen2".into(),
                model_id: "metaspace-public-source".into(),
                chat_template: None,
                eos_token_ids: vec![],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        assert_eq!(model.encode("hello world", false).unwrap(), [2, 7]);
        let source = model
            .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                memory_limits: crate::memory::limits(original_sources::CAPACITY),
                ..Default::default()
            },
            ..Default::default()
        };
        let cancel = eredu_core::GenerationCancellationToken::new();
        let output = model
            .generate_managed_plain_text(
                &source,
                ManagedPlainTextRequest::new("hello world", settings),
                &cancel,
                &mut |_| {},
            )
            .unwrap()
            .unwrap();
        assert_eq!(output.token_ids.as_ref(), &[2, 7]);
        assert_eq!(output.text.as_ref(), expected);
        drop((source, model));
        assert!(
            pool.live_charge_bytes().unwrap() > 0,
            "escaped output retains its actual source"
        );
        drop(output);
        assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    }
}

fn embedded_model(pool: &MemoryLedger) -> LoadedModel<MockBackend> {
    use eredu_gguf::{MetadataArray as A, MetadataValue as V};
    let metadata = std::collections::HashMap::from([
        ("general.architecture".into(), V::String("qwen3".into())),
        ("tokenizer.ggml.model".into(), V::String("gpt2".into())),
        (
            "tokenizer.ggml.tokens".into(),
            V::Array(A::String(
                ["<eos>", "h", "e", "l", "o", "he", "ll", "hell", "hello"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            )),
        ),
        (
            "tokenizer.ggml.merges".into(),
            V::Array(A::String(
                ["h e", "l l", "he ll", "hell o"]
                    .into_iter()
                    .map(str::to_owned)
                    .collect(),
            )),
        ),
        ("tokenizer.ggml.eos_token_id".into(), V::Uint32(0)),
        ("tokenizer.ggml.add_eos_token".into(), V::Bool(true)),
    ]);
    let loaded = eredu_text::gguf::from_metadata(&metadata).unwrap().unwrap();
    let mut tokenizer = ChatTokenizer::from_tokenizer(loaded.tokenizer);
    tokenizer.set_template_kwargs(loaded.template_kwargs);
    let backend = MockBackend::default();
    backend.calls.borrow_mut().pool = Some(pool.clone());
    LoadedModel::from_runtime(
        ModelRuntime::prepare(backend, ()).unwrap(), tokenizer,
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen3".into(), model_id: "embedded-complete-source".into(),
            chat_template: Some(ModelChatTemplate::Named(std::collections::BTreeMap::from([
                ("default".into(), "{{ messages[0].content }}{% if add_generation_prompt %}|assistant{% endif %}".into()),
                ("tool_use".into(), "tool:{{ messages[0].content }}{% if add_generation_prompt %}|assistant{% endif %}".into()),
            ]))),
            eos_token_ids: vec![0], checkpoint_generation_config: None,
        },
    ).unwrap()
}

#[test]
fn ordered_normalization_runs_public_plain_source_with_exact_retained_configuration() {
    use eredu::api::{
        ManagedPlainTextRequest, PreparedChatGenerationSettings, TextInferencePolicy,
    };
    for normalizer in [
        serde_json::json!({"type":"Lowercase"}),
        serde_json::json!({"type":"Sequence","normalizers":[
            {"type":"Prepend","prepend":"É"},
            {"type":"Sequence","normalizers":[{"type":"Lowercase"},{"type":"NFC"}]},
            {"type":"Replace","pattern":{"String":"é"},"content":""}]}),
    ] {
        let pool = crate::memory::host_ledger(original_sources::CAPACITY, 0).unwrap();
        let config = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":normalizer,
            "pre_tokenizer":{"type":"Whitespace"},"post_processor":null,
            "decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true},
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":2,"world":7,"▁world":8},"unk_token":"[UNK]"}});
        let tokenizer = ChatTokenizer::from_bytes(config.to_string().as_bytes()).unwrap();
        let backend = MockBackend::default();
        backend.calls.borrow_mut().pool = Some(pool.clone());
        backend.calls.borrow_mut().scripted_tokens.extend([2, 8]);
        let mut model = LoadedModel::from_runtime(
            ModelRuntime::prepare(backend, ()).unwrap(),
            tokenizer,
            LoadedTextModelConfig {
                model_family: ModelKind::Qwen2,
                effective_model_type: "qwen2".into(),
                model_id: "ordered-normalization-source".into(),
                chat_template: None,
                eos_token_ids: vec![],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        assert_eq!(model.encode("HELLO WORLD", false).unwrap(), [2, 7]);
        let source = model
            .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(2),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                memory_limits: crate::memory::limits(original_sources::CAPACITY),
                ..Default::default()
            },
            ..Default::default()
        };
        let output = model
            .generate_managed_plain_text(
                &source,
                ManagedPlainTextRequest::new("HELLO WORLD", settings),
                &eredu_core::GenerationCancellationToken::new(),
                &mut |_| {},
            )
            .unwrap()
            .unwrap();
        assert_eq!(output.token_ids.as_ref(), [2, 8]);
        assert_eq!(output.text.as_ref(), "hello world");
        drop((source, model));
        assert!(pool.live_charge_bytes().unwrap() > 0);
        drop(output);
        assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    }
}

#[test]
fn ordered_pretokenizers_run_public_plain_source_with_exact_retained_configuration() {
    use eredu::api::{
        ManagedPlainTextRequest, PreparedChatGenerationSettings, TextInferencePolicy,
    };
    let pool = crate::memory::host_ledger(original_sources::CAPACITY, 0).unwrap();
    let config = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "added_tokens":[],"normalizer":{"type":"Lowercase"},
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[
            {"type":"Whitespace"},{"type":"Sequence","pretokenizers":[
                {"type":"Digits","individual_digits":false},{"type":"Whitespace"},
                {"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true}]}]},
        "post_processor":null,
        "decoder":{"type":"Metaspace","replacement":"▁","prepend_scheme":"never","split":true},
        "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":2,"12":3,"world":7,"▁world":8},"unk_token":"[UNK]"}});
    let tokenizer = ChatTokenizer::from_bytes(config.to_string().as_bytes()).unwrap();
    let backend = MockBackend::default();
    backend.calls.borrow_mut().pool = Some(pool.clone());
    backend.calls.borrow_mut().scripted_tokens.extend([2, 8]);
    let mut model = LoadedModel::from_runtime(
        ModelRuntime::prepare(backend, ()).unwrap(),
        tokenizer,
        LoadedTextModelConfig {
            model_family: ModelKind::Qwen2,
            effective_model_type: "qwen2".into(),
            model_id: "ordered-pretokenizer-source".into(),
            chat_template: None,
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    assert_eq!(model.encode("HELLO12 WORLD", false).unwrap(), [2, 3, 7]);
    let source = model
        .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(2),
            ..Default::default()
        },
        inference: TextInferencePolicy {
            memory_limits: crate::memory::limits(original_sources::CAPACITY),
            ..Default::default()
        },
        ..Default::default()
    };
    let output = model
        .generate_managed_plain_text(
            &source,
            ManagedPlainTextRequest::new("HELLO12 WORLD", settings),
            &eredu_core::GenerationCancellationToken::new(),
            &mut |_| {},
        )
        .unwrap()
        .unwrap();
    assert_eq!(output.token_ids.as_ref(), [2, 8]);
    assert_eq!(output.text.as_ref(), "hello world");
    drop((source, model));
    assert!(pool.live_charge_bytes().unwrap() > 0);
    drop(output);
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}

#[test]
fn regex_feature_selected_sources_preserve_public_plain_configuration_and_word_semantics() {
    use eredu::api::{
        ManagedPlainTextRequest, PreparedChatGenerationSettings, TextInferencePolicy,
    };
    let variants = [
        serde_json::json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true}),
        serde_json::json!({"type":"Sequence","pretokenizers":[
            {"type":"Split","pattern":{"Regex":"\\w+|[^\\w\\s]+"},"behavior":"Isolated","invert":false},
            {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]}),
        serde_json::json!({"type":"Whitespace"}),
    ];
    for (index, pre) in variants.into_iter().enumerate() {
        let pool = crate::memory::host_ledger(original_sources::CAPACITY, 0).unwrap();
        let config = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
            "added_tokens":[],"normalizer":null,"pre_tokenizer":pre,"post_processor":null,
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
            "model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":2,"a²b":3,"a":4,"²":5,"b":6,"a\u{200c}b":7,"\u{200c}":8,"'²":9,"s":10,"a¼b":11,"¼":12,"'¼":13},"unk_token":"[UNK]"}});
        let tokenizer = ChatTokenizer::from_bytes(config.to_string().as_bytes()).unwrap();
        let backend = MockBackend::default();
        backend.calls.borrow_mut().pool = Some(pool.clone());
        backend.calls.borrow_mut().scripted_tokens.extend([2]);
        let mut model = LoadedModel::from_runtime(
            ModelRuntime::prepare(backend, ()).unwrap(),
            tokenizer,
            LoadedTextModelConfig {
                model_family: ModelKind::Qwen2,
                effective_model_type: "qwen2".into(),
                model_id: "regex-feature-source".into(),
                chat_template: None,
                eos_token_ids: vec![],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        if index == 2 {
            assert_eq!(
                model.encode("a²b '²s a¼b '¼s a\u{200c}b", false).unwrap(),
                if cfg!(feature = "onig") {
                    vec![3, 9, 10, 11, 13, 10, 4, 8, 6]
                } else {
                    vec![4, 5, 6, 9, 10, 4, 12, 6, 13, 10, 7]
                }
            );
        }
        let source = model
            .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
            .unwrap();
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(1),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                memory_limits: crate::memory::limits(original_sources::CAPACITY),
                ..Default::default()
            },
            ..Default::default()
        };
        let output = model
            .generate_managed_plain_text(
                &source,
                ManagedPlainTextRequest::new(
                    if index == 2 {
                        "a²b '²s a¼b '¼s a\u{200c}b"
                    } else {
                        "hello"
                    },
                    settings,
                ),
                &eredu_core::GenerationCancellationToken::new(),
                &mut |_| {},
            )
            .unwrap()
            .unwrap();
        assert_eq!(output.token_ids.as_ref(), [2]);
        assert_eq!(output.text.as_ref(), "hello");
        drop((source, model));
        assert!(pool.live_charge_bytes().unwrap() > 0);
        drop(output);
        assert_eq!(pool.live_charge_bytes().unwrap(), 0);
    }
}

#[test]
fn retained_embedded_configuration_compiles_without_artifacts_and_matches_selected_template() {
    let pool = crate::memory::host_ledger(8 * 1024 * 1024 * 1024, 0).unwrap();
    let mut model = embedded_model(&pool);
    assert_eq!(model.encode("hello", true).unwrap(), [8, 0]);
    let baseline = pool.live_charge_bytes().unwrap();
    let tokenizer = model
        .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
        .unwrap();
    let cancel = eredu_core::GenerationCancellationToken::new();
    let default = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancel,
        )
        .unwrap()
        .unwrap();
    let tools = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            true,
            &cancel,
        )
        .unwrap()
        .unwrap();
    let request = ChatTemplateRequest {
        messages: vec![serde_json::json!({"role":"user","content":"hello"})],
        add_generation_prompt: true,
        tool_choice: ToolChoice::None,
        ..Default::default()
    };
    let chat = model
        .prepare_chat(
            &default,
            &request,
            &crate::memory::limits(original_sources::CAPACITY),
            &cancel,
        )
        .unwrap()
        .unwrap();
    assert_eq!(chat.rendered_prompt(), "hello|assistant");
    assert_eq!(
        chat.template_identity(),
        &ChatTemplateIdentity::Named("default".into())
    );
    assert_eq!(
        model
            .prepare_chat(
                &tools,
                &request,
                &crate::memory::limits(original_sources::CAPACITY),
                &cancel
            )
            .unwrap_err()
            .input_rejection(),
        Some(eredu_core::TokenInputRejection::IdentityMismatch)
    );
    drop(chat);
    model.set_chat_template(Some("replacement {{ messages[0].content }}".into()));
    assert_eq!(
        model
            .prepare_chat(
                &default,
                &request,
                &crate::memory::limits(original_sources::CAPACITY),
                &cancel
            )
            .unwrap_err()
            .input_rejection(),
        Some(eredu_core::TokenInputRejection::IdentityMismatch)
    );
    let replaced = model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancel,
        )
        .unwrap()
        .unwrap();
    let chat = model
        .prepare_chat(
            &replaced,
            &request,
            &crate::memory::limits(original_sources::CAPACITY),
            &cancel,
        )
        .unwrap()
        .unwrap();
    assert_eq!(chat.rendered_prompt(), "replacement hello");
    drop((chat, replaced, tools, default, tokenizer));
    assert_eq!(pool.live_charge_bytes().unwrap(), baseline);
}

#[test]
fn retained_sources_preserve_cancellation_foreign_pool_and_original_refusal() {
    let pool = crate::memory::host_ledger(8 * 1024 * 1024 * 1024, 0).unwrap();
    let model = embedded_model(&pool);
    let loaded_bytes = pool.live_charge_bytes().unwrap();
    let loading_peak = pool.payload_peak_bytes().unwrap();
    let tokenizer = model
        .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
        .unwrap();
    let cancel = eredu_core::GenerationCancellationToken::new();
    cancel.cancel();
    let before = pool.live_charge_bytes().unwrap();
    assert!(model
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &cancel
        )
        .unwrap()
        .is_none());
    assert_eq!(pool.live_charge_bytes().unwrap(), before);
    let other = embedded_model(&crate::memory::host_ledger(8 * 1024 * 1024 * 1024, 0).unwrap());
    assert!(other
        .compile_managed_chat_source(
            &tokenizer,
            ChatSourceInput::RetainedConfiguration,
            false,
            &Default::default()
        )
        .is_err());
    // The exact loading peak includes temporary metadata. The later source
    // compiler must obtain an additional grant before serializing its input.
    let small = crate::memory::host_ledger(loading_peak, 0).unwrap();
    let model = embedded_model(&small);
    let error = model
        .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
        .unwrap_err();
    let eredu::api::ManagedPlainTextSourceError::Source(
        eredu_runtime::working_memory::OriginalTokenizerSourceError::Input(failure),
    ) = &error
    else {
        panic!("unexpected source refusal: {error:?}")
    };
    assert!(failure.accounting_failure().is_some());
    assert_eq!(failure.input_capacity(), 0);
    assert_eq!(small.live_charge_bytes().unwrap(), loaded_bytes);
}

#[test]
fn retained_gemma_configuration_preserves_nontrivial_embedded_score_bits() {
    use eredu_gguf::{MetadataArray as A, MetadataValue as V};
    let mut tokens = [
        "<unk>", "<eos>", "▁", "h", "e", "l", "o", "hello", "▁hello", "\n",
    ]
    .map(str::to_owned)
    .to_vec();
    tokens.extend((0..=255).map(|byte| format!("<0x{byte:02X}>")));
    let mut state = 0x92d68ca2u32;
    let scores: Vec<f32> = tokens
        .iter()
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            -f32::from_bits((126 << 23) | (state & 0x7f_ffff))
        })
        .collect();
    let metadata = std::collections::HashMap::from([
        ("general.architecture".into(), V::String("gemma4".into())),
        ("tokenizer.ggml.model".into(), V::String("llama".into())),
        ("tokenizer.ggml.tokens".into(), V::Array(A::String(tokens))),
        ("tokenizer.ggml.scores".into(), V::Array(A::Float32(scores))),
        ("tokenizer.ggml.unknown_token_id".into(), V::Uint32(0)),
        ("tokenizer.ggml.eos_token_id".into(), V::Uint32(1)),
    ]);
    let loaded = eredu_text::gguf::from_metadata(&metadata).unwrap().unwrap();
    let mut tokenizer = ChatTokenizer::from_tokenizer(loaded.tokenizer);
    tokenizer.set_template_kwargs(loaded.template_kwargs);
    let expected = tokenizer
        .encode("hello\nhello", true)
        .unwrap()
        .get_ids()
        .to_vec();
    let pool = crate::memory::host_ledger(8 * 1024 * 1024 * 1024, 0).unwrap();
    let backend = MockBackend::default();
    backend.calls.borrow_mut().pool = Some(pool.clone());
    let model = LoadedModel::from_runtime(
        ModelRuntime::prepare(backend, ()).unwrap(),
        tokenizer,
        LoadedTextModelConfig {
            model_family: ModelKind::Gemma4,
            effective_model_type: "gemma4".into(),
            model_id: "embedded-scored-source".into(),
            chat_template: None,
            eos_token_ids: vec![1],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    assert_eq!(model.encode("hello\nhello", true).unwrap(), expected);
    let before = pool.live_charge_bytes().unwrap();
    // Public source publication compares every actual f64 score bit, including
    // these exact f32-to-f64 metadata conversions; equality is never relaxed.
    let source = model
        .compile_managed_plain_text_source(TokenizerSourceInput::RetainedConfiguration)
        .unwrap();
    assert!(pool.live_charge_bytes().unwrap() > before);
    drop(source);
    assert_eq!(pool.live_charge_bytes().unwrap(), before);
}

/// Source-only qualification: no tensor payload is read or executed. Set the
/// exact pinned artifact path explicitly; the cache remains read-only.
#[test]
#[ignore = "requires EREDU_PINNED_GGUF_SOURCE with separately verified provenance"]
fn pinned_gguf_retained_source_preserves_ids_template_and_retirement() {
    use eredu_runtime::working_memory::OriginalTokenizerInput;
    use eredu_text::chat_storage::{ChatRenderContext, ChatTemplatePlan};
    let path = std::env::var_os("EREDU_PINNED_GGUF_SOURCE").expect("pinned GGUF path");
    let start = std::time::Instant::now();
    let metadata = eredu_gguf::Reader::open(path)
        .unwrap()
        .into_metadata()
        .into_iter()
        .collect();
    let loaded = eredu_text::gguf::from_metadata(&metadata).unwrap().unwrap();
    let template = ModelChatTemplate::Single(
        metadata
            .get("tokenizer.chat_template")
            .and_then(eredu_gguf::MetadataValue::as_str)
            .unwrap()
            .to_owned(),
    );
    let mut selected = ChatTokenizer::from_tokenizer(loaded.tokenizer);
    selected.set_template_kwargs(loaded.template_kwargs);
    let load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pool = crate::memory::host_ledger(u64::MAX, 0).unwrap();
    let start = std::time::Instant::now();
    let source = pool
        .compile_tokenizer_source_for_generation(OriginalTokenizerInput::Configuration(&selected))
        .unwrap();
    let compile_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert!(source.matches_configuration(&selected));
    let texts = [
        "",
        "Hello, world!",
        "1234567890\r\n",
        "Héllo 世界 العربية",
        "e\u{301} café 👩\u{200d}💻",
        "<|im_start|>user\nhello<|im_end|>",
    ];
    for text in texts {
        for add_special in [false, true] {
            let expected = selected.encode(text, add_special).unwrap();
            let actual = pool
                .encode_tokenizer_ids(&source, text, add_special)
                .unwrap();
            assert_eq!(
                actual.ids(),
                expected.get_ids(),
                "{text:?}, special={add_special}"
            );
        }
    }
    let start = std::time::Instant::now();
    let chat = pool
        .compile_chat_template(
            ChatTemplatePlan::prepare_model(&template, "pinned-embedded-source", false).unwrap(),
        )
        .unwrap();
    let template_ms = start.elapsed().as_secs_f64() * 1000.0;
    assert!(chat.matches_configuration(&template, "pinned-embedded-source"));
    let messages = vec![serde_json::json!({"role":"user","content":"Héllo 世界"})];
    let expected: Vec<_> = [false, true]
        .into_iter()
        .map(|generation| {
            selected
                .apply_chat_template_json(
                    template.clone(),
                    [messages.clone()],
                    None,
                    "pinned-embedded-source",
                    generation,
                    None,
                )
                .unwrap()
                .remove(0)
        })
        .collect();
    let rendered = pool
        .render_original_chat(
            &chat,
            &source,
            ChatRenderContext::from_json(&messages, Some(selected.template_kwargs()), None)
                .unwrap(),
        )
        .unwrap();
    for (generation, expected) in [false, true].into_iter().zip(&expected) {
        assert_eq!(rendered.prompt(generation), expected);
    }
    println!(
        "source_only load_ms={load_ms:.3} compile_ms={compile_ms:.3} template_ms={template_ms:.3} token_count={} source_bytes={} template_bytes={} peak_bytes={} id_comparisons={} render_comparisons=2 prefix_error_bytes={}",
        source.token_count(),
        source.original_bytes(),
        chat.original_bytes(),
        pool.payload_peak_bytes().unwrap(),
        texts.len() * 2,
        std::mem::size_of::<eredu_runtime::working_memory::OriginalTokenizerPrefixError>()
    );
    drop((source, chat, rendered));
    assert_eq!(pool.live_charge_bytes().unwrap(), 0);
}
