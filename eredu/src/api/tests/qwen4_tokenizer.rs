//! Opt-in real released BPE validation; the artifact stays outside the tracked tree.
use super::*;

#[test]
#[ignore = "requires pinned Flash-Next tokenizer sidecars in EREDU_QWEN4_TOKENIZER_DIR"]
fn qwen4_exp_released_tokenizer_accepts_atomic_tagged_tools_and_rejects_bad_arguments() {
    let directory = std::env::var_os("EREDU_QWEN4_TOKENIZER_DIR")
        .map(std::path::PathBuf::from)
        .expect("set EREDU_QWEN4_TOKENIZER_DIR to tokenizer.json and sidecars from Qwen/Qwen3.8-Flash-Next revision de4b8e4d43b917e7706784d8bb445c9af86a3540; model weights are not required");
    let raw = load_tokenizer(&directory).expect("load pinned released Flash-Next tokenizer");
    let eos = eos_token_ids_from_sidecar_dir(&directory).unwrap();
    assert_eq!(eos, [248044, 248046]);
    for (spelling, id) in [
        ("<|endoftext|>", 248044),
        ("<|im_start|>", 248045),
        ("<|im_end|>", 248046),
        ("<tool_call>", 248058),
        ("</tool_call>", 248059),
    ] {
        assert_eq!(raw.token_to_id(spelling), Some(id));
        assert_eq!(raw.encode(spelling, false).unwrap().get_ids(), &[id]);
    }
    let mut tokenizer = ChatTokenizer::from_tokenizer(raw.clone());
    tokenizer.set_template_kwargs(load_tokenizer_template_kwargs(&directory).unwrap());
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos);
    assert!(
        compiler.is_ok(),
        "real tokenizer compiler: {:?}",
        compiler.as_ref().err()
    );
    let prepared = prepare_chat_from_parts(
        &mut tokenizer,
        load_chat_template(&directory)
            .unwrap()
            .expect("pinned chat template"),
        "Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540",
        &eos,
        Some(&compiler),
        ChatTemplateRequest {
            messages: vec![json!({"role": "user", "content": "Look up 12345."})],
            tools: vec![production_tool("lookup")],
            tool_choice: ToolChoice::Required,
            enable_thinking: Some(false),
            add_generation_prompt: true,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        prepared.format_profile_identity(),
        Some("qwen3.8.tagged-parameter-tools.v1")
    );
    assert!(matches!(
        prepared.native_tool_support(),
        NativeToolSupport::Supported
    ));
    assert!(prepared.preserved_structural_token_ids().contains(&248046));
    assert_eq!(prepared.profile_stop_sequences(), ["<|im_end|>"]);
    let plan = prepared.tool_runtime_plan().unwrap();
    let output = concat!(
        "<tool_call>\n<function=lookup>\n<parameter=value>\n12345\n",
        "</parameter>\n</function>\n</tool_call>"
    );
    let ids = raw.encode(output, false).unwrap();
    assert!(ids.get_ids().contains(&248058));
    assert!(ids.get_ids().contains(&248059));
    assert_eq!(raw.decode(ids.get_ids(), false).unwrap(), output);
    let mut grammar = plan.generation_constraint().grammar_state();
    for (offset, &id) in ids.get_ids().iter().enumerate() {
        grammar
            .commit(id)
            .unwrap_or_else(|error| panic!("released BPE token {offset} ({id}) rejected: {error}"));
    }
    assert!(grammar.is_complete().unwrap());

    let malformed = output.replace("12345", "not_an_integer");
    let mut grammar = plan.generation_constraint().grammar_state();
    assert!(
        raw.encode(malformed, false)
            .unwrap()
            .get_ids()
            .iter()
            .any(|&id| grammar.commit(id).is_err()),
        "integer schema must reject a text argument"
    );

    let split = output.find("345").unwrap();
    let prefix = &output[..split];
    let mut incomplete_grammar = plan.generation_constraint().grammar_state();
    for &id in raw.encode(prefix, false).unwrap().get_ids() {
        incomplete_grammar.commit(id).unwrap();
    }
    assert!(!incomplete_grammar.is_complete().unwrap());
    let mut parser = plan.create_parser().unwrap();
    parser.push(prefix).unwrap();
    let mut fork = parser.fork().unwrap();
    let mut incomplete = parser.fork().unwrap();
    for parser in [&mut parser, &mut fork] {
        parser.push(&output[split..]).unwrap();
        parser.push("<|im_end|>").unwrap();
        parser.finish(FinishReason::Eos).unwrap();
        let arguments = tool_argument_events(parser.events());
        assert_eq!(arguments.len(), 1);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&arguments[0]).unwrap(),
            json!({"value": 12345})
        );
        assert!(parser.events().contains(&SemanticEvent::ToolCallEnd));
    }
    assert_eq!(parser.events(), fork.events());
    assert_eq!(
        incomplete.finish(FinishReason::MaxTokens).unwrap_err(),
        "incomplete tagged-parameter tool call"
    );
    assert!(!incomplete.events().contains(&SemanticEvent::ToolCallEnd));
}
