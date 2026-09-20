//! Source-owned compiler and controller conformance, with direct dependency
//! parser oracles for token masks and fixed expected activation boundaries.
use super::*;
use crate::runtime::chat::dialect::{
    DECLARATIVE_DIALECT, DeclarativeDialectSpec, DeclarativePayloadShape, ExactEnvelope,
    GenerationPromptBehavior, JsonFunctionEnvelope, ParallelCallLayout,
};
use std::sync::atomic::AtomicUsize;
use tokenizers::{AddedToken, decoders::byte_level::ByteLevel, models::bpe::BPE};

const FUNCTION: JsonFunctionEnvelope = JsonFunctionEnvelope {
    envelope: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    name_field: "name",
    arguments_field: "arguments",
    call_id: None,
};
const SPEC: DeclarativeDialectSpec = DeclarativeDialectSpec {
    generation_prompt_behavior: GenerationPromptBehavior::HonorRequest,
    reasoning_template_kwarg: "enable_thinking",
    supports_tool_reasoning: true,
    output: ExactEnvelope {
        prefix: r#"{"calls":"#,
        suffix: "}",
    },
    call: ExactEnvelope {
        prefix: "",
        suffix: "",
    },
    payload_shape: DeclarativePayloadShape::JsonList,
    json_function: Some(&FUNCTION),
    reasoning_channel: None,
    text_channel: None,
    raw_text_before_calls: false,
    call_separator: ",",
    parallel_layout: ParallelCallLayout::SingleEnvelope,
    protocol_max_tools: None,
    protocol_max_calls: None,
    auto_activation_trigger: Some(r#"{"calls":"#),
    required_structural_tokens: &[],
    stop_sequences: &[],
};
pub(super) const PARAMETERS: DialectParameters = DialectParameters::Declarative(&SPEC);

fn tokenizer() -> (ChatTokenizer, [u32; 2]) {
    let vocabulary = (b'!'..=b'~')
        .map(|byte| char::from(byte).to_string())
        .chain(["Ġ".to_owned()])
        .enumerate()
        .map(|(id, token)| (token, id as u32))
        .collect::<tokenizers::models::bpe::Vocab>();
    let model = BPE::builder()
        .vocab_and_merges(vocabulary, Vec::new())
        .build()
        .unwrap();
    let mut raw = tokenizers::Tokenizer::new(model);
    raw.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    raw.with_decoder(Some(ByteLevel::default()));
    raw.add_special_tokens([
        AddedToken::from("<eos>", true).normalized(false),
        AddedToken::from("<|end|>", true).normalized(false),
    ])
    .unwrap();
    let eos = [
        raw.token_to_id("<eos>").unwrap(),
        raw.token_to_id("<|end|>").unwrap(),
    ];
    (ChatTokenizer::from_tokenizer(raw), eos)
}

fn tool(parameters: Value) -> Value {
    json!({"type":"function", "function":{"name":"check", "parameters":parameters}})
}

pub(super) fn ordinary_tools() -> Vec<Value> {
    vec![tool(
        json!({"type":"object", "properties":{"value":{"enum":[17,23]}},
        "required":["value"], "additionalProperties":false}),
    )]
}

fn plan(
    compiler: &ConstraintCompiler,
    tools: &[Value],
    choice: ToolChoice,
) -> GenerationRuntimePlan {
    compiler
        .compile_generation_plan(
            &DECLARATIVE_DIALECT,
            PARAMETERS,
            tools,
            choice,
            ParallelToolCallPolicy::Disabled,
            vec!["<|end|>".to_owned()],
            vec![compiler.eos_token_ids[1]],
            vec!["<|end|>".to_owned()],
            true,
        )
        .unwrap()
}

/// Source-owned fixture construction, using the same prospective compiler
/// contract as public preparation. Draft selection is explicit in these fixtures;
/// public default-draft behavior has its own conformance cases.
fn original_plan(
    pool: &eredu_runtime::working_memory::WorkingMemoryPool,
    tokenizer: &ChatTokenizer,
    eos: &[u32; 2],
    tools: &[Value],
    choice: ToolChoice,
) -> (
    GenerationRuntimePlan,
    eredu_runtime::working_memory::OriginalControllerCompilation,
) {
    use eredu_runtime::working_memory::{
        ControllerCompilationOutput, ControllerCompilationSources, InferenceExecutionIdentity,
        OriginalChatProfilePreparation, OriginalControllerCompiler, OriginalTokenizer,
    };
    struct Output(GenerationRuntimePlan);
    impl ControllerCompilationOutput for Output {
        fn controller_sources(&self) -> ControllerCompilationSources<'_> {
            self.0.controller_sources()
        }
    }
    #[derive(Debug, thiserror::Error)]
    enum Failure {
        #[error(transparent)]
        Source(#[from] ConstraintCompilerSourceError),
        #[error(transparent)]
        Plan(#[from] preparation_error::PreparationFailure),
        #[error(transparent)]
        Funding(#[from] eredu_core::HostMetadataFundingError),
    }
    struct Compile<'a> {
        tokenizer: OriginalTokenizer,
        eos: &'a [u32; 2],
        tools: &'a [Value],
        choice: ToolChoice,
    }
    impl OriginalControllerCompiler for Compile<'_> {
        type Output = Output;
        type Error = Failure;
        fn compile(self, funding: &eredu_core::HostMetadataFunding) -> Result<Output, Failure> {
            let compiler =
                ConstraintCompiler::from_original_tokenizer(self.tokenizer, self.eos, funding)?;
            // Actual fixture-owned structural ID, spelling and stop vectors,
            // including the string allocations moved into the semantic plan.
            funding.reserve_metadata(
                2 * std::mem::size_of::<Vec<String>>()
                    + std::mem::size_of::<Vec<u32>>()
                    + 2 * std::mem::size_of::<String>()
                    + std::mem::size_of::<u32>()
                    + 2 * "<|end|>".len(),
            )?;
            Ok(Output(compiler.compile_generation_plan(
                &DECLARATIVE_DIALECT,
                PARAMETERS,
                self.tools,
                self.choice,
                ParallelToolCallPolicy::Disabled,
                vec!["<|end|>".into()],
                vec![self.eos[1]],
                vec!["<|end|>".into()],
                true,
            )?))
        }
    }
    let source = tokenizer.to_string(false).unwrap();
    let source = pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(source.as_bytes())
                .unwrap()
                .with_encode_special_tokens(tokenizer.get_encode_special_tokens()),
        )
        .unwrap();
    let template = pool
        .compile_chat_template(
            eredu_text::chat_storage::ChatTemplatePlan::prepare_utf8("fixture", "fixture").unwrap(),
        )
        .unwrap();
    let preparation = OriginalChatProfilePreparation::new(
        &template,
        &source,
        &InferenceExecutionIdentity::default(),
        pool.effective_capacity().unwrap(),
    )
    .unwrap();
    let mut tools = tools.to_vec();
    for tool in &mut tools {
        if let Some(parameters) = tool["function"]["parameters"].as_object_mut() {
            parameters.insert(
                "$schema".into(),
                Value::String("http://json-schema.org/draft-07/schema#".into()),
            );
        }
    }
    let (output, receipt) = preparation
        .compile_controller(Compile {
            tokenizer: source,
            eos,
            tools: &tools,
            choice,
        })
        .unwrap();
    (output.0, receipt)
}

fn commit_text(
    mut state: grammar_source::OriginalGrammarState,
    text: &str,
) -> grammar_source::OriginalGrammarState {
    for &byte in text.as_bytes() {
        let token = state
            .parser()
            .vocabulary()
            .trie_source()
            .trie()
            .token_id_at_bytes(&[byte])
            .unwrap();
        state = state.compute_mask().unwrap();
        assert!(
            state.token_mask().unwrap().is_allowed(token),
            "token {token}, text {text:?}"
        );
        state = state.commit(token).unwrap();
    }
    state
}

struct DropWitness(Arc<AtomicUsize>);
impl Drop for DropWitness {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}
fn authority(drops: &Arc<AtomicUsize>) -> HostPreparationAuthority {
    HostPreparationAuthority::retain(DropWitness(drops.clone()))
}

#[test]
fn source_plan_clones_and_paid_parser_forks_retain_exact_declarations() {
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
    let pool = WorkingMemoryPool::new(1 << 30, 0).unwrap();
    let (tokenizer, eos) = tokenizer();
    let (prepared, compilation) =
        original_plan(&pool, &tokenizer, &eos, &ordinary_tools(), ToolChoice::Auto);
    let copied = prepared.clone();
    assert_eq!(prepared, copied);
    assert!(
        prepared
            .generation_constraint()
            .inner
            .fixture_matcher
            .is_none()
    );
    let recipe = &copied.generation_constraint().inner.recipe;
    assert!(
        recipe
            .source()
            .same_storage(prepared.generation_constraint().inner.recipe.source())
    );
    assert_eq!(copied.tool_call_trigger(), Some(r#"{"calls":"#));
    assert_eq!(
        copied
            .semantic_plan()
            .structural_tokens()
            .collect::<Vec<_>>(),
        vec![(eos[1], "<|end|>")]
    );
    let bytes = recipe.source().as_ref().to_vec();
    let trie = recipe.original_trie().unwrap();
    assert_eq!(recipe.trie_info(), Some(*trie.trie().info()));
    assert_eq!(trie.trie().eos_tokens(), eos);
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 30)
        .unwrap();
    let state = copied
        .generation_constraint()
        .inner
        .original_grammar_state(&compilation, &funding)
        .unwrap();
    drop((prepared, tokenizer));
    assert_eq!(
        copied
            .generation_constraint()
            .inner
            .recipe
            .source()
            .as_ref(),
        bytes
    );
    let mut state = commit_text(state, r#"{"calls":[{"name":"check","arguments":{"value":"#);
    let mut fork = state.try_copy(&funding).unwrap();
    assert_eq!(
        state.parser().parser().final_bytes(),
        fork.parser().parser().final_bytes()
    );
    state = state.compute_mask().unwrap();
    fork = fork.compute_mask().unwrap();
    assert_eq!(state.token_mask(), fork.token_mask());
    drop((copied, compilation, funding));
    assert!(pool.used_bytes().unwrap() > 0);
    state = commit_text(state, "17}}]}");
    fork = commit_text(fork, "23}}]}");
    let (next, complete) = state.is_complete().unwrap();
    state = next;
    assert!(complete);
    let (next, complete) = fork.is_complete().unwrap();
    fork = next;
    assert!(complete);
    state = state.compute_mask().unwrap();
    fork = fork.compute_mask().unwrap();
    for token in eos {
        assert!(state.token_mask().unwrap().is_allowed(token));
        assert!(fork.token_mask().unwrap().is_allowed(token));
    }
    state = state.commit(eos[0]).unwrap();
    fork = fork.commit(eos[1]).unwrap();
    let (state, terminal) = state.is_terminal().unwrap();
    assert!(terminal);
    let (fork, terminal) = fork.is_terminal().unwrap();
    assert!(terminal);
    drop(state);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(fork);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_retains_decoder_even_when_current_spelling_map_matches() {
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
    let pool = WorkingMemoryPool::new(1 << 30, 0).unwrap();
    let (original, eos) = tokenizer();
    let tools = vec![tool(
        json!({"type":"object", "properties":{"value":{"const":"two words"}},
        "required":["value"], "additionalProperties":false}),
    )];
    let (prepared, compilation) =
        original_plan(&pool, &original, &eos, &tools, ToolChoice::Required);
    let mut other_raw = (*original).clone();
    other_raw.with_decoder(Some(
        tokenizers::decoders::byte_fallback::ByteFallback::new(),
    ));
    let other = ChatTokenizer::from_tokenizer(other_raw);
    assert_eq!(
        eredu_text::tokenizer::vocabulary_fingerprint(&original),
        eredu_text::tokenizer::vocabulary_fingerprint(&other)
    );
    let space = original.token_to_id("Ġ").unwrap();
    let other_environment =
        crate::runtime::chat::tokenizer_env::from_tokenizer(&other, &eos).unwrap();
    assert_eq!(other_environment.tok_trie().token(space), "Ġ".as_bytes());
    drop((original, other_environment, other));
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 30)
        .unwrap();
    let state = prepared
        .generation_constraint()
        .inner
        .original_grammar_state(&compilation, &funding)
        .unwrap();
    assert_eq!(
        state
            .parser()
            .vocabulary()
            .trie_source()
            .trie()
            .token(space),
        b" "
    );
    let state = commit_text(
        state,
        r#"{"calls":[{"name":"check","arguments":{"value":"two words"}}]}"#,
    );
    let (_, complete) = state.is_complete().unwrap();
    assert!(complete);
}

#[test]
fn selected_fallback_grammar_keeps_original_rejecting_schema() {
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
    let (tokenizer, eos) = tokenizer();
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos).unwrap();
    let original_tools = vec![tool(json!(false))];
    let strict = DECLARATIVE_DIALECT
        .constraint_configuration(
            PARAMETERS,
            ToolDeclarations::prepare(
                &original_tools,
                &crate::runtime::chat::preparation_memory::PreparationFunding::unmanaged(),
            )
            .unwrap()
            .as_slice(),
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            &[],
            &crate::runtime::chat::preparation_memory::PreparationFunding::unmanaged(),
        )
        .unwrap();
    assert!(
        compiler.compile_matcher(strict.grammar).is_err(),
        "fixture must select the fallback"
    );
    let pool = WorkingMemoryPool::new(1 << 30, 0).unwrap();
    let (prepared, compilation) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &original_tools,
        ToolChoice::Required,
    );
    let syntax_tools = vec![tool(json!(true))];
    let fallback = DECLARATIVE_DIALECT
        .constraint_configuration(
            PARAMETERS,
            ToolDeclarations::prepare(
                &syntax_tools,
                &crate::runtime::chat::preparation_memory::PreparationFunding::unmanaged(),
            )
            .unwrap()
            .as_slice(),
            ToolChoice::Required,
            ParallelToolCallPolicy::Disabled,
            &[],
            &crate::runtime::chat::preparation_memory::PreparationFunding::unmanaged(),
        )
        .unwrap();
    let recipe = &prepared.generation_constraint().inner.recipe;
    assert_eq!(
        serde_json::to_value(recipe.grammar().unwrap()).unwrap(),
        serde_json::to_value(fallback.grammar).unwrap()
    );
    assert_eq!(recipe.tools().unwrap(), original_tools);
    let fingerprint = prepared.generation_constraint().fingerprint;
    drop((compiler, tokenizer));
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 30)
        .unwrap();
    let grammar = prepared
        .generation_constraint()
        .inner
        .original_grammar_state(&compilation, &funding)
        .unwrap();
    let output = r#"{"calls":[{"name":"check","arguments":{}}]}"#;
    let grammar = commit_text(grammar, output);
    let (grammar, complete) = grammar.is_complete().unwrap();
    assert!(complete);
    let destination_drops = Arc::new(AtomicUsize::new(0));
    let destination = authority(&destination_drops);
    let mut semantic = prepared
        .semantic_plan()
        .create_parser_with_stops_under_authority(std::iter::empty(), &destination)
        .unwrap();
    let error = semantic.push(output).unwrap_err();
    assert!(error.contains("do not match its schema"), "{error}");
    assert!(
        !semantic
            .events()
            .contains(&eredu_core::generation::SemanticEvent::ToolCallEnd)
    );
    assert_eq!(prepared.generation_constraint().fingerprint, fingerprint);
    drop((prepared, grammar, destination));
    assert_eq!(destination_drops.load(Ordering::SeqCst), 0);
    drop(semantic);
    assert_eq!(destination_drops.load(Ordering::SeqCst), 1);
}

#[test]
fn missing_source_refusal_keeps_the_valid_recipe_unchanged() {
    use eredu_runtime::working_memory::{InferenceExecutionIdentity, WorkingMemoryPool};
    let pool = WorkingMemoryPool::new(1 << 30, 0).unwrap();
    let (tokenizer, eos) = tokenizer();
    let (prepared, compilation) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &ordinary_tools(),
        ToolChoice::Required,
    );
    let valid = &prepared.generation_constraint().inner.recipe;
    let before = valid.source().as_ref().to_vec();
    let invalid = ConstraintBlueprint {
        recipe: valid.clone(),
        fixture_matcher: None,
        declaration: None,
    };
    let funding = pool
        .prepare_workspace_metadata(&InferenceExecutionIdentity::default(), 1 << 30)
        .unwrap();
    let error = invalid
        .original_grammar_state(&compilation, &funding)
        .unwrap_err();
    assert!(error.to_string().contains("source"), "{error}");
    assert_eq!(valid.source().as_ref(), before);
    drop((invalid, error));
    let state = prepared
        .generation_constraint()
        .inner
        .original_grammar_state(&compilation, &funding)
        .unwrap();
    let state = commit_text(
        state,
        r#"{"calls":[{"name":"check","arguments":{"value":23}}]}"#,
    );
    let (_, complete) = state.is_complete().unwrap();
    assert!(complete);
}

#[test]
fn original_forbidden_startup_uses_retained_tokenizer_and_shared_branch_selection() {
    use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
    use eredu_runtime::working_memory::WorkingMemoryPool;
    #[derive(Debug)]
    struct Account(Arc<AtomicUsize>);
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            self.0.fetch_add(bytes, Ordering::SeqCst);
            Ok(())
        }
    }
    let (mut tokenizer, eos) = tokenizer();
    let pool = WorkingMemoryPool::new(1 << 26, 0).unwrap();
    let tokenizer_json = tokenizer.to_string(false).unwrap();
    let original_tokenizer = pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(tokenizer_json.as_bytes())
                .unwrap(),
        )
        .unwrap();
    let compiler = ConstraintCompiler::from_tokenizer_with_authority(
        &tokenizer,
        &eos,
        &HostPreparationAuthority::unmanaged(),
    )
    .unwrap();
    let prepared = plan(&compiler, &ordinary_tools(), ToolChoice::None);
    let automatic = plan(&compiler, &ordinary_tools(), ToolChoice::Auto);
    let required = plan(&compiler, &ordinary_tools(), ToolChoice::Required);
    let space = tokenizer.token_to_id("Ġ").unwrap();
    tokenizer
        .add_special_tokens([AddedToken::from("Ġ", true).normalized(false)])
        .unwrap();
    drop(compiler);
    let used = Arc::new(AtomicUsize::new(0));
    let funding = HostMetadataFunding::new(Account(used.clone())).unwrap();
    let mut original = ConstraintController::from_original_forbidden_generation_plan_with(
        &prepared,
        SharedTokenFilter::new(TokenFilter::All),
        32,
        &funding,
        |trigger| Ok(pool.compile_forbidden_tokenizer_source(&original_tokenizer, trigger)?),
    )
    .unwrap();
    assert!(used.load(Ordering::SeqCst) > 0);
    assert!(original.prepared_plain_source().is_none());
    let source = original.prepared_forbidden_source().unwrap();
    assert_eq!(
        source.inputs().token_bytes(space as usize),
        Some(b" ".as_slice())
    );
    let escaped = source.inputs().clone();
    let trigger = prepared.tool_call_trigger().unwrap();
    let ids: Vec<_> = trigger
        .chars()
        .map(|c| tokenizer.token_to_id(&c.to_string()).unwrap())
        .collect();
    for length in 0..ids.len() {
        let actual = source.decision_at(&ids[..length]).unwrap();
        let prefix = &trigger.as_bytes()[..length];
        for token in 0..source.inputs().vocabulary_len() as u32 {
            assert_eq!(
                actual.allows(token),
                !prefix
                    .iter()
                    .copied()
                    .chain(
                        source
                            .inputs()
                            .token_bytes(token as usize)
                            .unwrap()
                            .iter()
                            .copied()
                    )
                    .collect::<Vec<_>>()
                    .windows(trigger.len())
                    .any(|bytes| bytes == trigger.as_bytes()),
                "prefix {length}, token {token}"
            );
        }
    }
    for token in &ids[..ids.len() - 1] {
        original
            .prepared_forbidden_mutation()
            .unwrap()
            .commit(*token)
            .unwrap();
    }
    assert!(matches!(
        original
            .prepared_forbidden_mutation()
            .unwrap()
            .commit(*ids.last().unwrap()),
        Err(eredu_core::speculative::ForbiddenControllerError::Forbidden(_))
    ));
    for other in [&automatic, &required] {
        let error = ConstraintController::from_original_forbidden_generation_plan_with(
            other,
            SharedTokenFilter::new(TokenFilter::All),
            32,
            &funding,
            |_| panic!("grammar branch entered forbidden compiler"),
        )
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .contains("separately qualified grammar source")
        );
    }
    drop((
        original,
        prepared,
        automatic,
        required,
        tokenizer,
        original_tokenizer,
        funding,
    ));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(escaped.token_bytes(space as usize), Some(b" ".as_slice()));
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn paid_channel_destinations_share_split_transitions_copies_and_escaped_reasoning() {
    use crate::runtime::generation::streaming::prepared_channels::PreparedChannelParser;
    use eredu_core::generation::{FinishReason, SemanticEvent};
    use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
    use eredu_runtime::working_memory::WorkingMemoryPool;
    use std::sync::atomic::AtomicBool;
    static CHANNELS: DeclarativeDialectSpec = DeclarativeDialectSpec {
        output: ExactEnvelope {
            prefix: "<call>",
            suffix: "</call>",
        },
        reasoning_channel: Some(crate::runtime::chat::dialect::DelimitedChannel {
            prefix: "<think>",
            suffix: "</think>",
            required: false,
            prefix_in_prompt: false,
        }),
        auto_activation_trigger: Some("<call>"),
        ..SPEC
    };
    #[derive(Debug)]
    struct Account {
        bytes: Arc<AtomicUsize>,
        retired: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, n: usize) -> Result<(), HostMetadataFundingError> {
            self.bytes.fetch_add(n, Ordering::SeqCst);
            Ok(())
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let (tokenizer, eos) = tokenizer();
    let compiler = ConstraintCompiler::from_tokenizer_with_authority(
        &tokenizer,
        &eos,
        &HostPreparationAuthority::unmanaged(),
    )
    .unwrap();
    let prepared = compiler
        .compile_generation_plan(
            &DECLARATIVE_DIALECT,
            DialectParameters::Declarative(&CHANNELS),
            &ordinary_tools(),
            ToolChoice::None,
            ParallelToolCallPolicy::Disabled,
            vec!["<|end|>".to_owned()],
            vec![eos[1]],
            vec!["<|end|>".to_owned()],
            true,
        )
        .unwrap();
    let pool = WorkingMemoryPool::new(1 << 26, 0).unwrap();
    let original = pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(
                prepared
                    .generation_constraint()
                    .inner
                    .recipe
                    .tokenizer_object_json()
                    .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let source = pool
        .compile_forbidden_tokenizer_source(&original, b"<call>")
        .unwrap();
    drop(compiler);
    let bytes = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        bytes: bytes.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    let input = "hello<think>réason🦀</think>world";
    let mut escaped = None;
    for split in input
        .char_indices()
        .map(|(index, _)| index)
        .chain([input.len()])
    {
        let mut paid = PreparedChannelParser::prepare(
            prepared.semantic_plan(),
            &original,
            &source,
            input.len(),
            &funding,
        )
        .unwrap();
        let mut ordinary = prepared
            .semantic_plan()
            .create_parser_with_stops(std::iter::empty::<&str>())
            .unwrap();
        paid.push(&input[..split]).unwrap();
        ordinary.push(&input[..split]).unwrap();
        let mut copied = paid.copy(&funding).unwrap();
        paid.push(&input[split..]).unwrap();
        copied.push(&input[split..]).unwrap();
        ordinary.push(&input[split..]).unwrap();
        paid.finish().unwrap();
        copied.finish().unwrap();
        ordinary.finish(FinishReason::Eos).unwrap();
        let events = paid.take_events().unwrap();
        let copied_events = copied.take_events().unwrap();
        let expected: Vec<_> = ordinary
            .events()
            .iter()
            .filter(|event| !matches!(event, SemanticEvent::Finished { .. }))
            .cloned()
            .collect();
        assert_eq!(&*events, expected.as_slice(), "split {split}");
        assert_eq!(&*copied_events, &*events, "copied split {split}");
        if escaped.is_none() {
            escaped = events
                .into_iter()
                .find(|event| matches!(event, SemanticEvent::ReasoningDelta(_)));
        }
    }
    let mut failed =
        PreparedChannelParser::prepare(prepared.semantic_plan(), &original, &source, 64, &funding)
            .unwrap();
    let failure = failed.push("visible<call>").unwrap_err();
    assert!(failure.to_string().contains("tool-payload state"));
    let prefix = failed.take_events().unwrap();
    assert_eq!(&*prefix, &[SemanticEvent::TextDelta("visible".into())]);
    assert!(failed.finish().is_err());
    drop((
        failed, failure, prefix, source, original, prepared, tokenizer, funding,
    ));
    assert!(bytes.load(Ordering::SeqCst) > 0);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(!retired.load(Ordering::SeqCst));
    let SemanticEvent::ReasoningDelta(reasoning) = escaped.as_ref().unwrap() else {
        unreachable!()
    };
    assert!(!reasoning.is_empty());
    assert_eq!(reasoning.snapshot_copy_bytes(), 0);
    drop(escaped);
    assert!(retired.load(Ordering::SeqCst));
}

#[test]
fn original_grammar_vocabulary_uses_normalized_historical_source_and_retires_all_owners() {
    use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
    use eredu_runtime::working_memory::WorkingMemoryPool;
    use std::sync::atomic::AtomicBool;
    #[derive(Debug)]
    struct Account {
        bytes: Arc<AtomicUsize>,
        retired: Arc<AtomicBool>,
        refused: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, n: usize) -> Result<(), HostMetadataFundingError> {
            if self.refused.load(Ordering::SeqCst) {
                return Err(HostMetadataFundingError::Capacity {
                    required: u64::try_from(n).unwrap(),
                    available: 0,
                });
            }
            self.bytes.fetch_add(n, Ordering::SeqCst);
            Ok(())
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let (mut tokenizer, eos) = tokenizer();
    tokenizer
        .with_normalizer(Some(tokenizers::normalizers::Prepend::new("xx".into())))
        .unwrap();
    tokenizer.set_encode_special_tokens(true);
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos).unwrap();
    let oracle = plan(&compiler, &ordinary_tools(), ToolChoice::Required);
    let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
    let (prepared, compilation) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &ordinary_tools(),
        ToolChoice::Required,
    );
    let (foreign_plan, foreign_compilation) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &ordinary_tools(),
        ToolChoice::Required,
    );
    let other = plan(&compiler, &ordinary_tools(), ToolChoice::Auto);
    let environment = oracle
        .generation_constraint()
        .grammar_matcher()
        .tok_env()
        .unwrap();
    let expected = environment.tokenize_bytes(b"ab<eos>");
    let info = *environment.tok_trie().info();
    tokenizer.set_encode_special_tokens(false);
    let bytes = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    let refused = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        bytes: bytes.clone(),
        retired: retired.clone(),
        refused: refused.clone(),
    })
    .unwrap();
    let source = prepared
        .generation_constraint()
        .inner
        .original_grammar_vocabulary(&compilation, &funding)
        .unwrap();
    assert!(bytes.load(Ordering::SeqCst) > 0);
    assert!(source.matches_plan(&prepared));
    let declared = source.compiled_declaration();
    let historical = prepared
        .generation_constraint()
        .inner
        .declaration
        .as_ref()
        .unwrap();
    let actual = historical
        .template(&prepared.generation_constraint().inner.recipe)
        .unwrap()
        .grammar();
    assert!(std::ptr::eq(declared, actual));
    assert_eq!(declared.start(), actual.start());
    assert_eq!(
        declared.rules_of(declared.start()),
        actual.rules_of(actual.start())
    );
    assert_eq!(declared.parametric(), actual.parametric());
    assert!(!source.matches_plan(&other));
    assert!(
        historical
            .template(&prepared.generation_constraint().inner.recipe)
            .unwrap()
            .matches_source(source.trie_source())
    );
    assert_eq!(source.trie_source().trie().info(), &info);
    assert_eq!(source.trie_source().trie().eos_tokens(), eos);
    for id in 0..info.vocab_size {
        assert_eq!(
            source.trie_source().trie().token(id),
            environment.tok_trie().token(id)
        );
    }
    use super::grammar_source::GrammarTokenizationMode as Mode;
    for input in [
        b"".as_slice(),
        b"ab<eos>".as_slice(),
        b"ab\xfe<eos>".as_slice(),
        b"a\xff[1]b\xff<eos>c\xff[9999]".as_slice(),
    ] {
        for mode in [Mode::Plain, Mode::Special, Mode::Marker] {
            let (expected, fixed) = match mode {
                Mode::Plain => (environment.tokenize_bytes(input), 0),
                Mode::Special => (environment.tokenize_bytes_special(input), 0),
                Mode::Marker => environment.tokenize_bytes_marker(input),
            };
            let result = source.tokenize_bytes(input, mode).unwrap();
            assert_eq!(result.ids(), expected);
            assert_eq!(result.fixed_tokens(), fixed);
            assert!(result.matches_plan(&prepared));
            assert!(!result.matches_plan(&other));
        }
    }
    let escaped = source.tokenize_bytes(b"ab<eos>", Mode::Plain).unwrap();
    let foreign = foreign_plan
        .generation_constraint()
        .inner
        .original_grammar_vocabulary(&foreign_compilation, &funding)
        .unwrap();
    let mut callbacks = 0;
    let failed = source
        .tokenize_with(b"a\xff[1]b", Mode::Marker, |actual, text| {
            callbacks += 1;
            if callbacks == 1 {
                actual.encode_tokenizer_ids(text)
            } else {
                foreign.trie_source().encode_tokenizer_ids(text)
            }
        })
        .unwrap_err();
    assert_eq!(callbacks, 2);
    assert_eq!(failed.retained_encoding_chunks(), 2);
    assert!(failed.retained_encoding_bytes() > 0);
    drop(foreign);
    let previous_charge = bytes.load(Ordering::SeqCst);
    let mut state = prepared
        .generation_constraint()
        .inner
        .original_grammar_state(&compilation, &funding)
        .unwrap();
    assert!(bytes.load(Ordering::SeqCst) > previous_charge);
    assert!(state.parser().vocabulary().matches_plan(&prepared));
    let output = br#"{"calls":[{"name":"check","arguments":{"value":17}}]}"#;
    let mut ordinary = oracle.generation_constraint().grammar_matcher();
    for (position, &byte) in output.iter().enumerate() {
        state = state.compute_mask().unwrap();
        assert_eq!(
            state.token_mask().unwrap(),
            &ordinary.compute_mask_or_eos().unwrap(),
            "canonical mask at {position}"
        );
        let token = state
            .parser()
            .vocabulary()
            .trie_source()
            .trie()
            .token_id_at_bytes(&[byte])
            .expect("fixture emits every byte");
        state = state.commit(token).unwrap();
        ordinary.consume_token(token).unwrap();
    }
    assert_eq!(state.parser().parser().final_bytes(), output);
    let (mut state, accepted) = state.is_complete().unwrap();
    assert!(accepted);
    let copy_retired = Arc::new(AtomicBool::new(false));
    let copy_funding = HostMetadataFunding::new(Account {
        bytes: Arc::new(AtomicUsize::new(0)),
        retired: copy_retired.clone(),
        refused: Arc::new(AtomicBool::new(false)),
    })
    .unwrap();
    let copied = state.try_copy(&copy_funding).unwrap();
    assert_eq!(copied.parser().parser().final_bytes(), output);
    assert!(
        copied
            .parser()
            .vocabulary()
            .trie_source()
            .same_source(state.parser().vocabulary().trie_source())
    );
    // Copied execution uses its destination account after the original payer refuses.
    refused.store(true, Ordering::SeqCst);
    let copied = copied.compute_mask().unwrap();
    assert_eq!(
        copied.token_mask().unwrap(),
        &ordinary.compute_mask_or_eos().unwrap()
    );
    let copied = copied.commit(eos[1]).unwrap();
    let (copied, terminal) = copied.is_terminal().unwrap();
    assert!(terminal);
    refused.store(false, Ordering::SeqCst);
    drop((copied, copy_funding));
    assert!(copy_retired.load(Ordering::SeqCst));

    let failed_copy_retired = Arc::new(AtomicBool::new(false));
    let failed_copy_refused = Arc::new(AtomicBool::new(false));
    let fail_funding = HostMetadataFunding::new(Account {
        bytes: Arc::new(AtomicUsize::new(0)),
        retired: failed_copy_retired.clone(),
        refused: failed_copy_refused.clone(),
    })
    .unwrap();
    failed_copy_refused.store(true, Ordering::SeqCst);
    let failed_copy = state.try_copy(&fail_funding).unwrap_err();
    assert_eq!(state.parser().parser().final_bytes(), output);
    drop(fail_funding);
    assert!(!failed_copy_retired.load(Ordering::SeqCst));
    state = state.compute_mask().unwrap();
    assert_eq!(
        state.token_mask().unwrap(),
        &ordinary.compute_mask_or_eos().unwrap()
    );
    drop((
        source,
        prepared,
        compilation,
        oracle,
        foreign_plan,
        foreign_compilation,
        other,
        compiler,
        tokenizer,
        environment,
        funding,
        ordinary,
    ));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(pool.used_bytes().unwrap() > 0);
    assert_eq!(
        state
            .parser()
            .vocabulary()
            .trie_source()
            .trie()
            .token(eos[0]),
        b"\xff<eos>"
    );
    refused.store(true, Ordering::SeqCst);
    let failed_operation = state.compute_mask().unwrap_err();
    assert_eq!(escaped.ids(), expected);
    drop(escaped);
    assert_eq!(failed.retained_encoding_chunks(), 2);
    drop(failed);
    assert!(!retired.load(Ordering::SeqCst));
    drop(failed_operation);
    assert!(!retired.load(Ordering::SeqCst));
    assert!(!failed_copy_retired.load(Ordering::SeqCst));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(failed_copy);
    assert!(failed_copy_retired.load(Ordering::SeqCst));
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_declaration_uses_compiled_exact_recipe_and_keeps_source_on_inspection_refusal() {
    use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
    use eredu_runtime::working_memory::WorkingMemoryPool;
    use std::sync::atomic::AtomicBool;
    #[derive(Debug)]
    struct Account {
        calls: Arc<AtomicUsize>,
        fail_at: Arc<AtomicUsize>,
        bytes: Arc<AtomicUsize>,
        retired: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if call == self.fail_at.load(Ordering::SeqCst) {
                return Err(HostMetadataFundingError::Capacity {
                    required: u64::try_from(bytes).unwrap(),
                    available: 0,
                });
            }
            self.bytes.fetch_add(bytes, Ordering::SeqCst);
            Ok(())
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let (tokenizer, eos) = tokenizer();
    let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
    let (prepared, compilation) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &ordinary_tools(),
        ToolChoice::Required,
    );
    let (equal, equal_compilation) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &ordinary_tools(),
        ToolChoice::Required,
    );
    let actual = prepared.generation_constraint().inner.clone();
    let equal_recipe = &equal.generation_constraint().inner.recipe;
    assert!(!actual.recipe.source().same_storage(equal_recipe.source()));
    assert_eq!(
        actual.recipe.source().as_ref(),
        equal_recipe.source().as_ref()
    );
    let declaration = actual.declaration.as_ref().unwrap().clone();
    assert!(declaration.template(equal_recipe).is_err());
    assert!(declaration.template(&actual.recipe).is_ok());
    let wrong = ConstraintBlueprint {
        recipe: equal_recipe.clone(),
        declaration: Some(declaration.clone()),
        fixture_matcher: None,
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let fail_at = Arc::new(AtomicUsize::new(usize::MAX));
    let bytes = Arc::new(AtomicUsize::new(0));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        calls: calls.clone(),
        fail_at: fail_at.clone(),
        bytes: bytes.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    let before = bytes.load(Ordering::SeqCst);
    let destination = actual.original_grammar_declaration(&funding).unwrap();
    assert!(bytes.load(Ordering::SeqCst) > before);
    assert!(std::ptr::eq(
        destination.grammar(),
        declaration.template(&actual.recipe).unwrap().grammar()
    ));
    assert_eq!(
        destination.grammar().start(),
        declaration
            .template(&actual.recipe)
            .unwrap()
            .grammar()
            .start()
    );
    assert!(wrong.original_grammar_declaration(&funding).is_err());
    // Refuse the adapter's fixed controls before cloning the immutable owner.
    let failed_call = calls.load(Ordering::SeqCst) + 1;
    fail_at.store(failed_call, Ordering::SeqCst);
    let failed = actual.original_grammar_declaration(&funding).unwrap_err();
    assert_eq!(calls.load(Ordering::SeqCst), failed_call);
    let mut cause: &(dyn std::error::Error + 'static) = &failed;
    let refusal = loop {
        if let Some(refusal) = cause.downcast_ref::<HostMetadataFundingError>() {
            break refusal;
        }
        cause = cause.source().expect("typed metadata refusal preserved");
    };
    assert!(matches!(
        refusal,
        HostMetadataFundingError::Capacity { available: 0, .. }
    ));
    drop((
        actual,
        wrong,
        prepared,
        equal,
        compilation,
        equal_compilation,
        tokenizer,
        declaration,
        funding,
    ));
    assert!(pool.used_bytes().unwrap() > 0);
    assert!(!retired.load(Ordering::SeqCst));
    drop(destination);
    assert!(pool.used_bytes().unwrap() > 0);
    assert!(!retired.load(Ordering::SeqCst));
    drop(failed);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_active_grammar_shares_commit_eos_completion_and_failed_source_custody() {
    use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
    use eredu_runtime::working_memory::{WorkingMemoryError, WorkingMemoryPool};
    use std::sync::atomic::AtomicBool;
    #[derive(Debug)]
    struct Account {
        refused: Arc<AtomicBool>,
        retired: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, n: usize) -> Result<(), HostMetadataFundingError> {
            if self.refused.load(Ordering::SeqCst) {
                Err(HostMetadataFundingError::Capacity {
                    required: u64::try_from(n).unwrap(),
                    available: 0,
                })
            } else {
                Ok(())
            }
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let (tokenizer, eos) = tokenizer();
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos).unwrap();
    let prepared = plan(&compiler, &ordinary_tools(), ToolChoice::Required);
    let pool = WorkingMemoryPool::new(1 << 26, 0).unwrap();
    let validity = pool
        .prepare_shared_token_filter(|| eredu_core::TokenFilter::All)
        .unwrap();
    let (source_plan, compilation) = original_plan(
        &pool,
        &tokenizer,
        &eos,
        &ordinary_tools(),
        ToolChoice::Required,
    );
    let original_blueprint = source_plan.generation_constraint().inner.clone();

    let refused = Arc::new(AtomicBool::new(false));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Account {
        refused: refused.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    let mut original = original_blueprint
        .original_grammar_state(&compilation, &funding)
        .unwrap();
    let mut ordinary = prepared.generation_constraint().grammar_matcher();
    let output = br#"{"calls":[{"name":"check","arguments":{"value":17}}]}"#;
    // A provisional grammar owns its independently copied parser and complete
    // canonical history. Terminal aliases need not enter the parser token row.
    #[inline(never)]
    fn check_sampler<C: eredu_core::SpeculativeTokenFilterController>(
        source: &C,
        terminal: &C,
        first: u32,
        disallowed: u32,
        pool: &WorkingMemoryPool,
    ) {
        use eredu_core::speculative::PreparedGrammarController;
        use eredu_runtime::generation::{
            ConstrainedSampler, DefaultSampler, MirostatV2Sampler, PreparedAdaptiveCommitError,
            PreparedGrammarSamplerCause, SpeculativeSampler,
        };
        type B = eredu_runtime::working_memory::WorkspaceSamplingBackend;
        type S<C> = ConstrainedSampler<DefaultSampler, C>;
        let retired = Arc::new(AtomicBool::new(false));
        let funding = HostMetadataFunding::new(Account {
            refused: Arc::new(AtomicBool::new(false)),
            retired: retired.clone(),
        })
        .unwrap();
        let sampler = S::new(DefaultSampler, source.clone());
        assert!(<S<C> as SpeculativeSampler<B>>::prepared_controller(&sampler).is_none());
        let plan = <S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&sampler).unwrap();
        assert!(plan.copy_metadata_bytes().is_some());
        assert!(plan.controller_source().unwrap().history().is_empty());
        let original_source = plan.controller_source().unwrap();
        let trie = original_source
            .tokenizer()
            .downcast_ref::<eredu_runtime::working_memory::OriginalTokenTrieSource>()
            .unwrap();
        trie.validate_grammar_source(original_source, pool).unwrap();
        let foreign = WorkingMemoryPool::new(1 << 30, 0).unwrap();
        assert!(matches!(
            trie.validate_grammar_source(original_source, &foreign),
            Err(WorkingMemoryError::IdentityMismatch)
        ));
        let choice = plan.choice(0.0, &funding).unwrap();
        assert!(plan.matches_choice(&choice));
        assert!(choice.greedy(0.0).is_ok());
        assert!(choice.categorical(0.75).is_err());

        assert!(plan.greedy(0.0).is_ok());
        assert!(plan.categorical(0.75).is_ok());
        let (decision, policy) = plan.logits(&[], &funding).unwrap().into_parts();
        assert_eq!(policy.bind(0.0, 0).unwrap().history_len(), 0);
        assert!(decision.before_forcing().unwrap().allows(first));
        let vocabulary = decision.before_forcing().unwrap().vocabulary();
        assert!(decision.capture_domain().unwrap().filter.allows(first));
        let forced = plan
            .force(
                first,
                eredu_runtime::TokenDomain::new(vocabulary),
                0,
                &funding,
            )
            .unwrap();
        let forced_plan =
            <S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&forced).unwrap();
        assert!(!forced_plan.matches_choice(&choice));
        let forced_choice = forced_plan.choice(0.75, &funding).unwrap();
        assert!(forced_choice.categorical(0.75).is_ok());

        let (forced_decision, _) = forced_plan.logits(&[], &funding).unwrap().into_parts();
        assert_eq!(forced_decision.forced_token(), Some(first));
        let shape = [1, i32::try_from(vocabulary).unwrap()];
        let mask = forced_decision.mask_plan(&shape).unwrap();
        let mut bits = Vec::with_capacity(mask.elements());
        mask.fill(&mut bits).unwrap();
        assert_eq!(bits.iter().filter(|&&bit| !bit).count(), 1);
        let repeated = forced_plan
            .force(
                first,
                eredu_runtime::TokenDomain::new(vocabulary),
                0,
                &funding,
            )
            .err()
            .unwrap();
        assert!(
            matches!(repeated.cause(), PreparedGrammarSamplerCause::Decision(cause)
            if matches!(cause.cause(), eredu_runtime::execution_control::PreparedGrammarChoiceCause::Choice(
                eredu_runtime::execution_control::TokenChoiceError::AlreadyPending)))
        );
        drop(repeated);
        let snapshot = <S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&forced)
            .unwrap()
            .copy(&funding)
            .unwrap();
        let cleared = <S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&forced)
            .unwrap()
            .clear(&funding)
            .unwrap();
        assert!(
            !<S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&snapshot)
                .unwrap()
                .matches_choice(&forced_choice)
        );
        assert!(
            !<S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&cleared)
                .unwrap()
                .matches_choice(&forced_choice)
        );

        assert_eq!(
            <S<C> as SpeculativeSampler<B>>::control_pending_forced(&cleared),
            None
        );
        assert_eq!(
            <S<C> as SpeculativeSampler<B>>::control_pending_forced(&snapshot),
            Some(first)
        );
        let committed = <S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&forced)
            .unwrap()
            .commit(first, None, &funding)
            .unwrap();
        assert_eq!(
            committed
                .controller()
                .prepared_grammar()
                .unwrap()
                .prepared_grammar_source()
                .history(),
            &[first]
        );
        assert!(
            snapshot
                .controller()
                .prepared_grammar()
                .unwrap()
                .prepared_grammar_source()
                .history()
                .is_empty()
        );
        assert_eq!(
            <S<C> as SpeculativeSampler<B>>::control_pending_forced(&committed),
            None
        );
        assert!(
            !<S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&committed)
                .unwrap()
                .prefix_is_complete(&[first], &funding)
                .unwrap()
        );
        let ended = S::new(DefaultSampler, terminal.clone());
        let ended_plan =
            <S<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&ended).unwrap();
        assert!(
            ended_plan
                .prefix_is_complete(ended_plan.controller_source().unwrap().history(), &funding)
                .unwrap()
        );
        type A<C> = ConstrainedSampler<MirostatV2Sampler, C>;
        let adaptive = A::new(MirostatV2Sampler::new(3.5, 0.2).unwrap(), source.clone());
        let invalid = <A<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&adaptive)
            .unwrap()
            .commit(disallowed, Some(0.0), &funding)
            .err()
            .unwrap();
        assert!(matches!(
            invalid.cause(),
            PreparedGrammarSamplerCause::Adaptive(PreparedAdaptiveCommitError::Probability)
        ));
        drop(invalid);
        let accepted = <A<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&adaptive)
            .unwrap()
            .commit(first, Some(0.25), &funding)
            .unwrap();
        let mut expected = MirostatV2Sampler::new(3.5, 0.2).unwrap();
        expected.accept_token(first, 0.25).unwrap();
        assert_eq!(accepted.policy().mu().to_bits(), expected.mu().to_bits());
        assert_eq!(accepted.policy().generated_tokens(), &[first]);
        assert!(adaptive.policy().generated_tokens().is_empty());
        let failure = <A<C> as SpeculativeSampler<B>>::prepared_grammar_controller(&adaptive)
            .unwrap()
            .commit(disallowed, Some(0.25), &funding)
            .err()
            .unwrap();
        assert!(matches!(
            failure.cause(),
            PreparedGrammarSamplerCause::Operation(_)
        ));
        drop((
            choice,
            forced_choice,
            forced_decision,
            forced,
            snapshot,
            cleared,
            committed,
            ended,
            accepted,
            adaptive,
            sampler,
            funding,
        ));
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(!retired.load(Ordering::SeqCst));
        drop(decision);
        assert!(retired.load(Ordering::SeqCst));
    }
    #[inline(never)]
    fn check_copy_bound<G: eredu_core::speculative::PreparedGrammarController>(
        grammar: &G,
        capacity: usize,
    ) {
        #[derive(Debug)]
        struct CopyAccount {
            spent: Arc<AtomicUsize>,
            limit: Arc<AtomicUsize>,
            retired: Arc<AtomicBool>,
        }
        impl HostMetadataAccount for CopyAccount {
            fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
                let current = self.spent.load(Ordering::SeqCst);
                let next = current
                    .checked_add(bytes)
                    .ok_or(HostMetadataFundingError::Overflow)?;
                let limit = self.limit.load(Ordering::SeqCst);
                if next > limit {
                    return Err(HostMetadataFundingError::Capacity {
                        required: u64::try_from(next).unwrap(),
                        available: u64::try_from(limit).unwrap(),
                    });
                }
                self.spent.store(next, Ordering::SeqCst);
                Ok(())
            }
        }
        impl Drop for CopyAccount {
            fn drop(&mut self) {
                self.retired.store(true, Ordering::SeqCst);
            }
        }
        let required = grammar.prepared_grammar_copy_bytes(capacity).unwrap();
        assert!(
            grammar
                .prepared_grammar_copy_bytes(grammar.prepared_grammar_source().history().len() - 1)
                .is_none()
        );
        for shortage in [0, 1] {
            let spent = Arc::new(AtomicUsize::new(0));
            let limit = Arc::new(AtomicUsize::new(usize::MAX));
            let retired = Arc::new(AtomicBool::new(false));
            let funding = HostMetadataFunding::new(CopyAccount {
                spent: spent.clone(),
                limit: limit.clone(),
                retired: retired.clone(),
            })
            .unwrap();
            spent.store(0, Ordering::SeqCst);
            limit.store(required - shortage, Ordering::SeqCst);
            let result = grammar.copy_prepared_grammar(capacity, &funding);
            if shortage == 0 {
                let copied = result.unwrap();
                assert_eq!(spent.load(Ordering::SeqCst), required);
                assert!(grammar.prepared_grammar_source().matches_copy(
                    copied.prepared_grammar_source(),
                    capacity,
                    &funding
                ));
                drop(funding);
                assert!(!retired.load(Ordering::SeqCst));
                drop(copied);
            } else {
                let failure = result.unwrap_err();
                assert!(spent.load(Ordering::SeqCst) < required);
                drop(funding);
                assert!(!retired.load(Ordering::SeqCst));
                drop(failure);
            }
            assert!(retired.load(Ordering::SeqCst));
        }
    }
    #[inline(never)]
    fn check_dynamic<C: eredu_core::SpeculativeTokenFilterController>(
        source: &C,
        tokens: &[u32],
        eos: u32,
        disallowed: u32,
        ordinary: &Matcher,
        branch_retired: Arc<AtomicBool>,
        branch_funding: HostMetadataFunding,
        pool: &WorkingMemoryPool,
    ) {
        use eredu_core::speculative::PreparedGrammarController;
        use eredu_runtime::execution_control::{PreparedGrammarBranch, PreparedGrammarBranchCause};
        let grammar = source.prepared_grammar().unwrap();
        let decision_retired = Arc::new(AtomicBool::new(false));
        let decision_funding = HostMetadataFunding::new(Account {
            refused: Arc::new(AtomicBool::new(false)),
            retired: decision_retired.clone(),
        })
        .unwrap();
        let branch =
            PreparedGrammarBranch::at(grammar, &tokens, tokens.len() + 2, &decision_funding)
                .unwrap();
        assert!(grammar.prepared_grammar_source().history().is_empty());
        assert_eq!(
            branch.controller().prepared_grammar_source().history(),
            tokens
        );
        assert!(
            branch
                .controller()
                .prepared_grammar_source()
                .funding()
                .same_account(&decision_funding)
        );
        assert!(
            grammar
                .prepared_grammar_source()
                .tokenizer()
                .same_borrowed_source(branch.controller().prepared_grammar_source().tokenizer())
        );
        let mut expected = ordinary.deep_clone();
        for &token in tokens {
            expected.consume_token(token).unwrap();
        }
        let expected_mask = expected.compute_mask_or_eos().unwrap();
        let packed = branch.mask().unwrap();
        for token in 0..expected_mask.len() {
            assert_eq!(
                packed.allows(token as u32),
                expected_mask.is_allowed(token as u32)
            );
        }
        let committed = branch
            .into_controller()
            .commit_prepared_grammar(eos)
            .unwrap();
        assert_eq!(
            committed.prepared_grammar_source().history().last(),
            Some(&eos)
        );
        assert_eq!(
            committed.prepared_grammar_source().history().len(),
            tokens.len() + 1
        );
        let (committed, terminal) = committed.prepared_grammar_terminal().unwrap();
        assert!(terminal);
        check_copy_bound(&committed, tokens.len() + 2);
        let copied = committed
            .copy_prepared_grammar(tokens.len() + 2, &branch_funding)
            .unwrap();
        assert!(committed.prepared_grammar_source().matches_copy(
            copied.prepared_grammar_source(),
            tokens.len() + 2,
            &branch_funding
        ));
        let (copied, terminal) = copied.prepared_grammar_terminal().unwrap();
        assert!(terminal);
        let installed = source
            .replace_prepared_grammar(copied, &branch_funding)
            .unwrap();
        assert_eq!(
            installed
                .prepared_grammar()
                .unwrap()
                .prepared_grammar_source()
                .history(),
            committed.prepared_grammar_source().history()
        );
        check_sampler(source, &installed, tokens[0], disallowed, pool);
        assert!(grammar.prepared_grammar_source().history().is_empty());
        let divergent =
            PreparedGrammarBranch::at(&committed, &tokens[..1], tokens.len() + 2, &branch_funding)
                .unwrap_err();
        assert!(matches!(
            divergent.cause(),
            PreparedGrammarBranchCause::History(_)
        ));
        drop(divergent);
        let failed_retired = Arc::new(AtomicBool::new(false));
        let failed_funding = HostMetadataFunding::new(Account {
            refused: Arc::new(AtomicBool::new(false)),
            retired: failed_retired.clone(),
        })
        .unwrap();
        let failure = PreparedGrammarBranch::at(
            grammar,
            &[tokens[0], disallowed],
            tokens.len() + 2,
            &failed_funding,
        )
        .unwrap_err();
        assert!(matches!(
            failure.cause(),
            PreparedGrammarBranchCause::Operation(_)
        ));
        assert!(grammar.prepared_grammar_source().history().is_empty());
        drop((
            failed_funding,
            decision_funding,
            branch_funding,
            committed,
            installed,
        ));
        assert!(!failed_retired.load(Ordering::SeqCst));
        assert!(decision_retired.load(Ordering::SeqCst));
        assert!(!branch_retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(failed_retired.load(Ordering::SeqCst));
    }
    {
        let branch_retired = Arc::new(AtomicBool::new(false));
        let branch_funding = HostMetadataFunding::new(Account {
            refused: Arc::new(AtomicBool::new(false)),
            retired: branch_retired.clone(),
        })
        .unwrap();
        let source = original
            .try_copy(&branch_funding)
            .unwrap()
            .into_controller(output.len() + 2, validity.clone())
            .unwrap();
        let tokens = output
            .iter()
            .map(|&byte| {
                original
                    .parser()
                    .vocabulary()
                    .trie_source()
                    .trie()
                    .token_id_at_bytes(&[byte])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let disallowed = original
            .parser()
            .vocabulary()
            .trie_source()
            .trie()
            .token_id_at_bytes(b"!")
            .unwrap();
        let shared = ConstraintController::from_prepared_grammar(source, &branch_funding).unwrap();
        let alias = shared.clone();
        assert!(std::ptr::eq(
            shared.prepared_grammar().unwrap(),
            alias.prepared_grammar().unwrap()
        ));
        check_dynamic(
            &shared,
            &tokens,
            eos[1],
            disallowed,
            &ordinary,
            branch_retired.clone(),
            branch_funding,
            &pool,
        );
        drop(shared);
        assert!(!branch_retired.load(Ordering::SeqCst));
        drop(alias);
        assert!(branch_retired.load(Ordering::SeqCst));
    }
    for &byte in output {
        original = original.compute_mask().unwrap();
        let expected = ordinary.compute_mask_or_eos().unwrap();
        assert_eq!(original.token_mask().unwrap(), &expected);
        {
            let packed = original.packed_filter(&validity).unwrap();
            assert!(std::ptr::eq(
                packed.words().as_ptr(),
                original.token_mask().unwrap().as_slice().as_ptr()
            ));
            let width = packed.vocabulary() + 3;
            let shape = [2, i32::try_from(width).unwrap()];
            let plan =
                eredu_runtime::generation::TokenMaskPlan::packed(packed, &shape, None).unwrap();
            let mut actual = Vec::with_capacity(plan.elements());
            plan.fill(&mut actual).unwrap();
            let expected_row = (0..width)
                .map(|id| id >= expected.len() || !expected.is_allowed(id as u32))
                .collect::<Vec<_>>();
            assert_eq!(&actual[..width], expected_row.as_slice());
            assert_eq!(&actual[width..], expected_row.as_slice());
            let domain = eredu_core::capture::CaptureTokenDomain {
                filter: eredu_core::capture::CaptureTokenFilter::Packed(packed),
                tokenizer_validity: &validity,
            };
            assert_eq!(
                domain.summary(width as u32).allowed_tokens,
                (0..expected.len())
                    .filter(|&id| expected.is_allowed(id as u32))
                    .count() as u64
            );
            let forced = (0..expected.len())
                .find(|&id| expected.is_allowed(id as u32))
                .unwrap() as u32;
            let plan =
                eredu_runtime::generation::TokenMaskPlan::packed(packed, &shape, Some(forced))
                    .unwrap();
            let mut only_forced = Vec::with_capacity(plan.elements());
            plan.fill(&mut only_forced).unwrap();
            assert_eq!(only_forced.iter().filter(|&&invalid| !invalid).count(), 2);
            assert!(!only_forced[forced as usize]);
            assert!(
                eredu_runtime::generation::TokenMaskPlan::packed(
                    packed,
                    &shape,
                    Some(width as u32)
                )
                .is_err()
            );
        }
        let token = original
            .parser()
            .vocabulary()
            .trie_source()
            .trie()
            .token_id_at_bytes(&[byte])
            .unwrap();
        ordinary.consume_token(token).unwrap();
        original = original.commit(token).unwrap();
        let (next, complete) = original.is_complete().unwrap();
        original = next;
        assert_eq!(complete, ordinary.is_accepting().unwrap());
        let (next, terminal) = original.is_terminal().unwrap();
        original = next;
        assert_eq!(
            terminal,
            (ordinary.is_accepting().unwrap() && ordinary.is_stopped())
        );
    }
    original = original.compute_mask().unwrap();
    let expected_mask = ordinary.compute_mask_or_eos().unwrap();
    assert_eq!(original.token_mask().unwrap(), &expected_mask);
    assert!(expected_mask.is_allowed(eos[1]));
    assert!(ordinary.is_accepting().unwrap());
    original = original.commit(eos[1]).unwrap();
    let (next, terminal) = original.is_terminal().unwrap();
    original = next;
    assert!(terminal);
    let copied_retired = Arc::new(AtomicBool::new(false));
    let copy_funding = HostMetadataFunding::new(Account {
        refused: Arc::new(AtomicBool::new(false)),
        retired: copied_retired.clone(),
    })
    .unwrap();
    let copied = original.try_copy(&copy_funding).unwrap();
    let (copied, terminal) = copied.is_terminal().unwrap();
    assert!(terminal);
    assert_eq!(
        copied.parser().parser().final_bytes(),
        original.parser().parser().final_bytes()
    );
    drop(copy_funding);
    assert!(!copied_retired.load(Ordering::SeqCst));
    drop(copied);
    assert!(copied_retired.load(Ordering::SeqCst));

    // Refuse the first constructor frame after the original tokenizer/trie
    // source exists. The failed startup must retain that actual source prefix.
    let startup_refused = Arc::new(AtomicBool::new(false));
    let startup_retired = Arc::new(AtomicBool::new(false));
    let startup_funding = HostMetadataFunding::new(Account {
        refused: startup_refused.clone(),
        retired: startup_retired.clone(),
    })
    .unwrap();
    startup_refused.store(true, Ordering::SeqCst);
    let startup_failure = original_blueprint
        .original_grammar_state(&compilation, &startup_funding)
        .unwrap_err();
    refused.store(true, Ordering::SeqCst);
    let failure = original.compute_mask().unwrap_err();
    drop((
        funding,
        startup_funding,
        original_blueprint,
        source_plan,
        compilation,
        validity,
        prepared,
        compiler,
        tokenizer,
        ordinary,
    ));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(!startup_retired.load(Ordering::SeqCst));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(failure);
    assert!(retired.load(Ordering::SeqCst));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(startup_failure);
    assert!(startup_retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn original_auto_grammar_matches_split_and_atomic_activation_masks_and_paid_copy() {
    use eredu_core::speculative::PreparedGrammarController;
    use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
    use eredu_runtime::working_memory::WorkingMemoryPool;
    use std::sync::atomic::AtomicBool;
    #[derive(Debug)]
    struct Account {
        spent: Arc<AtomicUsize>,
        limit: Arc<AtomicUsize>,
        retired: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            let current = self.spent.load(Ordering::SeqCst);
            let next = current
                .checked_add(bytes)
                .ok_or(HostMetadataFundingError::Overflow)?;
            let limit = self.limit.load(Ordering::SeqCst);
            if next > limit {
                return Err(HostMetadataFundingError::Capacity {
                    required: u64::try_from(next).unwrap(),
                    available: u64::try_from(limit).unwrap(),
                });
            }
            self.spent.store(next, Ordering::SeqCst);
            Ok(())
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    #[inline(never)]
    fn check_copy<G: PreparedGrammarController>(source: &G, capacity: usize) {
        let required = source.prepared_grammar_copy_bytes(capacity).unwrap();
        for shortage in [0, 1] {
            let spent = Arc::new(AtomicUsize::new(0));
            let limit = Arc::new(AtomicUsize::new(usize::MAX));
            let retired = Arc::new(AtomicBool::new(false));
            let funding = HostMetadataFunding::new(Account {
                spent: spent.clone(),
                limit: limit.clone(),
                retired: retired.clone(),
            })
            .unwrap();
            spent.store(0, Ordering::SeqCst);
            limit.store(required - shortage, Ordering::SeqCst);
            let result = source.copy_prepared_grammar(capacity, &funding);
            if shortage == 0 {
                let copy = result.unwrap();
                assert_eq!(spent.load(Ordering::SeqCst), required);
                assert!(source.prepared_grammar_source().matches_copy(
                    copy.prepared_grammar_source(),
                    capacity,
                    &funding
                ));
                let expected = source.prepared_grammar_mask().unwrap();
                let actual = copy.prepared_grammar_mask().unwrap();
                for id in 0..expected.vocabulary() {
                    assert_eq!(actual.allows(id as u32), expected.allows(id as u32));
                }
                drop(funding);
                assert!(!retired.load(Ordering::SeqCst));
                drop(copy);
            } else {
                let failure = result.unwrap_err();
                assert!(spent.load(Ordering::SeqCst) < required);
                drop(funding);
                assert!(!retired.load(Ordering::SeqCst));
                drop(failure);
            }
            assert!(retired.load(Ordering::SeqCst));
        }
    }
    for atomic in [false, true] {
        let (mut tokenizer, eos) = tokenizer();
        if atomic {
            tokenizer
                .add_tokens([AddedToken::from(r#"{"calls":"#, false).normalized(false)])
                .unwrap();
        }
        let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos).unwrap();
        let prepared = plan(&compiler, &ordinary_tools(), ToolChoice::Auto);
        let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
        let (source_plan, compilation) =
            original_plan(&pool, &tokenizer, &eos, &ordinary_tools(), ToolChoice::Auto);
        let blueprint = source_plan.generation_constraint().inner.clone();
        let spent = Arc::new(AtomicUsize::new(0));
        let limit = Arc::new(AtomicUsize::new(usize::MAX));
        let retired = Arc::new(AtomicBool::new(false));
        let funding = HostMetadataFunding::new(Account {
            spent: spent.clone(),
            limit: limit.clone(),
            retired: retired.clone(),
        })
        .unwrap();
        let state = blueprint
            .original_grammar_state(&compilation, &funding)
            .unwrap();
        let text = r#"free{"calls":[{"name":"check","arguments":{"value":17}}]}"#;
        let tokens = tokenizer.encode(text, false).unwrap().get_ids().to_vec();
        let mut original = state
            .into_auto_controller(tokens.len() + 2, SharedTokenFilter::new(TokenFilter::All))
            .unwrap();
        let mut ordinary = prepared.generation_constraint().grammar_matcher();
        let preamble = tokenizer.encode("free", false).unwrap().len();
        let activation = preamble + tokenizer.encode(r#"{"calls":"#, false).unwrap().len();
        let mut history = Vec::new();
        for (index, &token) in tokens.iter().enumerate() {
            original = original.compute_prepared_grammar_mask().unwrap();
            let expected = (index >= activation).then(|| ordinary.compute_mask_or_eos().unwrap());
            let actual = original.prepared_grammar_mask().unwrap();
            for id in 0..actual.vocabulary() {
                assert_eq!(
                    actual.allows(id as u32),
                    expected
                        .as_ref()
                        .is_none_or(|mask| mask.is_allowed(id as u32)),
                    "atomic={atomic} index={index} token={id}"
                );
            }
            assert!(actual.allows(token));
            if index == 3 {
                check_copy(&original, tokens.len() + 2);
            }
            original = original.commit_prepared_grammar(token).unwrap();
            if index >= preamble {
                ordinary.consume_token(token).unwrap();
            }
            history.push(token);
            assert_eq!(original.prepared_grammar_source().history(), history);
        }
        assert!(ordinary.is_accepting().unwrap());
        assert!(ordinary.compute_mask_or_eos().unwrap().is_allowed(eos[1]));
        original = original.commit_prepared_grammar(eos[1]).unwrap();
        let (original, terminal) = original.prepared_grammar_terminal().unwrap();
        assert!(terminal);
        limit.store(spent.load(Ordering::SeqCst), Ordering::SeqCst);
        let failure = original.compute_prepared_grammar_mask().unwrap_err();
        drop((
            funding,
            blueprint,
            source_plan,
            compilation,
            compiler,
            prepared,
            tokenizer,
            ordinary,
            pool,
        ));
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }
}

#[test]
fn original_tool_schema_callback_binds_compilation_receipt_and_keeps_argument_failure_custody() {
    use eredu_core::speculative::PreparedGrammarController;
    use eredu_nn::workspace::{HostMetadataAccount, HostMetadataFunding, HostMetadataFundingError};
    use eredu_runtime::working_memory::{
        OriginalSemanticControllerSource, WorkingMemoryError, WorkingMemoryPool,
    };
    use std::sync::atomic::AtomicBool;
    #[derive(Debug)]
    struct Payer {
        refuse: Arc<AtomicBool>,
        retired: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Payer {
        fn reserve_metadata(&self, n: usize) -> Result<(), HostMetadataFundingError> {
            if self.refuse.load(Ordering::SeqCst) {
                Err(HostMetadataFundingError::Capacity {
                    required: n as u64,
                    available: 0,
                })
            } else {
                Ok(())
            }
        }
    }
    impl Drop for Payer {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }
    let (tokenizer, eos) = tokenizer();
    let compiler = ConstraintCompiler::from_tokenizer(&tokenizer, &eos).unwrap();
    let tools = vec![tool(
        serde_json::json!({"type":"object", "properties":{"value":{"type":"integer"}}, "required":["value"], "additionalProperties":false}),
    )];
    let prepared = plan(&compiler, &tools, ToolChoice::Required);
    let pool = WorkingMemoryPool::new(1 << 27, 0).unwrap();
    let foreign = WorkingMemoryPool::new(1 << 27, 0).unwrap();
    let validity = pool
        .prepare_shared_token_filter(|| TokenFilter::All)
        .unwrap();
    let (source_plan, compilation) =
        original_plan(&pool, &tokenizer, &eos, &tools, ToolChoice::Required);
    let blueprint = source_plan.generation_constraint().inner.clone();
    let recipe = blueprint.recipe.clone();
    let schemas = source_plan
        .semantic_plan()
        .tool_schemas
        .as_ref()
        .unwrap()
        .clone();
    let refuse = Arc::new(AtomicBool::new(false));
    let retired = Arc::new(AtomicBool::new(false));
    let funding = HostMetadataFunding::new(Payer {
        refuse: refuse.clone(),
        retired: retired.clone(),
    })
    .unwrap();
    let callback = schemas.prepare(&recipe, &funding).unwrap();
    let grammar = blueprint
        .original_grammar_state(&compilation, &funding)
        .unwrap()
        .into_controller(128, validity)
        .unwrap();
    let source = OriginalSemanticControllerSource::Grammar(grammar.prepared_grammar_source());
    callback.validate_source(source, &pool).unwrap();
    assert!(matches!(
        callback.validate_source(source, &foreign),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    let (equal_plan, equal_compilation) =
        original_plan(&pool, &tokenizer, &eos, &tools, ToolChoice::Required);
    let equal_recipe = equal_plan.generation_constraint().inner.recipe.clone();
    assert!(schemas.prepare(&equal_recipe, &funding).is_err());
    funding
        .reserve_metadata(callback.failure_control_bytes().unwrap())
        .unwrap();
    callback
        .validate("check", r#"{"value":18446744073709551615}"#, &funding)
        .unwrap();
    funding
        .reserve_metadata(callback.failure_control_bytes().unwrap())
        .unwrap();
    let mismatch = callback
        .validate("check", r#"{"value":1,"\u0076alue":3.5}"#, &funding)
        .unwrap_err();
    assert!(mismatch.to_string().contains("do not match"));
    funding
        .reserve_metadata(callback.failure_control_bytes().unwrap())
        .unwrap();
    refuse.store(true, Ordering::SeqCst);
    let failure = callback
        .validate("check", r#"{"value":1}"#, &funding)
        .unwrap_err();
    let mut cause: &(dyn std::error::Error + 'static) = &failure;
    loop {
        if matches!(
            cause.downcast_ref::<HostMetadataFundingError>(),
            Some(HostMetadataFundingError::Capacity { available: 0, .. })
        ) {
            break;
        }
        cause = cause
            .source()
            .expect("actual schema funding cause is retained");
    }
    drop((
        grammar,
        blueprint,
        source_plan,
        compilation,
        equal_plan,
        equal_compilation,
        equal_recipe,
        callback,
        schemas,
        recipe,
        prepared,
        compiler,
        tokenizer,
        funding,
        mismatch,
    ));
    assert!(!retired.load(Ordering::SeqCst));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(failure);
    assert!(retired.load(Ordering::SeqCst));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[path = "scoped_frame_tests.rs"]
mod scoped_frame_tests;
