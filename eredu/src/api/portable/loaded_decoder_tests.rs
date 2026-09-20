//! Actual private LoadedModel cold composition. Request/driver behavior is
//! independently exercised through genuine runtime and native request fixtures.
use super::*;
use eredu_core::{
    BackendDescriptor, BackendFailure, BackendProvider, BackendSession, Completion,
    DeviceCapabilities, DeviceDescriptor, ObservationSet, PendingTextInput, PreparedModel,
    SessionCapabilities, Submission, TextPreparationInput, TextStepContext, TokenFilter,
    TokenFilterController,
};
use eredu_runtime::working_memory::{
    LoadedDecodeSource, LoadedDecodeSourceBackend, WorkingMemoryError, WorkingMemoryPool,
};
use eredu_text::decoder_storage::DecodeCompilePlan;
use std::{cell::Cell, rc::Rc};
struct Backend {
    pool: WorkingMemoryPool,
    compiles: Rc<Cell<usize>>,
}
struct Session;
#[derive(Clone)]
struct Token;
struct Done;
impl TokenOutput for Token {
    type Error = WorkingMemoryError;
    fn token_id(&self) -> Result<u32, Self::Error> {
        unreachable!("cold fixture")
    }
}
impl Completion for Done {
    type Error = WorkingMemoryError;
    fn is_complete(&self) -> Result<bool, Self::Error> {
        unreachable!("cold fixture")
    }
    fn wait(&self) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
}
impl BackendProvider for Backend {
    type ModelConfig = ();
    type Model = ();
    type Session = Session;
    type Error = WorkingMemoryError;
    fn descriptor(&self) -> BackendDescriptor {
        BackendDescriptor::new("loaded-source-cold", "1")
    }
    fn devices(&self) -> Result<Vec<(DeviceDescriptor, DeviceCapabilities)>, Self::Error> {
        Ok(vec![])
    }
    fn prepare_model(&self, _: ()) -> Result<PreparedModel<()>, Self::Error> {
        Ok(PreparedModel::new((), SessionCapabilities::default()))
    }
    fn create_session(&self, _: PreparedModel<()>) -> Result<Session, Self::Error> {
        Ok(Session)
    }
    fn session_capability_mismatch(
        &self,
        _: SessionCapabilities,
        _: SessionCapabilities,
    ) -> Self::Error {
        WorkingMemoryError::IdentityMismatch
    }
}
impl BackendSession<Backend> for Session {
    type PrefillInput = Vec<u32>;
    type DecodeInput = Token;
    type Output = Token;
    type Completion = Done;
    fn capabilities(&self) -> SessionCapabilities {
        SessionCapabilities::default()
    }
    fn prefill(
        &mut self,
        _: &Backend,
        _: Vec<u32>,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("cold fixture")
    }
    fn decode(
        &mut self,
        _: &Backend,
        _: Token,
    ) -> Result<Submission<Token, Done>, WorkingMemoryError> {
        unreachable!("cold fixture")
    }
    fn observe_output(&self, _: &Backend, _: &Token) -> Result<ObservationSet, WorkingMemoryError> {
        unreachable!("cold fixture")
    }
}
impl TextGenerationBackend for Backend {
    fn reset_session(_: &Self, _: &mut Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize_session(_: &Self, _: &Session) -> Result<(), BackendFailure> {
        Ok(())
    }
    type TextPreparation = ();
    type TextPreparationControl = ();
    type TextStepPermit = ();
    type Prompt = Vec<u32>;
    type Token = Token;
    type TextGenerationState = ();
    type TextCompletion = Done;
    fn begin_text_step<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &(),
        _: &(),
        _: &C,
        _: PendingTextInput<&Vec<u32>, &Token>,
        _: &TextStepContext,
    ) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
    fn finish_text_step(_: ()) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
    fn admit_text_preparation<C: TokenFilterController>(
        _: &ModelRuntime<Self>,
        _: &TextPreparationInput<'_, Vec<u32>>,
        _: TextGenerationConfig,
        _: &C,
    ) -> Result<(), BackendFailure> {
        unreachable!("cold fixture")
    }
    fn start_text_generation(_: &Self, _: TextGenerationConfig) -> Result<(), Self::Error> {
        unreachable!("cold fixture")
    }
    fn prepare_text_prompt(_: &Self, _: Vec<u32>) -> Result<Vec<u32>, Self::Error> {
        unreachable!("cold fixture")
    }
    fn submit_text_prefill(
        _: &mut ModelRuntime<Self>,
        _: Vec<u32>,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, Self::Error> {
        unreachable!("cold fixture")
    }
    fn submit_text_decode(
        _: &mut ModelRuntime<Self>,
        _: Token,
        _: &TokenFilter,
        _: &mut (),
    ) -> Result<Submission<Token, Done>, Self::Error> {
        unreachable!("cold fixture")
    }
}
impl LoadedDecodeSourceBackend for Backend {
    fn compile_loaded_decode_source(
        runtime: &ModelRuntime<Self>,
        plan: DecodeCompilePlan<'_>,
    ) -> Result<LoadedDecodeSource, BackendFailure> {
        let backend = runtime.backend();
        backend.compiles.set(backend.compiles.get() + 1);
        backend.pool.compile_decode_source(plan).map_err(|error| {
            BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error)
        })
    }
}
fn tokenizer() -> ChatTokenizer {
    ChatTokenizer::from_bytes(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"hello":0,"é":8,"[UNK]":1},"unk_token":"[UNK]"}}"#.as_bytes()).unwrap()
}
fn model(pool: &WorkingMemoryPool) -> (LoadedModel<Backend>, Rc<Cell<usize>>) {
    model_with_tokenizer(pool, tokenizer())
}
fn model_with_tokenizer(
    pool: &WorkingMemoryPool,
    tokenizer: ChatTokenizer,
) -> (LoadedModel<Backend>, Rc<Cell<usize>>) {
    let compiles = Rc::new(Cell::new(0));
    let runtime = ModelRuntime::prepare(
        Backend {
            pool: pool.clone(),
            compiles: compiles.clone(),
        },
        (),
    )
    .unwrap();
    let model = LoadedModel::from_runtime(
        runtime,
        tokenizer,
        LoadedTextModelConfig {
            model_family: ModelKind::Llama,
            effective_model_type: "llama".into(),
            model_id: "cold-source-fixture".into(),
            chat_template: None,
            eos_token_ids: vec![],
            checkpoint_generation_config: None,
        },
    )
    .unwrap();
    (model, compiles)
}
#[test]
fn private_loaded_composition_compiles_once_and_request_headers_outlive_the_model() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let (mut model, compiles) = model(&pool);
    assert!(model.compiled_decoder_input(3, true).unwrap().is_none());
    let legacy = model.text_decoder(true);
    model.prepare_compiled_decoder().unwrap();
    let held = pool.used_bytes().unwrap();
    assert!(held > 0);
    model.prepare_compiled_decoder().unwrap();
    assert_eq!(compiles.get(), 1);
    let first = model.compiled_decoder_input(3, true).unwrap().unwrap();
    let second = model.compiled_decoder_input(5, false).unwrap().unwrap();
    assert!(first.source().same_source(second.source()));
    assert_eq!(held, first.source().original_bytes());
    drop(model);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(first);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(second);
    assert_eq!(pool.used_bytes().unwrap(), 0);
    // Existing raw HF aliases are an explicitly separate resident obligation.
    assert_eq!(legacy.tokenizer.decode(&[0, 8], true).unwrap(), "hello é");
}
#[test]
fn private_loaded_compilation_one_short_rejects_before_source_installation() {
    let snapshot = tokenizer().snapshot();
    let required = WorkingMemoryPool::decode_source_required_bytes(
        &DecodeCompilePlan::prepare(&snapshot).unwrap(),
    )
    .unwrap();
    for bytes in [required - 1, required] {
        let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
        let (mut model, compiles) = model(&pool);
        let result = model.prepare_compiled_decoder();
        assert_eq!(compiles.get(), 1);
        if bytes < required {
            let error = result.unwrap_err();
            let original = std::error::Error::source(&error)
                .unwrap()
                .downcast_ref::<eredu_runtime::working_memory::LoadedDecodeSourceError>()
                .unwrap();
            assert_eq!(original.retained_bytes(), 0);
            assert!(model.compiled_decoder_input(3, true).unwrap().is_none());
            assert_eq!(pool.used_bytes().unwrap(), 0);
        } else {
            result.unwrap();
            assert_eq!(pool.used_bytes().unwrap(), required);
        }
        drop(model);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

impl eredu_runtime::working_memory::OriginalStopSourceBackend for Backend {
    fn compile_original_stop_source(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::stop_storage::StopCompilePlan<'_>,
    ) -> Result<eredu_runtime::working_memory::OriginalStopSource, BackendFailure> {
        runtime
            .backend()
            .pool
            .compile_stop_source(plan)
            .map_err(|error| {
                BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error)
            })
    }
}
#[test]
fn private_plain_requests_use_actual_overrides_and_share_only_selected_original_sources() {
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let (mut model, compiles) = model(&pool);
    model.prepare_compiled_decoder().unwrap();
    let first_strings = vec![String::from("halt"), String::from(""), String::from("halt")];
    let first = model.compile_request_stops(&first_strings).unwrap();
    let cold_decoder = model.compiled_decoder.as_ref().unwrap().original_bytes();
    let cold_first = first.original_bytes();
    assert_eq!(first.source().stops().collect::<Vec<_>>(), ["halt"]);
    let a = model
        .compiled_plain_decoder_input(&first, 3, true)
        .unwrap()
        .unwrap();
    let b = model
        .compiled_plain_decoder_input(&first, 5, false)
        .unwrap()
        .unwrap();
    assert!(a.source().same_source(b.source()));
    assert!(
        a.stop_source()
            .unwrap()
            .same_source(b.stop_source().unwrap())
    );
    let override_strings = vec![String::from("é!"), String::from("halt")];
    let changed = model.compile_request_stops(&override_strings).unwrap();
    let c = model
        .compiled_plain_decoder_input(&changed, 3, true)
        .unwrap()
        .unwrap();
    assert!(
        !a.stop_source()
            .unwrap()
            .same_source(c.stop_source().unwrap())
    );
    assert_eq!(
        c.stop_source()
            .unwrap()
            .source()
            .stops()
            .collect::<Vec<_>>(),
        ["é!", "halt"]
    );
    let empty = model.compile_request_stops(&[]).unwrap();
    let d = model
        .compiled_plain_decoder_input(&empty, 0, true)
        .unwrap()
        .unwrap();
    assert!(d.stop_source().unwrap().source().stops().next().is_none());
    assert_eq!(
        eredu_core::GenerationDecoderInput::output_kind(&d),
        eredu_core::GenerationDecoderOutput::PlainText
    );
    assert_eq!(compiles.get(), 1);
    let held = cold_decoder + cold_first + changed.original_bytes() + empty.original_bytes();
    drop((
        model,
        first,
        changed,
        empty,
        first_strings,
        override_strings,
    ));
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(a);
    assert_eq!(pool.used_bytes().unwrap(), held);
    drop(b);
    assert_eq!(pool.used_bytes().unwrap(), held - cold_first);
    drop(c);
    assert_eq!(
        pool.used_bytes().unwrap(),
        cold_decoder + d.stop_source().unwrap().original_bytes()
    );
    drop(d);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn private_stop_override_short_or_foreign_source_preserves_existing_compiled_decoder() {
    let snapshot = tokenizer().snapshot();
    let decoder_bytes = WorkingMemoryPool::decode_source_required_bytes(
        &DecodeCompilePlan::prepare(&snapshot).unwrap(),
    )
    .unwrap();
    let strings = vec![String::from("stop")];
    let stop_bytes = WorkingMemoryPool::stop_source_required_bytes(
        &eredu_text::stop_storage::StopCompilePlan::prepare(&strings).unwrap(),
    )
    .unwrap();
    let pool = WorkingMemoryPool::new(decoder_bytes + stop_bytes - 1, 0).unwrap();
    let (mut model, _) = model(&pool);
    model.prepare_compiled_decoder().unwrap();
    let error = model.compile_request_stops(&strings).unwrap_err();
    let source = std::error::Error::source(&error)
        .unwrap()
        .downcast_ref::<eredu_runtime::working_memory::OriginalStopSourceError>()
        .unwrap();
    assert_eq!(source.retained_bytes(), 0);
    assert_eq!(pool.used_bytes().unwrap(), decoder_bytes);
    let foreign = WorkingMemoryPool::new(stop_bytes, 0).unwrap();
    let stops = foreign
        .compile_stop_source(eredu_text::stop_storage::StopCompilePlan::prepare(&strings).unwrap())
        .unwrap();
    assert!(matches!(
        model.compiled_plain_decoder_input(&stops, 3, true),
        Err(WorkingMemoryError::IdentityMismatch)
    ));
    assert!(model.compiled_decoder_input(3, true).unwrap().is_some());
    drop((model, stops));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert_eq!(foreign.used_bytes().unwrap(), 0);
}

#[test]
fn actual_text_decoder_aliases_keep_cache_policy_after_loaded_model_drop() {
    use eredu_text::tokenizer::ModelCachePolicy;
    let json = br#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}}"#;
    let source =
        ChatTokenizer::from_bytes_with_cache_policy(json, ModelCachePolicy::disabled()).unwrap();
    let pool = WorkingMemoryPool::new(1_000_000, 0).unwrap();
    let (model, compiles) = model_with_tokenizer(&pool, source);
    let mut decoder = model.text_decoder(false);
    assert_eq!(decoder.step(0).unwrap().as_deref(), Some("a"));
    let mut clone = decoder.clone();
    assert_eq!(
        decoder.tokenizer.model_cache_policy(),
        Some(ModelCachePolicy::disabled())
    );
    drop(model);
    assert_eq!(decoder.step(1).unwrap().as_deref(), Some("b"));
    drop(decoder);
    assert_eq!(
        clone.tokenizer.model_cache_policy(),
        Some(ModelCachePolicy::disabled())
    );
    assert_eq!(clone.step(2).unwrap().as_deref(), Some("ab"));
    assert_eq!(compiles.get(), 0); // Decoder aliases do not recompile a source.
}

impl eredu_runtime::working_memory::OriginalTokenizerBackend for Backend {
    fn encode_original_tokenizer_ids(
        runtime: &ModelRuntime<Self>,
        source: &eredu_runtime::working_memory::OriginalTokenizer,
        input: &str,
        add_special_tokens: bool,
    ) -> Result<eredu_runtime::working_memory::OriginalEncodedTokenIds, BackendFailure> {
        runtime
            .backend()
            .pool
            .encode_tokenizer_ids(source, input, add_special_tokens)
            .map_err(
                eredu_runtime::working_memory::OriginalTokenizerEncodeError::into_backend_failure,
            )
    }
    fn compile_original_tokenizer_file(
        runtime: &ModelRuntime<Self>,
        read: eredu_checkpoint::artifact::PreparedArtifactFileRead,
    ) -> Result<eredu_runtime::working_memory::OriginalTokenizer, BackendFailure> {
        let backend = runtime.backend();
        backend.compiles.set(backend.compiles.get() + 1);
        backend.pool.compile_tokenizer_file(read).map_err(
            eredu_runtime::working_memory::OriginalTokenizerInputError::into_backend_failure,
        )
    }
    fn compile_original_tokenizer(
        runtime: &ModelRuntime<Self>,
        plan: eredu_text::tokenizer_storage::TokenizerPlan<'_>,
    ) -> Result<eredu_runtime::working_memory::OriginalTokenizer, BackendFailure> {
        let backend = runtime.backend();
        backend.compiles.set(backend.compiles.get() + 1);
        backend.pool.compile_tokenizer(plan).map_err(|error| {
            BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error)
        })
    }
}
#[test]
fn private_fresh_tokenizer_producer_uses_actual_pool_and_rejects_malformed_json_under_original_admission()
 {
    use eredu_text::tokenizer_storage::TokenizerPlan;
    let input=r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":[["h","i"]]}}"#.to_owned();
    let bytes = WorkingMemoryPool::tokenizer_required_bytes(
        &TokenizerPlan::prepare_json(input.as_bytes()).unwrap(),
    )
    .unwrap();
    for short in [true, false] {
        let pool = WorkingMemoryPool::new(bytes - u64::from(short), 0).unwrap();
        let compiles = Rc::new(Cell::new(0));
        // This producer needs the real runtime/pool, not a previously built ChatTokenizer.
        let runtime = ModelRuntime::prepare(
            Backend {
                pool: pool.clone(),
                compiles: compiles.clone(),
            },
            (),
        )
        .unwrap();
        let malformed = &input.as_bytes()[..input.len() - 1];
        assert!(crate::api::tokenizer::compile_original_tokenizer(&runtime, malformed).is_err());
        assert_eq!(compiles.get(), 1);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        let result = crate::api::tokenizer::compile_original_tokenizer(&runtime, input.as_bytes());
        assert_eq!(compiles.get(), 2);
        drop(runtime);
        if short {
            assert!(result.is_err());
            assert_eq!(pool.used_bytes().unwrap(), 0);
        } else {
            let source = result.unwrap();
            assert_eq!(source.token_id("hi"), Some(2));
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            let header = eredu_runtime::working_memory::AggregateGenerationDecoderInput::new(
                &source, 3, false,
            )
            .unwrap();
            assert!(header.source().same_source(&source));
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), bytes);
            drop(header);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        }
    }
}

#[cfg(unix)]
#[test]
fn private_consumed_file_producer_admits_i_then_c_and_preserves_borrowed_source_compatibility() {
    use eredu_checkpoint::artifact::PreparedArtifactFileRead;
    use eredu_text::tokenizer_storage::TokenizerPlan;
    use std::io::Write as _;
    let input = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":[["h","i"]]}}"#;
    let make_file = || {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(input.as_bytes()).unwrap();
        file
    };
    let i = WorkingMemoryPool::tokenizer_file_required_bytes(
        &PreparedArtifactFileRead::new(make_file()).unwrap(),
    )
    .unwrap();
    let c = WorkingMemoryPool::tokenizer_required_bytes(
        &TokenizerPlan::prepare_json(input.as_bytes()).unwrap(),
    )
    .unwrap();
    for capacity in [i - 1, i + c - 1, i + c] {
        let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
        let compiles = Rc::new(Cell::new(0));
        let runtime = ModelRuntime::prepare(
            Backend {
                pool: pool.clone(),
                compiles: compiles.clone(),
            },
            (),
        )
        .unwrap();
        let result = crate::api::tokenizer::compile_original_tokenizer_file(&runtime, make_file());
        assert_eq!(compiles.get(), 1);
        drop(runtime);
        if capacity < i + c {
            use std::error::Error as _;
            let error = result.unwrap_err();
            let source = error
                .source()
                .unwrap()
                .downcast_ref::<eredu_runtime::working_memory::OriginalTokenizerInputError>()
                .unwrap();
            assert_eq!(source.input_bytes(), if capacity < i { 0 } else { i });
            assert_eq!(pool.used_bytes().unwrap(), source.input_bytes());
            drop(error);
            assert_eq!(pool.used_bytes().unwrap(), 0);
        } else {
            let source = result.unwrap();
            assert_eq!(pool.used_bytes().unwrap(), c);
            let comparison = WorkingMemoryPool::new(c, 0).unwrap();
            let borrowed = comparison
                .compile_tokenizer(TokenizerPlan::prepare_json(input.as_bytes()).unwrap())
                .unwrap();
            assert_eq!(
                source.ids().collect::<Vec<_>>(),
                borrowed.ids().collect::<Vec<_>>()
            );
            for id in source.ids() {
                assert_eq!(source.spelling(id), borrowed.spelling(id));
            }
            let request = eredu_runtime::working_memory::AggregateGenerationDecoderInput::new(
                &source, 3, false,
            )
            .unwrap();
            drop(source);
            assert_eq!(pool.used_bytes().unwrap(), c);
            assert_eq!(request.source().token_id("hi"), Some(2));
            drop(request);
            assert_eq!(pool.used_bytes().unwrap(), 0);
            drop(borrowed);
            assert_eq!(comparison.used_bytes().unwrap(), 0);
        }
    }
}

#[test]
fn private_identity_encode_producer_retains_source_and_only_admits_actual_operation_destinations() {
    let input = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2},"merges":[["h","i"]]}}"#;
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let compiles = Rc::new(Cell::new(0));
    let runtime = ModelRuntime::prepare(
        Backend {
            pool: pool.clone(),
            compiles: compiles.clone(),
        },
        (),
    )
    .unwrap();
    let source =
        crate::api::tokenizer::compile_original_tokenizer(&runtime, input.as_bytes()).unwrap();
    let c = source.original_bytes();
    let expected =
        WorkingMemoryPool::tokenizer_encode_required_bytes(&source, "hihi", true).unwrap();
    let ids = crate::api::tokenizer::encode_original_tokenizer_ids(&runtime, &source, "hihi", true)
        .unwrap();
    assert_eq!(ids.ids(), [2, 2]);
    assert_eq!(ids.original_bytes(), expected);
    assert!(ids.matches_source(&source));
    assert_eq!(compiles.get(), 1, "encoding never reconstructs its source");
    drop((runtime, source));
    assert_eq!(pool.used_bytes().unwrap(), c + expected);
    let plan = eredu_core::TokenIdsInputPlan::new(ids.ids()).unwrap();
    assert_eq!(plan.tokens(), [2, 2]);
    drop(ids);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[cfg(unix)]
mod regex;

#[test]
fn original_text_default_adapters_reject_by_value_before_source_operations() {
    use eredu_runtime::working_memory::{
        OriginalTextSourceError, OriginalTokenizerBackend, OriginalTokenizerSourceError,
    };
    let pool = WorkingMemoryPool::new(10_000_000, 0).unwrap();
    let compiles = Rc::new(Cell::new(0));
    let runtime = ModelRuntime::prepare(
        Backend {
            pool: pool.clone(),
            compiles: compiles.clone(),
        },
        (),
    )
    .unwrap();
    let input=br#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#;
    let source = pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(input)
                .unwrap()
                .with_generation_domain()
                .unwrap(),
        )
        .unwrap();
    let c = source.original_bytes();
    assert!(matches!(
        Backend::compile_original_text_stop_source(
            &runtime,
            eredu_text::stop_storage::StopCompilePlan::prepare_refs(&["stop"]).unwrap()
        ),
        Err(OriginalTextSourceError::Domain(
            eredu_core::TokenInputRejection::Unsupported
        ))
    ));
    assert!(matches!(
        Backend::encode_original_text_ids(&runtime, &source, "a", true),
        Err(OriginalTextSourceError::Domain(
            eredu_core::TokenInputRejection::Unsupported
        ))
    ));
    let mut file = tempfile::tempfile().unwrap();
    std::io::Write::write_all(&mut file, input).unwrap();
    let read = eredu_checkpoint::artifact::PreparedArtifactFileRead::new(file).unwrap();
    assert!(matches!(
        Backend::compile_original_tokenizer_source_for_generation(
            &runtime,
            eredu_runtime::working_memory::OriginalTokenizerInput::File(read)
        ),
        Err(OriginalTokenizerSourceError::Domain(
            eredu_core::TokenInputRejection::Unsupported
        ))
    ));
    assert_eq!(compiles.get(), 0);
    assert_eq!(pool.used_bytes().unwrap(), c);
    drop((source, runtime));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
