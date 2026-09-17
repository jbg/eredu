use super::*;
use eredu_core::{
    GenerationCancellationToken, SpeculativeCallbackPublisher, SpeculativeConstraint,
    SpeculativePublisher, SpeculativeSemanticConstraint,
};
use eredu_text::{stop_storage::StopCompilePlan, tokenizer_storage::TokenizerPlan};

#[test]
fn plain_semantic_forks_preserve_stop_state_and_events_keep_the_actual_account() {
    let input = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,
        "pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel",
        "add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],
        "model":{"type":"BPE","vocab":{"h":0,"i":1},"merges":[]}}"#;
    let capacity = 64 * 1024 * 1024;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let source = pool
        .compile_tokenizer(TokenizerPlan::prepare_json(input.as_bytes()).unwrap())
        .unwrap();
    let strings = ["ih".to_owned()];
    let stops = pool
        .compile_stop_source(StopCompilePlan::prepare(&strings).unwrap())
        .unwrap();
    let source_bytes = pool.used_bytes().unwrap();
    let execution = InferenceExecutionIdentity::default();
    let prepared =
        OriginalSpeculativeSemanticPreparation::new(&source, &execution, capacity).unwrap();
    prepared.validate(&pool, &execution).unwrap();
    assert_eq!(prepared.capacity_bytes(), capacity);
    assert!(
        prepared
            .validate(&pool, &InferenceExecutionIdentity::default())
            .is_err()
    );
    let alias = prepared.clone();
    assert!(prepared.same_preparation(&alias));
    let other = OriginalSpeculativeSemanticPreparation::new(&source, &execution, capacity).unwrap();
    assert!(!prepared.same_preparation(&other));
    drop(alias);
    let semantic = prepared.prepare(&stops, 8, 2, true).unwrap();
    let concrete = semantic
        .prepared_source()
        .unwrap()
        .downcast_ref::<OriginalSpeculativePlainText>()
        .unwrap();
    assert!(concrete.tokenizer().same_source(&source));
    assert!(concrete.preparation().same_preparation(&prepared));
    assert!(
        concrete
            .validate_pool(&WorkingMemoryPool::new(capacity, 0).unwrap())
            .is_err()
    );
    let funding = concrete.metadata_funding().clone();
    let eos = [1, 7];
    let config = prepared.prepare_configuration(8, 2, 0.0, &eos).unwrap();
    prepared.validate_configuration(&config).unwrap();
    assert!(other.validate_configuration(&config).is_err());
    let ordinary =
        eredu_core::SpeculativeConfiguration::from(eredu_core::SpeculativeConfig::default());
    assert!(prepared.validate_configuration(&ordinary).is_err());
    drop(ordinary);
    assert_eq!(config.eos_token_ids, eos);
    let config = config.try_into_ordinary().unwrap_err();
    struct CallbackDrop<'a>(&'a WorkingMemoryPool, &'a std::cell::Cell<bool>);
    impl Drop for CallbackDrop<'_> {
        fn drop(&mut self) {
            assert!(self.0.used_bytes().unwrap() > 0);
            self.1.set(true);
        }
    }
    let callback_dropped = std::cell::Cell::new(false);
    let witness = CallbackDrop(&pool, &callback_dropped);
    let callback = prepared
        .prepare_callback(move |_| {
            let _ = &witness;
        })
        .unwrap();
    prepared.validate_callback(&callback).unwrap();
    assert!(other.validate_callback(&callback).is_err());
    drop((prepared, other));
    let mut canonical = SpeculativeSemanticConstraint::semantic(semantic);
    let cancel = GenerationCancellationToken::new();
    let mut published = Vec::new();
    assert!(!canonical.push_token(0).unwrap());
    {
        let mut publisher =
            SpeculativeCallbackPublisher::semantic_boxed(Box::new(|event| published.push(event)));
        publisher
            .publish_committed(&mut canonical, &[0], &cancel, false)
            .unwrap();
    }
    assert_eq!(published, [SemanticEvent::TextDelta("h".into())]);
    assert_eq!(
        serde_json::to_string(&published[0]).unwrap(),
        r#"{"TextDelta":"h"}"#
    );
    let mut tentative = canonical.fork().unwrap();
    assert!(!tentative.push_token(1).unwrap());
    assert!(tentative.push_token(0).unwrap());
    tentative.finish(FinishReason::StopSequence).unwrap();
    // The original source has neither consumed the tentative i nor matched ih.
    assert!(!canonical.push_token(0).unwrap());
    let bytes = canonical.control_snapshot_metadata_bytes().unwrap();
    funding
        .reserve_metadata(
            bytes
                + HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>().unwrap(),
        )
        .unwrap();
    let mut snapshot = canonical
        .fork_control_snapshot(HostPreparationAuthority::retain(funding.clone()))
        .unwrap();
    let retained_metadata = pool.used_bytes().unwrap() - source_bytes;
    drop((canonical, tentative, source, stops, funding));
    {
        let mut publisher =
            SpeculativeCallbackPublisher::semantic_boxed(Box::new(|event| published.push(event)));
        publisher
            .publish_committed(&mut snapshot, &[0, 0], &cancel, false)
            .unwrap();
    }
    drop(snapshot);
    assert_eq!(
        published,
        [
            SemanticEvent::TextDelta("h".into()),
            SemanticEvent::TextDelta("h".into())
        ]
    );
    assert_eq!(pool.used_bytes().unwrap(), retained_metadata);
    let alias = published[0].clone();
    drop(published);
    assert!(pool.used_bytes().unwrap() > 0);
    drop(alias);
    assert_eq!(pool.used_bytes().unwrap(), retained_metadata);
    drop(config);
    assert!(pool.used_bytes().unwrap() > 0);
    assert!(!callback_dropped.get());
    drop(callback);
    assert!(callback_dropped.get());
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn channel_semantic_owner_preserves_structural_stop_snapshot_and_escaped_custody() {
    use eredu_text::semantic_channels::{ChannelProgram, DelimitedChannel};
    fn drain(state: &mut SpeculativeSemanticConstraint, ids: &[u32]) -> Vec<SemanticEvent> {
        let mut rows = Vec::new();
        {
            let mut publisher = SpeculativeCallbackPublisher::semantic(|event| rows.push(event));
            publisher
                .publish_committed(state, ids, &GenerationCancellationToken::new(), false)
                .unwrap();
        }
        rows
    }
    let input = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
        "normalizer":null,"pre_tokenizer":null,"post_processor":null,
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "added_tokens":[
            {"id":2,"content":"<think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":3,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":4,"content":"<stop>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "model":{"type":"BPE","vocab":{"h":0,"i":1},"merges":[]}}).to_string();
    let capacity = 64 * 1024 * 1024;
    let pool = WorkingMemoryPool::new(capacity, 0).unwrap();
    let tokenizer = pool
        .compile_tokenizer(TokenizerPlan::prepare_json(input.as_bytes()).unwrap())
        .unwrap();
    let forbidden = pool
        .compile_forbidden_tokenizer_source(&tokenizer, b"<call>")
        .unwrap();
    let program = ChannelProgram {
        reasoning_channel: Some(DelimitedChannel {
            prefix: "<think>",
            suffix: "</think>",
            prefix_in_prompt: false,
        }),
        text_channel: None,
        tool_delimiter: "<call>",
        tool_is_json: false,
    };
    let baseline = pool.used_bytes().unwrap();
    assert!(
        pool.compile_semantic_channel_source(
            &tokenizer,
            &forbidden,
            program,
            &[(0, "<think>", false)]
        )
        .is_err()
    );
    assert_eq!(pool.used_bytes().unwrap(), baseline);
    use eredu_text::semantic_channels::{JsonEnvelope, JsonToolLayout, JsonToolProgram, JsonToolShape};
    use eredu_text::json_fragments::{JsonCallId, JsonFieldNames, JsonFieldRole, JsonFieldError, JsonValueKind};
    let field_name = String::from("function");
    let tools = JsonToolProgram {
        output: JsonEnvelope { prefix: "<call>", suffix: "</call>" },
        call: JsonEnvelope { prefix: "", suffix: "" },
        function: JsonEnvelope { prefix: "", suffix: "" },
        fields: JsonFieldNames { name: &field_name, arguments: "input", call_id: Some(JsonCallId { field: "callid", length: Some(2) }) },
        shape: JsonToolShape::Object, separator: "\n", layout: JsonToolLayout::RepeatedEnvelopes,
    };
    let channels = pool
        .compile_semantic_channel_source_with_tools(
            &tokenizer,
            super::super::OriginalSemanticControllerSource::Forbidden(&forbidden),
            program,
            &[
                (2, "<think>", false),
                (3, "</think>", false),
                (4, "<stop>", true),
            ],
            Some(tools),
        )
        .unwrap();
    assert_eq!(channels.json_tools(), Some(tools));
    assert_ne!(channels.json_tools().unwrap().fields.name.as_ptr(), field_name.as_ptr());
    drop(field_name);
    assert_eq!(channels.json_tools().unwrap().fields.name, "function");
    assert!(channels.same_source(&channels.clone()));
    let foreign = pool
        .compile_tokenizer(TokenizerPlan::prepare_json(input.as_bytes()).unwrap())
        .unwrap();
    assert!(channels.validate(&pool, &foreign).is_err());
    let stops = pool
        .compile_stop_source(StopCompilePlan::prepare_refs(&["ih"]).unwrap())
        .unwrap();
    let execution = InferenceExecutionIdentity::default();
    let prepared =
        OriginalSpeculativeSemanticPreparation::new(&tokenizer, &execution, capacity).unwrap();
    let other =
        OriginalSpeculativeSemanticPreparation::new(&foreign, &execution, capacity).unwrap();
    assert!(
        other
            .prepare_channels(&stops, &channels, 16, 4, true)
            .is_err()
    );
    drop((foreign, other));
    let owner = prepared
        .prepare_channels(&stops, &channels, 16, 4, true)
        .unwrap();
    let concrete = owner
        .prepared_source()
        .unwrap()
        .downcast_ref::<OriginalSpeculativePlainText>()
        .unwrap();
    concrete.validate_pool(&pool).unwrap();
    assert!(concrete.preparation().same_preparation(&prepared));
    let funding = concrete.metadata_funding().clone();
    let tool_input = r#"{"function":"check","input":{"value":1},"callid":"é🙂"}"#;
    let (object, _, done) = super::super::OriginalJsonObject::prepare(tool_input.len(), &funding).unwrap().push(tool_input).unwrap();
    assert!(done);
    let fields = channels.json_tools().unwrap().fields;
    let roles = object.fields().iter().map(|field| fields.inspect(field.key().as_str(), field.kind().unwrap(), field.string().map(|text| text.as_str())).unwrap()).collect::<Vec<_>>();
    assert_eq!(roles, [JsonFieldRole::Name, JsonFieldRole::Arguments, JsonFieldRole::CallId]);
    assert_eq!(fields.inspect("callid", JsonValueKind::String, Some("🙂")), Err(JsonFieldError::CallIdLength));
    assert_eq!(fields.complete(true, false, true), Err(JsonFieldError::MissingArguments));
    drop(object);
    {
        use super::super::semantic_channel_parser::tool_call::Call;
        fn events(funding: &WorkspaceMetadataFunding) -> SpeculativeBuffer<SemanticEvent> {
            let bytes = SpeculativeBuffer::<SemanticEvent>::retained_control_bytes(16).unwrap()
                + HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>().unwrap();
            funding.reserve_metadata(bytes).unwrap();
            SpeculativeBuffer::try_new_retained(16, HostPreparationAuthority::retain(funding.clone())).unwrap()
        }
        #[derive(Debug, thiserror::Error)]
        #[error("injected full-schema loan refusal")]
        struct RefusedSchema;
        let prefix = r#"{"function":"check","callid":"é🙂","input":{"value":"#;
        let suffix = "1}}";
        let mut original_events = events(&funding);
        let call = Call::prepare(&channels, prefix.len() + suffix.len(), 7, &funding).unwrap();
        let (call, used, complete) = call.push(prefix, &mut original_events).unwrap();
        assert_eq!(used, prefix.len()); assert!(!complete);
        assert!(matches!(&original_events[0], SemanticEvent::ToolCallStart { index: 7, id, name } if id == "é🙂" && name == "check"));
        assert!(call.copy_bytes().is_some());
        let fork = call.try_copy(&funding).unwrap();
        let mut fork_events = events(&funding);
        let (fork, _, done) = fork.push(suffix, &mut fork_events).unwrap(); assert!(done);
        let refusal = fork.complete_with(|name, arguments, _| {
            assert_eq!(name, "check"); assert_eq!(arguments, r#"{"value":1}"#);
            Err(RefusedSchema)
        }, &mut fork_events).unwrap_err();
        assert!(!fork_events.iter().any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
        let (call, _, done) = call.push(suffix, &mut original_events).unwrap(); assert!(done);
        // Neutral callback-order fixture only. Production supplies the original
        // full-schema producer; grammar acceptance never supplies this success.
        let call = call.complete_with(|_, _, _| Ok::<_, RefusedSchema>(()), &mut original_events).unwrap();
        assert!(matches!(original_events.last(), Some(SemanticEvent::ToolCallEnd)));
        let escaped = original_events[0].clone();
        drop((call, refusal, original_events, fork_events));
        assert!(matches!(escaped, SemanticEvent::ToolCallStart { .. }));
    }


    {
        use super::super::{OriginalToolValidation, OriginalSemanticControllerSource, OriginalSemanticChannelParser, OriginalForbiddenSource};
        use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
        #[derive(Debug, thiserror::Error)]
        #[error("injected full-schema callback failure")]
        struct SchemaRefusal;
        #[derive(Debug)]
        struct Validation {
            controller: OriginalForbiddenSource, refused: Arc<AtomicBool>,
            calls: Arc<AtomicUsize>, retired: Arc<AtomicBool>,
        }
        impl Drop for Validation { fn drop(&mut self) { self.retired.store(true, Ordering::SeqCst); } }
        impl OriginalToolValidation for Validation {
            fn failure_control_bytes(&self) -> Option<usize> {
                eredu_core::BackendFailure::source_retention_peak_bytes::<SchemaRefusal>()
            }
            fn validate_source(&self, source: OriginalSemanticControllerSource<'_>, pool: &WorkingMemoryPool) -> Result<(), super::super::WorkingMemoryError> {
                self.controller.validate_pool(pool)?;
                match source {
                    OriginalSemanticControllerSource::Forbidden(source) if self.controller.inputs().same_source(source.inputs()) => Ok(()),
                    _ => Err(super::super::WorkingMemoryError::IdentityMismatch),
                }
            }
            fn validate(&self, name: &str, arguments: &str, funding: &WorkspaceMetadataFunding) -> Result<(), eredu_core::BackendFailure> {
                // This fixture authenticates the consumer/ordering only; facade
                // coverage supplies the original full-schema implementation.
                let _ = funding; // The shared caller prepaid the actual error owner.
                assert_eq!(name, "check"); assert_eq!(arguments, r#"{"value":1}"#);
                self.calls.fetch_add(1, Ordering::SeqCst);
                if self.refused.load(Ordering::SeqCst) {
                    Err(eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::InvalidInput, SchemaRefusal))
                } else { Ok(()) }
            }
        }
        let refused = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(AtomicUsize::new(0));
        let retired = Arc::new(AtomicBool::new(false));
        let validation = Arc::new(Validation { controller: forbidden.clone(), refused: refused.clone(), calls: calls.clone(), retired: retired.clone() });
        let source = pool.compile_semantic_channel_source_with_validation(&tokenizer,
            OriginalSemanticControllerSource::Forbidden(&forbidden), program, &[], channels.json_tools(), Some(validation.clone())).unwrap();
        let prefix = r#"<call>{"function":"check","callid":"é🙂","input":{"value":"#;
        let suffix = "1}}</call>";
        let mut parser = OriginalSemanticChannelParser::prepare(&source, prefix.len()+suffix.len(), &funding).unwrap();
        parser.push(prefix).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let mut branch = parser.copy(&funding).unwrap();
        parser.push(suffix).unwrap();
        let completed = parser.take_events().unwrap();
        assert!(matches!(completed.last(), Some(SemanticEvent::ToolCallEnd)));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        refused.store(true, Ordering::SeqCst);
        let failure = branch.push(suffix).unwrap_err().into_output();
        let partial = branch.take_events().unwrap();
        assert!(!partial.iter().any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let alias = failure.clone(); assert_eq!(failure, alias);
        let mut cause: &(dyn std::error::Error + 'static) = &failure;
        loop {
            if cause.downcast_ref::<SchemaRefusal>().is_some() { break; }
            cause = cause.source().expect("actual schema refusal remains in output chain");
        }
        drop((parser, branch, validation, source, completed, partial, failure));
        assert!(!retired.load(Ordering::SeqCst));
        drop(alias);
        assert!(retired.load(Ordering::SeqCst));
    }


    let mut canonical = SpeculativeSemanticConstraint::semantic(owner);
    for id in [0, 2, 0, 1] {
        assert!(!canonical.push_token(id).unwrap());
    }
    let published = drain(&mut canonical, &[0, 2, 0, 1]);
    assert_eq!(
        published,
        [
            SemanticEvent::TextDelta("h".into()),
            SemanticEvent::ReasoningDelta("h".into())
        ]
    );
    let escaped = published[1].clone();
    drop(published);
    let mut tentative = canonical.fork().unwrap();
    assert!(tentative.push_token(0).unwrap());
    tentative.finish(FinishReason::StopSequence).unwrap();
    assert_eq!(
        drain(&mut tentative, &[0]),
        [SemanticEvent::Finished {
            reason: FinishReason::StopSequence
        }]
    );
    let bytes = canonical.control_snapshot_metadata_bytes().unwrap();
    funding
        .reserve_metadata(
            bytes
                + HostPreparationAuthority::retention_bytes::<WorkspaceMetadataFunding>().unwrap(),
        )
        .unwrap();
    let mut copied = canonical
        .fork_control_snapshot(HostPreparationAuthority::retain(funding.clone()))
        .unwrap();
    assert!(!copied.push_token(3).unwrap());
    assert!(!copied.push_token(0).unwrap());
    assert!(copied.push_token(4).unwrap());
    copied.finish(FinishReason::Eos).unwrap();
    assert_eq!(
        drain(&mut copied, &[3, 0, 4]),
        [
            SemanticEvent::ReasoningDelta("i".into()),
            SemanticEvent::TextDelta("h".into()),
            SemanticEvent::Finished {
                reason: FinishReason::StopSequence
            }
        ]
    );
    let mut cancelled = Vec::new();
    {
        let mut publisher = SpeculativeCallbackPublisher::semantic(|event| cancelled.push(event));
        publisher.publish_cancelled(&mut canonical).unwrap();
    }
    assert_eq!(
        cancelled,
        [SemanticEvent::Finished {
            reason: FinishReason::Cancelled
        }]
    );
    drop((
        cancelled, canonical, tentative, copied, funding, prepared, channels, forbidden, tokenizer,
        stops,
    ));
    assert!(pool.used_bytes().unwrap() > 0);
    drop(escaped);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
