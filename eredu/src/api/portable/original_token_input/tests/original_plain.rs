use super::super::plain::{
    start_original_plain_string_with_options_for, OriginalPlainSession, OriginalPlainStartError,
};
use super::*;
use std::io::Write as _;

/// Test shorthand preserving the fixture's concrete consumer error transport.
fn start_original_plain_string<'a, B: OriginalTokenizerBackend>(
    runtime: &'a mut ModelRuntime<B>,
    source: &OriginalTokenizer,
    input: &str,
    config: TextGenerationConfig,
    eos: &[u32],
    stops: &[&str],
    add_special_tokens: bool,
    skip_special_tokens: bool,
    cancellation: &GenerationCancellationToken,
) -> Result<Option<OriginalPlainSession<'a, B>>, OriginalPlainStartError<B::Error>> {
    start_original_plain_string_with_options_for::<B, OriginalPlainStartError<B::Error>>(
        runtime,
        source,
        input,
        config,
        eos,
        stops,
        add_special_tokens,
        skip_special_tokens,
        cancellation,
        None,
    )
}

// The supported regex profile needs about 56 MB of encoding workspace. Positive
// string fixtures reserve 64 MiB; exact short-admission cases below derive their
// limits from the actual C/S/E requirements and keep those limits unchanged.
const STRING_FIXTURE_CAPACITY: u64 = 64 * 1024 * 1024;
pub(super) const JSON: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true}]},"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[{"id":4,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2,"Ġ":3,"<S>":4,"Ġhi":8},"merges":[["h","i"],["Ġ","hi"]]}}"#;
fn source(runtime: &ModelRuntime<Backend>, from_file: bool) -> OriginalTokenizer {
    if from_file {
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(JSON.as_bytes()).unwrap();
        crate::api::tokenizer::compile_original_text_tokenizer_file(runtime, file).unwrap()
    } else {
        Backend::compile_original_tokenizer(
            runtime,
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(JSON.as_bytes())
                .unwrap()
                .with_generation_domain()
                .unwrap(),
        )
        .unwrap()
    }
}
#[test]
fn original_string_file_c_e_i_r_and_terminal_text_share_ordinary_and_manual_driver() {
    for manual in [false, true] {
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(STRING_FIXTURE_CAPACITY);
        // Both requests use the same application ceiling while the first R lives.
        facts.borrow_mut().shared_admission_capacity = Some(STRING_FIXTURE_CAPACITY);
        let source = source(&runtime, true);
        let cold = source.original_bytes();
        assert_eq!(
            pool.used_bytes().unwrap(),
            cold,
            "consumed tokenizer-file I has retired"
        );
        let ordinary = tokenizers::Tokenizer::from_bytes(JSON.as_bytes()).unwrap();
        let expected = ordinary.encode("hi hi<S>", true).unwrap();
        assert_eq!(expected.get_ids(), [2, 8, 4]);
        let mut previous = None;
        let mut previous_held = 0;
        for run in 0..2 {
            facts.borrow_mut().order.clear();
            facts.borrow_mut().ids.clear();
            let cancel = GenerationCancellationToken::new();
            let stops: &[&str] = if run == 0 { &["hi"] } else { &[] };
            let mut session = start_original_plain_string(
                &mut runtime,
                &source,
                "hi hi<S>",
                config(),
                &[],
                stops,
                true,
                true,
                &cancel,
            )
            .unwrap()
            .unwrap();
            assert_eq!(facts.borrow().ids, expected.get_ids());
            assert_eq!(facts.borrow().encodes, run + 1);
            assert_eq!(facts.borrow().stops, run + 1);
            assert_eq!(
                facts.borrow().order,
                ["admit", "bind", "input+R", "prompt", "sampling"]
            );
            assert!(
                facts.borrow().at_admission >= cold + previous_held + facts.borrow().encoded_bytes
            );
            let mut visible = String::new();
            let mut event = |event: GenerationPlainTextEvent<'_>| {
                if let GenerationPlainTextEvent::TextDelta(text) = event {
                    visible.push_str(text);
                }
            };
            let output = if manual {
                let mut address = None;
                while session.finish_reason().is_none() {
                    session = session.advance(&cancel, &mut event).unwrap();
                    let current = session.token_ids().as_ptr();
                    if let Some(previous) = address {
                        assert_eq!(current, previous);
                    } else {
                        address = Some(current);
                    }
                }
                session
                    .into_output()
                    .unwrap_or_else(|_| panic!("terminal session"))
            } else {
                session.run(&cancel, &mut event).unwrap()
            };
            assert_eq!(output.text.as_str(), visible);
            assert_eq!(output.text.as_str(), if run == 0 { "h " } else { "h hih" });
            assert_eq!(
                output.token_ids.as_ref(),
                if run == 0 {
                    &[0, 8][..]
                } else {
                    &[0, 8, 0][..]
                }
            );
            assert_eq!(facts.borrow().order[5], "delivery");
            let held = facts.borrow().held;
            assert_eq!(
                pool.used_bytes().unwrap(),
                cold + previous_held + held,
                "E and S have retired; results retain only original R"
            );
            if let Some(old) = previous.take() {
                drop(old);
            }
            previous_held = held;
            previous = Some(output);
        }
        let output = previous.unwrap();
        let text = output.text.clone();
        let address = text.as_str().as_ptr();
        let mut ids = output.token_ids.clone().into_iter();
        drop((output, source, runtime));
        assert_eq!(pool.used_bytes().unwrap(), previous_held);
        assert_eq!(text.as_str(), "h hih");
        assert_eq!(text.as_str().as_ptr(), address);
        assert_eq!(ids.next(), Some(0));
        drop(text);
        assert_eq!(pool.used_bytes().unwrap(), previous_held);
        drop(ids);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn original_string_preflight_cancellation_foreign_source_and_short_admission_preserve_owners() {
    let (mut runtime, facts, pool) = bare_runtime_with_capacity(STRING_FIXTURE_CAPACITY);
    let source = source(&runtime, false);
    let cold = source.original_bytes();
    let cancel = GenerationCancellationToken::new();
    cancel.cancel();
    assert!(start_original_plain_string(
        &mut runtime,
        &source,
        "hi",
        config(),
        &[],
        &[],
        true,
        true,
        &cancel
    )
    .unwrap()
    .is_none());
    assert_eq!((facts.borrow().encodes, facts.borrow().stops), (0, 0));
    let (mut foreign, foreign_facts, foreign_pool) = bare_runtime();
    let cancel = GenerationCancellationToken::new();
    assert!(start_original_plain_string(
        &mut foreign,
        &source,
        "hi",
        config(),
        &[],
        &[],
        true,
        true,
        &cancel
    )
    .is_err());
    assert_eq!(
        (foreign_facts.borrow().encodes, foreign_facts.borrow().stops),
        (0, 0)
    );
    assert_eq!(foreign_pool.used_bytes().unwrap(), 0);
    let decoder_only = pool
        .compile_tokenizer(
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(JSON.as_bytes()).unwrap(),
        )
        .unwrap();
    assert!(matches!(
        start_original_plain_string(
            &mut runtime,
            &decoder_only,
            "hi",
            config(),
            &[],
            &[],
            true,
            true,
            &cancel
        ),
        Err(OriginalPlainStartError::Input(
            TokenInputRejection::Unsupported
        ))
    ));
    drop(decoder_only);
    facts.borrow_mut().short = true;
    let failed = start_original_plain_string(
        &mut runtime,
        &source,
        "hi",
        config(),
        &[],
        &[],
        true,
        true,
        &cancel,
    );
    assert!(failed.is_err());
    drop(failed);
    assert_eq!((facts.borrow().encodes, facts.borrow().stops), (1, 1));
    assert_eq!(facts.borrow().order, ["admit"]);
    assert_eq!(pool.used_bytes().unwrap(), cold);
    facts.borrow_mut().short = false;
    let session = start_original_plain_string(
        &mut runtime,
        &source,
        "hi",
        config(),
        &[],
        &[],
        true,
        true,
        &cancel,
    )
    .unwrap()
    .unwrap();
    cancel.cancel();
    let output = session.run(&cancel, &mut |_| {}).unwrap();
    assert_eq!(output.finish_reason, FinishReason::Cancelled);
    assert!(output.text.as_str().is_empty());
    assert!(output.token_ids.is_empty());
    assert!(!facts.borrow().order.contains(&"submit"));
    drop((output, source, runtime, foreign));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
#[test]
fn generation_c_domain_matches_fresh_hf_for_declared_ids_duplicates_overlap_and_sparse_ids() {
    for declared in [0, 4, 1000, u32::MAX] {
        let mut value: serde_json::Value = serde_json::from_str(JSON).unwrap();
        value["added_tokens"] = serde_json::json!([
            {"id":declared,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
            {"id":declared,"content":"new","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
            {"id":declared,"content":"new","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
            {"id":declared,"content":"hi","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false}
        ]);
        let json = value.to_string();
        let (runtime, _, pool) = bare_runtime();
        let source = Backend::compile_original_tokenizer(
            &runtime,
            eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(json.as_bytes())
                .unwrap()
                .with_generation_domain()
                .unwrap(),
        )
        .unwrap();
        let ordinary = tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
        let domain = source.generation_domain().unwrap().allowed_mask().unwrap();
        for (id, &allowed) in domain.iter().enumerate() {
            let canonical = ordinary
                .id_to_token(id as u32)
                .and_then(|text| ordinary.token_to_id(&text))
                == Some(id as u32);
            assert_eq!(allowed, canonical, "declared={declared}, id={id}");
        }
        let vocab = ordinary.get_vocab(true); // oracle-owned ordinary compatibility allocation
        assert!(vocab.values().all(|id| (*id as usize) < domain.len()));
        assert_eq!(source.token_id("new"), ordinary.token_to_id("new"));
        drop((source, runtime));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[derive(Clone, Copy)]
enum Fault {
    None,
    MissingWitness,
    WrongTypeWitness,
    ForeignWitness,
    CopiedMask,
    MissingDeclaration,
    LargeFinal,
    PreOverride,
}
struct CheckedController {
    source: OriginalTokenizer,
    foreign: OriginalTokenizer,
    copied: TokenFilter,
    fault: Fault,
}
impl TokenFilterController for CheckedController {
    type Error = std::convert::Infallible;
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: self.source.generation_domain().unwrap().into(),
            additional_host_bytes: 0,
        })
    }
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithOriginalTokenDomain(OriginalSourceWitness::new(
            &self.source,
        ))
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(self.source.generation_domain().unwrap().clone())
    }
    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Self::Error> {
        let filter = if matches!(self.fault, Fault::LargeFinal) {
            TokenFilter::Allowed(vec![true; 32])
        } else {
            self.current_filter()?
        };
        let mut decision = TokenSamplingDecision::new(filter);
        if matches!(self.fault, Fault::PreOverride) {
            // Both final and preserved owned masks remain independently checked.
            decision.override_filter(self.source.generation_domain().unwrap().clone());
        }
        if !matches!(self.fault, Fault::MissingWitness) {
            let witness = if matches!(self.fault, Fault::ForeignWitness) {
                &self.foreign
            } else {
                &self.source
            };
            let mask = if matches!(self.fault, Fault::CopiedMask) {
                &self.copied
            } else {
                self.source.generation_domain().unwrap()
            };
            let witness = if matches!(self.fault, Fault::WrongTypeWitness) {
                OriginalSourceWitness::new(&self.fault)
            } else {
                OriginalSourceWitness::new(witness)
            };
            decision = decision.with_original_tokenizer_validity(mask, witness);
        }
        if !matches!(self.fault, Fault::MissingDeclaration) {
            decision = decision.with_controller_storage(self.inference_storage());
        }
        Ok(decision)
    }
    fn commit_token(&mut self, _: u32) -> Result<(), Self::Error> {
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
fn checked(
    source: &OriginalTokenizer,
    foreign: &OriginalTokenizer,
    fault: Fault,
) -> CheckedController {
    CheckedController {
        source: source.clone(),
        foreign: foreign.clone(),
        copied: source.generation_domain().unwrap().clone(),
        fault,
    }
}
#[test]
fn actual_core_iterator_authenticates_original_domain_after_callbacks_before_submission() {
    for fault in [
        Fault::MissingWitness,
        Fault::WrongTypeWitness,
        Fault::ForeignWitness,
        Fault::CopiedMask,
        Fault::MissingDeclaration,
        Fault::LargeFinal,
        Fault::PreOverride,
    ] {
        let (mut runtime, facts, pool) = bare_runtime();
        let source = self::source(&runtime, false);
        let foreign = self::source(&runtime, false);
        assert!(!source.same_source(&foreign));
        let controller = checked(&source, &foreign, fault);
        assert!(
            ControllerStorageContract::inspect(&controller).is_err(),
            "original mode cannot enter legacy registration/adoption"
        );
        let stops = pool
            .compile_stop_source(
                eredu_text::stop_storage::StopCompilePlan::prepare_refs(&[]).unwrap(),
            )
            .unwrap();
        let input =
            AggregateGenerationDecoderInput::new_plain_text(&source, &stops, 3, true).unwrap();
        let layout = GenerationSequenceConsumerLayout::for_driver_with_terminal_text::<
            RetainedGenerationSequence,
            WorkingMemoryError,
            WorkingMemoryError,
        >()
        .unwrap();
        let ids = [2];
        let mut generator = ControlledTextGeneration::from_token_ids_with_sequence(
            &mut runtime,
            TokenIdsInputPlan::new(&ids).unwrap(),
            config(),
            controller,
            None,
            GenerationSequenceRequest::new(3, &[])
                .with_decoder(&input)
                .with_consumer(&layout),
        )
        .unwrap();
        assert!(generator.next().unwrap().is_err());
        assert!(!facts.borrow().order.contains(&"submit"));
        drop(generator);
        drop((input, stops, source, foreign, runtime));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
#[test]
fn foreign_c_header_rejects_before_take_and_real_header_remains_usable() {
    let (mut runtime, facts, pool) = bare_runtime();
    let a = source(&runtime, false);
    let b = source(&runtime, false);
    let stops = pool
        .compile_stop_source(eredu_text::stop_storage::StopCompilePlan::prepare_refs(&[]).unwrap())
        .unwrap();
    let header = AggregateGenerationDecoderInput::new_plain_text(&b, &stops, 3, true).unwrap();
    let layout = GenerationSequenceConsumerLayout::for_driver_with_terminal_text::<
        RetainedGenerationSequence,
        WorkingMemoryError,
        WorkingMemoryError,
    >()
    .unwrap();
    let ids = [2];
    let error = ControlledTextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        TokenIdsInputPlan::new(&ids).unwrap(),
        config(),
        checked(&a, &b, Fault::None),
        None,
        GenerationSequenceRequest::new(3, &[])
            .with_decoder(&header)
            .with_consumer(&layout),
    );
    assert!(error.is_err());
    drop(error);
    assert_eq!(facts.borrow().order, ["admit"]);
    let generator = ControlledTextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        TokenIdsInputPlan::new(&ids).unwrap(),
        config(),
        checked(&b, &a, Fault::None),
        None,
        GenerationSequenceRequest::new(3, &[])
            .with_decoder(&header)
            .with_consumer(&layout),
    )
    .unwrap();
    assert_eq!(facts.borrow().ids, ids);
    assert!(!facts.borrow().order.contains(&"submit"));
    drop(generator);
    drop((header, stops, a, b, runtime));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_phase_cancellation_and_eos_mismatch_precede_original_input_and_submission() {
    for phase in 0..3 {
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(STRING_FIXTURE_CAPACITY);
        let source = source(&runtime, false);
        let cold = source.original_bytes();
        let cancellation = GenerationCancellationToken::new();
        match phase {
            0 => cancellation.cancel(),
            1 => facts.borrow_mut().cancel_after_stops = Some(cancellation.clone()),
            _ => facts.borrow_mut().cancel_after_encode = Some(cancellation.clone()),
        }
        assert!(start_original_plain_string(
            &mut runtime,
            &source,
            "hi",
            config(),
            &[],
            &["halt"],
            true,
            true,
            &cancellation
        )
        .unwrap()
        .is_none());
        assert_eq!(facts.borrow().validations, usize::from(phase != 0));
        assert_eq!(facts.borrow().stops, usize::from(phase != 0));
        assert_eq!(facts.borrow().encodes, usize::from(phase == 2));
        assert!(facts.borrow().order.is_empty());
        assert_eq!(
            pool.used_bytes().unwrap(),
            cold,
            "actual S/E retire on phase cancellation"
        );
        drop((source, runtime));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
    let (mut runtime, facts, pool) = bare_runtime();
    let source = source(&runtime, false);
    let cancellation = GenerationCancellationToken::new();
    assert!(matches!(
        start_original_plain_string(
            &mut runtime,
            &source,
            "hi",
            config(),
            &[7],
            &[],
            true,
            true,
            &cancellation
        ),
        Err(OriginalPlainStartError::Input(
            TokenInputRejection::InvalidToken
        ))
    ));
    assert_eq!(
        (
            facts.borrow().validations,
            facts.borrow().stops,
            facts.borrow().encodes
        ),
        (0, 0, 0)
    );
    let mut sampling = config().sampling();
    sampling.max_new_tokens = Some(0);
    assert!(matches!(
        start_original_plain_string(
            &mut runtime,
            &source,
            "hi",
            TextGenerationConfig::new(sampling),
            &[],
            &[],
            true,
            true,
            &cancellation
        ),
        Err(OriginalPlainStartError::Generation(
            eredu_core::GenerationError::ZeroTokenBudget
        ))
    ));
    assert_eq!(
        (
            facts.borrow().validations,
            facts.borrow().stops,
            facts.borrow().encodes
        ),
        (0, 0, 0)
    );
    drop((source, runtime));
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
#[ignore = "requires the ten pinned complete tokenizer artifact paths; required serial release oracle"]
fn ten_released_files_use_private_original_string_startup_and_retained_terminal_output() {
    use sha2::{Digest, Sha256};
    #[derive(serde::Deserialize)]
    struct Artifact {
        label: String,
        origin: String,
        sha256: String,
    }
    let manifest = std::env::var_os("EREDU_ORIGINAL_TEXT_ARTIFACTS")
        .expect("exact pinned artifact-inputs.json");
    let artifacts: Vec<Artifact> =
        serde_json::from_slice(&std::fs::read(manifest).unwrap()).unwrap();
    assert_eq!(artifacts.len(), 10);
    let mut rows = Vec::new();
    for artifact in artifacts {
        let bytes = std::fs::read(&artifact.origin).unwrap();
        let sha = Sha256::digest(&bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        assert_eq!(sha, artifact.sha256);
        let ordinary = tokenizers::Tokenizer::from_bytes(&bytes).unwrap();
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(u64::MAX);
        facts.borrow_mut().shared_admission_capacity = Some(u64::MAX);
        let source = crate::api::tokenizer::compile_original_text_tokenizer_file(
            &runtime,
            std::fs::File::open(&artifact.origin).unwrap(),
        )
        .unwrap();
        let c = source.original_bytes();
        assert_eq!(pool.used_bytes().unwrap(), c);
        let domain = source.generation_domain().unwrap().allowed_mask().unwrap();
        for (id, &actual) in domain.iter().enumerate() {
            let expected = ordinary
                .id_to_token(id as u32)
                .and_then(|spelling| ordinary.token_to_id(&spelling))
                == Some(id as u32);
            assert_eq!(
                actual, expected,
                "{} canonical domain id {id}",
                artifact.label
            );
        }
        assert!(ordinary
            .get_vocab(true)
            .values()
            .all(|id| (*id as usize) < domain.len()));
        let domain_positions = domain.len();
        let a = source.token_id("a").unwrap();
        let b = source.token_id("b").unwrap();
        assert!(!source.is_special("a") && !source.is_special("b"));
        facts.borrow_mut().prediction_ids = Some([a, b, a]);
        facts.borrow_mut().output_width = Some(domain.len());
        let mut returns = Vec::new();
        for manual in [false, true] {
            for add_special in [false, true] {
                for skip_special in [false, true] {
                    let text = "Hello e\u{301} world! १२ 123\n";
                    let expected = ordinary.encode(text, add_special).unwrap();
                    facts.borrow_mut().ids.clear();
                    let cancellation = GenerationCancellationToken::new();
                    let mut session = start_original_plain_string(
                        &mut runtime,
                        &source,
                        text,
                        config(),
                        &[],
                        &[],
                        add_special,
                        skip_special,
                        &cancellation,
                    )
                    .unwrap()
                    .unwrap();
                    assert_eq!(
                        facts.borrow().ids,
                        expected.get_ids(),
                        "{} original E→I",
                        artifact.label
                    );
                    let mut chunks = String::new();
                    let mut emit = |event: GenerationPlainTextEvent<'_>| {
                        if let GenerationPlainTextEvent::TextDelta(text) = event {
                            chunks.push_str(text);
                        }
                    };
                    let output = if manual {
                        while session.finish_reason().is_none() {
                            session = session.advance(&cancellation, &mut emit).unwrap();
                        }
                        session.into_output().unwrap_or_else(|_| panic!("terminal"))
                    } else {
                        session.run(&cancellation, &mut emit).unwrap()
                    };
                    assert_eq!(output.token_ids.as_ref(), [a, b, a]);
                    assert_eq!(
                        output.text.as_str(),
                        ordinary.decode(&[a, b, a], skip_special).unwrap()
                    );
                    assert_eq!(output.text.as_str(), chunks);
                    assert!(!chunks.is_empty());
                    let held = facts.borrow().held;
                    assert_eq!(pool.used_bytes().unwrap(), c + held);
                    let retained = output.text.clone();
                    drop(output);
                    assert_eq!(pool.used_bytes().unwrap(), c + held);
                    drop(retained);
                    assert_eq!(pool.used_bytes().unwrap(), c);
                    returns.push((manual, add_special, skip_special));
                }
            }
        }
        assert_eq!(facts.borrow().encodes, 8);
        assert_eq!(facts.borrow().stops, 8);
        let final_text = {
            let cancel = GenerationCancellationToken::new();
            let session = start_original_plain_string(
                &mut runtime,
                &source,
                "hello",
                config(),
                &[],
                &["b"],
                false,
                true,
                &cancel,
            )
            .unwrap()
            .unwrap();
            let output = session.run(&cancel, &mut |_| {}).unwrap();
            assert_eq!(output.text.as_str(), "a");
            assert_eq!(output.finish_reason, FinishReason::StopSequence);
            output.text.clone()
        };
        let held = facts.borrow().held;
        drop((source, runtime));
        assert_eq!(pool.used_bytes().unwrap(), held);
        assert_eq!(final_text.as_str(), "a");
        drop(final_text);
        assert_eq!(pool.used_bytes().unwrap(), 0);
        rows.push(serde_json::json!({"label":artifact.label,"sha256":sha,"domain_positions":domain_positions,"c":c,"ordinary_manual_special_skip_cases":returns.len(),"stop_override":true}));
    }
    println!(
        "ORIGINAL_TEXT_ORACLE_JSON={}",
        serde_json::json!({"artifacts":rows,"count":10,"private_string_cases":90,"scope":"source/domain/encode/real neutral I-R provider/private facade; native matrix is separate"})
    );
}

#[test]
fn original_text_source_rejections_stay_by_value_before_their_actual_allowance() {
    let (runtime, _, pool) = bare_runtime_with_capacity(u64::MAX);
    let sample = source(&runtime, false);
    let c = sample.original_bytes();
    let s = WorkingMemoryPool::stop_source_required_bytes(
        &eredu_text::stop_storage::StopCompilePlan::prepare_refs(&["halt"]).unwrap(),
    )
    .unwrap();
    let e = WorkingMemoryPool::tokenizer_encode_required_bytes(&sample, "hi", true).unwrap();
    drop((sample, runtime));
    assert_eq!(pool.used_bytes().unwrap(), 0);
    let mut file = tempfile::tempfile().unwrap();
    std::io::Write::write_all(&mut file, JSON.as_bytes()).unwrap();
    let read = eredu_checkpoint::artifact::PreparedArtifactFileRead::new(file).unwrap();
    let i = WorkingMemoryPool::tokenizer_file_required_bytes(&read).unwrap();
    let (runtime, facts, pool) = bare_runtime_with_capacity(i - 1);
    let error =
        Backend::compile_original_tokenizer_source_for_generation(&runtime, eredu_runtime::working_memory::OriginalTokenizerInput::File(read)).unwrap_err();
    let OriginalTokenizerSourceError::Input(cause) = &error else {
        panic!("actual file admission cause")
    };
    assert_eq!(cause.input_bytes(), 0);
    assert!(
        matches!(cause.accounting_failure(), Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes == i && *available_bytes == i - 1)
    );
    assert_eq!(pool.used_bytes().unwrap(), 0);
    assert!(facts.borrow().order.is_empty());
    drop((runtime, error));
    for (capacity, phase) in [(c + s - 1, 0), (c + s + e - 1, 1)] {
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(capacity);
        let source = source(&runtime, false);
        let result = start_original_plain_string(
            &mut runtime,
            &source,
            "hi",
            config(),
            &[],
            &["halt"],
            true,
            true,
            &GenerationCancellationToken::new(),
        );
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("one-short source admission succeeded"),
        };
        match &error {
            OriginalPlainStartError::Source(OriginalTextSourceError::Stop(error)) if phase == 0 => {
                assert_eq!(error.retained_bytes(), 0);
                assert!(
                    matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes==s && *available_bytes==s-1)
                );
            }
            OriginalPlainStartError::Source(OriginalTextSourceError::Encode(error))
                if phase == 1 =>
            {
                assert_eq!(error.retained_bytes(), 0);
                assert!(
                    matches!(error.accounting_failure(),Some(WorkingMemoryError::BudgetExceeded { required_bytes, available_bytes }) if *required_bytes==e && *available_bytes==e-1)
                );
            }
            _ => panic!("wrong actual source cause: {error:?}"),
        }
        assert_eq!(facts.borrow().stops, 1);
        assert_eq!(facts.borrow().encodes, 0);
        assert!(facts.borrow().order.is_empty());
        assert_eq!(pool.used_bytes().unwrap(), c);
        drop((source, runtime));
        assert_eq!(pool.used_bytes().unwrap(), 0);
        drop(error); // unadmitted closed causes own no payload/allowance
    }
}

#[test]
fn public_managed_plain_entry_authenticates_source_and_shares_controlled_output() {
    use crate::api::{ManagedPlainTextRequest, PreparedChatGenerationSettings};
    let mut previous = None;
    for manual in [false, true] {
        let (runtime, facts, pool) = bare_runtime_with_capacity(STRING_FIXTURE_CAPACITY);
        facts.borrow_mut().shared_admission_capacity = Some(STRING_FIXTURE_CAPACITY);
        let tokenizer = ChatTokenizer::from_bytes(JSON.as_bytes()).unwrap();
        let mut model = LoadedModel::from_runtime(
            runtime,
            tokenizer,
            LoadedTextModelConfig {
                model_family: ModelKind::Llama,
                effective_model_type: "llama".into(),
                model_id: "public-original-plain".into(),
                chat_template: None,
                eos_token_ids: vec![],
                checkpoint_generation_config: None,
            },
        )
        .unwrap();
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(JSON.as_bytes()).unwrap();
        let source = model.compile_managed_plain_text_source(file).unwrap();
        let source_bytes = pool.used_bytes().unwrap();
        // Same vocabulary and pipeline, different BPE merge policy: reject and refund C.
        let mut changed = tempfile::tempfile().unwrap();
        changed
            .write_all(
                JSON.replace("[\"h\",\"i\"],[\"Ġ\",\"hi\"]", "[\"h\",\"i\"]")
                    .as_bytes(),
            )
            .unwrap();
        let rejected = model
            .compile_managed_plain_text_source(changed)
            .unwrap_err();
        assert_eq!(
            rejected.input_rejection(),
            Some(TokenInputRejection::IdentityMismatch)
        );
        drop(rejected);
        assert_eq!(pool.used_bytes().unwrap(), source_bytes);
        let settings = PreparedChatGenerationSettings {
            overrides: GenerationConfigOverrides {
                max_new_tokens: Some(3),
                ..Default::default()
            },
            inference: TextInferencePolicy {
                managed_memory_capacity_bytes: Some(STRING_FIXTURE_CAPACITY),
                ..Default::default()
            },
            ..Default::default()
        };
        let request = ManagedPlainTextRequest::new("hi hi<S>", settings);
        let mut short = request;
        short.settings.inference.managed_memory_capacity_bytes = Some(1);
        let short_error = match model.start_managed_plain_text(
            &source,
            short,
            &GenerationCancellationToken::new(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("tiny ceiling must reject before source preparation"),
        };
        assert_eq!(facts.borrow().encodes, 0);
        assert_eq!(facts.borrow().stops, 0);
        assert_eq!(pool.used_bytes().unwrap(), source_bytes);
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(&short_error);
        let mut classified = false;
        while let Some(error) = cause {
            classified |= matches!(
                error.downcast_ref::<WorkingMemoryError>(),
                Some(WorkingMemoryError::CapacityBelowUsage {
                    capacity_bytes: 1,
                    ..
                })
            );
            cause = error.source();
        }
        assert!(classified, "{short_error:?}");
        drop(short_error);
        let cancelled = GenerationCancellationToken::new();
        cancelled.cancel();
        assert!(model
            .start_managed_plain_text(&source, request, &cancelled)
            .unwrap()
            .is_none());
        assert_eq!(facts.borrow().encodes, 0);
        assert_eq!(facts.borrow().stops, 0);
        let cancellation = GenerationCancellationToken::new();
        let mut visible = String::new();
        let mut event = |event: GenerationPlainTextEvent<'_>| {
            if let GenerationPlainTextEvent::TextDelta(text) = event {
                visible.push_str(text);
            }
        };
        let output = if manual {
            let mut session = model
                .start_managed_plain_text(&source, request, &cancellation)
                .unwrap()
                .unwrap();
            let mut address = None;
            while session.finish_reason().is_none() {
                session = session.advance(&cancellation, &mut event).unwrap();
                if let Some(previous) = address {
                    assert_eq!(session.token_ids().as_ptr(), previous);
                }
                address = Some(session.token_ids().as_ptr());
            }
            session
                .into_output()
                .unwrap_or_else(|_| panic!("terminal session"))
        } else {
            model
                .generate_managed_plain_text(&source, request, &cancellation, &mut event)
                .unwrap()
                .unwrap()
        };
        assert_eq!(visible, "h hih");
        assert_eq!(facts.borrow().ids, [2, 8, 4]);
        assert_eq!(facts.borrow().encodes, 1);
        assert_eq!(facts.borrow().stops, 1);
        if let Some((text, ids, reason)) = previous {
            assert_eq!(output.text.as_str(), text);
            assert_eq!(output.token_ids.as_ref(), ids);
            assert_eq!(output.finish_reason, reason);
        }
        previous = Some((
            output.text.as_str().to_owned(),
            output.token_ids.as_ref().to_vec(),
            output.finish_reason,
        ));
        let ptr = output.text.as_str().as_ptr();
        drop((source, model));
        assert_eq!(output.text.as_str().as_ptr(), ptr);
        assert!(pool.used_bytes().unwrap() > 0);
        drop(output);
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}

#[test]
fn original_plain_readiness_source_is_explicit_through_shared_startup_and_advancement() {
    let mut runs=Vec::new();
    for manual in [false,true] {
        let (mut runtime,facts,_pool)=bare_runtime_with_capacity(STRING_FIXTURE_CAPACITY);
        facts.borrow_mut().shared_admission_capacity=Some(STRING_FIXTURE_CAPACITY);
        facts.borrow_mut().preparation_control_enabled=true;
        let tokenizer=source(&runtime,true);
        let cancellation=GenerationCancellationToken::new();
        let mut session=start_original_plain_string(&mut runtime,&tokenizer,"hi hi<S>",config(),
            &[],&[],true,true,&cancellation).unwrap().unwrap();
        assert_eq!(facts.borrow().preparation_control_calls,
            [(Stage::Admission,Status::Ready),(Stage::Prompt,Status::Ready),(Stage::Sampling,Status::Ready)]);
        assert_eq!(facts.borrow().preparation_control_drops,0);
        let output=if manual {
            while session.finish_reason().is_none() {
                session=session.advance(&cancellation,&mut |_|{}).unwrap();
            }
            session.into_output().unwrap_or_else(|_|panic!("terminal"))
        } else {session.run(&cancellation,&mut |_|{}).unwrap()};
        assert_eq!(facts.borrow().preparation_control_drops,1);
        let calls=facts.borrow().preparation_control_calls.clone();
        assert!(calls.iter().any(|(stage,_)|*stage==Stage::Prediction));
        assert!(calls.iter().any(|(stage,_)|*stage==Stage::Decision));
        assert!(calls.iter().any(|(stage,_)|*stage==Stage::Commitment));
        assert!(calls.iter().any(|(stage,_)|*stage==Stage::Delivery));
        runs.push((output.token_ids.as_slice().to_vec(),output.text.as_str().to_owned(),calls));
    }
    assert_eq!(runs[0],runs[1]);
    let (mut runtime,facts,_pool)=bare_runtime_with_capacity(STRING_FIXTURE_CAPACITY);
    facts.borrow_mut().shared_admission_capacity=Some(STRING_FIXTURE_CAPACITY);
    facts.borrow_mut().preparation_control_failed=true;
    let tokenizer=source(&runtime,true);
    let result=start_original_plain_string(&mut runtime,&tokenizer,"hi",config(),&[],&[],true,true,
        &GenerationCancellationToken::new());
    assert!(result.is_err());
    assert!(facts.borrow().order.is_empty(),"failed source cannot enter model admission or ordinary consensus");
    assert!(facts.borrow().preparation_control_calls.is_empty());
    assert_eq!(facts.borrow().total_agreements,0);
}

#[test]
fn peer_startup_refusal_uses_the_source_authenticated_plain_driver() {
    for stage in [Stage::Admission, Stage::Prompt, Stage::Sampling] {
        let (mut runtime, facts, pool) = bare_runtime_with_capacity(STRING_FIXTURE_CAPACITY);
        facts.borrow_mut().shared_admission_capacity = Some(STRING_FIXTURE_CAPACITY);
        let tokenizer = source(&runtime, true);
        let baseline = pool.used_bytes().unwrap();
        facts.borrow_mut().reject = Some(stage);
        let failure = start_original_plain_string(&mut runtime, &tokenizer, "hi hi<S>",
            config(), &[], &[], true, true, &GenerationCancellationToken::new());
        assert!(failure.is_err());
        assert!(!facts.borrow().order.contains(&"submit"));
        assert_eq!(facts.borrow().ids.len(), if stage == Stage::Admission { 0 } else { 3 });
        drop(failure);
        assert_eq!(pool.used_bytes().unwrap(), baseline);
        drop((tokenizer, runtime));
        assert_eq!(pool.used_bytes().unwrap(), 0);
    }
}
