//! Native realization of the same C witness and E→I/R mechanisms. Facade policy
//! stays in the facade; this test controller supplies only the identical neutral
//! domain contract. Numerical reference uses the existing full logical K/V helper.
use super::*;
use eredu_core::{
    BackendFailure, GenerationPlainTextOutput, GenerationSequenceConsumerLayout, GenerationTiming,
    OriginalTokenDomainWitness, TextControllerStorage, TextControllerWorkspace, TokenFilter,
    TokenFilterController, TokenSamplingDecision,
};
use eredu_runtime::working_memory::{
    AggregateGenerationDecoderInput, OriginalChatBackend, OriginalEncodedTokenIds,
    OriginalRenderedChat, OriginalTokenizer, OriginalTokenizerBackend,
};
use eredu_text::{
    chat_storage::{ChatMessages, ChatTemplatePlan, TextMessage},
    stop_storage::StopCompilePlan,
    tokenizer_storage::TokenizerPlan,
};
pub(in crate::composition::mlx::session::model_session::text_quote) struct Domain(
    pub(in crate::composition::mlx::session::model_session::text_quote) OriginalTokenizer,
);
impl TokenFilterController for Domain {
    type Error = std::convert::Infallible;
    fn inference_workspace(&self, _: u64) -> Option<TextControllerWorkspace<'_>> {
        Some(TextControllerWorkspace {
            filter: self.0.generation_domain().unwrap().into(),
            additional_host_bytes: 0,
        })
    }
    fn inference_storage(&self) -> TextControllerStorage<'_> {
        TextControllerStorage::RunOwnedWithOriginalTokenDomain(OriginalTokenDomainWitness::new(
            &self.0,
        ))
    }
    fn current_filter(&mut self) -> Result<TokenFilter, Self::Error> {
        Ok(self.0.generation_domain().unwrap().clone())
    }
    fn current_decision(&mut self) -> Result<TokenSamplingDecision<'_>, Self::Error> {
        Ok(TokenSamplingDecision::new(self.current_filter()?)
            .with_original_tokenizer_validity(
                self.0.generation_domain().unwrap(),
                OriginalTokenDomainWitness::new(&self.0),
            )
            .with_controller_storage(self.inference_storage()))
    }
    fn commit_token(&mut self, id: u32) -> Result<(), Self::Error> {
        assert!(self.0.generation_domain().unwrap().allows(id));
        Ok(())
    }
    fn is_complete(&mut self) -> Result<bool, Self::Error> {
        Ok(false)
    }
}
pub(in crate::composition::mlx::session::model_session::text_quote) fn json() -> String {
    let mut vocab = serde_json::Map::new();
    for id in 0..32 {
        let token = match id {
            2 => "a".to_owned(),
            5 => "b".to_owned(),
            7 => "c".to_owned(),
            3 => "d".to_owned(),
            11 => "e".to_owned(),
            _ => char::from_u32(0x400 + id).unwrap().to_string(),
        };
        vocab.insert(token, id.into());
    }
    serde_json::json!({"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":vocab,"merges":[]}}).to_string()
}
// This changes only fixture spellings, preserving the 32-wide native model and
// the five existing IDs. Every byte of the actual rendered chat is represented.
fn chat_json(prompt: &str) -> String {
    let mut value: serde_json::Value = serde_json::from_str(&json()).unwrap();
    let vocab = value["model"]["vocab"].as_object_mut().unwrap();
    let mut free = (0..32).filter(|id| ![2, 5, 7, 3, 11].contains(id));
    for ch in prompt.chars() {
        let spelling = ch.to_string();
        if !vocab.contains_key(&spelling) {
            let id = free
                .next()
                .expect("actual chat alphabet fits the native vocabulary");
            vocab.remove(&char::from_u32(0x400 + id).unwrap().to_string());
            vocab.insert(spelling, id.into());
        }
    }
    assert_eq!(vocab.len(), 32);
    value.to_string()
}
fn consumer() -> GenerationSequenceConsumerLayout {
    GenerationSequenceConsumerLayout::for_driver_with_terminal_text::<
        eredu_core::RetainedGenerationSequence,
        Error,
        BackendFailure,
    >()
    .unwrap()
}
fn finish_sequence(
    mut sequence: eredu_core::RetainedGenerationSequence,
) -> GenerationPlainTextOutput {
    let pending = sequence.finish_plain_text().unwrap();
    assert!(pending.is_empty());
    let reason = sequence.finish_reason().unwrap();
    GenerationPlainTextOutput::from_retained(
        sequence.into_token_ids(),
        reason,
        GenerationTiming::default(),
    )
    .unwrap()
}
fn project(sequence: &mut eredu_core::RetainedGenerationSequence, id: u32) {
    assert!(!sequence.project_plain_text(id).unwrap().stop_matched);
    sequence
        .commit(id, TokenTerminalSignals::default())
        .unwrap();
}
fn run_original(
    runtime: &mut Runtime,
    source: &OriginalTokenizer,
    encoded: OriginalEncodedTokenIds,
    rendered: Option<OriginalRenderedChat>,
    expected_input: &[u32],
    route: usize,
    policy: &[&str],
    tracking_capacity: u64,
) -> (GenerationPlainTextOutput, u64) {
    let probe = Probe::new(runtime, None, false);
    let stops = MlxBackend::compile_original_text_stop_source(
        runtime,
        StopCompilePlan::prepare_refs(policy).unwrap(),
    )
    .unwrap();
    let header = AggregateGenerationDecoderInput::new_plain_text(source, &stops, 4, true).unwrap();
    let layout = consumer();
    let cfg = config(4, u64::MAX).with_inference_policy(eredu_core::TextInferencePolicy {
        prefill_chunk_positions: std::num::NonZeroU64::new(2),
        graph_metadata_capacity_bytes: std::num::NonZeroU64::new(GRAPH),
        submission_tracking_capacity_bytes: std::num::NonZeroU64::new(tracking_capacity),
        ..config(4, u64::MAX).inference_policy()
    });
    assert!(encoded.matches_source(source));
    assert_eq!(encoded.ids(), expected_input);
    if let Some(rendered) = &rendered {
        assert!(rendered.tokenizer_source().same_source(source));
        assert!(rendered.accepts_consumer(&layout));
    }
    let output = if route < 2 {
        let mut generation = ControlledTextGeneration::from_token_ids_with_sequence(
            runtime,
            TokenIdsInputPlan::new(encoded.ids()).unwrap(),
            cfg,
            Domain(source.clone()),
            None,
            GenerationSequenceRequest::new(4, &[])
                .with_decoder(&header)
                .with_consumer(&layout),
        )
        .unwrap();
        drop(encoded); // genuine I already copied; no native reference to E remains
        drop(rendered); // H survives E and the actual I handoff, then retires
        let sequence = generation.take_prepared_sequence().unwrap();
        let local = sequence.prepare_storage().map(Some).map_err(|error| {
            BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error)
        });
        let mut sequence = generation
            .finish_text_preparation_cancellable(
                eredu_core::run_preparation::TextPreparationStage::Delivery,
                local,
                |e| e,
            )
            .unwrap()
            .unwrap();
        if route == 0 {
            while sequence.finish_reason().is_none() {
                let token = generation.next().unwrap().unwrap_or_else(|error| {
                    use std::error::Error as _;
                    let mut causes = vec![error.to_string()];
                    let mut source = error.source();
                    while let Some(cause) = source {
                        causes.push(cause.to_string());
                        source = cause.source();
                    }
                    panic!("original generation failed: {}", causes.join(" -> "));
                });
                project(&mut sequence, token.token_id());
                drop(token);
                assert!(generation.take_captured_delivery().unwrap().is_none());
            }
        } else {
            for _ in 0..4 {
                let token = generation.next().unwrap().unwrap();
                project(&mut sequence, token.token_id());
                drop(token);
                assert!(generation.take_captured_delivery().unwrap().is_none());
            }
        }
        finish_sequence(sequence)
    } else {
        let mut driver = TextGenerationDriver::new(runtime);
        let mut run = driver
            .start_token_ids_with_sequence(
                TokenIdsInputPlan::new(encoded.ids()).unwrap(),
                cfg,
                Domain(source.clone()),
                None,
                GenerationSequenceRequest::new(4, &[])
                    .with_decoder(&header)
                    .with_consumer(&layout),
            )
            .unwrap();
        drop(encoded);
        drop(rendered);
        let sequence = driver.take_prepared_sequence(&mut run).unwrap().unwrap();
        let local = sequence.prepare_storage().map(Some).map_err(|error| {
            BackendFailure::new(eredu_core::BackendFailureKind::ResourceExhausted, error)
        });
        let mut sequence = driver
            .runtime()
            .finish_text_preparation_cancellable(
                eredu_core::run_preparation::TextPreparationStage::Delivery,
                local,
                |e| e,
            )
            .unwrap()
            .unwrap();
        for _ in 0..4 {
            let token = driver.advance(&mut run).unwrap().unwrap();
            project(&mut sequence, token.token_id());
            drop(token);
            assert!(driver.take_completed_delivery(&mut run).unwrap().is_none());
        }
        finish_sequence(sequence)
    };
    let preparation = probe.take();
    let quote = preparation.quote.as_ref().unwrap();
    assert!(quote.storage_contract().has_original_domain());
    assert_eq!(probe.0.calls.get(), 1);
    assert_eq!(probe.0.decoder_takes.get(), 1);
    assert!(
        !quote
            .original_token_input()
            .unwrap()
            .is_unclaimed_for_test()
    );
    assert!(quote.sequence.as_ref().unwrap().pending.borrow().is_none());
    assert!(quote.graph_quota.as_ref().unwrap().occupied_bytes() <= GRAPH as usize);
    let held = probe.facts().held;
    drop((preparation, probe));
    (output, held)
}
#[test]
fn canonical_c_e_i_r_text_matches_four_predictions_and_full_kv_across_cached_residency_routes() {
    compare_original(false);
}
#[test]
fn original_j_h_chat_matches_four_predictions_and_full_kv_across_cached_residency_routes() {
    compare_original(true);
}
fn compare_original(chat: bool) {
    let stream = stream();
    let config_source = include_str!("original_chat_tokenizer_config.json");
    let messages = [TextMessage {
        role: "system",
        content: "abcde",
    }];
    // Ordinary public template selection/rendering provides the full byte oracle;
    // the native adapter below only supplies original J/H/C and E/I/R mechanisms.
    let chat_prompt = if chat {
        let selected = eredu_text::tokenizer::load_model_chat_template_from_str(config_source)
            .unwrap()
            .unwrap();
        let mut tokenizer =
            eredu_text::tokenizer::Tokenizer::from_bytes(json().as_bytes()).unwrap();
        tokenizer
            .apply_chat_template_json(
                selected,
                [vec![
                    serde_json::json!({"role":"system", "content":"abcde"}),
                ]],
                None,
                "native",
                false,
                None,
            )
            .unwrap()
            .pop()
            .unwrap()
    } else {
        "abcde".to_owned()
    };
    let json = if chat {
        chat_json(&chat_prompt)
    } else {
        json()
    };
    let ordinary = eredu_text::tokenizer::Tokenizer::from_bytes(json.as_bytes())
        .unwrap()
        .snapshot();
    let input = ordinary.encode(chat_prompt.as_str(), false).unwrap();
    let expected_input = input.get_ids();
    if !chat {
        assert_eq!(expected_input, [2, 5, 7, 3, 11]);
    }
    assert!(!expected_input.is_empty());
    let positions_per_request = expected_input.len() + 3;
    // The character-level chat tokenizer emits 35 prompt tokens. Two cached
    // requests also retain three decoded positions each and reserve one final
    // output position. Keep the ordinary 32-position fixtures unchanged, and
    // give this comparison's actual source model enough context for both runs.
    let load_pair = |pool: &WorkingMemoryPool, residency| {
        if !chat {
            return load(&stream, pool, residency);
        }
        let maximum_positions = 2 * positions_per_request + 1;
        match residency {
            0 => host::runtime_with_context(&stream, pool, None, maximum_positions),
            1 => host::runtime_with_context(&stream, pool, Some(1), maximum_positions),
            2 => disk::load_runtime_with_context(&stream, pool, true, maximum_positions),
            _ => unreachable!(),
        }
    };
    for residency in 0..3 {
        for route in 0..3 {
            let mut reference = Vec::new();
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = load_pair(&pool, residency);
            for request in 0..2 {
                let mut ids = expected_input.to_vec();
                let values = run(&mut runtime, route, false, &mut ids);
                runtime.synchronize().unwrap();
                reference.push((
                    values,
                    logical_kv(&runtime, positions_per_request * (request + 1)),
                ));
            }
            finish(runtime, &stream);
            settle(&pool, 0);
            let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
            let (mut runtime, _artifact) = load_pair(&pool, residency);
            let source = MlxBackend::compile_original_tokenizer(
                &runtime,
                TokenizerPlan::prepare_json(json.as_bytes())
                    .unwrap()
                    .with_generation_domain()
                    .unwrap(),
            )
            .unwrap();
            assert_eq!(
                source.generation_domain().unwrap(),
                &TokenFilter::Allowed(vec![true; 32])
            );
            let template = chat.then(|| {
                MlxBackend::compile_original_chat_template(
                    &runtime,
                    ChatTemplatePlan::prepare_config(config_source.as_bytes(), "native").unwrap(),
                )
                .unwrap()
            });
            // A source declaration alone cannot enable a legacy/no-claim start.
            // Rejection precedes Prompt/Sampling, leaving the genuine C reusable.
            if residency == 0 && route == 0 {
                let before = pool.used_bytes().unwrap();
                let result = ControlledTextGeneration::new(
                    &mut runtime,
                    expected_input.to_vec(),
                    config(4, u64::MAX),
                    Domain(source.clone()),
                );
                assert!(matches!(
                    result,
                    Err(eredu_core::ControlledTextGenerationError::Preparation(_))
                ));
                drop(result);
                assert_eq!(pool.used_bytes().unwrap(), before);
            }
            let mut outputs = Vec::new();
            for (request, (expected, expected_kv)) in reference.iter().enumerate() {
                let rendered = template.as_ref().map(|template| {
                    let render = MlxBackend::render_original_chat(
                        &runtime,
                        template,
                        &source,
                        ChatMessages::from_text(&messages),
                        consumer(),
                    )
                    .unwrap();
                    assert!(render.has_sources(template, &source));
                    assert_eq!(render.prompt(false), chat_prompt);
                    assert_eq!(
                        render.prompt(true),
                        format!("{chat_prompt}<|im_start|>assistant\n")
                    );
                    render
                });
                let text = rendered
                    .as_ref()
                    .map_or("abcde", |render| render.prompt(false));
                let encoded =
                    MlxBackend::encode_original_text_ids(&runtime, &source, text, !chat).unwrap();
                let (output, held) = run_original(
                    &mut runtime,
                    &source,
                    encoded,
                    rendered,
                    expected_input,
                    route,
                    if request == 0 { &["NEVER"] } else { &[] },
                    1 << 20,
                );
                assert_eq!(output.token_ids.as_ref(), expected);
                assert_eq!(
                    output.text.as_str(),
                    ordinary.decode(expected, true).unwrap()
                );
                assert!(!output.text.as_str().is_empty());
                runtime.synchronize().unwrap();
                let actual = logical_kv(&runtime, positions_per_request * (request + 1));
                assert_eq!(actual.len(), expected_kv.len());
                for ((shape, values), (other_shape, other)) in actual.iter().zip(expected_kv) {
                    assert_eq!(shape, other_shape);
                    assert_eq!(values.len(), other.len());
                    assert!(
                        values
                            .iter()
                            .zip(other)
                            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5)
                    );
                }
                outputs.push((output, held));
            }
            let total = outputs.iter().map(|(_, held)| held).sum::<u64>();
            finish(runtime, &stream);
            settle(
                &pool,
                source.original_bytes()
                    + template.as_ref().map_or(0, |j| j.original_bytes())
                    + total,
            );
            drop(template);
            settle(&pool, source.original_bytes() + total);
            drop(source);
            settle(&pool, total);
            let mut remaining = total;
            while let Some((output, held)) = outputs.pop() {
                let text = output.text.clone();
                let ids = output.token_ids.clone().into_iter();
                drop(output);
                settle(&pool, remaining);
                drop(text);
                settle(&pool, remaining);
                drop(ids);
                remaining -= held;
                settle(&pool, remaining);
            }
        }
    }
}

fn first_original_resident_request(
    load: impl Fn(&Stream, &WorkingMemoryPool) -> (Runtime, tempfile::TempDir),
    snapshot: impl Fn(&Runtime, usize) -> Vec<(Vec<i32>, Vec<f32>)>,
    derive_tracking: bool,
) {
    let stream = stream();
    let input = [2, 5, 7, 3, 11];
    let ordinary_pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut ordinary, _ordinary_artifact) = load(&stream, &ordinary_pool);
    // No managed sequence/control request participates in the ordinary
    // numerical reference. Both sides still use the shared text driver.
    let ordinary_config = TextGenerationConfig::new(config(4, u64::MAX).sampling())
        .with_seed(19)
        .with_inference_policy(eredu_core::TextInferencePolicy {
            prefill_chunk_positions: std::num::NonZeroU64::new(2),
            ..Default::default()
        });
    let expected = TextGeneration::new(&mut ordinary, input.to_vec(), ordinary_config)
        .unwrap()
        .map(|token| token.unwrap().token_id().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 4);
    ordinary.synchronize().unwrap();
    let expected_kv = snapshot(&ordinary, 8);
    assert!(
        expected_kv
            .iter()
            .flat_map(|(_, values)| values)
            .any(|v| *v != 0.0)
    );
    finish(ordinary, &stream);
    settle(&ordinary_pool, 0);

    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let (mut runtime, _artifact) = load(&stream, &pool);
    assert!(
        runtime
            .session()
            .payload
            .model
            .erased()
            .retained_inference_authority()
            .unwrap()
            .admission()
            .is_none()
    );
    let source = MlxBackend::compile_original_tokenizer(
        &runtime,
        TokenizerPlan::prepare_json(json().as_bytes())
            .unwrap()
            .with_generation_domain()
            .unwrap(),
    )
    .unwrap();
    let encoded = MlxBackend::encode_original_text_ids(&runtime, &source, "abcde", true).unwrap();
    let tracking_capacity = if derive_tracking {
        let geometry = eredu_core::InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: input.len() as u64,
            max_output_tokens: 4,
            prefill_chunk_positions: 2,
            output: eredu_core::OutputDemand::LastPosition,
        };
        let (_, _, recipe) = runtime
            .session()
            .payload
            .model
            .quote_registered_resident_text_with_sampling_recipe(
                geometry,
                chunked_original(config(4, u64::MAX)),
                source.generation_domain().unwrap(),
                &pool,
            )
            .unwrap();
        let storage = recipe.record_storage_requirement().unwrap();
        assert!(storage.minimum_capacity > 0);
        let full = storage
            .full_capacity
            .expect("complete selected resident Record producer");
        let too_small = full - 1;
        assert!(matches!(
            storage.validate_capacity(too_small),
            Err(eredu_runtime::working_memory::WorkingMemoryError::SubmissionTrackingCapacity {
                required_bytes,
                configured_bytes,
            }) if required_bytes == full && configured_bytes == too_small
        ));
        eprintln!(
            "hybrid Record bytes: minimum={}, constructors={}, full={}",
            storage.minimum_capacity, storage.known_constructor_bytes, full
        );
        // Complete Record fit includes every attempted constructor and its
        // physical split-tail requirement. Other native domains stay separate.
        full
    } else {
        1 << 20
    };
    // Real public controlled startup: genuine C/E/I/R and no predecessor request.
    let (output, held) = run_original(
        &mut runtime,
        &source,
        encoded,
        None,
        &input,
        0,
        &[],
        tracking_capacity,
    );
    assert_eq!(output.token_ids.as_ref(), expected);
    assert!(!output.text.as_str().is_empty());
    runtime.synchronize().unwrap();
    let actual = snapshot(&runtime, 8);
    assert_eq!(actual.len(), expected_kv.len());
    for ((shape, values), (expected_shape, expected_values)) in actual.iter().zip(&expected_kv) {
        assert_eq!(shape, expected_shape);
        assert_eq!(values.len(), expected_values.len());
        assert!(
            values
                .iter()
                .zip(expected_values)
                .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5)
        );
    }
    finish(runtime, &stream);
    settle(&pool, source.original_bytes() + held);
    drop(source);
    settle(&pool, held);
    drop(output);
    settle(&pool, 0);
}

#[test]
fn first_original_resident_request_matches_ordinary_tokens_and_nonzero_kv() {
    first_original_resident_request(|stream, pool| load(stream, pool, 0), logical_kv, false);
}

#[test]
fn first_original_hybrid_resident_request_matches_ordinary_tokens_and_all_state() {
    use crate::composition::mlx::replicated_text::tests::{
        qwen_hybrid_config, tiny_heterogeneous_artifact,
    };
    first_original_resident_request(
        |stream, pool| {
            let mut config = qwen_hybrid_config();
            config["vocab_size"] = 32.into();
            config["num_hidden_layers"] = 3.into();
            config["layer_types"] =
                serde_json::json!(["linear_attention", "full_attention", "linear_attention"]);
            host::runtime_from_artifact(stream, pool, None, tiny_heterogeneous_artifact(config))
        },
        |runtime, positions| {
            let plan = runtime
                .session()
                .payload
                .model
                .erased()
                .prepare_resident_decoder_copy()
                .unwrap();
            let mut state = Vec::new();
            let mut kinds = [0usize; 3];
            plan.visit_operands(&mut |array| {
                let mut shape = array.shape().to_vec();
                let evaluated = array.evaluated().unwrap();
                let values = evaluated.as_slice::<f32>();
                let logical = match shape.as_slice() {
                    // This fixture declares two KV heads of width eight.
                    [1, 2, capacity, 8] => {
                        kinds[0] += 1;
                        assert!(*capacity as usize >= positions);
                        let mut logical = Vec::new();
                        for head in 0..2 {
                            let start = head * *capacity as usize * 8;
                            logical.extend_from_slice(&values[start..start + positions * 8]);
                        }
                        shape[2] = positions as i32;
                        logical
                    }
                    // Every recurrent layer retains all four value heads.
                    [1, 4, 8, 8] => {
                        kinds[1] += 1;
                        values.to_vec()
                    }
                    // The causal convolution retains three positions and 64 channels.
                    [1, 3, 64] => {
                        kinds[2] += 1;
                        values.to_vec()
                    }
                    other => panic!("unexpected fixture state geometry: {other:?}"),
                };
                assert!(logical.iter().all(|value| value.is_finite()));
                assert!(logical.iter().any(|value| *value != 0.0));
                state.push((shape, logical));
            });
            assert_eq!(
                kinds,
                [2, 2, 2],
                "K/V plus both recurrent and convolution states"
            );
            state
        },
        true,
    );
}

#[test]
fn native_recipe_roots_preserve_selected_layerwise_operation_obligations() {
    let stream = stream();
    let geometry = eredu_core::InferenceGeometry {
        batch_size: 1,
        cached_positions: 0,
        input_positions: 5,
        max_output_tokens: 4,
        prefill_chunk_positions: 2,
        output: eredu_core::OutputDemand::LastPosition,
    };
    let config = chunked_original(config(4, u64::MAX));
    let graph = config
        .inference_policy()
        .graph_metadata_capacity_bytes
        .unwrap();
    for residency in 0..3 {
        let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
        let (runtime, _artifact) = load(&stream, &pool, residency);
        let model = &runtime.session().payload.model;
        let (quote, storage, recipe) = model
            .quote_registered_resident_text_with_sampling_recipe(
                geometry,
                config,
                &TokenFilter::All,
                &pool,
            )
            .unwrap();
        let retained_sources = model.layerwise_workspace().unwrap();
        let roots = recipe.maximum_roots().unwrap();
        assert!(roots > 0);
        let facts = model
            .erased()
            .prefill_control_facts(
                &pool,
                geometry,
                graph,
                Some(roots),
                retained_sources.as_ref(),
                None,
                None,
            )
            .unwrap()
            .unwrap();
        assert_eq!(facts.root_capacity(), roots);
        if residency == 0 {
            assert_eq!(facts.operation_control_bytes(), Some(0));
            assert!(facts.total_bytes().unwrap().is_some());
        } else {
            // A valid numerical recipe does not pay the independent selected
            // host promotion/direct-read and operation-bank owners.
            assert_eq!(facts.operation_control_bytes(), None);
            assert_eq!(facts.total_bytes().unwrap(), None);
        }
        drop((quote, storage, recipe, retained_sources));
        finish(runtime, &stream);
        settle(&pool, 0);
    }
}
