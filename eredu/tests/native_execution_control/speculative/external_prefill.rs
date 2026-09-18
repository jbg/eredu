//! Real Gemma assistant spans through ordinary and controlled public drivers.
use super::*;
use std::{cell::RefCell, rc::Rc};

fn sources() -> (Fixture, Fixture) {
    let (target, draft) = artifacts();
    for root in [&target.0, &draft.0] {
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("config.json")).unwrap()).unwrap();
        config["eos_token_id"] = serde_json::json!([]);
        config["text_config"]["eos_token_id"] = serde_json::json!([]);
        std::fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        let words = WordLevel::builder()
            .vocab((0..64).map(|id| (format!("word{id}"), id)).collect())
            .unk_token("word0".into())
            .build()
            .unwrap();
        let mut tokenizer = Tokenizer::new(words);
        tokenizer.with_pre_tokenizer(Some(Whitespace::default()));
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer.save(root.join("tokenizer.json"), false).unwrap();
        std::fs::write(
            root.join("chat_template.jinja"),
            "{% for m in messages %}{{ m.content }}{% endfor %} reply: ",
        )
        .unwrap();
    }
    (target, draft)
}

#[test]
#[cfg_attr(feature = "metal", ignore = "run CPU-only native initialization")]
fn gemma_external_shared_spans_match_full_controlled_and_snapshot_on_every_residency() {
    for residency in super::super::v3_components::residencies() {
        let (target, draft) = sources();
        selective_control_artifacts(
            target,
            draft,
            false,
            residency,
            true,
            0.0,
            std::num::NonZeroU64::new(2),
        );
    }
}

#[derive(Default)]
struct Trace {
    chunks: Vec<(u64, u64)>,
    rows: Vec<(u64, String, Vec<i32>)>,
    context: Vec<(u64, String, Vec<i32>)>,
    finished: Vec<bool>,
}
#[derive(Clone, Copy)]
enum Placement {
    Rows(u64),
    Context(u64),
}
struct Observer {
    trace: Rc<RefCell<Trace>>,
    placement: Option<Placement>,
    cancel: Option<eredu_core::GenerationCancellationToken>,
    context_supported: bool,
    sequence: bool,
}
impl<T: eredu_nn::Tensor, E> eredu_runtime::ActivationObserver<T, E> for Observer {
    fn requires_sequence_readout(&self) -> bool {
        self.sequence
    }
    fn supports_prefill_spans(&self) -> bool {
        true
    }
    fn supports_prefill_context(&self) -> bool {
        self.context_supported
    }
    fn begin_prefill_chunk(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
    ) -> Result<(), E> {
        let mut trace = self.trace.borrow_mut();
        let span = (chunk.input.start, chunk.input.end);
        if trace.chunks.last() != Some(&span) {
            trace.chunks.push(span);
        }
        self.placement = Some(Placement::Rows(chunk.input.start));
        Ok(())
    }
    fn begin_prefill_context(&mut self, frontier: u64) -> Result<(), E> {
        self.placement = Some(Placement::Context(frontier));
        Ok(())
    }
    fn observe(&mut self, path: &str, value: &T) -> Result<(), E> {
        match self.placement {
            Some(Placement::Rows(start)) => {
                self.trace
                    .borrow_mut()
                    .rows
                    .push((start, path.into(), value.shape().to_vec()));
                if start == 0 {
                    if let Some(cancel) = &self.cancel {
                        cancel.cancel();
                    }
                }
            }
            Some(Placement::Context(end)) => {
                self.trace
                    .borrow_mut()
                    .context
                    .push((end, path.into(), value.shape().to_vec()))
            }
            None => (),
        }
        Ok(())
    }
    fn finish_prefill(&mut self, committed: bool) {
        self.trace.borrow_mut().finished.push(committed);
        self.placement = None;
    }
}

fn values(
    records: &[eredu_core::speculative::SpeculativePredictionCapture],
) -> Vec<(SpeculativeCaptureRole, u64, Vec<(u32, f32)>)> {
    records
        .iter()
        .map(|record| {
            let Some(CapturePayload::Candidates(values)) =
                &record.capture.as_step().records[0].payload
            else {
                panic!("expected actual sampling candidates")
            };
            let mut values = values
                .candidates
                .iter()
                .map(|c| (c.token_id, c.score))
                .collect::<Vec<_>>();
            values.sort_by_key(|x| x.0);
            (record.role, record.position, values)
        })
        .collect()
}

fn has_typed_observer_refusal(mut error: &(dyn std::error::Error + 'static)) -> bool {
    loop {
        if matches!(
            error.downcast_ref::<eredu_core::speculative::SpeculativeControlError>(),
            Some(eredu_core::speculative::SpeculativeControlError::Unsupported(_))
        ) {
            return true;
        }
        let Some(source) = error.source() else {
            return false;
        };
        error = source;
    }
}

#[test]
#[cfg_attr(feature = "metal", ignore = "run CPU-only native initialization")]
fn gemma_external_context_attribution_cancellation_reuses_and_refusal_stays_typed() {
    let (target, draft) = sources();
    let plan = ExecutionPlan::fully_resident(local_device_plan(LocalDevice::Cpu).unwrap())
        .with_drafting(DraftingPlan::External {
            model: draft.0.display().to_string(),
            placement: DraftPlacementPlan::Target,
            max_draft_tokens: 2,
            lookahead: false,
            adaptive_lookahead: false,
        });
    let mut loaded =
        LoadedModel::load_execution_plan(&MlxBackendFactory::default(), &target.0, &plan).unwrap();
    let options = loaded.speculative_generation_options().unwrap().unwrap();
    let (model, drafting) = loaded.parts_mut();
    let chat = model
        .source_chat_with_capacity(
            ChatTemplateRequest {
                messages: vec![serde_json::json!({"role":"user", "content":"hello"})],
                add_generation_prompt: true,
                ..Default::default()
            },
            ORIGINAL_CAPACITY,
        )
        .unwrap();
    let settings = PreparedChatGenerationSettings {
        overrides: GenerationConfigOverrides {
            max_new_tokens: Some(13),
            temperature: Some(0.0),
            ..Default::default()
        },
        seed: 17,
        ..Default::default()
    };
    macro_rules! request {
        ($chunk:expr, $cancel:expr) => {
            PreparedChatSpeculativeRequest {
                chat: &chat,
                input: eredu::api::PreparedChatPrompt::TokenIds(&[1, 3, 2, 4, 5, 6]),
                output_mode: eredu::api::PreparedChatOutputMode::Text,
                skip_special_tokens: true,
                drafting: drafting.as_speculative_draft().unwrap(),
                settings: chat_settings(
                    &chat,
                    PreparedChatGenerationSettings {
                        inference: eredu_core::TextInferencePolicy {
                            prefill_chunk_positions: $chunk,
                            ..Default::default()
                        },
                        ..settings
                    },
                ),
                options: options.clone(),
                caller_stop_sequences: &[],
                cancellation: $cancel,
                on_event: |_| {},
            }
        };
    }
    let usage = CaptureUsage {
        captures: 1024,
        retained_bytes: 32 << 20,
        host_bytes: 32 << 20,
        encoded_bytes: 32 << 20,
    };
    let capture = model
        .prepare_speculative_capture(
            settings,
            CapturePlan {
                schema_version: CAPTURE_SCHEMA_VERSION,
                selections: vec![CaptureSelection {
                    id: "scores".into(),
                    path: eredu_core::MODEL_LOGITS_OBSERVATION_PATH.into(),
                    schedule: Default::default(),
                    slices: vec![],
                    transform: CaptureTransform::TopCandidates { count: 64 },
                }],
                limits: CaptureLimits {
                    per_step: usage,
                    cumulative: usage,
                    physical_native_bytes: None,
                    on_limit: CaptureLimitPolicy::Fail,
                },
            },
        )
        .unwrap();
    let control = ControlledSpeculativeOptions {
        capture: Some(capture),
        ..Default::default()
    };
    let mut baseline_scores = Vec::new();
    let baseline = model
        .generate_observed_prepared_chat_speculative(
            request!(None, Default::default()),
            control.clone(),
            |step| {
                baseline_scores.extend(step.captures.iter().cloned());
                ControlFlow::Continue(())
            },
        )
        .unwrap();
    let trace = Rc::new(RefCell::new(Trace::default()));
    let eredu_core::SpeculativeDraft::External(drafter) = drafting.as_speculative_draft().unwrap()
    else {
        panic!()
    };
    drafter
        .install_external_observers(
            Observer {
                trace: trace.clone(),
                placement: None,
                cancel: None,
                context_supported: true,
                sequence: true,
            },
            eredu_runtime::NoopObserver,
        )
        .unwrap();
    let mut observed_scores = Vec::new();
    let selected = model
        .generate_observed_prepared_chat_speculative(
            request!(std::num::NonZeroU64::new(2), Default::default()),
            control.clone(),
            |step| {
                observed_scores.extend(step.captures.iter().cloned());
                ControlFlow::Continue(())
            },
        )
        .unwrap();
    assert_eq!(selected.token_ids(), baseline.token_ids());
    assert_eq!(selected.token_ids().len(), 13);
    assert!(selected.stats().rounds() >= 3);
    let actual = values(&observed_scores);
    let expected = values(&baseline_scores);
    assert_eq!(actual.len(), expected.len());
    assert!(expected.iter().flat_map(|x| &x.2).any(|x| x.1 != 0.0));
    for (a, e) in actual.iter().zip(&expected) {
        assert_eq!((a.0, a.1), (e.0, e.1));
        assert_eq!(a.2.len(), e.2.len());
        for (a, e) in a.2.iter().zip(&e.2) {
            assert_eq!(a.0, e.0);
            assert!((a.1 - e.1).abs() <= 2e-4 * (1.0 + e.1.abs()));
        }
    }
    {
        let trace = trace.borrow();
        assert_eq!(trace.chunks, [(0, 2), (2, 4), (4, 6)]);
        assert_eq!(trace.finished, [true]);
        assert!(!trace.context.is_empty());
        for (end, _, shape) in &trace.context {
            assert_eq!(shape.len(), 4);
            assert_eq!(shape[2] as u64, *end);
        }
        assert!(trace.context.iter().any(|x| x.0 == 6));
        let logits_rows: Vec<_> = trace
            .rows
            .iter()
            .filter(|(_, path, _)| path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            .map(|(start, _, shape)| (*start, shape[1]))
            .collect();
        assert_eq!(logits_rows, [(0, 2), (2, 2), (4, 2)]);
    }
    // Default None reaches the same actual shared transaction, with one
    // demanded vocabulary row and complete target context. Scores are nonzero
    // and compared with the full-sequence observer run above.
    let default_trace = Rc::new(RefCell::new(Trace::default()));
    let eredu_core::SpeculativeDraft::External(drafter) = drafting.as_speculative_draft().unwrap()
    else {
        panic!()
    };
    drafter
        .install_external_observers(
            Observer {
                trace: default_trace.clone(),
                placement: None,
                cancel: None,
                context_supported: true,
                sequence: false,
            },
            eredu_runtime::NoopObserver,
        )
        .unwrap();
    let mut default_scores = Vec::new();
    let default = model
        .generate_observed_prepared_chat_speculative(
            request!(None, Default::default()),
            control.clone(),
            |step| {
                default_scores.extend(step.captures.iter().cloned());
                ControlFlow::Continue(())
            },
        )
        .unwrap();
    assert_eq!(default.token_ids(), baseline.token_ids());
    let actual_default = values(&default_scores);
    assert_eq!(actual_default.len(), expected.len());
    for (a, e) in actual_default.iter().zip(&expected) {
        assert_eq!((a.0, a.1), (e.0, e.1));
        assert_eq!(a.2.len(), e.2.len());
        for (a, e) in a.2.iter().zip(&e.2) {
            assert_eq!(a.0, e.0);
            assert!((a.1 - e.1).abs() <= 2e-4 * (1.0 + e.1.abs()));
        }
    }
    {
        let trace = default_trace.borrow();
        assert_eq!(trace.chunks, [(0, 6)]);
        assert_eq!(trace.finished, [true]);
        let rows = trace
            .rows
            .iter()
            .filter(|(_, path, _)| path == eredu_core::MODEL_LOGITS_OBSERVATION_PATH)
            .map(|(start, _, shape)| (*start, shape[1]))
            .collect::<Vec<_>>();
        assert_eq!(rows, [(0, 1)]);
        assert!(!trace.context.is_empty());
        assert!(
            trace
                .context
                .iter()
                .all(|(end, _, shape)| *end == 6 && shape[2] == 6)
        );
    }
    for supported in [true, false] {
        let trace = Rc::new(RefCell::new(Trace::default()));
        let cancellation = eredu_core::GenerationCancellationToken::new();
        let eredu_core::SpeculativeDraft::External(drafter) =
            drafting.as_speculative_draft().unwrap()
        else {
            panic!()
        };
        drafter
            .install_external_observers(
                Observer {
                    trace: trace.clone(),
                    placement: None,
                    cancel: supported.then(|| cancellation.clone()),
                    context_supported: supported,
                    sequence: true,
                },
                eredu_runtime::NoopObserver,
            )
            .unwrap();
        let mut captures = Vec::new();
        let result = model.generate_observed_prepared_chat_speculative(
            request!(std::num::NonZeroU64::new(2), cancellation),
            control.clone(),
            |step| {
                captures.extend(step.captures.iter().cloned());
                ControlFlow::Continue(())
            },
        );
        if supported {
            let result = result.unwrap();
            assert!(result.token_ids().is_empty());
            assert_eq!(result.stats().target_tokens(), 2);
            assert_eq!(result.stats().draft_tokens(), 0);
            assert_eq!(result.stats().rounds(), 0);
            assert!(captures.is_empty());
            assert_eq!(trace.borrow().chunks, [(0, 2)]);
        } else {
            let Err(error) = result else {
                panic!("incompatible observer was accepted");
            };
            assert!(has_typed_observer_refusal(&error), "{error:?}");
            assert!(trace.borrow().chunks.is_empty());
        }
        assert_eq!(trace.borrow().finished, [false]);
        let eredu_core::SpeculativeDraft::External(drafter) =
            drafting.as_speculative_draft().unwrap()
        else {
            panic!()
        };
        let install = drafter
            .install_external_observers(eredu_runtime::NoopObserver, eredu_runtime::NoopObserver);
        if supported {
            install.unwrap();
            let retry = model
                .generate_prepared_chat_speculative(request!(
                    std::num::NonZeroU64::new(2),
                    Default::default()
                ))
                .unwrap();
            assert_eq!(retry.token_ids(), baseline.token_ids());
        } else {
            // Existing DrafterOperation conservatively fences every failed
            // operation, even when this observer rejected before target work.
            assert!(install.is_err());
        }
    }
}
