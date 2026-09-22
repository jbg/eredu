use super::*;
use crate::composition::mlx::session::model_session::text_quote::preparation::tests::with_foreign_runtime;
#[cfg(test)]
use crate::memory_fixture::LedgerFixture as _;
use eredu_core::{TokenIdsInputPlan, TokenInputRejection};
use safemlx::SubmissionScopeOwnerCause;

// Explicit fixture ceiling, not a production default or graph-fit report.
const GRAPH: u64 = 4 << 20;

struct GraphRequest {
    probe: Probe,
    arena: Option<safemlx::SubmissionGraphQuota>,
    observations: usize,
    calls: crate::backend::submission_recovery::prediction::test_counts::Calls,
}
impl GraphRequest {
    fn new(runtime: &Runtime) -> Self {
        Self {
            probe: Probe::new(runtime, None, false),
            arena: None,
            observations: 0,
            calls: crate::backend::submission_recovery::prediction::test_counts::Calls::new(),
        }
    }
    fn observe(&mut self) {
        // Clone only the existing quote handle; release the probe loan before
        // native inspection and before dropping any owning or error value.
        let quote = {
            self.probe
                .0
                .admitted
                .borrow()
                .as_ref()
                .unwrap()
                .quote
                .as_ref()
                .unwrap()
                .clone()
        };
        assert_eq!(self.probe.0.calls.get(), 1, "one genuine I/R extraction");
        assert_eq!(quote.source_capacity_bytes, 5 * 4);
        let input = quote.original_token_input().expect("genuine original I");
        assert!(!input.is_unclaimed_for_test());
        let replay = input.take().unwrap_err();
        assert_eq!(
            cause::<TokenInputRejection>(&replay),
            &TokenInputRejection::Unavailable,
            "the original destination cannot be taken again or refilled"
        );
        drop(replay);
        assert!(quote.sequence.as_ref().unwrap().pending.borrow().is_none());
        assert_eq!(
            quote
                .config
                .inference_policy()
                .graph_metadata_capacity_bytes,
            std::num::NonZeroU64::new(GRAPH)
        );
        let arena = quote.graph_quota.as_ref().expect("same original graph Q");
        if let Some(original) = &self.arena {
            assert!(original.same_arena(arena), "no replacement or refill arena");
        } else {
            self.arena = Some(arena.clone());
        }
        let occupied = arena.occupied_bytes();
        assert!(occupied > 0 && occupied <= GRAPH as usize);
        let facts = self.probe.facts();
        assert!(!facts.source && facts.r > 0 && facts.held > GRAPH);
        assert!(quote.original_controls().is_some());
        self.observations += 1;
    }
    fn finish(mut self) -> (safemlx::SubmissionGraphQuota, u64) {
        assert_eq!(self.observations, 5, "startup and all four predictions");
        assert_eq!(self.calls.original(), [4; 5]);
        let held = self.probe.facts().held;
        let arena = self.arena.take().unwrap();
        drop(self.probe.take());
        // This fixture carries no probe-held quote, preparation or role
        // counter into the next request, only the actual arena. The runtime
        // retains its own cached state and original request owners.
        drop(self);
        (arena, held)
    }
}

fn run(runtime: &mut Runtime, route: usize, original: bool, ids: &mut [u32]) -> Vec<u32> {
    run_with_graph(runtime, route, original, ids, None)
}
enum FixtureInput<'a> {
    Mutable(&'a mut [u32]),
    Encoded(&'a eredu_runtime::working_memory::OriginalEncodedTokenIds),
}
impl FixtureInput<'_> {
    fn tokens(&self) -> &[u32] {
        match self {
            Self::Mutable(ids) => ids,
            Self::Encoded(ids) => ids.ids(),
        }
    }
    fn after_startup(&mut self) {
        if let Self::Mutable(ids) = self {
            ids.fill(1);
        }
    }
}
fn run_with_graph(
    runtime: &mut Runtime,
    route: usize,
    original: bool,
    ids: &mut [u32],
    graph: Option<&mut GraphRequest>,
) -> Vec<u32> {
    run_inputs(runtime, route, original, FixtureInput::Mutable(ids), graph)
}
fn run_inputs(
    runtime: &mut Runtime,
    route: usize,
    original: bool,
    mut ids: FixtureInput<'_>,
    mut graph: Option<&mut GraphRequest>,
) -> Vec<u32> {
    assert!(graph.is_none() || original);
    let cfg = config(4, u64::MAX).with_inference_policy(eredu_core::TextInferencePolicy {
        prefill_chunk_positions: std::num::NonZeroU64::new(2),
        graph_metadata_capacity_bytes: graph
            .as_ref()
            .and_then(|_| std::num::NonZeroU64::new(GRAPH)),
        submission_tracking_capacity_bytes: graph
            .as_ref()
            .and_then(|_| std::num::NonZeroU64::new(1 << 20)),
        ..config(4, u64::MAX).inference_policy().clone()
    });
    let request = GenerationSequenceRequest::new(4, &[]);
    let mut output = Vec::new();
    match route {
        0 => {
            let mut generation = if original {
                TextGeneration::from_token_ids_with_sequence(
                    runtime,
                    TokenIdsInputPlan::new(ids.tokens()).unwrap(),
                    cfg,
                    eredu_core::TokenFilter::All,
                    None,
                    request,
                )
            } else {
                TextGeneration::from_input_with_sequence(
                    runtime,
                    TextGenerationInput::TokenIds(ids.tokens().to_vec()),
                    cfg,
                    eredu_core::TokenFilter::All,
                    None,
                    request,
                )
            }
            .unwrap();
            ids.after_startup();
            if let Some(graph) = &mut graph {
                graph.observe();
            }
            let mut sequence = generation
                .take_prepared_sequence()
                .unwrap()
                .prepare_storage()
                .unwrap();
            for _ in 0..4 {
                let token = generation.next().unwrap().unwrap();
                let id = token.token_id().unwrap();
                output.push(id);
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
                drop(token);
                assert!(generation.take_captured_delivery().unwrap().is_none());
                if let Some(graph) = &mut graph {
                    graph.observe();
                }
            }
            assert_eq!(sequence.tokens(), output);
        }
        1 => {
            let mut generation = if original {
                ControlledTextGeneration::from_token_ids_with_sequence(
                    runtime,
                    TokenIdsInputPlan::new(ids.tokens()).unwrap(),
                    cfg,
                    disk::Controller::default(),
                    None,
                    request,
                )
            } else {
                ControlledTextGeneration::from_input_with_sequence(
                    runtime,
                    TextGenerationInput::TokenIds(ids.tokens().to_vec()),
                    cfg,
                    disk::Controller::default(),
                    None,
                    request,
                )
            }
            .unwrap();
            ids.after_startup();
            if let Some(graph) = &mut graph {
                graph.observe();
            }
            let mut sequence = generation
                .take_prepared_sequence()
                .unwrap()
                .prepare_storage()
                .unwrap();
            for _ in 0..4 {
                let token = generation.next().unwrap().unwrap();
                let id = token.token_id();
                output.push(id);
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
                drop(token);
                assert!(generation.take_captured_delivery().unwrap().is_none());
                if let Some(graph) = &mut graph {
                    graph.observe();
                }
            }
            assert_eq!(sequence.tokens(), output);
        }
        _ => {
            let mut driver = TextGenerationDriver::new(runtime);
            let mut generation = if original {
                driver.start_token_ids_with_sequence(
                    TokenIdsInputPlan::new(ids.tokens()).unwrap(),
                    cfg,
                    disk::Controller::default(),
                    None,
                    request,
                )
            } else {
                driver.start_input_with_sequence(
                    TextGenerationInput::TokenIds(ids.tokens().to_vec()),
                    cfg,
                    disk::Controller::default(),
                    None,
                    request,
                )
            }
            .unwrap();
            ids.after_startup();
            if let Some(graph) = &mut graph {
                graph.observe();
            }
            let mut sequence = driver
                .take_prepared_sequence(&mut generation)
                .unwrap()
                .unwrap()
                .prepare_storage()
                .unwrap();
            for _ in 0..4 {
                let token = driver.advance(&mut generation).unwrap().unwrap();
                let id = token.token_id();
                output.push(id);
                sequence
                    .commit(id, TokenTerminalSignals::default())
                    .unwrap();
                drop(token);
                assert!(
                    driver
                        .take_completed_delivery(&mut generation)
                        .unwrap()
                        .is_none()
                );
                if let Some(graph) = &mut graph {
                    graph.observe();
                }
            }
            assert_eq!(sequence.tokens(), output);
        }
    }
    output
}
fn logical_kv(runtime: &Runtime, positions: usize) -> Vec<(Vec<i32>, Vec<f32>)> {
    let plan = runtime
        .session()
        .payload
        .model
        .erased()
        .prepare_resident_decoder_copy()
        .unwrap();
    let mut snapshot = Vec::new();
    plan.visit_operands(&mut |array| {
        let mut shape = array.shape().to_vec();
        assert_eq!(shape.len(), 4, "actual Llama fixture K/V geometry");
        let capacity = shape[2] as usize;
        assert!(capacity >= positions);
        let width = shape[3] as usize;
        let evaluated = array.evaluated().unwrap();
        let values = evaluated.as_slice::<f32>();
        let mut logical = Vec::new();
        for head in 0..(shape[0] * shape[1]) as usize {
            let start = head * capacity * width;
            logical.extend_from_slice(&values[start..start + positions * width]);
        }
        assert!(logical.iter().all(|v| v.is_finite()));
        shape[2] = positions as i32;
        snapshot.push((shape, logical));
    });
    assert_eq!(snapshot.len(), 6, "three actual layers, K and V each");
    snapshot
}
#[test]
fn original_token_input_matches_actual_legacy_predictions_all_routes_and_residencies_after_caller_mutation()
 {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    compare_predictions(false);
}
#[test]
fn original_token_input_and_graph_share_real_cached_execution_and_retire_each_original_arena() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    compare_predictions(true);
}
fn compare_predictions(with_graph: bool) {
    let stream = stream();
    for residency in 0..3 {
        for route in 0..3 {
            let mut reference = None;
            for original in [false, true] {
                let pool = crate::tests::support::test_utils::initialize_original_sources();
                let (mut runtime, _artifact) = load(&stream, &pool, residency);
                let mut snapshots = Vec::new();
                let mut arenas = Vec::new();
                for request in 0..2 {
                    let mut graph = (with_graph && original).then(|| GraphRequest::new(&runtime));
                    let mut ids = [2, 5, 7, 3, 11];
                    let values =
                        run_with_graph(&mut runtime, route, original, &mut ids, graph.as_mut());
                    assert_eq!(ids, [1; 5]);
                    assert_eq!(values.len(), 4);
                    runtime.synchronize().unwrap();
                    let positions = 8 * (request + 1);
                    let presence = runtime.session().payload.model.erased().state_snapshot();
                    assert!(
                        presence
                            .iter()
                            .all(|(offset, _)| *offset == positions as i32)
                    );
                    snapshots.push((values, presence, logical_kv(&runtime, positions)));
                    if let Some(graph) = graph {
                        let (arena, held) = graph.finish();
                        assert!(
                            arenas.iter().all(
                                |(prior, _): &(safemlx::SubmissionGraphQuota, u64)| {
                                    !arena.same_arena(prior)
                                }
                            ),
                            "each genuine request admits its own arena"
                        );
                        arenas.push((arena, held));
                    }
                }
                if let Some(expected) = &reference {
                    let expected: &Vec<(Vec<u32>, _, Vec<(Vec<i32>, Vec<f32>)>)> = expected;
                    for (actual, expected) in snapshots.iter().zip(expected) {
                        assert_eq!(actual.0, expected.0);
                        assert_eq!(actual.1, expected.1);
                        assert_eq!(actual.2.len(), expected.2.len());
                        for ((shape, values), (other_shape, other_values)) in
                            actual.2.iter().zip(&expected.2)
                        {
                            assert_eq!(shape, other_shape);
                            assert_eq!(values.len(), other_values.len());
                            for (value, other) in values.iter().zip(other_values) {
                                assert!(
                                    (value - other).abs() <= 1e-5,
                                    "full logical cached state differs"
                                );
                            }
                        }
                    }
                } else {
                    reference = Some(snapshots);
                }
                finish(runtime, &stream);
                if !arenas.is_empty() {
                    assert_eq!(arenas.len(), 2);
                    // Cached arrays and all model/request owners must retire
                    // first. Derive the surviving sum from each actual original
                    // fact, never from GRAPH or a per-request size assumption.
                    let mut held = arenas
                        .iter()
                        .try_fold(0u64, |sum, (_, bytes)| sum.checked_add(*bytes))
                        .unwrap();
                    settle(&pool, held);
                    for (arena, _) in &arenas {
                        assert_eq!(arena.occupied_bytes(), 0);
                    }
                    while let Some((arena, original_held)) = arenas.pop() {
                        drop(arena);
                        held = held.checked_sub(original_held).unwrap();
                        settle(&pool, held);
                    }
                } else {
                    settle(&pool, 0);
                }
            }
        }
    }
}
#[test]
fn original_token_input_exact_and_one_short_precede_native_upload_for_optional_capture_rows() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    for (capture, rows) in [(false, false), (true, false), (true, true)] {
        let mut required = None;
        for pass in 0..3 {
            let pool = crate::tests::support::test_utils::initialize_original_sources();
            let (mut runtime, _artifact) = load(&stream, &pool, 0);
            let source = capture.then(|| source(&runtime));
            let probe = Probe::new(&runtime, source.as_ref(), rows);
            let baseline = pool.fixture_host_charge().unwrap();
            let before = paths::session_input_creation_attempts();
            let observed = paths::snapshot();
            let capacity = required.map_or(u64::MAX, |n| baseline + n - u64::from(pass == 1));
            let result = TextGeneration::from_token_ids_with_sequence(
                &mut runtime,
                TokenIdsInputPlan::new(&[2, 5, 7]).unwrap(),
                config(4, capacity),
                eredu_core::TokenFilter::All,
                source.as_ref().map(|s| TextPreparationOptions {
                    interventions: None,
                    capture: Some(s.clone()),
                }),
                GenerationSequenceRequest::new(4, &[]),
            );
            if pass == 1 {
                let error = result.err().unwrap();
                assert!(matches!(
                    cause::<WorkingMemoryError>(&error),
                    WorkingMemoryError::Domain(
                        eredu_core::MemoryDomainError::BudgetExceeded { .. }
                    )
                ));
                assert!(probe.0.admitted.borrow().is_none());
                assert_eq!(paths::session_input_creation_attempts(), before);
                assert_eq!(paths::snapshot(), observed);
                assert_eq!(pool.fixture_host_charge().unwrap(), baseline);
            } else {
                let mut generation = result.unwrap();
                let sequence = generation.take_prepared_sequence().unwrap();
                let facts = probe.facts();
                if pass == 0 {
                    required = Some(facts.required);
                } else {
                    assert_eq!(required, Some(facts.required));
                }
                let prepared = probe.take();
                assert!(
                    prepared
                        .quote
                        .as_ref()
                        .unwrap()
                        .original_token_input()
                        .is_some()
                );
                assert!(
                    prepared
                        .quote
                        .as_ref()
                        .unwrap()
                        .original_token_input()
                        .unwrap()
                        .take()
                        .is_err()
                );
                drop((sequence, generation, prepared));
            }
            drop((probe, source));
            finish(runtime, &stream);
            settle(&pool, 0);
        }
    }
}
#[test]
fn original_input_busy_retry_preserves_actual_pointer_work_and_node_without_recopy() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::DeferOriginalPrompt);
    let mut ids = [2, 5, 7];
    let error = TextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        TokenIdsInputPlan::new(&ids).unwrap(),
        config(4, u64::MAX),
        eredu_core::TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .err()
    .unwrap();
    assert_eq!(cause::<Rejection>(&error), &Rejection::Unavailable);
    drop(error);
    ids.fill(99); // completed owned construction, before actual Prompt dispatch
    let preparation = probe.take();
    let quote = preparation.quote.as_ref().unwrap();
    let scopes = quote.preparation_scopes.as_ref().unwrap();
    let before = paths::session_input_creation_attempts();
    with_foreign_runtime(|| {
        let mut first = None;
        for _ in 0..4 {
            let error =
                MlxBackend::prepare_original_text_prompt_admitted(runtime.backend(), &preparation)
                    .unwrap_err();
            assert!(matches!(
                cause::<Error>(&error),
                Error::PreparationScope(SubmissionScopeOwnerCause::RuntimeBusy)
            ));
            let identities = scopes.pending_prompt_identities().unwrap();
            if let Some(first) = first {
                assert_eq!(identities, first);
            } else {
                first = Some(identities);
            }
            assert_eq!(paths::session_input_creation_attempts(), before);
        }
    });
    let prompt =
        MlxBackend::prepare_original_text_prompt_admitted(runtime.backend(), &preparation).unwrap();
    assert_eq!(
        prompt.parts[0]
            .payload()
            .value()
            .evaluated()
            .unwrap()
            .as_slice::<u32>(),
        &[2, 5, 7]
    );
    let replay: Vec<_> = (0..16)
        .map(|_| {
            MlxBackend::prepare_original_text_prompt_admitted(runtime.backend(), &preparation)
                .unwrap_err()
        })
        .collect();
    assert!(
        replay
            .iter()
            .all(|e| cause::<TokenInputRejection>(e) == &TokenInputRejection::Unavailable)
    );
    drop((prompt, preparation, probe));
    finish(runtime, &stream);
    settle(&pool, 0);
    drop(replay);
}

#[test]
fn foreign_equal_value_claim_is_rejected_before_i_or_r_take_and_current_bank_remains_usable() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    let old_ids = [2, 5, 7, 3, 11];
    let mut current_ids = old_ids;
    assert_ne!(old_ids.as_ptr(), current_ids.as_ptr());
    probe.mode(Mode::Defer);
    let error = TextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        TokenIdsInputPlan::new(&old_ids).unwrap(),
        config(4, u64::MAX),
        eredu_core::TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .err()
    .unwrap();
    assert_eq!(cause::<Rejection>(&error), &Rejection::Unavailable);
    drop(error);
    let previous = probe.take();
    probe.target(previous.clone());
    probe.mode(Mode::CheckForeignOriginalInput);
    // The scoped hook rejects the genuine current claim against the old bank,
    // then rejects its old preparation against the current bank. Normal startup
    // still consumes the current bank and predicts from the original IDs.
    let output = run(&mut runtime, 0, true, &mut current_ids);
    assert_eq!(output.len(), 4);
    let current = probe.take();
    assert!(
        previous
            .quote
            .as_ref()
            .unwrap()
            .original_token_input()
            .unwrap()
            .is_unclaimed_for_test()
    );
    assert!(
        !current
            .quote
            .as_ref()
            .unwrap()
            .original_token_input()
            .unwrap()
            .is_unclaimed_for_test()
    );
    assert!(
        previous
            .quote
            .as_ref()
            .unwrap()
            .sequence
            .as_ref()
            .unwrap()
            .pending
            .borrow()
            .is_some()
    );
    assert!(
        current
            .quote
            .as_ref()
            .unwrap()
            .sequence
            .as_ref()
            .unwrap()
            .pending
            .borrow()
            .is_none()
    );
    drop((previous, current, probe));
    finish(runtime, &stream);
    settle(&pool, 0);
}

#[test]
fn original_encoding_lends_real_native_input_with_same_predictions_and_logical_kv() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    use eredu_runtime::working_memory::OriginalTokenizerBackend;
    use eredu_text::tokenizer_storage::TokenizerPlan;
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"a":2,"b":5,"c":7,"d":3,"e":11},"merges":[]}}"#;
    let stream = stream();
    for residency in 0..3 {
        for route in 0..3 {
            let mut reference: Option<Vec<(Vec<u32>, Vec<(Vec<i32>, Vec<f32>)>)>> = None;
            for encoded_route in [false, true] {
                let pool = crate::tests::support::test_utils::initialize_original_sources();
                let (mut runtime, _artifact) = load(&stream, &pool, residency);
                let source = encoded_route.then(|| {
                    MlxBackend::compile_original_tokenizer(
                        &runtime,
                        TokenizerPlan::prepare_json(json.as_bytes()).unwrap(),
                    )
                    .unwrap()
                });
                let mut actual = Vec::new();
                for request in 0..2 {
                    let ids = source.as_ref().map(|source| {
                        MlxBackend::encode_original_tokenizer_ids(&runtime, source, "abcde", true)
                            .unwrap()
                    });
                    let values = if let Some(ids) = &ids {
                        assert_eq!(ids.ids(), [2, 5, 7, 3, 11]);
                        run_inputs(&mut runtime, route, true, FixtureInput::Encoded(ids), None)
                    } else {
                        run(&mut runtime, route, false, &mut [2, 5, 7, 3, 11])
                    };
                    assert_eq!(values.len(), 4);
                    runtime.synchronize().unwrap();
                    actual.push((values, logical_kv(&runtime, 8 * (request + 1))));
                    if let Some(ids) = ids {
                        assert!(ids.matches_source(source.as_ref().unwrap()));
                        assert_eq!(ids.ids(), [2, 5, 7, 3, 11]);
                        let before = pool.fixture_host_charge().unwrap();
                        let e = ids.original_bytes();
                        drop(ids);
                        assert_eq!(pool.fixture_host_charge().unwrap(), before - e);
                    }
                }
                if let Some(reference) = &reference {
                    for ((ids, kv), (expected, other)) in actual.iter().zip(reference) {
                        assert_eq!(ids, expected);
                        assert_eq!(kv.len(), other.len());
                        for ((shape, values), (other_shape, other_values)) in kv.iter().zip(other) {
                            assert_eq!(shape, other_shape);
                            assert_eq!(values.len(), other_values.len());
                            assert!(
                                values
                                    .iter()
                                    .zip(other_values)
                                    .all(|(a, b)| (a - b).abs() <= 1e-5)
                            );
                        }
                    }
                } else {
                    reference = Some(actual);
                }
                let c = source.as_ref().map_or(0, |s| s.original_bytes());
                finish(runtime, &stream);
                settle(&pool, c);
                drop(source);
                settle(&pool, 0);
            }
        }
    }
}

#[test]
fn prepared_media_rebind_leaves_genuine_original_token_prompt_authority_untouched() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    use eredu_runtime::execution_control::TextSnapshotBackend;
    let stream = stream();
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut runtime, _artifact) = load(&stream, &pool, 0);
    let probe = Probe::new(&runtime, None, false);
    probe.mode(Mode::DeferOriginalPrompt);
    let ids = [2, 5, 7];
    let error = TextGeneration::from_token_ids_with_sequence(
        &mut runtime,
        TokenIdsInputPlan::new(&ids).unwrap(),
        config(4, u64::MAX),
        eredu_core::TokenFilter::All,
        None,
        GenerationSequenceRequest::new(4, &[]),
    )
    .err()
    .unwrap();
    assert_eq!(cause::<Rejection>(&error), &Rejection::Unavailable);
    drop(error);
    let preparation = probe.take();
    let prompt =
        MlxBackend::prepare_original_text_prompt_admitted(runtime.backend(), &preparation).unwrap();
    assert!(prompt.has_original_input_custody());
    let quote = prompt.quote.as_ref().unwrap().clone();
    let before = quote
        .original_token_input()
        .unwrap()
        .is_unclaimed_for_test();
    let mut pending = Some(eredu_core::PendingTextInput::Prefill(prompt));
    MlxBackend::rebind_pending_capture(&runtime, None, None, &mut pending).unwrap();
    let Some(eredu_core::PendingTextInput::Prefill(prompt)) = pending else {
        panic!("same pending input")
    };
    assert!(prompt.quote.as_ref().unwrap().same_owner(&quote));
    assert_eq!(
        quote
            .original_token_input()
            .unwrap()
            .is_unclaimed_for_test(),
        before
    );
    assert_eq!(
        prompt.parts[0]
            .payload()
            .value()
            .evaluated()
            .unwrap()
            .as_slice::<u32>(),
        &ids
    );
    drop((prompt, quote, preparation, probe));
    finish(runtime, &stream);
    settle(&pool, 0);
}

pub(in crate::composition::mlx::session::model_session::text_quote) mod original_plain;

#[test]
fn host_only_sequence_keeps_shared_predictions_and_host_custody_without_native_arenas() {
    if !crate::tests::support::native_process::enter("native-sequence-source") {
        return;
    }
    let stream = stream();
    let input = [2, 5, 7, 3, 11];
    let pool = crate::tests::support::test_utils::initialize_original_sources();
    let (mut ordinary, _artifact) = load(&stream, &pool, 0);
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
    ordinary.synchronize().unwrap();
    let expected_kv = logical_kv(&ordinary, 8);
    assert!(expected_kv.iter().flat_map(|(_, v)| v).any(|v| *v != 0.0));
    finish(ordinary, &stream);
    settle(&pool, 0);

    // Existing uninterrupted and controlled sequence drivers both request H
    // without native Graph/Record arenas. Neither may silently gain those arenas.
    for route in [0, 1] {
        let pool = crate::tests::support::test_utils::initialize_original_sources();
        let (mut runtime, _artifact) = load(&stream, &pool, 0);
        let probe = Probe::new(&runtime, None, false);
        let calls = crate::backend::submission_recovery::prediction::test_counts::Calls::new();
        let mut ids = input;
        let actual = run(&mut runtime, route, false, &mut ids);
        assert_eq!(actual, expected);
        runtime.synchronize().unwrap();
        let actual_kv = logical_kv(&runtime, 8);
        assert_eq!(actual_kv.len(), expected_kv.len());
        for ((shape, values), (expected_shape, expected_values)) in
            actual_kv.iter().zip(&expected_kv)
        {
            assert_eq!(shape, expected_shape);
            assert_eq!(values.len(), expected_values.len());
            assert!(
                values
                    .iter()
                    .zip(expected_values)
                    .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5)
            );
        }
        let preparation = probe.take();
        let quote = preparation.quote.as_ref().unwrap();
        assert!(quote.sequence.is_some());
        assert!(quote.record_quota.is_none());
        assert!(quote.graph_quota.is_none());
        assert!(quote.native_storage.is_none());
        assert!(
            calls.original().iter().all(|n| *n > 0),
            "all five genuine host roles execute"
        );
        assert_eq!(calls.legacy(), [0; 5]);
        drop((preparation, probe, calls));
        finish(runtime, &stream);
        settle(&pool, 0);
    }
}
