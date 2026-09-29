//! Nonzero target and MTP through the shared speculative driver.
use super::executor::Materializer;
use super::installed::*;
use super::*;
use eredu_architectures::prediction_extension::*;
use eredu_architectures::speculative_execution::{
    EmbeddedExecutorTypes, EmbeddedPredictionExecutor, EmbeddedPredictionOutput,
    ReplicatedMaterializedPredictionStrategy, ReplicatedPredictionInput,
    ReplicatedPredictionNative, SpeculativeTensorMechanisms,
};
use eredu_core::SpeculativeExecutor;
use eredu_runtime::*;

#[derive(Clone)]
struct Input {
    tokens: NumericTensor,
    identity: PreparedInputCacheIdentity,
    chunk_tokens: usize,
    fail_after_chunks: Option<usize>,
}
impl Input {
    fn new(tokens: &[u32]) -> Self {
        let tokens_tensor = tokens_tensor(tokens);
        let part = PreparedInputPart::new(
            eredu_core::InputModality::Text,
            PreparedInputPayload::TokenIds(tokens_tensor.clone()),
            [],
        )
        .unwrap();
        let prepared = PreparedModelInput::new(vec![part], |t| {
            eredu_core::InputTensorIdentity::new(
                eredu_core::checkpoint::TensorDtype::U32,
                t.shape.iter().map(|d| *d as usize).collect(),
            )
        })
        .unwrap();
        Self {
            tokens: tokens_tensor,
            chunk_tokens: usize::MAX,
            fail_after_chunks: None,
            identity: prepared
                .cache_identity(format!("qwen4-fixture-{tokens:?}"))
                .unwrap(),
        }
    }

    fn with_chunk_tokens(mut self, tokens: usize) -> Self {
        self.chunk_tokens = tokens;
        self
    }
}
struct Inputs;
impl ReplicatedPredictionInput<Target, NumericBackend, State, Error> for Inputs {
    type Input = Input;
    fn with_prefill_chunks(
        &mut self,
        input: Input,
        maximum_chunk_tokens: usize,
        _: &NumericContext,
        mut operation: impl for<'a> FnMut(
            TargetInput<'a, NumericTensor>,
            NumericTensor,
            Option<&'a PreparedInputCacheIdentity>,
        ) -> Result<
            <Target as LayeredArchitecture<NumericBackend, State>>::ForwardContext,
            Error,
        >,
    ) -> Result<(), Error> {
        let maximum_chunk_tokens = maximum_chunk_tokens.min(input.chunk_tokens);
        for (chunk, start) in (0..input.tokens.shape[1] as usize)
            .step_by(maximum_chunk_tokens)
            .enumerate()
        {
            let end = (start + maximum_chunk_tokens).min(input.tokens.shape[1] as usize);
            let tokens = input.tokens.axis_slice(1, start, end);
            operation(
                <Target as ReplicatedTextArchitecture<NumericBackend, State>>::text_input(
                    &tokens, None,
                ),
                tokens.clone(),
                Some(&input.identity),
            )?;
            if input.fail_after_chunks == Some(chunk + 1) {
                return Err(Error::backend("injected later prefill chunk failure"));
            }
        }
        Ok(())
    }
    fn with_decode<R>(
        &mut self,
        tokens: &NumericTensor,
        _: &NumericContext,
        operation: impl for<'a> FnOnce(TargetInput<'a, NumericTensor>) -> Result<R, Error>,
    ) -> Result<R, Error> {
        operation(<Target as ReplicatedTextArchitecture<
            NumericBackend,
            State,
        >>::text_input(tokens, None))
    }
}
struct Mechanisms;
struct Types;
struct PhaseTrace(
    std::rc::Rc<RefCell<Vec<(eredu_core::speculative::SpeculativeActivationPhase, usize)>>>,
);
impl ActivationObserver<NumericTensor, Error> for PhaseTrace {
    fn observe(&mut self, _: &str, _: &NumericTensor) -> Result<(), Error> {
        Ok(())
    }
}
impl eredu_runtime::inspection::SpeculativeActivationObserver<NumericTensor, Error> for PhaseTrace {
    fn begin_activation_invocation(
        &mut self,
        phase: eredu_core::speculative::SpeculativeActivationPhase,
        sequence: usize,
    ) -> Result<(), Error> {
        self.0.borrow_mut().push((phase, sequence));
        Ok(())
    }
    fn complete_activation_invocation(&mut self) -> Result<(), Error> {
        Ok(())
    }
    fn finish_activation_invocation(&mut self, success: bool) {
        assert!(success);
    }
}
impl EmbeddedExecutorTypes for Types {
    type Input = Input;
    type Logits = NumericTensor;
    type Context<'a> = &'a NumericContext;
    type Completion = NumericCompletion;
    type Telemetry = ();
    type Error = Error;
    fn erased_type_mismatch(value: &'static str) -> Error {
        Error::backend(value)
    }
}
impl ReplicatedPredictionNative<Target, NumericBackend, State, Mechanisms> for Materializer {
    type Input = Input;
    type Telemetry = ();
    type ExecutorTypes = Types;
    fn executor_context<'a>(
        ctx: <Types as EmbeddedExecutorTypes>::Context<'a>,
    ) -> <Mechanisms as SpeculativeTensorMechanisms>::Context<'a> {
        ctx
    }
    fn target_context<'a>(
        ctx: <Mechanisms as SpeculativeTensorMechanisms>::Context<'a>,
    ) -> &'a NumericContext {
        ctx
    }
    fn control_state_estimate(
        state: &State,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::snapshots::state(state)
    }
    fn control_state_snapshot(
        state: &State,
        _: &NumericContext,
    ) -> Result<Option<State>, eredu_core::speculative::SpeculativeControlError> {
        super::snapshots::copied();
        Ok(Some(state.clone()))
    }
    fn checkpoint(state: &State) -> Result<State, Error> {
        Ok(state.clone())
    }
    fn restore(state: &mut State, checkpoint: &State, _: &NumericContext) -> Result<(), Error> {
        state.clone_from(checkpoint);
        Ok(())
    }
    fn generation(state: &State) -> Result<u64, Error> {
        Ok(state.clone().layer(0).unwrap().position() as u64)
    }
    fn token(token: u32, _: &NumericContext) -> Result<NumericTensor, Error> {
        Ok(tokens_tensor(&[token]))
    }
    fn shape(tensor: &NumericTensor) -> &[i32] {
        tensor.shape()
    }
    fn validate<T>(operation: impl FnOnce() -> Result<T, Error>) -> Result<T, Error> {
        operation()
    }
    fn session_error(error: impl std::fmt::Display) -> Error {
        Error::backend(error.to_string())
    }
    fn session_failure(error: eredu_core::BackendFailure) -> Error {
        Error::backend_source(error)
    }
    fn take_telemetry() -> Result<(), Error> {
        Ok(())
    }
}
impl SpeculativeTensorMechanisms for Mechanisms {
    fn activation_observer<'a>(
        plan: &eredu_core::speculative::AdmittedSpeculativeActivations,
        request: eredu_core::SpeculativeRequestId,
        _: Self::Context<'a>,
    ) -> Result<
        Option<
            Box<dyn eredu_runtime::inspection::SpeculativeActivationObserver<NumericTensor, Error>>,
        >,
        eredu_core::speculative::SpeculativeControlError,
    > {
        control::captures::observer(plan, request)
    }

    type Tensor = NumericTensor;
    type Logits = NumericTensor;
    type Context<'a> = &'a NumericContext;
    type Completion = NumericCompletion;
    type Error = Error;
    fn control_tensor_estimate(
        value: &NumericTensor,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        super::snapshots::tensor(value)
    }
    fn control_tensor_snapshot<'a>(
        value: &NumericTensor,
        _: Self::Context<'a>,
    ) -> Result<Option<NumericTensor>, eredu_core::speculative::SpeculativeControlError> {
        super::snapshots::copied();
        Ok(Some(value.clone()))
    }
    fn observation_error(message: &'static str) -> Error {
        Error::backend(message)
    }
    fn empty_prediction_input() -> Error {
        Error::backend("empty prediction input")
    }
    fn fused_prediction_exhausted() -> Error {
        Error::backend("fused prediction exhausted")
    }
    fn invalid_prediction_commit(verified: usize, available: usize) -> Error {
        Error::backend(format!("invalid commit {verified}/{available}"))
    }
    fn invalid_prediction_output(
        logits: usize,
        capture: usize,
        tokens: usize,
        expected: Option<usize>,
    ) -> Error {
        Error::backend(format!(
            "invalid output {logits}/{capture}/{tokens}/{expected:?}"
        ))
    }
    fn invalid_fused_capacity(requested: usize, available: usize) -> Error {
        Error::backend(format!("invalid capacity {requested}/{available}"))
    }
    fn sequence_len(value: &NumericTensor) -> Result<usize, Error> {
        Ok(value.shape[1] as usize)
    }
    fn logits_row<'a>(
        value: &NumericTensor,
        row: usize,
        ctx: &'a NumericContext,
    ) -> Result<NumericTensor, Error> {
        Self::tensor_row(value, row, ctx)
    }
    fn tensor_row<'a>(
        value: &NumericTensor,
        row: usize,
        _: &'a NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(value.axis_slice(1, row, row + 1))
    }
    fn tensor_prefix<'a>(
        value: &NumericTensor,
        end: usize,
        _: &'a NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(value.axis_slice(1, 0, end))
    }
    fn token_range<'a>(
        value: &NumericTensor,
        start: usize,
        end: usize,
        _: &'a NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(value.axis_slice(1, start, end))
    }
    fn token_prefix<'a>(
        value: &NumericTensor,
        end: usize,
        ctx: &'a NumericContext,
    ) -> Result<NumericTensor, Error> {
        Self::tensor_prefix(value, end, ctx)
    }
    fn target_tokens<'a>(tokens: &[u32], _: &'a NumericContext) -> Result<NumericTensor, Error> {
        Ok(tokens_tensor(tokens))
    }
    fn fused_logits_row<'a>(
        value: &NumericTensor,
        row: usize,
        ctx: &'a NumericContext,
    ) -> Result<NumericTensor, Error> {
        Self::logits_row(value, row, ctx)
    }
    fn submit_verification_completion<'a>(
        _: &EmbeddedPredictionOutput<NumericTensor>,
        _: &NumericTensor,
        _: &'a NumericContext,
    ) -> Result<NumericCompletion, Error> {
        Ok(NumericCompletion::immediate())
    }
}

fn contract_request(
    target: &str,
    batch: usize,
    sequence: usize,
    draft: usize,
    topology: (usize, usize, usize, usize),
) -> EmbeddedSpeculativeContractRequest {
    use std::num::NonZeroUsize;
    let identity = |v: &str| SpeculativeIdentity::new(v).unwrap();
    EmbeddedSpeculativeContractRequest::new(
        identity(target),
        identity("fixture-source"),
        identity("float32"),
        eredu_core::ParallelRankTopology::new(
            eredu_core::ParallelTopology::new(topology.0, topology.1, topology.2, topology.3)
                .unwrap(),
            0,
        )
        .unwrap(),
        identity("text-token-ids"),
        NonZeroUsize::new(batch).unwrap(),
        NonZeroUsize::new(sequence).unwrap(),
        NonZeroUsize::new(draft).unwrap(),
    )
}

fn selection(
    selected: &eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution,
) -> SelectedSpeculativeRealization {
    let target = selected.target_state_fingerprint().unwrap();
    for request in [
        contract_request("wrong-target", 1, 8, 2, (1, 1, 1, 1)),
        contract_request(&target, 3, 8, 2, (1, 1, 1, 1)),
        contract_request(&target, 1, 1024, 2, (1, 1, 1, 1)),
        contract_request(&target, 1, 8, 3, (1, 1, 1, 1)),
        contract_request(&target, 1, 2, 2, (1, 1, 1, 1)),
        contract_request(&target, 1, 8, 2, (1, 1, 1, 2)),
    ] {
        assert!(selected.speculative_contract(request).is_err());
    }
    let distributed = selected
        .speculative_contract(contract_request(&target, 1, 8, 2, (2, 2, 1, 1)))
        .unwrap();
    assert!(distributed
        .requirements()
        .mechanisms()
        .mechanisms()
        .contains(&SpeculativeMechanism::Communication));
    assert_eq!(
        distributed.target_capture().entries()[0].shape(),
        [1, 8, 2, 2]
    );
    let contract = selected
        .speculative_contract(contract_request(&target, 1, 8, 2, (1, 1, 1, 1)))
        .unwrap();
    assert_eq!(contract.target_capture().entries()[0].shape(), [1, 8, 2, 2]);
    assert!(select_speculative_realization(
        contract.requirements(),
        &contract.selection_request(SpeculativePlacementRequest::Single),
        &SpeculativeMechanismCapabilities::new([])
    )
    .is_err());
    let facts = SpeculativeMechanismCapabilities::new(
        contract
            .requirements()
            .mechanisms()
            .mechanisms()
            .iter()
            .copied(),
    );
    select_speculative_realization(
        contract.requirements(),
        &contract.selection_request(SpeculativePlacementRequest::Single),
        &facts,
    )
    .unwrap()
}

pub(super) fn check(
    selected: &eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
) {
    let reads = prepared
        .artifact()
        .source_diagnostics()
        .unwrap()
        .physical_reads;
    selection(selected);
    let changed_hash = eredu_architectures::qwen4_exp::ngram::NGramHashSpec::new(
        32,
        7,
        3,
        1,
        vec![23703573157769, 9007199254740993, 8052911324071],
        vec![11, 13],
        vec![0, 11],
        24,
    )
    .unwrap();
    let alternate =
        BoundTargetSpec::new(prepared.spec().clone(), BTreeMap::from([(1, changed_hash)])).unwrap();
    assert_eq!(
        alternate.geometry().geometry_fingerprint(),
        prepared.spec().geometry_fingerprint()
    );
    assert_ne!(
        alternate.state_fingerprint(),
        selected.target_state_fingerprint().unwrap()
    );
    assert!(selected
        .speculative_contract(contract_request(
            alternate.state_fingerprint(),
            1,
            8,
            2,
            (1, 1, 1, 1)
        ))
        .is_err());
    assert_eq!(
        prepared
            .artifact()
            .source_diagnostics()
            .unwrap()
            .physical_reads,
        reads
    );
    scheduler_check(selected, prepared, ctx);
    control::check(selected, prepared, ctx);
    let speculative = selection(selected);
    for accepted in [0, 1, 2] {
        let (mut session, mut extension) = paired(selected, prepared, ctx);
        let (mut baseline, mut canonical_extension) = paired(selected, prepared, ctx);
        let mut canonical = <Executor as MaterializedPredictionExecutor<
            Target,
            NumericBackend,
            Materializer,
        >>::new_state(&canonical_extension);
        let mut strategy = ReplicatedMaterializedPredictionStrategy::<
            Target,
            NumericBackend,
            State,
            NumericReplicatedMechanisms,
            _,
            _,
            Inputs,
            Materializer,
            Mechanisms,
        >::new(&mut session, &mut extension, &speculative, Inputs, ctx);
        let mut cache = strategy.new_cache().unwrap();
        let phases = std::rc::Rc::new(RefCell::new(Vec::new()));
        let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::with_observers(
            &mut strategy,
            eredu_architectures::speculative_execution::EmbeddedPredictionObservers::default()
                .with_internal(PhaseTrace(phases.clone())),
        );
        let prompt = [3, 4, 7];
        if accepted == 0 {
            let target_before = cache.target().unwrap().clone();
            let prediction_before = cache.prediction().0.clone();
            let mut failed = Input::new(&prompt).with_chunk_tokens(1);
            failed.fail_after_chunks = Some(2);
            assert!(executor
                .prefill(failed, &mut cache, ctx)
                .err()
                .unwrap()
                .to_string()
                .contains("injected later prefill chunk failure"));
            assert_state_exact(
                cache.target().unwrap(),
                &target_before,
                prepared.spec().units.len(),
                "later chunk failure restores target state",
            );
            assert_state_exact(
                &cache.prediction().0,
                &prediction_before,
                2,
                "later chunk failure restores shifted prediction state",
            );
            phases.borrow_mut().clear();
        }
        let (logits, state, _) = executor
            .prefill(
                Input::new(&prompt).with_chunk_tokens(accepted as usize + 1),
                &mut cache,
                ctx,
            )
            .unwrap()
            .into_parts();
        use eredu_core::speculative::SpeculativeActivationPhase::{
            PredictionPrefill, TargetPrefill,
        };
        let expected_phases = match accepted {
            0 => vec![
                (TargetPrefill, 1),
                (TargetPrefill, 1),
                (PredictionPrefill, 1),
                (TargetPrefill, 1),
                (PredictionPrefill, 1),
            ],
            1 => vec![
                (TargetPrefill, 2),
                (PredictionPrefill, 1),
                (TargetPrefill, 1),
                (PredictionPrefill, 1),
            ],
            _ => vec![(TargetPrefill, 3), (PredictionPrefill, 2)],
        };
        assert_eq!(
            *phases.borrow(),
            expected_phases,
            "observations retain physical chunk and bridge widths"
        );
        let (expected, capture) = baseline
            .prefill_prediction_target(&tokens_tensor(&prompt), None, ctx)
            .unwrap();
        advance(
            &mut canonical_extension,
            &mut baseline,
            &capture.axis_slice(1, 0, 2),
            &tokens_tensor(&[4, 7]),
            &mut canonical,
            ctx,
        );
        assert_state_exact(
            cache.target().unwrap(),
            &baseline.checkpoint(ctx).unwrap(),
            prepared.spec().units.len(),
            "chunked prefill target matches whole request",
        );
        assert_state_exact(
            &cache.prediction().0,
            &canonical.0,
            2,
            "chunked prefill preserves every shifted prediction pair",
        );
        advance(
            &mut canonical_extension,
            &mut baseline,
            &capture.axis_slice(1, 2, 3),
            &tokens_tensor(&[5]),
            &mut canonical,
            ctx,
        );
        assert_tensor_exact(
            &logits,
            &expected.axis_slice(1, 2, 3),
            "driver prefill logits",
        );
        let mut draft = executor.begin_proposal(&state, 5, 2, ctx).unwrap();
        executor.proposal_logits(&mut draft, 5, ctx).unwrap();
        executor.proposal_logits(&mut draft, 9, ctx).unwrap();
        let checkpoint = executor.checkpoint(&cache).unwrap();
        let submission = executor
            .submit_verification(&[5, 9, 11], &mut cache, ctx)
            .unwrap();
        submission.completion.wait().unwrap();
        let committed = executor
            .commit_verification(
                submission.output,
                draft,
                &mut cache,
                &checkpoint,
                accepted + 1,
                ctx,
            )
            .unwrap();
        let (_, replayed) = committed.into_parts();
        assert_eq!(replayed, if accepted == 2 { 0 } else { accepted + 1 });
        let (expected, capture) = baseline
            .prefill_prediction_target(
                &NumericTensor::token_ids(&[5, 9, 11][..accepted + 1]),
                None,
                ctx,
            )
            .unwrap();
        if accepted > 0 {
            advance(
                &mut canonical_extension,
                &mut baseline,
                &capture.axis_slice(1, 0, accepted),
                &tokens_tensor(&[9, 11][..accepted]),
                &mut canonical,
                ctx,
            );
        }
        assert_state_exact(
            &cache.prediction().0,
            &canonical.0,
            2,
            "accepted prediction context excludes tentative captures",
        );
        assert!(expected.data.iter().all(|n| n.is_finite()));
        assert_state_exact(
            cache.target().unwrap(),
            &baseline.checkpoint(ctx).unwrap(),
            prepared.spec().units.len(),
            "verification committed target",
        );
    }
}

fn tokens_tensor(tokens: &[u32]) -> NumericTensor {
    NumericTensor::token_ids(&tokens.iter().map(|n| *n as usize).collect::<Vec<_>>())
}

fn scheduler_check(
    selected: &eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
) {
    use super::sampling::{Constraint, Publisher, Sampling};
    use eredu_core::generation::{
        GenerationCancellationToken, GenerationSequence, SpeculativeConfig,
        SpeculativeSchedulerOptions,
    };
    use eredu_core::{
        PreparedSpeculativeLane, SpeculativeExecutionTopology, SpeculativeOutputRuntime,
        SpeculativeRandomness,
    };
    let speculative = selection(selected);
    let prompt = [3, 4, 7];
    let (expected, frontiers) = oracle(selected, prepared, ctx, None);
    for drafts in [
        None,
        Some((expected.clone(), false)),
        Some((expected.clone(), true)),
    ] {
        let mut previous = None;
        for stepped in [false, true] {
            let (mut session, mut extension) = paired(selected, prepared, ctx);
            let mut strategy =
                ReplicatedMaterializedPredictionStrategy::<
                    Target,
                    NumericBackend,
                    State,
                    NumericReplicatedMechanisms,
                    _,
                    _,
                    Inputs,
                    Materializer,
                    Mechanisms,
                >::new(&mut session, &mut extension, &speculative, Inputs, ctx);
            let mut cache = strategy.new_cache().unwrap();
            let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
            let published = std::rc::Rc::new(RefCell::new(vec![]));
            let runtime = SpeculativeOutputRuntime::new(
                Sampling {
                    drafts: drafts.clone(),
                    ..Default::default()
                },
                GenerationSequence::new(12, []),
                Constraint,
                Publisher(published.clone()),
                GenerationCancellationToken::new(),
            );
            let lane = PreparedSpeculativeLane::new(
                &mut cache,
                Input::new(&prompt),
                SpeculativeConfig {
                    max_tokens: 12,
                    max_draft_tokens: 2,
                    temperature: 0.,
                    eos_token_ids: vec![],
                },
                runtime,
                SpeculativeRandomness::new(None, None),
            );
            let mut scheduler = SpeculativeScheduler::new(
                &mut executor,
                SpeculativeSchedulerOptions::default().with_lookahead(false),
                SpeculativeExecutionTopology::Single,
                false,
                false,
                ctx,
            )
            .unwrap();
            scheduler.submit(lane).unwrap();
            if stepped {
                while scheduler.step().unwrap() {}
            } else {
                scheduler.run().unwrap();
            }
            let result = scheduler.finish().unwrap().take_requests().pop().unwrap();
            assert_eq!(result.token_ids(), expected);
            assert_eq!(*published.borrow(), expected);
            let frontier = cache.target().unwrap().clone().layer(0).unwrap().position() as usize;
            let (target, prediction) = &frontiers[frontier - prompt.len()];
            assert_state_exact(
                cache.target().unwrap(),
                target,
                prepared.spec().units.len(),
                "scheduler canonical target",
            );
            assert_state_exact(
                &cache.prediction().0,
                prediction,
                2,
                "scheduler canonical prediction",
            );
            let accepts = result.stats().accept_lens().to_vec();
            if let Some((_, reject)) = &drafts {
                if *reject {
                    assert!(accepts.iter().all(|n| *n == 0));
                } else {
                    assert!(accepts.iter().any(|n| *n == 2));
                }
            }
            if let Some((target, prediction, previous_accepts)) = &previous {
                assert_eq!(&accepts, previous_accepts);
                assert_state_exact(
                    cache.target().unwrap(),
                    target,
                    prepared.spec().units.len(),
                    "stepped target state",
                );
                assert_state_exact(
                    &cache.prediction().0,
                    prediction,
                    2,
                    "stepped prediction state",
                );
            }
            previous = Some((
                cache.target().unwrap().clone(),
                cache.prediction().0.clone(),
                accepts,
            ));
        }
    }
}

#[path = "control.rs"]
mod control;

fn oracle(
    selected: &eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
    forced: Option<(usize, u32)>,
) -> (Vec<u32>, Vec<(State, State)>) {
    use super::sampling::argmax;
    let prompt = [3, 4, 7];
    let (mut baseline, mut canonical_extension) = paired(selected, prepared, ctx);
    let mut canonical = <Executor as MaterializedPredictionExecutor<
        Target,
        NumericBackend,
        Materializer,
    >>::new_state(&canonical_extension);
    let (logits, capture) = baseline
        .prefill_prediction_target(&tokens_tensor(&prompt), None, ctx)
        .unwrap();
    advance(
        &mut canonical_extension,
        &mut baseline,
        &capture.axis_slice(1, 0, 2),
        &tokens_tensor(&[4, 7]),
        &mut canonical,
        ctx,
    );
    let mut capture = capture.axis_slice(1, 2, 3);
    let mut logits = logits.axis_slice(1, 2, 3);
    let mut frontiers = vec![(baseline.checkpoint(ctx).unwrap(), canonical.0.clone())];
    let mut expected = vec![];
    for _ in 0..12 {
        let token = forced
            .filter(|(position, _)| *position == expected.len())
            .map_or_else(|| argmax(&logits), |(_, token)| token);
        expected.push(token);
        advance(
            &mut canonical_extension,
            &mut baseline,
            &capture,
            &tokens_tensor(&[token]),
            &mut canonical,
            ctx,
        );
        (logits, capture) = baseline
            .prefill_prediction_target(&tokens_tensor(&[token]), None, ctx)
            .unwrap();
        frontiers.push((baseline.checkpoint(ctx).unwrap(), canonical.0.clone()));
    }
    (expected, frontiers)
}

pub(super) fn check_joint_prefill_cap(
    selected: &eredu_architectures::qwen4_exp::prepared::SelectedTargetExecution,
    prepared: &PreparedTarget,
    ctx: &NumericContext,
) {
    let cap = prepared.spec().limits.qsa.tokens as usize;
    assert!(selected.prediction_spec().unwrap().limits.qsa.tokens as usize > cap);
    for capture_cap in [cap, 17] {
        let contract = selected
            .speculative_contract(contract_request(
                &selected.target_state_fingerprint().unwrap(),
                1,
                capture_cap,
                2,
                (1, 1, 1, 1),
            ))
            .unwrap();
        let facts = SpeculativeMechanismCapabilities::new(
            contract
                .requirements()
                .mechanisms()
                .mechanisms()
                .iter()
                .copied(),
        );
        let speculative = select_speculative_realization(
            contract.requirements(),
            &contract.selection_request(SpeculativePlacementRequest::Single),
            &facts,
        )
        .unwrap();
        let prompt: Vec<_> = [3, 4, 7, 9, 0].into_iter().cycle().take(cap + 1).collect();
        let mut reference: Option<(NumericTensor, State, State)> = None;
        for chunk in [usize::MAX, 1] {
            let (mut session, mut extension) = paired(selected, prepared, ctx);
            assert_eq!(
                <Executor as MaterializedPredictionExecutor<
                    Target,
                    NumericBackend,
                    Materializer,
                >>::maximum_prefill_chunk_tokens(&extension, &speculative),
                Some(cap.min(capture_cap)),
            );
            let mut strategy =
                ReplicatedMaterializedPredictionStrategy::<
                    Target,
                    NumericBackend,
                    State,
                    NumericReplicatedMechanisms,
                    _,
                    _,
                    Inputs,
                    Materializer,
                    Mechanisms,
                >::new(&mut session, &mut extension, &speculative, Inputs, ctx);
            let mut cache = strategy.new_cache().unwrap();
            let mut executor = EmbeddedPredictionExecutor::<_, Mechanisms>::new(&mut strategy);
            let (logits, _, count) = executor
                .prefill(
                    Input::new(&prompt).with_chunk_tokens(chunk),
                    &mut cache,
                    ctx,
                )
                .unwrap()
                .into_parts();
            assert_eq!(count, cap + 1);
            if let Some((expected, target, prediction)) = &reference {
                assert_tensor_exact(&logits, expected, "joint cap prefill logits");
                assert_state_exact(
                    cache.target().unwrap(),
                    target,
                    prepared.spec().units.len(),
                    "joint cap target state",
                );
                assert_state_exact(
                    &cache.prediction().0,
                    prediction,
                    2,
                    "joint cap prediction state",
                );
            } else {
                reference = Some((
                    logits,
                    cache.target().unwrap().clone(),
                    cache.prediction().0.clone(),
                ));
            }
        }
    }
}
