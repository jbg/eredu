//! Real masked attention through the single-cache speculative transaction.
use super::*;
use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    SpeculativeExecutor, SpeculativeProposalStep,
};
use eredu_runtime::speculative::{autoregressive::*, shared_backbone::*};

type Runtime = ResidentRuntime<TextModel<NumericBackend>, NumericBackend, State>;
struct Model {
    runtime: Runtime,
    canvas_widths: Vec<usize>,
    random_seeds: Vec<u64>,
    fail_verification: bool,
}
impl Model {
    fn new(context: &NumericContext) -> Self {
        Self {
            runtime: ResidentRuntime::new(TextModel::new(family(), context).unwrap(), context)
                .unwrap(),
            canvas_widths: Vec::new(),
            random_seeds: Vec::new(),
            fail_verification: false,
        }
    }
}
struct Done;
impl Completion for Done {
    type Error = Error;
    fn is_complete(&self) -> Result<bool, Error> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Error> {
        Ok(())
    }
}
impl BoundedCompletion for Done {
    fn supports_cancellation(mode: eredu_core::CompletionCancellationMode) -> bool {
        mode == eredu_core::CompletionCancellationMode::QuarantineUntilComplete
    }
    fn wait_bounded(self, _: BoundedCompletionWait) -> Result<BoundedCompletionOutcome, Error> {
        Ok(BoundedCompletionOutcome::Completed)
    }
}
struct Mechanisms;
impl AutoregressiveMechanisms for Mechanisms {
    type Model = Model;
    type Input = Vec<u32>;
    type State = State;
    type Checkpoint = State;
    type Output = NumericTensor;
    type Logits = NumericTensor;
    type Context<'a> = &'a NumericContext;
    type Completion = Done;
    type Telemetry = ();
    type Error = Error;
    fn empty(_: &mut Model, _: AutoregressivePass, _: &NumericContext) -> Result<State, Error> {
        Ok(state(&family()))
    }
    fn prefill(
        model: &mut Model,
        tokens: &Vec<u32>,
        cache: &mut State,
        pass: AutoregressivePass,
        context: &NumericContext,
    ) -> Result<(NumericTensor, usize), Error> {
        Ok((
            Self::decode(model, tokens, cache, pass, context)?,
            tokens.len(),
        ))
    }
    fn decode(
        model: &mut Model,
        tokens: &[u32],
        cache: &mut State,
        pass: AutoregressivePass,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let tokens =
            NumericTensor::token_ids(&tokens.iter().map(|x| *x as usize).collect::<Vec<_>>());
        let output =
            model
                .runtime
                .forward(input(&tokens, AttentionInvocation::CAUSAL), cache, context)?;
        if model.fail_verification && pass == AutoregressivePass::Verification {
            return Err(Error::backend(
                "injected failure after tentative verification append",
            ));
        }
        Ok(output)
    }
    fn checkpoint(cache: &State) -> Result<State, Error> {
        Ok(cache.clone())
    }
    fn restore(cache: &State, _: &NumericContext) -> Result<State, Error> {
        Ok(cache.clone())
    }
    fn estimate(cache: &State) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Self::estimate_state(cache)
    }
    fn estimate_state(cache: &State) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut bytes = std::mem::size_of::<State>() as u64;
        for layer in 0..cache.layout().len() {
            let mut copy = cache.clone();
            let cache = copy.layer(layer).unwrap();
            for tensor in [&cache.keys, &cache.values].into_iter().flatten() {
                bytes += tensor.data.len() as u64 * 4;
            }
        }
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }
    fn logits(
        output: &NumericTensor,
        index: usize,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(output.axis_slice(1, index, index + 1))
    }
    fn completion(_: &NumericTensor, _: &NumericContext) -> Result<Done, Error> {
        Ok(Done)
    }
    fn invalid(message: &'static str) -> Error {
        Error::backend(message)
    }
}
impl SharedBackboneMechanisms for Mechanisms {
    type ActivationTensor = NumericTensor;
    type DraftOptions = ();
    type ProposalRandomness = u64;
    type RefinementState = ();
    fn refinement_state_estimate(_: &()) -> Option<eredu_core::execution_control::SnapshotEstimate> { Some(eredu_core::execution_control::SnapshotEstimate { retained_bytes: 0, copy_bytes: 0 }) }
    fn prepare_draft_options(
        _: &Model,
        _: eredu_core::decoding::SpeculationOptions,
        _: f32,
    ) -> Result<(), eredu_core::decoding::DecodingModeError> {
        Ok(())
    }
    fn proposal_randomness_bytes(_: &u64) -> Option<u64> {
        Some(8)
    }
    fn draft_options_bytes(_: &()) -> Option<u64> {
        Some(0)
    }
    fn draft(
        model: &mut Model,
        prefix: &State,
        last: u32,
        capacity: usize,
        _: &(),
        randomness: &mut u64,
        context: &NumericContext,
        _: Option<&mut dyn eredu_runtime::inspection::SpeculativeActivationObserver<NumericTensor, Error>>,
    ) -> Result<Vec<u32>, Error> {
        // Full geometry survives a one-token budget. Two confidence transfers
        // freeze selected candidates while remaining positions are refined.
        model.canvas_widths.push(3);
        model.random_seeds.push(*randomness);
        *randomness += 1;
        let mut cache = prefix.clone();
        let mut canvas = [last, 0, 0];
        let mut eligible = vec![false, true, true];
        for _ in 0..2 {
            let tokens =
                NumericTensor::token_ids(&canvas.iter().map(|x| *x as usize).collect::<Vec<_>>());
            let logits = model.runtime.forward(
                input(&tokens, AttentionInvocation::DRAFT),
                &mut cache,
                context,
            )?;
            let rows: Vec<_> = logits
                .data
                .chunks(11)
                .map(|row| {
                    let mut row = row.to_vec();
                    row[0] = f32::NEG_INFINITY;
                    row
                })
                .collect();
            let sampled: Vec<_> = rows
                .iter()
                .map(|row| {
                    eredu_runtime::generation::refinement::masked::sample_position(
                        row,
                        eredu_core::refinement::MaskedSampling::GumbelPower,
                        0.0,
                        || panic!("greedy drafting consumes no noise"),
                    )
                    .unwrap()
                })
                .collect();
            let transferred = eredu_runtime::generation::refinement::masked::transfer_positions(
                &rows, &sampled, &eligible, 1, None,
            )
            .unwrap();
            for position in 1..3 {
                if transferred[position] {
                    canvas[position] = sampled[position];
                    eligible[position] = false;
                }
            }
        }
        assert_cache(&cache, prefix);
        Ok(canvas[1..=capacity].to_vec())
    }
}
fn assert_cache(actual: &State, expected: &State) {
    for layer in 0..2 {
        let mut actual = actual.clone();
        let mut expected = expected.clone();
        let actual = actual.layer(layer).unwrap();
        let expected = expected.layer(layer).unwrap();
        assert_eq!(actual.offset, expected.offset);
        for (a, e) in [
            (&actual.keys, &expected.keys),
            (&actual.values, &expected.values),
        ] {
            match (a, e) {
                (Some(a), Some(e)) => assert_tensor_close(a, e, "canonical accepted prefix"),
                (None, None) => (),
                _ => panic!("different cache frontiers"),
            }
        }
    }
}

#[test]
fn shared_backbone_masked_transactions_keep_one_cache_and_restore_proposal_randomness() {
    let context = NumericContext::default();
    for (capacity, accepted) in [(0, 1), (2, 1), (2, 2), (2, 3), (1, 2)] {
        let mut model = Model::new(&context);
        let storage = parameter_storage(&model.runtime);
        {
            let mut executor = SharedBackboneExecutor::<Mechanisms>::new(
                &mut model,
                std::num::NonZeroUsize::new(2).unwrap(),
            );
            let mut cache = executor.new_cache(&context).unwrap();
            let (_, seed, _) = executor
                .prefill(
                    SharedBackboneInput::new(vec![1, 4, 2], (), 17, 29),
                    &mut cache,
                    &context,
                )
                .unwrap()
                .into_parts();
            let prompt_cache = cache.clone();
            let durable = executor
                .control_snapshot(&cache, &seed, &context)
                .unwrap()
                .unwrap();
            let checkpoint = executor.checkpoint(&cache).unwrap();
            let mut draft = executor
                .begin_proposal(&seed, 5, capacity, &context)
                .unwrap();
            assert_cache(&cache, &prompt_cache);
            let mut tokens = vec![5];
            for _ in 0..capacity {
                let SpeculativeProposalStep::TargetMatchingToken(token) = executor
                    .proposal_step(&mut draft, *tokens.last().unwrap(), &context)
                    .unwrap()
                else {
                    panic!("masked drafting must not invent causal probabilities")
                };
                tokens.push(token);
            }
            let verification = executor
                .submit_verification(&tokens, &mut cache, &context)
                .unwrap()
                .wait()
                .unwrap();
            let (next_seed, replayed) = executor
                .commit_verification(
                    verification,
                    draft,
                    &mut cache,
                    &checkpoint,
                    accepted,
                    &context,
                )
                .unwrap()
                .into_parts();
            assert_eq!(replayed, if accepted < tokens.len() { accepted } else { 0 });
            let mut ordinary = Model::new(&context);
            let mut ordinary_cache = state(&family());
            let mut committed = vec![1, 4, 2];
            committed.extend_from_slice(&tokens[..accepted]);
            Mechanisms::decode(
                &mut ordinary,
                &committed,
                &mut ordinary_cache,
                AutoregressivePass::TargetPrefill,
                &context,
            )
            .unwrap();
            assert_cache(&cache, &ordinary_cache);
            // A mismatch/bonus is still pending: draft seeds it once, over the
            // accepted causal frontier, without appending it during refinement.
            executor.begin_proposal(&next_seed, 7, 1, &context).unwrap();
            assert_cache(&cache, &ordinary_cache);
            let restored = executor
                .restore_control_snapshot(&mut cache, &durable.0, &durable.1, &context)
                .unwrap()
                .unwrap();
            assert_cache(&cache, &prompt_cache);
            executor
                .begin_proposal(&restored, 5, capacity, &context)
                .unwrap();
        }
        assert_eq!(parameter_storage(&model.runtime), storage);
        if capacity == 0 {
            assert_eq!(
                model.canvas_widths,
                [3],
                "causal verification and its restored seed never refine a proposal canvas"
            );
            assert_eq!(
                model.random_seeds,
                [17],
                "causal verification preserves proposal randomness"
            );
        } else {
            assert_eq!(model.canvas_widths, [3, 3, 3]);
            assert_eq!(model.random_seeds, [17, 18, 17]);
        }
    }
}

#[test]
fn incremental_shared_target_matching_keeps_exact_causal_prefixes_and_isolated_cursors() {
    let context = NumericContext::default();
    for accept_second in [false, true] {
        let mut model = Model::new(&context);
        let storage = parameter_storage(&model.runtime);
        {
            let mut executor = SharedBackboneExecutor::<Mechanisms>::new(
                &mut model,
                std::num::NonZeroUsize::new(2).unwrap(),
            );
            let mut cache = executor.new_cache(&context).unwrap();
            executor
                .prefill(
                    SharedBackboneInput::new(vec![1, 4, 2], (), 17, 29),
                    &mut cache,
                    &context,
                )
                .unwrap();
            let mut transaction = SharedBackboneTransaction::submit(
                &mut executor,
                &mut cache,
                5,
                (),
                17,
                eredu_core::generation::SpeculativeSchedulerOptions::default()
                    .completion_wait()
                    .unwrap(),
                &context,
            )
            .unwrap();
            let proposals = transaction.proposals().to_vec();
            let mut fork = transaction.clone();
            // The evolving target constraint can force a proposal. Commitment
            // without the corresponding sampled target is rejected first.
            assert!(matches!(
                transaction.commit_target_token(&mut executor, proposals[0], &context),
                Err(SharedBackboneTransactionError::Policy(eredu_core::GenerationError::VerificationCommitmentMismatch))
            ));
            transaction.stage_target_token(proposals[0]).unwrap();
            assert!(matches!(
                transaction.stage_target_token((proposals[0] + 1) % 11),
                Err(SharedBackboneTransactionError::Policy(eredu_core::GenerationError::VerificationDecisionPending))
            ));
            assert!(matches!(
                transaction.logits(&executor, &context),
                Err(SharedBackboneTransactionError::Policy(eredu_core::GenerationError::VerificationDecisionPending))
            ));
            let (continues, first) = transaction
                .commit_target_token(&mut executor, proposals[0], &context)
                .unwrap();
            assert!(continues);
            let mut ordinary = Model::new(&context);
            let mut expected = state(&family());
            let mut prefix = vec![1, 4, 2, 5, proposals[0]];
            Mechanisms::decode(
                &mut ordinary,
                &prefix,
                &mut expected,
                AutoregressivePass::TargetPrefill,
                &context,
            )
            .unwrap();
            assert_cache(&first, &expected);
            let second = if accept_second {
                proposals[1]
            } else {
                (proposals[1] + 1) % 11
            };
            transaction.stage_target_token(second).unwrap();
            let (continues, second_cache) = transaction
                .commit_target_token(&mut executor, second, &context)
                .unwrap();
            assert_eq!(continues, accept_second);
            if accept_second {
                prefix.push(second);
                expected = state(&family());
                Mechanisms::decode(
                    &mut ordinary,
                    &prefix,
                    &mut expected,
                    AutoregressivePass::TargetPrefill,
                    &context,
                )
                .unwrap();
                // Full acceptance samples a bonus row. Its selected token
                // remains pending append, regardless of its proposal identity.
                transaction.stage_target_token(7).unwrap();
                let (continues, bonus_cache) = transaction
                    .commit_target_token(&mut executor, 7, &context)
                    .unwrap();
                assert!(!continues);
                assert_cache(&bonus_cache, &expected);
                assert!(matches!(
                    transaction.stage_target_token(7),
                    Err(SharedBackboneTransactionError::Policy(eredu_core::GenerationError::VerificationFinished))
                ));
            }
            assert_cache(&second_cache, &expected);
            let replacement = (proposals[0] + 1) % 11;
            fork.stage_target_token(replacement).unwrap();
            let (continues, fork_cache) = fork
                .commit_target_token(&mut executor, replacement, &context)
                .unwrap();
            assert!(!continues);
            assert!(matches!(
                fork.logits(&executor, &context),
                Err(SharedBackboneTransactionError::Policy(eredu_core::GenerationError::VerificationFinished))
            ));
            assert!(matches!(
                fork.commit_target_token(&mut executor, replacement, &context),
                Err(SharedBackboneTransactionError::Policy(eredu_core::GenerationError::VerificationFinished))
            ));
            let mut expected_fork = state(&family());
            Mechanisms::decode(
                &mut ordinary,
                &[1, 4, 2, 5],
                &mut expected_fork,
                AutoregressivePass::TargetPrefill,
                &context,
            )
            .unwrap();
            assert_cache(&fork_cache, &expected_fork);
            assert_eq!(
                *fork.proposal_randomness(),
                *transaction.proposal_randomness()
            );
        }
        assert_eq!(parameter_storage(&model.runtime), storage);
    }
}

#[test]
fn committed_block_suffix_settles_once_and_retains_the_final_causal_row() {
    let context = NumericContext::default();
    let mut model = Model::new(&context);
    let storage = parameter_storage(&model.runtime);
    {
        let mut executor = CausalExecutor::<Mechanisms, eredu_runtime::speculative::autoregressive::NoTargetRefinement>::new(&mut model, eredu_runtime::speculative::autoregressive::NoTargetRefinement);
        let mut cache = executor.new_cache(&context).unwrap();
        let (_, seed, _) = executor
            .prefill(eredu_runtime::speculative::autoregressive::AutoregressiveInput::new(vec![1, 4, 2], ()), &mut cache, &context)
            .unwrap()
            .into_parts();
        let saved = cache.clone();
        let wait = eredu_core::generation::SpeculativeSchedulerOptions::default()
            .completion_wait()
            .unwrap();
        assert!(matches!(
            eredu_core::speculative::settle_committed_suffix(
                &mut executor,
                &mut cache,
                &seed,
                &[],
                wait,
                |_| Ok(()),
                &context
            ),
            Err(eredu_core::SpeculativeDriverError::Generation(
                eredu_core::generation::GenerationError::EmptyCommittedSuffix
            ))
        ));
        assert_cache(&cache, &saved);
        let settled = eredu_core::speculative::settle_committed_suffix(
            &mut executor,
            &mut cache,
            &seed,
            &[5, 7, 8, 9, 3],
            wait,
            |_| Ok(()),
            &context,
        )
        .unwrap();
        let mut ordinary = Model::new(&context);
        let mut expected = state(&family());
        let scores = Mechanisms::decode(
            &mut ordinary,
            &[1, 4, 2, 5, 7, 8, 9, 3],
            &mut expected,
            AutoregressivePass::TargetPrefill,
            &context,
        )
        .unwrap();
        assert_cache(&cache, &expected);
        assert_tensor_close(
            &settled.logits,
            &Mechanisms::logits(&scores, 7, &context).unwrap(),
            "settled block next-token row",
        );
        assert_cache(&cache, &expected);
    }
    assert_eq!(parameter_storage(&model.runtime), storage);
    assert!(model.canvas_widths.is_empty());
    assert!(model.random_seeds.is_empty());
    assert!(
        model.canvas_widths.is_empty(),
        "causal block append performs no masked refinement"
    );
}

#[test]
fn failed_shared_verification_restores_the_canonical_prefix() {
    let context = NumericContext::default();
    let mut model = Model::new(&context);
    model.fail_verification = true;
    let storage = parameter_storage(&model.runtime);
    {
        let mut executor = SharedBackboneExecutor::<Mechanisms>::new(
            &mut model,
            std::num::NonZeroUsize::new(2).unwrap(),
        );
        let mut cache = executor.new_cache(&context).unwrap();
        let (_, seed, _) = executor
            .prefill(
                SharedBackboneInput::new(vec![1, 4, 2], (), 17, 29),
                &mut cache,
                &context,
            )
            .unwrap()
            .into_parts();
        let baseline = cache.clone();
        let unsupported = BoundedCompletionWait::new(
            std::time::Duration::from_secs(1),
            eredu_core::CompletionCancellationMode::NativeCancel,
        )
        .unwrap();
        assert!(matches!(
            SharedBackboneTransaction::submit(
                &mut executor,
                &mut cache,
                5,
                (),
                17,
                unsupported,
                &context
            ),
            Err(SharedBackboneTransactionError::UnsupportedCompletionCancellation { .. })
        ));
        assert_cache(&cache, &baseline);
        assert!(matches!(
            eredu_core::speculative::settle_committed_suffix(
                &mut executor,
                &mut cache,
                &seed,
                &[5],
                unsupported,
                |_| Ok(()),
                &context
            ),
            Err(eredu_core::SpeculativeDriverError::UnsupportedCompletionCancellation { .. })
        ));
        assert_cache(&cache, &baseline);
        let result = SharedBackboneTransaction::submit(
            &mut executor,
            &mut cache,
            5,
            (),
            17,
            eredu_core::generation::SpeculativeSchedulerOptions::default()
                .completion_wait()
                .unwrap(),
            &context,
        );
        assert!(matches!(
            result,
            Err(SharedBackboneTransactionError::Backend(_))
        ));
        assert_cache(&cache, &baseline);
    }
    assert_eq!(parameter_storage(&model.runtime), storage);
    assert_eq!(model.random_seeds, [17]);
}
