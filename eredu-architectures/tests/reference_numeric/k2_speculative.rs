//! Real dense/MoVA state through the independent-model speculative transaction.
use super::*;
use eredu_architectures::k2_horizon as family;
use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    GenerationCancellationToken, InferenceGeometry, OutputDemand, SpeculativeExecutor,
    SpeculativePrefillOutcome, Submission,
};
use eredu_runtime::speculative::autoregressive::*;
type State = DeviceState<NumericBackend, NumericHybridLayerState>;
struct Model {
    runtime: ResidentRuntime<family::LayeredModel<NumericBackend>, NumericBackend, State>,
    layout: eredu_runtime::StateLayout,
    chunk: u64,
    cancel_after: Option<usize>,
    spans: Vec<eredu_runtime::prefill::PrefillChunk>,
}
impl Model {
    fn new(config: &serde_json::Value, context: &NumericContext) -> Self {
        let args = family::model_args_from_config_value(config).unwrap();
        Self {
            chunk: 512,
            cancel_after: None,
            spans: Vec::new(),
            layout: decoder::state_layout(&args).unwrap(),
            runtime: ResidentRuntime::new(
                family::LayeredModel::new(args, context).unwrap(),
                context,
            )
            .unwrap(),
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
    fn wait_bounded(self, _: BoundedCompletionWait) -> Result<BoundedCompletionOutcome, Error> {
        Ok(BoundedCompletionOutcome::Completed)
    }
}
struct Mechanisms;
impl AutoregressiveMechanisms for Mechanisms {
    type Activation = NumericTensor;
    type Model = Model;
    type Input = Vec<u32>;
    type State = State;
    type Checkpoint = State;
    type Output = NumericTensor;
    type Logits = NumericTensor;
    type Context<'a> = &'a NumericContext;
    type Completion = Done;
    type Error = Error;
    type Telemetry = ();
    fn empty(model: &mut Model, _: AutoregressivePass, _: &NumericContext) -> Result<State, Error> {
        State::create(model.layout.clone(), |_, policy| {
            Ok::<_, Error>(NumericHybridLayerState::new(policy))
        })
        .map_err(Error::backend)
    }
    fn prefill(
        model: &mut Model,
        input: &Vec<u32>,
        state: &mut State,
        pass: AutoregressivePass,
        cancellation: &GenerationCancellationToken,
        context: &NumericContext,
    ) -> Result<SpeculativePrefillOutcome<AutoregressivePrefill<NumericTensor>>, Error> {
        use eredu_runtime::{prefill::*, working_memory::*};
        let output = match pass {
            AutoregressivePass::TargetPrefill => OutputDemand::LastPosition,
            AutoregressivePass::DraftPrefill => OutputDemand::StateOnly,
            _ => return Err(Error::backend("not a prefill pass")),
        };
        let geometry = InferenceGeometry {
            batch_size: 1,
            cached_positions: 0,
            input_positions: input.len() as u64,
            max_output_tokens: 0,
            prefill_chunk_positions: model.chunk.min(input.len() as u64),
            output,
        };
        let execution = InferenceExecutionIdentity::default();
        let request = crate::memory_fixture::request(&execution, geometry)
            .map_err(Error::backend_retained_source)?;
        let mut driver = PrefillDriver::new(&execution, request, geometry, cancellation.clone())
            .map_err(Error::backend_retained_source)?;
        let mut executor = NumericPrefill {
            model,
            input,
            state,
            context,
            cancellation,
        };
        let (outcome, logits) = driver
            .run_final(&mut executor)
            .map_err(Error::backend_retained_source)?;
        Ok(match outcome {
            PrefillOutcome::Cancelled => SpeculativePrefillOutcome::Cancelled {
                evaluated_tokens: driver.completed_positions() as usize,
            },
            PrefillOutcome::Complete => {
                SpeculativePrefillOutcome::Complete(AutoregressivePrefill {
                    logits,
                    evaluated_tokens: input.len(),
                })
            }
        })
    }
    fn decode(
        model: &mut Model,
        tokens: &[u32],
        state: &mut State,
        _: AutoregressivePass,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let tokens = NumericTensor::token_ids(
            &tokens
                .iter()
                .map(|token| *token as usize)
                .collect::<Vec<_>>(),
        );
        model.runtime.forward(
            decoder::LayeredInput {
                tokens: &tokens,
                mask: None,
            },
            state,
            context,
        )
    }
    fn checkpoint(state: &State) -> Result<State, Error> {
        Ok(state.clone())
    }
    fn restore(saved: &State, _: &NumericContext) -> Result<State, Error> {
        Ok(saved.clone())
    }
    fn estimate(saved: &State) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        Self::estimate_state(saved)
    }
    fn estimate_state(saved: &State) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        let mut bytes = std::mem::size_of::<State>() as u64;
        for index in 0..saved.layout().len() {
            let cache = saved.as_ref().get(index)?.attention.as_ref()?;
            for tensor in [&cache.keys, &cache.values].into_iter().flatten() {
                bytes = bytes.checked_add(tensor.data.len() as u64 * 4)?;
            }
        }
        Some(eredu_core::execution_control::SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }
    fn logits(
        output: &NumericTensor,
        position: usize,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(output.axis_slice(1, position, position + 1))
    }
    fn completion(_: &NumericTensor, _: &NumericContext) -> Result<Done, Error> {
        Ok(Done)
    }
    fn invalid(message: &'static str) -> Error {
        Error::backend(message)
    }
}
struct NumericPrefill<'a> {
    model: &'a mut Model,
    input: &'a [u32],
    state: &'a mut State,
    context: &'a NumericContext,
    cancellation: &'a GenerationCancellationToken,
}
struct Readout;
impl<C> eredu_runtime::LayeredTraversalHook<NumericBackend, C, Error> for Readout {}
impl eredu_runtime::prefill::PrefillExecutor for NumericPrefill<'_> {
    type Output = NumericTensor;
    type Completion = Done;
    type Error = Error;
    fn submit_chunk(
        &mut self,
        chunk: &eredu_runtime::prefill::PrefillChunk,
        _: eredu_runtime::working_memory::InferenceRequest,
    ) -> Result<Submission<Option<NumericTensor>, Done>, Error> {
        let tokens = NumericTensor::token_ids(
            &self.input[chunk.input.start as usize..chunk.input.end as usize]
                .iter()
                .map(|t| *t as usize)
                .collect::<Vec<_>>(),
        );
        let (output, _) = self
            .model
            .runtime
            .forward_with_traversal_hook_with_readout(
                decoder::LayeredInput {
                    tokens: &tokens,
                    mask: None,
                },
                self.state,
                self.context,
                &mut Readout,
                chunk.output,
            )?;
        self.model.spans.push(chunk.clone());
        if self.model.cancel_after == Some(self.model.spans.len()) {
            self.cancellation.cancel();
        }
        Ok(Submission {
            output,
            completion: Done,
        })
    }
}
fn assert_tensor(a: &NumericTensor, e: &NumericTensor) {
    assert_eq!(a.shape, e.shape);
    assert_eq!(a.data.len(), e.data.len());
    for (&a, &e) in a.data.iter().zip(&e.data) {
        assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
    }
}
fn assert_cache(actual: &State, expected: &State) {
    assert_eq!(actual.layout(), expected.layout());
    for index in 0..actual.layout().len() {
        let a = actual
            .as_ref()
            .get(index)
            .unwrap()
            .attention
            .as_ref()
            .unwrap();
        let e = expected
            .as_ref()
            .get(index)
            .unwrap()
            .attention
            .as_ref()
            .unwrap();
        assert_eq!(a.offset, e.offset);
        for (a, e) in [(&a.keys, &e.keys), (&a.values, &e.values)] {
            let (a, e) = (a.as_ref().unwrap(), e.as_ref().unwrap());
            assert_eq!(a.shape, e.shape);
            for (&a, &e) in a.data.iter().zip(&e.data) {
                assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
            }
        }
    }
}
#[test]
fn k2_independent_speculation_verifies_rejects_and_restores_every_prefix() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/k2_horizon/reference.json")).unwrap();
    let context = NumericContext::default();
    for target_family in ["dense", "mova"] {
        for accepted in 1..=4 {
            let mut target = Model::new(&fixture[target_family]["config"], &context);
            let mut draft = Model::new(&fixture["dense"]["config"], &context);
            let mut executor = AutoregressiveExecutor::<Mechanisms>::new(
                &mut target,
                &mut draft,
                std::num::NonZeroUsize::new(3).unwrap(),
            );
            let mut cache = executor.new_cache(&context).unwrap();
            let (_, seed, count) = executor
                .prefill(vec![1, 3, 2], &mut cache, &context)
                .unwrap()
                .into_parts();
            assert_eq!(count, 3);
            let original_target = cache.target.clone();
            let original_draft = cache.draft.clone();
            assert!(
                executor
                    .control_snapshot_estimate(&cache, &seed)
                    .unwrap()
                    .copy_bytes
                    > 0
            );
            let durable = executor
                .control_snapshot(&cache, &seed, &context)
                .unwrap()
                .unwrap();
            let checkpoint = executor.checkpoint(&cache).unwrap();
            let mut proposal = executor.begin_proposal(&seed, 4, 3, &context).unwrap();
            let mut sibling = proposal.clone();
            let first = executor
                .proposal_logits(&mut proposal, 4, &context)
                .unwrap();
            let _ = executor.proposal_logits(&mut sibling, 8, &context).unwrap();
            let mut repeated = executor.begin_proposal(&seed, 4, 3, &context).unwrap();
            assert_eq!(
                first.data,
                executor
                    .proposal_logits(&mut repeated, 4, &context)
                    .unwrap()
                    .data
            );
            assert_cache(&cache.draft, &original_draft);
            let verification = executor
                .submit_verification(&[4, 5, 6, 7], &mut cache, &context)
                .unwrap()
                .wait()
                .unwrap();
            for row in 0..4 {
                let logits = executor
                    .verification_logits(&verification, row, &context)
                    .unwrap();
                let expected = fixture[target_family]["logits"][3 + row]
                    .as_array()
                    .unwrap();
                for (&a, e) in logits.data.iter().zip(expected) {
                    let e = e.as_f64().unwrap() as f32;
                    assert!((a - e).abs() <= 1e-5 + 1e-4 * e.abs());
                }
            }
            let (_, replayed) = executor
                .commit_verification(
                    verification,
                    proposal,
                    &mut cache,
                    &checkpoint,
                    accepted,
                    &context,
                )
                .unwrap()
                .into_parts();
            assert_eq!(replayed, if accepted < 4 { accepted } else { 0 });
            let mut ordinary = Model::new(&fixture[target_family]["config"], &context);
            let mut ordinary_state =
                Mechanisms::empty(&mut ordinary, AutoregressivePass::TargetPrefill, &context)
                    .unwrap();
            Mechanisms::decode(
                &mut ordinary,
                &[1, 3, 2, 4, 5, 6, 7][..3 + accepted],
                &mut ordinary_state,
                AutoregressivePass::TargetPrefill,
                &context,
            )
            .unwrap();
            assert_cache(&cache.target, &ordinary_state);
            executor
                .restore_control_snapshot(&mut cache, &durable.0, &durable.1, &context)
                .unwrap()
                .unwrap();
            assert_cache(&cache.target, &original_target);
            assert_cache(&cache.draft, &original_draft);
            executor
                .submit_verification(&[4, 8], &mut cache, &context)
                .unwrap()
                .wait()
                .unwrap();
            executor
                .restore_checkpoint(&mut cache, &checkpoint, &context)
                .unwrap();
            assert_cache(&cache.target, &original_target);
            assert_cache(&cache.draft, &original_draft);
        }
    }
}

#[test]
fn k2_selective_prefill_and_three_verification_rounds_match_full_state() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/k2_horizon/reference.json")).unwrap();
    for family in ["dense", "mova"] {
        let context = NumericContext::default();
        let mut baseline = Model::new(&fixture[family]["config"], &context);
        let mut baseline_state =
            Mechanisms::empty(&mut baseline, AutoregressivePass::TargetPrefill, &context).unwrap();
        let prompt = vec![1, 3, 2, 4, 5];
        let full = Mechanisms::decode(
            &mut baseline,
            &prompt,
            &mut baseline_state,
            AutoregressivePass::TargetPrefill,
            &context,
        )
        .unwrap();
        assert!(full.data.iter().any(|x| x.abs() > 1e-6));
        let head = context
            .projections
            .lock()
            .unwrap()
            .last()
            .unwrap()
            .0
            .clone();
        let mut target = Model::new(&fixture[family]["config"], &context);
        target.chunk = 2;
        let mut draft = Model::new(&fixture["dense"]["config"], &context);
        draft.chunk = 2;
        let mut draft_reference = Model::new(&fixture["dense"]["config"], &context);
        let mut draft_state = Mechanisms::empty(
            &mut draft_reference,
            AutoregressivePass::DraftPrefill,
            &context,
        )
        .unwrap();
        Mechanisms::decode(
            &mut draft_reference,
            &prompt,
            &mut draft_state,
            AutoregressivePass::DraftPrefill,
            &context,
        )
        .unwrap();
        let mut executor = AutoregressiveExecutor::<Mechanisms>::new(
            &mut target,
            &mut draft,
            std::num::NonZeroUsize::new(3).unwrap(),
        );
        let mut cache = executor.new_cache(&context).unwrap();
        context.projections.lock().unwrap().clear();
        let (last, mut seed, count) = executor
            .prefill(prompt, &mut cache, &context)
            .unwrap()
            .into_parts();
        assert_eq!(count, 5);
        assert_tensor(&last, &full.axis_slice(1, 4, 5));
        assert_cache(&cache.target, &baseline_state);
        assert_cache(&cache.draft, &draft_state);
        assert_eq!(
            context
                .projections
                .lock()
                .unwrap()
                .iter()
                .filter(|(name, _)| name == &head)
                .map(|(_, shape)| shape[1])
                .collect::<Vec<_>>(),
            [1]
        );
        for (round, accepted) in [1, 3, 2].into_iter().enumerate() {
            let tokens = [6 + round as u32, 2, 1];
            let checkpoint = executor.checkpoint(&cache).unwrap();
            let proposal = executor
                .begin_proposal(&seed, tokens[0], 2, &context)
                .unwrap();
            context.projections.lock().unwrap().clear();
            let verification = executor
                .submit_verification(&tokens, &mut cache, &context)
                .unwrap()
                .wait()
                .unwrap();
            assert_eq!(
                context
                    .projections
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(name, _)| name == &head)
                    .map(|(_, shape)| shape[1])
                    .collect::<Vec<_>>(),
                [3]
            );
            let mut trial = baseline_state.clone();
            let all = Mechanisms::decode(
                &mut baseline,
                &tokens,
                &mut trial,
                AutoregressivePass::TargetPrefill,
                &context,
            )
            .unwrap();
            for row in 0..3 {
                assert_tensor(
                    &executor
                        .verification_logits(&verification, row, &context)
                        .unwrap(),
                    &all.axis_slice(1, row, row + 1),
                );
            }
            let (next, replayed) = executor
                .commit_verification(
                    verification,
                    proposal,
                    &mut cache,
                    &checkpoint,
                    accepted,
                    &context,
                )
                .unwrap()
                .into_parts();
            assert_eq!(replayed, if accepted == 3 { 0 } else { accepted });
            seed = next;
            Mechanisms::decode(
                &mut baseline,
                &tokens[..accepted],
                &mut baseline_state,
                AutoregressivePass::TargetPrefill,
                &context,
            )
            .unwrap();
            Mechanisms::decode(
                &mut draft_reference,
                &tokens[..accepted],
                &mut draft_state,
                AutoregressivePass::DraftPrefill,
                &context,
            )
            .unwrap();
            assert_cache(&cache.target, &baseline_state);
            assert_cache(&cache.draft, &draft_state);
        }
        drop(executor);
        for (model, demand) in [
            (&target, OutputDemand::LastPosition),
            (&draft, OutputDemand::StateOnly),
        ] {
            assert_eq!(
                model
                    .spans
                    .iter()
                    .map(|s| s.input.clone())
                    .collect::<Vec<_>>(),
                [0..2, 2..4, 4..5]
            );
            assert_eq!(
                model.spans.iter().map(|s| s.output).collect::<Vec<_>>(),
                [OutputDemand::StateOnly, OutputDemand::StateOnly, demand]
            );
        }
    }
}
#[test]
fn k2_target_and_draft_completed_span_cancellation_restore_joint_cache() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../fixtures/k2_horizon/reference.json")).unwrap();
    for cancel_draft in [false, true] {
        let context = NumericContext::default();
        let mut target = Model::new(&fixture["mova"]["config"], &context);
        target.chunk = 2;
        let mut draft = Model::new(&fixture["dense"]["config"], &context);
        draft.chunk = 2;
        if cancel_draft {
            draft.cancel_after = Some(1);
        } else {
            target.cancel_after = Some(1);
        }
        let mut executor = AutoregressiveExecutor::<Mechanisms>::new(
            &mut target,
            &mut draft,
            std::num::NonZeroUsize::new(2).unwrap(),
        );
        let mut cache = executor.new_cache(&context).unwrap();
        let checkpoint = executor.checkpoint(&cache).unwrap();
        let cancel = GenerationCancellationToken::new();
        assert!(matches!(
            executor
                .prefill_cancellable(vec![1, 3, 2, 4, 5], &mut cache, &cancel, &context)
                .unwrap(),
            SpeculativePrefillOutcome::Cancelled { evaluated_tokens } if evaluated_tokens == if cancel_draft {5} else {2}
        ));
        executor
            .restore_checkpoint(&mut cache, &checkpoint, &context)
            .unwrap();
        for state in [&cache.target, &cache.draft] {
            for i in 0..state.layout().len() {
                let layer = state.as_ref().get(i).unwrap();
                let kv = layer.attention.as_ref().unwrap();
                assert_eq!(kv.offset, 0);
                assert!(kv.keys.is_none());
                assert!(kv.values.is_none());
            }
        }
        drop(executor);
        assert_eq!(target.spans.len(), if cancel_draft { 3 } else { 1 });
        assert_eq!(draft.spans.len(), usize::from(cancel_draft));
    }
}
