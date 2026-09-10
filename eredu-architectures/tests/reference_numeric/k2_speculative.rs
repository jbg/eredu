//! Real dense/MoVA state through the independent-model speculative transaction.
use super::*;
use eredu_architectures::k2_horizon as family;
use eredu_core::{
    BoundedCompletion, BoundedCompletionOutcome, BoundedCompletionWait, Completion,
    SpeculativeExecutor,
};
use eredu_runtime::speculative::autoregressive::*;
type State = DeviceState<NumericBackend, NumericHybridLayerState>;
struct Model {
    runtime: ResidentRuntime<family::LayeredModel<NumericBackend>, NumericBackend, State>,
    layout: eredu_runtime::StateLayout,
}
impl Model {
    fn new(config: &serde_json::Value, context: &NumericContext) -> Self {
        let args = family::model_args_from_config_value(config).unwrap();
        Self {
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
        context: &NumericContext,
    ) -> Result<(NumericTensor, usize), Error> {
        Ok((
            Self::decode(model, input, state, pass, context)?,
            input.len(),
        ))
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
