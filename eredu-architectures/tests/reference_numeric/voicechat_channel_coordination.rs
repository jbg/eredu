//! Owner-only channel decisions retain the ordinary sampler and transaction.
use super::voicechat_agent_sampling::{Draw, Sampling};
use super::voicechat_rnnt_coordination::{Coordinator, coordinator_pair};
use super::voicechat_speech_sampling::{Host, Ready};
use super::*;
use eredu_architectures::voicechat::{
    agent_sampling::{AgentSampler, AgentSamplingPolicy},
    channel_decision::{
        ChannelDecisionPolicy, DecisionEvidence, DecisionInput, DecisionResolver, DecisionSource,
        ReplicatedChannelDecisions, select_channels,
    },
    channels::DualChannelLogits,
};
use eredu_core::{BackendFailure, scheduler::SemanticStateTransaction};
use eredu_runtime::{
    RealtimeGenerationContract, RealtimeGenerationState,
    replicated_execution::ReplicatedCoordinator,
};
use serde_json::Value;
use std::{cell::Cell, sync::atomic::Ordering};

thread_local! {
    static SELECTIONS: Cell<usize> = const { Cell::new(0) };
    static DRAWS: Cell<usize> = const { Cell::new(0) };
}

// Tensor transport around the existing independently checked scalar primitives.
struct TensorSampling;
impl SamplingBackend for TensorSampling {
    type Logits = NumericTensor;
    type Token = NumericTensor;
    type RandomState = Draw;
    type Context = NumericContext;
    type Error = Error;
    fn error(message: String) -> Error {
        Error::backend(message)
    }
    fn validate_token(
        token: &NumericTensor,
        domain: TokenDomain,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        let ids = token.to_i32_vec(context)?;
        if ids.len() != 1 || ids[0] < 0 || ids[0] as usize >= domain.cardinality() {
            return Err(Error::backend("invalid published channel token"));
        }
        Ok(token.clone())
    }
    fn scale_temperature(
        logits: &NumericTensor,
        temperature: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(NumericTensor::new(
            logits.shape.clone(),
            Sampling::scale_temperature(&logits.data, temperature, &RefCell::new(None))
                .map_err(Error::backend)?,
        ))
    }
    fn apply_penalties(
        logits: &NumericTensor,
        history: &[u32],
        config: PenaltyConfig,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(NumericTensor::new(
            logits.shape.clone(),
            Sampling::apply_penalties(&logits.data, history, config, &RefCell::new(None))
                .map_err(Error::backend)?,
        ))
    }
    fn apply_top_p(
        logits: NumericTensor,
        top_p: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        Ok(NumericTensor::new(
            logits.shape,
            Sampling::apply_top_p(logits.data, top_p, &RefCell::new(None))
                .map_err(Error::backend)?,
        ))
    }
    fn sample_raw(
        logits: &NumericTensor,
        temperature: f32,
        random: Option<&mut Draw>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        SELECTIONS.with(|count| count.set(count.get() + 1));
        let id = Sampling::sample_raw(&logits.data, temperature, random, &RefCell::new(None))
            .map_err(Error::backend)?;
        NumericTensor::from_i32_slice(&[id as i32], &[1], context)
    }
    fn sample_processed(
        logits: &NumericTensor,
        temperature: f32,
        random: Option<&mut Draw>,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        DRAWS.with(|count| count.set(count.get() + 1));
        let id = Sampling::sample_processed(&logits.data, temperature, random, &RefCell::new(None))
            .map_err(Error::backend)?;
        NumericTensor::from_i32_slice(&[id as i32], &[1], context)
    }
    fn token_id(token: &NumericTensor, context: &NumericContext) -> Result<u32, Error> {
        Ok(token.to_i32_vec(context)?[0] as u32)
    }
    fn token_probability(_: &NumericTensor, _: u32, _: &NumericContext) -> Result<f32, Error> {
        unreachable!()
    }
    fn apply_top_k(_: NumericTensor, _: i32, _: &NumericContext) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_min_p(_: NumericTensor, _: f32, _: &NumericContext) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_token_filter(
        _: &NumericTensor,
        _: &TokenFilter,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
    fn apply_mirostat(
        _: &NumericTensor,
        _: &[u32],
        _: PenaltyConfig,
        _: f32,
        _: f32,
        _: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        unreachable!()
    }
}

type Generation = RealtimeGenerationState<Host, Host, (), AgentSampler, Draw, Ready>;
fn generation(sampler: AgentSampler, random: Option<Draw>) -> Generation {
    Generation::new(
        Host,
        RealtimeGenerationContract::new((), 1),
        Host,
        (),
        vec![sampler],
        random,
    )
    .unwrap()
}

struct Resolver {
    owner: bool,
    fail: bool,
    preflights: Cell<usize>,
    calls: Cell<usize>,
}
impl DecisionResolver for Resolver {
    fn preflight(&self, _: TokenDomain) -> Result<(), BackendFailure> {
        assert!(self.owner, "followers must not evaluate host policy");
        self.preflights.set(self.preflights.get() + 1);
        Ok(())
    }
    fn resolve(&self, _: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure> {
        assert!(self.owner, "followers must not resolve host policy");
        self.calls.set(self.calls.get() + 1);
        if self.fail {
            return Err(BackendFailure::from_error(std::io::Error::other(
                "owner policy refused after draw",
            )));
        }
        Ok(ChannelDecisionPolicy::Tool {
            function_feedback: Some(9),
            spoken_text: Some(5),
        })
    }
}

// Corruption happens after a completed owner broadcast, so the remaining rank
// must learn of local validation failure before either adopts sampling state.
struct Wire {
    inner: Coordinator,
    corrupt: bool,
}
impl ReplicatedCoordinator<NumericTensor, NumericContext> for Wire {
    fn is_owner(&self) -> bool {
        self.inner.is_owner()
    }
    fn agree(
        &mut self,
        descriptor: [u64; 8],
        success: bool,
        context: &NumericContext,
    ) -> Result<bool, BackendFailure> {
        self.inner.agree(descriptor, success, context)
    }
    fn publish(
        &mut self,
        value: NumericTensor,
        context: &NumericContext,
    ) -> Result<NumericTensor, BackendFailure> {
        let value = self.inner.publish(value, context)?;
        if self.corrupt && !self.is_owner() {
            self.corrupt = false;
            return NumericTensor::from_i32_slice(&[-1], &[1], context)
                .map_err(BackendFailure::from_error);
        }
        Ok(value)
    }
}

#[test]
fn voicechat_channel_coordination_owner_policy_rollback_and_replay() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::AGENT_SAMPLING_REFERENCE_JSON).unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "nucleus_crossing")
        .unwrap();
    let (coordinators, timeouts) = coordinator_pair();
    std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .enumerate()
            .map(|(rank, coordinator)| {
                scope.spawn(move || {
                    let context = NumericContext::default();
                    let mut wire = Wire {
                        inner: coordinator,
                        corrupt: false,
                    };
                    let history: Vec<_> = case["history"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap() as u32)
                        .collect();
                    let mut sampler =
                        AgentSampler::new(AgentSamplingPolicy::new(0.9, 0.7, 1.4, 32).unwrap());
                    sampler.set_history(&history).unwrap();
                    let random = Draw {
                        uniform: case["uniform"].as_f64().unwrap() as f32,
                        count: 0,
                    };
                    let logits = DualChannelLogits {
                        agent: NumericTensor::new(
                            vec![1, 1, 16],
                            case["logits"]
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_f64().unwrap() as f32)
                                .collect(),
                        ),
                        function: NumericTensor::new(
                            vec![1, 1, 16],
                            (0..16).map(|id| if id == 6 { 4. } else { -3. }).collect(),
                        ),
                    };
                    let canonical =
                        generation(sampler.clone(), (rank == 0).then(|| random.clone()));
                    let mut resolver = Resolver {
                        owner: rank == 0,
                        fail: false,
                        preflights: Cell::new(0),
                        calls: Cell::new(0),
                    };

                    // Each mismatch is agreed before either rank consumes a draw.
                    for mismatch in 0..4 {
                        let mut branch = canonical.branch().unwrap();
                        if mismatch == 3 && rank == 1 {
                            branch.samplers_mut()[0].accept_effective(7).unwrap();
                        }
                        let identity = [9, 2, 10 + if mismatch == 0 { rank as u64 } else { 0 }];
                        let domain =
                            TokenDomain::new(if mismatch == 1 && rank == 1 { 15 } else { 16 });
                        let evidence = DecisionEvidence {
                            first_blank: Some(mismatch == 2 && rank == 1),
                            audio_frame_index: Some(4),
                        };
                        let mut coordination = ReplicatedChannelDecisions::new(&mut wire, identity);
                        assert!(
                            select_channels::<TensorSampling, _, _, _, _>(
                                &mut branch,
                                &logits,
                                DecisionSource::Resolved(&resolver),
                                evidence,
                                domain,
                                Some(&mut coordination),
                                &context
                            )
                            .is_err()
                        );
                        assert_eq!(
                            branch.random_state().map(|r| r.count),
                            (rank == 0).then_some(0)
                        );
                    }
                    SELECTIONS.with(|count| assert_eq!(count.get(), 0));
                    DRAWS.with(|count| assert_eq!(count.get(), 0));
                    assert_eq!(resolver.calls.get(), 0);

                    resolver.fail = true;
                    let mut branch = canonical.branch().unwrap();
                    let mut coordination = ReplicatedChannelDecisions::new(&mut wire, [9, 2, 20]);
                    let failure = select_channels::<TensorSampling, _, _, _, _>(
                        &mut branch,
                        &logits,
                        DecisionSource::Resolved(&resolver),
                        DecisionEvidence::default(),
                        TokenDomain::new(16),
                        Some(&mut coordination),
                        &context,
                    );
                    assert!(failure.is_err());
                    if rank == 0 {
                        assert!(
                            failure
                                .unwrap_err()
                                .to_string()
                                .contains("owner policy refused")
                        );
                    }
                    assert_eq!(
                        branch.random_state().map(|r| r.count),
                        (rank == 0).then_some(0)
                    );
                    assert_eq!(branch.samplers()[0].history(), history);
                    DRAWS.with(|count| assert_eq!(count.get(), usize::from(rank == 0)));

                    resolver.fail = false;
                    wire.corrupt = true;
                    let mut branch = canonical.branch().unwrap();
                    let mut coordination = ReplicatedChannelDecisions::new(&mut wire, [9, 2, 30]);
                    assert!(
                        select_channels::<TensorSampling, _, _, _, _>(
                            &mut branch,
                            &logits,
                            DecisionSource::Resolved(&resolver),
                            DecisionEvidence::default(),
                            TokenDomain::new(16),
                            Some(&mut coordination),
                            &context
                        )
                        .is_err()
                    );
                    assert_eq!(
                        branch.random_state().map(|r| r.count),
                        (rank == 0).then_some(0)
                    );
                    assert_eq!(branch.samplers()[0].history(), history);

                    wire.corrupt = false;
                    let mut branch = canonical.branch().unwrap();
                    let mut coordination = ReplicatedChannelDecisions::new(&mut wire, [9, 2, 40]);
                    let actual = select_channels::<TensorSampling, _, _, _, _>(
                        &mut branch,
                        &logits,
                        DecisionSource::Resolved(&resolver),
                        DecisionEvidence::default(),
                        TokenDomain::new(16),
                        Some(&mut coordination),
                        &context,
                    )
                    .unwrap();
                    assert_eq!(actual.raw_agent_id, case["expected"]["token"].as_u64().unwrap() as u32);
                    assert_eq!(
                        (
                            actual.raw_function_id,
                            actual.feedback.agent(),
                            actual.feedback.function(),
                            actual.spoken_text
                        ),
                        (6, 12, 9, 5)
                    );
                    let mut expected_history = history.clone();
                    expected_history.push(12);
                    assert_eq!(branch.samplers()[0].history(), expected_history);
                    assert_eq!(
                        branch.random_state().map(|r| r.count),
                        (rank == 0).then_some(1)
                    );
                    if rank == 1 {
                        SELECTIONS.with(|count| assert_eq!(count.get(), 0));
                        DRAWS.with(|count| assert_eq!(count.get(), 0));
                        assert_eq!(resolver.preflights.get(), 0);
                        assert_eq!(resolver.calls.get(), 0);
                    }
                    // Authority cannot be replayed even with a still-unpublished branch.
                    assert!(
                        select_channels::<TensorSampling, _, _, _, _>(
                            &mut branch,
                            &logits,
                            DecisionSource::Resolved(&resolver),
                            DecisionEvidence::default(),
                            TokenDomain::new(16),
                            Some(&mut coordination),
                            &context
                        )
                        .is_err()
                    );
                    assert_eq!(
                        branch.random_state().map(|r| r.count),
                        (rank == 0).then_some(1)
                    );

                    let mut ordinary = generation(sampler, Some(random)).branch().unwrap();
                    let expected = select_channels::<TensorSampling, _, _, _, _>(
                        &mut ordinary,
                        &logits,
                        ChannelDecisionPolicy::Tool {
                            function_feedback: Some(9),
                            spoken_text: Some(5),
                        }
                        .into(),
                        DecisionEvidence::default(),
                        TokenDomain::new(16),
                        None,
                        &context,
                    )
                    .unwrap();
                    assert_eq!(actual.raw_agent_id, expected.raw_agent_id);
                    assert_eq!(actual.feedback, expected.feedback);
                    assert_eq!(branch.samplers(), ordinary.samplers());
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }
    });
    assert_eq!(timeouts.load(Ordering::SeqCst), 0);
}
