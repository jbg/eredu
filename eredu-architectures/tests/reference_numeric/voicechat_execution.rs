//! Coupled equations use compact nonzero towers with the released frame clocks.
use super::voicechat_rnnt_coordination::{Coordinator, coordinator_pair};
use super::voicechat_transformer::CausalCache;
use super::*;
use eredu_architectures::voicechat::{
    input::{InputExecution, InputState},
    perception::PerceptionState,
    tts_ingress::PromptLayout,
};
use eredu_evaluation::voicechat::coupled::{config, controls, fixture, mapping};
use eredu_media::audio::streaming::{FinalPcmFrame, PcmFinishPolicy};
use eredu_runtime::replicated_execution::ReplicatedCoordinator;
use serde_json::Value;

fn input_state() -> InputState<NumericTensor> {
    eredu_evaluation::voicechat::coupled::input_state()
}
fn perception(
    context: &NumericContext,
) -> eredu_architectures::voicechat::perception_rnnt::PerceptionRnnt<NumericBackend> {
    eredu_evaluation::voicechat::coupled::perception::<NumericBackend>(context)
}
fn tts(
    fixture: &Value,
    context: &NumericContext,
) -> eredu_architectures::voicechat::tts_generator::TtsGenerator<NumericBackend> {
    eredu_evaluation::voicechat::coupled::tts::<NumericBackend>(fixture, context)
}
fn codec(
    fixture: &Value,
    context: &NumericContext,
) -> eredu_codec::ear::AudioDecoder<NumericBackend> {
    eredu_evaluation::voicechat::coupled::codec::<NumericBackend>(fixture, context)
}
fn prompt(fixture: &Value, context: &NumericContext) -> (NumericTensor, PromptLayout) {
    eredu_evaluation::voicechat::coupled::prompt(fixture, context)
}
fn caches() -> [CausalCache; 2] {
    [
        CausalCache(NumericCache::new(Some(3))),
        CausalCache(NumericCache::new(None)),
    ]
}

use eredu_architectures::voicechat::{
    agent_sampling::{AgentSampler, AgentSamplingPolicy},
    backbone_execution::{BackboneExecution, OrdinaryBackbone, ParallelBackbone},
    channel_decision::{ChannelDecisionPolicy, FeedbackTokens},
    channels::{ChannelFusion, PromptPolicy},
    execution::{Components, ExecutionCoordination, ExecutionError, Invocation, ToolAudio},
    function_head_layered::{FunctionHeadExecution, ParallelFunctionHead},
    output::OutputDecisions,
    perception_rnnt::PerceptionRnntLimits,
    rnnt::RnntLimits,
    schedule::{Progress, ScheduleLimits, SchedulePolicy, ScheduleState},
    speech::SpeechComponents,
    state::{NeuralStateParts, State, StateBranch},
    tts_conditioning_layered::ResidentConditioning,
    tts_feedback::{OutputPolicy, WaveformSilence},
    tts_mog::MogHead,
    tts_rvq::RvqRefinement,
    tts_transformer::TtsTransformer,
};
use eredu_core::{RealtimeChannelPayload, scheduler::SemanticStateTransaction};
use eredu_runtime::{
    CompletedRealtimeFrame, DenseSamplingBackend, GumbelTruncation, RealtimeFrameChannel,
    RealtimeFrameTransition, RealtimeGenerationBranch, RealtimeGenerationState,
    RealtimeHostChannelValues,
};
use std::convert::Infallible;

type Backbone = DeviceState<NumericBackend, NumericHybridLayerState>;
#[derive(Clone)]
struct Neural {
    backbone: Backbone,
    tts: [CausalCache; 2],
}
impl NeuralStateParts for Neural {
    type Backbone = Backbone;
    type TtsState = [CausalCache];
    fn parts_mut(&mut self) -> (&mut Backbone, &mut [CausalCache]) {
        (&mut self.backbone, &mut self.tts)
    }
}
impl SemanticStateTransaction for Neural {
    type Branch = Self;
    type Error = Infallible;
    fn branch(&self) -> Result<Self, Infallible> {
        // NumericTensor owns its arrays; cloning creates independent mutable
        // backbone and guidance cache storage, not native cache aliases.
        Ok(self.clone())
    }
    fn commit_branch(&mut self, branch: Self) -> Result<(), Infallible> {
        *self = branch;
        Ok(())
    }
}
struct Ready;
impl Completion for Ready {
    type Error = Infallible;
    fn is_complete(&self) -> Result<bool, Infallible> {
        Ok(true)
    }
    fn wait(&self) -> Result<(), Infallible> {
        Ok(())
    }
}
type Branch<
    C = PerceptionState<NumericTensor>,
    K = eredu_codec::ear_decoder::DecoderState<NumericTensor>,
> = RealtimeGenerationBranch<
    StateBranch<NumericTensor, Neural, C, K>,
    ScheduleState,
    (),
    AgentSampler,
    i32,
    Ready,
>;
type Generation<
    C = PerceptionState<NumericTensor>,
    K = eredu_codec::ear_decoder::DecoderState<NumericTensor>,
> = RealtimeGenerationState<
    State<NumericTensor, Neural, C, K>,
    ScheduleState,
    (),
    AgentSampler,
    i32,
    Ready,
>;

impl DenseSamplingBackend for NumericBackend {
    fn uniform_f32(
        shape: &[i32],
        random: &mut i32,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        assert_eq!(*random % 2, 0, "each slot starts with a uniform draw");
        *random += 1;
        NumericTensor::full_f32(0.5, shape, context)
    }
    fn standard_normal_f32(
        shape: &[i32],
        random: &mut i32,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        assert_eq!(*random % 2, 1, "Gaussian follows mixture selection");
        *random += 1;
        NumericTensor::full_f32(0.1, shape, context)
    }
    fn dense_shape(value: &NumericTensor) -> &[i32] {
        value.shape()
    }
    fn gumbel_max(
        logits: &NumericTensor,
        uniform: &NumericTensor,
        truncation: GumbelTruncation,
        epsilon: f32,
        context: &NumericContext,
    ) -> Result<NumericTensor, Error> {
        assert_eq!(logits.shape(), uniform.shape());
        assert!(uniform.data.iter().all(|v| *v == 0.5));
        assert!(matches!(truncation, GumbelTruncation::TopP(p) if p > 0. && p <= 1.));
        assert_eq!(epsilon, 1e-8);
        // Equal uniform draws add one common Gumbel term. Nucleus always keeps
        // the largest logit, so independent scalar argmax is the exact result.
        let width = *logits.shape().last().unwrap() as usize;
        let ids: Vec<_> = logits
            .data
            .chunks_exact(width)
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as i32
            })
            .collect();
        NumericTensor::from_i32_slice(&ids, &logits.shape()[..logits.shape().len() - 1], context)
    }
}

#[derive(Debug, PartialEq)]
struct Record {
    invocation: u64,
    before: Progress,
    after: Progress,
    decisions: Option<OutputDecisions>,
    codes: Vec<i32>,
    pcm: Vec<f32>,
    start_sample: Option<u64>,
    tail: bool,
    retained: Vec<f32>,
    backbone_retained_len: usize,
    projected: Option<Vec<f32>>,
    random: Option<i32>,
}
fn retain<
    C: eredu_nn::RetainedTensorRoots<NumericTensor>,
    K: eredu_nn::RetainedTensorRoots<NumericTensor>,
>(
    state: &StateBranch<NumericTensor, Neural, C, K>,
) -> Vec<f32> {
    let mut values = Vec::new();
    let mut add = |tensor: &NumericTensor| values.extend_from_slice(&tensor.data);
    for layer in state.neural.backbone.as_ref() {
        for tensor in layer.retained_values() {
            add(tensor);
        }
    }
    for cache in &state.neural.tts {
        for tensor in cache.0.keys.iter().chain(cache.0.values.iter()) {
            add(tensor);
        }
    }
    state.visit_component_tensors(&mut add);
    values
}
fn host(
    frame: &CompletedRealtimeFrame<NumericTensor, NumericTensor>,
) -> CompletedRealtimeFrame<RealtimeHostChannelValues, ()> {
    CompletedRealtimeFrame::new(
        frame.channels().clone(),
        frame.batch(),
        frame
            .values()
            .iter()
            .map(|value| {
                let payload = match &frame.channels().channel(value.channel()).unwrap().payload {
                    RealtimeChannelPayload::Categorical { .. } => {
                        RealtimeHostChannelValues::Categorical(
                            value.payload().data.iter().map(|v| *v as i32).collect(),
                        )
                    }
                    RealtimeChannelPayload::Dense { .. } => {
                        RealtimeHostChannelValues::Dense(value.payload().data.clone())
                    }
                };
                RealtimeFrameChannel::new(value.channel(), value.coordinate(), payload)
            })
            .collect(),
        vec![],
        frame.decision_channels().to_vec(),
        vec![],
    )
    .unwrap()
}
struct Transition<
    'a,
    R,
    F,
    I = eredu_architectures::voicechat::perception_rnnt::PerceptionRnnt<NumericBackend>,
    A = eredu_codec::ear::AudioDecoder<NumericBackend>,
> {
    components: Components<
        'a,
        NumericBackend,
        R,
        I,
        TtsTransformer<NumericBackend>,
        MogHead<NumericBackend>,
        RvqRefinement<NumericBackend>,
        ResidentConditioning<NumericBackend>,
        A,
        F,
    >,
    context: &'a NumericContext,
    codes: &'a NumericTensor,
    layout: &'a PromptLayout,
    coordinator: Option<Coordinator>,
    attempt: u64,
}
impl<R, F, I, A>
    RealtimeFrameTransition<
        StateBranch<NumericTensor, Neural, I::PerceptionCache, A::NeuralState>,
        ScheduleState,
        (),
        AgentSampler,
        i32,
        Ready,
    > for Transition<'_, R, F, I, A>
where
    R: BackboneExecution<NumericBackend, Backbone>,
    F: FunctionHeadExecution<NumericBackend>,
    I: InputExecution<NumericBackend>,
    A: eredu_codec::ear::AudioDecoderExecution<NumericBackend>,
{
    type Input = usize;
    type Output = Record;
    type Error = ExecutionError;
    fn execute(
        &mut self,
        index: &usize,
        branch: &mut Branch<I::PerceptionCache, A::NeuralState>,
    ) -> Result<(Record, Ready), ExecutionError> {
        let output = OutputPolicy {
            force_silence_on_eos: true,
            silence_on_pad: &[false],
            waveform: WaveformSilence::PreserveGenerated,
        };
        struct Policy {
            policy: ChannelDecisionPolicy,
            fresh: bool,
            owner: bool,
        }
        impl eredu_architectures::voicechat::channel_decision::DecisionResolver for Policy {
            fn preflight(&self, domain: TokenDomain) -> Result<(), eredu_core::BackendFailure> {
                assert!(self.owner, "follower cannot preflight host decision policy");
                self.policy
                    .preflight(domain)
                    .map_err(eredu_core::BackendFailure::from_error)
            }
            fn resolve(
                &self,
                input: eredu_architectures::voicechat::channel_decision::DecisionInput,
            ) -> Result<ChannelDecisionPolicy, eredu_core::BackendFailure> {
                assert!(self.owner, "follower cannot resolve host decision policy");
                assert_eq!(input.evidence.audio_frame_index, self.fresh.then_some(0));
                assert_eq!(input.evidence.first_blank.is_some(), self.fresh);
                Ok(self.policy)
            }
        }
        let live_policy = Policy {
            policy: ChannelDecisionPolicy::Main {
                agent_feedback: Some(3),
                function_feedback: Some(9),
                spoken_text: Some(3),
            },
            fresh: true,
            owner: self
                .coordinator
                .as_ref()
                .is_none_or(|coordinator| coordinator.is_owner()),
        };
        let tool_policy = Policy {
            policy: ChannelDecisionPolicy::Tool {
                function_feedback: Some(9),
                spoken_text: Some(3),
            },
            fresh: false,
            owner: self
                .coordinator
                .as_ref()
                .is_none_or(|coordinator| coordinator.is_owner()),
        };
        let invocation = match *index {
            0 => Invocation::Prefill {
                prompt: Some((
                    &[5, 7],
                    PromptPolicy {
                        repeat: 1,
                        max_tokens: 8,
                    },
                )),
                codes: self.codes,
                layout: self.layout,
                conditioning: None,
            },
            1 => Invocation::Live {
                decisions:
                    eredu_architectures::voicechat::channel_decision::DecisionSource::Resolved(
                        &live_policy,
                    ),
                output,
            },
            2 => Invocation::PerceptionOnly,
            3 => Invocation::Tool {
                audio: ToolAudio::Retained,
                decisions:
                    eredu_architectures::voicechat::channel_decision::DecisionSource::Resolved(
                        &tool_policy,
                    ),
                output,
            },
            4 => Invocation::Reminder { spoken: 3, output },
            5 => Invocation::Drain,
            7 => Invocation::ReplayAudio,
            _ => unreachable!(),
        };
        let mut coordination = self
            .coordinator
            .as_mut()
            .map(|coordinator| ExecutionCoordination::new(coordinator, [7, 11, self.attempt]));
        self.attempt += 1;
        let result = self.components.execute::<NumericBackend, _, _, _>(
            &invocation,
            branch,
            coordination.as_mut(),
            self.context,
        )?;
        assert_eq!(result.reminder_token, (*index == 4).then_some(3));
        let frame = host(&result.output.frame);
        // Numeric execution is synchronous. Observe and validate the actual host
        // payload before returning completion to either publication path.
        let view = result.output.projection.project(&frame)?;
        let record = Record {
            invocation: view.invocation_index,
            before: view.before,
            after: view.after,
            decisions: view.decisions,
            codes: view.generated_codes.unwrap_or_default().to_vec(),
            pcm: view
                .speech
                .map_or_else(Vec::new, |speech| speech.samples.to_vec()),
            start_sample: view.speech.map(|speech| speech.start_sample),
            tail: view.speech.is_some_and(|speech| speech.is_tail),
            retained: retain(branch.model_state()),
            backbone_retained_len: branch
                .model_state()
                .neural
                .backbone
                .as_ref()
                .iter()
                .flat_map(|layer| layer.retained_values())
                .map(|tensor| tensor.data.len())
                .sum(),
            projected: branch
                .model_state()
                .retained_audio
                .as_ref()
                .map(|value| value.data.clone()),
            random: branch.random_state().copied(),
        };
        Ok((record, Ready))
    }
}
fn run<R, F, I, A>(
    backbone: R,
    initial_backbone: Backbone,
    function: F,
    perception: I,
    input: InputState<NumericTensor, I::PerceptionCache>,
    codec: A,
    codec_fault: Option<&std::cell::Cell<Option<bool>>>,
    controlled: bool,
    coordinator: Option<Coordinator>,
    context: &NumericContext,
) -> Vec<Record>
where
    R: BackboneExecution<NumericBackend, Backbone>,
    F: FunctionHeadExecution<NumericBackend>,
    I: InputExecution<NumericBackend>,
    A: eredu_codec::ear::AudioDecoderExecution<NumericBackend>,
{
    run_trace(
        backbone,
        initial_backbone,
        function,
        perception,
        input,
        codec,
        codec_fault,
        controlled,
        coordinator,
        context,
        false,
    )
}
fn run_trace<R, F, I, A>(
    backbone: R,
    initial_backbone: Backbone,
    mut function: F,
    mut perception: I,
    mut input: InputState<NumericTensor, I::PerceptionCache>,
    mut codec: A,
    codec_fault: Option<&std::cell::Cell<Option<bool>>>,
    controlled: bool,
    coordinator: Option<Coordinator>,
    context: &NumericContext,
    replay: bool,
) -> Vec<Record>
where
    R: BackboneExecution<NumericBackend, Backbone>,
    F: FunctionHeadExecution<NumericBackend>,
    I: InputExecution<NumericBackend>,
    A: eredu_codec::ear::AudioDecoderExecution<NumericBackend>,
{
    let fixture = fixture();
    let fusion = ChannelFusion::from_config(&config().channels).unwrap();
    let mut tts = tts(&fixture, context);
    let controls = controls(&fixture);
    let mapping = mapping(&fixture);
    let (codes, layout) = prompt(&fixture, context);
    let empty_input = input.clone();
    let pcm: Vec<_> = (0..if replay { 3840i64 } else { 2560i64 })
        .map(|i| (((i * 73 + i * i * 19) % 2048) - 1024) as f32 / 8192.)
        .collect();
    // Deliberately non-frame-aligned ingress exercises retained perception carry.
    for chunk in pcm.chunks(157) {
        input.submit(chunk, 16000).unwrap();
    }
    input
        .finish(PcmFinishPolicy {
            final_frame: FinalPcmFrame::ZeroPad,
            trailing_silence_frames: 0,
        })
        .unwrap();
    let neural = Neural {
        backbone: initial_backbone,
        tts: caches(),
    };
    let policy = SchedulePolicy::from_config(
        &config(),
        ScheduleLimits {
            max_audio_frames: 4,
            max_backbone_positions: 64,
            max_tts_positions: 64,
            max_codec_frames: 4,
        },
    )
    .unwrap();
    let owner = coordinator
        .as_ref()
        .is_none_or(|coordinator| coordinator.is_owner());
    let coordinated = coordinator.is_some();
    let mut generation = Generation::<I::PerceptionCache, A::NeuralState>::new(
        State::new(input, neural, FeedbackTokens::new(12, 12).unwrap()),
        policy.generation_contract(),
        ScheduleState::new(policy),
        (),
        vec![AgentSampler::new(
            AgentSamplingPolicy::streaming_defaults(16).unwrap(),
        )],
        owner.then_some(0),
    )
    .unwrap();
    let components = Components {
        input: &mut perception,
        backbone,
        function: &mut function,
        fusion: &fusion,
        speech: SpeechComponents::new(&mut tts, &controls, &mut codec).unwrap(),
        characters: &mapping,
        max_character_cells: 1024,
        input_limits: PerceptionRnntLimits {
            frame_capacity: 4,
            rnnt: RnntLimits {
                max_symbols_per_frame: 2,
                hypothesis_capacity: 8,
            },
        },
        vocabulary: TokenDomain::new(24),
    };
    let mut transition = Transition {
        components,
        context,
        codes: &codes,
        layout: &layout,
        coordinator,
        attempt: 0,
    };
    let mut records = Vec::new();
    let mut projected_frames = std::collections::VecDeque::new();
    let steps: &[usize] = if replay {
        &[0, 1, 2, 2, 7, 7, 3, 4, 5]
    } else {
        &[0, 1, 2, 3, 4, 5]
    };
    for (ordinal, &step) in steps.iter().enumerate() {
        if coordinated && step == 1 {
            let snapshot = generation.branch().unwrap();
            let canonical_retained = retain(snapshot.model_state());
            let canonical_samplers = snapshot.samplers().to_vec();
            let canonical_random = snapshot.random_state().copied();
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(snapshot).unwrap();

            // Both are valid next operations separately. Different invocation
            // kinds must be refused before either rank consumes the audio frame.
            let mut mismatch = generation.branch().unwrap();
            let different_step = if owner { 1 } else { 2 };
            assert!(transition.execute(&different_step, &mut mismatch).is_err());
            assert_eq!(retain(mismatch.model_state()), canonical_retained);
            assert_eq!(mismatch.random_state().copied(), canonical_random);
            assert_eq!(mismatch.samplers(), canonical_samplers);
            mismatch.attach_submission_completion(Ready).unwrap();
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(mismatch).unwrap();

            // A follower without one ready PCM frame refuses the same live
            // request locally. The ready owner must also stop before perception.
            let mut incomplete = generation.branch().unwrap();
            if !owner {
                incomplete.model_state_mut().input = empty_input.clone();
            }
            let failure = transition.execute(&step, &mut incomplete);
            assert!(failure.is_err());
            if !owner {
                assert!(matches!(failure, Err(ExecutionError::NeedsInput)));
            }
            assert_eq!(incomplete.model_state().input.frontend().frames(), 0);
            assert_eq!(retain(incomplete.model_state()), canonical_retained);
            assert_eq!(incomplete.random_state().copied(), canonical_random);
            assert_eq!(incomplete.samplers(), canonical_samplers);
            incomplete.attach_submission_completion(Ready).unwrap();
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(incomplete).unwrap();
            assert_eq!(
                generation.model_state().input().frontend().queued_samples(),
                pcm.len()
            );
            assert_eq!(generation.revision(), 1);

            // Only the owner's character preparation fails. Every rank has
            // already executed perception/backbone and tentative channel policy;
            // neither can enter sampling/codec alone or publish that candidate.
            let mut failed = generation.branch().unwrap();
            transition.components.max_character_cells = if owner { 0 } else { 1024 };
            assert!(transition.execute(&step, &mut failed).is_err());
            assert_eq!(failed.model_state().input.frontend().frames(), 1);
            assert_eq!(failed.samplers()[0].history(), &[3]);
            assert_eq!(
                failed
                    .model_state_mut()
                    .neural
                    .backbone
                    .layer(0)
                    .unwrap()
                    .position(),
                5
            );
            failed.attach_submission_completion(Ready).unwrap();
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(failed).unwrap();
            let restored = generation.branch().unwrap();
            assert_eq!(retain(restored.model_state()), canonical_retained);
            assert_eq!(restored.random_state().copied(), canonical_random);
            assert_eq!(restored.samplers(), canonical_samplers);
            assert_eq!(restored.model_state().input.frontend().frames(), 0);
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(restored).unwrap();
            assert_eq!(generation.revision(), 1);
            transition.components.max_character_cells = 1024;
        }
        if let Some(fault) = codec_fault.filter(|_| step == 1 || step == 5) {
            let saved = generation.branch().unwrap();
            let retained = retain(saved.model_state());
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(saved).unwrap();
            let mut failed = generation.branch().unwrap();
            fault.set(Some(step == 5));
            assert!(transition.execute(&step, &mut failed).is_err());
            assert_eq!(fault.get(), None, "failure follows real codec evaluation");
            failed.attach_submission_completion(Ready).unwrap();
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(failed).unwrap();
            let restored = generation.branch().unwrap();
            assert_eq!(retain(restored.model_state()), retained);
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(restored).unwrap();
            assert_eq!(generation.revision(), ordinal as u64);
        }
        if step == 1 && controlled && !coordinated {
            let mut partial = generation.branch().unwrap();
            partial.model_state_mut().input = empty_input.clone();
            partial
                .model_state_mut()
                .input
                .submit(&pcm[..157], 16_000)
                .unwrap();
            assert!(matches!(
                transition.execute(&step, &mut partial),
                Err(ExecutionError::NeedsInput)
            ));
            assert_eq!(partial.model_state().input.frontend().queued_samples(), 157);
            assert_eq!(partial.random_state(), Some(&0));
            // Finishing makes the final partial chunk executable without
            // claiming or replacing the candidate that reported NeedsInput.
            partial
                .model_state_mut()
                .input
                .finish(PcmFinishPolicy {
                    final_frame: FinalPcmFrame::ZeroPad,
                    trailing_silence_frames: 0,
                })
                .unwrap();
            let (record, completion) = transition.execute(&step, &mut partial).unwrap();
            assert_eq!(record.after.audio_frames, 1);
            assert_eq!(record.pcm.len(), 1764);
            assert!(partial.model_state().input.frontend().is_drained());
            partial.attach_submission_completion(completion).unwrap();
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(partial).unwrap();
            assert_eq!(
                generation.model_state().input().frontend().queued_samples(),
                pcm.len()
            );
            let before = generation.branch().unwrap();
            let retained = retain(before.model_state());
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(before).unwrap();
            let mut failed = generation.branch().unwrap();
            transition.components.max_character_cells = 0;
            assert!(transition.execute(&step, &mut failed).is_err());
            assert_eq!(failed.model_state().input.frontend().frames(), 1);
            assert_eq!(generation.model_state().input().frontend().frames(), 0);
            assert!(
                transition.execute(&step, &mut failed).is_err(),
                "partially executed candidate cannot be reused"
            );
            failed.attach_submission_completion(Ready).unwrap();
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(failed).unwrap();
            let restored = generation.branch().unwrap();
            assert_eq!(retain(restored.model_state()), retained);
            assert_eq!(restored.random_state(), Some(&0));
            assert_eq!(
                restored.model_state().input.frontend().queued_samples(),
                pcm.len()
            );
            Generation::<I::PerceptionCache, A::NeuralState>::discard_branch(restored).unwrap();
            transition.components.max_character_cells = 1024;
        }
        let expected_backbone =
            if step == 7 {
                let audio = projected_frames
                    .pop_front()
                    .expect("one source frame per replay");
                let mut state = generation.model_state().neural().backbone.clone();
                let agent = NumericTensor::from_i32_slice(
                    &[if ordinal == 4 { 2 } else { 12 }],
                    &[1, 1],
                    context,
                )
                .unwrap();
                let function = NumericTensor::from_i32_slice(&[12], &[1, 1], context).unwrap();
                transition.components.backbone.execute_hidden(
                &fusion,
                eredu_architectures::voicechat::backbone_execution::BackboneRequest::Audio {
                    projected_audio: &audio,
                    agent_tokens: &agent,
                    function_tokens: &function,
                },
                &mut state,
                context,
            ).unwrap();
                Some(
                    state
                        .as_ref()
                        .iter()
                        .flat_map(|layer| layer.retained_values())
                        .flat_map(|tensor| tensor.data.iter().copied())
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
        let record = if controlled {
            let mut branch = generation.branch().unwrap();
            let (record, completion) = transition.execute(&step, &mut branch).unwrap();
            assert_eq!(generation.revision(), ordinal as u64);
            branch.attach_submission_completion(completion).unwrap();
            generation.commit_branch(branch).unwrap();
            record
        } else {
            generation
                .execute_frame_transition(&step, &mut transition)
                .unwrap()
        };
        assert_eq!(generation.schedule_state().progress(), &record.after);
        assert_eq!(
            generation
                .model_state()
                .retained_audio()
                .map(|value| &value.data),
            record.projected.as_ref(),
        );
        assert_eq!(
            generation.model_state().input().neural().frames(),
            record.after.audio_frames
        );
        if replay && step == 2 {
            projected_frames.push_back(generation.model_state().retained_audio().unwrap().clone());
            assert_eq!(
                generation.model_state().pending_audio_frames(),
                projected_frames.len()
            );
            if projected_frames.len() == 2 {
                assert_ne!(
                    projected_frames[0].data, projected_frames[1].data,
                    "ordered replay must distinguish the two acoustic frames"
                );
            }
        }
        if let Some(expected) = expected_backbone {
            assert_eq!(&record.retained[..record.backbone_retained_len], expected);
            assert_eq!(
                generation.model_state().pending_audio_frames(),
                projected_frames.len()
            );
            assert!(record.decisions.is_none());
            assert!(record.codes.is_empty());
            assert!(record.pcm.is_empty());
            assert!(record.start_sample.is_none());
            let mut expected_clocks = record.before;
            expected_clocks.backbone_positions += 1;
            assert_eq!(record.after, expected_clocks);
            assert_eq!(record.random, Some(10));
        }
        records.push(record);
    }
    assert_eq!(
        generation.samplers()[0].history(),
        if replay {
            &[2, 12, 12, 12][..]
        } else {
            &[3, 12][..]
        }
    );
    assert_eq!(generation.random_state().copied(), owner.then_some(30));
    records
}

#[test]
fn voicechat_execution_couples_components_through_shared_transactions_and_bounded_backbone() {
    let context = NumericContext::default();
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let model = || super::voicechat_backbone_execution::model(&fixture, &context);
    let controlled = run_trace(
        OrdinaryBackbone::new(&mut ResidentRuntime::new(model(), &context).unwrap()),
        super::voicechat_backbone_execution::state(),
        super::voicechat_backbone_execution::function(&fixture, &context),
        perception(&context),
        input_state(),
        codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
        None,
        true,
        None,
        &context,
        true,
    );
    let automatic = run_trace(
        OrdinaryBackbone::new(&mut ResidentRuntime::new(model(), &context).unwrap()),
        super::voicechat_backbone_execution::state(),
        super::voicechat_backbone_execution::function(&fixture, &context),
        perception(&context),
        input_state(),
        codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
        None,
        false,
        None,
        &context,
        true,
    );
    let bounded = run_trace(
        OrdinaryBackbone::new(&mut LayerwiseRuntime::new(
            model(),
            RebuildingUnitPolicy::default(),
        )),
        super::voicechat_backbone_execution::state(),
        super::voicechat_function_head_residency::bounded(&fixture),
        perception(&context),
        input_state(),
        codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
        None,
        false,
        None,
        &context,
        true,
    );
    assert_eq!(controlled, automatic);
    assert_eq!(controlled, bounded);
    let expected = [
        (0, 4, 37, 0, 0),
        (1, 5, 38, 1, 1764),
        (2, 5, 38, 1, 1764),
        (3, 5, 38, 1, 1764),
        (3, 6, 38, 1, 1764),
        (3, 7, 38, 1, 1764),
        (3, 8, 39, 2, 3528),
        (3, 8, 40, 3, 5292),
        (3, 8, 40, 3, 5300),
    ];
    for (index, (record, clocks)) in controlled.iter().zip(expected).enumerate() {
        assert_eq!(record.invocation, index as u64);
        let p = record.after;
        assert_eq!(
            (
                p.audio_frames,
                p.backbone_positions,
                p.tts_conditional_positions,
                p.codec_frames,
                p.output_samples
            ),
            clocks
        );
        assert_eq!(p.perception_frames, p.audio_frames);
        assert_eq!(p.rnnt_frames, p.audio_frames);
        assert_eq!(p.tts_unconditional_positions, p.tts_conditional_positions);
        assert!(record.retained.iter().any(|v| v.abs() > 1e-5));
    }
    assert_eq!(
        controlled
            .iter()
            .map(|r| r.start_sample)
            .collect::<Vec<_>>(),
        vec![
            None,
            Some(0),
            None,
            None,
            None,
            None,
            Some(1764),
            Some(3528),
            Some(5292)
        ]
    );
    assert_eq!(
        controlled.iter().map(|r| r.random).collect::<Vec<_>>(),
        vec![
            Some(0),
            Some(10),
            Some(10),
            Some(10),
            Some(10),
            Some(10),
            Some(20),
            Some(30),
            Some(30)
        ]
    );
    assert_eq!(controlled[1].pcm.len(), 1764);
    assert!(controlled[1].pcm.iter().any(|v| v.abs() > 1e-5));
    assert!(controlled[8].tail && controlled[8].after.codec_finished);
    assert_eq!(controlled[8].pcm.len(), 8);
}

#[test]
fn voicechat_execution_selected_tp_backbone_matches_coupled_cached_transitions() {
    let context = NumericContext::default();
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let ordinary = run(
        OrdinaryBackbone::new(
            &mut ResidentRuntime::new(
                super::voicechat_backbone_execution::model(&fixture, &context),
                &context,
            )
            .unwrap(),
        ),
        super::voicechat_backbone_execution::state(),
        super::voicechat_backbone_execution::function(&fixture, &context),
        perception(&context),
        input_state(),
        codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
        None,
        false,
        None,
        &context,
    );
    let group = NumericParallelGroup::new(2);
    let outputs =
        std::thread::scope(|scope| {
            let workers =
                (0..2)
                    .map(|rank| {
                        let fixture = &fixture;
                        let group = Arc::clone(&group);
                        scope.spawn(move || {
                let (context, mut runtime, mut function, state) =
                    super::voicechat_backbone_execution::parallel_components(fixture, rank);
                let parallel = NumericParallelContext::new(rank, group);
                // The selected providers implement the coupled contract directly;
                // Components cannot fall back to an ordinary LayeredInvocation.
                let records = run(
                    ParallelBackbone::new(&mut runtime, &parallel, None),
                    state,
                    ParallelFunctionHead::new(&mut function, &parallel),
                    perception(&context), input_state(),
                    codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
                    None, false,
        None, &context,
                );
                assert_eq!(runtime.policy().events,
                    expected_stream_events(&[(0, 0), (0, 1), (0, 2)]).repeat(3),
                    "prefill and the two backbone decode steps retain bounded layer leases");
                records
            })
                    })
                    .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
    for rank in outputs {
        assert_eq!(rank.len(), ordinary.len());
        for (actual, expected) in rank.iter().zip(&ordinary) {
            assert_eq!(
                (actual.invocation, actual.before, actual.after),
                (expected.invocation, expected.before, expected.after)
            );
            assert_eq!(actual.decisions, expected.decisions);
            assert_eq!(actual.codes, expected.codes);
            assert_eq!(
                (actual.start_sample, actual.tail, actual.random),
                (expected.start_sample, expected.tail, expected.random)
            );
            assert_eq!(actual.projected, expected.projected);
            // Attention and Mamba carry is rank-local; all subsequent shared
            // perception, guidance and codec carry still describes one stream.
            assert!(actual.backbone_retained_len < expected.backbone_retained_len);
            for (label, actual, expected) in [
                (
                    "coupled TP waveform",
                    actual.pcm.as_slice(),
                    expected.pcm.as_slice(),
                ),
                (
                    "coupled TP shared carry",
                    &actual.retained[actual.backbone_retained_len..],
                    &expected.retained[expected.backbone_retained_len..],
                ),
            ] {
                assert_tensor_close(
                    &NumericTensor::new(vec![actual.len() as i32], actual.to_vec()),
                    &NumericTensor::new(vec![expected.len() as i32], expected.to_vec()),
                    label,
                );
            }
        }
    }
}

#[test]
fn voicechat_execution_coordinated_owner_policy_failure_discard_and_retry() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let run_fixture = |coordinator: Option<Coordinator>| {
        let context = NumericContext::default();
        // One participant advances and commits explicitly while the other uses
        // the uninterrupted driver; both execute the same recorded transitions.
        let controlled = coordinator.as_ref().is_some_and(|value| value.is_owner());
        run(
            OrdinaryBackbone::new(
                &mut ResidentRuntime::new(
                    super::voicechat_backbone_execution::model(&fixture, &context),
                    &context,
                )
                .unwrap(),
            ),
            super::voicechat_backbone_execution::state(),
            super::voicechat_backbone_execution::function(&fixture, &context),
            perception(&context),
            input_state(),
            codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
            None,
            controlled,
            coordinator,
            &context,
        )
    };
    let ordinary = run_fixture(None);
    let (coordinators, timeouts) = coordinator_pair();
    let ranks = std::thread::scope(|scope| {
        let workers = coordinators
            .into_iter()
            .map(|coordinator| {
                let run_fixture = &run_fixture;
                scope.spawn(move || run_fixture(Some(coordinator)))
            })
            .collect::<Vec<_>>();
        workers
            .into_iter()
            .map(|worker| worker.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(
        ranks[0], ordinary,
        "owner replay matches ordinary execution after discarded failures"
    );
    for (mut follower, expected) in ranks.into_iter().nth(1).unwrap().into_iter().zip(&ordinary) {
        assert_eq!(follower.random, None, "followers never own or advance RNG");
        // Random ownership differs; committed samples and all retained neural
        // carry remain identical through prefill, audio, tool, reminder and drain.
        follower.random = expected.random;
        assert_eq!(&follower, expected);
    }
    assert_eq!(timeouts.load(std::sync::atomic::Ordering::SeqCst), 0);
}

/// A selected perception stage retains one local row, without remote tower caches.
#[derive(Debug, Clone, Default)]
struct LocalPerceptionCache {
    frames: u64,
    row: Option<NumericTensor>,
}
impl eredu_architectures::voicechat::perception_rnnt::PerceptionCacheState
    for LocalPerceptionCache
{
    fn frames(&self) -> u64 {
        self.frames
    }
}
impl eredu_nn::RetainedTensorRoots<NumericTensor> for LocalPerceptionCache {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a NumericTensor)) {
        if let Some(row) = &self.row {
            visitor(row);
        }
    }
}
struct SelectedInput {
    dimensions: eredu_architectures::voicechat::perception::PerceptionDimensions,
    rnnt: eredu_architectures::voicechat::rnnt::Rnnt<NumericBackend>,
}
impl InputExecution<NumericBackend> for SelectedInput {
    type PerceptionCache = LocalPerceptionCache;
    fn advance_pcm(
        &mut self,
        previous: &InputState<NumericTensor, LocalPerceptionCache>,
        limits: PerceptionRnntLimits,
        context: &NumericContext,
    ) -> Result<
        Option<
            eredu_architectures::voicechat::input::InputFrame<NumericTensor, LocalPerceptionCache>,
        >,
        eredu_architectures::voicechat::input::InputError,
    > {
        use eredu_architectures::voicechat::{
            input::{InputError, InputFrame},
            perception_rnnt::{PerceptionRnntError, PerceptionRnntState},
            rnnt::RnntExecution,
        };
        let Some(frontend) = previous.frontend().advance()? else {
            return Ok(None);
        };
        let frame = previous.neural().frames();
        if frame >= limits.frame_capacity {
            return Err(InputError::State("selected input frame capacity exhausted"));
        }
        // The provider checks the committed local history before computing its
        // next frame; discarded candidates must not replace this carry.
        assert_eq!(frontend.frame_index, frame);
        if let Some(row) = &previous.neural().perception().row {
            assert_eq!(row.data, vec![frame as f32, -0.5, 0.75, 1.0]);
        } else {
            assert_eq!(frame, 0);
        }
        let values = [frame as f32 + 1.0, -0.5, 0.75, 1.0];
        let row = NumericTensor::from_f32_slice(&values, &[1, 4], context)?;
        let projected = NumericTensor::from_f32_slice(&values, &[1, 1, 4], context)?;
        let decoded = self
            .rnnt
            .decode_frame(&row, previous.neural().rnnt(), limits.rnnt, context)
            .map_err(PerceptionRnntError::from)?;
        let neural = PerceptionRnntState::from_parts(
            (self.dimensions, self.rnnt.dimensions()),
            LocalPerceptionCache {
                frames: frame + 1,
                row: Some(row),
            },
            decoded.state,
        )?;
        Ok(Some(InputFrame {
            state: InputState::from_parts(frontend.state, neural)?,
            frame_index: frontend.frame_index,
            start_sample: frontend.start_sample,
            valid_samples: frontend.valid_samples,
            source: frontend.source,
            encoder: projected.clone(),
            projected,
            emitted: decoded.emitted,
            first_blank: decoded.first_blank,
            ended_blank: decoded.ended_blank,
            hit_symbol_limit: decoded.hit_symbol_limit,
        }))
    }
}

#[test]
fn selected_input_cache_commits_through_live_perception_and_tool_transactions() {
    use eredu_architectures::voicechat::perception_rnnt::PerceptionRnntState;
    let context = NumericContext::default();
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let execute = |controlled| {
        let (perception, rnnt) = perception(&context).into_parts();
        let selected = SelectedInput {
            dimensions: perception.dimensions(),
            rnnt,
        };
        let input = InputState::from_parts(
            input_state().frontend().clone(),
            PerceptionRnntState::from_perception(LocalPerceptionCache::default()).unwrap(),
        )
        .unwrap();
        run(
            OrdinaryBackbone::new(
                &mut ResidentRuntime::new(
                    super::voicechat_backbone_execution::model(&fixture, &context),
                    &context,
                )
                .unwrap(),
            ),
            super::voicechat_backbone_execution::state(),
            super::voicechat_backbone_execution::function(&fixture, &context),
            selected,
            input,
            codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
            None,
            controlled,
            None,
            &context,
        )
    };
    let controlled = execute(true);
    assert_eq!(controlled, execute(false));
    assert_eq!(controlled[0].projected, None);
    assert_eq!(
        controlled[1].projected.as_deref(),
        Some(&[1.0, -0.5, 0.75, 1.0][..])
    );
    for record in &controlled[2..] {
        assert_eq!(
            record.projected.as_deref(),
            Some(&[2.0, -0.5, 0.75, 1.0][..])
        );
        assert_eq!(record.after.perception_frames, 2);
        assert_eq!(record.after.rnnt_frames, 2);
    }
    assert_eq!(
        controlled[2].after.backbone_positions,
        controlled[1].after.backbone_positions
    );
    assert_eq!(
        controlled[3].after.backbone_positions,
        controlled[2].after.backbone_positions + 1
    );
    assert_eq!(controlled[1].pcm.len(), 1764);
    assert_eq!(controlled[3].pcm.len(), 1764);
}

#[derive(Clone)]
struct SelectedCodecState {
    inner: eredu_codec::ear_decoder::DecoderState<NumericTensor>,
    accepted: u64,
    finished: bool,
}
impl eredu_nn::RetainedTensorRoots<NumericTensor> for SelectedCodecState {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a NumericTensor)) {
        eredu_nn::RetainedTensorRoots::visit_retained_tensors(&self.inner, visitor);
    }
}
struct SelectedCodec {
    inner: eredu_codec::ear::AudioDecoder<NumericBackend>,
    fail: std::rc::Rc<std::cell::Cell<Option<bool>>>,
}
impl eredu_codec::ear::AudioDecoderExecution<NumericBackend> for SelectedCodec {
    type NeuralState = SelectedCodecState;
    fn config(&self) -> &eredu_codec::ear::AudioDecoderConfig {
        self.inner.config()
    }
    fn initial_state(
        &self,
        batch: i32,
    ) -> Result<
        eredu_codec::ear::AudioDecoderState<NumericTensor, SelectedCodecState>,
        eredu_codec::Error,
    > {
        Ok(self
            .inner
            .initial_state(batch)?
            .map_neural(|inner| SelectedCodecState {
                inner,
                accepted: 0,
                finished: false,
            }))
    }
    fn step(
        &mut self,
        codes: &NumericTensor,
        state: &eredu_codec::ear::AudioDecoderState<NumericTensor, SelectedCodecState>,
        constrain: bool,
        context: &NumericContext,
    ) -> Result<
        (
            NumericTensor,
            eredu_codec::ear::AudioDecoderState<NumericTensor, SelectedCodecState>,
        ),
        eredu_codec::Error,
    > {
        assert_eq!(state.neural().accepted, state.frames_seen());
        assert!(!state.neural().finished);
        let ordinary = state.clone().map_neural(|selected| selected.inner);
        let (pcm, next) = self.inner.step(codes, &ordinary, constrain, context)?;
        if self.fail.get() == Some(false) {
            self.fail.set(None);
            return Err(eredu_codec::Error::InvalidShape(
                "injected post-render failure".into(),
            ));
        }
        let accepted = next.frames_seen();
        Ok((
            pcm,
            next.map_neural(|inner| SelectedCodecState {
                inner,
                accepted,
                finished: false,
            }),
        ))
    }
    fn finish(
        &mut self,
        state: &eredu_codec::ear::AudioDecoderState<NumericTensor, SelectedCodecState>,
        constrain: bool,
        context: &NumericContext,
    ) -> Result<
        (
            Option<NumericTensor>,
            eredu_codec::ear::AudioDecoderState<NumericTensor, SelectedCodecState>,
        ),
        eredu_codec::Error,
    > {
        assert_eq!(state.neural().accepted, state.frames_seen());
        assert_eq!(state.neural().finished, state.is_finished());
        let ordinary = state.clone().map_neural(|selected| selected.inner);
        let (pcm, next) = self.inner.finish(&ordinary, constrain, context)?;
        if self.fail.get() == Some(true) {
            self.fail.set(None);
            return Err(eredu_codec::Error::InvalidShape(
                "injected post-drain failure".into(),
            ));
        }
        let accepted = next.frames_seen();
        Ok((
            pcm,
            next.map_neural(|inner| SelectedCodecState {
                inner,
                accepted,
                finished: true,
            }),
        ))
    }
}

#[test]
fn selected_codec_state_preserves_render_and_drain_candidates_until_commit() {
    let context = NumericContext::default();
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::BACKBONE_REFERENCE_JSON).unwrap();
    let execute = |controlled| {
        let fail = std::rc::Rc::new(std::cell::Cell::new(None));
        run(
            OrdinaryBackbone::new(
                &mut ResidentRuntime::new(
                    super::voicechat_backbone_execution::model(&fixture, &context),
                    &context,
                )
                .unwrap(),
            ),
            super::voicechat_backbone_execution::state(),
            super::voicechat_backbone_execution::function(&fixture, &context),
            perception(&context),
            input_state(),
            SelectedCodec {
                inner: codec(&eredu_evaluation::voicechat::coupled::fixture(), &context),
                fail: fail.clone(),
            },
            Some(&fail),
            controlled,
            None,
            &context,
        )
    };
    let controlled = execute(true);
    assert_eq!(controlled, execute(false));
    assert_eq!(controlled[1].start_sample, Some(0));
    assert_eq!(controlled[1].pcm.len(), 1764);
    assert!(controlled[1].pcm.iter().any(|value| value.abs() > 1e-5));
    assert_eq!(controlled[5].start_sample, Some(5292));
    assert_eq!(controlled[5].pcm.len(), 8);
    assert!(controlled[5].tail && controlled[5].after.codec_finished);
    assert_eq!(controlled[5].after.codec_frames, 3);
    assert_eq!(controlled[5].after.output_samples, 5300);
}

#[path = "voicechat_publication.rs"]
mod publication;
