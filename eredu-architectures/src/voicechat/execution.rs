//! Coupled component execution on the ordinary unpublished realtime branch.
//!
//! The enclosing submission scope owns native resources on success and failure.
//! It establishes exact completion and host validation before publishing this
//! candidate; these equations do not create a scheduler or execute tools.

#[path = "execution_coordination.rs"]
mod coordination;
pub use coordination::ExecutionCoordination;
use coordination::{ExecutionPhase, finish_child, gate, invocation_descriptor};

use super::function_head_layered::FunctionHeadExecution;
use super::tts_conditioning_layered::{ConditioningExecution, ResidentConditioning};
use super::tts_mog::MogHead;
use super::tts_mog_layered::MogExecution;
use super::tts_rvq::RvqRefinement;
use super::tts_rvq_layered::RvqExecution;
use super::tts_transformer::TtsTransformer;
use super::tts_transformer_layered::{CausalTransformerExecution, TransformerExecution};
use eredu_codec::ear::{AudioDecoder, AudioDecoderExecution};
use eredu_codec::ear_decoder::DecoderState;
use eredu_nn::RetainedTensorRoots;

use eredu_core::BackendFailure;
use eredu_nn::{DistributedNeuralBackend, GroupedNeuralBackend, Tensor, TensorElementType};
use eredu_runtime::{DenseSamplingBackend, RealtimeGenerationBranch, TokenDomain};

use super::{
    agent_sampling::AgentSampler,
    backbone_execution::{self, BackboneExecution, BackboneOutput, BackboneRequest},
    channel_decision::{
        self, ChannelDecision, ChannelDecisionPolicy, DecisionEvidence, DecisionInput,
        DecisionResolver, DecisionSource, FeedbackTokens,
    },
    channels::{ChannelFusion, FunctionHead, PromptPolicy},
    input::{InputExecution, InputFrame},
    output::{self, CandidateOutput, SpeechPayload},
    perception::PerceptionState,
    perception_rnnt::{PerceptionRnnt, PerceptionRnntLimits},
    progress,
    schedule::{AudioSource, ScheduleState, Step, StepPlan},
    speech::{SpeechComponents, SpeechOutput},
    state::{NeuralStateParts, StateBranch},
    tts_conditioning::CharacterMapping,
    tts_feedback::OutputPolicy,
    tts_ingress::{PromptLayout, SpeakerPrompt},
};

/// Explicit tool-step feature source. Silence is a prepared embedding, not live PCM.
pub enum ToolAudio<'a, T> {
    /// Consume exactly one queued live-audio frame.
    Fresh,
    /// Consume no live audio and use this independently prepared silence feature.
    Silence(&'a T),
    /// Use the latest committed or candidate projected perception output.
    Retained,
}

/// One architecture invocation with explicit conversational policy inputs.
pub enum Invocation<'a, T> {
    /// Startup with an optional framed agent prompt and prepared fixed-Aria codes.
    Prefill {
        /// Tokenized system prompt and its framing bounds; absent skips Nemotron.
        prompt: Option<(&'a [i32], PromptPolicy)>,
        /// Prepared speaker prompt codes.
        codes: &'a T,
        /// Exact prepared audio prompt mask layout.
        layout: &'a PromptLayout,
        /// Matching prepared text conditioning for the speaker prompt.
        conditioning: Option<&'a T>,
    },
    /// Fresh audio, one backbone invocation and one speech/codec frame.
    Live {
        /// Explicit overrides following raw selection.
        decisions: DecisionSource<'a>,
        /// Feedback and waveform silence choices.
        output: OutputPolicy<'a>,
    },
    /// Consume input during a tool wait without advancing conversation or speech.
    PerceptionOnly,
    /// Feed one retained tool-wait audio feature to the backbone without
    /// perception, sampling, speech or conversational publication.
    ReplayAudio,
    /// One function serialization or response-injection position.
    Tool {
        /// Feature source with independent live-audio attribution.
        audio: ToolAudio<'a, T>,
        /// Tool overrides, retaining PAD agent feedback.
        decisions: DecisionSource<'a>,
        /// Feedback and waveform silence choices.
        output: OutputPolicy<'a>,
    },
    /// Supplied speech without another backbone position or agent draw.
    Reminder {
        /// Token to condition through the retained character encoder.
        spoken: u32,
        /// Feedback and waveform silence choices.
        output: OutputPolicy<'a>,
    },
    /// Drain terminal codec overlap without another generated frame.
    Drain,
}

impl<T> RetainedTensorRoots<T> for Invocation<'_, T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        match self {
            Self::Prefill {
                codes,
                conditioning,
                ..
            } => {
                visitor(codes);
                if let Some(conditioning) = conditioning {
                    visitor(conditioning);
                }
            }
            Self::Tool {
                audio: ToolAudio::Silence(audio),
                ..
            } => visitor(audio),
            _ => {}
        }
    }
}

/// Borrowed execution components; their selected providers retain parameter ownership.
pub struct Components<
    'a,
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    R,
    I = PerceptionRnnt<B>,
    X = TtsTransformer<B>,
    Y = MogHead<B>,
    Z = RvqRefinement<B>,
    W = ResidentConditioning<B>,
    A = AudioDecoder<B>,
    F = FunctionHead<B>,
    E = X,
> {
    /// Shared perception and transcription encoder path.
    pub input: &'a mut I,
    /// Existing resident or bounded Nemotron runtime.
    pub backbone: R,
    /// Independently materialized function readout.
    pub function: &'a mut F,
    /// Released channel fusion equations.
    pub fusion: &'a ChannelFusion,
    /// Existing character/TTS/codec composition.
    pub speech: SpeechComponents<'a, B, X, Y, Z, W, A, E>,
    /// Prepared character assets, without artifact access during execution.
    pub characters: &'a CharacterMapping,
    /// Bound for one spoken token's character encoding.
    pub max_character_cells: usize,
    /// Explicit perception and retained transcription bounds.
    pub input_limits: PerceptionRnntLimits,
    /// Admitted common agent/function head domain.
    pub vocabulary: TokenDomain,
}

/// Native output resources and host metadata retained through exact completion.
pub struct ExecutionOutput<T: Tensor, C = PerceptionState<T>, K = DecoderState<T>> {
    /// Sparse output envelope and matching host validation/projection.
    pub output: CandidateOutput<T>,
    /// Actual supplied reminder token, absent for other invocation kinds.
    pub reminder_token: Option<u32>,
    /// New transcription and input attribution, absent for cached-audio steps.
    pub input: Option<InputFrame<T, C>>,
    /// Normalized prompt or replay state, without evaluating vocabulary heads.
    pub prompt_hidden: Option<T>,
    /// Shared normalized hidden and both heads, when Nemotron decoded.
    pub backbone: Option<BackboneOutput<T>>,
    /// Raw decision resources plus effective feedback and spoken identity.
    pub decisions: Option<ChannelDecision<T>>,
    /// Rendered speech and candidate codec/feedback carry.
    pub speech: Option<SpeechOutput<T, K>>,
    /// Terminal spectral overlap, when nonempty.
    pub tail: Option<T>,
}

impl<T: Tensor, C: RetainedTensorRoots<T>, K: RetainedTensorRoots<T>> RetainedTensorRoots<T>
    for ExecutionOutput<T, C, K>
{
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        self.output.visit_retained_tensors(visitor);
        if let Some(input) = &self.input {
            input.visit_retained_tensors(visitor);
        }
        if let Some(hidden) = &self.prompt_hidden {
            visitor(hidden);
        }
        if let Some(backbone) = &self.backbone {
            backbone.visit_retained_tensors(visitor);
        }
        if let Some(decisions) = &self.decisions {
            decisions.visit_retained_tensors(visitor);
        }
        if let Some(speech) = &self.speech {
            speech.visit_retained_tensors(visitor);
        }
        if let Some(tail) = &self.tail {
            visitor(tail);
        }
    }
}

impl<
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    R,
    I: InputExecution<B>,
    X: TransformerExecution<B>,
    Y: MogExecution<B>,
    Z: RvqExecution<B>,
    W: ConditioningExecution<B>,
    A: AudioDecoderExecution<B>,
    F: FunctionHeadExecution<B>,
    E: TransformerExecution<B>,
> Components<'_, B, R, I, X, Y, Z, W, A, F, E>
{
    /// Checks invocation, input readiness and component frontiers without
    /// claiming or advancing the candidate or submitting neural work.
    /// Native owners call this before opening their submission scope; execution
    /// repeats these same checks before claiming the candidate. Mutable access
    /// only borrows cache geometry through the existing joint-state contract.
    /// Coordinated callers enter `execute` on every participant even if local
    /// preflight fails; this local check does not perform admission agreement.
    pub fn preflight<D, N, P, C>(
        &self,
        invocation: &Invocation<'_, B::Tensor>,
        branch: &mut RealtimeGenerationBranch<
            StateBranch<B::Tensor, N, I::PerceptionCache, A::NeuralState>,
            ScheduleState,
            P,
            AgentSampler,
            D::RandomState,
            C,
        >,
        coordination: Option<&ExecutionCoordination<'_, B::Tensor>>,
    ) -> Result<StepPlan, ExecutionError>
    where
        D: DenseSamplingBackend<
                Logits = B::Tensor,
                Token = B::Tensor,
                Context = <B::Tensor as Tensor>::Context,
            >,
        D::RandomState: Clone,
        D::Error: std::error::Error + Send + Sync + 'static,
        N: NeuralStateParts,
        X: CausalTransformerExecution<B, N::TtsState>,
        R: BackboneExecution<B, N::Backbone>,
        R::Error: std::error::Error + Send + Sync + 'static,
    {
        if branch.has_submission_completion() {
            return Err(ExecutionError::Preflight(
                "branch already has submission completion",
            ));
        }
        let owner = coordination.is_none_or(ExecutionCoordination::is_owner);
        let step = match invocation {
            Invocation::Prefill { prompt, .. } => Step::Prefill {
                backbone_tokens: prompt
                    .map(|(tokens, policy)| policy.framed_length(tokens))
                    .transpose()?
                    .unwrap_or(0) as u64,
            },
            Invocation::Live { decisions, .. } => {
                if owner
                    && matches!(
                        decisions,
                        DecisionSource::Fixed(ChannelDecisionPolicy::Tool { .. })
                    )
                {
                    return Err(ExecutionError::Preflight(
                        "live invocation requires main decisions",
                    ));
                }
                Step::LiveAudio
            }
            Invocation::PerceptionOnly => Step::PerceptionOnly,
            Invocation::ReplayAudio => Step::ReplayAudio,
            Invocation::Tool {
                audio, decisions, ..
            } => {
                if owner
                    && matches!(
                        decisions,
                        DecisionSource::Fixed(ChannelDecisionPolicy::Main { .. })
                    )
                {
                    return Err(ExecutionError::Preflight(
                        "tool invocation requires tool decisions",
                    ));
                }
                Step::Tool {
                    audio: match audio {
                        ToolAudio::Fresh => AudioSource::Fresh,
                        ToolAudio::Silence(_) => AudioSource::Silence,
                        ToolAudio::Retained => AudioSource::Retained,
                    },
                }
            }
            Invocation::Reminder { spoken, .. } => {
                if *spoken as usize >= self.vocabulary.cardinality() {
                    return Err(ExecutionError::Preflight(
                        "reminder token exceeds vocabulary",
                    ));
                }
                Step::Reminder
            }
            Invocation::Drain => Step::Drain,
        };
        let plan = branch.schedule_state().plan(step)?;
        if plan.runs_perception() && !branch.model_state().input.frontend().has_ready_frame() {
            return Err(ExecutionError::NeedsInput);
        }
        if plan.runs_codec() && owner && branch.random_state().is_none() {
            return Err(super::speech_sampling::SpeechSamplingError::MissingRandomState.into());
        }
        if matches!(
            step,
            Step::LiveAudio | Step::Tool { .. } | Step::ReplayAudio
        ) {
            if branch.samplers().len() != 1 {
                return Err(ExecutionError::Preflight(
                    "coupled execution requires one agent sampler",
                ));
            }
            branch.samplers()[0]
                .preflight()
                .map_err(|e| ExecutionError::Decision(BackendFailure::from_error(e)))?;
            if branch.samplers()[0]
                .history()
                .iter()
                .any(|&id| id as usize >= self.vocabulary.cardinality())
            {
                return Err(ExecutionError::Preflight(
                    "retained agent history exceeds vocabulary",
                ));
            }
        }
        if matches!(step, Step::ReplayAudio) && branch.model_state().pending_audio_frames() == 0 {
            return Err(ExecutionError::Preflight("no queued audio for replay"));
        }
        if let Invocation::Prefill {
            layout,
            codes,
            prompt,
            ..
        } = invocation
        {
            if layout.batch() != 1
                || layout.frames() as u64 != plan.policy().tts_prefill_positions()
                || codes.shape() != [1, layout.frames(), 31]
                || codes.element_type() != Some(TensorElementType::I32)
            {
                return Err(ExecutionError::Preflight(
                    "speaker prompt differs from admitted startup",
                ));
            }
            if prompt.is_some_and(|(tokens, _)| {
                tokens
                    .iter()
                    .any(|&id| id as usize >= self.vocabulary.cardinality())
            }) {
                return Err(ExecutionError::Preflight("prompt token exceeds vocabulary"));
            }
        }
        if let Invocation::Live { decisions, output }
        | Invocation::Tool {
            decisions, output, ..
        } = invocation
        {
            if owner {
                decisions
                    .preflight(self.vocabulary)
                    .map_err(ExecutionError::Decision)?;
            }
            if output.silence_on_pad.len() != 1 {
                return Err(ExecutionError::Preflight(
                    "speech policy requires one stream",
                ));
            }
        }
        if let Invocation::Reminder { output, .. } = invocation {
            if output.silence_on_pad.len() != 1 {
                return Err(ExecutionError::Preflight(
                    "speech policy requires one stream",
                ));
            }
        }
        {
            let state = branch.model_state_mut();
            let (backbone, tts) = state.neural.parts_mut();
            let before = progress::observe::<B, _, _, _, _>(
                &state.input,
                &self.backbone,
                backbone,
                &self.speech.generator.transformer,
                tts,
                state.speech.as_ref(),
            )?;
            if &before != plan.before() {
                return Err(ExecutionError::Preflight(
                    "component frontiers differ from schedule",
                ));
            }
        }
        Ok(plan)
    }

    /// Executes exactly one invocation. Ordinary and controlled drivers call this
    /// same transition on their shared branch. Errors after execution starts
    /// require completion and branch discard; retry uses a fresh canonical branch.
    /// Optional coordination selects one policy/sampling owner and agrees component
    /// boundaries. Replicated inputs and prepared parameters must already match;
    /// providers with internal collectives must guard their own local failures.
    pub fn execute<D, N, P, C>(
        &mut self,
        invocation: &Invocation<'_, B::Tensor>,
        branch: &mut RealtimeGenerationBranch<
            StateBranch<B::Tensor, N, I::PerceptionCache, A::NeuralState>,
            ScheduleState,
            P,
            AgentSampler,
            D::RandomState,
            C,
        >,
        mut coordination: Option<&mut ExecutionCoordination<'_, B::Tensor>>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<ExecutionOutput<B::Tensor, I::PerceptionCache, A::NeuralState>, ExecutionError>
    where
        D: DenseSamplingBackend<
                Logits = B::Tensor,
                Token = B::Tensor,
                Context = <B::Tensor as Tensor>::Context,
            >,
        D::RandomState: Clone,
        D::Error: std::error::Error + Send + Sync + 'static,
        N: NeuralStateParts,
        X: CausalTransformerExecution<B, N::TtsState>,
        R: BackboneExecution<B, N::Backbone>,
        R::Error: std::error::Error + Send + Sync + 'static,
    {
        let local = self.preflight::<D, N, P, C>(invocation, branch, coordination.as_deref());
        if let Some(authority) = coordination.as_deref_mut() {
            let words = local
                .as_ref()
                .map(|plan| {
                    let mut words = invocation_descriptor(
                        invocation,
                        plan,
                        branch.model_state().feedback,
                        self.vocabulary,
                        self.input_limits,
                    );
                    words.push(branch.model_state().pending_audio_frames() as u64);
                    words.push(u64::from(branch.model_state().pending_audio_replaying()));
                    words
                })
                .unwrap_or_default();
            let agreement = authority.admit(&words, local.is_ok(), context);
            // A failing participant keeps its original portable/native source.
            if let Err(error) = local {
                return Err(error);
            }
            agreement?;
        }
        let plan = local?;
        let claimed = if branch.model_state_mut().begin_execution() {
            Ok(())
        } else {
            Err(ExecutionError::Preflight(
                "candidate invocation already started",
            ))
        };
        gate(&mut coordination, ExecutionPhase::Claim, claimed, context)?;
        let mut input_output = None;
        let mut prompt_hidden = None;
        let mut backbone_output = None;
        let mut decisions_output = None;
        let mut speech_output = None;
        let mut tail = None;
        if let Invocation::Prefill {
            prompt,
            codes,
            layout,
            conditioning,
        } = invocation
        {
            let state = branch.model_state_mut();
            let (backbone, caches) = state.neural.parts_mut();
            if let Some((tokens, policy)) = prompt {
                let hidden = self
                    .backbone
                    .execute_hidden(
                        self.fusion,
                        BackboneRequest::Prompt {
                            tokens,
                            policy: *policy,
                        },
                        backbone,
                        context,
                    )
                    .map_err(|e| ExecutionError::Backbone(BackendFailure::from_error(e)));
                prompt_hidden = Some(gate(
                    &mut coordination,
                    ExecutionPhase::PromptBackbone,
                    hidden,
                    context,
                )?);
            }
            let speech = self
                .speech
                .prefill(
                    codes,
                    layout,
                    SpeakerPrompt::Aria,
                    *conditioning,
                    caches,
                    context,
                )
                .map_err(ExecutionError::from);
            state.speech = Some(gate(
                &mut coordination,
                ExecutionPhase::SpeechPrefill,
                speech,
                context,
            )?);
        } else {
            if plan.runs_perception() {
                let frame = self
                    .input
                    .advance_pcm(&branch.model_state().input, self.input_limits, context)
                    .map_err(ExecutionError::from)
                    .and_then(|frame| frame.ok_or(ExecutionError::NeedsInput));
                let frame = gate(&mut coordination, ExecutionPhase::Input, frame, context)?;
                let retained = (|| -> Result<(), ExecutionError> {
                    let state = branch.model_state_mut();
                    if matches!(invocation, Invocation::PerceptionOnly) {
                        state.retain_pending_audio(
                            frame.projected.clone(),
                            plan.policy().limits().max_audio_frames,
                        )?;
                    }
                    state.input = frame.state.clone();
                    state.retained_audio = Some(frame.projected.clone());
                    Ok(())
                })();
                gate(
                    &mut coordination,
                    ExecutionPhase::InputRetention,
                    retained,
                    context,
                )?;
                input_output = Some(frame);
            }
            if matches!(invocation, Invocation::Live { .. }) {
                // A normal tool completion resumes live input without replay.
                // After interruption the facade drains this queue first.
                branch.model_state_mut().clear_pending_audio();
            }
            if matches!(invocation, Invocation::ReplayAudio) {
                if branch.model_state_mut().begin_pending_audio_replay() {
                    super::state::InterruptionFeedback::reset_interruption_feedback(branch);
                }
                let state = branch.model_state_mut();
                let prepared = (|| -> Result<_, ExecutionError> {
                    let audio = state.replay_pending_audio()?;
                    let feedback = state.feedback.materialize::<B::Tensor>(context)?;
                    Ok((audio, feedback))
                })();
                let (audio, feedback) = gate(
                    &mut coordination,
                    ExecutionPhase::BackboneIngress,
                    prepared,
                    context,
                )?;
                let (backbone, _) = state.neural.parts_mut();
                let hidden = self
                    .backbone
                    .execute_hidden(
                        self.fusion,
                        BackboneRequest::Audio {
                            projected_audio: &audio,
                            agent_tokens: &feedback.agent,
                            function_tokens: &feedback.function,
                        },
                        backbone,
                        context,
                    )
                    .map_err(|e| ExecutionError::Backbone(BackendFailure::from_error(e)));
                prompt_hidden = Some(gate(
                    &mut coordination,
                    ExecutionPhase::Backbone,
                    hidden,
                    context,
                )?);
                // These are effective silent feedback positions, not sampled
                // decisions. Preserve RNG and the sparse publication envelope.
                let pad = super::channels::TEXT_PAD_ID as u32;
                branch.model_state_mut().feedback = FeedbackTokens::new(pad, pad)
                    .map_err(|e| ExecutionError::Decision(BackendFailure::from_error(e)))?;
                branch.samplers_mut()[0]
                    .accept_effective(pad)
                    .map_err(|e| ExecutionError::Decision(BackendFailure::from_error(e)))?;
            }
            if let Invocation::Live { decisions, .. } | Invocation::Tool { decisions, .. } =
                invocation
            {
                let state = branch.model_state_mut();
                let prepared = (|| -> Result<_, ExecutionError> {
                    let audio = match invocation {
                        Invocation::Tool {
                            audio: ToolAudio::Silence(audio),
                            ..
                        } => (*audio).clone(),
                        _ => state
                            .retained_audio
                            .clone()
                            .ok_or(ExecutionError::Preflight("no retained projected audio"))?,
                    };
                    let feedback = if let Some([agent, function]) = plan.initial_feedback_tokens() {
                        FeedbackTokens::new(agent as u32, function as u32)
                            .map_err(|e| ExecutionError::Decision(BackendFailure::from_error(e)))?
                    } else {
                        state.feedback
                    };
                    let feedback = feedback.materialize::<B::Tensor>(context)?;
                    Ok((audio, feedback))
                })();
                let (audio, feedback) = gate(
                    &mut coordination,
                    ExecutionPhase::BackboneIngress,
                    prepared,
                    context,
                )?;
                let (backbone, _) = state.neural.parts_mut();
                let result = backbone_execution::execute::<B, _, _, _>(
                    &mut self.backbone,
                    self.function,
                    self.fusion,
                    BackboneRequest::Audio {
                        projected_audio: &audio,
                        agent_tokens: &feedback.agent,
                        function_tokens: &feedback.function,
                    },
                    backbone,
                    context,
                )
                .map_err(|e| ExecutionError::Backbone(BackendFailure::from_error(e)));
                let result = gate(&mut coordination, ExecutionPhase::Backbone, result, context)?;
                let resolver = InvocationDecisionResolver {
                    source: *decisions,
                    tool: matches!(invocation, Invocation::Tool { .. }),
                };
                let evidence =
                    input_output
                        .as_ref()
                        .map_or_else(DecisionEvidence::default, |input| DecisionEvidence {
                            first_blank: Some(input.first_blank),
                            audio_frame_index: Some(input.frame_index),
                        });
                let mut channel_authority = coordination
                    .as_deref_mut()
                    .map(|c| c.channels())
                    .transpose()?;
                let decision = channel_decision::select_channels::<D, _, _, _, _>(
                    branch,
                    &result.logits,
                    DecisionSource::Resolved(&resolver),
                    evidence,
                    self.vocabulary,
                    channel_authority
                        .as_mut()
                        .map(|c| c as &mut dyn channel_decision::ChannelDecisionCoordination<D>),
                    context,
                )
                .map_err(|e| ExecutionError::Decision(BackendFailure::from_error(e)));
                drop(channel_authority);
                let decision = finish_child(&mut coordination, decision)?;
                branch.model_state_mut().feedback = decision.feedback;
                decisions_output = Some(decision);
                backbone_output = Some(result);
            }
            let spoken = match invocation {
                Invocation::Live { output, .. } | Invocation::Tool { output, .. } => Some((
                    decisions_output
                        .as_ref()
                        .expect("backbone decision executed")
                        .spoken_text,
                    *output,
                )),
                Invocation::Reminder { spoken, output } => Some((*spoken, *output)),
                _ => None,
            };
            if let Some((spoken, policy)) = spoken {
                let frame = (|| -> Result<_, ExecutionError> {
                    let state = branch.model_state_mut();
                    let previous = state
                        .speech
                        .as_ref()
                        .ok_or(ExecutionError::Preflight("speech prefill is absent"))?;
                    let (_, caches) = state.neural.parts_mut();
                    Ok(self.speech.begin_text_frame(
                        previous,
                        &[spoken as i32],
                        self.characters,
                        self.max_character_cells,
                        policy,
                        caches,
                        context,
                    )?)
                })();
                let frame = gate(
                    &mut coordination,
                    ExecutionPhase::Conditioning,
                    frame,
                    context,
                )?;
                let mut speech_authority = coordination
                    .as_deref_mut()
                    .map(|c| c.speech())
                    .transpose()?;
                let rendered = self
                    .speech
                    .render_sampled_frame::<D, _, _, _, _, _>(
                        &frame,
                        branch,
                        speech_authority.as_mut(),
                        context,
                    )
                    .map_err(ExecutionError::from);
                drop(speech_authority);
                let rendered = finish_child(&mut coordination, rendered)?;
                branch.model_state_mut().speech = Some(rendered.state.clone());
                speech_output = Some(rendered);
            }
            if matches!(invocation, Invocation::Drain) {
                let state = branch.model_state_mut();
                let finished = (|| -> Result<_, ExecutionError> {
                    let previous = state
                        .speech
                        .as_ref()
                        .ok_or(ExecutionError::Preflight("speech prefill is absent"))?;
                    Ok(self.speech.finish(previous, context)?)
                })();
                let (pcm, next) =
                    gate(&mut coordination, ExecutionPhase::Drain, finished, context)?;
                state.speech = Some(next);
                tail = pcm;
            }
        }
        let assembled = (|| -> Result<_, ExecutionError> {
            let state = branch.model_state_mut();
            let (backbone, caches) = state.neural.parts_mut();
            let observed = progress::observe::<B, _, _, _, _>(
                &state.input,
                &self.backbone,
                backbone,
                &self.speech.generator.transformer,
                caches,
                state.speech.as_ref(),
            )?;
            let payload = if let Some(speech) = &speech_output {
                SpeechPayload::Frame {
                    generated: &speech.generated,
                    pcm: &speech.pcm,
                }
            } else if let Some(tail) = &tail {
                SpeechPayload::Tail(tail)
            } else {
                SpeechPayload::None
            };
            Ok(output::assemble(
                plan,
                observed,
                decisions_output.as_ref(),
                payload,
                context,
            )?)
        })();
        let output = gate(
            &mut coordination,
            ExecutionPhase::Output,
            assembled,
            context,
        )?;
        *branch.schedule_state_mut() = output.schedule.clone();
        Ok(ExecutionOutput {
            output,
            reminder_token: match invocation {
                Invocation::Reminder { spoken, .. } => Some(*spoken),
                _ => None,
            },
            prompt_hidden,
            input: input_output,
            backbone: backbone_output,
            decisions: decisions_output,
            speech: speech_output,
            tail,
        })
    }
}

// Resolved overrides obey the invocation's channel role before the shared
// sampler adopts its tentative RNG or effective history.
struct InvocationDecisionResolver<'a> {
    source: DecisionSource<'a>,
    tool: bool,
}

impl DecisionResolver for InvocationDecisionResolver<'_> {
    fn preflight(&self, domain: TokenDomain) -> Result<(), BackendFailure> {
        self.source.preflight(domain)
    }

    fn resolve(&self, input: DecisionInput) -> Result<ChannelDecisionPolicy, BackendFailure> {
        let policy = self.source.resolve(input)?;
        if matches!(policy, ChannelDecisionPolicy::Tool { .. }) != self.tool {
            return Err(BackendFailure::from_error(ExecutionError::Preflight(
                "resolved channel policy differs from invocation",
            )));
        }
        Ok(policy)
    }
}

/// Portable invocation or component failure, preserving native error sources.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    /// The candidate, invocation or retained component identities disagree.
    #[error("invalid VoiceChat execution: {0}")]
    Preflight(&'static str),
    /// A fresh-audio invocation has no complete queued frame.
    #[error("VoiceChat execution requires another complete input frame")]
    NeedsInput,
    /// Explicit scheduling or component capacity failed.
    #[error(transparent)]
    Schedule(#[from] super::schedule::ScheduleError),
    /// Observed cache frontiers are inconsistent.
    #[error(transparent)]
    Progress(#[from] progress::ProgressError),
    /// Host input, perception or transcription failed.
    #[error(transparent)]
    Input(#[from] super::input::InputError),
    /// Deferred projected audio cannot satisfy its retained capacity or replay.
    #[error(transparent)]
    PendingAudio(#[from] super::state::PendingAudioError),
    /// Prompt or channel fusion failed.
    #[error(transparent)]
    Channel(#[from] super::channels::ChannelError),
    /// Existing layered execution failed, retaining its original source.
    #[error("VoiceChat backbone execution failed")]
    Backbone(#[source] BackendFailure),
    /// Channel selection or feedback admission failed.
    #[error("VoiceChat channel decision failed")]
    Decision(#[source] BackendFailure),
    /// Character, causal speech or codec execution failed.
    #[error(transparent)]
    Speech(#[from] super::tts_mog::TtsError),
    /// Dense speech sampling failed.
    #[error(transparent)]
    Sampling(#[from] super::speech_sampling::SpeechSamplingError),
    /// Output geometry or receipt failed before publication.
    #[error(transparent)]
    Output(#[from] output::OutputError),
    /// Coupled participants disagree or selected communication failed.
    #[error("VoiceChat execution coordination failed")]
    Coordination(#[source] BackendFailure),
    /// Neutral tensor creation failed, retaining its native source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
}
