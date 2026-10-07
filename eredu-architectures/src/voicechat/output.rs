//! VoiceChat output attribution through the shared realtime completion envelope.
//!
//! Assembly is tentative. The submission owner retains native resources, performs
//! exact completion and host validation, then publishes state and output together.
//! Token decoding, tool parsing and conversational events remain facade policy.

use eredu_nn::RetainedTensorRoots;

use eredu_core::{RealtimeChannelId, RealtimeChannelLayout, RealtimeSlotCoordinate};
use eredu_nn::{Tensor, TensorElementType};
use eredu_runtime::{CompletedRealtimeFrame, RealtimeFrameChannel, RealtimeHostChannelValues};

use super::{
    channel_decision::{ChannelDecision, FeedbackTokens},
    schedule::{
        Progress, ScheduleState, Step, StepPlan, AGENT_CHANNEL, FUNCTION_CHANNEL, OUTPUT_CHANNEL,
        RVQ_CHANNEL, TAIL_CHANNEL,
    },
};

/// Component-owned payloads for one admitted invocation. Codes are the generated
/// codes before feedback substitutions; PCM is the waveform actually rendered.
pub enum SpeechPayload<'a, T> {
    /// Prefill or perception-only execution emits no speech.
    None,
    /// One complete codec frame, with native shapes `[1, 1, 31]` and `[1, 1, samples]`.
    Frame {
        /// Generated codes before recurrent feedback substitutions.
        generated: &'a T,
        /// Waveform after the explicit rendering policy.
        pcm: &'a T,
    },
    /// Nonempty codec overlap after drain, shaped `[1, 1, tail_samples]`.
    Tail(&'a T),
}

/// Host decision identities retained alongside their native completion resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputDecisions {
    /// Sampled agent identity before semantic overrides.
    pub raw_agent: u32,
    /// Greedy function identity before semantic overrides.
    pub raw_function: u32,
    /// Effective inputs for the next backbone invocation.
    pub feedback: FeedbackTokens,
    /// Text used to condition this invocation's speech.
    pub spoken_text: u32,
}

/// Unpublished envelope, matching schedule successor and host projection.
pub struct CandidateOutput<T> {
    /// Native values retained for shared completion and host observation.
    pub frame: CompletedRealtimeFrame<T, T>,
    /// Checked successor, published only with the matching model branch.
    pub schedule: ScheduleState,
    /// Host validation and semantic attribution for exactly this invocation.
    pub projection: OutputProjection,
}

impl<T> RetainedTensorRoots<T> for CandidateOutput<T> {
    fn visit_retained_tensors<'a>(&'a self, visitor: &mut dyn FnMut(&'a T)) {
        for value in self.frame.values().iter().chain(self.frame.aligned()) {
            visitor(value.payload());
        }
        for value in self.frame.diagnostics() {
            visitor(value);
        }
    }
}

/// Exact attribution retained until the shared host observer validates the frame.
#[derive(Debug, Clone)]
pub struct OutputProjection {
    channels: RealtimeChannelLayout,
    plan: StepPlan,
    coordinate: usize,
    decisions: Option<OutputDecisions>,
}

/// Borrowed validated PCM. Sample coordinates use the output rate and do not
/// advance the input-audio frontier during tool or reminder invocations.
#[derive(Debug, Clone, Copy)]
pub struct SpeechChunk<'a> {
    /// Finite mono PCM at the declared output rate.
    pub samples: &'a [f32],
    /// Output samples per second.
    pub sample_rate: u64,
    /// Absolute emitted-sample offset, independent of input-audio time.
    pub start_sample: u64,
    /// Declared codec latency, without subtracting it from emitted samples.
    pub delay_samples: u64,
    /// This chunk contains terminal codec overlap rather than another frame.
    pub is_tail: bool,
}

/// Host values ready for the enclosing publication boundary, not yet delivered.
#[derive(Debug)]
pub struct HostOutput<'a> {
    /// Zero-based invocation identity, including prefill and tool substeps.
    pub invocation_index: u64,
    /// Architecture invocation that produced these values.
    pub step: Step,
    /// Component frontiers before the invocation.
    pub before: Progress,
    /// Validated component frontiers after the invocation.
    pub after: Progress,
    /// Both raw decisions and effective identities when the backbone decoded.
    pub decisions: Option<OutputDecisions>,
    /// Generated complete code frame, absent for non-speech work or drain.
    pub generated_codes: Option<&'a [i32]>,
    /// Timestamped waveform or terminal overlap, when produced.
    pub speech: Option<SpeechChunk<'a>>,
}

/// Packages executed components without reading native values or establishing
/// completion. `observed` comes from component frontiers, not planned counters.
/// All sparse payload requirements and geometry are checked before reshaping.
pub fn assemble<T: Tensor>(
    plan: StepPlan,
    observed: Progress,
    decisions: Option<&ChannelDecision<T>>,
    speech: SpeechPayload<'_, T>,
    context: &T::Context,
) -> Result<CandidateOutput<T>, OutputError> {
    let schedule = plan.clone().complete(observed)?;
    let channels = plan.policy().frame_contract()?.channels().clone();
    let coordinate = usize::try_from(plan.invocation_index())
        .map_err(|_| OutputError::Payload("invocation exceeds shared coordinate range"))?;
    let has_heads = matches!(plan.step(), Step::LiveAudio | Step::Tool { .. });
    if has_heads != decisions.is_some() {
        return Err(OutputError::Payload(
            "decision presence differs from invocation",
        ));
    }
    if let Some(decision) = decisions {
        for token in [
            decision.raw_agent_id,
            decision.raw_function_id,
            decision.spoken_text,
        ] {
            if token >= 131072 {
                return Err(OutputError::Payload(
                    "decision ID exceeds released vocabulary",
                ));
            }
        }
        for token in [&decision.raw_agent, &decision.raw_function] {
            if !matches!(token.shape(), [] | [1] | [1, 1] | [1, 1, 1])
                || !matches!(
                    token.element_type(),
                    Some(TensorElementType::I32 | TensorElementType::U32)
                )
            {
                return Err(OutputError::Payload("decision requires one integer token"));
            }
        }
    }
    let is_frame = plan.runs_codec();
    let is_tail = matches!(plan.step(), Step::Drain) && plan.output_samples() > 0;
    let pcm_width = i32::try_from(plan.output_samples())
        .map_err(|_| OutputError::Payload("output width exceeds tensor indexing"))?;
    match &speech {
        SpeechPayload::Frame { generated, pcm } if is_frame => {
            if generated.shape() != [1, 1, 31]
                || generated.element_type() != Some(TensorElementType::I32)
            {
                return Err(OutputError::Payload(
                    "generated codes require I32 [1, 1, 31]",
                ));
            }
            validate_pcm(*pcm, pcm_width)?;
        }
        SpeechPayload::Tail(pcm) if is_tail => validate_pcm(*pcm, pcm_width)?,
        SpeechPayload::None if !is_frame && !is_tail => {}
        _ => {
            return Err(OutputError::Payload(
                "speech payload differs from invocation",
            ))
        }
    }
    let projection = OutputProjection {
        channels: channels.clone(),
        plan,
        coordinate,
        decisions: decisions.map(|d| OutputDecisions {
            raw_agent: d.raw_agent_id,
            raw_function: d.raw_function_id,
            feedback: d.feedback,
            spoken_text: d.spoken_text,
        }),
    };
    let entry = |id, payload| {
        RealtimeFrameChannel::new(
            id,
            Some(RealtimeSlotCoordinate::new(projection.coordinate, id)),
            payload,
        )
    };
    let mut values = Vec::new();
    let mut ordered = Vec::new();
    if let Some(d) = decisions {
        values.push(entry(AGENT_CHANNEL, d.raw_agent.reshape(&[1, 1], context)?));
        values.push(entry(
            FUNCTION_CHANNEL,
            d.raw_function.reshape(&[1, 1], context)?,
        ));
        ordered.extend([AGENT_CHANNEL, FUNCTION_CHANNEL]);
    }
    match speech {
        SpeechPayload::Frame { generated, pcm } => {
            values.push(entry(RVQ_CHANNEL, generated.reshape(&[1, 31], context)?));
            values.push(entry(
                OUTPUT_CHANNEL,
                pcm.reshape(&[1, pcm_width], context)?,
            ));
        }
        SpeechPayload::Tail(pcm) => {
            values.push(entry(TAIL_CHANNEL, pcm.reshape(&[1, pcm_width], context)?));
        }
        SpeechPayload::None => {}
    }
    // RVQ is a derived complete code frame, not another categorical decision.
    let frame = CompletedRealtimeFrame::new(channels, 1, values, vec![], ordered, vec![])?;
    Ok(CandidateOutput {
        frame,
        schedule,
        projection,
    })
}

fn validate_pcm<T: Tensor>(pcm: &T, width: i32) -> Result<(), OutputError> {
    if pcm.shape() != [1, 1, width]
        || !matches!(
            pcm.element_type(),
            Some(
                TensorElementType::F16
                    | TensorElementType::Bf16
                    | TensorElementType::F32
                    | TensorElementType::F64
            )
        )
    {
        return Err(OutputError::Payload(
            "PCM requires floating [1, 1, samples]",
        ));
    }
    Ok(())
}

impl OutputProjection {
    /// Exact source and output cadence admitted for this invocation.
    pub(super) const fn policy(&self) -> &super::schedule::SchedulePolicy {
        self.plan.policy()
    }

    /// Validates the shared observer's host envelope before semantic publication.
    /// Shared validation rejects nonfinite PCM and invalid categorical domains;
    /// this projection additionally binds sparse channels and coordinates to the
    /// exact invocation and verifies native decisions against retained host IDs.
    pub fn project<'a, D>(
        &self,
        frame: &'a CompletedRealtimeFrame<RealtimeHostChannelValues, D>,
    ) -> Result<HostOutput<'a>, OutputError> {
        eredu_runtime::validate_completed_realtime_host_frame(frame)?;
        let expected = self.expected_channels();
        let ordered = if self.decisions.is_some() {
            &[AGENT_CHANNEL, FUNCTION_CHANNEL][..]
        } else {
            &[]
        };
        if frame.channels() != &self.channels
            || frame.batch() != 1
            || !frame.aligned().is_empty()
            || frame.decision_channels() != ordered
            || frame.values().len() != expected.len()
            || frame.values().iter().any(|value| {
                !expected.contains(&value.channel())
                    || value.coordinate()
                        != Some(RealtimeSlotCoordinate::new(
                            self.coordinate,
                            value.channel(),
                        ))
            })
        {
            return Err(OutputError::Payload(
                "host envelope differs from admitted invocation",
            ));
        }
        let category = |id| match frame.value(id) {
            Some(RealtimeHostChannelValues::Categorical(v)) => Ok(v.as_slice()),
            _ => Err(OutputError::Payload("missing categorical output")),
        };
        if let Some(d) = self.decisions {
            if category(AGENT_CHANNEL)? != [d.raw_agent as i32]
                || category(FUNCTION_CHANNEL)? != [d.raw_function as i32]
            {
                return Err(OutputError::Payload(
                    "native and retained decision IDs differ",
                ));
            }
        }
        let generated_codes = self
            .plan
            .runs_codec()
            .then(|| category(RVQ_CHANNEL))
            .transpose()?;
        let speech = self
            .plan
            .output_start_sample()
            .map(|start_sample| {
                let is_tail = matches!(self.plan.step(), Step::Drain);
                let id = if is_tail {
                    TAIL_CHANNEL
                } else {
                    OUTPUT_CHANNEL
                };
                let Some(RealtimeHostChannelValues::Dense(samples)) = frame.value(id) else {
                    return Err(OutputError::Payload("missing PCM output"));
                };
                Ok(SpeechChunk {
                    samples,
                    sample_rate: self.plan.policy().output_sample_rate(),
                    start_sample,
                    delay_samples: self.plan.output_delay_samples(),
                    is_tail,
                })
            })
            .transpose()?;
        Ok(HostOutput {
            invocation_index: self.plan.invocation_index(),
            step: self.plan.step(),
            before: *self.plan.before(),
            after: *self.plan.expected(),
            decisions: self.decisions,
            generated_codes,
            speech,
        })
    }

    fn expected_channels(&self) -> Vec<RealtimeChannelId> {
        let mut expected = Vec::new();
        if self.decisions.is_some() {
            expected.extend([AGENT_CHANNEL, FUNCTION_CHANNEL]);
        }
        if self.plan.runs_codec() {
            expected.extend([RVQ_CHANNEL, OUTPUT_CHANNEL]);
        }
        if matches!(self.plan.step(), Step::Drain) && self.plan.output_samples() > 0 {
            expected.push(TAIL_CHANNEL);
        }
        expected
    }
}

/// Portable assembly/projection failures; neural errors retain native sources.
#[derive(Debug, thiserror::Error)]
pub enum OutputError {
    /// Selected geometry or observed component frontiers do not match the plan.
    #[error(transparent)]
    Schedule(#[from] super::schedule::ScheduleError),
    /// Required payloads, native geometry or host attribution are inconsistent.
    #[error("invalid VoiceChat output: {0}")]
    Payload(&'static str),
    /// Native normalization failed, preserving the underlying source.
    #[error(transparent)]
    Neural(#[from] eredu_nn::Error),
    /// Shared channel envelope admission failed.
    #[error(transparent)]
    Envelope(#[from] eredu_runtime::RealtimeCompletedFrameError),
    /// Observed host values have invalid geometry, domains or finiteness.
    #[error(transparent)]
    Host(#[from] eredu_runtime::RealtimeChannelIngressError),
}
