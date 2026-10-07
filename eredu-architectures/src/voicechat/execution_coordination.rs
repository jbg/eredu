//! One-shot agreement authority for the ordinary coupled component transition.

use eredu_nn::Tensor;
use eredu_runtime::replicated_execution::ReplicatedCoordinator;

use super::super::{
    channel_decision::ReplicatedChannelDecisions, speech_sampling::SpeechSamplingCoordination,
};
use super::ExecutionError;

const PROTOCOL: u64 = 0x564f494345455802;

/// Exact component boundaries within one coupled invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u64)]
pub(super) enum ExecutionPhase {
    Admission = 0,
    Claim = 1,
    PromptBackbone = 2,
    SpeechPrefill = 3,
    Input = 4,
    InputRetention = 12,
    BackboneIngress = 5,
    Backbone = 6,
    Conditioning = 8,
    Drain = 10,
    Output = 11,
}

/// One coupled invocation over retained bounded communication resources.
///
/// Request, turn and invocation identity are explicit recorded inputs. Admission
/// consumes this authority even when a participant fails. Retry uses a fresh
/// authority and unpublished branch. The authority owns neither model state nor
/// a scheduler and never establishes native completion or commits output.
/// Component providers retain responsibility for coordinating their internal
/// collectives before returning to the enclosing component boundary.
pub struct ExecutionCoordination<'a, T: Tensor> {
    coordinator: &'a mut dyn ReplicatedCoordinator<T, T::Context>,
    identity: [u64; 3],
    consumed: bool,
    admitted: bool,
    fenced: bool,
    channels_issued: bool,
    speech_issued: bool,
}

impl<'a, T: Tensor> ExecutionCoordination<'a, T> {
    /// Borrows selected communication and records the exact invocation identity.
    pub fn new(
        coordinator: &'a mut dyn ReplicatedCoordinator<T, T::Context>,
        identity: [u64; 3],
    ) -> Self {
        Self {
            coordinator,
            identity,
            consumed: false,
            admitted: false,
            fenced: false,
            channels_issued: false,
            speech_issued: false,
        }
    }

    /// Whether this participant alone resolves conversational policy and samples.
    pub fn is_owner(&self) -> bool {
        self.coordinator.is_owner()
    }

    /// Whether this authority has entered its single permitted invocation.
    pub fn is_consumed(&self) -> bool {
        self.consumed
    }

    /// Consumes authority and agrees exact invocation metadata before learned work.
    pub(super) fn admit(
        &mut self,
        words: &[u64],
        success: bool,
        context: &T::Context,
    ) -> Result<(), ExecutionError> {
        let fresh = !self.consumed;
        self.consumed = true;
        self.exchange(ExecutionPhase::Admission, words, success && fresh, context)?;
        self.admitted = true;
        Ok(())
    }

    /// Agrees local completion before another component or branch adoption.
    /// Local typed failures and their native sources take precedence over a peer
    /// failure or an additional failure of agreement itself.
    pub(super) fn gate<V>(
        &mut self,
        phase: ExecutionPhase,
        local: Result<V, ExecutionError>,
        context: &T::Context,
    ) -> Result<V, ExecutionError> {
        let agreement = self.exchange(phase, &[], self.admitted && local.is_ok(), context);
        let value = local?;
        agreement?;
        Ok(value)
    }

    /// Adopts an inner driver's already-agreed outcome without a recovery collective.
    /// An inner transport failure fences this enclosing authority immediately;
    /// its completion owner retains responsibility for failed native work.
    pub(super) fn finish_child<V>(
        &mut self,
        local: Result<V, ExecutionError>,
    ) -> Result<V, ExecutionError> {
        if local.is_err() {
            self.fenced = true;
        }
        local
    }

    /// Borrows the existing owner-only channel selection authority once.
    pub(super) fn channels(&mut self) -> Result<ReplicatedChannelDecisions<'_, T>, ExecutionError> {
        self.require_active()?;
        if self.channels_issued {
            return Err(ExecutionError::Preflight(
                "coupled invocation already issued channel authority",
            ));
        }
        self.channels_issued = true;
        Ok(ReplicatedChannelDecisions::new(
            self.coordinator,
            self.identity,
        ))
    }

    /// Borrows the existing owner-only speech sampling authority once.
    pub(super) fn speech(&mut self) -> Result<SpeechSamplingCoordination<'_, T>, ExecutionError> {
        self.require_active()?;
        if self.speech_issued {
            return Err(ExecutionError::Preflight(
                "coupled invocation already issued speech authority",
            ));
        }
        self.speech_issued = true;
        Ok(SpeechSamplingCoordination::new(
            self.coordinator,
            self.identity,
        ))
    }

    fn require_active(&self) -> Result<(), ExecutionError> {
        if !self.admitted || self.fenced {
            return Err(ExecutionError::Preflight(
                "coupled execution authority is not active",
            ));
        }
        Ok(())
    }

    fn exchange(
        &mut self,
        phase: ExecutionPhase,
        words: &[u64],
        success: bool,
        context: &T::Context,
    ) -> Result<(), ExecutionError> {
        if self.fenced {
            return Err(ExecutionError::Preflight(
                "coupled execution authority is fenced",
            ));
        }
        let result = (|| {
            // Agree variable metadata length before bounded word exchanges so a
            // mismatch cannot create different numbers of collective operations.
            if !self
                .coordinator
                .agree(
                    [
                        PROTOCOL,
                        self.identity[0],
                        self.identity[1],
                        self.identity[2],
                        phase as u64,
                        words.len() as u64,
                        0,
                        0,
                    ],
                    success,
                    context,
                )
                .map_err(ExecutionError::Coordination)?
            {
                return Err(ExecutionError::Preflight(
                    "coupled execution participants disagree or a participant failed",
                ));
            }
            for (index, chunk) in words.chunks(6).enumerate() {
                let mut descriptor = [u64::MAX; 8];
                descriptor[0] = PROTOCOL;
                descriptor[1] = index as u64;
                descriptor[2..2 + chunk.len()].copy_from_slice(chunk);
                if !self
                    .coordinator
                    .agree(descriptor, true, context)
                    .map_err(ExecutionError::Coordination)?
                {
                    return Err(ExecutionError::Preflight(
                        "coupled execution metadata differs between participants",
                    ));
                }
            }
            Ok(())
        })();
        if result.is_err() {
            self.fenced = true;
        }
        result
    }
}

pub(super) fn gate<T: Tensor, V>(
    coordination: &mut Option<&mut ExecutionCoordination<'_, T>>,
    phase: ExecutionPhase,
    local: Result<V, ExecutionError>,
    context: &T::Context,
) -> Result<V, ExecutionError> {
    match coordination.as_deref_mut() {
        Some(authority) => authority.gate(phase, local, context),
        None => local,
    }
}

pub(super) fn finish_child<T: Tensor, V>(
    coordination: &mut Option<&mut ExecutionCoordination<'_, T>>,
    local: Result<V, ExecutionError>,
) -> Result<V, ExecutionError> {
    match coordination.as_deref_mut() {
        Some(authority) => authority.finish_child(local),
        None => local,
    }
}

/// Exact host invocation semantics; tensor values retain their admitted source
/// identity and are not copied to host or hashed during inference.
pub(super) fn invocation_descriptor<T: Tensor>(
    invocation: &super::Invocation<'_, T>,
    plan: &super::super::schedule::StepPlan,
    feedback: super::super::channel_decision::FeedbackTokens,
    vocabulary: eredu_runtime::TokenDomain,
    input_limits: super::super::perception_rnnt::PerceptionRnntLimits,
) -> Vec<u64> {
    use super::{Invocation, ToolAudio};
    use eredu_runtime::RealtimeScheduleDescriptor;
    let policy = plan.policy().schedule_descriptor();
    let mut words = vec![policy.len() as u64];
    words.extend(policy.into_iter().map(u64::from));
    words.extend([
        plan.invocation_index(),
        vocabulary.cardinality() as u64,
        feedback.agent() as u64,
        feedback.function() as u64,
        u64::from(plan.uses_initial_feedback()),
        input_limits.frame_capacity,
        input_limits.rnnt.max_symbols_per_frame as u64,
        input_limits.rnnt.hypothesis_capacity as u64,
    ]);
    for progress in [plan.before(), plan.expected()] {
        words.extend([
            progress.audio_frames,
            progress.perception_frames,
            progress.rnnt_frames,
            progress.backbone_positions,
            progress.tts_conditional_positions,
            progress.tts_unconditional_positions,
            progress.codec_frames,
            progress.output_samples,
            u64::from(progress.codec_finished),
        ]);
    }
    match invocation {
        Invocation::Prefill {
            prompt,
            codes,
            layout,
            conditioning,
        } => {
            words.extend([0, u64::from(prompt.is_some())]);
            if let Some((tokens, policy)) = prompt {
                words.extend([
                    policy.repeat as u64,
                    policy.max_tokens as u64,
                    tokens.len() as u64,
                ]);
                words.extend(tokens.iter().map(|token| *token as u64));
            }
            tensor_descriptor(&mut words, *codes);
            words.extend([layout.batch() as u64, layout.frames() as u64]);
            let (bos, before_bos) = layout.masks();
            for mask in [bos, before_bos] {
                words.push(mask.len() as u64);
                words.extend(mask.iter().map(|value| *value as u64));
            }
            words.push(u64::from(conditioning.is_some()));
            if let Some(conditioning) = conditioning {
                tensor_descriptor(&mut words, *conditioning);
            }
        }
        Invocation::Live { output, .. } => {
            words.push(1);
            output_descriptor(&mut words, output);
        }
        Invocation::PerceptionOnly => words.push(2),
        Invocation::ReplayAudio => words.push(6),
        Invocation::Tool { audio, output, .. } => {
            words.push(3);
            match audio {
                ToolAudio::Fresh => words.push(0),
                ToolAudio::Silence(audio) => {
                    words.push(1);
                    tensor_descriptor(&mut words, *audio);
                }
                ToolAudio::Retained => words.push(2),
            }
            output_descriptor(&mut words, output);
        }
        Invocation::Reminder { spoken, output } => {
            words.extend([4, *spoken as u64]);
            output_descriptor(&mut words, output);
        }
        Invocation::Drain => words.push(5),
    }
    words
}

fn output_descriptor(words: &mut Vec<u64>, output: &super::super::tts_feedback::OutputPolicy<'_>) {
    use super::super::tts_feedback::WaveformSilence;
    words.extend([
        u64::from(output.force_silence_on_eos),
        match output.waveform {
            WaveformSilence::PreserveGenerated => 0,
            WaveformSilence::ApplySilence => 1,
        },
        output.silence_on_pad.len() as u64,
    ]);
    words.extend(output.silence_on_pad.iter().copied().map(u64::from));
}

fn tensor_descriptor<T: Tensor>(words: &mut Vec<u64>, tensor: &T) {
    use eredu_nn::TensorElementType;
    words.extend([
        match tensor.element_type() {
            None => 0,
            Some(TensorElementType::Bool) => 1,
            Some(TensorElementType::F16) => 2,
            Some(TensorElementType::Bf16) => 3,
            Some(TensorElementType::F32) => 4,
            Some(TensorElementType::F64) => 5,
            Some(TensorElementType::I8) => 6,
            Some(TensorElementType::I16) => 7,
            Some(TensorElementType::I32) => 8,
            Some(TensorElementType::I64) => 9,
            Some(TensorElementType::U8) => 10,
            Some(TensorElementType::U16) => 11,
            Some(TensorElementType::U32) => 12,
            Some(TensorElementType::U64) => 13,
            Some(TensorElementType::Complex64) => 14,
        },
        tensor.shape().len() as u64,
    ]);
    words.extend(tensor.shape().iter().map(|extent| *extent as u64));
}
