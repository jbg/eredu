//! Owned VoiceChat output observed before the shared scheduler publishes state.
//!
//! Native conversion is delegated to the ordinary host observer. This adapter
//! validates architecture attribution and retains only host values; text decoding
//! and conversational policy remain outside the neural execution path.

use eredu_core::BackendFailure;
use eredu_media::audio::streaming::PcmFrameSource;
use eredu_nn::Tensor;
use eredu_runtime::{CompletedRealtimeFrame, RealtimeFrameHostObserver, RealtimeHostChannelValues};

use super::{
    channels::TEXT_VOCABULARY_SIZE,
    execution::ExecutionOutput,
    output::{OutputDecisions, OutputError},
    schedule::{Progress, Step},
};

mod snapshot;

/// Owned finite mono speech, independently timestamped from input audio.
#[derive(Debug, Clone, PartialEq)]
pub struct OwnedSpeechChunk {
    /// Samples actually rendered, including explicit silence substitutions.
    pub samples: Vec<f32>,
    /// Output samples per second.
    pub sample_rate: u64,
    /// Absolute output-sample offset, including tool and reminder speech.
    pub start_sample: u64,
    /// Declared codec latency, without subtracting emitted samples.
    pub delay_samples: u64,
    /// Terminal overlap rather than another generated code frame.
    pub is_tail: bool,
}

/// Host transcription and exact source-audio attribution from one encoder step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptionOutput {
    /// Zero-based live-audio frame, independent of invocation identity.
    pub frame_index: u64,
    /// Source samples per second.
    pub sample_rate: u64,
    /// Absolute first cadence-sample index, including explicit continuation silence.
    pub start_sample: u64,
    /// Actual source samples, excluding explicit final-frame zero padding.
    pub valid_samples: usize,
    /// Original PCM or synthetic silence; continuation contributes no source samples.
    pub source: PcmFrameSource,
    /// Newly decoded RNNT tokens, without tokenizer or text policy.
    pub emitted: Vec<usize>,
    /// The first joint decision was blank.
    pub first_blank: bool,
    /// The final evaluated joint decision was blank.
    pub ended_blank: bool,
    /// A nonblank decision exhausted the per-frame symbol bound.
    pub hit_symbol_limit: bool,
}

/// Host-owned candidate cached by the shared prepublication completion wrapper.
/// Applications receive this only after its matching branch commits.
#[derive(Debug, Clone)]
pub struct OwnedHostOutput<D> {
    /// Zero-based invocation identity, including prefill and tool substeps.
    pub invocation_index: u64,
    /// Architecture invocation that produced this output.
    pub step: Step,
    /// Actual supplied token for a reminder invocation; absent otherwise.
    pub reminder_token: Option<u32>,
    /// Component frontiers before execution.
    pub before: Progress,
    /// Validated component frontiers after execution.
    pub after: Progress,
    /// Raw selections and effective feedback/spoken identities.
    pub decisions: Option<OutputDecisions>,
    /// Complete generated residual codes before feedback substitution.
    pub generated_codes: Option<Vec<i32>>,
    /// Timestamped waveform or terminal overlap.
    pub speech: Option<OwnedSpeechChunk>,
    /// Fresh-input transcription, absent on cached-audio invocations.
    pub transcription: Option<TranscriptionOutput>,
    /// Host diagnostics supplied by the delegated observer.
    pub diagnostics: Vec<D>,
}

/// Architecture projection around a family-independent native-to-host observer.
/// The delegate's diagnostic type contains host data, not native resources.
pub struct VoiceChatHostObserver<H> {
    observer: H,
}

impl<H> VoiceChatHostObserver<H> {
    /// Binds host conversion without allocating tensors or opening artifacts.
    pub const fn new(observer: H) -> Self {
        Self { observer }
    }
}

impl<T, D, H, C, K> RealtimeFrameHostObserver<ExecutionOutput<T, C, K>> for VoiceChatHostObserver<H>
where
    T: Tensor,
    C: super::perception_rnnt::PerceptionCacheState,
    D: Clone,
    H: RealtimeFrameHostObserver<
        CompletedRealtimeFrame<T, T>,
        Output = CompletedRealtimeFrame<RealtimeHostChannelValues, D>,
    >,
    H::Error: Send + Sync,
{
    type Output = OwnedHostOutput<D>;
    type Error = ObservationError;

    fn observe(
        &mut self,
        execution: &ExecutionOutput<T, C, K>,
    ) -> Result<Self::Output, Self::Error> {
        let frame = self
            .observer
            .observe(&execution.output.frame)
            .map_err(|error| ObservationError::Backend(BackendFailure::from_error(error)))?;
        let projected = execution.output.projection.project(&frame)?;
        let policy = execution.output.projection.policy();
        if execution.reminder_token.is_some() != (projected.step == Step::Reminder)
            || execution
                .reminder_token
                .is_some_and(|token| token as usize >= TEXT_VOCABULARY_SIZE)
        {
            return Err(ObservationError::Attribution(
                "reminder token differs from the invocation or text vocabulary",
            ));
        }
        if execution.output.schedule.progress() != &projected.after
            || execution.output.schedule.policy() != policy
        {
            return Err(ObservationError::Attribution(
                "candidate schedule differs from projected receipt",
            ));
        }
        let fresh = projected.after.audio_frames != projected.before.audio_frames;
        if fresh != execution.input.is_some() {
            return Err(ObservationError::Attribution(
                "fresh input presence differs from projected receipt",
            ));
        }
        let transcription = execution
            .input
            .as_ref()
            .map(|input| {
                let width = policy.input_samples_per_frame();
                let expected_start = projected.before.audio_frames.checked_mul(width).ok_or(
                    ObservationError::Attribution("source sample coordinate overflow"),
                )?;
                let valid = u64::try_from(input.valid_samples).map_err(|_| {
                    ObservationError::Attribution("source sample count exceeds coordinate range")
                })?;
                let valid_source = match input.source {
                    PcmFrameSource::Input => {
                        valid > 0
                            && valid <= width
                            && (valid == width || input.state.frontend().is_finished())
                    }
                    PcmFrameSource::Continuation => {
                        valid == 0 && input.state.frontend().is_finished()
                    }
                };
                if input.frame_index != projected.before.audio_frames
                    || input.start_sample != expected_start
                    || !valid_source
                    || input.state.frontend().frames() != projected.after.audio_frames
                    || input.state.neural().frames() != projected.after.perception_frames
                    || input.state.neural().rnnt().frames() != projected.after.rnnt_frames
                {
                    return Err(ObservationError::Attribution(
                        "source audio attribution differs from projected receipt",
                    ));
                }
                Ok(TranscriptionOutput {
                    frame_index: input.frame_index,
                    sample_rate: policy.input_sample_rate(),
                    start_sample: input.start_sample,
                    valid_samples: input.valid_samples,
                    source: input.source,
                    emitted: input.emitted.clone(),
                    first_blank: input.first_blank,
                    ended_blank: input.ended_blank,
                    hit_symbol_limit: input.hit_symbol_limit,
                })
            })
            .transpose()?;
        Ok(OwnedHostOutput {
            invocation_index: projected.invocation_index,
            step: projected.step,
            reminder_token: execution.reminder_token,
            before: projected.before,
            after: projected.after,
            decisions: projected.decisions,
            generated_codes: projected.generated_codes.map(<[i32]>::to_vec),
            speech: projected.speech.map(|speech| OwnedSpeechChunk {
                samples: speech.samples.to_vec(),
                sample_rate: speech.sample_rate,
                start_sample: speech.start_sample,
                delay_samples: speech.delay_samples,
                is_tail: speech.is_tail,
            }),
            transcription,
            diagnostics: frame.diagnostics().to_vec(),
        })
    }
}

/// Portable failures preventing publication, retaining native conversion sources.
#[derive(Debug, thiserror::Error)]
pub enum ObservationError {
    /// Native-to-host conversion failed after exact completion.
    #[error("VoiceChat host conversion failed")]
    Backend(#[source] BackendFailure),
    /// Shared channel validation or architecture projection failed.
    #[error(transparent)]
    Output(#[from] OutputError),
    /// Source input metadata does not describe the admitted invocation.
    #[error("invalid VoiceChat observation: {0}")]
    Attribution(&'static str),
}
