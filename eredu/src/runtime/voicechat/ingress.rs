//! Bounded PCM ingress whose head retires with the ordinary neural transaction.

use std::{cell::RefCell, rc::Rc};

use eredu_architectures::voicechat::{
    frontend::{
        validate_input_pcm, FrontendError, PcmPrecision, INPUT_FRAME_SAMPLES, INPUT_SAMPLE_RATE,
    },
    input::{InputError, PcmInputState},
    observation::OwnedHostOutput,
    schedule::{AudioSource, Step},
};
use eredu_media::audio::streaming::{
    AudioStreamError, FinalPcmFrame, PcmFinishPolicy, PcmFrame, PcmFrameQueue, PcmFrameSource,
};

/// Host ingress bounds and the admitted frontend's PCM precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatInputConfig {
    /// Maximum uncommitted samples, including a frame held by pending execution.
    pub capacity_samples: usize,
    /// Must match the retained architecture frontend; overflow is rejected at submission.
    pub pcm_precision: PcmPrecision,
}

impl Default for VoiceChatInputConfig {
    fn default() -> Self {
        Self {
            capacity_samples: 16 * INPUT_FRAME_SAMPLES,
            pcm_precision: PcmPrecision::Bf16,
        }
    }
}

/// Input accounting independent of tool substeps and output-audio time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VoiceChatInputStatus {
    /// All accepted source samples, excluding padding.
    pub accepted_samples: u64,
    /// Source samples whose perception transition committed.
    pub committed_samples: u64,
    /// Committed acoustic cadence frames, including declared continuation.
    pub committed_frames: u64,
    /// Committed synthetic silence frames, excluded from source-sample counts.
    pub committed_continuation_frames: u64,
    /// Declared synthetic frames still uncommitted, including a reserved one.
    pub remaining_continuation_frames: u64,
    /// Accepted samples still retained, including the reserved head.
    pub queued_samples: usize,
    /// Original samples in the outstanding receipt; zero for continuation or finish-only work.
    pub reserved_samples: usize,
    /// Whether one receipt is retained by queued or pending work.
    pub pending: bool,
    /// Whether external PCM admission is closed.
    pub finished: bool,
    /// Whether the finish policy has reached canonical preprocessing state.
    pub finish_declared: bool,
    /// Whether all source audio and declared continuation have committed.
    pub finish_committed: bool,
    /// Canonical execution was removed; further submissions are rejected.
    pub closed: bool,
}

/// Rejected input preserves accepted PCM and its committed frontier.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatInputError {
    /// The canonical execution owner has been cancelled, failed or dropped.
    #[error("VoiceChat PCM ingress owner is closed")]
    Closed,
    /// This low-level publication owner has no external ingress.
    #[error("VoiceChat PCM ingress is not configured")]
    Unavailable,
    /// A work receipt or snapshot copy retains input admission.
    #[error("VoiceChat PCM ingress has an outstanding receipt")]
    Busy,
    /// More samples or an explicit finish declaration are needed before enqueueing.
    #[error("VoiceChat PCM ingress needs more input")]
    NeedsInput,
    /// All input and its finish marker have committed.
    #[error("VoiceChat PCM ingress is drained")]
    Drained,
    /// A fresh-audio invocation requires a retained input permit.
    #[error("VoiceChat invocation requires a PCM input permit")]
    Required,
    /// Permit belongs to a different conversation's ingress.
    #[error("foreign VoiceChat PCM input permit")]
    ForeignPermit,
    /// Input remains unread or its finish marker is not committed.
    #[error("VoiceChat cannot drain before PCM ingress finishes")]
    DrainPending,
    /// The supplied branch, step or observed source clock differs from its receipt.
    #[error("VoiceChat PCM input attribution differs from its receipt")]
    Attribution,
    /// Cold input policy and frontend geometry disagree.
    #[error("invalid VoiceChat PCM input configuration: {0}")]
    Configuration(&'static str),
    /// Portable PCM validation or capacity rejection.
    #[error(transparent)]
    Audio(#[from] AudioStreamError),
    /// Architecture precision or frontend policy rejected the samples.
    #[error(transparent)]
    Frontend(#[from] FrontendError),
    /// Branch preprocessing cannot accept the retained receipt.
    #[error(transparent)]
    Input(#[from] InputError),
}

mod snapshot;

pub(super) struct Ingress {
    external: Rc<()>,
    fork_retention: Option<eredu_runtime::execution_control::SnapshotReservation>,
    config: VoiceChatInputConfig,
    queue: PcmFrameQueue,
    accepted: u64,
    committed: u64,
    frames: u64,
    continuation_frames: u64,
    finish_declared: bool,
    pending: bool,
    reserved: usize,
    finish_committed: bool,
    closed: bool,
}

/// Single external sender, usable while the scheduler owns canonical model state.
/// Submissions are atomic: backpressure consumes none of the rejected chunk.
/// Dropping a sender does not implicitly finish or discard accepted audio.
pub struct VoiceChatInputSender {
    ingress: Rc<RefCell<Ingress>>,
}

impl VoiceChatInputSender {
    pub(super) fn from_owner(ingress: Rc<RefCell<Ingress>>) -> Self {
        Self { ingress }
    }

    /// Accepts arbitrary finite mono 16 kHz chunks without advancing model state.
    pub fn submit(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), VoiceChatInputError> {
        let mut input = self
            .ingress
            .try_borrow_mut()
            .map_err(|_| VoiceChatInputError::Busy)?;
        if input.closed {
            return Err(VoiceChatInputError::Closed);
        }
        validate_input_pcm(samples, sample_rate, input.config.pcm_precision)?;
        let accepted = input
            .accepted
            .checked_add(
                u64::try_from(samples.len())
                    .map_err(|_| VoiceChatInputError::Configuration("sample clock overflow"))?,
            )
            .ok_or(VoiceChatInputError::Configuration("sample clock overflow"))?;
        input.queue.submit(samples, sample_rate)?;
        input.accepted = accepted;
        input.external = Rc::new(());
        Ok(())
    }

    /// Closes source admission and declares bounded silence continuation.
    /// Continuation is generated lazily through ordinary inference; it consumes
    /// acoustic/model capacity but never increases accepted source samples. The
    /// session drains codec overlap after this suffix and pending tool work finish.
    pub fn finish(&mut self, policy: PcmFinishPolicy) -> Result<(), VoiceChatInputError> {
        let mut input = self
            .ingress
            .try_borrow_mut()
            .map_err(|_| VoiceChatInputError::Busy)?;
        if input.closed {
            return Err(VoiceChatInputError::Closed);
        }
        if policy.final_frame == FinalPcmFrame::Partial {
            return Err(VoiceChatInputError::Configuration(
                "partial VoiceChat frames require zero padding",
            ));
        }
        let was_finished = input.queue.is_finished();
        input.queue.finish(policy)?;
        if !was_finished {
            input.external = Rc::new(());
        }
        Ok(())
    }

    /// Reports accepted and committed clocks, including in-flight retention.
    pub fn status(&self) -> VoiceChatInputStatus {
        self.ingress.borrow().status()
    }
}

impl Ingress {
    pub(super) fn new(
        config: VoiceChatInputConfig,
    ) -> Result<(Rc<RefCell<Self>>, VoiceChatInputSender), VoiceChatInputError> {
        let input = Rc::new(RefCell::new(Self {
            external: Rc::new(()),
            fork_retention: None,
            config,
            queue: PcmFrameQueue::new(
                INPUT_SAMPLE_RATE,
                INPUT_FRAME_SAMPLES,
                config.capacity_samples,
            )?,
            accepted: 0,
            committed: 0,
            frames: 0,
            continuation_frames: 0,
            finish_declared: false,
            pending: false,
            reserved: 0,
            finish_committed: false,
            closed: false,
        }));
        let sender = VoiceChatInputSender {
            ingress: Rc::clone(&input),
        };
        Ok((input, sender))
    }

    pub(super) fn reserve(
        owner: &Rc<RefCell<Self>>,
    ) -> Result<VoiceChatInputPermit, VoiceChatInputError> {
        let mut input = owner.borrow_mut();
        if input.closed {
            return Err(VoiceChatInputError::Closed);
        }
        if input.pending {
            return Err(VoiceChatInputError::Busy);
        }
        if input.finish_committed {
            return Err(VoiceChatInputError::Drained);
        }
        let frame = input.queue.peek_frame();
        if frame.is_none() && !input.queue.is_drained() {
            return Err(VoiceChatInputError::NeedsInput);
        }
        if frame.is_some() && input.frames.checked_add(1).is_none() {
            return Err(VoiceChatInputError::Configuration("frame clock overflow"));
        }
        let declaration =
            if !input.finish_declared && input.queue.queued_samples() <= INPUT_FRAME_SAMPLES {
                input.queue.finish_policy()
            } else {
                None
            };
        let finish = input.queue.is_finished()
            && match &frame {
                Some(frame) if frame.source == PcmFrameSource::Continuation => {
                    input.queue.remaining_continuation_frames() == 1
                }
                _ => {
                    input.queue.queued_samples() <= INPUT_FRAME_SAMPLES
                        && input.queue.remaining_continuation_frames() == 0
                }
            };
        input.reserved = frame.as_ref().map_or(0, |frame| frame.valid_samples);
        input.pending = true;
        Ok(VoiceChatInputPermit {
            ingress: Rc::clone(owner),
            frame,
            frame_index: input.frames,
            declaration,
            finish,
            active: true,
        })
    }

    pub(super) fn validate_drain(owner: &Rc<RefCell<Self>>) -> Result<(), VoiceChatInputError> {
        let input = owner.borrow();
        if input.closed {
            return Err(VoiceChatInputError::Closed);
        }
        if input.pending || !input.finish_committed || !input.queue.is_drained() {
            Err(VoiceChatInputError::DrainPending)
        } else {
            Ok(())
        }
    }

    pub(super) fn close(owner: &Rc<RefCell<Self>>) {
        owner.borrow_mut().closed = true;
    }

    pub(super) fn status(&self) -> VoiceChatInputStatus {
        VoiceChatInputStatus {
            accepted_samples: self.accepted,
            committed_samples: self.committed,
            committed_frames: self.frames,
            committed_continuation_frames: self.continuation_frames,
            remaining_continuation_frames: self.queue.remaining_continuation_frames(),
            queued_samples: self.queue.queued_samples(),
            reserved_samples: self.reserved,
            pending: self.pending,
            finished: self.queue.is_finished(),
            finish_declared: self.finish_declared,
            finish_committed: self.finish_committed,
            closed: self.closed,
        }
    }
}

/// Move-only head-frame or finish-marker receipt reserved before scheduler enqueue.
/// Accepted samples remain in the ingress queue until semantic commit. Discarding
/// the receipt releases admission and leaves the entire prefix available for retry.
#[must_use = "retain the input receipt with work until commit or discard"]
pub struct VoiceChatInputPermit {
    ingress: Rc<RefCell<Ingress>>,
    frame: Option<PcmFrame>,
    frame_index: u64,
    declaration: Option<PcmFinishPolicy>,
    finish: bool,
    active: bool,
}

impl VoiceChatInputPermit {
    pub(super) fn encode_descriptor(&self, output: &mut Vec<u32>) {
        use sha2::{Digest, Sha256};
        super::driver::encode_u64(output, self.frame_index);
        output.push(u32::from(self.finish));
        match self.declaration {
            Some(policy) => {
                output.push(1);
                output.push(match policy.final_frame {
                    FinalPcmFrame::Partial => 0,
                    FinalPcmFrame::ZeroPad => 1,
                    FinalPcmFrame::RequireComplete => 2,
                });
                super::driver::encode_u64(output, policy.trailing_silence_frames);
            }
            None => output.push(0),
        }
        output.push(match self.ingress.borrow().config.pcm_precision {
            PcmPrecision::F32 => 0,
            PcmPrecision::F16 => 1,
            PcmPrecision::Bf16 => 2,
        });
        if let Some(frame) = &self.frame {
            output.push(1);
            output.push(match frame.source {
                PcmFrameSource::Input => 0,
                PcmFrameSource::Continuation => 1,
            });
            super::driver::encode_u64(output, frame.start_sample);
            super::driver::encode_u64(output, frame.valid_samples as u64);
            let mut digest = Sha256::new();
            for sample in &frame.samples {
                digest.update(sample.to_bits().to_le_bytes());
            }
            for word in digest.finalize().chunks_exact(4) {
                output.push(u32::from_le_bytes(
                    word.try_into().expect("four-byte digest word"),
                ));
            }
        } else {
            output.push(0);
        }
    }
    /// True when no audio frame remains and only frontend finish must commit.
    /// Supply this receipt with a Drain invocation, not a fabricated perception step.
    pub fn is_finish_only(&self) -> bool {
        self.frame.is_none()
    }

    pub(super) fn belongs_to(&self, ingress: &Rc<RefCell<Ingress>>) -> bool {
        Rc::ptr_eq(&self.ingress, ingress)
    }

    pub(super) fn stage<B: PcmInputState>(
        &self,
        step: Step,
        branch: &mut B,
    ) -> Result<(), VoiceChatInputError> {
        if self.frame.is_some() != fresh(step) || (self.frame.is_none() && step != Step::Drain) {
            return Err(VoiceChatInputError::Attribution);
        }
        let input = self.ingress.borrow();
        let config = input.config;
        let status = branch.pcm_input_status();
        if status.frames != self.frame_index
            || status.queued_samples != 0
            || status.drained
            || status.finished != input.finish_declared
            || (input.finish_declared
                && status.remaining_continuation_frames
                    != input.queue.remaining_continuation_frames())
        {
            return Err(VoiceChatInputError::Attribution);
        }
        if status.pcm_precision != config.pcm_precision
            || status.queue_capacity_samples < INPUT_FRAME_SAMPLES
        {
            return Err(VoiceChatInputError::Configuration(
                "frontend precision or capacity differs",
            ));
        }
        if let Some(frame) = &self.frame {
            if frame.source == PcmFrameSource::Input {
                branch.submit_pcm(&frame.samples[..frame.valid_samples], INPUT_SAMPLE_RATE)?;
            }
        }
        if let Some(policy) = self.declaration {
            branch.finish_pcm(policy)?;
        }
        Ok(())
    }

    pub(super) fn validate_output<D>(
        &self,
        output: &OwnedHostOutput<D>,
    ) -> Result<(), VoiceChatInputError> {
        if output.before.audio_frames != self.frame_index {
            return Err(VoiceChatInputError::Attribution);
        }
        match (&self.frame, &output.transcription) {
            (Some(frame), Some(transcription))
                if fresh(output.step)
                    && output.after.audio_frames == self.frame_index + 1
                    && transcription.frame_index == self.frame_index
                    && transcription.start_sample == frame.start_sample
                    && transcription.valid_samples == frame.valid_samples
                    && transcription.source == frame.source
                    && transcription.sample_rate == u64::from(INPUT_SAMPLE_RATE) =>
            {
                Ok(())
            }
            (None, None)
                if output.step == Step::Drain && output.after.audio_frames == self.frame_index =>
            {
                Ok(())
            }
            _ => Err(VoiceChatInputError::Attribution),
        }
    }

    // Precommit validates attribution. Only appends and finish declarations may
    // occur behind this reserved head, so its prefix cannot change before adoption.
    pub(super) fn commit(mut self) {
        let mut input = self.ingress.borrow_mut();
        if self.frame.is_some() {
            let frame = input.queue.consume_frame().expect("reserved input head");
            input.committed += frame.valid_samples as u64;
            input.frames += 1;
            input.continuation_frames += u64::from(frame.source == PcmFrameSource::Continuation);
        }
        if self.declaration.is_some() {
            input.finish_declared = true;
        }
        if self.finish {
            input.finish_committed = true;
        }
        input.pending = false;
        input.reserved = 0;
        self.active = false;
    }
}

impl Drop for VoiceChatInputPermit {
    fn drop(&mut self) {
        if self.active {
            let mut input = self.ingress.borrow_mut();
            input.pending = false;
            input.reserved = 0;
        }
    }
}

fn fresh(step: Step) -> bool {
    matches!(
        step,
        Step::LiveAudio
            | Step::PerceptionOnly
            | Step::Tool {
                audio: AudioSource::Fresh
            }
    )
}

#[cfg(test)]
pub(crate) mod tests;
