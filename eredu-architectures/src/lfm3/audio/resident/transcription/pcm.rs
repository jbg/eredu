//! Bounded PCM framing for the shared LiveASR block driver.

use super::*;
use crate::lfm3::audio::encoder::MEL_BINS;
use eredu_media::audio::power_mel::{PowerMel, PowerMelState};

const BINS: usize = MEL_BINS as usize;

/// LiveASR with bounded host PCM preprocessing and explicit block advancement.
pub struct AudioPcmTranscription<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    session: AudioTranscription<B, H, PcmInput>,
    transform: PowerMel,
    frame_capacity: usize,
}

struct PcmInput {
    mel: PowerMelState,
    values: Vec<f32>,
    finished: bool,
}

impl SnapshotValue for PcmInput {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        use eredu_core::execution_control::ExecutionControlError;
        let bytes = self
            .mel
            .snapshot_storage_bytes()
            .and_then(|carry| {
                carry.checked_add(
                    (self.values.capacity() as u64).checked_mul(size_of::<f32>() as u64)?,
                )
            })
            .ok_or_else(|| failure(ExecutionControlError::Overflow))?;
        snapshot_value_estimate::<Self>([Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })])
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut values = Vec::new();
        values
            .try_reserve_exact(self.values.capacity())
            .map_err(failure)?;
        values.extend_from_slice(&self.values);
        Ok(Self {
            mel: self.mel.clone(),
            values,
            finished: self.finished,
        })
    }
}

/// Reusable saved PCM LiveASR continuation, including queued features and partial windows.
pub struct AudioPcmTranscriptionSnapshot<B, H>(AudioTranscriptionSnapshot<B, H, PcmInput>)
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>;

impl<B, H> AudioPcmTranscription<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend<Tensor: SnapshotValue>,
    H: ResidentAudioBinding<B, State: SnapshotValue>,
{
    /// Configures snapshot limits once for this stream.
    pub fn enable_snapshots(&mut self, limits: SnapshotLimits) -> Result<(), AudioSnapshotError> {
        self.session.enable_snapshots(limits)
    }

    /// Reports retained snapshots and cumulative copy charges.
    pub fn snapshot_usage(&self) -> Option<SnapshotUsage> {
        self.session.snapshot_usage()
    }

    /// Saves the exact PCM and model continuation.
    pub fn snapshot(&self) -> Result<AudioPcmTranscriptionSnapshot<B, H>, AudioSnapshotError> {
        self.session.snapshot().map(AudioPcmTranscriptionSnapshot)
    }

    /// Restores queued input and model state in one transaction.
    pub fn restore(
        &mut self,
        saved: &AudioPcmTranscriptionSnapshot<B, H>,
    ) -> Result<(), AudioSnapshotError> {
        self.session.restore(&saved.0)
    }
}

impl<B, H> ResidentAudio<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    /// Prepares live transcription with the retained 16-kHz PCM processor.
    pub fn prepare_pcm_transcription(
        self,
    ) -> Result<AudioPcmTranscription<B, H>, AudioTranscriptionError> {
        let transform = self.artifact.processor.transform()?;
        let mel = PowerMelState::default();
        let policy = Policy::new(&self.artifact)?;
        let frame_capacity = policy
            .frames_per_block
            .checked_mul(2)
            .ok_or(AudioTranscriptionError::Policy)?;
        transform
            .available_samples(&mel, frame_capacity)
            .ok_or(AudioTranscriptionError::Policy)?;
        // Reserve space for two complete blocks and the optional flush frame.
        let capacity = frame_capacity
            .checked_add(1)
            .and_then(|n| n.checked_mul(BINS))
            .ok_or(AudioTranscriptionError::Policy)?;
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(failure)?;
        Ok(AudioPcmTranscription {
            session: self.prepare_transcription(
                policy,
                PcmInput {
                    mel,
                    values,
                    finished: false,
                },
            )?,
            transform,
            frame_capacity,
        })
    }
}

impl<B, H> AudioPcmTranscription<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend + SubmissionBackend<Executor = Context<B>>,
    H: ResidentAudioBinding<B>,
{
    /// Maximum PCM samples accepted by the next submission.
    pub fn available_samples(&self) -> usize {
        if self.session.state.input.finished || self.is_finished() {
            return 0;
        }
        self.transform
            .available_samples(
                &self.session.state.input.mel,
                self.frame_capacity - self.session.state.input.values.len() / BINS,
            )
            .expect("validated PCM frame capacity")
    }

    /// Accepts a chunk of finite mono 16-kHz PCM without advancing the model.
    /// A rejected submission consumes no samples.
    pub fn submit(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
    ) -> Result<(), AudioTranscriptionError> {
        if self.is_finished() {
            return Err(AudioTranscriptionError::Finished);
        }
        if self.session.state.input.finished {
            return Err(AudioTranscriptionError::InputFinished);
        }
        let available = self.available_samples();
        if samples.len() > available {
            return Err(AudioTranscriptionError::Backpressure {
                incoming: samples.len(),
                available,
            });
        }
        let features = self.transform.advance(
            &mut self.session.state.input.mel,
            samples,
            sample_rate,
            false,
        )?;
        self.session.state.input.values.extend(features.values);
        Ok(())
    }

    /// Closes PCM input and right-pads the last incomplete window when needed.
    /// The caller can advance remaining blocks until the transcript is finished.
    pub fn finish_input(&mut self) -> Result<(), AudioTranscriptionError> {
        if self.is_finished() {
            return Err(AudioTranscriptionError::Finished);
        }
        if !self.session.state.input.finished {
            let features =
                self.transform
                    .advance(&mut self.session.state.input.mel, &[], 16_000, true)?;
            self.session.state.input.values.extend(features.values);
            self.session.state.input.finished = true;
        }
        Ok(())
    }

    /// Publishes one block or returns `None` when more input is needed.
    /// A restart applies only when a block is published. A rejected advance
    /// leaves the queued features and transcript frontier available for retry.
    pub fn advance(
        &mut self,
        restart: bool,
    ) -> Result<Option<AudioTranscriptionBlock>, AudioTranscriptionError> {
        if self.is_finished() {
            return Err(AudioTranscriptionError::Finished);
        }
        let available = self.session.state.input.values.len() / BINS;
        let cadence = self.session.frames_per_block();
        if available < cadence && !self.session.state.input.finished {
            return Ok(None);
        }
        let frames = available.min(cadence);
        let values = frames * BINS;
        let features = LogMelFeatures {
            values: self.session.state.input.values[..values].to_vec(),
            mask: vec![true; frames],
            frames,
            mel_bins: BINS,
        };
        let final_chunk = self.session.state.input.finished && available <= cadence;
        let block = self.session.advance(&features, final_chunk, restart)?;
        self.session.state.input.values.drain(..values);
        Ok(Some(block))
    }

    /// Whether the final transcript block was published or the session was cancelled.
    pub fn is_finished(&self) -> bool {
        self.session.is_finished()
    }

    /// Closes the session and discards queued input.
    pub fn cancel(&mut self) {
        self.session.state.input.values.clear();
        self.session.cancel();
    }
}
