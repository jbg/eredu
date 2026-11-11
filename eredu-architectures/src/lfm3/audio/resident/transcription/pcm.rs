//! Bounded PCM framing for the shared LiveASR block driver.

use super::*;
use crate::lfm3::audio::encoder::MEL_BINS;
use eredu_media::audio::power_mel::PowerMelStream;

const BINS: usize = MEL_BINS as usize;

/// LiveASR with bounded host PCM preprocessing and explicit block advancement.
pub struct AudioPcmTranscription<B, H>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    H: ResidentAudioBinding<B>,
{
    session: AudioTranscription<B, H>,
    mel: PowerMelStream<'static>,
    values: Vec<f32>,
    frame_capacity: usize,
    input_finished: bool,
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
        let mel = self.artifact.processor.transform()?.into_stream();
        let session = self.prepare_live_transcription()?;
        let frame_capacity = session
            .policy
            .frames_per_block
            .checked_mul(2)
            .ok_or(AudioTranscriptionError::Policy)?;
        mel.available_samples(frame_capacity)
            .ok_or(AudioTranscriptionError::Policy)?;
        // Reserve space for two complete blocks and the optional flush frame.
        let capacity = frame_capacity
            .checked_add(1)
            .and_then(|n| n.checked_mul(BINS))
            .ok_or(AudioTranscriptionError::Policy)?;
        let mut values = Vec::new();
        values.try_reserve_exact(capacity).map_err(failure)?;
        Ok(AudioPcmTranscription {
            session,
            mel,
            values,
            frame_capacity,
            input_finished: false,
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
        if self.input_finished || self.is_finished() {
            return 0;
        }
        self.mel
            .available_samples(self.frame_capacity - self.values.len() / BINS)
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
        if self.input_finished {
            return Err(AudioTranscriptionError::InputFinished);
        }
        let available = self.available_samples();
        if samples.len() > available {
            return Err(AudioTranscriptionError::Backpressure {
                incoming: samples.len(),
                available,
            });
        }
        let features = self.mel.advance(samples, sample_rate, false)?;
        self.values.extend(features.values);
        Ok(())
    }

    /// Closes PCM input and right-pads the last incomplete window when needed.
    /// The caller can advance remaining blocks until the transcript is finished.
    pub fn finish_input(&mut self) -> Result<(), AudioTranscriptionError> {
        if self.is_finished() {
            return Err(AudioTranscriptionError::Finished);
        }
        if !self.input_finished {
            let features = self.mel.advance(&[], 16_000, true)?;
            self.values.extend(features.values);
            self.input_finished = true;
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
        let available = self.values.len() / BINS;
        let cadence = self.session.frames_per_block();
        if available < cadence && !self.input_finished {
            return Ok(None);
        }
        let frames = available.min(cadence);
        let values = frames * BINS;
        let features = LogMelFeatures {
            values: self.values[..values].to_vec(),
            mask: vec![true; frames],
            frames,
            mel_bins: BINS,
        };
        let final_chunk = self.input_finished && available <= cadence;
        let block = self.session.advance(&features, final_chunk, restart)?;
        self.values.drain(..values);
        Ok(Some(block))
    }

    /// Whether the final transcript block was published or the session was cancelled.
    pub fn is_finished(&self) -> bool {
        self.session.finished
    }

    /// Closes the session and discards queued input.
    pub fn cancel(&mut self) {
        self.values.clear();
        self.session.cancel();
    }
}
