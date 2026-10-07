//! VoiceChat's released 80 ms mel-window schedule over bounded portable host audio.

use super::{artifact::RetainedVoiceChatArtifacts, config::PreprocessingConfig};
use eredu_core::{execution_control::SnapshotEstimate, BackendFailure};
use eredu_media::audio::{
    streaming::{
        AudioStreamError, CenteredSlaneyLogMelConfig, FinalPcmFrame, MelWindowBoundary,
        PcmFinishPolicy, PcmFrameQueue, PcmFrameSource, StreamingLogMelWindow,
    },
    LogMelFeatures,
};
use eredu_runtime::execution_control::{SnapshotReservation, SnapshotValue};

/// Released input cadence in samples at 16 kHz.
pub const INPUT_FRAME_SAMPLES: usize = 1280;
/// Released input sampling rate.
pub const INPUT_SAMPLE_RATE: u32 = 16000;

/// Precision applied to accepted PCM before F32 feature extraction.
/// The pinned native wrapper stores PCM in its selected compute dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmPrecision {
    /// Preserve the supplied F32 samples.
    F32,
    /// Round each sample to IEEE F16, then widen for host processing.
    F16,
    /// Round each sample to BF16, the pinned streaming default.
    Bf16,
}

/// Explicit host retention limits and initial subsampling phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrontendOptions {
    /// Input rounding before feature extraction; the pinned default is BF16.
    pub pcm_precision: PcmPrecision,
    /// Rolling PCM history in 80 ms frames; at least three, default 71.
    pub buffer_frames: usize,
    /// Maximum accepted, unprocessed PCM samples, including an incomplete frame.
    pub queue_capacity_samples: usize,
    /// Use nine zero mel rows and eight real rows on the first step, dropping two outputs.
    /// The released streaming configuration uses `false` (one real row, no drop).
    pub pad_and_drop_preencoded: bool,
}
impl Default for FrontendOptions {
    fn default() -> Self {
        Self {
            pcm_precision: PcmPrecision::Bf16,
            buffer_frames: 71,
            queue_capacity_samples: 16 * INPUT_FRAME_SAMPLES,
            pad_and_drop_preencoded: false,
        }
    }
}

/// Rejected audio, unsupported preprocessing policy or exhausted sample clock.
#[derive(Debug, thiserror::Error)]
pub enum FrontendError {
    /// Preparation did not retain perception's preprocessing buffers.
    #[error("VoiceChat frontend requires retained perception preprocessing buffers")]
    MissingPreprocessing,
    /// Configuration changes the released streaming features or cadence.
    #[error("{0}")]
    Configuration(&'static str),
    /// Host preprocessing or bounded admission failed without consuming the submission.
    #[error(transparent)]
    Audio(#[from] AudioStreamError),
    /// The stream's frame counter cannot represent another transition.
    #[error("VoiceChat audio frame clock exhausted")]
    ClockExhausted,
}

/// Validates released PCM admission, including overflow in the selected precision.
/// This check does not allocate or change preprocessing state.
pub fn validate_input_pcm(
    samples: &[f32],
    sample_rate: u32,
    precision: PcmPrecision,
) -> Result<(), FrontendError> {
    if sample_rate != INPUT_SAMPLE_RATE
        || samples
            .iter()
            .any(|&sample| !sample.is_finite() || !round_pcm(sample, precision).is_finite())
    {
        return Err(AudioStreamError::Invalid(
            "VoiceChat requires 16 kHz PCM finite in the selected precision".into(),
        )
        .into());
    }
    Ok(())
}

fn round_pcm(sample: f32, precision: PcmPrecision) -> f32 {
    match precision {
        PcmPrecision::F16 => half::f16::from_f32(sample).to_f32(),
        PcmPrecision::Bf16 => half::bf16::from_f32(sample).to_f32(),
        PcmPrecision::F32 => sample,
    }
}

/// PCM carry, finite-window mel cache and audio frontier, with independently cloneable state.
#[derive(Debug, Clone)]
pub struct VoiceChatFrontend {
    options: FrontendOptions,
    queue: PcmFrameQueue,
    window: StreamingLogMelWindow,
    frames: u64,
    mel_bins: usize,
    history_capacity: usize,
}

impl SnapshotValue for VoiceChatFrontend {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let bytes = self
            .queue
            .snapshot_storage_bytes()
            .and_then(|queue| queue.checked_add(self.window.snapshot_storage_bytes()?))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>() as u64))
            .ok_or_else(|| BackendFailure::from_error(AudioStreamError::SnapshotOverflow))?;
        // Counting component inline owners again conservatively covers aggregate
        // bookkeeping. Shared immutable FFT/filter tables incur no copy charge.
        Ok(SnapshotEstimate {
            retained_bytes: bytes,
            copy_bytes: bytes,
        })
    }

    fn copy_snapshot_value(&self, _: &SnapshotReservation) -> Result<Self, BackendFailure> {
        self.estimate_snapshot_value()?;
        Ok(Self {
            options: self.options,
            queue: self
                .queue
                .snapshot_copy()
                .map_err(BackendFailure::from_error)?,
            window: self
                .window
                .snapshot_copy()
                .map_err(BackendFailure::from_error)?,
            frames: self.frames,
            mel_bins: self.mel_bins,
            history_capacity: self.history_capacity,
        })
    }
}

/// One unpublished frontend transition. Retain `state` only with its neural transition.
#[derive(Debug, Clone)]
pub struct FrontendStep {
    /// Frame-major float32 mel values, including declared zero prefix rows.
    /// The mask describes host audio provenance, not encoder padding: every row
    /// participates in subsampling, including the zero prefix.
    pub mel: LogMelFeatures,
    /// Number of subsampler output rows to discard before the Conformer.
    pub drop_preencoded: usize,
    /// Zero-based audio frame identity, independent of tool-channel substeps.
    pub frame_index: u64,
    /// Absolute cadence sample position, including explicit continuation silence.
    pub start_sample: u64,
    /// Actual input samples, excluding explicit final padding.
    pub valid_samples: usize,
    /// Original PCM or bounded synthetic silence; both advance the mel cadence.
    pub source: PcmFrameSource,
    /// Candidate carry and feature cache; the parent is unchanged.
    pub state: VoiceChatFrontend,
}
impl VoiceChatFrontend {
    /// Applies the streaming reference's dither=0 and pad_to=0 overrides.
    ///
    /// Model configuration is already normalized; no `_target_` is executed.
    pub fn new(
        config: &PreprocessingConfig,
        options: FrontendOptions,
    ) -> Result<Self, FrontendError> {
        Self::build(config, options, StreamingLogMelWindow::new)
    }

    /// Constructs from canonical checkpoint coefficients retained during admission.
    /// No artifact is reopened and no coefficients are regenerated.
    pub fn from_retained(
        artifacts: &RetainedVoiceChatArtifacts,
        options: FrontendOptions,
    ) -> Result<Self, FrontendError> {
        let preprocessing = artifacts
            .preprocessing()
            .ok_or(FrontendError::MissingPreprocessing)?;
        Self::with_coefficients(
            &artifacts.plan().config().perception.preprocessor,
            options,
            preprocessing.window().to_vec(),
            preprocessing.filters().to_vec(),
        )
    }

    /// Uses explicit unpadded Hann and row-major mel-filter coefficients.
    /// Saved values are preserved; the shared host processor validates their geometry.
    pub fn with_coefficients(
        config: &PreprocessingConfig,
        options: FrontendOptions,
        window: Vec<f32>,
        filters: Vec<f32>,
    ) -> Result<Self, FrontendError> {
        Self::build(config, options, |geometry| {
            StreamingLogMelWindow::with_coefficients(geometry, window, filters)
        })
    }

    fn build(
        config: &PreprocessingConfig,
        options: FrontendOptions,
        create_window: impl FnOnce(
            CenteredSlaneyLogMelConfig,
        ) -> Result<StreamingLogMelWindow, AudioStreamError>,
    ) -> Result<Self, FrontendError> {
        if options.buffer_frames < 3 {
            return Err(FrontendError::Configuration(
                "PCM history requires at least three frames",
            ));
        }
        let history_capacity = options
            .buffer_frames
            .checked_mul(INPUT_FRAME_SAMPLES)
            .ok_or(FrontendError::Configuration(
                "PCM history capacity overflow",
            ))?;
        let window = create_window(preprocessing_geometry(config, history_capacity)?)?;
        Ok(Self {
            options,
            queue: PcmFrameQueue::new(
                INPUT_SAMPLE_RATE,
                INPUT_FRAME_SAMPLES,
                options.queue_capacity_samples,
            )?,
            window,
            frames: 0,
            mel_bins: config.features,
            history_capacity,
        })
    }

    /// Accepts arbitrary PCM boundaries atomically, returning typed backpressure at the bound.
    pub fn submit(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), FrontendError> {
        if self.queue.is_finished() {
            return Err(AudioStreamError::Finished.into());
        }
        validate_input_pcm(samples, sample_rate, self.options.pcm_precision)?;
        let available = self.options.queue_capacity_samples - self.queue.queued_samples();
        if samples.len() > available {
            return Err(AudioStreamError::Backpressure {
                incoming: samples.len(),
                available,
            }
            .into());
        }
        if self.options.pcm_precision == PcmPrecision::F32 {
            self.queue.submit(samples, sample_rate)?;
        } else {
            // Allocate only after both bounded and selected-precision admission.
            let rounded: Vec<_> = samples
                .iter()
                .map(|&sample| round_pcm(sample, self.options.pcm_precision))
                .collect();
            self.queue.submit(&rounded, sample_rate)?;
        }
        Ok(())
    }
    /// Stops input with explicit partial handling and finite silence continuation.
    pub fn finish(&mut self, policy: PcmFinishPolicy) -> Result<(), FrontendError> {
        if policy.final_frame == FinalPcmFrame::Partial {
            return Err(FrontendError::Configuration(
                "VoiceChat final PCM frames require complete or zero-padded cadence",
            ));
        }
        Ok(self.queue.finish(policy)?)
    }
    /// Accepted input samples waiting for advancement.
    pub fn queued_samples(&self) -> usize {
        self.queue.queued_samples()
    }
    /// Exact host retention, PCM precision and initial subsampling policy.
    pub fn options(&self) -> FrontendOptions {
        self.options
    }
    /// Whether a cadence frame is ready, including explicitly finished padding.
    /// This check consumes no PCM and performs no feature extraction.
    pub fn has_ready_frame(&self) -> bool {
        self.queue.has_ready_frame()
    }
    /// Whether accepted PCM and declared continuation have drained. Codec drain is separate.
    pub fn is_drained(&self) -> bool {
        self.queue.is_drained()
    }
    /// Whether input admission is closed, even while continuation frames remain.
    pub fn is_finished(&self) -> bool {
        self.queue.is_finished()
    }
    /// Synthetic cadence frames still owed by the installed finish declaration.
    pub fn remaining_continuation_frames(&self) -> u64 {
        self.queue.remaining_continuation_frames()
    }
    /// The committed input cadence frontier of this branch.
    pub fn frames(&self) -> u64 {
        self.frames
    }
    /// Number of PCM samples retained for the finite preprocessing window.
    pub fn retained_samples(&self) -> usize {
        self.window.retained_samples()
    }

    /// Prepares at most one cadence transition, preserving the parent on success or failure.
    pub fn advance(&self) -> Result<Option<FrontendStep>, FrontendError> {
        let mut next = self.clone();
        let Some(frame) = next.queue.next_frame() else {
            return Ok(None);
        };
        let next_frames = self
            .frames
            .checked_add(1)
            .ok_or(FrontendError::ClockExhausted)?;
        let retained = next.window.retained_samples();
        if retained > self.history_capacity - INPUT_FRAME_SAMPLES {
            next.window.discard_before(
                next.window.start_sample()
                    + (retained - (self.history_capacity - INPUT_FRAME_SAMPLES)) as u64,
            )?;
        }
        next.window.append(&frame.samples, INPUT_SAMPLE_RATE)?;
        let (start, end, prefix, drop_preencoded) =
            if self.frames == 0 {
                if self.options.pad_and_drop_preencoded {
                    (0, 8, 9, 2)
                } else {
                    (0, 1, 0, 0)
                }
            } else {
                // The reference clamps against the finite rolling window after rollover.
                let available_end = next.window.frame_count().checked_sub(7).ok_or(
                    FrontendError::Configuration("PCM window is shorter than streaming lookahead"),
                )?;
                let nominal_end = self
                    .frames
                    .checked_mul(8)
                    .and_then(|n| n.checked_add(1))
                    .ok_or(FrontendError::ClockExhausted)?;
                let end = nominal_end.min(available_end as u64) as usize;
                let main_start = end.checked_sub(8).ok_or(FrontendError::Configuration(
                    "PCM window is shorter than a mel step",
                ))?;
                let cache_start = main_start.saturating_sub(9);
                (cache_start, end, 9 - (main_start - cache_start), 2)
            };
        let extracted = next
            .window
            .extract(start..end, MelWindowBoundary::Reflect)?;
        let mel = if prefix == 0 {
            extracted
        } else {
            let mut values = vec![0.0; prefix * self.mel_bins];
            values.extend_from_slice(&extracted.values);
            let mut mask = vec![false; prefix];
            mask.extend_from_slice(&extracted.mask);
            LogMelFeatures {
                values,
                mask,
                frames: prefix + extracted.frames,
                mel_bins: self.mel_bins,
            }
        };
        next.frames = next_frames;
        Ok(Some(FrontendStep {
            mel,
            drop_preencoded,
            frame_index: self.frames,
            start_sample: frame.start_sample,
            valid_samples: frame.valid_samples,
            source: frame.source,
            state: next,
        }))
    }
}

/// Shared released preprocessing geometry; construction performs no FFT work.
pub(super) fn preprocessing_geometry(
    config: &PreprocessingConfig,
    capacity_samples: usize,
) -> Result<CenteredSlaneyLogMelConfig, FrontendError> {
    if config.sample_rate != INPUT_SAMPLE_RATE as usize
        || config.window_size != 0.025
        || config.window_stride != 0.01
        || config.window != "hann"
        || config.normalize != "NA"
        || config.n_fft != 512
        || config.features != 128
        || config.frame_splicing != 1
        || config.exact_pad
        || config.mag_power != 2.0
        || config.mel_norm != "slaney"
        || config.use_torchaudio
        || !config.log
        || config.log_zero_guard_type != "add"
        || config.pad_value != 0.0
    {
        return Err(FrontendError::Configuration(
            "unsupported VoiceChat preprocessing policy",
        ));
    }
    if !config.lowfreq.is_finite()
        || !config.highfreq.is_finite()
        || config.lowfreq < 0.0
        || config.highfreq <= config.lowfreq
        || config.highfreq > INPUT_SAMPLE_RATE as f64 / 2.0
        || !(config.preemph as f32).is_finite()
        || !(config.log_zero_guard_value as f32).is_finite()
        || config.log_zero_guard_value as f32 <= 0.0
    {
        return Err(FrontendError::Configuration(
            "invalid VoiceChat preprocessing coefficients",
        ));
    }
    Ok(CenteredSlaneyLogMelConfig {
        sample_rate: INPUT_SAMPLE_RATE,
        window_length: 400,
        fft_length: 512,
        hop_length: 160,
        mel_bins: config.features,
        min_frequency: config.lowfreq as f32,
        max_frequency: config.highfreq as f32,
        preemphasis: config.preemph as f32,
        log_floor: config.log_zero_guard_value as f32,
        capacity_samples,
    })
}
