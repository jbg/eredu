//! Bounded PCM framing and centered, pre-emphasized Slaney log-mel windows.
//!
//! Callers declare framing, history retention and feature ranges. This module
//! contains no model cadence, encoder cache or end-of-turn policy.

use super::LogMelFeatures;
#[cfg(feature = "audio")]
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::collections::VecDeque;
#[cfg(feature = "audio")]
use std::{collections::BTreeMap, sync::Arc};

/// A portable streaming audio failure. Rejected submissions consume no samples.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AudioStreamError {
    /// Invalid geometry, sample rate, sample value, or frame range.
    #[error("{0}")]
    Invalid(String),
    /// Input remains owned by the caller until capacity becomes available.
    #[error("audio capacity exhausted: {incoming} samples submitted, {available} available")]
    Backpressure {
        /// Number of samples submitted.
        incoming: usize,
        /// Number of free sample slots.
        available: usize,
    },
    /// Input was submitted after the framer was finished.
    #[error("audio stream is finished")]
    Finished,
    /// A finished stream cannot change its declared tail or restart it.
    #[error("audio finish policy differs: existing {existing:?}, requested {requested:?}")]
    FinishPolicyMismatch {
        /// Previously accepted finish policy.
        existing: PcmFinishPolicy,
        /// Rejected replacement policy.
        requested: PcmFinishPolicy,
    },
    /// The accepted input and declared tail cannot fit the sample clock.
    #[error("audio sample timeline overflows")]
    TimelineOverflow,
    /// A requested frame needs additional PCM or the final frame is incomplete.
    #[error("audio requires {required} samples, {available} available")]
    NeedMoreSamples {
        /// Required sample count.
        required: usize,
        /// Currently available sample count.
        available: usize,
    },
    /// FFT execution is disabled by the build configuration.
    #[error("streaming log-mel extraction requires the `audio` feature")]
    AudioFeatureDisabled,
    /// Host continuation storage cannot be represented by the portable byte counter.
    #[error("audio snapshot storage estimate overflows")]
    SnapshotOverflow,
    /// Independent host carry allocation failed before changing the source.
    #[error("audio snapshot allocation failed: {0}")]
    SnapshotAllocation(#[source] std::collections::TryReserveError),
}

fn sample_storage_bytes(inline: usize, capacity: usize) -> Option<u64> {
    u64::try_from(capacity)
        .ok()?
        .checked_mul(std::mem::size_of::<f32>() as u64)?
        .checked_add(u64::try_from(inline).ok()?)
}

fn copy_samples(samples: &VecDeque<f32>) -> Result<VecDeque<f32>, AudioStreamError> {
    let mut copied = VecDeque::new();
    copied
        .try_reserve_exact(samples.len())
        .map_err(AudioStreamError::SnapshotAllocation)?;
    copied.extend(samples.iter().copied());
    Ok(copied)
}

fn validate_pcm(
    samples: &[f32],
    actual_rate: u32,
    expected_rate: u32,
) -> Result<(), AudioStreamError> {
    if actual_rate != expected_rate {
        return Err(AudioStreamError::Invalid(format!(
            "expected {expected_rate} Hz PCM, got {actual_rate} Hz"
        )));
    }
    if samples.iter().any(|v| !v.is_finite()) {
        return Err(AudioStreamError::Invalid(
            "PCM samples must be finite".into(),
        ));
    }
    Ok(())
}

/// How finishing handles an incomplete cadence frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalPcmFrame {
    /// Return the remaining samples without padding.
    Partial,
    /// Pad the final frame with zeros, retaining its valid sample count.
    ZeroPad,
    /// Leave the stream open and report an error if a partial frame remains.
    RequireComplete,
}

/// Explicit final-frame handling and a finite, lazily produced silence tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmFinishPolicy {
    /// How to deliver an incomplete final input frame.
    pub final_frame: FinalPcmFrame,
    /// Full zero frames emitted after all accepted input, without source allocation.
    pub trailing_silence_frames: u64,
}

/// Distinguishes accepted input from explicitly requested continuation audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcmFrameSource {
    /// Frame containing accepted input, including possible final-frame padding.
    Input,
    /// Explicit silence following all accepted input.
    Continuation,
}

/// One framed PCM submission, with sample-clock attribution.
#[derive(Debug, Clone, PartialEq)]
pub struct PcmFrame {
    /// Absolute output sample index, including preceding padding and continuation.
    pub start_sample: u64,
    /// Whether this frame contains input or declared continuation.
    pub source: PcmFrameSource,
    /// Mono floating-point PCM; may contain explicit finish padding.
    pub samples: Vec<f32>,
    /// Original samples, excluding finish padding.
    pub valid_samples: usize,
}

/// Source attribution for a frame removed without materializing its samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcmFrameInfo {
    /// Absolute output sample index, including preceding padding and continuation.
    pub start_sample: u64,
    /// Whether this frame contains input or declared continuation.
    pub source: PcmFrameSource,
    /// Original samples, excluding explicit finish padding.
    pub valid_samples: usize,
}

/// A bounded cadence queue. Cloning preserves pending samples and finish state.
#[derive(Debug, Clone)]
pub struct PcmFrameQueue {
    sample_rate: u32,
    frame_samples: usize,
    capacity: usize,
    samples: VecDeque<f32>,
    consumed: u64,
    output_samples: u64,
    finish: Option<PcmFinishPolicy>,
    continuation_frames: u64,
}

impl PcmFrameQueue {
    /// Conservative logical storage for an independent copy of current PCM carry.
    /// Includes the inline owner and source allocation capacity, excluding allocator overhead.
    pub fn snapshot_storage_bytes(&self) -> Option<u64> {
        sample_storage_bytes(std::mem::size_of::<Self>(), self.samples.capacity())
    }

    /// Independently copies pending PCM, sample attribution and finish policy.
    /// No frame is consumed and no preprocessing work is performed.
    pub fn snapshot_copy(&self) -> Result<Self, AudioStreamError> {
        self.snapshot_storage_bytes()
            .ok_or(AudioStreamError::SnapshotOverflow)?;
        Ok(Self {
            sample_rate: self.sample_rate,
            frame_samples: self.frame_samples,
            capacity: self.capacity,
            samples: copy_samples(&self.samples)?,
            consumed: self.consumed,
            output_samples: self.output_samples,
            finish: self.finish,
            continuation_frames: self.continuation_frames,
        })
    }

    /// Creates a queue whose capacity includes both complete and partial frames.
    pub fn new(
        sample_rate: u32,
        frame_samples: usize,
        capacity: usize,
    ) -> Result<Self, AudioStreamError> {
        if sample_rate == 0 || frame_samples == 0 || capacity < frame_samples {
            return Err(AudioStreamError::Invalid("PCM queue requires a positive rate and frame length, and capacity at least one frame".into()));
        }
        Ok(Self {
            sample_rate,
            frame_samples,
            capacity,
            samples: VecDeque::new(),
            consumed: 0,
            output_samples: 0,
            finish: None,
            continuation_frames: 0,
        })
    }
    /// Atomically submits arbitrary chunk boundaries without dropping audio.
    pub fn submit(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), AudioStreamError> {
        if self.finish.is_some() {
            return Err(AudioStreamError::Finished);
        }
        validate_pcm(samples, sample_rate, self.sample_rate)?;
        let available = self.capacity - self.samples.len();
        if samples.len() > available {
            return Err(AudioStreamError::Backpressure {
                incoming: samples.len(),
                available,
            });
        }
        self.consumed
            .checked_add(self.samples.len() as u64)
            .and_then(|n| n.checked_add(samples.len() as u64))
            .ok_or(AudioStreamError::TimelineOverflow)?;
        self.samples.extend(samples);
        Ok(())
    }
    /// Stops admission and declares how to drain an incomplete final frame.
    pub fn finish(&mut self, policy: PcmFinishPolicy) -> Result<(), AudioStreamError> {
        if let Some(existing) = self.finish {
            return if existing == policy {
                Ok(())
            } else {
                Err(AudioStreamError::FinishPolicyMismatch {
                    existing,
                    requested: policy,
                })
            };
        }
        let remainder = self.samples.len() % self.frame_samples;
        if policy.final_frame == FinalPcmFrame::RequireComplete && remainder != 0 {
            return Err(AudioStreamError::NeedMoreSamples {
                required: self.frame_samples,
                available: remainder,
            });
        }
        let padding = if policy.final_frame == FinalPcmFrame::ZeroPad && remainder != 0 {
            self.frame_samples - remainder
        } else {
            0
        };
        policy
            .trailing_silence_frames
            .checked_mul(self.frame_samples as u64)
            .and_then(|n| n.checked_add(self.samples.len() as u64))
            .and_then(|n| n.checked_add(padding as u64))
            .and_then(|n| n.checked_add(self.output_samples))
            .ok_or(AudioStreamError::TimelineOverflow)?;
        self.finish = Some(policy);
        self.continuation_frames = policy.trailing_silence_frames;
        Ok(())
    }
    /// Whether advancement can remove a complete or declared final partial frame.
    pub fn has_ready_frame(&self) -> bool {
        (!self.samples.is_empty()
            && (self.samples.len() >= self.frame_samples || self.finish.is_some()))
            || self.continuation_frames != 0
    }
    /// Copies only the next complete or declared final partial frame without consuming it.
    /// Appending more PCM does not change an already complete head frame.
    pub fn peek_frame(&self) -> Option<PcmFrame> {
        if !self.has_ready_frame() {
            return None;
        }
        let valid_samples = self.samples.len().min(self.frame_samples);
        let source = if valid_samples == 0 {
            PcmFrameSource::Continuation
        } else {
            PcmFrameSource::Input
        };
        let mut samples: Vec<_> = self.samples.iter().take(valid_samples).copied().collect();
        if source == PcmFrameSource::Continuation
            || self
                .finish
                .is_some_and(|policy| policy.final_frame == FinalPcmFrame::ZeroPad)
        {
            samples.resize(self.frame_samples, 0.0);
        }
        Some(PcmFrame {
            start_sample: self.output_samples,
            source,
            samples,
            valid_samples,
        })
    }
    /// Removes one ready frame without allocating or copying its samples.
    /// The caller retains any previously peeked payload until its own commit boundary.
    pub fn consume_frame(&mut self) -> Option<PcmFrameInfo> {
        if !self.has_ready_frame() {
            return None;
        }
        let frame = PcmFrameInfo {
            start_sample: self.output_samples,
            source: if self.samples.is_empty() {
                PcmFrameSource::Continuation
            } else {
                PcmFrameSource::Input
            },
            valid_samples: self.samples.len().min(self.frame_samples),
        };
        drop(self.samples.drain(..frame.valid_samples));
        self.consumed += frame.valid_samples as u64;
        self.output_samples += if frame.source == PcmFrameSource::Continuation
            || self
                .finish
                .is_some_and(|policy| policy.final_frame == FinalPcmFrame::ZeroPad)
        {
            self.frame_samples as u64
        } else {
            frame.valid_samples as u64
        };
        if frame.source == PcmFrameSource::Continuation {
            self.continuation_frames -= 1;
        }
        Some(frame)
    }
    /// Removes one complete frame, or a declared final partial frame.
    pub fn next_frame(&mut self) -> Option<PcmFrame> {
        let frame = self.peek_frame()?;
        let _ = self.consume_frame();
        Some(frame)
    }
    /// Number of accepted samples that have not yet been delivered.
    pub fn queued_samples(&self) -> usize {
        self.samples.len()
    }
    /// Number of original input samples consumed, excluding padding and continuation.
    pub fn consumed_samples(&self) -> u64 {
        self.consumed
    }
    /// Exact finish declaration, retained even after its continuation is exhausted.
    pub fn finish_policy(&self) -> Option<PcmFinishPolicy> {
        self.finish
    }
    /// Declared continuation frames not yet consumed; peeking does not reduce this count.
    pub fn remaining_continuation_frames(&self) -> u64 {
        self.continuation_frames
    }
    /// Whether input admission has finished, including while queued frames remain.
    pub fn is_finished(&self) -> bool {
        self.finish.is_some()
    }
    /// Whether finish has been declared and all input and continuation delivered.
    pub fn is_drained(&self) -> bool {
        self.finish.is_some() && self.samples.is_empty() && self.continuation_frames == 0
    }
}

/// Centered symmetric-Hann, power-spectrum, area-normalized Slaney extraction.
#[derive(Debug, Clone, PartialEq)]
pub struct CenteredSlaneyLogMelConfig {
    /// Input sampling rate in hertz.
    pub sample_rate: u32,
    /// Symmetric Hann length, centered within the FFT window.
    pub window_length: usize,
    /// FFT length; even and at least the window length.
    pub fft_length: usize,
    /// Distance between frame centers in samples.
    pub hop_length: usize,
    /// Number of mel filters.
    pub mel_bins: usize,
    /// Lower filter frequency in hertz.
    pub min_frequency: f32,
    /// Upper filter frequency in hertz.
    pub max_frequency: f32,
    /// First-order pre-emphasis coefficient; the retained window's first sample is unchanged.
    pub preemphasis: f32,
    /// Positive additive floor before the natural logarithm.
    pub log_floor: f32,
    /// Maximum retained PCM samples. Appends beyond this bound fail atomically.
    pub capacity_samples: usize,
}
impl CenteredSlaneyLogMelConfig {
    fn validate(&self) -> Result<(), AudioStreamError> {
        if self.sample_rate == 0
            || self.window_length < 2
            || self.fft_length < self.window_length
            || !self.fft_length.is_multiple_of(2)
            || self.hop_length == 0
            || self.mel_bins == 0
            || self.capacity_samples <= self.fft_length / 2
            || !self.min_frequency.is_finite()
            || !self.max_frequency.is_finite()
            || self.min_frequency < 0.0
            || self.max_frequency <= self.min_frequency
            || self.max_frequency > self.sample_rate as f32 / 2.0
            || !self.preemphasis.is_finite()
            || !self.log_floor.is_finite()
            || self.log_floor <= 0.0
        {
            return Err(AudioStreamError::Invalid(
                "invalid centered Slaney log-mel configuration".into(),
            ));
        }
        self.mel_bins
            .checked_mul(self.fft_length / 2 + 1)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>()))
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .and_then(|_| self.capacity_samples.checked_add(self.fft_length))
            .filter(|extent| *extent <= isize::MAX as usize)
            .filter(|_| self.capacity_samples <= isize::MAX as usize / std::mem::size_of::<f32>())
            .ok_or_else(|| {
                AudioStreamError::Invalid("log-mel geometry exceeds addressable bounds".into())
            })?;
        Ok(())
    }
}

/// Right-edge semantics for a requested finite PCM window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MelWindowBoundary {
    /// Require real right context for every requested FFT frame.
    RequireLookahead,
    /// Reflect pre-emphasized samples at both finite-window boundaries.
    Reflect,
}

/// Bounded PCM history and reusable interior FFT rows.
///
/// `discard_before` explicitly advances the retained window origin. Frame indices
/// in `extract` are relative to that origin, including after a rollover. Clones
/// retain independent PCM/cache state and share only immutable FFT/filter tables.
#[derive(Clone)]
pub struct StreamingLogMelWindow {
    config: CenteredSlaneyLogMelConfig,
    samples: VecDeque<f32>,
    start_sample: u64,
    #[cfg(feature = "audio")]
    tables: Arc<MelTables>,
    #[cfg(feature = "audio")]
    cache: BTreeMap<u64, Vec<f32>>,
}
#[cfg(feature = "audio")]
struct MelTables {
    window: Vec<f32>,
    filters: Vec<f32>,
    fft: Arc<dyn Fft<f32>>,
}
impl std::fmt::Debug for StreamingLogMelWindow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamingLogMelWindow")
            .field("config", &self.config)
            .field("start_sample", &self.start_sample)
            .field("retained_samples", &self.samples.len())
            .finish()
    }
}
impl StreamingLogMelWindow {
    /// Conservative incremental logical storage for independently copied PCM and
    /// cached mel rows. Immutable FFT/filter tables stay shared through their Arc;
    /// their existing storage is not allocated or charged again by a host copy.
    /// The estimate includes a conservative tree-node allowance per cached row,
    /// and does not describe allocator-private overhead.
    pub fn snapshot_storage_bytes(&self) -> Option<u64> {
        let bytes = sample_storage_bytes(std::mem::size_of::<Self>(), self.samples.capacity())?;
        #[cfg(feature = "audio")]
        let bytes = self.cache.values().try_fold(bytes, |total, row| {
            // A full tree node allowance for every entry conservatively includes
            // keys, Vec owners, links and unused node slots in a sparse tree.
            total
                .checked_add(4096)?
                .checked_add(sample_storage_bytes(0, row.capacity())?)
        })?;
        Some(bytes)
    }

    /// Independently copies all mutable PCM/cache carry and the retained window
    /// origin. Existing immutable FFT/filter tables remain shared; no FFT runs.
    pub fn snapshot_copy(&self) -> Result<Self, AudioStreamError> {
        self.snapshot_storage_bytes()
            .ok_or(AudioStreamError::SnapshotOverflow)?;
        #[cfg(feature = "audio")]
        let cache = {
            let mut copied = BTreeMap::new();
            for (&center, row) in &self.cache {
                let mut values = Vec::new();
                values
                    .try_reserve_exact(row.len())
                    .map_err(AudioStreamError::SnapshotAllocation)?;
                values.extend_from_slice(row);
                copied.insert(center, values);
            }
            copied
        };
        Ok(Self {
            config: self.config.clone(),
            samples: copy_samples(&self.samples)?,
            start_sample: self.start_sample,
            #[cfg(feature = "audio")]
            tables: Arc::clone(&self.tables),
            #[cfg(feature = "audio")]
            cache,
        })
    }

    /// Validates geometry and creates bounded host preprocessing state.
    pub fn new(config: CenteredSlaneyLogMelConfig) -> Result<Self, AudioStreamError> {
        config.validate()?;
        #[cfg(not(feature = "audio"))]
        {
            let _ = config;
            Err(AudioStreamError::AudioFeatureDisabled)
        }
        #[cfg(feature = "audio")]
        {
            let window = (0..config.window_length)
                .map(|i| {
                    0.5 - 0.5
                        * (std::f32::consts::TAU * i as f32 / (config.window_length - 1) as f32)
                            .cos()
                })
                .collect();
            let filters = super::slaney_mel_filters(
                config.sample_rate,
                config.fft_length,
                config.mel_bins,
                config.min_frequency,
                config.max_frequency,
            );
            Self::with_coefficients(config, window, filters)
        }
    }

    /// Uses retained preprocessing coefficients without regenerating their values.
    ///
    /// `window` contains `window_length` finite, unpadded coefficients. `filters`
    /// is row-major `[mel_bins, fft_length / 2 + 1]`, finite and nonnegative.
    /// The window is centered in the FFT frame once, as in centered STFT.
    pub fn with_coefficients(
        config: CenteredSlaneyLogMelConfig,
        window: Vec<f32>,
        filters: Vec<f32>,
    ) -> Result<Self, AudioStreamError> {
        config.validate()?;
        if window.len() != config.window_length
            || filters.len() != config.mel_bins * (config.fft_length / 2 + 1)
            || window.iter().any(|value| !value.is_finite())
            || filters
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(AudioStreamError::Invalid(
                "invalid centered log-mel coefficients".into(),
            ));
        }
        #[cfg(not(feature = "audio"))]
        {
            Err(AudioStreamError::AudioFeatureDisabled)
        }
        #[cfg(feature = "audio")]
        {
            let mut padded_window = vec![0.0; config.fft_length];
            let offset = (config.fft_length - config.window_length) / 2;
            padded_window[offset..offset + window.len()].copy_from_slice(&window);
            let fft = FftPlanner::<f32>::new().plan_fft_forward(config.fft_length);
            Ok(Self {
                config,
                samples: VecDeque::new(),
                start_sample: 0,
                tables: Arc::new(MelTables {
                    window: padded_window,
                    filters,
                    fft,
                }),
                cache: BTreeMap::new(),
            })
        }
    }
    /// Appends PCM without implicit eviction; rejection preserves all state.
    pub fn append(&mut self, samples: &[f32], sample_rate: u32) -> Result<(), AudioStreamError> {
        validate_pcm(samples, sample_rate, self.config.sample_rate)?;
        let available = self.config.capacity_samples - self.samples.len();
        if samples.len() > available {
            return Err(AudioStreamError::Backpressure {
                incoming: samples.len(),
                available,
            });
        }
        self.end_sample()
            .checked_add(samples.len() as u64)
            .ok_or_else(|| AudioStreamError::Invalid("PCM sample clock overflow".into()))?;
        self.samples.extend(samples);
        Ok(())
    }
    /// Explicitly releases history before an absolute sample index.
    pub fn discard_before(&mut self, sample: u64) -> Result<(), AudioStreamError> {
        if sample < self.start_sample || sample > self.end_sample() {
            return Err(AudioStreamError::Invalid(
                "discard frontier lies outside retained PCM".into(),
            ));
        }
        let count = (sample - self.start_sample) as usize;
        self.samples.drain(..count);
        self.start_sample = sample;
        #[cfg(feature = "audio")]
        self.cache
            .retain(|&center, _| center.saturating_sub(sample) > self.config.fft_length as u64 / 2);
        Ok(())
    }
    /// Absolute retained window origin.
    pub fn start_sample(&self) -> u64 {
        self.start_sample
    }
    /// Exclusive absolute sample frontier.
    pub fn end_sample(&self) -> u64 {
        self.start_sample + self.samples.len() as u64
    }
    /// Number of retained input samples.
    pub fn retained_samples(&self) -> usize {
        self.samples.len()
    }
    /// Number of centered STFT frames for the current finite window.
    pub fn frame_count(&self) -> usize {
        if self.samples.is_empty() {
            0
        } else {
            self.samples.len() / self.config.hop_length + 1
        }
    }
    /// Extracts a local half-open frame range. No PCM is consumed.
    pub fn extract(
        &mut self,
        frames: std::ops::Range<usize>,
        boundary: MelWindowBoundary,
    ) -> Result<LogMelFeatures, AudioStreamError> {
        #[cfg(not(feature = "audio"))]
        {
            let _ = (frames, boundary);
            Err(AudioStreamError::AudioFeatureDisabled)
        }
        #[cfg(feature = "audio")]
        {
            self.extract_enabled(frames, boundary)
        }
    }
    #[cfg(feature = "audio")]
    fn extract_enabled(
        &mut self,
        frames: std::ops::Range<usize>,
        boundary: MelWindowBoundary,
    ) -> Result<LogMelFeatures, AudioStreamError> {
        if frames.start > frames.end || frames.end > self.frame_count() {
            return Err(AudioStreamError::Invalid(
                "mel frame range lies outside retained PCM".into(),
            ));
        }
        let half = self.config.fft_length / 2;
        if !frames.is_empty() && self.samples.len() <= half {
            return Err(AudioStreamError::NeedMoreSamples {
                required: half + 1,
                available: self.samples.len(),
            });
        }
        if !frames.is_empty() && boundary == MelWindowBoundary::RequireLookahead {
            let required = (frames.end - 1) * self.config.hop_length + half;
            if self.samples.len() < required {
                return Err(AudioStreamError::NeedMoreSamples {
                    required,
                    available: self.samples.len(),
                });
            }
        }
        let frame_count = frames.len();
        let mut values = Vec::with_capacity(
            frame_count
                .checked_mul(self.config.mel_bins)
                .ok_or_else(|| AudioStreamError::Invalid("feature allocation overflow".into()))?,
        );
        let frequency_bins = half + 1;
        let mut buffer = vec![Complex32::default(); self.config.fft_length];
        for frame in frames {
            let center = frame * self.config.hop_length;
            let absolute_center = self.start_sample + center as u64;
            if let Some(row) = self.cache.get(&absolute_center) {
                values.extend_from_slice(row);
                continue;
            }
            for (i, element) in buffer.iter_mut().enumerate() {
                let pos = center as isize + i as isize - half as isize;
                let index = if pos < 0 {
                    (-pos) as usize
                } else if pos as usize >= self.samples.len() {
                    2 * self.samples.len() - 2 - pos as usize
                } else {
                    pos as usize
                };
                let emphasized = if index == 0 {
                    self.samples[0]
                } else {
                    self.samples[index] - self.config.preemphasis * self.samples[index - 1]
                };
                *element = Complex32::new(emphasized * self.tables.window[i], 0.0);
            }
            self.tables.fft.process(&mut buffer);
            let powers: Vec<_> = buffer[..frequency_bins]
                .iter()
                .map(|value| {
                    let magnitude = (value.re * value.re + value.im * value.im).sqrt();
                    magnitude * magnitude
                })
                .collect();
            let row: Vec<_> = self
                .tables
                .filters
                .chunks_exact(frequency_bins)
                .map(|filter| {
                    (filter.iter().zip(&powers).map(|(w, p)| w * p).sum::<f32>()
                        + self.config.log_floor)
                        .ln()
                })
                .collect();
            values.extend_from_slice(&row);
            // Interior rows have neither reflected samples nor a reset pre-emphasis edge.
            if center > half && center + half <= self.samples.len() {
                let max_rows = self.config.capacity_samples / self.config.hop_length + 2;
                while self.cache.len() >= max_rows {
                    self.cache.pop_first();
                }
                self.cache.insert(absolute_center, row);
            }
        }
        Ok(LogMelFeatures {
            values,
            mask: vec![true; frame_count],
            frames: frame_count,
            mel_bins: self.config.mel_bins,
        })
    }
}
