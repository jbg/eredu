//! Uncentered periodic-Hann power-mel analysis with bounded sample overlap.

use super::{slaney_mel_filters, AudioWaveform, LogMelFeatures};
use crate::MediaError;
use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::{borrow::Cow, sync::Arc};

/// Host processing geometry for uncentered Slaney power-mel features.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerMelConfig {
    /// Required mono PCM sample rate in Hz.
    pub sample_rate: u32,
    /// FFT and periodic-Hann window length.
    pub fft_length: usize,
    /// Window stride in samples.
    pub hop_length: usize,
    /// Number of filters spanning zero to Nyquist.
    pub mel_bins: usize,
    /// Minimum base-ten log energy.
    pub log_floor: f32,
    /// Offset added after applying the floor.
    pub normalization_offset: f32,
    /// Divisor applied after adding the offset.
    pub normalization_divisor: f32,
}

/// Reusable immutable host transform.
#[derive(Clone)]
pub struct PowerMel {
    config: PowerMelConfig,
    window: Arc<[f32]>,
    filters: Arc<[f32]>,
    fft: Arc<dyn Fft<f32>>,
}

/// Mono stream continuation with fewer than one window of retained samples.
#[derive(Debug, Clone)]
pub struct PowerMelStream<'a> {
    transform: Cow<'a, PowerMel>,
    state: PowerMelState,
}

/// Mutable window carry, independent of the reusable transform.
#[derive(Debug, Clone, Default)]
pub struct PowerMelState {
    carry: Vec<f32>,
    emitted: bool,
    finished: bool,
}

impl std::fmt::Debug for PowerMel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PowerMel")
            .field("config", &self.config)
            .finish()
    }
}

impl PowerMel {
    /// Precomputes the window, mel filter bank and host FFT plan.
    pub fn new(config: PowerMelConfig) -> Result<Self, MediaError> {
        let bins = config.fft_length / 2 + 1;
        if config.sample_rate == 0
            || config.fft_length < 2
            || config.fft_length % 2 != 0
            || config.hop_length == 0
            || config.hop_length > config.fft_length
            || config.mel_bins == 0
            || config.mel_bins.checked_add(2).is_none()
            || config.mel_bins.checked_mul(bins).is_none()
            || !config.log_floor.is_finite()
            || !config.normalization_offset.is_finite()
            || !config.normalization_divisor.is_finite()
            || config.normalization_divisor <= 0.0
        {
            return Err(MediaError::invalid(
                "invalid uncentered power-mel configuration",
            ));
        }
        let step = (std::f64::consts::TAU / config.fft_length as f64) as f32;
        let window = (0..config.fft_length)
            .map(|index| 0.5 - 0.5 * (index as f32 * step).cos())
            .collect();
        let filters = slaney_mel_filters(
            config.sample_rate,
            config.fft_length,
            config.mel_bins,
            0.0,
            config.sample_rate as f32 / 2.0,
        );
        let fft = FftPlanner::new().plan_fft_forward(config.fft_length);
        Ok(Self {
            config,
            window,
            filters: filters.into(),
            fft,
        })
    }

    /// Computes the finite clip through the same finalization path as streaming.
    pub fn extract(&self, waveform: AudioWaveform<'_>) -> Result<LogMelFeatures, MediaError> {
        self.stream()
            .advance(waveform.samples(), waveform.sample_rate(), true)
    }

    /// Starts an independent input stream.
    pub fn stream(&self) -> PowerMelStream<'_> {
        PowerMelStream {
            transform: Cow::Borrowed(self),
            state: PowerMelState::default(),
        }
    }

    /// Starts an owned stream whose branches share the immutable transform.
    pub fn into_stream(self) -> PowerMelStream<'static> {
        PowerMelStream {
            transform: Cow::Owned(self),
            state: PowerMelState::default(),
        }
    }

    /// Maximum samples that finalize within a nonzero frame budget.
    pub fn max_samples_for_frames(&self, frames: usize) -> Option<usize> {
        frames
            .checked_sub(1)?
            .checked_mul(self.config.hop_length)?
            .checked_add(self.config.fft_length)
    }
}

impl PowerMelState {
    /// Logical bytes retained by a saved window, excluding the transform.
    pub fn snapshot_storage_bytes(&self) -> Option<u64> {
        u64::try_from(self.carry.capacity())
            .ok()?
            .checked_mul(size_of::<f32>() as u64)?
            .checked_add(size_of::<Self>() as u64)
    }
}

impl PowerMelStream<'_> {
    /// Maximum additional samples that emit at most the requested complete windows.
    pub fn available_samples(&self, frames: usize) -> Option<usize> {
        self.transform.available_samples(&self.state, frames)
    }

    /// Processes a PCM chunk with the stream's retained window.
    pub fn advance(
        &mut self,
        samples: &[f32],
        rate: u32,
        finalize: bool,
    ) -> Result<LogMelFeatures, MediaError> {
        self.transform
            .advance(&mut self.state, samples, rate, finalize)
    }
}

impl PowerMel {
    /// Maximum additional samples that emit at most `frames` complete windows.
    /// A partial window may remain in the stream until more samples arrive.
    pub fn available_samples(&self, state: &PowerMelState, frames: usize) -> Option<usize> {
        if state.finished {
            return Some(0);
        }
        frames
            .checked_mul(self.config.hop_length)?
            .checked_add(self.config.fft_length - 1)?
            .checked_sub(state.carry.len())
    }

    /// Emits complete windows and right-pads the final partial window, if any.
    /// A stream with no samples produces one zero-padded window on finalization.
    /// The continuation changes only after the entire feature chunk succeeds.
    pub fn advance(
        &self,
        state: &mut PowerMelState,
        samples: &[f32],
        sample_rate: u32,
        finalize: bool,
    ) -> Result<LogMelFeatures, MediaError> {
        let config = self.config;
        if state.finished {
            return Err(MediaError::invalid("power-mel stream is finalized"));
        }
        if sample_rate != config.sample_rate || samples.iter().any(|value| !value.is_finite()) {
            return Err(MediaError::invalid(
                "power-mel input requires finite PCM at the configured sample rate",
            ));
        }
        let sample_count = state
            .carry
            .len()
            .checked_add(samples.len())
            .ok_or_else(|| MediaError::invalid("power-mel sample count overflowed"))?;
        let overlap = config.fft_length - config.hop_length;
        let complete = if sample_count < config.fft_length {
            0
        } else {
            1 + (sample_count - config.fft_length) / config.hop_length
        };
        let consumed = complete * config.hop_length;
        let frames = complete
            + usize::from(
                finalize
                    && ((!state.emitted && complete == 0) || sample_count - consumed > overlap),
            );
        let value_count = frames
            .checked_mul(config.mel_bins)
            .ok_or_else(|| MediaError::invalid("power-mel feature size overflowed"))?;
        let mut values = Vec::new();
        values.try_reserve_exact(value_count).map_err(|error| {
            MediaError::invalid(format!("power-mel allocation failed: {error}"))
        })?;
        if frames > 0 {
            let transform = self;
            let mut spectrum = vec![Complex32::default(); config.fft_length];
            let mut scratch = vec![Complex32::default(); transform.fft.get_inplace_scratch_len()];
            let bins = config.fft_length / 2 + 1;
            let mut power = vec![0.0; bins];
            for frame in 0..frames {
                let start = frame * config.hop_length;
                for (tap, (value, window)) in
                    spectrum.iter_mut().zip(transform.window.iter()).enumerate()
                {
                    let index = start + tap;
                    let sample = if index < state.carry.len() {
                        state.carry[index]
                    } else {
                        samples
                            .get(index - state.carry.len())
                            .copied()
                            .unwrap_or(0.0)
                    };
                    *value = Complex32::new(sample * window, 0.0);
                }
                transform
                    .fft
                    .process_with_scratch(&mut spectrum, &mut scratch);
                for (energy, bin) in power.iter_mut().zip(&spectrum) {
                    *energy = bin.norm_sqr();
                }
                for filter in transform.filters.chunks_exact(bins) {
                    let energy = filter.iter().zip(&power).map(|(w, p)| w * p).sum::<f32>();
                    let value = (energy.log10().max(config.log_floor)
                        + config.normalization_offset)
                        / config.normalization_divisor;
                    if !energy.is_finite() || !value.is_finite() {
                        return Err(MediaError::invalid(
                            "power-mel feature arithmetic overflowed",
                        ));
                    }
                    values.push(value);
                }
            }
        }
        let carry = if finalize {
            Vec::new()
        } else {
            state
                .carry
                .iter()
                .chain(samples)
                .skip(consumed)
                .copied()
                .collect()
        };
        state.carry = carry;
        state.finished = finalize;
        state.emitted |= frames != 0;
        Ok(LogMelFeatures {
            values,
            mask: vec![true; frames],
            frames,
            mel_bins: config.mel_bins,
        })
    }
}
