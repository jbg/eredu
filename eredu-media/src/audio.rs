//! Shared PCM validation and log-mel feature extraction.

#[cfg(feature = "audio")]
use rustfft::{num_complex::Complex32, FftPlanner};

use crate::{MediaError, ProcessedMediaDtype};

/// Memory order of a log-mel feature buffer.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LogMelLayout {
    /// Contiguous row-major `[frames, mel_bins]` order.
    FramesMelBins,
}

/// Borrowed mono floating-point PCM waveform.
#[derive(Debug, Clone, Copy)]
pub struct AudioWaveform<'a> {
    samples: &'a [f32],
    sample_rate: u32,
}

impl<'a> AudioWaveform<'a> {
    /// Validates and creates a mono PCM waveform.
    pub fn new(samples: &'a [f32], sample_rate: u32) -> Result<Self, MediaError> {
        if samples.is_empty() {
            return Err(MediaError::invalid("audio waveform must not be empty"));
        }
        if sample_rate == 0 {
            return Err(MediaError::invalid("audio sample rate must be positive"));
        }
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(MediaError::invalid(
                "audio waveform samples must all be finite",
            ));
        }
        Ok(Self {
            samples,
            sample_rate,
        })
    }

    /// Returns the PCM samples.
    pub fn samples(self) -> &'a [f32] {
        self.samples
    }

    /// Returns the sampling rate in hertz.
    pub fn sample_rate(self) -> u32 {
        self.sample_rate
    }
}

/// Model-independent log-mel extraction parameters.
#[derive(Debug, Clone)]
pub struct LogMelConfig {
    /// Required input sampling rate.
    pub sample_rate: u32,
    /// Analysis frame length in samples.
    pub frame_length: usize,
    /// Frame step in samples.
    pub hop_length: usize,
    /// FFT length, at least `frame_length`.
    pub fft_length: usize,
    /// Number of HTK mel filters.
    pub mel_bins: usize,
    /// Lowest filter frequency.
    pub min_frequency: f32,
    /// Highest filter frequency.
    pub max_frequency: f32,
    /// Additive floor before taking the natural logarithm.
    pub mel_floor: f32,
    /// Maximum waveform length before truncation.
    pub max_samples: usize,
    /// Waveform padding multiple.
    pub pad_to_multiple: usize,
}

/// Owned model-ready features and their valid-frame mask.
#[derive(Debug, Clone, PartialEq)]
pub struct LogMelFeatures {
    /// Row-major `[frames, mel_bins]` feature values.
    pub values: Vec<f32>,
    /// True for frames whose analysis endpoint is real audio.
    pub mask: Vec<bool>,
    /// Number of feature frames.
    pub frames: usize,
    /// Number of mel bins.
    pub mel_bins: usize,
}

impl LogMelFeatures {
    /// Scalar dtype of every feature value.
    pub const fn dtype(&self) -> ProcessedMediaDtype {
        ProcessedMediaDtype::F32
    }

    /// Explicit feature memory layout.
    pub const fn layout(&self) -> LogMelLayout {
        LogMelLayout::FramesMelBins
    }

    /// Logical `[frames, mel_bins]` shape.
    pub const fn shape(&self) -> [usize; 2] {
        [self.frames, self.mel_bins]
    }

    /// Valid-frame mask paired one-for-one with the first shape dimension.
    pub fn frame_mask(&self) -> &[bool] {
        &self.mask
    }
}

/// Model-independent leading-zero Slaney log-mel extraction parameters.
#[derive(Debug, Clone)]
pub struct LeadingSlaneyLogMelConfig {
    /// Required input sampling rate.
    pub sample_rate: u32,
    /// FFT and periodic-Hann window length.
    pub fft_length: usize,
    /// Frame step in samples.
    pub hop_length: usize,
    /// Zero-valued samples prepended before the waveform.
    pub leading_zeros: usize,
    /// Number of area-normalized Slaney mel filters.
    pub mel_bins: usize,
    /// Lowest filter frequency.
    pub min_frequency: f32,
    /// Highest filter frequency.
    pub max_frequency: f32,
    /// Energy floor before taking the base-ten logarithm.
    pub energy_floor: f32,
}

/// Extracts semicausal periodic-Hann HTK log-mel features.
pub fn extract_log_mel(
    waveform: AudioWaveform<'_>,
    config: &LogMelConfig,
) -> Result<LogMelFeatures, MediaError> {
    #[cfg(feature = "audio")]
    {
        extract_log_mel_enabled(waveform, config)
    }
    #[cfg(not(feature = "audio"))]
    {
        let _ = (waveform, config);
        Err(MediaError::invalid(
            "HTK log-mel extraction requires the `audio` feature",
        ))
    }
}

#[cfg(feature = "audio")]
fn extract_log_mel_enabled(
    waveform: AudioWaveform<'_>,
    config: &LogMelConfig,
) -> Result<LogMelFeatures, MediaError> {
    if waveform.sample_rate != config.sample_rate {
        return Err(MediaError::invalid(format!(
            "audio processor requires {} Hz PCM, got {} Hz",
            config.sample_rate, waveform.sample_rate
        )));
    }
    if config.frame_length == 0
        || config.hop_length == 0
        || config.fft_length < config.frame_length
        || config.fft_length < 2
        || config.mel_bins == 0
        || config.pad_to_multiple == 0
        || config.max_samples == 0
        || !config.min_frequency.is_finite()
        || !config.max_frequency.is_finite()
        || config.min_frequency < 0.0
        || config.max_frequency <= config.min_frequency
        || config.max_frequency > config.sample_rate as f32 / 2.0
        || !config.mel_floor.is_finite()
        || config.mel_floor <= 0.0
    {
        return Err(MediaError::invalid(
            "invalid log-mel processor configuration",
        ));
    }

    let real_samples = waveform.samples.len().min(config.max_samples);
    let padded_samples = real_samples
        .div_ceil(config.pad_to_multiple)
        .checked_mul(config.pad_to_multiple)
        .ok_or_else(|| MediaError::invalid("audio padding geometry overflowed"))?;
    let left_padding = config.frame_length / 2;
    let frame_span = config.frame_length + 1;
    let total = left_padding
        .checked_add(padded_samples)
        .ok_or_else(|| MediaError::invalid("audio framing geometry overflowed"))?;
    let frames = total.saturating_sub(frame_span) / config.hop_length + 1;
    let mut padded = vec![0.0f32; total];
    padded[left_padding..left_padding + real_samples]
        .copy_from_slice(&waveform.samples[..real_samples]);

    let window = (0..config.frame_length)
        .map(|index| {
            0.5 - 0.5
                * (2.0 * std::f32::consts::PI * index as f32 / config.frame_length as f32).cos()
        })
        .collect::<Vec<_>>();
    let mel_filters = htk_mel_filters(config);
    let frequency_bins = config.fft_length / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.fft_length);
    let mut spectrum = vec![Complex32::default(); config.fft_length];
    let mut values = vec![0.0f32; frames * config.mel_bins];
    let mut mask = vec![false; frames];

    for frame in 0..frames {
        spectrum.fill(Complex32::default());
        let start = frame * config.hop_length;
        for index in 0..config.frame_length {
            spectrum[index].re = padded[start + index] * window[index];
        }
        fft.process(&mut spectrum);
        for mel in 0..config.mel_bins {
            let mut magnitude = 0.0f32;
            for frequency in 0..frequency_bins {
                magnitude +=
                    spectrum[frequency].norm() * mel_filters[frequency * config.mel_bins + mel];
            }
            values[frame * config.mel_bins + mel] = (magnitude + config.mel_floor).ln();
        }
        let endpoint = start + frame_span - 1;
        mask[frame] = endpoint >= left_padding && endpoint < left_padding + real_samples;
        if !mask[frame] {
            values[frame * config.mel_bins..(frame + 1) * config.mel_bins].fill(0.0);
        }
    }

    Ok(LogMelFeatures {
        values,
        mask,
        frames,
        mel_bins: config.mel_bins,
    })
}

/// Extracts leading-zero periodic-Hann, magnitude, Slaney log-mel features.
pub fn extract_leading_slaney_log_mel(
    waveform: AudioWaveform<'_>,
    config: &LeadingSlaneyLogMelConfig,
) -> Result<LogMelFeatures, MediaError> {
    #[cfg(feature = "audio")]
    {
        extract_leading_slaney_log_mel_enabled(waveform, config)
    }
    #[cfg(not(feature = "audio"))]
    {
        let _ = (waveform, config);
        Err(MediaError::invalid(
            "Slaney log-mel extraction requires the `audio` feature",
        ))
    }
}

#[cfg(feature = "audio")]
fn extract_leading_slaney_log_mel_enabled(
    waveform: AudioWaveform<'_>,
    config: &LeadingSlaneyLogMelConfig,
) -> Result<LogMelFeatures, MediaError> {
    if waveform.sample_rate != config.sample_rate {
        return Err(MediaError::invalid(format!(
            "audio processor requires {} Hz PCM, got {} Hz",
            config.sample_rate, waveform.sample_rate
        )));
    }
    if config.fft_length == 0
        || config.hop_length == 0
        || config.mel_bins == 0
        || !config.min_frequency.is_finite()
        || !config.max_frequency.is_finite()
        || config.min_frequency < 0.0
        || config.max_frequency <= config.min_frequency
        || config.max_frequency > config.sample_rate as f32 / 2.0
        || !config.energy_floor.is_finite()
        || config.energy_floor <= 0.0
    {
        return Err(MediaError::invalid(
            "invalid leading Slaney log-mel configuration",
        ));
    }

    let frames = waveform.samples.len().div_ceil(config.hop_length);
    let padded_len = frames
        .checked_sub(1)
        .map_or(config.leading_zeros, |last_frame| {
            config.leading_zeros.max(
                last_frame
                    .saturating_mul(config.hop_length)
                    .saturating_add(config.fft_length),
            )
        });
    let mut padded = vec![0.0f32; padded_len];
    let waveform_end = config
        .leading_zeros
        .checked_add(waveform.samples.len())
        .ok_or_else(|| MediaError::invalid("audio framing geometry overflowed"))?;
    if waveform_end > padded.len() {
        return Err(MediaError::invalid(
            "audio framing does not contain the waveform",
        ));
    }
    padded[config.leading_zeros..waveform_end].copy_from_slice(waveform.samples);

    let filters = slaney_mel_filters(config);
    let frequency_bins = config.fft_length / 2 + 1;
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(config.fft_length);
    let mut spectrum = vec![Complex32::default(); config.fft_length];
    let mut values = vec![0.0f32; frames * config.mel_bins];
    for frame in 0..frames {
        spectrum.fill(Complex32::default());
        let start = frame * config.hop_length;
        for index in 0..config.fft_length {
            let window = 0.5
                - 0.5
                    * (2.0 * std::f32::consts::PI * index as f32 / config.fft_length as f32).cos();
            spectrum[index].re = padded[start + index] * window;
        }
        fft.process(&mut spectrum);
        for mel in 0..config.mel_bins {
            let mut energy = 0.0f32;
            for frequency in 0..frequency_bins {
                energy += spectrum[frequency].norm() * filters[mel * frequency_bins + frequency];
            }
            values[frame * config.mel_bins + mel] = energy.max(config.energy_floor).log10();
        }
    }
    Ok(LogMelFeatures {
        values,
        mask: vec![true; frames],
        frames,
        mel_bins: config.mel_bins,
    })
}

#[cfg(feature = "audio")]
fn htk_mel_filters(config: &LogMelConfig) -> Vec<f32> {
    let frequency_bins = config.fft_length / 2 + 1;
    let hertz_to_mel = |frequency: f32| 2595.0 * (1.0 + frequency / 700.0).log10();
    let mel_to_hertz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);
    let mel_min = hertz_to_mel(config.min_frequency);
    let mel_max = hertz_to_mel(config.max_frequency);
    let centers = (0..config.mel_bins + 2)
        .map(|index| {
            let mel = mel_min + (mel_max - mel_min) * index as f32 / (config.mel_bins + 1) as f32;
            mel_to_hertz(mel)
        })
        .collect::<Vec<_>>();
    let mut filters = vec![0.0f32; frequency_bins * config.mel_bins];
    for frequency in 0..frequency_bins {
        let hertz =
            config.sample_rate as f32 * 0.5 * frequency as f32 / (frequency_bins - 1) as f32;
        for mel in 0..config.mel_bins {
            let down = (hertz - centers[mel]) / (centers[mel + 1] - centers[mel]);
            let up = (centers[mel + 2] - hertz) / (centers[mel + 2] - centers[mel + 1]);
            filters[frequency * config.mel_bins + mel] = down.min(up).max(0.0);
        }
    }
    filters
}

#[cfg(feature = "audio")]
fn slaney_mel_filters(config: &LeadingSlaneyLogMelConfig) -> Vec<f32> {
    let hz_to_mel = |hz: f64| {
        if hz < 1_000.0 {
            hz / (200.0 / 3.0)
        } else {
            15.0 + (hz / 1_000.0).ln() / (6.4f64.ln() / 27.0)
        }
    };
    let mel_to_hz = |mel: f64| {
        if mel < 15.0 {
            mel * (200.0 / 3.0)
        } else {
            1_000.0 * ((mel - 15.0) * (6.4f64.ln() / 27.0)).exp()
        }
    };
    let mel_min = hz_to_mel(config.min_frequency as f64);
    let mel_max = hz_to_mel(config.max_frequency as f64);
    let edges = (0..config.mel_bins + 2)
        .map(|index| {
            mel_to_hz(mel_min + (mel_max - mel_min) * index as f64 / (config.mel_bins + 1) as f64)
        })
        .collect::<Vec<_>>();
    let frequency_bins = config.fft_length / 2 + 1;
    let mut filters = vec![0.0f32; config.mel_bins * frequency_bins];
    for mel in 0..config.mel_bins {
        let normalization = 2.0 / (edges[mel + 2] - edges[mel]);
        for frequency in 0..frequency_bins {
            let hz = config.sample_rate as f64 * frequency as f64 / config.fft_length as f64;
            let lower = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
            let upper = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
            filters[mel * frequency_bins + frequency] =
                (lower.min(upper).max(0.0) * normalization) as f32;
        }
    }
    filters
}

#[cfg(all(test, feature = "audio"))]
mod tests {
    use super::{
        extract_leading_slaney_log_mel, extract_log_mel, AudioWaveform, LeadingSlaneyLogMelConfig,
        LogMelConfig,
    };

    #[test]
    fn validates_waveform_and_extracts_masked_features() {
        let samples = vec![0.0f32; 16_000];
        let waveform = AudioWaveform::new(&samples, 16_000).unwrap();
        let features = extract_log_mel(
            waveform,
            &LogMelConfig {
                sample_rate: 16_000,
                frame_length: 320,
                hop_length: 160,
                fft_length: 512,
                mel_bins: 128,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
                mel_floor: 1e-3,
                max_samples: 480_000,
                pad_to_multiple: 128,
            },
        )
        .unwrap();
        assert_eq!(features.frames, 99);
        assert_eq!(features.values.len(), 99 * 128);
        assert!(features.mask.iter().all(|valid| *valid));
        assert!(features.values.iter().all(|value| value.is_finite()));
        assert!(features
            .values
            .iter()
            .all(|value| (*value - 1e-3_f32.ln()).abs() < 1e-6));
    }

    #[test]
    fn nonzero_impulse_has_stable_htk_log_mel_values() {
        let samples = [1.0f32, 0.0, 0.0, 0.0];
        let features = extract_log_mel(
            AudioWaveform::new(&samples, 8).unwrap(),
            &LogMelConfig {
                sample_rate: 8,
                frame_length: 4,
                hop_length: 1,
                fft_length: 4,
                mel_bins: 1,
                min_frequency: 0.0,
                max_frequency: 4.0,
                mel_floor: 1e-6,
                max_samples: 4,
                pad_to_multiple: 4,
            },
        )
        .unwrap();
        assert_eq!(features.frames, 2);
        assert_eq!(features.mask, [true, true]);
        assert_eq!(features.values.len(), 2);
        assert!((features.values[0] - -0.001_431_296_9).abs() < 1e-6);
        assert!((features.values[1] - -0.694_577_46).abs() < 1e-6);
    }

    #[test]
    fn leading_slaney_frontend_uses_input_divided_by_hop_frames() {
        let samples = vec![0.0f32; 801];
        let features = extract_leading_slaney_log_mel(
            AudioWaveform::new(&samples, 16_000).unwrap(),
            &LeadingSlaneyLogMelConfig {
                sample_rate: 16_000,
                fft_length: 1_600,
                hop_length: 800,
                leading_zeros: 800,
                mel_bins: 80,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
                energy_floor: 1e-10,
            },
        )
        .unwrap();
        assert_eq!(features.frames, 2);
        assert_eq!(features.values.len(), 2 * 80);
        assert!(features
            .values
            .iter()
            .all(|value| (*value + 10.0).abs() < 1e-6));
        assert!(features.mask.iter().all(|valid| *valid));
    }

    #[test]
    fn rejects_invalid_waveforms_and_configs() {
        assert!(AudioWaveform::new(&[], 16_000).is_err());
        assert!(AudioWaveform::new(&[0.0], 0).is_err());
        assert!(AudioWaveform::new(&[f32::NAN], 16_000).is_err());

        let waveform = AudioWaveform::new(&[0.0; 320], 8_000).unwrap();
        assert!(extract_log_mel(
            waveform,
            &LogMelConfig {
                sample_rate: 16_000,
                frame_length: 320,
                hop_length: 160,
                fft_length: 512,
                mel_bins: 128,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
                mel_floor: 1e-3,
                max_samples: 480_000,
                pad_to_multiple: 128,
            }
        )
        .is_err());

        let waveform = AudioWaveform::new(&[0.0; 320], 16_000).unwrap();
        assert!(extract_leading_slaney_log_mel(
            waveform,
            &LeadingSlaneyLogMelConfig {
                sample_rate: 16_000,
                fft_length: 1_600,
                hop_length: 0,
                leading_zeros: 800,
                mel_bins: 80,
                min_frequency: 0.0,
                max_frequency: 8_000.0,
                energy_floor: 1e-10,
            }
        )
        .is_err());
    }
}
