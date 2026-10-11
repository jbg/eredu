//! Host echo cancellation and configurable WebRTC noise suppression.

use crate::MediaError;
use webrtc_audio_processing::{config::EchoCanceller as WebRtcEcho, Config, Processor};

/// Capture noise suppression strength; stronger settings can distort speech.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum NoiseSuppression {
    #[default]
    /// Disable noise suppression.
    Off,
    /// Mild attenuation with less speech distortion.
    Low,
    /// Moderate attenuation.
    Moderate,
    /// Strong attenuation.
    High,
    /// Strongest attenuation, with greater risk of speech distortion.
    VeryHigh,
}
impl NoiseSuppression {
    fn native(self) -> Option<webrtc_audio_processing::config::NoiseSuppressionLevel> {
        use webrtc_audio_processing::config::NoiseSuppressionLevel as Level;
        match self {
            Self::Off => None,
            Self::Low => Some(Level::Low),
            Self::Moderate => Some(Level::Moderate),
            Self::High => Some(Level::High),
            Self::VeryHigh => Some(Level::VeryHigh),
        }
    }
}

/// Optional host filtering without gain normalization.
#[derive(Debug, Default, Clone, Copy)]
pub struct CaptureProcessingConfig {
    /// Enable full AEC3; a matching rendered reference must be supplied.
    pub echo_cancellation: bool,
    /// Capture-side stationary noise suppression strength.
    pub noise_suppression: NoiseSuppression,
}

/// Mono WebRTC processing in 10 ms frames, with optional echo cancellation and
/// noise suppression. Automatic gain control is disabled.
pub struct CaptureProcessor {
    processor: Processor,
    frame_samples: usize,
}

impl CaptureProcessor {
    /// Opens an independent capture processor at 16, 32 or 48 kHz.
    pub fn new(sample_rate: u32, config: CaptureProcessingConfig) -> Result<Self, MediaError> {
        if !matches!(sample_rate, 16_000 | 32_000 | 48_000) {
            return Err(MediaError::invalid(
                "capture processing requires 16, 32 or 48 kHz",
            ));
        }
        let processor = Processor::new(sample_rate)
            .map_err(|e| MediaError::invalid(format!("capture processing: {e}")))?;
        processor.set_config(Config {
            echo_canceller: config.echo_cancellation.then_some(WebRtcEcho::Full {
                stream_delay_ms: None,
            }),
            high_pass_filter: Some(Default::default()),
            noise_suppression: config.noise_suppression.native().map(|level| {
                webrtc_audio_processing::config::NoiseSuppression {
                    level,
                    ..Default::default()
                }
            }),
            ..Config::default()
        });
        Ok(Self {
            processor,
            frame_samples: sample_rate as usize / 100,
        })
    }

    /// Number of mono samples in each required 10 ms frame.
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    fn validate(&self, samples: &[f32]) -> Result<(), MediaError> {
        if samples.len() != self.frame_samples || samples.iter().any(|s| !s.is_finite()) {
            return Err(MediaError::invalid("invalid capture-processing frame"));
        }
        Ok(())
    }

    /// Analyzes actual rendered PCM, including intentional silence and underruns.
    /// Submit reference frames in playback order before corresponding capture.
    pub fn analyze_render(&mut self, samples: &[f32]) -> Result<(), MediaError> {
        self.validate(samples)?;
        self.processor
            .analyze_render_frame([samples])
            .map_err(|e| MediaError::invalid(format!("echo reference: {e}")))
    }

    /// Applies configured echo/noise suppression to microphone PCM in place.
    pub fn process_capture(&mut self, samples: &mut [f32]) -> Result<(), MediaError> {
        self.validate(samples)?;
        self.processor
            .process_capture_frame([samples])
            .map_err(|e| MediaError::invalid(format!("echo capture: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_suppression_attenuates_stationary_noise_without_a_render_stream() {
        let mut processor = CaptureProcessor::new(
            16_000,
            CaptureProcessingConfig {
                noise_suppression: NoiseSuppression::High,
                ..Default::default()
            },
        )
        .unwrap();
        let mut seed = 23_u32;
        let (mut before, mut after, mut sine, mut cosine) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        for start in (0..96_000).step_by(160) {
            let mut mic = [0.0; 160];
            for (j, sample) in mic.iter_mut().enumerate() {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let i = start + j;
                *sample = ((seed >> 8) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.04;
                if i >= 64_000 {
                    *sample += 0.12 * (i as f32 * std::f32::consts::TAU * 440.0 / 16_000.0).sin();
                }
                if (32_000..64_000).contains(&i) {
                    before += f64::from(*sample).powi(2);
                }
            }
            processor.process_capture(&mut mic).unwrap();
            for (j, &sample) in mic.iter().enumerate() {
                let i = start + j;
                if (32_000..64_000).contains(&i) {
                    after += f64::from(sample).powi(2);
                }
                if i >= 72_000 {
                    let phase = i as f64 * std::f64::consts::TAU * 440.0 / 16_000.0;
                    sine += f64::from(sample) * phase.sin();
                    cosine += f64::from(sample) * phase.cos();
                }
            }
        }
        let attenuation = 10.0 * (before / after).log10();
        let amplitude = 2.0 * sine.hypot(cosine) / 24_000.0;
        eprintln!("stationary noise attenuation={attenuation:.1} dB, near-end tone amplitude={amplitude:.3}");
        assert!(attenuation > 8.0, "noise attenuation {attenuation} dB");
        assert!(
            amplitude > 0.06 && amplitude < 0.14,
            "near-end tone {amplitude}"
        );
    }

    #[test]
    fn delayed_echo_is_reduced_and_independent_speech_survives_double_talk() {
        for noise_suppression in [NoiseSuppression::Off, NoiseSuppression::Moderate] {
            let rate = 16_000;
            let mut canceller = CaptureProcessor::new(
                rate,
                CaptureProcessingConfig {
                    echo_cancellation: true,
                    noise_suppression,
                },
            )
            .unwrap();
            let count = rate as usize * 14;
            let delay = 592; // 37 ms acoustic path, plus a weaker 5 ms reflection.
            let mut seed = 7_u32;
            let mut far = vec![0.0_f32; count];
            for x in &mut far {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                *x = ((seed >> 8) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.4;
            }
            let mut before = 0.0;
            let mut after = 0.0;
            let mut near_energy = 0.0;
            let mut capture_energy = 0.0;
            let mut projections = [[0.0_f64; 2]; 2];
            for start in (0..count).step_by(160) {
                canceller.analyze_render(&far[start..start + 160]).unwrap();
                let mut mic = [0.0; 160];
                for (j, x) in mic.iter_mut().enumerate() {
                    let i = start + j;
                    let echo = if i >= delay {
                        0.5 * far[i - delay]
                    } else {
                        0.0
                    } + if i >= delay + 80 {
                        0.2 * far[i - delay - 80]
                    } else {
                        0.0
                    };
                    let near = if i >= 10 * rate as usize {
                        0.12 * (i as f32 * std::f32::consts::TAU * 440.0 / rate as f32).sin()
                            + 0.08 * (i as f32 * std::f32::consts::TAU * 710.0 / rate as f32).sin()
                    } else {
                        0.0
                    };
                    *x = echo + near;
                    if (7 * rate as usize..10 * rate as usize).contains(&i) {
                        before += f64::from(echo).powi(2);
                    }
                    if i >= 11 * rate as usize {
                        near_energy += f64::from(near).powi(2);
                    }
                }
                canceller.process_capture(&mut mic).unwrap();
                for (j, &x) in mic.iter().enumerate() {
                    let i = start + j;
                    if (7 * rate as usize..10 * rate as usize).contains(&i) {
                        after += f64::from(x).powi(2);
                    }
                    if i >= 11 * rate as usize {
                        capture_energy += f64::from(x).powi(2);
                        for (projection, frequency) in projections.iter_mut().zip([440.0, 710.0]) {
                            let phase =
                                i as f64 * std::f64::consts::TAU * frequency / f64::from(rate);
                            projection[0] += f64::from(x) * phase.sin();
                            projection[1] += f64::from(x) * phase.cos();
                        }
                    }
                }
            }
            let reduction_db = 10.0 * (before / after).log10();
            let retained = capture_energy / near_energy;
            eprintln!("noise={noise_suppression:?}, echo reduction={reduction_db:.1} dB, double-talk energy ratio={retained:.3}");
            assert!(reduction_db > 15.0, "echo reduction {reduction_db} dB");
            assert!(
                retained > 0.3 && retained < 1.5,
                "independent speech energy ratio {retained}"
            );
            for (projection, amplitude) in projections.iter().zip([0.12, 0.08]) {
                let recovered = 2.0 * projection[0].hypot(projection[1]) / (3 * rate) as f64;
                assert!(
                    recovered > amplitude * 0.5,
                    "near-end component {recovered} versus {amplitude}"
                );
            }
        }
    }
}
