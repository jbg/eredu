//! Host acoustic echo cancellation with an independent playback reference.

use crate::MediaError;
use webrtc_audio_processing::{config::EchoCanceller as WebRtcEcho, Config, Processor};

/// Mono WebRTC AEC3 processing in 10 ms frames. Playback is analyzed without
/// modification; capture remains live during playback for double-talk/barge-in.
/// No noise suppression or automatic gain control is enabled.
pub struct EchoCanceller {
    processor: Processor,
    frame_samples: usize,
}

impl EchoCanceller {
    /// Opens an independent adaptive echo filter at 16, 32 or 48 kHz.
    pub fn new(sample_rate: u32) -> Result<Self, MediaError> {
        if !matches!(sample_rate, 16_000 | 32_000 | 48_000) {
            return Err(MediaError::invalid(
                "echo cancellation requires 16, 32 or 48 kHz",
            ));
        }
        let processor = Processor::new(sample_rate)
            .map_err(|e| MediaError::invalid(format!("echo cancellation: {e}")))?;
        processor.set_config(Config {
            echo_canceller: Some(WebRtcEcho::Full {
                stream_delay_ms: None,
            }),
            high_pass_filter: Some(Default::default()),
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
            return Err(MediaError::invalid("invalid echo-cancellation frame"));
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

    /// Removes estimated speaker echo from microphone PCM in place.
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
    fn delayed_echo_is_reduced_and_independent_speech_survives_double_talk() {
        let rate = 16_000;
        let mut canceller = EchoCanceller::new(rate).unwrap();
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
                        let phase = i as f64 * std::f64::consts::TAU * frequency / f64::from(rate);
                        projection[0] += f64::from(x) * phase.sin();
                        projection[1] += f64::from(x) * phase.cos();
                    }
                }
            }
        }
        let reduction_db = 10.0 * (before / after).log10();
        let retained = capture_energy / near_energy;
        eprintln!("echo reduction={reduction_db:.1} dB, double-talk energy ratio={retained:.3}");
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
