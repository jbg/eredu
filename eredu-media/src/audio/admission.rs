//! Streaming level-based admission with bounded lookahead and a continuous PCM clock.

use crate::MediaError;
use std::collections::VecDeque;

/// Energy admission parameters. This detects sustained level, not speaker identity
/// or linguistic content. Quiet material outside admitted windows becomes silence.
#[derive(Debug, Clone, Copy)]
pub struct SpeechAdmissionConfig {
    /// RMS opening level, in -100..=0 dBFS.
    pub threshold_dbfs: f32,
    /// Closing level is this many dB below opening, in 0..=40.
    pub hysteresis_db: f32,
    /// Sustained opening-level duration, in 10..=1000 ms.
    pub min_speech_ms: u32,
    /// Preceding audio retained on opening, in 0..=1000 ms, rounded up to 10 ms.
    pub preroll_ms: u32,
    /// Quiet audio retained before closing, in 0..=2000 ms, rounded up to 10 ms.
    pub hangover_ms: u32,
}
impl Default for SpeechAdmissionConfig {
    fn default() -> Self {
        Self {
            threshold_dbfs: -40.0,
            hysteresis_db: 6.0,
            min_speech_ms: 80,
            preroll_ms: 160,
            hangover_ms: 300,
        }
    }
}

/// Bounded 10 ms energy windows, delayed by pre-roll plus minimum speech duration.
/// Every source sample is returned once, unchanged or zeroed, including at EOF.
pub struct SpeechAdmission {
    frame_len: usize,
    minimum: usize,
    preroll_frames: usize,
    hangover: usize,
    delay_frames: usize,
    opening_energy: f64,
    closing_energy: f64,
    partial: Vec<f32>,
    delayed: VecDeque<(Vec<f32>, bool)>,
    candidate: usize,
    quiet: usize,
    open: bool,
    finished: bool,
}
impl SpeechAdmission {
    /// Creates a bounded gate at a positive sample rate divisible by 100.
    pub fn new(rate: u32, config: SpeechAdmissionConfig) -> Result<Self, MediaError> {
        if rate == 0
            || rate > 384_000
            || rate % 100 != 0
            || !config.threshold_dbfs.is_finite()
            || !(-100.0..=0.0).contains(&config.threshold_dbfs)
            || !config.hysteresis_db.is_finite()
            || !(0.0..=40.0).contains(&config.hysteresis_db)
            || !(10..=1000).contains(&config.min_speech_ms)
            || config.preroll_ms > 1000
            || config.hangover_ms > 2000
        {
            return Err(MediaError::invalid(
                "invalid speech admission configuration",
            ));
        }
        let frame_len = rate as usize / 100;
        let preroll_frames = config.preroll_ms.div_ceil(10) as usize;
        Ok(Self {
            frame_len,
            minimum: (u64::from(rate) * u64::from(config.min_speech_ms)).div_ceil(1000) as usize,
            preroll_frames,
            hangover: config.hangover_ms.div_ceil(10) as usize * frame_len,
            delay_frames: preroll_frames + config.min_speech_ms.div_ceil(10) as usize,
            opening_energy: 10.0_f64.powf(f64::from(config.threshold_dbfs) / 10.0),
            closing_energy: 10.0_f64
                .powf(f64::from(config.threshold_dbfs - config.hysteresis_db) / 10.0),
            partial: Vec::with_capacity(frame_len),
            delayed: VecDeque::new(),
            candidate: 0,
            quiet: 0,
            open: false,
            finished: false,
        })
    }
    /// Fixed lookahead in milliseconds; rounding uses whole 10 ms windows.
    pub fn delay_ms(&self) -> usize {
        self.delay_frames * 10
    }

    /// Consumes finite mono PCM, returning delayed admitted PCM or silence.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>, MediaError> {
        if self.finished || samples.len() > 1_048_576 || samples.iter().any(|v| !v.is_finite()) {
            return Err(MediaError::invalid(
                "invalid or finished speech admission input",
            ));
        }
        let mut output = Vec::with_capacity(samples.len());
        for &sample in samples {
            self.partial.push(sample);
            if self.partial.len() == self.frame_len {
                let frame =
                    std::mem::replace(&mut self.partial, Vec::with_capacity(self.frame_len));
                self.process(frame);
                if self.delayed.len() > self.delay_frames {
                    self.emit(&mut output);
                }
            }
        }
        Ok(output)
    }
    fn process(&mut self, frame: Vec<f32>) {
        let energy = frame.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / frame.len() as f64;
        let mut admit = false;
        let mut opened = false;
        if self.open {
            self.quiet = if energy >= self.closing_energy {
                0
            } else {
                self.quiet + frame.len()
            };
            admit = self.quiet <= self.hangover;
            if self.quiet != 0 && self.quiet >= self.hangover {
                self.open = false;
            }
        } else {
            self.candidate = if energy >= self.opening_energy {
                self.candidate + frame.len()
            } else {
                0
            };
            if self.candidate >= self.minimum {
                self.open = true;
                self.quiet = 0;
                admit = true;
                opened = true;
            }
        }
        self.delayed.push_back((frame, admit));
        if opened {
            let count = self.preroll_frames + self.candidate.div_ceil(self.frame_len);
            for (_, admitted) in self.delayed.iter_mut().rev().take(count) {
                *admitted = true;
            }
            self.candidate = 0;
        }
    }
    fn emit(&mut self, output: &mut Vec<f32>) {
        let (mut frame, admitted) = self.delayed.pop_front().expect("queued admission frame");
        if !admitted {
            frame.fill(0.0);
        }
        output.extend(frame);
    }
    /// Flushes remaining real samples without adding lookahead or frame padding.
    pub fn finish(&mut self) -> Result<Vec<f32>, MediaError> {
        if self.finished {
            return Err(MediaError::invalid("speech admission is already finished"));
        }
        self.finished = true;
        if !self.partial.is_empty() {
            let frame = std::mem::take(&mut self.partial);
            self.process(frame);
        }
        let mut output = Vec::new();
        while !self.delayed.is_empty() {
            self.emit(&mut output);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sustained_level_admits_preroll_and_hangover_but_rejects_isolated_peaks() {
        let mut pcm = vec![0.001; 14 * 160 + 21];
        for frame in [5, 8, 9, 12] {
            pcm[frame * 160..(frame + 1) * 160].fill(0.1);
        }
        let config = SpeechAdmissionConfig {
            threshold_dbfs: -30.0,
            min_speech_ms: 20,
            preroll_ms: 20,
            hangover_ms: 20,
            ..Default::default()
        };
        let run = |chunk| {
            let mut gate = SpeechAdmission::new(16_000, config).unwrap();
            let mut out = Vec::new();
            for part in pcm.chunks(chunk) {
                out.extend(gate.push(part).unwrap());
            }
            out.extend(gate.finish().unwrap());
            out
        };
        let mut expected = vec![0.0; pcm.len()];
        expected[6 * 160..12 * 160].copy_from_slice(&pcm[6 * 160..12 * 160]);
        assert_eq!(run(13), expected);
        assert_eq!(run(640), expected);
    }
    #[test]
    fn hysteresis_retains_quieter_speech_and_partial_eof_never_qualifies_early() {
        let config = SpeechAdmissionConfig {
            threshold_dbfs: -20.0,
            min_speech_ms: 20,
            preroll_ms: 0,
            hangover_ms: 0,
            hysteresis_db: 6.0,
        };
        let mut gate = SpeechAdmission::new(16_000, config).unwrap();
        let mut pcm = vec![0.2; 320];
        pcm.extend([0.07; 160]); // Below opening level, above closing level.
        pcm.extend([0.0; 160]);
        let mut out = gate.push(&pcm).unwrap();
        out.extend(gate.finish().unwrap());
        assert_eq!(out, pcm);
        let mut gate = SpeechAdmission::new(16_000, config).unwrap();
        let mut out = gate.push(&[0.2; 319]).unwrap();
        out.extend(gate.finish().unwrap());
        assert_eq!(out, vec![0.0; 319]);
    }
}
