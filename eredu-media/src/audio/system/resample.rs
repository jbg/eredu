use super::SystemAudioError;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Bounded mono band-limited conversion with phase retained across host chunks.
/// Equal-rate input is passed through exactly. Finish returns the delayed suffix
/// and trims padding to the rationally computed output sample count.
pub struct PcmResampler {
    from: u32,
    to: u32,
    sinc: Option<SincFixedIn<f32>>,
    input: Vec<f32>,
    output: Vec<Vec<f32>>,
    received: u64,
    emitted: u64,
    finished: bool,
}
impl PcmResampler {
    /// Creates a converter for rates between 8 kHz and 384 kHz.
    pub fn new(from: u32, to: u32) -> Result<Self, SystemAudioError> {
        if from < 8_000 || to < 8_000 || from > 384_000 || to > 384_000 {
            return Err(SystemAudioError::Invalid(
                "sample rates must be in 8000..=384000",
            ));
        }
        let sinc = if from == to {
            None
        } else {
            Some(
                SincFixedIn::new(
                    to as f64 / from as f64,
                    1.0,
                    SincInterpolationParameters {
                        sinc_len: 128,
                        f_cutoff: 0.94,
                        interpolation: SincInterpolationType::Linear,
                        oversampling_factor: 128,
                        window: WindowFunction::BlackmanHarris2,
                    },
                    256,
                    1,
                )
                .map_err(|e| SystemAudioError::Resample(e.to_string()))?,
            )
        };
        let output = sinc
            .as_ref()
            .map_or_else(|| vec![Vec::new()], |s| s.output_buffer_allocate(true));
        Ok(Self {
            from,
            to,
            sinc,
            input: Vec::with_capacity(256),
            output,
            received: 0,
            emitted: 0,
            finished: false,
        })
    }
    /// Converts a finite bounded chunk; work runs on the host caller, never an audio callback.
    pub fn push(&mut self, samples: &[f32]) -> Result<Vec<f32>, SystemAudioError> {
        if self.finished || samples.len() > 1_048_576 || samples.iter().any(|v| !v.is_finite()) {
            return Err(SystemAudioError::Invalid(
                "invalid or finished resampler input",
            ));
        }
        self.received = self
            .received
            .checked_add(samples.len() as u64)
            .ok_or(SystemAudioError::Invalid("sample clock overflow"))?;
        if self.sinc.is_none() {
            self.emitted += samples.len() as u64;
            return Ok(samples.to_vec());
        }
        let mut result = Vec::new();
        for &sample in samples {
            self.input.push(sample);
            if self.input.len() == 256 {
                self.convert(&mut result)?;
            }
        }
        Ok(result)
    }
    fn convert(&mut self, result: &mut Vec<f32>) -> Result<(), SystemAudioError> {
        let (_, count) = self
            .sinc
            .as_mut()
            .unwrap()
            .process_into_buffer(&[self.input.as_slice()], &mut self.output, None)
            .map_err(|e| SystemAudioError::Resample(e.to_string()))?;
        result.extend_from_slice(&self.output[0][..count]);
        self.emitted += count as u64;
        self.input.clear();
        Ok(())
    }
    /// Flushes delayed real samples exactly once, without exposing zero padding.
    pub fn finish(&mut self) -> Result<Vec<f32>, SystemAudioError> {
        if self.finished {
            return Err(SystemAudioError::Invalid("resampler is already finished"));
        }
        self.finished = true;
        if self.sinc.is_none() {
            return Ok(Vec::new());
        }
        let target = self
            .received
            .checked_mul(self.to as u64)
            .ok_or(SystemAudioError::Invalid("sample clock overflow"))?
            .div_ceil(self.from as u64);
        let remaining = target.saturating_sub(self.emitted) as usize;
        let mut result = Vec::new();
        while result.len() < remaining {
            self.input.resize(256, 0.0);
            self.convert(&mut result)?;
        }
        result.truncate(remaining);
        self.emitted = target;
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fragmented_conversion_preserves_phase_duration_and_antialiasing() {
        let input = (0..48_000)
            .map(|i| (i as f32 * std::f32::consts::TAU * 1000.0 / 48000.0).sin())
            .collect::<Vec<_>>();
        let convert = |chunk| {
            let mut r = PcmResampler::new(48_000, 16_000).unwrap();
            let mut out = Vec::new();
            for part in input.chunks(chunk) {
                out.extend(r.push(part).unwrap());
            }
            out.extend(r.finish().unwrap());
            out
        };
        let a = convert(137);
        let b = convert(1024);
        assert_eq!(a, b);
        assert_eq!(a.len(), 16000);
        let tone = &a[100..15900];
        let (mut sine, mut cosine) = (0.0_f64, 0.0_f64);
        for (i, &sample) in tone.iter().enumerate() {
            let phase = i as f64 * std::f64::consts::TAU * 1000.0 / 16000.0;
            sine += sample as f64 * phase.sin();
            cosine += sample as f64 * phase.cos();
        }
        let amplitude = 2.0 * sine.hypot(cosine) / tone.len() as f64;
        assert!(
            (amplitude - 1.0).abs() < 0.01,
            "in-band amplitude {amplitude}"
        );
        let mut r = PcmResampler::new(48_000, 16_000).unwrap();
        let high = (0..48000)
            .map(|i| (i as f32 * std::f32::consts::TAU * 12000.0 / 48000.0).sin())
            .collect::<Vec<_>>();
        let mut out = r.push(&high).unwrap();
        out.extend(r.finish().unwrap());
        let energy = out[100..15900].iter().map(|v| v * v).sum::<f32>() / 15800.0;
        assert!(energy < 0.001, "aliased energy {energy}");
    }
    #[test]
    fn identity_and_odd_length_flush_preserve_sample_counts() {
        let input = [0.125, -0.25, 0.75, 0.0, 0.5];
        let mut r = PcmResampler::new(16000, 16000).unwrap();
        assert_eq!(r.push(&input).unwrap(), input);
        assert!(r.finish().unwrap().is_empty());
        assert!(r.push(&input).is_err());
        let mut r = PcmResampler::new(22050, 48000).unwrap();
        let mut out = r.push(&input).unwrap();
        out.extend(r.finish().unwrap());
        assert_eq!(out.len(), 11);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
