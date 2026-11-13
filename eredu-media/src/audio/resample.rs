use std::{borrow::Cow, f32::consts::PI};

use super::AudioWaveform;
use crate::MediaError;

/// Resamples finite mono PCM using a six-zero Hann-windowed sinc and 0.99 rolloff.
///
/// The waveform is zero-extended at both ends. The output contains the ceiling
/// of the rate-scaled sample count.
pub fn resample_hann_sinc(
    waveform: AudioWaveform<'_>,
    target_rate: u32,
) -> Result<Cow<'_, [f32]>, MediaError> {
    if target_rate == 0 {
        return Err(MediaError::invalid(
            "target audio sample rate must be positive",
        ));
    }
    let samples = waveform.samples();
    let source_rate = waveform.sample_rate();
    if source_rate == target_rate {
        return Ok(Cow::Borrowed(samples));
    }

    const ZEROS: f32 = 6.0;
    let (mut divisor, mut remainder) = (source_rate, target_rate);
    while remainder != 0 {
        (divisor, remainder) = (remainder, divisor % remainder);
    }
    let (source, target) = (source_rate / divisor, target_rate / divisor);
    let cutoff = f64::from(source.min(target)) * 0.99;
    let radius = (f64::from(ZEROS) * f64::from(source) / cutoff).ceil() as usize;
    let scale = (cutoff / f64::from(source)) as f32;
    let count = (samples.len() as u128 * u128::from(target)).div_ceil(u128::from(source));
    let count = usize::try_from(count)
        .map_err(|_| MediaError::invalid("resampled waveform length overflowed"))?;
    let mut output = Vec::new();
    output.try_reserve_exact(count).map_err(|error| {
        MediaError::invalid(format!("resampled waveform allocation failed: {error}"))
    })?;
    for index in 0..count {
        let numerator = index as u128 * u128::from(source);
        let center = (numerator / u128::from(target)) as usize;
        let base = (index as u128 / u128::from(target) * u128::from(source)) as usize;
        let phase = (index as u128 % u128::from(target)) as f32 / target as f32;
        let first = center.saturating_sub(radius);
        let last = center.saturating_add(radius).min(samples.len() - 1);
        let mut sum = 0.0;
        for (sample_index, sample) in (first..=last).zip(&samples[first..=last]) {
            let offset = (sample_index as i128 - base as i128) as f32;
            let distance = ((offset / source as f32 - phase) * cutoff as f32).clamp(-ZEROS, ZEROS);
            let window = (distance * PI / ZEROS / 2.0).cos().powi(2);
            let angle = distance * PI;
            let sinc = if angle == 0.0 {
                1.0
            } else {
                angle.sin() / angle
            };
            sum += sample * (sinc * (window * scale));
        }
        output.push(sum);
    }
    Ok(Cow::Owned(output))
}
