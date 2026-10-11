//! Timestamped playback-reference delivery; all DSP runs on the capture reader.

use super::{PcmResampler, StampedSample, SystemAudioError};
use crate::audio::echo::EchoCanceller;
use cpal::{FromSample, Sample};
use rtrb::{Consumer, Producer, RingBuffer};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::Instant,
};

pub(super) struct RenderTap {
    pub origin: Instant,
    rate: u32,
    queue: Producer<StampedSample>,
    overflow: Arc<AtomicU64>,
    enabled: Arc<AtomicBool>,
}
pub(super) struct RenderReference {
    rate: u32,
    queue: Consumer<StampedSample>,
    overflow: Arc<AtomicU64>,
    enabled: Arc<AtomicBool>,
}

pub(super) fn reference(
    rate: u32,
    capacity: usize,
    origin: Instant,
) -> (RenderTap, RenderReference) {
    let (producer, consumer) = RingBuffer::new(capacity);
    let overflow = Arc::new(AtomicU64::new(0));
    let enabled = Arc::new(AtomicBool::new(true));
    (
        RenderTap {
            origin,
            rate,
            queue: producer,
            overflow: overflow.clone(),
            enabled: enabled.clone(),
        },
        RenderReference {
            rate,
            queue: consumer,
            overflow,
            enabled,
        },
    )
}

impl RenderTap {
    pub fn capture<T: Sample>(&mut self, pcm: &[T], channels: usize, first_ns: u64)
    where
        f32: FromSample<T>,
    {
        if !self.enabled.load(Ordering::Acquire) {
            return;
        }
        // Playback is mono replicated to every output channel. Read the actual
        // scalar delivered to the device, after clipping/format conversion.
        for (i, frame) in pcm.chunks_exact(channels).enumerate() {
            if !self.enabled.load(Ordering::Acquire) {
                break;
            }
            let sample = StampedSample {
                value: f32::from_sample(frame[0]),
                at_ns: first_ns + i as u64 * 1_000_000_000 / u64::from(self.rate),
            };
            if self.queue.push(sample).is_err() && self.enabled.load(Ordering::Acquire) {
                self.overflow.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

pub(super) struct EchoInput {
    reference: RenderReference,
    render_resampler: PcmResampler,
    canceller: EchoCanceller,
    capture: VecDeque<f32>,
    render: VecDeque<f32>,
    rate: u32,
    origin_ns: Option<u64>,
    processed: u64,
}

impl EchoInput {
    pub fn new(reference: RenderReference, rate: u32) -> Result<Self, SystemAudioError> {
        Ok(Self {
            render_resampler: PcmResampler::new(reference.rate, rate)?,
            reference,
            canceller: EchoCanceller::new(rate)
                .map_err(|e| SystemAudioError::Echo(e.to_string()))?,
            capture: VecDeque::new(),
            render: VecDeque::new(),
            rate,
            origin_ns: None,
            processed: 0,
        })
    }

    pub fn check(&self) -> Result<(), SystemAudioError> {
        let lost = self.reference.overflow.load(Ordering::Acquire);
        if lost != 0 {
            return Err(SystemAudioError::Echo(format!(
                "playback reference overflow: {lost} samples"
            )));
        }
        Ok(())
    }

    pub fn stop_reference(&self) {
        self.reference.enabled.store(false, Ordering::Release);
    }

    fn analyze_until(&mut self, capture_end_ns: u64) -> Result<(), SystemAudioError> {
        self.check()?;
        let frame_len = self.canceller.frame_samples();
        let mut native = [0.0; 1024];
        loop {
            let mut count = 0;
            while count < native.len()
                && self
                    .reference
                    .queue
                    .peek()
                    .is_ok_and(|s| s.at_ns <= capture_end_ns)
            {
                native[count] = self
                    .reference
                    .queue
                    .pop()
                    .expect("single reference reader")
                    .value;
                count += 1;
            }
            if count == 0 {
                break;
            }
            self.render
                .extend(self.render_resampler.push(&native[..count])?);
            while self.render.len() >= frame_len {
                let frame: Vec<_> = self.render.drain(..frame_len).collect();
                self.canceller
                    .analyze_render(&frame)
                    .map_err(|e| SystemAudioError::Echo(e.to_string()))?;
            }
        }
        Ok(())
    }

    pub fn push_capture(
        &mut self,
        pcm: &[f32],
        first_ns: u64,
    ) -> Result<Vec<f32>, SystemAudioError> {
        self.origin_ns.get_or_insert(first_ns);
        self.capture.extend(pcm);
        let mut output = Vec::with_capacity(pcm.len());
        let frame_len = self.canceller.frame_samples();
        while self.capture.len() >= frame_len {
            let mut frame: Vec<_> = self.capture.drain(..frame_len).collect();
            self.process(&mut frame)?;
            output.extend(frame);
        }
        Ok(output)
    }

    fn process(&mut self, frame: &mut [f32]) -> Result<(), SystemAudioError> {
        let end_ns = self.origin_ns.unwrap_or(0)
            + (self.processed + frame.len() as u64) * 1_000_000_000 / u64::from(self.rate);
        self.analyze_until(end_ns)?;
        self.canceller
            .process_capture(frame)
            .map_err(|e| SystemAudioError::Echo(e.to_string()))?;
        self.processed += frame.len() as u64;
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Vec<f32>, SystemAudioError> {
        let valid = self.capture.len();
        if valid == 0 {
            return Ok(Vec::new());
        }
        let mut frame: Vec<_> = self.capture.drain(..).collect();
        frame.resize(self.canceller.frame_samples(), 0.0);
        self.process(&mut frame)?;
        frame.truncate(valid);
        Ok(frame)
    }
}

impl Drop for EchoInput {
    fn drop(&mut self) {
        self.stop_reference();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_is_bounded_timestamped_and_stops_with_capture() {
        let (mut tap, reference) = reference(16_000, 4, Instant::now());
        tap.capture(&[8192_i16, 8192, 0, 0, -16384, -16384], 2, 1_000_000);
        let mut input = EchoInput::new(reference, 16_000).unwrap();
        assert_eq!(input.reference.queue.pop().unwrap().value, 0.25);
        let silent = input.reference.queue.pop().unwrap();
        assert_eq!((silent.value, silent.at_ns), (0.0, 1_062_500));
        assert_eq!(input.reference.queue.pop().unwrap().value, -0.5);
        let mut output = input.push_capture(&[0.0; 161], 0).unwrap();
        output.extend(input.finish().unwrap());
        assert_eq!(output.len(), 161); // Final frame padding never enters model PCM.
        input.stop_reference();
        tap.capture(&[0.5_f32; 20], 1, 0);
        assert!(input.reference.queue.pop().is_err());
        assert!(input.check().is_ok());
        let (mut tap, reference) = super::reference(16_000, 1, Instant::now());
        tap.capture(&[0.0_f32; 2], 1, 0);
        assert!(EchoInput::new(reference, 16_000).unwrap().check().is_err());
    }

    #[test]
    fn timestamped_reference_cancels_echo_through_rate_conversion() {
        let native_rate = 48_000;
        let count = native_rate as usize * 10;
        let mut seed = 13_u32;
        let far: Vec<_> = (0..count)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / (1_u32 << 24) as f32 - 0.5) * 0.4
            })
            .collect();
        let (mut tap, reference) = reference(native_rate, 8192, Instant::now());
        let mut echo = EchoInput::new(reference, 16_000).unwrap();
        let mut microphone = PcmResampler::new(native_rate, 16_000).unwrap();
        let mut before = Vec::new();
        let mut after = Vec::new();
        let started = Instant::now();
        for start in (0..count).step_by(480) {
            // Output device starts 5 ms after capture. The physical echo takes
            // another 30 ms to arrive. References are timestamped in the future
            // as a real output callback can be, rather than fed at inference time.
            tap.capture(
                &far[start..start + 480],
                1,
                5_000_000 + start as u64 * 1_000_000_000 / u64::from(native_rate),
            );
            let mic: Vec<_> = (start..start + 480)
                .map(|i| if i >= 1680 { 0.6 * far[i - 1680] } else { 0.0 })
                .collect();
            let input = microphone.push(&mic).unwrap();
            before.extend_from_slice(&input);
            after.extend(
                echo.push_capture(
                    &input,
                    start as u64 * 1_000_000_000 / u64::from(native_rate),
                )
                .unwrap(),
            );
        }
        let input = microphone.finish().unwrap();
        before.extend_from_slice(&input);
        after.extend(echo.push_capture(&input, 0).unwrap());
        after.extend(echo.finish().unwrap());
        assert_eq!(after.len(), 160_000);
        let energy = |pcm: &[f32]| {
            pcm[128_000..]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
        };
        let reduction = 10.0 * (energy(&before) / energy(&after)).log10();
        eprintln!("timestamped/resampled echo reduction={reduction:.1} dB; host processing {:.3} ms per 80 ms of PCM", started.elapsed().as_secs_f64() * 1000.0 / 125.0);
        assert!(reduction > 15.0, "aligned echo reduction {reduction} dB");
    }
}
