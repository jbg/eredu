//! Optional cross-platform host capture and playback through CPAL.
//! Callbacks only convert scalar formats/downmix channels and access bounded
//! single-producer queues. Resampling, waiting and application work are host-owned.

use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, Sample, SampleFormat, SizedSample,
};
use rtrb::{Consumer, Producer, RingBuffer};
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

mod resample;
pub use resample::PcmResampler;

/// Host-device, queue or resampling failure, independent of inference backends.
#[derive(Debug, thiserror::Error)]
pub enum SystemAudioError {
    /// Invalid host processing geometry or lifecycle.
    #[error("{0}")]
    Invalid(&'static str),
    /// Host audio API failure.
    #[error("system audio: {0}")]
    Device(String),
    /// Band-limited conversion failure.
    #[error("audio resampling: {0}")]
    Resample(String),
    /// Input cannot preserve its continuous sample timeline within the bounded queue.
    #[error("microphone queue overflow: {0} samples; inference cannot keep up")]
    InputOverflow(u64),
    /// Output cannot accept more samples within its bounded queue.
    #[error("speaker queue remains full")]
    OutputFull,
}

/// Enumerated host device name and directions.
#[derive(Debug, Clone)]
pub struct AudioDevice {
    /// Name accepted by explicit device selection.
    pub name: String,
    /// Whether the device accepts capture streams.
    pub input: bool,
    /// Whether the device accepts playback streams.
    pub output: bool,
}
/// Lists devices on CPAL's default platform host without starting a stream.
pub fn devices() -> Result<Vec<AudioDevice>, SystemAudioError> {
    cpal::default_host()
        .devices()
        .map_err(device_error)?
        .map(|d| {
            Ok(AudioDevice {
                name: device_name(&d)?,
                input: d.default_input_config().is_ok(),
                output: d.default_output_config().is_ok(),
            })
        })
        .collect()
}
fn device_name(device: &Device) -> Result<String, SystemAudioError> {
    device
        .description()
        .map(|d| d.name().to_owned())
        .map_err(device_error)
}
fn device_error(error: impl std::fmt::Display) -> SystemAudioError {
    SystemAudioError::Device(error.to_string())
}
fn select_device(name: Option<&str>, input: bool) -> Result<Device, SystemAudioError> {
    let host = cpal::default_host();
    if let Some(name) = name {
        let mut matches =
            host.devices()
                .map_err(device_error)?
                .filter_map(|d| match device_name(&d) {
                    Ok(n) if n == name => Some(d),
                    _ => None,
                });
        let d = matches
            .next()
            .ok_or_else(|| SystemAudioError::Device(format!("no audio device named {name:?}")))?;
        if matches.next().is_some() {
            return Err(SystemAudioError::Device(format!(
                "ambiguous audio device name {name:?}"
            )));
        }
        Ok(d)
    } else {
        (if input {
            host.default_input_device()
        } else {
            host.default_output_device()
        })
        .ok_or_else(|| {
            SystemAudioError::Device(format!(
                "no default {} device",
                if input { "input" } else { "output" }
            ))
        })
    }
}
fn capacity(rate: u32, milliseconds: u32) -> Result<usize, SystemAudioError> {
    if !(160..=10_000).contains(&milliseconds) {
        return Err(SystemAudioError::Invalid(
            "audio queue must be between 160 and 10000 milliseconds",
        ));
    }
    Ok((rate as usize * milliseconds as usize).div_ceil(1000))
}

/// Selected native microphone format and its delivered mono rate.
#[derive(Debug, Clone)]
pub struct AudioInputConfiguration {
    /// Exact selected host device name.
    pub device: String,
    /// Native interleaved channel count, averaged to mono.
    pub channels: u16,
    /// Native samples per second per channel.
    pub sample_rate: u32,
    /// Host scalar representation, reported for diagnostics.
    pub sample_format: String,
    /// Mono samples per second delivered to callers.
    pub output_sample_rate: u32,
}

/// Bounded microphone capture converted to a caller-selected mono sample rate.
pub struct SystemAudioInput {
    configuration: AudioInputConfiguration,
    stream: cpal::Stream,
    queue: Consumer<f32>,
    overflow: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
    resampler: PcmResampler,
    pending: std::collections::VecDeque<f32>,
    stopped: bool,
    flushed: bool,
}
impl SystemAudioInput {
    /// Opens the selected/default input device. Capture begins only at `start`.
    pub fn new(
        name: Option<&str>,
        sample_rate: u32,
        queue_ms: u32,
    ) -> Result<Self, SystemAudioError> {
        let device = select_device(name, true)?;
        let supported = device.default_input_config().map_err(device_error)?;
        let rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        if channels == 0 {
            return Err(SystemAudioError::Invalid("audio device has no channels"));
        }
        let (producer, queue) = RingBuffer::new(capacity(rate, queue_ms)?);
        let overflow = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicBool::new(false));
        let format = supported.sample_format();
        let configuration = AudioInputConfiguration {
            device: device_name(&device)?,
            channels: supported.channels(),
            sample_rate: rate,
            sample_format: format!("{format:?}"),
            output_sample_rate: sample_rate,
        };
        let config = supported.into();
        macro_rules! build {
            ($t:ty) => {
                input_stream::<$t>(
                    &device,
                    config,
                    channels,
                    producer,
                    overflow.clone(),
                    failed.clone(),
                )
            };
        }
        let stream = match format {
            SampleFormat::I8 => build!(i8),
            SampleFormat::I16 => build!(i16),
            SampleFormat::I24 => build!(cpal::I24),
            SampleFormat::I32 => build!(i32),
            SampleFormat::I64 => build!(i64),
            SampleFormat::U8 => build!(u8),
            SampleFormat::U16 => build!(u16),
            SampleFormat::U24 => build!(cpal::U24),
            SampleFormat::U32 => build!(u32),
            SampleFormat::U64 => build!(u64),
            SampleFormat::F32 => build!(f32),
            SampleFormat::F64 => build!(f64),
            _ => return Err(SystemAudioError::Invalid("unsupported input sample format")),
        }?;
        Ok(Self {
            configuration,
            stream,
            queue,
            overflow,
            failed,
            resampler: PcmResampler::new(rate, sample_rate)?,
            pending: Default::default(),
            stopped: false,
            flushed: false,
        })
    }
    /// Returns the actual selected format without acquiring another stream.
    pub fn configuration(&self) -> &AudioInputConfiguration {
        &self.configuration
    }
    /// Starts microphone capture.
    pub fn start(&self) -> Result<(), SystemAudioError> {
        self.stream.play().map_err(device_error)
    }
    /// Stops capture while preserving all samples already queued.
    pub fn stop(&mut self) -> Result<(), SystemAudioError> {
        self.stream.pause().map_err(device_error)?;
        self.stopped = true;
        Ok(())
    }
    /// Returns pending (`None`), samples, or drained EOF (`Some(0)`).
    pub fn read(&mut self, buffer: &mut [f32]) -> Result<Option<usize>, SystemAudioError> {
        self.check()?;
        if buffer.is_empty() {
            return Err(SystemAudioError::Invalid("empty microphone read buffer"));
        }
        let mut native = [0.0; 1024];
        let mut count = 0;
        if self.pending.is_empty() {
            while count < native.len() {
                match self.queue.pop() {
                    Ok(v) => {
                        native[count] = v;
                        count += 1;
                    }
                    Err(_) => break,
                }
            }
            if count != 0 {
                self.pending.extend(self.resampler.push(&native[..count])?);
            }
            if self.stopped && self.queue.slots() == 0 && !self.flushed {
                self.pending.extend(self.resampler.finish()?);
                self.flushed = true;
            }
        }
        let count = buffer.len().min(self.pending.len());
        for value in &mut buffer[..count] {
            *value = self.pending.pop_front().unwrap();
        }
        Ok(if count != 0 || self.flushed {
            Some(count)
        } else {
            None
        })
    }
    /// Reports callback errors or continuity loss to the application thread.
    pub fn check(&self) -> Result<(), SystemAudioError> {
        if self.failed.load(Ordering::Acquire) {
            return Err(SystemAudioError::Device(
                "input stream callback failed or received nonfinite PCM".into(),
            ));
        }
        let overflow = self.overflow.load(Ordering::Acquire);
        if overflow != 0 {
            return Err(SystemAudioError::InputOverflow(overflow));
        }
        Ok(())
    }
}
fn input_stream<T: SizedSample + Send + 'static>(
    device: &Device,
    config: cpal::StreamConfig,
    channels: usize,
    mut queue: Producer<f32>,
    overflow: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
) -> Result<cpal::Stream, SystemAudioError>
where
    f32: FromSample<T>,
{
    let error = failed.clone();
    device
        .build_input_stream(
            config,
            move |data: &[T], _| {
                capture(data, channels, &mut queue, &overflow, &failed);
            },
            move |_| {
                error.store(true, Ordering::Release);
            },
            None,
        )
        .map_err(device_error)
}
fn capture<T: Sample>(
    data: &[T],
    channels: usize,
    queue: &mut Producer<f32>,
    overflow: &AtomicU64,
    failed: &AtomicBool,
) where
    f32: FromSample<T>,
{
    if channels == 0 || data.len() % channels != 0 {
        failed.store(true, Ordering::Release);
        return;
    }
    for frame in data.chunks_exact(channels) {
        let sample = frame.iter().map(|v| f32::from_sample(*v)).sum::<f32>() / channels as f32;
        if !sample.is_finite() {
            failed.store(true, Ordering::Release);
            continue;
        }
        if queue.push(sample).is_err() {
            overflow.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct PlaybackSample {
    epoch: u64,
    value: f32,
}
struct PlaybackState {
    origin: Instant,
    ready: AtomicBool,
    epoch: AtomicU64,
    written: AtomicU64,
    completed: AtomicU64,
    underruns: AtomicU64,
    failed: AtomicBool,
    started: AtomicBool,
    expecting: AtomicBool,
}
/// Bounded mono playback with stale queued generations discarded on interruption.
pub struct SystemAudioOutput {
    stream: cpal::Stream,
    queue: Producer<PlaybackSample>,
    state: Arc<PlaybackState>,
    resampler: PcmResampler,
    from: u32,
    to: u32,
    max_queued: usize,
    preroll: usize,
    queued_since_interrupt: usize,
}
impl SystemAudioOutput {
    /// Opens and starts the selected/default output, initially playing silence.
    pub fn new(
        name: Option<&str>,
        sample_rate: u32,
        queue_ms: u32,
    ) -> Result<Self, SystemAudioError> {
        let device = select_device(name, false)?;
        let supported = device.default_output_config().map_err(device_error)?;
        let rate = supported.sample_rate();
        let channels = supported.channels() as usize;
        if channels == 0 {
            return Err(SystemAudioError::Invalid("audio device has no channels"));
        }
        let max_queued = capacity(rate, queue_ms)?;
        let (queue, consumer) = RingBuffer::new(max_queued);
        let state = Arc::new(PlaybackState {
            origin: Instant::now(),
            ready: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            written: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            started: AtomicBool::new(false),
            expecting: AtomicBool::new(false),
        });
        let format = supported.sample_format();
        let config = supported.into();
        macro_rules! build {
            ($t:ty) => {
                output_stream::<$t>(&device, config, channels, consumer, state.clone())
            };
        }
        let stream = match format {
            SampleFormat::I8 => build!(i8),
            SampleFormat::I16 => build!(i16),
            SampleFormat::I24 => build!(cpal::I24),
            SampleFormat::I32 => build!(i32),
            SampleFormat::I64 => build!(i64),
            SampleFormat::U8 => build!(u8),
            SampleFormat::U16 => build!(u16),
            SampleFormat::U24 => build!(cpal::U24),
            SampleFormat::U32 => build!(u32),
            SampleFormat::U64 => build!(u64),
            SampleFormat::F32 => build!(f32),
            SampleFormat::F64 => build!(f64),
            _ => {
                return Err(SystemAudioError::Invalid(
                    "unsupported output sample format",
                ))
            }
        }?;
        stream.play().map_err(device_error)?;
        Ok(Self {
            stream,
            queue,
            state,
            resampler: PcmResampler::new(sample_rate, rate)?,
            from: sample_rate,
            to: rate,
            max_queued,
            preroll: (rate as usize * 160).div_ceil(1000),
            queued_since_interrupt: 0,
        })
    }
    /// Queues model PCM, waiting only on the application thread when playback lags.
    pub fn write(&mut self, samples: &[f32]) -> Result<(), SystemAudioError> {
        if !samples.is_empty()
            && !self.state.expecting.swap(true, Ordering::AcqRel)
            && self.queue.slots() == self.max_queued
        {
            self.queued_since_interrupt = 0;
            self.state.ready.store(false, Ordering::Release);
        }
        let output = self.resampler.push(samples)?;
        self.enqueue(&output)
    }
    fn enqueue(&mut self, samples: &[f32]) -> Result<(), SystemAudioError> {
        let epoch = self.state.epoch.load(Ordering::Acquire);
        let mut consumed = 0;
        let deadline = Instant::now() + Duration::from_secs(5);
        while consumed < samples.len() {
            self.check()?;
            while consumed < samples.len() && self.queue.slots() != 0 {
                self.queue
                    .push(PlaybackSample {
                        epoch,
                        value: samples[consumed],
                    })
                    .map_err(|_| SystemAudioError::OutputFull)?;
                self.state.written.fetch_add(1, Ordering::Release);
                consumed += 1;
                self.queued_since_interrupt += 1;
                if self.queued_since_interrupt >= self.preroll {
                    self.state.ready.store(true, Ordering::Release);
                }
            }
            if consumed < samples.len() {
                if Instant::now() >= deadline {
                    return Err(SystemAudioError::OutputFull);
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(())
    }
    /// Flushes a phrase's delayed suffix and returns its playback completion ticket.
    pub fn flush_segment(&mut self) -> Result<u64, SystemAudioError> {
        let output = self.resampler.finish()?;
        self.enqueue(&output)?;
        self.resampler = PcmResampler::new(self.from, self.to)?;
        self.state.ready.store(true, Ordering::Release);
        self.state.expecting.store(false, Ordering::Release);
        Ok(self.ticket())
    }
    /// Opaque completion position for samples already queued to the native stream.
    pub fn ticket(&self) -> u64 {
        self.state.written.load(Ordering::Acquire)
    }
    /// Whether the host playback timestamp for that range has passed.
    pub fn completed(&self, ticket: u64) -> bool {
        self.state.completed.load(Ordering::Acquire) >= ticket
    }
    /// Clears delayed and queued speech without racing the callback's consumer.
    pub fn interrupt(&mut self) -> Result<(), SystemAudioError> {
        self.state.epoch.fetch_add(1, Ordering::AcqRel);
        self.state.expecting.store(false, Ordering::Release);
        self.state.ready.store(false, Ordering::Release);
        self.queued_since_interrupt = 0;
        self.resampler = PcmResampler::new(self.from, self.to)?;
        Ok(())
    }
    /// Flushes the delayed suffix and waits for its playback completion.
    pub fn finish(&mut self) -> Result<(), SystemAudioError> {
        self.state.expecting.store(false, Ordering::Release);
        let output = self.resampler.finish()?;
        self.enqueue(&output)?;
        self.state.ready.store(true, Ordering::Release);
        let ticket = self.ticket();
        let deadline = Instant::now() + Duration::from_secs(5);
        while !self.completed(ticket) {
            self.check()?;
            if Instant::now() >= deadline {
                return Err(SystemAudioError::OutputFull);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        Ok(())
    }
    /// Native mono frames starved while more speech is expected. Explicit phrase
    /// endings, interruption and final drain do not count intentional silence.
    pub fn underrun_samples(&self) -> u64 {
        self.state.underruns.load(Ordering::Acquire)
    }
    /// Queued native PCM capacity in samples.
    pub fn capacity_samples(&self) -> usize {
        self.max_queued
    }
    /// Checks native stream health on the host thread.
    pub fn check(&self) -> Result<(), SystemAudioError> {
        if self.state.failed.load(Ordering::Acquire) {
            Err(SystemAudioError::Device(
                "output stream callback failed".into(),
            ))
        } else {
            Ok(())
        }
    }
}
impl Drop for SystemAudioOutput {
    fn drop(&mut self) {
        let _ = self.stream.pause();
    }
}
#[derive(Clone, Copy, Default)]
struct PlaybackCompletion {
    ticket: u64,
    at: Duration,
}
struct PlaybackConsumer {
    queue: Consumer<PlaybackSample>,
    consumed: u64,
    pending: [PlaybackCompletion; 128],
    head: usize,
    len: usize,
}
impl PlaybackConsumer {
    fn new(queue: Consumer<PlaybackSample>) -> Self {
        Self {
            queue,
            consumed: 0,
            pending: [PlaybackCompletion::default(); 128],
            head: 0,
            len: 0,
        }
    }
    fn render<T: Sample + FromSample<f32>>(
        &mut self,
        data: &mut [T],
        channels: usize,
        state: &PlaybackState,
        now: Duration,
        playback_end: Duration,
    ) {
        while self.len != 0 && self.pending[self.head].at <= now {
            state
                .completed
                .store(self.pending[self.head].ticket, Ordering::Release);
            self.head = (self.head + 1) % self.pending.len();
            self.len -= 1;
        }
        data.fill(T::EQUILIBRIUM);
        if channels == 0 || data.len() % channels != 0 {
            state.failed.store(true, Ordering::Release);
            return;
        }
        let before = self.consumed;
        let epoch = state.epoch.load(Ordering::Acquire);
        while self.queue.peek().is_ok_and(|sample| sample.epoch < epoch) {
            let _ = self.queue.pop();
            self.consumed += 1;
        }
        if state.ready.load(Ordering::Acquire) {
            for frame in data.chunks_exact_mut(channels) {
                let sample = loop {
                    // An interruption can arrive while this native buffer is being
                    // filled. Never retire newer queued speech against an older
                    // observed epoch; leave it for the current generation's preroll.
                    let epoch = state.epoch.load(Ordering::Acquire);
                    match self.queue.peek() {
                        Ok(sample) if sample.epoch < epoch => {
                            let _ = self.queue.pop();
                            self.consumed += 1;
                        }
                        Ok(sample)
                            if sample.epoch > epoch || !state.ready.load(Ordering::Acquire) =>
                        {
                            break 0.0
                        }
                        Ok(_) => {
                            let sample = self.queue.pop().expect("single callback consumer");
                            self.consumed += 1;
                            state.started.store(true, Ordering::Relaxed);
                            break sample.value;
                        }
                        Err(_) => {
                            if state.started.load(Ordering::Relaxed)
                                && state.expecting.load(Ordering::Acquire)
                                && state.ready.load(Ordering::Acquire)
                            {
                                state.underruns.fetch_add(1, Ordering::Relaxed);
                            }
                            break 0.0;
                        }
                    }
                };
                frame.fill(T::from_sample(sample.clamp(-1.0, 1.0)));
            }
        }
        if self.consumed != before {
            if self.len == self.pending.len() {
                state.failed.store(true, Ordering::Release);
                return;
            }
            // Cumulative tickets cannot complete before an earlier native buffer.
            let earlier = if self.len == 0 {
                now
            } else {
                self.pending[(self.head + self.len - 1) % self.pending.len()].at
            };
            self.pending[(self.head + self.len) % self.pending.len()] = PlaybackCompletion {
                ticket: self.consumed,
                at: playback_end.max(earlier),
            };
            self.len += 1;
        }
    }
}
fn output_stream<T: SizedSample + FromSample<f32> + Send + 'static>(
    device: &Device,
    config: cpal::StreamConfig,
    channels: usize,
    queue: Consumer<PlaybackSample>,
    state: Arc<PlaybackState>,
) -> Result<cpal::Stream, SystemAudioError> {
    let rate = config.sample_rate;
    let error = state.clone();
    let mut consumer = PlaybackConsumer::new(queue);
    device
        .build_output_stream(
            config,
            move |data: &mut [T], info| {
                let now = state.origin.elapsed();
                let timestamp = info.timestamp();
                let end = now
                    + timestamp.playback.duration_since(timestamp.callback)
                    + Duration::from_secs_f64((data.len() / channels) as f64 / rate as f64);
                consumer.render(data, channels, &state, now, end);
            },
            move |_| {
                error.failed.store(true, Ordering::Release);
            },
            None,
        )
        .map_err(device_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_downmixes_integer_pcm_and_reports_continuity_loss() {
        let (mut producer, mut consumer) = RingBuffer::new(2);
        let overflow = AtomicU64::new(0);
        let failed = AtomicBool::new(false);
        capture(
            &[16384_i16, -8192, 8192, 8192, 32767, 32767],
            2,
            &mut producer,
            &overflow,
            &failed,
        );
        assert_eq!(consumer.pop().unwrap(), 0.125);
        assert_eq!(consumer.pop().unwrap(), 0.25);
        assert_eq!(overflow.load(Ordering::Relaxed), 1);
        assert!(!failed.load(Ordering::Relaxed));
        capture(&[f32::NAN, 0.0], 2, &mut producer, &overflow, &failed);
        assert!(failed.load(Ordering::Relaxed));
        assert!(consumer.pop().is_err());
    }
    #[test]
    fn playback_preroll_interrupt_and_completion_obey_native_timestamps() {
        let (mut producer, queue) = RingBuffer::new(8);
        let state = PlaybackState {
            origin: Instant::now(),
            ready: AtomicBool::new(false),
            epoch: AtomicU64::new(0),
            written: AtomicU64::new(0),
            completed: AtomicU64::new(0),
            underruns: AtomicU64::new(0),
            failed: AtomicBool::new(false),
            started: AtomicBool::new(false),
            expecting: AtomicBool::new(false),
        };
        producer
            .push(PlaybackSample {
                epoch: 0,
                value: 0.25,
            })
            .unwrap();
        producer
            .push(PlaybackSample {
                epoch: 0,
                value: 0.75,
            })
            .unwrap();
        let mut consumer = PlaybackConsumer::new(queue);
        let mut stereo = [1.0_f32; 2];
        let ms = Duration::from_millis;
        consumer.render(&mut stereo, 2, &state, ms(0), ms(10));
        assert_eq!(stereo, [0.0, 0.0]); // preroll retains speech
        state.ready.store(true, Ordering::Release);
        consumer.render(&mut stereo, 2, &state, ms(1), ms(11));
        assert_eq!(stereo, [0.25, 0.25]);
        assert_eq!(state.completed.load(Ordering::Acquire), 0); // encoded is not played
        state.epoch.store(1, Ordering::Release);
        state.ready.store(false, Ordering::Release);
        consumer.render(&mut stereo, 2, &state, ms(2), ms(12));
        assert_eq!(stereo, [0.0, 0.0]); // stale queued speech is dropped even during preroll
        consumer.render(&mut stereo, 2, &state, ms(11), ms(21));
        assert_eq!(state.completed.load(Ordering::Acquire), 1);
        consumer.render(&mut stereo, 2, &state, ms(12), ms(22));
        assert_eq!(state.completed.load(Ordering::Acquire), 2); // no new speech required to retire dropped ticket
        producer
            .push(PlaybackSample {
                epoch: 1,
                value: -0.5,
            })
            .unwrap();
        state.ready.store(true, Ordering::Release);
        state.expecting.store(true, Ordering::Release);
        consumer.render(&mut stereo, 2, &state, ms(13), ms(23));
        assert_eq!(stereo, [-0.5, -0.5]);
        consumer.render(&mut stereo, 2, &state, ms(23), ms(33));
        assert_eq!(state.completed.load(Ordering::Acquire), 3);
        assert_eq!(stereo, [0.0, 0.0]);
        assert_eq!(state.underruns.load(Ordering::Acquire), 1);
        state.expecting.store(false, Ordering::Release);
        consumer.render(&mut stereo, 2, &state, ms(24), ms(34));
        assert_eq!(state.underruns.load(Ordering::Acquire), 1);
        producer
            .push(PlaybackSample {
                epoch: 2,
                value: 0.375,
            })
            .unwrap();
        consumer.render(&mut stereo, 2, &state, ms(25), ms(35));
        assert_eq!(stereo, [0.0, 0.0]); // a newer queued generation survives an older observation
        state.epoch.store(2, Ordering::Release);
        consumer.render(&mut stereo, 2, &state, ms(26), ms(36));
        assert_eq!(stereo, [0.375, 0.375]);
        consumer.render(&mut stereo, 2, &state, ms(36), ms(46));
        assert_eq!(state.completed.load(Ordering::Acquire), 4);
        assert!(!state.failed.load(Ordering::Acquire));
    }
}
