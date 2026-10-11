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

mod processing;
mod resample;
use crate::audio::processing::{CaptureProcessingConfig, NoiseSuppression};
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
    /// Capture filtering or playback-reference continuity failure.
    #[error("audio capture processing: {0}")]
    Processing(String),
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
    queue: Consumer<StampedSample>,
    origin: Instant,
    processing: Option<processing::CaptureInput>,
    capture_enabled: Arc<AtomicBool>,
    permissions: CapturePermissions,
    overflow: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
    resampler: PcmResampler,
    pending: std::collections::VecDeque<f32>,
    stopped: bool,
    flushed: bool,
}
impl SystemAudioInput {
    /// Opens the selected/default input device with optional noise suppression.
    /// Capture begins only at `start`. Filtering uses 16, 32 or 48 kHz output;
    /// suppression off retains the ordinary resampler's full rate range.
    pub fn new(
        name: Option<&str>,
        sample_rate: u32,
        queue_ms: u32,
        noise_suppression: NoiseSuppression,
    ) -> Result<Self, SystemAudioError> {
        Self::open(
            name,
            sample_rate,
            queue_ms,
            CaptureProcessingConfig {
                noise_suppression,
                ..Default::default()
            },
        )
    }
    fn open(
        name: Option<&str>,
        sample_rate: u32,
        queue_ms: u32,
        processing: CaptureProcessingConfig,
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
        let origin = Instant::now();
        let capture_enabled = Arc::new(AtomicBool::new(true));
        macro_rules! build {
            ($t:ty) => {
                input_stream::<$t>(
                    &device,
                    config,
                    channels,
                    producer,
                    overflow.clone(),
                    failed.clone(),
                    origin,
                    capture_enabled.clone(),
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
            origin,
            processing: if processing.echo_cancellation
                || processing.noise_suppression != NoiseSuppression::Off
            {
                Some(processing::CaptureInput::new(sample_rate, processing)?)
            } else {
                None
            },
            capture_enabled,
            permissions: CapturePermissions::new(rate, sample_rate),
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
    /// Enables or silences captured PCM at its source position. Filtering still
    /// consumes the real signal so echo/noise estimation remains active.
    pub fn set_capture_enabled(&self, enabled: bool) {
        self.capture_enabled.store(enabled, Ordering::Release);
    }
    /// Starts microphone capture.
    pub fn start(&self) -> Result<(), SystemAudioError> {
        self.stream.play().map_err(device_error)
    }
    /// Stops capture while preserving all samples already queued.
    pub fn stop(&mut self) -> Result<(), SystemAudioError> {
        self.stream.pause().map_err(device_error)?;
        self.stopped = true;
        if let Some(echo) = &self.processing {
            echo.stop_reference();
        }
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
        let mut first_ns = 0;
        if self.pending.is_empty() {
            while count < native.len() {
                match self.queue.pop() {
                    Ok(v) => {
                        if count == 0 {
                            first_ns = v.at_ns;
                        }
                        native[count] = v.value;
                        self.permissions.native.push_back(v.admitted);
                        count += 1;
                    }
                    Err(_) => break,
                }
            }
            if count != 0 {
                let samples = self.resampler.push(&native[..count])?;
                self.permissions.convert(samples.len());
                if let Some(echo) = &mut self.processing {
                    self.pending.extend(echo.push_capture(&samples, first_ns)?);
                } else {
                    self.pending.extend(samples);
                }
            }
            if self.stopped && self.queue.slots() == 0 && !self.flushed {
                let samples = self.resampler.finish()?;
                self.permissions.convert(samples.len());
                if let Some(echo) = &mut self.processing {
                    self.pending.extend(echo.push_capture(&samples, first_ns)?);
                    self.pending.extend(echo.finish()?);
                } else {
                    self.pending.extend(samples);
                }
                self.flushed = true;
            }
        }
        let count = buffer.len().min(self.pending.len());
        for value in &mut buffer[..count] {
            let sample = self.pending.pop_front().unwrap();
            *value = if self
                .permissions
                .converted
                .pop_front()
                .expect("one permission per processed sample")
            {
                sample
            } else {
                0.0
            };
        }
        Ok(if count != 0 || self.flushed {
            Some(count)
        } else {
            None
        })
    }
    /// Reports callback errors or continuity loss to the application thread.
    pub fn check(&self) -> Result<(), SystemAudioError> {
        if let Some(echo) = &self.processing {
            echo.check()?;
        }
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

#[derive(Debug, Clone, Copy)]
struct StampedSample {
    value: f32,
    at_ns: u64,
    admitted: bool,
}

// Keep admission attached to the captured samples across resampling and the
// processor's 10 ms buffering, rather than applying the latest key state to old PCM.
struct CapturePermissions {
    native: std::collections::VecDeque<bool>,
    converted: std::collections::VecDeque<bool>,
    from: u64,
    to: u64,
    native_start: u64,
    emitted: u64,
}
impl CapturePermissions {
    fn new(from: u32, to: u32) -> Self {
        Self {
            native: Default::default(),
            converted: Default::default(),
            from: u64::from(from),
            to: u64::from(to),
            native_start: 0,
            emitted: 0,
        }
    }
    fn convert(&mut self, count: usize) {
        for _ in 0..count {
            let source = self.emitted * self.from / self.to;
            while self.native_start < source {
                self.native.pop_front().expect("captured permission");
                self.native_start += 1;
            }
            self.converted
                .push_back(*self.native.front().expect("captured permission"));
            self.emitted += 1;
        }
    }
}

/// Opens ordinary capture/playback with a shared timestamped speaker reference
/// and host AEC3 processing. Capture starts only when the caller calls `start`.
/// The processed input rate must be 16, 32 or 48 kHz.
pub fn duplex(
    input_name: Option<&str>,
    output_name: Option<&str>,
    input_rate: u32,
    output_rate: u32,
    queue_ms: u32,
    noise_suppression: NoiseSuppression,
) -> Result<(SystemAudioInput, SystemAudioOutput), SystemAudioError> {
    let mut input = SystemAudioInput::open(
        input_name,
        input_rate,
        queue_ms,
        CaptureProcessingConfig {
            echo_cancellation: true,
            noise_suppression,
        },
    )?;
    let (output, reference) =
        SystemAudioOutput::open(output_name, output_rate, queue_ms, Some(input.origin))?;
    input
        .processing
        .as_mut()
        .expect("AEC configured")
        .attach_reference(reference.expect("requested playback reference"))?;
    Ok((input, output))
}

fn input_stream<T: SizedSample + Send + 'static>(
    device: &Device,
    config: cpal::StreamConfig,
    channels: usize,
    mut queue: Producer<StampedSample>,
    overflow: Arc<AtomicU64>,
    failed: Arc<AtomicBool>,
    origin: Instant,
    capture_enabled: Arc<AtomicBool>,
) -> Result<cpal::Stream, SystemAudioError>
where
    f32: FromSample<T>,
{
    let error = failed.clone();
    let sample_rate = config.sample_rate;
    device
        .build_input_stream(
            config,
            move |data: &[T], info| {
                let time = info.timestamp();
                let start = origin
                    .elapsed()
                    .saturating_sub(time.callback.duration_since(time.capture));
                capture(
                    data,
                    channels,
                    &mut queue,
                    &overflow,
                    &failed,
                    start.as_nanos() as u64,
                    sample_rate,
                    capture_enabled.load(Ordering::Acquire),
                );
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
    queue: &mut Producer<StampedSample>,
    overflow: &AtomicU64,
    failed: &AtomicBool,
    first_ns: u64,
    sample_rate: u32,
    admitted: bool,
) where
    f32: FromSample<T>,
{
    if channels == 0 || data.len() % channels != 0 {
        failed.store(true, Ordering::Release);
        return;
    }
    for (i, frame) in data.chunks_exact(channels).enumerate() {
        let sample = frame.iter().map(|v| f32::from_sample(*v)).sum::<f32>() / channels as f32;
        if !sample.is_finite() {
            failed.store(true, Ordering::Release);
            continue;
        }
        if queue
            .push(StampedSample {
                value: sample,
                at_ns: first_ns + i as u64 * 1_000_000_000 / u64::from(sample_rate),
                admitted,
            })
            .is_err()
        {
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
        Self::open(name, sample_rate, queue_ms, None).map(|(output, _)| output)
    }
    fn open(
        name: Option<&str>,
        sample_rate: u32,
        queue_ms: u32,
        reference_origin: Option<Instant>,
    ) -> Result<(Self, Option<processing::RenderReference>), SystemAudioError> {
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
        let (tap, reference) = match reference_origin {
            Some(origin) => {
                let (tap, reference) = processing::reference(rate, max_queued, origin);
                (Some(tap), Some(reference))
            }
            None => (None, None),
        };
        macro_rules! build {
            ($t:ty) => {
                output_stream::<$t>(&device, config, channels, consumer, state.clone(), tap)
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
        Ok((
            Self {
                stream,
                queue,
                state,
                resampler: PcmResampler::new(sample_rate, rate)?,
                from: sample_rate,
                to: rate,
                max_queued,
                preroll: (rate as usize * 160).div_ceil(1000),
                queued_since_interrupt: 0,
            },
            reference,
        ))
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
    mut tap: Option<processing::RenderTap>,
) -> Result<cpal::Stream, SystemAudioError>
where
    f32: FromSample<T>,
{
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
                if let Some(tap) = &mut tap {
                    let start = tap.origin.elapsed()
                        + timestamp.playback.duration_since(timestamp.callback);
                    tap.capture(data, channels, start.as_nanos() as u64);
                }
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
    fn captured_ptt_permissions_follow_pcm_through_conversion_and_processor_buffering() {
        for (from, to) in [(48_000, 16_000), (16_000, 48_000)] {
            let mut resampler = PcmResampler::new(from, to).unwrap();
            let mut permissions = CapturePermissions::new(from, to);
            let mut processor = processing::CaptureInput::new(
                to,
                CaptureProcessingConfig {
                    noise_suppression: NoiseSuppression::Low,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut output = Vec::new();
            let mut masks = Vec::new();
            for start in (0..from as usize).step_by(137) {
                let end = (start + 137).min(from as usize);
                permissions
                    .native
                    .extend((start..end).map(|i| i >= from as usize / 4 && i < from as usize / 2));
                let samples = resampler.push(&vec![0.1; end - start]).unwrap();
                permissions.convert(samples.len());
                let samples = processor.push_capture(&samples, 0).unwrap();
                for _ in &samples {
                    masks.push(permissions.converted.pop_front().unwrap());
                }
                output.extend(samples);
            }
            let samples = resampler.finish().unwrap();
            permissions.convert(samples.len());
            let mut samples = processor.push_capture(&samples, 0).unwrap();
            samples.extend(processor.finish().unwrap());
            for _ in &samples {
                masks.push(permissions.converted.pop_front().unwrap());
            }
            output.extend(samples);
            assert_eq!(output.len(), to as usize);
            assert_eq!(masks.len(), to as usize);
            assert!(masks[..to as usize / 4].iter().all(|v| !v));
            assert!(masks[to as usize / 4..to as usize / 2].iter().all(|v| *v));
            assert!(masks[to as usize / 2..].iter().all(|v| !v));
            assert!(permissions.converted.is_empty());
        }
    }
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
            0,
            48_000,
            true,
        );
        assert_eq!(consumer.pop().unwrap().value, 0.125);
        assert_eq!(consumer.pop().unwrap().value, 0.25);
        assert_eq!(overflow.load(Ordering::Relaxed), 1);
        assert!(!failed.load(Ordering::Relaxed));
        capture(
            &[f32::NAN, 0.0],
            2,
            &mut producer,
            &overflow,
            &failed,
            0,
            48_000,
            true,
        );
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
