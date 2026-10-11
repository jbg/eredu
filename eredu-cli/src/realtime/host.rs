use super::{run::AudioIo, Args};
use anyhow::{ensure, Result};
#[cfg(feature = "system-audio")]
use eredu_media::audio::system::{SystemAudioInput, SystemAudioOutput};
use eredu_media::audio::wav::{WavReader, WavWriter};
use std::{
    fs::File,
    io::{BufReader, BufWriter, Write},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

pub struct HostIo {
    file: Option<WavReader<BufReader<File>>>,
    wav: Option<WavWriter<BufWriter<File>>>,
    input_recording: Option<WavWriter<BufWriter<File>>>,
    accepted_samples: u64,
    input_energy: f64,
    input_peak: f32,
    #[cfg(feature = "system-audio")]
    microphone: Option<SystemAudioInput>,
    #[cfg(feature = "system-audio")]
    speaker: Option<SystemAudioOutput>,
    start: Instant,
    stop: Arc<AtomicUsize>,
    duration: Option<Duration>,
    paced: bool,
    read_samples: u64,
    input_stopped: bool,
    underruns: u64,
}
impl HostIo {
    pub fn new(
        file: Option<WavReader<BufReader<File>>>,
        wav: Option<WavWriter<BufWriter<File>>>,
        input_recording: Option<WavWriter<BufWriter<File>>>,
        args: &Args,
        stop: Arc<AtomicUsize>,
        max_seconds: f64,
    ) -> Result<Self> {
        #[cfg(feature = "system-audio")]
        let microphone = if file.is_none() {
            Some(SystemAudioInput::new(
                args.input_device.as_deref(),
                16_000,
                args.audio_buffer_ms,
            )?)
        } else {
            None
        };
        #[cfg(feature = "system-audio")]
        let speaker = if args.no_playback {
            None
        } else {
            Some(SystemAudioOutput::new(
                args.output_device.as_deref(),
                22_050,
                args.audio_buffer_ms,
            )?)
        };
        let start = Instant::now();
        #[cfg(feature = "system-audio")]
        if let Some(microphone) = &microphone {
            let format = microphone.configuration();
            eprintln!(
                "microphone: {:?}, {} Hz, {} channels, {} -> mono {} Hz",
                format.device,
                format.sample_rate,
                format.channels,
                format.sample_format,
                format.output_sample_rate
            );
            microphone.start()?;
        }
        Ok(Self {
            file,
            wav,
            input_recording,
            accepted_samples: 0,
            input_energy: 0.0,
            input_peak: 0.0,
            #[cfg(feature = "system-audio")]
            microphone,
            #[cfg(feature = "system-audio")]
            speaker,
            start,
            stop,
            duration: (!args.offline)
                .then(|| Duration::from_secs_f64(args.seconds.unwrap_or(max_seconds))),
            paced: !args.offline,
            read_samples: 0,
            input_stopped: false,
            underruns: 0,
        })
    }
    pub fn finish(&mut self, success: bool) -> Result<()> {
        #[cfg(feature = "system-audio")]
        if let Some(microphone) = &mut self.microphone {
            let _ = microphone.stop();
        }
        // Finalize committed file output independently of native playback success.
        let wav: Result<_> = self
            .wav
            .take()
            .map(|wav| -> Result<()> {
                let mut file = wav.finish()?;
                file.flush()?;
                Ok(())
            })
            .transpose();
        let recording: Result<_> = self
            .input_recording
            .take()
            .map(|wav| -> Result<()> {
                let mut file = wav.finish()?;
                file.flush()?;
                Ok(())
            })
            .transpose();
        let rms = (self.input_energy / self.accepted_samples.max(1) as f64).sqrt();
        eprintln!(
            "input audio: {} source samples accepted, RMS={:.6} ({:.1} dBFS), peak={:.6}",
            self.accepted_samples,
            rms,
            20.0 * rms.log10(),
            self.input_peak
        );
        #[cfg(feature = "system-audio")]
        let playback = if let Some(speaker) = &mut self.speaker {
            let result = if success {
                speaker.finish()
            } else {
                speaker.interrupt()
            };
            self.underruns = speaker.underrun_samples();
            result.map_err(anyhow::Error::from)
        } else {
            Ok(())
        };
        #[cfg(not(feature = "system-audio"))]
        let playback: Result<()> = {
            let _ = success;
            Ok(())
        };
        eprintln!("speaker underrun samples: {}", self.underruns);
        wav?;
        recording?;
        playback?;
        Ok(())
    }
}
impl AudioIo for HostIo {
    fn read(&mut self, buffer: &mut [f32]) -> Result<Option<usize>> {
        self.check()?;
        if !self.input_stopped
            && (self.stop.load(Ordering::Relaxed) != 0
                || self.duration.is_some_and(|d| self.start.elapsed() >= d))
        {
            self.input_stopped = true;
            #[cfg(feature = "system-audio")]
            if let Some(microphone) = &mut self.microphone {
                microphone.stop()?;
            }
        }
        if let Some(file) = &mut self.file {
            if self.input_stopped {
                return Ok(Some(0));
            }
            let count = if self.paced {
                let available = (self.start.elapsed().as_secs_f64() * 16000.0) as u64;
                usize::try_from(available.saturating_sub(self.read_samples))
                    .unwrap_or(usize::MAX)
                    .min(buffer.len())
            } else {
                buffer.len()
            };
            if count == 0 {
                return Ok(None);
            }
            let count = file.read_samples(&mut buffer[..count])?;
            self.read_samples += count as u64;
            return Ok(Some(count));
        }
        #[cfg(feature = "system-audio")]
        if let Some(microphone) = &mut self.microphone {
            return Ok(microphone.read(buffer)?);
        }
        anyhow::bail!("no audio input configured")
    }
    fn accepted_input(&mut self, samples: &[f32]) -> Result<()> {
        if let Some(recording) = &mut self.input_recording {
            recording.write_samples(samples)?;
        }
        self.accepted_samples += samples.len() as u64;
        for &sample in samples {
            self.input_energy += f64::from(sample).powi(2);
            self.input_peak = self.input_peak.max(sample.abs());
        }
        Ok(())
    }
    fn write(&mut self, samples: &[f32]) -> Result<()> {
        if let Some(wav) = &mut self.wav {
            wav.write_samples(samples)?;
        }
        #[cfg(feature = "system-audio")]
        if let Some(speaker) = &mut self.speaker {
            speaker.write(samples)?;
        }
        Ok(())
    }
    fn clock(&self, recorded: Duration) -> Duration {
        if self.paced {
            self.start.elapsed()
        } else {
            recorded
        }
    }
    fn idle(&mut self, until: Duration) -> Result<()> {
        if self.paced {
            while self.start.elapsed() < until {
                self.check()?;
                std::thread::sleep(
                    until
                        .saturating_sub(self.start.elapsed())
                        .min(Duration::from_millis(2)),
                );
            }
            std::thread::sleep(Duration::from_millis(1));
        } else {
            std::thread::yield_now();
        }
        self.check()
    }
    fn check(&self) -> Result<()> {
        ensure!(
            self.stop.load(Ordering::Relaxed) < 2,
            "cancelled by second Ctrl-C"
        );
        #[cfg(feature = "system-audio")]
        {
            if let Some(microphone) = &self.microphone {
                microphone.check()?;
            }
            if let Some(speaker) = &self.speaker {
                speaker.check()?;
            }
        }
        Ok(())
    }
    fn playback_ticket(&mut self, output_samples: u64) -> Result<u64> {
        #[cfg(feature = "system-audio")]
        if let Some(speaker) = &mut self.speaker {
            return Ok(speaker.flush_segment()?);
        }
        Ok(output_samples)
    }
    fn played(&self, ticket: u64) -> bool {
        #[cfg(feature = "system-audio")]
        if let Some(speaker) = &self.speaker {
            return speaker.completed(ticket);
        }
        let _ = ticket;
        true
    }
    fn interrupt(&mut self) -> Result<()> {
        #[cfg(feature = "system-audio")]
        if let Some(speaker) = &mut self.speaker {
            speaker.interrupt()?;
        }
        Ok(())
    }
}
