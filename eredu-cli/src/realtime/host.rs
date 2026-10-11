use super::{run::AudioIo, Args};
use anyhow::{ensure, Result};
use eredu_media::audio::admission::{SpeechAdmission, SpeechAdmissionConfig};
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
    admission: Option<SpeechAdmission>,
    admitted: std::collections::VecDeque<f32>,
    admission_finished: bool,
    #[cfg(feature = "system-audio")]
    ptt: Option<super::ptt::Ptt>,
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
        let (microphone, speaker) = if file.is_none()
            && !args.no_playback
            && !args.no_echo_cancellation
        {
            let (input, output) = eredu_media::audio::system::duplex(
                args.input_device.as_deref(),
                args.output_device.as_deref(),
                16_000,
                22_050,
                args.audio_buffer_ms,
                args.noise_suppression.processing(),
            )?;
            eprintln!("echo cancellation: WebRTC AEC3; microphone remains live during playback");
            (Some(input), Some(output))
        } else {
            let input = if file.is_none() {
                Some(SystemAudioInput::new(
                    args.input_device.as_deref(),
                    16_000,
                    args.audio_buffer_ms,
                    args.noise_suppression.processing(),
                )?)
            } else {
                None
            };
            let output = if args.no_playback {
                None
            } else {
                Some(SystemAudioOutput::new(
                    args.output_device.as_deref(),
                    22_050,
                    args.audio_buffer_ms,
                )?)
            };
            (input, output)
        };
        let admission = args
            .speech_admission
            .then(|| {
                SpeechAdmission::new(
                    16_000,
                    SpeechAdmissionConfig {
                        threshold_dbfs: args.speech_threshold_dbfs,
                        hysteresis_db: args.speech_hysteresis_db as f32,
                        min_speech_ms: args.speech_min_ms,
                        preroll_ms: args.speech_preroll_ms,
                        hangover_ms: args.speech_hangover_ms,
                    },
                )
            })
            .transpose()?;
        if let Some(admission) = &admission {
            eprintln!("speech admission: threshold={} dBFS, hysteresis={} dB, minimum={} ms, pre-roll={} ms, hangover={} ms, lookahead={} ms", args.speech_threshold_dbfs, args.speech_hysteresis_db, args.speech_min_ms, args.speech_preroll_ms, args.speech_hangover_ms, admission.delay_ms());
        }
        #[cfg(feature = "system-audio")]
        if !matches!(args.noise_suppression, super::NoiseSuppression::Off) {
            eprintln!("noise suppression: WebRTC {:?}", args.noise_suppression);
        }
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
            microphone.set_capture_enabled(args.push_to_talk.is_none());
            microphone.start()?;
        }
        #[cfg(feature = "system-audio")]
        let ptt = args
            .push_to_talk
            .map(|mode| super::ptt::Ptt::new(mode, &args.ptt_key, stop.clone()))
            .transpose()?;
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
            admission,
            admitted: Default::default(),
            admission_finished: false,
            #[cfg(feature = "system-audio")]
            ptt,
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
    fn poll_controls(&mut self) -> Result<()> {
        #[cfg(feature = "system-audio")]
        if let Some(ptt) = &mut self.ptt {
            if let Some(enabled) = ptt.poll()? {
                if let Some(microphone) = &self.microphone {
                    microphone.set_capture_enabled(enabled);
                }
            }
        }
        Ok(())
    }
    fn read_source(&mut self, buffer: &mut [f32]) -> Result<Option<usize>> {
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
    pub fn finish(&mut self, success: bool) -> Result<()> {
        #[cfg(feature = "system-audio")]
        drop(self.ptt.take());
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
        self.poll_controls()?;
        if self.admission.is_none() {
            return self.read_source(buffer);
        }
        ensure!(!buffer.is_empty(), "empty audio read buffer");
        let mut source = [0.0; 1024];
        while self.admitted.is_empty() && !self.admission_finished {
            match self.read_source(&mut source)? {
                Some(0) => {
                    self.admitted
                        .extend(self.admission.as_mut().unwrap().finish()?);
                    self.admission_finished = true;
                }
                Some(count) => self
                    .admitted
                    .extend(self.admission.as_mut().unwrap().push(&source[..count])?),
                None => return Ok(None),
            }
        }
        let count = buffer.len().min(self.admitted.len());
        for value in &mut buffer[..count] {
            *value = self.admitted.pop_front().unwrap();
        }
        Ok(if count != 0 || self.admission_finished {
            Some(count)
        } else {
            None
        })
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
        self.poll_controls()?;
        if self.paced {
            while self.start.elapsed() < until {
                self.poll_controls()?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[derive(Parser)]
    struct Options {
        #[command(flatten)]
        args: Args,
    }

    #[test]
    fn speech_admission_in_host_io_preserves_clock_and_flushes_exact_source_tail() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let mut writer = WavWriter::new(file.reopen().unwrap(), 16_000).unwrap();
        let mut pcm = vec![0.01; 967];
        pcm[320..640].fill(0.2);
        writer.write_samples(&pcm).unwrap();
        writer.finish().unwrap();
        let args = Options::try_parse_from([
            "test",
            "--input",
            "fixture.wav",
            "--no-playback",
            "--offline",
            "--speech-admission",
            "--speech-threshold-dbfs",
            "-20",
            "--speech-min-ms",
            "20",
            "--speech-preroll-ms",
            "0",
            "--speech-hangover-ms",
            "0",
        ])
        .unwrap()
        .args;
        assert!(args.push_to_talk.is_none());
        let source = WavReader::new(BufReader::new(file.reopen().unwrap())).unwrap();
        let mut io = HostIo::new(
            Some(source),
            None,
            None,
            &args,
            Arc::new(AtomicUsize::new(0)),
            10.0,
        )
        .unwrap();
        let mut output = Vec::new();
        let mut buffer = [0.0; 13];
        loop {
            match io.read(&mut buffer).unwrap() {
                Some(0) => break,
                Some(count) => {
                    io.accepted_input(&buffer[..count]).unwrap();
                    output.extend_from_slice(&buffer[..count]);
                }
                None => panic!("offline input must make progress"),
            }
        }
        let mut expected = vec![0.0; pcm.len()];
        expected[320..640].fill(0.2);
        assert_eq!(output, expected);
        assert_eq!(io.accepted_samples, 967);
    }
}
