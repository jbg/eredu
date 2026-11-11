//! CLI host I/O for one audio request.

use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    path::PathBuf,
};

use anyhow::{bail, ensure, Context, Result};
use clap::{Args as ClapArgs, ValueEnum};
use eredu::api::{
    audio::{AudioTask, AudioTaskOutput, PreparedAudioTask},
    realtime::{
        prepare_realtime_model, RealtimeArtifactRequest, RealtimeArtifactSource,
        RealtimeAssetDirectories,
    },
};
use eredu_backend_mlx::native::ExecutionContext;
use eredu_media::audio::wav::{WavReader, WavWriter};

use crate::{common::output_file, resolve_model, CachedGgufRole, CliDevice};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Task {
    Transcribe,
    Translate,
    Speak,
    Respond,
}

#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Task to perform.
    #[arg(value_enum)]
    task: Task,
    /// Local model path or cached Hugging Face model identifier.
    #[arg(long)]
    model: String,
    /// Cached model revision.
    #[arg(long)]
    revision: Option<String>,
    /// Local projector GGUF for a split checkpoint.
    #[arg(long, requires = "assets")]
    projector: Option<PathBuf>,
    /// Local model asset directory, required for split GGUF.
    #[arg(long)]
    assets: Option<PathBuf>,
    /// Input mono WAV at the model's required sample rate.
    #[arg(long, conflicts_with = "microphone_seconds")]
    input: Option<PathBuf>,
    /// Capture this many seconds from a microphone.
    #[arg(long)]
    microphone_seconds: Option<f64>,
    /// Microphone device name.
    #[arg(long, requires = "microphone_seconds")]
    input_device: Option<String>,
    /// Text to synthesize, or an optional instruction accompanying a spoken request.
    #[arg(long)]
    text: Option<String>,
    /// Source language for translation.
    #[arg(long)]
    source_language: Option<String>,
    /// Target language for translation.
    #[arg(long)]
    language: Option<String>,
    /// Write synthesized PCM to a new WAV file.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Play synthesized PCM on the system output.
    #[arg(long)]
    play: bool,
    /// Speaker device name.
    #[arg(long, requires = "play")]
    output_device: Option<String>,
    /// Execution device: cpu or gpu:N.
    #[arg(long)]
    device: Option<CliDevice>,
    /// Maximum committed text tokens and speech frames in the response.
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(1..))]
    max_slots: u32,
    /// Response sampling seed.
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

pub fn run(args: Args) -> Result<()> {
    let task = match args.task {
        Task::Transcribe => AudioTask::Transcribe,
        Task::Translate => AudioTask::Translate {
            source_language: args
                .source_language
                .as_deref()
                .filter(|v| !v.trim().is_empty())
                .context("translation requires --source-language")?,
            language: args
                .language
                .as_deref()
                .filter(|v| !v.trim().is_empty())
                .context("translation requires --language")?,
        },
        Task::Speak => AudioTask::Speak {
            text: args
                .text
                .as_deref()
                .filter(|v| !v.trim().is_empty())
                .context("speech synthesis requires --text")?,
        },
        Task::Respond => AudioTask::Respond {
            instruction: args.text.as_deref().unwrap_or(""),
        },
    };
    ensure!(
        (args.language.is_none() && args.source_language.is_none())
            || matches!(args.task, Task::Translate),
        "language options apply to translation"
    );
    ensure!(
        args.text.is_none() || matches!(args.task, Task::Speak | Task::Respond),
        "--text applies to speak or respond"
    );
    let takes_pcm = !matches!(args.task, Task::Speak);
    ensure!(
        takes_pcm == (args.input.is_some() || args.microphone_seconds.is_some()),
        "this task {}",
        if takes_pcm {
            "requires --input or --microphone-seconds"
        } else {
            "takes text input only"
        }
    );
    if let Some(seconds) = args.microphone_seconds {
        ensure!(
            seconds.is_finite() && seconds > 0.0,
            "--microphone-seconds must be finite and positive"
        );
    }
    #[cfg(not(feature = "system-audio"))]
    ensure!(
        args.microphone_seconds.is_none() && !args.play,
        "device I/O requires the system-audio feature"
    );

    let model = resolve_model(
        &args.model,
        args.revision.as_deref(),
        CachedGgufRole::Target,
    )?
    .path;
    let source = match &args.projector {
        Some(projector) => RealtimeArtifactSource::AudioGguf {
            backbone: model,
            projector: projector.clone(),
        },
        None => {
            ensure!(
                !model
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf")),
                "a split audio GGUF requires --projector and --assets"
            );
            RealtimeArtifactSource::SafeTensors { path: model }
        }
    };
    let artifact = prepare_realtime_model(&RealtimeArtifactRequest {
        source,
        assets: RealtimeAssetDirectories {
            model: args.assets.clone(),
            text_backbone: None,
        },
    })?;
    let prepared = PreparedAudioTask::new(&artifact, task, args.max_slots as usize, args.seed)
        .map_err(anyhow::Error::from_boxed)?;
    let input_rate = prepared.input_sample_rate();
    let output_rate = prepared.output_sample_rate();
    ensure!(
        output_rate.is_some() == (args.output.is_some() || args.play),
        "{}",
        if output_rate.is_some() {
            "speech output requires --output or --play"
        } else {
            "this task outputs text to stdout"
        }
    );
    let mut wav_input = args
        .input
        .as_ref()
        .map(|path| WavReader::new(BufReader::new(File::open(path)?)).map_err(anyhow::Error::from))
        .transpose()?;
    if let Some(input) = &wav_input {
        ensure!(
            Some(input.sample_rate()) == input_rate,
            "expected a mono {} Hz WAV",
            input_rate.unwrap_or_default()
        );
        ensure!(input.sample_count() > 0, "audio input is empty");
    }
    let weights = ExecutionContext::cpu()?;
    let execution = match args.device.unwrap_or_default() {
        CliDevice::Cpu => ExecutionContext::cpu()?,
        CliDevice::Gpu(index) => ExecutionContext::accelerator(usize::try_from(index)?)?,
    };
    let mut session = prepared
        .load(&weights, &execution)
        .map_err(anyhow::Error::from_boxed)?;
    let mut stdout = io::stdout().lock();
    let mut publish_input = |samples: &[f32]| -> Result<()> {
        let text = session.push(samples).map_err(anyhow::Error::from_boxed)?;
        stdout.write_all(text.as_bytes())?;
        stdout.flush()?;
        Ok(())
    };
    if let Some(input) = &mut wav_input {
        let mut buffer = [0.0; 640];
        loop {
            let count = input.read_samples(&mut buffer)?;
            if count == 0 {
                break;
            }
            publish_input(&buffer[..count])?;
        }
    }
    #[cfg(feature = "system-audio")]
    if let Some(seconds) = args.microphone_seconds {
        capture(
            &args,
            input_rate.context("model has no microphone input")?,
            seconds,
            publish_input,
        )?;
    }
    if takes_pcm {
        session.finish_input().map_err(anyhow::Error::from_boxed)?;
    }

    let mut wav_output = args
        .output
        .as_ref()
        .map(|path| -> Result<_> {
            let rate = output_rate.context("model has no PCM output")?;
            Ok(WavWriter::new(BufWriter::new(output_file(path)?), rate)?)
        })
        .transpose()?;
    #[cfg(feature = "system-audio")]
    let mut speaker = if args.play {
        Some(eredu_media::audio::system::SystemAudioOutput::new(
            args.output_device.as_deref(),
            output_rate.context("model has no speaker output")?,
            2000,
        )?)
    } else {
        None
    };
    while let Some(output) = session.next_output().map_err(anyhow::Error::from_boxed)? {
        match output {
            AudioTaskOutput::Text(text) => {
                stdout.write_all(text.as_bytes())?;
                stdout.flush()?;
            }
            AudioTaskOutput::Pcm(samples) => {
                if output_rate.is_none() {
                    bail!("selected text task emitted PCM");
                }
                if let Some(wav) = &mut wav_output {
                    wav.write_samples(&samples)?;
                }
                #[cfg(feature = "system-audio")]
                if let Some(speaker) = &mut speaker {
                    speaker.write(&samples)?;
                }
            }
        }
    }
    if let Some(wav) = wav_output {
        wav.finish()?.flush()?;
    }
    #[cfg(feature = "system-audio")]
    if let Some(speaker) = &mut speaker {
        speaker.finish()?;
    }
    if output_rate.is_none() {
        writeln!(stdout)?;
    }
    Ok(())
}

#[cfg(feature = "system-audio")]
fn capture(
    args: &Args,
    rate: u32,
    seconds: f64,
    mut publish: impl FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    use eredu_media::audio::{processing::NoiseSuppression, system::SystemAudioInput};
    use std::time::{Duration, Instant};

    let samples = (seconds * f64::from(rate)).round();
    ensure!(
        samples >= 1.0 && samples < usize::MAX as f64,
        "microphone sample count is out of range"
    );
    let mut remaining = samples as usize;
    let mut input = SystemAudioInput::new(
        args.input_device.as_deref(),
        rate,
        2000,
        NoiseSuppression::Off,
    )?;
    input.start()?;
    let mut buffer = [0.0; 640];
    let mut last_sample = Instant::now();
    while remaining != 0 {
        let capacity = buffer.len().min(remaining);
        match input.read(&mut buffer[..capacity])? {
            Some(0) => bail!("microphone closed before capture completed"),
            Some(count) => {
                remaining -= count;
                last_sample = Instant::now();
                publish(&buffer[..count])?;
            }
            None => {
                ensure!(
                    last_sample.elapsed() < Duration::from_secs(5),
                    "microphone stopped delivering audio"
                );
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }
    input.stop()?;
    Ok(())
}
