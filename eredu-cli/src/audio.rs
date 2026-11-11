//! CLI entry point for turn-based audio tasks and full-duplex conversation.

pub(crate) mod pcm;

use std::{
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use clap::{Args as ClapArgs, Subcommand};
use eredu::api::{
    audio::{AudioTask, AudioTaskOutput, PreparedAudioTask},
    realtime::{
        prepare_realtime_model, RealtimeArtifactRequest, RealtimeArtifactSource,
        RealtimeAssetDirectories,
    },
};

use crate::{resolve_model, CachedGgufRole, CliDevice};

#[derive(Debug, ClapArgs)]
pub struct Args {
    /// Report audio timing statistics to stderr.
    #[arg(long, global = true)]
    timing: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Stream an utterance into a transcript.
    Transcribe(PcmTurnArgs),
    /// Translate an utterance into text.
    Translate(TranslateArgs),
    /// Synthesize speech from text.
    Speak(SpeakArgs),
    /// Generate a spoken response to one utterance.
    Respond(RespondArgs),
    /// Run a full-duplex conversation using system audio or a WAV.
    Realtime(crate::realtime::Args),
}

#[derive(Debug, ClapArgs)]
struct PcmTurnArgs {
    #[command(flatten)]
    model: ModelArgs,
    #[command(flatten)]
    input: InputArgs,
}

#[derive(Debug, ClapArgs)]
struct TranslateArgs {
    #[command(flatten)]
    audio: PcmTurnArgs,
    /// Source language for translation.
    #[arg(long, value_parser = nonblank)]
    source_language: String,
    /// Target language for translation.
    #[arg(long, value_parser = nonblank)]
    language: String,
}

#[derive(Debug, ClapArgs)]
struct SpeakArgs {
    #[command(flatten)]
    model: ModelArgs,
    /// Text to synthesize.
    #[arg(long, value_parser = nonblank)]
    text: String,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, ClapArgs)]
struct RespondArgs {
    #[command(flatten)]
    audio: PcmTurnArgs,
    /// Optional instruction accompanying the spoken request.
    #[arg(long)]
    text: Option<String>,
    #[command(flatten)]
    output: OutputArgs,
}

#[derive(Debug, ClapArgs)]
struct ModelArgs {
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

#[derive(Debug, ClapArgs)]
#[cfg_attr(feature = "system-audio", group(skip))]
#[cfg_attr(feature = "system-audio", command(group(
    clap::ArgGroup::new("audio_input").args(["input", "microphone_seconds"]).required(true)
)))]
struct InputArgs {
    /// Input mono WAV at the model's required sample rate.
    #[arg(long)]
    #[cfg_attr(not(feature = "system-audio"), arg(required = true))]
    input: Option<PathBuf>,
    /// Capture this many seconds from a microphone.
    #[cfg(feature = "system-audio")]
    #[arg(long, value_parser = positive_seconds)]
    microphone_seconds: Option<f64>,
    /// Microphone device name.
    #[cfg(feature = "system-audio")]
    #[arg(long, requires = "microphone_seconds")]
    input_device: Option<String>,
}

#[derive(Debug, ClapArgs)]
#[cfg_attr(feature = "system-audio", group(skip))]
#[cfg_attr(feature = "system-audio", command(group(
    clap::ArgGroup::new("audio_output").args(["output", "play"]).required(true).multiple(true)
)))]
struct OutputArgs {
    /// Write synthesized PCM to a new WAV file.
    #[arg(long)]
    #[cfg_attr(not(feature = "system-audio"), arg(required = true))]
    output: Option<PathBuf>,
    /// Play synthesized PCM on the system output.
    #[cfg(feature = "system-audio")]
    #[arg(long)]
    play: bool,
    /// Speaker device name.
    #[cfg(feature = "system-audio")]
    #[arg(long, requires = "play")]
    output_device: Option<String>,
}

fn nonblank(value: &str) -> Result<String, &'static str> {
    if value.trim().is_empty() {
        Err("value must not be blank")
    } else {
        Ok(value.to_owned())
    }
}

#[cfg(feature = "system-audio")]
fn positive_seconds(value: &str) -> Result<f64, &'static str> {
    match value.parse::<f64>() {
        Ok(seconds) if seconds.is_finite() && seconds > 0.0 => Ok(seconds),
        _ => Err("duration must be finite and positive"),
    }
}

pub fn run(args: Args, total_started: Instant) -> Result<()> {
    let Args { timing, command } = args;
    match command {
        Command::Transcribe(args) => run_turn(
            AudioTask::Transcribe,
            args.model,
            Some(args.input),
            None,
            timing,
            total_started,
        ),
        Command::Translate(args) => run_turn(
            AudioTask::Translate {
                source_language: &args.source_language,
                language: &args.language,
            },
            args.audio.model,
            Some(args.audio.input),
            None,
            timing,
            total_started,
        ),
        Command::Speak(args) => run_turn(
            AudioTask::Speak { text: &args.text },
            args.model,
            None,
            Some(args.output),
            timing,
            total_started,
        ),
        Command::Respond(args) => run_turn(
            AudioTask::Respond {
                instruction: args.text.as_deref().unwrap_or(""),
            },
            args.audio.model,
            Some(args.audio.input),
            Some(args.output),
            timing,
            total_started,
        ),
        Command::Realtime(args) => crate::realtime::run(args, timing, total_started),
    }
}

fn run_turn(
    task: AudioTask<'_>,
    args: ModelArgs,
    input: Option<InputArgs>,
    output: Option<OutputArgs>,
    timing: bool,
    total_started: Instant,
) -> Result<()> {
    let load_started = timing.then(Instant::now);
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
    let mut wav_input = input
        .as_ref()
        .and_then(|input| input.input.as_deref())
        .map(|path| pcm::open_wav(path, input_rate.context("model has no PCM input")?))
        .transpose()?;
    if let Some(input) = &wav_input {
        ensure!(input.sample_count() > 0, "audio input is empty");
    }
    let (weights, execution) = pcm::execution_contexts(args.device.unwrap_or_default().local())?;
    let mut session = prepared
        .load(&weights, &execution)
        .map_err(anyhow::Error::from_boxed)?;
    let load_elapsed = load_started.map(|start| start.elapsed());
    if let Some(elapsed) = load_elapsed {
        eprintln!("model loaded in {:.3} s", elapsed.as_secs_f64());
    }
    let task_started = timing.then(Instant::now);
    let mut first_output = None;
    let mut observe_output = |nonempty: bool| {
        if nonempty && first_output.is_none() {
            first_output = task_started.map(|start| start.elapsed());
        }
    };
    let mut stdout = io::stdout().lock();
    let mut publish_input = |samples: &[f32]| -> Result<()> {
        let text = session.push(samples).map_err(anyhow::Error::from_boxed)?;
        observe_output(!text.is_empty());
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
    if let Some(input) = &input {
        if let Some(seconds) = input.microphone_seconds {
            capture(
                input.input_device.as_deref(),
                input_rate.context("model has no microphone input")?,
                seconds,
                publish_input,
            )?;
        }
    }
    if input.is_some() {
        session.finish_input().map_err(anyhow::Error::from_boxed)?;
    }

    let mut sink = pcm::PcmOutput {
        wav: output
            .as_ref()
            .and_then(|output| output.output.as_deref())
            .map(|path| pcm::create_wav(path, output_rate.context("model has no PCM output")?))
            .transpose()?,
        #[cfg(feature = "system-audio")]
        speaker: if let Some(output) = output.filter(|output| output.play) {
            Some(eredu_media::audio::system::SystemAudioOutput::new(
                output.output_device.as_deref(),
                output_rate.context("model has no speaker output")?,
                2000,
            )?)
        } else {
            None
        },
    };
    while let Some(output) = session.next_output().map_err(anyhow::Error::from_boxed)? {
        match output {
            AudioTaskOutput::Text(text) => {
                observe_output(!text.is_empty());
                stdout.write_all(text.as_bytes())?;
                stdout.flush()?;
            }
            AudioTaskOutput::Pcm(samples) => {
                if output_rate.is_none() {
                    bail!("selected text task emitted PCM");
                }
                observe_output(!samples.is_empty());
                sink.write(&samples)?;
            }
        }
    }
    if let Some(wav) = sink.wav {
        pcm::finish_wav(wav)?;
    }
    #[cfg(feature = "system-audio")]
    if let Some(speaker) = &mut sink.speaker {
        speaker.finish()?;
    }
    if output_rate.is_none() {
        writeln!(stdout)?;
    }
    if let (Some(load), Some(task)) = (load_elapsed, task_started) {
        let task_elapsed = task.elapsed();
        let total = total_started.elapsed();
        eprintln!("load_time: {:.3} s", load.as_secs_f64());
        eprintln!("audio_task_time: {:.3} s", task_elapsed.as_secs_f64());
        match first_output {
            Some(elapsed) => eprintln!("time_to_first_output: {:.3} s", elapsed.as_secs_f64()),
            None => eprintln!("time_to_first_output: n/a"),
        }
        eprintln!("total_execution_time: {:.3} s", total.as_secs_f64());
    }
    Ok(())
}

#[cfg(feature = "system-audio")]
fn capture(
    device: Option<&str>,
    rate: u32,
    seconds: f64,
    mut publish: impl FnMut(&[f32]) -> Result<()>,
) -> Result<()> {
    use eredu_media::audio::{processing::NoiseSuppression, system::SystemAudioInput};
    use std::time::Duration;

    let samples = (seconds * f64::from(rate)).round();
    ensure!(
        samples >= 1.0 && samples < usize::MAX as f64,
        "microphone sample count is out of range"
    );
    let mut remaining = samples as usize;
    let mut input = SystemAudioInput::new(device, rate, 2000, NoiseSuppression::Off)?;
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
