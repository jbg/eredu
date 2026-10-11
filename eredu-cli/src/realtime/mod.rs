//! Application-owned PCM I/O and options over the shared controlled speech session.

mod display;
mod events;
mod host;
mod load;
mod run;

use super::{resolve_model, CachedGgufRole, CliDevice};
use anyhow::{ensure, Context, Result};
use clap::{Args as ClapArgs, ValueEnum};
use eredu::api::realtime::RealtimeArtifactSource;

use eredu_architectures::voicechat::quantization::SpeechQuantizationPolicy;
use eredu_media::audio::wav::{WavReader, WavWriter};
use std::{
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{atomic::AtomicUsize, Arc},
    time::Instant,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuantizedComponent {
    Backbone,
    FunctionHead,
    Tts,
}

#[derive(Debug, ClapArgs)]
#[command(group(clap::ArgGroup::new("bounded_residency")
    .args(["layerwise_host", "dense_disk_stream"])
    .requires_all(["backbone_device_bytes", "backbone_host_bytes", "speech_device_bytes", "speech_host_bytes"])))]
pub struct Args {
    #[command(flatten)]
    common: crate::common::CommonOptions,

    /// JSON conversation, auxiliary-asset, residency and resource configuration.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Official VoiceChat auxiliary assets, or their cached repository ID.
    #[arg(long)]
    official_assets: Option<String>,
    /// Base tokenizer/configuration assets, or their cached repository ID.
    #[arg(long)]
    base_assets: Option<String>,
    /// List system audio devices and exit without loading a model.
    #[arg(long)]
    list_audio_devices: bool,
    /// Mono 16 kHz WAV input. Omit to use the microphone.
    #[arg(long)]
    input: Option<PathBuf>,
    /// Record source PCM accepted by the session as mono 16 kHz float32 WAV.
    #[arg(long)]
    record_input: Option<PathBuf>,
    /// Hide input transcription, grouped on stderr by default.
    #[arg(long)]
    no_transcription: bool,
    /// Save committed mono 22.05 kHz float32 speech to a new WAV.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Save committed events and wall-clock execution measurements as JSONL.
    #[arg(long)]
    events: Option<PathBuf>,
    /// Recorded tool responses; tools themselves are supplied by the JSON config.
    #[arg(long)]
    script: Option<PathBuf>,
    /// Named microphone; omission selects the platform default.
    #[arg(long, conflicts_with = "input")]
    input_device: Option<String>,
    /// Named speaker; omission selects the platform default.
    #[arg(long, conflicts_with = "no_playback")]
    output_device: Option<String>,
    /// Disable system playback (useful for unattended WAV benchmarks).
    #[arg(long)]
    no_playback: bool,
    /// Process file input without wall-clock pacing; requires --input --no-playback.
    #[arg(long, requires_all = ["input", "no_playback"])]
    offline: bool,
    /// Maximum microphone duration before graceful input closure and response tail.
    #[arg(long)]
    seconds: Option<f64>,
    /// Finite native input/output queue duration, in milliseconds.
    #[arg(long, default_value_t = 2000, value_parser = clap::value_parser!(u32).range(160..=10000))]
    audio_buffer_ms: u32,
    /// Matrix groups affected by --quantize.
    #[arg(
        long,
        value_enum,
        value_delimiter = ',',
        default_value = "backbone,function-head,tts",
        requires = "quantize"
    )]
    quantize_components: Vec<QuantizedComponent>,
    /// Floating storage for eligible unquantized speech/perception parameters.
    #[arg(long, value_enum)]
    speech_weights: Option<load::SpeechFloatChoice>,
    /// Floating arithmetic for eligible unquantized speech/perception operations.
    #[arg(long, value_enum)]
    speech_compute: Option<load::SpeechFloatChoice>,
    /// System prompt override (ASCII).
    #[arg(long)]
    system_prompt: Option<String>,
    #[arg(long)]
    backbone_device_bytes: Option<u64>,
    #[arg(long)]
    backbone_host_bytes: Option<u64>,
    /// Each independent speech provider's device allowance.
    #[arg(long)]
    speech_device_bytes: Option<u64>,
    /// Each independent speech provider's host allowance; zero permits direct disk streaming.
    #[arg(long)]
    speech_host_bytes: Option<u64>,
    #[arg(long)]
    max_frames: Option<u64>,
    #[arg(long)]
    max_model_steps: Option<u64>,
    #[arg(long)]
    interruption_speech_frames: Option<u64>,
    #[arg(long, default_value_t = 640)]
    chunk_samples: usize,
    #[arg(long, default_value_t = 200)]
    tail_frames: u64,
    #[arg(long, default_value_t = 100000)]
    max_steps: u64,
    #[cfg(feature = "projection-profiling")]
    #[arg(long)]
    projection_profile: Option<PathBuf>,
    #[cfg(feature = "projection-profiling")]
    #[arg(long, default_value_t = 22, requires = "projection_profile")]
    profile_advance: u64,
    #[cfg(feature = "projection-profiling")]
    #[arg(long, default_value_t = 5, requires = "projection_profile")]
    profile_samples: usize,
    #[cfg(feature = "projection-profiling")]
    #[arg(long, requires = "projection_profile")]
    profile_metal: Option<PathBuf>,
    #[cfg(feature = "projection-profiling")]
    #[arg(long, default_value_t = 0, requires = "profile_metal")]
    profile_metal_case: usize,
}

fn json<T: serde::de::DeserializeOwned>(path: &Path, limit: u64) -> Result<T> {
    let mut bytes = Vec::new();
    File::open(path)?.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}
fn source(path: PathBuf) -> RealtimeArtifactSource {
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
    {
        RealtimeArtifactSource::Gguf { path }
    } else {
        RealtimeArtifactSource::SafeTensors { path }
    }
}
fn configuration(args: &Args) -> Result<load::Config> {
    let mut config = if let Some(path) = &args.config {
        let mut config: load::Config = json(path, 1_048_576)?;
        config.resolve_paths(path.parent().unwrap_or_else(|| Path::new(".")));
        config
    } else {
        let model = args
            .common
            .model
            .as_deref()
            .context("provide --model or --config")?;
        let path = resolve_model(
            model,
            args.common.revision.as_deref(),
            CachedGgufRole::Target,
        )?
        .path;
        let official = args.official_assets.as_deref().unwrap_or(model);
        let base = args.base_assets.as_deref().context("provide --base-assets (base tokenizer/configuration directory or cached repository ID)")?;
        serde_json::from_value(serde_json::json!({
            "official_assets":resolve_model(official, None, CachedGgufRole::Target)?.path,
            "base_assets":resolve_model(base, None, CachedGgufRole::Target)?.path,
            "source":source(path), "system_prompt":"You are a helpful assistant.", "tools":[],
            "device_parameter_bytes":68719476736_u64, "host_parameter_bytes":68719476736_u64,
            "max_branch_bytes":8589934592_u64, "max_total_copy_bytes":1099511627776_u64
        }))?
    };
    if let Some(model) = &args.common.model {
        config.source = source(
            resolve_model(
                model,
                args.common.revision.as_deref(),
                CachedGgufRole::Target,
            )?
            .path,
        );
    }
    if let Some(official) = &args.official_assets {
        config.official_assets = resolve_model(official, None, CachedGgufRole::Target)?.path;
    }
    if let Some(base) = &args.base_assets {
        config.base_assets = resolve_model(base, None, CachedGgufRole::Target)?.path;
    }
    if let Some(seed) = args.common.seed {
        config.seed = seed;
    }
    macro_rules! set { ($($field:ident),*) => { $(if let Some(value) = &args.$field { config.$field = value.clone(); })* }; }
    set!(
        speech_weights,
        speech_compute,
        system_prompt,
        max_frames,
        max_model_steps
    );
    if let Some(value) = args.interruption_speech_frames {
        config.interruption_speech_frames = Some(value);
    }
    if let Some(value) = args.common.device_budget_bytes {
        config.device_parameter_bytes = value;
    }
    if let Some(value) = args.common.host_budget_bytes {
        config.host_parameter_bytes = value;
    }
    if args.common.layerwise_host || args.common.dense_disk_stream {
        config.parameter_residency = if args.common.layerwise_host {
            load::ParameterResidency::Host {
                backbone_device_bytes: args
                    .backbone_device_bytes
                    .context("bounded speech residency requires --backbone-device-bytes")?,
                backbone_host_bytes: args
                    .backbone_host_bytes
                    .context("bounded speech residency requires --backbone-host-bytes")?,
                speech_device_bytes: args
                    .speech_device_bytes
                    .context("bounded speech residency requires --speech-device-bytes")?,
                speech_host_bytes: args
                    .speech_host_bytes
                    .context("bounded speech residency requires --speech-host-bytes")?,
                depth: args.common.device_layer_window,
            }
        } else {
            load::ParameterResidency::Disk {
                backbone_device_bytes: args
                    .backbone_device_bytes
                    .context("bounded speech residency requires --backbone-device-bytes")?,
                backbone_host_bytes: args
                    .backbone_host_bytes
                    .context("bounded speech residency requires --backbone-host-bytes")?,
                speech_device_bytes: args
                    .speech_device_bytes
                    .context("bounded speech residency requires --speech-device-bytes")?,
                speech_host_bytes: args
                    .speech_host_bytes
                    .context("bounded speech residency requires --speech-host-bytes")?,
            }
        };
    }
    config.validate()?;
    Ok(config)
}
fn quantization(args: &Args) -> Result<SpeechQuantizationPolicy> {
    let selected = args.common.weight_quantization()?;
    let mut policy = SpeechQuantizationPolicy::default();
    for component in &args.quantize_components {
        match component {
            QuantizedComponent::Backbone => policy.backbone = selected,
            QuantizedComponent::FunctionHead => policy.function_head = selected,
            QuantizedComponent::Tts => policy.tts = selected,
        }
    }
    Ok(policy)
}
fn output_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {}; output paths must be new", path.display()))
}

pub fn run(args: Args) -> Result<()> {
    if args.list_audio_devices {
        #[cfg(feature = "system-audio")]
        {
            for device in eredu_media::audio::system::devices()? {
                println!(
                    "{}\tinput={}\toutput={}",
                    device.name, device.input, device.output
                );
            }
            return Ok(());
        }
        #[cfg(not(feature = "system-audio"))]
        anyhow::bail!("device enumeration requires the `system-audio` feature");
    }
    args.common.validate()?;
    ensure!(
        args.chunk_samples > 0 && args.chunk_samples <= 1_048_576 && args.max_steps > 0,
        "invalid chunk or step limit"
    );
    ensure!(
        args.seconds.is_none_or(|v| v.is_finite() && v > 0.0),
        "--seconds must be finite and positive"
    );
    #[cfg(not(feature = "system-audio"))]
    ensure!(
        args.input.is_some() && args.no_playback,
        "microphone and speaker require the `system-audio` feature"
    );
    let reserved_frames = args
        .tail_frames
        .checked_add(2)
        .context("response tail frame count overflow")?;
    let policy = quantization(&args)?;
    let config = configuration(&args)?;
    ensure!(
        config.max_frames > reserved_frames,
        "--max-frames must exceed the response tail plus two frames"
    );
    let scripts = args
        .script
        .as_ref()
        .map(|p| json::<Vec<run::ScriptedToolResponse>>(p, 4_194_304))
        .transpose()?
        .unwrap_or_default();
    run::validate_scripts(&scripts)?;
    let input = args
        .input
        .as_ref()
        .map(|p| WavReader::new(BufReader::new(File::open(p)?)))
        .transpose()?;
    if let Some(input) = &input {
        ensure!(
            input.sample_rate() == 16_000,
            "input must be mono 16 kHz WAV"
        );
        ensure!(
            input
                .sample_count()
                .div_ceil(1280)
                .checked_add(args.tail_frames)
                .is_some_and(|frames| frames <= config.max_frames),
            "input plus response tail exceeds --max-frames"
        );
    }
    let paths = [
        args.output.as_ref(),
        args.events.as_ref(),
        args.record_input.as_ref(),
    ]
    .into_iter()
    .flatten()
    .map(|p| destination(p))
    .collect::<Result<Vec<_>>>()?;
    for path in &paths {
        ensure!(
            !path.try_exists()?,
            "output already exists: {}",
            path.display()
        );
    }
    ensure!(
        paths.iter().collect::<std::collections::HashSet<_>>().len() == paths.len(),
        "speech, input recording and events must have different paths"
    );
    #[cfg(feature = "projection-profiling")]
    if let Some(path) = &args.projection_profile {
        let path = destination(path)?;
        ensure!(
            !path.try_exists()? && !paths.contains(&path),
            "profile output must be a new distinct path"
        );
        ensure!(
            args.offline,
            "diagnostic replay requires --offline to avoid interrupting live audio"
        );
    }
    args.common.configure_runtime()?;
    let device = args.common.device.unwrap_or_else(|| match config.device {
        load::Device::Cpu => {
            if args.config.is_some() {
                CliDevice::Cpu
            } else {
                CliDevice::default()
            }
        }
        load::Device::Gpu => CliDevice::Gpu(0),
    });
    let started = Instant::now();
    let sampling = eredu_architectures::voicechat::agent_sampling::AgentSamplingPolicy::new(
        args.common.temperature.unwrap_or(0.0),
        args.common.top_p.unwrap_or(1.0),
        args.common.repeat_penalty.unwrap_or(1.0),
        usize::try_from(config.max_model_steps)?,
    )?;
    let mut session = load::load(
        &config,
        policy,
        device.local(),
        args.common.cached_shards,
        sampling,
    )?;
    eprintln!(
        "loaded in {:.3} s on {device}",
        started.elapsed().as_secs_f64()
    );
    let stop = Arc::new(AtomicUsize::new(0));
    #[cfg(feature = "system-audio")]
    {
        let signal = stop.clone();
        ctrlc::set_handler(move || {
            signal.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })?;
    }
    let waveform = args
        .output
        .as_ref()
        .map(|p| -> Result<_> { Ok(WavWriter::new(BufWriter::new(output_file(p)?), 22_050)?) })
        .transpose()?;
    let input_recording = args
        .record_input
        .as_ref()
        .map(|p| -> Result<_> { Ok(WavWriter::new(BufWriter::new(output_file(p)?), 16_000)?) })
        .transpose()?;
    let mut io = host::HostIo::new(
        input,
        waveform,
        input_recording,
        &args,
        stop,
        (config.max_frames - reserved_frames) as f64 * 0.08,
    )?;
    let event_file: Box<dyn Write> = args
        .events
        .as_ref()
        .map(|p| output_file(p).map(|f| Box::new(BufWriter::new(f)) as Box<dyn Write>))
        .transpose()?
        .unwrap_or_else(|| Box::new(std::io::sink()));
    let mut events = events::Events::new(event_file, config.max_frames, !args.no_transcription)?;
    let options = run::RunOptions {
        chunk_samples: args.chunk_samples,
        tail_frames: args.tail_frames,
        max_steps: args.max_steps,
        #[cfg(feature = "projection-profiling")]
        profile: args
            .projection_profile
            .as_ref()
            .map(|p| run::ProfileOptions {
                output: p.clone(),
                advance: args.profile_advance,
                samples: args.profile_samples,
                metal: args.profile_metal.clone(),
                metal_case: args.profile_metal_case,
            }),
    };
    eprintln!(
        "{}; Ctrl-C closes input and drains the response, a second Ctrl-C cancels",
        if args.input.is_none() {
            "microphone ready"
        } else {
            "WAV input ready"
        }
    );
    let result = run::run_io(&mut session, &mut io, &mut events, options, scripts);
    let finalize = io.finish(result.is_ok());
    let event_finish = events.flush();
    events.summary();
    result?;
    finalize?;
    event_finish?;
    Ok(())
}
fn destination(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent
        .canonicalize()?
        .join(path.file_name().context("output requires a file name")?))
}
