//! Bounded WAV input and recorded tool responses through the public VoiceChat session.

#[path = "voicechat_wav/load.rs"]
mod load;
#[path = "voicechat_wav/run.rs"]
mod run;

use std::{
    collections::BTreeMap,
    fs::{File, OpenOptions},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{ensure, Context, Result};
use eredu_media::audio::wav::{WavReader, WavWriter};
use serde::de::DeserializeOwned;

const HELP: &str = "Usage: voicechat_wav --config CONFIG.json --input INPUT.wav \
    --output SPEECH.wav --events EVENTS.jsonl [--script RESPONSES.json] \
    [--chunk-samples 640] [--tail-frames 200] [--max-steps 100000]

Input: mono 16 kHz PCM16/24/32 or float32 WAV. Output: mono 22.05 kHz float32 WAV.
Output files must not already exist. Speech chunks are concatenated; JSONL records
retain audio timestamps and sample offsets. Execution uses recorded time, without
real-time pacing. Tool scripts supply responses; this example does not execute tools.
After EOF, bounded silence supports active tool work, followed by --tail-frames
live response frames before drain. Explicit session and step bounds remain enforced.
See doc/voicechat-validation.md for configuration, pinned sources and validation limits.";

struct Arguments {
    config: PathBuf,
    input: PathBuf,
    output: PathBuf,
    events: PathBuf,
    script: Option<PathBuf>,
    run: run::RunOptions,
}

impl Arguments {
    fn parse() -> Result<Option<Self>> {
        let mut args = std::env::args_os().skip(1);
        let mut values = BTreeMap::new();
        while let Some(key) = args.next() {
            if key == "--help" || key == "-h" {
                println!("{HELP}");
                return Ok(None);
            }
            let key = key.to_str().context("option name is not UTF-8")?;
            ensure!(
                matches!(
                    key,
                    "--config"
                        | "--input"
                        | "--output"
                        | "--events"
                        | "--script"
                        | "--chunk-samples"
                        | "--tail-frames"
                        | "--max-steps"
                ),
                "unknown option {key}; use --help"
            );
            let value = args
                .next()
                .with_context(|| format!("missing value for {key}"))?;
            ensure!(
                values.insert(key.to_owned(), value).is_none(),
                "duplicate option {key}"
            );
        }
        let mut required = |name: &str| -> Result<PathBuf> {
            Ok(values
                .remove(name)
                .with_context(|| format!("missing {name}; use --help"))?
                .into())
        };
        let config = required("--config")?;
        let input = required("--input")?;
        let output = required("--output")?;
        let events = required("--events")?;
        let script = values.remove("--script").map(PathBuf::from);
        let mut number = |name: &str, default: u64| -> Result<u64> {
            match values.remove(name) {
                None => Ok(default),
                Some(value) => value
                    .to_str()
                    .context("numeric option is not UTF-8")?
                    .parse()
                    .with_context(|| format!("invalid nonnegative integer for {name}")),
            }
        };
        let chunk_samples = usize::try_from(number("--chunk-samples", 640)?)?;
        ensure!(
            (1..=1_048_576).contains(&chunk_samples),
            "--chunk-samples must be in 1..=1048576"
        );
        let tail_frames = number("--tail-frames", 200)?;
        let max_steps = number("--max-steps", 100_000)?;
        ensure!(max_steps != 0, "--max-steps must be positive");
        Ok(Some(Self {
            config,
            input,
            output,
            events,
            script,
            run: run::RunOptions {
                chunk_samples,
                tail_frames,
                max_steps,
            },
        }))
    }
}

fn read_json<T: DeserializeOwned>(path: &Path, limit: u64) -> Result<T> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() as u64 <= limit,
        "{} exceeds {limit} bytes",
        path.display()
    );
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn create_output(path: &Path) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("creating {}; output paths must be new", path.display()))
}

fn output_destination(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    Ok(parent
        .canonicalize()?
        .join(path.file_name().context("output requires a file name")?))
}

fn main() -> Result<()> {
    let Some(args) = Arguments::parse()? else {
        return Ok(());
    };
    let mut config: load::Config = read_json(&args.config, 1_048_576)?;
    config.resolve_paths(args.config.parent().unwrap_or_else(|| Path::new(".")));
    config.validate()?;
    let scripts = match &args.script {
        Some(path) => read_json(path, 4_194_304)?,
        None => Vec::<run::ScriptedToolResponse>::new(),
    };
    run::validate_scripts(&scripts)?;
    let mut input = WavReader::new(BufReader::new(File::open(&args.input)?))?;
    ensure!(
        input.sample_rate() == 16_000,
        "input WAV must be mono 16 kHz"
    );
    let input_frames = input.sample_count().div_ceil(1280);
    ensure!(
        input_frames
            .checked_add(args.run.tail_frames)
            .is_some_and(|frames| frames <= config.max_frames),
        "input plus trailing silence exceeds config.max_frames"
    );
    // Fail before opening weights if either output is already present. create_new
    // below also protects against a file appearing after this check.
    let output_path = output_destination(&args.output)?;
    let events_path = output_destination(&args.events)?;
    for path in [&output_path, &events_path] {
        ensure!(
            !path.try_exists()?,
            "output already exists: {}",
            path.display()
        );
    }
    ensure!(
        output_path != events_path,
        "speech and event paths must differ"
    );
    let mut session = load::load(&config)?;
    let mut speech = WavWriter::new(BufWriter::new(create_output(&output_path)?), 22_050)?;
    let mut events = BufWriter::new(create_output(&events_path)?);
    let result = run::run(
        &mut session,
        &mut input,
        &mut speech,
        &mut events,
        args.run,
        scripts,
    );
    // Finalize committed samples even when a later transition fails. The driver
    // cancels with its last recorded timestamp and retains completion ownership.
    let waveform_result = speech.finish().and_then(|mut file| {
        file.flush()?;
        Ok(())
    });
    let events_result = events.flush();
    if let Err(error) = result {
        if let Err(finalize) = waveform_result {
            return Err(error.context(format!("finalizing partial WAV also failed: {finalize}")));
        }
        if let Err(flush) = events_result {
            return Err(error.context(format!("flushing events also failed: {flush}")));
        }
        return Err(error);
    }
    waveform_result?;
    events_result?;
    eprintln!(
        "wrote {} and {}",
        args.output.display(),
        args.events.display()
    );
    Ok(())
}
