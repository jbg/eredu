//! Emit reproducible checkpoint correctness and performance evidence.
//!
//! The JSON sidecar contains provenance, token IDs, timings, and memory. The
//! SafeTensors sidecar contains the prefill logits and one row of logits for
//! every token fed through the decode cache.

use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use clap::{Parser, ValueEnum};
use safemlx::{
    memory,
    ops::indexing::{NewAxis, TryIndexOp},
    transforms::async_eval_timed,
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use eredu::{
    api::LoadedModel,
    runtime::media::input::{InputPart, ModelInput},
};
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
use serde::Serialize;

const SCHEMA_VERSION: u32 = 1;
const DEFAULT_PROMPT: &str = "The capital of France is";

#[derive(Debug, Parser)]
#[command(
    about = "Probe a local model checkpoint and emit JSON plus SafeTensors evidence",
    after_help = "Example:\n  cargo run -p eredu --features cuda --example checkpoint_probe -- --model snapshots/model --device gpu --output validation/results/model"
)]
struct Args {
    /// Local Hugging Face-compatible checkpoint directory.
    #[arg(long)]
    model: PathBuf,

    /// Execution device.
    #[arg(long, value_enum, default_value_t = DeviceKind::Gpu)]
    device: DeviceKind,

    /// Device index (normally zero for a single-GPU job).
    #[arg(long, default_value_t = 0)]
    device_index: i32,

    /// Text to tokenize. Ignored only when --input-ids is supplied.
    #[arg(long)]
    prompt: Option<String>,

    /// Exact comma-delimited prompt token IDs; bypasses tokenization.
    #[arg(long, value_delimiter = ',', num_args = 1.., conflicts_with = "prompt")]
    input_ids: Option<Vec<u32>>,

    /// Ask the tokenizer to add its configured special tokens.
    #[arg(long, default_value_t = false)]
    add_special_tokens: bool,

    /// Number of greedy tokens to feed through the cache when no teacher-forced IDs are given.
    #[arg(long, default_value_t = 16)]
    decode_steps: usize,

    /// Exact comma-delimited decode tokens to feed; overrides --decode-steps.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    teacher_forced_ids: Option<Vec<u32>>,

    /// Untimed full runs before the measured run.
    #[arg(long, default_value_t = 1)]
    warmup_runs: usize,

    /// Output prefix; `.json` and `.safetensors` are appended/replaced.
    #[arg(long)]
    output: PathBuf,

    /// Replace existing probe artifacts.
    #[arg(long, default_value_t = false)]
    overwrite: bool,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum DeviceKind {
    Cpu,
    Gpu,
}

impl DeviceKind {
    fn mlx(self) -> DeviceType {
        match self {
            Self::Cpu => DeviceType::Cpu,
            Self::Gpu => DeviceType::Gpu,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
        }
    }
}

#[derive(Debug, Serialize)]
struct ProbeReport {
    schema_version: u32,
    model: ModelReport,
    runtime: RuntimeReport,
    input: InputReport,
    output: OutputReport,
    timings: TimingReport,
    memory: MemoryReport,
}

#[derive(Debug, Serialize)]
struct ModelReport {
    checkpoint_path: String,
    model_type: String,
    model_id_for_template: String,
}

#[derive(Debug, Serialize)]
struct RuntimeReport {
    crate_name: &'static str,
    crate_version: &'static str,
    os: &'static str,
    arch: &'static str,
    cuda_feature: bool,
    device_type: &'static str,
    device_index: i32,
    device_description: String,
}

#[derive(Debug, Serialize)]
struct InputReport {
    prompt: Option<String>,
    add_special_tokens: bool,
    token_ids: Vec<u32>,
    token_count: usize,
}

#[derive(Debug, Serialize)]
struct OutputReport {
    tensor_file: String,
    tensors: BTreeMap<&'static str, TensorReport>,
    fed_token_ids: Vec<u32>,
    greedy_token_ids: Vec<u32>,
    decoded_greedy_tokens: Option<String>,
}

#[derive(Debug, Serialize)]
struct TensorReport {
    dtype: &'static str,
    shape: Vec<usize>,
    semantics: &'static str,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    scope: &'static str,
    load_wall_seconds: f64,
    warmup_runs: usize,
    warmup_wall_seconds: f64,
    prefill_wall_seconds: f64,
    prefill_device_seconds: f64,
    prefill_tokens_per_device_second: Option<f64>,
    decode_wall_seconds: Vec<f64>,
    decode_device_seconds: Vec<f64>,
    decode_total_wall_seconds: f64,
    decode_total_device_seconds: f64,
    decode_tokens_per_device_second: Option<f64>,
}

#[derive(Debug, Serialize)]
struct MemoryReport {
    units: &'static str,
    scope: &'static str,
    after_load: MemorySnapshot,
    before_measured_run: MemorySnapshot,
    after_measured_run: MemorySnapshot,
    measured_run_peak_active_bytes: usize,
    process_peak_rss_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct MemorySnapshot {
    active_bytes: usize,
    cache_bytes: usize,
    peak_active_bytes: usize,
}

struct RunOutput {
    prefill_logits: Vec<f32>,
    decode_logits: Vec<f32>,
    vocab_size: usize,
    fed_token_ids: Vec<u32>,
    greedy_token_ids: Vec<u32>,
    prefill_wall: Duration,
    prefill_device: Duration,
    decode_wall: Vec<Duration>,
    decode_device: Vec<Duration>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.device_index >= 0,
        "--device-index must be non-negative"
    );
    ensure!(
        args.model.is_dir(),
        "checkpoint is not a directory: {}",
        args.model.display()
    );

    let json_path = args.output.with_extension("json");
    let tensor_path = args.output.with_extension("safetensors");
    ensure_distinct_outputs(&json_path, &tensor_path)?;
    ensure_output_policy(&json_path, &tensor_path, args.overwrite)?;
    create_parent(&json_path)?;
    create_parent(&tensor_path)?;

    let device = Device::new(args.device.mlx(), args.device_index);
    let device_description = device.to_string();
    let context = ExecutionContext::new(device);
    let stream = context.stream();
    let weights_context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let weights_stream = weights_context.stream();

    memory::reset_peak_memory()?;
    let load_started = Instant::now();
    let mut model = LoadedModel::load(&args.model, stream, weights_stream)
        .with_context(|| format!("failed to load checkpoint {}", args.model.display()))?;
    stream.synchronize()?;
    weights_stream.synchronize()?;
    let load_wall = load_started.elapsed();
    let after_load = memory_snapshot()?;

    let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);
    let (prompt_text, input_ids) = match args.input_ids {
        Some(ids) => (None, ids),
        None => (
            Some(prompt.to_owned()),
            model.encode(prompt, args.add_special_tokens)?,
        ),
    };
    ensure!(!input_ids.is_empty(), "the prompt encoded to zero tokens");

    let forced_ids = args.teacher_forced_ids.as_deref();
    let decode_steps = forced_ids.map_or(args.decode_steps, <[u32]>::len);
    let warmup_started = Instant::now();
    for _ in 0..args.warmup_runs {
        run_probe(&mut model, &input_ids, forced_ids, decode_steps, stream)?;
    }
    let warmup_wall = warmup_started.elapsed();

    memory::reset_peak_memory()?;
    let before_measured_run = memory_snapshot()?;
    let run = run_probe(&mut model, &input_ids, forced_ids, decode_steps, stream)?;
    let after_measured_run = memory_snapshot()?;
    let measured_run_peak_active_bytes = memory::peak_memory()?;

    write_tensors(&tensor_path, &input_ids, &run)?;

    let decode_total_wall: Duration = run.decode_wall.iter().copied().sum();
    let decode_total_device: Duration = run.decode_device.iter().copied().sum();
    let prefill_tokens_per_device_second = (!run.prefill_device.is_zero())
        .then(|| input_ids.len() as f64 / run.prefill_device.as_secs_f64());
    let decode_tokens_per_device_second = (!run.fed_token_ids.is_empty()
        && !decode_total_device.is_zero())
    .then(|| run.fed_token_ids.len() as f64 / decode_total_device.as_secs_f64());

    let tensors = tensor_report(&input_ids, &run);
    // Under teacher forcing these are independent predictions conditioned on
    // the supplied path, not one contiguous generated sequence.
    let decoded_greedy_tokens = forced_ids
        .is_none()
        .then(|| model.decode(&run.greedy_token_ids, false).ok())
        .flatten();
    let report = ProbeReport {
        schema_version: SCHEMA_VERSION,
        model: ModelReport {
            checkpoint_path: args.model.display().to_string(),
            model_type: model.model_type().to_owned(),
            model_id_for_template: model.model_id_for_template().to_owned(),
        },
        runtime: RuntimeReport {
            crate_name: env!("CARGO_PKG_NAME"),
            crate_version: env!("CARGO_PKG_VERSION"),
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            cuda_feature: cfg!(feature = "cuda"),
            device_type: args.device.label(),
            device_index: args.device_index,
            device_description,
        },
        input: InputReport {
            prompt: prompt_text,
            add_special_tokens: args.add_special_tokens,
            token_count: input_ids.len(),
            token_ids: input_ids,
        },
        output: OutputReport {
            tensor_file: tensor_path.display().to_string(),
            tensors,
            fed_token_ids: run.fed_token_ids,
            greedy_token_ids: run.greedy_token_ids,
            decoded_greedy_tokens,
        },
        timings: TimingReport {
            scope: "wall timings cover graph construction and synchronized model evaluation; device timings cover the model graph and exclude logit readback/serialization",
            load_wall_seconds: load_wall.as_secs_f64(),
            warmup_runs: args.warmup_runs,
            warmup_wall_seconds: warmup_wall.as_secs_f64(),
            prefill_wall_seconds: run.prefill_wall.as_secs_f64(),
            prefill_device_seconds: run.prefill_device.as_secs_f64(),
            prefill_tokens_per_device_second,
            decode_wall_seconds: durations_as_seconds(&run.decode_wall),
            decode_device_seconds: durations_as_seconds(&run.decode_device),
            decode_total_wall_seconds: decode_total_wall.as_secs_f64(),
            decode_total_device_seconds: decode_total_device.as_secs_f64(),
            decode_tokens_per_device_second,
        },
        memory: MemoryReport {
            units: "bytes",
            scope: "MLX allocator counters; process_peak_rss_bytes is OS-wide process peak RSS",
            after_load,
            before_measured_run,
            after_measured_run,
            measured_run_peak_active_bytes,
            process_peak_rss_bytes: process_peak_rss_bytes(),
        },
    };

    let json = serde_json::to_vec_pretty(&report)?;
    fs::write(&json_path, json)
        .with_context(|| format!("failed to write {}", json_path.display()))?;
    println!("json={}", json_path.display());
    println!("safetensors={}", tensor_path.display());
    Ok(())
}

fn run_probe(
    model: &mut LoadedModel,
    input_ids: &[u32],
    teacher_forced_ids: Option<&[u32]>,
    decode_steps: usize,
    stream: &Stream,
) -> Result<RunOutput> {
    let mut cache = model.new_cache();
    let prompt_tokens = Array::from(input_ids).try_index_device(NewAxis, stream)?;
    let prompt_parts = [InputPart::text_token_ids(&prompt_tokens)];

    let prefill_started = Instant::now();
    let prefill =
        model.prefill_input_with_cache(ModelInput::new(&prompt_parts), &mut cache, stream)?;
    let prefill_timed = async_eval_timed([&prefill], stream)?;
    let prefill_device = prefill_timed.elapsed()?;
    let prefill_wall = prefill_started.elapsed();
    let (prefill_logits, vocab_size) = copy_logits(&prefill, stream)?;
    let mut greedy_token_ids = vec![argmax(&prefill_logits)?];
    let mut fed_token_ids = Vec::with_capacity(decode_steps);
    let mut decode_logits = Vec::with_capacity(decode_steps.saturating_mul(vocab_size));
    let mut decode_wall = Vec::with_capacity(decode_steps);
    let mut decode_device = Vec::with_capacity(decode_steps);

    for step in 0..decode_steps {
        let token_id = teacher_forced_ids
            .map(|ids| ids[step])
            .unwrap_or_else(|| *greedy_token_ids.last().expect("prefill prediction exists"));
        fed_token_ids.push(token_id);
        let token = Array::from(&[token_id][..]).try_index_device(NewAxis, stream)?;

        let wall_started = Instant::now();
        let logits = model.decode_text_with_cache(&token, &mut cache, stream)?;
        let timed = async_eval_timed([&logits], stream)?;
        let device_elapsed = timed.elapsed()?;
        let wall_elapsed = wall_started.elapsed();
        let (values, step_vocab_size) = copy_logits(&logits, stream)?;
        ensure!(
            step_vocab_size == vocab_size,
            "vocabulary changed from {vocab_size} to {step_vocab_size} at decode step {step}"
        );
        greedy_token_ids.push(argmax(&values)?);
        decode_logits.extend(values);
        decode_wall.push(wall_elapsed);
        decode_device.push(device_elapsed);
    }

    Ok(RunOutput {
        prefill_logits,
        decode_logits,
        vocab_size,
        fed_token_ids,
        greedy_token_ids,
        prefill_wall,
        prefill_device,
        decode_wall,
        decode_device,
    })
}

fn copy_logits(logits: &Array, stream: &Stream) -> Result<(Vec<f32>, usize)> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 2 && shape[0] == 1 && shape[1] > 0,
        "expected logits shaped [1, vocab], got {shape:?}"
    );
    let vocab_size = usize::try_from(shape[1])?;
    let logits = logits.as_type::<f32>(stream)?;
    let evaluated = logits.evaluated()?;
    Ok((evaluated.as_slice::<f32>().to_vec(), vocab_size))
}

fn argmax(values: &[f32]) -> Result<u32> {
    let (index, _) = values
        .iter()
        .enumerate()
        .filter(|(_, value)| !value.is_nan())
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .context("cannot choose argmax from empty or all-NaN logits")?;
    u32::try_from(index).context("vocabulary index does not fit in u32")
}

fn write_tensors(path: &Path, input_ids: &[u32], run: &RunOutput) -> Result<()> {
    let input_bytes = u32_bytes(input_ids);
    let fed_bytes = u32_bytes(&run.fed_token_ids);
    let greedy_bytes = u32_bytes(&run.greedy_token_ids);
    let prefill_bytes = f32_bytes(&run.prefill_logits);
    let decode_bytes = f32_bytes(&run.decode_logits);
    let views = [
        (
            "input.token_ids",
            TensorView::new(SafeDtype::U32, vec![input_ids.len()], &input_bytes)?,
        ),
        (
            "output.fed_token_ids",
            TensorView::new(SafeDtype::U32, vec![run.fed_token_ids.len()], &fed_bytes)?,
        ),
        (
            "output.greedy_token_ids",
            TensorView::new(
                SafeDtype::U32,
                vec![run.greedy_token_ids.len()],
                &greedy_bytes,
            )?,
        ),
        (
            "prefill.logits",
            TensorView::new(SafeDtype::F32, vec![1, run.vocab_size], &prefill_bytes)?,
        ),
        (
            "decode.logits",
            TensorView::new(
                SafeDtype::F32,
                vec![run.fed_token_ids.len(), run.vocab_size],
                &decode_bytes,
            )?,
        ),
    ];
    let metadata = HashMap::from([
        ("schema_version".to_owned(), SCHEMA_VERSION.to_string()),
        (
            "decode_semantics".to_owned(),
            "row i is the next-token logits after feeding output.fed_token_ids[i]".to_owned(),
        ),
    ]);
    serialize_to_file(views, Some(metadata), path)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn tensor_report(input_ids: &[u32], run: &RunOutput) -> BTreeMap<&'static str, TensorReport> {
    BTreeMap::from([
        (
            "decode.logits",
            TensorReport {
                dtype: "F32",
                shape: vec![run.fed_token_ids.len(), run.vocab_size],
                semantics: "row i predicts the token after fed_token_ids[i]",
            },
        ),
        (
            "input.token_ids",
            TensorReport {
                dtype: "U32",
                shape: vec![input_ids.len()],
                semantics: "exact prompt IDs supplied to the model",
            },
        ),
        (
            "output.fed_token_ids",
            TensorReport {
                dtype: "U32",
                shape: vec![run.fed_token_ids.len()],
                semantics: "tokens appended to the KV cache",
            },
        ),
        (
            "output.greedy_token_ids",
            TensorReport {
                dtype: "U32",
                shape: vec![run.greedy_token_ids.len()],
                semantics: "argmax of prefill logits followed by every decode row",
            },
        ),
        (
            "prefill.logits",
            TensorReport {
                dtype: "F32",
                shape: vec![1, run.vocab_size],
                semantics: "next-token logits after the full prompt",
            },
        ),
    ])
}

fn memory_snapshot() -> Result<MemorySnapshot> {
    Ok(MemorySnapshot {
        active_bytes: memory::active_memory()?,
        cache_bytes: memory::cache_memory()?,
        peak_active_bytes: memory::peak_memory()?,
    })
}

fn durations_as_seconds(values: &[Duration]) -> Vec<f64> {
    values.iter().map(Duration::as_secs_f64).collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn ensure_distinct_outputs(json: &Path, tensors: &Path) -> Result<()> {
    if json == tensors {
        bail!("JSON and SafeTensors output paths must differ");
    }
    Ok(())
}

fn ensure_output_policy(json: &Path, tensors: &Path, overwrite: bool) -> Result<()> {
    if !overwrite {
        ensure!(
            !json.exists(),
            "refusing to replace {}; pass --overwrite to replace probe artifacts",
            json.display()
        );
        ensure!(
            !tensors.exists(),
            "refusing to replace {}; pass --overwrite to replace probe artifacts",
            tensors.display()
        );
    }
    Ok(())
}

fn create_parent(path: &Path) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    Ok(())
}

#[cfg(unix)]
fn process_peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage initializes the supplied rusage on success.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return None;
    }
    // SAFETY: the successful call above initialized usage.
    let rss = unsafe { usage.assume_init() }.ru_maxrss;
    let rss = u64::try_from(rss).ok()?;
    if cfg!(target_os = "macos") {
        Some(rss)
    } else {
        rss.checked_mul(1024)
    }
}

#[cfg(not(unix))]
fn process_peak_rss_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_ignores_nan_and_uses_first_for_ties() {
        assert_eq!(argmax(&[f32::NAN, 4.0, 4.0, 3.0]).unwrap(), 1);
    }

    #[test]
    fn byte_encoders_are_little_endian() {
        assert_eq!(u32_bytes(&[0x0102_0304]), [4, 3, 2, 1]);
        assert_eq!(f32_bytes(&[1.0]), 1.0f32.to_le_bytes());
    }

    #[test]
    fn safetensors_contract_round_trips() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("probe.safetensors");
        let run = RunOutput {
            prefill_logits: vec![1.0, 2.0, 3.0],
            decode_logits: vec![4.0, 5.0, 6.0],
            vocab_size: 3,
            fed_token_ids: vec![2],
            greedy_token_ids: vec![2, 1],
            prefill_wall: Duration::ZERO,
            prefill_device: Duration::ZERO,
            decode_wall: vec![Duration::ZERO],
            decode_device: vec![Duration::ZERO],
        };

        write_tensors(&path, &[7, 8], &run).unwrap();
        let bytes = fs::read(path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(tensors.tensor("prefill.logits").unwrap().shape(), [1, 3]);
        assert_eq!(tensors.tensor("decode.logits").unwrap().shape(), [1, 3]);
        assert_eq!(tensors.tensor("input.token_ids").unwrap().shape(), [2]);
    }
}
