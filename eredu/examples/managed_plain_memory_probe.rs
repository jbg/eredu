//! Public plain-text token and memory probe; run one mode per fresh process.
//!
//! managed_plain_memory_probe CHECKPOINT PROMPT.txt ordinary|managed|controlled OUTPUT.json
//!   [--max-tokens N] [--chunk N] [--budget BYTES] [--tokenizer TOKENIZER.json]
//!   [--device cpu|accelerator|metal] [--seed N] [--no-special-tokens] [--reference REPORT.json]
//!
//! `--budget` is required for managed/controlled and is not applied to ordinary.
//! The ordinary public token iterator runs to its finite output limit; it does
//! not implement the managed text driver's EOS stopping. Reports expose the
//! first EOS position and full produced IDs, including any ordinary post-EOS tail.
//! An optional reference must contain prompt_token_ids and generated_token_ids;
//! comparison is exact token equality, never numerical score equivalence.
//!
//! Allocator peaks are reset after loading, reference encoding and source setup.
//! They include retained model allocations. RSS samples are not process peaks;
//! on macOS, use `/usr/bin/time -l` for an independent process high-water mark.
//! The requested budget covers the managed domain, not all process memory or
//! this probe's reference IDs, telemetry, JSON, or caller-owned output copies.
use anyhow::{Context, bail, ensure};
use eredu::api::{
    LoadedModel, LocalDevice, ManagedPlainTextRequest, ManagedPlainTextSource,
    PreparedChatGenerationSettings, local_device_plan, reset_local_allocator_peak,
};
use eredu_backend_mlx::{MlxBackendFactory, backend::MlxBackend};
use eredu_core::{
    ExecutionPlan, FinishReason, GenerationCancellationToken, GenerationConfigOverrides,
    GenerationPlainTextEvent, GenerationPlainTextOutput, ResolvedGenerationConfig,
    TextGenerationConfig, TextInferencePolicy,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::Write, num::NonZeroU64, path::PathBuf, process::Command, time::Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ordinary,
    Managed,
    Controlled,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::Managed => "managed",
            Self::Controlled => "controlled",
        }
    }
}
struct Options {
    checkpoint: PathBuf,
    prompt: PathBuf,
    mode: Mode,
    output: PathBuf,
    max_tokens: usize,
    chunk: Option<NonZeroU64>,
    budget: Option<u64>,
    tokenizer: Option<PathBuf>,
    device: String,
    seed: u64,
    special_tokens: bool,
    reference: Option<PathBuf>,
}
impl Options {
    fn parse() -> anyhow::Result<Self> {
        let mut args = std::env::args().skip(1);
        let usage = "usage: managed_plain_memory_probe CHECKPOINT PROMPT.txt ordinary|managed|controlled OUTPUT.json [--max-tokens N] [--chunk N] [--budget BYTES] [--tokenizer FILE] [--device cpu|accelerator|metal] [--seed N] [--no-special-tokens] [--reference REPORT.json]";
        let checkpoint = args.next().context(usage)?.into();
        let prompt = args.next().context(usage)?.into();
        let mode = match args.next().context(usage)?.as_str() {
            "ordinary" => Mode::Ordinary,
            "managed" => Mode::Managed,
            "controlled" => Mode::Controlled,
            other => bail!("unknown mode {other}; {usage}"),
        };
        let output = args.next().context(usage)?.into();
        let mut options = Self {
            checkpoint,
            prompt,
            mode,
            output,
            max_tokens: 32,
            chunk: None,
            budget: None,
            tokenizer: None,
            device: "accelerator".into(),
            seed: 0,
            special_tokens: true,
            reference: None,
        };
        while let Some(flag) = args.next() {
            if flag == "--no-special-tokens" {
                options.special_tokens = false;
                continue;
            }
            let value = args
                .next()
                .with_context(|| format!("missing value for {flag}"))?;
            match flag.as_str() {
                "--max-tokens" => options.max_tokens = value.parse()?,
                "--chunk" => options.chunk = Some(value.parse()?),
                "--budget" => options.budget = Some(value.parse()?),
                "--tokenizer" => options.tokenizer = Some(value.into()),
                "--device" => {
                    ensure!(
                        matches!(value.as_str(), "cpu" | "accelerator" | "metal"),
                        "device must be cpu, accelerator or metal"
                    );
                    options.device = value;
                }
                "--seed" => options.seed = value.parse()?,
                "--reference" => options.reference = Some(value.into()),
                _ => bail!("unknown option {flag}; {usage}"),
            }
        }
        ensure!(options.max_tokens > 0, "max-tokens must be positive");
        ensure!(
            options.mode == Mode::Ordinary || options.budget.is_some(),
            "managed and controlled require --budget BYTES"
        );
        ensure!(
            !options.output.exists(),
            "output already exists: {}",
            options.output.display()
        );
        // Keep supplied artifacts and comparison evidence separate from output.
        ensure!(
            options.output != options.prompt
                && options.output != options.checkpoint
                && options.reference.as_ref() != Some(&options.output)
                && options.tokenizer.as_ref() != Some(&options.output),
            "output must be a distinct path"
        );
        Ok(options)
    }
}

fn process_rss_bytes() -> anyhow::Result<u64> {
    let result = Command::new("/bin/ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()?;
    ensure!(
        result.status.success(),
        "ps failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let kib: u64 = std::str::from_utf8(&result.stdout)?.trim().parse()?;
    kib.checked_mul(1024).context("process RSS overflow")
}
fn sample(model: &LoadedModel<MlxBackend<'_>>) -> Value {
    // This public method synchronizes before sampling the MLX counters.
    let allocator = model.allocator_telemetry();
    let rss = process_rss_bytes();
    json!({
        "mlx": allocator.as_ref().ok(),
        "mlx_error": allocator.as_ref().err().map(ToString::to_string),
        "process_rss_bytes": rss.as_ref().ok(),
        "process_rss_error": rss.as_ref().err().map(ToString::to_string),
    })
}
fn error_chain(error: &anyhow::Error) -> Vec<String> {
    error.chain().map(ToString::to_string).collect()
}

#[derive(Default)]
struct Events {
    deltas: usize,
    text_bytes: usize,
    finishes: usize,
    finished: Option<FinishReason>,
}
impl Events {
    fn emit(&mut self, event: GenerationPlainTextEvent<'_>) {
        match event {
            GenerationPlainTextEvent::TextDelta(text) => {
                self.deltas += 1;
                self.text_bytes += text.len();
            }
            GenerationPlainTextEvent::Finished { reason } => {
                self.finishes += 1;
                self.finished = Some(reason);
            }
        }
    }
}
enum Output {
    Raw(Vec<u32>),
    Plain(GenerationPlainTextOutput),
}
impl Output {
    fn ids(&self) -> &[u32] {
        match self {
            Self::Raw(ids) => ids,
            Self::Plain(output) => output.token_ids.as_ref(),
        }
    }
    fn finish_reason(&self) -> Option<FinishReason> {
        match self {
            Self::Raw(_) => None,
            Self::Plain(output) => Some(output.finish_reason),
        }
    }
}
// Caller-owned scalar observation of the accepted report, retained even if a
// later generation step fails. No Admission/report/account is copied or held.
#[derive(Clone, Copy)]
struct SelectedPreparation {
    prefill_chunk_positions: u64,
    incremental_required_bytes: u64,
}
impl SelectedPreparation {
    fn from_report(report: eredu_core::TextPreparationReport<'_>) -> Self {
        Self {
            prefill_chunk_positions: report.geometry.prefill_chunk_positions,
            incremental_required_bytes: report.admission.incremental_required_bytes,
        }
    }
}

struct Completed {
    output: Output,
    first_token_seconds: Option<f64>,
    seconds: f64,
    advances: Option<usize>,
    events: Events,
}
fn generate(
    model: &mut LoadedModel<MlxBackend<'_>>,
    options: &Options,
    prompt: &str,
    ordinary_prompt: Vec<u32>,
    source: Option<&ManagedPlainTextSource>,
    settings: PreparedChatGenerationSettings,
    resolved: ResolvedGenerationConfig,
    selected: &mut Option<SelectedPreparation>,
) -> anyhow::Result<Completed> {
    let started = Instant::now();
    if options.mode == Mode::Ordinary {
        let mut ids = Vec::with_capacity(options.max_tokens);
        let mut first = None;
        let config = TextGenerationConfig::new(resolved)
            .with_seed(settings.seed)
            .with_inference_policy(settings.inference);
        let generation = model.generate_tokens(ordinary_prompt, config)?;
        *selected = generation.preparation_report().map(SelectedPreparation::from_report);
        for token in generation {
            ids.push(token?.token_id()?);
            first.get_or_insert_with(|| started.elapsed().as_secs_f64());
        }
        return Ok(Completed {
            output: Output::Raw(ids),
            first_token_seconds: first,
            seconds: started.elapsed().as_secs_f64(),
            advances: None,
            events: Events::default(),
        });
    }
    let source = source.context("managed tokenizer source missing")?;
    let cancellation = GenerationCancellationToken::new();
    let mut request = ManagedPlainTextRequest::new(prompt, settings);
    request.add_special_tokens = options.special_tokens;
    let mut events = Events::default();
    let mut emit = |event: GenerationPlainTextEvent<'_>| events.emit(event);
    eprintln!("{} preparation started", options.mode.name());
    let session = model
        .start_managed_plain_text(source, request, &cancellation)?
        .context("unexpected pre-start cancellation")?;
    *selected = session.preparation_report().map(SelectedPreparation::from_report);
    if let Some(report) = selected.as_ref() {
        eprintln!(
            "{} preparation completed in {:.3}s: chunk={}, admitted_incremental_bytes={}",
            options.mode.name(),
            started.elapsed().as_secs_f64(),
            report.prefill_chunk_positions,
            report.incremental_required_bytes,
        );
    }
    let (output, advances) = if options.mode == Mode::Managed {
        (session.run(&cancellation, &mut emit)?, None)
    } else {
        let mut session = session;
        let mut advances = 0;
        while session.finish_reason().is_none() {
            session = session.advance(&cancellation, &mut emit)?;
            advances += 1;
        }
        let output = session
            .into_output()
            .map_err(|_| anyhow::anyhow!("session is not terminal"))?;
        (output, Some(advances))
    };
    let seconds = started.elapsed().as_secs_f64();
    ensure!(
        events.finishes == 1 && events.finished == Some(output.finish_reason),
        "unexpected terminal event count"
    );
    ensure!(
        events.text_bytes == output.text.as_str().len(),
        "borrowed text delivery length disagrees with output"
    );
    Ok(Completed {
        first_token_seconds: output.timing.time_to_first_token().map(|d| d.as_secs_f64()),
        output: Output::Plain(output),
        seconds,
        advances,
        events,
    })
}

#[derive(Deserialize)]
struct Reference {
    prompt_token_ids: Vec<u32>,
    generated_token_ids: Vec<u32>,
}
fn compare(reference: &Reference, prompt: &[u32], generated: &[u32]) -> Value {
    let common = reference
        .generated_token_ids
        .iter()
        .zip(generated)
        .take_while(|(left, right)| left == right)
        .count();
    json!({
        "prompt_ids_equal": reference.prompt_token_ids == prompt,
        "generated_ids_equal": reference.generated_token_ids == generated,
        "exact_match": reference.prompt_token_ids == prompt && reference.generated_token_ids == generated,
        "common_generated_prefix_length": common,
        "reference_generated_length": reference.generated_token_ids.len(),
        "actual_generated_length": generated.len(),
        "comparison": "exact token IDs only; no logits or numerical score comparison",
    })
}

fn run(options: &Options) -> anyhow::Result<bool> {
    let prompt = std::fs::read_to_string(&options.prompt).context("read UTF-8 prompt")?;
    let reference = options
        .reference
        .as_ref()
        .map(|path| -> anyhow::Result<Reference> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    let device = match options.device.as_str() {
        "metal" => eredu_core::DevicePlan::new("mlx", "metal:0")?,
        "cpu" => local_device_plan(LocalDevice::Cpu)?,
        _ => local_device_plan(LocalDevice::Accelerator(0))?,
    };
    let execution = ExecutionPlan::fully_resident(device);
    let load_started = Instant::now();
    let (mut model, _) = LoadedModel::load_execution_plan(
        &MlxBackendFactory::default(),
        &options.checkpoint,
        &execution,
    )?
    .into_parts();
    let after_load = sample(&model);
    let load_seconds = load_started.elapsed().as_secs_f64();
    let encoding_started = Instant::now();
    let prompt_ids = model.encode(&prompt, options.special_tokens)?;
    ensure!(!prompt_ids.is_empty(), "prompt encoded to no tokens");
    let reference_encoding_seconds = encoding_started.elapsed().as_secs_f64();
    let tokenizer_fingerprint = model
        .tokenizer()
        .fingerprint()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let tokenizer_path = options.tokenizer.clone().unwrap_or_else(|| {
        let root = if options.checkpoint.is_dir() {
            options.checkpoint.as_path()
        } else {
            options
                .checkpoint
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
        };
        root.join("tokenizer.json")
    });
    let setup_started = Instant::now();
    let source = if options.mode == Mode::Ordinary {
        Ok(None)
    } else {
        std::fs::File::open(&tokenizer_path)
            .context("open matching tokenizer.json; use --tokenizer for a sidecar")
            .and_then(|file| Ok(Some(model.compile_managed_plain_text_source(file)?)))
    };
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    let overrides = GenerationConfigOverrides {
        do_sample: Some(false),
        temperature: Some(0.0),
        top_k: Some(1),
        top_p: Some(1.0),
        min_p: Some(0.0),
        repetition_penalty: Some(1.0),
        repeat_last_n: Some(0),
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        max_new_tokens: Some(options.max_tokens),
    };
    let resolved = model.resolve_generation_config(overrides)?;
    let settings = PreparedChatGenerationSettings {
        overrides,
        seed: options.seed,
        inference: TextInferencePolicy {
            prefill_chunk_positions: options.chunk,
            managed_memory_capacity_bytes: if options.mode == Mode::Ordinary {
                None
            } else {
                options.budget
            },
            submission_tracking_capacity_bytes: None,
            graph_metadata_capacity_bytes: None,
        },
        ..Default::default()
    };
    let ordinary_prompt = if options.mode == Mode::Ordinary {
        prompt_ids.clone()
    } else {
        Vec::new()
    };
    let before_generation = sample(&model);
    reset_local_allocator_peak()?;
    // Keep failure custody or the successful retained output alive through the final sample.
    let mut selected = None;
    let result = match source.as_ref() {
        Ok(source) => generate(
            &mut model,
            options,
            &prompt,
            ordinary_prompt,
            source.as_ref(),
            settings,
            resolved,
            &mut selected,
        ),
        Err(error) => Err(anyhow::anyhow!("tokenizer source setup failed: {error:#}")),
    };
    let after_generation = sample(&model);
    let mut report = json!({
        "schema_version": 1, "mode": options.mode.name(), "checkpoint": options.checkpoint,
        "checkpoint_canonical_path": std::fs::canonicalize(&options.checkpoint).ok(),
        "checkpoint_identity_scope": "path only; pin weights/revision separately for independent reproduction",
        "model_family": model.model_family().canonical_name(), "effective_model_type": model.effective_model_type(),
        "execution_plan": execution, "prompt_text": prompt,
        "prompt_sha256": Sha256::digest(prompt.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
        "prompt_token_ids": prompt_ids, "prompt_encoding": "LoadedModel::encode, outside generation timing",
        "add_special_tokens": options.special_tokens, "tokenizer_vocabulary_fingerprint": tokenizer_fingerprint,
        "managed_tokenizer_source": if options.mode == Mode::Ordinary { None } else { Some(&tokenizer_path) },
        "managed_input_encoding": if options.mode == Mode::Ordinary { Value::Null } else { json!("independently admitted from text using the authenticated same tokenizer configuration") },
        "eos_token_ids": model.eos_token_ids(), "resolved_generation_config": resolved, "seed": options.seed,
        "inference_policy": settings.inference, "requested_total_budget_bytes": options.budget,
        "selected_prefill_chunk_positions": selected.map(|value| value.prefill_chunk_positions),
        "admitted_incremental_required_bytes": selected.map(|value| value.incremental_required_bytes),
        "selected_chunk_note": "actual accepted request geometry; null means no accepted-admission report was available",
        "admission_requirement_note": "historical accepted incremental working-memory bound including proved existing-storage credit; not measured high-water or current managed-domain total",
        "managed_capacity_scope": "managed domain including retained residency, not total process RSS",
        "output_scope": if options.mode == Mode::Ordinary { "full raw finite iterator" } else { "shared driver committed output through termination" },
        "termination_policy": if options.mode == Mode::Ordinary { "raw public token iterator; no EOS stopping" }
            else { "shared managed plain-text termination; EOS/max-tokens; no literal stop strings" },
        "load_and_sync_seconds": load_seconds, "reference_encoding_seconds": reference_encoding_seconds,
        "managed_source_setup_seconds": setup_seconds, "generation_attempted": source.is_ok(),
        "memory": { "after_load": after_load, "before_generation": before_generation, "after_generation_with_output_or_error_retained": after_generation },
        "process_peak_rss_bytes": Value::Null,
        "measurement_notes": "MLX peak reset immediately before generation; counters include retained baseline. RSS samples are not high-water; use /usr/bin/time -l on macOS. Probe buffers and serialization are outside managed budget.",
        "score_comparison": "none; use the separate historical readout_memory_probe for full-score evidence",
    });
    let mut success = false;
    match &result {
        Ok(completed) => {
            let ids = completed.output.ids();
            report["status"] = json!("ok");
            report["generated_token_ids"] = json!(ids);
            report["first_generated_eos_position"] =
                json!(ids.iter().position(|id| model.eos_token_ids().contains(id)));
            report["finish_reason"] = json!(completed.output.finish_reason());
            // Independent public decoding is outside measured generation. A decode
            // failure must not discard the actual token and memory evidence.
            for (key, skip_special) in [
                ("decoded_output_skip_special", true),
                ("decoded_output_with_special", false),
            ] {
                match model.decode(ids, skip_special) {
                    Ok(text) => report[key] = json!({"text": text}),
                    Err(error) => report[key] = json!({"error": error.to_string()}),
                }
            }
            report["visible_managed_text"] = match &completed.output {
                Output::Plain(output) => json!(output.text.as_str()),
                Output::Raw(_) => Value::Null,
            };
            report["generation_seconds"] = json!(completed.seconds);
            report["first_token_seconds"] = json!(completed.first_token_seconds);
            report["first_token_timing_scope"] = json!(if options.mode == Mode::Ordinary {
                "wall time through first public token observation"
            } else {
                "shared driver first-commit active timing"
            });
            report["controlled_advance_calls"] = json!(completed.advances);
            report["text_delta_events"] = json!(completed.events.deltas);
            report["text_delta_bytes"] = json!(completed.events.text_bytes);
            success = true;
            if let Some(reference) = &reference {
                let comparison = compare(reference, &prompt_ids, ids);
                success = comparison["exact_match"].as_bool().unwrap_or(false);
                if !success {
                    report["status"] = json!("token_mismatch");
                }
                report["reference_comparison"] = comparison;
                report["reference_path"] = json!(options.reference);
            }
        }
        Err(error) => {
            report["status"] = json!("error");
            report["error_phase"] = json!(if source.is_err() {
                "managed_source_setup"
            } else {
                "generation"
            });
            report["error_chain"] = json!(error_chain(error));
            if let Err(error) = &source {
                report["source_setup_error_chain"] = json!(error_chain(error));
            }
            report["generated_prefix_available"] = json!(false);
        }
    }
    write_report(options, &report)?;
    eprintln!(
        "{} report written to {}",
        options.mode.name(),
        options.output.display()
    );
    Ok(success)
}
fn write_report(options: &Options, report: &Value) -> anyhow::Result<()> {
    let bytes = serde_json::to_vec_pretty(report)?;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&options.output)
        .context("create new report")?;
    file.write_all(&bytes)?;
    file.write_all(b"\n")?;
    Ok(())
}
fn main() -> anyhow::Result<()> {
    let options = Options::parse()?;
    let success = match run(&options) {
        Ok(success) => success,
        Err(error) => {
            let report = json!({
                "schema_version": 1, "mode": options.mode.name(), "status": "probe_error",
                "checkpoint": options.checkpoint, "prompt_path": options.prompt,
                "error_chain": error_chain(&error),
                "scope": "setup or report failure; no successful inference or budget-enforcement claim",
            });
            if let Err(write_error) = write_report(&options, &report) {
                eprintln!("could not write error report: {write_error:#}");
            }
            return Err(error);
        }
    };
    ensure!(
        success,
        "generation failed or reference token IDs differed; see {}",
        options.output.display()
    );
    Ok(())
}
