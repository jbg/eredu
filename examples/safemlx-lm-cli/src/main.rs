use std::{
    collections::HashSet,
    fmt,
    io::{self, IsTerminal, Read, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    str::FromStr,
    time::Instant,
};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use hf_hub::{cache::CachedRevisionInfo, HFClientSync};
use safemlx::{
    error::Exception,
    ops::indexing::TryIndexOp,
    random::RandomState,
    transforms::{async_eval, eval},
    Array, Device, DeviceType, ExecutionContext, Stream,
};
use safemlx_lm::{
    api::{
        GenerationCancellationToken, LoadedModel, ModelLoadOptions,
        PreparedChatEmbeddedMtpGenerationRequest, PreparedChatGenerationRequest,
        PreparedChatGenerationSettings, PreparedChatInput, PreparedChatMtpGenerationOptions,
        PreparedChatMtpGenerationRequest, TextDecoder,
    },
    error::Error as LmError,
    runtime::chat::{
        ChatTemplateRequest, NativeToolSupport, ParallelToolCallPolicy, SemanticSupport, ToolChoice,
    },
    runtime::checkpoint::quantization::{AffineQuantization, WeightQuantization},
    runtime::execution::layerwise::{
        LayerwiseLoadOptions, NonExpertWeightResidency, WeightResidency,
    },
    runtime::generation::sampler::{
        DefaultSampler, GenerationSampler, MirostatV2Sampler, Sampler, SpeculativeSampler,
    },
    runtime::generation::speculative::{
        LoadedDrafter, MtpComponentTimingGuard, MtpConfig, MtpExecutionStreams,
        MtpSchedulerOptions, MtpStats,
    },
    runtime::generation::streaming::{FinishReason, SemanticEvent},
    runtime::media::input::{InputPart, ModelInput},
    runtime::residency::dense_stream::DenseDiskStreamLoadOptions,
    runtime::residency::expert_cache::{
        ExpertCacheLoadOptions, ExpertPassStatistics, ExpertTierStatistics,
    },
    runtime::residency::policy::{
        CacheEvictionPolicy, MemoryTier, OffloadConfig, TransferDirection,
    },
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ExpertCacheEviction {
    Lru,
    Lfu,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, ValueEnum)]
enum ThinkingMode {
    /// Preserve the model chat template's default.
    Auto,
    /// Ask a compatible chat template to enable thinking/reasoning.
    On,
    /// Ask a compatible chat template to disable thinking/reasoning.
    Off,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, ValueEnum)]
enum LoadQuantizationMode {
    #[default]
    Affine,
    Mxfp4,
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, ValueEnum)]
enum CliToolChoice {
    None,
    #[default]
    Auto,
    Required,
}

impl From<CliToolChoice> for ToolChoice {
    fn from(value: CliToolChoice) -> Self {
        match value {
            CliToolChoice::None => Self::None,
            CliToolChoice::Auto => Self::Auto,
            CliToolChoice::Required => Self::Required,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum CliDevice {
    Cpu,
    Gpu(i32),
}

impl CliDevice {
    fn mlx(self) -> Device {
        match self {
            Self::Cpu => Device::new(DeviceType::Cpu, 0),
            Self::Gpu(index) => Device::new(DeviceType::Gpu, index),
        }
    }
}

impl fmt::Display for CliDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => formatter.write_str("cpu"),
            Self::Gpu(index) => write!(formatter, "gpu:{index}"),
        }
    }
}

impl FromStr for CliDevice {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value == "cpu" {
            return Ok(Self::Cpu);
        }
        let Some(index) = value.strip_prefix("gpu:") else {
            return Err("expected `cpu` or `gpu:N`".into());
        };
        let index = index
            .parse::<i32>()
            .map_err(|_| "GPU index must be a non-negative integer".to_string())?;
        if index < 0 {
            return Err("GPU index must be a non-negative integer".into());
        }
        Ok(Self::Gpu(index))
    }
}

#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
enum MtpDraftDevice {
    #[default]
    Target,
    Device(CliDevice),
}

impl fmt::Display for MtpDraftDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target => formatter.write_str("target"),
            Self::Device(device) => device.fmt(formatter),
        }
    }
}

impl FromStr for MtpDraftDevice {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value == "target" {
            Ok(Self::Target)
        } else {
            value.parse().map(Self::Device)
        }
    }
}

impl ThinkingMode {
    const fn enabled(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::On => Some(true),
            Self::Off => Some(false),
        }
    }
}

impl From<ExpertCacheEviction> for CacheEvictionPolicy {
    fn from(value: ExpertCacheEviction) -> Self {
        match value {
            ExpertCacheEviction::Lru => Self::LeastRecentlyUsed,
            ExpertCacheEviction::Lfu => Self::LeastFrequentlyUsed,
        }
    }
}

#[derive(Clone)]
enum CliSampler {
    Greedy(DefaultSampler),
    Configured(GenerationSampler),
    MirostatV2(MirostatV2Sampler),
}

impl Sampler for CliSampler {
    fn sample(
        &mut self,
        logits: &Array,
        temp: f32,
        prng_state: Option<&mut RandomState>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Greedy(sampler) => sampler.sample(logits, temp, prng_state, stream),
            Self::Configured(sampler) => sampler.sample(logits, temp, prng_state, stream),
            Self::MirostatV2(sampler) => sampler.sample(logits, temp, prng_state, stream),
        }
    }
}

impl SpeculativeSampler for CliSampler {
    fn supports_exact_optimistic_promotion(&self) -> bool {
        match self {
            Self::Greedy(sampler) => sampler.supports_exact_optimistic_promotion(),
            Self::Configured(sampler) => sampler.supports_exact_optimistic_promotion(),
            Self::MirostatV2(sampler) => sampler.supports_exact_optimistic_promotion(),
        }
    }

    fn process_logits(
        &mut self,
        logits: &Array,
        temperature: f32,
        history: &[u32],
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Greedy(sampler) => sampler.process_logits(logits, temperature, history, stream),
            Self::Configured(sampler) => {
                sampler.process_logits(logits, temperature, history, stream)
            }
            Self::MirostatV2(sampler) => {
                sampler.process_logits(logits, temperature, history, stream)
            }
        }
    }

    fn commit_token(
        &mut self,
        processed_logits: &Array,
        token: u32,
        stream: &Stream,
    ) -> Result<(), Exception> {
        match self {
            Self::Greedy(sampler) => sampler.commit_token(processed_logits, token, stream),
            Self::Configured(sampler) => sampler.commit_token(processed_logits, token, stream),
            Self::MirostatV2(sampler) => sampler.commit_token(processed_logits, token, stream),
        }
    }
}

/// Generate text with a model supported by safemlx-lm.
#[derive(Debug, Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    /// Model directory, GGUF file, or cached Hugging Face model identifier.
    /// Append `:QUANT` to select a cached GGUF quantization.
    #[arg(short, long, value_name = "PATH_OR_ID")]
    model: String,

    /// External assistant directory, GGUF file, or cached Hugging Face identifier.
    /// A bare GGUF repository ID selects its unique draft sidecar.
    #[arg(long, value_name = "PATH_OR_ID")]
    draft_model: Option<String>,

    /// Main model execution device: `cpu` or `gpu:N`.
    #[arg(long, default_value = "gpu:0", value_name = "DEVICE")]
    device: CliDevice,

    /// Process-global MLX allocator-cache limit in bytes; zero disables caching.
    #[arg(long, value_name = "BYTES")]
    mlx_cache_limit_bytes: Option<u64>,

    /// Maximum speculative tokens proposed before each target verification.
    #[arg(long, default_value_t = 3, value_name = "TOKENS")]
    mtp_draft_tokens: usize,

    /// Disable same-request optimistic MTP lookahead for equivalent A/B runs.
    #[arg(long)]
    disable_mtp_lookahead: bool,

    /// Keep MTP lookahead enabled even when retained work does not cover discards.
    #[arg(long)]
    disable_mtp_adaptive_lookahead: bool,

    /// External MTP assistant placement: `target`, `cpu`, or `gpu:N`.
    ///
    /// `target` reuses the main execution stream. An explicit device creates
    /// a distinct draft stream even when it names the main physical device.
    #[arg(long, default_value = "target", value_name = "PLACEMENT")]
    mtp_draft_device: MtpDraftDevice,

    /// Prompt text. Reads the prompt from stdin when omitted and stdin is piped.
    #[arg(value_name = "PROMPT")]
    prompt: Option<String>,

    /// Cached Hugging Face revision (a ref such as `main` or a commit hash).
    #[arg(long, value_name = "REVISION")]
    revision: Option<String>,

    /// Acknowledge target and draft GGUFs from different cached commits.
    ///
    /// Mixed revisions are permitted by default after runtime compatibility
    /// validation; this flag suppresses the provenance warning.
    #[arg(long)]
    allow_mixed_revisions: bool,

    /// Require target and draft GGUFs from one cached repository commit.
    #[arg(long, conflicts_with = "allow_mixed_revisions")]
    require_same_revision: bool,

    /// Maximum number of tokens to generate. Generation stops earlier at EOS.
    #[arg(short = 'n', long, default_value_t = 256, value_name = "TOKENS")]
    max_tokens: usize,

    /// Sampling temperature. Zero selects greedy decoding.
    #[arg(short = 't', long, default_value_t = 0.0, value_name = "FLOAT")]
    temperature: f32,

    /// Keep only the K most likely tokens. Zero disables top-k filtering.
    #[arg(long, default_value_t = 40, value_name = "K")]
    top_k: i32,

    /// Keep the smallest token set with at least this cumulative probability.
    #[arg(long, default_value_t = 0.95, value_name = "FLOAT")]
    top_p: f32,

    /// Remove tokens below this fraction of the most likely token's probability.
    #[arg(long, default_value_t = 0.05, value_name = "FLOAT")]
    min_p: f32,

    /// Use adaptive Mirostat V2 sampling instead of top-k, top-p, and min-p.
    #[arg(long)]
    mirostat_v2: bool,

    /// Target Mirostat V2 surprise in bits.
    #[arg(long, default_value_t = 5.0, value_name = "FLOAT")]
    mirostat_tau: f32,

    /// Mirostat V2 adaptation rate.
    #[arg(long, default_value_t = 0.1, value_name = "FLOAT")]
    mirostat_eta: f32,

    /// Penalty for repeating a token. One disables the penalty.
    #[arg(long, default_value_t = 1.0, value_name = "FLOAT")]
    repeat_penalty: f32,

    /// Number of generated tokens considered for repetition penalties; -1 means all.
    #[arg(long, default_value_t = 64, value_name = "TOKENS")]
    repeat_last_n: i32,

    /// Penalty proportional to the number of times a token was generated.
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    frequency_penalty: f32,

    /// Penalty applied once when a token has already been generated.
    #[arg(long, default_value_t = 0.0, value_name = "FLOAT")]
    presence_penalty: f32,

    /// Random seed used when temperature is non-zero.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Quantize eligible dense weights to this bit width while loading.
    #[arg(long, value_name = "BITS")]
    quantize: Option<i32>,

    /// Encoding used for load-time quantization. MXFP4 requires --quantize 4.
    #[arg(long, value_enum, default_value_t = LoadQuantizationMode::Affine)]
    quantization_mode: LoadQuantizationMode,

    /// Number of adjacent weights sharing quantization parameters.
    #[arg(long, default_value_t = 64, value_name = "WEIGHTS")]
    quantization_group_size: i32,

    /// Keep repeated model layers on the host and use a bounded device window.
    #[arg(long)]
    layerwise_host: bool,

    /// Stream ordinary execution layers through bounded disk, host, and device caches.
    #[arg(long)]
    dense_disk_stream: bool,

    /// Dense-stream host lookahead; use zero with a zero host budget.
    #[arg(long, default_value_t = 2, value_name = "LAYERS")]
    dense_host_lookahead: usize,

    /// Maximum queued dense-stream background host materializations.
    #[arg(long, default_value_t = 2, value_name = "REQUESTS")]
    dense_background_queue: usize,

    /// Cache routed experts independently for any supported MoE model.
    #[arg(long)]
    expert_cache: bool,

    /// Maximum repeated layers resident on the execution device.
    #[arg(long, default_value_t = 1, value_name = "LAYERS")]
    device_layer_window: usize,

    /// Optional logical device parameter budget in bytes.
    #[arg(long, value_name = "BYTES")]
    device_budget_bytes: Option<u64>,

    /// Optional logical host parameter budget in bytes.
    #[arg(long, value_name = "BYTES")]
    host_budget_bytes: Option<u64>,

    /// Optional logical device budget for cached routed experts.
    #[arg(long, value_name = "BYTES")]
    expert_cache_device_budget_bytes: Option<u64>,

    /// Optional logical host budget for cached routed experts; zero uses disk fallback.
    #[arg(long, value_name = "BYTES")]
    expert_cache_host_budget_bytes: Option<u64>,

    /// Hard maximum bytes for one temporary compact expert bank.
    #[arg(long, default_value_t = 1_073_741_824, value_name = "BYTES")]
    expert_cache_scratch_bytes: u64,

    /// Soft compact expert-bank target used to split multi-token prefill routing.
    #[arg(long, default_value_t = 1_073_741_824, value_name = "BYTES")]
    expert_cache_prefill_bank_bytes: u64,

    /// Deterministic expert cache eviction ordering.
    #[arg(long, value_enum, default_value_t = ExpertCacheEviction::Lru)]
    expert_cache_eviction: ExpertCacheEviction,

    /// Measure cold prefill, repeated prefill, and one cached decode separately.
    #[arg(long)]
    expert_cache_benchmark: bool,

    /// Maximum simultaneously mapped safetensors shards or cached GGUF readers.
    #[arg(long, default_value_t = 4, value_name = "SHARDS")]
    mapped_shards: usize,

    /// Pass the prompt directly instead of applying the model's chat template.
    #[arg(long)]
    raw: bool,

    /// JSON file containing an array of OpenAI-shaped function tools.
    #[arg(long, value_name = "PATH")]
    tools: Option<PathBuf>,

    /// Native tool selection policy used with `--tools`.
    #[arg(long, value_enum, default_value_t = CliToolChoice::Auto)]
    tool_choice: CliToolChoice,

    /// Enable parallel native calls, optionally bounded to this many calls.
    #[arg(long, value_name = "CALLS")]
    max_parallel_tool_calls: Option<NonZeroUsize>,

    /// Additional decoded stop sequence for prepared native-tool generation.
    #[arg(long = "stop", value_name = "TEXT")]
    stop_sequences: Vec<String>,

    /// Control thinking/reasoning in chat templates that support `enable_thinking`.
    ///
    /// `auto` preserves the model's default. Explicit `on` or `off` fails when
    /// the model's chat template does not expose a compatible switch.
    #[arg(long, value_enum, default_value_t = ThinkingMode::Auto)]
    thinking: ThinkingMode,

    /// Allow `--thinking on` to fall back to unparsed raw response text.
    #[arg(long)]
    allow_unparsed_reasoning: bool,

    /// Print reasoning content, model resolution, and generation statistics to stderr.
    #[arg(short, long)]
    verbose: bool,

    /// Print load and generation timing statistics to stderr.
    #[arg(long)]
    timing: bool,

    /// Synchronize Gemma 4 component boundaries and report component timings.
    ///
    /// This changes scheduling and should be used for diagnosis, not headline
    /// throughput measurements.
    #[arg(long)]
    profile_components: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StopReason {
    Eos,
    StopSequence,
    GrammarComplete,
    MaxTokens,
    Cancelled,
    GeneratorExhausted,
}

impl StopReason {
    const fn label(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::StopSequence => "stop_sequence",
            Self::GrammarComplete => "grammar_complete",
            Self::MaxTokens => "max_tokens",
            Self::Cancelled => "cancelled",
            Self::GeneratorExhausted => "generator_exhausted",
        }
    }
}

impl From<FinishReason> for StopReason {
    fn from(value: FinishReason) -> Self {
        match value {
            FinishReason::Eos => Self::Eos,
            FinishReason::StopSequence => Self::StopSequence,
            FinishReason::GrammarComplete => Self::GrammarComplete,
            FinishReason::MaxTokens => Self::MaxTokens,
            FinishReason::Cancelled => Self::Cancelled,
        }
    }
}

fn stop_reason(output_ids: &[u32], eos_token_ids: &[u32], max_tokens: usize) -> StopReason {
    if output_ids
        .last()
        .is_some_and(|token| eos_token_ids.contains(token))
    {
        StopReason::Eos
    } else if output_ids.len() >= max_tokens {
        StopReason::MaxTokens
    } else {
        StopReason::GeneratorExhausted
    }
}

fn should_report_stop_reason(stop_reason: StopReason, verbose: bool) -> bool {
    verbose || stop_reason == StopReason::MaxTokens
}

fn write_timing_report(
    stderr: &mut impl Write,
    load_elapsed: std::time::Duration,
    generation_elapsed: std::time::Duration,
    time_to_first_token: Option<std::time::Duration>,
    generated_tokens: usize,
    total_elapsed: std::time::Duration,
) -> Result<()> {
    let token_rate = if generation_elapsed.is_zero() {
        0.0
    } else {
        generated_tokens as f64 / generation_elapsed.as_secs_f64()
    };
    writeln!(stderr, "load_time: {:.3} s", load_elapsed.as_secs_f64())?;
    writeln!(
        stderr,
        "generation_time: {:.3} s",
        generation_elapsed.as_secs_f64()
    )?;
    match time_to_first_token {
        Some(elapsed) => {
            writeln!(
                stderr,
                "time_to_first_token: {:.3} s",
                elapsed.as_secs_f64()
            )?;
            let decode_tokens = generated_tokens.saturating_sub(1);
            let decode_elapsed = generation_elapsed.saturating_sub(elapsed);
            let decode_token_rate = if decode_elapsed.is_zero() {
                0.0
            } else {
                decode_tokens as f64 / decode_elapsed.as_secs_f64()
            };
            writeln!(stderr, "decode_token_rate: {decode_token_rate:.2} tokens/s")?;
        }
        None => writeln!(stderr, "time_to_first_token: n/a")?,
    }
    writeln!(stderr, "token_rate: {token_rate:.2} tokens/s")?;
    writeln!(
        stderr,
        "total_execution_time: {:.3} s",
        total_elapsed.as_secs_f64()
    )?;
    Ok(())
}

fn write_streamed_token(
    decoder: &mut TextDecoder,
    stdout: &mut impl Write,
    streamed_text: &mut String,
    token_id: u32,
) -> Result<()> {
    if let Some(text) = decoder.step(token_id)? {
        stdout.write_all(text.as_bytes())?;
        stdout.flush()?;
        streamed_text.push_str(&text);
    }
    Ok(())
}

fn write_semantic_event(
    event: &SemanticEvent,
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    streamed_text: &mut String,
    reasoning_stream: &mut ReasoningStream,
    verbose: bool,
) -> Result<()> {
    let visible = match event {
        SemanticEvent::TextDelta(text) => Some(text.clone()),
        SemanticEvent::ToolCallStart { index, id, name } => Some(format!(
            "\n{{\"tool_call\":{{\"index\":{index},\"id\":{},\"name\":{},\"arguments\":",
            serde_json::to_string(id)?,
            serde_json::to_string(name)?,
        )),
        SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => Some(json_fragment.clone()),
        SemanticEvent::ToolCallEnd => Some("}}\n".into()),
        SemanticEvent::ReasoningDelta(text) => {
            if verbose {
                reasoning_stream.write_delta(stderr, text)?;
            }
            None
        }
        SemanticEvent::Finished { .. } => {
            if verbose {
                reasoning_stream.close(stderr)?;
            }
            None
        }
    };
    if let Some(visible) = visible {
        if verbose {
            reasoning_stream.close(stderr)?;
            reasoning_stream.announce_visible(stderr)?;
        }
        stdout.write_all(visible.as_bytes())?;
        stdout.flush()?;
        streamed_text.push_str(&visible);
    }
    Ok(())
}

#[derive(Default)]
struct ReasoningStream {
    open: bool,
    ends_with_newline: bool,
    visible_announced: bool,
}

impl ReasoningStream {
    fn write_delta(&mut self, stderr: &mut impl Write, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        if !self.open {
            writeln!(stderr, "--- reasoning content (stderr) ---")?;
            self.open = true;
        }
        stderr.write_all(text.as_bytes())?;
        self.ends_with_newline = text.ends_with('\n');
        stderr.flush()?;
        Ok(())
    }

    fn close(&mut self, stderr: &mut impl Write) -> Result<()> {
        if !self.open {
            return Ok(());
        }
        if !self.ends_with_newline {
            writeln!(stderr)?;
        }
        writeln!(stderr, "--- end reasoning content (stderr) ---")?;
        stderr.flush()?;
        self.open = false;
        self.ends_with_newline = false;
        Ok(())
    }

    fn announce_visible(&mut self, stderr: &mut impl Write) -> Result<()> {
        if !self.visible_announced {
            writeln!(stderr, "--- generated content (stdout) ---")?;
            stderr.flush()?;
            self.visible_announced = true;
        }
        Ok(())
    }
}

fn use_semantic_generation(
    semantic_support: &SemanticSupport,
    tool_support: &NativeToolSupport,
    tools_requested: bool,
) -> Result<bool> {
    if tools_requested {
        return match tool_support {
            NativeToolSupport::Supported => Ok(true),
            NativeToolSupport::Unsupported { reason } => {
                bail!("native tool calling is unavailable: {reason}")
            }
        };
    }
    match semantic_support {
        SemanticSupport::Supported => Ok(true),
        SemanticSupport::Unsupported { .. } => Ok(false),
    }
}

fn execution_contexts(
    device: CliDevice,
    mtp_draft_device: MtpDraftDevice,
) -> (ExecutionContext, Option<ExecutionContext>) {
    let execution = ExecutionContext::new(device.mlx());
    let draft = match mtp_draft_device {
        MtpDraftDevice::Target => None,
        MtpDraftDevice::Device(device) => Some(ExecutionContext::new(device.mlx())),
    };
    (execution, draft)
}

fn requested_load_quantization(args: &Cli) -> Result<Option<WeightQuantization>> {
    match (args.quantize, args.quantization_mode) {
        (Some(bits), LoadQuantizationMode::Affine) => Ok(Some(
            AffineQuantization::new(args.quantization_group_size, bits)?.into(),
        )),
        (Some(4), LoadQuantizationMode::Mxfp4) => Ok(Some(WeightQuantization::MxFp4)),
        (Some(_), LoadQuantizationMode::Mxfp4) => {
            bail!("--quantization-mode mxfp4 requires --quantize 4")
        }
        (None, LoadQuantizationMode::Mxfp4) => {
            bail!("--quantization-mode requires --quantize")
        }
        (None, LoadQuantizationMode::Affine) => Ok(None),
    }
}

fn main() -> Result<()> {
    let total_started = Instant::now();
    let args = Cli::parse();
    validate_args(&args)?;
    if let Some(bytes) = args.mlx_cache_limit_bytes {
        let bytes = usize::try_from(bytes).context("--mlx-cache-limit-bytes exceeds usize")?;
        safemlx::memory::set_cache_limit(bytes)
            .context("failed to set the MLX allocator-cache limit")?;
    }
    let prompt = read_prompt(args.prompt.as_deref())?;
    let (resolved_model, resolved_draft) = resolve_model_pair(
        &args.model,
        args.draft_model.as_deref(),
        args.revision.as_deref(),
    )?;
    if let Some(draft) = &resolved_draft {
        let mixed_revisions =
            validate_artifact_pair(&resolved_model, draft, args.require_same_revision)?;
        if mixed_revisions && !args.allow_mixed_revisions {
            eprintln!(
                "warning: target and draft artifacts come from different cached commits of {:?}; repository revision is provenance, not an MTP compatibility key (use --require-same-revision to enforce provenance equality or --allow-mixed-revisions to suppress this warning)",
                resolved_model.repository.as_deref().unwrap_or("unknown"),
            );
        }
    }
    let model_path = resolved_model.path;
    let draft_model_path = resolved_draft.map(|artifact| artifact.path);

    if args.verbose {
        eprintln!("--- safemlx diagnostics (stderr) ---");
        eprintln!("model: {}", model_path.display());
        eprintln!("device: {}", args.device);
        if let Some(bytes) = args.mlx_cache_limit_bytes {
            eprintln!("mlx_cache_limit: {}", format_bytes(bytes as usize));
        }
        if let Some(path) = &draft_model_path {
            eprintln!("draft_model: {}", path.display());
            eprintln!("mtp_draft_device: {}", args.mtp_draft_device);
        }
    }

    let (execution, draft_execution) = execution_contexts(args.device, args.mtp_draft_device);
    let weights = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = execution.stream();
    let draft_stream = draft_execution
        .as_ref()
        .map_or(stream, ExecutionContext::stream);
    if args.verbose {
        // Capture the complete model-load and generation high-water mark.
        safemlx::memory::reset_peak_memory()?;
    }
    let load_started = Instant::now();
    let mut load_options = requested_load_quantization(&args)?.map_or_else(
        ModelLoadOptions::default,
        ModelLoadOptions::with_quantization,
    );
    let dense_options = || -> Result<DenseDiskStreamLoadOptions> {
        let defaults = DenseDiskStreamLoadOptions::default();
        let mut options = DenseDiskStreamLoadOptions::new(
            args.device_budget_bytes
                .unwrap_or(defaults.device_budget_bytes),
            args.host_budget_bytes.unwrap_or(defaults.host_budget_bytes),
            args.dense_host_lookahead,
            args.dense_background_queue,
        )?;
        options.max_mapped_shards = args.mapped_shards;
        options.sample_mlx_memory = args.verbose;
        options.sample_process_memory = args.verbose;
        Ok(options)
    };
    if args.expert_cache {
        let non_expert = LayerwiseLoadOptions {
            offload: OffloadConfig::new(
                args.device_budget_bytes,
                args.host_budget_bytes,
                args.device_layer_window,
            )?,
            max_mapped_shards: args.mapped_shards,
            sample_mlx_memory: args.verbose,
            sample_process_memory: args.verbose,
            ..LayerwiseLoadOptions::default()
        };
        let experts = OffloadConfig::new(
            args.expert_cache_device_budget_bytes,
            args.expert_cache_host_budget_bytes,
            1,
        )?
        .with_eviction_policy(args.expert_cache_eviction.into());
        let expert_options = ExpertCacheLoadOptions::new(
            experts,
            args.expert_cache_scratch_bytes,
            args.expert_cache_prefill_bank_bytes,
        )?;
        let non_experts = if args.dense_disk_stream {
            NonExpertWeightResidency::DenseDiskStream(dense_options()?)
        } else if args.layerwise_host {
            NonExpertWeightResidency::LayerwiseHost(non_expert)
        } else {
            NonExpertWeightResidency::FullyResident
        };
        load_options = load_options.with_weight_residency(WeightResidency::with_expert_cache(
            non_experts,
            expert_options,
        ));
    } else if args.dense_disk_stream {
        load_options = load_options
            .with_weight_residency(WeightResidency::dense_disk_stream(dense_options()?));
    } else if args.layerwise_host {
        let offload = OffloadConfig::new(
            args.device_budget_bytes,
            args.host_budget_bytes,
            args.device_layer_window,
        )?;
        load_options = load_options.with_weight_residency(WeightResidency::layerwise_host(
            LayerwiseLoadOptions {
                offload,
                max_mapped_shards: args.mapped_shards,
                sample_mlx_memory: args.verbose,
                sample_process_memory: args.verbose,
                ..LayerwiseLoadOptions::default()
            },
        ));
    }
    let mut model =
        LoadedModel::load_with_options(&model_path, load_options, stream, weights.stream())
            .with_context(|| format!("failed to load model from {}", model_path.display()))?;
    if draft_model_path.is_some()
        && !matches!(
            model.mtp_capability(),
            safemlx_lm::runtime::generation::speculative::MtpCapability::Ready {
                checkpoint:
                    safemlx_lm::runtime::generation::speculative::MtpCheckpointKind::Separate
            }
        )
    {
        bail!(
            "--draft-model cannot be used with target capability {:?}",
            model.mtp_capability()
        );
    }
    let mut drafter = draft_model_path
        .as_ref()
        .map(|path| {
            let options = requested_load_quantization(&args)?.map_or_else(
                ModelLoadOptions::default,
                ModelLoadOptions::with_quantization,
            );
            LoadedDrafter::load_with_options(path, options, draft_stream, weights.stream())
                .with_context(|| format!("failed to load draft model from {}", path.display()))
        })
        .transpose()?;
    if let Some(drafter) = &drafter {
        model
            .validate_drafter_compatibility(drafter)
            .context("target and draft models are not MTP-compatible")?;
    }
    stream.synchronize()?;
    if draft_stream != stream {
        draft_stream.synchronize()?;
    }
    let load_elapsed = load_started.elapsed();

    let tools_requested = args.tools.is_some();
    let tools = args
        .tools
        .as_deref()
        .map(read_tools)
        .transpose()?
        .unwrap_or_default();
    let (prepared_chat, rendered_prompt, add_special_tokens) = if args.raw {
        (None, prompt, true)
    } else {
        let request = ChatTemplateRequest {
            messages: vec![serde_json::json!({
                "role": "user",
                "content": prompt.clone(),
            })],
            tools,
            tool_choice: if tools_requested {
                args.tool_choice.into()
            } else {
                ToolChoice::None
            },
            parallel_tool_calls: match args.max_parallel_tool_calls {
                Some(max_calls) => ParallelToolCallPolicy::Enabled {
                    max_calls: Some(max_calls),
                },
                None => ParallelToolCallPolicy::Disabled,
            },
            enable_thinking: args.thinking.enabled(),
            allow_unparsed_reasoning: args.allow_unparsed_reasoning,
            add_generation_prompt: true,
            ..ChatTemplateRequest::default()
        };
        match model.prepare_chat(request) {
            Ok(prepared) => {
                let semantic = use_semantic_generation(
                    prepared.semantic_support(),
                    prepared.native_tool_support(),
                    tools_requested,
                )?;
                let rendered_prompt = prepared.rendered_prompt().to_owned();
                if semantic {
                    if args.verbose {
                        let label = if tools_requested {
                            "native_tool_profile"
                        } else {
                            "semantic_profile"
                        };
                        eprintln!(
                            "{label}: {}",
                            prepared.format_profile_identity().unwrap_or("unregistered")
                        );
                    }
                    (Some(prepared), rendered_prompt, false)
                } else {
                    if args.verbose {
                        eprintln!(
                            "semantic_profile: unavailable ({}); using templated text fallback",
                            prepared
                                .native_tool_support()
                                .unsupported_reason()
                                .unwrap_or("unknown reason")
                        );
                    }
                    (None, rendered_prompt, false)
                }
            }
            Err(LmError::MissingChatTemplate) if !tools_requested => (None, prompt, true),
            Err(error) => return Err(error.into()),
        }
    };

    let tokens = model.encode_to_array(&rendered_prompt, add_special_tokens, stream)?;
    if tokens.shape()[1] == 0 {
        bail!("the prompt produced no input tokens");
    }

    if args.expert_cache_benchmark {
        run_expert_cache_benchmark(&mut model, &tokens, stream)?;
    }

    let eos_token_ids = model.eos_token_ids().to_vec();
    let mut cache = model.new_cache();
    let configured_sampler = GenerationSampler::new()
        .top_k(args.top_k)
        .top_p(args.top_p)
        .min_p(args.min_p)
        .penalties(
            args.repeat_penalty,
            args.repeat_last_n,
            args.frequency_penalty,
            args.presence_penalty,
        );
    // Probability filters cannot change a greedy argmax. Avoid their full-vocabulary
    // sorting and softmax work, as well as GenerationSampler's token-history readback,
    // when no repetition/frequency/presence penalty is active.
    let penalties_active =
        args.repeat_penalty != 1.0 || args.frequency_penalty != 0.0 || args.presence_penalty != 0.0;
    let mut sampler = if args.mirostat_v2 {
        CliSampler::MirostatV2(
            MirostatV2Sampler::new(args.mirostat_tau, args.mirostat_eta)?.penalties(
                args.repeat_penalty,
                args.repeat_last_n,
                args.frequency_penalty,
                args.presence_penalty,
            ),
        )
    } else if args.temperature == 0.0 && !penalties_active {
        CliSampler::Greedy(DefaultSampler)
    } else {
        CliSampler::Configured(configured_sampler)
    };
    let prng_key = (args.temperature != 0.0)
        .then(|| safemlx::random::key(args.seed))
        .transpose()?;
    let mut output_ids = Vec::with_capacity(args.max_tokens);
    let profile_gemma4 = args.profile_components && model.model_type() == "gemma4";
    if profile_gemma4 {
        safemlx_lm::architectures::gemma4::model::set_perf_profiling(true);
        safemlx_lm::architectures::gemma4::model::reset_perf_stats();
    }
    let generation_started = Instant::now();
    let mut time_to_first_token = None;
    let mut mtp_stats: Option<MtpStats> = None;
    let mut decoder = model.text_decoder(true);
    let mut streamed_text = String::new();
    let mut reasoning_stream = ReasoningStream::default();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    if args.verbose && prepared_chat.is_none() {
        eprintln!("--- generated content (stdout) ---");
    }
    let stderr = io::stderr();
    let mut stderr = stderr.lock();

    let embedded_mtp = args.mtp_draft_tokens > 0
        && matches!(
            model.mtp_capability(),
            safemlx_lm::runtime::generation::speculative::MtpCapability::Ready {
                checkpoint:
                    safemlx_lm::runtime::generation::speculative::MtpCheckpointKind::Embedded
            }
        );
    let _component_timing_guard = args.verbose.then(MtpComponentTimingGuard::enable);
    let scheduler_options = MtpSchedulerOptions {
        adaptive_lookahead: !args.disable_mtp_adaptive_lookahead,
        ..MtpSchedulerOptions::default()
    }
    .with_lookahead(!args.disable_mtp_lookahead);
    let mut prepared_finish_reason = None;
    if let Some(prepared) = &prepared_chat {
        let settings = PreparedChatGenerationSettings {
            temperature: args.temperature,
            max_tokens: NonZeroUsize::new(args.max_tokens).expect("validated non-zero max tokens"),
            prng_key,
        };
        let mut semantic_error = None;
        if let Some(drafter) = drafter.as_mut() {
            let cancellation = GenerationCancellationToken::new();
            let cancel_on_error = cancellation.clone();
            let output = model.generate_prepared_chat_mtp(PreparedChatMtpGenerationRequest {
                input: PreparedChatInput::rendered_prompt(prepared),
                drafter,
                cache: &mut cache,
                sampling_policy: sampler,
                settings,
                options: PreparedChatMtpGenerationOptions {
                    max_draft_tokens: NonZeroUsize::new(args.mtp_draft_tokens)
                        .expect("external MTP validates non-zero draft tokens"),
                    scheduler: scheduler_options,
                },
                caller_stop_sequences: &args.stop_sequences,
                streams: MtpExecutionStreams::new(stream, draft_stream)?,
                cancellation,
                on_event: |event| {
                    if time_to_first_token.is_none()
                        && !matches!(event, SemanticEvent::Finished { .. })
                    {
                        time_to_first_token = Some(generation_started.elapsed());
                    }
                    if semantic_error.is_none() {
                        semantic_error = write_semantic_event(
                            &event,
                            &mut stdout,
                            &mut stderr,
                            &mut streamed_text,
                            &mut reasoning_stream,
                            args.verbose,
                        )
                        .err();
                        if semantic_error.is_some() {
                            cancel_on_error.cancel();
                        }
                    }
                },
            })?;
            output_ids = output.token_ids;
            mtp_stats = Some(output.stats);
            prepared_finish_reason = Some(output.finish_reason);
        } else if embedded_mtp {
            let cancellation = GenerationCancellationToken::new();
            let cancel_on_error = cancellation.clone();
            let output = model.generate_prepared_chat_embedded_mtp(
                PreparedChatEmbeddedMtpGenerationRequest {
                    input: PreparedChatInput::rendered_prompt(prepared),
                    cache: &mut cache,
                    sampling_policy: sampler,
                    settings,
                    options: PreparedChatMtpGenerationOptions {
                        max_draft_tokens: NonZeroUsize::new(args.mtp_draft_tokens)
                            .expect("embedded MTP validates non-zero draft tokens"),
                        scheduler: scheduler_options,
                    },
                    caller_stop_sequences: &args.stop_sequences,
                    stream,
                    cancellation,
                    on_event: |event| {
                        if time_to_first_token.is_none()
                            && !matches!(event, SemanticEvent::Finished { .. })
                        {
                            time_to_first_token = Some(generation_started.elapsed());
                        }
                        if semantic_error.is_none() {
                            semantic_error = write_semantic_event(
                                &event,
                                &mut stdout,
                                &mut stderr,
                                &mut streamed_text,
                                &mut reasoning_stream,
                                args.verbose,
                            )
                            .err();
                            if semantic_error.is_some() {
                                cancel_on_error.cancel();
                            }
                        }
                    },
                },
            )?;
            output_ids = output.token_ids;
            mtp_stats = Some(output.stats);
            prepared_finish_reason = Some(output.finish_reason);
        } else {
            let cancellation = GenerationCancellationToken::new();
            let cancel_on_error = cancellation.clone();
            let output = model.generate_prepared_chat(PreparedChatGenerationRequest {
                input: PreparedChatInput::rendered_prompt(prepared),
                cache: &mut cache,
                sampling_policy: sampler,
                settings,
                caller_stop_sequences: &args.stop_sequences,
                stream,
                cancellation,
                on_event: |event| {
                    if time_to_first_token.is_none()
                        && !matches!(event, SemanticEvent::Finished { .. })
                    {
                        time_to_first_token = Some(generation_started.elapsed());
                    }
                    if semantic_error.is_none() {
                        semantic_error = write_semantic_event(
                            &event,
                            &mut stdout,
                            &mut stderr,
                            &mut streamed_text,
                            &mut reasoning_stream,
                            args.verbose,
                        )
                        .err();
                        if semantic_error.is_some() {
                            cancel_on_error.cancel();
                        }
                    }
                },
            })?;
            output_ids = output.token_ids;
            prepared_finish_reason = Some(output.finish_reason);
        }
        if let Some(error) = semantic_error {
            return Err(error);
        }
    } else if let Some(drafter) = drafter.as_mut() {
        let parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: args.max_tokens,
            max_draft_tokens: args.mtp_draft_tokens,
            temperature: args.temperature,
            eos_token_ids: eos_token_ids.clone(),
        };
        let mtp_streams = MtpExecutionStreams::new(stream, draft_stream)?;
        let (tokens, stats) = model
            .generate_mtp_input_with_sampler_callback_and_streams_and_options(
                drafter,
                &mut cache,
                input,
                &config,
                prng_key,
                &mut sampler,
                mtp_streams,
                scheduler_options,
                |token_id| {
                    if time_to_first_token.is_none() {
                        time_to_first_token = Some(generation_started.elapsed());
                    }
                    if eos_token_ids.contains(&token_id) {
                        Ok(())
                    } else {
                        write_streamed_token(
                            &mut decoder,
                            &mut stdout,
                            &mut streamed_text,
                            token_id,
                        )
                        .map_err(|error| Exception::custom(error.to_string()))
                    }
                },
            )?;
        output_ids = tokens;
        mtp_stats = Some(stats);
    } else if embedded_mtp {
        let parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&parts);
        let config = MtpConfig {
            max_tokens: args.max_tokens,
            max_draft_tokens: args.mtp_draft_tokens,
            temperature: args.temperature,
            eos_token_ids: eos_token_ids.clone(),
        };
        let (tokens, stats) = model.generate_embedded_mtp_input_with_sampler_callback(
            &mut cache,
            input,
            &config,
            prng_key,
            &mut sampler,
            stream,
            |token_id| {
                if time_to_first_token.is_none() {
                    time_to_first_token = Some(generation_started.elapsed());
                }
                if eos_token_ids.contains(&token_id) {
                    Ok(())
                } else {
                    write_streamed_token(&mut decoder, &mut stdout, &mut streamed_text, token_id)
                        .map_err(|error| Exception::custom(error.to_string()))
                }
            },
        )?;
        output_ids = tokens;
        mtp_stats = Some(stats);
    } else {
        let parts = [InputPart::text_token_ids(&tokens)];
        let input = ModelInput::new(&parts);
        let mut generator = model.generate_input_with_cache_sampler(
            &mut cache,
            args.temperature,
            input,
            prng_key,
            stream,
            sampler,
        );

        let mut current = generator.next().transpose()?;
        for index in 0..args.max_tokens {
            let Some(token) = current.take() else {
                break;
            };
            // Start the following decode before reading the current token back to
            // the CPU. This mirrors mlx-lm's one-token async evaluation pipeline.
            let next = if index + 1 < args.max_tokens {
                let next = generator.next();
                if let Some(Ok(next_token)) = next.as_ref() {
                    async_eval([next_token])?;
                }
                next
            } else {
                None
            };
            let token_id = token.item::<u32>(stream);
            if time_to_first_token.is_none() {
                time_to_first_token = Some(generation_started.elapsed());
            }
            output_ids.push(token_id);
            if eos_token_ids.contains(&token_id) {
                break;
            }
            write_streamed_token(&mut decoder, &mut stdout, &mut streamed_text, token_id)?;
            current = next.transpose()?;
        }
    }
    if args.verbose {
        reasoning_stream.close(&mut stderr)?;
    }
    drop(stderr);

    let generation_elapsed = generation_started.elapsed();
    let stop_reason = prepared_finish_reason
        .map(StopReason::from)
        .unwrap_or_else(|| stop_reason(&output_ids, &eos_token_ids, args.max_tokens));
    if prepared_chat.is_none() && stop_reason == StopReason::Eos {
        output_ids.pop();
    }

    if prepared_chat.is_none() {
        let output = model.decode(&output_ids, true)?;
        let remaining = output.strip_prefix(&streamed_text).with_context(|| {
            "incremental tokenizer output did not match the final decoded response"
        })?;
        stdout.write_all(remaining.as_bytes())?;
        if !output.ends_with('\n') {
            writeln!(stdout)?;
        }
    } else if !streamed_text.ends_with('\n') {
        writeln!(stdout)?;
    }
    stdout.flush()?;

    if args.verbose {
        eprintln!("--- safemlx diagnostics (stderr) ---");
    }
    if tools_requested || should_report_stop_reason(stop_reason, args.verbose) {
        eprintln!("stop_reason: {}", stop_reason.label());
    }

    if args.verbose {
        stream.synchronize()?;
        let peak_memory = safemlx::memory::peak_memory()?;
        let active_memory = safemlx::memory::active_memory()?;
        let cache_memory = safemlx::memory::cache_memory()?;
        let total_elapsed = total_started.elapsed();
        eprintln!(
            "model_type: {}, prompt_tokens: {}, generated_tokens: {}",
            model.model_type(),
            tokens.shape()[1],
            output_ids.len(),
        );
        write_timing_report(
            &mut io::stderr().lock(),
            load_elapsed,
            generation_elapsed,
            time_to_first_token,
            output_ids.len(),
            total_elapsed,
        )?;
        if let Some(stats) = &mtp_stats {
            eprintln!("mtp_stream_topology: {}", stats.stream_topology);
            eprintln!(
                "mtp_rounds: {}, mtp_draft_tokens: {}, mtp_accepted_tokens: {}, mtp_accept_rate: {:.3}, mtp_optimistic_blocks: {}, mtp_optimistic_bonus_tokens: {}, mtp_optimistic_bonus_matches: {}, mtp_optimistic_bonus_mismatches: {}, mtp_consumed_optimistic_tokens: {}, mtp_reused_optimistic_blocks: {}, mtp_reused_optimistic_tokens: {}, mtp_discarded_optimistic_blocks: {}, mtp_discarded_optimistic_tokens: {}, mtp_adaptive_lookahead_disabled: {}, mtp_cross_request_draft_opportunities: {}",
                stats.rounds,
                stats.draft_tokens,
                stats.accepted_tokens,
                stats.accept_rate(),
                stats.optimistic_draft_blocks,
                stats.optimistic_target_bonus_tokens,
                stats.optimistic_bonus_matches,
                stats.optimistic_bonus_mismatches,
                stats.consumed_optimistic_tokens,
                stats.reused_optimistic_blocks,
                stats.reused_optimistic_tokens,
                stats.discarded_optimistic_blocks,
                stats.discarded_optimistic_tokens,
                stats.adaptive_lookahead_disabled,
                stats.cross_request_draft_opportunities,
            );
            eprintln!(
                "mtp_optimistic_draft_time: {:.3} s, mtp_verification_in_flight_time: {:.3} s",
                stats.optimistic_draft_time.as_secs_f64(),
                stats.verification_in_flight_time.as_secs_f64(),
            );
            if stats.component_timings_collected {
                eprintln!(
                    "mtp_draft_context_time: {:.3} s, mtp_draft_assistant_time: {:.3} s, mtp_draft_head_time: {:.3} s, mtp_target_verification_time: {:.3} s",
                    stats.draft_context_time.as_secs_f64(),
                    stats.draft_assistant_time.as_secs_f64(),
                    stats.draft_head_time.as_secs_f64(),
                    stats.target_verification_time.as_secs_f64(),
                );
            }
            eprintln!("mtp_accept_lens: {:?}", stats.accept_lens);
        }
        eprintln!("mlx_peak_memory: {}", format_bytes(peak_memory));
        eprintln!("mlx_active_memory: {}", format_bytes(active_memory));
        eprintln!("mlx_cache_memory: {}", format_bytes(cache_memory));
        if let Some(stats) = model.native_quantization_stats() {
            eprintln!(
                "native_quantization: {} tensors / {}, fallback: {} tensors / {} checkpoint bytes",
                stats.native_tensor_count,
                format_bytes(stats.native_bytes as usize),
                stats.fallback_tensor_count,
                format_bytes(stats.fallback_checkpoint_bytes as usize),
            );
            eprintln!(
                "native_quantization_formats: Q4_K={} tensors/{}, Q5_1={} tensors/{}, Q8_0={} tensors/{}",
                stats.q4k_tensor_count,
                format_bytes(stats.q4k_bytes as usize),
                stats.q5_1_tensor_count,
                format_bytes(stats.q5_1_bytes as usize),
                stats.q8_0_tensor_count,
                format_bytes(stats.q8_0_bytes as usize),
            );
        }
        if profile_gemma4 {
            if let Some(stats) = safemlx_lm::architectures::gemma4::model::perf_stats() {
                eprintln!(
                    "gemma4_components_s: embed={:.6}, per_layer_inputs={:.6}, attention={:.6}, mlp={:.6}, per_layer_residual={:.6}, final_norm={:.6}, lm_head={:.6}, total={:.6}",
                    stats.embed_s,
                    stats.per_layer_inputs_s,
                    stats.attention_s,
                    stats.mlp_s,
                    stats.per_layer_residual_s,
                    stats.final_norm_s,
                    stats.lm_head_s,
                    stats.component_total_s(),
                );
            }
            safemlx_lm::architectures::gemma4::model::set_perf_profiling(false);
        }
        if let Some(report) = model.residency_report()? {
            let offload = report.offload();
            eprintln!(
                "residency_current_host_device: {} / {} bytes",
                offload.resident_bytes().get(MemoryTier::Host),
                offload.resident_bytes().get(MemoryTier::Device)
            );
            eprintln!(
                "residency_peak_host_device: {} / {} bytes",
                offload.peak_resident_bytes().get(MemoryTier::Host),
                offload.peak_resident_bytes().get(MemoryTier::Device)
            );
            for direction in TransferDirection::ALL {
                let transfer = offload.transfer(direction);
                if transfer.count() > 0 {
                    eprintln!(
                        "residency_{direction:?}: {} transfers, {} bytes",
                        transfer.count(),
                        transfer.bytes()
                    );
                }
            }
            eprintln!(
                "weight_store: {}",
                format_weight_store_diagnostics(report.weight_store())
            );
            if let Some(materialization) = report.materialization() {
                eprintln!(
                    "ordinary_weight_quantization: {} weights, {} tiles, {} source bytes -> {} packed bytes, {} peak working-set bytes",
                    materialization.transformed_weights,
                    materialization.source_tiles,
                    materialization.source_bytes_read,
                    materialization.output_bytes,
                    materialization.peak_planned_working_set_bytes
                );
            }
        }
        if let Some(report) = model.expert_cache_report()? {
            eprintln!(
                "expert_cache_owned: {} experts, {} bytes",
                report.owned_experts, report.owned_bytes
            );
            eprintln!(
                "expert_cache_current_host_device: {} / {} experts, {} / {} bytes",
                report.host_resident_experts,
                report.device_resident_experts,
                report.host_resident_bytes,
                report.device_resident_bytes
            );
            eprintln!("expert_cache_prefill: {:?}", report.prefill);
            eprintln!("expert_cache_decode: {:?}", report.decode);
            if let Some(materialization) = &report.materialization {
                eprintln!(
                    "expert_cache_quantization: {} weights, {} tiles, {} source bytes -> {} packed bytes, {} peak working-set bytes",
                    materialization.transformed_weights,
                    materialization.source_tiles,
                    materialization.source_bytes_read,
                    materialization.output_bytes,
                    materialization.peak_planned_working_set_bytes
                );
            }
        }
        if eos_token_ids.is_empty() {
            eprintln!("warning: the model config contains no EOS token id");
        } else {
            match stop_reason {
                StopReason::MaxTokens => {
                    eprintln!("warning: generation reached --max-tokens before EOS");
                }
                StopReason::GeneratorExhausted => {
                    eprintln!("warning: the token generator ended before EOS");
                }
                StopReason::Eos
                | StopReason::StopSequence
                | StopReason::GrammarComplete
                | StopReason::Cancelled => {}
            }
        }
    } else if args.timing {
        stream.synchronize()?;
        write_timing_report(
            &mut io::stderr().lock(),
            load_elapsed,
            generation_elapsed,
            time_to_first_token,
            output_ids.len(),
            total_started.elapsed(),
        )?;
    }

    Ok(())
}

fn format_bytes(bytes: usize) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;

    let bytes_float = bytes as f64;
    let (value, unit) = if bytes_float >= GIB {
        (bytes_float / GIB, "GiB")
    } else if bytes_float >= MIB {
        (bytes_float / MIB, "MiB")
    } else if bytes_float >= KIB {
        (bytes_float / KIB, "KiB")
    } else {
        (bytes_float, "B")
    };
    format!("{value:.2} {unit} ({bytes} bytes)")
}

fn format_weight_store_diagnostics(
    diagnostics: &safemlx_lm::runtime::checkpoint::store::WeightStoreDiagnostics,
) -> String {
    format!(
        "backend={:?}, mapping_hits={}, mapping_misses={}, evictions={}, currently_mapped_shards={}, touched_shards={}, physical_reads={}, physical_read_bytes={}, coalesced_group_hits={}",
        diagnostics.backend,
        diagnostics.mapping_hits,
        diagnostics.mapping_misses,
        diagnostics.evictions,
        diagnostics.currently_mapped_shards,
        diagnostics.touched_shard_paths.len(),
        diagnostics.physical_reads,
        diagnostics.physical_read_bytes,
        diagnostics.coalesced_group_hits,
    )
}

#[derive(Clone, Copy)]
struct ExpertBenchmarkSnapshot {
    prefill: ExpertPassStatistics,
    decode: ExpertPassStatistics,
    host_experts: usize,
    device_experts: usize,
    host_bytes: u64,
    device_bytes: u64,
}

fn expert_benchmark_snapshot(model: &LoadedModel) -> Result<ExpertBenchmarkSnapshot> {
    let report = model
        .expert_cache_report()?
        .context("sparse expert cache benchmark requires an expert-cache model")?;
    Ok(ExpertBenchmarkSnapshot {
        prefill: report.prefill,
        decode: report.decode,
        host_experts: report.host_resident_experts,
        device_experts: report.device_resident_experts,
        host_bytes: report.host_resident_bytes,
        device_bytes: report.device_resident_bytes,
    })
}

fn tier_delta(after: ExpertTierStatistics, before: ExpertTierStatistics) -> ExpertTierStatistics {
    ExpertTierStatistics {
        requests: after.requests.saturating_sub(before.requests),
        hits: after.hits.saturating_sub(before.hits),
        misses: after.misses.saturating_sub(before.misses),
        evictions: after.evictions.saturating_sub(before.evictions),
        eviction_bytes: after.eviction_bytes.saturating_sub(before.eviction_bytes),
    }
}

fn print_expert_benchmark_result(
    label: &str,
    elapsed: std::time::Duration,
    before: ExpertPassStatistics,
    after: ExpertPassStatistics,
    occupancy: ExpertBenchmarkSnapshot,
) {
    let host = tier_delta(after.host, before.host);
    let device = tier_delta(after.device, before.device);
    eprintln!(
        "expert_cache_benchmark_{label}: latency={:.3}s routes={} distinct={} coalesced={} compact_banks={} compact_bytes={} host_hits={} host_misses={} host_evictions={} device_hits={} device_misses={} device_evictions={} host_resident={}({} bytes) device_resident={}({} bytes)",
        elapsed.as_secs_f64(),
        after.requested_routes.saturating_sub(before.requested_routes),
        after.distinct_experts.saturating_sub(before.distinct_experts),
        after
            .coalesced_duplicates
            .saturating_sub(before.coalesced_duplicates),
        after.compact_banks.saturating_sub(before.compact_banks),
        after
            .compact_bank_bytes
            .saturating_sub(before.compact_bank_bytes),
        host.hits,
        host.misses,
        host.evictions,
        device.hits,
        device.misses,
        device.evictions,
        occupancy.host_experts,
        occupancy.host_bytes,
        occupancy.device_experts,
        occupancy.device_bytes,
    );
}

fn run_expert_cache_benchmark(
    model: &mut LoadedModel,
    tokens: &Array,
    stream: &Stream,
) -> Result<()> {
    let parts = [InputPart::text_token_ids(tokens)];
    let input = ModelInput::new(&parts);

    let before_cold = expert_benchmark_snapshot(model)?;
    let mut cold_cache = model.new_cache();
    let started = Instant::now();
    let logits = model.prefill_input_with_cache(input, &mut cold_cache, stream)?;
    eval([&logits])?;
    stream.synchronize()?;
    let cold_elapsed = started.elapsed();
    let after_cold = expert_benchmark_snapshot(model)?;
    print_expert_benchmark_result(
        "cold_prefill",
        cold_elapsed,
        before_cold.prefill,
        after_cold.prefill,
        after_cold,
    );

    let before_repeated = after_cold;
    let mut repeated_cache = model.new_cache();
    let started = Instant::now();
    let logits = model.prefill_input_with_cache(input, &mut repeated_cache, stream)?;
    eval([&logits])?;
    stream.synchronize()?;
    let repeated_elapsed = started.elapsed();
    let after_repeated = expert_benchmark_snapshot(model)?;
    print_expert_benchmark_result(
        "repeated_prefill",
        repeated_elapsed,
        before_repeated.prefill,
        after_repeated.prefill,
        after_repeated,
    );

    let last = tokens.try_index_device((.., tokens.dim(1) - 1..), stream)?;
    let decode_parts = [InputPart::text_token_ids(&last)];
    let decode_input = ModelInput::new(&decode_parts);
    let before_decode = after_repeated;
    let started = Instant::now();
    let logits = model.prefill_input_with_cache(decode_input, &mut repeated_cache, stream)?;
    eval([&logits])?;
    stream.synchronize()?;
    let decode_elapsed = started.elapsed();
    let after_decode = expert_benchmark_snapshot(model)?;
    print_expert_benchmark_result(
        "cached_decode",
        decode_elapsed,
        before_decode.decode,
        after_decode.decode,
        after_decode,
    );
    Ok(())
}

fn validate_args(args: &Cli) -> Result<()> {
    if args.max_tokens == 0 {
        bail!("--max-tokens must be greater than zero");
    }
    if args.draft_model.is_some() && args.mtp_draft_tokens == 0 {
        bail!("--mtp-draft-tokens must be greater than zero when --draft-model is used");
    }
    if args.mtp_draft_device != MtpDraftDevice::Target && args.draft_model.is_none() {
        bail!(
            "--mtp-draft-device {} requires an external --draft-model",
            args.mtp_draft_device,
        );
    }
    if !args.temperature.is_finite() || args.temperature < 0.0 {
        bail!("--temperature must be a finite, non-negative number");
    }
    if args.top_k < 0 {
        bail!("--top-k must be non-negative");
    }
    if !args.top_p.is_finite() || !(0.0..=1.0).contains(&args.top_p) {
        bail!("--top-p must be between zero and one");
    }
    if !args.min_p.is_finite() || !(0.0..=1.0).contains(&args.min_p) {
        bail!("--min-p must be between zero and one");
    }
    if args.mirostat_v2 && args.temperature == 0.0 {
        bail!("--mirostat-v2 requires --temperature greater than zero");
    }
    if !args.mirostat_tau.is_finite() || args.mirostat_tau <= 0.0 {
        bail!("--mirostat-tau must be a finite number greater than zero");
    }
    if !args.mirostat_eta.is_finite() || args.mirostat_eta <= 0.0 {
        bail!("--mirostat-eta must be a finite number greater than zero");
    }
    if !args.repeat_penalty.is_finite() || args.repeat_penalty <= 0.0 {
        bail!("--repeat-penalty must be a finite number greater than zero");
    }
    if args.repeat_last_n < -1 {
        bail!("--repeat-last-n must be -1 or greater");
    }
    if !args.frequency_penalty.is_finite() || !args.presence_penalty.is_finite() {
        bail!("frequency and presence penalties must be finite numbers");
    }
    requested_load_quantization(args)?;
    if args.layerwise_host && args.quantize.is_some() {
        bail!("--quantize is not supported with --layerwise-host; use matching checkpoint-native quantization");
    }
    if args.dense_disk_stream && args.quantize.is_some() {
        bail!("--quantize is not supported with --dense-disk-stream; use matching checkpoint-native weights");
    }
    if args.dense_disk_stream && args.layerwise_host {
        bail!("--dense-disk-stream conflicts with --layerwise-host");
    }
    if args.expert_cache_benchmark && !args.expert_cache {
        bail!("--expert-cache-benchmark requires --expert-cache");
    }
    if args.expert_cache_scratch_bytes == 0 {
        bail!("--expert-cache-scratch-bytes must be greater than zero");
    }
    if args.expert_cache_prefill_bank_bytes == 0 {
        bail!("--expert-cache-prefill-bank-bytes must be greater than zero");
    }
    if args.expert_cache_prefill_bank_bytes > args.expert_cache_scratch_bytes {
        bail!("--expert-cache-prefill-bank-bytes cannot exceed --expert-cache-scratch-bytes");
    }
    if args.device_layer_window == 0 {
        bail!("--device-layer-window must be greater than zero");
    }
    if args.mapped_shards == 0 {
        bail!("--mapped-shards must be greater than zero");
    }
    if args.revision.is_some() && Path::new(&args.model).exists() {
        bail!("--revision can only be used with a Hugging Face model identifier");
    }
    if args.raw && args.thinking != ThinkingMode::Auto {
        bail!("--thinking on/off cannot be used with --raw because raw prompts bypass the chat template");
    }
    if args.allow_unparsed_reasoning && args.thinking != ThinkingMode::On {
        bail!("--allow-unparsed-reasoning requires --thinking on");
    }
    if args.raw && args.tools.is_some() {
        bail!(
            "--tools cannot be used with --raw because raw prompts bypass native tool preparation"
        );
    }
    if args.tools.is_none() && args.tool_choice != CliToolChoice::Auto {
        bail!("--tool-choice requires --tools");
    }
    if args.tools.is_none() && args.max_parallel_tool_calls.is_some() {
        bail!("--max-parallel-tool-calls requires --tools");
    }
    if args.tools.is_none() && !args.stop_sequences.is_empty() {
        bail!("--stop requires --tools");
    }
    Ok(())
}

fn read_prompt(argument: Option<&str>) -> Result<String> {
    if let Some(prompt) = argument {
        return Ok(prompt.to_owned());
    }
    if io::stdin().is_terminal() {
        bail!("provide PROMPT as an argument or pipe it on stdin");
    }

    let mut prompt = String::new();
    io::stdin()
        .read_to_string(&mut prompt)
        .context("failed to read the prompt from stdin")?;
    if prompt.is_empty() {
        bail!("stdin contained no prompt");
    }
    Ok(prompt)
}

fn read_tools(path: &Path) -> Result<Vec<serde_json::Value>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open tool definitions {}", path.display()))?;
    let tools: serde_json::Value = serde_json::from_reader(file)
        .with_context(|| format!("failed to parse tool definitions {}", path.display()))?;
    tools
        .as_array()
        .cloned()
        .with_context(|| format!("tool definitions {} must be a JSON array", path.display()))
}

fn resolve_model(
    spec: &str,
    requested_revision: Option<&str>,
    gguf_role: CachedGgufRole,
) -> Result<ResolvedModel> {
    let path = Path::new(spec);
    if path.exists() {
        return Ok(ResolvedModel {
            // Keep the logical filename: Hugging Face snapshot files are
            // symlinks to extensionless blobs, while format dispatch uses the
            // user-visible extension.
            path: std::path::absolute(path)
                .with_context(|| format!("failed to resolve model path {}", path.display()))?,
            repository: None,
            commit_hash: None,
        });
    }

    let (repo_id, quantization) = split_hf_model_spec(spec)?;

    let client = HFClientSync::new().context("failed to initialize the Hugging Face cache")?;
    let cache = client
        .scan_cache()
        .send()
        .context("failed to scan the Hugging Face cache")?;
    let repo = cache
        .repos
        .iter()
        .find(|repo| repo.repo_type == "model" && repo.repo_id == repo_id)
        .with_context(|| {
            format!(
                "{repo_id:?} is not an existing path or a model in the local Hugging Face cache at {}",
                cache.cache_dir.display()
            )
        })?;
    match quantization {
        Some(quantization) => {
            let path = select_cached_gguf_from_revisions(
                &repo.revisions,
                requested_revision,
                quantization,
                gguf_role,
            )
            .with_context(|| {
                format!(
                    "could not select GGUF quantization {quantization:?} for Hugging Face model {repo_id:?}"
                )
            })?;
            let commit_hash = repo
                .revisions
                .iter()
                .find(|revision| revision.files.iter().any(|file| file.file_path == path))
                .map(|revision| revision.commit_hash.clone())
                .with_context(|| {
                    format!(
                        "selected cached artifact {} has no owning revision",
                        path.display()
                    )
                })?;
            Ok(ResolvedModel {
                path,
                repository: Some(repo_id.to_owned()),
                commit_hash: Some(commit_hash),
            })
        }
        None => {
            let revision =
                select_revision(&repo.revisions, requested_revision).with_context(|| {
                    format!("could not select a cached revision for Hugging Face model {repo_id:?}")
                })?;
            let path = if gguf_role == CachedGgufRole::MtpDraft
                && !revision.files.iter().any(|file| {
                    file.file_path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"))
                }) {
                select_unique_cached_gguf(revision, gguf_role)?
                    .unwrap_or_else(|| revision.snapshot_path.clone())
            } else {
                revision.snapshot_path.clone()
            };
            Ok(ResolvedModel {
                path,
                repository: Some(repo_id.to_owned()),
                commit_hash: Some(revision.commit_hash.clone()),
            })
        }
    }
}

fn resolve_model_pair(
    target_spec: &str,
    draft_spec: Option<&str>,
    requested_revision: Option<&str>,
) -> Result<(ResolvedModel, Option<ResolvedModel>)> {
    let Some(draft_spec) = draft_spec else {
        return Ok((
            resolve_model(target_spec, requested_revision, CachedGgufRole::Target)?,
            None,
        ));
    };

    if requested_revision.is_none() {
        if let Some(pair) = resolve_common_cached_gguf_pair(target_spec, draft_spec)? {
            return Ok((pair.0, Some(pair.1)));
        }
    }

    Ok((
        resolve_model(target_spec, requested_revision, CachedGgufRole::Target)?,
        Some(resolve_model(
            draft_spec,
            requested_revision,
            CachedGgufRole::MtpDraft,
        )?),
    ))
}

fn resolve_common_cached_gguf_pair(
    target_spec: &str,
    draft_spec: &str,
) -> Result<Option<(ResolvedModel, ResolvedModel)>> {
    if Path::new(target_spec).exists() || Path::new(draft_spec).exists() {
        return Ok(None);
    }
    let (target_repo, Some(target_quantization)) = split_hf_model_spec(target_spec)? else {
        return Ok(None);
    };
    let (draft_repo, Some(draft_quantization)) = split_hf_model_spec(draft_spec)? else {
        return Ok(None);
    };
    if target_repo != draft_repo {
        return Ok(None);
    }

    let client = HFClientSync::new().context("failed to initialize the Hugging Face cache")?;
    let cache = client
        .scan_cache()
        .send()
        .context("failed to scan the Hugging Face cache")?;
    let Some(repo) = cache
        .repos
        .iter()
        .find(|repo| repo.repo_type == "model" && repo.repo_id == target_repo)
    else {
        return Ok(None);
    };
    let Some((revision, target_path, draft_path)) = select_cached_gguf_pair_from_revisions(
        &repo.revisions,
        target_quantization,
        draft_quantization,
    ) else {
        return Ok(None);
    };
    let repository = Some(target_repo.to_owned());
    let commit_hash = Some(revision.commit_hash.clone());
    Ok(Some((
        ResolvedModel {
            path: target_path,
            repository: repository.clone(),
            commit_hash: commit_hash.clone(),
        },
        ResolvedModel {
            path: draft_path,
            repository,
            commit_hash,
        },
    )))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedModel {
    path: PathBuf,
    repository: Option<String>,
    commit_hash: Option<String>,
}

fn validate_artifact_pair(
    target: &ResolvedModel,
    draft: &ResolvedModel,
    require_same_revision: bool,
) -> Result<bool> {
    let same_repository = target.repository.is_some() && target.repository == draft.repository;
    let mixed_commits = target.commit_hash.is_some()
        && draft.commit_hash.is_some()
        && target.commit_hash != draft.commit_hash;
    let mixed_revisions = same_repository && mixed_commits;
    if mixed_revisions && require_same_revision {
        bail!(
            "target artifact {} resolves to cached commit {}, but draft artifact {} resolves to commit {}; --require-same-revision forbids mixing revisions from repository {:?}",
            target.path.display(),
            target.commit_hash.as_deref().unwrap_or("unknown"),
            draft.path.display(),
            draft.commit_hash.as_deref().unwrap_or("unknown"),
            target.repository.as_deref().unwrap_or("unknown"),
        );
    }
    Ok(mixed_revisions)
}

fn split_hf_model_spec(spec: &str) -> Result<(&str, Option<&str>)> {
    let Some((repo_id, quantization)) = spec.rsplit_once(':') else {
        return Ok((spec, None));
    };
    if repo_id.is_empty() {
        bail!("Hugging Face model identifier before ':' must not be empty");
    }
    if quantization.is_empty() {
        bail!("GGUF quantization selector after ':' must not be empty");
    }
    Ok((repo_id, Some(quantization)))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum QuantizationMatch {
    Exact,
    UnslothAlias,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CachedGgufRole {
    Target,
    MtpDraft,
}

fn select_cached_gguf(
    revision: &CachedRevisionInfo,
    quantization: &str,
    role: CachedGgufRole,
) -> Result<PathBuf> {
    let files = revision
        .files
        .iter()
        .map(|file| file.file_path.as_path())
        .collect::<Vec<_>>();
    select_cached_gguf_path(&files, quantization, role)
}

fn select_unique_cached_gguf(
    revision: &CachedRevisionInfo,
    role: CachedGgufRole,
) -> Result<Option<PathBuf>> {
    let mut candidates = revision
        .files
        .iter()
        .filter_map(|file| {
            let path = file.file_path.as_path();
            if !path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
            {
                return None;
            }
            let stem = path.file_stem()?.to_str()?;
            let (stem, first_shard) = strip_gguf_shard_suffix(stem);
            (first_shard && gguf_role_matches(stem, role)).then_some(path.to_owned())
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();

    match candidates.as_slice() {
        [] => Ok(None),
        [path] => Ok(Some(path.clone())),
        _ => bail!(
            "cached repository contains multiple draft GGUF files: {}; append :QUANT to --draft-model",
            format_cached_paths(&candidates.iter().map(PathBuf::as_path).collect::<Vec<_>>())
        ),
    }
}

fn select_cached_gguf_from_revisions(
    revisions: &[CachedRevisionInfo],
    requested_revision: Option<&str>,
    quantization: &str,
    role: CachedGgufRole,
) -> Result<PathBuf> {
    let preferred = select_revision(revisions, requested_revision)?;
    if requested_revision.is_some() {
        return select_cached_gguf(preferred, quantization, role);
    }
    if let Ok(path) = select_cached_gguf(preferred, quantization, role) {
        return Ok(path);
    }

    // Individual `hf_hub_download` calls can leave files from one repository
    // in separate commit snapshots. Search all cached snapshots when `main`
    // does not contain the requested quantization, while treating repeated
    // pointers to the same content-addressed blob as one candidate.
    let mut seen_blobs = HashSet::new();
    let files = revisions
        .iter()
        .flat_map(|revision| &revision.files)
        .filter(|file| seen_blobs.insert(&file.blob_path))
        .map(|file| file.file_path.as_path())
        .collect::<Vec<_>>();
    select_cached_gguf_path(&files, quantization, role)
}

fn select_cached_gguf_pair_from_revisions<'a>(
    revisions: &'a [CachedRevisionInfo],
    target_quantization: &str,
    draft_quantization: &str,
) -> Option<(&'a CachedRevisionInfo, PathBuf, PathBuf)> {
    let mut ordered = revisions.iter().collect::<Vec<_>>();
    ordered.sort_unstable_by(|left, right| {
        let left_is_main = left.refs.iter().any(|name| name == "main");
        let right_is_main = right.refs.iter().any(|name| name == "main");
        right_is_main
            .cmp(&left_is_main)
            .then_with(|| right.last_modified.cmp(&left.last_modified))
    });
    ordered.into_iter().find_map(|revision| {
        let target =
            select_cached_gguf(revision, target_quantization, CachedGgufRole::Target).ok()?;
        let draft =
            select_cached_gguf(revision, draft_quantization, CachedGgufRole::MtpDraft).ok()?;
        Some((revision, target, draft))
    })
}

fn select_cached_gguf_path(
    files: &[&Path],
    quantization: &str,
    role: CachedGgufRole,
) -> Result<PathBuf> {
    let selector = quantization.to_ascii_uppercase();
    let unsloth_alias = (!selector.starts_with("UD-")).then(|| format!("UD-{selector}"));
    let mut gguf_files = Vec::new();
    let mut candidates = Vec::new();

    for &path in files {
        if !path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            continue;
        }
        gguf_files.push(path);

        let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        let (stem, first_shard) = strip_gguf_shard_suffix(stem);
        if !first_shard || !gguf_role_matches(stem, role) {
            continue;
        }
        let stem = stem.to_ascii_uppercase();
        let matched = if quantization_suffix_matches(&stem, &selector) {
            if unsloth_alias
                .as_deref()
                .is_some_and(|alias| quantization_suffix_matches(&stem, alias))
            {
                QuantizationMatch::UnslothAlias
            } else {
                QuantizationMatch::Exact
            }
        } else {
            continue;
        };
        candidates.push((path, matched));
    }

    if candidates
        .iter()
        .any(|(_, matched)| *matched == QuantizationMatch::Exact)
    {
        candidates.retain(|(_, matched)| *matched == QuantizationMatch::Exact);
    }
    candidates.sort_unstable_by_key(|(path, _)| *path);

    match candidates.as_slice() {
        [(path, _)] => Ok((*path).to_owned()),
        [] => {
            let available = format_cached_paths(&gguf_files);
            if available.is_empty() {
                bail!("the selected cached revision contains no GGUF files");
            }
            bail!(
                "no cached GGUF filename matches quantization {quantization:?}; available GGUF files: {available}"
            )
        }
        _ => {
            let paths = candidates.iter().map(|(path, _)| *path).collect::<Vec<_>>();
            bail!(
                "quantization {quantization:?} matches multiple cached GGUF files: {}",
                format_cached_paths(&paths)
            )
        }
    }
}

fn gguf_role_matches(stem: &str, role: CachedGgufRole) -> bool {
    let stem = stem.to_ascii_lowercase();
    let mtp_sidecar = stem.starts_with("mtp-")
        || stem == "dflash"
        || stem.starts_with("dflash-")
        || stem.starts_with("dflash_");
    match role {
        CachedGgufRole::Target => !mtp_sidecar && !stem.starts_with("mmproj-"),
        CachedGgufRole::MtpDraft => mtp_sidecar,
    }
}

fn quantization_suffix_matches(stem: &str, selector: &str) -> bool {
    let Some(prefix) = stem.strip_suffix(selector) else {
        return false;
    };
    prefix.is_empty()
        || prefix
            .chars()
            .next_back()
            .is_some_and(|separator| matches!(separator, '-' | '.' | '_'))
}

fn strip_gguf_shard_suffix(stem: &str) -> (&str, bool) {
    let Some((prefix, count)) = stem.rsplit_once("-of-") else {
        return (stem, true);
    };
    let Some((base, number)) = prefix.rsplit_once('-') else {
        return (stem, true);
    };
    let canonical_number = number.len() == 5 && number.bytes().all(|byte| byte.is_ascii_digit());
    let canonical_count = count.len() == 5 && count.bytes().all(|byte| byte.is_ascii_digit());
    if canonical_number && canonical_count {
        (base, number == "00001")
    } else {
        (stem, true)
    }
}

fn format_cached_paths(paths: &[&Path]) -> String {
    let mut paths = paths
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    paths.sort_unstable();
    paths.join(", ")
}

fn select_revision<'a>(
    revisions: &'a [CachedRevisionInfo],
    requested: Option<&str>,
) -> Result<&'a CachedRevisionInfo> {
    if let Some(requested) = requested {
        return revisions
            .iter()
            .find(|revision| {
                revision.commit_hash == requested
                    || revision.refs.iter().any(|name| name == requested)
            })
            .with_context(|| format!("revision {requested:?} is not present in the cache"));
    }

    revisions
        .iter()
        .find(|revision| revision.refs.iter().any(|name| name == "main"))
        .or_else(|| {
            revisions
                .iter()
                .max_by_key(|revision| revision.last_modified)
        })
        .context("the cached repository contains no snapshots")
}

#[cfg(test)]
mod tests {
    use std::{
        path::Path,
        time::{Duration, SystemTime},
    };

    use clap::{CommandFactory, Parser};
    use hf_hub::cache::{CachedFileInfo, CachedRevisionInfo};

    use super::{
        eval, execution_contexts, format_bytes, format_weight_store_diagnostics,
        requested_load_quantization, select_cached_gguf_from_revisions,
        select_cached_gguf_pair_from_revisions, select_cached_gguf_path, select_revision,
        select_unique_cached_gguf, should_report_stop_reason, split_hf_model_spec, stop_reason,
        use_semantic_generation, validate_args, validate_artifact_pair, write_semantic_event,
        write_timing_report, Array, CachedGgufRole, Cli, CliDevice, CliToolChoice, DeviceType,
        MtpDraftDevice, MtpExecutionStreams, MtpSchedulerOptions, NativeToolSupport,
        ReasoningStream, ResolvedModel, SemanticEvent, SemanticSupport, StopReason,
        WeightQuantization,
    };

    fn revision(hash: &str, refs: &[&str], modified: u64) -> CachedRevisionInfo {
        CachedRevisionInfo {
            commit_hash: hash.to_owned(),
            snapshot_path: hash.into(),
            files: Vec::new(),
            size_on_disk: 0,
            refs: refs.iter().map(|value| (*value).to_owned()).collect(),
            last_modified: SystemTime::UNIX_EPOCH + Duration::from_secs(modified),
        }
    }

    fn cached_file(file_path: &str, blob_path: &str) -> CachedFileInfo {
        CachedFileInfo {
            file_name: file_path.to_owned(),
            file_path: file_path.into(),
            blob_path: blob_path.into(),
            size_on_disk: 0,
            blob_last_accessed: SystemTime::UNIX_EPOCH,
            blob_last_modified: SystemTime::UNIX_EPOCH,
        }
    }

    #[test]
    fn command_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_timing_without_enabling_verbose_output() {
        let args = Cli::try_parse_from(["safemlx-lm", "--model", "model-id", "--timing", "prompt"])
            .unwrap();
        assert!(args.timing);
        assert!(!args.verbose);
    }

    #[test]
    fn timing_report_contains_only_timing_statistics() {
        let mut stderr = Vec::new();
        write_timing_report(
            &mut stderr,
            Duration::from_millis(1250),
            Duration::from_millis(2000),
            Some(Duration::from_millis(500)),
            7,
            Duration::from_millis(4000),
        )
        .unwrap();

        assert_eq!(
            String::from_utf8(stderr).unwrap(),
            "load_time: 1.250 s\n\
             generation_time: 2.000 s\n\
             time_to_first_token: 0.500 s\n\
             decode_token_rate: 4.00 tokens/s\n\
             token_rate: 3.50 tokens/s\n\
             total_execution_time: 4.000 s\n"
        );
    }

    #[test]
    fn rejects_raw_thinking_modes() {
        let raw = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--raw",
            "--thinking",
            "off",
            "prompt",
        ])
        .unwrap();
        assert!(validate_args(&raw)
            .unwrap_err()
            .to_string()
            .contains("raw prompts bypass the chat template"));
    }

    #[test]
    fn parses_native_tool_options_and_keeps_raw_generation_explicit() {
        let tools = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--tools",
            "tools.json",
            "--tool-choice",
            "required",
            "--max-parallel-tool-calls",
            "2",
            "--stop",
            "<done>",
            "prompt",
        ])
        .unwrap();
        assert_eq!(tools.tool_choice, CliToolChoice::Required);
        assert_eq!(tools.max_parallel_tool_calls.unwrap().get(), 2);
        assert_eq!(tools.stop_sequences, ["<done>"]);
        validate_args(&tools).unwrap();

        let raw_tools = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--raw",
            "--tools",
            "tools.json",
            "prompt",
        ])
        .unwrap();
        assert!(validate_args(&raw_tools)
            .unwrap_err()
            .to_string()
            .contains("--tools cannot be used with --raw"));

        let raw =
            Cli::try_parse_from(["safemlx-lm", "--model", "model-id", "--raw", "prompt"]).unwrap();
        assert!(
            validate_args(&raw).is_ok(),
            "raw unconstrained generation remains intentionally supported"
        );
    }

    #[test]
    fn registered_profiles_use_semantic_generation_without_tools() {
        assert!(use_semantic_generation(
            &SemanticSupport::Supported,
            &NativeToolSupport::Supported,
            false
        )
        .unwrap());
        assert!(use_semantic_generation(
            &SemanticSupport::Supported,
            &NativeToolSupport::Supported,
            true
        )
        .unwrap());

        let unsupported = NativeToolSupport::Unsupported {
            reason: "unregistered template".into(),
        };
        let semantic_unsupported = SemanticSupport::Unsupported {
            reason: "unregistered template".into(),
        };
        assert!(!use_semantic_generation(&semantic_unsupported, &unsupported, false).unwrap());
        assert!(
            use_semantic_generation(&SemanticSupport::Supported, &unsupported, true)
                .unwrap_err()
                .to_string()
                .contains("native tool calling is unavailable")
        );
    }

    #[test]
    fn semantic_output_hides_reasoning_and_writes_visible_text() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut streamed_text = String::new();
        let mut reasoning_stream = ReasoningStream::default();
        write_semantic_event(
            &SemanticEvent::ReasoningDelta("private thought".into()),
            &mut stdout,
            &mut stderr,
            &mut streamed_text,
            &mut reasoning_stream,
            false,
        )
        .unwrap();
        write_semantic_event(
            &SemanticEvent::TextDelta("visible answer".into()),
            &mut stdout,
            &mut stderr,
            &mut streamed_text,
            &mut reasoning_stream,
            false,
        )
        .unwrap();

        assert_eq!(stdout, b"visible answer");
        assert!(stderr.is_empty());
        assert_eq!(streamed_text, "visible answer");
    }

    #[test]
    fn verbose_semantic_output_streams_reasoning_in_event_order() {
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut streamed_text = String::new();
        let mut reasoning_stream = ReasoningStream::default();

        write_semantic_event(
            &SemanticEvent::ReasoningDelta("private ".into()),
            &mut stdout,
            &mut stderr,
            &mut streamed_text,
            &mut reasoning_stream,
            true,
        )
        .unwrap();
        assert!(stdout.is_empty());
        assert_eq!(stderr, b"--- reasoning content (stderr) ---\nprivate ");

        write_semantic_event(
            &SemanticEvent::ReasoningDelta("thought\nsecond line".into()),
            &mut stdout,
            &mut stderr,
            &mut streamed_text,
            &mut reasoning_stream,
            true,
        )
        .unwrap();
        assert_eq!(
            stderr,
            b"--- reasoning content (stderr) ---\nprivate thought\nsecond line"
        );

        write_semantic_event(
            &SemanticEvent::TextDelta("visible answer".into()),
            &mut stdout,
            &mut stderr,
            &mut streamed_text,
            &mut reasoning_stream,
            true,
        )
        .unwrap();
        assert_eq!(stdout, b"visible answer");
        assert_eq!(streamed_text, "visible answer");
        assert_eq!(
            stderr,
            b"--- reasoning content (stderr) ---\nprivate thought\nsecond line\n\
              --- end reasoning content (stderr) ---\n\
              --- generated content (stdout) ---\n"
        );
    }

    #[test]
    fn classifies_generation_stop_reason() {
        assert_eq!(stop_reason(&[4, 2], &[2], 10), StopReason::Eos);
        assert_eq!(stop_reason(&[4, 5], &[2], 2), StopReason::MaxTokens);
        assert_eq!(stop_reason(&[4], &[2], 2), StopReason::GeneratorExhausted);
    }

    #[test]
    fn reports_only_max_tokens_without_verbose_output() {
        assert!(should_report_stop_reason(StopReason::MaxTokens, false));
        assert!(!should_report_stop_reason(StopReason::Eos, false));
        assert!(!should_report_stop_reason(
            StopReason::GeneratorExhausted,
            false
        ));
        assert!(should_report_stop_reason(StopReason::Eos, true));
    }

    #[test]
    fn validates_load_time_quantization_arguments() {
        let valid = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--quantize",
            "4",
            "prompt",
        ])
        .unwrap();
        validate_args(&valid).unwrap();

        let invalid = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--quantize",
            "7",
            "prompt",
        ])
        .unwrap();
        assert!(validate_args(&invalid)
            .unwrap_err()
            .to_string()
            .contains("bits must be one of"));

        let mxfp4 = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--quantize",
            "4",
            "--quantization-mode",
            "mxfp4",
            "prompt",
        ])
        .unwrap();
        validate_args(&mxfp4).unwrap();
        assert_eq!(
            requested_load_quantization(&mxfp4).unwrap(),
            Some(WeightQuantization::MxFp4)
        );

        let invalid_mxfp4 = Cli {
            quantize: Some(3),
            ..mxfp4
        };
        assert!(validate_args(&invalid_mxfp4)
            .unwrap_err()
            .to_string()
            .contains("requires --quantize 4"));
    }

    #[test]
    fn validates_mirostat_v2_arguments() {
        let valid = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--mirostat-v2",
            "--temperature",
            "1.0",
            "prompt",
        ])
        .unwrap();
        validate_args(&valid).unwrap();

        let zero_temperature = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--mirostat-v2",
            "prompt",
        ])
        .unwrap();
        assert!(validate_args(&zero_temperature)
            .unwrap_err()
            .to_string()
            .contains("--temperature greater than zero"));

        let speculative = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--draft-model",
            "draft-id",
            "--mirostat-v2",
            "--temperature",
            "1.0",
            "prompt",
        ])
        .unwrap();
        validate_args(&speculative).unwrap();
    }

    #[test]
    fn parses_main_and_draft_device_selectors() {
        assert_eq!("cpu".parse::<CliDevice>().unwrap(), CliDevice::Cpu);
        assert_eq!("gpu:0".parse::<CliDevice>().unwrap(), CliDevice::Gpu(0));
        assert_eq!("gpu:12".parse::<CliDevice>().unwrap(), CliDevice::Gpu(12));
        assert!("gpu".parse::<CliDevice>().is_err());
        assert!("gpu:-1".parse::<CliDevice>().is_err());
        assert!("gpu:one".parse::<CliDevice>().is_err());

        assert_eq!(
            "target".parse::<MtpDraftDevice>().unwrap(),
            MtpDraftDevice::Target
        );
        assert_eq!(
            "cpu".parse::<MtpDraftDevice>().unwrap(),
            MtpDraftDevice::Device(CliDevice::Cpu)
        );
        assert_eq!(
            "gpu:3".parse::<MtpDraftDevice>().unwrap(),
            MtpDraftDevice::Device(CliDevice::Gpu(3))
        );

        let defaults =
            Cli::try_parse_from(["safemlx-lm", "--model", "model-id", "prompt"]).unwrap();
        assert_eq!(defaults.device, CliDevice::Gpu(0));
        assert_eq!(defaults.mtp_draft_device, MtpDraftDevice::Target);

        let cpu_target = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--device",
            "cpu",
            "prompt",
        ])
        .unwrap();
        assert_eq!(cpu_target.device, CliDevice::Cpu);
        validate_args(&cpu_target).unwrap();
    }

    #[test]
    fn explicit_mtp_draft_devices_require_external_drafter() {
        for device in ["cpu", "gpu:0", "gpu:3"] {
            let without_drafter = Cli::try_parse_from([
                "safemlx-lm",
                "--model",
                "model-id",
                "--mtp-draft-device",
                device,
                "prompt",
            ])
            .unwrap();
            assert!(validate_args(&without_drafter)
                .unwrap_err()
                .to_string()
                .contains("requires an external --draft-model"));

            let with_drafter = Cli::try_parse_from([
                "safemlx-lm",
                "--model",
                "model-id",
                "--draft-model",
                "draft-id",
                "--mtp-draft-device",
                device,
                "prompt",
            ])
            .unwrap();
            validate_args(&with_drafter).unwrap();
        }
    }

    #[test]
    fn cpu_execution_device_runs_mlx_work() {
        let (context, no_draft) = execution_contexts(CliDevice::Cpu, MtpDraftDevice::Target);
        assert!(no_draft.is_none());
        assert_eq!(context.device().get_type().unwrap(), DeviceType::Cpu);
        let values = Array::from_slice(&[1.0f32, 2.0], &[2])
            .copy(context.stream())
            .unwrap();
        eval([&values]).unwrap();
        context.stream().synchronize().unwrap();
        assert_eq!(values.evaluated().unwrap().as_slice::<f32>(), &[1.0, 2.0]);

        let (target, draft) =
            execution_contexts(CliDevice::Cpu, MtpDraftDevice::Device(CliDevice::Cpu));
        let draft = draft.unwrap();
        assert_ne!(target.stream(), draft.stream());
        assert_eq!(
            MtpExecutionStreams::new(target.stream(), draft.stream())
                .unwrap()
                .topology(),
            safemlx_lm::runtime::generation::speculative::MtpStreamTopology::SameDeviceSplit
        );
    }

    #[test]
    #[ignore = "requires an MLX Metal device"]
    fn gpu_device_selectors_build_requested_mtp_topologies() {
        let (target, draft) =
            execution_contexts(CliDevice::Gpu(0), MtpDraftDevice::Device(CliDevice::Gpu(0)));
        assert_eq!(
            MtpExecutionStreams::new(target.stream(), draft.unwrap().stream())
                .unwrap()
                .topology(),
            safemlx_lm::runtime::generation::speculative::MtpStreamTopology::SameDeviceSplit
        );

        let (target, draft) =
            execution_contexts(CliDevice::Gpu(0), MtpDraftDevice::Device(CliDevice::Cpu));
        assert_eq!(
            MtpExecutionStreams::new(target.stream(), draft.unwrap().stream())
                .unwrap()
                .topology(),
            safemlx_lm::runtime::generation::speculative::MtpStreamTopology::CrossDeviceSplit
        );

        let (target, draft) =
            execution_contexts(CliDevice::Cpu, MtpDraftDevice::Device(CliDevice::Gpu(0)));
        assert_eq!(
            MtpExecutionStreams::new(target.stream(), draft.unwrap().stream())
                .unwrap()
                .topology(),
            safemlx_lm::runtime::generation::speculative::MtpStreamTopology::CrossDeviceSplit
        );
    }

    #[test]
    fn parses_mtp_lookahead_controls() {
        let args = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--draft-model",
            "draft-id",
            "--disable-mtp-lookahead",
            "--disable-mtp-adaptive-lookahead",
            "--verbose",
            "prompt",
        ])
        .unwrap();

        assert!(args.disable_mtp_lookahead);
        assert!(args.disable_mtp_adaptive_lookahead);
        let options = MtpSchedulerOptions {
            adaptive_lookahead: !args.disable_mtp_adaptive_lookahead,
            ..MtpSchedulerOptions::default()
        }
        .with_lookahead(!args.disable_mtp_lookahead);
        assert_eq!(options.lookahead_blocks, 0);
        assert!(!options.adaptive_lookahead);
    }

    #[test]
    fn parses_mlx_allocator_cache_limit() {
        let args = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--mlx-cache-limit-bytes",
            "17179869184",
            "prompt",
        ])
        .unwrap();

        assert_eq!(args.mlx_cache_limit_bytes, Some(17_179_869_184));
        validate_args(&args).unwrap();
    }

    #[test]
    fn parses_and_validates_expert_cache_prefill_bank_target() {
        let args = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--expert-cache",
            "--expert-cache-scratch-bytes",
            "4096",
            "--expert-cache-prefill-bank-bytes",
            "1024",
            "prompt",
        ])
        .unwrap();
        assert_eq!(args.expert_cache_scratch_bytes, 4096);
        assert_eq!(args.expert_cache_prefill_bank_bytes, 1024);
        validate_args(&args).unwrap();

        let invalid = Cli {
            expert_cache_prefill_bank_bytes: 4097,
            ..args
        };
        assert!(validate_args(&invalid).is_err());
    }

    #[test]
    fn validates_dense_stream_residency_conflicts() {
        for arguments in [
            vec!["--dense-disk-stream", "--layerwise-host"],
            vec!["--dense-disk-stream", "--quantize", "4"],
        ] {
            let mut command = vec!["safemlx-lm", "--model", "model-id"];
            command.extend(arguments);
            command.push("prompt");
            assert!(validate_args(&Cli::try_parse_from(command).unwrap()).is_err());
        }
    }

    #[test]
    fn accepts_combined_expert_cache_and_dense_streaming() {
        let arguments = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--expert-cache",
            "--dense-disk-stream",
            "prompt",
        ])
        .unwrap();
        validate_args(&arguments).unwrap();
    }

    #[test]
    fn accepts_load_time_quantization_with_a_resident_expert_cache() {
        let arguments = Cli::try_parse_from([
            "safemlx-lm",
            "--model",
            "model-id",
            "--expert-cache",
            "--quantize",
            "4",
            "prompt",
        ])
        .unwrap();
        validate_args(&arguments).unwrap();
    }

    #[test]
    fn selects_main_by_default() {
        let revisions = [revision("older", &["main"], 1), revision("newer", &[], 2)];
        assert_eq!(
            select_revision(&revisions, None).unwrap().commit_hash,
            "older"
        );
    }

    #[test]
    fn selects_requested_ref_or_hash() {
        let revisions = [
            revision("first", &["main"], 1),
            revision("second", &["experiment"], 2),
        ];
        assert_eq!(
            select_revision(&revisions, Some("experiment"))
                .unwrap()
                .commit_hash,
            "second"
        );
        assert_eq!(
            select_revision(&revisions, Some("first"))
                .unwrap()
                .commit_hash,
            "first"
        );
    }

    #[test]
    fn selects_newest_snapshot_without_main() {
        let revisions = [revision("older", &[], 1), revision("newer", &[], 2)];
        assert_eq!(
            select_revision(&revisions, None).unwrap().commit_hash,
            "newer"
        );
    }

    #[test]
    fn parses_hugging_face_quantization_selector() {
        assert_eq!(
            split_hf_model_spec("unsloth/model-GGUF:UD-Q4_K_M").unwrap(),
            ("unsloth/model-GGUF", Some("UD-Q4_K_M"))
        );
        assert_eq!(
            split_hf_model_spec("unsloth/model-GGUF").unwrap(),
            ("unsloth/model-GGUF", None)
        );
        assert!(split_hf_model_spec("unsloth/model-GGUF:").is_err());
    }

    #[test]
    fn selects_exact_and_unsloth_aliased_quantizations() {
        let q4 = Path::new("snapshot/model-Q4_K_M.gguf");
        let ud_q4 = Path::new("snapshot/model-UD-Q4_K_M.gguf");
        let files = [q4, ud_q4];

        assert_eq!(
            select_cached_gguf_path(&files, "UD-Q4_K_M", CachedGgufRole::Target).unwrap(),
            ud_q4
        );
        assert_eq!(
            select_cached_gguf_path(&files, "q4_k_m", CachedGgufRole::Target).unwrap(),
            q4
        );
        assert_eq!(
            select_cached_gguf_path(&[ud_q4], "Q4_K_M", CachedGgufRole::Target).unwrap(),
            ud_q4
        );
    }

    #[test]
    fn distinguishes_target_and_mtp_sidecar_at_same_quantization() {
        let target = Path::new("snapshot/gemma-4-26B-A4B-it-Q8_0.gguf");
        let draft = Path::new("snapshot/MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf");
        let files = [target, draft];

        assert_eq!(
            select_cached_gguf_path(&files, "Q8_0", CachedGgufRole::Target).unwrap(),
            target
        );
        assert_eq!(
            select_cached_gguf_path(&files, "Q8_0", CachedGgufRole::MtpDraft).unwrap(),
            draft
        );
    }

    #[test]
    fn selects_muse_dflash_as_the_unique_draft_sidecar() {
        let mut revision = revision("main", &["main"], 1);
        revision.files = vec![
            cached_file("main/muse-glimmer-30B-kquant-17gb.gguf", "blobs/target"),
            cached_file("main/mmproj-kquant.gguf", "blobs/projector"),
            cached_file("main/dflash-kquant.gguf", "blobs/dflash"),
        ];

        assert_eq!(
            select_unique_cached_gguf(&revision, CachedGgufRole::MtpDraft).unwrap(),
            Some(Path::new("main/dflash-kquant.gguf").into())
        );
        assert_eq!(
            select_cached_gguf_path(
                &revision
                    .files
                    .iter()
                    .map(|file| file.file_path.as_path())
                    .collect::<Vec<_>>(),
                "dflash-kquant",
                CachedGgufRole::MtpDraft,
            )
            .unwrap(),
            Path::new("main/dflash-kquant.gguf")
        );
    }

    #[test]
    fn bare_draft_repository_rejects_ambiguous_sidecars() {
        let mut revision = revision("main", &["main"], 1);
        revision.files = vec![
            cached_file("main/dflash-Q4_K.gguf", "blobs/q4"),
            cached_file("main/dflash-Q8_0.gguf", "blobs/q8"),
        ];

        let error = select_unique_cached_gguf(&revision, CachedGgufRole::MtpDraft)
            .unwrap_err()
            .to_string();
        assert!(error.contains("multiple draft GGUF files"));
        assert!(error.contains("append :QUANT"));
    }

    #[test]
    fn finds_quantizations_across_separate_cached_snapshots() {
        let mut target = revision("target", &[], 1);
        target.files = vec![cached_file(
            "target/model-UD-Q4_K_M.gguf",
            "blobs/target-q4",
        )];
        let mut main = revision("assistant", &["main"], 2);
        main.files = vec![cached_file(
            "assistant/MTP/mtp-assistant-Q8_0.gguf",
            "blobs/assistant-q8",
        )];
        let revisions = [target, main];

        assert_eq!(
            select_cached_gguf_from_revisions(&revisions, None, "Q4_K_M", CachedGgufRole::Target,)
                .unwrap(),
            Path::new("target/model-UD-Q4_K_M.gguf")
        );
        assert_eq!(
            select_cached_gguf_from_revisions(&revisions, None, "Q8_0", CachedGgufRole::MtpDraft,)
                .unwrap(),
            Path::new("assistant/MTP/mtp-assistant-Q8_0.gguf")
        );
    }

    #[test]
    fn explicit_revision_limits_quantization_selection() {
        let mut target = revision("target", &[], 1);
        target.files = vec![cached_file(
            "target/model-UD-Q4_K_M.gguf",
            "blobs/target-q4",
        )];
        let main = revision("assistant", &["main"], 2);
        let revisions = [target, main];

        assert!(select_cached_gguf_from_revisions(
            &revisions,
            Some("main"),
            "Q4_K_M",
            CachedGgufRole::Target,
        )
        .is_err());
    }

    #[test]
    fn prefers_a_complete_target_and_draft_snapshot() {
        let mut complete = revision("complete", &[], 1);
        complete.files = vec![
            cached_file("complete/model-UD-Q4_K_M.gguf", "blobs/complete-q4"),
            cached_file(
                "complete/MTP/mtp-model-Q8_0.gguf",
                "blobs/complete-draft-q8",
            ),
        ];
        let mut main = revision("main", &["main"], 2);
        main.files = vec![cached_file("main/model-UD-Q4_K_M.gguf", "blobs/main-q4")];
        let revisions = [complete, main];

        let (revision, target, draft) =
            select_cached_gguf_pair_from_revisions(&revisions, "Q4_K_M", "Q8_0").unwrap();
        assert_eq!(revision.commit_hash, "complete");
        assert_eq!(target, Path::new("complete/model-UD-Q4_K_M.gguf"));
        assert_eq!(draft, Path::new("complete/MTP/mtp-model-Q8_0.gguf"));
    }

    #[test]
    fn mixed_repository_revisions_are_advisory_unless_strictly_required() {
        let target = ResolvedModel {
            path: "target.gguf".into(),
            repository: Some("owner/model".into()),
            commit_hash: Some("target-commit".into()),
        };
        let draft = ResolvedModel {
            path: "draft.gguf".into(),
            repository: Some("owner/model".into()),
            commit_hash: Some("draft-commit".into()),
        };

        assert!(validate_artifact_pair(&target, &draft, false).unwrap());
        let error = validate_artifact_pair(&target, &draft, true).unwrap_err();
        assert!(error.to_string().contains("--require-same-revision"));

        let mut other_repository = draft;
        other_repository.repository = Some("owner/assistant".into());
        assert!(!validate_artifact_pair(&target, &other_repository, false).unwrap());
    }

    #[test]
    fn shared_blobs_are_not_ambiguous_across_snapshots() {
        let mut older = revision("older", &[], 1);
        older.files = vec![cached_file("older/model-Q4_K_M.gguf", "blobs/shared-q4")];
        let mut main = revision("main", &["main"], 2);
        main.files = vec![cached_file("main/other-Q8_0.gguf", "blobs/other-q8")];
        let mut newer = revision("newer", &[], 3);
        newer.files = vec![cached_file("newer/model-Q4_K_M.gguf", "blobs/shared-q4")];
        let revisions = [older, main, newer];

        assert_eq!(
            select_cached_gguf_from_revisions(&revisions, None, "Q4_K_M", CachedGgufRole::Target,)
                .unwrap(),
            Path::new("older/model-Q4_K_M.gguf")
        );
    }

    #[test]
    fn selects_first_shard_for_quantization() {
        let first = Path::new("snapshot/model-Q4_K_M-00001-of-00002.gguf");
        let second = Path::new("snapshot/model-Q4_K_M-00002-of-00002.gguf");
        assert_eq!(
            select_cached_gguf_path(&[second, first], "Q4_K_M", CachedGgufRole::Target,).unwrap(),
            first
        );
    }

    #[test]
    fn rejects_ambiguous_or_missing_quantizations() {
        let first = Path::new("snapshot/first-Q4_K_M.gguf");
        let second = Path::new("snapshot/second-Q4_K_M.gguf");
        let error = select_cached_gguf_path(&[first, second], "Q4_K_M", CachedGgufRole::Target)
            .unwrap_err()
            .to_string();
        assert!(error.contains("matches multiple cached GGUF files"));

        let error = select_cached_gguf_path(&[first], "Q8_0", CachedGgufRole::Target)
            .unwrap_err()
            .to_string();
        assert!(error.contains("available GGUF files"));
    }

    #[test]
    fn formats_memory_with_exact_bytes() {
        assert_eq!(format_bytes(512), "512.00 B (512 bytes)");
        assert_eq!(format_bytes(1536), "1.50 KiB (1536 bytes)");
        assert_eq!(
            format_bytes(3 * 1024 * 1024 * 1024),
            "3.00 GiB (3221225472 bytes)"
        );
    }

    #[test]
    fn concise_weight_store_diagnostics_omit_shard_paths() {
        let diagnostics = safemlx_lm::runtime::checkpoint::store::WeightStoreDiagnostics {
            backend: safemlx_lm::runtime::checkpoint::store::WeightStoreBackend::Safetensors,
            mapping_hits: 17,
            mapping_misses: 2,
            evictions: 1,
            currently_mapped_shards: 2,
            touched_shard_paths: vec![
                Path::new("/private/checkpoint/model-00001.safetensors").into(),
                Path::new("/tmp/quantized-00000.safetensors").into(),
            ],
            physical_reads: 3,
            physical_read_bytes: 4096,
            coalesced_group_hits: 4,
        };

        let formatted = format_weight_store_diagnostics(&diagnostics);
        assert!(formatted.contains("backend=Safetensors"));
        assert!(formatted.contains("touched_shards=2"));
        assert!(!formatted.contains("model-00001.safetensors"));
        assert!(!formatted.contains("quantized-00000.safetensors"));
    }
}
