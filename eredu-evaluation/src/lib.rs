//! Backend-neutral model evaluation drivers.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod checkpoint;
mod distribution;
mod evidence;
mod parity;
mod realtime;

pub use checkpoint::{
    compare_checkpoint_artifacts, CheckpointParityError, CheckpointParityOptions,
    CheckpointParityReport,
};
pub use distribution::{compare_distributions, DistributionError, DistributionMetrics};

pub use evidence::{
    observe_f32_tensor, observe_i32_tensor, observe_realtime_frame, summarize_latencies,
    EvaluationEvidence, EvidenceError, LatencySummary,
};
pub use parity::{
    compare_observations, LogitRowMetrics, LogitTolerance, NumericMetrics, NumericTolerance,
    ObservationParity, ParityComparison, ParityError, ParityMetrics, ParityPolicy, ParityReport,
    ParityRule,
};
pub use realtime::{encoded_audio_frames, run_realtime_trace, RealtimeTrace, RealtimeTraceError};

use std::{
    error::Error,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use eredu_architectures::moshi::personaplex_prompt::{
    wrap_system_prompt, AUDIO_TOKENS_PER_STREAM, SILENCE_TOKENS, SINE_TOKENS, TEXT_PADDING_TOKEN,
};
use eredu_codec::mimi::Mimi;
use eredu_core::{
    scheduler::{RequestId, SchedulerLimits},
    RealtimeBackend, RealtimeInputFrame, RealtimeModel, RealtimeOutputFrame, RealtimeSampling,
    RealtimeScheduler,
};
use eredu_nn::Tensor;
use sentencepiece_rs::SentencePieceProcessor;
use serde::Serialize;
use serde_json::json;

const SAMPLE_RATE: u32 = 24_000;
const FRAME_RATE: f64 = 12.5;
const FRAME_SAMPLES: usize = 1_920;
const DEADLINE_MS: f64 = 1_000.0 / FRAME_RATE;
const TAIL_ACTIVITY_FRAMES: usize = 3;
const ACTIVE_AUDIO_DBFS: f64 = -40.0;
const PROMPT_SILENCE_FRAMES: usize = 6;

/// Default PersonaPlex system instruction used by the evaluator.
pub const DEFAULT_TEXT_PROMPT: &str = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.";
/// Default deterministic sampling seed.
pub const DEFAULT_SAMPLING_SEED: u64 = 20_260_713;

/// Paths consumed and produced by one PersonaPlex comparison.
#[derive(Debug, Clone)]
pub struct PersonaPlexEvaluationPaths {
    /// Dense model artifact.
    pub dense_model: PathBuf,
    /// Quantized model artifact.
    pub quantized_model: PathBuf,
    /// SentencePiece text tokenizer.
    pub text_tokenizer: PathBuf,
    /// Mono 24 kHz raw `f32le` voice prompt.
    pub voice_prompt: PathBuf,
    /// Mono 24 kHz raw `f32le` user input.
    pub input: PathBuf,
    /// New output directory.
    pub output: PathBuf,
}

/// Controls for one PersonaPlex comparison.
#[derive(Debug, Clone)]
pub struct PersonaPlexEvaluationOptions {
    /// Maximum input frames, or all complete frames when omitted.
    pub frames: Option<usize>,
    /// Unwrapped system instruction.
    pub text_prompt: String,
    /// Root seed used independently by both model runs.
    pub sampling_seed: u64,
}

impl Default for PersonaPlexEvaluationOptions {
    fn default() -> Self {
        Self {
            frames: None,
            text_prompt: DEFAULT_TEXT_PROMPT.into(),
            sampling_seed: DEFAULT_SAMPLING_SEED,
        }
    }
}

/// Runs the complete PersonaPlex dense-versus-quantized evaluation.
///
/// The model loader is the only backend composition hook. Realtime inputs,
/// outputs, forcing, sampling, scheduling, diagnostics, codec execution, and
/// reporting use backend-neutral contracts.
pub fn run_personaplex_quantization<B, T, L>(
    paths: &PersonaPlexEvaluationPaths,
    options: &PersonaPlexEvaluationOptions,
    mimi: &mut Mimi<T>,
    context: &T::Context,
    mut load_model: L,
) -> Result<(), Box<dyn Error>>
where
    B: RealtimeBackend,
    T: Tensor,
    L: FnMut(&Path) -> Result<RealtimeModel<B>, Box<dyn Error>>,
{
    if paths.output.exists() {
        return Err(invalid(format!(
            "output directory already exists: {}",
            paths.output.display()
        )));
    }
    let voice_pcm = read_f32le(&paths.voice_prompt)?;
    if voice_pcm.len() < FRAME_SAMPLES {
        return Err(invalid("voice prompt contains no complete 80 ms frame"));
    }
    let input_pcm = read_f32le(&paths.input)?;
    let available_frames = input_pcm.len() / FRAME_SAMPLES;
    let frames = options
        .frames
        .unwrap_or(available_frames)
        .min(available_frames);
    if frames < 4 {
        return Err(invalid(format!(
            "input must contain at least four complete frames; found {available_frames}"
        )));
    }
    let input_pcm = &input_pcm[..frames * FRAME_SAMPLES];
    let input_tail = tail_max_rms_dbfs(input_pcm);
    let input_likely_truncated = input_tail > ACTIVE_AUDIO_DBFS;
    let input_warning = input_likely_truncated.then_some(
        "The final 240 ms contains active audio; the frame limit may truncate the user utterance.",
    );
    if let Some(warning) = input_warning {
        eprintln!("warning: {warning} tail_max_rms_dbfs={input_tail:.1}");
    }

    let codec_start = Instant::now();
    let voice_tokens = encode_pcm(mimi, &voice_pcm, context)?;
    let input_tokens = encode_pcm(mimi, input_pcm, context)?;
    if input_tokens.len() != frames {
        return Err(invalid(format!(
            "Mimi produced {} frames for {frames} PCM frames",
            input_tokens.len()
        )));
    }
    let encode_seconds = codec_start.elapsed().as_secs_f64();
    let offline = offline_roundtrip(mimi, input_pcm, context)?;
    let offline_tokens = offline.tokens;
    let offline_roundtrip = offline.pcm;
    let streaming_roundtrip = decode_tokens(mimi, &input_tokens, context, input_pcm.len())?;
    let codec_agreement = token_frame_agreement(&input_tokens, &offline_tokens);

    let tokenizer = SentencePieceProcessor::open(&paths.text_tokenizer)?;
    let wrapped_text_prompt = wrap_system_prompt(&options.text_prompt);
    let text_tokens = tokenizer
        .encode_to_ids(&wrapped_text_prompt)?
        .into_iter()
        .map(i32::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    if text_tokens.is_empty() {
        return Err(invalid("text prompt tokenized to an empty sequence"));
    }
    let prompt = PromptConditioning {
        voice_frames: voice_tokens,
        text_tokens,
    };

    let dense_load_start = Instant::now();
    let mut dense = load_model(&paths.dense_model)?;
    let dense_load_seconds = dense_load_start.elapsed().as_secs_f64();
    validate_personaplex_geometry(&dense)?;
    let dense_reference = run_model(
        &mut dense,
        &prompt,
        &input_tokens,
        RealtimeSampling::greedy(),
        RunMode::Diagnostics,
    )?;
    let sampling =
        RealtimeSampling::new(0.7, 0.8, options.sampling_seed)?.with_top_k(Some(25), Some(250))?;
    let dense_run = run_model(&mut dense, &prompt, &input_tokens, sampling, RunMode::Free)?;
    drop(dense);

    let quantized_load_start = Instant::now();
    let mut quantized = load_model(&paths.quantized_model)?;
    let quantized_load_seconds = quantized_load_start.elapsed().as_secs_f64();
    validate_personaplex_geometry(&quantized)?;
    let quantized_teacher = run_model(
        &mut quantized,
        &prompt,
        &input_tokens,
        RealtimeSampling::greedy(),
        RunMode::TeacherForced(&dense_reference.frames),
    )?;
    let quality = quality_summary(&dense_reference.frames, &quantized_teacher.frames)?;
    let quantized_run = run_model(
        &mut quantized,
        &prompt,
        &input_tokens,
        sampling,
        RunMode::Free,
    )?;

    let decode_start = Instant::now();
    let dense_pcm = decode_tokens(mimi, &dense_run.emitted_audio, context, input_pcm.len())?;
    let quantized_pcm =
        decode_tokens(mimi, &quantized_run.emitted_audio, context, input_pcm.len())?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let dense_tail = tail_max_rms_dbfs(&dense_pcm);
    let quantized_tail = tail_max_rms_dbfs(&quantized_pcm);
    let swap = options.sampling_seed & 1 == 1;
    let (sample_a, sample_b, label_a, label_b, tail_a, tail_b) = if swap {
        (
            &quantized_pcm,
            &dense_pcm,
            "quantized",
            "dense",
            quantized_tail,
            dense_tail,
        )
    } else {
        (
            &dense_pcm,
            &quantized_pcm,
            "dense",
            "quantized",
            dense_tail,
            quantized_tail,
        )
    };
    let truncated_a = tail_a > ACTIVE_AUDIO_DBFS;
    let truncated_b = tail_b > ACTIVE_AUDIO_DBFS;
    if truncated_a || truncated_b {
        eprintln!(
            "warning: generated speech is active at the output boundary; sample_a_tail_dbfs={tail_a:.1} sample_b_tail_dbfs={tail_b:.1}"
        );
    }
    let dense_performance = performance_summary(&dense_run.latencies_ms);
    let quantized_performance = performance_summary(&quantized_run.latencies_ms);
    let divergence = free_run_agreement(&dense_run.frames, &quantized_run.frames);

    fs::create_dir(&paths.output)?;
    write_wav_pcm16(&paths.output.join("input.wav"), input_pcm, SAMPLE_RATE)?;
    write_wav_pcm16(
        &paths.output.join("input_codec_roundtrip.wav"),
        &streaming_roundtrip,
        SAMPLE_RATE,
    )?;
    write_wav_pcm16(
        &paths.output.join("input_codec_roundtrip_offline.wav"),
        &offline_roundtrip,
        SAMPLE_RATE,
    )?;
    write_wav_pcm16(&paths.output.join("sample_a.wav"), sample_a, SAMPLE_RATE)?;
    write_wav_pcm16(&paths.output.join("sample_b.wav"), sample_b, SAMPLE_RATE)?;

    let metrics = json!({
        "format_version": 1,
        "methodology": "Both models use the public backend-neutral realtime scheduler, forcing, sampling, and observation contracts.",
        "input": {
            "path": paths.input,
            "sample_rate": SAMPLE_RATE,
            "frame_rate": FRAME_RATE,
            "frames": frames,
            "audio_seconds": frames as f64 / FRAME_RATE,
            "tail_max_rms_dbfs": input_tail,
            "likely_truncated": input_likely_truncated,
            "warning": input_warning,
        },
        "conditioning": {
            "voice_prompt_path": paths.voice_prompt,
            "voice_prompt_frames": prompt.voice_frames.len(),
            "text_tokenizer_path": paths.text_tokenizer,
            "text_prompt": options.text_prompt,
            "wrapped_text_prompt": wrapped_text_prompt,
            "text_prompt_tokens": prompt.text_tokens.len(),
            "silence_frames_after_voice": PROMPT_SILENCE_FRAMES,
            "silence_frames_after_text": PROMPT_SILENCE_FRAMES,
        },
        "codec_diagnostic": {
            "streaming_roundtrip": "input_codec_roundtrip.wav",
            "offline_roundtrip": "input_codec_roundtrip_offline.wav",
            "streaming_offline_token_agreement": codec_agreement,
        },
        "performance": {
            "frame_deadline_ms": DEADLINE_MS,
            "codec_encode_seconds": encode_seconds,
            "codec_decode_both_outputs_seconds": decode_seconds,
            "dense": { "load_seconds": dense_load_seconds, "model": dense_performance },
            "quantized": { "load_seconds": quantized_load_seconds, "model": quantized_performance },
        },
        "teacher_forced_quality": quality,
        "free_run_divergence_diagnostic": divergence,
        "listening_test": {
            "input": "input.wav",
            "sample_a": "sample_a.wav",
            "sample_b": "sample_b.wav",
            "sampling": {
                "seed": options.sampling_seed,
                "text_temperature": sampling.text_temperature(),
                "audio_temperature": sampling.audio_temperature(),
                "text_top_k": sampling.text_top_k(),
                "audio_top_k": sampling.audio_top_k(),
            },
            "sample_a_tail_max_rms_dbfs": tail_a,
            "sample_b_tail_max_rms_dbfs": tail_b,
            "sample_a_likely_truncated": truncated_a,
            "sample_b_likely_truncated": truncated_b,
            "input_warning": input_warning,
        },
    });
    fs::write(
        paths.output.join("metrics.json"),
        serde_json::to_vec_pretty(&metrics)?,
    )?;
    fs::write(
        paths.output.join("answer_key.json"),
        serde_json::to_vec_pretty(&json!({ "sample_a": label_a, "sample_b": label_b }))?,
    )?;
    fs::write(
        paths.output.join("listening_manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "format_version": 1,
            "trials": [{
                "id": "personaplex_quantization_001",
                "input": "input.wav",
                "codec_roundtrip": "input_codec_roundtrip.wav",
                "sample_a": "sample_a.wav",
                "sample_b": "sample_b.wav",
                "input_warning": input_warning,
                "sample_a_likely_truncated": truncated_a,
                "sample_b_likely_truncated": truncated_b,
            }],
        }))?,
    )?;
    fs::write(
        paths.output.join("token_diagnostics.json"),
        serde_json::to_vec_pretty(&json!({
            "input": input_tokens,
            "input_offline": offline_tokens,
            "conditioning": {
                "voice_prompt": prompt.voice_frames,
                "text_prompt": prompt.text_tokens,
                "silence_frames_after_voice": PROMPT_SILENCE_FRAMES,
                "silence_frames_after_text": PROMPT_SILENCE_FRAMES,
            },
            "sampling": {
                "seed": options.sampling_seed,
                "text_temperature": sampling.text_temperature(),
                "audio_temperature": sampling.audio_temperature(),
                "text_top_k": sampling.text_top_k(),
                "audio_top_k": sampling.audio_top_k(),
            },
            "dense_emitted": dense_run.emitted_audio,
            "dense_sampled_frames": reference_tokens(&dense_run.frames),
            "dense_greedy_emitted": dense_reference.emitted_audio,
            "dense_greedy_frames": reference_tokens(&dense_reference.frames),
            "quantized_emitted": quantized_run.emitted_audio,
        }))?,
    )?;
    Ok(())
}

fn validate_personaplex_geometry<B: RealtimeBackend>(
    model: &RealtimeModel<B>,
) -> Result<(), Box<dyn Error>> {
    let config = model.speech_config();
    if config.input_audio_codebooks() != AUDIO_TOKENS_PER_STREAM
        || config.generated_audio_codebooks() != AUDIO_TOKENS_PER_STREAM
    {
        return Err(invalid(format!(
            "PersonaPlex evaluation requires {AUDIO_TOKENS_PER_STREAM} input and generated codebooks, got {} and {}",
            config.input_audio_codebooks(),
            config.generated_audio_codebooks()
        )));
    }
    Ok(())
}

struct PromptConditioning {
    voice_frames: Vec<Vec<i32>>,
    text_tokens: Vec<i32>,
}

enum RunMode<'a> {
    Free,
    Diagnostics,
    TeacherForced(&'a [ReferenceFrame]),
}

struct ReferenceFrame {
    text_token: i32,
    decision_audio: Vec<i32>,
    sampled_audio: Vec<i32>,
    diagnostics: Vec<Vec<f32>>,
}

struct ModelRun {
    frames: Vec<ReferenceFrame>,
    emitted_audio: Vec<Vec<i32>>,
    latencies_ms: Vec<f64>,
}

fn run_model<B: RealtimeBackend>(
    model: &mut RealtimeModel<B>,
    prompt: &PromptConditioning,
    input_tokens: &[Vec<i32>],
    sampling: RealtimeSampling,
    mode: RunMode<'_>,
) -> Result<ModelRun, Box<dyn Error>> {
    let request = RequestId::new(0);
    let mut scheduler = RealtimeScheduler::new(model, SchedulerLimits::new(1, 1)?)?;
    scheduler.register_request(model, request, sampling)?;
    for frame in prompt_frames(prompt) {
        run_frame(model, &mut scheduler, request, frame)?;
    }
    let mut frames = Vec::with_capacity(input_tokens.len());
    let mut emitted_audio = Vec::new();
    let mut latencies_ms = Vec::with_capacity(input_tokens.len());
    for (index, tokens) in input_tokens.iter().enumerate() {
        let mut frame = RealtimeInputFrame::new(1, tokens.clone());
        match mode {
            RunMode::Free => {}
            RunMode::Diagnostics => frame = frame.with_diagnostics(),
            RunMode::TeacherForced(reference) => {
                let reference = reference
                    .get(index)
                    .ok_or_else(|| invalid("teacher-forced reference is shorter than input"))?;
                frame = frame
                    .with_forced_text(vec![reference.text_token])
                    .with_forced_generated_audio(reference.sampled_audio.clone())
                    .with_diagnostics();
            }
        }
        let start = Instant::now();
        let output = run_frame(model, &mut scheduler, request, frame)?;
        latencies_ms.push(start.elapsed().as_secs_f64() * 1_000.0);
        if let Some(tokens) = output.output_audio_tokens() {
            emitted_audio.push(tokens.to_vec());
        }
        frames.push(ReferenceFrame {
            text_token: *output
                .text_tokens()
                .first()
                .ok_or_else(|| invalid("realtime output has no text token"))?,
            decision_audio: output.decision_audio_tokens().to_vec(),
            sampled_audio: output.sampled_audio_tokens().to_vec(),
            diagnostics: output
                .diagnostics()
                .iter()
                .map(|diagnostic| diagnostic.logits().to_vec())
                .collect(),
        });
    }
    scheduler.finish_request(request)?;
    Ok(ModelRun {
        frames,
        emitted_audio,
        latencies_ms,
    })
}

fn run_frame<B: RealtimeBackend>(
    model: &mut RealtimeModel<B>,
    scheduler: &mut RealtimeScheduler<B>,
    request: RequestId,
    frame: RealtimeInputFrame,
) -> Result<RealtimeOutputFrame, Box<dyn Error>> {
    let input = model.backend().materialize_input(model.model(), &frame)?;
    scheduler.enqueue(model, request, input)?;
    loop {
        if let Some(completed) = scheduler.run_queued(model)?.pop() {
            return Ok(model.backend().observe_output(completed.output())?);
        }
        std::thread::yield_now();
    }
}

fn prompt_frames(prompt: &PromptConditioning) -> Vec<RealtimeInputFrame> {
    let forced = |audio: Vec<i32>, text: i32| {
        RealtimeInputFrame::new(1, SINE_TOKENS.to_vec())
            .with_forced_generated_audio(audio)
            .with_forced_text(vec![text])
    };
    let mut frames = prompt
        .voice_frames
        .iter()
        .cloned()
        .map(|audio| forced(audio, TEXT_PADDING_TOKEN))
        .collect::<Vec<_>>();
    frames.extend(
        std::iter::repeat_with(|| forced(SILENCE_TOKENS.to_vec(), TEXT_PADDING_TOKEN))
            .take(PROMPT_SILENCE_FRAMES),
    );
    frames.extend(
        prompt
            .text_tokens
            .iter()
            .map(|token| forced(SILENCE_TOKENS.to_vec(), *token)),
    );
    frames.extend(
        std::iter::repeat_with(|| forced(SILENCE_TOKENS.to_vec(), TEXT_PADDING_TOKEN))
            .take(PROMPT_SILENCE_FRAMES),
    );
    frames
}

fn encode_pcm<T: Tensor>(
    mimi: &mut Mimi<T>,
    pcm: &[f32],
    context: &T::Context,
) -> Result<Vec<Vec<i32>>, Box<dyn Error>> {
    mimi.reset_encode_state();
    let mut frames = Vec::with_capacity(pcm.len() / FRAME_SAMPLES);
    for frame in pcm.chunks_exact(FRAME_SAMPLES) {
        let frame = T::from_f32_slice(frame, &[1, 1, FRAME_SAMPLES as i32], context)?;
        if let Some(tokens) = mimi.encode_step(&frame, context)? {
            frames.push(tokens.to_i32_vec(context)?);
        }
    }
    Ok(frames)
}

fn decode_tokens<T: Tensor>(
    mimi: &mut Mimi<T>,
    frames: &[Vec<i32>],
    context: &T::Context,
    target_samples: usize,
) -> Result<Vec<f32>, Box<dyn Error>> {
    mimi.reset_decode_state();
    let mut pcm = Vec::with_capacity(target_samples);
    for frame in frames {
        let tokens = T::from_i32_slice(frame, &[1, frame.len() as i32], context)?;
        pcm.extend(mimi.decode_step(&tokens, context)?.to_f32_vec(context)?);
    }
    pcm.truncate(target_samples);
    pcm.resize(target_samples, 0.0);
    Ok(pcm)
}

fn offline_roundtrip<T: Tensor>(
    mimi: &mut Mimi<T>,
    pcm: &[f32],
    context: &T::Context,
) -> Result<OfflineRoundtrip, Box<dyn Error>> {
    let input = T::from_f32_slice(pcm, &[1, 1, pcm.len() as i32], context)?;
    let codes = mimi.encode(&input, context)?;
    let code_shape = codes.shape().to_vec();
    if code_shape.len() != 3 || code_shape[0] != 1 {
        return Err(invalid(format!(
            "offline Mimi codes have unexpected shape {code_shape:?}"
        )));
    }
    let values = codes.to_i32_vec(context)?;
    let codebooks = code_shape[1] as usize;
    let frame_count = code_shape[2] as usize;
    let mut frames = vec![vec![0; codebooks]; frame_count];
    for codebook in 0..codebooks {
        for frame in 0..frame_count {
            frames[frame][codebook] = values[codebook * frame_count + frame];
        }
    }
    let mut roundtrip = mimi.decode(&codes, context)?.to_f32_vec(context)?;
    roundtrip.truncate(pcm.len());
    roundtrip.resize(pcm.len(), 0.0);
    Ok(OfflineRoundtrip {
        tokens: frames,
        pcm: roundtrip,
    })
}

struct OfflineRoundtrip {
    tokens: Vec<Vec<i32>>,
    pcm: Vec<f32>,
}

#[derive(Debug, Clone, Default)]
struct DistributionAccumulator {
    count: usize,
    target_count: usize,
    kl_sum: f64,
    entropy_sum: f64,
    target_nll_delta_sum: f64,
    centered_rmse_sum: f64,
    top1_matches: usize,
    top5_overlap_sum: f64,
}

impl DistributionAccumulator {
    fn update(
        &mut self,
        dense: &[f32],
        candidate: &[f32],
        target: usize,
    ) -> Result<(), Box<dyn Error>> {
        let metrics = compare_distributions(
            dense,
            candidate,
            (target < dense.len()).then_some(target),
            5,
        )?;
        self.count += 1;
        self.kl_sum += metrics.kl_nats;
        self.entropy_sum += metrics.reference_entropy_nats;
        self.centered_rmse_sum += metrics.centered_logit_rmse;
        self.top1_matches += usize::from(metrics.top1_agreement);
        self.top5_overlap_sum += metrics.top_k_overlap;
        if let Some(delta) = metrics.target_nll_delta_nats {
            self.target_count += 1;
            self.target_nll_delta_sum += delta;
        }
        Ok(())
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
        self.target_count += other.target_count;
        self.kl_sum += other.kl_sum;
        self.entropy_sum += other.entropy_sum;
        self.target_nll_delta_sum += other.target_nll_delta_sum;
        self.centered_rmse_sum += other.centered_rmse_sum;
        self.top1_matches += other.top1_matches;
        self.top5_overlap_sum += other.top5_overlap_sum;
    }

    fn summary(&self) -> MetricSummary {
        let count = self.count.max(1) as f64;
        MetricSummary {
            distributions: self.count,
            target_distributions: self.target_count,
            mean_kl_nats: self.kl_sum / count,
            mean_dense_entropy_nats: self.entropy_sum / count,
            mean_target_nll_delta_nats: self.target_nll_delta_sum / self.target_count.max(1) as f64,
            mean_centered_logit_rmse: self.centered_rmse_sum / count,
            top1_agreement: self.top1_matches as f64 / count,
            mean_top5_overlap: self.top5_overlap_sum / count,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct MetricSummary {
    distributions: usize,
    target_distributions: usize,
    mean_kl_nats: f64,
    mean_dense_entropy_nats: f64,
    mean_target_nll_delta_nats: f64,
    mean_centered_logit_rmse: f64,
    top1_agreement: f64,
    mean_top5_overlap: f64,
}

#[derive(Debug, Clone, Serialize)]
struct QualitySummary {
    methodology: &'static str,
    text: MetricSummary,
    audio_generated: MetricSummary,
    audio_input_conditioned: MetricSummary,
    audio_overall: MetricSummary,
    audio_by_codebook: Vec<MetricSummary>,
}

fn quality_summary(
    dense: &[ReferenceFrame],
    candidate: &[ReferenceFrame],
) -> Result<QualitySummary, Box<dyn Error>> {
    if dense.len() != candidate.len() {
        return Err(invalid("teacher-forced run lengths differ"));
    }
    let mut text = DistributionAccumulator::default();
    let mut audio = Vec::<DistributionAccumulator>::new();
    for (dense, candidate) in dense.iter().zip(candidate) {
        if dense.diagnostics.len() != candidate.diagnostics.len() || dense.diagnostics.is_empty() {
            return Err(invalid(
                "teacher-forced diagnostic counts differ or are empty",
            ));
        }
        text.update(
            &dense.diagnostics[0],
            &candidate.diagnostics[0],
            dense.text_token as usize,
        )?;
        if audio.is_empty() {
            audio.resize(
                dense.diagnostics.len() - 1,
                DistributionAccumulator::default(),
            );
        }
        for (codebook, accumulator) in audio.iter_mut().enumerate() {
            accumulator.update(
                &dense.diagnostics[codebook + 1],
                &candidate.diagnostics[codebook + 1],
                *dense
                    .decision_audio
                    .get(codebook)
                    .ok_or_else(|| invalid("teacher-forced decision token is missing"))?
                    as usize,
            )?;
        }
    }
    let mut overall = DistributionAccumulator::default();
    for value in &audio {
        overall.merge(value);
    }
    let mut generated = DistributionAccumulator::default();
    for value in audio.iter().take(AUDIO_TOKENS_PER_STREAM) {
        generated.merge(value);
    }
    let mut input_conditioned = DistributionAccumulator::default();
    for value in audio.iter().skip(AUDIO_TOKENS_PER_STREAM) {
        input_conditioned.merge(value);
    }
    Ok(QualitySummary {
        methodology: "The candidate is teacher-forced onto the dense model's exact text and generated-audio history; KL uses the dense distribution as reference.",
        text: text.summary(),
        audio_generated: generated.summary(),
        audio_input_conditioned: input_conditioned.summary(),
        audio_overall: overall.summary(),
        audio_by_codebook: audio.iter().map(DistributionAccumulator::summary).collect(),
    })
}

#[derive(Debug, Clone, Serialize)]
struct PerformanceSummary {
    frames: usize,
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    deadline_misses: usize,
}

fn performance_summary(latencies: &[f64]) -> PerformanceSummary {
    let summary = summarize_latencies(latencies, Some(DEADLINE_MS))
        .expect("every model run records at least one finite nonnegative latency");
    PerformanceSummary {
        frames: summary.samples,
        mean_ms: summary.mean_ms,
        p50_ms: summary.p50_ms,
        p95_ms: summary.p95_ms,
        max_ms: summary.max_ms,
        deadline_misses: summary.deadline_misses,
    }
}

fn free_run_agreement(dense: &[ReferenceFrame], quantized: &[ReferenceFrame]) -> serde_json::Value {
    let frames = dense.len().min(quantized.len());
    let text_matches = dense
        .iter()
        .zip(quantized)
        .filter(|(left, right)| left.text_token == right.text_token)
        .count();
    let mut audio_matches = 0usize;
    let mut audio_total = 0usize;
    for (left, right) in dense.iter().zip(quantized) {
        for (left, right) in left.sampled_audio.iter().zip(&right.sampled_audio) {
            audio_matches += usize::from(left == right);
            audio_total += 1;
        }
    }
    json!({
        "frames": frames,
        "text_token_agreement": text_matches as f64 / frames.max(1) as f64,
        "audio_token_agreement": audio_matches as f64 / audio_total.max(1) as f64,
    })
}

fn reference_tokens(frames: &[ReferenceFrame]) -> Vec<serde_json::Value> {
    frames
        .iter()
        .map(|frame| json!({ "text": frame.text_token, "sampled_audio": frame.sampled_audio }))
        .collect()
}

fn token_frame_agreement(left: &[Vec<i32>], right: &[Vec<i32>]) -> f64 {
    let mut matches = 0usize;
    let mut total = 0usize;
    for (left, right) in left.iter().zip(right) {
        for (left, right) in left.iter().zip(right) {
            matches += usize::from(left == right);
            total += 1;
        }
    }
    matches as f64 / total.max(1) as f64
}

fn read_f32le(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid(format!(
            "raw f32le input length must be divisible by four, got {} bytes",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect())
}

fn rms_dbfs(samples: &[f32]) -> f64 {
    let mean_square = samples
        .iter()
        .map(|sample| (*sample as f64) * (*sample as f64))
        .sum::<f64>()
        / samples.len().max(1) as f64;
    20.0 * mean_square.sqrt().max(1e-12).log10()
}

fn tail_max_rms_dbfs(samples: &[f32]) -> f64 {
    samples
        .chunks_exact(FRAME_SAMPLES)
        .rev()
        .take(TAIL_ACTIVITY_FRAMES)
        .map(rms_dbfs)
        .fold(f64::NEG_INFINITY, f64::max)
}

fn write_wav_pcm16(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), Box<dyn Error>> {
    let data_bytes = u32::try_from(
        samples
            .len()
            .checked_mul(2)
            .ok_or_else(|| invalid("WAV size overflow"))?,
    )?;
    let mut file = fs::File::create(path)?;
    file.write_all(b"RIFF")?;
    file.write_all(&(36u32 + data_bytes).to_le_bytes())?;
    file.write_all(b"WAVEfmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_bytes.to_le_bytes())?;
    for sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Box<dyn Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distribution_metrics_are_exact() {
        let values = [0.0, 1.0, -1.0, 0.5, 0.25];
        let mut metric = DistributionAccumulator::default();
        metric.update(&values, &values, 1).unwrap();
        let summary = metric.summary();
        assert!(summary.mean_kl_nats.abs() < 1e-12);
        assert_eq!(summary.top1_agreement, 1.0);
        assert_eq!(summary.mean_top5_overlap, 1.0);
    }

    #[test]
    fn prompt_frames_preserve_released_conditioning_order() {
        let prompt = PromptConditioning {
            voice_frames: vec![vec![1; AUDIO_TOKENS_PER_STREAM]],
            text_tokens: vec![7, 8],
        };
        let frames = prompt_frames(&prompt);
        assert_eq!(
            frames.len(),
            1 + PROMPT_SILENCE_FRAMES + 2 + PROMPT_SILENCE_FRAMES
        );
        assert_eq!(frames[0].forced_generated_audio_tokens(), Some(&[1; 8][..]));
        assert_eq!(
            frames[1 + PROMPT_SILENCE_FRAMES].forced_text_tokens(),
            Some(&[7][..])
        );
    }
}
