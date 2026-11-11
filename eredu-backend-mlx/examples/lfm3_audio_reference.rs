//! Complete resident LFM3 Audio conditional reference case on widened source weights.

use clap::Parser;
use eredu_architectures::lfm3::{
    audio::{
        AudioPrompt, BindingSource, CompletedSpeechFrame, DepthDecoder, Encoder, Feedback,
        PreparedAudio, Projector,
    },
    CompleteLfm3Kernel, LayeredModel,
};
use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::MlxHybridState,
            checkpoint::binding::{
                build_neutral_module_bindings_with_recipes, materialize_module_bindings,
                populate_module_from_arrays_excluding,
            },
        },
        ExecutionContext,
    },
    native::MlxAudioModel,
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::TensorSelection,
};
use eredu_media::audio::{AudioWaveform, LogMelFeatures};
use eredu_nn::{Index, Parameterized, Tensor};
use eredu_runtime::ResidentRuntime;
use safemlx::{Array, Device, DeviceType};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    error::Error,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    checkpoint: PathBuf,
    /// Reference directory, or the pinned source manifest in GGUF mode.
    reference: PathBuf,
    output: PathBuf,
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "gpu"])]
    device: String,
    /// Compare the released LiveASR block protocol.
    #[arg(long)]
    live_asr: bool,
    /// Feed the LiveASR fixture as microphone-sized PCM chunks.
    #[arg(long, requires = "live_asr")]
    live_asr_pcm: bool,
    /// Compare restored streamed prompts with finite resident inference.
    #[arg(long, conflicts_with = "live_asr")]
    ingress_replay: bool,
    /// Released GGUF language weights.
    #[arg(long, requires = "gguf_projector")]
    gguf_backbone: Option<PathBuf>,
    /// Released GGUF speech input, depth and codec decoder weights.
    #[arg(long, requires = "gguf_backbone")]
    gguf_projector: Option<PathBuf>,
    /// Raw F32 score rows from the pinned llama.cpp Audio reference.
    #[arg(long, requires = "gguf_backbone", conflicts_with_all = ["ingress_replay", "live_asr"])]
    gguf_scores: Option<PathBuf>,
}

fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let mut source = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn bind<M: Parameterized<MlxTensor>>(
    module: &mut M,
    weights: &BindingSource,
    cpu: &<MlxTensor as Tensor>::Context,
    ctx: &<MlxTensor as Tensor>::Context,
) -> Result<(), Box<dyn Error>> {
    let source = weights.source();
    let mut recipes = weights.recipes().clone();
    for key in source.source_keys() {
        recipes
            .entry(key.clone())
            .or_insert_with(|| DerivedWeightRecipe::source(key, TensorSelection::Full));
    }
    for recipe in recipes.values_mut() {
        *recipe = DerivedWeightRecipe::Cast {
            input: Box::new(recipe.clone()),
            dtype: RecipeDtype::F32,
        };
    }
    let bindings =
        build_neutral_module_bindings_with_recipes(module, source.as_ref(), &mut recipes)?;
    let arrays = materialize_module_bindings(source.as_ref(), &bindings, cpu, ctx)?;
    populate_module_from_arrays_excluding(module, &arrays, |_| false)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if let (Some(backbone), Some(projector)) = (&args.gguf_backbone, &args.gguf_projector) {
        use eredu_architectures::realtime::{
            prepare_realtime_model, PreparedRealtimeArtifact, RealtimeArtifactRequest,
        };
        let manifest: Value = serde_json::from_slice(&std::fs::read(&args.reference)?)?;
        for (role, path) in [("backbone", backbone), ("projector", projector)] {
            let digest = sha256(path)?;
            if manifest["variants"]["Q4_K_M"][role]["publisher_sha256"] != digest {
                return Err(
                    format!("GGUF {role} hash differs from the pinned Q4_K_M release").into(),
                );
            }
        }
        let PreparedRealtimeArtifact::Lfm3Audio(artifact) = prepare_realtime_model(
            &RealtimeArtifactRequest::audio_gguf(backbone, projector, &args.checkpoint),
        )?
        else {
            return Err("expected an LFM3 Audio GGUF artifact".into());
        };
        let native = execution_context(&args.device);
        if let Some(scores) = &args.gguf_scores {
            return compare_gguf_scores(&args, &artifact, native, scores);
        }
        if !args.ingress_replay {
            return Err("GGUF acceptance requires --gguf-scores or --ingress-replay".into());
        }
        return compare_ingress_replay(&args, *artifact, native);
    }
    let reference_name = if args.live_asr {
        "live-reference.json"
    } else {
        "reference.json"
    };
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(args.reference.join(reference_name))?)?;
    let model_hash = sha256(&args.checkpoint.join("model.safetensors"))?;
    let codec_hash = sha256(&args.checkpoint.join("audio_tokenizer/model.safetensors"))?;
    if metadata["model_sha256"] != model_hash || metadata["codec_sha256"] != codec_hash {
        return Err("native weights differ from the reference artifacts".into());
    }
    let model = PreparedAudio::open(&args.checkpoint, 2)?;
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let native = execution_context(&args.device);
    if args.live_asr {
        return compare_live_asr(&args, &metadata, model, native);
    }
    if args.ingress_replay {
        return compare_ingress_replay(&args, model, native);
    }
    let tolerance = metadata["maximum_absolute_tolerance"]
        .as_f64()
        .ok_or("missing tolerance")?;
    let ctx = native.stream();
    let reference =
        Array::load_safetensors(args.reference.join("reference.safetensors"), cpu.stream())?;
    let prompt: Vec<i32> = serde_json::from_value(metadata["prompt"].clone())?;
    let decode: Vec<i32> = serde_json::from_value(metadata["decode"].clone())?;
    let mut maxima = BTreeMap::new();
    let mut greedy_rows = 0;
    let mut compare = |key: &str, native: &MlxTensor| -> Result<(), Box<dyn Error>> {
        let expected = &reference[key];
        assert_eq!(native.shape(), expected.shape());
        let actual = native.as_array().evaluated()?.try_to_vec::<f32>()?;
        let expected = expected.evaluated()?.try_to_vec::<f32>()?;
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| {
                assert!(a.is_finite() && b.is_finite());
                (a - b).abs()
            })
            .fold(0f32, f32::max);
        if key.ends_with(".logits") {
            for (a, b) in actual
                .chunks_exact(128000)
                .zip(expected.chunks_exact(128000))
            {
                let top = |x: &[f32]| {
                    x.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                };
                assert_eq!(top(a), top(b));
                greedy_rows += 1;
            }
        }
        println!("{key} max={max:.8}");
        maxima.insert(key.to_owned(), max);
        if f64::from(max) > tolerance {
            return Err("comparison threshold exceeded".into());
        }
        Ok(())
    };
    let mut encoder = Encoder::<MlxNeuralBackend>::new(&model.encoder, ctx)?;
    let mut projector = Projector::<MlxNeuralBackend>::new(&model.encoder, ctx)?;
    let mut feedback = Feedback::<MlxNeuralBackend>::new(&model.depth, ctx)?;
    bind(&mut encoder, &model.encoder_weights, cpu.stream(), ctx)?;
    bind(&mut projector, &model.encoder_weights, cpu.stream(), ctx)?;
    bind(&mut feedback, &model.depth_weights, cpu.stream(), ctx)?;
    let pcm = reference["pcm"].evaluated()?.try_to_vec::<f32>()?;
    let features = model
        .processor
        .transform()?
        .extract(AudioWaveform::new(&pcm, 16000)?)?;
    let features =
        MlxTensor::from_f32_slice(&features.values, &[1, features.frames as i32, 128], ctx)?
            .transpose_axes(&[0, 2, 1], ctx)?;
    compare("features", &features)?;
    let mut enc_state = MlxHybridState::device(model.encoder.state_layout().clone(), &[])?;
    let hidden = encoder.forward(&features, &mut enc_state, ctx)?;
    let projected = projector.forward(&hidden, ctx)?.squeeze_axes(&[0], ctx)?;
    let codes = MlxTensor::from_array(reference["codes"].copy(ctx)?);
    let replay = feedback.embed(&codes, ctx)?;
    let eos = feedback.embed(
        &MlxTensor::full_i32(model.depth.eos_code(), &[1, 12], ctx)?,
        ctx,
    )?;
    let (mut runtime, mut state) = bind_language(&model, cpu.stream(), ctx)?;
    let width = model.language.args().hidden_size;
    let hidden = AudioPrompt::new(
        &prompt,
        model.tokens,
        width,
        Some(&projected),
        Some(&replay),
        ctx,
    )?
    .forward_resident(&mut runtime, &mut state, ctx)?;
    compare("prefill.hidden", &hidden)?;
    let logits = runtime
        .architecture_mut()
        .kernel_mut()
        .static_modules_mut()
        .project_normalized(&hidden, ctx)?;
    compare("prefill.logits", &logits)?;
    let mut last = hidden;
    for (index, token) in decode.into_iter().enumerate() {
        let replacement = if token == model.tokens.output {
            Some(if index == 1 { &replay } else { &eos })
        } else {
            None
        };
        last = AudioPrompt::new(&[token], model.tokens, width, None, replacement, ctx)?
            .forward_resident(&mut runtime, &mut state, ctx)?;
        compare(&format!("decode.{index}.hidden"), &last)?;
        let logits = runtime
            .architecture_mut()
            .kernel_mut()
            .static_modules_mut()
            .project_normalized(&last, ctx)?;
        compare(&format!("decode.{index}.logits"), &logits)?;
    }
    let mut depth = DepthDecoder::<MlxNeuralBackend>::new(&model.depth, ctx)?;
    bind(&mut depth, &model.depth_weights, cpu.stream(), ctx)?;
    let order: Vec<usize> = serde_json::from_value(metadata["order"].clone())?;
    let mut frame = model.depth.begin_frame(&order)?;
    let last = last.index(&[Index::Full, Index::At(0)], ctx)?;
    while let Some((index, slots)) = frame.next_pass() {
        let logits = depth.score_frame(&last, &frame, ctx)?;
        compare(&format!("depth.{index}"), &logits)?;
        let sampled = slots
            .iter()
            .map(|&slot| {
                let row = logits.index(
                    &[
                        Index::At(0),
                        Index::At(slot as i32),
                        Index::Range(0, model.depth.eos_code()),
                    ],
                    ctx,
                )?;
                Ok(row.ranked_topk(1, ctx)?.1.to_i32_vec(ctx)?[0])
            })
            .collect::<Result<Vec<_>, eredu_nn::Error>>()?;
        frame.reveal(index, &sampled)?;
    }
    let Some(CompletedSpeechFrame::Audio(generated)) = frame.completed() else {
        return Err("expected speech frame".into());
    };
    assert_eq!(
        generated,
        reference["generated"].evaluated()?.try_to_vec::<i32>()?
    );
    let generated_ids = generated.to_vec();
    let generated = MlxTensor::from_i32_slice(generated, &[1, 12], ctx)?;
    let mut codec = eredu_codec::liquid::NeuralCodec::<MlxNeuralBackend>::new(&model.codec, ctx)?;
    bind(&mut codec, &model.codec_weights, cpu.stream(), ctx)?;
    let mut codec_state = MlxHybridState::device(model.codec.decoder_state_layout().clone(), &[])?;
    let pcm = codec.decode(&generated.expand_dims(-1, ctx)?, &mut codec_state, ctx)?;
    compare("speech.pcm", &pcm)?;
    let next = feedback.embed(&generated, ctx)?;
    let hidden = AudioPrompt::new(
        &[model.tokens.output],
        model.tokens,
        width,
        None,
        Some(&next),
        ctx,
    )?
    .forward_resident(&mut runtime, &mut state, ctx)?;
    compare("speech.hidden", &hidden)?;
    let logits = runtime
        .architecture_mut()
        .kernel_mut()
        .static_modules_mut()
        .project_normalized(&hidden, ctx)?;
    compare("speech.logits", &logits)?;
    std::fs::write(
        args.output,
        serde_json::to_vec_pretty(&json!({
            "device": args.device,
            "model_sha256": model_hash,
            "codec_sha256": codec_hash,
            "maximum_absolute_tolerance": tolerance,
            "maximum_absolute_errors": maxima,
            "greedy_rows_matched": greedy_rows,
            "speech_code_ids": generated_ids,
            "passed": true,
        }))?,
    )?;
    Ok(())
}

fn execution_context(device: &str) -> ExecutionContext {
    let device = if device == "gpu" {
        DeviceType::Gpu
    } else {
        DeviceType::Cpu
    };
    ExecutionContext::new(Device::new(device, 0))
}

#[derive(serde::Deserialize)]
struct LiveBlock {
    frames: usize,
    #[serde(rename = "final")]
    final_chunk: bool,
    restart: bool,
    audio_tokens: usize,
    closed: bool,
    tokens: Vec<i32>,
}

fn compare_live_asr(
    args: &Args,
    metadata: &Value,
    model: PreparedAudio,
    native: ExecutionContext,
) -> Result<(), Box<dyn Error>> {
    let bytes = std::fs::read(args.reference.join("live-pcm.f32"))?;
    if bytes.len() % 4 != 0 {
        return Err("invalid LiveASR PCM payload".into());
    }
    let pcm: Vec<_> = bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("four bytes")))
        .collect();
    let blocks: Vec<LiveBlock> = serde_json::from_value(metadata["blocks"].clone())?;
    let (matched_blocks, token_count) = if args.live_asr_pcm {
        compare_live_pcm(&pcm, &blocks, MlxAudioModel::load(model, native)?)?
    } else {
        compare_live_mel(&pcm, &blocks, model, native)?
    };
    let report = json!({
        "device": args.device,
        "input": if args.live_asr_pcm { "pcm" } else { "mel" },
        "wav_sha256": metadata["wav_sha256"],
        "matched_blocks": matched_blocks,
        "matched_tokens": token_count,
    });
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}

fn compare_live_mel(
    pcm: &[f32],
    blocks: &[LiveBlock],
    model: PreparedAudio,
    native: ExecutionContext,
) -> Result<(usize, usize), Box<dyn Error>> {
    let features = model
        .processor
        .transform()?
        .extract(AudioWaveform::new(pcm, 16_000)?)?;
    let mut session = MlxAudioModel::load(model, native)?.prepare_live_transcription()?;
    session.enable_snapshots(eredu_core::execution_control::SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 1 << 28,
        cumulative_copy_bytes: 1 << 28,
    })?;
    let mut cursor = 0;
    let mut token_count = 0;
    for (index, block) in blocks.iter().enumerate() {
        let end = cursor + block.frames;
        let features = LogMelFeatures {
            values: features.values[cursor * features.mel_bins..end * features.mel_bins].to_vec(),
            mask: features.mask[cursor..end].to_vec(),
            frames: block.frames,
            mel_bins: features.mel_bins,
        };
        let saved = (index == 1).then(|| session.snapshot()).transpose()?;
        let child = (index == 1).then(|| session.fork()).transpose()?;
        let result = session.advance(&features, block.final_chunk, block.restart)?;
        compare_live_block(index, &result, block);
        if let (Some(saved), Some(mut child)) = (saved, child) {
            let before = session.snapshot_usage().ok_or("missing fork usage")?;
            assert_eq!(child.snapshot_usage(), Some(before));
            assert!(child.fork().is_err());
            assert!(child.restore(&saved).is_err());
            assert_eq!(child.snapshot_usage(), Some(before));
            assert_eq!(
                child.advance(&features, block.final_chunk, block.restart)?,
                result
            );
            let after = child.snapshot_usage().ok_or("missing child growth usage")?;
            assert!(after.retained_bytes > before.retained_bytes);
            assert_eq!(after.cumulative_copy_bytes, before.cumulative_copy_bytes);
            drop(child);
            assert_eq!(
                session
                    .snapshot_usage()
                    .ok_or("missing released fork usage")?
                    .branches,
                0
            );
            session.restore(&saved)?;
            assert_eq!(
                session.advance(&features, block.final_chunk, block.restart)?,
                result
            );
        }
        token_count += result.tokens.len();
        cursor = end;
    }
    assert_eq!(cursor, features.frames);
    Ok((blocks.len(), token_count))
}

fn compare_live_pcm(
    pcm: &[f32],
    blocks: &[LiveBlock],
    model: MlxAudioModel,
) -> Result<(usize, usize), Box<dyn Error>> {
    use eredu_architectures::lfm3::audio::AudioTranscriptionError;
    use eredu_core::execution_control::SnapshotLimits;
    let mut session = model.prepare_pcm_transcription()?;
    session.enable_snapshots(SnapshotLimits {
        max_snapshots: 1,
        max_branches: 1,
        retained_bytes: 1 << 28,
        cumulative_copy_bytes: 1 << 28,
    })?;
    let capacity = session.available_samples();
    assert!(matches!(
        session.submit(&vec![0.0; capacity + 1], 16_000),
        Err(AudioTranscriptionError::Backpressure { available, .. }) if available == capacity
    ));
    assert_eq!(session.available_samples(), capacity);
    assert!(session.advance(false)?.is_none());
    let mut expected = blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.audio_tokens != 0 || block.final_chunk)
        .peekable();
    let mut matched = 0;
    let mut tokens = 0;
    let mut samples = 0;
    let mut saved = None;
    for chunk in pcm.chunks(1600) {
        session.submit(chunk, 16_000)?;
        samples += chunk.len();
        while let Some(&(index, block)) = expected.peek() {
            if block.final_chunk {
                break;
            }
            let Some(result) = session.advance(block.restart)? else {
                break;
            };
            compare_live_block(index, &result, block);
            expected.next();
            matched += 1;
            tokens += result.tokens.len();
            if matched == 1 {
                saved = Some((samples, session.available_samples(), session.snapshot()?));
                let usage = session.snapshot_usage();
                assert!(session.snapshot().is_err());
                assert_eq!(session.snapshot_usage(), usage);
            } else if let Some((start, capacity, snapshot)) = saved.take() {
                let copy_bytes = session
                    .snapshot_usage()
                    .ok_or("missing snapshot usage")?
                    .cumulative_copy_bytes;
                session.restore(&snapshot)?;
                assert_eq!(session.available_samples(), capacity);
                for chunk in pcm[start..samples].chunks(1600) {
                    session.submit(chunk, 16_000)?;
                }
                assert_eq!(session.advance(block.restart)?, Some(result));
                assert!(
                    session
                        .snapshot_usage()
                        .ok_or("missing restore usage")?
                        .cumulative_copy_bytes
                        > copy_bytes
                );
                drop(snapshot);
                assert_eq!(
                    session
                        .snapshot_usage()
                        .ok_or("missing snapshot usage")?
                        .retained_bytes,
                    0
                );
            }
        }
    }
    assert!(matched > 1 && saved.is_none());
    let before_finish = session.snapshot()?;
    let mut child = session.fork()?;
    assert_eq!(child.available_samples(), session.available_samples());
    session.finish_input()?;
    assert!(matches!(
        session.submit(&[], 16_000),
        Err(AudioTranscriptionError::InputFinished)
    ));
    let pending = expected.collect::<Vec<_>>();
    for &(index, block) in &pending {
        let result = session
            .advance(block.restart)?
            .ok_or("missing flushed LiveASR block")?;
        compare_live_block(index, &result, block);
        matched += 1;
        tokens += result.tokens.len();
    }
    assert!(session.is_finished());
    assert!(!child.is_finished());
    child.finish_input()?;
    for &(index, block) in &pending {
        let result = child
            .advance(block.restart)?
            .ok_or("missing child LiveASR block")?;
        compare_live_block(index, &result, block);
    }
    assert!(child.is_finished());
    drop(child);
    assert!(matches!(
        session.advance(false),
        Err(AudioTranscriptionError::Finished)
    ));
    session.restore(&before_finish)?;
    assert!(!session.is_finished());
    session.finish_input()?;
    for (index, block) in pending {
        let result = session
            .advance(block.restart)?
            .ok_or("missing replayed LiveASR block")?;
        compare_live_block(index, &result, block);
    }
    assert!(session.is_finished());
    Ok((matched, tokens))
}

fn compare_live_block(
    index: usize,
    result: &eredu_architectures::lfm3::audio::AudioTranscriptionBlock,
    block: &LiveBlock,
) {
    assert_eq!(result.audio_tokens, block.audio_tokens, "block {index}");
    assert_eq!(result.closed, block.closed, "block {index}");
    assert_eq!(result.tokens, block.tokens, "block {index}");
}

fn audio_fixture() -> Vec<f32> {
    (0..2151)
        .map(|sample| {
            let time = sample as f32 / 16_000.0;
            0.12 * (std::f32::consts::TAU * 413.0 * time).sin()
                + 0.05 * (std::f32::consts::TAU * 863.0 * time).sin()
        })
        .collect()
}

type LanguageRuntime =
    ResidentRuntime<LayeredModel<MlxNeuralBackend>, MlxNeuralBackend, MlxHybridState>;

fn bind_language(
    model: &PreparedAudio,
    cpu: &<MlxTensor as Tensor>::Context,
    ctx: &<MlxTensor as Tensor>::Context,
) -> Result<(LanguageRuntime, MlxHybridState), Box<dyn Error>> {
    let architecture = CompleteLfm3Kernel::<MlxNeuralBackend>::from_selected_declaration(
        model.language.clone(),
        model.language.definition().clone(),
        None,
        ctx,
    )?;
    let layout = architecture.definition().state_layout().clone();
    let mut runtime = ResidentRuntime::new(architecture, ctx)?;
    bind(
        runtime.architecture_mut().kernel_mut().static_modules_mut(),
        &model.language_weights,
        cpu,
        ctx,
    )?;
    for unit in runtime.units_mut().iter_mut().flatten() {
        bind(unit, &model.language_weights, cpu, ctx)?;
    }
    let state = MlxHybridState::device(layout, &[])?;
    Ok((runtime, state))
}

fn compare_gguf_scores(
    args: &Args,
    model: &PreparedAudio,
    native: ExecutionContext,
    scores: &Path,
) -> Result<(), Box<dyn Error>> {
    use eredu_core::{ObservationSet, ObservationValue, TensorObservation, TensorObservationData};
    use eredu_evaluation::{
        compare_observations, CheckpointParityOptions, ParityComparison, ParityPolicy,
    };

    let vocabulary = usize::try_from(model.language.args().vocab_size)?;
    let bytes = std::fs::read(scores)?;
    if bytes.len() != 9 * vocabulary * 4 {
        return Err("expected nine full-vocabulary F32 score rows".into());
    }
    let expected = bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes(value.try_into().expect("one F32 value")))
        .collect();
    let cpu = ExecutionContext::cpu()?;
    let ctx = native.stream();
    let mut encoder = Encoder::<MlxNeuralBackend>::new(&model.encoder, ctx)?;
    let mut projector = Projector::<MlxNeuralBackend>::new(&model.encoder, ctx)?;
    let mut feedback = Feedback::<MlxNeuralBackend>::new(&model.depth, ctx)?;
    bind(&mut encoder, &model.encoder_weights, cpu.stream(), ctx)?;
    bind(&mut projector, &model.encoder_weights, cpu.stream(), ctx)?;
    bind(&mut feedback, &model.depth_weights, cpu.stream(), ctx)?;
    let pcm = audio_fixture();
    let features = model
        .processor
        .transform()?
        .extract(AudioWaveform::new(&pcm, 16000)?)?;
    let features =
        MlxTensor::from_f32_slice(&features.values, &[1, features.frames as i32, 128], ctx)?
            .transpose_axes(&[0, 2, 1], ctx)?;
    let mut encoder_state = MlxHybridState::device(model.encoder.state_layout().clone(), &[])?;
    let input = projector
        .forward(&encoder.forward(&features, &mut encoder_state, ctx)?, ctx)?
        .squeeze_axes(&[0], ctx)?;
    let codes = (0..12).map(|book| 149 * book).collect::<Vec<_>>();
    let codes = MlxTensor::from_i32_slice(&codes, &[1, 12], ctx)?;
    let output = feedback.embed(&codes, ctx)?;
    let (mut runtime, mut state) = bind_language(model, cpu.stream(), ctx)?;
    let width = model.language.args().hidden_size;
    let mut rows = Vec::with_capacity(9 * vocabulary);
    let mut advance = |ids: &[i32],
                       input: Option<&MlxTensor>,
                       output: Option<&MlxTensor>|
     -> Result<(), Box<dyn Error>> {
        let hidden = AudioPrompt::new(ids, model.tokens, width, input, output, ctx)?
            .forward_resident(&mut runtime, &mut state, ctx)?;
        let last = hidden.index(&[Index::Full, Index::At(-1)], ctx)?;
        let logits = runtime
            .architecture_mut()
            .kernel_mut()
            .static_modules_mut()
            .project_normalized(&last, ctx)?;
        rows.extend(logits.as_array().evaluated()?.try_to_vec::<f32>()?);
        Ok(())
    };
    let speech = model.tokens.input;
    advance(&[124894, 5, speech, speech, 13, 21, 34], Some(&input), None)?;
    advance(&[model.tokens.output], None, Some(&output))?;
    for token in [5, 8, 13, 21, 34, 55, 3] {
        advance(std::slice::from_ref(&token), None, None)?;
    }
    let observe = |values| -> Result<ObservationSet, Box<dyn Error>> {
        let mut set = ObservationSet::new();
        set.insert(
            "logits",
            ObservationValue::Tensor(TensorObservation::new(
                vec![9, vocabulary],
                TensorObservationData::F32(values),
            )?),
        )?;
        Ok(set)
    };
    let tolerance = CheckpointParityOptions::default().logits;
    let policy = ParityPolicy {
        default: ParityComparison::Logits { tolerance },
        rules: vec![],
        require_same_paths: true,
    };
    let report = compare_observations(&observe(rows)?, &observe(expected)?, &policy)?;
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    if !report.passed {
        return Err("mixed Audio GGUF score rows exceed the shared logit tolerance".into());
    }
    Ok(())
}

fn compare_ingress_replay(
    args: &Args,
    artifact: PreparedAudio,
    execution: ExecutionContext,
) -> Result<(), Box<dyn Error>> {
    use eredu_architectures::lfm3::audio::{
        AudioAdvance, AudioIngressRequest, AudioMode, AudioRequest, AudioSnapshotError,
        PreparedResidentAudio,
    };
    use eredu_core::execution_control::SnapshotLimits;

    let pcm = audio_fixture();
    let prefix = [124894, 5];
    let suffix = [13, 21, 34];
    let finite = {
        let features = artifact
            .processor
            .transform()?
            .extract(AudioWaveform::new(&pcm, 16_000)?)?;
        let mut tokens = prefix.to_vec();
        tokens.extend([artifact.tokens.input; 2]);
        tokens.extend(suffix);
        let model = MlxAudioModel::load_prepared(
            PreparedResidentAudio::select(artifact.clone())?,
            &ExecutionContext::cpu()?,
            &execution,
        )?;
        model
            .start(
                AudioRequest {
                    tokens: &tokens,
                    features: &[features],
                    replay: &[],
                    mode: AudioMode::Speech,
                    max_slots: 3,
                },
                41,
            )?
            .events()
            .collect::<Result<Vec<_>, _>>()?
    };

    let mut ingress = MlxAudioModel::load(artifact, execution)?.prepare_ingress(
        AudioIngressRequest {
            prefix: &prefix,
            suffix: &suffix,
            mode: AudioMode::Speech,
            max_slots: 3,
        },
        41,
    )?;
    let limits = SnapshotLimits {
        max_snapshots: 2,
        max_branches: 1,
        retained_bytes: 1 << 28,
        cumulative_copy_bytes: 1 << 28,
    };
    ingress.enable_snapshots(limits)?;
    let early = ingress.submit(&pcm[..1055], 16_000)?;
    assert_eq!((early.samples, early.audio_tokens), (1055, 0));
    let before_first = ingress.snapshot()?;
    let mut child = ingress.fork()?;
    let initial_fork = ingress.snapshot_usage().ok_or("missing fork usage")?;
    assert_eq!(child.snapshot_usage(), Some(initial_fork));
    assert!(child.fork().is_err());
    assert!(child.restore(&before_first).is_err());
    assert_eq!(child.snapshot_usage(), Some(initial_fork));
    let late = ingress.submit(&pcm[1055..1520], 16_000)?;
    assert_eq!((late.samples, late.audio_tokens), (1520, 1));
    let after_first = ingress.snapshot()?;
    let usage = ingress.snapshot_usage().ok_or("missing ingress usage")?;
    assert!(ingress.snapshot().is_err());
    assert_eq!(ingress.snapshot_usage(), Some(usage));

    let complete = ingress.submit(&pcm[1520..], 16_000)?;
    assert_eq!(child.progress(), early);
    assert_eq!(child.submit(&pcm[1055..1520], 16_000)?, late);
    assert_eq!(child.submit(&pcm[1520..], 16_000)?, complete);
    let fork_usage = ingress.snapshot_usage().ok_or("missing grown fork usage")?;
    assert!(fork_usage.retained_bytes > usage.retained_bytes);
    assert_eq!(
        fork_usage.cumulative_copy_bytes,
        usage.cumulative_copy_bytes
    );
    ingress.restore(&after_first)?;
    assert_eq!(ingress.progress(), late);
    assert_eq!(ingress.submit(&pcm[1520..], 16_000)?, complete);
    ingress.restore(&before_first)?;
    assert_eq!(ingress.progress(), early);
    assert_eq!(ingress.submit(&pcm[1055..1520], 16_000)?, late);
    assert_eq!(ingress.submit(&pcm[1520..], 16_000)?, complete);

    let restored_usage = ingress.snapshot_usage().ok_or("missing restored usage")?;
    assert_eq!(restored_usage.retained_bytes, fork_usage.retained_bytes);
    assert!(restored_usage.cumulative_copy_bytes > usage.cumulative_copy_bytes);
    let mut session = ingress.finish()?;
    assert_eq!(session.snapshot_usage(), Some(restored_usage));
    assert!(matches!(
        session.enable_snapshots(limits),
        Err(AudioSnapshotError::AlreadyConfigured)
    ));
    drop((before_first, after_first));
    let child_retained = session.snapshot_usage().ok_or("missing response usage")?;
    assert_eq!(child_retained.snapshots, 0);
    assert_eq!(child_retained.branches, 1);
    assert!(child_retained.retained_bytes > 0);
    assert_eq!(
        child_retained.cumulative_copy_bytes,
        restored_usage.cumulative_copy_bytes
    );
    let output = session.events().collect::<Result<Vec<_>, _>>()?;
    assert_eq!(output, finite);
    let mut child = child.finish()?;
    let transferred = child
        .snapshot_usage()
        .ok_or("missing transferred fork usage")?;
    assert_eq!(transferred.branches, 1);
    assert_eq!(session.snapshot_usage(), Some(transferred));
    assert_eq!(
        transferred.cumulative_copy_bytes,
        child_retained.cumulative_copy_bytes
    );
    assert!(child.fork().is_err());
    assert_eq!(child.events().collect::<Result<Vec<_>, _>>()?, output);
    drop(child);
    let released = session
        .snapshot_usage()
        .ok_or("missing released fork usage")?;
    assert_eq!(released.branches, 0);
    assert_eq!(released.retained_bytes, 0);
    assert_eq!(
        released.cumulative_copy_bytes,
        transferred.cumulative_copy_bytes
    );
    let samples: usize = output
        .iter()
        .map(|event| match event {
            AudioAdvance::Speech { pcm, .. } => pcm.len(),
            _ => 0,
        })
        .sum();
    assert_eq!((output.len(), samples), (3, 5760));
    let report = json!({ "device": args.device, "matched_frames": output.len(), "matched_samples": samples });
    std::fs::write(&args.output, serde_json::to_vec_pretty(&report)?)?;
    println!("{report}");
    Ok(())
}
