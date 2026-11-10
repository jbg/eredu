//! Complete resident LFM3 Audio conditional reference case on widened source weights.

use clap::Parser;
use eredu_architectures::lfm3::{
    audio::{
        AudioPrompt, BindingSource, CompletedSpeechFrame, DepthDecoder, Encoder, Feedback,
        PreparedAudio, Projector,
    },
    CompleteLfm3Kernel,
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
    MlxTensor,
};
use eredu_checkpoint::{
    recipe::{DerivedWeightRecipe, RecipeDtype},
    store::TensorSelection,
};
use eredu_media::audio::AudioWaveform;
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
    reference: PathBuf,
    output: PathBuf,
    #[arg(long, default_value = "cpu", value_parser = ["cpu", "gpu"])]
    device: String,
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
    let metadata: Value =
        serde_json::from_slice(&std::fs::read(args.reference.join("reference.json"))?)?;
    let model_hash = sha256(&args.checkpoint.join("model.safetensors"))?;
    let codec_hash = sha256(&args.checkpoint.join("audio_tokenizer/model.safetensors"))?;
    if metadata["model_sha256"] != model_hash || metadata["codec_sha256"] != codec_hash {
        return Err("native weights differ from the reference artifacts".into());
    }
    let tolerance = metadata["maximum_absolute_tolerance"]
        .as_f64()
        .ok_or("missing tolerance")?;
    let model = PreparedAudio::open(&args.checkpoint, 2)?;
    let cpu = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let device = if args.device == "gpu" {
        DeviceType::Gpu
    } else {
        DeviceType::Cpu
    };
    let native = ExecutionContext::new(Device::new(device, 0));
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
        cpu.stream(),
        ctx,
    )?;
    for unit in runtime.units_mut().iter_mut().flatten() {
        bind(unit, &model.language_weights, cpu.stream(), ctx)?;
    }
    let mut state = MlxHybridState::device(layout, &[])?;
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
