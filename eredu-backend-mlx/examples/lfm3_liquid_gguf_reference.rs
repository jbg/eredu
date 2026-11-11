//! Compares released Audio GGUF synthesis with the pinned llama.cpp PCM runner.

use eredu_architectures::lfm3::audio::{AdmittedAudioGguf, Declaration, DepthDeclaration};
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
use eredu_checkpoint::recipe::{DerivedWeightRecipe, RecipeDtype};
use eredu_codec::liquid::{Config, Declaration as CodecDeclaration, NeuralDecoder};
use eredu_gguf::{Checkpoint, CheckpointHeader, Limits, QuantizedTensorRepresentation};
use eredu_nn::Tensor;
use safemlx::{Device, DeviceType};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{error::Error, fs, io::Read, path::PathBuf};

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let [manifest, header, projector, audio_config, codec_config, pcm_path] =
        std::env::args_os().skip(1).map(PathBuf::from).collect::<Vec<_>>().try_into()
            .map_err(|_| "expected source manifest, Audio backbone header, projector, Audio config, codec config and llama.cpp PCM")?;
    let manifest: Value = serde_json::from_slice(&fs::read(manifest)?)?;
    let header = fs::read(header)?;
    let digest = hex(&Sha256::digest(&header));
    let (variant, record) = manifest["variants"]
        .as_object()
        .ok_or("missing GGUF variants")?
        .iter()
        .find(|(_, record)| record["backbone"]["header_sha256"].as_str() == Some(digest.as_str()))
        .ok_or("Audio backbone header hash does not match a pinned publisher layout")?;
    let mut hasher = Sha256::new();
    let mut file = fs::File::open(&projector)?;
    let mut buffer = [0; 65536];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    if record["projector"]["publisher_sha256"].as_str() != Some(&hex(&hasher.finalize())) {
        return Err("Audio projector payload hash differs".into());
    }
    let representation = QuantizedTensorRepresentation::DecodedF32;
    let backbone = Checkpoint::from_headers_with_representation(
        &[CheckpointHeader {
            member: record["backbone"]["file"]
                .as_str()
                .ok_or("missing filename")?
                .into(),
            bytes: header,
            file_len: record["backbone"]["bytes"]
                .as_u64()
                .ok_or("missing file size")?,
        }],
        Limits::default(),
        representation,
    )?;
    let projector =
        Checkpoint::open_with_representation(&projector, Limits::default(), representation)?;
    let admitted = AdmittedAudioGguf::from_checkpoints(backbone, projector)?;
    let audio: Value = serde_json::from_slice(&fs::read(audio_config)?)?;
    let codec = CodecDeclaration::new(Config::from_json(&fs::read(codec_config)?)?)?;
    let (_, decoder) = admitted
        .speech_bindings(
            &Declaration::from_config(&audio)?,
            &DepthDeclaration::from_config(&audio)?,
        )?
        .retain_with_codec_decoder(&codec, 2)?;

    let context = ExecutionContext::new(Device::new(DeviceType::Cpu, 0));
    let stream = context.stream();
    let mut model = NeuralDecoder::<MlxNeuralBackend>::new(&codec, stream)?;
    let mut recipes = decoder.recipes().clone();
    for recipe in recipes.values_mut() {
        *recipe = DerivedWeightRecipe::Cast {
            input: Box::new(recipe.clone()),
            dtype: RecipeDtype::F32,
        };
    }
    let bindings = build_neutral_module_bindings_with_recipes(
        &model,
        decoder.source().as_ref(),
        &mut recipes,
    )?;
    let arrays = materialize_module_bindings(decoder.source().as_ref(), &bindings, stream, stream)?;
    populate_module_from_arrays_excluding(&mut model, &arrays, |_| false)?;

    let neural = MlxHybridState::device(codec.decoder_state_layout().clone(), &[])?;
    let mut state = model.decoder_state(neural, 1)?;
    let mut actual = Vec::new();
    let mut counts = Vec::new();
    let mut start = 0;
    for count in [1, 3, 6] {
        let codes = (0..12)
            .flat_map(|book| {
                (start..start + count)
                    .map(move |frame| (73 * frame + 149 * book + 31 * frame * book) % 2016)
            })
            .collect::<Vec<_>>();
        let codes = MlxTensor::from_i32_slice(&codes, &[1, 12, count], stream)?;
        let pcm = model.decode_chunk(Some(&codes), &mut state, false, stream)?;
        counts.push(pcm.dim(2));
        actual.extend(pcm.as_array().evaluated()?.try_to_vec::<f32>()?);
        start += count;
    }
    let pcm = model.decode_chunk(None, &mut state, true, stream)?;
    counts.push(pcm.dim(2));
    actual.extend(pcm.as_array().evaluated()?.try_to_vec::<f32>()?);
    let reference = fs::read(pcm_path)?;
    if counts != [1440, 5760, 11520, 480] || reference.len() != actual.len() * 4 {
        return Err(format!("unexpected PCM lengths {counts:?} and {}", reference.len()).into());
    }
    let mut maximum = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut squared_reference = 0.0f64;
    for (&actual, expected) in actual.iter().zip(reference.chunks_exact(4)) {
        let expected = f32::from_le_bytes(expected.try_into()?);
        if !actual.is_finite() || !expected.is_finite() {
            return Err("nonfinite PCM".into());
        }
        let delta = f64::from(actual) - f64::from(expected);
        maximum = maximum.max(delta.abs());
        squared_error += delta * delta;
        squared_reference += f64::from(expected).powi(2);
    }
    let relative = (squared_error / squared_reference).sqrt();
    println!("variant={variant} streamed_samples={counts:?} max_abs={maximum:.8} relative_l2={relative:.8}");
    if maximum > 0.05 || relative > 0.08 {
        return Err(format!("{variant} LiquidCodec PCM exceeds the reference tolerance").into());
    }
    Ok(())
}
