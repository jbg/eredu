//! Validates the pinned Audio GGUF descriptor prefixes through portable admission.

use eredu_architectures::lfm3::audio::{AdmittedAudioGguf, Declaration, DepthDeclaration};
use eredu_checkpoint::{
    recipe::DerivedWeightRecipe,
    store::{CheckpointLease, EncodedTensorLease, ReadPolicy, TensorReadRequest},
};
use eredu_codec::liquid::{Config as CodecConfig, Declaration as CodecDeclaration};
use eredu_gguf::{Checkpoint, CheckpointHeader, ConvertedTensor, GgmlType, Limits};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{error::Error, io::Read, path::Path};

fn checkpoint(directory: &Path, record: &Value) -> Result<Checkpoint, Box<dyn Error>> {
    let member = record["file"].as_str().ok_or("missing GGUF filename")?;
    let bytes = std::fs::read(directory.join(format!("{member}.header")))?;
    let sha256: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if record["header_sha256"].as_str() != Some(sha256.as_str()) {
        return Err(format!("header digest differs for {member}").into());
    }
    let checkpoint = Checkpoint::from_headers(
        &[CheckpointHeader {
            member: member.into(),
            bytes,
            file_len: record["bytes"].as_u64().ok_or("missing published size")?,
        }],
        Limits::default(),
    )?;
    Ok(checkpoint)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let manifest = args.next().ok_or("expected pinned manifest path")?;
    let directory = args.next().ok_or("expected downloaded header directory")?;
    let config = args
        .next()
        .map(|path| -> Result<Value, Box<dyn Error>> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        })
        .transpose()?;
    let payload = args.next();
    let codec = args
        .next()
        .map(|path| -> Result<CodecDeclaration, Box<dyn Error>> {
            Ok(CodecDeclaration::new(CodecConfig::from_json(
                &std::fs::read(path)?,
            )?)?)
        })
        .transpose()?;
    if codec.is_some() && config.is_none() {
        return Err("codec verification requires the pinned Audio config".into());
    }
    let directory = Path::new(&directory);
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let variants = manifest["variants"].as_object().ok_or("missing variants")?;
    for (variant, record) in variants {
        let admitted = AdmittedAudioGguf::from_checkpoints(
            checkpoint(directory, &record["backbone"])?,
            checkpoint(directory, &record["projector"])?,
        )?;
        let (backbone, projector) = admitted.layouts();
        println!(
            "{variant}: {} backbone tensors ({} parameters), {} projector tensors",
            admitted.backbone().tensors().count(),
            backbone.source_keys().len(),
            projector.source_keys().len()
        );
        if let Some(config) = &config {
            let bindings = admitted.speech_bindings(
                &Declaration::from_config(config)?,
                &DepthDeclaration::from_config(config)?,
            )?;
            println!(
                "  {} canonical input and {} depth/feedback bindings",
                bindings.input_recipes().len(),
                bindings.depth_recipes().len(),
            );
            if let Some(codec) = &codec {
                println!(
                    "  {} canonical LiquidCodec decoder bindings",
                    bindings.codec_decoder_bindings(codec)?.recipes().len()
                );
            }
        }
    }
    AdmittedAudioGguf::from_checkpoints(
        checkpoint(directory, &variants["Q4_K_M"]["backbone"])?,
        checkpoint(directory, &variants["Q8_0"]["projector"])?,
    )?;
    println!("Q4_K_M backbone + Q8_0 projector: admitted");
    if let Some(path) = payload {
        let config = config
            .as_ref()
            .ok_or("payload verification requires the pinned Audio config")?;
        verify_projector(
            directory,
            &variants["Q4_K_M"],
            config,
            Path::new(&path),
            codec.as_ref(),
        )?;
    }
    Ok(())
}

fn verify_projector(
    directory: &Path,
    record: &Value,
    config: &Value,
    path: &Path,
    codec: Option<&CodecDeclaration>,
) -> Result<(), Box<dyn Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut bytes = [0; 65536];
    loop {
        let read = file.read(&mut bytes)?;
        if read == 0 {
            break;
        }
        hasher.update(&bytes[..read]);
    }
    let digest: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if record["projector"]["publisher_sha256"].as_str() != Some(&digest) {
        return Err("projector payload hash differs from the pinned release".into());
    }
    let admitted = AdmittedAudioGguf::from_checkpoints(
        checkpoint(directory, &record["backbone"])?,
        Checkpoint::open(path)?,
    )?;
    let bindings = admitted.speech_bindings(
        &Declaration::from_config(config)?,
        &DepthDeclaration::from_config(config)?,
    )?;
    let sources = if let Some(codec) = codec {
        let (speech, _, decoder) = bindings.retain_with_codec_decoder(codec, 1, None)?;
        let key = "decoder.istft.window";
        let window = decoder.source().acquire_lease(TensorReadRequest {
            key: key.into(),
            selection: eredu_checkpoint::store::TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })?;
        assert_eq!(window.output_shape(), [1280]);
        let values = window
            .encoded_bytes()
            .ok_or("expected synthesized window bytes")?;
        assert_eq!(values.len(), 1280 * 4);
        assert_eq!(&values[..4], &0f32.to_le_bytes());
        assert_eq!(&values[640 * 4..641 * 4], &1f32.to_le_bytes());
        assert!(speech.input.source().source_metadata(key).is_err());
        assert!(decoder
            .source()
            .source_metadata("a.gen.wav.quant.first.codebook.weight")
            .is_err());
        println!("Q4_K_M projector: verified FSQ table and retained 1280 Hann samples");
        speech
    } else {
        bindings.retain(1, None)?
    };
    let recipe =
        &sources.depth.recipes()["model.depth_decoder.depth_embeddings.11.embedding.weight"];
    let DerivedWeightRecipe::Source { key, selection } = recipe else {
        return Err("packed codebook must lower to a direct bounded selection".into());
    };
    assert!(sources.input.source().source_metadata(key).is_err());
    let lease = sources.depth.source().acquire_lease(TensorReadRequest {
        key: key.clone(),
        selection: selection.clone(),
        policy: ReadPolicy::RequireBounded,
    })?;
    let CheckpointLease::Gguf(lease) = lease else {
        return Err("expected GGUF codebook lease".into());
    };
    let proof = lease.bounded_read_proof();
    assert!(proof.physically_bounded && proof.offset_bytes != 0);
    let ConvertedTensor::IQuant(packed) = lease.materialize_portable()?.into_converted() else {
        return Err("expected the bounded GGML selection to stay packed".into());
    };
    assert_eq!(packed.shape, [2017, 1024]);
    assert!(matches!(packed.ggml_type, GgmlType::Q4K | GgmlType::Q6K));
    assert_eq!(packed.data.len() as u64, proof.length_bytes);
    println!(
        "Q4_K_M projector: retained codebook 11 in {} encoded bytes",
        proof.length_bytes
    );
    Ok(())
}
