//! Validates the pinned Audio GGUF descriptor prefixes through portable admission.

use eredu_architectures::lfm3::audio::AdmittedAudioGguf;
use eredu_gguf::{Checkpoint, CheckpointHeader, Limits, QuantizedTensorRepresentation};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{error::Error, path::Path};

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
    let checkpoint = Checkpoint::from_headers_with_representation(
        &[CheckpointHeader {
            member: member.into(),
            bytes,
            file_len: record["bytes"].as_u64().ok_or("missing published size")?,
        }],
        Limits::default(),
        QuantizedTensorRepresentation::DecodedF32,
    )?;
    Ok(checkpoint)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let manifest = args.next().ok_or("expected pinned manifest path")?;
    let directory = args.next().ok_or("expected downloaded header directory")?;
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
    }
    AdmittedAudioGguf::from_checkpoints(
        checkpoint(directory, &variants["Q4_K_M"]["backbone"])?,
        checkpoint(directory, &variants["Q8_0"]["projector"])?,
    )?;
    println!("Q4_K_M backbone + Q8_0 projector: admitted");
    Ok(())
}
