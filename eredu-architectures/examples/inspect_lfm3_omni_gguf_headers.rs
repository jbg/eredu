//! Checks pinned Omni GGUF descriptors and the canonical component projections.

use eredu_architectures::lfm3::omni::AdmittedOmniGguf;
use eredu_codec::liquid::{Config, Declaration};
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
    if record["header_sha256"].as_str() != Some(&sha256) {
        return Err(format!("header digest differs for {member}").into());
    }
    Ok(Checkpoint::from_headers_with_representation(
        &[CheckpointHeader {
            member: member.into(),
            bytes,
            file_len: record["bytes"].as_u64().ok_or("missing published size")?,
        }],
        Limits::default(),
        QuantizedTensorRepresentation::DecodedF32,
    )?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let manifest = args.next().ok_or("expected pinned manifest path")?;
    let directory = args.next().ok_or("expected downloaded header directory")?;
    let assets = args.next().ok_or("expected pinned Omni asset directory")?;
    let assets = Path::new(&assets);
    let config = serde_json::from_slice(&std::fs::read(assets.join("config.json"))?)?;
    let codec = Declaration::new(Config::from_json(&std::fs::read(
        assets.join("audio_tokenizer/config.json"),
    )?)?)?;
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let variants = manifest["variants"].as_object().ok_or("missing variants")?;
    for (variant, record) in variants {
        let directory = Path::new(&directory);
        let admitted = AdmittedOmniGguf::from_checkpoints(
            checkpoint(directory, &record["backbone"])?,
            checkpoint(directory, &record["projector"])?,
            &config,
            &codec,
        )?;
        let (backbone, projector) = admitted.layouts();
        println!(
            "{variant}: {} backbone and {} projector sources",
            backbone.source_keys().len(),
            projector.source_keys().len(),
        );
    }
    Ok(())
}
