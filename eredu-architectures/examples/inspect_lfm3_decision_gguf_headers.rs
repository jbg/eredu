//! Validate the released d1 GGUF headers without opening weights.

use eredu_architectures::lfm3::decision::DecisionGgufLayouts;
use eredu_gguf::{Checkpoint, CheckpointHeader, Limits};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{error::Error, path::Path};

fn checkpoint(directory: &Path, record: &Value) -> Result<Checkpoint, Box<dyn Error>> {
    let member = record["file"].as_str().ok_or("missing GGUF filename")?;
    let bytes = std::fs::read(directory.join(format!("{member}.header")))?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if record["header_sha256"].as_str() != Some(&digest) {
        return Err(format!("header digest differs for {member}").into());
    }
    Ok(Checkpoint::from_headers(
        &[CheckpointHeader {
            member: member.into(),
            bytes,
            file_len: record["bytes"].as_u64().ok_or("missing published size")?,
        }],
        Limits::default(),
    )?)
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let manifest = args.next().ok_or("expected pinned manifest path")?;
    let directory = args.next().ok_or("expected downloaded header directory")?;
    let vl = args.next().ok_or("expected pinned d1-2B asset directory")?;
    let omni = args
        .next()
        .ok_or("expected pinned d1 Omni asset directory")?;
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    for (family, assets) in [("vl", vl), ("omni", omni)] {
        let files = manifest[family]["files"]
            .as_array()
            .ok_or("missing files")?;
        let (projectors, backbones): (Vec<_>, Vec<_>) = files.iter().partition(|record| {
            record["file"]
                .as_str()
                .is_some_and(|name| name.starts_with("mmproj-"))
        });
        let directory = Path::new(&directory);
        let projectors = projectors
            .into_iter()
            .map(|record| Ok::<_, Box<dyn Error>>((record, checkpoint(directory, record)?)))
            .collect::<Result<Vec<_>, _>>()?;
        for record in backbones {
            let backbone = checkpoint(directory, record)?;
            for (projector_record, projector) in &projectors {
                let layouts =
                    DecisionGgufLayouts::inspect(&backbone, Some(projector), Path::new(&assets))?;
                let projector_layout = layouts.projector.ok_or("projector layout was omitted")?;
                println!(
                    "{} + {}: {} backbone and {} projector sources",
                    record["file"],
                    projector_record["file"],
                    layouts.backbone.source_keys().len(),
                    projector_layout.source_keys().len(),
                );
            }
        }
    }
    Ok(())
}
