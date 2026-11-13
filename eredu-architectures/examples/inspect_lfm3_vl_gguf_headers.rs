//! Checks pinned VL backbone and projector headers and optional bounded projector reads.

use eredu_architectures::lfm3::vision::{
    resolve_gguf_projector, Declaration, PreparedVisionExecution, ProjectorConfig,
};
use eredu_architectures::{
    configuration::MODEL_CONFIGURATIONS, replicated_text::lfm3_vl_requirements,
};
use eredu_checkpoint::{
    recipe::RecipeDtype,
    store::{CheckpointLease, EncodedTensorLease, ReadPolicy, TensorReadRequest, TensorSelection},
};
use eredu_core::artifact::{
    inspect_artifact_metadata, ArtifactCompanionRole, ArtifactMetadata, CheckpointMetadata,
    GgufCompanionMetadata, MetadataProvenance,
};
use eredu_gguf::{Checkpoint, CheckpointHeader, ConvertedTensor, Limits};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, error::Error, io::Read, path::Path};

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let manifest = args
        .next()
        .ok_or("expected the pinned projector manifest")?;
    let directory = args.next().ok_or("expected the header directory")?;
    let config_path = args.next().ok_or("expected the pinned VL config.json")?;
    let payload = args.next();
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest)?)?;
    let config_bytes = std::fs::read(&config_path)?;
    let config: Value = serde_json::from_slice(&config_bytes)?;
    let declaration = Declaration::new(
        serde_json::from_value(config["vision_config"].clone())?,
        ProjectorConfig::from_config_value(&config)?,
        "model.vision_tower",
        "model.multi_modal_projector",
    )?;
    let projectors = manifest["projectors"]
        .as_object()
        .ok_or("missing projector manifest entries")?;
    for (variant, record) in projectors {
        let checkpoint = Checkpoint::from_headers(
            &[read_header(Path::new(&directory), record)?],
            Limits::default(),
        )?;
        resolve_gguf_projector(&checkpoint, &declaration)?;
        println!(
            "{variant}: {} projector tensors",
            checkpoint.tensors().count()
        );
    }
    let config_dir = Path::new(&config_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let sidecars = BTreeMap::from([
        ("config.json".into(), config_bytes),
        (
            "processor_config.json".into(),
            std::fs::read(config_dir.join("processor_config.json"))?,
        ),
    ]);
    let projector = read_header(Path::new(&directory), &projectors["F16"])?;
    let backbones = manifest["backbones"]
        .as_object()
        .ok_or("missing backbone manifest entries")?;
    for (variant, record) in backbones {
        let header = read_header(Path::new(&directory), record)?;
        let inspection = inspect_artifact_metadata(
            &ArtifactMetadata {
                provenance: MetadataProvenance {
                    source: header.member.clone(),
                    revision: manifest["revision"]
                        .as_str()
                        .ok_or("missing revision")?
                        .into(),
                },
                checkpoint: CheckpointMetadata::Gguf {
                    headers: vec![header],
                    companions: vec![GgufCompanionMetadata {
                        role: ArtifactCompanionRole::MediaProjector,
                        headers: vec![projector.clone()],
                    }],
                },
                sidecars: sidecars.clone(),
            },
            &MODEL_CONFIGURATIONS,
        );
        let requirements = lfm3_vl_requirements(&inspection?)?;
        println!(
            "{variant}: {} paired graph bindings",
            requirements.execution.parameters().len()
        );
    }
    if let Some(payload) = payload {
        let payload = Path::new(&payload);
        let record = projectors
            .values()
            .find(|record| record["file"].as_str().map(std::ffi::OsStr::new) == payload.file_name())
            .ok_or("payload filename differs from the pinned release")?;
        verify_payload(payload, record, declaration)?;
    }
    Ok(())
}

fn read_header(directory: &Path, record: &Value) -> Result<CheckpointHeader, Box<dyn Error>> {
    let member = record["file"]
        .as_str()
        .ok_or("missing checkpoint filename")?;
    let bytes = std::fs::read(directory.join(format!("{member}.header")))?;
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if Some(bytes.len() as u64) != record["header_bytes"].as_u64()
        || Some(digest.as_str()) != record["header_sha256"].as_str()
    {
        return Err(format!("header digest or length differs for {member}").into());
    }
    Ok(CheckpointHeader {
        member: member.into(),
        bytes,
        file_len: record["bytes"]
            .as_u64()
            .ok_or("missing checkpoint file length")?,
    })
}

fn verify_payload(
    path: &Path,
    record: &Value,
    declaration: Declaration,
) -> Result<(), Box<dyn Error>> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    let digest: String = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    if record["publisher_sha256"].as_str() != Some(digest.as_str()) {
        return Err("projector payload hash differs from the pinned release".into());
    }
    let prepared = PreparedVisionExecution::new_gguf(declaration, path, RecipeDtype::F32, 1)?;
    let (_, source, _, bindings) = prepared.into_parts();
    let keys = source.source_keys();
    for key in &keys {
        let lease = source.acquire_lease(TensorReadRequest {
            key: key.clone(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })?;
        let CheckpointLease::Gguf(lease) = lease else {
            return Err("expected a bounded GGUF lease".into());
        };
        if !lease.bounded_read_proof().physically_bounded {
            return Err(format!("unbounded projector payload read for {key}").into());
        }
        let expected = lease
            .output_shape()
            .iter()
            .map(|&dimension| dimension as u64)
            .collect::<Vec<_>>();
        let shape = match lease.materialize_portable()?.into_converted() {
            ConvertedTensor::Dense(tensor) => tensor.shape,
            ConvertedTensor::IQuant(tensor) => tensor.packed_shape()?,
            _ => return Err(format!("unexpected projector encoding for {key}").into()),
        };
        if shape != expected {
            return Err(format!("projector storage shape differs for {key}").into());
        }
    }
    println!(
        "verified {} bounded payloads and {} canonical bindings across {} vision units",
        keys.len(),
        bindings.iter().map(Vec::len).sum::<usize>(),
        bindings.len(),
    );
    Ok(())
}
