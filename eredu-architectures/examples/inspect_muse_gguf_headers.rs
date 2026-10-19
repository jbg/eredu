//! Validates pinned GGUF descriptor prefixes without opening weight payloads.

use eredu_architectures::configuration::{GgufModelConfig, MODEL_CONFIGURATIONS};
use eredu_checkpoint::validation::{validate_gguf_plan, CheckpointValidation};
use eredu_core::artifact::{
    inspect_artifact_metadata, ArtifactMetadata, CheckpointMetadata, GgufCompanionMetadata,
    MetadataProvenance,
};
use eredu_core::{ArtifactCompanionRole, ModelConfigurationResolver};
use eredu_gguf::{Checkpoint, CheckpointHeader, Limits, QuantizedTensorRepresentation};
use sha2::{Digest, Sha256};
use std::{error::Error, path::PathBuf};

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut arguments = std::env::args_os().skip(1);
    let directory = PathBuf::from(
        arguments
            .next()
            .ok_or("usage: inspect_muse_gguf_headers EVIDENCE_DIRECTORY [CATALOG_OUTPUT]")?,
    );
    let output = arguments.next().map(PathBuf::from);
    let mut evidence: serde_json::Value =
        serde_json::from_slice(&std::fs::read(directory.join("evidence.json"))?)?;
    let provenance = MetadataProvenance {
        source: evidence["repository"]
            .as_str()
            .ok_or("missing repository")?
            .into(),
        revision: evidence["revision"]
            .as_str()
            .ok_or("missing revision")?
            .into(),
    };
    let mut target = None;
    for (variant, record) in evidence["variants"]
        .as_object_mut()
        .ok_or("missing variants")?
    {
        let mut headers = Vec::new();
        for source in record["headers"].as_array().ok_or("missing headers")? {
            let member = source["file"].as_str().ok_or("missing member name")?;
            if !std::path::Path::new(member)
                .components()
                .all(|part| matches!(part, std::path::Component::Normal(_)))
            {
                return Err("invalid relative GGUF member name".into());
            }
            let bytes = std::fs::read(directory.join(format!("{member}.header.bin")))?;
            if digest(&bytes)
                != source["header_sha256"]
                    .as_str()
                    .ok_or("missing header digest")?
            {
                return Err("retained GGUF header digest mismatch".into());
            }
            headers.push(CheckpointHeader {
                member: member.to_owned(),
                bytes,
                file_len: source["bytes"]
                    .as_u64()
                    .ok_or("missing pinned file length")?,
            });
        }
        let checkpoint = Checkpoint::from_headers_with_representation(
            &headers,
            Limits::default(),
            QuantizedTensorRepresentation::Encoded,
        )?;
        record["reader_validation"] = "exact".into();
        let architecture = checkpoint
            .metadata()
            .get("general.architecture")
            .and_then(|value| value.as_str());
        if architecture == Some("muse-spark") {
            let (_, admitted) = MODEL_CONFIGURATIONS
                .resolve_gguf("muse-spark", &checkpoint)?
                .into_parts();
            let architecture = admitted
                .gguf_plan()
                .ok_or("registry omitted its GGUF plan")?;
            let GgufModelConfig::Muse(_) = architecture.model() else {
                return Err("registry selected a different construction family".into());
            };
            let validation = validate_gguf_plan(&checkpoint, architecture.checkpoint());
            if validation != CheckpointValidation::Exact {
                eprintln!("{variant}: {validation:?}");
                return Err("Muse GGUF tensor catalog does not match the exact schema".into());
            }
            record["text_schema_validation"] = "exact".into();
            println!(
                "{variant}: {} shards, {} exact text tensors",
                headers.len(),
                checkpoint.tensors().count()
            );
            target = Some(headers.clone());
        } else if architecture == Some("clip") {
            let primary = target
                .as_ref()
                .ok_or("media inspection requires a text variant in the same evidence set")?;
            let inspection = inspect_artifact_metadata(
                &ArtifactMetadata {
                    provenance: provenance.clone(),
                    checkpoint: CheckpointMetadata::Gguf {
                        headers: primary.clone(),
                        companions: vec![GgufCompanionMetadata {
                            role: ArtifactCompanionRole::MediaProjector,
                            headers: headers.clone(),
                        }],
                    },
                    sidecars: Default::default(),
                },
                &MODEL_CONFIGURATIONS,
            )?;
            let admitted = inspection
                .architecture_plan()
                .gguf_media_projector()
                .ok_or("public registry omitted its supplied media companion")?;
            let waveform = inspection
                .architecture_plan()
                .muse()
                .ok_or("public registry omitted its Muse processor")?
                .waveform();
            let has_audio = checkpoint
                .metadata()
                .get("clip.has_audio_encoder")
                .and_then(eredu_gguf::MetadataValue::as_bool)
                == Some(true);
            if waveform.is_some() != has_audio {
                return Err("public processor omitted or invented its waveform policy".into());
            }
            let validation = validate_gguf_plan(&checkpoint, admitted.checkpoint());
            if validation != CheckpointValidation::Exact {
                eprintln!("{variant}: {validation:?}");
                return Err("Muse GGUF media catalog does not match the exact schema".into());
            }
            record["media_schema_validation"] = "exact".into();
            println!(
                "{variant}: {} exact vision/audio tensors",
                checkpoint.tensors().count()
            );
        } else {
            println!(
                "{variant}: {} structurally valid tensors; no model schema result",
                checkpoint.tensors().count()
            );
        }
        let first = headers.first().ok_or("empty shard group")?;
        let first_json: serde_json::Value = serde_json::from_slice(&std::fs::read(
            directory.join(format!("{}.header.json", first.member)),
        )?)?;
        let source_metadata = first_json["metadata"]
            .as_object()
            .ok_or("missing source metadata")?;
        record["metadata"] = serde_json::Value::Object(
            source_metadata
                .iter()
                .filter(|(key, value)| {
                    !key.starts_with("tokenizer.")
                        || key.ends_with("_token_id") && !value.is_array()
                })
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        if let Some(tokens) = source_metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|value| value.as_array())
        {
            let special_tokens = ["<|audio|>", "<|video|>", "<|patch|>"]
                .into_iter()
                .filter_map(|token| {
                    tokens
                        .iter()
                        .position(|value| value.as_str() == Some(token))
                        .map(|id| (token, id))
                })
                .collect::<std::collections::BTreeMap<_, _>>();
            record["tokenizer"] = serde_json::json!({
                "tokens_count": tokens.len(),
                "tokens_json_sha256": digest(&serde_json::to_vec(tokens)?),
                "special_tokens": special_tokens,
            });
        }
    }
    if let Some(output) = output {
        std::fs::write(output, serde_json::to_vec(&evidence)?)?;
    }
    Ok(())
}
