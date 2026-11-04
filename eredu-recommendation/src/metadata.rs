use crate::{transport::Hub, ModelVariant};
use anyhow::{ensure, Result};
use eredu::api::{
    ArtifactMetadata, CheckpointMetadata, GgufCompanionMetadata, GgufHeader, MetadataProvenance,
    SafetensorsHeader,
};
use eredu_core::artifact::ArtifactCompanionRole;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
const HEADER_LIMIT: usize = 16 * 1024 * 1024;

pub(crate) async fn fetch_metadata(
    hub: &Hub,
    repo: &str,
    revision: &str,
    variant: &ModelVariant,
    manifest: &Value,
) -> Result<Result<ArtifactMetadata, Vec<String>>> {
    let names: BTreeSet<&str> = manifest["siblings"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Missing model file listing"))?
        .iter()
        .filter_map(|file| file["rfilename"].as_str())
        .collect();
    let mut sidecars = BTreeMap::new();
    let mut total_header_bytes = 0;
    let checkpoint = if variant.format == "safetensors" {
        let config = hub.file(repo, revision, "config.json").await?;
        let json = serde_json::from_slice(&config)?;
        if let Err(eredu_core::artifact::ArtifactError::UnsupportedModelType(model_type)) =
            eredu_architectures::configuration::resolve_model_identity(&json)
        {
            return Ok(Err(vec![format!(
                "Architecture {model_type} is unsupported by eredu."
            )]));
        }
        let index = if names.contains("model.safetensors.index.json") {
            Some(
                hub.file(repo, revision, "model.safetensors.index.json")
                    .await?,
            )
        } else {
            None
        };
        let members: BTreeSet<String> = match &index {
            Some(index) => serde_json::from_slice::<Value>(index)?["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("Invalid SafeTensors index"))?
                .values()
                .map(|name| {
                    name.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| anyhow::anyhow!("Invalid shard name"))
                })
                .collect::<Result<_>>()?,
            None => names
                .iter()
                .filter(|name| name.ends_with(".safetensors") && !name.contains('/'))
                .map(|name| (*name).into())
                .collect(),
        };
        ensure!(
            !members.is_empty() && members.len() <= 256,
            "Incomplete or oversized SafeTensors shard set"
        );
        let mut headers = Vec::new();
        for member in members {
            ensure!(
                names.contains(member.as_str()),
                "Missing indexed SafeTensors shard"
            );
            let (prefix, file_len) = hub.range(repo, revision, &member, 0, 7).await?;
            let length = u64::from_le_bytes(
                prefix
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("Invalid SafeTensors prefix"))?,
            );
            ensure!(
                length > 0 && length <= HEADER_LIMIT as u64 - 8 && length + 8 <= file_len,
                "Invalid or oversized SafeTensors header"
            );
            let (bytes, checked_len) = hub.range(repo, revision, &member, 0, length + 7).await?;
            ensure!(
                file_len == checked_len,
                "Model length changed during inspection"
            );
            total_header_bytes += bytes.len();
            ensure!(
                total_header_bytes <= 64 * 1024 * 1024,
                "Model headers exceed inspection budget"
            );
            headers.push(SafetensorsHeader {
                member,
                bytes,
                file_len,
            });
        }
        for name in [
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "processor_config.json",
        ] {
            if names.contains(name) {
                sidecars.insert(name.into(), hub.file(repo, revision, name).await?);
            }
        }
        CheckpointMetadata::SafeTensors {
            config,
            index,
            headers,
        }
    } else {
        let (members, projector) = crate::variants::gguf_members(variant)?;
        ensure!(
            !members.is_empty() && members.len() <= 256,
            "Incomplete or oversized GGUF shard set"
        );
        let mut headers = Vec::new();
        for member in members {
            let prefix = hub.range(repo, revision, &member, 0, 65535).await?;
            if headers.is_empty() {
                if let Some(architecture) = crate::gguf::prefix_architecture(&prefix.0)? {
                    if eredu_architectures::configuration::GgufArchitecture::resolve(&architecture)
                        .is_err()
                    {
                        return Ok(Err(vec![format!(
                            "GGUF architecture {architecture} is unsupported by eredu."
                        )]));
                    }
                }
            }
            let header = gguf_header_from_prefix(hub, repo, revision, &member, prefix).await?;
            if headers.is_empty() {
                if let Some(architecture) = crate::gguf::header_architecture(&header.bytes)? {
                    if eredu_architectures::configuration::GgufArchitecture::resolve(&architecture)
                        .is_err()
                    {
                        return Ok(Err(vec![format!(
                            "GGUF architecture {architecture} is unsupported by eredu."
                        )]));
                    }
                }
            }
            total_header_bytes += header.bytes.len();
            ensure!(
                total_header_bytes <= 64 * 1024 * 1024,
                "Model headers exceed inspection budget"
            );
            headers.push(header);
        }
        let companions = match projector {
            Some(member) => vec![GgufCompanionMetadata {
                role: ArtifactCompanionRole::MediaProjector,
                headers: vec![gguf_header(hub, repo, revision, &member).await?],
            }],
            None => Vec::new(),
        };
        CheckpointMetadata::Gguf {
            headers,
            companions,
        }
    };
    Ok(Ok(ArtifactMetadata {
        provenance: MetadataProvenance {
            source: format!("{repo}:{}", variant.variant_id),
            revision: revision.into(),
        },
        checkpoint,
        sidecars,
    }))
}

async fn gguf_header(hub: &Hub, repo: &str, revision: &str, member: &str) -> Result<GgufHeader> {
    let prefix = hub.range(repo, revision, member, 0, 65535).await?;
    gguf_header_from_prefix(hub, repo, revision, member, prefix).await
}

async fn gguf_header_from_prefix(
    hub: &Hub,
    repo: &str,
    revision: &str,
    member: &str,
    prefix: (Vec<u8>, u64),
) -> Result<GgufHeader> {
    let (mut bytes, file_len) = prefix;
    loop {
        match crate::gguf::header_end(&bytes)? {
            Ok(end) => {
                bytes.truncate(end);
                return Ok(GgufHeader {
                    member: member.into(),
                    bytes,
                    file_len,
                });
            }
            Err(required) => {
                let end = (required.max(bytes.len() * 2).min(HEADER_LIMIT) as u64).min(file_len);
                ensure!(
                    end > bytes.len() as u64,
                    "Incomplete or oversized GGUF header"
                );
                let (next, checked_len) = hub
                    .range(repo, revision, member, bytes.len() as u64, end - 1)
                    .await?;
                ensure!(
                    file_len == checked_len,
                    "Model length changed during inspection"
                );
                bytes.extend_from_slice(&next);
            }
        }
    }
}
