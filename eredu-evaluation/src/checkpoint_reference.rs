//! Retained publisher payload evidence shared by released-checkpoint comparisons.
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    path::{Component, Path},
};

/// SHA-256 identity for retained comparison inputs and outputs.
pub fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Identity of the exact retained-payload validator compiled into a runner.
pub fn validator_sha256() -> String {
    sha256(include_bytes!("checkpoint_reference.rs"))
}

fn pinned_artifact<'a>(
    inventory: &'a Value,
    repository: &str,
    revision: &str,
) -> Result<&'a Value, Box<dyn Error>> {
    inventory["artifacts"]
        .as_array()
        .ok_or("missing frozen inventory")?
        .iter()
        .find(|item| item["repository"] == repository && item["revision"] == revision)
        .ok_or_else(|| "checkpoint is absent from the pinned inventory".into())
}

/// Verifies the retrieved configuration, tokenizer and processor metadata pinned
/// by an inventory. Returns exact file identities without opening weight payloads.
pub fn verify_metadata(
    checkpoint: &Path,
    inventory: &Value,
    repository: &str,
    revision: &str,
) -> Result<BTreeMap<String, String>, Box<dyn Error>> {
    let pinned = pinned_artifact(inventory, repository, revision)?;
    let metadata = pinned["retrieved_metadata"]
        .as_object()
        .ok_or("missing pinned metadata")?;
    let mut identities = BTreeMap::new();
    for (name, record) in metadata {
        if name.is_empty()
            || !Path::new(name)
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
        {
            return Err("metadata names must remain inside the checkpoint".into());
        }
        let digest = sha256(&std::fs::read(checkpoint.join(name))?);
        if record["sha256"] != digest {
            return Err(format!("pinned metadata digest differs for {name}").into());
        }
        identities.insert(name.clone(), digest);
    }
    Ok(identities)
}

/// Matches retained full-payload verification to the pinned inventory and
/// reference, and rejects files changed since that verification. This does not
/// rehash the checkpoint or replace the source store's lifetime guards.
pub fn verify_payload(
    checkpoint: &Path,
    reference: &Value,
    inventory: &Value,
    repository: &str,
    revision: &str,
) -> Result<String, Box<dyn Error>> {
    let proof_path = checkpoint.join("payload-evidence.json");
    let proof_bytes = std::fs::read(&proof_path)?;
    let proof: Value = serde_json::from_slice(&proof_bytes)?;
    let proof_hash = sha256(&proof_bytes);
    if proof["repository"] != repository
        || proof["revision"] != revision
        || proof["all_weight_payload_sha256_verified"] != true
        || reference["repository"] != repository
        || reference["revision"] != revision
        || reference["payload_proof_sha256"] != proof_hash
    {
        return Err(
            "comparison requires matching complete pinned payload and reference evidence".into(),
        );
    }
    let pinned = pinned_artifact(inventory, repository, revision)?;
    let expected = pinned["files"]
        .as_object()
        .ok_or("missing publisher file hashes")?;
    let expected_names: BTreeSet<_> = expected
        .keys()
        .filter(|name| name.ends_with(".safetensors"))
        .map(String::as_str)
        .collect();
    let verified_at = std::fs::metadata(&proof_path)?.modified()?;
    let mut names = BTreeSet::new();
    for file in proof["files"]
        .as_array()
        .ok_or("missing verified payload files")?
    {
        let name = file["file"].as_str().ok_or("missing payload member name")?;
        if Path::new(name).file_name().and_then(|p| p.to_str()) != Some(name)
            || !names.insert(name)
            || !expected_names.contains(name)
            || file["verified"] != true
            || file["sha256"] != expected[name]["sha256"]
            || file["bytes"] != expected[name]["bytes"]
        {
            return Err("payload evidence differs from the frozen publisher inventory".into());
        }
        let metadata = std::fs::metadata(checkpoint.join(name))?;
        if Some(metadata.len()) != file["bytes"].as_u64() || metadata.modified()? > verified_at {
            return Err("verified payload changed after verification".into());
        }
    }
    if expected_names.is_empty() || names != expected_names {
        return Err(
            "complete publisher payload membership differs from retained verification".into(),
        );
    }
    Ok(proof_hash)
}
