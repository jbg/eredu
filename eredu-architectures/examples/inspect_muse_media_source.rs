//! Header-only admission of independently selected Muse BF16 media sources.
use eredu_architectures::muse::{DecoderConfig, media_companion::Bf16MediaCompanion};
use eredu_core::ArtifactCompanionRole;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = std::env::args().skip(1);
    let target = arguments.next().ok_or("usage: inspect_muse_media_source TARGET_CONFIG SOURCE_DIRECTORY vision|audio [vision|audio]")?;
    let source = arguments.next().ok_or("missing companion directory")?;
    let roles = arguments
        .map(|role| match role.as_str() {
            "vision" => Ok(ArtifactCompanionRole::Vision),
            "audio" => Ok(ArtifactCompanionRole::Audio),
            _ => Err("media role must be vision or audio"),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let config = DecoderConfig::from_hf_json(&std::fs::read(target)?)?;
    let companion = Bf16MediaCompanion::admit(&config, source, roles)?;
    let bound = companion.bind_source(2)?;
    companion.verify_unchanged()?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "identity": companion.configuration().identity.model_type(),
            "roles": companion.roles().iter().map(|(role, keys)| (format!("{role:?}"), keys.len())).collect::<std::collections::BTreeMap<_, _>>(),
            "recipes": companion.recipes().len(),
            "retained_files": companion.files().len(),
        "files": companion.files().iter().map(|file| serde_json::json!({
            "role":file.logical_role(),
            "member":file.path().file_name().and_then(|name| name.to_str()),
            "bytes":std::fs::metadata(file.path()).map(|metadata| metadata.len()).ok(),
        })).collect::<Vec<_>>(),
            "visible_source_keys": bound.source_keys().len(),
            "decoder_keys_visible": bound.source_keys().iter().any(|key| key.starts_with("model.language_model.")),
            "tensor_payloads_acquired": false,
        }))?
    );
    Ok(())
}
