//! Backend-independent text generation and EOS sidecar metadata.

use std::path::Path;

use safemlx_gguf::MetadataValue as GgufMetadataValue;
use safemlx_lm_core::generation::CheckpointGenerationConfig;
use serde::Deserialize;

use super::TextMetadataError;

#[derive(Debug, Clone, Default, Deserialize)]
struct EosTokenMetadata {
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
    #[serde(default)]
    text_config: Option<TextEosTokenMetadata>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TextEosTokenMetadata {
    #[serde(default)]
    eos_token_id: Option<TokenIdOrIds>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum TokenIdOrIds {
    Single(u32),
    Multiple(Vec<u32>),
}

impl TokenIdOrIds {
    fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Single(id) => vec![id],
            Self::Multiple(ids) => ids,
        }
    }
}

pub(crate) fn read_checkpoint_generation_config(
    sidecar_dir: &Path,
) -> Result<Option<CheckpointGenerationConfig>, TextMetadataError> {
    let path = sidecar_dir.join("generation_config.json");
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_reader(file)?))
}

fn append_unique_eos_token_ids(output: &mut Vec<u32>, ids: impl IntoIterator<Item = u32>) {
    for id in ids {
        if !output.contains(&id) {
            output.push(id);
        }
    }
}

pub(crate) fn merge_eos_token_id_sources(
    sources: impl IntoIterator<Item = impl IntoIterator<Item = u32>>,
) -> Vec<u32> {
    let mut output = Vec::new();
    for source in sources {
        append_unique_eos_token_ids(&mut output, source);
    }
    output
}

fn read_optional_eos_token_metadata(
    path: &Path,
) -> Result<Option<EosTokenMetadata>, TextMetadataError> {
    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(serde_json::from_reader(file)?))
}

pub(crate) fn eos_token_ids_from_sidecar_dir(
    sidecar_dir: &Path,
) -> Result<Vec<u32>, TextMetadataError> {
    let mut output = Vec::new();
    for filename in ["config.json", "generation_config.json"] {
        let Some(metadata) = read_optional_eos_token_metadata(&sidecar_dir.join(filename))? else {
            continue;
        };
        if let Some(ids) = metadata.eos_token_id {
            append_unique_eos_token_ids(&mut output, ids.into_vec());
        }
        if let Some(ids) = metadata
            .text_config
            .and_then(|text_config| text_config.eos_token_id)
        {
            append_unique_eos_token_ids(&mut output, ids.into_vec());
        }
    }
    Ok(output)
}

pub(crate) fn gguf_eos_token_ids(
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
) -> Result<Vec<u32>, TextMetadataError> {
    const KEY: &str = "tokenizer.ggml.eos_token_id";
    safemlx_lm_core::gguf_u32_metadata_values(KEY, metadata.get(KEY))
        .map_err(|error| TextMetadataError::UnsupportedArchitecture(error.to_string()))
}
