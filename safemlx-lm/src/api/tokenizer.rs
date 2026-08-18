//! Tokenizer, chat-template, and GGUF metadata loading.

use super::*;
use sha2::{Digest, Sha256};

/// Backend-independent tokenizer, chat-template, and text-sidecar failure.
#[derive(Debug, thiserror::Error)]
pub enum TextMetadataError {
    /// Artifact identity or model-kind normalization failed.
    #[error(transparent)]
    Artifact(#[from] safemlx_lm_core::artifact::ArtifactError),
    /// Filesystem metadata could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON metadata was invalid.
    #[error(transparent)]
    Deserialize(#[from] serde_json::Error),
    /// The portable tokenizer rejected its configuration or input.
    #[error(transparent)]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),
    /// Chat-template parsing or rendering metadata was invalid.
    #[error(transparent)]
    Template(#[from] safemlx_lm_utils::error::Error),
    /// Portable GGUF parsing failed.
    #[error(transparent)]
    Gguf(#[from] safemlx_gguf::Error),
    /// Embedded GGUF tokenizer metadata was invalid.
    #[error("GGUF tokenizer error: {0}")]
    GgufTokenizer(String),
    /// The artifact is valid but not supported by the text facade.
    #[error("unsupported text architecture: {0}")]
    UnsupportedArchitecture(String),
    /// Architecture-specific tokenizer reconstruction failed.
    #[error("tokenizer configuration error: {0}")]
    TokenizerConfiguration(String),
}

pub(crate) fn tokenizer_vocabulary_fingerprint(tokenizer: &Tokenizer) -> [u8; 32] {
    let vocabulary_size = tokenizer.get_vocab_size(true);
    let mut hasher = Sha256::new();
    hasher.update(b"safemlx-token-id-vocabulary-v1");
    hasher.update((vocabulary_size as u64).to_le_bytes());
    for token_id in 0..vocabulary_size {
        hasher.update((token_id as u64).to_le_bytes());
        match tokenizer.id_to_token(token_id as u32) {
            Some(token) => {
                hasher.update((token.len() as u64).to_le_bytes());
                hasher.update(token.as_bytes());
            }
            None => hasher.update(u64::MAX.to_le_bytes()),
        }
    }
    hasher.finalize().into()
}

/// Loads only the tokenizer from a supported model directory or GGUF file.
///
/// Loading from GGUF parses embedded tokenizer metadata without creating an
/// MLX stream. A sibling `tokenizer.json` remains a fallback for missing or
/// unsupported embedded tokenizer formats.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, TextMetadataError> {
    let model_dir = model_dir.as_ref();
    if is_gguf_file(model_dir) {
        return Ok(load_gguf_tokenizer(model_dir)?.tokenizer);
    }
    let metadata = read_model_metadata(model_dir)?;
    load_tokenizer_for_kind(
        ModelKind::from_model_type(&effective_model_type(&metadata))?,
        model_dir,
    )
}

pub(super) fn load_tokenizer_for_kind(
    kind: ModelKind,
    model_dir: &Path,
) -> Result<Tokenizer, TextMetadataError> {
    match kind {
        ModelKind::KimiLinear => {
            let converted = model_dir.join("tokenizer.json");
            if converted.exists() {
                Ok(Tokenizer::from_file(converted)?)
            } else {
                kimi_linear::load_tokenizer(model_dir)
                    .map_err(|error| TextMetadataError::TokenizerConfiguration(error.to_string()))
            }
        }
        ModelKind::PersonaPlex => Err(TextMetadataError::UnsupportedArchitecture(
            "PersonaPlex uses the released SentencePiece tokenizer; load it outside the chat tokenizer API".into(),
        )),
        _ => Ok(Tokenizer::from_file(model_dir.join("tokenizer.json"))?),
    }
}

/// Returns likely user-provided kwargs referenced by a model directory's chat template.
///
/// This reads tokenizer/chat-template metadata only and does not load model weights.
pub fn chat_template_kwargs(model_dir: impl AsRef<Path>) -> Result<Vec<String>, TextMetadataError> {
    let submitted_path = model_dir.as_ref();
    let (template, model_id, tokenizer_template_kwargs) = if is_gguf_file(submitted_path) {
        let metadata = portable_gguf_metadata(submitted_path)?;
        let sidecar_dir = gguf_sidecar_dir(submitted_path);
        let template = match metadata.get("tokenizer.chat_template") {
            Some(GgufMetadataValue::String(template)) => {
                Some(ModelChatTemplate::Single(template.clone()))
            }
            Some(_) => {
                return Err(TextMetadataError::GgufTokenizer(
                    "tokenizer.chat_template must be a string".into(),
                ));
            }
            None => load_chat_template(sidecar_dir)?,
        };
        let mut template_kwargs = gguf_tokenizer::template_kwargs(&metadata)
            .map_err(|error| TextMetadataError::GgufTokenizer(error.to_string()))?;
        template_kwargs.extend(load_tokenizer_template_kwargs(sidecar_dir)?);
        (
            template,
            submitted_path.display().to_string(),
            template_kwargs,
        )
    } else {
        (
            load_chat_template(submitted_path)?,
            submitted_path.display().to_string(),
            load_tokenizer_template_kwargs(submitted_path)?,
        )
    };
    let Some(template) = template else {
        return Ok(Vec::new());
    };
    let selected = template.select(None)?;
    Ok(
        inspect_chat_template_kwargs(selected.template(), &model_id)?
            .into_iter()
            .filter(|name| !tokenizer_template_kwargs.contains_key(name))
            .collect(),
    )
}

pub(super) fn read_model_metadata(model_dir: &Path) -> Result<ModelMetadata, TextMetadataError> {
    let config_path = model_dir.join("config.json");
    let file = std::fs::File::open(config_path)?;
    Ok(serde_json::from_reader(file)?)
}

pub(super) fn is_gguf_file(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

pub(crate) fn gguf_sidecar_dir(path: &Path) -> &Path {
    path.parent().unwrap_or_else(|| Path::new("."))
}

pub(super) fn load_gguf_tokenizer(gguf_file: &Path) -> Result<GgufTokenizer, TextMetadataError> {
    let metadata = portable_gguf_metadata(gguf_file)?;
    load_gguf_tokenizer_from_metadata(gguf_file, &metadata)
}

fn portable_gguf_metadata(
    gguf_file: &Path,
) -> Result<std::collections::HashMap<String, GgufMetadataValue>, TextMetadataError> {
    Ok(safemlx_gguf::Reader::open(gguf_file)?
        .metadata()
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

pub(super) fn load_gguf_tokenizer_from_metadata(
    gguf_file: &Path,
    metadata: &std::collections::HashMap<String, GgufMetadataValue>,
) -> Result<GgufTokenizer, TextMetadataError> {
    let sidecar_dir = gguf_sidecar_dir(gguf_file);
    if let Some(mut embedded) = gguf_tokenizer::from_metadata(metadata)
        .map_err(|error| TextMetadataError::GgufTokenizer(error.to_string()))?
    {
        embedded
            .template_kwargs
            .extend(load_tokenizer_template_kwargs(sidecar_dir)?);
        return Ok(embedded);
    }
    Ok(GgufTokenizer {
        tokenizer: Tokenizer::from_file(sidecar_dir.join("tokenizer.json"))?,
        template_kwargs: load_tokenizer_template_kwargs(sidecar_dir)?,
    })
}

pub(super) fn effective_model_type(metadata: &ModelMetadata) -> String {
    if metadata.model_type == "inkling_mm_model" {
        return metadata.model_type.clone();
    }
    if matches!(
        metadata.model_type.as_str(),
        "gemma4" | "gemma4_unified" | "qwen3_vl" | "qwen3_vl_moe" | "qwen3_5" | "qwen3_5_moe"
    ) {
        metadata
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    } else if ModelKind::from_model_type(&metadata.model_type).is_ok() {
        metadata.model_type.clone()
    } else {
        metadata
            .text_config
            .as_ref()
            .and_then(|text_config| text_config.model_type.clone())
            .unwrap_or_else(|| metadata.model_type.clone())
    }
}

pub(super) fn load_chat_template(
    model_dir: &Path,
) -> Result<Option<ModelChatTemplate>, TextMetadataError> {
    let config_path = model_dir.join("tokenizer_config.json");
    if config_path.exists() {
        if let Some(template) = load_model_chat_template_from_file(config_path)? {
            return Ok(Some(template));
        }
    }

    let jinja_path = model_dir.join("chat_template.jinja");
    if jinja_path.exists() {
        return Ok(Some(ModelChatTemplate::Single(std::fs::read_to_string(
            jinja_path,
        )?)));
    }

    if !model_dir.join("config.json").exists() {
        return Ok(None);
    }

    let metadata = read_model_metadata(model_dir)?;
    if matches!(metadata.model_type.as_str(), "gemma4" | "gemma4_unified")
        || metadata.text_config.as_ref().is_some_and(|text_config| {
            matches!(
                text_config.model_type.as_deref(),
                Some("gemma4_text" | "gemma4_unified_text")
            )
        })
    {
        return Ok(Some(ModelChatTemplate::Single(
            GEMMA4_TEXT_TEMPLATE.to_string(),
        )));
    }

    Ok(None)
}

pub(super) fn load_tokenizer_template_kwargs(
    model_dir: &Path,
) -> Result<Map<String, Value>, TextMetadataError> {
    let config_path = model_dir.join("tokenizer_config.json");
    if !config_path.exists() {
        return Ok(Map::new());
    }

    let value: Value = serde_json::from_reader(std::fs::File::open(config_path)?)?;
    let Some(object) = value.as_object() else {
        return Ok(Map::new());
    };

    Ok(object
        .iter()
        .filter(|(key, value)| key.ends_with("_token") && (value.is_string() || value.is_null()))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect())
}

const GEMMA4_TEXT_TEMPLATE: &str = r#"<bos>{% for message in messages %}{% set role = 'model' if message['role'] == 'assistant' else message['role'] %}<|turn>{{ role }}
{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for content in message['content'] %}{% if content['type'] == 'text' %}{{ content['text'] }}{% elif content['type'] == 'image' %}<|image>{% elif content['type'] == 'audio' %}<|audio>{% endif %}{% endfor %}{% endif %}<turn|>
{% endfor %}{% if add_generation_prompt %}<|turn>model
{% endif %}"#;

#[cfg(test)]
mod vocabulary_fingerprint_tests {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::AddedToken;

    use super::{tokenizer_vocabulary_fingerprint, Tokenizer};

    fn tokenizer(tokens: &[&str]) -> Tokenizer {
        let mut tokenizer = Tokenizer::new(WordLevel::default());
        tokenizer
            .add_tokens(
                tokens
                    .iter()
                    .map(|token| AddedToken::from((*token).to_owned(), false))
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        tokenizer
    }

    #[test]
    fn fingerprint_tracks_token_id_mapping() {
        let first = tokenizer(&["<unk>", "alpha", "beta"]);
        let identical = tokenizer(&["<unk>", "alpha", "beta"]);
        let remapped = tokenizer(&["<unk>", "beta", "alpha"]);

        assert_eq!(
            tokenizer_vocabulary_fingerprint(&first),
            tokenizer_vocabulary_fingerprint(&identical)
        );
        assert_ne!(
            tokenizer_vocabulary_fingerprint(&first),
            tokenizer_vocabulary_fingerprint(&remapped)
        );
    }
}
