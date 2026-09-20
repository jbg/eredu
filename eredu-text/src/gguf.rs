//! Reconstruction of executable tokenizers from portable GGUF metadata.

use std::collections::{HashMap, HashSet};

use eredu_gguf::MetadataValue as GgufMetadataValue;
use serde_json::{Map, Value};
use tokenizers::{
    AddedToken, DecoderWrapper, SplitDelimiterBehavior, Tokenizer,
    decoders::{
        byte_fallback::ByteFallback, fuse::Fuse, sequence::Sequence as DecoderSequence,
        strip::Strip,
    },
    models::{
        bpe::{BPE, Vocab},
        unigram::Unigram,
    },
    normalizers::{NFC, Replace},
    pre_tokenizers::{
        PreTokenizerWrapper,
        byte_level::ByteLevel,
        metaspace::{Metaspace, PrependScheme},
        sequence::Sequence as PreTokenizerSequence,
        split::{Split, SplitPattern},
    },
    processors::template::TemplateProcessing,
};

use crate::error::Error;

/// Portable tokenizer reconstructed from GGUF metadata.
pub struct GgufTokenizer {
    /// Executable Hugging Face tokenizer reconstructed from the metadata.
    pub tokenizer: Tokenizer,
    /// Template variables derived from named GGUF special-token IDs.
    pub template_kwargs: Map<String, Value>,
}

/// Reconstructs a supported tokenizer from portable GGUF metadata.
///
/// Returns `None` when the metadata does not describe a tokenizer family this
/// crate can reconstruct.
pub fn from_metadata(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Option<GgufTokenizer>, Error> {
    from_metadata_with_cache_policy(metadata, crate::tokenizer::ModelCachePolicy::default())
}

/// Reconstructs with a model-cache entry limit applied before first use.
/// Parsing and model construction use ordinary upstream allocations.
pub fn from_metadata_with_cache_policy(
    metadata: &HashMap<String, GgufMetadataValue>,
    policy: crate::tokenizer::ModelCachePolicy,
) -> Result<Option<GgufTokenizer>, Error> {
    if let Some(json) = metadata
        .get("tokenizer.huggingface.json")
        .and_then(GgufMetadataValue::as_str)
    {
        let tokenizer = policy.from_bytes(json.as_bytes())?;
        return Ok(Some(GgufTokenizer {
            tokenizer,
            template_kwargs: special_token_kwargs(metadata)?,
        }));
    }

    let Some(tokens) = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    else {
        return Ok(None);
    };
    let Some(model_type) = metadata
        .get("tokenizer.ggml.model")
        .and_then(GgufMetadataValue::as_str)
    else {
        return Ok(None);
    };
    let architecture = metadata
        .get("general.architecture")
        .and_then(GgufMetadataValue::as_str)
        .unwrap_or_default();

    let mut tokenizer = match (architecture, model_type) {
        ("gemma2" | "gemma4" | "gemma4_assistant" | "gemma4-assistant", _) => {
            build_gemma(tokens, metadata, policy)?
        }
        ("llama", _) => build_llama(tokens, metadata, policy)?,
        (_, "llama") => build_llama(tokens, metadata, policy)?,
        (_, "gpt2") => build_gpt(tokens, metadata, policy)?,
        _ => return Ok(None),
    };
    register_special_tokens(&mut tokenizer, tokens, metadata)?;
    configure_post_processor(&mut tokenizer, tokens, metadata)?;

    Ok(Some(GgufTokenizer {
        tokenizer,
        template_kwargs: special_token_kwargs(metadata)?,
    }))
}

/// Extracts chat-template variables derived from named GGUF special-token IDs.
pub fn template_kwargs(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Map<String, Value>, Error> {
    special_token_kwargs(metadata)
}

fn build_gpt(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
    policy: crate::tokenizer::ModelCachePolicy,
) -> Result<Tokenizer, Error> {
    let merges = required_merges(metadata)?;
    let mut builder = BPE::builder()
        .cache_capacity(policy.capacity)
        .vocab_and_merges(vocab(tokens)?, merges)
        .fuse_unk(true);
    if let Some(unknown) = special_token(metadata, tokens, "unknown_token_id")? {
        builder = builder.unk_token(unknown);
    }
    let model = builder.build()?;
    let mut tokenizer = Tokenizer::new(model);
    let pre_tokenizer = metadata
        .get("tokenizer.ggml.pre")
        .and_then(GgufMetadataValue::as_str)
        .unwrap_or_default();
    let architecture = metadata
        .get("general.architecture")
        .and_then(GgufMetadataValue::as_str)
        .unwrap_or_default();
    const GPT4O_PATTERN: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    if pre_tokenizer == "k2-horizon" {
        // Publisher pattern: combining marks and joiners stay in words, while
        // numbers split into groups of at most three digits. No NFC transform.
        const K2_HORIZON_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?(?:\p{L}|\p{M}|\x{200C}|\x{200D})+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Split(Split::new(
                SplitPattern::Regex(K2_HORIZON_PATTERN.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, true, false)),
        ])));
    } else if pre_tokenizer == "kimi-k2" || architecture == "kimi-linear" {
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Split(Split::new(
                SplitPattern::Regex(super::tiktoken::KIMI_K2_PATTERN.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));
    } else if matches!(pre_tokenizer, "gpt-4o" | "inkling")
        || matches!(architecture, "gpt-oss" | "inkling")
    {
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Split(Split::new(
                SplitPattern::Regex(GPT4O_PATTERN.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));
    } else if pre_tokenizer == "lfm2" || matches!(architecture, "lfm2" | "lfm2moe") {
        const LFM2_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Split(Split::new(
                SplitPattern::Regex(LFM2_PATTERN.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));
    } else if matches!(pre_tokenizer, "qwen2" | "qwen35")
        || matches!(architecture, "qwen3" | "qwen3moe" | "qwen35" | "qwen35moe")
    {
        const QWEN_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
        tokenizer.with_normalizer(Some(NFC))?;
        tokenizer.with_pre_tokenizer(Some(PreTokenizerSequence::new(vec![
            PreTokenizerWrapper::Split(Split::new(
                SplitPattern::Regex(QWEN_PATTERN.into()),
                SplitDelimiterBehavior::Isolated,
                false,
            )?),
            PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
        ])));
    } else {
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
    }
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, true)));
    Ok(tokenizer)
}

fn build_llama(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
    policy: crate::tokenizer::ModelCachePolicy,
) -> Result<Tokenizer, Error> {
    let merges = match metadata
        .get("tokenizer.ggml.merges")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(values) => parse_merges(values)?,
        None => derive_merges(tokens, metadata)?,
    };
    let mut builder = BPE::builder()
        .cache_capacity(policy.capacity)
        .vocab_and_merges(vocab(tokens)?, merges)
        .fuse_unk(true)
        .byte_fallback(true);
    if let Some(unknown) = special_token(metadata, tokens, "unknown_token_id")? {
        builder = builder.unk_token(unknown);
    }
    let mut tokenizer = Tokenizer::new(builder.build()?);
    let is_llama3 = metadata
        .get("tokenizer.ggml.model")
        .and_then(GgufMetadataValue::as_str)
        != Some("llama");
    let add_prefix_space =
        metadata_bool(metadata, "tokenizer.ggml.add_space_prefix")?.unwrap_or(!is_llama3);
    if is_llama3 {
        tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, true)));
    } else {
        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            '▁',
            if add_prefix_space {
                PrependScheme::Always
            } else {
                PrependScheme::Never
            },
            true,
        )));
    }
    tokenizer.with_decoder(Some(sentencepiece_decoder(add_prefix_space, is_llama3)?));
    Ok(tokenizer)
}

fn build_gemma(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
    policy: crate::tokenizer::ModelCachePolicy,
) -> Result<Tokenizer, Error> {
    let scores = required_scores(metadata, tokens.len())?;
    let vocabulary = tokens
        .iter()
        .zip(scores)
        .map(|(token, score)| {
            let token = if token == "<0x09>" {
                "\t".to_string()
            // Gemma GGUF vocabularies may contain literal newline tokens. Only
            // ordinary spaces use SentencePiece's metaspace representation;
            // converting every whitespace character would flatten line breaks.
            } else if !token.is_empty() && token.chars().all(|character| character == ' ') {
                "▁".repeat(token.chars().count())
            } else {
                token.clone()
            };
            (token, f64::from(score))
        })
        .collect();
    let unknown = special_id(metadata, "unknown_token_id")?;
    let mut model = Unigram::from(vocabulary, unknown, true)?;
    model.resize_cache(policy.capacity);
    let mut tokenizer = Tokenizer::new(model);
    let add_prefix_space =
        metadata_bool(metadata, "tokenizer.ggml.add_space_prefix")?.unwrap_or(true);
    tokenizer.with_normalizer(Some(Replace::new(" ", "▁")?))?;
    tokenizer.with_pre_tokenizer(Some(Metaspace::new(
        '▁',
        if add_prefix_space {
            PrependScheme::Always
        } else {
            PrependScheme::Never
        },
        true,
    )));
    tokenizer.with_decoder(Some(gemma_decoder(add_prefix_space)?));
    Ok(tokenizer)
}

fn gemma_decoder(add_prefix_space: bool) -> Result<DecoderSequence, Error> {
    let mut decoders = vec![
        DecoderWrapper::Replace(Replace::new("▁", " ")?),
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
    ];
    if add_prefix_space {
        decoders.push(DecoderWrapper::Strip(Strip::new(' ', 1, 0)));
    }
    Ok(DecoderSequence::new(decoders))
}

fn sentencepiece_decoder(
    add_prefix_space: bool,
    append_byte_level: bool,
) -> Result<DecoderSequence, Error> {
    let mut decoders = vec![
        DecoderWrapper::ByteFallback(ByteFallback::new()),
        DecoderWrapper::Fuse(Fuse::new()),
        DecoderWrapper::Replace(Replace::new("▁", " ")?),
    ];
    if append_byte_level {
        decoders.push(DecoderWrapper::ByteLevel(ByteLevel::new(
            false, false, true,
        )));
    }
    if add_prefix_space {
        decoders.push(DecoderWrapper::Strip(Strip::new(' ', 1, 0)));
    }
    Ok(DecoderSequence::new(decoders))
}

fn vocab(tokens: &[String]) -> Result<Vocab, Error> {
    tokens
        .iter()
        .enumerate()
        .map(|(id, token)| {
            u32::try_from(id)
                .map(|id| (token.clone(), id))
                .map_err(|_| tokenizer_error("vocabulary has more than u32::MAX entries"))
        })
        .collect()
}

fn required_merges(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Vec<(String, String)>, Error> {
    let values = metadata
        .get("tokenizer.ggml.merges")
        .and_then(GgufMetadataValue::as_strings)
        .ok_or_else(|| tokenizer_error("BPE metadata is missing tokenizer.ggml.merges"))?;
    parse_merges(values)
}

fn parse_merges(values: &[String]) -> Result<Vec<(String, String)>, Error> {
    values
        .iter()
        .map(|merge| {
            merge
                .split_once(' ')
                .map(|(left, right)| (left.to_string(), right.to_string()))
                .ok_or_else(|| tokenizer_error(format!("invalid BPE merge {merge:?}")))
        })
        .collect()
}

fn derive_merges(
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Vec<(String, String)>, Error> {
    let scores = required_scores(metadata, tokens.len())?;
    let token_scores = tokens
        .iter()
        .cloned()
        .zip(scores.iter().copied())
        .collect::<HashMap<_, _>>();
    let mut merges = Vec::new();
    for (token, score) in tokens.iter().zip(scores) {
        let mut local = token
            .char_indices()
            .skip(1)
            .filter_map(|(index, _)| {
                let (left, right) = token.split_at(index);
                Some((
                    left.to_string(),
                    right.to_string(),
                    *token_scores.get(left)?,
                    *token_scores.get(right)?,
                    score,
                ))
            })
            .collect::<Vec<_>>();
        local.sort_by(|a, b| b.2.total_cmp(&a.2).then_with(|| b.3.total_cmp(&a.3)));
        merges.extend(local);
    }
    merges.sort_by(|a, b| b.4.total_cmp(&a.4));
    Ok(merges
        .into_iter()
        .map(|(left, right, _, _, _)| (left, right))
        .collect())
}

fn required_scores(
    metadata: &HashMap<String, GgufMetadataValue>,
    expected: usize,
) -> Result<Vec<f32>, Error> {
    let scores = metadata
        .get("tokenizer.ggml.scores")
        .and_then(GgufMetadataValue::as_array)
        .and_then(|values| values.to_f32_vec())
        .ok_or_else(|| tokenizer_error("metadata is missing tokenizer.ggml.scores"))?;
    if scores.len() != expected {
        return Err(tokenizer_error(format!(
            "tokenizer.ggml.scores has {} entries for a {expected}-token vocabulary",
            scores.len()
        )));
    }
    Ok(scores)
}

fn register_special_tokens(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<(), Error> {
    let mut ids = HashSet::new();
    let mut user_defined = Vec::new();
    if let Some(types) = metadata
        .get("tokenizer.ggml.token_type")
        .and_then(GgufMetadataValue::as_array)
        .and_then(|values| values.to_i64_vec())
    {
        if types.len() != tokens.len() {
            return Err(tokenizer_error(
                "tokenizer.ggml.token_type length does not match the vocabulary",
            ));
        }
        ids.extend(
            types
                .iter()
                .enumerate()
                .filter_map(|(id, kind)| (*kind == 3).then_some(id)),
        );
        user_defined.extend(
            types
                .iter()
                .enumerate()
                .filter_map(|(id, kind)| (*kind == 4).then_some(id)),
        );
    }
    for name in [
        "bos_token_id",
        "eos_token_id",
        "unknown_token_id",
        "padding_token_id",
        "separator_token_id",
    ] {
        if name == "eos_token_id" {
            ids.extend(special_ids(metadata, name)?);
        } else if let Some(id) = special_id(metadata, name)? {
            ids.insert(id);
        }
    }
    for token in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] {
        if let Some(id) = tokens.iter().position(|candidate| candidate == token) {
            ids.insert(id);
        }
    }
    let added = ids
        .into_iter()
        .map(|id| {
            tokens
                .get(id)
                .cloned()
                .map(|token| AddedToken::from(token, true).normalized(false))
                .ok_or_else(|| {
                    tokenizer_error(format!("special token id {id} is outside the vocabulary"))
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    tokenizer.add_special_tokens(added)?;
    tokenizer.add_tokens(
        user_defined
            .into_iter()
            .map(|id| AddedToken::from(tokens[id].clone(), false).normalized(false)),
    )?;
    Ok(())
}

fn configure_post_processor(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<(), Error> {
    let bos = metadata_bool(metadata, "tokenizer.ggml.add_bos_token")?
        .unwrap_or(false)
        .then(|| special_token(metadata, tokens, "bos_token_id"))
        .transpose()?
        .flatten();
    let eos = metadata_bool(metadata, "tokenizer.ggml.add_eos_token")?
        .unwrap_or(false)
        .then(|| special_token(metadata, tokens, "eos_token_id"))
        .transpose()?
        .flatten();
    if bos.is_none() && eos.is_none() {
        return Ok(());
    }
    let mut single = Vec::new();
    let mut pair = Vec::new();
    let mut specials = Vec::new();
    if let Some(token) = &bos {
        single.push(token.clone());
        pair.push(token.clone());
        specials.push((token.clone(), tokenizer.token_to_id(token).unwrap()));
    }
    single.push("$A".into());
    pair.push("$A".into());
    if let Some(token) = &eos {
        single.push(token.clone());
        pair.push(token.clone());
        specials.push((token.clone(), tokenizer.token_to_id(token).unwrap()));
    }
    pair.push("$B:1".into());
    if let Some(token) = &eos {
        pair.push(format!("{token}:1"));
    }
    specials.sort_by_key(|(_, id)| *id);
    specials.dedup_by_key(|(_, id)| *id);
    tokenizer.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single(single.join(" "))
            .map_err(|error| tokenizer_error(error.to_string()))?
            .try_pair(pair.join(" "))
            .map_err(|error| tokenizer_error(error.to_string()))?
            .special_tokens(specials)
            .build()
            .map_err(|error| tokenizer_error(error.to_string()))?,
    ));
    Ok(())
}

fn special_token_kwargs(
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<Map<String, Value>, Error> {
    let Some(tokens) = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    else {
        return Ok(Map::new());
    };
    let mut kwargs = Map::new();
    for (metadata_name, kwarg_name) in [
        ("bos_token_id", "bos_token"),
        ("eos_token_id", "eos_token"),
        ("unknown_token_id", "unk_token"),
        ("padding_token_id", "pad_token"),
        ("separator_token_id", "sep_token"),
    ] {
        if let Some(token) = special_token(metadata, tokens, metadata_name)? {
            kwargs.insert(kwarg_name.into(), Value::String(token));
        }
    }
    Ok(kwargs)
}

fn special_token(
    metadata: &HashMap<String, GgufMetadataValue>,
    tokens: &[String],
    name: &str,
) -> Result<Option<String>, Error> {
    let id = if name == "eos_token_id" {
        special_ids(metadata, name)?.into_iter().next()
    } else {
        special_id(metadata, name)?
    };
    id.map(|id| {
        tokens.get(id).cloned().ok_or_else(|| {
            tokenizer_error(format!(
                "tokenizer.ggml.{name}={id} is outside the vocabulary"
            ))
        })
    })
    .transpose()
}

fn special_ids(
    metadata: &HashMap<String, GgufMetadataValue>,
    name: &str,
) -> Result<Vec<usize>, Error> {
    let key = format!("tokenizer.ggml.{name}");
    let Some(value) = metadata.get(&key) else {
        return Ok(Vec::new());
    };
    let values = value
        .to_i64_vec()
        .ok_or_else(|| tokenizer_error(format!("{key} must be an integer or integer array")))?;
    values
        .into_iter()
        .map(|id| {
            u32::try_from(id).map(|id| id as usize).map_err(|_| {
                tokenizer_error(format!(
                    "{key} must contain integers from 0 through {}",
                    u32::MAX
                ))
            })
        })
        .collect()
}

fn special_id(
    metadata: &HashMap<String, GgufMetadataValue>,
    name: &str,
) -> Result<Option<usize>, Error> {
    let key = format!("tokenizer.ggml.{name}");
    metadata
        .get(&key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|id| usize::try_from(id).ok())
                .ok_or_else(|| tokenizer_error(format!("{key} must be a non-negative integer")))
        })
        .transpose()
}

fn metadata_bool(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<bool>, Error> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| tokenizer_error(format!("{key} must be a boolean")))
        })
        .transpose()
}

fn tokenizer_error(message: impl Into<String>) -> Error {
    Error::InvalidTokenizer(format!("GGUF tokenizer error: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use eredu_gguf::{MetadataArray as GgufMetadataArray, MetadataValue as GgufMetadataValue};

    use super::*;

    #[test]
    fn k2_horizon_preserves_joined_words_marks_and_three_digit_boundaries() {
        use tokenizers::PreTokenizer;
        let metadata = HashMap::from([
            (
                "tokenizer.ggml.pre".into(),
                GgufMetadataValue::String("k2-horizon".into()),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![])),
            ),
        ]);
        let tokenizer = build_gpt(
            &["x".into()],
            &metadata,
            crate::tokenizer::ModelCachePolicy::default(),
        )
        .unwrap();
        let input = "a\u{301}b x\u{200c}y\u{200d}z 1234567";
        let mut pretokenized = tokenizers::PreTokenizedString::from(input);
        tokenizer
            .get_pre_tokenizer()
            .unwrap()
            .pre_tokenize(&mut pretokenized)
            .unwrap();
        let offsets = pretokenized
            .get_splits(
                tokenizers::OffsetReferential::Original,
                tokenizers::OffsetType::Byte,
            )
            .into_iter()
            .map(|(_, offset, _)| offset)
            .collect::<Vec<_>>();
        let pieces = offsets
            .iter()
            .map(|&(start, end)| &input[start..end])
            .collect::<Vec<_>>();
        assert_eq!(
            pieces,
            ["a\u{301}b", " x\u{200c}y\u{200d}z", " ", "123", "456", "7"]
        );
        assert!(tokenizer.get_normalizer().is_none());
    }

    #[test]
    fn builds_embedded_gpt_tokenizer() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("qwen3".into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                GgufMetadataValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "<eos>".into(),
                    "h".into(),
                    "e".into(),
                    "l".into(),
                    "o".into(),
                    "he".into(),
                    "ll".into(),
                    "hell".into(),
                    "hello".into(),
                    "<eos2>".into(),
                ])),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "h e".into(),
                    "l l".into(),
                    "he ll".into(),
                    "hell o".into(),
                ])),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Uint32(vec![0, 9])),
            ),
            (
                "tokenizer.ggml.add_eos_token".into(),
                GgufMetadataValue::Bool(true),
            ),
        ]);

        let loaded = from_metadata(&metadata).unwrap().unwrap();
        let encoding = loaded.tokenizer.encode("hello", true).unwrap();
        assert_eq!(encoding.get_ids(), &[8, 0]);
        assert_eq!(loaded.tokenizer.decode(&[0, 9], true).unwrap(), "");
        assert_eq!(loaded.template_kwargs["eos_token"], "<eos>");
    }

    #[test]
    fn uses_lfm2_pretokenizer_without_unicode_normalization() {
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("lfm2".into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                GgufMetadataValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.pre".into(),
                GgufMetadataValue::String("lfm2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "<eos>".into(),
                    "a".into(),
                ])),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![])),
            ),
        ]);
        let tokenizer = from_metadata(&metadata).unwrap().unwrap().tokenizer;
        let serialized = tokenizer.to_string(false).unwrap();
        assert!(serialized.contains(r"\\p{N}{1,3}"));
        assert!(serialized.contains("ByteLevel"));
        assert!(tokenizer.get_normalizer().is_none());
    }

    #[test]
    fn uses_gpt4o_family_pretokenizers_for_gpt_oss_and_inkling() {
        for (architecture, pre) in [("gpt-oss", "gpt-4o"), ("inkling", "inkling")] {
            let metadata = HashMap::from([
                (
                    "general.architecture".into(),
                    GgufMetadataValue::String(architecture.into()),
                ),
                (
                    "tokenizer.ggml.model".into(),
                    GgufMetadataValue::String("gpt2".into()),
                ),
                (
                    "tokenizer.ggml.pre".into(),
                    GgufMetadataValue::String(pre.into()),
                ),
                (
                    "tokenizer.ggml.tokens".into(),
                    GgufMetadataValue::Array(GgufMetadataArray::String(vec!["a".into()])),
                ),
                (
                    "tokenizer.ggml.merges".into(),
                    GgufMetadataValue::Array(GgufMetadataArray::String(vec![])),
                ),
            ]);
            let tokenizer = from_metadata(&metadata).unwrap().unwrap().tokenizer;
            let serialized = tokenizer.to_string(false).unwrap();
            assert!(serialized.contains(r"\\p{M}"));
            assert!(serialized.contains(r"\\p{N}{1,3}"));
            assert!(serialized.contains("ByteLevel"));
        }
    }

    #[test]
    fn derives_llama_merges_from_scores() {
        let metadata = sentencepiece_metadata("llama");
        let loaded = from_metadata(&metadata).unwrap().unwrap();
        let encoding = loaded.tokenizer.encode("hi", false).unwrap();
        assert_eq!(encoding.get_ids(), &[5]);
        assert_eq!(
            loaded.tokenizer.decode(encoding.get_ids(), false).unwrap(),
            "hi"
        );
    }

    #[test]
    fn builds_gemma_unigram_tokenizer() {
        for family in ["gemma2", "gemma4"] {
            let metadata = sentencepiece_metadata(family);
            let loaded = from_metadata(&metadata).unwrap().unwrap();
            let encoding = loaded.tokenizer.encode("hi", false).unwrap();
            assert_eq!(encoding.get_ids(), &[5]);
            assert_eq!(
                loaded.tokenizer.decode(encoding.get_ids(), false).unwrap(),
                "hi"
            );
        }
    }

    #[test]
    fn reconstructed_gemma_uses_the_original_source_and_id_workers() {
        for family in ["gemma2", "gemma4"] {
            for prefix in [false, true] {
                let mut metadata = sentencepiece_metadata(family);
                metadata.insert(
                    "tokenizer.ggml.add_space_prefix".into(),
                    GgufMetadataValue::Bool(prefix),
                );
                let selected = from_metadata(&metadata).unwrap().unwrap().tokenizer;
                let bytes = selected.to_string(false).unwrap();
                let original =
                    crate::tokenizer_storage::TokenizerPlan::prepare_json(bytes.as_bytes())
                        .unwrap()
                        .compile()
                        .unwrap();
                let selected_config =
                    crate::tokenizer::Tokenizer::from_bytes(bytes.as_bytes()).unwrap();
                assert!(original.matches_configuration(&selected_config));
                assert!(original.input_prefix_removal_is_identity() == !prefix);
                for text in ["hi", " hi", "hi hi", "hi\nhi", "hi\t hi", "<unk>hi", ""] {
                    let expected = selected.encode(text, false).unwrap();
                    let ids =
                        crate::tokenizer_storage::EncodeIdsPlan::prepare(&original, text, false)
                            .unwrap()
                            .encode()
                            .unwrap();
                    assert_eq!(ids.ids(), expected.get_ids(), "{family} {prefix} {text:?}");
                }
            }
        }
    }

    #[test]
    fn builds_gemma_assistant_unigram_tokenizers() {
        for architecture in ["gemma4_assistant", "gemma4-assistant"] {
            let metadata = sentencepiece_metadata(architecture);
            let loaded = from_metadata(&metadata).unwrap().unwrap();
            let encoding = loaded.tokenizer.encode("hi", false).unwrap();
            assert_eq!(encoding.get_ids(), &[5], "{architecture}");
            assert_eq!(
                loaded.tokenizer.decode(encoding.get_ids(), false).unwrap(),
                "hi",
                "{architecture}"
            );
        }
    }

    #[test]
    fn gemma_decoder_preserves_literal_newlines() {
        let metadata = sentencepiece_metadata("gemma4");
        let tokenizer = from_metadata(&metadata).unwrap().unwrap().tokenizer;

        assert_eq!(
            tokenizer.decode(&[5, 6, 4, 7, 4], false).unwrap(),
            "hi\nhi\n\nhi"
        );
        assert_eq!(tokenizer.decode(&[5, 6, 5], false).unwrap(), "hi\n hi");
        assert_eq!(tokenizer.decode(&[5, 6, 1, 5], false).unwrap(), "hi\n  hi");
        let encoding = tokenizer.encode("hi\nhi", false).unwrap();
        assert_eq!(
            tokenizer.decode(encoding.get_ids(), false).unwrap(),
            "hi\nhi"
        );
    }

    fn sentencepiece_metadata(architecture: &str) -> HashMap<String, GgufMetadataValue> {
        HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String(architecture.into()),
            ),
            (
                "tokenizer.ggml.model".into(),
                GgufMetadataValue::String("llama".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "<unk>".into(),
                    "▁".into(),
                    "h".into(),
                    "i".into(),
                    "hi".into(),
                    "▁hi".into(),
                    "\n".into(),
                    "\n\n".into(),
                ])),
            ),
            (
                "tokenizer.ggml.scores".into(),
                GgufMetadataValue::Array(GgufMetadataArray::Float32(vec![
                    0.0, 1.0, 2.0, 2.0, 3.0, 10.0, 1.0, 1.0,
                ])),
            ),
            (
                "tokenizer.ggml.unknown_token_id".into(),
                GgufMetadataValue::Uint32(0),
            ),
        ])
    }
    #[test]
    fn model_cache_policy_reaches_gguf_bpe_unigram_and_embedded_json() {
        use crate::tokenizer::ModelCachePolicy;
        for family in ["llama", "gemma2", "gemma4"] {
            let metadata = sentencepiece_metadata(family);
            let legacy = from_metadata(&metadata).unwrap().unwrap();
            let absent = from_metadata_with_cache_policy(&metadata, ModelCachePolicy::disabled())
                .unwrap()
                .unwrap();
            for input in ["hi", "hi hi", "hi\nhi"] {
                let a = legacy.tokenizer.encode(input, true).unwrap();
                let b = absent.tokenizer.encode(input, true).unwrap();
                assert!(!a.get_ids().is_empty());
                assert_eq!(a.get_ids(), b.get_ids());
                assert_eq!(a.get_offsets(), b.get_offsets());
                assert_eq!(
                    legacy.tokenizer.decode(a.get_ids(), false).unwrap(),
                    absent.tokenizer.decode(b.get_ids(), false).unwrap()
                );
            }
            assert_eq!(legacy.template_kwargs, absent.template_kwargs);
            let mut embedded = metadata.clone();
            embedded.insert(
                "tokenizer.huggingface.json".into(),
                GgufMetadataValue::String(legacy.tokenizer.to_string(false).unwrap()),
            );
            let restored = from_metadata_with_cache_policy(&embedded, ModelCachePolicy::disabled())
                .unwrap()
                .unwrap();
            assert_eq!(
                restored.tokenizer.encode("hi", false).unwrap().get_ids(),
                absent.tokenizer.encode("hi", false).unwrap().get_ids()
            );
        }
    }

    #[test]
    fn disabled_caching_preserves_gpt_builder_nonzero_merges() {
        let metadata = HashMap::from([
            (
                "tokenizer.ggml.model".into(),
                GgufMetadataValue::String("gpt2".into()),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec![
                    "a".into(),
                    "b".into(),
                    "ab".into(),
                ])),
            ),
            (
                "tokenizer.ggml.merges".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec!["a b".into()])),
            ),
        ]);
        let legacy = from_metadata(&metadata).unwrap().unwrap().tokenizer;
        let absent = from_metadata_with_cache_policy(
            &metadata,
            crate::tokenizer::ModelCachePolicy::disabled(),
        )
        .unwrap()
        .unwrap()
        .tokenizer;
        assert_eq!(absent.encode("abab", false).unwrap().get_ids(), &[2, 2]);
        assert_eq!(
            legacy.encode("abab", false).unwrap().get_offsets(),
            absent.encode("abab", false).unwrap().get_offsets()
        );
        assert_eq!(absent.decode(&[2, 2], false).unwrap(), "abab");
    }
}
