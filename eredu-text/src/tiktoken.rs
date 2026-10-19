//! Import rank-ordered tiktoken byte-BPE files into a fast tokenizer.

use std::{
    collections::{HashMap, HashSet},
    fs,
    path::Path,
};

use base64::{engine::general_purpose::STANDARD, Engine};
use serde::Deserialize;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{Vocab, BPE},
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    AddedToken, SplitDelimiterBehavior, Tokenizer,
};

use crate::error::Error;

/// Official Kimi/K2 Unicode- and Han-aware pre-tokenization expression.
pub(crate) const KIMI_K2_PATTERN: &str = concat!(
    r"[\p{Han}]+|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"\p{N}{1,3}|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|",
    r"\s*[\r\n]+|\s+(?!\S)|\s+"
);

/// Meta Llama 4's published O200K split expression, also used by Muse Spark.
pub(crate) const LLAMA4_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|",
    r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
);

#[derive(Clone, Copy)]
enum MergePolicy {
    RankedPair,
    VocabularySplits,
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedTokenConfig>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    special: Option<bool>,
}

fn tokenizer_error(message: impl Into<String>) -> Error {
    Error::InvalidTokenizer(format!("invalid tiktoken checkpoint: {}", message.into()))
}

fn read_ranks(path: &Path) -> Result<Vec<Vec<u8>>, Error> {
    let contents = fs::read_to_string(path)?;
    let line_count = contents.lines().count();
    let mut ranked = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        let mut fields = line.split_ascii_whitespace();
        let encoded = fields.next().ok_or_else(|| {
            tokenizer_error(format!("line {} has no encoded token", line_index + 1))
        })?;
        let rank = fields
            .next()
            .ok_or_else(|| tokenizer_error(format!("line {} has no rank", line_index + 1)))?
            .parse::<usize>()
            .map_err(|error| {
                tokenizer_error(format!(
                    "line {} has an invalid rank: {error}",
                    line_index + 1
                ))
            })?;
        if fields.next().is_some() {
            return Err(tokenizer_error(format!(
                "line {} has trailing fields",
                line_index + 1
            )));
        }
        let token = STANDARD.decode(encoded).map_err(|error| {
            tokenizer_error(format!(
                "line {} has invalid base64 token bytes: {error}",
                line_index + 1
            ))
        })?;
        if token.is_empty() {
            return Err(tokenizer_error(format!(
                "line {} contains an empty token",
                line_index + 1
            )));
        }
        if rank >= line_count {
            return Err(tokenizer_error(format!(
                "rank {rank} exceeds the {line_count}-entry file"
            )));
        }
        if rank >= ranked.len() {
            ranked.resize(rank + 1, None);
        }
        if ranked[rank].replace(token).is_some() {
            return Err(tokenizer_error(format!("duplicate rank {rank}")));
        }
    }
    ranked
        .into_iter()
        .enumerate()
        .map(|(rank, token)| token.ok_or_else(|| tokenizer_error(format!("missing rank {rank}"))))
        .collect()
}

fn byte_encoder() -> [char; 256] {
    let mut bytes = Vec::new();
    bytes.extend(33u16..=126);
    bytes.extend(161u16..=172);
    bytes.extend(174u16..=255);
    let mut codepoints = bytes.clone();
    let mut extra = 0u16;
    for byte in 0u16..=255 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(256 + extra);
            extra += 1;
        }
    }
    let mut result = ['\0'; 256];
    for (byte, codepoint) in bytes.into_iter().zip(codepoints) {
        result[usize::from(byte)] = char::from_u32(u32::from(codepoint)).unwrap();
    }
    result
}

fn encoded_token(bytes: &[u8], encoder: &[char; 256]) -> String {
    bytes
        .iter()
        .map(|byte| encoder[usize::from(*byte)])
        .collect()
}

fn bpe_parts(ranks: &HashMap<Vec<u8>, usize>, token: &[u8], max_rank: usize) -> Vec<Vec<u8>> {
    let mut parts = token.iter().map(|byte| vec![*byte]).collect::<Vec<_>>();
    loop {
        let candidate = parts
            .windows(2)
            .enumerate()
            .filter_map(|(index, pair)| {
                let mut merged = pair[0].clone();
                merged.extend_from_slice(&pair[1]);
                ranks
                    .get(&merged)
                    .copied()
                    .filter(|rank| *rank < max_rank)
                    .map(|rank| (rank, index, merged))
            })
            .min_by_key(|(rank, _, _)| *rank);
        let Some((_, index, merged)) = candidate else {
            break;
        };
        parts.splice(index..=index + 1, [merged]);
    }
    parts
}

fn build_byte_bpe(
    ranked: &[Vec<u8>],
    pattern: &str,
    policy: MergePolicy,
) -> Result<Tokenizer, Error> {
    let encoder = byte_encoder();
    let ranks = ranked
        .iter()
        .cloned()
        .enumerate()
        .map(|(rank, token)| (token, rank))
        .collect::<HashMap<_, _>>();
    if ranks.len() != ranked.len() {
        return Err(tokenizer_error("duplicate base token bytes"));
    }
    if (0..=255u8).any(|byte| !ranks.contains_key(&vec![byte])) {
        return Err(tokenizer_error("base vocabulary does not cover every byte"));
    }
    let vocab = ranked
        .iter()
        .enumerate()
        .map(|(rank, token)| {
            Ok((
                encoded_token(token, &encoder),
                u32::try_from(rank)
                    .map_err(|_| tokenizer_error("vocabulary exceeds u32 token IDs"))?,
            ))
        })
        .collect::<Result<Vocab, Error>>()?;
    let merges = match policy {
        MergePolicy::RankedPair => ranked
            .iter()
            .enumerate()
            .filter(|(_, token)| token.len() > 1)
            .filter_map(|(rank, token)| {
                let parts = bpe_parts(&ranks, token, rank);
                (parts.len() == 2).then(|| {
                    (
                        encoded_token(&parts[0], &encoder),
                        encoded_token(&parts[1], &encoder),
                    )
                })
            })
            .collect(),
        MergePolicy::VocabularySplits => ranked
            .iter()
            .flat_map(|token| {
                let mut splits = (1..token.len())
                    .filter_map(|index| {
                        let left = &token[..index];
                        let right = &token[index..];
                        Some((*ranks.get(left)?, *ranks.get(right)?, left, right))
                    })
                    .collect::<Vec<_>>();
                splits.sort_by_key(|(left, right, _, _)| (*left, *right));
                splits
                    .into_iter()
                    .map(|(_, _, left, right)| {
                        (
                            encoded_token(left, &encoder),
                            encoded_token(right, &encoder),
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect(),
    };
    let mut tokenizer = Tokenizer::new(
        BPE::builder()
            .vocab_and_merges(vocab, merges)
            .ignore_merges(matches!(policy, MergePolicy::VocabularySplits))
            .build()?,
    );
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(Split::new(
            SplitPattern::Regex(pattern.into()),
            SplitDelimiterBehavior::Isolated,
            false,
        )?),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ])));
    tokenizer.with_decoder(Some(ByteLevelDecoder::new(false, false, true)));
    Ok(tokenizer)
}

fn special_tokens(config_path: &Path, base_count: usize) -> Result<Vec<AddedToken>, Error> {
    let config: TokenizerConfig = serde_json::from_slice(&fs::read(config_path)?)?;
    let special_count = 258usize;
    let end = base_count
        .checked_add(special_count)
        .ok_or_else(|| tokenizer_error("special-token ID range overflows"))?;
    let configured = config
        .added_tokens_decoder
        .into_iter()
        .map(|(id, token)| {
            let id = id
                .parse::<usize>()
                .map_err(|error| tokenizer_error(format!("invalid special-token ID: {error}")))?;
            if !(base_count..end).contains(&id) {
                return Err(tokenizer_error(format!(
                    "special-token ID {id} is outside {base_count}..{end}"
                )));
            }
            Ok((id, token.content))
        })
        .collect::<Result<HashMap<_, _>, Error>>()?;
    Ok((base_count..end)
        .map(|id| {
            AddedToken::from(
                configured
                    .get(&id)
                    .cloned()
                    .unwrap_or_else(|| format!("<|reserved_token_{id}|>")),
                true,
            )
            .normalized(false)
        })
        .collect())
}

/// Loads the official Kimi/K2 `tiktoken.model` and registers its complete
/// contiguous reserved-token range without permitting partial registration.
pub fn load_kimi_k2(model_dir: &Path) -> Result<Tokenizer, Error> {
    let ranked = read_ranks(&model_dir.join("tiktoken.model"))?;
    let mut tokenizer = build_byte_bpe(&ranked, KIMI_K2_PATTERN, MergePolicy::RankedPair)?;
    let tokens = special_tokens(&model_dir.join("tokenizer_config.json"), ranked.len())?;
    tokenizer.add_special_tokens(tokens)?;
    for id in ranked.len()..ranked.len() + 258 {
        if tokenizer.id_to_token(id as u32).is_none() {
            return Err(tokenizer_error(format!(
                "special token {id} was not registered atomically"
            )));
        }
    }
    Ok(tokenizer)
}

/// Reconstructs the published Muse Spark L4 byte-BPE tokenizer from
/// `l4_200k_base` and `tokenizer_config.json`. Only the 1,818 declared added
/// tokens are registered; unused model-output rows remain outside the vocabulary.
/// Chat prompts supply their own BOS token, as in the published fast tokenizer.
pub fn load_muse_spark(model_dir: &Path) -> Result<Tokenizer, Error> {
    let ranked = read_ranks(&model_dir.join("l4_200k_base"))?;
    if ranked.len() != 200_000 {
        return Err(tokenizer_error("Muse Spark requires 200000 base ranks"));
    }
    let mut tokenizer = build_byte_bpe(&ranked, LLAMA4_PATTERN, MergePolicy::VocabularySplits)?;
    let config: TokenizerConfig =
        serde_json::from_slice(&fs::read(model_dir.join("tokenizer_config.json"))?)?;
    register_complete_tokens(&mut tokenizer, config, 200_000, 1_818)?;
    Ok(tokenizer)
}

fn register_complete_tokens(
    tokenizer: &mut Tokenizer,
    config: TokenizerConfig,
    base_count: u32,
    count: u32,
) -> Result<(), Error> {
    let end = base_count + count;
    let mut configured = std::collections::BTreeMap::new();
    let mut contents = HashSet::new();
    for (id, token) in config.added_tokens_decoder {
        let id = id
            .parse::<u32>()
            .map_err(|_| tokenizer_error("invalid added-token ID"))?;
        if !(base_count..end).contains(&id) || configured.contains_key(&id) {
            return Err(tokenizer_error(format!("unexpected added-token ID {id}")));
        }
        if token.single_word
            || token.lstrip
            || token.rstrip
            || token.normalized
            || token.special == Some(false)
        {
            return Err(tokenizer_error(format!(
                "added-token flags at {id} differ from the Muse Spark contract"
            )));
        }
        if token.content.is_empty()
            || !contents.insert(token.content.clone())
            || tokenizer.token_to_id(&token.content).is_some()
        {
            return Err(tokenizer_error(format!(
                "duplicate or empty added token at {id}"
            )));
        }
        configured.insert(id, token.content);
    }
    let tokens = (base_count..end)
        .map(|id| {
            configured
                .get(&id)
                .cloned()
                .map(|token| AddedToken::from(token, true).normalized(false))
                .ok_or_else(|| tokenizer_error(format!("missing added-token ID {id}")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    tokenizer.add_special_tokens(tokens)?;
    for (id, token) in configured {
        if tokenizer.token_to_id(&token) != Some(id) {
            return Err(tokenizer_error(format!("added-token ID {id} was remapped")));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{bpe_parts, byte_encoder, encoded_token, load_kimi_k2};
    use std::collections::HashMap;

    #[test]
    fn incomplete_or_colliding_added_tokens_never_return_a_partial_tokenizer() {
        for added in [
            serde_json::json!({"0":{"content":"first"}}),
            serde_json::json!({"0":{"content":"same"},"1":{"content":"same"}}),
            serde_json::json!({"0":{"content":"first"},"3":{"content":"third"}}),
            serde_json::json!({"0":{"content":"first"},"1":{"content":"second","lstrip":true}}),
        ] {
            let config =
                serde_json::from_value(serde_json::json!({"added_tokens_decoder":added})).unwrap();
            let mut tokenizer =
                tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default());
            assert!(super::register_complete_tokens(&mut tokenizer, config, 0, 2).is_err());
            assert_eq!(tokenizer.get_vocab_size(true), 0);
        }
    }

    #[test]
    fn gpt2_byte_alphabet_is_bijective() {
        let encoder = byte_encoder();
        assert_eq!(
            encoder
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            256
        );
        assert_eq!(encoded_token(b"hello", &encoder), "hello");
        assert_eq!(encoder[0], 'Ā');
        assert_eq!(encoder[b' ' as usize], 'Ġ');
    }

    #[test]
    #[ignore = "requires KIMI_LINEAR_MODEL_DIR with tiktoken.model"]
    fn official_kimi_tokenizer_round_trips_representative_text() {
        let dir = std::env::var_os("KIMI_LINEAR_MODEL_DIR")
            .map(std::path::PathBuf::from)
            .expect("KIMI_LINEAR_MODEL_DIR");
        let tokenizer = load_kimi_k2(&dir).unwrap();
        for text in [
            "Hello 世界!",
            "I'm testing contractions, whitespace, and emoji 🚀.",
            "第一行\n\nSecond line 123456",
        ] {
            let encoding = tokenizer.encode(text, false).unwrap();
            assert_eq!(tokenizer.decode(encoding.get_ids(), false).unwrap(), text);
        }
        assert_eq!(tokenizer.token_to_id("[BOS]"), Some(163584));
        assert_eq!(tokenizer.token_to_id("[EOS]"), Some(163585));
        assert_eq!(tokenizer.token_to_id("<|im_end|>"), Some(163586));
        assert_eq!(tokenizer.token_to_id("[UNK]"), Some(163838));
        assert_eq!(tokenizer.token_to_id("[PAD]"), Some(163839));
    }

    #[test]
    fn derives_the_last_ranked_merge() {
        let ranks = HashMap::from([
            (b"a".to_vec(), 0),
            (b"b".to_vec(), 1),
            (b"c".to_vec(), 2),
            (b"ab".to_vec(), 3),
            (b"abc".to_vec(), 4),
        ]);
        assert_eq!(
            bpe_parts(&ranks, b"abc", 4),
            vec![b"ab".to_vec(), b"c".to_vec()]
        );
    }
}
