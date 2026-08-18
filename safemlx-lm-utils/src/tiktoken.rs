//! Import rank-ordered tiktoken byte-BPE files into a fast tokenizer.

use std::{collections::HashMap, fs, path::Path};

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

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: HashMap<String, AddedTokenConfig>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
}

fn tokenizer_error(message: impl Into<String>) -> Error {
    Error::InvalidTokenizer(format!(
        "invalid tiktoken.model checkpoint: {}",
        message.into()
    ))
}

fn read_ranks(path: &Path) -> Result<Vec<Vec<u8>>, Error> {
    let contents = fs::read_to_string(path)?;
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

fn build_byte_bpe(ranked: &[Vec<u8>]) -> Result<Tokenizer, Error> {
    let encoder = byte_encoder();
    let ranks = ranked
        .iter()
        .cloned()
        .enumerate()
        .map(|(rank, token)| (token, rank))
        .collect::<HashMap<_, _>>();
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
    let merges = ranked
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
        .collect();
    let mut tokenizer = Tokenizer::new(BPE::builder().vocab_and_merges(vocab, merges).build()?);
    tokenizer.with_pre_tokenizer(Some(Sequence::new(vec![
        PreTokenizerWrapper::Split(Split::new(
            SplitPattern::Regex(KIMI_K2_PATTERN.into()),
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
    let mut tokenizer = build_byte_bpe(&ranked)?;
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

#[cfg(test)]
mod tests {
    use super::{bpe_parts, byte_encoder, encoded_token, load_kimi_k2};
    use std::collections::HashMap;

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
