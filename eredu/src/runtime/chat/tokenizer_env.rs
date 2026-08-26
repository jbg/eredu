use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use tokenizers::{
    normalizers, pre_tokenizers, DecoderWrapper, NormalizerWrapper, PreTokenizerWrapper, Tokenizer,
};

struct HuggingFaceTokenEnv {
    tokenizer: Tokenizer,
    trie: TokTrie,
}

impl TokenizerEnv for HuggingFaceTokenEnv {
    fn tok_trie(&self) -> &TokTrie {
        &self.trie
    }

    fn tokenize_bytes(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            self.tokenizer
                .encode(text, false)
                .expect("tokenizer rejected valid UTF-8")
                .get_ids()
                .to_vec()
        })
    }

    fn tokenize_bytes_special(&self, bytes: &[u8]) -> Vec<TokenId> {
        self.trie.tokenize_with_greedy_fallback(bytes, |text| {
            self.trie.tokenize_with_special(text, |text| {
                self.tokenizer
                    .encode(text, false)
                    .expect("tokenizer rejected valid UTF-8")
                    .get_ids()
                    .to_vec()
            })
        })
    }
}

pub(super) fn from_tokenizer(
    tokenizer: &ChatTokenizer,
    eos_token_ids: &[u32],
) -> Result<TokEnv, String> {
    let mut tokenizer = (**tokenizer).clone();
    remove_input_prefixes(&mut tokenizer)?;

    let decoder = DecoderKind::inspect(&tokenizer)?;
    let vocab_size = tokenizer.get_vocab_size(true) as u32;
    if let Some(id) = eos_token_ids.iter().find(|&&id| id >= vocab_size) {
        return Err(format!(
            "EOS token ID {id} is outside tokenizer vocabulary {vocab_size}"
        ));
    }

    let mut info = TokRxInfo::new(vocab_size, 0);
    let token_bytes = vocabulary_bytes(&tokenizer, &decoder, &mut info)?;
    if let Some(&primary_eos) = eos_token_ids.first() {
        info.tok_eos = primary_eos;
    }

    let mut trie = TokTrie::from(&info, &token_bytes);
    if eos_token_ids.len() > 1 {
        trie = trie.with_eos_tokens(eos_token_ids);
    }
    Ok(Arc::new(HuggingFaceTokenEnv { tokenizer, trie }))
}

fn remove_input_prefixes(tokenizer: &mut Tokenizer) -> Result<(), String> {
    fn without_prepend(normalizer: NormalizerWrapper) -> Option<NormalizerWrapper> {
        match normalizer {
            NormalizerWrapper::Prepend(_) => None,
            NormalizerWrapper::Sequence(sequence) => {
                let members = sequence
                    .as_ref()
                    .iter()
                    .cloned()
                    .filter_map(without_prepend)
                    .collect::<Vec<_>>();
                (!members.is_empty())
                    .then(|| NormalizerWrapper::Sequence(normalizers::Sequence::new(members)))
            }
            other => Some(other),
        }
    }

    fn without_metaspace_prefix(pre_tokenizer: PreTokenizerWrapper) -> PreTokenizerWrapper {
        match pre_tokenizer {
            PreTokenizerWrapper::Metaspace(mut metaspace) => {
                metaspace.prepend_scheme = pre_tokenizers::metaspace::PrependScheme::Never;
                PreTokenizerWrapper::Metaspace(metaspace)
            }
            PreTokenizerWrapper::Sequence(sequence) => {
                PreTokenizerWrapper::Sequence(pre_tokenizers::sequence::Sequence::new(
                    sequence
                        .as_ref()
                        .iter()
                        .cloned()
                        .map(without_metaspace_prefix)
                        .collect(),
                ))
            }
            other => other,
        }
    }

    if let Some(normalizer) = tokenizer.get_normalizer().cloned() {
        tokenizer
            .with_normalizer(without_prepend(normalizer))
            .map_err(|error| format!("failed to remove tokenizer input prefix: {error}"))?;
    }
    if let Some(pre_tokenizer) = tokenizer.get_pre_tokenizer().cloned() {
        tokenizer.with_pre_tokenizer(Some(without_metaspace_prefix(pre_tokenizer)));
    }
    Ok(())
}

enum DecoderKind {
    ByteLevel(HashMap<char, u8>),
    ByteFallback { space_marker: char },
}

impl DecoderKind {
    fn inspect(tokenizer: &Tokenizer) -> Result<Self, String> {
        #[derive(Default)]
        struct DecoderParts {
            byte_level: bool,
            byte_fallback: bool,
            space_marker: Option<char>,
        }

        fn visit(decoder: &DecoderWrapper, parts: &mut DecoderParts) -> Result<(), String> {
            match decoder {
                DecoderWrapper::ByteLevel(_) => parts.byte_level = true,
                DecoderWrapper::ByteFallback(_) => parts.byte_fallback = true,
                DecoderWrapper::Replace(replace) if replace.content == " " => {
                    let value = serde_json::to_value(replace).map_err(|error| {
                        format!("failed to inspect replacement decoder: {error}")
                    })?;
                    if let Some(pattern) = value["pattern"]["String"].as_str() {
                        let mut chars = pattern.chars();
                        if let (Some(marker), None) = (chars.next(), chars.next()) {
                            parts.space_marker = Some(marker);
                        }
                    }
                }
                DecoderWrapper::Sequence(sequence) => {
                    for member in sequence.get_decoders() {
                        visit(member, parts)?;
                    }
                }
                _ => {}
            }
            Ok(())
        }

        let mut parts = DecoderParts::default();
        if let Some(decoder) = tokenizer.get_decoder() {
            visit(decoder, &mut parts)?;
        }
        if parts.byte_fallback {
            Ok(Self::ByteFallback {
                space_marker: parts.space_marker.unwrap_or(' '),
            })
        } else if parts.byte_level {
            Ok(Self::ByteLevel(byte_level_alphabet()))
        } else {
            Err(format!(
                "cannot determine byte encoding from tokenizer decoder {:?}",
                tokenizer.get_decoder()
            ))
        }
    }

    fn token_bytes(&self, token: &str) -> Result<Vec<u8>, String> {
        match self {
            Self::ByteLevel(alphabet) => token
                .chars()
                .map(|character| {
                    alphabet.get(&character).copied().ok_or_else(|| {
                        format!(
                            "byte-level token {token:?} contains unmapped character {character:?}"
                        )
                    })
                })
                .collect(),
            Self::ByteFallback { space_marker } => {
                if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
                    u8::from_str_radix(&token[3..5], 16)
                        .map(|byte| vec![byte])
                        .map_err(|error| format!("invalid byte-fallback token {token:?}: {error}"))
                } else if token.starts_with("<0x") {
                    Err(format!("invalid byte-fallback token {token:?}"))
                } else {
                    Ok(token.replace(*space_marker, " ").into_bytes())
                }
            }
        }
    }
}

fn vocabulary_bytes(
    tokenizer: &Tokenizer,
    decoder: &DecoderKind,
    info: &mut TokRxInfo,
) -> Result<Vec<Vec<u8>>, String> {
    let mut token_bytes = vec![Vec::new(); info.vocab_size as usize];
    let mut special_ids = HashSet::new();

    for (id, added) in tokenizer.get_added_tokens_decoder() {
        let bracketed = added.content.starts_with('<') && added.content.ends_with('>');
        if added.special || bracketed {
            match added.content.as_str() {
                "</s>"
                | "<|endoftext|>"
                | "<|end_of_text|>"
                | "<｜end▁of▁sentence｜>"
                | "<eos>" => info.tok_eos = id,
                "<|end|>" | "<|eot_id|>" | "<|im_end|>" => info.tok_end_of_turn = Some(id),
                "<unk>" | "<|unk|>" => info.tok_unk = Some(id),
                "<pad>" | "<|pad|>" => info.tok_pad = Some(id),
                _ => {}
            }
            special_ids.insert(id);
        }
    }

    for id in 0..info.vocab_size {
        let token = tokenizer
            .id_to_token(id)
            .ok_or_else(|| format!("tokenizer vocabulary has no token for ID {id}"))?;
        token_bytes[id as usize] = if special_ids.contains(&id) {
            let mut bytes = Vec::with_capacity(token.len() + 1);
            bytes.push(TokTrie::SPECIAL_TOKEN_MARKER);
            bytes.extend_from_slice(token.as_bytes());
            bytes
        } else {
            decoder.token_bytes(&token)?
        };
    }

    Ok(token_bytes)
}

fn byte_level_alphabet() -> HashMap<char, u8> {
    let mut alphabet = HashMap::with_capacity(256);
    let mut escaped = 0x100;
    for byte in 0..=u8::MAX {
        let character = byte as char;
        if matches!(character, '!'..='~' | '\u{00a1}'..='\u{00ac}' | '\u{00ae}'..='\u{00ff}') {
            alphabet.insert(character, byte);
        } else {
            alphabet.insert(
                char::from_u32(escaped).expect("valid byte-level scalar"),
                byte,
            );
            escaped += 1;
        }
    }
    alphabet
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{decoders::byte_level::ByteLevel, models::bpe::BPE, AddedToken};

    #[test]
    fn current_hugging_face_tokenizer_builds_toktrie_without_serialization() {
        let model = BPE::builder()
            .vocab_and_merges([("a".to_owned(), 0), ("Ġb".to_owned(), 1)], Vec::new())
            .build()
            .unwrap();
        let mut raw = Tokenizer::new(model);
        raw.with_decoder(Some(ByteLevel::new(false, false, true)));
        assert_eq!(
            raw.add_special_tokens([AddedToken::from("<|end|>", true).normalized(false)])
                .unwrap(),
            1
        );
        let tokenizer = ChatTokenizer::from_tokenizer(raw);

        let environment = from_tokenizer(&tokenizer, &[2]).unwrap();

        assert_eq!(environment.tok_trie().token(0), b"a");
        assert_eq!(environment.tok_trie().token(1), b" b");
        assert_eq!(environment.tok_trie().token(2), b"\xff<|end|>");
        assert_eq!(environment.tok_trie().eos_tokens(), &[2]);
    }
}
