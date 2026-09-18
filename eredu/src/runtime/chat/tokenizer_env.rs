use std::{collections::HashSet, sync::Arc};

use eredu_text::tokenizer::Tokenizer as ChatTokenizer;
use llguidance::toktrie::{TokEnv, TokRxInfo, TokTrie, TokenId, TokenizerEnv};
use tokenizers::{NormalizerWrapper, PreTokenizerWrapper, Tokenizer, normalizers, pre_tokenizers};

pub(crate) mod recipe;

struct HuggingFaceTokenEnv {
    tokenizer: Tokenizer,
    trie: TokTrie,
    // The tokenizer and trie must retire before their preparation authority.
    _authority: eredu_core::HostPreparationAuthority,
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
    from_tokenizer_with_authority(
        tokenizer,
        eos_token_ids,
        &eredu_core::HostPreparationAuthority::unmanaged(),
    )
}

pub(super) fn from_tokenizer_with_authority(
    tokenizer: &ChatTokenizer,
    eos_token_ids: &[u32],
    authority: &eredu_core::HostPreparationAuthority,
) -> Result<TokEnv, String> {
    from_raw((**tokenizer).clone(), eos_token_ids, authority)
}

fn from_raw(
    tokenizer: Tokenizer,
    eos_token_ids: &[u32],
    authority: &eredu_core::HostPreparationAuthority,
) -> Result<TokEnv, String> {
    from_raw_with_info(tokenizer, eos_token_ids, None, authority)
}

fn from_raw_with_info(
    mut tokenizer: Tokenizer,
    eos_token_ids: &[u32],
    selected_info: Option<&TokRxInfo>,
    authority: &eredu_core::HostPreparationAuthority,
) -> Result<TokEnv, String> {
    remove_input_prefixes(&mut tokenizer)?;
    from_prepared_raw_with_info(tokenizer, eos_token_ids, selected_info, authority)
}

// Only the ordinary prefix worker and its closed frozen source can reach this.
// It consumes the same tokenizer object, with no repeated prefix mutation.
fn from_prepared_raw_with_info(
    tokenizer: Tokenizer,
    eos_token_ids: &[u32],
    selected_info: Option<&TokRxInfo>,
    authority: &eredu_core::HostPreparationAuthority,
) -> Result<TokEnv, String> {
    let decoder = DecoderKind::inspect(&tokenizer)?;
    let vocabulary = eredu_text::tokenizer::token_id_vocabulary(&tokenizer);
    let vocab_size = vocabulary
        .last_key_value()
        .and_then(|(&id, _)| id.checked_add(1))
        .ok_or("tokenizer has an empty or unrepresentable token ID domain")?;
    if let Some(id) = eos_token_ids.iter().find(|id| !vocabulary.contains_key(id)) {
        return Err(format!(
            "EOS token ID {id} has no consistent tokenizer mapping"
        ));
    }

    let mut info = TokRxInfo::new(vocab_size, eos_token_ids.first().copied().unwrap_or(llguidance::toktrie::INVALID_TOKEN));
    let token_bytes = vocabulary_bytes(&tokenizer, &vocabulary, &decoder, &mut info)?;
    if let Some(&primary_eos) = eos_token_ids.first() {
        info.tok_eos = primary_eos;
    }

    if let Some(selected) = selected_info {
        if selected.vocab_size != vocab_size
            || eos_token_ids
                .first()
                .is_some_and(|id| *id != selected.tok_eos)
        {
            return Err("frozen trie metadata differs from its tokenizer or EOS source".into());
        }
        info = *selected;
    }

    let mut trie = TokTrie::from(&info, &token_bytes);
    if eos_token_ids.len() > 1 {
        trie = trie.with_eos_tokens(eos_token_ids);
    }
    Ok(Arc::new(HuggingFaceTokenEnv {
        tokenizer,
        trie,
        _authority: authority.clone(),
    }))
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

struct DecoderKind(eredu_text::token_bytes::TokenByteEncoding);

impl DecoderKind {
    fn inspect(tokenizer: &Tokenizer) -> Result<Self, String> {
        eredu_text::token_bytes::TokenByteEncoding::inspect(tokenizer.get_decoder())
            .map(Self)
            .map_err(|_| {
                format!(
                    "cannot determine byte encoding from tokenizer decoder {:?}",
                    tokenizer.get_decoder()
                )
            })
    }

    fn token_bytes(&self, token: &str, special: bool) -> Result<Vec<u8>, String> {
        let failure = |cause: eredu_text::token_bytes::TokenByteError| match cause {
            eredu_text::token_bytes::TokenByteError::Fallback(Some(error)) => {
                format!("invalid byte-fallback token {token:?}: {error}")
            }
            _ => format!("invalid byte-fallback token {token:?}"),
        };
        let mut bytes = vec![0; self.0.trie_token_len(token, special).map_err(failure)?];
        self.0
            .write_trie_token(token, special, &mut bytes)
            .map_err(failure)?;
        Ok(bytes)
    }
}

fn vocabulary_bytes(
    tokenizer: &Tokenizer,
    vocabulary: &std::collections::BTreeMap<u32, String>,
    decoder: &DecoderKind,
    info: &mut TokRxInfo,
) -> Result<Vec<Vec<u8>>, String> {
    let mut token_bytes = vec![Vec::new(); info.vocab_size as usize];
    let mut special_ids = HashSet::new();

    for (id, added) in tokenizer.get_added_tokens_decoder() {
        if !vocabulary.contains_key(&id) {
            continue;
        }
        // Ordinary added tokens participate in text grammar literals and regexes.
        // In particular, promoting an ordinary </think> to a special token makes
        // it unreachable from a reasoning grammar that names its text spelling.
        // Dialects can still refer to any explicitly structural token by ID.
        if added.special {
            apply_special_metadata(info, id, &added.content);
            special_ids.insert(id);
        }
    }

    // Empty slots are absent from TokTrie, not invented tokenizer entries.
    for (&id, token) in vocabulary {
        token_bytes[id as usize] = decoder.token_bytes(token, special_ids.contains(&id))?;
    }

    Ok(token_bytes)
}

/// Shared metadata policy for ordinary and retained original tokenizer sources.
/// EOS is selected only by the exact configured list. Multiple recognized
/// spellings use the highest canonical ID independently of hash iteration order.
pub(super) fn apply_special_metadata(info: &mut TokRxInfo, id: TokenId, spelling: &str) {
    let selected = match spelling {
        "<|end|>" | "<|eot_id|>" | "<|im_end|>" => &mut info.tok_end_of_turn,
        "<unk>" | "<|unk|>" => &mut info.tok_unk,
        "<pad>" | "<|pad|>" => &mut info.tok_pad,
        _ => return,
    };
    *selected = Some(selected.map_or(id, |previous| previous.max(id)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::{AddedToken, decoders::byte_level::ByteLevel, models::bpe::BPE};

    #[test]
    fn sparse_ids_keep_their_positions_and_unmapped_eos_is_rejected() {
        let model = BPE::builder()
            .vocab_and_merges([("a".to_owned(), 0), ("b".to_owned(), 5)], Vec::new())
            .build()
            .unwrap();
        let mut raw = Tokenizer::new(model);
        raw.with_decoder(Some(ByteLevel::default()));
        let tokenizer = ChatTokenizer::from_tokenizer(raw);
        let environment = from_tokenizer(&tokenizer, &[5]).unwrap();
        assert_eq!(environment.tok_trie().vocab_size(), 6);
        assert_eq!(environment.tok_trie().token(5), b"b");
        assert!(environment.tok_trie().token(1).is_empty());
        assert!(
            from_tokenizer(&tokenizer, &[1])
                .err()
                .unwrap()
                .contains("no consistent tokenizer mapping")
        );
    }

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
    #[test]
    fn borrowed_packed_vocabulary_matches_actual_trie_for_sparse_special_and_fallback_bytes() {
        for (decoder, vocabulary) in [
            (
                serde_json::json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}),
                serde_json::json!({"a":0,"Ġb":2,"Ã©":4,"z":7,"<|tool|>":9}),
            ),
            (
                serde_json::json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"▁"},"content":" "}]}),
                serde_json::json!({"a":0,"<0xFF>":2,"é▁🙂":4,"z":7,"<|tool|>":9}),
            ),
        ] {
            let config = serde_json::json!({"version":"1.0","truncation":null,"padding":null,
                "normalizer":null,"pre_tokenizer":null,"post_processor":null,
                "decoder":decoder,"added_tokens":[],
                "model":{"type":"BPE","vocab":vocabulary,"merges":[]}});
            let mut raw = Tokenizer::from_bytes(config.to_string().as_bytes()).unwrap();
            raw.add_special_tokens([AddedToken::from("<|tool|>", true).normalized(false)])
                .unwrap();
            // Reserve its real ID before marking it special: allocating a new
            // added token from sparse vocabulary length can shadow existing ID 4.
            assert_eq!(raw.token_to_id("<|tool|>"), Some(9));
            assert_eq!(raw.id_to_token(9).as_deref(), Some("<|tool|>"));
            let frozen = raw.to_string(false).unwrap();
            let plan =
                eredu_text::tokenizer_storage::TokenizerPlan::prepare_json(frozen.as_bytes());
            let source = plan.unwrap().compile().unwrap();
            let ordinary = from_tokenizer(&ChatTokenizer::from_tokenizer(raw), &[7]).unwrap();
            let trie_plan = source.token_trie_vocabulary().unwrap();
            let mut trie_packed = vec![0; trie_plan.packed_bytes()];
            let trie_view = trie_plan.write_tokens(&mut trie_packed).unwrap();
            assert_eq!(trie_view.token_count(), ordinary.tok_trie().vocab_size());
            for token in 0..trie_view.token_count() {
                assert_eq!(
                    trie_view.token(token),
                    Some(ordinary.tok_trie().token(token as u32))
                );
            }
            assert_eq!(trie_view.token(trie_view.token_count()), None);
            assert_eq!(trie_view.token(9), Some(b"\xff<|tool|>".as_slice()));
            let plan = source.token_byte_vocabulary().unwrap();
            assert_eq!(plan.token_count(), ordinary.tok_trie().vocab_size());
            let mut packed = vec![0; plan.packed_bytes()];
            plan.write(&mut packed).unwrap();
            for token in 0..plan.token_count() {
                let bytes = ordinary.tok_trie().token(token as u32);
                let expected = bytes
                    .strip_prefix(&[TokTrie::SPECIAL_TOKEN_MARKER])
                    .unwrap_or(bytes);
                assert_eq!(
                    eredu_core::speculative::packed_controller_token_bytes(
                        &packed,
                        plan.token_count(),
                        plan.maximum_token_bytes(),
                        token
                    ),
                    Some(expected)
                );
            }
            let mut wrong = vec![91; plan.packed_bytes() - 1];
            assert!(matches!(
                plan.write(&mut wrong),
                Err(eredu_text::token_bytes::TokenByteError::Destination)
            ));
            assert!(wrong.iter().all(|byte| *byte == 91));
            assert!(plan.control_bytes().unwrap() > 0);
        }
    }
}
