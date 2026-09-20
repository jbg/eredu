use super::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use tokenizers::{AddedToken, decoders::byte_level::ByteLevel, models::bpe::BPE};

fn byte_level() -> Tokenizer {
    let model = BPE::builder()
        .vocab_and_merges(
            [
                ("a".to_owned(), 0),
                ("b".to_owned(), 1),
                ("ab".to_owned(), 2),
            ],
            vec![("a".to_owned(), "b".to_owned())],
        )
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
}

fn assert_same_environment(left: &TokEnv, right: &TokEnv) {
    assert_eq!(left.tok_trie().info(), right.tok_trie().info());
    assert_eq!(left.tok_trie().eos_tokens(), right.tok_trie().eos_tokens());
    for id in 0..left.tok_trie().info().vocab_size {
        assert_eq!(left.tok_trie().token(id), right.tok_trie().token(id));
    }
}

#[test]
fn nonzero_merges_added_tokens_and_nested_prefix_configuration_roundtrip() {
    let mut raw = byte_level();
    raw.add_tokens([AddedToken::from("ordinary", false).normalized(false)])
        .unwrap();
    raw.add_special_tokens([
        AddedToken::from("<eos>", true).normalized(false),
        AddedToken::from("<|end|>", true).normalized(false),
    ])
    .unwrap();
    let mut configuration = serde_json::to_value(&raw).unwrap();
    configuration["normalizer"] = serde_json::json!({
        "type": "Sequence", "normalizers": [
            {"type": "Prepend", "prepend": "prefix"},
            {"type": "Sequence", "normalizers": [
                {"type": "Lowercase"}, {"type": "Prepend", "prepend": "nested"}
            ]}
        ]
    });
    let raw: Tokenizer = serde_json::from_value(configuration).unwrap();
    let tokenizer = ChatTokenizer::from_tokenizer(raw);
    let authority = HostPreparationAuthority::unmanaged();
    let frozen = freeze(&tokenizer, &[5, 4], &authority).unwrap();
    let restored = restore(&frozen.bytes, &[5, 4], &authority).unwrap();
    assert_same_environment(&frozen.environment, &restored);
    let normalized = restore_prepared(
        frozen.grammar.bytes(),
        restored.tok_trie().eos_tokens(),
        Some(restored.tok_trie().info()),
        &authority,
    )
    .unwrap();
    assert_same_environment(&restored, &normalized);
    for text in [b"AB".as_slice(), b"ordinary", "é".as_bytes(), b"a"] {
        assert_eq!(
            normalized.tokenize_bytes(text),
            restored.tokenize_bytes(text)
        );
    }
    let frozen_normalized = decode(frozen.grammar.bytes()).unwrap();
    assert_eq!(
        frozen_normalized.get_encode_special_tokens(),
        frozen.grammar.encode_special_tokens()
    );

    assert_eq!(restored.tokenize_bytes(b"AB"), vec![2]);
    assert_eq!(restored.tokenize_bytes(b"ordinary"), vec![3]);
    assert_eq!(restored.tok_trie().token(3), b"ordinary");
    assert_eq!(restored.tok_trie().token(4), b"\xff<eos>");
    assert_eq!(restored.tok_trie().eos_tokens(), &[5, 4]);
    let decoded = decode(&frozen.bytes).unwrap();
    assert_eq!(
        serde_json::to_value(decoded.get_normalizer()).unwrap(),
        serde_json::to_value(tokenizer.get_normalizer()).unwrap()
    );
    // Normalization applies to the runtime environment; the frozen source still
    // contains the original complete tokenizer settings, including its prefixes.
    assert_eq!(
        decode(&frozen.bytes).unwrap().get_vocab(true),
        tokenizer.get_vocab(true)
    );
}

#[test]
fn sparse_slots_and_multibyte_fallback_keep_exact_bytes_and_metaspace_behavior() {
    let model = BPE::builder()
        .vocab_and_merges(
            [
                ("a".to_owned(), 0),
                ("▁".to_owned(), 2),
                ("<0xC3>".to_owned(), 5),
                ("<0xA9>".to_owned(), 7),
                ("é".to_owned(), 8),
            ],
            Vec::new(),
        )
        .build()
        .unwrap();
    let mut configuration = serde_json::to_value(Tokenizer::new(model)).unwrap();
    configuration["decoder"] = serde_json::json!({"type":"Sequence", "decoders":[
        {"type":"Replace", "pattern":{"String":"▁"}, "content":" "},
        {"type":"ByteFallback"}
    ]});
    configuration["pre_tokenizer"] = serde_json::json!({"type":"Sequence", "pretokenizers":[
        {"type":"Metaspace", "replacement":"▁", "prepend_scheme":"always", "split":true}
    ]});
    let raw: Tokenizer = serde_json::from_value(configuration).unwrap();
    let tokenizer = ChatTokenizer::from_tokenizer(raw);
    let authority = HostPreparationAuthority::unmanaged();
    let frozen = freeze(&tokenizer, &[7, 0], &authority).unwrap();
    let restored = restore(&frozen.bytes, &[7, 0], &authority).unwrap();
    assert_same_environment(&frozen.environment, &restored);
    let normalized = restore_prepared(
        frozen.grammar.bytes(),
        restored.tok_trie().eos_tokens(),
        Some(restored.tok_trie().info()),
        &authority,
    )
    .unwrap();
    assert_same_environment(&restored, &normalized);
    for text in [b"AB".as_slice(), b"ordinary", "é".as_bytes(), b"a"] {
        assert_eq!(
            normalized.tokenize_bytes(text),
            restored.tokenize_bytes(text)
        );
    }
    let frozen_normalized = decode(frozen.grammar.bytes()).unwrap();
    assert_eq!(
        frozen_normalized.get_encode_special_tokens(),
        frozen.grammar.encode_special_tokens()
    );

    assert_eq!(restored.tok_trie().vocab_size(), 9);
    for absent in [1, 3, 4, 6] {
        assert!(restored.tok_trie().token(absent).is_empty());
    }
    assert_eq!(restored.tok_trie().token(2), b" ");
    assert_eq!(restored.tok_trie().token(5), &[0xc3]);
    assert_eq!(restored.tok_trie().token(7), &[0xa9]);
    assert_eq!(restored.tok_trie().token(8), "é".as_bytes());
    assert_eq!(restored.tokenize_bytes(b"a"), vec![0]);
    assert_eq!(restored.tokenize_bytes("é".as_bytes()), vec![8]);
    assert!(
        freeze(&tokenizer, &[1], &authority)
            .err()
            .unwrap()
            .contains("no consistent tokenizer mapping")
    );
}

#[test]
fn frozen_recipe_preserves_special_splitting_flag_omitted_by_upstream_serde() {
    let model = BPE::builder()
        .vocab_and_merges([("a".to_owned(), 0), ("b".to_owned(), 1)], Vec::new())
        .build()
        .unwrap();
    let mut raw = Tokenizer::new(model);
    raw.with_decoder(Some(ByteLevel::default()));
    raw.add_special_tokens([AddedToken::from("ab", true).normalized(false)])
        .unwrap();
    raw.set_encode_special_tokens(true);
    let tokenizer = ChatTokenizer::from_tokenizer(raw);
    let authority = HostPreparationAuthority::unmanaged();
    let frozen = freeze(&tokenizer, &[1], &authority).unwrap();
    assert!(decode(&frozen.bytes).unwrap().get_encode_special_tokens());
    let serialized_raw = serde_json::to_vec(&*tokenizer).unwrap();
    let upstream_restored: Tokenizer = serde_json::from_slice(&serialized_raw).unwrap();
    assert!(!upstream_restored.get_encode_special_tokens());
    assert_eq!(
        upstream_restored.encode("ab", false).unwrap().get_ids(),
        &[2]
    );
    assert_eq!(tokenizer.encode("ab", false).unwrap().get_ids(), &[0, 1]);
    let restored = restore(&frozen.bytes, &[1], &authority).unwrap();
    assert_same_environment(&frozen.environment, &restored);
    let normalized = restore_prepared(
        frozen.grammar.bytes(),
        restored.tok_trie().eos_tokens(),
        Some(restored.tok_trie().info()),
        &authority,
    )
    .unwrap();
    assert_same_environment(&restored, &normalized);
    for text in [b"AB".as_slice(), b"ordinary", "é".as_bytes(), b"a"] {
        assert_eq!(
            normalized.tokenize_bytes(text),
            restored.tokenize_bytes(text)
        );
    }
    let frozen_normalized = decode(frozen.grammar.bytes()).unwrap();
    assert_eq!(
        frozen_normalized.get_encode_special_tokens(),
        frozen.grammar.encode_special_tokens()
    );

    assert_eq!(
        restored.tokenize_bytes(b"ab"),
        tokenizer.encode("ab", false).unwrap().get_ids()
    );
}

#[test]
fn sparse_added_token_remapping_is_rejected_before_a_recipe_can_escape() {
    let model = BPE::builder()
        .vocab_and_merges([("a".to_owned(), 0), ("b".to_owned(), 5)], Vec::new())
        .build()
        .unwrap();
    let mut raw = Tokenizer::new(model);
    raw.with_decoder(Some(ByteLevel::default()));
    // Separate additions allocate c after the existing high-ID added token.
    // Upstream deserialization adds the complete list in a single batch and
    // starts new IDs from the model entry count, silently remapping c to 2.
    raw.add_special_tokens([AddedToken::from("b", true).normalized(false)])
        .unwrap();
    raw.add_tokens([AddedToken::from("c", false).normalized(false)])
        .unwrap();
    assert_eq!(raw.token_to_id("c"), Some(6));
    let tokenizer = ChatTokenizer::from_tokenizer(raw);
    let raw_json = serde_json::to_vec(&*tokenizer).unwrap();
    let upstream_restored: Tokenizer = serde_json::from_slice(&raw_json).unwrap();
    assert_eq!(upstream_restored.token_to_id("c"), Some(2));
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(DropWitness(drops.clone()));
    let error = freeze(&tokenizer, &[5], &authority).err().unwrap();
    assert!(
        error.contains("changed serialized configuration"),
        "{error}"
    );
    drop((tokenizer, authority));
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

struct DropWitness(Arc<AtomicUsize>);
impl Drop for DropWitness {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn escaped_environment_alias_retains_authority_after_source_and_recipe_retire() {
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(DropWitness(drops.clone()));
    let tokenizer = ChatTokenizer::from_tokenizer(byte_level());
    let frozen = freeze(&tokenizer, &[1], &authority).unwrap();
    let restored = restore(&frozen.bytes, &[1], &authority).unwrap();
    let alias = restored.clone();
    drop((frozen, tokenizer, restored, authority));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    assert_eq!(alias.tokenize_bytes(b"ab"), vec![2]);
    drop(alias);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn malformed_or_changed_recipe_fails_without_escaping_preparation_authority() {
    let drops = Arc::new(AtomicUsize::new(0));
    let authority = HostPreparationAuthority::retain(DropWitness(drops.clone()));
    let tokenizer = ChatTokenizer::from_tokenizer(byte_level());
    let frozen = freeze(&tokenizer, &[1], &authority).unwrap();
    assert!(
        restore(b"{", &[1], &authority)
            .err()
            .unwrap()
            .contains("failed to restore frozen tokenizer")
    );
    let mut changed: serde_json::Value = serde_json::from_slice(&frozen.bytes).unwrap();
    changed["version"] = serde_json::json!(RECIPE_VERSION + 1);
    assert!(
        restore(&serde_json::to_vec(&changed).unwrap(), &[1], &authority)
            .err()
            .unwrap()
            .contains("unsupported frozen tokenizer recipe version")
    );
    // A reconstructed tokenizer with changed settings cannot pass the source
    // compatibility check, even though its spelling/ID vocabulary is unchanged.
    let mut reconstructed = decode(&frozen.bytes).unwrap();
    reconstructed.set_encode_special_tokens(true);
    assert_eq!(
        token_id_vocabulary(&reconstructed),
        token_id_vocabulary(&*tokenizer)
    );
    assert!(
        validate_configuration(&*tokenizer, &reconstructed, &frozen.bytes)
            .unwrap_err()
            .contains("changed serialized configuration")
    );
    drop((frozen, tokenizer));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    drop(authority);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
}

#[test]
fn recipe_restores_with_cache_policy_in_any_key_order() {
    use eredu_text::tokenizer::ModelCachePolicy;
    let legacy = byte_level();
    let json = serde_json::to_vec(&legacy).unwrap();
    let raw = ModelCachePolicy::disabled().from_bytes(&json).unwrap();
    let authority = HostPreparationAuthority::unmanaged();
    let source = ChatTokenizer::from_tokenizer_with_cache_policy(raw, ModelCachePolicy::disabled());
    let frozen = freeze(&source, &[1], &authority).unwrap();
    let alias = source.snapshot();
    drop(source);
    assert_eq!(alias.model_cache_policy(), Some(ModelCachePolicy::disabled()));
    for bytes in [
        frozen.bytes.clone(),
        format!(r#"{{"tokenizer":{},"encode_special_tokens":false,"version":3,"cache_policy":{{"capacity":0}}}}"#, std::str::from_utf8(&json).unwrap()).into_bytes(),
        format!(r#"{{"cache_policy":{{"capacity":0}},"version":3,"encode_special_tokens":false,"tokenizer":{}}}"#, std::str::from_utf8(&json).unwrap()).into_bytes(),
    ] {
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.encode("ab", false).unwrap().get_ids(), &[2]);
        assert_eq!(decoded.clone().encode("ab", false).unwrap().get_ids(), &[2]);
        let header: RecipeHeader = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(header.policy().unwrap(), ModelCachePolicy::disabled());
        let env = restore(&bytes, &[1], &authority).unwrap();
        assert_eq!(env.tokenize_bytes(b"ab"), vec![2]);
        assert_same_environment(&frozen.environment, &env);
    }
    // Superseded recipes are not reconstructed through a compatibility policy.
    for version in [1, 2] {
        let mut old: serde_json::Value = serde_json::from_slice(&frozen.bytes).unwrap();
        old["version"] = serde_json::json!(version);
        assert!(
            decode(&serde_json::to_vec(&old).unwrap())
                .unwrap_err()
                .contains("unsupported frozen tokenizer recipe version")
        );
    }
    let mut missing: serde_json::Value = serde_json::from_slice(&frozen.bytes).unwrap();
    missing.as_object_mut().unwrap().remove("cache_policy");
    assert!(
        decode(&serde_json::to_vec(&missing).unwrap())
            .unwrap_err()
            .contains("cache_policy")
    );
}

#[test]
fn cache_policy_recipe_rejects_invalid_headers_before_hf_and_preserves_unigram() {
    use eredu_text::tokenizer::ModelCachePolicy;
    let raw = Tokenizer::new(
        tokenizers::models::unigram::Unigram::from(
            vec![
                ("<unk>".into(), -10.0),
                ("a".into(), -2.0),
                ("b".into(), -2.0),
                ("ab".into(), 0.0),
            ],
            Some(0),
            false,
        )
        .unwrap(),
    );
    let mut raw = raw;
    raw.with_decoder(Some(ByteLevel::default()));
    let source = ChatTokenizer::from_tokenizer_with_cache_policy(raw, ModelCachePolicy::disabled());
    let authority = HostPreparationAuthority::unmanaged();
    let frozen = freeze(&source, &[1], &authority).unwrap();
    let decoded = decode(&frozen.bytes).unwrap();
    let header: RecipeHeader = serde_json::from_slice(&frozen.bytes).unwrap();
    assert_eq!(header.policy().unwrap(), ModelCachePolicy::disabled());
    assert_eq!(decoded.encode("ab", false).unwrap().get_ids(), &[3]);
    assert_eq!(
        restore(&frozen.bytes, &[1], &authority)
            .unwrap()
            .tokenize_bytes(b"ab"),
        vec![3]
    );
    for bad in [
        r#"{"tokenizer":{"model":{"type":"BPE"}},"version":3,"encode_special_tokens":false}"#,
        r#"{"tokenizer":{},"version":3,"cache_policy":{},"encode_special_tokens":false}"#,
        r#"{"version":3,"cache_policy":{"capacity":0},"tokenizer":{},"cache_policy":{"capacity":12},"encode_special_tokens":false}"#,
        r#"{"version":3,"cache_policy":"unknown","tokenizer":{},"encode_special_tokens":false}"#,
    ] {
        let error = decode(bad.as_bytes()).unwrap_err();
        assert!(
            error.contains("cache_policy")
                || error.contains("cache policy")
                || error.contains("capacity")
                || error.contains("expected struct ModelCachePolicy"),
            "{error}"
        );
        assert!(!error.contains("Missing vocab"));
    }
}
