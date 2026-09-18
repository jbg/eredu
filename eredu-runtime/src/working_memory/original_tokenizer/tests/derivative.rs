use super::*;
use eredu_text::token_trie_storage::TokRxInfo;

#[test]
fn identity_prefix_normalization_keeps_exact_source_without_new_reservation() {
    let bytes = WorkingMemoryPool::tokenizer_required_bytes(&plan(INPUT)).unwrap();
    let pool = WorkingMemoryPool::new(bytes, 0).unwrap();
    let original = pool.compile_tokenizer(plan(INPUT)).unwrap();
    let normalized = original.input_prefix_normalized_source().unwrap();
    assert!(normalized.same_source(&original));
    assert!(normalized.tokenization_is_canonical());
    assert_eq!(pool.peak_bytes().unwrap(), bytes);
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), bytes);
    drop(normalized);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn checked_derivative_retains_exact_original_and_does_not_grant_equal_source_identity() {
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let input = PREFIX_INPUT.replace(r#""decoder":null"#,
        r#""decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"▁"},"content":" "}]}"#);
    let original = pool.compile_tokenizer(plan(&input)).unwrap();
    let unrelated = pool.compile_tokenizer(plan(&input)).unwrap();
    let derivative = original.input_prefix_normalized_source().unwrap();
    let retained = original.original_bytes() + derivative.original_bytes();
    assert!(!derivative.same_source(&original));
    assert!(derivative.matches_semantic_root(&original));
    assert!(!derivative.matches_semantic_root(&unrelated));
    assert!(!original.matches_semantic_root(&derivative));
    assert!(
        derivative
            .input_prefix_normalized_source()
            .unwrap()
            .same_source(&derivative)
    );
    let info = TokRxInfo::new(5, u32::MAX);
    let trie = derivative.compile_token_trie_source(&info, &[]).unwrap();
    assert!(trie.matches_semantic_tokenizer(&original));
    assert!(trie.matches_semantic_tokenizer(&derivative));
    assert!(!trie.matches_semantic_tokenizer(&unrelated));
    assert!(trie.validate(&pool, &original, &info, &[]).is_err());
    trie.validate(&pool, &derivative, &info, &[]).unwrap();
    let trie_bytes = trie.original_bytes();
    drop((original, unrelated, derivative));
    assert_eq!(pool.used_bytes().unwrap(), retained + trie_bytes);
    let peer = trie.clone();
    std::thread::scope(|scope| {
        scope.spawn(move || drop(trie));
        scope.spawn(move || drop(peer));
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

const PREFIX_INPUT: &str = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"x"},{"type":"NFC"}]},"pre_tokenizer":null,"post_processor":null,"decoder":null,"added_tokens":[{"id":3,"content":"é","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],"model":{"type":"BPE","vocab":{"a":0,"e":1,"́":2,"é":3,"x":4},"merges":[]}}"#;

#[test]
fn source_native_prefix_projection_refreshes_domain_and_retains_original_through_alias_retirement()
{
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let original = pool
        .compile_tokenizer(plan(PREFIX_INPUT).with_generation_domain().unwrap())
        .unwrap();
    let original_bytes = original.original_bytes();
    assert_eq!(original.spelling(3), Some("xé"));
    assert_eq!(original.generation_domain().unwrap().allows(3), false);
    let derived = original.input_prefix_normalized_source().unwrap();
    let derived_bytes = derived.original_bytes();
    assert!(!derived.same_source(&original));
    assert!(derived.matches_semantic_root(&original));
    assert_eq!(derived.spelling(3), Some("é"));
    assert_eq!(derived.token_id("é"), Some(3));
    assert_eq!(derived.generation_domain().unwrap().allows(3), true);
    let original_ids = eredu_text::tokenizer_storage::EncodeIdsPlan::prepare(
        &original.payload().model,
        "aa",
        false,
    )
    .unwrap()
    .encode()
    .unwrap();
    let derived_ids = eredu_text::tokenizer_storage::EncodeIdsPlan::prepare(
        &derived.payload().model,
        "aa",
        false,
    )
    .unwrap()
    .encode()
    .unwrap();
    assert_eq!(original_ids.ids(), &[4, 0, 0]);
    assert_eq!(derived_ids.ids(), &[0, 0]);
    assert!(
        derived
            .input_prefix_normalized_source()
            .unwrap()
            .same_source(&derived)
    );
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), original_bytes + derived_bytes);
    let other = derived.clone();
    std::thread::scope(|scope| {
        scope.spawn(move || drop(derived));
        scope.spawn(move || drop(other));
    });
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn source_native_prefix_budget_refusal_retains_original_without_new_destinations() {
    let measure = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let original = measure
        .compile_tokenizer(plan(PREFIX_INPUT).with_generation_domain().unwrap())
        .unwrap();
    let derived = original.input_prefix_normalized_source().unwrap();
    let required = original.original_bytes() + derived.original_bytes();
    drop((derived, original));
    let pool = WorkingMemoryPool::new(required - 1, 0).unwrap();
    let original = pool
        .compile_tokenizer(plan(PREFIX_INPUT).with_generation_domain().unwrap())
        .unwrap();
    let root_bytes = original.original_bytes();
    let failure = original.input_prefix_normalized_source().unwrap_err();
    assert!(matches!(
        failure.accounting_failure(),
        Some(WorkingMemoryError::BudgetExceeded { .. })
    ));
    assert_eq!(failure.retained_bytes(), 0);
    drop(original);
    assert_eq!(pool.used_bytes().unwrap(), root_bytes);
    drop(failure);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}

#[test]
fn nested_metaspace_prefix_source_keeps_original_trie_and_decoder_custody() {
    let input = r#"{"version":"1.0","truncation":null,"padding":null,"normalizer":{"type":"Sequence","normalizers":[{"type":"Prepend","prepend":"x"},{"type":"NFC"}]},"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Sequence","pretokenizers":[{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true}]}]},"post_processor":null,"decoder":{"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"String":"▁"},"content":" "}]},"added_tokens":[{"id":5,"content":"é","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},{"id":6,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"model":{"type":"BPE","vocab":{"a":0,"b":1,"▁":2,"▁a":3,"x":4,"é":5},"merges":[["▁","a"]]}}"#;
    let pool = WorkingMemoryPool::new(1 << 24, 0).unwrap();
    let source = pool
        .compile_tokenizer(plan(input).with_generation_domain().unwrap())
        .unwrap();
    let projected = source.input_prefix_normalized_source().unwrap();
    let info = TokRxInfo::new(7, 6);
    let trie = projected.compile_token_trie_source(&info, &[6]).unwrap();
    assert!(trie.matches_semantic_tokenizer(&source));
    trie.validate(&pool, &projected, &info, &[6]).unwrap();
    assert!(trie.validate(&pool, &source, &info, &[6]).is_err());
    for (id, bytes) in [
        b"a".as_slice(),
        b"b",
        b" ",
        b" a",
        b"x",
        "é".as_bytes(),
        b"\xff<S>",
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(trie.trie().token(id as u32), bytes);
    }
    for (text, original, derived) in [
        ("a b", &[2, 4, 0, 2, 1][..], &[0, 2, 1][..]),
        ("é", &[5][..], &[5][..]),
        ("<S>a", &[6, 4, 0][..], &[6, 0][..]),
    ] {
        let before = eredu_text::tokenizer_storage::EncodeIdsPlan::prepare(
            &source.payload().model,
            text,
            false,
        )
        .unwrap()
        .encode()
        .unwrap();
        let after = eredu_text::tokenizer_storage::EncodeIdsPlan::prepare(
            &projected.payload().model,
            text,
            false,
        )
        .unwrap()
        .encode()
        .unwrap();
        assert_eq!(before.ids(), original, "original {text:?}");
        assert_eq!(after.ids(), derived, "derived {text:?}");
    }
    let retained = source.original_bytes() + projected.original_bytes() + trie.original_bytes();
    drop((source, projected));
    assert_eq!(pool.used_bytes().unwrap(), retained);
    assert_eq!(trie.trie().token(5), "é".as_bytes());
    drop(trie);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
