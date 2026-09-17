use super::*;
use crate::{Model, ModelCachePolicy, Tokenizer, Trainer};
use std::error::Error as _;

const HELLO: &str = r#"{"type":"BPE","vocab":{"h":0,"e":1,"l":2,"o":3,"he":4,"hel":5,"hell":6,"hello":7},"merges":[["h","e"],["he","l"],["hel","l"],["hell","o"]]}"#;
const CONFIGURED: &str = r###"{"type":"BPE","vocab":{"a":0,"##b</w>":1,"ab</w>":2,"[UNK]":3},"merges":[["a","##b</w>"]],"unk_token":"[UNK]","continuing_subword_prefix":"##","end_of_word_suffix":"</w>","fuse_unk":true}"###;

fn packed(input: &str) -> BPE {
    BpeCompilePlan::prepare_model_json(input.as_bytes())
        .unwrap()
        .compile()
        .unwrap()
}
fn legacy(input: &str) -> BPE {
    BPE::deserialize_with_cache_policy(
        &mut serde_json::Deserializer::from_str(input),
        ModelCachePolicy::NoModelCaches,
    )
    .unwrap()
}
fn parity(input: &str, words: &[&str]) -> BPE {
    let model = packed(input);
    let reference = legacy(input);
    assert!(!model.cache_is_enabled());
    assert_eq!(model, reference);
    assert_eq!(model.get_vocab(), reference.get_vocab());
    for word in words {
        assert_eq!(
            model.tokenize(word).unwrap(),
            reference.tokenize(word).unwrap()
        );
    }
    model
}
fn capacities(model: &BPE) -> [usize; 7] {
    let Storage::Packed(p) = &model.storage else {
        panic!("packed model")
    };
    [
        p.entries.capacity(),
        p.bytes.capacity(),
        p.ids.capacity(),
        p.merges.capacity(),
        model.unk_token.as_ref().map_or(0, String::capacity),
        model
            .continuing_subword_prefix
            .as_ref()
            .map_or(0, String::capacity),
        model
            .end_of_word_suffix
            .as_ref()
            .map_or(0, String::capacity),
    ]
}

#[test]
fn packed_model_uses_existing_nonzero_merge_engine_and_offsets() {
    let model = parity(
        HELLO,
        &["", "h", "he", "hell", "hello", "hellohello", "olé"],
    );
    let tokens = model.tokenize("hellohello").unwrap();
    assert_eq!(tokens.iter().map(|t| t.id).collect::<Vec<_>>(), [7, 7]);
    assert_eq!(
        tokens.iter().map(|t| t.offsets).collect::<Vec<_>>(),
        [(0, 5), (5, 10)]
    );
    assert_eq!(tokens[0].value, "hello");
}

#[test]
fn decoded_duplicate_spelling_is_last_wins_and_sparse_ids_remain_borrowed() {
    let input = r#"{"vocab":{"a":1,"b":2,"ab":3,"\u0061":7,"\uD83D\uDE03":4294967295},"merges":[["a","b"]]}"#;
    let plan = BpeCompilePlan::prepare_model_json(input.as_bytes()).unwrap();
    assert_eq!(plan.requirements().vocabulary_slots(), 5);
    assert_eq!(plan.requirements().spelling_bytes(), 9);
    let model = parity(input, &["a", "ab", "😃ab"]);
    assert_eq!(model.token_to_id("a"), Some(7));
    assert_eq!(model.id_to_token(1), None);
    assert_eq!(model.get_vocab_size(), 4);
    let tokenizer = Tokenizer::new(model);
    let view = tokenizer.decode_vocabulary();
    let mut ids: Vec<_> = view.ids().collect();
    ids.sort_unstable();
    assert_eq!(ids, [2, 3, 7, u32::MAX]);
    assert_eq!(view.id_to_token(u32::MAX), Some("😃"));
    assert_eq!(view.id_to_token(1), None);
}

#[test]
fn repeated_merge_pair_keeps_last_rank_after_legacy_comments() {
    let modern = r#"{"vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"abc":5},"merges":[["a","b"],["b","c"],["a","b"],["a","bc"]]}"#;
    let old = r##"{"vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"abc":5},"merges":["#version: 0.2","a b","b c","#version ignored","a b","a bc"]}"##;
    let a = parity(modern, &["abc", "ab", "bc", "abcabc"]);
    let b = parity(old, &["abc", "ab", "bc", "abcabc"]);
    assert_eq!(a, b);
    assert_eq!(a.tokenize("abc").unwrap()[0].id, 5);
    assert_eq!(a.tokenize("abc").unwrap().len(), 1);
    assert_eq!(
        BpeCompilePlan::prepare_model_json(old.as_bytes())
            .unwrap()
            .requirements()
            .merge_slots(),
        6
    );
}

#[test]
fn prefixes_suffixes_unknown_fusion_fallback_and_ignore_merges_match_legacy() {
    let configured = parity(CONFIGURED, &["", "ab", "ac", "zz", "abac"]);
    assert_eq!(configured.tokenize("ab").unwrap()[0].id, 2);
    let fallback = r#"{"vocab":{"a":0,"<0xC3>":1,"<0xA9>":2,"[UNK]":3},"merges":[],"unk_token":"[UNK]","byte_fallback":true,"fuse_unk":true}"#;
    let fallback = parity(fallback, &["é", "aé", "xyz", "éé"]);
    assert_eq!(
        fallback
            .tokenize("é")
            .unwrap()
            .iter()
            .map(|t| t.id)
            .collect::<Vec<_>>(),
        [1, 2]
    );
    let direct = r#"{"vocab":{"a":0,"b":1,"ab":2},"merges":[],"ignore_merges":true}"#;
    assert_eq!(
        parity(direct, &["ab", "aba"]).tokenize("ab").unwrap()[0].id,
        2
    );
    let spaces = r#"{"vocab":{"a":0," ":1,"a ":2,"b":3,"a b":4},"merges":[["a"," "],["a ","b"]]}"#;
    assert_eq!(
        parity(spaces, &["a b", "a ba b"]).tokenize("a b").unwrap()[0].id,
        4
    );
    let empty = r#"{"vocab":{"":0,"a":1},"merges":[],"unk_token":"","continuing_subword_prefix":"","end_of_word_suffix":""}"#;
    let empty = parity(empty, &["a", "b", "ab"]);
    assert_eq!(empty.unk_token.as_deref(), Some(""));
    assert_eq!(empty.continuing_subword_prefix.as_deref(), Some(""));
    assert_eq!(empty.end_of_word_suffix.as_deref(), Some(""));
}

#[test]
fn all_seven_capacity_overflow_frontiers_retain_actual_prior_destinations() {
    let q = BpeCompilePlan::prepare_model_json(CONFIGURED.as_bytes())
        .unwrap()
        .requirements();
    let expected = [
        q.slots,
        q.bytes,
        q.slots,
        q.merges,
        q.strings[0],
        q.strings[1],
        q.strings[2],
    ];
    assert!(expected.iter().all(|&n| n > 0));
    let mut failures = Vec::new();
    for stage in 0..7 {
        let mut plan = BpeCompilePlan::prepare_model_json(CONFIGURED.as_bytes()).unwrap();
        plan.fail_at = Some(stage);
        let failure = plan.compile().unwrap_err();
        assert!(failure.source_error().is_none());
        assert!(failure.allocation_error().is_some());
        assert!(failure
            .source()
            .unwrap()
            .downcast_ref::<TryReserveError>()
            .is_some());
        let actual = failure.buffer_capacities();
        assert_eq!(&actual[..stage], &expected[..stage]);
        assert!(actual[stage..].iter().all(|&n| n == 0));
        failures.push(failure);
    }
    // Keep every genuine capacity-overflow error and its partial buffers alive
    // together. The consumed plans expose neither retries nor partial extraction.
    for (stage, failure) in failures.iter().enumerate() {
        assert_eq!(&failure.buffer_capacities()[..stage], &expected[..stage]);
    }
    assert_eq!(capacities(&packed(CONFIGURED)), expected);
}

#[test]
fn post_allocation_semantic_failures_keep_the_complete_destination_population() {
    for (input, kind) in [
        (r#"{"vocab":{"a":0,"b":0},"merges":[]}"#, Kind::AmbiguousId),
        (
            r#"{"vocab":{"a":0,"b":1},"merges":[["a","x"]]}"#,
            Kind::MissingMergeToken,
        ),
        (
            r#"{"vocab":{"a":0,"b":1},"merges":[["a","b"]]}"#,
            Kind::MissingMergeToken,
        ),
        (
            r#"{"vocab":{"a":0,"é":1},"merges":[["a","é"]],"continuing_subword_prefix":"x"}"#,
            Kind::PrefixBoundary,
        ),
    ] {
        let plan = BpeCompilePlan::prepare_model_json(input.as_bytes()).unwrap();
        let q = plan.requirements();
        let failure = plan.compile().unwrap_err();
        assert_eq!(failure.source_error().unwrap().kind, kind);
        assert!(failure.allocation_error().is_none());
        assert!(failure
            .source()
            .unwrap()
            .downcast_ref::<BpeCompileError>()
            .is_some());
        assert_eq!(
            failure.buffer_capacities(),
            [
                q.slots,
                q.bytes,
                q.slots,
                q.merges,
                q.strings[0],
                q.strings[1],
                q.strings[2]
            ]
        );
        assert_eq!(failure.allocated_bytes(), Some(q.buffer_bytes()));
    }
}

#[test]
fn planning_rejects_invalid_and_unfinished_profiles_without_constructing_a_model() {
    for (input, kind) in [
        (r#"{"vocab":{},"merges":[],}"#, Kind::InvalidJson),
        (r#"{"vocab":{"\uD800":0},"merges":[]}"#, Kind::InvalidJson),
        (r#"{"vocab":{"\uDC00":0},"merges":[]}"#, Kind::InvalidJson),
        (r#"{"vocab":{"a":01},"merges":[]}"#, Kind::InvalidJson),
        (r#"{"vocab":{"a":1.0},"merges":[]}"#, Kind::InvalidId),
        (r#"{"vocab":{"a":-1},"merges":[]}"#, Kind::InvalidId),
        (r#"{"vocab":{"a":4294967296},"merges":[]}"#, Kind::InvalidId),
        (
            r#"{"vocab":{},"merges":[],"dropout":0}"#,
            Kind::DropoutProfile,
        ),
        (
            r#"{"vocab":{},"merges":[],"dropout":0.5}"#,
            Kind::DropoutProfile,
        ),
        (
            r#"{"vocab":{},"merges":[],"normalizer":null}"#,
            Kind::UnsupportedField,
        ),
        (
            r#"{"model":{"vocab":{},"merges":[]}}"#,
            Kind::UnsupportedField,
        ),
        (
            r#"{"vocab":{},"merges":[],"v\u006fcab":{}}"#,
            Kind::DuplicateField,
        ),
        (r#"{"type":"Unigram","vocab":{},"merges":[]}"#, Kind::NotBpe),
        (r#"{"vocab":{}}"#, Kind::MissingField),
        (
            r#"{"vocab":{},"merges":[["a","b"],"a b"]}"#,
            Kind::InvalidMerge,
        ),
        (
            r#"{"vocab":{},"merges":[["a","b","c"]]}"#,
            Kind::InvalidMerge,
        ),
        (r#"{"vocab":{},"merges":["a  b"]}"#, Kind::InvalidMerge),
    ] {
        assert_eq!(
            BpeCompilePlan::prepare_model_json(input.as_bytes())
                .unwrap_err()
                .kind,
            kind,
            "{input}"
        );
    }
    assert_eq!(
        BpeCompilePlan::prepare_model_json(&[0xff])
            .unwrap_err()
            .kind,
        Kind::InvalidUtf8
    );
    let deep = format!(
        "{}0{}",
        "[".repeat(json::DEPTH + 1),
        "]".repeat(json::DEPTH + 1)
    );
    assert_eq!(
        BpeCompilePlan::prepare_model_json(deep.as_bytes())
            .unwrap_err()
            .kind,
        Kind::DepthLimit
    );
    for (n, b, m, c) in [
        (usize::MAX, 0, 0, [0; 3]),
        (0, usize::MAX, 0, [0; 3]),
        (0, 0, usize::MAX, [0; 3]),
        (0, 0, 0, [usize::MAX, 0, 0]),
    ] {
        assert_eq!(
            requirements(n, b, m, c, 0).unwrap_err().kind,
            Kind::Overflow
        );
    }
}

#[test]
fn escaped_strings_field_order_and_fixed_parser_depth_preserve_values() {
    let input = r#"{"merges":[],"fuse_unk":null,"vocab":{"\"\\\/\b\f\n\r\t":0,"\u0061":1,"\uD83D\uDE03":2,"é":3},"type":"BPE"}"#;
    let model = parity(input, &["aé😃", ""]);
    assert_eq!(
        model.id_to_token(0).as_deref(),
        Some("\"\\/\u{8}\u{c}\n\r\t")
    );
    let nested = format!("{}0{}", "[".repeat(json::DEPTH), "]".repeat(json::DEPTH));
    let mut reader = Reader::new(
        &nested,
        Span {
            start: 0,
            end: nested.len(),
        },
    );
    reader.value().unwrap();
    reader.finish().unwrap();
    let q = BpeCompilePlan::prepare_model_json(HELLO.as_bytes())
        .unwrap()
        .requirements();
    assert_eq!(q.required_bytes(), q.buffer_bytes() + q.control_bytes());
    assert!(q.control_bytes() >= size_of::<Result<BPE, BpeCompileFailure>>() + json::stack_bytes());
}

#[test]
fn clones_serde_save_and_owned_vocab_remain_explicit_compatibility_paths() {
    let input = HELLO.to_owned();
    let model = packed(&input);
    drop(input);
    let mut cloned = model.clone();
    cloned.resize_cache(256);
    cloned.clear_cache();
    assert!(!cloned.cache_is_enabled());
    assert_eq!(cloned, model);
    assert_eq!(cloned.tokenize("hello").unwrap()[0].id, 7);
    let mut owned_vocab = cloned.get_vocab();
    owned_vocab.clear();
    assert_eq!(cloned.get_vocab_size(), 8);
    let encoded = serde_json::to_string(&model).unwrap();
    assert_eq!(legacy(&encoded), model);
    assert!(serde_json::from_str::<BPE>(&encoded)
        .unwrap()
        .cache_is_enabled());
    let dir = tempfile::tempdir().unwrap();
    let files = model.save(dir.path(), Some("packed")).unwrap();
    assert_eq!(files.len(), 2);
    let restored = BPE::from_file(files[0].to_str().unwrap(), files[1].to_str().unwrap())
        .cache_policy(ModelCachePolicy::NoModelCaches)
        .build()
        .unwrap();
    assert_eq!(restored, model);
}

#[test]
fn existing_trainer_replaces_packed_storage_as_one_legacy_table_set() {
    let mut trainer = super::super::BpeTrainer::builder()
        .show_progress(false)
        .min_frequency(1)
        .vocab_size(32)
        .build();
    trainer
        .feed(["abab", "abc", "abab"].iter(), |s| Ok(vec![s.to_owned()]))
        .unwrap();
    let mut a = packed(HELLO);
    let mut b = legacy(HELLO);
    assert_eq!(
        trainer.train(&mut a).unwrap(),
        trainer.train(&mut b).unwrap()
    );
    assert!(matches!(&a.storage, Storage::Legacy { .. }));
    assert_eq!(a, b);
    assert_eq!(
        a.tokenize("abababc").unwrap(),
        b.tokenize("abababc").unwrap()
    );
    assert!(a.get_vocab_size() > 3);
}

#[cfg(feature = "parity-aware-bpe")]
#[test]
fn parity_trainer_replaces_packed_storage_without_a_second_training_engine() {
    use super::super::{ParityBpeTrainer, ParityVariant};
    let mut trainer = ParityBpeTrainer::builder()
        .show_progress(false)
        .min_frequency(1)
        .num_merges(6)
        .variant(ParityVariant::Base)
        .build();
    trainer
        .feed_language_from_iter(0, ["aabb", "aabb"].iter(), |s| Ok(vec![s.to_owned()]))
        .unwrap();
    trainer
        .feed_language_from_iter(1, ["ccdd", "ccdd"].iter(), |s| Ok(vec![s.to_owned()]))
        .unwrap();
    let mut a = packed(HELLO);
    let mut b = legacy(HELLO);
    assert_eq!(
        trainer.do_train(&mut a).unwrap(),
        trainer.do_train(&mut b).unwrap()
    );
    assert!(matches!(&a.storage, Storage::Legacy { .. }));
    assert_eq!(a, b);
    assert_eq!(
        a.tokenize("aabbccdd").unwrap(),
        b.tokenize("aabbccdd").unwrap()
    );
}
