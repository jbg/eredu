use super::*;
use crate::{models::bpe::BPE, TokenizerCompilePlan};
use serde_json::{json, Value};
use std::error::Error as _;
fn source() -> Value {
    let added = |id, content, special| json!({"id":id,"content":content,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":special});
    json!({"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "added_tokens":[added(10,"<S>",true),added(11,"<S>x",false),added(12,"xx",false)],
        "model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"ab":3,"bc":4,"abc":5," ":6,"?":7,"é":8,"x":9},
            "merges":[["a","b"],["b","c"],["ab","c"]],"unk_token":"?","fuse_unk":false}})
}
fn packed(value: &Value) -> Tokenizer {
    TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
        .unwrap()
        .compile()
        .unwrap()
}
fn compare(actual: &Tokenizer, legacy: &Tokenizer, input: &str) {
    for special in [false, true] {
        let plan = EncodeIdsPlan::prepare(actual, input, special).unwrap();
        let limits = plan.requirements();
        let output = plan.encode().unwrap();
        assert_eq!(
            output.ids(),
            legacy.encode(input, special).unwrap().get_ids(),
            "{input:?}"
        );
        let capacities = output.capacities();
        assert!(capacities[0] >= limits.symbol_capacity());
        assert!(capacities[1] >= limits.merge_capacity());
        assert!(capacities[2] >= limits.id_capacity());
        assert!(output.ids().len() <= limits.id_capacity());
    }
}
#[test]
fn identity_ids_match_legacy_ranks_unicode_unknowns_and_raw_special_consumption(
) {
    for fuse in [false, true] {
        for ignore in [false, true] {
            let mut value = source();
            value["model"]["fuse_unk"] = json!(fuse);
            value["model"]["ignore_merges"] = json!(ignore);
            let mut actual = packed(&value);
            let mut legacy =
                Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
            for encode_special in [false, true] {
                actual.set_encode_special_tokens(encode_special);
                legacy.set_encode_special_tokens(encode_special);
                for input in [
                    "abc abcab",
                    "é🦀zé",
                    "<S>x<S>xx",
                    "<S><S>x",
                    "xxabcxx",
                    "\0\n\r?",
                    "zzaz",
                    "",
                    "a",
                ] {
                    compare(&actual, &legacy, input);
                }
            }
            assert_eq!(
                EncodeIdsPlan::prepare(&actual, "abc", true)
                    .unwrap()
                    .encode()
                    .unwrap()
                    .ids(),
                &[5]
            );
        }
    }
}
#[test]
fn reused_word_and_stale_merge_heap_match_all_nonzero_small_inputs() {
    let value = source();
    let actual = packed(&value);
    let legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    for number in 0..729usize {
        let mut n = number;
        let mut input = String::new();
        for _ in 0..6 {
            input.push(['a', 'b', 'c'][n % 3]);
            n /= 3;
        }
        input.push_str("<S>abc xx ");
        input.push_str(&"abc".repeat(number % 5));
        compare(&actual, &legacy, &input);
    }
}
#[test]
fn actual_profile_flags_reject_before_any_destination_exists() {
    let mut actual = packed(&source());
    actual.with_normalizer(Some(crate::normalizers::NFD));
    assert!(matches!(
        EncodeIdsPlan::prepare(&actual, "abc", false),
        Err(EncodeIdsError::PipelineProfile)
    ));
    let mut value = source();
    value["pre_tokenizer"] = json!({"type":"ByteLevel","use_regex":false});
    assert!(matches!(
        EncodeIdsPlan::prepare(&packed(&value), "abc", false),
        Err(EncodeIdsError::PipelineProfile)
    ));
    let mut value = source();
    value["post_processor"] = json!({"type":"ByteLevel"});
    assert!(matches!(
        EncodeIdsPlan::prepare(&packed(&value), "abc", false),
        Err(EncodeIdsError::PipelineProfile)
    ));
    let mut value = source();
    value["added_tokens"][0]["normalized"] = json!(true);
    // None normalization now supports both literal phases; word-boundary
    // matching still has no closed original workspace recipe.
    let actual = packed(&value);
    let legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    compare(&actual, &legacy, "<S>abc<S>x");
    value["added_tokens"][0]["single_word"] = json!(true);
    assert!(matches!(
        TokenizerCompilePlan::prepare_json(value.to_string().as_bytes()),
        Err(crate::TokenizerCompileError::Added(_))
    ));
    for field in ["continuing_subword_prefix", "end_of_word_suffix"] {
        let mut value = source();
        value["model"][field] = json!("#");
        value["model"]["merges"] = json!([]);
        assert!(matches!(
            EncodeIdsPlan::prepare(&packed(&value), "abc", false),
            Err(EncodeIdsError::ModelProfile)
        ));
    }
    let mut value = source();
    value["model"]["byte_fallback"] = json!(true);
    assert!(matches!(
        EncodeIdsPlan::prepare(&packed(&value), "abc", false),
        Err(EncodeIdsError::ModelProfile)
    ));
    let mut actual = packed(&source());
    actual.with_model(BPE::default());
    assert!(matches!(
        EncodeIdsPlan::prepare(&actual, "abc", false),
        Err(EncodeIdsError::ModelProfile)
    ));
    let mut padded = packed(&source());
    padded.with_padding(Some(Default::default()));
    assert!(matches!(
        EncodeIdsPlan::prepare(&padded, "abc", false),
        Err(EncodeIdsError::PipelineProfile)
    ));
    let mut truncated = packed(&source());
    truncated.with_truncation(Some(Default::default())).unwrap();
    assert!(matches!(
        EncodeIdsPlan::prepare(&truncated, "abc", false),
        Err(EncodeIdsError::PipelineProfile)
    ));
    // Actual HF mutation to an ordinary added matcher is not accepted as packed authority.
    let legacy =
        Tokenizer::from_bytes(source().to_string().as_bytes()).unwrap();
    assert!(matches!(
        EncodeIdsPlan::prepare(&legacy, "abc", false),
        Err(EncodeIdsError::AddedProfile)
    ));
}
#[test]
fn every_actual_target_reserve_retains_earlier_capacity_and_real_error() {
    let actual = packed(&source());
    for stage in 0..3 {
        let mut plan =
            EncodeIdsPlan::prepare(&actual, "abc<S>abcé", false).unwrap();
        plan.failure = Some(stage);
        let error = plan.encode().unwrap_err();
        assert_eq!(error.partial_id_count(), 0);
        for (index, capacity) in error.capacities().iter().enumerate() {
            assert_eq!(*capacity > 0, index < stage);
        }
        assert!(error
            .source()
            .unwrap()
            .source()
            .unwrap()
            .is::<TryReserveError>());
    }
}
#[test]
fn missing_unknown_is_fixed_and_retains_actual_prior_ids_and_word_storage() {
    let mut value = source();
    value["model"]["unk_token"] = json!("missing");
    let actual = packed(&value);
    let error = EncodeIdsPlan::prepare(&actual, "<S>az", false)
        .unwrap()
        .encode()
        .unwrap_err();
    assert!(matches!(error.cause(), EncodeIdsError::MissingUnknown));
    assert_eq!(error.partial_id_count(), 1);
    assert!(error.capacities().iter().all(|n| *n > 0));
    let legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    assert!(legacy
        .encode("<S>az", false)
        .unwrap_err()
        .is::<crate::models::bpe::Error>());
}
#[test]
fn persistent_worker_ids_never_grow_model_caches_with_positive_legacy_control()
{
    let worker = std::thread::spawn(|| {
        let value = source();
        let actual = packed(&value);
        let legacy =
            Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
        let before = BPE::tls_cache_population();
        let expected =
            legacy.encode("abc abc", false).unwrap().get_ids().to_vec();
        let populated = BPE::tls_cache_population();
        assert!(populated.0 > before.0 && populated.1 > before.1);
        for _ in 0..128 {
            let output = EncodeIdsPlan::prepare(&actual, "abc abc", true)
                .unwrap()
                .encode()
                .unwrap();
            assert_eq!(output.ids(), expected);
            assert_eq!(BPE::tls_cache_population(), populated);
        }
    });
    worker.join().unwrap();
}
