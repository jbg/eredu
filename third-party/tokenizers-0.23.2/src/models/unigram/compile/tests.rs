use super::*;
use crate::{EncodeIdsPlan, Model, TokenizerCompilePlan};
const MODEL: &[u8] = br#"{"type":"Unigram","unk_id":0,"vocab":[["<unk>",0.0],["a",-1.0],["b",-1.0],["ab",-2.0],["\u00e9",-1.25],["\u4e2d\u6587",-0.25],["a",-0.75]],"byte_fallback":false}"#;
#[test]
fn actual_source_matches_scored_duplicates_unknown_fusion_and_utf8() {
    let plan = UnigramCompilePlan::prepare_model_json(MODEL).unwrap();
    assert_eq!(plan.requirements().id_extent(), 7);
    let actual = plan.compile().unwrap();
    let ordinary: Unigram = serde_json::from_slice(MODEL).unwrap();
    assert!(actual.same_configuration(&ordinary));
    for text in [
        "",
        "a",
        "ab",
        "aba",
        "é中文",
        "🙂x🙂a🙂x",
        "é\u{301}",
        "\0a\0",
    ] {
        assert_eq!(
            actual.tokenize(text).unwrap(),
            ordinary.tokenize(text).unwrap(),
            "{text:?}"
        );
    }
    assert_eq!(actual.token_to_id("a"), Some(6));
}
#[test]
fn every_destination_reserve_retains_the_original_prefix() {
    let bound = UnigramCompilePlan::prepare_model_json(MODEL)
        .unwrap()
        .requirements()
        .buffer_bytes();
    let mut previous = 0;
    for stage in 0..18 {
        let failure = UnigramCompilePlan::prepare_model_json(MODEL)
            .unwrap()
            .fail_reservation(stage)
            .compile()
            .unwrap_err();
        let retained = failure.allocated_bytes().unwrap();
        assert!(
            retained >= previous && retained <= bound,
            "stage {stage}: {retained} after {previous}"
        );
        previous = retained;
    }
    assert!(previous > 0);
}
#[test]
fn floating_scores_use_the_ordinary_typed_json_worker() {
    for score in [
        "-0",
        "-0.0",
        "0.84551240822557006",
        "2.2250738585072014e-308",
        "1.7976931348623157e308",
        "1e-5000",
    ] {
        let source = format!("{{\"type\":\"Unigram\",\"unk_id\":0,\"vocab\":[[\"a\",{score}]],\"byte_fallback\":false}}");
        let actual = UnigramCompilePlan::prepare_model_json(source.as_bytes())
            .unwrap()
            .compile()
            .unwrap();
        let ordinary: Unigram = serde_json::from_str(&source).unwrap();
        assert_eq!(
            actual.vocab[0].1.to_bits(),
            ordinary.vocab[0].1.to_bits(),
            "{score}"
        );
    }
    let source = br#"{"type":"Unigram","unk_id":0,"vocab":[["a",1e5000]]}"#;
    assert!(UnigramCompilePlan::prepare_model_json(source)
        .unwrap()
        .compile()
        .is_err());
    assert!(serde_json::from_slice::<Unigram>(source).is_err());
}
#[test]
fn aggregate_ids_reuse_the_original_path_and_byte_fallback() {
    for byte_fallback in [false, true] {
        let mut vocab = vec![
            ("<unk>".to_owned(), -9.0),
            ("a".into(), -1.0),
            ("ab".into(), -1.2),
            ("▁".into(), -0.3),
            ("é".into(), -0.5),
        ];
        if byte_fallback {
            for b in 0..=255 {
                vocab.push((format!("<0x{b:02X}>"), -10.0));
            }
        }
        for scheme in ["always", "first", "never"] {
            let source = serde_json::json!({"version":"1.0", "added_tokens":[], "normalizer":null,
                "pre_tokenizer":{"type":"Metaspace", "replacement":"▁", "prepend_scheme":scheme, "split":true},
                "post_processor":null,"decoder":null,
                "model":{"type":"Unigram", "unk_id":0,"vocab":vocab,"byte_fallback":byte_fallback}}).to_string();
            let actual = TokenizerCompilePlan::prepare_json(source.as_bytes())
                .unwrap()
                .compile()
                .unwrap();
            let ordinary = crate::Tokenizer::from_bytes(source.as_bytes()).unwrap();
            assert!(actual.matches_compiled_configuration(&ordinary));
            for input in ["", "ab a", "é 🙂", "\0xx\0", "a\t ab\n é", "🙂🙂a🙂"] {
                let expected = ordinary.encode(input, false).unwrap();
                let plan = EncodeIdsPlan::prepare(&actual, input, false).unwrap();
                let output = plan.encode().unwrap();
                assert_eq!(
                    output.ids(),
                    expected.get_ids(),
                    "{scheme} {byte_fallback} {input:?}"
                );
                assert_eq!(
                    actual.encode(input, false).unwrap().get_offsets(),
                    expected.get_offsets()
                );
            }
        }
    }
}

#[cfg(feature = "tokenizer-compiler-test-support")]
#[test]
fn each_id_destination_failure_retains_real_path_storage() {
    let model: serde_json::Value = serde_json::from_slice(MODEL).unwrap();
    let source = serde_json::json!({"version":"1.0", "added_tokens":[], "normalizer":null, "pre_tokenizer":null,
        "post_processor":null,"decoder":null,"model":model}).to_string();
    let source = TokenizerCompilePlan::prepare_json(source.as_bytes())
        .unwrap()
        .compile()
        .unwrap();
    for stage in 0..6 {
        let failure = EncodeIdsPlan::prepare(&source, "ab🙂é", false)
            .unwrap()
            .fail_reservation(stage)
            .encode()
            .unwrap_err();
        assert!(matches!(failure.cause(), crate::EncodeIdsError::Reserve(_)));
        assert_eq!(failure.partial_id_count(), 0);
        if stage == 5 {
            assert!(failure.unigram_capacities()[0] > 0);
        }
    }
    let output = EncodeIdsPlan::prepare(&source, "ab🙂é", false)
        .unwrap()
        .encode()
        .unwrap();
    assert!(output.unigram_capacities().iter().all(|n| *n > 0));
}
