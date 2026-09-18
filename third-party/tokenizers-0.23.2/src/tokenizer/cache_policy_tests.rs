use super::*;
use crate::models::bpe::BPE;
use crate::utils::cache::cache_constructions;
const BPE_JSON: &str = r#"{"type":"BPE","vocab":{"<unk>":0,"a":1,"b":2,"ab":3," ":4},"merges":[["a","b"]],"unk_token":"<unk>"}"#;
const UNIGRAM: &str = r#"{"type":"Unigram","vocab":[["<unk>",-10.0],["a",-2.0],["b",-2.0],["ab",0.0],[" ",-2.0]],"unk_id":0,"byte_fallback":false}"#;
fn json(model: &str) -> String {
    format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{{"id":5,"content":"ZZ","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}},{{"id":6,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}}],"normalizer":{{"type":"Lowercase"}},"pre_tokenizer":null,"post_processor":null,"decoder":{{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}},"model":{model}}}"#
    )
}
fn parity(left: &Tokenizer, right: &Tokenizer) {
    for text in ["ABab", "AB ZZ<S>", "ab ab", "", "☃ab"] {
        let a = left.encode(text, false).unwrap();
        let b = right.encode(text, false).unwrap();
        assert_eq!(a.get_ids(), b.get_ids());
        assert_eq!(a.get_offsets(), b.get_offsets());
        assert_eq!(a.get_tokens(), b.get_tokens());
        assert_eq!(a.get_attention_mask(), b.get_attention_mask());
        for skip in [false, true] {
            assert_eq!(
                left.decode(a.get_ids(), skip).unwrap(),
                right.decode(b.get_ids(), skip).unwrap()
            );
        }
    }
    assert!(!right.encode("ABab", false).unwrap().get_ids().is_empty());
    assert_eq!(
        left.decode_vocabulary().id_to_token(5),
        right.decode_vocabulary().id_to_token(5)
    );
    assert_eq!(right.decode_vocabulary().id_to_token(5), Some("zz"));
}
#[test]
fn policy_precedes_tagged_and_legacy_model_construction_and_preserves_values() {
    for model in [BPE_JSON, UNIGRAM] {
        for tagged in [true, false] {
            let mut value: serde_json::Value = serde_json::from_str(model).unwrap();
            if !tagged {
                value.as_object_mut().unwrap().remove("type");
            }
            let bytes = json(&value.to_string());
            let legacy = Tokenizer::from_bytes(bytes.as_bytes()).unwrap();
            let count = cache_constructions();
            let absent = Tokenizer::from_bytes_with_cache_policy(
                bytes.as_bytes(),
                ModelCachePolicy::disabled(),
            )
            .unwrap();
            assert_eq!(cache_constructions(), count);
            assert_eq!(absent.model_cache_policy(), ModelCachePolicy::disabled());
            assert_eq!(legacy.model_cache_policy(), ModelCachePolicy::default());
            parity(&legacy, &absent);
            let alias = absent.clone();
            assert_eq!(cache_constructions(), count);
            parity(&legacy, &alias);
            let serialized = serde_json::to_vec(&absent).unwrap();
            let restored = Tokenizer::from_bytes_with_cache_policy(
                &serialized,
                ModelCachePolicy::disabled(),
            )
            .unwrap();
            assert_eq!(cache_constructions(), count);
            parity(&legacy, &restored);
            // Bare HF JSON deliberately does not transport resource policy.
            assert_eq!(
                Tokenizer::from_bytes(&serialized)
                    .unwrap()
                    .model_cache_policy(),
                ModelCachePolicy::default()
            );
        }
    }
}
#[test]
fn persistent_workers_keep_legacy_tls_but_no_model_cache_appears_for_absent_sources() {
    let mut workers = Vec::new();
    for _ in 0..2 {
        let (tx, rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let legacy = Tokenizer::from_bytes(json(BPE_JSON).as_bytes()).unwrap();
            legacy.encode("abab", false).unwrap();
            let populated = BPE::tls_cache_population();
            assert!(populated.0 > 0 && populated.1 > 0);
            drop(legacy);
            assert_eq!(BPE::tls_cache_population(), populated);
            while rx.recv().unwrap() {
                let before = cache_constructions();
                for model in [BPE_JSON, UNIGRAM] {
                    let source = Tokenizer::from_bytes_with_cache_policy(
                        json(model).as_bytes(),
                        ModelCachePolicy::disabled(),
                    )
                    .unwrap();
                    let copy = source.clone();
                    drop(source);
                    for _ in 0..3 {
                        assert!(!copy.encode("abab", false).unwrap().get_ids().is_empty());
                    }
                    drop(copy);
                }
                assert_eq!(cache_constructions(), before);
                assert_eq!(BPE::tls_cache_population(), populated);
                done_tx.send(()).unwrap();
            }
        });
        workers.push((tx, done_rx, worker));
    }
    for _ in 0..3 {
        for (tx, _, _) in &workers {
            tx.send(true).unwrap();
        }
        for (_, done, _) in &workers {
            done.recv().unwrap();
        }
    }
    for (tx, _, worker) in workers {
        tx.send(false).unwrap();
        worker.join().unwrap();
    }
}
#[test]
fn absent_models_keep_absence_under_clear_resize_and_constructor_failure() {
    for model in [BPE_JSON, UNIGRAM] {
        let mut tokenizer = Tokenizer::from_bytes_with_cache_policy(
            json(model).as_bytes(),
            ModelCachePolicy::disabled(),
        )
        .unwrap();
        let count = cache_constructions();
        let mut model = tokenizer.get_model().clone();
        match &mut model {
            ModelWrapper::BPE(model) => {
                model.clear_cache();
                model.resize_cache(999);
            }
            ModelWrapper::Unigram(model) => {
                model.clear_cache();
                model.resize_cache(999);
            }
            _ => unreachable!(),
        }
        tokenizer.with_model(model);
        assert_eq!(
            tokenizer.model_cache_policy(),
            ModelCachePolicy::disabled()
        );
        assert!(!tokenizer
            .encode("abab", false)
            .unwrap()
            .get_ids()
            .is_empty());
        assert_eq!(cache_constructions(), count);
    }
    let missing_merge = r#"{"type":"BPE","vocab":{"a":0,"b":1,"zz":2},"merges":[["a","b"]]}"#;
    let count = cache_constructions();
    assert!(Tokenizer::from_bytes_with_cache_policy(
        json(missing_merge).as_bytes(),
        ModelCachePolicy::disabled()
    )
    .is_err());
    assert_eq!(cache_constructions(), count);
    assert!(Tokenizer::from_bytes(json(missing_merge).as_bytes()).is_err());
    assert!(cache_constructions() > count); // Positive pre-error descriptor control.
    for bad in [
        "{",
        r#"{"version":"bad"}"#,
        r#"{"version":"1.0","model":{"type":"Unknown"}}"#,
    ] {
        let count = cache_constructions();
        assert!(Tokenizer::from_bytes_with_cache_policy(
            bad.as_bytes(),
            ModelCachePolicy::disabled()
        )
        .is_err());
        assert_eq!(cache_constructions(), count);
    }
}
#[test]
fn seeded_dispatch_preserves_word_models_and_trailing_input_rejection() {
    for model in [
        r###"{"type":"WordPiece","vocab":{"[UNK]":0,"ab":1},"unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100}"###,
        r#"{"type":"WordLevel","vocab":{"[UNK]":0,"ab":1},"unk_token":"[UNK]"}"#,
    ] {
        for tagged in [true, false] {
            let mut value: serde_json::Value = serde_json::from_str(model).unwrap();
            if !tagged {
                value.as_object_mut().unwrap().remove("type");
            }
            let bytes = json(&value.to_string());
            let a = Tokenizer::from_bytes(bytes.as_bytes()).unwrap();
            let b = Tokenizer::from_bytes_with_cache_policy(
                bytes.as_bytes(),
                ModelCachePolicy::disabled(),
            )
            .unwrap();
            assert_eq!(
                a.encode("ab", false).unwrap().get_ids(),
                b.encode("ab", false).unwrap().get_ids()
            );
            assert_eq!(
                serde_json::to_value(a.get_model()).unwrap(),
                serde_json::to_value(b.get_model()).unwrap()
            );
            assert!(Tokenizer::from_bytes_with_cache_policy(
                format!("{bytes} false").as_bytes(),
                ModelCachePolicy::disabled()
            )
            .is_err());
        }
    }
}

#[test]
fn explicit_capacity_is_applied_and_reported_before_model_construction() {
    for capacity in [0, 1, 37] {
        let policy = ModelCachePolicy { capacity };
        for source in [
            r#"{"model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#,
            r#"{"model":{"type":"Unigram","vocab":[["a",0.0]],"unk_id":0}}"#,
            r#"{"model":{"vocab":{"a":0},"merges":[]}}"#,
        ] {
            let tokenizer = Tokenizer::from_bytes_with_cache_policy(source, policy).unwrap();
            assert_eq!(tokenizer.model_cache_policy(), policy);
            assert_eq!(tokenizer.clone().model_cache_policy(), policy);
            assert_eq!(tokenizer.encode("a", false).unwrap().get_ids(), [0]);
        }
    }
}

#[test]
fn zero_resize_removes_the_model_cache_instead_of_reporting_false_absence() {
    for source in [BPE_JSON, UNIGRAM] {
        let mut tokenizer = Tokenizer::from_bytes(json(source)).unwrap();
        tokenizer.encode("abab", false).unwrap();
        match &mut tokenizer.model {
            ModelWrapper::BPE(model) => { model.resize_cache(0); assert!(!model.cache_is_enabled()); }
            ModelWrapper::Unigram(model) => { model.resize_cache(0); assert!(!model.cache_is_enabled()); }
            _ => unreachable!(),
        }
        assert_eq!(tokenizer.model_cache_policy(), ModelCachePolicy::disabled());
    }
}
