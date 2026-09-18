use super::*;
use crate::{pre_tokenizers::byte_level::ByteLevel, TokenizerCompilePlan};
use fancy_regex::workspace::{construction, Buffer, DelegateBuffer, PrepareFailure};
use serde_json::{json, Value};
use std::error::Error as _;

fn source(ignore: bool) -> Value {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for (id, c) in alphabet.into_iter().enumerate() {
        vocab.insert(c.to_string(), json!(id));
    }
    vocab.insert("hi".into(), json!(256));
    // Whole-token ID without a merge chain proves full mapped split consumption.
    vocab.insert("Hello".into(), json!(257));
    let added = |id, word, special| json!({"id":id,"content":word,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":special});
    json!({"version":"1.0","truncation":null,"padding":null,"normalizer":null,
      "pre_tokenizer":{"type":"Sequence","pretokenizers":[
        {"type":"Split","pattern":{"Regex":construction::patterns().nth(2).unwrap()},"behavior":"Isolated","invert":false},
        {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
      "post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
      "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
      "added_tokens":[added(258,"<S>",true),added(259,"<S>x",false),added(260,"xx",false)],
      "model":{"type":"BPE","vocab":vocab,"merges":[["h","i"]],"ignore_merges":ignore}})
}
fn packed(value: &Value) -> Tokenizer {
    TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
        .unwrap()
        .compile()
        .unwrap()
}
fn has_reserve(mut cause: Option<&(dyn std::error::Error + 'static)>) -> bool {
    while let Some(error) = cause {
        if error.is::<TryReserveError>() {
            return true;
        }
        cause = error.source();
    }
    false
}
#[test]
fn full_regex_pipeline_and_original_ids_match_ordinary_spelling_offsets_and_special_policy() {
    for ignore in [false, true] {
        let value = source(ignore);
        let mut actual = packed(&value);
        let mut legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
        for encode_special in [false, true] {
            actual.set_encode_special_tokens(encode_special);
            legacy.set_encode_special_tokens(encode_special);
            let bytes: String = (0u8..=255).map(char::from).collect();
            for text in [
                "Hello hi",
                "Hello",
                "hi<S>x<S>xxhi",
                " can't WE'LL 1234567\r\n ",
                "é e\u{301} 🦀 👨‍👩‍👧",
                "\0\u{7f}\t\n",
                "",
                &bytes,
            ] {
                for special in [false, true] {
                    let a = actual.encode(text, special).unwrap();
                    let b = legacy.encode(text, special).unwrap();
                    assert_eq!(a.get_ids(), b.get_ids(), "{text:?}");
                    assert_eq!(a.get_tokens(), b.get_tokens());
                    assert_eq!(a.get_offsets(), b.get_offsets());
                    let plan = EncodeIdsPlan::prepare(&actual, text, special).unwrap();
                    let limits = plan.requirements();
                    assert_eq!(limits.mapped_capacity(), 2 * text.len());
                    assert_eq!(limits.regex_delegate_count(), 0);
                    let ids = plan.encode().unwrap();
                    assert_eq!(ids.ids(), b.get_ids(), "{text:?}");
                    assert_eq!(ids.mapped_capacity(), limits.mapped_capacity());
                    assert_eq!(
                        ids.capacities(),
                        [
                            limits.symbol_capacity(),
                            limits.merge_capacity(),
                            limits.id_capacity()
                        ]
                    );
                }
            }
        }
        let ids = EncodeIdsPlan::prepare(&actual, "Hello", false)
            .unwrap()
            .encode()
            .unwrap();
        assert_eq!(ids.ids() == [257], ignore);
        let restored = Tokenizer::from_bytes_with_cache_policy(
            serde_json::to_string(&actual).unwrap().as_bytes(),
            crate::ModelCachePolicy::disabled(),
        )
        .unwrap();
        assert_eq!(
            restored.encode("Hello hi", false).unwrap().get_ids(),
            legacy.encode("Hello hi", false).unwrap().get_ids()
        );
        let restored_ids = EncodeIdsPlan::prepare(&restored, "Hello", false)
            .unwrap()
            .encode()
            .unwrap();
        assert_eq!(restored_ids.ids(), ids.ids());
    }
}
#[test]
fn decoded_pattern_and_component_settings_are_checked_before_model_construction() {
    use crate::TokenizerCompileErrorKind as K;
    for change in 0..7 {
        let mut value = source(true);
        let pre = &mut value["pre_tokenizer"]["pretokenizers"];
        match change {
            0 => pre[0]["pattern"]["Regex"] = json!("not the admitted pattern"),
            1 => pre[0]["invert"] = json!(true),
            2 => {
                pre[0].as_object_mut().unwrap().remove("invert");
            }
            3 => pre[0]["behavior"] = json!("Removed"),
            4 => pre[1]["add_prefix_space"] = json!(true),
            5 => pre[1]["use_regex"] = json!(true),
            _ => pre.as_array_mut().unwrap().swap(0, 1),
        }
        value["model"] = json!(false);
        assert_eq!(
            TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
                .unwrap_err()
                .kind(),
            if change == 2 {
                K::MissingField
            } else {
                K::RegexProfile
            }
        );
    }
    // Equivalent JSON spelling retains the original borrow and decodes to the same selected source.
    let json = source(true)
        .to_string()
        .replace("Isolated", "\\u0049solated");
    let actual = TokenizerCompilePlan::prepare_json(json.as_bytes())
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(
        EncodeIdsPlan::prepare(&actual, "Hello", false)
            .unwrap()
            .encode()
            .unwrap()
            .ids(),
        [257]
    );
    let mut transformed = packed(&source(true));
    transformed.with_normalizer(Some(crate::normalizers::NFD));
    assert!(matches!(
        EncodeIdsPlan::prepare(&transformed, "Hello", false),
        Err(EncodeIdsError::PipelineProfile)
    ));
}
#[test]
fn real_cold_regex_failures_retain_completed_model_and_original_nested_cause() {
    use construction::ConstructionFailure as F;
    for target in [F::Pattern, F::Instructions, F::LastDelegate, F::Completed] {
        let json = source(true).to_string();
        let failure = TokenizerCompilePlan::prepare_json(json.as_bytes())
            .unwrap()
            .fail_regex_construction(target)
            .compile()
            .unwrap_err();
        assert!(failure.completed_model());
        drop(json);
        let cause = failure
            .source()
            .unwrap()
            .downcast_ref::<construction::Failure>()
            .unwrap();
        if !matches!(target, F::Completed) {
            assert!(has_reserve(Some(cause)));
        }
        // This is a retained real constructor prefix; Completed is explicitly injected.
        assert!(!failure.to_string().is_empty());
    }
}
#[test]
fn mapped_and_every_actual_regex_reserve_fail_once_with_retired_nested_prefixes() {
    let actual = packed(&source(true));
    let input = "Hello hi<S>xé";
    let failure = EncodeIdsPlan::prepare(&actual, input, false)
        .unwrap()
        .fail_reservation(3)
        .encode()
        .unwrap_err();
    assert!(failure.capacities().iter().all(|&n| n > 0));
    assert_eq!(failure.mapped_capacity(), 0);
    assert!(has_reserve(Some(&failure)));
    let delegates = EncodeIdsPlan::prepare(&actual, input, false)
        .unwrap()
        .requirements()
        .regex_delegate_count();
    let outer = [Buffer::Saves, Buffer::Branches, Buffer::Undo];
    let inner = [
        DelegateBuffer::Epsilon,
        DelegateBuffer::CurrentDense,
        DelegateBuffer::CurrentSparse,
        DelegateBuffer::NextDense,
        DelegateBuffer::NextSparse,
        DelegateBuffer::CurrentSlots,
        DelegateBuffer::NextSlots,
    ];
    let targets = outer
        .iter()
        .copied()
        .map(PrepareFailure::Outer)
        .chain((0..delegates).flat_map(|ordinal| {
            inner
                .iter()
                .copied()
                .map(move |buffer| PrepareFailure::Delegate { ordinal, buffer })
        }));
    for target in targets {
        let failure = EncodeIdsPlan::prepare(&actual, input, false)
            .unwrap()
            .fail_regex_reservation(target)
            .unwrap()
            .encode()
            .unwrap_err();
        assert!(failure.capacities().iter().all(|&n| n > 0));
        assert_eq!(failure.mapped_capacity(), 2 * input.len());
        assert_eq!(failure.partial_id_count(), 0);
        let EncodeIdsError::RegexPreparation(retired) = failure.cause() else {
            panic!("actual regex failure")
        };
        assert!(retired.reserve_error().is_some());
        match target {
            PrepareFailure::Outer(buffer) => {
                assert_eq!(retired.buffer(), Some(buffer));
                assert_eq!(retired.retired_delegates(), 0);
            }
            PrepareFailure::Delegate { ordinal, buffer } => {
                assert_eq!(retired.retired_delegates(), ordinal);
                let (actual, nested) = retired.delegate_error().unwrap();
                assert_eq!(actual, ordinal);
                assert_eq!(nested.buffer(), buffer);
            }
        }
        assert!(has_reserve(Some(&failure)));
    }
}
#[test]
fn independent_workspaces_share_only_immutable_source_and_reuse_no_model_cache() {
    let worker = std::thread::spawn(|| {
        let value = source(true);
        let actual = packed(&value);
        let legacy = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
        let before = crate::models::bpe::BPE::tls_cache_population();
        legacy.encode("hihi hihi", false).unwrap();
        let positive = crate::models::bpe::BPE::tls_cache_population();
        assert!(positive.0 > before.0 && positive.1 > before.1);
        for _ in 0..32 {
            let a = EncodeIdsPlan::prepare(&actual, "Hello hi", false)
                .unwrap()
                .encode()
                .unwrap();
            let b = EncodeIdsPlan::prepare(&actual, "Hello hi", false)
                .unwrap()
                .encode()
                .unwrap();
            assert_eq!(a.ids(), b.ids());
            assert_ne!(a.ids().as_ptr(), b.ids().as_ptr());
            assert_eq!(crate::models::bpe::BPE::tls_cache_population(), positive);
        }
    });
    worker.join().unwrap();
}
