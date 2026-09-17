use super::*;
use crate::{
    pre_tokenizers::{digits::Digits, sequence::Sequence, PreTokenizerWrapper},
    EncodeIdsError, EncodeIdsPlan, OffsetReferential, OffsetType, Tokenizer, TokenizerCompilePlan,
};
use serde_json::{json, Value};
fn source() -> Value {
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for (id, c) in alphabet.into_iter().enumerate() {
        vocab.insert(c.to_string(), json!(id));
    }
    for (word, id) in [("hi", 256), ("12", 257), ("34", 258)] {
        vocab.insert(word.into(), json!(id));
    }
    let added = |id, word| json!({"id":id,"content":word,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true});
    json!({"version":"1.0","truncation":null,"padding":null,"normalizer":null,
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]},
        "post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true},
        "added_tokens":[added(259,"<S>"),added(260,"<N12>")],
        "model":{"type":"BPE","vocab":vocab,"merges":[["h","i"],["1","2"],["3","4"]]}})
}
fn packed(v: &Value) -> Tokenizer {
    TokenizerCompilePlan::prepare_json(v.to_string().as_bytes())
        .unwrap()
        .compile()
        .unwrap()
}
#[test]
fn complete_implicit_profile_matches_legacy_ids_spellings_offsets_and_numeric_boundaries() {
    let value = source();
    let mut actual = packed(&value);
    let mut ordinary = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    let bytes: String = (0u8..=255).map(char::from).collect();
    for encode_special in [false, true] {
        actual.set_encode_special_tokens(encode_special);
        ordinary.set_encode_special_tokens(encode_special);
        for text in [
            "hi 1234 hi",
            "a12b ٣٤५６",
            "¼Ⅷ²¹ 1\u{301}2",
            " A\u{200d}1\u{200d}B ",
            "hi<N12><S>1234",
            "WE'LL can't 123\r\n  ",
            "",
            &bytes,
        ] {
            for special in [false, true] {
                let a = actual.encode(text, special).unwrap();
                let b = ordinary.encode(text, special).unwrap();
                assert_eq!(a.get_ids(), b.get_ids(), "{text:?}");
                assert_eq!(a.get_tokens(), b.get_tokens());
                assert_eq!(a.get_offsets(), b.get_offsets());
                let plan = EncodeIdsPlan::prepare(&actual, text, special).unwrap();
                let facts = plan.requirements();
                let ids = plan.encode().unwrap();
                assert_eq!(ids.ids(), b.get_ids());
                assert_eq!(ids.mapped_capacity(), 2 * text.len());
                assert_eq!(
                    ids.capacities(),
                    [
                        facts.symbol_capacity(),
                        facts.merge_capacity(),
                        facts.id_capacity()
                    ]
                );
            }
        }
    }
    let numeric = EncodeIdsPlan::prepare(&actual, "1234", false)
        .unwrap()
        .encode()
        .unwrap();
    assert_eq!(numeric.ids().len(), 4);
    assert!(!numeric.ids().contains(&257));
    assert!(!numeric.ids().contains(&258));
}
#[test]
fn closed_byte_source_preserves_two_stage_serde_clone_and_ordinary_offset_worker() {
    let value = source();
    let actual = packed(&value);
    let cloned = actual.clone();
    drop(actual);
    let json = serde_json::to_value(&cloned).unwrap();
    assert_eq!(json["pre_tokenizer"], value["pre_tokenizer"]);
    let restored = Tokenizer::from_bytes(json.to_string().as_bytes()).unwrap();
    for text in ["hi 1234", " ³a1 \r\n", ""] {
        assert_eq!(
            cloned.encode(text, false).unwrap().get_ids(),
            restored.encode(text, false).unwrap().get_ids()
        );
    }
    assert!(matches!(
        EncodeIdsPlan::prepare(&restored, "hi", false),
        Err(EncodeIdsError::PipelineProfile)
    ));
    let compiled = CompiledByteLevel::from_parts(
        ByteLevel::new(false, true, true),
        CompiledRegexSplit::compile(
            fancy_regex::workspace::construction::Plan::new(
                super::super::byte_level::DEFAULT_PATTERN,
            )
            .unwrap(),
        )
        .unwrap(),
    );
    let compiled = Sequence::new(vec![
        PreTokenizerWrapper::Digits(Digits::new(true)),
        PreTokenizerWrapper::CompiledByteLevel(compiled),
    ]);
    let ordinary = Sequence::new(vec![
        Digits::new(true).into(),
        ByteLevel::new(false, true, true).into(),
    ]);
    for text in [" a123 \r\n", "٣\u{301}²x", "", " a1\t2"] {
        let mut a = PreTokenizedString::from(text);
        let mut b = PreTokenizedString::from(text);
        compiled.pre_tokenize(&mut a).unwrap();
        ordinary.pre_tokenize(&mut b).unwrap();
        for referential in [OffsetReferential::Original, OffsetReferential::Normalized] {
            let a: Vec<_> = a
                .get_splits(referential, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect();
            let b: Vec<_> = b
                .get_splits(referential, OffsetType::Byte)
                .into_iter()
                .map(|(s, o, _)| (s, o))
                .collect();
            assert_eq!(a, b);
        }
    }
}
#[test]
fn implicit_settings_and_default_use_regex_bind_before_model_construction() {
    let mut value = source();
    value["pre_tokenizer"]["pretokenizers"][1]
        .as_object_mut()
        .unwrap()
        .remove("use_regex");
    let tokenizer = packed(&value);
    assert_eq!(
        EncodeIdsPlan::prepare(&tokenizer, "1234", false)
            .unwrap()
            .encode()
            .unwrap()
            .ids()
            .len(),
        4
    );
    for change in 0..5 {
        let mut value = source();
        let pre = &mut value["pre_tokenizer"]["pretokenizers"];
        match change {
            0 => pre[0]["individual_digits"] = json!(false),
            1 => {
                pre[0].as_object_mut().unwrap().remove("individual_digits");
            }
            2 => pre[1]["add_prefix_space"] = json!(true),
            3 => pre.as_array_mut().unwrap().swap(0, 1),
            _ => {
                let extra = pre[0].clone();
                pre.as_array_mut().unwrap().push(extra);
            }
        }
        value["model"] = json!(false);
        assert_eq!(
            TokenizerCompilePlan::prepare_json(value.to_string().as_bytes())
                .unwrap_err()
                .kind(),
            crate::TokenizerCompileErrorKind::RegexProfile
        );
    }
}
#[cfg(feature = "tokenizer-compiler-test-support")]
#[test]
fn implicit_first_literal_and_each_real_workspace_frontier_keep_owned_causes() {
    use fancy_regex::workspace::{
        construction::{ConstructionFailure as C, ScratchFailureBuffer as S},
        Buffer as B, DelegateBuffer as D, PrepareFailure as F,
    };
    use std::error::Error as _;
    for target in [
        C::Pattern,
        C::Instructions,
        C::FirstLiteral,
        C::LastDelegate,
        C::Scratch(S::Stack),
        C::Scratch(S::Dense),
        C::Scratch(S::Sparse),
        C::Completed,
    ] {
        let json = source().to_string();
        let error = TokenizerCompilePlan::prepare_json(json.as_bytes())
            .unwrap()
            .fail_regex_construction(target)
            .compile()
            .unwrap_err();
        assert!(error.completed_model());
        drop(json);
        let mut cause = error.source();
        let mut reserve = false;
        while let Some(e) = cause {
            reserve |= e.is::<std::collections::TryReserveError>();
            cause = e.source();
        }
        if !matches!(target, C::Completed) {
            assert!(reserve);
        }
    }
    let actual = packed(&source());
    let text = "hi12 ٣<S>34";
    let delegates = EncodeIdsPlan::prepare(&actual, text, false)
        .unwrap()
        .requirements()
        .regex_delegate_count();
    assert!(delegates > 0);
    let inner = [
        D::Epsilon,
        D::CurrentDense,
        D::CurrentSparse,
        D::NextDense,
        D::NextSparse,
        D::CurrentSlots,
        D::NextSlots,
    ];
    let outer = [B::Delegates, B::Saves, B::Branches, B::Undo, B::Slots];
    let targets = outer
        .iter()
        .copied()
        .map(F::Outer)
        .chain((0..delegates).flat_map(|ordinal| {
            inner
                .iter()
                .copied()
                .map(move |buffer| F::Delegate { ordinal, buffer })
        }));
    for target in targets {
        let error = EncodeIdsPlan::prepare(&actual, text, false)
            .unwrap()
            .fail_regex_reservation(target)
            .unwrap()
            .encode()
            .unwrap_err();
        assert_eq!(error.partial_id_count(), 0);
        assert_eq!(error.mapped_capacity(), 2 * text.len());
        let EncodeIdsError::RegexPreparation(retired) = error.cause() else {
            panic!("actual failure")
        };
        assert!(retired.reserve_error().is_some());
        if let F::Delegate { ordinal, buffer } = target {
            let (actual, nested) = retired.delegate_error().unwrap();
            assert_eq!(actual, ordinal);
            assert_eq!(nested.buffer(), buffer);
            assert_eq!(retired.retired_delegates(), ordinal);
        }
    }
}
#[test]
fn accepted_source_never_initializes_ordinary_regex_in_a_fresh_process() {
    const CHILD: &str = "EREDU_COMPILED_BYTELEVEL_CHILD";
    const TEST:&str="pre_tokenizers::compiled_byte_level::tests::accepted_source_never_initializes_ordinary_regex_in_a_fresh_process";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", TEST, "--nocapture", "--test-threads=1"])
            .env(CHILD, "1")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("running 1 test"));
        return;
    }
    use super::super::byte_level::regex_initializations_for_test as count;
    assert_eq!(count(), 0);
    let value = source();
    let actual = packed(&value);
    for _ in 0..3 {
        actual.encode("hi 123", false).unwrap();
        EncodeIdsPlan::prepare(&actual, "hi 123", false)
            .unwrap()
            .encode()
            .unwrap();
    }
    assert_eq!(count(), 0);
    let mut input = PreTokenizedString::from("hello 123");
    ByteLevel::new(false, true, true)
        .pre_tokenize(&mut input)
        .unwrap();
    assert_eq!(count(), 1);
    let ordinary = Tokenizer::from_bytes(value.to_string().as_bytes()).unwrap();
    let before = crate::models::bpe::BPE::tls_cache_population();
    ordinary.encode("hihi 123 hihi", false).unwrap();
    let positive = crate::models::bpe::BPE::tls_cache_population();
    assert!(positive.0 > before.0 && positive.1 > before.1);
    for _ in 0..64 {
        EncodeIdsPlan::prepare(&actual, "hihi 123 hihi", false)
            .unwrap()
            .encode()
            .unwrap();
        assert_eq!(crate::models::bpe::BPE::tls_cache_population(), positive);
        assert_eq!(count(), 1);
    }
}
