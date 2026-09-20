use super::*;
use crate::decoder_storage::DecodeDestinations;
use serde_json::json;
fn added(text: &str) -> serde_json::Value {
    json!({"id":999,"content":text,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false})
}
fn source() -> serde_json::Value {
    json!({"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"a":0,"b":2},"merges":[]}})
}
#[test]
fn source_preserves_duplicate_added_ids_and_whole_token_bytelevel_fallback() {
    let mut value = source();
    let spelling = "Ġ🦀long";
    value["added_tokens"] = json!([added(spelling)]);
    let input = value.to_string();
    let plan = TokenizerPlan::prepare_json(input.as_bytes()).unwrap();
    let prepared = plan.compile().unwrap();
    drop(input);
    assert_eq!(prepared.token_id(spelling), Some(2));
    assert_eq!(prepared.spelling(2), Some(spelling));
    let actual = DecodeCompilePlan::prepare_metadata(&prepared.root().metadata)
        .unwrap()
        .requirements();
    assert_eq!(actual.id_slots(), 3);
    assert_eq!(actual.piece_bytes(), 1 + 2 * spelling.len());
    assert!(actual.piece_bytes() > 2 + spelling.len());
    let mut ids: Vec<_> = prepared.ids().collect();
    ids.sort_unstable();
    assert_eq!(ids, [0, 2, 2]);
    let mut storage = DecodeDestinations::new(prepared.decode_source(), 2, false).unwrap();
    storage.prepare_destinations().unwrap();
    let first = storage
        .step(prepared.decode_source(), 2)
        .unwrap()
        .unwrap()
        .to_owned();
    assert_eq!(first, spelling); // one nonalphabet scalar makes the entire token fall back.
    assert_eq!(
        storage.step(prepared.decode_source(), 0).unwrap(),
        Some("a")
    );
    storage.finish(prepared.decode_source()).unwrap();
    assert_eq!(
        prepared.root().tokenizer.decode(&[2, 0], false).unwrap(),
        format!("{spelling}a")
    );
    // Largest sparse ID changes neither visit count nor destination population.
    let mut sparse = source();
    sparse["model"]["vocab"] = json!({"a":0,"b":u32::MAX});
    let input = sparse.to_string();
    let plan = TokenizerPlan::prepare_json(input.as_bytes()).unwrap();
    let sparse = plan.compile().unwrap();
    drop(input);
    assert_eq!(sparse.spelling(u32::MAX), Some("b"));
    let mut destination = DecodeDestinations::new(sparse.decode_source(), 1, false).unwrap();
    destination.prepare_destinations().unwrap();
    assert_eq!(
        destination.step(sparse.decode_source(), u32::MAX).unwrap(),
        Some("b")
    );
    destination.finish(sparse.decode_source()).unwrap();
}
#[test]
fn one_root_constructs_fresh_hf_and_same_decoder_with_special_filtering() {
    let mut value = source();
    value["model"]["vocab"] = json!({"h":0,"i":1,"hi":2,"Ġ":3,"é":4});
    value["model"]["merges"] = json!([["h", "i"]]);
    let mut special = added("<S>");
    special["special"] = json!(true);
    value["added_tokens"] = json!([special]);
    let input = value.to_string();
    let prepared = TokenizerPlan::prepare_json(input.as_bytes())
        .unwrap()
        .compile()
        .unwrap();
    let legacy = tokenizers::Tokenizer::from_bytes(input.as_bytes()).unwrap();
    drop(input);
    let id = prepared.token_id("<S>").unwrap();
    assert!(prepared.is_special("<S>"));
    for skip in [false, true] {
        let mut storage = DecodeDestinations::new(prepared.decode_source(), 4, skip).unwrap();
        storage.prepare_destinations().unwrap();
        let mut output = String::new();
        for token in [2, 3, id, 2] {
            if let Some(text) = storage.step(prepared.decode_source(), token).unwrap() {
                output.push_str(text);
            }
        }
        storage.finish(prepared.decode_source()).unwrap();
        assert_eq!(output, legacy.decode(&[2, 3, id, 2], skip).unwrap());
    }
}
#[cfg(feature = "tokenizer-compiler-test-support")]
#[test]
fn all_actual_decode_reserve_failures_retain_completed_hf_and_partial_decoder() {
    let mut previous = 0;
    for stage in 0..3 {
        let input = source().to_string();
        let error = TokenizerPlan::prepare_json(input.as_bytes())
            .unwrap()
            .fail_decode_reservation(stage)
            .compile()
            .unwrap_err();
        drop(input);
        assert!(error.completed_tokenizer());
        assert!(error.root_failure().is_none());
        assert_eq!(error.tokenizer.as_ref().unwrap().token_to_id("b"), Some(2));
        let decoder = error.decode_failure().unwrap();
        assert!(matches!(decoder.cause(), DecodeSourceError::Allocation(_)));
        let retained = decoder.retained_buffer_bytes();
        assert_eq!(retained == 0, stage == 0);
        if stage > 0 {
            assert!(retained > previous);
        }
        previous = retained;
    }
}

#[test]
fn compiled_configuration_matches_ordinary_templates_but_rejects_policy_changes() {
    let mut value = source();
    value["model"]["vocab"] = json!({"h":0,"i":1,"hi":2,"<S>":3});
    value["model"]["merges"] = json!([["h", "i"]]);
    value["added_tokens"] = json!([{"id":3,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
    value["post_processor"] = json!({"type":"TemplateProcessing",
        "single":[{"SpecialToken":{"id":"<S>","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
        "pair":[{"SpecialToken":{"id":"<S>","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"Sequence":{"id":"B","type_id":1}}],
        "special_tokens":{"<S>":{"id":"<S>","ids":[3],"tokens":["<S>"]}}});
    let bytes = serde_json::to_vec(&value).unwrap();
    let compiled = TokenizerPlan::prepare_json(&bytes)
        .unwrap()
        .compile()
        .unwrap();
    let selected = crate::tokenizer::Tokenizer::from_bytes(&bytes).unwrap();
    assert!(compiled.matches_configuration(&selected));
    assert_eq!(selected.encode("hi", true).unwrap().get_ids(), [3, 2]);
    for change in 0..4 {
        let mut changed = value.clone();
        match change {
            0 => changed["model"]["merges"] = json!([]),
            1 => changed["added_tokens"][0]["lstrip"] = json!(true),
            2 => changed["post_processor"]["single"][0]["SpecialToken"]["type_id"] = json!(1),
            _ => changed["normalizer"] = json!({"type":"NFC"}),
        }
        let changed =
            crate::tokenizer::Tokenizer::from_bytes(serde_json::to_vec(&changed).unwrap()).unwrap();
        assert_eq!(
            crate::tokenizer::vocabulary_fingerprint(&selected),
            crate::tokenizer::vocabulary_fingerprint(&changed)
        );
        assert!(
            !compiled.matches_configuration(&changed),
            "policy change {change}"
        );
    }
}

#[test]
fn original_fallback_components_preserve_all_shared_orders_and_streaming_byte_boundaries() {
    use crate::decoder_storage::{DecodeStorageError, OwnedDecodeStorageError};
    let mut observed_prefix_failure = false;
    for order in 0..3 {
        for strip in [false, true] {
            let fallback = json!({"type":"ByteFallback"});
            let fuse = json!({"type":"Fuse"});
            let replace = json!({"type":"Replace","pattern":{"String":"▁"},"content":" "});
            let mut components = if order == 0 {
                vec![replace, fallback, fuse]
            } else {
                vec![fallback, fuse, replace]
            };
            if order == 2 {
                components.push(json!({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":false}));
            }
            if strip {
                components.push(json!({"type":"Strip","content":" ","start":1,"stop":0}));
            }
            let mut value = source();
            value["decoder"] = json!({"type":"Sequence","decoders":components});
            value["model"]["vocab"] = json!({"▁a":0,"<0xC3>":1,"<0xA9>":2,"<0xFF>":3,"🙂":4,"Ġb":5,"<S>":6,
                "<0xE2>":7,"<0x96>":8,"<0x81>":9});
            let mut special = added("<S>");
            special["id"] = json!(6);
            special["special"] = json!(true);
            value["added_tokens"] = json!([special]);
            let input = value.to_string();
            let plan = TokenizerPlan::prepare_json(input.as_bytes()).unwrap();
            let admitted = plan.requirements();
            let prepared = plan.compile().unwrap();
            let ordinary = tokenizers::Tokenizer::from_bytes(input.as_bytes()).unwrap();
            let selected = crate::tokenizer::Tokenizer::from_bytes(input.as_bytes()).unwrap();
            assert!(prepared.matches_configuration(&selected));
            drop(input);
            assert!(admitted.required_bytes() > 4);
            assert_eq!(
                serde_json::to_string(&prepared.root().tokenizer.get_decoder()).unwrap(),
                serde_json::to_string(&ordinary.get_decoder()).unwrap()
            );
            for skip in [false, true] {
                let ids = [0, 1, 2, 7, 8, 9, 3, 4, 6, 5];
                let mut storage =
                    DecodeDestinations::new(prepared.decode_source(), ids.len(), skip).unwrap();
                storage.prepare_destinations().unwrap();
                let (mut retained, mut prefix, mut index) = (Vec::new(), String::new(), 0);
                for id in ids {
                    let expected = tokenizers::tokenizer::step_decode_stream(
                        &ordinary,
                        vec![id],
                        skip,
                        &mut retained,
                        &mut prefix,
                        &mut index,
                    );
                    let actual = storage.step(prepared.decode_source(), id);
                    match expected {
                        Ok(text) => assert_eq!(
                            actual,
                            Ok(text.as_deref()),
                            "order={order} strip={strip} id={id}"
                        ),
                        Err(error) => {
                            let Some(tokenizers::tokenizer::DecodeStreamError::InvalidPrefix {
                                token_id,
                                expected_prefix,
                                actual_string,
                            }) = error.downcast_ref()
                            else {
                                panic!("unexpected HF error: {error}")
                            };
                            observed_prefix_failure = true;
                            assert_eq!(
                                actual,
                                Err(OwnedDecodeStorageError::Storage(
                                    DecodeStorageError::InvalidPrefix {
                                        token_id: *token_id,
                                        expected_bytes: expected_prefix.len(),
                                        actual_bytes: actual_string.len()
                                    }
                                ))
                            );
                            assert_eq!(storage.prefix(), expected_prefix);
                            assert_eq!(storage.candidate(), actual_string);
                        }
                    }
                    assert_eq!(storage.retained_ids(), retained);
                    assert_eq!(storage.prefix(), prefix);
                    assert_eq!(storage.prefix_index(), index);
                }
                let remaining = ordinary.decode(&retained, skip).unwrap();
                let expected_finish = if remaining.len() > prefix.len() {
                    Err(OwnedDecodeStorageError::Storage(
                        DecodeStorageError::IncompleteByteSequence,
                    ))
                } else {
                    Ok(())
                };
                assert_eq!(storage.finish(prepared.decode_source()), expected_finish);
                assert_eq!(storage.candidate(), remaining);
                assert_eq!(
                    prepared.root().tokenizer.decode(&ids, skip).unwrap(),
                    ordinary.decode(&ids, skip).unwrap()
                );
            }
        }
    }
    assert!(observed_prefix_failure);
}

#[test]
fn original_special_splitting_flag_preserves_actual_added_matcher_encoding() {
    for enabled in [false, true] {
        let mut value = source();
        value["normalizer"] = serde_json::Value::Null;
        value["pre_tokenizer"] = serde_json::Value::Null;
        value["post_processor"] = serde_json::Value::Null;
        value["decoder"] = serde_json::Value::Null;
        value["model"]["vocab"] = json!({"<":0,"S":1,">":2,"a":3});
        value["model"]["merges"] = json!([]);
        let mut special = added("<S>");
        special["id"] = json!(4);
        special["special"] = json!(true);
        special["normalized"] = json!(false);
        value["added_tokens"] = json!([special]);
        let input = value.to_string();
        let prepared = TokenizerPlan::prepare_json(input.as_bytes())
            .unwrap()
            .with_encode_special_tokens(enabled)
            .compile()
            .unwrap();
        let mut ordinary = tokenizers::Tokenizer::from_bytes(input.as_bytes()).unwrap();
        ordinary.set_encode_special_tokens(enabled);
        drop(input);
        assert_eq!(
            prepared.root().tokenizer.get_encode_special_tokens(),
            enabled
        );
        for skip in [false, true] {
            let ids = [0, 1, 2, 4, 3];
            let mut destination =
                DecodeDestinations::new(prepared.decode_source(), ids.len(), skip).unwrap();
            destination.prepare_destinations().unwrap();
            let mut joined = String::new();
            for id in ids {
                if let Some(text) = destination.step(prepared.decode_source(), id).unwrap() {
                    joined.push_str(text);
                }
            }
            destination.finish(prepared.decode_source()).unwrap();
            assert_eq!(joined, ordinary.decode(&ids, skip).unwrap());
        }
        for text in ["<S>", "a<S>a", "<S><S>"] {
            let expected = ordinary.encode(text, false).unwrap();
            let actual = EncodeIdsPlan::prepare(&prepared, text, false)
                .unwrap()
                .encode()
                .unwrap();
            assert_eq!(actual.ids(), expected.get_ids());
            if text == "<S>" {
                assert_eq!(
                    actual.ids(),
                    if enabled { &[0, 1, 2][..] } else { &[4][..] }
                );
            }
        }
        let selected = crate::tokenizer::Tokenizer::from_tokenizer(ordinary);
        assert!(prepared.matches_configuration(&selected));
    }
}

#[test]
fn original_bare_bytelevel_ids_share_full_span_mapping_and_special_matching() {
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    let mut alphabet: Vec<_> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut vocab = serde_json::Map::new();
    for (id, letter) in alphabet.into_iter().enumerate() {
        vocab.insert(letter.to_string(), json!(id));
    }
    vocab.insert("hi".into(), json!(256));
    vocab.insert("HelloĠhi".into(), json!(257));
    let mut value = source();
    value["model"] = json!({"type":"BPE","vocab":vocab,"merges":[["h","i"]],"ignore_merges":true});
    value["pre_tokenizer"] =
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false});
    value["added_tokens"] = json!([{"id":258,"content":"<S>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
    let input = value.to_string();
    for flag in [false, true] {
        let prepared = TokenizerPlan::prepare_json(input.as_bytes())
            .unwrap()
            .with_encode_special_tokens(flag)
            .compile()
            .unwrap();
        let mut ordinary = tokenizers::Tokenizer::from_bytes(input.as_bytes()).unwrap();
        ordinary.set_encode_special_tokens(flag);
        let all: String = (0u8..=255).map(char::from).collect();
        for text in ["Hello hi", "hi<S> hi", "é e\u{301} 🙂", "\0\n\t", "", &all] {
            let plan = EncodeIdsPlan::prepare(&prepared, text, false).unwrap();
            let actual = plan.encode().unwrap();
            assert_eq!(
                actual.ids(),
                ordinary.encode(text, false).unwrap().get_ids(),
                "{text:?}"
            );
        }
        assert_eq!(
            EncodeIdsPlan::prepare(&prepared, "Hello hi", false)
                .unwrap()
                .encode()
                .unwrap()
                .ids(),
            [257]
        );
    }
    let mut ordinary = tokenizers::Tokenizer::from_bytes(input.as_bytes()).unwrap();
    ordinary.with_pre_tokenizer(Some(ByteLevel::new(true, true, false)));
    let bytes = serde_json::to_vec(&ordinary).unwrap();
    let prepared = TokenizerPlan::prepare_json(&bytes)
        .unwrap()
        .compile()
        .unwrap();
    let ids = EncodeIdsPlan::prepare(&prepared, "hi", false)
        .unwrap()
        .encode()
        .unwrap();
    assert_eq!(ids.ids(), ordinary.encode("hi", false).unwrap().get_ids());
}

#[test]
fn wordlevel_source_authenticates_unknown_policy_and_whitespace() {
    let mut value = source();
    value["model"] =
        json!({"type":"WordLevel","vocab":{"[UNK]":0,"other":1,"é":5},"unk_token":"[UNK]"});
    value["pre_tokenizer"] = json!({"type":"Whitespace"});
    let bytes = serde_json::to_vec(&value).unwrap();
    let prepared = TokenizerPlan::prepare_json(&bytes)
        .unwrap()
        .compile()
        .unwrap();
    let selected = crate::tokenizer::Tokenizer::from_bytes(&bytes).unwrap();
    assert!(prepared.matches_configuration(&selected));
    assert!(prepared.input_prefix_removal_is_identity());
    for change in 0..3 {
        let mut different = value.clone();
        match change {
            0 => different["model"]["unk_token"] = json!("other"),
            1 => different["pre_tokenizer"] = serde_json::Value::Null,
            _ => different["model"]["vocab"]["é"] = json!(7),
        }
        let other =
            crate::tokenizer::Tokenizer::from_bytes(serde_json::to_vec(&different).unwrap())
                .unwrap();
        assert!(
            !prepared.matches_configuration(&other),
            "changed policy {change}"
        );
    }
}

#[test]
fn unigram_authentication_compares_scores_unknown_and_fallback_policy() {
    let mut value = source();
    value["model"] = json!({"type":"Unigram", "unk_id":0, "byte_fallback":false,
        "vocab":[["<unk>",-9.0],["a",-1.0],["ab",-1.0]]});
    value["pre_tokenizer"] = serde_json::Value::Null;
    let bytes = serde_json::to_vec(&value).unwrap();
    let prepared = TokenizerPlan::prepare_json(&bytes)
        .unwrap()
        .compile()
        .unwrap();
    let selected = crate::tokenizer::Tokenizer::from_bytes(&bytes).unwrap();
    assert!(prepared.matches_configuration(&selected));
    assert!(prepared.tokenization_is_canonical());
    for change in 0..3 {
        let mut value = value.clone();
        match change {
            0 => value["model"]["unk_id"] = serde_json::Value::Null,
            1 => value["model"]["byte_fallback"] = json!(true),
            _ => value["model"]["vocab"][2][1] = json!(-0.75),
        }
        let other =
            crate::tokenizer::Tokenizer::from_bytes(serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(
            !prepared.matches_configuration(&other),
            "changed scored model policy {change}"
        );
    }
    let mut changed = tokenizers::Tokenizer::from_bytes(&bytes).unwrap();
    let tokenizers::ModelWrapper::Unigram(mut model) = changed.get_model().clone() else {
        unreachable!()
    };
    model.alpha = Some(0.5);
    model.nbest_size = Some(2);
    changed.with_model(model);
    assert!(!prepared.matches_configuration(&crate::tokenizer::Tokenizer::from_tokenizer(changed)));
}

#[test]
fn prefix_copy_preserves_decode_and_changes_only_input_prefix_semantics() {
    let mut value = source();
    value["model"] = json!({"type":"Unigram", "unk_id":0, "byte_fallback":false,
        "vocab":[["<unk>",-10.0],["▁",-2.0],["a",-1.0],["b",-1.0],["▁a",-0.5]]});
    value["normalizer"] = json!({"type":"Sequence", "normalizers":[
        {"type":"NFC"}, {"type":"Sequence", "normalizers":[{"type":"Prepend", "prepend":"▁"}]}]});
    value["pre_tokenizer"] =
        json!({"type":"Metaspace", "replacement":"▁", "prepend_scheme":"always", "split":true});
    value["decoder"] =
        json!({"type":"Metaspace", "replacement":"▁", "prepend_scheme":"always", "split":true});
    let bytes = serde_json::to_vec(&value).unwrap();
    let original = TokenizerPlan::prepare_json(&bytes)
        .unwrap()
        .compile()
        .unwrap();
    let plan = original.input_prefix_plan().unwrap().unwrap();
    assert_eq!(
        plan.requirements().required_bytes(),
        original.root().estimate.construction(bytes.len()).unwrap()
    );
    let changed = plan.compile().unwrap();
    assert!(changed.is_input_prefix_derivative_of(&original));
    assert!(!std::ptr::eq(changed.input_view(), original.input_view()));
    assert!(changed.input_prefix_plan().unwrap().is_none());
    assert_eq!(
        EncodeIdsPlan::prepare(&original, "a", false)
            .unwrap()
            .encode()
            .unwrap()
            .ids(),
        [4]
    );
    assert_eq!(
        EncodeIdsPlan::prepare(&changed, "a", false)
            .unwrap()
            .encode()
            .unwrap()
            .ids(),
        [2]
    );
    assert_eq!(changed.input_view().decode(&[4, 3], false).unwrap(), "ab");
    assert_eq!(original.input_view().decode(&[4, 3], false).unwrap(), "ab");
    drop(original);
    assert_eq!(
        EncodeIdsPlan::prepare(&changed, "a b", false)
            .unwrap()
            .encode()
            .unwrap()
            .ids(),
        [2, 1, 3]
    );
}

#[test]
fn generation_geometry_handles_sparse_maps_scored_arrays_and_added_overlaps() {
    for (vocab, expected) in [
        (json!({"a":2,"b":900}), 901),
        (json!([["a", -1.0], ["a", -2.0], ["b", -3.0]]), 5),
    ] {
        let bytes =
            serde_json::to_vec(&json!({"model":{"vocab":vocab}, "added_tokens":[{},{}]})).unwrap();
        let plan = TokenizerPlan::prepare_json(&bytes)
            .unwrap()
            .with_generation_domain()
            .unwrap();
        assert_eq!(plan.generation_domain_extent(), Some(expected));
    }
    let input = source().to_string();
    let plan = TokenizerPlan::prepare_json(input.as_bytes())
        .unwrap()
        .with_memory_estimate(TokenizerMemoryEstimate {
            fixed_bytes: 100,
            bytes_per_source_byte: 3,
            bytes_per_text_byte: 7,
        })
        .unwrap();
    assert_eq!(plan.requirements().required_bytes(), 100 + 3 * input.len());
    let prepared = plan.compile().unwrap();
    assert_eq!(
        EncodeIdsPlan::prepare(&prepared, "é", false)
            .unwrap()
            .required_bytes(),
        114
    );
    assert!(
        TokenizerPlan::prepare_json(input.as_bytes())
            .unwrap()
            .with_memory_estimate(TokenizerMemoryEstimate {
                fixed_bytes: usize::MAX,
                ..Default::default()
            })
            .is_err()
    );
}

#[test]
fn duplicate_model_ids_reject_ambiguous_decoding_but_added_aliases_remain_valid() {
    use std::error::Error;
    let mut value = source();
    value["model"]["vocab"] = json!({"a":7,"b":7});
    let input = value.to_string();
    let error = TokenizerPlan::prepare_json(input.as_bytes())
        .unwrap()
        .compile()
        .unwrap_err();
    assert!(error.completed_tokenizer());
    assert!(matches!(
        error.source().unwrap().downcast_ref::<DecodeSourceError>(),
        Some(DecodeSourceError::DuplicateModelId(7))
    ));
    value["model"]["vocab"] = json!({"a":0,"b":2});
    value["added_tokens"] = json!([added("<S>")]);
    let input = value.to_string();
    let prepared = TokenizerPlan::prepare_json(input.as_bytes())
        .unwrap()
        .compile()
        .unwrap();
    assert_eq!(prepared.spelling(2), Some("<S>"));
}
