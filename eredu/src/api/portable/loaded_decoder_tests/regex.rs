use super::*;
use std::io::Write as _;
#[test]
fn private_file_regex_c_and_original_e_share_real_backend_pool_and_decoder_source() {
    file_profile(0);
}
#[test]
fn private_file_implicit_c_and_original_e_share_real_backend_pool_and_decoder_source() {
    file_profile(1);
}
#[test]
fn private_file_template_c_and_e_preserve_special_flags_and_source_ownership() {
    file_profile(2);
}
#[test]
fn private_file_two_phase_literals_preserve_both_flags_and_closed_backend_source() {
    file_profile(3);
}
#[test]
fn private_file_nfc_empty_affixes_use_actual_backend_i_c_e_and_shared_decoder() {
    file_profile(4);
}
fn file_profile(profile: u8) {
    let implicit = profile == 1;
    let json = r###"{"version":"1.0","truncation":null,"padding":null,"normalizer":null,"pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},"post_processor":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},"added_tokens":[],"model":{"type":"BPE","vocab":{"h":0,"i":1,"hi":2,"Ġ":3},"merges":[["h","i"]],"ignore_merges":true}}"###;
    let json = if implicit {
        let mut value: serde_json::Value = serde_json::from_str(json).unwrap();
        value["pre_tokenizer"]=serde_json::from_str(r###"{"type":"Sequence","pretokenizers":[{"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}]}"###).unwrap();
        value["post_processor"] = serde_json::Value::Null;
        value["model"]["ignore_merges"] = serde_json::json!(false);
        value["model"]["vocab"]["1"] = serde_json::json!(4);
        value["model"]["vocab"]["2"] = serde_json::json!(5);
        value["model"]["vocab"]["12"] = serde_json::json!(6);
        value["model"]["merges"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::json!(["1", "2"]));
        value.to_string()
    } else {
        json.to_owned()
    };
    let json = if profile >= 2 {
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["post_processor"] = serde_json::json!({"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],"pair":[{"SpecialToken":{"id":"start","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"start","type_id":1}},{"Sequence":{"id":"B","type_id":1}}],"special_tokens":{"start":{"id":"start","ids":[2],"tokens":["hi"]}}});
        value.to_string()
    } else {
        json
    };
    let json = if profile == 3 {
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["pre_tokenizer"]=serde_json::from_str(r###"{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}"###).unwrap();
        value["model"]["ignore_merges"] = serde_json::json!(false);
        value["added_tokens"]=serde_json::from_str(r#"[{"id":4,"content":"Mathias","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},{"id":5,"content":"python","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}]"#).unwrap();
        value.to_string()
    } else {
        json
    };
    let json = if profile == 4 {
        let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
        value["normalizer"] = serde_json::json!({"type":"NFC"});
        value["model"]["ignore_merges"] = serde_json::json!(false);
        value["model"]["continuing_subword_prefix"] = serde_json::json!("");
        value["model"]["end_of_word_suffix"] = serde_json::json!("");
        for (id, spelling) in [(4, "Ã"), (5, "©"), (6, "Ì"), (7, "Ī"), (8, "ģ")] {
            value["model"]["vocab"][spelling] = serde_json::json!(id);
        }
        value.to_string()
    } else {
        json
    };
    let pool = WorkingMemoryPool::new(u64::MAX, 0).unwrap();
    let compiles = Rc::new(Cell::new(0));
    let runtime = ModelRuntime::prepare(
        Backend {
            pool: pool.clone(),
            compiles: compiles.clone(),
        },
        (),
    )
    .unwrap();
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(json.as_bytes()).unwrap();
    let source = crate::api::tokenizer::compile_original_tokenizer_file(&runtime, file).unwrap();
    let c = source.original_bytes();
    assert_eq!(
        pool.used_bytes().unwrap(),
        c,
        "I retires after fresh C construction"
    );
    let legacy = tokenizers::Tokenizer::from_bytes(json.as_bytes()).unwrap();
    let mut held = Vec::new();
    let mut total = c;
    for text in if implicit {
        ["hihi12 hi", "hi12 hi", ""]
    } else if profile == 4 {
        ["hihi e\u{301} hi", "\u{344}", ""]
    } else if profile == 3 {
        ["Mathias hi python", "Mathiaspython", ""]
    } else {
        ["hihi hi", "hi hi", ""]
    } {
        for special in if profile >= 2 {
            &[false, true][..]
        } else {
            &[true][..]
        } {
            let special = *special;
            let e =
                WorkingMemoryPool::tokenizer_encode_required_bytes(&source, text, special).unwrap();
            let ids = crate::api::tokenizer::encode_original_tokenizer_ids(
                &runtime, &source, text, special,
            )
            .unwrap();
            assert_eq!(ids.ids(), legacy.encode(text, special).unwrap().get_ids());
            assert_eq!(ids.original_bytes(), e);
            if profile == 4 {
                if text == "\u{344}" {
                    assert_eq!(
                        ids.ids(),
                        if special {
                            &[2, 6, 7, 6, 8][..]
                        } else {
                            &[6, 7, 6, 8][..]
                        }
                    );
                }
            }
            assert!(ids.matches_source(&source));
            if ids.ids().is_empty() {
                assert!(matches!(
                    eredu_core::TokenIdsInputPlan::new(ids.ids()),
                    Err(eredu_core::TokenInputRejection::Empty)
                ));
            } else {
                let input = eredu_core::TokenIdsInputPlan::new(ids.ids()).unwrap();
                assert_eq!(input.tokens(), ids.ids());
            }
            total += e;
            held.push(ids);
        }
    }
    assert_eq!(compiles.get(), 1);
    let decoder =
        eredu_runtime::working_memory::AggregateGenerationDecoderInput::new(&source, 3, false)
            .unwrap();
    assert!(decoder.source().same_source(&source));
    drop((runtime, source));
    assert_eq!(pool.used_bytes().unwrap(), total);
    drop(held);
    assert_eq!(pool.used_bytes().unwrap(), c);
    drop(decoder);
    assert_eq!(pool.used_bytes().unwrap(), 0);
}
