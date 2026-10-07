use super::*;

#[test]
fn mismatched_asset_bytes_are_rejected_even_when_length_matches() {
    let request = VOICECHAT_ASSET_REQUESTS[0];
    let error =
        PreparedVoiceChatAssets::resolve::<std::io::Error>(|_| Ok(vec![0; request.byte_len]))
            .unwrap_err();
    assert!(matches!(
        error,
        VoiceChatAssetError::ContentMismatch {
            role: Role::Configuration,
            expected_len: 31984,
            actual_len: 31984,
            ..
        }
    ));
}

#[test]
fn failed_resolution_preserves_the_application_error() {
    let error = PreparedVoiceChatAssets::resolve(|_| {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "snapshot is missing",
        ))
    })
    .unwrap_err();
    let VoiceChatAssetError::Resolve { role, source } = error else {
        panic!("resolver error was erased");
    };
    assert_eq!(role, Role::Configuration);
    assert_eq!(
        source.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::NotFound
    );
}

#[test]
fn rnnt_vocabulary_must_match_the_joint_network_in_order() {
    let mut vocabulary = BTreeMap::new();
    for (token, id) in [
        ("<s>", 1),
        ("</s>", 2),
        ("<SPECIAL_11>", 11),
        ("<SPECIAL_12>", 12),
        ("<SPECIAL_20>", 20),
        ("<SPECIAL_21>", 21),
        ("<SPECIAL_22>", 22),
        ("^", 1094),
        ("$", 1036),
    ] {
        vocabulary.insert(token, id);
    }
    let tokenizer = serde_json::json!({"model": {"vocab": vocabulary}});
    let config = serde_json::json!({"_rnnt_merge_info": {
        "joint_config": {"vocabulary": ["a", "b"]},
        "decoder_config": {"vocab_size": 2},
    }});
    let error = validate_semantics(
        &config,
        &serde_json::json!({"eos_token_id": 12}),
        &tokenizer,
        &["b".into(), "a".into()],
    )
    .unwrap_err();
    assert!(matches!(error, VoiceChatAssetError::Inconsistent(_)));
}

/// Exercises the independently downloaded released assets without storing a
/// 17 MB tokenizer fixture in the source tree.
#[test]
#[ignore = "requires VOICECHAT_OFFICIAL_ASSETS and VOICECHAT_BASE_ASSETS pinned snapshot directories"]
fn released_assets_have_consistent_control_tokens_and_rnnt_vocabulary() {
    let official = std::env::var_os("VOICECHAT_OFFICIAL_ASSETS").expect("official asset directory");
    let base = std::env::var_os("VOICECHAT_BASE_ASSETS").expect("base asset directory");
    let prepared =
        PreparedVoiceChatAssets::from_local_directories(Path::new(&official), Path::new(&base))
            .unwrap();
    let config =
        crate::voicechat::config::VoiceChatConfig::from_value(prepared.configuration()).unwrap();
    assert_eq!(config.rnnt.vocabulary, prepared.rnnt_vocabulary());
    assert_eq!(prepared.rnnt_vocabulary().len(), 1024);
    assert_eq!(prepared.rnnt_vocabulary()[0], "<unk>");
    assert_eq!(prepared.backbone_configuration()["eos_token_id"], 12);
    let retained = prepared.asset(Role::TextTokenizer).bytes();
    assert!(Arc::ptr_eq(
        &retained,
        &prepared.clone().asset(Role::TextTokenizer).bytes()
    ));
    drop(prepared);
    assert_eq!(retained.len(), 17078330);
}

#[test]
#[ignore = "requires VOICECHAT_REFERENCE with downloaded pinned metadata and integer payload ranges"]
fn released_gguf_tokenizer_and_integer_buffers_match_independent_contracts() {
    use super::super::controls::{validate_control_tensors, validate_split_character_tables};
    use eredu_gguf::{MetadataArray, MetadataValue};
    let root = std::env::var_os("VOICECHAT_REFERENCE").expect("pinned reference directory");
    let root = Path::new(&root);
    let header: Value =
        serde_json::from_slice(&std::fs::read(root.join("gguf-2-header.json")).unwrap()).unwrap();
    let metadata = header["metadata"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(key, _)| key.starts_with("tokenizer."))
        .map(|(key, value)| {
            let value = match value {
                Value::String(value) => MetadataValue::String(value.clone()),
                Value::Bool(value) => MetadataValue::Bool(*value),
                Value::Number(value) => MetadataValue::Int64(value.as_i64().unwrap()),
                Value::Array(values) if key == "tokenizer.ggml.token_type" => MetadataValue::Array(
                    MetadataArray::Int64(values.iter().map(|v| v.as_i64().unwrap()).collect()),
                ),
                Value::Array(values) => MetadataValue::Array(MetadataArray::String(
                    values
                        .iter()
                        .map(|v| v.as_str().unwrap().to_owned())
                        .collect(),
                )),
                _ => panic!("unexpected tokenizer metadata"),
            };
            (key.clone(), value)
        })
        .collect::<BTreeMap<_, _>>();
    validate_split_backbone_tokenizer(&metadata).unwrap();
    let mut modified = metadata;
    let MetadataValue::Array(MetadataArray::String(merges)) =
        modified.get_mut("tokenizer.ggml.merges").unwrap()
    else {
        panic!("merges");
    };
    merges.swap(1000, 1001);
    assert!(validate_split_backbone_tokenizer(&modified).is_err());

    let header: Value =
        serde_json::from_slice(&std::fs::read(root.join("official-header.json")).unwrap()).unwrap();
    let mut integers = std::fs::read(root.join("official-integer-buffers.bin")).unwrap();
    validate_control_tensors::<std::io::Error>(|key, range| {
        let start = header[key]["data_offsets"][0].as_u64().unwrap() as usize;
        Ok(integers[start + 8 * range.start..start + 8 * range.end]
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    })
    .unwrap();
    integers[0] ^= 1;
    assert!(validate_control_tensors::<std::io::Error>(|key, range| {
        let start = header[key]["data_offsets"][0].as_u64().unwrap() as usize;
        Ok(integers[start + 8 * range.start..start + 8 * range.end]
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
            .collect())
    })
    .is_err());
    let mut characters = std::fs::read(root.join("split-character-buffers.bin")).unwrap();
    validate_split_character_tables::<std::io::Error>(|key, range| {
        let start = match key {
            "tts.subword.char_ids" => 0,
            "tts.subword.char_offsets" => 3564576,
            _ => unreachable!(),
        };
        Ok(characters[start + 4 * range.start..start + 4 * range.end]
            .chunks_exact(4)
            .map(|bytes| i64::from(i32::from_le_bytes(bytes.try_into().unwrap())))
            .collect())
    })
    .unwrap();
    characters[0] ^= 1;
    assert!(
        validate_split_character_tables::<std::io::Error>(|key, range| {
            let start = match key {
                "tts.subword.char_ids" => 0,
                "tts.subword.char_offsets" => 3564576,
                _ => unreachable!(),
            };
            Ok(characters[start + 4 * range.start..start + 4 * range.end]
                .chunks_exact(4)
                .map(|bytes| i64::from(i32::from_le_bytes(bytes.try_into().unwrap())))
                .collect())
        })
        .is_err()
    );
}
