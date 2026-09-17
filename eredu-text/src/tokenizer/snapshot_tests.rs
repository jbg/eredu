use super::*;
use tokenizers::{decoders::wordpiece::WordPiece, models::wordlevel::WordLevel, AddedToken};

fn tokenizer() -> Tokenizer {
    let model = WordLevel::builder()
        .vocab(
            [
                ("hello".into(), 0),
                ("##s".into(), 1),
                ("<stop>".into(), 2),
                ("[UNK]".into(), 3),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut raw = tokenizers::Tokenizer::new(model);
    raw.add_special_tokens([AddedToken::from("<stop>", true)])
        .unwrap();
    Tokenizer::from_tokenizer(raw)
}

#[test]
fn tokenizer_snapshot_preserves_configuration_after_wrapper_mutation_and_drop() {
    let mut wrapper = tokenizer();
    let original = wrapper.snapshot();
    let peer = original.clone();
    assert!(std::ptr::eq(&*wrapper, &*original));
    let fingerprint = vocabulary_fingerprint(&original);
    assert_eq!(original.decode(&[0, 1], false).unwrap(), "hello ##s");

    // This uses the existing public DerefMut configuration API. Escaped
    // snapshots keep their old decoder, vocabulary and fingerprint.
    wrapper.with_decoder(Some(WordPiece::new("##".into(), false)));
    assert!(!std::ptr::eq(&*wrapper, &*original));
    assert_eq!(wrapper.decode(&[0, 1], false).unwrap(), "hellos");
    assert_eq!(original.decode(&[0, 1], false).unwrap(), "hello ##s");
    assert_eq!(vocabulary_fingerprint(&wrapper), fingerprint);
    wrapper
        .add_tokens([AddedToken::from("new-token", false)])
        .unwrap();
    assert_ne!(vocabulary_fingerprint(&wrapper), fingerprint);
    assert!(original.token_to_id("new-token").is_none());
    let changed = wrapper.snapshot();
    drop(wrapper);
    drop(original);
    assert_eq!(peer.decode(&[0, 1], false).unwrap(), "hello ##s");
    assert_eq!(changed.decode(&[0, 1], false).unwrap(), "hellos");
    assert!(changed.token_to_id("new-token").is_some());
    assert_eq!(vocabulary_fingerprint(&peer), fingerprint);
}

#[test]
fn tokenizer_template_mutation_keeps_snapshot_and_fingerprint_without_copying_inner() {
    let mut wrapper = tokenizer();
    let snapshot = wrapper.snapshot();
    let fingerprint = vocabulary_fingerprint(&wrapper);
    for text in ["first", "second"] {
        wrapper.set_template_kwargs(serde_json::Map::from_iter([(
            "tone".into(),
            serde_json::json!(text),
        )]));
        wrapper.clear_chat_template_cache();
        let rendered = wrapper
            .apply_chat_template_json(
                "{{ tone }}:{{ messages[0].content }}",
                [vec![serde_json::json!({"role":"user","content":"hello"})]],
                None,
                "snapshot-test",
                false,
                None,
            )
            .unwrap();
        assert_eq!(rendered, [format!("{text}:hello")]);
        assert!(std::ptr::eq(&*wrapper, &*snapshot));
        assert_eq!(vocabulary_fingerprint(&wrapper), fingerprint);
    }
    drop(wrapper);
    assert_eq!(snapshot.decode(&[0, 2], true).unwrap(), "hello");
    assert_eq!(snapshot.decode(&[0, 2], false).unwrap(), "hello <stop>");
}
