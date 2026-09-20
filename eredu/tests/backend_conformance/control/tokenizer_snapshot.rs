use super::*;

#[test]
fn loaded_decoder_snapshots_outlive_model_with_unicode_forks_and_special_modes() {
    let first = 10;
    let tokenizer = ChatTokenizer::from_tokenizer(unicode_tokenizer(Some(first), 64));
    let original = tokenizer.snapshot();
    let mut model = unicode_model_from_tokenizer(tokenizer, QWEN_TEMPLATE);
    let fingerprint = *model.tokenizer_fingerprint();
    let expected_skip = model.decode(&[first, first + 1, first + 2], true).unwrap();
    let expected_include = model.decode(&[first, first + 1, first + 2], false).unwrap();
    assert_eq!(expected_skip, "é");
    assert_eq!(expected_include, "é<|im_end|>");
    let mut skip = model.text_decoder(true);
    let mut include = model.text_decoder(false);
    assert!(skip.step(first).unwrap().is_none());
    assert!(include.step(first).unwrap().is_none());
    let mut fork = skip.clone();

    // Changing mutable template state neither changes the token configuration
    // nor invalidates decoders which already carry an incomplete UTF-8 prefix.
    model.replace_template(Some("{{ messages[0].content }}".into()));
    assert_eq!(*model.tokenizer_fingerprint(), fingerprint);
    // Borrowed spelling has the same backing as the pre-load snapshot: changing
    // templates did not clone or replace the tokenizer configuration.
    assert!(std::ptr::eq(
        model.tokenizer().id_to_token(first).unwrap(),
        original.vocabulary().id_to_token(first).unwrap(),
    ));
    drop(original);
    drop(model);

    let mut skipped = skip.step(first + 1).unwrap().unwrap();
    skipped.push_str(skip.step(first + 2).unwrap().as_deref().unwrap_or(""));
    let mut included = include.step(first + 1).unwrap().unwrap();
    included.push_str(include.step(first + 2).unwrap().as_deref().unwrap_or(""));
    assert_eq!(skipped, expected_skip);
    assert_eq!(included, expected_include);
    drop((skip, include));
    assert_eq!(fork.step(first + 1).unwrap().as_deref(), Some("é"));
    assert!(fork.step(first + 2).unwrap().is_none());
}

#[test]
fn loaded_tokenizer_view_preserves_sparse_normalized_and_overlapping_metadata() {
    let words = WordLevel::builder()
        .vocab(
            [
                ("[UNK]".into(), 1),
                ("known".into(), 0),
                ("é".into(), 70_000),
            ]
            .into_iter()
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut raw = Tokenizer::new(words);
    raw.with_normalizer(Some(tokenizers::normalizers::Lowercase))
        .unwrap();
    raw.add_special_tokens([
        AddedToken::from("known", true).normalized(false),
        AddedToken::from("<LOUD>", true).normalized(true),
        AddedToken::from("<|im_end|>", true).normalized(false),
    ])
    .unwrap();
    let loud = raw.token_to_id("<LOUD>").unwrap();
    let mut expected: Vec<_> = raw.get_vocab(false).into_values().collect();
    expected.extend(
        raw.get_added_vocabulary()
            .get_added_tokens_decoder()
            .keys()
            .copied(),
    );
    expected.sort_unstable();
    let tokenizer = ChatTokenizer::from_tokenizer(raw);
    let original = tokenizer.snapshot();
    let model = unicode_model_from_tokenizer(tokenizer, QWEN_TEMPLATE);
    let view = model.tokenizer();
    let mut actual: Vec<_> = view.decode_ids().collect();
    actual.sort_unstable();
    assert_eq!(actual, expected);
    assert_eq!(actual.iter().filter(|&&id| id == 0).count(), 2);
    assert!(!actual.contains(&2));
    assert_eq!(view.token_to_id("é"), Some(70_000));
    assert_eq!(view.id_to_token(70_000), Some("é"));
    assert_eq!(view.id_to_token(u32::MAX), None);
    assert_eq!(view.token_to_id("<LOUD>"), Some(loud));
    assert_eq!(view.id_to_token(loud), Some("<loud>"));
    assert!(view.is_special_token("<LOUD>"));
    assert!(!view.is_special_token("<loud>"));
    assert!(view.is_special_token("known"));
    for id in actual {
        let spelling = view.id_to_token(id).unwrap();
        assert!(std::ptr::eq(
            spelling,
            original.vocabulary().id_to_token(id).unwrap()
        ));
    }
    assert!(std::ptr::eq(
        view.fingerprint(),
        model.tokenizer_fingerprint()
    ));
}
