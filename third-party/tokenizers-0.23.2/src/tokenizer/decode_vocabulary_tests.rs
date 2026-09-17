use super::*;
use crate::models::{bpe::BPE, unigram::Unigram, wordlevel::WordLevel, wordpiece::WordPiece};
use crate::normalizers::{
    replace::{Replace, ReplacePattern},
    Lowercase,
};

fn assert_view(tokenizer: &Tokenizer) {
    let view = tokenizer.decode_vocabulary();
    let mut actual = Vec::new();
    view.visit_ids(&mut |id| actual.push(id));
    let mut expected: Vec<_> = tokenizer.get_vocab(false).into_values().collect();
    expected.extend(
        tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .keys()
            .copied(),
    );
    let mut iterated: Vec<_> = view.ids().collect();
    iterated.sort_unstable();
    actual.sort_unstable();
    assert_eq!(iterated, actual);
    expected.sort_unstable();
    assert_eq!(actual, expected);
    actual.push(u32::MAX);
    for id in actual {
        assert_eq!(view.id_to_token(id), tokenizer.id_to_token(id).as_deref());
        if let Some(token) = view.id_to_token(id) {
            assert!(std::ptr::eq(token, view.id_to_token(id).unwrap()));
            assert_eq!(
                view.is_special_token(token),
                tokenizer.get_added_vocabulary().is_special_token(token)
            );
        }
    }
}

#[test]
fn borrowed_decode_maps_match_all_concrete_models_and_sparse_or_repeated_ids() {
    let vocab: ahash::AHashMap<String, u32> = vec![
        ("[UNK]".into(), 0),
        ("hi".into(), 9),
        ("there".into(), 70_000),
    ]
    .into_iter()
    .collect();
    let wordlevel = WordLevel::builder().vocab(vocab.clone()).build().unwrap();
    let wordpiece = WordPiece::builder().vocab(vocab.clone()).build().unwrap();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .build()
        .unwrap();
    // get_vocab uses the final forward ID for repeated spellings, whereas the
    // reverse vector still contains both rows. Enumeration must preserve that.
    let unigram = Unigram::from(
        vec![
            ("[UNK]".into(), 0.0),
            ("repeat".into(), -1.0),
            ("repeat".into(), -2.0),
            ("other".into(), -3.0),
        ],
        Some(0),
        false,
    )
    .unwrap();
    let models: Vec<ModelWrapper> = vec![
        wordlevel.into(),
        wordpiece.into(),
        bpe.into(),
        unigram.into(),
    ];
    for model in models {
        assert_view(&Tokenizer::new(model));
    }
}

#[test]
fn borrowed_decode_preserves_added_shadowing_and_normalized_special_spelling() {
    let mut tokenizer = Tokenizer::new(
        WordLevel::builder()
            .vocab(
                vec![("[UNK]".into(), 0), ("base".into(), 1)]
                    .into_iter()
                    .collect(),
            )
            .build()
            .unwrap(),
    );
    tokenizer.with_normalizer(Some(Lowercase)).unwrap();
    tokenizer
        .add_tokens([AddedToken::from("SAME", false).normalized(true)])
        .unwrap();
    let added = tokenizer.token_to_id("SAME").unwrap();
    tokenizer.with_model(
        WordLevel::builder()
            .vocab(
                vec![
                    ("SAME".into(), 99),
                    ("hidden".into(), added),
                    ("plain".into(), 10),
                ]
                .into_iter()
                .collect(),
            )
            .build()
            .unwrap(),
    );
    tokenizer
        .add_special_tokens([
            AddedToken::from("<LOUD>", true).normalized(true),
            AddedToken::from("<stop>", true),
        ])
        .unwrap();
    assert_view(&tokenizer);
    let view = tokenizer.decode_vocabulary();
    assert_eq!(view.id_to_token(added), Some("same"));
    assert_eq!(view.id_to_token(99), Some("SAME"));
    let loud = tokenizer.token_to_id("<LOUD>").unwrap();
    assert_eq!(view.id_to_token(loud), Some("<loud>"));
    assert!(!view.is_special_token(view.id_to_token(loud).unwrap()));
    assert!(view.is_special_token("<stop>"));
}

#[test]
fn replace_borrow_distinguishes_equal_text_with_different_pattern_kind() {
    let literal = Replace::new("▁", " ").unwrap();
    let regex = Replace::new(ReplacePattern::Regex("▁".into()), " ").unwrap();
    assert!(matches!(literal.pattern(), ReplacePattern::String(value) if value == "▁"));
    assert!(matches!(regex.pattern(), ReplacePattern::Regex(value) if value == "▁"));
    assert!(std::ptr::eq(literal.pattern(), literal.pattern()));
}
