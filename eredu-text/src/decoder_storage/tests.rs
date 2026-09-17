use super::*;
use crate::tokenizer::Tokenizer;
use tokenizers::{models::wordlevel::WordLevel, pre_tokenizers::byte_level::ByteLevel, AddedToken};

fn model(pieces: impl IntoIterator<Item = (String, u32)>) -> WordLevel {
    WordLevel::builder()
        .vocab(pieces.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap()
}

fn wrapper(pieces: &[(&str, u32)], bytes: bool) -> Tokenizer {
    let mut hf =
        tokenizers::Tokenizer::new(model(pieces.iter().map(|(s, id)| ((*s).to_owned(), *id))));
    if bytes {
        hf.with_decoder(Some(ByteLevel::default()));
    }
    Tokenizer::from_tokenizer(hf)
}

struct Buffers {
    ids: Vec<u32>,
    raw: Vec<u8>,
    candidate: Vec<u8>,
    prefix: Vec<u8>,
}
impl Buffers {
    fn new(layout: &DecodeStreamLayout<'_>) -> Self {
        Self {
            ids: vec![u32::MAX; layout.token_capacity()],
            raw: vec![0xcd; layout.raw_capacity()],
            candidate: vec![0xcd; layout.text_capacity()],
            prefix: vec![0xcd; layout.text_capacity()],
        }
    }
    fn stream<'a, 'b>(&'b mut self, layout: DecodeStreamLayout<'a>) -> DecodeStreamState<'a, 'b> {
        DecodeStreamState::new(
            layout,
            &mut self.ids,
            &mut self.raw,
            &mut self.candidate,
            &mut self.prefix,
        )
        .unwrap()
    }
}

fn differential(
    snapshot: &TokenizerSnapshot,
    source: &PreparedDecodeSource,
    tokens: &[u32],
    skip: bool,
) -> Vec<Option<String>> {
    let layout = DecodeStreamLayout::for_source(source, tokens.len(), skip).unwrap();
    let mut buffers = Buffers::new(&layout);
    let addresses = (
        buffers.ids.as_ptr(),
        buffers.raw.as_ptr(),
        buffers.candidate.as_ptr(),
        buffers.prefix.as_ptr(),
    );
    let mut stream = buffers.stream(layout);
    let (mut ids, mut prefix, mut index) = (Vec::new(), String::new(), 0);
    let mut emitted = Vec::new();
    for (step, id) in tokens.iter().enumerate() {
        let expected = tokenizers::tokenizer::step_decode_stream(
            snapshot,
            vec![*id],
            skip,
            &mut ids,
            &mut prefix,
            &mut index,
        )
        .unwrap();
        let actual = stream.step(*id).unwrap().map(str::to_owned);
        assert_eq!(actual, expected, "step {step}, token {id}, skip={skip}");
        assert_eq!(stream.retained_ids(), ids, "retained IDs step {step}");
        assert_eq!(stream.prefix(), prefix, "prefix step {step}");
        assert_eq!(stream.prefix_index(), index, "prefix index step {step}");
        assert_eq!(stream.successful_calls(), step + 1);
        emitted.push(actual);
    }
    let residual = snapshot.decode(&ids, skip).unwrap();
    let expected_finish = if residual.len() > prefix.len() {
        Err(DecodeStorageError::IncompleteByteSequence)
    } else {
        Ok(())
    };
    assert_eq!(stream.finish(), expected_finish);
    // Finish does not flush, clear state, add a successful call, or terminate
    // the underlying HF helper. Its candidate is the actual residual decode.
    assert_eq!(stream.retained_ids(), ids);
    assert_eq!(stream.prefix(), prefix);
    assert_eq!(stream.prefix_index(), index);
    assert_eq!(stream.candidate(), residual);
    assert_eq!(stream.successful_calls(), tokens.len());
    assert_eq!(stream.finish(), expected_finish);
    let before = (
        stream.retained_ids().to_vec(),
        stream.prefix().to_owned(),
        stream.candidate().to_owned(),
        stream.prefix_index(),
    );
    assert_eq!(stream.step(999), Err(DecodeStorageError::CallLimit));
    assert_eq!(
        (
            stream.retained_ids().to_vec(),
            stream.prefix().to_owned(),
            stream.candidate().to_owned(),
            stream.prefix_index()
        ),
        before
    );
    drop(stream);
    assert_eq!(
        (
            buffers.ids.as_ptr(),
            buffers.raw.as_ptr(),
            buffers.candidate.as_ptr(),
            buffers.prefix.as_ptr()
        ),
        addresses
    );
    emitted
}

#[test]
fn plain_join_preserves_empty_unknown_special_and_unicode_frontiers() {
    let mut tokenizer = wrapper(
        &[("", 0), ("hello", 7), ("é", 11), ("🦀", 21), ("<stop>", 80)],
        false,
    );
    tokenizer
        .add_special_tokens([AddedToken::from("<stop>", true)])
        .unwrap();
    let snapshot = tokenizer.snapshot();
    let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
    assert_eq!(
        snapshot.decode(&[0, 7, 999, 0, 11, 21, 80], false).unwrap(),
        " hello  é 🦀 <stop>"
    );
    for skip in [false, true] {
        for tokens in [
            &[][..],
            &[999, 0, 999, 0],
            &[0, 7, 999, 0, 11, 21, 80],
            &[80, 7, 80, 0, 80],
        ] {
            differential(&snapshot, &source, tokens, skip);
        }
    }
}

// Test inputs use the published ByteLevel alphabet, independently obtained from
// HF's pre-tokenizer for each byte spelling. Explicit high-byte spellings below
// also anchor the multi-token UTF-8 examples without calling our compiler map.
fn byte_wrapper() -> Tokenizer {
    let mut alphabet: Vec<char> = ByteLevel::alphabet().into_iter().collect();
    alphabet.sort_unstable();
    let mut pieces: Vec<(String, u32)> = alphabet
        .into_iter()
        .enumerate()
        .map(|(id, c)| (c.to_string(), id as u32))
        .collect();
    pieces.extend([
        ("Ġ🦀".into(), 300),
        ("�".into(), 301),
        ("".into(), 302),
        ("<stop>".into(), 303),
    ]);
    let mut hf = tokenizers::Tokenizer::new(model(pieces));
    hf.with_decoder(Some(ByteLevel::default()));
    hf.add_special_tokens([AddedToken::from("<stop>", true)])
        .unwrap();
    Tokenizer::from_tokenizer(hf)
}

fn ids(snapshot: &TokenizerSnapshot, spellings: &[&str]) -> Vec<u32> {
    spellings
        .iter()
        .map(|s| snapshot.token_to_id(s).unwrap())
        .collect()
}

#[test]
fn bytelevel_matches_hf_all_bytes_partial_unicode_invalid_and_full_token_fallback() {
    let tokenizer = byte_wrapper();
    let snapshot = tokenizer.snapshot();
    let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
    assert_eq!(snapshot.decode(&[300], false).unwrap(), "Ġ🦀"); // not " 🦀"
    let cases = [
        ids(&snapshot, &["h", "i", "Ġ", "Ã", "©", "!"]),
        ids(&snapshot, &["ð", "Ł", "¦", "Ģ", "!"]), // U+1F980 crab
        ids(&snapshot, &["Ã"]),
        ids(&snapshot, &["â", "Ĥ"]), // incomplete three-byte scalar
        ids(&snapshot, &["ÿ", "x", "À", "¯", "!"]), // invalid and overlong
        ids(&snapshot, &["í", "ł", "Ģ", "a"]), // UTF-8 surrogate
        vec![301],                   // literal U+FFFD also suppresses output and fails finish
        vec![302, 999, 300, 303, 302, 301, 999],
        (0..256).collect(),
    ];
    for skip in [false, true] {
        for tokens in &cases {
            differential(&snapshot, &source, tokens, skip);
        }
    }
    let crab = differential(&snapshot, &source, &cases[1], false);
    assert_eq!(
        crab,
        vec![None, None, None, Some("🦀".into()), Some("!".into())]
    );
    let fallback = differential(&snapshot, &source, &[300], false);
    assert_eq!(fallback, vec![Some("Ġ🦀".into())]);
}

#[test]
fn lossy_destination_matches_standard_invalid_subsequence_boundaries() {
    // Every one- and two-byte input, plus incomplete/overlong/surrogate/upper
    // boundary four-byte sequences, exercises replacement grouping, not merely
    // a copied capacity formula. Production never constructs a Cow/String.
    let mut destination = [0u8; 12];
    for first in 0..=255u8 {
        for second in 0..=255u8 {
            for raw in [&[first][..], &[first, second][..]] {
                let n = lossy_into(raw, &mut destination);
                assert_eq!(&destination[..n], String::from_utf8_lossy(raw).as_bytes());
            }
        }
    }
    for raw in [
        &[0xf0, 0x9f, 0xa6, 0x80][..],
        &[0xf0, 0x9f, 0xa6],
        &[0xed, 0xa0, 0x80],
        &[0xf4, 0x8f, 0xbf, 0xbf],
        &[0xf4, 0x90, 0x80, 0x80],
        &[0xe0, 0x80, 0x80],
        &[0xe2, 0x82, b'a', 0xff],
        &[0xf0, b'a', 0x9f, 0x80],
    ] {
        let n = lossy_into(raw, &mut destination);
        assert_eq!(&destination[..n], String::from_utf8_lossy(raw).as_bytes());
    }
}

#[test]
fn added_reverse_ids_shadow_model_without_losing_model_spelling_aliases() {
    let mut tokenizer = wrapper(&[("base", 0), ("[UNK]", 1)], false);
    tokenizer
        .add_tokens([AddedToken::from("same", false)])
        .unwrap();
    let added_id = tokenizer.token_to_id("same").unwrap();
    assert_eq!(added_id, 2);
    // Public model replacement leaves the actual added reverse owner intact.
    // get_vocab(true) erases model ID99 under the same spelling key; decoding
    // must still support it, as well as added ID2 shadowing model's "hidden".
    tokenizer.with_model(model([
        ("same".into(), 99),
        ("hidden".into(), 2),
        ("plain".into(), 10),
    ]));
    let snapshot = tokenizer.snapshot();
    assert!(!snapshot.get_vocab(true).values().any(|id| *id == 99));
    assert_eq!(
        snapshot.decode(&[2, 99, 10], false).unwrap(),
        "same same plain"
    );
    let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
    assert_eq!(source.token_count(), 3);
    differential(&snapshot, &source, &[2, 99, 10, 2, 999, 99], false);
}

#[test]
fn normalized_added_spellings_and_special_membership_follow_actual_hf_lookup() {
    let mut tokenizer = wrapper(&[("[UNK]", 0), ("A", 1)], true);
    tokenizer
        .with_normalizer(Some(tokenizers::normalizers::Lowercase))
        .unwrap();
    tokenizer
        .add_tokens([AddedToken::from(" Hi", false).normalized(true)])
        .unwrap();
    tokenizer
        .add_special_tokens([
            AddedToken::from("<special>", true),
            AddedToken::from("<LOUD>", true).normalized(true),
        ])
        .unwrap();
    let snapshot = tokenizer.snapshot();
    let added = snapshot.token_to_id(" Hi").unwrap();
    let special = snapshot.token_to_id("<special>").unwrap();
    let normalized_special = snapshot.token_to_id("<LOUD>").unwrap();
    assert_eq!(snapshot.id_to_token(added).as_deref(), Some(" hi"));
    assert_eq!(snapshot.decode(&[added], false).unwrap(), " hi");
    // HF tests special membership on the returned normalized spelling, which
    // may differ from the registered special content. Preserve that behavior.
    assert_eq!(
        snapshot.decode(&[normalized_special], true).unwrap(),
        "<loud>"
    );
    assert_eq!(snapshot.decode(&[special], true).unwrap(), "");
    let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
    for skip in [false, true] {
        differential(
            &snapshot,
            &source,
            &[added, special, 1, normalized_special, added, special],
            skip,
        );
    }
}

#[test]
fn compiled_source_survives_snapshot_drop_and_wrapper_cow_configuration_changes() {
    let mut tokenizer = wrapper(&[("Ġ", 0), ("hello", 1)], false);
    let old_snapshot = tokenizer.snapshot();
    let old = PreparedDecodeSource::prepare(&old_snapshot).unwrap();
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
        .add_tokens([AddedToken::from("tail", false)])
        .unwrap();
    let new_snapshot = tokenizer.snapshot();
    let new = PreparedDecodeSource::prepare(&new_snapshot).unwrap();
    differential(&old_snapshot, &old, &[0, 1, 0, 1], false);
    differential(&new_snapshot, &new, &[0, 1, 0, 1], false);
    drop((old_snapshot, new_snapshot, tokenizer));
    for (source, expected) in [(&old, "Ġ hello"), (&new, " hello")] {
        let layout = DecodeStreamLayout::for_source(source, 2, false).unwrap();
        let mut buffers = Buffers::new(&layout);
        let mut stream = buffers.stream(layout);
        let first = stream.step(0).unwrap().unwrap().to_owned();
        let second = stream.step(1).unwrap().unwrap();
        assert_eq!(first + second, expected);
        assert_eq!(stream.finish(), Ok(()));
    }
}

#[test]
fn all_destination_extents_reject_minus_one_or_extra_before_any_write() {
    let tokenizer = byte_wrapper();
    let source = PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap();
    for field in 0..4 {
        for extra in [false, true] {
            let layout = DecodeStreamLayout::for_source(&source, 4, false).unwrap();
            let mut buffers = Buffers::new(&layout);
            match field {
                0 => {
                    if extra {
                        buffers.ids.push(0xab);
                    } else {
                        buffers.ids.pop();
                    }
                }
                1 => {
                    if extra {
                        buffers.raw.push(0xab);
                    } else {
                        buffers.raw.pop();
                    }
                }
                2 => {
                    if extra {
                        buffers.candidate.push(0xab);
                    } else {
                        buffers.candidate.pop();
                    }
                }
                _ => {
                    if extra {
                        buffers.prefix.push(0xab);
                    } else {
                        buffers.prefix.pop();
                    }
                }
            }
            let before = (
                buffers.ids.clone(),
                buffers.raw.clone(),
                buffers.candidate.clone(),
                buffers.prefix.clone(),
            );
            let error = DecodeStreamState::new(
                layout,
                &mut buffers.ids,
                &mut buffers.raw,
                &mut buffers.candidate,
                &mut buffers.prefix,
            )
            .unwrap_err();
            let expected_field = [
                DecodeBuffer::History,
                DecodeBuffer::Raw,
                DecodeBuffer::Candidate,
                DecodeBuffer::Prefix,
            ][field];
            assert!(
                matches!(error, DecodeStorageError::Extent { buffer, expected, actual } if buffer == expected_field && if extra { actual == expected + 1 } else { actual + 1 == expected })
            );
            assert_eq!(
                (buffers.ids, buffers.raw, buffers.candidate, buffers.prefix),
                before
            );
        }
    }
    differential(
        &tokenizer.snapshot(),
        &source,
        &ids(&tokenizer.snapshot(), &["Ã", "©", "h", "i"]),
        false,
    );
}

#[test]
fn successful_call_limit_survives_repeated_compaction_and_zero_call_finish() {
    let tokenizer = wrapper(&[("a", 0)], true);
    let source = PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap();
    let layout = DecodeStreamLayout::for_source(&source, 12, false).unwrap();
    let mut buffers = Buffers::new(&layout);
    let mut stream = buffers.stream(layout);
    for n in 1..=12 {
        assert_eq!(stream.step(0), Ok(Some("a")));
        assert_eq!(stream.retained_ids(), &[0]);
        assert_eq!(stream.successful_calls(), n);
    }
    let before = (
        stream.ids.to_vec(),
        stream.raw.to_vec(),
        stream.candidate.to_vec(),
        stream.prefix.to_vec(),
        stream.prefix_index(),
    );
    assert_eq!(stream.step(0), Err(DecodeStorageError::CallLimit));
    assert_eq!(
        (
            stream.ids.to_vec(),
            stream.raw.to_vec(),
            stream.candidate.to_vec(),
            stream.prefix.to_vec(),
            stream.prefix_index()
        ),
        before
    );
    assert_eq!(stream.finish(), Ok(()));
    // Zero-output completion has no buffers and still performs the empty finish
    // check. Initial cancellation is simply no step/finish; no hidden flush.
    let zero = DecodeStreamLayout::for_source(&source, 0, false).unwrap();
    assert_eq!(zero.buffer_bytes(), 0);
    let mut empty = Buffers::new(&zero);
    let mut zero_stream = empty.stream(zero);
    assert_eq!(zero_stream.successful_calls(), 0);
    assert_eq!(zero_stream.finish(), Ok(()));
    assert_eq!(zero_stream.step(0), Err(DecodeStorageError::CallLimit));
    assert_eq!(zero_stream.successful_calls(), 0);
}

#[test]
fn invalid_prefix_preserves_hf_failure_frontier_and_bounded_retry_rejection() {
    let tokenizer = wrapper(&[("a", 0), ("b", 1)], false);
    let snapshot = tokenizer.snapshot();
    let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
    let layout = DecodeStreamLayout::for_source(&source, 3, false).unwrap();
    let mut buffers = Buffers::new(&layout);
    let mut stream = buffers.stream(layout);
    // Deliberately seed the same inconsistent diagnostic frontier in each
    // implementation; no public restore API or natural failure claim is made.
    stream.ids[0] = 0;
    stream.progress.ids_len = 1;
    stream.prefix[0] = b'z';
    stream.progress.prefix_len = 1;
    stream.progress.prefix_index = 1;
    let (mut hf_ids, mut hf_prefix, mut hf_index) = (vec![0], "z".to_owned(), 1);
    for id in [1, 0] {
        let expected = tokenizers::tokenizer::step_decode_stream(
            &snapshot,
            vec![id],
            false,
            &mut hf_ids,
            &mut hf_prefix,
            &mut hf_index,
        )
        .unwrap_err();
        let Some(tokenizers::tokenizer::DecodeStreamError::InvalidPrefix {
            token_id,
            expected_prefix,
            actual_string,
        }) = expected.downcast_ref::<tokenizers::tokenizer::DecodeStreamError>()
        else {
            panic!("wrong HF error: {expected}")
        };
        assert_eq!(
            stream.step(id),
            Err(DecodeStorageError::InvalidPrefix {
                token_id: *token_id,
                expected_bytes: expected_prefix.len(),
                actual_bytes: actual_string.len()
            })
        );
        assert_eq!(stream.retained_ids(), hf_ids);
        assert_eq!(stream.prefix(), expected_prefix);
        assert_eq!(stream.prefix(), hf_prefix);
        assert_eq!(stream.candidate(), actual_string);
        assert_eq!(stream.prefix_index(), hf_index);
        assert_eq!(stream.successful_calls(), 0);
    }
    let before = (
        stream.ids.to_vec(),
        stream.raw.to_vec(),
        stream.candidate.to_vec(),
        stream.prefix.to_vec(),
        stream.prefix_index(),
    );
    assert_eq!(stream.step(1), Err(DecodeStorageError::HistoryLimit));
    assert_eq!(
        (
            stream.ids.to_vec(),
            stream.raw.to_vec(),
            stream.candidate.to_vec(),
            stream.prefix.to_vec(),
            stream.prefix_index()
        ),
        before
    );
}

#[test]
fn unsupported_decoder_and_unrepresentable_extents_reject_without_legacy_change() {
    let mut tokenizer = wrapper(&[("hello", 0), ("##s", 1)], false);
    tokenizer.with_decoder(Some(tokenizers::decoders::wordpiece::WordPiece::new(
        "##".into(),
        false,
    )));
    let snapshot = tokenizer.snapshot();
    assert_eq!(snapshot.decode(&[0, 1], false).unwrap(), "hellos");
    assert!(matches!(
        PreparedDecodeSource::prepare(&snapshot),
        Err(DecodeSourceError::UnsupportedDecoder)
    ));
    assert_eq!(snapshot.decode(&[0, 1], false).unwrap(), "hellos");
    let plain = wrapper(&[("longer", 0)], false);
    let bytes = wrapper(&[("longer", 0)], true);
    let empty = wrapper(&[("", 0)], true);
    let empty_source = PreparedDecodeSource::prepare(&empty.snapshot()).unwrap();
    assert!(matches!(
        DecodeStreamLayout::for_source(&empty_source, (isize::MAX as usize / 4) + 1, false),
        Err(DecodeStorageError::Overflow)
    ));
    for wrapper in [plain, bytes] {
        let source = PreparedDecodeSource::prepare(&wrapper.snapshot()).unwrap();
        assert!(matches!(
            DecodeStreamLayout::for_source(&source, usize::MAX, false),
            Err(DecodeStorageError::Overflow)
        ));
    }
}

mod pipelines;

mod owned;

mod compiler;
