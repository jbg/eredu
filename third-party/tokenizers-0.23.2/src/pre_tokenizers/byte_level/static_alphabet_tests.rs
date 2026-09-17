use super::{alphabet::*, ByteLevel, REGEX_INITIALIZATIONS};
use crate::models::bpe::BPE;
use crate::normalizers::byte_level::ByteLevel as ByteNormalizer;
use crate::{
    Decoder, Encoding, ModelCachePolicy, NormalizedString, Normalizer, OffsetReferential,
    OffsetType, PostProcessor, PreTokenizedString, PreTokenizer, Token, Tokenizer,
};
use ahash::{AHashMap, AHashSet};
use std::sync::atomic::Ordering;

// Independent test oracle: upstream 0.23.2's ordered-list construction, retained
// only here. The production table is computed by disjoint scalar ranges.
fn upstream_alphabet() -> AHashMap<u8, char> {
    let mut bytes = Vec::new();
    bytes.extend(b'!'..=b'~');
    bytes.extend(b'\xA1'..=b'\xAC');
    bytes.extend(b'\xAE'..=b'\xFF');
    let mut scalars: Vec<u32> = bytes.iter().map(|b| u32::from(*b)).collect();
    let mut extra = 0;
    for byte in 0..=255u8 {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            scalars.push(256 + extra);
            extra += 1;
        }
    }
    bytes
        .into_iter()
        .zip(scalars.into_iter().map(|c| char::from_u32(c).unwrap()))
        .collect()
}

fn encoded(input: &str, alphabet: &AHashMap<u8, char>) -> String {
    input.as_bytes().iter().map(|b| alphabet[b]).collect()
}

#[test]
fn all_bytes_and_every_unicode_scalar_match_the_upstream_alphabet() {
    let expected = upstream_alphabet();
    let reverse: AHashMap<char, u8> = expected.iter().map(|(&b, &c)| (c, b)).collect();
    assert_eq!(reverse.len(), 256);
    for byte in 0..=255u8 {
        assert_eq!(BYTE_TO_CHAR[usize::from(byte)], expected[&byte]);
    }
    for scalar in 0..=0x10ffff {
        if let Some(c) = char::from_u32(scalar) {
            assert_eq!(char_to_byte(c), reverse.get(&c).copied(), "{scalar:#x}");
        }
    }
    let alphabet: AHashSet<char> = expected.values().copied().collect();
    assert_eq!(ByteLevel::alphabet(), alphabet);
    assert_eq!(ByteNormalizer::alphabet(), alphabet);

    // Exercise the real decoder with every byte, including invalid UTF-8 runs.
    let all_bytes: Vec<u8> = (0..=255).collect();
    let all_chars = all_bytes.iter().map(|b| expected[b]).collect();
    assert_eq!(
        ByteLevel::default().decode_chain(vec![all_chars]).unwrap(),
        vec![String::from_utf8_lossy(&all_bytes).into_owned()]
    );
}

#[test]
fn unicode_normalization_keeps_exact_original_alignment() {
    let alphabet = upstream_alphabet();
    let input = "A\0\u{7f}\u{a0}é☃🦀\r\n";
    let mut expected_text = String::new();
    let mut expected_alignment = Vec::new();
    for (start, c) in input.char_indices() {
        let end = start + c.len_utf8();
        for byte in &input.as_bytes()[start..end] {
            let mapped = alphabet[byte];
            expected_text.push(mapped);
            // NormalizedString records an original span for every output byte.
            expected_alignment.extend(std::iter::repeat_n((start, end), mapped.len_utf8()));
        }
    }
    let expected = NormalizedString::new(input.into(), expected_text, expected_alignment, 0);
    let mut actual = NormalizedString::from(input);
    ByteNormalizer::new().normalize(&mut actual).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.alignments_original(), expected.alignments_original());
}

#[test]
fn regex_disabled_unicode_prefix_and_offsets_are_preserved() {
    let alphabet = upstream_alphabet();
    for input in ["é☃\u{a0}\tZ\r\n", " leading 🦀", "\0a"] {
        for prefix in [false, true] {
            let byte_level = ByteLevel::new(prefix, true, false);
            let mut actual = PreTokenizedString::from(input);
            byte_level.pre_tokenize(&mut actual).unwrap();
            let logical = if prefix && !input.starts_with(' ') {
                format!(" {input}")
            } else {
                input.to_owned()
            };
            let expected = encoded(&logical, &alphabet);
            let original = actual.get_splits(OffsetReferential::Original, OffsetType::Byte);
            assert_eq!(original.len(), 1);
            assert_eq!(
                (original[0].0, original[0].1),
                (expected.as_str(), (0, input.len()))
            );
            let normalized = actual.get_splits(OffsetReferential::Normalized, OffsetType::Byte);
            assert_eq!(normalized.len(), 1);
            assert_eq!(normalized[0].1, (0, expected.len()));
            assert_eq!(
                byte_level.decode_chain(vec![expected]).unwrap(),
                vec![logical]
            );
        }
    }
}

#[test]
fn decoder_fallback_is_whole_token_and_utf8_spans_token_boundaries() {
    let alphabet = upstream_alphabet();
    let byte_level = ByteLevel::default();
    let split: Vec<String> = "☃🦀"
        .as_bytes()
        .iter()
        .map(|b| alphabet[b].to_string())
        .collect();
    assert_eq!(byte_level.decode_chain(split).unwrap(), vec!["☃🦀"]);
    for invalid in [vec![0xe2, 0x28, 0xa1], vec![0xf0, 0x9f], vec![0xff, b'a']] {
        let tokens = invalid.iter().map(|b| alphabet[b].to_string()).collect();
        assert_eq!(
            byte_level.decode_chain(tokens).unwrap(),
            vec![String::from_utf8_lossy(&invalid).into_owned()]
        );
    }
    // Ġ and é are mapped characters. A later snowman invalidates their mapped
    // prefix for this token only; the next token still decodes its own Ġ.
    assert_eq!(
        byte_level
            .decode_chain(vec!["Ġé☃".into(), "Ġ".into()])
            .unwrap(),
        vec!["Ġé☃ "]
    );
    assert_eq!(
        byte_level
            .decode_chain(vec!["Ġ\0".into(), "A".into()])
            .unwrap(),
        vec!["Ġ\0A"]
    );
}

#[test]
fn postprocessing_preserves_prefix_unicode_space_overflow_and_serialization() {
    let mut base = Encoding::from_tokens(
        vec![
            Token::new(7, "ĠhiĠ".into(), (0, 4)),
            Token::new(8, "\u{2003}x\u{2003}".into(), (4, 11)),
            Token::new(9, "Ġ".into(), (11, 12)),
        ],
        0,
    );
    base.set_overflowing(vec![base.clone()]);
    for prefix in [false, true] {
        for trim in [false, true] {
            for regex in [false, true] {
                let byte_level = ByteLevel::new(prefix, trim, regex);
                let json = serde_json::to_string(&byte_level).unwrap();
                assert_eq!(
                    json,
                    format!(
                        r#"{{"type":"ByteLevel","add_prefix_space":{prefix},"trim_offsets":{trim},"use_regex":{regex}}}"#
                    )
                );
                let restored: ByteLevel = serde_json::from_str(&json).unwrap();
                assert_eq!(restored, byte_level);
                let mut expected = vec![base.clone(), base.clone()];
                for (index, encoding) in expected.iter_mut().enumerate() {
                    if trim {
                        let offsets = [(usize::from(!prefix), 3), (5, 10), (12, 12)];
                        encoding.get_offsets_mut().copy_from_slice(&offsets);
                        encoding.get_overflowing_mut()[0]
                            .get_offsets_mut()
                            .copy_from_slice(&offsets);
                    }
                    encoding.set_sequence_id(index);
                }
                assert_eq!(
                    restored
                        .process_encodings(vec![base.clone(), base.clone()], false)
                        .unwrap(),
                    expected
                );
            }
        }
    }
    let normalizer = ByteNormalizer::new();
    assert_eq!(
        serde_json::to_string(&normalizer).unwrap(),
        r#"{"type":"ByteLevel"}"#
    );
    let restored: ByteNormalizer = serde_json::from_str(r#"{"type":"ByteLevel"}"#).unwrap();
    let mut actual = NormalizedString::from("a b");
    restored.normalize(&mut actual).unwrap();
    assert_eq!(actual.get(), "aĠb");
}

fn tokenizer(policy: ModelCachePolicy) -> Tokenizer {
    let vocab: AHashMap<String, u32> = upstream_alphabet()
        .into_iter()
        .map(|(b, c)| (c.to_string(), u32::from(b)))
        .collect();
    let model = BPE::builder()
        .vocab_and_merges(vocab, vec![])
        .cache_policy(policy)
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(ByteLevel::new(false, false, false)));
    tokenizer.with_decoder(Some(ByteLevel::new(false, false, false)));
    tokenizer
}

#[test]
fn persistent_workers_encode_with_cloned_and_restored_sources_after_owner_drop() {
    let mut workers = Vec::new();
    for _ in 0..2 {
        let (send, receive) = std::sync::mpsc::channel();
        let (done, wait) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            while receive.recv().unwrap() {
                let legacy = tokenizer(ModelCachePolicy::Legacy);
                let original = tokenizer(ModelCachePolicy::NoModelCaches);
                let cloned = original.clone();
                let serialized = serde_json::to_vec(&original).unwrap();
                let restored = Tokenizer::from_bytes_with_cache_policy(
                    &serialized,
                    ModelCachePolicy::NoModelCaches,
                )
                .unwrap();
                drop(original);
                let input = "\0\u{7f}\u{a0}Hé☃ 🦀\r\n";
                let expected_ids: Vec<u32> =
                    input.as_bytes().iter().map(|&b| u32::from(b)).collect();
                let mut expected_offsets = Vec::new();
                for (start, c) in input.char_indices() {
                    expected_offsets.extend(std::iter::repeat_n(
                        (start, start + c.len_utf8()),
                        c.len_utf8(),
                    ));
                }
                assert_eq!(legacy.model_cache_policy(), ModelCachePolicy::Legacy);
                for source in [&cloned, &restored, &legacy] {
                    let output = source.encode(input, false).unwrap();
                    assert_eq!(output.get_ids(), expected_ids);
                    assert_eq!(output.get_offsets(), expected_offsets);
                    assert_eq!(source.decode(output.get_ids(), false).unwrap(), input);
                }
                assert_eq!(cloned.model_cache_policy(), ModelCachePolicy::NoModelCaches);
                assert_eq!(
                    restored.model_cache_policy(),
                    ModelCachePolicy::NoModelCaches
                );
                done.send(()).unwrap();
            }
        });
        workers.push((send, wait, worker));
    }
    for _ in 0..3 {
        for (send, _, _) in &workers {
            send.send(true).unwrap();
        }
        for (_, wait, _) in &workers {
            wait.recv().unwrap();
        }
    }
    for (send, _, worker) in workers {
        send.send(false).unwrap();
        worker.join().unwrap();
    }
}

#[test]
fn regex_is_initialized_only_by_enabled_work_in_a_fresh_process() {
    const CHILD: &str = "EREDU_BYTELEVEL_REGEX_CHILD";
    const TEST: &str = "pre_tokenizers::byte_level::static_alphabet_tests::regex_is_initialized_only_by_enabled_work_in_a_fresh_process";
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
    assert_eq!(REGEX_INITIALIZATIONS.load(Ordering::SeqCst), 0);
    for _ in 0..3 {
        for prefix in [false, true] {
            let mut input = PreTokenizedString::from("é☃ two\n");
            ByteLevel::new(prefix, true, false)
                .pre_tokenize(&mut input)
                .unwrap();
            assert_eq!(
                input
                    .get_splits(OffsetReferential::Original, OffsetType::Byte)
                    .len(),
                1
            );
        }
    }
    assert_eq!(REGEX_INITIALIZATIONS.load(Ordering::SeqCst), 0);
    let mut input = PreTokenizedString::from("hello world!");
    ByteLevel::new(false, true, true)
        .pre_tokenize(&mut input)
        .unwrap();
    assert_eq!(
        input
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .len(),
        3
    );
    assert_eq!(REGEX_INITIALIZATIONS.load(Ordering::SeqCst), 1);
    ByteLevel::new(false, true, true)
        .pre_tokenize(&mut input)
        .unwrap();
    assert_eq!(REGEX_INITIALIZATIONS.load(Ordering::SeqCst), 1);
}
