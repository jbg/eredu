use super::*;
use eredu_gguf::{MetadataArray, MetadataValue};
use serde_json::{json, Value};
use std::collections::HashMap;

pub(super) fn pipeline(order: char, strip: bool) -> Value {
    let replace = json!({"type":"Replace","pattern":{"String":"▁"},"content":" "});
    let mut rows = match order {
        'A' => vec![
            replace,
            json!({"type":"ByteFallback"}),
            json!({"type":"Fuse"}),
        ],
        'B' | 'C' => vec![
            json!({"type":"ByteFallback"}),
            json!({"type":"Fuse"}),
            replace,
        ],
        _ => panic!("fixture order"),
    };
    if order == 'C' {
        rows.push(json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":true}));
    }
    if strip {
        rows.push(json!({"type":"Strip","content":" ","start":1,"stop":0}));
    }
    json!({"type":"Sequence","decoders":rows})
}

pub(super) fn with_pipeline(config: Value) -> Tokenizer {
    let mut pieces: Vec<_> = (0..=255u32).map(|i| (format!("<0x{i:02X}>"), i)).collect();
    pieces.extend([
        ("".into(), 300),
        ("a".into(), 301),
        ("b".into(), 302),
        ("▁".into(), 303),
        ("Ġ".into(), 304),
        ("🦀".into(), 305),
        ("  \n".into(), 306),
        ("é".into(), 307),
        ("<stop>".into(), 308),
        ("<0xff>".into(), 309),
        ("<0x+F>".into(), 310),
        ("<0xé>".into(), 311),
        ("<0xGG>".into(), 312),
        ("<0XFF>".into(), 313),
        ("<0xF>".into(), 314),
        ("[UNK]".into(), 315),
    ]);
    let mut hf = tokenizers::Tokenizer::new(model(pieces));
    hf.with_decoder(Some(
        serde_json::from_value::<DecoderWrapper>(config).unwrap(),
    ));
    hf.add_special_tokens([AddedToken::from("<stop>", true)])
        .unwrap();
    Tokenizer::from_tokenizer(hf)
}

// Compare the complete real HF frontier, including post-append InvalidPrefix.
// Candidate is the pre-compaction decode, not merely the emitted suffix.
fn trace(
    snapshot: &TokenizerSnapshot,
    tokens: &[u32],
    skip: bool,
) -> Vec<Result<Option<String>, String>> {
    let source = PreparedDecodeSource::prepare(snapshot).unwrap();
    trace_source(snapshot, &source, tokens, skip)
}

pub(super) fn trace_source(
    snapshot: &TokenizerSnapshot,
    source: &PreparedDecodeSource,
    tokens: &[u32],
    skip: bool,
) -> Vec<Result<Option<String>, String>> {
    let layout = DecodeStreamLayout::for_source(source, tokens.len(), skip).unwrap();
    let mut buffers = Buffers::new(&layout);
    let addresses = (
        buffers.ids.as_ptr(),
        buffers.raw.as_ptr(),
        buffers.candidate.as_ptr(),
        buffers.prefix.as_ptr(),
    );
    let mut stream = buffers.stream(layout);
    let (mut ids, mut prefix, mut index, mut successful) = (vec![], String::new(), 0, 0);
    let mut results = vec![];
    for &id in tokens {
        let mut appended = ids.clone();
        appended.push(id);
        let candidate = snapshot.decode(&appended, skip).unwrap();
        let expected = tokenizers::tokenizer::step_decode_stream(
            snapshot,
            vec![id],
            skip,
            &mut ids,
            &mut prefix,
            &mut index,
        );
        let actual = stream.step(id).map(|text| text.map(str::to_owned));
        match expected {
            Ok(text) => {
                assert_eq!(actual, Ok(text.clone()), "id={id}, skip={skip}");
                successful += 1;
                results.push(Ok(text));
            }
            Err(error) => {
                let Some(tokenizers::tokenizer::DecodeStreamError::InvalidPrefix {
                    token_id,
                    expected_prefix,
                    actual_string,
                }) = error.downcast_ref()
                else {
                    panic!("unexpected HF error: {error}");
                };
                assert_eq!(
                    actual,
                    Err(DecodeStorageError::InvalidPrefix {
                        token_id: *token_id,
                        expected_bytes: expected_prefix.len(),
                        actual_bytes: actual_string.len()
                    })
                );
                assert_eq!(stream.prefix(), expected_prefix);
                assert_eq!(stream.candidate(), actual_string);
                results.push(Err(actual_string.clone()));
            }
        }
        assert_eq!(stream.retained_ids(), ids, "IDs at token {id}");
        assert_eq!(stream.prefix(), prefix, "prefix at token {id}");
        assert_eq!(stream.prefix_index(), index, "index at token {id}");
        assert_eq!(stream.candidate(), candidate, "candidate at token {id}");
        assert_eq!(stream.successful_calls(), successful);
    }
    let remaining = snapshot.decode(&ids, skip).unwrap();
    let finish = if remaining.len() > prefix.len() {
        Err(DecodeStorageError::IncompleteByteSequence)
    } else {
        Ok(())
    };
    assert_eq!(stream.finish(), finish);
    assert_eq!(stream.finish(), finish);
    assert_eq!(stream.candidate(), remaining);
    assert_eq!(stream.retained_ids(), ids);
    assert_eq!(stream.prefix(), prefix);
    assert_eq!(stream.prefix_index(), index);
    assert_eq!(stream.successful_calls(), successful);
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
    results
}

fn decode_fixed(snapshot: &TokenizerSnapshot, tokens: &[u32], skip: bool) -> String {
    let source = PreparedDecodeSource::prepare(snapshot).unwrap();
    let layout = DecodeStreamLayout::for_source(&source, tokens.len(), skip).unwrap();
    let mut buffers = Buffers::new(&layout);
    let n = source.decode_into(tokens, skip, &mut buffers.raw, &mut buffers.candidate);
    let actual = std::str::from_utf8(&buffers.candidate[..n]).unwrap();
    assert_eq!(actual, snapshot.decode(tokens, skip).unwrap());
    actual.to_owned()
}

#[test]
fn singleton_bytelevel_matches_actual_lfm_configuration_and_direct_stream() {
    let tokenizer = byte_wrapper();
    let direct = tokenizer.snapshot();
    let mut hf = (*direct).clone();
    // Exact decoder object from all four archived LFM tokenizer revisions.
    hf.with_decoder(Some(serde_json::from_value::<DecoderWrapper>(json!({"type":"Sequence","decoders":[{"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}]})).unwrap()));
    let wrapped = Tokenizer::from_tokenizer(hf).snapshot();
    let tokens = [
        ids(&direct, &["h", "Ġ", "Ã", "©", "ð", "Ł", "¦", "Ģ"]),
        vec![300, 302, 999, 303, 301],
    ]
    .concat();
    for skip in [false, true] {
        assert_eq!(
            trace(&wrapped, &tokens, skip),
            trace(&direct, &tokens, skip)
        );
    }
}

#[test]
fn actual_gguf_constructors_preserve_all_six_pipeline_frontiers() {
    for (architecture, model_tag) in [("gemma4", "llama"), ("llama", "llama"), ("llama", "gpt2")] {
        for strip in [false, true] {
            let pieces = [
                "<unk>", "▁", "a", "b", "é", "🦀", "Ġ", "\n", "<0x61>", "<0xFF>", "<0xC3>",
                "<0xA9>", "<0xE2>", "<0x96>", "<0x81>",
            ];
            let metadata = HashMap::from([
                (
                    "general.architecture".into(),
                    MetadataValue::String(architecture.into()),
                ),
                (
                    "tokenizer.ggml.model".into(),
                    MetadataValue::String(model_tag.into()),
                ),
                (
                    "tokenizer.ggml.add_space_prefix".into(),
                    MetadataValue::Bool(strip),
                ),
                (
                    "tokenizer.ggml.tokens".into(),
                    MetadataValue::Array(MetadataArray::String(
                        pieces.iter().map(|s| (*s).into()).collect(),
                    )),
                ),
                (
                    "tokenizer.ggml.scores".into(),
                    MetadataValue::Array(MetadataArray::Float32(vec![1.; pieces.len()])),
                ),
                (
                    "tokenizer.ggml.merges".into(),
                    MetadataValue::Array(MetadataArray::String(vec![])),
                ),
                (
                    "tokenizer.ggml.unknown_token_id".into(),
                    MetadataValue::Uint32(0),
                ),
            ]);
            let actual = crate::gguf::from_metadata(&metadata).unwrap().unwrap();
            let tokenizer = Tokenizer::from_tokenizer(actual.tokenizer);
            let snapshot = tokenizer.snapshot();
            for skip in [false, true] {
                for sequence in [
                    vec![],
                    ids(&snapshot, &["▁", "a", "\n", "é", "🦀"]),
                    ids(&snapshot, &["<0xC3>", "<0xA9>", "b"]),
                    ids(&snapshot, &["<0xE2>", "<0x96>", "<0x81>", "a"]),
                    ids(&snapshot, &["<0x61>", "<0xFF>", "b"]),
                    vec![0, 999, 1, 2],
                ] {
                    trace(&snapshot, &sequence, skip);
                    decode_fixed(&snapshot, &sequence, skip);
                }
            }
        }
    }
}

#[test]
fn nanbeige_decoder_configuration_and_ordered_byte_runs_match_hf() {
    // Exact decoder from the released-derived Nanbeige 0e137298 fixture,
    // SHA256 9e63c959710c22fa1fde5200d4b36867d5797204e652573a36b73b96b1352eed.
    // This vocabulary is deliberately synthetic; no full release claim follows.
    let nanbeige = json!({"type":"Sequence","decoders":[{"type":"Replace","pattern":{"String":"▁"},"content":" "},{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Strip","content":" ","start":1,"stop":0}]});
    let tokenizer = with_pipeline(nanbeige);
    let snapshot = tokenizer.snapshot();
    assert_eq!(decode_fixed(&snapshot, &[303, 303, 301], false), " a");
    assert_eq!(decode_fixed(&snapshot, &[306], false), " \n");
    for order in ['A', 'B', 'C'] {
        for strip in [false, true] {
            let tokenizer = with_pipeline(pipeline(order, strip));
            let snapshot = tokenizer.snapshot();
            for skip in [false, true] {
                for sequence in [
                    vec![0xC3, 0xA9, 302],
                    vec![0x61, 0xFF, 302],
                    vec![0x61, 0xFF, 300, 302],
                    vec![0xE2, 0x96, 0x81, 301],
                    vec![0xE2, 0x96, 300, 0x81, 302],
                    vec![0xC3, 999, 308, 0xA9, 302],
                    vec![300, 300, 303, 303, 301, 306],
                    vec![309, 310, 311, 312, 313, 314, 302],
                ] {
                    trace(&snapshot, &sequence, skip);
                    decode_fixed(&snapshot, &sequence, skip);
                }
            }
            assert_eq!(decode_fixed(&snapshot, &[0x61, 0xFF], false), "��");
            assert_eq!(decode_fixed(&snapshot, &[0x61, 300, 0xFF], false), "a�");
            assert_eq!(
                decode_fixed(&snapshot, &[0xC3, 999, 308, 0xA9], true),
                if order == 'C' { "�" } else { "é" }
            );
        }
    }
}

#[test]
fn post_fuse_bytelevel_uses_one_whole_token_and_replace_keeps_stage_order() {
    let a = with_pipeline(pipeline('A', false));
    let b = with_pipeline(pipeline('B', false));
    let c = with_pipeline(pipeline('C', false));
    assert_eq!(decode_fixed(&a.snapshot(), &[0xE2, 0x96, 0x81], false), "▁");
    assert_eq!(decode_fixed(&b.snapshot(), &[0xE2, 0x96, 0x81], false), " ");
    assert_eq!(decode_fixed(&c.snapshot(), &[304, 305], false), "Ġ🦀");
    assert_eq!(decode_fixed(&c.snapshot(), &[304, 301], false), " a");
    assert_eq!(decode_fixed(&c.snapshot(), &[0xFF, 304], false), "�Ġ");
    // Reuse one compiled source and exact destinations while comparing every
    // two-byte run: whole-run rejection and ByteLevel's later transform differ.
    for snapshot in [a.snapshot(), b.snapshot(), c.snapshot()] {
        let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
        let layout = DecodeStreamLayout::for_source(&source, 2, false).unwrap();
        let mut buffers = Buffers::new(&layout);
        for byte in 0..=255u32 {
            for second in 0..=255u32 {
                let ids = [byte, second];
                let n = source.decode_into(&ids, false, &mut buffers.raw, &mut buffers.candidate);
                assert_eq!(
                    &buffers.candidate[..n],
                    snapshot.decode(&ids, false).unwrap().as_bytes(),
                    "bytes={ids:?}"
                );
            }
        }
    }
}

#[test]
fn fresh_fallback_invalid_prefix_preserves_failure_then_history_limit() {
    let tokenizer = with_pipeline(pipeline('B', false));
    let snapshot = tokenizer.snapshot();
    assert_eq!(
        trace(&snapshot, &[0x61, 0xFF, 302, 301], false),
        vec![
            Ok(Some("a".into())),
            Ok(None),
            Err("��b".into()),
            Err("��ba".into())
        ]
    );
    let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
    let layout = DecodeStreamLayout::for_source(&source, 4, false).unwrap();
    let mut buffers = Buffers::new(&layout);
    let mut stream = buffers.stream(layout);
    assert_eq!(stream.step(0x61), Ok(Some("a")));
    assert_eq!(stream.step(0xFF), Ok(None));
    assert!(matches!(
        stream.step(302),
        Err(DecodeStorageError::InvalidPrefix { token_id: 302, .. })
    ));
    assert!(matches!(
        stream.step(301),
        Err(DecodeStorageError::InvalidPrefix { token_id: 301, .. })
    ));
    assert_eq!(stream.retained_ids(), &[0x61, 0xFF, 302, 301]);
    assert_eq!(stream.successful_calls(), 2);
    let before = (
        stream.ids.to_vec(),
        stream.raw.to_vec(),
        stream.candidate.to_vec(),
        stream.prefix.to_vec(),
        stream.prefix_index(),
    );
    assert_eq!(stream.step(300), Err(DecodeStorageError::HistoryLimit));
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
fn pipeline_extents_reject_before_mutation_and_calls_do_not_recycle() {
    for order in ['A', 'B', 'C'] {
        let tokenizer = with_pipeline(pipeline(order, false));
        let source = PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap();
        for field in 0..4 {
            let layout = DecodeStreamLayout::for_source(&source, 4, false).unwrap();
            let mut buffers = Buffers::new(&layout);
            match field {
                0 => {
                    buffers.ids.pop();
                }
                1 => {
                    buffers.raw.pop();
                }
                2 => {
                    buffers.candidate.pop();
                }
                _ => {
                    buffers.prefix.pop();
                }
            }
            let before = (
                buffers.ids.clone(),
                buffers.raw.clone(),
                buffers.candidate.clone(),
                buffers.prefix.clone(),
            );
            assert!(matches!(
                DecodeStreamState::new(
                    layout,
                    &mut buffers.ids,
                    &mut buffers.raw,
                    &mut buffers.candidate,
                    &mut buffers.prefix
                ),
                Err(DecodeStorageError::Extent { .. })
            ));
            assert_eq!(
                (buffers.ids, buffers.raw, buffers.candidate, buffers.prefix),
                before
            );
        }
        let layout = DecodeStreamLayout::for_source(&source, 12, false).unwrap();
        let mut buffers = Buffers::new(&layout);
        let mut stream = buffers.stream(layout);
        for n in 1..=12 {
            assert_eq!(stream.step(301), Ok(Some("a")));
            assert_eq!(stream.retained_ids(), &[301]);
            assert_eq!(stream.successful_calls(), n);
        }
        let before = (
            stream.ids.to_vec(),
            stream.raw.to_vec(),
            stream.candidate.to_vec(),
            stream.prefix.to_vec(),
        );
        assert_eq!(stream.step(301), Err(DecodeStorageError::CallLimit));
        assert_eq!(
            (
                stream.ids.to_vec(),
                stream.raw.to_vec(),
                stream.candidate.to_vec(),
                stream.prefix.to_vec()
            ),
            before
        );
        assert!(matches!(
            DecodeStreamLayout::for_source(&source, usize::MAX, false),
            Err(DecodeStorageError::Overflow)
        ));
        let zero = DecodeStreamLayout::for_source(&source, 0, false).unwrap();
        let mut buffers = Buffers::new(&zero);
        let mut stream = buffers.stream(zero);
        assert_eq!(stream.finish(), Ok(()));
        assert_eq!(stream.step(301), Err(DecodeStorageError::CallLimit));
        assert_eq!(stream.successful_calls(), 0);
    }
}

#[test]
fn pipeline_source_keeps_added_shadowing_specials_and_independent_lifetime() {
    let mut tokenizer = wrapper(&[("base", 0), ("[UNK]", 1)], false);
    tokenizer
        .add_tokens([AddedToken::from("<0xC3>", false)])
        .unwrap();
    let added = tokenizer.token_to_id("<0xC3>").unwrap();
    assert_eq!(added, 2);
    tokenizer.with_model(model([
        ("<0xC3>".into(), 99),
        ("shadowed".into(), 2),
        ("<0xA9>".into(), 10),
        ("b".into(), 11),
        ("[UNK]".into(), 0),
    ]));
    tokenizer.with_decoder(Some(
        serde_json::from_value::<DecoderWrapper>(pipeline('B', false)).unwrap(),
    ));
    tokenizer
        .add_special_tokens([AddedToken::from("<0xFF>", true)])
        .unwrap();
    let special = tokenizer.token_to_id("<0xFF>").unwrap();
    let snapshot = tokenizer.snapshot();
    assert!(!snapshot.get_vocab(true).values().any(|id| *id == 99));
    assert_eq!(snapshot.decode(&[2, 10], false).unwrap(), "é");
    assert_eq!(snapshot.decode(&[99, 10], false).unwrap(), "é");
    for skip in [false, true] {
        trace(&snapshot, &[2, 10, 11, 99, 10, special, 11], skip);
    }
    let source = PreparedDecodeSource::prepare(&snapshot).unwrap();
    tokenizer.with_decoder(Some(ByteLevel::default()));
    drop((snapshot, tokenizer));
    let layout = DecodeStreamLayout::for_source(&source, 3, false).unwrap();
    let mut buffers = Buffers::new(&layout);
    let mut stream = buffers.stream(layout);
    assert_eq!(stream.step(2), Ok(None));
    assert_eq!(stream.step(10), Ok(Some("é")));
    assert_eq!(stream.step(11), Ok(Some("b")));
}

#[test]
fn other_sequence_algebra_and_replace_or_strip_configs_stay_typed_unsupported() {
    let mut cases = vec![
        json!({"type":"Sequence","decoders":[]}),
        json!({"type":"Sequence","decoders":[pipeline('A', false)]}),
    ];
    let mut regex = pipeline('A', false);
    regex["decoders"][0]["pattern"] = json!({"Regex":"▁"});
    cases.push(regex);
    let mut content = pipeline('A', false);
    content["decoders"][0]["content"] = json!("xx");
    cases.push(content);
    let mut strip = pipeline('B', true);
    strip["decoders"][3]["start"] = json!(2);
    cases.push(strip);
    let mut stop = pipeline('B', true);
    stop["decoders"][3]["stop"] = json!(1);
    cases.push(stop);
    let mut reordered = pipeline('B', false);
    reordered["decoders"].as_array_mut().unwrap().swap(0, 1);
    cases.push(reordered);
    for config in cases {
        let tokenizer = with_pipeline(config);
        assert!(matches!(
            PreparedDecodeSource::prepare(&tokenizer.snapshot()),
            Err(DecodeSourceError::UnsupportedDecoder)
        ));
    }
    // Empty Sequence is empty-separator joining; it must not become plain join.
    let tokenizer = with_pipeline(json!({"type":"Sequence","decoders":[]}));
    assert_eq!(tokenizer.decode(&[301, 302], false).unwrap(), "ab");
}
