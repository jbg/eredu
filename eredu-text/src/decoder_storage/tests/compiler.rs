use super::*;
use std::{error::Error as _, mem::size_of};
use tokenizers::models::{bpe::BPE, unigram::Unigram, wordpiece::WordPiece};

#[test]
fn plan_prices_duplicate_sparse_ids_while_publication_keeps_unique_destinations() {
    let mut tokenizer = wrapper(&[("base", 0), ("[UNK]", 1)], false);
    tokenizer
        .add_tokens([AddedToken::from("same", false)])
        .unwrap();
    tokenizer.with_model(model([
        ("same".into(), 99),
        ("hidden".into(), 2),
        ("plain".into(), 10),
    ]));
    let snapshot = tokenizer.snapshot();
    let plan = DecodeCompilePlan::prepare(&snapshot).unwrap();
    let required = plan.requirements();
    assert_eq!(required.id_slots(), 4);
    assert_eq!(required.piece_bytes(), 17); // same, same, plain, same
    assert_eq!(
        required.buffer_bytes(),
        4 * (size_of::<u32>() + size_of::<Record>()) + 17
    );
    let source = plan.compile().unwrap();
    assert_eq!(source.records.capacity(), 3);
    assert_eq!(source.bytes.capacity(), 13);
    assert_eq!(
        source.storage_bytes(),
        size_of::<PreparedDecodeSource>() + 3 * size_of::<Record>() + 13
    );
    assert!(source.storage_bytes() < required.required_bytes());
    assert_eq!(
        source.records.iter().map(|row| row.id).collect::<Vec<_>>(),
        [2, 10, 99]
    );
    differential(&snapshot, &source, &[2, 99, 10, 999, 2], false);
}

#[test]
fn compiled_nonzero_sources_match_hf_across_all_concrete_model_maps() {
    let vocab: std::collections::HashMap<String, u32> = [
        ("[UNK]".into(), 0),
        ("hi".into(), 7),
        ("é".into(), 55),
        ("🦀".into(), 900),
    ]
    .into_iter()
    .collect();
    // Construct through the published JSON model formats to avoid fixture-only
    // access to each model's concrete hash-map implementation.
    let bpe: BPE = serde_json::from_str(
        &serde_json::json!({"type":"BPE","vocab":vocab,"merges":[]}).to_string(),
    )
    .unwrap();
    let wordpiece: WordPiece = serde_json::from_str(&serde_json::json!({"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":vocab}).to_string()).unwrap();
    let unigram = Unigram::from(
        vec![
            ("[UNK]".into(), 0.0),
            ("hi".into(), -1.0),
            ("é".into(), -2.0),
            ("🦀".into(), -3.0),
            ("hi".into(), -4.0),
        ],
        Some(0),
        false,
    )
    .unwrap();
    let models: [tokenizers::models::ModelWrapper; 3] =
        [bpe.into(), wordpiece.into(), unigram.into()];
    for model in models {
        let tokenizer = Tokenizer::from_tokenizer(tokenizers::Tokenizer::new(model));
        let snapshot = tokenizer.snapshot();
        let plan = DecodeCompilePlan::prepare(&snapshot).unwrap();
        let mut expected: Vec<_> = snapshot.get_vocab(false).into_values().collect();
        expected.sort_unstable();
        expected.dedup();
        assert_eq!(plan.requirements().id_slots(), expected.len());
        let source = plan.compile().unwrap();
        assert_eq!(
            source
                .records
                .iter()
                .map(|record| record.id)
                .collect::<Vec<_>>(),
            expected
        );
        let tokens = ids(&snapshot, &["hi", "é", "🦀", "hi"]);
        differential(&snapshot, &source, &tokens, false);
    }
}

#[test]
fn checked_compiler_preserves_every_pipeline_frontier_and_actual_invalid_prefix() {
    let mut invalid_prefixes = 0;
    for order in ['A', 'B', 'C'] {
        for strip in [false, true] {
            let tokenizer = pipelines::with_pipeline(pipelines::pipeline(order, strip));
            let snapshot = tokenizer.snapshot();
            let plan = DecodeCompilePlan::prepare(&snapshot).unwrap();
            let source = plan.compile().unwrap();
            for skip in [false, true] {
                for tokens in [
                    vec![0x61, 0xff, 302, 301],
                    vec![303, 301, 0xc3, 0xa9, 305, 308, 302],
                    vec![0xe2, 0x96, 0x81, 301, 0xff, 302],
                    vec![304, 305, 304, 301, 0xc3, 0xa9],
                    vec![300, 999, 309, 310, 311, 312, 313, 314],
                ] {
                    invalid_prefixes += pipelines::trace_source(&snapshot, &source, &tokens, skip)
                        .iter()
                        .filter(|result| result.is_err())
                        .count();
                }
            }
        }
    }
    assert!(
        invalid_prefixes > 0,
        "the fixture must actually reach HF InvalidPrefix"
    );
    let tokenizer = byte_wrapper();
    let snapshot = tokenizer.snapshot();
    let source = DecodeCompilePlan::prepare(&snapshot)
        .unwrap()
        .compile()
        .unwrap();
    for skip in [false, true] {
        differential(
            &snapshot,
            &source,
            &ids(&snapshot, &["h", "Ġ", "Ã", "©", "ð", "Ł", "¦", "Ģ"]),
            skip,
        );
    }
}

#[test]
fn checked_compiler_uses_normalized_spelling_for_special_membership() {
    let mut tokenizer = wrapper(&[("[UNK]", 0), ("A", 1)], true);
    tokenizer
        .with_normalizer(Some(tokenizers::normalizers::Lowercase))
        .unwrap();
    tokenizer
        .add_tokens([AddedToken::from(" Hi", false).normalized(true)])
        .unwrap();
    tokenizer
        .add_special_tokens([
            AddedToken::from("<LOUD>", true).normalized(true),
            AddedToken::from("<stop>", true),
        ])
        .unwrap();
    let snapshot = tokenizer.snapshot();
    let source = DecodeCompilePlan::prepare(&snapshot)
        .unwrap()
        .compile()
        .unwrap();
    let loud = snapshot.token_to_id("<LOUD>").unwrap();
    assert_eq!(source.piece(loud, true), Some(&b"<loud>"[..]));
    let tokens = ids(&snapshot, &[" Hi", "<LOUD>", "A", "<stop>", " Hi"]);
    for skip in [false, true] {
        differential(&snapshot, &source, &tokens, skip);
    }
}

#[test]
fn partial_allocation_errors_retain_every_successful_reservation_frontier() {
    let tokenizer = wrapper(&[("first", 2), ("é", 500), ("🦀", 900)], false);
    let snapshot = tokenizer.snapshot();
    let mut errors = Vec::new();
    for stage in 0..3 {
        let plan = DecodeCompilePlan::prepare(&snapshot).unwrap();
        let required = plan.requirements();
        let error = plan.fail_reservation(stage).compile().unwrap_err();
        let expected = match stage {
            0 => 0,
            1 => 3 * size_of::<u32>(),
            2 => 3 * (size_of::<u32>() + size_of::<Record>()),
            _ => unreachable!(),
        };
        assert_eq!(error.retained_buffer_bytes(), expected);
        assert!(error.retained_buffer_bytes() <= required.buffer_bytes());
        let DecodeSourceError::Allocation(cause) = error.cause() else {
            panic!("actual reserve error lost");
        };
        assert!(std::ptr::eq(
            error
                .source()
                .unwrap()
                .downcast_ref::<DecodeSourceError>()
                .unwrap(),
            error.cause()
        ));
        assert!(std::ptr::eq(
            error
                .cause()
                .source()
                .unwrap()
                .downcast_ref::<std::collections::TryReserveError>()
                .unwrap(),
            cause
        ));
        errors.push(error);
    }
    // Errors own only their original buffers; the immutable HF borrow ends at
    // compile return. Keeping all failures does not retain or need the snapshot.
    drop(snapshot);
    drop(tokenizer);
    assert_eq!(
        errors
            .iter()
            .map(DecodeCompileFailure::retained_buffer_bytes)
            .sum::<usize>(),
        6 * size_of::<u32>() + 3 * size_of::<Record>()
    );
}

#[test]
fn planning_rejects_unsupported_decoders_and_all_host_extent_overflows() {
    for decoder in [
        serde_json::json!({"type":"WordPiece","prefix":"##","cleanup":true}),
        serde_json::json!({"type":"Sequence","decoders":[{"type":"ByteFallback"},{"type":"Fuse"},{"type":"Replace","pattern":{"Regex":"▁"},"content":" "}]}),
    ] {
        let mut tokenizer = wrapper(&[("hello", 0)], false);
        tokenizer.with_decoder(Some(
            serde_json::from_value::<DecoderWrapper>(decoder).unwrap(),
        ));
        let snapshot = tokenizer.snapshot();
        assert!(matches!(
            DecodeCompilePlan::prepare(&snapshot),
            Err(DecodeSourceError::UnsupportedDecoder)
        ));
        assert!(matches!(
            PreparedDecodeSource::prepare(&snapshot),
            Err(DecodeSourceError::UnsupportedDecoder)
        ));
    }
    for (ids, bytes) in [
        (usize::MAX, 0),
        (0, usize::MAX),
        ((isize::MAX as usize / size_of::<Record>()) + 1, 0),
        (
            isize::MAX as usize / size_of::<Record>(),
            isize::MAX as usize,
        ),
    ] {
        assert!(matches!(
            super::super::compiler::overflow_requirements(ids, bytes),
            Err(DecodeSourceError::Overflow)
        ));
    }
}

#[test]
fn successful_program_is_independent_of_snapshot_and_preserves_fixed_storage() {
    let mut tokenizer = wrapper(&[("hi", 0), ("é", 1), ("🦀", 2)], false);
    let snapshot = tokenizer.snapshot();
    let plan = DecodeCompilePlan::prepare(&snapshot).unwrap();
    tokenizer.with_model(model([("changed".into(), 0)]));
    let source = plan.compile().unwrap();
    assert_eq!(tokenizer.decode(&[0], false).unwrap(), "changed");
    let expected = snapshot.decode(&[0, 1, 2], false).unwrap();
    let capacities = (source.records.capacity(), source.bytes.capacity());
    let addresses = (source.records.as_ptr(), source.bytes.as_ptr());
    drop(snapshot);
    drop(tokenizer);
    let layout = DecodeStreamLayout::for_source(&source, 3, false).unwrap();
    let mut buffers = Buffers::new(&layout);
    let mut stream = buffers.stream(layout);
    let mut text = String::new();
    for id in [0, 1, 2] {
        if let Some(piece) = stream.step(id).unwrap() {
            text.push_str(piece);
        }
    }
    assert_eq!(text, expected);
    assert_eq!(stream.finish(), Ok(()));
    assert_eq!(
        (source.records.capacity(), source.bytes.capacity()),
        capacities
    );
    assert_eq!((source.records.as_ptr(), source.bytes.as_ptr()), addresses);
}

#[test]
fn empty_and_zero_byte_programs_need_no_byte_destination() {
    for entries in [&[][..], &[("", 91)][..]] {
        let tokenizer = wrapper(entries, false);
        let snapshot = tokenizer.snapshot();
        let plan = DecodeCompilePlan::prepare(&snapshot).unwrap();
        assert_eq!(plan.requirements().piece_bytes(), 0);
        let source = plan.compile().unwrap();
        assert_eq!(source.bytes.capacity(), 0);
        assert_eq!(source.token_count(), entries.len());
        assert_eq!(source.records.capacity(), entries.len());
        differential(&snapshot, &source, &[91, u32::MAX, 91], false);
    }
}
