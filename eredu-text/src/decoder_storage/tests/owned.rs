use super::*;

fn compare(snapshot: &TokenizerSnapshot, tokens: &[u32], skip: bool) {
    let source = PreparedDecodeSource::prepare(snapshot).unwrap();
    let layout = DecodeStreamLayout::for_source(&source, tokens.len(), skip).unwrap();
    let mut buffers = Buffers::new(&layout);
    let mut borrowed = buffers.stream(layout);
    let mut owned = OwnedDecodeStorage::new(
        PreparedDecodeSource::prepare(snapshot).unwrap(),
        tokens.len(),
        skip,
    )
    .unwrap();
    let source_address = std::ptr::from_ref(owned.source());
    assert!(owned.retained_ids().is_empty());
    if !tokens.is_empty() {
        assert_eq!(
            owned.step(tokens[0]),
            Err(OwnedDecodeStorageError::Unprepared)
        );
        assert_eq!(owned.successful_calls(), 0);
    }
    owned.prepare_destinations().unwrap();
    // Move the complete source/storage owner, preserving physical source identity.
    let mut owned = Box::new(owned);
    assert_eq!(source_address, std::ptr::from_ref(owned.source()));
    let (mut ids, mut prefix, mut index, mut successful) = (vec![], String::new(), 0, 0);
    for &id in tokens {
        let hf = tokenizers::tokenizer::step_decode_stream(
            snapshot,
            vec![id],
            skip,
            &mut ids,
            &mut prefix,
            &mut index,
        );
        let actual = owned.step(id).map(|s| s.map(str::to_owned));
        let reference = borrowed.step(id).map(|s| s.map(str::to_owned));
        assert_eq!(actual, reference.map_err(OwnedDecodeStorageError::Storage));
        match hf {
            Ok(text) => {
                assert_eq!(actual, Ok(text));
                successful += 1;
            }
            Err(error) => {
                let tokenizers::tokenizer::DecodeStreamError::InvalidPrefix {
                    token_id,
                    expected_prefix,
                    actual_string,
                } = error.downcast_ref().expect("actual HF stream diagnostic");
                assert_eq!(
                    actual,
                    Err(OwnedDecodeStorageError::Storage(
                        DecodeStorageError::InvalidPrefix {
                            token_id: *token_id,
                            expected_bytes: expected_prefix.len(),
                            actual_bytes: actual_string.len(),
                        }
                    ))
                );
            }
        }
        assert_eq!(owned.retained_ids(), ids);
        assert_eq!(owned.prefix(), prefix);
        assert_eq!(owned.prefix_index(), index);
        assert_eq!(owned.successful_calls(), successful);
        assert_eq!(owned.candidate(), borrowed.candidate());
        assert_eq!(owned.retained_ids(), borrowed.retained_ids());
    }
    assert_eq!(
        owned.finish(),
        borrowed.finish().map_err(OwnedDecodeStorageError::Storage)
    );
    assert_eq!(owned.candidate(), snapshot.decode(&ids, skip).unwrap());
    assert_eq!(owned.prefix(), prefix);
    assert_eq!(source_address, std::ptr::from_ref(owned.source()));
}

#[test]
fn owning_and_borrowed_streams_share_complete_hf_frontiers_across_source_modes() {
    let plain = wrapper(&[("", 0), ("hello", 1), ("é", 2), ("🦀", 3)], false);
    compare(&plain.snapshot(), &[0, 1, 99, 2, 3, 0, 1], false);
    let bytes = wrapper(&[("â", 0), ("Ĥ", 1), ("¬", 2), ("Ġ", 3), ("x", 4)], true);
    compare(&bytes.snapshot(), &[0, 1, 2, 3, 4, 0], false);
    for order in ['A', 'B', 'C'] {
        for strip in [false, true] {
            let tokenizer =
                super::pipelines::with_pipeline(super::pipelines::pipeline(order, strip));
            for skip in [false, true] {
                compare(
                    &tokenizer.snapshot(),
                    &[303, 0xe2, 0x96, 0x81, 308, 301, 999, 0xff, 300, 302],
                    skip,
                );
                compare(&tokenizer.snapshot(), &[0x61, 0xff, 302, 301], skip);
                // This real transition family includes shrinking/invalid prefixes.
                compare(
                    &tokenizer.snapshot(),
                    &[300, 303, 303, 301, 300, 0xff, 0x61, 304],
                    skip,
                );
            }
        }
    }
}

#[test]
fn dormant_zero_limit_and_prepared_exhaustion_preserve_source_and_state() {
    let tokenizer = wrapper(&[("hello", 0)], false);
    let mut empty = OwnedDecodeStorage::new(
        PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap(),
        0,
        false,
    )
    .unwrap();
    assert_eq!(empty.destination_bytes(), 0);
    assert_eq!(empty.finish(), Ok(()));
    assert_eq!(
        empty.step(0),
        Err(OwnedDecodeStorageError::Storage(
            DecodeStorageError::CallLimit
        ))
    );
    let mut one = OwnedDecodeStorage::new(
        PreparedDecodeSource::prepare(&tokenizer.snapshot()).unwrap(),
        1,
        false,
    )
    .unwrap();
    one.prepare_destinations().unwrap();
    assert_eq!(one.step(0), Ok(Some("hello")));
    let before = (
        one.retained_ids().to_vec(),
        one.prefix().to_owned(),
        one.candidate().to_owned(),
    );
    one.prepare_destinations().unwrap(); // ready does not allocate/reset history
    assert_eq!(
        one.step(0),
        Err(OwnedDecodeStorageError::Storage(
            DecodeStorageError::CallLimit
        ))
    );
    assert_eq!(
        (
            one.retained_ids().to_vec(),
            one.prefix().to_owned(),
            one.candidate().to_owned()
        ),
        before
    );
}
