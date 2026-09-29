use super::*;
fn lookup() -> eredu_runtime::RowLookupSpec {
    eredu_runtime::RowLookupSpec {
        parameter: eredu_nn::ParameterId::new("table.weight").unwrap(),
        bank: 1,
        unit: 3,
        rows: 24,
        dimensions: 2,
        encoding: eredu_runtime::RowEncoding::Dense,
        output_type: eredu_nn::TensorElementType::F32,
    }
}
#[test]
fn header_embedding_geometry_is_available_before_exact_hash_binding() {
    let specification = NGramEmbeddingSpec::new(1007, 7, 3, 1, lookup(), 5, 16).unwrap();
    assert_eq!(specification.output_width(), 4);
    assert_eq!(specification.order(), 3);
    assert_eq!(specification.padding_token(), 7);
    assert_eq!(specification.max_tokens(), 8);
    assert_eq!(specification.lookup_spec(), &lookup());
    let policy = specification.history_policy();
    let exact = NGramHashSpec::new(
        1007,
        7,
        3,
        1,
        vec![9_007_199_254_740_993, i64::MAX, i64::MIN],
        vec![11, 13],
        vec![0, 11],
        24,
    )
    .unwrap();
    for mismatch in [
        NGramHashSpec {
            vocabulary: 1008,
            ..exact.clone()
        },
        NGramHashSpec {
            eos: 8,
            ..exact.clone()
        },
        NGramHashSpec::new(1007, 7, 2, 1, vec![3, 5], vec![24], vec![0], 24).unwrap(),
        NGramHashSpec::new(
            1007,
            7,
            3,
            2,
            vec![3, 5, 7],
            vec![5, 5, 5, 9],
            vec![0, 5, 10, 15],
            24,
        )
        .unwrap(),
        NGramHashSpec {
            table_rows: 25,
            ..exact.clone()
        },
    ] {
        assert!(matches!(
            NGramEmbedding::new(specification.clone(), mismatch),
            Err(NGramEmbeddingError::Hash(NGramError::Constants))
        ));
    }
    let embedding = NGramEmbedding::new(specification, exact.clone()).unwrap();
    assert_eq!(embedding.hash, exact);
    assert_eq!(embedding.specification().history_policy(), policy);
}
#[test]
fn header_embedding_rejects_unusable_or_overflowing_geometry_without_literals() {
    for (vocabulary, eos, order, heads, max_rows) in [
        (0, 0, 3, 1, 16),
        (i32::MAX as u64 + 1, 7, 3, 1, 16),
        (1007, 1007, 3, 1, 16),
        (1007, 7, 1, 1, 16),
        (1007, 7, 3, 0, 16),
        (1007, 7, i32::MAX as usize + 1, 1, 16),
        (1007, 7, 3, usize::MAX, 16),
        (1007, 7, 3, 1, 1),
        (1007, 7, 3, 1, i32::MAX as usize + 1),
    ] {
        assert!(
            NGramEmbeddingSpec::new(vocabulary, eos, order, heads, lookup(), 5, max_rows).is_err()
        );
    }
    for table in [
        eredu_runtime::RowLookupSpec {
            rows: 0,
            ..lookup()
        },
        eredu_runtime::RowLookupSpec {
            rows: i64::MAX as u64 + 1,
            ..lookup()
        },
        eredu_runtime::RowLookupSpec {
            dimensions: 0,
            ..lookup()
        },
        eredu_runtime::RowLookupSpec {
            dimensions: i32::MAX,
            ..lookup()
        },
        eredu_runtime::RowLookupSpec {
            output_type: eredu_nn::TensorElementType::I64,
            ..lookup()
        },
    ] {
        assert!(NGramEmbeddingSpec::new(1007, 7, 3, 1, table, 5, 16).is_err());
    }
}
fn spec() -> NGramHashSpec {
    NGramHashSpec::new(
        1 << 32,
        7,
        4,
        2,
        vec![
            9223372036854775783,
            9007199254740993,
            -9223372036854775765,
            0x123456789abcdef,
        ],
        vec![101, 103, 107, 109, 113, 127],
        vec![0, 101, 204, 311, 420, 533],
        660,
    )
    .unwrap()
}
const TOKENS: [u64; 16] = [16777217, 3, 7, 4, 5, 7, 6, 1, 2, 7, 7, 16777219, 7, 4, 3, 2];
// Independent Python integer implementation of the pinned reference's shifted
// token windows, explicit signed-int64 wrapping and positive remainders.
const EXPECTED: [u64; 96] = [
    90, 140, 302, 354, 454, 547, 17, 142, 230, 398, 431, 591, 41, 200, 213, 368, 524, 541, 56, 178,
    248, 399, 529, 600, 1, 146, 294, 407, 528, 650, 59, 194, 247, 393, 464, 616, 10, 132, 237, 390,
    441, 611, 6, 125, 223, 333, 421, 589, 1, 121, 294, 336, 462, 646, 34, 102, 248, 362, 429, 576,
    77, 188, 250, 416, 467, 651, 44, 197, 256, 417, 521, 628, 36, 139, 206, 373, 520, 599, 56, 178,
    248, 399, 529, 600, 43, 188, 280, 393, 514, 636, 74, 141, 262, 317, 483, 575,
];
#[test]
fn exact_hashes_include_eos_boundaries_large_constants_and_signed_overflow() {
    let spec = spec();
    let initial = spec.initial_history(2).unwrap();
    let result = spec.select(Some(&TOKENS), 2, 8, &initial, 96).unwrap();
    assert_eq!(result.rows, EXPECTED);
    assert_eq!(result.next_history, [7, 6, 1, 4, 3, 2]);
    // EOS is included in the current n-gram. Subsequent inputs resume with EOS
    // padding; semantic termination belongs to the generation facade.
    assert_ne!(&result.rows[12..18], &result.rows[30..36]);
    assert_eq!(&result.rows[18..24], &result.rows[78..84]);
}
#[test]
fn arbitrary_prefill_chunks_decode_and_restored_histories_match_reference() {
    let spec = spec();
    for chunks in [vec![8], vec![1; 8], vec![1, 2, 1, 3, 1], vec![3, 5]] {
        let mut history = spec.initial_history(2).unwrap();
        let mut position = 0;
        let mut rows = [Vec::new(), Vec::new()];
        for n in chunks {
            let input = (0..2)
                .flat_map(|batch| {
                    TOKENS[batch * 8 + position..batch * 8 + position + n]
                        .iter()
                        .copied()
                })
                .collect::<Vec<_>>();
            let saved = history.clone();
            let result = spec.select(Some(&input), 2, n, &history, 96).unwrap();
            let replay = spec.select(Some(&input), 2, n, &saved, 96).unwrap();
            assert_eq!(result, replay);
            for b in 0..2 {
                rows[b].extend_from_slice(&result.rows[b * n * 6..(b + 1) * n * 6]);
            }
            // A rejected speculative proposal is a separate history value.
            let rejected = spec.select(Some(&[99, 100]), 2, 1, &saved, 12).unwrap();
            assert_ne!(rejected.next_history, saved);
            assert_eq!(history, saved);
            history = result.next_history;
            position += n;
        }
        assert_eq!(rows.concat(), EXPECTED);
        assert_eq!(history, [7, 6, 1, 4, 3, 2]);
    }
}
#[test]
fn missing_ids_bad_companions_and_lookup_bounds_are_typed() {
    let spec = spec();
    let history = spec.initial_history(1).unwrap();
    assert_eq!(
        spec.select(None, 1, 1, &history, 6),
        Err(NGramError::MissingTokenIds)
    );
    assert_eq!(
        spec.select(Some(&[1]), 1, 1, &history, 5),
        Err(NGramError::Budget {
            required: 6,
            limit: 5
        })
    );
    assert_eq!(
        spec.select(Some(&[1 << 32]), 1, 1, &history, 6),
        Err(NGramError::Token(1 << 32))
    );
    assert_eq!(
        spec.select(Some(&[1]), 2, 1, &history, 6),
        Err(NGramError::Shape)
    );
    assert!(NGramHashSpec::new(10, 7, 3, 1, vec![1, 2], vec![3, 5], vec![0, 3], 8).is_err());
    assert!(NGramHashSpec::new(10, 7, 3, 1, vec![1, 2, 3], vec![0, 5], vec![0, 3], 8).is_err());
    assert!(NGramHashSpec::new(10, 7, 3, 1, vec![1, 2, 3], vec![3, 5], vec![0, 4], 8).is_err());
    assert_eq!(history, [7, 7, 7]);
}
#[test]
fn splitmix_initialization_matches_pinned_integer_reference() {
    assert_eq!(
        layer_multipliers(248320, 3, 2, 0).unwrap(),
        [19168946543173, 7589933366025, 6060819858713]
    );
    assert!(layer_multipliers(0, 3, 0, 0).is_err());
}
