use super::*;
use std::error::Error;

fn tag(flags: ExprFlags, tag: ExprTag) -> u32 {
    flags.0 | tag as u32
}

fn check_sequence(cases: &[(Expr<'_>, Vec<u32>)]) {
    let words = cases.iter().map(|(_, words)| words.len()).sum();
    let scratch = cases
        .iter()
        .map(|(_, words)| words.len())
        .max()
        .unwrap_or(0);
    let mut prepared =
        PreparedVecHashCons::try_new_with_scratch(words, cases.len(), scratch).unwrap();
    let capacity = prepared.retained_capacity_bytes().unwrap();
    let mut growing = PreparedVecHashCons::empty_with_funding(crate::ParserAllocationFunding::unenforced()).unwrap();
    for (expression, expected) in cases {
        assert_eq!(expression.encoded_word_len().unwrap(), expected.len());
        let growing_id = expression.try_intern_encoded(&mut growing).unwrap();
        let prepared_id = expression.try_intern_encoded(&mut prepared).unwrap();
        assert_eq!(prepared_id, growing_id);
        assert_eq!(growing.get(growing_id), expected);
        assert_eq!(prepared.get(prepared_id), expected);
        assert_eq!(prepared.retained_capacity_bytes().unwrap(), capacity);
    }
    let count = prepared.len();
    for (expression, expected) in cases.iter().rev() {
        let prepared_id = expression.try_intern_encoded(&mut prepared).unwrap();
        assert_eq!(prepared_id, expression.try_intern_encoded(&mut growing).unwrap());
        assert_eq!(prepared.get(prepared_id), expected);
        assert_eq!(prepared.len(), count);
        assert_eq!(prepared.retained_capacity_bytes().unwrap(), capacity);
    }
}

#[test]
fn every_expression_variant_preserves_words_ids_and_flags() {
    let args = [ExprRef::new(17), ExprRef::new(23), ExprRef::new(41)];
    let byte_set = [0x1, 0x8000_0000, 0x1234_5678, 0xffff_ffff, 0, 3, 5, 9];
    let positive = ExprFlags::POSITIVE;
    let nullable = ExprFlags::POSITIVE_NULLABLE;
    let zero = ExprFlags::ZERO;
    let mut cases = vec![
        (Expr::EmptyString, vec![tag(nullable, ExprTag::EmptyString)]),
        (Expr::NoMatch, vec![tag(zero, ExprTag::NoMatch)]),
        (
            Expr::Byte(0xa7),
            vec![tag(positive, ExprTag::Byte), 0xa7u32.to_le()],
        ),
        (
            Expr::RemainderIs {
                divisor: 17,
                remainder: 0,
                scale: 3,
                fractional_part: false,
            },
            vec![tag(nullable, ExprTag::RemainderIs), 17, 0, 3, 0],
        ),
        (
            Expr::RemainderIs {
                divisor: 23,
                remainder: 5,
                scale: 2,
                fractional_part: true,
            },
            vec![tag(positive, ExprTag::RemainderIs), 23, 5, 2, 1],
        ),
        (
            Expr::Lookahead(nullable, args[0], 7),
            vec![tag(nullable, ExprTag::Lookahead), 17, 7],
        ),
        (Expr::Not(zero, args[1]), vec![tag(zero, ExprTag::Not), 23]),
        (
            Expr::Repeat(positive, args[0], 2, u32::MAX),
            vec![tag(positive, ExprTag::Repeat), 17, 2, u32::MAX],
        ),
        (
            Expr::Concat(nullable, [args[0], args[1]]),
            vec![tag(nullable, ExprTag::Concat), 17, 23],
        ),
    ];
    for length in [0, 1, 3, 8] {
        let mut words = vec![tag(positive, ExprTag::ByteSet)];
        words.extend_from_slice(&byte_set[..length]);
        cases.push((Expr::ByteSet(&byte_set[..length]), words));
    }
    for length in [0, 1, 3] {
        let mut or = vec![tag(nullable, ExprTag::Or)];
        or.extend(args[..length].iter().map(|arg| arg.0));
        cases.push((Expr::Or(nullable, &args[..length]), or));
        let mut and = vec![tag(positive, ExprTag::And)];
        and.extend(args[..length].iter().map(|arg| arg.0));
        cases.push((Expr::And(positive, &args[..length]), and));
    }
    check_sequence(&cases);
}

#[test]
fn byte_concat_boundary_lengths_preserve_byte_order_and_zero_padding() {
    let bytes: Vec<_> = (0..31).map(|index| 0x80 + index as u8).collect();
    let mut cases = Vec::new();
    for length in [0usize, 1, 3, 4, 30, 31] {
        let mut payload = vec![0; (length + 1).div_ceil(4) * 4];
        payload[0] = length as u8;
        payload[1..length + 1].copy_from_slice(&bytes[..length]);
        let mut words = vec![tag(ExprFlags::NULLABLE, ExprTag::ByteConcat), 23];
        words.extend(payload.chunks_exact(4).map(|part| {
            // The old representation writes byte payloads directly into the
            // word buffer, preserving the same memory bytes on every endian.
            u32::from_ne_bytes(part.try_into().unwrap())
        }));
        assert_eq!(words.len(), 3 + length / 4);
        let expression = Expr::ByteConcat(ExprFlags::NULLABLE, &bytes[..length], ExprRef::new(23));
        cases.push((expression, words));
    }
    check_sequence(&cases);
    for (_, words) in &cases {
        let raw: &[u8] = bytemuck::cast_slice(&words[2..]);
        let length = raw[0] as usize;
        assert_eq!(&raw[1..length + 1], &bytes[..length]);
        assert!(raw[length + 1..].iter().all(|byte| *byte == 0));
        match Expr::from_slice(words) {
            Expr::ByteConcat(flags, decoded, tail) => {
                assert_eq!(flags.0, ExprFlags::NULLABLE.0);
                assert_eq!(decoded, &bytes[..length]);
                assert_eq!(tail, ExprRef::new(23));
            }
            _ => panic!("wrong decoded expression variant"),
        }
    }
}

#[test]
fn exact_capacity_and_duplicates_do_not_grow_prepared_storage() {
    let args = [ExprRef::new(17), ExprRef::new(23), ExprRef::new(41)];
    let expression = Expr::Or(ExprFlags::POSITIVE, &args);
    let words = expression.encoded_word_len().unwrap();
    let mut table = PreparedVecHashCons::try_new_with_scratch(words, 1, words).unwrap();
    let capacity = table.retained_capacity_bytes().unwrap();
    let id = expression.try_intern_encoded(&mut table).unwrap();
    assert_eq!(id, 0);
    assert_eq!(table.len(), 1);
    assert_eq!(expression.try_intern_encoded(&mut table).unwrap(), id);
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
    assert_eq!(
        table.get(id),
        &[tag(ExprFlags::POSITIVE, ExprTag::Or), 17, 23, 41]
    );
}

#[test]
fn one_word_short_retains_typed_storage_error_and_allows_later_smaller_insert() {
    let args = [ExprRef::new(17), ExprRef::new(23), ExprRef::new(41)];
    let expression = Expr::And(ExprFlags::POSITIVE, &args);
    let words = expression.encoded_word_len().unwrap();
    let mut table = PreparedVecHashCons::try_new_with_scratch(words - 1, 1, words).unwrap();
    let capacity = table.retained_capacity_bytes().unwrap();
    let error = expression.try_intern_encoded(&mut table).unwrap_err();
    assert!(matches!(error,
        ExprEncodingError::Storage(HashConsCapacityError::WordsExceeded {
            required_words,
            capacity_words,
        }) if required_words == words && capacity_words == words - 1
    ));
    assert!(error
        .source()
        .unwrap()
        .downcast_ref::<HashConsCapacityError>()
        .is_some());
    assert_eq!(table.len(), 0);
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
    assert_eq!(Expr::Byte(0xe3).try_intern_encoded(&mut table).unwrap(), 0);
    assert_eq!(
        table.get(0),
        &[tag(ExprFlags::POSITIVE, ExprTag::Byte), 0xe3u32.to_le()]
    );
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
}

#[test]
fn one_scratch_word_short_rejects_before_publication_and_reuses_scratch() {
    let args = [ExprRef::new(17), ExprRef::new(23), ExprRef::new(41)];
    let expression = Expr::Or(ExprFlags::POSITIVE_NULLABLE, &args);
    let words = expression.encoded_word_len().unwrap();
    let mut table = PreparedVecHashCons::try_new_with_scratch(words, 1, words - 1).unwrap();
    let capacity = table.retained_capacity_bytes().unwrap();
    let error = expression.try_intern_encoded(&mut table).unwrap_err();
    assert!(matches!(error,
        ExprEncodingError::Storage(HashConsCapacityError::ScratchExceeded {
            required_words,
            capacity_words,
        }) if required_words == words && capacity_words == words - 1
    ));
    assert!(error
        .source()
        .unwrap()
        .downcast_ref::<HashConsCapacityError>()
        .is_some());
    assert_eq!(table.len(), 0);
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
    assert_eq!(Expr::Byte(0xf1).try_intern_encoded(&mut table).unwrap(), 0);
    assert_eq!(
        table.get(0),
        &[tag(ExprFlags::POSITIVE, ExprTag::Byte), 0xf1u32.to_le()]
    );
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
}

#[test]
fn entry_capacity_failure_preserves_existing_encoding_and_duplicate_hits() {
    let mut table = PreparedVecHashCons::try_new_with_scratch(8, 1, 5).unwrap();
    let id = Expr::EmptyString.try_intern_encoded(&mut table).unwrap();
    let capacity = table.retained_capacity_bytes().unwrap();
    let error = Expr::RemainderIs {
        divisor: 17,
        remainder: 3,
        scale: 2,
        fractional_part: true,
    }
    .try_intern_encoded(&mut table)
    .unwrap_err();
    assert!(matches!(
        error,
        ExprEncodingError::Storage(HashConsCapacityError::EntriesExceeded {
            required_entries: 2,
            capacity_entries: 1,
        })
    ));
    assert_eq!(table.len(), 1);
    assert_eq!(
        table.get(id),
        &[tag(ExprFlags::POSITIVE_NULLABLE, ExprTag::EmptyString)]
    );
    assert_eq!(
        Expr::EmptyString.try_intern_encoded(&mut table).unwrap(),
        id
    );
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
}

#[test]
fn invalid_byte_concat_is_rejected_before_prepared_insertion() {
    let bytes = [0xa7; 32];
    let expression = Expr::ByteConcat(ExprFlags::POSITIVE, &bytes, ExprRef::EMPTY_STRING);
    let mut table = PreparedVecHashCons::try_new_with_scratch(2, 1, 2).unwrap();
    let capacity = table.retained_capacity_bytes().unwrap();
    for error in [
        expression.encoded_word_len().unwrap_err(),
        expression.try_intern_encoded(&mut table).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            ExprEncodingError::InvalidByteConcatLength {
                length: 32,
                maximum: 31
            }
        ));
        assert!(error.source().is_none());
    }
    assert_eq!(table.len(), 0);
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
    assert_eq!(Expr::Byte(0x81).try_intern_encoded(&mut table).unwrap(), 0);
    assert_eq!(table.retained_capacity_bytes().unwrap(), capacity);
}

#[test]
fn encoded_word_geometry_checks_overflow_without_allocating_a_large_slice() {
    let mut count = EncodedWordCount(3);
    assert!(matches!(
        count.add(usize::MAX),
        Err(ExprEncodingError::WordCountOverflow)
    ));
    assert_eq!(count.0, 3);
    let maximum = (u32::MAX as usize).min(isize::MAX as usize / std::mem::size_of::<u32>());
    count.0 = maximum;
    assert!(matches!(
        count.add(1),
        Err(ExprEncodingError::WordCountOverflow)
    ));
    assert_eq!(count.0, maximum);
    count.0 = 0;
    count.add(maximum).unwrap();
    assert_eq!(count.0, maximum);
}

#[test]
fn borrowed_original_expressions_encode_without_rebasing_children_or_faking_expr_ids() {
    let mut source = ExprSet::new(256, crate::ParserAllocationFunding::unenforced()).unwrap();
    let literal = source.mk_literal("marigold").unwrap();
    let expression = source.get(literal);
    let words = expression.encoded_word_len().unwrap();
    let before = source.exprs.retained_capacity_bytes().unwrap();
    let mut raw = PreparedVecHashCons::try_new_with_scratch(words + 1, 2, words).unwrap();
    // Unlike ExprSet's reserved seed layout, an empty raw table assigns id 0
    // to the first real expression. The API exposes that fact as a raw u32.
    let empty = source
        .get(ExprRef::EMPTY_STRING)
        .try_intern_encoded(&mut raw)
        .unwrap();
    assert_eq!(empty, 0);
    assert_ne!(empty, ExprRef::EMPTY_STRING.as_u32());
    let raw_id = expression.try_intern_encoded(&mut raw).unwrap();
    assert_eq!(raw.get(raw_id), source.exprs.get(literal.as_u32()));
    assert_eq!(source.exprs.retained_capacity_bytes().unwrap(), before);
    assert_eq!(source.mk_literal("marigold").unwrap(), literal);
}
