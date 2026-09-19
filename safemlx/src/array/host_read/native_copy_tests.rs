use super::*;
use crate::{
    error::NativeBytesCopyError,
    ops::{broadcast_to, indexing::TryIndexOp},
    utils::allocation_test::measure,
};

fn check(array: &Array, expected: &[u8]) {
    let evaluated = array.evaluated().unwrap();
    let mut guarded = vec![0xa5; expected.len() + 2];
    let (result, allocations) =
        measure(|| evaluated.try_copy_native_bytes_into(&mut guarded[1..expected.len() + 1]));
    assert_eq!(result, Ok(()));
    assert_eq!(allocations, 0);
    assert_eq!(&guarded[1..expected.len() + 1], expected);
    assert_eq!(guarded[0], 0xa5);
    assert_eq!(guarded[expected.len() + 1], 0xa5);

    for length in [expected.len().checked_sub(1), Some(expected.len() + 1)]
        .into_iter()
        .flatten()
    {
        let mut wrong = vec![0x5a; length];
        let (result, allocations) = measure(|| evaluated.try_copy_native_bytes_into(&mut wrong));
        assert_eq!(
            result,
            Err(NativeBytesCopyError::DestinationLength {
                expected: expected.len(),
                found: length,
            })
        );
        assert_eq!(allocations, 0);
        assert!(wrong.iter().all(|&byte| byte == 0x5a));
    }

    // The allocating API supplies a positive instrumentation control and keeps
    // the same independently encoded representation as the caller-owned copy.
    let (owned, allocations) = measure(|| evaluated.to_native_bytes());
    assert_eq!(owned, expected);
    assert_eq!(allocations, usize::from(!expected.is_empty()));
}

#[test]
fn native_copy_encodes_every_dtype_in_logical_order_without_allocating() {
    let stream = crate::test_stream();
    macro_rules! reversed {
        ($values:expr, $encode:expr) => {{
            let values = $values;
            let expected: Vec<u8> = values.into_iter().rev().flat_map($encode).collect();
            let source = Array::from_slice(&values, &[2, 2]);
            let reverse = source
                .as_strided(&[2, 2][..], &[-2, -1][..], 3, stream)
                .unwrap();
            check(&reverse, &expected);
        }};
    }
    reversed!([false, true, true, false], |v| [u8::from(v)]);
    reversed!([1_u8, 5, 128, 255], |v| [v]);
    reversed!([1_u16, 5, 32768, u16::MAX], u16::to_ne_bytes);
    reversed!([1_u32, 5, 1 << 31, u32::MAX], u32::to_ne_bytes);
    reversed!([1_u64, 5, 1 << 63, u64::MAX], u64::to_ne_bytes);
    reversed!([-1_i8, 5, i8::MIN, i8::MAX], i8::to_ne_bytes);
    reversed!([-1_i16, 5, i16::MIN, i16::MAX], i16::to_ne_bytes);
    reversed!([-1_i32, 5, i32::MIN, i32::MAX], i32::to_ne_bytes);
    reversed!([-1_i64, 5, i64::MIN, i64::MAX], i64::to_ne_bytes);
    reversed!(
        [f32::from_bits(0x7fc01234), -0.0, f32::INFINITY, -2.5],
        f32::to_ne_bytes
    );
    reversed!(
        [0x7e12, 0x8000, 0x7c00, 0xc100].map(half::f16::from_bits),
        |v: half::f16| v.to_bits().to_ne_bytes()
    );
    reversed!(
        [0x7fc1, 0x8000, 0x7f80, 0xc020].map(half::bf16::from_bits),
        |v: half::bf16| v.to_bits().to_ne_bytes()
    );
    reversed!(
        [
            complex64::new(1.0, -2.0),
            complex64::new(-0.0, 4.0),
            complex64::new(-5.0, f32::INFINITY),
            complex64::new(7.0, -8.0),
        ],
        |v: complex64| v.re.to_ne_bytes().into_iter().chain(v.im.to_ne_bytes())
    );
    let values = [
        f64::from_bits(0x7ff8000000001234),
        -0.0,
        f64::INFINITY,
        -2.5,
    ];
    let source = Array::from_slice_f64(&values, &[2, 2]);
    let reverse = source
        .as_strided(&[2, 2][..], &[-2, -1][..], 3, stream)
        .unwrap();
    check(
        &reverse,
        &values
            .into_iter()
            .rev()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
}

#[test]
fn native_copy_handles_partial_rows_broadcast_empty_scalar_and_unaligned_storage() {
    let stream = crate::test_stream();
    let source = Array::from_iter(1..=30_i32, &[2, 5, 3]);
    let prefix = source.try_index_device((.., ..2, ..), stream).unwrap();
    let transpose = prefix.transpose_axes(&[2, 0, 1], stream).unwrap();
    check(
        &transpose,
        &[1_i32, 4, 16, 19, 2, 5, 17, 20, 3, 6, 18, 21]
            .into_iter()
            .flat_map(i32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
    let scalar = Array::from_slice(&[-2.5_f32], &[]);
    check(&scalar, &(-2.5_f32).to_ne_bytes());
    let broadcast = broadcast_to(&scalar, &[4097], stream).unwrap();
    check(&broadcast, &(-2.5_f32).to_ne_bytes().repeat(4097));
    check(&Array::from_slice::<u32>(&[], &[2, 0, 3]), &[]);

    let values = [0x01234567_u32, 0x89abcdef, 0xffeeddcc];
    let mut encoded = vec![0xff];
    encoded.extend(values.into_iter().flat_map(u32::to_ne_bytes));
    let source = Array::from_slice(&encoded, &[encoded.len() as i32]);
    let unaligned = source
        .try_index_device(1.., stream)
        .unwrap()
        .view::<u32>(stream)
        .unwrap();
    assert_eq!(
        unaligned.evaluated().unwrap().try_as_slice::<u32>(),
        Err(AsSliceError::Misaligned)
    );
    let reverse = unaligned
        .as_strided(&[3][..], &[-1][..], 2, stream)
        .unwrap();
    check(
        &reverse,
        &values
            .into_iter()
            .rev()
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>(),
    );
}
