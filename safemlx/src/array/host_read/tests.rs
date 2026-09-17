use super::*;
use crate::ops::{broadcast_to, indexing::TryIndexOp};

fn check<T: FromSliceElement + Copy + PartialEq + std::fmt::Debug>(array: &Array, expected: &[T]) {
    let value = array.evaluated().unwrap();
    let copied = value.try_to_vec::<T>().unwrap();
    assert_eq!(copied, expected);
    assert_eq!(copied.capacity(), expected.len());
    let mut forward = value.try_iter::<T>().unwrap();
    let mut backward = expected.iter().copied().rev();
    assert_eq!(forward.len(), expected.len());
    while let Some(last) = backward.next() {
        assert_eq!(forward.next_back(), Some(last));
    }
    assert_eq!(forward.next(), None);
    assert_eq!(forward.len(), 0);
    assert!(value.try_iter::<T>().unwrap().eq(expected.iter().copied()));
    let reference = Array::from_slice(expected, array.shape())
        .into_evaluated()
        .unwrap();
    assert!(value.equal_values(&reference));
    assert!(reference.equal_values(&value));
    assert_eq!(value.to_native_bytes(), reference.to_native_bytes());
    assert_eq!(
        value.deep_clone().unwrap().try_to_vec::<T>().unwrap(),
        expected
    );
}

#[test]
fn host_reads_follow_transposes_and_capacity_backed_prefixes() {
    let stream = crate::test_stream();
    let source = Array::from_iter(1..=30_i32, &[2, 5, 3]);
    let prefix = source.try_index_device((.., ..2, ..), stream).unwrap();
    let evaluated = prefix.evaluated().unwrap();
    assert_eq!(prefix.signed_strides(), [15, 3, 1]);
    assert_eq!(
        evaluated.try_as_slice::<i32>(),
        Err(AsSliceError::NonContiguous)
    );
    check(&prefix, &[1, 2, 3, 4, 5, 6, 16, 17, 18, 19, 20, 21]);
    let transpose = prefix.transpose_axes(&[2, 0, 1], stream).unwrap();
    check(&transpose, &[1, 4, 16, 19, 2, 5, 17, 20, 3, 6, 18, 21]);
}

#[test]
fn host_reads_broadcast_without_reading_logical_size_from_small_backing() {
    let stream = crate::test_stream();
    let scalar = Array::from_slice(&[7_i32], &[]);
    let repeated = broadcast_to(&scalar, &[4097], stream).unwrap();
    let value = repeated.evaluated().unwrap();
    assert_eq!(value.as_array().signed_strides(), [0]);
    assert_eq!(
        value.try_as_slice::<i32>(),
        Err(AsSliceError::NonContiguous)
    );
    check(&repeated, &vec![7_i32; 4097]);
    let row = Array::from_slice(&[1_i32, 5, 9], &[1, 3]);
    let repeated = broadcast_to(&row, &[2, 4, 3], stream).unwrap();
    check(&repeated, &[1, 5, 9].repeat(8));
}

#[test]
fn host_reads_signed_offsets_and_ignores_singleton_axis_stride() {
    let stream = crate::test_stream();
    let source = Array::from_iter(1..=6_i32, &[6]);
    let reversed = source
        .as_strided(&[2, 3][..], &[-3, -1][..], 5, stream)
        .unwrap();
    let value = reversed.evaluated().unwrap();
    assert_eq!(value.as_array().signed_strides(), [-3, -1]);
    assert_eq!(
        value.try_as_slice::<i32>(),
        Err(AsSliceError::NonContiguous)
    );
    check(&reversed, &[6, 5, 4, 3, 2, 1]);
    let rows = source
        .as_strided(&[2, 3][..], &[-3, 1][..], 3, stream)
        .unwrap();
    check(&rows, &[4, 5, 6, 1, 2, 3]);
    let singleton = source
        .as_strided(&[1, 3][..], &[i64::MAX, 1][..], 1, stream)
        .unwrap();
    assert_eq!(
        singleton
            .evaluated()
            .unwrap()
            .try_as_slice::<i32>()
            .unwrap(),
        [2, 3, 4]
    );
}

#[test]
fn host_reads_unaligned_views_without_forming_rust_references() {
    let stream = crate::test_stream();
    let values = [1.25_f32, -2.5, 3.75];
    let mut bytes = vec![0xff];
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let source = Array::from_slice(&bytes, &[bytes.len() as i32]);
    let unaligned = source
        .try_index_device(1.., stream)
        .unwrap()
        .view::<f32>(stream)
        .unwrap();
    let value = unaligned.evaluated().unwrap();
    assert_eq!(value.try_as_slice::<f32>(), Err(AsSliceError::Misaligned));
    check(&unaligned, &values);
    assert_eq!(value.to_native_bytes(), &bytes[1..]);
    let reverse = unaligned
        .as_strided(&[3][..], &[-1][..], 2, stream)
        .unwrap();
    check(&reverse, &[3.75_f32, -2.5, 1.25]);
}

#[test]
fn host_reads_empty_and_scalar_views_preserve_type_checks() {
    let stream = crate::test_stream();
    let empty = Array::from_slice::<i32>(&[], &[2, 0, 3]);
    let value = empty.evaluated().unwrap();
    assert!(value.try_as_slice::<i32>().unwrap().is_empty());
    assert!(matches!(
        value.try_as_slice::<f32>(),
        Err(AsSliceError::DtypeMismatch { .. })
    ));
    assert!(matches!(
        value.try_to_vec::<f32>(),
        Err(AsSliceError::DtypeMismatch { .. })
    ));
    check(&empty, &[] as &[i32]);
    let reshaped = empty
        .as_strided(&[0, 3][..], &[3, 1][..], 0, stream)
        .unwrap();
    check(&reshaped, &[] as &[i32]);
    let scalar = Array::from_slice(&[11_i32], &[]);
    check(&scalar, &[11_i32]);
    assert_eq!(
        scalar.evaluated().unwrap().try_as_slice::<i32>().unwrap(),
        [11]
    );
}

#[test]
fn host_reads_native_bytes_cover_all_scalar_representations() {
    let stream = crate::test_stream();
    macro_rules! reversed {
        ($values:expr) => {{
            let values = $values;
            let a = Array::from_slice(&values, &[2, 2]);
            let v = a.as_strided(&[2, 2][..], &[-2, -1][..], 3, stream).unwrap();
            check(&v, &[values[3], values[2], values[1], values[0]]);
        }};
    }
    reversed!([false, true, true, false]);
    reversed!([1_u8, 5, 128, 255]);
    reversed!([1_u16, 5, 32768, 65535]);
    reversed!([1_u32, 5, 1 << 31, u32::MAX]);
    reversed!([1_u64, 5, 1 << 63, u64::MAX]);
    reversed!([-1_i8, 5, -128, 127]);
    reversed!([-1_i16, 5, i16::MIN, i16::MAX]);
    reversed!([-1_i32, 5, i32::MIN, i32::MAX]);
    reversed!([-1_i64, 5, i64::MIN, i64::MAX]);
    reversed!([1.25_f32, -2.5, -0.0, 3.75]);
    reversed!([1.25, -2.5, -0.0, 3.75].map(half::f16::from_f32));
    reversed!([1.25, -2.5, -0.0, 3.75].map(half::bf16::from_f32));
    reversed!([
        complex64::new(1.0, -2.0),
        complex64::new(3.0, 4.0),
        complex64::new(-5.0, 6.0),
        complex64::new(7.0, -8.0)
    ]);
    let values = [1.25_f64, -2.5, -0.0, 3.75];
    let source = Array::from_slice_f64(&values, &[2, 2]);
    let reverse = source
        .as_strided(&[2, 2][..], &[-2, -1][..], 3, stream)
        .unwrap();
    let actual = reverse.evaluated().unwrap();
    let expected = [3.75_f64, -0.0, -2.5, 1.25];
    assert_eq!(actual.try_to_vec::<f64>().unwrap(), expected);
    assert_eq!(
        actual.to_native_bytes(),
        expected
            .into_iter()
            .flat_map(f64::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    assert!(actual.equal_values(
        &Array::from_slice_f64(&expected, &[2, 2])
            .evaluated()
            .unwrap()
    ));
}

#[test]
fn host_reads_equality_preserves_nan_and_signed_zero_semantics() {
    let stream = crate::test_stream();
    let values = [f32::from_bits(0x7fc01234), -0.0, f32::INFINITY];
    let source = Array::from_slice(&values, &[3]);
    let reverse = source.as_strided(&[3][..], &[-1][..], 2, stream).unwrap();
    let value = reverse.evaluated().unwrap();
    assert!(!value.equal_values(&value));
    assert_eq!(
        value.to_native_bytes(),
        values
            .into_iter()
            .rev()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>()
    );
    let negative = Array::from_slice(&[-0.0_f32], &[]);
    let zeroes = broadcast_to(&negative, &[5], stream).unwrap();
    let positive = Array::from_slice(&[0.0_f32; 5], &[5]);
    assert!(zeroes
        .evaluated()
        .unwrap()
        .equal_values(&positive.evaluated().unwrap()));
}

#[test]
fn strided_view_rejects_out_of_source_geometry_and_signed_overflow() {
    let stream = crate::test_stream();
    let source = Array::from_iter(0..6_i32, &[6]);
    for (shape, strides, offset) in [
        (vec![3, 3], vec![3, 1], 0),
        (vec![2, 3], vec![-3, 1], 0),
        (vec![2, 3], vec![1], 0),
        (vec![-1, 3], vec![3, 1], 0),
        (vec![2], vec![i64::MAX], 0),
        (vec![0], vec![i64::MIN], 0),
        (vec![0], vec![1], 7),
        (vec![1], vec![1], usize::MAX),
    ] {
        assert!(source
            .as_strided(shape.as_slice(), strides.as_slice(), offset, stream)
            .is_err());
    }
    assert!(source
        .as_strided(&[i32::MAX; 3][..], None, 0, stream)
        .is_err());
    assert!(matches!(
        checked_layout(&[2], &[i64::MIN], 4),
        Err(AsSliceError::TooLarge)
    ));
    assert!(matches!(
        checked_layout(&[2], &[i64::MAX], 4),
        Err(AsSliceError::TooLarge)
    ));
}

#[cfg(all(target_vendor = "apple", feature = "metal", not(feature = "cuda")))]
#[test]
#[ignore = "requires exclusive Metal allocation measurement; use --test-threads=1"]
fn host_reads_do_not_allocate_native_compaction_or_payload_staging() {
    let stream = Stream::new_with_device(&crate::Device::new(crate::DeviceType::Gpu, 0));
    let source = Array::from_iter((1..=42).map(|n| n as f32 * 0.25), &[2, 7, 3]);
    let prefix = source.try_index_device((.., ..3, ..), &stream).unwrap();
    let transposed = prefix.transpose_axes(&[2, 0, 1], &stream).unwrap();
    let reverse = source
        .as_strided(&[42][..], &[-1][..], 41, &stream)
        .unwrap();
    let scalar = Array::from_slice(&[2.5_f32], &[]);
    let broadcast = broadcast_to(&scalar, &[4097], &stream).unwrap();
    let arrays = [source, prefix, transposed, reverse, broadcast];
    crate::transforms::eval(&arrays).unwrap();
    stream.synchronize().unwrap();
    for array in &arrays {
        let value = array.evaluated().unwrap();
        let before = crate::memory::active_memory().unwrap();
        crate::memory::reset_peak_memory().unwrap();
        let values = value.try_to_vec::<f32>().unwrap();
        let bytes = value.to_native_bytes();
        assert!(value.equal_values(&value));
        assert_eq!(values.len(), array.size());
        assert_eq!(bytes.len(), array.nbytes());
        assert_eq!(crate::memory::active_memory().unwrap(), before);
        // reset_peak_memory clears the counter; only a subsequent native
        // allocation raises it to the current active size.
        assert_eq!(crate::memory::peak_memory().unwrap(), 0);
    }
}

#[test]
fn borrowed_lookup_preserves_original_signed_strides_and_typed_bounds() {
    let stream = crate::test_stream();
    let source = Array::from_iter(1..=6_i32, &[6]);
    let reverse = source.as_strided(&[2, 3][..], &[-3, -1][..], 5, stream).unwrap();
    let completed = reverse.evaluated().unwrap();
    for (index, expected) in [(4, 2), (0, 6), (5, 1), (2, 4)] {
        assert_eq!(completed.try_get::<i32>(index).unwrap(), Some(expected));
    }
    assert_eq!(completed.try_get::<i32>(6).unwrap(), None);
    assert_eq!(completed.try_get::<i32>(usize::MAX).unwrap(), None);
    assert!(matches!(completed.try_get::<f32>(0), Err(AsSliceError::DtypeMismatch { .. })));
    let empty = Array::from_slice::<i32>(&[], &[0]);
    assert_eq!(empty.evaluated().unwrap().try_get::<i32>(0).unwrap(), None);
    let scalar = Array::from_slice(&[half::f16::from_f32(-1.75)], &[]);
    let repeated = broadcast_to(&scalar, &[4097], stream).unwrap();
    let completed = repeated.evaluated().unwrap();
    assert_eq!(completed.try_get::<half::f16>(4096).unwrap().unwrap().to_f32(), -1.75);
    let mut bytes = vec![0xff];
    bytes.extend_from_slice(&2.5_f32.to_ne_bytes());
    let bytes = Array::from_slice(&bytes, &[5]);
    let unaligned = bytes.try_index_device(1.., stream).unwrap().view::<f32>(stream).unwrap();
    assert_eq!(unaligned.evaluated().unwrap().try_get::<f32>(0).unwrap(), Some(2.5));
}
