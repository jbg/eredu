use super::*;
use crate::{
    error::NativeBytesCopyError,
    ops::{broadcast_to, indexing::TryIndexOp},
    utils::allocation_test::measure,
};

#[test]
fn completed_region_reader_preserves_signed_layout_and_refuses_without_writes() {
    let stream = crate::test_stream();
    let source = Array::from_slice(
        &[1_f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        &[3, 4],
    );
    let reversed = source
        .as_strided(&[3, 4][..], &[-4, -1][..], 11, stream)
        .unwrap();
    reversed.evaluated().unwrap();
    let mut output = [-99_f32; 4];
    let (result, allocations) = measure(|| {
        reversed.try_completed()?.try_map_region_into::<f32, f32>(
            &[1, 1],
            &[2, 2],
            &mut output,
            |v| v * 2.,
        )
    });
    assert_eq!(result, Ok(()));
    assert_eq!(allocations, 0);
    assert_eq!(output, [14., 12., 6., 4.]);
    for (starts, shape) in [([2, 3], [2, 2]), ([u64::MAX, 0], [2, 2]), ([0, 0], [1, 3])] {
        let mut sentinel = [-99_f32; 4];
        assert!(
            reversed
                .try_completed()
                .unwrap()
                .try_map_region_into::<f32, f32>(&starts, &shape, &mut sentinel, |v| v)
                .is_err()
        );
        assert_eq!(sentinel, [-99.; 4]);
    }
    let lazy = source.add(&source, stream).unwrap();
    assert!(matches!(
        lazy.try_completed(),
        Err(crate::error::CompletedReadbackError::Attribution)
    ));
    assert!(Array::completed_borrow_control_bytes().is_some_and(|n| n > 0));
    assert!(
        EvaluatedArray::completed_region_readback_control_bytes::<f32, f32>()
            .is_some_and(|n| n > 0)
    );
}

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

    let width = array.item_size();
    let elements = expected.len() / width;
    for start in 0..=elements {
        let end = (start + 2).min(elements);
        guarded.fill(0xa5);
        let length = (end - start) * width;
        let (result, allocations) = measure(|| {
            evaluated.try_copy_native_element_range_into(start, &mut guarded[1..length + 1])
        });
        assert_eq!(result, Ok(()));
        assert_eq!(allocations, 0);
        assert_eq!(
            &guarded[1..length + 1],
            &expected[start * width..end * width]
        );
        assert_eq!(guarded[0], 0xa5);
        assert!(guarded[length + 1..].iter().all(|&byte| byte == 0xa5));
    }
    for (start, length) in [(elements + 1, 0), (elements, width), (usize::MAX, width)] {
        let mut invalid = [0x5a; 16];
        assert!(
            evaluated
                .try_copy_native_element_range_into(start, &mut invalid[1..length + 1])
                .is_err()
        );
        assert!(invalid.iter().all(|&byte| byte == 0x5a));
    }
    if width > 1 {
        let mut invalid = [0x5a; 16];
        assert!(
            evaluated
                .try_copy_native_element_range_into(0, &mut invalid[1..width])
                .is_err()
        );
        assert!(invalid.iter().all(|&byte| byte == 0x5a));
    }
    assert!(EvaluatedArray::native_element_range_readback_control_bytes().is_some_and(|n| n > 0));

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

#[test]
fn completed_readback_keeps_backing_identity_and_uses_only_caller_storage() {
    let stream = crate::test_stream();
    let original = Array::from_slice(&[1.25_f32, -2.5, 7.0, 9.5, -4.0, 3.0], &[2, 3]);
    let copied = original.copy(stream).unwrap();
    let reversed = copied
        .as_strided(&[2, 3][..], &[-3, -1][..], 5, stream)
        .unwrap();
    let evaluated = reversed.evaluated().unwrap();
    let before = reversed.allocation_info().unwrap().unwrap();
    let mut output = [0.0_f32; 6];
    let (result, allocations) = measure(|| evaluated.try_copy_into(&mut output));
    assert_eq!(result, Ok(()));
    assert_eq!(allocations, 0);
    assert_eq!(output, [3.0, -4.0, 9.5, 7.0, -2.5, 1.25]);
    assert_eq!(reversed.allocation_info().unwrap().unwrap(), before);
    let mut wrong = [13.0_f32; 5];
    assert_eq!(
        evaluated.try_copy_into(&mut wrong),
        Err(crate::error::CompletedReadbackError::DestinationLength {
            expected: 6,
            found: 5
        })
    );
    assert_eq!(wrong, [13.0; 5]);
    let mut wrong_type = [5_i32; 6];
    assert!(matches!(
        evaluated.try_copy_into(&mut wrong_type),
        Err(crate::error::CompletedReadbackError::Source(
            AsSliceError::DtypeMismatch { .. }
        ))
    ));
    assert_eq!(wrong_type, [5; 6]);
    let empty = Array::from_slice::<f32>(&[], &[0]);
    assert_eq!(
        empty.evaluated().unwrap().try_copy_into::<f32>(&mut []),
        Ok(())
    );
}

#[test]
fn completed_mapped_readback_converts_strided_values_without_staging() {
    let stream = crate::test_stream();
    let values = [
        half::bf16::from_f32(1.5),
        half::bf16::from_f32(-3.25),
        half::bf16::from_f32(7.0),
    ];
    let source = Array::from_slice(&values, &[3]);
    let reversed = source.as_strided(&[3][..], &[-1][..], 2, stream).unwrap();
    let evaluated = reversed.evaluated().unwrap();
    let mut output = [0.0_f32; 3];
    let (result, allocations) = measure(|| evaluated.try_map_into(&mut output, half::bf16::to_f32));
    assert_eq!(result, Ok(()));
    assert_eq!(allocations, 0);
    assert_eq!(output, [7.0, -3.25, 1.5]);
    let source = Array::from_slice(&[u32::MAX, 7_u32, 23], &[3]);
    let mut wide = [0_u64; 3];
    source
        .evaluated()
        .unwrap()
        .try_map_into::<u32, u64>(&mut wide, u64::from)
        .unwrap();
    assert_eq!(wide, [u64::from(u32::MAX), 7, 23]);
}

#[cfg(any(feature = "metal", feature = "cuda"))]
#[test]
#[ignore = "requires native accelerators; use --test-threads=1"]
fn completed_readback_from_actual_accelerators_preserves_each_source_backing() {
    let devices = crate::physical_memory_topology().unwrap();
    assert!(
        !devices.is_empty(),
        "native accelerator validation requires a device"
    );
    for device in devices {
        let stream = crate::Stream::new_with_device(&crate::Device::new(
            crate::DeviceType::Gpu,
            device.ordinal as i32,
        ));
        let left = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]);
        let right = Array::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2]);
        let result = left.matmul(&right, &stream).unwrap();
        let completed = result.evaluated().unwrap();
        let before = result.allocation_info().unwrap().unwrap();
        let mut output = [0.0_f32; 4];
        completed.try_copy_into(&mut output).unwrap();
        assert_eq!(output, [19.0, 22.0, 43.0, 50.0]);
        let mut converted = [0_f64; 4];
        completed
            .try_map_into::<f32, f64>(&mut converted, f64::from)
            .unwrap();
        assert_eq!(converted, [19.0, 22.0, 43.0, 50.0]);
        assert_eq!(result.allocation_info().unwrap(), Some(before));
    }
}

#[test]
fn availability_observation_publishes_completed_event_without_evaluating_lazy_sources() {
    let stream = crate::test_stream();
    let source = Array::from_slice(&[1i32, -4, 9], &[3]);
    let lazy = source.add(&source, stream).unwrap();
    let alias = lazy.clone();
    let (pending, allocations) = measure(|| lazy.try_observe_availability());
    assert_eq!(pending.unwrap(), Some(false));
    assert_eq!(allocations, 0);
    assert!(matches!(
        lazy.try_completed(),
        Err(crate::error::CompletedReadbackError::Attribution)
    ));
    assert!(
        alias
            .try_descriptor()
            .unwrap()
            .facts()
            .allocation()
            .is_none()
    );
    let event = crate::transforms::async_eval_with_event([&lazy]).unwrap();
    event.synchronize().unwrap();
    let (ready, allocations) = measure(|| lazy.try_observe_availability());
    assert_eq!(ready.unwrap(), Some(true));
    assert_eq!(allocations, 0);
    let mut output = [0i32; 3];
    lazy.try_completed()
        .unwrap()
        .try_copy_into(&mut output)
        .unwrap();
    assert_eq!(output, [2, -8, 18]);
    let allocation = lazy.allocation_info().unwrap();
    assert!(allocation.is_some());
    assert_eq!(alias.allocation_info().unwrap(), allocation);
    let (repeated, allocations) = measure(|| alias.try_observe_availability());
    assert_eq!(repeated.unwrap(), Some(true));
    assert_eq!(allocations, 0);
    assert_eq!(alias.allocation_info().unwrap(), allocation);
}
