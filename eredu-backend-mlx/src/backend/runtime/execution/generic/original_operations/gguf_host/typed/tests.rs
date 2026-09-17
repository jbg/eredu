use super::*;
use eredu_gguf::{Endian, GgmlType, TensorDescriptor};
fn descriptor(ty: GgmlType, dimensions: Vec<u64>, bytes: u64) -> TensorDescriptor {
    TensorDescriptor {
        name: "actual.typed.weight".into(),
        dimensions,
        ggml_type: ty,
        relative_offset: 0,
        data_offset: 0,
        byte_len: bytes,
    }
}

#[test]
fn selected_requests_keep_encoded_extent_separate_from_declared_dense_shape() {
    for (ty, dtype, width) in [
        (GgmlType::F32, LogicalDtype::F32, 4),
        (GgmlType::F16, LogicalDtype::F16, 2),
        (GgmlType::Bf16, LogicalDtype::Bf16, 2),
        (GgmlType::I8, LogicalDtype::I8, 1),
        (GgmlType::I16, LogicalDtype::I16, 2),
        (GgmlType::I32, LogicalDtype::I32, 4),
        (GgmlType::I64, LogicalDtype::I64, 8),
        (GgmlType::F64, LogicalDtype::F64, 8),
    ] {
        for endian in [Endian::Little, Endian::Big] {
            let plan =
                ConversionPlan::new(&descriptor(ty, vec![3], (width * 4) as u64), endian).unwrap();
            let request = TypedOutputRequest::for_output(&plan, 0).unwrap();
            assert_eq!(request.dtype, dtype);
            assert_eq!(request.kind, TransformKind::NativeBytes { width });
            assert_eq!(
                (request.input_elements, request.output_elements),
                (width * 4, 4)
            );
            assert_eq!(request.declared_shape_elements, Some(3));
            assert!(request.accepts_input(width * 4));
            assert!(!request.accepts_input(width * 3));
        }
    }
    let plan =
        ConversionPlan::new(&descriptor(GgmlType::F32, vec![3], 17), Endian::Little).unwrap();
    let request = TypedOutputRequest::for_output(&plan, 0).unwrap();
    assert_eq!(
        (
            request.input_elements,
            request.output_elements,
            request.declared_shape_elements
        ),
        (17, 4, Some(3))
    );
    assert_eq!(request.requested_layout().unwrap().size(), 16);
    assert!(TypedOutputRequest::for_output(&plan, 1).is_none());
}

#[test]
fn affine_and_mxfp4_requests_retain_real_g2_representations_and_order() {
    let affine =
        ConversionPlan::new(&descriptor(GgmlType::Q4_0, vec![32, 2], 36), Endian::Big).unwrap();
    let a: Vec<_> = (0..3)
        .map(|i| TypedOutputRequest::for_output(&affine, i).unwrap())
        .collect();
    assert_eq!(
        a.iter()
            .map(|r| (r.dtype, r.kind, r.output_elements))
            .collect::<Vec<_>>(),
        vec![
            (LogicalDtype::U32, TransformKind::Move, 8),
            (LogicalDtype::F16, TransformKind::HalfBits, 2),
            (LogicalDtype::F16, TransformKind::HalfBits, 2),
        ]
    );
    let mxfp4 = ConversionPlan::new(
        &descriptor(GgmlType::MxFp4, vec![32, 2], 34),
        Endian::Little,
    )
    .unwrap();
    let weights = TypedOutputRequest::for_output(&mxfp4, 0).unwrap();
    let scales = TypedOutputRequest::for_output(&mxfp4, 1).unwrap();
    assert_eq!(
        (weights.dtype, weights.kind, weights.output_elements),
        (LogicalDtype::U32, TransformKind::Move, 8)
    );
    assert_eq!(
        (scales.dtype, scales.kind, scales.output_elements),
        (LogicalDtype::U8, TransformKind::Move, 2)
    );
    let packed =
        ConversionPlan::new(&descriptor(GgmlType::Q8_0, vec![32, 2], 68), Endian::Little).unwrap();
    let packed = TypedOutputRequest::for_output(&packed, 0).unwrap();
    assert_eq!(
        (packed.dtype, packed.kind, packed.output_elements),
        (LogicalDtype::U8, TransformKind::Move, 68)
    );
}

#[test]
fn destination_writes_keep_final_allocation_and_reject_reuse_without_refill() {
    let mut destination = TypedDestination::<u32>::try_new(4).unwrap();
    let pointer = destination.values().as_ptr();
    let capacity = destination.values().capacity();
    destination
        .fill(4, [0x7fc0_0021, 0x8000_0000, 7, u32::MAX].into_iter())
        .unwrap();
    assert_eq!(
        destination.values().as_slice(),
        &[0x7fc0_0021, 0x8000_0000, 7, u32::MAX]
    );
    assert_eq!(destination.values().as_ptr(), pointer);
    assert_eq!(destination.values().capacity(), capacity);
    assert!(matches!(
        destination.fill(1, [9].into_iter()),
        Err(DestinationCause::AlreadyFilled)
    ));
    let values = destination.into_values();
    assert_eq!((values.as_ptr(), values.capacity()), (pointer, capacity));
    assert_eq!(values.as_slice()[0], 0x7fc0_0021);
    let mut zero = TypedDestination::<u8>::try_new(0).unwrap();
    zero.fill(0, [].into_iter()).unwrap();
    assert!(zero.into_values().as_slice().is_empty());
    let (failed, cause) = TypedDestination::<u64>::try_new(usize::MAX).unwrap_err();
    assert!(matches!(cause, DestinationCause::Layout));
    assert!(failed.values().as_slice().is_empty());
}

#[test]
fn short_storage_and_unexpected_iterator_extent_retain_actual_accepted_prefix() {
    let mut short = TypedDestination::<i8> {
        values: Vec::new().into(),
        requested: 2,
        started: false,
    };
    assert!(matches!(
        short.fill(2, [1, 2].into_iter()),
        Err(DestinationCause::Capacity {
            requested: 2,
            actual: 0
        })
    ));
    assert_eq!(short.values().capacity(), 0);
    // Internal contract fault: the bounded loop must still never grow even if
    // an iterator lies. The accepted prefix stays in the same final destination.
    struct Longer {
        values: std::array::IntoIter<u32, 3>,
    }
    impl Iterator for Longer {
        type Item = u32;
        fn next(&mut self) -> Option<u32> {
            self.values.next()
        }
        fn size_hint(&self) -> (usize, Option<usize>) {
            (2, Some(2))
        }
    }
    impl ExactSizeIterator for Longer {
        fn len(&self) -> usize {
            2
        }
    }
    let mut destination = TypedDestination::try_new(2).unwrap();
    let pointer = destination.values().as_ptr();
    let capacity = destination.values().capacity();
    let cause = destination
        .fill(
            2,
            Longer {
                values: [11, 23, 37].into_iter(),
            },
        )
        .unwrap_err();
    assert!(matches!(
        cause,
        DestinationCause::Extent {
            requested: 2,
            actual: 3
        }
    ));
    assert_eq!(destination.values().as_slice(), &[11, 23]);
    assert_eq!(
        (
            destination.values().as_ptr(),
            destination.values().capacity()
        ),
        (pointer, capacity)
    );
}
