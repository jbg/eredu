use super::*;

fn fixture() -> (RoutedUnitCapture, ResolvedCaptureSlice) {
    let rows = [(3, 2), (1, 0), (3, 0), (1, 2)]
        .into_iter()
        .map(|(token, slot)| RoutedUnitCaptureRow {
            source_peer: None,
            token,
            slot,
            expert: (token + slot) % 5,
            coefficient: 0.2 + (slot as f32) * 0.1,
            unit_start: 1,
            unit_stride: 2,
            values: crate::TensorObservation::new(
                vec![2],
                crate::TensorObservationData::F32(vec![token as f32 + 0.25, f32::INFINITY]),
            )
            .unwrap(),
        })
        .collect();
    (
        RoutedUnitCapture {
            geometry: RoutedUnitGeometry {
                experts: 5,
                units_per_expert: 4,
                routes_per_token: 3,
            },
            source_token_ranges: vec![[2, 4], [0, 2]],
            rows,
        },
        ResolvedCaptureSlice {
            starts: vec![1, 0, 1],
            ends: vec![4, 3, 4],
            strides: vec![2, 2, 2],
            shape: vec![2, 2, 2],
        },
    )
}

#[test]
fn prepaid_routed_validation_preserves_sparse_values_and_detects_duplicate_identity() {
    let (mut value, slice) = fixture();
    let original = value.clone();
    let mut short = [RoutedUnitRowIdentity::default(); 3];
    assert_eq!(
        value.validate_rows_with_scratch(&slice, &mut short),
        Err(RoutedUnitValidationError::Scratch)
    );
    assert_eq!(value, original);
    let mut scratch = [RoutedUnitRowIdentity::default(); 4];
    value
        .validate_rows_with_scratch(&slice, &mut scratch)
        .unwrap();
    assert_eq!(value, original);
    let data = value.rows[0].values.data();
    let crate::TensorObservationData::F32(data) = data else {
        panic!("fixture precision");
    };
    let backing = data.as_ptr();
    value
        .finish_ordinary_with_scratch(&slice, 4, &mut scratch)
        .unwrap();
    assert_eq!(value.source_token_ranges, [[0, 2], [2, 4]]);
    assert_eq!(
        value
            .rows
            .iter()
            .map(|row| (row.token, row.slot))
            .collect::<Vec<_>>(),
        [(1, 0), (1, 2), (3, 0), (3, 2)]
    );
    let crate::TensorObservationData::F32(data) = value.rows[3].values.data() else {
        panic!("precision");
    };
    assert_eq!(data.as_ptr(), backing);
    assert!(data[1].is_infinite());
    let mut ordinary = original.clone();
    ordinary.finish_ordinary(&slice, 4).unwrap();
    assert_eq!(ordinary, value);

    let (mut duplicate, _) = fixture();
    duplicate.rows[1] = duplicate.rows[0].clone();
    // A different expert/coefficient does not change original route identity.
    duplicate.rows[1].expert = 0;
    duplicate.rows[1].coefficient = 0.75;
    assert_eq!(
        duplicate.validate_rows_with_scratch(&slice, &mut scratch),
        Err(RoutedUnitValidationError::Row)
    );
    duplicate.rows[1].source_peer = Some(1);
    duplicate
        .validate_rows_with_scratch(&slice, &mut scratch)
        .unwrap();
    assert_eq!(
        duplicate.finish_ordinary_with_scratch(&slice, 4, &mut scratch),
        Err(RoutedUnitValidationError::Incomplete)
    );
}

#[test]
fn prepaid_routed_validation_keeps_typed_geometry_and_chunk_refusals() {
    let (mut value, mut slice) = fixture();
    let mut scratch = [RoutedUnitRowIdentity::default(); 4];
    slice.strides[0] = 0;
    assert_eq!(
        value.validate_rows_with_scratch(&slice, &mut scratch),
        Err(RoutedUnitValidationError::SliceGeometry)
    );
    slice.strides[0] = 2;
    value.rows[0].coefficient = f32::NAN;
    assert_eq!(
        value.validate_rows_with_scratch(&slice, &mut scratch),
        Err(RoutedUnitValidationError::Row)
    );
    value.rows[0].coefficient = 0.5;
    value.rows[0].unit_stride = 1;
    assert_eq!(
        value.validate_rows_with_scratch(&slice, &mut scratch),
        Err(RoutedUnitValidationError::Selection)
    );
    value.rows[0].unit_stride = 2;
    value.source_token_ranges[0] = [1, 4];
    assert_eq!(
        value.finish_ordinary_with_scratch(&slice, 4, &mut scratch),
        Err(RoutedUnitValidationError::Chunks)
    );
    value.source_token_ranges = vec![[0, 4]];
    value.rows.pop();
    assert_eq!(
        value.finish_ordinary_with_scratch(&slice, 4, &mut scratch),
        Err(RoutedUnitValidationError::Incomplete)
    );
}
