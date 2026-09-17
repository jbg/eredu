use super::*;
#[allow(dead_code)]
#[path = "original.rs"]
mod original;

fn descriptor(ty: GgmlType, dims: Vec<u64>) -> TensorDescriptor {
    let (values, bytes) = ty.block_and_bytes().unwrap();
    TensorDescriptor {
        name: "layer.α.weight".into(),
        byte_len: dims.iter().product::<u64>() / values * bytes,
        dimensions: dims,
        ggml_type: ty,
        relative_offset: 64,
        data_offset: 256,
    }
}
fn axis_compare(d: &TensorDescriptor, selection: TensorSelection) {
    let old = original::TensorSelectionPlan::new(d, selection.clone());
    let ordinary = TensorSelectionPlan::new(d, selection.clone());
    let mut scratch = AxisScratch::empty();
    scratch.prepare(d, &selection).unwrap();
    let capacities = (
        scratch.indices.capacity(),
        scratch.ranges.capacity(),
        scratch.dimensions.capacity(),
        scratch.spans.capacity(),
        scratch.selected.dimensions.capacity(),
    );
    let addresses = (
        scratch.indices.as_ptr(),
        scratch.ranges.as_ptr(),
        scratch.dimensions.as_ptr(),
        scratch.spans.as_ptr(),
        scratch.selected.name.as_ptr(),
        scratch.selected.dimensions.as_ptr(),
    );
    {
        let fixed = axis_plan(d, &selection, AxisStorage::new(Some(&mut scratch)));
        match old {
            Ok(old) => {
                let ordinary = ordinary.unwrap();
                let fixed = fixed.unwrap();
                assert_eq!(ordinary.selected_descriptor(), old.selected_descriptor());
                assert_eq!(&*fixed.selected_descriptor, old.selected_descriptor());
                assert_eq!(ordinary.selection(), old.selection());
                assert_eq!(ordinary.gguf_dimension(), old.gguf_dimension());
                assert_eq!(ordinary.alignment(), old.alignment());
                assert_eq!(
                    fixed.view().encoded_spans().collect::<Vec<_>>(),
                    old.encoded_spans().collect::<Vec<_>>()
                );
                assert_eq!(
                    ordinary.encoded_spans().collect::<Vec<_>>(),
                    old.encoded_spans().collect::<Vec<_>>()
                );
                assert_eq!(fixed.encoded_byte_len, old.encoded_byte_len());
            }
            Err(old) => {
                assert_eq!(ordinary.unwrap_err().to_string(), old.to_string());
                let Err(MetadataDestinationError::Gguf(error)) = fixed else {
                    panic!("original validation cause")
                };
                assert_eq!(error.to_string(), old.to_string());
                assert_eq!(format!("{error:?}"), format!("{old:?}"));
            }
        }
    }
    assert_eq!(
        capacities,
        (
            scratch.indices.capacity(),
            scratch.ranges.capacity(),
            scratch.dimensions.capacity(),
            scratch.spans.capacity(),
            scratch.selected.dimensions.capacity()
        )
    );
    assert_eq!(
        addresses,
        (
            scratch.indices.as_ptr(),
            scratch.ranges.as_ptr(),
            scratch.dimensions.as_ptr(),
            scratch.spans.as_ptr(),
            scratch.selected.name.as_ptr(),
            scratch.selected.dimensions.as_ptr()
        )
    );
}
#[test]
fn axis_storage_preserves_original_checks_order_and_compact_repetitions() {
    for ty in [
        GgmlType::F32,
        GgmlType::Q4_0,
        GgmlType::Q8_0,
        GgmlType::MxFp4,
    ] {
        let d = descriptor(ty, vec![64, 4, 3]);
        for selection in [
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 3,
            },
            TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 3,
            },
            TensorSelection::Range {
                axis: 2,
                start: 32,
                end: 64,
            },
            TensorSelection::Indices {
                axis: 0,
                indices: vec![2, 0, 2],
            },
            TensorSelection::Indices {
                axis: 1,
                indices: vec![3, 0, 1, 1],
            },
            TensorSelection::Indices {
                axis: 2,
                indices: (32..64).chain(0..32).chain(32..64).collect(),
            },
            TensorSelection::Indices {
                axis: 2,
                indices: vec![1, 0, 1],
            },
            TensorSelection::Range {
                axis: 9,
                start: usize::MAX,
                end: 0,
            },
            TensorSelection::Indices {
                axis: 1,
                indices: vec![],
            },
            TensorSelection::Range {
                axis: 1,
                start: 1,
                end: 5,
            },
        ] {
            axis_compare(&d, selection);
        }
    }
    let d = descriptor(GgmlType::F32, vec![2, 1_000_000]);
    let selection = TensorSelection::Range {
        axis: 1,
        start: 1,
        end: 2,
    };
    let mut scratch = AxisScratch::empty();
    scratch.prepare(&d, &selection).unwrap();
    let plan = axis_plan(&d, &selection, AxisStorage::new(Some(&mut scratch))).unwrap();
    assert_eq!(plan.repetitions, 1_000_000);
    assert_eq!(plan.relative_spans.len(), 1);
    assert_eq!(plan.view().encoded_spans().count(), 1_000_000);
}

#[test]
fn span_storage_preserves_original_rank_changes_offsets_and_first_errors() {
    for ty in [GgmlType::F32, GgmlType::Q4_0, GgmlType::Q8_0] {
        let d = descriptor(ty, vec![64, 4]);
        for selection in [
            DenseTensorSpan::new(32, vec![1, 2, 32]).unwrap(),
            DenseTensorSpan::new(64, vec![64]).unwrap(),
            DenseTensorSpan::new(1, vec![32]).unwrap(),
            DenseTensorSpan::new(250, vec![32]).unwrap(),
        ] {
            let old = original::DenseTensorSpanPlan::new(&d, selection.clone());
            let ordinary = DenseTensorSpanPlan::new(&d, selection.clone());
            let mut selected = empty_descriptor();
            prepare_descriptor(
                &mut selected,
                &d,
                d.dimensions.len().max(selection.shape.len()),
            )
            .unwrap();
            let address = (selected.name.as_ptr(), selected.dimensions.as_ptr());
            let fixed = span_plan(&d, &selection, Some(&mut selected));
            match old {
                Ok(old) => {
                    let ordinary = ordinary.unwrap();
                    let fixed = fixed.unwrap();
                    assert_eq!(ordinary.selected_descriptor(), old.selected_descriptor());
                    assert_eq!(&*fixed.selected_descriptor, old.selected_descriptor());
                    assert_eq!(fixed.encoded_span, old.encoded_span());
                }
                Err(old) => {
                    assert_eq!(ordinary.unwrap_err().to_string(), old.to_string());
                    let Err(MetadataDestinationError::Gguf(error)) = fixed else {
                        panic!("original span error")
                    };
                    assert_eq!(format!("{error:?}"), format!("{old:?}"));
                }
            }
            assert_eq!(
                address,
                (selected.name.as_ptr(), selected.dimensions.as_ptr())
            );
        }
    }
    for d in [
        TensorDescriptor {
            dimensions: vec![],
            ..descriptor(GgmlType::F32, vec![4])
        },
        TensorDescriptor {
            dimensions: vec![u64::MAX, 2],
            ..descriptor(GgmlType::F32, vec![4])
        },
        TensorDescriptor {
            byte_len: 1,
            ..descriptor(GgmlType::F32, vec![4])
        },
        TensorDescriptor {
            data_offset: u64::MAX - 1,
            ..descriptor(GgmlType::F32, vec![4])
        },
    ] {
        axis_compare(
            &d,
            TensorSelection::Range {
                axis: 0,
                start: 0,
                end: 1,
            },
        );
    }
}

#[test]
fn fixed_scratch_failure_keeps_prefix_and_preceding_original_error() {
    let d = descriptor(GgmlType::F32, vec![4, 3]);
    let selection = TensorSelection::Indices {
        axis: 0,
        indices: vec![2, 0, 2],
    };
    let mut scratch = AxisScratch::empty();
    scratch.prepare(&d, &selection).unwrap();
    scratch.spans = Vec::new();
    let result = axis_plan(&d, &selection, AxisStorage::new(Some(&mut scratch)));
    assert!(matches!(
        result,
        Err(MetadataDestinationError::Capacity {
            field: "relative spans",
            required: 1,
            capacity: 0
        })
    ));
    assert_eq!(scratch.indices, [2, 0, 2]);
    assert_eq!(scratch.ranges, [(2, 1), (0, 1), (2, 1)]);
    let invalid = TensorSelection::Indices {
        axis: 8,
        indices: vec![usize::MAX],
    };
    assert!(matches!(
        axis_plan(&d, &invalid, AxisStorage::new(Some(&mut scratch))),
        Err(MetadataDestinationError::Gguf(_))
    ));
}
