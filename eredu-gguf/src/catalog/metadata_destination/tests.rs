use super::*;
use crate::{GgmlType, PreparedConversion, TensorInput, Writer, WriterOptions};
#[path = "original.rs"]
mod original;

pub(super) fn file(ty: GgmlType, endian: Endian) -> (tempfile::TempDir, Checkpoint) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("metadata.gguf");
    let raw = if ty == GgmlType::F32 {
        (0..128)
            .flat_map(|i| {
                let x = (i as f32 - 31.0) * 0.25;
                match endian {
                    Endian::Little => x.to_le_bytes(),
                    Endian::Big => x.to_be_bytes(),
                }
            })
            .collect::<Vec<_>>()
    } else {
        let size = ty.block_and_bytes().unwrap().1 as usize;
        (0..4)
            .flat_map(|block| {
                let mut b = vec![0; size];
                if ty == GgmlType::MxFp4 {
                    b[0] = 130;
                    for i in 0..16 {
                        b[i + 1] = (i as u8) | ((15 - i as u8) << 4);
                    }
                } else {
                    b[..2].copy_from_slice(&match endian {
                        Endian::Little => 0x3c00_u16.to_le_bytes(),
                        Endian::Big => 0x3c00_u16.to_be_bytes(),
                    });
                    for (i, v) in b.iter_mut().enumerate().skip(2) {
                        *v = ((i + block) % 63 + 1) as u8;
                    }
                }
                b
            })
            .collect::<Vec<_>>()
    };
    Writer::new(WriterOptions {
        version: 3,
        endian,
        alignment: 32,
    })
    .unwrap()
    .write(
        File::create(&path).unwrap(),
        &BTreeMap::new(),
        &[TensorInput {
            name: "prefix.α.weight",
            dimensions: &[64, 2],
            ggml_type: ty,
            data: &raw,
        }],
    )
    .unwrap();
    let checkpoint = Checkpoint::open(&path).unwrap();
    (dir, checkpoint)
}
fn selected(d: &TensorDescriptor, selection: MetadataSelection<'_>) -> TensorDescriptor {
    match selection {
        MetadataSelection::Full => d.clone(),
        MetadataSelection::Axis(s) => TensorSelectionPlan::new(d, s.clone())
            .unwrap()
            .selected_descriptor()
            .clone(),
        MetadataSelection::Span(s) => DenseTensorSpanPlan::new(d, s.clone())
            .unwrap()
            .selected_descriptor()
            .clone(),
    }
}
pub(super) fn old(
    materializer: &mut TensorMaterializer,
    selection: MetadataSelection<'_>,
) -> ConvertedCheckpointTensor {
    match selection {
        MetadataSelection::Full => materializer.old_converted_tensor_with_storage(
            "prefix.α.weight",
            RawStorage::Ordinary,
            None,
        ),
        MetadataSelection::Axis(s) => materializer.old_converted_tensor_selected_with_storage(
            "prefix.α.weight",
            s,
            RawStorage::Ordinary,
            None,
        ),
        MetadataSelection::Span(s) => materializer.old_converted_dense_tensor_span_with_storage(
            "prefix.α.weight",
            s,
            RawStorage::Ordinary,
            None,
        ),
    }
    .unwrap()
}

#[test]
fn final_result_moves_all_cold_metadata_and_matches_original_materializer() {
    for endian in [Endian::Little, Endian::Big] {
        for ty in [
            GgmlType::F32,
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::MxFp4,
        ] {
            let (_dir, checkpoint) = file(ty, endian);
            let mut materializer = checkpoint.materializer();
            let mut ordinary = checkpoint.materializer();
            let axis = TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            };
            let span = DenseTensorSpan::new(32, vec![1, 1, 32]).unwrap();
            for selection in [
                MetadataSelection::Full,
                MetadataSelection::Axis(&axis),
                MetadataSelection::Span(&span),
            ] {
                let expected = old(&mut ordinary, selection);
                let source = materializer.metadata_source("prefix.α.weight").unwrap();
                let d = selected(&source.tensor.descriptor, selection);
                let layouts = source.layouts(selection).unwrap();
                let mut metadata = PreparedTensorMetadata::prepare(source, selection).unwrap();
                let final_d = if matches!(selection, MetadataSelection::Full) {
                    &metadata.base
                } else {
                    &metadata.final_descriptor
                };
                let pointers = (
                    final_d.name.as_ptr(),
                    final_d.dimensions.as_ptr(),
                    metadata.names.as_ptr(),
                );
                let names: Vec<_> = metadata.names.iter().map(|name| name.as_ptr()).collect();
                let requested = layouts.requested_buffer_bytes();
                assert!(metadata.capacity_bytes().unwrap() >= requested);
                let mut raw = vec![0; d.byte_len as usize];
                let mut conversion = PreparedConversion::prepare(d, endian).unwrap();
                let result = materializer
                    .converted_tensor_with_metadata(
                        "prefix.α.weight",
                        selection,
                        &mut raw,
                        &mut conversion,
                        &mut metadata,
                    )
                    .unwrap();
                assert_eq!(result, expected);
                assert_eq!(result.descriptor.name.as_ptr(), pointers.0);
                assert_eq!(result.descriptor.dimensions.as_ptr(), pointers.1);
                assert_eq!(result.output_names.as_ptr(), pointers.2);
                assert_eq!(
                    result
                        .output_names
                        .iter()
                        .map(|name| name.as_ptr())
                        .collect::<Vec<_>>(),
                    names
                );
                assert!(raw.iter().any(|b| *b != 0));
                assert!(metadata.completed().is_none());
                let count = match (ty, endian) {
                    (GgmlType::MxFp4, _) => 2,
                    (GgmlType::Q4_0, _) | (GgmlType::Q8_0, Endian::Big) => 3,
                    _ => 1,
                };
                assert_eq!(result.output_names().len(), count);
                let (descriptor, names, payload) = result.into_parts();
                assert_eq!(descriptor.name.as_ptr(), pointers.0);
                assert_eq!(names.as_ptr(), pointers.2);
                assert_eq!(payload, expected.into_converted());
            }
        }
    }
}

#[test]
fn equal_extent_reordered_duplicate_selections_and_names_cannot_rebind_destination() {
    let (_dir, checkpoint) = file(GgmlType::Q4_0, Endian::Little);
    for mutation in 0..5 {
        let mut materializer = checkpoint.materializer();
        let bound = TensorSelection::Indices {
            axis: 0,
            indices: vec![0, 1],
        };
        let foreign = TensorSelection::Indices {
            axis: 0,
            indices: if mutation == 1 {
                vec![0, 0]
            } else {
                vec![1, 0]
            },
        };
        let source = materializer.metadata_source("prefix.α.weight").unwrap();
        let d = selected(&source.tensor.descriptor, MetadataSelection::Axis(&bound));
        let mut metadata =
            PreparedTensorMetadata::prepare(source, MetadataSelection::Axis(&bound)).unwrap();
        if mutation == 2 {
            metadata.names.swap(0, 1)
        }
        if mutation == 3 {
            metadata.names[0].replace_range(..6, "wrong.");
        }
        if mutation == 4 {
            metadata.location.tensor_index = 1;
        }
        let selection = if mutation < 2 { &foreign } else { &bound };
        let mut raw = vec![0x5a; d.byte_len as usize];
        let mut conversion = PreparedConversion::prepare(d, Endian::Little).unwrap();
        assert!(matches!(
            materializer.converted_tensor_with_metadata(
                "prefix.α.weight",
                MetadataSelection::Axis(selection),
                &mut raw,
                &mut conversion,
                &mut metadata
            ),
            Err(ReadDestinationError::Metadata(
                MetadataDestinationError::Binding
            ))
        ));
        assert!(metadata.used);
        assert!(metadata.completed.is_none());
        assert!(raw.iter().all(|b| *b == 0x5a));
        assert_eq!(metadata.names.len(), 3);
        assert!(matches!(
            materializer.converted_tensor_with_metadata(
                "prefix.α.weight",
                MetadataSelection::Axis(&bound),
                &mut raw,
                &mut conversion,
                &mut metadata
            ),
            Err(ReadDestinationError::Metadata(
                MetadataDestinationError::Used
            ))
        ));
    }
}

#[test]
fn late_descriptor_refusal_retains_completed_payload_and_all_other_owners() {
    for span_case in [false, true] {
        let (_dir, checkpoint) = file(GgmlType::F32, Endian::Big);
        let mut materializer = checkpoint.materializer();
        let mut ordinary = checkpoint.materializer();
        let axis = TensorSelection::Range {
            axis: 0,
            start: 1,
            end: 2,
        };
        let span = DenseTensorSpan::new(32, vec![32]).unwrap();
        let selection = if span_case {
            MetadataSelection::Span(&span)
        } else {
            MetadataSelection::Axis(&axis)
        };
        let expected = old(&mut ordinary, selection);
        let source = materializer.metadata_source("prefix.α.weight").unwrap();
        let d = selected(&source.tensor.descriptor, selection);
        let mut metadata = PreparedTensorMetadata::prepare(source, selection).unwrap();
        let names = metadata.names.as_ptr();
        let base = metadata.base.name.as_ptr();
        metadata.final_descriptor.dimensions = Vec::new();
        let mut raw = vec![0; d.byte_len as usize];
        let mut conversion = PreparedConversion::prepare(d, Endian::Big).unwrap();
        let error = materializer
            .converted_tensor_with_metadata(
                "prefix.α.weight",
                selection,
                &mut raw,
                &mut conversion,
                &mut metadata,
            )
            .unwrap_err();
        assert!(matches!(
            error,
            ReadDestinationError::Metadata(MetadataDestinationError::Capacity {
                field: "descriptor dimensions",
                capacity: 0,
                ..
            })
        ));
        assert_eq!(metadata.completed(), Some(expected.converted()));
        assert!(raw.iter().any(|b| *b != 0));
        assert_eq!(metadata.names.as_ptr(), names);
        assert_eq!(metadata.base.name.as_ptr(), base);
        assert!(metadata.used);
    }
}

#[test]
fn prepared_plan_validation_runs_after_reopen_and_before_payload_read() {
    let (dir, checkpoint) = file(GgmlType::F32, Endian::Little);
    let mut materializer = checkpoint.materializer();
    let invalid = TensorSelection::Range {
        axis: 9,
        start: 0,
        end: 1,
    };
    let source = materializer.metadata_source("prefix.α.weight").unwrap();
    let d = source.tensor.descriptor.clone();
    let mut metadata =
        PreparedTensorMetadata::prepare(source, MetadataSelection::Axis(&invalid)).unwrap();
    assert!(materializer.open_shard_path().is_none());
    let mut raw = vec![0x5a; d.byte_len as usize];
    let mut conversion = PreparedConversion::prepare(d.clone(), Endian::Little).unwrap();
    let expected = TensorSelectionPlan::new(&d, invalid.clone()).unwrap_err();
    let error = materializer
        .converted_tensor_with_metadata(
            "prefix.α.weight",
            MetadataSelection::Axis(&invalid),
            &mut raw,
            &mut conversion,
            &mut metadata,
        )
        .unwrap_err();
    let ReadDestinationError::Gguf(error) = error else {
        panic!("old plan error")
    };
    assert_eq!(error.to_string(), expected.to_string());
    assert!(materializer.open_shard_path().is_some());
    assert!(raw.iter().all(|b| *b == 0x5a));
    let mut materializer = checkpoint.materializer();
    let source = materializer.metadata_source("prefix.α.weight").unwrap();
    let mut metadata =
        PreparedTensorMetadata::prepare(source, MetadataSelection::Axis(&invalid)).unwrap();
    std::fs::remove_file(dir.path().join("metadata.gguf")).unwrap();
    assert!(matches!(
        materializer.converted_tensor_with_metadata(
            "prefix.α.weight",
            MetadataSelection::Axis(&invalid),
            &mut raw,
            &mut conversion,
            &mut metadata
        ),
        Err(ReadDestinationError::Gguf(Error::Shard { .. }))
    ));
    assert!(!metadata.used);
    assert!(metadata.completed.is_none());
}

#[test]
fn shared_metadata_retains_exact_catalog_row_after_materializer_drop() {
    let (_directory, checkpoint) = file(GgmlType::Q4_0, Endian::Little);
    let expected = &checkpoint.shards[0].tensors[0];
    let tensor = std::ptr::from_ref(expected);
    let descriptor_name = expected.descriptor.name.as_ptr();
    let names = expected
        .outputs
        .iter()
        .map(|o| o.name.as_ptr())
        .collect::<Vec<_>>();
    let ordinary = checkpoint.materializer();
    assert!(matches!(
        ordinary.checkpoint,
        MaterializerCheckpoint::Owned(_)
    ));
    assert!(ordinary
        .shared_metadata_source("prefix.α.weight")
        .unwrap()
        .is_none());
    let materializer = checkpoint.into_shared_materializer();
    let owner = materializer
        .shared_metadata_source("prefix.α.weight")
        .unwrap()
        .unwrap();
    assert_eq!(std::ptr::from_ref(owner.source().tensor), tensor);
    let other = owner.clone();
    drop(materializer);
    drop(owner);
    let source = other.source();
    assert_eq!(std::ptr::from_ref(source.tensor), tensor);
    assert_eq!(source.tensor.descriptor.name.as_ptr(), descriptor_name);
    assert_eq!(
        source
            .tensor
            .outputs
            .iter()
            .map(|o| o.name.as_ptr())
            .collect::<Vec<_>>(),
        names
    );
    assert_eq!(source.tensor.outputs.len(), 3);
}
