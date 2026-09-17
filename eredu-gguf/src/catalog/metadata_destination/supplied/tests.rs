use super::*;
use crate::catalog::metadata_destination::tests::{file, old};
use crate::supplied_storage::tests::{own, Provider};
use crate::GgmlType;
#[test]
fn supplied_catalog_names_and_selected_metadata_move_intact_and_match_old_driver() {
    for endian in [Endian::Little, Endian::Big] {
        for ty in [
            GgmlType::F32,
            GgmlType::Q8_0,
            GgmlType::Q4_0,
            GgmlType::MxFp4,
        ] {
            let (_dir, checkpoint) = file(ty, endian);
            let axis = TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            };
            let range = TensorSelection::Range {
                axis: 0,
                start: 1,
                end: 2,
            };
            let span = DenseTensorSpan::new(32, vec![1, 1, 32]).unwrap();
            for selection in [
                MetadataSelection::Full,
                MetadataSelection::Axis(&axis),
                MetadataSelection::Axis(&range),
                MetadataSelection::Span(&span),
            ] {
                let mut materializer = checkpoint.materializer();
                let mut ordinary = checkpoint.materializer();
                let expected = old(&mut ordinary, selection);
                let source = materializer.metadata_source("prefix.α.weight").unwrap();
                let descriptor = &source.tensor.descriptor;
                let mut provider = Provider::default();
                let physical =
                    crate::StoredPhysicalDescriptor::prepare(descriptor, selection, &mut provider)
                        .unwrap();
                let mut raw = vec![0; physical.descriptor().byte_len as usize];
                let conversion = crate::StoredConversion::prepare(
                    physical.into_descriptor(),
                    endian,
                    &mut provider,
                )
                .unwrap();
                let metadata =
                    StoredTensorMetadata::prepare(source, selection, &mut provider).unwrap();
                let name_pointers = metadata.names.iter().map(str::as_ptr).collect::<Vec<_>>();
                let selected = if matches!(selection, MetadataSelection::Full) {
                    metadata.base.view()
                } else {
                    metadata.final_descriptor.view()
                };
                let name = selected.name.as_ptr();
                let mut pair = StoredTensorPair::try_new(metadata, conversion).unwrap();
                materializer
                    .converted_tensor_with_stored_pair(
                        "prefix.α.weight",
                        selection,
                        &mut raw,
                        &mut pair,
                    )
                    .unwrap();
                let result = pair.into_result().unwrap();
                assert_eq!(result.descriptor.view(), expected.descriptor().view());
                assert_eq!(result.descriptor.view().name.as_ptr(), name);
                assert_eq!(
                    result.output_names.iter().collect::<Vec<_>>(),
                    expected
                        .output_names()
                        .iter()
                        .map(String::as_str)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    result
                        .output_names
                        .iter()
                        .map(str::as_ptr)
                        .collect::<Vec<_>>(),
                    name_pointers
                );
                assert!(raw.iter().any(|&b| b != 0));
                assert_eq!(own(result.converted), expected.into_converted());
            }
        }
    }
}
#[test]
fn late_supplied_metadata_refusal_keeps_completed_output_and_prevents_result_escape() {
    let (_dir, checkpoint) = file(GgmlType::Q4_0, Endian::Little);
    let selection = TensorSelection::Indices {
        axis: 0,
        indices: vec![1, 0, 1],
    };
    let selection = MetadataSelection::Axis(&selection);
    let mut materializer = checkpoint.materializer();
    let source = materializer.metadata_source("prefix.α.weight").unwrap();
    let mut provider = Provider::default();
    let physical = crate::StoredPhysicalDescriptor::prepare(
        &source.tensor.descriptor,
        selection,
        &mut provider,
    )
    .unwrap();
    let mut raw = vec![0; physical.descriptor().byte_len as usize];
    let conversion =
        crate::StoredConversion::prepare(physical.into_descriptor(), Endian::Little, &mut provider)
            .unwrap();
    let mut metadata = StoredTensorMetadata::prepare(source, selection, &mut provider).unwrap();
    // Remove only the final-descriptor destination; physical planning/conversion
    // still use their actual complete supplied owners and succeed first.
    metadata.final_descriptor.dimensions = StoredBuffer::default();
    let mut pair = StoredTensorPair::try_new(metadata, conversion).unwrap();
    let error = materializer
        .converted_tensor_with_stored_pair("prefix.α.weight", selection, &mut raw, &mut pair)
        .unwrap_err();
    assert!(matches!(
        error,
        ReadDestinationError::Metadata(MetadataDestinationError::Capacity { .. })
    ));
    assert!(pair.conversion_completed());
    assert!(raw.iter().any(|&b| b != 0));
    let pair = pair.into_result().unwrap_err();
    let (metadata, conversion) = pair.into_parts();
    assert!(conversion.completed());
    assert!(!metadata.completed);
}
#[test]
fn supplied_metadata_rejects_same_extent_foreign_selection_before_read() {
    let (_dir, checkpoint) = file(GgmlType::F32, Endian::Little);
    let bound = TensorSelection::Indices {
        axis: 0,
        indices: vec![0, 1],
    };
    let foreign = TensorSelection::Indices {
        axis: 0,
        indices: vec![1, 0],
    };
    let mut materializer = checkpoint.materializer();
    let source = materializer.metadata_source("prefix.α.weight").unwrap();
    let mut provider = Provider::default();
    let physical = crate::StoredPhysicalDescriptor::prepare(
        &source.tensor.descriptor,
        MetadataSelection::Axis(&bound),
        &mut provider,
    )
    .unwrap();
    let mut raw = vec![0x5a; physical.descriptor().byte_len as usize];
    let conversion =
        crate::StoredConversion::prepare(physical.into_descriptor(), Endian::Little, &mut provider)
            .unwrap();
    let metadata =
        StoredTensorMetadata::prepare(source, MetadataSelection::Axis(&bound), &mut provider)
            .unwrap();
    let mut pair = StoredTensorPair::try_new(metadata, conversion).unwrap();
    assert!(matches!(
        materializer.converted_tensor_with_stored_pair(
            "prefix.α.weight",
            MetadataSelection::Axis(&foreign),
            &mut raw,
            &mut pair
        ),
        Err(ReadDestinationError::Metadata(
            MetadataDestinationError::Binding
        ))
    ));
    assert!(!pair.conversion_completed());
    assert!(raw.iter().all(|&b| b == 0x5a));
}

fn prepared_pair(
    materializer: &TensorMaterializer,
    selection: MetadataSelection<'_>,
) -> (
    StoredTensorPair<crate::supplied_storage::tests::Family>,
    Vec<u8>,
) {
    let source = materializer.metadata_source("prefix.α.weight").unwrap();
    let mut provider = Provider::default();
    let physical = crate::StoredPhysicalDescriptor::prepare(
        &source.tensor.descriptor,
        selection,
        &mut provider,
    )
    .unwrap();
    let raw = vec![0; physical.descriptor().byte_len as usize];
    let conversion =
        crate::StoredConversion::prepare(physical.into_descriptor(), Endian::Little, &mut provider)
            .unwrap();
    let metadata = StoredTensorMetadata::prepare(source, selection, &mut provider).unwrap();
    (
        StoredTensorPair::try_new(metadata, conversion).unwrap(),
        raw,
    )
}

#[test]
fn completed_same_extent_different_selection_members_cannot_be_cross_paired() {
    let (_dir, checkpoint) = file(GgmlType::F32, Endian::Little);
    let a = TensorSelection::Indices {
        axis: 0,
        indices: vec![0, 1],
    };
    let b = TensorSelection::Indices {
        axis: 0,
        indices: vec![1, 0],
    };
    let mut materializer = checkpoint.materializer();
    let (mut first, mut first_raw) = prepared_pair(&materializer, MetadataSelection::Axis(&a));
    let (mut second, mut second_raw) = prepared_pair(&materializer, MetadataSelection::Axis(&b));
    materializer
        .converted_tensor_with_stored_pair(
            "prefix.α.weight",
            MetadataSelection::Axis(&a),
            &mut first_raw,
            &mut first,
        )
        .unwrap();
    materializer
        .converted_tensor_with_stored_pair(
            "prefix.α.weight",
            MetadataSelection::Axis(&b),
            &mut second_raw,
            &mut second,
        )
        .unwrap();
    assert_ne!(first_raw, second_raw);
    let (first_metadata, first_conversion) = first.into_parts();
    let (second_metadata, second_conversion) = second.into_parts();
    assert_eq!(
        first_conversion.descriptor(),
        second_conversion.descriptor()
    );
    let first_name = first_metadata.names.get(0).unwrap().as_ptr();
    let second_name = second_metadata.names.get(0).unwrap().as_ptr();
    let first_data = first_conversion.descriptor().name.as_ptr();
    let second_data = second_conversion.descriptor().name.as_ptr();
    let (first_metadata, second_conversion) =
        StoredTensorPair::try_new(first_metadata, second_conversion).unwrap_err();
    let (second_metadata, first_conversion) =
        StoredTensorPair::try_new(second_metadata, first_conversion).unwrap_err();
    assert_eq!(first_metadata.names.get(0).unwrap().as_ptr(), first_name);
    assert_eq!(second_metadata.names.get(0).unwrap().as_ptr(), second_name);
    assert_eq!(first_conversion.descriptor().name.as_ptr(), first_data);
    assert_eq!(second_conversion.descriptor().name.as_ptr(), second_data);
    assert!(first_conversion.completed() && second_conversion.completed());
}

#[test]
fn failed_pair_attempt_before_read_prevents_repairing_either_member() {
    let (_dir, checkpoint) = file(GgmlType::F32, Endian::Little);
    let mut materializer = checkpoint.materializer();
    let (mut pair, mut raw) = prepared_pair(&materializer, MetadataSelection::Full);
    assert!(materializer
        .converted_tensor_with_stored_pair("absent", MetadataSelection::Full, &mut raw, &mut pair)
        .is_err());
    assert!(materializer.open_shard_path().is_none());
    assert!(!pair.conversion_completed());
    assert!(raw.iter().all(|b| *b == 0));
    assert!(matches!(
        materializer.converted_tensor_with_stored_pair(
            "prefix.α.weight",
            MetadataSelection::Full,
            &mut raw,
            &mut pair
        ),
        Err(ReadDestinationError::Metadata(
            MetadataDestinationError::Used
        ))
    ));
    let (metadata, conversion) = pair.into_parts();
    let (fresh, _) = prepared_pair(&materializer, MetadataSelection::Full);
    let (fresh_metadata, fresh_conversion) = fresh.into_parts();
    let (metadata, fresh_conversion) =
        StoredTensorPair::try_new(metadata, fresh_conversion).unwrap_err();
    let (fresh_metadata, conversion) =
        StoredTensorPair::try_new(fresh_metadata, conversion).unwrap_err();
    assert!(metadata.pair_attempted);
    assert!(!fresh_metadata.pair_attempted);
    assert!(!conversion.completed() && !fresh_conversion.completed());
}
