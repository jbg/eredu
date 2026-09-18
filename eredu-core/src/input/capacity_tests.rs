use super::*;
use std::mem::size_of;

fn tensor(dtype: TensorDtype, dims: &[usize], spare: usize) -> InputTensorIdentity {
    let mut shape = Vec::with_capacity(dims.len() + spare);
    shape.extend_from_slice(dims);
    InputTensorIdentity::new(dtype, shape).unwrap()
}

fn image(reverse: bool) -> InputPartDescriptor {
    let grid = (
        InputMetadataKey::PatchGrid,
        tensor(TensorDtype::I32, &[1, 3], 0),
    );
    let positions = (
        InputMetadataKey::PatchPositions,
        tensor(TensorDtype::I32, &[7, 2], 0),
    );
    InputPartDescriptor::new_with_extents(
        InputModality::Image,
        InputPayloadKind::Tensor,
        tensor(TensorDtype::F32, &[7, 16], 0),
        if reverse {
            [positions, grid]
        } else {
            [grid, positions]
        },
        [InputExtent::PatchGrid {
            time: 1,
            height: 2,
            width: 4,
        }],
    )
    .unwrap()
}

#[test]
fn fixed_entries_preserve_sorted_lookup_wire_and_legacy_map_serde() {
    let unordered = PreparedInputIdentity::new(vec![image(true)]).unwrap();
    let ordered = PreparedInputIdentity::new(vec![image(false)]).unwrap();
    assert_eq!(unordered, ordered);
    assert_eq!(
        unordered.encode_words().unwrap(),
        ordered.encode_words().unwrap()
    );
    let map = unordered.parts()[0].metadata();
    assert_eq!(map.len(), 2);
    assert!(!map.is_empty());
    assert_eq!(
        map.keys().copied().collect::<Vec<_>>(),
        [
            InputMetadataKey::PatchGrid,
            InputMetadataKey::PatchPositions
        ]
    );
    assert_eq!(map.iter().len(), 2);
    assert_eq!(map.values().len(), 2);
    assert_eq!(map[&InputMetadataKey::PatchPositions].shape(), [7, 2]);
    assert!(map.contains_key(&InputMetadataKey::PatchGrid));
    assert!(map.get(&InputMetadataKey::AudioMask).is_none());

    // The wire representation contains two serialized BTreeMaps. Construct that
    // representation independently rather than comparing a serializer to itself.
    #[derive(Serialize)]
    struct LegacyPart<'a> {
        modality: InputModality,
        payload_kind: InputPayloadKind,
        payload: &'a InputTensorIdentity,
        metadata: BTreeMap<InputMetadataKey, &'a InputTensorIdentity>,
        extents: BTreeMap<u32, InputExtent>,
    }
    #[derive(Serialize)]
    struct Legacy<'a> {
        parts: Vec<LegacyPart<'a>>,
    }
    let part = &unordered.parts()[0];
    let legacy = Legacy {
        parts: vec![LegacyPart {
            modality: part.modality(),
            payload_kind: part.payload_kind(),
            payload: part.payload(),
            metadata: part.metadata().iter().map(|(k, v)| (*k, v)).collect(),
            extents: part
                .extents()
                .map(|extent| (extent.key(), extent))
                .collect(),
        }],
    };
    let encoded = serde_json::to_string(&legacy).unwrap();
    assert_eq!(serde_json::to_string(&unordered).unwrap(), encoded);
    let restored: PreparedInputIdentity = serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored, unordered);
    assert_eq!(
        PreparedInputIdentity::decode_words(&restored.encode_words().unwrap()).unwrap(),
        unordered
    );
}

#[test]
fn capacity_includes_spare_shapes_parts_and_encoded_names_once() {
    let mut encoded = String::with_capacity(97);
    encoded.push_str("packed-q4");
    let name_capacity = encoded.capacity();
    let payload = tensor(TensorDtype::Encoded(encoded), &[1, 7, 16], 9);
    let payload_heap = payload.shape.capacity() * size_of::<usize>() + name_capacity;
    assert_eq!(
        payload.capacity_bytes(),
        Some((size_of::<InputTensorIdentity>() + payload_heap) as u64)
    );
    let grid = tensor(TensorDtype::I32, &[1, 3], 5);
    let grid_heap = grid.shape.capacity() * size_of::<usize>();
    let part = InputPartDescriptor::new_with_extents(
        InputModality::Video,
        InputPayloadKind::Tensor,
        payload,
        [(InputMetadataKey::PatchGrid, grid)],
        [InputExtent::PatchGrid {
            time: 1,
            height: 2,
            width: 4,
        }],
    )
    .unwrap();
    let part_heap = payload_heap
        + grid_heap
        + size_of::<(InputMetadataKey, InputTensorIdentity)>()
        + size_of::<(u32, InputExtent)>();
    assert_eq!(
        part.capacity_bytes(),
        Some((size_of::<InputPartDescriptor>() + part_heap) as u64)
    );
    let mut parts = Vec::with_capacity(13);
    parts.push(part);
    let parts_capacity = parts.capacity();
    let identity = PreparedInputIdentity::new(parts).unwrap();
    let expected = size_of::<PreparedInputIdentity>()
        + parts_capacity * size_of::<InputPartDescriptor>()
        + part_heap;
    let before = (
        identity.parts.as_ptr(),
        identity.parts[0].payload.shape.as_ptr(),
    );
    for _ in 0..3 {
        assert_eq!(identity.capacity_bytes(), Some(expected as u64));
    }
    assert_eq!(
        before,
        (
            identity.parts.as_ptr(),
            identity.parts[0].payload.shape.as_ptr()
        )
    );
    assert!(identity.capacity_bytes().unwrap() > identity.logical_metadata_bytes().unwrap());
    let copy = identity.clone();
    assert_eq!(copy, identity);
    assert!(copy.capacity_bytes().unwrap() <= expected as u64);
}

#[test]
fn fixed_entries_keep_duplicate_and_modality_errors_before_publication() {
    let payload = || tensor(TensorDtype::F32, &[2, 8], 0);
    let value = || tensor(TensorDtype::I32, &[1, 3], 0);
    assert_eq!(
        InputPartDescriptor::new(
            InputModality::Image,
            InputPayloadKind::Tensor,
            payload(),
            [
                (InputMetadataKey::PatchGrid, value()),
                (InputMetadataKey::PatchGrid, value()),
            ]
        )
        .unwrap_err(),
        PreparedInputError::DuplicateMetadata {
            key: InputMetadataKey::PatchGrid
        }
    );
    let extent = InputExtent::PatchGrid {
        time: 1,
        height: 2,
        width: 4,
    };
    assert_eq!(
        InputPartDescriptor::new_with_extents(
            InputModality::Image,
            InputPayloadKind::Tensor,
            payload(),
            [],
            [extent, extent]
        )
        .unwrap_err(),
        PreparedInputError::DuplicateExtent { extent }
    );
    assert_eq!(
        InputPartDescriptor::new(
            InputModality::Audio,
            InputPayloadKind::Tensor,
            payload(),
            [(InputMetadataKey::PatchGrid, value())]
        )
        .unwrap_err(),
        PreparedInputError::IncompatibleMetadata {
            modality: InputModality::Audio,
            key: InputMetadataKey::PatchGrid
        }
    );
    assert_eq!(
        InputPartDescriptor::new_with_extents(
            InputModality::Text,
            InputPayloadKind::TokenIds,
            payload(),
            [],
            [extent]
        )
        .unwrap_err(),
        PreparedInputError::IncompatibleExtent {
            modality: InputModality::Text,
            extent
        }
    );
}
