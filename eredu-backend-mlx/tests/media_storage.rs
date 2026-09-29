//! Scoped native storage observations of the actual owned Flash-Next request.
use eredu_architectures::{
    qwen::{ingress::PreparedInputTensorRole as IngressRole, vision::VisionConfigSource},
    qwen4_exp::{
        conditional::prefill::{MediaPrefillRequest, MediaPrefillTensorRole as Role},
        config::MediaTokens,
        media::{MediaAdmissionConfig, PreparedMediaTensorRole as PreparedRole},
    },
};
use eredu_backend_mlx::{backend::runtime::media::input::MlxTensorInputInspector, MlxTensor};
use eredu_core::{InputMetadataKey as Key, InputModality as Modality};
use eredu_nn::{tensor_storage::TensorStorageSnapshot, Index, Tensor};
use eredu_runtime::{
    input::PreparedInputTensorRole as OriginalRole, PreparedInputInspector, PreparedInputPart,
    PreparedInputPayload as Payload, PreparedModelInput,
};
use safemlx::{Array, Device, DeviceType, Stream};

fn policy() -> MediaAdmissionConfig {
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    let vision = serde_json::from_value::<VisionConfigSource>(serde_json::json!({
        "depth": 2, "hidden_size": 8, "intermediate_size": 12,
        "num_heads": 2, "num_position_embeddings": 16, "in_channels": 3,
        "patch_size": 2, "spatial_merge_size": 2, "temporal_patch_size": 2,
        "out_hidden_size": 32, "hidden_act": "gelu_pytorch_tanh",
        "deepstack_visual_indexes": []
    }))
    .unwrap()
    .normalize_qwen3_vl()
    .unwrap();
    // The returned geometry is source-free. No checkpoint or tower parameters
    // survive this helper; observation cannot reopen its temporary artifacts.
    MediaAdmissionConfig::new(
        fixture.gguf.spec(),
        &vision,
        &MediaTokens {
            image: 12,
            video: 13,
            start: 14,
            end: 15,
        },
    )
    .unwrap()
}

fn text(ids: &[u32]) -> PreparedInputPart<MlxTensor> {
    PreparedInputPart::new(
        Modality::Text,
        Payload::TokenIds(MlxTensor::from_array(Array::from_slice(
            ids,
            &[1, ids.len() as i32],
        ))),
        [],
    )
    .unwrap()
}

fn media(modality: Modality, pixels: MlxTensor, grid: [i32; 3]) -> PreparedInputPart<MlxTensor> {
    PreparedInputPart::new(
        modality,
        Payload::Tensor(pixels),
        [(
            Key::PatchGrid,
            MlxTensor::from_array(Array::from_slice(&grid, &[1, 3])),
        )],
    )
    .unwrap()
}

fn find(snapshot: &TensorStorageSnapshot<Role>, role: Role) -> usize {
    snapshot
        .values
        .iter()
        .position(|value| value.role == role)
        .unwrap()
}

fn prepared(role: IngressRole) -> Role {
    Role::Prepared(PreparedRole::Ingress(role))
}

#[test]
fn media_request_storage_keeps_original_views_metadata_and_lazy_concatenation() {
    let policy = policy();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let source_values: Vec<f32> = (0..32 * 24).map(|i| (i as f32 - 127.) / 256.).collect();
    let backing = MlxTensor::from_array(Array::from_slice(&source_values, &[32, 24]));
    let image = backing
        .index(&[Index::Range(4, 20), Index::Full], &stream)
        .unwrap();
    image.as_array().evaluated().unwrap();
    let expected_image = image
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    let video = MlxTensor::from_array(Array::from_slice(&source_values[..8 * 24], &[8, 24]));
    let original = PreparedModelInput::new(
        vec![
            text(&[3, 14]),
            media(Modality::Image, image.clone(), [1, 4, 4]),
            text(&[15, 4, 14]),
            media(Modality::Video, video, [1, 4, 2]),
            text(&[15, 6]),
        ],
        |value| MlxTensorInputInspector.identity(value),
    )
    .unwrap();
    let admitted = policy.admit(&original, &MlxTensorInputInspector).unwrap();
    let request = MediaPrefillRequest::new(admitted.into_composite(), &stream).unwrap();
    let cloned = request.clone();
    let snapshot = request.inspect_storage(None).unwrap();
    snapshot.validate().unwrap();
    assert_eq!(snapshot.values.len(), 16);
    let original_image = find(&snapshot, Role::Original(OriginalRole::Payload { part: 1 }));
    let original_video = find(&snapshot, Role::Original(OriginalRole::Payload { part: 3 }));
    let patches = find(&snapshot, prepared(IngressRole::Pixels));
    assert_eq!(
        snapshot.values[original_image].logical_bytes,
        Some(16 * 24 * 4)
    );
    assert_eq!(
        snapshot.values[original_video].logical_bytes,
        Some(8 * 24 * 4)
    );
    assert_eq!(snapshot.values[patches].logical_bytes, Some(24 * 24 * 4));
    assert_eq!(snapshot.storage.values[patches], None);
    let image_backing = snapshot.storage.values[original_image].unwrap();
    assert!(
        snapshot.storage.backings[image_backing]
            .allocator_capacity_bytes
            .unwrap()
            >= 32 * 24 * 4
    );
    for part in [0, 2, 4] {
        let source = find(&snapshot, Role::Original(OriginalRole::Payload { part }));
        let ingress = find(&snapshot, prepared(IngressRole::Tokens(part)));
        assert!(snapshot.storage.values[source].is_some());
        assert_eq!(
            snapshot.storage.values[source],
            snapshot.storage.values[ingress]
        );
    }
    for part in [1, 3] {
        let grid = find(
            &snapshot,
            Role::Original(OriginalRole::Metadata {
                part,
                key: Key::PatchGrid,
            }),
        );
        assert_eq!(snapshot.values[grid].logical_bytes, Some(12));
        assert!(snapshot.storage.values[grid].is_some());
    }
    assert_eq!(request.inspect_storage(None).unwrap(), snapshot);
    assert_eq!(
        image.as_array().evaluated().unwrap().as_slice::<f32>(),
        expected_image
    );
    drop((original, backing, image, request));
    assert_eq!(cloned.inspect_storage(None).unwrap(), snapshot);

    // A supplied encoder continuation is observed in the same namespace, with
    // no claim that inspection completes or validates an encoder computation.
    let source = Array::from_slice(&vec![0.25f32; 6 * 32], &[1, 6, 32]);
    let projection = MlxTensor::from_array(source.add(&source, &stream).unwrap());
    let before = cloned.inspect_storage(Some(&projection)).unwrap();
    let output = find(&before, Role::EncoderOutput);
    assert_eq!(before.values[output].logical_bytes, Some(6 * 32 * 4));
    assert_eq!(before.storage.values[output], None);
    assert!(!projection.as_array().is_available().unwrap());
    assert!(projection
        .as_array()
        .evaluated()
        .unwrap()
        .as_slice::<f32>()
        .iter()
        .all(|&v| v == 0.5));
    let after = cloned.inspect_storage(Some(&projection)).unwrap();
    assert!(after.storage.values[output].is_some());
    assert_eq!(after.storage.values[patches], None);
    drop((cloned, projection, source, stream));
    after.validate().unwrap();
    assert_eq!(after.values.len(), 17);
}

#[test]
fn projected_request_storage_preserves_embeddings_and_explicit_original_id_aliases() {
    let policy = policy();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let original = PreparedModelInput::new(
        vec![PreparedInputPart::new(
            Modality::Text,
            Payload::Embeddings(MlxTensor::from_array(Array::from_slice(
                &vec![0.375f32; 3 * 32],
                &[1, 3, 32],
            ))),
            [(
                Key::OriginalTokenIds,
                MlxTensor::from_array(Array::from_slice(&[3u32, 4, 5], &[1, 3])),
            )],
        )
        .unwrap()],
        |value| MlxTensorInputInspector.identity(value),
    )
    .unwrap();
    let request = MediaPrefillRequest::new(
        policy
            .admit(&original, &MlxTensorInputInspector)
            .unwrap()
            .into_composite(),
        &stream,
    )
    .unwrap();
    let snapshot = request.inspect_storage(None).unwrap();
    let source_embeddings = find(&snapshot, Role::Original(OriginalRole::Payload { part: 0 }));
    let prepared_embeddings = find(&snapshot, prepared(IngressRole::Projected(0)));
    let source_ids = find(
        &snapshot,
        Role::Original(OriginalRole::Metadata {
            part: 0,
            key: Key::OriginalTokenIds,
        }),
    );
    let prepared_ids = find(&snapshot, prepared(IngressRole::Tokens(0)));
    assert_eq!(snapshot.values.len(), 7);
    assert_eq!(
        snapshot.values[source_embeddings].logical_bytes,
        Some(3 * 32 * 4)
    );
    assert_eq!(snapshot.values[source_ids].logical_bytes, Some(3 * 4));
    assert!(snapshot.storage.values[source_embeddings].is_some());
    assert!(snapshot.storage.values[source_ids].is_some());
    assert_eq!(
        snapshot.storage.values[source_embeddings],
        snapshot.storage.values[prepared_embeddings]
    );
    assert_eq!(
        snapshot.storage.values[source_ids],
        snapshot.storage.values[prepared_ids]
    );
    assert!(!snapshot
        .values
        .iter()
        .any(|value| value.role == prepared(IngressRole::Pixels)));
    let clone = request.clone();
    drop((original, request));
    assert_eq!(clone.inspect_storage(None).unwrap(), snapshot);
    drop((clone, stream));
    snapshot.validate().unwrap();
}
