use super::*;
use eredu_checkpoint::{
    gguf_store::GgufLease,
    store::{CheckpointLease, ReadPolicy, TensorReadRequest, TensorSelection},
};
use eredu_gguf::{ConvertedTensor, DenseDtype};
use std::path::Path;

fn prepared(path: &Path, captured: bool) -> PreparedModelSources {
    let inspection = if captured {
        crate::configuration::inspect_artifact_with_prepared_gguf_headers(path)
    } else {
        crate::configuration::inspect_artifact(path)
    }
    .unwrap();
    let selected = crate::select_preparation(
        &inspection,
        &eredu_runtime::NormalizedLoadRequest::default(),
        &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
    )
    .unwrap();
    let plan = eredu_core::plan_model_preparation(
        inspection,
        eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::FullyResident),
        selected.session_capabilities(),
    )
    .unwrap();
    prepare_model_sources(plan, selected).unwrap()
}

fn lease(source: &RetainedCheckpointSource, key: &str) -> Box<GgufLease> {
    let CheckpointLease::Gguf(lease) = source
        .acquire_lease(TensorReadRequest {
            key: key.into(),
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap()
    else {
        panic!("GGUF lease expected")
    };
    lease
}

fn values(tensor: eredu_gguf::ConvertedCheckpointTensor, value: f32) {
    let ConvertedTensor::Dense(tensor) = tensor.into_converted() else {
        panic!("dense F32 expected")
    };
    assert_eq!(tensor.dtype, DenseDtype::F32);
    assert!(!tensor.data.is_empty());
    assert_eq!(tensor.data.len() % 4, 0);
    for bytes in tensor.data.chunks_exact(4) {
        assert_eq!(f32::from_le_bytes(bytes.try_into().unwrap()), value);
    }
}

// Keep the established complete Gemma4 fixture and change only actual F32 payloads.
fn payload(path: &Path, value: f32) -> Vec<u8> {
    let checkpoint = eredu_gguf::Checkpoint::open(path).unwrap();
    let mut bytes = std::fs::read(path).unwrap();
    for tensor in checkpoint.tensors() {
        let descriptor = tensor.descriptor();
        assert_eq!(descriptor.ggml_type, eredu_gguf::GgmlType::F32);
        let start = usize::try_from(descriptor.data_offset).unwrap();
        let length = usize::try_from(descriptor.byte_len).unwrap();
        for element in bytes[start..start + length].chunks_exact_mut(4) {
            element.copy_from_slice(&value.to_le_bytes());
        }
    }
    std::fs::write(path, &bytes).unwrap();
    bytes
}

fn changed_header(original: &[u8]) -> Vec<u8> {
    let key = b"general.architecture";
    let key_start = original
        .windows(key.len())
        .position(|bytes| bytes == key)
        .unwrap();
    // GGUF string value: metadata type u32 followed by string length u64.
    let value_start = key_start + key.len() + 4 + 8;
    let mut changed = original.to_vec();
    changed[value_start] = b'X';
    assert_ne!(original[value_start], changed[value_start]);
    changed
}

#[test]
fn captured_selection_retains_exact_lazy_primary_companion_and_projection_owners() {
    let root = tests::gemma4_gguf_fixture();
    let path = root.path().join("model.gguf");
    payload(&path, 1.25);
    payload(&root.path().join("mmproj.gguf"), -3.5);
    let sources = prepared(&path, true);
    let ordinary = prepared(&path, false);
    assert!(!sources.graph().source_identity().is_resolved());
    assert_eq!(
        sources.primary().source_keys(),
        ordinary.primary().source_keys()
    );
    assert_eq!(
        sources.target().source_keys(),
        ordinary.target().source_keys()
    );
    let role = GgufCompanionRole::MediaProjector;
    assert_eq!(
        sources.companion(&role).unwrap().source_keys(),
        ordinary.companion(&role).unwrap().source_keys()
    );
    let primary_storage = sources.primary().source_storage().unwrap().unwrap();
    let ordinary_storage = ordinary.primary().source_storage().unwrap().unwrap();
    assert!(primary_storage.bytes().unwrap() > ordinary_storage.bytes().unwrap());
    assert_eq!(
        sources
            .inference_blueprint()
            .source_storage()
            .unwrap()
            .unwrap()
            .owner_count(),
        2
    );
    for source in [
        sources.primary(),
        sources.companion(&role).unwrap(),
        sources.target(),
    ] {
        let diagnostics = source.source_diagnostics().unwrap();
        assert_eq!(diagnostics.physical_reads, 0);
        assert_eq!(diagnostics.physical_read_bytes, 0);
        assert!(diagnostics.payload_shard_paths.is_empty());
    }
    let clone = sources.clone();
    assert!(sources.same_selected_sources(&clone));
    let admitted = sources.inspection().validated_gguf().unwrap();
    for (a, b) in admitted
        .checkpoint()
        .shards()
        .iter()
        .zip(clone.inspection().gguf_checkpoint().unwrap().shards())
    {
        assert!(std::ptr::eq(
            a.prepared_header().unwrap(),
            b.prepared_header().unwrap()
        ));
    }
    let key = sources.primary().source_keys().into_iter().next().unwrap();
    let primary = lease(sources.primary(), &key);
    let companion_key = sources
        .companion(&role)
        .unwrap()
        .source_keys()
        .into_iter()
        .next()
        .unwrap();
    let companion = lease(sources.companion(&role).unwrap(), &companion_key);
    drop(sources);
    drop(clone);
    // The leases still own the selected captured stores after every graph view dies.
    values(primary.materialize_portable().unwrap(), 1.25);
    values(companion.materialize_portable().unwrap(), -3.5);
}

#[test]
fn captured_primary_and_companion_late_refusals_retain_source_through_restore_and_view_drop() {
    let root = tests::gemma4_gguf_fixture();
    let paths = [
        root.path().join("model.gguf"),
        root.path().join("mmproj.gguf"),
    ];
    let original = [payload(&paths[0], 1.25), payload(&paths[1], -3.5)];
    let sources = prepared(&paths[0], true);
    let ordinary = prepared(&paths[0], false);
    let role = GgufCompanionRole::MediaProjector;
    let mut failures = Vec::new();
    for (index, (captured_source, ordinary_source)) in [
        (sources.primary(), ordinary.primary()),
        (
            sources.companion(&role).unwrap(),
            ordinary.companion(&role).unwrap(),
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let key = captured_source.source_keys().into_iter().next().unwrap();
        // Mutate before this store's first cold read; no cache-hit revalidation is assumed.
        std::fs::write(&paths[index], changed_header(&original[index])).unwrap();
        let actual = *lease(captured_source, &key);
        let identity = actual.identity().clone();
        let failure = actual
            .prepare_portable_read()
            .unwrap()
            .prepare_conversion()
            .unwrap()
            .prepare_result_metadata()
            .unwrap()
            .materialize()
            .unwrap_err();
        match failure.store_error().unwrap() {
            StoreError::GgufPreparedHeaderChanged {
                key: actual,
                source,
            } => {
                assert_eq!(actual, &key);
                assert!(source.prepared_header_change().is_some());
            }
            other => panic!("unexpected refusal: {other:?}"),
        }
        assert_eq!(failure.lease().identity(), &identity);
        values(
            lease(ordinary_source, &key).materialize_portable().unwrap(),
            if index == 0 { 1.25 } else { -3.5 },
        );
        std::fs::write(&paths[index], &original[index]).unwrap();
        payload(&paths[index], 19.0 + index as f32);
        values(
            failure.lease().materialize_portable().unwrap(),
            19.0 + index as f32,
        );
        failures.push(failure);
    }
    drop(sources);
    drop(ordinary);
    // Old typed failures retain their own real leases independently of graph owners.
    for (index, failure) in failures.iter().enumerate() {
        assert!(matches!(
            failure.store_error(),
            Some(StoreError::GgufPreparedHeaderChanged { .. })
        ));
        values(
            failure.lease().materialize_portable().unwrap(),
            19.0 + index as f32,
        );
    }
}

#[test]
fn captured_inspection_cannot_substitute_for_another_selection_of_identical_bytes() {
    let root = tests::gemma4_gguf_fixture();
    let path = root.path().join("model.gguf");
    let captured =
        crate::configuration::inspect_artifact_with_prepared_gguf_headers(&path).unwrap();
    for replacement in [
        crate::configuration::inspect_artifact(&path).unwrap(),
        crate::configuration::inspect_artifact_with_prepared_gguf_headers(&path).unwrap(),
    ] {
        let selected = crate::select_preparation(
            &captured,
            &eredu_runtime::NormalizedLoadRequest::default(),
            &crate::preparation_selection::tests::BoundedIndependentAdapter::default(),
        )
        .unwrap();
        assert!(!captured
            .admission_token()
            .same_admission(&replacement.admission_token()));
        let plan = eredu_core::plan_model_preparation(
            replacement,
            eredu_core::PreparationPolicy::new(None, eredu_core::ResidencyRequest::FullyResident),
            selected.session_capabilities(),
        )
        .unwrap();
        assert!(matches!(
            prepare_model_sources(plan, selected),
            Err(PreparedModelSourcesError::InvalidSelection(_))
        ));
    }
}

#[test]
fn retained_gguf_roles_select_cold_reader_index_inventory_but_inspection_stays_ordinary() {
    use eredu_checkpoint::gguf_store::{open_prepared_gguf_source, GgufWeightStore};
    let root = tests::gemma4_gguf_fixture();
    let sources = prepared(&root.path().join("model.gguf"), true);
    let architecture = sources.inspection().architecture_plan();
    let primary_plan = architecture.gguf_plan().unwrap();
    let projector = architecture.gguf_media_projector().unwrap();
    let validated = sources.inspection().validated_gguf().unwrap();
    let companion = validated
        .companion(&GgufCompanionRole::MediaProjector)
        .unwrap();
    let maximum = sources.selected().text_realization().max_cached_shards();
    for (actual, checkpoint, plan, mapping) in [
        (
            sources.primary(),
            validated.checkpoint(),
            primary_plan.checkpoint(),
            projector.primary_tensor_mapping(),
        ),
        (
            sources
                .companion(&GgufCompanionRole::MediaProjector)
                .unwrap(),
            companion.checkpoint(),
            projector.checkpoint(),
            projector.tensor_mapping(),
        ),
    ] {
        let expected = open_prepared_gguf_source_with_reader_buffers(
            checkpoint.clone(),
            plan,
            mapping,
            maximum,
        )
        .unwrap();
        let ordinary =
            open_prepared_gguf_source(checkpoint.clone(), plan, mapping, maximum).unwrap();
        assert!(expected.prepared_reader_storage_bytes().unwrap() > 0);
        assert!(expected.prepared_materializer_storage_bytes().unwrap() > 0);
        assert!(ordinary.prepared_reader_storage_bytes().is_none());
        assert!(ordinary.prepared_materializer_storage_bytes().is_none());
        let inventory = actual.source_storage().unwrap().unwrap();
        assert_eq!(
            inventory.bytes().unwrap(),
            expected.source_storage().unwrap().unwrap().bytes().unwrap()
        );
        assert!(
            inventory.bytes().unwrap()
                > ordinary.source_storage().unwrap().unwrap().bytes().unwrap()
        );
        assert_eq!(actual.source_diagnostics().unwrap().physical_reads, 0);
    }
    let inventory = sources.source_storage().unwrap().unwrap();
    assert_eq!(inventory.owner_count(), 2);
    assert_eq!(
        inventory.bytes().unwrap(),
        sources
            .primary()
            .source_storage()
            .unwrap()
            .unwrap()
            .bytes()
            .unwrap()
            + sources
                .companion(&GgufCompanionRole::MediaProjector)
                .unwrap()
                .source_storage()
                .unwrap()
                .unwrap()
                .bytes()
                .unwrap()
    );

    // The actual metadata-only recipe constructor remains on its ordinary route.
    let inspection_only =
        crate::replicated_text::inspection_recipe_source(sources.inspection()).unwrap();
    let expected = GgufWeightStore::builder()
        .add_checkpoint(
            validated.checkpoint().clone(),
            primary_plan.checkpoint(),
            projector.primary_tensor_mapping(),
        )
        .unwrap()
        .add_checkpoint(
            companion.checkpoint().clone(),
            projector.checkpoint(),
            projector.tensor_mapping(),
        )
        .unwrap()
        .build()
        .unwrap();
    assert!(expected.prepared_reader_storage_bytes().is_none());
    assert!(expected.prepared_materializer_storage_bytes().is_none());
    assert_eq!(
        inspection_only
            .source_storage()
            .unwrap()
            .unwrap()
            .bytes()
            .unwrap(),
        expected.source_storage().unwrap().unwrap().bytes().unwrap()
    );
    assert_eq!(
        inspection_only.source_diagnostics().unwrap().physical_reads,
        0
    );
}
