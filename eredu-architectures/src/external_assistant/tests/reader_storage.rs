use super::*;
use eredu_checkpoint::{
    gguf_store::GgufWeightStore,
    store::{CheckpointLease, ReadPolicy, TensorReadRequest, TensorSelection},
};
use eredu_gguf::{ConvertedTensor, GgmlType, TensorInput, Writer};

#[test]
fn selected_gguf_assistant_keeps_one_actual_reader_index_source_through_preparation() {
    let config = gemma4::AssistantConfig::from_json(GEMMA_ASSISTANT.as_bytes()).unwrap();
    let plan = gemma4::assistant_gguf_plan(&config).unwrap();
    let tensors = plan
        .common_tensors
        .iter()
        .chain(
            plan.layout_groups
                .iter()
                .filter(|g| g.required)
                .filter_map(|g| g.variants.first())
                .flat_map(|v| &v.tensors),
        )
        .filter(|t| t.requirement == TensorRequirement::Required)
        .map(|t| {
            (
                t.key.clone(),
                t.shape.iter().rev().map(|d| *d as u64).collect::<Vec<_>>(),
                (0..t.shape.iter().product::<usize>())
                    .flat_map(|i| {
                        if i % 2 == 0 {
                            1.25f32.to_le_bytes()
                        } else {
                            (-3.5f32).to_le_bytes()
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let inputs = tensors
        .iter()
        .map(|(name, dimensions, data)| TensorInput {
            name,
            dimensions,
            data,
            ggml_type: GgmlType::F32,
        })
        .collect::<Vec<_>>();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("assistant.gguf");
    Writer::default()
        .write(
            std::fs::File::create(&path).unwrap(),
            &BTreeMap::new(),
            &inputs,
        )
        .unwrap();
    let checkpoint = Checkpoint::open_with_prepared_headers(&path).unwrap();
    let resolution = resolve_gguf_plan(&checkpoint, &plan).unwrap();
    let tensor_mapping = checkpoint
        .translated_outputs(gemma4::translate_assistant_gguf_weight_name)
        .unwrap();
    let expected = GgufWeightStore::builder()
        .max_cached_readers(2)
        .unwrap()
        .add_resolved_checkpoint(checkpoint.clone(), &resolution, &tensor_mapping)
        .unwrap()
        .build_with_prepared_reader_buffers()
        .unwrap();
    assert!(expected.prepared_reader_storage_bytes().unwrap() > 0);
    assert!(expected.prepared_materializer_storage_bytes().unwrap() > 0);
    let selected = select_external_materialization(
        PreparedExternalAssistant::<Gemma4AssistantArchitecture> {
            checkpoint: ExternalAssistantCheckpoint::Gguf {
                checkpoint,
                resolution,
                tensor_mapping,
            },
            config,
            _architecture: PhantomData,
        },
        None,
        2,
        &|descriptor, transforms| {
            (!transforms && descriptor.executable() == LinearFormat::Dense)
                .then_some(WeightLoweringKind::Direct)
        },
    )
    .unwrap();
    assert!(!selected.tasks.is_empty());
    let original = selected.prepared_source.as_ref().unwrap().clone();
    assert_eq!(
        original.source_storage().unwrap().unwrap().bytes().unwrap(),
        expected.source_storage().unwrap().unwrap().bytes().unwrap()
    );
    assert_eq!(original.source_diagnostics().unwrap().physical_reads, 0);
    let prepared = selected.prepare_source(2).unwrap();
    assert!(original.same_source(&prepared.source));
    let key = prepared.tasks[0].name().to_owned();
    let (source, _, _, _, _, _) = prepared.into_parts();
    drop(original);
    let CheckpointLease::Gguf(lease) = source
        .acquire_lease(TensorReadRequest {
            key,
            selection: TensorSelection::Full,
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap()
    else {
        panic!("GGUF lease")
    };
    drop(source);
    let ConvertedTensor::Dense(converted) = lease.materialize_portable().unwrap().into_converted()
    else {
        panic!("dense fixture")
    };
    assert!(!converted.data.is_empty());
    for (i, bytes) in converted.data.chunks_exact(4).enumerate() {
        assert_eq!(
            f32::from_le_bytes(bytes.try_into().unwrap()),
            if i % 2 == 0 { 1.25 } else { -3.5 }
        );
    }
}
