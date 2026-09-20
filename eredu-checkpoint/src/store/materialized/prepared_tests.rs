use super::*;
use safetensors::tensor::{TensorView, serialize_to_file};

fn memory(rows: &[(&str, &[u8])]) -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors(
        rows.iter()
            .map(|(key, bytes)| ((*key).into(), Dtype::U8, vec![bytes.len()], bytes.to_vec())),
    )
    .unwrap()
}
fn snapshot(source: RetainedCheckpointSource) -> PreparedCheckpointSource {
    let catalog = source
        .source_keys()
        .into_iter()
        .map(|key| {
            let row = PreparedTensorSource {
                metadata: source.source_metadata(&key).unwrap(),
                provenance: source.source_provenance(&key).unwrap(),
            };
            (key, row)
        })
        .collect();
    PreparedCheckpointSource::new(source, catalog).unwrap()
}
fn overlay(source: RetainedCheckpointSource) -> RetainedCheckpointSource {
    Arc::new(MaterializedCheckpointSource::new(
        source,
        memory(&[("replacement", &[9, 8])]),
        BTreeSet::new(),
        BTreeSet::new(),
    ))
    .into()
}
fn fixtures() -> (tempfile::TempDir, [RetainedCheckpointSource; 2]) {
    let directory = tempfile::tempdir().unwrap();
    serialize_to_file(
        [
            ("a", TensorView::new(Dtype::U8, vec![2], &[3, 7]).unwrap()),
            ("b", TensorView::new(Dtype::U8, vec![2], &[11, 17]).unwrap()),
        ],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let file: RetainedCheckpointSource =
        Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap()).into();
    let memory: RetainedCheckpointSource =
        Arc::new(memory(&[("a", &[3, 7]), ("b", &[11, 17])])).into();
    (directory, [memory, file])
}

#[test]
fn prepared_overlay_reads_memory_and_files_without_a_recipe_cache() {
    let (_directory, sources) = fixtures();
    for source in sources {
        let overlay = overlay(source);
        assert!(overlay.recipe_cache().is_none());
        let root: RetainedCheckpointSource = Arc::new(snapshot(overlay)).into();
        let keys = [String::from("replacement"), String::from("replacement")];
        let replacement = MemoryEncodedReadPlan::from_source(&root, &keys)
            .unwrap()
            .unwrap()
            .construct(())
            .unwrap();
        let keys = [String::from("b"), String::from("a"), String::from("b")];
        let passthrough =
            if let Some(plan) = MemoryEncodedReadPlan::from_source(&root, &keys).unwrap() {
                plan.construct(()).unwrap()
            } else {
                SafetensorsEncodedReadPlan::from_source(&root, &keys)
                    .unwrap()
                    .unwrap()
                    .construct(())
                    .unwrap()
            };
        assert_eq!(root.source_diagnostics().unwrap().physical_reads, 0);
        drop(root);
        let mut bytes = [0; 4];
        replacement.read_into(&mut bytes).unwrap();
        assert_eq!(bytes, [9, 8, 9, 8]);
        let mut bytes = [0; 6];
        passthrough.read_into(&mut bytes).unwrap();
        assert_eq!(bytes, [11, 17, 3, 7, 11, 17]);
    }
}

#[test]
fn prepared_overlay_rejects_each_metadata_and_provenance_substitution() {
    let (_directory, sources) = fixtures();
    for (file, source) in sources.into_iter().enumerate() {
        for field in 0..11 {
            let mut prepared = snapshot(overlay(source.clone()));
            // Fault injection into the private admitted snapshot models a stale
            // catalog and proves that cacheless encoded reads validate every field.
            let row = prepared.catalog.get_mut("b").unwrap();
            match field {
                0 => row.metadata.name = "different".into(),
                1 => row.metadata.logical_shape = vec![1, 2],
                2 => row.metadata.physical_shape = vec![1, 2],
                3 => row.metadata.stored_dtype = StoredDtype::I8,
                4 => row.metadata.encoded_byte_len = 1,
                5 => row.metadata.backing_shard = Some(PathBuf::from("different")),
                6 => row.provenance.catalog_key = "different".into(),
                7 => row.provenance.physical_tensor = "different".into(),
                8 => row.provenance.output = "different".into(),
                9 => row.provenance.backing_shard = Some(PathBuf::from("different")),
                10 => {
                    row.provenance.source_encoding =
                        crate::SourceTensorEncoding::RecipeOutput(StoredDtype::U8)
                }
                _ => unreachable!(),
            }
            let root: RetainedCheckpointSource = Arc::new(prepared).into();
            let keys = [String::from("a"), String::from("b")];
            assert!(
                matches!(root.prepare_encoded_read(&keys), Err(StoreError::PreparedCatalogMismatch { key }) if key == "b")
            );
            if file == 0 {
                assert!(
                    matches!(
                        MemoryEncodedReadPlan::from_source(&root, &keys),
                        Err(MemoryEncodedReadRouteError::PreparedCatalogMismatch { index: 1 })
                    ),
                    "field {field}"
                );
            } else {
                assert!(
                    matches!(
                        SafetensorsEncodedReadPlan::from_source(&root, &keys).map_err(|e| e.kind()),
                        Err(
                            SafetensorsEncodedReadPlanErrorKind::PreparedCatalogMismatch {
                                index: 1
                            }
                        )
                    ),
                    "field {field}"
                );
            }
        }
    }
}

#[test]
fn prepared_overlay_keeps_missing_mixed_empty_and_outer_authorization_behavior() {
    let root: RetainedCheckpointSource = Arc::new(snapshot(overlay(
        Arc::new(memory(&[("a", &[3, 7])])).into(),
    )))
    .into();
    let mixed = [String::from("a"), String::from("replacement")];
    assert!(root.prepare_encoded_read(&mixed).unwrap().is_none());
    assert!(
        MemoryEncodedReadPlan::from_source(&root, &mixed)
            .unwrap()
            .is_none()
    );
    let missing = [String::from("replacement"), String::from("absent")];
    assert!(
        matches!(root.prepare_encoded_read(&missing), Err(StoreError::UnknownTensor { key }) if key == "absent")
    );
    assert!(matches!(
        MemoryEncodedReadPlan::from_source(&root, &missing),
        Err(MemoryEncodedReadRouteError::Source(
            MemoryEncodedReadPlanError::UnknownTensor { index: 1 }
        ))
    ));
    assert_eq!(
        MemoryEncodedReadPlan::from_source(&root, &[])
            .unwrap()
            .unwrap()
            .construct(())
            .unwrap()
            .byte_len(),
        0
    );
    let restricted: RetainedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(root, "only a", BTreeSet::from(["a".into()]))
            .unwrap(),
    )
    .into();
    assert!(matches!(
        MemoryEncodedReadPlan::from_source(&restricted, &mixed),
        Err(MemoryEncodedReadRouteError::UnauthorizedTensor { index: 1 })
    ));
    let error = match SafetensorsEncodedReadPlan::from_source(&restricted, &mixed) {
        Err(error) => error,
        _ => panic!("outer authorization precedes leaf selection"),
    };
    drop(restricted);
    assert_eq!(error.contract(), Some("only a"));
}

#[test]
fn nested_prepared_catalog_failures_preserve_inner_first_order() {
    let (_directory, sources) = fixtures();
    for (file, source) in sources.into_iter().enumerate() {
        let mut inner = snapshot(overlay(source));
        inner.catalog.get_mut("b").unwrap().provenance.output = "stale inner".into();
        let mut outer = snapshot(overlay(Arc::new(inner).into()));
        outer.catalog.get_mut("a").unwrap().provenance.output = "stale outer".into();
        let root: RetainedCheckpointSource = Arc::new(outer).into();
        let keys = [String::from("a"), String::from("b")];
        assert!(
            matches!(root.prepare_encoded_read(&keys), Err(StoreError::PreparedCatalogMismatch { key }) if key == "b")
        );
        if file == 0 {
            assert!(matches!(
                MemoryEncodedReadPlan::from_source(&root, &keys),
                Err(MemoryEncodedReadRouteError::PreparedCatalogMismatch { index: 1 })
            ));
        } else {
            assert!(matches!(
                SafetensorsEncodedReadPlan::from_source(&root, &keys).map_err(|e| e.kind()),
                Err(SafetensorsEncodedReadPlanErrorKind::PreparedCatalogMismatch { index: 1 })
            ));
        }
    }
}
