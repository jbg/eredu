use super::*;
use crate::store::{
    CompositeCheckpointSource, PreparedCheckpointSource, PreparedTensorSource,
    RestrictedCheckpointSource, SelectedGgufConversionPlan, SharedCheckpointSource,
};
use eredu_gguf::{Endian, WriterOptions};

fn file(path: &Path, ty: GgmlType, endian: Endian) {
    let (block_values, block_bytes) = ty.block_and_bytes().unwrap();
    let raw = if ty == GgmlType::F32 {
        (0..128)
            .flat_map(|i| {
                let value = (i as f32 - 31.0) * 0.25;
                match endian {
                    Endian::Little => value.to_le_bytes(),
                    Endian::Big => value.to_be_bytes(),
                }
            })
            .collect::<Vec<_>>()
    } else {
        (0..128 / block_values * block_bytes)
            .map(|i| (i.wrapping_mul(13).wrapping_add(37) % 256) as u8)
            .collect()
    };
    Writer::new(WriterOptions {
        version: 3,
        endian,
        alignment: 32,
    })
    .unwrap()
    .write(
        File::create(path).unwrap(),
        &BTreeMap::new(),
        &[TensorInput {
            name: "matrix.weight",
            dimensions: &[64, 2],
            ggml_type: ty,
            data: &raw,
        }],
    )
    .unwrap();
}
fn counters(source: &GgufWeightStore) -> (u64, u64, u64, u64, u64, u64, Vec<Option<PathBuf>>) {
    let diagnostics = source.diagnostics().unwrap();
    let readers = source.inner.readers.lock().unwrap();
    (
        diagnostics.physical_reads,
        diagnostics.physical_read_bytes,
        diagnostics.cache_hits,
        diagnostics.cache_misses,
        diagnostics.evictions,
        readers.tick,
        readers
            .materializers
            .iter()
            .map(|m| m.open_shard_path().map(Path::to_path_buf))
            .collect(),
    )
}
fn shapes(tensor: &ConvertedTensor) -> Vec<(Vec<u64>, LogicalDtype)> {
    match tensor {
        ConvertedTensor::Dense(t) => vec![(t.shape.clone(), t.dtype.into())],
        ConvertedTensor::IQuant(t) => vec![(t.packed_shape().unwrap(), LogicalDtype::U8)],
        ConvertedTensor::Affine(t) => vec![
            (t.weight_shape.clone(), LogicalDtype::U32),
            (t.scale_shape.clone(), LogicalDtype::F16),
            (t.scale_shape.clone(), LogicalDtype::F16),
        ],
        ConvertedTensor::MxFp4(t) => vec![
            (t.weight_shape.clone(), LogicalDtype::U32),
            (t.scale_shape.clone(), LogicalDtype::U8),
        ],
    }
}
fn planned(plan: &GgufConversionPlan) -> Vec<(Vec<u64>, LogicalDtype)> {
    plan.conversion()
        .outputs()
        .iter()
        .map(|p| (p.shape().to_vec(), p.dtype()))
        .collect()
}
#[test]
fn selected_plans_match_real_groups_for_endian_formats_and_ordered_physical_selections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("selected.gguf");
    for endian in [Endian::Little, Endian::Big] {
        for ty in [
            GgmlType::F32,
            GgmlType::Q4_0,
            GgmlType::Q8_0,
            GgmlType::MxFp4,
        ] {
            file(&path, ty, endian);
            let source = test_store(&path);
            for key in source.keys() {
                let unit = source.inner.catalog[&key]
                    .logical_last_units_per_block
                    .unwrap();
                for selection in [
                    TensorSelection::Full,
                    TensorSelection::Range {
                        axis: 0,
                        start: 1,
                        end: 2,
                    },
                    TensorSelection::Indices {
                        axis: 0,
                        indices: vec![1, 0, 1],
                    },
                    TensorSelection::Contiguous {
                        offset_elements: unit,
                        shape: vec![1, unit],
                    },
                ] {
                    let request = TensorReadRequest {
                        key: key.clone(),
                        selection,
                        policy: ReadPolicy::RequireBounded,
                    };
                    let before = counters(&source);
                    let plan = source.conversion_plan(&request).unwrap();
                    assert_eq!(counters(&source), before);
                    let lease = source.acquire(request.clone()).unwrap();
                    assert!(plan.matches_lease(&lease));
                    assert_eq!(plan.output_name(), lease.logical_output_name());
                    assert_eq!(plan.output_shape(), lease.output_shape());
                    assert_eq!(
                        plan.requested_read_bytes(),
                        lease.bounded_read_proof().length_bytes
                    );
                    assert_eq!(
                        plan.relative_read_offset_bytes(),
                        lease.bounded_read_proof().offset_bytes
                    );
                    assert_eq!(
                        plan.physically_bounded(),
                        lease.bounded_read_proof().physically_bounded
                    );
                    let from_lease = lease.conversion_plan().unwrap();
                    assert_eq!(plan.identity(), from_lease.identity());
                    assert_eq!(planned(&plan), planned(&from_lease));
                    assert_eq!(counters(&source), before);
                    let prepared = lease
                        .prepare_portable_read()
                        .unwrap()
                        .prepare_conversion()
                        .unwrap();
                    assert_eq!(
                        plan.conversion().requested_layouts().vectors,
                        prepared.conversion_layouts().unwrap().vectors
                    );
                    let output = prepared.materialize().unwrap();
                    assert_eq!(
                        planned(&plan),
                        shapes(output.converted()),
                        "{ty:?} {endian:?} {key}"
                    );
                    if ty == GgmlType::F32
                        && matches!(request.selection, TensorSelection::Indices { .. })
                    {
                        let ConvertedTensor::Dense(dense) = output.converted() else {
                            panic!("dense")
                        };
                        let values: Vec<_> = dense
                            .data
                            .chunks_exact(4)
                            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                            .collect();
                        assert_eq!(values.len(), 192);
                        assert_eq!(values[0], 8.25);
                        assert_eq!(values[64], -7.75);
                        assert_eq!(&values[..64], &values[128..]);
                    }
                }
            }
        }
    }
}
#[test]
fn fallback_plan_tracks_full_physical_group_and_companion_identity_without_reading() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fallback.gguf");
    file(&path, GgmlType::Q4_0, Endian::Big);
    let source = test_store(&path);
    let mut request = TensorReadRequest {
        key: "matrix.weight".into(),
        selection: TensorSelection::Range {
            axis: 1,
            start: 1,
            end: 2,
        },
        policy: ReadPolicy::RequireBounded,
    };
    let before = counters(&source);
    let error = source.conversion_plan(&request).unwrap_err();
    let ordinary = source.acquire(request.clone()).unwrap_err();
    assert!(matches!(
        error,
        GgufConversionPlanError::Store(StoreError::BoundedSelectionUnavailable { .. })
    ));
    assert_eq!(error.to_string(), ordinary.to_string());
    request.policy = ReadPolicy::AllowFullTensorRead;
    let plan = source.conversion_plan(&request).unwrap();
    assert!(!plan.selection_is_materialized());
    assert!(!plan.physically_bounded());
    assert_eq!(plan.output_shape(), [2, 1]);
    assert_eq!(plan.conversion().descriptor().dimensions, [64, 2]);
    let lease = source.acquire(request).unwrap();
    assert!(plan.matches_lease(&lease));
    assert_eq!(
        plan.requested_read_bytes(),
        lease.bounded_read_proof().length_bytes
    );
    assert_eq!(
        plan.relative_read_offset_bytes(),
        lease.bounded_read_proof().offset_bytes
    );
    assert_eq!(
        plan.physically_bounded(),
        lease.bounded_read_proof().physically_bounded
    );
    assert_eq!(counters(&source), before);
    assert_eq!(
        planned(&plan),
        shapes(lease.materialize_portable().unwrap().converted())
    );
    let full = |key: &str| TensorReadRequest {
        key: key.into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    let weights = source.conversion_plan(&full("matrix.weight")).unwrap();
    let scales = source.conversion_plan(&full("matrix.scales")).unwrap();
    assert_eq!(weights.identity(), scales.identity());
    assert_ne!(weights.output_name(), scales.output_name());
    assert_eq!(planned(&weights), planned(&scales));
    let foreign = test_store(&path);
    let foreign_lease = foreign.acquire(full("matrix.weight")).unwrap();
    assert_ne!(weights.identity(), foreign_lease.identity());
    assert!(!weights.matches_lease(&foreign_lease));
}

#[test]
fn unavailable_source_query_does_not_call_ordinary_fallback_or_invent_zero_fit() {
    struct Unavailable;
    impl CheckpointSource for Unavailable {
        fn source_keys(&self) -> Vec<String> {
            panic!("no owned key fallback")
        }
        fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            panic!("no metadata fallback")
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("no acquisition fallback")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            panic!("no diagnostics fallback")
        }
    }
    let source: SharedCheckpointSource = Arc::new(Unavailable);
    let request = TensorReadRequest {
        key: "selected".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    assert!(SelectedGgufConversionPlan::query(source, request)
        .unwrap()
        .is_none());
}
#[test]
fn cold_source_query_retains_exact_root_metadata_and_authorization_without_payload_or_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("cold.gguf");
    file(&path, GgmlType::F32, Endian::Little);
    let leaf = Arc::new(test_store(&path));
    let restricted: SharedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            leaf.clone(),
            "selected",
            BTreeSet::from(["matrix.weight".into()]),
        )
        .unwrap(),
    );
    let restriction = restricted.clone();
    let composite: SharedCheckpointSource =
        Arc::new(CompositeCheckpointSource::new([restricted]).unwrap());
    let catalog = composite
        .source_keys()
        .into_iter()
        .map(|key| {
            let row = PreparedTensorSource {
                metadata: composite.source_metadata(&key).unwrap(),
                provenance: composite.source_provenance(&key).unwrap(),
            };
            (key, row)
        })
        .collect();
    let root: SharedCheckpointSource =
        Arc::new(PreparedCheckpointSource::new(composite, catalog).unwrap());
    let request = TensorReadRequest {
        key: "matrix.weight".into(),
        selection: TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        policy: ReadPolicy::RequireBounded,
    };
    let before = counters(&leaf);
    std::fs::remove_file(&path).unwrap(); // queries cannot be reopening this artifact
    let mut plans = Vec::new();
    for _ in 0..32 {
        let plan = SelectedGgufConversionPlan::query(root.clone(), request.clone())
            .unwrap()
            .unwrap();
        assert!(plan.matches_request(&root, &request));
        assert!(!plan.matches_request(&(leaf.clone() as SharedCheckpointSource), &request));
        assert!(plan.metadata_bytes().unwrap() > 0);
        assert_eq!(plan.physical().conversion().outputs()[0].shape(), [3, 64]);
        plans.push(plan);
    }
    assert_eq!(counters(&leaf), before);
    let denied = TensorReadRequest {
        key: "hidden".into(),
        ..request
    };
    let error = SelectedGgufConversionPlan::query(root.clone(), denied.clone()).unwrap_err();
    let ordinary = root.acquire_lease(denied).unwrap_err();
    assert_eq!(error.to_string(), ordinary.to_string());
    assert!(matches!(
        error,
        GgufConversionPlanError::Store(StoreError::UnknownTensor { .. })
    ));
    assert_eq!(counters(&leaf), before);
    let denied = TensorReadRequest {
        key: "hidden".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    let error = SelectedGgufConversionPlan::query(restriction.clone(), denied.clone()).unwrap_err();
    let ordinary = restriction.acquire_lease(denied).unwrap_err();
    assert_eq!(error.to_string(), ordinary.to_string());
    assert!(matches!(
        error,
        GgufConversionPlanError::Store(StoreError::UnauthorizedTensor { .. })
    ));
    assert_eq!(counters(&leaf), before);
    drop(restriction);
    let weak = Arc::downgrade(&root);
    drop((root, leaf));
    assert!(weak.upgrade().is_some());
    assert_eq!(
        plans[0].physical().conversion().requested_elements()[2],
        3 * 64 * 4
    );
    drop(plans);
    assert!(weak.upgrade().is_none());
}

#[test]
fn physical_group_identity_preserves_source_aliases_selection_and_weak_retirement() {
    let directory = tempfile::tempdir().unwrap();
    let first_path = directory.path().join("first.gguf");
    let second_path = directory.path().join("second.gguf");
    for (path, first) in [(&first_path, 1.25_f32), (&second_path, -7.5_f32)] {
        let values = [first, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let bytes = values
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        write_tensor(path, "matrix.weight", &[4, 2], GgmlType::F32, &bytes);
    }
    let source = Arc::new(test_store(&first_path));
    let alias = source.as_ref().clone();
    let reopened = test_store(&first_path);
    let foreign = test_store(&second_path);
    let source_weak = source.inner.ordinary_weak();
    let before = counters(&source);
    let request = TensorReadRequest {
        key: "matrix.weight".into(),
        selection: TensorSelection::Indices {
            axis: 0,
            indices: vec![1, 0, 1],
        },
        policy: ReadPolicy::RequireBounded,
    };
    let plan = source.conversion_plan(&request).unwrap();
    let lease = source.acquire(request.clone()).unwrap();
    let alias_lease = alias.acquire(request.clone()).unwrap();
    let wrapped = RestrictedCheckpointSource::including(
        source.clone(),
        "same physical source",
        BTreeSet::from(["matrix.weight".into()]),
    )
    .unwrap();
    let wrapped_lease = wrapped.acquire_lease(request.clone()).unwrap();
    let CheckpointLease::Gguf(wrapped_lease) = wrapped_lease else {
        panic!("expected GGUF lease");
    };
    assert_eq!(plan.identity(), lease.identity());
    assert_eq!(lease.identity(), alias_lease.identity());
    assert_eq!(lease.identity(), wrapped_lease.identity());
    assert!(plan.matches_lease(&wrapped_lease));
    assert_ne!(
        lease.identity(),
        reopened.acquire(request.clone()).unwrap().identity()
    );
    assert_ne!(
        lease.identity(),
        foreign.acquire(request.clone()).unwrap().identity()
    );
    let reordered = source
        .acquire(TensorReadRequest {
            selection: TensorSelection::Indices {
                axis: 0,
                indices: vec![0, 1, 1],
            },
            ..request
        })
        .unwrap();
    assert_ne!(lease.identity(), reordered.identity());
    assert_eq!(counters(&source), before);

    let retained = lease.identity().clone();
    let retained_alias = wrapped_lease.identity().clone();
    drop((
        plan,
        lease,
        alias_lease,
        wrapped_lease,
        reordered,
        wrapped,
        alias,
        source,
    ));
    assert!(
        source_weak.upgrade().is_none(),
        "keys must not keep source payloads alive"
    );
    assert_eq!(retained, retained_alias);
    let replacement = test_store(&first_path);
    let replacement_lease = replacement
        .acquire(TensorReadRequest {
            key: "matrix.weight".into(),
            selection: TensorSelection::Indices {
                axis: 0,
                indices: vec![1, 0, 1],
            },
            policy: ReadPolicy::RequireBounded,
        })
        .unwrap();
    assert_ne!(&retained, replacement_lease.identity());
}
