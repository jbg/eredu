use super::*;
use crate::recipe::DerivedWeightRecipe;
use safetensors::tensor::Dtype as SafeDtype;
#[test]
fn borrowed_overlay_metadata_keeps_real_materialized_authority_through_resolved_view() {
    use crate::{
        StoredDtype,
        schema::{
            CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
            StoredDtypeConstraint,
        },
        store::{ResolvedCheckpointSource, SourceKeyAuthority, SourceMetadataBorrowError},
        validation::resolve_safetensors_plan,
    };
    let source: crate::store::RetainedCheckpointSource = (Arc::new(
        MemoryWeightStore::from_safetensors([
            (
                "weight".into(),
                SafeDtype::F32,
                vec![1],
                0.75f32.to_le_bytes().to_vec(),
            ),
            (
                "unchanged".into(),
                SafeDtype::F32,
                vec![1],
                (-2.5f32).to_le_bytes().to_vec(),
            ),
        ])
        .unwrap(),
    ))
    .into();
    let plan = SafetensorsCheckpointPlan::new(
        "selected",
        vec![SafetensorsTensorConstraint::required(
            "unchanged",
            vec![1],
            StoredDtypeConstraint::Exact(StoredDtype::F32),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let contract = resolve_safetensors_plan(source.as_ref(), &plan).unwrap();
    let overlay = Arc::new(MaterializedCheckpointSource {
        source,
        transformed: Arc::new(
            MemoryWeightStore::from_safetensors([(
                "packed".into(),
                SafeDtype::U32,
                vec![1],
                0x76543210u32.to_le_bytes().to_vec(),
            )])
            .unwrap(),
        )
        .into(),
        materialized_source_keys: BTreeSet::new(),
        materialized_source_shards: BTreeSet::new(),
    });
    let resolved = ResolvedCheckpointSource::new(overlay.clone(), contract);
    // This exact transformed memory source must qualify the same original
    // manager reader; no file descriptor or ordinary lease is substituted.
    let packed_read = DerivedWeightRecipe::source("packed", TensorSelection::Full)
        .prepare_encoded_read(&resolved)
        .unwrap()
        .unwrap();
    let mut packed_bytes = [0; 4];
    crate::recipe::EncodedRecipeRead::read_many_borrowed_into(
        std::iter::once(&packed_read),
        &mut [&mut packed_bytes],
    )
    .unwrap();
    assert_eq!(packed_bytes, 0x76543210u32.to_le_bytes());
    let unchanged_read = DerivedWeightRecipe::source("unchanged", TensorSelection::Full)
        .prepare_encoded_read(&resolved)
        .unwrap()
        .unwrap();
    let mut unchanged_bytes = [0; 4];
    crate::recipe::EncodedRecipeRead::read_many_borrowed_into(
        std::iter::once(&unchanged_read),
        &mut [&mut unchanged_bytes],
    )
    .unwrap();
    assert_eq!(unchanged_bytes, (-2.5f32).to_le_bytes());
    let packed = overlay
        .transformed
        .source_metadata_borrowed("packed")
        .unwrap();
    assert!(std::ptr::eq(
        packed,
        resolved.source_metadata_borrowed("packed").unwrap()
    ));
    let packed_loan = resolved.source_lease_controls("packed").unwrap();
    assert!(packed_loan.same_entry(&overlay.transformed.source_lease_controls("packed").unwrap()));
    assert!(!packed_loan.same_entry(&overlay.source.source_lease_controls("weight").unwrap()));
    assert!(
        resolved
            .source_lease_controls("unchanged")
            .unwrap()
            .same_entry(&overlay.source.source_lease_controls("unchanged").unwrap())
    );
    assert!(matches!(
        resolved.source_lease_controls("weight"),
        Err(crate::store::LeaseControlBorrowError::Source(
            SourceMetadataBorrowError::UnauthorizedTensor
        ))
    ));
    assert_eq!(packed.stored_dtype, StoredDtype::U32);
    assert_eq!(resolved.source_metadata("packed").unwrap(), *packed);
    assert_eq!(
        resolved.source_key_authority_borrowed("packed").unwrap(),
        SourceKeyAuthority::Materialized
    );
    let unchanged = overlay
        .source
        .source_metadata_borrowed("unchanged")
        .unwrap();
    assert!(std::ptr::eq(
        unchanged,
        resolved.source_metadata_borrowed("unchanged").unwrap()
    ));
    for key in ["weight", "absent"] {
        assert!(matches!(
            resolved.source_metadata_borrowed(key),
            Err(SourceMetadataBorrowError::UnauthorizedTensor)
        ));
        assert!(matches!(
            resolved.source_metadata(key),
            Err(StoreError::UnauthorizedTensor { .. })
        ));
    }
}

struct Renamed(MemoryWeightStore);
impl CheckpointSource for Renamed {
    fn source_keys(&self) -> Vec<String> {
        self.0.source_keys()
    }
    fn source_metadata(&self, key: &str) -> Result<crate::store::TensorMetadata, StoreError> {
        self.0.source_metadata(key)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.0.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.0.source_diagnostics()
    }
    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<crate::store::TensorSourceProvenance, StoreError> {
        let mut provenance = self.0.source_provenance(key)?;
        provenance.physical_tensor = format!("checkpoint.prefix.{key}");
        provenance.output = format!("selected.{key}");
        Ok(provenance)
    }
}
#[test]
fn bounded_overlay_preserves_original_passthrough_and_actual_output_provenance() {
    let source: crate::store::RetainedCheckpointSource = (Arc::new(Renamed(
        MemoryWeightStore::from_safetensors(["weight", "unchanged"].map(|key| {
            (
                key.to_owned(),
                SafeDtype::F32,
                vec![1],
                0.75f32.to_le_bytes().to_vec(),
            )
        }))
        .unwrap(),
    )))
    .into();
    let transformed = MemoryWeightStore::from_safetensors([(
        "weight".to_owned(),
        SafeDtype::U32,
        vec![1],
        0x76543210u32.to_le_bytes().to_vec(),
    )])
    .unwrap();
    let expected_output = transformed.source_provenance("weight").unwrap();
    let expected_passthrough = source.source_provenance("unchanged").unwrap();
    let overlay = MaterializedCheckpointSource {
        source,
        transformed: Arc::new(transformed).into(),
        materialized_source_keys: BTreeSet::new(),
        materialized_source_shards: BTreeSet::new(),
    };
    assert_eq!(
        overlay.source_provenance("unchanged").unwrap(),
        expected_passthrough
    );
    assert_eq!(
        overlay.source_provenance("weight").unwrap(),
        expected_output
    );
    assert_eq!(
        overlay.source_provenance("weight").unwrap().source_encoding,
        crate::SourceTensorEncoding::Safetensors(crate::StoredDtype::U32)
    );
}

fn memory(rows: &[(&str, &[u8])]) -> MemoryWeightStore {
    MemoryWeightStore::from_safetensors(
        rows.iter()
            .map(|(key, bytes)| ((*key).into(), Dtype::U8, vec![bytes.len()], bytes.to_vec())),
    )
    .unwrap()
}

#[test]
fn memory_replacements_keep_hidden_storage_and_reads_outlive_the_overlay() {
    let base = Arc::new(memory(&[("weight", &[1, 2]), ("unchanged", &[3, 4])]));
    let hidden = base.tensors["weight"].ordinary_weak();
    let converted = memory(&[("weight", &[9, 8]), ("packed", &[7, 6])]);
    let output = converted.tensors["weight"].ordinary_weak();
    let root: RetainedCheckpointSource = Arc::new(MaterializedCheckpointSource::new(
        base.clone().into(),
        converted,
        BTreeSet::from(["weight".into()]),
        BTreeSet::from([PathBuf::from("original.safetensors")]),
    ))
    .into();
    drop(base);
    assert!(hidden.upgrade().is_some());
    assert_eq!(root.source_keys(), ["packed", "unchanged", "weight"]);
    assert_eq!(root.materialized_source_keys(), ["weight"]);
    assert_eq!(
        root.materialized_source_shards(),
        [PathBuf::from("original.safetensors")]
    );
    assert_eq!(root.source_storage().unwrap().unwrap().owner_count(), 4);
    let mut owners = 0;
    assert!(root.visit_source_storage(&mut |_| owners += 1).unwrap());
    assert_eq!(owners, 4);
    let keys = ["weight".into(), "weight".into()];
    let plan = MemoryEncodedReadPlan::from_source(&root, &keys)
        .unwrap()
        .unwrap();
    let unchanged_keys = ["unchanged".into()];
    let unchanged = MemoryEncodedReadPlan::from_source(&root, &unchanged_keys)
        .unwrap()
        .unwrap()
        .construct(())
        .unwrap();
    for keys in [
        vec!["weight".into(), "unchanged".into()],
        vec!["packed".into(), "absent".into()],
    ] {
        assert!(root.prepare_encoded_read(&keys).unwrap().is_none());
        assert!(
            MemoryEncodedReadPlan::from_source(&root, &keys)
                .unwrap()
                .is_none()
        );
    }
    let empty = MemoryEncodedReadPlan::from_source(&root, &[])
        .unwrap()
        .unwrap()
        .construct(())
        .unwrap();
    assert_eq!(empty.byte_len(), 0);
    drop(empty);
    drop(root);
    assert!(output.upgrade().is_some());
    let read = plan.construct(()).unwrap();
    let mut bytes = [0; 4];
    read.read_into(&mut bytes).unwrap();
    assert_eq!(bytes, [9, 8, 9, 8]);
    let mut bytes = [0; 2];
    unchanged.read_into(&mut bytes).unwrap();
    assert_eq!(bytes, [3, 4]);
    drop(unchanged);
    assert!(hidden.upgrade().is_none());
    drop(read);
    assert!(output.upgrade().is_none());
}

#[test]
fn enclosing_contract_admits_actual_outputs_but_restriction_still_denies_them() {
    use crate::{schema::*, validation::resolve_safetensors_plan};
    let base: RetainedCheckpointSource =
        Arc::new(memory(&[("weight", &[1, 2]), ("unchanged", &[3, 4])])).into();
    let plan = SafetensorsCheckpointPlan::new(
        "selected",
        vec![SafetensorsTensorConstraint::required(
            "unchanged",
            vec![2],
            StoredDtypeConstraint::Exact(StoredDtype::U8),
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .unwrap();
    let contract = resolve_safetensors_plan(base.as_ref(), &plan).unwrap();
    let overlay = Arc::new(MaterializedCheckpointSource::new(
        base,
        memory(&[("weight", &[9, 8])]),
        BTreeSet::new(),
        BTreeSet::new(),
    ));
    let resolved: RetainedCheckpointSource =
        Arc::new(ResolvedCheckpointSource::new(overlay, contract)).into();
    let keys = ["weight".into()];
    let read = MemoryEncodedReadPlan::from_source(&resolved, &keys)
        .unwrap()
        .unwrap()
        .construct(())
        .unwrap();
    let mut bytes = [0; 2];
    read.read_into(&mut bytes).unwrap();
    assert_eq!(bytes, [9, 8]);
    let restricted: RetainedCheckpointSource = Arc::new(
        RestrictedCheckpointSource::including(
            resolved,
            "unchanged only",
            BTreeSet::from(["unchanged".into()]),
        )
        .unwrap(),
    )
    .into();
    assert!(matches!(
        MemoryEncodedReadPlan::from_source(&restricted, &keys),
        Err(MemoryEncodedReadRouteError::UnauthorizedTensor { index: 0 })
    ));
    let request = TensorReadRequest {
        key: "weight".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    assert!(PreparedCheckpointAcquisition::prepare(restricted, request).is_err());
}

#[test]
fn file_passthrough_uses_prepared_headers_and_memory_replacements_never_read_the_file() {
    use safetensors::tensor::{TensorView, serialize_to_file};
    let directory = tempfile::tempdir().unwrap();
    serialize_to_file(
        [
            (
                "weight",
                TensorView::new(Dtype::U8, vec![2], &[1, 2]).unwrap(),
            ),
            (
                "unchanged",
                TensorView::new(Dtype::U8, vec![2], &[3, 4]).unwrap(),
            ),
        ],
        None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    std::fs::write(
        directory.path().join("model.safetensors.index.json"),
        r#"{"weight_map":{"weight":"model.safetensors","unchanged":"model.safetensors"}}"#,
    )
    .unwrap();
    let base = Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
    let root: RetainedCheckpointSource = Arc::new(MaterializedCheckpointSource::new(
        base.clone().into(),
        memory(&[("weight", &[9, 8])]),
        BTreeSet::new(),
        BTreeSet::new(),
    ))
    .into();
    let keys = ["unchanged".into()];
    assert!(matches!(
        SafetensorsEncodedReadPlan::from_source(&root, &keys).map_err(|e| e.kind()),
        Err(SafetensorsEncodedReadPlanErrorKind::HeaderUnavailable { index: 0 })
    ));
    let replaced_keys = ["weight".into()];
    assert!(
        SafetensorsEncodedReadPlan::from_source(&root, &replaced_keys)
            .unwrap()
            .is_none()
    );
    let replacement = MemoryEncodedReadPlan::from_source(&root, &replaced_keys)
        .unwrap()
        .unwrap()
        .construct(())
        .unwrap();
    assert_eq!(base.diagnostics().unwrap().physical_reads, 0);
    WeightStore::metadata(base.as_ref(), "unchanged").unwrap();
    let read = SafetensorsEncodedReadPlan::from_source(&root, &keys)
        .unwrap()
        .unwrap()
        .construct(())
        .unwrap();
    let request = TensorReadRequest {
        key: "weight".into(),
        selection: TensorSelection::Full,
        policy: ReadPolicy::RequireBounded,
    };
    let acquisition = PreparedCheckpointAcquisition::prepare(root.clone(), request)
        .unwrap()
        .unwrap();
    drop((root, base, keys, replaced_keys));
    let mut bytes = [0; 2];
    read.read_into(&mut bytes).unwrap();
    assert_eq!(bytes, [3, 4]);
    replacement.read_into(&mut bytes).unwrap();
    assert_eq!(bytes, [9, 8]);
    let lease = acquisition.acquire().unwrap();
    assert_eq!(lease.encoded_bytes().unwrap(), [9, 8]);
}
