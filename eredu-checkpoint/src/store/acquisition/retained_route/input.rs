//! Borrowed declarations used to estimate cold materialization host overhead.
use super::*;
use std::mem::size_of;

fn names<'a>(mut keys: impl Iterator<Item = &'a String>) -> Option<usize> {
    keys.try_fold(0usize, |n, key| {
        n.checked_add(size_of::<String>())?.checked_add(key.len())
    })
}

pub(in crate::store) fn bytes(source: &RetainedCheckpointSource) -> Option<usize> {
    let owner = source.acquisition_owner()?;
    let bytes = match owner.route() {
        Route::Unavailable => return None,
        Route::Memory(store) => names(store.tensors.keys())?,
        Route::Gguf(store) => names(store.catalog_keys())?,
        Route::Safetensors(store) => names(store.catalog.keys())?,
        Route::Materialized(store) => store.materialization_input_bytes()?,
        Route::Prepared(store) => {
            names(store.catalog.keys())?.checked_add(bytes(&store.source)?)?
        }
        Route::Restricted(store) => bytes(&store.source)?.checked_add(store.contract.len())?,
        Route::Resolved(store) => {
            bytes(&store.source)?.checked_add(store.contract.identity().len())?
        }
        Route::Composite(store) => store
            .sources
            .iter()
            .try_fold(0usize, |n, child| n.checked_add(bytes(child)?))?,
    };
    size_of::<RetainedCheckpointSource>().checked_add(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn memory(key: &str, payload: usize) -> RetainedCheckpointSource {
        Arc::new(
            MemoryWeightStore::from_safetensors([(
                key.into(),
                safetensors::Dtype::U8,
                vec![payload],
                vec![1; payload],
            )])
            .unwrap(),
        )
        .into()
    }

    #[test]
    fn input_estimate_tracks_declarations_without_counting_tensor_payloads() {
        let small = memory("weight", 1);
        let large = memory("weight", 4096);
        assert_eq!(bytes(&small), bytes(&large));
        assert_eq!(
            bytes(&memory("weight.longer", 1)).unwrap() - bytes(&small).unwrap(),
            7
        );
        let composite = RetainedCheckpointSource::from_composite(
            CompositeCheckpointSource::new([small.clone(), memory("other", 1)]).unwrap(),
        );
        let restricted: RetainedCheckpointSource = Arc::new(
            RestrictedCheckpointSource::including(
                composite.clone(),
                "selected",
                ["weight".into()].into(),
            )
            .unwrap(),
        )
        .into();
        // Cold source_keys on a restricted view still builds its child's keys.
        assert!(bytes(&restricted).unwrap() > bytes(&composite).unwrap());
        let catalog = small
            .source_keys()
            .into_iter()
            .map(|key| {
                let row = crate::store::PreparedTensorSource {
                    metadata: small.source_metadata(&key).unwrap(),
                    provenance: small.source_provenance(&key).unwrap(),
                };
                (key, row)
            })
            .collect();
        let prepared = RetainedCheckpointSource::from_prepared_with_custody(
            PreparedCheckpointSource::new(small.clone(), catalog).unwrap(),
            (),
        );
        assert!(bytes(&prepared).unwrap() > bytes(&small).unwrap());
        let overlay =
            RetainedCheckpointSource::from_materialized(MaterializedCheckpointSource::new(
                prepared.clone(),
                MemoryWeightStore::default(),
                ["weight".into()].into(),
                [std::path::PathBuf::from("consumed/model.safetensors")].into(),
            ));
        assert!(bytes(&overlay).unwrap() > bytes(&prepared).unwrap());
    }

    #[test]
    fn input_estimate_does_not_open_a_safetensors_payload() {
        use safetensors::tensor::{TensorView, serialize_to_file};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        serialize_to_file(
            [(
                "weight",
                TensorView::new(safetensors::Dtype::U8, vec![2], &[3, 7]).unwrap(),
            )],
            None,
            &path,
        )
        .unwrap();
        let source = RetainedCheckpointSource::from_safetensors(
            SafetensorsWeightStore::open(&path).unwrap(),
        );
        let before = bytes(&source).unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(bytes(&source), Some(before));
        assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    }

    struct Forwarded(RetainedCheckpointSource);
    impl CheckpointSource for Forwarded {
        fn prepared_acquisition_owner(self: Arc<Self>) -> Option<PreparedAcquisitionOwner> {
            self.0.acquisition_owner()
        }
        fn source_keys(&self) -> Vec<String> {
            panic!("ordinary catalog callback")
        }
        fn source_metadata(&self, _: &str) -> Result<TensorMetadata, StoreError> {
            panic!("metadata callback")
        }
        fn acquire_lease(&self, _: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
            panic!("payload callback")
        }
        fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
            panic!("diagnostics callback")
        }
    }
    #[test]
    fn foreign_acquisition_loan_cannot_supply_estimate_inputs() {
        let source: RetainedCheckpointSource = Arc::new(Forwarded(memory("weight", 1))).into();
        assert!(bytes(&source).is_none());
    }
}
