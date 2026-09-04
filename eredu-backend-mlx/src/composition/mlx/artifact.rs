//! Prepared SafeTensors inputs for MLX materialization.

use std::{path::Path, sync::Arc};

use crate::backend::error::Error;
use eredu_checkpoint::{
    safetensors::SafetensorsShards,
    schema::SafetensorsCheckpointPlan,
    store::{
        CheckpointLease, CheckpointSource, EncodedTensorLease, SafetensorsWeightStore,
        SharedCheckpointSource, StoreError, TensorMetadata, TensorReadRequest,
        WeightStoreDiagnostics,
    },
    StoredDtype,
};
use eredu_core::{
    checkpoint::{TensorCatalog, TensorDtype},
    ModelConfiguration,
};

/// Authoritative SafeTensors inputs retained from neutral preparation.
///
/// Family composition receives this object instead of an artifact path so it
/// cannot rediscover configuration or checkpoint topology after admission.
pub struct PreparedSafetensorsArtifact {
    architecture: eredu_architectures::configuration::SafetensorsArchitecturePlan,
    store: SharedCheckpointSource,
}

impl PreparedSafetensorsArtifact {
    pub fn open(
        _configuration: ModelConfiguration,
        architecture: eredu_architectures::configuration::SafetensorsArchitecturePlan,
        catalog: TensorCatalog,
        shards: SafetensorsShards,
        max_cached_shards: usize,
    ) -> Result<Self, Error> {
        #[cfg(test)]
        super::path_instrumentation::payload_open();
        let store = open_admitted_catalog_bound_store(catalog, shards, max_cached_shards)?;
        let resolution = architecture
            .checkpoint_resolution()
            .ok_or_else(|| {
                Error::ArchitectureModel(
                    "prepared SafeTensors artifact omitted its admitted checkpoint layout".into(),
                )
            })?
            .clone();
        let store = eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolution);
        Ok(Self {
            architecture,
            store: Arc::new(store),
        })
    }

    pub fn architecture(&self) -> &eredu_architectures::configuration::SafetensorsArchitecturePlan {
        &self.architecture
    }

    pub fn model(&self) -> &eredu_architectures::configuration::SafetensorsModelConfig {
        self.architecture.model()
    }

    pub fn store(&self) -> SharedCheckpointSource {
        Arc::clone(&self.store)
    }
}

/// Reopens an assistant checkpoint through its admitted catalog and strict schema.
pub(crate) fn open_prepared_safetensors_checkpoint(
    path: &Path,
    catalog: TensorCatalog,
    plan: &SafetensorsCheckpointPlan,
    admitted_resolution: &eredu_checkpoint::validation::ResolvedCheckpointPlan,
    max_cached_shards: usize,
) -> Result<SharedCheckpointSource, Error> {
    #[cfg(test)]
    super::path_instrumentation::payload_open();
    let store = open_catalog_bound_store(path, catalog, max_cached_shards)?;
    let resolution = eredu_checkpoint::validation::resolve_safetensors_plan(store.as_ref(), plan)
        .map_err(|failure| {
        Error::ArchitectureModel(format!(
            "prepared external-assistant SafeTensors contract did not revalidate: {failure:?}"
        ))
    })?;
    if &resolution != admitted_resolution {
        return Err(changed_artifact(
            "strict external-assistant layout resolution differs from preparation",
        ));
    }
    Ok(Arc::new(
        eredu_checkpoint::store::ResolvedCheckpointSource::new(store, resolution),
    ))
}

fn open_catalog_bound_store(
    path: &Path,
    catalog: TensorCatalog,
    max_cached_shards: usize,
) -> Result<SharedCheckpointSource, Error> {
    let store = SafetensorsWeightStore::open_with_max_cached_shards(path, max_cached_shards)?;
    validate_prepared_catalog(&catalog, &store)?;
    Ok(Arc::new(PreparedCatalogSource {
        catalog,
        source: Arc::new(store),
    }))
}

fn open_admitted_catalog_bound_store(
    catalog: TensorCatalog,
    shards: SafetensorsShards,
    max_cached_shards: usize,
) -> Result<SharedCheckpointSource, Error> {
    let store = SafetensorsWeightStore::open_admitted(shards, max_cached_shards)?;
    validate_prepared_catalog(&catalog, &store)?;
    Ok(Arc::new(PreparedCatalogSource {
        catalog,
        source: Arc::new(store),
    }))
}

struct PreparedCatalogSource {
    catalog: TensorCatalog,
    source: SharedCheckpointSource,
}

impl CheckpointSource for PreparedCatalogSource {
    fn source_keys(&self) -> Vec<String> {
        self.catalog
            .descriptors()
            .map(|tensor| tensor.name.clone())
            .collect()
    }

    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let tensor = self
            .catalog
            .get(key)
            .ok_or_else(|| StoreError::UnknownTensor { key: key.into() })?;
        let metadata = self.source.source_metadata(key)?;
        validate_descriptor(tensor, &metadata)?;
        Ok(metadata)
    }

    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        let tensor = self
            .catalog
            .get(&request.key)
            .ok_or_else(|| StoreError::UnknownTensor {
                key: request.key.clone(),
            })?;
        let lease = self.source.acquire_lease(request)?;
        validate_descriptor(tensor, lease.metadata())?;
        Ok(lease)
    }

    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }
}

fn validate_prepared_catalog(
    catalog: &TensorCatalog,
    store: &dyn CheckpointSource,
) -> Result<(), Error> {
    let prepared_keys = catalog
        .descriptors()
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    let current_keys = store.source_keys();
    if current_keys != prepared_keys {
        return Err(changed_artifact(format!(
            "tensor names differ (prepared {}, current {})",
            prepared_keys.len(),
            current_keys.len()
        )));
    }
    for tensor in catalog.descriptors() {
        let metadata = store.source_metadata(&tensor.name)?;
        if !descriptor_matches(tensor, &metadata) {
            return Err(changed_artifact(format!(
                "tensor {:?} metadata differs from the authoritative preparation catalog",
                tensor.name
            )));
        }
    }
    Ok(())
}

fn validate_descriptor(
    tensor: &eredu_core::checkpoint::TensorDescriptor,
    metadata: &TensorMetadata,
) -> Result<(), StoreError> {
    if descriptor_matches(tensor, metadata) {
        return Ok(());
    }
    Err(StoreError::PreparedCatalogMismatch {
        key: tensor.name.clone(),
    })
}

fn descriptor_matches(
    tensor: &eredu_core::checkpoint::TensorDescriptor,
    metadata: &TensorMetadata,
) -> bool {
    let prepared_storage = tensor.storage.as_ref();
    metadata.logical_shape == tensor.shape
        && prepared_storage.map(|storage| storage.length) == Some(metadata.encoded_byte_len)
        && prepared_storage
            .zip(metadata.backing_shard.as_deref())
            .is_some_and(|(storage, shard)| Path::new(&storage.member) == shard)
        && same_dtype(&tensor.dtype, &metadata.stored_dtype)
}

fn changed_artifact(detail: impl Into<String>) -> Error {
    Error::ArchitectureModel(format!(
        "SafeTensors artifact changed after preparation: {}",
        detail.into()
    ))
}

fn same_dtype(prepared: &TensorDtype, current: &StoredDtype) -> bool {
    match (prepared, current) {
        (TensorDtype::Bool, StoredDtype::Bool)
        | (TensorDtype::F32, StoredDtype::F32)
        | (TensorDtype::F16, StoredDtype::F16)
        | (TensorDtype::Bf16, StoredDtype::BF16)
        | (TensorDtype::I8, StoredDtype::I8)
        | (TensorDtype::U8, StoredDtype::U8)
        | (TensorDtype::U16, StoredDtype::U16)
        | (TensorDtype::U32, StoredDtype::U32)
        | (TensorDtype::U64, StoredDtype::U64)
        | (TensorDtype::I16, StoredDtype::I16)
        | (TensorDtype::I32, StoredDtype::I32)
        | (TensorDtype::I64, StoredDtype::I64)
        | (TensorDtype::F64, StoredDtype::F64)
        | (TensorDtype::Complex64, StoredDtype::C64) => true,
        (TensorDtype::Encoded(name), current) => encoded_dtype_name(current) == name,
        _ => false,
    }
}

fn encoded_dtype_name(dtype: &StoredDtype) -> &str {
    match dtype {
        StoredDtype::Bool => "BOOL",
        StoredDtype::U8 => "U8",
        StoredDtype::I8 => "I8",
        StoredDtype::I16 => "I16",
        StoredDtype::U16 => "U16",
        StoredDtype::F16 => "F16",
        StoredDtype::BF16 => "BF16",
        StoredDtype::I32 => "I32",
        StoredDtype::U32 => "U32",
        StoredDtype::F32 => "F32",
        StoredDtype::F64 => "F64",
        StoredDtype::I64 => "I64",
        StoredDtype::U64 => "U64",
        StoredDtype::C64 => "C64",
        StoredDtype::F8E4M3 => "F8_E4M3",
        StoredDtype::F4 => "F4",
        StoredDtype::F8E8M0 => "F8_E8M0",
        StoredDtype::F8E5M2 => "F8_E5M2",
        StoredDtype::Other(name) => name,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        open_prepared_safetensors_checkpoint, PreparedCatalogSource, PreparedSafetensorsArtifact,
    };
    use eredu_checkpoint::schema::{
        CatalogPolicy, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint,
    };
    use eredu_checkpoint::store::{SafetensorsWeightStore, SharedCheckpointSource};
    use eredu_core::{
        checkpoint::{TensorCatalog, TensorDescriptor, TensorDtype, TensorStorage},
        LoadingProtocol, ModelConfiguration,
    };
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    fn configuration(json: serde_json::Value) -> ModelConfiguration {
        ModelConfiguration::new(
            "llama",
            "llama",
            "llama",
            LoadingProtocol::Model,
            Some(json),
        )
        .unwrap()
    }

    fn architecture() -> eredu_architectures::configuration::SafetensorsArchitecturePlan {
        eredu_architectures::configuration::resolve_model_config(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64
        }))
        .unwrap()
        .architecture
    }

    fn catalog_for(path: &std::path::Path, shape: Vec<usize>, length: u64) -> TensorCatalog {
        TensorCatalog::new([TensorDescriptor {
            name: "weight".into(),
            shape,
            dtype: TensorDtype::F32,
            storage: Some(TensorStorage {
                member: path.join("model.safetensors").display().to_string(),
                offset: 0,
                length,
            }),
        }])
        .unwrap()
    }

    fn write_tensor(path: &std::path::Path, values: &[f32]) {
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let shape = vec![values.len()];
        let view = TensorView::new(Dtype::F32, shape, &bytes).unwrap();
        serialize_to_file([("weight", view)], None, &path.join("model.safetensors")).unwrap();
    }

    #[test]
    fn composition_keeps_prepared_configuration_when_sidecar_changes() {
        let directory = tempfile::tempdir().unwrap();
        write_tensor(directory.path(), &[0.0]);
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&serde_json::json!({
                "model_type": "llama",
                "hidden_size": 32
            }))
            .unwrap(),
        )
        .unwrap();

        let source = eredu_checkpoint::store::SafetensorsWeightStore::open_with_max_cached_shards(
            directory.path(),
            1,
        )
        .unwrap();
        let artifact = PreparedSafetensorsArtifact {
            architecture: architecture(),
            store: std::sync::Arc::new(PreparedCatalogSource {
                catalog: catalog_for(directory.path(), vec![1], 4),
                source: std::sync::Arc::new(source),
            }),
        };

        let eredu_architectures::configuration::SafetensorsModelConfig::Llama(args) =
            artifact.model()
        else {
            panic!("expected prepared Llama geometry");
        };
        assert_eq!(args.hidden_size, 16);
    }

    #[test]
    fn composition_rejects_store_metadata_that_differs_from_preparation() {
        let directory = tempfile::tempdir().unwrap();
        write_tensor(directory.path(), &[0.0, 0.0]);
        let error = PreparedSafetensorsArtifact::open(
            configuration(serde_json::json!({"model_type": "llama"})),
            architecture(),
            catalog_for(directory.path(), vec![1], 4),
            eredu_checkpoint::safetensors::SafetensorsShards::discover(directory.path()).unwrap(),
            1,
        )
        .err()
        .expect("changed tensor metadata must be rejected");

        assert!(
            error
                .to_string()
                .contains("authoritative preparation catalog"),
            "{error}"
        );
    }

    #[test]
    fn external_assistant_store_revalidates_the_prepared_catalog_before_loading() {
        let directory = tempfile::tempdir().unwrap();
        write_tensor(directory.path(), &[0.0]);
        let catalog = catalog_for(directory.path(), vec![1], 4);
        let plan = SafetensorsCheckpointPlan::new(
            "test assistant",
            vec![SafetensorsTensorConstraint::required(
                "weight",
                vec![1],
                StoredDtypeConstraint::Floating,
            )],
            Vec::new(),
            CatalogPolicy::strict(),
        )
        .unwrap();
        let source: SharedCheckpointSource =
            std::sync::Arc::new(SafetensorsWeightStore::open(directory.path()).unwrap());
        let resolution =
            eredu_checkpoint::validation::resolve_safetensors_plan(source.as_ref(), &plan).unwrap();

        write_tensor(directory.path(), &[0.0, 0.0]);
        let error =
            open_prepared_safetensors_checkpoint(directory.path(), catalog, &plan, &resolution, 1)
                .err()
                .expect("changed assistant catalog must be rejected");

        assert!(error.to_string().contains("changed after preparation"));
    }
}
