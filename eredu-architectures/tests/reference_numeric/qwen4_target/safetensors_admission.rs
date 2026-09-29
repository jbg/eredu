//! Header admission must finish before reading exact integer controls.
use eredu_architectures::qwen4_exp::{
    checkpoint::schema::SafetensorsEncoding,
    config::Config,
    prepared::{PreparationError, PreparedTarget, SafetensorsTargetPlan},
};
use eredu_checkpoint::{
    store::{
        CheckpointLease, CheckpointSource, SafetensorsWeightStore, SharedCheckpointSource,
        StoreError, TensorMetadata, TensorReadRequest, TensorSourceProvenance,
        WeightStoreDiagnostics,
    },
    validation::{CatalogTensorMetadata, SafetensorsCatalog},
};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

struct Header(BTreeMap<String, CatalogTensorMetadata>);
impl Header {
    fn from_source(source: &dyn CheckpointSource) -> Self {
        Self(
            source
                .source_keys()
                .into_iter()
                .map(|key| {
                    let metadata = source.source_metadata(&key).unwrap();
                    (
                        key,
                        CatalogTensorMetadata {
                            shape: metadata.logical_shape,
                            stored_dtype: metadata.stored_dtype,
                        },
                    )
                })
                .collect(),
        )
    }
}
impl SafetensorsCatalog for Header {
    fn keys(&self) -> Vec<String> {
        self.0.keys().cloned().collect()
    }
    fn metadata(&self, key: &str) -> Result<CatalogTensorMetadata, String> {
        self.0
            .get(key)
            .cloned()
            .ok_or_else(|| format!("missing {key}"))
    }
}
#[derive(Clone, Copy)]
enum Change {
    None,
    Shape,
    PhysicalShape,
    ByteLength,
    Alias,
    ControlProvenance,
    TableProvenance,
}
struct CountedSource {
    inner: SharedCheckpointSource,
    change: Change,
    reads: Mutex<Vec<String>>,
}
const EMBEDDING: &str = "model.embed_tokens.weight";
const ALIAS: &str = "model.language_model.embed_tokens.weight";
impl CountedSource {
    fn new(inner: SharedCheckpointSource, change: Change) -> Arc<Self> {
        Arc::new(Self {
            inner,
            change,
            reads: Mutex::default(),
        })
    }
    fn original<'a>(&self, key: &'a str) -> &'a str {
        if matches!(self.change, Change::Alias) && key == ALIAS {
            EMBEDDING
        } else {
            key
        }
    }
    fn assert_unread(&self) {
        assert!(self.reads.lock().unwrap().is_empty());
        assert_eq!(self.source_diagnostics().unwrap().physical_reads, 0);
    }
}
impl CheckpointSource for CountedSource {
    fn source_keys(&self) -> Vec<String> {
        self.inner
            .source_keys()
            .into_iter()
            .map(|key| {
                if matches!(self.change, Change::Alias) && key == EMBEDDING {
                    ALIAS.into()
                } else {
                    key
                }
            })
            .collect()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let mut metadata = self.inner.source_metadata(self.original(key))?;
        metadata.name = key.into();
        if matches!(self.change, Change::Shape) && key == EMBEDDING {
            metadata.logical_shape[0] += 1;
        }
        if matches!(self.change, Change::PhysicalShape) && key == EMBEDDING {
            metadata.physical_shape[0] += 1;
        }
        if matches!(self.change, Change::ByteLength) && key == EMBEDDING {
            metadata.encoded_byte_len += 4;
        }
        Ok(metadata)
    }
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        let mut provenance = self.inner.source_provenance(self.original(key))?;
        provenance.catalog_key = key.into();
        if (matches!(self.change, Change::ControlProvenance) && key.ends_with("layer_multipliers"))
            || (matches!(self.change, Change::TableProvenance)
                && key.ends_with("ngram_embedding.shard_0.weight"))
        {
            provenance.backing_shard = Some(std::path::PathBuf::from(
                "different-admitted-shard.safetensors",
            ));
        }
        Ok(provenance)
    }
    fn acquire_lease(&self, mut request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.reads.lock().unwrap().push(request.key.clone());
        request.key = self.original(&request.key).to_owned();
        self.inner.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.inner.source_diagnostics()
    }
}
fn encoding() -> SafetensorsEncoding {
    SafetensorsEncoding::from_json(&serde_json::json!({})).unwrap()
}
pub(super) fn fixture() -> (tempfile::TempDir, Config, SharedCheckpointSource) {
    let directory = tempfile::tempdir().unwrap();
    let fixture = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    let mut config =
        Config::from_gguf(&eredu_gguf::Checkpoint::open(&fixture.gguf_path).unwrap()).unwrap();
    config.ngram.source = eredu_architectures::qwen4_exp::config::NGramSourceLayout::Safetensors {
        vocabulary_base: 5,
        vocabulary_alignment: 1,
        shards: 1,
        seed: 1,
    };
    let source = Arc::new(SafetensorsWeightStore::open(&fixture.safetensors_path).unwrap());
    (directory, config, source)
}

pub(super) fn physical_sources(
    source: &dyn CheckpointSource,
) -> BTreeMap<String, eredu_runtime::ReplicatedTextPhysicalSource> {
    source
        .source_keys()
        .into_iter()
        .map(|key| {
            let metadata = source.source_metadata(&key).unwrap();
            let provenance = source.source_provenance(&key).unwrap();
            let physical = eredu_runtime::ReplicatedTextPhysicalSource::new(
                &key,
                provenance.physical_tensor,
                provenance.backing_shard.unwrap(),
                provenance.output,
                provenance.source_encoding,
                metadata.encoded_byte_len,
            )
            .unwrap();
            (key, physical)
        })
        .collect()
}

#[derive(Default)]
struct CountingSupport(std::cell::Cell<usize>);
impl eredu_runtime::RowLookupMechanismSupport for CountingSupport {
    fn storage(&self) -> Option<eredu_runtime::AddressableStorageCapabilities> {
        self.0.set(self.0.get() + 1);
        eredu_runtime::RowLookupMechanismSupport::storage(&super::cold::Support)
    }
    fn workspace(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<Option<eredu_runtime::RowLookupWorkspace>, eredu_runtime::RowLookupError> {
        self.0.set(self.0.get() + 1);
        eredu_runtime::RowLookupMechanismSupport::workspace(&super::cold::Support, descriptor)
    }
    fn decode_memory(
        &self,
        descriptor: &eredu_runtime::RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_runtime::RowLookupError>
    {
        self.0.set(self.0.get() + 1);
        eredu_runtime::RowLookupMechanismSupport::decode_memory(&super::cold::Support, descriptor)
    }
}

#[test]
fn safetensors_target_header_rejects_ordinary_catalog_before_integer_reads() {
    let (_directory, config, source) = fixture();
    let counted = CountedSource::new(source, Change::None);
    let header = Header::from_source(counted.as_ref());
    let plan = SafetensorsTargetPlan::prepare(&header, config.clone(), encoding()).unwrap();
    assert!(!plan.resolution().source_keys().is_empty());
    let cold = plan
        .target_spec(eredu_evaluation::qwen4_exp::limits())
        .unwrap();
    let cold_layout = cold.state_layout().unwrap();
    let cold_identity = cold.geometry_fingerprint();
    let row_limits = eredu_runtime::RowLookupLimits {
        requests: 128,
        rows_per_acquisition: 2,
        acquisition_bytes: 128,
        host_bytes: 1 << 16,
        output_bytes: 32768,
    };
    let descriptors = plan
        .row_descriptors(
            eredu_evaluation::qwen4_exp::limits(),
            row_limits,
            eredu_core::residency::ResidencyPolicy::Cacheable,
        )
        .unwrap();
    let support = CountingSupport::default();
    let row_admission = eredu_runtime::SelectedRowLookupPlans::select(
        descriptors,
        eredu_runtime::ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(4096), Some(0), 1).unwrap(),
            1 << 20,
            1 << 20,
        )
        .unwrap(),
        0,
        &support,
    )
    .unwrap();
    let cold_queries = support.0.get();
    assert!(cold_queries > 0);
    let cold_execution = plan
        .clone()
        .execution_plan(
            eredu_evaluation::qwen4_exp::limits(),
            super::stream_bindings(&cold),
            row_admission.clone(),
            physical_sources(counted.as_ref()),
        )
        .unwrap();
    assert_eq!(
        cold_execution.requirements().text().architecture_identity(),
        cold_identity
    );
    assert_eq!(
        cold_execution.requirements().text().state_layout(),
        &cold_layout
    );
    assert_eq!(
        cold_execution.requirements().row_lookups(),
        Some(&row_admission)
    );
    let capabilities = super::cold::capabilities(cold_execution.requirements(), None);
    let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
        eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        )
        .with_session(eredu_core::SessionCapabilities::new(false, true, true))
        .with_exact_completion(true),
        eredu_runtime::WeightResidency::fully_resident(),
    )
    .unwrap();
    let selected_cold = cold_execution
        .clone()
        .select(&request, &capabilities, None)
        .unwrap();
    for residency in [
        eredu_runtime::LayerWeightResidency::LayerwiseHost(
            eredu_runtime::LayerwiseLoadOptions::new(
                eredu_core::residency::OffloadConfig::new(Some(1 << 20), Some(1 << 20), 1).unwrap(),
            ),
        ),
        eredu_runtime::LayerWeightResidency::DenseDiskStream(
            eredu_runtime::DenseDiskStreamLoadOptions::new(1 << 20, 0, 0, 0).unwrap(),
        ),
    ] {
        let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
            eredu_runtime::ReplicatedTextSelectionRequest::new(
                residency,
                eredu_runtime::CacheResidencyPolicy::Device,
            )
            .with_session(eredu_core::SessionCapabilities::new(false, true, true))
            .with_exact_completion(true),
            eredu_runtime::WeightResidency::with_layers(residency),
        )
        .unwrap();
        cold_execution
            .clone()
            .select(&request, &capabilities, None)
            .unwrap();
    }
    assert!(cold_execution
        .clone()
        .select(
            &request,
            &capabilities.clone().with_exact_completion(false),
            None
        )
        .is_err());
    counted.assert_unread();
    assert_eq!(
        support.0.get(),
        cold_queries,
        "complete cold selection retains row admission"
    );
    assert_eq!(cold.embedding.dimensions, config.hidden_size);
    assert_eq!(
        cold.units.len(),
        config.layers.len() + config.ngram.layers.len()
    );
    counted.assert_unread();
    for missing in [false, true] {
        let mut malformed = Header::from_source(counted.as_ref());
        if missing {
            malformed.0.remove(EMBEDDING);
        } else {
            malformed.0.get_mut(EMBEDDING).unwrap().shape[0] += 1;
        }
        assert!(SafetensorsTargetPlan::prepare(&malformed, config.clone(), encoding()).is_err());
        counted.assert_unread();
    }
    let invalid_source = CountedSource::new(counted.inner.clone(), Change::Shape);
    assert!(PreparedTarget::safetensors(
        invalid_source.clone(),
        config.clone(),
        encoding(),
        eredu_evaluation::qwen4_exp::limits(),
    )
    .is_err());
    invalid_source.assert_unread();
    let prepared = plan
        .bind(counted.clone(), eredu_evaluation::qwen4_exp::limits())
        .unwrap();
    assert_eq!(prepared.spec().embedding.dimensions, config.hidden_size);
    assert_eq!(prepared.spec().state_layout().unwrap(), cold_layout);
    assert_eq!(prepared.spec().geometry_fingerprint(), cold_identity);
    let reads = counted.reads.lock().unwrap();
    assert_eq!(reads.len(), 3);
    assert!(reads.iter().all(|key| key.ends_with("layer_multipliers")
        || key.ends_with("ngram_heads_vocab_sizes")
        || key.ends_with("ngram_heads_offsets")));
    drop(reads);
    let before = counted.source_diagnostics().unwrap().physical_reads;
    let bound_execution = prepared
        .execution_plan(super::stream_bindings(&cold), row_admission)
        .unwrap();
    assert_eq!(
        bound_execution.requirements(),
        cold_execution.requirements()
    );
    assert_eq!(counted.source_diagnostics().unwrap().physical_reads, before);
    let selected = selected_cold.bind(counted.clone(), None).unwrap();
    assert_eq!(
        selected.realization().text().requirements(),
        cold_execution.requirements().text()
    );
    assert_eq!(
        support.0.get(),
        cold_queries,
        "binding must consume retained row admission"
    );
    assert_eq!(counted.reads.lock().unwrap().len(), 6);
    assert_eq!(
        counted.source_diagnostics().unwrap().physical_reads,
        before + 3
    );
}

#[test]
fn safetensors_target_binding_rejects_changed_metadata_or_alias_resolution_before_reads() {
    let (_directory, config, source) = fixture();
    let header = Header::from_source(source.as_ref());
    for change in [
        Change::Shape,
        Change::PhysicalShape,
        Change::ByteLength,
        Change::Alias,
    ] {
        let plan = SafetensorsTargetPlan::prepare(&header, config.clone(), encoding()).unwrap();
        let changed = CountedSource::new(source.clone(), change);
        if matches!(change, Change::Alias) {
            let alias_catalog: &dyn CheckpointSource = changed.as_ref();
            assert!(
                SafetensorsTargetPlan::prepare(alias_catalog, config.clone(), encoding()).is_ok()
            );
        }
        assert!(plan
            .bind(changed.clone(), eredu_evaluation::qwen4_exp::limits())
            .is_err());
        changed.assert_unread();
    }
}

#[test]
fn safetensors_complete_cold_plan_rejects_replaced_physical_identity_before_control_reads() {
    let (_directory, config, source) = fixture();
    let header = Header::from_source(source.as_ref());
    let plan = SafetensorsTargetPlan::prepare(&header, config, encoding()).unwrap();
    let limits = eredu_evaluation::qwen4_exp::limits();
    let spec = plan.target_spec(limits).unwrap();
    let rows = eredu_runtime::SelectedRowLookupPlans::select(
        plan.row_descriptors(
            limits,
            eredu_runtime::RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 1 << 16,
                output_bytes: 32768,
            },
            eredu_core::residency::ResidencyPolicy::Cacheable,
        )
        .unwrap(),
        eredu_runtime::ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(4096), Some(0), 1).unwrap(),
            1 << 20,
            1 << 20,
        )
        .unwrap(),
        0,
        &CountingSupport::default(),
    )
    .unwrap();
    let physical = physical_sources(source.as_ref());
    let descriptor = rows.descriptors().entries().values().next().unwrap();
    let mut wrong_bank = descriptor.spec().clone();
    wrong_bank.bank += 1;
    let wrong_descriptor = eredu_runtime::RowLookupDescriptor::new(
        descriptor.range().clone(),
        descriptor.metadata().clone(),
        wrong_bank,
        descriptor.scale().cloned(),
        descriptor.limits(),
    )
    .unwrap();
    let wrong_rows = eredu_runtime::SelectedRowLookupPlans::select(
        eredu_runtime::RowLookupDescriptors::new([wrong_descriptor], spec.units.len()).unwrap(),
        rows.options(),
        0,
        &CountingSupport::default(),
    )
    .unwrap();
    assert!(
        plan.execution_plan(
            limits,
            super::stream_bindings(&spec),
            wrong_rows,
            physical.clone(),
        )
        .is_err(),
        "complete cold plan must reject a different lexical bank identity"
    );
    assert_eq!(source.source_diagnostics().unwrap().physical_reads, 0);
    let mut missing = physical.clone();
    missing.remove(EMBEDDING);
    assert!(plan
        .clone()
        .execution_plan(limits, super::stream_bindings(&spec), rows.clone(), missing)
        .is_err());
    let cold = plan
        .execution_plan(limits, super::stream_bindings(&spec), rows, physical)
        .unwrap();
    let capabilities = super::cold::capabilities(cold.requirements(), None);
    let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
        eredu_runtime::ReplicatedTextSelectionRequest::new(
            eredu_runtime::LayerWeightResidency::FullyResident,
            eredu_runtime::CacheResidencyPolicy::Device,
        )
        .with_session(eredu_core::SessionCapabilities::new(false, true, true))
        .with_exact_completion(true),
        eredu_runtime::WeightResidency::fully_resident(),
    )
    .unwrap();
    let selected = cold.select(&request, &capabilities, None).unwrap();
    for change in [
        Change::ControlProvenance,
        Change::TableProvenance,
        Change::ByteLength,
        Change::PhysicalShape,
        Change::Alias,
    ] {
        let changed = CountedSource::new(source.clone(), change);
        assert!(selected.clone().bind(changed.clone(), None).is_err());
        changed.assert_unread();
    }
}

#[test]
fn safetensors_target_literal_hash_failure_occurs_only_during_binding() {
    let (directory, config, _) = fixture();
    let bytes = std::fs::read(directory.path().join("safetensors/model.safetensors")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let invalid_offsets = [0i64, 6]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    let owned_values = tensors
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            let data = if name.ends_with("ngram_heads_offsets") {
                invalid_offsets.clone()
            } else {
                tensor.data().to_vec()
            };
            (name, tensor.dtype(), tensor.shape().to_vec(), data)
        })
        .collect::<Vec<_>>();
    let values = owned_values.iter().map(|(name, dtype, shape, bytes)| {
        (
            name.as_str(),
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
        )
    });
    let invalid_path = directory.path().join("invalid-controls");
    std::fs::create_dir(&invalid_path).unwrap();
    safetensors::tensor::serialize_to_file(values, None, &invalid_path.join("model.safetensors"))
        .unwrap();
    let source = CountedSource::new(
        Arc::new(SafetensorsWeightStore::open(&invalid_path).unwrap()),
        Change::None,
    );
    let header = Header::from_source(source.as_ref());
    let plan = SafetensorsTargetPlan::prepare(&header, config, encoding()).unwrap();
    source.assert_unread();
    assert!(plan
        .bind(source.clone(), eredu_evaluation::qwen4_exp::limits())
        .is_err());
    let reads = source.reads.lock().unwrap();
    assert!(reads.iter().any(|key| key.ends_with("ngram_heads_offsets")));
    assert!(reads.iter().all(|key| key.ends_with("layer_multipliers")
        || key.ends_with("ngram_heads_vocab_sizes")
        || key.ends_with("ngram_heads_offsets")));
}

#[test]
fn safetensors_target_binds_claimed_view_without_unused_rotary_buffers() {
    let (_directory, config, source) = fixture();
    let counted = CountedSource::new(source, Change::None);
    let mut header = Header::from_source(counted.as_ref());
    let unused = "model.rotary_emb.inv_freq";
    header.0.insert(
        unused.into(),
        CatalogTensorMetadata {
            shape: vec![4],
            stored_dtype: eredu_checkpoint::StoredDtype::F32,
        },
    );
    let plan = SafetensorsTargetPlan::prepare(&header, config, encoding()).unwrap();
    assert!(!plan.resolution().source_keys().contains(unused));
    let resolved: SharedCheckpointSource =
        Arc::new(eredu_checkpoint::store::ResolvedCheckpointSource::new(
            counted.clone(),
            plan.resolution().clone(),
        ));
    assert!(resolved.source_metadata(unused).is_err());
    counted.assert_unread();
    let prepared = plan
        .bind(resolved, eredu_evaluation::qwen4_exp::limits())
        .unwrap();
    assert!(prepared.artifact().source_metadata(unused).is_err());
    assert_eq!(counted.reads.lock().unwrap().len(), 3);
}

#[test]
fn safetensors_target_rejects_invalid_execution_limits_before_source_reads() {
    let (_directory, config, source) = fixture();
    let counted = CountedSource::new(source, Change::None);
    let header = Header::from_source(counted.as_ref());
    let valid = eredu_evaluation::qwen4_exp::limits();
    let mut short_history = valid;
    short_history.history_tokens = short_history.qsa.tokens - 1;
    let mut zero_batch = valid;
    zero_batch.qsa.batch = 0;
    let mut short_lookup = valid;
    short_lookup.lookup_rows = short_lookup.invocation_tokens
        * (config.ngram.order - 1) as usize
        * config.ngram.heads as usize
        - 1;
    let mut oversized_lookup = valid;
    oversized_lookup.lookup_rows = i32::MAX as usize + 1;
    let mut zero_tile = valid;
    zero_tile.tile_blocks = 0;
    let mut no_selection_workspace = valid;
    no_selection_workspace.selection_workspace = 0;
    for invalid in [
        short_history,
        zero_batch,
        short_lookup,
        oversized_lookup,
        zero_tile,
        no_selection_workspace,
    ] {
        let plan = SafetensorsTargetPlan::prepare(&header, config.clone(), encoding()).unwrap();
        assert!(matches!(
            plan.bind(counted.clone(), invalid),
            Err(PreparationError::Neural(_))
        ));
        counted.assert_unread();
    }
}

#[test]
fn safetensors_identical_headers_with_distinct_valid_literals_bind_distinct_state_identity() {
    let (directory, config, original) = fixture();
    let bytes = std::fs::read(directory.path().join("safetensors/model.safetensors")).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let alternate = [1i64, 9007199254740995, 37]
        .into_iter()
        .flat_map(i64::to_le_bytes)
        .collect::<Vec<_>>();
    let owned = tensors
        .tensors()
        .into_iter()
        .map(|(name, tensor)| {
            let data = if name.ends_with("layer_multipliers") {
                alternate.clone()
            } else {
                tensor.data().to_vec()
            };
            (name, tensor.dtype(), tensor.shape().to_vec(), data)
        })
        .collect::<Vec<_>>();
    let views = owned.iter().map(|(name, dtype, shape, data)| {
        (
            name.as_str(),
            safetensors::tensor::TensorView::new(*dtype, shape.clone(), data).unwrap(),
        )
    });
    let path = directory.path().join("alternate-valid-hash");
    std::fs::create_dir(&path).unwrap();
    safetensors::tensor::serialize_to_file(views, None, &path.join("model.safetensors")).unwrap();
    let first = CountedSource::new(original, Change::None);
    let second = CountedSource::new(
        Arc::new(SafetensorsWeightStore::open(&path).unwrap()),
        Change::None,
    );
    let first_header = Header::from_source(first.as_ref());
    let second_header = Header::from_source(second.as_ref());
    assert_eq!(first_header.0, second_header.0);
    let first_plan =
        SafetensorsTargetPlan::prepare(&first_header, config.clone(), encoding()).unwrap();
    let second_plan = SafetensorsTargetPlan::prepare(&second_header, config, encoding()).unwrap();
    let limits = eredu_evaluation::qwen4_exp::limits();
    let first_geometry = first_plan.target_spec(limits).unwrap();
    let second_geometry = second_plan.target_spec(limits).unwrap();
    assert_eq!(
        first_geometry.geometry_fingerprint(),
        second_geometry.geometry_fingerprint()
    );
    assert_eq!(
        first_geometry.state_layout().unwrap(),
        second_geometry.state_layout().unwrap()
    );
    first.assert_unread();
    second.assert_unread();
    let first_target = first_plan.bind(first, limits).unwrap();
    let second_target = second_plan.bind(second, limits).unwrap();
    assert_ne!(
        first_target.bound_spec().unwrap().state_fingerprint(),
        second_target.bound_spec().unwrap().state_fingerprint()
    );
    let ids = [3u64, 4, 6, 9];
    let first_rows = first_target.tables()[&0]
        .hash
        .select(Some(&ids), 1, 4, &[0, 0], 8)
        .unwrap();
    let second_rows = second_target.tables()[&0]
        .hash
        .select(Some(&ids), 1, 4, &[0, 0], 8)
        .unwrap();
    assert_eq!(first_rows.rows, [3, 8, 3, 10, 0, 11, 2, 9]);
    assert_eq!(second_rows.rows, [3, 8, 4, 9, 3, 10, 4, 11]);
}
