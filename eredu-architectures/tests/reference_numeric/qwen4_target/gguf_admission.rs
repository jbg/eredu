//! Header-only GGUF selection binds exact retained sources without payload reads.
use super::*;
use eredu_architectures::qwen4_exp::prepared::GgufTargetPlan;
use eredu_checkpoint::store::{
    CheckpointLease, CheckpointSource, SharedCheckpointSource, StoreError, TensorMetadata,
    TensorReadRequest, TensorSourceProvenance, WeightStoreDiagnostics,
};
use eredu_runtime::*;
use std::sync::atomic::{AtomicUsize, Ordering};

const EMBEDDING: &str = "model.embed_tokens.weight";
const TABLE: &str = "per_layer_token_embd.weight";

#[derive(Clone, Copy)]
enum Change {
    None,
    Shape,
    PhysicalShape,
    ByteLength,
    PhysicalTensor,
    Output,
    Encoding,
    TableShard,
    Missing,
}
struct CountedSource {
    source: SharedCheckpointSource,
    change: Change,
    reads: AtomicUsize,
}
impl CountedSource {
    fn new(source: SharedCheckpointSource, change: Change) -> Arc<Self> {
        Arc::new(Self {
            source,
            change,
            reads: AtomicUsize::new(0),
        })
    }
    fn assert_unread(&self) {
        assert_eq!(self.reads.load(Ordering::Relaxed), 0);
        assert_eq!(self.source.source_diagnostics().unwrap().physical_reads, 0);
    }
}
impl CheckpointSource for CountedSource {
    fn source_keys(&self) -> Vec<String> {
        self.source
            .source_keys()
            .into_iter()
            .filter(|key| !(matches!(self.change, Change::Missing) && key == EMBEDDING))
            .collect()
    }
    fn source_metadata(&self, key: &str) -> Result<TensorMetadata, StoreError> {
        let mut metadata = self.source.source_metadata(key)?;
        if key == EMBEDDING {
            match self.change {
                Change::Shape => metadata.logical_shape[0] += 1,
                Change::PhysicalShape => metadata.physical_shape[0] += 1,
                Change::ByteLength => metadata.encoded_byte_len += 4,
                _ => (),
            }
        }
        Ok(metadata)
    }
    fn source_provenance(&self, key: &str) -> Result<TensorSourceProvenance, StoreError> {
        let mut provenance = self.source.source_provenance(key)?;
        if key == EMBEDDING {
            match self.change {
                Change::PhysicalTensor => provenance.physical_tensor = "other.weight".into(),
                Change::Output => provenance.output = "other.output".into(),
                Change::Encoding => {
                    provenance.source_encoding = eredu_checkpoint::SourceTensorEncoding::Safetensors(
                        eredu_checkpoint::StoredDtype::F32,
                    )
                }
                _ => (),
            }
        }
        if key == TABLE && matches!(self.change, Change::TableShard) {
            provenance.backing_shard = Some("different-table.gguf".into());
        }
        Ok(provenance)
    }
    fn acquire_lease(&self, request: TensorReadRequest) -> Result<CheckpointLease, StoreError> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.source.acquire_lease(request)
    }
    fn source_diagnostics(&self) -> Result<WeightStoreDiagnostics, StoreError> {
        self.source.source_diagnostics()
    }
}

#[derive(Default)]
struct Support(std::cell::Cell<usize>);
impl RowLookupMechanismSupport for Support {
    fn storage(&self) -> Option<AddressableStorageCapabilities> {
        self.0.set(self.0.get() + 1);
        RowLookupMechanismSupport::storage(&super::cold::Support)
    }
    fn workspace(
        &self,
        descriptor: &RowLookupDescriptor,
    ) -> Result<Option<RowLookupWorkspace>, RowLookupError> {
        self.0.set(self.0.get() + 1);
        RowLookupMechanismSupport::workspace(&super::cold::Support, descriptor)
    }
    fn decode_memory(
        &self,
        descriptor: &RowLookupDescriptor,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, RowLookupError> {
        self.0.set(self.0.get() + 1);
        RowLookupMechanismSupport::decode_memory(&super::cold::Support, descriptor)
    }
}

pub(super) fn cold_plan(
    header: GgufTargetPlan,
) -> eredu_architectures::qwen4_exp::prepared::TargetPreparationPlan {
    cold_plan_with_support(header, &Support::default())
}
fn cold_plan_with_support(
    header: GgufTargetPlan,
    support: &Support,
) -> eredu_architectures::qwen4_exp::prepared::TargetPreparationPlan {
    let limits = eredu_evaluation::qwen4_exp::limits();
    let spec = header.target_spec(limits).unwrap();
    let descriptors = header
        .row_descriptors(
            limits,
            RowLookupLimits {
                requests: 128,
                rows_per_acquisition: 2,
                acquisition_bytes: 128,
                host_bytes: 1 << 16,
                output_bytes: 32768,
            },
            eredu_core::residency::ResidencyPolicy::Cacheable,
        )
        .unwrap();
    let admission = SelectedRowLookupPlans::select(
        descriptors,
        ParameterBankLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(4096), Some(0), 1).unwrap(),
            1 << 20,
            1 << 20,
        )
        .unwrap(),
        0,
        support,
    )
    .unwrap();
    header
        .execution_plan(limits, super::stream_bindings(&spec), admission)
        .unwrap()
}

#[test]
fn gguf_complete_cold_selection_binds_without_reopening_or_reading_payloads() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("cold.gguf");
    super::super::qwen4_gguf::Fixture::new().write(&path);
    let checkpoint = eredu_gguf::Checkpoint::open(&path).unwrap();
    let header = GgufTargetPlan::prepare(&checkpoint).unwrap();
    let source = CountedSource::new(
        super::super::qwen4_gguf::open_text_source(header.text_plan()),
        Change::None,
    );
    let spec = header
        .target_spec(eredu_evaluation::qwen4_exp::limits())
        .unwrap();
    let support = Support::default();
    // Neither complete requirement authoring nor binding may reopen the artifact.
    std::fs::remove_file(&path).unwrap();
    let plan = cold_plan_with_support(header, &support);
    let queries = support.0.get();
    assert!(queries > 0);
    assert_eq!(
        plan.requirements().text().architecture_identity(),
        spec.geometry_fingerprint()
    );
    assert_eq!(
        plan.requirements().text().state_layout(),
        &spec.state_layout().unwrap()
    );
    let capabilities = super::cold::capabilities(plan.requirements(), None);
    for residency in [
        LayerWeightResidency::FullyResident,
        LayerWeightResidency::LayerwiseHost(LayerwiseLoadOptions::new(
            eredu_core::residency::OffloadConfig::new(Some(1 << 24), Some(1 << 24), 1).unwrap(),
        )),
        LayerWeightResidency::DenseDiskStream(
            DenseDiskStreamLoadOptions::new(1 << 24, 0, 0, 0).unwrap(),
        ),
    ] {
        let request = eredu_architectures::routed_text::RoutedTextSelectionRequest::new(
            ReplicatedTextSelectionRequest::new(residency, CacheResidencyPolicy::Device)
                .with_session(eredu_core::SessionCapabilities::new(false, true, true))
                .with_exact_completion(true),
            WeightResidency::with_layers(residency),
        )
        .unwrap();
        let selected = plan.clone().select(&request, &capabilities, None).unwrap();
        let bound = selected.bind(source.clone(), None).unwrap();
        bound
            .prepare::<NumericBackend, State>(&NumericContext::default())
            .unwrap();
        source.assert_unread();
        assert_eq!(
            support.0.get(),
            queries,
            "binding must preserve cold row selection"
        );
    }
}

#[test]
fn gguf_target_binding_rejects_changed_exact_headers_and_provenance_before_reads() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("exact.gguf");
    super::super::qwen4_gguf::Fixture::new().write(&path);
    let header = GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(&path).unwrap()).unwrap();
    let source = super::super::qwen4_gguf::open_text_source(header.text_plan());
    let cold = cold_plan(header.clone());
    for change in [
        Change::Shape,
        Change::PhysicalShape,
        Change::ByteLength,
        Change::PhysicalTensor,
        Change::Output,
        Change::Encoding,
        Change::TableShard,
        Change::Missing,
    ] {
        let altered = CountedSource::new(source.clone(), change);
        assert!(header
            .clone()
            .bind(altered.clone(), eredu_evaluation::qwen4_exp::limits())
            .is_err());
        assert!(cold.clone().bind(altered.clone(), None).is_err());
        altered.assert_unread();
    }
    let second_path = directory.path().join("same-headers-other-file.gguf");
    super::super::qwen4_gguf::Fixture::new().write(&second_path);
    let second =
        GgufTargetPlan::prepare(&eredu_gguf::Checkpoint::open(second_path).unwrap()).unwrap();
    let other = CountedSource::new(
        super::super::qwen4_gguf::open_text_source(second.text_plan()),
        Change::None,
    );
    assert!(cold.bind(other.clone(), None).is_err());
    other.assert_unread();
}
