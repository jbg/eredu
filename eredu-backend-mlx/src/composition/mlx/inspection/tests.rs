use super::*;

struct IndependentMechanisms;

impl eredu_architectures::PreparationMechanismProvider for IndependentMechanisms {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        super::super::structural::preparation_mechanism_capabilities()
    }

    fn supports_grouped_operation(
        &self,
        requirement: eredu_runtime::GroupedOperationRequirement,
    ) -> bool {
        super::super::replicated_text::GROUPED_OPERATION_CAPABILITIES.contains(&requirement)
    }

    fn replicated_text_capabilities(
        &self,
        requirements: &eredu_runtime::ReplicatedTextRequirements,
        request: &eredu_runtime::ReplicatedTextSelectionRequest,
    ) -> eredu_runtime::BackendMechanismCapabilities {
        super::super::replicated_text::capabilities(requirements, request)
    }

    fn processor_capabilities(&self) -> eredu_runtime::MediaPrimitiveCapabilities {
        super::super::processor::capabilities()
    }

    fn speculative_capabilities(&self) -> eredu_runtime::SpeculativeMechanismCapabilities {
        super::super::speculative::speculative_mechanism_capabilities()
    }

    fn communication_capabilities(&self) -> eredu_runtime::CommunicationCapabilities {
        crate::backend::runtime::distributed::topology::mlx_communication_capabilities()
    }
}

fn write_safetensors_fixture(root: &Path) {
    use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "intermediate_size": 32,
        "num_attention_heads": 4,
        "rms_norm_eps": 0.00001,
        "vocab_size": 64
    });
    std::fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::resolve_model_config(&config).unwrap();
    let checkpoint = resolved.architecture.checkpoint();
    let tensors = checkpoint
        .common_tensors
        .iter()
        .chain(
            checkpoint
                .layout_groups
                .iter()
                .filter_map(|group| group.variants.first())
                .flat_map(|variant| variant.tensors.iter()),
        )
        .map(|tensor| {
            let elements = tensor.shape.iter().product::<usize>();
            (
                tensor.key.clone(),
                tensor.shape.clone(),
                vec![0; elements * 4],
            )
        })
        .collect::<Vec<_>>();
    let views = tensors.iter().map(|(name, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        )
    });
    serialize_to_file(views, None, &root.join("model.safetensors")).unwrap();
}

#[test]
fn valid_inspection_never_opens_payloads_or_realizes_native_resources() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    crate::tests::support::path_instrumentation::reset();

    let report = inspect_model(root.path(), MlxInspectionOptions::default()).unwrap();

    assert_eq!(
        report.requested_load,
        eredu_core::InspectionReadiness::Ready
    );
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(counts.payload_opens, 0);
    assert_eq!(counts.architecture_constructions, 0);
    assert_eq!(counts.state_allocations, 0);
    assert_eq!(counts.materializations, 0);
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::communication_realization_attempts(),
        0
    );
}

#[test]
fn retained_inspection_selection_is_consumed_by_source_preparation() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    let outcome = inspect_model_preparation(root.path(), MlxInspectionOptions::default()).unwrap();
    let admission = outcome.selected().unwrap().preparation().admission();

    let sources = eredu_architectures::prepare_inspected_model_sources(outcome)
        .unwrap()
        .unwrap();

    assert_eq!(sources.selected().admission(), admission);
    let diagnostics = sources.complete().source_diagnostics().unwrap();
    assert_eq!(diagnostics.physical_reads, 0);
    assert_eq!(diagnostics.physical_read_bytes, 0);
}

#[test]
fn independent_capability_adapters_produce_the_same_report_and_retained_admission() {
    let root = tempfile::tempdir().unwrap();
    write_safetensors_fixture(root.path());
    let inspection = eredu_architectures::configuration::inspect_artifact(root.path()).unwrap();
    let request = MlxLoadRequest::default();
    let (normalized, _rank) = request.checked_normalized().unwrap();

    let first = eredu_architectures::inspect_selected_model(
        inspection.clone(),
        normalized,
        &preparation_mechanisms(),
        media_feature_availability(),
    );
    let second = eredu_architectures::inspect_selected_model(
        inspection,
        normalized,
        &IndependentMechanisms,
        media_feature_availability(),
    );

    assert_eq!(first.report(), second.report());
    let first = first.selected().unwrap().preparation();
    let second = second.selected().unwrap().preparation();
    assert_eq!(first.admission(), second.admission());
    assert_eq!(first.session_capabilities(), second.session_capabilities());
}

#[test]
fn missing_artifact_is_returned_as_a_total_report() {
    let root = tempfile::tempdir().unwrap();
    let report = inspect_model(
        root.path().join("missing.gguf"),
        MlxInspectionOptions::default(),
    )
    .unwrap();

    assert_eq!(report.container, eredu_core::InspectionReadiness::Missing);
    assert_eq!(
        report.requested_load,
        eredu_core::InspectionReadiness::Missing
    );
}

#[test]
fn unsupported_container_is_returned_as_a_total_report() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("weights.bin");
    std::fs::write(&path, b"not a supported model container").unwrap();

    let report = inspect_model(&path, MlxInspectionOptions::default()).unwrap();

    assert_eq!(
        report.container,
        eredu_core::InspectionReadiness::Unsupported
    );
    assert_eq!(
        report.requested_load,
        eredu_core::InspectionReadiness::Unsupported
    );
}

#[test]
fn invalid_configuration_is_returned_as_a_total_report() {
    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("config.json"), b"{").unwrap();

    let report = inspect_model(root.path(), MlxInspectionOptions::default()).unwrap();

    assert_ne!(
        report.requested_load,
        eredu_core::InspectionReadiness::Ready
    );
    assert!(!report.issues.is_empty());
}
