//! Cold distributed predictor admission, placement, and retained companion authority.
use super::*;
use eredu_architectures::{
    prepared_sources::prepare_model_sources,
    select_preparation,
    selected_execution::{
        SelectedCompositePartitionedExecution, SelectedDensePartitionedExecution,
        SelectedExecutionDispatcher, SelectedRoutedPartitionedExecution,
    },
};
use eredu_runtime::*;

struct Ownership;
impl SelectedExecutionDispatcher for Ownership {
    type Output = PartitionOwnership;
    type Error = String;
    fn replicated(self, _: SelectedReplicatedTextRealization) -> Result<Self::Output, String> {
        panic!("expected routed partition")
    }
    fn routed(
        self,
        _: eredu_architectures::SelectedRoutedTextRealization,
    ) -> Result<Self::Output, String> {
        panic!("expected routed partition")
    }
    fn composite(
        self,
        _: eredu_architectures::replicated_text::SelectedCompositeTextRealization,
    ) -> Result<Self::Output, String> {
        panic!("expected routed partition")
    }
    fn partitioned_dense(
        self,
        _: SelectedDensePartitionedExecution,
    ) -> Result<Self::Output, String> {
        panic!("expected routed partition")
    }
    fn partitioned_routed(
        self,
        selected: SelectedRoutedPartitionedExecution,
    ) -> Result<Self::Output, String> {
        Ok(selected.requirements().ownership().clone())
    }
    fn partitioned_composite(
        self,
        _: SelectedCompositePartitionedExecution,
    ) -> Result<Self::Output, String> {
        panic!("expected routed partition")
    }
}
fn request(rank: usize) -> NormalizedLoadRequest {
    super::super::partition_selection::request(2, 2, 1, rank)
        .with_drafting(DraftingLoadRequest::embedded(1).unwrap())
}
fn mechanisms() -> super::super::registry::Mechanisms {
    super::super::registry::Mechanisms {
        prediction: true,
        ..Default::default()
    }
}

#[test]
fn qwen4_partitioned_prediction_cold_state_vocabulary_and_missing_mechanisms() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let inspection =
        eredu_architectures::configuration::inspect_artifact(&fixtures.safetensors_path).unwrap();
    // All cold ranks must work from the retained headers, without reading weights.
    let payload = fixtures.safetensors_path.join("model.safetensors");
    let hidden = fixtures.safetensors_path.join("unavailable.safetensors");
    std::fs::rename(&payload, &hidden).unwrap();
    let selections = (0..4)
        .map(|rank| select_preparation(&inspection, &request(rank), &mechanisms()))
        .collect::<Vec<_>>();
    std::fs::rename(&hidden, &payload).unwrap();
    for (rank, selected) in selections.into_iter().enumerate() {
        let selected = selected.unwrap();
        let topology = selected.embedded_prediction_topology().unwrap().unwrap();
        assert_eq!(topology.state.len(), 1);
        assert!(
            matches!(&topology.state[0].policy,
                eredu_core::cache::LayerCachePolicy::KeyValueWithState { num_key_value_heads, head_dim, .. }
                    if num_key_value_heads.get() == 1 && head_dim.get() == 8
            ),
            "TP-local prediction state on rank {rank}"
        );
        assert_eq!(
            &topology.target_features.entries()[0].shape()[2..],
            &[2, 32]
        );
        let ownership = selected.execution().clone().dispatch(Ownership).unwrap();
        for role in ["embedding", "output"] {
            assert!(
                ownership
                    .replicated_static_roles()
                    .iter()
                    .any(|actual| actual == role),
                "prediction vocabulary {role} must remain available on pipeline rank {rank}"
            );
        }
    }
    let inspected = eredu_architectures::model_inspection::inspect_selected_model(
        inspection.clone(),
        &request(3),
        &mechanisms(),
        eredu_core::MediaFeatureAvailability {
            image: false,
            audio: false,
        },
    );
    let retained = inspected
        .selected()
        .expect("distributed MTP inspection remains selectable");
    assert_eq!(
        inspected.report().resources.embedded_draft_capacity.value(),
        Some(&1)
    );
    let prediction = retained.preparation().prediction_realization().unwrap();
    for mechanism in [
        SpeculativeMechanism::Communication,
        SpeculativeMechanism::StateStorage,
    ] {
        assert!(prediction
            .requirements()
            .mechanisms()
            .mechanisms()
            .contains(&mechanism));
    }
    let topology = retained
        .preparation()
        .embedded_prediction_topology()
        .unwrap()
        .unwrap();
    assert!(matches!(&topology.state[0].policy,
        eredu_core::cache::LayerCachePolicy::KeyValueWithState { num_key_value_heads, .. }
            if num_key_value_heads.get() == 1
    ));
    let with_prediction = retained
        .preparation()
        .execution()
        .parameter_resources()
        .unwrap();
    assert_eq!(
        inspected
            .report()
            .resources
            .materialized_parameter_bytes
            .value(),
        Some(&with_prediction.parameter_bytes)
    );
    // This report is explicitly unsharded; verify the dominant MTP weights are
    // included without manufacturing a new per-allocation or per-rank estimate.
    let target_only = select_preparation(
        &inspection,
        &request(3).with_drafting(DraftingLoadRequest::Disabled),
        &mechanisms(),
    )
    .unwrap();
    let target_only = target_only.execution().parameter_resources().unwrap();
    assert!(with_prediction.parameter_bytes > target_only.parameter_bytes);
    assert!(with_prediction.expert_bytes > target_only.expert_bytes);
    for missing in [
        super::super::registry::Mechanisms {
            prediction_state: false,
            ..mechanisms()
        },
        super::super::registry::Mechanisms {
            communication: false,
            ..mechanisms()
        },
    ] {
        assert!(matches!(
            select_preparation(&inspection, &request(0), &missing),
            Err(eredu_architectures::PreparationSelectionError::Qwen4TargetSelection(_))
        ));
    }
}

#[test]
fn qwen4_partitioned_gguf_prediction_requires_and_retains_companion_authority() {
    use eredu_architectures::qwen4_exp::prepared::{PreparationError, TargetLoadError};
    use eredu_architectures::PreparationSelectionError;
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_prediction_weights(&fixtures.safetensors_path).unwrap();
    let inspection =
        eredu_architectures::configuration::inspect_artifact(&fixtures.gguf_path).unwrap();
    assert!(
        matches!(select_preparation(&inspection, &request(3), &mechanisms()),
        Err(PreparationSelectionError::Qwen4TargetLoad(error))
            if matches!(error.as_ref(), TargetLoadError::Preparation(PreparationError::MissingPrediction)))
    );
    let request = request(3).with_prediction_source(fixtures.safetensors_path.clone());
    let hidden = fixtures.gguf_path.with_extension("unavailable");
    std::fs::rename(&fixtures.gguf_path, &hidden).unwrap();
    let selected = select_preparation(&inspection, &request, &mechanisms());
    std::fs::rename(&hidden, &fixtures.gguf_path).unwrap();
    assert!(
        selected.is_ok(),
        "GGUF cold selection must retain the original header authority"
    );
    // Reinspect the restored file identity before testing ordinary role binding.
    let inspection =
        eredu_architectures::configuration::inspect_artifact(&fixtures.gguf_path).unwrap();
    let selected = select_preparation(&inspection, &request, &mechanisms()).unwrap();
    let topology = selected.embedded_prediction_topology().unwrap().unwrap();
    let admission =
        eredu_core::ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
    let sources = prepare_model_sources(admission, selected).unwrap();
    let prediction = sources
        .extension()
        .expect("retained separate prediction role");
    assert_eq!(prediction.source_diagnostics().unwrap().physical_reads, 0);
    assert!(prediction
        .source_keys()
        .iter()
        .all(|key| key.starts_with("mtp.")));
    assert!(sources
        .target()
        .source_metadata("mtp.fc_hidden.weight")
        .is_err());
    assert_eq!(
        prediction
            .source_provenance("mtp.fc_hidden.weight")
            .unwrap()
            .backing_shard,
        Some(
            fixtures
                .safetensors_path
                .join("model.safetensors")
                .canonicalize()
                .unwrap()
        )
    );
    for (name, physical) in [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("lm_head.weight", "output.weight"),
    ] {
        let provenance = sources.target().source_provenance(name).unwrap();
        assert_eq!(provenance.physical_tensor, physical);
        assert_eq!(
            provenance
                .backing_shard
                .map(|path| path.canonicalize().unwrap()),
            Some(fixtures.gguf_path.canonicalize().unwrap())
        );
        assert!(prediction.source_metadata(name).is_err());
    }
    assert_eq!(
        sources
            .prepare_discovery(Default::default(), Default::default())
            .embedded_prediction_topology()
            .unwrap(),
        Some(&topology)
    );
}
