//! Ordinary metadata selection keeps bound tokenizer policy and exact source roles.
use super::super::registry::Mechanisms;
use super::*;
use eredu_architectures::{
    configuration::inspect_artifact,
    processor_plan::{GgufSpecialTokenIds, GgufSpecialTokenKind, QwenMediaTokenIds},
    select_preparation, PreparationMechanismProvider, PreparationSelectionError,
};
use eredu_core::{GgufCompanionRole, InputModality};
use eredu_runtime::*;

struct NoFacts;
impl PreparationMechanismProvider for NoFacts {
    fn preparation_capabilities(&self) -> eredu_core::PreparationMechanismCapabilities {
        panic!("unresolved IDs queried preparation facts")
    }
    fn supports_grouped_operation(&self, _: GroupedOperationRequirement) -> bool {
        panic!("unresolved IDs queried grouped facts")
    }
    fn replicated_text_capabilities(
        &self,
        _: &ReplicatedTextRequirements,
        _: &ReplicatedTextSelectionRequest,
    ) -> BackendMechanismCapabilities {
        panic!("unresolved IDs queried text facts")
    }
    fn processor_capabilities(&self) -> MediaPrimitiveCapabilities {
        panic!("unresolved IDs queried media facts")
    }
    fn speculative_capabilities(&self) -> SpeculativeMechanismCapabilities {
        panic!("unresolved IDs queried prediction facts")
    }
    fn communication_capabilities(&self) -> CommunicationCapabilities {
        panic!("unresolved IDs queried communication facts")
    }
}

#[test]
fn gguf_ordinary_media_registry_selects_from_retained_headers_and_separates_source_roles() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    let projector = directory.path().join("mmproj-qwen4.gguf");
    write(&projector, false, false, 32, false);
    let mut inspection = inspect_artifact(&fixtures.gguf_path).unwrap();
    let policy = MediaExecutionPolicy::new(
        ProcessorSelectionRequest::new([
            InputModality::Text,
            InputModality::Image,
            InputModality::Video,
        ])
        .with_projected_embeddings(true),
    )
    .unwrap();
    let request = super::super::load_policy::request(LayerWeightResidency::FullyResident)
        .with_media_execution(MediaLoadRequest::Required(policy.clone()));
    assert!(matches!(
        select_preparation(&inspection, &request, &NoFacts),
        Err(PreparationSelectionError::UnresolvedGgufSpecialTokens(
            GgufSpecialTokenKind::Qwen
        ))
    ));
    let ids = GgufSpecialTokenIds::Qwen(QwenMediaTokenIds {
        image_token_id: media().image,
        video_token_id: media().video,
        vision_start_token_id: media().start,
        vision_end_token_id: media().end,
    });
    inspection
        .architecture_plan_mut()
        .bind_gguf_special_token_ids(ids)
        .unwrap();
    let independent = inspect_artifact(&fixtures.gguf_path).unwrap();
    let hidden_target = fixtures.gguf_path.with_extension("unavailable");
    let hidden_projector = projector.with_extension("unavailable");
    std::fs::rename(&fixtures.gguf_path, &hidden_target).unwrap();
    std::fs::rename(&projector, &hidden_projector).unwrap();
    let mechanisms = Mechanisms::default();
    let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
    assert!(selected.execution().processor().is_some());
    assert!(selected.allows_media_projector());
    let report = eredu_architectures::model_inspection::inspect_selected_model(
        inspection.clone(),
        &request,
        &mechanisms,
        eredu_core::MediaFeatureAvailability {
            image: true,
            audio: false,
        },
    );
    assert!(
        report.selected().is_some(),
        "conditional cold report failed: {:?}",
        report.report()
    );
    assert_eq!(
        report.report().expected_modalities,
        vec![
            eredu_core::inspection::ArtifactModality::Text,
            eredu_core::inspection::ArtifactModality::Image,
            eredu_core::inspection::ArtifactModality::Video,
        ]
    );
    let missing_rows = Mechanisms {
        rows: false,
        ..Default::default()
    };
    assert!(select_preparation(&inspection, &request, &missing_rows).is_err());
    assert!(missing_rows.row_queries.get() > 0);
    let changed = request
        .clone()
        .with_media_execution(MediaLoadRequest::Required(
            MediaExecutionPolicy::new(policy.processor().clone().with_raw_media(true)).unwrap(),
        ));
    assert!(
        select_preparation(&inspection, &changed, &mechanisms).is_err(),
        "cached prepared-only selection must not authorize unsupported raw media"
    );
    let raw_mechanisms = Mechanisms {
        raw_media: true,
        ..Default::default()
    };
    let raw_without_limits = select_preparation(&inspection, &changed, &raw_mechanisms).unwrap();
    assert!(raw_without_limits
        .execution()
        .processor()
        .unwrap()
        .raw_media());
    // Explicit zero budgets admit readiness but deny execution. Selection is cold:
    // target and projector payload paths are still unavailable here.
    let MediaLoadRequest::Required(raw_policy) = changed.media_execution() else {
        unreachable!()
    };
    let raw_policy =
        raw_policy
            .clone()
            .with_processor_budget(processor_resources::ProcessorRequestBudget {
                decoded_input_bytes: 0,
                host_buffer_bytes: 0,
                output_tensor_bytes: 0,
                planning_items: 0,
                decoder_positions: 0,
            });
    let bounded_raw = changed
        .clone()
        .with_media_execution(MediaLoadRequest::Required(raw_policy));
    let raw = select_preparation(&inspection, &bounded_raw, &raw_mechanisms).unwrap();
    assert!(raw.execution().processor().unwrap().raw_media());
    assert!(select_preparation(&inspection, &changed, &raw_mechanisms)
        .unwrap()
        .execution()
        .processor()
        .unwrap()
        .raw_media());
    assert!(
        select_preparation(&inspection, &bounded_raw, &mechanisms).is_err(),
        "cached raw readiness cannot replace missing primitive facts"
    );
    let restored = select_preparation(&inspection, &request, &mechanisms).unwrap();
    assert_eq!(
        selected.text_realization().requirements(),
        restored.text_realization().requirements()
    );
    let disabled = request
        .clone()
        .with_media_execution(MediaLoadRequest::Disabled);
    assert!(matches!(
        select_preparation(&inspection, &disabled, &mechanisms),
        Err(PreparationSelectionError::MediaDisabledForComposite)
    ));
    let unspecified = request
        .clone()
        .with_media_execution(MediaLoadRequest::ArchitectureDefault);
    select_preparation(&inspection, &unspecified, &mechanisms)
        .expect("an explicit projector needs no custom media policy");
    assert!(matches!(
        select_preparation(&independent, &request, &NoFacts),
        Err(PreparationSelectionError::UnresolvedGgufSpecialTokens(
            GgufSpecialTokenKind::Qwen
        ))
    ));
    std::fs::rename(&hidden_target, &fixtures.gguf_path).unwrap();
    std::fs::rename(&hidden_projector, &projector).unwrap();
    let admitted =
        eredu_core::ModelPreparationPlan::from_retained_admission(inspection, selected.admission())
            .unwrap();
    let sources =
        eredu_architectures::prepared_sources::prepare_model_sources(admitted, selected).unwrap();
    let companion = sources
        .companion(&GgufCompanionRole::MediaProjector)
        .unwrap();
    assert_eq!(sources.companions().count(), 1);
    assert!(sources
        .resolutions()
        .target_companion(&GgufCompanionRole::MediaProjector)
        .is_some());
    assert!(sources
        .primary()
        .source_keys()
        .iter()
        .all(|name| !name.starts_with("model.visual.")));
    assert!(companion
        .source_keys()
        .iter()
        .all(|name| name.starts_with("model.visual.")));
    assert!(sources
        .target()
        .source_keys()
        .iter()
        .any(|name| name.starts_with("model.visual.")));
    assert_eq!(companion.source_diagnostics().unwrap().physical_reads, 0);
    for key in companion.source_keys() {
        let provenance = companion.source_provenance(&key).unwrap();
        assert_eq!(
            provenance.backing_shard.as_deref(),
            Some(projector.as_path())
        );
        assert_eq!(
            sources.target().source_provenance(&key).unwrap(),
            provenance
        );
    }
    assert!(sources.extension().is_none());
}

#[test]
fn safetensors_declared_vision_is_selected_by_default_and_can_be_disabled() {
    let directory = tempfile::tempdir().unwrap();
    let fixtures = eredu_evaluation::qwen4_exp::PreparedFixtures::write(directory.path()).unwrap();
    eredu_evaluation::qwen4_exp::add_vision_weights(&fixtures.safetensors_path, 2);
    let inspection = inspect_artifact(&fixtures.safetensors_path).unwrap();
    let request = NormalizedLoadRequest::default();
    let mechanisms = Mechanisms::default();
    let selected = select_preparation(&inspection, &request, &mechanisms).unwrap();
    assert!(selected.execution().processor().is_some());
    let selected = select_preparation(
        &inspection,
        &request.with_media_execution(MediaLoadRequest::Disabled),
        &mechanisms,
    )
    .unwrap();
    assert!(selected.execution().processor().is_none());
}
