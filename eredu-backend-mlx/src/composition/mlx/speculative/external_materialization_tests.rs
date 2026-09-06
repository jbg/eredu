use eredu_architectures::{
    gemma4, ExternalAssistantArchitecture, ExternalAssistantTargetProfile,
    MaterializedExternalAssistantVisitor,
};
use eredu_checkpoint::schema::StoredDtypeConstraint;
use safemlx::{Device, DeviceType, Stream};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

use super::{MlxAssistantPreparationVisitor, MlxExternalAssistant};

pub(super) const ASSISTANT_CONFIG: &str = r#"{
  "model_type":"gemma4_assistant","backbone_hidden_size":32,
  "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
  "text_config":{"model_type":"gemma4_text","hidden_size":32,
    "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
    "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
    "vocab_size":32,"max_position_embeddings":128,"tie_word_embeddings":false,
    "attention_k_eq_v":false,"layer_types":["full_attention"]}
}"#;

pub(super) fn assistant_artifact() -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("config.json"), ASSISTANT_CONFIG).unwrap();
    let config = gemma4::AssistantConfig::from_json(ASSISTANT_CONFIG.as_bytes()).unwrap();
    let plan = gemma4::assistant_safetensors_plan(&config).unwrap();
    assert!(plan.layout_groups.is_empty());
    let tensors = plan
        .common_tensors
        .iter()
        .map(|tensor| {
            assert_eq!(tensor.dtype, StoredDtypeConstraint::Floating);
            let bytes = vec![0; tensor.shape.iter().product::<usize>() * 4];
            (tensor.key.clone(), tensor.shape.clone(), bytes)
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
    directory
}

fn target_profile() -> ExternalAssistantTargetProfile {
    let assistant = gemma4::AssistantConfig::from_json(ASSISTANT_CONFIG.as_bytes()).unwrap();
    let mut text = assistant.text_config;
    let mut publisher = *text.layer_schedule.get(0).unwrap();
    publisher.key_value = eredu_nn::AttentionStateSource::Publish {
        value: eredu_nn::AttentionValueSource::Projected,
    };
    text.layer_schedule = eredu_core::LayerSchedule::new(1, vec![publisher]).unwrap();
    ExternalAssistantTargetProfile::Gemma4(gemma4::FamilyConfig {
        model_type: "gemma4".into(),
        text,
        vision: None,
        image_token_id: None,
        video_token_id: None,
        audio: None,
        audio_token_id: None,
    })
}

fn visitor(stream: &Stream) -> MlxAssistantPreparationVisitor {
    MlxAssistantPreparationVisitor {
        stream: stream.clone(),
        weights_stream: stream.clone(),
    }
}

fn selected(
    preparation: eredu_architectures::ExternalAssistantPreparation,
    quantization: Option<eredu_core::QuantizationRequest>,
) -> eredu_architectures::SelectedExternalAssistantPreparation {
    preparation
        .select_materialization(
            quantization,
            eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS,
            |descriptor, transforms| {
                if transforms && super::super::replicated_text::supports_transform(descriptor) {
                    Some(eredu_runtime::WeightLoweringKind::Transform)
                } else if !transforms && super::super::replicated_text::supports_direct(descriptor)
                {
                    Some(eredu_runtime::WeightLoweringKind::Direct)
                } else {
                    None
                }
            },
        )
        .unwrap()
}

struct InspectMaterialized;

impl MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor> for InspectMaterialized {
    type Output = (String, Option<eredu_checkpoint::WeightQuantization>);

    fn visit<A: ExternalAssistantArchitecture>(
        self,
        assistant: &mut MlxExternalAssistant<A>,
    ) -> Self::Output {
        (
            A::configuration_model_type(&assistant.config).to_owned(),
            A::quantization(&assistant.config),
        )
    }
}

#[test]
fn family_blind_mlx_materializer_revalidates_catalog_without_reloading_target() {
    let artifact = assistant_artifact();
    let compatible = selected(
        eredu_architectures::prepare_external_assistant(artifact.path()).unwrap(),
        None,
    )
    .prove_target_compatibility(&target_profile())
    .unwrap();
    let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
    crate::tests::support::path_instrumentation::reset();

    let prepared = compatible
        .prepare_source(eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS)
        .unwrap();
    let mut materialized = prepared.visit(visitor(&stream)).unwrap();
    assert_eq!(
        materialized.visit(InspectMaterialized),
        ("gemma4_assistant".into(), None)
    );
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(counts.payload_opens, 0);
    assert_eq!(counts.constructors, 0);
}

#[test]
fn family_blind_mlx_materializer_applies_architecture_load_time_format() {
    let artifact = assistant_artifact();
    let compatible = selected(
        eredu_architectures::prepare_external_assistant(artifact.path()).unwrap(),
        Some(eredu_core::QuantizationRequest::MxFp4),
    )
    .prove_target_compatibility(&target_profile())
    .unwrap();
    let stream = Stream::try_new_with_device(&Device::new(DeviceType::Cpu, 0)).unwrap();
    crate::tests::support::path_instrumentation::reset();

    let prepared = compatible
        .prepare_source(eredu_checkpoint::store::DEFAULT_MAX_CACHED_SHARDS)
        .unwrap();
    let mut materialized = prepared.visit(visitor(&stream)).unwrap();
    assert_eq!(
        materialized.visit(InspectMaterialized),
        (
            "gemma4_assistant".into(),
            Some(eredu_checkpoint::WeightQuantization::MxFp4)
        )
    );
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(counts.payload_opens, 0);
    assert_eq!(counts.constructors, 0);
}

#[test]
fn unsupported_target_lowering_fails_before_native_resources() {
    let plan = eredu_core::ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
    )
    .with_weight_transformation(eredu_core::WeightTransformationPlan::Affine {
        bits: 4,
        group_size: 32_768,
    })
    .with_drafting(eredu_core::DraftingPlan::External {
        model: "unopened-assistant".into(),
        placement: eredu_core::DraftPlacementPlan::Target,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let target = super::super::replicated_text::tests::tiny_artifact("llama", false);
    let inspection = eredu_architectures::configuration::inspect_artifact(target.path())
        .expect("tiny target inspection");
    let factory = crate::composition::mlx::automatic::MlxBackendFactory::default();
    crate::tests::support::path_instrumentation::reset();

    let error = match eredu_core::select_execution_plan_target(&factory, &plan, inspection) {
        Ok(_) => panic!("invalid target packing geometry unexpectedly selected"),
        Err(error) => error,
    };

    assert!(
        error.to_string().contains("select_model_preparation"),
        "{error}"
    );
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(counts.payload_opens, 0);
    assert_eq!(counts.constructors, 0);
    assert_eq!(counts.materializations, 0);
}

#[test]
fn incompatible_target_assistant_pair_fails_before_native_target_resources() {
    let assistant = assistant_artifact();
    let target = super::super::replicated_text::tests::tiny_artifact("llama", false);
    let plan = eredu_core::ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
    )
    .with_drafting(eredu_core::DraftingPlan::External {
        model: assistant.path().display().to_string(),
        placement: eredu_core::DraftPlacementPlan::Target,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let factory = crate::composition::mlx::automatic::MlxBackendFactory::default();
    let inspection = eredu_architectures::configuration::inspect_artifact(target.path())
        .expect("tiny target inspection");
    let selected_target = eredu_core::select_execution_plan_target(&factory, &plan, inspection)
        .expect("ordinary target selection");
    let preparation = eredu_architectures::prepare_external_assistant(assistant.path())
        .expect("assistant inspection");
    crate::tests::support::path_instrumentation::reset();

    let error = eredu_core::select_execution_plan_drafting(
        &factory,
        &plan,
        &selected_target,
        Some(eredu_core::ExternalDraftArtifact {
            preparation,
            tokenizer_compatibility: eredu_core::TokenizerCompatibilityProof::prove(
                [3; 32], [3; 32],
            )
            .unwrap(),
        }),
    )
    .expect_err("Llama has no external-assistant target contract");

    assert!(error
        .to_string()
        .contains("does not admit an external assistant"));
    assert_eq!(
        crate::tests::support::path_instrumentation::target_native_resource_realization_attempts(),
        0
    );
    let counts = crate::tests::support::path_instrumentation::snapshot();
    assert_eq!(counts.payload_opens, 0);
    assert_eq!(counts.constructors, 0);
    assert_eq!(counts.materializations, 0);
}
