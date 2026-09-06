use super::*;
use eredu_core::{
    ArtifactInspection, AutomaticPlanningError, DevicePlan, DraftPlacementPlan, DraftingPlan,
    ExecutionPlan, ExternalDraftArtifact, SpeculativeExecutionTopology, WeightTransformationPlan,
};

fn target_inspection(
    vocabulary: usize,
) -> (
    tempfile::TempDir,
    ArtifactInspection<crate::processor_plan::ArtifactArchitecturePlan>,
) {
    let assistant: serde_json::Value = serde_json::from_str(GEMMA_ASSISTANT).unwrap();
    let mut config = serde_json::json!({
        "model_type": "gemma4",
        "tie_word_embeddings": false,
        "text_config": assistant["text_config"],
    });
    config["text_config"]["vocab_size"] = vocabulary.into();
    config["text_config"]["num_hidden_layers"] = 2.into();
    config["text_config"]["num_kv_shared_layers"] = 1.into();
    config["text_config"]["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
    let config = serde_json::to_string(&config).unwrap();
    let family = gemma4::FamilyConfig::from_hf_json(config.as_bytes()).unwrap();
    let schema = gemma4::safetensors_plan(&family).unwrap();
    let root = sparse_safetensors_artifact(&config, &schema);
    let inspection = crate::configuration::inspect_artifact(root.path()).unwrap();
    (root, inspection)
}

fn execution_plan(placement: DraftPlacementPlan, capacity: usize) -> ExecutionPlan {
    ExecutionPlan::fully_resident(DevicePlan::new("foreign", "device:0").unwrap()).with_drafting(
        DraftingPlan::External {
            model: "retained-assistant".into(),
            placement,
            max_draft_tokens: capacity,
            lookahead: false,
            adaptive_lookahead: false,
        },
    )
}

fn artifact(path: &std::path::Path) -> ExternalDraftArtifact<ExternalAssistantPreparation> {
    ExternalDraftArtifact {
        preparation: prepare_external_assistant(path).unwrap(),
        tokenizer_compatibility: TokenizerCompatibilityProof::prove([23; 32], [23; 32]).unwrap(),
    }
}

fn direct_lowering(
    descriptor: &WeightLoweringDescriptor,
    transforms: bool,
) -> Option<WeightLoweringKind> {
    (!transforms
        && matches!(
            descriptor.source(),
            SourceTensorEncoding::Safetensors(StoredDtype::F32)
        )
        && descriptor.executable() == LinearFormat::Dense)
        .then_some(WeightLoweringKind::Direct)
}

// Deliberately independent backend facts, not derived from the requested contract.
fn capabilities() -> SpeculativeMechanismCapabilities {
    SpeculativeMechanismCapabilities::new([
        SpeculativeMechanism::TensorOperations,
        SpeculativeMechanism::NeuralOperations,
        SpeculativeMechanism::PayloadMaterialization,
        SpeculativeMechanism::LogitsProcessing,
        SpeculativeMechanism::Sampling,
        SpeculativeMechanism::Randomness,
        SpeculativeMechanism::StateStorage,
        SpeculativeMechanism::StorageResidency,
        SpeculativeMechanism::ExactCompletion,
        SpeculativeMechanism::Observation,
        SpeculativeMechanism::QueueBinding,
        SpeculativeMechanism::Agreement,
        SpeculativeMechanism::Publication,
        SpeculativeMechanism::SameDeviceHandoff,
        SpeculativeMechanism::CrossDeviceTransfer,
    ])
}

#[test]
fn execution_plan_retains_exact_source_capture_tokenizer_capacity_and_placement() {
    let root = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
    let (_target_root, target) = target_inspection(32);
    for (placement, expected) in [
        (
            DraftPlacementPlan::Target,
            SpeculativeExecutionTopology::Single,
        ),
        (
            DraftPlacementPlan::Device {
                device: DevicePlan::new("foreign", "device:0").unwrap(),
            },
            SpeculativeExecutionTopology::SameDeviceSplit,
        ),
        (
            DraftPlacementPlan::Device {
                device: DevicePlan::new("foreign", "device:1").unwrap(),
            },
            SpeculativeExecutionTopology::CrossDeviceSplit,
        ),
    ] {
        let mut selected = prepare_execution_plan_assistant(
            &execution_plan(placement, 2),
            &target,
            artifact(root.path()),
            direct_lowering,
            &capabilities(),
        )
        .unwrap();
        let requirements = selected.preparation.selected().requirements().clone();
        assert_eq!(
            requirements.strategy().class(),
            SpeculativeStrategyClass::External
        );
        assert_eq!(requirements.strategy().proposal_capacity().get(), 2);
        assert_eq!(
            selected.preparation.selected().placement().topology(),
            expected
        );
        let proof = selected.tokenizer_compatibility;
        // Replacing the outer transport label cannot re-pair the opaque source
        // and contract with another tokenizer authority during materialization.
        selected.tokenizer_compatibility =
            TokenizerCompatibilityProof::prove([24; 32], [24; 32]).unwrap();
        let mut materialized = selected
            .preparation
            .materialize(InspectPreparation)
            .unwrap();
        assert_eq!(materialized.selected().requirements(), &requirements);
        assert_eq!(materialized.tokenizer_compatibility(), proof);
        assert_eq!(materialized.selected().placement().topology(), expected);
        assert_eq!(
            materialized.capture(),
            &crate::composite_execution::ExternalPredictionCaptureRequest::Gemma4SharedAttention {
                final_hidden_path: "model.language_model.layers.1.output".into(),
            },
        );
        let inspected = materialized.visit(TakeInspection);
        assert_eq!(inspected.model_type, "gemma4_assistant");
        assert_eq!(
            inspected.tokenizer_model_kind,
            crate::configuration::ModelKind::Gemma4
        );
    }
}

#[test]
fn execution_plan_rejects_every_missing_required_mechanism_before_source_preparation() {
    let root = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
    let (_target_root, target) = target_inspection(32);
    let retained = artifact(root.path());
    std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
    for missing in capabilities().mechanisms() {
        let placement = match missing {
            SpeculativeMechanism::CrossDeviceTransfer => DraftPlacementPlan::Device {
                device: DevicePlan::new("foreign", "device:1").unwrap(),
            },
            _ => DraftPlacementPlan::Device {
                device: DevicePlan::new("foreign", "device:0").unwrap(),
            },
        };
        let limited = SpeculativeMechanismCapabilities::new(
            capabilities()
                .mechanisms()
                .iter()
                .copied()
                .filter(|fact| fact != missing),
        );
        let result = prepare_execution_plan_assistant(
            &execution_plan(placement, 2),
            &target,
            retained.clone(),
            direct_lowering,
            &limited,
        );
        assert!(
            matches!(result, Err(AutomaticPlanningError::Invalid(ref message))
            if message.contains(&format!("{missing:?}"))),
            "missing {missing:?} must fail before source access"
        );
    }
    let result = prepare_execution_plan_assistant(
        &execution_plan(DraftPlacementPlan::Target, 2),
        &target,
        retained,
        direct_lowering,
        &capabilities(),
    );
    assert!(
        matches!(
            result,
            Err(AutomaticPlanningError::Backend {
                operation: "prepare_external_drafter_source",
                ..
            })
        ),
        "complete capability facts must reach the deliberately missing source"
    );
}

#[test]
fn execution_plan_validates_policy_lowering_compatibility_and_capacity_before_source() {
    let root = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
    let (_target_root, target) = target_inspection(32);
    let (_incompatible_root, incompatible) = target_inspection(64);
    let retained = artifact(root.path());
    std::fs::remove_file(root.path().join("model.safetensors")).unwrap();
    let invalid = execution_plan(DraftPlacementPlan::Target, 2).with_max_cached_shards(0);
    assert!(matches!(
        prepare_execution_plan_assistant(
            &invalid,
            &target,
            retained.clone(),
            |_, _| panic!("invalid policy reached lowering"),
            &capabilities()
        ),
        Err(AutomaticPlanningError::Invalid(_))
    ));
    assert!(matches!(
        prepare_execution_plan_assistant(
            &execution_plan(DraftPlacementPlan::Target, 2),
            &target,
            retained.clone(),
            |_, _| None,
            &capabilities()
        ),
        Err(AutomaticPlanningError::Backend {
            operation: "select_external_drafter",
            ..
        })
    ));
    assert!(matches!(prepare_execution_plan_assistant(
        &execution_plan(DraftPlacementPlan::Target, 2), &incompatible, retained.clone(), direct_lowering, &capabilities()
    ), Err(AutomaticPlanningError::Invalid(ref message)) if message.contains("incompatible")));
    assert!(matches!(prepare_execution_plan_assistant(
        &execution_plan(DraftPlacementPlan::Target, 4), &target, retained, direct_lowering, &capabilities()
    ), Err(AutomaticPlanningError::Invalid(ref message)) if message.contains("exceeds architecture capacity 3")));
}

#[test]
fn execution_plan_forwards_the_neutral_weight_transformation() {
    let root = safetensors_artifact(GEMMA_ASSISTANT, gemma_tensors());
    let (_target_root, target) = target_inspection(32);
    let transformations = std::cell::Cell::new(0);
    let plan = execution_plan(DraftPlacementPlan::Target, 2).with_weight_transformation(
        WeightTransformationPlan::Affine {
            bits: 4,
            group_size: 32,
        },
    );
    prepare_execution_plan_assistant(
        &plan,
        &target,
        artifact(root.path()),
        |descriptor, transforms| {
            if transforms
                && matches!(
                    descriptor.source(),
                    SourceTensorEncoding::Safetensors(StoredDtype::F32)
                )
                && matches!(descriptor.executable(), LinearFormat::Affine(_))
            {
                transformations.set(transformations.get() + 1);
                Some(WeightLoweringKind::Transform)
            } else {
                direct_lowering(descriptor, transforms)
            }
        },
        &capabilities(),
    )
    .unwrap();
    assert!(transformations.get() > 0);
}

#[test]
fn execution_plan_reader_bound_governs_the_exact_prepared_source() {
    use eredu_checkpoint::store::{ReadPolicy, TensorReadRequest, TensorSelection};

    let root = tempfile::tempdir().unwrap();
    std::fs::write(root.path().join("config.json"), GEMMA_ASSISTANT).unwrap();
    let tensors = gemma_tensors();
    let mut weight_map = BTreeMap::new();
    let mut keys = Vec::new();
    for shard in 0..3 {
        let name = format!("model-{shard}.safetensors");
        let entries = tensors.iter().skip(shard).step_by(3).collect::<Vec<_>>();
        keys.push(entries[0].0.clone());
        for (key, _, _) in &entries {
            weight_map.insert(key.clone(), name.clone());
        }
        let views = entries.iter().map(|(key, shape, bytes)| {
            (
                key.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        });
        serialize_to_file(views, None, &root.path().join(name)).unwrap();
    }
    std::fs::write(
        root.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
    )
    .unwrap();

    struct CheckReaderBound(Vec<String>);
    impl ExternalAssistantPreparationVisitor for CheckReaderBound {
        type Output<A: ExternalAssistantArchitecture> = ();
        type Error = Infallible;

        fn visit<A: ExternalAssistantArchitecture>(
            self,
            prepared: PreparedExternalAssistantSource<A>,
        ) -> Result<(), Infallible> {
            let (source, _, _, _, _, _) = prepared.into_parts();
            let acquire = |key: &String| {
                source.acquire_lease(TensorReadRequest {
                    key: key.clone(),
                    selection: TensorSelection::Full,
                    policy: ReadPolicy::RequireBounded,
                })
            };
            let first = acquire(&self.0[0]).unwrap();
            let second = acquire(&self.0[1]).unwrap();
            let before = source.source_diagnostics().unwrap();
            assert!(matches!(
                acquire(&self.0[2]),
                Err(StoreError::CapacityExhausted { maximum: 2, .. })
            ));
            drop(first);
            let third = acquire(&self.0[2]).unwrap();
            let diagnostics = source.source_diagnostics().unwrap();
            assert_eq!(diagnostics.currently_cached_shards, 2);
            assert_eq!(diagnostics.evictions, before.evictions + 1);
            drop((second, third));
            Ok(())
        }
    }

    let (_target_root, target) = target_inspection(32);
    let plan = execution_plan(DraftPlacementPlan::Target, 2).with_max_cached_shards(2);
    prepare_execution_plan_assistant(
        &plan,
        &target,
        artifact(root.path()),
        direct_lowering,
        &capabilities(),
    )
    .unwrap()
    .preparation
    .materialize(CheckReaderBound(keys))
    .unwrap();
}
