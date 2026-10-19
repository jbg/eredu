use super::*;
use crate::backend::submission_recovery;
use eredu_architectures::{gemma4, MaterializedExternalAssistantVisitor};
use safemlx::{Device, DeviceType, SubmissionScope};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

fn drafter(
    stream: &Stream,
) -> (
    MlxDrafter,
    eredu_core::ModelRuntime<crate::composition::mlx::MlxBackend<'static>>,
) {
    use super::super::test_fixture::{assistant_artifact, ASSISTANT_CONFIG};

    let assistant = assistant_artifact();
    let source: serde_json::Value = serde_json::from_str(ASSISTANT_CONFIG).unwrap();
    let mut config = serde_json::json!({
        "model_type": "gemma4",
        "tie_word_embeddings": false,
        "text_config": source["text_config"],
    });
    config["text_config"]["num_hidden_layers"] = 2.into();
    config["text_config"]["num_kv_shared_layers"] = 1.into();
    config["text_config"]["layer_types"] = serde_json::json!(["full_attention", "full_attention"]);
    let config = serde_json::to_vec(&config).unwrap();
    let family = gemma4::FamilyConfig::from_hf_json(&config).unwrap();
    let schema = gemma4::safetensors_plan(&family).unwrap();
    assert!(schema.layout_groups.is_empty());
    let target = tempfile::tempdir().unwrap();
    std::fs::write(target.path().join("config.json"), config).unwrap();
    let tensors = schema
        .common_tensors
        .iter()
        .map(|tensor| {
            (
                tensor.key.clone(),
                tensor.shape.clone(),
                vec![0; tensor.shape.iter().product::<usize>() * 4],
            )
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(key, shape, bytes)| {
            (
                key.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &target.path().join("model.safetensors")).unwrap();
    let inspection = eredu_architectures::configuration::inspect_artifact(target.path()).unwrap();
    let plan = eredu_core::ExecutionPlan::fully_resident(
        eredu_core::DevicePlan::new("mlx", "cpu:0").unwrap(),
    )
    .with_drafting(eredu_core::DraftingPlan::External {
        residency: eredu_core::ResidencyPlan::FullyResident,
        model: assistant.path().display().to_string(),
        placement: eredu_core::DraftPlacementPlan::Target,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let factory = super::super::super::automatic::MlxBackendFactory::default();
    let target_selection =
        eredu_core::select_execution_plan_target(&factory, &plan, inspection.clone()).unwrap();
    let prepared = eredu_architectures::prepare_execution_plan_assistant(
        &plan,
        &inspection,
        target_selection.selected_preparation().neutral(),
        eredu_core::ExternalDraftArtifact {
            preparation: eredu_architectures::prepare_external_assistant(assistant.path()).unwrap(),
            tokenizer_compatibility: TokenizerCompatibilityProof::prove([9; 32], [9; 32]).unwrap(),
        },
        |descriptor, transforms| {
            (!transforms && super::super::super::replicated_text::supports_direct(descriptor))
                .then_some(eredu_runtime::WeightLoweringKind::Direct)
        },
        &speculative_mechanism_capabilities(),
    )
    .unwrap();
    let runtime = eredu_core::realize_execution_plan_target(&factory, &plan, target_selection)
        .unwrap()
        .into_runtime()
        .unwrap();
    let drafter = MlxDrafter::materialize(
        eredu_architectures::PreparedExternalDraft::Assistant(prepared.preparation),
        runtime.session(),
        stream,
        stream,
    )
    .unwrap();
    (drafter, runtime)
}

struct CountVisit<'a>(&'a Cell<usize>);

impl MaterializedExternalAssistantVisitor<MlxAssistantPreparationVisitor> for CountVisit<'_> {
    type Output = Result<(), Error>;

    fn visit<A: eredu_architectures::ExternalAssistantArchitecture>(
        self,
        _: &mut MlxExternalAssistant<A>,
    ) -> Self::Output {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
}

#[test]
fn terminal_retained_drafter_owner_rejects_mutation_until_exclusive_without_poisoning() {
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    // Keep the bound target alive; dropping it would inject unrelated target
    // retirement into the lock-contention boundary being measured here.
    let (mut drafter, mut target) = drafter(&stream);
    let mut scope = SubmissionScope::begin().unwrap();
    let value = Array::from_slice(&[2.0_f32], &[1]).square(&stream).unwrap();
    eval([&value]).unwrap();
    scope.seal();
    assert!(scope.status().has_work());
    let deadline = Instant::now() + Duration::from_secs(10);
    while !scope.progress().is_settled() {
        assert!(Instant::now() < deadline, "native scope did not settle");
        std::thread::yield_now();
    }
    assert!(!scope.status().failed());
    let recovery = Recovery::with_probe(
        DrafterRetention {
            _payload: Rc::new(RefCell::new(Some(Rc::clone(&drafter.payload)))),
            poisoned: Rc::clone(&drafter.poisoned),
        },
        scope,
    );
    let (ready_tx, ready_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let holder = std::thread::spawn(move || loop {
        if safemlx::try_with_submission_retirement(|| {
            ready_tx.send(()).unwrap();
            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        })
        .is_some()
        {
            break;
        }
        std::thread::yield_now();
    });
    ready_rx.recv_timeout(Duration::from_secs(10)).unwrap();
    drop(recovery);
    assert_eq!(Rc::strong_count(&drafter.payload), 2);
    assert!(!drafter.poisoned.get());
    let visits = Cell::new(0);
    let start = Instant::now();
    let preparation_calls = Cell::new(0);
    let error = match drafter.prepare_operation(|local| {
        preparation_calls.set(preparation_calls.get() + 1);
        assert!(local.is_err(), "failed acquisition still enters readiness");
        local
    }) {
        Ok(_) => panic!("retained drafter must reject acquisition"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("still retained by prior work"));
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(visits.get(), 0);
    assert_eq!(preparation_calls.get(), 1);
    assert!(!drafter.poisoned.get());
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    submission_recovery::reap();
    crate::backend::submission_recovery::wait_for_retirement(|| {
        Rc::strong_count(&drafter.payload) == 1
    });
    assert_eq!(Rc::strong_count(&drafter.payload), 1);
    let error = match drafter.prepare_operation(|local| {
        local?;
        Err(Error::Speculative(
            "peer rejected mutation ownership".into(),
        ))
    }) {
        Ok(_) => panic!("peer rejection must reject acquisition"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("peer rejected mutation ownership"));
    assert_eq!(visits.get(), 0);
    assert!(!drafter.poisoned.get());
    // A real local assistant acquisition failure occurs inside the target's
    // unused preparation scope. Neither executable may be visited or poisoned.
    let retained = Rc::clone(&drafter.payload);
    let failed = target
        .session_mut()
        .with_prepared_model_operation(
            |local| {
                local?;
                drafter.prepare_operation(|local| {
                    assert!(local.is_err());
                    local
                })
            },
            |_, draft| draft.visit(CountVisit(&visits)),
        )
        .unwrap_err();
    assert!(failed.to_string().contains("still retained by prior work"));
    assert_eq!(visits.get(), 0);
    target.session().ensure_no_submission_in_flight().unwrap();
    assert!(!drafter.poisoned.get());
    drop(retained);
    target
        .session_mut()
        .with_prepared_model_operation(
            |local| {
                local?;
                drafter.prepare_operation(|local| local)
            },
            |_, draft| draft.visit(CountVisit(&visits)),
        )
        .unwrap();
    assert_eq!(visits.get(), 1);
    assert!(!drafter.poisoned.get());
}
