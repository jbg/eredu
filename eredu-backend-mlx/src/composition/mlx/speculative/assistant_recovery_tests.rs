use super::*;
use crate::backend::submission_recovery;
use eredu_architectures::{gemma4, MaterializedExternalAssistantVisitor};
use safemlx::{Device, DeviceType, SubmissionScope};
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use std::{
    sync::mpsc,
    time::{Duration, Instant},
};

fn drafter(stream: &Stream) -> MlxDrafter {
    use super::super::external_materialization_tests::{assistant_artifact, ASSISTANT_CONFIG};

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
        model: assistant.path().display().to_string(),
        placement: eredu_core::DraftPlacementPlan::Target,
        max_draft_tokens: 2,
        lookahead: false,
        adaptive_lookahead: false,
    });
    let prepared = eredu_architectures::prepare_execution_plan_assistant(
        &plan,
        &inspection,
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
    MlxDrafter::materialize(prepared.preparation, stream, stream).unwrap()
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
    let mut drafter = drafter(&stream);
    let mut scope = SubmissionScope::begin().unwrap();
    let value = Array::from_slice(&[2.0_f32], &[1]).square(&stream).unwrap();
    eval([&value]).unwrap();
    scope.seal();
    assert!(scope.status().has_work());
    assert!(scope.status().is_settled());
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
    let error = drafter.visit(CountVisit(&visits)).unwrap_err();
    assert!(error.to_string().contains("still retained by prior work"));
    assert!(start.elapsed() < Duration::from_secs(1));
    assert_eq!(visits.get(), 0);
    assert!(!drafter.poisoned.get());
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    submission_recovery::reap();
    assert_eq!(Rc::strong_count(&drafter.payload), 1);
    drafter.visit(CountVisit(&visits)).unwrap();
    assert_eq!(visits.get(), 1);
    assert!(!drafter.poisoned.get());
}
