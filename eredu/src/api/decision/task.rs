use std::path::Path;

use anyhow::{ensure, Result};
use eredu_architectures::{
    decision_task::{DecisionInput, DecisionMedia, PreparedDecisionModel},
    processor_plan::ArtifactArchitecturePlan,
    ExternalDraftPreparation,
};
use eredu_core::{
    ExecutionPlan, Media, ModelConfigurationResolver, ModelLoadingBackend, TextGenerationBackend,
    WeightTransformationPlan,
};
use eredu_runtime::{execution_plan_layer_residency, ResidencyDiagnostics};

use super::*;

/// Evaluates a request using the artifact's declared decision or refinement capability.
/// The optional callback runs after binding and synchronization.
pub fn evaluate_decisions<F>(
    factory: &F,
    path: &Path,
    assets: Option<&Path>,
    request: &DecisionRequest,
    plan: &ExecutionPlan,
    on_loaded: Option<&mut dyn FnMut()>,
) -> Result<DecisionResponse>
where
    F: DecisionBackendFactory<DrafterPreparation = ExternalDraftPreparation>,
    F::Backend: TextGenerationBackend + ModelLoadingBackend,
    <F::Backend as ModelLoadingBackend>::ConfigurationResolver:
        ModelConfigurationResolver<ArtifactPlan = ArtifactArchitecturePlan>,
{
    request.validate()?;
    let media = DecisionMedia {
        audio: request
            .media
            .iter()
            .any(|item| matches!(item, Media::Audio(_)))
            || request.questions.iter().any(|item| item.spoken.is_some()),
        visual: request
            .media
            .iter()
            .any(|item| matches!(item, Media::Image(_) | Media::Video(_))),
    };
    if let Some(prepared) = PreparedDecisionModel::discover(
        path,
        plan.media_companions(),
        assets,
        media,
        plan.max_cached_shards(),
        plan.prepared_checkpoint_cache(),
    )? {
        ensure!(
            plan.topology().is_replicated(),
            "local decision evaluation requires a single rank"
        );
        ensure!(
            plan.weight_transformation() == WeightTransformationPlan::PreserveCheckpoint,
            "native decision evaluation requires checkpoint weight encodings"
        );
        ensure!(
            request.refinement_steps.is_none() && request.seed.is_none(),
            "native decisions do not use refinement_steps or seed"
        );
        let residency = execution_plan_layer_residency(
            plan.residency(),
            plan.max_cached_shards(),
            ResidencyDiagnostics::default(),
        )?;
        let mut model = factory.bind_decision(
            prepared,
            plan.device(),
            residency.into(),
            on_loaded.is_some(),
        )?;
        if let Some(callback) = on_loaded {
            callback();
        }
        let questions = request
            .questions
            .iter()
            .map(|item| item.question.clone())
            .collect::<Vec<_>>();
        let spoken = request
            .questions
            .iter()
            .map(|item| item.spoken.as_ref())
            .collect::<Vec<_>>();
        let mut progress = model.start(DecisionInput {
            state: request.state.as_deref(),
            media: &request.media,
            questions: &questions,
            spoken: &spoken,
        })?;
        let evaluation = loop {
            if let Some(result) = progress.advance()? {
                break result;
            }
            ensure!(
                !progress.is_finished(),
                "decision finished without publishing a result"
            );
        };
        return Ok(DecisionResponse {
            answers: request
                .questions
                .iter()
                .zip(evaluation.answers)
                .map(|(item, answer)| DecisionResult {
                    id: item.id.clone(),
                    answer,
                    label_mass: None,
                })
                .collect(),
            input_tokens: evaluation.input_tokens,
            generated_tokens: evaluation.generated_tokens,
            calibration_sha256: Some(
                evaluation
                    .calibration_sha256
                    .iter()
                    .map(|byte| format!("{byte:02x}"))
                    .collect(),
            ),
        });
    }
    ensure!(
        assets.is_none(),
        "decision sidecars require a native decision artifact"
    );
    let mut loaded = crate::api::LoadedModel::load_execution_plan(factory, path, plan)?;
    if let Some(callback) = on_loaded {
        loaded.model_mut().synchronize()?;
        callback();
    }
    evaluate_refinement(loaded.model_mut(), request)
}
