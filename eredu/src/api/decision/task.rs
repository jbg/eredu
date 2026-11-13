use std::path::Path;

use anyhow::{ensure, Context, Result};
use eredu_backend_mlx::native::ExecutionContext;
use eredu_checkpoint::recipe::RecipeDtype;
use eredu_core::{Media, ResidencyPlan};
use eredu_runtime::{execution_plan_layer_residency, LayerWeightResidency, ResidencyDiagnostics};

use super::*;
use crate::api::LocalDevice;

/// A prepared native decision operation with its decoded input.
pub struct PreparedDecisionTask {
    model: PreparedTextChoiceModel,
    request: DecisionRequest,
    max_cached_shards: usize,
}

impl PreparedDecisionTask {
    /// Discovers and prepares a native decision artifact. A checkpoint without a
    /// native decision declaration can be evaluated through text refinement.
    ///
    /// A split artifact uses its explicit media projector and local sidecars.
    pub fn discover(
        path: &Path,
        projector: Option<&Path>,
        assets: Option<&Path>,
        request: &DecisionRequest,
        max_cached_shards: usize,
    ) -> Result<Option<Self>> {
        request.validate()?;
        let audio = request
            .media
            .iter()
            .any(|media| matches!(media, Media::Audio(_)))
            || request.questions.iter().any(|q| q.spoken.is_some());
        let visual = request
            .media
            .iter()
            .any(|media| matches!(media, Media::Image(_) | Media::Video(_)));
        let dtype = RecipeDtype::F32;
        let model = if path.extension().is_some_and(|ext| ext == "gguf") {
            let assets = assets
                .or_else(|| path.parent())
                .context("decision sidecar directory is required")?;
            let Some(model) = PreparedTextChoiceModel::discover_gguf(
                path,
                projector,
                assets,
                dtype,
                max_cached_shards,
            )?
            else {
                return Ok(None);
            };
            ensure!(
                !(visual || audio) || projector.is_some(),
                "media requires a projector companion"
            );
            model
        } else {
            let root = if path.is_dir() {
                path
            } else {
                path.parent().context("model directory is required")?
            };
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join("config.json"))?)?;
            if !eredu_architectures::lfm3::decision::is_d1_config(&config) {
                return Ok(None);
            }
            ensure!(
                projector.is_none() && assets.is_none(),
                "native SafeTensors decisions use bundled companions"
            );
            ensure!(
                path.is_dir(),
                "native SafeTensors decisions require the checkpoint directory"
            );
            if audio {
                PreparedTextChoiceModel::open_with_omni(root, dtype, max_cached_shards)?
            } else if visual {
                PreparedTextChoiceModel::open_with_vision(root, dtype, max_cached_shards)?
            } else {
                PreparedTextChoiceModel::open(root, dtype, max_cached_shards)?
            }
        };
        ensure!(
            request.refinement_steps.is_none() && request.seed.is_none(),
            "native decisions do not use refinement_steps or seed"
        );
        Ok(Some(Self {
            model,
            request: request.clone(),
            max_cached_shards,
        }))
    }

    /// Loads the prepared graph and drains its controlled decision session.
    pub fn evaluate(
        self,
        device: LocalDevice,
        residency: &ResidencyPlan,
    ) -> Result<DecisionResponse> {
        let residency = execution_plan_layer_residency(
            residency,
            self.max_cached_shards,
            ResidencyDiagnostics::default(),
        )?;
        let residency: LayerWeightResidency = residency.into();
        let weights = ExecutionContext::cpu().map_err(eredu_core::BackendFailure::from_error)?;
        let execution = match device {
            LocalDevice::Cpu => ExecutionContext::cpu(),
            LocalDevice::Accelerator(index) => ExecutionContext::accelerator(index as usize),
        }
        .map_err(eredu_core::BackendFailure::from_error)?;
        let request = self.request;
        let questions: Vec<_> = request
            .questions
            .iter()
            .map(|q| q.question.clone())
            .collect();
        let state = request.state.as_deref();
        let audio = request
            .media
            .iter()
            .any(|media| matches!(media, Media::Audio(_)))
            || request.questions.iter().any(|q| q.spoken.is_some());
        let evaluation = if audio {
            #[cfg(all(feature = "audio", feature = "image"))]
            {
                let mut model = TextChoiceModel::load_with_omni(
                    self.model, residency, residency, residency, &weights, &execution,
                )?;
                let spoken = request
                    .questions
                    .iter()
                    .map(|q| q.spoken.as_ref())
                    .collect::<Vec<_>>();
                model.evaluate_with_spoken_questions(state, &request.media, &questions, &spoken)?
            }
            #[cfg(not(all(feature = "audio", feature = "image")))]
            anyhow::bail!("decision audio requires the audio and image features");
        } else if !request.media.is_empty() {
            #[cfg(feature = "image")]
            {
                let mut model = TextChoiceModel::load_with_vision(
                    self.model, residency, residency, &weights, &execution,
                )?;
                model.evaluate_with_visuals(state, &request.media, &questions)?
            }
            #[cfg(not(feature = "image"))]
            anyhow::bail!("visual decisions require the image feature");
        } else {
            let mut model = TextChoiceModel::load(self.model, residency, &weights, &execution)?;
            model.evaluate_decisions(state, &questions)?
        };
        Ok(DecisionResponse {
            answers: request
                .questions
                .into_iter()
                .zip(evaluation.answers)
                .map(|(q, answer)| DecisionResult {
                    id: q.id,
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
        })
    }
}

/// Evaluates a local artifact through its native decision or refinement capability.
pub fn evaluate_local_decisions(
    path: &Path,
    assets: Option<&Path>,
    request: &DecisionRequest,
    device: LocalDevice,
    plan: &eredu_core::ExecutionPlan,
) -> Result<DecisionResponse> {
    ensure!(
        plan.device() == &crate::api::local_device_plan(device)?,
        "decision execution plan and device must match"
    );
    let companions = plan.media_companions();
    if let Some(task) = PreparedDecisionTask::discover(
        path,
        companions.vision().or(companions.audio()),
        assets,
        request,
        plan.max_cached_shards(),
    )? {
        ensure!(
            companions
                .vision()
                .zip(companions.audio())
                .is_none_or(|(a, b)| a == b),
            "native decisions require one shared media projector"
        );
        ensure!(
            plan.topology().is_replicated(),
            "local decision evaluation requires a single rank"
        );
        ensure!(
            plan.weight_transformation()
                == eredu_core::WeightTransformationPlan::PreserveCheckpoint,
            "native decision evaluation requires checkpoint weight encodings"
        );
        return task.evaluate(device, plan.residency());
    }
    ensure!(
        assets.is_none(),
        "decision sidecars require a native decision artifact"
    );
    let factory = eredu_backend_mlx::MlxBackendFactory::default();
    let mut loaded = crate::api::LoadedModel::load_execution_plan(&factory, path, plan)?;
    super::evaluate_refinement(&mut loaded.model_mut(), request)
}
