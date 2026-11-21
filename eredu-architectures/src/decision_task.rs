//! Prepared decision execution and host-facing advancement.

use std::path::Path;

use eredu_checkpoint::{prepared_cache::PreparedCheckpointCachePolicy, recipe::RecipeDtype};
use eredu_core::{
    decision::{DecisionBinding, DecisionBindingFactory, DecisionEvaluation, DecisionQuestion},
    Audio, BackendFailure, DevicePlan, ExecutionPlanBackendFactory, Media, MediaCompanionSources,
};
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, Tensor, TensorParallelGroupedNeuralBackend,
};
use eredu_runtime::{
    decision_binding::PreparedDecisionProgram, ComponentRankMechanisms, DecoderPipelineState,
    LayerWeightResidency, LocalComponentMechanisms, RuntimeStateComponents,
};

use crate::lfm3::decision::{self, execution::TextChoiceModel, PreparedTextChoiceModel};

/// Binds a retained decision artifact using a configured backend factory.
pub trait DecisionBackendFactory:
    ExecutionPlanBackendFactory<crate::processor_plan::ArtifactArchitecturePlan>
    + DecisionBindingFactory
{
    /// Realizes the retained artifact on the plan's selected device.
    fn bind_decision(
        &self,
        model: PreparedDecisionModel,
        device: &DevicePlan,
        residency: LayerWeightResidency,
        synchronize_load: bool,
    ) -> Result<Box<dyn BoundDecisionModel>, DecisionTaskError>;
}

impl<D, F> DecisionBackendFactory for F
where
    D: DecisionBinding,
    F: ExecutionPlanBackendFactory<crate::processor_plan::ArtifactArchitecturePlan>
        + DecisionBindingFactory<Binding = D>,
    <D::Backend as eredu_core::backend::LocalComponentStorage>::CacheStorage:
        eredu_runtime::state::HybridCacheStorage,
    PreparedDecisionModel: PreparedDecisionProgram<
        D::Backend,
        eredu_runtime::decision_binding::DecisionState<D::Backend>,
        D,
        D::Parallel,
        D::InputLowering,
        Model = Box<dyn BoundDecisionModel>,
        Error = DecisionTaskError,
    >,
{
    fn bind_decision(
        &self,
        model: PreparedDecisionModel,
        device: &DevicePlan,
        residency: LayerWeightResidency,
        synchronize_load: bool,
    ) -> Result<Box<dyn BoundDecisionModel>, DecisionTaskError> {
        eredu_runtime::decision_binding::bind_decision_model(
            self,
            model,
            device,
            residency,
            synchronize_load,
        )
    }
}

/// Media requirements for selecting a finite decision task.
#[derive(Clone, Copy, Debug, Default)]
pub struct DecisionMedia {
    /// Image or video evidence.
    pub visual: bool,
    /// Audio evidence or spoken questions.
    pub audio: bool,
}

/// Retained model-family plan for a native finite decision.
pub struct PreparedDecisionModel {
    model: PreparedTextChoiceModel,
    media: DecisionMedia,
}

/// Failure to prepare, bind, or advance a native decision.
#[derive(Debug, thiserror::Error)]
pub enum DecisionTaskError {
    /// The selected adapter lacks a required decision mechanism.
    #[error("decision mechanism unavailable: {0}")]
    Unsupported(&'static str),
    /// The artifact or supplied input cannot be used for this decision.
    #[error("invalid decision input: {0}")]
    Input(&'static str),
    /// Architecture-owned preparation or readout failed.
    #[error("decision preparation or readout failed: {0}")]
    Preparation(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The selected backend failed to execute the decision.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// The request was cancelled before publication.
    #[error("decision evaluation was cancelled")]
    Cancelled,
}

impl From<decision::execution::DecisionExecutionError> for DecisionTaskError {
    fn from(error: decision::execution::DecisionExecutionError) -> Self {
        use decision::execution::DecisionExecutionError as Error;
        match error {
            Error::Backend(error) => Self::Backend(error),
            Error::Cancelled => Self::Cancelled,
            other => Self::Preparation(Box::new(other)),
        }
    }
}

/// A decoded decision invocation in caller order.
pub struct DecisionInput<'a> {
    /// Shared textual evidence.
    pub state: Option<&'a str>,
    /// Shared media evidence.
    pub media: &'a [Media],
    /// Questions and their readout settings.
    pub questions: &'a [DecisionQuestion],
    /// One optional spoken clip per question, or an empty slice for written input.
    pub spoken: &'a [Option<&'a Audio>],
}

/// A native decision model with host-only request and result types.
pub trait BoundDecisionModel {
    /// Prepares and starts a cancellable finite invocation.
    fn start(
        &mut self,
        input: DecisionInput<'_>,
    ) -> Result<Box<dyn DecisionProgress + '_>, DecisionTaskError>;
}

/// Host operations on a controlled decision invocation.
pub trait DecisionProgress {
    /// Advances one bounded step, publishing only the complete decision evaluation.
    fn advance(&mut self) -> Result<Option<DecisionEvaluation>, DecisionTaskError>;
    /// Requests cancellation.
    fn cancel(&mut self);
    /// Whether the invocation is terminal.
    fn is_finished(&self) -> bool;
}

impl PreparedDecisionModel {
    /// Discovers native decision support and admits the exact companions for the request.
    pub fn discover(
        path: &Path,
        companions: &MediaCompanionSources,
        assets: Option<&Path>,
        media: DecisionMedia,
        max_cached_shards: usize,
        prepared_cache: Option<&PreparedCheckpointCachePolicy>,
    ) -> Result<Option<Self>, DecisionTaskError> {
        let prepare = |error| DecisionTaskError::Preparation(Box::new(error));
        let dtype = RecipeDtype::F32;
        let projector = companions.vision().or(companions.audio());
        let model = if path.extension().is_some_and(|ext| ext == "gguf") {
            let assets = assets
                .or_else(|| path.parent())
                .ok_or(DecisionTaskError::Input(
                    "decision sidecar directory is required",
                ))?;
            let Some(model) = PreparedTextChoiceModel::discover_gguf(
                path,
                projector,
                assets,
                dtype,
                max_cached_shards,
                prepared_cache,
            )
            .map_err(prepare)?
            else {
                return Ok(None);
            };
            if companions
                .vision()
                .zip(companions.audio())
                .is_some_and(|(vision, audio)| vision != audio)
            {
                return Err(DecisionTaskError::Input(
                    "native decisions require one shared media projector",
                ));
            }
            if (media.visual || media.audio) && projector.is_none() {
                return Err(DecisionTaskError::Input(
                    "media requires a projector companion",
                ));
            }
            model
        } else {
            let root = if path.is_dir() {
                Some(path)
            } else {
                path.parent()
            }
            .ok_or(DecisionTaskError::Input("model directory is required"))?;
            let config = decision::read(root, "config.json").map_err(prepare)?;
            let config = serde_json::from_slice(&config)
                .map_err(decision::DecisionError::from)
                .map_err(prepare)?;
            if !decision::is_d1_config(&config) {
                return Ok(None);
            }
            if projector.is_some() || assets.is_some() || !path.is_dir() {
                return Err(DecisionTaskError::Input("native SafeTensors decisions require a checkpoint directory with bundled companions"));
            }
            if media.audio {
                PreparedTextChoiceModel::open_with_omni(root, dtype, max_cached_shards)
            } else if media.visual {
                PreparedTextChoiceModel::open_with_vision(root, dtype, max_cached_shards)
            } else {
                PreparedTextChoiceModel::open(root, dtype, max_cached_shards)
            }
            .map_err(prepare)?
        };
        Ok(Some(Self { model, media }))
    }

    /// Binds the admitted task using ordinary component mechanisms.
    pub fn bind<B, S, M, C, I>(
        self,
        residency: LayerWeightResidency,
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
        input_lowering: I,
    ) -> Result<Box<dyn BoundDecisionModel>, DecisionTaskError>
    where
        B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
        B::Tensor: 'static,
        <B::Tensor as Tensor>::Context: Clone + 'static,
        S: DecoderPipelineState<B> + 'static,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        M: LocalComponentMechanisms<B, S> + ComponentRankMechanisms<B, S, C> + 'static,
        C: 'static,
        I: DecisionInputLowering<B::Tensor> + 'static,
    {
        let model: TextChoiceModel<B, S, M, C, I> = if self.media.audio {
            #[cfg(all(feature = "image", feature = "audio"))]
            {
                TextChoiceModel::load_with_omni(
                    self.model,
                    residency,
                    residency,
                    residency,
                    mechanisms,
                    context,
                    input_lowering,
                )?
            }
            #[cfg(not(all(feature = "image", feature = "audio")))]
            return Err(DecisionTaskError::Unsupported("audio decision processing"));
        } else if self.media.visual {
            #[cfg(feature = "image")]
            {
                TextChoiceModel::load_with_vision(
                    self.model,
                    residency,
                    residency,
                    mechanisms,
                    context,
                    input_lowering,
                )?
            }
            #[cfg(not(feature = "image"))]
            return Err(DecisionTaskError::Unsupported("visual decision processing"));
        } else {
            TextChoiceModel::load(self.model, residency, mechanisms, context, input_lowering)?
        };
        Ok(Box::new(model))
    }
}

impl<B, S, M, C, I> eredu_runtime::decision_binding::PreparedDecisionProgram<B, S, M, C, I>
    for PreparedDecisionModel
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    <B::Tensor as Tensor>::Context: Clone + 'static,
    S: DecoderPipelineState<B> + 'static,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    M: LocalComponentMechanisms<B, S> + ComponentRankMechanisms<B, S, C> + 'static,
    C: 'static,
    I: DecisionInputLowering<B::Tensor> + 'static,
{
    type Model = Box<dyn BoundDecisionModel>;
    type Error = DecisionTaskError;

    fn bind(
        self,
        residency: LayerWeightResidency,
        mechanisms: &M,
        context: &<B::Tensor as Tensor>::Context,
        input_lowering: I,
    ) -> Result<Self::Model, Self::Error> {
        PreparedDecisionModel::bind::<B, S, M, C, I>(
            self,
            residency,
            mechanisms,
            context,
            input_lowering,
        )
    }
}

/// Input lowering required by the configured decision media features.
#[cfg(feature = "image")]
pub use eredu_core::input::MediaInputLowering as DecisionInputLowering;

/// Text-only decisions require no native media ingress mechanisms.
#[cfg(not(feature = "image"))]
pub trait DecisionInputLowering<T> {}

#[cfg(not(feature = "image"))]
impl<T> DecisionInputLowering<T> for () {}

impl<B, S, M, C, I> BoundDecisionModel for TextChoiceModel<B, S, M, C, I>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    <B::Tensor as Tensor>::Context: Clone,
    S: DecoderPipelineState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    M: LocalComponentMechanisms<B, S> + ComponentRankMechanisms<B, S, C>,
    I: DecisionInputLowering<B::Tensor>,
{
    fn start(
        &mut self,
        input: DecisionInput<'_>,
    ) -> Result<Box<dyn DecisionProgress + '_>, DecisionTaskError> {
        if !input.spoken.is_empty() && input.spoken.len() != input.questions.len() {
            return Err(DecisionTaskError::Input(
                "spoken instructions must align with questions",
            ));
        }
        let has_audio = input.spoken.iter().any(Option::is_some)
            || input
                .media
                .iter()
                .any(|item| matches!(item, Media::Audio(_)));
        let session = if has_audio {
            #[cfg(all(feature = "image", feature = "audio"))]
            {
                if input.spoken.is_empty() {
                    self.start_with_media(input.state, input.media, input.questions)?
                } else {
                    self.start_with_spoken_questions(
                        input.state,
                        input.media,
                        input.questions,
                        input.spoken,
                    )?
                }
            }
            #[cfg(not(all(feature = "image", feature = "audio")))]
            return Err(DecisionTaskError::Unsupported("audio decision processing"));
        } else if !input.media.is_empty() {
            #[cfg(feature = "image")]
            {
                self.start_with_visuals(input.state, input.media, input.questions)?
            }
            #[cfg(not(feature = "image"))]
            return Err(DecisionTaskError::Unsupported("visual decision processing"));
        } else {
            self.start_decisions(input.state, input.questions)?
        };
        Ok(Box::new(session))
    }
}

impl<B, S, M, C, I> DecisionProgress for decision::execution::DecisionSession<'_, B, S, M, C, I>
where
    B: TensorParallelGroupedNeuralBackend + DistributedNeuralBackend + 'static,
    B::Tensor: 'static,
    <B::Tensor as Tensor>::Context: Clone,
    S: DecoderPipelineState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    M: LocalComponentMechanisms<B, S> + ComponentRankMechanisms<B, S, C>,
{
    fn advance(&mut self) -> Result<Option<DecisionEvaluation>, DecisionTaskError> {
        Ok(self.advance()?)
    }

    fn cancel(&mut self) {
        self.cancel();
    }

    fn is_finished(&self) -> bool {
        self.is_finished()
    }
}
