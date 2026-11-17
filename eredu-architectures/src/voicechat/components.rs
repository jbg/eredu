//! Retained construction of the components used by the coupled executor.
//!
//! This driver owns family composition. Native callers supply ordinary neural,
//! parameter and cache mechanisms; they do not resolve artifacts or interpret
//! checkpoint layouts. Session admission and publication remain caller-owned.

use super::rnnt::RnntExecution;
use eredu_codec::{
    ear_decoder_layered::DecoderExecution, ear_encoder_layered::EncoderExecution,
    prvq_layered::QuantizerExecution,
};
use eredu_core::BackendFailure;
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, NeuralBackend, Tensor,
    TensorElementType,
};
use eredu_runtime::{
    ComponentPreparation, DenseSamplingBackend, LayerRuntimeState, LayeredInvocation,
    LayeredTensorCohort, LocalComponentMechanisms, ParameterUnitMechanisms,
    RealtimeGenerationBranch, RealtimeJointStateParts, RuntimeStateComponents, SubmissionBackend,
};

use crate::nemotron_h;

use super::{
    agent_sampling::{AgentSampler, AgentSamplingPolicy},
    artifact::RetainedVoiceChatArtifacts,
    assembly::LearnedSpeechComponents,
    backbone::{BackboneConstructionError, PreparedBackboneParameters},
    checkpoint::Component,
    component_residency::{
        VoiceChatResidencyError, VoiceChatResidencyPolicy, VoiceChatResidencyReport,
    },
    execution::{Components, ExecutionOutput, Invocation},
    frontend::FrontendOptions,
    function_head_layered::FunctionHeadExecution,
    host::{HostPreparationError, PreparedHostConditioning},
    input::{InputError, InputState},
    perception_layered::PerceptionExecution,
    perception_rnnt::PerceptionRnntLimits,
    preparation::{GenerationPreparationError, PreparedGeneration},
    provider_construction::{
        PreparedSpeechProviders, ProviderConstructionError, SpeechConditioning, SpeechDecoder,
        SpeechEncoder, SpeechFunctionHead, SpeechMog, SpeechPerception, SpeechQuantizer,
        SpeechRnnt, SpeechRvq, SpeechTransformer,
    },
    schedule::{ScheduleLimits, ScheduleState},
    silence::{PreparedSilenceConditioning, SilenceConditioningError},
    state::{NeuralStateGeometry, NeuralStateGeometryError},
    tts_conditioning_layered::ConditioningExecution,
    tts_mog::TtsError,
    tts_mog_layered::MogExecution,
    tts_prompt::PreparedTtsPrompt,
    tts_rvq_layered::RvqExecution,
    tts_transformer_layered::{CausalTransformerExecution, TransformerExecution},
};

/// Retained-source construction failure; no partially loaded aggregate is returned.
#[derive(Debug, thiserror::Error)]
pub enum VoiceChatConstructionError {
    /// Exact host coefficients cannot prepare the tool-conditioning carrier.
    #[error(transparent)]
    Silence(#[from] SilenceConditioningError),
    /// Retained configurations cannot describe the coupled neural state.
    #[error(transparent)]
    StateGeometry(#[from] NeuralStateGeometryError),
    /// Native submission or completion failed, preserving the original source.
    #[error("VoiceChat component materialization failed: {0}")]
    Backend(#[from] eredu_core::BackendFailure),
    /// The selected artifact task did not retain a required companion component.
    #[error("coupled VoiceChat construction requires retained component {0:?}")]
    MissingComponent(Component),
    /// Complete parameter allowances fail cold admission.
    #[error(transparent)]
    Residency(#[from] VoiceChatResidencyError),
    /// Retained host controls or tokenizer cannot be reconstructed.
    #[error(transparent)]
    Host(#[from] HostPreparationError),
    /// The exact preprocessing buffers or host queue policy cannot be used.
    #[error(transparent)]
    Input(#[from] InputError),
    /// Ordinary Nemotron construction or binding failed.
    #[error(transparent)]
    Backbone(#[from] BackboneConstructionError),
    /// Speech component construction or binding failed.
    #[error(transparent)]
    Speech(#[from] ProviderConstructionError),
}

/// Cold authority for complete coupled component construction from retained stores.
///
/// Every required role is checked before tokenizer reconstruction or native work.
/// This type borrows the admitted stores, preserving their guards and provenance.
pub struct PreparedVoiceChatConstruction<'a> {
    artifacts: &'a RetainedVoiceChatArtifacts,
    backbone: PreparedBackboneParameters,
    speech: PreparedSpeechProviders,
    host: PreparedHostConditioning,
    silence: PreparedSilenceConditioning,
    neural_state_geometry: NeuralStateGeometry,
}

impl<'a> PreparedVoiceChatConstruction<'a> {
    /// Selects load-time matrix recipes without changing neural state geometry.
    pub fn with_quantization(
        mut self,
        policy: super::quantization::SpeechQuantizationPolicy,
    ) -> Result<Self, VoiceChatConstructionError> {
        if let Some(quantization) = policy.backbone {
            self.backbone = self
                .backbone
                .with_quantization(quantization)
                .map_err(BackboneConstructionError::from)?;
        }
        self.speech = self
            .speech
            .with_quantization(policy, self.artifacts.plan().parameters())?;
        Ok(self)
    }

    /// Admits the coupled component set without reopening any artifact.
    pub fn from_retained(
        artifacts: &'a RetainedVoiceChatArtifacts,
    ) -> Result<Self, VoiceChatConstructionError> {
        for component in [
            Component::Perception,
            Component::Backbone,
            Component::FunctionHead,
            Component::Rnnt,
            Component::Tts,
            Component::CodecEncoder,
            Component::CodecDecoder,
            Component::CodecQuantizer,
            Component::Controls,
        ] {
            if artifacts.component_source(component).is_none() {
                return Err(VoiceChatConstructionError::MissingComponent(component));
            }
        }
        // Host coefficients must exist before native construction is possible.
        if artifacts.preprocessing().is_none() {
            return Err(
                InputError::from(super::frontend::FrontendError::MissingPreprocessing).into(),
            );
        }
        let backbone = PreparedBackboneParameters::from_retained(artifacts)
            .map_err(BackboneConstructionError::from)?;
        let speech = PreparedSpeechProviders::from_retained(artifacts)?;
        let neural_state_geometry =
            NeuralStateGeometry::new(backbone.declaration(), speech.transformer_declaration())?;
        Ok(Self {
            artifacts,
            backbone,
            speech,
            host: PreparedHostConditioning::from_retained(artifacts)?,
            silence: PreparedSilenceConditioning::from_retained(artifacts)?,
            neural_state_geometry,
        })
    }

    /// Selects architecture-owned precision before native construction and pool admission.
    pub fn with_speech_precision(
        mut self,
        precision: super::precision::SpeechPrecision,
    ) -> Result<Self, VoiceChatConstructionError> {
        self.speech = self
            .speech
            .with_precision(precision, self.artifacts.plan().parameters())?;
        Ok(self)
    }

    /// Exact independently selected speech parameter requirements.
    pub fn speech_providers(&self) -> &PreparedSpeechProviders {
        &self.speech
    }

    /// Exact backbone recipes, including embedding and agent readout.
    pub fn backbone_parameters(&self) -> &PreparedBackboneParameters {
        &self.backbone
    }

    /// Exact reconstructed tokenizer, controls and character mapping for cold host policy.
    pub fn host_conditioning(&self) -> &PreparedHostConditioning {
        &self.host
    }

    /// Cold neural geometry consumed by cache admission and native construction.
    pub fn neural_state_geometry(&self) -> &NeuralStateGeometry {
        &self.neural_state_geometry
    }

    /// Derives shared generation policy from the exact retained controls and config.
    /// This remains cold and precedes materialization or native cache allocation.
    pub fn prepare_generation(
        &self,
        limits: ScheduleLimits,
        sampling: AgentSamplingPolicy,
    ) -> Result<PreparedGeneration, GenerationPreparationError> {
        PreparedGeneration::new(
            self.artifacts.plan().config(),
            &self.host.tokens,
            limits,
            sampling,
        )
    }

    /// Binds speech and backbone through their shared parameter mechanisms.
    ///
    /// Returns empty input carry separately so the caller can put it under the
    /// same transaction as its neural caches. No prompt is executed, no cache is
    /// allocated, and no scheduler is created. The selected backbone and every
    /// independent speech pool pass aggregate budget admission before any binding.
    /// The enclosing native owner protects submission lifetime on every path.
    pub fn materialize<B, S, H, C>(
        self,
        options: FrontendOptions,
        policy: &VoiceChatResidencyPolicy,
        mechanisms: &H,
    ) -> Result<
        (BoundVoiceChatComponents<B, S, H, C>, InputState<B::Tensor>),
        VoiceChatConstructionError,
    >
    where
        B: GroupedNeuralBackend
            + DistributedNeuralBackend
            + SubmissionBackend<Executor = <<B as NeuralBackend>::Tensor as Tensor>::Context>
            + 'static,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        H: ParameterUnitMechanisms<B>
            + LocalComponentMechanisms<B, S>
            + ComponentPreparation<Context = <B::Tensor as Tensor>::Context>,
        C: LayeredTensorCohort<B>,
    {
        mechanisms.prepare_components(move |mechanisms, context| {
            let residency_report = policy.validate(&self.backbone, &self.speech)?;
            let input = InputState::from_retained(self.artifacts, options)?;
            let (speech, _) =
                self.speech
                    .materialize::<B, H, C>(policy.speech(), mechanisms, context)?;
            let backbone = mechanisms.bind_local(|| {
                self.backbone
                    .prepare_local::<B, S>(policy.backbone(), context)
                    .map_err(eredu_core::BackendFailure::from_error)
            })?;
            Ok((
                VoiceChatComponents {
                    backbone,
                    state: std::marker::PhantomData,
                    speech,
                    host: self.host,
                    silence: self.silence,
                    residency_report,
                },
                input,
            ))
        })
    }
}

/// Unpublished local VoiceChat generation candidate.
pub type LocalInvocationBranch<T, N, R, C> = RealtimeGenerationBranch<
    super::state::StateBranch<T, N>,
    ScheduleState,
    AgentSamplingPolicy,
    AgentSampler,
    R,
    C,
>;

/// Selected provider types assembled through a native binding visitor.
pub type BoundVoiceChatComponents<B, S, H, C> = VoiceChatComponents<
    B,
    S,
    <H as LocalComponentMechanisms<B, S>>::Local<nemotron_h::LayeredModel<B>>,
    SpeechPerception<B, H, C>,
    SpeechTransformer<B, H, C>,
    SpeechMog<B, H, C>,
    SpeechRvq<B, H, C>,
    SpeechConditioning<B, H, C>,
    SpeechDecoder<B, H, C>,
    SpeechQuantizer<B, H, C>,
    SpeechEncoder<B, H, C>,
    SpeechFunctionHead<B, H, C>,
    SpeechRnnt<B, H, C>,
>;

/// Bound components for the existing coupled executor, with exact retained host assets.
///
/// This aggregate owns learned weights. The input carry, neural transaction,
/// RNG, submission completion and shared session scheduler have separate owners.
pub struct VoiceChatComponents<B, S, R, P, X, Y, Z, W, D, Q, H, F, J>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    /// Ordinary Nemotron target with the original retained parameter recipes.
    backbone: R,
    state: std::marker::PhantomData<S>,
    /// Perception, transcription, function head, TTS and codec components.
    speech: LearnedSpeechComponents<B, P, X, Y, Z, W, D, Q, H, F, J>,
    /// Exact tokenizer, character mapping and channel controls.
    host: PreparedHostConditioning,
    silence: PreparedSilenceConditioning,
    residency_report: VoiceChatResidencyReport,
}

/// Architecture operations which prepare immutable VoiceChat startup tensors.
pub trait VoiceChatStartupComponents<T: Tensor> {
    /// Prepares fixed-Aria startup with the admitted codec and codebooks.
    fn prepare_prompt(
        &mut self,
        batch: i32,
        max_token_positions: usize,
        context: &T::Context,
    ) -> Result<PreparedTtsPrompt<T>, TtsError>;
    /// Prepares tool silence without mutating live conversation state.
    fn prepare_silence(
        &mut self,
        dtype: TensorElementType,
        context: &T::Context,
    ) -> Result<T, SilenceConditioningError>;
}

impl<B, S, R, P, X, Y, Z, W, D, Q, H, F, J> VoiceChatStartupComponents<B::Tensor>
    for VoiceChatComponents<B, S, R, P, X, Y, Z, W, D, Q, H, F, J>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    R: LayeredInvocation<B, S, Architecture = nemotron_h::LayeredModel<B>>,
    P: PerceptionExecution<B>,
    X: TransformerExecution<B>,
    Y: MogExecution<B>,
    Z: RvqExecution<B>,
    W: ConditioningExecution<B>,
    D: DecoderExecution<B>,
    Q: QuantizerExecution<B>,
    H: EncoderExecution<B>,
    F: FunctionHeadExecution<B>,
    J: RnntExecution<B>,
{
    /// Prepares fixed-Aria startup using the admitted encoder and shared codebooks.
    /// Cache ownership and prompt execution remain with the enclosing session.
    fn prepare_prompt(
        &mut self,
        batch: i32,
        max_token_positions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedTtsPrompt<B::Tensor>, TtsError> {
        self.speech
            .prepare_prompt(batch, max_token_positions, context)
    }

    /// Prepares tool conditioning without touching live input, RNNT or neural caches.
    /// The enclosing native owner establishes completion before exposing the tensor.
    fn prepare_silence(
        &mut self,
        dtype: TensorElementType,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, SilenceConditioningError> {
        self.speech.prepare_silence(&self.silence, dtype, context)
    }
}

impl<B, S, R, P, X, Y, Z, W, D, Q, H, F, J>
    VoiceChatComponents<B, S, R, P, X, Y, Z, W, D, Q, H, F, J>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
    R: LayeredInvocation<B, S, Architecture = nemotron_h::LayeredModel<B>>,
    P: PerceptionExecution<B>,
    X: TransformerExecution<B>,
    Y: MogExecution<B>,
    Z: RvqExecution<B>,
    W: ConditioningExecution<B>,
    D: DecoderExecution<B>,
    Q: QuantizerExecution<B>,
    H: EncoderExecution<B>,
    F: FunctionHeadExecution<B>,
    J: RnntExecution<B>,
{
    /// Admitted parameter pools and conservative materialization reservation.
    pub fn residency_report(&self) -> &VoiceChatResidencyReport {
        &self.residency_report
    }

    /// Runs local preflight and then enters the caller's submission scope before execution.
    #[allow(clippy::too_many_arguments)]
    pub fn invoke_local<Sampling, N, C>(
        &mut self,
        invocation: &Invocation<'_, B::Tensor>,
        branch: &mut LocalInvocationBranch<B::Tensor, N, Sampling::RandomState, C>,
        input_limits: PerceptionRnntLimits,
        max_character_cells: usize,
        context: &<B::Tensor as Tensor>::Context,
        begin: impl FnOnce(
            &LocalInvocationBranch<B::Tensor, N, Sampling::RandomState, C>,
        ) -> Result<(), BackendFailure>,
    ) -> Result<ExecutionOutput<B::Tensor>, BackendFailure>
    where
        Sampling: DenseSamplingBackend<
            Logits = B::Tensor,
            Token = B::Tensor,
            Context = <B::Tensor as Tensor>::Context,
        >,
        Sampling::RandomState: Clone,
        Sampling::Error: std::error::Error + Send + Sync + 'static,
        R::Error: std::error::Error + Send + Sync + 'static,
        N: RealtimeJointStateParts<Left = S>,
        X: CausalTransformerExecution<B, N::Right>,
    {
        let mut execution = self
            .execution(input_limits, max_character_cells)
            .map_err(BackendFailure::from_error)?;
        execution
            .preflight::<Sampling, _, _, _>(invocation, branch, None)
            .map_err(BackendFailure::from_error)?;
        begin(branch)?;
        execution
            .execute::<Sampling, _, _, _>(invocation, branch, None, context)
            .map_err(BackendFailure::from_error)
    }

    /// Borrows the existing coupled transition; no independent engine is created.
    pub fn execution(
        &mut self,
        input_limits: PerceptionRnntLimits,
        max_character_cells: usize,
    ) -> Result<
        Components<
            '_,
            B,
            super::backbone_execution::OrdinaryBackbone<'_, B, R>,
            super::perception_rnnt::PerceptionRnnt<B, P, J>,
            X,
            Y,
            Z,
            W,
            eredu_codec::ear::AudioDecoder<B, D, Q>,
            F,
        >,
        TtsError,
    > {
        self.speech.execution(
            super::backbone_execution::OrdinaryBackbone::new(&mut self.backbone),
            &self.host,
            input_limits,
            max_character_cells,
        )
    }
}
