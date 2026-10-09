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
use eredu_nn::{
    AttentionCache, DistributedNeuralBackend, GroupedNeuralBackend, Tensor, TensorElementType,
};
use eredu_runtime::{
    LayerRuntimeState, LayerWeightResidency, LayeredInvocation, RuntimeStateComponents,
};

use crate::nemotron_h;

use super::{
    agent_sampling::AgentSamplingPolicy,
    artifact::RetainedVoiceChatArtifacts,
    assembly::LearnedSpeechComponents,
    backbone::{BackboneConstructionError, PreparedBackboneParameters},
    checkpoint::Component,
    component_residency::{
        VoiceChatResidencyError, VoiceChatResidencyPolicy, VoiceChatResidencyReport,
    },
    execution::Components,
    frontend::FrontendOptions,
    function_head_layered::FunctionHeadExecution,
    host::{HostPreparationError, PreparedHostConditioning},
    input::{InputError, InputState},
    perception_layered::PerceptionExecution,
    perception_rnnt::PerceptionRnntLimits,
    preparation::{GenerationPreparationError, PreparedGeneration},
    provider_construction::{
        PreparedSpeechProviders, ProviderConstructionError, SpeechProviderBinder,
    },
    schedule::ScheduleLimits,
    silence::{PreparedSilenceConditioning, SilenceConditioningError},
    state::{NeuralStateGeometry, NeuralStateGeometryError},
    tts_conditioning_layered::ConditioningExecution,
    tts_mog::TtsError,
    tts_mog_layered::MogExecution,
    tts_prompt::PreparedTtsPrompt,
    tts_rvq_layered::RvqExecution,
    tts_transformer_layered::TransformerExecution,
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
    Backend(#[source] eredu_core::BackendFailure),
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

/// Binds the selected backbone through the same layered invocation used by the executor.
/// The architecture retains artifact interpretation and component composition.
pub trait BackboneProviderBinder<B, S>: SpeechProviderBinder<B>
where
    B: GroupedNeuralBackend + DistributedNeuralBackend,
    S: LayerRuntimeState<B>,
    S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
{
    /// Ordinary Nemotron runtime with the selected parameter provider.
    type Backbone: LayeredInvocation<B, S, Architecture = nemotron_h::LayeredModel<B>>;

    /// Materializes exact retained recipes under the admitted residency policy.
    fn backbone(
        &mut self,
        prepared: &PreparedBackboneParameters,
        residency: &LayerWeightResidency,
    ) -> Result<Self::Backbone, eredu_core::BackendFailure>;
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
        let neural_state_geometry =
            NeuralStateGeometry::new(backbone.args(), &artifacts.plan().config().tts.backbone)?;
        Ok(Self {
            artifacts,
            backbone,
            speech: PreparedSpeechProviders::from_retained(artifacts)?,
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
    pub fn materialize<B, S, V>(
        self,
        options: FrontendOptions,
        policy: &VoiceChatResidencyPolicy,
        visitor: &mut V,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<
        (BoundVoiceChatComponents<B, S, V>, InputState<B::Tensor>),
        VoiceChatConstructionError,
    >
    where
        B: GroupedNeuralBackend + DistributedNeuralBackend,
        S: LayerRuntimeState<B>,
        S::LayerState: AttentionCache<B::Tensor> + RuntimeStateComponents<B>,
        V: BackboneProviderBinder<B, S>,
    {
        let residency_report = policy.validate(&self.backbone, &self.speech)?;
        let input = InputState::from_retained(self.artifacts, options)?;
        let (speech, _) = self
            .speech
            .materialize::<B, V>(policy.speech(), visitor, context)?;
        let backbone = visitor
            .backbone(&self.backbone, &policy.backbone())
            .map_err(VoiceChatConstructionError::Backend)?;
        Ok((
            VoiceChatComponents {
                backbone,
                state: std::marker::PhantomData,
                speech,
                host: self.host,
                silence: self.silence,
                neural_state_geometry: self.neural_state_geometry,
                residency_report,
            },
            input,
        ))
    }
}

/// Selected provider types assembled through a native binding visitor.
pub type BoundVoiceChatComponents<B, S, V> = VoiceChatComponents<
    B,
    S,
    <V as BackboneProviderBinder<B, S>>::Backbone,
    <V as SpeechProviderBinder<B>>::Perception,
    <V as SpeechProviderBinder<B>>::Transformer,
    <V as SpeechProviderBinder<B>>::Mog,
    <V as SpeechProviderBinder<B>>::Rvq,
    <V as SpeechProviderBinder<B>>::Conditioning,
    <V as SpeechProviderBinder<B>>::Decoder,
    <V as SpeechProviderBinder<B>>::Quantizer,
    <V as SpeechProviderBinder<B>>::Encoder,
    <V as SpeechProviderBinder<B>>::Function,
    <V as SpeechProviderBinder<B>>::Rnnt,
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
    neural_state_geometry: NeuralStateGeometry,
    residency_report: VoiceChatResidencyReport,
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

    /// Exact geometry retained from cold construction, including both TTS rows.
    pub fn neural_state_geometry(&self) -> &NeuralStateGeometry {
        &self.neural_state_geometry
    }

    /// Exact tokenizer, character mapping and channel controls.
    pub fn host_conditioning(&self) -> &PreparedHostConditioning {
        &self.host
    }

    /// Prepares fixed-Aria startup using the admitted encoder and shared codebooks.
    /// Cache ownership and prompt execution remain with the enclosing session.
    pub fn prepare_prompt(
        &mut self,
        batch: i32,
        max_token_positions: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PreparedTtsPrompt<B::Tensor>, TtsError> {
        self.speech
            .prepare_prompt(batch, max_token_positions, context)
    }

    /// Immutable host policy and exact coefficients for one-second tool silence.
    pub fn silence_conditioning(&self) -> &PreparedSilenceConditioning {
        &self.silence
    }

    /// Prepares tool conditioning without touching live input, RNNT or neural caches.
    /// The enclosing native owner establishes completion before exposing the tensor.
    pub fn prepare_silence(
        &mut self,
        dtype: TensorElementType,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<B::Tensor, SilenceConditioningError> {
        self.speech.prepare_silence(&self.silence, dtype, context)
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
