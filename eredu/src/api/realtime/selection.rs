//! Portable cold selection of realtime sources and execution resources.

use super::RealtimeLoadOptions;
use crate::runtime::voicechat::{
    PreparedVoiceChatSession, VoiceChatLoadingError, VoiceChatStartError,
};
use eredu_architectures::{
    lfm3::audio::PreparedAudio,
    moshi::{
        prepare_moshi_realtime_with_support, MoshiPreparationPlan, MoshiRealtimePreparationError,
        PreparedMoshiRealtimeSource,
    },
    realtime::{PreparedRealtimeArtifact, RealtimeArtifactVisitor},
    voicechat::{
        artifact::RetainedVoiceChatArtifacts,
        component_residency::{
            VoiceChatResidencyError, VoiceChatResidencyPolicy, VoiceChatResidencyReport,
        },
        state::VoiceChatStateSelectionError,
    },
};
use eredu_core::BackendFailure;
use eredu_runtime::{RealtimeCopyLimits, RealtimeLoadSupport, SelectedStateRealization};

/// Portable failure during cold selection or native binding.
#[derive(Debug, thiserror::Error)]
pub enum RealtimeLoadError {
    /// The requested host protocol differs from the discovered architecture.
    #[error("realtime artifact requires {required} input, but loading requested {requested}")]
    Protocol {
        /// Protocol declared by the discovered architecture.
        required: &'static str,
        /// Protocol selected by the application.
        requested: &'static str,
    },
    /// Categorical architecture selection or source preparation failed.
    #[error(transparent)]
    Categorical(#[from] MoshiRealtimePreparationError),
    /// The selected backend cannot retain speech state and read completed outputs.
    #[error(transparent)]
    SpeechCapability(#[from] eredu_core::SessionCapabilityError),
    /// Speech host preparation failed before construction.
    #[error(transparent)]
    SpeechPreparation(#[from] VoiceChatLoadingError),
    /// Exact parameter pools cannot admit the selected graph.
    #[error(transparent)]
    Residency(#[from] VoiceChatResidencyError),
    /// Selected cache mechanisms cannot implement the declared state.
    #[error(transparent)]
    State(#[from] VoiceChatStateSelectionError),
    /// Speech session binding failed.
    #[error(transparent)]
    SpeechStart(#[from] VoiceChatStartError),
    /// Native failure with its original error retained as source.
    #[error("realtime backend failed: {0}")]
    Backend(#[source] BackendFailure),
}

/// Retained VoiceChat policy, state and source authority.
pub struct PreparedSpeechSelection<'a> {
    prepared: PreparedVoiceChatSession<'a>,
    resources: PreparedSpeechResources,
    report: VoiceChatResidencyReport,
}
/// Portable selected source and resource authority for realtime binding.
pub enum PreparedRealtimeSelection<'a> {
    /// Prepared categorical source and selected realtime realization.
    Categorical(PreparedMoshiRealtimeSource),
    /// Exact PCM conversation plan.
    Speech(Box<PreparedSpeechSelection<'a>>),
}

/// Selects exact realtime resources from an admitted artifact and cold backend facts.
pub fn prepare_realtime_selection<'a>(
    artifact: &'a PreparedRealtimeArtifact,
    options: &RealtimeLoadOptions,
    support: &impl RealtimeLoadSupport,
) -> Result<PreparedRealtimeSelection<'a>, RealtimeLoadError> {
    artifact.visit(Select { options, support })
}

impl PreparedRealtimeSelection<'_> {
    /// Admitted speech parameter pools for a PCM conversation.
    pub fn speech_residency_report(&self) -> Option<VoiceChatResidencyReport> {
        match self {
            Self::Speech(selection) => Some(selection.report),
            _ => None,
        }
    }
}

impl<'a> PreparedSpeechSelection<'a> {
    /// Exposes the admitted policy and startup selections alongside the owned session.
    pub fn into_startup(self) -> (PreparedVoiceChatSession<'a>, PreparedSpeechResources) {
        (self.prepared, self.resources)
    }
}

/// Resources retained for portable VoiceChat startup.
pub struct PreparedSpeechResources {
    parameters: VoiceChatResidencyPolicy,
    backbone: SelectedStateRealization,
    tts: SelectedStateRealization,
    copy_limits: RealtimeCopyLimits,
    seed: u64,
}

impl PreparedSpeechResources {
    /// Complete parameter policy admitted for binding.
    pub fn parameters(&self) -> &VoiceChatResidencyPolicy {
        &self.parameters
    }

    /// Exact selected state, copy limits and RNG seed.
    pub fn startup(&self) -> crate::runtime::voicechat::VoiceChatLoadSelection<'_> {
        crate::runtime::voicechat::VoiceChatLoadSelection {
            backbone: &self.backbone,
            tts: &self.tts,
            copy_limits: self.copy_limits,
            seed: self.seed,
        }
    }
}

struct Select<'a, S> {
    options: &'a RealtimeLoadOptions,
    support: &'a S,
}
impl<'a, S: RealtimeLoadSupport> RealtimeArtifactVisitor<'a> for Select<'_, S> {
    type Output = PreparedRealtimeSelection<'a>;
    type Error = RealtimeLoadError;
    fn moshi(self, artifact: &'a MoshiPreparationPlan) -> Result<Self::Output, Self::Error> {
        let RealtimeLoadOptions::Categorical(request) = self.options else {
            return Err(RealtimeLoadError::Protocol {
                required: "categorical",
                requested: self.options.protocol(),
            });
        };
        Ok(PreparedRealtimeSelection::Categorical(
            prepare_moshi_realtime_with_support(artifact.clone(), request, self.support)?,
        ))
    }
    fn lfm3_audio(self, _: &'a PreparedAudio) -> Result<Self::Output, Self::Error> {
        Err(RealtimeLoadError::Protocol {
            required: "prepared Audio task",
            requested: self.options.protocol(),
        })
    }

    fn voicechat(
        self,
        artifacts: &'a RetainedVoiceChatArtifacts,
    ) -> Result<Self::Output, Self::Error> {
        let RealtimeLoadOptions::Speech(options) = self.options else {
            return Err(RealtimeLoadError::Protocol {
                required: "PCM conversation",
                requested: self.options.protocol(),
            });
        };
        eredu_core::SessionCapabilities::new(true, true, false)
            .validate(&self.support.facts().session)?;
        let prepared = PreparedVoiceChatSession::from_retained(artifacts, &options.conversation)?
            .with_quantization(options.quantization)?;
        let construction = prepared.construction();
        let [backbone, tts] = construction
            .neural_state_geometry()
            .select_state_realizations(&options.state, self.support.independent_state_support())?;
        let report = options.parameters.validate(
            construction.backbone_parameters(),
            construction.speech_providers(),
        )?;
        Ok(PreparedRealtimeSelection::Speech(Box::new(
            PreparedSpeechSelection {
                prepared,
                report,
                resources: PreparedSpeechResources {
                    parameters: options.parameters.clone(),
                    backbone,
                    tts,
                    copy_limits: RealtimeCopyLimits {
                        max_branch_bytes: options.max_branch_bytes,
                        max_total_copy_bytes: options.max_total_copy_bytes,
                    },
                    seed: options.seed,
                },
            },
        )))
    }
}
