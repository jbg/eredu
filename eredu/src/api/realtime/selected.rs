//! Backend-selected binding of a retained realtime plan.

use eredu_core::DevicePlan;
use eredu_runtime::RealtimeBackendFactory;

use super::{
    categorical_binding::bind_categorical, prepare_realtime_selection, CategoricalRealtimeModel,
    PreparedRealtimeArtifact, PreparedRealtimeSelection, RealtimeLoadError, RealtimeLoadOptions,
    SpeechRealtimeSession,
};

/// Bound execution retaining the artifact's declared host input protocol.
pub enum RealtimeSession {
    /// Categorical frames use the shared realtime scheduler and model state.
    Categorical(Box<dyn CategoricalRealtimeModel>),
    /// PCM speech uses the shared controlled conversation session.
    Speech(Box<dyn SpeechRealtimeSession>),
}

/// A retained protocol selection tied to the factory that admitted it.
pub struct PreparedRealtime<'a, F> {
    factory: &'a F,
    selected: PreparedRealtimeSelection<'a>,
}

impl<F> PreparedRealtime<'_, F> {
    /// Admitted speech parameter pools for a PCM conversation.
    pub fn speech_residency_report(
        &self,
    ) -> Option<eredu_architectures::voicechat::component_residency::VoiceChatResidencyReport> {
        self.selected.speech_residency_report()
    }
}

/// Selects a realtime protocol using the chosen backend's cold support predicates.
pub fn prepare_realtime<'a, F: RealtimeBackendFactory>(
    factory: &'a F,
    artifact: &'a PreparedRealtimeArtifact,
    options: &RealtimeLoadOptions,
) -> Result<PreparedRealtime<'a, F>, RealtimeLoadError> {
    let selected = prepare_realtime_selection(artifact, options, &factory.realtime_support())?;
    Ok(PreparedRealtime { factory, selected })
}

/// Binds a retained selection through the factory that admitted it.
pub fn bind_realtime<F: RealtimeBackendFactory>(
    prepared: PreparedRealtime<'_, F>,
    device: &DevicePlan,
) -> Result<RealtimeSession, RealtimeLoadError> {
    let mechanisms = prepared
        .factory
        .open_realtime(device)
        .map_err(RealtimeLoadError::Backend)?;
    match prepared.selected {
        PreparedRealtimeSelection::Categorical(selected) => bind_categorical(selected, mechanisms)
            .map(RealtimeSession::Categorical)
            .map_err(RealtimeLoadError::Backend),
        PreparedRealtimeSelection::Speech(selected) => {
            crate::runtime::voicechat::bind_speech(*selected, &mechanisms)
                .map(RealtimeSession::Speech)
                .map_err(Into::into)
        }
    }
}
