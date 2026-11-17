//! Backend-selected binding of a retained realtime plan.

use eredu_core::DevicePlan;
use eredu_runtime::RealtimeLoadSupport;

use super::{
    prepare_realtime_selection, CategoricalRealtimeModel, PreparedRealtimeArtifact,
    PreparedRealtimeSelection, PreparedSpeechSelection, RealtimeLoadError, RealtimeLoadOptions,
    SpeechRealtimeSession,
};
use eredu_architectures::moshi::PreparedMoshiRealtimeSource;

/// Supplies cold capability facts and binds the protocol selected by the artifact.
pub trait RealtimeBackendFactory {
    /// Capability predicates used before sources or devices are bound.
    type Support: RealtimeLoadSupport;

    /// Reports exact, side-effect-free support for local realtime loading.
    fn realtime_support(&self) -> Self::Support;

    /// Binds a retained categorical execution on the selected device.
    fn bind_categorical(
        &self,
        selected: PreparedMoshiRealtimeSource,
        device: &DevicePlan,
    ) -> Result<Box<dyn CategoricalRealtimeModel>, RealtimeLoadError>;

    /// Binds a retained PCM speech execution on the selected device.
    fn bind_speech(
        &self,
        selected: PreparedSpeechSelection<'_>,
        device: &DevicePlan,
    ) -> Result<Box<dyn SpeechRealtimeSession>, RealtimeLoadError>;
}

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
    match prepared.selected {
        PreparedRealtimeSelection::Categorical(selected) => prepared
            .factory
            .bind_categorical(selected, device)
            .map(RealtimeSession::Categorical),
        PreparedRealtimeSelection::Speech(selected) => prepared
            .factory
            .bind_speech(*selected, device)
            .map(RealtimeSession::Speech),
    }
}

#[cfg(feature = "mlx")]
mod mlx {
    use super::*;
    use eredu_backend_mlx::{
        native::{ExecutionContext, MlxRealtimeExecutionContext, MlxRealtimeSupport},
        MlxBackendFactory,
    };
    use eredu_core::BackendFailure;

    fn backend(error: impl std::error::Error + Send + Sync + 'static) -> RealtimeLoadError {
        RealtimeLoadError::Backend(BackendFailure::from_error(error))
    }

    impl RealtimeBackendFactory for MlxBackendFactory {
        type Support = MlxRealtimeSupport;

        fn realtime_support(&self) -> Self::Support {
            MlxRealtimeSupport::new(false)
        }

        fn bind_categorical(
            &self,
            selected: PreparedMoshiRealtimeSource,
            device: &DevicePlan,
        ) -> Result<Box<dyn CategoricalRealtimeModel>, RealtimeLoadError> {
            let weights = ExecutionContext::cpu().map_err(backend)?;
            let execution = self.execution_context(device).map_err(backend)?;
            let context = MlxRealtimeExecutionContext::new(execution.stream(), weights.stream());
            let model = context.bind_realtime(selected, None).map_err(backend)?;
            Ok(Box::new(std::rc::Rc::new(
                super::super::selected_categorical_mlx::MlxCategoricalModel::new(context, model),
            )))
        }

        fn bind_speech(
            &self,
            selected: PreparedSpeechSelection<'_>,
            device: &DevicePlan,
        ) -> Result<Box<dyn SpeechRealtimeSession>, RealtimeLoadError> {
            let weights = ExecutionContext::cpu().map_err(backend)?;
            let execution = self.execution_context(device).map_err(backend)?;
            let (prepared, resources) = selected.into_startup();
            Ok(crate::runtime::voicechat::start_mlx_voicechat(
                prepared,
                resources.parameters(),
                resources.startup(),
                &weights,
                &execution,
            )?)
        }
    }
}
