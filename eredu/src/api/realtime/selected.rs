//! Native binding of a retained realtime selection.

use super::{
    prepare_realtime_selection, PreparedRealtimeSelection, RealtimeLoadError, RealtimeLoadOptions,
};
use crate::runtime::voicechat::{start_mlx_voicechat, MlxVoiceChatSession};
use eredu_architectures::{
    moshi::MoshiRealtimeExecution, realtime::PreparedRealtimeArtifact,
    voicechat::component_residency::VoiceChatResidencyReport,
};
use eredu_backend_mlx::native::{
    ExecutionContext, MlxRealtimeExecution, MlxRealtimeExecutionContext, MlxRealtimeSupport,
};
use eredu_core::BackendFailure;

fn backend(error: impl std::error::Error + Send + Sync + 'static) -> RealtimeLoadError {
    RealtimeLoadError::Backend(BackendFailure::from_error(error))
}

/// Complete cold selection for local MLX binding.
pub struct PreparedMlxRealtime<'a> {
    selection: PreparedRealtimeSelection<'a>,
}

/// Bound execution retaining the architecture's declared input protocol.
pub enum LoadedMlxRealtime {
    /// Categorical frames use the shared realtime scheduler and model state.
    Categorical {
        /// Native stream mechanisms retained for submissions and state creation.
        context: MlxRealtimeExecutionContext,
        /// Architecture-owned selected categorical execution.
        model: MoshiRealtimeExecution<MlxRealtimeExecution>,
    },
    /// PCM speech uses the shared controlled/iterator conversation session.
    Speech(MlxVoiceChatSession),
}

/// Selects from a discovered artifact without creating a native device or tensor.
pub fn prepare_mlx_realtime<'a>(
    artifact: &'a PreparedRealtimeArtifact,
    options: &RealtimeLoadOptions,
) -> Result<PreparedMlxRealtime<'a>, RealtimeLoadError> {
    let selection = prepare_realtime_selection(artifact, options, &MlxRealtimeSupport::new(false))?;
    Ok(PreparedMlxRealtime { selection })
}

impl PreparedMlxRealtime<'_> {
    /// Exact aggregate speech parameter admission, when the protocol uses PCM.
    pub fn speech_residency_report(&self) -> Option<VoiceChatResidencyReport> {
        self.selection.speech_residency_report()
    }

    /// Materializes the retained selection through the existing submission owners.
    /// Sources and host policy cannot change between selection and this binding.
    pub fn load(
        self,
        weights: &ExecutionContext,
        execution: &ExecutionContext,
    ) -> Result<LoadedMlxRealtime, RealtimeLoadError> {
        match self.selection {
            PreparedRealtimeSelection::Categorical(selected) => {
                let context =
                    MlxRealtimeExecutionContext::new(execution.stream(), weights.stream());
                let model = context.bind_realtime(selected, None).map_err(backend)?;
                Ok(LoadedMlxRealtime::Categorical { context, model })
            }
            PreparedRealtimeSelection::Speech(selected) => {
                let (prepared, resources) = selected.into_startup();
                Ok(LoadedMlxRealtime::Speech(start_mlx_voicechat(
                    prepared,
                    resources.parameters(),
                    resources.startup(),
                    weights,
                    execution,
                )?))
            }
        }
    }
}
