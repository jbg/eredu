//! Native binding of the portable facade submission driver.

use eredu_architectures::voicechat::perception_rnnt::PerceptionRnntLimits;
use eredu_backend_mlx::{
    native::{MlxVoiceChatBranch, MlxVoiceChatExecution},
    MlxTensor,
};
use eredu_core::{BackendFailure, RealtimeDecisionDiagnostics};

use super::{VoiceChatExecutionError, VoiceChatExecutionInputs, VoiceChatSubmittedWork};
use crate::runtime::voicechat::{VoiceChatPublicationBranch, VoiceChatWork};

/// Submits prepared facade work through the ordinary MLX completion mechanism.
///
/// The caller supplies already materialized execution and prepared input tensors.
/// No model is selected or loaded here. The portable driver owns invocation and
/// rendering policy; MLX owns scopes, retained native resources and recovery.
pub fn submit_mlx_voicechat(
    work: &VoiceChatWork<RealtimeDecisionDiagnostics>,
    branch: &mut VoiceChatPublicationBranch<MlxVoiceChatBranch, RealtimeDecisionDiagnostics>,
    execution: &mut MlxVoiceChatExecution,
    inputs: &VoiceChatExecutionInputs<'_, MlxTensor>,
    input_limits: PerceptionRnntLimits,
    max_character_cells: usize,
) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError> {
    work.submit(branch, inputs, |branch, invocation| {
        execution
            .submit(branch, invocation, input_limits, max_character_cells)
            .map_err(|error| {
                BackendFailure::from_error(error).with_operation("MLX VoiceChat submission")
            })
    })
}
