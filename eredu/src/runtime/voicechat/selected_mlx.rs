//! Selected native mechanisms for the portable prepared-session driver.

use std::sync::Arc;

use eredu_architectures::voicechat::{
    component_residency::VoiceChatResidencyPolicy,
    components::{BoundVoiceChatComponents, LocalInvocationBranch},
    observation::VoiceChatHostObserver,
    preparation::Generation,
};
use eredu_backend_mlx::{
    backend::{
        nn::shared::MlxNeuralBackend,
        runtime::{
            cache::state::{MlxHybridState, MlxJointState, MlxJointStateBranch},
            generation::MlxSamplingBackend,
            realtime::MlxRealtimeChannelObserver,
        },
    },
    native::{
        ExecutionContext, MlxComponentBinder, MlxRealtimeCompletion, MlxRealtimeExecutionContext,
        MlxReplicatedCoordinator, RandomState, SharedRealtimeExecution,
    },
    MlxTensor,
};
use eredu_core::RealtimeDecisionDiagnostics;

use super::{
    PreparedVoiceChatSession, VoiceChatExecutionError, VoiceChatExecutionInputs, VoiceChatExecutor,
    VoiceChatLoadSelection, VoiceChatPublicationBranch, VoiceChatStartError,
    VoiceChatStartupInputs, VoiceChatSubmittedWork, VoiceChatWork,
};

type MlxVoiceChatComponents = BoundVoiceChatComponents<
    MlxNeuralBackend,
    MlxHybridState,
    MlxComponentBinder,
    MlxReplicatedCoordinator,
>;
type MlxVoiceChatGeneration =
    Generation<MlxTensor, MlxJointState, RandomState, MlxRealtimeCompletion>;
type MlxVoiceChatBranch =
    LocalInvocationBranch<MlxTensor, MlxJointStateBranch, RandomState, MlxRealtimeCompletion>;

/// Native execution resources and immutable startup inputs for a bound session.
/// Neural caches, RNG, conversation policy and delivered events live in the
/// ordinary session transaction and are never copied by this executor.
struct MlxVoiceChatExecutor {
    execution: SharedRealtimeExecution<MlxVoiceChatComponents>,
    inputs: Arc<VoiceChatStartupInputs<MlxTensor>>,
}

impl crate::api::realtime::ShareSpeechExecution for MlxVoiceChatExecutor {
    fn share_for_fork(&self) -> Self {
        Self {
            execution: self.execution.share(),
            inputs: Arc::clone(&self.inputs),
        }
    }
}

impl VoiceChatExecutor<MlxVoiceChatGeneration, RealtimeDecisionDiagnostics>
    for MlxVoiceChatExecutor
{
    fn is_busy(&self) -> bool {
        self.execution.is_busy()
    }

    fn submit(
        &mut self,
        work: &VoiceChatWork<RealtimeDecisionDiagnostics>,
        branch: &mut VoiceChatPublicationBranch<MlxVoiceChatBranch, RealtimeDecisionDiagnostics>,
    ) -> Result<VoiceChatSubmittedWork, VoiceChatExecutionError> {
        let inputs = VoiceChatExecutionInputs {
            prompt: Some(&self.inputs.prompt),
            speaker_codes: self.inputs.speaker.codes(),
            speaker_layout: self.inputs.speaker.layout(),
            speaker_conditioning: None,
            silence: Some(&self.inputs.silence),
            speech: self.inputs.speech,
        };
        work.submit(branch, &inputs, |branch, invocation| {
            let observer = VoiceChatHostObserver::new(MlxRealtimeChannelObserver::new(
                self.execution.stream(),
            ));
            self.execution
                .submit(
                    branch,
                    invocation,
                    |branch| {
                        let mut roots = branch
                            .model_state()
                            .neural
                            .retained_arrays()
                            .into_iter()
                            .cloned()
                            .collect::<Vec<_>>();
                        branch.model_state().visit_component_tensors(&mut |tensor| {
                            roots.push(tensor.as_array().clone());
                        });
                        roots.extend(
                            branch
                                .random_state()
                                .map(|random| random.as_array().clone()),
                        );
                        roots
                    },
                    |branch| branch.model_state().neural.retention(),
                    observer,
                    |components, branch, context, begin| {
                        components.invoke_local::<MlxSamplingBackend, _, _>(
                            invocation,
                            branch,
                            self.inputs.input_limits,
                            self.inputs.max_character_cells,
                            context,
                            begin,
                        )
                    },
                )
                .map_err(|error| error.with_operation("MLX VoiceChat submission"))
        })
    }
}

/// Binds an admitted request without reopening artifacts or advancing live conversation.
/// Fixed-Aria and silence preparation establish native completion before their
/// tensors are exposed. Prompt prefill is queued on the ordinary session and
/// executes only when the application advances it with an explicit timestamp.
pub(crate) fn start_mlx_voicechat(
    prepared: PreparedVoiceChatSession<'_>,
    residency: &VoiceChatResidencyPolicy,
    options: VoiceChatLoadSelection<'_>,
    weights: &ExecutionContext,
    execution: &ExecutionContext,
) -> Result<Box<dyn crate::api::realtime::SpeechRealtimeSession>, VoiceChatStartError> {
    let mechanisms = MlxRealtimeExecutionContext::new(execution.stream(), weights.stream());
    let session = prepared.start(
        residency,
        options,
        &mechanisms,
        |construction, frontend, residency| {
            let binder = MlxComponentBinder::new(weights.stream(), execution.stream());
            let (components, input) = construction.materialize(frontend, residency, &binder)?;
            Ok((
                SharedRealtimeExecution::new(components, execution.stream()),
                input,
            ))
        },
        |execution, inputs| MlxVoiceChatExecutor {
            execution,
            inputs: Arc::new(inputs),
        },
    )?;
    Ok(Box::new(session))
}
