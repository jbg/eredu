//! Architecture-owned speech startup and invocation over native mechanisms.

use eredu_core::realtime_mechanisms::{RealtimeQueue, RealtimeStartupMechanisms};
use std::sync::Arc;

use eredu_architectures::voicechat::{
    components::{BoundVoiceChatComponents, LocalInvocationBranch},
    observation::VoiceChatHostObserver,
    preparation::Generation,
};
use eredu_core::RealtimeDecisionDiagnostics;
use eredu_runtime::{SharedRealtimeExecution, SpeechRealtimeMechanisms};

use super::{
    VoiceChatExecutionError, VoiceChatExecutionInputs, VoiceChatExecutor,
    VoiceChatPublicationBranch, VoiceChatStartError, VoiceChatStartupInputs,
    VoiceChatSubmittedWork, VoiceChatWork,
};
use crate::api::realtime::{PreparedSpeechSelection, ShareSpeechExecution, SpeechRealtimeSession};

type NativeTensor<M> = <<M as eredu_core::realtime_mechanisms::SpeechRealtimeNative>::Neural as eredu_nn::NeuralBackend>::Tensor;

type Components<M> = BoundVoiceChatComponents<
    <M as eredu_core::realtime_mechanisms::SpeechRealtimeNative>::Neural,
    eredu_runtime::state::HybridState<
        <M as eredu_core::realtime_mechanisms::SpeechRealtimeNative>::CacheBackend,
    >,
    <M as eredu_core::realtime_mechanisms::SpeechRealtimeNative>::Components,
    std::convert::Infallible,
>;
type SpeechGeneration<M> = Generation<
    NativeTensor<M>,
    eredu_runtime::SpeechRealtimeState<M>,
    <M as RealtimeStartupMechanisms>::Random,
    <M as RealtimeStartupMechanisms>::Completion,
>;
type SpeechBranch<M> = LocalInvocationBranch<
    NativeTensor<M>,
    eredu_runtime::SpeechRealtimeStateBranch<M>,
    <M as RealtimeStartupMechanisms>::Random,
    <M as RealtimeStartupMechanisms>::Completion,
>;

struct SpeechExecutor<M: SpeechRealtimeMechanisms> {
    execution: SharedRealtimeExecution<Components<M>, M::Queue>,
    inputs: Arc<VoiceChatStartupInputs<NativeTensor<M>>>,
}

impl<M: SpeechRealtimeMechanisms> ShareSpeechExecution for SpeechExecutor<M> {
    fn share_for_fork(&self) -> Self {
        Self {
            execution: self.execution.share(),
            inputs: Arc::clone(&self.inputs),
        }
    }
}

impl<M: SpeechRealtimeMechanisms>
    VoiceChatExecutor<SpeechGeneration<M>, RealtimeDecisionDiagnostics> for SpeechExecutor<M>
{
    fn is_busy(&self) -> bool {
        self.execution.is_busy()
    }

    fn submit(
        &mut self,
        work: &VoiceChatWork<RealtimeDecisionDiagnostics>,
        branch: &mut VoiceChatPublicationBranch<SpeechBranch<M>, RealtimeDecisionDiagnostics>,
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
            eredu_runtime::submit_speech_realtime_frame::<M, _, _, _, _, _, _, _, _>(
                &self.execution,
                branch,
                invocation,
                |state| &state.neural,
                |state, visit| state.visit_component_tensors(visit),
                VoiceChatHostObserver::new(eredu_runtime::RealtimeChannelObserver::new(
                    self.execution.queue().readback(),
                )),
                |components, branch, context, begin| {
                    components.invoke_local::<M::Sampling, _, _>(
                        invocation,
                        branch,
                        self.inputs.input_limits,
                        self.inputs.max_character_cells,
                        context,
                        begin,
                    )
                },
            )
            .map_err(|error| error.with_operation("speech invocation submission"))
        })
    }
}

pub(crate) fn bind_speech<M: SpeechRealtimeMechanisms>(
    selected: PreparedSpeechSelection<'_>,
    mechanisms: &M,
) -> Result<Box<dyn SpeechRealtimeSession>, VoiceChatStartError> {
    let (prepared, resources) = selected.into_startup();
    let binder = mechanisms.components();
    let session = prepared.start(
        resources.parameters(),
        resources.startup(),
        mechanisms,
        |construction, frontend, residency| {
            let (components, input) = construction
                .materialize::<M::Neural, eredu_runtime::state::HybridState<M::CacheBackend>, _, std::convert::Infallible>(
                    frontend, residency, &binder,
                )?;
            Ok((
                SharedRealtimeExecution::new(components, mechanisms.speech_queue()),
                input,
            ))
        },
        |execution, inputs| SpeechExecutor::<M> {
            execution,
            inputs: Arc::new(inputs),
        },
    )?;
    Ok(Box::new(session))
}
