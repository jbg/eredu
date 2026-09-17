//! Consumer of the shared JSON framing/field workers and original schema loan.
use super::super::OriginalToolValidation;
use super::*;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub(super) enum ToolCause {
    #[error(transparent)]
    Funding(#[from] eredu_nn::workspace::WorkspaceMetadataFundingError),
    #[error(transparent)]
    Call(#[from] tool_call::Failure),
    #[error(transparent)]
    Complete(#[from] tool_call::CompletionFailure<BackendFailure>),
    #[error(transparent)]
    Frame(#[from] channels::JsonFrameFailure),
    #[error("original tool call/source population changed")]
    Source,
    #[error("original tool call index overflow")]
    Overflow,
}
#[derive(Debug)]
struct ToolFailure {
    cause: ToolCause,
    pending: SpeculativeBuffer<u8>,
    call: Option<Call>,
    source: OriginalSemanticChannelSource,
    funding: WorkspaceMetadataFunding,
}
impl std::fmt::Display for ToolFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.cause, f)
    }
}
impl std::error::Error for ToolFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            ToolCause::Funding(error) => Some(error),
            ToolCause::Call(error) => Some(error),
            ToolCause::Complete(error) => Some(error),
            ToolCause::Frame(error) => Some(error),
            _ => Some(&self.cause),
        }
    }
}
pub(super) fn control_bytes() -> Option<usize> {
    let parts = [
        SharedBackendFailure::control_bytes::<ToolFailure>()?,
        size_of::<ToolFailure>(),
        size_of::<ToolCause>(),
        size_of::<Result<bool, Cause>>(),
        size_of::<Result<(Call, usize, bool), tool_call::Failure>>(),
        size_of::<Result<Call, tool_call::CompletionFailure<BackendFailure>>>(),
        size_of::<Result<(), BackendFailure>>(),
        size_of::<Arc<dyn OriginalToolValidation>>(),
        size_of::<(
            &dyn OriginalToolValidation,
            &str,
            &str,
            &WorkspaceMetadataFunding,
        )>(),
        size_of::<Result<channels::JsonFrameStep, channels::JsonFrameFailure>>(),
        size_of::<(&mut OriginalSemanticChannelParser, JsonFrame)>(),
        size_of::<(&mut OriginalSemanticChannelParser, ToolCause)>(),
        size_of::<(usize, usize, bool)>(),
        size_of::<Option<Call>>(),
        size_of::<Result<Call, tool_call::Failure>>(),
        size_of::<Option<&Arc<dyn OriginalToolValidation>>>(),
        size_of::<Result<&str, std::str::Utf8Error>>(),
        size_of::<SharedBackendFailure>(),
        size_of::<Option<usize>>(),
        size_of::<Result<(), eredu_nn::workspace::WorkspaceMetadataFundingError>>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}
/// Allocation was reserved by the enclosing parser invocation/copy census.
pub(super) fn retained(
    cause: ToolCause,
    pending: SpeculativeBuffer<u8>,
    call: Option<Call>,
    source: &OriginalSemanticChannelSource,
    funding: &WorkspaceMetadataFunding,
) -> Cause {
    Cause::Tool(SharedBackendFailure::new(
        BackendFailureKind::InvalidInput,
        ToolFailure {
            cause,
            pending,
            call,
            source: source.clone(),
            funding: funding.clone(),
        },
    ))
}
impl OriginalSemanticChannelParser {
    fn consume_tool_bytes(&mut self, count: usize) {
        self.pending.copy_within(count..self.used, 0);
        self.used -= count;
    }
    fn tool_failure(&mut self, cause: ToolCause) -> Cause {
        self.used = 0;
        retained(
            cause,
            std::mem::take(&mut self.pending),
            self.call.take(),
            &self.source,
            &self.funding,
        )
    }
    pub(super) fn process_tool(&mut self, frame: JsonFrame) -> Result<bool, Cause> {
        if frame == JsonFrame::Payload {
            if self.used == 0 {
                return Ok(true);
            }
            if self.call.is_none() {
                self.call = Some(
                    Call::prepare(&self.source, self.limit, self.tool_index, &self.funding)
                        .map_err(|cause| self.tool_failure(cause.into()))?,
                );
            }
            let call = self.call.take().expect("prepared call");
            let pending = std::str::from_utf8(&self.pending[..self.used]).expect("UTF-8 input");
            let (call, consumed, complete) = call
                .push(pending, &mut self.events)
                .map_err(|cause| self.tool_failure(cause.into()))?;
            self.call = Some(call);
            self.consume_tool_bytes(consumed);
            if complete {
                self.state = Cursor::Tool(JsonFrame::AfterPayload);
            }
            return Ok(!complete);
        }
        let Some(program) = self.source.json_tools() else {
            return Err(self.tool_failure(ToolCause::Source));
        };
        let pending = std::str::from_utf8(&self.pending[..self.used]).expect("UTF-8 input");
        let step = match channels::json_frame_step(program, frame, pending) {
            Ok(step) => step,
            Err(failure) => {
                self.consume_tool_bytes(failure.consumed);
                return Err(self.tool_failure(failure.into()));
            }
        };
        self.consume_tool_bytes(step.consume_before);
        if step.end_call {
            let Some(validation) = self.source.tool_validation().cloned() else {
                return Err(self.tool_failure(ToolCause::Source));
            };
            let controls = validation.failure_control_bytes()
                .ok_or_else(|| self.tool_failure(ToolCause::Overflow))?;
            self.funding.reserve_metadata(controls)
                .map_err(|cause| self.tool_failure(cause.into()))?;
            let Some(call) = self.call.take() else {
                return Err(self.tool_failure(ToolCause::Source));
            };
            // The same field owner publishes ToolCallEnd only after this actual
            // full-schema callback succeeds. Its failure owns the call prefix.
            let completed = call
                .complete_with(
                    |name, arguments, funding| validation.validate(name, arguments, funding),
                    &mut self.events,
                )
                .map_err(|cause| self.tool_failure(cause.into()))?;
            drop(completed);
            self.tool_index = self
                .tool_index
                .checked_add(1)
                .ok_or_else(|| self.tool_failure(ToolCause::Overflow))?;
        }
        self.consume_tool_bytes(step.consume_after);
        self.state = match step.next {
            JsonFrame::Outside => Cursor::Outside,
            frame => Cursor::Tool(frame),
        };
        Ok(step.wait)
    }
}
