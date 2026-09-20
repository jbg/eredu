//! Paid consumer of the same tagged call and wrapper workers as ordinary parsing.
use super::*;
use eredu_text::semantic_channels::tagged::{
    self as shared, TaggedCall, TaggedEvent, TaggedFrame, TaggedSchemas,
};
struct Schemas<'a> {
    source: &'a dyn super::super::OriginalToolValidation,
    funding: &'a HostMetadataFunding,
}
impl TaggedSchemas for Schemas<'_> {
    type Error = BackendFailure;
    fn contains_tool(&self, name: &str) -> bool {
        self.source.contains_tagged_tool(name)
    }
    fn missing_required(
        &self,
        name: &str,
        parameters: &eredu_text::semantic_channels::tagged::TaggedParameters,
    ) -> bool {
        self.source.tagged_missing_required(name, parameters)
    }
    fn parse_parameter(
        &self,
        name: &str,
        parameter: &str,
        declared: Option<&str>,
        raw: &str,
    ) -> Result<serde_json::Value, BackendFailure> {
        self.source
            .parse_tagged_parameter(name, parameter, declared, raw, self.funding)
    }
}
pub(super) fn controls() -> Option<usize> {
    size_of::<TaggedCall>()
        .checked_add(size_of::<Schemas<'_>>())?
        .checked_add(size_of::<Result<SemanticText, Cause>>())?
        .checked_add(size_of::<
            [u8; eredu_text::json_fragments::GENERATED_CALL_ID_BYTES],
        >())
}
fn text(value: &str, funding: &HostMetadataFunding) -> Result<SemanticText, tool::ToolCause> {
    let bytes = SemanticText::retained_control_bytes(value.len())
        .and_then(|n| {
            n.checked_add(HostPreparationAuthority::retention_bytes::<
                HostMetadataFunding,
            >()?)
        })
        .ok_or(tool::ToolCause::Overflow)?;
    funding.reserve_metadata(bytes)?;
    SemanticText::try_copy_retained(value, HostPreparationAuthority::retain(funding.clone()))
        .map_err(tool::ToolCause::Text)
}
struct Identifier {
    bytes: [u8; eredu_text::json_fragments::GENERATED_CALL_ID_BYTES],
    len: usize,
}
impl std::fmt::Write for Identifier {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        let end = self.len.checked_add(value.len()).ok_or(std::fmt::Error)?;
        self.bytes
            .get_mut(self.len..end)
            .ok_or(std::fmt::Error)?
            .copy_from_slice(value.as_bytes());
        self.len = end;
        Ok(())
    }
}
impl OriginalSemanticChannelParser {
    pub(super) fn process_tagged(&mut self, frame: TaggedFrame) -> Result<bool, Cause> {
        let Some(program) = self.source.tagged_tools() else {
            return Err(self.tool_failure(tool::ToolCause::Source));
        };
        let Some(validation) = self.source.tool_validation().cloned() else {
            return Err(self.tool_failure(tool::ToolCause::Source));
        };
        self.funding
            .reserve_metadata(validation.failure_control_bytes().ok_or(Cause::Overflow)?)?;
        if frame == TaggedFrame::Payload {
            if self.used == 0 {
                return Ok(true);
            }
            let call = self.tagged.get_or_insert_with(TaggedCall::default);
            let schemas = Schemas {
                source: validation.as_ref(),
                funding: &self.funding,
            };
            let pending = std::str::from_utf8(&self.pending[..self.used]).expect("UTF-8 input");
            let result = call.advance(program.encoding, pending, &schemas);
            let (consumed, wait, event) =
                result.map_err(|error| self.tool_failure(tool::ToolCause::Tagged(error)))?;
            self.consume_tool_bytes(consumed);
            match event {
                TaggedEvent::None => {}
                TaggedEvent::Start => {
                    let call = self.tagged.as_ref().expect("started tagged call");
                    let name =
                        text(call.name(), &self.funding).map_err(|e| self.tool_failure(e))?;
                    let mut id = Identifier {
                        bytes: [0; eredu_text::json_fragments::GENERATED_CALL_ID_BYTES],
                        len: 0,
                    };
                    eredu_text::json_fragments::write_generated_call_id(&mut id, self.tool_index)
                        .map_err(|_| self.tool_failure(tool::ToolCause::Overflow))?;
                    let id = text(
                        std::str::from_utf8(&id.bytes[..id.len]).expect("ASCII identifier"),
                        &self.funding,
                    )
                    .map_err(|e| self.tool_failure(e))?;
                    self.events
                        .try_push(SemanticEvent::ToolCallStart {
                            index: self.tool_index,
                            id,
                            name,
                        })
                        .map_err(|_| self.tool_failure(tool::ToolCause::Capacity))?;
                }
                TaggedEvent::Complete => {
                    let arguments = text(
                        self.tagged.as_ref().expect("complete call").arguments(),
                        &self.funding,
                    )
                    .map_err(|e| self.tool_failure(e))?;
                    self.events
                        .try_push(SemanticEvent::ToolArgumentsDelta {
                            index: self.tool_index,
                            json_fragment: arguments,
                        })
                        .map_err(|_| self.tool_failure(tool::ToolCause::Capacity))?;
                    self.state = Cursor::Tagged(TaggedFrame::AfterPayload);
                }
            }
            return Ok(wait);
        }
        let pending = std::str::from_utf8(&self.pending[..self.used]).expect("UTF-8 input");
        let step = shared::tagged_frame_step(program, frame, pending)
            .map_err(|error| self.tool_failure(tool::ToolCause::TaggedSyntax(error)))?;
        self.consume_tool_bytes(step.consumed);
        if step.end_call {
            let call = self.tagged.as_ref().ok_or(Cause::ToolPayload)?;
            validation
                .validate(call.name(), call.arguments(), &self.funding)
                .map_err(|error| self.tool_failure(tool::ToolCause::Schema(error)))?;
            self.events
                .try_push(SemanticEvent::ToolCallEnd)
                .map_err(|_| self.tool_failure(tool::ToolCause::Capacity))?;
            self.tagged = None;
            self.tool_index = self
                .tool_index
                .checked_add(1)
                .ok_or_else(|| self.tool_failure(tool::ToolCause::Overflow))?;
        }
        self.state = match step.next {
            TaggedFrame::Outside => Cursor::Outside,
            frame => Cursor::Tagged(frame),
        };
        Ok(step.wait)
    }
}
