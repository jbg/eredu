//! Structurally recognized Gemma channel protocol.

use serde_json::Value;

use super::{
    dialect::{
        ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
        DECLARATIVE_DIALECT,
    },
    ParallelToolCallPolicy, ToolChoice, GEMMA4_STRUCTURAL_TOOL_SPEC,
};
use crate::runtime::generation::streaming::{ProtocolParser, SemanticEventSink};

pub(crate) const CHANNEL_OPEN: &str = "<|channel>";
pub(crate) const CHANNEL_CLOSE: &str = "<channel|>";
pub(crate) const TOOL_CALL_OPEN: &str = "<|tool_call>";
pub(crate) const TOOL_CALL_CLOSE: &str = "<tool_call|>";
pub(crate) const STRING_DELIMITER: &str = "<|\"|>";
pub(crate) const TOOL_RESPONSE_OPEN: &str = "<|tool_response>";
pub(crate) const TURN_CLOSE: &str = "<turn|>";

#[derive(Debug)]
pub(crate) struct GemmaChannelDialect;

pub(crate) static GEMMA_CHANNEL_DIALECT: GemmaChannelDialect = GemmaChannelDialect;
pub(crate) static GEMMA_TOOL_DIALECT: GemmaToolDialect = GemmaToolDialect;
static GEMMA_CHANNEL_PARAMETERS: () = ();

pub(crate) fn parameters() -> DialectParameters {
    DialectParameters::Custom(&GEMMA_CHANNEL_PARAMETERS)
}

#[derive(Debug)]
pub(crate) struct GemmaToolDialect;

impl FormatDialect for GemmaToolDialect {
    fn supports_reasoning_parsing(&self, _parameters: DialectParameters) -> bool {
        true
    }

    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        DECLARATIVE_DIALECT.generation_prompt_behavior(parameters)
    }

    fn reasoning_template_kwarg(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static str, String> {
        DECLARATIVE_DIALECT.reasoning_template_kwarg(parameters)
    }

    fn supports_tool_reasoning(&self, parameters: DialectParameters) -> Result<bool, String> {
        DECLARATIVE_DIALECT.supports_tool_reasoning(parameters)
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        tools: &[Value],
        tool_choice: ToolChoice,
        parallel_tool_calls: ParallelToolCallPolicy,
        resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        DECLARATIVE_DIALECT.constraint_configuration(
            parameters,
            tools,
            tool_choice,
            parallel_tool_calls,
            resolved_structural_token_ids,
        )
    }

    fn auto_activation_trigger(
        &self,
        parameters: DialectParameters,
    ) -> Result<Option<&'static str>, String> {
        DECLARATIVE_DIALECT.auto_activation_trigger(parameters)
    }

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        DECLARATIVE_DIALECT.required_structural_tokens(parameters)
    }

    fn stop_sequences(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        DECLARATIVE_DIALECT.stop_sequences(parameters)
    }

    fn incremental_parser_state(
        &self,
        _parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(GemmaToolParser::default()))
    }
}

impl FormatDialect for GemmaChannelDialect {
    fn supports_reasoning_parsing(&self, _parameters: DialectParameters) -> bool {
        true
    }

    fn generation_prompt_behavior(
        &self,
        _parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Ok(GenerationPromptBehavior::HonorRequest)
    }

    fn constraint_configuration(
        &self,
        _parameters: DialectParameters,
        _tools: &[Value],
        _tool_choice: ToolChoice,
        _parallel_tool_calls: ParallelToolCallPolicy,
        _resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        Err("Gemma channel semantics do not imply constrained tool generation".into())
    }

    fn auto_activation_trigger(
        &self,
        _parameters: DialectParameters,
    ) -> Result<Option<&'static str>, String> {
        Ok(None)
    }

    fn required_structural_tokens(
        &self,
        _parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Ok(&[CHANNEL_OPEN, CHANNEL_CLOSE])
    }

    fn stop_sequences(
        &self,
        _parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Ok(&[])
    }

    fn incremental_parser_state(
        &self,
        _parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(GemmaChannelParser::default()))
    }
}

#[derive(Debug, Default)]
enum GemmaChannelState {
    #[default]
    Visible,
    ChannelHeader(String),
    Reasoning,
}

#[derive(Debug, Default)]
struct GemmaChannelParser {
    state: GemmaChannelState,
}

#[derive(Debug, Default)]
enum GemmaToolState {
    #[default]
    Visible,
    ChannelHeader(String),
    Reasoning,
    ToolCall(String),
}

#[derive(Debug, Default)]
struct GemmaToolParser {
    state: GemmaToolState,
}

impl GemmaToolParser {
    fn finish_tool_call(
        &mut self,
        payload: String,
        sink: &mut SemanticEventSink,
    ) -> Result<(), String> {
        let mut parser = DECLARATIVE_DIALECT.incremental_parser_state(
            DialectParameters::Declarative(&GEMMA4_STRUCTURAL_TOOL_SPEC),
        )?;
        parser.push(&format!("{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"), sink)?;
        self.state = GemmaToolState::Visible;
        Ok(())
    }
}

impl ProtocolParser for GemmaToolParser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match &mut self.state {
            GemmaToolState::Visible => sink.text(text),
            GemmaToolState::Reasoning => sink.reasoning(text),
            GemmaToolState::ChannelHeader(header) => {
                header.push_str(text);
                let Some(newline) = header.find('\n') else {
                    if header.len() > 64 {
                        return Err("Gemma channel header exceeds 64 bytes".into());
                    }
                    return Ok(());
                };
                let body = header[newline + 1..].to_owned();
                if &header[..newline] != "thought" {
                    return Err(format!(
                        "unsupported Gemma channel {:?}; expected \"thought\"",
                        &header[..newline]
                    ));
                }
                self.state = GemmaToolState::Reasoning;
                sink.reasoning(body);
            }
            GemmaToolState::ToolCall(payload) => {
                if [
                    CHANNEL_OPEN,
                    CHANNEL_CLOSE,
                    TOOL_CALL_OPEN,
                    TOOL_CALL_CLOSE,
                    STRING_DELIMITER,
                    TOOL_RESPONSE_OPEN,
                    TURN_CLOSE,
                ]
                .iter()
                .any(|marker| text.contains(marker))
                {
                    return Err(
                        "Gemma tool payload contains a structural marker without its special-token identity"
                            .into(),
                    );
                }
                payload.push_str(text);
            }
        }
        Ok(())
    }

    fn structural(
        &mut self,
        _token_id: u32,
        spelling: &str,
        sink: &mut SemanticEventSink,
    ) -> Result<(), Self::Error> {
        match (&mut self.state, spelling) {
            (GemmaToolState::Visible, CHANNEL_OPEN) => {
                self.state = GemmaToolState::ChannelHeader(String::new());
                Ok(())
            }
            (GemmaToolState::Visible, TOOL_CALL_OPEN) => {
                self.state = GemmaToolState::ToolCall(String::new());
                Ok(())
            }
            (GemmaToolState::Reasoning, CHANNEL_CLOSE) => {
                self.state = GemmaToolState::Visible;
                Ok(())
            }
            (GemmaToolState::ToolCall(payload), STRING_DELIMITER) => {
                payload.push_str(STRING_DELIMITER);
                Ok(())
            }
            (GemmaToolState::ToolCall(_), TOOL_CALL_CLOSE) => {
                let GemmaToolState::ToolCall(payload) =
                    std::mem::replace(&mut self.state, GemmaToolState::Visible)
                else {
                    unreachable!()
                };
                self.finish_tool_call(payload, sink)
            }
            _ => Err(format!(
                "Gemma structural token {spelling:?} is invalid in parser state {:?}",
                self.state
            )),
        }
    }

    fn stop(&mut self, sequence: &str, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match self.state {
            GemmaToolState::Visible => Ok(()),
            _ => Err(format!(
                "Gemma stop token {sequence:?} occurred inside an incomplete construct"
            )),
        }
    }

    fn finish(&mut self, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match self.state {
            GemmaToolState::Visible => Ok(()),
            GemmaToolState::ChannelHeader(_) => {
                Err("generation ended inside a Gemma channel header".into())
            }
            GemmaToolState::Reasoning => {
                Err("generation ended before the Gemma reasoning channel closed".into())
            }
            GemmaToolState::ToolCall(_) => {
                Err("generation ended before the Gemma tool call closed".into())
            }
        }
    }
}

impl ProtocolParser for GemmaChannelParser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match &mut self.state {
            GemmaChannelState::Visible => sink.text(text),
            GemmaChannelState::Reasoning => sink.reasoning(text),
            GemmaChannelState::ChannelHeader(header) => {
                header.push_str(text);
                let Some(newline) = header.find('\n') else {
                    if header.len() > 64 {
                        return Err("Gemma channel header exceeds 64 bytes".into());
                    }
                    return Ok(());
                };
                let body = header[newline + 1..].to_owned();
                if &header[..newline] != "thought" {
                    return Err(format!(
                        "unsupported Gemma channel {:?}; expected \"thought\"",
                        &header[..newline]
                    ));
                }
                self.state = GemmaChannelState::Reasoning;
                sink.reasoning(body);
            }
        }
        Ok(())
    }

    fn structural(
        &mut self,
        _token_id: u32,
        spelling: &str,
        _sink: &mut SemanticEventSink,
    ) -> Result<(), Self::Error> {
        match (&self.state, spelling) {
            (GemmaChannelState::Visible, CHANNEL_OPEN) => {
                self.state = GemmaChannelState::ChannelHeader(String::new());
                Ok(())
            }
            (GemmaChannelState::Reasoning, CHANNEL_CLOSE) => {
                self.state = GemmaChannelState::Visible;
                Ok(())
            }
            _ => Err(format!(
                "Gemma structural token {spelling:?} is invalid in parser state {:?}",
                self.state
            )),
        }
    }

    fn stop(&mut self, sequence: &str, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match self.state {
            GemmaChannelState::Visible => Ok(()),
            _ => Err(format!(
                "Gemma stop token {sequence:?} occurred inside an incomplete channel"
            )),
        }
    }

    fn finish(&mut self, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        match self.state {
            GemmaChannelState::Visible => Ok(()),
            GemmaChannelState::ChannelHeader(_) => {
                Err("generation ended inside a Gemma channel header".into())
            }
            GemmaChannelState::Reasoning => {
                Err("generation ended before the Gemma reasoning channel closed".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::*;
    use crate::runtime::generation::streaming::{
        CommittedTokenPipeline, FinishReason, RawTokenDecoder, SemanticEvent, SemanticEventSink,
        TokenDecoderBackend, ToolRuntimeParser,
    };

    #[test]
    fn structural_channel_identity_controls_reasoning_state() {
        let mut parser = GemmaChannelParser::default();
        let mut sink = SemanticEventSink::default();
        parser.push("before <|channel>", &mut sink).unwrap();
        parser.structural(10, CHANNEL_OPEN, &mut sink).unwrap();
        parser.push("thought\nprivate", &mut sink).unwrap();
        parser.structural(11, CHANNEL_CLOSE, &mut sink).unwrap();
        parser.push("visible", &mut sink).unwrap();

        assert_eq!(
            sink.events(),
            [
                SemanticEvent::TextDelta("before <|channel>".into()),
                SemanticEvent::ReasoningDelta("private".into()),
                SemanticEvent::TextDelta("visible".into()),
            ]
        );
    }

    #[test]
    fn rejects_structural_tokens_outside_valid_states() {
        let mut parser = GemmaChannelParser::default();
        let mut sink = SemanticEventSink::default();
        assert!(parser
            .structural(11, CHANNEL_CLOSE, &mut sink)
            .unwrap_err()
            .contains("invalid"));
    }

    #[derive(Debug)]
    struct StructuralBackend;

    impl TokenDecoderBackend for StructuralBackend {
        type Error = Infallible;

        fn decode_token(
            &mut self,
            token_id: u32,
            _preserve_special: bool,
        ) -> Result<Vec<u8>, Self::Error> {
            Ok(match token_id {
                1 => b"literal <|channel>".to_vec(),
                2 => b"thought\nprivate".to_vec(),
                3 => b"visible".to_vec(),
                10 | 11 => Vec::new(),
                _ => unreachable!(),
            })
        }
    }

    #[test]
    fn committed_pipeline_distinguishes_ids_from_marker_like_text() {
        let parser = ToolRuntimeParser::new(
            Box::new(GemmaChannelParser::default()),
            std::iter::empty(),
            std::iter::empty(),
        );
        let decoder = RawTokenDecoder::with_structural_tokens(
            StructuralBackend,
            [
                (10, CHANNEL_OPEN.to_owned()),
                (11, CHANNEL_CLOSE.to_owned()),
            ],
        );
        let mut pipeline = CommittedTokenPipeline::new(decoder, parser);
        let mut events = Vec::new();
        for token in [1, 10, 2, 11, 3] {
            pipeline
                .push(token, &mut |event| events.push(event))
                .unwrap();
        }
        pipeline
            .finish(FinishReason::MaxTokens, &mut |event| events.push(event))
            .unwrap();

        assert_eq!(
            events,
            [
                SemanticEvent::TextDelta("literal <|channel>".into()),
                SemanticEvent::ReasoningDelta("private".into()),
                SemanticEvent::TextDelta("visible".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::MaxTokens
                },
            ]
        );
    }

    #[test]
    fn tool_parser_treats_marker_like_text_as_text_and_structural_stops_as_stops() {
        let mut parser = ToolRuntimeParser::new_with_structural_stops(
            Box::new(GemmaToolParser::default()),
            [TURN_CLOSE],
            std::iter::empty(),
            [TURN_CLOSE],
        );
        parser.push("literal <|channel> and <turn|>").unwrap();
        parser.push_structural(10, CHANNEL_OPEN).unwrap();
        parser.push("thought\nprivate").unwrap();
        parser.push_structural(11, CHANNEL_CLOSE).unwrap();
        parser.push("visible").unwrap();
        parser.push_structural(16, TURN_CLOSE).unwrap();

        assert_eq!(
            parser.events(),
            [
                SemanticEvent::TextDelta("literal <|channel> and <turn|>".into()),
                SemanticEvent::ReasoningDelta("private".into()),
                SemanticEvent::TextDelta("visible".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::StopSequence
                },
            ]
        );
    }
}
