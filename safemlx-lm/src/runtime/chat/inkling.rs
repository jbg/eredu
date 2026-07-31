//! Inkling's structurally framed reasoning and visible-text protocol.

use serde_json::Value;

use super::{
    dialect::{
        ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
    },
    ParallelToolCallPolicy, ToolChoice,
};
use crate::runtime::generation::streaming::{ProtocolParser, SemanticEventSink};

pub(crate) const MESSAGE_MODEL: &str = "<|message_model|>";
pub(crate) const CONTENT_TEXT: &str = "<|content_text|>";
pub(crate) const CONTENT_THINKING: &str = "<|content_thinking|>";
pub(crate) const END_MESSAGE: &str = "<|end_message|>";
pub(crate) const END_SAMPLING: &str = "<|content_model_end_sampling|>";

const STRUCTURAL_TOKENS: &[&str] = &[
    MESSAGE_MODEL,
    CONTENT_TEXT,
    CONTENT_THINKING,
    END_MESSAGE,
    END_SAMPLING,
];
const STOPS: &[&str] = &[END_SAMPLING];

#[derive(Debug)]
pub(crate) struct InklingMessageDialect;

pub(crate) static INKLING_MESSAGE_DIALECT: InklingMessageDialect = InklingMessageDialect;

#[derive(Debug)]
pub(crate) struct InklingMessageParameters;

pub(crate) static INKLING_MESSAGE_PARAMETERS: InklingMessageParameters = InklingMessageParameters;

pub(crate) fn parameters() -> DialectParameters {
    DialectParameters::Custom(&INKLING_MESSAGE_PARAMETERS)
}

impl InklingMessageDialect {
    fn parameters(
        parameters: DialectParameters,
    ) -> Result<&'static InklingMessageParameters, String> {
        parameters.custom::<InklingMessageParameters>()
    }
}

impl FormatDialect for InklingMessageDialect {
    fn supports_reasoning_parsing(&self, parameters: DialectParameters) -> bool {
        Self::parameters(parameters).is_ok()
    }

    fn generation_prompt_behavior(
        &self,
        parameters: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Self::parameters(parameters)?;
        Ok(GenerationPromptBehavior::HonorRequest)
    }

    fn reasoning_template_kwarg(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static str, String> {
        Self::parameters(parameters)?;
        Ok("reasoning_effort")
    }

    fn constraint_configuration(
        &self,
        parameters: DialectParameters,
        _tools: &[Value],
        _tool_choice: ToolChoice,
        _parallel_tool_calls: ParallelToolCallPolicy,
        _resolved_structural_token_ids: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        Self::parameters(parameters)?;
        Err("Inkling message semantics do not imply constrained tool generation".into())
    }

    fn auto_activation_trigger(
        &self,
        parameters: DialectParameters,
    ) -> Result<Option<&'static str>, String> {
        Self::parameters(parameters)?;
        Ok(None)
    }

    fn required_structural_tokens(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::parameters(parameters)?;
        Ok(STRUCTURAL_TOKENS)
    }

    fn stop_sequences(
        &self,
        parameters: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Self::parameters(parameters)?;
        Ok(STOPS)
    }

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Self::parameters(parameters)?;
        Ok(Box::new(InklingMessageParser::default()))
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    #[default]
    Channel,
    Reasoning,
    ModelAfterReasoning,
    Text,
    AfterText,
}

#[derive(Debug, Default)]
struct InklingMessageParser {
    state: ParserState,
}

impl InklingMessageParser {
    fn unexpected(&self, spelling: &str) -> String {
        format!(
            "unexpected Inkling structural token {spelling:?} while parsing {:?}",
            self.state
        )
    }
}

impl ProtocolParser for InklingMessageParser {
    type Error = String;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        if text.is_empty() {
            return Ok(());
        }
        match self.state {
            ParserState::Reasoning => sink.reasoning(text),
            ParserState::Text => sink.text(text),
            _ => {
                return Err(format!(
                    "unexpected ordinary Inkling output while parsing {:?}",
                    self.state
                ));
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
        self.state = match (self.state, spelling) {
            (ParserState::Channel, CONTENT_THINKING) => ParserState::Reasoning,
            (ParserState::Channel, CONTENT_TEXT) => ParserState::Text,
            (ParserState::Reasoning, END_MESSAGE) => ParserState::ModelAfterReasoning,
            (ParserState::ModelAfterReasoning, MESSAGE_MODEL) => ParserState::Channel,
            (ParserState::Text, END_MESSAGE) => ParserState::AfterText,
            _ => return Err(self.unexpected(spelling)),
        };
        Ok(())
    }

    fn stop(&mut self, sequence: &str, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        if sequence != END_SAMPLING {
            return Err(format!("unexpected Inkling stop sequence {sequence:?}"));
        }
        if self.state != ParserState::AfterText {
            return Err(format!(
                "Inkling sampling ended while parsing {:?}, expected a completed text frame",
                self.state
            ));
        }
        Ok(())
    }

    fn finish(&mut self, _sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        // A caller token limit may stop inside either channel. Content already
        // emitted remains correctly classified without manufacturing closure.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::generation::streaming::{FinishReason, SemanticEvent, ToolRuntimeParser};

    fn new_parser() -> ToolRuntimeParser {
        ToolRuntimeParser::new_with_structural_stops(
            Box::new(InklingMessageParser::default()),
            STOPS.iter().copied(),
            std::iter::empty(),
            STOPS.iter().copied(),
        )
    }

    #[test]
    fn parser_separates_reasoning_and_visible_text() {
        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        parser.push("private thought").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        parser.push_structural(3, MESSAGE_MODEL).unwrap();
        parser.push_structural(4, CONTENT_TEXT).unwrap();
        parser.push("visible answer").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        assert!(parser.push_structural(5, END_SAMPLING).unwrap());
        assert_eq!(
            parser.events(),
            [
                SemanticEvent::ReasoningDelta("private thought".into()),
                SemanticEvent::TextDelta("visible answer".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::StopSequence,
                },
            ]
        );
    }

    #[test]
    fn parser_preserves_partial_reasoning_at_token_limit() {
        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        parser.push("partial thought").unwrap();
        parser.finish(FinishReason::MaxTokens).unwrap();
        assert_eq!(
            parser.events(),
            [
                SemanticEvent::ReasoningDelta("partial thought".into()),
                SemanticEvent::Finished {
                    reason: FinishReason::MaxTokens,
                },
            ]
        );
    }

    #[test]
    fn ordinary_marker_text_cannot_change_structural_state() {
        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        parser
            .push("literal <|end_message|> remains reasoning")
            .unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        parser.push_structural(3, MESSAGE_MODEL).unwrap();
        parser.push_structural(4, CONTENT_TEXT).unwrap();
        parser.push("answer").unwrap();
        parser.push_structural(2, END_MESSAGE).unwrap();
        parser.push_structural(5, END_SAMPLING).unwrap();

        assert!(parser.events().contains(&SemanticEvent::ReasoningDelta(
            "literal <|end_message|> remains reasoning".into()
        )));
        assert!(parser
            .events()
            .contains(&SemanticEvent::TextDelta("answer".into())));
    }

    #[test]
    fn malformed_channel_transitions_fail_closed() {
        let mut parser = new_parser();
        assert!(parser.push("unframed text").is_err());

        let mut parser = new_parser();
        parser.push_structural(1, CONTENT_THINKING).unwrap();
        assert!(parser.push_structural(4, CONTENT_TEXT).is_err());

        let mut parser = new_parser();
        parser.push_structural(4, CONTENT_TEXT).unwrap();
        parser.push("incomplete answer").unwrap();
        assert!(parser.push_structural(5, END_SAMPLING).is_err());
    }
}
