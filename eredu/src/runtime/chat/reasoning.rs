//! Delimited reasoning without tool parsing or vocabulary constraints.

use super::dialect::{
    ConstraintConfiguration, DialectParameters, FormatDialect, GenerationPromptBehavior,
};
use super::{ParallelToolCallPolicy, ToolChoice};
use crate::runtime::generation::streaming::{ProtocolParser, SemanticEventSink};
use eredu_core::SemanticEvent;
use serde_json::Value;

#[derive(Debug, Clone, Copy)]
pub(crate) enum InitialChannel {
    Generated,
    Reasoning,
    Text,
}

#[derive(Debug)]
pub(crate) struct ReasoningFormat {
    pub(crate) opening: &'static str,
    pub(crate) closing: &'static str,
    pub(crate) initial: InitialChannel,
}

pub(crate) static THINK_FORMATS: [ReasoningFormat; 3] = [
    ReasoningFormat {
        opening: "<think>",
        closing: "</think>",
        initial: InitialChannel::Generated,
    },
    ReasoningFormat {
        opening: "<think>",
        closing: "</think>",
        initial: InitialChannel::Reasoning,
    },
    ReasoningFormat {
        opening: "<think>",
        closing: "</think>",
        initial: InitialChannel::Text,
    },
];

#[derive(Debug)]
pub(crate) struct ReasoningDialect;
pub(crate) static REASONING_DIALECT: ReasoningDialect = ReasoningDialect;

impl FormatDialect for ReasoningDialect {
    fn generation_prompt_behavior(
        &self,
        _: DialectParameters,
    ) -> Result<GenerationPromptBehavior, String> {
        Ok(GenerationPromptBehavior::HonorRequest)
    }

    fn supports_reasoning_parsing(&self, _: DialectParameters) -> bool {
        true
    }

    fn constrains_semantic_output(&self) -> bool {
        false
    }

    fn constraint_configuration(
        &self,
        _: DialectParameters,
        _: &[Value],
        _: ToolChoice,
        _: ParallelToolCallPolicy,
        _: &[u32],
    ) -> Result<ConstraintConfiguration, String> {
        Err("delimited reasoning does not provide native tool constraints".into())
    }

    fn auto_activation_trigger(
        &self,
        _: DialectParameters,
    ) -> Result<Option<&'static str>, String> {
        Ok(None)
    }
    fn required_structural_tokens(
        &self,
        _: DialectParameters,
    ) -> Result<&'static [&'static str], String> {
        Ok(&[])
    }
    fn stop_sequences(&self, _: DialectParameters) -> Result<&'static [&'static str], String> {
        Ok(&[])
    }

    fn incremental_parser_state(
        &self,
        parameters: DialectParameters,
    ) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        let format = parameters.custom::<ReasoningFormat>()?;
        Ok(Box::new(ReasoningParser {
            format,
            channel: format.initial,
            pending: String::new(),
        }))
    }
}

#[derive(Clone)]
struct ReasoningParser {
    format: &'static ReasoningFormat,
    channel: InitialChannel,
    pending: String,
}

impl ReasoningParser {
    fn emit(&self, text: String, sink: &mut SemanticEventSink) {
        match self.channel {
            InitialChannel::Reasoning => sink.reasoning(text),
            InitialChannel::Generated | InitialChannel::Text => sink.text(text),
        }
    }
}

impl ProtocolParser for ReasoningParser {
    type Error = String;

    fn snapshot_storage_bytes(&self) -> Option<u64> {
        (std::mem::size_of::<Self>() as u64).checked_add(self.pending.len() as u64)
    }

    fn continuation_storage_bytes(&self, input: u64) -> Option<u64> {
        // At most one delimiter's lookbehind survives a push; include transient input.
        self.snapshot_storage_bytes()?.checked_add(input)
    }

    fn continuation_event_storage_bytes(&self, input: u64, tokens: u64) -> Option<u64> {
        // One token may flush lookbehind, cross a channel, and emit visible text.
        tokens
            .checked_mul(3)?
            .checked_add(5)?
            .checked_mul(std::mem::size_of::<SemanticEvent>() as u64)?
            .checked_add(input)?
            .checked_add(self.pending.len() as u64)?
            .checked_mul(2)
    }

    fn fork_box(&self) -> Result<Box<dyn ProtocolParser<Error = String>>, String> {
        Ok(Box::new(self.clone()))
    }

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), String> {
        self.pending.push_str(text);
        if matches!(self.channel, InitialChannel::Generated) {
            if self.pending.starts_with(self.format.opening) {
                self.pending.drain(..self.format.opening.len());
                self.channel = InitialChannel::Reasoning;
            } else if self.format.opening.starts_with(&self.pending) {
                return Ok(());
            } else {
                self.channel = InitialChannel::Text;
            }
        }
        if matches!(self.channel, InitialChannel::Reasoning) {
            if let Some(end) = self.pending.find(self.format.closing) {
                sink.reasoning(self.pending[..end].to_owned());
                self.pending.drain(..end + self.format.closing.len());
                self.channel = InitialChannel::Text;
            } else {
                let retained = (1..self.format.closing.len().min(self.pending.len() + 1))
                    .rev()
                    .find(|&len| {
                        self.pending.is_char_boundary(self.pending.len() - len)
                            && self
                                .format
                                .closing
                                .starts_with(&self.pending[self.pending.len() - len..])
                    })
                    .unwrap_or(0);
                let ready = self.pending.len() - retained;
                sink.reasoning(self.pending.drain(..ready).collect::<String>());
                return Ok(());
            }
        }
        sink.text(std::mem::take(&mut self.pending));
        Ok(())
    }

    fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), String> {
        let pending = std::mem::take(&mut self.pending);
        self.emit(pending, sink);
        Ok(())
    }
}
