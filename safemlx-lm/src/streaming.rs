//! Protocol-independent semantic streaming.
//!
//! This module defines the events consumed by API adapters. The parsing
//! machinery deliberately knows nothing about a model family: format profiles
//! supply parsers and ordinary generation supplies committed tokenizer ids.

#![allow(dead_code)]

use std::collections::BTreeSet;
use std::fmt;
use std::num::NonZeroUsize;

/// Why a semantic response stream finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A checkpoint EOS token was generated.
    Eos,
    /// A profile or caller stop sequence matched.
    StopSequence,
    /// A generation grammar reached its accepting state.
    GrammarComplete,
    /// The caller's generation token limit was reached.
    MaxTokens,
}

/// A protocol-neutral incremental response event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticEvent {
    /// An incremental reasoning-text fragment.
    ReasoningDelta(String),
    /// An incremental visible-text fragment.
    TextDelta(String),
    /// The beginning of one canonical tool call.
    ToolCallStart {
        /// Zero-based tool-call position in the assistant turn.
        index: usize,
        /// Stable tool-call identifier.
        id: String,
        /// Tool name selected by the model.
        name: String,
    },
    /// An incremental JSON argument fragment for a tool call.
    ToolArgumentsDelta {
        /// Zero-based tool-call position in the assistant turn.
        index: usize,
        /// A fragment of the tool call's JSON arguments.
        json_fragment: String,
    },
    /// The current tool call ended.
    ToolCallEnd,
    /// The semantic response stream ended.
    Finished {
        /// The condition that ended the stream.
        reason: FinishReason,
    },
}

/// Canonical parser-owned representation of a tool call being assembled.
#[derive(Debug, Clone, PartialEq, Eq)]
struct InProgressToolCall {
    index: usize,
    id: String,
    name: String,
    arguments: String,
}

impl InProgressToolCall {
    fn new(index: usize, id: String, name: String) -> Self {
        Self {
            index,
            id,
            name,
            arguments: String::new(),
        }
    }

    fn start_event(&self) -> SemanticEvent {
        SemanticEvent::ToolCallStart {
            index: self.index,
            id: self.id.clone(),
            name: self.name.clone(),
        }
    }

    fn append_arguments(&mut self, fragment: &str) -> SemanticEvent {
        self.arguments.push_str(fragment);
        SemanticEvent::ToolArgumentsDelta {
            index: self.index,
            json_fragment: fragment.to_owned(),
        }
    }
}

/// Token decoder used before UTF-8 assembly and protocol parsing.
///
/// Backends decide how ordinary tokenizer pieces become raw bytes. The
/// `preserve_special` flag is true only for profile-designated structural token
/// IDs; other special tokens can therefore remain skipped.
pub(crate) trait TokenDecoderBackend {
    type Error;

    fn decode_token(
        &mut self,
        token_id: u32,
        preserve_special: bool,
    ) -> Result<Vec<u8>, Self::Error>;

    fn finish(&mut self) -> Result<Vec<u8>, Self::Error> {
        Ok(Vec::new())
    }
}

/// A raw decoded token with its structural identity retained.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RawToken {
    token_id: u32,
    bytes: Vec<u8>,
    structural: bool,
}

/// Decodes raw token pieces while retaining designated structural specials.
pub(crate) struct RawTokenDecoder<D> {
    backend: D,
    structural_token_ids: BTreeSet<u32>,
}

impl<D> RawTokenDecoder<D>
where
    D: TokenDecoderBackend,
{
    pub(crate) fn new(backend: D, structural_token_ids: impl IntoIterator<Item = u32>) -> Self {
        Self {
            backend,
            structural_token_ids: structural_token_ids.into_iter().collect(),
        }
    }

    fn push(&mut self, token_id: u32) -> Result<RawToken, D::Error> {
        let structural = self.structural_token_ids.contains(&token_id);
        let bytes = self.backend.decode_token(token_id, structural)?;
        Ok(RawToken {
            token_id,
            bytes,
            structural,
        })
    }

    fn finish(&mut self) -> Result<Vec<u8>, D::Error> {
        self.backend.finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Utf8BufferError {
    Invalid { valid_up_to: usize },
    Incomplete,
}

impl fmt::Display for Utf8BufferError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid { valid_up_to } => {
                write!(formatter, "invalid UTF-8 after byte {valid_up_to}")
            }
            Self::Incomplete => formatter.write_str("incomplete UTF-8 at end of stream"),
        }
    }
}

/// Buffers byte-fallback pieces until a complete UTF-8 prefix is available.
#[derive(Debug, Default)]
struct Utf8Buffer {
    pending: Vec<u8>,
}

impl Utf8Buffer {
    fn push(&mut self, bytes: &[u8]) -> Result<String, Utf8BufferError> {
        self.pending.extend_from_slice(bytes);

        match std::str::from_utf8(&self.pending) {
            Ok(text) => {
                let text = text.to_owned();
                self.pending.clear();
                Ok(text)
            }
            Err(error) if error.error_len().is_none() => {
                let valid_up_to = error.valid_up_to();
                let valid = String::from_utf8(self.pending[..valid_up_to].to_vec())
                    .expect("from_utf8 reported this prefix as valid");
                self.pending.drain(..valid_up_to);
                Ok(valid)
            }
            Err(error) => Err(Utf8BufferError::Invalid {
                valid_up_to: error.valid_up_to(),
            }),
        }
    }

    fn finish(self) -> Result<(), Utf8BufferError> {
        if self.pending.is_empty() {
            Ok(())
        } else {
            Err(Utf8BufferError::Incomplete)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct StopMatch {
    visible: String,
    matched: bool,
}

/// Incremental multi-pattern stop matcher with overlap-safe lookbehind.
///
/// Bytes that might still become a stop remain private. Once a complete stop
/// matches, neither it nor any following bytes can become visible.
#[derive(Debug)]
struct StopMatcher {
    stops: Vec<Vec<u8>>,
    pending: Vec<u8>,
    matched: bool,
}

impl StopMatcher {
    fn new<'a, 'b>(
        profile_stops: impl IntoIterator<Item = &'a str>,
        caller_stops: impl IntoIterator<Item = &'b str>,
    ) -> Self {
        let mut stops = Vec::new();
        for stop in profile_stops {
            if !stop.is_empty() && !stops.iter().any(|existing| existing == stop.as_bytes()) {
                stops.push(stop.as_bytes().to_vec());
            }
        }
        for stop in caller_stops {
            if !stop.is_empty() && !stops.iter().any(|existing| existing == stop.as_bytes()) {
                stops.push(stop.as_bytes().to_vec());
            }
        }
        Self {
            stops,
            pending: Vec::new(),
            matched: false,
        }
    }

    fn push(&mut self, text: &str) -> StopMatch {
        if self.matched {
            return StopMatch {
                visible: String::new(),
                matched: true,
            };
        }

        self.pending.extend_from_slice(text.as_bytes());
        if let Some(position) = self.first_match_position() {
            let visible = String::from_utf8(self.pending[..position].to_vec())
                .expect("stop matching preserves UTF-8 boundaries");
            self.pending.clear();
            self.matched = true;
            return StopMatch {
                visible,
                matched: true,
            };
        }

        let lookbehind = self.longest_possible_stop_prefix();
        let visible_len = self.pending.len() - lookbehind;
        let visible = String::from_utf8(self.pending[..visible_len].to_vec())
            .expect("stop matching preserves UTF-8 boundaries");
        self.pending.drain(..visible_len);
        StopMatch {
            visible,
            matched: false,
        }
    }

    fn finish(&mut self) -> String {
        if self.matched {
            return String::new();
        }
        String::from_utf8(std::mem::take(&mut self.pending))
            .expect("stop matching preserves UTF-8 boundaries")
    }

    fn first_match_position(&self) -> Option<usize> {
        (0..self.pending.len()).find(|&position| {
            self.stops
                .iter()
                .any(|stop| self.pending[position..].starts_with(stop))
        })
    }

    fn longest_possible_stop_prefix(&self) -> usize {
        let max = self
            .stops
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or_default()
            .min(self.pending.len());
        (1..=max)
            .rev()
            .find(|&length| {
                self.stops
                    .iter()
                    .any(|stop| stop.starts_with(&self.pending[self.pending.len() - length..]))
            })
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PatternKind {
    Trigger,
    Tag,
    Delimiter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PatternPiece {
    Text(String),
    Match { index: usize, kind: PatternKind },
}

/// Buffers suffixes that could be partial triggers, tags, or delimiters.
#[derive(Debug)]
pub(crate) struct PartialPatternBuffer {
    patterns: Vec<(PatternKind, Vec<u8>)>,
    pending: Vec<u8>,
}

impl PartialPatternBuffer {
    pub(crate) fn new(patterns: impl IntoIterator<Item = (PatternKind, &'static str)>) -> Self {
        Self {
            patterns: patterns
                .into_iter()
                .filter(|(_, pattern)| !pattern.is_empty())
                .map(|(kind, pattern)| (kind, pattern.as_bytes().to_vec()))
                .collect(),
            pending: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> Vec<PatternPiece> {
        self.pending.extend_from_slice(text.as_bytes());
        let mut pieces = Vec::new();

        loop {
            let Some((position, index)) = self.first_match() else {
                let lookbehind = self.longest_possible_prefix();
                let visible_len = self.pending.len() - lookbehind;
                if visible_len > 0 {
                    pieces.push(PatternPiece::Text(
                        String::from_utf8(self.pending[..visible_len].to_vec())
                            .expect("pattern matching preserves UTF-8 boundaries"),
                    ));
                    self.pending.drain(..visible_len);
                }
                break;
            };

            if position > 0 {
                pieces.push(PatternPiece::Text(
                    String::from_utf8(self.pending[..position].to_vec())
                        .expect("pattern matching preserves UTF-8 boundaries"),
                ));
            }
            let pattern_len = self.patterns[index].1.len();
            self.pending.drain(..position + pattern_len);
            pieces.push(PatternPiece::Match {
                index,
                kind: self.patterns[index].0,
            });
        }
        pieces
    }

    pub(crate) fn finish(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            None
        } else {
            Some(
                String::from_utf8(std::mem::take(&mut self.pending))
                    .expect("pattern matching preserves UTF-8 boundaries"),
            )
        }
    }

    fn first_match(&self) -> Option<(usize, usize)> {
        for position in 0..self.pending.len() {
            if let Some(index) = self
                .patterns
                .iter()
                .position(|(_, pattern)| self.pending[position..].starts_with(pattern))
            {
                return Some((position, index));
            }
        }
        None
    }

    fn longest_possible_prefix(&self) -> usize {
        let max = self
            .patterns
            .iter()
            .map(|(_, pattern)| pattern.len())
            .max()
            .unwrap_or_default()
            .min(self.pending.len());
        (1..=max)
            .rev()
            .find(|&length| {
                self.patterns.iter().any(|(_, pattern)| {
                    pattern.starts_with(&self.pending[self.pending.len() - length..])
                })
            })
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JsonFragmentError {
    MissingContainer,
    TrailingData,
}

/// Incrementally identifies one complete JSON object or array.
#[derive(Debug, Default)]
pub(crate) struct JsonFragmentBuffer {
    fragment: String,
    depth: usize,
    in_string: bool,
    escaped: bool,
    started: bool,
    complete: bool,
}

impl JsonFragmentBuffer {
    /// Appends input and returns `(consumed_bytes, complete)`.
    pub(crate) fn push(&mut self, input: &str) -> Result<(usize, bool), JsonFragmentError> {
        if self.complete {
            return if input.is_empty() {
                Ok((0, true))
            } else {
                Err(JsonFragmentError::TrailingData)
            };
        }

        let mut consumed = 0;
        for character in input.chars() {
            let character_len = character.len_utf8();
            if !self.started {
                if character.is_whitespace() {
                    self.fragment.push(character);
                    consumed += character_len;
                    continue;
                }
                if !matches!(character, '{' | '[') {
                    return Err(JsonFragmentError::MissingContainer);
                }
                self.started = true;
                self.depth = 1;
                self.fragment.push(character);
                consumed += character_len;
                continue;
            }

            self.fragment.push(character);
            consumed += character_len;
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if character == '\\' {
                    self.escaped = true;
                } else if character == '"' {
                    self.in_string = false;
                }
                continue;
            }

            match character {
                '"' => self.in_string = true,
                '{' | '[' => self.depth += 1,
                '}' | ']' => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.complete = true;
                        break;
                    }
                }
                _ => {}
            }
        }

        Ok((consumed, self.complete))
    }

    pub(crate) fn fragment(&self) -> &str {
        &self.fragment
    }

    fn is_complete(&self) -> bool {
        self.complete
    }
}

/// Sink offered to protocol parsers so semantic state and emitted events stay
/// synchronized.
#[derive(Debug, Default)]
pub(crate) struct SemanticEventSink {
    events: Vec<SemanticEvent>,
    active_tool_call: Option<InProgressToolCall>,
    next_tool_index: usize,
}

impl SemanticEventSink {
    pub(crate) fn events(&self) -> &[SemanticEvent] {
        &self.events
    }

    pub(crate) fn reasoning(&mut self, delta: impl Into<String>) {
        let delta = delta.into();
        if !delta.is_empty() {
            self.events.push(SemanticEvent::ReasoningDelta(delta));
        }
    }

    pub(crate) fn text(&mut self, delta: impl Into<String>) {
        let delta = delta.into();
        if !delta.is_empty() {
            self.events.push(SemanticEvent::TextDelta(delta));
        }
    }

    pub(crate) fn next_tool_index(&self) -> usize {
        self.next_tool_index
    }

    pub(crate) fn start_tool_call(&mut self, id: String, name: String) {
        debug_assert!(self.active_tool_call.is_none());
        let call = InProgressToolCall::new(self.next_tool_index, id, name);
        self.next_tool_index += 1;
        self.events.push(call.start_event());
        self.active_tool_call = Some(call);
    }

    pub(crate) fn tool_arguments(&mut self, fragment: &str) {
        if fragment.is_empty() {
            return;
        }
        let event = self
            .active_tool_call
            .as_mut()
            .expect("arguments require an active tool call")
            .append_arguments(fragment);
        self.events.push(event);
    }

    pub(crate) fn end_tool_call(&mut self) {
        debug_assert!(self.active_tool_call.is_some());
        self.active_tool_call = None;
        self.events.push(SemanticEvent::ToolCallEnd);
    }

    fn finish(&mut self, reason: FinishReason) {
        self.events.push(SemanticEvent::Finished { reason });
    }
}

/// Incremental parser contract implemented by an exact format profile.
pub(crate) trait ProtocolParser: Send {
    type Error;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error>;

    fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), Self::Error>;
}

impl<P> ProtocolParser for Box<P>
where
    P: ProtocolParser + ?Sized,
{
    type Error = P::Error;

    fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        (**self).push(text, sink)
    }

    fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
        (**self).finish(sink)
    }
}

/// Applies stop matching before any bytes can reach a protocol parser.
struct SemanticStream<P> {
    parser: P,
    stops: StopMatcher,
    sink: SemanticEventSink,
    finished: bool,
}

impl<P> SemanticStream<P>
where
    P: ProtocolParser,
{
    fn new<'a, 'b>(
        parser: P,
        profile_stops: impl IntoIterator<Item = &'a str>,
        caller_stops: impl IntoIterator<Item = &'b str>,
    ) -> Self {
        Self {
            parser,
            stops: StopMatcher::new(profile_stops, caller_stops),
            sink: SemanticEventSink::default(),
            finished: false,
        }
    }

    /// Returns true when a stop matched and the caller should halt decoding.
    fn push(&mut self, text: &str) -> Result<bool, P::Error> {
        if self.finished {
            return Ok(true);
        }

        let stop_match = self.stops.push(text);
        self.parser.push(&stop_match.visible, &mut self.sink)?;
        if stop_match.matched {
            self.parser.finish(&mut self.sink)?;
            self.sink.finish(FinishReason::StopSequence);
            self.finished = true;
        }
        Ok(stop_match.matched)
    }

    fn finish(&mut self, reason: FinishReason) -> Result<(), P::Error> {
        if self.finished {
            return Ok(());
        }
        let visible = self.stops.finish();
        self.parser.push(&visible, &mut self.sink)?;
        self.parser.finish(&mut self.sink)?;
        self.sink.finish(reason);
        self.finished = true;
        Ok(())
    }

    fn events(&self) -> &[SemanticEvent] {
        &self.sink.events
    }
}

/// Opaque, independent protocol parsing state for one prepared tool generation.
///
/// Instances are created by [`crate::chat::ToolRuntimePlan::create_parser`].
/// Each instance owns its parser, semantic event sink, and profile stop matcher.
pub struct ToolRuntimeParser {
    stream: SemanticStream<Box<dyn ProtocolParser<Error = String>>>,
}

impl fmt::Debug for ToolRuntimeParser {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ToolRuntimeParser")
            .field("finished", &self.stream.finished)
            .field("event_count", &self.stream.sink.events.len())
            .finish_non_exhaustive()
    }
}

impl ToolRuntimeParser {
    pub(crate) fn new<'a, 'b>(
        parser: Box<dyn ProtocolParser<Error = String>>,
        profile_stops: impl IntoIterator<Item = &'a str>,
        caller_stops: impl IntoIterator<Item = &'b str>,
    ) -> Self {
        Self {
            stream: SemanticStream::new(parser, profile_stops, caller_stops),
        }
    }

    /// Pushes decoded text and returns whether a profile stop sequence matched.
    pub fn push(&mut self, text: &str) -> Result<bool, String> {
        self.stream.push(text)
    }

    /// Finishes parsing with the generation's terminal reason.
    ///
    /// This is a no-op if a profile stop already finished the parser.
    pub fn finish(&mut self, reason: FinishReason) -> Result<(), String> {
        self.stream.finish(reason)
    }

    /// Returns all semantic events emitted by this parser instance.
    pub fn events(&self) -> &[SemanticEvent] {
        self.stream.events()
    }

    /// Removes and returns events emitted since the previous drain.
    ///
    /// Ordinary generation drains this queue after every committed token, so
    /// callers observe deltas while decoding is still in progress rather than
    /// after the complete response has been retained.
    pub fn take_events(&mut self) -> Vec<SemanticEvent> {
        std::mem::take(&mut self.stream.sink.events)
    }

    /// Returns whether this parser has reached a terminal condition.
    pub fn is_finished(&self) -> bool {
        self.stream.finished
    }
}

/// Owns decoding, UTF-8 assembly, stop matching, protocol parsing, and event
/// delivery for committed ordinary-generation tokens.
pub(crate) struct CommittedTokenPipeline<D> {
    decoder: RawTokenDecoder<D>,
    utf8: Utf8Buffer,
    parser: ToolRuntimeParser,
}

impl<D> CommittedTokenPipeline<D>
where
    D: TokenDecoderBackend,
    D::Error: fmt::Display,
{
    pub(crate) fn new(decoder: RawTokenDecoder<D>, parser: ToolRuntimeParser) -> Self {
        Self {
            decoder,
            utf8: Utf8Buffer::default(),
            parser,
        }
    }

    /// Commits one tokenizer id and reports whether a decoded stop matched.
    pub(crate) fn push(
        &mut self,
        token_id: u32,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<bool, String> {
        let raw = self
            .decoder
            .push(token_id)
            .map_err(|error| format!("token decoding failed: {error}"))?;
        let text = self
            .utf8
            .push(&raw.bytes)
            .map_err(|error| format!("UTF-8 assembly failed: {error}"))?;
        let matched = self.parser.push(&text)?;
        self.drain_into(emit);
        Ok(matched)
    }

    /// Finalizes parsing exactly once and immediately delivers buffered events.
    pub(crate) fn finish(
        &mut self,
        reason: FinishReason,
        emit: &mut impl FnMut(SemanticEvent),
    ) -> Result<(), String> {
        let trailing = self
            .decoder
            .finish()
            .map_err(|error| format!("token decoding failed: {error}"))?;
        let text = self
            .utf8
            .push(&trailing)
            .map_err(|error| format!("UTF-8 assembly failed: {error}"))?;
        self.parser.push(&text)?;
        std::mem::take(&mut self.utf8)
            .finish()
            .map_err(|error| format!("UTF-8 assembly failed: {error}"))?;
        self.parser.finish(reason)?;
        self.drain_into(emit);
        Ok(())
    }

    fn drain_into(&mut self, emit: &mut impl FnMut(SemanticEvent)) {
        for event in self.parser.take_events() {
            emit(event);
        }
    }
}

/// Architecture-independent source of sampled, grammar-committed token ids.
pub(crate) trait CommittedTokenSource {
    type Error: fmt::Display;

    fn next_token(&mut self) -> Result<Option<u32>, Self::Error>;

    fn grammar_is_complete(&mut self) -> Result<bool, Self::Error>;
}

/// Drives the production committed-token boundary without owning a second
/// model loop. The source remains the architecture-dispatched iterator.
pub(crate) fn drive_committed_generation<D, T>(
    source: &mut T,
    pipeline: &mut CommittedTokenPipeline<D>,
    eos_token_ids: &[u32],
    max_tokens: NonZeroUsize,
    emit: &mut impl FnMut(SemanticEvent),
) -> Result<(Vec<u32>, FinishReason), String>
where
    D: TokenDecoderBackend,
    D::Error: fmt::Display,
    T: CommittedTokenSource,
{
    let max_tokens = max_tokens.get();
    let mut token_ids = Vec::with_capacity(max_tokens);
    for index in 0..max_tokens {
        let token_id = source
            .next_token()
            .map_err(|error| format!("model generation failed: {error}"))?
            .ok_or_else(|| "architecture generation ended without a terminal token".to_owned())?;
        token_ids.push(token_id);

        let stop_matched = pipeline.push(token_id, emit)?;
        let grammar_complete = if stop_matched {
            false
        } else {
            source
                .grammar_is_complete()
                .map_err(|error| format!("grammar completion check failed: {error}"))?
        };
        let reason = stop_matched
            .then_some(FinishReason::StopSequence)
            .or_else(|| grammar_complete.then_some(FinishReason::GrammarComplete))
            .or_else(|| {
                eos_token_ids
                    .contains(&token_id)
                    .then_some(FinishReason::Eos)
            })
            .or_else(|| (index + 1 == max_tokens).then_some(FinishReason::MaxTokens));

        if let Some(reason) = reason {
            pipeline.finish(reason, emit)?;
            return Ok((token_ids, reason));
        }
    }

    unreachable!("a non-zero max token limit always terminates the generation loop")
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::VecDeque, convert::Infallible, num::NonZeroUsize, rc::Rc};

    use super::{
        drive_committed_generation, CommittedTokenPipeline, CommittedTokenSource, FinishReason,
        JsonFragmentBuffer, PartialPatternBuffer, PatternKind, PatternPiece, ProtocolParser,
        RawTokenDecoder, SemanticEvent, SemanticEventSink, SemanticStream, StopMatcher,
        TokenDecoderBackend, ToolRuntimeParser, Utf8Buffer, Utf8BufferError,
    };

    const REASONING_START: &str = "<r>";
    const REASONING_END: &str = "</r>";
    const TOOL_START: &str = "<call:";
    const TOOL_END: &str = "</call>";

    #[derive(Debug)]
    enum SyntheticState {
        Text(PartialPatternBuffer),
        Reasoning(PartialPatternBuffer),
        ToolHeader(String),
        ToolArguments(JsonFragmentBuffer),
        ToolEnd(PartialPatternBuffer),
    }

    #[derive(Debug)]
    struct SyntheticParser {
        state: SyntheticState,
    }

    impl Default for SyntheticParser {
        fn default() -> Self {
            Self {
                state: SyntheticState::Text(text_patterns()),
            }
        }
    }

    fn text_patterns() -> PartialPatternBuffer {
        PartialPatternBuffer::new([
            (PatternKind::Tag, REASONING_START),
            (PatternKind::Trigger, TOOL_START),
        ])
    }

    fn reasoning_patterns() -> PartialPatternBuffer {
        PartialPatternBuffer::new([(PatternKind::Tag, REASONING_END)])
    }

    fn tool_end_patterns() -> PartialPatternBuffer {
        PartialPatternBuffer::new([(PatternKind::Delimiter, TOOL_END)])
    }

    impl ProtocolParser for SyntheticParser {
        type Error = String;

        fn push(&mut self, text: &str, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
            for character in text.chars() {
                let mut encoded = [0; 4];
                let piece = character.encode_utf8(&mut encoded);
                match &mut self.state {
                    SyntheticState::Text(buffer) => {
                        for matched in buffer.push(piece) {
                            match matched {
                                PatternPiece::Text(text) => sink.text(text),
                                PatternPiece::Match { index: 0, .. } => {
                                    self.state = SyntheticState::Reasoning(reasoning_patterns());
                                }
                                PatternPiece::Match { index: 1, .. } => {
                                    self.state = SyntheticState::ToolHeader(String::new());
                                }
                                PatternPiece::Match { index, .. } => {
                                    unreachable!("unexpected text pattern {index}")
                                }
                            }
                        }
                    }
                    SyntheticState::Reasoning(buffer) => {
                        for matched in buffer.push(piece) {
                            match matched {
                                PatternPiece::Text(text) => sink.reasoning(text),
                                PatternPiece::Match { index: 0, .. } => {
                                    self.state = SyntheticState::Text(text_patterns());
                                }
                                PatternPiece::Match { index, .. } => {
                                    unreachable!("unexpected reasoning pattern {index}")
                                }
                            }
                        }
                    }
                    SyntheticState::ToolHeader(header) => {
                        if character == '>' {
                            let (id, name) = header
                                .split_once(':')
                                .ok_or_else(|| "synthetic tool header needs id:name".to_owned())?;
                            sink.start_tool_call(id.to_owned(), name.to_owned());
                            self.state =
                                SyntheticState::ToolArguments(JsonFragmentBuffer::default());
                        } else {
                            header.push(character);
                        }
                    }
                    SyntheticState::ToolArguments(json) => {
                        let (consumed, complete) =
                            json.push(piece).map_err(|error| format!("{error:?}"))?;
                        sink.tool_arguments(&piece[..consumed]);
                        if complete {
                            self.state = SyntheticState::ToolEnd(tool_end_patterns());
                        }
                    }
                    SyntheticState::ToolEnd(buffer) => {
                        for matched in buffer.push(piece) {
                            match matched {
                                PatternPiece::Text(text) if text.trim().is_empty() => {}
                                PatternPiece::Text(text) => {
                                    return Err(format!(
                                        "unexpected text before synthetic tool end: {text:?}"
                                    ));
                                }
                                PatternPiece::Match { index: 0, .. } => {
                                    sink.end_tool_call();
                                    self.state = SyntheticState::Text(text_patterns());
                                }
                                PatternPiece::Match { index, .. } => {
                                    unreachable!("unexpected tool-end pattern {index}")
                                }
                            }
                        }
                    }
                }
            }
            Ok(())
        }

        fn finish(&mut self, sink: &mut SemanticEventSink) -> Result<(), Self::Error> {
            match &mut self.state {
                SyntheticState::Text(buffer) => {
                    if let Some(text) = buffer.finish() {
                        sink.text(text);
                    }
                }
                SyntheticState::Reasoning(buffer) => {
                    if let Some(text) = buffer.finish() {
                        sink.reasoning(text);
                    }
                }
                SyntheticState::ToolHeader(_) | SyntheticState::ToolArguments(_) => {}
                SyntheticState::ToolEnd(_) => {}
            }
            Ok(())
        }
    }

    fn split_points(text: &str) -> impl Iterator<Item = usize> + '_ {
        (0..=text.len()).filter(|&index| text.is_char_boundary(index))
    }

    fn visible_text(events: &[SemanticEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::TextDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn reasoning_text(events: &[SemanticEvent]) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ReasoningDelta(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn tool_arguments(events: &[SemanticEvent], index: usize) -> String {
        events
            .iter()
            .filter_map(|event| match event {
                SemanticEvent::ToolArgumentsDelta {
                    index: event_index,
                    json_fragment,
                } if *event_index == index => Some(json_fragment.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn utf8_buffer_accepts_every_split_boundary() {
        let text = "ASCII café 東京 🦀";
        for first in 0..=text.len() {
            for second in first..=text.len() {
                let mut buffer = Utf8Buffer::default();
                let mut decoded = String::new();
                decoded.push_str(&buffer.push(&text.as_bytes()[..first]).unwrap());
                decoded.push_str(&buffer.push(&text.as_bytes()[first..second]).unwrap());
                decoded.push_str(&buffer.push(&text.as_bytes()[second..]).unwrap());
                buffer.finish().unwrap();
                assert_eq!(decoded, text, "byte splits {first}, {second}");
            }
        }
    }

    #[test]
    fn utf8_buffer_rejects_invalid_and_incomplete_final_bytes() {
        let mut incomplete = Utf8Buffer::default();
        assert_eq!(incomplete.push(&[0xf0, 0x9f]).unwrap(), "");
        assert_eq!(incomplete.finish(), Err(Utf8BufferError::Incomplete));

        let mut invalid = Utf8Buffer::default();
        assert!(matches!(
            invalid.push(&[b'a', 0xff]),
            Err(Utf8BufferError::Invalid { valid_up_to: 1 })
        ));
    }

    #[derive(Debug)]
    struct SyntheticTokenBackend;

    impl TokenDecoderBackend for SyntheticTokenBackend {
        type Error = Infallible;

        fn decode_token(
            &mut self,
            token_id: u32,
            preserve_special: bool,
        ) -> Result<Vec<u8>, Self::Error> {
            let bytes = match (token_id, preserve_special) {
                (1, _) => b"before".to_vec(),
                (2, true) => REASONING_START.as_bytes().to_vec(),
                (2, false) => Vec::new(),
                (3, _) => "pensée".as_bytes().to_vec(),
                (4, true) => REASONING_END.as_bytes().to_vec(),
                (4, false) => Vec::new(),
                (5, _) => b"after".to_vec(),
                (6, _) => Vec::new(),
                _ => unreachable!("unknown synthetic token"),
            };
            Ok(bytes)
        }
    }

    #[test]
    fn raw_decoding_preserves_structural_tokens_at_every_token_boundary() {
        let ids = [1, 2, 3, 6, 4, 5];
        for split in 0..=ids.len() {
            let mut decoder = RawTokenDecoder::new(SyntheticTokenBackend, [2, 4]);
            let mut bytes = Vec::new();
            for &id in ids[..split].iter().chain(&ids[split..]) {
                let token = decoder.push(id).unwrap();
                assert_eq!(token.structural, matches!(id, 2 | 4));
                bytes.extend(token.bytes);
            }
            assert_eq!(
                String::from_utf8(bytes).unwrap(),
                "before<r>pensée</r>after",
                "token split {split}"
            );
        }
    }

    #[test]
    fn overlapping_stops_match_at_every_split_without_leaking() {
        let input = "visible abab tail";
        for split in split_points(input) {
            let mut matcher = StopMatcher::new(["aba"], ["bab", "STOP"]);
            let first = matcher.push(&input[..split]);
            let second = matcher.push(&input[split..]);
            let visible = first.visible + &second.visible + &matcher.finish();
            assert_eq!(visible, "visible ", "split {split}");
            assert!(first.matched || second.matched, "split {split}");
            assert!(!visible.contains("aba"));
        }
    }

    #[test]
    fn profile_and_caller_stops_share_one_matcher() {
        for (input, expected) in [("text</s>tail", "text"), ("text STOP tail", "text ")] {
            for split in split_points(input) {
                let mut matcher = StopMatcher::new(["</s>"], ["STOP"]);
                let first = matcher.push(&input[..split]);
                let second = matcher.push(&input[split..]);
                assert_eq!(
                    first.visible + &second.visible + &matcher.finish(),
                    expected,
                    "input {input:?}, split {split}"
                );
            }
        }
    }

    #[test]
    fn partial_stop_is_released_at_eof() {
        let input = "visible STO";
        for split in split_points(input) {
            let mut matcher = StopMatcher::new([], ["STOP"]);
            let first = matcher.push(&input[..split]);
            let second = matcher.push(&input[split..]);
            assert!(!first.matched && !second.matched);
            assert_eq!(first.visible + &second.visible + &matcher.finish(), input);
        }
    }

    #[test]
    fn partial_trigger_tag_and_delimiter_match_at_every_boundary() {
        let input = "a<r>b</r>c::d";
        for split in split_points(input) {
            let mut buffer = PartialPatternBuffer::new([
                (PatternKind::Trigger, "<r>"),
                (PatternKind::Tag, "</r>"),
                (PatternKind::Delimiter, "::"),
            ]);
            let pieces = buffer
                .push(&input[..split])
                .into_iter()
                .chain(buffer.push(&input[split..]))
                .collect::<Vec<_>>();
            assert_eq!(
                pieces,
                [
                    PatternPiece::Text("a".into()),
                    PatternPiece::Match {
                        index: 0,
                        kind: PatternKind::Trigger
                    },
                    PatternPiece::Text("b".into()),
                    PatternPiece::Match {
                        index: 1,
                        kind: PatternKind::Tag
                    },
                    PatternPiece::Text("c".into()),
                    PatternPiece::Match {
                        index: 2,
                        kind: PatternKind::Delimiter
                    },
                    PatternPiece::Text("d".into()),
                ],
                "split {split}"
            );
            assert_eq!(buffer.finish(), None);
        }
    }

    #[test]
    fn mismatched_trigger_prefix_becomes_visible() {
        let input = "plain <rx text";
        for split in split_points(input) {
            let mut buffer = PartialPatternBuffer::new([(PatternKind::Trigger, REASONING_START)]);
            let mut visible = String::new();
            for piece in buffer
                .push(&input[..split])
                .into_iter()
                .chain(buffer.push(&input[split..]))
            {
                match piece {
                    PatternPiece::Text(text) => visible.push_str(&text),
                    PatternPiece::Match { .. } => panic!("mismatched trigger must not match"),
                }
            }
            visible.push_str(&buffer.finish().unwrap_or_default());
            assert_eq!(visible, input, "split {split}");
        }
    }

    #[test]
    fn json_fragments_complete_at_every_split_boundary() {
        let json = r#" {"text":"brace } and quote \"","nested":[1,{"ok":true}]} "#;
        let complete_len = json.trim_end().len();
        for split in split_points(json) {
            let mut buffer = JsonFragmentBuffer::default();
            let (first_consumed, first_complete) = buffer.push(&json[..split]).unwrap();
            let mut consumed = first_consumed;
            if !first_complete {
                let (second_consumed, _) = buffer.push(&json[split..]).unwrap();
                consumed += second_consumed;
            }
            assert!(buffer.is_complete(), "split {split}");
            assert_eq!(buffer.fragment(), &json[..complete_len], "split {split}");
            assert_eq!(consumed, complete_len, "split {split}");
        }
    }

    #[test]
    fn json_fragment_can_remain_incomplete_at_eof() {
        let mut buffer = JsonFragmentBuffer::default();
        let fragment = r#"{"nested":[1,2"#;
        assert_eq!(buffer.push(fragment).unwrap(), (fragment.len(), false));
        assert_eq!(buffer.fragment(), fragment);
        assert!(!buffer.is_complete());
    }

    #[test]
    fn synthetic_parser_is_split_independent() {
        let input = r#"<r>why 🦀</r>Hello <call:id-7:weather>{"city":"Bogotá"}</call> done"#;
        for split in split_points(input) {
            let mut stream = SemanticStream::new(SyntheticParser::default(), [], []);
            assert!(!stream.push(&input[..split]).unwrap());
            assert!(!stream.push(&input[split..]).unwrap());
            stream.finish(FinishReason::Eos).unwrap();
            let events = stream.events();

            assert_eq!(reasoning_text(events), "why 🦀", "split {split}");
            assert_eq!(visible_text(events), "Hello  done", "split {split}");
            assert_eq!(tool_arguments(events, 0), r#"{"city":"Bogotá"}"#);
            assert!(events.contains(&SemanticEvent::ToolCallStart {
                index: 0,
                id: "id-7".into(),
                name: "weather".into(),
            }));
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(event, SemanticEvent::ToolCallEnd))
                    .count(),
                1,
                "split {split}"
            );
            assert_eq!(
                events.last(),
                Some(&SemanticEvent::Finished {
                    reason: FinishReason::Eos
                })
            );
        }
    }

    #[test]
    fn matched_stop_never_reaches_visible_output() {
        let input = "safe STOP must remain hidden";
        for split in split_points(input) {
            let mut stream = SemanticStream::new(SyntheticParser::default(), ["</s>"], ["STOP"]);
            let first_matched = stream.push(&input[..split]).unwrap();
            let second_matched = stream.push(&input[split..]).unwrap();
            assert!(first_matched || second_matched);
            assert_eq!(visible_text(stream.events()), "safe ");
            assert_eq!(
                stream.events().last(),
                Some(&SemanticEvent::Finished {
                    reason: FinishReason::StopSequence
                })
            );
        }
    }

    #[test]
    fn partial_stop_at_eof_is_parsed_before_finished() {
        let input = "safe STO";
        for split in split_points(input) {
            let mut stream = SemanticStream::new(SyntheticParser::default(), [], ["STOP"]);
            stream.push(&input[..split]).unwrap();
            stream.push(&input[split..]).unwrap();
            stream.finish(FinishReason::MaxTokens).unwrap();
            assert_eq!(visible_text(stream.events()), input);
            assert_eq!(
                stream.events().last(),
                Some(&SemanticEvent::Finished {
                    reason: FinishReason::MaxTokens
                })
            );
        }
    }

    #[test]
    fn incomplete_final_calls_do_not_emit_tool_call_end() {
        for input in [
            r#"<call:id:name>{"partial":"#,
            r#"<call:id:name>{"complete":true}"#,
            r#"<call:id:name>{"complete":true}</ca"#,
        ] {
            for split in split_points(input) {
                let mut stream = SemanticStream::new(SyntheticParser::default(), [], []);
                stream.push(&input[..split]).unwrap();
                stream.push(&input[split..]).unwrap();
                stream.finish(FinishReason::Eos).unwrap();

                assert!(stream.events().contains(&SemanticEvent::ToolCallStart {
                    index: 0,
                    id: "id".into(),
                    name: "name".into(),
                }));
                assert!(!stream
                    .events()
                    .iter()
                    .any(|event| matches!(event, SemanticEvent::ToolCallEnd)));
                assert_eq!(
                    stream.events().last(),
                    Some(&SemanticEvent::Finished {
                        reason: FinishReason::Eos
                    })
                );
            }
        }
    }

    #[derive(Debug, Default)]
    struct SyntheticCommittedDecoder;

    impl TokenDecoderBackend for SyntheticCommittedDecoder {
        type Error = Infallible;

        fn decode_token(
            &mut self,
            token_id: u32,
            preserve_special: bool,
        ) -> Result<Vec<u8>, Self::Error> {
            Ok(match (token_id, preserve_special) {
                (1000, true) => REASONING_START.as_bytes().to_vec(),
                (1001, true) => REASONING_END.as_bytes().to_vec(),
                (1000 | 1001, false) => Vec::new(),
                (id, _) => vec![u8::try_from(id).expect("synthetic ordinary token is one byte")],
            })
        }
    }

    struct SyntheticModel {
        tokens: VecDeque<u32>,
        grammar_after: Option<usize>,
        next_calls: Rc<Cell<usize>>,
    }

    impl SyntheticModel {
        fn new(
            tokens: impl IntoIterator<Item = u32>,
            grammar_after: Option<usize>,
            next_calls: Rc<Cell<usize>>,
        ) -> Self {
            Self {
                tokens: tokens.into_iter().collect(),
                grammar_after,
                next_calls,
            }
        }
    }

    impl CommittedTokenSource for SyntheticModel {
        type Error = Infallible;

        fn next_token(&mut self) -> Result<Option<u32>, Self::Error> {
            self.next_calls.set(self.next_calls.get() + 1);
            Ok(self.tokens.pop_front())
        }

        fn grammar_is_complete(&mut self) -> Result<bool, Self::Error> {
            Ok(self
                .grammar_after
                .is_some_and(|count| self.next_calls.get() >= count))
        }
    }

    fn synthetic_generation(
        bytes: &[u8],
        grammar_after: Option<usize>,
        eos_token_ids: &[u32],
        max_tokens: usize,
        caller_stops: &[&str],
    ) -> (Vec<u32>, FinishReason, Vec<SemanticEvent>, usize) {
        let calls = Rc::new(Cell::new(0));
        let mut source = SyntheticModel::new(
            bytes.iter().copied().map(u32::from),
            grammar_after,
            calls.clone(),
        );
        let parser = ToolRuntimeParser::new(
            Box::new(SyntheticParser::default()),
            std::iter::empty(),
            caller_stops.iter().copied(),
        );
        let decoder = RawTokenDecoder::new(SyntheticCommittedDecoder, []);
        let mut pipeline = CommittedTokenPipeline::new(decoder, parser);
        let mut events = Vec::new();
        let (tokens, reason) = drive_committed_generation(
            &mut source,
            &mut pipeline,
            eos_token_ids,
            NonZeroUsize::new(max_tokens).unwrap(),
            &mut |event| events.push(event),
        )
        .unwrap();
        (tokens, reason, events, calls.get())
    }

    fn assert_exactly_one_finished(events: &[SemanticEvent], reason: FinishReason) {
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, SemanticEvent::Finished { .. }))
                .count(),
            1
        );
        assert_eq!(events.last(), Some(&SemanticEvent::Finished { reason }));
    }

    #[test]
    fn synthetic_model_stop_has_precedence_and_never_leaks() {
        let input = b"safe STOP hidden";
        let terminal_len = b"safe STOP".len();
        let (tokens, reason, events, calls) = synthetic_generation(
            input,
            Some(terminal_len),
            &[u32::from(b'P')],
            terminal_len,
            &["STOP"],
        );

        assert_eq!(reason, FinishReason::StopSequence);
        assert_eq!(tokens.len(), terminal_len);
        assert_eq!(calls, terminal_len);
        assert_eq!(visible_text(&events), "safe ");
        assert_exactly_one_finished(&events, reason);
    }

    #[test]
    fn synthetic_model_grammar_completion_precedes_eos_and_max_without_resampling() {
        let input = b"hello";
        let (tokens, reason, events, calls) = synthetic_generation(
            input,
            Some(input.len()),
            &[u32::from(b'o')],
            input.len(),
            &[],
        );

        assert_eq!(reason, FinishReason::GrammarComplete);
        assert_eq!(tokens.len(), input.len());
        assert_eq!(
            calls,
            input.len(),
            "no token sampled after grammar completion"
        );
        assert_eq!(visible_text(&events), "hello");
        assert_exactly_one_finished(&events, reason);
    }

    #[test]
    fn synthetic_model_eos_precedes_max_tokens() {
        let (tokens, reason, events, calls) =
            synthetic_generation(b"hi", None, &[u32::from(b'i')], 2, &[]);

        assert_eq!(reason, FinishReason::Eos);
        assert_eq!(tokens, [u32::from(b'h'), u32::from(b'i')]);
        assert_eq!(calls, 2);
        assert_exactly_one_finished(&events, reason);
    }

    #[test]
    fn synthetic_model_max_tokens_finalizes_once() {
        let (tokens, reason, events, calls) = synthetic_generation(b"more", None, &[], 2, &[]);

        assert_eq!(reason, FinishReason::MaxTokens);
        assert_eq!(tokens, [u32::from(b'm'), u32::from(b'o')]);
        assert_eq!(calls, 2);
        assert_eq!(visible_text(&events), "mo");
        assert_exactly_one_finished(&events, reason);
    }

    #[test]
    fn synthetic_model_delivers_events_before_completion_and_preserves_only_structural_specials() {
        let calls = Rc::new(Cell::new(0));
        let tokens = [
            u32::from(b'a'),
            1000,
            u32::from(b'x'),
            1001,
            u32::from(b'b'),
        ];
        let mut source = SyntheticModel::new(tokens, None, calls.clone());
        let parser = ToolRuntimeParser::new(
            Box::new(SyntheticParser::default()),
            std::iter::empty(),
            std::iter::empty(),
        );
        let decoder = RawTokenDecoder::new(SyntheticCommittedDecoder, [1000, 1001]);
        let mut pipeline = CommittedTokenPipeline::new(decoder, parser);
        let mut delivered = Vec::new();
        let (committed, reason) = drive_committed_generation(
            &mut source,
            &mut pipeline,
            &[],
            NonZeroUsize::new(tokens.len()).unwrap(),
            &mut |event| delivered.push((calls.get(), event)),
        )
        .unwrap();
        let events = delivered
            .iter()
            .map(|(_, event)| event.clone())
            .collect::<Vec<_>>();

        assert_eq!(committed, tokens);
        assert_eq!(reason, FinishReason::MaxTokens);
        assert!(
            delivered
                .iter()
                .any(|(at_call, event)| *at_call < tokens.len()
                    && !matches!(event, SemanticEvent::Finished { .. })),
            "at least one semantic delta must arrive while the model can still be sampled"
        );
        assert_eq!(visible_text(&events), "ab");
        assert_eq!(reasoning_text(&events), "x");
        assert!(!format!("{events:?}").contains(REASONING_START));
        assert!(!format!("{events:?}").contains(REASONING_END));
        assert_exactly_one_finished(&events, reason);
    }
}
