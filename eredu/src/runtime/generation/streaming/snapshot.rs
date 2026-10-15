use super::*;
use crate::runtime::generation::storage::{snapshot_fields, SnapshotStorage};

snapshot_fields!(InProgressToolCall {
    index,
    id,
    name,
    arguments
});
snapshot_fields!(Utf8Buffer { pending });
snapshot_fields!(StopMatcher {
    stops,
    pending,
    matched
});
snapshot_fields!(PartialPatternBuffer { patterns, pending });
snapshot_fields!(JsonFragmentBuffer {
    fragment,
    depth,
    in_string,
    escaped,
    started,
    complete
});
snapshot_fields!(SemanticEventSink {
    events,
    active_tool_call,
    next_tool_index,
    tool_calls_enabled,
    tool_schemas,
    tool_names
});
snapshot_fields!(ToolRuntimeParser { stream });

impl SnapshotStorage for PatternKind {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Trigger => Some(0),
            #[cfg(test)]
            Self::Tag | Self::Delimiter => Some(0),
        }
    }
}
impl SnapshotStorage for SemanticEvent {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::TextDelta(v) | Self::ReasoningDelta(v) => v.heap_bytes(),
            Self::ToolCallStart { index: _, id, name } => {
                id.heap_bytes()?.checked_add(name.heap_bytes()?)
            }
            Self::ToolArgumentsDelta {
                index: _,
                json_fragment,
            } => json_fragment.heap_bytes(),
            Self::ToolCallCompleted {
                id,
                name,
                arguments,
                ..
            } => id
                .heap_bytes()?
                .checked_add(name.heap_bytes()?)?
                .checked_add(arguments.heap_bytes()?),
            Self::Finished { reason: _ } => Some(0),
        }
    }
}
impl SnapshotStorage for Box<dyn ProtocolParser<Error = String>> {
    fn heap_bytes(&self) -> Option<u64> {
        self.snapshot_storage_bytes()
    }
}
impl<P: SnapshotStorage> SnapshotStorage for SemanticStream<P> {
    fn heap_bytes(&self) -> Option<u64> {
        let Self {
            parser,
            stops,
            structural_stops,
            sink,
            finished: _,
        } = self;
        parser
            .heap_bytes()?
            .checked_add(stops.heap_bytes()?)?
            .checked_add(structural_stops.heap_bytes()?)?
            .checked_add(sink.heap_bytes()?)
    }
}
impl<D: TokenDecoderBackend> SnapshotStorage for RawTokenDecoder<D> {
    fn heap_bytes(&self) -> Option<u64> {
        let Self {
            backend,
            structural_token_ids,
            structural_token_spellings,
        } = self;
        backend
            .snapshot_storage_bytes()?
            .checked_sub(std::mem::size_of::<D>() as u64)?
            .checked_add(structural_token_ids.heap_bytes()?)?
            .checked_add(structural_token_spellings.heap_bytes()?)
    }
}
impl<D: TokenDecoderBackend> CommittedTokenPipeline<D> {
    pub(crate) fn snapshot_storage_bytes(&self) -> Option<u64> {
        let Self {
            decoder,
            utf8,
            parser,
        } = self;
        (std::mem::size_of::<Self>() as u64)
            .checked_add(decoder.heap_bytes()?)?
            .checked_add(utf8.heap_bytes()?)?
            .checked_add(parser.heap_bytes()?)
    }
}
impl CommittedGenerationCursor {
    pub(crate) fn max_predictions(&self) -> u64 {
        self.sequence.max_tokens() as u64
    }
    pub(crate) fn snapshot_storage_bytes(&self) -> Option<u64> {
        let Self {
            sequence,
            finish_reason: _,
            failed: _,
        } = self;
        (std::mem::size_of::<Self>() as u64)
            .checked_add(sequence.snapshot_storage_bytes()?)?
            .checked_sub(std::mem::size_of::<GenerationSequence>() as u64)
    }
}

impl<D: TokenDecoderBackend> CommittedTokenPipeline<D> {
    pub(crate) fn tool_calls_enabled(&self) -> bool {
        let sink = &self.parser.stream.sink;
        sink.tool_calls_enabled
            && sink
                .tool_schemas
                .as_ref()
                .is_none_or(|schemas| !schemas.is_empty())
    }

    pub(crate) fn continuation_event_storage_bytes(&self, max_predictions: u64) -> Option<u64> {
        let (_, input) = self
            .decoder
            .backend
            .continuation_storage_bounds(max_predictions)?;
        self.parser
            .stream
            .parser
            .continuation_event_storage_bytes(input, max_predictions)
            .and_then(|bytes| {
                bytes.checked_add(self.parser.stream.sink.tool_names.event_growth_bytes(input)?)
            })
    }

    pub(crate) fn continuation_storage_bytes(&self, max_predictions: u64) -> Option<u64> {
        let (decoder, input) = self
            .decoder
            .backend
            .continuation_storage_bounds(max_predictions)?;
        let parser = self
            .parser
            .stream
            .parser
            .continuation_storage_bytes(input)?;
        // UTF-8 and stop lookbehind retain input bytes. The sink can retain one
        // active call's ID, name and normalized arguments; queued deltas drain
        // synchronously before a branch boundary. Include those text copies and
        // generated u64 call IDs, plus the committed canonical cursor history.
        decoder
            .checked_add(parser)?
            .checked_add(input.checked_mul(10)?)?
            // A closing byte can complete an argument buffer inherited at the
            // saved boundary. Price its parsed object as well as fresh input.
            .checked_add(
                input
                    .checked_add(self.parser.stream.sink.active_tool_call.snapshot_bytes()?)?
                    .checked_mul(if self.tool_calls_enabled() {
                        6 * crate::runtime::generation::storage::JSON_STORAGE_PER_BYTE
                    } else {
                        0
                    })?,
            )?
            .checked_add(64)?
            .checked_add(max_predictions.checked_mul(4)?)?
            .checked_add(self.parser.stream.sink.tool_names.event_growth_bytes(input)?)
    }
}
