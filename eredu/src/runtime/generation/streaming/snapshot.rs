use super::*;
use crate::runtime::generation::storage::{SnapshotStorage, snapshot_fields};

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
snapshot_fields!(JsonFragmentBuffer { fragment, cursor });
impl SnapshotStorage for eredu_text::json_fragments::FragmentCursor {
    fn heap_bytes(&self) -> Option<u64> { Some(0) }
}

snapshot_fields!(SemanticEventSink {
    events,
    active_tool_call,
    next_tool_index,
    tool_calls_enabled,
    tool_schemas
});
impl SnapshotStorage for ToolRuntimeParser {
    fn heap_bytes(&self) -> Option<u64> {
        let Self {
            stream,
            host_preparation: _,
        } = self;
        // Shared exclusion custody carries no copied numerical payload. This
        // logical ledger remains distinct from physical memory admission.
        stream.heap_bytes()
    }
}

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
            Self::TextDelta(v) | Self::ReasoningDelta(v) => {
                u64::try_from(v.snapshot_copy_bytes()).ok()
            }
            Self::ToolCallStart { index: _, id, name } => {
                u64::try_from(id.snapshot_copy_bytes()).ok()?
                    .checked_add(u64::try_from(name.snapshot_copy_bytes()).ok()?)
            }
            Self::ToolArgumentsDelta {
                index: _,
                json_fragment,
            } => u64::try_from(json_fragment.snapshot_copy_bytes()).ok(),
            Self::ToolCallEnd | Self::Finished { reason: _ } => Some(0),
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
        self.sequence
            .as_ref()
            .expect("live legacy sequence")
            .max_tokens() as u64
    }
    pub(crate) fn snapshot_storage_bytes(&self) -> Option<u64> {
        let Self {
            sequence,
            finish_reason: _,
            failed: _,
        } = self;
        (std::mem::size_of::<Self>() as u64)
            .checked_add(sequence.as_ref()?.snapshot_storage_bytes()?)?
            .checked_sub(std::mem::size_of::<GenerationSequence>() as u64)
    }
}

impl<D: TokenDecoderBackend> CommittedTokenPipeline<D> {
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
            .checked_add(64)?
            .checked_add(max_predictions.checked_mul(4)?)
    }
}
