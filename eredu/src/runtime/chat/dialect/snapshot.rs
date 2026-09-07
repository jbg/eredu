use super::*;
use crate::runtime::generation::storage::{snapshot_fields, SnapshotStorage};

impl DeclarativeParser {
    pub(super) fn continuation_bound(&self, input_bytes: u64) -> Option<u64> {
        // Price all mutually exclusive states together: pending/raw JSON/name/
        // ID/argument/key strings; normalized structural JSON (at most six bytes
        // per input byte); parsed JSON values and object keys; container frames
        // and duplicate-key sets. A value, key or frame needs at least one input
        // byte. Schema tables are immutable and already included in the copy.
        let per_byte = 9u64
            .checked_add(6)?
            .checked_add(3)?
            .checked_add(std::mem::size_of::<Value>() as u64)?
            .checked_add(2 * std::mem::size_of::<String>() as u64)?
            .checked_add(std::mem::size_of::<StructuralContainer>() as u64)?;
        // Existing buffered input can become parsed/normalized state after the
        // fork. Count its whole live footprint as additional input to the bound.
        input_bytes
            .checked_add(self.snapshot_bytes()?)?
            .checked_mul(per_byte)
    }
}

snapshot_fields!(DeclarativeParser {
    spec,
    tagged_tools,
    state,
    pending
});
snapshot_fields!(TaggedToolSchema {
    parameters,
    required
});
snapshot_fields!(IncrementalJsonCall {
    phase,
    fragment,
    fields,
    name,
    id,
    arguments,
    arguments_seen,
    arguments_emitted,
    started,
    complete,
});
snapshot_fields!(JsonValueAccumulator { raw, kind });
snapshot_fields!(StructuralObjectNormalizer {
    string_delimiter,
    phase,
    containers,
    normalized,
    emitted,
    complete,
});

impl SnapshotStorage for ChannelKind {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Reasoning | Self::Text => Some(0),
        }
    }
}
impl SnapshotStorage for DeclarativeParserState {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::PrefilledChannelOrTool { kind, suffix } | Self::Channel { kind, suffix } => {
                kind.heap_bytes()?.checked_add(suffix.heap_bytes()?)
            }
            Self::JsonPayload(value) => value.heap_bytes(),
            Self::NamedJsonPayload { json, emitted: _ } => json.heap_bytes(),
            Self::TaggedParameterOrEnd { tool, arguments }
            | Self::TaggedParameterName { tool, arguments } => {
                tool.heap_bytes()?.checked_add(arguments.heap_bytes()?)
            }
            Self::TaggedParameterValue {
                tool,
                parameter,
                arguments,
            } => tool
                .heap_bytes()?
                .checked_add(parameter.heap_bytes()?)?
                .checked_add(arguments.heap_bytes()?),
            Self::StructuralPayload { normalizer } => normalizer.heap_bytes(),
            Self::Outside
            | Self::ToolStart
            | Self::JsonEnvelopeStart
            | Self::NamedJsonName
            | Self::TaggedFunctionPrefix
            | Self::TaggedFunctionName
            | Self::StructuralName { prefix_consumed: _ }
            | Self::AfterPayload
            | Self::AfterEnvelope
            | Self::ListItemOrEnd { allow_end: _ }
            | Self::ToolSuffix => Some(0),
        }
    }
}
impl SnapshotStorage for JsonCallPhase {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Start | Self::KeyOrEnd { allow_end: _ } | Self::AfterValue => Some(0),
            Self::Key { raw, escaped: _ } => raw.heap_bytes(),
            Self::Colon { key } | Self::ValueStart { key } => key.heap_bytes(),
            Self::Value { key, value } => key.heap_bytes()?.checked_add(value.heap_bytes()?),
        }
    }
}
impl SnapshotStorage for JsonValueKind {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Container {
                depth: _,
                in_string: _,
                escaped: _,
            }
            | Self::String { escaped: _ }
            | Self::Scalar => Some(0),
        }
    }
}
impl SnapshotStorage for StructuralPhase {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Start | Self::Value { allow_array_end: _ } | Self::String | Self::AfterValue => {
                Some(0)
            }
            Self::ObjectKey { key, allow_end: _ } => key.heap_bytes(),
            Self::Scalar { raw } => raw.heap_bytes(),
        }
    }
}
impl SnapshotStorage for StructuralContainer {
    fn heap_bytes(&self) -> Option<u64> {
        match self {
            Self::Object { keys } => keys.heap_bytes(),
            Self::Array => Some(0),
        }
    }
}
