//! Independent copies of committed, undelivered publications.

use eredu_core::{
    execution_control::{ExecutionControlError, SnapshotEstimate},
    generation::SemanticEvent,
    BackendFailure,
};
use eredu_runtime::execution_control::{
    snapshot_value_estimate, SnapshotReservation, SnapshotValue,
};

use super::*;

impl<D> Queue<D> {
    pub(in crate::runtime::voicechat) fn delivery_identity(&self) -> Rc<()> {
        self.delivered.clone()
    }
}

impl<D: SnapshotValue> SnapshotValue for Queue<D> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let slots = self
            .limits
            .max_publications
            .checked_mul(std::mem::size_of::<Queued<D>>())
            .and_then(|bytes| bytes.checked_add(64))
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        snapshot_value_estimate::<Self>(
            std::iter::once(Ok(SnapshotEstimate {
                retained_bytes: slots as u64,
                copy_bytes: slots as u64,
            }))
            .chain(
                self.output
                    .iter()
                    .map(|item| item.output.estimate_snapshot_value()),
            ),
        )
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        let mut output = VecDeque::new();
        output
            .try_reserve_exact(self.limits.max_publications)
            .map_err(BackendFailure::from_error)?;
        for item in &self.output {
            output.push_back(Queued {
                output: item.output.copy_snapshot_value(retention)?,
                charge: PublicationCharge(item.charge.0),
            });
        }
        Ok(Self {
            limits: self.limits,
            output,
            usage: self.usage,
            delivered: self.delivered.clone(),
            fork_retention: None,
            retained_growth: None,
            retained_estimator: None,
        })
    }
}

fn event_strings(event: &SemanticEvent) -> impl Iterator<Item = &String> {
    let fields = match event {
        SemanticEvent::ReasoningDelta(text) | SemanticEvent::TextDelta(text) => [Some(text), None],
        SemanticEvent::ToolCallStart { id, name, .. }
        | SemanticEvent::ToolCallCompleted { id, name, .. } => [Some(id), Some(name)],
        SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => [Some(json_fragment), None],
        SemanticEvent::Finished { .. } => [None, None],
    };
    fields.into_iter().flatten()
}

fn text_estimate(text: &String) -> Result<SnapshotEstimate, BackendFailure> {
    let bytes = (std::mem::size_of::<String>() as u64)
        .checked_add(text.capacity() as u64)
        .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
    Ok(SnapshotEstimate {
        retained_bytes: bytes,
        copy_bytes: bytes,
    })
}

impl<D: SnapshotValue> SnapshotValue for VoiceChatConversationOutput<D> {
    fn estimate_snapshot_value(&self) -> Result<SnapshotEstimate, BackendFailure> {
        let tool_slots = self
            .tools
            .as_ref()
            .map_or(0, |batch| batch.events.capacity())
            .checked_mul(std::mem::size_of::<SemanticEvent>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
        let tools = self
            .tools
            .iter()
            .flat_map(|batch| batch.events.iter())
            .map(|event| {
                let arguments = if let SemanticEvent::ToolCallCompleted { arguments, .. } = event {
                    Some(arguments_estimate(arguments))
                } else {
                    None
                };
                snapshot_value_estimate::<SemanticEvent>(
                    event_strings(event).map(text_estimate).chain(arguments),
                )
            });
        snapshot_value_estimate::<Self>(
            std::iter::once(self.frame.estimate_snapshot_value())
                .chain(std::iter::once(Ok(SnapshotEstimate {
                    retained_bytes: tool_slots,
                    copy_bytes: 0,
                })))
                .chain(tools)
                .chain(
                    self.agent_text
                        .iter()
                        .map(|event| text_estimate(&event.text)),
                )
                .chain(
                    self.transcription
                        .iter()
                        .map(|event| text_estimate(&event.text)),
                ),
        )
    }

    fn copy_snapshot_value(&self, retention: &SnapshotReservation) -> Result<Self, BackendFailure> {
        Ok(Self {
            frame: self.frame.copy_snapshot_value(retention)?,
            turn: self.turn,
            tools: self
                .tools
                .as_ref()
                .map(|batch| {
                    let mut events = Vec::new();
                    events
                        .try_reserve_exact(batch.events.len())
                        .map_err(BackendFailure::from_error)?;
                    for event in &batch.events {
                        events.push(copy_event(event)?);
                    }
                    Ok::<_, BackendFailure>(super::super::VoiceChatToolBatch {
                        ticket: batch.ticket,
                        events,
                    })
                })
                .transpose()?,
            playback: self.playback,
            agent_text: self
                .agent_text
                .as_ref()
                .map(|event| {
                    Ok::<_, BackendFailure>(super::super::VoiceChatAgentText {
                        text: copy_text(&event.text)?,
                    })
                })
                .transpose()?,
            transcription: self
                .transcription
                .as_ref()
                .map(|event| {
                    Ok::<_, BackendFailure>(super::super::VoiceChatTranscriptionText {
                        text: copy_text(&event.text)?,
                    })
                })
                .transpose()?,
        })
    }
}

fn copy_text(text: &str) -> Result<String, BackendFailure> {
    let mut copied = String::new();
    copied
        .try_reserve_exact(text.len())
        .map_err(BackendFailure::from_error)?;
    copied.push_str(text);
    Ok(copied)
}

fn copy_event(event: &SemanticEvent) -> Result<SemanticEvent, BackendFailure> {
    Ok(match event {
        SemanticEvent::ReasoningDelta(text) => SemanticEvent::ReasoningDelta(copy_text(text)?),
        SemanticEvent::TextDelta(text) => SemanticEvent::TextDelta(copy_text(text)?),
        SemanticEvent::ToolCallStart { index, id, name } => SemanticEvent::ToolCallStart {
            index: *index,
            id: copy_text(id)?,
            name: copy_text(name)?,
        },
        SemanticEvent::ToolArgumentsDelta {
            index,
            json_fragment,
        } => SemanticEvent::ToolArgumentsDelta {
            index: *index,
            json_fragment: copy_text(json_fragment)?,
        },
        SemanticEvent::ToolCallCompleted {
            index,
            id,
            name,
            arguments,
        } => SemanticEvent::ToolCallCompleted {
            index: *index,
            id: copy_text(id)?,
            name: copy_text(name)?,
            arguments: copy_arguments(arguments)?,
        },
        SemanticEvent::Finished { reason } => SemanticEvent::Finished { reason: *reason },
    })
}

// Include container slots and conservative tree-node overhead alongside all
// recursively owned strings. One KiB per entry also covers a sparse B-tree
// node. These copies are admitted before allocation.
fn arguments_estimate(
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<SnapshotEstimate, BackendFailure> {
    snapshot_value_estimate::<serde_json::Map<String, serde_json::Value>>(arguments.iter().map(
        |(key, value)| {
            let mut estimate = snapshot_value_estimate::<serde_json::Value>([
                text_estimate(key),
                json_estimate(value),
            ])?;
            estimate.retained_bytes = estimate
                .retained_bytes
                .checked_add(1024)
                .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
            estimate.copy_bytes = estimate
                .copy_bytes
                .checked_add(1024)
                .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?;
            Ok(estimate)
        },
    ))
}

fn json_estimate(value: &serde_json::Value) -> Result<SnapshotEstimate, BackendFailure> {
    match value {
        serde_json::Value::String(text) => text_estimate(text),
        serde_json::Value::Object(object) => arguments_estimate(object),
        serde_json::Value::Array(values) => {
            let slots = values
                .capacity()
                .checked_mul(std::mem::size_of::<serde_json::Value>())
                .ok_or_else(|| BackendFailure::from_error(ExecutionControlError::Overflow))?
                as u64;
            snapshot_value_estimate::<serde_json::Value>(
                std::iter::once(Ok(SnapshotEstimate {
                    retained_bytes: slots,
                    copy_bytes: slots,
                }))
                .chain(values.iter().map(json_estimate)),
            )
        }
        _ => snapshot_value_estimate::<serde_json::Value>([]),
    }
}

fn copy_arguments(
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<serde_json::Map<String, serde_json::Value>, BackendFailure> {
    arguments
        .iter()
        .map(|(key, value)| Ok((copy_text(key)?, copy_json(value)?)))
        .collect()
}

fn copy_json(value: &serde_json::Value) -> Result<serde_json::Value, BackendFailure> {
    Ok(match value {
        serde_json::Value::String(text) => serde_json::Value::String(copy_text(text)?),
        serde_json::Value::Object(object) => serde_json::Value::Object(copy_arguments(object)?),
        serde_json::Value::Array(values) => {
            let mut copied = Vec::new();
            copied
                .try_reserve_exact(values.len())
                .map_err(BackendFailure::from_error)?;
            for value in values {
                copied.push(copy_json(value)?);
            }
            serde_json::Value::Array(copied)
        }
        _ => value.clone(),
    })
}
