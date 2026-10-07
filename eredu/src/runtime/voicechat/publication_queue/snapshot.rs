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
        SemanticEvent::ToolCallStart { id, name, .. } => [Some(id), Some(name)],
        SemanticEvent::ToolArgumentsDelta { json_fragment, .. } => [Some(json_fragment), None],
        SemanticEvent::ToolCallEnd | SemanticEvent::Finished { .. } => [None, None],
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
                snapshot_value_estimate::<SemanticEvent>(event_strings(event).map(text_estimate))
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
        SemanticEvent::ToolCallEnd => SemanticEvent::ToolCallEnd,
        SemanticEvent::Finished { reason } => SemanticEvent::Finished { reason: *reason },
    })
}
