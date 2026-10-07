//! Moshi/PersonaPlex host output projection from declared realtime channels.

use eredu_core::{
    RealtimeChannelId, RealtimeChannelLayout, RealtimeDecisionDiagnostics, RealtimeOutputFrame,
    RealtimeSpeechConfig,
};
use eredu_runtime::{CompletedRealtimeFrame, RealtimeHostChannelValues, RealtimeIngressContract};

/// Architecture-owned mapping from canonical Moshi channels to public codec frames.
#[derive(Debug, Clone)]
pub struct MoshiRealtimeOutputProjection {
    schedule: RealtimeSpeechConfig,
    channels: RealtimeChannelLayout,
}

impl MoshiRealtimeOutputProjection {
    /// Retains the exact admitted schedule and channel declarations.
    pub fn new(contract: &RealtimeIngressContract) -> Self {
        Self {
            schedule: contract.schedule().clone(),
            channels: contract.channels().clone(),
        }
    }

    /// Collects host columns in batch-major text/depth/generated-audio order.
    /// An unready delayed frame has no aligned audio. Warmup frames carry
    /// padding values but no model decisions or decision diagnostics.
    /// Shared host validation runs before any architecture output is collected.
    pub fn project(
        &self,
        frame: &CompletedRealtimeFrame<RealtimeHostChannelValues, RealtimeDecisionDiagnostics>,
    ) -> Result<RealtimeOutputFrame, MoshiRealtimeOutputError> {
        eredu_runtime::validate_completed_realtime_host_frame(frame)
            .map_err(MoshiRealtimeOutputError::HostValues)?;
        if frame.channels() != &self.channels {
            return Err(MoshiRealtimeOutputError::LayoutMismatch);
        }
        let batch = frame.batch();
        let column = |id| {
            categorical_column(
                frame
                    .value(id)
                    .ok_or(MoshiRealtimeOutputError::MissingChannel(id))?,
                id,
                batch,
            )
        };
        let text = column(RealtimeChannelId::new(0))?.to_vec();
        let decisions = frame.decision_channels();
        if !decisions.is_empty()
            && (decisions.len() != self.schedule.depth_audio_codebooks() + 1
                || decisions
                    .iter()
                    .enumerate()
                    .any(|(index, id)| id.index() != index))
        {
            return Err(MoshiRealtimeOutputError::DecisionOrder);
        }
        let decision_columns = decisions
            .iter()
            .skip(1)
            .copied()
            .map(column)
            .collect::<Result<Vec<_>, _>>()?;
        let generated = self.schedule.generated_audio_codebooks();
        let sampled_columns = (1..=generated)
            .map(RealtimeChannelId::new)
            .map(column)
            .collect::<Result<Vec<_>, _>>()?;
        let aligned = if frame.aligned().is_empty() {
            None
        } else {
            if frame.aligned().len() != generated {
                return Err(MoshiRealtimeOutputError::AlignedChannels);
            }
            let columns = (1..=generated)
                .map(|index| {
                    let id = RealtimeChannelId::new(index);
                    let value = frame
                        .aligned()
                        .iter()
                        .find(|value| value.channel() == id)
                        .ok_or(MoshiRealtimeOutputError::MissingChannel(id))?;
                    categorical_column(value.payload(), id, batch)
                })
                .collect::<Result<Vec<_>, _>>()?;
            Some(batch_major(&columns, batch)?)
        };
        Ok(RealtimeOutputFrame::new(
            batch,
            text,
            batch_major(&decision_columns, batch)?,
            batch_major(&sampled_columns, batch)?,
            aligned,
            frame.diagnostics().to_vec(),
        ))
    }
}

fn categorical_column(
    values: &RealtimeHostChannelValues,
    channel: RealtimeChannelId,
    batch: usize,
) -> Result<&[i32], MoshiRealtimeOutputError> {
    match values {
        RealtimeHostChannelValues::Categorical(values) if values.len() == batch => Ok(values),
        _ => Err(MoshiRealtimeOutputError::Column(channel)),
    }
}

fn batch_major(columns: &[&[i32]], batch: usize) -> Result<Vec<i32>, MoshiRealtimeOutputError> {
    let count = batch
        .checked_mul(columns.len())
        .ok_or(MoshiRealtimeOutputError::Extent)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(count)
        .map_err(|_| MoshiRealtimeOutputError::Extent)?;
    for row in 0..batch {
        output.extend(columns.iter().map(|column| column[row]));
    }
    Ok(output)
}

/// A completed envelope cannot be interpreted as the retained Moshi schedule.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MoshiRealtimeOutputError {
    /// A host payload violates its declared geometry, kind or domain.
    #[error(transparent)]
    HostValues(#[from] eredu_runtime::RealtimeChannelIngressError),
    /// Completion and selected architecture differ.
    #[error("completed realtime layout differs from the selected Moshi layout")]
    LayoutMismatch,
    /// A required semantic channel is absent.
    #[error("completed Moshi frame is missing channel {0:?}")]
    MissingChannel(RealtimeChannelId),
    /// Moshi channels hold one categorical value per batch row.
    #[error("completed Moshi channel {0:?} is not a categorical batch column")]
    Column(RealtimeChannelId),
    /// Moshi decisions follow one text decision and all depth codebooks.
    #[error("completed Moshi decision channels differ from text-then-depth order")]
    DecisionOrder,
    /// Aligned output must contain the complete generated codebook set.
    #[error("completed Moshi aligned output does not contain all generated codebooks")]
    AlignedChannels,
    /// Host output cannot be allocated with checked geometry.
    #[error("completed Moshi host output extent exceeds available storage")]
    Extent,
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{RealtimeFrameConvention, RealtimeSlotCoordinate};
    use eredu_runtime::{RealtimeFrameChannel, TokenDomain};

    #[test]
    fn projection_preserves_batch_rows_depth_order_and_delayed_channel_identity() {
        let contract = RealtimeIngressContract::new(
            RealtimeSpeechConfig::new(
                4,
                2,
                2,
                4,
                0,
                0,
                RealtimeFrameConvention::AbsoluteDelayedSlots,
                vec![0, 0, 1, 0, 1],
            )
            .unwrap(),
            TokenDomain::new(32),
            TokenDomain::new(32),
        )
        .unwrap();
        let entry = |id, values: [i32; 2]| {
            let id = RealtimeChannelId::new(id);
            RealtimeFrameChannel::new(
                id,
                Some(RealtimeSlotCoordinate::new(2, id)),
                RealtimeHostChannelValues::Categorical(values.to_vec()),
            )
        };
        let frame = CompletedRealtimeFrame::new(
            contract.channels().clone(),
            2,
            vec![
                entry(3, [13, 23]),
                entry(0, [10, 20]),
                entry(2, [12, 22]),
                entry(4, [14, 24]),
                entry(1, [11, 21]),
            ],
            vec![entry(2, [2, 4]), entry(1, [1, 3])],
            (0..=4).map(RealtimeChannelId::new).collect(),
            vec![],
        )
        .unwrap();
        let projection = MoshiRealtimeOutputProjection::new(&contract);
        let output = projection.project(&frame).unwrap();
        assert_eq!(output.text_tokens(), [10, 20]);
        assert_eq!(
            output.decision_audio_tokens(),
            [11, 12, 13, 14, 21, 22, 23, 24]
        );
        assert_eq!(output.sampled_audio_tokens(), [11, 12, 21, 22]);
        assert_eq!(output.output_audio_tokens(), Some([1, 2, 3, 4].as_slice()));

        let warmup = CompletedRealtimeFrame::new(
            contract.channels().clone(),
            2,
            vec![entry(0, [0, 0]), entry(1, [0, 0]), entry(2, [0, 0])],
            vec![],
            vec![],
            vec![],
        )
        .unwrap();
        let output = projection.project(&warmup).unwrap();
        assert!(output.decision_audio_tokens().is_empty());
        assert_eq!(output.sampled_audio_tokens(), [0, 0, 0, 0]);
        assert!(output.output_audio_tokens().is_none());
    }
}
