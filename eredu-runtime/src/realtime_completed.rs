//! Validated categorical and dense outputs of a completed realtime transition.
//!
//! This envelope records channel identity, publication coordinates and ordered
//! categorical decisions. Payloads remain opaque; native mechanisms and host
//! observers validate their representation against the declared geometry.

use std::collections::BTreeSet;

use eredu_core::{
    RealtimeChannelId, RealtimeChannelLayout, RealtimeChannelPayload, RealtimeSlotCoordinate,
};

/// One opaque channel value, optionally attributed to a retained timeline position.
#[derive(Debug, Clone)]
pub struct RealtimeFrameChannel<T> {
    channel: RealtimeChannelId,
    coordinate: Option<RealtimeSlotCoordinate>,
    payload: T,
}

impl<T> RealtimeFrameChannel<T> {
    /// Supplies a channel payload for validation by [`CompletedRealtimeFrame::new`].
    /// Warm-up padding without a timeline position has no coordinate.
    pub const fn new(
        channel: RealtimeChannelId,
        coordinate: Option<RealtimeSlotCoordinate>,
        payload: T,
    ) -> Self {
        Self {
            channel,
            coordinate,
            payload,
        }
    }
    /// Architecture-local channel identity.
    pub const fn channel(&self) -> RealtimeChannelId {
        self.channel
    }
    /// Position in the channel's declared coordinate policy, when attributed.
    pub const fn coordinate(&self) -> Option<RealtimeSlotCoordinate> {
        self.coordinate
    }
    /// Opaque value with the channel's declared per-batch geometry.
    pub const fn payload(&self) -> &T {
        &self.payload
    }
    /// Consumes this attributed value into its payload.
    pub fn into_payload(self) -> T {
        self.payload
    }
}

/// One transition's current and delay-aligned outputs with exact channel identity.
///
/// Each collection contains a channel at most once. The same channel may occur
/// in both collections at different coordinates. Decision order is explicit and
/// independent of declaration or payload order; dense derived outputs consume
/// no categorical decision ordinal. An empty diagnostics list means capture was
/// disabled; otherwise it contains exactly one entry per ordered decision.
#[derive(Debug, Clone)]
pub struct CompletedRealtimeFrame<T, D> {
    channels: RealtimeChannelLayout,
    batch: usize,
    values: Vec<RealtimeFrameChannel<T>>,
    aligned: Vec<RealtimeFrameChannel<T>>,
    decision_channels: Vec<RealtimeChannelId>,
    diagnostics: Vec<D>,
}

impl<T, D> CompletedRealtimeFrame<T, D> {
    /// Validates all output attribution before completion or host publication.
    ///
    /// Coordinates must lie on their channel's declared cadence. Delay arithmetic
    /// belongs to architecture policy and is not applied again here. This checks
    /// geometry arithmetic for the batch without inspecting native tensor types.
    pub fn new(
        channels: RealtimeChannelLayout,
        batch: usize,
        values: Vec<RealtimeFrameChannel<T>>,
        aligned: Vec<RealtimeFrameChannel<T>>,
        decision_channels: Vec<RealtimeChannelId>,
        diagnostics: Vec<D>,
    ) -> Result<Self, RealtimeCompletedFrameError> {
        if batch == 0 {
            return Err(RealtimeCompletedFrameError::EmptyBatch);
        }
        for channel in channels.channels() {
            if channel.payload.elements(batch).is_none() {
                return Err(RealtimeCompletedFrameError::Geometry(channel.id));
            }
        }
        let validate_values = |payloads: &[RealtimeFrameChannel<T>], aligned: bool| {
            let mut seen = BTreeSet::new();
            for value in payloads {
                let channel = channels
                    .channel(value.channel)
                    .ok_or(RealtimeCompletedFrameError::UnknownChannel(value.channel))?;
                if !seen.insert(value.channel) {
                    return Err(if aligned {
                        RealtimeCompletedFrameError::DuplicateAligned(value.channel)
                    } else {
                        RealtimeCompletedFrameError::DuplicateValue(value.channel)
                    });
                }
                if aligned && value.coordinate.is_none() {
                    return Err(RealtimeCompletedFrameError::MissingAlignedCoordinate(
                        value.channel,
                    ));
                }
                if let Some(coordinate) = value.coordinate {
                    if coordinate.slot() != value.channel {
                        return Err(RealtimeCompletedFrameError::CoordinateChannel {
                            channel: value.channel,
                            coordinate,
                        });
                    }
                    if !channel.cadence.is_active(coordinate.position()) {
                        return Err(RealtimeCompletedFrameError::InactiveCoordinate(coordinate));
                    }
                }
            }
            Ok(seen)
        };
        let present = validate_values(&values, false)?;
        validate_values(&aligned, true)?;
        let mut decisions = BTreeSet::new();
        for &id in &decision_channels {
            let declaration = channels
                .channel(id)
                .ok_or(RealtimeCompletedFrameError::UnknownChannel(id))?;
            if !decisions.insert(id) {
                return Err(RealtimeCompletedFrameError::DuplicateDecision(id));
            }
            if !matches!(
                declaration.payload,
                RealtimeChannelPayload::Categorical { .. }
            ) {
                return Err(RealtimeCompletedFrameError::DenseDecision(id));
            }
            if !present.contains(&id) {
                return Err(RealtimeCompletedFrameError::MissingDecisionValue(id));
            }
        }
        if !diagnostics.is_empty() && diagnostics.len() != decision_channels.len() {
            return Err(RealtimeCompletedFrameError::DiagnosticCount {
                expected: decision_channels.len(),
                actual: diagnostics.len(),
            });
        }
        Ok(Self {
            channels,
            batch,
            values,
            aligned,
            decision_channels,
            diagnostics,
        })
    }
    /// Exact declared channel layout.
    pub const fn channels(&self) -> &RealtimeChannelLayout {
        &self.channels
    }
    /// Validated positive batch size.
    pub const fn batch(&self) -> usize {
        self.batch
    }
    /// Current outputs, including categorical decisions, dense values or padding.
    pub fn values(&self) -> &[RealtimeFrameChannel<T>] {
        &self.values
    }
    /// Available outputs selected at architecture-declared aligned coordinates.
    pub fn aligned(&self) -> &[RealtimeFrameChannel<T>] {
        &self.aligned
    }
    /// Categorical decisions in actual execution order, excluding dense derivations.
    pub fn decision_channels(&self) -> &[RealtimeChannelId] {
        &self.decision_channels
    }
    /// Optional diagnostics in the same order as [`Self::decision_channels`].
    pub fn diagnostics(&self) -> &[D] {
        &self.diagnostics
    }
    /// Looks up one current channel value without assuming a declaration order.
    pub fn value(&self, channel: RealtimeChannelId) -> Option<&T> {
        self.values
            .iter()
            .find(|value| value.channel == channel)
            .map(RealtimeFrameChannel::payload)
    }
    /// All current and aligned payloads whose resources require completion ownership.
    pub fn retained_payloads(&self) -> impl Iterator<Item = &T> {
        self.values
            .iter()
            .chain(&self.aligned)
            .map(RealtimeFrameChannel::payload)
    }
}

/// Invalid completed-channel attribution, independent of backend failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum RealtimeCompletedFrameError {
    /// A completed frame requires a positive batch.
    #[error("completed realtime frame has an empty batch")]
    EmptyBatch,
    /// Declared payload size overflows for the selected batch.
    #[error("completed realtime channel {0:?} has overflowing batch geometry")]
    Geometry(RealtimeChannelId),
    /// A value or decision names an undeclared channel.
    #[error("completed realtime channel {0:?} is not declared")]
    UnknownChannel(RealtimeChannelId),
    /// Current values repeat one channel.
    #[error("completed realtime channel {0:?} has duplicate current values")]
    DuplicateValue(RealtimeChannelId),
    /// Aligned values repeat one channel.
    #[error("completed realtime channel {0:?} has duplicate aligned values")]
    DuplicateAligned(RealtimeChannelId),
    /// An aligned output has no selected timeline coordinate.
    #[error("aligned realtime channel {0:?} has no timeline coordinate")]
    MissingAlignedCoordinate(RealtimeChannelId),
    /// Ordered decisions repeat one channel.
    #[error("completed realtime channel {0:?} has duplicate decisions")]
    DuplicateDecision(RealtimeChannelId),
    /// The payload's coordinate names a different channel.
    #[error("completed realtime channel {channel:?} disagrees with coordinate {coordinate:?}")]
    CoordinateChannel {
        /// Payload identity.
        channel: RealtimeChannelId,
        /// Inconsistent coordinate.
        coordinate: RealtimeSlotCoordinate,
    },
    /// A coordinate falls outside the channel's declared cadence.
    #[error("completed realtime coordinate {0:?} is inactive under its declared cadence")]
    InactiveCoordinate(RealtimeSlotCoordinate),
    /// Dense derived outputs cannot consume categorical decision ordinals.
    #[error("dense realtime channel {0:?} cannot be a categorical decision")]
    DenseDecision(RealtimeChannelId),
    /// A declared decision has no current value.
    #[error("completed realtime decision channel {0:?} has no current value")]
    MissingDecisionValue(RealtimeChannelId),
    /// Captured diagnostics do not cover the complete ordered decision set.
    #[error("completed realtime frame has {actual} diagnostics, expected {expected}")]
    DiagnosticCount {
        /// Required diagnostic count when capture is enabled.
        expected: usize,
        /// Actual diagnostic count.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::{RealtimeChannelCadence, RealtimeChannelDeclaration};

    fn id(value: usize) -> RealtimeChannelId {
        RealtimeChannelId::new(value)
    }
    fn layout() -> RealtimeChannelLayout {
        RealtimeChannelLayout::new(
            "mixed_frame_output".into(),
            vec![
                RealtimeChannelDeclaration {
                    id: id(7),
                    payload: RealtimeChannelPayload::Categorical {
                        width: 1,
                        cardinality: 13,
                    },
                    cadence: RealtimeChannelCadence {
                        period: 1,
                        phase: 0,
                    },
                    delay: 0,
                    feedback: vec![],
                },
                RealtimeChannelDeclaration {
                    id: id(11),
                    payload: RealtimeChannelPayload::Dense { shape: vec![2, 3] },
                    cadence: RealtimeChannelCadence {
                        period: 2,
                        phase: 1,
                    },
                    delay: 0,
                    feedback: vec![],
                },
            ],
        )
        .unwrap()
    }
    fn value(
        channel: usize,
        position: Option<usize>,
        payload: &'static str,
    ) -> RealtimeFrameChannel<&'static str> {
        RealtimeFrameChannel::new(
            id(channel),
            position.map(|n| RealtimeSlotCoordinate::new(n, id(channel))),
            payload,
        )
    }

    #[test]
    fn completed_channels_preserve_dense_coordinates_and_ordered_diagnostic_attribution() {
        let frame = CompletedRealtimeFrame::new(
            layout(),
            2,
            vec![
                value(11, Some(3), "current_pcm"),
                value(7, None, "forced_category"),
            ],
            vec![value(11, Some(1), "aligned_pcm")],
            vec![id(7)],
            vec!["category_logits"],
        )
        .unwrap();
        assert_eq!(frame.value(id(7)), Some(&"forced_category"));
        assert_eq!(frame.value(id(11)), Some(&"current_pcm"));
        assert_eq!(frame.value(id(0)), None);
        assert_eq!(frame.decision_channels(), &[id(7)]);
        assert_eq!(frame.diagnostics(), &["category_logits"]);
        assert_eq!(
            frame.values()[0].coordinate(),
            Some(RealtimeSlotCoordinate::new(3, id(11)))
        );
        assert_eq!(
            frame.aligned()[0].coordinate(),
            Some(RealtimeSlotCoordinate::new(1, id(11)))
        );
        assert_eq!(
            frame.retained_payloads().copied().collect::<Vec<_>>(),
            vec!["current_pcm", "forced_category", "aligned_pcm"]
        );
    }

    #[test]
    fn completed_channels_reject_misattribution_before_publication() {
        let finish = |values, aligned, decisions, diagnostics| {
            CompletedRealtimeFrame::new(layout(), 1, values, aligned, decisions, diagnostics)
                .unwrap_err()
        };
        assert_eq!(
            finish(
                vec![value(99, None, "unknown")],
                vec![],
                vec![],
                Vec::<&str>::new()
            ),
            RealtimeCompletedFrameError::UnknownChannel(id(99))
        );
        assert_eq!(
            finish(
                vec![value(7, None, "a"), value(7, None, "b")],
                vec![],
                vec![],
                vec![]
            ),
            RealtimeCompletedFrameError::DuplicateValue(id(7))
        );
        assert_eq!(
            finish(
                vec![],
                vec![value(11, Some(1), "a"), value(11, Some(3), "b")],
                vec![],
                vec![]
            ),
            RealtimeCompletedFrameError::DuplicateAligned(id(11))
        );
        assert_eq!(
            finish(
                vec![],
                vec![value(11, None, "unattributed")],
                vec![],
                vec![]
            ),
            RealtimeCompletedFrameError::MissingAlignedCoordinate(id(11))
        );
        let wrong = RealtimeSlotCoordinate::new(1, id(11));
        assert_eq!(
            finish(
                vec![RealtimeFrameChannel::new(id(7), Some(wrong), "wrong")],
                vec![],
                vec![],
                vec![]
            ),
            RealtimeCompletedFrameError::CoordinateChannel {
                channel: id(7),
                coordinate: wrong
            }
        );
        assert_eq!(
            finish(
                vec![value(11, Some(2), "off_cadence")],
                vec![],
                vec![],
                vec![]
            ),
            RealtimeCompletedFrameError::InactiveCoordinate(RealtimeSlotCoordinate::new(2, id(11)))
        );
        assert_eq!(
            finish(
                vec![value(11, Some(1), "dense")],
                vec![],
                vec![id(11)],
                vec![]
            ),
            RealtimeCompletedFrameError::DenseDecision(id(11))
        );
        assert_eq!(
            finish(vec![], vec![], vec![id(7)], vec![]),
            RealtimeCompletedFrameError::MissingDecisionValue(id(7))
        );
        assert_eq!(
            finish(
                vec![value(7, None, "category")],
                vec![],
                vec![id(7), id(7)],
                vec![]
            ),
            RealtimeCompletedFrameError::DuplicateDecision(id(7))
        );
        assert_eq!(
            finish(
                vec![value(7, None, "category")],
                vec![],
                vec![id(7)],
                vec!["a", "b"]
            ),
            RealtimeCompletedFrameError::DiagnosticCount {
                expected: 1,
                actual: 2
            }
        );
        assert_eq!(
            CompletedRealtimeFrame::<(), ()>::new(layout(), 0, vec![], vec![], vec![], vec![])
                .unwrap_err(),
            RealtimeCompletedFrameError::EmptyBatch
        );
        assert_eq!(
            CompletedRealtimeFrame::<(), ()>::new(
                layout(),
                usize::MAX,
                vec![],
                vec![],
                vec![],
                vec![]
            )
            .unwrap_err(),
            RealtimeCompletedFrameError::Geometry(id(11))
        );
    }
}
