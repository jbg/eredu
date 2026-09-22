//! Source-only native transport calls made by the shared receipt workers.
use super::*;

/// A checked upper population of one actual transport callback. This describes
/// calls only: it creates no communication occurrence, reservation or source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionCaptureTransportDemand {
    /// Revalidate the selected communication owner before the next protocol phase.
    Active,
    /// One exact protocol frame, with a bounded actual local word count.
    Gather {
        kind: PartitionCaptureFrameKind,
        maximum_words: usize,
    },
    /// One padded receipt writer, separate from the gather result and readout.
    Words { maximum_elements: usize },
    /// One rank's canonical receipt reconstruction scratch.
    Bytes { maximum_elements: usize },
}

/// Source votes for every scheduled row plus the one enclosing coordination
/// frame. Rejected/skipped receipts still vote their actual source outcome;
/// an empty schedule still coordinates its admitted frame. Interventions own
/// their additional member/evidence exchanges separately.
pub fn partition_capture_coordination_demands(
    source: &SharedCapturePlan,
    phase: CapturePhase,
    prediction: u64,
) -> impl Iterator<Item = PartitionCaptureTransportDemand> + '_ {
    use PartitionCaptureTransportDemand as D;
    std::iter::once(D::Active)
        .chain(
            source
                .admission()
                .plan()
                .selections
                .iter()
                .filter(move |selection| selection.schedule.includes(phase, prediction))
                .map(|_| D::Gather {
                    kind: PartitionCaptureFrameKind::Source,
                    maximum_words: PartitionCaptureFrameKind::Source
                        .fixed_words()
                        .expect("fixed source frame"),
                }),
        )
        .chain([
            D::Active,
            D::Gather {
                kind: PartitionCaptureFrameKind::Coordination,
                maximum_words: PartitionCaptureFrameKind::Coordination
                    .fixed_words()
                    .expect("fixed coordination frame"),
            },
        ])
}
impl PartitionCaptureReceiptPlan {
    /// The same admitted exchange and decoder callbacks, at this receipt's
    /// maximum encoded width. A failing execution consumes only a prefix.
    /// Decoder scratch is allocated for every rank, including inactive ranks,
    /// because padding and local-byte equality are checked before delivery.
    pub fn transport_demands(
        &self,
    ) -> Result<impl Iterator<Item = PartitionCaptureTransportDemand> + '_, CaptureError> {
        use PartitionCaptureTransportDemand as D;
        let width = self.maximum_payload_words()?;
        let bytes = width.checked_mul(4).ok_or(CaptureError::Overflow)?;
        let mut frames = self.protocol_frames()?;
        let (prepare, pwords) = frames.next().expect("preparation");
        let (payload, words) = frames.next().expect("payload");
        let (delivery, dwords) = frames.next().expect("delivery");
        Ok([
            D::Active,
            D::Active,
            D::Gather {
                kind: prepare,
                maximum_words: pwords,
            },
            D::Words {
                maximum_elements: width,
            },
            D::Gather {
                kind: payload,
                maximum_words: words,
            },
        ]
        .into_iter()
        .chain(std::iter::repeat_n(
            D::Bytes {
                maximum_elements: bytes,
            },
            self.world_size(),
        ))
        .chain(std::iter::once(D::Gather {
            kind: delivery,
            maximum_words: dwords,
        })))
    }
}
