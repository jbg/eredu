//! Exact reached protocol frames and custody-retaining host destinations.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// The existing receipt protocol's actual collective occurrence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionCaptureFrameKind {
    /// Pre-forward identity and quota agreement.
    Coordination,
    /// Actual selected producer's scalar witness, before native observation.
    Source,
    /// Producer presence, local status and bounded receipt length.
    Preparation,
    /// Padded receipt words, sized from the completed preparation exchange.
    Payload,
    /// Post-decoding all-rank delivery decision.
    Delivery,
    /// The existing post-edit identity, acceptance and affected-count receipt.
    InterventionReceipt,
}

/// Borrowed words produced by the shared capture protocol. This descriptive
/// source grants no native allocation or execution permission. Its private
/// constructor preserves the exact protocol population before backend entry.
#[derive(Debug)]
pub struct PartitionCaptureFrame<'a> {
    kind: PartitionCaptureFrameKind,
    rank: usize,
    participants: usize,
    words: &'a [u32],
    gathered_words: usize,
}
impl<'a> PartitionCaptureFrame<'a> {
    pub(in crate::capture::partition) fn new(
        kind: PartitionCaptureFrameKind,
        rank: usize,
        participants: usize,
        words: &'a [u32],
        maximum: usize,
    ) -> Result<Self, PartitionCaptureExchangeError> {
        let gathered_words = words
            .len()
            .checked_mul(participants)
            .ok_or(CaptureError::Overflow)?;
        let fixed_width = match kind {
            PartitionCaptureFrameKind::Payload => None,
            PartitionCaptureFrameKind::InterventionReceipt => Some(18),
            _ => Some(16),
        };
        if participants == 0
            || rank >= participants
            || words.is_empty()
            || words.len() > maximum
            || fixed_width.is_some_and(|width| words.len() != width)
            || gathered_words > isize::MAX as usize / size_of::<u32>()
        {
            return Err(PartitionCaptureExchangeError::Protocol(
                "capture frame population",
            ));
        }
        Ok(Self {
            kind,
            rank,
            participants,
            words,
            gathered_words,
        })
    }
    /// Actual shared protocol occurrence.
    pub const fn kind(&self) -> PartitionCaptureFrameKind {
        self.kind
    }
    /// Actual sender in the retained world.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Required participant population.
    pub const fn participants(&self) -> usize {
        self.participants
    }
    /// Canonical protocol words; the backend does not encode another frame.
    pub const fn words(&self) -> &[u32] {
        self.words
    }
    /// Exact rank-major completed destination length.
    pub const fn gathered_words(&self) -> usize {
        self.gathered_words
    }
}

/// A finite host destination whose contents retire before their cumulative
/// funding. No Vec can escape independently of this owner.
#[derive(Debug)]
pub struct PartitionCaptureBuffer<T> {
    values: Vec<T>,
    limit: usize,
    _funding: Option<HostMetadataFunding>,
}
/// Typed constructor failure retaining the original metadata account.
#[derive(Debug, thiserror::Error)]
pub enum PartitionCaptureStorageError {
    /// Reservation refused before constructing a destination.
    #[error("capture destination reservation: {cause}")]
    Funding {
        /// Exact fixed reservation failure.
        #[source]
        cause: HostMetadataFundingError,
        /// Original cumulative account; a failure never refunds prior work.
        funding: HostMetadataFunding,
    },
    /// The existing counted metadata vector producer failed.
    #[error("capture destination construction: {cause}")]
    Destination {
        /// Original typed producer error.
        #[source]
        cause: eredu_nn::Error,
        /// Account outliving the producer's error payload.
        funding: HostMetadataFunding,
    },
}
impl<T> PartitionCaptureBuffer<T> {
    pub(super) fn ordinary(values: Vec<T>) -> Self {
        let limit = values.capacity();
        Self {
            values,
            limit,
            _funding: None,
        }
    }
    /// Reserve and construct one exact destination through the existing counted
    /// metadata vector producer. This grants no numerical/native backing.
    pub fn funded(
        capacity: usize,
        funding: &HostMetadataFunding,
    ) -> Result<Self, PartitionCaptureStorageError> {
        let controls = [
            size_of::<Self>(),
            size_of::<Option<HostMetadataFunding>>(),
            size_of::<Result<Self, PartitionCaptureStorageError>>(),
            size_of::<PartitionCaptureStorageError>(),
            size_of::<(&HostMetadataFunding, usize)>(),
        ];
        let bytes = controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| PartitionCaptureStorageError::Funding {
                cause: HostMetadataFundingError::Overflow,
                funding: funding.clone(),
            })?;
        funding
            .reserve_metadata(bytes)
            .map_err(|cause| PartitionCaptureStorageError::Funding {
                cause,
                funding: funding.clone(),
            })?;
        let values = funding.metadata_vec(capacity).map_err(|cause| {
            PartitionCaptureStorageError::Destination {
                cause,
                funding: funding.clone(),
            }
        })?;
        Ok(Self {
            values,
            limit: capacity,
            _funding: Some(funding.clone()),
        })
    }
    /// Fixed capacity established before the actual writer runs.
    pub const fn limit(&self) -> usize {
        self.limit
    }
}
impl<T: Copy> PartitionCaptureBuffer<T> {
    /// Copy into the existing destination without growth or a fresh reservation.
    pub fn extend_from_slice(&mut self, values: &[T]) -> Result<(), PartitionCaptureExchangeError> {
        if values.len() > self.limit.saturating_sub(self.values.len()) {
            return Err(PartitionCaptureExchangeError::Protocol(
                "capture destination capacity",
            ));
        }
        self.values.extend_from_slice(values);
        Ok(())
    }
    pub(super) fn resize(
        &mut self,
        length: usize,
        value: T,
    ) -> Result<(), PartitionCaptureExchangeError> {
        if length > self.limit {
            return Err(PartitionCaptureExchangeError::Protocol(
                "capture destination capacity",
            ));
        }
        self.values.resize(length, value);
        Ok(())
    }
}
impl<T> std::ops::Deref for PartitionCaptureBuffer<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        &self.values
    }
}
impl<T> std::ops::DerefMut for PartitionCaptureBuffer<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.values
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_nn::workspace::HostMetadataAccount;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };

    #[derive(Debug)]
    struct Account {
        used: Arc<AtomicUsize>,
        refuse: Arc<AtomicBool>,
        retired: Arc<AtomicBool>,
    }
    impl HostMetadataAccount for Account {
        fn reserve_metadata(&self, bytes: usize) -> Result<(), HostMetadataFundingError> {
            if self.refuse.load(Ordering::SeqCst) {
                return Err(HostMetadataFundingError::Overflow);
            }
            self.used.fetch_add(bytes, Ordering::SeqCst);
            Ok(())
        }
    }
    impl Drop for Account {
        fn drop(&mut self) {
            self.retired.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn capture_destinations_keep_spent_funding_through_output_and_refusal() {
        let used = Arc::new(AtomicUsize::new(0));
        let refuse = Arc::new(AtomicBool::new(false));
        let retired = Arc::new(AtomicBool::new(false));
        let funding = HostMetadataFunding::new(Account {
            used: used.clone(),
            refuse: refuse.clone(),
            retired: retired.clone(),
        })
        .unwrap();
        let mut words = PartitionCaptureBuffer::<u32>::funded(4, &funding).unwrap();
        words
            .extend_from_slice(&[17, 0x03bb, 65539, u32::MAX])
            .unwrap();
        let spent = used.load(Ordering::SeqCst);
        assert!(spent > 16);
        assert!(words.extend_from_slice(&[23]).is_err());
        assert_eq!(&*words, &[17, 0x03bb, 65539, u32::MAX]);
        assert_eq!(used.load(Ordering::SeqCst), spent);
        refuse.store(true, Ordering::SeqCst);
        let failure = PartitionCaptureBuffer::<u8>::funded(37, &funding).unwrap_err();
        drop(funding);
        assert!(!retired.load(Ordering::SeqCst));
        drop(words);
        assert!(!retired.load(Ordering::SeqCst));
        drop(failure);
        assert!(retired.load(Ordering::SeqCst));
    }

    #[test]
    fn capture_frame_retains_actual_nonzero_protocol_population() {
        let words = [
            0x4552_4353,
            2,
            1,
            3,
            11,
            0,
            7,
            13,
            19,
            29,
            31,
            37,
            41,
            43,
            0,
            0,
        ];
        let frame =
            PartitionCaptureFrame::new(PartitionCaptureFrameKind::Coordination, 1, 3, &words, 16)
                .unwrap();
        assert_eq!(frame.words().as_ptr(), words.as_ptr());
        assert_eq!(frame.gathered_words(), 48);
        assert_eq!((frame.rank(), frame.participants()), (1, 3));
        let payload = [0xcebbu32, 0x12345678, 0xff];
        let frame =
            PartitionCaptureFrame::new(PartitionCaptureFrameKind::Payload, 2, 3, &payload, 5)
                .unwrap();
        assert_eq!(frame.words(), payload);
        assert_eq!(frame.gathered_words(), 9);
        assert!(
            PartitionCaptureFrame::new(PartitionCaptureFrameKind::Payload, 2, 3, &payload, 2)
                .is_err()
        );
        assert!(PartitionCaptureFrame::new(
            PartitionCaptureFrameKind::Preparation,
            0,
            3,
            &payload,
            16
        )
        .is_err());
    }
}

impl<T> AsRef<[T]> for PartitionCaptureBuffer<T> {
    fn as_ref(&self) -> &[T] { &self.values }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
