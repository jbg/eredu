//! Prepaid bounded receipt transport with a final all-rank delivery decision.
use super::*;
use eredu_core::{
    consensus::BoundedConsensusTransport, BackendFailure, BoundedCompletion, BoundedCompletionWait,
    BoundedSubmissionOutcome, Completion, CompletionCancellationMode,
};

mod delivery;
pub(crate) use delivery::{PartitionCaptureDecoder, PartitionCapturePayload};

const MAGIC: u32 = 0x4552_4350;
const VERSION: u32 = 1;
const HEADER_WORDS: usize = 16;

/// Native facts and shared failure ownership for a capture receipt exchange.
/// Implementations reuse their selected portable-word transport. Estimates must
/// include retained input/output and native temporaries plus host resolution;
/// runtime separately prices its framing buffers. These are logical reservations,
/// never an assertion about physical allocator ceilings.
pub trait PartitionCaptureTransport: BoundedConsensusTransport {
    /// Rank in the exact retained participant topology.
    fn capture_rank(&self) -> usize;
    /// Selected bounded wait; querying it must not submit work.
    fn capture_wait(&self) -> Result<BoundedCompletionWait, CaptureError>;
    /// Rejects a previously failed shared execution/communication owner.
    fn ensure_capture_active(&self) -> Result<(), BackendFailure>;
    /// Side-effect-free upper bound for one equal-word gather and host resolution.
    fn estimate_capture_gather(&self, local_words: usize) -> Result<CaptureUsage, CaptureError>;
    /// Allocate the actual payload writer before encoding. Original adapters
    /// override this with the retained metadata account; ordinary storage is unchanged.
    fn capture_word_destination(&self, capacity: usize)
        -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::ordinary(Vec::with_capacity(capacity)))
    }
    /// Allocate the actual receipt decoder's byte scratch before reconstruction.
    fn capture_byte_destination(&self, capacity: usize)
        -> Result<PartitionCaptureBuffer<u8>, PartitionCaptureExchangeError> {
        Ok(PartitionCaptureBuffer::ordinary(Vec::with_capacity(capacity)))
    }
    /// Execute one exact shared protocol frame. Original adapters use their
    /// explicitly lent request source and independent authenticated child role.
    /// The default keeps the ordinary bounded submission/completion worker.
    fn gather_capture_frame(&self, frame: &PartitionCaptureFrame<'_>, wait: BoundedCompletionWait)
        -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError>
    where Self: Sized, Self::Error: Send + Sync + 'static,
        <Self::Completion as Completion>::Error: Send + Sync + 'static,
    {
        ordinary_gather_capture_words(self, wait, frame)
    }
    /// Fences this execution after transport or unagreed protocol failure. This does not establish
    /// native completion: the exact completion owner still settles or quarantines
    /// submitted work before releasing resources, including on polling errors.
    fn fail_capture_exchange(&self, error: &PartitionCaptureExchangeError);
}

/// Cooperative receipt exchange phase, distinct from a model computation phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionCaptureExchangeStage {
    /// Every rank compared the same live run, layout and prepaid work before forward.
    Coordination,
    /// Every rank supplied its local preparation status and bounded receipt length.
    Preparation,
    /// Completed payload transport and local decoding/assembly.
    Delivery,
}

/// Typed exchange failure; native sources stay behind the neutral failure type.
#[derive(Debug, thiserror::Error)]
pub enum PartitionCaptureExchangeError {
    /// The original canonical record writer retained its exact account.
    #[error(transparent)]
    Encoding(#[from] PartitionCaptureEncodingError),
    /// The original scheduled tensor decoder retained its spent claim and H.
    #[error(transparent)]
    TensorDestination(#[from] crate::working_memory::PartitionCaptureTensorDecodeError),
    /// The exact existing scheduled frame rejected a destination or receipt.
    #[error(transparent)]
    ScheduledDestination(#[from] crate::working_memory::CaptureRunHostError),
    /// Exact paid host destination failed; its metadata account stays retained.
    #[error(transparent)]
    Storage(#[from] PartitionCaptureStorageError),
    /// Invalid input or exhausted capture reservation.
    #[error(transparent)]
    Capture(#[from] CaptureError),
    /// Work exceeded an already prepaid allowance. This is a bound violation,
    /// not optional exhaustion of the application's capture budget.
    #[error("partition capture exceeded its prepaid allowance: {source}")]
    PrepaidBound {
        /// Original typed quota failure, retained for diagnostics.
        #[source]
        source: CaptureError,
    },
    /// Received fragments could not establish a complete value.
    #[error(transparent)]
    Merge(#[from] PartitionCaptureMergeError),
    /// Native submission, completion, or output resolution failed.
    #[error(transparent)]
    Backend(#[from] BackendFailure),
    /// Bounded completion applied the selected safe disposition.
    #[error("partition capture transport deadline exceeded ({cancellation:?})")]
    Deadline {
        /// Disposition actually applied by the completion owner.
        cancellation: CompletionCancellationMode,
    },
    /// At least one peer rejected preparation or delivery; no result is released.
    #[error("partition capture rank {rank} rejected {stage:?}")]
    PeerRejected {
        /// First rejecting world rank.
        rank: usize,
        /// Rejected phase.
        stage: PartitionCaptureExchangeStage,
    },
    /// All ranks completed a rejection decision; retains this rank's precise cause.
    /// The enclosing forward driver decides skip, rollback, or termination policy.
    #[error("partition capture rank {rank} rejected {stage:?}: {source}")]
    LocalRejected {
        /// Rank reporting the retained local failure.
        rank: usize,
        /// Rejected phase.
        stage: PartitionCaptureExchangeStage,
        /// Original typed preparation, decoding, or assembly failure.
        #[source]
        source: Box<PartitionCaptureExchangeError>,
    },
    /// The collective result did not satisfy its exact rank-major frame contract.
    #[error("partition capture protocol mismatch: {0}")]
    Protocol(&'static str),
}

/// Move-only, prepaid exchange for one selected forward receipt.
///
/// Admit this on every rank before forward work and agree admission through the
/// enclosing session's control protocol. An admission error submits no collective;
/// it cannot itself coordinate peers that have not obtained this authority. Once
/// admitted, all ranks (including inactive pipeline stages) call `exchange` at the
/// same post-forward boundary, supplying local failure instead of returning early.
/// This must never be called from a hook reached by only the active pipeline stage.
/// The enclosing run owns monotone epochs and the final generation commit.
pub struct PartitionCaptureExchange<'a, T: PartitionCaptureTransport> {
    transport: &'a T,
    receipt: Option<PartitionCaptureReceiptPlan>,
    wait: BoundedCompletionWait,
    rank: usize,
    participants: usize,
    digest: [u32; 8],
    payload_words: usize,
    reserved: CaptureUsage,
}

impl<'a, T: PartitionCaptureTransport> PartitionCaptureExchange<'a, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    pub(super) fn transport(&self) -> &'a T {
        self.transport
    }
    pub(crate) const fn local_rank(&self) -> usize { self.rank }
    /// Reserves both control rounds and the largest allowed payload gather before
    /// native work. Credits cannot be refunded, cloned, or consumed a second time.
    /// Decoding and assembly retain their ordinary separate ledger reservations;
    /// their failures can still vote using these already-paid control credits.
    pub fn admit(
        transport: &'a T,
        receipt: PartitionCaptureReceiptPlan,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureExchangeError> {
        transport.ensure_capture_active()?;
        let participants = transport.participant_count();
        let rank = transport.capture_rank();
        if participants != receipt.world_size() || rank >= participants {
            return Err(PartitionCaptureExchangeError::Protocol("admitted topology"));
        }
        u32::try_from(participants).map_err(|_| CaptureError::Overflow)?;
        let wait = transport.capture_wait()?;
        if !T::Completion::supports_cancellation(wait.cancellation()) {
            return Err(
                CaptureError::Unsupported("capture transport cancellation policy".into()).into(),
            );
        }
        let mut digest = [0; 8];
        for (word, bytes) in digest
            .iter_mut()
            .zip(receipt.identity().as_bytes().chunks_exact(8))
        {
            *word = u32::from_str_radix(
                std::str::from_utf8(bytes).map_err(|_| CaptureError::Overflow)?,
                16,
            )
            .map_err(|_| CaptureError::Overflow)?;
        }
        let payload_words = word_count(receipt.max_record_bytes())?;
        let reserved = Self::estimate_usage(transport, &receipt)?;
        ledger.reserve_quota(reserved)?;
        Ok(Self {
            transport,
            receipt: Some(receipt),
            wait,
            rank,
            participants,
            digest,
            payload_words,
            reserved,
        })
    }

    /// Cold upper bound used when a live session prepays a global step. Admission
    /// repeats this calculation against its move-only local allowance.
    pub fn estimate_usage(
        transport: &T,
        receipt: &PartitionCaptureReceiptPlan,
    ) -> Result<CaptureUsage, CaptureError> {
        let participants = transport.participant_count();
        if participants != receipt.world_size() {
            return Err(CaptureError::Invalid(
                "capture exchange topology differs".into(),
            ));
        }
        let payload_words = word_count(receipt.max_record_bytes())?;
        gather_usage(transport, participants, HEADER_WORDS)?
            .checked_mul(2)?
            .checked_add(gather_usage(transport, participants, payload_words)?)
    }

    /// Retained authority used to encode this rank's actual completed fragments.
    pub fn receipt_plan(&self) -> &PartitionCaptureReceiptPlan {
        self.receipt
            .as_ref()
            .expect("receipt authority is consumed only during exchange")
    }

    // The closed original delivery may move its actual receipt into a local
    // hook owner while the transport continuation remains idle. No protocol
    // operation is available through that continuation until ownership returns.
    pub(crate) fn take_hook_receipt(&mut self) -> Option<PartitionCaptureReceiptPlan> {
        self.receipt.take()
    }
    pub(crate) fn restore_hook_receipt(&mut self, receipt: PartitionCaptureReceiptPlan)
        -> Result<(), PartitionCaptureReceiptPlan> {
        if self.receipt.is_some() || receipt.world_size()!=self.participants
            || word_count(receipt.max_record_bytes()).ok()!=Some(self.payload_words)
            || receipt.identity().len()!=64 {
            return Err(receipt);
        }
        // The same exact SHA-256 words that admitted all protocol frames.
        for (expected,bytes) in self.digest.iter().zip(receipt.identity().as_bytes().chunks_exact(8)) {
            let actual=std::str::from_utf8(bytes).ok().and_then(|text|u32::from_str_radix(text,16).ok());
            if actual!=Some(*expected){return Err(receipt);}
        }
        self.receipt=Some(receipt);Ok(())
    }

    /// Nonrefundable transport reservation, separate from producer/decode charges.
    pub const fn reserved(&self) -> CaptureUsage {
        self.reserved
    }

    /// Exchanges one optional producer receipt or a local failure, then validates
    /// every receipt and agrees success before releasing the assembled host value.
    /// Nonproducers supply `Ok(None)` and participate in both control rounds.
    /// Native preparation failures may be passed as `Backend`; no output is forced
    /// and no unsuccessful/partial capture is converted into a measured zero.
    pub fn exchange(
        self,
        local: Result<Option<Vec<u8>>, PartitionCaptureExchangeError>,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<ReceivedPartitionCapture, PartitionCaptureExchangeError> {
        self.exchange_into(local.map(|bytes|bytes.map(PartitionCaptureBuffer::ordinary)),
            delivery::OrdinaryDecoder { ledger })
    }

    /// Same protocol worker for a closed original destination. Decoder controls
    /// must be reserved before forward; only the actual receipt/host consumer
    /// inside this crate can select this entry.
    pub(crate) fn exchange_into<D: PartitionCaptureDecoder<T>>(
        self,
        local: Result<Option<PartitionCaptureBuffer<u8>>, PartitionCaptureExchangeError>,
        decoder: D,
    ) -> Result<D::Output, PartitionCaptureExchangeError> {
        let transport = self.transport;
        let result = self.exchange_inner(local, decoder);
        if let Err(error) = &result {
            if !matches!(
                error,
                PartitionCaptureExchangeError::LocalRejected { .. }
                    | PartitionCaptureExchangeError::PeerRejected { .. }
            ) {
                transport.fail_capture_exchange(error);
            }
        }
        result
    }

    fn exchange_inner<D: PartitionCaptureDecoder<T>>(
        mut self,
        local: Result<Option<PartitionCaptureBuffer<u8>>, PartitionCaptureExchangeError>,
        decoder: D,
    ) -> Result<D::Output, PartitionCaptureExchangeError> {
        self.transport.ensure_capture_active()?;
        // Validate local bytes without parsing before announcing readiness. Keeping
        // the original error allows a failing rank to return its precise cause.
        let local = local.and_then(|bytes| {
            let producer = self.receipt_plan().producer(self.rank).is_some();
            if producer != bytes.is_some()
                || bytes.as_ref().is_some_and(|bytes| {
                    bytes.is_empty() || bytes.len() as u64 > self.receipt_plan().max_record_bytes()
                })
            {
                Err(
                    CaptureError::Invalid("local capture receipt presence or byte bound".into())
                        .into(),
                )
            } else {
                Ok(bytes)
            }
        });
        let (status, length) = match &local {
            Err(_) => (0, 0),
            Ok(None) => (1, 0),
            Ok(Some(bytes)) => (2, bytes.len() as u64),
        };
        let header = self.header(0, status, length);
        let gathered = self.gather(PartitionCaptureFrameKind::Preparation, &header)?;
        let mut max_length = 0;
        let mut rejected = None;
        for (rank, frame) in gathered.chunks_exact(HEADER_WORDS).enumerate() {
            self.validate_header(frame, rank, 0)?;
            let length = frame[14] as u64 | ((frame[15] as u64) << 32);
            match frame[13] {
                0 if length == 0 => {
                    rejected.get_or_insert(rank);
                }
                1 if length == 0 && self.receipt_plan().producer(rank).is_none() => (),
                2 if length > 0
                    && length <= self.receipt_plan().max_record_bytes()
                    && self.receipt_plan().producer(rank).is_some() =>
                {
                    max_length = max_length.max(length);
                }
                _ => {
                    return Err(PartitionCaptureExchangeError::Protocol(
                        "producer readiness or byte bound",
                    ))
                }
            }
        }
        if let Some(rank) = rejected {
            return Err(self.rejection(
                rank,
                PartitionCaptureExchangeStage::Preparation,
                local.err(),
            ));
        }
        let local = local?;
        let width = word_count(max_length)?;
        let mut payload = self.transport.capture_word_destination(width)?;
        payload.resize(width, 0)?;
        if let Some(bytes) = &local {
            for (word, bytes) in payload.iter_mut().zip(bytes.chunks(4)) {
                let mut padded = [0; 4];
                padded[..bytes.len()].copy_from_slice(bytes);
                *word = u32::from_le_bytes(padded);
            }
        }
        let received = self.gather(PartitionCaptureFrameKind::Payload, &payload)?;
        let receipt = self.receipt.take().expect("one exchange consumes its receipt once");
        let payload = PartitionCapturePayload::new(self.transport, &gathered, &received, width,
            self.rank, local.as_deref());
        let result = decoder.decode(receipt, payload);
        // Credits were reserved before any producer work or parser reservation;
        // even an exhausted ledger can participate in this final decision.
        let verdict = self.header(1, u32::from(result.is_ok()), 0);
        let verdicts = self.gather(PartitionCaptureFrameKind::Delivery, &verdict)?;
        let mut rejected = None;
        for (rank, frame) in verdicts.chunks_exact(HEADER_WORDS).enumerate() {
            self.validate_header(frame, rank, 1)?;
            if frame[14..] != [0, 0] || frame[13] > 1 {
                return Err(PartitionCaptureExchangeError::Protocol("delivery verdict"));
            }
            if frame[13] == 0 {
                rejected.get_or_insert(rank);
            }
        }
        if let Some(rank) = rejected {
            return Err(self.rejection(
                rank,
                PartitionCaptureExchangeStage::Delivery,
                result.err(),
            ));
        }
        result
    }

    fn rejection(
        &self,
        first_rank: usize,
        stage: PartitionCaptureExchangeStage,
        local: Option<PartitionCaptureExchangeError>,
    ) -> PartitionCaptureExchangeError {
        match local {
            Some(source) => PartitionCaptureExchangeError::LocalRejected {
                rank: self.rank,
                stage,
                source: Box::new(source),
            },
            None => PartitionCaptureExchangeError::PeerRejected {
                rank: first_rank,
                stage,
            },
        }
    }

    fn header(&self, phase: u32, status: u32, length: u64) -> [u32; HEADER_WORDS] {
        protocol_header(self.rank, self.participants, &self.digest, phase, status, length)
    }

    fn validate_header(&self, frame: &[u32], rank: usize, phase: u32)
        -> Result<(), PartitionCaptureExchangeError> {
        validate_protocol_header(frame, rank, self.participants, &self.digest, phase)
    }

    fn gather(&self, kind: PartitionCaptureFrameKind, words: &[u32])
        -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError> {
        let maximum = if kind == PartitionCaptureFrameKind::Payload { self.payload_words } else { HEADER_WORDS };
        let frame = PartitionCaptureFrame::new(kind, self.rank, self.participants, words, maximum)?;
        gather_capture_words(self.transport, self.wait, &frame)
    }
}

// One canonical writer/validator for fixed receipt and producer-source frames.
pub(super) fn protocol_header(rank: usize, participants: usize, digest: &[u32; 8],
    phase: u32, status: u32, length: u64) -> [u32; HEADER_WORDS] {
    let mut words = [0; HEADER_WORDS];
    words[..5].copy_from_slice(&[MAGIC, VERSION, phase, rank as u32, participants as u32]);
    words[5..13].copy_from_slice(digest);
    words[13..].copy_from_slice(&[status, length as u32, (length >> 32) as u32]);
    words
}
pub(super) fn validate_protocol_header(frame: &[u32], rank: usize, participants: usize,
    digest: &[u32; 8], phase: u32) -> Result<(), PartitionCaptureExchangeError> {
    if frame.len() != HEADER_WORDS
        || frame[..5] != [MAGIC, VERSION, phase, rank as u32, participants as u32]
        || frame[5..13] != *digest {
        return Err(PartitionCaptureExchangeError::Protocol("receipt identity, phase, or sender"));
    }
    Ok(())
}

pub(super) fn gather_capture_words<T: PartitionCaptureTransport>(
    transport: &T, wait: BoundedCompletionWait, frame: &PartitionCaptureFrame<'_>,
) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError>
where T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    let gathered = transport.gather_capture_frame(frame, wait)?;
    if gathered.len() != frame.gathered_words() {
        return Err(PartitionCaptureExchangeError::Protocol("rank-major gather length"));
    }
    Ok(gathered)
}

fn ordinary_gather_capture_words<T: PartitionCaptureTransport>(
    transport: &T, wait: BoundedCompletionWait, frame: &PartitionCaptureFrame<'_>,
) -> Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError>
where T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    let submission = transport.submit_all_gather_words(frame.words()).map_err(BackendFailure::from_error)?;
    let output = match submission.wait_bounded(wait).map_err(BackendFailure::from_error)? {
        BoundedSubmissionOutcome::Completed(output) => output,
        BoundedSubmissionOutcome::DeadlineExceeded { cancellation } => {
            return Err(PartitionCaptureExchangeError::Deadline { cancellation })
        }
    };
    let gathered = transport.resolve_all_gather_words(output).map_err(BackendFailure::from_error)?;
    Ok(PartitionCaptureBuffer::ordinary(gathered))
}

fn word_count(bytes: u64) -> Result<usize, CaptureError> {
    usize::try_from(add(bytes, 3)? / 4).map_err(|_| CaptureError::Overflow)
}

pub(super) fn gather_usage<T: PartitionCaptureTransport>(
    transport: &T,
    participants: usize,
    words: usize,
) -> Result<CaptureUsage, CaptureError> {
    let gathered = words
        .checked_mul(participants)
        .ok_or(CaptureError::Overflow)?;
    // Vec's allocation bound is in bytes, even for word buffers. Validate it
    // while still cold, including on hosts whose usize is narrower than u64.
    for count in [words, gathered] {
        count
            .checked_mul(4)
            .filter(|bytes| *bytes <= isize::MAX as usize)
            .ok_or(CaptureError::Overflow)?;
    }
    // Input words, gathered host words and byte reconstruction (including slack).
    transport
        .estimate_capture_gather(words)?
        .checked_add(CaptureUsage {
            host_bytes: add(4096, mul(add(words as u64, mul(gathered as u64, 2)?)?, 4)?)?,
            ..Default::default()
        })
}
