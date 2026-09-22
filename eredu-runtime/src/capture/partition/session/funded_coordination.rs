//! Original source custody around the existing pre-forward coordination worker.
use super::*;
use eredu_core::{BoundedCompletion, BoundedCompletionWait};
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

use crate::working_memory::PartitionCaptureRankSource;
#[derive(Clone, Copy, PartialEq, Eq)]
enum SourceMode {
    Selected,
    Contiguous,
    Routed,
}
struct SourceVote {
    dtype: Option<TensorDtype>,
    ranks: Option<Vec<PartitionCaptureRankSource>>,
}
pub(crate) struct PartitionCaptureRoutedVote {
    pub(crate) dtype: TensorDtype,
    pub(crate) ranks: Vec<PartitionCaptureRankSource>,
}

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("partition capture coordination source: {0}")]
    Source(&'static str),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Capture(#[from] CaptureError),
    #[error(transparent)]
    Context(#[from] eredu_nn::Error),
    #[error(transparent)]
    Exchange(#[from] PartitionCaptureExchangeError),
}
/// A failed coordination retains the original shared capture source and every
/// spent metadata reservation, including native error custody in its source.
#[derive(Debug, thiserror::Error)]
#[error("original partition capture coordination: {cause}")]
pub struct PartitionCaptureCoordinationError {
    #[source]
    cause: Cause,
    _source: SharedCapturePlan,
    _metadata: HostMetadataFunding,
}

/// One source-bound, move-only pre-forward vote. The enclosing observer adds
/// every scheduled complete-producer receipt in canonical selection order, then
/// compares the final nonrefundable ledger through the ordinary frame worker.
/// This host owner supplies no native transport or execution authority.
#[must_use = "coordinate once before any selected native capture work"]
pub struct PreparedPartitionCaptureCoordination<'a, T: PartitionCaptureTransport> {
    transport: &'a T,
    context: PartitionCaptureContext,
    wait: BoundedCompletionWait,
    rank: usize,
    participants: usize,
    epoch: DistributedCommitEpoch,
    digest: Sha256,
    next: usize,
    included: usize,
    intervention_last: Option<usize>,
    reserved: CaptureUsage,
    source: SharedCapturePlan,
    metadata: HostMetadataFunding,
}
impl<T: PartitionCaptureTransport> std::fmt::Debug for PreparedPartitionCaptureCoordination<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedPartitionCaptureCoordination")
            .field("epoch", &self.epoch)
            .field("included", &self.included)
            .finish_non_exhaustive()
    }
}
impl<'a, T: PartitionCaptureTransport> PreparedPartitionCaptureCoordination<'a, T>
where
    T::Error: Send + Sync + 'static,
    <T::Completion as Completion>::Error: Send + Sync + 'static,
{
    /// Metadata debited by this exact constructor and its paid context copy.
    /// Transport callbacks and logical capture quota remain separate sources.
    pub fn prepare_metadata_bytes(context: &PartitionCaptureContext) -> Option<usize> {
        Self::prepare_metadata_bytes_with_run_length(context, context.run_identity.len())
    }
    /// The same constructor with a bounded future epoch label. The existing
    /// context supplies every other label and validates the minimum run length.
    pub fn prepare_metadata_bytes_with_run_length(
        context: &PartitionCaptureContext,
        run_length: usize,
    ) -> Option<usize> {
        if run_length < context.run_identity.len() {
            return None;
        }
        Self::prepare_metadata_bytes_for_labels(
            context.artifact_identity.len(),
            context.execution_identity.len(),
            run_length,
            context.overlay_identity.as_ref().map(String::len),
            context.capture_plan_identity.len(),
        )
    }
    /// Constructor and exact UTF-8 label-copy controls from the retained run
    /// source, including a frame with no selected receipt. Lengths describe
    /// storage only and supply no context or execution identity.
    pub fn prepare_metadata_bytes_for_labels(
        artifact_length: usize,
        execution_length: usize,
        run_length: usize,
        overlay_length: Option<usize>,
        capture_identity_length: usize,
    ) -> Option<usize> {
        [
            artifact_length,
            execution_length,
            run_length,
            capture_identity_length,
        ]
        .into_iter()
        .chain(overlay_length)
        .try_fold(Self::prepare_control_bytes()?, |bytes, length| {
            bytes.checked_add(eredu_nn::workspace::WorkspaceContext::metadata_string_bytes(length)?)
        })
    }
    /// One selected producer's Source vote, including a skipped producer row.
    /// Native exchange buffers and callback metadata are quoted by the transport.
    pub fn selected_source_metadata_bytes() -> Option<usize> {
        Self::source_control_bytes()
    }
    /// One contiguous-source vote through its actual wrapper and common worker.
    pub fn contiguous_source_metadata_bytes() -> Option<usize> {
        Self::contiguous_control_bytes()?.checked_add(Self::source_control_bytes()?)
    }
    /// One routed-source vote, including the exact retained producer-rank table.
    pub fn routed_source_metadata_bytes(receipt: &PartitionCaptureReceiptPlan) -> Option<usize> {
        Self::routed_control_bytes()?
            .checked_add(Self::source_control_bytes()?)?
            .checked_add(eredu_nn::workspace::WorkspaceContext::metadata_vec_bytes::<
                PartitionCaptureRankSource,
            >(receipt.producers().count())?)
    }
    fn prepare_control_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>() * 2,
            size_of::<PartitionCaptureContext>(),
            size_of::<Result<PartitionCaptureContext, eredu_nn::Error>>(),
            size_of::<Result<Self, PartitionCaptureCoordinationError>>(),
            size_of::<PartitionCaptureCoordinationError>(),
            size_of::<Cause>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, CaptureSelection>>>(),
            size_of::<HashWriter<'_>>(),
            size_of::<serde_json::Serializer<&mut HashWriter<'_>>>(),
            crate::capture::RECORD_ENCODING_CONTROL_BYTES,
            size_of::<(
                &T,
                &SharedCapturePlan,
                &PartitionCaptureContext,
                &HostMetadataFunding,
                &mut dyn CaptureReservation,
            )>(),
            size_of::<[&String; 4]>(),
            size_of::<[u8; 8]>(),
            size_of::<CaptureUsage>() * 3,
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    /// Final coordination debit, excluding the transport's exchange callbacks.
    pub fn coordinate_metadata_bytes() -> Option<usize> {
        let parts = [
            size_of::<Self>(),
            size_of::<Result<(), PartitionCaptureExchangeError>>(),
            size_of::<Result<(), PartitionCaptureCoordinationError>>(),
            size_of::<PartitionCaptureCoordinationError>(),
            size_of::<[u32; 16]>(),
            size_of::<[u8; 32]>(),
            size_of::<PartitionCaptureFrame<'_>>(),
            size_of::<PartitionCaptureBuffer<u32>>(),
            size_of::<(
                &T,
                BoundedCompletionWait,
                usize,
                usize,
                DistributedCommitEpoch,
                &[u8; 32],
            )>(),
            size_of::<std::slice::ChunksExact<'_, u32>>() * 2,
            size_of::<std::slice::ChunksExact<'_, u8>>(),
            size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<Sha256>(),
            size_of::<(&CaptureLedger, usize, CaptureUsage)>(),
        ];
        parts
            .into_iter()
            .try_fold(size_of_val(&parts), usize::checked_add)
    }

    fn source_control_bytes() -> Option<usize> {
        let producers =
            None::<&PartitionCaptureReceiptPlan>.map(PartitionCaptureReceiptPlan::producers);
        let controls = [
            size_of::<SourceVote>(),
            size_of::<Result<SourceVote, PartitionCaptureCoordinationError>>(),
            size_of::<Result<SourceVote, PartitionCaptureExchangeError>>(),
            size_of::<SourceMode>(),
            size_of::<Option<Vec<PartitionCaptureRankSource>>>(),
            size_of::<PartitionCaptureRankSource>(),
            size_of_val(&producers) * 2,
            size_of::<(bool, usize)>(),
            size_of::<Sha256>(),
            size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<[u32; 8]>(),
            size_of::<[u32; 16]>() * 2,
            size_of::<PartitionCaptureFrame<'_>>(),
            size_of::<PartitionCaptureBuffer<u32>>(),
            size_of::<Result<PartitionCaptureBuffer<u32>, PartitionCaptureExchangeError>>(),
            size_of::<Result<Option<TensorDtype>, PartitionCaptureCoordinationError>>(),
            size_of::<PartitionCaptureCoordinationError>(),
            size_of::<Cause>(),
            size_of::<HashWriter<'_>>(),
            size_of::<serde_json::Serializer<&mut HashWriter<'_>>>(),
            crate::capture::RECORD_ENCODING_CONTROL_BYTES,
            size_of::<std::iter::Enumerate<std::slice::ChunksExact<'_, u32>>>(),
            size_of::<std::slice::ChunksExact<'_, u8>>(),
            size_of::<Option<TensorDtype>>() * 3,
            size_of::<(
                &Self,
                usize,
                Option<&PartitionCaptureReceiptPlan>,
                usize,
                Option<&TensorDtype>,
                Option<&CaptureSkipReason>,
                CaptureUsage,
                &CaptureLedger,
                SourceMode,
                bool,
            )>(),
        ];
        controls
            .into_iter()
            .try_fold(size_of_val(&controls), usize::checked_add)
    }

    /// Debit for including one successful receipt in the common transcript.
    pub fn include_metadata_bytes() -> Option<usize> {
        Some(
            size_of::<(
                &Self,
                &PartitionCaptureReceiptPlan,
                TensorDtype,
                CaptureUsage,
            )>() + size_of::<Result<(), PartitionCaptureCoordinationError>>()
                + size_of::<PartitionCaptureCoordinationError>()
                + size_of::<[u8; 8]>(),
        )
    }

    /// Debit for including one permitted limit skip in the common transcript.
    pub fn skip_metadata_bytes() -> Option<usize> {
        Some(
            size_of::<(&Self, usize, &CaptureSkipReason)>()
                + size_of::<PartitionCaptureCoordinationError>()
                + size_of::<Result<(), PartitionCaptureCoordinationError>>()
                + size_of::<HashWriter<'_>>()
                + size_of::<serde_json::Serializer<&mut HashWriter<'_>>>()
                + crate::capture::RECORD_ENCODING_CONTROL_BYTES,
        )
    }

    fn contiguous_control_bytes() -> Option<usize> {
        Some(
            size_of::<(
                &Self,
                usize,
                &PartitionCaptureReceiptPlan,
                Option<&TensorDtype>,
                CaptureUsage,
                &CaptureLedger,
            )>() + size_of::<Result<TensorDtype, PartitionCaptureCoordinationError>>(),
        )
    }

    fn routed_control_bytes() -> Option<usize> {
        Some(
            size_of::<PartitionCaptureRoutedVote>()
                + size_of::<(
                    &Self,
                    usize,
                    &PartitionCaptureReceiptPlan,
                    bool,
                    Option<&TensorDtype>,
                    CaptureUsage,
                    &CaptureLedger,
                )>(),
        )
    }

    /// Debit for including one actual intervention in the existing transcript.
    pub fn intervention_metadata_bytes() -> Option<usize> {
        Some(
            size_of::<(&mut Self, &PreparedPartitionIntervention<'_, T>)>()
                + size_of::<Result<(), PartitionCaptureCoordinationError>>(),
        )
    }

    /// Reserve the same fixed coordination exchange before optional producer
    /// work. Context strings are copied through the shared paid UTF-8 producer; the source
    /// labels alone do not authenticate a loaded native world or request.
    pub fn prepare(
        transport: &'a T,
        source: &SharedCapturePlan,
        context: &PartitionCaptureContext,
        metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureCoordinationError> {
        let error = |cause| PartitionCaptureCoordinationError {
            cause,
            _source: source.clone(),
            _metadata: metadata.clone(),
        };
        let controls = Self::prepare_control_bytes()
            .ok_or_else(|| error(Cause::Source("constructor controls overflow")))?;
        metadata
            .reserve_metadata(controls)
            .map_err(|cause| error(cause.into()))?;
        let participants = transport.participant_count();
        let rank = transport.capture_rank();
        let epoch = DistributedCommitEpoch::new(context.forward_epoch)
            .ok_or_else(|| error(Cause::Source("missing forward epoch")))?;
        if participants == 0
            || rank >= participants
            || u32::try_from(participants).is_err()
            || context.capture_plan_identity != source.admission().identity()
            || context.prediction >= source.admission().request().max_predictions
            || [
                &context.artifact_identity,
                &context.execution_identity,
                &context.run_identity,
                &context.capture_plan_identity,
            ]
            .into_iter()
            .chain(context.overlay_identity.iter())
            .any(|value| value.is_empty() || value.len() > 256)
        {
            return Err(error(Cause::Source(
                "context or participant source differs",
            )));
        }
        source
            .admission()
            .geometry_at(context.phase, context.prediction, context.invocation)
            .map_err(|cause| error(cause.into()))?;
        transport
            .ensure_capture_active()
            .map_err(PartitionCaptureExchangeError::from)
            .map_err(|cause| error(cause.into()))?;
        let wait = transport
            .capture_wait()
            .map_err(|cause| error(cause.into()))?;
        if !T::Completion::supports_cancellation(wait.cancellation()) {
            return Err(error(Cause::Source(
                "completion cannot apply the selected disposition",
            )));
        }
        // Preserve the ordinary global quota equation. Physical frame, output
        // and native storage are paid separately by their actual constructors.
        let scheduled = source
            .admission()
            .plan()
            .selections
            .iter()
            .filter(|selection| {
                selection
                    .schedule
                    .includes(context.phase, context.prediction)
            })
            .count();
        let source_votes = super::super::exchange::gather_usage(transport, participants, 16)
            .and_then(|usage| usage.checked_mul(scheduled as u64))
            .map_err(|cause| error(cause.into()))?;
        let reserved = super::super::exchange::gather_usage(transport, participants, 16)
            .and_then(|usage| {
                usage.checked_add(CaptureUsage {
                    host_bytes: 8192,
                    ..Default::default()
                })
            })
            .and_then(|usage| usage.checked_add(source_votes))
            .and_then(|usage| usage.checked_mul(participants as u64))
            .map_err(|cause| error(cause.into()))?;
        ledger
            .reserve_quota(reserved)
            .map_err(|cause| error(cause.into()))?;
        let mut digest = Sha256::new();
        digest.update(b"eredu.funded.partition.capture.coordination.v1\0");
        // The same canonical Context serializer feeds the digest directly.
        // Every field is a closed borrowed string/scalar; no JSON Vec is built.
        serde_json::to_writer(&mut HashWriter(&mut digest), context)
            .expect("closed context and infallible digest writer");
        let context = super::super::receipt::copy_context(context, metadata)
            .map_err(|cause| error(cause.into()))?;
        Ok(Self {
            transport,
            context,
            wait,
            rank,
            participants,
            epoch,
            digest,
            next: 0,
            included: 0,
            intervention_last: None,
            reserved,
            source: source.clone(),
            metadata: metadata.clone(),
        })
    }

    /// Resolve the actual selected producer's scalar after all ranks prepared
    /// their finite destinations. Absent local witnesses remain absent until
    /// this authenticated rank-major exchange supplies the producer's fact.
    /// Every scheduled row, including a skipped row, follows this same frame.
    pub(crate) fn coordinate_source(
        &self,
        index: usize,
        receipt: Option<&PartitionCaptureReceiptPlan>,
        producer: usize,
        local: Option<&TensorDtype>,
        skipped: Option<&CaptureSkipReason>,
        usage: CaptureUsage,
        ledger: &CaptureLedger,
    ) -> Result<Option<TensorDtype>, PartitionCaptureCoordinationError> {
        self.coordinate_source_worker(
            index,
            receipt,
            producer,
            local,
            skipped,
            usage,
            ledger,
            SourceMode::Selected,
            false,
        )
        .map(|vote| vote.dtype)
    }

    /// The same single Source frame validates all actual nonempty contiguous
    /// producers. Empty projections need no invented scalar, while an advertised
    /// empty/replica witness must still agree with the actual common precision.
    pub(crate) fn coordinate_contiguous_source(
        &self,
        index: usize,
        receipt: &PartitionCaptureReceiptPlan,
        local: Option<&TensorDtype>,
        usage: CaptureUsage,
        ledger: &CaptureLedger,
    ) -> Result<TensorDtype, PartitionCaptureCoordinationError> {
        self.metadata
            .reserve_metadata(
                Self::contiguous_control_bytes()
                    .ok_or_else(|| self.error(Cause::Source("source wrapper controls overflow")))?,
            )
            .map_err(|cause| self.error(cause.into()))?;
        if !receipt.shared_plan_source().same_storage(&self.source)
            || !same_forward(&self.context, receipt.context())
            || receipt.context().selection_index != index
            || receipt.world_size() != self.participants
            || receipt.producers().any(|(rank, p)| {
                rank >= self.participants
                    || p.fragments().len() > 1
                    || receipt.routed_producer(rank).is_some()
            })
        {
            return Err(self.error(Cause::Source("contiguous receipt source differs")));
        }
        let producer = receipt
            .producers()
            .find(|(_, p)| !p.fragments().is_empty())
            .or_else(|| receipt.producers().next())
            .map(|(rank, _)| rank)
            .ok_or_else(|| {
                self.error(Cause::Source("contiguous receipt has no actual producers"))
            })?;
        self.coordinate_source_worker(
            index,
            Some(receipt),
            producer,
            local,
            None,
            usage,
            ledger,
            SourceMode::Contiguous,
            false,
        )?
        .dtype
        .ok_or_else(|| self.error(Cause::Source("contiguous source has no scalar witness")))
    }

    /// Sparse source presence remains distinct from a routed-result witness.
    /// An idle producer acknowledges its original ownership without fabricating
    /// a dtype; real result witnesses establish the common precision.
    pub(crate) fn coordinate_routed_source(
        &self,
        index: usize,
        receipt: &PartitionCaptureReceiptPlan,
        present: bool,
        local: Option<&TensorDtype>,
        usage: CaptureUsage,
        ledger: &CaptureLedger,
    ) -> Result<PartitionCaptureRoutedVote, PartitionCaptureCoordinationError> {
        self.metadata
            .reserve_metadata(
                Self::routed_control_bytes()
                    .ok_or_else(|| self.error(Cause::Source("source wrapper controls overflow")))?,
            )
            .map_err(|e| self.error(e.into()))?;
        if !receipt.shared_plan_source().same_storage(&self.source)
            || !same_forward(&self.context, receipt.context())
            || receipt.context().selection_index != index
            || receipt.world_size() != self.participants
            || receipt.producers().any(|(rank, _)| {
                rank >= self.participants || receipt.routed_producer(rank).is_none()
            })
            || local.is_some() && !present
            || receipt.producer(self.rank).is_some() && !present
        {
            return Err(self.error(Cause::Source("sparse receipt source differs")));
        }
        let producer = receipt
            .producers()
            .next()
            .map(|(rank, _)| rank)
            .ok_or_else(|| self.error(Cause::Source("sparse receipt has no actual producers")))?;
        let vote = self.coordinate_source_worker(
            index,
            Some(receipt),
            producer,
            local,
            None,
            usage,
            ledger,
            SourceMode::Routed,
            present,
        )?;
        Ok(PartitionCaptureRoutedVote {
            dtype: vote.dtype.ok_or_else(|| {
                self.error(Cause::Source("sparse source has no result scalar witness"))
            })?,
            ranks: vote
                .ranks
                .ok_or_else(|| self.error(Cause::Source("sparse source has no witness table")))?,
        })
    }

    fn coordinate_source_worker(
        &self,
        index: usize,
        receipt: Option<&PartitionCaptureReceiptPlan>,
        producer: usize,
        local: Option<&TensorDtype>,
        skipped: Option<&CaptureSkipReason>,
        usage: CaptureUsage,
        ledger: &CaptureLedger,
        mode: SourceMode,
        present: bool,
    ) -> Result<SourceVote, PartitionCaptureCoordinationError> {
        let bytes = Self::source_control_bytes()
            .ok_or_else(|| self.error(Cause::Source("source frame controls overflow")))?;
        self.metadata
            .reserve_metadata(bytes)
            .map_err(|cause| self.error(cause.into()))?;
        if self.next_scheduled() != Some(index)
            || producer >= self.participants
            || receipt.is_some() == skipped.is_some()
            || !matches!(
                local,
                None | Some(TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16)
            )
        {
            return Err(self.error(Cause::Source("source row or scalar declaration differs")));
        }
        let mut digest = self.digest.clone();
        digest.update(b"actual producer scalar source\0");
        match mode {
            SourceMode::Selected => {}
            SourceMode::Contiguous => digest.update(b"all contiguous fragment producers\0"),
            SourceMode::Routed => digest.update(b"actual sparse result witnesses\0"),
        };
        digest.update((index as u64).to_le_bytes());
        digest.update((producer as u64).to_le_bytes());
        if let Some(receipt) = receipt {
            digest.update(receipt.identity().as_bytes());
        }
        if let Some(reason) = skipped {
            serde_json::to_writer(&mut HashWriter(&mut digest), reason)
                .expect("closed skip and infallible digest writer");
        }
        hash_usage(&mut digest, usage);
        hash_usage(&mut digest, ledger.step());
        hash_usage(&mut digest, ledger.total());
        let digest = digest.finalize();
        let mut words_digest = [0; 8];
        for (out, bytes) in words_digest.iter_mut().zip(digest.chunks_exact(4)) {
            *out = u32::from_le_bytes(bytes.try_into().expect("four-byte digest chunk"));
        }
        let scalar = if skipped.is_some() {
            0
        } else {
            match local {
                None => 0,
                Some(TensorDtype::F16) => 1,
                Some(TensorDtype::F32) => 2,
                Some(TensorDtype::Bf16) => 3,
                _ => unreachable!("validated scalar"),
            }
        };
        let mut routed_ranks = if mode == SourceMode::Routed {
            Some(
                self.metadata
                    .metadata_vec(
                        receipt
                            .expect("routed source has receipt")
                            .producers()
                            .count(),
                    )
                    .map_err(|e| self.error(e.into()))?,
            )
        } else {
            None
        };
        let result = (|| -> Result<SourceVote, PartitionCaptureExchangeError> {
            let words = super::super::exchange::protocol_header(
                self.rank,
                self.participants,
                &words_digest,
                3,
                scalar
                    | if mode == SourceMode::Routed && present {
                        8
                    } else {
                        0
                    },
                index as u64,
            );
            let frame = PartitionCaptureFrame::new(
                PartitionCaptureFrameKind::Source,
                self.rank,
                self.participants,
                &words,
                16,
            )?;
            let gathered =
                super::super::exchange::gather_capture_words(self.transport, self.wait, &frame)?;
            for (rank, header) in gathered.chunks_exact(16).enumerate() {
                super::super::exchange::validate_protocol_header(
                    header,
                    rank,
                    self.participants,
                    &words_digest,
                    3,
                )?;
                let source = header[13];
                let valid = if mode == SourceMode::Routed {
                    source & !11 == 0
                        && (source & 3 == 0 || source & 8 != 0)
                        && (receipt.is_none_or(|receipt| receipt.producer(rank).is_none())
                            || source & 8 != 0)
                } else {
                    source <= 3
                };
                if !valid || header[14] != index as u32 || header[15] != (index as u64 >> 32) as u32
                {
                    return Err(PartitionCaptureExchangeError::Protocol(
                        "source scalar or selection differs",
                    ));
                }
            }
            let actual = if mode == SourceMode::Routed {
                gathered
                    .chunks_exact(16)
                    .map(|header| header[13] & 3)
                    .find(|scalar| *scalar != 0)
                    .unwrap_or(0)
            } else {
                gathered[producer * 16 + 13]
            };
            if skipped.is_some() {
                if gathered.chunks_exact(16).any(|header| header[13] != 0) {
                    return Err(PartitionCaptureExchangeError::Protocol(
                        "skipped source has an active scalar",
                    ));
                }
                return Ok(SourceVote {
                    dtype: None,
                    ranks: None,
                });
            }
            if mode == SourceMode::Contiguous
                && receipt.is_none_or(|receipt| {
                    receipt.producers().any(|(rank, projection)| {
                        !projection.fragments().is_empty() && gathered[rank * 16 + 13] == 0
                    })
                })
            {
                return Err(PartitionCaptureExchangeError::Protocol(
                    "a nonempty fragment producer has no scalar witness",
                ));
            }
            if actual == 0
                || gathered.chunks_exact(16).any(|header| {
                    let scalar = header[13] & 3;
                    scalar != 0 && scalar != actual
                })
            {
                return Err(PartitionCaptureExchangeError::Protocol(
                    "selected producer scalar is missing or inconsistent",
                ));
            }
            let dtype = |scalar| match scalar {
                1 => Some(TensorDtype::F16),
                2 => Some(TensorDtype::F32),
                3 => Some(TensorDtype::Bf16),
                _ => None,
            };
            if let Some(ranks) = &mut routed_ranks {
                for (producer, _) in receipt.expect("routed source receipt").producers() {
                    ranks.push(PartitionCaptureRankSource {
                        producer,
                        dtype: dtype(gathered[producer * 16 + 13] & 3),
                    });
                }
            }
            Ok(SourceVote {
                dtype: dtype(actual),
                ranks: routed_ranks.take(),
            })
        })();
        result.map_err(|cause| {
            self.transport.fail_capture_exchange(&cause);
            self.error(cause.into())
        })
    }

    /// Add exactly the next scheduled receipt and its source-derived floating
    /// transform cost. These are logical facts only; the backend must separately
    /// match dtype/cost against the actual source before performing a transform.
    pub fn include(
        &mut self,
        receipt: &PartitionCaptureReceiptPlan,
        dtype: TensorDtype,
        usage: CaptureUsage,
    ) -> Result<(), PartitionCaptureCoordinationError> {
        let controls = Self::include_metadata_bytes()
            .ok_or_else(|| self.error(Cause::Source("row controls overflow")))?;
        self.metadata
            .reserve_metadata(controls)
            .map_err(|cause| self.error(cause.into()))?;
        let index = self.next_scheduled();
        let actual = receipt.context();
        if index != Some(actual.selection_index)
            || receipt.world_size() != self.participants
            || !receipt.shared_plan_source().same_storage(&self.source)
            || !same_forward(&self.context, actual)
            || !matches!(
                dtype,
                TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16
            )
        {
            return Err(self.error(Cause::Source("receipt order, source or forward differs")));
        }
        self.digest
            .update((actual.selection_index as u64).to_le_bytes());
        self.digest.update(receipt.identity().as_bytes());
        self.digest.update([match dtype {
            TensorDtype::F16 => 0,
            TensorDtype::F32 => 1,
            TensorDtype::Bf16 => 2,
            _ => unreachable!("checked scalar format"),
        }]);
        hash_usage(&mut self.digest, usage);
        self.next = actual
            .selection_index
            .checked_add(1)
            .ok_or_else(|| self.error(Cause::Source("selection cursor overflow")))?;
        self.included = self
            .included
            .checked_add(1)
            .ok_or_else(|| self.error(Cause::Source("receipt count overflow")))?;
        Ok(())
    }

    pub(crate) fn include_skip(
        &mut self,
        index: usize,
        reason: &CaptureSkipReason,
    ) -> Result<(), PartitionCaptureCoordinationError> {
        let controls = Self::skip_metadata_bytes()
            .ok_or_else(|| self.error(Cause::Source("row controls overflow")))?;
        self.metadata
            .reserve_metadata(controls)
            .map_err(|cause| self.error(cause.into()))?;
        if self.next_scheduled() != Some(index)
            || !matches!(reason, CaptureSkipReason::Limit { .. })
        {
            return Err(self.error(Cause::Source("skipped receipt order or reason differs")));
        }
        self.digest.update((index as u64).to_le_bytes());
        self.digest.update(b"skipped receipt\0");
        serde_json::to_writer(&mut HashWriter(&mut self.digest), reason)
            .expect("closed skip and infallible digest writer");
        self.next = index
            .checked_add(1)
            .ok_or_else(|| self.error(Cause::Source("selection cursor overflow")))?;
        self.included = self
            .included
            .checked_add(1)
            .ok_or_else(|| self.error(Cause::Source("receipt count overflow")))?;
        Ok(())
    }

    /// Original operation transcripts follow the scheduled capture rows in the
    /// same pre-forward digest. This introduces no additional coordination vote.
    pub(crate) fn include_intervention(
        &mut self,
        work: &PreparedPartitionIntervention<'_, T>,
    ) -> Result<(), PartitionCaptureCoordinationError> {
        self.metadata
            .reserve_metadata(
                Self::intervention_metadata_bytes()
                    .ok_or_else(|| self.error(Cause::Source("source wrapper controls overflow")))?,
            )
            .map_err(|cause| self.error(cause.into()))?;
        if self.next_scheduled().is_some()
            || work.epoch() != self.epoch
            || work.coordinate() != (self.context.phase, self.context.prediction)
            || work.original().plan().admission().request() != self.source.admission().request()
            || self
                .intervention_last
                .is_some_and(|last| last >= work.operation())
        {
            return Err(self.error(Cause::Source(
                "original intervention transcript order or frame differs",
            )));
        }
        self.digest.update(b"original intervention\0");
        self.digest.update((work.operation() as u64).to_le_bytes());
        self.digest.update(work.descriptor());
        hash_usage(&mut self.digest, work.global_reserved());
        self.intervention_last = Some(work.operation());
        Ok(())
    }

    /// Actual global logical reservation, never refundable or transferable.
    pub const fn global_reserved(&self) -> CaptureUsage {
        self.reserved
    }

    /// Consume the one vote after every local destination has been prepared.
    /// The shared frame worker compares the same transcript on every rank. A
    /// completed disagreement stays distinct from unsafe transport failure.
    pub fn coordinate(
        self,
        ledger: &CaptureLedger,
    ) -> Result<(), PartitionCaptureCoordinationError> {
        let controls = Self::coordinate_metadata_bytes()
            .ok_or_else(|| self.error(Cause::Source("completion controls overflow")))?;
        self.metadata
            .reserve_metadata(controls)
            .map_err(|cause| self.error(cause.into()))?;
        if self.next_scheduled().is_some() {
            return Err(self.error(Cause::Source("scheduled receipt is missing")));
        }
        // A local copy is a fixed digest state, never an allocation or a second
        // vote authority. Keep the whole source owner for failure construction.
        let mut digest = self.digest.clone();
        digest.update((self.included as u64).to_le_bytes());
        hash_usage(&mut digest, ledger.step());
        hash_usage(&mut digest, ledger.total());
        let digest = digest.finalize().into();
        super::coordination::exchange_coordination(
            self.transport,
            self.wait,
            self.rank,
            self.participants,
            self.epoch,
            &digest,
        )
        .map_err(|cause| {
            if !matches!(
                cause,
                PartitionCaptureExchangeError::PeerRejected {
                    stage: PartitionCaptureExchangeStage::Coordination,
                    ..
                }
            ) {
                self.transport.fail_capture_exchange(&cause);
            }
            self.error(cause.into())
        })
    }

    fn next_scheduled(&self) -> Option<usize> {
        self.source
            .admission()
            .plan()
            .selections
            .iter()
            .enumerate()
            .skip(self.next)
            .find_map(|(index, selection)| {
                selection
                    .schedule
                    .includes(self.context.phase, self.context.prediction)
                    .then_some(index)
            })
    }
    fn error(&self, cause: Cause) -> PartitionCaptureCoordinationError {
        PartitionCaptureCoordinationError {
            cause,
            _source: self.source.clone(),
            _metadata: self.metadata.clone(),
        }
    }
}

fn same_forward(a: &PartitionCaptureContext, b: &PartitionCaptureContext) -> bool {
    a.artifact_identity == b.artifact_identity
        && a.execution_identity == b.execution_identity
        && a.run_identity == b.run_identity
        && a.overlay_identity == b.overlay_identity
        && a.capture_plan_identity == b.capture_plan_identity
        && a.phase == b.phase
        && a.prediction == b.prediction
        && a.forward_epoch == b.forward_epoch
        && a.invocation == b.invocation
        && a.invocation_window == b.invocation_window
}
pub(super) struct HashWriter<'a>(pub(super) &'a mut Sha256);
impl std::io::Write for HashWriter<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
