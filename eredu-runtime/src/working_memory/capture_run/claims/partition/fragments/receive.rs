//! Contiguous producer envelopes use the ordinary fixed reader once per rank.
use super::*;

#[derive(Debug, thiserror::Error)]
enum ReceiveCause {
    #[error("contiguous receipt identity, shape, charge or fragment count differs")]
    Source,
    #[error(transparent)]
    Destination(#[from] PartitionFragmentDestinationError),
    #[error(transparent)]
    Decode(#[from] PartitionCaptureTensorDecodeError),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Memory(#[from] WorkingMemoryError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Routed(#[from] RoutedReceiveCause),
}
/// Exact reader error plus the original source, metadata and host account. A
/// rejected payload remains in its underlying typed destination failure.
#[derive(Debug, thiserror::Error)]
#[error("contiguous capture receipt: {cause}")]
pub(crate) struct PartitionFragmentReceiveError {
    #[source]
    cause: ReceiveCause,
    _source: SharedCapturePlan,
    _custody: CaptureTensorCustody,
    _metadata: HostMetadataFunding,
}
impl PreparedPartitionFragmentDestinations {
    /// Decode only a real remote producer from the enclosing unique-rank exchange.
    /// Contiguous projection supplies zero or one fragment. For an empty source,
    /// the enclosing retained per-rank source supplies its actual optional dtype;
    /// no two-byte/F32 inference or dummy native fragment is made here.
    pub(crate) fn decode_contiguous_producer(
        &mut self,
        receipt: &PartitionCaptureReceiptPlan,
        producer: usize,
        bytes: &[u8],
        empty_dtype: Option<&TensorDtype>,
        funding: &HostMetadataFunding,
    ) -> Result<(), PartitionFragmentReceiveError> {
        let source = self.source.clone();
        let custody = self.custody.share_scheduled();
        let error = |cause| PartitionFragmentReceiveError {
            cause,
            _source: source.clone(),
            _custody: custody.share_scheduled(),
            _metadata: funding.clone(),
        };

        funding
            .reserve_metadata(
                receive_contiguous_controls().ok_or_else(|| error(ReceiveCause::Source))?,
            )
            .map_err(|e| error(e.into()))?;
        custody.validate().map_err(|e| error(e.into()))?;
        if !self.matches(receipt)
            || producer == self.allowance.local_rank()
            || bytes.len() as u64 > receipt.max_record_bytes()
        {
            return Err(error(ReceiveCause::Source));
        }
        let projection = receipt
            .producer(producer)
            .ok_or_else(|| error(ReceiveCause::Source))?;
        if receipt.routed_producer(producer).is_some() {
            return self.decode_routed_producer(receipt, producer, bytes, empty_dtype, funding);
        }
        if projection.fragments().len() > 1 {
            return Err(error(ReceiveCause::Source));
        }
        if projection.fragments().is_empty() {
            if !matches!(
                empty_dtype,
                None | Some(TensorDtype::F16 | TensorDtype::F32 | TensorDtype::Bf16)
            ) {
                return Err(error(ReceiveCause::Source));
            }
            funding
                .reserve_metadata(
                    TensorReader::construction_control_bytes()
                        .ok_or_else(|| error(ReceiveCause::Source))?,
                )
                .map_err(|e| error(e.into()))?;
            reserve_parser_headroom(bytes, funding).map_err(|e| error(e.into()))?;
            let mut reader = TensorReader::new_empty(
                receipt.context(),
                receipt.identity(),
                producer,
                source.admission(),
                receipt.combination(),
                empty_dtype.cloned(),
            );
            let parsed = json::parse(bytes, |event| reader.event(event));
            parsed.map_err(|e| error(e.into()))?;
            if !reader.complete() {
                return Err(error(ReceiveCause::Source));
            }
            return Ok(());
        }
        // Expected charge is the same ordinary metadata + actual native quote,
        // not a sender-supplied reservation or a recomputed probability table.
        let (dtype, charged, native) = self
            .allowance
            .fragment_record_charge(receipt, producer, 0)
            .ok_or_else(|| error(ReceiveCause::Source))?;
        let destination = self
            .take_received(receipt, producer, 0)
            .map_err(|e| error(e.into()))?;
        let expected = PartitionCaptureTensorReceipt {
            context: receipt.context(),
            identity: receipt.identity(),
            producer,
            dtype: dtype.clone(),
            charged,
        };
        let value = match destination {
            PartitionFragmentDestination::Routed(_) => return Err(error(ReceiveCause::Source)),
            PartitionFragmentDestination::Tensor(claim) => PartitionFragmentValue::Tensor(
                claim
                    .decode_partition_receipt_combined(
                        bytes,
                        expected,
                        funding,
                        receipt.combination(),
                    )
                    .map_err(|e| error(e.into()))?,
            ),
            PartitionFragmentDestination::Summary(claim) => PartitionFragmentValue::Summary(
                claim
                    .decode_partition_receipt(bytes, expected, funding)
                    .map_err(|e| error(e.into()))?,
            ),
            PartitionFragmentDestination::Histogram(claim) => PartitionFragmentValue::Histogram(
                claim
                    .decode_partition_receipt(bytes, expected, funding)
                    .map_err(|e| error(e.into()))?,
            ),
        };
        self.record(receipt, producer, 0, &dtype, native, value)
            .map_err(|e| error(e.into()))
    }
}

mod routed;
pub(in crate::working_memory::capture_run::claims::partition) use routed::{
    RoutedReceiveCause, RoutedReceiver,
};
impl PreparedPartitionFragmentDestinations {
    fn decode_routed_producer(
        &mut self,
        receipt: &PartitionCaptureReceiptPlan,
        producer: usize,
        bytes: &[u8],
        dtype: Option<&TensorDtype>,
        funding: &HostMetadataFunding,
    ) -> Result<(), PartitionFragmentReceiveError> {
        let source = self.source.clone();
        let custody = self.custody.share_scheduled();
        let error = |cause| PartitionFragmentReceiveError {
            cause,
            _source: source.clone(),
            _custody: custody.share_scheduled(),
            _metadata: funding.clone(),
        };

        funding
            .reserve_metadata(receive_routed_controls().ok_or_else(|| error(ReceiveCause::Source))?)
            .map_err(|e| error(e.into()))?;
        reserve_parser_headroom(bytes, funding).map_err(|e| error(e.into()))?;
        let mut destination =
            RoutedReceiver::new(self, receipt, producer, dtype).map_err(|e| error(e.into()))?;
        let mut reader = TensorReader::new_routed(
            &mut destination,
            receipt,
            producer,
            source.admission(),
            dtype.cloned(),
        );
        let parsed = json::parse(bytes, |event| reader.event(event));
        let complete = reader.complete();
        drop(reader);
        if let Some(cause) = destination.take_error() {
            return Err(error(cause.into()));
        }
        parsed.map_err(|e| error(e.into()))?;
        if !complete {
            return Err(error(ReceiveCause::Source));
        }
        Ok(())
    }
}

fn receive_contiguous_controls() -> Option<usize> {
    let parts = [
        size_of::<(
            &mut PreparedPartitionFragmentDestinations,
            &PartitionCaptureReceiptPlan,
            usize,
            &[u8],
            Option<&TensorDtype>,
            &HostMetadataFunding,
        )>(),
        size_of::<SharedCapturePlan>(),
        size_of::<CaptureTensorCustody>(),
        size_of::<PartitionCaptureTensorReceipt<'_>>(),
        size_of::<PartitionFragmentDestination<'_, '_>>(),
        size_of::<PartitionFragmentValue>(),
        size_of::<Option<(TensorDtype, CaptureUsage, CaptureUsage)>>(),
        size_of::<(TensorDtype, CaptureUsage, CaptureUsage)>(),
        size_of::<PartitionFragmentReceiveError>(),
        size_of::<ReceiveCause>(),
        size_of::<Result<(), PartitionFragmentReceiveError>>(),
        size_of::<TensorReader<'_, '_, '_>>(),
        size_of::<Option<&CaptureSlicePartition>>(),
        size_of::<Option<TensorDtype>>(),
        size_of::<bool>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

fn receive_routed_controls() -> Option<usize> {
    let parts = [
        RoutedReceiver::control_bytes()?,
        TensorReader::construction_control_bytes()?,
        size_of::<RoutedReceiver<'_>>(),
        size_of::<TensorReader<'_, '_, '_>>(),
        size_of::<SharedCapturePlan>(),
        size_of::<CaptureTensorCustody>(),
        size_of::<PartitionFragmentReceiveError>(),
        size_of::<Result<(), PartitionFragmentReceiveError>>(),
        size_of::<(
            &mut PreparedPartitionFragmentDestinations,
            &PartitionCaptureReceiptPlan,
            usize,
            &[u8],
            Option<&TensorDtype>,
            &HostMetadataFunding,
        )>(),
    ];
    parts
        .into_iter()
        .try_fold(size_of_val(&parts), usize::checked_add)
}

pub(super) fn receiver_metadata_bytes(
    receipt: &PartitionCaptureReceiptPlan,
    local_rank: usize,
) -> Option<usize> {
    let maximum = usize::try_from(receipt.max_record_bytes()).ok()?;
    let mut bytes = 0usize;
    let selection = &receipt.shared_plan_source().admission().plan().selections
        [receipt.context().selection_index];
    for (producer, projection) in receipt.producers() {
        if producer == local_rank {
            continue;
        }
        bytes = bytes.checked_add(receive_contiguous_controls()?)?;
        let received = if receipt.routed_producer(producer).is_some() {
            receive_routed_controls()?.checked_add(super::super::parser_metadata_bytes(maximum)?)?
        } else if projection.fragments().is_empty() {
            TensorReader::construction_control_bytes()?
                .checked_add(super::super::parser_metadata_bytes(maximum)?)?
        } else {
            // The same additive source lowers its selected transform to a raw
            // tensor before native production and canonical receipt decoding.
            let transform = if receipt.combination() == PartitionCaptureCombination::SumF64ToF32 {
                &CaptureTransform::FullTensor
            } else {
                &selection.transform
            };
            super::super::decode_metadata_bytes(transform, maximum)?
        };
        bytes = bytes.checked_add(received)?;
    }
    Some(bytes)
}
