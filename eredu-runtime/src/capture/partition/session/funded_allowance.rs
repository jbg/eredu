//! The ordinary global receipt equation, lent as one local original allowance.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("partition capture allowance source: {0}")]
    Source(&'static str),
    #[error(transparent)]
    Funding(#[from] HostMetadataFundingError),
    #[error(transparent)]
    Capture(#[from] CaptureError),
}
/// Refusal retains the exact admission and all spent metadata charges. Logical
/// capture credits already consumed by this attempt are never returned.
#[derive(Debug, thiserror::Error)]
#[error("original partition capture allowance: {cause}")]
pub struct PartitionCaptureAllowanceError {
    #[source]
    cause: Cause,
    _source: SharedCapturePlan,
    _metadata: HostMetadataFunding,
}

/// Local move-only credits cut from the same complete global receipt equation
/// used by ordinary partition capture. This is logical capture allowance only;
/// the original frame, metadata and native transport keep their own accounts.
#[must_use = "lend the allowance to this receipt's transform, exchange and evidence"]
#[derive(Debug)]
pub struct PreparedPartitionCaptureAllowance {
    quota: CaptureQuota,
    global: CaptureUsage,
    descriptor: [u8; 32],
    selection_index: usize,
    local_rank: usize,
    producer: usize,
    estimate: PartitionCaptureNativeEstimate,
    remote_attempted: bool,
    source: SharedCapturePlan,
    metadata: HostMetadataFunding,
}
impl PreparedPartitionCaptureAllowance {
    /// Reserve the existing common exchange/delivery/evidence and producer
    /// transformation costs. Receipt construction has its separate preparation
    /// reservation. An exact native source must recheck `estimate` before use;
    /// this structured diagnostic does not certify a native operation.
    pub fn prepare<T: PartitionCaptureTransport>(
        transport: &T, receipt: &mut PartitionCaptureReceiptPlan,
        estimate: PartitionCaptureNativeEstimate, metadata: &HostMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureAllowanceError>
    where T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        let source = receipt.shared_plan_source().clone();
        let error = |cause| PartitionCaptureAllowanceError {
            cause, _source: source.clone(), _metadata: metadata.clone(),
        };
        let parts = [
            size_of::<Self>() * 2, size_of::<Cause>(),
            PartitionCaptureReceiptPlan::record_bound_control_bytes()
                .ok_or_else(|| error(Cause::Source("receipt bound controls overflow")))?,
            size_of::<Result<Self, PartitionCaptureAllowanceError>>(),
            size_of::<PartitionCaptureAllowanceError>(),
            size_of::<ReceiptWorkCosts>() * 2, size_of::<CaptureQuota>() * 2,
            size_of::<PreparedPartitionRemoteCharge<'_>>() * 2,
            size_of::<Result<PreparedPartitionRemoteCharge<'_>, PartitionCaptureAllowanceError>>(),
            size_of::<CaptureUsage>() * 8, size_of::<PartitionCaptureNativeEstimate>() * 2,
            size_of::<Sha256>(), size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<(&T, &PartitionCaptureReceiptPlan, &HostMetadataFunding,
                &mut dyn CaptureReservation, &SharedCapturePlan)>(),
            size_of::<Option<(usize, &CaptureSlicePartition)>>(),
            size_of::<std::slice::Iter<'_, CaptureFragmentGeometry>>(),
            size_of::<(usize, usize, usize, [u8; 8])>(),
        ];
        let controls = parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| error(Cause::Source("constructor controls overflow")))?;
        metadata.reserve_metadata(controls).map_err(|cause| error(cause.into()))?;
        let retained = &source;
        let mut producers = receipt.producers();
        let (producer, projection) = producers.next().ok_or_else(|| error(Cause::Source("receipt has no producer")))?;
        if producers.next().is_some() || receipt.combination() != PartitionCaptureCombination::Disjoint
            || projection.fragments().len() != 1 || projection.local_shape() != projection.global_shape()
            || receipt.context().invocation.is_some() || receipt.routed_producer(producer).is_some()
            || receipt.context().capture_plan_identity != retained.admission().identity()
            || transport.participant_count() != receipt.world_size()
            || transport.capture_rank() >= receipt.world_size()
        { return Err(error(Cause::Source("receipt is not the selected complete producer"))); }
        drop(producers);
        // The private shared worker has no arbitrary callback in this closed
        // route: the one fragment consumes the same retained scalar estimate.
        let costs = receipt_work_costs(transport, receipt, |_, _, _| Ok(estimate),
            |_| Err(invalid("complete producer unexpectedly requires sparse metadata")))
            .map_err(|cause| error(cause.into()))?;
        let quota = ledger.reserve_quota(costs.global)
            .and_then(|mut global| global.reserve_quota(costs.local))
            .map_err(|cause| error(cause.into()))?;
        Ok(Self { quota, global: costs.global, descriptor: costs.descriptor,
            selection_index: receipt.context().selection_index,
            local_rank: transport.capture_rank(), producer, estimate, remote_attempted: false,
            source: retained.clone(), metadata: metadata.clone() })
    }
    /// Borrow the one local allowance. Its sub-reservations never touch the
    /// parent global ledger again and cannot reset, clone or refund credits.
    pub fn quota_mut(&mut self) -> &mut CaptureQuota { &mut self.quota }
    /// The full nonrefundable charge made for this receipt's work on all ranks.
    pub const fn global_reserved(&self) -> CaptureUsage { self.global }
    /// Canonical ordinary receipt/cost descriptor for the preparation vote.
    pub const fn descriptor(&self) -> &[u8; 32] { &self.descriptor }
    /// Original shared admission; source equality is physical, not label-only.
    pub fn source(&self) -> &SharedCapturePlan { &self.source }
    /// Metadata custody remains attached until the local allowance retires.
    pub fn metadata(&self) -> &HostMetadataFunding { &self.metadata }
}

/// A one-use acknowledgment of an already globally charged remote producer.
/// This private value cannot issue local/native credits or allocate frame H.
#[derive(Debug)]
pub(crate) struct PreparedPartitionRemoteCharge<'a> {
    source: &'a SharedCapturePlan,
    index: usize,
    usage: CaptureUsage,
    _exclusive: std::marker::PhantomData<&'a mut PreparedPartitionCaptureAllowance>,
}
impl PreparedPartitionCaptureAllowance {
    pub(crate) fn take_remote_charge(&mut self) -> Result<PreparedPartitionRemoteCharge<'_>, PartitionCaptureAllowanceError> {
        let controls = size_of::<PreparedPartitionRemoteCharge<'_>>()
            + size_of::<Result<PreparedPartitionRemoteCharge<'_>, PartitionCaptureAllowanceError>>()
            + size_of::<PartitionCaptureAllowanceError>() + size_of::<&mut Self>();
        self.metadata.reserve_metadata(controls).map_err(|cause| PartitionCaptureAllowanceError {
            cause: cause.into(), _source: self.source.clone(), _metadata: self.metadata.clone(),
        })?;
        if self.remote_attempted || self.local_rank == self.producer {
            return Err(PartitionCaptureAllowanceError { cause: Cause::Source("remote reservation is unavailable or spent"),
                _source: self.source.clone(), _metadata: self.metadata.clone() });
        }
        // Claim before any downstream validation. Dropping the token cannot
        // make the same global charge available for another local progress row.
        self.remote_attempted = true;
        Ok(PreparedPartitionRemoteCharge { source: &self.source, index: self.selection_index,
            usage: self.estimate.capture, _exclusive: std::marker::PhantomData })
    }
}
impl PreparedPartitionRemoteCharge<'_> {
    pub(crate) fn validate(&self, source: &SharedCapturePlan, index: usize) -> bool {
        self.source.same_storage(source) && self.index == index
    }
    pub(crate) fn usage(&self) -> CaptureUsage { self.usage }
}

impl PartitionCaptureAllowanceError {
    pub(crate) fn limit_skip(&self) -> Option<CaptureSkipReason> {
        match &self.cause {
            Cause::Capture(cause) => match cause.cause() {
                CaptureError::Limit { budget, cumulative } => Some(CaptureSkipReason::Limit {
                    budget: *budget, cumulative: *cumulative }), _ => None,
            }, _ => None,
        }
    }
}
