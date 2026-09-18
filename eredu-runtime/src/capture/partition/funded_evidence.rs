//! Paid provenance from the retained receipt, published only after delivery.
use super::*;
use eredu_nn::workspace::{HostMetadataFunding, HostMetadataFundingError};
use std::mem::{size_of, size_of_val};

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("partition evidence source: {0}")]
    Source(&'static str),
    #[error(transparent)] Funding(#[from] HostMetadataFundingError),
    #[error(transparent)] Destination(#[from] eredu_nn::Error),
    #[error(transparent)] Capture(#[from] CaptureError),
    #[error(transparent)] Ownership(#[from] receipt::ReceiptConstructionCause),
}
#[derive(Debug, thiserror::Error)]
#[error("prepared partition evidence: {cause}")]
pub(crate) struct PartitionCaptureEvidenceError {
    #[source] cause: Cause,
    _source: SharedCapturePlan,
    _metadata: HostMetadataFunding,
}

/// Payload precedes its immutable source and physical metadata custody. Only
/// the closed receipt constructor can create this value; no native permission.
#[derive(Debug)]
pub(crate) struct PreparedPartitionCaptureEvidence {
    pub(crate) value: PartitionCaptureEvidence,
    pub(crate) source: SharedCapturePlan,
    pub(crate) metadata: HostMetadataFunding,
    // Exact final record charge from the closed complete or contiguous source.
    // A contribution row alone excludes the global dense-assembly worker.
    charged: CaptureUsage,
    routed_pending:Vec<bool>,
}
enum Charges<'a> {
    Complete { charged: CaptureUsage, quota: &'a mut dyn CaptureReservation },
    Contiguous(&'a mut PreparedPartitionFragmentAllowance),
}
impl Charges<'_> {
    fn charged(&self, receipt: &PartitionCaptureReceiptPlan, producer: usize, fragment: usize)
        -> Option<CaptureUsage>
    {
        match self {
            Self::Complete { charged, .. } => Some(*charged),
            Self::Contiguous(source) => source.fragment_record_charge(receipt, producer, fragment).map(|(_, charged, _)| charged),
        }
    }
    fn reserve(&mut self, usage: CaptureUsage) -> Result<Option<CaptureSkipReason>, CaptureError> {
        match self {
            Self::Complete { quota, .. } => quota.reserve(usage),
            Self::Contiguous(source) => source.quota_mut().reserve(usage),
        }
    }
}
impl PreparedPartitionCaptureEvidence {
    pub(crate) fn charged(&self) -> CaptureUsage { self.charged }
    pub(crate) fn prepare(receipt: &PartitionCaptureReceiptPlan, charged: CaptureUsage,
        metadata: &HostMetadataFunding, quota: &mut dyn CaptureReservation)
        -> Result<Self, PartitionCaptureEvidenceError>
    {
        Self::prepare_sources(receipt, Charges::Complete { charged, quota }, metadata)
    }
    /// The same contribution writer, using every actual source-bound fragment
    /// charge. Empty ranks remain producers with no invented contribution row.
    pub(crate) fn prepare_contiguous(receipt: &PartitionCaptureReceiptPlan,
        allowance: &mut PreparedPartitionFragmentAllowance, metadata: &HostMetadataFunding)
        -> Result<Self, PartitionCaptureEvidenceError>
    {
        Self::prepare_sources(receipt, Charges::Contiguous(allowance), metadata)
    }
    fn prepare_sources(receipt: &PartitionCaptureReceiptPlan, mut charges: Charges<'_>,
        metadata: &HostMetadataFunding) -> Result<Self, PartitionCaptureEvidenceError>
    {
        let source = receipt.shared_plan_source().clone();
        let error = |cause| PartitionCaptureEvidenceError { cause,
            _source: source.clone(), _metadata: metadata.clone() };
        let producers = receipt.producers();
        let controls = [size_of::<Self>() * 2, size_of::<PartitionCaptureEvidence>() * 2,
            size_of::<PartitionCaptureContributionRecord>() * 2,
            size_of::<PartitionCaptureRegion>() * 4, size_of::<PartitionCaptureContext>() * 2,
            size_of::<PartitionCaptureEvidenceError>(), size_of::<Cause>(),
            size_of::<Result<Self, PartitionCaptureEvidenceError>>(),
            size_of::<(&PartitionCaptureReceiptPlan, CaptureUsage, &HostMetadataFunding,
                &mut dyn CaptureReservation)>(), size_of::<Option<(usize, &CaptureSlicePartition)>>(),
            size_of::<(&PartitionCaptureReceiptPlan, &mut PreparedPartitionFragmentAllowance, &HostMetadataFunding)>(),
            size_of::<(&PartitionCaptureReceiptPlan, Charges<'_>, &HostMetadataFunding)>(),
            size_of::<Charges<'_>>() * 2, size_of::<(usize, usize)>(),size_of::<Vec<bool>>(),
            size_of::<(&Self,&PartitionCaptureReceiptPlan)>(),size_of::<bool>(),
            size_of::<RoutedUnitCaptureOwnership>()*2,size_of::<RoutedUnitCaptureProvenance>()*2,
            size_of::<Result<RoutedUnitCaptureOwnership,receipt::ReceiptConstructionCause>>(),
            size_of::<Vec<[u64;2]>>(),size_of::<Option<RoutedUnitCaptureProvenance>>(),
            size_of::<CaptureUsage>() * 4, size_of::<Result<CaptureUsage,CaptureError>>(), size_of::<Option<CaptureSkipReason>>(),
            size_of::<Option<CaptureUsage>>(), size_of_val(&producers) * 2,
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, eredu_core::capture::CaptureFragmentGeometry>>>(),
            size_of::<[Vec<u64>; 4]>() * 2, size_of::<Result<PartitionCaptureRegion, eredu_nn::Error>>()];
        metadata.reserve_metadata(controls.into_iter().try_fold(size_of_val(&controls), usize::checked_add)
            .ok_or_else(|| error(Cause::Source("constructor controls overflow")))?)
            .map_err(|cause| error(cause.into()))?;
        let admitted = &source;
        if receipt.context().invocation.is_some()
            || matches!(&charges, Charges::Contiguous(allowance) if !allowance.matches(receipt))
        { return Err(error(Cause::Source("contribution source differs"))); }
        let mut producer_count = 0usize;
        let mut fragment_count = 0usize;
        for (producer, projection) in producers {
            if receipt.routed_producer(producer).is_none()&&projection.fragments().len()>1
                || receipt.routed_producer(producer).is_some()&&matches!(&charges,Charges::Complete{..}) {
                return Err(error(Cause::Source("expected qualified fragment producer")));
            }
            if matches!(&charges, Charges::Complete { .. })
                && (receipt.combination() != PartitionCaptureCombination::Disjoint
                    || projection.fragments().len() != 1 || projection.global_shape() != projection.local_shape())
            { return Err(error(Cause::Source("expected one complete nonrouted producer"))); }
            for fragment in 0..projection.fragments().len() {
                charges.charged(receipt, producer, fragment)
                    .ok_or_else(|| error(Cause::Source("fragment charge lacks its retained native source")))?;
            }
            producer_count = producer_count.checked_add(1).ok_or_else(|| error(Cause::Source("producer count overflow")))?;
            fragment_count = fragment_count.checked_add(projection.fragments().len())
                .ok_or_else(|| error(Cause::Source("fragment count overflow")))?;
        }
        if producer_count == 0 || (matches!(&charges, Charges::Complete { .. }) && producer_count != 1) {
            return Err(error(Cause::Source("producer count differs")));
        }
        let charged = match &charges {
            Charges::Complete { charged, .. } => *charged,
            Charges::Contiguous(allowance) => allowance.assembly_record_charge(receipt)
                .map_err(|cause| error(cause.into()))?,
        };
        let cost = session::evidence_usage(receipt).map_err(|cause| error(cause.into()))?;
        if let Some(reason) = charges.reserve(cost).map_err(|cause| error(cause.into()))? {
            return Err(error(match reason {
                CaptureSkipReason::Limit { budget, cumulative } => CaptureError::Limit { budget, cumulative }.into(),
                _ => Cause::Source("prepaid evidence reservation rejected"),
            }));
        }
        let context = receipt::copy_context(receipt.context(), metadata).map_err(|cause| error(cause.into()))?;
        let identity = metadata.metadata_string(format_args!("{}", receipt.identity()))
            .map_err(|cause| error(cause.into()))?;
        let mut producer_rows = metadata.metadata_vec(producer_count).map_err(|cause| error(cause.into()))?;
        let mut contributions = metadata.metadata_vec(fragment_count).map_err(|cause| error(cause.into()))?;
        let mut routed_pending=metadata.metadata_vec(fragment_count).map_err(|cause|error(cause.into()))?;
        for (producer, projection) in receipt.producers() {
            producer_rows.push(producer);
            for (fragment_index, fragment) in projection.fragments().iter().enumerate() {
                let routed=if let Some(ownership)=receipt.routed_producer(producer) {
                    let maximum=ownership.maximum_source_rows(projection.global_shape()[0],projection.global_shape()[1])
                        .map_err(|cause|error(cause.into()))?.min(receipt.max_record_bytes());
                    let ranges=metadata.metadata_vec(usize::try_from(maximum).map_err(|_|error(Cause::Source("range capacity overflow")))?)
                        .map_err(|cause|error(cause.into()))?;
                    Some(RoutedUnitCaptureProvenance{ownership:receipt::copy_routed_ownership(ownership,metadata).map_err(|cause|error(cause.into()))?,
                        source_token_ranges:ranges})
                }else{None};
                routed_pending.push(routed.is_some());
                contributions.push(PartitionCaptureContributionRecord {
                    producer_rank: producer,
                    local: copy_region(fragment.local(), metadata).map_err(|cause| error(cause.into()))?,
                    destination: copy_region(fragment.destination(), metadata).map_err(|cause| error(cause.into()))?,
                    charged: charges.charged(receipt, producer, fragment_index)
                        .ok_or_else(|| error(Cause::Source("retained fragment charge changed")))?,
                    routed,
                });
            }
        }
        Ok(Self { value: PartitionCaptureEvidence { schema_version: PARTITION_CAPTURE_SCHEMA_VERSION,
            combination: receipt.combination(), context, receipt_plan_identity: identity,
            producers: producer_rows, contributions }, source: admitted.clone(), metadata: metadata.clone(), charged, routed_pending })
    }
    pub(crate) fn matches(&self, receipt: &PartitionCaptureReceiptPlan) -> bool {
        self.matches_receipt(receipt) && self.routed_pending.iter().all(|pending|!*pending)
    }
    /// Authenticate the original receipt while a local hook still accumulates
    /// routed ranges. Final delivery separately requires all provenance rows.
    pub(crate) fn matches_receipt(&self, receipt: &PartitionCaptureReceiptPlan) -> bool {
        receipt.shared_plan_source().same_storage(&self.source)
            && &self.value.context == receipt.context()
            && self.value.receipt_plan_identity == receipt.identity()
            && self.value.combination == receipt.combination()
    }
    /// Bind one completed sparse fragment's actual native ranges into the paid
    /// provenance row. The original receipt supplies ownership; no sender map is copied.
    pub(crate) fn record_routed_ranges(&mut self,receipt:&PartitionCaptureReceiptPlan,producer:usize,fragment:usize,ranges:&[[u64;2]])
        ->Result<(),PartitionCaptureEvidenceError> {
        let source=self.source.clone();let metadata=self.metadata.clone();
        let error=|cause|PartitionCaptureEvidenceError{cause,_source:source.clone(),_metadata:metadata.clone()};
        let parts=[size_of::<(&mut Self,&PartitionCaptureReceiptPlan,usize,usize,&[[u64;2]])>(),
            size_of::<SharedCapturePlan>(),size_of::<HostMetadataFunding>(),size_of::<Option<usize>>(),
            size_of::<Result<(),PartitionCaptureEvidenceError>>(),size_of::<(usize,u64)>(),size_of::<std::slice::Iter<'_,[u64;2]>>()];
        metadata.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or_else(||error(Cause::Source("range binding controls overflow")))?).map_err(|e|error(e.into()))?;
        if !receipt.shared_plan_source().same_storage(&source)||self.value.receipt_plan_identity!=receipt.identity()
            ||&self.value.context!=receipt.context(){return Err(error(Cause::Source("range receipt differs")));}
        let index=self.value.contributions.iter().enumerate().filter(|(_,row)|row.producer_rank==producer).nth(fragment)
            .map(|(index,_)|index).ok_or_else(||error(Cause::Source("range fragment missing")))?;
        if self.routed_pending.get(index)!=Some(&true){return Err(error(Cause::Source("range fragment already spent")));}
        self.routed_pending[index]=false;
        let destination=self.value.contributions[index].routed.as_mut().ok_or_else(||error(Cause::Source("range ownership missing")))?;
        let ownership=receipt.routed_producer(producer).ok_or_else(||error(Cause::Source("range producer missing")))?;
        let projection=receipt.producer(producer).ok_or_else(||error(Cause::Source("range projection missing")))?;
        if destination.ownership!=*ownership||ranges.len()>destination.source_token_ranges.capacity(){return Err(error(Cause::Source("range destination differs")));}
        let maximum=ownership.maximum_source_rows(projection.global_shape()[0],projection.global_shape()[1]).map_err(|e|error(e.into()))?;
        let mut end=0;
        for &[start,next] in ranges{if start!=end||next<=start||next>maximum{return Err(error(Cause::Source("range coverage differs")));}end=next;}
        if ownership.source_peer.is_none()&&end!=projection.global_shape()[0]{return Err(error(Cause::Source("range source incomplete")));}
        destination.source_token_ranges.extend_from_slice(ranges);Ok(())
    }
}
fn copy_region(source: &ResolvedCaptureSlice, metadata: &HostMetadataFunding)
    -> Result<PartitionCaptureRegion, eredu_nn::Error>
{
    let mut starts = metadata.metadata_vec(source.starts.len())?;
    let mut ends = metadata.metadata_vec(source.ends.len())?;
    let mut strides = metadata.metadata_vec(source.strides.len())?;
    let mut shape = metadata.metadata_vec(source.shape.len())?;
    starts.extend_from_slice(&source.starts); ends.extend_from_slice(&source.ends);
    strides.extend_from_slice(&source.strides); shape.extend_from_slice(&source.shape);
    Ok(PartitionCaptureRegion { starts, ends, strides, shape })
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
