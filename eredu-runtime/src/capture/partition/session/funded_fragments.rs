//! Exact fragment sources lent to the ordinary global receipt cost worker.
use super::*;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::{size_of, size_of_val};

/// One actual native quote, bound to its retained receipt fragment. This is a
/// descriptive source, not a substitute for the native producer's work grant.
#[derive(Debug)]
pub struct PartitionCaptureFragmentSource<'a> {
    /// Authoritative world rank, including its actual TP term identity.
    pub producer: usize,
    /// Exact fragment ordinal; empty producers supply no native source row.
    pub fragment: usize,
    /// Actual complete invocation-local tensor shape before slicing.
    pub local_shape: &'a [u64],
    /// Exact normalized local coordinates of this fragment.
    pub local_slice: &'a ResolvedCaptureSlice,
    /// Actual native transform. Additive reductions must quote raw Slice (or
    /// bounded Preview), never a shard-local nonlinear statistic.
    pub transform: &'a CaptureTransform,
    /// Actual scalar representation from the native cold source.
    pub dtype: TensorDtype,
    /// Existing native estimate for this exact local shape/slice/transform.
    pub estimate: PartitionCaptureNativeEstimate,
}
/// Native fragment geometry and estimate before the distributed source vote.
/// It supplies no scalar witness and cannot issue a native fragment loan.
#[derive(Debug)]
pub struct PartitionCaptureFragmentGeometry<'a> {
    /// Authoritative world rank of this actual contribution.
    pub producer:usize,
    /// Exact local fragment ordinal.
    pub fragment:usize,
    /// Actual complete invocation-local source shape.
    pub local_shape:&'a [u64],
    /// Normalized local selected coordinates.
    pub local_slice:&'a ResolvedCaptureSlice,
    /// Existing native transform, including raw additive terms.
    pub transform:&'a CaptureTransform,
    /// Scalar-independent F32 readout and host estimate for this exact source.
    pub estimate:PartitionCaptureNativeEstimate,
}
mod routed;
pub use routed::{PartitionCaptureRoutedFragmentGeometry,PartitionCaptureRoutedFragmentSource};
mod voted;
pub(crate) use voted::{PreparedPartitionFragmentSourceAllowance,SourceBindingError};
use voted::Sources;
#[derive(Debug)]
struct Row {
    producer: usize,
    fragment: usize,
    dtype: Option<TensorDtype>,
    estimate: PartitionCaptureNativeEstimate,
    taken: bool,
}
#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("partition fragment source: {0}")]
    Source(&'static str),
    #[error(transparent)] Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)] Destination(#[from] eredu_nn::Error),
    #[error(transparent)] Capture(#[from] CaptureError),
}
/// A failure retains the original admission and paid metadata. Any global
/// credits consumed before refusal remain spent in their original ledger.
#[derive(Debug, thiserror::Error)]
#[error("original partition fragment allowance: {cause}")]
pub struct PartitionCaptureFragmentAllowanceError {
    #[source] cause: Cause,
    _source: Option<SharedCapturePlan>,
    _metadata: WorkspaceMetadataFunding,
}

/// Per-rank/per-fragment sources and a local allowance from the ordinary global
/// cost equation. No rank can consume another rank's native source. The receipt
/// identity is retained *after* source-derived encoded bounds are finalized.
#[must_use = "consume only with this receipt's prepared fragment destinations"]
#[derive(Debug)]
pub struct PreparedPartitionFragmentAllowance {
    rows: Vec<Row>,
    assembly_claimed: bool,
    identity: String,
    quota: CaptureQuota,
    global: CaptureUsage,
    descriptor: [u8; 32],
    rank: usize,
    source: SharedCapturePlan,
    metadata: WorkspaceMetadataFunding,
}
/// Move-only native credits retaining the exact source and metadata account.
#[must_use = "spend through the source-bound fragment destination"]
#[derive(Debug)]
pub struct PreparedPartitionFragmentLoan {
    quota: CaptureQuota,
    source: SharedCapturePlan,
    _metadata: WorkspaceMetadataFunding,
}
impl PreparedPartitionFragmentLoan {
    /// Lend only this fragment's already charged native credits.
    pub fn quota_mut(&mut self) -> &mut CaptureQuota { &mut self.quota }
    /// Original immutable source retained through every escaped loan.
    pub fn source(&self) -> &SharedCapturePlan { &self.source }
}
impl PreparedPartitionFragmentAllowance {
    /// Authenticate every real fragment in canonical producer/fragment order,
    /// then reserve exactly the same global/local costs as ordinary delivery.
    /// Empty producers remain in the receipt but require no invented native row.
    pub fn prepare<T: PartitionCaptureTransport>(
        transport: &T, receipt: &mut PartitionCaptureReceiptPlan,
        sources: &[PartitionCaptureFragmentSource<'_>], metadata: &WorkspaceMetadataFunding,
        ledger: &mut dyn CaptureReservation,
    ) -> Result<Self, PartitionCaptureFragmentAllowanceError>
    where T::Error: Send + Sync + 'static,
        <T::Completion as Completion>::Error: Send + Sync + 'static,
    {
        Self::prepare_sources(transport,receipt,Sources::Typed(sources),metadata,ledger)
    }
    fn prepare_sources<T:PartitionCaptureTransport>(transport:&T,receipt:&mut PartitionCaptureReceiptPlan,
        sources:Sources<'_,'_>,metadata:&WorkspaceMetadataFunding,ledger:&mut dyn CaptureReservation)
        ->Result<Self,PartitionCaptureFragmentAllowanceError>
    where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
        let source = receipt.shared_plan_source().cloned();
        let error = |cause| PartitionCaptureFragmentAllowanceError {
            cause, _source: source.clone(), _metadata: metadata.clone(),
        };
        let parts = [size_of::<Self>() * 2, size_of::<Row>() * 2,
            size_of::<PartitionCaptureFragmentSource<'_>>(),size_of::<PartitionCaptureFragmentGeometry<'_>>(),
            size_of::<Sources<'_,'_>>(),size_of::<voted::Source<'_>>(),size_of::<std::ops::Range<usize>>(),
            size_of::<Result<Self, PartitionCaptureFragmentAllowanceError>>(),
            size_of::<PartitionCaptureFragmentAllowanceError>(), size_of::<Cause>(),
            size_of::<ReceiptWorkCosts>() * 2, size_of::<CaptureQuota>() * 2,
            size_of::<CaptureUsage>() * 8, size_of::<PartitionCaptureNativeEstimate>() * 2,
            size_of::<Sha256>(), size_of::<sha2::digest::Output<Sha256>>(),
            size_of::<Option<(usize, &CaptureSlicePartition)>>(),
            size_of::<std::iter::Enumerate<std::slice::Iter<'_, CaptureFragmentGeometry>>>(),
            size_of::<std::slice::Iter<'_, Row>>(), size_of::<(usize, usize, usize, [u8; 8])>(),
            size_of::<Option<&PartitionCaptureFragmentSource<'_>>>(),
            size_of::<CaptureTransform>(), size_of::<(usize, bool)>(),
            size_of::<(&mut usize, &Vec<Row>)>(),
            size_of::<Result<PartitionCaptureNativeEstimate, CaptureError>>(),
            size_of::<Result<(), CaptureError>>(),
            size_of::<(&T, &mut PartitionCaptureReceiptPlan, &[PartitionCaptureFragmentSource<'_>],
                &WorkspaceMetadataFunding, &mut dyn CaptureReservation)>(),
            size_of::<(&T,&mut PartitionCaptureReceiptPlan,Sources<'_,'_>,&WorkspaceMetadataFunding,&mut dyn CaptureReservation)>(),
            PartitionCaptureReceiptPlan::record_bound_control_bytes()
                .ok_or_else(|| error(Cause::Source("record-bound controls overflow")))?,
        ];
        metadata.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| error(Cause::Source("fragment allowance controls overflow")))?)
            .map_err(|cause| error(cause.into()))?;
        if sources.routed() {
            metadata.reserve_metadata(routed::control_bytes().ok_or_else(||error(Cause::Source("sparse source controls overflow")))?)
                .map_err(|cause|error(cause.into()))?;
        }
        let retained = source.as_ref().ok_or_else(|| error(Cause::Source("receipt has no original source")))?;
        if receipt.context().capture_plan_identity != retained.admission().identity()
            || receipt.context().invocation.is_some()
            || transport.participant_count() != receipt.world_size()
            || transport.capture_rank() >= receipt.world_size()
            || sources.routed() != receipt.producers().any(|(rank, _)| receipt.routed_producer(rank).is_some())
        { return Err(error(Cause::Source("receipt, invocation or transport differs"))); }
        let selection = &retained.admission().plan().selections[receipt.context().selection_index];
        if if sources.routed() { !matches!(selection.transform,CaptureTransform::RoutedUnits) }
            else { !matches!(selection.transform, CaptureTransform::FullTensor | CaptureTransform::Slice
                | CaptureTransform::Preview { .. } | CaptureTransform::Summary | CaptureTransform::Histogram { .. }) }
        { return Err(error(Cause::Source("fragment delivery requires a floating tensor transform"))); }
        let raw;
        let transform = if receipt.combination() == PartitionCaptureCombination::SumF64ToF32 {
            raw = super::super::sum::native_transform(&selection.transform); &raw
        } else { &selection.transform };
        let mut cursor = 0usize;
        for (rank, projection) in receipt.producers() {
            for (fragment, geometry) in projection.fragments().iter().enumerate() {
                let row = sources.get(cursor).ok_or_else(|| error(Cause::Source("missing actual fragment source")))?;
                let matched=match row.geometry {
                    voted::Geometry::Tensor{shape,slice,transform:actual}=>
                        shape==projection.local_shape()&&slice==geometry.local()&&actual==transform,
                    voted::Geometry::Routed(request)=>routed::matches(receipt,rank,fragment,request),
                };
                if row.producer != rank || row.fragment != fragment || !matched
                    || !(matches!(row.dtype,None|Some(TensorDtype::F16|TensorDtype::Bf16|TensorDtype::F32))
                        || sources.routed()&&row.dtype==Some(&TensorDtype::F64))
                    || row.estimate.generated_creation_bytes != 0
                { return Err(error(Cause::Source("native fragment identity, geometry or transform differs"))); }
                cursor += 1;
            }
        }
        if cursor != sources.len() { return Err(error(Cause::Source("unexpected native fragment source"))); }
        // Pay every source row and the final receipt's fixed SHA-256 text before
        // the fallible quota worker can consume global credits.
        let mut rows = metadata.metadata_vec(sources.len()).map_err(|cause| error(cause.into()))?;
        metadata.reserve_metadata(64).map_err(|cause| error(cause.into()))?;
        let mut identity = String::with_capacity(64);
        for index in 0..sources.len() { let row=sources.get(index).expect("source range");
            rows.push(Row { producer: row.producer, fragment: row.fragment,
            dtype: row.dtype.cloned(), estimate: row.estimate, taken: false }); }
        let mut cursor = 0usize;
        let costs = receipt_work_costs(transport, receipt, |_, producer, fragment| {
            let row = rows.get(cursor).ok_or_else(|| invalid("missing prepared native fragment"))?;
            if row.producer != producer || row.fragment != fragment {
                return Err(invalid("prepared native fragment order differs"));
            }
            cursor += 1; Ok(row.estimate)
        }, |_| {
            // The ordinary routed worker retains each local fragment's charge.
            // Here the existing paid Row table already owns the same source;
            // its native and metadata credits are spent by the bank separately.
            if sources.routed() {Ok(())} else {Err(invalid("dense fragments cannot create routed metadata"))}
        })
            .map_err(|cause| error(cause.into()))?;
        if cursor != rows.len() || receipt.identity().len() != 64 {
            return Err(error(Cause::Source("finalized receipt differs from its sources")));
        }
        identity.push_str(receipt.identity());
        let quota = ledger.reserve_quota(costs.global)
            .and_then(|mut global| global.reserve_quota(costs.local))
            .map_err(|cause| error(cause.into()))?;
        Ok(Self { rows, assembly_claimed: false, identity, quota, global: costs.global, descriptor: costs.descriptor,
            rank: transport.capture_rank(), source: retained.clone(), metadata: metadata.clone() })
    }
    /// Compare physical admission and finalized descriptor before any source use.
    pub fn matches(&self, receipt: &PartitionCaptureReceiptPlan) -> bool {
        receipt.shared_plan_source().is_some_and(|source| self.source.same_storage(source))
            && receipt.identity() == self.identity
    }
    /// The global charge includes every producer and each rank's delivery work.
    pub const fn global_reserved(&self) -> CaptureUsage { self.global }
    /// This rank's ordinary local producer/common allowance.
    pub fn local_reserved(&self) -> CaptureUsage { self.quota.limit() }
    /// Actual spent local work; read-only and never refunded.
    pub fn local_used(&self) -> CaptureUsage { self.quota.used() }
    /// Descriptor from the same ordinary cost writer for all-rank coordination.
    pub const fn descriptor(&self) -> &[u8; 32] { &self.descriptor }
    /// Existing source owner; source rows do not manufacture a new admission.
    pub fn source(&self) -> &SharedCapturePlan { &self.source }
    /// Authenticate one local source exactly once before lending a child native
    /// quota. Failure cannot make the attempted source reusable. Metadata and
    /// assembly credits remain in the parent allowance for their own consumers.
    pub fn take_local_fragment(&mut self, receipt: &PartitionCaptureReceiptPlan, fragment: usize,
        dtype: &TensorDtype, actual: PartitionCaptureNativeEstimate)
        -> Result<PreparedPartitionFragmentLoan, PartitionCaptureFragmentAllowanceError>
    {
        let error = |cause| PartitionCaptureFragmentAllowanceError { cause,
            _source: Some(self.source.clone()), _metadata: self.metadata.clone() };
        let parts = [size_of::<(&mut Self, &PartitionCaptureReceiptPlan, usize, &TensorDtype)>(),
            size_of::<PartitionCaptureNativeEstimate>(), size_of::<CaptureQuota>(),
            size_of::<Result<PreparedPartitionFragmentLoan, PartitionCaptureFragmentAllowanceError>>(),
            size_of::<PreparedPartitionFragmentLoan>() * 2,
            size_of::<PartitionCaptureFragmentAllowanceError>(), size_of::<Cause>(),
            size_of::<Option<&mut Row>>(), size_of::<std::slice::IterMut<'_, Row>>(),
        ];
        self.metadata.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts), usize::checked_add)
            .ok_or_else(|| error(Cause::Source("fragment loan controls overflow")))?)
            .map_err(|cause| error(cause.into()))?;
        if !self.matches(receipt) { return Err(error(Cause::Source("fragment receipt source differs"))); }
        let row = self.rows.iter_mut().find(|row| row.producer == self.rank && row.fragment == fragment)
            .ok_or_else(|| error(Cause::Source("rank has no such native fragment")))?;
        if row.taken { return Err(error(Cause::Source("native fragment already attempted"))); }
        row.taken = true;
        if row.dtype.as_ref() != Some(dtype) || row.estimate.capture != actual.capture
            || row.estimate.generated_creation_bytes != actual.generated_creation_bytes
        { return Err(error(Cause::Source("actual fragment scalar or estimate differs"))); }
        let quota = self.quota.reserve_quota(actual.capture).map_err(|cause| error(cause.into()))?;
        Ok(PreparedPartitionFragmentLoan { quota, source: self.source.clone(), _metadata: self.metadata.clone() })
    }
    /// Common paid receipt/assembly/evidence credits, lent only inside runtime.
    pub(crate) fn quota_mut(&mut self) -> &mut CaptureQuota { &mut self.quota }
}

impl PreparedPartitionFragmentAllowance {
    /// Actual rank whose local native credits were reserved by this source.
    pub const fn local_rank(&self)->usize {self.rank}
    pub(crate) fn fragment_source(&self,producer:usize,fragment:usize)->Option<(&TensorDtype,PartitionCaptureNativeEstimate)> {
        self.rows.iter().find(|row|row.producer==producer&&row.fragment==fragment).and_then(|row|row.dtype.as_ref().map(|dtype|(dtype,row.estimate)))
    }
}

impl PreparedPartitionFragmentAllowance {
    /// Same ordinary fragment metadata/native charge, with no reservation or
    /// reconstruction of a native selection. The caller supplies the receipt
    /// already finalized by this allowance.
    pub(crate) fn fragment_record_charge(&self,receipt:&PartitionCaptureReceiptPlan,producer:usize,fragment:usize)
        ->Option<(TensorDtype,CaptureUsage,CaptureUsage)> {
        if !self.matches(receipt){return None;}
        let (dtype,native)=self.fragment_source(producer,fragment)?;
        Some((dtype.clone(),self.fragment_record_overhead(receipt,producer)?.checked_add(native.capture).ok()?,native.capture))
    }
    /// Ordinary metadata body separate from the independently consumed native loan.
    pub(crate) fn fragment_record_overhead(&self,receipt:&PartitionCaptureReceiptPlan,producer:usize)->Option<CaptureUsage> {
        if !self.matches(receipt){return None;}
        let source=self.source.admission();let index=receipt.context().selection_index;
        let selection=source.plan().selections.get(index)?;let point=source.points().get(index)?;
        let projection=receipt.producer(producer)?;
        let metadata=super::super::fragment_metadata_usage(selection,point,projection.global_shape().len()).ok()?
            .checked_add(if receipt.combination()==PartitionCaptureCombination::SumF64ToF32 {
                metadata_reservation(selection,point).ok()?
            } else {CaptureUsage::default()}).ok()?;
        Some(metadata)
    }
}

mod assembly_charge;
pub(crate) use assembly_charge::PreparedPartitionAssemblyCharge;

impl PartitionCaptureFragmentAllowanceError {
    pub(crate) fn limit_skip(&self) -> Option<CaptureSkipReason> {
        match &self.cause {
            Cause::Capture(cause) => match cause.cause() {
                CaptureError::Limit { budget, cumulative } => Some(CaptureSkipReason::Limit {
                    budget: *budget, cumulative: *cumulative }),
                _ => None,
            },
            _ => None,
        }
    }
}
