//! Borrow the ordinary backend worker while the local receipt/host owner is detached.
use super::*;
use crate::capture::{FundedCaptureError,ScheduledCaptureBackend};
use crate::capture::partition::{PartitionPrefillCapturePlan,PartitionPrefillCaptureKind,PartitionPrefillCaptureSourceError};
use eredu_core::{InferenceGeometry,capture::{CapturePrefillGeometryError,CaptureTensorGeometryError}};

#[derive(Debug,thiserror::Error)]
pub(super) enum LocalCause<E:std::error::Error+'static> {
    #[error("partition local source, scalar or estimate differs")] Identity,
    #[error(transparent)] Routed(#[from] eredu_core::capture::RoutedUnitValidationError),
    #[error(transparent)] Source(#[from] PartitionPrefillCaptureSourceError),
    #[error(transparent)] Invocation(#[from] crate::capture::partition::PartitionInvocationCaptureSourceError),
    #[error(transparent)] Rows(#[from] CapturePrefillGeometryError),
    #[error(transparent)] Geometry(#[from] CaptureTensorGeometryError),
    #[error(transparent)] Host(#[from] CaptureRunHostError),
    #[error(transparent)] Destination(#[from] PartitionFragmentDestinationError),
    #[error(transparent)] Capture(#[from] CaptureError),
    #[error(transparent)] Funding(#[from] WorkspaceMetadataFundingError),
    #[error(transparent)] Work(#[from] FundedCaptureError<E>),
}
/// A failed callback consumes its owner; unfinished chunks cannot be returned
/// as completed. Its actual native error and partial host/source remain together.
#[derive(Debug,thiserror::Error)]
#[error("partition local callback: {cause}")]
pub(crate) struct PartitionLocalCaptureFailure<E:std::error::Error+'static> {
    #[source] pub(super) cause:LocalCause<E>,
    pub(super) _hook:PartitionLocalCaptureHook,
}
impl PartitionLocalCaptureHook {
    pub(crate) fn observe_prefill_program<T,E:std::error::Error+Send+Sync+'static>(self,
        backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,
        inference:InferenceGeometry,chunk:u64)
        ->Result<Self,crate::capture::partition::PartitionCaptureProgramError> {
        self.observe_prefill(backend,value,inference,chunk).map_err(|failure|{
            let source=failure._hook.source().clone();let metadata=failure._hook.funding().clone();
            crate::capture::partition::PartitionCaptureProgramError::local(failure,source,metadata)
        })
    }

    /// Exact local projection from this receipt, using the same native backend
    /// methods and fixed host claims as the ordinary funded observer. The caller
    /// separately owns canonical global hook/epoch progression and transport.
    pub(crate) fn observe_prefill<T,E:std::error::Error+Send+Sync+'static>(mut self,
        backend:&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,
        inference:InferenceGeometry,chunk:u64)->Result<Self,PartitionLocalCaptureFailure<E>> {
        let result=(||{
            let controls=control_bytes::<T,E>(&self.receipt).ok_or(LocalCause::Identity)?;
            self.metadata.reserve_metadata(controls)?;
            let receipt=&self.receipt;let bank=&mut self.bank;
            let rank=bank.allowance.local_rank();
            let source=PartitionPrefillCapturePlan::prepare(receipt,rank,0,inference)?;
            match source.kind() {
                PartitionPrefillCaptureKind::Tensor(rows)=>{
                    let fragment=rows.fragment(chunk)?;
                    let dtype=backend.validate_prefill_source(value,&fragment)?;
                    let usage=backend.estimate_prefill(rows.logical_geometry())?;
                    validate_or_begin::<E>(bank,receipt,chunk,&dtype,usage)?;
                    let claim=bank.take_local_prefill_chunk(receipt,0,&fragment)?;
                    if fragment.output_elements()==0 {claim.prepare()?.finish()?;}
                    else {backend.transform_prefill_fragment(value,claim)?;}
                },
                PartitionPrefillCaptureKind::Summary(plan)=>{
                    let fragment=plan.fragment(chunk)?;
                    let geometry=CaptureSummaryGeometry::prepare_partition(plan.admission(),plan.selection_index(),
                        CapturePhase::Prefill,0,None,receipt.producer(rank).ok_or(LocalCause::Identity)?,0)?
                        .fragment(&fragment)?;
                    let dtype=backend.validate_summary_source(value,&geometry)?;
                    let usage=backend.estimate_prefill_summary(plan)?;
                    validate_or_begin::<E>(bank,receipt,chunk,&dtype,usage)?;
                    let PartitionFragmentDestination::Summary(claim)=bank.take_local_prefill_reduction(receipt,0,&fragment)?
                        else{return Err(LocalCause::Identity);};
                    let value=if fragment.selected_elements()==0 {claim.finish_empty().map_err(FundedCaptureError::from)?}
                        else{backend.transform_summary(value,claim)?};
                    bank.record_local_prefill_reduction(receipt,0,&fragment,PartitionFragmentValue::Summary(value))?;
                },
                PartitionPrefillCaptureKind::Histogram(plan)=>{
                    let fragment=plan.fragment(chunk)?;
                    let geometry=CaptureHistogramGeometry::prepare_partition(plan.admission(),plan.selection_index(),
                        CapturePhase::Prefill,0,None,receipt.producer(rank).ok_or(LocalCause::Identity)?,0)?
                        .fragment(&fragment)?;
                    let dtype=backend.validate_histogram_source(value,&geometry)?;
                    let usage=backend.estimate_prefill_histogram(plan)?;
                    validate_or_begin::<E>(bank,receipt,chunk,&dtype,usage)?;
                    let PartitionFragmentDestination::Histogram(claim)=bank.take_local_prefill_reduction(receipt,0,&fragment)?
                        else{return Err(LocalCause::Identity);};
                    let value=if fragment.selected_elements()==0 {claim.prepare().map_err(FundedCaptureError::from)?
                        .finish(0,0,0).map_err(FundedCaptureError::from)?}
                        else{backend.transform_histogram(value,claim)?};
                    bank.record_local_prefill_reduction(receipt,0,&fragment,PartitionFragmentValue::Histogram(value))?;
                },
            }
            bank.complete_local_prefill_chunk(receipt,0,chunk)?;
            Ok(())
        })();
        match result {Ok(())=>Ok(self),Err(cause)=>Err(PartitionLocalCaptureFailure{cause,_hook:self})}
    }
}
fn validate_or_begin<E:std::error::Error+'static>(bank:&mut PreparedPartitionFragmentDestinations,
    receipt:&PartitionCaptureReceiptPlan,chunk:u64,dtype:&TensorDtype,usage:CaptureUsage)->Result<(),LocalCause<E>> {
    let estimate=PartitionCaptureNativeEstimate{capture:usage,generated_creation_bytes:0};
    if !bank.allowance.fragment_source(bank.allowance.local_rank(),0)
        .is_some_and(|(expected,actual)|expected==dtype && actual==estimate) {return Err(LocalCause::Identity);}
    if chunk==0 {
        let mut loan=bank.begin_local_prefill(receipt,0,dtype,estimate)?;
        // These are the existing whole-fragment child credits. They are spent
        // once; subsequent chunks validate the same estimate without a refund.
        loan.quota_mut().reserve_quota(usage)?;
    }
    Ok(())
}
fn control_bytes<T,E:std::error::Error+Send+Sync+'static>(receipt:&PartitionCaptureReceiptPlan)->Option<usize> {
    let transform=&receipt.shared_plan_source()?.admission().plan().selections.get(receipt.context().selection_index)?.transform;
    let raw=receipt.combination()==PartitionCaptureCombination::SumF64ToF32
        || matches!(transform,CaptureTransform::FullTensor|CaptureTransform::Slice|CaptureTransform::Preview{..});
    let worker=if raw {
        let parts=[size_of::<CapturePrefillFragment<'_,'_>>(),
            size_of::<crate::working_memory::CapturePrefillFragmentClaim<'_,'_,'_,'_>>(),
            size_of::<crate::working_memory::CapturePrefillFragmentWriter<'_,'_,'_,'_>>(),
            size_of::<Result<crate::working_memory::CapturePrefillFragmentClaim<'_,'_,'_,'_>,PartitionFragmentDestinationError>>(),
            size_of::<Result<crate::working_memory::CapturePrefillFragmentWriter<'_,'_,'_,'_>,CaptureRunHostError>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?
    }else if matches!(transform,CaptureTransform::Summary) {
        let parts=[size_of::<CapturePrefillTransformFragment<'_,'_>>(),size_of::<CaptureSummaryGeometry<'_>>(),
            size_of::<crate::working_memory::CaptureSummaryClaim<'_,'_>>(),size_of::<crate::working_memory::ClaimedCaptureSummary>(),
            size_of::<Result<crate::working_memory::ClaimedCaptureSummary,FundedCaptureError<E>>>(),
            size_of::<Result<crate::working_memory::ClaimedCaptureSummary,crate::working_memory::CaptureSummaryFailure>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?
    }else if matches!(transform,CaptureTransform::Histogram{..}) {
        let parts=[size_of::<CapturePrefillTransformFragment<'_,'_>>(),size_of::<CaptureHistogramGeometry<'_>>(),
            size_of::<crate::working_memory::CaptureHistogramClaim<'_,'_>>(),size_of::<crate::working_memory::ClaimedCaptureHistogram>(),
            size_of::<crate::working_memory::ScheduledCaptureHistogram<'_,'_>>(),
            size_of::<Result<crate::working_memory::ClaimedCaptureHistogram,FundedCaptureError<E>>>(),
            size_of::<Result<crate::working_memory::ClaimedCaptureHistogram,crate::working_memory::CaptureHistogramFailure>>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)?
    }else{return None;};
    let parts=[crate::capture::partition::PartitionCaptureLocalHook::control_bytes()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<PartitionLocalCaptureFailure<E>>()?,
        size_of::<crate::capture::partition::PartitionCaptureLocalHook>()*2,
        size_of::<crate::capture::partition::PartitionCaptureProgramError>(),
        size_of::<Result<crate::capture::partition::PartitionCaptureLocalHook,crate::capture::partition::PartitionCaptureProgramError>>(),
        PartitionPrefillCapturePlan::control_bytes()?,size_of::<PartitionLocalCaptureHook>()*2,
        size_of::<PartitionLocalCaptureFailure<E>>(),size_of::<LocalCause<E>>(),size_of::<FundedCaptureError<E>>(),
        size_of::<Result<PartitionLocalCaptureHook,PartitionLocalCaptureFailure<E>>>(),
        size_of::<Result<(),LocalCause<E>>>(),size_of::<(PartitionLocalCaptureHook,
            &mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,&T,InferenceGeometry,u64)>(),
        size_of::<(&mut PreparedPartitionFragmentDestinations,&PartitionCaptureReceiptPlan,u64,&TensorDtype,CaptureUsage)>(),
        size_of::<PartitionCaptureNativeEstimate>(),size_of::<PreparedPartitionFragmentLoan>(),
        size_of::<CaptureUsage>(),size_of::<TensorDtype>(),size_of::<InferenceGeometry>(),
        worker,size_of::<(&mut PartitionLocalCaptureHook,&mut dyn ScheduledCaptureBackend<Tensor=T,Error=E>,&T,InferenceGeometry,u64)>(),
        size_of::<PartitionFragmentDestination<'_,'_>>(),size_of::<PartitionFragmentValue>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}
