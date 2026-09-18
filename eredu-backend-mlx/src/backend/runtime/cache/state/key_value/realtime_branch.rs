//! Actual transaction branch, borrowing the existing state and checkpoint policy.
use super::*;
use crate::backend::{array_copy::{OriginalCopyLayoutBuilder,RealtimeCopyPlan,RealtimeCopyContext},error::Error};
use eredu_core::BackendFailure;
use eredu_nn::workspace::{WorkspaceContext,HostMetadataFunding};
use eredu_runtime::working_memory::{OriginalRealtimeNative,OriginalRealtimeBudgetCustody,WorkingMemoryError};
use sha2::{Digest,Sha256};
use safemlx::{PreparedArrayClone,PreparedInputRuntime,PreparedStreamCopy,StreamCopyPlan};
use std::{mem::{size_of,size_of_val},time::Duration};

/// Exact borrowed source. Inspection follows the same Concat checkpoint branch
/// as ordinary transactions; it does not manufacture a state or a copy grant.
pub(crate) struct RealtimeKvBranchPlan<'a> {
    source:&'a MlxKeyValueState,arrays:usize,copies:usize,
    paged_identity:[u8;32],paged_host:usize,paged_inspection:usize,
    reporting_identity:[u8;32],reporting_host:usize,reporting_inspection:usize,
}
struct CopySources {
    _arrays:Vec<Array>,_stream:PreparedStreamCopy<OriginalRealtimeBudgetCustody>,
    _funding:HostMetadataFunding,
}
#[derive(Debug,thiserror::Error)]
#[error("realtime state branch: {cause}")]
struct Failure {
    #[source] cause:Error,
    funding:HostMetadataFunding,
}
fn overflow()->Error {Error::PrefillControl(WorkingMemoryError::Overflow)}
fn mismatch()->Error {Error::PrefillControl(WorkingMemoryError::IdentityMismatch)}
fn sum(parts:&[usize])->Option<usize>{parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)}
fn alias_bytes()->Option<usize>{PreparedArrayClone::control_bytes()?.checked_add(Array::inspection_clone_handle_bytes())}
fn alias(source:&Array,funding:&HostMetadataFunding)->Result<Array,Error> {
    funding.reserve_metadata(alias_bytes().ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
    let mut slot=PreparedArrayClone::try_prepare_for_inspection().map_err(Error::OriginalSamplingClone)?;
    slot.fill_for_inspection(source).map_err(Error::OriginalSamplingClone)
}
fn copy_callback<'a>(source:Option<&'a RealtimeKvBranchPlan<'a>>,funding:Option<&'a HostMetadataFunding>)
    ->impl FnMut(&CopySources,&RealtimeCopyContext<'_>)->Result<MlxKeyValueTransactionBranch,Error>+'a {
    move |_,copy|source.expect("active branch source").construct(funding.expect("active branch funding"),Some(copy))
}
fn paged_managers(source:&MlxKeyValueState)->impl Iterator<Item=&PagedKeyValueCache> {
    source.layers.slots().iter().enumerate().filter_map(move |(index,layer)| {
        let MlxKeyValueLayerState::Paged(cache)=layer else{return None};
        (!source.layers.slots()[..index].iter().any(|prior|matches!(prior,
            MlxKeyValueLayerState::Paged(prior) if prior.manager().same_catalog(cache.manager()))))
            .then_some(cache)
    })
}
fn reporting_layers<'a>(source:&'a MlxKeyValueState,manager:&'a CacheResidencyManager)
    ->impl Iterator<Item=usize>+Clone+'a {
    source.layers.slots().iter().filter_map(move |layer|match layer {
        MlxKeyValueLayerState::Paged(cache) if cache.manager().same_catalog(manager)=>Some(cache.global_layer()),
        _=>None,
    })
}
impl<'a> RealtimeKvBranchPlan<'a> {
    pub(crate) fn matches_source(&self,state:&MlxKeyValueState)->bool {std::ptr::eq(self.source,state)}
    pub(crate) fn inspect(source:&'a MlxKeyValueState)->Result<Self,Error> {
        let mut arrays=0usize;let mut copies=0usize;let mut failed=false;
        let mut paged_hash=Sha256::new();let mut paged_host=0usize;let mut paged_inspection=0usize;
        for layer in source.layers.slots() {
            match layer {
                MlxKeyValueLayerState::Stateless=>{},
                MlxKeyValueLayerState::Device(cache)=>cache.visit_checkpoint_operands(&mut |_,copy| {
                    match arrays.checked_add(1){Some(n)=>arrays=n,None=>failed=true};
                    if copy {match copies.checked_add(1){Some(n)=>copies=n,None=>failed=true};}
                }),
                MlxKeyValueLayerState::Paged(cache)=>{
                    let source=cache.inspect_realtime_branch()?;
                    arrays=arrays.checked_add(source.arrays).ok_or_else(overflow)?;
                    paged_host=paged_host.checked_add(source.host_bytes().ok_or_else(overflow)?).ok_or_else(overflow)?;
                    paged_inspection=paged_inspection.checked_add(PagedKeyValueCache::realtime_branch_inspection_controls()
                        .ok_or_else(overflow)?).ok_or_else(overflow)?;
                    paged_hash.update(source.identity);
                },
            }
        }
        let mut reporting_hash=Sha256::new();let mut reporting_host=0usize;let mut reporting_inspection=0usize;
        for cache in paged_managers(source) {
            let future=reporting_layers(source,cache.manager());
            let controls=CacheResidencyManager::realtime_reporting_inspection_controls(&future).ok_or_else(overflow)?;
            let reporting=cache.manager().inspect_realtime_reporting(future)
                .map_err(|cause|Error::StorageSource(BackendFailure::from_error(cause)))?;
            reporting_hash.update(reporting.identity());
            reporting_host=reporting_host.checked_add(reporting.host_bytes().ok_or_else(overflow)?).ok_or_else(overflow)?;
            reporting_inspection=reporting_inspection.checked_add(controls).ok_or_else(overflow)?;
        }
        if failed{return Err(overflow());}Ok(Self{source,arrays,copies,
            paged_identity:paged_hash.finalize().into(),paged_host,paged_inspection,
            reporting_identity:reporting_hash.finalize().into(),reporting_host,reporting_inspection})
    }
    fn visit(&self,visit:&mut dyn FnMut(&Array,bool)) {
        for layer in self.source.layers.slots() {
            match layer {
                MlxKeyValueLayerState::Device(cache)=>cache.visit_checkpoint_operands(visit),
                MlxKeyValueLayerState::Paged(cache)=>cache.visit_realtime_tail_operands(visit),
                MlxKeyValueLayerState::Stateless=>{},
            }
        }
    }
    pub(crate) fn copy_plan<'s>(&self,runtime:&PreparedInputRuntime,stream:&'s Stream)
        ->Result<Option<RealtimeCopyPlan<'s>>,Error> {
        let mut builder=OriginalCopyLayoutBuilder::new();let mut error=None;
        self.visit(&mut |array,copy| {
            if copy && error.is_none() {
                error=builder.push_retained_source(array).and_then(|()|builder.push_operand(array)).err();
            }
        });
        if let Some(cause)=error{return Err(Error::StorageSource(BackendFailure::from_error(cause)));}
        builder.finish_realtime(runtime,stream).map_err(|cause|Error::StorageSource(BackendFailure::from_error(cause)))
    }
    fn fixed_bytes(&self)->Option<usize> {
        let callback=copy_callback(None,None);
        let parts=[size_of_val(&callback),size_of::<Sha256>(),size_of::<Result<Self,Error>>(),size_of::<Self>(),size_of::<MlxKeyValueState>(),size_of::<MlxKeyValueTransactionBranch>(),
            size_of::<(&Self,&HostMetadataFunding,Option<&RealtimeCopyContext<'_>>)>(),
            size_of::<Option<RealtimeCopyPlan<'_>>>(),size_of::<CopySources>(),
            size_of::<Result<MlxKeyValueTransactionBranch,Error>>(),size_of::<Failure>(),
            BackendFailure::source_retention_peak_bytes::<Failure>()?];sum(&parts)
    }
    /// Complete host branching cost, excluding the separately counted native
    /// copy role and the caller's already-retained source state itself.
    pub(crate) fn host_bytes(&self)->Option<usize> {
        let count=self.source.layers.len();
        let aliases=self.arrays.checked_sub(self.copies)?;
        let mut bytes=self.fixed_bytes()?.checked_add(self.paged_host)?
            .checked_add(self.reporting_host)?.checked_add(self.reporting_inspection.checked_mul(2)?)?
            .checked_add(self.paged_inspection.checked_mul(2)?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<MlxKeyValueLayerState>(count)?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<Option<PagedKeyValueTransactionCheckpoint>>(count)?)?
            .checked_add(eredu_runtime::HostSlotTable::<MlxKeyValueLayerState>::host_source_control_bytes()?)?
            .checked_add(self.source.inference_retention.host_clone_bytes()?)?
            .checked_add(aliases.checked_mul(alias_bytes()?)?)?;
        for layer in self.source.layers.slots() {
            if let MlxKeyValueLayerState::Device(cache)=layer {
                bytes=bytes.checked_add(cache.checkpoint_clone_control_bytes::<Error>()?)?;
            }
        }
        if self.copies!=0 {
            bytes=bytes.checked_add(WorkspaceContext::metadata_vec_bytes::<Array>(self.arrays)?)?
                .checked_add(self.arrays.checked_mul(alias_bytes()?)?)?
                .checked_add(StreamCopyPlan::<OriginalRealtimeBudgetCustody>::constructor_storage_bytes().ok()?)?;
        }
        Some(bytes)
    }
    pub(crate) fn copy_control_bytes(&self,plan:&RealtimeCopyPlan<'_>,runtime:&PreparedInputRuntime)->Option<usize> {
        plan.control_bytes::<CopySources,MlxKeyValueTransactionBranch>(runtime)
    }
    fn construct(&self,funding:&HostMetadataFunding,copy:Option<&RealtimeCopyContext<'_>>)
        ->Result<MlxKeyValueTransactionBranch,Error> {
        // Revalidate the exact inspected manager frontiers before constructing
        // any branch. This source check grants no copy or mutation authority.
        funding.reserve_metadata(self.paged_inspection.checked_add(self.reporting_inspection).ok_or_else(overflow)?)
            .map_err(Error::WorkspacePlanning)?;
        let actual=Self::inspect(self.source)?;
        if actual.arrays!=self.arrays || actual.copies!=self.copies || actual.paged_identity!=self.paged_identity
            || actual.paged_host!=self.paged_host || actual.paged_inspection!=self.paged_inspection
            || actual.reporting_identity!=self.reporting_identity || actual.reporting_host!=self.reporting_host
            || actual.reporting_inspection!=self.reporting_inspection {
            return Err(mismatch());
        }
        for cache in paged_managers(self.source) {
            let future=reporting_layers(self.source,cache.manager());
            funding.reserve_metadata(CacheResidencyManager::realtime_reporting_inspection_controls(&future)
                .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)?;
            let reporting=cache.manager().inspect_realtime_reporting(future.clone())
                .map_err(|cause|Error::StorageSource(BackendFailure::from_error(cause)))?;
            cache.manager().prepare_realtime_reporting(reporting,future,funding)?;
        }
        let count=self.source.layers.len();
        let mut layers=funding.metadata_vec(count).map_err(Error::Neural)?;
        let mut rollback=funding.metadata_vec(count).map_err(Error::Neural)?;
        let mut copied=0usize;let mut paged=false;let mut built_hash=Sha256::new();
        for layer in self.source.layers.slots() {
            let mut checkpoint=None;
            layers.push(match layer {
                MlxKeyValueLayerState::Stateless=>MlxKeyValueLayerState::Stateless,
                MlxKeyValueLayerState::Device(cache)=>{
                    funding.reserve_metadata(cache.checkpoint_clone_control_bytes::<Error>().ok_or_else(overflow)?)
                        .map_err(Error::WorkspacePlanning)?;
                    MlxKeyValueLayerState::Device(cache.checkpoint_clone_with(&mut |source,requires_copy| {
                        if requires_copy {
                            let copy=copy.ok_or_else(mismatch)?;
                            copy.retain_source(source.source())?;
                            copied=copied.checked_add(1).ok_or_else(overflow)?;
                            copy.copy(source)
                        } else {alias(source.source(),funding)}
                    })?)},
                MlxKeyValueLayerState::Paged(cache)=>{
                    funding.reserve_metadata(PagedKeyValueCache::realtime_branch_inspection_controls().ok_or_else(overflow)?)
                        .map_err(Error::WorkspacePlanning)?;
                    let source=cache.inspect_realtime_branch()?;
                    built_hash.update(source.identity);
                    let (branch,receipt)=cache.prepare_realtime_branch(source,funding,&mut |array|alias(array,funding))?;
                    checkpoint=Some(receipt);paged=true;MlxKeyValueLayerState::Paged(branch)
                },
            });rollback.push(checkpoint);
        }
        let built_identity:[u8;32]=built_hash.finalize().into();
        if copied!=self.copies || built_identity!=self.paged_identity {return Err(mismatch());}
        let state=MlxKeyValueState {
            layout:self.source.layout.clone(),global_layer_start:self.source.global_layer_start,
            layers:eredu_runtime::HostSlotTable::from_boxed_with_host_source(layers.into_boxed_slice(),funding)
                .map_err(Error::StorageSource)?,
            paged_transaction_branch:paged,
            inference_retention:self.source.inference_retention.clone_with_host_source(funding)
                .map_err(Error::StorageSource)?,
        };
        Ok(MlxKeyValueTransactionBranch{state,paged_rollback:rollback})
    }
    /// Uses the existing state field assembly and isolated numerical worker.
    /// Copy completion is established here, before the scheduler returns a branch.
    pub(crate) fn prepare(self,copy:Option<RealtimeCopyPlan<'_>>,claim:Option<OriginalRealtimeNative>,
        runtime:&PreparedInputRuntime,stream:&Stream,funding:&HostMetadataFunding,
        timeout:Option<Duration>)->Result<MlxKeyValueTransactionBranch,BackendFailure> {
        funding.reserve_metadata(self.fixed_bytes().ok_or_else(||overflow().into_backend_failure())?)
            .map_err(|cause|Error::WorkspacePlanning(cause).into_backend_failure())?;
        let fail=|cause|BackendFailure::from_error(Failure{cause,funding:funding.clone()});
        match (self.copies,copy,claim) {
            (0,None,None)=>self.construct(funding,None).map_err(fail),
            (copies,Some(plan),Some(claim)) if copies!=0=>{
                let controls=StreamCopyPlan::<OriginalRealtimeBudgetCustody>::constructor_storage_bytes()
                    .map_err(|cause|fail(Error::StorageSource(BackendFailure::from_error(cause))))?;
                funding.reserve_metadata(controls).map_err(|cause|fail(Error::WorkspacePlanning(cause)))?;
                let mut arrays=funding.metadata_vec(self.arrays).map_err(|cause|fail(Error::Neural(cause)))?;
                let mut failed=None;
                self.visit(&mut |array,_|if failed.is_none(){match alias(array,funding){
                    Ok(array)=>arrays.push(array),Err(cause)=>failed=Some(cause)}});
                if let Some(cause)=failed{return Err(fail(cause));}
                let copied_stream=StreamCopyPlan::capture(stream)
                    .map_err(|cause|fail(Error::StorageSource(BackendFailure::from_error(cause))))?
                    .realize(claim.budget_custody())
                    .map_err(|cause|fail(Error::StorageSource(BackendFailure::from_error(cause))))?;
                let sources=CopySources{_arrays:arrays,_stream:copied_stream,_funding:funding.clone()};
                plan.run(sources,claim,runtime,funding,timeout,&mut copy_callback(Some(&self),Some(funding)))
            },
            _=>Err(fail(mismatch())),
        }
    }
}

use eredu_nn::workspace::WorkspaceMetadataAllocation;
