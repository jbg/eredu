//! Actual scheduler branch sourced from its unchanged canonical state.
use super::*;
use super::original_execution::PreparedOriginalRealtimeFrame;
use crate::backend::{array_copy::RealtimeCopyPlan,
    runtime::cache::state::RealtimeKvBranchPlan};
use eredu_core::{BackendFailure,HostMetadataFunding,HostMetadataFundingError};
use eredu_nn::workspace::WorkspaceMetadataFunding;
use eredu_runtime::{RealtimePayloadState,RealtimeSessionState,Sampler,
    working_memory::{OriginalRealtimeNative,RealtimeNativeRequirements,WorkingMemoryError}};
use safemlx::PreparedInputRuntime;
use std::{mem::{size_of,size_of_val},time::Duration};

type Payload=RealtimePayloadState<MlxKeyValueState,MlxTensor>;
type PayloadBranch=RealtimePayloadBranch<MlxKeyValueTransactionBranch,MlxTensor>;
/// Canonical scheduler state carrying an admitted native frame preparation.
pub type OriginalRealtimeSessionState=RealtimeSessionState<RealtimePayloadState<MlxKeyValueState,MlxTensor>,GenerationSampler,
    RandomState,MlxRealtimeCompletion,PreparedOriginalRealtimeFrame>;
type AliasFn=fn(&MlxTensor,&HostMetadataFunding)->Result<MlxTensor,BackendFailure>;
type SamplerFn=fn(&GenerationSampler,&HostMetadataFunding)->Result<GenerationSampler,BackendFailure>;
type RandomFn=fn(&RandomState,&HostMetadataFunding)->Result<RandomState,BackendFailure>;

pub(super) struct RealtimeBranchSource<'a> {
    canonical:&'a OriginalRealtimeSessionState,
    kv:RealtimeKvBranchPlan<'a>,copy:Option<RealtimeCopyPlan<'a>>,
    runtime:&'a PreparedInputRuntime,stream:&'a Stream,
}
#[derive(Debug,thiserror::Error)]
#[error("realtime prepared branch: {cause}")]
struct Failure {#[source] cause:BackendFailure,_funding:HostMetadataFunding}
pub(super) fn alias_bytes()->Option<usize> {MlxTensor::host_clone_bytes()}
pub(super) fn alias(value:&MlxTensor,funding:&HostMetadataFunding)->Result<MlxTensor,BackendFailure> {
    value.clone_with_host_source(funding)
}
fn sampler(value:&GenerationSampler,funding:&HostMetadataFunding)->Result<GenerationSampler,BackendFailure> {
    <GenerationSampler as Sampler<MlxSamplingBackend>>::clone_with_host_source(value,funding).map_err(Into::into)
}
fn random(value:&RandomState,funding:&HostMetadataFunding)->Result<RandomState,BackendFailure> {
    value.clone_with_host_source(funding)
}
fn kv_callback<'a>(source:Option<RealtimeKvBranchPlan<'a>>,copy:Option<RealtimeCopyPlan<'a>>,
    claim:Option<OriginalRealtimeNative>,runtime:Option<&'a PreparedInputRuntime>,
    stream:Option<&'a Stream>,timeout:Option<Duration>)
    ->impl FnOnce(&MlxKeyValueState,&HostMetadataFunding)->Result<MlxKeyValueTransactionBranch,BackendFailure>+'a {
    move |actual,funding| {
        let source=source.ok_or(HostMetadataFundingError::Unavailable)?;
        if !source.matches_source(actual){return Err(Error::PrefillControl(WorkingMemoryError::IdentityMismatch).into_backend_failure());}
        let funding:WorkspaceMetadataFunding=funding.clone().into();
        source.prepare(copy,claim,runtime.ok_or(HostMetadataFundingError::Unavailable)?,
            stream.ok_or(HostMetadataFundingError::Unavailable)?,&funding,timeout)
    }
}
fn payload_callback<K>(source:K)
    ->impl FnOnce(&Payload,&HostMetadataFunding)->Result<PayloadBranch,BackendFailure>
where K:FnOnce(&MlxKeyValueState,&HostMetadataFunding)->Result<MlxKeyValueTransactionBranch,BackendFailure> {
    move |payload,funding|payload.branch_with_host_source(funding,source,alias as AliasFn)
}
fn payload_controls<F>(payload:&Payload,_source:&F)->Option<usize> {payload.branch_host_bytes::<F,AliasFn>()}
fn session_controls<F>(canonical:&OriginalRealtimeSessionState,_source:&F)->Option<usize> {
    canonical.branch_host_bytes::<F,SamplerFn,RandomFn>()
}
impl<'a> RealtimeBranchSource<'a> {
    /// No branch, clone or native work is made during source inspection.
    pub(super) fn inspect(canonical:&'a OriginalRealtimeSessionState,runtime:&'a PreparedInputRuntime,
        stream:&'a Stream)->Result<Self,Error> {
        let kv=RealtimeKvBranchPlan::inspect(canonical.generation().model_state().model_state())?;
        let copy=kv.copy_plan(runtime,stream)?;
        Ok(Self{canonical,kv,copy,runtime,stream})
    }
    fn fixed_controls(&self)->Option<usize> {
        let callback=kv_callback(None,None,None,None,None,None);
        let payload=payload_callback(callback);
        let parts=[size_of::<Self>(),size_of::<Result<Self,Error>>(),size_of_val(&payload),
            size_of::<Result<MlxFrameSessionBranch<PreparedOriginalRealtimeFrame>,BackendFailure>>(),
            size_of::<Failure>(),size_of::<Option<OriginalRealtimeNative>>(),
            size_of::<WorkspaceMetadataFunding>(),size_of::<HostMetadataFunding>(),
            size_of::<(&mut PreparedOriginalRealtimeFrame,Option<Duration>)>(),
            BackendFailure::source_retention_peak_bytes::<Failure>()?];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    /// Actual shells, history and descriptor copies. The optional numerical
    /// copy phase's native controls are a separate returned source.
    pub(super) fn host_bytes(&self)->Option<usize> {
        let generation=self.canonical.generation();let payload=generation.model_state();
        let kv=kv_callback(None,None,None,None,None,None);
        let payload_bytes=payload_controls(payload,&kv)?;
        let callback=payload_callback(kv);
        let mut bytes=self.fixed_controls()?.checked_add(session_controls(self.canonical,&callback)?)?
            .checked_add(payload_bytes)?.checked_add(self.kv.host_bytes()?)?
            .checked_add(payload.payload_history().len().checked_mul(alias_bytes()?)?)?;
        for sampler in generation.samplers(){bytes=bytes.checked_add(sampler.host_clone_bytes()?)?;}
        if generation.random_state().is_some(){bytes=bytes.checked_add(RandomState::host_clone_bytes()?)?;}
        Some(bytes)
    }
    pub(super) fn preparation(&self)->Result<Option<(RealtimeNativeRequirements,usize)>,Error> {
        self.copy.as_ref().map(|copy|Ok((copy.requirements().map_err(Error::PrefillControl)?,
            self.kv.copy_control_bytes(copy,self.runtime).ok_or(Error::PrefillControl(WorkingMemoryError::Overflow))?)))
            .transpose()
    }
    /// Creates the scheduler branch only after complete frame admission.
    /// Required native state copies finish before another item is prepared.
    pub(super) fn prepare(self,prepared:&mut PreparedOriginalRealtimeFrame,timeout:Option<Duration>)
        ->Result<MlxFrameSessionBranch<PreparedOriginalRealtimeFrame>,BackendFailure> {
        let funding:HostMetadataFunding=prepared.funding().clone().into();
        let result=(|| {
            funding.reserve_metadata(self.fixed_controls().ok_or(HostMetadataFundingError::Overflow)?)?;
            let claim=if self.copy.is_some(){Some(prepared.claim_preparation().map_err(Error::into_backend_failure)?)}else{None};
            let callback=kv_callback(Some(self.kv),self.copy,claim,Some(self.runtime),Some(self.stream),timeout);
            self.canonical.branch_with_host_source(&funding,payload_callback(callback),sampler as SamplerFn,random as RandomFn)
        })();
        result.map_err(|cause|BackendFailure::from_error(Failure{cause,_funding:funding}))
    }
}
