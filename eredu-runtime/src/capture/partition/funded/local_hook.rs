//! Opaque move-only loan from the sealed program to the shared native callback.
use super::*;
use crate::working_memory::PartitionLocalCaptureHook;

/// Actual projected receipt and original fragment destinations, detached from
/// transport while the same backend is mutably borrowed. Only runtime programs
/// can create or unwrap this loan; public collectors cannot forge a destination.
#[derive(Debug)]
pub struct PartitionCaptureLocalHook(PartitionLocalCaptureHook);
#[derive(Debug,thiserror::Error)]
#[error("a projected local hook was returned to a complete-producer program")]
struct RejectedHook { _hook:PartitionLocalCaptureHook }

impl PartitionCaptureLocalHook {
    pub(crate) fn control_bytes()->Option<usize> {
        let parts=[eredu_core::BackendFailure::source_retention_peak_bytes::<RejectedHook>()?,
            size_of::<Self>()*2,size_of::<Option<Self>>(),size_of::<Result<Option<Self>,PartitionCaptureProgramError>>(),
            size_of::<Result<(),PartitionCaptureProgramError>>(),size_of::<RejectedHook>(),
            size_of::<SharedCapturePlan>(),size_of::<HostMetadataFunding>()];
        parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
    }
    pub(crate) fn new(hook:PartitionLocalCaptureHook)->Self {Self(hook)}
    pub(crate) fn into_inner(self)->PartitionLocalCaptureHook {self.0}
    pub(crate) fn observe<T,E:std::error::Error+Send+Sync+'static>(self,
        backend:&mut dyn crate::capture::ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T,
        inference:eredu_core::InferenceGeometry,chunk:u64)->Result<Self,PartitionCaptureProgramError> {
        self.0.observe_prefill_program(backend,value,inference,chunk).map(Self)
    }
    pub(crate) fn observe_invocation<T,E:std::error::Error+Send+Sync+'static>(self,
        backend:&mut dyn crate::capture::ScheduledCaptureBackend<Tensor=T,Error=E>,value:&T)
        ->Result<Self,PartitionCaptureProgramError> {
        self.0.observe_invocation_program(backend,value).map(Self)
    }
    /// Fixed callback controls for the actual sparse local hook transitions.
    pub fn routed_control_bytes<T,E:std::error::Error+Send+Sync+'static>()->Option<usize> {
        PartitionLocalCaptureHook::routed_control_bytes::<T,E>()
    }
    pub(crate) fn begin_routed<T,E:std::error::Error+Send+Sync+'static>(self,
        backend:&mut dyn crate::capture::ScheduledCaptureBackend<Tensor=T,Error=E>,
        invocation:&crate::RoutedUnitInvocation<'_,T>,prefill:Option<(eredu_core::InferenceGeometry,u64)>)
        ->Result<Self,PartitionCaptureProgramError> {
        self.0.begin_routed_program(backend,invocation,prefill).map(Self)
    }
    pub(crate) fn observe_routed<T,E:std::error::Error+Send+Sync+'static>(self,
        backend:&mut dyn crate::capture::ScheduledCaptureBackend<Tensor=T,Error=E>,batch:&crate::RoutedUnitBatch<'_,T>)
        ->Result<Self,PartitionCaptureProgramError> {
        self.0.observe_routed_program(backend,batch).map(Self)
    }
    pub(crate) fn finish_routed<T,E:std::error::Error+Send+Sync+'static>(self,success:bool)
        ->Result<Self,PartitionCaptureProgramError> {
        self.0.finish_routed_program::<T,E>(success).map(Self)
    }
    pub(super) fn reject(self)->PartitionCaptureProgramError {
        let source=self.0.source().clone();let metadata=self.0.funding().clone();
        PartitionCaptureProgramError::local(RejectedHook{_hook:self.0},source,metadata)
    }
}
impl PartitionCaptureProgramError {
    pub(crate) fn local<E:std::error::Error+Send+Sync+'static>(cause:E,
        source:SharedCapturePlan,metadata:HostMetadataFunding)->Self {
        Self{cause:Cause::Local(eredu_core::BackendFailure::from_error(cause)),_source:source,_metadata:metadata}
    }
}
