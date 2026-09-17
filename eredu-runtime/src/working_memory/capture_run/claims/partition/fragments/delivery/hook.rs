//! Move the actual host/receipt owner out while a backend borrows its native state.
use super::*;

/// No transport borrow remains in the local hook. Its bank and receipt are the
/// actual originals; splitting their borrows permits existing native callbacks.
#[derive(Debug)]
pub(crate) struct PartitionLocalCaptureHook {
    bank:PreparedPartitionFragmentDestinations,
    receipt:PartitionCaptureReceiptPlan,
    metadata:WorkspaceMetadataFunding,
    routed:Option<routed::Invocation>,
    routed_source:Option<crate::capture::partition::PreparedPartitionRoutedLocalSource>,
}
impl PartitionLocalCaptureHook {
    pub(crate) fn bind_routed_source(&mut self,source:crate::capture::partition::PreparedPartitionRoutedLocalSource) {
        assert!(self.routed_source.is_none()&&self.routed.is_none(),"original hook has no active sparse source");
        self.routed_source=Some(source);
    }
    pub(crate) fn source(&self)->&SharedCapturePlan {&self.bank.source}
    pub(crate) fn funding(&self)->&WorkspaceMetadataFunding {&self.metadata}
    pub(crate) fn parts(&mut self)->(&PartitionCaptureReceiptPlan,&mut PreparedPartitionFragmentDestinations) {
        (&self.receipt,&mut self.bank)
    }
}
/// Idle transport source with no receipt authority or exchange method. It must
/// consume the same original host owner before normal protocol work resumes.
pub(crate) struct PartitionCaptureHookContinuation<'t,T:PartitionCaptureTransport> {
    exchange:PartitionCaptureExchange<'t,T>,
    ranks:Vec<PartitionCaptureRankSource>,
    evidence:PreparedPartitionCaptureEvidence,
    dtype:TensorDtype,
    source:SharedCapturePlan,
    custody:CaptureTensorCustody,
    metadata:WorkspaceMetadataFunding,
}
impl<T:PartitionCaptureTransport> fmt::Debug for PartitionCaptureHookContinuation<'_,T> {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result {
        f.debug_struct("PartitionCaptureHookContinuation").field("ranks",&self.ranks).finish_non_exhaustive()
    }
}
/// A refused return retains both actual host accounts and the rejected receipt.
#[derive(Debug,thiserror::Error)]
#[error("partition local hook does not belong to its original continuation")]
pub(crate) struct PartitionCaptureHookReturnError {
    _hook:PartitionLocalCaptureHook,
    _source:SharedCapturePlan,
    _expected:CaptureTensorCustody,
    _metadata:WorkspaceMetadataFunding,
}
impl<'t,T:PartitionCaptureTransport> PreparedPartitionFragmentDelivery<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    /// Taking the entire owner prevents an outstanding host claim from also
    /// borrowing the backend that owns this program. No receipt or payload is copied and
    /// no protocol phase can run until the continuation is reunited.
    pub(crate) fn into_local_hook(self)->(PartitionCaptureHookContinuation<'t,T>,PartitionLocalCaptureHook) {
        let Self{mut exchange,bank,ranks,evidence,dtype,metadata}=self;
        let receipt=exchange.take_hook_receipt().expect("one unexchanged original receipt");
        let source=bank.source.clone();let custody=bank.custody.share_scheduled();
        (PartitionCaptureHookContinuation{exchange,ranks,evidence,dtype,source,custody,metadata:metadata.clone()},
            PartitionLocalCaptureHook{bank,receipt,metadata,routed:None,routed_source:None})
    }
}
impl<'t,T:PartitionCaptureTransport> PartitionCaptureHookContinuation<'t,T>
where T::Error:Send+Sync+'static,<T::Completion as Completion>::Error:Send+Sync+'static {
    pub(crate) fn resume(self,hook:PartitionLocalCaptureHook)
        ->Result<PreparedPartitionFragmentDelivery<'t,T>,PartitionCaptureHookReturnError> {
        self.resume_routed(hook).map(|(delivery,_)|delivery)
    }
    pub(crate) fn resume_routed(self,hook:PartitionLocalCaptureHook)
        ->Result<(PreparedPartitionFragmentDelivery<'t,T>,Option<crate::capture::partition::PreparedPartitionRoutedLocalSource>),PartitionCaptureHookReturnError> {
        let Self{mut exchange,ranks,evidence,dtype,source,custody,metadata}=self;
        let error=|hook|PartitionCaptureHookReturnError{_hook:hook,_source:source.clone(),_expected:custody.share_scheduled(),_metadata:metadata.clone()};
        if hook.routed.is_some() || !custody.same_schedule(&hook.bank.custody) || !source.same_storage(&hook.bank.source)
            || !metadata.same_account(&hook.metadata)
            || !hook.bank.matches(&hook.receipt) || !evidence.matches_receipt(&hook.receipt)
            || hook.bank.allowance.local_rank()!=exchange.local_rank() {
            return Err(error(hook));
        }
        let PartitionLocalCaptureHook{bank,receipt,metadata:hook_metadata,routed:_,routed_source}=hook;
        if let Err(receipt)=exchange.restore_hook_receipt(receipt) {return Err(error(PartitionLocalCaptureHook{bank,receipt,metadata:hook_metadata,routed:None,routed_source}));}
        Ok((PreparedPartitionFragmentDelivery{exchange,bank,ranks,evidence,dtype,metadata},routed_source))
    }
}
pub(super) fn control_bytes<T:PartitionCaptureTransport>()->Option<usize> {
    let parts=[size_of::<Option<crate::capture::partition::PreparedPartitionRoutedLocalSource>>(),size_of::<(PreparedPartitionFragmentDelivery<'_,T>,Option<crate::capture::partition::PreparedPartitionRoutedLocalSource>)>(),size_of::<PartitionLocalCaptureHook>()*2,size_of::<PartitionCaptureHookContinuation<'_,T>>()*2,
        size_of::<(PartitionCaptureHookContinuation<'_,T>,PartitionLocalCaptureHook)>(),
        size_of::<Result<PreparedPartitionFragmentDelivery<'_,T>,PartitionCaptureHookReturnError>>(),
        size_of::<PartitionCaptureHookReturnError>(),size_of::<Result<(),PartitionCaptureReceiptPlan>>(),
        size_of::<Option<PartitionCaptureReceiptPlan>>(),size_of::<(&mut PartitionLocalCaptureHook,)>(),
        size_of::<(&PartitionCaptureReceiptPlan,&mut PreparedPartitionFragmentDestinations)>(),
        size_of::<(PartitionCaptureHookContinuation<'_,T>,PartitionLocalCaptureHook)>(),
        size_of::<(&SharedCapturePlan,&CaptureTensorCustody,&WorkspaceMetadataFunding)>(),
        size_of::<(&mut PartitionCaptureExchange<'_,T>,PartitionCaptureReceiptPlan)>(),
        size_of::<std::iter::Zip<std::slice::Iter<'_,u32>,std::slice::ChunksExact<'_,u8>>>(),
        size_of::<Option<u32>>(),size_of::<Result<u32,std::num::ParseIntError>>(),
        size_of::<SharedCapturePlan>(),size_of::<CaptureTensorCustody>(),size_of::<WorkspaceMetadataFunding>()];
    parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
}

mod observe;
pub(crate) use observe::PartitionLocalCaptureFailure;

mod invocation;

mod routed;
