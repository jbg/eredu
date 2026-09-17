//! The canonical host header is copied by the existing immutable input worker.
//! Its exact bytes stay beside the logical tensor for later wire validation.
use super::*;
use crate::MlxTensor;
use eredu_runtime::{PreparedBoundaryFrames,RoleExactBoundaryValue};
use safemlx::{CompletedOwnedHostCopy,OriginalScopeObserver,OwnedHostCopyBuffer,
    OwnedHostCopyPlan,PreparedInputRuntime,PreparedSubmissionGraphQuota};

/// Actual completed header plus its unmodified canonical bytes and payload.
/// Native storage retires before the expected header and logical tensor.
pub(crate) struct OriginalBoundaryHeader {
    header:CompletedOwnedHostCopy,
    value:RoleExactBoundaryValue<MlxTensor>,
}
impl OriginalBoundaryHeader {
    pub(crate) fn header(&self)->&CompletedOwnedHostCopy{&self.header}
    pub(crate) fn value(&self)->&RoleExactBoundaryValue<MlxTensor>{&self.value}
    pub(crate) fn into_parts(self)->(CompletedOwnedHostCopy,RoleExactBoundaryValue<MlxTensor>){
        (self.header,self.value)
    }
}
/// Exact route/order source; all native values retire before its source and H.
/// This grants no frame operation, route submission or completion authority.
pub(crate) struct OriginalBoundaryHeaders {
    values:Vec<OriginalBoundaryHeader>,
    route:CommunicationRouteId,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
impl OriginalBoundaryHeaders {
    pub(crate) fn values(&self)->&[OriginalBoundaryHeader]{&self.values}
    pub(crate) fn route(&self)->CommunicationRouteId{self.route}
    pub(crate) fn source(&self)->&RetainedCommunicationSource{&self.source}
    pub(crate) fn funding(&self)->&WorkspaceMetadataFunding{&self.funding}
    pub(crate) fn into_parts(self)->(Vec<OriginalBoundaryHeader>,RetainedCommunicationSource,WorkspaceMetadataFunding){
        (self.values,self.source,self.funding)
    }
}
// The source arena holds only this paid, array-free custody. It cannot keep the
// request, observer, source table, frame vector or native payload alive.
#[derive(Clone)]
struct Custody { source:RetainedCommunicationSource, funding:WorkspaceMetadataFunding }
struct Bytes<'a>(&'a [u8]);
impl OwnedHostCopyBuffer<u8> for Bytes<'_> {
    fn as_slice(&self)->&[u8]{self.0}
    fn capacity(&self)->usize{self.0.len()}
}
impl OriginalCommunicationSource<'_> {
    /// Consume only a frame packet minted by the shared canonical writer and
    /// authenticate the exact stored route. Caller-provided bytes/counts cannot
    /// independently create an original header source through this entry.
    pub(crate) fn prepare_boundary_headers(&self,frames:PreparedBoundaryFrames<MlxTensor>,
        route:&CommunicationRouteRealization,runtime:&PreparedInputRuntime,
        observer:&OriginalScopeObserver)->Result<OriginalBoundaryHeaders,Error>{
        reserve(&self.funding,&[
            size_of::<OriginalBoundaryHeaders>(),size_of::<PreparedBoundaryFrames<MlxTensor>>(),
            size_of::<Result<OriginalBoundaryHeaders,Error>>(),
            size_of::<(&Self,&CommunicationRouteRealization,&PreparedInputRuntime,&OriginalScopeObserver)>(),
            size_of::<Vec<OriginalBoundaryHeader>>(),size_of::<Vec<RoleExactBoundaryValue<MlxTensor>>>(),
            size_of::<std::vec::IntoIter<RoleExactBoundaryValue<MlxTensor>>>(),
            size_of::<RoleExactBoundaryValue<MlxTensor>>(),size_of::<Result<(),std::collections::TryReserveError>>(),
            size_of::<Option<usize>>(),size_of::<std::alloc::Layout>(),
            OriginalScopeObserver::control_bytes().ok_or_else(overflow)?,
            failure_control_bytes().ok_or_else(overflow)?,
        ])?;
        self.validate()?;
        let fail=|cause|failure(cause,&self.source,&self.funding);
        let order=self.source.manifest().routes().iter().position(|item|item.id()==frames.route())
            .ok_or_else(||fail(Cause::Identity))?;
        if !frames.source().same_source(&self.source) || !frames.funding().same_account(&self.funding)
            || frames.route()!=route.descriptor().id() || !self.matches_route(order,route)
            || route.group().is_none() || route.endpoint().is_none() {
            return Err(fail(Cause::Identity));
        }
        let current=OriginalScopeObserver::require_current().map_err(|cause|fail(Cause::Native(cause)))?;
        if !current.same_scope(observer){return Err(fail(Cause::Identity));}
        let selected_route=frames.route();
        let (values,source,funding)=frames.into_parts();
        reserve(&funding,&[std::alloc::Layout::array::<OriginalBoundaryHeader>(values.len())
            .map_err(|_|overflow())?.size()])?;
        let mut outputs=Vec::new();outputs.try_reserve_exact(values.len())
            .map_err(|cause|fail(Cause::BoundaryDestination(cause)))?;
        for value in values {
            let header=upload(self,runtime,value.header(),observer)?;
            outputs.push(OriginalBoundaryHeader{header,value});
        }
        Ok(OriginalBoundaryHeaders{values:outputs,route:selected_route,source,funding})
    }
}
fn upload(source:&OriginalCommunicationSource<'_>,runtime:&PreparedInputRuntime,
    bytes:&[u8],observer:&OriginalScopeObserver)->Result<CompletedOwnedHostCopy,Error>{
    let funding=source.funding();
    reserve(funding,&[
        size_of::<(&OriginalCommunicationSource<'_>,&PreparedInputRuntime,&[u8],&OriginalScopeObserver)>(),
        size_of::<[i32;1]>(),size_of::<Result<i32,std::num::TryFromIntError>>(),
        size_of::<OwnedHostCopyPlan<'_,u8>>(),size_of::<Result<OwnedHostCopyPlan<'_,u8>,safemlx::OwnedHostCopyCause>>(),
        size_of::<Custody>(),size_of::<safemlx::OwnedHostCopyFacts>(),
        size_of::<Result<PreparedSubmissionGraphQuota<Custody>,safemlx::SubmissionGraphQuotaError<Custody>>>(),
        size_of::<Result<CompletedOwnedHostCopy,Error>>(),
        CompletedOwnedHostCopy::control_bytes().ok_or_else(overflow)?,
        failure_control_bytes().ok_or_else(overflow)?,
    ])?;
    let fail=|cause|failure(cause,source.source(),funding);
    let len=i32::try_from(bytes.len()).map_err(|_|fail(Cause::BoundaryHeader{
        cause:safemlx::OwnedHostCopyCause::Invalid,native:None}))?;
    if len==0{return Err(fail(Cause::BoundaryHeader{cause:safemlx::OwnedHostCopyCause::Invalid,native:None}));}
    let shape=[len];
    let plan=OwnedHostCopyPlan::<u8>::new(runtime,&shape,bytes.len())
        .map_err(|cause|fail(Cause::BoundaryHeader{cause,native:None}))?;
    // H owns the actual immutable source arena and backing. This is separate
    // from the later frame graph/data admission; no scalar Q credit is added.
    reserve(funding,&[plan.control_bytes_with_buffer::<Custody,Bytes<'_>>().ok_or_else(overflow)?,
        plan.facts().backing_bytes()])?;
    let custody=Custody{source:source.source().clone(),funding:funding.clone()};
    let prepared=PreparedSubmissionGraphQuota::try_new(plan.facts().metadata_bytes(),custody)
        .map_err(|error|{let(cause,owner)=error.into_parts();let error=fail(Cause::SourceGraph(cause));drop(owner);error})?;
    let slot=plan.prepare(prepared).map_err(|error|{
        let cause=fail(Cause::BoundaryHeader{cause:error.cause(),native:None});drop(error);cause
    })?;
    let header=slot.try_fill_owned_completed(Bytes(bytes),observer).map_err(|mut error|{
        let cause=fail(Cause::BoundaryHeader{cause:error.cause(),native:error.take_native_source()});
        drop(error);cause
    })?;
    // Preserve the successful native birth rather than accepting a raw Array
    // paired with descriptive byte counts or a different completed source.
    header.observe().map_err(|cause|fail(Cause::Buffer(cause)))?;
    Ok(header)
}
fn overflow()->Error{Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)}
fn reserve(funding:&WorkspaceMetadataFunding,bytes:&[usize])->Result<(),Error>{
    funding.reserve_metadata(bytes.iter().copied().try_fold(size_of_val(bytes),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
