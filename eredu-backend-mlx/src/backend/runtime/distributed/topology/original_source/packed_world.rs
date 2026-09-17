//! Exact logical selection and its existing packed native-world operation.
use super::*;
use super::parallel::OriginalParallelSource;
use crate::backend::runtime::distributed::group::LogicalPackedWorldPlan;
use safemlx::{Array,Dtype,distributed::{GroupWorkerOperation,
    GroupCpuLayoutStorage,GroupCpuCompletionStorage,GroupCpuCompletionLayoutStorage,
    OwnedGroupCpuCompletionLayoutStorage}};
fn overflow()->Error{Error::WorkspacePlanning(WorkspaceMetadataFundingError::Overflow)}
fn reserve(funding:&WorkspaceMetadataFunding,parts:&[usize])->Result<(),Error>{
    funding.reserve_metadata(parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
        .ok_or_else(overflow)?).map_err(Error::WorkspacePlanning)
}
impl OriginalCommunicationSource<'_>{
    pub(crate) fn packed_world_plan(&self,order:usize)->Result<Option<LogicalPackedWorldPlan<'_>>,Error>{
        reserve(&self.funding,&[size_of::<(&Self,usize)>(),size_of::<Option<LogicalPackedWorldPlan<'_>>>(),
            size_of::<Result<Option<LogicalPackedWorldPlan<'_>>,Error>>(),
            size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>(),
            failure_control_bytes().ok_or_else(overflow)?])?;
        self.validate()?;
        let(group,_,wave)=self.group(order).ok_or_else(||failure(Cause::Identity,&self.source,&self.funding))?;
        if !self.matches_group(order,group){return Err(failure(Cause::Identity,&self.source,&self.funding));}
        let plan=group.logical_packed_world_plan().map_err(|cause|failure(Cause::LogicalExchange(cause),&self.source,&self.funding))?;
        if plan.is_some() && (!wave || !group.native_group().shares_native_handle(self.world().native_group())) {
            return Err(failure(Cause::LogicalWorldTransport,&self.source,&self.funding));
        }
        Ok(plan)
    }

}
/// Original logical group plus its exact selected internal native-world Sum.
/// The closed source keeps full completion and physical backing facts separate
/// from permission; no input or observer belongs to this immutable quote.
pub(crate) struct OwnedPackedWorldCompletion {
    native:OwnedPackedWorldSource,
    owner:OriginalParallelSource,
}
/// Closed physical-world source shared by model tensors and control votes.
/// Its owned native layout retains the actual physical Group; binding still
/// requires the original selected communication source and funding account.
pub(crate) struct OwnedPackedWorldSource {
    native:OwnedGroupCpuCompletionLayoutStorage,
    order:usize,
    backing:usize,
    source:RetainedCommunicationSource,
    funding:WorkspaceMetadataFunding,
}
impl OriginalParallelSource {
    pub(crate) fn packed_world_completion(&self,order:usize,shape:&[i32],dtype:Dtype)->Result<OwnedPackedWorldCompletion,Error>{
        let source=self.communication_source()?;
        reserve(self.funding(),&[size_of::<OwnedPackedWorldCompletion>(),size_of::<Result<OwnedPackedWorldCompletion,Error>>(),
            size_of::<(&Self,usize,&[i32],Dtype)>()])?;
        let native=source.packed_world_completion_source(order,shape,dtype,self.initialized_runtime())?;
        Ok(OwnedPackedWorldCompletion{native,owner:self.clone()})
    }
}
impl OriginalCommunicationSource<'_> {
    pub(crate) fn packed_world_completion_source(&self,order:usize,shape:&[i32],dtype:Dtype,runtime:&safemlx::PreparedInputRuntime)->Result<OwnedPackedWorldSource,Error>{
        reserve(self.funding(),&[size_of::<OwnedPackedWorldSource>(),size_of::<Result<OwnedPackedWorldSource,Error>>(),
            size_of::<GroupCpuLayoutStorage<'_>>(),size_of::<GroupCpuCompletionLayoutStorage<'_>>(),
            size_of::<OwnedGroupCpuCompletionLayoutStorage>(),
            size_of::<(&Self,usize,&[i32],Dtype,&safemlx::PreparedInputRuntime)>(),failure_control_bytes().ok_or_else(overflow)?])?;
        let plan=self.packed_world_plan(order)?.ok_or_else(||failure(Cause::Identity,self.source(),self.funding()))?;
        if plan.group().size()!=self.group(order).ok_or_else(||failure(Cause::Identity,self.source(),self.funding()))?.1.members().len()
            || shape.first().copied()!=i32::try_from(plan.world_size()).ok(){
            return Err(failure(Cause::Identity,self.source(),self.funding()));
        }
        let persistent=self.world_persistent()?;
        if persistent.native().has_unqualified_storage() || !persistent.native().is_for(self.world().native_group())
            || !persistent.source().same_source(self.source()) {
            return Err(failure(Cause::Resource,self.source(),self.funding()));
        }
        reserve(self.funding(),&[self.world().native_group().cpu_layout_storage_control_bytes().ok_or_else(overflow)?])?;
        let native=self.world().native_group().cpu_layout_storage(shape,dtype,GroupWorkerOperation::Sum)
            .map_err(|_|failure(Cause::NativeLayout,self.source(),self.funding()))?;
        reserve(self.funding(),&[native.completion_layout_control_bytes().ok_or_else(overflow)?,native.backing_control_bytes().ok_or_else(overflow)?])?;
        let backing=native.backing_capacity(runtime)
            .map_err(|cause|failure(Cause::Buffer(cause),self.source(),self.funding()))?;
        let native=native.with_completion_layout().map_err(|_|failure(Cause::NativeLayout,self.source(),self.funding()))?;
        reserve(self.funding(),&[native.ownership_control_bytes().ok_or_else(overflow)?])?;
        let native=native.try_into_owned().map_err(|_|failure(Cause::Resource,self.source(),self.funding()))?;
        Ok(OwnedPackedWorldSource{native,order,backing,source:self.source().clone(),funding:self.funding().clone()})
    }
}
impl OwnedPackedWorldCompletion {
    pub(crate) fn graph_capacity(&self)->usize{self.native.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize{self.native.record_capacity()}
    pub(crate) fn backing_capacity(&self)->usize{self.native.backing_capacity()}
    pub(crate) fn births(&self)->usize{self.native.births()}
    pub(crate) fn shape(&self)->&[i32]{self.native.shape()}
    pub(crate) fn bind_actual<'a>(&'a self,source:&'a OriginalCommunicationSource<'_>,input:&'a Array)
        ->Result<OriginalCommunicationCompletedOperation<'a>,Error>{
        reserve(&self.native.funding,&[size_of::<(&Self,&OriginalCommunicationSource<'_>,&Array)>(),
            size_of::<Result<OriginalCommunicationCompletedOperation<'a>,Error>>(),failure_control_bytes().ok_or_else(overflow)?])?;
        if !self.owner.declaration_source().same_source(source.source()){
            return Err(failure(Cause::Identity,&self.native.source,&self.native.funding));
        }
        self.native.bind_actual(source,input)
    }
}
impl OwnedPackedWorldSource {
    pub(crate) fn graph_capacity(&self)->usize{self.native.graph_capacity()}
    pub(crate) fn record_capacity(&self)->usize{self.native.record_capacity()}
    pub(crate) fn backing_capacity(&self)->usize{self.backing}
    pub(crate) fn births(&self)->usize{self.native.operation().evaluation().logical_backing_population().0}
    pub(crate) fn shape(&self)->&[i32]{self.native.operation().shape()}
    pub(crate) fn bind_actual<'a>(&'a self,source:&'a OriginalCommunicationSource<'_>,input:&'a Array)
        ->Result<OriginalCommunicationCompletedOperation<'a>,Error>{
        reserve(&self.funding,&[size_of::<(&Self,&OriginalCommunicationSource<'_>,&Array)>(),
            size_of::<GroupCpuCompletionStorage<'a>>(),size_of::<OriginalCommunicationCompletedOperation<'a>>(),
            size_of::<Result<OriginalCommunicationCompletedOperation<'a>,Error>>(),
            self.native.binding_control_bytes().ok_or_else(overflow)?,failure_control_bytes().ok_or_else(overflow)?])?;
        source.validate()?;
        if !self.source.same_source(source.source()) || !self.funding.same_account(source.funding())
            || source.packed_world_plan(self.order)?.is_none(){
            return Err(failure(Cause::Identity,&self.source,&self.funding));
        }
        let persistent=source.world_persistent()?;
        if persistent.native().has_unqualified_storage() || !persistent.native().is_for(source.world().native_group()){
            return Err(failure(Cause::Resource,&self.source,&self.funding));
        }
        let native=self.native.bind_actual_in_group(source.world().native_group(),input)
            .map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OriginalCommunicationCompletedOperation::from_retained_layout(native,persistent,self.source.clone(),self.funding.clone()))
    }
}
