//! Shared native/cold addressable declaration and exact borrowed callback frame.
use super::*;
use eredu_nn::workspace::{WorkspaceAddressableRegionView,WorkspaceExpertKernel,WorkspaceMetadataError};

// Every generic field is a borrowed tensor/context/observer. There are no
// by-value backend tensors or provider objects in this frame. Both native
// execution and cold source construction use this same layout.
pub(super) struct Callback<'input,'observer,'context,P,T:Tensor> {
    pub request:Option<RoutedExpertRequest<'input,'observer,T>>,
    pub error:Option<RoutedTextExecutionError>,
    pub chunks:eredu_runtime::expert::AddressableChunkPlan,
    pub partitions:Option<usize>,
    pub context:&'context T::Context,
    pub run:fn(&mut P,RoutedExpertRequest<'input,'observer,T>,Option<usize>,
        eredu_runtime::expert::AddressableChunkPlan,Option<&eredu_nn::workspace::WorkspaceMetadataFunding>,&T::Context)
        ->Result<eredu_nn::TensorParallelGroupedOutput<T>,RoutedTextExecutionError>,
}
pub(crate) fn callback_control_bytes<T:Tensor>()->Option<usize> {
    std::mem::size_of::<Callback<'_,'_,'_,(),T>>()
        .checked_add(std::mem::size_of::<&mut Callback<'_,'_,'_,(),T>>())?
        // The actual adapter contains one owner reference, one accessor fn
        // pointer and one callback trait-object loan; no P or M by value.
        .checked_add(eredu_runtime::expert::BorrowedIndexedInvocation::<(),(),T>::control_bytes())
}
pub(super) fn declaration<'a>(owner_group:&'a str,bank:RoutedBankId,unit:usize,
    pass:eredu_runtime::ExpertPass,chunks:eredu_runtime::expert::AddressableChunkPlan,
    local_members:Option<&'a [usize]>,kernel:WorkspaceExpertKernel<'a>,partitions:Option<usize>,
    compact_scratch_bytes:u64,bulk_target_bytes:u64,callback_control_bytes:usize)
    ->Result<WorkspaceAddressableRegionView<'a>,WorkspaceMetadataError> {
    let source=WorkspaceAddressableRegionView {owner_group,bank:bank.value(),unit,
        prefill:matches!(pass,eredu_runtime::ExpertPass::Prefill),chunks:chunks.workspace_source(),
        local_members,kernel,tensor_partitions:partitions,compact_scratch_bytes,bulk_target_bytes,
        callback_control_bytes};
    source.validate()?;Ok(source)
}
impl SelectedRoutedBank {
    /// Borrows the already selected physical equation and member-byte policy.
    /// This creates only a descriptor; the backend must bind its real bank source.
    pub(crate) fn addressable_workspace_source<'a,T:Tensor>(&'a self,
        request:&RoutedExpertRequest<'_, '_,T>,options:eredu_runtime::ParameterBankLoadOptions,
        partitions:Option<usize>,context:&eredu_nn::workspace::WorkspaceContext)->Result<WorkspaceAddressableRegionView<'a>,eredu_nn::Error> {
        context.charge_metadata(std::mem::size_of::<(
            &Self,&RoutedExpertRequest<'_,'_,T>,eredu_runtime::ParameterBankLoadOptions,Option<usize>,
            WorkspaceAddressableRegionView<'_>,WorkspaceExpertKernel<'_>,Option<&[usize]>,
            eredu_nn::workspace::ExpertRegionInputShape,Option<u64>,
            eredu_runtime::expert::AddressableChunkPlan,Result<WorkspaceAddressableRegionView<'_>,eredu_nn::Error>,
            std::slice::Iter<'_,eredu_runtime::AddressableBankMember>,[usize;4],
        )>())?;
        validate_route_cardinality(request.routes,*self.routes_by_unit.get(&request.layer)
            .ok_or(WorkspaceMetadataError::Unqualified)?)
            .map_err(|_|eredu_nn::Error::from(WorkspaceMetadataError::Unqualified))?;
        let (kernel,local_members)=match &self.plan {
            RoutedGroupedPlan::Linear(plan)=>(WorkspaceExpertKernel::Linear(plan.unit_spec(self.owner_group.as_str(),request.layer)
                .ok_or(WorkspaceMetadataError::Unqualified)?),(!plan.unit_is_replicated(request.layer)).then(||plan.local_global_group_indices())),
            RoutedGroupedPlan::Gated(plan)=>(WorkspaceExpertKernel::Gated(plan.unit_spec(self.owner_group.as_str(),request.layer)
                .ok_or(WorkspaceMetadataError::Unqualified)?),(!plan.unit_is_replicated(request.layer)).then(||plan.local_global_group_indices())),
            RoutedGroupedPlan::Relu2(plan)=>(WorkspaceExpertKernel::Relu2(plan.unit_spec(self.owner_group.as_str(),request.layer)
                .ok_or(WorkspaceMetadataError::Unqualified)?),(!plan.unit_is_replicated(request.layer)).then(||plan.local_global_group_indices())),
        };
        let members=match kernel {WorkspaceExpertKernel::Linear(v)=>v.group_count(),WorkspaceExpertKernel::Gated(v)=>v.group_count(),WorkspaceExpertKernel::Relu2(v)=>v.group_count()};
        let geometry=eredu_nn::workspace::ExpertRegionInputShape::inspect(request.input.shape(),request.routes.group_indices().shape())?;
        let access=request.pass.parameter_bank_access();
        let maximum=if access==eredu_runtime::ParameterBankAccess::Bulk {
            Some(self.addressable_members.iter().filter(|member|member.key().unit()==request.layer)
                .map(|member|member.selected_bytes()).max().ok_or(WorkspaceMetadataError::Unqualified)?)
        }else{None};
        let chunks=eredu_runtime::expert::AddressableChunkPlan::new(
            usize::try_from(geometry.rows).map_err(|_|WorkspaceMetadataError::Overflow)?,
            usize::try_from(geometry.routes).map_err(|_|WorkspaceMetadataError::Overflow)?,
            usize::try_from(members).map_err(|_|WorkspaceMetadataError::Overflow)?,access,maximum,
            options.prefill_compact_bank_target_bytes()).map_err(eredu_nn::Error::backend_source)?;
        declaration(self.owner_group.as_str(),request.bank,request.layer,request.pass,chunks,local_members,kernel,
            partitions,options.compact_bank_scratch_bytes(),options.prefill_compact_bank_target_bytes(),
            callback_control_bytes::<T>().ok_or(WorkspaceMetadataError::Overflow)?).map_err(Into::into)
    }
}
