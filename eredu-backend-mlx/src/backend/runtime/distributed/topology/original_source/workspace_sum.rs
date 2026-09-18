//! One ordinary Sum equation, its exact native Group and the shared layout source.
use super::*;
use eredu_nn::workspace::{WorkspaceOperationView,WorkspaceOperationKindView,WorkspaceCollectiveView,
    WorkspaceDtype,WorkspaceFloatingType};
use safemlx::distributed::{GroupCpuLayoutStorage,GroupWorkerOperation};

/// Same immutable group/source and checked workspace equation. It can describe
/// a lazy model partial without constructing or evaluating a placeholder array.
/// Actual input binding still precedes the existing original constructor.
pub(crate) struct OriginalWorkspaceSum<'a> {
    native:GroupCpuLayoutStorage<'a>,
    persistent:OriginalCommunicatorPersistent<'a>,
    source:RetainedCommunicationSource,
    funding:HostMetadataFunding,
}
impl OriginalCommunicationSource<'_> {
    pub(crate) fn sum_workspace_storage<'a>(&'a self,id:CollectiveGroupId,equation:WorkspaceOperationView<'a>)
        ->Result<OriginalWorkspaceSum<'a>,Error> {
        self.validate()?;
        let (order,input,dtype)=sum_layout(&self.source,&self.funding,id,equation)?;
        let group=self.group(order).ok_or_else(||failure(Cause::Resource,&self.source,&self.funding))?.0;
        let persistent=self.group_persistent(order)?;
        if group.is_logical() || persistent.native().has_unqualified_storage() ||
            !persistent.native().is_for(group.native_group()) || !persistent.source().same_source(&self.source) {
            return Err(failure(Cause::Resource,&self.source,&self.funding));
        }
        self.funding.reserve_metadata(group.native_group().cpu_layout_storage_control_bytes()
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let native=group.native_group().cpu_layout_storage(input.shape(),dtype,GroupWorkerOperation::Sum)
            .map_err(|_|failure(Cause::Resource,&self.source,&self.funding))?;
        Ok(OriginalWorkspaceSum{native,persistent,source:self.source.clone(),funding:self.funding.clone()})
    }
}
impl<'a> OriginalWorkspaceSum<'a> {
    pub(crate) fn native(&self)->&GroupCpuLayoutStorage<'a>{&self.native}
    pub(crate) fn bind_actual<'input>(self,source:&OriginalCommunicationSource<'_>,input:&'input Array)
        ->Result<OriginalCommunicationOperation<'input>,Error> where 'a:'input {
        let parts=[size_of::<Self>(),size_of::<OriginalCommunicationOperation<'a>>(),
            size_of::<Result<OriginalCommunicationOperation<'a>,Error>>(),
            size_of::<(&OriginalCommunicationSource<'_>,&Array)>(),
            self.native.binding_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?];
        self.funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        source.validate()?;
        if !self.source.same_source(source.source()) {return Err(failure(Cause::Identity,&self.source,&self.funding));}
        let native=self.native.bind_actual(input).map_err(|cause|failure(Cause::NativeBinding { index:None,ordinal:None,cause },&self.source,&self.funding))?;
        Ok(OriginalCommunicationOperation::from_layout_binding(native,self.persistent,self.source,self.funding))
    }
}

pub(super) fn sum_layout<'a>(source:&RetainedCommunicationSource,funding:&HostMetadataFunding,
    id:CollectiveGroupId,equation:WorkspaceOperationView<'a>)
    ->Result<(usize,eredu_nn::workspace::WorkspaceLayoutView<'a>,safemlx::Dtype),Error> {
    let (order,input,dtype,operation)=parallel_layout(source,funding,id,equation)?;
    if operation!=GroupWorkerOperation::Sum{return Err(failure(Cause::Resource,source,funding));}
    Ok((order,input,dtype))
}

pub(super) fn parallel_layout<'a>(source:&RetainedCommunicationSource,funding:&HostMetadataFunding,
    id:CollectiveGroupId,equation:WorkspaceOperationView<'a>)
    ->Result<(usize,eredu_nn::workspace::WorkspaceLayoutView<'a>,safemlx::Dtype,GroupWorkerOperation),Error> {
        let parts=[size_of::<Option<(usize,&[usize])>>(),size_of::<GroupWorkerOperation>(),
            size_of::<(usize,eredu_nn::workspace::WorkspaceLayoutView<'a>,safemlx::Dtype,GroupWorkerOperation)>(),
            size_of::<[Option<usize>;5]>(),size_of::<[usize;6]>(),size_of::<std::slice::Iter<'_,usize>>(),
            size_of::<OriginalWorkspaceSum<'a>>(),size_of::<Result<OriginalWorkspaceSum<'a>,Error>>(),
            size_of::<(&RetainedCommunicationSource,&HostMetadataFunding,CollectiveGroupId,WorkspaceOperationView<'a>)>(),
            size_of::<eredu_nn::workspace::WorkspaceLayoutView<'a>>()*2,
            size_of::<Option<usize>>(),size_of::<Option<([eredu_nn::workspace::WorkspaceLayoutView<'a>;1],[eredu_nn::workspace::WorkspaceLayoutView<'a>;1])>>(),
            size_of::<Option<(&Group,&CommunicationGroupDescriptor,bool)>>(),
            size_of::<Result<eredu_runtime::CommunicationGroupOperation<'a>,eredu_runtime::CommunicationGroupOperationError>>(),
            size_of::<Result<(),eredu_runtime::CommunicationTensorContractError>>(),
            size_of::<std::array::IntoIter<bool,2>>(),size_of::<[bool;2]>(),
            size_of::<Option<eredu_nn::workspace::WorkspaceRepresentation>>(),
            size_of::<Option<WorkspaceFloatingType>>(),size_of::<TensorDtype>(),size_of::<safemlx::Dtype>(),
            CommunicationManifest::group_operation_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            CommunicationOperationRequirement::tensor_metadata_control_bytes().and_then(|n|n.checked_mul(2))
                .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?,
            failure_control_bytes().ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?];
        funding.reserve_metadata(parts.into_iter().try_fold(size_of_val(&parts),usize::checked_add)
            .ok_or(Error::WorkspacePlanning(HostMetadataFundingError::Overflow))?).map_err(Error::WorkspacePlanning)?;
        let (operation,native,partitions,rank,wire)=match equation.kind {
            WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Sum{partitions,rank})=>
                (CommunicationOperation::AllReduceSum,GroupWorkerOperation::Sum,partitions,rank,None),
            WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::Broadcast{group,partitions,rank,..}) if group==id =>
                (CommunicationOperation::Broadcast,GroupWorkerOperation::Sum,partitions,rank,None),
            WorkspaceOperationKindView::Collective(WorkspaceCollectiveView::GatherFirstAxis{axis,rank,peer_widths})=>
                (CommunicationOperation::AllGatherUneven,GroupWorkerOperation::Gather,peer_widths.len(),rank,Some((axis,peer_widths))),
            _=>return Err(failure(Cause::Resource,source,funding)),
        };
        let selected=source.manifest().select_group_operation(id,operation)
            .map_err(|cause|failure(Cause::Rank(cause),source,funding))?;
        let ([input],[output])=equation.inputs.array::<1>().zip(equation.outputs.array::<1>())
            .ok_or_else(||failure(Cause::Resource,source,funding))?;
        let requirement=selected.requirement();
        if partitions!=selected.descriptor().members().len() || Some(rank)!=selected.descriptor().local_index()
            || !requirement.exact_completion() {
            return Err(failure(Cause::Identity,source,funding));
        }
        let (original_input_elements,original_output_elements)=if let Some((axis,widths))=wire {
            let max=widths.iter().copied().max().filter(|&n|n!=0)
                .ok_or_else(||failure(Cause::Identity,source,funding))?;
            let width=widths.get(rank).copied().ok_or_else(||failure(Cause::Identity,source,funding))?;
            let first=input.shape().first().copied().and_then(|n|usize::try_from(n).ok());
            let height=first.and_then(|n|n.checked_mul(partitions)).and_then(|n|i32::try_from(n).ok());
            if input.shape().get(axis).copied().and_then(|n|usize::try_from(n).ok())!=Some(max)
                || input.shape().len()!=output.shape().len() || input.dtype()!=output.dtype()
                || output.shape().first().copied()!=height
                || input.shape().iter().skip(1).ne(output.shape().iter().skip(1)) {
                return Err(failure(Cause::Identity,source,funding));
            }
            let non_axis=input.elements().ok().and_then(|n|usize::try_from(n).ok()).and_then(|n|n.checked_div(max))
                .ok_or_else(||failure(Cause::Resource,source,funding))?;
            let total=widths.iter().try_fold(0usize,|n,&v|n.checked_add(v));
            (non_axis.checked_mul(width),total.and_then(|n|n.checked_mul(non_axis)))
        } else {
            if input!=output{return Err(failure(Cause::Identity,source,funding));}
            let elements=input.elements().ok().and_then(|n|usize::try_from(n).ok());
            (elements,elements)
        };
        // Floating logical storage is conservative F32. Only actual propagated
        // representation can select the native reduction's real scalar type.
        let dtype=match input.dtype() {
            WorkspaceDtype::Float32=>match input.representation().map(|value|value.dtype()) {
                Some(WorkspaceFloatingType::Float32)=>safemlx::Dtype::Float32,
                Some(WorkspaceFloatingType::Float16)=>safemlx::Dtype::Float16,
                Some(WorkspaceFloatingType::Bfloat16)=>safemlx::Dtype::Bfloat16,
                None=>return Err(failure(Cause::WorkspaceRepresentation {
                    operation, group: id, rank, partitions,
                    input_rank: input.shape().len(), input_width: input.shape().last().copied(),
                    input_elements: input.elements().ok(),
                },source,funding)),
            },
            WorkspaceDtype::Int32=>safemlx::Dtype::Int32,WorkspaceDtype::Bool=>safemlx::Dtype::Bool,
            WorkspaceDtype::Uint8=>safemlx::Dtype::Uint8,WorkspaceDtype::Uint32=>safemlx::Dtype::Uint32,
        };
        if output.representation().is_some_and(|value|Some(value.dtype())!=input.representation().map(|value|value.dtype())) {
            return Err(failure(Cause::Identity,source,funding));
        }
        let portable=crate::tensor::portable_dtype(dtype);
        for completed in [false,true] {
            requirement.validate_tensor_metadata(&portable,input.shape().len(),if completed {original_output_elements} else {original_input_elements},completed)
                .map_err(|cause|failure(Cause::Tensor(cause),source,funding))?;
        }
        Ok((selected.order(),input,dtype,native))
}
