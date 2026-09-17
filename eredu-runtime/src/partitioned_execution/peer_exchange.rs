//! Original variable exchange reuses the ordinary selected operation and count
//! matrix. No shape vector or diagnostic string is created on this path.
use super::*;
use eredu_nn::workspace::{WorkspaceMetadataFunding, WorkspaceMetadataFundingError};
use std::mem::size_of;

#[derive(Debug, thiserror::Error)]
enum Cause {
    #[error("variable exchange count matrix differs from the selected group")]
    Matrix,
    #[error("variable exchange source has invalid tensor geometry")]
    Geometry,
    #[error("variable exchange has no exact native transport source")]
    NativeSource,
    #[error("variable exchange has no funded actual tensor metadata source")]
    Metadata,
    #[error(transparent)]
    Group(#[from] crate::CommunicationGroupOperationError),
    #[error(transparent)]
    Tensor(#[from] crate::CommunicationTensorContractError),
    #[error(transparent)]
    Funding(#[from] WorkspaceMetadataFundingError),
}
#[derive(Debug, thiserror::Error)]
#[error("{cause}")]
struct Failure { #[source] cause:Cause, funding:WorkspaceMetadataFunding }
fn fail(cause:Cause,funding:&WorkspaceMetadataFunding)->PartitionExecutionError {
    PartitionExecutionError::PreparedCommunication {
        operation:CommunicationOperation::VariableAllToAll,
        phase:DistributedExecutionPhase::Execution,completion:false,
        source:eredu_core::BackendFailure::new(eredu_core::BackendFailureKind::Other,
            Failure {cause,funding:funding.clone()}),
    }
}
fn fixed_controls<B:VariableAllToAllBackend>()->Option<usize>{
    let extents=[size_of::<Cause>(),size_of::<Failure>(),size_of::<WorkspaceMetadataFunding>(),
        size_of::<[B::Tensor;2]>(),size_of::<(&B::Tensor,&CommunicationPeerCounts,usize,CollectiveGroupId,&B::Executor,Option<&B::ParallelContext>,&[usize],bool)>(),
        size_of::<(&[i32],&[i32])>(),size_of::<Result<Option<(TensorDtype,usize,Option<usize>)>,WorkspaceMetadataFundingError>>(),
        size_of::<Option<&eredu_core::ErasedSharedStorageOwner>>(),size_of::<crate::CommunicationPeerMatrix<'_>>(),size_of::<Option<crate::CommunicationPeerMatrix<'_>>>(),
        size_of::<Result<B::Tensor,PartitionExecutionError>>(),size_of::<Result<Option<B::Tensor>,B::CommunicationError>>(),
        size_of::<Result<Result<B::Tensor,PartitionExecutionError>,B::CommunicationError>>(),
        size_of::<(TensorDtype,usize,Option<usize>)>(),size_of::<Result<(),Cause>>(),
        size_of::<(usize,usize,usize,usize,bool)>(),size_of::<(Option<usize>,Option<usize>)>(),
        size_of::<std::iter::Enumerate<std::slice::Iter<'_,i32>>>(),
        size_of::<std::slice::Iter<'_,usize>>(),size_of::<B::CommunicationError>(),
        eredu_core::BackendFailure::source_retention_peak_bytes::<Failure>()?,
        eredu_core::BackendFailure::source_retention_peak_bytes::<B::CommunicationError>()?];
    extents.into_iter().try_fold(size_of_val(&extents),usize::checked_add)
}
use std::mem::size_of_val;
impl<B,G,R,I> PartitionCommunication<B,G,R,I>
where B:VariableAllToAllBackend,G:Borrow<B::CommunicationGroup>,R:Borrow<B::CommunicationRoute>,I:CommunicationTensorMetadata<B> {
    /// Runs the existing variable exchange with an explicit retained count source.
    /// Ordinary contexts use the ordinary worker. Original contexts must provide
    /// a complete native source for this matrix and exact selected resource.
    pub fn variable_all_to_all_with_source(&self,value:B::Tensor,counts:&CommunicationPeerCounts,
        axis:usize,group:CollectiveGroupId,executor:&B::Executor,
        current:Option<&B::ParallelContext>,matrix:&[usize],transposed:bool,
    )->Result<B::Tensor,PartitionExecutionError>{
        self.variable_all_to_all_with_count_source(value,counts,axis,group,executor,current,matrix,transposed,None)
    }
    /// Retains the actual completed count-source owner through native binding.
    pub fn variable_all_to_all_with_count_source(&self,value:B::Tensor,counts:&CommunicationPeerCounts,
        axis:usize,group:CollectiveGroupId,executor:&B::Executor,
        current:Option<&B::ParallelContext>,matrix:&[usize],transposed:bool,
        completed_source:Option<&eredu_core::ErasedSharedStorageOwner>,
    )->Result<B::Tensor,PartitionExecutionError>{
        let Some(current)=current else {return self.variable_all_to_all(value,counts,axis,group,executor);};
        B::with_parallel_control_context(current,|prepared|{
            let Some((context,funding))=prepared else {
                return self.variable_all_to_all(value,counts,axis,group,executor);
            };
            let controls=fixed_controls::<B>().ok_or(PartitionExecutionError::CommunicationShapeOverflow)?;
            funding.reserve_metadata(controls).map_err(|cause|fail(cause.into(),funding))?;
            self.ensure_active()?;
            let result=(||{
                let selected=self.manifest.select_group_operation(group,CommunicationOperation::VariableAllToAll)
                    .map_err(|cause|fail(cause.into(),funding))?;
                let descriptor=selected.descriptor();
                let native=self.groups[selected.order()].resource.borrow();
                let source=crate::CommunicationPeerMatrix::checked(descriptor,matrix,transposed,counts)
                    .ok_or_else(||fail(Cause::Matrix,funding))?.with_completed_source(completed_source);
                let requirement=Self::group_requirement(descriptor,CommunicationOperation::VariableAllToAll);
                let max=requirement.limits().and_then(|limits|limits.max_count_per_peer())
                    .ok_or_else(||fail(Cause::Matrix,funding))?;
                if matrix.iter().any(|&count|count>max){return Err(fail(Cause::Matrix,funding));}
                let send=counts.send().iter().try_fold(0usize,|n,&count|n.checked_add(count))
                    .ok_or_else(||fail(Cause::Matrix,funding))?;
                let receive=counts.receive().iter().try_fold(0usize,|n,&count|n.checked_add(count))
                    .ok_or_else(||fail(Cause::Matrix,funding))?;
                let input_shape=value.shape();
                if input_shape.get(axis).and_then(|&n|usize::try_from(n).ok())!=Some(send)
                    || input_shape.iter().any(|&n|n<0) || i32::try_from(receive).is_err(){
                    return Err(fail(Cause::Geometry,funding));
                }
                let (dtype,rank,elements)=self.inspector.fixed_metadata_with_funding(&value,funding)
                    .map_err(|cause|fail(cause.into(),funding))?.ok_or_else(||fail(Cause::Metadata,funding))?;
                let input_elements=input_shape.iter().try_fold(1usize,|n,&dim|n.checked_mul(usize::try_from(dim).ok()?));
                if rank!=input_shape.len() || elements!=input_elements {return Err(fail(Cause::Geometry,funding));}
                requirement.validate_tensor_metadata(&dtype,rank,elements,false)
                    .map_err(|cause|fail(cause.into(),funding))?;
                let expected_elements=input_shape.iter().enumerate().try_fold(1usize,|n,(index,&dim)|
                    n.checked_mul(if index==axis {receive}else{usize::try_from(dim).ok()?}));
                requirement.validate_tensor_metadata(&dtype,rank,expected_elements,true)
                    .map_err(|cause|fail(cause.into(),funding))?;
                let output=B::complete_prepared_variable_all_to_all(&value,counts,axis,&source,native,context,executor,funding)
                    .map_err(|cause|PartitionExecutionError::PreparedCommunication {
                        operation:CommunicationOperation::VariableAllToAll,phase:DistributedExecutionPhase::Execution,
                        completion:false,source:eredu_core::BackendFailure::from_error(cause)})?
                    .ok_or_else(||fail(Cause::NativeSource,funding))?;
                let shape=output.shape();
                if shape.len()!=rank || shape.iter().enumerate().any(|(index,&dim)|
                    if index==axis {usize::try_from(dim).ok()!=Some(receive)}else{dim!=input_shape[index]}){
                    return Err(fail(Cause::Geometry,funding));
                }
                let (actual_dtype,actual_rank,actual_elements)=self.inspector.fixed_metadata_with_funding(&output,funding)
                    .map_err(|cause|fail(cause.into(),funding))?.ok_or_else(||fail(Cause::Metadata,funding))?;
                if actual_dtype!=dtype || actual_rank!=rank || actual_elements!=expected_elements {
                    return Err(fail(Cause::Geometry,funding));
                }
                requirement.validate_tensor_metadata(&actual_dtype,actual_rank,actual_elements,true)
                    .map_err(|cause|fail(cause.into(),funding))?;
                Ok(output)
            })();
            result.map_err(|cause|self.output_contract_error(cause,CommunicationOperation::VariableAllToAll,
                DistributedExecutionPhase::Execution,None))
        }).map_err(|cause|{
            self.authority.fence_protocol_failure(CommunicationOperation::VariableAllToAll,DistributedExecutionPhase::Execution,None);
            PartitionExecutionError::PreparedCommunication {operation:CommunicationOperation::VariableAllToAll,
                phase:DistributedExecutionPhase::Execution,completion:false,
                source:eredu_core::BackendFailure::from_error(cause)}
        })?
    }
}
