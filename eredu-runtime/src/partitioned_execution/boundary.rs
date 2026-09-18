//! Paid mechanical validation and framing in the existing boundary driver.
use super::*;
use crate::{PreparedBoundarySource,PreparedBoundaryFrameCause};
use eredu_nn::workspace::HostMetadataFundingError;
use std::{alloc::Layout,mem::{size_of,size_of_val}};
#[derive(Debug,thiserror::Error)]
#[error("{cause}")]
struct Failure<E:std::error::Error+'static>{
    #[source] cause:E,
    _source:PreparedBoundarySource,
}
pub(super) struct Controls<'a,E>{source:&'a PreparedBoundarySource,marker:PhantomData<fn()->E>}
impl<'a,E:std::error::Error+Send+Sync+'static> Controls<'a,E>{
    pub(super) fn new(source:&'a PreparedBoundarySource)->Result<Self,PartitionExecutionError>{
        let value=Self{source,marker:PhantomData};
        let frames=[size_of::<Self>(),size_of::<Result<Self,PartitionExecutionError>>(),
            size_of::<crate::PreparedBoundaryFrameError>(),size_of::<Failure<E>>(),
            size_of::<eredu_core::BackendFailure>(),size_of::<CommunicationPoison>(),
            size_of::<std::sync::MutexGuard<'_,Option<CommunicationPoison>>>(),
            size_of::<(TensorDtype,usize,Option<usize>)>(),size_of::<Result<Option<bool>,HostMetadataFundingError>>(),
            size_of::<Result<(),crate::CommunicationTensorContractError>>(),
            size_of::<Result<(),crate::CommunicationManifestError>>(),
            CommunicationOperationRequirement::tensor_metadata_control_bytes().ok_or_else(||value.overflow())?,
            eredu_core::BackendFailure::source_retention_peak_bytes::<Failure<E>>().ok_or_else(||value.overflow())?];
        value.reserve(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add).ok_or_else(||value.overflow())?)?;
        Ok(value)
    }
    fn overflow(&self)->PartitionExecutionError{self.funding_error(HostMetadataFundingError::Overflow)}
    fn funding_error(&self,cause:HostMetadataFundingError)->PartitionExecutionError{
        self.source.error(PreparedBoundaryFrameCause::Funding(cause)).into()
    }
    fn contract(&self)->PartitionExecutionError{self.source.error(PreparedBoundaryFrameCause::Contract).into()}
    fn reserve(&self,bytes:usize)->Result<(),PartitionExecutionError>{
        self.source.funding().reserve_metadata(bytes).map_err(|cause|self.funding_error(cause))
    }
    fn capacity(&self,cause:std::collections::TryReserveError)->PartitionExecutionError{
        self.source.error(PreparedBoundaryFrameCause::Capacity(cause)).into()
    }
    fn roles<T>(&self,values:&[crate::ArchitectureBoundaryValue<T>],schema:&ResolvedBoundaryWireSchema,
        wire:PipelineWireContract)->Result<Vec<crate::BoundaryRoleContract>,PartitionExecutionError>{
        if values.len()!=1+schema.auxiliary().len(){return Err(self.contract());}
        let frames=[size_of::<Vec<crate::BoundaryRoleContract>>(),size_of::<Vec<usize>>(),
            size_of::<Result<Vec<crate::BoundaryRoleContract>,PartitionExecutionError>>(),
            size_of::<Result<(),std::collections::TryReserveError>>(),size_of::<String>(),
            size_of::<Result<crate::BoundaryRoleContract,crate::CommunicationManifestError>>(),
            Layout::array::<crate::BoundaryRoleContract>(values.len()).map_err(|_|self.overflow())?.size()];
        self.reserve(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add).ok_or_else(||self.overflow())?)?;
        let mut roles=Vec::new();roles.try_reserve_exact(values.len()).map_err(|cause|self.capacity(cause))?;
        for (value,spec) in values.iter().zip(std::iter::once(schema.primary()).chain(schema.auxiliary())){
            if value.role()!=spec.role(){return Err(self.contract());}
            // The same BoundaryRoleContract constructor consumes its input Vec
            // and creates Fixed dimensions; both exact destinations are paid.
            let bytes=Layout::array::<usize>(spec.shape().len()).map_err(|_|self.overflow())?.size()
                .checked_add(Layout::array::<crate::BoundaryDimensionContract>(spec.shape().len()).map_err(|_|self.overflow())?.size())
                .and_then(|n|n.checked_add(value.role().len())).ok_or_else(||self.overflow())?;
            self.reserve(bytes)?;
            let mut shape=Vec::new();shape.try_reserve_exact(spec.shape().len()).map_err(|cause|self.capacity(cause))?;
            for &dimension in spec.shape(){shape.push(usize::try_from(dimension).map_err(|_|self.contract())?);}
            roles.push(crate::BoundaryRoleContract::new(value.role(),boundary_dtype(spec,wire),shape)
                .map_err(|_|self.contract())?);
        }
        Ok(roles)
    }
    fn validate_tensor<B:NeuralBackend,I:CommunicationTensorMetadata<B>>(&self,inspector:&I,value:&B::Tensor,
        spec:&ResolvedBoundaryTensorSpec,wire:PipelineWireContract,completed:bool)->Result<(),PartitionExecutionError>{
        let funding=self.source.funding();
        let matches=inspector.matches_shape_with_funding(value,spec.shape(),funding)
            .map_err(|cause|self.funding_error(cause))?.ok_or_else(||self.contract())?;
        let (dtype,rank,elements)=inspector.fixed_metadata_with_funding(value,funding)
            .map_err(|cause|self.funding_error(cause))?.ok_or_else(||self.contract())?;
        if !matches||dtype!=boundary_dtype(spec,wire){return Err(self.contract());}
        self.source.descriptor().requirement().validate_tensor_metadata(&dtype,rank,elements,completed)
            .map_err(|_|self.contract())
    }
    pub(super) fn validate_tagged<B:NeuralBackend,I:CommunicationTensorMetadata<B>>(&self,inspector:&I,
        values:&[crate::ArchitectureBoundaryValue<B::Tensor>],schema:&ResolvedBoundaryWireSchema,
        wire:PipelineWireContract)->Result<(),PartitionExecutionError>{
        let contract=self.source.descriptor().boundary_contract().ok_or_else(||self.contract())?;
        if contract.schema()!=schema.identity(){return Err(self.contract());}
        let roles=self.roles(values,schema,wire)?;
        contract.validate_invocation(&roles).map_err(|_|self.contract())?;
        for (value,spec) in values.iter().zip(std::iter::once(schema.primary()).chain(schema.auxiliary())){
            self.validate_tensor::<B,I>(inspector,value.tensor(),spec,wire,false)?;
        }
        Ok(())
    }
    fn validate_output<B:NeuralBackend,I:CommunicationTensorMetadata<B>>(&self,inspector:&I,
        values:&[B::Tensor],schema:&ResolvedBoundaryWireSchema,wire:PipelineWireContract)->Result<(),PartitionExecutionError>{
        if values.len()!=1+schema.auxiliary().len(){return Err(self.contract());}
        for (value,spec) in values.iter().zip(std::iter::once(schema.primary()).chain(schema.auxiliary())){
            self.validate_tensor::<B,I>(inspector,value,spec,wire,true)?;
        }
        Ok(())
    }
    fn failure(&self,authority:&PartitionCommunicationAuthority,cause:E,completion:bool)->PartitionExecutionError{
        authority.mark_poisoned(CommunicationPoison{operation:CommunicationOperation::SendReceive,
            phase:DistributedExecutionPhase::Execution,route:Some(self.source.descriptor().id()),
            cancellation:authority.policy.expect("selected bounded boundary completion").cancellation()});
        PartitionExecutionError::PreparedCommunication{operation:CommunicationOperation::SendReceive,
            phase:DistributedExecutionPhase::Execution,completion,
            source:eredu_core::BackendFailure::from_error(Failure{cause,_source:self.source.clone()})}
    }
}
impl<B,G,R,I> PartitionCommunication<B,G,R,I>
where B:CommunicationBackend+PointToPointBackend,G:Borrow<B::CommunicationGroup>,R:Borrow<B::CommunicationRoute>,
    I:CommunicationTensorMetadata<B>,
{
    pub(super) fn transfer_boundary_prepared(&self,route:CommunicationRouteId,
        values:Vec<crate::ArchitectureBoundaryValue<B::Tensor>>,schema:&ResolvedBoundaryWireSchema,
        wire:PipelineWireContract,executor:&B::Executor,source:&PreparedBoundarySource,context:&B::ParallelContext)
        ->Result<Vec<B::Tensor>,PartitionExecutionError>{
        let controls=Controls::<B::CommunicationError>::new(source)?;
        self.ensure_active()?;
        let (descriptor,native)=self.route(route)?;
        if descriptor!=source.descriptor(){return Err(controls.contract());}
        controls.validate_tagged::<B,I>(&self.inspector,&values,schema,wire)?;
        let roles=controls.roles(&values,schema,wire)?;
        let frames=[Layout::array::<B::Tensor>(values.len()).map_err(|_|controls.overflow())?.size(),
            size_of::<Vec<B::Tensor>>(),size_of::<crate::PreparedBoundaryFrames<B::Tensor>>(),
            size_of::<eredu_core::Submission<Vec<B::Tensor>,B::CommunicationCompletion>>(),
            size_of::<Result<Option<eredu_core::Submission<Vec<B::Tensor>,B::CommunicationCompletion>>,B::CommunicationError>>(),
            size_of::<BoundedSubmissionOutcome<Vec<B::Tensor>>>(),size_of::<Result<Vec<B::Tensor>,PartitionExecutionError>>()];
        controls.reserve(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add).ok_or_else(||controls.overflow())?)?;
        let mut tensors=Vec::new();tensors.try_reserve_exact(values.len()).map_err(|cause|controls.capacity(cause))?;
        for value in values {tensors.push(value.into_parts().1);}
        let frames=source.frame_values(&roles,tensors)?;
        let submission=B::send_receive_prepared(frames,native,context,executor)
            .map_err(|cause|controls.failure(&self.authority,cause,false))?.ok_or_else(||controls.contract())?;
        let output=self.authority.wait_with_error(submission,CommunicationOperation::SendReceive,
            DistributedExecutionPhase::Execution,Some(route),|cause|controls.failure(&self.authority,cause,true))?;
        controls.validate_output::<B,I>(&self.inspector,&output,schema,wire).map_err(|cause|{
            self.authority.fence_protocol_failure(CommunicationOperation::SendReceive,DistributedExecutionPhase::Execution,Some(route));cause
        })?;
        Ok(output)
    }
}

impl<B,G,R,I> PartitionCommunication<B,G,R,I>
where B:CommunicationBackend,G:Borrow<B::CommunicationGroup>,R:Borrow<B::CommunicationRoute>,
    I:CommunicationTensorMetadata<B>,
{
    pub(super) fn complete_boundary_prepared(&self,
        values:&[crate::ArchitectureBoundaryValue<B::Tensor>],route:CommunicationRouteId,
        executor:&B::Executor,before_agreement:bool,source:&PreparedBoundarySource,context:&B::ParallelContext,
    )->Result<(),PartitionExecutionError>{
        let controls=Controls::<B::CommunicationError>::new(source)?;
        self.ensure_active()?;
        let (descriptor,native)=self.route(route)?;
        if descriptor!=source.descriptor()||descriptor.source()!=self.manifest.rank(){return Err(controls.contract());}
        let phase=DistributedExecutionPhase::BoundarySourceCompletion(route);
        let failure=|cause,completion|{
            if !before_agreement {
                self.authority.fence_protocol_failure(CommunicationOperation::SendReceive,phase,Some(route));
            }
            PartitionExecutionError::PreparedCommunication{
                operation:CommunicationOperation::SendReceive,phase,completion,
                source:eredu_core::BackendFailure::from_error(Failure{cause,_source:source.clone()}),
            }
        };
        let frames=[size_of::<eredu_core::Submission<(),B::CommunicationCompletion>>(),
            size_of::<Result<Option<eredu_core::Submission<(),B::CommunicationCompletion>>,B::CommunicationError>>(),
            size_of::<BoundedSubmissionOutcome<()>>(),size_of_val(&failure)];
        controls.reserve(frames.into_iter().try_fold(size_of_val(&frames),usize::checked_add).ok_or_else(||controls.overflow())?)?;
        let submission=B::submit_prepared_boundary_dependencies(values,source,native,context,executor)
            .map_err(|cause|failure(cause,false))?.ok_or_else(||controls.contract())?;
        if before_agreement {
            self.authority.wait_before_failure_agreement_with_error(submission,
                CommunicationOperation::SendReceive,phase,Some(route),|cause|failure(cause,true))
        } else {
            self.authority.wait_with_error(submission,
                CommunicationOperation::SendReceive,phase,Some(route),|cause|failure(cause,true))
        }
    }
}
