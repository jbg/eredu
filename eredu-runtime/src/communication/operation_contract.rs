//! Allocation-free selected rank/tensor checks shared by ordinary and prepared consumers.
use super::*;

/// Finite reason that one actual tensor does not satisfy its selected operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CommunicationTensorContractError {
    /// The logical dtype is absent from the selected declaration.
    #[error("communication tensor dtype is not selected")]
    Dtype,
    /// A tensor was supplied to a declaration without tensor limits.
    #[error("communication operation has no tensor limits")]
    MissingLimits,
    /// Rank or checked element population exceeds the selected side's limits.
    #[error("communication tensor exceeds selected limits")]
    Limits,
}
impl CommunicationOperationRequirement {
    /// Fixed frames for the allocation-free tensor validator, before invoking it.
    pub fn tensor_metadata_control_bytes()->Option<usize> {
        let parts=[std::mem::size_of::<(&Self,&TensorDtype,usize,Option<usize>,bool)>(),
            std::mem::size_of::<CommunicationTensorLimits>(),std::mem::size_of::<usize>(),
            std::mem::size_of::<std::slice::Iter<'_,TensorDtype>>(),
            std::mem::size_of::<Result<(),CommunicationTensorContractError>>()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
    }
    /// Check one actual tensor without copying its shape or formatting an error.
    /// `elements=None` represents checked dimension-product overflow. Completion
    /// selects the declared result bound; it does not certify native completion.
    pub fn validate_tensor_metadata(&self, dtype: &TensorDtype, rank: usize,
        elements: Option<usize>, completed: bool) -> Result<(), CommunicationTensorContractError> {
        if !self.dtypes.contains(dtype) { return Err(CommunicationTensorContractError::Dtype); }
        let limits=self.limits.ok_or(CommunicationTensorContractError::MissingLimits)?;
        let maximum=if completed {limits.max_output_tensor_elements()} else {limits.max_tensor_elements()};
        if rank>limits.max_tensor_rank() || elements.is_none_or(|elements| elements>maximum) {
            return Err(CommunicationTensorContractError::Limits);
        }
        Ok(())
    }
}

/// Finite selected group refusal, without an allocated resource label.
#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum CommunicationGroupOperationError {
    /// The opaque group ID is absent from this exact rank's manifest.
    #[error("unknown communication group {0:?}")]
    Unknown(CollectiveGroupId),
    /// This rank has no local member slot in the selected group.
    #[error("local rank is not a member of communication group {0:?}")]
    NotMember(CollectiveGroupId),
    /// This group's declaration does not select the requested operation.
    #[error("communication group {group:?} does not select {operation:?}")]
    NotSelected {
        /// Exact opaque group ID requested by the caller.
        group: CollectiveGroupId,
        /// Requested semantic operation.
        operation: CommunicationOperation,
    },
}

/// Borrowed exact declaration selected from one immutable rank-local manifest.
/// This is descriptive metadata, never a native resource or submission grant.
#[derive(Clone, Copy, Debug)]
pub struct CommunicationGroupOperation<'a> {
    order:usize,
    descriptor:&'a CommunicationGroupDescriptor,
    requirement:&'a CommunicationOperationRequirement,
}
impl<'a> CommunicationGroupOperation<'a> {
    /// The selected declaration's authoritative order within the manifest.
    pub fn order(self)->usize {self.order}
    /// The same selected group declaration; no reconstruction or deep clone.
    pub fn descriptor(self)->&'a CommunicationGroupDescriptor {self.descriptor}
    /// The same selected semantic operation and its tensor/completion limits.
    pub fn requirement(self)->&'a CommunicationOperationRequirement {self.requirement}
}
impl CommunicationManifest {
    /// Fixed controls for the exact finite declaration lookup, with no native credit.
    pub fn group_operation_control_bytes()->Option<usize> {
        let parts=[std::mem::size_of::<(&Self,CollectiveGroupId,CommunicationOperation)>(),
            std::mem::size_of::<std::iter::Enumerate<std::slice::Iter<'_,CommunicationGroupDescriptor>>>(),
            std::mem::size_of::<(usize,&CommunicationGroupDescriptor)>(),
            std::mem::size_of::<std::slice::Iter<'_,CommunicationOperationRequirement>>(),
            std::mem::size_of::<&CommunicationOperationRequirement>(),
            std::mem::size_of::<CommunicationGroupOperation<'_>>(),
            std::mem::size_of::<Result<CommunicationGroupOperation<'_>,CommunicationGroupOperationError>>()];
        parts.into_iter().try_fold(std::mem::size_of_val(&parts),usize::checked_add)
    }
    /// Resolve exact group membership and operation selection with no allocation.
    /// Source identity, active authority and actual native resources are separate.
    pub fn select_group_operation(&self, id:CollectiveGroupId, operation:CommunicationOperation)
        ->Result<CommunicationGroupOperation<'_>,CommunicationGroupOperationError> {
        for (order,descriptor) in self.groups.iter().enumerate() {
            if descriptor.id()!=id {continue;}
            if descriptor.local_index().is_none() {return Err(CommunicationGroupOperationError::NotMember(id));}
            for requirement in descriptor.requirements().operations() {
                if requirement.operation()==operation {
                    return Ok(CommunicationGroupOperation{order,descriptor,requirement});
                }
            }
            return Err(CommunicationGroupOperationError::NotSelected{group:id,operation});
        }
        Err(CommunicationGroupOperationError::Unknown(id))
    }
}

#[cfg(test)]
mod tests;
