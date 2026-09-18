//! Typed erasure of the actual selected resident or bounded operation slot.
use crate::backend::{error::Error,nn::workspace::SpeculativeNumericalRecipe,
    runtime::execution::generic::{RealtimeNeuralPlan,QualifiedRealtimeNeuralPlan,RealtimeNeuralOwner,
        RealtimeLayerwisePlan,QualifiedRealtimeLayerwisePlan,LayerwiseWorkspace,MlxResidentPolicy,MlxLayerwisePolicy},
    submission_recovery::native_role::realtime::RealtimeOperationClaim};
use eredu_nn::workspace::{WorkspaceContext,WorkspaceMetadataError,HostMetadataFunding};
use std::mem::{size_of,size_of_val};
use crate::backend::nn::workspace::MetalAllocationFacts;
use eredu_runtime::working_memory::{WorkingMemoryPool,OriginalHostSourceBank,HostSourceConstructionFacts};
use safemlx::Stream;

/// Actual selected policies supply their own immutable layout and install slot.
/// The generic parallel wrapper delegates without reconstructing family state.
pub(super) trait RealtimeOperationPolicy<U:'static> {
    fn realtime_operation_plan(&self,stream:&Stream,allocation:MetalAllocationFacts,
        pool:&WorkingMemoryPool,context:&WorkspaceContext)->Result<RealtimeOperationPlan,Error>;
}
impl<U:'static> RealtimeOperationPolicy<U> for MlxResidentPolicy<U> {
    fn realtime_operation_plan(&self,stream:&Stream,_allocation:MetalAllocationFacts,
        _pool:&WorkingMemoryPool,context:&WorkspaceContext)->Result<RealtimeOperationPlan,Error> {
        RealtimeOperationPlan::new(self.original_realtime_plan(stream,context)?,context)
    }
}
impl<U:'static> RealtimeOperationPolicy<U> for MlxLayerwisePolicy<U,()> {
    fn realtime_operation_plan(&self,stream:&Stream,allocation:MetalAllocationFacts,
        pool:&WorkingMemoryPool,context:&WorkspaceContext)->Result<RealtimeOperationPlan,Error> {
        let source=self.layerwise_workspace_with_metadata(allocation,context)?;
        RealtimeOperationPlan::new_bounded(self.realtime_plan(source,stream,pool,context)?,context)
    }
}


/// Move-only descriptive model source; constructing or cloning accounting
/// handles cannot create it. The concrete slot remains inside this owner.
pub(crate) struct RealtimeOperationPlan { source:Box<dyn Source>, _funding:HostMetadataFunding }
pub(crate) struct RealtimeOperationRecipe { source:Box<dyn Qualified>, _funding:HostMetadataFunding }
trait Source {
    fn qualify(self:Box<Self>,recipe:SpeculativeNumericalRecipe,layerwise:Option<&LayerwiseWorkspace>,context:&WorkspaceContext)
        ->Result<RealtimeOperationRecipe,Error>;
}
trait Qualified {
    fn recipe(&self)->&SpeculativeNumericalRecipe;
    fn control_bytes(&self)->Option<u64>;
    fn source_facts(&self)->Option<HostSourceConstructionFacts>;
    fn activate(self:Box<Self>,claim:RealtimeOperationClaim<'_>,source:Option<OriginalHostSourceBank>)->Result<RealtimeNeuralOwner,Error>;
}
fn reserve(parts:&[usize],context:&WorkspaceContext)->Result<(),Error> {
    let bytes=parts.iter().copied().try_fold(size_of_val(parts),usize::checked_add)
        .ok_or_else(||Error::Neural(WorkspaceMetadataError::Overflow.into()))?;
    context.charge_metadata(bytes).map_err(|cause|Error::Neural(cause.into()))
}
impl RealtimeOperationPlan {
    pub(super) fn new<U:'static>(plan:RealtimeNeuralPlan<U>,context:&WorkspaceContext)->Result<Self,Error> {
        let funding=context.metadata_funding().ok_or_else(||Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
        reserve(&[size_of::<HostMetadataFunding>(),size_of::<Self>(),size_of::<RealtimeNeuralPlan<U>>(),size_of::<Box<dyn Source>>(),
            size_of::<Result<Self,Error>>()],context)?;
        Ok(Self{source:Box::new(plan),_funding:funding})
    }
    fn new_bounded<U:'static>(plan:RealtimeLayerwisePlan<U>,context:&WorkspaceContext)->Result<Self,Error> {
        let funding=context.metadata_funding().ok_or_else(||Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
        reserve(&[size_of::<HostMetadataFunding>(),size_of::<Self>(),size_of::<RealtimeLayerwisePlan<U>>(),
            size_of::<Box<dyn Source>>(),size_of::<Result<Self,Error>>()],context)?;
        Ok(Self{source:Box::new(plan),_funding:funding})
    }
    pub(crate) fn qualify(self,recipe:SpeculativeNumericalRecipe,layerwise:Option<&LayerwiseWorkspace>,context:&WorkspaceContext)
        ->Result<RealtimeOperationRecipe,Error> { self.source.qualify(recipe,layerwise,context) }
}
impl<U:'static> Source for RealtimeNeuralPlan<U> {
    fn qualify(self:Box<Self>,recipe:SpeculativeNumericalRecipe,layerwise:Option<&LayerwiseWorkspace>,context:&WorkspaceContext)
        ->Result<RealtimeOperationRecipe,Error> {
        if layerwise.is_some(){return Err(Error::Neural(WorkspaceMetadataError::Unqualified.into()));}
        let funding=context.metadata_funding().ok_or_else(||Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
        reserve(&[size_of::<HostMetadataFunding>(),size_of::<RealtimeOperationRecipe>(),size_of::<QualifiedRealtimeNeuralPlan<U>>(),
            size_of::<Box<dyn Qualified>>(),size_of::<Result<RealtimeOperationRecipe,Error>>(),
            size_of::<(RealtimeNeuralPlan<U>,SpeculativeNumericalRecipe,&WorkspaceContext)>()],context)?;
        Ok(RealtimeOperationRecipe{source:Box::new((*self).qualify(recipe,context)?),_funding:funding})
    }
}
impl<U:'static> Qualified for QualifiedRealtimeNeuralPlan<U> {
    fn recipe(&self)->&SpeculativeNumericalRecipe { self.recipe() }
    fn control_bytes(&self)->Option<u64> { self.control_bytes() }
    fn source_facts(&self)->Option<HostSourceConstructionFacts> {None}
    fn activate(self:Box<Self>,claim:RealtimeOperationClaim<'_>,source:Option<OriginalHostSourceBank>)->Result<RealtimeNeuralOwner,Error> {
        if source.is_some(){return Err(Error::Neural(WorkspaceMetadataError::Unqualified.into()));}
        (*self).prepare(claim)
    }
}
impl RealtimeOperationRecipe {
    pub(crate) fn recipe(&self)->&SpeculativeNumericalRecipe { self.source.recipe() }
    pub(crate) fn control_bytes(&self)->Option<u64> { self.source.control_bytes() }
    pub(crate) fn source_facts(&self)->Option<HostSourceConstructionFacts> {self.source.source_facts()}
    pub(crate) fn activate(self,claim:RealtimeOperationClaim<'_>,source:Option<OriginalHostSourceBank>)->Result<RealtimeNeuralOwner,Error> {
        self.source.activate(claim,source)
    }
}

impl<U:'static> Source for RealtimeLayerwisePlan<U> {
    fn qualify(self:Box<Self>,recipe:SpeculativeNumericalRecipe,layerwise:Option<&LayerwiseWorkspace>,context:&WorkspaceContext)
        ->Result<RealtimeOperationRecipe,Error> {
        let source=layerwise.ok_or_else(||Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
        let funding=context.metadata_funding().ok_or_else(||Error::Neural(WorkspaceMetadataError::Unqualified.into()))?;
        reserve(&[size_of::<HostMetadataFunding>(),size_of::<RealtimeOperationRecipe>(),size_of::<QualifiedRealtimeLayerwisePlan<U>>(),
            size_of::<Box<dyn Qualified>>(),size_of::<Result<RealtimeOperationRecipe,Error>>(),
            size_of::<(RealtimeLayerwisePlan<U>,SpeculativeNumericalRecipe,&LayerwiseWorkspace,&WorkspaceContext)>()],context)?;
        Ok(RealtimeOperationRecipe{source:Box::new((*self).qualify(recipe,source,context)?),_funding:funding})
    }
}
impl<U:'static> Qualified for QualifiedRealtimeLayerwisePlan<U> {
    fn recipe(&self)->&SpeculativeNumericalRecipe {self.recipe()}
    fn control_bytes(&self)->Option<u64> {self.control_bytes()}
    fn source_facts(&self)->Option<HostSourceConstructionFacts> {self.source_facts()}
    fn activate(self:Box<Self>,claim:RealtimeOperationClaim<'_>,source:Option<OriginalHostSourceBank>)
        ->Result<RealtimeNeuralOwner,Error> {(*self).prepare(claim,source)}
}
