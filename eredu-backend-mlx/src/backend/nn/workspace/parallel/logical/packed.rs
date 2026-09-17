//! The existing packed-world branch, quoted one actual completion stage at a time.
use super::*;
use crate::backend::{nn::logical_collective::packed,
    runtime::distributed::topology::original_source::{OriginalCommunicationSource,OwnedPackedWorldCompletion}};
pub(crate) struct PackedWorldQuote {
    pub(crate) world:OwnedPackedWorldCompletion,
    pub(crate) pack_recipe:SpeculativeNumericalRecipe,
    pub(crate) pack_capacity:BoundaryStageCapacity,
    pub(crate) result_recipe:SpeculativeNumericalRecipe,
    pub(crate) result_capacity:BoundaryStageCapacity,
    pub(crate) slot:usize,
    pub(crate) members:usize,
    pub(crate) member_ranks:Vec<usize>,
}
impl PackedWorldQuote {
    pub(super) fn prepare(source:&OriginalParallelSource,actual:&OriginalCommunicationSource<'_>,order:usize,
        kind:LogicalCollectiveKind,input:&WorkspaceLayout,dtype:safemlx::Dtype,
        mechanism:ResidentExecutionMechanisms,context:&WorkspaceContext)->Result<Self,Error>{
        context.charge_metadata(size_of::<(Self,Result<Self,Error>,[WorkspaceTensor;4],
            WorkspaceTraceReport,SpeculativeNumericalRecipe)>())?;
        let invalid=||context.metadata_error(format_args!("packed world differs from selected subgroup source"));
        let plan=actual.packed_world_plan(order).map_err(|cause|source.neural_error(cause))?.ok_or_else(invalid)?;
        let member_count=plan.members().len();
        let mut member_ranks=context.metadata_vec(member_count)?;
        member_ranks.extend_from_slice(plan.members());
        let slot=match kind{LogicalCollectiveKind::Sum=>plan.representative(),LogicalCollectiveKind::Gather=>plan.world_rank()};
        let prototype=WorkspaceTensor::existing(input.clone(),context)?;
        let ops=logical_collective::Workspace(context);
        context.begin_span();
        let packed=packed::pack(&ops,&prototype,slot,plan.world_size())?;
        let packed_layout=packed.layout().clone();
        let report=context.finish_report(&[packed])?;
        let pack_recipe=super::super::numerical(&report,1,mechanism,context)?;
        let pack_capacity=boundary::capacity(pack_recipe,source.initialized_runtime(),context)?;
        let world=source.packed_world_completion(order,packed_layout.shape(),dtype)
            .map_err(|cause|source.neural_error(cause))?;
        let completed=WorkspaceTensor::existing(packed_layout,context)?;
        context.begin_span();
        let result=match kind {
            LogicalCollectiveKind::Sum=>packed::sum_result(&ops,&completed,slot)?,
            LogicalCollectiveKind::Gather=>{
                let stacked=packed::gather_stacked(&ops,&completed,&member_ranks)?;
                packed::flatten(&ops,&stacked,input.shape(),member_count)?
            }
        };
        let report=context.finish_report(&[result])?;
        let result_recipe=super::super::numerical(&report,1,mechanism,context)?;
        let result_capacity=boundary::capacity(result_recipe,source.initialized_runtime(),context)?;
        Ok(Self {world,pack_recipe,pack_capacity,result_recipe,result_capacity,slot,members:member_count,member_ranks})
    }
    pub(crate) fn scratch(&self)->Option<usize>{self.pack_capacity.backing.checked_add(self.world.backing_capacity())}
    pub(crate) fn maximum_births(&self)->Option<usize>{self.pack_recipe.storage.maximum_births()
        .checked_add(self.world.births())?.checked_add(self.result_recipe.storage.maximum_births())}
}
