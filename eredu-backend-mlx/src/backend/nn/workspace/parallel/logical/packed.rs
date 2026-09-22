//! The existing packed-world branch, quoted one actual completion stage at a time.
use super::*;
use crate::backend::{
    nn::logical_collective::packed,
    runtime::distributed::topology::original_source::{
        OriginalCommunicationSource, OwnedPackedWorldCompletion,
    },
};
pub(crate) struct PackedWorldQuote {
    pub(crate) world: OwnedPackedWorldCompletion,
    pub(crate) pack_recipe: SpeculativeNumericalRecipe,
    pub(crate) pack_scratch: Option<WorkspaceAllocationPopulation>,
    pub(crate) pack_capacity: BoundaryStageCapacity,
    pub(crate) result_recipe: SpeculativeNumericalRecipe,
    pub(crate) result_scratch: Option<WorkspaceAllocationPopulation>,
    pub(crate) operation_source: Option<WorkspaceAllocationPopulation>,
    pub(crate) output_source: Option<WorkspaceAllocationPopulation>,
    pub(crate) result_capacity: BoundaryStageCapacity,
    pub(crate) slot: usize,
    pub(crate) members: usize,
    pub(crate) member_ranks: Vec<usize>,
    pub(crate) ordinary_completion: Option<OrdinaryCallControls>,
}
impl PackedWorldQuote {
    pub(super) fn prepare(
        source: &OriginalParallelSource,
        actual: &OriginalCommunicationSource<'_>,
        order: usize,
        kind: LogicalCollectiveKind,
        input: &WorkspaceLayout,
        dtype: safemlx::Dtype,
        mechanism: ResidentExecutionMechanisms,
        context: &WorkspaceContext,
    ) -> Result<Self, Error> {
        context.charge_metadata(size_of::<(
            Self,
            Result<Self, Error>,
            [WorkspaceTensor; 4],
            WorkspaceTraceReport,
            SpeculativeNumericalRecipe,
            Option<WorkspaceAllocationPopulation>,
        )>())?;
        let invalid = || {
            context.metadata_error(format_args!(
                "packed world differs from selected subgroup source"
            ))
        };
        let plan = actual
            .packed_world_plan(order)
            .map_err(|cause| source.neural_error(cause))?
            .ok_or_else(invalid)?;
        let ordinary_completion=crate::backend::runtime::distributed::completion::MlxCommunicationCompletion::ordinary_collective_completion_call_controls(
            actual.group(order).ok_or_else(invalid)?.0,&[]);
        let member_count = plan.members().len();
        let mut member_ranks = context.metadata_vec(member_count)?;
        member_ranks.extend_from_slice(plan.members());
        let slot = match kind {
            LogicalCollectiveKind::Sum => plan.representative(),
            LogicalCollectiveKind::Gather => plan.world_rank(),
        };
        let prototype = WorkspaceTensor::existing(input.clone(), context)?;
        let ops = logical_collective::Workspace(context);
        context.begin_span();
        let packed = packed::pack(&ops, &prototype, slot, plan.world_size())?;
        let packed_layout = packed.layout().clone();
        let pack_scratch = context.new_allocation_scratch()?;
        let report = context.finish_report(&[packed])?;
        let pack_recipe = super::super::numerical(&report, 1, mechanism, context)?;
        let pack_capacity = boundary::capacity(pack_recipe, source.initialized_runtime(), context)?;
        let world = source
            .packed_world_completion(order, packed_layout.shape(), dtype)
            .map_err(|cause| source.neural_error(cause))?;
        let completed = WorkspaceTensor::existing(packed_layout, context)?;
        context.begin_span();
        let result = match kind {
            LogicalCollectiveKind::Sum => packed::sum_result(&ops, &completed, slot)?,
            LogicalCollectiveKind::Gather => {
                let stacked = packed::gather_stacked(&ops, &completed, &member_ranks)?;
                packed::flatten(&ops, &stacked, input.shape(), member_count)?
            }
        };
        let result_scratch = context.new_allocation_scratch()?;
        let report = context.finish_report(&[result])?;
        let result_recipe = super::super::numerical(&report, 1, mechanism, context)?;
        let result_capacity =
            boundary::capacity(result_recipe, source.initialized_runtime(), context)?;
        let (operation_source, output_source) = match (&pack_scratch, &result_scratch) {
            (Some(pack), Some(result)) => {
                let world = super::super::scratch::native_cpu(
                    context,
                    mechanism,
                    world.backing_capacity(),
                    world.births(),
                )?;
                let (scratch, output) =
                    allocation_populations(context, kind, pack, &world, result)?;
                (Some(scratch), Some(output))
            }
            _ => (None, None),
        };
        Ok(Self {
            world,
            pack_recipe,
            pack_scratch,
            pack_capacity,
            result_recipe,
            result_scratch,
            operation_source,
            output_source,
            result_capacity,
            slot,
            members: member_count,
            member_ranks,
            ordinary_completion,
        })
    }
    pub(crate) fn scratch(&self) -> Option<usize> {
        self.pack_capacity
            .backing
            .checked_add(self.world.backing_capacity())
    }
    pub(crate) fn maximum_births(&self) -> Option<usize> {
        self.pack_recipe
            .storage
            .maximum_births()
            .checked_add(self.world.births())?
            .checked_add(self.result_recipe.storage.maximum_births())
    }
}

/// Split the actual completed world source according to the shared extraction
/// worker. A Sum view escapes with that backing; Gather allocates its result.
/// Each source belongs to exactly one lifetime population.
pub(in crate::backend::nn::workspace::parallel) fn allocation_populations(
    context: &WorkspaceContext,
    kind: LogicalCollectiveKind,
    pack: &WorkspaceAllocationPopulation,
    world: &WorkspaceAllocationPopulation,
    result: &WorkspaceAllocationPopulation,
) -> Result<(WorkspaceAllocationPopulation, WorkspaceAllocationPopulation), Error> {
    match kind {
        LogicalCollectiveKind::Sum => Ok((
            pack.clone(),
            context.combine_scratch_populations(&[(world, 1), (result, 1)])?,
        )),
        LogicalCollectiveKind::Gather => Ok((
            context.combine_scratch_populations(&[(pack, 1), (world, 1)])?,
            result.clone(),
        )),
    }
}
