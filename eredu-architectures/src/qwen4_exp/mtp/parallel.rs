//! Exact rank-local prediction equations, sharing target partition rules.
use super::{PredictionSpec, PredictionUnit};
use crate::{
    qwen4_exp::target::parallel_geometry::{local_expert_spec, parallel_layout, Selections},
    ExpertRealizationPlan,
};
use eredu_checkpoint::store::TensorSelection;
use eredu_core::ParallelRankTopology;
use eredu_nn::{
    DistributedNeuralBackend, Error, GroupedGatedProductSpec, GroupedNeuralBackend, Tensor,
};
use eredu_runtime::{ArchitectureParameterDescription, ExecutionGroupId, LocalModelLayout};
use std::collections::BTreeMap;

#[cfg(test)]
mod tests;

const OWNER: &str = "prediction";
fn invalid(detail: impl std::fmt::Display) -> Error {
    Error::backend(format!("qwen4_exp prediction partition: {detail}"))
}

/// Retained tensor-rank equations and exact physical parameter selections.
/// Fusion, residual mixing, routers, indexers and readout remain replicated.
#[derive(Debug, Clone)]
pub struct PredictionTensorPartition {
    rank: usize,
    ranks: usize,
    local: PredictionSpec,
    selections: BTreeMap<String, Vec<TensorSelection>>,
}
impl PredictionTensorPartition {
    /// Rank within the tensor group.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Number of tensor ranks.
    pub const fn ranks(&self) -> usize {
        self.ranks
    }
    /// Local equations and state geometry before optional expert ownership.
    pub fn local_spec(&self) -> &PredictionSpec {
        &self.local
    }
    /// Exact physical ranges, including encoded companions and fused segments.
    pub fn parameter_selections(&self, parameter: &str) -> Option<&[TensorSelection]> {
        self.selections.get(parameter).map(Vec::as_slice)
    }
    /// Validates an independently described local module against retained ranges.
    pub fn local_layout(
        &self,
        global: &ArchitectureParameterDescription,
        local: &ArchitectureParameterDescription,
    ) -> Result<LocalModelLayout, Error> {
        parallel_layout::local_layout(&self.selections, global, local)
    }
    /// Projects physical prediction coordinates without constructing native modules.
    /// Prediction replicas outside the tensor group retain every expert.
    pub(crate) fn projected_layout(
        &self,
        global: &ArchitectureParameterDescription,
    ) -> Result<LocalModelLayout, Error> {
        parallel_layout::project_local_layout(&self.selections, global, None)
    }
    /// Composes tensor ranges with exact balanced packed-expert ownership.
    pub fn local_expert_layout(
        &self,
        global: &ArchitectureParameterDescription,
        tensor_local: &ArchitectureParameterDescription,
        expert_local: &ArchitectureParameterDescription,
        expert_rank: usize,
        expert_ranks: usize,
    ) -> Result<LocalModelLayout, Error> {
        let tensor = self.local_layout(global, tensor_local)?;
        let topology = eredu_core::ParallelTopology::new(1, 1, expert_ranks, 1).map_err(invalid)?;
        let rank = ParallelRankTopology::new(topology, expert_rank).map_err(invalid)?;
        let realization = self.local.expert_realization(rank)?;
        let groups = realization.local_global_group_indices();
        let first = *groups
            .first()
            .ok_or_else(|| invalid("expert owner is empty"))?;
        parallel_layout::local_expert_layout(
            tensor,
            expert_local,
            self.local
                .units
                .iter()
                .map(|unit| &unit.feed_forward.feed_forward.experts),
            realization.global_expert_count(),
            first..first + groups.len(),
            expert_ranks,
        )
    }
    /// Constructs one local depth while retaining its tensor collective coordinates.
    pub fn construct_unit<B: GroupedNeuralBackend + DistributedNeuralBackend>(
        &self,
        depth: usize,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionUnit<B>, Error> {
        let unit = self
            .local
            .units
            .get(depth)
            .ok_or_else(|| invalid("prediction depth is out of range"))?;
        PredictionUnit::new_partitioned(unit.clone(), self.rank, self.ranks, context)
    }
    /// Constructs one local depth after validating the complete expert realization.
    pub fn construct_unit_with_experts<B: GroupedNeuralBackend + DistributedNeuralBackend>(
        &self,
        depth: usize,
        realization: &ExpertRealizationPlan<GroupedGatedProductSpec>,
        context: &<B::Tensor as Tensor>::Context,
    ) -> Result<PredictionUnit<B>, Error> {
        let local = self.local.with_expert_realization(realization)?;
        let unit = local
            .units
            .get(depth)
            .ok_or_else(|| invalid("prediction depth is out of range"))?;
        PredictionUnit::new_partitioned(unit.clone(), self.rank, self.ranks, context)
    }
}
impl PredictionSpec {
    /// Derives local mixer/FFN equations with shared target sharding and encoding rules.
    pub fn tensor_partition(
        &self,
        rank: usize,
        ranks: usize,
    ) -> Result<PredictionTensorPartition, Error> {
        if ranks == 0 || ranks > i32::MAX as usize || rank >= ranks {
            return Err(invalid("invalid rank or group size"));
        }
        self.state_layout()?;
        let mut local = self.clone();
        let mut selections = Selections(BTreeMap::new());
        for unit in &mut local.units {
            selections.partition_sublayers(&mut unit.mixer, &mut unit.feed_forward, rank, ranks)?;
        }
        local.state_layout()?;
        Ok(PredictionTensorPartition {
            rank,
            ranks,
            local,
            selections: selections.0,
        })
    }
    /// Authors prediction-bank ownership from already selected tensor geometry.
    pub fn expert_realization(
        &self,
        topology: ParallelRankTopology,
    ) -> Result<ExpertRealizationPlan<GroupedGatedProductSpec>, Error> {
        self.state_layout()?;
        let global = self.global_expert_count()?;
        let groups = eredu_core::balanced_contiguous_range(
            global,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(invalid)?;
        let owner = ExecutionGroupId::new(OWNER).map_err(invalid)?;
        let units = self
            .units
            .iter()
            .map(|unit| {
                Ok((
                    (owner.clone(), unit.depth),
                    local_expert_spec(&unit.feed_forward.feed_forward.experts, groups.clone())?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>, Error>>()?;
        ExpertRealizationPlan::balanced(global, topology, units).map_err(invalid)
    }
    /// Retains router identities while applying an exact prediction expert plan.
    pub fn with_expert_realization(
        &self,
        realization: &ExpertRealizationPlan<GroupedGatedProductSpec>,
    ) -> Result<Self, Error> {
        self.state_layout()?;
        let global = self.global_expert_count()?;
        let groups = eredu_core::balanced_contiguous_range(
            global,
            realization.expert_parallel_size(),
            realization.expert_parallel_rank(),
            false,
        )
        .map_err(invalid)?;
        if realization.global_expert_count() != global
            || realization.local_global_group_indices() != groups.clone().collect::<Vec<_>>()
            || realization.unit_specs().len() != self.units.len()
        {
            return Err(invalid(
                "expert realization differs from prediction ownership",
            ));
        }
        let mut local = self.clone();
        for unit in &mut local.units {
            let expected =
                local_expert_spec(&unit.feed_forward.feed_forward.experts, groups.clone())?;
            let selected = realization
                .unit_spec(OWNER, unit.depth)
                .ok_or_else(|| invalid("expert realization omits prediction depth"))?;
            if selected != &expected {
                return Err(invalid(
                    "expert realization differs from prediction parameters or equation",
                ));
            }
            unit.feed_forward.feed_forward.experts = selected.clone();
            unit.feed_forward.validate()?;
        }
        local.state_layout()?;
        Ok(local)
    }
    fn global_expert_count(&self) -> Result<usize, Error> {
        let first = self
            .units
            .first()
            .ok_or_else(|| invalid("prediction has no depths"))?;
        let global = usize::try_from(first.feed_forward.feed_forward.experts.group_count())
            .map_err(invalid)?;
        if self.units.iter().any(|unit| {
            unit.feed_forward.feed_forward.experts.group_count() as usize != global
                || unit
                    .feed_forward
                    .feed_forward
                    .router
                    .selection()
                    .group_count() as usize
                    != global
        }) {
            return Err(invalid(
                "expert realization requires global bank cardinality",
            ));
        }
        Ok(global)
    }
}
