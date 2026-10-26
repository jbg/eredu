//! Complete DSpark unit placement with sharded body and replicated scoring heads.
use super::{Equation, MuseAssistantDefinition, MuseAssistantPlacement};
use eredu_core::ParallelRankTopology;
use eredu_nn::Error;
use eredu_runtime::{
    derive_partitioned_local_layout, ArchitectureParameterDescription, ExecutionGraph,
};

impl MuseAssistantDefinition {
    pub(crate) fn target_placement(&self, rank: ParallelRankTopology) -> MuseAssistantPlacement {
        match &self.equation {
            Equation::DSpark { body, .. }
                if rank.tensor_parallel_size() > 1
                    && rank.pipeline_parallel_size() == 1
                    && rank.expert_parallel_size() == 1
                    && rank.data_parallel_size() == 1
                    && body.partitions_heads_evenly(rank.tensor_parallel_size()) =>
            {
                MuseAssistantPlacement::TensorParallel(rank)
            }
            _ => MuseAssistantPlacement::Replicated,
        }
    }

    /// Sum operations required by the complete selected proposal block.
    /// Context projection and scoring heads are replicas and need no collective.
    /// Prepared sources retain their floating compute precisions. Standalone
    /// normalized definitions conservatively admit all supported precisions.
    pub fn tensor_parallel_requirements(
        &self,
    ) -> Result<Option<eredu_runtime::CommunicationGroupRequirements>, Error> {
        use eredu_runtime::{
            CommunicationGroupRequirements, CommunicationOperation,
            CommunicationOperationRequirement, CommunicationTensorLimits,
        };
        if self.placement == MuseAssistantPlacement::Replicated {
            return Ok(None);
        }
        let Equation::DSpark { body, .. } = &self.equation else {
            return Err(Error::backend(
                "selected parallel assistant has no reduction equation",
            ));
        };
        let geometry = body.geometry();
        let elements = usize::try_from(geometry.hidden)
            .ok()
            .and_then(|hidden| usize::try_from(geometry.block).ok()?.checked_mul(hidden))
            .ok_or_else(|| Error::backend("assistant reduction extent overflow"))?;
        let limits =
            CommunicationTensorLimits::new(1, 3, elements, None).map_err(Error::backend_source)?;
        let reduction = CommunicationOperationRequirement::tensors(
            CommunicationOperation::AllReduceSum,
            self.compute_precisions.clone(),
            limits,
            true,
        )
        .map_err(Error::backend_source)?;
        CommunicationGroupRequirements::new([reduction])
            .map(Some)
            .map_err(Error::backend_source)
    }

    pub(in crate::muse::assistant) fn source_for_selected(
        &self,
        selected: &Self,
    ) -> Result<Self, Error> {
        let (
            Equation::DSpark { body, heads },
            Equation::DSpark {
                body: selected_body,
                heads: selected_heads,
            },
        ) = (&self.equation, &selected.equation)
        else {
            return Err(Error::backend(
                "local assistant source geometry is implemented for DSpark",
            ));
        };
        let geometry = |g: super::super::dspark::DsparkHeadGeometry| {
            (
                g.hidden,
                g.vocabulary,
                g.rank,
                g.output_multiplier,
                g.soft_cap,
            )
        };
        if geometry(heads.geometry()) != geometry(selected_heads.geometry()) {
            return Err(Error::backend(
                "assistant source and selected head geometries differ",
            ));
        }
        let mut source = Self::dspark(body.source_for_selected(selected_body)?, heads.clone())?;
        source.placement = selected.placement;
        source.compute_precisions = selected.compute_precisions.clone();
        Ok(source)
    }

    /// Selects DSpark body TP and complete scoring replicas under one unit budget.
    /// This does not admit a public distributed speculative session.
    pub fn partition_tensor_parallel(
        &self,
        rank: ParallelRankTopology,
    ) -> Result<(Self, ArchitectureParameterDescription), Error> {
        let Equation::DSpark { body, heads } = &self.equation else {
            return Err(Error::backend(
                "tensor-parallel assistant unit construction is implemented for DSpark",
            ));
        };
        let (body, body_description) = body.partition_tensor_parallel(rank)?;
        let head_description = heads.parameter_description()?;
        let groups = body_description
            .groups()
            .iter()
            .chain(head_description.groups())
            .cloned()
            .collect::<Vec<_>>();
        let mut definition = Self::dspark(body, heads.clone())?;
        let graph = ExecutionGraph::chain((0..definition.unit_layout().group_count()).map(|i| {
            definition
                .unit_layout()
                .group_id(i)
                .unwrap()
                .as_str()
                .to_owned()
        }))
        .map_err(Error::backend_source)?;
        let description = ArchitectureParameterDescription::from_owned(
            &graph,
            definition.unit_layout(),
            groups,
        )
        .map_err(Error::backend_source)?;
        let layout = derive_partitioned_local_layout(&description, rank).map_err(Error::backend)?;
        // Body construction and complete binding consume the same exact selections.
        let (_, selected_body) = body_description.partition_layout().unwrap();
        for (name, tensor) in selected_body.tensors() {
            if layout.tensor(name) != Some(tensor) {
                return Err(Error::backend(
                    "complete DSpark placement differs from selected body layout",
                ));
            }
        }
        let description = description
            .with_partition_layout(rank, layout)
            .map_err(Error::backend_source)?;
        definition.placement = MuseAssistantPlacement::TensorParallel(rank);
        definition.compute_precisions = self.compute_precisions.clone();
        Ok((definition, description))
    }
}
