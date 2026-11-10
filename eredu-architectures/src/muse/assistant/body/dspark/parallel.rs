//! DSpark body parameter ownership and TP-local neural geometry.

use super::DsparkBodyDefinition;
use eredu_core::ParallelRankTopology;
use eredu_nn::{DeclareLogicalParameters, Error};
use eredu_runtime::{
    derive_partitioned_local_layout, ArchitectureParameterDescription, ExecutionGraph,
    ExecutionGroupId, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
    ParameterGroupSpec, ParameterRole,
};

impl DsparkBodyDefinition {
    pub(in crate::muse::assistant) fn partitions_heads_evenly(&self, parts: usize) -> bool {
        let heads = self.parameters.blocks[0].attention.projection.geometry;
        heads.query_heads() as usize % parts == 0 && heads.key_value_heads() as usize % parts == 0
    }

    /// Declares context, decoder and final-norm parameters at their actual unit
    /// boundaries, including all selected physical encoding companions.
    pub fn parameter_description(&self) -> Result<ArchitectureParameterDescription, Error> {
        let edge = &self.parameters.edge;
        let replicated = |name, parameters: Vec<eredu_nn::LogicalParameterDeclaration<'_>>| {
            eredu_runtime::logical_parameter_group(
                name,
                ParameterRole::Replicated,
                None,
                &parameters,
                |_| MemberSharding::Replicated,
            )
            .map_err(Error::backend_source)
        };
        let mut owned = Vec::new();
        let mut append = |group_id: &str, unit: usize, groups: Vec<ParameterGroupSpec>| {
            let id = ExecutionGroupId::new(group_id).map_err(Error::backend_source)?;
            owned.extend(groups.into_iter().map(|group| {
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::execution_unit(id.clone(), unit),
                    group,
                )
            }));
            Ok::<_, Error>(())
        };
        let mut context = edge.projection.logical_parameters()?;
        context.extend(edge.norms[0].logical_parameters()?);
        append(
            "assistant.context",
            0,
            vec![replicated("assistant.context".to_owned(), context)?],
        )?;
        for layer in 0..self.geometry.layers {
            let root = format!("layers.{layer}");
            let feed_forward = format!("{root}.feed_forward");
            let block = &self.parameters.blocks[layer];
            let mut groups = block.attention.parameter_groups(false)?;
            groups.push(crate::muse::text::specs::dense_parameter_group(
                &feed_forward,
                &block.dense,
            )?);
            let scalar = &block.scalar;
            groups.push(replicated(
                format!("{root}.normalization_and_residual"),
                scalar.logical_parameters()?,
            )?);
            append("assistant.decoder", layer, groups)?;
        }
        append(
            "assistant.final_norm",
            0,
            vec![replicated(
                "assistant.final_norm".to_owned(),
                edge.norms[1].logical_parameters()?,
            )?],
        )?;
        let graph = ExecutionGraph::chain([
            "assistant.context",
            "assistant.decoder",
            "assistant.final_norm",
        ])
        .map_err(Error::backend_source)?;
        ArchitectureParameterDescription::from_owned(&graph, self.geometry.unit_layout(), owned)
            .map_err(Error::backend_source)
    }

    /// Selects exact TP placement and derives local head, K/V and dense widths.
    /// This body primitive does not place assistant heads or pipeline stages.
    pub fn partition_tensor_parallel(
        &self,
        rank: ParallelRankTopology,
    ) -> Result<(Self, ArchitectureParameterDescription), Error> {
        if rank.pipeline_parallel_size() != 1
            || rank.expert_parallel_size() != 1
            || rank.data_parallel_size() != 1
        {
            return Err(Error::backend(
                "DSpark body TP construction requires singleton pipeline, expert and data axes",
            ));
        }
        let parts = rank.tensor_parallel_size();
        if !self.partitions_heads_evenly(parts) {
            return Err(Error::backend(
                "DSpark body TP construction requires evenly partitioned query and key/value heads",
            ));
        }
        let description = self.parameter_description()?;
        let layout = derive_partitioned_local_layout(&description, rank).map_err(Error::backend)?;
        let width = |name: &str| {
            layout
                .tensor(name)
                .and_then(|tensor| tensor.local_shape().first())
                .and_then(|width| i32::try_from(*width).ok())
                .filter(|width| *width > 0)
                .ok_or_else(|| Error::backend(format!("invalid DSpark local width for {name}")))
        };
        let mut local = self.clone();
        let mut parameters = self.parameters.as_ref().clone();
        parameters.blocks = self
            .parameters
            .blocks
            .iter()
            .enumerate()
            .map(|(layer, parameters)| {
                let root = format!("layers.{layer}");
                let head = parameters.attention.projection.geometry.key_dimensions();
                let queries = width(&format!("{root}.self_attn.q_proj.weight"))?;
                let keys = width(&format!("{root}.self_attn.k_proj.weight"))?;
                if queries % head != 0 || keys % head != 0 {
                    return Err(Error::backend(
                        "DSpark local projection splits an attention head",
                    ));
                }
                parameters.select_dimensions(
                    queries / head,
                    keys / head,
                    width(&format!("{root}.feed_forward.gate_proj.weight"))?,
                )
            })
            .collect::<Result<_, Error>>()?;
        let keys = parameters.blocks[0]
            .attention
            .projection
            .geometry
            .key_value_heads();
        if parameters
            .blocks
            .iter()
            .any(|b| b.attention.projection.geometry.key_value_heads() != keys)
        {
            return Err(Error::backend(
                "DSpark body context requires uniform local key/value heads",
            ));
        }
        parameters.validate()?;
        local.local_parameters = Some(parameters);
        local.geometry.key_value_heads = keys;
        let description = description
            .with_partition_layout(rank, layout)
            .map_err(Error::backend_source)?;
        Ok((local, description))
    }
}
