//! Explicit replicated scoring-head ownership alongside a sharded DSpark body.
use super::DsparkHeadDefinition;
use eredu_nn::Error;
use eredu_runtime::{
    expand_linear_format_parameter_group, ArchitectureParameterDescription, ExecutionGraph,
    ExecutionGroupId, MemberSharding, OwnedParameterGroupSpec, ParameterGroupOwner,
    ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
};

impl DsparkHeadDefinition {
    /// Declares complete scoring replicas at their independent acquisition units.
    /// Exact I64 token mapping remains separate from floating-point projections.
    pub fn parameter_description(&self) -> Result<ArchitectureParameterDescription, Error> {
        let g = self.geometry;
        let h = g.hidden as usize;
        let v = g.vocabulary as usize;
        let r = g.rank as usize;
        let ids = (0..self.layout.group_count())
            .map(|i| self.layout.group_id(i).unwrap().as_str().to_owned())
            .collect::<Vec<_>>();
        let graph = ExecutionGraph::chain(ids.clone()).map_err(Error::backend_source)?;
        let declarations = [
            vec![("lm_head.weight", vec![v, h])],
            vec![("markov_head.markov_w1.weight", vec![v, r])],
            vec![("markov_head.markov_w2.weight", vec![v, r])],
            vec![
                ("confidence_head.proj.weight", vec![1, h + r]),
                ("confidence_head.proj.bias", vec![1]),
            ],
            vec![("d2t", vec![v])],
        ];
        let mut owned = Vec::new();
        for (id, members) in ids.iter().zip(declarations) {
            let group = ParameterGroupSpec::new(
                id,
                ParameterRole::Replicated,
                members.into_iter().map(|(name, shape)| {
                    ParameterMemberSpec::new(name, shape, MemberSharding::Replicated)
                }),
            )
            .map_err(Error::backend_source)?;
            let group = expand_linear_format_parameter_group(group, |member| {
                crate::linear_format::standard_parallel_linear_format(
                    member,
                    self.formats
                        .get(member.target())
                        .copied()
                        .map(Into::into)
                        .unwrap_or(eredu_checkpoint::LinearFormat::Dense),
                )
            })
            .map_err(Error::backend_source)?;
            let owner = ParameterGroupOwner::execution_unit(
                ExecutionGroupId::new(id).map_err(Error::backend_source)?,
                0,
            );
            owned.push(OwnedParameterGroupSpec::new(owner, group));
        }
        ArchitectureParameterDescription::new(
            &graph,
            &self.layout,
            owned.iter().map(|group| group.group().clone()),
            owned.clone(),
        )
        .map_err(Error::backend_source)
    }
}
