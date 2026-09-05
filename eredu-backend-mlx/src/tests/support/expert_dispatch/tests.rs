#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use eredu_core::{ParallelRankTopology, ParallelTopology};
    use eredu_runtime::ExecutionGroupId;

    use super::*;

    #[test]
    fn native_assignment_lowers_architecture_owner_map_verbatim() {
        let topology = ParallelTopology::new(1, 1, 2, 1).unwrap();
        let rank = ParallelRankTopology::new(topology, 1).unwrap();
        let decoder = ExecutionGroupId::new("text_decoder").unwrap();
        let plan = eredu_architectures::ExpertRealizationPlan::balanced(
            5,
            rank,
            BTreeMap::from([((decoder, 0), ())]),
        )
        .unwrap();

        let assignment = ExpertAssignment::from_realization(&plan).unwrap();
        assert_eq!(assignment.global_expert_count(), 5);
        assert_eq!(assignment.local_global_group_indices(), [3, 4]);
        assert_eq!(
            (0..5)
                .map(|expert| assignment.owner(expert).unwrap())
                .collect::<Vec<_>>(),
            plan.owners()
        );
    }
}
