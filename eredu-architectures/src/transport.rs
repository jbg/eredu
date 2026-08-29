//! Shared architecture-owned execution-group transport policies.

use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParallelSubgroup, ArchitectureStatePartitionPlan,
    ArchitectureStatePartitionRule, StateLayout,
};

/// Unit-aligned state owned by one pipeline-distributed execution group.
pub(crate) fn pipeline_state(group: usize, layout: &StateLayout) -> ArchitectureStatePartitionPlan {
    ArchitectureStatePartitionPlan::new([ArchitectureStatePartitionRule::group_units(
        group,
        0..layout.len(),
    )])
}

/// Unit-aligned primary state followed by state attached to the output owner.
pub(crate) fn pipeline_with_output_state(
    group: usize,
    primary_layers: usize,
    layout: &StateLayout,
) -> ArchitectureStatePartitionPlan {
    let mut rules = vec![ArchitectureStatePartitionRule::group_units(
        group,
        0..primary_layers,
    )];
    if primary_layers < layout.len() {
        rules.push(ArchitectureStatePartitionRule::output_owner(
            primary_layers..layout.len(),
        ));
    }
    ArchitectureStatePartitionPlan::new(rules)
}

/// Standard pipeline-balanced text-decoder transport.
pub(crate) fn decoder() -> ArchitectureGroupTransport {
    ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::Decoder,
        first_owner_static_roles: vec!["embedding".into()],
        last_owner_static_roles: vec!["norm".into(), "output".into()],
        merge_destination: ArchitectureMergeDestination::LastOwner,
        parallel_subgroup: Some(ArchitectureParallelSubgroup::Decoder),
        request_optional: false,
    }
}

/// Output-owner embedded-prediction transport without pinned modules.
pub(crate) fn prediction() -> ArchitectureGroupTransport {
    ArchitectureGroupTransport {
        placement: ArchitectureGroupPlacement::OutputOwner,
        kind: ArchitectureGroupKind::Prediction,
        first_owner_static_roles: Vec::new(),
        last_owner_static_roles: Vec::new(),
        merge_destination: ArchitectureMergeDestination::OutputOwner,
        parallel_subgroup: Some(ArchitectureParallelSubgroup::Decoder),
        request_optional: false,
    }
}
