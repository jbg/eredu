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
    decoder_declaration().into_owned()
}

pub(crate) fn decoder_declaration() -> eredu_runtime::ArchitectureGroupTransportDeclaration<'static>
{
    eredu_runtime::ArchitectureGroupTransportDeclaration {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::Decoder,
        first_owner_static_roles: &["embedding"],
        last_owner_static_roles: &["norm", "output"],
        merge_destination: ArchitectureMergeDestination::LastOwner,
        parallel_subgroup: Some(ArchitectureParallelSubgroup::Decoder),
        request_optional: false,
    }
}

/// Output-owner embedded-prediction transport without pinned modules.
pub(crate) fn prediction() -> ArchitectureGroupTransport {
    prediction_declaration().into_owned()
}

pub(crate) fn prediction_declaration()
-> eredu_runtime::ArchitectureGroupTransportDeclaration<'static> {
    eredu_runtime::ArchitectureGroupTransportDeclaration {
        placement: ArchitectureGroupPlacement::OutputOwner,
        kind: ArchitectureGroupKind::Prediction,
        first_owner_static_roles: &[],
        last_owner_static_roles: &[],
        merge_destination: ArchitectureMergeDestination::OutputOwner,
        parallel_subgroup: Some(ArchitectureParallelSubgroup::Decoder),
        request_optional: false,
    }
}

/// Shared conditional vision encoder followed by decoder ingress.
pub(crate) fn vision_declaration() -> eredu_runtime::ArchitectureGroupTransportDeclaration<'static>
{
    eredu_runtime::ArchitectureGroupTransportDeclaration {
        placement: ArchitectureGroupPlacement::Pipeline,
        kind: ArchitectureGroupKind::VisionEncoder,
        first_owner_static_roles: &["vision"],
        last_owner_static_roles: &[],
        merge_destination: ArchitectureMergeDestination::FirstPipelineOwner,
        parallel_subgroup: Some(ArchitectureParallelSubgroup::TensorSharded),
        request_optional: true,
    }
}
