//! Shared architecture-owned execution-group transport policies.

use eredu_runtime::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureGroupTransport,
    ArchitectureMergeDestination, ArchitectureParallelSubgroup,
};

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
