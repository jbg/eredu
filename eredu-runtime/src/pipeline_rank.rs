//! Rank-local ownership, state and parameter projection from retained declarations.
use crate::{
    ArchitectureGroupKind, ArchitectureGroupPlacement, ArchitectureMergeDestination,
    PartitionOwnership, PartitionState,
};
use eredu_core::ParallelRankTopology;
use std::ops::Range;

/// Invalid rank-local projection of a retained execution declaration.
#[derive(Debug, thiserror::Error)]
pub enum PipelineRankError {
    /// Global graph or pipeline placement is inconsistent.
    #[error(transparent)]
    Placement(#[from] crate::PipelinePlacementError),
    /// Embedded prediction uses its separately retained placement contract.
    #[error("embedded prediction is active in group {group}")]
    ActivePrediction {
        /// Declared prediction group.
        group: crate::ExecutionGroupId,
    },
    /// A group is absent from the canonical invocation layout.
    #[error("execution group {group} has no unit geometry")]
    MissingGroup {
        /// Missing execution-group slot.
        group: usize,
    },
    /// Contiguous decoder-state offsets overflowed.
    #[error("decoder state geometry overflowed")]
    StateOffsetOverflow,
    /// Contiguous state fallback cannot represent the selected state intervals.
    #[error("rank-local state ranges are not contiguous")]
    NonContiguousState,
    /// An explicit architecture state mapping is invalid.
    #[error(transparent)]
    StatePartition(#[from] crate::ArchitectureStatePartitionError),
    /// State layout slicing is invalid.
    #[error(transparent)]
    State(#[from] crate::StateError),
    /// Static-role or local-state ownership is inconsistent.
    #[error(transparent)]
    Ownership(#[from] crate::ArchitecturePartitionError),
    /// Publication coordinates are inconsistent with the selected topology.
    #[error(transparent)]
    Topology(#[from] eredu_core::TopologyError),
    /// The output group has no publication destination.
    #[error("output group has no publication ownership")]
    MissingPublicationOwner,
}

/// Partition construction or binding disagrees with the retained rank admission.
#[derive(Debug, thiserror::Error)]
pub enum PipelineRankBindingError {
    /// The declared groups, state or boundary cannot form a valid partition.
    #[error(transparent)]
    Partition(#[from] crate::ArchitecturePartitionError),
    /// Canonical execution groups or dependencies differ.
    #[error("concrete architecture execution graph differs from admission")]
    Graph,
    /// Global invocation identities or ordering differ.
    #[error("concrete architecture unit layout differs from admission")]
    Units,
    /// Input, output or pinned-role ownership differs.
    #[error("concrete architecture ownership {actual:?} differs from admitted {expected:?}")]
    Ownership {
        /// Constructed partition ownership.
        actual: Box<PartitionOwnership>,
        /// Retained rank ownership.
        expected: Box<PartitionOwnership>,
    },
    /// Ordered group-global invocation intervals differ.
    #[error("concrete architecture groups {actual:?} differ from admitted {expected:?}")]
    Groups {
        /// Constructed group identities and intervals.
        actual: Vec<(crate::ExecutionGroupId, Range<usize>)>,
        /// Retained group identities and intervals.
        expected: Vec<(crate::ExecutionGroupId, Range<usize>)>,
    },
    /// Exact local state geometry or global offsets differ.
    #[error("concrete architecture state {actual:?} differs from admitted {expected:?}")]
    State {
        /// Constructed mutable-state mapping.
        actual: Box<Option<PartitionState>>,
        /// Retained mutable-state mapping.
        expected: Box<Option<PartitionState>>,
    },
}

/// One architecture group interval and its semantic pipeline owner.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PipelineGroupOwnership {
    group: crate::ExecutionGroupId,
    units: Range<usize>,
    pipeline_owner: usize,
}

impl PipelineGroupOwnership {
    /// Stable architecture execution-group identity.
    pub const fn group(&self) -> &crate::ExecutionGroupId {
        &self.group
    }

    /// Group-local architecture-global units owned by this rank.
    pub fn units(&self) -> Range<usize> {
        self.units.clone()
    }

    /// Semantic pipeline coordinate owning these units.
    pub const fn pipeline_owner(&self) -> usize {
        self.pipeline_owner
    }
}

/// Source-free local projection of groups, static roles, state and parameter consumers.
/// This grants no checkpoint, native resource, execution or communication authority.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineRankProjection {
    topology: ParallelRankTopology,
    groups: Vec<PipelineGroupOwnership>,
    ownership: PartitionOwnership,
    publication_owner: usize,
    state: Option<PartitionState>,
    parameter_targets: Vec<String>,
}
impl PipelineRankProjection {
    /// Projects one actual rank. Parameter targets may be omitted for ownership-only
    /// inspection without allocating their names. Prediction placement is separate.
    pub fn new(
        execution: &crate::ReplicatedTextRequirements,
        state_layout: Option<&crate::StateLayout>,
        topology: ParallelRankTopology,
        include_parameter_targets: bool,
    ) -> Result<Self, PipelineRankError> {
        project_pipeline_rank(execution, state_layout, topology, include_parameter_targets)
    }
    /// Actual global and Cartesian rank coordinates.
    pub const fn topology(&self) -> ParallelRankTopology {
        self.topology
    }
    /// Ordered group-global intervals owned by this pipeline stage.
    pub fn groups(&self) -> &[PipelineGroupOwnership] {
        &self.groups
    }
    /// Declared input, output and static-role ownership.
    pub const fn ownership(&self) -> &PartitionOwnership {
        &self.ownership
    }
    /// Global rank supplying the authoritative output.
    pub const fn publication_owner(&self) -> usize {
        self.publication_owner
    }
    /// Exact selected state geometry when state is requested.
    pub const fn state(&self) -> Option<&PartitionState> {
        self.state.as_ref()
    }
    /// Canonical logical parameter consumers retained on this rank.
    pub fn parameter_targets(&self) -> &[String] {
        &self.parameter_targets
    }
    /// Constructs a partition using exactly this rank's admitted groups and ownership.
    /// Family declarations supply parameters, state geometry and transport; the result
    /// is checked against the retained execution and local state before it is returned.
    #[allow(clippy::too_many_arguments)]
    pub fn construct_partition<G, W: crate::ArchitectureBoundary>(
        &self,
        execution: &crate::ReplicatedTextRequirements,
        parameters: &crate::ArchitectureParameterDescription,
        complete_state: &crate::StateLayout,
        state_plan: &crate::ArchitectureStatePartitionPlan,
        local_geometry: G,
        boundary_schema: W,
    ) -> Result<crate::ArchitecturePartition<G, W>, PipelineRankBindingError> {
        let partition = crate::ArchitecturePartition::from_description(
            parameters,
            self.groups
                .iter()
                .map(|group| (group.group().as_str(), group.units())),
            self.ownership.clone(),
            complete_state,
            state_plan,
            local_geometry,
            boundary_schema,
        )?;
        self.validate_partition(execution, &partition)?;
        Ok(partition)
    }

    /// Validates a complete constructed partition against this retained admission.
    /// The execution declaration is the one used to project this rank; geometry,
    /// native bindings and communication authority remain separate contracts.
    pub fn validate_partition<G, W>(
        &self,
        execution: &crate::ReplicatedTextRequirements,
        partition: &crate::ArchitecturePartition<G, W>,
    ) -> Result<(), PipelineRankBindingError> {
        if partition.graph() != execution.execution_graph() {
            return Err(PipelineRankBindingError::Graph);
        }
        if partition.unit_layout() != execution.execution_units() {
            return Err(PipelineRankBindingError::Units);
        }
        if partition.ownership() != self.ownership() {
            return Err(PipelineRankBindingError::Ownership {
                actual: Box::new(partition.ownership().clone()),
                expected: Box::new(self.ownership.clone()),
            });
        }
        if partition.groups().len() != self.groups.len()
            || partition
                .groups()
                .iter()
                .zip(&self.groups)
                .any(|(actual, expected)| {
                    actual.group() != expected.group() || actual.global_units() != expected.units()
                })
        {
            return Err(PipelineRankBindingError::Groups {
                actual: partition
                    .groups()
                    .iter()
                    .map(|group| (group.group().clone(), group.global_units()))
                    .collect(),
                expected: self
                    .groups
                    .iter()
                    .map(|group| (group.group().clone(), group.units()))
                    .collect(),
            });
        }
        if partition.state() != self.state() {
            return Err(PipelineRankBindingError::State {
                actual: Box::new(partition.state().cloned()),
                expected: Box::new(self.state.clone()),
            });
        }
        Ok(())
    }

    /// Consumes the projection for ordinary architecture/native construction.
    pub fn into_parts(
        self,
    ) -> (
        Vec<PipelineGroupOwnership>,
        PartitionOwnership,
        usize,
        Option<PartitionState>,
        Vec<String>,
    ) {
        (
            self.groups,
            self.ownership,
            self.publication_owner,
            self.state,
            self.parameter_targets,
        )
    }
}

fn project_pipeline_rank(
    execution: &crate::ReplicatedTextRequirements,
    state_layout: Option<&crate::StateLayout>,
    topology: ParallelRankTopology,
    include_parameter_targets: bool,
) -> Result<PipelineRankProjection, PipelineRankError> {
    let pp_size = topology.pipeline_parallel_size();
    let pp_rank = topology.pipeline_parallel_rank();
    let mut groups = Vec::new();
    let mut static_roles = Vec::new();
    let mut owns_input = false;
    let output_group = execution.execution_graph().output();
    let mut owns_output = false;
    let mut output_pipeline = None;
    let mut decoder_state_offset = 0usize;
    let mut local_state_ranges = Vec::new();
    let placements = crate::PipelinePlacementPlan::new(
        execution.execution_graph(),
        execution.execution_units(),
        execution.group_transports(),
        pp_size,
    )?;

    for (group_index, transport) in execution.group_transports().iter().enumerate() {
        if transport.kind == ArchitectureGroupKind::Prediction {
            return Err(PipelineRankError::ActivePrediction {
                group: execution
                    .execution_units()
                    .group_id(group_index)
                    .expect("validated group")
                    .clone(),
            });
        }
        let count = execution
            .execution_units()
            .group_range(group_index)
            .ok_or(PipelineRankError::MissingGroup { group: group_index })?
            .len();
        let owners = &placements.owners()[group_index];
        let first_owner = owners.first().map_or(
            if transport.placement == ArchitectureGroupPlacement::OutputOwner {
                pp_size - 1
            } else {
                0
            },
            |(owner, _)| *owner,
        );
        let last_owner = owners.last().map_or(first_owner, |(owner, _)| *owner);
        let merge_owner = match transport.merge_destination {
            ArchitectureMergeDestination::LastOwner => last_owner,
            ArchitectureMergeDestination::FirstPipelineOwner => 0,
            ArchitectureMergeDestination::OutputOwner => pp_size - 1,
        };
        if execution
            .execution_graph()
            .dependencies(group_index)
            .is_some_and(|dependencies| dependencies.is_empty())
            && first_owner == pp_rank
        {
            owns_input = true;
        }
        if group_index == output_group {
            output_pipeline = Some(merge_owner);
            owns_output = merge_owner == pp_rank;
        }
        let first_role_start = static_roles.len();
        if first_owner == pp_rank {
            static_roles.extend(transport.first_owner_static_roles.iter().cloned());
        }
        if last_owner == pp_rank {
            if last_owner == first_owner {
                let first_roles = static_roles[first_role_start..].to_vec();
                static_roles.extend(
                    transport
                        .last_owner_static_roles
                        .iter()
                        .filter(|role| !first_roles.contains(role))
                        .cloned(),
                );
            } else {
                static_roles.extend(transport.last_owner_static_roles.iter().cloned());
            }
        }
        if let Some((_, units)) = owners.iter().find(|(owner, _)| *owner == pp_rank) {
            groups.push(PipelineGroupOwnership {
                group: execution
                    .execution_units()
                    .group_id(group_index)
                    .expect("validated execution layout contains every group")
                    .clone(),
                units: units.clone(),
                pipeline_owner: pp_rank,
            });
            if transport.kind == ArchitectureGroupKind::Decoder {
                local_state_ranges
                    .push(decoder_state_offset + units.start..decoder_state_offset + units.end);
            }
        }
        if transport.kind == ArchitectureGroupKind::Decoder {
            decoder_state_offset = decoder_state_offset
                .checked_add(count)
                .ok_or(PipelineRankError::StateOffsetOverflow)?;
        }
    }
    local_state_ranges.sort_by_key(|range| range.start);
    let state = if let (Some(layout), Some(plan)) = (state_layout, execution.state_partition_plan())
    {
        plan.resolve_partition(
            layout,
            execution.execution_units(),
            |index| {
                let id = execution.execution_units().group_id(index)?;
                groups
                    .iter()
                    .find(|group| &group.group == id)
                    .map(|group| group.units.clone())
            },
            owns_output,
        )?
    } else if let (Some(state_layout), Some(first)) = (state_layout, local_state_ranges.first()) {
        let start = first.start;
        let end = local_state_ranges
            .iter()
            .try_fold(start, |frontier, range| {
                (range.start == frontier)
                    .then_some(range.end)
                    .ok_or(PipelineRankError::NonContiguousState)
            })?;
        Some(PartitionState::new(state_layout.slice(start..end)?, start)?)
    } else {
        None
    };
    let ownership =
        PartitionOwnership::new(owns_input, owns_output, static_roles).and_then(|ownership| {
            ownership
                .with_replicated_static_roles(execution.replicated_static_roles().iter().cloned())
        })?;
    let parameter_targets = execution
        .parameters()
        .iter()
        .filter(|_| include_parameter_targets)
        .filter(|parameter| match parameter.owner() {
            crate::ReplicatedTextParameterOwner::StaticRole(role) => {
                ownership.stores_static_role(role)
            }
            crate::ReplicatedTextParameterOwner::StaticUnitConsumers { role, consumers } => {
                ownership.replicated_static_roles().contains(role)
                    || consumers.iter().any(|(group, unit)| {
                        groups.iter().any(|owned| {
                            owned.group.as_str() == group && owned.units.contains(unit)
                        })
                    })
            }
            crate::ReplicatedTextParameterOwner::ExecutionUnit { group, unit } => groups
                .iter()
                .any(|owned| owned.group.as_str() == group && owned.units.contains(unit)),
        })
        .map(|parameter| parameter.name().to_owned())
        .collect();
    Ok(PipelineRankProjection {
        topology,
        groups,
        ownership,
        publication_owner: topology.global_rank_for(eredu_core::ParallelCoordinates::new(
            0,
            output_pipeline.ok_or(PipelineRankError::MissingPublicationOwner)?,
            0,
            topology.data_parallel_rank(),
        ))?,
        state,
        parameter_targets,
    })
}
