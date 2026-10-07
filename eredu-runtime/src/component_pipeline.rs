//! Cold consistency checks shared by layered component pipelines.
use crate::{
    ArchitectureBoundary, ArchitectureGroupKind, ArchitectureParameterDescription,
    ArchitecturePartition, BoundaryDimensionContract, BoundaryTensorDimension, BoundaryTensorDtype,
    BoundaryWireSchema, CommunicationManifest, CommunicationRouteDescriptor, ExecutionUnitAddress,
    LocalModelLayout, PartitionExecutionError, PartitionedExecutionPlan, PipelineActivationDtype,
    StateLayout,
};
use eredu_core::{checkpoint::TensorDtype, CollectiveGroupId};

/// Cold mismatch between retained component execution and pipeline binding.
#[derive(Debug, thiserror::Error)]
pub enum PipelineBindingError {
    /// Communication selection violates its exact completion or publication contract.
    #[error(transparent)]
    Communication(#[from] PartitionExecutionError),
    /// The selected tensor cohort cannot safely propagate local unit failures.
    #[error(transparent)]
    LocalAgreement(#[from] crate::PipelineLocalAgreementError),
    /// Architecture-declared parameter, state, or boundary selection differs.
    #[error("component pipeline binding: {0}")]
    Contract(&'static str),
}
/// Validates exact retained parameters, stage ownership, collective topology and wire geometry.
/// `boundary_schema` supplies the architecture's declaration for each directed
/// route, independently of that route's manifest contract. Uniform-width
/// architectures return their partition schema; changing-width architectures
/// derive each declaration from their retained unit geometry and selected cuts.
#[allow(clippy::too_many_arguments)]
pub fn validate_component_pipeline<G, A: ArchitectureBoundary>(
    description: &ArchitectureParameterDescription,
    partition: &ArchitecturePartition<G, A>,
    units: impl Iterator<Item = ExecutionUnitAddress>,
    state_layout: Option<&StateLayout>,
    local_layout: Option<&LocalModelLayout>,
    manifest: &CommunicationManifest,
    plan: &PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
    activation_dtype: PipelineActivationDtype,
    group_contracts: &[(ArchitectureGroupKind, bool)],
    state_group: usize,
    maximum_sequence: usize,
    mut boundary_schema: impl FnMut(
        &CommunicationRouteDescriptor,
    ) -> Result<BoundaryWireSchema, PipelineBindingError>,
) -> Result<(), PipelineBindingError> {
    use PipelineBindingError::Contract;
    if plan.graph() != description.graph()
        || !units.eq(partition.units())
        || state_layout != partition.state().map(|state| state.layout())
        || plan.wire_contract().activation_dtype() != activation_dtype
    {
        return Err(Contract(
            "prepared parameters, graph, state or wire selection differs",
        ));
    }
    if plan.group_contracts() != group_contracts {
        return Err(Contract(
            "execution group semantics differ from the architecture",
        ));
    }
    for (group, driver) in plan.drivers().iter().enumerate() {
        let owned = partition
            .groups()
            .iter()
            .find(|owned| owned.group_index() == group);
        match (owned, driver) {
            (Some(owned), Some(driver))
                if driver.range() == owned.global_units()
                    && driver.owns_input() == partition.ownership().owns_input()
                    && driver.owns_output() == partition.ownership().owns_output()
                    && driver.optional_state_layout()
                        == if group == state_group {
                            state_layout
                        } else {
                            None
                        } => {}
            (None, None) => {}
            _ => {
                return Err(Contract(
                    "execution driver differs from selected stage ownership",
                ))
            }
        }
    }
    plan.validate_manifest(&manifest)?;
    plan.validate_exact_commit_agreement(&manifest)?;
    if manifest.completion_policy().is_none() || plan.publication_authority(&manifest)?.is_none() {
        return Err(Contract(
            "pipeline requires bounded completion and exact output publication",
        ));
    }
    let session = manifest
        .groups()
        .iter()
        .find(|group| Some(group.id()) == plan.commit_barrier())
        .ok_or(Contract("session agreement group is absent"))?;
    if session.members().len() != manifest.world_size() {
        return Err(Contract(
            "session agreement must include every participating rank",
        ));
    }
    let publication = plan.publication().expect("validated output publication");
    let publication_group = manifest
        .groups()
        .iter()
        .find(|group| group.id() == publication.group)
        .ok_or(Contract("publication group is absent"))?;
    if publication_group.members().len() != manifest.world_size() {
        return Err(Contract(
            "output publication must include every participating rank",
        ));
    }
    let activation_tensor_dtype = match activation_dtype {
        PipelineActivationDtype::Float32 => TensorDtype::F32,
        PipelineActivationDtype::Float16 => TensorDtype::F16,
        PipelineActivationDtype::Bfloat16 => TensorDtype::Bf16,
    };
    if !publication_group
        .requirements()
        .operations()
        .iter()
        .any(|operation| {
            operation.operation() == crate::CommunicationOperation::Broadcast
                && operation.exact_completion()
                && operation.dtypes().contains(&activation_tensor_dtype)
        })
    {
        return Err(Contract(
            "publication does not support the selected activation precision",
        ));
    }
    let local_groups = partition.groups();
    let first = local_groups
        .first()
        .ok_or(Contract("stage owns no execution groups"))?
        .group_index();
    let last = local_groups.last().expect("nonempty stage").group_index();
    let incoming = plan
        .routes()
        .iter()
        .filter(|route| route.destination_rank == manifest.rank())
        .collect::<Vec<_>>();
    let outgoing = plan
        .routes()
        .iter()
        .filter(|route| route.source_rank == manifest.rank())
        .collect::<Vec<_>>();
    if incoming.len() != usize::from(!partition.ownership().owns_input())
        || outgoing.len() != usize::from(!partition.ownership().owns_output())
        || incoming
            .iter()
            .any(|route| route.destination_group != first)
        || outgoing.iter().any(|route| route.source_group != last)
    {
        return Err(Contract(
            "routes differ from the stage's single ingress and egress",
        ));
    }
    match (local_layout, tensor_group) {
        (Some(_), Some(id)) => {
            let group = manifest
                .groups()
                .iter()
                .find(|group| group.id() == id)
                .ok_or(Contract("tensor group is absent"))?;
            if group.local_index().is_none() {
                return Err(Contract("selected tensor group does not contain this rank"));
            }
            crate::validate_pipeline_local_agreement(manifest, id)?;
            if !group.requirements().operations().iter().any(|operation| {
                operation.operation() == crate::CommunicationOperation::AllReduceSum
                    && operation.exact_completion()
                    && operation.dtypes().contains(&TensorDtype::F32)
            }) {
                return Err(Contract("tensor group requires exact F32 sum reduction"));
            }
            let topology = eredu_core::ParallelRankTopology::new(
                eredu_core::ParallelTopology::new(group.members().len(), 1, 1, 1)
                    .map_err(|_| Contract("invalid tensor group size"))?,
                group.local_index().expect("validated tensor rank"),
            )
            .map_err(|_| Contract("invalid tensor rank"))?;
            let expected = crate::derive_partitioned_local_layout(description, topology)
                .map_err(|_| Contract("tensor parameter placement cannot be derived"))?;
            if local_layout != Some(&expected) {
                return Err(Contract(
                    "tensor group differs from retained parameter placement",
                ));
            }
        }
        (None, None) => {}
        _ => {
            return Err(Contract(
                "tensor placement and communicator selection disagree",
            ))
        }
    }
    for route in manifest.routes() {
        let schema = boundary_schema(route)?;
        let contract = route
            .boundary_contract()
            .ok_or(Contract("route lacks exact boundary roles"))?;
        let symbolic = std::iter::once(schema.primary())
            .chain(schema.auxiliary())
            .collect::<Vec<_>>();
        if contract.schema() != schema.identity() || contract.roles().len() != symbolic.len() {
            return Err(Contract("route boundary identity or roles differ"));
        }
        for (actual, expected) in contract.roles().iter().zip(symbolic) {
            let dtype = match expected.dtype() {
                BoundaryTensorDtype::Int32 => TensorDtype::I32,
                BoundaryTensorDtype::Float32 => TensorDtype::F32,
                BoundaryTensorDtype::Activation => activation_tensor_dtype.clone(),
                _ => return Err(Contract("unsupported boundary scalar type")),
            };
            if actual.role() != expected.role()
                || actual.dtype() != &dtype
                || actual.shape().len() != expected.shape().len()
            {
                return Err(Contract("route role, scalar type or tensor rank differs"));
            }
            for (actual, expected) in actual.shape().iter().zip(expected.shape()) {
                let maximum = match actual {
                    BoundaryDimensionContract::Fixed(n) => *n,
                    BoundaryDimensionContract::Variable { maximum } => *maximum,
                };
                let valid = match expected {
                    BoundaryTensorDimension::Fixed(n) => {
                        matches!(actual, BoundaryDimensionContract::Fixed(m) if *m == *n as usize)
                    }
                    BoundaryTensorDimension::Batch => maximum > 0 && maximum <= i32::MAX as usize,
                    BoundaryTensorDimension::Sequence => maximum > 0 && maximum <= maximum_sequence,
                };
                if !valid {
                    return Err(Contract(
                        "route tensor dimensions exceed the component contract",
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Checks a linear sequence of global units split across pipeline routes.
/// Architectures supply the exact cut for each route; tensor cohorts must own
/// the same interval so that every member executes the same collective sequence.
/// Compose this with [`validate_component_pipeline`] to validate the plan's
/// endpoints, wire contract, state, parameter placement and completion authority.
#[allow(clippy::too_many_arguments)]
pub fn validate_component_pipeline_ranges(
    count: usize,
    local_range: std::ops::Range<usize>,
    group: usize,
    cuts: impl IntoIterator<Item = (crate::CommunicationRouteId, usize)>,
    manifest: &CommunicationManifest,
    plan: &PartitionedExecutionPlan,
    tensor_group: Option<CollectiveGroupId>,
) -> Result<(), PipelineBindingError> {
    use std::collections::{BTreeMap, BTreeSet};
    use PipelineBindingError::Contract;
    let mut selected = BTreeMap::new();
    for (id, cut) in cuts {
        if cut == 0 || cut >= count || selected.insert(id, cut).is_some() {
            return Err(Contract("invalid or repeated pipeline cut"));
        }
    }
    let route_ids = manifest
        .routes()
        .iter()
        .map(|route| route.id())
        .collect::<BTreeSet<_>>();
    if selected.keys().copied().collect::<BTreeSet<_>>() != route_ids
        || plan
            .routes()
            .iter()
            .map(|route| route.route)
            .collect::<BTreeSet<_>>()
            != route_ids
        || plan
            .routes()
            .iter()
            .any(|route| route.source_group != group || route.destination_group != group)
    {
        return Err(Contract(
            "pipeline cuts must exactly cover the selected group routes",
        ));
    }
    let mut incoming = vec![None; manifest.world_size()];
    let mut outgoing = vec![None; manifest.world_size()];
    for route in manifest.routes() {
        let cut = selected[&route.id()];
        if incoming[route.destination()].replace(cut).is_some()
            || outgoing[route.source()].replace(cut).is_some()
        {
            return Err(Contract(
                "linear stage requires one ingress and one egress at most",
            ));
        }
    }
    let ranges = incoming
        .iter()
        .zip(&outgoing)
        .map(|(start, end)| start.unwrap_or(0)..end.unwrap_or(count))
        .collect::<Vec<_>>();
    if ranges.iter().any(|range| range.is_empty()) || ranges[manifest.rank()] != local_range {
        return Err(Contract(
            "route cuts disagree with contiguous stage ownership",
        ));
    }
    if let Some(id) = tensor_group {
        let cohort = manifest
            .groups()
            .iter()
            .find(|candidate| candidate.id() == id)
            .ok_or(Contract("tensor group is absent"))?;
        if cohort
            .members()
            .iter()
            .any(|rank| ranges[*rank] != ranges[manifest.rank()])
        {
            return Err(Contract("tensor participants own different pipeline units"));
        }
    }
    Ok(())
}
