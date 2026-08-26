//! Architecture-neutral placement of pipeline execution-group DAGs.
//!
//! Semantic dependencies, physical PP ownership, Cartesian subgroup activity,
//! payload contracts, static tensors, and checkpoint/residency bindings are
//! validated before a weight store is opened. Routes are derived from group
//! ownership and never from numeric pipeline adjacency.

use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

use crate::backend::error::Error;
use eredu_runtime::{
    ArchitecturePartition, ArchitecturePartitionError, ExecutionGraph, ExecutionGroupSpec,
    ExecutionUnitLayout, OwnedParameterGroupSpec, PartitionOwnership, PartitionState, StateLayout,
};

/// Semantic role of one placed execution group.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum ExecutionGroupKind {
    /// Ordered text decoder blocks.
    Decoder,
    /// Output-owner embedded prediction blocks.
    Prediction,
    /// Ordered visual encoder blocks.
    VisionEncoder,
    /// Ordered audio encoder blocks.
    AudioEncoder,
    /// Learned modality projection.
    Projector,
    /// Learned or structural modality merge.
    Merger,
    /// Assembly of modality and text payloads.
    ModalityFinalization,
}

/// Active Cartesian subgroups for a placed group.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ActiveParallelSubgroup {
    /// Units use the semantic TP parameter plan.
    pub tensor_parallel: bool,
    /// Routed components use the EP assignment.
    pub expert_parallel: bool,
}

impl ActiveParallelSubgroup {
    /// TP-sharded, non-routed encoder/projection group.
    pub const fn tensor_sharded() -> Self {
        Self {
            tensor_parallel: true,
            expert_parallel: false,
        }
    }

    /// Decoder group composing with all configured axes.
    pub const fn decoder() -> Self {
        Self {
            tensor_parallel: true,
            expert_parallel: true,
        }
    }
}

/// One PP owner and its contiguous group-global unit interval.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlacedUnitRange {
    /// Owning PP coordinate.
    pub pp_rank: usize,
    /// Contiguous group-global units.
    pub global_units: Range<usize>,
}

/// Static tensor role and its unique PP owner.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct StaticTensorOwnership {
    /// Architecture-authored static role.
    pub role: String,
    /// Owning PP coordinate.
    pub pp_rank: usize,
}

/// Residency identity attached to a placed group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ResidencyBinding {
    /// Prefix used for offload units and telemetry.
    pub unit_prefix: String,
    /// Whether absent request media skips lease acquisition.
    pub request_optional: bool,
}

/// Complete physical placement of one semantic execution group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionGroupPlacement {
    /// Stable group identity.
    pub id: String,
    /// Semantic role.
    pub kind: ExecutionGroupKind,
    /// Dependencies in declaration order.
    pub dependencies: Vec<String>,
    /// Complete group-global geometry.
    pub global_unit_range: Range<usize>,
    /// Ordered PP path and rank-local intervals.
    pub owners: Vec<PlacedUnitRange>,
    /// Active Cartesian subgroups.
    pub active_subgroup: ActiveParallelSubgroup,
    /// Unique static tensor ownership.
    pub static_tensors: Vec<StaticTensorOwnership>,
    /// PP coordinate responsible for the consumer merge.
    pub merge_destination: usize,
    /// Residency binding.
    pub residency: ResidencyBinding,
}

impl ExecutionGroupPlacement {
    /// Returns the local unit interval for `pp_rank`.
    pub fn local_units(&self, pp_rank: usize) -> Option<Range<usize>> {
        self.owners
            .iter()
            .find(|owner| owner.pp_rank == pp_rank)
            .map(|owner| owner.global_units.clone())
    }

    /// Returns the first PP owner.
    pub fn first_owner(&self) -> Option<usize> {
        self.owners.first().map(|owner| owner.pp_rank)
    }
}

/// Planner input for one architecture-authored group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExecutionGroupPlacementRequest {
    /// Stable identity and semantic dependencies.
    pub spec: ExecutionGroupSpec,
    /// Semantic role.
    pub kind: ExecutionGroupKind,
    /// Number of ordered units.
    pub unit_count: usize,
    /// Ordered allowed PP coordinates.
    pub rank_path: Vec<usize>,
    /// Active Cartesian subgroups.
    pub active_subgroup: ActiveParallelSubgroup,
    /// Static roles assigned to the first owner.
    pub first_owner_static_roles: Vec<String>,
    /// Static roles assigned to the terminal owner.
    pub last_owner_static_roles: Vec<String>,
    /// Explicit merge coordinate; defaults to the terminal owner.
    pub merge_destination: Option<usize>,
    /// Residency binding.
    pub residency: ResidencyBinding,
    /// Checkpoint binding group.
    pub checkpoint_group: String,
}

/// Explicit topology-planned transfer between group owners.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlacementRoute {
    /// Producer group.
    pub from_group: String,
    /// Consumer group.
    pub to_group: String,
    /// Producer/merge PP coordinate.
    pub from_pp_rank: usize,
    /// Consumer ingress PP coordinate.
    pub to_pp_rank: usize,
    /// Ordered topology path including endpoints.
    pub pp_path: Vec<usize>,
}

/// One validated authoritative placed execution-group DAG.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlacedExecutionDag {
    groups: Vec<ExecutionGroupPlacement>,
    routes: Vec<PlacementRoute>,
    semantic: ExecutionGraph,
    unit_layout: ExecutionUnitLayout,
    pp_rank_count: usize,
}

/// Runtime reason that two graph-ready groups cannot overlap.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum PlacedGroupSerialReason {
    /// The backend does not permit independent streams on a shared PP owner.
    SharedRankBackend {
        /// Conflicting PP coordinates.
        pp_ranks: Vec<usize>,
    },
    /// Both groups would use one bounded rank-local residency window.
    SharedResidencyWindow {
        /// Conflicting PP coordinates.
        pp_ranks: Vec<usize>,
    },
    /// TP/EP collective submission order must remain identical on shared owners.
    CollectiveOrdering {
        /// Conflicting PP coordinates.
        pp_ranks: Vec<usize>,
    },
}

/// Runtime resources which determine whether graph-ready groups may overlap.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PlacedGroupConcurrencyPolicy {
    /// The local backend supports distinct execution streams.
    pub rank_local_streams: bool,
    /// Rank-local media units share one bounded residency controller/window.
    pub shared_residency_window: bool,
    /// Active tensor-parallel degree.
    pub tensor_parallel_size: usize,
    /// Active expert-parallel degree.
    pub expert_parallel_size: usize,
}

impl PlacedExecutionDag {
    /// Balances non-empty groups over their allowed PP paths and validates all
    /// graph, ownership, schema, static-role, and route invariants.
    pub fn plan(
        pp_rank_count: usize,
        requests: Vec<ExecutionGroupPlacementRequest>,
        output: impl AsRef<str>,
    ) -> Result<Self, Error> {
        if pp_rank_count == 0 {
            return Err(Error::Parallel(
                "placed execution DAG requires at least one PP rank".into(),
            ));
        }
        let semantic = ExecutionGraph::new(
            requests
                .iter()
                .map(|request| request.spec.clone())
                .collect(),
            output.as_ref(),
        )?;
        let unit_layout =
            ExecutionUnitLayout::new(&semantic, requests.iter().map(|request| request.unit_count))
                .map_err(|error| {
                    Error::Parallel(format!("invalid execution-unit placement: {error}"))
                })?;
        let mut groups = Vec::with_capacity(requests.len());
        for request in requests {
            let id = request.spec.id().to_string();
            validate_rank_path(&id, pp_rank_count, &request.rank_path)?;
            let owners = balanced_ranges(request.unit_count, &request.rank_path);
            let first = owners
                .first()
                .map_or(request.rank_path[0], |owner| owner.pp_rank);
            let last = owners.last().map_or(first, |owner| owner.pp_rank);
            let merge_destination = request.merge_destination.unwrap_or(last);
            if merge_destination >= pp_rank_count {
                return Err(Error::Parallel(format!(
                    "execution group {id:?} has impossible merge destination PP rank {merge_destination}"
                )));
            }
            let static_tensors = static_owners(
                &id,
                first,
                last,
                request.first_owner_static_roles,
                request.last_owner_static_roles,
            )?;
            groups.push(ExecutionGroupPlacement {
                id,
                kind: request.kind,
                dependencies: request.spec.dependencies().to_vec(),
                global_unit_range: 0..request.unit_count,
                owners,
                active_subgroup: request.active_subgroup,
                static_tensors,
                merge_destination,
                residency: request.residency,
            });
        }
        let by_id = groups
            .iter()
            .enumerate()
            .map(|(index, group)| (group.id.as_str(), index))
            .collect::<BTreeMap<_, _>>();
        let mut routes = Vec::new();
        for group in &groups {
            for owners in group.owners.windows(2) {
                let from = owners[0].pp_rank;
                let to = owners[1].pp_rank;
                routes.push(PlacementRoute {
                    from_group: group.id.clone(),
                    to_group: group.id.clone(),
                    from_pp_rank: from,
                    to_pp_rank: to,
                    pp_path: vec![from, to],
                });
            }
        }
        for consumer in &groups {
            let to = consumer.first_owner().unwrap_or(consumer.merge_destination);
            for dependency in &consumer.dependencies {
                let producer = &groups[*by_id.get(dependency.as_str()).expect("validated DAG")];
                let from = producer.merge_destination;
                routes.push(PlacementRoute {
                    from_group: producer.id.clone(),
                    to_group: consumer.id.clone(),
                    from_pp_rank: from,
                    to_pp_rank: to,
                    pp_path: if from == to {
                        vec![from]
                    } else {
                        vec![from, to]
                    },
                });
            }
        }
        validate_segment_graph(&groups, &routes)?;
        Ok(Self {
            groups,
            routes,
            semantic,
            unit_layout,
            pp_rank_count,
        })
    }

    /// Returns groups in architecture order.
    pub fn groups(&self) -> &[ExecutionGroupPlacement] {
        &self.groups
    }
    /// Returns topology-planned routes.
    pub fn routes(&self) -> &[PlacementRoute] {
        &self.routes
    }
    /// Returns stable topological group order.
    pub fn execution_order(&self) -> &[usize] {
        self.semantic.execution_order()
    }
    /// Returns the shared semantic DAG used by the ready-set scheduler.
    pub const fn semantic(&self) -> &ExecutionGraph {
        &self.semantic
    }
    #[cfg(test)]
    fn unit_layout(&self) -> &ExecutionUnitLayout {
        &self.unit_layout
    }
    #[cfg(test)]
    fn group(&self, id: &str) -> Option<&ExecutionGroupPlacement> {
        self.groups.iter().find(|group| group.id == id)
    }
    /// Resolves a stable architecture slot by group identity.
    pub fn group_index(&self, id: &str) -> Option<usize> {
        self.groups.iter().position(|group| group.id == id)
    }
    /// Returns dependency slots in declaration/schema order.
    pub fn dependency_indices(&self, group: usize) -> Option<&[usize]> {
        self.semantic.dependencies(group)
    }
    /// Determines whether two ready groups may submit rank-local compute in
    /// parallel. Transport remains ordered separately in stable group order.
    pub fn concurrency_compatibility(
        &self,
        left: usize,
        right: usize,
        policy: PlacedGroupConcurrencyPolicy,
    ) -> Result<(), PlacedGroupSerialReason> {
        let left = &self.groups[left];
        let right = &self.groups[right];
        let left_ranks = group_ranks(left).into_iter().collect::<BTreeSet<_>>();
        let right_ranks = group_ranks(right).into_iter().collect::<BTreeSet<_>>();
        let shared = left_ranks
            .intersection(&right_ranks)
            .copied()
            .collect::<Vec<_>>();
        if shared.is_empty() {
            return Ok(());
        }
        if !policy.rank_local_streams {
            return Err(PlacedGroupSerialReason::SharedRankBackend { pp_ranks: shared });
        }
        if policy.shared_residency_window {
            return Err(PlacedGroupSerialReason::SharedResidencyWindow { pp_ranks: shared });
        }
        let tensor_collective = policy.tensor_parallel_size > 1
            && (left.active_subgroup.tensor_parallel || right.active_subgroup.tensor_parallel);
        let expert_collective = policy.expert_parallel_size > 1
            && (left.active_subgroup.expert_parallel || right.active_subgroup.expert_parallel);
        if tensor_collective || expert_collective {
            return Err(PlacedGroupSerialReason::CollectiveOrdering { pp_ranks: shared });
        }
        Ok(())
    }
    /// Iterates rank-local group intervals.
    pub fn local_groups(
        &self,
        pp_rank: usize,
    ) -> impl Iterator<Item = (&ExecutionGroupPlacement, Range<usize>)> {
        self.groups
            .iter()
            .filter_map(move |group| group.local_units(pp_rank).map(|range| (group, range)))
    }

    /// Realizes rank ownership directly from the concrete neutral
    /// architecture's canonical graph and unit layout.
    #[allow(clippy::too_many_arguments)]
    pub fn realize_architecture_partition<B, S, M, G, A>(
        &self,
        architecture: &M,
        pp_rank: usize,
        state: Option<(StateLayout, usize)>,
        local_geometry: G,
        parameter_bindings: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<ArchitecturePartition<G, A>, Error>
    where
        B: eredu_nn::NeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        M: eredu_runtime::PartitionedLayeredArchitecture<B, S, Boundary = A>,
        M::Error: std::fmt::Display,
        A: eredu_runtime::ArchitectureBoundary,
    {
        let boundary_schema = architecture
            .boundary_schema()
            .map_err(|error| Error::ArchitectureModel(error.to_string()))?;
        self.realize_architecture_partition_with_boundary::<B, S, M, G, A>(
            architecture,
            pp_rank,
            state,
            local_geometry,
            boundary_schema,
            parameter_bindings,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn realize_architecture_partition_with_boundary<B, S, M, G, A>(
        &self,
        architecture: &M,
        pp_rank: usize,
        state: Option<(StateLayout, usize)>,
        local_geometry: G,
        boundary_schema: A,
        parameter_bindings: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<ArchitecturePartition<G, A>, Error>
    where
        B: eredu_nn::NeuralBackend,
        S: eredu_runtime::RuntimeState<B>,
        M: eredu_runtime::LayeredArchitecture<B, S>,
        M::Error: std::fmt::Display,
        A: eredu_runtime::ArchitectureBoundary,
    {
        if pp_rank >= self.pp_rank_count {
            return Err(Error::Parallel(format!(
                "cannot realize PP rank {pp_rank} from {} planned ranks",
                self.pp_rank_count
            )));
        }
        let group_ranges = self
            .local_groups(pp_rank)
            .map(|(group, units)| (group.id.clone(), units))
            .collect::<Vec<_>>();
        let static_roles = self
            .groups
            .iter()
            .flat_map(|group| &group.static_tensors)
            .filter(|owner| owner.pp_rank == pp_rank)
            .map(|owner| owner.role.clone())
            .collect::<Vec<_>>();
        let owns_input = self.groups.iter().enumerate().any(|(index, group)| {
            self.semantic
                .dependencies(index)
                .is_some_and(|dependencies| dependencies.is_empty())
                && group.first_owner().unwrap_or(group.merge_destination) == pp_rank
        });
        let owns_output = self.groups[self.semantic.output()].merge_destination == pp_rank;
        let ownership = PartitionOwnership::new(owns_input, owns_output, static_roles)
            .map_err(placement_partition_error)?;
        let state = state
            .map(|(layout, offset)| PartitionState::new(layout, offset))
            .transpose()
            .map_err(placement_partition_error)?;
        ArchitecturePartition::from_architecture::<B, S, M, _>(
            architecture,
            group_ranges,
            ownership,
            state,
            local_geometry,
            boundary_schema,
            parameter_bindings,
        )
        .map_err(placement_partition_error)
    }
}

fn placement_partition_error(error: ArchitecturePartitionError) -> Error {
    Error::Parallel(format!("invalid architecture partition placement: {error}"))
}

fn validate_rank_path(id: &str, rank_count: usize, ranks: &[usize]) -> Result<(), Error> {
    if ranks.is_empty() {
        return Err(Error::Parallel(format!(
            "execution group {id:?} has no PP rank path"
        )));
    }
    let mut seen = BTreeSet::new();
    for &rank in ranks {
        if rank >= rank_count || !seen.insert(rank) {
            return Err(Error::Parallel(format!(
                "execution group {id:?} has invalid or ambiguous PP owner {rank}"
            )));
        }
    }
    Ok(())
}

fn balanced_ranges(unit_count: usize, ranks: &[usize]) -> Vec<PlacedUnitRange> {
    if unit_count == 0 {
        return Vec::new();
    }
    let active = ranks.len().min(unit_count);
    let base = unit_count / active;
    let remainder = unit_count % active;
    let mut start = 0;
    ranks
        .iter()
        .copied()
        .take(active)
        .enumerate()
        .map(|(index, pp_rank)| {
            let end = start + base + usize::from(index < remainder);
            let owner = PlacedUnitRange {
                pp_rank,
                global_units: start..end,
            };
            start = end;
            owner
        })
        .collect()
}

fn static_owners(
    id: &str,
    first: usize,
    last: usize,
    first_roles: Vec<String>,
    last_roles: Vec<String>,
) -> Result<Vec<StaticTensorOwnership>, Error> {
    let mut roles = BTreeMap::<String, usize>::new();
    for (owner, declared) in [(first, first_roles), (last, last_roles)] {
        for role in declared {
            if let Some(previous) = roles.insert(role.clone(), owner) {
                return Err(Error::Parallel(format!(
                    "execution group {id:?} ambiguously assigns static role {role:?} to PP ranks {previous} and {owner}"
                )));
            }
        }
    }
    Ok(roles
        .into_iter()
        .map(|(role, pp_rank)| StaticTensorOwnership { role, pp_rank })
        .collect())
}

/// Validate physical `(group, rank)` nodes. Ranks are not collapsed because a
/// PP coordinate may execute several topologically ordered groups.
fn validate_segment_graph(
    groups: &[ExecutionGroupPlacement],
    routes: &[PlacementRoute],
) -> Result<(), Error> {
    let mut nodes = BTreeMap::new();
    for group in groups {
        let ranks = group_ranks(group);
        for rank in ranks {
            let next = nodes.len();
            if nodes.insert((group.id.clone(), rank), next).is_some() {
                return Err(Error::Parallel(format!(
                    "execution group {:?} has ambiguous ownership on PP rank {rank}",
                    group.id
                )));
            }
        }
    }
    let mut edges = vec![Vec::new(); nodes.len()];
    let mut indegree = vec![0usize; nodes.len()];
    for group in groups {
        for ranks in group_ranks(group).windows(2) {
            add_edge(
                &nodes,
                &mut edges,
                &mut indegree,
                (&group.id, ranks[0]),
                (&group.id, ranks[1]),
            )?;
        }
    }
    for route in routes {
        add_edge(
            &nodes,
            &mut edges,
            &mut indegree,
            (&route.from_group, route.from_pp_rank),
            (&route.to_group, route.to_pp_rank),
        )?;
    }
    let mut ready = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index))
        .collect::<BTreeSet<_>>();
    let mut visited = 0;
    while let Some(node) = ready.pop_first() {
        visited += 1;
        for &dependent in &edges[node] {
            indegree[dependent] -= 1;
            if indegree[dependent] == 0 {
                ready.insert(dependent);
            }
        }
    }
    if visited != nodes.len() {
        return Err(Error::Parallel(
            "placed execution-group rank graph contains a cycle".into(),
        ));
    }
    Ok(())
}

fn group_ranks(group: &ExecutionGroupPlacement) -> Vec<usize> {
    if group.owners.is_empty() {
        vec![group.merge_destination]
    } else {
        group.owners.iter().map(|owner| owner.pp_rank).collect()
    }
}

fn add_edge(
    nodes: &BTreeMap<(String, usize), usize>,
    edges: &mut [Vec<usize>],
    indegree: &mut [usize],
    from: (&str, usize),
    to: (&str, usize),
) -> Result<(), Error> {
    let resolve = |node: (&str, usize)| {
        nodes
            .get(&(node.0.to_string(), node.1))
            .copied()
            .ok_or_else(|| Error::Parallel("placement route references an unowned group".into()))
    };
    let from = resolve(from)?;
    let to = resolve(to)?;
    if from != to && !edges[from].contains(&to) {
        edges[from].push(to);
        indegree[to] += 1;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{nn::shared::MlxNeuralBackend, runtime::cache::state::MlxHybridState};
    use eredu_nn::{ParameterVisitor, ParameterVisitorMut, Parameterized};
    use eredu_runtime::{
        ArchitectureBoundary, ArchitectureGroupKind, ArchitectureGroupPlacement,
        ArchitectureGroupTransport, ArchitectureMergeDestination, ArchitectureParallelSubgroup,
        ArchitectureParameterDescription, ArchitectureParameters, ExecutionGroupId,
        LayeredArchitecture, LayeredForwardState, MemberSharding, OwnedParameterGroupSpec,
        ParameterGroupOwner, ParameterGroupSpec, ParameterMemberSpec, ParameterRole,
        StaticParameterVisitor, StaticParameterVisitorMut,
    };
    use safemlx::Stream;

    #[derive(Debug)]
    struct EmptyModule;

    impl Parameterized<crate::MlxTensor> for EmptyModule {
        fn visit_parameters<'a, V>(&'a self, _visitor: &mut V)
        where
            V: ParameterVisitor<'a, crate::MlxTensor>,
        {
        }

        fn visit_parameters_mut<'a, V>(&'a mut self, _visitor: &mut V)
        where
            V: ParameterVisitorMut<'a, crate::MlxTensor>,
        {
        }

        fn set_trainable(&mut self, _trainable: bool) {}
    }

    struct FixtureArchitecture {
        graph: ExecutionGraph,
        unit_counts: Vec<usize>,
        static_modules: EmptyModule,
    }

    impl FixtureArchitecture {
        fn new(placed: &PlacedExecutionDag) -> Self {
            let unit_counts = (0..placed.semantic.groups().len())
                .map(|group| {
                    placed
                        .unit_layout
                        .group_range(group)
                        .expect("fixture layout covers every semantic group")
                        .len()
                })
                .collect();
            Self {
                graph: placed.semantic.clone(),
                unit_counts,
                static_modules: EmptyModule,
            }
        }
    }

    impl ArchitectureParameters<MlxNeuralBackend> for FixtureArchitecture {
        type DefinitionError = String;

        fn state_layout(&self) -> Result<eredu_runtime::StateLayout, Self::DefinitionError> {
            let policies = (0..self.unit_counts.iter().sum())
                .map(|_| {
                    eredu_core::cache::LayerCachePolicy::key_value(
                        eredu_core::AttentionPolicy::Full,
                        1,
                        1,
                    )
                    .expect("fixture cache policy")
                })
                .collect::<Vec<_>>();
            eredu_runtime::StateLayout::new(
                eredu_core::LayerSchedule::new(policies.len(), policies)
                    .map_err(|error| error.to_string())?,
            )
            .map_err(|error| error.to_string())
        }

        fn parameter_description(
            &self,
            _context: &Stream,
        ) -> Result<ArchitectureParameterDescription, Self::DefinitionError> {
            let layout = ExecutionUnitLayout::new(&self.graph, self.unit_counts.clone())
                .map_err(|error| error.to_string())?;
            ArchitectureParameterDescription::new(&self.graph, &layout, [], [])
                .map_err(|error| error.to_string())
        }

        fn visit_static_parameters<V>(&self, visitor: &mut V) -> Result<(), V::Error>
        where
            V: StaticParameterVisitor<MlxNeuralBackend>,
        {
            visitor.visit("static", &self.static_modules)
        }

        fn visit_static_parameters_mut<V>(&mut self, visitor: &mut V) -> Result<(), V::Error>
        where
            V: StaticParameterVisitorMut<MlxNeuralBackend>,
        {
            visitor.visit_mut("static", &mut self.static_modules)
        }
    }

    impl LayeredArchitecture<MlxNeuralBackend, MlxHybridState> for FixtureArchitecture {
        type Input<'a> = &'a crate::MlxTensor;
        type StaticModules = EmptyModule;
        type Unit = EmptyModule;
        type ForwardContext = ();
        type RetainedContextValues<'a> = std::iter::Empty<&'a crate::MlxTensor>;
        type Error = String;

        fn group_transport(&self, _group: usize) -> ArchitectureGroupTransport {
            ArchitectureGroupTransport {
                placement: ArchitectureGroupPlacement::Pipeline,
                kind: ArchitectureGroupKind::Decoder,
                first_owner_static_roles: Vec::new(),
                last_owner_static_roles: Vec::new(),
                merge_destination: ArchitectureMergeDestination::LastOwner,
                parallel_subgroup: Some(ArchitectureParallelSubgroup::Decoder),
                request_optional: false,
            }
        }

        fn model_identity(&self) -> &str {
            "placement-fixture"
        }

        fn execution_graph(&self) -> Result<ExecutionGraph, Self::Error> {
            Ok(self.graph.clone())
        }

        fn group_unit_count(&self, group: usize) -> Result<usize, Self::Error> {
            self.unit_counts
                .get(group)
                .copied()
                .ok_or_else(|| format!("unknown fixture group {group}"))
        }

        fn unit_path(&self, group: usize, index: usize) -> Result<String, Self::Error> {
            Ok(format!("fixture.{group}.{index}"))
        }

        fn static_modules(&self) -> &Self::StaticModules {
            &self.static_modules
        }

        fn static_modules_mut(&mut self) -> &mut Self::StaticModules {
            &mut self.static_modules
        }

        fn build_unit(
            &self,
            _group: usize,
            _index: usize,
            _context: &Stream,
        ) -> Result<Self::Unit, Self::Error> {
            Ok(EmptyModule)
        }

        fn begin_forward<'a>(
            &mut self,
            _input: Self::Input<'a>,
            _state: &mut MlxHybridState,
            _context: &Stream,
        ) -> Result<LayeredForwardState<crate::MlxTensor, Self::ForwardContext>, Self::Error>
        {
            Err("placement fixture does not execute".into())
        }

        fn begin_execution_group(
            &mut self,
            _group: usize,
            initial: &crate::MlxTensor,
            _dependencies: &[&crate::MlxTensor],
            _state: &mut MlxHybridState,
            _forward: &mut Self::ForwardContext,
            _context: &Stream,
        ) -> Result<crate::MlxTensor, Self::Error> {
            Ok(initial.clone())
        }

        fn forward_unit(
            &mut self,
            _group: usize,
            _index: usize,
            _unit: &mut Self::Unit,
            hidden: &crate::MlxTensor,
            _state: &mut MlxHybridState,
            _forward: &mut Self::ForwardContext,
            _context: &Stream,
        ) -> Result<crate::MlxTensor, Self::Error> {
            Ok(hidden.clone())
        }

        fn finish_forward(
            &mut self,
            hidden: &crate::MlxTensor,
            _state: &mut MlxHybridState,
            _forward: &Self::ForwardContext,
            _context: &Stream,
        ) -> Result<crate::MlxTensor, Self::Error> {
            Ok(hidden.clone())
        }

        fn retained_context_values<'a>(
            &'a self,
            _forward: &'a Self::ForwardContext,
            _group: usize,
            _index: usize,
        ) -> Self::RetainedContextValues<'a> {
            std::iter::empty()
        }
    }

    fn realize_partition<G, A: eredu_runtime::ArchitectureBoundary>(
        placed: &PlacedExecutionDag,
        pp_rank: usize,
        state: Option<(StateLayout, usize)>,
        local_geometry: G,
        boundary_schema: A,
        parameter_bindings: impl IntoIterator<Item = OwnedParameterGroupSpec>,
    ) -> Result<ArchitecturePartition<G, A>, Error> {
        placed.realize_architecture_partition_with_boundary::<
            MlxNeuralBackend,
            MlxHybridState,
            _,
            _,
            _,
        >(
            &FixtureArchitecture::new(placed),
            pp_rank,
            state,
            local_geometry,
            boundary_schema,
            parameter_bindings,
        )
    }

    fn request(
        id: &str,
        dependencies: &[&str],
        kind: ExecutionGroupKind,
        units: usize,
        ranks: &[usize],
    ) -> ExecutionGroupPlacementRequest {
        ExecutionGroupPlacementRequest {
            spec: if dependencies.is_empty() {
                ExecutionGroupSpec::root(id)
            } else {
                ExecutionGroupSpec::with_dependencies(id, dependencies.iter().copied())
            },
            kind,
            unit_count: units,
            rank_path: ranks.to_vec(),
            active_subgroup: if kind == ExecutionGroupKind::Decoder {
                ActiveParallelSubgroup::decoder()
            } else {
                ActiveParallelSubgroup::tensor_sharded()
            },
            first_owner_static_roles: vec![format!("{id}.input")],
            last_owner_static_roles: vec![format!("{id}.output")],
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: id.into(),
                request_optional: kind != ExecutionGroupKind::Decoder,
            },
            checkpoint_group: id.into(),
        }
    }

    fn parameter(logical: &str, target: &str) -> ParameterGroupSpec {
        ParameterGroupSpec::new(
            logical,
            ParameterRole::Replicated,
            [ParameterMemberSpec::new(
                target,
                vec![2, 2],
                MemberSharding::Replicated,
            )],
        )
        .unwrap()
    }

    fn state_layout(layers: usize) -> StateLayout {
        StateLayout::new(
            eredu_core::LayerSchedule::new(
                layers,
                vec![eredu_core::cache::LayerCachePolicy::NoState; layers],
            )
            .unwrap(),
        )
        .unwrap()
    }

    fn partition_fixture() -> PlacedExecutionDag {
        PlacedExecutionDag::plan(
            2,
            vec![
                request("vision", &[], ExecutionGroupKind::VisionEncoder, 4, &[0, 1]),
                request("text", &["vision"], ExecutionGroupKind::Decoder, 2, &[1]),
            ],
            "text",
        )
        .unwrap()
    }

    #[test]
    fn realizes_neutral_partition_without_absorbing_transport_or_residency() {
        #[derive(Debug, Clone, Eq, PartialEq)]
        struct Geometry(&'static str);

        let placed = partition_fixture();
        let partition = realize_partition(
            &placed,
            1,
            Some((state_layout(2), 7)),
            Geometry("family-local"),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
            [OwnedParameterGroupSpec::new(
                ParameterGroupOwner::execution_unit(ExecutionGroupId::new("text").unwrap(), 0),
                parameter("text.layer", "model.layers.0.weight"),
            )],
        )
        .unwrap();

        assert_eq!(
            partition
                .graph()
                .groups()
                .iter()
                .map(ExecutionGroupSpec::id)
                .collect::<Vec<_>>(),
            ["vision", "text"]
        );
        assert_eq!(partition.unit_layout().group_range(0), Some(0..4));
        assert_eq!(partition.unit_layout().group_range(1), Some(4..6));
        assert_eq!(partition.groups()[0].group().as_str(), "vision");
        assert_eq!(partition.groups()[0].global_units(), 2..4);
        assert_eq!(partition.groups()[1].group().as_str(), "text");
        assert_eq!(partition.groups()[1].global_units(), 0..2);
        assert!(!partition.ownership().owns_input());
        assert!(partition.ownership().owns_output());
        assert_eq!(
            partition.ownership().static_roles(),
            ["vision.output", "text.input", "text.output"]
        );
        assert_eq!(partition.state().unwrap().global_layer_offset(), 7);
        assert_eq!(partition.state().unwrap().global_layers(), 7..9);
        assert_eq!(partition.local_geometry(), &Geometry("family-local"));
        assert!(partition
            .boundary_schema()
            .wire_schema()
            .unwrap()
            .auxiliary()
            .is_empty());
        assert_eq!(partition.parameter_bindings().len(), 1);
        assert_eq!(
            partition.parameter_bindings()[0].members()[0].target(),
            "model.layers.0.weight"
        );

        let ingress = realize_partition(
            &placed,
            0,
            None,
            (),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
            std::iter::empty(),
        )
        .unwrap();
        assert!(ingress.ownership().owns_input());
        assert!(!ingress.ownership().owns_output());
        assert_eq!(ingress.ownership().static_roles(), ["vision.input"]);

        assert_eq!(placed.unit_layout().group_range(0), Some(0..4));
        assert_eq!(
            placed.group("vision").unwrap().residency.unit_prefix,
            "vision"
        );
        assert!(placed
            .routes()
            .iter()
            .any(|route| route.from_pp_rank == 0 && route.to_pp_rank == 1));
    }

    #[test]
    fn output_owned_prediction_selects_untied_shared_embedding_through_mtp_role() {
        let mut target = request("target", &[], ExecutionGroupKind::Decoder, 4, &[0, 1]);
        target.first_owner_static_roles = vec!["embedding".into()];
        target.last_owner_static_roles = vec!["norm".into(), "output".into()];

        let mut prediction = request("mtp_0", &["target"], ExecutionGroupKind::Decoder, 1, &[1]);
        prediction.first_owner_static_roles = vec!["mtp".into()];
        prediction.last_owner_static_roles.clear();

        let placed = PlacedExecutionDag::plan(2, vec![target, prediction], "mtp_0").unwrap();
        let embedding = OwnedParameterGroupSpec::new(
            ParameterGroupOwner::static_any_of(["embedding", "mtp"]),
            parameter("embedding", "model.embed_tokens.weight"),
        );
        let output = realize_partition(
            &placed,
            1,
            None,
            (),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
            [embedding.clone()],
        )
        .unwrap();
        assert!(output.ownership().owns_static_role("mtp"));
        assert!(output.ownership().owns_static_role("output"));
        assert!(!output.ownership().owns_static_role("embedding"));
        assert_eq!(
            output.parameter_bindings(),
            std::slice::from_ref(&embedding)
        );

        let mut target = request("target", &[], ExecutionGroupKind::Decoder, 4, &[0, 1]);
        target.first_owner_static_roles = vec!["embedding".into()];
        target.last_owner_static_roles = vec!["norm".into(), "output".into()];
        let mut prediction = request("mtp_0", &["target"], ExecutionGroupKind::Decoder, 1, &[1]);
        prediction.first_owner_static_roles.clear();
        prediction.last_owner_static_roles.clear();
        let missing_role = PlacedExecutionDag::plan(2, vec![target, prediction], "mtp_0").unwrap();
        assert!(matches!(
            realize_partition(
                &missing_role,
                1,
                None,
                (),
                eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
                [embedding]
            ),
            Err(Error::Parallel(message)) if message.contains("non-local parameter owner")
        ));
    }

    #[test]
    fn translates_neutral_partition_validation_at_the_placement_boundary() {
        let placed = partition_fixture();
        let error = realize_partition(
            &placed,
            1,
            Some((state_layout(2), usize::MAX)),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
            std::iter::empty(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Parallel(message)
                if message.starts_with("invalid architecture partition placement:")
                    && message.contains("overflowed usize")
        ));

        let error = realize_partition(
            &placed,
            1,
            None,
            (),
            eredu_runtime::NoAuxiliaryBoundarySchema::new(8),
            [
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("text.input"),
                    parameter("first", "shared.weight"),
                ),
                OwnedParameterGroupSpec::new(
                    ParameterGroupOwner::static_role("text.output"),
                    parameter("second", "shared.weight"),
                ),
            ],
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Parallel(message)
                if message.starts_with("invalid architecture partition placement:")
                    && message.contains("shared.weight")
        ));
    }

    #[test]
    fn balances_encoder_and_routes_non_adjacent_dependencies() {
        let graph = PlacedExecutionDag::plan(
            4,
            vec![
                request(
                    "vision",
                    &[],
                    ExecutionGroupKind::VisionEncoder,
                    7,
                    &[0, 1, 2],
                ),
                request(
                    "projector",
                    &["vision"],
                    ExecutionGroupKind::Projector,
                    1,
                    &[3],
                ),
                request(
                    "text",
                    &["projector"],
                    ExecutionGroupKind::Decoder,
                    8,
                    &[1, 2, 3],
                ),
            ],
            "text",
        )
        .unwrap();
        assert_eq!(
            graph.group("vision").unwrap().owners,
            vec![
                PlacedUnitRange {
                    pp_rank: 0,
                    global_units: 0..3
                },
                PlacedUnitRange {
                    pp_rank: 1,
                    global_units: 3..5
                },
                PlacedUnitRange {
                    pp_rank: 2,
                    global_units: 5..7
                },
            ]
        );
        assert!(graph.routes().iter().any(|route| {
            route.from_group == "vision" && route.to_group == "vision" && route.pp_path == [0, 1]
        }));
        assert!(graph.routes().iter().any(|route| {
            route.from_group == "projector" && route.to_group == "text" && route.pp_path == [3, 1]
        }));
    }

    #[test]
    fn routes_exclude_unowned_ranks_and_preserve_non_adjacent_owners() {
        let single = PlacedExecutionDag::plan(
            4,
            vec![request(
                "vision",
                &[],
                ExecutionGroupKind::VisionEncoder,
                1,
                &[0, 1, 2, 3],
            )],
            "vision",
        )
        .unwrap();
        assert!(single.routes().is_empty());

        let non_adjacent = PlacedExecutionDag::plan(
            4,
            vec![request(
                "vision",
                &[],
                ExecutionGroupKind::VisionEncoder,
                2,
                &[0, 2],
            )],
            "vision",
        )
        .unwrap();
        assert_eq!(non_adjacent.routes()[0].pp_path, [0, 2]);
    }

    #[test]
    fn differently_ordered_encoder_paths_do_not_form_a_synthetic_rank_cycle() {
        let conflicting = vec![
            request("vision", &[], ExecutionGroupKind::VisionEncoder, 2, &[0, 1]),
            request("audio", &[], ExecutionGroupKind::AudioEncoder, 2, &[1, 0]),
            request(
                "merge",
                &["vision", "audio"],
                ExecutionGroupKind::Merger,
                1,
                &[0],
            ),
        ];
        let graph = PlacedExecutionDag::plan(2, conflicting, "merge").unwrap();
        assert_eq!(graph.group("vision").unwrap().owners[0].pp_rank, 0);
        assert_eq!(graph.group("audio").unwrap().owners[0].pp_rank, 1);
    }

    #[test]
    fn compatibility_explicitly_serializes_collectives_and_shared_windows() {
        let graph = PlacedExecutionDag::plan(
            2,
            vec![
                request("vision", &[], ExecutionGroupKind::VisionEncoder, 2, &[0, 1]),
                request("audio", &[], ExecutionGroupKind::AudioEncoder, 2, &[0, 1]),
                request(
                    "merge",
                    &["vision", "audio"],
                    ExecutionGroupKind::Merger,
                    1,
                    &[0],
                ),
            ],
            "merge",
        )
        .unwrap();
        let resident_pp = PlacedGroupConcurrencyPolicy {
            rank_local_streams: true,
            shared_residency_window: false,
            tensor_parallel_size: 1,
            expert_parallel_size: 1,
        };
        assert_eq!(graph.concurrency_compatibility(0, 1, resident_pp), Ok(()));
        assert!(matches!(
            graph.concurrency_compatibility(
                0,
                1,
                PlacedGroupConcurrencyPolicy {
                    shared_residency_window: true,
                    ..resident_pp
                }
            ),
            Err(PlacedGroupSerialReason::SharedResidencyWindow { .. })
        ));
        assert!(matches!(
            graph.concurrency_compatibility(
                0,
                1,
                PlacedGroupConcurrencyPolicy {
                    tensor_parallel_size: 2,
                    ..resident_pp
                }
            ),
            Err(PlacedGroupSerialReason::CollectiveOrdering { .. })
        ));
    }

    #[test]
    fn rejects_cycle_disconnect_and_ambiguous_owner() {
        let cycle = vec![
            request("a", &["b"], ExecutionGroupKind::VisionEncoder, 1, &[0]),
            request("b", &["a"], ExecutionGroupKind::Decoder, 1, &[1]),
        ];
        assert!(PlacedExecutionDag::plan(2, cycle, "b").is_err());
        let disconnected = vec![
            request("unused", &[], ExecutionGroupKind::AudioEncoder, 1, &[0]),
            request("text", &[], ExecutionGroupKind::Decoder, 1, &[1]),
        ];
        assert!(PlacedExecutionDag::plan(2, disconnected, "text").is_err());
        let ambiguous = request("text", &[], ExecutionGroupKind::Decoder, 2, &[0, 0]);
        assert!(PlacedExecutionDag::plan(2, vec![ambiguous], "text").is_err());
    }
}
