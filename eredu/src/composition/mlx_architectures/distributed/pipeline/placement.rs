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

use crate::backend::mlx::error::Error;
use eredu_runtime::{ExecutionGraph, ExecutionGroupSpec};

/// Semantic role of one placed execution group.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum ExecutionGroupKind {
    /// Ordered text decoder blocks.
    Decoder,
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

/// One tensor in a routed group payload.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PayloadField {
    /// Stable semantic name.
    pub name: String,
    /// Symbolic shape resolved from request/config geometry.
    pub shape: Vec<String>,
    /// Whether the field is absent when a modality is absent.
    pub optional: bool,
}

impl PayloadField {
    /// Creates a required symbolic field.
    pub fn required(
        name: impl Into<String>,
        shape: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            shape: shape.into_iter().map(Into::into).collect(),
            optional: false,
        }
    }

    /// Marks the field request-optional.
    pub const fn optional(mut self) -> Self {
        self.optional = true;
        self
    }
}

/// Ordered payload contract at a group boundary.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct PayloadSchema {
    /// Schema identity used for route compatibility.
    pub id: String,
    /// Ordered tensor fields.
    pub fields: Vec<PayloadField>,
}

impl PayloadSchema {
    /// Creates a named schema.
    pub fn new(id: impl Into<String>, fields: Vec<PayloadField>) -> Self {
        Self {
            id: id.into(),
            fields,
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

/// Checkpoint selection attached to a placed group.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct CheckpointBinding {
    /// Stable architecture binding group.
    pub group: String,
    /// Static roles selected on declared owners.
    pub static_roles: Vec<String>,
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
    /// Input payload contract.
    pub input_schema: PayloadSchema,
    /// Output payload contract.
    pub output_schema: PayloadSchema,
    /// PP coordinate responsible for the consumer merge.
    pub merge_destination: usize,
    /// Residency binding.
    pub residency: ResidencyBinding,
    /// Checkpoint binding.
    pub checkpoint: CheckpointBinding,
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
    /// Input payload contract.
    pub input_schema: PayloadSchema,
    /// Output payload contract.
    pub output_schema: PayloadSchema,
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
    /// Payload carried by the route.
    pub payload_schema: PayloadSchema,
}

/// One validated authoritative placed execution-group DAG.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct PlacedExecutionDag {
    groups: Vec<ExecutionGroupPlacement>,
    routes: Vec<PlacementRoute>,
    semantic: ExecutionGraph,
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
        pipeline_stages: usize,
        requests: Vec<ExecutionGroupPlacementRequest>,
        output: impl AsRef<str>,
    ) -> Result<Self, Error> {
        if pipeline_stages == 0 {
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
        let mut groups = Vec::with_capacity(requests.len());
        for request in requests {
            let id = request.spec.id().to_string();
            validate_rank_path(&id, pipeline_stages, &request.rank_path)?;
            validate_schema(&id, "input", &request.input_schema)?;
            validate_schema(&id, "output", &request.output_schema)?;
            let owners = balanced_ranges(request.unit_count, &request.rank_path);
            let first = owners
                .first()
                .map_or(request.rank_path[0], |owner| owner.pp_rank);
            let last = owners.last().map_or(first, |owner| owner.pp_rank);
            let merge_destination = request.merge_destination.unwrap_or(last);
            if merge_destination >= pipeline_stages {
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
                checkpoint: CheckpointBinding {
                    group: request.checkpoint_group,
                    static_roles: static_tensors
                        .iter()
                        .map(|owner| owner.role.clone())
                        .collect(),
                },
                static_tensors,
                input_schema: request.input_schema,
                output_schema: request.output_schema,
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
                    payload_schema: group.output_schema.clone(),
                });
            }
        }
        for consumer in &groups {
            let to = consumer.first_owner().unwrap_or(consumer.merge_destination);
            for dependency in &consumer.dependencies {
                let producer = &groups[*by_id.get(dependency.as_str()).expect("validated DAG")];
                if producer.output_schema != consumer.input_schema {
                    return Err(Error::Parallel(format!(
                        "execution-group payload mismatch: {:?} outputs {:?}, but {:?} expects {:?}",
                        producer.id, producer.output_schema.id, consumer.id, consumer.input_schema.id
                    )));
                }
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
                    payload_schema: producer.output_schema.clone(),
                });
            }
        }
        validate_segment_graph(&groups, &routes)?;
        Ok(Self {
            groups,
            routes,
            semantic,
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
    pub(crate) const fn semantic(&self) -> &ExecutionGraph {
        &self.semantic
    }
    /// Resolves a group by identity.
    pub fn group(&self, id: &str) -> Option<&ExecutionGroupPlacement> {
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
}

fn validate_rank_path(id: &str, stages: usize, ranks: &[usize]) -> Result<(), Error> {
    if ranks.is_empty() {
        return Err(Error::Parallel(format!(
            "execution group {id:?} has no PP rank path"
        )));
    }
    let mut seen = BTreeSet::new();
    for &rank in ranks {
        if rank >= stages || !seen.insert(rank) {
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

fn validate_schema(group: &str, direction: &str, schema: &PayloadSchema) -> Result<(), Error> {
    if schema.id.is_empty() {
        return Err(Error::Parallel(format!(
            "execution group {group:?} has an empty {direction} schema id"
        )));
    }
    let mut names = BTreeSet::new();
    for field in &schema.fields {
        if field.name.is_empty()
            || !names.insert(field.name.as_str())
            || field.shape.is_empty()
            || field.shape.iter().any(String::is_empty)
        {
            return Err(Error::Parallel(format!(
                "execution group {group:?} has malformed {direction} payload field {:?}",
                field.name
            )));
        }
    }
    Ok(())
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

    fn schema(id: &str) -> PayloadSchema {
        PayloadSchema::new(
            id,
            vec![PayloadField::required(
                "hidden",
                ["batch", "sequence", "hidden"],
            )],
        )
    }

    fn request(
        id: &str,
        dependencies: &[&str],
        kind: ExecutionGroupKind,
        units: usize,
        ranks: &[usize],
        input: &str,
        output: &str,
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
            input_schema: schema(input),
            output_schema: schema(output),
            merge_destination: None,
            residency: ResidencyBinding {
                unit_prefix: id.into(),
                request_optional: kind != ExecutionGroupKind::Decoder,
            },
            checkpoint_group: id.into(),
        }
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
                    "pixels",
                    "encoded",
                ),
                request(
                    "projector",
                    &["vision"],
                    ExecutionGroupKind::Projector,
                    1,
                    &[3],
                    "encoded",
                    "decoder",
                ),
                request(
                    "text",
                    &["projector"],
                    ExecutionGroupKind::Decoder,
                    8,
                    &[1, 2, 3],
                    "decoder",
                    "logits",
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
                "pixels",
                "encoded",
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
                "pixels",
                "encoded",
            )],
            "vision",
        )
        .unwrap();
        assert_eq!(non_adjacent.routes()[0].pp_path, [0, 2]);
    }

    #[test]
    fn differently_ordered_encoder_paths_do_not_form_a_synthetic_rank_cycle() {
        let conflicting = vec![
            request(
                "vision",
                &[],
                ExecutionGroupKind::VisionEncoder,
                2,
                &[0, 1],
                "media",
                "media",
            ),
            request(
                "audio",
                &[],
                ExecutionGroupKind::AudioEncoder,
                2,
                &[1, 0],
                "media",
                "media",
            ),
            request(
                "merge",
                &["vision", "audio"],
                ExecutionGroupKind::Merger,
                1,
                &[0],
                "media",
                "media",
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
                request(
                    "vision",
                    &[],
                    ExecutionGroupKind::VisionEncoder,
                    2,
                    &[0, 1],
                    "media",
                    "encoded",
                ),
                request(
                    "audio",
                    &[],
                    ExecutionGroupKind::AudioEncoder,
                    2,
                    &[0, 1],
                    "media",
                    "encoded",
                ),
                request(
                    "merge",
                    &["vision", "audio"],
                    ExecutionGroupKind::Merger,
                    1,
                    &[0],
                    "encoded",
                    "encoded",
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
    fn rejects_cycle_disconnect_ambiguous_owner_and_payload_mismatch() {
        let cycle = vec![
            request(
                "a",
                &["b"],
                ExecutionGroupKind::VisionEncoder,
                1,
                &[0],
                "x",
                "x",
            ),
            request("b", &["a"], ExecutionGroupKind::Decoder, 1, &[1], "x", "x"),
        ];
        assert!(PlacedExecutionDag::plan(2, cycle, "b").is_err());
        let disconnected = vec![
            request(
                "unused",
                &[],
                ExecutionGroupKind::AudioEncoder,
                1,
                &[0],
                "audio",
                "audio",
            ),
            request(
                "text",
                &[],
                ExecutionGroupKind::Decoder,
                1,
                &[1],
                "text",
                "logits",
            ),
        ];
        assert!(PlacedExecutionDag::plan(2, disconnected, "text").is_err());
        let ambiguous = request(
            "text",
            &[],
            ExecutionGroupKind::Decoder,
            2,
            &[0, 0],
            "text",
            "logits",
        );
        assert!(PlacedExecutionDag::plan(2, vec![ambiguous], "text").is_err());
        let mismatch = vec![
            request(
                "vision",
                &[],
                ExecutionGroupKind::VisionEncoder,
                1,
                &[0],
                "pixels",
                "encoded",
            ),
            request(
                "text",
                &["vision"],
                ExecutionGroupKind::Decoder,
                1,
                &[1],
                "wrong",
                "logits",
            ),
        ];
        assert!(PlacedExecutionDag::plan(2, mismatch, "text").is_err());
    }
}
