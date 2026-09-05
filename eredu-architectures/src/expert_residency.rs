//! Backend-neutral topology and checkpoint recipes for independent expert residency.

use std::collections::{BTreeMap, BTreeSet};

use eredu_checkpoint::recipe::DerivedWeightRecipe;
use eredu_checkpoint::{recipe::RecipeCatalog, store::TensorSelection};
use eredu_core::{
    balanced_contiguous_range, CollectiveGroupDescriptor, CollectiveGroupId, ParallelAxis,
    ParallelRankTopology,
};
use eredu_nn::Tensor;
use eredu_runtime::{
    AddressableExpertRouteProvider, AddressableExpertRouteRequest, AddressableGatedProductBank,
    CollectiveBackend, CommunicationPeerCounts, CommunicationTensorMetadata, EvenGatherBackend,
    ExecutionGroupId, ExpertPass, ExpertRouteCombination, ExpertRouteExchange,
    ExpertRouteTensorMovement, ParameterBankKey, PartitionCommunication,
    RoutedExpertTensorParallelOutput, VariableAllToAllBackend,
};

/// Complete architecture-derived ownership and rank-local bank construction plan.
///
/// The plan is deliberately independent of a concrete collective runtime. It
/// fixes the global-to-owner mapping once and carries the architecture's exact
/// rank-local construction specification for every routed execution unit.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertRealizationPlan<S> {
    global_expert_count: usize,
    expert_parallel_size: usize,
    expert_parallel_rank: usize,
    owners: Vec<usize>,
    local_global_group_indices: Vec<usize>,
    collective_members: Vec<usize>,
    collective_local_rank: usize,
    unit_specs: BTreeMap<(ExecutionGroupId, usize), S>,
}

impl<S> ExpertRealizationPlan<S> {
    /// Creates a balanced contiguous realization for one architecture rank.
    pub fn balanced(
        global_expert_count: usize,
        topology: ParallelRankTopology,
        unit_specs: BTreeMap<(ExecutionGroupId, usize), S>,
    ) -> Result<Self, ExpertRealizationPlanError> {
        if global_expert_count == 0 {
            return Err(ExpertRealizationPlanError::EmptyExpertBank);
        }
        if unit_specs.is_empty() {
            return Err(ExpertRealizationPlanError::EmptyUnitSchedule);
        }
        let mut owners = vec![0; global_expert_count];
        for owner in 0..topology.expert_parallel_size() {
            let range = balanced_contiguous_range(
                global_expert_count,
                topology.expert_parallel_size(),
                owner,
                false,
            )
            .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))?;
            owners[range].fill(owner);
        }
        let local = balanced_contiguous_range(
            global_expert_count,
            topology.expert_parallel_size(),
            topology.expert_parallel_rank(),
            false,
        )
        .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))?;
        let collective = topology
            .subgroup(ParallelAxis::Expert)
            .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))?;
        Ok(Self {
            global_expert_count,
            expert_parallel_size: topology.expert_parallel_size(),
            expert_parallel_rank: topology.expert_parallel_rank(),
            owners,
            local_global_group_indices: local.collect(),
            collective_members: collective.global_ranks().to_vec(),
            collective_local_rank: collective.rank(),
            unit_specs,
        })
    }

    /// Returns the checkpoint-global routed expert count used by preflight.
    pub const fn global_expert_count(&self) -> usize {
        self.global_expert_count
    }

    /// Returns the expert-axis rank count used to derive ownership.
    pub const fn expert_parallel_size(&self) -> usize {
        self.expert_parallel_size
    }

    /// Returns this rank's coordinate on the expert axis.
    pub const fn expert_parallel_rank(&self) -> usize {
        self.expert_parallel_rank
    }

    /// Returns one owner rank for every checkpoint-global expert identity.
    pub fn owners(&self) -> &[usize] {
        &self.owners
    }

    /// Returns this rank's global expert identities in owner-local order.
    pub fn local_global_group_indices(&self) -> &[usize] {
        &self.local_global_group_indices
    }

    /// Returns the exact rank-local bank specification for an execution unit.
    pub fn unit_spec(&self, owner_group: &str, owner_unit: usize) -> Option<&S> {
        self.unit_specs
            .iter()
            .find(|((group, unit), _)| group.as_str() == owner_group && *unit == owner_unit)
            .map(|(_, spec)| spec)
    }

    /// Returns whether the plan declares any routed unit in an execution group.
    pub fn has_routed_units_in_group(&self, owner_group: &str) -> bool {
        self.unit_specs
            .keys()
            .any(|(group, _)| group.as_str() == owner_group)
    }

    /// Returns every routed execution unit and its rank-local bank specification.
    pub fn unit_specs(&self) -> &BTreeMap<(ExecutionGroupId, usize), S> {
        &self.unit_specs
    }

    pub(crate) fn try_map_unit_specs<T>(
        self,
        mut map: impl FnMut(S) -> Result<T, String>,
    ) -> Result<ExpertRealizationPlan<T>, String> {
        let unit_specs = self
            .unit_specs
            .into_iter()
            .map(|(address, spec)| map(spec).map(|mapped| (address, mapped)))
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(ExpertRealizationPlan {
            global_expert_count: self.global_expert_count,
            expert_parallel_size: self.expert_parallel_size,
            expert_parallel_rank: self.expert_parallel_rank,
            owners: self.owners,
            local_global_group_indices: self.local_global_group_indices,
            collective_members: self.collective_members,
            collective_local_rank: self.collective_local_rank,
            unit_specs,
        })
    }

    /// Translates the architecture's semantic route axis to an opaque generic group.
    pub fn collective_group(
        &self,
        id: CollectiveGroupId,
    ) -> Result<CollectiveGroupDescriptor, ExpertRealizationPlanError> {
        CollectiveGroupDescriptor::new(
            id,
            self.collective_members.clone(),
            self.collective_local_rank,
        )
        .map_err(|error| ExpertRealizationPlanError::InvalidTopology(error.to_string()))
    }
}

/// Failure while translating and executing one routed plan through generic mechanisms.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum RoutedMechanismExecutionError {
    /// The architecture plan did not contain the requested local bank member or unit.
    #[error("invalid routed mechanism plan: {0}")]
    InvalidPlan(String),
    /// A mechanism-only bank or collective operation failed.
    #[error("routed mechanism execution failed: {0}")]
    Mechanism(String),
}

/// Architecture-owned peer counts for a forward expert dispatch and its return path.
///
/// Rows in `count_matrix` are source ranks and columns are destination ranks in
/// the selected opaque group's ordered membership. The backend never receives
/// that routing meaning: it sees only the lowered per-peer counts for one
/// variable-count exchange.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertRouteCountPlan {
    group: CollectiveGroupId,
    group_size: usize,
    local_rank: usize,
    count_matrix: Vec<usize>,
    forward: CommunicationPeerCounts,
    reverse: CommunicationPeerCounts,
}

impl ExpertRouteCountPlan {
    /// Validates count consensus and derives exact forward and reverse exchanges.
    pub fn from_consensus(
        group: CollectiveGroupId,
        local_rank: usize,
        local_send_counts: Vec<usize>,
        count_matrix: Vec<usize>,
    ) -> Result<Self, RoutedMechanismExecutionError> {
        let group_size = local_send_counts.len();
        if group_size == 0 || local_rank >= group_size {
            return Err(RoutedMechanismExecutionError::InvalidPlan(format!(
                "expert route rank {local_rank} is outside group size {group_size}"
            )));
        }
        let expected = group_size.checked_mul(group_size).ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route count matrix size overflowed usize".into(),
            )
        })?;
        if count_matrix.len() != expected {
            return Err(RoutedMechanismExecutionError::InvalidPlan(format!(
                "expert route count consensus has {} entries, expected {expected}",
                count_matrix.len()
            )));
        }
        let local_start = local_rank.checked_mul(group_size).ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route count row offset overflowed usize".into(),
            )
        })?;
        if count_matrix[local_start..local_start + group_size] != local_send_counts {
            return Err(RoutedMechanismExecutionError::InvalidPlan(
                "expert route count consensus changed the local send row".into(),
            ));
        }
        let receive = (0..group_size)
            .map(|source| count_matrix[source * group_size + local_rank])
            .collect::<Vec<_>>();
        local_send_counts
            .iter()
            .chain(&receive)
            .try_fold(0usize, |total, count| total.checked_add(*count))
            .ok_or_else(|| {
                RoutedMechanismExecutionError::InvalidPlan(
                    "expert route peer count total overflowed usize".into(),
                )
            })?;
        let forward =
            CommunicationPeerCounts::new(local_send_counts.clone(), receive.clone(), group_size)
                .map_err(|error| RoutedMechanismExecutionError::InvalidPlan(error.to_string()))?;
        let reverse = CommunicationPeerCounts::new(receive, local_send_counts, group_size)
            .map_err(|error| RoutedMechanismExecutionError::InvalidPlan(error.to_string()))?;
        Ok(Self {
            group,
            group_size,
            local_rank,
            count_matrix,
            forward,
            reverse,
        })
    }

    /// Opaque communication group selected before model construction.
    pub const fn group(&self) -> CollectiveGroupId {
        self.group
    }

    /// Local position in the opaque group's ordered membership.
    pub const fn local_rank(&self) -> usize {
        self.local_rank
    }

    /// Ordered group cardinality.
    pub const fn group_size(&self) -> usize {
        self.group_size
    }

    /// Source-major count matrix produced by neutral count consensus.
    pub fn count_matrix(&self) -> &[usize] {
        &self.count_matrix
    }

    /// Destination-major counts for dispatching rows to their selected owners.
    pub const fn forward(&self) -> &CommunicationPeerCounts {
        &self.forward
    }

    /// Exact transpose direction returning owner results to source ranks.
    pub const fn reverse(&self) -> &CommunicationPeerCounts {
        &self.reverse
    }

    fn validate_rows(
        &self,
        rows: usize,
        direction: ExpertRouteExchangeDirection,
    ) -> Result<(), RoutedMechanismExecutionError> {
        let counts = match direction {
            ExpertRouteExchangeDirection::Forward => self.forward(),
            ExpertRouteExchangeDirection::Reverse => self.reverse(),
        };
        let expected = counts
            .send()
            .iter()
            .try_fold(0usize, |total, count| total.checked_add(*count));
        if expected != Some(rows) {
            return Err(RoutedMechanismExecutionError::InvalidPlan(format!(
                "expert route exchange planned {expected:?} send rows but payload has {rows}"
            )));
        }
        Ok(())
    }
}

/// Direction of one half of the architecture-owned expert exchange.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertRouteExchangeDirection {
    /// Source rows are sent to their selected expert owners.
    Forward,
    /// Owner results are returned in the exact inverse peer layout.
    Reverse,
}

/// Destination-major route packing selected from architecture-global expert IDs.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertRoutePackingPlan {
    source_tokens: usize,
    routes_per_token: usize,
    send_counts: Vec<usize>,
    packed_route_positions: Vec<usize>,
    packed_token_indices: Vec<usize>,
    packed_global_experts: Vec<usize>,
    packed_owner_local_experts: Vec<usize>,
}

impl ExpertRoutePackingPlan {
    /// Validates route IDs and orders them stably by selected owner rank.
    pub fn new<S>(
        realization: &ExpertRealizationPlan<S>,
        source_tokens: usize,
        routes_per_token: usize,
        global_experts: &[usize],
    ) -> Result<Self, RoutedMechanismExecutionError> {
        if routes_per_token == 0
            || source_tokens.checked_mul(routes_per_token) != Some(global_experts.len())
        {
            return Err(RoutedMechanismExecutionError::InvalidPlan(
                "expert route IDs do not match source token and route cardinality".into(),
            ));
        }
        let mut owner_local = vec![0usize; realization.global_expert_count()];
        let mut next_local = vec![0usize; realization.expert_parallel_size()];
        for (global, owner) in realization.owners().iter().copied().enumerate() {
            let local = next_local.get_mut(owner).ok_or_else(|| {
                RoutedMechanismExecutionError::InvalidPlan(format!(
                    "global expert {global} has owner {owner} outside the selected group"
                ))
            })?;
            owner_local[global] = *local;
            *local = local.checked_add(1).ok_or_else(|| {
                RoutedMechanismExecutionError::InvalidPlan(
                    "owner-local expert identity overflowed usize".into(),
                )
            })?;
        }
        let mut by_owner = vec![Vec::new(); realization.expert_parallel_size()];
        for (position, global) in global_experts.iter().copied().enumerate() {
            let owner = realization.owners().get(global).copied().ok_or_else(|| {
                RoutedMechanismExecutionError::InvalidPlan(format!(
                    "route position {position} selects invalid global expert {global}"
                ))
            })?;
            by_owner[owner].push((position, global, owner_local[global]));
        }
        let send_counts = by_owner.iter().map(Vec::len).collect::<Vec<_>>();
        let mut packed_route_positions = Vec::with_capacity(global_experts.len());
        let mut packed_token_indices = Vec::with_capacity(global_experts.len());
        let mut packed_global_experts = Vec::with_capacity(global_experts.len());
        let mut packed_owner_local_experts = Vec::with_capacity(global_experts.len());
        for routes in by_owner {
            for (position, global, local) in routes {
                packed_route_positions.push(position);
                packed_token_indices.push(position / routes_per_token);
                packed_global_experts.push(global);
                packed_owner_local_experts.push(local);
            }
        }
        Ok(Self {
            source_tokens,
            routes_per_token,
            send_counts,
            packed_route_positions,
            packed_token_indices,
            packed_global_experts,
            packed_owner_local_experts,
        })
    }

    /// Source token rows before route expansion.
    pub const fn source_tokens(&self) -> usize {
        self.source_tokens
    }

    /// Selected expert slots per source token.
    pub const fn routes_per_token(&self) -> usize {
        self.routes_per_token
    }

    /// Exact destination-major route counts, including idle peers.
    pub fn send_counts(&self) -> &[usize] {
        &self.send_counts
    }

    /// Original flattened route positions in destination-major transport order.
    pub fn packed_route_positions(&self) -> &[usize] {
        &self.packed_route_positions
    }

    /// Source token row for every packed route.
    pub fn packed_token_indices(&self) -> &[usize] {
        &self.packed_token_indices
    }

    /// Architecture-global expert identity for every packed route.
    pub fn packed_global_experts(&self) -> &[usize] {
        &self.packed_global_experts
    }

    /// Dense owner-local expert identity for grouped execution.
    pub fn packed_owner_local_experts(&self) -> &[usize] {
        &self.packed_owner_local_experts
    }
}

/// Performs one neutral count-consensus gather before any variable exchange.
#[allow(clippy::too_many_arguments)]
pub fn agree_expert_route_counts<B, G, R, I>(
    group: CollectiveGroupId,
    local_rank: usize,
    local_send_counts: Vec<usize>,
    communication: &PartitionCommunication<B, G, R, I>,
    executor: &B::Executor,
    context: &<B::Tensor as Tensor>::Context,
) -> Result<ExpertRouteCountPlan, RoutedMechanismExecutionError>
where
    B: EvenGatherBackend,
    G: std::borrow::Borrow<B::CommunicationGroup>,
    R: std::borrow::Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    let group_size = local_send_counts.len();
    let descriptor = communication
        .manifest()
        .groups()
        .iter()
        .find(|candidate| candidate.id() == group)
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert count-consensus group is absent from the communication manifest".into(),
            )
        })?;
    if descriptor.members().len() != group_size || descriptor.local_index() != Some(local_rank) {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "expert count consensus differs from opaque group membership".into(),
        ));
    }
    let shape = [i32::try_from(group_size).map_err(|_| {
        RoutedMechanismExecutionError::InvalidPlan(
            "expert route group size exceeds i32 tensor geometry".into(),
        )
    })?];
    let values = local_send_counts
        .iter()
        .map(|count| {
            i32::try_from(*count).map_err(|_| {
                RoutedMechanismExecutionError::InvalidPlan(
                    "expert route count exceeds i32 consensus geometry".into(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let local = B::Tensor::from_i32_slice(&values, &shape, context)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let gathered = communication
        .all_gather_even(local, 0, group, executor)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let matrix = gathered
        .to_i32_vec(context)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?
        .into_iter()
        .map(|count| {
            usize::try_from(count).map_err(|_| {
                RoutedMechanismExecutionError::InvalidPlan(
                    "expert route count consensus returned a negative count".into(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ExpertRouteCountPlan::from_consensus(group, local_rank, local_send_counts, matrix)
}

/// Exchanges already packed expert rows through an opaque generic mechanism.
#[allow(clippy::too_many_arguments)]
pub fn exchange_expert_rows<B, G, R, I>(
    counts: &ExpertRouteCountPlan,
    direction: ExpertRouteExchangeDirection,
    value: B::Tensor,
    communication: &PartitionCommunication<B, G, R, I>,
    executor: &B::Executor,
) -> Result<B::Tensor, RoutedMechanismExecutionError>
where
    B: VariableAllToAllBackend,
    G: std::borrow::Borrow<B::CommunicationGroup>,
    R: std::borrow::Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    let descriptor = communication
        .manifest()
        .groups()
        .iter()
        .find(|candidate| candidate.id() == counts.group())
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route group is absent from the communication manifest".into(),
            )
        })?;
    if descriptor.members().len() != counts.group_size()
        || descriptor.local_index() != Some(counts.local_rank())
    {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "expert route count plan differs from opaque group membership".into(),
        ));
    }
    let maximum = descriptor
        .requirements()
        .operations()
        .iter()
        .find(|requirement| {
            requirement.operation() == eredu_runtime::CommunicationOperation::VariableAllToAll
        })
        .and_then(|requirement| requirement.limits())
        .and_then(|limits| limits.max_count_per_peer())
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route group omitted variable-count exchange limits".into(),
            )
        })?;
    if counts
        .forward()
        .send()
        .iter()
        .chain(counts.forward().receive())
        .any(|count| *count > maximum)
    {
        return Err(RoutedMechanismExecutionError::InvalidPlan(format!(
            "expert route peer count exceeds selected maximum {maximum}"
        )));
    }
    let shape = value.shape();
    let rows = shape
        .first()
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route payload must have a nonnegative leading row dimension".into(),
            )
        })?;
    counts.validate_rows(rows, direction)?;
    let peer_counts = match direction {
        ExpertRouteExchangeDirection::Forward => counts.forward(),
        ExpertRouteExchangeDirection::Reverse => counts.reverse(),
    };
    let expected_rows = peer_counts
        .receive()
        .iter()
        .try_fold(0usize, |total, count| {
            total.checked_add(*count).ok_or_else(|| {
                RoutedMechanismExecutionError::InvalidPlan(
                    "expert route receive row count overflowed".into(),
                )
            })
        })?;
    let output = communication
        .variable_all_to_all(value, peer_counts, 0, counts.group(), executor)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let actual_rows = output
        .shape()
        .first()
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            RoutedMechanismExecutionError::Mechanism(
                "expert route exchange returned an invalid leading row dimension".into(),
            )
        })?;
    if actual_rows != expected_rows {
        return Err(RoutedMechanismExecutionError::Mechanism(format!(
            "expert route exchange returned {actual_rows} rows, expected {expected_rows}"
        )));
    }
    Ok(output)
}

/// Borrowed opaque communication adapter for one half of expert exchange.
///
/// Integer metadata uses the same selected variable-count operation as tensor
/// rows, so manifest dtype/count/shape validation and exact completion remain
/// centralized in [`PartitionCommunication`].
pub struct PartitionExpertRouteExchange<'a, B, G, R, I>
where
    B: VariableAllToAllBackend,
{
    counts: &'a ExpertRouteCountPlan,
    direction: ExpertRouteExchangeDirection,
    communication: &'a PartitionCommunication<B, G, R, I>,
    executor: &'a B::Executor,
    context: &'a <B::Tensor as Tensor>::Context,
}

impl<'a, B, G, R, I> PartitionExpertRouteExchange<'a, B, G, R, I>
where
    B: VariableAllToAllBackend,
{
    /// Binds one checked direction to already realized opaque resources.
    pub const fn new(
        counts: &'a ExpertRouteCountPlan,
        direction: ExpertRouteExchangeDirection,
        communication: &'a PartitionCommunication<B, G, R, I>,
        executor: &'a B::Executor,
        context: &'a <B::Tensor as Tensor>::Context,
    ) -> Self {
        Self {
            counts,
            direction,
            communication,
            executor,
            context,
        }
    }

    fn expected_counts(&self) -> &CommunicationPeerCounts {
        match self.direction {
            ExpertRouteExchangeDirection::Forward => self.counts.forward(),
            ExpertRouteExchangeDirection::Reverse => self.counts.reverse(),
        }
    }
}

impl<B, G, R, I> ExpertRouteExchange<B::Tensor> for PartitionExpertRouteExchange<'_, B, G, R, I>
where
    B: VariableAllToAllBackend,
    G: std::borrow::Borrow<B::CommunicationGroup>,
    R: std::borrow::Borrow<B::CommunicationRoute>,
    I: CommunicationTensorMetadata<B>,
{
    type Error = RoutedMechanismExecutionError;

    fn exchange_tensor(
        &mut self,
        counts: &CommunicationPeerCounts,
        value: B::Tensor,
    ) -> Result<B::Tensor, Self::Error> {
        if counts != self.expected_counts() {
            return Err(RoutedMechanismExecutionError::InvalidPlan(
                "expert route exchange received counts for the wrong direction".into(),
            ));
        }
        exchange_expert_rows(
            self.counts,
            self.direction,
            value,
            self.communication,
            self.executor,
        )
    }

    fn exchange_indices(
        &mut self,
        counts: &CommunicationPeerCounts,
        values: Vec<usize>,
    ) -> Result<Vec<usize>, Self::Error> {
        if counts != self.expected_counts() {
            return Err(RoutedMechanismExecutionError::InvalidPlan(
                "expert route metadata received counts for the wrong direction".into(),
            ));
        }
        let values = values
            .into_iter()
            .map(|value| {
                i32::try_from(value).map_err(|_| {
                    RoutedMechanismExecutionError::InvalidPlan(
                        "expert route metadata exceeds i32 transport geometry".into(),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rows = i32::try_from(values.len()).map_err(|_| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route metadata row count exceeds i32 geometry".into(),
            )
        })?;
        let tensor = B::Tensor::from_i32_slice(&values, &[rows, 1], self.context)
            .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
        let exchanged = exchange_expert_rows(
            self.counts,
            self.direction,
            tensor,
            self.communication,
            self.executor,
        )?;
        // Metadata is resolved on the host below. Complete the selected native
        // dependency on every participant first, including peers whose exact
        // exchange has zero rows. Otherwise an empty peer can advance to the
        // next world-backed collective while a non-empty peer is still
        // materializing this exchange.
        self.communication
            .complete_execution_dependencies(std::iter::once(&exchanged), self.executor)
            .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
        exchanged
            .to_i32_vec(self.context)
            .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?
            .into_iter()
            .map(|value| {
                usize::try_from(value).map_err(|_| {
                    RoutedMechanismExecutionError::Mechanism(
                        "expert route exchange returned negative metadata".into(),
                    )
                })
            })
            .collect()
    }
}

/// Forces the shared exchange engine through its complete-output provider path.
struct CompleteRouteProvider<'a, P>(&'a mut P);

impl<T, P> AddressableExpertRouteProvider<T> for CompleteRouteProvider<'_, P>
where
    P: AddressableExpertRouteProvider<T>,
{
    type Error = P::Error;

    fn execute_addressable_routes(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<T, Self::Error> {
        self.0.execute_addressable_routes(request)
    }

    fn execute_addressable_routes_tensor_parallel(
        &mut self,
        request: AddressableExpertRouteRequest<'_, T>,
    ) -> Result<RoutedExpertTensorParallelOutput<T>, Self::Error> {
        self.0
            .execute_addressable_routes(request)
            .map(RoutedExpertTensorParallelOutput::Complete)
    }
}

/// Executes a complete architecture-owned expert dispatch and return protocol.
///
/// The caller supplies architecture-selected destination-major packing and
/// source-major count consensus. All structural relationships are checked
/// before tensor movement, provider execution, or communication. The local
/// provider receives one route per row and applies its coefficient; duplicate
/// source-token routes are combined additively only after the exact reverse
/// permutation has been verified.
#[allow(clippy::too_many_arguments)]
pub fn execute_expert_route_exchange<S, T, M, X, P>(
    realization: &ExpertRealizationPlan<S>,
    packing: &ExpertRoutePackingPlan,
    counts: &ExpertRouteCountPlan,
    input: &T,
    selected_scores: &T,
    coefficients: &T,
    unit: usize,
    pass: ExpertPass,
    movement: &mut M,
    forward: &mut X,
    reverse: &mut X,
    provider: &mut P,
) -> Result<T, RoutedMechanismExecutionError>
where
    M: ExpertRouteTensorMovement<T>,
    M::Error: std::fmt::Display,
    X: ExpertRouteExchange<T>,
    X::Error: std::fmt::Display,
    P: AddressableExpertRouteProvider<T>,
    P::Error: std::fmt::Display,
{
    let mut provider = CompleteRouteProvider(provider);
    match execute_expert_route_exchange_tensor_parallel(
        realization,
        packing,
        counts,
        input,
        selected_scores,
        coefficients,
        unit,
        pass,
        movement,
        forward,
        reverse,
        &mut provider,
    )? {
        RoutedExpertTensorParallelOutput::Complete(output) => Ok(output),
        RoutedExpertTensorParallelOutput::Partial(_) => {
            Err(RoutedMechanismExecutionError::InvalidPlan(
                "complete expert exchange received an unconsumed tensor-parallel partial".into(),
            ))
        }
    }
}

/// Executes an architecture-owned expert dispatch without collapsing TP work.
///
/// Both the rank-local activation contribution and its optional replicated
/// selection-weighted down bias traverse the exact inverse route permutation.
/// The returned partial is still awaiting the caller's tensor all-sum; adding
/// `post_reduce` before that sum would multiply the bias by the TP group size.
#[allow(clippy::too_many_arguments)]
pub fn execute_expert_route_exchange_tensor_parallel<S, T, M, X, P>(
    realization: &ExpertRealizationPlan<S>,
    packing: &ExpertRoutePackingPlan,
    counts: &ExpertRouteCountPlan,
    input: &T,
    selected_scores: &T,
    coefficients: &T,
    unit: usize,
    pass: ExpertPass,
    movement: &mut M,
    forward: &mut X,
    reverse: &mut X,
    provider: &mut P,
) -> Result<RoutedExpertTensorParallelOutput<T>, RoutedMechanismExecutionError>
where
    M: ExpertRouteTensorMovement<T>,
    M::Error: std::fmt::Display,
    X: ExpertRouteExchange<T>,
    X::Error: std::fmt::Display,
    P: AddressableExpertRouteProvider<T>,
    P::Error: std::fmt::Display,
{
    validate_expert_route_execution(
        realization,
        packing,
        counts,
        input,
        selected_scores,
        coefficients,
        movement,
    )?;

    let packed_input = movement
        .gather_rows(input, packing.packed_token_indices())
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let packed_scores = movement
        .gather_route_values(selected_scores, packing.packed_route_positions())
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let packed_coefficients = movement
        .gather_route_values(coefficients, packing.packed_route_positions())
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let packed_rows = packing.packed_route_positions().len();
    let hidden = movement.shape(input)[1];
    validate_packed_tensor_shape(movement, &packed_input, packed_rows, Some(hidden), "input")?;
    validate_packed_tensor_shape(movement, &packed_scores, packed_rows, Some(1), "scores")?;
    validate_packed_tensor_shape(
        movement,
        &packed_coefficients,
        packed_rows,
        Some(1),
        "coefficients",
    )?;

    let received_global_experts = forward
        .exchange_indices(counts.forward(), packing.packed_global_experts().to_vec())
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let received_local_experts = forward
        .exchange_indices(
            counts.forward(),
            packing.packed_owner_local_experts().to_vec(),
        )
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let received_route_tags = forward
        .exchange_indices(counts.forward(), packing.packed_route_positions().to_vec())
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let received_input = forward
        .exchange_tensor(counts.forward(), packed_input)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let received_scores = forward
        .exchange_tensor(counts.forward(), packed_scores)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let received_coefficients = forward
        .exchange_tensor(counts.forward(), packed_coefficients)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;

    let received_rows = counts
        .forward()
        .receive()
        .iter()
        .try_fold(0usize, |total, count| total.checked_add(*count))
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route receive row count overflowed usize".into(),
            )
        })?;
    if received_global_experts.len() != received_rows
        || received_local_experts.len() != received_rows
        || received_route_tags.len() != received_rows
    {
        return Err(RoutedMechanismExecutionError::Mechanism(format!(
            "expert route metadata returned ({}, {}, {}) rows, expected {received_rows}",
            received_global_experts.len(),
            received_local_experts.len(),
            received_route_tags.len()
        )));
    }
    validate_packed_tensor_shape(
        movement,
        &received_input,
        received_rows,
        Some(hidden),
        "received input",
    )?;
    validate_packed_tensor_shape(
        movement,
        &received_scores,
        received_rows,
        Some(1),
        "received scores",
    )?;
    validate_packed_tensor_shape(
        movement,
        &received_coefficients,
        received_rows,
        Some(1),
        "received coefficients",
    )?;
    validate_received_expert_identities(
        realization,
        &received_global_experts,
        &received_local_experts,
    )?;

    let output = provider
        .execute_addressable_routes_tensor_parallel(AddressableExpertRouteRequest {
            unit,
            input: &received_input,
            global_experts: &received_global_experts,
            owner_local_experts: &received_local_experts,
            selected_scores: &received_scores,
            coefficients: &received_coefficients,
            pass,
            access: pass.parameter_bank_access(),
            combination: ExpertRouteCombination::CoefficientWeightedSum,
        })
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let (complete, reducible, post_reduce) = match output {
        RoutedExpertTensorParallelOutput::Complete(output) => (true, output, None),
        RoutedExpertTensorParallelOutput::Partial(output) => {
            let (reducible, post_reduce) = output.into_parts();
            (false, reducible, post_reduce)
        }
    };
    validate_packed_tensor_shape(
        movement,
        &reducible,
        received_rows,
        Some(hidden),
        "provider activation contribution",
    )?;
    if let Some(bias) = post_reduce.as_ref() {
        validate_packed_tensor_shape(
            movement,
            bias,
            received_rows,
            Some(hidden),
            "provider post-reduction bias",
        )?;
    }

    let returned = reverse
        .exchange_tensor(counts.reverse(), reducible)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let returned_bias = post_reduce
        .map(|bias| {
            reverse
                .exchange_tensor(counts.reverse(), bias)
                .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))
        })
        .transpose()?;
    let returned_route_tags = reverse
        .exchange_indices(counts.reverse(), received_route_tags)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    validate_packed_tensor_shape(
        movement,
        &returned,
        packed_rows,
        Some(hidden),
        "returned output",
    )?;
    if let Some(bias) = returned_bias.as_ref() {
        validate_packed_tensor_shape(
            movement,
            bias,
            packed_rows,
            Some(hidden),
            "returned post-reduction bias",
        )?;
    }
    if returned_route_tags != packing.packed_route_positions() {
        return Err(RoutedMechanismExecutionError::Mechanism(
            "expert route return order differs from the dispatched route permutation".into(),
        ));
    }
    let output = movement
        .scatter_add_rows(
            returned,
            packing.packed_token_indices(),
            packing.source_tokens(),
        )
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let post_reduce = returned_bias
        .map(|bias| {
            movement
                .scatter_add_rows(
                    bias,
                    packing.packed_token_indices(),
                    packing.source_tokens(),
                )
                .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))
        })
        .transpose()?;
    Ok(if complete {
        RoutedExpertTensorParallelOutput::Complete(output)
    } else {
        RoutedExpertTensorParallelOutput::Partial(eredu_nn::TensorParallelGroupedOutput::new(
            output,
            post_reduce,
        ))
    })
}

fn validate_received_expert_identities<S>(
    realization: &ExpertRealizationPlan<S>,
    global_experts: &[usize],
    owner_local_experts: &[usize],
) -> Result<(), RoutedMechanismExecutionError> {
    if global_experts.len() != owner_local_experts.len() {
        return Err(RoutedMechanismExecutionError::Mechanism(
            "expert route global and owner-local identity counts differ".into(),
        ));
    }
    let local_globals = realization.local_global_group_indices();
    for (row, (&global, &owner_local)) in global_experts.iter().zip(owner_local_experts).enumerate()
    {
        if local_globals.get(owner_local) != Some(&global)
            || realization.owners().get(global) != Some(&realization.expert_parallel_rank())
        {
            return Err(RoutedMechanismExecutionError::Mechanism(format!(
                "expert route row {row} maps global expert {global} to invalid owner-local identity {owner_local}"
            )));
        }
    }
    Ok(())
}

fn validate_expert_route_execution<S, T, M>(
    realization: &ExpertRealizationPlan<S>,
    packing: &ExpertRoutePackingPlan,
    counts: &ExpertRouteCountPlan,
    input: &T,
    selected_scores: &T,
    coefficients: &T,
    movement: &M,
) -> Result<(), RoutedMechanismExecutionError>
where
    M: ExpertRouteTensorMovement<T>,
{
    if counts.group_size() != realization.expert_parallel_size()
        || counts.local_rank() != realization.expert_parallel_rank()
        || counts.group_size() != packing.send_counts().len()
        || counts.forward().send() != packing.send_counts()
    {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "expert route packing differs from count consensus".into(),
        ));
    }
    let route_count = packing
        .source_tokens()
        .checked_mul(packing.routes_per_token())
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route cardinality overflowed usize".into(),
            )
        })?;
    let arrays = [
        packing.packed_route_positions(),
        packing.packed_token_indices(),
        packing.packed_global_experts(),
        packing.packed_owner_local_experts(),
    ];
    if arrays.iter().any(|values| values.len() != route_count) {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "expert route packing arrays have inconsistent cardinality".into(),
        ));
    }
    let mut owner_local = vec![0usize; realization.global_expert_count()];
    let mut next_local = vec![0usize; realization.expert_parallel_size()];
    for (global, owner) in realization.owners().iter().copied().enumerate() {
        let local = next_local.get_mut(owner).ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert realization contains an owner outside its selected group".into(),
            )
        })?;
        owner_local[global] = *local;
        *local = local.checked_add(1).ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "owner-local expert identity overflowed usize".into(),
            )
        })?;
    }
    let mut packed_offset = 0usize;
    for (owner, owner_count) in packing.send_counts().iter().copied().enumerate() {
        let end = packed_offset.checked_add(owner_count).ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route owner block overflowed usize".into(),
            )
        })?;
        for packed in packed_offset..end {
            let global = packing.packed_global_experts()[packed];
            if realization.owners().get(global) != Some(&owner)
                || owner_local.get(global) != packing.packed_owner_local_experts().get(packed)
            {
                return Err(RoutedMechanismExecutionError::InvalidPlan(
                    "expert route packing changed expert ownership or local identity".into(),
                ));
            }
        }
        packed_offset = end;
    }
    if packed_offset != route_count {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "expert route owner blocks do not consume every packed row".into(),
        ));
    }
    let mut seen = vec![false; route_count];
    for (packed, position) in packing.packed_route_positions().iter().copied().enumerate() {
        if position >= route_count || std::mem::replace(&mut seen[position], true) {
            return Err(RoutedMechanismExecutionError::InvalidPlan(
                "expert route packing is not an exact route permutation".into(),
            ));
        }
        if packing.packed_token_indices()[packed] != position / packing.routes_per_token() {
            return Err(RoutedMechanismExecutionError::InvalidPlan(
                "expert route packing changed source-token association".into(),
            ));
        }
    }
    let input_shape = movement.shape(input);
    let selection_shape = [packing.source_tokens(), packing.routes_per_token()];
    if input_shape.len() != 2 || input_shape.first() != Some(&packing.source_tokens()) {
        return Err(RoutedMechanismExecutionError::InvalidPlan(format!(
            "expert route input shape {input_shape:?} is not [source_tokens, hidden]"
        )));
    }
    if movement.shape(selected_scores) != selection_shape
        || movement.shape(coefficients) != selection_shape
    {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "expert route score and coefficient shapes differ from route geometry".into(),
        ));
    }
    Ok(())
}

fn validate_packed_tensor_shape<T, M>(
    movement: &M,
    value: &T,
    rows: usize,
    trailing: Option<usize>,
    name: &str,
) -> Result<(), RoutedMechanismExecutionError>
where
    M: ExpertRouteTensorMovement<T>,
{
    let shape = movement.shape(value);
    if shape.len() != 2
        || shape.first() != Some(&rows)
        || trailing.is_some_and(|trailing| shape.get(1) != Some(&trailing))
    {
        return Err(RoutedMechanismExecutionError::Mechanism(format!(
            "expert route {name} has invalid shape {shape:?}"
        )));
    }
    Ok(())
}

/// Executes an architecture-owned route through generic bank, grouped, and collective mechanisms.
#[allow(clippy::too_many_arguments)]
pub fn execute_routed_gated_product<B, P>(
    plan: &ExpertRealizationPlan<eredu_nn::GroupedGatedProductSpec>,
    owner_group: &str,
    owner_unit: usize,
    local_bank_member: usize,
    input: &B::Tensor,
    routes: &eredu_nn::GroupSelection<B::Tensor>,
    bank: &mut P,
    collective: &CollectiveGroupDescriptor,
    group: &B::Group,
    executor: &B::Executor,
    context: &<B::Tensor as eredu_nn::Tensor>::Context,
) -> Result<B::Tensor, RoutedMechanismExecutionError>
where
    B: eredu_nn::GroupedNeuralBackend + CollectiveBackend,
    P: AddressableGatedProductBank<B>,
    P::Error: std::fmt::Display,
    B::CollectiveError: std::fmt::Display,
{
    let expected = plan
        .collective_group(collective.id())
        .map_err(|error| RoutedMechanismExecutionError::InvalidPlan(error.to_string()))?;
    if &expected != collective {
        return Err(RoutedMechanismExecutionError::InvalidPlan(
            "collective membership does not match the architecture route plan".into(),
        ));
    }
    let global_member = plan
        .local_global_group_indices()
        .get(local_bank_member)
        .copied()
        .ok_or_else(|| {
            RoutedMechanismExecutionError::InvalidPlan(format!(
                "local bank member {local_bank_member} is outside the selected bank"
            ))
        })?;
    let spec = plan.unit_spec(owner_group, owner_unit).ok_or_else(|| {
        RoutedMechanismExecutionError::InvalidPlan(format!(
            "execution unit {owner_group:?}/{owner_unit} has no grouped bank"
        ))
    })?;
    let key = ParameterBankKey::new(owner_unit, global_member);
    let groups = bank
        .acquire(key, spec, context)
        .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    let output =
        eredu_nn::GroupedGatedProductOperator::forward_grouped(groups, input, routes, context)
            .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))?;
    if collective.members().len() > 1 {
        B::all_to_all(output, group, executor)
            .map_err(|error| RoutedMechanismExecutionError::Mechanism(error.to_string()))
    } else {
        Ok(output)
    }
}

/// Invalid architecture-derived expert realization.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ExpertRealizationPlanError {
    /// The architecture declared no routed experts.
    #[error("expert realization requires at least one routed expert")]
    EmptyExpertBank,
    /// The architecture declared no routed execution units.
    #[error("expert realization requires at least one routed execution unit")]
    EmptyUnitSchedule,
    /// The requested rank topology cannot own a non-empty expert partition.
    #[error("invalid expert realization topology: {0}")]
    InvalidTopology(String),
}

/// Placement of one expert relative to an expert-parallel axis.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[non_exhaustive]
pub enum ExpertResidencyDistribution {
    /// Assign the expert by its global router identity across expert ranks.
    ExpertParallel,
    /// Materialize the expert on every rank that owns its execution unit.
    Replicated,
}

/// Selects architecture-canonical expert outputs for one rank's global IDs.
///
/// The expert axis is part of the architecture's canonical parameter geometry.
/// Applying the selection to derived outputs lets the checkpoint recipe layer
/// push it through fused, stacked, or transposed physical layouts without a
/// backend recovering storage geometry.
pub(crate) fn select_rank_local_expert_recipes<C: RecipeCatalog + ?Sized>(
    catalog: &C,
    global_experts: usize,
    expert_axis: usize,
    group_indices: &[usize],
    outputs: impl IntoIterator<Item = (String, DerivedWeightRecipe)>,
) -> Result<BTreeMap<String, DerivedWeightRecipe>, String> {
    if group_indices.is_empty() {
        return Err("rank-local expert recipes require at least one expert".into());
    }
    let mut unique = BTreeSet::new();
    for &expert in group_indices {
        if expert >= global_experts {
            return Err(format!(
                "rank-local expert {expert} is outside {global_experts} experts"
            ));
        }
        if !unique.insert(expert) {
            return Err(format!(
                "rank-local expert recipe contains duplicate expert {expert}"
            ));
        }
    }
    let selection = TensorSelection::Indices {
        axis: expert_axis,
        indices: group_indices.to_vec(),
    };
    outputs
        .into_iter()
        .map(|(target, recipe)| {
            recipe
                .select_bounded(catalog, selection.clone())
                .map(|recipe| (target, recipe))
                .map_err(|error| error.to_string())
        })
        .collect()
}

/// One architecture-logical parameter in an independently resident expert.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertParameterRecipe {
    binding_name: String,
    logical_target: String,
    recipe: DerivedWeightRecipe,
    role: ExpertParameterRole,
    metadata: Option<eredu_checkpoint::recipe::RecipeMetadata>,
}

/// Quantization semantics of one independently resident expert parameter.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ExpertParameterRole {
    /// Preserve this binding exactly as declared by the architecture.
    Preserved,
    /// Quantize this projection and publish companions under these exact local names.
    QuantizableProjection {
        /// Local binding name for packed quantization scales.
        scales_binding: String,
        /// Local binding name for packed affine biases, when the format uses them.
        biases_binding: String,
    },
}

impl ExpertParameterRole {
    /// Declares a projection eligible for load-time quantization.
    pub fn quantizable_projection(
        scales_binding: impl Into<String>,
        biases_binding: impl Into<String>,
    ) -> Self {
        Self::QuantizableProjection {
            scales_binding: scales_binding.into(),
            biases_binding: biases_binding.into(),
        }
    }
}

impl ExpertParameterRecipe {
    /// Creates one exact local binding and its architecture-logical destination.
    pub fn new(
        binding_name: impl Into<String>,
        logical_target: impl Into<String>,
        recipe: DerivedWeightRecipe,
        role: ExpertParameterRole,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        let binding_name = binding_name.into();
        if binding_name.trim().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyBindingName);
        }
        let logical_target = logical_target.into();
        if logical_target.trim().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyLogicalTarget {
                binding: binding_name,
            });
        }
        if recipe.source_keys().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyRecipe {
                binding: binding_name,
            });
        }
        if let ExpertParameterRole::QuantizableProjection {
            scales_binding,
            biases_binding,
        } = &role
        {
            for companion in [scales_binding, biases_binding] {
                if companion.trim().is_empty() {
                    return Err(ExpertResidencyCatalogError::EmptyQuantizationCompanion {
                        binding: binding_name,
                    });
                }
                if companion == &binding_name {
                    return Err(
                        ExpertResidencyCatalogError::QuantizationCompanionCollision {
                            binding: binding_name,
                            companion: companion.clone(),
                        },
                    );
                }
            }
            if scales_binding == biases_binding {
                return Err(
                    ExpertResidencyCatalogError::QuantizationCompanionCollision {
                        binding: binding_name,
                        companion: scales_binding.clone(),
                    },
                );
            }
        }
        Ok(Self {
            binding_name,
            logical_target,
            recipe,
            role,
            metadata: None,
        })
    }

    /// Returns the stable name used by an acquired expert bank.
    pub fn binding_name(&self) -> &str {
        &self.binding_name
    }

    /// Returns the exact architecture parameter destination.
    pub fn logical_target(&self) -> &str {
        &self.logical_target
    }

    /// Returns the checkpoint-derived recipe for this expert-local value.
    pub const fn recipe(&self) -> &DerivedWeightRecipe {
        &self.recipe
    }

    /// Returns the architecture-declared parameter and quantization semantics.
    pub const fn role(&self) -> &ExpertParameterRole {
        &self.role
    }

    /// Returns admission-time metadata for this expert-local recipe output.
    pub const fn metadata(&self) -> Option<&eredu_checkpoint::recipe::RecipeMetadata> {
        self.metadata.as_ref()
    }

    /// Consumes the declaration into a named handoff artifact.
    pub fn into_artifact(self) -> ExpertParameterArtifact {
        ExpertParameterArtifact {
            binding_name: Some(self.binding_name),
            logical_target: Some(self.logical_target),
            recipe: Some(self.recipe),
            role: Some(self.role),
        }
    }
}

/// Named consuming artifact for one expert-local parameter declaration.
pub struct ExpertParameterArtifact {
    binding_name: Option<String>,
    logical_target: Option<String>,
    recipe: Option<DerivedWeightRecipe>,
    role: Option<ExpertParameterRole>,
}

impl ExpertParameterArtifact {
    /// Takes the local binding name exactly once.
    pub fn take_binding_name(&mut self) -> String {
        self.binding_name
            .take()
            .expect("binding name already taken")
    }
    /// Takes the architecture-logical destination exactly once.
    pub fn take_logical_target(&mut self) -> String {
        self.logical_target
            .take()
            .expect("logical target already taken")
    }
    /// Takes the checkpoint-derived recipe exactly once.
    pub fn take_recipe(&mut self) -> DerivedWeightRecipe {
        self.recipe.take().expect("expert recipe already taken")
    }
    /// Takes parameter-role semantics exactly once.
    pub fn take_role(&mut self) -> ExpertParameterRole {
        self.role
            .take()
            .expect("expert parameter role already taken")
    }
}

/// One independently addressable expert and its owning execution unit.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertResidencyUnit {
    identity: ParameterBankKey,
    owner_group: ExecutionGroupId,
    owner_unit: usize,
    unit_path: String,
    distribution: ExpertResidencyDistribution,
    parameters: Vec<ExpertParameterRecipe>,
    byte_len: Option<u64>,
}

impl ExpertResidencyUnit {
    /// Creates one complete atomic expert unit.
    pub fn new(
        identity: ParameterBankKey,
        owner_group: ExecutionGroupId,
        owner_unit: usize,
        unit_path: impl Into<String>,
        distribution: ExpertResidencyDistribution,
        parameters: impl IntoIterator<Item = ExpertParameterRecipe>,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        let unit_path = unit_path.into();
        if unit_path.trim().is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyUnitPath { identity });
        }
        let parameters = parameters.into_iter().collect::<Vec<_>>();
        if parameters.is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyUnit { identity });
        }
        let mut names = BTreeSet::new();
        let mut targets = BTreeSet::new();
        for parameter in &parameters {
            if !names.insert(parameter.binding_name.as_str()) {
                return Err(ExpertResidencyCatalogError::DuplicateBinding {
                    identity,
                    binding: parameter.binding_name.clone(),
                });
            }
            if !targets.insert(parameter.logical_target.as_str()) {
                return Err(ExpertResidencyCatalogError::DuplicateLogicalTarget {
                    identity,
                    target: parameter.logical_target.clone(),
                });
            }
            if let ExpertParameterRole::QuantizableProjection {
                scales_binding,
                biases_binding,
            } = parameter.role()
            {
                for companion in [scales_binding, biases_binding] {
                    if parameters
                        .iter()
                        .any(|candidate| candidate.binding_name() == companion)
                    {
                        return Err(
                            ExpertResidencyCatalogError::QuantizationCompanionCollision {
                                binding: parameter.binding_name.clone(),
                                companion: companion.clone(),
                            },
                        );
                    }
                }
            }
        }
        Ok(Self {
            identity,
            owner_group,
            owner_unit,
            unit_path,
            distribution,
            parameters,
            byte_len: None,
        })
    }

    /// Returns the global cache identity selected by architecture routing.
    pub const fn identity(&self) -> ParameterBankKey {
        self.identity
    }

    /// Returns the canonical execution group that owns this expert.
    pub const fn owner_group(&self) -> &ExecutionGroupId {
        &self.owner_group
    }

    /// Returns the architecture-global unit index inside the owning group.
    pub const fn owner_unit(&self) -> usize {
        self.owner_unit
    }

    /// Returns the architecture execution-unit path that owns these parameters.
    pub fn unit_path(&self) -> &str {
        &self.unit_path
    }

    /// Returns how this expert participates in expert-parallel placement.
    pub const fn distribution(&self) -> ExpertResidencyDistribution {
        self.distribution
    }

    /// Returns every exact expert-local parameter recipe.
    pub fn parameters(&self) -> &[ExpertParameterRecipe] {
        &self.parameters
    }

    /// Attaches exact admitted materialized bytes when already derived upstream.
    pub fn with_byte_len(mut self, byte_len: u64) -> Result<Self, ExpertResidencyCatalogError> {
        if byte_len == 0 {
            return Err(ExpertResidencyCatalogError::InvalidByteGeometry {
                identity: self.identity,
                detail: "materialized byte count is zero".into(),
            });
        }
        self.byte_len = Some(byte_len);
        Ok(self)
    }

    /// Returns exact admitted materialized bytes for this atomic unit.
    pub const fn byte_len(&self) -> Option<u64> {
        self.byte_len
    }

    /// Consumes the unit into its parameter recipes.
    pub fn into_parameters(self) -> Vec<ExpertParameterRecipe> {
        self.parameters
    }
}

/// Complete architecture-owned schedule for independent expert residency.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertResidencyCatalog {
    units: Vec<ExpertResidencyUnit>,
}

impl ExpertResidencyCatalog {
    /// Validates a non-empty catalog with globally unique cache identities.
    pub fn new(
        units: impl IntoIterator<Item = ExpertResidencyUnit>,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        let units = units.into_iter().collect::<Vec<_>>();
        if units.is_empty() {
            return Err(ExpertResidencyCatalogError::EmptyCatalog);
        }
        let mut identities = BTreeSet::new();
        for unit in &units {
            if !identities.insert(unit.identity) {
                return Err(ExpertResidencyCatalogError::DuplicateIdentity(
                    unit.identity,
                ));
            }
        }
        Ok(Self { units })
    }

    /// Returns the deterministic architecture order of resident expert units.
    pub fn units(&self) -> &[ExpertResidencyUnit] {
        &self.units
    }

    /// Returns one atomic unit by its architecture-translated bank identity.
    pub fn unit(&self, identity: ParameterBankKey) -> Option<&ExpertResidencyUnit> {
        self.units.iter().find(|unit| unit.identity == identity)
    }

    /// Returns every canonical parameter assigned to addressable storage.
    pub fn logical_targets(&self) -> BTreeSet<&str> {
        self.units
            .iter()
            .flat_map(|unit| unit.parameters.iter())
            .map(ExpertParameterRecipe::logical_target)
            .collect()
    }

    /// Infers and retains exact materialized bytes from admitted recipe metadata.
    pub fn with_inferred_byte_geometry<C: RecipeCatalog + ?Sized>(
        mut self,
        catalog: &C,
    ) -> Result<Self, ExpertResidencyCatalogError> {
        for unit in &mut self.units {
            let bytes = unit
                .parameters
                .iter_mut()
                .try_fold(0u64, |total, parameter| {
                    let metadata = parameter.recipe().infer(catalog).map_err(|error| {
                        ExpertResidencyCatalogError::InvalidByteGeometry {
                            identity: unit.identity,
                            detail: error.to_string(),
                        }
                    })?;
                    let bytes = metadata.byte_len;
                    parameter.metadata = Some(metadata);
                    total.checked_add(bytes).ok_or_else(|| {
                        ExpertResidencyCatalogError::InvalidByteGeometry {
                            identity: unit.identity,
                            detail: "materialized byte count overflowed".into(),
                        }
                    })
                })?;
            if bytes == 0 {
                return Err(ExpertResidencyCatalogError::InvalidByteGeometry {
                    identity: unit.identity,
                    detail: "materialized byte count is zero".into(),
                });
            }
            unit.byte_len = Some(bytes);
        }
        Ok(self)
    }

    /// Consumes the catalog into its deterministic architecture order.
    pub fn into_units(self) -> Vec<ExpertResidencyUnit> {
        self.units
    }

    /// Consumes the catalog and retains units owned by a caller's execution partition.
    ///
    /// Selection is expressed only in the architecture's canonical group-local address
    /// space so adapters do not flatten heterogeneous execution groups back into layer
    /// ordinals.
    pub fn into_units_selected_by_owner(
        self,
        mut owns_unit: impl FnMut(&ExecutionGroupId, usize) -> bool,
    ) -> impl Iterator<Item = ExpertResidencyUnit> {
        self.units
            .into_iter()
            .filter(move |unit| owns_unit(unit.owner_group(), unit.owner_unit()))
    }
}

impl IntoIterator for ExpertResidencyCatalog {
    type Item = ExpertResidencyUnit;
    type IntoIter = std::vec::IntoIter<ExpertResidencyUnit>;

    fn into_iter(self) -> Self::IntoIter {
        self.units.into_iter()
    }
}

/// Invalid architecture-owned expert residency topology.
#[derive(Debug, Clone, Eq, PartialEq, thiserror::Error)]
pub enum ExpertResidencyCatalogError {
    /// A local acquired-bank name is empty.
    #[error("expert residency binding name must not be empty")]
    EmptyBindingName,
    /// A binding has no exact architecture destination.
    #[error("expert residency binding {binding:?} has an empty logical target")]
    EmptyLogicalTarget {
        /// Invalid local binding name.
        binding: String,
    },
    /// A binding recipe has no checkpoint inputs.
    #[error("expert residency binding {binding:?} has no checkpoint recipe source")]
    EmptyRecipe {
        /// Invalid local binding name.
        binding: String,
    },
    /// A quantizable binding did not declare a usable companion name.
    #[error("expert residency binding {binding:?} has an empty quantization companion")]
    EmptyQuantizationCompanion {
        /// Invalid projection binding.
        binding: String,
    },
    /// A packed companion collides with its projection or another declared binding.
    #[error("expert residency binding {binding:?} quantization companion {companion:?} collides with an existing binding")]
    QuantizationCompanionCollision {
        /// Quantizable projection binding.
        binding: String,
        /// Colliding packed companion binding.
        companion: String,
    },
    /// An expert is not attached to an architecture execution unit.
    #[error("expert {identity:?} has an empty architecture unit path")]
    EmptyUnitPath {
        /// Invalid expert identity.
        identity: ParameterBankKey,
    },
    /// An expert has no checkpoint-backed parameters.
    #[error("expert {identity:?} has no residency parameters")]
    EmptyUnit {
        /// Invalid expert identity.
        identity: ParameterBankKey,
    },
    /// Admitted recipes did not yield a finite nonzero atomic byte count.
    #[error("expert {identity:?} has invalid materialized byte geometry: {detail}")]
    InvalidByteGeometry {
        /// Invalid unit.
        identity: ParameterBankKey,
        /// Metadata inference failure.
        detail: String,
    },
    /// One acquired bank name is repeated inside an expert.
    #[error("expert {identity:?} repeats local binding {binding:?}")]
    DuplicateBinding {
        /// Invalid expert identity.
        identity: ParameterBankKey,
        /// Repeated local binding.
        binding: String,
    },
    /// One architecture target is repeated inside an expert.
    #[error("expert {identity:?} repeats logical target {target:?}")]
    DuplicateLogicalTarget {
        /// Invalid expert identity.
        identity: ParameterBankKey,
        /// Repeated logical destination.
        target: String,
    },
    /// No independently resident experts were declared.
    #[error("architecture declares no independently resident experts")]
    EmptyCatalog,
    /// Two units use the same router/cache identity.
    #[error("architecture repeats expert residency identity {0:?}")]
    DuplicateIdentity(ParameterBankKey),
}

#[cfg(test)]
mod tests {
    use super::*;
    use eredu_core::ParallelTopology;

    #[derive(Debug, Clone, PartialEq)]
    struct NumericTensor {
        rows: usize,
        columns: usize,
        values: Vec<f32>,
    }

    impl NumericTensor {
        fn new(rows: usize, columns: usize, values: Vec<f32>) -> Self {
            assert_eq!(values.len(), rows * columns);
            Self {
                rows,
                columns,
                values,
            }
        }
    }

    #[derive(Default)]
    struct NumericMovement {
        gather_rows: usize,
        gather_routes: usize,
        scatter_rows: usize,
        scattered_destinations: Vec<usize>,
    }

    impl ExpertRouteTensorMovement<NumericTensor> for NumericMovement {
        type Error = &'static str;

        fn shape(&self, value: &NumericTensor) -> Vec<usize> {
            vec![value.rows, value.columns]
        }

        fn gather_rows(
            &mut self,
            value: &NumericTensor,
            rows: &[usize],
        ) -> Result<NumericTensor, Self::Error> {
            self.gather_rows += 1;
            let mut values = Vec::with_capacity(rows.len() * value.columns);
            for row in rows {
                if *row >= value.rows {
                    return Err("row outside tensor");
                }
                values.extend_from_slice(
                    &value.values[*row * value.columns..(*row + 1) * value.columns],
                );
            }
            Ok(NumericTensor::new(rows.len(), value.columns, values))
        }

        fn gather_route_values(
            &mut self,
            value: &NumericTensor,
            flattened_routes: &[usize],
        ) -> Result<NumericTensor, Self::Error> {
            self.gather_routes += 1;
            let values = flattened_routes
                .iter()
                .map(|position| {
                    value
                        .values
                        .get(*position)
                        .copied()
                        .ok_or("route outside tensor")
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(NumericTensor::new(values.len(), 1, values))
        }

        fn scatter_add_rows(
            &mut self,
            value: NumericTensor,
            destination_rows: &[usize],
            output_rows: usize,
        ) -> Result<NumericTensor, Self::Error> {
            self.scatter_rows += 1;
            self.scattered_destinations = destination_rows.to_vec();
            if value.rows != destination_rows.len() {
                return Err("scatter row cardinality mismatch");
            }
            let mut output = vec![0.0; output_rows * value.columns];
            for (source, destination) in destination_rows.iter().copied().enumerate() {
                if destination >= output_rows {
                    return Err("scatter destination outside tensor");
                }
                for column in 0..value.columns {
                    output[destination * value.columns + column] +=
                        value.values[source * value.columns + column];
                }
            }
            Ok(NumericTensor::new(output_rows, value.columns, output))
        }
    }

    #[derive(Default)]
    struct IdentityExchange {
        calls: usize,
        fail_on: Option<usize>,
        reorder_indices: bool,
        corrupt_index_call: Option<usize>,
    }

    impl IdentityExchange {
        fn begin(
            &mut self,
            counts: &CommunicationPeerCounts,
            rows: usize,
        ) -> Result<(), &'static str> {
            self.calls += 1;
            if self.fail_on == Some(self.calls) {
                return Err("exchange failed");
            }
            let send = counts.send().iter().sum::<usize>();
            let receive = counts.receive().iter().sum::<usize>();
            if rows != send || send != receive {
                return Err("identity exchange requires balanced row totals");
            }
            Ok(())
        }
    }

    impl ExpertRouteExchange<NumericTensor> for IdentityExchange {
        type Error = &'static str;

        fn exchange_tensor(
            &mut self,
            counts: &CommunicationPeerCounts,
            value: NumericTensor,
        ) -> Result<NumericTensor, Self::Error> {
            self.begin(counts, value.rows)?;
            Ok(value)
        }

        fn exchange_indices(
            &mut self,
            counts: &CommunicationPeerCounts,
            mut values: Vec<usize>,
        ) -> Result<Vec<usize>, Self::Error> {
            self.begin(counts, values.len())?;
            if self.reorder_indices {
                values.reverse();
            }
            if self.corrupt_index_call == Some(self.calls) && !values.is_empty() {
                values[0] = 0;
            }
            Ok(values)
        }
    }

    #[derive(Default)]
    struct NumericProvider {
        calls: usize,
        fail: bool,
        observed: Option<(usize, ExpertPass, eredu_runtime::ParameterBankAccess)>,
        consumed_global_experts: Vec<usize>,
        consumed_experts: Vec<usize>,
        addressable_keys: Vec<ParameterBankKey>,
    }

    impl AddressableExpertRouteProvider<NumericTensor> for NumericProvider {
        type Error = &'static str;

        fn execute_addressable_routes(
            &mut self,
            request: AddressableExpertRouteRequest<'_, NumericTensor>,
        ) -> Result<NumericTensor, Self::Error> {
            self.calls += 1;
            if self.fail {
                return Err("provider failed");
            }
            if request.input.rows != request.global_experts.len()
                || request.input.rows != request.owner_local_experts.len()
                || request.coefficients.rows != request.input.rows
                || request.selected_scores.rows != request.input.rows
                || request.combination != ExpertRouteCombination::CoefficientWeightedSum
            {
                return Err("provider request mismatch");
            }
            self.observed = Some((request.unit, request.pass, request.access));
            self.consumed_global_experts = request.global_experts.to_vec();
            self.consumed_experts = request.owner_local_experts.to_vec();
            self.addressable_keys = (0..request.input.rows)
                .map(|row| {
                    request
                        .addressable_bank_key(row)
                        .ok_or("missing global addressable key")
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut output = request.input.clone();
            for row in 0..output.rows {
                let coefficient = request.coefficients.values[row];
                let expert_scale = (request.owner_local_experts[row] + 1) as f32;
                for column in 0..output.columns {
                    output.values[row * output.columns + column] *= coefficient * expert_scale;
                }
            }
            Ok(output)
        }
    }

    struct TensorParallelNumericProvider {
        tensor_rank: usize,
        calls: usize,
    }

    impl AddressableExpertRouteProvider<NumericTensor> for TensorParallelNumericProvider {
        type Error = &'static str;

        fn execute_addressable_routes(
            &mut self,
            _request: AddressableExpertRouteRequest<'_, NumericTensor>,
        ) -> Result<NumericTensor, Self::Error> {
            Err("TP provider requires the structured output path")
        }

        fn execute_addressable_routes_tensor_parallel(
            &mut self,
            request: AddressableExpertRouteRequest<'_, NumericTensor>,
        ) -> Result<RoutedExpertTensorParallelOutput<NumericTensor>, Self::Error> {
            self.calls += 1;
            if request.input.rows != request.global_experts.len()
                || request.input.rows != request.owner_local_experts.len()
                || request.coefficients.rows != request.input.rows
            {
                return Err("TP provider request mismatch");
            }
            let contribution_scale = (self.tensor_rank + 1) as f32;
            let mut reducible = request.input.clone();
            let mut post_reduce = NumericTensor::new(
                request.input.rows,
                request.input.columns,
                vec![0.0; request.input.rows * request.input.columns],
            );
            for row in 0..request.input.rows {
                let coefficient = request.coefficients.values[row];
                let expert_scale = (request.owner_local_experts[row] + 1) as f32;
                let bias = coefficient * (request.global_experts[row] + 1) as f32;
                for column in 0..request.input.columns {
                    reducible.values[row * request.input.columns + column] *=
                        coefficient * expert_scale * contribution_scale;
                    post_reduce.values[row * request.input.columns + column] = bias;
                }
            }
            Ok(RoutedExpertTensorParallelOutput::Partial(
                eredu_nn::TensorParallelGroupedOutput::new(reducible, Some(post_reduce)),
            ))
        }
    }

    fn numeric_exchange_fixture() -> (
        ExpertRealizationPlan<()>,
        ExpertRoutePackingPlan,
        ExpertRouteCountPlan,
        NumericTensor,
        NumericTensor,
        NumericTensor,
    ) {
        let topology = ParallelTopology::new(1, 1, 2, 1).unwrap();
        let realization = ExpertRealizationPlan::balanced(
            4,
            ParallelRankTopology::new(topology, 1).unwrap(),
            BTreeMap::from([((ExecutionGroupId::new("decoder").unwrap(), 7), ())]),
        )
        .unwrap();
        let packing = ExpertRoutePackingPlan::new(&realization, 2, 2, &[2, 3, 2, 3]).unwrap();
        let counts = ExpertRouteCountPlan::from_consensus(
            CollectiveGroupId::new(71),
            1,
            vec![0, 4],
            vec![0, 0, 0, 4],
        )
        .unwrap();
        (
            realization,
            packing,
            counts,
            NumericTensor::new(2, 2, vec![10.0, 1.0, 20.0, 2.0]),
            NumericTensor::new(2, 2, vec![0.1, 0.2, 0.3, 0.4]),
            NumericTensor::new(2, 2, vec![0.5, 0.25, 1.0, 2.0]),
        )
    }

    fn run_numeric_exchange(
        realization: &ExpertRealizationPlan<()>,
        packing: &ExpertRoutePackingPlan,
        counts: &ExpertRouteCountPlan,
        input: &NumericTensor,
        scores: &NumericTensor,
        coefficients: &NumericTensor,
        movement: &mut NumericMovement,
        forward: &mut IdentityExchange,
        reverse: &mut IdentityExchange,
        provider: &mut NumericProvider,
    ) -> Result<NumericTensor, RoutedMechanismExecutionError> {
        execute_expert_route_exchange(
            realization,
            packing,
            counts,
            input,
            scores,
            coefficients,
            7,
            ExpertPass::Prefill,
            movement,
            forward,
            reverse,
            provider,
        )
    }

    #[test]
    fn expert_realization_is_the_complete_balanced_owner_map() {
        let topology = ParallelTopology::new(1, 1, 3, 1).unwrap();
        let plans = (0..3)
            .map(|rank| {
                let decoder = ExecutionGroupId::new("text_decoder").unwrap();
                ExpertRealizationPlan::balanced(
                    8,
                    ParallelRankTopology::new(topology, rank).unwrap(),
                    BTreeMap::from([((decoder, 4), format!("rank-{rank}-bank"))]),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        assert_eq!(plans[0].owners(), [0, 0, 0, 1, 1, 1, 2, 2]);
        assert_eq!(plans[0].local_global_group_indices(), [0, 1, 2]);
        assert_eq!(plans[1].local_global_group_indices(), [3, 4, 5]);
        assert_eq!(plans[2].local_global_group_indices(), [6, 7]);
        assert_eq!(
            plans[2].unit_spec("text_decoder", 4).map(String::as_str),
            Some("rank-2-bank")
        );
        assert!(plans[2].has_routed_units_in_group("text_decoder"));
        assert!(!plans[2].has_routed_units_in_group("mtp.0"));
    }

    #[test]
    fn expert_realization_rejects_empty_ranks_and_unit_schedules() {
        let rank =
            ParallelRankTopology::new(ParallelTopology::new(1, 1, 3, 1).unwrap(), 0).unwrap();
        assert!(matches!(
            ExpertRealizationPlan::<()>::balanced(
                2,
                rank,
                BTreeMap::from([((ExecutionGroupId::new("decoder").unwrap(), 0), ())])
            ),
            Err(ExpertRealizationPlanError::InvalidTopology(_))
        ));
        assert!(matches!(
            ExpertRealizationPlan::<()>::balanced(3, rank, BTreeMap::new()),
            Err(ExpertRealizationPlanError::EmptyUnitSchedule)
        ));
    }

    #[test]
    fn expert_route_counts_preserve_uneven_and_zero_peer_rows() {
        let plan = ExpertRouteCountPlan::from_consensus(
            CollectiveGroupId::new(17),
            1,
            vec![0, 2, 1],
            vec![2, 0, 1, 0, 2, 1, 4, 0, 0],
        )
        .unwrap();

        assert_eq!(plan.forward().send(), [0, 2, 1]);
        assert_eq!(plan.forward().receive(), [0, 2, 0]);
        assert_eq!(plan.reverse().send(), [0, 2, 0]);
        assert_eq!(plan.reverse().receive(), [0, 2, 1]);
        assert_eq!(plan.count_matrix(), [2, 0, 1, 0, 2, 1, 4, 0, 0]);
    }

    #[test]
    fn expert_route_count_consensus_rejects_changed_local_row() {
        let error = ExpertRouteCountPlan::from_consensus(
            CollectiveGroupId::new(18),
            1,
            vec![0, 2],
            vec![1, 0, 1, 1],
        )
        .unwrap_err();

        assert_eq!(
            error,
            RoutedMechanismExecutionError::InvalidPlan(
                "expert route count consensus changed the local send row".into()
            )
        );
    }

    #[test]
    fn expert_route_packing_is_stable_and_retains_idle_destinations() {
        let topology = ParallelTopology::new(1, 1, 3, 1).unwrap();
        let realization = ExpertRealizationPlan::balanced(
            6,
            ParallelRankTopology::new(topology, 0).unwrap(),
            BTreeMap::from([((ExecutionGroupId::new("decoder").unwrap(), 0), ())]),
        )
        .unwrap();
        let packing = ExpertRoutePackingPlan::new(&realization, 2, 2, &[5, 4, 1, 0]).unwrap();

        assert_eq!(packing.send_counts(), [2, 0, 2]);
        assert_eq!(packing.packed_route_positions(), [2, 3, 0, 1]);
        assert_eq!(packing.packed_token_indices(), [1, 1, 0, 0]);
        assert_eq!(packing.packed_global_experts(), [1, 0, 5, 4]);
        assert_eq!(packing.packed_owner_local_experts(), [1, 0, 1, 0]);
    }

    #[test]
    fn expert_exchange_combines_duplicate_token_routes_in_exact_source_order() {
        let (realization, packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange::default();
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider::default();

        let output = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap();

        assert_eq!(counts.forward().send(), [0, 4]);
        assert_eq!(
            output,
            NumericTensor::new(2, 2, vec![10.0, 1.0, 100.0, 10.0])
        );
        assert_eq!(provider.calls, 1);
        assert_eq!(provider.consumed_global_experts, [2, 3, 2, 3]);
        assert_eq!(provider.consumed_experts, [0, 1, 0, 1]);
        assert_eq!(
            provider.addressable_keys,
            [
                ParameterBankKey::new(7, 2),
                ParameterBankKey::new(7, 3),
                ParameterBankKey::new(7, 2),
                ParameterBankKey::new(7, 3),
            ]
        );
        assert_eq!(
            provider.observed,
            Some((
                7,
                ExpertPass::Prefill,
                eredu_runtime::ParameterBankAccess::Bulk
            ))
        );
        assert_eq!((forward.calls, reverse.calls), (6, 2));
        assert_eq!(
            (
                movement.gather_rows,
                movement.gather_routes,
                movement.scatter_rows
            ),
            (1, 2, 1)
        );
        assert_eq!(movement.scattered_destinations, [0, 0, 1, 1]);
    }

    #[test]
    fn tp2_ep2_reverse_exchange_preserves_bias_for_one_post_sum_addition() {
        let (realization, packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        let mut rank_outputs = Vec::new();

        for tensor_rank in 0..2 {
            let mut movement = NumericMovement::default();
            let mut forward = IdentityExchange::default();
            let mut reverse = IdentityExchange::default();
            let mut provider = TensorParallelNumericProvider {
                tensor_rank,
                calls: 0,
            };
            let output = execute_expert_route_exchange_tensor_parallel(
                &realization,
                &packing,
                &counts,
                &input,
                &scores,
                &coefficients,
                7,
                ExpertPass::Decode,
                &mut movement,
                &mut forward,
                &mut reverse,
                &mut provider,
            )
            .unwrap();
            let RoutedExpertTensorParallelOutput::Partial(output) = output else {
                panic!("TP expert exchange collapsed a rank-local partial");
            };
            let (reducible, post_reduce) = output.into_parts();
            rank_outputs.push((reducible, post_reduce.unwrap()));
            assert_eq!(provider.calls, 1);
            assert_eq!((forward.calls, reverse.calls), (6, 3));
            assert_eq!(movement.scatter_rows, 2);
        }

        assert_eq!(rank_outputs[0].1, rank_outputs[1].1);
        let mut completed = rank_outputs[0].0.clone();
        for (index, value) in completed.values.iter_mut().enumerate() {
            *value += rank_outputs[1].0.values[index] + rank_outputs[0].1.values[index];
        }
        assert_eq!(
            completed,
            NumericTensor::new(2, 2, vec![32.5, 5.5, 311.0, 41.0])
        );

        let mut incorrectly_added_per_rank = rank_outputs[0].0.clone();
        for (index, value) in incorrectly_added_per_rank.values.iter_mut().enumerate() {
            *value += rank_outputs[1].0.values[index]
                + rank_outputs[0].1.values[index]
                + rank_outputs[1].1.values[index];
        }
        assert_ne!(completed, incorrectly_added_per_rank);
    }

    #[test]
    fn malformed_expert_assignment_fails_before_payload_provider_or_exchange() {
        let (realization, mut packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        packing.packed_owner_local_experts[0] = 99;
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange::default();
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider::default();

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RoutedMechanismExecutionError::InvalidPlan(_)
        ));
        assert_eq!((movement.gather_rows, movement.gather_routes), (0, 0));
        assert_eq!((forward.calls, reverse.calls, provider.calls), (0, 0, 0));
    }

    #[test]
    fn changed_global_identity_fails_before_payload_provider_or_exchange() {
        let (realization, mut packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        packing.packed_global_experts[0] = 0;
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange::default();
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider::default();

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(error.to_string().contains("ownership or local identity"));
        assert_eq!((movement.gather_rows, movement.gather_routes), (0, 0));
        assert_eq!((forward.calls, reverse.calls, provider.calls), (0, 0, 0));
    }

    #[test]
    fn exchanged_global_local_mismatch_stops_before_provider_and_return_exchange() {
        let (realization, packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange {
            corrupt_index_call: Some(1),
            ..IdentityExchange::default()
        };
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider::default();

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(error.to_string().contains("maps global expert"));
        assert_eq!((forward.calls, reverse.calls, provider.calls), (6, 0, 0));
        assert_eq!(movement.scatter_rows, 0);
    }

    #[test]
    fn malformed_route_order_fails_before_payload_provider_or_exchange() {
        let (realization, mut packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        packing.packed_route_positions[1] = packing.packed_route_positions[0];
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange::default();
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider::default();

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RoutedMechanismExecutionError::InvalidPlan(_)
        ));
        assert_eq!((movement.gather_rows, movement.gather_routes), (0, 0));
        assert_eq!((forward.calls, reverse.calls, provider.calls), (0, 0, 0));
    }

    #[test]
    fn malformed_peer_counts_fail_before_payload_provider_or_exchange() {
        let (realization, mut packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        packing.send_counts = vec![0, 1, 1, 2];
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange::default();
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider::default();

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            RoutedMechanismExecutionError::InvalidPlan(_)
        ));
        assert_eq!((movement.gather_rows, movement.gather_routes), (0, 0));
        assert_eq!((forward.calls, reverse.calls, provider.calls), (0, 0, 0));
    }

    #[test]
    fn reordered_expert_return_is_rejected_before_scatter() {
        let (realization, packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange::default();
        let mut reverse = IdentityExchange {
            reorder_indices: true,
            ..IdentityExchange::default()
        };
        let mut provider = NumericProvider::default();

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(error.to_string().contains("return order"));
        assert_eq!(provider.calls, 1);
        assert_eq!(movement.scatter_rows, 0);
        assert_eq!((forward.calls, reverse.calls), (6, 2));
    }

    #[test]
    fn provider_failure_stops_before_reverse_exchange() {
        let (realization, packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange::default();
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider {
            fail: true,
            ..NumericProvider::default()
        };

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(error.to_string().contains("provider failed"));
        assert_eq!((forward.calls, provider.calls, reverse.calls), (6, 1, 0));
        assert_eq!(movement.scatter_rows, 0);
    }

    #[test]
    fn forward_exchange_failure_stops_before_provider_and_reverse() {
        let (realization, packing, counts, input, scores, coefficients) =
            numeric_exchange_fixture();
        let mut movement = NumericMovement::default();
        let mut forward = IdentityExchange {
            fail_on: Some(3),
            ..IdentityExchange::default()
        };
        let mut reverse = IdentityExchange::default();
        let mut provider = NumericProvider::default();

        let error = run_numeric_exchange(
            &realization,
            &packing,
            &counts,
            &input,
            &scores,
            &coefficients,
            &mut movement,
            &mut forward,
            &mut reverse,
            &mut provider,
        )
        .unwrap_err();

        assert!(error.to_string().contains("exchange failed"));
        assert_eq!((forward.calls, provider.calls, reverse.calls), (3, 0, 0));
        assert_eq!(movement.scatter_rows, 0);
    }
}
