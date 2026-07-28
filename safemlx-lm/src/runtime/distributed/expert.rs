//! Reusable expert-parallel assignment, routing, and exchange infrastructure.
//!
//! Pure expert parallelism keeps ordinary model state replicated and partitions
//! only routed expert banks.  [`dispatch_replicated`] exploits the replicated
//! token layout: ranks compact only routes owned by their experts and all-sum
//! the resulting token buffer.  [`all_to_all_v`] is the general sharded-token
//! transport.  It is intentionally an all-gather fallback and therefore uses
//! `O(group_size)` temporary replication until MLX exposes native all-to-all.

use std::{
    cell::Cell,
    time::{Duration, Instant},
};

use safemlx::{
    distributed::{self, Group},
    ops::{concatenate_axis, indexing::TryIndexOp, r#where, segment_sum_by_index, zeros_dtype},
    transforms::eval,
    Array, Dtype, Stream,
};

use crate::{
    error::Error,
    nn::moe::{PackedRelu2Experts, PackedSwiGluExperts},
};

thread_local! {
    static EAGER_TIMING_PROFILING: Cell<bool> = const { Cell::new(false) };
}

/// Scoped opt-in profiling mode for expert-parallel phase timings.
///
/// MLX executes lazily, so ordinary phase timings primarily describe graph
/// submission. While this guard is alive, expert-parallel code materializes
/// phase outputs before stopping each timer. This makes the measurements useful
/// for benchmarks, at the cost of extra synchronization and changed scheduling.
#[must_use]
pub struct ExpertParallelTimingGuard {
    previous: bool,
}

impl Drop for ExpertParallelTimingGuard {
    fn drop(&mut self) {
        EAGER_TIMING_PROFILING.with(|enabled| enabled.set(self.previous));
    }
}

/// Enables device-complete expert-parallel phase timings for the current thread.
pub fn profile_expert_parallel_timings() -> ExpertParallelTimingGuard {
    let previous = EAGER_TIMING_PROFILING.with(|enabled| {
        let previous = enabled.get();
        enabled.set(true);
        previous
    });
    ExpertParallelTimingGuard { previous }
}

pub(crate) fn timing_profiling_enabled() -> bool {
    EAGER_TIMING_PROFILING.with(Cell::get)
}

pub(crate) fn materialize_timing_phase<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    if timing_profiling_enabled() {
        eval(outputs)?;
    }
    Ok(())
}

/// Policy used to assign global routed experts to ranks.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ExpertAssignmentPolicy {
    /// Balanced contiguous ranges, with lower ranks receiving any remainder.
    BalancedContiguous,
    /// Expert `e` is owned by rank `e % group_size`.
    RoundRobin,
    /// Explicit global-expert-to-owner-rank table.
    Explicit(Vec<usize>),
}

/// Validated bidirectional mapping between checkpoint-global and owner-local ids.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertAssignment {
    global_expert_count: usize,
    group_size: usize,
    rank: usize,
    policy: ExpertAssignmentPolicy,
    owners: Vec<usize>,
    owner_local: Vec<usize>,
    local_global: Vec<usize>,
}

impl ExpertAssignment {
    /// Creates the default balanced contiguous assignment.
    pub fn balanced(global_experts: usize, group_size: usize, rank: usize) -> Result<Self, Error> {
        Self::balanced_with_empty(global_experts, group_size, rank, false)
    }

    /// Creates a balanced assignment and optionally permits empty ranks.
    pub fn balanced_with_empty(
        global_experts: usize,
        group_size: usize,
        rank: usize,
        allow_empty: bool,
    ) -> Result<Self, Error> {
        validate_dimensions(global_experts, group_size, rank, allow_empty)?;
        let base = global_experts / group_size;
        let extra = global_experts % group_size;
        let mut owners = Vec::with_capacity(global_experts);
        for owner in 0..group_size {
            owners.extend(std::iter::repeat_n(
                owner,
                base + usize::from(owner < extra),
            ));
        }
        Self::from_owners_impl(
            owners,
            group_size,
            rank,
            ExpertAssignmentPolicy::BalancedContiguous,
            allow_empty,
        )
    }

    /// Creates a deterministic round-robin assignment.
    pub fn round_robin(
        global_experts: usize,
        group_size: usize,
        rank: usize,
    ) -> Result<Self, Error> {
        validate_dimensions(global_experts, group_size, rank, false)?;
        let owners = (0..global_experts)
            .map(|expert| expert % group_size)
            .collect();
        Self::from_owners_impl(
            owners,
            group_size,
            rank,
            ExpertAssignmentPolicy::RoundRobin,
            false,
        )
    }

    /// Creates an assignment from one owner rank per global expert.
    pub fn explicit(owners: Vec<usize>, group_size: usize, rank: usize) -> Result<Self, Error> {
        let policy = ExpertAssignmentPolicy::Explicit(owners.clone());
        Self::from_owners_impl(owners, group_size, rank, policy, false)
    }

    /// Creates an explicit assignment and optionally permits empty ranks.
    pub fn explicit_with_empty(
        owners: Vec<usize>,
        group_size: usize,
        rank: usize,
        allow_empty: bool,
    ) -> Result<Self, Error> {
        let policy = ExpertAssignmentPolicy::Explicit(owners.clone());
        Self::from_owners_impl(owners, group_size, rank, policy, allow_empty)
    }

    fn from_owners_impl(
        owners: Vec<usize>,
        group_size: usize,
        rank: usize,
        policy: ExpertAssignmentPolicy,
        allow_empty: bool,
    ) -> Result<Self, Error> {
        validate_dimensions(owners.len(), group_size, rank, allow_empty)?;
        if let Some((expert, owner)) = owners
            .iter()
            .copied()
            .enumerate()
            .find(|(_, owner)| *owner >= group_size)
        {
            return Err(Error::Parallel(format!(
                "global expert {expert} has invalid owner rank {owner} for EP size {group_size}"
            )));
        }
        let mut next_local = vec![0usize; group_size];
        let mut owner_local = Vec::with_capacity(owners.len());
        let mut local_global = Vec::new();
        for (global, owner) in owners.iter().copied().enumerate() {
            owner_local.push(next_local[owner]);
            next_local[owner] = next_local[owner].checked_add(1).ok_or_else(|| {
                Error::Parallel("owner-local expert index overflowed usize".into())
            })?;
            if owner == rank {
                local_global.push(global);
            }
        }
        if !allow_empty && next_local.contains(&0) {
            return Err(Error::Parallel(format!(
                "expert assignment creates an empty rank: counts {next_local:?}"
            )));
        }
        if owners.len() > i32::MAX as usize
            || next_local.iter().any(|count| *count > i32::MAX as usize)
        {
            return Err(Error::Parallel(
                "expert assignment exceeds MLX i32 indexing limits".into(),
            ));
        }
        Ok(Self {
            global_expert_count: owners.len(),
            group_size,
            rank,
            policy,
            owners,
            owner_local,
            local_global,
        })
    }

    /// Total checkpoint-global routed expert count.
    pub const fn global_expert_count(&self) -> usize {
        self.global_expert_count
    }
    /// EP group size.
    pub const fn group_size(&self) -> usize {
        self.group_size
    }
    /// Current rank within the EP group.
    pub const fn rank(&self) -> usize {
        self.rank
    }
    /// Assignment policy.
    pub fn policy(&self) -> &ExpertAssignmentPolicy {
        &self.policy
    }
    /// Global expert ids owned by this rank, in owner-local order.
    pub fn local_global_expert_ids(&self) -> &[usize] {
        &self.local_global
    }
    /// Number of experts owned by this rank.
    pub fn local_expert_count(&self) -> usize {
        self.local_global.len()
    }
    /// Returns the owner rank of a global expert.
    pub fn owner(&self, global: usize) -> Option<usize> {
        self.owners.get(global).copied()
    }
    /// Returns the owner-local id of a global expert.
    pub fn owner_local_id(&self, global: usize) -> Option<usize> {
        self.owner_local.get(global).copied()
    }
    /// Returns the global id corresponding to a local id on this rank.
    pub fn global_id(&self, local: usize) -> Option<usize> {
        self.local_global.get(local).copied()
    }
    /// Complete global-to-owner mapping.
    pub fn owners(&self) -> &[usize] {
        &self.owners
    }
    /// Complete global-to-owner-local mapping.
    pub fn owner_local_ids(&self) -> &[usize] {
        &self.owner_local
    }
}

fn validate_dimensions(
    global_experts: usize,
    group_size: usize,
    rank: usize,
    allow_empty: bool,
) -> Result<(), Error> {
    if global_experts == 0 || group_size == 0 {
        return Err(Error::Parallel(
            "expert count and EP size must be nonzero".into(),
        ));
    }
    if rank >= group_size {
        return Err(Error::Parallel(format!(
            "EP rank {rank} is outside size {group_size}"
        )));
    }
    if !allow_empty && global_experts < group_size {
        return Err(Error::Parallel(format!(
            "cannot assign {global_experts} experts to {group_size} non-empty ranks"
        )));
    }
    Ok(())
}

/// Token ownership layout used by expert dispatch.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TokenLayout {
    /// Every EP rank has identical hidden rows and router results.
    Replicated,
    /// Each source rank owns disjoint hidden rows and exchanges routes.
    Sharded,
}

/// Transport selected for expert routes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertExchangeStrategy {
    /// Compact local routes, execute local experts, and all-sum token outputs.
    ReplicatedInputAllSum,
    /// Variable-count all-to-all emulated with padded all-gather.
    AllGatherAllToAllV,
}

/// Per-dispatch counters used by diagnostics and benchmark probes.
#[derive(Debug, Clone, Default, Eq, PartialEq)]
pub struct RoutingStatistics {
    /// Total selected routes visible to the source rank.
    pub total_routes: usize,
    /// Selected routes owned and executed by this rank.
    pub local_routes: usize,
    /// Routes sent by a sharded-input exchange.
    pub sent_routes: usize,
    /// Routes received by a sharded-input exchange.
    pub received_routes: usize,
    /// Padding rows introduced by the fallback transport.
    pub padding_routes: usize,
    /// Explicit host-visible synchronization points.
    pub synchronization_count: usize,
    /// Payload bytes transferred logically, excluding backend internals.
    pub exchanged_bytes: usize,
    /// Time spent waiting for explicit route metadata synchronization.
    pub synchronization_time: Duration,
    /// Wall time spent computing router decisions.
    pub router_time: Duration,
    /// Wall time spent validating and compacting owner-local routes.
    pub compaction_time: Duration,
    /// Wall time spent in route transport collectives.
    pub exchange_time: Duration,
    /// Wall time spent in local expert computation.
    pub expert_time: Duration,
    /// Wall time spent reducing or recombining routed outputs.
    pub reduction_time: Duration,
    /// Wall time spent computing replicated shared experts.
    pub shared_expert_time: Duration,
    /// End-to-end wall time summed across represented MoE blocks or one dispatch.
    pub total_time: Duration,
    /// End-to-end wall time for the complete model forward containing those blocks.
    pub model_time: Duration,
}

impl RoutingStatistics {
    /// Adds counters and measured synchronization time from another dispatch.
    pub fn accumulate(&mut self, other: &Self) {
        self.total_routes += other.total_routes;
        self.local_routes += other.local_routes;
        self.sent_routes += other.sent_routes;
        self.received_routes += other.received_routes;
        self.padding_routes += other.padding_routes;
        self.synchronization_count += other.synchronization_count;
        self.exchanged_bytes += other.exchanged_bytes;
        self.synchronization_time += other.synchronization_time;
        self.router_time += other.router_time;
        self.compaction_time += other.compaction_time;
        self.exchange_time += other.exchange_time;
        self.expert_time += other.expert_time;
        self.reduction_time += other.reduction_time;
        self.shared_expert_time += other.shared_expert_time;
        self.total_time += other.total_time;
        self.model_time += other.model_time;
    }
}

/// Compact device-side routes owned by the current rank.
pub struct DispatchedRoutes {
    /// Hidden rows in stable original route order.
    pub hidden: Array,
    /// Checkpoint-global expert ids.
    pub global_expert_ids: Array,
    /// Dense owner-local ids passed to grouped kernels.
    pub local_expert_ids: Array,
    /// Original flattened route positions.
    pub original_route_indices: Array,
    /// Source token indices.
    pub token_indices: Array,
    /// Top-k slot indices.
    pub slot_indices: Array,
    /// Route weights, not yet applied.
    pub weights: Array,
}

/// Result of a replicated-input expert dispatch.
pub struct ReturnedRoutes {
    /// Rank-local weighted token buffer before the collective.
    pub local_output: Array,
    /// Exact routed output after all-sum.
    pub reduced_output: Array,
    /// Dispatch counters.
    pub statistics: RoutingStatistics,
}

/// Architecture-specific execution behind the common route dispatcher.
pub trait LocalExpertBank {
    /// Executes compact hidden rows using dense owner-local expert ids.
    /// Returned route rows must be unweighted and retain input order.
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error>;
}

pub(crate) fn unit_route_weights(
    routes: i32,
    dtype: Dtype,
    stream: &Stream,
) -> Result<Array, Error> {
    Ok(safemlx::ops::ones_dtype(&[routes, 1], dtype, stream)?)
}

impl LocalExpertBank for PackedSwiGluExperts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

impl LocalExpertBank for PackedRelu2Experts {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_expert_ids: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_expert_ids.reshape(&[-1, 1], stream)?;
        let weights = unit_route_weights(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

/// Compacts routes owned by this rank with exactly one scalar synchronization.
pub fn compact_local_routes(
    hidden_states: &Array,
    expert_ids: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<(DispatchedRoutes, RoutingStatistics), Error> {
    if expert_ids.ndim() != 2 || weights.shape() != expert_ids.shape() {
        return Err(Error::Parallel(format!(
            "expert ids and weights must have matching [tokens, top_k] shapes, got {:?} and {:?}",
            expert_ids.shape(),
            weights.shape()
        )));
    }
    if hidden_states.ndim() != 2 || hidden_states.dim(0) != expert_ids.dim(0) {
        return Err(Error::Parallel(format!(
            "hidden states must be [tokens, hidden] matching route tokens, got {:?}",
            hidden_states.shape()
        )));
    }
    if !matches!(
        expert_ids.dtype(),
        Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
    ) {
        return Err(Error::Parallel(format!(
            "expert ids must use an integer dtype, got {:?}",
            expert_ids.dtype()
        )));
    }
    if !weights.dtype().is_float() || !hidden_states.dtype().is_float() {
        return Err(Error::Parallel(
            "route weights and hidden states must be floating point".into(),
        ));
    }
    let flat_ids = expert_ids
        .reshape(&[-1], stream)?
        .as_dtype(Dtype::Int32, stream)?;
    let valid = flat_ids.ge(Array::from_int(0), stream)?.logical_and(
        flat_ids.lt(
            Array::from_int(assignment.global_expert_count as i32),
            stream,
        )?,
        stream,
    )?;
    let invalid = valid.logical_not(stream)?.count_nonzero(stream)?;
    // Use a safe placeholder for invalid ids so validation and the compact
    // count can share the same single host synchronization below.
    let safe_ids = r#where(
        &valid,
        flat_ids.clone(),
        Array::zeros::<i32>(&[flat_ids.size() as i32], stream)?,
        stream,
    )?;
    let owners = Array::from_slice(
        &assignment
            .owners
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>(),
        &[assignment.global_expert_count as i32],
    );
    let owner_local = Array::from_slice(
        &assignment
            .owner_local
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>(),
        &[assignment.global_expert_count as i32],
    );
    let route_owners = owners.take(&safe_ids, stream)?;
    let mask = route_owners
        .eq(Array::from_int(assignment.rank as i32), stream)?
        .logical_and(valid, stream)?;
    let compact = mask.compact_indices(stream)?;
    let started = std::time::Instant::now();
    eval([&invalid, &compact.count])?;
    let synchronization_time = started.elapsed();
    if invalid.clone().try_item::<i32>(stream)? != 0 {
        return Err(Error::Parallel(
            "route contains a globally invalid expert id".into(),
        ));
    }
    let local_routes = compact.count.clone().try_item::<i32>(stream)? as usize;
    let positions = compact
        .indices
        .try_index_device(..local_routes as i32, stream)?;
    let global_expert_ids = flat_ids.take(&positions, stream)?;
    let local_expert_ids = owner_local.take(&global_expert_ids, stream)?;
    let top_k = expert_ids.dim(1);
    let token_indices = positions.floor_divide(Array::from_int(top_k), stream)?;
    let slot_indices = positions.remainder(Array::from_int(top_k), stream)?;
    let hidden = hidden_states.take_axis(&token_indices, 0, stream)?;
    let route_weights = weights.reshape(&[-1], stream)?.take(&positions, stream)?;
    Ok((
        DispatchedRoutes {
            hidden,
            global_expert_ids,
            local_expert_ids,
            original_route_indices: positions,
            token_indices,
            slot_indices,
            weights: route_weights,
        },
        RoutingStatistics {
            total_routes: expert_ids.size(),
            local_routes,
            synchronization_count: 1,
            synchronization_time,
            ..RoutingStatistics::default()
        },
    ))
}

/// Executes compact local routes and exactly recombines them across EP ranks.
pub fn dispatch_replicated(
    hidden_states: &Array,
    expert_ids: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    bank: &mut impl LocalExpertBank,
    group: &Group,
    stream: &Stream,
) -> Result<ReturnedRoutes, Error> {
    dispatch_replicated_with(
        hidden_states,
        expert_ids,
        weights,
        assignment,
        group,
        stream,
        |routes, stream| {
            bank.execute_local_routes(&routes.hidden, &routes.local_expert_ids, stream)
        },
    )
}

/// Dispatches replicated routes while delegating exact local route execution.
///
/// The callback receives both global and owner-local ids after the existing
/// validated route compaction, so cache-backed banks can retain global identity
/// without duplicating transport or recombination.
pub fn dispatch_replicated_with<F>(
    hidden_states: &Array,
    expert_ids: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    group: &Group,
    stream: &Stream,
    execute: F,
) -> Result<ReturnedRoutes, Error>
where
    F: FnOnce(&DispatchedRoutes, &Stream) -> Result<Array, Error>,
{
    let total_started = Instant::now();
    if group.rank() != assignment.rank || group.size() != assignment.group_size {
        return Err(Error::Parallel(
            "expert assignment does not match the supplied group".into(),
        ));
    }
    let compaction_started = Instant::now();
    let (routes, mut statistics) =
        compact_local_routes(hidden_states, expert_ids, weights, assignment, stream)?;
    materialize_timing_phase([
        &routes.hidden,
        &routes.global_expert_ids,
        &routes.local_expert_ids,
        &routes.original_route_indices,
        &routes.token_indices,
        &routes.slot_indices,
        &routes.weights,
    ])?;
    statistics.compaction_time += compaction_started.elapsed();
    let expert_started = Instant::now();
    let local_output = if statistics.local_routes == 0 {
        zeros_dtype(hidden_states.shape(), hidden_states.dtype(), stream)?
    } else {
        let output = execute(&routes, stream)?;
        if output.ndim() != 2 || output.dim(0) != statistics.local_routes as i32 {
            return Err(Error::Parallel(format!(
                "local expert bank returned invalid shape {:?}",
                output.shape()
            )));
        }
        let weighted = output.multiply(routes.weights.expand_dims(1, stream)?, stream)?;
        segment_sum_by_index(
            weighted,
            &routes.token_indices,
            hidden_states.dim(0),
            stream,
        )?
    };
    materialize_timing_phase([&local_output])?;
    statistics.expert_time += expert_started.elapsed();
    let reduction_started = Instant::now();
    let reduced_output = distributed::all_sum(&local_output, group, stream)?;
    materialize_timing_phase([&reduced_output])?;
    statistics.reduction_time += reduction_started.elapsed();
    statistics.total_time = total_started.elapsed();
    Ok(ReturnedRoutes {
        local_output,
        reduced_output,
        statistics,
    })
}

/// Result of one variable-count all-to-all fallback.
pub struct ExchangeResult {
    /// Received rows concatenated in source-rank order.
    pub received: Array,
    /// Number of logical rows received from every source rank.
    pub source_counts: Vec<usize>,
    /// Transport counters.
    pub statistics: RoutingStatistics,
}

/// Destination-major route blocks for sharded-token expert dispatch.
///
/// Every vector has exactly one block per destination EP rank and matching
/// leading row counts. Global expert ids and original flattened route indices
/// remain visible at this transport boundary.
pub struct ShardedRouteBlocks {
    /// Hidden activation rows addressed to each expert owner.
    pub hidden: Vec<Array>,
    /// Checkpoint-global expert ids for each row.
    pub global_expert_ids: Vec<Array>,
    /// Original source-rank flattened route indices for each row.
    pub original_route_indices: Vec<Array>,
    /// Route weights for each row, applied exactly once by the owner.
    pub weights: Vec<Array>,
    /// Number of top-k slots per source token.
    pub top_k: i32,
    /// Number of tokens owned by this source rank.
    pub source_tokens: i32,
}

/// Returned source-local output from sharded-input dispatch.
pub struct ShardedReturnedRoutes {
    /// Weighted route output reduced to source token order.
    pub output: Array,
    /// Transport and execution counters.
    pub statistics: RoutingStatistics,
}

fn validate_sharded_blocks(blocks: &ShardedRouteBlocks, world: usize) -> Result<(), Error> {
    if blocks.top_k <= 0 || blocks.source_tokens < 0 {
        return Err(Error::Parallel(
            "sharded dispatch requires positive top_k and nonnegative source token count".into(),
        ));
    }
    if blocks.hidden.len() != world
        || blocks.global_expert_ids.len() != world
        || blocks.original_route_indices.len() != world
        || blocks.weights.len() != world
    {
        return Err(Error::Parallel(format!(
            "sharded dispatch requires {world} blocks for every payload and metadata field"
        )));
    }
    for destination in 0..world {
        let rows = blocks.hidden[destination].dim(0);
        if blocks.hidden[destination].ndim() != 2
            || blocks.global_expert_ids[destination].shape() != [rows]
            || blocks.original_route_indices[destination].shape() != [rows]
            || blocks.weights[destination].shape() != [rows]
        {
            return Err(Error::Parallel(format!(
                "destination {destination} sharded route fields have inconsistent row counts"
            )));
        }
    }
    Ok(())
}

/// Exchanges sharded-token routes, executes owner-local experts, and returns
/// exact weighted results to their source ranks.
///
/// All payload and metadata exchange uses [`all_to_all_v`]. Collectives are
/// entered in a fixed order on every rank, including ranks with zero routes.
pub fn dispatch_sharded(
    blocks: ShardedRouteBlocks,
    assignment: &ExpertAssignment,
    bank: &mut impl LocalExpertBank,
    group: &Group,
    stream: &Stream,
) -> Result<ShardedReturnedRoutes, Error> {
    let total_started = Instant::now();
    if group.rank() != assignment.rank() || group.size() != assignment.group_size() {
        return Err(Error::Parallel(
            "expert assignment does not match the supplied group".into(),
        ));
    }
    validate_sharded_blocks(&blocks, group.size())?;
    let total_routes = blocks
        .hidden
        .iter()
        .map(|block| block.dim(0) as usize)
        .sum();
    let hidden = all_to_all_v(&blocks.hidden, group, stream)?;
    let global_ids = all_to_all_v(&blocks.global_expert_ids, group, stream)?;
    let route_indices = all_to_all_v(&blocks.original_route_indices, group, stream)?;
    let weights = all_to_all_v(&blocks.weights, group, stream)?;
    if hidden.source_counts != global_ids.source_counts
        || hidden.source_counts != route_indices.source_counts
        || hidden.source_counts != weights.source_counts
    {
        return Err(Error::Parallel(
            "sharded route payload and metadata receive counts diverged".into(),
        ));
    }
    let received_routes = hidden.received.dim(0);
    let owner_local = Array::from_slice(
        &assignment
            .owner_local_ids()
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>(),
        &[assignment.global_expert_count() as i32],
    );
    let local_ids =
        owner_local.take(&global_ids.received.as_dtype(Dtype::Int32, stream)?, stream)?;
    let expert_started = Instant::now();
    let weighted = if received_routes == 0 {
        let mut shape = hidden.received.shape().to_vec();
        shape[0] = 0;
        zeros_dtype(&shape, hidden.received.dtype(), stream)?
    } else {
        bank.execute_local_routes(&hidden.received, &local_ids, stream)?
            .multiply(weights.received.expand_dims(1, stream)?, stream)?
    };
    materialize_timing_phase([&weighted])?;
    let expert_time = expert_started.elapsed();
    let mut output_to_source = Vec::with_capacity(group.size());
    let mut indices_to_source = Vec::with_capacity(group.size());
    let mut offset = 0i32;
    for count in &hidden.source_counts {
        let end = offset + *count as i32;
        output_to_source.push(weighted.try_index_device(offset..end, stream)?);
        indices_to_source.push(
            route_indices
                .received
                .try_index_device(offset..end, stream)?,
        );
        offset = end;
    }
    let returned_output = all_to_all_v(&output_to_source, group, stream)?;
    let returned_indices = all_to_all_v(&indices_to_source, group, stream)?;
    if returned_output.source_counts != returned_indices.source_counts {
        return Err(Error::Parallel(
            "returned sharded outputs and route indices diverged".into(),
        ));
    }
    let token_indices = returned_indices
        .received
        .as_dtype(Dtype::Int32, stream)?
        .floor_divide(Array::from_int(blocks.top_k), stream)?;
    let reduction_started = Instant::now();
    let output = segment_sum_by_index(
        returned_output.received.clone(),
        token_indices,
        blocks.source_tokens,
        stream,
    )?;
    materialize_timing_phase([&output])?;
    let reduction_time = reduction_started.elapsed();
    let mut statistics = RoutingStatistics {
        total_routes,
        local_routes: received_routes as usize,
        sent_routes: total_routes,
        received_routes: received_routes as usize,
        expert_time,
        reduction_time,
        ..Default::default()
    };
    for exchange in [
        hidden,
        global_ids,
        route_indices,
        weights,
        returned_output,
        returned_indices,
    ] {
        statistics.padding_routes += exchange.statistics.padding_routes;
        statistics.synchronization_count += exchange.statistics.synchronization_count;
        statistics.exchanged_bytes += exchange.statistics.exchanged_bytes;
        statistics.synchronization_time += exchange.statistics.synchronization_time;
        statistics.exchange_time += exchange.statistics.exchange_time;
    }
    statistics.total_time = total_started.elapsed();
    Ok(ShardedReturnedRoutes { output, statistics })
}

/// Exchanges destination-major variable-sized blocks using padded all-gather.
///
/// `send_blocks[d]` contains rows addressed to destination rank `d`; all
/// blocks must have the same trailing shape and dtype.  The fallback gathers
/// `group_size` destination blocks from every source, so peak transfer storage
/// and bandwidth are `O(group_size)` larger than a native all-to-all.
pub fn all_to_all_v(
    send_blocks: &[Array],
    group: &Group,
    stream: &Stream,
) -> Result<ExchangeResult, Error> {
    let total_started = Instant::now();
    let world = group.size();
    if send_blocks.len() != world || send_blocks.is_empty() {
        return Err(Error::Parallel(format!(
            "all_to_all_v requires {world} destination blocks"
        )));
    }
    if send_blocks.iter().any(|block| block.ndim() == 0) {
        return Err(Error::Parallel(
            "all_to_all_v blocks must have a leading row dimension".into(),
        ));
    }
    let dtype = send_blocks[0].dtype();
    let first_shape = send_blocks[0].shape();
    let tail = &first_shape[1..];
    if send_blocks
        .iter()
        .any(|block| block.dtype() != dtype || &block.shape()[1..] != tail)
    {
        return Err(Error::Parallel(
            "all_to_all_v blocks must share dtype and trailing shape".into(),
        ));
    }
    let local_counts = send_blocks
        .iter()
        .map(|block| block.dim(0))
        .collect::<Vec<_>>();
    // Materialize the tiny host count vector onto the explicit execution
    // stream before entering the collective.
    let counts = Array::from_slice(&local_counts, &[world as i32]).copy(stream)?;
    let gathered_counts = distributed::all_gather(&counts, group, stream)?;
    let started = std::time::Instant::now();
    let evaluated_counts = gathered_counts.evaluated()?;
    let synchronization_time = started.elapsed();
    let all_counts = evaluated_counts.as_slice::<i32>();
    let max_rows = all_counts.iter().copied().max().unwrap_or(0) as usize;
    if max_rows == 0 {
        let mut shape = send_blocks[0].shape().to_vec();
        shape[0] = 0;
        let exchange_time = total_started.elapsed();
        return Ok(ExchangeResult {
            received: zeros_dtype(&shape, dtype, stream)?,
            source_counts: vec![0; world],
            statistics: RoutingStatistics {
                synchronization_count: 1,
                synchronization_time,
                exchange_time,
                total_time: exchange_time,
                ..Default::default()
            },
        });
    }
    let mut padded = Vec::with_capacity(world);
    for block in send_blocks {
        let rows = block.dim(0) as usize;
        if rows == max_rows {
            padded.push(block.clone());
        } else {
            let mut shape = block.shape().to_vec();
            shape[0] = (max_rows - rows) as i32;
            let padding = zeros_dtype(&shape, dtype, stream)?;
            padded.push(concatenate_axis(&[block, &padding], 0, stream)?);
        }
    }
    let refs = padded.iter().collect::<Vec<_>>();
    let packed = concatenate_axis(&refs, 0, stream)?;
    let gathered = distributed::all_gather(&packed, group, stream)?;
    let mut received = Vec::with_capacity(world);
    let mut source_counts = Vec::with_capacity(world);
    for source in 0..world {
        let count = all_counts[source * world + group.rank()] as usize;
        source_counts.push(count);
        let start = (source * world * max_rows + group.rank() * max_rows) as i32;
        received.push(gathered.try_index_device(start..start + count as i32, stream)?);
    }
    let refs = received.iter().collect::<Vec<_>>();
    let received = concatenate_axis(&refs, 0, stream)?;
    materialize_timing_phase([&received])?;
    let sent_routes = local_counts
        .iter()
        .map(|value| *value as usize)
        .sum::<usize>();
    let received_routes = source_counts.iter().sum::<usize>();
    let padding_routes = world * world * max_rows
        - all_counts
            .iter()
            .map(|value| *value as usize)
            .sum::<usize>();
    let row_bytes = tail
        .iter()
        .map(|dimension| *dimension as usize)
        .product::<usize>()
        * send_blocks[0].item_size();
    let exchange_time = total_started.elapsed();
    Ok(ExchangeResult {
        received,
        source_counts,
        statistics: RoutingStatistics {
            sent_routes,
            received_routes,
            padding_routes,
            synchronization_count: 1,
            synchronization_time,
            exchange_time,
            total_time: exchange_time,
            exchanged_bytes: world * world * max_rows * row_bytes,
            ..Default::default()
        },
    })
}
