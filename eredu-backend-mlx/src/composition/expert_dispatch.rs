//! Architecture-owned expert assignment and routing over generic MLX collectives.
//!
//! Pure expert parallelism keeps ordinary model state replicated and partitions
//! only routed expert banks. [`crate::composition::expert_dispatch::dispatch_replicated`]
//! exploits the replicated
//! token layout: ranks compact only routes owned by their experts and all-sum
//! the resulting token buffer. Sharded-token dispatch uses one reusable
//! [`AllToAllVPlan`](crate::composition::expert_dispatch::AllToAllVPlan)
//! and compact variable-count exchanges in both directions.

use std::{
    cell::Cell,
    time::{Duration, Instant},
};

use eredu_nn::TensorParallelGroupedOutput;
use safemlx::{
    ops::{concatenate_axis, indexing::TryIndexOp, r#where, zeros_dtype},
    transforms::{depends, eval},
    Array, Dtype, Stream,
};

use crate::{
    backend::compaction::{compact_indices, count_nonzero},
    backend::error::Error,
    backend::nn::grouped::{PackedGatedProductGroups, PackedRelu2Groups},
    backend::nn::grouping::segment_sum_by_index,
    backend::runtime::distributed::{self as distributed, Group},
};

impl LocalExpertBank for crate::backend::nn::shared::MlxGroupedRelu2 {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        let routes = eredu_nn::GroupSelection::new(
            crate::MlxTensor::from_array(ids),
            crate::MlxTensor::from_array(weights.clone()),
            crate::MlxTensor::from_array(weights),
        );
        eredu_nn::GroupedRelu2Operator::forward_grouped(
            self,
            &crate::MlxTensor::from_array(hidden.clone()),
            &routes,
            stream,
        )
        .map(|value| value.as_array().clone())
        .map_err(Error::from)
    }
}

impl LocalExpertBank for crate::backend::nn::shared::MlxGroupedGatedProduct {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        let routes = eredu_nn::GroupSelection::new(
            crate::MlxTensor::from_array(ids),
            crate::MlxTensor::from_array(weights.clone()),
            crate::MlxTensor::from_array(weights),
        );
        eredu_nn::GroupedGatedProductOperator::forward_grouped(
            self,
            &crate::MlxTensor::from_array(hidden.clone()),
            &routes,
            stream,
        )
        .map(|value| value.as_array().clone())
        .map_err(Error::from)
    }
}

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

/// Returns whether eager expert-parallel phase timing is enabled on this thread.
pub fn timing_profiling_enabled() -> bool {
    EAGER_TIMING_PROFILING.with(Cell::get)
}

/// Materializes a phase's outputs when eager timing is enabled.
pub fn materialize_timing_phase<'a>(
    outputs: impl IntoIterator<Item = &'a Array>,
) -> safemlx::error::Result<()> {
    if timing_profiling_enabled() {
        eval(outputs)?;
    }
    Ok(())
}

/// Validated bidirectional mapping between checkpoint-global and owner-local ids.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ExpertAssignment {
    global_expert_count: usize,
    group_size: usize,
    rank: usize,
    owners: Vec<usize>,
    owner_local: Vec<usize>,
    local_global: Vec<usize>,
}

impl ExpertAssignment {
    /// Lowers an architecture-derived realization without recomputing ownership.
    pub fn from_realization<S>(
        plan: &eredu_architectures::ExpertRealizationPlan<S>,
    ) -> Result<Self, Error> {
        Self::from_owners(
            plan.owners().to_vec(),
            plan.expert_parallel_size(),
            plan.expert_parallel_rank(),
        )
    }

    fn from_owners(owners: Vec<usize>, group_size: usize, rank: usize) -> Result<Self, Error> {
        validate_dimensions(owners.len(), group_size, rank)?;
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
        if next_local.contains(&0) {
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
    /// Global expert ids owned by this rank, in owner-local order.
    pub fn local_global_group_indices(&self) -> &[usize] {
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

fn validate_dimensions(global_experts: usize, group_size: usize, rank: usize) -> Result<(), Error> {
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
    if global_experts < group_size {
        return Err(Error::Parallel(format!(
            "cannot assign {global_experts} experts to {group_size} non-empty ranks"
        )));
    }
    Ok(())
}

/// Route transport selected for a sharded expert exchange.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq)]
pub enum RoutedTransport {
    /// No routed payload transport was used.
    #[default]
    None,
    /// A native MLX distributed group executed the exchange.
    Native,
    /// Eredu's topology-planned neighbor routes executed the exchange.
    Logical,
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
    /// Padding rows introduced by transport; zero for all-to-all-v.
    pub padding_routes: usize,
    /// Useful logical payload bytes sent to all destinations.
    pub useful_sent_bytes: usize,
    /// Useful logical payload bytes received from all sources.
    pub useful_received_bytes: usize,
    /// Padding payload bytes transferred by the selected transport.
    pub padding_bytes: usize,
    /// Backend physical or Ring hop bytes, when exposed exactly.
    pub backend_physical_bytes: Option<usize>,
    /// Measured temporary/staging high-water bytes, when exposed exactly.
    pub temporary_high_water_bytes: Option<usize>,
    /// Conservative Eredu-visible retained-plus-Ring-staging payload bound.
    /// Backend-internal buffers are excluded when the backend does not expose them.
    pub payload_allocation_upper_bound_bytes: usize,
    /// Count-matrix consensus operations performed for this dispatch.
    pub count_consensus_count: usize,
    /// Time spent materializing count-matrix consensus.
    pub count_consensus_time: Duration,
    /// Other explicit host-visible validation synchronizations.
    pub host_synchronization_count: usize,
    /// Time spent in explicit host-visible validation synchronization.
    pub host_synchronization_time: Duration,
    /// Native versus topology-routed payload transport.
    pub routed_transport: RoutedTransport,
    /// Wall time spent computing router decisions.
    pub router_time: Duration,
    /// Wall time spent validating and compacting owner-local routes.
    pub compaction_time: Duration,
    /// Wall time spent in route transport collectives.
    pub payload_exchange_time: Duration,
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
        let self_has_routed_transport = self.routed_transport != RoutedTransport::None;
        let other_has_routed_transport = other.routed_transport != RoutedTransport::None;
        self.total_routes += other.total_routes;
        self.local_routes += other.local_routes;
        self.sent_routes += other.sent_routes;
        self.received_routes += other.received_routes;
        self.padding_routes += other.padding_routes;
        self.useful_sent_bytes += other.useful_sent_bytes;
        self.useful_received_bytes += other.useful_received_bytes;
        self.padding_bytes += other.padding_bytes;
        self.backend_physical_bytes = match (self_has_routed_transport, other_has_routed_transport)
        {
            (false, false) => None,
            (false, true) => other.backend_physical_bytes,
            (true, false) => self.backend_physical_bytes,
            (true, true) => match (self.backend_physical_bytes, other.backend_physical_bytes) {
                (Some(left), Some(right)) => left.checked_add(right),
                _ => None,
            },
        };
        self.temporary_high_water_bytes =
            match (self_has_routed_transport, other_has_routed_transport) {
                (false, false) => None,
                (false, true) => other.temporary_high_water_bytes,
                (true, false) => self.temporary_high_water_bytes,
                (true, true) => match (
                    self.temporary_high_water_bytes,
                    other.temporary_high_water_bytes,
                ) {
                    (Some(left), Some(right)) => Some(left.max(right)),
                    _ => None,
                },
            };
        self.payload_allocation_upper_bound_bytes += other.payload_allocation_upper_bound_bytes;
        self.count_consensus_count += other.count_consensus_count;
        self.count_consensus_time += other.count_consensus_time;
        self.host_synchronization_count += other.host_synchronization_count;
        self.host_synchronization_time += other.host_synchronization_time;
        if other.routed_transport != RoutedTransport::None {
            self.routed_transport = other.routed_transport;
        }
        self.router_time += other.router_time;
        self.compaction_time += other.compaction_time;
        self.payload_exchange_time += other.payload_exchange_time;
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
    pub global_group_indices: Array,
    /// Dense owner-local ids passed to grouped kernels.
    pub local_group_indices: Array,
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
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error>;

    /// Executes owner-local rows while separating replicated TP down bias.
    fn execute_local_routes_tensor_parallel(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, Error> {
        if partitions == 0 {
            return Err(Error::Parallel(
                "tensor-parallel partition count must be positive".into(),
            ));
        }
        Ok(TensorParallelGroupedOutput::new(
            self.execute_local_routes(hidden, local_group_indices, stream)?,
            None,
        ))
    }
}

/// Creates one unit weight for every routed token.
pub fn unit_coefficients(routes: i32, dtype: Dtype, stream: &Stream) -> Result<Array, Error> {
    Ok(safemlx::ops::ones_dtype(&[routes, 1], dtype, stream)?)
}

impl LocalExpertBank for PackedGatedProductGroups {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }

    fn execute_local_routes_tensor_parallel(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        partitions: usize,
        stream: &Stream,
    ) -> Result<TensorParallelGroupedOutput<Array>, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward_tensor_parallel(hidden, &ids, &weights, partitions, stream)?)
    }
}

impl LocalExpertBank for PackedRelu2Groups {
    fn execute_local_routes(
        &mut self,
        hidden: &Array,
        local_group_indices: &Array,
        stream: &Stream,
    ) -> Result<Array, Error> {
        let ids = local_group_indices.reshape(&[-1, 1], stream)?;
        let weights = unit_coefficients(hidden.dim(0), hidden.dtype(), stream)?;
        Ok(self.forward(hidden, &ids, &weights, stream)?)
    }
}

/// Compacts routes owned by this rank with exactly one scalar synchronization.
pub fn compact_local_routes(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    stream: &Stream,
) -> Result<(DispatchedRoutes, RoutingStatistics), Error> {
    if group_indices.ndim() != 2 || weights.shape() != group_indices.shape() {
        return Err(Error::Parallel(format!(
            "expert ids and weights must have matching [tokens, top_k] shapes, got {:?} and {:?}",
            group_indices.shape(),
            weights.shape()
        )));
    }
    if hidden_states.ndim() != 2 || hidden_states.dim(0) != group_indices.dim(0) {
        return Err(Error::Parallel(format!(
            "hidden states must be [tokens, hidden] matching route tokens, got {:?}",
            hidden_states.shape()
        )));
    }
    if !matches!(
        group_indices.dtype(),
        Dtype::Int32 | Dtype::Uint32 | Dtype::Int64 | Dtype::Uint64
    ) {
        return Err(Error::Parallel(format!(
            "expert ids must use an integer dtype, got {:?}",
            group_indices.dtype()
        )));
    }
    if !weights.dtype().is_float() || !hidden_states.dtype().is_float() {
        return Err(Error::Parallel(
            "route weights and hidden states must be floating point".into(),
        ));
    }
    let flat_ids = group_indices
        .reshape(&[-1], stream)?
        .as_dtype(Dtype::Int32, stream)?;
    let valid = flat_ids.ge(Array::from_int(0), stream)?.logical_and(
        flat_ids.lt(
            Array::from_int(assignment.global_expert_count as i32),
            stream,
        )?,
        stream,
    )?;
    let invalid = count_nonzero(&valid.logical_not(stream)?, stream)?;
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
    let compact = compact_indices(&mask, stream)?;
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
    let global_group_indices = flat_ids.take(&positions, stream)?;
    let local_group_indices = owner_local.take(&global_group_indices, stream)?;
    let top_k = group_indices.dim(1);
    let token_indices = positions.floor_divide(Array::from_int(top_k), stream)?;
    let slot_indices = positions.remainder(Array::from_int(top_k), stream)?;
    let hidden = hidden_states.take_axis(&token_indices, 0, stream)?;
    let coefficients = weights.reshape(&[-1], stream)?.take(&positions, stream)?;
    Ok((
        DispatchedRoutes {
            hidden,
            global_group_indices,
            local_group_indices,
            original_route_indices: positions,
            token_indices,
            slot_indices,
            weights: coefficients,
        },
        RoutingStatistics {
            total_routes: group_indices.size(),
            local_routes,
            host_synchronization_count: 1,
            host_synchronization_time: synchronization_time,
            ..RoutingStatistics::default()
        },
    ))
}

/// Executes compact local routes and exactly recombines them across EP ranks.
pub fn dispatch_replicated(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    bank: &mut impl LocalExpertBank,
    group: &Group,
    stream: &Stream,
) -> Result<ReturnedRoutes, Error> {
    dispatch_replicated_with(
        hidden_states,
        group_indices,
        weights,
        assignment,
        group,
        stream,
        |routes, stream| {
            bank.execute_local_routes(&routes.hidden, &routes.local_group_indices, stream)
        },
    )
}

/// EP-reduced rank-local TP contribution with replicated bias kept separate.
pub struct TensorParallelReturnedRoutes {
    /// Tensor-parallel projection contribution and literal post-reduce bias.
    pub output: TensorParallelGroupedOutput<Array>,
    /// Dispatch counters shared with ordinary expert execution.
    pub statistics: RoutingStatistics,
}

/// Dispatches replicated routes across EP while preserving the TP bias split.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_replicated_tensor_parallel(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    bank: &mut impl LocalExpertBank,
    group: &Group,
    partitions: usize,
    stream: &Stream,
) -> Result<TensorParallelReturnedRoutes, Error> {
    let hidden_dimensions = hidden_states.dim(-1);
    let returned = dispatch_replicated_with_output_dimensions(
        hidden_states,
        group_indices,
        weights,
        assignment,
        group,
        2 * hidden_dimensions,
        stream,
        |routes, stream| {
            let output = bank.execute_local_routes_tensor_parallel(
                &routes.hidden,
                &routes.local_group_indices,
                partitions,
                stream,
            )?;
            let (reducible, post_reduce) = output.into_parts();
            let post_reduce = match post_reduce {
                Some(bias) => bias,
                None => zeros_dtype(reducible.shape(), reducible.dtype(), stream)?,
            };
            Ok(concatenate_axis(&[reducible, post_reduce], -1, stream)?)
        },
    )?;
    let reducible = returned
        .reduced_output
        .try_index_device((.., ..hidden_dimensions), stream)?;
    let post_reduce = returned
        .reduced_output
        .try_index_device((.., hidden_dimensions..), stream)?;
    Ok(TensorParallelReturnedRoutes {
        output: TensorParallelGroupedOutput::new(reducible, Some(post_reduce)),
        statistics: returned.statistics,
    })
}

/// Dispatches singleton-owner routes while preserving the TP bias split.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_local_tensor_parallel(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    bank: &mut impl LocalExpertBank,
    partitions: usize,
    stream: &Stream,
) -> Result<TensorParallelReturnedRoutes, Error> {
    if assignment.rank != 0 || assignment.group_size != 1 {
        return Err(Error::Parallel(
            "collective-free tensor-parallel expert dispatch requires a singleton rank-zero assignment"
                .into(),
        ));
    }
    let hidden_dimensions = hidden_states.dim(-1);
    let returned = dispatch_owned_with(
        hidden_states,
        group_indices,
        weights,
        assignment,
        None,
        2 * hidden_dimensions,
        stream,
        |routes, stream| {
            let output = bank.execute_local_routes_tensor_parallel(
                &routes.hidden,
                &routes.local_group_indices,
                partitions,
                stream,
            )?;
            let (reducible, post_reduce) = output.into_parts();
            let post_reduce = match post_reduce {
                Some(bias) => bias,
                None => zeros_dtype(reducible.shape(), reducible.dtype(), stream)?,
            };
            Ok(concatenate_axis(&[reducible, post_reduce], -1, stream)?)
        },
    )?;
    let reducible = returned
        .reduced_output
        .try_index_device((.., ..hidden_dimensions), stream)?;
    let post_reduce = returned
        .reduced_output
        .try_index_device((.., hidden_dimensions..), stream)?;
    Ok(TensorParallelReturnedRoutes {
        output: TensorParallelGroupedOutput::new(reducible, Some(post_reduce)),
        statistics: returned.statistics,
    })
}

/// Dispatches replicated routes while delegating exact local route execution.
///
/// The callback receives both global and owner-local ids after the existing
/// validated route compaction, so cache-backed banks can retain global identity
/// without duplicating transport or recombination.
pub fn dispatch_replicated_with<F>(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    group: &Group,
    stream: &Stream,
    execute: F,
) -> Result<ReturnedRoutes, Error>
where
    F: FnOnce(&DispatchedRoutes, &Stream) -> Result<Array, Error>,
{
    dispatch_replicated_with_output_dimensions(
        hidden_states,
        group_indices,
        weights,
        assignment,
        group,
        hidden_states.dim(-1),
        stream,
        execute,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_replicated_with_output_dimensions<F>(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    group: &Group,
    output_dimensions: i32,
    stream: &Stream,
    execute: F,
) -> Result<ReturnedRoutes, Error>
where
    F: FnOnce(&DispatchedRoutes, &Stream) -> Result<Array, Error>,
{
    if group.rank() != assignment.rank || group.size() != assignment.group_size {
        return Err(Error::Parallel(
            "expert assignment does not match the supplied group".into(),
        ));
    }
    dispatch_owned_with(
        hidden_states,
        group_indices,
        weights,
        assignment,
        Some(group),
        output_dimensions,
        stream,
        execute,
    )
}

/// Dispatches routes to a singleton expert owner without creating a collective.
///
/// This is the EP-degree-one specialization of [`dispatch_replicated_with`].
/// It retains the same validation, route compaction, cache callback, weighted
/// recombination, and telemetry while making the absence of an EP communicator
/// explicit.
pub fn dispatch_local_with<F>(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    stream: &Stream,
    execute: F,
) -> Result<ReturnedRoutes, Error>
where
    F: FnOnce(&DispatchedRoutes, &Stream) -> Result<Array, Error>,
{
    if assignment.rank != 0 || assignment.group_size != 1 {
        return Err(Error::Parallel(
            "collective-free expert dispatch requires a singleton rank-zero assignment".into(),
        ));
    }
    dispatch_owned_with(
        hidden_states,
        group_indices,
        weights,
        assignment,
        None,
        hidden_states.dim(-1),
        stream,
        execute,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_owned_with<F>(
    hidden_states: &Array,
    group_indices: &Array,
    weights: &Array,
    assignment: &ExpertAssignment,
    group: Option<&Group>,
    output_dimensions: i32,
    stream: &Stream,
    execute: F,
) -> Result<ReturnedRoutes, Error>
where
    F: FnOnce(&DispatchedRoutes, &Stream) -> Result<Array, Error>,
{
    let total_started = Instant::now();
    let compaction_started = Instant::now();
    let (routes, mut statistics) =
        compact_local_routes(hidden_states, group_indices, weights, assignment, stream)?;
    materialize_timing_phase([
        &routes.hidden,
        &routes.global_group_indices,
        &routes.local_group_indices,
        &routes.original_route_indices,
        &routes.token_indices,
        &routes.slot_indices,
        &routes.weights,
    ])?;
    statistics.compaction_time += compaction_started.elapsed();
    let expert_started = Instant::now();
    let local_output = if statistics.local_routes == 0 {
        zeros_dtype(
            &[hidden_states.dim(0), output_dimensions],
            hidden_states.dtype(),
            stream,
        )?
    } else {
        let output = execute(&routes, stream)?;
        if output.ndim() != 2
            || output.dim(0) != statistics.local_routes as i32
            || output.dim(1) != output_dimensions
        {
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
    let reduced_output = match group {
        Some(group) => {
            let reduction_started = Instant::now();
            let output = distributed::all_sum(&local_output, group, stream)?;
            materialize_timing_phase([&output])?;
            statistics.reduction_time += reduction_started.elapsed();
            output
        }
        None => local_output.clone(),
    };
    statistics.total_time = total_started.elapsed();
    Ok(ReturnedRoutes {
        local_output,
        reduced_output,
        statistics,
    })
}

/// Reusable count and receive-layout plan for a complete expert dispatch.
#[derive(Debug, Clone)]
pub struct AllToAllVPlan {
    send_counts: Vec<usize>,
    recv_counts: Vec<usize>,
    count_matrix: Vec<usize>,
    count_consensus_count: usize,
    count_consensus_time: Duration,
}

impl AllToAllVPlan {
    /// Gather one source/destination count matrix and derive this rank's
    /// source-major receive counts.
    pub fn new(send_counts: &[usize], group: &Group, stream: &Stream) -> Result<Self, Error> {
        if send_counts.len() != group.size() {
            return Err(Error::Parallel(format!(
                "all-to-all-v plan requires {} send counts, got {}",
                group.size(),
                send_counts.len()
            )));
        }
        let local_counts = send_counts
            .iter()
            .map(|count| {
                i32::try_from(*count)
                    .map_err(|_| Error::Parallel("all-to-all-v count exceeds i32".into()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (count_matrix, count_consensus_count, count_consensus_time) = if group.size() == 1 {
            (send_counts.to_vec(), 0, Duration::ZERO)
        } else {
            let counts = Array::from_slice(
                &local_counts,
                &[i32::try_from(group.size())
                    .map_err(|_| Error::Parallel("EP group size exceeds i32".into()))?],
            )
            .copy(stream)?;
            let gathered = distributed::all_gather(&counts, group, stream)?;
            let started = Instant::now();
            let evaluated = gathered.evaluated()?;
            let elapsed = started.elapsed();
            let values = evaluated.as_slice::<i32>();
            let expected = group
                .size()
                .checked_mul(group.size())
                .ok_or_else(|| Error::Parallel("count matrix size overflowed usize".into()))?;
            if values.len() != expected {
                return Err(Error::Parallel(format!(
                    "all-to-all-v count matrix has {} entries, expected {expected}",
                    values.len()
                )));
            }
            let values = values
                .iter()
                .map(|value| {
                    usize::try_from(*value).map_err(|_| {
                        Error::Parallel(
                            "all-to-all-v count matrix contains a negative count".into(),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            (values, 1, elapsed)
        };
        let recv_counts = (0..group.size())
            .map(|source| count_matrix[source * group.size() + group.rank()])
            .collect();
        Ok(Self {
            send_counts: send_counts.to_vec(),
            recv_counts,
            count_matrix,
            count_consensus_count,
            count_consensus_time,
        })
    }

    /// Destination-major row counts supplied by this rank.
    pub fn send_counts(&self) -> &[usize] {
        &self.send_counts
    }

    /// Source-major row counts expected by this rank.
    pub fn recv_counts(&self) -> &[usize] {
        &self.recv_counts
    }

    /// Row-major source/destination count matrix materialized by the plan.
    pub fn count_matrix(&self) -> &[usize] {
        &self.count_matrix
    }

    /// Return the reverse exchange without another count consensus.
    pub fn reverse(&self) -> Self {
        let size = self.send_counts.len();
        let mut count_matrix = vec![0; self.count_matrix.len()];
        for source in 0..size {
            for destination in 0..size {
                count_matrix[destination * size + source] =
                    self.count_matrix[source * size + destination];
            }
        }
        Self {
            send_counts: self.recv_counts.clone(),
            recv_counts: self.send_counts.clone(),
            count_matrix,
            count_consensus_count: 0,
            count_consensus_time: Duration::ZERO,
        }
    }

    fn consensus_statistics(&self) -> RoutingStatistics {
        RoutingStatistics {
            count_consensus_count: self.count_consensus_count,
            count_consensus_time: self.count_consensus_time,
            ..Default::default()
        }
    }

    /// Apply this plan to one compact destination-major field.
    pub fn exchange(
        &self,
        input: &Array,
        group: &Group,
        stream: &Stream,
    ) -> Result<ExchangeResult, Error> {
        let send_rows = self.send_counts.iter().try_fold(0usize, |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| Error::Parallel("all-to-all-v send rows overflowed usize".into()))
        })?;
        if usize::try_from(input.dim(0)).ok() != Some(send_rows) {
            return Err(Error::Parallel(format!(
                "all-to-all-v planned {send_rows} send rows but payload has {} rows",
                input.dim(0)
            )));
        }
        let row_elements = input.shape()[1..]
            .iter()
            .try_fold(1usize, |size, dimension| {
                let dimension = usize::try_from(*dimension).map_err(|_| {
                    Error::Parallel("all-to-all-v trailing shape is negative".into())
                })?;
                size.checked_mul(dimension)
                    .ok_or_else(|| Error::Parallel("all-to-all-v row size overflowed usize".into()))
            })?;
        let row_bytes = row_elements
            .checked_mul(input.item_size())
            .ok_or_else(|| Error::Parallel("all-to-all-v row bytes overflowed usize".into()))?;
        let recv_rows = self.recv_counts.iter().try_fold(0usize, |total, count| {
            total
                .checked_add(*count)
                .ok_or_else(|| Error::Parallel("all-to-all-v receive rows overflowed usize".into()))
        })?;
        let useful_sent_bytes = send_rows
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::Parallel("all-to-all-v sent bytes overflowed usize".into()))?;
        let useful_received_bytes = recv_rows.checked_mul(row_bytes).ok_or_else(|| {
            Error::Parallel("all-to-all-v received bytes overflowed usize".into())
        })?;
        // Store-and-forward Ring implementations retain one outgoing and one
        // incoming packet. Native mesh collectives use no larger bound.
        let routing_window = self
            .send_counts
            .iter()
            .chain(&self.recv_counts)
            .copied()
            .max()
            .unwrap_or(0)
            .checked_mul(row_bytes)
            .and_then(|bytes| bytes.checked_mul(2))
            .ok_or_else(|| {
                Error::Parallel("all-to-all-v routing window overflowed usize".into())
            })?;
        let allocation_bound = useful_sent_bytes
            .checked_add(useful_received_bytes)
            .and_then(|bytes| bytes.checked_add(routing_window))
            .ok_or_else(|| {
                Error::Parallel("all-to-all-v allocation bound overflowed usize".into())
            })?;

        let started = Instant::now();
        let received =
            distributed::all_to_all_v(input, &self.send_counts, &self.recv_counts, group, stream)?;
        materialize_timing_phase([&received])?;
        let payload_exchange_time = started.elapsed();
        Ok(ExchangeResult {
            received,
            source_counts: self.recv_counts.clone(),
            statistics: RoutingStatistics {
                sent_routes: send_rows,
                received_routes: recv_rows,
                useful_sent_bytes,
                useful_received_bytes,
                padding_routes: 0,
                padding_bytes: 0,
                payload_allocation_upper_bound_bytes: allocation_bound,
                payload_exchange_time,
                routed_transport: if group.is_logical() {
                    RoutedTransport::Logical
                } else {
                    RoutedTransport::Native
                },
                ..Default::default()
            },
        })
    }
}

/// Result of one planned variable-count all-to-all exchange.
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
    pub global_group_indices: Vec<Array>,
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
        || blocks.global_group_indices.len() != world
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
            || blocks.global_group_indices[destination].shape() != [rows]
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

fn compact_blocks(blocks: &[Array], stream: &Stream) -> Result<Array, Error> {
    let references = blocks.iter().collect::<Vec<_>>();
    Ok(concatenate_axis(&references, 0, stream)?)
}

/// Exchanges sharded-token routes, executes owner-local experts, and returns
/// exact weighted results to their source ranks.
///
/// All payload and metadata exchange uses one [`AllToAllVPlan`]. Collectives
/// are dependency-ordered on every rank, including ranks with zero routes.
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
    let send_counts = blocks
        .hidden
        .iter()
        .map(|block| {
            usize::try_from(block.dim(0))
                .map_err(|_| Error::Parallel("sharded route count is negative".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let total_routes = send_counts.iter().try_fold(0usize, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| Error::Parallel("sharded route count overflowed usize".into()))
    })?;
    let plan = AllToAllVPlan::new(&send_counts, group, stream)?;
    let hidden = plan.exchange(&compact_blocks(&blocks.hidden, stream)?, group, stream)?;
    let compact_global_ids = compact_blocks(&blocks.global_group_indices, stream)?;
    let compact_global_ids = depends([&compact_global_ids], [&hidden.received])?
        .pop()
        .ok_or_else(|| Error::Parallel("all-to-all-v dependency produced no payload".into()))?;
    let global_ids = plan.exchange(&compact_global_ids, group, stream)?;
    let compact_route_indices = compact_blocks(&blocks.original_route_indices, stream)?;
    let compact_route_indices = depends([&compact_route_indices], [&global_ids.received])?
        .pop()
        .ok_or_else(|| Error::Parallel("all-to-all-v dependency produced no payload".into()))?;
    let route_indices = plan.exchange(&compact_route_indices, group, stream)?;
    let compact_weights = compact_blocks(&blocks.weights, stream)?;
    let compact_weights = depends([&compact_weights], [&route_indices.received])?
        .pop()
        .ok_or_else(|| Error::Parallel("all-to-all-v dependency produced no payload".into()))?;
    let weights = plan.exchange(&compact_weights, group, stream)?;
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
        let empty = zeros_dtype(&shape, hidden.received.dtype(), stream)?;
        depends([&empty], [&weights.received])?
            .pop()
            .ok_or_else(|| Error::Parallel("all-to-all-v dependency produced no payload".into()))?
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
        let count = i32::try_from(*count)
            .map_err(|_| Error::Parallel("all-to-all-v receive count exceeds i32".into()))?;
        let end = offset
            .checked_add(count)
            .ok_or_else(|| Error::Parallel("all-to-all-v receive offset exceeds i32".into()))?;
        output_to_source.push(weighted.try_index_device(offset..end, stream)?);
        indices_to_source.push(
            route_indices
                .received
                .try_index_device(offset..end, stream)?,
        );
        offset = end;
    }
    let reverse_plan = plan.reverse();
    let returned_output =
        reverse_plan.exchange(&compact_blocks(&output_to_source, stream)?, group, stream)?;
    let compact_returned_indices = compact_blocks(&indices_to_source, stream)?;
    let compact_returned_indices =
        depends([&compact_returned_indices], [&returned_output.received])?
            .pop()
            .ok_or_else(|| Error::Parallel("all-to-all-v dependency produced no payload".into()))?;
    let returned_indices = reverse_plan.exchange(&compact_returned_indices, group, stream)?;
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
    statistics.accumulate(&plan.consensus_statistics());
    for exchange in [
        hidden,
        global_ids,
        route_indices,
        weights,
        returned_output,
        returned_indices,
    ] {
        statistics.padding_routes = statistics
            .padding_routes
            .checked_add(exchange.statistics.padding_routes)
            .ok_or_else(|| {
                Error::Parallel("all-to-all-v padding row count overflowed usize".into())
            })?;
        statistics.useful_sent_bytes = statistics
            .useful_sent_bytes
            .checked_add(exchange.statistics.useful_sent_bytes)
            .ok_or_else(|| {
                Error::Parallel("all-to-all-v sent byte total overflowed usize".into())
            })?;
        statistics.useful_received_bytes = statistics
            .useful_received_bytes
            .checked_add(exchange.statistics.useful_received_bytes)
            .ok_or_else(|| {
                Error::Parallel("all-to-all-v received byte total overflowed usize".into())
            })?;
        statistics.padding_bytes = statistics
            .padding_bytes
            .checked_add(exchange.statistics.padding_bytes)
            .ok_or_else(|| {
                Error::Parallel("all-to-all-v padding byte total overflowed usize".into())
            })?;
        statistics.payload_allocation_upper_bound_bytes = statistics
            .payload_allocation_upper_bound_bytes
            .checked_add(exchange.statistics.payload_allocation_upper_bound_bytes)
            .ok_or_else(|| {
                Error::Parallel("all-to-all-v allocation bound overflowed usize".into())
            })?;
        if let Some(bytes) = exchange.statistics.temporary_high_water_bytes {
            statistics.temporary_high_water_bytes = Some(
                statistics
                    .temporary_high_water_bytes
                    .map_or(bytes, |current| current.max(bytes)),
            );
        }
        statistics.payload_exchange_time += exchange.statistics.payload_exchange_time;
        statistics.routed_transport = exchange.statistics.routed_transport;
    }
    statistics.total_time = total_started.elapsed();
    Ok(ShardedReturnedRoutes { output, statistics })
}

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
