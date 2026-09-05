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
