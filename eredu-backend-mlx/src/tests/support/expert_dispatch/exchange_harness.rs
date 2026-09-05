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
