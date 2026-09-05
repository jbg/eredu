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
