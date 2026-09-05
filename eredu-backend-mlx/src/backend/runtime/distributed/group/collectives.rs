use super::handle::record_native_collective_submission;
use super::*;

fn depends_on(value: &Array, dependency: &Array) -> Result<Array> {
    safemlx::transforms::depends([value], [dependency])?
        .pop()
        .ok_or_else(|| Exception::custom("MLX depends returned no output"))
}

pub(super) fn pack_logical_value(
    input: &Array,
    slot: usize,
    native_size: usize,
    stream: &Stream,
) -> Result<Array> {
    let mut shape = Vec::with_capacity(input.ndim() + 1);
    shape.push(
        i32::try_from(native_size)
            .map_err(|_| Exception::custom("native world size does not fit in i32"))?,
    );
    shape.extend_from_slice(input.shape());
    let mut packed = zeros_dtype(&shape, input.dtype(), stream)?;
    packed.try_index_mut_device(
        i32::try_from(slot).map_err(|_| Exception::custom("logical slot does not fit in i32"))?,
        input,
        stream,
    )?;
    Ok(packed)
}

fn logical_pair_peer(group: &Group) -> Option<usize> {
    let logical = group.logical.as_ref()?;
    (logical.global_ranks.len() == 2).then(|| logical.global_ranks[1 - logical.rank])
}

fn native_send(input: &Array, destination: usize, group: &Group, stream: &Stream) -> Result<Array> {
    native::send(input, destination, &group.native, stream)
}

fn native_recv_like(like: &Array, source: usize, group: &Group, stream: &Stream) -> Result<Array> {
    native::recv_like(like, source, &group.native, stream)
}

pub(super) fn logical_direct_exchange(
    input: &Array,
    group: &Group,
    stream: &Stream,
) -> Result<Option<Array>> {
    let Some(peer) = logical_pair_peer(group) else {
        return Ok(None);
    };
    let logical = group
        .logical
        .as_ref()
        .expect("logical pair requires logical group");
    let native_rank = group.native.rank();
    let native_size = group.native.size();
    let direct = (native_rank + 1) % native_size == peer || (peer + 1) % native_size == native_rank;
    let rounds = if direct {
        1
    } else if native_size.is_multiple_of(2) && (native_rank + native_size / 2) % native_size == peer
    {
        if !logical.world_collective_wave {
            return Err(Exception::custom(
                "non-neighbor logical pair requires a consensus-proven world participation wave",
            ));
        }
        native_size / 2
    } else {
        return Ok(None);
    };
    let zero = zeros_dtype(&[], input.dtype(), stream)?;
    let mut exchanged = input.clone();
    for _ in 0..rounds {
        let (destination, source) = if direct {
            (peer, peer)
        } else {
            (
                (native_rank + 1) % native_size,
                (native_rank + native_size - 1) % native_size,
            )
        };
        let sent = native_send(&exchanged, destination, group, stream)?;
        let received = native_recv_like(&exchanged, source, group, stream)?;
        exchanged = received.add(sent.multiply(&zero, stream)?, stream)?;
        safemlx::transforms::async_eval_with_event([&exchanged])?.synchronize()?;
    }
    Ok(Some(exchanged))
}

fn logical_routed_values(
    input: &Array,
    group: &Group,
    stream: &Stream,
) -> Result<Option<Vec<(usize, Array)>>> {
    let Some(routes) = group
        .logical
        .as_ref()
        .and_then(|logical| logical.routes.as_ref())
    else {
        return Ok(None);
    };
    if routes
        .iter()
        .any(|route| route.exchanges.iter().any(Option::is_some))
        && !group
            .logical
            .as_ref()
            .expect("logical routes require a logical group")
            .world_collective_wave
    {
        return Err(Exception::custom(format!(
            "routed logical collective over members {:?} requires a consensus-proven world participation wave",
            group
                .logical
                .as_ref()
                .expect("logical routes require a logical group")
                .global_ranks
        )));
    }
    let zero = zeros_dtype(&[], input.dtype(), stream)?;
    let mut values = Vec::with_capacity(routes.len());
    for route in routes {
        let mut routed = input.clone();
        for peer in route.exchanges.iter().flatten() {
            let sent = native_send(&routed, *peer, group, stream)?;
            let received = native_recv_like(&routed, *peer, group, stream)?;
            routed = received.add(sent.multiply(&zero, stream)?, stream)?;
            safemlx::transforms::async_eval_with_event([&routed])?.synchronize()?;
        }
        values.push((route.source_rank, routed));
    }
    Ok(Some(values))
}

fn logical_all_sum(input: &Array, group: &Group, stream: &Stream) -> Result<Array> {
    if let Some(values) = logical_routed_values(input, group, stream)? {
        let mut values = values.into_iter();
        let (_, mut reduced) = values
            .next()
            .ok_or_else(|| Exception::custom("logical routes cannot be empty"))?;
        for (_, value) in values {
            reduced = reduced.add(value, stream)?;
        }
        return Ok(reduced);
    }
    if let Some(peer) = logical_direct_exchange(input, group, stream)? {
        return input.add(peer, stream);
    }
    let logical = group.logical.as_ref().expect("logical group");
    if !logical.world_collective_wave {
        return Err(Exception::custom(
            "logical subgroup cannot use a world collective without a consensus-proven participation wave",
        ));
    }
    let representative = logical.global_ranks[0];
    let packed = pack_logical_value(input, representative, group.native.size(), stream)?;
    native::all_sum(&packed, &group.native, stream)?.try_index_device(
        i32::try_from(representative)
            .map_err(|_| Exception::custom("logical representative does not fit in i32"))?,
        stream,
    )
}

fn logical_all_gather_stacked(input: &Array, group: &Group, stream: &Stream) -> Result<Array> {
    if let Some(mut values) = logical_routed_values(input, group, stream)? {
        values.sort_unstable_by_key(|(source_rank, _)| *source_rank);
        let values = values
            .into_iter()
            .map(|(_, value)| value)
            .collect::<Vec<_>>();
        return stack_axis(&values, 0, stream);
    }
    if let Some(peer) = logical_direct_exchange(input, group, stream)? {
        return if group.rank() == 0 {
            stack_axis(&[input.clone(), peer], 0, stream)
        } else {
            stack_axis(&[peer, input.clone()], 0, stream)
        };
    }
    let logical = group.logical.as_ref().expect("logical group");
    if !logical.world_collective_wave {
        return Err(Exception::custom(
            "logical subgroup cannot use a world collective without a consensus-proven participation wave",
        ));
    }
    let packed = pack_logical_value(input, group.native.rank(), group.native.size(), stream)?;
    let gathered = native::all_sum(&packed, &group.native, stream)?;
    let indices = logical
        .global_ranks
        .iter()
        .map(|rank| {
            i32::try_from(*rank)
                .map_err(|_| Exception::custom("logical member rank does not fit in i32"))
        })
        .collect::<Result<Vec<_>>>()?;
    let length = i32::try_from(indices.len())
        .map_err(|_| Exception::custom("logical subgroup size does not fit in i32"))?;
    gathered.take_axis(Array::from_slice(&indices, &[length]), 0, stream)
}

fn all_sum_unchecked(input: &Array, group: &Group, stream: &Stream) -> Result<Array> {
    record_native_collective_submission(group);
    match group.logical {
        Some(_) => logical_all_sum(input, group, stream),
        None => native::all_sum(input, &group.native, stream),
    }
}

/// Executes one tensor-carrying sum-based collective under its exact contract.
pub(crate) fn all_sum_for(
    operation: CommunicationOperation,
    input: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_tensor(operation, input, false)?;
    group.validate_expected_output(operation, input.dtype(), input.ndim(), input.size())?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let output = all_sum_unchecked(input, group, stream.as_ref())?;
    group.validate_tensor(operation, &output, true)?;
    if output.shape() != input.shape() {
        return Err(Exception::custom(format!(
            "{operation:?} completed with shape {:?}, expected {:?}",
            output.shape(),
            input.shape()
        )));
    }
    Ok(output)
}

/// Executes one payload-free sum-based collective under its exact contract.
pub(crate) fn payload_free_all_sum_for(
    operation: CommunicationOperation,
    token: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_payload_free(operation)?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let output = all_sum_unchecked(token, group, stream.as_ref())?;
    if output.shape() != token.shape() {
        return Err(Exception::custom(format!(
            "{operation:?} completed with shape {:?}, expected {:?}",
            output.shape(),
            token.shape()
        )));
    }
    Ok(output)
}

/// Sums `input` element-wise across a native or logical group.
pub fn all_sum(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    all_sum_for(CommunicationOperation::AllReduceSum, input, group, stream)
}

/// Gathers `input` from every rank, concatenating along axis zero.
pub(crate) fn all_gather_for(
    operation: CommunicationOperation,
    input: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_tensor(operation, input, false)?;
    let output_elements = input
        .size()
        .checked_mul(group.size())
        .ok_or_else(|| Exception::custom("all-gather output elements overflow usize"))?;
    let output_rank = if input.ndim() == 0 { 1 } else { input.ndim() };
    group.validate_expected_output(operation, input.dtype(), output_rank, output_elements)?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let output = all_gather_unchecked(input, group, stream)?;
    group.validate_tensor(operation, &output, true)?;
    Ok(output)
}

pub(crate) fn all_gather_unchecked(
    input: &Array,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    record_native_collective_submission(group);
    let stream = stream.as_ref();
    let output = if group.logical.is_none() {
        native::all_gather(input, &group.native, stream)?
    } else {
        let stacked = logical_all_gather_stacked(input, group, stream)?;
        if input.ndim() == 0 {
            stacked
        } else {
            let mut shape = input.shape().to_vec();
            shape[0] = shape[0]
                .checked_mul(
                    i32::try_from(group.size())
                        .map_err(|_| Exception::custom("logical group size does not fit in i32"))?,
                )
                .ok_or_else(|| Exception::custom("logical all-gather shape exceeds i32"))?;
            stacked.reshape(&shape, stream)?
        }
    };
    let mut expected = input.shape().to_vec();
    if input.ndim() == 0 {
        expected = vec![i32::try_from(group.size())
            .map_err(|_| Exception::custom("group size does not fit in i32"))?];
    } else {
        expected[0] = expected[0]
            .checked_mul(
                i32::try_from(group.size())
                    .map_err(|_| Exception::custom("group size does not fit in i32"))?,
            )
            .ok_or_else(|| Exception::custom("all-gather output shape exceeds i32"))?;
    }
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "all-gather completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(output)
}

/// Gathers `input` from every rank, concatenating along axis zero.
pub fn all_gather(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    all_gather_for(CommunicationOperation::AllGatherEven, input, group, stream)
}

fn validate_all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
) -> Result<Vec<i32>> {
    if input.ndim() == 0 {
        return Err(Exception::custom(
            "all_to_all_v input must have a leading row dimension",
        ));
    }
    if send_counts.len() != group.size() || recv_counts.len() != group.size() {
        return Err(Exception::custom(format!(
            "all_to_all_v requires {} send counts and receive counts, got {} and {}",
            group.size(),
            send_counts.len(),
            recv_counts.len()
        )));
    }
    let send_rows = send_counts.iter().try_fold(0usize, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| Exception::custom("all_to_all_v send count sum overflowed usize"))
    })?;
    if usize::try_from(input.dim(0)).ok() != Some(send_rows) {
        return Err(Exception::custom(format!(
            "all_to_all_v send count sum {send_rows} does not match input row count {}",
            input.dim(0)
        )));
    }
    if send_counts[group.rank()] != recv_counts[group.rank()] {
        return Err(Exception::custom(format!(
            "all_to_all_v self send count {} does not match self receive count {}",
            send_counts[group.rank()],
            recv_counts[group.rank()]
        )));
    }
    let mut offsets = Vec::with_capacity(send_counts.len());
    let mut offset = 0usize;
    for &count in send_counts {
        offsets.push(
            i32::try_from(offset)
                .map_err(|_| Exception::custom("all_to_all_v row offset exceeds i32"))?,
        );
        offset = offset
            .checked_add(count)
            .ok_or_else(|| Exception::custom("all_to_all_v row offset overflowed usize"))?;
    }
    Ok(offsets)
}

fn concatenate_leading_blocks(
    input: &Array,
    counts: &[usize],
    logical_order: impl Iterator<Item = usize>,
    stream: &Stream,
) -> Result<Array> {
    let mut offsets = Vec::with_capacity(counts.len());
    let mut offset = 0_i32;
    for count in counts {
        offsets.push(offset);
        offset = offset
            .checked_add(
                i32::try_from(*count)
                    .map_err(|_| Exception::custom("all_to_all_v block exceeds i32"))?,
            )
            .ok_or_else(|| Exception::custom("all_to_all_v block offset exceeds i32"))?;
    }
    let blocks = logical_order
        .filter_map(|logical| {
            let count = counts[logical];
            (count > 0).then_some((logical, count))
        })
        .map(|(logical, count)| {
            let end = offsets[logical]
                .checked_add(
                    i32::try_from(count)
                        .map_err(|_| Exception::custom("all_to_all_v block exceeds i32"))?,
                )
                .ok_or_else(|| Exception::custom("all_to_all_v block end exceeds i32"))?;
            input.try_index_device(offsets[logical]..end, stream)
        })
        .collect::<Result<Vec<_>>>()?;
    match blocks.as_slice() {
        [] => {
            let mut shape = input.shape().to_vec();
            shape[0] = 0;
            zeros_dtype(&shape, input.dtype(), stream)
        }
        [block] => Ok(block.clone()),
        blocks => concatenate_axis(&blocks.iter().collect::<Vec<_>>(), 0, stream),
    }
}

fn logical_world_all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
    stream: &Stream,
) -> Result<Array> {
    let logical = group.logical.as_ref().expect("logical group");
    let world_size = group.native.size();
    let mut world_send = vec![0_usize; world_size];
    let mut world_recv = vec![0_usize; world_size];
    for (logical_rank, global_rank) in logical.global_ranks.iter().copied().enumerate() {
        world_send[global_rank] = send_counts[logical_rank];
        world_recv[global_rank] = recv_counts[logical_rank];
    }
    let canonical_order = logical
        .global_ranks
        .windows(2)
        .all(|members| members[0] < members[1]);
    let world_input = if canonical_order {
        input.clone()
    } else {
        concatenate_leading_blocks(
            input,
            send_counts,
            (0..world_size).filter_map(|global_rank| {
                logical
                    .global_ranks
                    .iter()
                    .position(|member| *member == global_rank)
            }),
            stream,
        )?
    };
    let world_output = native::all_to_all_v(
        &world_input,
        &world_send,
        &world_recv,
        &group.native,
        stream,
    )?;
    if canonical_order {
        Ok(world_output)
    } else {
        let world_receive_counts = (0..world_size)
            .map(|global_rank| world_recv[global_rank])
            .collect::<Vec<_>>();
        concatenate_leading_blocks(
            &world_output,
            &world_receive_counts,
            logical.global_ranks.iter().copied(),
            stream,
        )
    }
}

/// Exchanges variable-sized leading-axis blocks across a native or logical group.
pub fn all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let _setup = group.begin_bounded_setup()?;
    group.validate_tensor(CommunicationOperation::VariableAllToAll, input, false)?;
    group.validate_peer_counts(
        CommunicationOperation::VariableAllToAll,
        send_counts,
        recv_counts,
    )?;
    let expected_rows = recv_counts.iter().try_fold(0usize, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| Exception::custom("all_to_all_v receive count sum overflowed usize"))
    })?;
    let trailing_elements = input.shape()[1..]
        .iter()
        .try_fold(1usize, |total, dimension| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| Exception::custom("all_to_all_v input has a negative dimension"))?;
            total
                .checked_mul(dimension)
                .ok_or_else(|| Exception::custom("all_to_all_v output elements overflow usize"))
        })?;
    let output_elements = expected_rows
        .checked_mul(trailing_elements)
        .ok_or_else(|| Exception::custom("all_to_all_v output elements overflow usize"))?;
    group.validate_expected_output(
        CommunicationOperation::VariableAllToAll,
        input.dtype(),
        input.ndim(),
        output_elements,
    )?;
    let offsets = validate_all_to_all_v(input, send_counts, recv_counts, group)?;
    if let Some(setup) = &_setup {
        setup.check()?;
    }
    let stream = stream.as_ref();
    record_native_collective_submission(group);
    if group.logical.is_none() {
        let output = native::all_to_all_v(input, send_counts, recv_counts, &group.native, stream)?;
        validate_all_to_all_output(&output, input, expected_rows, group)?;
        return Ok(output);
    }
    if group.size() == 1 {
        let output = input.clone();
        validate_all_to_all_output(&output, input, expected_rows, group)?;
        return Ok(output);
    }
    let logical = group.logical.as_ref().expect("logical group");
    if logical.world_collective_wave {
        let output = logical_world_all_to_all_v(input, send_counts, recv_counts, group, stream)?;
        validate_all_to_all_output(&output, input, expected_rows, group)?;
        return Ok(output);
    }
    let routes = if let Some(routes) = &logical.routes {
        if routes
            .iter()
            .any(|route| route.exchanges.iter().any(Option::is_some))
            && !logical.world_collective_wave
        {
            return Err(Exception::custom(
                "routed logical exchange requires a consensus-proven world participation wave",
            ));
        }
        routes
            .iter()
            .map(|route| (route.source_rank, route.exchanges.clone()))
            .collect::<Vec<_>>()
    } else if logical.global_ranks.len() == 2 {
        let peer = logical.global_ranks[1 - logical.rank];
        let rank = group.native.rank();
        let size = group.native.size();
        if (rank + 1) % size != peer && (peer + 1) % size != rank {
            return Err(Exception::custom(
                "all_to_all_v logical pair has no topology-planned neighbor route",
            ));
        }
        vec![
            (logical.rank, Vec::new()),
            (1 - logical.rank, vec![Some(peer)]),
        ]
    } else {
        return Err(Exception::custom(
            "all_to_all_v logical subgroups larger than two require topology-planned neighbor routes",
        ));
    };
    let mut received = Vec::with_capacity(routes.len());
    for (source_rank, exchanges) in routes {
        let shift =
            (logical.rank + logical.global_ranks.len() - source_rank) % logical.global_ranks.len();
        let destination = (logical.rank + shift) % logical.global_ranks.len();
        let end = offsets[destination]
            .checked_add(
                i32::try_from(send_counts[destination]).map_err(|_| {
                    Exception::custom("all_to_all_v logical send count exceeds i32")
                })?,
            )
            .ok_or_else(|| Exception::custom("all_to_all_v logical slice end overflowed i32"))?;
        let mut routed = input.try_index_device(offsets[destination]..end, stream)?;
        for peer in exchanges.into_iter().flatten() {
            let count = Array::from_slice(&[i64::from(routed.dim(0))], &[1]).copy(stream)?;
            let sent_count = native_send(&count, peer, group, stream)?;
            let received_count = native_recv_like(&count, peer, group, stream)?;
            let received_count = depends_on(&received_count, &sent_count)?;
            safemlx::transforms::async_eval_with_event([&received_count])?.synchronize()?;
            let incoming_rows = i32::try_from(received_count.evaluated()?.as_slice::<i64>()[0])
                .map_err(|_| {
                    Exception::custom("all_to_all_v logical routed row count is outside i32")
                })?;
            if incoming_rows < 0 {
                return Err(Exception::custom(
                    "all_to_all_v logical routed row count is negative",
                ));
            }
            let mut shape = input.shape().to_vec();
            shape[0] = incoming_rows;
            let empty = zeros_dtype(&shape, input.dtype(), stream)?;
            let sent = native_send(&routed, peer, group, stream)?;
            let incoming = native_recv_like(&empty, peer, group, stream)?;
            routed = depends_on(&incoming, &sent)?;
        }
        if usize::try_from(routed.dim(0)).ok() != Some(recv_counts[source_rank]) {
            return Err(Exception::custom(format!(
                "all_to_all_v logical route from source {source_rank} produced {} rows, expected {}",
                routed.dim(0), recv_counts[source_rank]
            )));
        }
        received.push((source_rank, routed));
    }
    received.sort_unstable_by_key(|(source_rank, _)| *source_rank);
    let arrays = received.iter().map(|(_, array)| array).collect::<Vec<_>>();
    let output = concatenate_axis(&arrays, 0, stream)?;
    validate_all_to_all_output(&output, input, expected_rows, group)?;
    Ok(output)
}

fn validate_all_to_all_output(
    output: &Array,
    input: &Array,
    expected_rows: usize,
    group: &Group,
) -> Result<()> {
    group.validate_tensor(CommunicationOperation::VariableAllToAll, output, true)?;
    let mut expected = input.shape().to_vec();
    expected[0] = i32::try_from(expected_rows)
        .map_err(|_| Exception::custom("all_to_all_v output rows exceed i32"))?;
    if output.shape() != expected {
        return Err(Exception::custom(format!(
            "VariableAllToAll completed with shape {:?}, expected {expected:?}",
            output.shape()
        )));
    }
    Ok(())
}
