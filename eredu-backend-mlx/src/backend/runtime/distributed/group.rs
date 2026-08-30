//! Backend-owned logical groups layered over native MLX communication groups.

use safemlx::{
    distributed as native,
    error::{Exception, Result},
    ops::{
        concatenate_axis, indexing::TryIndexMutOp, indexing::TryIndexOp, stack_axis, zeros_dtype,
    },
    Array, Dtype, Stream,
};

/// A native MLX group or a backend-owned logical subgroup of one native world.
pub struct Group {
    native: native::Group,
    logical: Option<LogicalSubgroup>,
}

#[derive(Debug)]
struct LogicalSubgroup {
    global_ranks: Vec<usize>,
    rank: usize,
    routes: Option<Vec<LogicalRoute>>,
}

#[derive(Debug)]
struct LogicalRoute {
    source_rank: usize,
    exchanges: Vec<Option<usize>>,
}

impl Group {
    /// Wraps a native MLX group without changing its rank semantics.
    pub fn native(group: &native::Group) -> Self {
        Self {
            native: group.clone(),
            logical: None,
        }
    }

    #[cfg(test)]
    pub(crate) const fn native_group(&self) -> &native::Group {
        &self.native
    }

    /// Returns this process's rank in this native or logical group.
    pub fn rank(&self) -> usize {
        self.logical
            .as_ref()
            .map_or_else(|| self.native.rank(), |logical| logical.rank)
    }

    /// Returns this native or logical group's size.
    pub fn size(&self) -> usize {
        self.logical
            .as_ref()
            .map_or_else(|| self.native.size(), |logical| logical.global_ranks.len())
    }

    /// Returns whether this is a backend-routed logical subgroup.
    pub fn is_logical(&self) -> bool {
        self.logical.is_some()
    }

    /// Attempts a backend-native split.
    pub fn split(&self, color: i32, key: Option<i32>) -> Result<Self> {
        if self.logical.is_some() {
            return Err(Exception::custom(
                "backend-native splitting of a logical subgroup is unsupported",
            ));
        }
        Ok(Self {
            native: self.native.split(color, key)?,
            logical: None,
        })
    }

    /// Creates a logical subgroup over the same native world group.
    pub fn logical_subgroup(&self, global_ranks: &[usize]) -> Result<Self> {
        if self.logical.is_some() {
            return Err(Exception::custom(
                "logical subgroups must be derived from a native world group",
            ));
        }
        if global_ranks.is_empty() {
            return Err(Exception::custom("logical subgroup cannot be empty"));
        }
        let native_size = self.native.size();
        let mut seen = vec![false; native_size];
        for &rank in global_ranks {
            if rank >= native_size {
                return Err(Exception::custom(format!(
                    "logical subgroup rank {rank} is outside native world size {native_size}"
                )));
            }
            if std::mem::replace(&mut seen[rank], true) {
                return Err(Exception::custom(format!(
                    "logical subgroup repeats global rank {rank}"
                )));
            }
        }
        let native_rank = self.native.rank();
        let rank = global_ranks
            .iter()
            .position(|rank| *rank == native_rank)
            .ok_or_else(|| {
                Exception::custom(format!(
                    "native rank {native_rank} is not a member of logical subgroup {global_ranks:?}"
                ))
            })?;
        Ok(Self {
            native: self.native.clone(),
            logical: Some(LogicalSubgroup {
                global_ranks: global_ranks.to_vec(),
                rank,
                routes: None,
            }),
        })
    }

    /// Creates a logical subgroup with topology-planned native-world routes.
    pub fn logical_subgroup_with_routes(
        &self,
        global_ranks: &[usize],
        routes: Vec<(usize, Vec<Option<usize>>)>,
    ) -> Result<Self> {
        let mut group = self.logical_subgroup(global_ranks)?;
        let native_rank = group.native.rank();
        let native_size = group.native.size();
        let mut seen = vec![false; global_ranks.len()];
        let routes = routes
            .into_iter()
            .map(|(source_rank, exchanges)| {
                if source_rank >= global_ranks.len()
                    || std::mem::replace(&mut seen[source_rank], true)
                {
                    return Err(Exception::custom(format!(
                        "logical route source rank {source_rank} is missing or repeated for subgroup size {}",
                        global_ranks.len()
                    )));
                }
                for peer in exchanges.iter().flatten() {
                    if *peer >= native_size
                        || !((native_rank + 1) % native_size == *peer
                            || (*peer + 1) % native_size == native_rank)
                    {
                        return Err(Exception::custom(format!(
                            "logical route from native rank {native_rank} uses non-neighbor peer {peer}"
                        )));
                    }
                }
                Ok(LogicalRoute {
                    source_rank,
                    exchanges,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if seen.iter().any(|present| !present) {
            return Err(Exception::custom(
                "logical routes do not cover every subgroup source rank",
            ));
        }
        group.logical.as_mut().expect("logical subgroup").routes = Some(routes);
        Ok(group)
    }
}

impl std::fmt::Debug for Group {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Group")
            .field("rank", &self.rank())
            .field("size", &self.size())
            .field("logical", &self.is_logical())
            .finish()
    }
}

fn depends_on(value: &Array, dependency: &Array) -> Result<Array> {
    safemlx::transforms::depends([value], [dependency])?
        .pop()
        .ok_or_else(|| Exception::custom("MLX depends returned no output"))
}

fn pack_logical_value(
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

fn logical_direct_exchange(input: &Array, group: &Group, stream: &Stream) -> Result<Option<Array>> {
    let Some(peer) = logical_pair_peer(group) else {
        return Ok(None);
    };
    let native_rank = group.native.rank();
    let native_size = group.native.size();
    let direct = (native_rank + 1) % native_size == peer || (peer + 1) % native_size == native_rank;
    let rounds = if direct {
        1
    } else if native_size.is_multiple_of(2) && (native_rank + native_size / 2) % native_size == peer
    {
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

/// Sums `input` element-wise across a native or logical group.
pub fn all_sum(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    match group.logical {
        Some(_) => logical_all_sum(input, group, stream.as_ref()),
        None => native::all_sum(input, &group.native, stream),
    }
}

/// Gathers `input` from every rank, concatenating along axis zero.
pub fn all_gather(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    if group.logical.is_none() {
        return native::all_gather(input, &group.native, stream);
    }
    let stacked = logical_all_gather_stacked(input, group, stream.as_ref())?;
    if input.ndim() == 0 {
        return Ok(stacked);
    }
    let mut shape = input.shape().to_vec();
    shape[0] = shape[0]
        .checked_mul(
            i32::try_from(group.size())
                .map_err(|_| Exception::custom("logical group size does not fit in i32"))?,
        )
        .ok_or_else(|| Exception::custom("logical all-gather shape exceeds i32"))?;
    stacked.reshape(&shape, stream)
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

/// Exchanges variable-sized leading-axis blocks across a native or logical group.
pub fn all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    if group.logical.is_none() {
        return native::all_to_all_v(input, send_counts, recv_counts, &group.native, stream);
    }
    let offsets = validate_all_to_all_v(input, send_counts, recv_counts, group)?;
    let stream = stream.as_ref();
    if group.size() == 1 {
        return Ok(input.clone());
    }
    let logical = group.logical.as_ref().expect("logical group");
    let routes = if let Some(routes) = &logical.routes {
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
    concatenate_axis(&arrays, 0, stream)
}

fn checked_peer(peer: usize, group: &Group, role: &str) -> Result<()> {
    if group.size() == 1 {
        return Err(Exception::custom(format!(
            "cannot use a {role} rank with a singleton distributed group"
        )));
    }
    if peer >= group.size() {
        return Err(Exception::custom(format!(
            "invalid {role} rank {peer} for distributed group of size {}",
            group.size()
        )));
    }
    Ok(())
}

/// Sends `input` to a rank in a native or logical group.
pub fn send(
    input: &Array,
    destination: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    checked_peer(destination, group, "destination")?;
    if group.logical.is_none() {
        return native::send(input, destination, &group.native, stream);
    }
    let stream = stream.as_ref();
    if let Some(exchanged) = logical_direct_exchange(input, group, stream)? {
        return Ok(exchanged);
    }
    let logical = group.logical.as_ref().expect("logical group");
    let source_global = logical.global_ranks[logical.rank];
    let packed = pack_logical_value(input, source_global, group.native.size(), stream)?;
    native::all_sum(&packed, &group.native, stream)?.try_index_device(
        i32::try_from(source_global)
            .map_err(|_| Exception::custom("logical source rank does not fit in i32"))?,
        stream,
    )
}

/// Receives an array from a rank in a native or logical group.
pub fn recv(
    shape: &[i32],
    dtype: Dtype,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    checked_peer(source, group, "source")?;
    if group.logical.is_none() {
        return native::recv(shape, dtype, source, &group.native, stream);
    }
    let stream = stream.as_ref();
    let empty = zeros_dtype(shape, dtype, stream)?;
    recv_like(&empty, source, group, stream)
}

/// Receives from a rank using `like` for shape and dtype.
fn recv_like(
    like: &Array,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    checked_peer(source, group, "source")?;
    if group.logical.is_none() {
        return native::recv_like(like, source, &group.native, stream);
    }
    let stream = stream.as_ref();
    if let Some(exchanged) = logical_direct_exchange(like, group, stream)? {
        return Ok(exchanged);
    }
    let logical = group.logical.as_ref().expect("logical group");
    let source_global = logical.global_ranks[source];
    let packed = pack_logical_value(like, source_global, group.native.size(), stream)?;
    native::all_sum(&packed, &group.native, stream)?.try_index_device(
        i32::try_from(source_global)
            .map_err(|_| Exception::custom("logical source rank does not fit in i32"))?,
        stream,
    )
}
