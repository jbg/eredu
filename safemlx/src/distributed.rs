//! Distributed communication groups and operations.
//!
//! MLX caches initialized backends process-wide. [`init`] preserves that
//! behavior. With `strict == false`, failure to establish the requested
//! backend returns a usable size-one group; collectives on that group return
//! their input unchanged. Point-to-point operations still reject a singleton.
//!
//! Backend setup follows MLX 0.32:
//!
//! - Ring uses `MLX_RANK` and a JSON file named by `MLX_HOSTFILE`.
//! - MPI dynamically loads Open MPI; `MLX_MPI_LIBNAME` can override its library.
//! - JACCL uses `MLX_RANK`, `MLX_IBV_DEVICES`, and
//!   `MLX_JACCL_COORDINATOR` (or their `JACCL_*` aliases) on supported macOS
//!   systems.
//! - NCCL uses `NCCL_HOST_IP`, `NCCL_PORT`, `MLX_RANK`, and
//!   `MLX_WORLD_SIZE` in NCCL-enabled builds.
//!
//! Operations are lazy, just like other MLX array operations. Evaluate their
//! returned arrays at the synchronization points required by the application.

use std::{ffi::c_char, marker::PhantomData, rc::Rc, str::FromStr};

use crate::{
    error::{Exception, Result},
    ops::indexing::{TryIndexMutOp, TryIndexOp},
    ops::{concatenate_axis, stack_axis, zeros_dtype},
    utils::{guard::Guarded, runtime_lock, SUCCESS},
    Array, Device, DeviceType, Dtype, Stream,
};

/// A distributed communication backend supported by MLX.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Let MLX choose an available backend.
    Any,
    /// MLX's TCP ring backend.
    Ring,
    /// Open MPI, loaded dynamically by MLX.
    Mpi,
    /// JACCL over RDMA on supported Apple systems.
    Jaccl,
    /// NCCL in CUDA/NCCL-enabled builds.
    Nccl,
}

impl Backend {
    /// Return the backend name accepted by MLX.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::Ring => "ring",
            Self::Mpi => "mpi",
            Self::Jaccl => "jaccl",
            Self::Nccl => "nccl",
        }
    }

    fn as_c_ptr(self) -> *const c_char {
        match self {
            // A null backend selects MLX's default/"any" behavior.
            Self::Any => std::ptr::null(),
            Self::Ring => c"ring".as_ptr(),
            Self::Mpi => c"mpi".as_ptr(),
            Self::Jaccl => c"jaccl".as_ptr(),
            Self::Nccl => c"nccl".as_ptr(),
        }
    }
}

impl FromStr for Backend {
    type Err = Exception;

    fn from_str(value: &str) -> Result<Self> {
        if value.as_bytes().contains(&0) {
            return Err(Exception::custom(
                "distributed backend name contains an interior NUL byte",
            ));
        }

        match value {
            "any" => Ok(Self::Any),
            "ring" => Ok(Self::Ring),
            "mpi" => Ok(Self::Mpi),
            "jaccl" => Ok(Self::Jaccl),
            "nccl" => Ok(Self::Nccl),
            _ => Err(Exception::custom(format!(
                "unknown distributed backend {value:?}; expected any, ring, mpi, jaccl, or nccl"
            ))),
        }
    }
}

impl TryFrom<&str> for Backend {
    type Error = Exception;

    fn try_from(value: &str) -> Result<Self> {
        value.parse()
    }
}

impl TryFrom<String> for Backend {
    type Error = Exception;

    fn try_from(value: String) -> Result<Self> {
        value.parse()
    }
}

impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// An owned MLX distributed group or a logical view of an owned world group.
///
/// The final related group frees the native handle on drop. The type is
/// intentionally neither `Clone`, `Send`, nor `Sync`: not every communication
/// backend documents cross-thread group access.
pub struct Group {
    native: Rc<NativeGroup>,
    logical: Option<LogicalSubgroup>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

struct NativeGroup {
    c_group: safemlx_sys::mlx_distributed_group,
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

impl Drop for NativeGroup {
    fn drop(&mut self) {
        let _guard = runtime_lock::enter();
        // SAFETY: this is the sole owner of the native group handle.
        let status = unsafe { safemlx_sys::mlx_distributed_group_free(self.c_group) };
        debug_assert_eq!(status, SUCCESS);
    }
}

impl Group {
    pub(crate) fn from_owned_ptr(c_group: safemlx_sys::mlx_distributed_group) -> Self {
        Self {
            native: Rc::new(NativeGroup { c_group }),
            logical: None,
            _not_send_or_sync: PhantomData,
        }
    }

    fn native_rank(&self) -> usize {
        let _guard = runtime_lock::enter();
        // SAFETY: `self.native` owns a successfully initialized, non-empty group.
        let rank = unsafe { safemlx_sys::mlx_distributed_group_rank(self.native.c_group) };
        usize::try_from(rank).expect("MLX returned a negative distributed rank")
    }

    fn native_size(&self) -> usize {
        let _guard = runtime_lock::enter();
        // SAFETY: `self.native` owns a successfully initialized, non-empty group.
        let size = unsafe { safemlx_sys::mlx_distributed_group_size(self.native.c_group) };
        usize::try_from(size).expect("MLX returned a negative distributed group size")
    }

    /// Initialize and own the process's group for `backend`.
    ///
    /// If `strict` is `false` and MLX cannot initialize the backend, the result
    /// is a size-one group. If `strict` is `true`, initialization instead
    /// returns the MLX error.
    pub fn init(strict: bool, backend: Backend) -> Result<Self> {
        init(strict, backend)
    }

    /// Return this process's zero-based rank within the group.
    pub fn rank(&self) -> usize {
        self.logical
            .as_ref()
            .map_or_else(|| self.native_rank(), |logical| logical.rank)
    }

    /// Return the number of processes in the group.
    pub fn size(&self) -> usize {
        self.logical
            .as_ref()
            .map_or_else(|| self.native_size(), |logical| logical.global_ranks.len())
    }

    /// Split the group by `color`, optionally ordering new ranks by `key`.
    ///
    /// A missing or negative key asks MLX to use the current group rank. Backend
    /// support varies: MLX 0.32 supports splitting with MPI and NCCL, while its
    /// singleton, Ring, and JACCL groups return an error.
    pub fn split(&self, color: i32, key: Option<i32>) -> Result<Self> {
        if self.logical.is_some() {
            return Err(Exception::custom(
                "backend-native splitting of a logical subgroup is unsupported",
            ));
        }
        let _guard = runtime_lock::enter();
        Self::try_from_op(|res| {
            // SAFETY: `res` is an initialized output guard and `self.c_group`
            // remains alive for the duration of this call.
            unsafe {
                safemlx_sys::mlx_distributed_group_split(
                    res,
                    self.native.c_group,
                    color,
                    key.unwrap_or(-1),
                )
            }
        })
    }

    /// Creates a logical subgroup over the same native world group.
    ///
    /// Logical subgroups are the correctness fallback for backends such as
    /// Ring and JACCL which do not implement native group splitting. Two-rank
    /// neighbor or antipodal groups use ordered peer exchange. Callers with an
    /// authoritative topology can instead install neighbor routes through
    /// [`Self::logical_subgroup_with_routes`] so arbitrary-degree independent
    /// subgroups progress without world-wide collective participation. Other
    /// layouts pack values by subgroup identity into native-world collectives.
    pub fn logical_subgroup(&self, global_ranks: &[usize]) -> Result<Self> {
        if self.logical.is_some() {
            return Err(Exception::custom(
                "logical subgroups must be derived from a native world group",
            ));
        }
        if global_ranks.is_empty() {
            return Err(Exception::custom("logical subgroup cannot be empty"));
        }
        let native_size = self.native_size();
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
        let native_rank = self.native_rank();
        let rank = global_ranks
            .iter()
            .position(|rank| *rank == native_rank)
            .ok_or_else(|| {
                Exception::custom(format!(
                    "native rank {native_rank} is not a member of logical subgroup {global_ranks:?}"
                ))
            })?;
        Ok(Self {
            native: Rc::clone(&self.native),
            logical: Some(LogicalSubgroup {
                global_ranks: global_ranks.to_vec(),
                rank,
                routes: None,
            }),
            _not_send_or_sync: PhantomData,
        })
    }

    /// Creates a logical subgroup with topology-planned neighbor routes.
    ///
    /// Each route identifies the subgroup rank whose original value arrives
    /// locally after the listed rounds. A round either exchanges the current
    /// value with one native-world neighbor or remains idle. This lets Ring
    /// execute independent stage-local collectives without requiring every
    /// world rank to enter a packed fallback collective.
    pub fn logical_subgroup_with_routes(
        &self,
        global_ranks: &[usize],
        routes: Vec<(usize, Vec<Option<usize>>)>,
    ) -> Result<Self> {
        let mut group = self.logical_subgroup(global_ranks)?;
        let native_rank = group.native_rank();
        let native_size = group.native_size();
        let mut seen = vec![false; global_ranks.len()];
        let routes = routes
            .into_iter()
            .map(|(source_rank, exchanges)| {
                if source_rank >= global_ranks.len() || std::mem::replace(&mut seen[source_rank], true)
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Group")
            .field("rank", &self.rank())
            .field("size", &self.size())
            .finish()
    }
}

/// Check whether MLX was built with support for `backend`.
///
/// Availability does not imply that the backend's required environment or
/// communication peers are configured; use [`init`] to establish a group.
pub fn is_available(backend: Backend) -> bool {
    let _guard = runtime_lock::enter();
    // SAFETY: the pointer is null or a static NUL-terminated backend name.
    unsafe { safemlx_sys::mlx_distributed_is_available(backend.as_c_ptr()) }
}

/// Initialize and own the process's group for `backend`.
///
/// MLX caches backend initialization process-wide. With `strict == false`, a
/// backend that cannot be established produces a usable size-one group.
pub fn init(strict: bool, backend: Backend) -> Result<Group> {
    let _guard = runtime_lock::enter();
    Group::try_from_op(|res| {
        // SAFETY: `res` is owned by the output guard and the backend pointer is
        // null or a static NUL-terminated string.
        unsafe { safemlx_sys::mlx_distributed_init(res, strict, backend.as_c_ptr()) }
    })
}

fn collective(
    input: &Array,
    group: &Group,
    stream: &Stream,
    op: unsafe extern "C" fn(
        *mut safemlx_sys::mlx_array,
        safemlx_sys::mlx_array,
        safemlx_sys::mlx_distributed_group,
        safemlx_sys::mlx_stream,
    ) -> i32,
) -> Result<Array> {
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all borrowed handles remain alive for the call and `res` is
        // an owned array output guard.
        unsafe { op(res, input.as_ptr(), group.native.c_group, stream.as_ptr()) }
    })
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
    if logical.global_ranks.len() != 2 {
        return None;
    }
    Some(logical.global_ranks[1 - logical.rank])
}

fn native_send(input: &Array, destination: usize, group: &Group, stream: &Stream) -> Result<Array> {
    let destination = i32::try_from(destination)
        .map_err(|_| Exception::custom("destination rank does not fit in i32"))?;
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all input handles remain alive and `destination` is a native
        // group rank validated by the caller.
        unsafe {
            safemlx_sys::mlx_distributed_send(
                res,
                input.as_ptr(),
                destination,
                group.native.c_group,
                stream.as_ptr(),
            )
        }
    })
}

fn native_recv_like(like: &Array, source: usize, group: &Group, stream: &Stream) -> Result<Array> {
    let source =
        i32::try_from(source).map_err(|_| Exception::custom("source rank does not fit in i32"))?;
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all borrowed handles remain alive and `source` is a native
        // group rank validated by the caller.
        unsafe {
            safemlx_sys::mlx_distributed_recv_like(
                res,
                like.as_ptr(),
                source,
                group.native.c_group,
                stream.as_ptr(),
            )
        }
    })
}

fn logical_direct_exchange(input: &Array, group: &Group, stream: &Stream) -> Result<Option<Array>> {
    let Some(peer) = logical_pair_peer(group) else {
        return Ok(None);
    };
    let native_rank = group.native_rank();
    let native_size = group.native_size();
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
        // Preserve the lazy send as an explicit dependency of the returned
        // value. Without it, evaluating only the receive side can leave both
        // peers waiting.
        exchanged = received.add(sent.multiply(&zero, stream)?, stream)?;
        crate::transforms::async_eval_with_event([&exchanged])?.synchronize()?;
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
        for peer in &route.exchanges {
            let Some(peer) = peer else {
                continue;
            };
            let sent = native_send(&routed, *peer, group, stream)?;
            let received = native_recv_like(&routed, *peer, group, stream)?;
            routed = received.add(sent.multiply(&zero, stream)?, stream)?;
            crate::transforms::async_eval_with_event([&routed])?.synchronize()?;
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
    let logical = group
        .logical
        .as_ref()
        .expect("logical collective requires logical membership");
    let representative = logical.global_ranks[0];
    let packed = pack_logical_value(input, representative, group.native_size(), stream)?;
    let reduced = collective(&packed, group, stream, safemlx_sys::mlx_distributed_all_sum)?;
    reduced.try_index_device(
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
        let values = if group.rank() == 0 {
            [input.clone(), peer]
        } else {
            [peer, input.clone()]
        };
        return stack_axis(&values, 0, stream);
    }
    let logical = group
        .logical
        .as_ref()
        .expect("logical collective requires logical membership");
    let native_rank = group.native_rank();
    let packed = pack_logical_value(input, native_rank, group.native_size(), stream)?;
    let gathered = collective(&packed, group, stream, safemlx_sys::mlx_distributed_all_sum)?;
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

/// Sum `input` element-wise across `group` on `stream`.
pub fn all_sum(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    if group.logical.is_some() {
        return logical_all_sum(input, group, stream.as_ref());
    }
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_sum,
    )
}

/// Take the element-wise maximum of `input` across `group` on `stream`.
pub fn all_max(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    if group.logical.is_some() {
        return logical_all_gather_stacked(input, group, stream.as_ref())?.max_axis(
            0,
            Some(false),
            stream,
        );
    }
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_max,
    )
}

/// Take the element-wise minimum of `input` across `group` on `stream`.
pub fn all_min(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    if group.logical.is_some() {
        return logical_all_gather_stacked(input, group, stream.as_ref())?.min_axis(
            0,
            Some(false),
            stream,
        );
    }
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_min,
    )
}

/// Gather `input` from every rank, concatenating along axis zero.
///
/// Scalar inputs become a one-dimensional result for non-singleton groups.
pub fn all_gather(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    if group.size() > 1 {
        if let Some(&first_dim) = input.shape().first() {
            let group_size = i32::try_from(group.size())
                .map_err(|_| Exception::custom("distributed group size does not fit in i32"))?;
            first_dim.checked_mul(group_size).ok_or_else(|| {
                Exception::custom("all-gather output's first dimension exceeds i32")
            })?;
        }
    }
    if group.logical.is_some() {
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
        return stacked.reshape(&shape, stream);
    }
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_gather,
    )
}

/// Gather equal-shaped shards along an arbitrary existing tensor axis.
///
/// MLX's primitive all-gather always concatenates complete rank payloads on
/// axis zero. This helper recovers those payloads and concatenates them along
/// `axis` without forcing evaluation.
pub fn all_gather_axis(
    input: &Array,
    axis: i32,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Exception::custom(
            "axis all-gather requires a non-scalar input",
        ));
    }
    let ndim_i32 =
        i32::try_from(ndim).map_err(|_| Exception::custom("input rank does not fit in i32"))?;
    let axis = if axis < 0 { axis + ndim_i32 } else { axis };
    if !(0..ndim_i32).contains(&axis) {
        return Err(Exception::custom(format!(
            "all-gather axis {axis} is outside input rank {ndim}"
        )));
    }
    if axis == 0 {
        return all_gather(input, group, stream);
    }
    // MLX's primitive consumes each rank's complete payload and concatenates
    // those payloads on axis zero. Gather before rearranging, recover each
    // rank's original tensor by slicing axis zero, then concatenate the rank
    // tensors along the requested axis. Passing a moved (strided) view to the
    // collective gathers its underlying storage order for multi-row tensors.
    let gathered = all_gather(input, group, stream)?;
    let rank_height = input.shape()[0];
    let mut shards = Vec::with_capacity(group.size());
    for rank in 0..group.size() {
        let start = i32::try_from(rank)
            .ok()
            .and_then(|rank| rank.checked_mul(rank_height))
            .ok_or_else(|| Exception::custom("gathered rank offset exceeds i32"))?;
        let end = start
            .checked_add(rank_height)
            .ok_or_else(|| Exception::custom("gathered rank end exceeds i32"))?;
        shards.push(gathered.try_index_device(start..end, stream)?);
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    concatenate_axis(&shard_refs, axis, stream)
}

/// Gather unequal contiguous shards along an arbitrary tensor axis.
///
/// `widths` contains the logical width contributed by every rank, in group
/// rank order. Local shards are padded to the largest width for the primitive
/// all-gather, then padding is removed before the original axis is restored.
/// This is useful for balanced vocabulary partitions when the vocabulary size
/// is not divisible by the tensor-parallel degree.
pub fn all_gather_uneven_axis(
    input: &Array,
    axis: i32,
    widths: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    if widths.len() != group.size() {
        return Err(Exception::custom(format!(
            "uneven all-gather received {} widths for group size {}",
            widths.len(),
            group.size()
        )));
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Exception::custom(
            "uneven axis all-gather requires a non-scalar input",
        ));
    }
    let ndim_i32 =
        i32::try_from(ndim).map_err(|_| Exception::custom("input rank does not fit in i32"))?;
    let axis = if axis < 0 { axis + ndim_i32 } else { axis };
    if !(0..ndim_i32).contains(&axis) {
        return Err(Exception::custom(format!(
            "uneven all-gather axis {axis} is outside input rank {ndim}"
        )));
    }
    let rank = group.rank();
    let local_width = usize::try_from(input.shape()[axis as usize])
        .map_err(|_| Exception::custom("input shape contains a negative dimension"))?;
    if local_width != widths[rank] {
        return Err(Exception::custom(format!(
            "rank {rank} local width {local_width} does not match declared width {}",
            widths[rank]
        )));
    }
    let max_width = widths.iter().copied().max().unwrap_or(0);
    if max_width == 0 {
        return Err(Exception::custom(
            "uneven all-gather requires at least one non-empty shard",
        ));
    }

    let padded = if local_width == max_width {
        input.clone()
    } else {
        let mut padding_shape = input.shape().to_vec();
        padding_shape[axis as usize] = i32::try_from(max_width - local_width)
            .map_err(|_| Exception::custom("padding width does not fit in i32"))?;
        let padding = zeros_dtype(&padding_shape, input.dtype(), stream)?;
        concatenate_axis(&[input, &padding], axis, stream)?
    };
    let gathered = all_gather_axis(&padded, axis, group, stream)?;
    let group_size = i32::try_from(group.size())
        .map_err(|_| Exception::custom("distributed group size does not fit in i32"))?;
    let padded_shards = gathered.split(group_size, Some(axis), stream)?;
    let mut shards = Vec::with_capacity(widths.len());
    for (padded, &width) in padded_shards.into_iter().zip(widths) {
        if width == max_width {
            shards.push(padded);
        } else {
            let width = i32::try_from(width)
                .map_err(|_| Exception::custom("shard width does not fit in i32"))?;
            shards.push(
                padded
                    .split_axis(&[width], Some(axis), stream)?
                    .into_iter()
                    .next()
                    .expect("one split index produces a leading shard"),
            );
        }
    }
    let shard_refs = shards.iter().collect::<Vec<_>>();
    concatenate_axis(&shard_refs, axis, stream)
}

/// Sum across `group` and scatter equal axis-zero chunks to each rank.
pub fn sum_scatter(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    let group_size = group.size();
    if group_size > 1 {
        let first_dim = input
            .shape()
            .first()
            .copied()
            .ok_or_else(|| Exception::custom("sum-scatter requires a non-scalar input"))?;
        let first_dim = usize::try_from(first_dim)
            .map_err(|_| Exception::custom("sum-scatter input shape contains a negative size"))?;
        if first_dim % group_size != 0 {
            return Err(Exception::custom(format!(
                "sum-scatter input's first dimension ({first_dim}) is not divisible by group size ({group_size})"
            )));
        }
    }
    if group.logical.is_some() {
        let reduced = logical_all_sum(input, group, stream.as_ref())?;
        return reduced
            .split(
                i32::try_from(group_size)
                    .map_err(|_| Exception::custom("logical group size does not fit in i32"))?,
                Some(0),
                stream,
            )?
            .into_iter()
            .nth(group.rank())
            .ok_or_else(|| Exception::custom("logical sum-scatter rank is missing"));
    }
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_sum_scatter,
    )
}

fn checked_peer(peer: usize, group: &Group, role: &str) -> Result<i32> {
    let size = group.size();
    if size == 1 {
        return Err(Exception::custom(format!(
            "cannot use a {role} rank with a singleton distributed group"
        )));
    }
    if peer >= size {
        return Err(Exception::custom(format!(
            "invalid {role} rank {peer} for distributed group of size {size}"
        )));
    }
    i32::try_from(peer).map_err(|_| Exception::custom(format!("{role} rank does not fit in i32")))
}

/// Lazily send `input` to rank `destination` on `stream`.
///
/// Ring only supports direct neighbors; other backend restrictions are
/// reported by MLX.
pub fn send(
    input: &Array,
    destination: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let destination = checked_peer(destination, group, "destination")?;
    let stream = stream.as_ref();
    if let Some(logical) = &group.logical {
        if let Some(exchanged) = logical_direct_exchange(input, group, stream)? {
            return Ok(exchanged);
        }
        let destination =
            usize::try_from(destination).expect("checked non-negative logical destination rank");
        let _destination_global = logical.global_ranks[destination];
        let source_global = logical.global_ranks[logical.rank];
        let packed = pack_logical_value(input, source_global, group.native_size(), stream)?;
        let exchanged = collective(&packed, group, stream, safemlx_sys::mlx_distributed_all_sum)?;
        let sent = exchanged.try_index_device(
            i32::try_from(source_global)
                .map_err(|_| Exception::custom("logical source rank does not fit in i32"))?,
            stream,
        )?;
        return Ok(sent);
    }
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all input handles remain alive and `res` is an owned output
        // guard. `destination` was range-checked above.
        unsafe {
            safemlx_sys::mlx_distributed_send(
                res,
                input.as_ptr(),
                destination,
                group.native.c_group,
                stream.as_ptr(),
            )
        }
    })
}

/// Lazily receive an array of `shape` and `dtype` from rank `source`.
pub fn recv(
    shape: &[i32],
    dtype: Dtype,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    if shape.iter().any(|dimension| *dimension < 0) {
        return Err(Exception::custom(
            "receive shape dimensions must be non-negative",
        ));
    }
    let source = checked_peer(source, group, "source")?;
    let stream = stream.as_ref();
    if let Some(logical) = &group.logical {
        let empty = zeros_dtype(shape, dtype, stream)?;
        if let Some(exchanged) = logical_direct_exchange(&empty, group, stream)? {
            return Ok(exchanged);
        }
        let source = usize::try_from(source).expect("checked non-negative logical source rank");
        let source_global = logical.global_ranks[source];
        let packed = pack_logical_value(&empty, source_global, group.native_size(), stream)?;
        let exchanged = collective(&packed, group, stream, safemlx_sys::mlx_distributed_all_sum)?;
        return exchanged.try_index_device(
            i32::try_from(source_global)
                .map_err(|_| Exception::custom("logical source rank does not fit in i32"))?,
            stream,
        );
    }
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: `shape` and all borrowed handles remain alive for this call;
        // `source` and every shape dimension were validated above.
        unsafe {
            safemlx_sys::mlx_distributed_recv(
                res,
                shape.as_ptr(),
                shape.len(),
                dtype.into(),
                source,
                group.native.c_group,
                stream.as_ptr(),
            )
        }
    })
}

/// Lazily receive from rank `source`, using `like` for shape and dtype.
pub fn recv_like(
    like: &Array,
    source: usize,
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let source = checked_peer(source, group, "source")?;
    let stream = stream.as_ref();
    if let Some(logical) = &group.logical {
        let empty = zeros_dtype(like.shape(), like.dtype(), stream)?;
        if let Some(exchanged) = logical_direct_exchange(&empty, group, stream)? {
            return Ok(exchanged);
        }
        let source = usize::try_from(source).expect("checked non-negative logical source rank");
        let source_global = logical.global_ranks[source];
        let packed = pack_logical_value(&empty, source_global, group.native_size(), stream)?;
        let exchanged = collective(&packed, group, stream, safemlx_sys::mlx_distributed_all_sum)?;
        return exchanged.try_index_device(
            i32::try_from(source_global)
                .map_err(|_| Exception::custom("logical source rank does not fit in i32"))?,
            stream,
        );
    }
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: all borrowed handles remain alive and `source` was checked.
        unsafe {
            safemlx_sys::mlx_distributed_recv_like(
                res,
                like.as_ptr(),
                source,
                group.native.c_group,
                stream.as_ptr(),
            )
        }
    })
}

/// Select a process-local device by explicit local index.
///
/// Do not pass [`Group::rank`] blindly: global ranks span machines and need not
/// match local device indices. Launchers should pass a local rank explicitly.
/// In the common one-process-per-visible-GPU setup, `CUDA_VISIBLE_DEVICES`
/// exposes one GPU to each process, so `local_index` is usually zero even when
/// the global distributed rank is nonzero.
pub fn device_for_local_rank(device_type: DeviceType, local_index: usize) -> Result<Device> {
    let local_index = i32::try_from(local_index)
        .map_err(|_| Exception::custom("local device index does not fit in i32"))?;
    Ok(Device::new(device_type, local_index))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn singleton() -> Group {
        init(false, Backend::Ring).unwrap()
    }

    #[test]
    fn backend_names_are_checked_before_ffi() {
        assert_eq!(Backend::try_from("ring").unwrap(), Backend::Ring);
        assert!(Backend::try_from("ring\0other").is_err());
        assert!(Backend::try_from("future-backend").is_err());
    }

    #[test]
    fn non_strict_group_is_usable() {
        let group = singleton();
        assert_eq!(group.rank(), 0);
        assert_eq!(group.size(), 1);
        assert!(format!("{group:?}").contains("rank: 0"));
        assert!(format!("{group:?}").contains("size: 1"));
    }

    #[test]
    fn singleton_collectives_preserve_values() {
        let group = singleton();
        let stream = crate::test_stream();
        let input = Array::arange::<_, f32>(Some(1), 3, None::<i32>, stream).unwrap();

        for output in [
            all_sum(&input, &group, stream).unwrap(),
            all_max(&input, &group, stream).unwrap(),
            all_min(&input, &group, stream).unwrap(),
            all_gather(&input, &group, stream).unwrap(),
            sum_scatter(&input, &group, stream).unwrap(),
        ] {
            assert_eq!(output.shape(), &[2]);
            assert_eq!(output.dtype(), Dtype::Float32);
            let output = output.evaluated().unwrap();
            assert_eq!(output.as_slice::<f32>(), &[1.0, 2.0]);
        }
    }

    #[test]
    fn singleton_split_reports_backend_support() {
        let group = singleton();
        match group.split(0, None) {
            Ok(subgroup) => {
                assert_eq!(subgroup.rank(), 0);
                assert_eq!(subgroup.size(), 1);
            }
            Err(error) => assert!(error.what().contains("split")),
        }
    }

    #[test]
    fn validates_point_to_point_inputs() {
        let group = singleton();
        let stream = crate::test_stream();
        let input = Array::arange::<_, i32>(Some(0), 1, None::<i32>, stream).unwrap();
        assert!(send(&input, 0, &group, stream).is_err());
        assert!(recv(&[-1], Dtype::Int32, 0, &group, stream).is_err());
        assert!(recv_like(&input, 0, &group, stream).is_err());
    }

    #[test]
    fn local_device_index_is_explicit() {
        let device = device_for_local_rank(DeviceType::Cpu, 0).unwrap();
        assert_eq!(device.get_index().unwrap(), 0);
        assert!(device_for_local_rank(DeviceType::Cpu, usize::MAX).is_err());
    }
}
