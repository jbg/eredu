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
    utils::{guard::Guarded, runtime_lock, SUCCESS},
    Array, Dtype, Stream,
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

/// An owned MLX distributed group.
///
/// The final related group frees the native handle on drop. The type is
/// intentionally neither `Clone`, `Send`, nor `Sync`: not every communication
/// backend documents cross-thread group access.
#[derive(Clone)]
pub struct Group {
    native: Rc<NativeGroup>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

struct NativeGroup {
    c_group: safemlx_sys::mlx_distributed_group,
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
    /// Returns whether two handles retain the same native communicator.
    pub fn shares_native_handle(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.native, &other.native)
    }

    pub(crate) fn from_owned_ptr(c_group: safemlx_sys::mlx_distributed_group) -> Self {
        Self {
            native: Rc::new(NativeGroup { c_group }),
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
        self.native_rank()
    }

    /// Return the number of processes in the group.
    pub fn size(&self) -> usize {
        self.native_size()
    }

    /// Split the group by `color`, optionally ordering new ranks by `key`.
    ///
    /// A missing or negative key asks MLX to use the current group rank. Backend
    /// support varies: MLX 0.32 supports splitting with MPI and NCCL, while its
    /// singleton, Ring, and JACCL groups return an error.
    pub fn split(&self, color: i32, key: Option<i32>) -> Result<Self> {
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

fn validate_all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
) -> Result<(Vec<i64>, Vec<i64>, Vec<i32>)> {
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
    let recv_rows = recv_counts.iter().try_fold(0usize, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| Exception::custom("all_to_all_v receive count sum overflowed usize"))
    })?;
    let input_rows = usize::try_from(input.dim(0))
        .map_err(|_| Exception::custom("all_to_all_v input row count is negative"))?;
    if send_rows != input_rows {
        return Err(Exception::custom(format!(
            "all_to_all_v send count sum {send_rows} does not match input row count {input_rows}"
        )));
    }
    i32::try_from(recv_rows)
        .map_err(|_| Exception::custom("all_to_all_v output row count exceeds i32"))?;

    let mut row_bytes = input.item_size();
    for &dimension in &input.shape()[1..] {
        let dimension = usize::try_from(dimension)
            .map_err(|_| Exception::custom("all_to_all_v trailing shape is negative"))?;
        row_bytes = row_bytes
            .checked_mul(dimension)
            .ok_or_else(|| Exception::custom("all_to_all_v row byte size overflowed usize"))?;
    }
    row_bytes
        .checked_mul(recv_rows)
        .ok_or_else(|| Exception::custom("all_to_all_v output byte size overflowed usize"))?;
    for (peer, (&send, &recv)) in send_counts.iter().zip(recv_counts).enumerate() {
        row_bytes.checked_mul(send).ok_or_else(|| {
            Exception::custom(format!(
                "all_to_all_v send byte size for peer {peer} overflowed usize"
            ))
        })?;
        row_bytes.checked_mul(recv).ok_or_else(|| {
            Exception::custom(format!(
                "all_to_all_v receive byte size for peer {peer} overflowed usize"
            ))
        })?;
    }
    if group.size() == 1 && send_rows != recv_rows {
        return Err(Exception::custom(
            "all_to_all_v singleton send and receive counts must match",
        ));
    }
    if send_counts[group.rank()] != recv_counts[group.rank()] {
        return Err(Exception::custom(format!(
            "all_to_all_v self send count {} does not match self receive count {}",
            send_counts[group.rank()],
            recv_counts[group.rank()]
        )));
    }

    let send = send_counts
        .iter()
        .map(|count| {
            i64::try_from(*count)
                .map_err(|_| Exception::custom("all_to_all_v send count exceeds i64"))
        })
        .collect::<Result<Vec<_>>>()?;
    let recv = recv_counts
        .iter()
        .map(|count| {
            i64::try_from(*count)
                .map_err(|_| Exception::custom("all_to_all_v receive count exceeds i64"))
        })
        .collect::<Result<Vec<_>>>()?;
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
    Ok((send, recv, offsets))
}

fn native_all_to_all_v(
    input: &Array,
    send_counts: &[i64],
    recv_counts: &[i64],
    group: &Group,
    stream: &Stream,
) -> Result<Array> {
    let _guard = runtime_lock::enter();
    Array::try_from_op(|res| {
        // SAFETY: count slices and all borrowed MLX handles remain alive for
        // the call. The returned primitive retains its own group and counts.
        unsafe {
            safemlx_sys::mlx_distributed_all_to_all_v(
                res,
                input.as_ptr(),
                send_counts.as_ptr(),
                send_counts.len(),
                recv_counts.as_ptr(),
                recv_counts.len(),
                group.native.c_group,
                stream.as_ptr(),
            )
        }
    })
}

/// Sum `input` element-wise across `group` on `stream`.
pub fn all_sum(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_sum,
    )
}

/// Take the element-wise maximum of `input` across `group` on `stream`.
pub fn all_max(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_max,
    )
}

/// Take the element-wise minimum of `input` across `group` on `stream`.
pub fn all_min(input: &Array, group: &Group, stream: impl AsRef<Stream>) -> Result<Array> {
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
    collective(
        input,
        group,
        stream.as_ref(),
        safemlx_sys::mlx_distributed_all_gather,
    )
}

/// Exchange variable-sized leading-axis row blocks between all ranks.
///
/// `send_counts[d]` rows are addressed to destination group rank `d`, and the
/// compact input is concatenated in destination-rank order. `recv_counts[s]`
/// rows are expected from source group rank `s`; the compact output is
/// concatenated in source-rank order. The operation is lazy for native MLX
/// groups. Logical Ring/JACCL subgroups use their topology-planned neighbor
/// routes and exchange only the addressed payload, with small routed count
/// headers materialized to size intermediate receives.
pub fn all_to_all_v(
    input: &Array,
    send_counts: &[usize],
    recv_counts: &[usize],
    group: &Group,
    stream: impl AsRef<Stream>,
) -> Result<Array> {
    let stream = stream.as_ref();
    let (send, recv, _) = validate_all_to_all_v(input, send_counts, recv_counts, group)?;
    if group.size() == 1 {
        return Ok(input.clone());
    }
    native_all_to_all_v(input, &send, &recv, group, stream)
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
    fn singleton_all_to_all_v_is_a_validated_identity() {
        let group = singleton();
        let stream = crate::test_stream();
        let input = Array::arange::<_, i32>(Some(1), 5, None::<i32>, stream)
            .unwrap()
            .reshape(&[2, 2], stream)
            .unwrap();
        let output = all_to_all_v(&input, &[2], &[2], &group, stream).unwrap();
        drop(input);
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.evaluated().unwrap().as_slice::<i32>(), &[1, 2, 3, 4]);

        let empty = crate::ops::zeros_dtype(&[0, 3], Dtype::Float32, stream).unwrap();
        let output = all_to_all_v(&empty, &[0], &[0], &group, stream).unwrap();
        assert_eq!(output.shape(), &[0, 3]);
    }

    #[test]
    fn all_to_all_v_reports_exact_validation_errors() {
        let group = singleton();
        let stream = crate::test_stream();
        let scalar = Array::from_int(1);
        assert_eq!(
            all_to_all_v(&scalar, &[1], &[1], &group, stream)
                .unwrap_err()
                .what(),
            "all_to_all_v input must have a leading row dimension"
        );

        let input = Array::arange::<_, i32>(Some(0), 2, None::<i32>, stream).unwrap();
        assert_eq!(
            all_to_all_v(&input, &[], &[2], &group, stream)
                .unwrap_err()
                .what(),
            "all_to_all_v requires 1 send counts and receive counts, got 0 and 1"
        );
        assert_eq!(
            all_to_all_v(&input, &[1], &[1], &group, stream)
                .unwrap_err()
                .what(),
            "all_to_all_v send count sum 1 does not match input row count 2"
        );
        assert_eq!(
            all_to_all_v(&input, &[2], &[1], &group, stream)
                .unwrap_err()
                .what(),
            "all_to_all_v singleton send and receive counts must match"
        );
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
}
