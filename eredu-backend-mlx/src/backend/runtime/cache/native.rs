//! MLX storage and transfer operations for the portable live cache.

mod copy_scope;
mod paged_attention;
mod pooling_masks;

use eredu_checkpoint::StoredDtype;
use eredu_core::{
    cache::mechanisms::{
        CacheHostBuffer, CacheHostTensor, CacheProcessUsage, CacheTransferLane, LiveCacheBackend,
    },
    BackendFailure,
};
use safemlx::{
    host_transfer_capacity_upper_bound,
    transforms::{async_eval_with_event, eval},
    Array, Device, DeviceType, Dtype, Event, HostTransferBuffer, HostTransferPolicy,
    ImmutableHostTransferBuffer, Stream,
};
use std::sync::{Arc, Mutex};

mod streams;

/// MLX tensor and transfer types used by the portable live cache.
#[derive(Debug, Clone)]
pub struct MlxCacheBackend;

/// Concrete MLX transfer device.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct CacheTransferDevice {
    device_type: DeviceType,
    index: i32,
}

impl CacheTransferDevice {
    const CPU: Self = Self {
        device_type: DeviceType::Cpu,
        index: 0,
    };
}

impl CacheHostBuffer for MlxCacheBackend {
    type Buffer = HostTransferBuffer;

    fn allocate(shape: &[usize], dtype: StoredDtype) -> Result<Self::Buffer, BackendFailure> {
        let dtype = match dtype {
            StoredDtype::Bool => Dtype::Bool,
            StoredDtype::U8 => Dtype::Uint8,
            StoredDtype::U16 => Dtype::Uint16,
            StoredDtype::U32 => Dtype::Uint32,
            StoredDtype::U64 => Dtype::Uint64,
            StoredDtype::I8 => Dtype::Int8,
            StoredDtype::I16 => Dtype::Int16,
            StoredDtype::I32 => Dtype::Int32,
            StoredDtype::I64 => Dtype::Int64,
            StoredDtype::F16 => Dtype::Float16,
            StoredDtype::F32 => Dtype::Float32,
            StoredDtype::F64 => Dtype::Float64,
            StoredDtype::BF16 => Dtype::Bfloat16,
            StoredDtype::C64 => Dtype::Complex64,
            other => {
                return Err(BackendFailure::from_error(
                    safemlx::error::Exception::custom(format!(
                        "unsupported host cache dtype {other:?}"
                    )),
                ));
            }
        };
        let shape = shape
            .iter()
            .map(|&dim| i32::try_from(dim).map_err(BackendFailure::from_error))
            .collect::<Result<Vec<_>, _>>()?;
        HostTransferBuffer::new(&shape, dtype, HostTransferPolicy::Transfer)
            .map_err(BackendFailure::from_error)
    }

    fn bytes_mut(buffer: &mut Self::Buffer) -> Result<&mut [u8], BackendFailure> {
        buffer.as_bytes_mut().map_err(BackendFailure::from_error)
    }
}

impl LiveCacheBackend for MlxCacheBackend {
    type Tensor = Array;
    type Host = Arc<ImmutableHostTransferBuffer>;
    type Device = CacheTransferDevice;
    type Queue = Stream;
    type Event = Event;

    fn shape(tensor: &Array) -> &[i32] {
        tensor.shape()
    }

    fn dtype(tensor: &Array) -> StoredDtype {
        storage_dtype(tensor.dtype())
    }

    fn bytes(tensor: &Array) -> u64 {
        tensor.nbytes() as u64
    }

    fn copy_tensor(tensor: &Array, queue: &Stream) -> Result<Array, BackendFailure> {
        tensor
            .contiguous(false, queue)
            .and_then(|array| array.deep_clone())
            .map_err(BackendFailure::from_error)
    }

    fn settle_tensors(tensors: &[&Array]) -> Result<(), BackendFailure> {
        if tensors.is_empty() {
            return Ok(());
        }
        eval(tensors.iter().copied()).map_err(BackendFailure::from_error)
    }

    fn device_ready(tensors: [&Array; 2]) -> Result<Event, BackendFailure> {
        async_eval_with_event(tensors).map_err(BackendFailure::from_error)
    }

    fn device(queue: &Stream) -> Result<CacheTransferDevice, BackendFailure> {
        let device = queue.get_device().map_err(BackendFailure::from_error)?;
        Ok(CacheTransferDevice {
            device_type: device.get_type().map_err(BackendFailure::from_error)?,
            index: device.get_index().map_err(BackendFailure::from_error)?,
        })
    }

    fn host_device() -> CacheTransferDevice {
        CacheTransferDevice::CPU
    }

    fn transfer_queue(
        device: CacheTransferDevice,
        lane: CacheTransferLane,
    ) -> Result<Stream, BackendFailure> {
        streams::cache_transfer_stream(device, lane)
    }

    fn host_capacity_bound(bytes: u64) -> Result<u64, BackendFailure> {
        let bytes = usize::try_from(bytes).map_err(BackendFailure::from_error)?;
        host_transfer_capacity_upper_bound(bytes, HostTransferPolicy::Transfer)
            .map(|size| size as u64)
            .map_err(BackendFailure::from_error)
    }

    fn host_capacity(host: &Self::Host) -> Result<u64, BackendFailure> {
        host.capacity()
            .map(|size| size as u64)
            .map_err(BackendFailure::from_error)
    }

    fn host_payload(host: &Self::Host) -> Result<CacheHostTensor<'_>, BackendFailure> {
        let shape = host
            .shape()
            .map_err(BackendFailure::from_error)?
            .into_iter()
            .map(|dim| usize::try_from(dim).map_err(BackendFailure::from_error))
            .collect::<Result<Vec<_>, _>>()?;
        let dtype = storage_dtype(host.dtype().map_err(BackendFailure::from_error)?);
        let bytes = host.as_bytes().map_err(BackendFailure::from_error)?;
        Ok(CacheHostTensor {
            shape,
            dtype,
            bytes,
        })
    }

    fn freeze_host(buffer: HostTransferBuffer) -> Self::Host {
        Arc::new(buffer.freeze())
    }

    fn copy_to_host<const N: usize>(
        tensors: [&Array; N],
        queue: &Stream,
    ) -> Result<[Self::Host; N], BackendFailure> {
        let mut transfers = Vec::with_capacity(N);
        let mut failure = None;
        for tensor in tensors {
            match HostTransferBuffer::copy_from_array(tensor, HostTransferPolicy::Transfer, queue) {
                Ok(transfer) => transfers.push(transfer),
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
        let mut buffers = Vec::with_capacity(N);
        for transfer in transfers {
            match transfer.synchronize() {
                Ok(buffer) => buffers.push(Self::freeze_host(buffer)),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        match failure {
            Some(error) => Err(BackendFailure::from_error(error)),
            None => Ok(buffers
                .try_into()
                .ok()
                .expect("one host buffer per submitted tensor")),
        }
    }

    fn copy_to_device<const N: usize>(
        buffers: [&Self::Host; N],
        queue: &Stream,
    ) -> Result<[(Array, Event); N], BackendFailure> {
        let mut transfers = Vec::with_capacity(N);
        for buffer in buffers {
            match buffer.copy_to_array(queue) {
                Ok(transfer) => transfers.push(transfer.into_parts()),
                Err(error) => {
                    // A partially submitted transfer still owns its native allocation.
                    // Drain its event before the portable cache releases the reservation.
                    for (_, event) in transfers {
                        let _ = event.synchronize();
                    }
                    return Err(BackendFailure::from_error(error));
                }
            }
        }
        Ok(transfers
            .try_into()
            .ok()
            .expect("one device tensor per submitted buffer"))
    }

    fn wait_on(event: &Event, queue: &Stream) -> Result<(), BackendFailure> {
        event.wait_on(queue).map_err(BackendFailure::from_error)
    }

    fn synchronize(event: Event) -> Result<(), BackendFailure> {
        event.synchronize().map_err(BackendFailure::from_error)
    }

    fn process_usage() -> Option<CacheProcessUsage> {
        safemlx::system::process_usage().map(|usage| CacheProcessUsage {
            peak_rss: usage.peak_rss,
            minor_faults: usage.minor_page_faults,
            major_faults: usage.major_page_faults,
        })
    }
}

fn storage_dtype(dtype: Dtype) -> StoredDtype {
    match dtype {
        Dtype::Bool => StoredDtype::Bool,
        Dtype::Uint8 => StoredDtype::U8,
        Dtype::Uint16 => StoredDtype::U16,
        Dtype::Uint32 => StoredDtype::U32,
        Dtype::Uint64 => StoredDtype::U64,
        Dtype::Int8 => StoredDtype::I8,
        Dtype::Int16 => StoredDtype::I16,
        Dtype::Int32 => StoredDtype::I32,
        Dtype::Int64 => StoredDtype::I64,
        Dtype::Float16 => StoredDtype::F16,
        Dtype::Float32 => StoredDtype::F32,
        Dtype::Float64 => StoredDtype::F64,
        Dtype::Bfloat16 => StoredDtype::BF16,
        Dtype::Complex64 => StoredDtype::C64,
    }
}

impl eredu_core::cache::mechanisms::LiveCacheAppendBackend for MlxCacheBackend {
    fn slice_tokens(
        tensor: &Array,
        axis: usize,
        range: std::ops::Range<i32>,
        queue: &Stream,
    ) -> Result<Array, BackendFailure> {
        use safemlx::ops::indexing::TryIndexOp;
        let value = match (tensor.ndim(), axis) {
            (3, 1) => tensor.try_index_device((.., range, ..), queue),
            (4, 2) => tensor.try_index_device((.., .., range, ..), queue),
            _ => Err(safemlx::error::Exception::custom(
                "unsupported cache tensor slice axis",
            )),
        };
        value.map_err(BackendFailure::from_error)
    }

    fn concatenate_tokens(
        slices: &[&Array],
        axis: usize,
        queue: &Stream,
    ) -> Result<Array, BackendFailure> {
        let axis = i32::try_from(axis).map_err(BackendFailure::from_error)?;
        safemlx::ops::concatenate_axis(slices, axis, queue).map_err(BackendFailure::from_error)
    }

    fn compact_tokens(tensor: &Array, queue: &Stream) -> Result<Array, BackendFailure> {
        let copy = Self::copy_tensor(tensor, queue)?;
        eval([&copy]).map_err(BackendFailure::from_error)?;
        Ok(copy)
    }
}

pub(super) struct CacheReadRetention<R>(pub(super) R);

impl<R: 'static> crate::backend::submission_recovery::Retention for CacheReadRetention<R> {
    fn observe(&self, _: crate::backend::submission_recovery::Status) {}
}

impl<R: 'static> eredu_core::cache::mechanisms::CacheReadSubmission
    for crate::backend::submission_recovery::Recovery<CacheReadRetention<R>>
{
    fn settle(mut self) -> Result<(), BackendFailure> {
        self.seal();
        let status = self.finish();
        if status.failed || status.blocked {
            return Err(BackendFailure::from_error(
                safemlx::error::Exception::custom(
                    "cache read failed; unresolved source lease retained",
                ),
            ));
        }
        Ok(())
    }
}

impl eredu_core::cache::mechanisms::LiveCacheReadSubmissionBackend for MlxCacheBackend {
    fn begin_cache_read(
        _: &Stream,
        retention: impl Sized + 'static,
    ) -> Result<impl eredu_core::cache::mechanisms::CacheReadSubmission, BackendFailure> {
        crate::backend::submission_recovery::Recovery::begin(CacheReadRetention(retention))
            .map_err(BackendFailure::from_error)
    }
}

pub(super) fn with_cache_read<R: 'static, T>(
    queue: &Stream,
    retention: R,
    read: impl FnOnce() -> Result<T, safemlx::error::Exception>,
) -> Result<T, BackendFailure> {
    use eredu_core::cache::mechanisms::{CacheReadSubmission, LiveCacheReadSubmissionBackend};
    let scope = MlxCacheBackend::begin_cache_read(queue, retention)?;
    let value = read().map_err(BackendFailure::from_error)?;
    scope.settle()?;
    Ok(value)
}

impl eredu_core::cache::mechanisms::LiveCacheSelectedReadBackend for MlxCacheBackend {
    fn copy_token_slice(
        tensor: &Array,
        axis: usize,
        range: std::ops::Range<i32>,
        queue: &Stream,
        retention: impl Sized + 'static,
    ) -> Result<Array, BackendFailure> {
        use eredu_core::cache::mechanisms::LiveCacheAppendBackend;
        with_cache_read(queue, retention, || {
            let part = Self::slice_tokens(tensor, axis, range, queue)
                .map_err(safemlx::error::Exception::from_source)?;
            let part = part.contiguous(false, queue)?.deep_clone()?;
            eval([&part])?;
            Ok(part)
        })
    }

    fn gather_token_rows(
        tensors: [&Array; 2],
        batch: i32,
        rows: &[i32],
        queue: &Stream,
        retention: impl Sized + 'static,
    ) -> Result<[Array; 2], BackendFailure> {
        use safemlx::ops::indexing::{take_axis, TryIndexOp};
        with_cache_read(queue, retention, || {
            let indices = Array::from_slice(rows, &[rows.len() as i32]);
            let gather = |tensor: &Array| {
                take_axis(
                    tensor.try_index_device((batch..batch + 1, .., .., ..), queue)?,
                    &indices,
                    2,
                    queue,
                )
            };
            let pair = [gather(tensors[0])?, gather(tensors[1])?];
            eval(&pair)?;
            Ok(pair)
        })
    }
}

impl eredu_core::cache::mechanisms::CacheTensorStorage for MlxCacheBackend {
    type Value = crate::MlxTensor;

    fn into_value(tensor: Array) -> Self::Value {
        crate::MlxTensor::from_array(tensor)
    }

    fn into_storage(value: Self::Value) -> Array {
        value.into_array()
    }

    fn storage_ref(value: &Self::Value) -> &Array {
        value.as_array()
    }

    fn borrow_value(tensor: &Array) -> &Self::Value {
        use ref_cast::RefCast;
        crate::MlxTensor::ref_cast(tensor)
    }

    fn zeroed_cache_tensor(
        shape: &[i32],
        like: &Array,
        stream: &Stream,
    ) -> Result<Array, eredu_core::BackendFailure> {
        safemlx::ops::zeros_dtype(shape, like.dtype(), stream)
            .map_err(eredu_core::BackendFailure::from_error)
    }

    fn allocate_tokens(
        like: &Array,
        axis: usize,
        capacity: i32,
        queue: &Stream,
    ) -> Result<Array, BackendFailure> {
        let mut shape = like.shape().to_vec();
        shape[axis] = capacity;
        safemlx::ops::zeros_dtype(&shape, like.dtype(), queue).map_err(BackendFailure::from_error)
    }

    fn write_tokens(
        mut target: Array,
        axis: usize,
        positions: std::ops::Range<i32>,
        source: &Array,
        queue: &Stream,
    ) -> Result<Array, BackendFailure> {
        use safemlx::ops::indexing::TryIndexMutOp;
        match axis {
            1 => target.try_index_mut_device((.., positions, ..), source, queue),
            2 => target.try_index_mut_device((.., .., positions, ..), source, queue),
            _ => Err(safemlx::error::Exception::custom(
                "unsupported cache write axis",
            )),
        }
        .map_err(BackendFailure::from_error)?;
        Ok(target)
    }

    fn snapshot_tensor_costs() -> Option<eredu_core::cache::mechanisms::CacheSnapshotTensorCosts> {
        Some(eredu_core::cache::mechanisms::CacheSnapshotTensorCosts {
            tensor_metadata: 4096,
            dimension_metadata: 16,
            copy_buffers: 2,
            floating_width: 8,
        })
    }

    fn fork_tensor(tensor: &Array) -> Result<Array, BackendFailure> {
        tensor
            .clone()
            .deep_clone()
            .map_err(BackendFailure::from_error)
    }
}

impl eredu_core::cache::mechanisms::CompressedCacheStorage for MlxCacheBackend {
    fn snapshot_tensor_estimate(
        tensor: &Array,
    ) -> Option<eredu_core::execution_control::SnapshotEstimate> {
        crate::composition::mlx::speculative::state_snapshot::estimate_array(tensor)
    }
}

impl eredu_core::cache::mechanisms::ResidentKeyValueStorage for MlxCacheBackend {
    type Prefix = safemlx::PrefixAppendBuffer;

    fn resident_attention(
        request: eredu_nn::AttentionRequest<'_, Self::Value>,
        queue: &Stream,
    ) -> Result<Self::Value, eredu_nn::Error> {
        crate::backend::nn::attention::attention_with_softcap(
            request.queries.as_array(),
            request.keys.as_array(),
            request.values.as_array(),
            request.scale,
            request.mask.map(Self::Value::as_array),
            request.sinks.map(Self::Value::as_array),
            request.softcap,
            request.arithmetic,
            queue,
        )
        .map(Self::Value::from_array)
        .map_err(eredu_nn::Error::backend_source)
    }

    fn describe_resident_append(
        invocation: &eredu_nn::mechanism_memory::MechanismInvocation,
        storage: eredu_core::cache::mechanisms::ResidentAppendStorage<'_, Array>,
    ) -> Result<eredu_nn::mechanism_memory::MechanismMemoryContract, eredu_nn::Error> {
        use eredu_core::cache::mechanisms::ResidentAppendBacking;
        use eredu_nn::mechanism_memory::*;
        use eredu_nn::Error as ComputeError;
        let MechanismInvocation::CacheUpdate {
            batch,
            heads,
            key_width,
            value_width,
            element,
            ..
        } = *invocation
        else {
            return Err(ComputeError::backend(
                "cache update memory requires a cache append invocation",
            ));
        };
        for (array, width) in [storage.keys, storage.values]
            .into_iter()
            .zip([key_width, value_width])
            .filter_map(|(array, width)| array.map(|array| (array, width)))
        {
            let shape = array.shape();
            if shape.len() != 4
                || shape[0] as u64 != batch
                || shape[1] as u64 != heads
                || shape[3] as u64 != width
                || crate::backend::nn::memory::element_type(array.dtype()) != Some(element)
            {
                return Err(ComputeError::backend(
                    "cache memory append shape/dtype differs from installed arrays",
                ));
            }
        }
        let mut result = crate::backend::nn::memory::describe(invocation)?;
        result
            .storage
            .retain(|storage| storage.name == "native_workspace");
        result
            .missing
            .retain(|reason| !reason.starts_with("cache append backing"));
        let backing_positions = storage.positions;
        let backing = storage.backing;
        let key_only = storage.key_only;
        for (name, width) in [("cache_keys", key_width), ("cache_values", value_width)] {
            if key_only && name == "cache_values" {
                continue;
            }
            let payload = crate::backend::nn::memory::bytes(
                &[batch, heads, backing_positions, width],
                element_bytes(element),
            )?;
            let mut storage = crate::backend::nn::memory::allocation(
                name,
                payload,
                MechanismStorageRole::State,
                StorageRetention::Returned,
            );
            storage.detail = match backing {
                ResidentAppendBacking::BorrowedInput => {
                    storage.backing = MechanismBacking::Unknown;
                    "empty cache borrows appended tensor backing; no independent cache allocation"
                }
                ResidentAppendBacking::ReservedPrefix => {
                    storage.backing = MechanismBacking::Owner(name.into());
                    "append reserves an unused suffix of private owner capacity; published prefixes remain immutable"
                }
                ResidentAppendBacking::ExistingCapacity => {
                    storage.backing = MechanismBacking::Owner(name.into());
                    "append fits actual reserved cache capacity; mutation reuses owner backing"
                }
                ResidentAppendBacking::NewAllocation => {
                    "selected cache append allocates concatenated/padded backing; old storage remains a lazy graph dependency"
                }
            }.into();
            result.storage.push(storage);
        }
        result.missing.push("cache input layouts and native update copy-on-write/old-buffer retention require completion/lifetime composition".into());
        result.validate()?;
        Ok(result)
    }

    fn new_prefix(
        tensor: &Array,
        capacity: i32,
        queue: &Stream,
    ) -> Result<Self::Prefix, BackendFailure> {
        Self::Prefix::new(tensor, capacity, -2, queue).map_err(BackendFailure::from_error)
    }

    fn append_prefix(
        prefix: &Self::Prefix,
        update: &Array,
        capacity: i32,
        queue: &Stream,
    ) -> Result<Self::Prefix, BackendFailure> {
        prefix
            .appended(update, capacity, queue)
            .map_err(BackendFailure::from_error)
    }

    fn prefix_view(prefix: &Self::Prefix, queue: &Stream) -> Result<Array, BackendFailure> {
        prefix.prefix(queue).map_err(BackendFailure::from_error)
    }

    fn truncate_prefix(prefix: &Self::Prefix, length: i32) -> Result<Self::Prefix, BackendFailure> {
        prefix.truncated(length).map_err(BackendFailure::from_error)
    }

    fn prefix_capacity(prefix: &Self::Prefix) -> i32 {
        prefix.capacity()
    }

    fn prefix_spare_bytes(prefix: &Self::Prefix) -> u64 {
        prefix.spare_capacity_bytes() as u64
    }

    fn can_append_prefix_without_copy(prefix: &Self::Prefix, count: i32) -> bool {
        prefix.can_append_without_copy(count)
    }
}
