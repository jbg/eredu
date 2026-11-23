use std::{ops::Range, sync::Arc};

use candle_core::{DType, Tensor as NativeTensor};
use eredu_checkpoint::{recipe::RecipeDtype, StoredDtype};
use eredu_core::{
    cache::mechanisms::{
        CacheHostBuffer, CacheHostTensor, CacheTensorStorage, CacheTransferLane,
        LiveCacheAppendBackend, LiveCacheBackend, ResidentAppendStorage, ResidentKeyValueStorage,
    },
    BackendFailure, BackendFailureKind,
};
use eredu_nn::{
    mechanism_memory::{MechanismInvocation, MechanismMemoryContract},
    AttentionRequest, Error, Index, NeuralBackend, Tensor,
};

use crate::{checkpoint, CandleContext, CandleNeuralBackend, CandleTensor};

pub(crate) mod paged;
mod pooling;

/// Native tensor storage for the portable resident cache.
#[derive(Clone, Debug)]
pub struct CandleCacheStorage;

/// Exact native context that a cache transfer can share with execution.
#[derive(Clone, Debug)]
pub struct CandleCacheDevice(CandleContext);

impl PartialEq for CandleCacheDevice {
    fn eq(&self, other: &Self) -> bool {
        self.0.device.same_device(&other.0.device)
    }
}

impl Eq for CandleCacheDevice {}

/// Owned contiguous bytes for a completed cache transfer.
#[derive(Debug)]
pub struct CandleCacheBuffer {
    shape: Vec<usize>,
    dtype: StoredDtype,
    bytes: Box<[u8]>,
}

/// Immutable storage for a visible key/value prefix and reserved capacity.
#[derive(Clone, Debug)]
pub struct CandleCachePrefix {
    backing: CandleTensor,
    length: i32,
}

fn invalid(message: &'static str) -> BackendFailure {
    BackendFailure::new(BackendFailureKind::InvalidInput, Error::backend(message))
}

fn tensor_size(tensor: &CandleTensor) -> u64 {
    (tensor.value.elem_count() as u64) * (tensor.value.dtype().size_in_bytes() as u64)
}

fn selection(
    tensor: &CandleTensor,
    axis: usize,
    range: Range<i32>,
) -> Result<Vec<Index>, BackendFailure> {
    let mut indexes = vec![Index::Full; tensor.shape().len()];
    let dimension = indexes
        .get_mut(axis)
        .ok_or_else(|| invalid("cache axis is out of range"))?;
    if range.start < 0 || range.end < range.start || range.end > tensor.shape()[axis] {
        return Err(invalid("cache interval is outside the selected axis"));
    }
    *dimension = Index::Range(range.start, range.end);
    Ok(indexes)
}

fn encode(tensor: &CandleTensor) -> Result<Vec<u8>, BackendFailure> {
    let result = (|| {
        let flat = tensor.value.flatten_all()?;
        macro_rules! scalar_bytes {
            ($scalar:ty, $bits:expr) => {{
                flat.to_vec1::<$scalar>()?
                    .into_iter()
                    .flat_map($bits)
                    .collect()
            }};
        }
        Ok(match flat.dtype() {
            DType::U8 => flat.to_vec1::<u8>()?,
            DType::U32 => scalar_bytes!(u32, u32::to_le_bytes),
            DType::I16 => scalar_bytes!(i16, i16::to_le_bytes),
            DType::I32 => scalar_bytes!(i32, i32::to_le_bytes),
            DType::I64 => scalar_bytes!(i64, i64::to_le_bytes),
            DType::F32 => scalar_bytes!(f32, f32::to_le_bytes),
            DType::F64 => scalar_bytes!(f64, f64::to_le_bytes),
            DType::F16 => {
                scalar_bytes!(half::f16, |value: half::f16| value.to_bits().to_le_bytes())
            }
            DType::BF16 => scalar_bytes!(half::bf16, |value: half::bf16| value
                .to_bits()
                .to_le_bytes()),
            _ => candle_core::bail!("unsupported CPU cache transfer dtype"),
        })
    })();
    result.map_err(BackendFailure::from_error)
}

impl CacheHostBuffer for CandleCacheStorage {
    type Buffer = CandleCacheBuffer;

    fn allocate(shape: &[usize], dtype: StoredDtype) -> Result<Self::Buffer, BackendFailure> {
        let width = checkpoint::native_dtype(&RecipeDtype::from(dtype.clone()))
            .map_err(BackendFailure::from_error)?
            .size_in_bytes();
        let length = shape
            .iter()
            .try_fold(width, |n, &dimension| n.checked_mul(dimension))
            .ok_or_else(|| invalid("cache buffer byte count overflows"))?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|error| BackendFailure::new(BackendFailureKind::ResourceExhausted, error))?;
        bytes.resize(length, 0);
        Ok(CandleCacheBuffer {
            shape: shape.to_vec(),
            dtype,
            bytes: bytes.into_boxed_slice(),
        })
    }

    fn bytes_mut(buffer: &mut Self::Buffer) -> Result<&mut [u8], BackendFailure> {
        Ok(&mut buffer.bytes)
    }
}

impl LiveCacheBackend for CandleCacheStorage {
    type Tensor = CandleTensor;
    type Host = Arc<CandleCacheBuffer>;
    type Device = CandleCacheDevice;
    type Queue = CandleContext;
    type Event = ();

    fn shape(tensor: &CandleTensor) -> &[i32] {
        tensor.shape()
    }

    fn dtype(tensor: &CandleTensor) -> StoredDtype {
        eredu_checkpoint::store::stored_dtype_from_safetensors(tensor.value.dtype().into())
    }

    fn bytes(tensor: &CandleTensor) -> u64 {
        tensor_size(tensor)
    }

    fn copy_tensor(
        tensor: &CandleTensor,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        Self::compact_tokens(tensor, queue)
    }

    fn settle_tensors(tensors: &[&CandleTensor]) -> Result<(), BackendFailure> {
        CandleContext::settle_roots(tensors.iter().copied())
    }

    fn device_ready(tensors: [&CandleTensor; 2]) -> Result<(), BackendFailure> {
        Self::settle_tensors(&tensors)
    }

    fn device(queue: &CandleContext) -> Result<Self::Device, BackendFailure> {
        queue.check_healthy()?;
        Ok(CandleCacheDevice(queue.clone()))
    }

    fn host_device() -> Self::Device {
        CandleCacheDevice(CandleContext::cpu())
    }

    fn transfer_queue(
        device: Self::Device,
        _: CacheTransferLane,
    ) -> Result<CandleContext, BackendFailure> {
        device.0.check_healthy()?;
        Ok(device.0)
    }

    fn host_capacity_bound(bytes: u64) -> Result<u64, BackendFailure> {
        Ok(bytes)
    }

    fn host_capacity(host: &Self::Host) -> Result<u64, BackendFailure> {
        Ok(host.bytes.len() as u64)
    }

    fn host_payload(host: &Self::Host) -> Result<CacheHostTensor<'_>, BackendFailure> {
        Ok(CacheHostTensor {
            shape: host.shape.clone(),
            dtype: host.dtype.clone(),
            bytes: &host.bytes,
        })
    }

    fn freeze_host(buffer: Self::Buffer) -> Self::Host {
        Arc::new(buffer)
    }

    fn copy_to_host<const N: usize>(
        tensors: [&CandleTensor; N],
        queue: &CandleContext,
    ) -> Result<[Self::Host; N], BackendFailure> {
        queue.check_healthy()?;
        CandleContext::settle_roots(tensors)?;
        let mut outputs = Vec::with_capacity(N);
        for tensor in tensors {
            outputs.push(Self::freeze_host(CandleCacheBuffer {
                shape: tensor.value.dims().to_vec(),
                dtype: Self::dtype(tensor),
                bytes: encode(tensor)?.into_boxed_slice(),
            }));
        }
        outputs
            .try_into()
            .map_err(|_| invalid("cache transfer output count differs"))
    }

    fn copy_to_device<const N: usize>(
        buffers: [&Self::Host; N],
        queue: &CandleContext,
    ) -> Result<[(CandleTensor, ()); N], BackendFailure> {
        queue.check_healthy()?;
        let mut outputs = Vec::with_capacity(N);
        for buffer in buffers {
            let tensor = checkpoint::import_bytes(
                &buffer.bytes,
                &buffer.dtype.clone().into(),
                &buffer.shape,
                queue,
            )
            .map_err(BackendFailure::from_error)?;
            outputs.push((tensor, ()));
        }
        if let Err(error) = queue.synchronize() {
            queue.quarantine((buffers.map(Arc::clone), outputs));
            return Err(error);
        }
        outputs
            .try_into()
            .map_err(|_| invalid("cache transfer output count differs"))
    }

    fn wait_on(_: &(), queue: &CandleContext) -> Result<(), BackendFailure> {
        queue.check_healthy()
    }

    fn synchronize(_: ()) -> Result<(), BackendFailure> {
        Ok(())
    }
}

impl LiveCacheAppendBackend for CandleCacheStorage {
    fn slice_tokens(
        tensor: &CandleTensor,
        axis: usize,
        range: Range<i32>,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        tensor
            .index(&selection(tensor, axis, range)?, queue)
            .map_err(BackendFailure::from_error)
    }

    fn concatenate_tokens(
        slices: &[&CandleTensor],
        axis: usize,
        _: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let values: Vec<_> = slices.iter().map(|tensor| &tensor.value).collect();
        CandleTensor::wrap(NativeTensor::cat(&values, axis)).map_err(BackendFailure::from_error)
    }

    fn compact_tokens(
        tensor: &CandleTensor,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        queue.validate_tensor_device(tensor)?;
        let result = tensor.compact(queue).map_err(BackendFailure::from_error)?;
        if let Err(error) = queue.synchronize() {
            queue.quarantine((tensor.clone(), result));
            return Err(error);
        }
        Ok(result)
    }
}

impl CacheTensorStorage for CandleCacheStorage {
    type Value = CandleTensor;

    fn into_value(tensor: CandleTensor) -> CandleTensor {
        tensor
    }
    fn into_storage(value: CandleTensor) -> CandleTensor {
        value
    }
    fn storage_ref(tensor: &CandleTensor) -> &CandleTensor {
        tensor
    }
    fn borrow_value(tensor: &CandleTensor) -> &CandleTensor {
        tensor
    }

    fn zeroed_cache_tensor(
        shape: &[i32],
        like: &CandleTensor,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let shape: Vec<_> = shape
            .iter()
            .map(|&dim| usize::try_from(dim))
            .collect::<Result<_, _>>()
            .map_err(BackendFailure::from_error)?;
        CandleTensor::wrap(NativeTensor::zeros(
            shape,
            like.value.dtype(),
            &queue.device,
        ))
        .map_err(BackendFailure::from_error)
    }

    fn allocate_tokens(
        like: &CandleTensor,
        axis: usize,
        capacity: i32,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        let mut shape = like.shape().to_vec();
        *shape
            .get_mut(axis)
            .ok_or_else(|| invalid("cache allocation axis is out of range"))? = capacity;
        Self::zeroed_cache_tensor(&shape, like, queue)
    }

    fn write_tokens(
        target: CandleTensor,
        axis: usize,
        positions: Range<i32>,
        source: &CandleTensor,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        selection(&target, axis, positions.clone())?;
        let mut expected = target.shape().to_vec();
        expected[axis] = positions.end - positions.start;
        if source.shape() != expected || target.value.dtype() != source.value.dtype() {
            return Err(invalid(
                "cache update shape or type differs from the selected interval",
            ));
        }
        let left = Self::slice_tokens(&target, axis, 0..positions.start, queue)?;
        let right = Self::slice_tokens(&target, axis, positions.end..target.shape()[axis], queue)?;
        Self::concatenate_tokens(&[&left, source, &right], axis, queue)
    }

    fn snapshot_tensor_costs() -> Option<eredu_core::cache::mechanisms::CacheSnapshotTensorCosts> {
        Some(eredu_core::cache::mechanisms::CacheSnapshotTensorCosts {
            tensor_metadata: 4096,
            dimension_metadata: 16,
            copy_buffers: 2,
            floating_width: 8,
        })
    }

    fn fork_tensor(tensor: &CandleTensor) -> Result<CandleTensor, BackendFailure> {
        let result = tensor
            .independent_copy()
            .map_err(BackendFailure::from_error)?;
        CandleContext::settle_roots([&result])?;
        Ok(result)
    }
}

impl ResidentKeyValueStorage for CandleCacheStorage {
    type Prefix = CandleCachePrefix;

    fn resident_attention(
        request: AttentionRequest<'_, CandleTensor>,
        queue: &CandleContext,
    ) -> Result<CandleTensor, Error> {
        CandleNeuralBackend::attention_with_sinks(request, queue)
    }

    fn describe_resident_append(
        invocation: &MechanismInvocation,
        _: ResidentAppendStorage<'_, CandleTensor>,
    ) -> Result<MechanismMemoryContract, Error> {
        MechanismMemoryContract::unknown(invocation)
    }

    fn new_prefix(
        tensor: &CandleTensor,
        capacity: i32,
        queue: &CandleContext,
    ) -> Result<Self::Prefix, BackendFailure> {
        let length = tensor
            .shape()
            .get(2)
            .copied()
            .ok_or_else(|| invalid("cache prefix needs a sequence axis"))?;
        if capacity < length {
            return Err(invalid("cache capacity cannot truncate a new prefix"));
        }
        let backing = if capacity == length {
            Self::compact_tokens(tensor, queue)?
        } else {
            CandleTensor::wrap(
                tensor
                    .value
                    .pad_with_zeros(2, 0, (capacity - length) as usize),
            )
            .map_err(BackendFailure::from_error)?
        };
        Ok(CandleCachePrefix { backing, length })
    }

    fn append_prefix(
        prefix: &Self::Prefix,
        update: &CandleTensor,
        capacity: i32,
        queue: &CandleContext,
    ) -> Result<Self::Prefix, BackendFailure> {
        let committed = Self::prefix_view(prefix, queue)?;
        let joined = Self::concatenate_tokens(&[&committed, update], 2, queue)?;
        Self::new_prefix(&joined, capacity, queue)
    }

    fn prefix_view(
        prefix: &Self::Prefix,
        queue: &CandleContext,
    ) -> Result<CandleTensor, BackendFailure> {
        Self::slice_tokens(&prefix.backing, 2, 0..prefix.length, queue)
    }

    fn truncate_prefix(prefix: &Self::Prefix, length: i32) -> Result<Self::Prefix, BackendFailure> {
        if !(0..=prefix.length).contains(&length) {
            return Err(invalid("cache truncation lies outside the visible prefix"));
        }
        Ok(CandleCachePrefix {
            backing: prefix.backing.clone(),
            length,
        })
    }

    fn prefix_capacity(prefix: &Self::Prefix) -> i32 {
        prefix.backing.dim(2)
    }

    fn prefix_spare_bytes(prefix: &Self::Prefix) -> u64 {
        let capacity = Self::prefix_capacity(prefix) as u64;
        tensor_size(&prefix.backing)
            .checked_div(capacity)
            .unwrap_or(0)
            * (capacity - prefix.length as u64)
    }

    fn can_append_prefix_without_copy(_: &Self::Prefix, _: i32) -> bool {
        false
    }
}

/// Retains cache state and copy authority through native device settlement.
pub struct CandleCopyScope<R: 'static>(crate::device::CandleSettlement<R>);

impl<R: 'static> eredu_core::cache::mechanisms::CacheCopyScope<CandleTensor, R>
    for CandleCopyScope<R>
{
    fn resources(&self) -> &R {
        self.0.resources()
    }

    fn resources_mut(&mut self) -> &mut R {
        self.0.resources_mut()
    }

    fn settle(&mut self, roots: &[CandleTensor]) -> Result<(), BackendFailure> {
        for root in roots {
            self.0.queue.validate_tensor_device(root)?;
        }
        self.0.settle()
    }

    fn finish(self) {
        self.0.release_settled();
    }
}

impl eredu_core::cache::mechanisms::CacheCopySubmission for CandleCacheStorage {
    type CopyScope<R: 'static> = CandleCopyScope<R>;

    fn begin_copy<R: 'static>(
        queue: &CandleContext,
        resources: R,
    ) -> Result<Self::CopyScope<R>, BackendFailure> {
        queue.check_healthy()?;
        Ok(CandleCopyScope(crate::device::CandleSettlement::new(
            queue, resources,
        )))
    }

    fn reclaim_copies() {
        CandleContext::reclaim_retired();
    }
}
