use super::*;
use eredu_core::cache::mechanisms::{CacheHostBuffer, LiveCacheAppendBackend};
use std::time::Duration;

#[derive(Clone, Debug)]
struct MemoryCacheBackend;

#[derive(Clone, Debug)]
struct Tensor {
    shape: Vec<i32>,
    dtype: StoredDtype,
    bytes: Arc<[u8]>,
}

#[derive(Debug)]
struct WritableTensor {
    shape: Vec<i32>,
    dtype: StoredDtype,
    bytes: Vec<u8>,
}

impl CacheHostBuffer for MemoryCacheBackend {
    type Buffer = WritableTensor;

    fn allocate(shape: &[usize], dtype: StoredDtype) -> Result<Self::Buffer, BackendFailure> {
        let byte_width = match dtype {
            StoredDtype::F32 | StoredDtype::I32 => 4,
            _ => {
                return Err(BackendFailure::from_error(std::io::Error::other(
                    "unsupported test dtype",
                )));
            }
        };
        let length = shape.iter().product::<usize>() * byte_width;
        Ok(WritableTensor {
            shape: shape
                .iter()
                .map(|&dim| i32::try_from(dim).unwrap())
                .collect(),
            dtype,
            bytes: vec![0; length],
        })
    }

    fn bytes_mut(buffer: &mut Self::Buffer) -> Result<&mut [u8], BackendFailure> {
        Ok(&mut buffer.bytes)
    }
}

impl LiveCacheBackend for MemoryCacheBackend {
    type Tensor = Tensor;
    type Host = Tensor;
    type Device = ();
    type Queue = ();
    type Event = ();

    fn shape(tensor: &Tensor) -> &[i32] {
        &tensor.shape
    }
    fn dtype(tensor: &Tensor) -> StoredDtype {
        tensor.dtype.clone()
    }
    fn bytes(tensor: &Tensor) -> u64 {
        tensor.bytes.len() as u64
    }
    fn copy_tensor(tensor: &Tensor, _: &()) -> Result<Tensor, BackendFailure> {
        Ok(Tensor {
            bytes: Arc::from(tensor.bytes.as_ref()),
            ..tensor.clone()
        })
    }
    fn settle_tensors(_: &[&Tensor]) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn device_ready(_: [&Tensor; 2]) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn device(_: &()) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn host_device() {}
    fn transfer_queue(_: (), _: CacheTransferLane) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn host_capacity_bound(bytes: u64) -> Result<u64, BackendFailure> {
        Ok(bytes)
    }
    fn host_capacity(host: &Tensor) -> Result<u64, BackendFailure> {
        Ok(Self::bytes(host))
    }
    fn host_payload(host: &Tensor) -> Result<CacheHostTensor<'_>, BackendFailure> {
        Ok(CacheHostTensor {
            shape: host
                .shape
                .iter()
                .map(|&dim| usize::try_from(dim).unwrap())
                .collect(),
            dtype: host.dtype.clone(),
            bytes: &host.bytes,
        })
    }
    fn freeze_host(buffer: WritableTensor) -> Tensor {
        Tensor {
            shape: buffer.shape,
            dtype: buffer.dtype,
            bytes: buffer.bytes.into(),
        }
    }
    fn copy_to_host<const N: usize>(
        tensors: [&Tensor; N],
        _: &(),
    ) -> Result<[Tensor; N], BackendFailure> {
        Ok(tensors.map(Clone::clone))
    }
    fn copy_to_device<const N: usize>(
        buffers: [&Tensor; N],
        _: &(),
    ) -> Result<[(Tensor, ()); N], BackendFailure> {
        Ok(buffers.map(|tensor| (tensor.clone(), ())))
    }
    fn wait_on(_: &(), _: &()) -> Result<(), BackendFailure> {
        Ok(())
    }
    fn synchronize(_: ()) -> Result<(), BackendFailure> {
        Ok(())
    }
}

impl LiveCacheAppendBackend for MemoryCacheBackend {
    fn slice_tokens(
        tensor: &Tensor,
        axis: usize,
        range: std::ops::Range<i32>,
        _: &(),
    ) -> Result<Tensor, BackendFailure> {
        assert_eq!(axis, 2);
        let mut shape = tensor.shape.clone();
        shape[axis] = range.end - range.start;
        let byte_range = range.start as usize * 4..range.end as usize * 4;
        Ok(Tensor {
            shape,
            dtype: tensor.dtype.clone(),
            bytes: Arc::from(&tensor.bytes[byte_range]),
        })
    }

    fn concatenate_tokens(
        slices: &[&Tensor],
        axis: usize,
        _: &(),
    ) -> Result<Tensor, BackendFailure> {
        assert_eq!(axis, 2);
        let mut shape = slices[0].shape.clone();
        shape[axis] = slices.iter().map(|tensor| tensor.shape[axis]).sum();
        Ok(Tensor {
            shape,
            dtype: slices[0].dtype.clone(),
            bytes: slices
                .iter()
                .flat_map(|tensor| tensor.bytes.iter().copied())
                .collect(),
        })
    }

    fn compact_tokens(tensor: &Tensor, queue: &()) -> Result<Tensor, BackendFailure> {
        Self::copy_tensor(tensor, queue)
    }
}

type Cache = CacheResidencyManager<MemoryCacheBackend>;
type TestPage = Page<MemoryCacheBackend>;
type TestHostPage = HostCacheBlock<MemoryCacheBackend>;
type TestStorage = LiveCacheBlockStorage<MemoryCacheBackend>;
type TestRecord = CacheBlockRecord<MemoryCacheBackend>;
type TestState = CacheManagerState<MemoryCacheBackend>;
type TestDiskTicket = DiskTicket<MemoryCacheBackend>;
type TestDemotionTicket = HostDemotionTicket<MemoryCacheBackend>;

fn disk_test_id(start: i64) -> CacheBlockId {
    CacheBlockId {
        session_id: 7,
        global_layer: 0,
        representation: CacheRepresentation::KeyValue,
        start,
        end: start + 1,
        rank: None,
    }
}

fn missing_location(root: &Path, name: &str) -> DiskLocation {
    DiskLocation {
        path: root.join(name),
        first_name: "keys".into(),
        second_name: "values".into(),
        persistent: false,
        source: None,
        logical_bytes: 8,
    }
}

fn page(value: i32) -> TestPage {
    let tensor = Tensor {
        shape: vec![1, 1, 1, 1],
        dtype: StoredDtype::I32,
        bytes: Arc::from(value.to_le_bytes()),
    };
    CacheBlockArrays::KeyValue {
        keys: tensor.clone(),
        values: tensor,
    }
}

fn test_device_block() -> TestPage {
    page(0)
}
fn test_host_block() -> TestHostPage {
    HostCacheBlock::from_device_arrays(&test_device_block(), &()).unwrap()
}

fn pending_demotion() -> HostDemotionCompletion<MemoryCacheBackend> {
    HostDemotionCompletion {
        result: Mutex::new(None),
        ready: Condvar::new(),
    }
}

include!("tests/transactions.rs");

#[test]
fn independent_backend_spills_and_reloads_exact_bytes() {
    let directory = tempfile::tempdir().unwrap();
    let options = PagedCacheOptions::new(1, 16, 8, 1)
        .unwrap()
        .with_live_disk(directory.path(), 64, 1)
        .unwrap();
    let cache = Cache::new(options).unwrap();
    let stream = LiveCacheStream {
        global_layer: 0,
        representation: CacheRepresentation::KeyValue,
        rank: None,
        protected_prefix: 0,
    };
    let mut state = CacheStreamState {
        tail: None,
        tail_start: 0,
        offset: 0,
    };
    for value in [16_777_217, -24, 123, 345] {
        cache
            .append_pages(&stream, &mut state, page(value), None, &())
            .unwrap();
    }
    assert_eq!(state.offset, 4);
    let selected = cache
        .layer_block_ids(0, CacheRepresentation::KeyValue, 0, 1, 0)
        .unwrap()
        .remove(0);
    let lease = cache.lease_block(&selected, &()).unwrap();
    assert_eq!(
        lease.arrays().arrays()[0].bytes.as_ref(),
        &16_777_217_i32.to_le_bytes()
    );
    assert_eq!(
        lease.arrays().arrays()[1].bytes.as_ref(),
        &16_777_217_i32.to_le_bytes()
    );
    assert!(cache.report().unwrap().disk_demotions > 0);
    drop(lease);
    cache.clear().unwrap();
    drop(cache);
    assert!(directory.path().read_dir().unwrap().next().is_none());
}
