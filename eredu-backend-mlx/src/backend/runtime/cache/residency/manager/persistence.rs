//! Safetensors serialization and host-buffer reconstruction.

use super::*;

pub(in super::super) fn write_live_block(
    directory: &Path,
    id: &CacheBlockId,
    block: &HostCacheBlock,
) -> Result<DiskLocation, CacheResidencyError> {
    let publication = LiveCacheBlockPublication::begin(directory, id);
    save_host_cache_block(publication.staging_path(), block)?;
    sync_file(publication.staging_path())?;
    let path = publication.commit()?;
    let names = array_names(id.representation);
    Ok(DiskLocation {
        path,
        first_name: names.0.into(),
        second_name: names.1.into(),
        persistent: false,
        source: None,
        logical_bytes: block.bytes()?,
        payload_sha256: None,
    })
}

pub(in super::super) fn save_block_arrays(
    path: &Path,
    arrays: &CacheBlockArrays,
) -> Result<(), CacheResidencyError> {
    let names = array_names(arrays.representation());
    let values = arrays.arrays();
    // The native serializer rejects empty arrays, but empty companion storage
    // is valid for key-only attention and named records. Use the shared bounded
    // host encoding, which preserves those exact zero-width declarations.
    if values.iter().any(|value| value.shape().contains(&0)) {
        let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
        return save_host_cache_block(path, &HostCacheBlock::from_device_arrays(arrays, &stream)?);
    }
    Array::save_safetensors([(names.0, values[0]), (names.1, values[1])], None, path).map_err(
        |source| CacheResidencyError::Runtime(format!("save {}: {source}", path.display())),
    )
}

pub(in super::super) fn save_host_cache_block(
    path: &Path,
    block: &HostCacheBlock,
) -> Result<(), CacheResidencyError> {
    let names = array_names(block.representation());
    let [first, second] = block.buffers();
    let first_shape = host_shape_to_stored(first)?;
    let second_shape = host_shape_to_stored(second)?;
    let first_dtype = host_dtype_to_stored(first)?;
    let second_dtype = host_dtype_to_stored(second)?;
    let first_bytes = first
        .as_bytes()
        .map_err(|source| transfer_error("read first host cache payload", source))?;
    let second_bytes = second
        .as_bytes()
        .map_err(|source| transfer_error("read second host cache payload", source))?;
    let first_view = TensorView::new(first_dtype, first_shape, first_bytes).map_err(|source| {
        CacheResidencyError::Runtime(format!("create first host cache tensor view: {source}"))
    })?;
    let second_view =
        TensorView::new(second_dtype, second_shape, second_bytes).map_err(|source| {
            CacheResidencyError::Runtime(format!("create second host cache tensor view: {source}"))
        })?;
    serialize_to_file([(names.0, first_view), (names.1, second_view)], None, path).map_err(
        |source| CacheResidencyError::Runtime(format!("save {}: {source}", path.display())),
    )
}

pub(in super::super) fn host_shape_to_stored(
    buffer: &ImmutableHostTransferBuffer,
) -> Result<Vec<usize>, CacheResidencyError> {
    buffer
        .shape()
        .map_err(|source| transfer_error("inspect host cache shape", source))?
        .into_iter()
        .map(|dimension| {
            usize::try_from(dimension).map_err(|_| {
                CacheResidencyError::Runtime(
                    "host cache shape contains a negative dimension".into(),
                )
            })
        })
        .collect()
}

pub(in super::super) fn host_dtype_to_stored(
    buffer: &ImmutableHostTransferBuffer,
) -> Result<StoredDtype, CacheResidencyError> {
    let dtype = buffer
        .dtype()
        .map_err(|source| transfer_error("inspect host cache dtype", source))?;
    Ok(match dtype {
        Dtype::Bool => StoredDtype::BOOL,
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
    })
}

pub(in super::super) fn stored_dtype_to_host(
    dtype: StoredDtype,
) -> Result<Dtype, CacheResidencyError> {
    match dtype {
        StoredDtype::BOOL => Ok(Dtype::Bool),
        StoredDtype::U8 => Ok(Dtype::Uint8),
        StoredDtype::U16 => Ok(Dtype::Uint16),
        StoredDtype::U32 => Ok(Dtype::Uint32),
        StoredDtype::U64 => Ok(Dtype::Uint64),
        StoredDtype::I8 => Ok(Dtype::Int8),
        StoredDtype::I16 => Ok(Dtype::Int16),
        StoredDtype::I32 => Ok(Dtype::Int32),
        StoredDtype::I64 => Ok(Dtype::Int64),
        StoredDtype::F16 => Ok(Dtype::Float16),
        StoredDtype::F32 => Ok(Dtype::Float32),
        StoredDtype::F64 => Ok(Dtype::Float64),
        StoredDtype::BF16 => Ok(Dtype::Bfloat16),
        StoredDtype::C64 => Ok(Dtype::Complex64),
        other => Err(CacheResidencyError::ArrayMismatch(format!(
            "unsupported host cache dtype {other:?}"
        ))),
    }
}

pub(in super::super) fn load_host_cache_block_direct(
    location: &DiskLocation,
    representation: CacheRepresentation,
) -> Result<HostCacheBlock, CacheResidencyError> {
    let opened;
    let source = if let Some(source) = &location.source {
        source.as_ref()
    } else {
        opened = eredu_runtime::RetainedCacheShard::open(&location.path, location.logical_bytes)?;
        &opened
    };
    if source.metadata().tensors().len() != 2 {
        return Err(CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: "unexpected extra arrays".into(),
        });
    }
    let allocate = |name: &str| -> Result<HostTransferBuffer, CacheResidencyError> {
        let info =
            source
                .metadata()
                .info(name)
                .ok_or_else(|| CacheResidencyError::MalformedShard {
                    path: location.path.clone(),
                    reason: format!("missing array {name}"),
                })?;
        let shape = info
            .shape
            .iter()
            .map(|n| {
                i32::try_from(*n).map_err(|_| {
                    CacheResidencyError::ArrayMismatch("cache block dimension exceeds i32".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        HostTransferBuffer::new(
            &shape,
            stored_dtype_to_host(info.dtype)?,
            HostTransferPolicy::Transfer,
        )
        .map_err(|source| transfer_error("allocate disk-loaded host cache buffer", source))
    };
    let mut first = allocate(&location.first_name)?;
    let mut second = allocate(&location.second_name)?;
    source.read_into(
        &mut [
            (
                &location.first_name,
                first
                    .as_bytes_mut()
                    .map_err(|e| transfer_error("first cache buffer", e))?,
            ),
            (
                &location.second_name,
                second
                    .as_bytes_mut()
                    .map_err(|e| transfer_error("second cache buffer", e))?,
            ),
        ],
        location.payload_sha256.as_deref(),
    )?;
    Ok(HostCacheBlock::from_buffers(
        representation,
        first.freeze(),
        second.freeze(),
    ))
}

pub(in super::super) fn remove_ephemeral_file(record: &CacheBlockRecord) {
    if let Some(location) = record.disk() {
        if !location.persistent {
            let _ = fs::remove_file(&location.path);
        }
    }
}

pub(in super::super) fn array_names(
    representation: CacheRepresentation,
) -> (&'static str, &'static str) {
    match representation {
        CacheRepresentation::KeyValue => ("keys", "values"),
        CacheRepresentation::AppendStream { .. } => ("records", "reserved"),
        CacheRepresentation::CompressedLatentRotary => ("latent", "rotary_key"),
    }
}
