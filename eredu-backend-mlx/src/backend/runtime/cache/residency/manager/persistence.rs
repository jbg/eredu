//! Safetensors serialization and host-buffer reconstruction.

use super::*;

pub(in super::super) fn write_live_block(
    directory: &Path,
    id: &CacheBlockId,
    block: &HostCacheBlock,
) -> Result<DiskLocation, CacheResidencyError> {
    let publication = LiveCacheBlockPublication::begin(directory, id);
    let layout = save_live_host_cache_block(publication.staging_path(), block)?;
    sync_file(publication.staging_path())?;
    let live_source = publication.commit_owned_with_layout(layout)?;
    let path = live_source.path().to_path_buf();
    let names = array_names(id.representation);
    Ok(DiskLocation::ordinary(DiskLocationData {
        path,
        first_name: names.0.into(),
        second_name: names.1.into(),
        persistent: false,
        buffered: None,
        payload_sha256: None,
        payload_verification: Arc::new(OnceLock::new()),
        live_source: Some(live_source),
        persistent_source: None,
    }))
}

// Both ordinary live writes and the future original task consume the same
// neutral header/source worker. The live publication already owns a unique
// staging path, so no secondary tempfile or metadata-index reconstruction is needed.
fn save_live_host_cache_block(
    path: &Path,
    block: &HostCacheBlock,
) -> Result<eredu_runtime::cache::CacheShardLayout, CacheResidencyError> {
    let [first, second] = block.buffers();
    let shapes = [host_shape_to_stored(first)?, host_shape_to_stored(second)?];
    let dtypes = [host_dtype_to_stored(first)?, host_dtype_to_stored(second)?];
    let bytes = [
        first
            .as_bytes()
            .map_err(|source| transfer_error("read first live cache payload", source))?,
        second
            .as_bytes()
            .map_err(|source| transfer_error("read second live cache payload", source))?,
    ];
    let source = eredu_runtime::cache::CacheShardMetadata::ordinary(
        block.representation(),
        [&shapes[0], &shapes[1]],
        dtypes,
        [bytes[0].len(), bytes[1].len()],
    )?;
    let layout = source.into_ordinary_layout()?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|source| CacheResidencyError::Io {
            action: "create unique live cache staging file",
            path: path.to_path_buf(),
            source,
        })?;
    layout.write_to(&mut file, bytes)?;
    Ok(layout)
}

pub(in super::super) fn save_block_arrays(
    path: &Path,
    arrays: &CacheBlockArrays,
) -> Result<(), CacheResidencyError> {
    let names = array_names(arrays.representation());
    let values = arrays.arrays();
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
    Ok(host_scalar_to_stored(dtype))
}

pub(in super::super) fn host_scalar_to_stored(dtype: Dtype) -> StoredDtype {
    match dtype {
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
    }
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

pub(in super::super) fn verify_disk_payload(
    location: &DiskLocation,
) -> Result<(), CacheResidencyError> {
    let Some(expected) = &location.payload_sha256 else {
        return Ok(());
    };
    let verification = location.payload_verification.get_or_init(|| {
        let actual = if let Some(buffered) = &location.buffered {
            if buffered.len() < 8 {
                return Err("file is too short for a safetensors header".into());
            }
            let mut length_bytes = [0u8; 8];
            length_bytes.copy_from_slice(&buffered[..8]);
            let header_len = usize::try_from(u64::from_le_bytes(length_bytes))
                .map_err(|_| "safetensors header length exceeds addressable memory".to_string())?;
            let data_start = 8usize
                .checked_add(header_len)
                .filter(|start| *start <= buffered.len())
                .ok_or_else(|| {
                    "safetensors header extends beyond the buffered shard".to_string()
                })?;
            sha256_hex(Sha256::digest(&buffered[data_start..]))
        } else {
            hash_prompt_cache_shard_payload(&location.path).map_err(|error| error.to_string())?
        };
        if &actual == expected {
            Ok(())
        } else {
            Err(format!(
                "payload SHA-256 mismatch: expected {expected}, computed {actual}"
            ))
        }
    });
    verification
        .as_ref()
        .map_err(|reason| CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: reason.clone(),
        })
        .copied()
}

pub(in super::super) fn load_host_cache_block_direct(
    location: &DiskLocation,
    representation: CacheRepresentation,
) -> Result<HostCacheBlock, CacheResidencyError> {
    let authenticated = location.file_source();
    if authenticated.is_none() {
        verify_disk_payload(location)?;
    }
    let owned;
    let bytes = if let Some(buffered) = &location.buffered {
        buffered.as_ref()
    } else if let Some(source) = authenticated.as_ref().filter(|_| cfg!(unix)) {
        let bytes = source
            .file_bytes()
            .ok_or(eredu_runtime::cache::CacheShardError::Header)?;
        let file = File::open(source.path()).map_err(|source| CacheResidencyError::Io {
            action: "open authenticated cache shard",
            path: location.path.clone(),
            source,
        })?;
        let mut output = vec![0; bytes];
        source.read_ordinary_into(file, &mut output)?;
        owned = output;
        owned.as_slice()
    } else {
        owned = fs::read(&location.path).map_err(|source| CacheResidencyError::Io {
            action: "read cache block shard",
            path: location.path.clone(),
            source,
        })?;
        owned.as_slice()
    };
    if let Some(layout) = authenticated.as_ref().and_then(|source| source.layout()) {
        if layout.names() != [location.first_name.as_str(), location.second_name.as_str()]
            || layout.names() != eredu_runtime::cache::cache_shard_tensor_names(representation)
        {
            return Err(eredu_runtime::cache::CacheShardError::Header.into());
        }
        let [first, second] = layout.tensors(bytes)?;
        return Ok(HostCacheBlock::from_buffers(
            representation,
            host_buffer_from_parts(first.dtype(), first.shape(), first.data())?,
            host_buffer_from_parts(second.dtype(), second.shape(), second.data())?,
        ));
    }
    let tensors = safetensors::SafeTensors::deserialize(bytes).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: error.to_string(),
        }
    })?;
    if tensors.names().len() != 2 {
        return Err(CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: "unexpected extra arrays".into(),
        });
    }
    let first = tensors.tensor(&location.first_name).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: error.to_string(),
        }
    })?;
    let second = tensors.tensor(&location.second_name).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: location.path.clone(),
            reason: error.to_string(),
        }
    })?;
    Ok(HostCacheBlock::from_buffers(
        representation,
        host_buffer_from_view(first)?,
        host_buffer_from_view(second)?,
    ))
}

pub(in super::super) fn host_buffer_from_view(
    view: safetensors::tensor::TensorView<'_>,
) -> Result<ImmutableHostTransferBuffer, CacheResidencyError> {
    host_buffer_from_parts(view.dtype(), view.shape(), view.data())
}
fn host_buffer_from_parts(
    dtype: StoredDtype,
    shape: &[usize],
    data: &[u8],
) -> Result<ImmutableHostTransferBuffer, CacheResidencyError> {
    let shape = shape
        .iter()
        .map(|dimension| {
            i32::try_from(*dimension).map_err(|_| {
                CacheResidencyError::ArrayMismatch(
                    "cache block dimension exceeds the MLX i32 shape range".into(),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dtype = stored_dtype_to_host(dtype)?;
    let mut buffer = HostTransferBuffer::new(&shape, dtype, HostTransferPolicy::Transfer)
        .map_err(|source| transfer_error("allocate disk-loaded host cache buffer", source))?;
    let destination = buffer
        .as_bytes_mut()
        .map_err(|source| transfer_error("access disk-loaded host cache buffer", source))?;
    if destination.len() != data.len() {
        return Err(CacheResidencyError::ArrayMismatch(
            "cache block payload length does not match its shape and dtype".into(),
        ));
    }
    destination.copy_from_slice(data);
    Ok(buffer.freeze())
}

pub(in super::super) fn array_names(
    representation: CacheRepresentation,
) -> (&'static str, &'static str) {
    match representation {
        CacheRepresentation::KeyValue => ("keys", "values"),
        CacheRepresentation::CompressedLatentRotary => ("latent", "rotary_key"),
    }
}

pub(in super::super) fn buffer_prompt_cache_shard(
    path: &Path,
) -> Result<Arc<[u8]>, CacheResidencyError> {
    let buffered = fs::read(path).map_err(|source| CacheResidencyError::Io {
        action: "read prompt cache shard",
        path: path.to_path_buf(),
        source,
    })?;
    safetensors::SafeTensors::deserialize(&buffered).map_err(|error| {
        CacheResidencyError::MalformedShard {
            path: path.to_path_buf(),
            reason: error.to_string(),
        }
    })?;
    Ok(Arc::from(buffered))
}
