//! Completed source streaming through the canonical two-tensor shard writer.
use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataAllocation, WorkspaceMetadataError};
use eredu_runtime::cache::{CacheShardLayout, CacheShardMetadata};
use safemlx::EvaluatedArray;
use std::{
    io::Write,
    mem::{size_of, size_of_val},
};

#[derive(Debug, thiserror::Error)]
enum ReadbackCause {
    #[error("prompt-cache source has unsupported geometry or scalar byte order")]
    Geometry,
    #[error(transparent)]
    Completed(#[from] safemlx::error::CompletedReadbackError),
    #[error(transparent)]
    Readback(#[from] safemlx::error::NativeBytesCopyError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Header(#[from] eredu_checkpoint::safetensors::SafetensorsHeaderError),
    #[error(transparent)]
    Host(#[from] safemlx::error::Exception),
}
fn failure(context: &WorkspaceContext, cause: ReadbackCause) -> CacheResidencyError {
    CacheResidencyError::Preparation(context.metadata_source(cause))
}
pub(super) fn controls(
    context: &WorkspaceContext,
    bytes: Option<usize>,
) -> Result<(), CacheResidencyError> {
    context
        .charge_metadata(bytes.ok_or_else(|| {
            CacheResidencyError::Preparation(WorkspaceMetadataError::Overflow.into())
        })?)
        .map_err(|cause| CacheResidencyError::Preparation(cause.into()))
}

/// The source remains borrowed from its pinned canonical snapshot or exclusive
/// mutable-state owner throughout this synchronous read. No evaluation, graph,
/// tensor staging allocation, cache mutation or execution authority is created.
pub(super) fn write_completed_block(
    path: &Path,
    representation: CacheRepresentation,
    arrays: [&Array; 2],
    context: &WorkspaceContext,
) -> Result<String, CacheResidencyError> {
    let frames = [
        size_of::<(&Path, CacheRepresentation, [&Array; 2], &WorkspaceContext)>(),
        size_of::<[Vec<usize>; 2]>(),
        size_of::<[StoredDtype; 2]>(),
        size_of::<[usize; 2]>(),
        size_of::<File>(),
        size_of::<Result<File, std::io::Error>>(),
        size_of::<Result<(), CacheResidencyError>>(),
        size_of::<[EvaluatedArray<'_>; 2]>(),
    ];
    controls(
        context,
        frames
            .into_iter()
            .try_fold(size_of_val(&frames), usize::checked_add),
    )?;
    if !cfg!(target_endian = "little") {
        return Err(failure(context, ReadbackCause::Geometry));
    }
    let mut shapes = [Vec::new(), Vec::new()];
    let mut bytes = [0; 2];
    let mut dtypes = [StoredDtype::F32; 2];
    for (index, array) in arrays.into_iter().enumerate() {
        shapes[index] = context
            .metadata_vec(array.ndim())
            .map_err(CacheResidencyError::Preparation)?;
        for dimension in array.shape() {
            shapes[index].push(
                usize::try_from(*dimension)
                    .map_err(|_| failure(context, ReadbackCause::Geometry))?,
            );
        }
        bytes[index] = array.nbytes();
        dtypes[index] = host_scalar_to_stored(array.dtype());
    }
    controls(
        context,
        Array::completed_borrow_control_bytes().and_then(|n| n.checked_mul(2)),
    )?;
    let completed = [
        arrays[0]
            .try_completed()
            .map_err(|cause| failure(context, cause.into()))?,
        arrays[1]
            .try_completed()
            .map_err(|cause| failure(context, cause.into()))?,
    ];
    let layout = CacheShardMetadata::prepare(
        representation,
        [&shapes[0], &shapes[1]],
        dtypes,
        bytes,
        context,
    )?
    .into_prepared_layout(context)?;
    let mut hasher = Sha256::new();
    controls(context, Some(size_of::<Sha256>()))?;
    let writer =
        |index: usize, file: &mut File| write_array(file, &completed[index], &mut hasher, context);
    fn quote<E, F>(_: &F) -> Option<usize> {
        CacheShardLayout::write_with_control_bytes::<E, F>()
    }
    controls(context, quote::<CacheResidencyError, _>(&writer))?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|cause| failure(context, cause.into()))?;
    layout.write_with(&mut file, writer)?;
    file.sync_all()
        .map_err(|cause| failure(context, cause.into()))?;
    hash_string(hasher, context)
}

fn write_array(
    file: &mut File,
    array: &EvaluatedArray<'_>,
    hasher: &mut Sha256,
    context: &WorkspaceContext,
) -> Result<(), CacheResidencyError> {
    const CHUNK: usize = 4096;
    controls(
        context,
        Some(size_of::<(
            [u8; CHUNK],
            usize,
            usize,
            usize,
            usize,
            &mut File,
            &EvaluatedArray<'_>,
            &WorkspaceContext,
        )>()),
    )?;
    let native = array.as_array();
    let width = native.item_size();
    if width == 0 || width > CHUNK {
        return Err(failure(context, ReadbackCause::Geometry));
    }
    let mut buffer = [0; CHUNK];
    let mut offset = 0;
    while offset < native.size() {
        let count = (CHUNK / width).min(native.size() - offset);
        let destination = &mut buffer[..count * width];
        controls(
            context,
            EvaluatedArray::native_element_range_readback_control_bytes(),
        )?;
        array
            .try_copy_native_element_range_into(offset, destination)
            .map_err(|cause| failure(context, cause.into()))?;
        hasher.update(&*destination);
        file.write_all(destination)
            .map_err(|cause| failure(context, cause.into()))?;
        offset += count;
    }
    Ok(())
}

fn hash_string(hasher: Sha256, context: &WorkspaceContext) -> Result<String, CacheResidencyError> {
    let digest = hasher.finalize();
    let mut text = context
        .metadata_string(format_args!("{:064}", ""))
        .map_err(CacheResidencyError::Preparation)?;
    text.clear();
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in digest {
        text.push(HEX[usize::from(byte >> 4)] as char);
        text.push(HEX[usize::from(byte & 15)] as char);
    }
    Ok(text)
}

/// The fixed-state writer uses the same counted canonical header producer as
/// cache block shards, with its actual single state tensor declaration.
pub(super) fn write_completed_state(
    path: &Path,
    array: &Array,
    context: &WorkspaceContext,
) -> Result<String, CacheResidencyError> {
    use eredu_checkpoint::safetensors::SafetensorsHeaderPlan;
    controls(
        context,
        SafetensorsHeaderPlan::control_bytes()
            .and_then(|n| n.checked_add(Array::completed_borrow_control_bytes()?))
            .and_then(|n| {
                n.checked_add(size_of::<(
                    [(&str, safetensors::tensor::TensorInfo); 1],
                    Vec<u8>,
                    EvaluatedArray<'_>,
                    File,
                    Sha256,
                    [u8; 32],
                )>())
            }),
    )?;
    if !cfg!(target_endian = "little") {
        return Err(failure(context, ReadbackCause::Geometry));
    }
    let mut shape = context
        .metadata_vec(array.ndim())
        .map_err(CacheResidencyError::Preparation)?;
    for dimension in array.shape() {
        shape.push(
            usize::try_from(*dimension).map_err(|_| failure(context, ReadbackCause::Geometry))?,
        );
    }
    let source = [(
        "state",
        safetensors::tensor::TensorInfo {
            dtype: host_scalar_to_stored(array.dtype()),
            shape,
            data_offsets: (0, array.nbytes()),
        },
    )];
    let plan =
        SafetensorsHeaderPlan::prepare(&source).map_err(|cause| failure(context, cause.into()))?;
    let mut header = context
        .metadata_vec(plan.header_bytes())
        .map_err(CacheResidencyError::Preparation)?;
    header.resize(plan.header_bytes(), 0);
    plan.write_header(&mut header)
        .map_err(|cause| failure(context, cause.into()))?;
    let completed = array
        .try_completed()
        .map_err(|cause| failure(context, cause.into()))?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|cause| failure(context, cause.into()))?;
    file.write_all(&header)
        .map_err(|cause| failure(context, cause.into()))?;
    let mut hasher = Sha256::new();
    write_array(&mut file, &completed, &mut hasher, context)?;
    file.sync_all()
        .map_err(|cause| failure(context, cause.into()))?;
    hash_string(hasher, context)
}

pub(super) fn write_host_block(
    path: &Path,
    representation: CacheRepresentation,
    buffers: [&ImmutableHostTransferBuffer; 2],
    context: &WorkspaceContext,
) -> Result<String, CacheResidencyError> {
    controls(
        context,
        safemlx::HostTransferDescriptor::<4>::control_bytes()
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| {
                n.checked_add(size_of::<(
                    [safemlx::HostTransferDescriptor<4>; 2],
                    [Vec<usize>; 2],
                    File,
                    Sha256,
                    [&[u8]; 2],
                )>())
            }),
    )?;
    let descriptors = [
        buffers[0].try_fixed_descriptor::<4>(),
        buffers[1].try_fixed_descriptor::<4>(),
    ];
    let descriptors = [
        descriptors[0]
            .as_ref()
            .map_err(|_| failure(context, ReadbackCause::Geometry))?,
        descriptors[1]
            .as_ref()
            .map_err(|_| failure(context, ReadbackCause::Geometry))?,
    ];
    let mut shapes = [Vec::new(), Vec::new()];
    for index in 0..2 {
        shapes[index] = context
            .metadata_vec(descriptors[index].shape().len())
            .map_err(CacheResidencyError::Preparation)?;
        for dimension in descriptors[index].shape() {
            shapes[index].push(
                usize::try_from(*dimension)
                    .map_err(|_| failure(context, ReadbackCause::Geometry))?,
            );
        }
    }
    let layout = CacheShardMetadata::prepare(
        representation,
        [&shapes[0], &shapes[1]],
        descriptors.map(|d| host_scalar_to_stored(d.dtype())),
        descriptors.map(|d| d.nbytes()),
        context,
    )?
    .into_prepared_layout(context)?;
    let mut hasher = Sha256::new();
    controls(
        context,
        ImmutableHostTransferBuffer::byte_borrow_control_bytes().and_then(|n| n.checked_mul(2)),
    )?;
    let writer = |index: usize, file: &mut File| {
        let bytes = buffers[index]
            .as_bytes()
            .map_err(|cause| failure(context, cause.into()))?;
        hasher.update(bytes);
        file.write_all(bytes)
            .map_err(|cause| failure(context, cause.into()))
    };
    fn quote<E, F>(_: &F) -> Option<usize> {
        CacheShardLayout::write_with_control_bytes::<E, F>()
    }
    controls(context, quote::<CacheResidencyError, _>(&writer))?;
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|cause| failure(context, cause.into()))?;
    layout.write_with(&mut file, writer)?;
    file.sync_all()
        .map_err(|cause| failure(context, cause.into()))?;
    hash_string(hasher, context)
}
