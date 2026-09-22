//! One completed-source copy worker for funded capture and retained inspection.
use super::*;
use eredu_core::SharedTensorObservation;
use eredu_runtime::working_memory::{
    PendingStorageAllocation, StorageAllocation, StoragePublicationLayout, WorkingMemoryError,
};
use safemlx::{ArrayElement, EvaluatedArray};
use std::{cmp::Ordering, mem::size_of};

#[derive(Clone, Debug)]
struct ObservationBacking(Arc<u8>);
impl PartialEq for ObservationBacking {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for ObservationBacking {}
impl PartialOrd for ObservationBacking {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ObservationBacking {
    fn cmp(&self, other: &Self) -> Ordering {
        Arc::as_ptr(&self.0).cmp(&Arc::as_ptr(&other.0))
    }
}
type ObservationCustody = PendingStorageAllocation<ObservationBacking>;
fn memory(error: WorkingMemoryError) -> Error {
    Error::text_admission(error)
}

fn output_width(dtype: Dtype) -> Result<usize, Error> {
    match dtype {
        Dtype::Bool => Ok(size_of::<bool>()),
        Dtype::Uint8 | Dtype::Uint16 | Dtype::Uint32 | Dtype::Uint64 => Ok(size_of::<u64>()),
        Dtype::Int8 | Dtype::Int16 | Dtype::Int32 | Dtype::Int64 => Ok(size_of::<i64>()),
        Dtype::Float16 | Dtype::Bfloat16 | Dtype::Float32 | Dtype::Float64 => Ok(size_of::<f32>()),
        Dtype::Complex64 => Err(Error::ArchitectureModel(
            "complex activation observation is unsupported".into(),
        )),
    }
}
fn worker_controls(dtype: Dtype) -> Result<usize, Error> {
    let bytes = match dtype {
        Dtype::Bool => EvaluatedArray::completed_mapped_readback_control_bytes::<bool, bool>(),
        Dtype::Uint8 => EvaluatedArray::completed_mapped_readback_control_bytes::<u8, u64>(),
        Dtype::Uint16 => EvaluatedArray::completed_mapped_readback_control_bytes::<u16, u64>(),
        Dtype::Uint32 => EvaluatedArray::completed_mapped_readback_control_bytes::<u32, u64>(),
        Dtype::Uint64 => EvaluatedArray::completed_mapped_readback_control_bytes::<u64, u64>(),
        Dtype::Int8 => EvaluatedArray::completed_mapped_readback_control_bytes::<i8, i64>(),
        Dtype::Int16 => EvaluatedArray::completed_mapped_readback_control_bytes::<i16, i64>(),
        Dtype::Int32 => EvaluatedArray::completed_mapped_readback_control_bytes::<i32, i64>(),
        Dtype::Int64 => EvaluatedArray::completed_mapped_readback_control_bytes::<i64, i64>(),
        Dtype::Float16 => {
            EvaluatedArray::completed_mapped_readback_control_bytes::<half::f16, f32>()
        }
        Dtype::Bfloat16 => {
            EvaluatedArray::completed_mapped_readback_control_bytes::<half::bf16, f32>()
        }
        Dtype::Float32 => EvaluatedArray::completed_mapped_readback_control_bytes::<f32, f32>(),
        Dtype::Float64 => EvaluatedArray::completed_mapped_readback_control_bytes::<f64, f32>(),
        Dtype::Complex64 => {
            return Err(Error::ArchitectureModel(
                "complex activation observation is unsupported".into(),
            ))
        }
    };
    bytes.ok_or_else(|| memory(WorkingMemoryError::UnknownBound))
}
fn bytes(count: usize, width: usize) -> Result<u64, Error> {
    count
        .checked_mul(width)
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))
}

/// Independently owned inspection output. Both backing allocations and the
/// shared-control owner receive their grant before either vector is allocated.
pub(in super::super) fn observe_tensor_retained(
    value: &MlxTensor,
    stream: &Stream,
) -> Result<SharedTensorObservation, Error> {
    let pool = crate::backend::managed_memory::try_ledger()?;
    observe_tensor_in(value, stream, &pool)
}

pub(super) fn observe_tensor_in(
    value: &MlxTensor,
    stream: &Stream,
    pool: &eredu_runtime::working_memory::MemoryLedger,
) -> Result<SharedTensorObservation, Error> {
    let dtype = value.as_array().dtype();
    let shape_bytes = bytes(value.shape().len(), size_of::<usize>())?;
    let data_bytes = bytes(value.as_array().size(), output_width(dtype)?)?;
    let key_allocations = size_of::<usize>()
        .checked_mul(6)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let controls = worker_controls(dtype)?
        .checked_add(key_allocations)
        .and_then(|n| n.checked_add(size_of::<[ObservationBacking; 2]>()))
        .and_then(|n| {
            n.checked_add(size_of::<(
                Vec<usize>,
                TensorObservationData,
                TensorObservation,
            )>())
        })
        .and_then(|n| n.checked_add(size_of::<Result<SharedTensorObservation, Error>>()))
        .and_then(|n| n.checked_add(size_of::<Result<(), std::collections::TryReserveError>>()))
        .and_then(|n| u64::try_from(n).ok())
        .and_then(|n| {
            n.checked_add(SharedTensorObservation::retained_control_bytes::<
                ObservationCustody,
            >()?)
        })
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let prepared = StoragePublicationLayout::<ObservationBacking>::pending(2)
        .and_then(|layout| layout.with_additional_host_metadata(controls))
        .and_then(|layout| layout.fund(pool))
        .map_err(memory)?;
    let shape_key = ObservationBacking(Arc::new(0));
    let data_key = ObservationBacking(Arc::new(0));
    let placement = pool.host_placement_handle();
    let mut custody = prepared
        .reserve_storage([
            (
                shape_key,
                StorageAllocation::new(shape_bytes, placement.clone()),
            ),
            (data_key, StorageAllocation::new(data_bytes, placement)),
        ])
        .map_err(memory)?;
    let observation = observe_tensor(value, stream)?;
    custody.publish().map_err(memory)?;
    Ok(SharedTensorObservation::retain(observation, custody))
}

fn values<S: ArrayElement + Copy + Default, T: Copy + Default>(
    source: &EvaluatedArray<'_>,
    map: fn(S) -> T,
) -> Result<Vec<T>, Error> {
    let elements = source.as_array().size();
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|error| Error::Other(Box::new(error)))?;
    if output.capacity() != elements {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    output.resize(elements, T::default());
    if elements != 0 {
        source
            .try_map_into(&mut output, map)
            .map_err(|error| Error::Other(Box::new(error)))?;
    }
    Ok(output)
}

/// Fills the destination already funded by the capture owner or the retained
/// inspection constructor above. Completion is synchronous; no staging vector
/// or native conversion graph is created by scalar normalization.
pub(in super::super) fn observe_tensor(
    value: &MlxTensor,
    _stream: &Stream,
) -> Result<TensorObservation, Error> {
    #[cfg(test)]
    super::super::bounded_capture::record_host_read(value.as_array().size());
    let source = value.as_array().evaluated()?;
    if source.as_array().size() != 0 && source.as_array().allocation_info()?.is_none() {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    let rank = value.shape().len();
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(rank)
        .map_err(|error| Error::Other(Box::new(error)))?;
    if shape.capacity() != rank {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    for &dimension in value.shape() {
        shape.push(usize::try_from(dimension).map_err(|_| {
            Error::ArchitectureModel(format!(
                "observed tensor has negative dimension {dimension}"
            ))
        })?);
    }
    let data = match value.as_array().dtype() {
        Dtype::Bool => TensorObservationData::Bool(values::<bool, bool>(&source, |v| v)?),
        Dtype::Uint8 => TensorObservationData::U64(values::<u8, u64>(&source, u64::from)?),
        Dtype::Uint16 => TensorObservationData::U64(values::<u16, u64>(&source, u64::from)?),
        Dtype::Uint32 => TensorObservationData::U64(values::<u32, u64>(&source, u64::from)?),
        Dtype::Uint64 => TensorObservationData::U64(values::<u64, u64>(&source, |v| v)?),
        Dtype::Int8 => TensorObservationData::I64(values::<i8, i64>(&source, i64::from)?),
        Dtype::Int16 => TensorObservationData::I64(values::<i16, i64>(&source, i64::from)?),
        Dtype::Int32 => TensorObservationData::I64(values::<i32, i64>(&source, i64::from)?),
        Dtype::Int64 => TensorObservationData::I64(values::<i64, i64>(&source, |v| v)?),
        Dtype::Float16 => {
            TensorObservationData::F32(values::<half::f16, f32>(&source, |v| v.to_f32())?)
        }
        Dtype::Bfloat16 => {
            TensorObservationData::F32(values::<half::bf16, f32>(&source, |v| v.to_f32())?)
        }
        Dtype::Float32 => TensorObservationData::F32(values::<f32, f32>(&source, |v| v)?),
        Dtype::Float64 => TensorObservationData::F32(values::<f64, f32>(&source, |v| v as f32)?),
        Dtype::Complex64 => {
            return Err(Error::ArchitectureModel(
                "complex activation observation is unsupported".into(),
            ))
        }
    };
    TensorObservation::new(shape, data).map_err(Error::observation)
}
