//! Completed tensor exports retain their independently admitted host backing.
use eredu_core::HostTensorBuffer;
use eredu_nn::Error;
use eredu_runtime::working_memory::{
    PendingStorageAllocation, StorageAllocation, StoragePublicationLayout, WorkingMemoryError,
};
use safemlx::{ArrayElement, Dtype, EvaluatedArray};
use std::{cmp::Ordering, mem::size_of, sync::Arc};

// Closed keys are authenticated by a paid, retained allocation. Pointer reuse
// cannot alias a live registration because that registration owns the same Arc.
#[derive(Clone, Debug)]
struct ReadbackKey(Arc<u8>);
impl PartialEq for ReadbackKey {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for ReadbackKey {}
impl PartialOrd for ReadbackKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ReadbackKey {
    fn cmp(&self, other: &Self) -> Ordering {
        Arc::as_ptr(&self.0).cmp(&Arc::as_ptr(&other.0))
    }
}

type OutputCustody = PendingStorageAllocation<ReadbackKey>;
fn memory(error: WorkingMemoryError) -> Error {
    Error::backend_retained_source(error)
}

fn export_as<S: ArrayElement + Copy + Default, T: ArrayElement + Copy + Default>(
    source: &EvaluatedArray<'_>,
    map: fn(S) -> T,
) -> Result<HostTensorBuffer<T>, Error> {
    let pool =
        crate::backend::managed_memory::try_ledger().map_err(Error::backend_retained_source)?;
    let elements = source.as_array().size();
    if elements != 0
        && source
            .as_array()
            .allocation_info()
            .map_err(Error::backend_retained_source)?
            .is_none()
    {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    let payload = elements
        .checked_mul(size_of::<T>())
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    // Arc<u8> uses the qualified two-word Arc header and the aligned u8 body.
    let key_allocation = size_of::<usize>()
        .checked_mul(3)
        .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    let controls = [
        EvaluatedArray::completed_mapped_readback_control_bytes::<S, T>()
            .ok_or_else(|| memory(WorkingMemoryError::UnknownBound))?,
        key_allocation,
        size_of::<ReadbackKey>(),
        size_of::<Vec<T>>(),
        size_of::<HostTensorBuffer<T>>(),
        size_of::<eredu_core::HostTensorBufferIntoIter<T>>(),
        HostTensorBuffer::<T>::custody_allocation_bytes::<OutputCustody>(),
        size_of::<Result<HostTensorBuffer<T>, Error>>(),
        size_of::<Result<(), std::collections::TryReserveError>>(),
    ]
    .into_iter()
    .try_fold(0usize, usize::checked_add)
    .and_then(|bytes| u64::try_from(bytes).ok())
    .ok_or_else(|| memory(WorkingMemoryError::Overflow))?;
    // Preparation owns every descriptor/key/control before it allocates. The
    // following single physical grant respects all live request domain ceilings.
    let prepared = StoragePublicationLayout::<ReadbackKey>::pending(1)
        .and_then(|layout| layout.with_additional_host_metadata(controls))
        .and_then(|layout| layout.fund(&pool))
        .map_err(memory)?;
    let key = ReadbackKey(Arc::new(0));
    let mut custody = prepared
        .reserve_storage([(
            key,
            StorageAllocation::new(payload, pool.host_placement_handle()),
        )])
        .map_err(memory)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(elements)
        .map_err(Error::backend_retained_source)?;
    values.resize(elements, T::default());
    // The qualified global allocator exposes the exact requested Vec capacity.
    if values.capacity() != elements {
        return Err(memory(WorkingMemoryError::UnknownBound));
    }
    custody.publish().map_err(memory)?;
    if source.as_array().dtype() == T::DTYPE {
        source
            .try_copy_into(&mut values)
            .map_err(Error::backend_retained_source)?;
    } else {
        source
            .try_map_into(&mut values, map)
            .map_err(Error::backend_retained_source)?;
    }
    Ok(HostTensorBuffer::new(values, custody))
}

pub(super) fn export_f32(source: &EvaluatedArray<'_>) -> Result<HostTensorBuffer<f32>, Error> {
    match source.as_array().dtype() {
        Dtype::Bool => export_as::<bool, f32>(source, |value| u8::from(value) as f32),
        Dtype::Uint8 => export_as::<u8, f32>(source, |value| value as f32),
        Dtype::Uint16 => export_as::<u16, f32>(source, |value| value as f32),
        Dtype::Uint32 => export_as::<u32, f32>(source, |value| value as f32),
        Dtype::Uint64 => export_as::<u64, f32>(source, |value| value as f32),
        Dtype::Int8 => export_as::<i8, f32>(source, |value| value as f32),
        Dtype::Int16 => export_as::<i16, f32>(source, |value| value as f32),
        Dtype::Int32 => export_as::<i32, f32>(source, |value| value as f32),
        Dtype::Int64 => export_as::<i64, f32>(source, |value| value as f32),
        Dtype::Float16 => export_as::<half::f16, f32>(source, |value| value.to_f32() as f32),
        Dtype::Bfloat16 => export_as::<half::bf16, f32>(source, |value| value.to_f32() as f32),
        Dtype::Float32 => export_as::<f32, f32>(source, |value| value as f32),
        Dtype::Float64 => export_as::<f64, f32>(source, |value| value as f32),
        Dtype::Complex64 => export_as::<safemlx::complex64, f32>(source, |value| value.re as f32),
    }
}

pub(super) fn export_i32(source: &EvaluatedArray<'_>) -> Result<HostTensorBuffer<i32>, Error> {
    match source.as_array().dtype() {
        Dtype::Bool => export_as::<bool, i32>(source, |value| u8::from(value) as i32),
        Dtype::Uint8 => export_as::<u8, i32>(source, |value| value as i32),
        Dtype::Uint16 => export_as::<u16, i32>(source, |value| value as i32),
        Dtype::Uint32 => export_as::<u32, i32>(source, |value| value as i32),
        Dtype::Uint64 => export_as::<u64, i32>(source, |value| value as i32),
        Dtype::Int8 => export_as::<i8, i32>(source, |value| value as i32),
        Dtype::Int16 => export_as::<i16, i32>(source, |value| value as i32),
        Dtype::Int32 => export_as::<i32, i32>(source, |value| value as i32),
        Dtype::Int64 => export_as::<i64, i32>(source, |value| value as i32),
        Dtype::Float16 => export_as::<half::f16, i32>(source, |value| value.to_f32() as i32),
        Dtype::Bfloat16 => export_as::<half::bf16, i32>(source, |value| value.to_f32() as i32),
        Dtype::Float32 => export_as::<f32, i32>(source, |value| value as i32),
        Dtype::Float64 => export_as::<f64, i32>(source, |value| value as i32),
        Dtype::Complex64 => export_as::<safemlx::complex64, i32>(source, |value| value.re as i32),
    }
}
