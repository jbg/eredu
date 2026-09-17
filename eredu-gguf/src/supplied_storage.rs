//! Owning initialized storage for the existing GGUF conversion workers.
use crate::{ConversionDestinationError, MetadataDestinationError};
use std::{fmt, marker::PhantomData};

/// An owning initialized buffer. Its allocation custody must stay inside this
/// value until all of its elements and allocation have been destroyed.
pub trait InitializedStorage<T>: AsRef<[T]> + AsMut<[T]> + fmt::Debug + 'static {
    /// Actual exposed element capacity, not an allocation grant.
    fn capacity(&self) -> usize;
}

/// Types owned by a supplied conversion, independent of the provider's borrow.
pub trait StorageFamily: fmt::Debug + 'static {
    /// The intact allocation owner for this element type.
    type Buffer<T: Copy + fmt::Debug + 'static>: InitializedStorage<T>;
    /// Actual allocation/funding cause; successful prefixes remain in buffers.
    type Error: std::error::Error + 'static;
}

/// Storage realization only; never source selection, cache or conversion policy.
pub trait StorageProvider {
    /// Owning types that can outlive this mutable provider loan.
    type Family: StorageFamily;
    /// Prepare exactly one initialized destination at the reached reserve point.
    /// A failure returns any allocated prefix, even when its exposed extent is
    /// wrong. No successful allocation may be detached from its custody.
    fn prepare<T: Copy + fmt::Debug + 'static>(
        &mut self,
        elements: usize,
        initializer: T,
    ) -> Result<
        <Self::Family as StorageFamily>::Buffer<T>,
        (
            <Self::Family as StorageFamily>::Error,
            Option<<Self::Family as StorageFamily>::Buffer<T>>,
        ),
    >;
}

/// Typed provider or shared-worker refusal. The surrounding destination owns
/// the exact successful/failed storage prefix; this cause does not clone it.
#[derive(Debug)]
pub enum SuppliedStorageError<E> {
    /// Actual provider error, without formatting/erasing its source.
    Provider(E),
    /// The provider returned an extent that does not match the requested slot.
    Extent {
        /// Requested initialized elements.
        expected: usize,
        /// Actual initialized elements.
        actual: usize,
        /// Actual exposed capacity.
        capacity: usize,
    },
    /// Existing conversion validation or checked write refusal.
    Conversion(ConversionDestinationError),
    /// Existing metadata/physical-plan validation or checked write refusal.
    Metadata(MetadataDestinationError),
}
impl<E: fmt::Display> fmt::Display for SuppliedStorageError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Provider(e) => e.fmt(f),
            Self::Conversion(e) => e.fmt(f),
            Self::Metadata(e) => e.fmt(f),
            Self::Extent { expected, actual, capacity } => write!(
                f,
                "GGUF supplied buffer expected {expected} initialized elements, got {actual} with capacity {capacity}"
            ),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for SuppliedStorageError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Provider(e) => Some(e),
            Self::Conversion(e) => Some(e),
            Self::Metadata(e) => Some(e),
            Self::Extent { .. } => None,
        }
    }
}
impl<E> From<ConversionDestinationError> for SuppliedStorageError<E> {
    fn from(e: ConversionDestinationError) -> Self {
        Self::Conversion(e)
    }
}
impl<E> From<MetadataDestinationError> for SuppliedStorageError<E> {
    fn from(e: MetadataDestinationError) -> Self {
        Self::Metadata(e)
    }
}

/// Intact initialized allocation plus the shared worker's emitted prefix.
/// Initialized spare elements are not reported as converted output.
#[derive(Debug)]
pub struct StoredBuffer<F: StorageFamily, T: Copy + fmt::Debug + 'static> {
    storage: Option<F::Buffer<T>>,
    used: usize,
    limit: usize,
    _element: PhantomData<T>,
}
impl<F: StorageFamily, T: Copy + fmt::Debug + 'static> Default for StoredBuffer<F, T> {
    fn default() -> Self {
        Self {
            storage: None,
            used: 0,
            limit: 0,
            _element: PhantomData,
        }
    }
}
impl<F: StorageFamily, T: Copy + fmt::Debug + 'static> StoredBuffer<F, T> {
    /// Copy one actual metadata slice into supplied storage. A refusal returns
    /// the exact attempted owner, including an extent-mismatched allocation.
    /// The provider owns admission; slice length is not allocator qualification.
    pub fn try_copy<P: StorageProvider<Family = F>>(
        provider: &mut P,
        values: &[T],
        initializer: T,
    ) -> Result<Self, (SuppliedStorageError<F::Error>, Self)> {
        let mut out = Self::default();
        let result = out
            .prepare(provider, values.len(), initializer)
            .and_then(|()| {
                out.copy_metadata(values, "supplied metadata copy")
                    .map_err(Into::into)
            });
        match result {
            Ok(()) => Ok(out),
            Err(cause) => Err((cause, out)),
        }
    }

    /// Named caller/worker transports for `try_copy`, separate from the
    /// provider's allocation/implementation and the supplied backing extent.
    /// This describes managed values, not compiler ABI/process stack usage.
    pub fn copy_control_bytes() -> Option<usize> {
        [
            std::mem::size_of::<Self>(),
            std::mem::size_of::<Result<Self, (SuppliedStorageError<F::Error>, Self)>>(),
            std::mem::size_of::<Result<(), SuppliedStorageError<F::Error>>>(),
            std::mem::size_of::<Result<F::Buffer<T>, (F::Error, Option<F::Buffer<T>>)>>(),
            std::mem::size_of::<F::Buffer<T>>(),
            std::mem::size_of::<SuppliedStorageError<F::Error>>(),
            std::mem::size_of::<(&mut Self, &[T], &mut (), T, usize, usize, usize)>(),
        ]
        .into_iter()
        .try_fold(0usize, usize::checked_add)
    }

    /// Move the intact allocation owner and its checked emitted prefix. This
    /// never exposes a growing Vec or separates an allocation from its custody.
    pub fn into_storage(self) -> (Option<F::Buffer<T>>, usize, usize) {
        (self.storage, self.used, self.limit)
    }

    pub(crate) fn prepare<P: StorageProvider<Family = F>>(
        &mut self,
        provider: &mut P,
        count: usize,
        initial: T,
    ) -> Result<(), SuppliedStorageError<F::Error>> {
        debug_assert!(self.storage.is_none());
        self.limit = count;
        match provider.prepare(count, initial) {
            Ok(storage) => self.storage = Some(storage),
            Err((cause, storage)) => {
                self.storage = storage;
                return Err(SuppliedStorageError::Provider(cause));
            }
        }
        let storage = self.storage.as_ref().expect("successful supplied storage");
        if storage.as_ref().len() != count || storage.capacity() < count {
            return Err(SuppliedStorageError::Extent {
                expected: count,
                actual: storage.as_ref().len(),
                capacity: storage.capacity(),
            });
        }
        Ok(())
    }
    /// Shared worker's initialized emitted prefix.
    pub fn as_slice(&self) -> &[T] {
        match &self.storage {
            Some(storage) => &storage.as_ref()[..self.used],
            None => &[],
        }
    }
    /// Actual backing capacity, separate from emitted length.
    pub fn capacity(&self) -> usize {
        self.storage
            .as_ref()
            .map_or(0, InitializedStorage::capacity)
    }
    /// Admitted requested element limit.
    pub fn requested_elements(&self) -> usize {
        self.limit
    }
    /// Emitted element count, not the fully initialized allocation length.
    pub fn len(&self) -> usize {
        self.used
    }
    /// Whether no elements have been emitted.
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }
    pub(crate) fn loan(&mut self) -> FixedSlice<'_, T> {
        let data = self.storage.as_mut().map_or(&mut [][..], |s| s.as_mut());
        FixedSlice {
            data,
            used: &mut self.used,
            limit: self.limit,
        }
    }
    pub(crate) fn prepared(&self) -> bool {
        self.storage.is_some()
    }
    pub(crate) fn emitted_mut(&mut self) -> &mut [T] {
        match self.storage.as_mut() {
            Some(s) => &mut s.as_mut()[..self.used],
            None => &mut [],
        }
    }
    pub(crate) fn replace_reversed(
        &mut self,
        values: &[T],
        field: &'static str,
    ) -> Result<(), MetadataDestinationError> {
        self.check_metadata(values.len(), field)?;
        if let Some(storage) = &mut self.storage {
            for (out, value) in storage.as_mut().iter_mut().zip(values.iter().rev()) {
                *out = *value;
            }
        }
        self.used = values.len();
        Ok(())
    }
    pub(crate) fn check_metadata(
        &self,
        count: usize,
        field: &'static str,
    ) -> Result<(), MetadataDestinationError> {
        let available = self
            .storage
            .as_ref()
            .map_or(0, |s| s.as_ref().len())
            .min(self.limit);
        if count > available {
            return Err(MetadataDestinationError::Capacity {
                field,
                required: count,
                capacity: available,
            });
        }
        Ok(())
    }
    pub(crate) fn copy_metadata(
        &mut self,
        values: &[T],
        field: &'static str,
    ) -> Result<(), MetadataDestinationError> {
        self.check_metadata(values.len(), field)?;
        if let Some(storage) = &mut self.storage {
            storage.as_mut()[..values.len()].copy_from_slice(values);
        }
        self.used = values.len();
        Ok(())
    }
}
impl<F: StorageFamily, T: Copy + fmt::Debug + 'static> AsRef<[T]> for StoredBuffer<F, T> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

/// Borrowed fixed initialized storage. All writes use its checked logical length.
pub(crate) struct FixedSlice<'a, T> {
    pub(crate) data: &'a mut [T],
    pub(crate) used: &'a mut usize,
    pub(crate) limit: usize,
}
impl<T> FixedSlice<'_, T> {
    pub(crate) fn check(&self, required: usize) -> Result<(), ConversionDestinationError> {
        if required > self.limit || required > self.data.len() {
            return Err(ConversionDestinationError::Capacity {
                required,
                limit: self.limit,
                capacity: self.data.len(),
            });
        }
        Ok(())
    }
    pub(crate) fn clear(&mut self) {
        *self.used = 0;
    }
    pub(crate) fn push(&mut self, value: T) -> Result<(), ConversionDestinationError> {
        let n = self
            .used
            .checked_add(1)
            .ok_or(ConversionDestinationError::Layout)?;
        self.check(n)?;
        self.data[*self.used] = value;
        *self.used = n;
        Ok(())
    }
    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data[..*self.used]
    }
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..*self.used]
    }
}
impl<T: Copy> FixedSlice<'_, T> {
    pub(crate) fn resize(
        &mut self,
        len: usize,
        value: T,
    ) -> Result<(), ConversionDestinationError> {
        self.check(len)?;
        if len > *self.used {
            self.data[*self.used..len].fill(value);
        }
        *self.used = len;
        Ok(())
    }
}

/// Descriptor with supplied name/dimension owners. Borrowed views cannot outlive
/// these allocations, including a late failed materialization's descriptor.
#[derive(Debug)]
pub struct StoredDescriptor<F: StorageFamily> {
    pub(crate) name: StoredBuffer<F, u8>,
    pub(crate) dimensions: StoredBuffer<F, u64>,
    pub(crate) ggml_type: crate::GgmlType,
    pub(crate) relative_offset: u64,
    pub(crate) data_offset: u64,
    pub(crate) byte_len: u64,
}
impl<F: StorageFamily> Default for StoredDescriptor<F> {
    fn default() -> Self {
        Self {
            name: StoredBuffer::default(),
            dimensions: StoredBuffer::default(),
            ggml_type: crate::GgmlType::F32,
            relative_offset: 0,
            data_offset: 0,
            byte_len: 0,
        }
    }
}
impl<F: StorageFamily> StoredDescriptor<F> {
    pub(crate) fn prepare<P: StorageProvider<Family = F>>(
        &mut self,
        provider: &mut P,
        source: crate::TensorDescriptorView<'_>,
        rank: usize,
    ) -> Result<(), SuppliedStorageError<F::Error>> {
        self.reserve_only(provider, source.name, rank)?;
        self.copy_from(source).map_err(Into::into)
    }
    pub(crate) fn reserve_only<P: StorageProvider<Family = F>>(
        &mut self,
        provider: &mut P,
        name: &str,
        rank: usize,
    ) -> Result<(), SuppliedStorageError<F::Error>> {
        self.name.prepare(provider, name.len(), 0)?;
        self.name
            .copy_metadata(name.as_bytes(), "descriptor name")?;
        self.dimensions.prepare(provider, rank, 0)
    }
    pub(crate) fn copy_from(
        &mut self,
        source: crate::TensorDescriptorView<'_>,
    ) -> Result<(), MetadataDestinationError> {
        self.name
            .check_metadata(source.name.len(), "descriptor name")?;
        self.dimensions
            .check_metadata(source.dimensions.len(), "descriptor dimensions")?;
        self.name
            .copy_metadata(source.name.as_bytes(), "descriptor name")?;
        self.dimensions
            .copy_metadata(source.dimensions, "descriptor dimensions")?;
        self.ggml_type = source.ggml_type;
        self.relative_offset = source.relative_offset;
        self.data_offset = source.data_offset;
        self.byte_len = source.byte_len;
        Ok(())
    }
    /// Exact immutable borrowed fields; no String or Vec is reconstructed.
    pub fn view(&self) -> crate::TensorDescriptorView<'_> {
        crate::TensorDescriptorView {
            name: std::str::from_utf8(self.name.as_slice())
                .expect("descriptor copied from actual UTF-8 source"),
            dimensions: self.dimensions.as_slice(),
            ggml_type: self.ggml_type,
            relative_offset: self.relative_offset,
            data_offset: self.data_offset,
            byte_len: self.byte_len,
        }
    }
    pub(crate) fn loan(&mut self) -> FixedDescriptor<'_> {
        FixedDescriptor {
            name: self.name.loan(),
            dimensions: self.dimensions.loan(),
            ggml_type: &mut self.ggml_type,
            relative_offset: &mut self.relative_offset,
            data_offset: &mut self.data_offset,
            byte_len: &mut self.byte_len,
        }
    }
}

pub(crate) struct FixedDescriptor<'a> {
    pub(crate) name: FixedSlice<'a, u8>,
    pub(crate) dimensions: FixedSlice<'a, u64>,
    pub(crate) ggml_type: &'a mut crate::GgmlType,
    pub(crate) relative_offset: &'a mut u64,
    pub(crate) data_offset: &'a mut u64,
    pub(crate) byte_len: &'a mut u64,
}
impl FixedDescriptor<'_> {
    pub(crate) fn view(&self) -> crate::TensorDescriptorView<'_> {
        crate::TensorDescriptorView {
            name: std::str::from_utf8(self.name.as_slice()).expect("fixed descriptor UTF-8"),
            dimensions: self.dimensions.as_slice(),
            ggml_type: *self.ggml_type,
            relative_offset: *self.relative_offset,
            data_offset: *self.data_offset,
            byte_len: *self.byte_len,
        }
    }
    pub(crate) fn copy_from(
        &mut self,
        source: crate::TensorDescriptorView<'_>,
    ) -> Result<(), MetadataDestinationError> {
        metadata_capacity(
            "descriptor name",
            source.name.len(),
            self.name.limit.min(self.name.data.len()),
        )?;
        metadata_capacity(
            "descriptor dimensions",
            source.dimensions.len(),
            self.dimensions.limit.min(self.dimensions.data.len()),
        )?;
        self.name.data[..source.name.len()].copy_from_slice(source.name.as_bytes());
        *self.name.used = source.name.len();
        self.dimensions.data[..source.dimensions.len()].copy_from_slice(source.dimensions);
        *self.dimensions.used = source.dimensions.len();
        *self.ggml_type = source.ggml_type;
        *self.relative_offset = source.relative_offset;
        *self.data_offset = source.data_offset;
        *self.byte_len = source.byte_len;
        Ok(())
    }
}
pub(crate) fn metadata_capacity(
    field: &'static str,
    required: usize,
    capacity: usize,
) -> Result<(), MetadataDestinationError> {
    if required > capacity {
        Err(MetadataDestinationError::Capacity {
            field,
            required,
            capacity,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests;

mod requests;
pub use requests::StorageRequestBound;
