//! The same publication record with ordinary or source-paid metadata backing.
use super::*;
use eredu_nn::workspace::{WorkspaceContext, WorkspaceMetadataFunding};
use std::mem::size_of;

/// Metadata only. It cannot publish a block, enter a native scope or authorize
/// materialization. A prepared caller must authenticate its actual arrays and
/// exact source/role before transferring it to the canonical manager record.
pub(crate) struct CacheBlockMetadata {
    representation: CacheRepresentation,
    shapes: [Vec<i32>; 2],
    dtypes: [String; 2],
    native_dtypes: [Dtype; 2],
    bytes: u64,
    funding: Option<WorkspaceMetadataFunding>,
}
impl CacheBlockMetadata {
    pub(super) fn ordinary(arrays: &CacheBlockArrays) -> Self {
        let roots = arrays.arrays();
        Self {
            representation: arrays.representation(),
            shapes: arrays.shapes(),
            dtypes: arrays.dtypes(),
            native_dtypes: [roots[0].dtype(), roots[1].dtype()],
            bytes: arrays.bytes(),
            funding: None,
        }
    }
    pub(crate) fn floating_dtype_bytes(dtype: Dtype) -> Option<u64> {
        match dtype {
            Dtype::Float32 => Some(4),
            Dtype::Float16 | Dtype::Bfloat16 => Some(2),
            _ => None,
        }
    }
    pub(crate) fn floating_bytes(
        shapes: [&[i32]; 2],
        dtypes: [Dtype; 2],
    ) -> Result<u64, CacheSourceError> {
        let mut bytes = 0u64;
        for (shape, dtype) in shapes.into_iter().zip(dtypes) {
            let width = Self::floating_dtype_bytes(dtype).ok_or(CacheSourceError::Geometry)?;
            if shape.is_empty() || shape.iter().any(|dimension| *dimension <= 0) {
                return Err(CacheSourceError::Geometry);
            }
            let one = shape
                .iter()
                .try_fold(width, |bytes, dimension| {
                    bytes.checked_mul(*dimension as u64)
                })
                .ok_or(CacheSourceError::Overflow)?;
            bytes = bytes.checked_add(one).ok_or(CacheSourceError::Overflow)?;
        }
        Ok(bytes)
    }
    pub(crate) fn f32_control_bytes(shapes: [&[i32]; 2]) -> Option<usize> {
        Self::floating_control_bytes(shapes, [Dtype::Float32; 2])
    }
    pub(crate) fn floating_control_bytes(shapes: [&[i32]; 2], dtypes: [Dtype; 2]) -> Option<usize> {
        struct Length(usize);
        impl std::fmt::Write for Length {
            fn write_str(&mut self, value: &str) -> std::fmt::Result {
                self.0 = self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;
                Ok(())
            }
        }
        let mut bytes = Self::fixed_controls()?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<i32>(
                shapes[0].len(),
            )?)?
            .checked_add(WorkspaceContext::metadata_vec_bytes::<i32>(
                shapes[1].len(),
            )?)?;
        for dtype in dtypes {
            Self::floating_dtype_bytes(dtype)?;
            let mut length = Length(0);
            std::fmt::write(&mut length, format_args!("{:?}", dtype)).ok()?;
            bytes = bytes.checked_add(WorkspaceContext::metadata_string_bytes(length.0)?)?;
        }
        bytes.checked_add(size_of::<(Length, std::fmt::Arguments<'_>, Dtype, usize)>())
    }
    pub(crate) fn prepare_f32(
        representation: CacheRepresentation,
        shapes: [&[i32]; 2],
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        Self::prepare_floating(representation, shapes, [Dtype::Float32; 2], context)
    }
    /// Copies the actual floating source declaration. No conversion is applied
    /// to source or destination; the shared native copy worker preserves bits.
    pub(crate) fn prepare_floating(
        representation: CacheRepresentation,
        shapes: [&[i32]; 2],
        dtypes: [Dtype; 2],
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        let failure = |cause| CacheSourceFailure::source(cause, context);
        context
            .charge_metadata(
                Self::fixed_controls().ok_or_else(|| failure(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        let bytes = Self::floating_bytes(shapes, dtypes).map_err(failure)?;
        let mut first = context
            .metadata_vec(shapes[0].len())
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        first.extend_from_slice(shapes[0]);
        let mut second = context
            .metadata_vec(shapes[1].len())
            .map_err(|cause| CacheSourceFailure::metadata(cause, context))?;
        second.extend_from_slice(shapes[1]);
        let names = [
            context
                .metadata_string(format_args!("{:?}", dtypes[0]))
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
            context
                .metadata_string(format_args!("{:?}", dtypes[1]))
                .map_err(|cause| CacheSourceFailure::metadata(cause, context))?,
        ];
        Ok(Self {
            representation,
            shapes: [first, second],
            dtypes: names,
            native_dtypes: dtypes,
            bytes,
            funding: context.metadata_funding(),
        })
    }
    pub(crate) fn validate_arrays(
        &self,
        arrays: &CacheBlockArrays,
    ) -> Result<(), CacheSourceError> {
        if arrays.representation() != self.representation {
            return Err(CacheSourceError::Geometry);
        }
        for (index, array) in arrays.arrays().into_iter().enumerate() {
            if array.shape() != self.shapes[index].as_slice()
                || array.dtype() != self.native_dtypes[index]
            {
                return Err(CacheSourceError::Geometry);
            }
        }
        if arrays.bytes() != self.bytes {
            return Err(CacheSourceError::Geometry);
        }
        Ok(())
    }
    /// Caller validated the actual source/role and arrays before moving these
    /// buffers. Physical arrays and metadata retire before the paying account.
    pub(super) fn into_record(
        self,
        id: CacheBlockId,
        arrays: CacheBlockArrays,
        imported: bool,
    ) -> CacheBlockRecord {
        self.into_storage_record(MlxCacheBlockStorage::device(id, arrays, None), imported)
    }
    pub(super) fn into_host_record(
        self,
        id: CacheBlockId,
        host: HostCacheBlock,
        imported: bool,
    ) -> CacheBlockRecord {
        self.into_storage_record(MlxCacheBlockStorage::host(id, host, None), imported)
    }
    /// Same canonical record for a validated physical source. The caller owns
    /// source/payload validation and retains rejected storage until unlock.
    pub(super) fn into_storage_record(
        self,
        physical: MlxCacheBlockStorage,
        imported: bool,
    ) -> CacheBlockRecord {
        CacheBlockRecord {
            physical,
            bytes: self.bytes,
            shapes: self.shapes,
            dtypes: self.dtypes,
            imported,
            original_discard: None,
            _metadata_funding: self.funding,
        }
    }
    pub(super) fn validate_host(&self, host: &HostCacheBlock) -> Result<(), CacheSourceError> {
        if host.representation() != self.representation {
            return Err(CacheSourceError::Geometry);
        }
        let mut bytes = 0u64;
        for (index, buffer) in host.buffers().into_iter().enumerate() {
            let descriptor = buffer
                .try_fixed_descriptor::<4>()
                .map_err(CacheSourceError::HostDescriptor)?;
            if descriptor.shape() != self.shapes[index].as_slice()
                || descriptor.dtype() != self.native_dtypes[index]
            {
                return Err(CacheSourceError::Geometry);
            }
            bytes = bytes
                .checked_add(
                    u64::try_from(descriptor.nbytes()).map_err(|_| CacheSourceError::Overflow)?,
                )
                .ok_or(CacheSourceError::Overflow)?;
        }
        if bytes != self.bytes {
            return Err(CacheSourceError::Geometry);
        }
        Ok(())
    }
    pub(crate) fn fixed_controls() -> Option<usize> {
        let frames = [
            size_of::<Self>(),
            size_of::<CacheBlockRecord>(),
            size_of::<CacheBlockArrays>(),
            size_of::<HostCacheBlock>(),
            safemlx::HostTransferDescriptor::<4>::control_bytes()?.checked_mul(2)?,
            size_of::<[&Array; 2]>(),
            size_of::<[&[i32]; 2]>(),
            size_of::<[Dtype; 2]>(),
            size_of::<[Vec<i32>; 2]>(),
            size_of::<[String; 2]>(),
            size_of::<(&WorkspaceContext, CacheRepresentation)>(),
            size_of::<(usize, u64)>(),
            size_of::<
                std::iter::Zip<std::array::IntoIter<&[i32], 2>, std::array::IntoIter<Dtype, 2>>,
            >(),
            size_of::<(
                CacheRepresentation,
                [&[i32]; 2],
                [Dtype; 2],
                &WorkspaceContext,
            )>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<std::slice::Iter<'_, i32>>(),
            size_of::<Option<u64>>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<(), CacheSourceError>>(),
            size_of::<([&[i32]; 2], [Dtype; 2], Result<u64, CacheSourceError>)>(),
        ];
        frames
            .into_iter()
            .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
    }
}

/// Paid destination for the finite native floating declarations. It is not a
/// publishable record until actual source-qualified output arrays bind it.
/// Empty cache geometry does not select a native dtype.
pub(crate) struct PreparedFloatingBlockMetadata {
    metadata: CacheBlockMetadata,
}
struct FloatingTextLength(usize);
impl std::fmt::Write for FloatingTextLength {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        self.0 = self.0.checked_add(value.len()).ok_or(std::fmt::Error)?;
        Ok(())
    }
}
struct FloatingTextOutput<'a>(&'a mut String);
impl std::fmt::Write for FloatingTextOutput<'_> {
    fn write_str(&mut self, value: &str) -> std::fmt::Result {
        if value.len() > self.0.capacity() - self.0.len() {
            return Err(std::fmt::Error);
        }
        self.0.push_str(value);
        Ok(())
    }
}
impl PreparedFloatingBlockMetadata {
    pub(crate) fn prepare(
        representation: CacheRepresentation,
        shapes: [&[i32]; 2],
        context: &WorkspaceContext,
    ) -> Result<Self, CacheSourceFailure> {
        let fail = |cause| CacheSourceFailure::source(cause, context);
        let frames = [
            size_of::<Self>(),
            size_of::<CacheBlockMetadata>(),
            size_of::<FloatingTextLength>(),
            size_of::<FloatingTextOutput<'_>>(),
            size_of::<std::fmt::Arguments<'_>>(),
            size_of::<std::fmt::Error>(),
            size_of::<(Dtype, usize, usize)>(),
            size_of::<[Dtype; 3]>(),
            size_of::<std::array::IntoIter<Dtype, 3>>(),
            size_of::<std::iter::Enumerate<std::array::IntoIter<&Array, 2>>>(),
            size_of::<(&Self, &CacheBlockArrays, [Dtype; 2], u64)>(),
            size_of::<(&WorkspaceContext, CacheRepresentation, [&[i32]; 2])>(),
            size_of::<Result<Self, CacheSourceFailure>>(),
            size_of::<Result<CacheBlockMetadata, CacheSourceError>>(),
            size_of::<Result<(), CacheSourceError>>(),
            // Two bound-output validations followed by final record validation.
            CacheBlockMetadata::fixed_controls()
                .and_then(|n| n.checked_mul(3))
                .ok_or_else(|| fail(CacheSourceError::Overflow))?,
        ];
        context
            .charge_metadata(
                frames
                    .into_iter()
                    .try_fold(std::mem::size_of_val(&frames), usize::checked_add)
                    .ok_or_else(|| fail(CacheSourceError::Overflow))?,
            )
            .map_err(|cause| CacheSourceFailure::metadata(cause.into(), context))?;
        // Measure the actual shared Debug formatter; no spelling or byte-width
        // guess determines which paid text destination is largest.
        let mut largest = Dtype::Float32;
        let mut maximum = 0;
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            let mut length = FloatingTextLength(0);
            std::fmt::write(&mut length, format_args!("{:?}", dtype))
                .map_err(|_| fail(CacheSourceError::Overflow))?;
            if length.0 > maximum {
                largest = dtype;
                maximum = length.0;
            }
        }
        Ok(Self {
            metadata: CacheBlockMetadata::prepare_floating(
                representation,
                shapes,
                [largest; 2],
                context,
            )?,
        })
    }
    pub(crate) fn validate_arrays(
        &self,
        arrays: &CacheBlockArrays,
    ) -> Result<(), CacheSourceError> {
        if arrays.representation() != self.metadata.representation {
            return Err(CacheSourceError::Geometry);
        }
        let roots = arrays.arrays();
        for (index, array) in roots.into_iter().enumerate() {
            if array.shape() != self.metadata.shapes[index].as_slice()
                || CacheBlockMetadata::floating_dtype_bytes(array.dtype()).is_none()
            {
                return Err(CacheSourceError::Geometry);
            }
            let mut length = FloatingTextLength(0);
            std::fmt::write(&mut length, format_args!("{:?}", array.dtype()))
                .map_err(|_| CacheSourceError::Overflow)?;
            if length.0 > self.metadata.dtypes[index].capacity() {
                return Err(CacheSourceError::Geometry);
            }
        }
        let bytes = CacheBlockMetadata::floating_bytes(
            [&self.metadata.shapes[0], &self.metadata.shapes[1]],
            [roots[0].dtype(), roots[1].dtype()],
        )?;
        if bytes != arrays.bytes() {
            return Err(CacheSourceError::Geometry);
        }
        Ok(())
    }
    pub(crate) fn bind(
        mut self,
        arrays: &CacheBlockArrays,
    ) -> Result<CacheBlockMetadata, CacheSourceError> {
        self.validate_arrays(arrays)?;
        let roots = arrays.arrays();
        for (index, array) in roots.into_iter().enumerate() {
            let text = &mut self.metadata.dtypes[index];
            text.clear();
            std::fmt::write(
                &mut FloatingTextOutput(text),
                format_args!("{:?}", array.dtype()),
            )
            .map_err(|_| CacheSourceError::Geometry)?;
            self.metadata.native_dtypes[index] = array.dtype();
        }
        self.metadata.bytes = arrays.bytes();
        self.metadata.validate_arrays(arrays)?;
        Ok(self.metadata)
    }
}

/// Canonical insertion shared by ordinary and explicitly prepared callers.
/// An unaccepted native record is returned intact for retirement after unlock.
pub(super) fn insert_record(
    state: &mut CacheManagerState,
    id: CacheBlockId,
    record: CacheBlockRecord,
    protected_prefix: bool,
    prepared: bool,
) -> Result<(), (CacheResidencyError, CacheBlockRecord)> {
    if state.blocks.contains_key(&id) {
        return Err((CacheLifecycleError::DuplicateBlock(id).into(), record));
    }
    if prepared {
        let population = match state.blocks.len().checked_add(1) {
            Some(value) => value,
            None => {
                return Err((
                    CacheLifecycleError::from(
                        eredu_runtime::cache::CacheTableCapacityError::Exhausted,
                    )
                    .into(),
                    record,
                ));
            }
        };
        if let Err(cause) = state.blocks.validate_prepared_population(population) {
            return Err((CacheLifecycleError::from(cause).into(), record));
        }
    }
    let inserted = if prepared {
        state
            .lifecycle
            .insert_prepared(id.clone(), protected_prefix)
    } else {
        state.lifecycle.insert(id.clone(), protected_prefix)
    };
    if let Err(cause) = inserted {
        return Err((cause.into(), record));
    }
    // Same guard and validated capacity; no allocation can occur here when prepared.
    let prior = state.blocks.insert(id, record);
    debug_assert!(prior.is_none());
    Ok(())
}
