//! G1/G2/G3 supplied storage over the original source/cache/materializer driver.
use super::*;
use eredu_gguf::{
    InitializedStorage, MetadataDestinationError, MetadataSelection, StorageFamily,
    StorageProvider, StoredCheckpointTensor, StoredConversion, StoredPhysicalDescriptor,
    StoredTensorMetadata, StoredTensorPair, SuppliedStorageError,
};

/// One exact source authorization followed by the existing reached storage requests.
/// The same provider realizes G1 and all subsequent typed G2/G3 buffers. It does
/// not choose a selection, format, reader or conversion policy.
pub trait GgufStorageProvider: StorageProvider {
    /// Authorize this actual consumed lease and prepare its one initialized G1 buffer.
    /// This runs only after the original checked raw layout, before G2 planning.
    fn prepare_raw(
        &mut self,
        request: GgufRawStorageRequest<'_>,
    ) -> Result<
        <Self::Family as StorageFamily>::Buffer<u8>,
        (
            <Self::Family as StorageFamily>::Error,
            Option<<Self::Family as StorageFamily>::Buffer<u8>>,
        ),
    >;
}
#[derive(Debug)]
enum StoredCause<E> {
    Storage(SuppliedStorageError<E>),
    Original(Cause),
}
impl<E: std::error::Error + 'static> std::fmt::Display for StoredCause<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Storage(e) => e.fmt(f),
            Self::Original(e) => e.fmt(f),
        }
    }
}
impl<E: std::error::Error + 'static> std::error::Error for StoredCause<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Storage(e) => e,
            Self::Original(e) => e,
        })
    }
}
impl<E> From<Cause> for StoredCause<E> {
    fn from(e: Cause) -> Self {
        Self::Original(e)
    }
}
impl<E> From<StoreError> for StoredCause<E> {
    fn from(e: StoreError) -> Self {
        Self::Original(Cause::Store(e))
    }
}
impl<E> From<SuppliedStorageError<E>> for StoredCause<E> {
    fn from(e: SuppliedStorageError<E>) -> Self {
        Self::Storage(e)
    }
}

#[derive(Debug)]
enum StoredOutputs<F: StorageFamily> {
    Preparing {
        metadata: Option<StoredTensorMetadata<F>>,
        conversion: Option<StoredConversion<F>>,
    },
    Paired(StoredTensorPair<F>),
}
impl<F: StorageFamily> Default for StoredOutputs<F> {
    fn default() -> Self {
        Self::Preparing {
            metadata: None,
            conversion: None,
        }
    }
}
impl<F: StorageFamily> StoredOutputs<F> {
    fn conversion_completed(&self) -> bool {
        match self {
            Self::Preparing { conversion, .. } => {
                conversion.as_ref().is_some_and(StoredConversion::completed)
            }
            Self::Paired(pair) => pair.conversion_completed(),
        }
    }
}

// Field order is the actual retirement order, including panic unwinding through
// borrowed read/conversion work. The same Box remains last throughout the call.
#[derive(Debug)]
struct StoredOwners<F: StorageFamily> {
    outputs: StoredOutputs<F>,
    physical: Option<StoredPhysicalDescriptor<F>>,
    raw: Option<F::Buffer<u8>>,
    lease: Box<GgufLease>,
}
/// Actual source/provider/read failure retaining every reached G1/G2/G3 owner.
/// A late metadata failure keeps the completed conversion in the same owner.
#[derive(Debug)]
pub struct StoredGgufFailure<F: StorageFamily> {
    cause: StoredCause<F::Error>,
    owners: StoredOwners<F>,
}
impl<F: StorageFamily> StoredGgufFailure<F> {
    /// The exact input lease allocation, without cloning or reacquisition.
    pub fn lease(&self) -> &GgufLease {
        &self.owners.lease
    }
    /// Actual supplied storage refusal, when the storage provider failed.
    pub fn storage_error(&self) -> Option<&SuppliedStorageError<F::Error>> {
        match &self.cause {
            StoredCause::Storage(e) => Some(e),
            _ => None,
        }
    }
    /// Whether completed conversion output remains retained after a later failure.
    pub fn conversion_completed(&self) -> bool {
        self.owners.outputs.conversion_completed()
    }
    /// Initialized/partially read raw bytes; this is not successful-read credit.
    pub fn raw_bytes(&self) -> Option<&[u8]> {
        self.owners.raw.as_ref().map(AsRef::as_ref)
    }
}
impl<F: StorageFamily> std::fmt::Display for StoredGgufFailure<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.cause.fmt(f)
    }
}
impl<F: StorageFamily> std::error::Error for StoredGgufFailure<F> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause.source()
    }
}
fn selection(lease: &GgufLease) -> MetadataSelection<'_> {
    match lease.identity.selection.as_ref() {
        None => MetadataSelection::Full,
        Some(GgufPhysicalSelection::Axis(s)) => MetadataSelection::Axis(s),
        Some(GgufPhysicalSelection::DenseSpan(s)) => MetadataSelection::Span(s),
    }
}
impl<F: StorageFamily> StoredOwners<F> {
    fn run<P: GgufStorageProvider<Family = F>>(
        &mut self,
        provider: &mut P,
    ) -> Result<StoredCheckpointTensor<F>, StoredCause<F::Error>> {
        let length = usize::try_from(self.lease.proof.length_bytes)
            .ok()
            .and_then(|n| Layout::array::<u8>(n).ok())
            .ok_or(Cause::Layout)?;
        match provider.prepare_raw(GgufRawStorageRequest {
            lease: &self.lease,
            layout: length,
        }) {
            Ok(raw) => self.raw = Some(raw),
            Err((cause, raw)) => {
                self.raw = raw;
                return Err(SuppliedStorageError::Provider(cause).into());
            }
        }
        let raw = self.raw.as_ref().expect("successful raw preparation");
        if raw.as_ref().len() != length.size() || raw.capacity() < length.size() {
            return Err(SuppliedStorageError::Extent {
                expected: length.size(),
                actual: raw.as_ref().len(),
                capacity: raw.capacity(),
            }
            .into());
        }
        let physical = match StoredPhysicalDescriptor::prepare(
            &self.lease.entry.physical_descriptor,
            selection(&self.lease),
            provider,
        ) {
            Ok(physical) => physical,
            Err(failure) => {
                let (cause, physical) = failure.into_parts();
                self.physical = Some(physical);
                // The original early selection worker wraps its GGUF error in
                // the source's logical key context, before G2 reserve begins.
                return Err(match cause {
                    SuppliedStorageError::Metadata(MetadataDestinationError::Gguf(error)) => {
                        StoredCause::Original(Cause::Store(gguf_error(
                            &self.lease.entry.metadata.name,
                            error,
                        )))
                    }
                    error => StoredCause::Storage(error),
                });
            }
        };
        let crate::SourceTensorEncoding::Gguf { endian, .. } = self.lease.entry.source_encoding
        else {
            unreachable!("actual GGUF catalog encoding")
        };
        let StoredOutputs::Preparing {
            metadata,
            conversion,
        } = &mut self.outputs
        else {
            unreachable!("one preparation attempt")
        };
        match StoredConversion::prepare(physical.into_descriptor(), endian, provider) {
            Ok(value) => *conversion = Some(value),
            Err(failure) => {
                let (cause, value) = failure.into_parts();
                *conversion = Some(value);
                return Err(cause.into());
            }
        }
        // Resolve and retain the actual immutable row under the existing lock,
        // then release that loan before any public provider callback. No reader
        // state, catalog contents, or names are cloned/reconstructed.
        let metadata_source = {
            let lease = &self.lease;
            let cache = lease
                .store
                .readers
                .lock()
                .map_err(|_| StoreError::Internal("GGUF reader cache is poisoned".into()))?;
            let materializer =
                cache
                    .materializers
                    .get(lease.entry.checkpoint)
                    .ok_or_else(|| {
                        gguf_error(
                            &lease.entry.metadata.name,
                            "catalog references an unknown checkpoint",
                        )
                    })?;
            materializer
                .shared_metadata_source(&lease.entry.physical_name)
                .map_err(|error| gguf_error(&lease.entry.metadata.name, error))?
                .ok_or_else(|| {
                    gguf_error(
                        &lease.entry.metadata.name,
                        "GGUF store requires shared catalog metadata",
                    )
                })?
        };
        match StoredTensorMetadata::prepare(
            metadata_source.source(),
            selection(&self.lease),
            provider,
        ) {
            Ok(value) => *metadata = Some(value),
            Err(failure) => {
                let (cause, value) = failure.into_parts();
                *metadata = Some(value);
                return Err(cause.into());
            }
        }
        drop(metadata_source);
        let prepared_metadata = metadata.take().expect("G3 owner");
        let prepared_conversion = conversion.take().expect("G2 owner");
        match StoredTensorPair::try_new(prepared_metadata, prepared_conversion) {
            Ok(pair) => self.outputs = StoredOutputs::Paired(pair),
            Err((returned_metadata, returned_conversion)) => {
                *metadata = Some(returned_metadata);
                *conversion = Some(returned_conversion);
                unreachable!("newly prepared destinations have not been executed")
            }
        }
        let lease = &self.lease;
        let raw = self.raw.as_mut().expect("G1 owner");
        let StoredOutputs::Paired(pair) = &mut self.outputs else {
            unreachable!("paired before materialization")
        };
        materialize_worker(lease, |cache| {
            cache.materialize_worker(
                lease.entry.checkpoint,
                &lease.entry.physical_name,
                lease.store.max_cached_readers,
                &lease.entry.metadata.name,
                |materializer| {
                    materializer.converted_tensor_with_stored_pair(
                        &lease.entry.physical_name,
                        selection(lease),
                        raw.as_mut(),
                        pair,
                    )
                },
            )
        })?;
        let StoredOutputs::Paired(pair) = std::mem::take(&mut self.outputs) else {
            unreachable!("same pair after materialization")
        };
        match pair.into_result() {
            Ok(output) => Ok(output),
            Err(pair) => {
                self.outputs = StoredOutputs::Paired(pair);
                unreachable!("successful shared driver completes the same pair")
            }
        }
    }
}
impl GgufLease {
    /// Materialize with intact supplied G1/G2/G3 owners and return this same Box.
    /// Reuses the ordinary physical planner, conversion, metadata, reader-cache
    /// and statistics workers. No source format or admission policy is delegated.
    pub fn materialize_prepared_boxed_with_destinations<P: GgufStorageProvider>(
        lease: Box<Self>,
        mut provider: P,
    ) -> Result<(StoredCheckpointTensor<P::Family>, Box<Self>), StoredGgufFailure<P::Family>> {
        let mut owners = StoredOwners {
            outputs: StoredOutputs::default(),
            physical: None,
            raw: None,
            lease,
        };
        match owners.run(&mut provider) {
            Ok(output) => {
                drop(owners.raw.take());
                Ok((output, owners.lease))
            }
            Err(cause) => Err(StoredGgufFailure { cause, owners }),
        }
    }
}
