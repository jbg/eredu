//! Complete result metadata bound to the already-owned physical source.
use super::*;
use eredu_gguf::{
    MetadataDestinationError, MetadataLayouts, MetadataSelection, PreparedTensorMetadata,
};

/// Owns final result metadata and plan scratch before G2 output/raw/source owners.
/// Reader/cache/parser and backend/native storage remain separate.
#[derive(Debug)]
pub struct PreparedGgufTensor<S: GgufRawStorage = Vec<u8>> {
    metadata: PreparedTensorMetadata,
    conversion: PreparedGgufConversion<S>,
}
/// Retains metadata, completed or partial outputs, raw bytes and the exact source.
#[derive(Debug)]
pub struct PreparedGgufTensorFailure<S: GgufRawStorage = Vec<u8>> {
    cause: Cause,
    metadata: Option<PreparedTensorMetadata>,
    conversion: PreparedGgufConversion<S>,
}
impl<S: GgufRawStorage> PreparedGgufTensorFailure<S> {
    /// Original checkpoint error, if source processing failed.
    pub fn store_error(&self) -> Option<&StoreError> {
        match &self.cause {
            Cause::Store(error) => Some(error),
            _ => None,
        }
    }
    /// Actual metadata binding, reserve, capacity or layout failure.
    pub fn metadata_error(&self) -> Option<&MetadataDestinationError> {
        match &self.cause {
            Cause::Metadata(error) => Some(error),
            _ => None,
        }
    }
    /// Retained complete or partially prepared metadata, including late completed output.
    pub fn metadata(&self) -> Option<&PreparedTensorMetadata> {
        self.metadata.as_ref()
    }
    /// Actual raw bytes, including any partial failed read; not successful-read credit.
    pub fn raw_bytes(&self) -> &[u8] {
        self.conversion.read.raw.as_ref()
    }
    /// Genuine source lease retained until every destination retires.
    pub fn lease(&self) -> &GgufLease {
        self.conversion.lease()
    }
}
impl<S: GgufRawStorage> std::fmt::Display for PreparedGgufTensorFailure<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.cause {
            Cause::Store(e) => e.fmt(f),
            Cause::Conversion(e) => e.fmt(f),
            Cause::Metadata(e) => e.fmt(f),
            Cause::Reserve(e) => e.fmt(f),
            Cause::Layout => f.write_str("GGUF raw destination layout is not representable"),
            Cause::Length { expected, actual } => write!(
                f,
                "GGUF raw destination has {actual} bytes; expected {expected}"
            ),
        }
    }
}
impl<S: GgufRawStorage> std::error::Error for PreparedGgufTensorFailure<S> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.cause {
            Cause::Store(e) => Some(e),
            Cause::Conversion(e) => Some(e),
            Cause::Metadata(e) => Some(e),
            Cause::Reserve(e) => Some(e),
            _ => None,
        }
    }
}
fn selection(lease: &GgufLease) -> MetadataSelection<'_> {
    match lease.identity.selection.as_ref() {
        None => MetadataSelection::Full,
        Some(GgufPhysicalSelection::Axis(s)) => MetadataSelection::Axis(s),
        Some(GgufPhysicalSelection::DenseSpan(s)) => MetadataSelection::Span(s),
    }
}
impl<S: GgufRawStorage> PreparedGgufConversion<S> {
    /// Adds final result metadata from this source's actual retained materializer catalog.
    /// No reader is opened, no cache is replaced and no allocating plan sizes storage.
    pub fn prepare_result_metadata(
        self,
    ) -> Result<PreparedGgufTensor<S>, PreparedGgufTensorFailure<S>> {
        let mut metadata = None;
        let prepared = (|| {
            let lease = self.lease();
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
            let source = materializer
                .metadata_source(&lease.entry.physical_name)
                .map_err(|error| gguf_error(&lease.entry.metadata.name, error))?;
            match PreparedTensorMetadata::prepare(source, selection(lease)) {
                Ok(destination) => {
                    metadata = Some(destination);
                    Ok(())
                }
                Err(failure) => {
                    let (cause, destination) = failure.into_parts();
                    metadata = Some(destination);
                    Err(Cause::Metadata(cause))
                }
            }
        })();
        match prepared {
            Ok(()) => Ok(PreparedGgufTensor {
                metadata: metadata.expect("prepared metadata"),
                conversion: self,
            }),
            Err(cause) => Err(PreparedGgufTensorFailure {
                cause,
                metadata,
                conversion: self,
            }),
        }
    }
}
impl<S: GgufRawStorage> PreparedGgufTensor<S> {
    /// Exact source/request/selection owner determining this only permitted materialization.
    pub fn lease(&self) -> &GgufLease {
        self.conversion.lease()
    }
    /// Source-derived requested metadata layouts, excluding allocator charge.
    pub fn metadata_layouts(&self) -> MetadataLayouts {
        self.metadata.layouts()
    }
    /// Actual retained metadata buffer capacities in bytes.
    pub fn metadata_capacity_bytes(&self) -> Option<usize> {
        self.metadata.capacity_bytes()
    }
    /// Inline combined owner, including the previously prepared G2 owners.
    pub fn owner_layout(&self) -> Layout {
        Layout::new::<Self>()
    }
    /// Executes the original cache, reopen, selection, read and conversion order.
    /// The intact owner preserves metadata/output-before-raw/source retirement.
    pub fn materialize(self) -> Result<ConvertedCheckpointTensor, PreparedGgufTensorFailure<S>> {
        self.materialize_retaining().map(|(output, _owner)| output)
    }
    pub(in crate::gguf_store::read_destination) fn materialize_retaining(
        mut self,
    ) -> Result<(ConvertedCheckpointTensor, Self), PreparedGgufTensorFailure<S>> {
        match super::super::materialize_with_metadata(
            &self.conversion.read.lease,
            Some(self.conversion.read.raw.as_mut()),
            Some(&mut self.conversion.conversion),
            Some(&mut self.metadata),
        ) {
            Ok(output) => Ok((output, self)),
            Err(cause) => Err(PreparedGgufTensorFailure {
                cause,
                metadata: Some(self.metadata),
                conversion: self.conversion,
            }),
        }
    }
}

impl<S: GgufRawStorage> PreparedGgufTensor<S> {
    pub(in crate::gguf_store::read_destination) fn into_boxed_lease(self) -> Box<GgufLease> {
        let super::super::LeaseOwner::Boxed(lease) = self.conversion.read.lease else {
            unreachable!("boxed preparation retains its actual lease representation")
        };
        lease
    }
}
