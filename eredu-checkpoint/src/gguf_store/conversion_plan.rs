//! Source-owned physical selection and requested conversion metadata, without I/O.
use super::*;
use eredu_gguf::{ConversionDestinationError, ConversionPlan};

/// A cold source/selection or conversion-layout failure, retaining its typed cause.
#[derive(Debug)]
pub enum GgufConversionPlanError {
    /// Existing source authorization or physical selection error.
    Store(StoreError),
    /// Existing conversion shape/dispatch or checked layout error.
    Conversion(ConversionDestinationError),
}
impl From<StoreError> for GgufConversionPlanError {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}
impl From<ConversionDestinationError> for GgufConversionPlanError {
    fn from(error: ConversionDestinationError) -> Self {
        Self::Conversion(error)
    }
}
impl std::fmt::Display for GgufConversionPlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Store(e) => e.fmt(f),
            Self::Conversion(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for GgufConversionPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Store(e) => e,
            Self::Conversion(e) => e,
        })
    }
}

/// Requested physical-group geometry from an actual retained GGUF source.
/// Owns only metadata and source aliases, never a payload or an acquired lease.
/// This is not permission to acquire, a capacity bound or a complete peak estimate.
#[derive(Debug)]
pub struct GgufConversionPlan {
    conversion: ConversionPlan,
    identity: GgufLeaseIdentity,
    output_name: String,
    output_shape: Vec<usize>,
    selection_is_materialized: bool,
    read: ReadExtent,
    source: crate::store::SourceHandle<StoreInner>,
}
#[derive(Debug)]
struct ReadExtent {
    physically_bounded: bool,
    offset: u64,
    bytes: u64,
}
impl GgufConversionPlan {
    /// Borrow only the actual source catalog's concrete incoming control owner.
    /// This retained source plan cannot substitute another catalog or invoke a
    /// custom provider. The caller validates its own closed accounting type.
    pub fn catalog_control_owner<C: std::any::Any>(&self) -> Option<&C> {
        self.source.catalog.origin()
    }

    /// The shared converter's requested layouts and ordered output geometry.
    pub fn conversion(&self) -> &ConversionPlan {
        &self.conversion
    }
    /// Physical group identity; companion outputs can share this identity.
    pub fn identity(&self) -> &GgufLeaseIdentity {
        &self.identity
    }
    /// Actual requested logical companion in that physical group.
    pub fn output_name(&self) -> &str {
        &self.output_name
    }
    /// Logical requested shape, distinct from a full-read fallback's physical shape.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }
    /// Whether the physical read already materializes the requested selection.
    pub fn selection_is_materialized(&self) -> bool {
        self.selection_is_materialized
    }
    /// Encoded bytes requested by the physical plan; no bytes have been read.
    pub fn requested_read_bytes(&self) -> u64 {
        self.read.bytes
    }
    /// Physical byte offset relative to the complete tensor's payload.
    pub fn relative_read_offset_bytes(&self) -> u64 {
        self.read.offset
    }
    /// Whether the planned physical request is bounded to the requested selection.
    /// This describes the request, not completed I/O or an accepted reservation.
    pub fn physically_bounded(&self) -> bool {
        self.read.physically_bounded
    }
    /// Cumulative G1/physical/G2/G3 supplied requests for this retained source.
    /// Does not allocate payload, open a reader, or certify allocator capacity.
    pub fn supplied_storage_bound(
        &self,
    ) -> Result<eredu_gguf::StorageRequestBound, GgufConversionPlanError> {
        let metadata = {
            let cache = self
                .source
                .readers
                .lock()
                .map_err(|_| StoreError::Internal("GGUF reader cache is poisoned".into()))?;
            let materializer = cache
                .materializers
                .get(self.identity.checkpoint)
                .ok_or_else(|| {
                    StoreError::Internal("catalog references an unknown checkpoint".into())
                })?;
            materializer
                .shared_metadata_source(&self.identity.physical_name)
                .map_err(|error| gguf_error(&self.output_name, error))?
                .ok_or_else(|| {
                    StoreError::Internal("GGUF store requires shared catalog metadata".into())
                })?
        };
        let selection = match self.identity.selection.as_ref() {
            None => eredu_gguf::MetadataSelection::Full,
            Some(GgufPhysicalSelection::Axis(value)) => eredu_gguf::MetadataSelection::Axis(value),
            Some(GgufPhysicalSelection::DenseSpan(value)) => {
                eredu_gguf::MetadataSelection::Span(value)
            }
        };
        let source = metadata.source();
        let bound = (|| {
            let mut out =
                eredu_gguf::StorageRequestBound::one::<u8>(usize::try_from(self.read.bytes).ok()?)?;
            out = out.checked_add(source.physical_storage_bound(selection)?)?;
            out = out.checked_add(self.conversion.supplied_storage_bound()?)?;
            out.checked_add(source.supplied_storage_bound(selection)?)
        })()
        .ok_or(ConversionDestinationError::Layout)?;
        Ok(bound)
    }
    /// Named temporary owners used by the cold supplied-request query, excluding
    /// already retained source/plan backing and independent cold error producers.
    pub fn supplied_storage_control_bytes() -> Option<usize> {
        use std::mem::size_of;
        size_of::<eredu_gguf::SharedTensorMetadataSource>()
            .checked_add(size_of::<eredu_gguf::TensorMetadataSource<'static>>())?
            .checked_add(size_of::<eredu_gguf::MetadataSelection<'static>>())?
            .checked_add(size_of::<std::sync::MutexGuard<'static, ReaderCache>>())?
            .checked_add(size_of::<eredu_gguf::StorageRequestBound>())?
            .checked_add(size_of::<
                Result<eredu_gguf::StorageRequestBound, GgufConversionPlanError>,
            >())
    }
    /// Compare actual source and physical selection; issues no acquisition authority.
    pub fn matches_lease(&self, lease: &GgufLease) -> bool {
        self.source.same(&lease.store) && self.identity == lease.identity
    }
    /// Actual plan metadata capacities, excluding retained source storage,
    /// allocator charge and every tensor/native payload.
    pub fn metadata_bytes(&self) -> Option<usize> {
        use std::alloc::Layout;
        let selection = match self.identity.selection.as_ref() {
            Some(GgufPhysicalSelection::Axis(GgufTensorSelection::Indices { indices, .. })) => {
                Layout::array::<usize>(indices.capacity()).ok()?.size()
            }
            Some(GgufPhysicalSelection::DenseSpan(span)) => span.shape_storage_layout()?.size(),
            _ => 0,
        };
        Layout::new::<Self>()
            .size()
            .checked_add(
                self.conversion
                    .metadata_bytes()?
                    .checked_sub(Layout::new::<ConversionPlan>().size())?,
            )?
            .checked_add(self.identity.physical_name.capacity())?
            .checked_add(self.output_name.capacity())?
            .checked_add(
                Layout::array::<usize>(self.output_shape.capacity())
                    .ok()?
                    .size(),
            )?
            .checked_add(selection)
    }
}

/// Shared with actual G2 preparation, preserving its selected descriptor checks.
pub(super) fn selected_descriptor(
    entry: &CatalogEntry,
    selection: Option<&GgufPhysicalSelection>,
) -> Result<(TensorDescriptor, eredu_gguf::Endian), StoreError> {
    let descriptor = &entry.physical_descriptor;
    let selected = match selection {
        None => descriptor.clone(),
        Some(GgufPhysicalSelection::Axis(selection)) => {
            TensorSelectionPlan::new(descriptor, selection.clone())
                .map_err(|e| gguf_error(&entry.metadata.name, e))?
                .selected_descriptor()
                .clone()
        }
        Some(GgufPhysicalSelection::DenseSpan(selection)) => {
            DenseTensorSpanPlan::new(descriptor, selection.clone())
                .map_err(|e| gguf_error(&entry.metadata.name, e))?
                .selected_descriptor()
                .clone()
        }
    };
    let crate::SourceTensorEncoding::Gguf { endian, .. } = entry.source_encoding else {
        unreachable!("GGUF entries retain the actual catalog encoding")
    };
    Ok((selected, endian))
}
impl GgufWeightStore {
    /// Inspect the actual bounded physical request, using existing selection and
    /// fallback rules. No reader is opened, lease acquired or payload allocated.
    pub fn conversion_plan(
        &self,
        request: &TensorReadRequest,
    ) -> Result<GgufConversionPlan, GgufConversionPlanError> {
        let entry =
            self.inner
                .catalog
                .get(&request.key)
                .ok_or_else(|| StoreError::UnknownTensor {
                    key: request.key.clone(),
                })?;
        let (output_shape, read) = lease_destination::plan_request(entry, request)?;
        let (descriptor, endian) = selected_descriptor(entry, read.physical_selection.as_ref())?;
        let conversion = ConversionPlan::new(&descriptor, endian)?;
        let physically_bounded =
            matches!(request.selection, TensorSelection::Full) || read.physical_selection.is_some();
        Ok(GgufConversionPlan {
            conversion,
            identity: GgufLeaseIdentity {
                source: self.inner.identity(),
                checkpoint: entry.checkpoint,
                physical_name: entry.physical_name.clone(),
                selection: read.physical_selection,
            },
            output_name: entry.original_name.clone(),
            output_shape,
            selection_is_materialized: read.selection_is_materialized,
            read: ReadExtent {
                physically_bounded,
                offset: read.physical_offset,
                bytes: read.physical_byte_len,
            },
            source: self.inner.clone(),
        })
    }
}
impl GgufLease {
    /// Inspect this same retained physical selection before allocating G1/G2
    /// payload destinations. The result grants no budget or successful conversion.
    pub fn conversion_plan(&self) -> Result<GgufConversionPlan, GgufConversionPlanError> {
        let (descriptor, endian) =
            selected_descriptor(&self.entry, self.identity.selection.as_ref())?;
        Ok(GgufConversionPlan {
            conversion: ConversionPlan::new(&descriptor, endian)?,
            identity: self.identity.clone(),
            output_name: self.entry.original_name.clone(),
            output_shape: self.output_shape.clone(),
            selection_is_materialized: self.selection_is_materialized,
            read: ReadExtent {
                physically_bounded: self.proof.physically_bounded,
                offset: self.proof.offset_bytes,
                bytes: self.proof.length_bytes,
            },
            source: self.store.clone(),
        })
    }
}

mod acquisition;
