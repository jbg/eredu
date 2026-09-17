use crate::convert::{
    affine_shapes, conversion_kind, iquant_packed_shape, mxfp4_shapes, ConversionKind,
};
use crate::reader::{CatalogReader, FileReader};
use crate::reader::{RawStorage, ReadDestinationError};
use crate::PreparedHeader;
use crate::{
    ConvertedTensor, DenseDtype, DenseTensorSpan, DenseTensorSpanPlan, Endian, Error, Limits,
    MetadataValue, Reader, Result, TensorDescriptor, TensorSelection, TensorSelectionPlan,
};
use std::collections::{BTreeMap, HashMap};
#[cfg(test)]
use std::fs::File;

use std::path::{Path, PathBuf};
use std::sync::Arc;

mod materializer_index;
mod metadata_destination;
mod reader_storage;
use materializer_index::MaterializerIndex;
use metadata_destination::MetadataPolicy;
pub use metadata_destination::{
    MetadataLayouts, MetadataPreparationFailure, MetadataSelection, PreparedTensorMetadata,
    SharedTensorMetadataSource, StoredCheckpointTensor, StoredMetadataFailure, StoredOutputNames,
    StoredTensorMetadata, StoredTensorPair, TensorMetadataSource,
};
use reader_storage::{open_with_reader_storage, ReaderStorage};

const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS_COUNT: &str = "split.tensors.count";

/// Scalar encoding of one logical tensor produced by GGUF conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalDtype {
    F32,
    F16,
    Bf16,
    I8,
    I16,
    U8,
    U32,
    I32,
    I64,
    F64,
}

impl From<DenseDtype> for LogicalDtype {
    fn from(value: DenseDtype) -> Self {
        match value {
            DenseDtype::F32 => Self::F32,
            DenseDtype::F16 => Self::F16,
            DenseDtype::Bf16 => Self::Bf16,
            DenseDtype::I8 => Self::I8,
            DenseDtype::I16 => Self::I16,
            DenseDtype::I32 => Self::I32,
            DenseDtype::I64 => Self::I64,
            DenseDtype::F64 => Self::F64,
        }
    }
}

/// Name, shape, and dtype of one converted tensor without its payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalTensorLayout {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: LogicalDtype,
}

/// One physical GGUF tensor and the logical tensors its conversion produces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogTensor {
    descriptor: TensorDescriptor,
    outputs: Vec<LogicalTensorLayout>,
    affine: Option<(u8, u32)>,
}

impl CatalogTensor {
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    pub fn outputs(&self) -> &[LogicalTensorLayout] {
        &self.outputs
    }

    /// Packed affine `(bits, group_size)`, or `None` for dense and native-block outputs.
    pub fn affine(&self) -> Option<(u8, u32)> {
        self.affine
    }

    /// Whether this physical tensor expands into packed MXFP4 weights and scales.
    pub fn is_mxfp4(&self) -> bool {
        self.descriptor.ggml_type == crate::GgmlType::MxFp4
    }
}

/// Header-only description of one GGUF payload shard.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogShard {
    path: PathBuf,
    split_no: usize,
    version: u32,
    endian: Endian,
    alignment: u64,
    tensors: Vec<CatalogTensor>,
    prepared_header: Option<Arc<PreparedHeader>>,
}

impl CatalogShard {
    /// Actual cold header capability, if this source explicitly fixes reopened header bytes.
    pub fn prepared_header(&self) -> Option<&PreparedHeader> {
        self.prepared_header.as_deref()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn split_no(&self) -> usize {
        self.split_no
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn endian(&self) -> Endian {
        self.endian
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn tensors(&self) -> &[CatalogTensor] {
        &self.tensors
    }
}

/// A validated, streaming handle for a single-file or sharded GGUF checkpoint.
///
/// Opening a checkpoint parses every shard header and tensor descriptor but does
/// not read or convert tensor payloads. Payloads are materialized one physical
/// tensor at a time through [`Self::converted_tensors`] or
/// [`Self::for_each_converted_tensor`].
#[derive(Debug, Clone)]
pub struct Checkpoint {
    metadata: BTreeMap<String, MetadataValue>,
    shards: Vec<CatalogShard>,
    physical_tensor_count: usize,
    limits: Limits,
}

/// One materialized physical GGUF tensor and its converted logical output group.
#[derive(Debug, Clone, PartialEq)]
pub struct ConvertedCheckpointTensor {
    shard_index: usize,
    tensor_index: usize,
    descriptor: TensorDescriptor,
    output_names: Vec<String>,
    converted: ConvertedTensor,
}

/// One physical GGUF tensor retained in its checkpoint-native byte encoding.
///
/// This is the ownership seam used by device backends which can consume GGUF
/// blocks directly. The byte vector is intentionally not interpreted or
/// repacked by this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RawCheckpointTensor {
    shard_index: usize,
    tensor_index: usize,
    endian: Endian,
    descriptor: TensorDescriptor,
    data: Vec<u8>,
}

impl RawCheckpointTensor {
    /// Zero-based index of the shard containing this tensor.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Zero-based tensor index within the containing shard.
    pub fn tensor_index(&self) -> usize {
        self.tensor_index
    }

    /// Endianness declared by the containing GGUF shard.
    pub fn endian(&self) -> Endian {
        self.endian
    }

    /// Physical GGUF tensor descriptor.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Checkpoint-native tensor payload.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Consume this tensor and return its checkpoint-native payload.
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }
}

impl ConvertedCheckpointTensor {
    /// Zero-based index of the shard containing this tensor.
    pub fn shard_index(&self) -> usize {
        self.shard_index
    }

    /// Zero-based tensor index within the containing shard.
    pub fn tensor_index(&self) -> usize {
        self.tensor_index
    }

    /// Physical GGUF tensor descriptor.
    pub fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    /// Catalog-owned logical names in the same order as the converted outputs.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Converted dense tensor or atomic affine tensor group.
    pub fn converted(&self) -> &ConvertedTensor {
        &self.converted
    }

    /// Consume the item and return its converted tensor group.
    pub fn into_converted(self) -> ConvertedTensor {
        self.converted
    }

    /// Consume the item into its physical descriptor, logical output names,
    /// and converted tensor group.
    pub fn into_parts(self) -> (TensorDescriptor, Vec<String>, ConvertedTensor) {
        (self.descriptor, self.output_names, self.converted)
    }
}

/// Fallible iterator that materializes one physical tensor at a time.
pub struct ConvertedTensorIter<'a> {
    checkpoint: &'a Checkpoint,
    shard_index: usize,
    tensor_index: usize,
    reader: Option<CatalogReader<FileReader>>,
    finished: bool,
    header_scratch: Vec<u8>,
    reader_storage: ReaderStorage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TensorLocation {
    shard_index: usize,
    tensor_index: usize,
}

/// Indexed materializer that reuses the currently open GGUF shard reader.
///
/// Ordinary constructors use a hash index; explicit coordinate constructors
/// binary-search retained physical names. Consecutive requests from the same
/// shard reuse one parsed reader. Switching shards opens and validates
/// the replacement before retiring the previous reader; an outer cache can call
/// the explicit close operation first when enforcing its reader ceiling.
pub struct TensorMaterializer {
    locations: MaterializerIndex,
    reader: Option<(usize, CatalogReader<FileReader>)>,
    header_scratch: Vec<u8>,
    reader_storage: ReaderStorage,
    // Shared custody retires after index, reader and scratch storage.
    checkpoint: MaterializerCheckpoint,
}

// Ordinary constructors retain their original inline owner. Only an explicit
// cold sharing request introduces the one Arc allocation; the actual checkpoint
// and all of its name/descriptor allocations move without cloning.
enum MaterializerCheckpoint {
    Owned(Checkpoint),
    Shared(prepared_materializer::SharedCheckpoint),
}
impl std::ops::Deref for MaterializerCheckpoint {
    type Target = Checkpoint;
    fn deref(&self) -> &Checkpoint {
        match self {
            Self::Owned(checkpoint) => checkpoint,
            Self::Shared(checkpoint) => checkpoint,
        }
    }
}
impl From<Checkpoint> for MaterializerCheckpoint {
    fn from(checkpoint: Checkpoint) -> Self {
        Self::Owned(checkpoint)
    }
}

impl std::fmt::Debug for TensorMaterializer {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("TensorMaterializer")
            .field("tensor_count", &self.locations.len())
            .field(
                "open_shard_index",
                &self.reader.as_ref().map(|(index, _)| index),
            )
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ConvertedTensorIter<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConvertedTensorIter")
            .field("shard_index", &self.shard_index)
            .field("tensor_index", &self.tensor_index)
            .field("finished", &self.finished)
            .finish_non_exhaustive()
    }
}

/// A logical layout paired with the pre-translation name that produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranslatedTensorLayout {
    pub physical_name: String,
    pub original_name: String,
    pub layout: LogicalTensorLayout,
}

impl Checkpoint {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_limits(path, Limits::default())
    }

    pub fn open_with_limits(path: impl AsRef<Path>, limits: Limits) -> Result<Self> {
        Self::open_policy(path.as_ref(), limits, false)
    }
    /// Opens a source whose later successful header reads must match these actual cold bytes.
    ///
    /// Unlike ordinary sources, even otherwise valid metadata changes are rejected.
    /// Payload bytes and filesystem identity are not made immutable. Capture storage
    /// is cold source ownership, not a complete parser or request bound.
    pub fn open_with_prepared_headers(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_prepared_headers_and_limits(path, Limits::default())
    }
    /// Uses explicit parser limits with the prepared-header source contract.
    pub fn open_with_prepared_headers_and_limits(
        path: impl AsRef<Path>,
        limits: Limits,
    ) -> Result<Self> {
        Self::open_policy(path.as_ref(), limits, true)
    }
    fn open_policy(path: &Path, limits: Limits, capture: bool) -> Result<Self> {
        let path = path.as_ref();
        validate_extension(path)?;
        let first = open_catalog_reader(path, limits.clone(), capture)?;
        let metadata = first.metadata().clone();
        let split_count = split_value(&metadata, SPLIT_COUNT)?.unwrap_or(0);
        if split_count <= 1 {
            let tensors = catalog_tensors(first.tensors(), first.endian())?;
            if let Some(split_no) = split_value(&metadata, SPLIT_NO)? {
                if split_no != 0 {
                    return Err(shard_error(format!(
                        "single-shard GGUF {:?} has {SPLIT_NO}={split_no}, expected 0",
                        path.display()
                    )));
                }
            }
            if let Some(expected_tensors) = split_value(&metadata, SPLIT_TENSORS_COUNT)? {
                if expected_tensors != tensors.len() {
                    return Err(shard_error(format!(
                        "single-shard GGUF declares {expected_tensors} tensors in {SPLIT_TENSORS_COUNT}, but {} were cataloged",
                        tensors.len()
                    )));
                }
            }
            validate_logical_names(tensors.iter())?;
            return Ok(Self {
                physical_tensor_count: tensors.len(),
                metadata,
                shards: vec![CatalogShard {
                    path: path.to_path_buf(),
                    split_no: 0,
                    version: first.version(),
                    endian: first.endian(),
                    alignment: first.alignment(),
                    prepared_header: first.captured_header().cloned(),
                    tensors,
                }],
                limits,
            });
        }

        let first_split_no = required_split_value(&metadata, SPLIT_NO, path)?;
        if first_split_no != 0 {
            return Err(shard_error(format!(
                "sharded GGUF must be loaded from its first shard, but {:?} has {SPLIT_NO}={first_split_no}",
                path.display()
            )));
        }
        let expected_tensors = required_split_value(&metadata, SPLIT_TENSORS_COUNT, path)?;
        let paths = shard_paths(path, split_count)?;
        let mut shards = Vec::with_capacity(split_count);
        let mut names = HashMap::<String, PathBuf>::new();
        let first_tensors = catalog_tensors(first.tensors(), first.endian())?;
        for tensor in &first_tensors {
            names.insert(tensor.descriptor.name.clone(), path.to_path_buf());
        }
        let mut physical_tensor_count = first_tensors.len();
        shards.push(CatalogShard {
            path: path.to_path_buf(),
            split_no: 0,
            version: first.version(),
            endian: first.endian(),
            alignment: first.alignment(),
            prepared_header: first.captured_header().cloned(),
            tensors: first_tensors,
        });

        for (split_no, shard_path) in paths.into_iter().enumerate().skip(1) {
            if !shard_path.is_file() {
                return Err(shard_error(format!(
                    "missing GGUF shard {:?}",
                    shard_path.display()
                )));
            }
            let reader = open_catalog_reader(&shard_path, limits.clone(), capture)?;
            let shard_metadata = reader.metadata();
            let actual_split_no = required_split_value(shard_metadata, SPLIT_NO, &shard_path)?;
            if actual_split_no != split_no {
                return Err(shard_error(format!(
                    "GGUF shard {:?} has {SPLIT_NO}={actual_split_no}, expected {split_no}",
                    shard_path.display()
                )));
            }
            let actual_count = required_split_value(shard_metadata, SPLIT_COUNT, &shard_path)?;
            if actual_count != split_count {
                return Err(shard_error(format!(
                    "GGUF shard {:?} has {SPLIT_COUNT}={actual_count}, expected {split_count}",
                    shard_path.display()
                )));
            }
            let actual_tensors =
                required_split_value(shard_metadata, SPLIT_TENSORS_COUNT, &shard_path)?;
            if actual_tensors != expected_tensors {
                return Err(shard_error(format!(
                    "GGUF shard {:?} has {SPLIT_TENSORS_COUNT}={actual_tensors}, expected {expected_tensors}",
                    shard_path.display()
                )));
            }

            let tensors = catalog_tensors(reader.tensors(), reader.endian())?;
            for tensor in &tensors {
                let source = tensor.descriptor.name.clone();
                if let Some(previous) = names.insert(source.clone(), shard_path.clone()) {
                    return Err(shard_error(format!(
                        "tensor {source:?} is duplicated across GGUF shards {:?} and {:?}",
                        previous.display(),
                        shard_path.display()
                    )));
                }
            }
            physical_tensor_count = physical_tensor_count
                .checked_add(tensors.len())
                .ok_or(Error::Overflow("sharded tensor count"))?;
            shards.push(CatalogShard {
                path: shard_path,
                split_no,
                version: reader.version(),
                endian: reader.endian(),
                alignment: reader.alignment(),
                prepared_header: reader.captured_header().cloned(),
                tensors,
            });
        }

        if physical_tensor_count != expected_tensors {
            return Err(shard_error(format!(
                "sharded GGUF declares {expected_tensors} tensors in {SPLIT_TENSORS_COUNT}, but {physical_tensor_count} were cataloged"
            )));
        }
        validate_logical_names(shards.iter().flat_map(|shard| shard.tensors.iter()))?;
        Ok(Self {
            metadata,
            shards,
            physical_tensor_count,
            limits,
        })
    }

    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.metadata
    }

    pub fn shards(&self) -> &[CatalogShard] {
        &self.shards
    }

    pub fn physical_tensor_count(&self) -> usize {
        self.physical_tensor_count
    }

    pub fn tensors(&self) -> impl Iterator<Item = &CatalogTensor> {
        self.shards.iter().flat_map(|shard| shard.tensors.iter())
    }

    pub fn logical_outputs(&self) -> impl Iterator<Item = &LogicalTensorLayout> {
        self.tensors().flat_map(|tensor| tensor.outputs.iter())
    }

    /// Iterate over converted tensor groups without retaining earlier payloads.
    pub fn converted_tensors(&self) -> ConvertedTensorIter<'_> {
        ConvertedTensorIter {
            checkpoint: self,
            shard_index: 0,
            tensor_index: 0,
            reader: None,
            finished: false,
            header_scratch: self.header_scratch(),
            reader_storage: ReaderStorage::Ordinary,
        }
    }

    /// Create an indexed named-tensor materializer with bounded reader reuse.
    pub fn materializer(&self) -> TensorMaterializer {
        let locations = self.materializer_locations();
        TensorMaterializer {
            header_scratch: self.header_scratch(),
            reader_storage: ReaderStorage::Ordinary,
            checkpoint: self.clone().into(),
            locations: locations.into(),
            reader: None,
        }
    }

    /// Create a lazy named-tensor materializer by moving this owned catalog.
    ///
    /// The physical-name index is constructed in the same order as
    /// [`Self::materializer`], without cloning the retained checkpoint metadata.
    pub fn into_materializer(self) -> TensorMaterializer {
        let locations = self.materializer_locations();
        TensorMaterializer {
            header_scratch: self.header_scratch(),
            reader_storage: ReaderStorage::Ordinary,
            checkpoint: self.into(),
            locations: locations.into(),
            reader: None,
        }
    }

    /// Move this catalog into one shared immutable owner for metadata loans
    /// that must outlive a mutable reader-cache borrow. The existing index and
    /// scratch construction are unchanged. This adds one cold Arc allocation;
    /// it does not qualify that allocation or grant source/admission authority.
    pub fn into_shared_materializer(self) -> TensorMaterializer {
        self.into_materializer().share_catalog()
    }

    fn header_scratch(&self) -> Vec<u8> {
        let bytes = self
            .shards
            .iter()
            .filter_map(|s| s.prepared_header.as_ref())
            .map(|h| h.scratch_len())
            .max()
            .unwrap_or(0);
        vec![0; bytes]
    }

    fn materializer_locations(&self) -> HashMap<String, TensorLocation> {
        self.shards
            .iter()
            .enumerate()
            .flat_map(|(shard_index, shard)| {
                shard
                    .tensors
                    .iter()
                    .enumerate()
                    .map(move |(tensor_index, tensor)| {
                        (
                            tensor.descriptor.name.clone(),
                            TensorLocation {
                                shard_index,
                                tensor_index,
                            },
                        )
                    })
            })
            .collect()
    }

    /// Materialize and visit one physical GGUF tensor at a time.
    ///
    /// Dense tensors are delivered as one dense output. Packed affine tensors
    /// are delivered as one atomic weight/scales/biases group.
    pub fn for_each_converted_tensor<F>(&self, mut visitor: F) -> Result<()>
    where
        F: FnMut(ConvertedCheckpointTensor) -> Result<()>,
    {
        for tensor in self.converted_tensors() {
            visitor(tensor?)?;
        }
        Ok(())
    }

    /// Translate every logical tensor name and reject collisions before payload reads.
    pub fn translated_outputs<F>(&self, mut translate: F) -> Result<Vec<TranslatedTensorLayout>>
    where
        F: FnMut(&str) -> String,
    {
        let mut owners = HashMap::<String, String>::new();
        let mut translated = Vec::new();
        for tensor in self.tensors() {
            for output in tensor.outputs() {
                let name = translate(&output.name);
                if let Some(first_source) = owners.insert(name.clone(), output.name.clone()) {
                    return Err(Error::TranslatedTensorCollision {
                        name,
                        first_source,
                        second_source: output.name.clone(),
                    });
                }
                translated.push(TranslatedTensorLayout {
                    physical_name: tensor.descriptor.name.clone(),
                    original_name: output.name.clone(),
                    layout: LogicalTensorLayout {
                        name,
                        shape: output.shape.clone(),
                        dtype: output.dtype,
                    },
                });
            }
        }
        Ok(translated)
    }
}

impl TensorMaterializer {
    /// Actual independent initialized parser-scratch backing layout. This excludes
    /// immutable shared headers and all allocator, map, reader and native controls.
    pub fn prepared_header_scratch_layout(&self) -> Option<std::alloc::Layout> {
        std::alloc::Layout::array::<u8>(self.header_scratch.capacity()).ok()
    }

    /// Path of the shard containing `name`, without opening its payload reader.
    pub fn shard_path_for_tensor(&self, name: &str) -> Result<&Path> {
        self.shard_source_for_tensor(name).map(|(_, path)| path)
    }

    /// Borrow the immutable catalog shards retained by this materializer.
    /// Indices refer only to this catalog; they grant no source admission authority.
    pub fn shards(&self) -> &[CatalogShard] {
        &self.checkpoint.shards
    }

    /// Borrow the actual shard index and path for a physical tensor, without I/O.
    pub fn shard_source_for_tensor(&self, name: &str) -> Result<(usize, &Path)> {
        let location =
            self.locations
                .get(name, &self.checkpoint)
                .ok_or_else(|| Error::InvalidTensor {
                    tensor: name.to_string(),
                    reason: "tensor is not present in the checkpoint".into(),
                })?;
        Ok((
            location.shard_index,
            &self.checkpoint.shards[location.shard_index].path,
        ))
    }

    /// Path of the currently cached shard reader, if any.
    pub fn open_shard_path(&self) -> Option<&Path> {
        self.reader
            .as_ref()
            .map(|(index, _)| self.checkpoint.shards[*index].path.as_path())
    }

    /// Close the currently cached shard reader.
    pub fn close_reader(&mut self) -> Option<PathBuf> {
        let index = self.take_reader_index()?;
        Some(self.checkpoint.shards[index].path.clone())
    }

    /// Close the cached reader without allocating an owned copy of its path.
    ///
    /// Returns whether a reader was present. Its file, input buffer and parsed
    /// header are retired before this method returns.
    pub fn close_reader_without_path(&mut self) -> bool {
        self.take_reader_index().is_some()
    }

    fn take_reader_index(&mut self) -> Option<usize> {
        let (index, reader) = self.reader.take()?;
        self.reader_storage.retire(reader);
        Some(index)
    }

    fn location_and_open(&mut self, name: &str) -> Result<(TensorLocation, Endian)> {
        let location =
            self.locations
                .get(name, &self.checkpoint)
                .ok_or_else(|| Error::InvalidTensor {
                    tensor: name.to_string(),
                    reason: "tensor is not present in the checkpoint".into(),
                })?;
        let shard = &self.checkpoint.shards[location.shard_index];
        if self
            .reader
            .as_ref()
            .is_none_or(|(shard_index, _)| *shard_index != location.shard_index)
        {
            let reader = open_with_reader_storage(
                shard,
                self.checkpoint.limits.clone(),
                &mut self.header_scratch,
                &mut self.reader_storage,
            )?;
            if let Some((_, previous)) = self.reader.replace((location.shard_index, reader)) {
                self.reader_storage.retire(previous);
            }
        }
        Ok((location, shard.endian))
    }
    fn location_and_reader(
        &mut self,
        name: &str,
    ) -> Result<(TensorLocation, TensorDescriptor, Endian)> {
        let (location, endian) = self.location_and_open(name)?;
        Ok((
            location,
            self.checkpoint.shards[location.shard_index].tensors[location.tensor_index]
                .descriptor
                .clone(),
            endian,
        ))
    }

    /// Materialize one physical tensor by its GGUF name.
    pub fn converted_tensor(&mut self, name: &str) -> Result<ConvertedCheckpointTensor> {
        self.converted_tensor_with_storage(name, RawStorage::Ordinary, None, None)
            .map_err(ReadDestinationError::ordinary)
    }
    /// Materialize a bounded physical axis selection.
    pub fn converted_tensor_selected(
        &mut self,
        name: &str,
        selection: &TensorSelection,
    ) -> Result<ConvertedCheckpointTensor> {
        self.converted_tensor_selected_with_storage(
            name,
            selection,
            RawStorage::Ordinary,
            None,
            None,
        )
        .map_err(ReadDestinationError::ordinary)
    }
    /// Materialize a block-aligned contiguous physical span.
    pub fn converted_dense_tensor_span(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
    ) -> Result<ConvertedCheckpointTensor> {
        self.converted_dense_tensor_span_with_storage(
            name,
            selection,
            RawStorage::Ordinary,
            None,
            None,
        )
        .map_err(ReadDestinationError::ordinary)
    }
    /// Reuses the actual cached reader and ordinary conversion with exact raw storage.
    /// Header reuse follows the source contract. Plans, names and converted
    /// outputs in this raw-only call retain their ordinary storage.
    pub fn converted_tensor_with_raw_destination(
        &mut self,
        name: &str,
        raw: &mut [u8],
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_tensor_with_storage(name, RawStorage::Borrowed(raw), None, None)
    }
    /// Materializes the actual axis plan with an exact borrowed encoded buffer.
    pub fn converted_tensor_selected_with_raw_destination(
        &mut self,
        name: &str,
        selection: &TensorSelection,
        raw: &mut [u8],
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_tensor_selected_with_storage(
            name,
            selection,
            RawStorage::Borrowed(raw),
            None,
            None,
        )
    }
    /// Materializes the actual contiguous plan with exact borrowed encoded storage.
    pub fn converted_dense_tensor_span_with_raw_destination(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
        raw: &mut [u8],
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_dense_tensor_span_with_storage(
            name,
            selection,
            RawStorage::Borrowed(raw),
            None,
            None,
        )
    }
    /// Materialize one physical tensor by its GGUF name.
    fn converted_tensor_with_storage(
        &mut self,
        name: &str,
        storage: RawStorage<'_>,
        conversion: Option<&mut crate::PreparedConversion>,
        destination: Option<&mut PreparedTensorMetadata>,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_tensor_worker(name, storage, conversion, MetadataPolicy::new(destination))
    }
    fn converted_tensor_worker<'a, C: crate::reader::ConversionDestination>(
        &mut self,
        name: &str,
        storage: RawStorage<'_>,
        conversion: C,
        mut metadata: MetadataPolicy<'a>,
    ) -> std::result::Result<
        <C::Output as metadata_destination::CatalogOutput>::Result,
        ReadDestinationError,
    >
    where
        C::Output: metadata_destination::CatalogOutput,
    {
        use metadata_destination::CatalogOutput;
        let (location, endian) = self.location_and_open(name)?;
        let descriptor = metadata.base(
            &self.checkpoint.shards[location.shard_index].tensors[location.tensor_index].descriptor,
            location,
            endian,
        )?;
        let output_names = metadata.names(
            &self.checkpoint.shards[location.shard_index].tensors[location.tensor_index].outputs,
        )?;
        metadata.full()?;
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_tensor_view_with_storage(descriptor.view(), storage, conversion)
            .map_err(|source| {
                source.with_shard(&self.checkpoint.shards[location.shard_index].path)
            })?;
        let converted = converted.park(&mut metadata);
        Ok(C::Output::finish(
            location,
            descriptor,
            output_names,
            converted,
        ))
    }

    /// Materialize a bounded selection along one logical row-major tensor axis.
    fn converted_tensor_selected_with_storage(
        &mut self,
        name: &str,
        selection: &TensorSelection,
        storage: RawStorage<'_>,
        conversion: Option<&mut crate::PreparedConversion>,
        destination: Option<&mut PreparedTensorMetadata>,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_tensor_selected_worker(
            name,
            selection,
            storage,
            conversion,
            MetadataPolicy::new(destination),
        )
    }
    fn converted_tensor_selected_worker<'a, C: crate::reader::ConversionDestination>(
        &mut self,
        name: &str,
        selection: &TensorSelection,
        storage: RawStorage<'_>,
        conversion: C,
        mut metadata: MetadataPolicy<'a>,
    ) -> std::result::Result<
        <C::Output as metadata_destination::CatalogOutput>::Result,
        ReadDestinationError,
    >
    where
        C::Output: metadata_destination::CatalogOutput,
    {
        use metadata_destination::CatalogOutput;
        let (location, endian) = self.location_and_open(name)?;
        let mut descriptor = metadata.base(
            &self.checkpoint.shards[location.shard_index].tensors[location.tensor_index].descriptor,
            location,
            endian,
        )?;
        let output_names = metadata.names(
            &self.checkpoint.shards[location.shard_index].tensors[location.tensor_index].outputs,
        )?;
        let plan = metadata.axis(&descriptor, selection)?;
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_tensor_plan_with_storage(plan.view(), storage, conversion)
            .map_err(|source| {
                source.with_shard(&self.checkpoint.shards[location.shard_index].path)
            })?;
        let converted = converted.park(&mut metadata);
        descriptor = metadata.selected(plan.selected_descriptor())?;
        Ok(C::Output::finish(
            location,
            descriptor,
            output_names,
            converted,
        ))
    }

    /// Materialize one bounded contiguous span from an unquantized dense tensor.
    fn converted_dense_tensor_span_with_storage(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
        storage: RawStorage<'_>,
        conversion: Option<&mut crate::PreparedConversion>,
        destination: Option<&mut PreparedTensorMetadata>,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_dense_tensor_span_worker(
            name,
            selection,
            storage,
            conversion,
            MetadataPolicy::new(destination),
        )
    }
    fn converted_dense_tensor_span_worker<'a, C: crate::reader::ConversionDestination>(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
        storage: RawStorage<'_>,
        conversion: C,
        mut metadata: MetadataPolicy<'a>,
    ) -> std::result::Result<
        <C::Output as metadata_destination::CatalogOutput>::Result,
        ReadDestinationError,
    >
    where
        C::Output: metadata_destination::CatalogOutput,
    {
        use metadata_destination::CatalogOutput;
        let (location, endian) = self.location_and_open(name)?;
        let descriptor = metadata.base(
            &self.checkpoint.shards[location.shard_index].tensors[location.tensor_index].descriptor,
            location,
            endian,
        )?;
        let output_names = metadata.names(
            &self.checkpoint.shards[location.shard_index].tensors[location.tensor_index].outputs,
        )?;
        let plan = metadata.span(&descriptor, selection)?;
        let converted = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_dense_tensor_span_with_storage(plan.view(), storage, conversion)
            .map_err(|source| {
                source.with_shard(&self.checkpoint.shards[location.shard_index].path)
            })?;
        let converted = converted.park(&mut metadata);
        Ok(C::Output::finish(
            location,
            metadata.selected(plan.selected_descriptor())?,
            output_names,
            converted,
        ))
    }

    /// Materialize one physical tensor without converting its native encoding.
    pub fn raw_tensor(&mut self, name: &str) -> Result<RawCheckpointTensor> {
        let (location, descriptor, endian) = self.location_and_reader(name)?;
        let data = self
            .reader
            .as_mut()
            .expect("requested shard reader opened above")
            .1
            .read_raw(&descriptor)
            .map_err(|source| Error::Shard {
                path: self.checkpoint.shards[location.shard_index].path.clone(),
                source: Box::new(source),
            })?;
        Ok(RawCheckpointTensor {
            shard_index: location.shard_index,
            tensor_index: location.tensor_index,
            endian,
            descriptor,
            data,
        })
    }
}

impl Iterator for ConvertedTensorIter<'_> {
    type Item = Result<ConvertedCheckpointTensor>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        loop {
            let Some(shard) = self.checkpoint.shards.get(self.shard_index) else {
                self.finished = true;
                return None;
            };
            if self.tensor_index >= shard.tensors.len() {
                self.shard_index += 1;
                self.tensor_index = 0;
                if let Some(reader) = self.reader.take() {
                    self.reader_storage.retire(reader);
                }
                continue;
            }
            if self.reader.is_none() {
                match open_with_reader_storage(
                    shard,
                    self.checkpoint.limits.clone(),
                    &mut self.header_scratch,
                    &mut self.reader_storage,
                ) {
                    Ok(reader) => self.reader = Some(reader),
                    Err(error) => {
                        self.finished = true;
                        return Some(Err(error));
                    }
                }
            }

            let tensor_index = self.tensor_index;
            let catalog_tensor = &shard.tensors[tensor_index];
            let descriptor = catalog_tensor.descriptor.clone();
            let output_names = catalog_tensor
                .outputs
                .iter()
                .map(|output| output.name.clone())
                .collect();
            self.tensor_index += 1;
            let converted = self
                .reader
                .as_mut()
                .expect("reader opened above")
                .read_tensor(&descriptor)
                .map_err(|source| Error::Shard {
                    path: shard.path.clone(),
                    source: Box::new(source),
                });
            return Some(converted.map(|converted| ConvertedCheckpointTensor {
                shard_index: self.shard_index,
                tensor_index,
                descriptor,
                output_names,
                converted,
            }));
        }
    }
}

#[cfg(test)]
fn validate_reopened_shard(
    reader: CatalogReader<FileReader>,
    shard: &CatalogShard,
) -> Result<CatalogReader<FileReader>> {
    validate_reopened_shard_ref(&reader, shard)?;
    Ok(reader)
}

fn validate_reopened_shard_ref(
    reader: &CatalogReader<FileReader>,
    shard: &CatalogShard,
) -> Result<()> {
    let unchanged = reader.version() == shard.version
        && reader.endian() == shard.endian
        && reader.alignment() == shard.alignment
        && reader.tensors().len() == shard.tensors.len()
        && reader
            .tensors()
            .iter()
            .zip(&shard.tensors)
            .all(|(actual, cataloged)| actual == &cataloged.descriptor);
    if unchanged {
        Ok(())
    } else {
        Err(shard_error(format!(
            "GGUF shard {:?} changed after the checkpoint was opened",
            shard.path.display()
        )))
    }
}

fn catalog_tensors(descriptors: &[TensorDescriptor], endian: Endian) -> Result<Vec<CatalogTensor>> {
    descriptors
        .iter()
        .map(|descriptor| catalog_tensor(descriptor, endian))
        .collect()
}

fn catalog_tensor(descriptor: &TensorDescriptor, endian: Endian) -> Result<CatalogTensor> {
    let kind = conversion_kind(descriptor.ggml_type, endian)?;
    if let ConversionKind::IQuant = kind {
        return Ok(CatalogTensor {
            descriptor: descriptor.clone(),
            outputs: vec![LogicalTensorLayout {
                name: descriptor.name.clone(),
                shape: iquant_packed_shape(&descriptor.row_major_shape(), descriptor.ggml_type)?,
                dtype: LogicalDtype::U8,
            }],
            affine: None,
        });
    }
    if let ConversionKind::Dense(dtype) = kind {
        return Ok(CatalogTensor {
            descriptor: descriptor.clone(),
            outputs: vec![LogicalTensorLayout {
                name: descriptor.name.clone(),
                shape: descriptor.row_major_shape(),
                dtype: dtype.into(),
            }],
            affine: None,
        });
    }
    if let ConversionKind::MxFp4 = kind {
        let prefix = descriptor.name.strip_suffix(".weight").ok_or_else(|| {
            Error::tensor(&descriptor.name, "MXFP4 tensor name must end in .weight")
        })?;
        let (weight_shape, scale_shape) = mxfp4_shapes(descriptor)?;
        return Ok(CatalogTensor {
            descriptor: descriptor.clone(),
            outputs: vec![
                LogicalTensorLayout {
                    name: descriptor.name.clone(),
                    shape: weight_shape,
                    dtype: LogicalDtype::U32,
                },
                LogicalTensorLayout {
                    name: format!("{prefix}.scales"),
                    shape: scale_shape,
                    dtype: LogicalDtype::U8,
                },
            ],
            affine: None,
        });
    }

    let ConversionKind::Affine { bits, group_size } = kind else {
        unreachable!("dense, IQ, and MXFP4 conversions returned above");
    };
    let prefix = descriptor.name.strip_suffix(".weight").ok_or_else(|| {
        Error::tensor(
            &descriptor.name,
            "quantized tensor name must end in .weight",
        )
    })?;
    let (weight_shape, scale_shape) = affine_shapes(descriptor, bits, group_size)?;
    let outputs = vec![
        LogicalTensorLayout {
            name: descriptor.name.clone(),
            shape: weight_shape,
            dtype: LogicalDtype::U32,
        },
        LogicalTensorLayout {
            name: format!("{prefix}.scales"),
            shape: scale_shape.clone(),
            dtype: LogicalDtype::F16,
        },
        LogicalTensorLayout {
            name: format!("{prefix}.biases"),
            shape: scale_shape,
            dtype: LogicalDtype::F16,
        },
    ];
    Ok(CatalogTensor {
        descriptor: descriptor.clone(),
        outputs,
        affine: Some((bits, group_size)),
    })
}

fn validate_logical_names<'a>(tensors: impl Iterator<Item = &'a CatalogTensor>) -> Result<()> {
    let mut owners = HashMap::<String, String>::new();
    for tensor in tensors {
        for output in &tensor.outputs {
            if let Some(first_source) =
                owners.insert(output.name.clone(), tensor.descriptor.name.clone())
            {
                return Err(Error::DuplicateLogicalTensor {
                    name: output.name.clone(),
                    first_source,
                    second_source: tensor.descriptor.name.clone(),
                });
            }
        }
    }
    Ok(())
}

fn split_value(metadata: &BTreeMap<String, MetadataValue>, key: &str) -> Result<Option<usize>> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    shard_error(format!(
                        "GGUF metadata key {key:?} must be a non-negative integer scalar"
                    ))
                })
        })
        .transpose()
}

fn required_split_value(
    metadata: &BTreeMap<String, MetadataValue>,
    key: &str,
    path: &Path,
) -> Result<usize> {
    split_value(metadata, key)?.ok_or_else(|| {
        shard_error(format!(
            "GGUF shard {:?} is missing required metadata key {key:?}",
            path.display()
        ))
    })
}

fn shard_paths(path: &Path, split_count: usize) -> Result<Vec<PathBuf>> {
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            shard_error(format!(
                "GGUF path {:?} has a non-UTF-8 extension",
                path.display()
            ))
        })?;
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .ok_or_else(|| {
            shard_error(format!(
                "GGUF path {:?} has a non-UTF-8 filename",
                path.display()
            ))
        })?;
    let invalid_name = || {
        shard_error(format!(
            "sharded GGUF filename {:?} must end in -00001-of-NNNNN.gguf",
            path.display()
        ))
    };
    let (prefix_and_no, filename_count) = stem.rsplit_once("-of-").ok_or_else(&invalid_name)?;
    let (prefix, filename_no) = prefix_and_no.rsplit_once('-').ok_or_else(&invalid_name)?;
    let valid_digits =
        |value: &str| value.len() == 5 && value.bytes().all(|byte| byte.is_ascii_digit());
    if prefix.is_empty() || !valid_digits(filename_no) || !valid_digits(filename_count) {
        return Err(invalid_name());
    }
    let filename_no = filename_no
        .parse::<usize>()
        .map_err(|error| shard_error(format!("invalid GGUF shard number: {error}")))?;
    let filename_count = filename_count
        .parse::<usize>()
        .map_err(|error| shard_error(format!("invalid GGUF shard count: {error}")))?;
    if filename_no != 1 {
        return Err(shard_error(format!(
            "sharded GGUF must be loaded from shard 00001, got {filename_no:05}"
        )));
    }
    if filename_count != split_count {
        return Err(shard_error(format!(
            "GGUF filename declares {filename_count} shards, but {SPLIT_COUNT}={split_count}"
        )));
    }
    if split_count > 99_999 {
        return Err(shard_error(format!(
            "GGUF shard count {split_count} cannot be represented by the canonical five-digit filename"
        )));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    Ok((1..=split_count)
        .map(|index| {
            parent.join(format!(
                "{prefix}-{index:05}-of-{split_count:05}.{extension}"
            ))
        })
        .collect())
}

fn validate_extension(path: &Path) -> Result<()> {
    if path
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
    {
        Ok(())
    } else {
        Err(shard_error(format!(
            "checkpoint path {:?} must have a .gguf extension",
            path.display()
        )))
    }
}

fn open_catalog_reader(
    path: &Path,
    limits: Limits,
    capture: bool,
) -> Result<CatalogReader<FileReader>> {
    if capture {
        CatalogReader::open_captured(path, limits)
            .map(CatalogReader::into_file_reader)
            .map_err(|source| Error::Shard {
                path: path.to_path_buf(),
                source: Box::new(source),
            })
    } else {
        open_reader(path, limits)
    }
}
fn open_reopened_reader(
    shard: &CatalogShard,
    limits: Limits,
    scratch: &mut [u8],
) -> Result<CatalogReader<FileReader>> {
    if let Some(header) = &shard.prepared_header {
        CatalogReader::open_prepared(&shard.path, limits, header, scratch)
            .map(CatalogReader::into_file_reader)
            .map_err(|source| Error::Shard {
                path: shard.path.to_path_buf(),
                source: Box::new(source),
            })
    } else {
        open_reader(&shard.path, limits)
    }
}

fn open_reader(path: &Path, limits: Limits) -> Result<CatalogReader<FileReader>> {
    Reader::open_with_limits(path, limits)
        .map(CatalogReader::ordinary)
        .map(CatalogReader::into_file_reader)
        .map_err(|source| Error::Shard {
            path: path.to_path_buf(),
            source: Box::new(source),
        })
}

fn shard_error(message: impl Into<String>) -> Error {
    Error::InvalidShardSet(message.into())
}

impl TensorMaterializer {
    /// Executes the original read and conversion using both prepared destinations.
    /// Result metadata remains owning. Header reuse follows the source contract,
    /// and cache ownership and operation order are unchanged.
    pub fn converted_tensor_with_destinations(
        &mut self,
        name: &str,
        raw: &mut [u8],
        conversion: &mut crate::PreparedConversion,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_tensor_with_storage(name, RawStorage::Borrowed(raw), Some(conversion), None)
    }
    /// Executes the original read and conversion using both prepared destinations.
    /// Result metadata remains owning. Header reuse follows the source contract,
    /// and cache ownership and operation order are unchanged.
    pub fn converted_tensor_selected_with_destinations(
        &mut self,
        name: &str,
        selection: &TensorSelection,
        raw: &mut [u8],
        conversion: &mut crate::PreparedConversion,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_tensor_selected_with_storage(
            name,
            selection,
            RawStorage::Borrowed(raw),
            Some(conversion),
            None,
        )
    }
    /// Executes the original read and conversion using both prepared destinations.
    /// Result metadata remains owning. Header reuse follows the source contract,
    /// and cache ownership and operation order are unchanged.
    pub fn converted_dense_tensor_span_with_destinations(
        &mut self,
        name: &str,
        selection: &DenseTensorSpan,
        raw: &mut [u8],
        conversion: &mut crate::PreparedConversion,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        self.converted_dense_tensor_span_with_storage(
            name,
            selection,
            RawStorage::Borrowed(raw),
            Some(conversion),
            None,
        )
    }
}

impl TensorMaterializer {
    /// Executes the original full/selected driver using complete prepared result metadata.
    /// Reopen validation, physical planning, read and conversion ordering are unchanged.
    pub fn converted_tensor_with_metadata(
        &mut self,
        name: &str,
        selection: MetadataSelection<'_>,
        raw: &mut [u8],
        conversion: &mut crate::PreparedConversion,
        metadata: &mut PreparedTensorMetadata,
    ) -> std::result::Result<ConvertedCheckpointTensor, ReadDestinationError> {
        match selection {
            MetadataSelection::Full => self.converted_tensor_with_storage(
                name,
                RawStorage::Borrowed(raw),
                Some(conversion),
                Some(metadata),
            ),
            MetadataSelection::Axis(selection) => self.converted_tensor_selected_with_storage(
                name,
                selection,
                RawStorage::Borrowed(raw),
                Some(conversion),
                Some(metadata),
            ),
            MetadataSelection::Span(selection) => self.converted_dense_tensor_span_with_storage(
                name,
                selection,
                RawStorage::Borrowed(raw),
                Some(conversion),
                Some(metadata),
            ),
        }
    }
}

#[cfg(test)]
mod cache_controls_tests;

#[cfg(test)]
mod header_reuse_tests;

impl TensorMaterializer {
    /// Executes the unchanged full/axis/span driver into supplied output owners.
    /// The caller retains conversion buffers, completed output and all metadata
    /// through any read, conversion or late final-descriptor refusal.
    pub(crate) fn converted_tensor_with_supplied_metadata<F: crate::StorageFamily>(
        &mut self,
        name: &str,
        selection: MetadataSelection<'_>,
        raw: &mut [u8],
        conversion: &mut crate::StoredConversion<F>,
        metadata: &mut StoredTensorMetadata<F>,
    ) -> std::result::Result<(), ReadDestinationError> {
        let policy = metadata.policy();
        let result = match selection {
            MetadataSelection::Full => {
                self.converted_tensor_worker(name, RawStorage::Borrowed(raw), conversion, policy)
            }
            MetadataSelection::Axis(selection) => self.converted_tensor_selected_worker(
                name,
                selection,
                RawStorage::Borrowed(raw),
                conversion,
                policy,
            ),
            MetadataSelection::Span(selection) => self.converted_dense_tensor_span_worker(
                name,
                selection,
                RawStorage::Borrowed(raw),
                conversion,
                policy,
            ),
        };
        if result.is_ok() {
            metadata.mark_completed();
        }
        result
    }
}

impl TensorMaterializer {
    /// Execute the ordinary selected read/conversion/metadata driver into one
    /// private destination pair. Failure preserves that same pair, including
    /// completed conversion before any late metadata refusal.
    pub fn converted_tensor_with_stored_pair<F: crate::StorageFamily>(
        &mut self,
        name: &str,
        selection: MetadataSelection<'_>,
        raw: &mut [u8],
        pair: &mut StoredTensorPair<F>,
    ) -> std::result::Result<(), ReadDestinationError> {
        let (metadata, conversion) = pair.loans().map_err(ReadDestinationError::Metadata)?;
        self.converted_tensor_with_supplied_metadata(name, selection, raw, conversion, metadata)
    }
}

mod prepared_materializer;
pub use prepared_materializer::{PreparedMaterializerFailure, PreparedMaterializerStorage};
