//! Pure-Rust GGUF container I/O and GGML tensor conversion.
//!
//! The crate has no tensor-framework or native-code dependency. [`Reader`]
//! parses descriptors with configurable resource limits and reads one tensor at
//! a time. [`Checkpoint`] validates complete single-file or sharded checkpoints
//! without reading tensor payloads and then streams their conversion. [`Writer`]
//! emits deterministic GGUF v3 files to seekable outputs. Encodings with native
//! execution support remain in their checkpoint block representation through
//! conversion so runtimes can execute them without affine expansion.

#![forbid(unsafe_code)]

mod catalog;
mod codebook;
mod convert;
mod error;
mod format;
mod iquant;
mod iquant_tables;
mod reader;
mod supplied_storage;
mod writer;

pub use catalog::{
    CatalogShard, CatalogTensor, Checkpoint, ConvertedCheckpointTensor, ConvertedTensorIter,
    LogicalDtype, LogicalTensorLayout, MetadataLayouts, MetadataPreparationFailure,
    MetadataSelection, PreparedTensorMetadata, RawCheckpointTensor, SharedTensorMetadataSource,
    TensorMaterializer, TensorMetadataSource, TranslatedTensorLayout,
};
pub use codebook::IQuantCodebook;
pub use convert::{
    convert_affine, AffineTensor, ConversionDestinationError, ConversionLayouts,
    ConversionOutputPlan, ConversionPlan, ConvertedTensor, DenseDtype, DenseTensor, IQuantTensor,
    MxFp4Tensor, PreparedConversion, PreparedConversionFailure, StoredConversion,
    StoredConversionFailure, StoredConvertedTensor,
};
pub use error::{Error, Result, UnsupportedGgmlType};
pub use format::{
    Endian, GgmlType, MetadataArray, MetadataValue, TensorDescriptor, TensorDescriptorView,
    DEFAULT_ALIGNMENT,
};
pub use reader::{
    DenseTensorSpan, DenseTensorSpanPlan, EncodedSpan, HeaderStorageKind, HeaderStorageStep,
    Limits, MetadataDestinationError, PreparedHeader, PreparedHeaderChanged, ReadDestinationError,
    Reader, ReaderBuffer, ReaderBufferPreparationFailure, SelectionAlignment,
    StoredPhysicalDescriptor, StoredPhysicalFailure, TensorSelection, TensorSelectionPlan,
};
pub use writer::{TensorInput, Writer, WriterOptions};

pub use supplied_storage::{
    InitializedStorage, StorageFamily, StorageProvider, StorageRequestBound, StoredBuffer,
    StoredDescriptor, SuppliedStorageError,
};

pub use catalog::{
    StoredCheckpointTensor, StoredMetadataFailure, StoredOutputNames, StoredTensorMetadata,
    StoredTensorPair,
};

pub use convert::{packed_iquant_shape, ConvertedParts};

pub use catalog::{PreparedMaterializerFailure, PreparedMaterializerStorage};
