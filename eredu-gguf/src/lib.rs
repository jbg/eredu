//! Pure-Rust GGUF container I/O and GGML tensor conversion.
//!
//! The crate has no tensor-framework or native-code dependency. [`Reader`]
//! parses descriptors with configurable resource limits and reads one tensor at
//! a time. [`Checkpoint`] validates complete single-file or sharded checkpoints
//! without reading tensor payloads and then streams their conversion. [`Writer`]
//! emits deterministic GGUF v3 files to seekable outputs. Encodings with native
//! execution support remain in their checkpoint block representation through
//! conversion so runtimes can execute them without affine expansion.

mod catalog;
mod codebook;
mod convert;
mod error;
mod format;
mod iquant;
mod iquant_tables;
mod reader;
mod writer;

pub use catalog::{
    CatalogShard, CatalogTensor, Checkpoint, ConvertedCheckpointTensor, ConvertedTensorIter,
    LogicalDtype, LogicalTensorLayout, RawCheckpointTensor, TensorMaterializer,
    TranslatedTensorLayout,
};
pub use codebook::IQuantCodebook;
pub use convert::{
    convert_affine, AffineTensor, ConvertedTensor, DenseDtype, DenseTensor, IQuantTensor,
    MxFp4Tensor,
};
pub use error::{Error, Result};
pub use format::{
    Endian, GgmlType, MetadataArray, MetadataValue, TensorDescriptor, DEFAULT_ALIGNMENT,
};
pub use reader::{
    DenseTensorSpan, DenseTensorSpanPlan, EncodedSpan, Limits, Reader, SelectionAlignment,
    TensorSelection, TensorSelectionPlan,
};
pub use writer::{TensorInput, Writer, WriterOptions};
